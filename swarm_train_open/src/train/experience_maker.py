'''
This experience_maker logic has been modified significantly from the original openrlhf implementation
due to bugs, performance reasons and the requirement to customize sampling logic.
Everything worked out except for our attempt to implement sampling while param updating (seems to cause deadlock due to
some known cuda reason).
'''


import time

from openrlhf.trainer.ppo_utils.experience_maker import *
from openrlhf.trainer.ray.launcher import DistributedTorchRayActor
import ray.actor
from ray.util.collective import init_collective_group

from src.swarmenv.swarmbench.train_env import output_root,redirect_env
from src.swarmenv.swarmbench.framework import SwarmFramework

import os
import random
import gc
from typing import Any


# We apply a RawSamples class for re-batching samples.
# It is ok to simply use the original Samples class. However, it stores padded sequences and will cause performance
# issues since padding will be applied again to match the seq len during re-batching.
@dataclass
class RawSamples:
    sequences: Any
    max_input_len: Any
    eos_token_id: Any
    pad_token_id: Any
    prompts: Any
    labels: Any


def split_samples(samples_list: List[RawSamples], batch_size: int = 2, sample_fraction: float = 0.1) -> List[
    Samples]:

    all_samples = []
    all_rewards = []

    for sample in samples_list:
        # flatten
        for i in range(len(sample.sequences)):
            reward = float(sample.labels[i])
            all_rewards.append(reward)

            all_samples.append(
                RawSamples(
                    sequences=sample.sequences[i],
                    max_input_len=None,
                    eos_token_id=sample.eos_token_id,
                    pad_token_id=sample.pad_token_id,
                    prompts=sample.prompts[i],
                    labels=sample.labels[i]
                )
            )

    # reward + 0.1, avoid 0 weight
    weights = [reward + 0.1 for reward in all_rewards]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    num_samples_to_select = int(len(all_samples) * sample_fraction)
    selected_indices = random.choices(range(len(all_samples)), weights=normalized_weights, k=num_samples_to_select)
    selected_samples = [all_samples[i] for i in selected_indices]

    # pack samples into Samples object
    result_samples = []
    current_batch = []

    # A nasty openrlhf bug, has to apply a new batch_size instead of rollout batch_size to avoid OOM.
    for sample_data in selected_samples:
        current_batch.append(sample_data)
        if len(current_batch) >= batch_size:
            result_samples.append(create_samples_from_batch(current_batch))
            current_batch = []

    # last batch
    if current_batch:
        result_samples.append(create_samples_from_batch(current_batch))

    return result_samples


def create_samples_from_batch(batch: List[RawSamples]) -> RawSamples:
    sequences = [sample.sequences for sample in batch]
    max_input_len = max([len(sample.sequences[0]) for sample in batch])
    max_output_len = max([len(sample.sequences[1]) for sample in batch])
    eos_token_id = batch[0].eos_token_id
    pad_token_id = batch[0].pad_token_id
    prompts = [sample.prompts for sample in batch]
    labels = [sample.labels for sample in batch]

    seqs = []
    # Moved padding logic here.
    for input_ids, output_ids in sequences:
        # left padding input
        input_ids = [pad_token_id] * (max_input_len - len(input_ids)) + input_ids

        # right padding output
        output_ids = output_ids + [pad_token_id] * (max_output_len - len(output_ids))

        # concat input and output
        seqs.append(input_ids + output_ids)

    return RawSamples(
        sequences=torch.tensor(seqs),
        max_input_len=max_input_len,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        prompts=prompts,
        labels=labels
    )


class RemoteExperienceMakerMixin(BaseExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        # if self.custom_reward_func:
        #     self.custom_reward_func = ray.remote(self.custom_reward_func)

        redirect_env()
        self.swarms = [... for _ in range(5)]


    def get_world_size(self):
        return torch.distributed.get_world_size()

    def get_rank(self):
        return torch.distributed.get_rank()

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        samples_list = self.sample(all_prompts, all_labels, **generate_kwargs)
        experiences = self.samples_to_experience(samples_list, **generate_kwargs)
        return experiences

    def sample(self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs) -> List[Samples]:
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            try:
                torch.cuda.synchronize()
            except Exception as e:
                print(e)

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = self.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        try:
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            torch.cuda.synchronize()
        except Exception as e:
            print(e)
        return samples_list

    def samples_to_experience(self, samples_list, **generate_kwargs) -> List[Experience]:
        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        if not self.packing_samples:
            samples_list = split_samples(samples_list, 1)
        processed_samples_list = []
        for samples in samples_list:
            sequences, attention_mask, action_mask = self.process_sequences(
                samples.sequences, samples.max_input_len, samples.eos_token_id, samples.pad_token_id
            )
            sequences = sequences.to(torch.cuda.current_device())
            attention_mask = attention_mask.to(torch.cuda.current_device())
            action_mask = action_mask.to(torch.cuda.current_device())
            processed_samples_list.append(
                Samples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    num_actions=action_mask.size(1),
                    packed_seq_lens=None,
                    response_length=action_mask.float().sum(dim=-1),
                    total_length=attention_mask.float().sum(dim=-1),
                    prompts=samples.prompts,
                    labels=samples.labels,
                    pad_len=None,
                )
            )
        samples_list = processed_samples_list

        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences, **generate_kwargs)

        # send experience to critic
        if self.critic is not None:
            for experience in experiences:
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences


    @torch.no_grad()
    def make_experience(self, samples_list: List[Samples]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        if dist.get_rank() == 0:
            logger.info(f"🚀 Starting experience making with {len(samples_list) * dist.get_world_size()} batches")

        args = self.strategy.args
        device = torch.cuda.current_device()
        experiences = []

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        num_actions_list = [s.num_actions for s in samples_list]
        packed_seq_lens_list = [s.packed_seq_lens for s in samples_list]
        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]

        # Move data to CPU for remote processing
        sequences_cpu_list = [seq.to("cpu") for seq in sequences_list]
        attention_mask_cpu_list = [mask.to("cpu") for mask in attention_mask_list]

        # Batch call initial model
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=sequences_cpu_list,
                num_actions=num_actions_list,
                attention_mask=attention_mask_cpu_list,
                logps_allgather=[True] * len(samples_list),
                packed_seq_lens=packed_seq_lens_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(samples_list))

        # Batch call critic model
        if self.critic is not None:
            value_ref = self.critic.forward_batch.remote(
                sequences=sequences_cpu_list,
                num_actions=num_actions_list,
                attention_mask=attention_mask_cpu_list,
                packed_seq_lens=packed_seq_lens_list,
            )
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put([None] * len(samples_list))

        # Batch call reward model
        r_refs = []
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(
                    rm.forward_batch.remote(
                        sequences=sequences_cpu_list,
                        attention_mask=attention_mask_cpu_list,
                        packed_seq_lens=packed_seq_lens_list,
                        pad_sequence=[True] * len(samples_list),
                    )
                )
        else:
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                queries_list = []
                for i, (seq, packed_lens) in enumerate(zip(sequences_cpu_list, packed_seq_lens_list)):
                    if not self.packing_samples:
                        queries = self.tokenizer.batch_decode(seq, skip_special_tokens=False)
                    else:
                        sequences_list = []
                        offset = 0
                        tokens_list = seq.tolist()[0]
                        for length in packed_lens:
                            sequences_list.append(tokens_list[offset : offset + length])
                            offset += length
                        queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    queries_list.extend(queries)

                if self.custom_reward_func:
                    r = None
                    rewards_list = self.custom_reward_func(queries_list, prompts_list, labels_list)
                else:
                    rank = self.get_rank() // self.strategy.ring_attn_size
                    rm = self.remote_rm_url[rank % len(self.remote_rm_url)]
                    r = remote_rm_fn_ray.remote(rm, queries=queries_list, prompts=prompts_list, labels=labels_list)
                r_refs.append(r)
            else:
                r_refs.append(ray.put([None] * len(samples_list)))

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # Batch call actor model

        action_log_probs_list = []

        self.actor.eval()
        for seq, num_acts, attn_mask, packed_lens in zip(
            sequences_cpu_list, num_actions_list, attention_mask_cpu_list, packed_seq_lens_list
        ):
            action_log_probs = self.actor(
                seq.to(device),
                num_acts,
                attn_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_lens,
            )
            action_log_probs_list.append(action_log_probs)
        dist.barrier()
        torch.cuda.synchronize()
        self.actor.train()  # Reset model state

        actor_value_rm_time = time.time() - start_time

        # Wait for all remote calls to complete
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start

        base_action_log_probs_list, value_list = ref_values[0], ref_values[1]
        if self.remote_rm_url is not None and isinstance(rewards_list, torch.Tensor):
            rewards_list = rewards_list.chunk(len(samples_list))

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value, rewards) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list, rewards_list)
        ):
            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)

            # Broadcast rewards to all ring attention ranks when using remote RM
            rewards = [rewards]
            if self.remote_rm_url and self.strategy.ring_attn_group is not None:
                if self.strategy.ring_attn_rank == 0:
                    dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
                else:
                    dist.broadcast_object_list(
                        rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                    )
            r = rewards[0].to(device)

            if (self.initial_model is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=samples.action_mask,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            sequences = samples.sequences
            attention_mask = samples.attention_mask
            if not self.packing_samples:
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            else:
                num_actions = samples.num_actions
                packed_seq_lens = samples.packed_seq_lens
                if self.strategy.ring_attn_group is not None:
                    assert samples.pad_len is not None
                    sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                        pad_len=samples.pad_len,
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        action_log_probs=action_log_probs,
                        values=value,
                        kl=kl,
                    )
                # Convert tensor into list of tensors for easier manipulation within dataset
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                if value is not None:
                    value = unpacking_samples(value, num_actions)
                if base_action_log_probs is not None:
                    base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

            if not args.use_kl_loss:
                base_action_log_probs = None

            info = {
                "kl": kl_mean,
                "reward": r,
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "num_actions": samples.num_actions,
            }

            if self.strategy.args.perf:
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,
                action_log_probs,
                base_action_log_probs,
                value,
                None,
                None,
                attention_mask,
                samples.action_mask,
                info,
                kl,
            )

            experiences.append(experience)

        end_time = time.time()
        duration = end_time - start_time
        if dist.get_rank() == 0:
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Experience making completed in {time_str}")
        return experiences

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # get rewards from experiences
        rewards = [experience.info["reward"] for experience in experiences]

        # reward shaping
        if args.advantage_estimator == "rloo":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                    kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if kwargs["gamma"] != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    if dist.get_rank() == 0:
                        logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    kwargs["gamma"] = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    def gen_queries(self, samples):
        responses = []
        seqs = [input_ids + output_ids for s in samples for input_ids, output_ids in s.sequences]
        if not self.packing_samples:
            responses = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        else:
            packed_seq_lens_list = [s.packed_seq_lens for s in samples]
            for seq, packed_lens in zip(seqs, packed_seq_lens_list):
                sequences_list = []
                offset = 0
                tokens_list = seq.tolist()[0]
                for length in packed_lens:
                    sequences_list.append(tokens_list[offset: offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=True)
                responses.extend(queries)
        return responses

    def generate(self, prompts, swarm_ids, **generate_kwargs):
        prompts = [
            self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ) for prompt in prompts
        ]
        samples = self._generate_vllm(prompts, [str(x) for x in swarm_ids], **generate_kwargs)
        responses = [query[len(prompt):] for prompt, query in zip(prompts, self.gen_queries(samples))]
        return responses, samples

    def init_swarm(self, i):
        self.swarms[i] = SwarmFramework()
        self.swarms[i].run_level(
            model=SwarmFramework.model_config(
                'gpt-4o-mini',  # Just a placeholder, not really using.
                'xxx',
                'xxx'
            ),
            level=random.choice(['Flocking', 'Pursuit', 'Synchronization', 'Foraging', 'Transport']),
            log_dir=os.path.join(output_root, 'openrlhf'),
            max_round=100,
            height=10,
            width=10,
            num_agents=random.choice([8, 12, 16]),
            seed=random.randint(1, 100000000),
            view_size=5,
        )
    
    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """

        if self.vllm_engines is None:
            print('This should never be called!!!')
            return self._generate_with_hf(all_prompts, all_labels, **generate_kwargs)

        all_samples = []

        for i in range(10):

            all_prompts = []
            swarm_ids = []
            keys = [[] for _ in self.swarms]

            global_id = 0
            for i in range(len(self.swarms)):
                if not isinstance(self.swarms[i], SwarmFramework):
                    self.init_swarm(i)
                prompts = [(k, v) for k, v in self.swarms[i].framework.env.gen_prompts().items()]
                if len(prompts) == 0:
                    self.init_swarm(i)
                    prompts = [(k, v) for k, v in self.swarms[i].framework.env.gen_prompts().items()]
                keys[i] = [k for k, v in prompts]
                ids = [global_id + j for j in range(len(prompts))]
                global_id += len(prompts)
                all_prompts.extend(prompts)
                swarm_ids.extend(ids)

            responses, samples = self.generate([prompt for _, prompt in all_prompts], swarm_ids, **generate_kwargs)
            rewards = []

            for i in range(len(self.swarms)):
                reward = self.swarms[i].framework.env.apply_response({
                    k: response for (k, v), response in zip(all_prompts, responses) if k in keys[i]
                })
                rewards.extend(reward)

            for sample in samples:
                sample.labels = [f'{rewards[int(label)]}' for label in sample.labels]

            all_samples.extend(samples)

        return all_samples

    @torch.no_grad()
    def _generate_with_hf(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = self.get_rank() // self.strategy.ring_attn_size
        world_size = self.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Waiting for all requests to be sent
        if self.strategy.ring_attn_group is not None:
            if self.ring_rank0_group is None:
                world_size = dist.get_world_size()
                ring_rank0 = [
                    i * self.strategy.ring_attn_size for i in range(world_size // self.strategy.ring_attn_size)
                ]
                self.ring_rank0_group = dist.new_group(ranks=ring_rank0)
            dist.barrier(group=self.ring_rank0_group)
        else:
            dist.barrier()
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(e)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    sequences.append((list(output.prompt_token_ids), list(output.outputs[0].token_ids)))

                samples_list.append(
                    RawSamples(
                        sequences=sequences,
                        max_input_len=max_input_len,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        prompts=prompts,
                        labels=labels
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        return samples_list

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None