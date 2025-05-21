set -x

ray stop

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://huggingface.co
# export HF_ENDPOINT=https://hf-mirror.com
# export HTTP_PROXY=http://127.0.0.1:7897
# export HTTPS_PROXY=http://127.0.0.1:7897
export HF_HUB_ENABLE_HF_TRANSFER=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #<------
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_LAUNCH_BLOCKING=1
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
#export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
source ~/.bashrc

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 4
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 # <--- recommend setting

#   --colocate_all_models \
#   --pretrain unsloth/Qwen3-1.7B-unsloth-bnb-4bit \
#   --pretrain Qwen/Qwen3-1.7B \
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "."}' \
   -- conda run -n swarmbench_RL --no-capture-output python ./src/train/train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 3 \
   --vllm_num_engines 3 \
   --vllm_tensor_parallel_size 1 \
   --pretrain unsloth/Qwen3-1.7B \
   --grad_accum_dtype bf16 \
   --remote_rm_url src/train/reward_func.py \
   --ckpt_path /root/shared-nvme/swarm_openrlhf/take/checkpoint \
   --load_checkpoint \
   --save_hf_ckpt \
   --save_path /root/shared-nvme/swarm_openrlhf/take/model \
   --save_steps 2 \
   --micro_train_batch_size 2 \
   --train_batch_size 12 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 96 \
   --max_samples 10000 \
   --max_ckpt_num 10 \
   --max_epochs 1 \
   --num_episodes 10 \
   --prompt_max_len 3071 \
   --generate_max_len 2047 \
   --zero_stage 3 \
   --vllm_gpu_memory_utilization 0.75 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data RLHFlow/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --n_samples_per_prompt 1
#   --use_wandb xxx
#   --packing_samples \