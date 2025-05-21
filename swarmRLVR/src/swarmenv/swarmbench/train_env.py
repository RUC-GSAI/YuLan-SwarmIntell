from src.swarmenv.swarmbench.environment import SwarmEnvironment
from src.swarmenv.swarmbench.logger import SwarmLogger

import asyncio

output_root = "./"
# Mute logging.
class SwarmDummyLogger(SwarmLogger):

    def __init__(self, name, meta, log_dir="structured_logs"):
        super().__init__(name, meta, log_dir)
        pass

    def log_game_state(self, env, round_num, timestamp):
        pass

    def log_agent_action(self, env, agent_id, prompt, response, action, message,
                         api_call_time, action_success, view, total_llm_tokens):
        pass

    def save_game_logs(self):
        pass

    def save_agent_logs(self):
        pass


class SwarmTrainEnv(SwarmEnvironment):

    def __init__(self, *args, **kwargs):
        self.pending = {}
        super().__init__(*args, **kwargs)

    def start(self):
        pass

    def render(self, name=None):
        if name is None:
            return ''
        return super().render(name)

    def gen_prompts(self):
        """
        Synchronously extract prompts from the environment.
        :return: A dict where key is SwarmAgent and value is prompt.
        """
        if self.is_done():
            return {}
        self.pending = {}
        for agent in self.agents.values():
            self.notify(agent)
            if agent.escaped:
                continue
            self.pending[agent] = agent.gen_prompt(self.obs(agent), agent.msgs)
            agent.msgs = {}
        return self.pending

    async def _apply_response(self, responses):
        actions = {
            agent.name: agent.to_action(res)
            for agent, res in responses.items() if agent in self.agents.values()
        }
        await asyncio.gather(*[self.act(self.agents[name], action) for name, action in actions.items()])
        self.update()
        return [self.tasks[self.task].rewards[agent] + (('move' in actions[agent.name]) + ('speak' in actions[agent.name])) * 0.1
                for agent in responses]

    def apply_response(self, responses):
        """
        Applies the responses to the agents.
        :param responses: A dict where key is SwarmAgent and value is prompt.
        :return: List of rewards.
        """
        return asyncio.run(self._apply_response(responses))


def redirect_env():
    """
    Make sure to call this again on the subprocesses when training in distributed environment.
    """
    import src.swarmenv.swarmbench.framework
    src.swarmenv.swarmbench.framework.env_cls = SwarmTrainEnv
    src.swarmenv.swarmbench.framework.logger_cls = SwarmDummyLogger
redirect_env()
