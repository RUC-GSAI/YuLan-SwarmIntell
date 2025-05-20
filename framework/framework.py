from framework.llm import Chat, UserAgent


class ModelConfig:
    model: str
    api_base: str
    api_key: str

    def as_dict(self):
        return {
            'model': self.model,
            'api_base': self.api_base,
            'api_key': self.api_key
        }


class Framework:

    def __init__(self):
        self.env = None
        self.agents = None
        self.logger = None

    def start(self, agent_cls, env_cls, logger_cls, env_name, num_agents, model,
              agent_args=None, logger_args=None, env_args=None):
        if agent_args is None:
            agent_args = {}
        
        if logger_args is None:
            logger_args = {}
            
        if env_args is None:
            env_args = {}

        if isinstance(model, ModelConfig):
            model = [model for _ in range(num_agents)]
        self.agents = {}
        for i in range(num_agents):
            agent_name = f'Agent_{i}'
            self.agents[agent_name] = agent_cls(agent_name, Chat(
                model[i].api_base, model[i].api_key, model[i].model), **agent_args)
        
        self.logger = logger_cls(env_name, **logger_args)

        self.env = env_cls(env_name, self.agents, self.logger, **env_args)

        self.env.start()

        return self.env
