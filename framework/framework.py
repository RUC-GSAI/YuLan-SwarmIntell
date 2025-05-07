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
    """
    Framework class, used to integrate agents, environments, and loggers
    Not intended to be used as a base class, but rather to be directly instantiated to run the simulation
    """

    def __init__(self):
        self.env = None
        self.agents = None
        self.logger = None

    def start(self, agent_cls, env_cls, logger_cls, env_name, num_agents, model,
              agent_args=None, logger_args=None, env_args=None):
        """
        Start the simulation framework
        
        Parameters:
            agent_cls: Agent class
            env_cls: Environment class
            logger_cls: Logger class
            env_name: Environment name
            num_agents: Number of agents
            agent_args: Agent initialization parameters
            logger_args: Logger initialization parameters
            env_args: Environment initialization parameters
        """
        if agent_args is None:
            agent_args = {}
        
        if logger_args is None:
            logger_args = {}
            
        if env_args is None:
            env_args = {}

        if isinstance(model, ModelConfig):
            model = [model for _ in range(num_agents)]
        # create agent
        self.agents = {}
        for i in range(num_agents):
            agent_name = f'Agent_{i}'
            self.agents[agent_name] = agent_cls(agent_name, Chat(
                model[i].api_base, model[i].api_key, model[i].model), **agent_args)
        
        # create logger
        self.logger = logger_cls(env_name, **logger_args)
        
        # create environment
        self.env = env_cls(env_name, self.agents, self.logger, **env_args)
        
        # start simulation
        self.env.start()
        
        return self.env
