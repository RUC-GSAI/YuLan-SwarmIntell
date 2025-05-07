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
    框架类，用于整合代理、环境和日志记录器
    不作为基类使用，而是直接实例化来运行模拟
    """

    def __init__(self):
        self.env = None
        self.agents = None
        self.logger = None

    def start(self, agent_cls, env_cls, logger_cls, env_name, num_agents, model,
              agent_args=None, logger_args=None, env_args=None):
        """
        启动模拟框架
        
        参数:
            agent_cls: 代理类
            env_cls: 环境类
            logger_cls: 日志记录器类
            env_name: 环境名称
            num_agents: 代理数量
            agent_args: 代理初始化参数
            logger_args: 日志记录器初始化参数
            env_args: 环境初始化参数
        """
        if agent_args is None:
            agent_args = {}
        
        if logger_args is None:
            logger_args = {}
            
        if env_args is None:
            env_args = {}

        if isinstance(model, ModelConfig):
            model = [model for _ in range(num_agents)]
        # 创建代理
        self.agents = {}
        for i in range(num_agents):
            agent_name = f'Agent_{i}'
            self.agents[agent_name] = agent_cls(agent_name, Chat(
                model[i].api_base, model[i].api_key, model[i].model), **agent_args)
        
        # 创建日志记录器
        self.logger = logger_cls(env_name, **logger_args)
        
        # 创建环境
        self.env = env_cls(env_name, self.agents, self.logger, **env_args)
        
        # 启动模拟
        self.env.start()
        
        return self.env
