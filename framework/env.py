import asyncio

class Environment:
    def __init__(self, env_name, agents, logger):
        self.agents = agents
        self.env_name = env_name
        self.logger = logger

    def start(self):
        asyncio.run(self.loop())

    async def loop(self):
        while not self.is_done():
            for agent in self.agents.values():
                self.notify(agent)
            await self.step()
            self.update()

    async def step(self):
        tasks = {
            name: agent.decision(self.obs(agent)) for name, agent in self.agents.items()
        }
        
        # wait all the task finish
        results = await asyncio.gather(*tasks.values())
        actions = {name: result for name, result in zip(tasks.keys(), results)}
        
        # execute all the actions
        return {
            name: await self.act(self.agents[name], action) for name, action in actions.items()
        }

    async def act(self, agent, action):
        return {}

    def notify(self, agent):
        pass

    def reset(self):
        pass

    def obs(self, agent):
        return {}

    def is_done(self):
        pass

    def update(self):
        pass
