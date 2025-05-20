from traceback import print_exc


class Agent:

    def __init__(self, name, brain, sys_prompt):
        self.name = name
        self.brain = brain
        self.msgs = {}
        self.sys_prompt = sys_prompt
        self.brain.system(self.sys_prompt)

    async def decision(self, obs):
        while True:
            try:
                res = self.to_action(await self.brain.generate(self.gen_prompt(obs, self.msgs)))
                break
            except Exception as e:
                print_exc()
        # print(res)
        self.msgs = {}
        return res

    def notify(self, name, msg):
        self.msgs[name] = msg

    def gen_prompt(self, obs, msgs):
        return ''

    def to_action(self, response):
        return {}
