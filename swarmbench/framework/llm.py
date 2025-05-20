from openai import AsyncOpenAI


class Chat:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        self.model = model
        self.msg = []
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}
        self.stream = stream
        self.api_key = api_key
        self.base_url = base_url
        self.usage = 0

    def _gen_wrapper(self, response):
        tmp = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                s = chunk.choices[0].delta.content
                tmp.append(s)
                yield s
        self.msg.append({'role': 'assistant', 'content': ''.join(tmp)})

    def _format_response(self, response):
        if self.stream:
            return self._gen_wrapper(response)
        else:
            s = response.choices[0].message.content
            self.msg.append({'role': 'assistant', 'content': s})
            return s

    def jump_back(self):
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]

    def system(self, content):
        self.sys_prompt = {'role': 'system', 'content': content}

    async def generate(self, content):
        self.msg.append({'role': 'user', 'content': content})
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[self.sys_prompt] + (self.msg if self.memory else self.msg[-1:]),  # 包含系统提示和消息历史
            stream=self.stream,
            timeout=3600,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        self.usage += response.usage.completion_tokens
        await client.close()

        return self._format_response(response)


class UserAgent:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        self.system_prompt = ''
        self.usage = 0

    async def generate(self, content):
        print('SYSTEM')
        print(self.system_prompt)
        print('PROMPT')
        print(content)
        return input('>>>')

    def system(self, content):
        self.system_prompt = content
