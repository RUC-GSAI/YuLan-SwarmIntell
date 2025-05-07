from openai import AsyncOpenAI, OpenAI  # 导入OpenAI客户端，包括同步和异步版本
import asyncio  # 导入异步IO库，用于支持异步编程


class Chat:
    """
    Chat类封装了与LLM的交互逻辑
    提供了基于OpenAI API的异步消息生成功能
    """
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        """
        初始化Chat实例
        
        参数:
            model: 使用的LLM模型名称
            stream: 是否使用流式响应模式
        """
        self.model = model  # 存储模型名称
        self.msg = []  # 存储消息历史，格式为[{role: str, content: str}, ...]
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}  # 系统提示，默认为空
        self.stream = stream  # 是否使用流式响应
        self.api_key = api_key
        self.base_url = base_url
        self.usage = 0
        # 注意: 不在初始化时创建全局client，以避免event loop资源混乱
        # 下面的代码被注释掉了，解释了为什么每次调用都创建新的client
        # 可能会导致底层event loop等资源的混乱，因此不创建全局client，而是仅在调用时创建一次性的client
        # self.client = AsyncOpenAI(api_key=CONFIG.api_key, base_url=CONFIG.base_url)

    def _gen_wrapper(self, response):
        """
        处理流式响应的生成器包装函数
        
        参数:
            response: OpenAI流式响应对象
            
        返回:
            生成器，逐个产出响应片段
            
        流程:
            1. 遍历响应片段
            2. 收集所有内容
            3. 最后将完整内容添加到消息历史
        """
        tmp = []  # 临时存储所有响应片段
        for chunk in response:  # 遍历响应流中的每个块
            if chunk.choices[0].delta.content is not None:  # 如果块包含内容
                s = chunk.choices[0].delta.content  # 获取内容
                tmp.append(s)  # 添加到临时存储
                yield s  # 产出给调用者
        # 流式响应结束后，将完整内容添加到消息历史
        self.msg.append({'role': 'assistant', 'content': ''.join(tmp)})

    def _format_response(self, response):
        """
        格式化LLM响应
        
        参数:
            response: OpenAI响应对象
            
        返回:
            格式化后的响应字符串或生成器
            
        流程:
            1. 根据流式设置选择不同处理方式
            2. 将响应添加到消息历史
            3. 返回格式化后的响应
        """
        if self.stream:  # 如果使用流式响应
            return self._gen_wrapper(response)  # 返回生成器
        else:  # 如果使用普通响应
            s = response.choices[0].message.content  # 获取响应内容
            self.msg.append({'role': 'assistant', 'content': s})  # 添加到消息历史
            return s  # 返回响应内容

    def jump_back(self):
        """
        在生成错误时回退对话历史
        用于在生成无效响应时恢复到上一个状态
        
        流程:
            1. 删除所有连续的assistant消息
            2. 删除最后一条user消息
        """
        # TODO: 这个方法在出错时会删除最后一条user消息，可能导致对话上下文丢失。
        # 考虑添加一个参数来控制是否删除最后一条用户消息，或者记录被删除的消息以便后续分析
        # 删除所有连续的assistant消息
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        # 删除最后一条user消息
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]  # 这行会删除最后一条用户消息

    def system(self, content):
        """
        设置系统提示
        
        参数:
            content: 系统提示内容
        """
        self.sys_prompt = {'role': 'system', 'content': content}  # 更新系统提示

    async def generate(self, content):
        """
        异步生成LLM响应
        
        参数:
            content: 用户消息内容
            
        返回:
            LLM生成的响应
            
        流程:
            1. 将用户消息添加到历史
            2. 创建一次性的AsyncOpenAI客户端
            3. 发送请求并等待响应
            4. 关闭客户端
            5. 格式化并返回响应
        """
        # 将用户消息添加到历史
        self.msg.append({'role': 'user', 'content': content})
        # print('Generating...')
        
        # 创建一次性的AsyncOpenAI客户端
        # 每次调用都创建新的客户端，避免event loop资源混乱
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 异步调用OpenAI API生成响应
        response = await client.chat.completions.create(
            model=self.model,  # 使用指定的模型
            messages=[self.sys_prompt] + (self.msg if self.memory else self.msg[-1:]),  # 包含系统提示和消息历史
            stream=self.stream,  # 是否使用流式响应
            timeout=600
        )
        self.usage += response.usage.completion_tokens
        # 关闭客户端，释放资源
        await client.close()
        # print('Generated!')
        
        # 格式化并返回响应
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


async def demo():
    """
    演示Chat类使用的异步函数
    """
    chat = Chat()  # 创建Chat实例
    response = await chat.generate('What is the capital of France?')  # 异步生成响应
    print(response)  # 打印响应


if __name__ == '__main__':
    """
    脚本执行入口，运行演示函数
    """
    asyncio.run(demo())  # 运行异步演示函数
    # 注释掉的代码是同步版本的演示，当前不可用
    # chat = Chat()
    # print(chat.generate('What is the capital of France?'))
