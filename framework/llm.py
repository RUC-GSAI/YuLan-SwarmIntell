from openai import AsyncOpenAI, OpenAI
import asyncio 


class Chat:
    """
    The Chat class encapsulates the interaction logic with the LLM (Large Language Model).
    It provides asynchronous message generation functionality based on the OpenAI API.
    """
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        """
        Initialize a Chat instance.
        
        Parameters:
            model: The name of the LLM model to use.
            stream: Whether to use the streaming response mode.
        """
        self.model = model  # Store the model name
        self.msg = []  # Store the message history, format: [{'role': str, 'content': str}, ...]
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}  # System prompt, default is empty
        self.stream = stream  # Whether to use streaming response
        self.api_key = api_key
        self.base_url = base_url
        self.usage = 0
        # Note: Do not create a global client during initialization to avoid event loop resource conflicts.
        # The code below has been commented out, which explains why a new client is created each time it is called.
        # This can lead to confusion with the underlying event loop and other resources, so a new one-time client is created instead of a global one.
        # self.client = AsyncOpenAI(api_key=CONFIG.api_key, base_url=CONFIG.base_url)

    def _gen_wrapper(self, response):
        """
        Wrapper function for processing the streaming response generator.
        
        Parameters:
            response: The OpenAI streaming response object.
            
        Returns:
            A generator that yields individual response chunks.
            
        Flow:
            1. Iterate through the response chunks.
            2. Collect all the content.
            3. Finally, add the complete content to the message history.
        """
        tmp = []  # Temporary storage for all response chunks
        for chunk in response:  # Iterate through each chunk in the response stream
            if chunk.choices[0].delta.content is not None:  # If the chunk contains content
                s = chunk.choices[0].delta.content  # Get the content
                tmp.append(s)  # Add it to the temporary storage
                yield s  # Yield it to the caller
        # After the streaming response is finished, add the complete content to the message history
        self.msg.append({'role': 'assistant', 'content': ''.join(tmp)})

    def _format_response(self, response):
        """
        Format the LLM response.
        
        Parameters:
            response: The OpenAI response object.
            
        Returns:
            The formatted response string or generator.
            
        Flow:
            1. Choose the appropriate processing method based on the streaming setting.
            2. Add the response to the message history.
            3. Return the formatted response.
        """
        if self.stream:  # If using streaming response
            return self._gen_wrapper(response)  # Return the generator
        else:  # If using regular response
            s = response.choices[0].message.content  # Get the response content
            self.msg.append({'role': 'assistant', 'content': s})  # Add it to the message history
            return s  # Return the response content

    def jump_back(self):
        """
        Jump back in the conversation history when an error occurs.
        Used to restore the previous state when an invalid response is generated.
        
        Flow:
            1. Delete all consecutive assistant messages.
            2. Delete the last user message.
        """
        # TODO: This method deletes the last user message when an error occurs, which may lead to a loss of conversational context.
        # Consider adding a parameter to control whether the last user message should be deleted, or record the deleted messages for later analysis.
        # Delete all consecutive assistant messages
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        # Delete the last user message
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]  # This line will delete the last user message

    def system(self, content):
        """
        Set the system prompt.
        
        Parameters:
            content: The system prompt content.
        """
        self.sys_prompt = {'role': 'system', 'content': content}  # Update the system prompt

    async def generate(self, content):
        """
        Asynchronously generate an LLM response.
        
        Parameters:
            content: The user message content.
            
        Returns:
            The LLM-generated response.
            
        Flow:
            1. Add the user message to the history.
            2. Create a one-time AsyncOpenAI client.
            3. Send the request and wait for the response.
            4. Close the client.
            5. Format and return the response.
        """
        # Add the user message to the history
        self.msg.append({'role': 'user', 'content': content})
        # print('Generating...')
        
        # Create a one-time AsyncOpenAI client
        # A new client is created each time it is called to avoid event loop resource conflicts
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Asynchronously call the OpenAI API to generate the response
        response = await client.chat.completions.create(
            model=self.model,  # Use the specified model
            messages=[self.sys_prompt] + (self.msg if self.memory else self.msg[-1:]),  # Include the system prompt and message history
            stream=self.stream,  # Whether to use streaming response
            timeout=600
        )
        self.usage += response.usage.completion_tokens
        # Close the client to release resources
        await client.close()
        # print('Generated!')
        
        # Format and return the response
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
    Demonstration of the asynchronous function used in the Chat class
    """
    chat = Chat()  # Create a Chat instance
    response = await chat.generate('What is the capital of France?')  # Asynchronously generate a response
    print(response)  # Print the response


if __name__ == '__main__':
    """
    Script execution entry point, running the demonstration function
    """
    asyncio.run(demo())  # Run the asynchronous demonstration function
    # The commented-out code is the synchronous version of the demonstration, which is currently not available
    # chat = Chat()
    # print(chat.generate('What is the capital of France?'))
