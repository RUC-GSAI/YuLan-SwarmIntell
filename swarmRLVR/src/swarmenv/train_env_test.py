import os

if __name__ == '__main__':
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from swarmbench.train_env import *
import asyncio


if __name__ == '__main__':
    swarm = SwarmFramework()
    swarm.run_task(
        model=SwarmFramework.model_config(
            'gpt-4o-mini',
            'sk-cxNX9FBAstiFSwdkUEjlEYOSejrKh9JceOy3Ubo9gQwOLTyC',
            'https://us.ifopen.ai/v1'
        ),
        task='Transport',
        log_dir='train_test',
        max_round=10,
        height=10,
        width=10
    )

    # async def generate():
    #     return await asyncio.gather(*[k.brain.generate(v) for k, v in prompts])

    while True:
        prompts = [(k, v) for k, v in swarm.framework.env.gen_prompts().items()]
        responses = generate([prompt for _, prompt in prompts])
        # for (k, v), response in zip(prompts, responses):
        #     print(k)
        #     print(response)
        #     print('\n' * 2)
        # responses = asyncio.run(generate())
        swarm.framework.env.apply_response({k: response for (k, v), response in zip(prompts, responses)})
