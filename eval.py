from swarmbench import SwarmFramework


if __name__ == '__main__':
    name = 1
    for task in ('Transport', 'Pursuit'): ########## Task Names, avaliable: {'Transport', 'Pursuit', 'Synchronization', 'Foraging', 'Flocking'}
        for model in ('gpt-4o-mini',): ########### Models
            for seed in (27, 42):
                SwarmFramework.submit(
                    f'exp_{name}',
                    SwarmFramework.model_config(model, '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', 'https://openrouter.ai/api/v1'), ########## API
                    task,
                    log_dir='./logs', ########## Logging
                    num_agents=10,
                    max_round=100,
                    width=10,
                    height=10,
                    seed=seed,
                    view_size=5
                )
                name += 1

    SwarmFramework.run_all(max_parallel=4)
