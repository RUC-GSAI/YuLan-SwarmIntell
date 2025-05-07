from swarmbench import SwarmFramework


if __name__ == '__main__':
    name = 1
    for level in ('Transport', 'Pursuit'):
        for model in ('gpt-4o-mini', 'gpt-3.5-turbo'):
            for seed in (27, 42):
                SwarmFramework.submit(
                    f'exp_{name}',
                    SwarmFramework.model_config(model, 'REPLACE: API_KEY', 'REPLACE: API_BASE'),
                    level,
                    log_dir='REPLACE: LOG_DIR',
                    num_agents=10,
                    max_round=100,
                    width=10,
                    height=10,
                    seed=seed,
                    view_size=9
                )
                name += 1

    SwarmFramework.run_all(max_parallel=4)
