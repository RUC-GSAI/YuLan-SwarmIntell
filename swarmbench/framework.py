import os
from framework.framework import Framework, ModelConfig
from swarmbench.agent import SwarmAgent
from swarmbench.environment import SwarmEnvironment
from swarmbench.logger import SwarmLogger
from contextlib import contextmanager

from queue import Queue
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

stdout = sys.stdout


@contextmanager
def silence():
    # Save the current stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
    yield  # Allow the code to run
    sys.stdout = original_stdout  # Restore stdout


def output(s):
    stdout.write(f'{s}')


class SwarmFramework:

    instances = {}
    submission = {}

    def __init__(self, name=''):
        self.name = name
        self.framework = None

    @property
    def status(self):
        if self.framework is None or self.framework.env is None:
            return 'pending'
        if self.framework.env.done:
            return 'done'
        return 'running'

    def run_task(self, model, task, log_dir=None,
                  num_agents=10, max_round=100, width=12, height=12, seed=42, view_size=9):
        if self.status != 'pending':
            raise RuntimeError(f'Cannot run task because task is already {self.status}.')
        env_name = self.name
        
        # Prepare agent parameters
        sys_prompt = "You are a agent. You need to cooperate with other agents and finish a given task."
        agent_args = {"sys_prompt": sys_prompt}
        
        # Prepare logger parameters
        meta = {
            'model': model.model if isinstance(model, ModelConfig) else [m.model for m in model],
            'task': task,
            'num_agents': num_agents,
            'max_round': max_round,
            'width': width,
            'height': height,
            'seed': seed,
            'view_size': view_size,
        }
        logger_args = {"log_dir": log_dir, "meta": meta}
        
        # Prepare environment parameters
        env_args = {"task": task, "seed": seed, "max_round": max_round,
                    'width': width, 'height': height, 'view_size': view_size}

        self.framework = Framework()
        # Use the framework to start the game
        self.framework.start(
            agent_cls=SwarmAgent,
            env_cls=SwarmEnvironment,
            logger_cls=SwarmLogger,
            env_name=env_name,
            num_agents=num_agents,
            model=model,
            agent_args=agent_args,
            logger_args=logger_args,
            env_args=env_args
        )

    @classmethod
    def model_config(cls, model, api_key, api_base):
        cfg = ModelConfig()
        cfg.api_key = api_key
        cfg.api_base = api_base
        cfg.model = model
        return cfg

    @classmethod
    def submit(cls, name, model, task, log_dir=None,
               num_agents=10, max_round=100, width=12, height=12, seed=42, view_size=9):
        kwargs = {
            'model': model,
            'task': task,
            'log_dir': log_dir,
            'num_agents': num_agents,
            'max_round': max_round,
            'width': width,
            'height': height,
            'seed': seed,
            'view_size': view_size
        }

        if name in cls.submission:
            raise ValueError(f"Name ({name}) already exists.")
        cls.submission[name] = kwargs

    @classmethod
    def run_all(cls, max_parallel=None):
        for name, args in cls.submission.items():
            cls.instances[name] = cls(name=name)

        def wrapper(name):
            cls.instances[name].run_task(**cls.submission[name])

        def daemon():
            max_name_len = max([len(name) for name in cls.submission])
            max_progress_len = max([len(f'{d["max_round"]}/{d["max_round"]}')
                                    for d in cls.submission.values()])
            fmt_str = f'{{:<{max_name_len}}} - {{:>{max_progress_len}}}'
            prev_prog = -1

            while True:
                dones = 0
                total_progress = 0
                progress = 0
                brief = []
                for name, instance in cls.instances.items():
                    total_progress += cls.submission[name]['max_round']
                    if instance.status == 'running':
                        cur_progress = instance.framework.env.round
                    elif instance.status == 'done':
                        dones += 1
                        cur_progress = cls.submission[name]['max_round']
                    else:
                        cur_progress = 0
                    progress += cur_progress
                    brief.append(fmt_str.format(name, f"{cur_progress}/{cls.submission[name]['max_round']}"))
                prog = progress / total_progress
                brief.append(f'Progress: {prog:.2%}')

                if prog != prev_prog:
                    output('\n'.join(brief))
                    output('\n')
                    prev_prog = prog

                if dones == len(cls.submission):
                    break
                time.sleep(1)

        with silence(), ThreadPoolExecutor(
                max_workers=len(cls.instances) if max_parallel is None else max_parallel
        ) as executor:
            for name in cls.instances:
                executor.submit(wrapper, name)
            daemon_thread = Thread(target=daemon)
            daemon_thread.start()
        daemon_thread.join()

        cls.instances = {}
        cls.submission = {}
