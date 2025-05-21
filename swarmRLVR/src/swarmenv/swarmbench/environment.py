import math

import numpy as np
import random
import time
from colorama import init, Fore, Back, Style
from src.swarmenv.framework.env import Environment
from src.swarmenv.swarmbench.task import *
from src.swarmenv.swarmbench.physics import Mesh, Node

dirc_map = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}


class SwarmEnvironment(Environment):
    def __init__(self, env_name, agents, logger, task=1,
                 max_round=100, width=12, height=12, seed=42, view_size=9):
        super().__init__(env_name, agents, logger)
        self.task = task
        self.width = width
        self.height = height
        self.grid = np.full((self.height, self.width), '.', dtype=str)
        self.round = 0
        self.done = False
        self.messages = []
        self.tasks = {
            'Transport': Transport(seed),
            'Flocking': Flocking(seed),
            'Pursuit': Pursuit(seed),
            'Synchronization': Synchronization(seed),
            'Foraging': Foraging(seed)
        }
        self.push_i = {}
        self.push_j = {}
        self.actions = {}
        self.meshes = []
        self.agent_meshes = {}
        self.mesh_map = np.full_like(self.grid, None, dtype=object)
        self.max_round = max_round
        self.__grid_dtype = None
        self.seed = seed
        self.view_size = view_size
        init()
        self.reset()

    @property
    def grid_dtype(self):
        if self.__grid_dtype is None:
            self.__grid_dtype = f'<U{max([len(s) for s in self.agents]) - 5}'
        return self.__grid_dtype

    def reset(self):
        self.__grid_dtype = None
        self.grid = np.full((self.height, self.width), '.', dtype=str)
        self.mesh_map = np.full_like(self.grid, None, dtype=object)
        self.messages = []
        self.round = 0
        self.meshes = []
        self.agent_meshes = {}
        self.tasks[self.task].reset(self)

        self.logger.log_game_state(self, 0, time.time())
        print(self.render())
        return self.obs(None)

    def is_done(self):
        if not self.done:
            task_done = self.tasks[self.task].is_done(self)
            if task_done or self.round >= self.max_round:
                self.done = True
                return True
            return False
        return True

    def obs(self, agent):
        if agent is None:
            return {
                'task': self.tasks[self.task],
                'round': self.round,
                'grid': self.grid.copy(),
                **self.tasks[self.task].task_obs(None)
            }

        name = agent.name
        view = self.get_view(name)
        x, y = self.agent_meshes[name].pos[1], self.agent_meshes[name].pos[0]
        visible_messages = self.get_agent_visible_messages(name)

        return {
            'task': self.tasks[self.task],
            'round': self.round,
            'view': view,
            'position': {'x': x, 'y': y},
            'messages': visible_messages,
            'name': name,
            **self.tasks[self.task].task_obs(agent)
        }

    def notify(self, agent):
        name = agent.name
        if name not in self.agent_meshes:
            return

        for msg in self.messages:
            if msg['round'] == self.round - 1:
                speaker_name = msg['agent_name']
                if self.is_visible(name, speaker_name):
                    agent.notify(speaker_name, msg['message'])

    def is_visible(self, observer_name, target_name):
        if observer_name not in self.agent_meshes or target_name not in self.agent_meshes:
            return False

        observer_x, observer_y = self.agent_meshes[observer_name].pos[1], self.agent_meshes[observer_name].pos[0]
        target_x, target_y = self.agent_meshes[target_name].pos[1], self.agent_meshes[target_name].pos[0]

        view_size = self.view_size
        half_view = view_size // 2

        return abs(observer_x - target_x) <= half_view and abs(observer_y - target_y) <= half_view

    def get_agent_visible_messages(self, agent_name):
        if agent_name not in self.agent_meshes:
            return []

        visible_messages = []
        for msg in self.messages:
            if msg['round'] == self.round - 1:
                speaker_name = msg['agent_name']
                if self.is_visible(agent_name, speaker_name):
                    visible_messages.append(msg)

        return visible_messages

    async def act(self, agent, action):
        name = agent.name
        self.actions[name] = action
        if name not in self.agent_meshes:
            return {'success': False, 'error': 'Agent not found'}

        result = {'success': False}

        if 'move' in action and action['move'] in dirc_map:
            result['success'] = self.move_agent(name, action['move'])

        if 'speak' in action and action['speak']:
            self.messages.append({
                'round': self.round,
                'agent_name': name,
                'message': action['speak']
            })
            result['message_sent'] = True

        self.logger.log_agent_action(
            self, name,
            action.get('prompt', ''),
            action.get('response', ''),
            action.get('move', 'STAY'),
            action.get('speak', ''),
            action.get('api_call_time', 0),
            result['success'],
            self.get_view(name),
            self.agents[name].brain.usage
        )

        return result

    def move_agent(self, name, direction):
        if name not in self.agent_meshes or direction not in dirc_map:
            return False

        di, dj = dirc_map[direction]
        mesh = self.agent_meshes[name]

        (self.push_i if dj == 0 else self.push_j)[mesh] = (di if dj == 0 else dj) * 2

        return False

    def update_B(self):
        for di, dj in dirc_map.values():
            Mesh.detect_all(self.mesh_map, di, dj)
            for mesh, force in (self.push_i if dj == 0 else self.push_j).items():
                if mesh.node is None:
                    continue
                mesh.node.force = force * (di if dj == 0 else dj)
            heads = Mesh.build_dag(self.mesh_map)
            nodes = Node.pass_forward(heads)
            Mesh.move_all(self.grid, self.mesh_map, nodes, di, dj)
        self.push_i = {}
        self.push_j = {}

    def update(self):
        self.update_B()
        self.tasks[self.task].update(self, self.actions)
        for name, mesh in self.agent_meshes.items():
            i, j = mesh.pos
            if not (0 <= i < self.height and 0 <= j < self.width):
                mesh.remove(self.grid, self.mesh_map)
                self.agents[name].escaped = True
        self.round += 1
        self.logger.log_game_state(self, self.round, time.time())
        print(self.render())
        self.actions = {}

    def get_view(self, name):
        if name not in self.agent_meshes:
            return None
        x, y = self.agent_meshes[name].pos[1], self.agent_meshes[name].pos[0]
        view_size = self.view_size
        half_view = view_size // 2

        view = np.full((view_size, view_size), '.', dtype=self.grid_dtype)
        grid = self.grid.copy().astype(self.grid_dtype)
        for name, mesh in self.agent_meshes.items():
            i, j = mesh.pos
            if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                continue
            if grid[i, j] != 'A':
                grid[i, j] = '$' + name[6:]
            else:
                grid[i, j] = name[6:]

        for i in range(view_size):
            for j in range(view_size):
                grid_x = x + (j - half_view)
                grid_y = y + (i - half_view)

                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    view[i, j] = grid[grid_y, grid_x]
                else:
                    view[i, j] = '*'

        return view

    def render(self, name=None):
        if name:
            view = self.get_view(name)
            if view is None:
                return "Agent not found"

            view_str = f"Agent {name}'s View (Round {self.round}):\n"
            view_str += "  " + " ".join(str(i % 10) for i in range(view.shape[1])) + "\n"

            for i in range(view.shape[0]):
                view_str += f"{i % 10} "
                for j in range(view.shape[1]):
                    view_str += view[i, j] + " "
                view_str += "\n"

            return view_str

        else:
            grid_str = f"Global View (Round {self.round}):\n"
            grid_str += "   " + " ".join(str(i % 10) for i in range(self.width)) + "\n"

            for i in range(self.height):
                grid_str += f"{i:2d} "
                for j in range(self.width):
                    cell = self.grid[i, j]
                    if cell[0] == 'A':
                        grid_str += Back.GREEN + cell[0] + Style.RESET_ALL + " "
                    elif cell[0] == 'W':
                        grid_str += Back.RED + cell + Style.RESET_ALL + " "
                    elif cell[0] == 'B':
                        grid_str += Back.YELLOW + cell + Style.RESET_ALL + " "
                    elif cell[0] == 'X':
                        grid_str += Back.MAGENTA + cell + Style.RESET_ALL + " "
                    else:
                        grid_str += cell + " "
                grid_str += "\n"

            return grid_str


if __name__ == '__main__':
    grid = np.full((10, 10), '.', dtype=str)
    shape_h = np.array(
        [['B', 'B', 'B', 'B']]
    )
    shape_v = np.array(
        [['B'], ['B'], ['B'], ['B']]
    )
    meshes = [
        Mesh(pos=(1, 1), shape=shape_h, static=False),
        Mesh(pos=(4, 1), shape=shape_v, static=False)
    ]
    for mesh in meshes:
        grid[mesh.pos[0]:mesh.pos[0] + mesh.shape.shape[0], mesh.pos[1]:mesh.pos[1] + mesh.shape[1]] = mesh
    print(grid)
