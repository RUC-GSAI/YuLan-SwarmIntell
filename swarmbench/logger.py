from util import safe_open, SafeOpenWrapper

import os
import time
import json
import numpy as np
from framework.logger import Logger
from threading import Lock, current_thread
from io import TextIOWrapper
import builtins


class SwarmLogger(Logger):

    def __init__(self, name, meta, log_dir="structured_logs"):
        super().__init__(name)
        self.log_dir = os.path.realpath(log_dir)
        with SafeOpenWrapper.get_lock(self.log_dir):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        # init logger
        timestamp = f'{int(time.time())}-{current_thread().getName()}'
        self.game_log_file = os.path.join(log_dir, f"game_log_{timestamp}.json")
        self.agent_log_file = os.path.join(log_dir, f"agent_log_{timestamp}.json")
        with safe_open(os.path.join(self.log_dir, f"meta_log.json"), "a") as f:
            pass
        with safe_open(os.path.join(self.log_dir, f"meta_log.json"), "r+") as f:
            try:
                meta_log = json.load(f)
            except json.decoder.JSONDecodeError:
                meta_log = {}
            meta_log[timestamp] = meta
            f.truncate(0)
            f.seek(0)
            json.dump(meta_log, f)

        # init log data
        self.game_logs = []
        self.agent_logs = []

    def log_game_state(self, env, round_num, timestamp):
        """Log game state"""
        # Convert numpy array to list
        grid_list = env.grid.tolist()
        
        # Get agent information
        agents_info = []
        for name, mesh in env.agent_meshes.items():
            agents_info.append({
                "id": name,
                "x": mesh.pos[1],
                "y": mesh.pos[0]
            })

        # Get messages
        messages = [m for m in env.messages if m["round"] == round_num-1]
        
        game_state = {
            "timestamp": timestamp,
            "level": env.level,
            "round": round_num,
            "grid": grid_list,
            "agents": agents_info,
            **env.levels[env.level].level_obs(None),
            "messages": messages
        }
        self.game_logs.append(game_state)
        self.save_game_logs()
    
    def log_agent_action(self, env, agent_id, prompt, response, action, message,
                         api_call_time, action_success, view, total_llm_tokens):
        """Record agent actions"""
        # Convert the vision numpy array to a list
        view_list = view.tolist() if view is not None else None
        
        agent_log = {
            "timestamp": time.time(),
            "level": env.level,
            "round": env.round,
            "agent_id": agent_id,
            "view": view_list,
            "prompt": prompt,
            "response": response,
            "action": action,
            "message": message,
            "api_call_time": api_call_time,
            "action_success": action_success,
            'total_llm_tokens': total_llm_tokens,
        }
        self.agent_logs.append(agent_log)
        self.save_agent_logs()
    
    def save_game_logs(self):
        """Save game logs"""
        with safe_open(self.game_log_file, 'w') as f:
            json.dump(self.game_logs, f, indent=2)

    def save_agent_logs(self):
        """Save agent logs"""
        with safe_open(self.agent_log_file, 'w') as f:
            json.dump(self.agent_logs, f, indent=2)

    def info(self, msg):
        """
        Output informational logs
        Overriding the info method of Logger
        """
        super().info(msg)
        print(msg)

    def debug(self, msg):
        """
        Output debug logs
        Overriding the debug method of Logger
        """
        super().debug(msg)
        if os.environ.get('DEBUG'):
            print(f"[DEBUG] {msg}")

    def error(self, msg):
        """
        Output error logs
        Overriding the error method of Logger
        """
        super().error(msg)
        print(f"[ERROR] {msg}")