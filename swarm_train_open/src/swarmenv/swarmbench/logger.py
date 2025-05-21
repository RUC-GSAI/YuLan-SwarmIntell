from src.swarmenv.util import safe_open, SafeOpenWrapper

import os
import time
import json
from src.swarmenv.framework.logger import Logger
from threading import current_thread


class SwarmLogger(Logger):

    def __init__(self, name, meta, log_dir="structured_logs"):
        super().__init__(name)
        self.log_dir = os.path.realpath(log_dir)
        with SafeOpenWrapper.get_lock(self.log_dir):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

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

        self.game_logs = []
        self.agent_logs = []

    def log_game_state(self, env, round_num, timestamp):
        grid_list = env.grid.tolist()
        
        agents_info = []
        for name, mesh in env.agent_meshes.items():
            agents_info.append({
                "id": name,
                "x": mesh.pos[1],
                "y": mesh.pos[0]
            })

        messages = [m for m in env.messages if m["round"] == round_num-1]
        
        game_state = {
            "timestamp": timestamp,
            "task": env.task,
            "round": round_num,
            "grid": grid_list,
            "agents": agents_info,
            **env.tasks[env.task].task_obs(None),
            "messages": messages
        }
        self.game_logs.append(game_state)
        self.save_game_logs()
    
    def log_agent_action(self, env, agent_id, prompt, response, action, message,
                         api_call_time, action_success, view, total_llm_tokens):
        view_list = view.tolist() if view is not None else None
        
        agent_log = {
            "timestamp": time.time(),
            "task": env.task,
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
        with safe_open(self.game_log_file, 'w') as f:
            json.dump(self.game_logs, f, indent=2)
    
    def save_agent_logs(self):
        with safe_open(self.agent_log_file, 'w') as f:
            json.dump(self.agent_logs, f, indent=2)
    
    def info(self, msg):
        super().info(msg)
        print(msg)
    
    def debug(self, msg):
        super().debug(msg)
        if os.environ.get('DEBUG'):
            print(f"[DEBUG] {msg}")
    
    def error(self, msg):
        super().error(msg)
        print(f"[ERROR] {msg}")
