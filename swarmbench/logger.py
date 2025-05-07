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

        # 初始化日志文件
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

        # 初始化日志数据
        self.game_logs = []
        self.agent_logs = []

    def log_game_state(self, env, round_num, timestamp):
        """记录游戏状态"""
        # 将numpy数组转换为列表
        grid_list = env.grid.tolist()
        
        # 获取代理信息
        agents_info = []
        for name, mesh in env.agent_meshes.items():
            agents_info.append({
                "id": name,
                "x": mesh.pos[1],
                "y": mesh.pos[0]
            })

        # 获取消息
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
        """记录代理动作"""
        # 将视野numpy数组转换为列表
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
        """保存游戏日志"""
        with safe_open(self.game_log_file, 'w') as f:
            json.dump(self.game_logs, f, indent=2)
    
    def save_agent_logs(self):
        """保存代理日志"""
        with safe_open(self.agent_log_file, 'w') as f:
            json.dump(self.agent_logs, f, indent=2)
    
    def info(self, msg):
        """
        输出信息日志
        重写Logger的info方法
        """
        super().info(msg)
        print(msg)
    
    def debug(self, msg):
        """
        输出调试日志
        重写Logger的debug方法
        """
        super().debug(msg)
        if os.environ.get('DEBUG'):
            print(f"[DEBUG] {msg}")
    
    def error(self, msg):
        """
        输出错误日志
        重写Logger的error方法
        """
        super().error(msg)
        print(f"[ERROR] {msg}")
