import os
import sys
import numpy as np
from colorama import init, Fore, Back, Style

#  Configuration 
DEFAULT_LOG_DIR = 'experiment_01'
DEFAULT_TIME = 0.2
DEFAULT_MAX_GRIDS_PER_ROW = 6
GRID_SEPARATOR = "  |  "
LATEX_GRID_SEPARATOR = "~~|~~"
LATEX_AGENT_VIEWS_PER_ROW = 4
DEFAULT_ANIMATION_FPS = 10
DEFAULT_GIF_DPI = 150 # Lower DPI for GIFs for smaller file sizes
DEFAULT_VIDEO_DPI = 150 # DPI for PNGs used in video

#  Additional Configuration for Metrics 
DEFAULT_EMBEDDING_MODEL = 'all-mpnet-base-v2'
MOVE_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'] # STAY is often treated specially
ACTUAL_MOVE_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'] # For metrics like move attempts
COORDINATION_ACTIONS = ACTUAL_MOVE_ACTIONS + ['STAY']
ACTION_VECTORS = {
    'UP': np.array([0, -1]),
    'DOWN': np.array([0, 1]),
    'LEFT': np.array([-1, 0]),
    'RIGHT': np.array([1, 0]),
    'STAY': np.array([0, 0])
}

#  Agent Color Definitions 
AGENT_MESSAGE_COLORS = [ Fore.LIGHTRED_EX, Fore.RED, Fore.LIGHTYELLOW_EX, Fore.YELLOW, Fore.LIGHTGREEN_EX, Fore.GREEN, Fore.LIGHTCYAN_EX, Fore.CYAN, Fore.LIGHTBLUE_EX, Fore.BLUE, Fore.LIGHTMAGENTA_EX, Fore.MAGENTA]
AGENT_MESSAGE_COLORS = list(dict.fromkeys(AGENT_MESSAGE_COLORS))
AGENT_GRID_COLORS_MAP = { Fore.LIGHTRED_EX: (Back.LIGHTRED_EX, Fore.WHITE), Fore.RED: (Back.RED, Fore.WHITE), Fore.LIGHTYELLOW_EX: (Back.LIGHTYELLOW_EX, Fore.BLACK), Fore.YELLOW: (Back.YELLOW, Fore.BLACK), Fore.LIGHTGREEN_EX: (Back.LIGHTGREEN_EX, Fore.BLACK), Fore.GREEN: (Back.GREEN, Fore.BLACK), Fore.LIGHTCYAN_EX: (Back.LIGHTCYAN_EX, Fore.BLACK), Fore.CYAN: (Back.CYAN, Fore.BLACK), Fore.LIGHTBLUE_EX: (Back.LIGHTBLUE_EX, Fore.WHITE), Fore.BLUE: (Back.BLUE, Fore.WHITE), Fore.LIGHTMAGENTA_EX: (Back.LIGHTMAGENTA_EX, Fore.BLACK), Fore.MAGENTA: (Back.MAGENTA, Fore.WHITE)}
for msg_color in AGENT_MESSAGE_COLORS:
    if msg_color not in AGENT_GRID_COLORS_MAP: print(f"{Fore.RED}Error: Color {msg_color} missing map! Exiting."); sys.exit(1)
DEFAULT_GRID_AGENT_COLOR = (Back.WHITE, Fore.BLACK)
GENERIC_AGENT_GRID_COLOR = Back.GREEN + Fore.BLACK


#  Cache directories setup 
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.game_log_visualizer_cache')
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, 'embeddings')
METRICS_CACHE_DIR = os.path.join(CACHE_DIR, 'metrics')

