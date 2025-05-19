
import json
import os
import sys
import time
from collections import defaultdict, Counter
import re
import math 
import argparse
import subprocess 
import hashlib
import pickle
import numpy as np
from colorama import init, Fore, Back, Style
from utils.helper import *
from utils.constants import *
#  ANSI Code Stripping Regex 
ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import sem
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    METRICS_LIBRARIES_AVAILABLE = True
except ImportError as e:
    METRICS_LIBRARIES_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: Some libraries for calculating metrics are not available: {e}")
    print(f"To enable metrics in visualization, install them with: pip install pandas matplotlib seaborn scipy sentence-transformers scikit-learn{Style.RESET_ALL}")

#  Terminal Rendering Function (Unchanged) 
def render_terminal(grid, agent_message_colors_map, coord_to_agent_id_map):
    # (Code is identical to the previous correct version)
    try:
        if not isinstance(grid, (list, np.ndarray)) or not grid: return ["(Empty/invalid grid)"], 20
        try: grid_np = np.array(grid, dtype=object)
        except Exception as e: error_str = f"(Grid conversion err: {e})"; return [error_str], max(25, get_visual_width(error_str))
        if grid_np.ndim != 2: return [f"(Invalid grid dim {grid_np.ndim})"], 25
        height, width = grid_np.shape
        if height == 0 or width == 0: return ["(Empty HxW grid)"], 18
    except Exception as e: error_str = f"(Render Setup Error: {e})"; return [error_str], max(25, get_visual_width(error_str))
    rendered_lines = []; max_visual_width = 0
    header_line = "   " + " ".join(f"{i % 100:2d}" for i in range(width)); rendered_lines.append(header_line)
    for i in range(height):
        row_str_prefix = f"{i:2d} "; segments_for_row = []
        for j in range(width):
            try: cell = grid_np[i, j]; original_cell_str = str(cell) if cell is not None and str(cell).strip() != '' else '.'
            except IndexError: original_cell_str = '?'; cell = '?'
            color_code = ""; display_content = original_cell_str
            agent_id_at_coord = coord_to_agent_id_map.get((i, j))
            if agent_id_at_coord and agent_id_at_coord in agent_message_colors_map:
                match = re.search(r'\d+$', agent_id_at_coord); agent_num_str = match.group()[-2:] if match else '?'; display_content = agent_num_str
                message_color = agent_message_colors_map[agent_id_at_coord]
                if original_cell_str == 'a': color_code = message_color
                else: back_color, text_color = AGENT_GRID_COLORS_MAP.get(message_color, DEFAULT_GRID_AGENT_COLOR); color_code = back_color + text_color
            else:
                if original_cell_str == 'P': color_code = Back.WHITE + Fore.BLACK
                elif original_cell_str == 'Y': color_code = Back.CYAN + Fore.BLACK
                elif original_cell_str == 'W': color_code = Back.RED + Fore.WHITE
                elif original_cell_str == 'B': color_code = Back.YELLOW + Fore.BLACK
                elif original_cell_str == 'X': color_code = Back.MAGENTA + Fore.WHITE
                elif original_cell_str == 'A': color_code = GENERIC_AGENT_GRID_COLOR; display_content = 'A'
                elif original_cell_str == '.': display_content = '.'
            segment_text = "{:<2}".format(display_content)
            if color_code: segments_for_row.append(color_code + segment_text + Style.RESET_ALL)
            else: segments_for_row.append(segment_text)
        current_line = row_str_prefix + " ".join(segments_for_row); rendered_lines.append(current_line)
    max_visual_width = 0;
    for line in rendered_lines: max_visual_width = max(max_visual_width, get_visual_width(line))
    return rendered_lines, max_visual_width

#  LaTeX Color Definitions 
COLORAMA_TO_LATEX = { Fore.LIGHTRED_EX: ("termLightRed", "1.0, 0.4, 0.4"), Fore.RED: ("termRed", "0.8, 0.0, 0.0"), Fore.LIGHTYELLOW_EX:("termLightYellow", "1.0, 1.0, 0.4"), Fore.YELLOW: ("termYellow", "0.8, 0.8, 0.0"), Fore.LIGHTGREEN_EX:("termLightGreen", "0.4, 1.0, 0.4"), Fore.GREEN: ("termGreen", "0.0, 0.8, 0.0"), Fore.LIGHTCYAN_EX: ("termLightCyan", "0.4, 1.0, 1.0"), Fore.CYAN: ("termCyan", "0.0, 0.8, 0.8"), Fore.LIGHTBLUE_EX:("termLightBlue", "0.4, 0.4, 1.0"), Fore.BLUE: ("termBlue", "0.0, 0.0, 0.8"), Fore.LIGHTMAGENTA_EX:("termLightMagenta", "1.0, 0.4, 1.0"), Fore.MAGENTA: ("termMagenta", "0.8, 0.0, 0.8"), Fore.WHITE: ("termWhite", "0.9, 0.9, 0.9"), Fore.BLACK: ("termBlack", "0.1, 0.1, 0.1"), Back.LIGHTRED_EX: ("termBgLightRed", "1.0, 0.4, 0.4"), Back.RED: ("termBgRed", "0.8, 0.0, 0.0"), Back.LIGHTYELLOW_EX:("termBgLightYellow", "1.0, 1.0, 0.4"), Back.YELLOW: ("termBgYellow", "0.8, 0.8, 0.0"), Back.LIGHTGREEN_EX:("termBgLightGreen", "0.4, 1.0, 0.4"), Back.GREEN: ("termBgGreen", "0.0, 0.8, 0.0"), Back.LIGHTCYAN_EX: ("termBgLightCyan", "0.4, 1.0, 1.0"), Back.CYAN: ("termBgCyan", "0.0, 0.8, 0.8"), Back.LIGHTBLUE_EX:("termBgLightBlue", "0.4, 0.4, 1.0"), Back.BLUE: ("termBgBlue", "0.0, 0.0, 0.8"), Back.LIGHTMAGENTA_EX:("termBgLightMagenta", "1.0, 0.4, 1.0"), Back.MAGENTA: ("termBgMagenta", "0.8, 0.0, 0.8"), Back.WHITE: ("termBgWhite", "0.9, 0.9, 0.9"), Back.BLACK: ("termBgBlack", "0.1, 0.1, 0.1"), }

def generate_latex_color_definitions():
    defs = []; unique_defs = {}
    for _, (name, rgb) in COLORAMA_TO_LATEX.items():
        if name not in unique_defs: defs.append(f"\\definecolor{{{name}}}{{rgb}}{{{rgb}}}"); unique_defs[name] = True
    if "anthropicOrange" not in unique_defs:
        defs.append(f"\\definecolor{{anthropicOrange}}{{rgb}}{{0.93, 0.43, 0.2}}")
        unique_defs["anthropicOrange"] = True
    if "anthropicDarkOrange" not in unique_defs:
        defs.append(f"\\definecolor{{anthropicDarkOrange}}{{rgb}}{{0.78, 0.31, 0.09}}")
        unique_defs["anthropicDarkOrange"] = True
    if "termBgDefault" not in unique_defs: defs.append(f"\\definecolor{{termBgDefault}}{{gray}}{{0.95}}")
    
    # Add colors for metrics plots
    metric_colors = {
        "infoColor": "0.0, 0.36, 0.84",       # #0f5dd7
        "msgLengthColor": "0.12, 0.65, 0.45", # #1fa774
        "questionColor": "0.63, 0.38, 1.0",   # #a160ff
        "digitColor": "0.94, 0.51, 0.23",     # #f0833a
        "dirEntropyColor": "0.98, 0.32, 0.42", # #f9516a
        "stillnessColor": "0.2, 0.88, 0.78",  # #33e0c8
        "dominantColor": "1.0, 0.76, 0.03",   # #ffc107
        "polarizationColor": "0.54, 0.17, 0.89", # #8A2BE2
        "moveDistColor": "0.9, 0.1, 0.29",    # #e6194B
        "exploreColor": "1.0, 0.88, 0.1",     # #ffe119
        "structureColor": "0.29, 0.0, 0.51",  # #4B0082
        "pushColor": "1.0, 0.27, 0.0"         # #FF4500
    }
    
    for name, rgb in metric_colors.items():
        if name not in unique_defs:
            defs.append(f"\\definecolor{{{name}}}{{rgb}}{{{rgb}}}")
            unique_defs[name] = True
            
    return "\n".join(defs)
