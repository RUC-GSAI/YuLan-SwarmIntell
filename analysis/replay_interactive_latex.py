
# python analysis/replay_interactive_latex.py -v --log-dir experiment_v01

import json
import os
import sys
import time
from collections import defaultdict
import re
import math # For ceiling division
import argparse
import platform # To check OS
import atexit

#  Third-party Imports 
import numpy as np
from colorama import init, Fore, Back, Style, Cursor

#  Try importing readchar 
try:
    import readchar
    # Define common ANSI escape codes for arrow keys
    # These might vary slightly, but are common defaults
    ARROW_UP    = "\x1b[A"
    ARROW_DOWN  = "\x1b[B"
    ARROW_RIGHT = "\x1b[C"
    ARROW_LEFT  = "\x1b[D"
    # Readchar might map these differently on some systems (e.g., Windows)
    # We will check both readchar constants and these sequences
except ImportError:
    print(f"{Fore.RED}Error: The 'readchar' library is required for interactive mode.")
    print("Please install it using: pip install readchar")
    sys.exit(1)

#  Try importing pyperclip for clipboard functionality 
try:
    import pyperclip
except ImportError:
    print(f"{Fore.YELLOW}Warning: The 'pyperclip' library is needed for copy-to-clipboard functionality.")
    print("Install it using: pip install pyperclip")
    pyperclip = None # Set to None if import fails

#  Configuration 
DEFAULT_LOG_DIR = 'experiment_v37'
DEFAULT_TIME = 0.2 # Currently unused due to manual control
DEFAULT_MAX_GRIDS_PER_ROW = 6 # For terminal display of agent views if global is shown
GRID_SEPARATOR = "  |  "
LATEX_GRID_SEPARATOR = "~~|~~"
LATEX_AGENT_VIEWS_PER_ROW = 4

#  Colorama Initialization 
init(autoreset=True)

#  Agent Color Definitions 
AGENT_MESSAGE_COLORS = [ Fore.LIGHTRED_EX, Fore.RED, Fore.LIGHTYELLOW_EX, Fore.YELLOW, Fore.LIGHTGREEN_EX, Fore.GREEN, Fore.LIGHTCYAN_EX, Fore.CYAN, Fore.LIGHTBLUE_EX, Fore.BLUE, Fore.LIGHTMAGENTA_EX, Fore.MAGENTA]
AGENT_MESSAGE_COLORS = list(dict.fromkeys(AGENT_MESSAGE_COLORS))
AGENT_GRID_COLORS_MAP = { Fore.LIGHTRED_EX: (Back.LIGHTRED_EX, Fore.WHITE), Fore.RED: (Back.RED, Fore.WHITE), Fore.LIGHTYELLOW_EX: (Back.LIGHTYELLOW_EX, Fore.BLACK), Fore.YELLOW: (Back.YELLOW, Fore.BLACK), Fore.LIGHTGREEN_EX: (Back.LIGHTGREEN_EX, Fore.BLACK), Fore.GREEN: (Back.GREEN, Fore.BLACK), Fore.LIGHTCYAN_EX: (Back.LIGHTCYAN_EX, Fore.BLACK), Fore.CYAN: (Back.CYAN, Fore.BLACK), Fore.LIGHTBLUE_EX: (Back.LIGHTBLUE_EX, Fore.WHITE), Fore.BLUE: (Back.BLUE, Fore.WHITE), Fore.LIGHTMAGENTA_EX: (Back.LIGHTMAGENTA_EX, Fore.BLACK), Fore.MAGENTA: (Back.MAGENTA, Fore.WHITE)}
for msg_color in AGENT_MESSAGE_COLORS:
    if msg_color not in AGENT_GRID_COLORS_MAP: print(f"{Fore.RED}Error: Color {msg_color} missing map! Exiting."); sys.exit(1)
DEFAULT_GRID_AGENT_COLOR = (Back.WHITE, Fore.BLACK)
GENERIC_AGENT_GRID_COLOR = Back.GREEN + Fore.BLACK

#  ANSI Code Stripping Regex 
ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def get_visual_width(text):
    """Calculates the visible width of a string by removing ANSI escape codes."""
    return len(ansi_escape_pattern.sub('', text))

def pad_visual_width(text, width):
    """Pads a string with spaces to reach a desired visual width."""
    current_visual_width = get_visual_width(text)
    padding = max(0, width - current_visual_width)
    return text + ' ' * padding

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
    """Generates LaTeX \\definecolor commands."""
    defs = []; unique_defs = {}
    for _, (name, rgb) in COLORAMA_TO_LATEX.items():
        if name not in unique_defs: defs.append(f"\\definecolor{{{name}}}{{rgb}}{{{rgb}}}"); unique_defs[name] = True
    if "termBgDefault" not in unique_defs: defs.append(f"\\definecolor{{termBgDefault}}{{gray}}{{0.95}}")
    return "\n".join(defs)

def render_latex_frame(step_data, agent_message_colors_map, coord_to_agent_id_map,
                       views_by_round, all_agent_ids_in_log, game_info, timestamp,
                       cmd_show_views, # From command line args
                       show_global_map_toggle, show_agent_views_toggle, show_messages_toggle, # Toggles
                       messages_by_round, round_num_int,
                       current_step_index, # <<< 新增参数
                       generate_content_only=False):
    """
    Generates a LaTeX string representation of the current game state frame.
    If generate_content_only is True, outputs only the resizebox/minipage content.
    """
    grid_data = step_data.get('grid'); agents_list = step_data.get('agents'); round_num = step_data['round']
    latex_string = ""
    if not generate_content_only:
        latex_string += "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
        latex_string += "\\usepackage[table]{xcolor}\n\\usepackage{graphicx}\n\\usepackage{geometry}\n\\usepackage{float}\n"  # 添加 float 包
        latex_string += "\\geometry{a4paper, margin=0.2cm, top=0.2cm, bottom=0.2cm, left=0.2cm, right=0.2cm}\n"
        latex_string += "\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{0pt}\n\\definecolor{darkgray}{gray}{0.3}\n\\definecolor{lightgray}{gray}{0.9}\n"
        latex_string += generate_latex_color_definitions() + "\n\n"
        latex_string += "\\newcommand{\\termcell}[3]{% BgC, FgC, T\n  \\multicolumn{1}{@{}>{\\columncolor{#1}}c@{}|}{\\textcolor{#2}{\\texttt{\\detokenize{#3}}}}}\n"
        latex_string += "\\newcommand{\\termcellfg}[2]{% FgC, T\n  \\multicolumn{1}{@{}>{\\columncolor{lightgray}}c@{}|}{\\textcolor{#1}{\\texttt{\\detokenize{#2}}}}}\n"
        latex_string += "\\newcommand{\\termcellbg}[2]{% BgC, T\n  \\multicolumn{1}{@{}>{\\columncolor{#1}}c@{}|}{\\texttt{\\detokenize{#2}}}}\n"
        latex_string += "\\newcommand{\\termcelldef}[1]{% T\n  \\multicolumn{1}{@{}>{\\columncolor{lightgray}}c@{}|}{\\texttt{\\detokenize{#1}}}}\n"
        latex_string += "\\begin{document}\n\n"

    def sort_key(agent_id): match = re.search(r'\d+$', agent_id); return int(match.group()) if match else agent_id
    def render_single_grid_latex(grid, local_agent_message_colors_map, local_coord_to_agent_id_map):
        grid_latex = ""; width = 0
        try:
            if not isinstance(grid, (list, np.ndarray)) or not grid: return "\\texttt{(Empty/inv grid)}\n", 1
            grid_np = np.array(grid, dtype=object)
            if grid_np.ndim != 2: return f"\\texttt{{(Invalid dim {grid_np.ndim})}}\n", 1
            height, width = grid_np.shape
            if height == 0 or width == 0: return "\\texttt{(Empty HxW)}\n", 1
            col_spec = "l|" + "c" * width + "|"; grid_latex += f"\\setlength{{\\tabcolsep}}{{3pt}}%"; grid_latex += f"\\renewcommand{{\\arraystretch}}{{1.0}}%\n"
            grid_latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"; header_row = "   & "+" & ".join(f"\\texttt{{{i%100:<2d}}}" for i in range(width))+" \\\\\n"
            grid_latex += header_row; grid_latex += "\\hline\n"
            for i in range(height):
                row_latex = f"\\texttt{{{i:<2d}}} & "; row_cells = []
                for j in range(width):
                    try: cell = grid_np[i, j]; original_cell_str = str(cell) if cell is not None and str(cell).strip() != '' else '.'
                    except IndexError: original_cell_str = '?'; cell = '?'
                    latex_cell_cmd="\\termcelldef"; latex_params=["{.}"]
                    agent_id_at_coord = local_coord_to_agent_id_map.get((i,j)); display_content=original_cell_str
                    safe_display_content = str(display_content).replace('_','\\_').replace('{','\\{').replace('}','\\}')
                    if agent_id_at_coord and agent_id_at_coord in local_agent_message_colors_map:
                        match=re.search(r'\d+$', agent_id_at_coord); agent_num_str=match.group()[-2:] if match else '?'; display_content=agent_num_str
                        safe_display_content = str(display_content).replace('_','\\_').replace('{','\\{').replace('}','\\}')
                        message_color = local_agent_message_colors_map[agent_id_at_coord]
                        if original_cell_str == 'a': fg_color_name,_=COLORAMA_TO_LATEX.get(message_color,("termBlack","")); latex_cell_cmd="\\termcellfg"; latex_params=[f"{{{fg_color_name}}}", f"{{{safe_display_content:<2}}}"]
                        else: grid_back_color, grid_text_color = AGENT_GRID_COLORS_MAP.get(message_color, DEFAULT_GRID_AGENT_COLOR); bg_color_name,_=COLORAMA_TO_LATEX.get(grid_back_color,("termBgWhite","")); fg_color_name,_=COLORAMA_TO_LATEX.get(grid_text_color,("termBlack","")); latex_cell_cmd="\\termcell"; latex_params=[f"{{{bg_color_name}}}", f"{{{fg_color_name}}}", f"{{{safe_display_content:<2}}}"]
                    else:
                        color_tuple = None
                        if original_cell_str=='P': color_tuple=(Back.WHITE, Fore.BLACK)
                        elif original_cell_str=='Y': color_tuple=(Back.CYAN, Fore.BLACK)
                        elif original_cell_str=='W': color_tuple=(Back.RED, Fore.WHITE)
                        elif original_cell_str=='B': color_tuple=(Back.YELLOW, Fore.BLACK)
                        elif original_cell_str=='X': color_tuple=(Back.MAGENTA, Fore.WHITE)
                        elif original_cell_str=='A': color_tuple=(GENERIC_AGENT_GRID_COLOR[0], GENERIC_AGENT_GRID_COLOR[1]); safe_display_content='A'
                        elif original_cell_str=='.': safe_display_content='.'
                        if color_tuple:
                            back_c, fore_c = color_tuple
                            if back_c and fore_c: bg_name,_=COLORAMA_TO_LATEX.get(back_c,("lightgray","")); fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack","")); latex_cell_cmd="\\termcell"; latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{safe_display_content:<2}}}"]
                            elif back_c: bg_name,_=COLORAMA_TO_LATEX.get(back_c,("lightgray","")); latex_cell_cmd="\\termcellbg"; latex_params=[f"{{{bg_name}}}", f"{{{safe_display_content:<2}}}"]
                            elif fore_c: fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack","")); latex_cell_cmd="\\termcellfg"; latex_params=[f"{{{fg_name}}}", f"{{{safe_display_content:<2}}}"]
                        else: latex_cell_cmd="\\termcelldef"; latex_params=[f"{{{safe_display_content:<2}}}"]
                    padded_safe_content = "{:<2}".format(safe_display_content)
                    if latex_cell_cmd=="\\termcell": latex_params=[latex_params[0],latex_params[1],f"{{{padded_safe_content}}}"]
                    elif latex_cell_cmd=="\\termcellfg" or latex_cell_cmd=="\\termcellbg": latex_params=[latex_params[0],f"{{{padded_safe_content}}}"]
                    else: latex_params=[f"{{{padded_safe_content}}}"]
                    cell_str=f"{latex_cell_cmd}{''.join(latex_params)}"; row_cells.append(cell_str)
                row_latex += " & ".join(row_cells) + " \\\\\n"; grid_latex += row_latex
            grid_latex += "\\hline\n\\end{tabular}\n"; return grid_latex, width
        except Exception as e: escaped_error=str(e).replace('{','\\{').replace('}','\\}').replace('_','\\_'); return f"\\texttt{{(Err grid: {escaped_error})}}\n", 1

    # 使用 [H] 选项固定位置，并添加居中对齐
    latex_string += "\\begin{figure}[H] % Start of fixed position figure\n"
    latex_string += "\\centering\n"  # 确保内容居中对齐
    latex_string += "\\resizebox{0.98\\textwidth}{!}{%\n"; latex_string += "\\begin{minipage}{\\textwidth}\n\n"
    
    global_header=None; global_table=None; agent_view_latex_data=[]; grid_section_string=""; has_printed_grid=False; message_block=""
    if show_global_map_toggle:
        try: global_header_text="\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ Global\\ Map \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ "; table_code,_=render_single_grid_latex(grid_data,agent_message_colors_map,coord_to_agent_id_map); global_header=f"\\texttt{{{global_header_text}}}"; global_table=table_code
        except Exception as e: error_latex=f"\\texttt{{(Err prep global R{round_num}: {e})}}"; global_header=f"\\texttt{{Global Err R{round_num}}}"; global_table=error_latex
    if cmd_show_views and show_agent_views_toggle:
        agent_views_this_round = views_by_round.get(round_num_int, {})
        if agent_views_this_round:
            agent_ids_in_step={aid for aid in coord_to_agent_id_map.values() if aid}
            try: agents_with_views_in_step=sorted(list(agent_ids_in_step & agent_views_this_round.keys()),key=sort_key); other_agents_with_views=sorted([aid for aid in agent_views_this_round if aid not in agent_ids_in_step],key=sort_key); agents_to_render_views=agents_with_views_in_step+other_agents_with_views
            except Exception: agents_to_render_views = sorted(list(agent_views_this_round.keys()))
            for agent_id in agents_to_render_views:
                view_grid = agent_views_this_round.get(agent_id)
                if view_grid:
                    try: agent_color_fore=agent_message_colors_map.get(agent_id, Fore.WHITE); latex_fg_color_name,_=COLORAMA_TO_LATEX.get(agent_color_fore,("termWhite","")); safe_agent_id_header=agent_id.replace('_','\\_'); view_header_text=f"\\ \\ \\ \\ \\ \\ View\\ {safe_agent_id_header}\\ \\ \\ \\ \\ \\ "; latex_header_str=f"\\textcolor{{{latex_fg_color_name}}}{{{{\\texttt{{{view_header_text}}}}}}}"; view_table_latex,view_cols=render_single_grid_latex(view_grid,agent_message_colors_map,{}); agent_view_latex_data.append((latex_header_str,view_table_latex,view_cols))
                    except Exception as e: safe_agent_id_err=agent_id.replace('_','\\_'); error_latex=f"\\texttt{{(Err prep view {safe_agent_id_err}: {e})}}"; agent_view_latex_data.append((f"\\texttt{{View {safe_agent_id_err} Err}}",error_latex,1))

    if global_table: grid_section_string += "\\noindent\n"+global_header+" \\\\\n"+global_table+"\n\\vspace{1.5em}\n\n"; has_printed_grid=True
    if agent_view_latex_data:
        num_agent_grids=len(agent_view_latex_data); num_agent_chunks=math.ceil(num_agent_grids/LATEX_AGENT_VIEWS_PER_ROW)
        for chunk_index in range(num_agent_chunks):
            start_index=chunk_index*LATEX_AGENT_VIEWS_PER_ROW; end_index=start_index+LATEX_AGENT_VIEWS_PER_ROW; current_chunk_data=agent_view_latex_data[start_index:end_index]
            if not current_chunk_data: continue
            num_cols_this_chunk=len(current_chunk_data)
            grid_section_string += "\\noindent\n{\\setlength{\\tabcolsep}{1pt}%\n"; grid_section_string += "\\begin{tabular}{"+" ".join(["l"]*num_cols_this_chunk)+"} \n"
            header_cells=[h for h,_,_ in current_chunk_data]; grid_section_string+=" & ".join(header_cells)+" \\\\\n"
            grid_table_cells=[g for _,g,_ in current_chunk_data]; grid_section_string+=" & ".join(grid_table_cells)+" \\\\\n"; grid_section_string+="\\end{tabular}}\n\n\\vspace{1em}\n\n"; has_printed_grid=True
    if not has_printed_grid and (show_global_map_toggle or (cmd_show_views and show_agent_views_toggle)): grid_section_string += "\\texttt{(No grids selected/found)}\n\n\\vspace{1em}\n\n"

    if grid_section_string: latex_string += "{%\n\\centering\n"+grid_section_string+"\n}%\n"

    if show_messages_toggle:
        agent_messages_this_round = messages_by_round.get(round_num_int, {})
        if agent_messages_this_round or (not has_printed_grid and not show_global_map_toggle and not (cmd_show_views and show_agent_views_toggle)):
             safe_msg_header=" Agent Messages ".replace('_','\\_'); message_block+=f"\\texttt{{{safe_msg_header}}} \\\\\n"
             if agent_messages_this_round:
                agent_ids_in_step={aid for aid in coord_to_agent_id_map.values() if aid}
                try: agents_msg_in_step=sorted(list(agent_ids_in_step&agent_messages_this_round.keys()),key=sort_key); other_agents_with_msg=sorted([aid for aid in agent_messages_this_round if aid not in agent_ids_in_step],key=sort_key); agents_to_display_msg=agents_msg_in_step+other_agents_with_msg
                except Exception: agents_to_display_msg = sorted(list(agent_messages_this_round.keys()))
                msg_count = 0
                for agent_id in agents_to_display_msg:
                    message=agent_messages_this_round.get(agent_id,""); cleaned_message=' '.join(str(message).split())
                    if not cleaned_message: continue
                    agent_color_fore=agent_message_colors_map.get(agent_id,Fore.WHITE); latex_fg_color_name,_=COLORAMA_TO_LATEX.get(agent_color_fore,("termWhite",""))
                    escaped_message=cleaned_message.replace('\\','\\textbackslash{}').replace('%','\\%').replace('#','\\#').replace('&','\\&')
                    safe_agent_id_msg=agent_id.replace('_','\\_'); part1=f"\\textcolor{{{latex_fg_color_name}}}{{{{\\texttt{{{safe_agent_id_msg}:}}}}}}"; part2=f"\\texttt{{\\detokenize{{{escaped_message}}}}}"; message_block+=f"{part1} {part2} \\\\\n"; msg_count+=1
                if msg_count==0: message_block+="\\texttt{(No valid messages)}\\\\\n"
             else: message_block+="\\texttt{(No messages this round)}\\\\\n"
    elif not has_printed_grid: message_block+="\\texttt{(Grids and Messages hidden)}\n"

    if message_block:
        if has_printed_grid: latex_string += "\\vspace{1em}\n"
        latex_string += "\\noindent\n"+message_block

    # 移除caption和label，仅保留figure环境结束
    latex_string += "\n\\end{minipage}%\n"; latex_string += "} % End resizebox\n"
    latex_string += "\\end{figure}\n"  # 简化图形环境结束

    if not generate_content_only: latex_string += "\n\\end{document}\n"
    return latex_string
#  New function to batch export LaTeX frames 
def batch_export_latex_frames(game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log, 
                             info, timestamp, cmd_show_views, output_dir, messages_by_round, prefix="frame"):
    """批量导出所有帧的LaTeX文件到指定目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Exporting {len(game_steps)} frames to {output_dir}...")
    
    for i, step in enumerate(game_steps):
        if not isinstance(step, dict) or 'round' not in step:
            print(f"Skipping invalid step at index {i}")
            continue
            
        round_num = step['round']
        try:
            round_num_int = int(round_num)
        except (ValueError, TypeError):
            print(f"Skipping step {i} with invalid round '{round_num}'")
            continue
        
        # 准备坐标到代理ID的映射
        coord_to_agent_id_map = {}
        agents_list = step.get('agents', [])
        if isinstance(agents_list, list):
            for agent in agents_list:
                if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                    try:
                        y = int(agent['y'])
                        x = int(agent['x'])
                        agent_id_str = str(agent['id'])
                        coord_to_agent_id_map[(y, x)] = agent_id_str
                    except (ValueError, TypeError):
                        continue
        
        # 生成LaTeX内容
        try:
            latex_content = render_latex_frame(
                step, agent_message_colors_map, coord_to_agent_id_map,
                views_by_round, all_agent_ids_in_log, info, timestamp,
                cmd_show_views, True, True, True,  # 启用全局地图、代理视图和消息
                messages_by_round, round_num_int, i,
                generate_content_only=False  # 生成完整的LaTeX文档
            )
            
            # 创建带有填充的文件名（确保按顺序排序）
            filename = f"{prefix}_{i+1:04d}.tex"
            filepath = os.path.join(output_dir, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)
                
            print(f"Exported frame {i+1}/{len(game_steps)}: {filepath}")
            
        except Exception as e:
            print(f"Error exporting frame {i+1}: {e}")
    
    print(f"Export complete. {len(game_steps)} frames exported to {output_dir}")
    return True

def convert_latex_to_video(latex_dir, output_video, fps=10):
    """
    将LaTeX文件转换为视频。
    需要系统安装：pdflatex, pdf2image (使用pip安装), ffmpeg
    """
    import subprocess
    import glob
    from pdf2image import convert_from_path
    
    print(f"Converting LaTeX files to PDF...")
    # 1. 编译所有LaTeX文件为PDF
    latex_files = sorted(glob.glob(os.path.join(latex_dir, "*.tex")))
    pdf_dir = os.path.join(latex_dir, "pdf")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    for i, latex_file in enumerate(latex_files):
        base_name = os.path.basename(latex_file).replace('.tex', '')
        pdf_file = os.path.join(pdf_dir, f"{base_name}.pdf")
        
        print(f"Compiling {i+1}/{len(latex_files)}: {latex_file}")
        try:
            # 使用pdflatex编译
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", pdf_dir, latex_file],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {latex_file}: {e}")
            continue
    
    # 2. 将PDF转换为图像
    print(f"Converting PDFs to images...")
    img_dir = os.path.join(latex_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    for i, pdf_file in enumerate(pdf_files):
        base_name = os.path.basename(pdf_file).replace('.pdf', '')
        img_file = os.path.join(img_dir, f"{base_name}.png")
        
        print(f"Converting {i+1}/{len(pdf_files)}: {pdf_file}")
        try:
            # 使用pdf2image转换PDF为PNG
            images = convert_from_path(pdf_file, dpi=300)
            if images:
                images[0].save(img_file, 'PNG')
        except Exception as e:
            print(f"Error converting {pdf_file}: {e}")
            continue
    
    # 3. 使用ffmpeg将图像序列合并为视频
    print(f"Creating video from images...")
    try:
        # 修改的ffmpeg命令以确保视频宽度和高度是偶数
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), 
            "-pattern_type", "glob", "-i", os.path.join(img_dir, "*.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽度和高度是偶数
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
        ], check=True)
        print(f"Video created successfully: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"Conversion failed!")
        return False
#  Main Execution Logic 
if __name__ == '__main__':
    #  Argument Parsing 
    parser = argparse.ArgumentParser(description='Replay game logs with interactive controls and LaTeX export.')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help=f'Log directory (default: {DEFAULT_LOG_DIR})')
    parser.add_argument('--time', type=float, default=DEFAULT_TIME, help=f'Time delay (inactive)')
    parser.add_argument('--show-views', '-v', action='store_true', help='Allow display/toggle of individual agent views.')
    parser.add_argument('--max-grids', type=int, default=DEFAULT_MAX_GRIDS_PER_ROW,
                        help=f'Max AGENT grids per horizontal row in terminal (default: {DEFAULT_MAX_GRIDS_PER_ROW}). LaTeX uses {LATEX_AGENT_VIEWS_PER_ROW}.')
    parser.add_argument('--debug', action='store_true', help='Print debug information.')
    # 新增导出相关参数
    parser.add_argument('--export-all', action='store_true', help='Export all frames as LaTeX files')
    parser.add_argument('--output-dir', type=str, default='latex_frames', help='Directory to save exported LaTeX files')
    parser.add_argument('--create-video', action='store_true', help='Create video from exported LaTeX files')
    parser.add_argument('--video-file', type=str, default='output_video.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for output video')
    args = parser.parse_args()

    log_dir = args.log_dir; TIME = args.time; cmd_show_views = args.show_views
    max_grids_per_row_terminal = args.max_grids; debug_mode = args.debug

    print(f"Log Dir: {log_dir}, Allow Views: {cmd_show_views}, Max Grids/Row (Term): {max_grids_per_row_terminal}, Debug: {debug_mode}")
    print("-" * 30); time.sleep(0.5)

    #  Load Metadata 
    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path):
        print(f"{Fore.RED}Meta log not found: {meta_log_path}")
        sys.exit(1)
    try:
        with open(meta_log_path) as f:
             meta = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error reading meta log {meta_log_path}: {e}")
        sys.exit(1)

    #  Game Selection 
    first_timestamp = next(iter(meta), None)
    if not first_timestamp: print(f"{Fore.RED}No games found."); sys.exit(1)
    timestamp = first_timestamp; info = meta[timestamp]
    print(f"{Style.BRIGHT}Loading game: {timestamp}: {info}{Style.RESET_ALL}")
    game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')
    agent_log_path = os.path.join(log_dir, f'agent_log_{timestamp}.json')

    #  Load Game Steps & Agent Data 
    game_steps = []; messages_by_round = defaultdict(dict); views_by_round = defaultdict(dict); all_agent_ids_in_log = set()
    try:
        if os.path.exists(game_log_path):
             with open(game_log_path, encoding='utf-8') as f: game_steps = json.load(f)
        else: print(f"{Fore.RED}Game log missing."); sys.exit(1)
        if not game_steps: print(f"{Fore.RED}Game log empty."); sys.exit(1)
        if os.path.exists(agent_log_path):
            with open(agent_log_path, encoding='utf-8') as f: agent_records = json.load(f)
            for record in agent_records:
                if isinstance(record, dict) and 'round' in record and 'agent_id' in record:
                    try:
                        round_num_int = int(record['round']); agent_id = str(record['agent_id']); all_agent_ids_in_log.add(agent_id)
                        if msg := record.get('message'): messages_by_round[round_num_int][agent_id] = msg
                        if view := record.get('view'):
                            if isinstance(view, list) and all(isinstance(row, list) for row in view): views_by_round[round_num_int][agent_id] = view
                    except (ValueError, TypeError): pass
        else: print(f"{Fore.YELLOW}Warn: Agent log missing.")
    except Exception as e: print(f"{Fore.RED}Fatal Err loading logs: {e}"); sys.exit(1)

    #  Assign Colors 
    def sort_key(agent_id): match = re.search(r'\d+$', agent_id); return int(match.group()) if match else agent_id
    try: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log), key=sort_key)
    except Exception: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log))
    agent_message_colors_map = {}
    num_unique_colors = len(AGENT_MESSAGE_COLORS)
    if num_unique_colors > 0:
        for i, agent_id in enumerate(sorted_agent_ids_list): agent_message_colors_map[agent_id] = AGENT_MESSAGE_COLORS[i % num_unique_colors]

    #  Export All LaTeX Frames if requested 
    if args.export_all:
        export_success = batch_export_latex_frames(
            game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log,
            info, timestamp, cmd_show_views, args.output_dir, messages_by_round
        )
        
        # If video creation is also requested
        if args.create_video and export_success:
            try:
                from pdf2image import convert_from_path
                video_success = convert_latex_to_video(args.output_dir, args.video_file, args.fps)
                if video_success:
                    print(f"Video creation complete: {args.video_file}")
                else:
                    print("Failed to create video. Check error messages above.")
            except ImportError:
                print("Error: The 'pdf2image' library is required for video creation.")
                print("Please install it using: pip install pdf2image")
                print("Also ensure you have pdflatex and ffmpeg installed on your system.")
        

        sys.exit(0)

    #  Interactive Replay Loop 
    current_step_index = 0; paused = True; latex_mode = False
    last_total_lines_printed = 0; needs_redraw = True; current_latex_output = ""
    copy_confirmation_msg = ""
    show_global_map_toggle = True; show_agent_views_toggle = True; show_messages_toggle = True

    while True:
        #  Step Boundary Check 
        if current_step_index < 0: current_step_index = 0
        if current_step_index >= len(game_steps): current_step_index = len(game_steps) - 1

        #  Get Step Data (Corrected Error Handling) 
        step = game_steps[current_step_index]
        if not isinstance(step, dict) or 'round' not in step:
            copy_confirmation_msg = f"{Fore.YELLOW}Invalid step data at index {current_step_index}. Skipping.{Style.RESET_ALL}"
            needs_redraw = True
            if current_step_index < len(game_steps) - 1: current_step_index += 1
            elif current_step_index > 0: current_step_index -= 1
            else: print(f"{Fore.RED}Cannot proceed from invalid step 0.{Style.RESET_ALL}"); break
            continue

        round_num = step['round']
        try: round_num_int = int(round_num)
        except (ValueError, TypeError):
            copy_confirmation_msg = f"{Fore.YELLOW}Invalid round number '{round_num}' at step {current_step_index}. Skipping.{Style.RESET_ALL}"
            needs_redraw = True
            if current_step_index < len(game_steps) - 1: current_step_index += 1
            elif current_step_index > 0: current_step_index -= 1
            else: print(f"{Fore.RED}Cannot proceed from step 0 with invalid round.{Style.RESET_ALL}"); break
            continue

        #  Display Logic 
        if needs_redraw:
            print('\033[H\033[J', end='') # Clear Screen
            current_lines_printed = 0
            display_message = copy_confirmation_msg; copy_confirmation_msg = "" # Show status message once

            if latex_mode:
                # (LaTeX rendering logic unchanged)
                print(f"{Style.BRIGHT} LaTeX Mode  (Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num}){Style.RESET_ALL}")
                print("Generating LaTeX snippet...")
                sys.stdout.flush()
                grid_data = step.get('grid'); agents_list = step.get('agents'); coord_to_agent_id_map = {}
                if isinstance(agents_list, list):
                    for agent in agents_list:
                        if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                            try: y = int(agent['y']); x = int(agent['x']); agent_id_str = str(agent['id']); coord_to_agent_id_map[(y, x)] = agent_id_str
                            except (ValueError, TypeError): continue
                try:
                    current_latex_output = render_latex_frame(
                        step, agent_message_colors_map, coord_to_agent_id_map,
                        views_by_round, all_agent_ids_in_log, info, timestamp,
                        cmd_show_views, show_global_map_toggle, show_agent_views_toggle, show_messages_toggle,
                        messages_by_round, round_num_int, current_step_index, generate_content_only=True
                    )
                except Exception as e:
                    current_latex_output = f"% Error generating LaTeX snippet: {e}"
                    display_message = f"{Fore.RED} Error during LaTeX generation: {e}{Style.RESET_ALL}"
                print('\033[H\033[J', end='')
                print(f"{Style.BRIGHT} LaTeX Snippet (Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num}) {Style.RESET_ALL}")
                print(current_latex_output); print("-" * 60)
                if display_message: print(display_message)
                else: copy_hint = "| [C] Copy Snippet" if pyperclip else "(Install 'pyperclip' for [C])"; print(f"[Enter] Terminal View {copy_hint} | [Q] Quit")
                print("-" * 60)

            else: # Terminal Mode
                # (Terminal rendering logic unchanged)
                current_latex_output = ""
                grid_data = step.get('grid'); agents_list = step.get('agents'); score = step.get('score', 'N/A'); level = step.get('level', 'N/A'); coord_to_agent_id_map = {}; agent_ids_in_step = set()
                agents_list_valid = isinstance(agents_list, list)
                if agents_list_valid:
                    for agent in agents_list:
                        if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                            try: y = int(agent['y']); x = int(agent['x']); agent_id_str = str(agent['id']); coord_to_agent_id_map[(y, x)] = agent_id_str; agent_ids_in_step.add(agent_id_str)
                            except (ValueError, TypeError): continue

                model_name = info.get("model", "N/A"); num_agents_meta = info.get("num_agents", "N/A"); num_agents_actual = len(coord_to_agent_id_map)
                pause_indicator = f'{Style.BRIGHT + Fore.YELLOW}> PAUSED <{Style.RESET_ALL}' if paused else ""; round_info_line = (f'{pause_indicator}\n'
                                   f'Game: {timestamp} | Frame: {current_step_index+1}/{len(game_steps)} | Round: {round_num:<3} | Level: {level:<2} | '
                                   f'Score: {score:<4} | Agents: {num_agents_actual:<2} (Meta: {num_agents_meta}) | Model: {model_name}')
                print(round_info_line); current_lines_printed += round_info_line.count('\n') + 1
                if not agents_list_valid: print(f"{Fore.YELLOW}Warn: 'agents' list missing R{round_num}."); current_lines_printed += 1

                all_rendered_grids = [];
                if show_global_map_toggle and grid_data:
                    try: g_lines, g_width = render_terminal(grid_data, agent_message_colors_map, coord_to_agent_id_map); all_rendered_grids.append((" Global Map ", g_lines, g_width))
                    except Exception as e: all_rendered_grids.append((f"Global Error", [f"{Fore.RED}(Err global: {e})"], 30))
                elif show_global_map_toggle: all_rendered_grids.append((f"Global Error", ["(No grid data)"], 16))
                if cmd_show_views and show_agent_views_toggle:
                    agent_views_this_round = views_by_round.get(round_num_int, {})
                    if agent_views_this_round:
                        try:
                           def sort_key(agent_id): match = re.search(r'\d+$', agent_id); return int(match.group()) if match else agent_id
                           agents_with_views_in_step=sorted(list(agent_ids_in_step & agent_views_this_round.keys()), key=sort_key); other_agents_with_views=sorted([aid for aid in agent_views_this_round if aid not in agent_ids_in_step], key=sort_key); agents_to_render_views = agents_with_views_in_step + other_agents_with_views
                        except Exception: agents_to_render_views = sorted(list(agent_views_this_round.keys()))
                        for agent_id in agents_to_render_views:
                            view_grid = agent_views_this_round.get(agent_id)
                            if view_grid:
                                try: v_lines, v_width = render_terminal(view_grid, agent_message_colors_map, {}); agent_color = agent_message_colors_map.get(agent_id, Fore.WHITE); v_header = f"{agent_color}- View {agent_id} -{Style.RESET_ALL}"; all_rendered_grids.append((v_header, v_lines, v_width))
                                except Exception as e: all_rendered_grids.append((f"View {agent_id} Error", [f"{Fore.RED}(Err view {agent_id}: {e})"], 30))

                if all_rendered_grids:
                    num_grids_total = len(all_rendered_grids); effective_max_grids = max(1, max_grids_per_row_terminal); num_chunks = math.ceil(num_grids_total / effective_max_grids)
                    for chunk_index in range(num_chunks):
                        start_index = chunk_index * effective_max_grids; end_index = start_index + effective_max_grids; current_chunk_data = all_rendered_grids[start_index:end_index]
                        if not current_chunk_data: continue
                        if chunk_index > 0: print(""); current_lines_printed += 1
                        max_height_chunk = max((len(lines) for _, lines, _ in current_chunk_data if lines), default=0); header_line = ""
                        for header, _, width in current_chunk_data: header_line += pad_visual_width(header, width) + GRID_SEPARATOR
                        print(header_line.rstrip(GRID_SEPARATOR).rstrip()); current_lines_printed += 1
                        for i in range(max_height_chunk):
                            combined_line = ""
                            for idx, (_, lines, width) in enumerate(current_chunk_data): line_seg = lines[i] if i < len(lines) else ' ' * width; padded_segment = pad_visual_width(line_seg, width); combined_line += padded_segment + GRID_SEPARATOR
                            print(combined_line.rstrip(GRID_SEPARATOR).rstrip()); current_lines_printed += 1
                elif show_global_map_toggle or (cmd_show_views and show_agent_views_toggle): print("(No grids selected/found)"); current_lines_printed += 1

                if show_messages_toggle:
                    agent_messages_this_round = messages_by_round.get(round_num_int, {}); print_msg_header = agent_messages_this_round or not (show_global_map_toggle or (cmd_show_views and show_agent_views_toggle))
                    if print_msg_header: print("\n Agent Messages "); current_lines_printed += 2
                    if agent_messages_this_round:
                        try:
                           def sort_key(agent_id): match = re.search(r'\d+$', agent_id); return int(match.group()) if match else agent_id
                           agents_msg_in_step = sorted(list(agent_ids_in_step & agent_messages_this_round.keys()), key=sort_key); other_agents_with_msg = sorted([aid for aid in agent_messages_this_round if aid not in agent_ids_in_step], key=sort_key); agents_to_display_msg = agents_msg_in_step + other_agents_with_msg
                        except Exception: agents_to_display_msg = sorted(list(agent_messages_this_round.keys()))
                        msg_count = 0
                        for agent_id in agents_to_display_msg:
                            message = agent_messages_this_round.get(agent_id, ""); cleaned_message = ' '.join(str(message).split())
                            if not cleaned_message: continue
                            agent_color = agent_message_colors_map.get(agent_id, Fore.WHITE); print(f"{agent_color}{agent_id}:{Style.RESET_ALL} {cleaned_message}"); current_lines_printed += 1; msg_count += 1
                        if msg_count == 0: print("(No valid messages)"); current_lines_printed += 1
                    elif print_msg_header: print("(No messages this round)"); current_lines_printed +=1
                elif not (show_global_map_toggle or (cmd_show_views and show_agent_views_toggle)): print("(Messages hidden)"); current_lines_printed += 1

                print("-" * 30); g_stat = f"{Fore.GREEN}ON{Style.RESET_ALL}" if show_global_map_toggle else f"{Fore.RED}OFF{Style.RESET_ALL}"
                a_stat = f"{Fore.GREEN}ON{Style.RESET_ALL}" if show_agent_views_toggle else f"{Fore.RED}OFF{Style.RESET_ALL}"
                m_stat = f"{Fore.GREEN}ON{Style.RESET_ALL}" if show_messages_toggle else f"{Fore.RED}OFF{Style.RESET_ALL}"
                view_toggle_hint = f"[A]gViews:{a_stat}" if cmd_show_views else f"{Fore.LIGHTBLACK_EX}(Views disabld){Style.RESET_ALL}"
                print(f"[G]lobal:{g_stat} | {view_toggle_hint} | [M]sgs:{m_stat}"); print("[Space] Pause | [<-] Prev | [->] Next | [Enter] LaTeX | [Q] Quit"); current_lines_printed += 3
                if display_message: print(display_message); current_lines_printed += 1
                last_total_lines_printed = current_lines_printed

            needs_redraw = False
            sys.stdout.flush()

        #  Input Handling 
        try:
            key = readchar.readkey() # Read raw key
            key_lower = key.lower() # Use lowercase for most comparisons
            if debug_mode: # Optional debug print
                print(f"DEBUG: Key pressed: {key!r} | Lower: {key_lower!r}")
                sys.stdout.flush(); time.sleep(0.1) # Show debug info briefly
        except KeyboardInterrupt:
             key = '\x03'
             key_lower = '\x03'

        #  Process Input 
        action_taken = False # Flag to track if key was processed
        if key:
            original_index = current_step_index # Store index before potential change

            if key_lower == 'q' or key == '\x03':
                print("\nExiting."); break

            elif key == readchar.key.SPACE and not latex_mode:
                paused = not paused; action_taken = True
            #  Modified Arrow Key Handling 
            elif (key == readchar.key.RIGHT or key == ARROW_RIGHT) and not latex_mode:
                action_taken = True # We processed the key, even if index doesn't change
                if current_step_index < len(game_steps) - 1:
                    current_step_index += 1
                else:
                    copy_confirmation_msg = f"{Fore.YELLOW}Already at last frame!{Style.RESET_ALL}"
            elif (key == readchar.key.LEFT or key == ARROW_LEFT) and not latex_mode:
                action_taken = True # We processed the key
                if current_step_index > 0:
                    current_step_index -= 1
                else:
                    copy_confirmation_msg = f"{Fore.YELLOW}Already at first frame!{Style.RESET_ALL}"
            #  End Modified Arrow Key Handling 

            elif key_lower == readchar.key.ENTER: # Assuming ENTER is consistent
                latex_mode = not latex_mode; paused = True; current_latex_output = ""
                action_taken = True
            elif key_lower == 'c' and latex_mode:
                 action_taken = True
                 if pyperclip and current_latex_output:
                    try: pyperclip.copy(current_latex_output); copy_confirmation_msg = f"{Fore.GREEN}LaTeX snippet copied!{Style.RESET_ALL}"
                    except Exception as clip_err: copy_confirmation_msg = f"{Fore.RED}Copy error: {clip_err}{Style.RESET_ALL}"
                 elif not pyperclip: copy_confirmation_msg = f"{Fore.YELLOW}Copy requires 'pyperclip'.{Style.RESET_ALL}"
                 else: copy_confirmation_msg = f"{Fore.YELLOW}No snippet generated/visible.{Style.RESET_ALL}"
            elif key_lower == 'g':
                show_global_map_toggle = not show_global_map_toggle; action_taken = True
            elif key_lower == 'a' and cmd_show_views:
                show_agent_views_toggle = not show_agent_views_toggle; action_taken = True
            elif key_lower == 'm':
                show_messages_toggle = not show_messages_toggle; action_taken = True
            else:
                 # Key not recognized for a specific action
                 # We might still want to redraw if debug was enabled to clear the debug print
                 if debug_mode: needs_redraw = True
                 pass # Or print an "unknown key" message?

            #  Set redraw flag based on whether state changed 
            if action_taken or current_step_index != original_index or copy_confirmation_msg:
                 needs_redraw = True

    #  Cleanup 
    print("Replay finished.")