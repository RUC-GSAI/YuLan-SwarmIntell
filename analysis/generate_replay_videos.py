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
import subprocess # <--- ADD THIS LINE

# --- Third-party Imports ---
import numpy as np
from colorama import init, Fore, Back, Style, Cursor

# --- Try importing readchar ---
try:
    import readchar
    ARROW_UP    = "\x1b[A"
    ARROW_DOWN  = "\x1b[B"
    ARROW_RIGHT = "\x1b[C"
    ARROW_LEFT  = "\x1b[D"
except ImportError:
    # Keep readchar optional if only exporting
    readchar = None
    ARROW_UP, ARROW_DOWN, ARROW_RIGHT, ARROW_LEFT = "", "", "", ""
    print(f"{Fore.YELLOW}Warning: The 'readchar' library is required for interactive mode, but not for export.")
    print(f"To use interactive mode, install it using: pip install readchar{Style.RESET_ALL}")


# --- Try importing pyperclip for clipboard functionality ---
try:
    import pyperclip
except ImportError:
    pyperclip = None # Set to None if import fails
    # Do not print warning here, only if copy is attempted.

# --- Configuration ---
DEFAULT_LOG_DIR = 'experiment_01'
DEFAULT_TIME = 0.2
DEFAULT_MAX_GRIDS_PER_ROW = 6
GRID_SEPARATOR = "  |  "
LATEX_GRID_SEPARATOR = "~~|~~"
LATEX_AGENT_VIEWS_PER_ROW = 4
DEFAULT_ANIMATION_FPS = 10
DEFAULT_GIF_DPI = 150 # Lower DPI for GIFs for smaller file sizes
DEFAULT_VIDEO_DPI = 150 # DPI for PNGs used in video

# --- Colorama Initialization ---
init(autoreset=True)

# --- Agent Color Definitions ---
AGENT_MESSAGE_COLORS = [ Fore.LIGHTRED_EX, Fore.RED, Fore.LIGHTYELLOW_EX, Fore.YELLOW, Fore.LIGHTGREEN_EX, Fore.GREEN, Fore.LIGHTCYAN_EX, Fore.CYAN, Fore.LIGHTBLUE_EX, Fore.BLUE, Fore.LIGHTMAGENTA_EX, Fore.MAGENTA]
AGENT_MESSAGE_COLORS = list(dict.fromkeys(AGENT_MESSAGE_COLORS))
AGENT_GRID_COLORS_MAP = { Fore.LIGHTRED_EX: (Back.LIGHTRED_EX, Fore.WHITE), Fore.RED: (Back.RED, Fore.WHITE), Fore.LIGHTYELLOW_EX: (Back.LIGHTYELLOW_EX, Fore.BLACK), Fore.YELLOW: (Back.YELLOW, Fore.BLACK), Fore.LIGHTGREEN_EX: (Back.LIGHTGREEN_EX, Fore.BLACK), Fore.GREEN: (Back.GREEN, Fore.BLACK), Fore.LIGHTCYAN_EX: (Back.LIGHTCYAN_EX, Fore.BLACK), Fore.CYAN: (Back.CYAN, Fore.BLACK), Fore.LIGHTBLUE_EX: (Back.LIGHTBLUE_EX, Fore.WHITE), Fore.BLUE: (Back.BLUE, Fore.WHITE), Fore.LIGHTMAGENTA_EX: (Back.LIGHTMAGENTA_EX, Fore.BLACK), Fore.MAGENTA: (Back.MAGENTA, Fore.WHITE)}
for msg_color in AGENT_MESSAGE_COLORS:
    if msg_color not in AGENT_GRID_COLORS_MAP: print(f"{Fore.RED}Error: Color {msg_color} missing map! Exiting."); sys.exit(1)
DEFAULT_GRID_AGENT_COLOR = (Back.WHITE, Fore.BLACK)
GENERIC_AGENT_GRID_COLOR = Back.GREEN + Fore.BLACK

# --- ANSI Code Stripping Regex ---
ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def get_visual_width(text):
    return len(ansi_escape_pattern.sub('', text))

def pad_visual_width(text, width):
    current_visual_width = get_visual_width(text)
    padding = max(0, width - current_visual_width)
    return text + ' ' * padding

# --- Terminal Rendering Function (Unchanged) ---
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

# --- LaTeX Color Definitions ---
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
    return "\n".join(defs)

# --- LaTeX Frame Rendering (Unchanged from your provided code) ---
def render_latex_frame(step_data, agent_message_colors_map, coord_to_agent_id_map,
                      views_by_round, all_agent_ids_in_log, game_info, timestamp,
                      cmd_show_views, 
                      show_global_map_toggle, show_agent_views_toggle, show_messages_toggle, 
                      messages_by_round, round_num_int,
                      current_step_index, 
                      game_steps=None, 
                      generate_content_only=False):
    # (Code is identical to your provided version, ensure it's correct and complete)
    grid_data = step_data.get('grid'); agents_list = step_data.get('agents'); round_num = step_data['round']
    latex_string = ""
    if not generate_content_only:
        latex_string += "\\documentclass[landscape]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
        latex_string += "\\usepackage[table]{xcolor}\n\\usepackage{graphicx}\n\\usepackage{geometry}\n\\usepackage{float}\n"
        latex_string += "\\usepackage{enumitem}\n" 
        latex_string += "\\usepackage{amsmath}\n\\usepackage{amssymb}\n" 
        latex_string += "\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
        latex_string += "\\geometry{paperwidth=14in, paperheight=10in, margin=0.3cm, top=0.3cm, bottom=0.3cm, left=0.3cm, right=0.3cm}\n"
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
            grid_list = [[str(cell) if cell is not None else '.' for cell in row] for row in grid]
            grid_np = np.array(grid_list, dtype=object)
            if grid_np.ndim != 2: return f"\\texttt{{(Invalid dim {grid_np.ndim})}}\n", 1
            height, width_g = grid_np.shape
            if height == 0 or width_g == 0: return "\\texttt{(Empty HxW)}\n", 1
            width = width_g # Assign to outer scope var

            col_spec = "l|" + "c" * width + "|"; grid_latex += f"\\setlength{{\\tabcolsep}}{{3pt}}%"; grid_latex += f"\\renewcommand{{\\arraystretch}}{{1.0}}%\n"
            grid_latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"; header_row = "   & "+" & ".join(f"\\texttt{{{i%100:<2d}}}" for i in range(width))+" \\\\\n"
            grid_latex += header_row; grid_latex += "\\hline\n"

            for i in range(height):
                row_latex = f"\\texttt{{{i:<2d}}} & "; row_cells = []
                for j in range(width):
                    try:
                        cell = grid_np[i, j]
                        original_cell_str = str(cell).strip() if cell is not None and str(cell).strip() != '' else '.'
                    except IndexError:
                        original_cell_str = '?'; cell = '?'

                    latex_cell_cmd = "\\termcelldef" 
                    latex_params = ["{.}"]          
                    agent_id_at_coord = local_coord_to_agent_id_map.get((i,j))

                    if agent_id_at_coord and agent_id_at_coord in local_agent_message_colors_map:
                        match = re.search(r'\d+$', agent_id_at_coord)
                        agent_num_str = match.group()[-2:] if match else '?'
                        display_content = agent_num_str 
                        safe_display_content = str(display_content).replace('_','\\_').replace('{','\\{').replace('}','\\}')
                        padded_safe_content = "{:<2}".format(safe_display_content) 

                        message_color = local_agent_message_colors_map[agent_id_at_coord]
                        if original_cell_str == 'a': 
                            fg_color_name, _ = COLORAMA_TO_LATEX.get(message_color, ("termBlack", ""))
                            latex_cell_cmd = "\\termcellfg"
                            latex_params = [f"{{{fg_color_name}}}", f"{{{padded_safe_content}}}"]
                        else: 
                            grid_back_color, grid_text_color = AGENT_GRID_COLORS_MAP.get(message_color, DEFAULT_GRID_AGENT_COLOR)
                            bg_color_name, _ = COLORAMA_TO_LATEX.get(grid_back_color, ("termBgWhite", "")) 
                            fg_color_name, _ = COLORAMA_TO_LATEX.get(grid_text_color, ("termBlack", "")) 
                            latex_cell_cmd = "\\termcell"
                            latex_params = [f"{{{bg_color_name}}}", f"{{{fg_color_name}}}", f"{{{padded_safe_content}}}"]
                    else:
                        safe_display_content = str(original_cell_str).replace('_','\\_').replace('{','\\{').replace('}','\\}')
                        padded_safe_content = "{:<2}".format(safe_display_content)

                        if original_cell_str == 'W':
                            latex_cell_cmd = "\\termcell"
                            bg_color_name = "anthropicOrange" 
                            fg_color_name = "termBlack"       
                            padded_safe_content = "{:<2}".format('W') 
                            latex_params = [f"{{{bg_color_name}}}", f"{{{fg_color_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'P':
                            back_c, fore_c = (Back.WHITE, Fore.BLACK)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("white",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'Y':
                            back_c, fore_c = (Back.CYAN, Fore.BLACK)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("cyan",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'B':
                            back_c, fore_c = (Back.YELLOW, Fore.BLACK)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("yellow",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'X':
                            back_c, fore_c = (Back.MAGENTA, Fore.WHITE)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("magenta",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termWhite",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'A':
                            back_c, fore_c = GENERIC_AGENT_GRID_COLOR
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("blue",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termWhite",""))
                            latex_cell_cmd="\\termcell"
                            padded_safe_content = "{:<2}".format('A') 
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == '.':
                            latex_cell_cmd = "\\termcelldef" 
                            padded_safe_content = "{:<2}".format('.')
                            latex_params = [f"{{{padded_safe_content}}}"]
                        else:
                            latex_cell_cmd = "\\termcelldef"
                            latex_params = [f"{{{padded_safe_content}}}"]

                    cell_str = f"{latex_cell_cmd}{''.join(latex_params)}"
                    row_cells.append(cell_str)
                row_latex += " & ".join(row_cells) + " \\\\\n"; grid_latex += row_latex
            grid_latex += "\\hline\n\\end{tabular}\n"
            return grid_latex, width
        except Exception as e:
            escaped_error = str(e).replace('{','\\{').replace('}','\\}').replace('_','\\_')
            import traceback
            print(f"Error rendering grid to LaTeX: {e}")
            traceback.print_exc()
            return f"\\texttt{{(Err grid: {escaped_error})}}\n", 1

    latex_string += "\\begin{figure}[p]\n"
    latex_string += "\\centering\n" 
    latex_string += "\\begin{tabular}{@{}p{0.60\\textwidth}@{}p{0.38\\textwidth}@{}}\n"
    latex_string += "\\begin{minipage}[t]{0.60\\textwidth}\n" 
    latex_string += "\\resizebox{\\textwidth}{!}{%\n" 
    latex_string += "\\begin{minipage}{\\textwidth}\n\n"
    
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
             safe_msg_header="--- Agent Messages ---".replace('_','\\_'); message_block+=f"\\texttt{{{safe_msg_header}}} \\\\\n"
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

    latex_string += "\\vspace{1em}\n"
    latex_string += "\\noindent\\begin{minipage}[t][8cm][t]{\\textwidth} \n"
    latex_string += "\\fontsize{8pt}{9pt}\\selectfont \n"
    if message_block:
        latex_string += message_block
    else:
        latex_string += "\\texttt{(No messages)}\n"
    latex_string += "\\end{minipage}\n"
    latex_string += "\n\\end{minipage}%\n"; latex_string += "} \n"
    latex_string += "\\end{minipage}\n" 
    latex_string += "&\n" 

    latex_string += "\\begin{minipage}[t]{0.38\\textwidth}\n" 
    latex_string += "\\vspace{0.5cm}\n" 

    all_scores = []
    try:
        if game_steps:
            for i, game_step_data in enumerate(game_steps[:current_step_index + 1]):
                if isinstance(game_step_data, dict) and 'score' in game_step_data:
                    try:
                        score_str = str(game_step_data['score']).strip()
                        score_str = ''.join(c for c in score_str if c.isdigit() or c == '.')
                        score_val = float(score_str) if score_str else 0.0
                        all_scores.append((i, score_val))
                    except (ValueError, TypeError):
                        prev_score = all_scores[-1][1] if all_scores else 0.0
                        all_scores.append((i, prev_score))
                else:
                    prev_score = all_scores[-1][1] if all_scores else 0.0
                    all_scores.append((i, prev_score))
    except Exception as e:
        if not all_scores: all_scores = [(0, 0.0)]
        latex_string += "% Error processing scores: " + str(e).replace('%','\\%') + "\n"

    latex_string += "\\begin{tikzpicture}\n"
    latex_string += "\\begin{axis}[\n"
    latex_string += "    title={Score Progression},\n"
    latex_string += "    xlabel={Frame},\n"
    latex_string += "    ylabel={Score},\n"
    latex_string += "    title style={align=center},\n" 

    try:
        if all_scores:
            score_values = [s[1] for s in all_scores]
            min_score_val = min(score_values)
            max_score_val = max(score_values)
            if min_score_val == max_score_val:
                min_score_val = min_score_val - 0.5 if min_score_val != 0 else 0
                max_score_val = max_score_val + 0.5 if max_score_val !=0 else 1.0
            else:
                range_size = max_score_val - min_score_val
                min_score_val = max(0, min_score_val - 0.1 * range_size) 
                max_score_val = max_score_val + 0.1 * range_size
            latex_string += f"    ymin={min_score_val}, ymax={max_score_val},\n"
        else:
            latex_string += "    ymin=0, ymax=1,\n"
    except Exception:
        latex_string += "    ymin=0, ymax=1,\n"
        
    latex_string += "    grid=major,\n"
    latex_string += "    legend pos=north west,\n"
    latex_string += "    width=\\textwidth,\n"
    latex_string += "    height=7cm,\n" 
    latex_string += "    scaled ticks=false,\n"
    latex_string += "    tick label style={/pgf/number format/fixed},\n"
    latex_string += "    axis background/.style={fill=white, opacity=0.8},\n" 
    latex_string += "    title style={font=\\small},\n" 
    latex_string += "    label style={font=\\small},\n" 
    latex_string += "    tick label style={font=\\scriptsize, /pgf/number format/fixed},\n" 
    latex_string += "    no markers\n"
    latex_string += "]\n"

    latex_string += "\\addplot[\n"
    latex_string += "    color=blue,\n"
    latex_string += "    line width=1.0pt,\n"
    latex_string += "    mark=none\n"
    latex_string += "] coordinates {\n"
    for frame, score_val in all_scores:
        latex_string += f"    ({frame},{score_val})\n"
    latex_string += "};\n"

    try:
        if all_scores:
            current_score_val = all_scores[-1][1]
            latex_string += f"\\node[circle, fill=red, inner sep=2pt] at (axis cs:{current_step_index},{current_score_val}) {{}};\n"
    except Exception: pass

    latex_string += "\\end{axis}\n"
    latex_string += "\\end{tikzpicture}\n"
    latex_string += "\\vspace{0.5cm}\n"
    latex_string += "\\scriptsize{\\textbf{Legend:}}\n"
    latex_string += "\\begin{itemize}[nosep, leftmargin=*]\n"
    latex_string += "\\scriptsize\n"
    latex_string += "\\item[\\textcolor{blue}{---}] Score value\n" 
    latex_string += "\\item[\\textcolor{red}{$\\bullet$}] Current frame\n"
    latex_string += "\\end{itemize}\n"
    latex_string += "\\vspace{-0.5cm}\n" 
    latex_string += "\\end{minipage}\n" 
    latex_string += "\\end{tabular}\n" 
    latex_string += "\\end{figure}\n" 

    if not generate_content_only: 
        latex_string += "\n\\end{document}\n"
    return latex_string

# --- New function to batch export LaTeX frames ---
def batch_export_latex_frames(game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log, 
                             info, timestamp, cmd_show_views, output_dir, messages_by_round, prefix="frame"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Exporting {len(game_steps)} frames to {output_dir}...")
    for i, step in enumerate(game_steps):
        if not isinstance(step, dict) or 'round' not in step:
            print(f"{Fore.YELLOW}Skipping invalid step at index {i}{Style.RESET_ALL}")
            continue
        round_num_str = step['round']
        try: round_num_int = int(round_num_str)
        except (ValueError, TypeError):
            print(f"{Fore.YELLOW}Skipping step {i} with invalid round '{round_num_str}'{Style.RESET_ALL}")
            continue
        
        coord_to_agent_id_map = {}
        agents_list = step.get('agents', [])
        if isinstance(agents_list, list):
            for agent in agents_list:
                if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                    try:
                        y, x = int(agent['y']), int(agent['x'])
                        agent_id_str = str(agent['id'])
                        coord_to_agent_id_map[(y, x)] = agent_id_str
                    except (ValueError, TypeError): continue
        
        try:
            latex_content = render_latex_frame(
                step, agent_message_colors_map, coord_to_agent_id_map,
                views_by_round, all_agent_ids_in_log, info, timestamp,
                cmd_show_views, True, True, True, 
                messages_by_round, round_num_int, i,
                game_steps=game_steps, 
                generate_content_only=False
            )
            filename = f"{prefix}_{i:04d}.tex" # Start frame index from 0 for consistency with current_step_index
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            if (i+1) % 20 == 0 or i == len(game_steps)-1 : # Print progress periodically
                print(f"  Exported frame {i+1}/{len(game_steps)}: {filepath}")
        except Exception as e:
            print(f"{Fore.RED}Error exporting frame {i+1} (round {round_num_str}): {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    print(f"Export complete. Frames exported to {output_dir}")
    return True

def _ensure_pdflatex_ffmpeg_installed():
    """Checks for pdflatex and ffmpeg."""
    try:
        subprocess.run(["pdflatex", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Fore.RED}Error: 'pdflatex' command not found. Please install a LaTeX distribution (e.g., MiKTeX, TeX Live).{Style.RESET_ALL}")
        return False
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Fore.RED}Error: 'ffmpeg' command not found. Please install ffmpeg and ensure it's in your PATH.{Style.RESET_ALL}")
        return False
    return True

def _generate_images_from_latex(latex_dir, image_prefix="frame", dpi=150, force_recompile=False):
    """
    Compiles .tex files in latex_dir to .pdf, then converts .pdf to .png images.
    Stores intermediate PDFs in a 'pdf_gen' subdirectory and PNGs in 'image_gen'.
    Returns the path to the image directory or None on failure.
    """
    import subprocess
    import glob
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print(f"{Fore.RED}Error: The 'pdf2image' library is required for creating animations.")
        print(f"Please install it using: pip install pdf2image{Style.RESET_ALL}")
        print(f"You might also need to install poppler: https://pdf2image.readthedocs.io/en/latest/installation.html")
        return None

    pdf_dir = os.path.join(latex_dir, "pdf_gen")
    img_dir = os.path.join(latex_dir, "image_gen")

    if force_recompile or not os.path.exists(pdf_dir) or not os.path.exists(img_dir):
        if os.path.exists(pdf_dir) and force_recompile:
            import shutil
            shutil.rmtree(pdf_dir)
        if os.path.exists(img_dir) and force_recompile:
            import shutil
            shutil.rmtree(img_dir)

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        latex_files = sorted(glob.glob(os.path.join(latex_dir, f"{image_prefix}_*.tex")))
        if not latex_files:
            print(f"{Fore.YELLOW}No LaTeX files found with prefix '{image_prefix}_' in {latex_dir}. Trying '*.tex'.{Style.RESET_ALL}")
            latex_files = sorted(glob.glob(os.path.join(latex_dir, "*.tex")))
            if not latex_files:
                print(f"{Fore.RED}Error: No LaTeX files found in {latex_dir}. Cannot generate images.{Style.RESET_ALL}")
                return None
        
        print(f"Compiling {len(latex_files)} LaTeX files to PDF (DPI for images: {dpi})...")
        for i, latex_file in enumerate(latex_files):
            base_name = os.path.basename(latex_file).replace('.tex', '')
            pdf_file_path = os.path.join(pdf_dir, f"{base_name}.pdf")
            img_file_path = os.path.join(img_dir, f"{base_name}.png")

            # Skip if PNG already exists and not forcing recompile (though force_recompile handles parent dir)
            if os.path.exists(img_file_path) and not force_recompile:
                if (i + 1) % 50 == 0 or i == len(latex_files) -1 :
                    print(f"  Skipping already generated image {i+1}/{len(latex_files)}: {img_file_path}")
                continue
            
            if (i + 1) % 20 == 0 or i == 0 or i == len(latex_files) -1 :
                 print(f"  Processing LaTeX file {i+1}/{len(latex_files)}: {latex_file}")

            try:
                # Compile LaTeX to PDF
                if not os.path.exists(pdf_file_path) or force_recompile:
                    compile_process = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "-output-directory", pdf_dir, latex_file],
                        capture_output=True, text=True, errors='ignore', timeout=60 # ADDED errors='ignore' HERE
                    )
                    if compile_process.returncode != 0:
                        # The stdout/stderr might now be incomplete if bad bytes were ignored,
                        # but the error from pdflatex (if any) should still be indicated by returncode.
                        print(f"{Fore.RED}Error compiling {latex_file}:{Style.RESET_ALL}\nPDFLATEX STDOUT:\n{compile_process.stdout}\nPDFLATEX STDERR:\n{compile_process.stderr}")
                        # Try to continue with other files if one fails
                        continue
                
                # Convert PDF to PNG
                if os.path.exists(pdf_file_path): # Check if PDF was created
                    images = convert_from_path(pdf_file_path, dpi=dpi, first_page=1, last_page=1)
                    if images:
                        images[0].save(img_file_path, 'PNG')
                    else:
                        print(f"{Fore.YELLOW}Warning: No image generated from {pdf_file_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Warning: PDF file {pdf_file_path} not found after compilation attempt.{Style.RESET_ALL}")

            except subprocess.TimeoutExpired:
                print(f"{Fore.RED}Timeout compiling {latex_file}. Skipping.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error processing {latex_file} to image: {e}{Style.RESET_ALL}")
                # import traceback # For more detailed debugging if needed
                # traceback.print_exc()
    else:
        print(f"Found existing compiled PDFs in {pdf_dir} and images in {img_dir}. Using them. (Use --force-recompile to regenerate)")

    # Final check if images were produced
    if not glob.glob(os.path.join(img_dir, "*.png")):
        print(f"{Fore.RED}No PNG images found in {img_dir} after generation process.{Style.RESET_ALL}")
        return None
        
    return img_dir


def convert_images_to_video(img_dir, output_video, fps=10, image_prefix="frame"):
    import subprocess
    import glob
    print(f"Creating video from images in {img_dir}...")
    
    # Correctly glob for images based on prefix
    image_pattern_glob = os.path.join(img_dir, f"{image_prefix}_*.png")
    
    # Check if specific pattern yields files, otherwise fall back to generic
    if not glob.glob(image_pattern_glob):
        print(f"{Fore.YELLOW}No images found with pattern {image_pattern_glob}. Trying generic '*.png'.{Style.RESET_ALL}")
        image_pattern_glob = os.path.join(img_dir, "*.png")
        if not glob.glob(image_pattern_glob):
            print(f"{Fore.RED}Error: No PNG images found in {img_dir}. Cannot create video.{Style.RESET_ALL}")
            return False

    try:
        ffmpeg_command = [
            "ffmpeg", "-y", 
            "-framerate", str(fps),
            "-pattern_type", "glob", "-i", image_pattern_glob,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", 
            output_video
        ]
        print(f"Executing: {' '.join(ffmpeg_command)}")
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"Video created successfully: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Error creating video: {e}{Style.RESET_ALL}")
        print(f"FFmpeg stdout:\n{e.stdout}")
        print(f"FFmpeg stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"{Fore.RED}Error: ffmpeg not found. Please ensure it's installed and in your PATH.{Style.RESET_ALL}")
        return False

def convert_images_to_gif(img_dir, output_gif, fps=10, width=-1, height=-1, image_prefix="frame", dither_algo="sierra2_4a", gifsicle_optimize=True):
    import subprocess
    import glob
    import shutil # For checking gifsicle

    print(f"Creating GIF from images in {img_dir} (FPS: {fps}, Res: {width}x{height}, Dither: {dither_algo})...")

    image_pattern_glob = os.path.join(img_dir, f"{image_prefix}_*.png")
    
    if not glob.glob(image_pattern_glob):
        print(f"{Fore.YELLOW}No images found with pattern {image_pattern_glob}. Trying generic '*.png'.{Style.RESET_ALL}")
        image_pattern_glob = os.path.join(img_dir, "*.png")
        if not glob.glob(image_pattern_glob):
            print(f"{Fore.RED}Error: No PNG images found in {img_dir}. Cannot create GIF.{Style.RESET_ALL}")
            return False

    scale_params = []
    if width != -1:
        scale_params.append(f"width={width}")
    if height != -1:
        scale_params.append(f"height={height}")
    
    scale_filter_str = ""
    if scale_params:
        if len(scale_params) == 1:
            if 'width' in scale_params[0] and height == -1 :
                scale_params.append("height=-2") 
            elif 'height' in scale_params[0] and width == -1:
                scale_params.append("width=-2")
        scale_filter_str = f"scale={':'.join(scale_params)}:flags=lanczos,"
    
    # Add dither algorithm to paletteuse
    palette_use_options = f"dither={dither_algo}"

    if scale_filter_str:
        vf_filter = f"{scale_filter_str}split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse={palette_use_options}"
    else:
        vf_filter = f"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse={palette_use_options}"

    temp_gif_path = output_gif + ".tmp.gif" # Create a temporary GIF for ffmpeg output

    try:
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob", "-i", image_pattern_glob,
            "-vf", vf_filter,
            "-loop", "0", # Make GIF loop indefinitely
            temp_gif_path # Output to temp file first
        ]
        print(f"Executing FFmpeg: {' '.join(ffmpeg_command)}")
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        
        if gifsicle_optimize and shutil.which("gifsicle"):
            print(f"Optimizing with gifsicle...")
            # Common gifsicle optimization levels: -O1, -O2, -O3 (higher is more aggressive)
            # --lossy can significantly reduce size but affects quality
            # gifsicle_command = ["gifsicle", "-O3", temp_gif_path, "-o", output_gif]
            # For potentially very large files, try adding lossy compression. Adjust lossiness as needed.
            # Start with a moderate lossy value if needed.
            gifsicle_command = ["gifsicle", "-O3", "--colors", "256", temp_gif_path, "-o", output_gif]
            # To add lossy:
            # gifsicle_command = ["gifsicle", "-O3", "--lossy=80", "--colors", "256", temp_gif_path, "-o", output_gif]

            print(f"Executing Gifsicle: {' '.join(gifsicle_command)}")
            process_gifsicle = subprocess.run(gifsicle_command, capture_output=True, text=True, check=True)
            os.remove(temp_gif_path) # Remove temp file
            print(f"GIF created and optimized successfully: {output_gif}")
        else:
            if gifsicle_optimize and not shutil.which("gifsicle"):
                print(f"{Fore.YELLOW}Warning: gifsicle not found. Skipping gifsicle optimization. Install with 'sudo apt install gifsicle'.{Style.RESET_ALL}")
            os.rename(temp_gif_path, output_gif) # If not using gifsicle, rename temp to final
            print(f"GIF created successfully (no gifsicle optimization): {output_gif}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Error during GIF creation/optimization: {e}{Style.RESET_ALL}")
        if hasattr(e, 'cmd') and 'gifsicle' in e.cmd:
             print(f"Gifsicle stdout:\n{e.stdout}")
             print(f"Gifsicle stderr:\n{e.stderr}")
        else:
            print(f"FFmpeg stdout:\n{e.stdout}")
            print(f"FFmpeg stderr:\n{e.stderr}")
        if os.path.exists(temp_gif_path):
            os.remove(temp_gif_path)
        return False
    except FileNotFoundError:
        print(f"{Fore.RED}Error: ffmpeg or gifsicle not found. Please ensure they are installed and in your PATH.{Style.RESET_ALL}")
        if os.path.exists(temp_gif_path):
            os.remove(temp_gif_path)
        return False
    finally:
        if os.path.exists(temp_gif_path) and not os.path.exists(output_gif): # Cleanup if error before final rename/remove
            os.remove(temp_gif_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay game logs with interactive controls and LaTeX/Animation export.')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help=f'Log directory (default: {DEFAULT_LOG_DIR})')
    parser.add_argument('--time', type=float, default=DEFAULT_TIME, help=f'Time delay (inactive)')
    parser.add_argument('--show-views', '-v', action='store_true', help='Allow display/toggle of individual agent views.')
    parser.add_argument('--max-grids', type=int, default=DEFAULT_MAX_GRIDS_PER_ROW, help=f'Max AGENT grids per horizontal row in terminal (default: {DEFAULT_MAX_GRIDS_PER_ROW}).')
    parser.add_argument('--debug', action='store_true', help='Print debug information.')
    
    # Export options
    parser.add_argument('--export-all-latex', action='store_true', help='Export all frames as LaTeX files to --output-dir.')
    parser.add_argument('--output-dir', type=str, default='latex_frames_export', help='Directory to save exported LaTeX files and generated animations.')
    parser.add_argument('--force-recompile', action='store_true', help='Force re-compilation of LaTeX to PDF and PDF to PNG, even if intermediate files exist.')
    
    # Video options
    parser.add_argument('--create-video', action='store_true', help='Create video from exported LaTeX frames. Requires --export-all-latex implicitly.')
    parser.add_argument('--video-file', type=str, default='output_video.mp4', help='Output video file name (saved in --output-dir).')
    parser.add_argument('--video-dpi', type=int, default=DEFAULT_VIDEO_DPI, help=f'DPI for PNGs used for video (default: {DEFAULT_VIDEO_DPI}).')
    
    # GIF options
    parser.add_argument('--create-gif', action='store_true', help='Create GIF from exported LaTeX frames. Requires --export-all-latex implicitly.')
    parser.add_argument('--gif-file', type=str, default='output_animation.gif', help='Output GIF file name (saved in --output-dir).')
    parser.add_argument('--gif-dpi', type=int, default=DEFAULT_GIF_DPI, help=f'DPI for PNGs used for GIF (default: {DEFAULT_GIF_DPI}).')
    parser.add_argument('--gif-width', type=int, default=-1, help='Width for output GIF in pixels (-1 for original/auto based on height).')
    parser.add_argument('--gif-height', type=int, default=-1, help='Height for output GIF in pixels (-1 for original/auto based on width).')
    
    # Common animation options
    parser.add_argument('--fps', type=int, default=DEFAULT_ANIMATION_FPS, help=f'Frames per second for output video/GIF (default: {DEFAULT_ANIMATION_FPS}).')
    
    parser.add_argument('--gif-dither', type=str, default='sierra2_4a', 
                        help='Dithering algorithm for GIF (e.g., sierra2_4a, bayer, none). See ffmpeg paletteuse docs.')
    parser.add_argument('--no-gifsicle', action='store_true',
                        help='Disable optimization attempt with gifsicle even if it is installed.')

    parser.add_argument('--model-name', type=str, default=None,
                        help='Specify the model name to filter game logs for export/animation. Processes the first match found in meta_log.json.')
    # --- END OF ADDITION ---

    args = parser.parse_args()

    # If creating video or GIF, ensure LaTeX export is also triggered
    if args.create_video or args.create_gif:
        args.export_all_latex = True 
        if not _ensure_pdflatex_ffmpeg_installed():
            sys.exit(1)


    log_dir = args.log_dir; TIME = args.time; cmd_show_views = args.show_views
    max_grids_per_row_terminal = args.max_grids; debug_mode = args.debug
    # --- Load Metadata, Game Steps & Agent Data ---
    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path):
        print(f"{Fore.RED}Meta log not found: {meta_log_path}{Style.RESET_ALL}")
        sys.exit(1)
    try:
        # Ensure UTF-8 encoding when reading JSON
        with open(meta_log_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error reading meta log {meta_log_path}: {e}{Style.RESET_ALL}")
        sys.exit(1)

    if not meta: # Check if meta log is empty or invalid
        print(f"{Fore.RED}Meta log is empty or invalid: {meta_log_path}{Style.RESET_ALL}")
        sys.exit(1)

    timestamp = None
    info = None

    if args.model_name:
        print(f"{Style.BRIGHT}Attempting to find game log for model: {args.model_name}{Style.RESET_ALL}")
        found_match = False
        # Iterate through meta log entries. Python 3.7+ dicts preserve insertion order.
        # For older versions, the "first" match might not be chronologically first.
        for ts_key, game_info_val in meta.items():
            if isinstance(game_info_val, dict) and game_info_val.get('model') == args.model_name:
                timestamp = ts_key
                info = game_info_val
                found_match = True
                print(f"{Fore.GREEN}Found matching game log for model '{args.model_name}' with timestamp: {timestamp}{Style.RESET_ALL}")
                break # Process the first match found
        if not found_match:
            print(f"{Fore.RED}Error: No game log found for model name '{args.model_name}' in {meta_log_path}{Style.RESET_ALL}")
            print(f"Available models/entries in meta_log (first few shown if many):")
            count = 0
            for ts_example, info_example in meta.items():
                if isinstance(info_example, dict):
                    print(f"  Timestamp: {ts_example}, Model: {info_example.get('model', 'N/A')}")
                else:
                    print(f"  Timestamp: {ts_example}, Invalid entry format.")
                count += 1
                if count >= 5: # Print a few examples
                    if len(meta) > 5: print("  ...")
                    break
            sys.exit(1)
    else:
        # Original behavior: process the first entry if no model_name is specified
        try:
            # Get the first key from the meta dictionary
            timestamp = next(iter(meta.keys()))
            info = meta[timestamp]
            print(f"{Style.BRIGHT}No model name specified. Processing the first game log found in meta_log.{Style.RESET_ALL}")
            print(f"  Timestamp: {timestamp}, Model: {info.get('model', 'N/A') if isinstance(info, dict) else 'N/A (invalid info format)'}")
        except StopIteration: # Handles empty meta dict
            print(f"{Fore.RED}No games found in meta log: {meta_log_path}{Style.RESET_ALL}")
            sys.exit(1)
        except KeyError: # Should not happen if next(iter()) worked, but good for safety
            print(f"{Fore.RED}Error accessing first game entry in meta log.{Style.RESET_ALL}")
            sys.exit(1)

    if not timestamp or not info: # Should be caught by earlier checks, but as a final safeguard
        print(f"{Fore.RED}Fatal: Could not determine a game log to process. Timestamp or info is missing.{Style.RESET_ALL}")
        sys.exit(1)

    # The rest of the script will now use the selected 'timestamp' and 'info'
    print(f"{Style.BRIGHT}Selected game for processing: Timestamp {timestamp}, Model {info.get('model', 'N/A')}{Style.RESET_ALL}")
    game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')
    agent_log_path = os.path.join(log_dir, f'agent_log_{timestamp}.json')

    game_steps = []; messages_by_round = defaultdict(dict); views_by_round = defaultdict(dict); all_agent_ids_in_log = set()
    try:
        if os.path.exists(game_log_path):
             with open(game_log_path, encoding='utf-8') as f: game_steps = json.load(f)
        else: print(f"{Fore.RED}Game log missing: {game_log_path}"); sys.exit(1)
        if not game_steps: print(f"{Fore.RED}Game log empty: {game_log_path}"); sys.exit(1)
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
        else: print(f"{Fore.YELLOW}Warn: Agent log missing: {agent_log_path}")
    except Exception as e: print(f"{Fore.RED}Fatal Err loading logs: {e}"); sys.exit(1)

    def sort_key_fn(agent_id_str): match = re.search(r'\d+$', agent_id_str); return int(match.group()) if match else agent_id_str
    try: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log), key=sort_key_fn)
    except Exception: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log))
    agent_message_colors_map = {}
    num_unique_colors = len(AGENT_MESSAGE_COLORS)
    if num_unique_colors > 0:
        for i, agent_id in enumerate(sorted_agent_ids_list): agent_message_colors_map[agent_id] = AGENT_MESSAGE_COLORS[i % num_unique_colors]
    
    # --- Export All LaTeX Frames if requested ---
    if args.export_all_latex:
        print(f"{Style.BRIGHT}--- Starting Batch Export ---{Style.RESET_ALL}")
        print(f"Game: {timestamp}")
        print(f"Output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        export_success = batch_export_latex_frames(
            game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log,
            info, timestamp, cmd_show_views, args.output_dir, messages_by_round, prefix="frame"
        )
        
        # Path to directory containing generated PNGs.
        # This will be set by the first process (video or gif) that generates images.
        generated_image_dir = None
        generated_image_dpi = -1 # DPI of the images in generated_image_dir

        if export_success and args.create_video:
            
            if generated_image_dir and generated_image_dpi == args.gif_dpi : 
                gif_file_path = os.path.join(args.output_dir, args.gif_file)
                gif_success = convert_images_to_gif(
                    generated_image_dir, gif_file_path, args.fps, 
                    args.gif_width, args.gif_height, image_prefix="frame",
                    dither_algo=args.gif_dither, # Pass dither argument
                    gifsicle_optimize=not args.no_gifsicle # Pass gifsicle flag
                )
            
            print(f"{Style.BRIGHT}--- Creating Video ---{Style.RESET_ALL}")
            # Generate images if not already generated or if DPI differs or force_recompile
            if not generated_image_dir or generated_image_dpi != args.video_dpi or args.force_recompile:
                print(f"Generating images for video (DPI: {args.video_dpi})...")
                generated_image_dir = _generate_images_from_latex(
                    args.output_dir, 
                    image_prefix="frame", 
                    dpi=args.video_dpi, 
                    force_recompile=args.force_recompile
                )
                if generated_image_dir:
                    generated_image_dpi = args.video_dpi
            else:
                print(f"Reusing existing images from {generated_image_dir} (DPI: {generated_image_dpi}).")

            if generated_image_dir:
                video_file_path = os.path.join(args.output_dir, args.video_file)
                video_success = convert_images_to_video(generated_image_dir, video_file_path, args.fps, image_prefix="frame")
                if video_success:
                    print(f"{Fore.GREEN}Video creation complete: {video_file_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to create video.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to generate/find images for video.{Style.RESET_ALL}")

        if export_success and args.create_gif:
            print(f"{Style.BRIGHT}--- Creating GIF ---{Style.RESET_ALL}")
            # Generate images if not already generated or if DPI differs or force_recompile
            # Note: If force_recompile is true, this will always run _generate_images_from_latex
            #       even if DPIs match. This is intended by force_recompile.
            if not generated_image_dir or generated_image_dpi != args.gif_dpi or args.force_recompile:
                print(f"Generating images for GIF (DPI: {args.gif_dpi})...")
                # If force_recompile is true, we want to recompile, even if generated_image_dir exists.
                # The _generate_images_from_latex function itself handles the force_recompile flag for its internal directories.
                current_force_recompile_for_gif_images = args.force_recompile
                if generated_image_dir and generated_image_dpi == args.gif_dpi and not args.force_recompile:
                    # This case means images exist at the correct DPI, and we are not forcing recompile, so skip generation.
                    print(f"Reusing existing images from {generated_image_dir} (DPI: {generated_image_dpi}) for GIF.")
                else:
                    # This will run if:
                    # 1. No images generated yet (generated_image_dir is None)
                    # 2. Images exist, but DPI is different
                    # 3. args.force_recompile is True (overrides everything)
                    generated_image_dir_for_gif = _generate_images_from_latex( # Use a temp var here
                        args.output_dir, 
                        image_prefix="frame", 
                        dpi=args.gif_dpi, 
                        force_recompile=current_force_recompile_for_gif_images # Pass the correct force flag
                    )
                    if generated_image_dir_for_gif:
                        generated_image_dir = generated_image_dir_for_gif # Update the main variable
                        generated_image_dpi = args.gif_dpi
                    # If _generate_images_from_latex fails, generated_image_dir will remain as it was or None.
            else:
                 print(f"Reusing existing images from {generated_image_dir} (DPI: {generated_image_dpi}) for GIF.")


            if generated_image_dir and generated_image_dpi == args.gif_dpi : # Ensure images are for GIF DPI
                gif_file_path = os.path.join(args.output_dir, args.gif_file)
                gif_success = convert_images_to_gif(
                    generated_image_dir, gif_file_path, args.fps, 
                    args.gif_width, args.gif_height, image_prefix="frame"
                )
                if gif_success:
                    print(f"{Fore.GREEN}GIF creation complete: {gif_file_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to create GIF.{Style.RESET_ALL}")
            elif not generated_image_dir:
                 print(f"{Fore.RED}Failed to generate/find images for GIF.{Style.RESET_ALL}")
            elif generated_image_dpi != args.gif_dpi:
                 print(f"{Fore.RED}Images found but at wrong DPI ({generated_image_dpi}) for GIF (expected {args.gif_dpi}). This shouldn't happen with the new logic.{Style.RESET_ALL}")

        
        print(f"{Style.BRIGHT}--- Batch Export Finished ---{Style.RESET_ALL}")
        sys.exit(0) # Exit after export operations
    # --- Interactive Replay Loop ---
    if not readchar:
        print(f"{Fore.RED}Cannot start interactive mode: 'readchar' library not found or failed to import.{Style.RESET_ALL}")
        print("If you only intended to export, the process might have completed above.")
        sys.exit(1)

    current_step_index = 0; paused = True; latex_mode = False
    last_total_lines_printed = 0; needs_redraw = True; current_latex_output = ""
    copy_confirmation_msg = ""
    show_global_map_toggle = True; show_agent_views_toggle = True; show_messages_toggle = True

    print(f"{Style.BRIGHT}Starting Interactive Replay for game: {timestamp}{Style.RESET_ALL}")
    print(f"Model: {info.get('model', 'N/A')}, Agents in meta: {info.get('num_agents', 'N/A')}")
    print("Press 'Q' to quit, [Space] to Play/Pause, [<-][->] to navigate frames.")
    time.sleep(1) # Give user time to read initial messages

    while True:
        if current_step_index < 0: current_step_index = 0
        if current_step_index >= len(game_steps): current_step_index = len(game_steps) - 1
        step = game_steps[current_step_index]
        if not isinstance(step, dict) or 'round' not in step:
            copy_confirmation_msg = f"{Fore.YELLOW}Invalid step data at index {current_step_index}. Skipping.{Style.RESET_ALL}"
            needs_redraw = True
            if current_step_index < len(game_steps) - 1: current_step_index += 1
            elif current_step_index > 0: current_step_index -= 1
            else: print(f"{Fore.RED}Cannot proceed from invalid step 0.{Style.RESET_ALL}"); break
            continue
        round_num_str = step['round']
        try: round_num_int = int(round_num_str)
        except (ValueError, TypeError):
            copy_confirmation_msg = f"{Fore.YELLOW}Invalid round '{round_num_str}' at step {current_step_index}. Skipping.{Style.RESET_ALL}"
            needs_redraw = True
            if current_step_index < len(game_steps) - 1: current_step_index += 1
            elif current_step_index > 0: current_step_index -= 1
            else: print(f"{Fore.RED}Cannot proceed from step 0 with invalid round.{Style.RESET_ALL}"); break
            continue

        if needs_redraw:
            print('\033[H\033[J', end='') 
            current_lines_printed = 0
            display_message = copy_confirmation_msg; copy_confirmation_msg = "" 

            if latex_mode:
                print(f"{Style.BRIGHT}--- LaTeX Mode --- (Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num_str}){Style.RESET_ALL}")
                print("Generating LaTeX snippet...")
                sys.stdout.flush()
                grid_data = step.get('grid'); agents_list = step.get('agents'); coord_to_agent_id_map = {}
                if isinstance(agents_list, list):
                    for agent in agents_list:
                        if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                            try: y,x = int(agent['y']),int(agent['x']); agent_id_str = str(agent['id']); coord_to_agent_id_map[(y, x)] = agent_id_str
                            except (ValueError, TypeError): continue
                try:
                    current_latex_output = render_latex_frame(
                        step, agent_message_colors_map, coord_to_agent_id_map,
                        views_by_round, all_agent_ids_in_log, info, timestamp,
                        cmd_show_views, show_global_map_toggle, show_agent_views_toggle, show_messages_toggle,
                        messages_by_round, round_num_int, current_step_index, 
                        game_steps=game_steps, generate_content_only=True
                    )
                except Exception as e:
                    current_latex_output = f"% Error generating LaTeX snippet: {e}"
                    display_message = f"{Fore.RED} Error during LaTeX generation: {e}{Style.RESET_ALL}"
                print('\033[H\033[J', end='')
                print(f"{Style.BRIGHT}--- LaTeX Snippet (Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num_str}) ---{Style.RESET_ALL}")
                print(current_latex_output); print("-" * 60)
                if display_message: print(display_message)
                copy_hint = "| [C] Copy Snippet" if pyperclip else "| (Install 'pyperclip' for [C])"
                print(f"[Enter] Terminal View {copy_hint} | [Q] Quit")
                print("-" * 60)

            else: # Terminal Mode
                current_latex_output = ""
                grid_data = step.get('grid'); agents_list = step.get('agents'); score = step.get('score', 'N/A'); level = step.get('level', 'N/A'); coord_to_agent_id_map = {}; agent_ids_in_step = set()
                agents_list_valid = isinstance(agents_list, list)
                if agents_list_valid:
                    for agent in agents_list:
                        if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                            try: y,x = int(agent['y']),int(agent['x']); agent_id_str = str(agent['id']); coord_to_agent_id_map[(y, x)] = agent_id_str; agent_ids_in_step.add(agent_id_str)
                            except (ValueError, TypeError): continue

                model_name = info.get("model", "N/A"); num_agents_meta = info.get("num_agents", "N/A"); num_agents_actual = len(coord_to_agent_id_map)
                pause_indicator = f'{Style.BRIGHT + Fore.YELLOW}> PAUSED <{Style.RESET_ALL}' if paused else ""; round_info_line = (f'{pause_indicator}\n'
                                   f'Game: {timestamp} | Frame: {current_step_index+1}/{len(game_steps)} | Round: {round_num_str:<3} | Level: {level:<2} | '
                                   f'Score: {score:<4} | Agents: {num_agents_actual:<2} (Meta: {num_agents_meta}) | Model: {model_name}')
                print(round_info_line); current_lines_printed += round_info_line.count('\n') + 1
                if not agents_list_valid: print(f"{Fore.YELLOW}Warn: 'agents' list missing R{round_num_str}."); current_lines_printed += 1

                all_rendered_grids = [];
                if show_global_map_toggle and grid_data:
                    try: g_lines, g_width = render_terminal(grid_data, agent_message_colors_map, coord_to_agent_id_map); all_rendered_grids.append(("--- Global Map ---", g_lines, g_width))
                    except Exception as e: all_rendered_grids.append((f"Global Error", [f"{Fore.RED}(Err global: {e})"], 30))
                elif show_global_map_toggle: all_rendered_grids.append((f"Global Error", ["(No grid data)"], 16))
                if cmd_show_views and show_agent_views_toggle:
                    agent_views_this_round = views_by_round.get(round_num_int, {})
                    if agent_views_this_round:
                        try:
                           agents_with_views_in_step=sorted(list(agent_ids_in_step & agent_views_this_round.keys()), key=sort_key_fn); other_agents_with_views=sorted([aid for aid in agent_views_this_round if aid not in agent_ids_in_step], key=sort_key_fn); agents_to_render_views = agents_with_views_in_step + other_agents_with_views
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
                        max_height_chunk = max((len(lines) for _, lines, _ in current_chunk_data if lines), default=0); header_line_str = ""
                        for header, _, width in current_chunk_data: header_line_str += pad_visual_width(header, width) + GRID_SEPARATOR
                        print(header_line_str.rstrip(GRID_SEPARATOR).rstrip()); current_lines_printed += 1
                        for i in range(max_height_chunk):
                            combined_line = ""
                            for idx, (_, lines, width) in enumerate(current_chunk_data): line_seg = lines[i] if i < len(lines) else ' ' * width; padded_segment = pad_visual_width(line_seg, width); combined_line += padded_segment + GRID_SEPARATOR
                            print(combined_line.rstrip(GRID_SEPARATOR).rstrip()); current_lines_printed += 1
                elif show_global_map_toggle or (cmd_show_views and show_agent_views_toggle): print("(No grids selected/found)"); current_lines_printed += 1

                if show_messages_toggle:
                    agent_messages_this_round = messages_by_round.get(round_num_int, {}); print_msg_header = agent_messages_this_round or not (show_global_map_toggle or (cmd_show_views and show_agent_views_toggle))
                    if print_msg_header: print("\n--- Agent Messages ---"); current_lines_printed += 2
                    if agent_messages_this_round:
                        try:
                           agents_msg_in_step = sorted(list(agent_ids_in_step & agent_messages_this_round.keys()), key=sort_key_fn); other_agents_with_msg = sorted([aid for aid in agent_messages_this_round if aid not in agent_ids_in_step], key=sort_key_fn); agents_to_display_msg = agents_msg_in_step + other_agents_with_msg
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
            needs_redraw = False; sys.stdout.flush()

        try:
            if not paused and not latex_mode: # Auto-advance if not paused and in terminal mode
                time.sleep(0.1) # Small delay for auto-advance to be visible, adjust as needed
                key_char_to_process = readchar.readchar() if platform.system() != "Windows" else readchar.readkey() # Check for key press during auto-advance
                if key_char_to_process: # If a key was pressed, process it
                     key = key_char_to_process
                else: # No key pressed, simulate next frame
                    if current_step_index < len(game_steps) -1:
                        current_step_index += 1
                        needs_redraw = True
                    else: # At last frame, pause
                        paused = True
                        copy_confirmation_msg = f"{Fore.YELLOW}End of log. Paused.{Style.RESET_ALL}"
                        needs_redraw = True
                    continue # Skip normal input reading for this iteration
            else: # Paused or in LaTeX mode, wait for key
                key = readchar.readkey()
        except KeyboardInterrupt: key = '\x03' # Ctrl+C

        key_lower = key.lower()
        if debug_mode: print(f"DEBUG: Key pressed: {key!r} | Lower: {key_lower!r}"); sys.stdout.flush(); time.sleep(0.1) 
        
        action_taken = False; original_index = current_step_index
        if key_lower == 'q' or key == '\x03': print("\nExiting interactive replay."); break
        elif key == readchar.key.SPACE and not latex_mode: paused = not paused; action_taken = True
        elif (key == readchar.key.RIGHT or key == ARROW_RIGHT) and not latex_mode:
            action_taken = True
            if current_step_index < len(game_steps) - 1: current_step_index += 1
            else: copy_confirmation_msg = f"{Fore.YELLOW}Already at last frame!{Style.RESET_ALL}"
        elif (key == readchar.key.LEFT or key == ARROW_LEFT) and not latex_mode:
            action_taken = True
            if current_step_index > 0: current_step_index -= 1
            else: copy_confirmation_msg = f"{Fore.YELLOW}Already at first frame!{Style.RESET_ALL}"
        elif key_lower == readchar.key.ENTER: 
            latex_mode = not latex_mode; paused = True; current_latex_output = ""
            action_taken = True
        elif key_lower == 'c' and latex_mode:
             action_taken = True
             if pyperclip and current_latex_output:
                try: pyperclip.copy(current_latex_output); copy_confirmation_msg = f"{Fore.GREEN}LaTeX snippet copied!{Style.RESET_ALL}"
                except Exception as clip_err: copy_confirmation_msg = f"{Fore.RED}Copy error: {clip_err}{Style.RESET_ALL}"
             elif not pyperclip: copy_confirmation_msg = f"{Fore.YELLOW}Copy requires 'pyperclip' (pip install pyperclip).{Style.RESET_ALL}"
             else: copy_confirmation_msg = f"{Fore.YELLOW}No snippet generated/visible.{Style.RESET_ALL}"
        elif key_lower == 'g': show_global_map_toggle = not show_global_map_toggle; action_taken = True
        elif key_lower == 'a' and cmd_show_views: show_agent_views_toggle = not show_agent_views_toggle; action_taken = True
        elif key_lower == 'm': show_messages_toggle = not show_messages_toggle; action_taken = True
        else:
             if debug_mode: needs_redraw = True # Redraw to clear debug line
        if action_taken or current_step_index != original_index or copy_confirmation_msg: needs_redraw = True

    print("Replay finished.")
