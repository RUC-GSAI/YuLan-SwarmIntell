
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
global metrics_cache
global embedding_cache
metrics_cache = {}
try:
    import readchar
    ARROW_UP    = "\x1b[A"
    ARROW_DOWN  = "\x1b[B"
    ARROW_RIGHT = "\x1b[C"
    ARROW_LEFT  = "\x1b[D"
except ImportError:
    readchar = None
    ARROW_UP, ARROW_DOWN, ARROW_RIGHT, ARROW_LEFT = "", "", "", ""
    print(f"{Fore.YELLOW}Warning: The 'readchar' library is required for interactive mode, but not for export.")
    print(f"To use interactive mode, install it using: pip install readchar{Style.RESET_ALL}")


#  Try importing pyperclip for clipboard functionality 
try:
    import pyperclip
except ImportError:
    pyperclip = None # Set to None if import fails
    # Do not print warning here, only if copy is attempted.

#  Try importing libraries for metrics calculation 
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


from utils.constants import *
from utils.helper import *
from utils.latex import *


#  Colorama Initialization 
init(autoreset=True)




os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
os.makedirs(METRICS_CACHE_DIR, exist_ok=True)

#  Embedding caching system 
class EmbeddingCache:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.cache = {}
        self.cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{model_name.replace('/', '_')}_cache.pkl")
        self.load_cache()
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving embedding cache: {e}")
    
    def get_model(self):
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                return None
        return self.model
    
    def get_embedding(self, text):
        """Get embedding for text, using cache if available"""
        if not text:
            return None
            
        # Create a hash of the text to use as key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        model = self.get_model()
        if model is None:
            return None
            
        try:
            embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
            self.cache[text_hash] = embedding
            # Save periodically (every 100 new embeddings)
            if len(self.cache) % 100 == 0:
                self.save_cache()
            return embedding
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def get_embeddings(self, texts):
        """Get embeddings for multiple texts, using cache where available"""
        if not texts:
            return []
            
        # Check which texts need to be computed
        uncached_texts = []
        uncached_indices = []
        text_hashes = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            text_hashes.append(text_hash)
            
            if text_hash not in self.cache:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts
        embeddings = [None] * len(texts)
        if uncached_texts:
            model = self.get_model()
            if model is not None:
                try:
                    new_embeddings = model.encode(uncached_texts, show_progress_bar=False, convert_to_numpy=True)
                    
                    # Update cache with new embeddings
                    for i, idx in enumerate(uncached_indices):
                        text_hash = text_hashes[idx]
                        self.cache[text_hash] = new_embeddings[i]
                        embeddings[idx] = new_embeddings[i]
                        
                    # Save cache after batch update
                    self.save_cache()
                except Exception as e:
                    print(f"Error encoding texts: {e}")
        
        # Fill in cached embeddings
        for i, text_hash in enumerate(text_hashes):
            if i not in uncached_indices:
                embeddings[i] = self.cache[text_hash]
                
        return embeddings

# Global embedding cache instance
embedding_cache = None
metrics_cache = {}


def render_latex_frame(step_data, agent_message_colors_map, coord_to_agent_id_map,
                      views_by_round, all_agent_ids_in_log, game_info, timestamp,
                      cmd_show_views, 
                      show_global_map_toggle, show_agent_views_toggle, show_messages_toggle, 
                      messages_by_round, round_num_int,
                      current_step_index, 
                      game_steps=None, agent_log=None,
                      generate_content_only=False, embedding_model=None):
    
    # Initialize embedding cache if needed
    global embedding_cache
    if embedding_cache is None and METRICS_LIBRARIES_AVAILABLE and embedding_model is not None:
        embedding_cache = EmbeddingCache(model_name=DEFAULT_EMBEDDING_MODEL)
        
    # Calculate metrics for this frame
    current_metrics = {}
    if METRICS_LIBRARIES_AVAILABLE and embedding_cache is not None:
        current_metrics = calculate_metrics_for_frame(
            step_data, messages_by_round, game_steps, current_step_index, 
            agent_log=agent_log, embedding_model=embedding_model
        )
    
    grid_data = step_data.get('grid'); agents_list = step_data.get('agents'); round_num = step_data['round']
    latex_string = ""
    if not generate_content_only:
        latex_string += "\\documentclass[landscape]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
        latex_string += "\\usepackage[table]{xcolor}\n\\usepackage{graphicx}\n\\usepackage{geometry}\n\\usepackage{float}\n"
        latex_string += "\\usepackage{enumitem}\n" 
        latex_string += "\\usepackage{amsmath}\n\\usepackage{amssymb}\n" 
        latex_string += "\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
        latex_string += "\\geometry{paperwidth=14in, paperheight=10in, margin=0.5in}\n"
        latex_string += "\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{0pt}\n"
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
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("termBgWhite",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'Y':
                            back_c, fore_c = (Back.CYAN, Fore.BLACK)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("termBgCyan",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'B':
                            back_c, fore_c = (Back.YELLOW, Fore.BLACK)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("termBgYellow",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'X':
                            back_c, fore_c = (Back.MAGENTA, Fore.WHITE)
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("termBgMagenta",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termWhite",""))
                            latex_cell_cmd="\\termcell"
                            latex_params=[f"{{{bg_name}}}", f"{{{fg_name}}}", f"{{{padded_safe_content}}}"]
                        elif original_cell_str == 'A':
                            back_c, fore_c = GENERIC_AGENT_GRID_COLOR
                            bg_name,_=COLORAMA_TO_LATEX.get(back_c,("termBgGreen",""))
                            fg_name,_=COLORAMA_TO_LATEX.get(fore_c,("termBlack",""))
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

    # Start the actual document structure
    latex_string += "\\begin{figure}[p]\n"
    latex_string += "\\centering\n" 
    
    # Use a more balanced two-column layout with horizontal space between them
    latex_string += "\\begin{minipage}[t]{0.45\\textwidth}\n" 
    
    # Left column - grids
    global_header=None; global_table=None; agent_view_latex_data=[]; grid_section_string=""; has_printed_grid=False; message_block=""
    
    if show_global_map_toggle:
        try: 
            global_header_text="\\centering Global Map\\\\"; 
            table_code,_=render_single_grid_latex(grid_data,agent_message_colors_map,coord_to_agent_id_map); 
            global_header=f"\\textbf{{{global_header_text}}}"; 
            global_table=table_code
        except Exception as e: 
            error_latex=f"\\texttt{{(Err prep global R{round_num}: {e})}}"; 
            global_header=f"\\texttt{{Global Err R{round_num}}}"; 
            global_table=error_latex
    
    if cmd_show_views and show_agent_views_toggle:
        agent_views_this_round = views_by_round.get(round_num_int, {})
        if agent_views_this_round:
            agent_ids_in_step={aid for aid in coord_to_agent_id_map.values() if aid}
            try: 
                agents_with_views_in_step=sorted(list(agent_ids_in_step & agent_views_this_round.keys()),key=sort_key); 
                other_agents_with_views=sorted([aid for aid in agent_views_this_round if aid not in agent_ids_in_step],key=sort_key); 
                agents_to_render_views=agents_with_views_in_step+other_agents_with_views
            except Exception: 
                agents_to_render_views = sorted(list(agent_views_this_round.keys()))
                
            for agent_id in agents_to_render_views:
                view_grid = agent_views_this_round.get(agent_id)
                if view_grid:
                    try: 
                        agent_color_fore=agent_message_colors_map.get(agent_id, Fore.WHITE); 
                        latex_fg_color_name,_=COLORAMA_TO_LATEX.get(agent_color_fore,("termWhite","")); 
                        safe_agent_id_header=agent_id.replace('_','\\_'); 
                        view_header_text=f"\\centering View {safe_agent_id_header}"; 
                        latex_header_str=f"\\textcolor{{{latex_fg_color_name}}}{{\\textbf{{{view_header_text}}}}}"; 
                        view_table_latex,view_cols=render_single_grid_latex(view_grid,agent_message_colors_map,{}); 
                        agent_view_latex_data.append((latex_header_str,view_table_latex,view_cols))
                    except Exception as e: 
                        safe_agent_id_err=agent_id.replace('_','\\_'); 
                        error_latex=f"\\texttt{{(Err prep view {safe_agent_id_err}: {e})}}"; 
                        agent_view_latex_data.append((f"\\texttt{{View {safe_agent_id_err} Err}}",error_latex,1))

    # Assemble the global grid and agent views
    if global_table: 
        grid_section_string += "\\begin{center}\n"+global_header+"\n"+global_table+"\n\\end{center}\\vspace{1em}\n\n"
        has_printed_grid=True
        
    if agent_view_latex_data:
        num_agent_grids=len(agent_view_latex_data); 
        num_agent_chunks=math.ceil(num_agent_grids/LATEX_AGENT_VIEWS_PER_ROW)
        
        for chunk_index in range(num_agent_chunks):
            start_index=chunk_index*LATEX_AGENT_VIEWS_PER_ROW; 
            end_index=start_index+LATEX_AGENT_VIEWS_PER_ROW; 
            current_chunk_data=agent_view_latex_data[start_index:end_index]
            
            if not current_chunk_data: continue
            
            # 修改这部分，解决嵌套tabular问题
            num_cols_this_chunk=len(current_chunk_data)
            grid_section_string += "\\noindent\n{\\setlength{\\tabcolsep}{1pt}%\n"; 
            
            # 使用简单的表格结构，避免嵌套引起的问题
            grid_section_string += "\\begin{tabular}{" + " ".join(["p{0.23\\textwidth}"]*num_cols_this_chunk) + "}\n"
            
            # 使用minipage环境包装每个单元格内容，防止内部格式影响外部表格
            header_row = []
            for header, _, _ in current_chunk_data:
                header_row.append(f"\\begin{{minipage}}[t]{{\\linewidth}}\\centering {header}\\end{{minipage}}")
            grid_section_string += " & ".join(header_row) + " \\\\\n"
            
            grid_row = []
            for _, grid_content, _ in current_chunk_data:
                grid_row.append(f"\\begin{{minipage}}[t]{{\\linewidth}}{grid_content}\\end{{minipage}}")
            grid_section_string += " & ".join(grid_row) + " \\\\\n"
            
            grid_section_string+="\\end{tabular}}\n\n\\vspace{0.5em}\n\n"; 
            has_printed_grid=True
            
    if not has_printed_grid and (show_global_map_toggle or (cmd_show_views and show_agent_views_toggle)): 
        grid_section_string += "\\texttt{(No grids selected/found)}\n\n\\vspace{0.5em}\n\n"

    if grid_section_string: 
        latex_string += grid_section_string

    # Messages section
    if show_messages_toggle:
        agent_messages_this_round = messages_by_round.get(round_num_int, {})
        
        if agent_messages_this_round or (not has_printed_grid and not show_global_map_toggle and not (cmd_show_views and show_agent_views_toggle)):
            safe_msg_header=" Agent Messages ".replace('_','\\_'); 
            message_block += f"\\noindent\\textbf{{{safe_msg_header}}}\\\\\n"
            
            if agent_messages_this_round:
                agent_ids_in_step={aid for aid in coord_to_agent_id_map.values() if aid}
                try: 
                    agents_msg_in_step=sorted(list(agent_ids_in_step&agent_messages_this_round.keys()),key=sort_key); 
                    other_agents_with_msg=sorted([aid for aid in agent_messages_this_round if aid not in agent_ids_in_step],key=sort_key); 
                    agents_to_display_msg=agents_msg_in_step+other_agents_with_msg
                except Exception: 
                    agents_to_display_msg = sorted(list(agent_messages_this_round.keys()))
                    
                msg_count = 0
                for agent_id in agents_to_display_msg:
                    message=agent_messages_this_round.get(agent_id,""); 
                    cleaned_message=' '.join(str(message).split())
                    
                    if not cleaned_message: continue
                    
                    agent_color_fore=agent_message_colors_map.get(agent_id,Fore.WHITE); 
                    latex_fg_color_name,_=COLORAMA_TO_LATEX.get(agent_color_fore,("termWhite",""))
                    escaped_message=cleaned_message.replace('\\','\\textbackslash{}').replace('%','\\%').replace('#','\\#').replace('&','\\&')
                    safe_agent_id_msg=agent_id.replace('_','\\_'); 
                    part1=f"\\textcolor{{{latex_fg_color_name}}}{{\\texttt{{{safe_agent_id_msg}:}}}}"; 
                    part2=f"\\texttt{{\\detokenize{{{escaped_message}}}}}"; 
                    message_block+=f"{part1} {part2} \\\\\n"; 
                    msg_count+=1
                    
                if msg_count==0: 
                    message_block+="\\texttt{(No valid messages)}\\\\\n"
            else: 
                message_block+="\\texttt{(No messages this round)}\\\\\n"
    elif not has_printed_grid: 
        message_block+="\\texttt{(Grids and Messages hidden)}\n"

    # Calculate available height for left column and create minipage with appropriate height
    if message_block:
        latex_string += "\\vspace{0.7em}\n"
        latex_string += "\\begin{minipage}[t][15cm][t]{\\textwidth} \n" # 固定高度，确保左侧有足够空间
        latex_string += "\\fontsize{8pt}{9pt}\\selectfont \n"
        latex_string += message_block
        latex_string += "\\end{minipage}\n"
    
    # Close left column
    latex_string += "\\end{minipage}\n" 
    
    # Add horizontal space between columns
    latex_string += "\\hfill\n" 
    
    # Start right column
    latex_string += "\\begin{minipage}[t]{0.5\\textwidth}\n" 
    
    # Main score graph
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

    # Score graph
    latex_string += "\\vspace{-0.3cm}\n" # Start plots higher up
    latex_string += "\\begin{tikzpicture}\n"
    latex_string += "\\begin{axis}[\n"
    latex_string += "    title={Score Progression},\n"
    latex_string += "    xlabel={Frame},\n"
    latex_string += "    ylabel={Score},\n"
    latex_string += "    title style={font=\\bfseries},\n" 

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
    latex_string += "    height=4.8cm,\n" # Adjusted height
    latex_string += "    scaled ticks=false,\n"
    latex_string += "    tick label style={/pgf/number format/fixed},\n"
    latex_string += "    axis background/.style={fill=white, opacity=0.8},\n" 
    latex_string += "    label style={font=\\small},\n" 
    latex_string += "    tick label style={font=\\footnotesize},\n" 
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

    # Now add the metrics data visualization with adjusted heights
    if METRICS_LIBRARIES_AVAILABLE and current_metrics:
        # Define metrics groups and their corresponding colors
        metric_groups = [
            {
                "title": "Message Metrics",
                "metrics": [
                    {"name": "info_homogeneity", "label": "Info Homogeneity", "color": "infoColor", "scale": 2},
                    {"name": "mean_message_length", "label": "Msg Length ÷200", "color": "msgLengthColor", "scale": 200},
                    {"name": "prop_question_sentences", "label": "Question Prop.", "color": "questionColor", "scale": 1},
                    {"name": "prop_digit_chars", "label": "Digit Char Prop.", "color": "digitColor", "scale": 1}
                ]
            },
            {
                "title": "Movement Metrics",
                "metrics": [
                    {"name": "directional_entropy", "label": "Dir. Entropy ÷2", "color": "dirEntropyColor", "scale": 2},
                    {"name": "stillness_proportion", "label": "Stillness Prop.", "color": "stillnessColor", "scale": 1},
                    {"name": "dominant_action_prop", "label": "Dominant Action", "color": "dominantColor", "scale": 1},
                    {"name": "polarization_index", "label": "Polarization", "color": "polarizationColor", "scale": 1}
                ]
            },
            {
                "title": "Exploration Metrics",
                "metrics": [
                    {"name": "avg_moving_distance", "label": "Moving Dist. ÷50", "color": "moveDistColor", "scale": 50},
                    {"name": "exploration_rate", "label": "Explore Rate ÷100", "color": "exploreColor", "scale": 100},
                    {"name": "local_structure_preservation_count", "label": "Structure ÷10", "color": "structureColor", "scale": 10},
                    {"name": "agent_push_events", "label": "Push Events ÷4", "color": "pushColor", "scale": 4}
                ]
            }
        ]

        # 为每个指标组创建时间序列图，这里使用等间距分布所有图表
        # 计算右侧总共可用的高度 - 减去Score Graph和标题的空间
        # total_available_height = 15 - 5.0  # 总高度减去score graph的高度和间距
        # chart_height = total_available_height / 3.0 - 0.5  # 平均分配给3个图表，略微减少以容纳标题
        
        chart_height = 4.8  # 使用与score graph相同的固定高度
        
        # Cache of metric history data to avoid recalculating
        metric_history_cache = {}
        
        for i, group in enumerate(metric_groups):
            # Use a box around the title for better visual separation
            latex_string += f"\\vspace{{0.25cm}}\\noindent{{\\colorbox{{lightgray}}{{\\makebox[\\linewidth]{{\\textbf{{{group['title']}}}}}}}}}\\vspace{{0.1cm}}\n"
            
            latex_string += "\\begin{tikzpicture}[scale=0.9]\n"
            latex_string += "\\begin{axis}[\n"
            latex_string += "    xlabel={Frame},\n"
            latex_string += "    ylabel={Metric Value},\n"
            latex_string += "    width=\\textwidth,\n"
            latex_string += f"    height={chart_height}cm,\n" # 使用计算的高度
            latex_string += "    legend style={at={(0.5,1.03)}, anchor=south, legend columns=-1, inner sep=0pt, font=\\footnotesize},\n"
            latex_string += "    tick label style={font=\\tiny},\n"
            latex_string += "    label style={font=\\tiny},\n"
            latex_string += "    scale only axis,\n"
            latex_string += "    enlarge x limits=false,\n"

            latex_string += "    scaled y ticks=false,\n"
            latex_string += "    ymin=0,\n"
            latex_string += "    ymax=1.1,\n"
            latex_string += "    ytick={0,0.25,0.5,0.75,1.0},\n"
            latex_string += "    xmin=0, xmax=" + str(current_step_index) + ",\n"
            latex_string += "    xtick={0," + str(int(current_step_index/4)) + "," + str(int(current_step_index/2)) + "," + str(int(3*current_step_index/4)) + "," + str(current_step_index) + "},\n"
            latex_string += "    ymajorgrids=true,\n"
            latex_string += "    grid style={dotted,gray},\n"
            latex_string += "    no markers\n"
            latex_string += "]\n"
            
            # 为每个指标添加完整的时间序列线
            for metric in group["metrics"]:
                metric_name = metric["name"]
                scale = metric["scale"]
                
                # Use cached metric history if available
                cache_key = f"{metric_name}_{current_step_index}"
                if cache_key in metric_history_cache:
                    metric_history = metric_history_cache[cache_key]
                else:
                    # 收集该指标的历史数据
                    metric_history = []
                    try:
                        for step_idx in range(current_step_index + 1):
                            # Reuse previously calculated metrics from cache
                            step = game_steps[step_idx]
                            step_cache_key = f"{step.get('timestamp', 'unknown')}_{step_idx}"
                            
                            if step_cache_key in metrics_cache and metric_name in metrics_cache[step_cache_key]:
                                raw_value = metrics_cache[step_cache_key][metric_name]
                            else:
                                # Calculate metrics for this step if not cached
                                step_metrics = calculate_metrics_for_frame(
                                    step, messages_by_round, game_steps, step_idx, 
                                    agent_log=agent_log, embedding_model=embedding_model
                                )
                                raw_value = step_metrics.get(metric_name, 0.0)
                                
                            scaled_value = raw_value / scale
                            metric_history.append((step_idx, scaled_value))
                    except Exception as e:
                        # If error, at least add current frame's value
                        if metric_name in current_metrics:
                            raw_value = current_metrics[metric_name]
                            scaled_value = raw_value / scale
                            metric_history.append((current_step_index, scaled_value))
                    
                    # Cache the metric history
                    metric_history_cache[cache_key] = metric_history
                
                # 绘制完整的时间序列线
                if metric_history:
                    latex_string += f"\\addplot[color={metric['color']}, line width=1.2pt] coordinates {{\n"
                    for frame, value in metric_history:
                        latex_string += f"    ({frame},{value})\n"
                    latex_string += "};\n"
                    latex_string += f"\\addlegendentry{{{metric['label']}}}\n"
                
            latex_string += "\\end{axis}\n"
            latex_string += "\\end{tikzpicture}\n"
    else:
        # If metrics not available, add a note
        if not METRICS_LIBRARIES_AVAILABLE:
            latex_string += "\\vspace{0.5cm}\n"
            latex_string += "\\begin{center}\n"
            latex_string += "\\textit{Statistical metrics not available. Install required packages.}\n"
            latex_string += "\\end{center}\n"
            latex_string += "\\vspace{0.5cm}\n"

    # Compact legend for score graph
    latex_string += "\\vspace{0.2cm}\n"
    latex_string += "\\noindent{\\scriptsize\\textbf{Score Legend:} "
    latex_string += "{\\color{blue}\\rule{1em}{2pt}} Score value, "
    latex_string += "{\\color{red}$\\bullet$} Current frame}\n"
    
    # Add vertical space filler to make right column match left column height
    latex_string += "\\vfill\n"
    
    # Close right column
    latex_string += "\\end{minipage}\n" 
    latex_string += "\\end{figure}\n" 

    if not generate_content_only: 
        latex_string += "\n\\end{document}\n"
    return latex_string


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


def _generate_images_from_latex(latex_dir, image_prefix="frame", dpi=150, force_recompile=False, max_workers=None):
    """
    Compiles .tex files in latex_dir to .pdf, then converts .pdf to .png images.
    Stores intermediate PDFs in a 'pdf_gen' subdirectory and PNGs in 'image_gen'.
    Returns the path to the image directory or None on failure.
    Uses multiprocessing to speed up compilation.
    """
    import subprocess
    import glob
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print(f"{Fore.RED}Error: The 'pdf2image' library is required for creating animations.")
        print(f"Please install it using: pip install pdf2image{Style.RESET_ALL}")
        print(f"You might also need to install poppler: https://pdf2image.readthedocs.io/en/latest/installation.html")
        return None

    # 如果没有指定最大工作进程数，使用 CPU 数量的 90%
    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.9))
    print(f"Using {max_workers} parallel workers for LaTeX compilation")

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
        
        print(f"Compiling {len(latex_files)} LaTeX files to PDF (DPI for images: {dpi}, Workers: {max_workers})...")
        
        # 使用进程池并行处理文件
        total_files = len(latex_files)
        processed_count = 0
        success_count = 0
        error_count = 0
        
        # 使用ProcessPoolExecutor管理进程池
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务列表，包含索引和文件路径以及所有需要的参数
            tasks = [(i, latex_file, pdf_dir, img_dir, dpi, force_recompile) 
                     for i, latex_file in enumerate(latex_files)]
            
            # 提交所有任务，获取future对象
            futures = {executor.submit(_process_single_latex_file, task_info): task_info 
                      for task_info in tasks}
            
            # 处理已完成的任务结果
            for future in as_completed(futures):
                index, img_path, error, status = future.result()
                processed_count += 1
                
                if status == "success":
                    success_count += 1
                elif status == "already_exists":
                    success_count += 1
                    status_str = "skipped"
                else:
                    error_count += 1
                    status_str = "ERROR"
                
                # 定期输出进度信息 
                if processed_count % 10 == 0 or processed_count == total_files:
                    print(f"  Progress: {processed_count}/{total_files} files ({success_count} success, {error_count} errors)")
                
                # 输出详细信息
                if error:
                    print(f"  {Fore.RED}[{index+1}/{total_files}] {status_str}: {error}{Style.RESET_ALL}")
                elif processed_count % 50 == 0 or processed_count == total_files:
                    status_display = "Compiled" if status == "success" else "Skipped"
                    print(f"  [{index+1}/{total_files}] {status_display}: {os.path.basename(img_path) if img_path else 'N/A'}")

        print(f"Compilation complete: {success_count} successful, {error_count} failed out of {total_files} files")
    else:
        print(f"Found existing compiled PDFs in {pdf_dir} and images in {img_dir}. Using them. (Use --force-recompile to regenerate)")

    # 最后检查是否生成了图片
    image_files = glob.glob(os.path.join(img_dir, "*.png"))
    if not image_files:
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


def batch_export_latex_frames(game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log, 
                             info, timestamp, cmd_show_views, output_dir, messages_by_round, prefix="frame",
                             agent_log=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Exporting {len(game_steps)} frames to {output_dir}...")
    
    # Initialize embedding model for metrics if available
    embedding_model = None
    global embedding_cache
    
    if METRICS_LIBRARIES_AVAILABLE:
        try:
            print(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}...")
            # Initialize global embedding cache
            embedding_cache = EmbeddingCache(model_name=DEFAULT_EMBEDDING_MODEL)
            
            # Get the model if needed for direct calls
            embedding_model = embedding_cache.get_model()
            
            print("Embedding model and cache loaded for metrics calculation.")
        except Exception as e:
            print(f"Warning: Could not load embedding model for metrics: {e}")
    
    # Add timestamp to all steps for caching purposes
    for step in game_steps:
        if isinstance(step, dict) and 'timestamp' not in step:
            step['timestamp'] = timestamp
    
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
                game_steps=game_steps, agent_log=agent_log,
                generate_content_only=False, embedding_model=embedding_model
            )
            filename = f"{prefix}_{i:04d}.tex" # Start frame index from 0 for consistency with current_step_index
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            if (i+1) % 10 == 0 or i == len(game_steps)-1 : # Print progress periodically
                print(f"  Exported frame {i+1}/{len(game_steps)}: {filepath}")
        except Exception as e:
            print(f"{Fore.RED}Error exporting frame {i+1} (round {round_num_str}): {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    # Save final cache
    if embedding_cache is not None:
        embedding_cache.save_cache()
        
    print(f"Export complete. Frames exported to {output_dir}")
    return True


# 在模块级别定义这个函数
def _process_single_latex_file(file_info):
    """处理单个LaTeX文件的编译和转换
    
    Args:
        file_info: 包含 (index, latex_file, pdf_dir, img_dir, dpi, force_recompile) 的元组
    """
    import subprocess
    import os
    from pdf2image import convert_from_path
    
    index, latex_file, pdf_dir, img_dir, dpi, force_recompile = file_info
    base_name = os.path.basename(latex_file).replace('.tex', '')
    pdf_file_path = os.path.join(pdf_dir, f"{base_name}.pdf")
    img_file_path = os.path.join(img_dir, f"{base_name}.png")

    # 如果图片已存在且不强制重新编译，则跳过
    if os.path.exists(img_file_path) and not force_recompile:
        return index, img_file_path, None, "already_exists"

    try:
        # 编译 LaTeX 为 PDF
        if not os.path.exists(pdf_file_path) or force_recompile:
            compile_process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", pdf_dir, latex_file],
                capture_output=True, text=True, errors='ignore', timeout=60
            )
            if compile_process.returncode != 0:
                return index, None, f"Error compiling {latex_file}: returncode {compile_process.returncode}", "compile_error"
        
        # 转换 PDF 为 PNG
        if os.path.exists(pdf_file_path):
            images = convert_from_path(pdf_file_path, dpi=dpi, first_page=1, last_page=1)
            if images:
                images[0].save(img_file_path, 'PNG')
                return index, img_file_path, None, "success"
            else:
                return index, None, f"No image generated from {pdf_file_path}", "no_image"
        else:
            return index, None, f"PDF file {pdf_file_path} not found after compilation attempt", "no_pdf"

    except subprocess.TimeoutExpired:
        return index, None, f"Timeout compiling {latex_file}", "timeout"
    except Exception as e:
        return index, None, f"Error processing {latex_file} to image: {e}", "exception"



def calculate_metrics_for_frame(step_data, messages_by_round, game_steps, current_step_index, agent_log=None, embedding_model=None):
    """Calculate metrics for a specific frame with caching"""
    global metrics_cache
    
    # Create cache key based on step_index and game_id
    cache_key = f"{step_data.get('timestamp', 'unknown')}_{current_step_index}"
    
    # Return cached metrics if available
    if cache_key in metrics_cache:
        return metrics_cache[cache_key]
    
    metrics = {}
    try:
        # Basic info retrieval 
        round_num = int(step_data['round'])
        grid_data = step_data.get('grid', [])
        agents_list = step_data.get('agents', [])
        
        # Position analysis
        positions = {}
        for agent in agents_list:
            if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                try:
                    agent_id = str(agent['id'])
                    positions[agent_id] = (int(agent['x']), int(agent['y']))
                except (ValueError, TypeError): 
                    continue
        
        # Previous positions for comparison
        prev_positions = {}
        if current_step_index > 0 and game_steps:
            prev_step = game_steps[current_step_index-1]
            if isinstance(prev_step, dict) and 'agents' in prev_step:
                for agent in prev_step.get('agents', []):
                    if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                        try:
                            agent_id = str(agent['id'])
                            prev_positions[agent_id] = (int(agent['x']), int(agent['y']))
                        except (ValueError, TypeError):
                            continue

        # Action analysis
        actions_in_round = []
        if agent_log:
            for entry in agent_log:
                if isinstance(entry, dict) and 'round' in entry and entry['round'] == round_num and 'action' in entry:
                    action = entry.get('action')
                    if action:
                        actions_in_round.append(action)
        
        # Message analysis
        round_messages = []
        for agent_id, msg in messages_by_round.get(round_num, {}).items():
            if msg:
                round_messages.append(str(msg))
        
        # Calculate metrics
        
        # 1. Directional entropy
        move_actions = [a for a in actions_in_round if a in ACTUAL_MOVE_ACTIONS]
        metrics['directional_entropy'] = shannon_entropy(move_actions)
        
        # 2. Stillness proportion
        if actions_in_round:
            stay_count = actions_in_round.count('STAY')
            metrics['stillness_proportion'] = stay_count / len(actions_in_round)
        else:
            metrics['stillness_proportion'] = 0.0
            
        # 3. Message metrics
        if round_messages:
            message_lengths = [len(msg) for msg in round_messages]
            metrics['mean_message_length'] = np.mean(message_lengths)
            metrics['std_message_length'] = np.std(message_lengths) if len(message_lengths) >= 2 else 0.0
        else:
            metrics['mean_message_length'] = 0.0
            metrics['std_message_length'] = 0.0
        
        # 4. Count of questions in messages
        question_mark_count = 0
        total_chars_in_round = 0
        digit_chars_in_round = 0
        
        if round_messages:
            for msg in round_messages:
                if "?" in msg:
                    question_mark_count += 1
                for char in msg:
                    total_chars_in_round += 1
                    if char.isdigit():
                        digit_chars_in_round += 1
                
            metrics['prop_question_sentences'] = question_mark_count / len(round_messages) if round_messages else 0.0
            metrics['prop_digit_chars'] = digit_chars_in_round / total_chars_in_round if total_chars_in_round > 0 else 0.0
        else:
            metrics['prop_question_sentences'] = 0.0
            metrics['prop_digit_chars'] = 0.0
        
        # 5. Information homogeneity (using cached embeddings)
        if embedding_model and len(round_messages) >= 2 and embedding_cache is not None:
            try:
                unique_messages = list(set(round_messages))
                if len(unique_messages) >= 2:
                    # Use embedding cache for faster processing
                    embeddings = []
                    for msg in unique_messages:
                        embedding = embedding_cache.get_embedding(msg)
                        if embedding is not None:
                            embeddings.append(embedding)
                    
                    if len(embeddings) >= 2:
                        embeddings = np.array(embeddings)
                        cos_sim_matrix = cosine_similarity(embeddings)
                        upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
                        pairwise_sims = cos_sim_matrix[upper_triangle_indices]
                        metrics['info_homogeneity'] = np.nanmean(pairwise_sims) if len(pairwise_sims) > 0 else 0.0
                    else:
                        metrics['info_homogeneity'] = 1.0  # Perfect homogeneity if couldn't compute
                else:
                    metrics['info_homogeneity'] = 1.0  # Perfect homogeneity if only one unique message
            except Exception as e:
                print(f"Error calculating info_homogeneity: {e}")
                metrics['info_homogeneity'] = 0.0
        else:
            metrics['info_homogeneity'] = 0.0
        
        # 6. Movement metrics
        cumulative_distances = {}
        for agent_id, curr_pos in positions.items():
            if agent_id in prev_positions:
                prev_pos = prev_positions[agent_id]
                distance_moved = manhattan_distance(prev_pos, curr_pos)
                cumulative_distances[agent_id] = distance_moved
        
        if cumulative_distances:
            metrics['avg_moving_distance'] = sum(cumulative_distances.values()) / len(cumulative_distances)
        else:
            metrics['avg_moving_distance'] = 0.0
            
        # 7. Exploration rate (count of unique cells explored so far)
        explored_cells = set()
        for i in range(current_step_index + 1):
            if i < len(game_steps) and 'agents' in game_steps[i]:
                for agent in game_steps[i].get('agents', []):
                    if isinstance(agent, dict) and 'x' in agent and 'y' in agent:
                        try:
                            explored_cells.add((int(agent['x']), int(agent['y'])))
                        except (ValueError, TypeError):
                            continue
        metrics['exploration_rate'] = len(explored_cells)
        
        # 8. Coordination metrics
        actions_for_coordination = [a for a in actions_in_round if a in COORDINATION_ACTIONS]
        if actions_for_coordination:
            # Dominant action proportion
            action_counts = Counter(actions_for_coordination)
            max_freq = action_counts.most_common(1)[0][1] if action_counts else 0
            metrics['dominant_action_prop'] = max_freq / len(actions_for_coordination) if len(actions_for_coordination) > 0 else 0.0
            
            # Polarization index
            sum_vector = np.zeros(2)
            for action in actions_for_coordination:
                sum_vector += ACTION_VECTORS.get(action, np.array([0,0]))
            avg_vector = sum_vector / len(actions_for_coordination) if len(actions_for_coordination) > 0 else np.zeros(2)
            metrics['polarization_index'] = np.linalg.norm(avg_vector)
        else:
            metrics['dominant_action_prop'] = 0.0
            metrics['polarization_index'] = 0.0
        
        # 9. Local structure preservation
        metrics['local_structure_preservation_count'] = 0
        agent_ids = list(positions.keys())
        if len(agent_ids) >= 2 and prev_positions:
            for i in range(len(agent_ids)):
                for j in range(i+1, len(agent_ids)):
                    agent_i = agent_ids[i]
                    agent_j = agent_ids[j]
                    if agent_i in prev_positions and agent_j in prev_positions and agent_i in positions and agent_j in positions:
                        prev_dist = manhattan_distance(prev_positions[agent_i], prev_positions[agent_j])
                        curr_dist = manhattan_distance(positions[agent_i], positions[agent_j])
                        # Check if agents maintained adjacency
                        if prev_dist == 1 and curr_dist == 1:
                            metrics['local_structure_preservation_count'] += 1
        
        # 10. Agent push events
        metrics['agent_push_events'] = 0
        if prev_positions and current_step_index > 0:
            prev_round = int(game_steps[current_step_index-1].get('round', 0))
            actions_prev_round = {}
            # Get actions from previous round
            if agent_log:
                for entry in agent_log:
                    if isinstance(entry, dict) and 'round' in entry and entry['round'] == prev_round and 'action' in entry and 'agent_id' in entry:
                        actions_prev_round[entry['agent_id']] = entry['action']
            
            # Check for push events
            for agent_A_id, pos_A_prev in prev_positions.items():
                if agent_A_id not in actions_prev_round or agent_A_id not in positions:
                    continue
                    
                action_A = actions_prev_round.get(agent_A_id)
                if action_A not in ACTUAL_MOVE_ACTIONS:
                    continue
                    
                for agent_B_id, pos_B_prev in prev_positions.items():
                    if agent_A_id == agent_B_id or agent_B_id not in positions:
                        continue
                    
                    # Check if agents were adjacent
                    if manhattan_distance(pos_A_prev, pos_B_prev) != 1:
                        continue
                        
                    # Calculate intended position after A's move
                    intended_A_pos = tuple(np.array(pos_A_prev) + ACTION_VECTORS[action_A])
                    
                    # Check if A moved into B's previous position
                    if intended_A_pos == pos_B_prev and positions[agent_A_id] == pos_B_prev:
                        # Calculate expected position for B if pushed
                        expected_B_pos = tuple(np.array(pos_B_prev) + ACTION_VECTORS[action_A])
                        
                        # Check if B actually moved in the expected direction
                        if positions[agent_B_id] == expected_B_pos:
                            metrics['agent_push_events'] += 1
        
        # Store metrics in cache
        metrics_cache[cache_key] = metrics
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {}

# 

# 
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

    #  Model selection arguments 
    parser.add_argument('--model-name', type=str, default=None,
                        help='Specify the model name to filter game logs. Processes the first match found in meta_log.json if --draw-best/worst not used.')
    parser.add_argument('--draw-best', action='store_true',
                        help='Replay/Export the game with the highest final score for the specified --model-name. Requires --model-name.')
    parser.add_argument('--draw-worst', action='store_true',
                        help='Replay/Export the game with the lowest final score for the specified --model-name. Requires --model-name.')

    # Metrics options
    parser.add_argument('--disable-metrics', action='store_true',
                        help='Disable the calculation and display of metrics in exported frames.')
    
    # Embedding cache options
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all embedding cache before starting.')

    # 在 argparse 部分添加以下参数
    parser.add_argument('--max-workers', type=int, default=64, 
                    help='Maximum number of parallel workers for LaTeX compilation. Default: 90% of available CPU cores.')


    args = parser.parse_args()

    if args.draw_best and args.draw_worst:
        print(f"{Fore.RED}Error: --draw-best and --draw-worst are mutually exclusive.{Style.RESET_ALL}")
        sys.exit(1)
    # if (args.draw_best or args.draw_worst) and not args.model_name:
    #     print(f"{Fore.RED}Error: --draw-best or --draw-worst requires --model-name to be specified.{Style.RESET_ALL}")
    #     sys.exit(1)
    
    debug_mode = args.debug
    
    # Handle cache clearing
    if args.clear_cache:
        try:
            import shutil
            print(f"Clearing cache directory: {CACHE_DIR}")
            if os.path.exists(EMBEDDING_CACHE_DIR):
                shutil.rmtree(EMBEDDING_CACHE_DIR)
                os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
            if os.path.exists(METRICS_CACHE_DIR):
                shutil.rmtree(METRICS_CACHE_DIR)
                os.makedirs(METRICS_CACHE_DIR, exist_ok=True)
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"{Fore.RED}Error clearing cache: {e}{Style.RESET_ALL}")
            if args.debug:
                import traceback
                traceback.print_exc()



    if args.create_video or args.create_gif:
        args.export_all_latex = True 
        if not _ensure_pdflatex_ffmpeg_installed():
            sys.exit(1)

    log_dir = args.log_dir
    TIME = args.time
    cmd_show_views = args.show_views
    max_grids_per_row_terminal = args.max_grids
    
    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path):
        print(f"{Fore.RED}Meta log not found: {meta_log_path}{Style.RESET_ALL}")
        sys.exit(1)
    try:
        with open(meta_log_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error reading meta log {meta_log_path}: {e}{Style.RESET_ALL}")
        sys.exit(1)

    if not meta:
        print(f"{Fore.RED}Meta log is empty or invalid: {meta_log_path}{Style.RESET_ALL}")
        sys.exit(1)

    timestamp = None
    info = None
    game_selection_criteria_message = "" # To describe how the game was selected

    if args.draw_best or args.draw_worst:
        # This block handles --draw-best and --draw-worst
        # args.model_name is guaranteed to be set due to earlier checks
        criteria = "highest" if args.draw_best else "lowest"
        print(f"{Style.BRIGHT}Searching for game with {criteria} score for model: {args.model_name}{Style.RESET_ALL}")
        
        candidate_games = [] # List of (score, ts_key, game_info_val)
        for ts_key, game_info_val in meta.items():
            # if isinstance(game_info_val, dict) and game_info_val.get('model') == args.model_name:
            if isinstance(game_info_val, dict) and args.model_name in game_info_val.get('model'):
                temp_game_log_path = os.path.join(log_dir, f'game_log_{ts_key}.json')
                if os.path.exists(temp_game_log_path):
                    try:
                        with open(temp_game_log_path, 'r', encoding='utf-8') as f_game:
                            temp_game_steps = json.load(f_game)
                        if temp_game_steps and isinstance(temp_game_steps, list) and len(temp_game_steps) > 0:
                            last_step = temp_game_steps[-1]
                            if isinstance(last_step, dict) and 'score' in last_step:
                                try:
                                    # Attempt to clean and convert score
                                    score_str = str(last_step['score']).strip()
                                    # Remove non-numeric characters except decimal point and negative sign at the start
                                    score_str_cleaned = re.sub(r'[^\d.-]', '', score_str) 
                                    if score_str_cleaned and score_str_cleaned != '-' and score_str_cleaned != '.': # Ensure not empty/invalid after cleaning
                                        score = float(score_str_cleaned)
                                        candidate_games.append((score, ts_key, game_info_val))
                                    else:
                                         if debug_mode: print(f"{Fore.YELLOW}Debug: Score became empty/invalid after cleaning for {ts_key}, original: '{last_step['score']}', cleaned: '{score_str_cleaned}'{Style.RESET_ALL}")
                                except (ValueError, TypeError) as e_score:
                                    if debug_mode: print(f"{Fore.YELLOW}Debug: Could not parse score '{last_step['score']}' (cleaned: '{score_str_cleaned}') for {ts_key}: {e_score}{Style.RESET_ALL}")
                            else:
                                if debug_mode: print(f"{Fore.YELLOW}Debug: No score in last step or last step invalid for {ts_key}{Style.RESET_ALL}")
                        else:
                            if debug_mode: print(f"{Fore.YELLOW}Debug: Empty or invalid game_steps for {ts_key}{Style.RESET_ALL}")
                    except Exception as e_load:
                        if debug_mode: print(f"{Fore.YELLOW}Debug: Error loading or parsing game log {ts_key}: {e_load}{Style.RESET_ALL}")
                else:
                    if debug_mode: print(f"{Fore.YELLOW}Debug: Game log file not found for {ts_key}: {temp_game_log_path}{Style.RESET_ALL}")
        
        if not candidate_games:
            print(f"{Fore.RED}Error: No suitable game logs found with parseable scores for model '{args.model_name}' to determine {criteria} score.{Style.RESET_ALL}")
            sys.exit(1)

        # Sort candidates: descending for best, ascending for worst
        candidate_games.sort(key=lambda x: x[0], reverse=args.draw_best)
        
        selected_score, timestamp, info = candidate_games[0]
        game_selection_criteria_message = f" (Model: {args.model_name}, {criteria.capitalize()} Score: {selected_score})"
        print(f"{Fore.GREEN}Selected game with {criteria} score ({selected_score}) for model '{args.model_name}' with timestamp: {timestamp}{Style.RESET_ALL}")

    elif args.model_name:
        # This block handles --model-name without --draw-best/worst
        print(f"{Style.BRIGHT}Attempting to find first matching game log for model: {args.model_name}{Style.RESET_ALL}")
        found_match = False
        for ts_key, game_info_val in meta.items():
            if isinstance(game_info_val, dict) and game_info_val.get('model') == args.model_name:
                timestamp = ts_key
                info = game_info_val
                found_match = True
                game_selection_criteria_message = f" (Model: {args.model_name}, First Match)"
                print(f"{Fore.GREEN}Found matching game log for model '{args.model_name}' with timestamp: {timestamp}{Style.RESET_ALL}")
                break 
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
                if count >= 5: 
                    if len(meta) > 5: print("  ...")
                    break
            sys.exit(1)
    else:
        # Original behavior: process the first entry if no model_name is specified
        try:
            timestamp = next(iter(meta.keys()))
            info = meta[timestamp]
            game_selection_criteria_message = " (First Log Entry)"
            print(f"{Style.BRIGHT}No model name specified. Processing the first game log found in meta_log.{Style.RESET_ALL}")
        except StopIteration:
            print(f"{Fore.RED}No games found in meta log: {meta_log_path}{Style.RESET_ALL}")
            sys.exit(1)
        except KeyError: 
            print(f"{Fore.RED}Error accessing first game entry in meta log.{Style.RESET_ALL}")
            sys.exit(1)

    if not timestamp or not info: 
        print(f"{Fore.RED}Fatal: Could not determine a game log to process. Timestamp or info is missing.{Style.RESET_ALL}")
        sys.exit(1)

    print(f"{Style.BRIGHT}Selected game for processing{game_selection_criteria_message}:{Style.RESET_ALL}")
    print(f"  Timestamp: {timestamp}, Model from meta: {info.get('model', 'N/A') if isinstance(info, dict) else 'N/A (invalid info)'}")

    game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')
    agent_log_path = os.path.join(log_dir, f'agent_log_{timestamp}.json')

    game_steps = []; agent_log = None; messages_by_round = defaultdict(dict); views_by_round = defaultdict(dict); all_agent_ids_in_log = set()
    try:
        if os.path.exists(game_log_path):
             with open(game_log_path, encoding='utf-8') as f: game_steps = json.load(f)
        else: print(f"{Fore.RED}Game log missing: {game_log_path}"); sys.exit(1)
        if not game_steps: print(f"{Fore.RED}Game log empty: {game_log_path}"); sys.exit(1)
        
        # Add timestamp to all steps for caching purposes
        for step in game_steps:
            if isinstance(step, dict) and 'timestamp' not in step:
                step['timestamp'] = timestamp
                
        if os.path.exists(agent_log_path):
            with open(agent_log_path, encoding='utf-8') as f: 
                agent_log = json.load(f)
                for record in agent_log:
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
    
    if args.export_all_latex:
        print(f"{Style.BRIGHT} Starting Batch Export {Style.RESET_ALL}")
        print(f"Game log: {timestamp}{game_selection_criteria_message}") # Used descriptive message
        print(f"Output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize embedding model for metrics calculation
        embedding_model = None
        
        
        if METRICS_LIBRARIES_AVAILABLE and not args.disable_metrics:
            try:
                print(f"Loading embedding model for metrics calculation...")
                # Initialize global embedding cache
                embedding_cache = EmbeddingCache(model_name=DEFAULT_EMBEDDING_MODEL)
                
                # Get the model if needed for direct calls
                embedding_model = embedding_cache.get_model()
                
                print(f"Embedding model loaded successfully.")
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Could not load embedding model for metrics: {e}. Some metrics will not be calculated.{Style.RESET_ALL}")

        export_success = batch_export_latex_frames(
            game_steps, agent_message_colors_map, views_by_round, all_agent_ids_in_log,
            info, timestamp, cmd_show_views, args.output_dir, messages_by_round, prefix="frame",
            agent_log=agent_log  # Pass agent_log for metrics calculation
        )
        
        generated_image_dir = None
        generated_image_dpi = -1

        if export_success and args.create_video:
            print(f"{Style.BRIGHT} Creating Video {Style.RESET_ALL}")
            if not generated_image_dir or generated_image_dpi != args.video_dpi or args.force_recompile:
                print(f"Generating images for video (DPI: {args.video_dpi})...")
                generated_image_dir = _generate_images_from_latex(
                    args.output_dir, 
                    image_prefix="frame", 
                    dpi=args.video_dpi, 
                    force_recompile=args.force_recompile,
                    max_workers=args.max_workers
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

        if export_success and args.create_gif: # Corrected nesting
            print(f"{Style.BRIGHT} Creating GIF {Style.RESET_ALL}")
            if not generated_image_dir or generated_image_dpi != args.gif_dpi or args.force_recompile:
                print(f"Generating images for GIF (DPI: {args.gif_dpi})...")
                current_force_recompile_for_gif_images = args.force_recompile
                if generated_image_dir and generated_image_dpi == args.gif_dpi and not args.force_recompile:
                    print(f"Reusing existing images from {generated_image_dir} (DPI: {generated_image_dpi}) for GIF.")
                else:
                    generated_image_dir_for_gif = _generate_images_from_latex(
                        args.output_dir, 
                        image_prefix="frame", 
                        dpi=args.gif_dpi, 
                        force_recompile=current_force_recompile_for_gif_images,
                        max_workers=args.max_workers
                    )
                    if generated_image_dir_for_gif:
                        generated_image_dir = generated_image_dir_for_gif
                        generated_image_dpi = args.gif_dpi
            else:
                 print(f"Reusing existing images from {generated_image_dir} (DPI: {generated_image_dpi}) for GIF.")

            if generated_image_dir and generated_image_dpi == args.gif_dpi :
                gif_file_path = os.path.join(args.output_dir, args.gif_file)
                gif_success = convert_images_to_gif(
                    generated_image_dir, gif_file_path, args.fps, 
                    args.gif_width, args.gif_height, image_prefix="frame",
                    dither_algo=args.gif_dither,
                    gifsicle_optimize=not args.no_gifsicle
                )
                if gif_success:
                    print(f"{Fore.GREEN}GIF creation complete: {gif_file_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to create GIF.{Style.RESET_ALL}")
            elif not generated_image_dir:
                 print(f"{Fore.RED}Failed to generate/find images for GIF.{Style.RESET_ALL}")
            elif generated_image_dpi != args.gif_dpi:
                 print(f"{Fore.RED}Images found but at wrong DPI ({generated_image_dpi}) for GIF (expected {args.gif_dpi}). This indicates an issue in image generation logic.{Style.RESET_ALL}")
        
        print(f"{Style.BRIGHT} Batch Export Finished {Style.RESET_ALL}")
        sys.exit(0)

    if not readchar:
        print(f"{Fore.RED}Cannot start interactive mode: 'readchar' library not found or failed to import.{Style.RESET_ALL}")
        print("If you only intended to export, the process might have completed above.")
        sys.exit(1)

    current_step_index = 0; paused = True; latex_mode = False
    last_total_lines_printed = 0; needs_redraw = True; current_latex_output = ""
    copy_confirmation_msg = ""
    show_global_map_toggle = True; show_agent_views_toggle = True; show_messages_toggle = True

    print(f"{Style.BRIGHT}Starting Interactive Replay for game log: {timestamp}{game_selection_criteria_message}{Style.RESET_ALL}") # Used descriptive message
    model_display = info.get('model', 'N/A') if isinstance(info, dict) else 'N/A'
    num_agents_display = info.get('num_agents', 'N/A') if isinstance(info, dict) else 'N/A'
    print(f"Model from meta: {model_display}, Agents in meta: {num_agents_display}")
    print("Press 'Q' to quit, [Space] to Play/Pause, [<-][->] to navigate frames.")
    time.sleep(1)

    # Initialize embedding model for metrics calculation in interactive mode
    embedding_model = None
    # global embedding_cache
    
    if METRICS_LIBRARIES_AVAILABLE and not args.disable_metrics:
        try:
            print(f"Loading embedding model for metrics calculation...")
            # Initialize global embedding cache
            embedding_cache = EmbeddingCache(model_name=DEFAULT_EMBEDDING_MODEL)
            
            # Get the model if needed for direct calls
            embedding_model = embedding_cache.get_model()
            
            print(f"Embedding model loaded successfully.")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load embedding model for metrics: {e}. Some metrics will not be calculated.{Style.RESET_ALL}")

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
                # Updated LaTeX mode header
                print(f"{Style.BRIGHT} LaTeX Mode  (Game: {timestamp}{game_selection_criteria_message} | Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num_str}){Style.RESET_ALL}")
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
                        game_steps=game_steps, agent_log=agent_log,
                        generate_content_only=True, embedding_model=embedding_model
                    )
                except Exception as e:
                    current_latex_output = f"% Error generating LaTeX snippet: {e}"
                    display_message = f"{Fore.RED} Error during LaTeX generation: {e}{Style.RESET_ALL}"
                print('\033[H\033[J', end='')
                print(f"{Style.BRIGHT} LaTeX Snippet (Game: {timestamp}{game_selection_criteria_message} | Frame: {current_step_index+1}/{len(game_steps)}, Round: {round_num_str}) {Style.RESET_ALL}")
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

                model_name_from_info = info.get("model", "N/A") if isinstance(info, dict) else "N/A"
                num_agents_meta = info.get("num_agents", "N/A") if isinstance(info, dict) else "N/A"
                num_agents_actual = len(coord_to_agent_id_map)
                pause_indicator = f'{Style.BRIGHT + Fore.YELLOW}> PAUSED <{Style.RESET_ALL}' if paused else ""
                
                # Updated Terminal mode header
                round_info_line = (f'{pause_indicator}\n'
                                   f'Game: {timestamp}{game_selection_criteria_message}\n'
                                   f'Frame: {current_step_index+1}/{len(game_steps)} | Round: {round_num_str:<3} | Level: {level:<2} | '
                                   f'Score: {score:<4} | Agents: {num_agents_actual:<2} (Meta: {num_agents_meta}) | Model (meta): {model_name_from_info}')
                print(round_info_line); current_lines_printed += round_info_line.count('\n') + 1
                if not agents_list_valid: print(f"{Fore.YELLOW}Warn: 'agents' list missing R{round_num_str}."); current_lines_printed += 1

                all_rendered_grids = [];
                if show_global_map_toggle and grid_data:
                    try: g_lines, g_width = render_terminal(grid_data, agent_message_colors_map, coord_to_agent_id_map); all_rendered_grids.append((" Global Map ", g_lines, g_width))
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
                    if print_msg_header: print("\n Agent Messages "); current_lines_printed += 2
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
            # Simplified auto-advance and input handling
            if not paused and not latex_mode:
                time.sleep(0.05) # Small delay for frame visibility
                if current_step_index < len(game_steps) - 1:
                    current_step_index += 1
                    needs_redraw = True
                else:
                    paused = True # Pause at the end of the log
                    copy_confirmation_msg = f"{Fore.YELLOW}End of log. Paused.{Style.RESET_ALL}"
                    needs_redraw = True
                
                if needs_redraw: # If auto-advanced, skip reading key for this iteration, redraw on next.
                    continue 
            
            # Always read key if paused, in LaTeX mode, or after an auto-advance attempt that didn't redraw.
            key = readchar.readkey()

        except KeyboardInterrupt: key = '\x03'

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
        
        if action_taken or current_step_index != original_index or copy_confirmation_msg: needs_redraw = True
        if debug_mode and not needs_redraw and not (key_lower == 'q' or key == '\x03') : needs_redraw = True

    # Save the embedding cache before exiting
    if embedding_cache is not None:
        embedding_cache.save_cache()

    print("Replay finished.")
