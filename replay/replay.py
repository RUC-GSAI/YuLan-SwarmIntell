import numpy as np
from colorama import init, Fore, Back, Style, Cursor
import json
import os
import sys
import time
from collections import defaultdict
import re
import argparse
import math # For ceiling division

# --- Configuration ---
DEFAULT_LOG_DIR = '../experiment_v15'
DEFAULT_TIME = 0.0
DEFAULT_MAX_GRIDS_PER_ROW = 4 # NEW: Default limit for horizontal grids
GRID_SEPARATOR = "  |  "

# --- Colorama Initialization ---
init(autoreset=True)

# --- Agent Color Definitions ---
# (Keep existing definitions)
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
    """Calculates the visible width of a string by removing ANSI escape codes."""
    return len(ansi_escape_pattern.sub('', text))

def pad_visual_width(text, width):
    """Pads a string with spaces to reach a desired visual width."""
    current_visual_width = get_visual_width(text)
    padding = max(0, width - current_visual_width)
    return text + ' ' * padding

# --- Rendering Function (Returns list of UNPADDED lines and max visual width) ---
def render(grid, agent_message_colors_map, coord_to_agent_id_map):
    """
    Renders a grid. Returns list of UNPADDED lines and the calculated max visual width.
    """
    try:
        # Basic validation
        if not isinstance(grid, (list, np.ndarray)) or not grid: return ["(Empty/invalid grid)"], 20
        grid_np = np.array(grid, dtype=object)
        if grid_np.ndim != 2: return [f"(Invalid grid dim {grid_np.ndim})"], 25
        height, width = grid_np.shape
        if height == 0 or width == 0: return ["(Empty HxW grid)"], 18
    except Exception as e:
        error_str = f"(Render Error: {e})"
        return [error_str], max(20, len(error_str))

    rendered_lines = []
    max_visual_width = 0

    # Column headers
    header_line = "   " + " ".join(str(i % 10) for i in range(width))
    rendered_lines.append(header_line)
    # max_visual_width updated within the loop now

    for i in range(height):
        row_str_prefix = f"{i:2d} "
        current_line = row_str_prefix
        for j in range(width):
            try: cell = grid_np[i, j]
            except IndexError: cell = '?'
            original_cell_char = str(cell) if cell is not None and str(cell).strip() != '' else '.'
            color_code = ""
            final_char = original_cell_char
            agent_id_at_coord = coord_to_agent_id_map.get((i, j))

            # Coloring Logic (same as before)
            if agent_id_at_coord and agent_id_at_coord in agent_message_colors_map:
                match = re.search(r'\d+$', agent_id_at_coord)
                final_char = match.group()[-1] if match else '?'
                message_color = agent_message_colors_map[agent_id_at_coord]
                if original_cell_char == 'a': color_code = message_color
                else: back_color, text_color = AGENT_GRID_COLORS_MAP.get(message_color, DEFAULT_GRID_AGENT_COLOR); color_code = back_color + text_color
            else:
                if original_cell_char == 'P': color_code = Back.WHITE + Fore.BLACK
                elif original_cell_char == 'Y': color_code = Back.CYAN + Fore.BLACK
                elif original_cell_char == 'W': color_code = Back.RED + Fore.WHITE
                elif original_cell_char == 'B': color_code = Back.YELLOW + Fore.BLACK
                elif original_cell_char == 'X': color_code = Back.MAGENTA + Fore.WHITE
                elif original_cell_char == 'A': color_code = GENERIC_AGENT_GRID_COLOR

            # Append character
            segment = final_char + " "
            if color_code: current_line += color_code + segment + Style.RESET_ALL
            else: current_line += segment

        current_line = current_line.rstrip()
        rendered_lines.append(current_line)

    # Calculate max visual width AFTER all lines are generated
    max_visual_width = 0
    for line in rendered_lines:
        max_visual_width = max(max_visual_width, get_visual_width(line))

    # Return UNPADDED lines and the calculated max width
    return rendered_lines, max_visual_width


# --- Main Execution Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay game logs with optional agent views horizontally, chunked.')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help=f'Log directory (default: {DEFAULT_LOG_DIR})')
    parser.add_argument('--time', type=float, default=DEFAULT_TIME, help=f'Time delay between steps (default: {DEFAULT_TIME}s)')
    parser.add_argument('--show-views', '-v', action='store_true', help='Display individual agent views horizontally.')
    parser.add_argument('--max-grids', type=int, default=DEFAULT_MAX_GRIDS_PER_ROW,
                        help=f'Max TOTAL grids (global + agent views) per horizontal row (default: {DEFAULT_MAX_GRIDS_PER_ROW}).')
    parser.add_argument('--debug', action='store_true', help='Print debug information about widths.')
    args = parser.parse_args()

    log_dir = args.log_dir
    TIME = args.time
    # show_agent_views = args.show_views
    show_agent_views = True
    max_grids_per_row = args.max_grids # Use the new name
    debug_mode = args.debug

    print(f"Log Dir: {log_dir}, Time: {TIME}s, Show Views: {show_agent_views}, Max Grids/Row: {max_grids_per_row}, Debug: {debug_mode}")
    print("-" * 30); time.sleep(0.5) # Shorter sleep

    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path): print(f"{Fore.RED}Error: Meta log not found: {meta_log_path}"); sys.exit(1)
    try:
        with open(meta_log_path) as f: meta = json.load(f)
    except Exception as e: print(f"{Fore.RED}Error reading meta log {meta_log_path}: {e}"); sys.exit(1)

    meta = sorted([(t, i) for t, i in meta.items()])

    for timestamp, info in meta:
        print(f"{Style.BRIGHT}Loading game: {timestamp}: {info}{Style.RESET_ALL}")
        game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')
        agent_log_path = os.path.join(log_dir, f'agent_log_{timestamp}.json')

        # Load Game Steps & Agent Data (simplified loading)
        game_steps = []; messages_by_round = defaultdict(dict); views_by_round = defaultdict(dict); all_agent_ids_in_log = set()
        try: # Combine loading for brevity
            if os.path.exists(game_log_path):
                with open(game_log_path, encoding='utf-8') as f: game_steps = json.load(f)
            else: print(f"{Fore.YELLOW}Warn: Game log missing: {game_log_path}"); continue
            if not game_steps: print(f"{Fore.YELLOW}Warn: Game log empty: {game_log_path}"); continue

            if os.path.exists(agent_log_path):
                with open(agent_log_path, encoding='utf-8') as f: agent_records = json.load(f)
                for record in agent_records:
                    if isinstance(record, dict) and 'round' in record and 'agent_id' in record:
                        try:
                            round_num_int = int(record['round']); agent_id = str(record['agent_id'])
                            all_agent_ids_in_log.add(agent_id)
                            if msg := record.get('message'): messages_by_round[round_num_int][agent_id] = msg
                            if view := record.get('view'):
                                if isinstance(view, list) and view: views_by_round[round_num_int][agent_id] = view
                        except (ValueError, TypeError): pass
            else: print(f"{Fore.YELLOW}Warn: Agent log missing: {agent_log_path}")
        except Exception as e: print(f"{Fore.RED}Err loading logs for {timestamp}: {e}"); continue


        # Assign Colors (simplified)
        def sort_key(agent_id): match = re.search(r'\d+$', agent_id); return int(match.group()) if match else agent_id
        try: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log), key=sort_key)
        except Exception: sorted_agent_ids_list = sorted(list(all_agent_ids_in_log))
        agent_message_colors_map = {}
        num_unique_colors = len(AGENT_MESSAGE_COLORS)
        if num_unique_colors > 0:
            for i, agent_id in enumerate(sorted_agent_ids_list): agent_message_colors_map[agent_id] = AGENT_MESSAGE_COLORS[i % num_unique_colors]

        # --- Replay Loop ---
        refresh = False
        last_total_lines_printed = 0
        num_agents_meta = info.get("num_agents", "N/A")

        for step_index, step in enumerate(game_steps):
            if not isinstance(step, dict) or 'round' not in step: continue
            round_num = step['round']
            try: round_num_int = int(round_num)
            except (ValueError, TypeError): continue
            grid_data = step.get('grid')
            if not grid_data: print(f"{Fore.YELLOW}Warn: Missing grid R{round_num}"); continue

            # Agent coords (CORRECTED BLOCK)
            coord_to_agent_id_map = {}; agent_ids_in_step = set()
            agents_list = step.get('agents'); agents_list_valid = isinstance(agents_list, list)
            if agents_list_valid:
                 for agent in agents_list:
                     if (isinstance(agent, dict) and all(k in agent for k in ('x','y','id'))):
                          try: # Add try-except for coord conversion
                              y = int(agent['y']) # Separate line
                              x = int(agent['x']) # Separate line
                              agent_id_str = str(agent['id'])
                              coord_to_agent_id_map[(y, x)] = agent_id_str
                              agent_ids_in_step.add(agent_id_str) # Separate line - FIX HERE
                          except (ValueError, TypeError):
                              # Optional: Add a warning if you want to know about malformed entries
                              # print(f"{Fore.YELLOW}Warn: Skipping agent with invalid coords/id: {agent}")
                              continue # Skip malformed agent entries


            # --- Refresh Screen ---
            if refresh:
                if last_total_lines_printed > 0: print(f'\033[{last_total_lines_printed}A\033[J', end='')
                else: print('\033[H\033[J', end='')
            else: refresh = True; time.sleep(0.1)

            current_lines_printed = 0

            # --- Print Round Info ---
            model_name = info.get("model", "N/A"); num_agents_actual = len(coord_to_agent_id_map)
            round_info_line = f'Timestamp: {timestamp} | Round: {round_num:<3} | Agents: {num_agents_actual:<2} (Meta: {num_agents_meta}) | Model: {model_name}'
            print(round_info_line); current_lines_printed += 1
            if not agents_list_valid: print(f"{Fore.YELLOW}Warn: 'agents' list missing/invalid R{round_num}."); current_lines_printed += 1

            # --- Prepare ALL Grid Data First ---
            all_rendered_grids = [] # List of (header, unpadded_lines_list, visual_width)

            # 1. Render Global Grid
            try:
                global_lines, global_width = render(grid_data, agent_message_colors_map, coord_to_agent_id_map)
                all_rendered_grids.append(("--- Global Map ---", global_lines, global_width))
            except Exception as e:
                error_line = f"{Fore.RED}(Err rendering global R{round_num}: {e})"
                all_rendered_grids.append((f"Global Error", [error_line], len(error_line)))

            # 2. Render Agent Views (if requested)
            agent_views_this_round = views_by_round.get(round_num_int, {})
            if show_agent_views and agent_views_this_round:
                agents_with_views = sorted(list(agent_ids_in_step & agent_views_this_round.keys()), key=sort_key)
                if not agents_with_views: agents_with_views = sorted([aid for aid in sorted_agent_ids_list if aid in agent_views_this_round], key=sort_key)

                # NO limit applied here yet, render all requested views
                for agent_id in agents_with_views:
                    view_grid = agent_views_this_round.get(agent_id)
                    if view_grid:
                        try:
                            view_lines, view_width = render(view_grid, agent_message_colors_map, {})
                            agent_color = agent_message_colors_map.get(agent_id, Fore.WHITE)
                            view_header = f"{agent_color}--- View {agent_id} ---{Style.RESET_ALL}"
                            all_rendered_grids.append((view_header, view_lines, view_width))
                        except Exception as e:
                             error_line = f"{Fore.RED}(Err rendering view {agent_id} R{round_num}: {e})"
                             all_rendered_grids.append((f"View {agent_id} Error", [error_line], max(25, len(error_line))))


            # --- Print Grids Horizontally IN CHUNKS ---
            if all_rendered_grids:
                num_grids_total = len(all_rendered_grids)
                # Use max() to ensure at least 1 grid per row if user enters 0 or negative
                effective_max_grids = max(1, max_grids_per_row)
                num_chunks = math.ceil(num_grids_total / effective_max_grids)


                for chunk_index in range(num_chunks):
                    start_index = chunk_index * effective_max_grids
                    end_index = start_index + effective_max_grids
                    current_chunk_data = all_rendered_grids[start_index:end_index]

                    if not current_chunk_data: continue

                    if chunk_index > 0: print(""); current_lines_printed += 1

                    max_height_chunk = 0
                    if current_chunk_data:
                        max_height_chunk = max(len(lines) for _, lines, _ in current_chunk_data if lines)

                    # Print Headers for this chunk
                    header_line = ""
                    for header, _, width in current_chunk_data:
                        padded_header = pad_visual_width(header, width)
                        header_line += padded_header + GRID_SEPARATOR
                    print(header_line.rstrip())
                    current_lines_printed += 1

                    # Print Grid Rows for this chunk
                    for i in range(max_height_chunk):
                        combined_line = ""
                        for idx, (_, lines, width) in enumerate(current_chunk_data):
                            actual_grid_index = start_index + idx
                            if i < len(lines):
                                line_segment = lines[i]
                                padded_segment = pad_visual_width(line_segment, width)
                            else:
                                padded_segment = ' ' * width

                            if debug_mode and i < len(lines):
                                 print(f"[D Chk{chunk_index} Grid{actual_grid_index}] L{i} W={width} Raw='{line_segment.strip()}' VisW={get_visual_width(line_segment)} PadW={get_visual_width(padded_segment)}")
                            elif debug_mode:
                                 print(f"[D Chk{chunk_index} Grid{actual_grid_index}] L{i} W={width} Padding Only")

                            combined_line += padded_segment + GRID_SEPARATOR
                        print(combined_line.rstrip())
                        current_lines_printed += 1
            else:
                print("(No grids to display)"); current_lines_printed += 1


            # --- Display Agent Messages (Below ALL Grid Chunks) ---
            agent_messages_this_round = messages_by_round.get(round_num_int, {})
            if agent_messages_this_round:
                print("\n--- Agent Messages ---"); current_lines_printed += 2
                agents_to_display_msg = sorted(list(agent_ids_in_step & agent_messages_this_round.keys()), key=sort_key)
                if not agents_to_display_msg: agents_to_display_msg = sorted([aid for aid in sorted_agent_ids_list if aid in agent_messages_this_round], key=sort_key)
                msg_count = 0
                for agent_id in agents_to_display_msg:
                    message = agent_messages_this_round.get(agent_id, "")
                    cleaned_message = ' '.join(str(message).split())
                    if not cleaned_message: continue
                    agent_color = agent_message_colors_map.get(agent_id, Fore.WHITE)
                    formatted_message = f"{agent_color}{agent_id}:{Style.RESET_ALL} {cleaned_message}"
                    print(formatted_message); current_lines_printed += 1; msg_count += 1
                if msg_count == 0: print("(No valid messages)"); current_lines_printed += 1


            # --- End of Frame ---
            last_total_lines_printed = current_lines_printed
            sys.stdout.flush()
            time.sleep(max(0.05, TIME))


        print("\n" + "="*50 + "\n"); time.sleep(max(0.5, TIME))

    print("Replay finished.")