# -*- coding: utf-8 -*-
# v1.12: Refactored into main.py + gui_utils package (with log loading fix)

import pygame
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict
import re
import argparse
import math
# import random # No longer needed here, moved to drawing if specific layouts needed it
import glob # To find log files

# --- Import GUI utilities ---
from gui_utils import initialization, layout, drawing, slider, constants as C # Use alias

# --- Configuration ---
DEFAULT_LOG_DIR = 'logs' # Default directory name
DEFAULT_TIME_MS = 200 # Base Delay in milliseconds (for 1.0x speed)
WAIT_BETWEEN_GAMES_S = 2.0 # Wait time in seconds between sequential games
SPEED_MULTIPLIERS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] # Added slower/faster options
# Ensure 1.0x is present for default index calculation
if 1.0 not in SPEED_MULTIPLIERS:
    SPEED_MULTIPLIERS.append(1.0)
    SPEED_MULTIPLIERS.sort()
DEFAULT_SPEED_INDEX = SPEED_MULTIPLIERS.index(1.0)
WIDTH, HEIGHT = 1700, 950 # Slightly larger screen dimensions
MAX_MESSAGES_DISPLAY = 50 # Limit lines in the message panel

# --- Regex ---
MENTION_REGEX = re.compile(r'(?:agent|Agent|@|#)\s?\{?(\d+)\}?')
LOG_IDENTIFIER_REGEX = re.compile(r'(?:game|agent)_log_([\w-]+)\.json')

# --- Utility Functions (Non-GUI) ---
def parse_mentions(message, sender_id, all_agent_ids):
    """Extracts mentioned agent IDs from a message string."""
    mentions = []
    if not message or not isinstance(message, str):
        return mentions
    try:
        # Find all numbers potentially representing agent IDs
        found_ids = MENTION_REGEX.findall(message)
        for mentioned_num_str in found_ids:
            # Assume standard Agent_N format
            mentioned_id = f"Agent_{mentioned_num_str}"
            # Add if it's a valid agent ID in the known set and not the sender mentioning themselves
            if mentioned_id != sender_id and mentioned_id in all_agent_ids:
                 mentions.append((sender_id, mentioned_id))
    except Exception as e:
        print(f"Warn: Error parsing mentions in message from {sender_id}: {e}")
    # Return unique pairs
    return list(set(mentions))

def get_agent_color_map(agent_ids):
    """Assigns colors to agent IDs based on their number."""
    color_map = {}
    # Sort agents by their numerical ID for consistent color assignment
    def sort_key(agent_id):
        match = re.search(r'\d+$', agent_id)
        return int(match.group()) if match else float('inf') # Put non-matching last
    sorted_ids = sorted(list(agent_ids), key=sort_key)

    num_colors = len(C.PYGAME_AGENT_COLORS)
    if num_colors > 0:
        for i, agent_id in enumerate(sorted_ids):
            color_map[agent_id] = C.PYGAME_AGENT_COLORS[i % num_colors] # Cycle through colors
    else:
        # Fallback if no colors are defined (shouldn't happen with constants.py)
        print("Warning: No agent colors defined in constants.")
        for agent_id in sorted_ids:
            color_map[agent_id] = C.WHITE # Default to white
    return color_map

# --- Log Discovery and Selection ---
def find_available_games(log_dir):
    """Finds valid game/agent log pairs in the specified directory."""
    identifiers = set()
    abs_log_dir = os.path.abspath(log_dir)
    print(f"Searching for logs in: {abs_log_dir}")

    if not os.path.isdir(abs_log_dir):
        print(f"Error: Log directory not found: {abs_log_dir}")
        return []

    game_files = glob.glob(os.path.join(abs_log_dir, 'game_log_*.json'))

    for game_file in game_files:
        match = LOG_IDENTIFIER_REGEX.search(os.path.basename(game_file))
        if match:
            identifier = match.group(1)
            agent_file = os.path.join(abs_log_dir, f'agent_log_{identifier}.json')
            meta_file = os.path.join(abs_log_dir, 'meta_log.json') # Meta log is optional

            # Require both game and agent log to exist
            if os.path.exists(agent_file):
                 identifiers.add(identifier)
                 meta_exists = os.path.exists(meta_file)
                 print(f"  Found: Game '{identifier}' (game+agent{' +meta' if meta_exists else ''})")
            else:
                 print(f"  Warn: Skipping game log '{identifier}', missing required agent log: {agent_file}")
        else:
             # This might indicate a file that looks like a log but doesn't match pattern
             print(f"  Debug: Filename pattern mismatch for {os.path.basename(game_file)}")

    if not identifiers:
        print("No valid game/agent log pairs found.")
        return []

    # Sort identifiers, trying to parse numbers for more natural sorting
    def sort_key(id_str):
        parts = id_str.split('-Thread-') # Handle common naming convention
        t = 0
        th = 0
        try:
            # Try to extract numbers, default to 0 if parsing fails
            if parts[0].isdigit(): t = int(parts[0])
            if len(parts) > 1 and parts[1].isdigit(): th = int(parts[1])
        except ValueError:
            pass # Keep defaults if not parsable
        # Sort primarily by the first number, then by the thread number
        # If no 'Thread' part, use 0 for thread number
        return (t, th, id_str) # Include original string as tertiary key for stability
    return sorted(list(identifiers), key=sort_key)


def display_selection_menu(game_identifiers):
    """Displays the game selection menu in the console and gets user choice."""
    print("\nAvailable Game Logs:")
    if not game_identifiers:
        print(f"No valid log pairs found.")
        return None # Indicate no choice possible

    for i, identifier in enumerate(game_identifiers):
        print(f"  {i + 1}: {identifier}")

    print(f"\n  {len(game_identifiers) + 1}: Play All Sequentially")
    print("  0: Quit")

    while True:
        try:
            choice = input(f"Enter choice (0-{len(game_identifiers)+1}): ")
            choice_num = int(choice)
            if 0 <= choice_num <= len(game_identifiers) + 1:
                return choice_num # Return the valid choice number
            else:
                print("Invalid choice number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
             print("\nInput stream closed. Exiting.")
             return 0 # Treat EOF as quitting


# --- Main Replay Logic ---

def run_replay(screen, clock, fonts, log_dir, game_identifier, base_time_delay_ms):
    """Runs the replay visualization for a SINGLE specified game identifier."""
    print(f"\n--- Loading Replay Data for: {game_identifier} ---")
    abs_log_dir = os.path.abspath(log_dir)
    game_log_path = os.path.join(abs_log_dir, f'game_log_{game_identifier}.json')
    agent_log_path = os.path.join(abs_log_dir, f'agent_log_{game_identifier}.json')
    meta_log_path = os.path.join(abs_log_dir, 'meta_log.json') # Optional

    game_steps = [] # List to store data for each round [{grid, agents, score, ...}]
    messages_by_round = defaultdict(dict) # {round_idx: {agent_id: message}}
    all_agent_ids_in_log = set() # Collect all unique agent IDs for THIS game
    score_history = [] # List of (round_index, score) tuples for THIS game
    model_name = "N/A"
    load_success = True

    # --- Load Data ---
    try:
        # --- Load optional metadata ---
        if os.path.exists(meta_log_path):
            try:
                with open(meta_log_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    # Find the metadata specific to this game identifier
                    model_name = meta_data.get(game_identifier, {}).get("model", "N/A")
                print(f"  Loaded metadata: Model = {model_name}")
            except json.JSONDecodeError as e:
                print(f"Warn: Meta log '{meta_log_path}' decode error: {e}")
            except Exception as e:
                print(f"Warn: Error reading meta log: {e}")
        else:
            print(f"Info: Optional meta log missing '{meta_log_path}'.")

        # --- Load required game log ---
        if not os.path.exists(game_log_path):
            print(f"Error: Required game log missing: {game_log_path}")
            return False # Indicate failure to run this replay
        with open(game_log_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
            # Assume game_data is a list of steps/rounds
            if not isinstance(game_data, list):
                raise TypeError("Game log is not a list of rounds/steps.")

            # Clear data structures for this game replay
            game_steps.clear()
            score_history.clear()
            all_agent_ids_in_log.clear()
            messages_by_round.clear() # Clear messages too

            for round_index, step_data in enumerate(game_data):
                if not isinstance(step_data, dict):
                     print(f"Warn: Skipping invalid step data at index {round_index} (not a dict).")
                     continue

                # --- ADJUSTED DATA EXTRACTION (Matches original script logic) ---
                grid_state = step_data.get('grid')      # Expect 'grid' key
                agents_list = step_data.get('agents')   # Expect 'agents' key with a LIST
                score = step_data.get('score', 0)       # Score handling

                # Check if the essential keys are present and 'agents' is a list
                if grid_state is None or not isinstance(agents_list, list):
                    print(f"Warn: Skipping round {round_index+1}, missing 'grid' key or 'agents' is not a list.")
                    continue

                # --- Process the 'agents' list ---
                processed_agent_positions = {} # Stores {agent_id: (row, col)} for this step
                current_agent_ids_in_step = set()
                for agent_dict in agents_list:
                    # Check if item in list is a dict with required keys
                    if isinstance(agent_dict, dict) and all(k in agent_dict for k in ('x', 'y', 'id')):
                        try:
                            # Original script used 'y', 'x' for (row, col)
                            r, c = int(agent_dict['y']), int(agent_dict['x'])
                            agent_id = str(agent_dict['id'])
                            processed_agent_positions[agent_id] = (r, c) # Store as {id: (r,c)} tuple
                            current_agent_ids_in_step.add(agent_id)
                        except (ValueError, TypeError, KeyError) as e:
                            print(f"Warn: Skipping invalid agent data in round {round_index+1}: {agent_dict} ({e})")
                            # Continue to next agent in the list
                    else:
                        # Handle cases where item in agents_list is not the expected dict format
                        print(f"Warn: Unexpected item format in 'agents' list for round {round_index+1}: {agent_dict}")

                # Update the set of all agents seen in this game log
                all_agent_ids_in_log.update(current_agent_ids_in_step)

                # Append the correctly processed step data to game_steps list
                game_steps.append({
                    'round': round_index,
                    'grid': grid_state,
                    'agents': processed_agent_positions, # Store the processed DICT {id: (r,c)}
                    'score': score
                })

                # Append score to history if it's numeric
                numeric_score = None
                if isinstance(score, (int, float)):
                    numeric_score = float(score)
                elif isinstance(score, str): # Try converting string scores
                    try: numeric_score = float(score)
                    except ValueError: pass # Ignore non-numeric string scores

                if numeric_score is not None:
                    score_history.append((round_index, numeric_score))
                # --- END ADJUSTED DATA EXTRACTION ---

        print(f"  Loaded {len(game_steps)} rounds from game log.") # This count should now be accurate
        if not game_steps and game_data: # Check if loading failed despite file having data
             print("  Warning: Game log file contained data, but no valid rounds could be parsed.")


        # --- Load required agent log ---
        if not os.path.exists(agent_log_path):
            print(f"Error: Required agent log missing: {agent_log_path}")
            # Decide if agent log is truly required. If not, maybe just warn.
            # For now, assume messages/mentions are important, so fail if missing.
            return False # Indicate failure
        with open(agent_log_path, 'r', encoding='utf-8') as f:
            agent_data = json.load(f)
            # Assume agent_data is a list of message entries
            if not isinstance(agent_data, list):
                raise TypeError("Agent log is not a list of message entries.")

            num_messages_loaded = 0
            for msg_entry in agent_data:
                if not isinstance(msg_entry, dict):
                    print(f"Warn: Skipping invalid message entry (not a dict).")
                    continue
                # Use .get with default None to avoid KeyErrors
                round_idx_str = msg_entry.get('round')
                agent_id = msg_entry.get('agent_id')
                message = msg_entry.get('message')

                # Check if essential fields are missing
                if round_idx_str is None or agent_id is None or message is None:
                    print(f"Warn: Skipping message entry, missing 'round', 'agent_id', or 'message': {msg_entry}")
                    continue

                # Convert round index to integer, handle potential errors
                try:
                    round_idx = int(round_idx_str)
                except (ValueError, TypeError):
                     print(f"Warn: Skipping message entry, invalid round format: {round_idx_str}")
                     continue

                # Ensure agent_id is a string
                agent_id = str(agent_id)

                # Store message, indexed by round and then agent
                messages_by_round[round_idx][agent_id] = str(message) # Ensure message is string
                # Ensure agent ID from message log is also tracked
                all_agent_ids_in_log.add(agent_id)
                num_messages_loaded += 1
        print(f"  Loaded {num_messages_loaded} messages for {len(messages_by_round)} unique rounds from agent log.")
        print(f"  Total unique agents identified across logs: {len(all_agent_ids_in_log)}")

    except json.JSONDecodeError as e:
        print(f"Fatal Error: Failed to decode JSON in log files: {e}")
        load_success = False
    except FileNotFoundError as e:
        # Specific file not found errors printed above
        print(f"Fatal Error: A required log file was not found.")
        load_success = False
    except TypeError as e:
         print(f"Fatal Error: Unexpected data structure encountered in log file: {e}")
         load_success = False
    except Exception as e:
        print(f"Fatal Error: An unexpected error occurred during log loading: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for unexpected errors
        load_success = False

    if not load_success or not game_steps:
        print("--- Replay loading failed or no valid rounds found. Aborting replay. ---")
        # Optional: Add a small delay so user can read the error
        time.sleep(2)
        return False # Indicate failure

    # --- Post-Load Setup ---
    agent_color_map = get_agent_color_map(all_agent_ids_in_log)
    ui_layout = layout.calculate_grid_layout(WIDTH, HEIGHT)

    # --- Pygame Replay Loop Initialization ---
    current_round_index = 0
    max_round_index = len(game_steps) - 1 # Max valid index for game_steps
    paused = False
    running = True
    dragging_slider = False
    current_speed_index = DEFAULT_SPEED_INDEX
    current_speed_multiplier = SPEED_MULTIPLIERS[current_speed_index]
    time_delay_ms = base_time_delay_ms / current_speed_multiplier if current_speed_multiplier > 0 else float('inf')
    last_update_time = time.time() * 1000 # Use milliseconds for comparison
    handle_rect = None # Store slider handle rect for collision

    # --- Main Replay Loop ---
    print("--- Starting Replay ---")
    print("Controls: SPACE=Pause/Play, RIGHT/LEFT=Step (when paused), ESC=Quit")
    while running: # Loop continues as long as running is True
        current_time_ms = time.time() * 1000

        # --- Time-based Round Progression ---
        # Check if we should advance the round index
        if not paused and current_round_index < max_round_index and (current_time_ms - last_update_time >= time_delay_ms):
            current_round_index += 1
            last_update_time = current_time_ms
        elif not paused and current_round_index >= max_round_index:
             # Reached the end of the replay naturally
             print("--- Replay Finished ---")
             running = False # Set running to False to exit the loop
             # No break needed here, the loop condition will handle it

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("--- Replay Quit by User (Window Close) ---")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Playing")
                    # Reset update timer when resuming to avoid sudden jump
                    if not paused: last_update_time = current_time_ms
                elif event.key == pygame.K_RIGHT: # Step forward
                    current_round_index = min(current_round_index + 1, max_round_index)
                    paused = True # Ensure pause when manually stepping
                    print(f"Stepped forward to round {current_round_index + 1}/{max_round_index + 1}")
                elif event.key == pygame.K_LEFT: # Step backward
                    current_round_index = max(current_round_index - 1, 0)
                    paused = True # Ensure pause when manually stepping
                    print(f"Stepped back to round {current_round_index + 1}/{max_round_index + 1}")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    print("--- Replay Quit by User (ESC) ---")
            # Slider Event Handling
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and handle_rect and handle_rect.collidepoint(event.pos):
                     dragging_slider = True
                # Allow clicking directly on track
                elif event.button == 1 and ui_layout['speed_slider'].collidepoint(event.pos):
                    new_index = slider.get_slider_index_from_pos(ui_layout['speed_slider'], len(SPEED_MULTIPLIERS), event.pos[0])
                    if new_index != current_speed_index:
                        current_speed_index = new_index
                        current_speed_multiplier = SPEED_MULTIPLIERS[current_speed_index]
                        time_delay_ms = base_time_delay_ms / current_speed_multiplier if current_speed_multiplier > 0 else float('inf')
                        print(f"Speed changed to {current_speed_multiplier}x")
                    # Start dragging even if clicked on track
                    dragging_slider = True # Allows dragging after initial click
            if event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1:
                     dragging_slider = False
            if event.type == pygame.MOUSEMOTION:
                 if dragging_slider:
                     # Update slider position based on mouse X
                     new_index = slider.get_slider_index_from_pos(ui_layout['speed_slider'], len(SPEED_MULTIPLIERS), event.pos[0])
                     if new_index != current_speed_index:
                         current_speed_index = new_index
                         current_speed_multiplier = SPEED_MULTIPLIERS[current_speed_index]
                         time_delay_ms = base_time_delay_ms / current_speed_multiplier if current_speed_multiplier > 0 else float('inf')
                         # No print here, avoid spamming console during drag
                         # Consider updating a visual speed indicator instead if needed

        # --- Prepare Data for Current Frame ---
        # Use the current_round_index (which is managed by time/stepping logic)
        # No need to clamp here, as the progression logic prevents going out of bounds
        # and stepping logic also clamps.
        if 0 <= current_round_index <= max_round_index:
            current_step = game_steps[current_round_index]
        else:
            # This case should ideally not be reached if loop logic is correct
            # but as a fallback, use the last valid step
            current_step = game_steps[max_round_index]
            print(f"Warn: current_round_index ({current_round_index}) was out of bounds [0, {max_round_index}]. Using last valid step.")

        grid_data = current_step['grid']
        agent_positions_this_step = current_step['agents'] # {agent_id: (r, c)}
        current_score = current_step['score']

        # Map agent positions for grid drawing: {(r, c): agent_id}
        agent_coords_map = {pos: aid for aid, pos in agent_positions_this_step.items()}

        # Get messages for this round index
        messages_this_round = messages_by_round.get(current_round_index, {}) # Get dict for the round, or empty dict

        # Parse mentions from this round's messages
        mentions_this_round = []
        for sender_id, message in messages_this_round.items():
            # Only parse mentions for agents present in the log (all_agent_ids_in_log)
            if sender_id in all_agent_ids_in_log:
                mentions_this_round.extend(parse_mentions(message, sender_id, all_agent_ids_in_log))
        mentions_this_round = list(set(mentions_this_round)) # Ensure uniqueness

        # --- Drawing ---
        screen.fill(C.BLACK) # Clear screen

        # Draw Grid and get agent cell center coordinates
        agent_cell_centers = drawing.draw_grid(
            surface=screen,
            grid_data=grid_data,
            agent_coords_map=agent_coords_map,
            agent_color_map=agent_color_map,
            target_rect=ui_layout['global_grid'],
            cell_font=fonts['grid_cell']
        )

        # Draw Mention Lines on the Grid
        drawing.draw_mention_lines(
            surface=screen,
            mentions_list=mentions_this_round,
            messages_this_round=messages_this_round,
            agent_cell_centers=agent_cell_centers,
            msg_font=fonts['small']
        )

        # Draw Mention Graph Panel
        # Consider showing mentions aggregated over time vs just current round?
        # For now, showing current round's mentions.
        drawing.draw_mention_graph(
            surface=screen,
            agent_ids=all_agent_ids_in_log, # Show all agents ever seen in graph
            mentions_list=mentions_this_round, # Edges based on current round
            target_rect=ui_layout['mention_graph'],
            font=fonts['small']
        )

        # Draw Score Curve Panel (only up to the current round)
        scores_to_plot = [sh for sh in score_history if sh[0] <= current_round_index]
        drawing.draw_score_curve(
            surface=screen,
            score_history=scores_to_plot,
            target_rect=ui_layout['score_curve'],
            font=fonts['small']
        )

        # Draw Messages Panel (show messages for current round)
        drawing.draw_messages(
            surface=screen,
            messages_dict=messages_this_round,
            agent_color_map=agent_color_map,
            target_rect=ui_layout['messages'],
            font=fonts['small'],
            max_lines=MAX_MESSAGES_DISPLAY
        )

        # Draw Speed Slider
        handle_rect = drawing.draw_speed_slider(
            surface=screen,
            slider_rect=ui_layout['speed_slider'],
            font=fonts['small'],
            current_index=current_speed_index,
            speed_multipliers=SPEED_MULTIPLIERS,
            is_dragging=dragging_slider
        )

        # --- Draw Status Text ---
        status_font = fonts['medium']
        # Ensure score is formatted nicely
        score_display = f"{current_score:.2f}" if isinstance(current_score, (int, float)) else str(current_score)
        status_text = f"Game: {game_identifier} | Model: {model_name} | Round: {current_round_index + 1}/{max_round_index + 1} | Score: {score_display} | Speed: {current_speed_multiplier}x {'[PAUSED]' if paused else ''}"
        status_surf = status_font.render(status_text, True, C.WHITE)
        # Position status text near the bottom left
        status_rect = status_surf.get_rect(bottomleft=(10, HEIGHT - 10))
        screen.blit(status_surf, status_rect)

        # --- Update Display ---
        pygame.display.flip()

        # --- Frame Limiting ---
        clock.tick(60) # Limit to 60 FPS

    # --- End of Replay Loop (while running:) ---
    # Determine if the loop ended because the user quit or because it finished
    user_quit = not (current_round_index >= max_round_index and not paused)

    # If the loop finished because replay ended naturally, wait briefly
    if not user_quit:
        time.sleep(1.0) # Brief pause after natural finish

    # Return True if replay finished naturally, False if user quit (ESC or window close)
    return not user_quit


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pygame visualization for agent simulation logs.")
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                        help=f"Directory containing the log files (default: {DEFAULT_LOG_DIR})")
    parser.add_argument('--game_id', type=str, default=None,
                        help="Specific game identifier to replay (optional). If not provided, a menu will be shown.")
    parser.add_argument('--speed', type=float, default=1.0,
                        help=f"Initial speed multiplier (e.g., 0.5, 1.0, 2.0). Must be one of {SPEED_MULTIPLIERS}.")

    args = parser.parse_args()

    log_directory = args.log_dir

    # Validate initial speed setting
    try:
        initial_speed_index = SPEED_MULTIPLIERS.index(args.speed)
        DEFAULT_SPEED_INDEX = initial_speed_index # Override default if valid speed provided
    except ValueError:
        print(f"Warning: Provided speed {args.speed}x is not available in {SPEED_MULTIPLIERS}. Using default {SPEED_MULTIPLIERS[DEFAULT_SPEED_INDEX]}x.")
        # Keep the original DEFAULT_SPEED_INDEX

    available_games = find_available_games(log_directory)

    if not available_games:
        print(f"\nNo replayable games found in '{log_directory}'. Exiting.")
        sys.exit(1)

    selected_game_identifiers = []
    if args.game_id:
        if args.game_id in available_games:
            print(f"Replaying specified game: {args.game_id}")
            selected_game_identifiers = [args.game_id]
        else:
            print(f"Error: Specified game ID '{args.game_id}' not found in available logs. Please choose from the list:")
            # Fall through to menu selection
            args.game_id = None # Clear invalid ID
            # Add a small pause so user can see the error before menu
            time.sleep(1)
    # else: # No need for this else if we fall through
    #     print("\nNo specific game ID provided via arguments.")


    if not selected_game_identifiers: # If no valid game_id from args, show menu
        choice = display_selection_menu(available_games)

        if choice == 0:
            print("Exiting.")
            sys.exit(0)
        elif choice == len(available_games) + 1:
            print("Selected: Play All Sequentially")
            selected_game_identifiers = available_games
        elif 1 <= choice <= len(available_games):
            selected_identifier = available_games[choice - 1]
            print(f"Selected: {selected_identifier}")
            selected_game_identifiers = [selected_identifier]
        else:
            # Should not happen with input validation, but good practice
            print("Invalid selection state. Exiting.")
            sys.exit(1)

    # --- Initialize Pygame ---
    # Add basic error handling for Pygame setup itself
    try:
        screen, clock, fonts = initialization.setup_pygame(WIDTH, HEIGHT, "Agent Replay Visualizer")
    except SystemExit: # Catch sys.exit called by setup_pygame on failure
        print("Exiting due to Pygame initialization failure.")
        sys.exit(1) # Ensure exit code indicates error
    except Exception as e:
        print(f"An unexpected error occurred during Pygame setup: {e}")
        sys.exit(1)


    # --- Run Replay(s) ---
    continue_sequence = True # Flag to control sequential play loop
    for i, game_id in enumerate(selected_game_identifiers):
        if not continue_sequence: # Check if sequence was aborted in previous iteration
            break

        print(f"\n--- Starting Replay {i+1}/{len(selected_game_identifiers)}: {game_id} ---")

        # Check if Pygame display is still active before running replay
        if not pygame.display.get_init() or not pygame.display.get_active():
             print("Error: Pygame display is not active. Aborting sequence.")
             continue_sequence = False
             break

        # run_replay returns True if finished naturally, False if user quit
        replay_finished_naturally = run_replay(screen, clock, fonts, log_directory, game_id, DEFAULT_TIME_MS)

        if not replay_finished_naturally:
             # User quit the replay (ESC or window close)
             print("User quit detected during replay. Stopping sequence.")
             continue_sequence = False
             break # Exit the loop over game identifiers

        # --- Wait Between Games (Only if sequence active, replay finished OK, and not the last game) ---
        if continue_sequence and i < len(selected_game_identifiers) - 1:
            print(f"\nFinished {game_id}. Waiting {WAIT_BETWEEN_GAMES_S} seconds before next replay...")
            start_wait = time.time()
            wait_end_time = start_wait + WAIT_BETWEEN_GAMES_S

            while time.time() < wait_end_time:
                 if not continue_sequence: break # Check flag again in case of quit event

                 # Keep pygame responsive during wait to handle quit events
                 for event in pygame.event.get():
                      if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                           continue_sequence = False
                           print("Quit during wait period.")
                           break
                 if not continue_sequence: break # Exit wait loop if quit detected

                 # Draw a waiting message
                 try:
                    if pygame.display.get_init() and pygame.display.get_active():
                        screen.fill(C.BLACK)
                        next_game_index = i + 1
                        wait_text = f"Next: {selected_game_identifiers[next_game_index]} ({next_game_index + 1}/{len(selected_game_identifiers)})"
                        wait_surf = fonts['medium'].render(wait_text, True, C.GRAY)
                        screen.blit(wait_surf, wait_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
                        time_left = max(0, wait_end_time - time.time())
                        time_surf = fonts['small'].render(f"Starting in {time_left:.1f}s... (ESC to cancel)", True, C.GRAY)
                        screen.blit(time_surf, time_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20)))
                        pygame.display.flip()
                    else:
                        print("Display lost during wait. Aborting sequence.")
                        continue_sequence = False; break
                 except Exception as draw_wait_e: print(f"Warn: Could not draw wait screen: {draw_wait_e}")


                 pygame.time.wait(50) # Small wait to prevent busy-looping CPU

            if not continue_sequence:
                print("Aborting sequence after wait period.")
                break # Exit outer for loop (over game identifiers)

    # --- Cleanup ---
    print("\nCleaning up Pygame...")
    pygame.quit() # Quit Pygame properly
    print("Exiting program.")
    sys.exit(0) # Explicitly exit with success code