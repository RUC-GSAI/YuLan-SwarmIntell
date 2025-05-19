import numpy as np
import json
import os
import sys
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import re
from utils.load_fonts import *

DEFAULT_LOG_DIR = 'experiment_v01'
os.makedirs('./figs',exist_ok=True)
DEFAULT_OUTPUT_FILE = 'figs/model_score_trends.pdf'
DEFAULT_PAD_SCORE_UP_TO_ROUND = 100

MODEL_NAME_MAPPING = {
    'DeepSeek-V3': 'deepseek-v3',
    'claude-3-5-haiku-20241022': 'claude-3.5-haiku',
    'claude-3-7-sonnet-20250219': 'claude-3.7-sonnet',
    'deepseek-ai/DeepSeek-R1': 'deepseek-r1',
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'gpt-4.1': 'gpt-4.1',                   
    'gpt-4.1-mini': 'gpt-4.1-mini',         
    'gpt-4o': 'gpt-4o',                     
    'o3-mini': 'o3-mini',                   
    'o4-mini': 'o4-mini'   ,                 
    'Meta-Llama-3.1-70B-Instruct': 'llama-3.1-70b',
    'meta-llama/llama-4-scout': 'llama-4-scout',
    'qwen/qwq-32b':'qwq-32b'
}

NATURE_COLORS = [
    '#4878D0',  # Medium blue - primary
    '#EE854A',  # Burnt orange
    '#6ACC64',  # Medium green
    '#D65F5F',  # Medium red
    '#956CB4',  # Medium purple
    '#8C613C',  # Brown
    '#DC7EC0',  # Medium pink
    '#82C6E2',  # Sky blue
    '#D5BB67',  # Gold/khaki
    '#5DA5DA',  # Light blue
    '#60BD68',  # Brighter green
    '#B2912F',  # Dark gold
    '#B276B2',  # Light purple
    '#DECF3F',  # Yellow
    '#F15854',  # Bright red
    '#4D4D4D'   # Dark gray
]


EXPERIMENT_NAME_MAPPING = {
    'experiment_v01': 'Flocking',
    'experiment_v02': 'Pursuit',
    'experiment_v03': 'Synchronize',
    'experiment_v04': 'Foraging',
    'experiment_v05': 'Transport'
}


# Helper function to get the display name
def get_display_name(raw_name):
    """Returns the mapped display name or the raw name if no mapping exists."""
    return MODEL_NAME_MAPPING.get(raw_name, raw_name) # Fallback to raw_name

def parse_game_logs_for_scores(log_dir, pad_score_up_to_round): # <<< ADDED ARG
    """
    Parses game logs to extract scores per round for each model across multiple games.
    Validates that scores within each game are non-decreasing.
    Pads the final score of games ending early up to pad_score_up_to_round.
    Applies score adjustment for specified models.

    Args:
        log_dir (str): The path to the log directory.
        pad_score_up_to_round (int): The round number to pad scores up to.

    Returns:
        tuple: A tuple containing:
            - scores_data (dict): A nested dictionary:
              { raw_model_name: { round_number: [score_game1, score_game2, ...] } }
            - max_round (int): The maximum round number encountered across all games
                               (considering padding).
            - processed_games (int): Count of successfully processed games.
            - failed_games (int): Count of games that failed processing.
        Returns (None, -1, 0, 0) if meta_log is not found or unreadable.
        Raises ValueError if a game log contains decreasing scores.
    """
    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path):
        print(f"Error: Meta log not found: {meta_log_path}", file=sys.stderr)
        return None, -1, 0, 0
    try:
        with open(meta_log_path) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Error reading meta log {meta_log_path}: {e}", file=sys.stderr)
        return None, -1, 0, 0

    scores_data = defaultdict(lambda: defaultdict(list))
    max_round = -1 # Tracks max round across all logs, including padding target
    processed_games = 0
    failed_games = 0

    print(f"Analyzing logs in directory: {log_dir}")
    print(f"Padding scores up to round: {pad_score_up_to_round}") # <<< INFO

    for timestamp, info in meta.items():
        raw_model_name = info.get("model")
        if not raw_model_name:
            print(f"Warning: Skipping entry for timestamp {timestamp} due to missing 'model' key.", file=sys.stderr)
            failed_games += 1
            continue

        game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')

        if not os.path.exists(game_log_path):
            print(f"Warning: Game log file not found: {game_log_path} (referenced in meta_log).", file=sys.stderr)
            failed_games += 1
            continue

        try:
            with open(game_log_path, encoding='utf-8') as f:
                game_steps = json.load(f)

            if not game_steps:
                print(f"Warning: Empty game log file: {game_log_path}. Skipping.", file=sys.stderr)
                failed_games += 1
                continue

            #  Collect scores for the current game 
            current_game_round_scores = [] # Store as (round_num_int, score_float, step_data)
            max_round_in_this_game = -1
            game_had_valid_step = False

            for step in game_steps:
                if not isinstance(step, dict):
                    continue # Skip invalid step format

                round_num = step.get('round')
                score = step.get('score')

                try:
                    if round_num is None: raise ValueError("Missing round")
                    round_num_int = int(round_num)

                    if score is None: raise ValueError("Missing score")
                    score_float = float(score)

                    # Store temporary data for validation and finding max round/score later
                    current_game_round_scores.append((round_num_int, score_float, step))

                    # Track max round *observed* in this specific game
                    max_round_in_this_game = max(max_round_in_this_game, round_num_int)
                    game_had_valid_step = True

                except (ValueError, TypeError) as e:
                    # print(f"Debug: Skipping invalid step in {game_log_path}: {step} due to {e}", file=sys.stderr)
                    pass # Silently skip steps with invalid round/score

            #  Validate, Pad, and Store scores if the game had valid steps 
            if game_had_valid_step:
                # 1. Sort the collected steps by round number
                current_game_round_scores.sort(key=lambda x: x[0])

                # 2. Perform non-decreasing check
                last_score_validation = -float('inf') # Initialize for validation
                last_round_validation = -1
                offending_step = None

                for r_val, current_score_val, step_data in current_game_round_scores:
                    if current_score_val < last_score_validation:
                        offending_step = step_data
                        error_msg = (
                            f"Validation Error: Score decreased in game log {game_log_path}.\n"
                            f"  Model: {raw_model_name} (Scores might be adjusted if this is '{MODEL_TO_ADJUST_SCORE}')\n"
                            f"  Round {r_val}: Score = {current_score_val}\n"
                            f"  Previous Round {last_round_validation}: Score = {last_score_validation}\n"
                            f"  Offending Step Data: {json.dumps(offending_step)}"
                        )
                        raise ValueError(error_msg)
                    last_score_validation = current_score_val
                    last_round_validation = r_val

                # 3. If check passed, add validated scores to the main data structure
                #    and perform padding if needed.
                last_round_in_game_final = -1
                last_score_in_game_final = np.nan # Use NaN if no valid score found

                for r, score_val, _ in current_game_round_scores:
                    scores_data[raw_model_name][r].append(score_val)
                    last_round_in_game_final = r
                    last_score_in_game_final = score_val
                    # Update global max_round based on observed rounds
                    max_round = max(max_round, r)

                # 4.  Apply Padding 
                if last_round_in_game_final >= 0 and last_round_in_game_final < pad_score_up_to_round:
                    # print(f"Debug: Padding game {timestamp} for model {raw_model_name} from round {last_round_in_game_final + 1} to {pad_score_up_to_round} with score {last_score_in_game_final}")
                    for r_pad in range(last_round_in_game_final + 1, pad_score_up_to_round + 1):
                        scores_data[raw_model_name][r_pad].append(last_score_in_game_final)
                    # Ensure the global max_round reflects the padding target
                    max_round = max(max_round, pad_score_up_to_round)

                processed_games += 1
            else:
                # Game file existed but contained no valid steps
                print(f"Warning: No valid steps found in game log: {game_log_path}. Skipping.", file=sys.stderr)
                failed_games += 1

        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding JSON in {game_log_path}: {e}, skipping.", file=sys.stderr)
            failed_games += 1
        except ValueError as e: # Catch the specific validation error
            print(f"\n SCORE VALIDATION FAILED ", file=sys.stderr)
            print(e, file=sys.stderr)
            print(f" Stopping execution due to validation error. ", file=sys.stderr)
            raise # Re-raise to stop the program
        except Exception as e:
            print(f"Error processing game log {game_log_path}: {e}", file=sys.stderr)
            failed_games += 1

    print(f"Finished analyzing. Processed data from {processed_games} games.")
    if failed_games > 0:
         print(f"Warning: Skipped or failed to fully process {failed_games} log entries/files.")

    if not scores_data and processed_games == 0:
        print("Warning: No valid score data found for any model after processing.", file=sys.stderr)

    # Ensure max_round is at least 0 if padding was requested but no games were processed
    if pad_score_up_to_round >= 0 and max_round < 0 and processed_games > 0:
         max_round = max(max_round, 0) # At least have round 0 if padding was > 0
    elif pad_score_up_to_round >=0 and processed_games == 0 and failed_games > 0:
         pass


    return scores_data, max_round, processed_games, failed_games


def plot_score_trends(scores_data, max_round, output_filename, log_dir_name):
    """
    Generates and saves a line plot of score trends per model with std dev shading,
    using mapped display names for the legend. (No changes needed here)

    Args:
        scores_data (dict): Data structure from parse_game_logs_for_scores.
                           Keys are raw model names.
        max_round (int): The maximum round number for the x-axis.
        output_filename (str): Path to save the plot image.
        log_dir_name (str): Name of the log directory for the title.
    """
    if not scores_data or max_round < 0:
        print(f"No data to plot for {log_dir_name} (max_round={max_round}).")
        return

    #  Plotting setup (unchanged) 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    scaling = 0.7
    fig, ax = plt.subplots(figsize=(10*scaling, 6*scaling))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    rounds = np.arange(max_round + 1) # <<< Will now use potentially padded max_round

    # Sort by raw model name for consistent processing order
    raw_model_names = sorted(scores_data.keys())

    legend_handles = []
    legend_labels = []

    # Get unique display names to assign colors
    display_names = sorted(list(set([get_display_name(raw_name) for raw_name in raw_model_names])))
    color_map = {name: NATURE_COLORS[i % len(NATURE_COLORS)] for i, name in enumerate(display_names)}

    line_styles = ['-', '--', '-.', ':']

    for i, raw_model_name in enumerate(raw_model_names):
        model_rounds_data = scores_data[raw_model_name]
        display_name = get_display_name(raw_model_name)

        mean_scores = np.full(max_round + 1, np.nan)
        std_devs = np.full(max_round + 1, np.nan)

        for r in rounds:
            # This loop correctly handles padded data as it's now in model_rounds_data
            if r in model_rounds_data and model_rounds_data[r]:
                scores_at_round = np.array(model_rounds_data[r])
                valid_scores = scores_at_round[~np.isnan(scores_at_round)] # Handle potential NaNs just in case
                if len(valid_scores) > 0:
                    mean_scores[r] = np.mean(valid_scores)
                    if len(valid_scores) > 1:
                         std_devs[r] = np.std(valid_scores)
                    else:
                         std_devs[r] = 0
                # else leave as NaN

        model_color = color_map[display_name]
        line_style = line_styles[i // len(NATURE_COLORS) % len(line_styles)] if len(display_names) > len(NATURE_COLORS) else '-'

        valid_indices = ~np.isnan(mean_scores)
        if np.any(valid_indices):
             line, = ax.plot(rounds[valid_indices], mean_scores[valid_indices], label=display_name,
                            color=model_color, linestyle=line_style, linewidth=2.5)

             upper_bound = mean_scores + std_devs
             lower_bound = mean_scores - std_devs
             ax.fill_between(rounds[valid_indices], lower_bound[valid_indices], upper_bound[valid_indices], color=model_color, alpha=0.15, interpolate=True)

             legend_handles.append(line)
             legend_labels.append(display_name)
        else:
            print(f"Warning: No valid mean scores calculated for model '{display_name}' in '{log_dir_name}'. Skipping plotting.")

    ax.set_xlabel('Round Number')
    ax.set_ylabel('Average Score')
    ax.set_ylim(bottom=0)

    if legend_handles:
        sorted_legend = sorted(zip(legend_labels, legend_handles), key=lambda x: x[0])
        sorted_labels = [label for label, handle in sorted_legend]
        sorted_handles = [handle for label, handle in sorted_legend]
        legend = ax.legend(sorted_handles, sorted_labels, title='Models',
                bbox_to_anchor=(1.05, 1), loc='upper left',
                frameon=False)
        # legend.get_title().set_fontweight('bold')
    else:
         print(f"Warning: No models had valid data to plot legend for {log_dir_name}.")

    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#E0E0E0')
    ax.set_xlim(0, max_round) # <<< Uses the potentially padded max_round
    plt.tight_layout(rect=[0, 0, 0.85, 1] if legend_handles else [0,0,1,1])

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)
    finally:
        plt.close(fig) # Close figure after saving


def plot_multiple_score_trends(all_data, output_filename):
    """
    Plot multiple log directories as subplots with shared legend, using mapped
    display names. (Relies on max_round from parser, which now includes padding)

    Args:
        all_data (list): List of tuples (scores_data, max_round, log_dir_name)
                         scores_data keys are raw model names.
        output_filename (str): Path to save the plot image
    """
    if not all_data:
        print("No data to plot.")
        return

    all_data = [(data, max_round, dir_name) for data, max_round, dir_name in all_data if data is not None and max_round >= 0]

    if not all_data:
        print("No valid data to plot for multi-plot.")
        return



    #  Plotting setup (largely unchanged) 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    n_plots = len(all_data)
    fig_width = max(10, 4 * n_plots)
    legend_relative_width = 0.15 if n_plots <= 3 else 0.1
    fig = plt.figure(figsize=(fig_width*0.7, 5*0.7))

    plot_width_ratio = (1 - legend_relative_width) / n_plots
    from matplotlib.gridspec import GridSpec
    width_ratios = [plot_width_ratio] * n_plots + [legend_relative_width]
    gs = GridSpec(1, n_plots + 1, width_ratios=width_ratios)

    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

    #  Model color/style mapping (unchanged) 
    all_raw_model_names = set()
    for scores_data, _, _ in all_data:
        if scores_data:
             all_raw_model_names.update(scores_data.keys())
    all_raw_model_names = sorted(list(all_raw_model_names))
    all_display_names = sorted(list(set(get_display_name(name) for name in all_raw_model_names)))
    color_map = {name: NATURE_COLORS[i % len(NATURE_COLORS)] for i, name in enumerate(all_display_names)}
    line_styles = ['-', '--', '-.', ':']
    style_map = {}
    num_colors = len(NATURE_COLORS)
    num_styles = len(line_styles)
    if len(all_display_names) > num_colors:
        for i, name in enumerate(all_display_names):
            color_idx = i % num_colors
            style_idx = (i // num_colors) % num_styles
            style_map[name] = line_styles[style_idx]
    else:
        for name in all_display_names:
            style_map[name] = '-'

    legend_handles = {}
    legend_labels_dict = {}

    #  Find overall max round AFTER considering padding from individual parsers 
    max_round_overall = -1
    for _, max_round_single, _ in all_data:
         max_round_overall = max(max_round_overall, max_round_single)

    if max_round_overall < 0:
        print("Error: Could not determine a valid maximum round for plotting across datasets.")
        plt.close(fig)
        return


    #  Plotting loop (uses max_round_single from parser results) 
    for i, (scores_data, max_round_single, log_dir_name) in enumerate(all_data):
        ax = axes[i]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # max_round_overall = max(max_round_overall, max_round_single) # Done before loop now
        rounds = np.arange(max_round_single + 1) # Use the max_round specific to this dataset


        # Get the display name for the experiment
        display_experiment_name = EXPERIMENT_NAME_MAPPING.get(log_dir_name, log_dir_name) # Fallback to original dir name if not mapped

        ax.set_title(display_experiment_name, fontsize=12, pad=10)

        if not scores_data:
            print(f"Warning: No score data available for subplot {log_dir_name}. Plotting empty axes.")
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='grey')
            ax.set_xlabel('Round Number')
            if i == 0: ax.set_ylabel('Average Score')
            ax.set_ylim(bottom=0)
            # Set xlim even for empty plot for consistency, using overall max
            ax.set_xlim(0, max_round_overall if max_round_overall > 0 else 1)
            continue

        # Plot each model's data
        for raw_model_name in sorted(scores_data.keys()):
            model_rounds_data = scores_data[raw_model_name]
            display_name = get_display_name(raw_model_name)

            mean_scores = np.full(max_round_single + 1, np.nan)
            std_devs = np.full(max_round_single + 1, np.nan)

            for r in rounds: # Iterate up to this subplot's max_round
                if r in model_rounds_data and model_rounds_data[r]:
                    scores_at_round = np.array(model_rounds_data[r])
                    valid_scores = scores_at_round[~np.isnan(scores_at_round)]
                    if len(valid_scores) > 0:
                        mean_scores[r] = np.mean(valid_scores)
                        if len(valid_scores) > 1:
                           std_devs[r] = np.std(valid_scores)
                        else:
                           std_devs[r] = 0

            line_color = color_map[display_name]
            line_style = style_map[display_name]

            valid_indices = ~np.isnan(mean_scores)
            if np.any(valid_indices):
                line, = ax.plot(rounds[valid_indices], mean_scores[valid_indices], linewidth=2.5,
                               color=line_color, linestyle=line_style)

                upper_bound = mean_scores + std_devs
                lower_bound = mean_scores - std_devs
                ax.fill_between(rounds[valid_indices], lower_bound[valid_indices], upper_bound[valid_indices],
                                color=line_color, alpha=0.15, interpolate=True)

                if display_name not in legend_labels_dict:
                     legend_handles[display_name] = line
                     legend_labels_dict[display_name] = display_name

        ax.set_ylim(bottom=0)
        if i == 0:
            ax.set_ylabel('Average Score')
        ax.set_xlabel('Round Number')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#E0E0E0')
        # Set xlim specific to this subplot initially - will be adjusted later
        ax.set_xlim(0, max_round_single) # Set xlim specific to this subplot's data initially

    # Set consistent xlim across all plots using the overall max round
    for ax in axes:
         ax.set_xlim(0, max_round_overall)

    # Create sorted legend
    sorted_display_names = sorted(legend_labels_dict.keys())
    final_legend_handles = [legend_handles[name] for name in sorted_display_names]
    final_legend_labels = [legend_labels_dict[name] for name in sorted_display_names]

    if final_legend_handles:
        legend_ax = fig.add_subplot(gs[0, -1])
        legend_ax.axis('off')
        legend = legend_ax.legend(final_legend_handles, final_legend_labels,
                            title='Models',
                            loc='center left',
                            frameon=False)
        # legend.get_title().set_fontweight('bold')

    plt.tight_layout()

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Multi-plot saved successfully to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


#  Main Execution Logic 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze game logs to generate score trend plots per model with score validation and padding.')
    parser.add_argument('--log-dir', type=str, nargs='+', default=[DEFAULT_LOG_DIR],
                        help=f'One or more log directories containing meta_log.json and game logs (default: {DEFAULT_LOG_DIR})')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output filename for the plot image (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--multi-plot', action='store_true',
                        help='Create a single figure with multiple subplots (one per log directory) with shared legend')
    parser.add_argument('--pad-rounds', type=int, default=DEFAULT_PAD_SCORE_UP_TO_ROUND, # <<< NEW ARG
                        help=f'Pad final scores of shorter games up to this round number (default: {DEFAULT_PAD_SCORE_UP_TO_ROUND})')
    args = parser.parse_args()

    #  Execution based on arguments 
    try:
        pad_target = args.pad_rounds # <<< Get padding value

        if len(args.log_dir) > 1 and args.multi_plot:
            all_plot_data = []
            for log_dir in args.log_dir:
                print(f"\n Processing Directory: {log_dir} ")
                # <<< Pass pad_target to parser
                scores_data, max_round, processed_count, failed_count = parse_game_logs_for_scores(log_dir, pad_target)
                if scores_data is not None and max_round >= 0:
                    if processed_count > 0 or (scores_data and max_round >=0) : # Include if processing happened OR if padding created data
                         all_plot_data.append((scores_data, max_round, os.path.basename(log_dir)))
                    else:
                         print(f"Note: {log_dir} processed {processed_count} games but yielded no plottable score sequences.")
                         # Include empty plot data if desired
                         all_plot_data.append(({}, max_round if max_round >=0 else -1 , os.path.basename(log_dir)))

                elif scores_data is None and max_round == -1:
                    print(f"Could not process {log_dir} due to critical errors (e.g., missing meta_log). Skipping.", file=sys.stderr)
                else:
                    print(f"Could not gather valid score data from {log_dir} (Processed: {processed_count}, Failed: {failed_count}). Skipping.", file=sys.stderr)
                    # Represent skipped plot with empty data if necessary
                    all_plot_data.append(({}, -1, os.path.basename(log_dir)))


            # Filter out entries that genuinely have no chance of plotting (max_round < 0) before plotting
            valid_plot_data = [(d, mr, n) for d, mr, n in all_plot_data if mr >= 0]

            if valid_plot_data:
                print("\n Generating Multi-Plot ")
                plot_multiple_score_trends(valid_plot_data, args.output)
            else:
                print("Could not create multi-plot: No valid data found in any directory after processing.", file=sys.stderr)
                if not all_plot_data: # Check if the original list was also empty (meaning critical errors everywhere)
                     sys.exit(1) # Exit if no dirs could even be processed minimally

        else: # Single plot per directory or single directory specified
            for log_dir in args.log_dir:
                print(f"\n Processing Directory: {log_dir} ")
                if len(args.log_dir) > 1 and not args.multi_plot:
                    base_name = os.path.basename(log_dir)
                    safe_base_name = re.sub(r'[^\w\-]+', '_', base_name)
                    output_file = os.path.join(os.path.dirname(args.output), f"{safe_base_name}_{os.path.basename(args.output)}")
                else:
                    output_file = args.output

                # <<< Pass pad_target to parser
                scores_data, max_round, processed_count, failed_count = parse_game_logs_for_scores(log_dir, pad_target)

                if scores_data is not None and max_round >= 0:
                     if scores_data: # Check if there's actually data (could be empty even if max_round >= 0 due to padding only)
                         print(f"\n Generating Plot for {log_dir} ")
                         plot_score_trends(scores_data, max_round, output_file, os.path.basename(log_dir))
                     else:
                         print(f"No plottable score data found in {log_dir} after processing {processed_count} games (max_round={max_round}). Skipping plot generation.")

                elif processed_count == 0 and failed_count > 0:
                    print(f"Could not generate plot for {log_dir}: No games were successfully processed.", file=sys.stderr)
                elif scores_data is None and max_round == -1:
                    print(f"Could not generate plot for {log_dir} due to critical errors reading logs (e.g., missing meta_log).", file=sys.stderr)
                else: # Includes cases where processing happened but yielded no score data
                    print(f"Could not generate plot for {log_dir}: No valid score data found (Processed: {processed_count}, Failed: {failed_count}, Max Round: {max_round}).", file=sys.stderr)

    except ValueError as validation_error:
        # The error message from parse_game_logs_for_scores already includes details.
        print(f"Exiting due to score validation error.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nAnalysis complete.")