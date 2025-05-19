import os
import json
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from utils.metrics import (
    shannon_entropy, manhattan_distance, calculate_directional_entropy,
    calculate_stillness_proportion, calculate_message_length_metrics,
    calculate_message_content_metrics, calculate_info_homogeneity,
    calculate_norm_mean_message_embedding, calculate_avg_moving_distance,
    calculate_exploration_rate, calculate_coordination_metrics,
    calculate_local_structure_preservation, calculate_agent_push_events,
    calculate_all_metrics, MOVE_ACTIONS, ACTUAL_MOVE_ACTIONS, 
    COORDINATION_ACTIONS, ACTION_VECTORS
)

DEFAULT_EMBEDDING_MODEL = 'all-mpnet-base-v2'
FIGSIZE_ANALYSIS_PLOT = (10, 4)

from utils.load_fonts import *

#  Helper Functions 

class SingleModelMultiGameAnalyzer:
    def __init__(self, log_dir, model_name_exact, output_pdf_path):
        self.log_dir = log_dir
        self.model_name_exact = model_name_exact
        self.output_pdf_path = output_pdf_path
        print(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}...")
        self.embedding_model = None
        try:
            self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Warning: Could not load embedding model {DEFAULT_EMBEDDING_MODEL}. Metrics requiring embeddings will be skipped. Error: {e}")

        print(f"Initialized analyzer for model: '{self.model_name_exact}' in dir: '{self.log_dir}'")

    def _load_game_data_for_model(self):
        meta_log_path = os.path.join(self.log_dir, 'meta_log.json')
        if not os.path.exists(meta_log_path):
            print(f"Error: meta_log.json not found in {self.log_dir}")
            return []
        with open(meta_log_path, 'r') as f: meta_data = json.load(f)
        game_files_data = []
        print(f"Scanning meta_log for games by model '{self.model_name_exact}'...")
        for timestamp, info in meta_data.items():
            if info.get("model") == self.model_name_exact:
                game_log_path = os.path.join(self.log_dir, f'game_log_{timestamp}.json')
                agent_log_path = os.path.join(self.log_dir, f'agent_log_{timestamp}.json')
                if os.path.exists(game_log_path) and os.path.exists(agent_log_path):
                    game_files_data.append({
                        'timestamp': timestamp,
                        'game_log_path': game_log_path,
                        'agent_log_path': agent_log_path
                    })
                else:
                    print(f"Warning: Missing game or agent log for {timestamp} (model {self.model_name_exact})")
        print(f"Found {len(game_files_data)} game(s) for model '{self.model_name_exact}'.")
        return game_files_data

    def _calculate_metrics_for_game(self, game_log_path, agent_log_path):
        with open(game_log_path, 'r') as f: game_log = json.load(f)
        with open(agent_log_path, 'r') as f: agent_log = json.load(f)

        actions_by_agent_by_round = defaultdict(lambda: defaultdict(str))
        messages_by_round = defaultdict(list)
        
        positions_by_round = defaultdict(dict)
        initial_positions = {}
        cumulative_distances_by_round = defaultdict(dict)
        cumulative_distances_by_agent = defaultdict(float)
        move_attempts_by_round = defaultdict(int) 
        move_successes_by_round = defaultdict(int) 
        explored_cells = set()
        
        for entry in agent_log:
            round_num = int(entry['round'])
            agent_id = entry.get('agent_id', 'unknown_agent')
            
            if 'action' in entry and agent_id != 'unknown_agent': 
                actions_by_agent_by_round[round_num][agent_id] = str(entry['action'])
            
            msg_content = str(entry.get('message', '')).strip()
            if msg_content:
                messages_by_round[round_num].append(msg_content)
        
        for round_data_idx, round_data in enumerate(game_log):
            round_num_gl = int(round_data['round'])
            if 'agents' in round_data:
                for agent_state in round_data['agents']:
                    agent_id_gl = agent_state.get('id', f'unknown_agent_{round_num_gl}_{agent_state.get("x",0)}_{agent_state.get("y",0)}')
                    if 'x' in agent_state and 'y' in agent_state:
                        current_pos_gl = (agent_state['x'], agent_state['y'])
                        positions_by_round[round_num_gl][agent_id_gl] = current_pos_gl
                        explored_cells.add(current_pos_gl)

                        if agent_id_gl not in initial_positions:
                            initial_positions[agent_id_gl] = current_pos_gl
                            cumulative_distances_by_agent[agent_id_gl] = 0.0
                        
                        if round_num_gl > 0 and agent_id_gl in positions_by_round.get(round_num_gl-1, {}):
                            prev_pos_gl = positions_by_round[round_num_gl-1][agent_id_gl]
                            distance_moved = manhattan_distance(prev_pos_gl, current_pos_gl)
                            cumulative_distances_by_agent[agent_id_gl] += distance_moved
                        
                        cumulative_distances_by_round[round_num_gl][agent_id_gl] = cumulative_distances_by_agent[agent_id_gl]

                    if 'action_result' in agent_state:
                        action_result = agent_state.get('action_result', False)
                        action = agent_state.get('action', '')
                        if action in ACTUAL_MOVE_ACTIONS:
                            move_attempts_by_round[round_num_gl] += 1
                            if action_result:
                                move_successes_by_round[round_num_gl] += 1
                                
        num_rounds_game_log = len(game_log)
        game_metrics = defaultdict(list)
        
        explored_cells_by_round = {}
        current_explored_cells = set()

        for i in range(num_rounds_game_log):
            round_data = game_log[i]
            current_round_num = int(round_data['round'])
            game_metrics['score'].append(float(round_data.get('score', 0)))
            
            round_actions_all_agents = list(actions_by_agent_by_round.get(current_round_num, {}).values())
            round_messages_str = messages_by_round.get(current_round_num, [])

            # Message Metrics
            if round_messages_str:
                # Message content metrics
                prop_question, prop_digit = calculate_message_content_metrics(round_messages_str)
                game_metrics['prop_question_sentences'].append(prop_question)
                game_metrics['prop_digit_chars'].append(prop_digit)
                
                # Message length metrics
                mean_length, std_length = calculate_message_length_metrics(round_messages_str)
                game_metrics['mean_message_length'].append(mean_length)
                game_metrics['std_message_length'].append(std_length)
            else:
                game_metrics['prop_question_sentences'].append(0.0)
                game_metrics['prop_digit_chars'].append(0.0)
                game_metrics['mean_message_length'].append(0.0)
                game_metrics['std_message_length'].append(0.0)

            # Embedding-based Metrics
            if self.embedding_model:
                unique_round_messages_str = list(set(round_messages_str))
                try:
                    # Information homogeneity
                    info_homogeneity = calculate_info_homogeneity(
                        unique_round_messages_str, 
                        embedding_model=self.embedding_model
                    )
                    game_metrics['info_homogeneity'].append(info_homogeneity)
                    
                    # Norm of mean message embedding
                    norm_embedding = calculate_norm_mean_message_embedding(
                        unique_round_messages_str, 
                        embedding_model=self.embedding_model
                    )
                    game_metrics['norm_mean_message_embedding'].append(norm_embedding)
                except Exception as e:
                    print(f"Warning: Error calculating embedding metrics in round {current_round_num}: {e}")
                    game_metrics['info_homogeneity'].append(np.nan)
                    game_metrics['norm_mean_message_embedding'].append(np.nan)
            else:
                game_metrics['info_homogeneity'].append(np.nan)
                game_metrics['norm_mean_message_embedding'].append(np.nan)

            # Movement and Coordination Metrics
            game_metrics['directional_entropy'].append(
                calculate_directional_entropy(round_actions_all_agents)
            )
            
            game_metrics['stillness_proportion'].append(
                calculate_stillness_proportion(round_actions_all_agents)
            )
            
            # Calculate coordination metrics
            dominant_action_prop, polarization_index = calculate_coordination_metrics(round_actions_all_agents)
            game_metrics['dominant_action_prop'].append(dominant_action_prop)
            game_metrics['polarization_index'].append(polarization_index)

            # Distance metrics
            round_distances = cumulative_distances_by_round.get(current_round_num, {})
            if round_distances:
                avg_distance = sum(round_distances.values()) / len(round_distances)
                game_metrics['avg_moving_distance'].append(avg_distance)
            else:
                game_metrics['avg_moving_distance'].append(0.0)
            
            # Exploration metrics
            if current_round_num in positions_by_round:
                for pos in positions_by_round[current_round_num].values():
                    current_explored_cells.add(pos)
            
            explored_cells_by_round[current_round_num] = len(current_explored_cells)
            game_metrics['exploration_rate'].append(explored_cells_by_round[current_round_num])

            # Structure preservation and agent push metrics
            if current_round_num > 0:
                prev_round_positions_map = positions_by_round.get(current_round_num - 1, {})
                actions_at_prev_decision_phase = actions_by_agent_by_round.get(current_round_num - 1, {})
                current_round_positions_map = positions_by_round.get(current_round_num, {})
                
                # Calculate local structure preservation
                local_structures_preserved_this_round = calculate_local_structure_preservation(
                    current_round_positions_map, prev_round_positions_map
                )
                
                # Calculate agent push events
                agent_pushes_this_round = calculate_agent_push_events(
                    current_round_positions_map, prev_round_positions_map, actions_at_prev_decision_phase
                )
            else:
                local_structures_preserved_this_round = 0
                agent_pushes_this_round = 0
                
            game_metrics['local_structure_preservation_count'].append(local_structures_preserved_this_round)
            game_metrics['agent_push_events'].append(agent_pushes_this_round)

        return game_metrics

    def analyze(self):
        game_files = self._load_game_data_for_model()
        if not game_files:
            print("No game data found for the specified model. Aborting analysis.")
            return

        all_games_metrics_raw = []
        print(f"Calculating metrics for {len(game_files)} games...")
        for game_file_info in tqdm(game_files, desc="Processing Games"):
            metrics = self._calculate_metrics_for_game(
                game_file_info['game_log_path'],
                game_file_info['agent_log_path']
            )
            all_games_metrics_raw.append(metrics)

        if not all_games_metrics_raw:
            print("No metrics could be calculated. Aborting.")
            return

        max_rounds = 0
        for game_m in all_games_metrics_raw:
            if 'score' in game_m and len(game_m['score']) > max_rounds:
                max_rounds = len(game_m['score'])
        print(f"Max rounds found across games: {max_rounds}")
        if max_rounds == 0:
            print("No rounds found in any game. Aborting plot generation.")
            return

        metric_keys = [
            'score', 'info_homogeneity', 'norm_mean_message_embedding',
            'directional_entropy', 'stillness_proportion',
            'mean_message_length', 'std_message_length',
            'avg_moving_distance', 'exploration_rate', 
            'dominant_action_prop', 'polarization_index',
            'local_structure_preservation_count', 'agent_push_events',
            'prop_question_sentences', 'prop_digit_chars' # Added new metric keys
        ]
        if self.embedding_model is None:
             if 'info_homogeneity' in metric_keys: metric_keys.remove('info_homogeneity')
             if 'norm_mean_message_embedding' in metric_keys: metric_keys.remove('norm_mean_message_embedding')

        all_games_metrics_padded = defaultdict(list)
        for game_m_raw in all_games_metrics_raw:
            game_len = len(game_m_raw.get('score', []))
            for key in metric_keys:
                metric_series = game_m_raw.get(key, [np.nan] * game_len)
                if not isinstance(metric_series, list): 
                    metric_series = list(metric_series) if hasattr(metric_series, '__iter__') else [np.nan] * game_len
                padded_series = list(metric_series) + [np.nan] * (max_rounds - len(metric_series))
                all_games_metrics_padded[key].append(padded_series)

        aggregated_metrics = {}
        for key in metric_keys:
            if key in all_games_metrics_padded and all_games_metrics_padded[key]:
                stacked_array = np.array(all_games_metrics_padded[key])
                if stacked_array.size > 0 and not np.all(np.isnan(stacked_array)):
                     aggregated_metrics[f'mean_{key}'] = np.nanmean(stacked_array, axis=0)
                     aggregated_metrics[f'std_{key}'] = np.nanstd(stacked_array, axis=0)
                else:
                    aggregated_metrics[f'mean_{key}'] = np.full(max_rounds, np.nan)
                    aggregated_metrics[f'std_{key}'] = np.full(max_rounds, np.nan)
            else:
                aggregated_metrics[f'mean_{key}'] = np.full(max_rounds, np.nan)
                aggregated_metrics[f'std_{key}'] = np.full(max_rounds, np.nan)

        final_scores = [game_m['score'][-1] if game_m.get('score') and len(game_m['score']) > 0 else -np.inf for game_m in all_games_metrics_raw]
        if not final_scores or np.all(np.isneginf(final_scores)):
            print("Warning: No valid final scores found. Max score game plot will be based on first game or be empty.")
            max_score_idx = 0
            max_score_game_metrics_raw = all_games_metrics_raw[0] if all_games_metrics_raw else {}
        else:
            max_score_idx = np.argmax(final_scores)
            max_score_game_metrics_raw = all_games_metrics_raw[max_score_idx]

        max_score_game_metrics_padded = {}
        game_to_pad = max_score_game_metrics_raw
        game_len_max_score = len(game_to_pad.get('score', []))
        for key in metric_keys:
            metric_series = game_to_pad.get(key, [np.nan] * game_len_max_score)
            if not isinstance(metric_series, list):
                 metric_series = list(metric_series) if hasattr(metric_series, '__iter__') else [np.nan] * game_len_max_score
            padded_series = list(metric_series) + [np.nan] * (max_rounds - len(metric_series))
            max_score_game_metrics_padded[key] = np.array(padded_series)

        self.plot_results(max_score_game_metrics_padded, aggregated_metrics, max_rounds, metric_keys)


    def plot_results(self, max_score_game_metrics, aggregated_metrics, num_total_rounds, available_metric_keys):
        if num_total_rounds <= 1:
            print("Not enough rounds to plot (requires at least 2 rounds total). Skipping plot generation.")
            return

        num_rounds_to_plot = num_total_rounds - 1
        if num_rounds_to_plot <= 0:
            print("Not enough rounds to plot after excluding the last one. Skipping plot generation.")
            return

        rounds_x_plot = np.arange(num_rounds_to_plot)

        score_plot_info = {'score': {'label': 'Score', 'max_color': 'black', 'mean_color': '#404040', 'fill_color': '#D3D3D3'}}

        metric_config = {
            'info_homogeneity': {'base_label': 'Information\nHomogeneity', 'color': '#0f5dd7', 'scale_factor': 2, 'panel': 2},
            # 'norm_mean_message_embedding': {'base_label': 'Norm of\nMessage\nEmbedding', 'color': '#a160ff', 'scale_factor': 1, 'panel': 2}, # Potentially replace
            'mean_message_length': {'base_label': 'Message\nLength', 'color': '#1fa774', 'scale_factor': 200, 'panel': 2}, 
            # 'std_message_length': {'base_label': 'SD Message\nLength', 'color': '#f0833a', 'scale_factor': 100, 'panel': 2}, # Potentially replace
            
            'prop_question_sentences': {'base_label': 'Question Prop.\n(Messages)', 'color': '#a160ff', 'scale_factor': 1, 'panel': 2}, # New, using a freed color
            'prop_digit_chars': {'base_label': 'Digit Char\nProportion', 'color': '#f0833a', 'scale_factor': 1, 'panel': 2}, # New, using a freed color

            'directional_entropy': {'base_label': 'Directional\nEntropy', 'color': '#f9516a', 'scale_factor': 2, 'panel': 3},
            'stillness_proportion': {'base_label': 'Stillness\nProportion', 'color': '#33e0c8', 'scale_factor': 1, 'panel': 3},
            'dominant_action_prop': {'base_label': 'Dominant Action\nProportion', 'color': '#ffc107', 'scale_factor': 1, 'panel': 3}, 
            'polarization_index': {'base_label': 'Polarization\nIndex', 'color': '#8A2BE2', 'scale_factor': 1, 'panel': 3},       
            
            'avg_moving_distance': {'base_label': 'Moving\nDistance', 'color': '#e6194B', 'scale_factor': 50, 'panel': 4},
            'exploration_rate': {'base_label': 'Exploration\nRate', 'color': '#ffe119', 'scale_factor': 100, 'panel': 4},
            'local_structure_preservation_count': {'base_label': 'Local Structure\nPreservation', 'color': '#4B0082', 'scale_factor': 10, 'panel': 4},
            'agent_push_events': {'base_label': 'Agent Push\nEvents', 'color': '#FF4500', 'scale_factor': 4, 'panel': 4}
        }
        
        # Remove old metrics if they are fully replaced and not just commented out for choice
        if 'norm_mean_message_embedding' in metric_config and 'norm_mean_message_embedding' not in available_metric_keys :
            del metric_config['norm_mean_message_embedding']
        if 'std_message_length' in metric_config and 'std_message_length' not in available_metric_keys:
             del metric_config['std_message_length']


        for key, config in metric_config.items():
            if config['scale_factor'] != 1:
                config['display_label'] = f"{config['base_label']} ÷{config['scale_factor']}"
            else:
                config['display_label'] = config['base_label']
        
        active_metrics = {k: v for k, v in metric_config.items() if k in available_metric_keys}

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharey=False)
        panel_labels = {(0, 0): 'a', (0, 1): 'b', (1, 0): 'c', (1, 1): 'd'}
        
        for metric_key_iter in active_metrics:
            scale_factor = active_metrics[metric_key_iter]['scale_factor']
            if scale_factor != 1:
                if f'mean_{metric_key_iter}' in aggregated_metrics and aggregated_metrics[f'mean_{metric_key_iter}'] is not None:
                    aggregated_metrics[f'mean_{metric_key_iter}'] = aggregated_metrics[f'mean_{metric_key_iter}'] / scale_factor
                if f'std_{metric_key_iter}' in aggregated_metrics and aggregated_metrics[f'std_{metric_key_iter}'] is not None:
                    aggregated_metrics[f'std_{metric_key_iter}'] = aggregated_metrics[f'std_{metric_key_iter}'] / scale_factor
                if metric_key_iter in max_score_game_metrics and max_score_game_metrics[metric_key_iter] is not None:
                    max_score_game_metrics[metric_key_iter] = max_score_game_metrics[metric_key_iter] / scale_factor
        
        ax_score = axes[0, 0]
        metric_key_score = 'score'
        info_score = score_plot_info[metric_key_score]
        score_data_to_plot = []

        if metric_key_score in max_score_game_metrics:
            max_trace_score = max_score_game_metrics.get(metric_key_score)
            if max_trace_score is not None and len(max_trace_score) >= num_rounds_to_plot:
                max_data = max_trace_score[:num_rounds_to_plot]
                if not np.all(np.isnan(max_data)): score_data_to_plot.append(max_data)
                ax_score.plot(rounds_x_plot, max_data, label=f'Maximum ', color=info_score['max_color'], linestyle='-', linewidth=1.5, alpha=0.9)

        mean_trace_score = aggregated_metrics.get(f'mean_{metric_key_score}')
        std_trace_score = aggregated_metrics.get(f'std_{metric_key_score}')

        if mean_trace_score is not None and std_trace_score is not None \
        and len(mean_trace_score) >= num_rounds_to_plot and len(std_trace_score) >= num_rounds_to_plot:
            mean_plot = mean_trace_score[:num_rounds_to_plot]
            std_plot = std_trace_score[:num_rounds_to_plot]
            if not np.all(np.isnan(mean_plot)): score_data_to_plot.append(mean_plot)
            std_plot_safe = np.where(np.isnan(mean_plot), np.nan, std_plot)
            ax_score.plot(rounds_x_plot, mean_plot, label=f'Mean', color=info_score['mean_color'], linestyle='--', linewidth=1.5)
            with np.errstate(invalid='ignore'):
                lower_bound = np.maximum(0, mean_plot - std_plot_safe) if np.all(mean_plot[~np.isnan(mean_plot)] >= 0) else mean_plot - std_plot_safe
                upper_bound = mean_plot + std_plot_safe
            if not np.all(np.isnan(upper_bound)): score_data_to_plot.append(upper_bound)
            if not np.all(np.isnan(lower_bound)): score_data_to_plot.append(lower_bound)
            ax_score.fill_between(rounds_x_plot, lower_bound, upper_bound, color=info_score['fill_color'], alpha=0.4, label=f'Mean ± Std. Dev.')
        
        if score_data_to_plot:
            valid_data_for_ylim = [d[~np.isnan(d)] for d in score_data_to_plot if d is not None and d[~np.isnan(d)].size > 0]
            if valid_data_for_ylim:
                all_values = np.concatenate(valid_data_for_ylim)
                if all_values.size > 0 :
                    ymin_score = 0 
                    ymax_score = np.nanmax(all_values) * 1.25 
                    ax_score.set_ylim(ymin_score, ymax_score)

        ax_score.set_ylabel('Score', fontsize=10)
        ax_score.set_xlabel('Round', fontsize=10)
        ax_score.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize='small', frameon=False, ncol=3) 
        ax_score.grid(True, linestyle=':', alpha=0.6)
        ax_score.spines['top'].set_visible(False)
        ax_score.spines['right'].set_visible(False)
        ax_score.tick_params(direction='in', top=False, right=False)
        ax_score.text(-0.15, 1.05, panel_labels[(0, 0)], transform=ax_score.transAxes, fontsize=16, fontweight='bold', va='top')

        panel_metrics = {2: [], 3: [], 4: []}
        for metric_key, config in active_metrics.items():
            panel = config['panel']
            if panel in panel_metrics:
                panel_metrics[panel].append(metric_key)
        
        panel_to_subplot = {2: (0, 1), 3: (1, 0), 4: (1, 1)}
        
        for panel_idx, (row, col) in panel_to_subplot.items():
            ax = axes[row, col]
            metrics_for_panel = panel_metrics.get(panel_idx, [])
            legend_handles = []
            all_values_in_panel_for_ylim = []
            
            for metric_key_panel in metrics_for_panel:
                config = active_metrics[metric_key_panel]
                label = config['display_label']
                color = config['color']
                
                max_trace = max_score_game_metrics.get(metric_key_panel)
                if max_trace is not None and len(max_trace) >= num_rounds_to_plot:
                    max_data = max_trace[:num_rounds_to_plot]
                    if not np.all(np.isnan(max_data)): all_values_in_panel_for_ylim.extend(max_data[~np.isnan(max_data)])
                    ax.plot(rounds_x_plot, max_data, color=color, linestyle='-', linewidth=1.2, alpha=0.4)

                mean_trace = aggregated_metrics.get(f'mean_{metric_key_panel}')
                std_trace = aggregated_metrics.get(f'std_{metric_key_panel}')

                if mean_trace is not None and std_trace is not None \
                and len(mean_trace) >= num_rounds_to_plot and len(std_trace) >= num_rounds_to_plot:
                    mean_plot = mean_trace[:num_rounds_to_plot]
                    std_plot = std_trace[:num_rounds_to_plot]
                    std_plot_safe = np.where(np.isnan(mean_plot), np.nan, std_plot) 
                    if not np.all(np.isnan(mean_plot)): all_values_in_panel_for_ylim.extend(mean_plot[~np.isnan(mean_plot)])
                    line, = ax.plot(rounds_x_plot, mean_plot, label=label, color=color, linestyle='--', linewidth=1.35)
                    legend_handles.append(line)
                    with np.errstate(invalid='ignore'):
                        is_metric_non_negative = np.all(mean_plot[~np.isnan(mean_plot)] >= 0) if not np.all(np.isnan(mean_plot)) else True
                        
                        # For these new metrics, they are proportions, so non-negative.
                        # We can keep the existing logic for lower_bound_condition or refine if needed.
                        # The existing logic already excludes info_homogeneity and norm_mean_message_embedding from strict non-negativity.
                        # If 'norm_mean_message_embedding' is removed, this check might become simpler or might need adjustment based on what remains.
                        lower_bound_condition = is_metric_non_negative and \
                                                metric_key_panel not in ['info_homogeneity', 'norm_mean_message_embedding'] # norm_mean_message_embedding might be removed
                        
                        lower_bound = np.maximum(0, mean_plot - std_plot_safe) if lower_bound_condition else mean_plot - std_plot_safe
                        upper_bound = mean_plot + std_plot_safe
                    if not np.all(np.isnan(lower_bound)): all_values_in_panel_for_ylim.extend(lower_bound[~np.isnan(lower_bound)])
                    if not np.all(np.isnan(upper_bound)): all_values_in_panel_for_ylim.extend(upper_bound[~np.isnan(upper_bound)])
                    ax.fill_between(rounds_x_plot, lower_bound, upper_bound, color=color, alpha=0.15)
            
            if all_values_in_panel_for_ylim: 
                is_panel_generally_non_negative = all(
                    (np.all(aggregated_metrics.get(f'mean_{mk}', np.array([0]))[~np.isnan(aggregated_metrics.get(f'mean_{mk}', np.array([0])))] >= 0)
                     if aggregated_metrics.get(f'mean_{mk}') is not None and not np.all(np.isnan(aggregated_metrics.get(f'mean_{mk}'))) else True)
                    for mk in metrics_for_panel
                ) and all(
                     active_metrics[mk]['base_label'] not in ['Information\nHomogeneity', 'Norm of\nMessage\nEmbedding'] # This check might change if norm_mean_message_embedding is removed
                     for mk in metrics_for_panel if mk in active_metrics # ensure mk is valid
                )

                ymin_panel_val = np.nanmin(all_values_in_panel_for_ylim) if all_values_in_panel_for_ylim else 0
                ymax_panel_val = np.nanmax(all_values_in_panel_for_ylim) if all_values_in_panel_for_ylim else 1
                ymin_panel = 0 if is_panel_generally_non_negative and ymin_panel_val >= 0 else ymin_panel_val
                range_val = ymax_panel_val - ymin_panel
                if range_val == 0 or np.isnan(range_val): 
                    if ymin_panel == 0 and ymax_panel_val == 0:
                        ymin_panel = -0.1
                        ymax_panel_val = 0.1
                    elif ymin_panel == ymax_panel_val:
                         ymin_panel -= abs(ymin_panel * 0.1) if ymin_panel != 0 else 0.1
                         ymax_panel_val += abs(ymax_panel_val * 0.1) if ymax_panel_val != 0 else 0.1
                    else:
                        ymin_panel -= 0.1
                        ymax_panel_val += 0.1
                    range_val = ymax_panel_val - ymin_panel

                final_ymin = ymin_panel - 0.05 * range_val
                if is_panel_generally_non_negative and ymin_panel_val >=0 and final_ymin < 0:
                    final_ymin = 0
                ax.set_ylim(final_ymin, ymax_panel_val + 0.25 * range_val) 
            
            ax.set_ylabel("Metric Value", fontsize=10)
            ax.set_xlabel('Round', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='in', top=False, right=False)
            ax.text(-0.15, 1.05, panel_labels[(row, col)], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
            
            if legend_handles:
                if len(legend_handles) <= 2: ncols_legend = len(legend_handles)
                elif len(legend_handles) <= 4 : ncols_legend = 2
                else: ncols_legend = 3 
                ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                          fontsize='small', frameon=False, ncol=ncols_legend)

        plt.tight_layout(rect=[0, 0, 1, 0.98]) 
        os.makedirs(os.path.dirname(self.output_pdf_path), exist_ok=True)
        try:
            plt.savefig(self.output_pdf_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {self.output_pdf_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multiple game logs for a single model and generate plots.")
    parser.add_argument('--log-dir', type=str, required=True, help="Directory containing meta_log.json and game/agent logs.")
    parser.add_argument('--model-name', type=str, required=True, help="Exact model name string used in meta_log.json (e.g., 'claude-3-opus-20240229').")
    parser.add_argument('--output-pdf', type=str, default="model_dynamics_analysis.pdf", help="Path to save the output PDF plot.")

    args = parser.parse_args()

    if args.output_pdf == "model_dynamics_analysis.pdf": 
        sanitized_model_name = re.sub(r'[\\/*?:"<>|]', "_", args.model_name) 
        output_filename = f"{sanitized_model_name}_dynamics_analysis.pdf"
        default_output_dir = os.path.join(args.log_dir, "analysis_plots")
        if not os.path.exists(default_output_dir):
            try:
                os.makedirs(default_output_dir, exist_ok=True)
            except OSError as e:
                print(f"Warning: Could not create default output directory {default_output_dir}: {e}")
                default_output_dir = "." 
        
        args.output_pdf = os.path.join(default_output_dir, output_filename)
        print(f"Output PDF not specified, defaulting to: {args.output_pdf}")

    analyzer = SingleModelMultiGameAnalyzer(
        log_dir=args.log_dir,
        model_name_exact=args.model_name,
        output_pdf_path=args.output_pdf
    )
    analyzer.analyze()
    
    
"""
python analysis/plot_metrics.py --log-dir experiment_v33 --model-name "gemini-2.0-flash" --output-pdf ./figs/mc_gemini-pursuit_analysis.pdf

"""
