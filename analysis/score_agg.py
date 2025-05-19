# -*- coding: utf-8 -*-

import numpy as np
from colorama import init, Fore, Style
import json
import os
import sys
from collections import defaultdict
import argparse

init(autoreset=True)

def calculate_average_scores(log_dir):
    meta_log_path = os.path.join(log_dir, 'meta_log.json')
    if not os.path.exists(meta_log_path):
        print(f"{Fore.RED}Error: Meta log not found: {meta_log_path}", file=sys.stderr)
        return None
    try:
        with open(meta_log_path, encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error reading meta log {meta_log_path}: {e}", file=sys.stderr)
        return None

    scores_by_model = defaultdict(list)
    print(f"Analyzing logs in: {log_dir}")

    for timestamp, info in meta.items():
        model_name = info.get("model")
        if not model_name:
            continue

        game_log_path = os.path.join(log_dir, f'game_log_{timestamp}.json')
        if not os.path.exists(game_log_path):
            continue

        try:
            with open(game_log_path, encoding='utf-8') as f:
                game_steps = json.load(f)
            
            final_score = game_steps[-1].get('score') 
            if isinstance(final_score, (int, float)):
                scores_by_model[model_name].append(float(final_score))
        except Exception:
            continue
    
    if not scores_by_model:
        print("No valid scores found in game logs.")
        return []

    results_list = []
    for model, scores in scores_by_model.items():
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            results_list.append((model, avg, std, len(scores)))

    results_list.sort(key=lambda item: item[1], reverse=True)
    return results_list

def print_score_table(results_list):
    max_model_len = len("Model")
    max_score_len = len("Avg Score")
    max_std_len = len("± Std Dev")
    max_games_len = len("Games")

    for model, avg_score, std_dev, game_count in results_list:
        max_model_len = max(max_model_len, len(model))
        max_score_len = max(max_score_len, len(f"{avg_score:.2f}"))
        max_std_len = max(max_std_len, len(f"± {std_dev:.2f}"))
        max_games_len = max(max_games_len, len(str(game_count)))

    header = (f"{'Model':<{max_model_len}} | "
              f"{'Avg Score':>{max_score_len}} | "
              f"{'± Std Dev':>{max_std_len}} | "
              f"{'Games':>{max_games_len}}")
    print("\n" + Style.BRIGHT + header + Style.RESET_ALL)
    print("-" * len(header))

    for model, avg_score, std_dev, game_count in results_list:
        std_dev_str = f"± {std_dev:.2f}"
        print(f"{model:<{max_model_len}} | "
              f"{avg_score:>{max_score_len}.2f} | "
              f"{std_dev_str:>{max_std_len}} | "
              f"{game_count:>{max_games_len}}")
    print("-" * len(header))

DEFAULT_LOG_DIR = 'experiment_xx'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and display average scores from game logs.')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR,
                        help=f'Log directory (default: {DEFAULT_LOG_DIR})')
    args = parser.parse_args()

    results = calculate_average_scores(args.log_dir)

    if results is None:
        sys.exit(1)
    elif not results:
        pass
    else:
        print_score_table(results)
        
        
"""
python analysis/score_agg.py --log-dir experiment_xx
"""