#!/bin/bash

models=(
    "DeepSeek-V3"
    "claude-3-5-haiku-20241022"
    "claude-3-7-sonnet-20250219"
    "deepseek-ai/DeepSeek-R1"
    "gemini-2.0-flash"
    "gpt-4.1"
    "gpt-4.1-mini"
    "gpt-4o"
    "o3-mini"
    "o4-mini"
    "Meta-Llama-3.1-70B-Instruct"
    "meta-llama/llama-4-scout"
    "qwen/qwq-32b"
)

experiments=(
    "v01 flocking"
    "v02 pursuit"
    "v03 synchronize"
    "v04 foraging"
    "v05 transport"
)

for model in "${models[@]}"; do
    model_name_for_file=$(echo "$model" | tr '/.' '-')
    for exp in "${experiments[@]}"; do
        exp_num=$(echo "$exp" | cut -d' ' -f1)
        exp_name=$(echo "$exp" | cut -d' ' -f2)
        python analysis/plot_metrics.py \
            --log-dir "experiment_${exp_num}" \
            --model-name "$model" \
            --output-pdf "./figs/mc_${model_name_for_file}-${exp_name}_analysis.pdf"
    done
done