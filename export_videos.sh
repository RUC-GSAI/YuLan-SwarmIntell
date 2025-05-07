for i in {01..05}; do
    rm ./latex_output -rf
    python ./analysis/generate_replay_videos.py --log-dir experiment_v$i -v --export-all --create-video --fps 5 --video-file v$i.mp4 --output-dir latex_output
done