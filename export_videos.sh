for i in {01..05}; do
    rm ./latex_output -rf
    python ./analysis/generate_replay_videos.py --log-dir experiment_v$i -v --create-gif --gif-file v$i.gif --gif-dpi 240 --gif-width 1280 --gif-dither none --force-recompile
done
