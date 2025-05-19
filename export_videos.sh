# for i in {01..05}; do
#     rm ./latex_output -rf
#     python ./analysis/visualizer.py --log-dir experiment_v$i -v --create-gif --gif-file v$i.gif --gif-dpi 240 --gif-width 1280 --gif-dither none --force-recompile
# done

for i in 01 02 03 04 05; do
    for name in "gemini-2.0-flash" "o4-mini" "claude-3-7-sonnet-20250219"; do
        rm ./latex_frames_export/*.tex -rf
        rm ./latex_frames_export/image_gen -rf
        rm ./latex_frames_export/pdf_gen -rf
        rm ./latex_frames_export/pdf_gen_tmp -rf
        python ./analysis/visualizer.py --log-dir experiment_v$i -v --draw-best --model-name $name --create-gif --gif-file v${i}_${name}_best.gif --gif-dpi 240 --gif-width 1280 --gif-dither none --force-recompile
        # rm ./latex_frames_export/*.tex -rf
        # rm ./latex_frames_export/image_gen -rf
        # rm ./latex_frames_export/pdf_gen -rf
        # rm ./latex_frames_export/pdf_gen_tmp -rf
        # python ./analysis/visualizer.py --log-dir experiment_v$i -v --draw-worst --model-name $name --create-gif --gif-file v${i}_${name}_worst.gif --gif-dpi 240 --gif-width 1280 --gif-dither none --force-recompile
    done
done