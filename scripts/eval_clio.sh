#!/bin/bash

# Evaluate VGGT-SLAM on the three Clio scenes (cubicle, apartment, office).
# Runs all three scenes first, then prints ATE for each at the end.

img_base="/home/dominic/cam_ws/imgs"
eval_script="$(pwd)/scripts/eval_clio.py"

scenes=(cubicle apartment office)

for scene in "${scenes[@]}"; do
    echo "==== Running VGGT-SLAM on: $scene ===="
    python main.py \
        --image_folder "${img_base}/${scene}/images" \
        --log_results \
        --skip_dense_log \
        --log_path "${scene}.txt" \
        --max_loops 1 \
        --lc_thres 0.80 \
        --submap_size 8
done

echo ""
echo "==== Clio Results ===="
for scene in "${scenes[@]}"; do
    echo "---- $scene ----"
    python "$eval_script" --scene "$scene" --est_traj "${scene}.txt" --vis
done
