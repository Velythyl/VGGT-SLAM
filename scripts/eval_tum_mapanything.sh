#!/bin/bash

# Evaluate MapAnything or PI3 on TUM RGB-D benchmark sequences.
# Usage: ./scripts/eval_tum_mapanything.sh [stride] [model]
#   stride: downsample factor — load every nth image (default: 1)
#   model:  mapanything | pi3 (default: mapanything)

abs_dir="/home/dominic/Documents"
stride=${1:-1}          # Default to 1 (all images) if not provided
model=${2:-mapanything} # Default to mapanything if not provided
dataset_path="${abs_dir%/}/MASt3R-SLAM/datasets/tum/"
gt_path="${abs_dir%/}/MASt3R-SLAM/datasets/tum/"
log_path="$(pwd)/logs/tum_${model}_results_s${stride}.txt"
infer_script="$(pwd)/map-anything/scripts/infer_tum.py"

mkdir -p "$(pwd)/logs"

datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# Number of full runs
n=1  # <-- change as needed

# If file doesn't exist, write header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    count=0

    for dataset in "${datasets[@]}"; do
        echo "Running ${model} on $dataset (Run $run, stride=$stride)"
        dataset_name="${dataset_path}${dataset}/rgb"
        python "$infer_script" \
            --image_folder "$dataset_name" \
            --stride "$stride" \
            --model "$model" \
            --log_path "$(pwd)/logs/${dataset}_${model}_run${run}_s${stride}.txt"
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        est_path="$(pwd)/logs/${dataset}_${model}_run${run}_s${stride}.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        echo "$run,$dataset,$rmse" >> "$log_path"

        total_rmse=$(echo "$total_rmse + $rmse" | bc -l)
        count=$((count + 1))
    done

    avg_rmse=$(echo "$total_rmse / $count" | bc -l)
    echo "$run,Average,$avg_rmse" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
done
