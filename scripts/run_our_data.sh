#!/bin/bash

# Usage: bash scripts/run_our_data.sh <dataset_root>
# Example: bash scripts/run_our_data.sh /media/asus/H1/datasets

DATASET_ROOT=${1:?"Usage: bash scripts/run_our_data.sh <dataset_root>"}

declare -A SCENES
# SeathruNeRF scenes
SCENES["10_SeathruNeRF_dataset/Panama"]="config/Panama.yaml"
SCENES["10_SeathruNeRF_dataset/Curasao"]="config/Curasao.yaml"
SCENES["10_SeathruNeRF_dataset/JapaneseGardens"]="config/Jap_RedSea.yaml"
SCENES["10_SeathruNeRF_dataset/IUI3"]="config/RedSea.yaml"

# WaterSplat-SLAM scenes
SCENES["pool_up"]="config/pool_up.yaml"
SCENES["pool_up2"]="config/pool_up2.yaml"
SCENES["5_pool"]="config/5_pool.yaml"
SCENES["pipe_local"]="config/pipe_local.yaml"
SCENES["big_gate"]="config/big_gate.yaml"
SCENES["undistorted5"]="config/undistorted5.yaml"

echo "Start evaluating on underwater datasets..."

for scene in "${!SCENES[@]}"; do
    config="${SCENES[$scene]}"
    dataset_path="${DATASET_ROOT}/${scene}"

    if [ ! -d "$dataset_path" ]; then
        echo "Skipping $scene (not found at $dataset_path)"
        continue
    fi

    echo "Running on $scene with $config ..."
    python main.py --dataset "$dataset_path" --config "$config"
    echo "$scene done!"
    echo "---"
done

echo "All scenes done!"
