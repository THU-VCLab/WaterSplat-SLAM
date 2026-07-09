#!/bin/bash

SEATHRU_ROOT=${1:?"Usage: bash scripts/run_our_data.sh <seathru_root> <watersplat_root>"}
WATERSPLAT_ROOT=${2:?"Usage: bash scripts/run_our_data.sh <seathru_root> <watersplat_root>"}

declare -A SCENES
SCENES["${SEATHRU_ROOT}/4_Curasao"]="config/Curasao.yaml"
SCENES["${SEATHRU_ROOT}/5_IUI3-RedSea"]="config/RedSea.yaml"
SCENES["${SEATHRU_ROOT}/6_JapaneseGradens-RedSea"]="config/Jap_RedSea.yaml"
SCENES["${SEATHRU_ROOT}/7_Panama"]="config/Panama.yaml"

SCENES["${WATERSPLAT_ROOT}/big_gate"]="config/big_gate.yaml"
SCENES["${WATERSPLAT_ROOT}/pipe_local"]="config/pipe_local.yaml"
SCENES["${WATERSPLAT_ROOT}/pool_up2"]="config/pool_up2.yaml"
SCENES["${WATERSPLAT_ROOT}/pool_loop"]="config/pool_loop.yaml"

echo "Start evaluating on underwater datasets..."

for scene in "${!SCENES[@]}"; do
    config="${SCENES[$scene]}"

    if [ ! -d "$scene" ]; then
        echo "Skipping $scene (directory not found)"
        continue
    fi

    echo "Running on $scene with $config ..."
    python main.py --dataset "$scene" --config "$config"
    echo "$scene done!"
    echo "---"
done

echo "All scenes done!"
