#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: bash mast3r_slam/run_eval.sh <dataset_root> <output_root> <scene1> [scene2 ...]"
  echo "Example: bash mast3r_slam/run_eval.sh WaterSplat_datasets outputs pool_loop"
  exit 1
fi

data_path="$1"
refpose_path="$2"
shift 2
scenes="$@"

echo "Start evaluating trajectories..."

for sc in ${scenes}
do
  echo Running on $sc ...
    python mast3r_slam/evo_eval.py \
    --colmap_pose_dir "${data_path}/${sc}" \
    --ref_pose_dir "${refpose_path}/${sc}/traj_full.txt" \
    --save_dir "${refpose_path}/${sc}" \
    --data_name "${sc}" \
    --image_subdir "images" \
    --sparse_dir "sparse/0"

  echo $sc done!
done

echo "All scenes done!"
