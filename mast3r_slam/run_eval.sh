#!/bin/bash

# scenes="office0 office1 office2 office3 office4 room0 room1 room2"
# scenes="big_gate Cursao JapRedSea panama RedSea pipe_local pool_up pool_up2 undistorted5"#traj_kf.txt
scenes="pool_up pool_up2 undistorted5 5_pool"
# scenes="undistorted5"
data_path="/home/robolab/Watersplatting-SLAM/ours_underwater/17_blue_rov"
refpose_path="/home/robolab/HI-SLAM2/outputs"

echo "Start evaluating on Sea dataset..."

for sc in ${scenes}
do
  echo Running on $sc ...
    python evo_eval.py \
    --colmap_pose_dir "${data_path}/${sc}" \
    --ref_pose_dir "${refpose_path}/${sc}/traj_full.txt" \
    --save_dir "${refpose_path}/${sc}" \
    --data_name "${sc}" \
    --image_subdir "images" \
    --sparse_dir "sparse/0"

  echo $sc done!
done

echo "All scenes done!"