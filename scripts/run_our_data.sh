#!/bin/bash

scenes="undistorted5"
echo "Start evaluating on underwater dataset..."

for sc in ${scenes}
do
  echo Running on $sc ...
    python main.py --dataset ours_underwater/17_blue_rov/${sc} --config config/base.yaml --no-viz
  echo $sc done!
done

echo "All scenes done!"
