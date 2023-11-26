#!/bin/bash

# Assuming SLURM_ARRAY_TASK_ID is set
IMG_INDEX=$SLURM_ARRAY_TASK_ID

# Convert the index to the corresponding image path
IMG_PATH=$(sed "${IMG_INDEX}q;d" /home/haoranch/projects/HuBMAP/2D-3D/script/img_files_list_${1}_${2}.txt)


echo "$IMG_PATH"
echo "$1"
echo "$2"
echo "$3"
echo "$4"

bash /home/haoranch/projects/HuBMAP/2D-3D/script/run_aics_classic.sh "$IMG_PATH" "$3" "$4"
bash /home/haoranch/projects/HuBMAP/2D-3D/script/run_CellProfiler.sh "$IMG_PATH" "$3" "$4"
bash /home/haoranch/projects/HuBMAP/2D-3D/script/run_cellsegm.sh "$IMG_PATH" "$3" "$4"
bash /home/haoranch/projects/HuBMAP/2D-3D/script/run_CellX.sh "$IMG_PATH" "$3" "$4"


