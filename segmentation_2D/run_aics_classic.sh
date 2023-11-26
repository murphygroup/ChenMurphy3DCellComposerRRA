#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate aics_classic
python /home/haoranch/projects/HuBMAP/2D-3D/script/aics_classic_wrapper.py $1 $2 $3
conda deactivate


