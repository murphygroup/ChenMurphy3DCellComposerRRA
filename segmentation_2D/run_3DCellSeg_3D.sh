#!/bin/bash

source /home/hrchen/anaconda3/etc/profile.d/conda.sh
conda activate 3DCellSeg
python /home/hrchen/Documents/Research/hubmap/script/2D-3D/3DCellSeg_wrapper.py $1
conda deactivate

