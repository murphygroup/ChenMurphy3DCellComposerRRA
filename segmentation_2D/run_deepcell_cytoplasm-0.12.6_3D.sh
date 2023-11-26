#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepcell-0.12.6
python /home/hrchen/Documents/Research/hubmap/script/2D-3D/deepcell_wrapper_cytoplasm-0.12.6_3D.py $1 $2
conda deactivate




