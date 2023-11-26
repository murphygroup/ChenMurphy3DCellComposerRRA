#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepcell-0.12.6
python /home/haoranch/projects/HuBMAP/2D-3D/script/deepcell_wrapper_membrane-0.12.6_3D.py $1 $2
conda deactivate




