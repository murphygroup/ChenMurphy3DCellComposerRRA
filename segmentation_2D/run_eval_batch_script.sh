#!/bin/bash
#SBATCH --partition=pool1,model1,model2,model3,model4,pool3-bigmem,gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --job-name=match_3D_nuclei_random_gaussian_3_aics_ml_00

python /home/haoranch/projects/HuBMAP/2D-3D/script/match_3D_nuclei.py /home/haoranch/projects/HuBMAP/2D-3D/data/AICS/AICS_tublin/AICS_911/random_gaussian_3 aics_ml 0.0
