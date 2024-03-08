import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave
import bz2
import pickle
import time
import subprocess
if __name__ == '__main__':
	data_dir = '/home/haoranch/projects/HuBMAP/2D-3D/data/AICS'
	img_3D_dir_list = [os.path.join(root, d) for root, dirs, _ in os.walk(data_dir) for d in dirs if d == "original" or d.startswith("random_gaussian_")]
	img_3D_dir_list.sort()
	print(img_3D_dir_list)
	np.save('/home/haoranch/projects/HuBMAP/2D-3D/script/AICS_img_3D_dir_list.npy', img_3D_dir_list)
	methods = ['deepcell_membrane-0.12.6','cellpose-2.2.2']
	methods = ['aics_ml']
	JI_threshold_list = ['0.0']


	miss_count = 0
	for JI_threshold in JI_threshold_list:
		for img_dir in img_3D_dir_list:
			for method in methods:
				file_path = f'{img_dir}/metrics/metrics_{method}_{JI_threshold}.npy'
				if not os.path.exists(file_path) or (time.time() - os.path.getmtime(file_path)) > 20000:
				#if True:
					miss_count += 1
					print(f'srun -p pool1,model1,model2,model3,model4,pool3-bigmem -n 1 -c 4 --mem 20G --exclude=compute-0-30,compute-0-14  python /home/haoranch/projects/HuBMAP/2D-3D/script/match_3D_nuclei.py {img_dir} {method} {JI_threshold} &')

					batch_script = f"""#!/bin/bash
#SBATCH --partition=model1,model2,model3,model4,pool3-bigmem,gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --job-name=3D_eval
#SBATCH --exclude=compute-0-30,compute-0-14

python /home/haoranch/projects/HuBMAP/2D-3D/script/single_method_eval_3D_AICS.py {img_dir} {method} {JI_threshold}
"""
					script_file = '/home/haoranch/projects/HuBMAP/2D-3D/script/batch_script.sh'
					with open(script_file, 'w') as file:
						file.write(batch_script)

					subprocess.run(['sbatch', script_file])









