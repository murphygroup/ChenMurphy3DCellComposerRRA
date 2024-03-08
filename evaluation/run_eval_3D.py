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
	data_dir = './data/IMC_3D'
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original*'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	methods = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX', 'cellsegm']
	JI_threshold_list = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7']

	miss_count = 0
	for JI_threshold in JI_threshold_list:
		for img_dir in img_3D_dir_list:
			for method in methods:
				file_path = f'{img_dir}/metrics/metrics_{method}_{JI_threshold}.npy'
				if not os.path.exists(file_path) or (time.time() - os.path.getmtime(file_path)) > 18400:
				#if True:
					miss_count += 1

					batch_script = f"""#!/bin/bash
#SBATCH --partition=pool1,model1,model2,model3,model4,pool3-bigmem,gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=46G
#SBATCH --job-name=3D_eval
#SBATCH --exclude=compute-0-30,compute-0-14

python ./single_method_eval_3D.py {img_dir} {method} {JI_threshold}
"""
					script_file = './batch_script.sh'
					with open(script_file, 'w') as file:
						file.write(batch_script)

					subprocess.run(['sbatch', script_file])
				#else:
				#	print(f"Skipping submission for {file_path} as it exists and is recent.")








