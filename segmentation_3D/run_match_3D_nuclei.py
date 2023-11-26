import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave
import bz2
import pickle
import matplotlib.pyplot as plt
if __name__ == '__main__':
	#data_dir = '/home/haoranch/projects/HuBMAP/2D-3D/data/IMC_3D'
	data_dir = '/home/haoranch/projects/HuBMAP/2D-3D/data/AICS'
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original*'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	axes = ['XY', 'XZ', 'YZ']
	#methods = ['artificial', 'aics_classic', 'cellpose', 'CellProfiler', 'CellX', 'cellsegm', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'cellpose_new', 'cellpose', 'deepcell_membrane', 'deepcell_cytoplasm']
	methods = ['deepcell_membrane-0.12.6']
	#methods = ['artificial', 'aics_classic', 'CellProfiler', 'CellX', 'cellsegm']
	JI_threshold_list = ['0.3','0.1','0.5']
	#JI_threshold_list = ['0.3']
	for JI_threshold in JI_threshold_list[:1]:
		for img_dir in img_3D_dir_list[:1]:
			for method in methods[:1]:
				if True: 
						if not os.path.exists(f'{img_dir}/mask_{method}_matched_3D_final_{JI_threshold}.pkl'):
						#if True:
							print(f'{img_dir}/mask_{method}_matched_3D_final_{JI_threshold}.pkl')
							#os.system(f'srun -p short1,pool1,model1,model2,model3,model4,pool3-bigmem,gpu -t 72:00:00 -n 1 -c 4 --mem 20G python /home/haoranch/projects/HuBMAP/2D-3D/script/match_3D_nuclei.py {img_dir} {method} {JI_threshold} &')
				# 		break
				# 	break
				# break
