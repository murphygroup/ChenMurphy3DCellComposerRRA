import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave
import bz2
import pickle
import matplotlib.pyplot as plt
if __name__ == '__main__':
	data_dir = '/home/haoranch/projects/HuBMAP/2D-3D/data/IMC_3D'
	#data_dir = '/home/haoranch/projects/HuBMAP/2D-3D/data/AICS'
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original*'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	#methods = ['artificial', 'aics_classic', 'cellpose', 'CellProfiler', 'CellX', 'cellsegm', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'cellpose_new', 'cellpose', 'deepcell_membrane', 'deepcell_cytoplasm']
	methods = ['deepcell_membrane-0.12.6']
	for img_dir in img_3D_dir_list[:1]:
		for method in methods:
			#if not os.path.exists(join(img_dir, 'mask_' + method + '_matched_3D.pkl')):
			if True:
				print('srun -p pool1,model1,model2,model3,model4,pool3-bigmem -n 1 -c 4 --mem 20G python /home/haoranch/projects/HuBMAP/2D-3D/script/match_3D_cells.py ' + img_dir + ' ' + method + ' &')
				os.system('srun -p pool1,model1,model2,model3,model4,pool3-bigmem -n 1 -c 4 --mem 46G python /home/haoranch/projects/HuBMAP/2D-3D/script/match_3D_cells.py ' + img_dir + ' ' + method + ' &')
