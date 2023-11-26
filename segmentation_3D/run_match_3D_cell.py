import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave
import bz2
import pickle
import matplotlib.pyplot as plt
if __name__ == '__main__':
	data_dir = '/home/haoranch/projects/HuBMAP/IMC_3D/florida-3d-imc/'
	#img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	img_3D_dir_list = sorted(glob.glob(data_dir + '/a296c763352828159f3adfa495becf3e/random*')) + sorted(glob.glob(data_dir + '/cd880c54e0095bad5200397588eccf81/random*')) + sorted(glob.glob(data_dir + '/d3130f4a89946cc6b300b115a3120b7a/random*'))
	#methods = ['artificial', 'aics_classic', 'cellpose', 'CellProfiler', 'CellX', 'cellsegm', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'cellpose_new', 'cellpose', 'deepcell_membrane', 'deepcell_cytoplasm']
	methods = ['artificial']
	for img_dir in img_3D_dir_list:
		for method in methods:
			if not os.path.exists(join(img_dir, 'mask_' + method + '_matched_3D_no_bits.npy')):
				os.system('srun -p model1,model2,model3,model4,pool1,pool3-bigmem,gpu,short1,interactive -n 1 -c 4 --mem 10G python /home/haoranch/projects/HuBMAP/match_3D_cells.py ' + img_dir + ' ' + method + ' &')
		# 		break
		# 	break
		# break
