import glob
from os.path import join
import os
import sys

def get_JI_list(data_type):
	JI_list = []
	if data_type == 'IMC_3D':
		JI_range = 8
	elif data_type == 'AICS':
		JI_range = 10
	
	for i in range(0, JI_range, 1):
		value = round(i * 0.1, 1)
		JI_list.append(str(value))
	return JI_list

if __name__ == '__main__':
	data_type = sys.argv[1]
	data_dir = f'../data/{data_type}'
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original*'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	axes = ['XY', 'XZ', 'YZ']
	
	if data_dir == 'IMC_3D':
		methods = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX', 'cellsegm']
		JI_threshold_list = get_JI_list('IMC_3D')
	if data_dir == 'AICS':
		methods = ['deepcell_membrane-0.12.6', 'cellpose-2.2.2']
		JI_threshold_list = get_JI_list('AICS')

	for JI_threshold in JI_threshold_list:
		for img_dir in img_3D_dir_list:
			for method in methods:
				for axis in axes:
					os.system(f'python match_2D_cells.py {img_dir} {method} {axis} {JI_threshold} &')
				os.system(f'python match_3D_cells.py {img_dir} {method} {JI_threshold} &')
				os.system(f'python match_3D_nuclei.py {img_dir} {method} {JI_threshold} &')

