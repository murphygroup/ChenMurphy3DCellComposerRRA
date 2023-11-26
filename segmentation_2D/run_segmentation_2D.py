import os
import numpy as np
from os.path import join
import argparse
import glob
import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave

import numpy as np
import glob
from os.path import join
import os
from skimage.io import imread, imsave
import bz2
import pickle


def extract_number(s):
	return int(''.join(filter(str.isdigit, s)))

def combine_slices(data_dir):
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original'), recursive=True)) + sorted(
		glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	# axes = ['XY', 'XZ', 'YZ']
	axes = ['XY', 'XZ', 'YZ']
	methods = ['aics_classic', 'cellpose', 'CellProfiler', 'CellX', 'cellsegm']
	
	for img_dir in img_3D_dir_list:
		img_shape = imread(join(img_dir, 'nucleus.tif')).shape
		
		for method in methods:
			for axis in axes:
				slice_shape = imread(f'{img_dir}/slices/slice_{axis}_0/nucleus.tif').shape
				
				if True:
					img_3D_pieces = list()
					slice_dir_list = sorted(glob.glob(f'{img_dir}/slices/slice_{axis}*', recursive=True),
					                        key=extract_number)
					for slice_dir in slice_dir_list:
						try:
							img_slice = pickle.load(bz2.BZ2File(f'{slice_dir}/nuclear_mask_{method}.pkl', 'r'))
							# print(slice_dir)
							if img_slice.shape != slice_shape:
								print(slice_dir)
						except:
							print(slice_dir)
							img_slice = np.zeros(slice_shape)
						img_3D_pieces.append(img_slice)
					img_3D = np.stack(img_3D_pieces, axis=0)
					print(img_3D.shape)
					print(len(np.unique(img_3D)))
					imsave(f'{img_dir}/nuclear_mask_{method}_{axis}.tif', img_3D)
					pickle.dump(img_3D, bz2.BZ2File(f'{img_dir}/nuclear_mask_{method}_{axis}.pkl', 'w'))
		
def split_slices(data_dir):
	img_3D_dir_list = sorted(glob.glob(join(data_dir, '**', 'original'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'random_gaussian_*'), recursive=True))
	img_names = ['nucleus', 'cytoplasm', 'membrane']
	axes = ['XY','XZ','YZ']
	for img_dir in img_3D_dir_list:
		for img_name in img_names:
			print(join(img_dir, img_name + '.tif'))
			img = imread(join(img_dir, img_name + '.tif'))
			for axis in axes:
				if axis == 'XY':
					img_rotated = img.copy()
				elif axis == 'XZ':
					img_rotated = np.rot90(img, k=1, axes=(2, 0))
				elif axis == 'YZ':
					img_rotated = np.rot90(img, k=1, axes=(1, 0))
				for slice_index in range(img_rotated.shape[0]):
					img_slice = img_rotated[slice_index]
					print(img_slice.shape)
					# os.system('rm -rf ' + join(img_dir, str(axis) + '_' + str(slice_index)))
					slice_dir = join(img_dir, 'slice_' + str(axis) + '_' + str(slice_index))
					if not os.path.exists(slice_dir):
						os.makedirs(slice_dir)
					save_dir = join(slice_dir, img_name + '.tif')
					imsave(save_dir, img_slice.astype(np.uint16))
				
def segmentation_2D(data_dir):
	split_slices(data_dir)
	axes = ['XY', 'XZ', 'YZ']
	#axes = ['XZ']
	for axis in axes:
		img_files = sorted(glob.glob(f'{data_dir}/**/slice_{axis}*', recursive=True))
		if axis == 'XY':
			pixel_size = '1000'
		else:
			pixel_size = '2000'
		percentage = '100'
		print(len(img_files))
		total_tasks = len(img_files)
		chunk_size = 1000
		total_chunks = (total_tasks + chunk_size - 1) // chunk_size
		print(total_chunks)
		for i in range(total_chunks):
			start_idx = i * chunk_size
			end_idx = min(start_idx + chunk_size, total_tasks)
			with open(f'./segmentation_2D/img_files_list_{axis}_{i}.txt', 'w') as f:
				for img in img_files[start_idx:end_idx]:
					f.write(img + '\n')
			if end_idx == total_tasks:
				end_idx_batch = total_tasks - 1000 * (total_chunks-1)
			else:
				end_idx_batch = 1000
			print(end_idx_batch)
			os.system(f'sbatch --array=1-{end_idx_batch} -p short1,pool1,pool3-bigmem,model1,model2,model3,model4 --mem 5G -o ./segmentation_2D/output_%A_%a.out ./segmentation_2D/run_2D_slice_segmentation.sh {axis} {i} {percentage} {pixel_size}')
		
	combine_slices(data_dir)
