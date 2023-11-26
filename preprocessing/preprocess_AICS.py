from skimage.io import imread, imsave
import numpy as np
import glob
import os
from os.path import join
import bz2
import pickle

markers = ['golgi', 'mito', 'actin', 'lysosome', 'tublin']


if __name__ == '__main__':
	for marker in markers:
		data_dir_list = sorted(glob.glob(f'{os.getcwd()}/data/AICS/AICS_{marker}/**/AICS-*ome.tif', recursive=True))
		for data_dir in data_dir_list:
			# if not os.path.exists(join(os.path.dirname(data_dir), 'membrane.tif')):
			if True:
				image = imread(data_dir)
				membrane = image[:, 0, :, :]
				cytoplasm = image[:, 1, :, :]
				nucleus = image[:, 2, :, :]
				cell_mask = image[:, 5, :, :]
				nuclear_mask = image[:, 6, :, :]
				if not os.path.exists(join(os.path.dirname(data_dir), 'original')):
					os.makedirs(join(os.path.dirname(data_dir), 'original'))
				imsave(os.path.dirname(data_dir) + '/original/' + 'nucleus.tif', nucleus)
				imsave(os.path.dirname(data_dir) + '/original/' + 'cytoplasm.tif', cytoplasm)
				imsave(os.path.dirname(data_dir) + '/original/' + 'membrane.tif', membrane)
				# imsave(os.path.dirname(data_dir) + '/original/' + 'mask_aics_ml_matched_3D_final.tif', cell_mask)
				# imsave(os.path.dirname(data_dir) + '/original/' + 'nuclear_mask_aics_ml_matched_3D_final.tif', nuclear_mask)
				pickle.dump(cell_mask, bz2.BZ2File(os.path.dirname(data_dir) + '/original/' + 'mask_aics_ml_matched_3D_final.pkl','w'))
				pickle.dump(nuclear_mask, bz2.BZ2File(os.path.dirname(data_dir) + '/original/' + 'nuclear_mask_aics_ml_matched_3D_final.pkl', 'w'))
		
		