import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pandas as pd
from skimage.segmentation import find_boundaries
import pickle
import bz2

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)


def get_volumes_from_indices(series):
	return series.apply(lambda x: len(x[0]))
	
if __name__ == '__main__':
 
	data_dir = sys.argv[1]
	method = sys.argv[2]
	JI_thre = sys.argv[3]

	mask_XY = pickle.load(bz2.BZ2File(f'{data_dir}/mask_{method}_matched_stack_XY_{JI_thre}.pkl', 'r'))
	mask_XZ = pickle.load(bz2.BZ2File(f'{data_dir}/mask_{method}_matched_stack_XZ_{JI_thre}.pkl', 'r'))
	mask_YZ = pickle.load(bz2.BZ2File(f'{data_dir}/mask_{method}_matched_stack_YZ_{JI_thre}.pkl', 'r'))
	
	mask_XY_2D_slices = pickle.load(bz2.BZ2File(f'{data_dir}/mask_{method}_XY.pkl', 'r'))
	mask_XZ = np.rot90(mask_XZ, k=1, axes=(0, 2))
	mask_YZ = np.rot90(mask_YZ, k=1, axes=(0, 1))

	X_max = np.max(mask_YZ) + 1
	Y_max = np.max(mask_XZ) + 1
	Z_max = np.max(mask_XY) + 1
	segmentation = np.zeros(mask_XY.shape, dtype=np.int64)
	index_list = [0]
	index_cumulative = [0]
	X_list = [0]
	Y_list = [0]
	Z_list = [0]
	
	for z in range(mask_XY.shape[0]):
		for x in range(mask_XY.shape[1]):
			for y in range(mask_XY.shape[2]):
				Z = mask_XY[z, x, y]
				Y = mask_XZ[z, x, y]
				X = mask_YZ[z, x, y]
				if Z == 0:
					index_1D = 0
				else:
					index_1D = Y + X * Y_max + Z * X_max * Y_max
				segmentation[z, x, y] = index_1D
	
	pickle.dump(segmentation, bz2.BZ2File(f'{data_dir}/mask_{method}_nonrepaired_3D_{JI_thre}.pkl', 'w'))	
	
	cell_coords = get_indices_pandas(segmentation)[1:]
	
	
	cell_volumes = get_volumes_from_indices(cell_coords)
	sorted_cells = sorted(cell_volumes.keys(), key=lambda x: cell_volumes[x], reverse=True)

	XY_coords = get_indices_pandas(mask_XY.astype(int))

	segmentation_XY_repaired = np.zeros(segmentation.shape, dtype=np.int64)
	segmentation_XY_repaired_binary = np.zeros(segmentation.shape)

	for cell_index in sorted_cells:
		current_coords = cell_coords[cell_index]
		if not np.any(segmentation_XY_repaired_binary[current_coords]) and len(np.unique(current_coords[0])) > 3:
			Z = int(cell_index // (X_max * Y_max))
			XY_coords_Z = XY_coords[Z]
			z_min = min(current_coords[0])
			z_max = max(current_coords[0])
			x_min = min(XY_coords_Z[1])
			x_max = max(XY_coords_Z[1])
			y_min = min(XY_coords_Z[2])
			y_max = max(XY_coords_Z[2])
			impute_range = np.where((XY_coords_Z[0] >= z_min) & (XY_coords_Z[0] <= z_max) & (XY_coords_Z[1] >= x_min) & (XY_coords_Z[1] <= x_max) & (XY_coords_Z[2] >= y_min) & (XY_coords_Z[2] <= y_max))
			XY_coords_Z = list(XY_coords_Z)
			XY_coords_Z[0] = XY_coords_Z[0][impute_range]
			XY_coords_Z[1] = XY_coords_Z[1][impute_range]
			XY_coords_Z[2] = XY_coords_Z[2][impute_range]
			XY_coords_Z = tuple(XY_coords_Z)
			segmentation_XY_repaired[XY_coords_Z] = cell_index
			segmentation_XY_repaired_binary[XY_coords_Z] = 1
		else:
			segmentation_XY_repaired_binary[current_coords] = 1
	segmentation_XY_repaired_output = np.zeros(segmentation_XY_repaired.shape, dtype=int)
	segmentation_XY_repaired_coords = get_indices_pandas(segmentation_XY_repaired)[1:]
	cell_idx = 1
	for coord_idx in segmentation_XY_repaired_coords.index:
		segmentation_XY_repaired_output[segmentation_XY_repaired_coords[coord_idx]] = cell_idx
		cell_idx += 1
	pickle.dump(segmentation_XY_repaired, bz2.BZ2File(f'{data_dir}/mask_{method}_matched_3D_{JI_thre}.pkl', 'w'))
	

