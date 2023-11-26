import numpy as np
import bz2
import pickle
from skimage.segmentation import find_boundaries
from scipy.sparse import csr_matrix
import pandas as pd
from skimage.io import imsave
import sys
from os.path import join
# from aicsimageio.writers import OmeTiffWriter
import os

def match_3D_slice(mask_3D, cell_mask_3D):
	for z in range(mask_3D.shape[0]):
		for y in range(mask_3D.shape[1]):
			for x in range(mask_3D.shape[2]):
				if mask_3D[z, y, x] != 0:
					mask_3D[z, y, x] = cell_mask_3D[z, y, x]
	return mask_3D


def get_compartments_diff(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	diff = np.array(list(a - b))
	return diff


def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
	# a = set((tuple(i) for i in cell_arr))
	# b = set((tuple(i) for i in cell_membrane_arr))
	# c = set((tuple(i) for i in nuclear_arr))
	
	a = set(zip(cell_arr[0], cell_arr[1], cell_arr[2]))
	b = set(zip(cell_membrane_arr[0], cell_membrane_arr[1], cell_membrane_arr[2]))
	c = set(zip(nuclear_arr[0], nuclear_arr[1], nuclear_arr[2]))
	d = a - b
	mismatch_pixel_num = len(list(c - d))
	# print(mismatch_pixel_num)
	mismatch_fraction = len(list(c - d)) / len(list(c))
	if not mismatch_repair:
		if mismatch_pixel_num == 0:
			return np.array(list(a)), np.array(list(c)), 0
		else:
			return False, False, False
	else:
		if mismatch_pixel_num < len(nuclear_arr[0]):
			return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
		else:
			return False, False, False


def append_coord(rlabel_mask, indices, maxvalue):
	masked_imgs_coord = [[[], []] for i in range(maxvalue)]
	for i in range(0, len(rlabel_mask)):
		masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
		masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])
	return masked_imgs_coord

def unravel_indices(labeled_mask, maxvalue):
	rlabel_mask = labeled_mask.reshape(-1)
	indices = np.arange(len(rlabel_mask))
	indices = np.unravel_index(indices, (labeled_mask.shape[0], labeled_mask.shape[1]))
	masked_imgs_coord = append_coord(rlabel_mask, indices, maxvalue)
	masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))
	return masked_imgs_coord

def get_coordinates(mask):
	print("Getting cell coordinates...")
	cell_num = np.unique(mask)
	maxvalue = len(cell_num)
	channel_coords = unravel_indices(mask, maxvalue)
	return channel_coords

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_pandas(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def show_plt(mask):
	plt.imshow(mask)
	plt.show()
	plt.clf()

def list_remove(c_list, indexes):
	for index in sorted(indexes, reverse=True):
		del c_list[index]
	return c_list

def filter_cells(coords, mask):
	# completely mismatches
	no_cells = []
	for i in range(len(coords)):
		if np.sum(mask[coords[i]]) == 0:
			no_cells.append(i)
	new_coords = list_remove(coords.copy(), no_cells)
	return new_coords

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask, mode='inner')
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def get_mask(cell_list, mask_shape):
	mask = np.zeros((mask_shape))
	for cell_num in range(len(cell_list)):
		# mask[tuple(cell_list[cell_num].T)] = cell_num
		z_coords, x_coords, y_coords = zip(*cell_list[cell_num])
		mask[z_coords, x_coords, y_coords] = cell_num
	return mask

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)

def get_membrane_mask(original_mask):
	slice_boundary_slices = []
	for slice_idx in range(original_mask.shape[0]):
		slice_boundary_slices.append(find_boundaries(original_mask[slice_idx]))
	slice_boundary = np.stack(slice_boundary_slices)
	return get_indexed_mask(original_mask, slice_boundary)
	
def match_repair_cell_nucleus_3D(nuclear_mask, cell_mask):
	
	cell_membrane_mask = get_boundary(cell_mask)
	
	cell_coords = get_indices_pandas(cell_mask)[1:]
	nucleus_coords = get_indices_pandas(nuclear_mask)[1:]
	cell_membrane_coords = get_indices_pandas(cell_membrane_mask)[1:]

	
	# cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
	# cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
	# nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
	cell_matched_index_list = []
	nucleus_matched_index_list = []
	cell_matched_list = []
	nucleus_matched_list = []
	repaired_num = 0
	mismatch_repair = True
	#print(len(np.unique(nuclear_slice)))
	
	
	for i in cell_coords.index:
		# print(i)
		if len(cell_coords[i]) != 0:
			# if len(np.unique(cell_coords[i][0])) >= 3:
			# 	print(i)
			current_cell_coords = cell_coords[i]
			nuclear_search_num = np.unique(nuclear_mask[current_cell_coords]).astype(np.int64)
			# print(nuclear_search_num)
			best_mismatch_fraction = 1
			whole_cell_best = []
			for j in nuclear_search_num:
				#print(j)
				if j != 0:
					if (j not in nucleus_matched_index_list) and (i not in cell_matched_index_list):
						# print(j-1)
						whole_cell, nucleus, mismatch_fraction = get_matched_cells(cell_coords[i], cell_membrane_coords[i], nucleus_coords[j], mismatch_repair=mismatch_repair)
						if type(whole_cell) != bool:
							if mismatch_fraction < best_mismatch_fraction:
								best_mismatch_fraction = mismatch_fraction
								whole_cell_best = whole_cell
								nucleus_best = nucleus
								i_ind = i
								j_ind = j
			if best_mismatch_fraction < 1 and best_mismatch_fraction > 0:
				repaired_num += 1
			
			if len(whole_cell_best) > 0:
				cell_matched_list.append(whole_cell_best)
				nucleus_matched_list.append(nucleus_best)
				cell_matched_index_list.append(i_ind)
				nucleus_matched_index_list.append(j_ind)
	
	cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, nuclear_mask.shape)

	return cell_matched_mask, nuclear_matched_mask


if __name__ == '__main__':
	
	
	file_dir = sys.argv[1]
	method = sys.argv[2]
	JI_thre = sys.argv[3]

	cell_mask_3D = pickle.load(bz2.BZ2File(f'{file_dir}/mask_{method}_matched_3D_final.pkl', 'r'))
	nuclear_mask_3D = pickle.load(bz2.BZ2File(f'{file_dir}/nuclear_mask_{method}_matched_3D_final.pkl', 'r'))
	matched_cell_3D, matched_nuclear_3D = match_repair_cell_nucleus_3D(nuclear_mask_3D,cell_mask_3D)
	
	pickle.dump(matched_cell_3D, bz2.BZ2File(f'{file_dir}/mask_{method}_matched_3D_final_{JI_thre}.pkl', 'w'))
	pickle.dump(matched_nuclear_3D, bz2.BZ2File(f'{file_dir}/nuclear_mask_{method}_matched_3D_final_{JI_thre}.pkl', 'w'))
