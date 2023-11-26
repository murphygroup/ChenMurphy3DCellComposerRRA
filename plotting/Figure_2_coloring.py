import numpy as np
import pickle
import bz2
import pandas as pd
import random
random.seed(12345)

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)

def find_neighbors(segmentation, i, j, k):
	# Get the label of the cell at the current voxel
	current_label = segmentation[i, j, k]
	
	# List of possible neighbors' coordinates
	neighbors_coords = [(i - 1, j, k), (i + 1, j, k),
	                    (i, j - 1, k), (i, j + 1, k),
	                    (i, j, k - 1), (i, j, k + 1)]
	
	# Retrieve the labels of the neighbors
	neighbors = set()
	for x, y, z in neighbors_coords:
		if 0 <= x < segmentation.shape[0] and 0 <= y < segmentation.shape[1] and 0 <= z < segmentation.shape[2]:
			if segmentation[x, y, z] != current_label:
				neighbors.add(segmentation[x, y, z])
	
	return neighbors


def color_cells(segmentation, cell_coords, cmap):
	# Unique labels representing different cells
	# labels = np.unique(segmentation)
	# labels = labels[labels != 0]  # Assuming 0 is background or not a cell
	
	# Dictionary to store color assignments
	color_assignments = {}
	
	# Define the colors, you can use more if needed

	cell_num = 1
	for label in cell_coords.index:
		print(cell_num)
		cell_num += 1
		available_colors = set(cmap)
		
		# For each voxel in the segmentation, find neighbors and reduce available colors
		voxel_indices = np.vstack(cell_coords[label]).T
		for i, j, k in voxel_indices:
			neighbors = find_neighbors(segmentation, i, j, k)
			for neighbor in neighbors:
				if neighbor in color_assignments:
					available_colors.discard(color_assignments[neighbor])
		
		# Assign the first available color to the current label
		chosen_color = random.choice(list(available_colors))
		color_assignments[label] = chosen_color
		
	return color_assignments

def coloring(seg, coords, cols, cmap):
	seg_colored = np.zeros(seg.shape)
	for cell_idx in coords.index:
		seg_colored[coords[cell_idx]] = cmap.index(cols[cell_idx]) + 1
	return seg_colored

# Example 3D numpy array
# color_map = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown']
methods = ['deepcell_membrane-0.12.6', 'CellProfiler', 'cellpose-2.2.2_2D-3D', '3DCellSeg']
# methods = []

for method in methods:
	color_map = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
	try:
		segmentation = pickle.load(bz2.BZ2File(f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.3.pkl', 'r')).astype(np.int64)
	except:
		segmentation = pickle.load(bz2.BZ2File(f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.0.pkl', 'r')).astype(np.int64)
	segmentation = segmentation[:,350:450,350:450]
	cell_coords = get_indices_pandas(segmentation)[1:]
	
	cell_colors = color_cells(segmentation, cell_coords, color_map)
	segmentation_colored = coloring(segmentation, cell_coords, cell_colors, color_map)
	pickle.dump(segmentation_colored, bz2.BZ2File(f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_colored.pkl', 'w'))
	
	print(cell_colors)
