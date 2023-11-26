import numpy as np
import matplotlib.pyplot as plt
import pickle
import bz2
import pandas as pd
import random
random.seed(12)
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



method = 'deepcell_membrane-0.12.6'
axis = 'YZ'
segmentation = pickle.load(bz2.BZ2File(
	f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_stack_{axis}_0.0.pkl',
	'r')).astype(np.int64)
# segmentation = pickle.load(bz2.BZ2File(
# 	f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.3.pkl',
# 	'r')).astype(np.int64)
# segmentation = segmentation[:10,350:400,350:400]
if axis == 'XY':
	segmentation = segmentation[20:30, 500:550, 500:550]
	segmentation = np.transpose(segmentation, (0, 2, 1))
	segmentation = np.flip(segmentation, axis=2)

elif axis == 'YZ':
	segmentation = segmentation[500:550, 20:30, 500:550]
	segmentation = np.transpose(segmentation, (0, 2, 1))
	segmentation = np.flip(segmentation, axis=0)

elif axis == 'XZ':
	segmentation = segmentation[500:550, 500:550, 20:30]
	segmentation = np.transpose(segmentation, (0, 2, 1))
	segmentation = np.flip(segmentation, axis=(0, 1, 2))

	
cell_coords = get_indices_pandas(segmentation)[1:]

color_map = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'test', 'test2']

cell_colors = color_cells(segmentation, cell_coords, color_map)
segmentation_colored = coloring(segmentation, cell_coords, cell_colors, color_map)


# cell = np.flip(cell, axis=0)

# Create a color array (50x100x100x4) for RGBA values
colors = np.zeros(segmentation_colored.shape + (4,))
colors[segmentation_colored == 1] = [1, 0, 0, 1]  # Red for value 0
colors[segmentation_colored == 2] = [0, 1, 0, 1]  # Green for value 1
colors[segmentation_colored == 3] = [0, 0, 1, 1]  # Blue for value 2
colors[segmentation_colored == 4] = [0, 1, 1, 1]  # Blue for value 2
colors[segmentation_colored == 5] = [1, 0, 1, 1]  # Blue for value 2
colors[segmentation_colored == 6] = [1, 1, 0, 1]  # Blue for value 2
colors[segmentation_colored == 7] = [1.0, 0.5, 0.0, 1]  # Blue for value 2
colors[segmentation_colored == 8] = [0.5, 0.0, 0.5, 1]  # Blue for value 2
colors[segmentation_colored == 9] = [0.6, 0.2, 0.2, 1]  # Blue for value 2
colors[segmentation_colored == 10] = [0.8, 0.4, 0.3, 1]  # Blue for value 2
colors[segmentation_colored == 11] = [0.2, 0.4, 0.3, 1]  # Blue for value 2

# Make sure the arrays are contiguous in memory
segmentation_colored = np.ascontiguousarray(segmentation_colored)
colors = np.ascontiguousarray(colors)

# Start plotting
fig = plt.figure(figsize=(10, 10))  # Increase figure size
ax = fig.add_subplot(111, projection='3d')

# Use ax.voxels to visualize the 3D numpy array with different colors
ax.voxels(segmentation_colored, facecolors=colors, edgecolors=None)  # No edge coloring

if axis == 'XY':
	ax.set_box_aspect([segmentation.shape[0]*2, segmentation.shape[1], segmentation.shape[2]])
elif axis == 'YZ':
	ax.set_box_aspect([segmentation.shape[0], segmentation.shape[1], segmentation.shape[2]*2])
elif axis == 'XZ':
	ax.set_box_aspect([segmentation.shape[0], segmentation.shape[1]*2, segmentation.shape[2]])
ax.view_init(elev=30, azim=-25)
# Hide grid
ax.grid(False)

# Hide ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_frame_on(False)

plt.savefig(f'/home/hrchen/Documents/Research/hubmap/script/2D-3D/fig/2D_stack_{axis}', bbox_inches='tight', pad_inches=0,
            transparent=True)
plt.clf()

