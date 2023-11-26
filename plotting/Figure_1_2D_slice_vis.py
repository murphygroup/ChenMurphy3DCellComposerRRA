import numpy as np
import matplotlib.pyplot as plt
import bz2
import pickle
from matplotlib.colors import ListedColormap
import random
random.seed(0)

def dfs(graph, node, visited, cell_colors):
	visited.add(node)
	available_colors = {1, 2, 3, 4, 5}

	for neighbor in graph[node]:
		if neighbor in visited:
			available_colors.discard(cell_colors.get(neighbor, None))

	if not available_colors:
		print(f"Error: No available colors for node {node}, neighbors: {graph[node]}")
		return
	
	chosen_color = random.choice(list(available_colors))
	# chosen_color = available_colors.pop()
	cell_colors[node] = chosen_color

	for neighbor in graph[node]:
		if neighbor not in visited:
			dfs(graph, neighbor, visited, cell_colors)

def cell_coloring(cell_2D):
	rows, cols = cell_2D.shape
	graph = {}
	for i in range(rows):
		for j in range(cols):
			node = cell_2D[i, j]
			if node == 0:  # Skip background cells
				continue
			if node not in graph:
				graph[node] = set()
			
			for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				new_x, new_y = i + dx, j + dy
				if 0 <= new_x < rows and 0 <= new_y < cols:
					neighbor = cell_2D[new_x, new_y]
					if neighbor == 0:  # Skip background cells
						continue
					if node != neighbor:
						graph[node].add(neighbor)
	
	# Apply DFS to color cells
	visited = set()
	cell_colors = {}
	for node in graph.keys():
		if node not in visited:
			dfs(graph, node, visited, cell_colors)
	
	# Create a new 2D array to store the color codes
	color_2D_array = np.zeros((rows, cols))
	
	for i in range(rows):
		for j in range(cols):
			node = cell_2D[i, j]
			if node in cell_colors:
				color_2D_array[i, j] = cell_colors[node]
			else:
				color_2D_array[i, j] = 0  # Set background color to black
	return color_2D_array
# Load your 2D array
# axes = ['XY', 'YZ', 'XZ']
axes = ['XY']
method = 'deepcell_membrane-0.12.6'
save_dir = '/home/hrchen/Documents/Research/hubmap/script/2D-3D/fig'
for axis in axes:
	cell_2D = pickle.load(bz2.BZ2File(
		f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_{axis}.pkl',
		'r'))
	# cell_2D = pickle.load(bz2.BZ2File(
	# f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_stack_{axis}_0.0.pkl',
	# 'r')).astype(np.int64)
	if axis == 'XY':
		cell_2D = cell_2D[20:30, 300:750, 300:750]
		cell_2D = np.flip(cell_2D, axis=0)
		aspect_ratio = 1
	elif axis == 'YZ':
		cell_2D = cell_2D[500:550, 20:30, 500:550]
		cell_2D = np.flip(cell_2D, axis=1)
		aspect_ratio = 2
	elif axis == 'XZ':
		cell_2D = cell_2D[500:550, 500:550, 20:30]
		cell_2D = np.flip(cell_2D, axis=2)
		aspect_ratio = 0.5
	
	from matplotlib.colors import ListedColormap
	
	border_thickness = 1  # Set the border thickness in pixels
	
	for i in range(1):
		color_2D_array = cell_coloring(cell_2D[i])
		
		# Add a black border around the image array
		h, w = color_2D_array.shape
		bordered_array = np.zeros((h + 2 * border_thickness, w + 2 * border_thickness))
		bordered_array[border_thickness:h + border_thickness, border_thickness:w + border_thickness] = color_2D_array
		
		if 0 in color_2D_array:
			custom_cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
		else:
			custom_cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'purple'])

		# Display the color-coded cells with black border using matplotlib
		plt.imshow(color_2D_array, cmap=custom_cmap, aspect=aspect_ratio)
		
		# Remove tick labels
		plt.tick_params(
			axis='both',  # changes apply to both x and y-axis
			which='both',  # both major and minor ticks are affected
			bottom=False,  # ticks along the bottom edge are off
			left=False,  # ticks along the left edge are off
			labelbottom=False,  # labels along the bottom edge are off
			labelleft=False  # labels along the left edge are off
		)
		
		plt.savefig(f'/home/hrchen/Documents/Research/hubmap/script/2D-3D/fig/2D_{axis}_{i}', bbox_inches='tight', dpi=100,
		            pad_inches=0, transparent=True)
		plt.clf()

#
# from PIL import Image, ImageDraw
#
# # Create the first image with a red square
# img1 = Image.new('RGB', (100, 100), color='white')
# draw1 = ImageDraw.Draw(img1)
# draw1.rectangle([20, 20, 80, 80], fill='red')
#
# # Create the second image with a blue square
# img2 = Image.new('RGB', (100, 100), color='white')
# draw2 = ImageDraw.Draw(img2)
# draw2.rectangle([20, 20, 80, 80], fill='blue')
#
# # Create a new empty white image
# final_img = Image.new('RGB', (120, 120), color='white')
#
# # Paste the first image, slightly offset
# final_img.paste(img1, (10, 10))
#
# # Paste the second image, with more offset to create a "shadow" or "3D" effect
# final_img.paste(img2, (20, 20))
#
# # Save the final image
# final_img.save('/home/hrchen/Documents/Research/hubmap/script/2D-3D/fig/stacked_image.jpg')
#
#





