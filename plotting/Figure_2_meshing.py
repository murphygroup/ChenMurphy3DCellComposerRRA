import numpy as np
import bz2
import pickle
from skimage import measure
from skimage.io import imsave
import pandas as pd

print("creating blender's mesh files for Figure 2...")

methods = ['deepcell_membrane-0.12.6', 'CellProfiler', 'cellpose-2.2.2_2D-3D', '3DCellSeg']
# methods = ['deepcell_membrane-0.12.6']

def write_to_mtl(materials, filename):
	with open(filename, 'w') as f:
		for material_id, color in materials.items():
			# print(material_id)
			f.write(f'newmtl Color_{material_id}\n')
			f.write(f'Kd {color[0]} {color[1]} {color[2]}\n')  # Diffuse color


def write_to_obj(verts, faces, groups, colors, filename):
	with open(filename, 'w') as f:
		f.write(
			f'mtllib ../data/masks/IMC_3D/florida-3d-imc/d3130f4a89946cc6b300b115a3120b7a/original/cell_mesh_{method}.mtl\n')
		
		# Write vertices
		for v in verts:
			f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
		
		# Keep track of the last group number to identify when to change groups
		last_group = None
		
		# Write faces
		for face_idx in range(len(faces)):
			# Take the value from the first vertex in the face as the group number
			face = faces[face_idx]
			group = groups[face_idx][0]
			color = colors[face_idx][0]
			
			# If this is a new group, write a new group tag
			if group != last_group:
				f.write(f'g Cell_{group}\n')
				last_group = group
				# print(group)
				f.write(f'usemtl Color_{color}\n')
				# if color == 0:
				# 	print(face_idx)
			
			# print(group_number != last_group)
			# Write the face
			f.write('f {0} {1} {2}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


# print(face)


def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)

# Extract the 2D contours (meshes) from the slices
def get_2D_mesh(slice_data, level=1.9):
	return measure.find_contours(slice_data, level=level)

# Convert 2D contours to 3D
def convert_2D_contour_to_3D(contour, z_value):
	return np.hstack((np.full((contour.shape[0], 1), z_value), contour))

# Triangulate the 2D contours to create triangles from them
def triangulate_2D_contour(contour):
	triangles = []
	for i in range(1, len(contour) - 1):
		triangles.append([0, i, i + 1])
	return triangles

for method in methods:
	try:
		mask = pickle.load(bz2.BZ2File(
			f'../data/masks/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.3.pkl',
			'r')).astype(np.int64)
	except:
		mask = pickle.load(bz2.BZ2File(
			f'../data/masks/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.0.pkl',
			'r')).astype(np.int64)
	mask_colored = pickle.load(bz2.BZ2File(
		f'../data/masks/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_colored.pkl',
		'r')).astype(np.int64)
	# data = np.zeros(mask.shape, dtype=np.int16)
	# cell_id_max = len(cell_coords)
	# cell_id = 1
	# for i in cell_coords.index:
	# 	data[cell_coords[i]] = cell_id
	# 	cell_id += 1
	data = mask[:,350:450,350:450]
	cell_coords = get_indices_pandas(data)[1:]
	
	# imsave(f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.3.tif', data.astype(np.int16))
	
	# cell_coords = get_indices_pandas(data)[1:]
	
	all_verts = []
	all_faces = []
	all_values = []
	all_groups = []
	all_colors = []
	offset = 0  # To keep track of the index offset for faces when combining multiple cells
	
	for cell_index in cell_coords.index:
		# print(cell_index)
		current_coords = cell_coords[cell_index]
		current_mask = np.zeros(data.shape)
		current_mask[current_coords] = 2
		
		# 3D mesh
		current_color = mask_colored[current_coords][0]
		verts, faces, normals, values = measure.marching_cubes(current_mask, level=1.999)
		faces += offset
		offset += len(verts)

		# 2D mesh for start and end slice to close the hole
		if len(np.unique(current_coords[0])) >= 2:
			z_start = np.min(current_coords[0])
			if np.sum(current_mask[z_start] != 0) > np.sum(current_mask[z_start+1] != 0):
				start_slice_mesh = get_2D_mesh(current_mask[z_start])
				start_2D_contours = [convert_2D_contour_to_3D(contour, z_start) for contour in start_slice_mesh]
				start_triangles = [triangulate_2D_contour(contour) for contour in start_2D_contours]
				start_2D_contours = np.vstack(start_2D_contours)
				start_triangles = np.vstack(start_triangles)
				verts = np.vstack([verts, start_2D_contours])
				start_triangles += offset
				offset += len(start_2D_contours)
				faces = np.vstack([faces, start_triangles])
			
			z_end = np.max(current_coords[0])
			if np.sum(current_mask[z_end] != 0) > np.sum(current_mask[z_end-1] != 0):
				end_slice_mesh = get_2D_mesh(current_mask[z_end])
				end_2D_contours = [convert_2D_contour_to_3D(contour, z_end) for contour in end_slice_mesh]
				end_triangles = [triangulate_2D_contour(contour) for contour in end_2D_contours]
				end_2D_contours = np.vstack(end_2D_contours)
				end_triangles = np.vstack(end_triangles)
				verts = np.vstack([verts, end_2D_contours])
				end_triangles += offset
				offset += len(end_2D_contours)
				faces = np.vstack([faces, end_triangles])
		
		# Append the new vertices, faces, and values to the main list
		all_verts.extend(verts)
		all_faces.extend(faces)
		all_values.extend(values)
		all_groups.extend(np.repeat(cell_index, len(faces)))
		all_colors.extend(np.repeat(current_color, len(faces)))
		# all_colors.extend(np.random.randint(1, 5, len(faces)))
		# Update the offset for the next iteration

	all_verts = np.vstack(all_verts)
	all_faces = np.vstack(all_faces)
	all_groups = np.vstack(all_groups)
	all_colors = np.vstack(all_colors)
	
	color_map = {
		1: (1.0, 0.0, 0.0),
		2: (0.0, 1.0, 0.0),
		3: (0.0, 0.0, 1.0),
		4: (1.0, 1.0, 0.0),
		5: (0.0, 1.0, 1.0),
		6: (1.0, 0.0, 1.0),
		7: (1.0, 0.5, 0.0),
		8: (0.5, 0.0, 0.5),
		9: (0.6, 0.2, 0.2)
	}
	write_to_mtl(color_map, f"../fig/cell_mesh_{method}.mtl")
	
	
	
	write_to_obj(all_verts, all_faces, all_groups, all_colors,
				 f"../fig/Fig_2_cell_mesh_{method}.obj")
	
print('completed!')