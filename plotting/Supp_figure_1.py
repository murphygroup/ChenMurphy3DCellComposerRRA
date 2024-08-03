import pickle
import bz2
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

JI_threshold = 0.2
IMCpath = './data/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e'
IMAGEpath = IMCpath

mask_2D_XY_stack = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_aics_classic_matched_3D_final_0.0.pkl', 'rb'))
mask_2D_XY_stack[mask_2D_XY_stack > 0] = 1

mask_2D_XY_stack = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_deepcell_membrane-0.12.6_matched_stack_XY_{JI_threshold}.pkl', 'rb'))
mask_2D_XZ_stack = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_deepcell_membrane-0.12.6_matched_stack_XZ_{JI_threshold}.pkl', 'rb'))
mask_2D_YZ_stack = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_deepcell_membrane-0.12.6_matched_stack_YZ_{JI_threshold}.pkl', 'rb'))
mask_2D_XZ_stack = np.rot90(mask_2D_XZ_stack, k=1, axes=(0, 2))
mask_2D_YZ_stack = np.rot90(mask_2D_YZ_stack, k=1, axes=(0, 1))
mask_no_repair = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_deepcell_membrane-0.12.6_nonrepaired_3D_{JI_threshold}.pkl', 'rb'))
mask_repair = pickle.load(bz2.BZ2File(f'{IMAGEpath}/original/mask_deepcell_membrane-0.12.6_matched_3D_{JI_threshold}.pkl', 'rb'))

def create_slice_fig_new(volume, slice_type, index=None, repaired_coords=None, fragment_idx=None):
    if repaired_coords is not None:
        volume_copy = np.zeros_like(volume)
        volume_copy[repaired_coords] = volume[repaired_coords]
        volume = volume_copy
    XZ = volume[:, index[0], :]
    YZ = volume[:, :, index[1]]

    def reindex_and_convert(array):
        if len(np.unique(array)) > 2:
            unique_values = np.unique(array[array > 0])
            reindex_map = {0: 0, fragment_idx: 2}
            reindex_map.update({value: i + 1 for i, value in enumerate(unique_values) if value != fragment_idx})
            result = np.vectorize(lambda x: reindex_map.get(x, 0), otypes=[int])(array)
            return result
        else:
            return np.where(array > 0, 1, 0).astype(int)

    binary_XZ = reindex_and_convert(XZ).astype(np.uint8)
    binary_YZ = reindex_and_convert(YZ).astype(np.uint8)

    custom_colors = ['white', 'darkblue', 'cyan']
    cmap = matplotlib.colors.ListedColormap(custom_colors)

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    aspect_ratio_XZ = binary_XZ.shape[1] / binary_XZ.shape[0]
    aspect_ratio_YZ = binary_YZ.shape[1] / binary_YZ.shape[0]

    base_height = 4
    width_XZ = base_height * aspect_ratio_XZ
    width_YZ = base_height * aspect_ratio_YZ
    fig_width = max(width_XZ, width_YZ)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 2 * base_height))
    ax1.imshow(binary_XZ, cmap=cmap, origin='lower', aspect='equal')
    ax1.set_title('XZ View', fontsize=40)
    ax1.axis('off')

    ax2.imshow(binary_YZ, cmap=cmap, origin='lower', aspect='equal')
    ax2.set_title('YZ View', fontsize=40)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'/home/hrchen/Documents/Research/hubmap/script/2D-3D/fig/supp_fig_1_2D_{slice_type}.png', dpi=300)

def isolate_cell(segmentation, cell_index):
    isolated_cell = np.zeros_like(segmentation)
    isolated_cell[segmentation == cell_index] = cell_index
    cell_coords = np.where(isolated_cell == cell_index)
    isolated_cell = isolated_cell[min(cell_coords[0]):max(cell_coords[0])+1,
                                  min(cell_coords[1]):max(cell_coords[1])+1,
                                  min(cell_coords[2]):max(cell_coords[2])+1]
    return isolated_cell, cell_coords

def crop_mask(mask, bounding_box):
    cropped_mask = mask[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3], bounding_box[4]:bounding_box[5]]
    return cropped_mask

Z = 20
X = 164
Y = 353
vis_coords = [9, 14]

isolated_cell_no_repair, cell_coords_no_repair = isolate_cell(mask_no_repair, mask_no_repair[Z, Y, X])
isolated_cell_mask_XY, cell_coords_XY = isolate_cell(mask_2D_XY_stack, mask_2D_XY_stack[Z, Y, X])
isolated_cell_mask_XZ, cell_coords_XZ = isolate_cell(mask_2D_XZ_stack, mask_2D_XZ_stack[Z, Y, X])
isolated_cell_mask_YZ, cell_coords_YZ = isolate_cell(mask_2D_YZ_stack, mask_2D_YZ_stack[Z, Y, X])
isolated_cell_repair, cell_coords_repair = isolate_cell(mask_repair, mask_repair[Z, Y, X])

Z_min = min(cell_coords_XZ[0])
Z_max = max(cell_coords_XZ[0]) + 1
Y_min = min(cell_coords_XY[1])
Y_max = max(cell_coords_XZ[1]) + 1
X_min = min(cell_coords_XZ[2])
X_max = max(cell_coords_XZ[2]) + 1

bounding_box = (Z_min, Z_max, Y_min, Y_max, X_min, X_max)
cropped_cell_mask_XY = crop_mask(mask_2D_XY_stack, bounding_box)
cropped_cell_mask_XZ = crop_mask(mask_2D_XZ_stack, bounding_box)
cropped_cell_mask_YZ = crop_mask(mask_2D_YZ_stack, bounding_box)
cropped_cell_mask_no_repair = crop_mask(mask_no_repair, bounding_box)
cropped_cell_mask_repair = crop_mask(mask_repair, bounding_box)

cropped_cell_mask_XY[np.where(cropped_cell_mask_XY != mask_2D_XY_stack[Z, Y, X])] = 0
cropped_cell_mask_XZ[np.where(cropped_cell_mask_XZ != mask_2D_XZ_stack[Z, Y, X])] = 0
cropped_cell_mask_YZ[np.where(cropped_cell_mask_YZ != mask_2D_YZ_stack[Z, Y, X])] = 0
cropped_cell_mask_no_repair[np.where(cropped_cell_mask_no_repair != mask_no_repair[Z, Y, X])] = 0
cropped_cell_mask_repair[np.where(cropped_cell_mask_repair != mask_repair[Z, Y, X])] = 0

dual_color = True
if dual_color:
    color_idx = 2
    mask_no_repair_coords = np.where(cropped_cell_mask_no_repair != 0)
    cropped_cell_mask_XY[mask_no_repair_coords] = color_idx
    cropped_cell_mask_XZ[mask_no_repair_coords] = color_idx
    cropped_cell_mask_YZ[mask_no_repair_coords] = color_idx
    cropped_cell_mask_repair[mask_no_repair_coords] = color_idx

repaired_coords = np.where(cropped_cell_mask_repair != 0)
create_slice_fig_new(cropped_cell_mask_no_repair, 'fragment', vis_coords, repaired_coords, mask_repair[Z, Y, X])
create_slice_fig_new(cropped_cell_mask_repair, 'repaired', vis_coords)
create_slice_fig_new(cropped_cell_mask_XY, 'Z_stack', vis_coords)
create_slice_fig_new(cropped_cell_mask_XZ, 'Y_stack', vis_coords)
create_slice_fig_new(cropped_cell_mask_YZ, 'X_stack', vis_coords)
