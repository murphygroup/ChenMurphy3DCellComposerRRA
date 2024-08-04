"""
This is a script to create cell number files for plotting Supplementary Figures 3 and 6.
It is not part of the main evaluation but is used for posterior analysis.
"""


import glob
import bz2
import pickle
import numpy as np

print('creating cell number files...')

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

def get_cell_num(data_type):
    JI_list = get_JI_list(data_type)
    method = 'deepcell_membrane-0.12.6'
    mask_dir_list = sorted(glob.glob(f'../data/masks/{data_type}/**/original', recursive=True))
    for mask_dir in mask_dir_list:
        print(mask_dir)
        cell_num_list = []
        for JI in JI_list:
            mask_path = f'{mask_dir}/mask_{method}_matched_3D_final_{JI}.pkl'
            with bz2.BZ2File(mask_path, 'r') as file:
                mask = pickle.load(file)
                cell_num_list.append(len(np.unique(mask)) - 1)
        np.save(f'{mask_dir}/cell_num_JI.npy', cell_num_list)

if __name__ == '__main__':
    get_cell_num('IMC_3D')
    get_cell_num('AICS')

print('completed!')
