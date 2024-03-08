import glob
import numpy as np
import pickle
import bz2
import pandas as pd
from os.path import join
import os

if __name__ == '__main__':
	
	data_dir = './data/IMC_3D/florida-3d-imc'
	methods_2D = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic',
	              'CellProfiler', 'CellX', 'cellsegm']
	methods_vis = ['DeepCell_mem', 'DeepCell_cyto', 'Cellpose', 'ACSS(classic)', 'CellProfiler', 'CellX',
	           'CellSegm']
	
	def load_pkl(file_path):
		return pickle.load(bz2.BZ2File(file_path, 'rb'))
	
	
	img_dir_list = sorted(glob.glob(f'{data_dir}/**/original', recursive=True))
	
	# Initialize an empty dictionary to hold cell count data
	
	for img_dir in img_dir_list[:1]:
		print(img_dir)
		cell_counts_data = {method: [] for method in methods_2D}
		
		img_name = img_dir.split('/')[-2]
		for method in methods_2D:
			optimal_JI = np.load(f'{img_dir}/metrics/optimal_JI_{method}.npy')
			img_dir_store = img_dir.replace("IMC_3D", "IMC_3D_store")
			
			cell_counts_img_2D = []
			cell_counts_img_prelim_3D = []
			for axis in ['XY', 'XZ', 'YZ']:
				img_2D_path = f'{img_dir_store}/mask_{method}_{axis}.pkl'
				img_2D = load_pkl(img_2D_path)
				cell_counts_img_2D.append(sum(len(np.unique(x)) - 1 for x in img_2D))
				
				img_prelim_3D_path = f'{img_dir_store}/mask_{method}_matched_stack_{axis}_{optimal_JI}.pkl'
				img_prelim_3D = load_pkl(img_prelim_3D_path)
				cell_counts_img_prelim_3D.append(len(np.unique(img_prelim_3D)) - 1)
			
			img_before_nuclear_matching_path = f'{img_dir_store}/mask_{method}_matched_3D_{optimal_JI}.pkl'
			img_before_nuclear_matching = load_pkl(img_before_nuclear_matching_path)
			cell_counts_img_before_nuclear_matching = len(np.unique(img_before_nuclear_matching)) - 1
			
			img_after_nuclear_matching_path = f'{img_dir_store}/mask_{method}_matched_3D_final_{optimal_JI}.pkl'
			img_after_nuclear_matching = load_pkl(img_after_nuclear_matching_path)
			cell_counts_img_after_nuclear_matching = len(np.unique(img_after_nuclear_matching)) - 1
			
			# Store all counts for the current method
			cell_counts_data[method].extend(cell_counts_img_2D)
			cell_counts_data[method].append(cell_counts_img_before_nuclear_matching)
			cell_counts_data[method].append(cell_counts_img_before_nuclear_matching-cell_counts_img_after_nuclear_matching)
			cell_counts_data[method].append(cell_counts_img_after_nuclear_matching)
	
		# Convert the dictionary to a pandas DataFrame
		cell_counts_df = pd.DataFrame(cell_counts_data)
		cell_counts_df.index = ['2D cells along all slices of Z axis',
		                        '2D cells along all slices of Y axis',
		                        '2D cells along all slices of X axis',
		                        'preliminary 3D cells before 3D nuclear matching',
		                        'preliminary 3D cells removed due to no corresponding 3D nuclei',
		                        'matched 3D cells after 3D nuclear matching (final 3D cells)']
		cell_counts_df.columns = methods_vis
		cell_counts_df.index.name = 'Number of cells / Method'
		

		cell_counts_df.to_csv(f'./table/cell_counts_{img_name}.csv')
