import glob
import numpy as np
import pickle
from os.path import join
import os
import shutil
import time

print('check necessary files for plotting figures...')
time.sleep(1)

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



def move_metrics():
	# Define the base paths
	metrics_base_path = "../data/metrics"
	masks_base_path = "../data/masks"
	
	# Walk through the metrics base path
	for root, dirs, files in os.walk(metrics_base_path):
		if 'metrics' in dirs:
			# Construct the full path of the metrics directory
			metrics_dir = os.path.join(root, 'metrics')
			
			# Extract the relative path from the metrics base path
			relative_path = os.path.relpath(metrics_dir, metrics_base_path)
			
			# Construct the corresponding path in the masks base path
			destination_dir = os.path.join(masks_base_path, relative_path)
			
			# Ensure the destination parent directory exists
			os.makedirs(os.path.dirname(destination_dir), exist_ok=True)
			
			# Move the metrics directory to the corresponding masks directory
			shutil.move(metrics_dir, os.path.dirname(destination_dir))
	

def move_supp_data():
	# Define the base paths
	
	# # Define the path to the zip file and the extraction directory
	# zip_file_path = "../data/supp_data.zip"
	# extraction_directory = "../data/"
	#
	# # Run the bash command to unzip the file
	# subprocess.run(["unzip", zip_file_path, "-d", extraction_directory])
	#
	# print("Unzipping complete!")
	
	supp_data_base_path = "../data/supp_data"
	masks_base_path = "../data/masks"
	
	# Walk through the metrics base path
	for root, dirs, files in os.walk(supp_data_base_path):
		for file in files:
			# Construct the full path of the source file
			source_file = os.path.join(root, file)
			
			# Extract the relative path from the metrics base path
			relative_path = os.path.relpath(source_file, supp_data_base_path)
			
			# Construct the corresponding path in the masks base path
			destination_file = os.path.join(masks_base_path, relative_path)
			
			# Ensure the destination parent directory exists
			os.makedirs(os.path.dirname(destination_file), exist_ok=True)
			
			# Copy the file to the corresponding masks directory
			shutil.copy2(source_file, destination_file)



def create_optimal_JI_files(data_type):
	data_dir = f'../data/masks/{data_type}'
	pca_dir = '../data/PCA_model'

	methods_IMC = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm']
	methods_AICS = ['deepcell_membrane-0.12.6', 'cellpose-2.2.2']
	JI_list = get_JI_list(data_type)
	if data_type == 'IMC_3D':
		methods = methods_IMC
	elif data_type == 'AICS':
		methods = methods_AICS
		
	noise_list = ['original', 'random_gaussian_1', 'random_gaussian_2', 'random_gaussian_3']
	ss, pca = pickle.load(open(join(pca_dir, f'pca_{data_type}.pkl'), 'rb'))

	
	for noise in noise_list:
		img_dir_list = sorted(glob.glob(f'{data_dir}/**/{noise}', recursive=True))
		for img_dir in img_dir_list:
			for method in methods:
				metrics_dir_list = sorted(glob.glob(f'{img_dir}/metrics/metrics_{method}_0.*.npy', recursive=True))
				current_metrics_pieces = list()
				for metrics_dir in metrics_dir_list:
					current_metrics = np.load(metrics_dir)
					if data_type == 'AICS':
						current_metrics = np.delete(current_metrics, 5, axis=1)
					current_metrics_pieces.append(current_metrics)
					
				current_metrics = np.vstack(current_metrics_pieces)
				current_metrics_z = ss.transform(current_metrics)
				current_metrics_z_pca = pca.transform(current_metrics_z)
				weighted_score = np.exp(sum(current_metrics_z_pca[:,i] * pca.explained_variance_ratio_[i] for i in range(2)))
				optimal_JI_method_image = JI_list[np.argmax(weighted_score)]
				np.save(f'{img_dir}/metrics/optimal_JI_{method}.npy', optimal_JI_method_image)
				np.save(f'{img_dir}/metrics/quality_scores_JI_{method}.npy', weighted_score)
				
move_metrics()
move_supp_data()
create_optimal_JI_files('IMC_3D')
create_optimal_JI_files('AICS')
print('completed!')
