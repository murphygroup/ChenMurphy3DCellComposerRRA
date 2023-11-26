import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from os.path import join
import os
import matplotlib.pyplot as plt
import bz2

if __name__ == '__main__':
	data_type = 'IMC_3D'
	data_dir = f'/data/3D/{data_type}'
	pca_dir = '/home/hrchen/Documents/Research/hubmap/script/2D-3D/PCA_model'
	cmap = ['deepskyblue', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan', 'darkgrey']
	marker = ["o", "s", "^", "P", "D", "*", "X", "h", "v", "d", "p"]
	methods = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm']
	# methods_vis = ['DeepCell+our method JI=0.1', 'DeepCell+our method JI=0.3', 'DeepCell+our method JI=0.5', 'DeepCell+cellstitch', 'AICS(ML)']
	# method = 'deepcell_membrane-0.12.6'
	# method = 'CellProfiler'
	# base_method = "_"
	# base_method_vis = ""
	#
	# methods = ['deepcell_membrane-0.12.6_0.0']
	# methods_vis = ['0.0']
	JI_list = []
	
	for i in range(0, 8, 1):
		value = round(i * 0.1, 1)
		JI_list.append(str(value))
		# methods.append(base_method + str(value))
		# methods_vis.append(base_method_vis + str(value))
		
	# noise_list = ['original', 'random_gaussian_1', 'random_gaussian_2', 'random_gaussian_3']
	noise_list = ['original']

	
	ss, pca = pickle.load(open(join(pca_dir, f'pca_{data_type}.pkl'), 'rb'))
	pc1_explained = "{:.0f}%".format(pca.explained_variance_ratio_[0] * 100)
	pc2_explained = "{:.0f}%".format(pca.explained_variance_ratio_[1] * 100)
	
	for method in methods:
		weighted_score_list = []
		for JI in JI_list:
			# print(JI)
			avg_method_metrics_pca_pieces = list()
			avg_method_metrics_pieces = list()
			for noise in noise_list:
				metrics_dir_list = sorted(glob.glob(f'{data_dir}/**/{noise}/**/metrics_{method}_{JI}.npy', recursive=True))
				current_metrics_pieces = list()
				for metrics_dir in metrics_dir_list:
					current_metrics_pieces.append(np.load(metrics_dir))
					# print(metrics_dir)
					# print(np.load(metrics_dir).shape)
				current_metrics = np.vstack(current_metrics_pieces)
				# current_metrics = np.delete(current_metrics, 0, axis=1)
				# current_metrics = np.delete(current_metrics, -3, axis=1)
	
				current_metrics_z = ss.transform(current_metrics)
				avg_method_metrics_pieces.append(np.average(current_metrics_z))
				current_metrics_pca = pca.transform(current_metrics_z)
				avg_current_metrics_pca = np.average(current_metrics_pca, axis=0)
				avg_method_metrics_pca_pieces.append(avg_current_metrics_pca)
				if noise == 'original':
					weighted_score = np.exp(sum(avg_current_metrics_pca[i] * pca.explained_variance_ratio_[i] for i in range(2)))
					weighted_score_list.append(weighted_score)
					print(method)
					# print(noise)
					# print(np.average(current_metrics_z, axis=0))
					
			avg_method_metrics_pca = np.vstack(avg_method_metrics_pca_pieces)
			method_idx = JI_list.index(JI)
			plt.plot(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], color=cmap[method_idx])
			plt.scatter(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], edgecolors=cmap[method_idx], s=50, facecolors='none', marker = marker[method_idx])
			plt.scatter(avg_method_metrics_pca[0, 0], avg_method_metrics_pca[0, 1], edgecolors=cmap[method_idx], s=50, facecolors=cmap[method_idx], marker = marker[method_idx], label=JI_list[method_idx])
			# print(avg_method_metrics_pieces)
		
		plt.xlabel(f'PC1({pc1_explained})')
		plt.ylabel(f'PC2({pc2_explained})')
		plt.legend(title='JI Threshold')
		plt.tight_layout()  # Adjust layout for better display
	
		plt.savefig(f'{os.path.dirname(pca_dir)}/fig/{data_type}_JI_PCA.png', dpi=500)
		plt.clf()
		
		# sorted_data = sorted(zip(weighted_score_list, JI_list), reverse=False)
		# weighted_score_list_sorted = [item[0] for item in sorted_data]
		# methods_vis_sorted = [item[1] for item in sorted_data]
		np.save(f'{os.path.dirname(pca_dir)}/fig/{data_type}_{method}_JI_score.npy', weighted_score_list)
		optimal_JI = JI_list[np.argmax(weighted_score_list)]
		np.save(f'{os.path.dirname(pca_dir)}/fig/{data_type}_{method}_optimal_JI.npy', optimal_JI)
		print(optimal_JI)
		plt.plot(JI_list, weighted_score_list)
		
		plt.xlabel('Jaccard Index Threshold')
		plt.ylabel('Quality Score')
		plt.tight_layout()  # Adjust layout for better display
		plt.savefig(f'{os.path.dirname(pca_dir)}/fig/{data_type}_{method}_JI_score.png', dpi=500)
		plt.clf()
	
	# mask_dir_list = sorted(glob.glob(f'{data_dir}/**/original', recursive=True))
	# for mask_dir in mask_dir_list:
	# 	cell_num_list = []
	# 	for JI in JI_list:
	# 		mask = pickle.load(bz2.BZ2File(f'{mask_dir}/mask_{method}_matched_3D_final_{JI}.pkl', 'r'))
	# 		cell_num_list.append(len(np.unique(mask)-1))
	# 	print(cell_num_list)
	# 	np.save(f'{mask_dir}/cell_num_JI.npy', cell_num_list)
	