import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from os.path import join
import os
import matplotlib.pyplot as plt
import sys
import umap
import pandas as pd
if __name__ == '__main__':
	
	print('plotting Fig 3...')

	data_dir = f'../data/masks/IMC_3D/florida-3d-imc'
	pca_dir = '../data/PCA_model'
	cmap = ['deepskyblue', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan', 'darkgrey']
	marker = ["o", "o", "s", "^", "P", "D", "*", "X", "h", "v", "d", "p"]


	methods = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm', 'cellpose-2.2.2_2D-3D', 'aics_ml', '3DCellSeg']
	methods_vis = ['DeepCell_mem', 'DeepCell_cyto', 'Cellpose', 'ACSS(classic)', 'CellProfiler', 'CellX',
	           'CellSegm', 'Cellpose2Dto3D', 'ACSS', '3DCellSeg']
	methods_2D = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm']
	noise_list = ['original', 'random_gaussian_1', 'random_gaussian_2', 'random_gaussian_3']
	
	
	ss, pca_model = pickle.load(open(join(pca_dir, 'pca_IMC_3D.pkl'), 'rb'))
	pc1_explained = "{:.0f}%".format(pca_model.explained_variance_ratio_[0] * 100)
	pc2_explained = "{:.0f}%".format(pca_model.explained_variance_ratio_[1] * 100)
	weighted_score_list = []
	for method in methods:
		avg_method_metrics_pca_pieces = list()
		for noise in noise_list:
			# print(method)
			# print(noise)
			if method in methods_2D:
				metrics_dir_list = list()
				img_dir_list = sorted(glob.glob(f'{data_dir}/**/original', recursive=True))
				for img_dir in img_dir_list:
					optimal_JI = np.load(f'{img_dir}/metrics/optimal_JI_{method}.npy')
					# print(optimal_JI)
					metrics_dir_list.append(f'{os.path.dirname(img_dir)}/{noise}/metrics/metrics_{method}_{optimal_JI}.npy')
			else:
				metrics_dir_list = sorted(glob.glob(f'{data_dir}/**/{noise}/**/metrics_{method}_0.0.npy', recursive=True))

			current_metrics_pieces = list()
			for metrics_dir in metrics_dir_list:
				current_current_metrics = np.load(metrics_dir)

				current_metrics_pieces.append(current_current_metrics)
			current_metrics = np.vstack(current_metrics_pieces)
			current_metrics_z = ss.transform(current_metrics)
			current_metrics_pca = pca_model.transform(current_metrics_z)
			# print(current_metrics)
			

			avg_current_metrics_pca = np.average(current_metrics_pca, axis=0)


			avg_method_metrics_pca_pieces.append(avg_current_metrics_pca)

			if noise == 'original':
				weighted_score = np.exp(sum(avg_current_metrics_pca[i] * pca_model.explained_variance_ratio_[i] for i in range(2)))
				weighted_score_list.append(weighted_score)

		avg_method_metrics_pca = np.vstack(avg_method_metrics_pca_pieces)
		method_idx = methods.index(method)
		plt.plot(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], color=cmap[method_idx])
		plt.scatter(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], edgecolors=cmap[method_idx], s=50, facecolors='none', marker = marker[method_idx])
		plt.scatter(avg_method_metrics_pca[0, 0], avg_method_metrics_pca[0, 1], edgecolors=cmap[method_idx], s=50, facecolors=cmap[method_idx], marker = marker[method_idx], label=methods_vis[method_idx])

	

	plt.xlabel(f'PC1({pc1_explained})')
	plt.ylabel(f'PC2({pc2_explained})')

	plt.savefig(f'../fig/Fig_3_IMC_PCA.png', dpi=500, bbox_inches='tight')
	plt.clf()
	
	# Sort the data
	colors = ['green'] * 7 + ['blue'] * 3
	sorted_data = sorted(zip(weighted_score_list, methods_vis, colors), reverse=False)
	weighted_score_list_sorted = [item[0] for item in sorted_data]
	methods_vis_sorted = [item[1] for item in sorted_data]
	colors_sorted = [item[2] for item in sorted_data]
	
	# Create the horizontal bar plot with color coding
	bars = plt.barh(methods_vis_sorted, weighted_score_list_sorted, color=colors_sorted)
	
	# Create dummy bars for legend
	bar1 = plt.barh([0], [0], color='green', label='3DCellComposer')
	bar2 = plt.barh([0], [0], color='blue', label='Other Methods')
	
	# Remove the dummy bars
	plt.gca().cla()
	
	# Add the real bars again
	plt.barh(methods_vis_sorted, weighted_score_list_sorted, color=colors_sorted)
	# print(weighted_score_list)
	# Label the axes
	plt.xlabel('Quality Score')
	plt.ylabel('Methods')

	plt.legend(handles=[bar1, bar2], title=None, loc=4)
	
	# Save the plot
	plt.savefig(f'../fig/Fig_3_IMC_score.png', dpi=500, bbox_inches='tight')
	
	# Clear the plot for future use
	plt.clf()

	print('completed!')
	
