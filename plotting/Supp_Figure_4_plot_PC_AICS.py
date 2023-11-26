import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from os.path import join
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

if __name__ == '__main__':
	channel_list = ['actin','lysosome','golgi','tublin','mito']
	channel_list_vis = ['Actin filaments','Lysosome','Golgi apparatus','Microtubule','Mitochondria']
	pca_dir = '/home/hrchen/Documents/Research/hubmap/script/2D-3D/PCA_model'
	cmap = ['deepskyblue', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan', 'darkgrey']
	marker = ["o", "s", "^", "P", "D", "*", "X", "h", "v", "d", "p"]
	methods = ['deepcell_membrane-0.12.6', 'cellpose-2.2.2', 'aics_ml']
	methods_vis = ['3DCellComposer w/ DeepCell', '3DCellComposer w/ Cellpose', 'ACSS']
	noise_list = ['original', 'random_gaussian_1', 'random_gaussian_2', 'random_gaussian_3']
	# noise_list = ['original']
	ss, pca = pickle.load(open(join(pca_dir, 'pca_AICS.pkl'), 'rb'))
	pc1_explained = "{:.0f}%".format(pca.explained_variance_ratio_[0] * 100)
	pc2_explained = "{:.0f}%".format(pca.explained_variance_ratio_[1] * 100)
	
	for channel in channel_list:
		data_dir = f'/data/3D/AICS/AICS_{channel}'

		weighted_score_list = []
		for method in methods:
			avg_method_metrics_pca_pieces = list()
			avg_method_metrics_pieces = list()
			for noise in noise_list:
				if method != 'aics_ml':
					metrics_dir_list = list()
					img_dir_list = sorted(glob.glob(f'{data_dir}/**/original', recursive=True))
					for img_dir in img_dir_list:
						optimal_JI = np.load(f'{img_dir}/metrics/optimal_JI_{method}.npy')
						print(optimal_JI)
						metrics_dir_list.append(
							f'{os.path.dirname(img_dir)}/{noise}/metrics/metrics_{method}_{optimal_JI}.npy')
				else:
					metrics_dir_list = sorted(
						glob.glob(f'{data_dir}/**/{noise}/**/metrics_{method}_0.0.npy', recursive=True))
				
				current_metrics_pieces = list()
				for metrics_dir in metrics_dir_list:
					current_metrics_pieces.append(np.load(metrics_dir))
				current_metrics = np.vstack(current_metrics_pieces)
				current_metrics = np.delete(current_metrics, 5, axis=1)

				# current_metrics = np.delete(current_metrics, -1, axis=1)
				# current_metrics = np.delete(current_metrics, -3, axis=1)
				# current_metrics = np.nan_to_num(current_metrics, nan=1)
				
				current_metrics_z = ss.transform(current_metrics)
				avg_method_metrics_pieces.append(np.average(current_metrics_z))
				current_metrics_pca = pca.transform(current_metrics_z)
				print(method)
				print(noise)
				print(current_metrics)

				avg_current_metrics_pca = np.average(current_metrics_pca, axis=0)
				print(avg_current_metrics_pca)
	
				avg_method_metrics_pca_pieces.append(avg_current_metrics_pca)
				
				if noise == 'original':
					weighted_score = sum(avg_current_metrics_pca[i] * pca.explained_variance_ratio_[i] for i in range(2))
					weighted_score_list.append(weighted_score)
	
			avg_method_metrics_pca = np.vstack(avg_method_metrics_pca_pieces)
			method_idx = methods.index(method)
			channel_idx = channel_list.index(channel)
			print(avg_method_metrics_pca[:, 0])
			plt.plot(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], color=cmap[method_idx])
			plt.scatter(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], edgecolors=cmap[method_idx], s=20, facecolors='none', marker = marker[channel_idx])
			plt.scatter(avg_method_metrics_pca[0, 0], avg_method_metrics_pca[0, 1], edgecolors='black', s=50, facecolors=cmap[method_idx], marker = marker[channel_idx])
			# print(avg_method_metrics_pieces)
	

	plt.xlabel(f'PC1({pc1_explained})')
	plt.ylabel(f'PC2({pc2_explained})')
	
	color_handles = [patches.Patch(color=color, label=methods_vis[i]) for i, color in
	                 enumerate(cmap[:len(methods_vis)])]
	
	marker_handles = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='', label=channel_list_vis[i]) for i, marker
	                  in enumerate(marker[:len(channel_list_vis)])]
	
	# Add the legend to the plot
	plt.legend(handles=color_handles + marker_handles, loc='upper left')
	plt.tight_layout()
	plt.savefig(f'{os.path.dirname(pca_dir)}/fig/AICS_all_PCA.png', dpi=500)
	plt.clf()
	
	# sorted_data = sorted(zip(weighted_score_list, methods_vis), reverse=False)
	# weighted_score_list_sorted = [item[0] for item in sorted_data]
	# methods_vis_sorted = [item[1] for item in sorted_data]

	# plt.barh(methods_vis_sorted, weighted_score_list_sorted)
	#
	# plt.xlabel('Quality Score')
	# plt.ylabel('Methods')
	#
	# plt.tight_layout()  # Adjust layout for better display
	# plt.savefig(f'{os.path.dirname(pca_dir)}/fig/AICS_test_{channel}_score.png', dpi=500)
	# plt.clf()