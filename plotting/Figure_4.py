import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from os.path import join
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.legend as mlegend
import pandas as pd

if __name__ == '__main__':
	channel_list = ['actin', 'lysosome', 'golgi', 'tublin', 'mito']
	channel_list_vis = ['Actin filaments', 'Lysosome', 'Golgi apparatus', 'Microtubule', 'Mitochondria']
	pca_dir = '../data/PCA_model'
	cmap = ['deepskyblue', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate',
	        'darkcyan', 'darkgrey']
	marker = ["o", "s", "^", "P", "D", "*", "X", "h", "v", "d", "p"]
	methods = ['deepcell_membrane-0.12.6', 'cellpose-2.2.2', 'aics_ml']
	methods_vis = ['3DCellComposer w/ DeepCell', '3DCellComposer w/ Cellpose', 'ACSS']
	noise_list = ['original']
	
	ss, pca = pickle.load(open(join(pca_dir, 'pca_ACSS.pkl'), 'rb'))
	pc1_explained = "{:.0f}%".format(pca.explained_variance_ratio_[0] * 100)
	pc2_explained = "{:.0f}%".format(pca.explained_variance_ratio_[1] * 100)
	fig, ax = plt.subplots()
	
	weighted_score_list = []
	for channel in channel_list:
		data_dir = f'../data/masks/AICS/AICS_{channel}'
		weighted_score_channel_list = []
		
		for method in methods:
			avg_method_metrics_pca_pieces = []
			for noise in noise_list:
				if method != 'aics_ml':
					metrics_dir_list = []
					img_dir_list = sorted(glob.glob(f'{data_dir}/**/original', recursive=True))
					for img_dir in img_dir_list:
						optimal_JI = np.load(f'{img_dir}/metrics/optimal_JI_{method}.npy')
						metrics_dir_list.append(
							f'{os.path.dirname(img_dir)}/{noise}/metrics/metrics_{method}_{optimal_JI}.npy')
				else:
					metrics_dir_list = sorted(
						glob.glob(f'{data_dir}/**/{noise}/**/metrics_{method}_0.0.npy', recursive=True))
				
				current_metrics_pieces = [np.load(metrics_dir) for metrics_dir in metrics_dir_list]
				current_metrics = np.vstack(current_metrics_pieces)
				
				current_metrics = np.delete(current_metrics, 5, axis=1)
				
				current_metrics_z = ss.transform(current_metrics)
				current_metrics_pca = pca.transform(current_metrics_z)
				
				avg_current_metrics_pca = np.average(current_metrics_pca, axis=0)
				avg_method_metrics_pca_pieces.append(avg_current_metrics_pca)
				
				if noise == 'original':
					weighted_score = np.exp(
						sum(avg_current_metrics_pca[i] * pca.explained_variance_ratio_[i] for i in range(2)))
					weighted_score_channel_list.append(weighted_score)
			
			avg_method_metrics_pca = np.vstack(avg_method_metrics_pca_pieces)
			method_idx = methods.index(method)
			channel_idx = channel_list.index(channel)
			
			ax.plot(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], color=cmap[method_idx])
			ax.scatter(avg_method_metrics_pca[:, 0], avg_method_metrics_pca[:, 1], edgecolors=cmap[method_idx], s=20,
			           facecolors='none', marker=marker[channel_idx])
			ax.scatter(avg_method_metrics_pca[0, 0], avg_method_metrics_pca[0, 1], edgecolors='black', s=50,
			           facecolors=cmap[method_idx], marker=marker[channel_idx])
		
		weighted_score_list.append(weighted_score_channel_list)
	
	plt.xlabel(f'PC1({pc1_explained})')
	plt.ylabel(f'PC2({pc2_explained})')
	
	color_handles = [patches.Patch(color=color, label=methods_vis[i]) for i, color in
	                 enumerate(cmap[:len(methods_vis)])]
	marker_handles = [plt.Line2D([0], [0], marker=marker, color='k', linestyle='', label=channel_list_vis[i]) for
	                  i, marker in enumerate(marker[:len(channel_list_vis)])]
	
	plt.xlim([3.58, 4.25])
	upper_left_legend = mlegend.Legend(ax, handles=color_handles, labels=methods_vis, loc='upper left')
	ax.add_artist(upper_left_legend)
	lower_right_legend = mlegend.Legend(ax, handles=marker_handles, labels=channel_list_vis, loc='lower right')
	ax.add_artist(lower_right_legend)
	plt.tight_layout()
	plt.savefig(f'../fig/Fig_4_ACSS_original_PCA.png', dpi=500)
	plt.clf()
	
	weighted_score_list = np.vstack(weighted_score_list)
	weighted_score_pd = pd.DataFrame(data=weighted_score_list, index=channel_list_vis, columns=methods_vis)
	colors = ['deepskyblue', 'darkred', 'darkgoldenrod']
	
	weighted_score_pd.plot(kind='barh', color=colors)
	plt.xlabel('Quality Score')
	plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.1), ncol=3)
	plt.tight_layout()
	plt.savefig(f'../fig/Fig_4_ACSS_original_scores.png', dpi=500)
