import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from os.path import join
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data_dir = './data'
	datasets = ['IMC_3D', 'AICS']
	pca_dir = './PCA_model'
	methods_IMC = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm', 'cellpose-2.2.2_2D-3D', 'aics_ml', '3DCellSeg']
	methods_2D = ['deepcell_membrane-0.12.6', 'deepcell_cytoplasm-0.12.6', 'cellpose-2.2.2', 'aics_classic', 'CellProfiler', 'CellX',
	           'cellsegm']
	methods_AICS = ['deepcell_membrane-0.12.6', 'cellpose-2.2.2', 'aics_ml']

	for dataset in datasets:
		metrics_dir_list = []
		if dataset == 'IMC_3D':
			methods = methods_IMC
		elif dataset == 'AICS':
			methods = methods_AICS
		for method in methods:
			if method in methods_2D:
				metrics_dir_list = metrics_dir_list + sorted(glob.glob(f'{data_dir}/{dataset}/**/metrics/metrics_{method}_0.*.npy', recursive=True))
			else:
				metrics_dir_list = metrics_dir_list + sorted(glob.glob(f'{data_dir}/{dataset}/**/metrics/metrics_{method}_0.0.npy', recursive=True))
		metrics_pieces = list()
		for metrics_dir in metrics_dir_list:
			current_metrics = np.load(metrics_dir)
			if current_metrics.shape[1] == 15:
				current_metrics = np.delete(current_metrics, 5, axis=1)
			metrics_pieces.append(current_metrics)
		metrics = np.vstack(metrics_pieces)

		ss = StandardScaler().fit(metrics)
		metrics_z = ss.transform(metrics)
		pca_model = PCA(n_components=2).fit(metrics_z)
		if pca_model.components_[1][0] < 0:
			pca_model.components_[1] = -pca_model.components_[1]
		print(pca_model.components_)
		print(pca_model.explained_variance_ratio_)

		pickle.dump([ss, pca_model], open(join(pca_dir, f'pca_{dataset}.pkl'), "wb"))
	
		loadings = pca_model.components_
		metric_names = ['NC','FFC','1-FBC','FCF','1/(CSCV+1)', 'WACS','1/(ACVF+1)','FPCF',
		           '1/(ACVC_NUC+1)','FPCC_NUC','AS_NUC','1/(ACVC_CEN+1)','FPCC_CEN','AS_CEN']


		fig, ax = plt.subplots()
		
		width = 0.4
		
		ind = np.arange(loadings.shape[1])
		
		p1 = ax.bar(ind, loadings[0, :], width, color='blue', label='PC1')
		p2 = ax.bar(ind + width, loadings[1, :], width, color='green', label='PC2')
		
		ax.set_xlabel('Metric', fontsize=12)
		ax.set_ylabel('Loading', fontsize=12)
		ax.set_xticks(ind + width / 2)
		ax.set_xticklabels(metric_names, rotation=45, fontsize=12, ha='right',  rotation_mode="anchor")
		ax.legend(fontsize=12)
		ax.tick_params(axis='y', labelsize=13)
		plt.tight_layout()
		plt.savefig(f'./fig/{dataset}_loadings.png', dpi=500)

