import matplotlib.pyplot as plt
import numpy as np
import glob
import random
random.seed(3)

print('plotting Supp Fig 3...')

data_type = 'IMC_3D'
data_dir = f'../data/metrics/{data_type}'
IMC_structure_list = ['d3130f4a89946cc6b300b115a3120b7a','cd880c54e0095bad5200397588eccf81','a296c763352828159f3adfa495becf3e']
IMC_structure_list_vis = ['SPLEEN', 'Thymus','Lymph Node']
figure_dir = '../fig'

# Create a figure and a set of subplots - 2 rows, 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

for i, IMC_structure in enumerate(IMC_structure_list):
    # Find the subplot location
    ax1 = axs[i]

    mask_dir_list = sorted(glob.glob(f'{data_dir}/**/{IMC_structure}/**/original', recursive=True))
    mask_dir = random.choice(mask_dir_list)
    method = 'deepcell_membrane-0.12.6'

    for axis in ['XY', 'XZ','YZ']:
        JI_list_image = np.load(f'{mask_dir}/best_JI_list_{axis}.npy').tolist()
        JI_list_list = JI_list_list + JI_list_image

    cell_num_JI_image = np.load(f'{mask_dir}/cell_num_JI.npy')
    quality_score = np.load(f'{mask_dir}/metrics/quality_scores_JI_{method}.npy')

    labels = [round(i * 0.1, 1) for i in range(8)]
    bins = np.arange(0, 0.9, 0.1)
    counts, _ = np.histogram(JI_list_list, bins)
    cumulative_counts = np.cumsum(counts[::-1])[::-1]

    # Plotting for this structure
    ax1.plot(np.arange(0, 0.8, 0.1), cumulative_counts, marker='^', color='blue')
    ax1.set_xlabel('Jaccard Index Threshold')
    ax1.set_ylabel('Number of Pairs of Matched 2D Cells', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(np.arange(0, 0.8, 0.1))
    ax1.invert_xaxis()

    # Secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(labels, cell_num_JI_image, marker='s', color='green')
    ax2.set_ylabel('Number of Final 3D Cells', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Third y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(labels, quality_score, marker='o', linestyle='-', color='red')
    ax3.set_ylabel('Quality Score', color='r')
    ax3.tick_params('y', colors='r')

    # Set the title for each subplot
    ax1.set_title(IMC_structure_list_vis[i])


plt.tight_layout()

# Save the figure
plt.savefig(f'{figure_dir}/Supp_Fig_3_{data_type}_JI_vs_cell_num.png', dpi=500)
plt.close(fig)
