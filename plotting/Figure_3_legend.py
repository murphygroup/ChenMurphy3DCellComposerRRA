import matplotlib.pyplot as plt

save_dir = '../fig'
print("plotting Fig 3's legend...")

# Your color, marker and method information
cmap = ['deepskyblue', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan', 'darkgrey']
marker = ["o", "o", "s", "^", "P", "D", "*", "X", "h", "v"]
methods_vis = ['DeepCell_mem', 'DeepCell_cyto', 'Cellpose', 'ACSS(classic)', 'CellProfiler', 'CellX', 'CellSegm', 'Cellpose2Dto3D', 'ACSS', '3DCellSeg']

legend_elements = []

# Create scatter plots off the visible axis
for i, method in enumerate(methods_vis):
    legend_elements.append(plt.scatter([], [], c=cmap[i], marker=marker[i], label=methods_vis[i]))

# Create main plot title and labels
plt.title('Comparison of Various Methods')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Create legends
leg1 = plt.legend(handles=legend_elements[:7], title='3DCellComposer', loc='lower left', bbox_to_anchor=(0, -0.5), ncol=2)
leg2 = plt.legend(handles=[legend_elements[7]], title='Other 2D to 3D Method', loc='lower center', bbox_to_anchor=(0.66, 0.08))
leg3 = plt.legend(handles=legend_elements[8:], title='3D Methods', loc='lower right', bbox_to_anchor=(0.89, -0.5), ncol=2)
# leg1 = plt.legend(handles=legend_elements[:7], title='3DCellComposer', loc='lower left', bbox_to_anchor=(0, 0), ncol=1)
# leg2 = plt.legend(handles=[legend_elements[7]], title='Other 2D to 3D Method', loc='lower center', bbox_to_anchor=(0.124, -0.26))
# leg3 = plt.legend(handles=legend_elements[8:], title='3D Methods', loc='lower right', bbox_to_anchor=(0.277, -0.63), ncol=1)

# Add the legends to the current Axes
ax = plt.gca()
ax.add_artist(leg1)
ax.add_artist(leg2)

# Hide the points and axes since they're not needed
ax.axis('off')

# Adjust layout
plt.tight_layout(rect=[0, 0.3, 1, 1])

# plt.show()
plt.savefig(f'{save_dir}/Fig_3_legends.png', dpi=500)
plt.clf()
print('completed!')