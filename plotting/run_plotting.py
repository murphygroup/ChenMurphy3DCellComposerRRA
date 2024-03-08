import subprocess
import os

plotting_dir = "./plotting"

scripts = [
	"Figure_1_B.py",
	"Figure_1_C.py",
	"Figure_1_D.py",
	"Figure_2_coloring.py",
	"Figure_2_meshing.py",
	"Supp_figure_2_4.py",
	"Figure_3.py",
	"Figure_3_legend.py",
	"Figure_4.py",
	"Supp_figure_1.py",
	"Supp_figure_3.py",
	"Supp_figure_5.py",
	"Supp_figure_6.py"
	"Supp_table_7.py"
]

for script in scripts:
	script_path = os.path.join(plotting_dir, script)
	subprocess.run(["python", script_path])
