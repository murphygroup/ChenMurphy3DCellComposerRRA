import subprocess
import os

plotting_dir = "./plotting"

scripts = [
    "Figure_1_B.py",
    "Figure_1_C.py",
    "Figure_1_D.py",
    "Figure_2_coloring.py",
    "Figure_2_meshing.py",
    "Figure_3.py",
    "Figure_3_legend.py",
    "Figure_4.py",
    "Supp_figure_2.py",
    "Supp_figure_4.py",
    "Supp_figure_5.py"
]

for script in scripts:
    script_path = os.path.join(plotting_dir, script)
    subprocess.run(["python", script_path])

