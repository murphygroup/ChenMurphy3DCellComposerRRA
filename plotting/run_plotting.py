import subprocess
import os
import time
plotting_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(plotting_dir)

print('installing necessary packages...')
time.sleep(1)
# Run 'pip install -r requirements.txt'
requirements_file = os.path.join(plotting_dir, 'requirements.txt')

if os.path.exists(requirements_file):
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
else:
    print("requirements.txt file not found.")
print('completed!')

if not os.path.exists('../fig'):
    os.makedirs('../fig')
    
if not os.path.exists('../table'):
    os.makedirs('../table')

scripts = [
	"preprocessing.py",
    "Figure_1_B.py",
    "Figure_1_C.py",
    "Figure_1_D.py",
    "Figure_2_coloring.py",
    "Figure_2_meshing.py",
    "Figure_3.py",
    "Figure_3_legend.py",
    "Figure_4.py",
    "Supp_figure_1.py",
    "Supp_figure_2_4.py",
    "Supp_figure_3.py",
    "Supp_figure_5.py",
    "Supp_figure_6.py",
    "Supp_table_7.py"
]

for script_path in scripts:
    subprocess.run(["python", script_path])
