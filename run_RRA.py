import subprocess
import os


def run_download_hiPSC_script():
	script_path = './downloading/download_hiPSC.py'
	print('Downloading 3D hiPSC multiplexed cell culture images..')
	# Execute the script
	subprocess.run(['python', script_path])


def run_download_IMC_script():
	script_path = './downloading/download_IMC.py'
	print('Downloading 3D IMC multiplexed tissue images..')
	# Execute the script
	subprocess.run(['python', script_path])


def run_preprocess_IMC_script():
	script_path = './preprocessing/preprocess_IMC.py'
	print('Preprocessing 3D IMC multiplexed tissue images..')
	# Execute the script
	subprocess.run(['python', script_path])


def run_preprocess_hiPSC_script():
	script_path = './preprocessing/preprocess_AICS.py'
	print('Preprocessing 3D hiPSC multiplexed cell culture images..')
	# Execute the script
	subprocess.run(['python', script_path])


def run_perturb_IMC_script():
	script_path = './preprocessing/noise_maker.py'
	print('Perturbing 3D IMC multiplexed tissue images w/ Gaussian noise..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/IMC_3D', '3', '5'])


def run_perturb_hiPSC_script():
	script_path = './preprocessing/noise_maker.py'
	print('Perturbing 3D hiPSC multiplexed cell culture images w/ Gaussian noise..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/AICS', '3', '30'])


def run_segmentation_2D_IMC_script():
	script_path = './segmentation_2D/run_segmentation_2D.py'
	print('Segmenting slices of 3D IMC multiplexed tissue images..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/IMC_3D'])


def run_segmentation_2D_hiPSC_script():
	script_path = './segmentation_2D/run_segmentation_2D.py'
	print('Segmenting slices of 3D hiPSC multiplexed cell culture images..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/AICS'])


def run_segmentation_3D_IMC_script():
	script_path = './segmentation_3D/run_segmentation_3D.py'
	print('Perturbing 3D IMC multiplexed tissue images w/ Gaussian noise..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/IMC_3D'])


def run_segmentation_3D_hiPSC_script():
	script_path = './segmentation_3D/run_segmentation_3D.py'
	print('Perturbing 3D hiPSC multiplexed cell culture images w/ Gaussian noise..')
	# Execute the script
	subprocess.run(['python', script_path, f'{os.getcwd()}/data/AICS'])


# Run the function
if __name__ == "__main__":
	print("Current working directory:", os.getcwd())
	# run_download_IMC_script()
	# run_download_hiPSC_script()
	# run_preprocess_IMC_script()
	# run_preprocess_hiPSC_script()
	run_perturb_IMC_script()
	run_perturb_hiPSC_script()
	run_segmentation_2D_IMC_script()
	run_segmentation_2D_hiPSC_script()
	run_segmentation_3D_IMC_script()
	run_segmentation_3D_hiPSC_script()
	run_evaluation_IMC_script()
	run_evaluation_hiPSC_script()
	run_plotting_IMC_script()
	run_plotting_hiPSC_script()

# from skimage.io import imread
# import numpy as np
#
# imread('/home/hrchen/Documents/Research/hubmap/script/2D-3D/ChenMurphy3DCellComposerRRA/data/AICS/AICS_golgi/AICS_15978/random_gaussian_10/nucleus.tif')
# imread('/data/3D/AICS/AICS_golgi/AICS_15978/original/nucleus.tif')
# imread('/data/3D/AICS/AICS_golgi/AICS_15978/random_gaussian_2/nucleus.tif')