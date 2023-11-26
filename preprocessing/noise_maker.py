import numpy as np
import os
import cv2
from skimage.io import imread
from skimage.io import imsave
from os.path import join
import sys
import matplotlib.pyplot as plt
import argparse
import glob
np.random.seed(3)


def add_noise(noise_typ, image, sigma):
	if noise_typ == "gauss":
		row, col, ch = image.shape
		mean = 0
		gauss = np.random.normal(mean, sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row, col, ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		          for i in image.shape]
		out[coords] = 1
		
		# Pepper mode
		num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
		          for i in image.shape]
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ == "speckle":
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy
	
if __name__ == '__main__':
	data_dir = sys.argv[1]
	noise_num = int(sys.argv[2])
	noise_interval = int(sys.argv[3])
	img_dir_list = sorted(glob.glob(f'{data_dir}/**/**/original'))
	# for i in range(args.noise_num + 1):
	for img_dir in img_dir_list:
		for i in range(1, noise_num+1):
			gaussian_dir = join(os.path.dirname(img_dir), 'random_gaussian_' + str(i))
			if not os.path.exists(join(gaussian_dir, 'membrane.tif')):
				try:
					os.makedirs(gaussian_dir)
				except:
					pass
			for img_name in ['nucleus', 'cytoplasm', 'membrane']:
				img = imread(join(img_dir, img_name + '.tif'))
				# w = int(sys.argv[4])
				# h = int(sys.argv[5])
				# center = [img.shape[0] / 2, img.shape[1] / 2]
				# x = center[1] - w/2
				# y = center[0] - h/2
				# img = img[int(y):int(y+h), int(x):int(x+w)]
				# img = np.expand_dims(img, axis=-1)
				img_noisy = add_noise('gauss', img, i * noise_interval)
				# img_noisy = np.squeeze(img_noisy, axis=-1)
				img_noisy[np.where(img_noisy < 0)] = 0
				img_noisy[np.where(img_noisy > 65535)] = 65535
				img_noisy = img_noisy.astype('uint16')
				imsave(join(gaussian_dir, img_name + '.tif'), img_noisy)
				
					# for img_name in ['nucleus', 'cytoplasm', 'membrane']:
					# 	img = imread(join(img_dir, img_name + '.tif'))
					# 	img = np.expand_dims(img, axis=-1)
					# 	img_noisy = add_noise('gauss', img, i * args.noise_interval)
					# 	img_noisy = np.squeeze(img_noisy, axis=-1)
					# 	img_noisy[np.where(img_noisy < 0)] = 0
					# 	img_noisy[np.where(img_noisy > 65535)] = 65535
					# 	img_noisy = img_noisy.astype('uint16')
					# 	imsave(join(gaussian_dir, img_name + '.tif'), img_noisy)
				
				# # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
				# from scipy.ndimage import gaussian_filter
				# img_blur = gaussian_filter(img, sigma=i / 5)
				# img_blur = np.squeeze(img_blur, axis=-1)
				# img_blur[np.where(img_blur > 65535)] = 65535
				# smooth_dir = join(file_dir, img_dir, 'random_smooth_' + str(i))
				# if not os.path.exists(smooth_dir):
				# 	os.makedirs(smooth_dir)
				# imsave(join(smooth_dir, img_name + '.tif'), img_blur)
