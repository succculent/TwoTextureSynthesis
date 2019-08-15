import numpy as np
from scipy.ndimage import gaussian_filter

def gen_mask(size): #size must be multiple of 16
	#This is where the mask is defined

	###This is one example of a mask###
	temp = np.zeros((4,4))
	for i in range(0, 4):
		for j in range(0, 4):
			if ((i+j)%2 == 0):
				temp[i, j] = 1
	clean_mask = np.zeros((size,size))
	k = int(size/4)
	for i in range(0, size):
		for j in range(0, size):
			clean_mask[i, j] = temp[i//k, j//k]
	mask = gaussian_filter(clean_mask, sigma=size/16)
	return mask