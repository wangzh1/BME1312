import math
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

from CS_mask import cartesian_mask
from helper_fn import imsshow

dataset = np.load('./cine.npz')['dataset']

# Index of cine img to process
CINE_INDEX = 100
mask = cartesian_mask(shape=(1, 20, 192, 192), acc=8, sample_n=10, centred=True)
imsshow(mask[0, :, :, :], num_col=5, cmap='gray')

# To get the first column of each image, use indexing with ':,0,0'
# Squeeze the third dimension to get a 20*192 matrix

mask_get_ky = mask[0, :, :, 0:1]

squeezed_matrix = np.squeeze(mask_get_ky, axis=2)

# Transpose the result to get a 192*20 matrix
result_matrix = np.transpose(squeezed_matrix) # [20, 192, 1]

# Transpose the result to get a 192*20 matrix
plt.imshow(result_matrix, aspect=0.3, cmap='gray')
plt.xticks(np.arange(0, 20, 4))
plt.show()