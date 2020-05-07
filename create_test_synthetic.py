from skimage import io
import os
from PIL import Image
import scipy
from scipy import misc, ndimage
import scipy.io as sio
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ops_dataset import (
    create_noise,
    create_hist,
    center_of_mass
)



# Notes : here we add noise at the level of HR histogram and then we downsample to get input and features. 
### --- Define Scale 
scale = 4
downsample_type = 'nearest'

### --- Import data 
print('Import data ...')
Import_Dir = '/Users/aliceruget/Documents/PhD/Dataset/'

depth = sio.loadmat(os.path.join(Import_Dir,'RAW','Middlebury','Middlebury_dataset' ,'2005', 'depth_data.mat'))['depth_data_2005']
intensity = sio.loadmat(os.path.join(Import_Dir,'RAW','Middlebury','Middlebury_dataset' ,'2005', 'intensity_data.mat'))['intensity_data_2005']

I_up = np.squeeze(depth[0,0])
intensity_image = np.squeeze(intensity[0,0])

### --- Crop images modulo scale
h, w = I_up.shape
h = h - np.mod(h, 16)
w = w - np.mod(w, 16)
I_up = I_up[0:h, 0:w]
intensity_image = intensity_image[0:h, 0:w]

### --- Normalize 
print('Normalize  ...')
min_up , max_up  = np.min(I_up), np.max(I_up)
I_up = (I_up- min_up)/(max_up-min_up)
min_i, max_i = np.min(intensity_image), np.max(intensity_image)
intensity_image = (intensity_image-min_i)/(max_i-min_i)


### --- Create Histograms 
print('Create Histograms  ...')
patch_depth_LR_norm = {}
patch_depth_LR_norm[0] = I_up
patch_intensity_norm = {}
patch_intensity_norm[0] = intensity_image
patch_histogram = create_hist(patch_depth_LR_norm, patch_intensity_norm, 1)
print(patch_histogram[0].shape)

### --- Create Noisy Histograms 
print('Create Noisy Histograms  ...')
SBR_mean = 0.41 #0.9
no_ambient = 0 
patch_histogram = create_noise(patch_histogram, SBR_mean, no_ambient)
histogram = patch_histogram[0]
print(histogram.shape)

### --- Create HR intensity 
print('Create HR intensity  ...')
intensity_image = np.sum(histogram, 2)
print(intensity_image.shape)

### --- Normalize intensity again 
print('Normalize intensity  ...')
min_i, max_i = np.min(intensity_image), np.max(intensity_image)
intensity_image = (intensity_image - min_i)/(max_i - min_i)


### --- Downsample Histograms 
print('Downsample Histograms  ...')
Nx = histogram.shape[0]
Ny = histogram.shape[1]
Nbins = histogram.shape[2]
scale = 4
Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale),Nbins))
i_x = 0
for x in range(0,histogram.shape[0],scale):
    #print(i_x)
    i_y = 0
    for y in range(0,histogram.shape[1],scale):
        #print(i_y)
        for index_x in range(scale):
            for index_y in range(scale):
                Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * histogram[x + index_x, y + index_y, :]
        i_y = i_y + 1
    i_x = i_x + 1

histogram = Hist_LR
print(histogram.shape)

### --- Center of Mass 
depth = center_of_mass(histogram, 'one_image')
print(depth.shape)

### --- Upsampling to get Input 
print('Input ...')
scale = 4
depth_up = np.kron(depth , np.ones((scale,scale)))
print(depth_up.shape)

### ---  Upsampling to get Feature 1 
print('Feature 1 ...')
scale_1 = 2
image_up = np.kron(depth , np.ones((scale_1,scale_1)))
list_pool_1 = image_up
print(list_pool_1.shape)

### --- Feature 2 
print('Feature 2 ...')
list_pool_2 = depth
print(list_pool_2.shape)

### --- Feature 3 
print('Feature 3 ...')
scale_3 = 2
Nbins = 15
Nx = histogram.shape[0]
Ny = histogram.shape[1]
Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
i_x = 0
for x in range(0,histogram.shape[0],scale_3):
    i_y = 0
    for y in range(0,histogram.shape[1],scale_3):
        for index_x in range(scale_3):
            for index_y in range(scale_3):
                Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
        i_y = i_y + 1
    i_x = i_x + 1
list_pool_3 = center_of_mass(Hist_LR, 'one_image')
print(list_pool_3.shape)

### --- Feature 4 
print('Feature 4 ...')
scale_3 = 4
Nbins = 15
Nx = histogram.shape[0]
Ny = histogram.shape[1]
Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
i_x = 0
for x in range(0,histogram.shape[0],scale_3):
    i_y = 0
    for y in range(0,histogram.shape[1],scale_3):
        for index_x in range(scale_3):
            for index_y in range(scale_3):
                Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
        i_y = i_y + 1
    i_x = i_x + 1
list_pool_4 = center_of_mass(Hist_LR, 'one_image')
print(list_pool_4.shape)

### --- Save 
print('Save ...')
Dir = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset/create_testing/Synthetic_data'
save_path = os.path.join(Dir, 'depth_'+str(scale), 'DATA_TEST_art')
print(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

sio.savemat(os.path.join(save_path,  "0_Df_down.mat" ),{'I_down':np.squeeze(depth_up)})
sio.savemat(os.path.join(save_path,  "0_Df.mat" ),{'I_up':np.squeeze(I_up)})
sio.savemat(os.path.join(save_path,  "0_pool1.mat" ),{'list_pool_1':np.squeeze(list_pool_1)})
sio.savemat(os.path.join(save_path,  "0_pool2.mat" ),{'list_pool_2':np.squeeze(list_pool_2)})
sio.savemat(os.path.join(save_path,  "0_pool3.mat" ),{'list_pool_3':np.squeeze(list_pool_3)})
sio.savemat(os.path.join(save_path,  "0_pool4.mat" ),{'list_pool_4':np.squeeze(list_pool_4)})
imageio.imwrite(os.path.join(save_path,  "0_RGB.bmp" ), intensity_image)