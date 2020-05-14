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

from ops_dataset import (
    create_noise,
    create_hist,
    center_of_mass
)


print('Attention create noise avec des 1 pour no ambient !')
print('Attention def SBR different de b_val now !')
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
I_up = I_up[0:h, 0:400]
intensity_image = intensity_image[0:h, 0:400]
#intensity_image_ref = intensity_image

### --- Normalize 
print('Normalize  ...')
min_up , max_up  = np.min(I_up), np.max(I_up)
I_up = (I_up- min_up)/(max_up-min_up)
min_i, max_i = np.amin(intensity_image), np.amax(intensity_image)
intensity_image = (intensity_image - min_i)/(max_i - min_i)
#intensity_image_ref = intensity_image
#rescale
#min_r, max_r = 0.8, 1
#intensity_image = intensity_image * 0.4 + 0.6

### --- Create Histograms 
print('Create Histograms  ...')
patch_depth_LR_norm = {}
patch_depth_LR_norm[0] = I_up
patch_intensity_norm = {}
patch_intensity_norm[0] = intensity_image
intensity_level = 10#3000
patch_histogram_before = create_hist(patch_depth_LR_norm, patch_intensity_norm, intensity_level)
print(patch_histogram_before[0].shape)
histogram_before = patch_histogram_before[0]
intensity_image_ref = np.sum(histogram_before, axis = 2)

#depth_HR_before_noise =  center_of_mass(patch_histogram_before[0], 'one_image')

### --- Create Noisy Histograms 
print('Create Noisy Histograms  ...')
SBR_mean = 0.04#0.4 #0.9
ambient_type = 'constant_SBR'
patch_histogram = create_noise(patch_histogram_before, SBR_mean, ambient_type)
histogram = patch_histogram[0]
print(histogram.shape)
depth_HR_before_down =  center_of_mass(histogram, 'one_image')

### --- Create HR intensity 
print('Create HR intensity  ...')
intensity_image_withb = np.sum(histogram, axis = 2)
#intensity_image = np.sum(histogram, 2)
#background = np.median(histogram,axis = 2)
background = SBR_mean*np.ones((histogram.shape[0], histogram.shape[1],histogram.shape[2]))
print(background.shape)

new_histogram = histogram-background
Nx = new_histogram.shape[0]
Ny = new_histogram.shape[1]
Nbins = new_histogram.shape[2]
for x in range(Nx):
    for y in range(Ny):
        for t in range(Nbins):
            if new_histogram[x,y,t]<0:
                new_histogram[x,y,t]=0
max_val = np.amax(new_histogram)
min_val = np.amin(new_histogram)
min_array = min_val*np.ones([Nx, Ny, Nbins])
max_array = max_val*np.ones([Nx, Ny, Nbins])
new_histogram = (new_histogram - min_array)/(max_val-min_val)

intensity_image = np.sum(new_histogram, axis = 2)
print(intensity_image.shape)

### --- Normalize intensity again 
print('Normalize intensity  ...')
#min_i, max_i = np.amin(intensity_image), np.amax(intensity_image)
#intensity_image = (intensity_image - min_i)/(max_i - min_i)
intensity_image = intensity_image

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
#save_path = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset/TEST/Synthetic_data/depth_4/DATA_TEST_art_i10_b1/'
save_path = os.path.join(Dir, 'depth_'+str(scale), 'DATA_TEST_art_rescaled_intensity_level='+str(intensity_level)+'_ambient_type='+str(ambient_type)+'_background='+str(SBR_mean))
#'_rescaled'+
print(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

sio.savemat(os.path.join(save_path,  "0_Df_down.mat" ),{'I_down':np.squeeze(depth_up)})
sio.savemat(os.path.join(save_path,  "0_Df.mat" ),{'I_up':np.squeeze(I_up)})
sio.savemat(os.path.join(save_path,  "0_pool1.mat" ),{'list_pool_1':np.squeeze(list_pool_1)})
sio.savemat(os.path.join(save_path,  "0_pool2.mat" ),{'list_pool_2':np.squeeze(list_pool_2)})
sio.savemat(os.path.join(save_path,  "0_pool3.mat" ),{'list_pool_3':np.squeeze(list_pool_3)})
sio.savemat(os.path.join(save_path,  "0_pool4.mat" ),{'list_pool_4':np.squeeze(list_pool_4)})
sio.savemat(os.path.join(save_path,  "0_RGB.mat" ),{'intensity_image':np.squeeze(intensity_image)})
sio.savemat(os.path.join(save_path,  "0_RGB_withb.mat" ),{'intensity_image_withb':np.squeeze(intensity_image_withb)})
sio.savemat(os.path.join(save_path,  "0_RGB_ref.mat" ),{'intensity_image_ref':np.squeeze(intensity_image_ref)})
#imageio.imwrite(os.path.join(save_path,  "0_RGB.bmp" ), intensity_image)
#imageio.imwrite(os.path.join(save_path,  "ref_RGB.bmp" ), intensity_image_ref)

sio.savemat(os.path.join(save_path,  "hist_before.mat" ),{'hist_before':np.squeeze(patch_histogram_before[0])})
sio.savemat(os.path.join(save_path,  "hist_after.mat" ),{'hist_after':np.squeeze(patch_histogram[0])})
sio.savemat(os.path.join(save_path,  "depth_HR_before_down.mat" ),{'depth_HR_before_down':np.squeeze(depth_HR_before_down)})
sio.savemat(os.path.join(save_path,  "depth_HR_after_down.mat" ),{'depth_HR_after_down':np.squeeze(depth)})
#sio.savemat(os.path.join(save_path,  "depth_HR_before_noise.mat" ),{'depth_HR_before_noise':np.squeeze(depth_HR_before_noise)})


fig, axarr = plt.subplots(2,4)
a1 = axarr[0,0].imshow(np.squeeze(I_up), cmap="gray")
axarr[0,0].set_title("Depth HR")
axarr[0,0].axis('off')
cbar1 = fig.colorbar(a1, ax =axarr[0,0],  cmap='gray')

a2 = axarr[0,1].imshow(np.squeeze(intensity_image), cmap="hot")
axarr[0,1].set_title("Intensity")
axarr[0,1].axis('off')
cbar1 = fig.colorbar(a1, ax =axarr[0,1],  cmap='hot')

a3 = axarr[0,2].imshow(np.squeeze(depth_up), cmap="gray")
axarr[0,2].set_title("Depth LR")
axarr[0,2].axis('off')
cbar1 = fig.colorbar(a3, ax =axarr[0,2],  cmap='gray')

axarr[0,3].axis('off')

a10 = axarr[1,0].imshow(np.squeeze(list_pool_1), cmap="gray")
axarr[1,0].set_title("Feature 1")
axarr[1,0].axis('off')
cbar10 = fig.colorbar(a10, ax =axarr[1,0], cmap='gray')

a11 = axarr[1,1].imshow(np.squeeze(list_pool_2), cmap="gray")
axarr[1,1].set_title("Feature 2")
axarr[1,1].axis('off')
cbar11 = fig.colorbar(a11, ax =axarr[1,1], cmap='gray')

a12 = axarr[1,2].imshow(np.squeeze(list_pool_3), cmap="gray")
axarr[1,2].set_title("Feature 3")
axarr[1,2].axis('off')
cbar12 = fig.colorbar(a12, ax =axarr[1,2], cmap='gray')

a13 = axarr[1,3].imshow(np.squeeze(list_pool_4), cmap="gray")
axarr[1,3].set_title("Feature 4")
axarr[1,3].axis('off')
cbar13 = fig.colorbar(a13, ax =axarr[1,3],cmap='gray')
plt.savefig(os.path.join(save_path, 'inputs.png'))
#plt.show()
plt.clf()
plt.cla()