import argparse
from datetime import datetime
import os
import yaml
import scipy
from scipy import io as sio
import scipy.misc
import numpy as np
import glob 
import random
import matplotlib.pyplot as plt
import skimage
import skimage.transform
#import cv2
import math
import time
from ops_dataset import *
from ops import *
import time

# ---- Inputs --------------------------------------------------------------------------------
parser = argparse.ArgumentParser('')
parser.add_argument('--config', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--downsample_type', type=str)
parser.add_argument('--data_type', type=str)
parser.add_argument('--Dir_import', type=str)

args = parser.parse_args()
Dir = args.out_dir #home/DepthSR_Net/Dataset
config = yaml.load(open(args.config, 'r'))
downsample_type = args.downsample_type
data_type = args.data_type
image_size = config['image_size']
stride = config['stride']
scale = config['scale']
batch_size = config['batch_size']
Dir_import = args.Dir_import
Directory = os.path.join(Dir, 'depth_'+str(scale))
if not os.path.exists(Directory):
    os.mkdir(Directory)
#d = datetime.now()
Directory = os.path.join(Directory,'DATA_TRAIN_DEPTH_noisy_intensity10_SBR_0_004_'+data_type+'_'+downsample_type)
if not os.path.exists(Directory):
    os.mkdir(Directory)

# ---- 1. Import data --------------------------------------------------------------------------
print(scale)
start_time_initial = time.time()
#print('!!!!!!!!!!!!!!ATTENTION PAS dE NOISE POISSON ICI !!!!!!!!!!!!!!')
print('ATTENTION SBR ! = b_val = 1')
print('ATTENTION intensity level of 10')
#print('ATTENTION QUE 2 IMAGES')
#print('ATTENTION QUE de 0 a 287')
#print('ATTENTION ratio 2')
print('Import data...\n')
if data_type == 'total':
    depth = sio.loadmat(os.path.join(Dir_import,'Raw_data' ,'Raw_data_Middlebury_MPI', 'depth_total.mat'))['depth_total']
    intensity = sio.loadmat(os.path.join(Dir_import, 'Raw_data','Raw_data_Middlebury_MPI', 'intensity_total.mat'))['intensity_total']
elif data_type == 'MPI':
    depth = sio.loadmat(os.path.join(Dir_import,'Raw_data' ,'Raw_data_Middlebury_MPI', 'depth_total.mat'))['depth_data_MPI']
    intensity = sio.loadmat(os.path.join(Dir_import, 'Raw_data','Raw_data_Middlebury_MPI', 'intensity_total.mat'))['intensity_data_MPI']
elif data_type == 'Middlebury':
    depth = sio.loadmat(os.path.join(Dir_import,'Raw_data' ,'Raw_data_Middlebury_MPI', 'depth_total.mat'))['depth_data_2006']
    intensity = sio.loadmat(os.path.join(Dir_import, 'Raw_data','Raw_data_Middlebury_MPI', 'intensity_total.mat'))['intensity_data_2006']
else:
    raise Exception('Data_type must be either total, MPI or Middlebury')

depth = np.squeeze(depth)
intensity = np.squeeze(intensity)
print(depth.shape)
depth = depth#[17:19]
print(depth.shape)
intensity = intensity#[17:19]

# ---- 1.bis Check NaN Inf values  -------------------------------------------------------------

print('Check NaN Inf values...')
count_nan , count_inf = 0 , 0
depth_new = {}
intensity_new = {}
new_index = 0
print('Initial nb of images = '+str(depth.shape[0]))
for index in range(0,depth.shape[0],1):
    depth_im = depth[index]
    intensity_im = intensity[index]
    
    if np.any(np.isnan(np.ndarray.flatten(depth_im))):
        count_nan = count_nan + 1
    elif np.any(np.isinf(np.ndarray.flatten(depth_im))):
        count_inf = count_inf + 1
    else:  
        depth_new[new_index]        = depth_im#[0:192]#287 
        intensity_new[new_index]    = intensity_im#[0:192]
        new_index = new_index + 1

depth = depth_new
intensity = intensity_new

#print('count_nan = '+str(count_nan))
#print('count_inf = '+str(count_inf) + '\n')
print('Final nb of images = '+str(len(depth))+ '\n')


# ---- 2. Split Dataset into Training and Validation  ------------------------------------------
print('Split Dataset into Training and Validation ratio...')
ratio_train_test = 1/8#
random.seed(2000)
indexes             =  np.random.permutation(len(depth))
index_validation    = indexes[range(0 , int(ratio_train_test*len(depth)) , 1)]
index_training      = indexes[range(int(ratio_train_test*len(depth)) , len(depth) , 1)]

intensity_validation = {}
depth_validation     = {}
new_index = 0 
for index in index_validation:
    intensity_validation[new_index] = intensity[index]
    depth_validation[new_index]     = depth[index]
    new_index = new_index + 1

intensity_training = {}
depth_training     = {}
new_index = 0 
for index in index_training:
    intensity_training[new_index] = intensity[index]
    depth_training[new_index]     = depth[index]
    new_index = new_index + 1   

print('Training : '+str(len(depth_training))+ ' images')
print('Validation : '+str(len(depth_validation))+ ' images\n')

# ---- 3. Flipping and rotation of Training and Validation dataset ---------------------------------------------
print('Flipping and rotation of Training dataset...')
intensity_training_aug = {}
depth_training_aug = {}
i = 0
for index in range(0 , len(intensity_training),1):
    intensity_im        = intensity_training[index]
    depth_im            = depth_training[index]

    intensity_im_flip   = np.flipud(intensity_im)
    depth_im_flip       = np.flipud(depth_im)

    for angle in range(4):
        intensity_im        = np.rot90(intensity_im)
        depth_im            = np.rot90(depth_im)
        intensity_im_flip   = np.rot90(intensity_im_flip)
        depth_im_flip       = np.rot90(depth_im_flip)

        intensity_training_aug[i]   = intensity_im
        intensity_training_aug[i+1] = intensity_im_flip
        depth_training_aug[i]       = depth_im
        depth_training_aug[i+1]     = depth_im_flip

        i = i + 2

print('Flipping and rotation of Validation dataset...')
intensity_validation_aug = {}
depth_validation_aug = {}
i = 0
for index in range(0 , len(intensity_validation),1):
    intensity_im        = intensity_validation[index]
    depth_im            = depth_validation[index]

    intensity_im_flip   = np.flipud(intensity_im)
    depth_im_flip       = np.flipud(depth_im)

    for angle in range(4):
        intensity_im        = np.rot90(intensity_im)
        depth_im            = np.rot90(depth_im)
        intensity_im_flip   = np.rot90(intensity_im_flip)
        depth_im_flip       = np.rot90(depth_im_flip)

        intensity_validation_aug[i]   = intensity_im
        intensity_validation_aug[i+1] = intensity_im_flip
        depth_validation_aug[i]       = depth_im
        depth_validation_aug[i+1]     = depth_im_flip

        i = i + 2
print('Training : '+str(len(depth_training_aug))+ ' images')
print('Validation : '+str(len(depth_validation))+ ' images\n')

# ---- 4. Create Patches -------------------------------------------------------------------------
print('Create Patches ...')

patch_training_intensity , patch_training_depth = create_patches(intensity_training_aug , depth_training_aug , image_size , stride)
print('Training : '+str(len(patch_training_intensity)) + ' patches')

patch_validation_intensity , patch_validation_depth = create_patches(intensity_validation_aug , depth_validation_aug , image_size , stride)
print('Validation : '+str(len(patch_validation_depth)) + ' patches\n')

# ---- 5. Normalization ----------------------------------------------------------------------------
print('Normalization ...')

patch_training_depth_norm = {}
patch_training_intensity_norm = {}
count = 0
index_save = 0
for index in range(len(patch_training_intensity)):
    intensity = patch_training_intensity[index]
    depth = patch_training_depth[index]
    min_i = np.amin(intensity)
    max_i = np.amax(intensity)
    min_d = np.amin(depth)
    max_d = np.amax(depth)
    if min_i == max_i:
        count = count + 1
    elif min_d == max_d:
        count = count + 1
    else:
        patch_training_depth_norm[index_save] = (depth - min_d) / (max_d - min_d)
        patch_training_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
        index_save = index_save + 1


index_save = 0
patch_validation_depth_norm = {}
patch_validation_intensity_norm = {}
for index in range(len(patch_validation_intensity)):
    intensity = patch_validation_intensity[index]
    depth = patch_validation_depth[index]
    min_i = np.amin(intensity)
    max_i = np.amax(intensity)
    min_d = np.amin(depth)
    max_d = np.amax(depth)
    if min_i == max_i:
        count = count + 1
    elif min_d == max_d:
        count = count + 1
    else:
        patch_validation_depth_norm[index_save] = (depth - min_d) / (max_d - min_d)
        patch_validation_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
        index_save = index_save + 1


print('Training : '+ str(len(patch_training_depth_norm))+' patches')

print('Validation : '+ str(len(patch_validation_depth_norm))+' patches\n')

# ---- 6. Create Histograms ------------------------------------------------------------------------
Nbins = 15
intensity_level = 10 #3000
print("Create Histograms ...")
Histogram_training_depth_LR = create_hist(patch_training_depth_norm , patch_training_intensity_norm,intensity_level)
print('Training : ' + str(len(Histogram_training_depth_LR))+' histograms of size '+ str(Histogram_training_depth_LR[0].shape))


Histogram_validation_depth_LR = create_hist(patch_validation_depth_norm , patch_validation_intensity_norm,intensity_level)
print('Validation : '+ str(len(Histogram_validation_depth_LR))+' histograms of size '+ str(Histogram_training_depth_LR[0].shape)+'\n')

# ---- 7. Add Noise ------------------------------------------------------------------------
print("Create Noisy Histograms ...")
SBR_mean = 0.004 #0.9
ambient_type = 'constant_SBR'
Histogram_training_depth_LR_noisy = create_noise(Histogram_training_depth_LR, SBR_mean, ambient_type)
print('Training : ' + str(len(Histogram_training_depth_LR_noisy))+' histograms of size '+ str(Histogram_training_depth_LR_noisy[0].shape))

Histogram_validation_depth_LR_noisy = create_noise(Histogram_validation_depth_LR, SBR_mean, ambient_type)
print('Validation : '+ str(len(Histogram_validation_depth_LR_noisy))+' histograms of size '+ str(Histogram_validation_depth_LR_noisy[0].shape)+'\n')

# ---- 8. Create HR intensity -------------------------------------
print("Create Intensity ...")
Histogram_validation_depth_LR_noisy
nb_patches = len(Histogram_training_depth_LR_noisy)
Intensity_training = {}
for index in range(nb_patches):
    patch = Histogram_training_depth_LR_noisy[index]
    #print(patch.shape)
    Intensity_training[index] = np.sum(patch, 2)

nb_patches = len(Histogram_validation_depth_LR_noisy)
Intensity_validation = {}
for index in range(nb_patches):
    patch = Histogram_validation_depth_LR_noisy[index]
    #print(patch.shape)
    Intensity_validation[index] = np.sum(patch, 2)
print('Training : ' + str(len(Intensity_training))+' intensity maps of size '+ str(Intensity_training[0].shape))
print('Validation : ' + str(len(Intensity_validation))+' intensity maps of size '+ str(Intensity_validation[0].shape)+'\n')

# -- Normalize intensity again -------------------------------------
print('Normalize intensity ...')
patch_training_intensity_norm = {}
count = 0
index_save = 0
for index in range(len(Intensity_training)):
    intensity = Intensity_training[index]
    min_i = np.amin(intensity)
    max_i = np.amax(intensity)
    if min_i == max_i:
        count = count + 1
    else:
        patch_training_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
        index_save = index_save + 1

index_save = 0
patch_validation_intensity_norm = {}
for index in range(len(Intensity_validation)):
    intensity = Intensity_validation[index]
    min_i = np.amin(intensity)
    max_i = np.amax(intensity)
    if min_i == max_i:
        count = count + 1
    else:
        patch_validation_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
        index_save = index_save + 1

print('Training : '+ str(len(patch_training_intensity_norm))+' patches')
print('Validation : '+ str(len(patch_validation_intensity_norm))+' patches\n')

# ---- 8. Downsample Histograms -----------------------------------------------------------------------
Histogram_training_depth_LR_DS = {}
for patch_idx in range(0, len(Histogram_training_depth_LR_noisy)): 
    histogram = Histogram_training_depth_LR_noisy[patch_idx]
    Nx = histogram.shape[0]
    Ny = histogram.shape[1]
    Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale), Nbins))
    #print(Hist_LR.shape)
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

    Histogram_training_depth_LR_DS[patch_idx] = Hist_LR

Histogram_validation_depth_LR_DS = {}
for patch_idx in range(0, len(Histogram_validation_depth_LR_noisy)): 
    histogram = Histogram_validation_depth_LR_noisy[patch_idx]
    Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale),Nbins))
    i_x = 0
    for x in range(0,histogram.shape[0],scale):
        i_y = 0
        for y in range(0,histogram.shape[1],scale):
            for index_x in range(scale):
                for index_y in range(scale):
                    Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * histogram[x + index_x, y + index_y, :]
            i_y = i_y + 1
        i_x = i_x + 1

    Histogram_validation_depth_LR_DS[patch_idx] = Hist_LR


# ---- 9. Center of Mass -------------------------------------------------------------------

print("Compute Input ...")
Patches_training_depth_LR = center_of_mass(Histogram_training_depth_LR_DS, 'dict')
print('Training : '+str(len(Patches_training_depth_LR))+' depths of size '+ str(Patches_training_depth_LR[0].shape))

Patches_validation_depth_LR = center_of_mass(Histogram_validation_depth_LR_DS, 'dict')
print('Validation : '+str(len(Patches_validation_depth_LR))+' depths of size '+ str(Patches_validation_depth_LR[0].shape)+'\n')

# ---- 10. Upsampling to get Input -------------------------------------------------------------------

Training_input = {}
for patch_idx in range(0, len(Patches_training_depth_LR)): 
    image = Patches_training_depth_LR[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Training_input[patch_idx] = image_up

Validation_input = {}
for patch_idx in range(0, len(Patches_validation_depth_LR)): 
    image = Patches_validation_depth_LR[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Validation_input[patch_idx] = image_up


# ----11. Downsample to get Feature 1 ------------------------------------------------------------
list_pool_1 = {}
for patch_idx in range(0, len(Training_input)): 
    image = Training_input[patch_idx]
    image = np.reshape(image,[1, image.shape[0],image.shape[1],1])
    image = max_pool_2x2(image)
    list_pool_1[patch_idx] = image


# ---- 10. Feature 2 ------------------------------------------------------------
list_pool_2 = {}
for patch_idx in range(0, len(list_pool_1)): 
    image = list_pool_1[patch_idx]
    image = max_pool_2x2(image)
    list_pool_2[patch_idx] = image


# ---- 11. Feature 3  ---------------------------------------------------------
list_pool_3 = {}
for patch_idx in range(0, len(list_pool_2)): 
    image = list_pool_2[patch_idx]
    image = max_pool_2x2(image)
    list_pool_3[patch_idx] = image

# ---- 11. Feature 4  ---------------------------------------------------------
list_pool_4 = {}
for patch_idx in range(0, len(list_pool_2)): 
    image = list_pool_3[patch_idx]
    image = max_pool_2x2(image)
    list_pool_4[patch_idx] = image



print(str(len(list_pool_1))+' depths for Feature 1 of size '+ str(list_pool_1[0].shape))
print(str(len(list_pool_2))+' depths for Feature 2 of size '+ str(list_pool_2[0].shape))
print(str(len(list_pool_3))+' depths for Feature 3 of size '+ str(list_pool_3[0].shape))
print(str(len(list_pool_4))+' depths for Feature 4 of size '+ str(list_pool_4[0].shape))


# ---- 11. Save --------------------------------------------------------------------------------------

#training
print('Save ...')
index_save = 0
print(len(patch_training_depth_norm))
print(Directory + '\n')
image_size = 96 
image_size_2 = 48
image_size_4 = 24
image_size_8 = 12
image_size_16 = 6
for index_batch in range(0 , len(patch_training_depth_norm) - batch_size , batch_size):
    #if np.mod(index_batch, 5000):
    #    print(index_batch)
    for index_image in range(0 , batch_size , 1):
        #print(index_image)
        
        depth_HR    = np.reshape(patch_training_depth_norm[index_batch + index_image] , (1, image_size,image_size))
        intensity   = np.reshape(patch_training_intensity_norm[index_batch + index_image] , (1, image_size,image_size))
        depth_LR = np.reshape(Training_input[index_batch + index_image] , (1, image_size,image_size))
        pool_1 = np.reshape(list_pool_1[index_batch + index_image] , (1, image_size_2 ,image_size_2))
        pool_2 = np.reshape(list_pool_2[index_batch + index_image] , (1, image_size_4 ,image_size_4))
        pool_3 = np.reshape(list_pool_3[index_batch + index_image] , (1, image_size_8 ,image_size_8))
        pool_4 = np.reshape(list_pool_4[index_batch + index_image] , (1, image_size_16 ,image_size_16))

        if index_image == 0:
            batch_depth_HR  = depth_HR
            batch_intensity = intensity
            batch_depth_LR  = depth_LR
            batch_pool_1    = pool_1
            batch_pool_2    = pool_2
            batch_pool_3    = pool_3
            batch_pool_4    = pool_4   
            
        else:
            batch_depth_HR  = np.concatenate((batch_depth_HR , depth_HR), axis=0)
            batch_intensity = np.concatenate((batch_intensity , intensity), axis=0)
            batch_depth_LR  = np.concatenate((batch_depth_LR , depth_LR), axis=0)
            batch_pool_1    = np.concatenate((batch_pool_1 , pool_1), axis=0)
            batch_pool_2    = np.concatenate((batch_pool_2 , pool_2), axis=0)
            batch_pool_3    = np.concatenate((batch_pool_3 , pool_3), axis=0)
            batch_pool_4    = np.concatenate((batch_pool_4 , pool_4), axis=0)

    dict_HR      = {}
    dict_hist_LR = {}
    dict_i       = {}
    dict_LR      = {}
    dict_pool1 = {}
    dict_pool2 = {}
    dict_pool3 = {}
    dict_pool4 = {}

    dict_HR['batch_depth_HR'] = batch_depth_HR
    dict_i['batch_intensity'] = batch_intensity
    dict_LR['batch_depth_LR']  = batch_depth_LR
    dict_pool1['batch_pool1'] = batch_pool_1
    dict_pool2['batch_pool2'] = batch_pool_2
    dict_pool3['batch_pool3'] = batch_pool_3
    dict_pool4['batch_pool4'] = batch_pool_4

    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_depth_label.mat'), dict_HR)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_I_add.mat'), dict_i)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_depth_down.mat'), dict_LR)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool1.mat'), dict_pool1)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool2.mat'), dict_pool2)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool3.mat'), dict_pool3)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool4.mat'), dict_pool4)
    index_save = index_save + 1

print('Training : '+str(index_save)+' batches')

