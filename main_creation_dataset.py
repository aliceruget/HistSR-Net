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
import time
from create_histograms import *

# ---- Inputs --------------------------------------------------------------------------------

parser = argparse.ArgumentParser('')
parser.add_argument('--config', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--data_type', type=str)
parser.add_argument('--Dir_import', type=str)

args = parser.parse_args()
Dir = args.out_dir #home/DepthSR_Net/Dataset
config = yaml.load(open(args.config, 'r'))
image_size = config['image_size']
stride = config['stride']
scale = config['scale']
batch_size = config['batch_size']
data_type = args.data_type
Dir_import = args.Dir_import

Directory = os.path.join(Dir, 'depth_'+str(scale))
if not os.path.exists(Directory):
    os.mkdir(Directory)

Directory = os.path.join(Directory,'DATA_TRAIN_HISTOGRAM_noisy_intensity10_SBR_0_004_'+data_type)
if not os.path.exists(Directory):
    os.mkdir(Directory)


# ---- Create LR Histogram and HR Intensity ------------------------------------------------------
Histogram_training_depth_LR_DS, Histogram_validation_depth_LR_DS, patch_training_intensity_norm, patch_validation_intensity_norm,patch_training_depth_norm,patch_validation_depth_norm = create_histograms(config,data_type,Dir_import)


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


# ----11. Upsampling to get Feature 1 ------------------------------------------------------------
list_pool_1 = {}
scale_1 = 2
for patch_idx in range(0, len(Patches_training_depth_LR)): 
    image = Patches_training_depth_LR[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1[patch_idx] = image_up

list_pool_1_val = {}
for patch_idx in range(0, len(Patches_validation_depth_LR)): 
    image = Patches_validation_depth_LR[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1_val[patch_idx] = image_up

# ---- 10. Feature 2 ------------------------------------------------------------
list_pool_2 = Patches_training_depth_LR
list_pool_2_val = Patches_validation_depth_LR


# ---- 11. Feature 3 : DS histogram + center of mass -----------------------------------------
pool_3_hist = {}
scale_3 = 2
Nbins = 15
for patch_idx in range(0, len(Histogram_training_depth_LR_DS)): 
    histogram = Histogram_training_depth_LR_DS[patch_idx]
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
    pool_3_hist[patch_idx] = Hist_LR
list_pool_3 = center_of_mass(pool_3_hist, 'dict')

pool_3_hist_val = {}
scale_3 = 2
Nbins = 15
for patch_idx in range(0, len(Histogram_validation_depth_LR_DS)): 
    histogram = Histogram_validation_depth_LR_DS[patch_idx]
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
    pool_3_hist_val[patch_idx] = Hist_LR
list_pool_3_val = center_of_mass(pool_3_hist_val, 'dict')


# ---- 12. Feature 4 : DS histogram + center of mass -----------------------------------------
pool_4_hist = {}
scale_3 = 4
Nbins = 15
for patch_idx in range(0, len(Histogram_training_depth_LR_DS)): 
    histogram = Histogram_training_depth_LR_DS[patch_idx]
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
    pool_4_hist[patch_idx] = Hist_LR
list_pool_4 = center_of_mass(pool_4_hist, 'dict')

pool_4_hist_val = {}
scale_3 = 4
Nbins = 15
for patch_idx in range(0, len(Histogram_validation_depth_LR_DS)): 
    histogram = Histogram_validation_depth_LR_DS[patch_idx]
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
    pool_4_hist_val[patch_idx] = Hist_LR
list_pool_4_val = center_of_mass(pool_4_hist_val, 'dict')

for scale in [2,4,8,16]:
    Nx = 96
    if scale == 2:
        image_size_2 = int(Nx/scale)
    elif scale == 4:
        image_size_4 = int(Nx/scale)
    elif scale == 8:
        image_size_8 = int(Nx/scale)
    elif scale == 16:
        image_size_16 = int(Nx/scale)

print(str(len(list_pool_1))+' depths for Feature 1 of size '+ str(list_pool_1[0].shape))
print(str(len(list_pool_2))+' depths for Feature 2 of size '+ str(list_pool_2[0].shape))
print(str(len(list_pool_3))+' depths for Feature 3 of size '+ str(list_pool_3[0].shape))
print(str(len(list_pool_4))+' depths for Feature 4 of size '+ str(list_pool_4[0].shape))


print(str(len(list_pool_1_val))+' depths for Feature 1 of size '+ str(list_pool_1_val[0].shape))
print(str(len(list_pool_2_val))+' depths for Feature 2 of size '+ str(list_pool_2_val[0].shape))
print(str(len(list_pool_3_val))+' depths for Feature 3 of size '+ str(list_pool_3_val[0].shape))
print(str(len(list_pool_4_val))+' depths for Feature 4 of size '+ str(list_pool_4_val[0].shape)+'\n')


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

#validation
index_save = 1
for index_image in range(0,len(patch_validation_depth_norm),1):
    dict_HR      = {}
    dict_i       = {}
    dict_LR      = {}
    dict_pool1 = {}
    dict_pool2 = {}
    dict_pool3 = {}
    dict_pool4 = {}

    dict_HR['depth_label'] = np.reshape(patch_validation_depth_norm[index_image], (96,96))
    dict_i['I_add'] = np.reshape(patch_validation_intensity_norm[index_image], (96,96))
    dict_LR['batch_depth_LR'] = np.reshape(Validation_input[index_image] , (96, 96))
    dict_pool1['batch_pool_1'] = np.reshape(list_pool_1_val[index_image] , (48 ,48))
    dict_pool2['batch_pool_2'] = np.reshape(list_pool_2_val[index_image] , (24 ,24))
    dict_pool3['batch_pool_3'] = np.reshape(list_pool_3_val[index_image] , (12 ,12))
    dict_pool4['batch_pool_4'] = np.reshape(list_pool_4_val[index_image] , (6 ,6))

    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_depth_label_test.mat'), dict_HR)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_I_add_test.mat'), dict_i)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_depth_down_test.mat'), dict_LR)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool1_test.mat'), dict_pool1)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool2_test.mat'), dict_pool2)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool3_test.mat'), dict_pool3)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool4_test.mat'), dict_pool4)

    index_save = index_save + 1

print('Validation : '+str(index_save)+' batches')
print('Done!')

# #save data for Abderrahim
# dictionnaire = {}
# for index in range(10):
#     dictionnaire['Depth_HR'] = patch_training_depth_norm[index]
#     dictionnaire['Depth_LR'] = Training_input[index]
#     dictionnaire['Hist_HR'] = Histogram_training_depth_LR_noisy[index]
#     dictionnaire['Intensity'] = patch_training_intensity_norm[index]
#     dictionnaire['Hist_feature_2'] = Histogram_training_depth_LR_DS[index]
#     dictionnaire['Hist_feature_3'] = pool_3_hist[index]
#     dictionnaire['Hist_feature_4'] = pool_4_hist[index]
#     dictionnaire['Depth_feature_1'] = list_pool_1[index]
#     dictionnaire['Depth_feature_2'] = list_pool_2[index]
#     dictionnaire['Depth_feature_3'] = list_pool_3[index]
#     dictionnaire['Depth_feature_4'] = list_pool_4[index]
#     scipy.io.savemat(os.path.join(Directory, str(index)+'_data.mat'), dictionnaire)


# --- Plot 
for index in range(10):
    fig, axarr = plt.subplots(2,4)
    a1 = axarr[0,0].imshow(np.squeeze(patch_training_depth_norm[index]), cmap="gray")
    axarr[0,0].set_title("Depth HR")
    axarr[0,0].axis('off')
    cbar1 = fig.colorbar(a1, ax =axarr[0,0],  cmap='gray')

    a2 = axarr[0,1].imshow(np.squeeze(patch_training_intensity_norm[index]), cmap="hot")
    axarr[0,1].set_title("Intensity")
    axarr[0,1].axis('off')
    cbar1 = fig.colorbar(a1, ax =axarr[0,1],  cmap='hot')

    a3 = axarr[0,2].imshow(np.squeeze(Training_input[index]), cmap="gray")
    axarr[0,2].set_title("Depth LR")
    axarr[0,2].axis('off')
    cbar1 = fig.colorbar(a3, ax =axarr[0,2],  cmap='gray')

    axarr[0,3].axis('off')

    a10 = axarr[1,0].imshow(np.squeeze(list_pool_1[index]), cmap="gray")
    axarr[1,0].set_title("Feature 1")
    axarr[1,0].axis('off')
    cbar10 = fig.colorbar(a10, ax =axarr[1,0], cmap='gray')

    a11 = axarr[1,1].imshow(np.squeeze(list_pool_2[index]), cmap="gray")
    axarr[1,1].set_title("Feature 2")
    axarr[1,1].axis('off')
    cbar11 = fig.colorbar(a11, ax =axarr[1,1], cmap='gray')

    a12 = axarr[1,2].imshow(np.squeeze(list_pool_3[index]), cmap="gray")
    axarr[1,2].set_title("Feature 3")
    axarr[1,2].axis('off')
    cbar12 = fig.colorbar(a12, ax =axarr[1,2], cmap='gray')

    a13 = axarr[1,3].imshow(np.squeeze(list_pool_4[index]), cmap="gray")
    axarr[1,3].set_title("Feature 4")
    axarr[1,3].axis('off')
    cbar13 = fig.colorbar(a13, ax =axarr[1,3],cmap='gray')
    plt.savefig(os.path.join(Directory, str(index)+'_inputs.png'))
    #plt.show()
    plt.clf()
    plt.cla()