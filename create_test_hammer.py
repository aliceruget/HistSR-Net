
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
from ops_dataset import center_of_mass
import numpy as np
import skimage.measure

# ---------------------------------------------------------------------------------
### 1. Clean Histograms from last and first row 
# ---------------------------------------------------------------------------------
print('Clean_Histogram...')
data_dir = '/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/Raw_histograms'
list_hist =glob.glob(os.path.join(data_dir,'0_hist.mat'))

for idx in range(len(list_hist)):
    hist_input_image   = glob.glob(os.path.join(data_dir,str(idx)+'_hist.mat'))
    histo = sio.loadmat(hist_input_image[0])['hist_LR_initial']
    histo[0,:,:]    = histo[1,:,:]
    histo[29,:,:]   = histo[28,:,:]
    histo[30,:,:]   = histo[28,:,:]
    histo[31,:,:]   = histo[28,:,:]
    image = center_of_mass(histo, 'one_image')
    sio.savemat('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/Clean_histograms/'+str(idx)+'_hist.mat', {'histogram' : histo})
    #sio.savemat('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/Depth/'+str(idx)+'_depth.mat', {'depth' : image})

# ---------------------------------------------------------------------------------
### 2. Prepare multi-scale calibration map 
# ---------------------------------------------------------------------------------

calibration = sio.loadmat('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/calibration/compensation_frame.mat')
calibration = 1/15 * calibration['compensation_frame'] 
calibration[0,:]   = calibration[1,:]
calibration[29,:]   = calibration[28,:]
calibration[30,:]   = calibration[28,:]
calibration[31,:]   = calibration[28,:]

calibration_input = np.kron(calibration,np.ones((4 , 4))) # input
calibration_input = np.reshape(calibration_input, (1,128,256,1))

calibration_pool1 = np.kron(calibration,np.ones((2 , 2))) 
calibration_pool1 = np.reshape(calibration_pool1, (1,64,128,1))

calibration_pool2 = np.reshape(calibration, (1,32,64,1))

calibration_pool3 = skimage.measure.block_reduce(calibration, (2,2), np.mean)
calibration_pool3 = np.reshape(calibration_pool3, (1,16,32,1))

calibration_pool4 = skimage.measure.block_reduce(calibration_pool3, (1,2,2,1), np.mean)
calibration_pool4 = np.reshape(calibration_pool4, (1,8,16,1))


# ---------------------------------------------------------------------------------
### 3. Compute Depth and Features
# ---------------------------------------------------------------------------------
print('Compute Depth and Features...')

data_dir =  '/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/Clean_histograms'
list_hist = glob.glob(os.path.join(data_dir,'0_hist.mat'))


for idx in range(0,len(list_hist)):
    h = 128
    w = 256
    c_dim = 1
    scale = 4
    Nbins = 15

    # Input
    hist_input_image   = glob.glob(os.path.join(data_dir,str(idx)+'_hist.mat'))
    histogram_down = sio.loadmat(hist_input_image[0])['histogram']
    histogram_down = histogram_down.reshape([1, int(h/scale), int(w/scale), Nbins])
    depth_down_before = center_of_mass(histogram_down.reshape([int(h/scale), int(w/scale), Nbins]), 'one_image')

    # -----------------------  Upsample Low resolution Histogram  --------------------------
    histogram_up = np.kron(histogram_down,np.ones((scale , scale , 1)))

    # -----------------------  From Histogram construct Input  --------------------------
    histogram_up =np.reshape(histogram_up, (histogram_up.shape[1], histogram_up.shape[2], histogram_up.shape[3]))
    depth_down = center_of_mass(histogram_up, 'one_image')
    #print('depth_down'+str(depth_down.shape))
    Nx = depth_down.shape[1]
    Ny = depth_down.shape[2]
    depth_down_bec = np.reshape(depth_down, [1,Nx, Ny,1])
    depth_down = depth_down_bec + calibration_input
    #print('depth_down'+str(depth_down.shape))

    # -----------------------  From Histogram construct Features Input Pyramid  --------------------------

    batch_histogram_down  = np.squeeze(histogram_up)
    Nx, Ny = batch_histogram_down.shape[0], batch_histogram_down.shape[1]
    Nbins = batch_histogram_down.shape[2]
        
    for scale in [2,4,8,16]:
        Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale),Nbins))
        i_x = 0
        for x in range(0,batch_histogram_down.shape[0],scale):
            i_y = 0
            for y in range(0,batch_histogram_down.shape[1],scale):
                for index_x in range(scale):
                    for index_y in range(scale):
                        Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * batch_histogram_down[x + index_x, y + index_y, :]
                i_y = i_y + 1
            i_x = i_x + 1
        Depth_LR = center_of_mass(Hist_LR, 'one_image')
        if scale == 2:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_1_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1])
            list_pool_1 = list_pool_1_bec + calibration_pool1
            #print('list_pool_1'+str(list_pool_1.shape))
        elif scale == 4:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_2_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1])
            list_pool_2 = list_pool_2_bec + calibration_pool2
            #print('list_pool_2'+str(list_pool_2.shape))
        elif scale == 8:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_3_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1]) 
            list_pool_3 = list_pool_3_bec + calibration_pool3
            #print('list_pool_3'+str(list_pool_3.shape))
            
        elif scale == 16:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_4_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1]) 
            list_pool_4 = list_pool_4_bec + calibration_pool4
            #print('list_pool_4'+str(list_pool_4.shape))
            

# ---------------------------------------------------------------------------------
### 5.   Save
# ---------------------------------------------------------------------------------
print('Save ...')

save_path = '/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/small_Data_TEST'
for idx in range(0,len(list_hist)):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sio.savemat(os.path.join(save_path,  str(idx)+"_Df_down.mat" ),{'I_down':np.squeeze(depth_down)})
    sio.savemat(os.path.join(save_path,  str(idx)+"_pool1.mat" ),{'list_pool_1':np.squeeze(list_pool_1)})
    sio.savemat(os.path.join(save_path,  str(idx)+"_pool2.mat" ),{'list_pool_2':np.squeeze(list_pool_2)})
    sio.savemat(os.path.join(save_path,  str(idx)+"_pool3.mat" ),{'list_pool_3':np.squeeze(list_pool_3)})
    sio.savemat(os.path.join(save_path,  str(idx)+"_pool4.mat" ),{'list_pool_4':np.squeeze(list_pool_4)})
    sio.savemat(os.path.join(save_path,  str(idx)+"_Df.mat" ),{'I_up':np.zeros((h,w))}) # stupid label


