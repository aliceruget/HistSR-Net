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
plt.clf()
plt.cla()
index_feature = 3
#data_dir = '/home/ar432/Hist_SR_Net/Dataset/Train/depth_4/DATA_TRAIN_HISTOGRAM_noisy_intensity10_SBR_0_004_MPI_nearest'
data_dir = '/home/ar432/Hist_SR_Net/Dataset/Train/depth_4/DATA_TRAIN_DEPTH_noisy_intensity10_SBR_0_004_MPI_nearest'
list_1 = glob.glob(os.path.join(data_dir,'*_patch_pool4.mat'))
list_dd = glob.glob(os.path.join(data_dir,'*_patch_depth_down.mat'))
Nb = len(list_1)

print(Nb)
tab_min =[]
tab_max =[]
tab_dif = []
for idx in range(Nb):
    patch_4 = glob.glob(os.path.join(data_dir,str(idx)+'_patch_pool'+str(index_feature)+'.mat'))
    dd = glob.glob(os.path.join(data_dir,str(idx)+'_patch_depth_down.mat'))

    feature_4 = sio.loadmat(patch_4[0])['batch_pool'+str(index_feature)]
    feature_4_up = np.kron(feature_4 , np.ones((1,8,8)))
    depth_down = sio.loadmat(dd[0])['batch_depth_LR']
    N = feature_4.shape[0]
    for index in range(N):
        patch_feature_4 = np.squeeze(feature_4_up[index,:,:])
        patch_dd = np.squeeze(depth_down[index,:,:])
        dif = np.abs(patch_feature_4 - patch_dd)
        dif_val = np.sum(dif)
        tab_dif.append(dif_val)


print(len(tab_dif))
plt.plot(tab_dif)
plt.savefig(os.path.join(data_dir, 'plot_dif_'+str(index_feature)+'_dd.png'))





