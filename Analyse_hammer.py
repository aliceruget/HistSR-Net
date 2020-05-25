
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
# save_path = '/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/SBR_function_ppp'


# data_dir = '/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data/Clean_histograms'
# list_hist =glob.glob(os.path.join(data_dir,'*_hist.mat'))
# histogram = sio.loadmat(list_hist[0])['histogram']


#data_dir = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset/create_testing/Synthetic_data/depth_4/DATA_TEST_art_rescaled_intensity_level=3000_no_ambient=0_background=0.4'
data_dir = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset/create_testing/Synthetic_data/depth_4/DATA_TEST_art_rescaled_intensity_level=3000_ambient_type=constant_SBR_background=0.4'
save_path = data_dir
list_hist =glob.glob(os.path.join(data_dir,'hist_after.mat'))
histogram = sio.loadmat(list_hist[0])['hist_after']

Nb_histogram = len(list_hist)

print('Nb_histogram '+ str(Nb_histogram))
Nx = histogram.shape[0] 
Ny = histogram.shape[1]
Nbins = histogram.shape[2]

ppp_tot_array = []
SBR_tot_array = []

number_background_zeros = 0
for idx in range(Nb_histogram):
    if idx != 31 and idx!=38:
        ppp_array = []
        SBR_array = []
        #hist_input_image   = glob.glob(os.path.join(data_dir,str(idx)+'_hist.mat'))
        #histogram = sio.loadmat(hist_input_image[0])['histogram']
        list_hist =glob.glob(os.path.join(data_dir,'hist_after.mat'))
        histogram = sio.loadmat(list_hist[0])['hist_after']
        histogram = np.squeeze(histogram)

        for i in range(Nx):
            for j in range(Ny):
                hist_one_pixel = np.squeeze(histogram[i,j,:])
                #print(type(hist_one_pixel))
                
                # SBR 
                b = np.median(hist_one_pixel)
                
                if b == 0 :
                    #print('Background is zero for idx='+ str(idx)+', i='+str(i)+', j='+str(j))
                    number_background_zeros = number_background_zeros + 1
                    #print(idx)

                else : 
                    # ppp
                    ppp = np.sum(hist_one_pixel)
                    #if ppp == 0:
                    #    print(idx)
                    ppp_array.append(ppp)
                    ppp_tot_array.append(ppp)

                    pos_max = np.argmax(hist_one_pixel)
                    range_center_of_mass = range(max(pos_max-1, 0), min(pos_max+2,Nbins))
                    
                    hist_one_pixel_no_noise = hist_one_pixel - b*np.ones(Nbins)
                    background = b * len(range_center_of_mass)
                    signal =np.sum(hist_one_pixel_no_noise[range_center_of_mass])

                    SBR = signal/background
                    #if SBR <5:
                    SBR_array.append(SBR)
                    SBR_tot_array.append(SBR)
    
        #print('number_background_zeros'+str(number_background_zeros))
    plt.scatter(ppp_array,SBR_array,marker='.')

    plt.xlabel("ppp")
    plt.ylabel("SBR")

    plt.title('Histogram '+ str(idx))

    
    plt.savefig(os.path.join(save_path, str(idx)+'_plot.png'))
    plt.clf()
    plt.cla()

sio.savemat(os.path.join(save_path,'ppp_tot_array.mat'), {'ppp_tot_array':ppp_tot_array})

plt.scatter(ppp_tot_array,SBR_tot_array,marker='.')
plt.title('Total array')
plt.xlabel("ppp")
plt.ylabel("SBR")
plt.savefig(os.path.join(save_path, 'total_plot.png'))
plt.show()
