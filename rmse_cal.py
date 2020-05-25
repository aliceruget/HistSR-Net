import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

def rmse(im1,im2):
#   import pdb  
#   pdb.set_trace()
  #diff = np.square(im1.astype(np.float)-im2.astype(np.float))
  diff = np.square(im1-im2)
  diff_sum = np.mean(diff)
  rmse = np.sqrt(diff_sum)
  return rmse    

data_dir = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset/create_testing/Synthetic_data/depth_4/DATA_TEST_art_rescaled_intensity_level=10_ambient_type=constant_SBR_background=0.004'
depth_down = sio.loadmat(os.path.join(data_dir, '0_Df_down.mat'))
depth_down = np.squeeze(depth_down['I_down'])
depth_up = sio.loadmat(os.path.join(data_dir, '0_Df.mat'))
depth_up = np.squeeze(depth_up['I_up'])

data_recon = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/New_noisy_MPI_i10_SBR0_004/Middlebury'
depth_recon = sio.loadmat(os.path.join(data_recon, 'parameters.mat'))
depth_recon = np.squeeze(depth_recon['result'])

data_recon_depth = '/Users/aliceruget/Documents/PhD/DepthSR_Net_AR.bak/Results/depth_4/MPI_histogram_i10_SBR-004/Middlebury'
depth_recon_depth = sio.loadmat(os.path.join(data_recon_depth, '0recon.mat'))
depth_recon_depth = np.squeeze(depth_recon_depth['recon'])

init_rmse = rmse(depth_up, depth_down)   
final_rmse = rmse(depth_up, depth_recon)
other_rmse = rmse(depth_up,depth_recon_depth)
print(init_rmse)
print(final_rmse)
print(other_rmse)