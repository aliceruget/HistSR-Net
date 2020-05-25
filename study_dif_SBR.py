import os 
import numpy as np 
import glob
import matplotlib.pyplot as plt
import scipy.io as sio

plt.clf()
plt.cla()
data_dir = '/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/New_noisy_MPI_i10_SBR0_004/Test_dif_SBR_below'
list_dir= sorted(glob.glob(os.path.join(data_dir,'SBR_*')))
init_rmse_tab = []
rmse_tab = []
for folder in list_dir:
    print(folder)
    data = sio.loadmat(os.path.join(folder,'parameters.mat'))
    init_rmse = data['init_rmse']
    init_rmse_tab.append(np.squeeze(init_rmse))
    rmse = data['rmse_value']
    rmse_tab.append(np.squeeze(rmse))

# N = len(rmse_tab)
# rmse_tab = rmse_tab[0:N-1]
# init_rmse_tab = init_rmse_tab[0:N-1]

#x = np.linspace(0.4,0.00004, num = 100)
# x = x[0:N-1]
# initial = plt.plot(x,init_rmse_tab, label='init rmse')
# final = plt.plot(x,rmse_tab,label='rmse')
# plt.legend()
# plt.show()
x = np.linspace(0.000004,0.004, num = 100)
initial = plt.plot(x,init_rmse_tab, label='init rmse')
final = plt.plot(x,rmse_tab,label='rmse')
plt.legend()
plt.show()