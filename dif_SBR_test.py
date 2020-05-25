
import glob
import os
import subprocess
import numpy as np



data_dir = '/home/ar432/Dataset/Test/dif_SBR_bis'
folder_list = glob.glob(os.path.join(data_dir, 'SBR_*'))
for folder in folder_list :
    print(folder)
    data_path = folder
    result_path = folder
    subprocess.run(["python3", "main_hist.py", \
        "--data_path="+str(data_path), "--is_train=0",\
            "--config="+str('/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml'),\
                "--checkpoint_dir="+str('/home/ar432/Hist_SR_Net/Checkpoint/New_input0_noisy_i10_SBR0_004_MPI'),\
                    "--result_path="+str(result_path), "--save_parameters=1",\
                        "--loss_type="+str('l2'), "--optimizer_type="+str('Adam')])
                