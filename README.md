# Dataset 
## Training dataset 

python3 create_train.py --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --out_dir='/home/ar432/Hist_SR_Net/Dataset/Train' --downsample_type='nearest' --data_type='MPI' --Dir_import='/home/ar432/DepthSR_Net/Dataset'

python3 create_train.py --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR.bak/Configs/cfg_original_scale4.yaml' --out_dir='/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset' --downsample_type='nearest' --data_type='MPI' --Dir_import='/Users/aliceruget/Documents/PhD/Dataset/RAW'

python3 main_creation_dataset.py --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --out_dir='/home/ar432/Hist_SR_Net/Dataset/Train' --data_type='total' --Dir_import='/home/ar432/DepthSR_Net/Dataset'

python3 main_creation_dataset.py --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR.bak/Configs/cfg_original_scale4.yaml' --out_dir='/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Dataset' --data_type='total' --Dir_import='/Users/aliceruget/Documents/PhD/Dataset/RAW'

python3 create_train_depth.py --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --out_dir='/home/ar432/Hist_SR_Net/Dataset/Train' --downsample_type='nearest' --data_type='MPI' --Dir_import='/home/ar432/DepthSR_Net/Dataset'

## Simulated Validation dataset (Middlebury image)

create_test_synthetic.py

## Real Validation dataset (Hammer data) 

create_test_hammer.py 


# Network 
## Training 

python3 main_hist.py --data_path='/home/ar432/Two-Depth-SR-Net/Dataset/Train/depth_4/DATA_TRAIN_HISTOGRAM_noisy_intensity3000_SBR_0_4_MPI' --is_train='1' --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --checkpoint_dir='/home/ar432/Hist_SR_Net/Checkpoint/New_input0_noisy_i3000_SBR0_4_MPI'  --result_path='/home/ar432/Hist_SR_Net/Results/New_input0_noisy_i1000_SBR0_4_MPI' --save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'

python3 main.py --data_path='/home/ar432/Two-Depth-SR-Net/Dataset/Train/depth_4/DATA_TRAIN_HISTOGRAM_noisy_intensity3000_SBR_0_4_MPI'  --is_train='1' --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --checkpoint_dir='/home/ar432/DepthSR_Net/Checkpoint/depth_4/MPI/histogram_i30000_SBR_0_4' --result_path='/home/ar432/DepthSR_Net/Results/depth_4/data_MPI/histogram_i3000_SBR0_4' --save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'

## Testing 

python3 main_hist.py --data_path='/home/ar432/Dataset/Test/Middlebury_i10_SBR_0_004' --is_train='0' --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --checkpoint_dir='/home/ar432/Hist_SR_Net/Checkpoint/New_input0_noisy_i10_SBR0_004_MPI' --result_path='/home/ar432/Hist_SR_Net/Results/New_input0_noisy_i10_SBR0_004_MPI/Middlebury' --save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'


python3 main_hist.py --data_path='/home/ar432/Dataset/Test/Middlebury_i10_SBR_0_004' --is_train='0' --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --checkpoint_dir='/home/ar432/Hist_SR_Net/Checkpoint/noisy_i10_SBR0_004_MPI'  --result_path='/home/ar432/Hist_SR_Net/Results/noisy_i10_SBR0_004_MPI/Middlebury_zeros_features' --save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'

python3 main.py --data_path='/home/ar432/Dataset/Test/Middlebury_i10_SBR_0_004' --is_train='0' --config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' --checkpoint_dir='/home/ar432/DepthSR_Net/Checkpoint/depth_4/MPI/histogram_i10_SBR0_004' --result_path='/home/ar432/DepthSR_Net/Results/depth_4/data_MPI/histogram_i10_SBR0_004/Middlebury_zeros_features' --save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'


