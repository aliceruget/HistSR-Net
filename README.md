# Dataset 
## Training dataset : create_train.py

python3 create_train.py 

--config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' 

--out_dir='/home/ar432/Hist_SR_Net/Dataset/Train' 

--downsample_type='nearest' 

--data_type='MPI' 

--Dir_import='/home/ar432/DepthSR_Net/Dataset'

## Simulated Validation dataset (Middlebury image): 

create_test_synthetic.py

## Real Validation dataset (Hammer data) : 

create_test_hammer.py 


# Network 
## Training : 

python3 main_hist.py 

--data_path='/home/ar432/Hist_SR_Net/Dataset/Train/depth_4/DATA_TRAIN_HISTOGRAM_MPIMPI_nearest' 

--is_train='1' 

--config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' 

--checkpoint_dir='/home/ar432/Hist_SR_Net/Checkpoint/new_MPI'  

--result_path='/home/ar432/Hist_SR_Net/Results/new_MPI/' 

--save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'


## Testing :

python3 main_hist.py 

--data_path='/home/ar432/Dataset/Test/Hammer_data' 

--is_train='0' 

--config='/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml' 

--checkpoint_dir='/home/ar432/Hist_SR_Net/Checkpoint/new_MPI'  

--result_path='/home/ar432/Hist_SR_Net/Results/new_MPI/Test_Hammer' 

--save_parameters='1'  --loss_type='l2' --optimizer_type='Adam'
