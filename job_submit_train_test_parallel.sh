#!/bin/bash

## DON'T USE SPACES AFTER COMMAS

# You must specify a valid email address!
#SBATCH --mail-user=negin.ghamsarian@unibe.ch
# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END


# Job name
#SBATCH --job-name="UDA_Cat3kToCaDIS"
# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long

# Runtime and memory
#SBATCH --time=24:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=8
#SBATCH --nodes=3
#SBATCH --ntasks=3
##SBATCH --ntasks-per-node=1

# on gpu partition
#SBATCH --gres=gpu:rtx3090:3

#SBATCH --output=../Results_ENCORE_Cat3kToCaDIS/code_outputs/Supervised_%j.out_
#SBATCH --error=../Results_ENCORE_Cat3kToCaDIS/code_errors/Supervised_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
module load Python/3.9.5-GCCcore-10.3.0 #cuda/10.2.89 cuDNN/8.2.1.32-CUDA-11.3.1

#eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#conda activate /storage/homefs/ng22l920/anaconda3/envs/PyTorch_GPU

#source /storage/homefs/ng22l920/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate PyTorch_GPU
# conda list torch
# conda activate /storage/homefs/ng22l920/anaconda3/envs/PyTorch_GPU
# conda list torch
#eval "$(conda shell.bash hook)"
#source activate ~//anaconda3/etc/profile.d/conda.sh
#source /storage/homefs/ng22l920/anaconda3/etc/profile.d/conda.sh
#conda init powershell 
#conda activate ~//anaconda3/envs/PyTorch_GPU
##conda init --all --dry-run --verbose
#source activate /storage/homefs/ng22l920/anaconda3/envs/PyTorch_GPU
#python Test_VisualStudio.py 

# srun --ntasks 1 --nodes 1 --gres gpu:rtx3090:1 --output ../Results_ENCORE_Cat3kToCaDIS/code_outputs/Supervised_%j.out_ --error ../Results_ENCORE_Cat3kToCaDIS/code_errors/Supervised_%j.err /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 Supervised_importlib.py --config 'configs_Cataract.Config_Supervised' &
# srun --ntasks 1 --nodes 1 --gres gpu:rtx3090:1 --output ../Results_ENCORE_Cat3kToCaDIS/code_outputs/ENCORE_%j.out_ --error ../Results_ENCORE_Cat3kToCaDIS/code_errors/ENCORE_%j.err /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 ENCORE.py --config 'configs_Cataract.Config_ENCORE' 
srun --ntasks 1 --nodes 1 --gres gpu:rtx3090:1 --output ../Results_ENCORE_Cat3kToCaDIS/code_outputs/SelfTraining_%j.out_ --error ../Results_ENCORE_Cat3kToCaDIS/code_errors/SelfTraining_%j.err /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 Self_Training.py --config 'configs_Cataract.Config_SelfTraining' &
srun --ntasks 1 --nodes 1 --gres gpu:rtx3090:1 --output ../Results_ENCORE_Cat3kToCaDIS/code_outputs/MeanTeacher_%j.out_ --error ../Results_ENCORE_Cat3kToCaDIS/code_errors/MeanTeacher_%j.err /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 Mean_Teacher.py --config 'configs_Cataract.Config_Mean_Teacher' &
srun --ntasks 1 --nodes 1 --gres gpu:rtx3090:1 --output ../Results_ENCORE_Cat3kToCaDIS/code_outputs/TemporalEnsembling_%j.out_ --error ../Results_ENCORE_Cat3kToCaDIS/code_errors/TemporalEnsembling_%j.err /storage/homefs/ng22l920/anaconda3/envs/PyTorch_RTX3090/bin/python3 Temporal_Ensembling.py --config 'configs_Cataract.Config_TemporalEnsembling' 

