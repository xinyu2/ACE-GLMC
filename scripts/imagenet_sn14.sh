#!/bin/bash
#SBATCH --partition=tao  ### Partition
#SBATCH --account=tao  ### Account
#SBATCH --job-name=cifarlt  ### Job Name
#SBATCH --time=16:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=4 ### Number of tasks (MPI processes)
#SBATCH --mem=300000 	### Memory(MB)

export PATH=/data/lab/tao/xinyu/software/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/data/lab/tao/xinyu/software/cuda-11.3/lib64:$LD_LIBRARY_PATH
module load python3/3.7.4
source $HOME/env4cv/bin/activate
cd $myenv/ACE-GLMC
# =======
#  Step1
# =======
time python main.py --dataset ImageNet-LT -a resnet32 --num_classes 1000 \
	--imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 \
	--momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 \
	--label_weighting 1.2 --contrast_weight 1 --lossfn ori --L1 0.0 --L2 0.0 --L3 0.0 --f0 0.0

# time python main.py --dataset cifar100 -a resnet32 --num_classes 100 \
# 	--imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 \
# 	--momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 \
# 	--label_weighting 1.2 --contrast_weight 1 --lossfn ace --L1 0.0 --L2 0.0 --L3 0.0 --f0 0.0
