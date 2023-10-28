#!/bin/bash
#SBATCH --partition=camas  ### Partition
#SBATCH --job-name=cifarlt  ### Job Name
#SBATCH --time=16:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=4 ### Number of tasks (MPI processes)
#SBATCH --mem=300000 	### Memory(MB)

module load python3/3.11.4
cd $MYSCRATCH
source $MYSCRATCH/env4t22/bin/activate
cd $MYSCRATCH/ACE-GLMC
# =======
#  Step1
# =======
time python main.py --dataset cifar10 -a resnet32 --num_classes 10 \
	--imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 \
	--momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 \
	--label_weighting 1.2 --contrast_weight 1 --L1 0.0 --L2 0.0 --L3 0.0 --f0 0.0

# Lm=( 0.0 )
# F0=( 0.0 )
# for lam in "${Lm[@]}" ; 
# do
# 	for f0 in "${F0[@]}" ; 
# 	do
		
# 	done	
# done
