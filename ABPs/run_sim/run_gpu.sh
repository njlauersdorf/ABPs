#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --nodes=1                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)
#SBATCH --constraint=rhel8
#SBATCH --exclude=g0605
# Don't run this for now SBATCH --exclude=g0605

echo test
echo $SHELL

source ~/.bashrc

echo test2
echo $SHELL

module load gcc/9.1.0
module load cuda/11.4

echo test3
echo $SHELL

conda activate rekt

echo test4
echo $SHELL

filename=$1

python3 $filename --mode=gpu # I want one gpu
