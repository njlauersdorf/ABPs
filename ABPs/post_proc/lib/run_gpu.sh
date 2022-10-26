#!/bin/sh
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --partition=gpu
#SBATCH --nodes=1                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)
#SBATCH --exclude=g0605
# Don't run this for now SBATCH --exclude=g0605

filename=$1

python3.5 $filename --mode=gpu # I want one gpu
