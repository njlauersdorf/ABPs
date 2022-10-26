#!/bin/sh
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --partition=gpu
#SBATCH --nodes=1                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)
#SBATCH --exclude=g0605
# Don't run this for now SBATCH --exclude=g0605
#source ~/.bashrc

module load gcc/9.1.0
module load cuda/11.4

conda init bash
conda activate rekt

filename=$1

python3 $filename --mode=gpu # I want one gpu
