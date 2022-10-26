#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --nodes=1                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)
#SBATCH --constraint=rhel8
#SBATCH --exclude=g0605
source ~/miniconda3/etc/profile.d/conda.sh

source ~/.bashrc

conda deactivate
# Don't run this for now SBATCH --exclude=g0605
module load gcc/9.1.0
module load cuda/11.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekt

filename=$1
python3 $filename --mode=gpu # I want one gpu
