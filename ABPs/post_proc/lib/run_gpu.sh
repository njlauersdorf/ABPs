#!/bin/bash                    
#SBATCH --partition=volta-gpu
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --ntasks=1                     # partition to run on
#SBATCH --cpus-per-task=8
#SBATCH --time=11-00:00                     # time (D-HH:MM)
source ~/.bashrc

# Don't run this for now SBATCH --exclude=g0605
module load gcc/9.1.0
module load cuda/11.4

#source ~/virtual_envs/rekt/bin/activate

conda activate rekt

filename=$1
python3 $filename --mode=gpu # I want one gpu
