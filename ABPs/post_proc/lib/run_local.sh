#!/bin/sh
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --partition=gpu                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)

#Python template file
inFile=$1

python $inFile
