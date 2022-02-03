#!/bin/sh
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --partition=gpu                     # partition to run on
#SBATCH --time=11-00:00                     # time (D-HH:MM)

inFile=$1  #Python template file 
hoomdPath=$2  #Path to HOOMD (I hard-coded this into python file)
gsdPath=$3  #Path you want to save the file (I manipulate this manually in python file)
pa=$4 #Particle A activity
pb=$5 #Particle B activity
xa=$6 #Particle fraction of type A
ep=$7 #Softness of both particles
seed1=$8 #Random seed 1
seed2=$9 #Random seed 2
seed3=${10} #Random seed 3
seed4=${11} #Random seed 4
seed5=${12} #Random seed 5
myFrame=${13} #current frame

python $inFile $hoomdPath $gsdPath $pa $pb $xa $ep $phi $seed1 $seed2 $seed3 $seed4 $seed5 $myFrame
