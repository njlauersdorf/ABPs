#!/bin/bash
#SBATCH -p general                          # partition to run on
#SBATCH -n 1                                # number of cores
#SBATCH -t 1-00:00                          # time (D-HH:MM)
#SBATCH --mem=100g
# Command to increase memory allocated --mem=100g

#This is the path to hoomd
hoomd_path=$1
#This is the path to the analysis file
script_path=$2
# This is the given operating system
os=$3

if [ $os == "windows" ]; then
    source ~/.profile
    conda activate rekt
fi


if [ $hoomd_path == "/nas/longleaf/home/njlauers/hoomd-blue/build" ]; then
    source ~/.bashrc
    conda activate rekt
fi

if [ $os == "mac" ]; then
    python3 $script_path/lib/average_radial_heterogeneity.py
elif [ $os == "windows" ]; then
    python $script_path/lib/average_radial_heterogeneity.py
fi

exit 0
