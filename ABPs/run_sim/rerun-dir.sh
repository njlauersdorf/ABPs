#!/bin/sh

script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/run_specific/run_gpu.sh'
submit='sbatch'

# Run from the directory where the slurm files are
for file in $(ls *.py)
do

    $submit $script_path $file

done
