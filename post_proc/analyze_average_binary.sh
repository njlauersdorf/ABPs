#!/bin/bash
#SBATCH -p general                          # partition to run on
#SBATCH -n 1                                # number of cores
#SBATCH -t 11-00:00                          # time (D-HH:MM)
#SBATCH --mem=10g
# Command to increase memory allocated --mem=100g

#This is the path to hoomd
hoomd_path=$1
#This is the path to save the text files
outpath=$2
#This is the path to the analysis file
script_path=$3
#This is the file name to analyze
fname=$4
# This is the given operating system
os=$5

if [ $os == "windows" ]; then
    source ~/.profile
    conda activate rekt
fi

echo hoomd_path
echo $hoomd_path
echo outpath
echo $outpath
echo script_path
echo $script_path
echo fname
echo $fname

if [ $hoomd_path == '/Users/nicklauersdorf/hoomd-blue/build' ]; then
    vars="$(python3 ${script_path}/get_parameters.py ${fname})"
fi

if [ $hoomd_path == '/c/Users/Nick/hoomd-blue/build' ]; then
    vars="$(python ${script_path}/get_parameters.py ${fname})"
fi

if [ $hoomd_path == "/nas/longleaf/home/njlauers/hoomd-blue/build" ]; then
    vars="$(python3.8 ${script_path}/get_parameters.py ${fname})"
    source ~/.bashrc
    conda activate rekt
fi

echo vars
echo $vars

pass=()
for i in $vars
do
    # Put in array to unpack
    pass+=($i)
done

if [ $os == "mac" ]; then
    python3 $script_path/lib/average_radial_heterogeneity.py $fname $hoomd_path $outpath
elif [ $os == "windows" ]; then
    python $script_path/lib/average_radial_heterogeneity.py $fname $hoomd_path $outpath
fi

exit 0
