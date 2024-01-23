#!/bin/sh

echo "|----------------------------------------------------------------------|"
echo "|          Nicholas Lauersdorf's HOOMD-Blue simulation module          |"
echo "|          **************************************************          |"
echo "|                                                                      |"
echo "| This is module intended for running simulations of Brownian and      |"
echo "| active Brownian particles subject to various initial and simulation  |"
echo "| conditions that the user prescribes through input in the command     |"
echo "| prompt and hard-coding via bash scripts. Of note, this module is     |"
echo "| intended for its primary use of binary active mixtures of two        |"
echo "| species with varying properties, namely different activities.        |"
echo "|                                                                      |"
echo "| Any problems or questions should be sent to njlauersdorf@gmail.com.  |"
echo "|----------------------------------------------------------------------|"

echo "Are you running on Longleaf (y/n)?"
read answer

if [ -z "$answer" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

if [ $answer == "y" ]; then
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/ABPs/post_proc"
    submit='sbatch'
else
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/ABPs/post_proc"
    submit='sh'
fi

current=$( date "+%m_%d_%y" )
this_path="$( pwd )"

outpath="$this_path"/

echo "|----------------------------------------------------------------------|"
echo "|          -----------------Operating Systems-----------------         |"
echo "| mac: macOS X operating system                                        |"
echo "| windows: windows operating system                                    |"
echo "| linux: linux operating system                                        |"
echo "|----------------------------------------------------------------------|"

echo "What is your operating system?"
read os

if [ -z "$os" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

if [ $os != "mac" ] &&  [ $os != "windows" ] && [ $os != "linux" ]; then
    echo 'Input must correspond to given options!' 
    exit 0 
fi

for file in $(ls *txt)
do
    $submit $script_path/analyze_average_binary.sh $hoomd_path $outpath $script_path $file $os
done
