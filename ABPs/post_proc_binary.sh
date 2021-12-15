#!/bin/sh

echo "Are you running on Longleaf (y/n)?"
read answer

if [ $answer == "y" ]; then
    hoomd_path='/nas/longleaf/home/njlauers/hoomd-blue/build'
    script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/post_proc'
    submit='sbatch'
else
    hoomd_path='/Users/nicklauersdorf/hoomd-blue/build'
    script_path='/Users/nicklauersdorf/hoomd-blue/build/njlauersdorf/ABPs/post_proc'
    submit='sh'
fi

current=$( date "+%m_%d_%y" )
this_path="$( pwd )"

mkdir ${current}_txt_files
mkdir ${current}_pic_files
mkdir ${current}_vid_files

txt_path="$this_path"/"$current"_txt_files/
pic_path="$this_path"/"$current"_pic_files/
vid_path="$this_path"/"$current"_vid_files/

echo "What is your bin size?"
read bin_size

echo "What is your step size?"
read time_step

echo "What do you want to analyze?"
read method
        
for file in $(ls *gsd)
do

    $submit $script_path/analyze_binary.sh $hoomd_path $txt_path $pic_path $vid_path $script_path $file $bin_size $time_step $method
    
done
