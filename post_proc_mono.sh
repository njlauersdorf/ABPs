#!/bin/sh

echo "Are you running on Longleaf (y/n)?"
read answer

if [ $answer == "y" ]; then
    hoomd_path='/nas/longleaf/home/njlauers/hoomd-blue/build'
    gsd_path='/Volumes/External/04_15_20_parent_preset_cluster_lattice_spacing=activity_dependent'
    script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/post_proc'
    submit='sbatch'
    module add python/3.5.1
else
    hoomd_path='/Users/nicklauersdorf/hoomd-blue/build'
    gsd_path='/Volumes/External/04_01_20_parent/gsd/'
    script_path='/Users/nicklauersdorf/hoomd-blue/build/post_proc'
    submit='sh'
fi

for file in $(ls *gsd)
do
    $submit $script_path/analyze_alignment_map_accurate_int2.sh $hoomd_path $gsd_path $script_path $file
    
done