#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:01:50 2020

@author: nicklauersdorf
"""

#!/bin/sh

current=$( date "+%m_%d_%y" )
this_path=$( pwd )

echo "Are you running on Longleaf (y/n)?"
read answer

if [ $answer == "y" ]; then
    hoomd_path='/nas/longleaf/home/njlauers/hoomd-blue/build'
    gsd_path='/nas/longleaf/home/njlauers/hoomd-blue/build/gsd'
    script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/run_gpu_nuc.sh'
#    template='/nas/longleaf/home/kolbt/whingdingdilly/run_specific/varyWellDepth.py'
    template='/nas/longleaf/home/njlauers/hoomd-blue/build/run_specific/epsilonKBT.py'
    sedtype='sed'
    submit='sbatch'
else
    hoomd_path='/Users/nicklauersdorf/hoomd-blue/build'
    gsd_path='/Users/nicklauersdorf/hoomd-blue/build/gsd'
    script_path='/Users/nicklauersdorf/hoomd-blue/build/run_gpu_nuc_BB.sh'
#    template='/Users/kolbt/Desktop/compiled/whingdingdilly/run_specific/varyWellDepth.py'
    template='/Users/nicklauersdorf/hoomd-blue/build/run_specific/BinaryBubbleSlowOut.py'
    sedtype='gsed'
    submit='sh'
fi

echo "GPU (y/n)?"
read gpu

if [ $gpu == "y" ]; then
    hoomd_path='/nas/longleaf/home/njlauers/hoomd-blue'
    script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/run_gpu_nuc.sh'
fi

# Default values for simulations
phiG=$(( 10 ))
runfor=$(( 150 ))
dump_freq=$(( 20000 ))
pef=$(( 500 ))
pes=$(( 150))
xf=$(( 50 ))
partNum=$(( 50 ))

# These are the parameters for my loop
a_count=$(( 1 ))
a_spacer=$(( 1 ))
a_max=$(( 1 ))

mkdir ${current}_parent
cd ${current}_parent

# this segment of code writes the infiles
while [ $a_count -le $a_max ]       # loop through N
do

    infile=pa${pe_a}_pb${pe_b}_xa${xa}_alpha${a_count}.py                       # set unique infile name
    $sedtype -e 's@\${hoomd_path}@'"${hoomd_path}"'@g' $template > $infile  # write path to infile (delimit with @)
    $sedtype -i 's/\${part_num}/'"${partNum}"'/g' $infile                   # write particle number
    $sedtype -i 's/\${phiG}/'"${phiG}"'/g' $infile                            # write particle number
    $sedtype -i 's/\${runfor}/'"${runfor}"'/g' $infile                      # write time in tau to infile
    $sedtype -i 's/\${dump_freq}/'"${dump_freq}"'/g' $infile                # write dump frequency to infile
    $sedtype -i 's/\${xf}/'"${xf}"'/g' $infile                     # write particle fraction to infile
    $sedtype -i 's/\${pef}/'"${pef}"'/g' $infile                          # write activity of A to infile
    $sedtype -i 's/\${pes}/'"${pes}"'/g' $infile                          # write activity of B to infile
    $sedtype -i 's/\${alpha}/'"${a_count}"'/g' $infile                      # write activity of B to infile
    $sedtype -i 's@\${gsd_path}@'"${gsd_path}"'@g' $infile                  # set gsd path variable

    $submit $script_path $infile

    a_count=$(( $a_count + $a_spacer ))

done