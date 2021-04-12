#!/bin/sh

current=$( date "+%m_%d_%y" )
this_path=$( pwd )

hoomd_path='/nas/longleaf/home/njlauers/hoomd-blue'
gsd_path='/nas/longleaf/home/njlauers/hoomd-blue/build/gsd'
script_path='/nas/longleaf/home/njlauers/hoomd-blue/build/run_gpu_nuc.sh'
#    template='/nas/longleaf/home/kolbt/whingdingdilly/run_specific/varyWellDepth.py'
template='/nas/longleaf/home/njlauers/hoomd-blue/build/run_specific/epsilonKBT.py'
sedtype='sed'
submit='sbatch'

# Default values for simulations
phi=$(( 60 ))
runfor=$(( 150 ))
dump_freq=$(( 20000 ))
pe_a=$(( 0 ))
pe_b=$(( 50 ))
xa=$(( 50 ))
partNum=$(( 10000 ))

# These are the parameters for my loop
a_count=$(( 1 ))
a_spacer=$(( 1 ))
a_max=$(( 1 ))


echo "Time to set some seeds!"
echo "Positional seed"
seed1=$(( 1))
echo "Equilibration seed"
seed2=$(( 1))
echo "Orientational seed"
seed3=$(( 1))
echo "A activity seed"
seed4=$(( 1))
echo "B activity seed"
seed5=$(( 1))


mkdir ${current}_parent
cd ${current}_parent

# this segment of code writes the infiles
while [ $a_count -le $a_max ]       # loop through N
do

    infile=pa${pe_a}_pb${pe_b}_xa${xa}_alpha${a_count}.py                       # set unique infile name
    $sedtype -e 's@\${hoomd_path}@'"${hoomd_path}"'@g' $template > $infile  # write path to infile (delimit with @)
    $sedtype -i 's/\${part_num}/'"${partNum}"'/g' $infile                   # write particle number
    $sedtype -i 's/\${phi}/'"${phi}"'/g' $infile                            # write particle number
    $sedtype -i 's/\${runfor}/'"${runfor}"'/g' $infile                      # write time in tau to infile
    $sedtype -i 's/\${dump_freq}/'"${dump_freq}"'/g' $infile                # write dump frequency to infile
    $sedtype -i 's/\${part_frac_a}/'"${xa}"'/g' $infile                     # write particle fraction to infile
    $sedtype -i 's/\${pe_a}/'"${pe_a}"'/g' $infile                          # write activity of A to infile
    $sedtype -i 's/\${pe_b}/'"${pe_b}"'/g' $infile                          # write activity of B to infile
    $sedtype -i 's/\${alpha}/'"${a_count}"'/g' $infile                      # write activity of B to infile
    $sedtype -i 's@\${gsd_path}@'"${gsd_path}"'@g' $infile                  # set gsd path variable
    $sedtype -i 's/\${seed1}/'"${seed1}"'/g' $infile                        # set your seeds
    $sedtype -i 's/\${seed2}/'"${seed2}"'/g' $infile
    $sedtype -i 's/\${seed3}/'"${seed3}"'/g' $infile
    $sedtype -i 's/\${seed4}/'"${seed4}"'/g' $infile
    $sedtype -i 's/\${seed5}/'"${seed5}"'/g' $infile

    $submit $script_path $infile

    a_count=$(( $a_count + $a_spacer ))

done