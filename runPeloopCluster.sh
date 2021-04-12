#!/bin/sh

current=$( date "+%m_%d_%y" )
this_path=$( pwd )

hoomd_path='/Users/nicklauersdorf/hoomd-blue/build/'
#/nas/longleaf/home/njlauers/hoomd-blue/build/
#gsd_path='/proj/dklotsalab/users/ABPs/binary_soft/random_init/'
script_path='/Users/nicklauersdorf/hoomd-blue/build/run_specific/run_gpu.sh'
#nas/longleaf/home/njlauers/hoomd-blue/build/run_gpu_nuc.sh
tempOne='/Users/nicklauersdorf/hoomd-blue/build/run_specific/binary_soft_clusters.py'
##'/nas/longleaf/home/njlauers/hoomd-blue/build/run_specific/epsilonKBT.py'
#tempTwo='/nas/longleaf/home/kolbt/whingdingdilly/run_specific/soft_clusters.py'
sedtype='gsed'
submit='sh'

# Default values for simulations
part_num=$(( 1000 ))
runfor=$(( 1 ))
dump_freq=$(( 20000 ))
# Lists for activity of A and B species
pa=(0)
# 100 150 200 250 300 350 400 450 500)
pb=(50 100)
# 100 150 200 250 300 350 400 450 500)
# List for particle fraction
xa=(0)
# List for phi
phi=(60)
# List for epsilon
#eps=(1.0 0.1 0.001) # LISTS CAN CONTAIN FLOATS!!!!
eps=(1.0) 
#(0.1 1.0)

seed1=5
echo $seed1
seed2=5
echo $seed2
seed3=5
echo $seed3
seed4=5
echo $seed4
seed5=5
echo $seed5

mkdir ${current}_parent
cd ${current}_parent

# Set up an index counter for each list (instantiate to 0th index)
paCount=$(( 0 ))
pbCount=$(( 0 ))
xaCount=$(( 0 ))
phiCount=$(( 0 ))
epsCount=$(( 0 ))
# Instantiating the above counters is REDUNDANT! I did this to make it clear how lists work

# Set my phi index counter

phiCount=$(( 0 ))
# Loop through phi list values
for i in ${phi[@]}
do
    # Set the value of phi
    in_phi=${phi[${phiCount}]}

    # Reset my epsilon index counter
    epsCount=$(( 0 ))
    # Loop through eps list values
    for i in ${eps[@]}
    do
        # Set the value of epsilon
        in_eps=${eps[${epsCount}]}


        # Reset my fraction index counter
        xaCount=$(( 0 ))
        # Loop through particle fraction
        for i in ${xa[@]}
        do
            # Set the value of xa
            in_xa=${xa[${xaCount}]}
            
            # Reset b activity index counter
            pbCount=$(( 0 ))
            # Loop through b species activity
            for i in ${pb[@]}
            do
                # Set the value of pb
                in_pb=${pb[${pbCount}]}
            
                paCount=$(( 0 ))
                # Loop through particle fraction
                for i in ${pa[@]}
                do
                    # Set the value of pa
                    in_pa=${pa[${paCount}]}
                    
                    # If pa <= pb we want to run a simulation
                    if [ ${in_pa} -le ${in_pb} ]; then
                        # The below "echo" is great for checking if your loops are working!
                        echo "Simulation of PeA=${in_pa}, PeB=${in_pb}"
                        
                        # Submit our simulation
                        sim=pa${in_pa}_pb${in_pb}_xa${in_xa}_eps${in_eps}_phi${in_phi}_N${part_num}.py
                        #'s/\${replace_in_text_File}/'"${variable_to_replace_with}"'/g'
                        $sedtype -e 's@\${hoomd_path}@'"${hoomd_path}"'@g' $tempOne > $sim
                        $sedtype -i 's/\${part_num}/'"${part_num}"'/g' $sim 
                        # write particle number
                        $sedtype -i 's/\${phi}/'"${in_phi}"'/g' $sim                                    # write area fraction
                        $sedtype -i 's/\${runfor}/'"${runfor}"'/g' $sim                                 # write time in tau to infile
                        $sedtype -i 's/\${dump_freq}/'"${dump_freq}"'/g' $sim                           # write dump frequency to infile
                        $sedtype -i 's/\${part_frac_a}/'"${in_xa}"'/g' $sim
                        $sedtype -i 's/\${pe_a}/'"${in_pa}"'/g' $sim                                      # write a activity to infile
                        $sedtype -i 's/\${pe_b}/'"${in_pb}"'/g' $sim                                      # write b activity to infile
                        #$sedtype -i 's/\${alpha}/'"${a_count}"'/g' $sim                                     # write a fraction to infile
                        $sedtype -i 's@\${gsd_path}@'"${gsd_path}"'@g' $sim
                        $sedtype -i 's/\${ep}/'"${in_eps}"'/g' $sim                                    # write epsilon to infile
                        $sedtype -i 's/\${seed1}/'"${seed1}"'/g' $sim                                   # set your seeds
                        $sedtype -i 's/\${seed2}/'"${seed2}"'/g' $sim
                        $sedtype -i 's/\${seed3}/'"${seed3}"'/g' $sim
                        $sedtype -i 's/\${seed4}/'"${seed4}"'/g' $sim
                        $sedtype -i 's/\${seed5}/'"${seed5}"'/g' $sim

                        $submit $script_path $sim
                    fi
                
                    # Increment paCount
                    paCount=$(( $paCount + 1 ))
                
                # End pa loop
                done
            
                echo ""
            
                # Increment pbCount
                pbCount=$(( $pbCount + 1 ))
            
            # End pb loop
            done
            
            # Increment xaCount
            xaCount=$(( $xaCount + 1 ))
        
        # End xa loop
        done
    
        # Increment epsCount
        epsCount=$(( epsCount + 1 ))
    
    # End eps loop
    done
    
    # Increment phiCount
    phiCount=$(( $phiCount + 1 ))

# End phi loop
done
