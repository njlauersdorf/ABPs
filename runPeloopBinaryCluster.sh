#!/bin/bash

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    LIST_WHITESPACES=`echo $LIST | tr "$DELIMITER" " "`
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
            return 0
        fi
    done
    return 1
}

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

current=$( date "+%m_%d_%y" )
this_path=$( pwd )


hoomd_path="$HOME/hoomd-blue/build/"

echo "Are you running on Longleaf (y/n)?"
read answer

if [ $answer == "y" ]; then
    sedtype='sed'
    submit='sbatch'
    #script_path="$HOME/hoomd-blue/build/run_test/run_gpu.sh"
    script_path="$HOME/klotsa/ABPs/post_proc/lib/run_gpu.sh"
    tempOne="$HOME/klotsa/ABPs/post_proc/lib/run_sim_sample.py"
    #tempOne="$HOME/hoomd-blue/build/run_test/run_sim_sample.py"
else
    sedtype='gsed'
    submit='sh'
    script_path="$HOME/klotsa/ABPs/post_proc/lib/run_local.sh"
    tempOne="$HOME/klotsa/ABPs/post_proc/lib/run_sim_sample.py"
fi

echo "|----------------------------------------------------------------------|"
echo "|      Possible simulation options and corresponding user inputs       |"
echo "|          **************************************************          |"
echo "|          -------------Random initial conditions-------------         |"
echo "| Random gas: random_init                                              |"
echo "|          ----------Near steady-state MIPS clusters----------         |"
echo "| Homogeneous cluster: homogeneous_cluster                             |"
echo "| 100% slow bulk, 100% fast interface cluster: slow_bulk_cluster       |"
echo "| 100% fast bulk, 100% slow interface cluster: fast_bulk_cluster       |"
echo "| Half-slow, half-fast cluster: half_cluster                           |"
echo "|             ----------Bulk only of MIPS cluster----------            |"
echo "| Constant pressure (changing volume): constant_pressure          |"
echo "|          -----------Elongated planar membranes--------------         |"
echo "| Slow planar membrane: slow_membrane                                  |"
echo "| Immobile planar membrane: immobile_membrane                          |"
echo "| Immobile oriented planar membrane: immobile_orient_membrane          |"
echo "| Slow constrained planar membrane: slow_constrained_membrane          |"
echo "| Slow adsorb constrained membrane: slow_adsorb_constrained_membrane   |"
echo "| Slow interior constrained membrane: slow_int_constrained_membrane    |"
echo "| Diffusive mixture membrane: slow_int_constrained_membrane_dif_temp   |"
echo "|----------------------------------------------------------------------|"


echo "What initial conditions do you want (see above for options)?"
read answer

init_cond=$answer

list_of_sims="random_init chemical_equilibrium homogeneous_cluster slow_bulk_cluster fast_bulk_cluster half_cluster constant_pressure slow_membrane immobile_membrane immobile_orient_membrane slow_constrained_membrane slow_adsorb_constrained_membrane slow_int_constrained_membrane slow_int_constrained_membrane_dif_temp"

if exists_in_list "$list_of_sims" " " $init_cond; then
    dont_run='no'
else
    dont_run='yes'
fi

list_of_elongated_sims="random_init slow_membrane immobile_membrane immobile_orient_membrane slow_constrained_membrane slow_adsorb_constrained_membrane slow_int_constrained_membrane slow_int_constrained_membrane_dif_temp"

if exists_in_list "$list_of_elongated_sims" " " $init_cond; then

    echo "What simulation box aspect ratio, i.e. length:width (#:#)?"
    read answer

    aspect_ratio=$answer
else
    aspect_ratio='1:1'
fi


if [ $dont_run == "no" ]; then
    # Default values for simulations
    declare -i part_num
    part_num=( 20000 )

    # Length of simulation in Brownian time steps
    declare -i runfor
    runfor=$(( 3 ))

    # Frequency for dumping simulation data
    declare -a dump_freq
    dump_freq=( 0.3 )
    #( 0.0025 )

    # Lists for activity of A and B species
    declare -a pa
    pa=(0)
    # 25 50 75)

    declare -a pb
    pb=(500)

    # List for particle fraction
    declare -a xa
    xa=(50)

    # List for phi
    declare -a phi
    phi=(60)
   
    # List for epsilon
    declare -a eps
    eps=(1.0)

    declare -a kT
    kT=(1.0)

    # Random seeds for initializing simulation
    seed1=$$
    echo $seed1
    seed2=$$
    echo $seed2
    seed3=$$
    echo $seed3
    seed4=$$
    echo $seed4
    seed5=$$
    echo $seed5

    # Parent directory to contain simulations
    mkdir ${current}_parent
    cd ${current}_parent

    # Set my phi index counter
    partCount=$(( 0 ))
    # Loop through part_num list values
    for i in ${part_num[@]}
    do

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

                                # Declare simulation file name
                                sim=${init_cond}_pa${in_pa}_pb${in_pb}_xa${in_xa}_eps${in_eps}_phi${in_phi}_N${part_num}.py
                                #'s/\${replace_in_text_File}/'"${variable_to_replace_with}"'/g'
                                $sedtype -e 's@\${hoomd_path}@'"${hoomd_path}"'@g' $tempOne > $sim              # write path to HOOMD-blue to infile
                                $sedtype -i 's@\${init_cond}@'"${init_cond}"'@g' $sim                           # write initial condition to infile
                                $sedtype -i 's/\${part_num}/'"${part_num}"'/g' $sim                             # write particle number
                                $sedtype -i 's/\${phi}/'"${in_phi}"'/g' $sim                                    # write area fraction
                                $sedtype -i 's/\${runfor}/'"${runfor}"'/g' $sim                                 # write time in tau to infile
                                $sedtype -i 's/\${dump_freq}/'"${dump_freq}"'/g' $sim                           # write dump frequency to infile
                                $sedtype -i 's/\${part_frac_a}/'"${in_xa}"'/g' $sim                             # write slow particle fraction to infile
                                $sedtype -i 's/\${pe_a}/'"${in_pa}"'/g' $sim                                    # write slow activity to infile
                                $sedtype -i 's/\${pe_b}/'"${in_pb}"'/g' $sim                                    # write fast activity to infile
                                $sedtype -i 's/\${ep}/'"${in_eps}"'/g' $sim                                     # write particle softness to infile
                                $sedtype -i 's/\${aspect_ratio}/'"${aspect_ratio}"'/g' $sim                     # write simulation box aspect ratio to infile
                                $sedtype -i 's/\${seed1}/'"${seed1}"'/g' $sim                                   # set your seeds
                                $sedtype -i 's/\${seed2}/'"${seed2}"'/g' $sim
                                $sedtype -i 's/\${seed3}/'"${seed3}"'/g' $sim
                                $sedtype -i 's/\${seed4}/'"${seed4}"'/g' $sim
                                $sedtype -i 's/\${seed5}/'"${seed5}"'/g' $sim
                                $submit $script_path $sim                                                       # submit corresponding run bash file for each simulation given parameters
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

        # Increment partCount
        partCount=$(( $partCount + 1 ))
    # End partNum loop
    done
else
    echo 'Did not initiate run!'
fi
