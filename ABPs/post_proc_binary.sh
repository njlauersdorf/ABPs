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

if [ $answer == "y" ]; then
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/klotsa/ABPs/post_proc"
    submit='sbatch'
else
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/klotsa/ABPs/post_proc"
    submit='sh'
fi

current=$( date "+%m_%d_%y" )
this_path="$( pwd )"

mkdir ${current}_txt_files
mkdir ${current}_pic_files
mkdir ${current}_vid_files

outpath="$this_path"/"$current"

echo "|----------------------------------------------------------------------|"
echo "|          -----------------Operating Systems-----------------         |"
echo "| mac: composition and area of each phase                              |"
echo "| windows: activity distribution of particles                          |"
echo "| linux: orientation distribution of particles                         |"
echo "|----------------------------------------------------------------------|"

echo "What is your operating system? (y/n)"
read os

echo "What is your bin size?"
read bin_size

echo "What is your step size?"
read time_step

echo "|----------------------------------------------------------------------|"
echo "|      Possible simulation options and corresponding user inputs       |"
echo "|          **************************************************          |"
echo "|          -----------------System Properties-----------------         |"
#echo "| phases: composition and area of each phase                           |"
#echo "| activity: activity distribution of particles                         |"
#echo "| orientation: orientation distribution of particles                   |"
echo "|          -----------------Kinetic Properties----------------         |"
#echo "| adsorption: adsorption to/desorption from cluster                    |"
echo "| adsorption-final: flux of particles between each phase               |"
#echo "| collision: collision frequency of gas particles                      |"
#echo "| fluctuations: fluctuations in cluster size                           |"
echo "|          ----------------Structural Properties--------------         |"
echo "| lattice-spacing: lattice spacing of cluster                          |"
echo "| compressibility: structural compressibility of particles in cluster  |"
echo "| structure-factor2: vorticity of particle motion                              |"
echo "| structure-factor: structural compressibility and S(k->0) of cluster  |"
echo "| structure-factor-freud: structural compressibility and S(k->0) of cluster using Freud module                        |"
#echo "| centrosymmetry: centrosymmetry parameter of particles                |"
#echo "| radial-df: radial distribution function of cluster                   |"
#echo "| angular-df: angular distribution function of cluster                 |"
#echo "| clustering-coefficient: clustering coefficient of particles          |"
#echo "| hexatic-order: hexatic order parameter of particles                  |"
#echo "| voronoi: voronoi tesselation of system                               |"
#echo "| translational-order: translational order of particles                |"
#echo "| steinhardt-order: Steinhardt order parameter of particles            |"
#echo "| neighbors: probability of being neighbors for each species in cluster|"
#echo "| domain-size: size of domains of like-species within cluster          |"
echo "|          ----------------Nematic Properties-----------------         |"
#echo "| nematic-order: nematic order of particles                            |"
#echo "| surface-align: alignment of particles toward cluster surface normal  |"
#echo "| interface-props: structural properties of interface                  |"
#echo "| com-align: alignment of particles toward cluster center of mass      |"
echo "|          ----------------Motion Properties------------------         |"
#echo "| cluster-msd: mean squared displacement of cluster center of mass     |"
#echo "| cluster-velocity: velocity of cluster center of mass                 |"
echo "| activity-com: tracked motion of cluster center of mass               |"
echo "| velocity-corr: local correlation in particle velocities              |"
#echo "| vorticity: vorticity of particle motion                              |"
#echo "| phase-velocity: velocity of particles in each phase                  |"
#echo "| penetration: ability for fast particles to penetrate slow membranes  |"
echo "|          ----------------Mechanical Properties-----------------      |"
echo "| int-press: interparticle pressure and stress of each phase           |"
echo "| bubble-interface-pressure: vorticity of particle motion                              |"
echo "| interparticle-pressure: vorticity of particle motion                              |"
echo "| interparticle-pressure-nlist: vorticity of particle motion                              |"
echo "| com-interface-pressure: interface pressure using center of mass alignment                             |"
echo "| surface-interface-pressure: interface pressure using surface normal alignment       |"
echo "| density: vorticity of particle motion                              |"
echo "| local-gas-density: local density within the gas phase only           |"
echo "| local-density: local density within the cluster only                 |"
echo "|----------------------------------------------------------------------|"

echo "What do you want to analyze?"
read method

echo "Do you want to generate plots? (y/n)"
read plot

if [ $method == "lattice_spacing" ]; then
  echo "Do you want to use parallel processing, namely for lattice spacing (y/n)?"
  read parallel
else
  parallel="n"
fi


for file in $(ls *gsd)
do
    if [ "$parallel" = "y" ]; then
        $submit $script_path/analyze_binary_parallel.sh $hoomd_path $outpath $script_path $file $bin_size $time_step $method $plot $os
    elif [ "$parallel" = "n" ]; then
        $submit $script_path/analyze_binary.sh $hoomd_path $outpath $script_path $file $bin_size $time_step $method $plot $os
    else
        echo "did not recognize response to parallel processing"
    fi

done
