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

echo "What is your bin size? (recommended = 5)"
read bin_size

if [ -z "$bin_size" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

if ! [[ "$bin_size" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]]; then 
    echo "Inputs must be a numbers" 
    exit 0 
fi

echo "What is your step size? (recommended = 1)"
read time_step

if [ -z "$time_step" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

if ! [[ "$time_step" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]]; then 
    echo "Inputs must be a numbers" 
    exit 0 
fi

echo "|----------------------------------------------------------------------|"
echo "|      Possible simulation options and corresponding user inputs       |"
echo "|          **************************************************          |"
echo "|          -----------------System Properties-----------------         |"
echo "| phases: composition and area of each phase                           |"
echo "| activity: activity distribution of particles                         |"
echo "| orientation: orientation distribution of particles                   |"
echo "|          -----------------Kinetic Properties----------------         |"
echo "| adsorption: adsorption to/desorption from cluster                    |"
echo "| kinetic-motion: cluster displacement driven by particle flux         |"
echo "| collision: collision frequency of gas particles                      |"
echo "| fluctuations: fluctuations in cluster size                           |"
echo "|          ----------------Structural Properties--------------         |"
echo "| lattice-spacing: lattice spacing of cluster                          |"
echo "| compressibility: structural compressibility of particles in cluster  |"
echo "| structure-factor-sep: structure factor given lattice spacings        |"
echo "| structure-factor-rdf: structure factor given g(r)                    |"
echo "| structure-factor-freud: structure factor using Freud algorithm       |"
echo "| centrosymmetry: centrosymmetry parameter of particles                |"
echo "| radial-df: radial distribution function of cluster                   |"
echo "| angular-df: angular distribution function of cluster                 |"
echo "| hexatic-order: hexatic order parameter of particles                  |"
echo "| voronoi: voronoi tesselation of system                               |"
echo "| translational-order: translational order of particles                |"
echo "| steinhardt-order: Steinhardt order parameter of particles            |"
echo "| density: density of the bulk                                         |"
echo "| local-gas-density: local density within the gas phase only           |"
echo "| local-density: local density within the cluster only                 |"
echo "|        ----------------Segregation Properties-----------------       |"
echo "| neighbors: probability of being neighbors for each species in cluster|"
echo "| domain-size: size of domains of like-species within cluster          |"
echo "| clustering-coefficient: clustering coefficient of particles          |"
echo "|          ----------------Nematic Properties-----------------         |"
echo "| nematic-order: nematic order of particles                            |"
echo "| surface-align: alignment of particles toward cluster surface normal  |"
echo "| interface-props: structural properties of interface                  |"
echo "| com-align: alignment of particles toward cluster center of mass      |"
echo "|          ----------------Motion Properties------------------         |"
echo "| cluster-msd: mean squared displacement of cluster center of mass     |"
echo "| cluster-velocity: velocity of cluster center of mass                 |"
echo "| activity-com: tracked motion of cluster center of mass               |"
echo "| velocity-corr: local correlation in particle velocities              |"
echo "| vorticity: vorticity of particle motion                              |"
echo "| phase-velocity: velocity of particles in each phase                  |"
echo "| penetration: ability for fast particles to penetrate slow membranes  |"
echo "|          ----------------Mechanical Properties-----------------      |"
echo "| int-press: interparticle pressure and stress of each phase           |"
echo "| int-press-dep: interparticle pressure of entire system (deprecated)  |"
echo "| int-press-nlist: interparticle pressure of entire system             |"
echo "| com-body-forces: interface pressure using center of mass             |"
echo "| surface-body-forces: interface pressure using surface normal         |"
echo "| bubble-body-forces: separate interface pressures using surface norm  |"
echo "|----------------------------------------------------------------------|"

echo "What do you want to analyze?"
read method

if [ -z "$method" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

echo "What is your starting frame for analysis? (default = start of simulation)"
read start_step

if [ -z "$start_step" ]; then 
    echo 'Set to default value'  
fi 

echo "What is your ending frame for analysis? (default = end of simulation)"
read end_step

if [ -z "$end_step" ]; then 
    echo 'Set to default value' 
fi 

echo "Do you want to generate plots? (y/n)"
read plot

if [ -z "$plot" ]; then 
    echo 'Inputs cannot be blank please try again!' 
    exit 0 
fi 

if [ $plot != "y" ] &&  [ $plot != "n" ]; then
    echo 'Input must correspond to given options!' 
    exit 0 
fi


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
        $submit $script_path/analyze_binary.sh $hoomd_path $outpath $script_path $file $bin_size $time_step $method $plot $os $start_step $end_step
    else
        echo "did not recognize response to parallel processing"
    fi

done
