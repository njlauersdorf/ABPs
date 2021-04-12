#!/bin/sh

current=$( date "+%m_%d_%y" )
this_path=$( pwd )

hoomd_path='/Users/nicklauersdorf/hoomd-blue/build/'
#/nas/longleaf/home/njlauers/hoomd-blue/build/
#gsd_path='/proj/dklotsalab/users/ABPs/binary_soft/random_init/'
script_path='/Users/nicklauersdorf/hoomd-blue/build/run_specific/run_gpu.sh'
#nas/longleaf/home/njlauers/hoomd-blue/build/run_gpu_nuc.sh
tempOne='/Users/nicklauersdorf/hoomd-blue/build/run_specific/epsilonKBT_wall.py'
##'/nas/longleaf/home/njlauers/hoomd-blue/build/run_specific/epsilonKBT.py'
#tempTwo='/nas/longleaf/home/kolbt/whingdingdilly/run_specific/soft_clusters.py'
sedtype='gsed'
submit='sh'


../SAMRAI-2.4.4/configure \
  --prefix=/nas/longleaf/home/njlauers/sfw/samrai/2.4.4/linux-g++-debug \
  --with-CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  --with-CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  --with-F77=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  --with-hdf5=/nas/longleaf/home/njlauers/sfw/linux/hdf5/1.10.6 \
  --without-petsc \
  --without-hypre \
  --with-silo=/nas/longleaf/home/njlauers/sfw/linux/silo/4.10 \
  --without-blaslapack \
  --without-cubes \
  --without-eleven \
  --without-kinsol \
  --without-petsc \
  --without-sundials \
  --without-x \
  --with-doxygen \
  --with-dot \
  --enable-debug \
  --disable-opt \
  --enable-implicit-template-instantiation \
  --disable-deprecated
  





export BOOST_ROOT=/nas/longleaf/home/njlauers/sfw/linux/boost/1.6.0
export PETSC_ARCH=linux-debug
export PETSC_DIR=/nas/longleaf/home/njlauers/sfw/petsc/3.12.2
../IBAMR/configure \
  CFLAGS="-g -O1 -Wall" \
  CXXFLAGS="-g -O1 -Wall" \
  FCFLAGS="-g -O1 -Wall" \
  CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  FC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  CPPFLAGS="-DOMPI_SKIP_MPICXX" \
  --with-hypre=$PETSC_DIR/$PETSC_ARCH \
  --with-samrai=/nas/longleaf/home/njlauers/sfw/samrai/2.4.4/linux-g++-debug \
  --with-hdf5=/nas/longleaf/home/njlauers/sfw/linux/hdf5/1.10.6 \
  --with-silo=/nas/longleaf/home/njlauers/sfw/linux/silo/4.10 \
  --with-boost=/nas/longleaf/home/njlauers/sfw/linux/boost/1.6.0 \
  --enable-libmesh \
  --with-libmesh=/nas/longleaf/home/njlauers/sfw/linux/libmesh/1.5.1/1.5.1-debug \
  --with-libmesh-method=dbg
  
  
  
  
  
  
  ../IBAMR/configure \
  CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  F77=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  FC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  MPICC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  MPICXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  CFLAGS="-O3 -pipe -Wall" \
  CXXFLAGS="-O3 -pipe -Wall" \
  FCFLAGS="-O3 -pipe -Wall" \
  CPPFLAGS="-DOMPI_SKIP_MPICXX" \
  --with-hypre=$PETSC_DIR/$PETSC_ARCH \
  --with-samrai=/nas/longleaf/home/njlauers/sfw/samrai/2.4.4/linux-g++-opt \
  --with-hdf5=/nas/longleaf/home/njlauers/sfw/linux/hdf5/1.10.6 \
  --with-silo=/nas/longleaf/home/njlauers/sfw/linux/silo/4.10 \
  --with-boost=/nas/longleaf/home/njlauers/sfw/linux/boost/1.6.0 \
  --enable-libmesh \
  --with-libmesh=/nas/longleaf/home/njlauers/sfw/linux/libmesh/1.5.1/1.5.1-opt \
  --with-libmesh-method=opt
  
  
  
  
  
  

../LIBMESH/configure \
    --prefix=/nas/longleaf/home/njlauers/sfw/linux/libmesh/1.5.1/1.5.1-opt \
    --with-methods=opt \
    PETSC_DIR=/nas/longleaf/home/njlauers/sfw/petsc/3.12.2 \
    PETSC_ARCH=linux-opt \
    CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
    CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
    FC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
    F77=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
    --enable-exodus \
    --enable-triangle \
    --enable-petsc-required \
    --disable-boost \
    --disable-eigen \
    --disable-hdf5 \
    --disable-openmp \
    --disable-perflog \
    --disable-pthreads \
    --with-thread-model=none \
    --disable-strict-lgpl \
    --disable-glibcxx-debugging


../LIBMESH/configure \
    --prefix=/nas/longleaf/home/njlauers/sfw/linux/libmesh/1.5.1/1.5.1-debug \
    --with-methods=dbg \
    PETSC_DIR=/nas/longleaf/home/njlauers/sfw/petsc/3.12.2 \
    PETSC_ARCH=linux-debug \
    CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
    CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
    FC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
    F77=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
    --enable-exodus \
    --enable-triangle \
    --enable-petsc-required \
    --disable-boost \
    --disable-eigen \
    --disable-hdf5 \
    --disable-openmp \
    --disable-perflog \
    --disable-pthreads \
    --with-thread-model=none \
    --disable-strict-lgpl \
    --disable-glibcxx-debugging
    


../SAMRAI-2.4.4/configure \
  CFLAGS="-O3" \
  CXXFLAGS="-O3" \
  FFLAGS="-O3" \
  --prefix=/nas/longleaf/home/njlauers/sfw/samrai/2.4.4/linux-g++-opt \
  --with-CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  --with-CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  --with-F77=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  --with-hdf5=/nas/longleaf/home/njlauers/sfw/linux/hdf5/1.10.6 \
  --without-hypre \
  --with-silo=/nas/longleaf/home/njlauers/sfw/linux/silo/4.10 \
  --without-blaslapack \
  --without-cubes \
  --without-eleven \
  --without-kinsol \
  --without-petsc \
  --without-sundials \
  --without-x \
  --with-doxygen \
  --with-dot \
  --disable-debug \
  --enable-opt \
  --enable-implicit-template-instantiation \
  --disable-deprecated
  
  
  
  
  
./configure \
  --CC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicc \
  --CXX=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpicxx \
  --FC=/nas/longleaf/home/njlauers/sfw/linux/openmpi/4.0.2/bin/mpif90 \
  --COPTFLAGS="-O3" \
  --CXXOPTFLAGS="-O3" \
  --FOPTFLAGS="-O3" \
  --PETSC_ARCH=$PETSC_ARCH \
  --with-debugging=0 \
  --download-hypre=1 \
  --with-x=0
  

./configure \

  --with-debugging=1 \
  --download-hypre=1 \
  --with-x=0
  
  
  
# Default values for simulations
part_num=$(( 300 ))
runfor=$(( 10 ))
dump_freq=$(( 20000 ))
# Lists for activity of A and B species

#PA MUST BE LESS THAN PB!
pa=(50)
# 100 150 200 250 300 350 400 450 500)
pb=(100)
# 100 150 200 250 300 350 400 450 500)
# List for particle fraction
xa=(50)
# List for phi
phi=(60)
# List for epsilon
#eps=(1.0 0.1 0.001) # LISTS CAN CONTAIN FLOATS!!!!
eps=(1.0) 
#(0.1 1.0)


seed1=$RANDOM
echo $seed1
seed2=$RANDOM
echo $seed2
seed3=$RANDOM
echo $seed3
seed4=$RANDOM
echo $seed4
seed5=$RANDOM
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
                        echo 6
                        echo $sim
                        $submit $script_path $sim
                        echo 7
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
