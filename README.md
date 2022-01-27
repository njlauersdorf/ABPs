# klotsa
***

## Table of Contents
1. [Introduction](#introduction)
2. [ABPs](#abps)
   - [General Info](#general-info)
   - [Technologies](#technologies)
   - [Installation](#installation)
     - [HOOMD-Blue](#HOOMD-Blue)
     - [Klotsa Github Repository](#Klotsa-Github-Repository)
   - [Running Code](#running-code)
     - [Submitting Simulations](#Submitting-Simulations)
     - [Submitting Post-Processing](#Submitting-Post-Processing)
   - [Collaboration](#collaboration)
3. [Personal](#personal)

## Introduction
This github profile consists of 2 parts currently: a collection of software and packages for simulating systems of active and passive Brownian particles in HOOMD-Blue using molecular dynamics, located in the /klotsa/ABPs folder, and a collection of various projects to develop and demonstrate my skills in various languages and applications (both front-end and back-end software development), located in /klotsa/personal.

# ABPs
***

## General Info
This github project consists of a collection of software and packages for simulating systems of active and passive Brownian particles in HOOMD-Blue using molecular dynamics, located in the /klotsa/ABPs folder. Simulation submission bash files read in the user's desired system (particle activities, particle softness, system density, particle fraction, particle size, and population size) and initial conditions (box dimensions, box shape, initial positions, whether it's a randomized gas phase or an instantiated liquid-gas phase separated system, and initial orientation, whether it's randomized or biased with local alignment, i.e. the liquid-gas interface) creates an initial hoomd snapshot/frame using these specified conditions, and runs and saves a simulation for the desired time at the desired stepsize. In addition, this software enables the post-processing of the outputted system over time with focus placed on steady-state measurements and characterization (see Lauersdorf, et al. (2021) for derivation of the theory these measurements led to). 

## Technologies

A list of technologies used within the project:
* HOOMD-Blue (https://hoomd-blue.readthedocs.io/en/stable/installation.html): Version 2.9.7
  * Python (https://www.python.org/downloads/release/python-380/): Version 3.8.1
  * Numpy (https://numpy.org/install/): Version 1.18.1
  * CMake (https://cmake.org/): Version 3.16.4
  * Clang (https://clang.llvm.org/): Version 9.0.1
  * OpenMPI (https://www.open-mpi.org/): Version 4.0.2 
* Python (https://www.python.org/downloads/release/python-380/): Version 3.8.1
  * Numpy (https://numpy.org/install/): Version 1.18.1
  * Matplotlib (https://matplotlib.org/): Version 3.1.3
  * Shapely (https://pypi.org/project/Shapely/): Version 1.7.0
  * SciPy (https://scipy.org/): Version 1.3.1
* Jupyter Notebook (https://jupyter-notebook.readthedocs.io/en/stable/changelog.html): Version 6.1.1
* FFMPEG (https://www.ffmpeg.org/): Version 3.4

## Installation

It is highly recommended one install's both HOOMD-Blue and this github repository in their user's home directory due to the pathing in the analysis files. 

### HOOMD-Blue

**For installation on local desktop:**

Install prerequisites.
```
$ cd ~
$ bash
$ source active [Virtual Environment]
$ conda install -c conda-forge sphinx git openmpi numpy cmake clang
```
Download HOOMD-Blue version 2.9.7
```
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```
or
```
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
```
Configure HOOMD-Blue. When configuring locally, be sure `-DENABLE_CUDA=OFF` in the `cmake` tags. When configuring locally, you installed Open MPI, let `-DENABLE_MPI=ON` in the `cmake` tags.
```
$ cd hoomd-blue
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=OFF -DENABLE_MPI=ON
```
Compile:
```
$ make -j4
```
Test your build. Since we built locally and do not have CUDA support, many tests will fail due to the requirement of a GPU.
```
$ ctest
```

**For installation on computing cluster (i.e. UNC's Longleaf):**

Install and save prerequisites to future logins on cluster.
```
$ module avail
$ module add git/2.33.0 cmake/3.18.0 clang/6.0 ffmpeg/3.4 geos/3.8.1 cuda/10.1 gcc/6.3.0 python/3.5.1
$ module save
```
Download HOOMD-Blue version 2.9.7
```
$ cd ~
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```
or
```
$ cd ~
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
```
Create and activate virtual environment for compiling HOOMD-Blue
```
$ python3 -m venv /path/to/new/virtual/environment --system-site-packages
$ source /path/to/new/virtual/environment/bin/activate
$ source activate <virtual environment>
```
Configure HOOMD-Blue. When configuring on cluster, be sure `-DENABLE_CUDA=ON` in the `cmake` tags as you will be using GPUs. When configuring locally, let `-DENABLE_MPI=ON` in the `cmake` tags.
```
$ cd hoomd-blue
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON
```
Compile:
```
$ make -j4
```
Test your build. Since we built locally and do not have CUDA support, many tests will fail due to the requirement of a GPU.
```
$ ctest
```

### Klotsa github repository
```
$ cd ~
$ git clone https://github.com/njlauersdorf/klotsa.git
```

## Running Code
This github project utilizes bash scripts to read in user's desired measurement/simulation type, select the desired python file to run (either for a simulation or post-processing) based on user input, and to read in the specified initial/system conditions into a template python file for a) post-processing of each .gsd file (simulation file) within the current directory or b) create a python file for instantiating a system and running that file to simulate each possible configuration of initial conditions inputted, which, in turn, outputs a .gsd file detailing the simulation at each time step.  


### Submitting simulations
The bash file used to submit a simulation is /klotsa/ABPs/runPeloopBinaryCluster.sh. Before submitting the bash file, one should manually input all desired possible physical conditions of each system to run as lists in each variable's location at the top of the page. The bash file will loop through all possible pairings of these variables and submit create and submit individual python files for each possible configuration based on template python files. Whether running on a cluster (i.e. Longleaf) or locally, one should submit this file as:
```
$ sh ~/klotsa/ABPs/runPeloopBinaryCluster.sh
```

If running on Longleaf, be sure you are running these simulations in the /proj/ (group workspace) or /pine/ (personal workspace) (note that /pine/ workspace has very limited storage compared to the group workspace). To determine which template python file to use, the user is prompted to answer a few questions that describe the initial conditions of the system and where the simulation is being run.

> Are you running on Longleaf (y/n)?

If true, answer `y` otherwise answer `n`. Answering this correctly will automatically specify the pathing to hoomd for you and proper slurm terminology for submitting a run to separate GPU nodes (as opposed to CPU nodes if ran locally). Your input is then followed by four yes-or-no questions in a row to specify the initial conditions

> Do you want homogeneous cluster (y/n)?
> 
> Do you want slow bulk, fast interface (y/n)?
> 
> Do you want fast bulk, slow interface (y/n)?
> 
> Do you want half slow, half fast cluster (y/n)?

Answering `y` to any of these will terminate the question asking process and choose that respective template python file for creating and submitting each simulation run. If `n` is answered to all of them, the submission will abort. Future initial conditions are planned to be added soon.

Once the file is submitted on Longleaf, a slurm-XXXXXXXXXXX.out file will be created that documents the progress of the run and which submission number it is (if ran on Longleaf). Be sure to check this to file to be sure the submission will take less than 11 days or else the simulation will be automatically aborted per Longleaf's maximum run length. Similarly, the slurm file is where to go to see if and what type of error occurred. There are two common errors: 1) a particle exits the simulation box, which occurs due to too great of particle overlap from too small of a time step (therefore, increase the time step and re-submit to fix this) and 2) the simulation finds a faulty GPU node. If the latter occurs, the simulation never starts and the slurm file remains empty despite the simulation queue (squeue -u <Longleaf username, i.e. Onyen>) saying the simulation is still running (and it will continue to take up a GPU node until it his the maximum time limit of 11:00:00 days unless cancelled beforehand (scancel <submission number>). If running locally, the estimated simulation time will be output to the terminal at a regular interval. In addition only the first error commonly occurs when you have too small of a time step. Testing a run locally (using the most active and hard spheres desired of your input systems) is a good method to find a good starting time step to use. Sometimes, a particle will exit the box later in the simulation, however, this is less common with the initial conditions being the main culprit for most errors.

### Submitting post-processing


## Collaboration

To collaborate or ask questions, please contact me at: njlauers@live.unc.edu. If you wish to use any part of this package in your research, please supply my name in the acknowledgments or cite any of my publications.
   
# Personal
