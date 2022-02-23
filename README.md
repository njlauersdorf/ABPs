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
* Anaconda (https://www.anaconda.com/): Version 4.11.0

## Installation

It is highly recommended one install's both HOOMD-Blue and this github repository in their user's home directory due to the pathing in the analysis files. Furthermore, this installation instruction and github repository is intended for use with HOOMD v2.9.7. This is not the most recent version. There is a beta release for v3.0.0. A large number of changes were made for how one submits simulations. The post processing should work similarly, however, simulation submission files need to be adjusted for use on newer HOOMD versions. One should reference the guide for the newest version (https://hoomd-blue.readthedocs.io/en/latest/) to determine how these submission files need to be modified.

### Step 1: Setting up your Mac to code

First, navigate to the app store and install Xcode. You can use this as an IDE if you'd like. This should take a couple hours to install. While this is installing, navigate to anaconda.com to install Anaconda Individual Edition to get access to conda/miniconda. This will be used for installing hoomd/prerequisites. In addition, you can install Spyder through Anaconda for a different IDE. Open the Anaconda installer that was downloaded and follow the instructions. Once the Anaconda installation finishes, open it from your Applications and create a virtual environment.

[Contribution guidelines for this project](docs/CONTRIBUTING.md)

### HOOMD-Blue

**For installation on local desktop:**

Installing Prerequisites:

BASH
First, you must switch your computer's shell to BASH from ZSH. To do this on a mac, open your Terminal and in the command line, enter:

```
$ echo '/usr/local/bin/bash' | sudo tee -a /etc/shells;
$ chsh -s /usr/local/bin/bash
```

Close your current terminal window and open a new one. To verify it worked, enter the following two lines. You should see similar outputs as shown below:

```
$ echo $SHELL
/usr/local/bin/bash
$ echo $BASH_VERSION
4.2.37(2)-release
```

Note, you should not see a path which ends in /zsh. However, if you do, your default shell is still ZSH, then perform the alternate steps noted here. First, close and reopen the command window. Then, input the following line: 

```
$ chsh -s /bin/bash
```
Test your default shell as shown above. If it's now BASH, you are good to go! 

Homebrew

Following the instructions at https://brew.sh/, in your command line, simply type the following:

```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Next, you need to make a virtual environment to install HOOMD prerequisite modules. To do this, download Anaconda. Open the Anaconda-Navigator and select 'create' under the listed environments. 

Install prerequisites.
```
$ cd ~
$ bash
$ conda install -c conda-forge sphinx git numpy cmake clang
```
Download HOOMD-Blue version 2.9.7
```
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```
or
```
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
```
Configure HOOMD-Blue. When configuring locally, be sure `-DENABLE_CUDA=OFF` in the `cmake` tags. When configuring locally, you installed Open MPI, let `-DENABLE_MPI=ON` in the `cmake` tags, allowing for use of a message passing interface for parallel programming.

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

Make it so HOOMD-Blue can find the correct Python later by changing the aliases to that on a Mac. This makes running uniform code much easier. Navigate to your .bash_profile and be sure it reads:
```
# User specific environment and startup programs
alias python='/usr/bin/python2.7'
alias python3='/usr/bin/python3.5'
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
Configure HOOMD-Blue. When configuring on cluster, be sure `-DENABLE_CUDA=ON` in the `cmake` tags as you will be using CUDA-supported GPUs and MPI is enabled, allowing for use of a message passing interface for parallel programming.
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

### Setting up GitHub

First, open your teerminal and navigate to your home directory and start the ssh-agent in the background:
```
$ cd ~
$ eval "$(ssh-agent -s)"
```

This should output:

> Agent pid #####

where ##### is some combination of integers.


Next, generate an ssh key with either of the following command while replacing your_email@example.com with the email you used for your Github profile:

```
$ ssh-keygen -t ed25519 -C "your_email@example.com"
```

If you are using a legacy system that doesn't support the Ed25519 algorithm you will get an output that reads:

> command not found

If that's the case, instead, use:

```
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

For me, either of these would read: 

```
$ ssh-keygen -t ed25519 -C "njlauersdorf@wisc.edu"
```
or
```
$ ssh-keygen -t rsa -b 4096 -C "njlauersdorf@wisc.edu"
```

Upon the succesful creation of an ssh key, the terminal should output: 

> Generating public/private rsa key pair.

and proceed with the following prompt: 

> Enter a file in which to save the key (/Users/you/.ssh/id_rsa):

I always just press the enter key and the ssh-keycode is generated in the default file location (/Users/you/.ssh/id_rsa). Alternatively, you can choose your own location in the /Users/you/.ssh folder. After responding to the above prompt, two more prompts will appear:

> Enter passphrase (empty for no passphrase): [Type a passphrase]
> Enter same passphrase again: [Type passphrase again]

Similar to the prompt before, I simply press enter for both of these prompts and leave the passphrase blank. 

##### Adding your SSH key to the ssh-agent

Now that you have your ssh-keycode, you will want to save it in your ssh-agent. Start the ssh-agent in the background with the following prompt:

```
$ eval "$(ssh-agent -s)"
```

The terminal will output prompt like the following:

> Agent pid 59566

Depending on your environment, you may need to use a different command. For example, you may need to use root access by running sudo -s -H before starting the ssh-agent, or you may need to use exec ssh-agent bash or exec ssh-agent zsh to run the ssh-agent.

Next, we have to verify the existence and contents of your ~/.ssh/config file.  

```
$ cd ~/.ssh
$ ls
```

if you do not see the config file, create one:

```
$ touch ~/.ssh/config
```

Since you have a config file now, you want to open the file and modify its contents:

```
$ open ~/.ssh/config
```

Make sure the config file reads. If it does not, copy and paste these lines into your file, ensuring that the lines under `Host *` are indented. Furthermore, if you chose not to add a passphrase to your key, you should omit the UseKeychain line. If you created your key with a different name, or if you are adding an existing key that has a different name, replace id_rsa in the command with the name of your private key file.

>Host *
>  AddKeysToAgent yes
>  UseKeychain yes
>  IdentityFile ~/.ssh/id_rsa

You may also choose to add additional lines such that your config file reads:

>Host *
>  ServerAliveInterval 60
>  ServerAliveCountMax 30
>  AddKeysToAgent yes
>  UseKeychain yes
>  IdentityFile ~/.ssh/id_rsa
  
These two additional lines refresh your connection to github (when uploading files) in case you have a large amount of data to upload or else the link to github will be automatically broken after a short amount of time.

If you see an error later on, such as:

> /Users/USER/.ssh/config: line 16: Bad configuration option: usekeychain

Then you want to add the following line to your config file indented under `Host *`:

```
IgnoreUnknown AddKeysToAgent,UseKeychain
```
That line was necessary for Longleaf, but was not necessary for setting up Github on my local computer. Now, add your SSH private key to the ssh-agent and store your passphrase in the keychain. If you created your key with a different name, or if you are adding an existing key that has a different name, replace id_rsa in the command with the name of your private key file.
  
```
$ ssh-add -K ~/.ssh/id_rsa
```

The -K option is Apple's standard version of ssh-add, which stores the passphrase in your keychain for you when you add an SSH key to the ssh-agent. If you chose not to add a passphrase to your key, run the command without the `-K` option. In MacOS Monterey (12.0), the -K and -A flags are deprecated and have been replaced by the --apple-use-keychain and --apple-load-keychain flags, respectively.

Finally, now that you have the ssh-key fully set up on your local computer or on the cluster (Longleaf), you want to copy your ssh-keycode so that you can put it in your Github profile. 

Next, is where Longleaf deviates from a local computer. On your local computer, copy the contents of the file to your clipboard that you saved your ssh-keycode in by inputting the following into your terminal (for a mac):
```
$ pbcopy < ~/.ssh/id_rsa.pub
```
If you want to set up your Github profile on Longleaf, simply download the file that you saved your ssh-keycode in to your local computer:

```
$ scp username@longleaf.unc.edu:~/.ssh/id_rsa.pub /~/Desktop/
```

Then, copy the contents of this file to your clipboard:

```
$ pbcopy < ~/Desktop/id_rsa.pub
```

With your ssh-key in your clipboard, navigate and login to Github. In the upper-right corner of any page, click your profile photo, then click Settings. In the user settings sidebar, click SSH and GPG keys. Click New SSH key or Add SSH key. In the "Title" field, add a descriptive label for the new key. For example, if you're using a personal Mac, you might call this key "Personal MacBook Air". Paste your key into the "Key" field. Click Add SSH key.
> 
#### Local installation
To set up github, first navigate to your home directory
####Klotsa github repository
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

If running on Longleaf, be sure you are running these simulations in the /proj/ (group workspace) or /pine/ (personal workspace) (note that /pine/ workspace has very limited storage compared to the group workspace). Upon submitting this bash file, a folder named /MM_DD_YYYY_parent, where MM is the month, DD is the day, and YYYY is the year, will be created where each python file for every run is created and saved. To determine which template python file to use, the user is prompted to answer a few questions that describe the initial conditions of the system and where the simulation is being run.

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
The bash file used to submit a simulation is /klotsa/ABPs/post_proc_binary.sh. This file will submit the specified post processing routine for every simulation file (.gsd) in the current directory. Submit this bash file similarly to running a simulation:
   
```
$ sh ~/klotsa/ABPs/post_proc_binary.sh
```
   
   
Upon submitting this bash file, three folders will be created: /MM_DD_YYYY_txt_files, /MM_DD_YYYY_pic_files, and /MM_DD_YYYY_vid_files, where they store the outputted .txt, .png, and .mp4 files respectively for all post-processing scripts started on that date (MM/DD/YYYY). In addition, there will be prompts to specify a few analytical details, such as:
   
> What is your bin size?
> 
> What is your step size?
> 
> What do you want to analyze?
   
   
where it seeks the length of the bin for meshing the system (typically 5), the time step size to be analyzed (always 1 unless we care about velocity), and our post-processing method. A second bash script will be submitted for each .gsd file in the current directory where this information is passed to. There, a python post-processing file will be ran on each .gsd file separately that corresponds to our response to the third prompt. These post processing files typically output a series of .png files for each frame and a .txt file (located in the /MM_DD_YYYY_txt_files_folder) compiling certain information at each time step. Once the python script finishes running, the second bash file will compile each series of .png files (located in the /MM_DD_YYYY_pic_files folder) into a .mp4 (located in the /MM_DD_YYYY_vid_files folder). The bash script will proceed to delete all .png files that were created by that post-processing script. Therefore, what we're left with will always be .mp4 files. or.txt files unless an error occurs. These .txt files are typically read into certain jupyter notebook scripts to analyze them averaged over time and to identify trends over our input parameters.

## Collaboration

To collaborate or ask questions, please contact me at: njlauers@live.unc.edu. If you wish to use any part of this package in your research, please supply my name in the acknowledgments or cite any of my publications.
   
# Personal
