# klotsa
***

## Table of Contents
1. [Introduction](#introduction)
2. [ABPs](#abps)
   - [General Info](#general-info)
   - [Technologies](#technologies)
   - [Installation](#installation)
     - [Prerequisites](#prerequisites)
     - [HOOMD-Blue](#HOOMD-Blue)
       - [Local install via conda](#local-install-via-conda)
       - [Local install via source](#local-install-via-source)
       - [Cluster install via source](#cluster-install-via-source)
     - [Github](#Github)
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

### Prerequisites

First, navigate to the app store and install Xcode. You can use this as an IDE if you'd like. This should take a couple hours to install. Once Xcode installation is complete, open Xcode and agree to the license agreement. Alternatively, in a Terminal window, you can run: 

```
sudo xcodebuild -license
```

While Xcode is installing, navigate to anaconda.com to install Anaconda Individual Edition to get access to conda/miniconda. This will be used for installing hoomd/prerequisites. In addition, you can install Spyder through Anaconda for a different IDE. Open the Anaconda installer that was downloaded and follow the instructions until the installation is complete. 

Once the installations for both Anaconda and Xcode finish (be sure Xcode installation is complete and has been launched at least once as Homebrew uses it and it can help identify the Xcode command line tools, which are needed for Homebrew), navigate to https://brew.sh and install Homebrew for your Mac. Per their website (though double check to be sure this command is up to date), open your Terminal and enter:

```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Once homebrew finishes its install, install a few prerequisites with it:

```
brew install gnu-sed
```

Now, it's time to set the default shell to BASH. Select the apple symbol>System Preferences>Users & Groups. Click the lock and verify your account password to enable changes. Right click on your user and select Advanced Options. Under Login Shell, click the dropdown arrow and see if /bin/bash is available. If so, select it and press OK. Your default shell is now set to BASH, so any newly opened Terminal windows will operate with BASH. To verify which shell you are using, enter:

```
$ echo $SHELL
```

If the terminal output reads: `/usr/local/bin/bash`, then you're good to go! You can skip to the step of creating a virtual environment via Anaconda below. However, if the terminal output reads: `/bin/zsh`, which is the default shell for Mac computers, or you do not see /bin/bash as an option, you must follow the below lines to install BASH and make it your default shell. In your Terminal, input:

```
brew install bash
```

Locate the new bash installation, which is probably either `/usr/local/bin/bash` or `/bin/bash`. To do this on a mac, open your Terminal and in the command line, enter the following line to identify the path to your BASH install:

```
$ echo '/path/to/bash' | sudo tee -a /etc/shells;
```

After entering your administrator's password, your terminal should output:

```
$ /path/to/bash
```

Now, it's time to change your default operating shell from ZSH to BASH with the following command:

```
$ chsh -s /path/to/bash
```

Now, open a new Terminal window and enter:

```
$ echo $SHELL
/path/to/bash
$ echo $BASH_VERSION
X.X.XX(X)-release
```

The output should read similar to above. If you open the new Terminal window and see a [Process completed] in the output line and are unable to type anything, the shell/path to shell that you changed to does not exist. You must navigate to Apple icon>System Preferences>Users & Groups and follow the instructions above for changing your shell from the currently non-existent path back to the default `/bin/zsh` shell. Then, you must find the correct path to the BASH install and try again.

Next, you need to make a virtual environment via Anaconda to install HOOMD and its prerequisite modules. To do this, download Anaconda. Open the Anaconda-Navigator and select 'create' under the listed environments. Enter a name for your environment and select Python 3.8 for your Python version (which is the same as on Longleaf). After your environment is created, open a terminal and download the HOOMD pre-requisites into this environment:

```
$ cd ~
$ bash
$ source activate [virtual environment name]
$ conda install -c conda-forge sphinx git numpy cmake clang openmpi gsd numpy matplotlib yaml llvm ipython gsd pybind11 eigen ffmpeg
$ conda install -c anaconda conda-package-handling
```

### Prerequisites

### HOOMD-Blue

#### Local install via conda

This only works on your local computer. One downside to this is that if you need to compile anything special (i.e. the compile/cmake commands), you can't modify them.  However, this is the easiest method as we'll mainly just be doing local testing before running on the cluster, so I suggest using this method of installing HOOMD.

```
$ conda install -c conda-forge hoomd
```

All done! That was easy! This conda command will configure HOOMD-Blue v2.9.7 for your computer. You can skip to the section titled "Setting Up Github" to test your HOOMD-Blue install. Simply clone the directory or download a ZIP file and unzip it. Run a HOOMD simulation by entering the following in your Terminal window:

```
$ bash
$ source activate [virtual environment name]
$ sh /Path/to/Klotsa/ABPs/runPeloopBinaryCluster.sh
```

Once the Brownian equilibration starts, you can close the Terminal window to cancel the run. A full simulation should work fine.

#### Local install via source

See the next section for installing HOOMD-Blue via source on the cluster. First, download HOOMD-Blue version 2.9.7:

```
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```

or you can run the following command to download the most recent version of HOOMD-Blue (v3.0.0 beta at the time of writing this). This download instruction and github is designed to be used for HOOMD-Blue version 2.9.7:

```
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
$ cd hoomd-blue
$ git fetch --all --tags
$ git checkout tags/v2.9.7
$ python3 ./install-prereq-headers.py
```

If you use the following command, these instructions and git repository will not fully apply due to large modifications in HOOMD-Blue's prerequisites and how it is run. If you chose the former, proceed with these instructions by untarring the downloaded folder:

```
$ tar -xzvf hoomd-v2.9.7.tar.gz
```

Configure HOOMD-Blue. When configuring locally, be sure `-DENABLE_CUDA=OFF` and `-DENABLE_MPI=OFF` in the `cmake` tags. 

```
$ cd hoomd-v2.9.7
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=OFF -DENABLE_MPI=OFF
```

You can enter the following to enter a GUI to better see and verify that HOOMD was compiled properly. Namely, be sure MPI and CUDA were both identified. 

```
$ ccmake .
```

If everything looks good, build HOOMD: 

```
$ cmake --build ./ -j4
```

Test your build, first, exit the compile node, claim a gpu node, and use the built-in test command

```
$ cd ~/hoomd-v2.9.7/build
$ ctest
```

Since we built locally on a Mac OS computer and, in turn, do not have CUDA support, many tests will fail due to the requirement of a GPU.

#### Cluster install via source

Login to Longleaf with SSH using your ONYEN as your username:

```
$ ssh username@longleaf.unc.edu
```

If this is your first time loggin in from this computer, enter 'yes' to the prompt of remoting into a new computer. Then, type your password (current password for your ONYEN) and press enter. You will log into a log-in node located in your user's folder within the `/nas` directory. This is your home directory (`cd ~`). 

If you run your compile in the /nas directory, you will run it in your login node which can have memory limitations. Modify your ~/.bashrc file to enable quick login to a compile node (you can use one without graphics for compilation) for these steps, although you can also login to a GPU node:

```
$ cd ~
$ nano .bashrc
```

copy and paste these commands at the bottom of your .bashrc file so that I have easy access to different options:

```
alias sinteractive="srun -t 8:00:00 -p interact -N 1 --mem=6G --x11=first --pty /bin/bash"
alias sinteractivecompile="srun --ntasks=1 --cpus-per-task=8 --mem=8G --time=8:00:00 --pty /bin/bash"
alias sinteractivegpu="srun --ntasks=1 --cpus-per-task=8 --mem=8G --time=8:00:00 --partition=gpu --gres=gpu:1 --qos=gpu_access --pty /bin/bash"
alias sinteractivevolta="srun --ntasks=1 --cpus-per-task=8 --mem=8G --time=8:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --pty /bin/bash"
```

Install prerequisite modules and save them for future logins on cluster. These are the versions I have currently installed, but you can install the most recent versions as well.

```
$ module avail
$ module add git/2.33.0 cmake/3.18.0 clang/6.0 ffmpeg/3.4 geos/3.8.1 cuda/10.1 gcc/6.3.0 python/3.5.1
$ module save
$ module list
```

Create a python virtual environment to load everything into with 'install' command later. 

```
$ cd ~
$ mkdir virtual_envs
$ python3 -m venv virtual_envs/hoomd297
$ source ~/virtual_envs/hoomd297/bin/activate
```

Next, download HOOMD-Blue version 2.9.7:

```
$ cd ~
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```

or you can run the following command to download the most recent version of HOOMD-Blue (v3.0.0 beta at the time of writing this). This download instruction and github is designed to be used for HOOMD-Blue version 2.9.7:

```
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
$ cd hoomd-blue
$ git fetch --all --tags
$ git checkout tags/v2.9.7
$ python3 ./install-prereq-headers.py
```

If you use the following command, these instructions and git repository will not fully apply due to large modifications in HOOMD-Blue's prerequisites and how it is run. If you chose the former, proceed with these instructions by untarring the downloaded folder:

```
$ tar -xzvf hoomd-v2.9.7.tar.gz
```

Configure HOOMD-Blue. First, activate a compile node for more memory. After you get the compile node, re-activate your virtual environment and ensure the python3 is pathed to your virtual environment's python3 before compiling:

```
$ sinteractivevolta
$ source ~/virtual_envs/hoomd297/bin/activate
$ which python3
~/virtual_envs/hoomd297/bin/python3
```

Next, compile the build. When configuring on cluster, be sure `-DENABLE_CUDA=ON` and `-DENABLE_CUDA=ON` in the `cmake` tags. The former enables use of CUDA-enabled GPUs for very quick simulations. The latter enables use of MPI, allowing for use of a message passing interface for parallel programming. The `CC=gcc CXX=g++` prefix specify your compilers. 

```
$ cd hoomd-v2.9.7
$ mkdir build
$ cd build
$ CC=gcc CXX=g++ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON
```

You can enter the following to enter a GUI to better see and verify that HOOMD was compiled properly. Namely, be sure MPI and CUDA were both identified and that the CMAKE_INSTALL_PREFIX is equal to the path to your virtual environment's python3. 

```
$ ccmake .
```

If everything looks good, you can either press 't' to enter advanced mode or 'q' to leave the GUI. Next, build HOOMD: 

```
$ cmake --build ./ -j4
```

To test your build use the built-in test command.

```
$ ctest
```

I had a 51% pass rate. It is likely MPI is not linked properly for these tests, so a similar pass rate should be fine. Since we have both MPI and CUDA enabled, more tests should pass than our local build.

Finally, install HOOMD-Blue into your Python environment:

```
$ cmake install
```

Before running HOOMD-Blue, be sure you always have `source ~/virtual_envs/hoomd297/bin/activate` included at the beginning of any bash scripts.

### Github

First, open your teerminal and navigate to your home directory and start the ssh-agent in the background:

```
$ cd ~
$ eval "$(ssh-agent -s)"
```

This should output:

> Agent pid #####

where ##### is some combination of integers. Depending on your environment, you may need to use a different command. For example, you may need to use root access by running sudo -s -H before starting the ssh-agent, or you may need to use exec ssh-agent bash or exec ssh-agent zsh to run the ssh-agent.


Next, generate an ssh key with either of the following command while replacing your_email@example.com held within the quotations (do not remove quotation marks!) with the email you used for your Github profile. The former command is intended for non-legacy systems (i.e. macOS Monterrey):

```
$ ssh-keygen -t ed25519 -C "your_email@example.com"
```

If you are using a non-legacy system (i.e. macOS Monterrey), you will see the following output:

> Generating public/private ed25519 key pair.

If you are using a legacy system that doesn't support the Ed25519 algorithm you will get an output that reads:

> command not found

If that's the case, instead, use:

```
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Upon the succesful creation of an ssh key, the terminal should output: 

> Generating public/private rsa key pair.

and proceed with the following prompt: 

> Enter a file in which to save the key (/Users/you/.ssh/id_XXX):

XXX is replaced with the type of keypair you generated in the previous step. I always just press the enter key and the ssh-keycode is generated in the default file location, which is /Users/you/.ssh/id_rsa for the latter method (rsa key pair) or /Users/you/.ssh/id_ed25519 (ed25519 key pair). For now on, id_XXX will refer to either id_rsa or id_ed25519 depending on which key generation option you used. Alternatively, you can choose your own location in the /Users/you/.ssh folder. After responding to the above prompt, two more prompts will appear:

> Enter passphrase (empty for no passphrase): [Type a passphrase]
> Enter same passphrase again: [Type passphrase again]

Similar to the prompt before, I simply press enter for both of these prompts and leave the passphrase blank. 

##### Adding your SSH key to the ssh-agent

Now that you have your ssh-keycode, you will want to save it in your ssh-agent. Start the ssh-agent in the background again with the following prompt:

```
$ eval "$(ssh-agent -s)"
```

Next, we have to verify the existence and contents of your ~/.ssh/config file.  

```
$ cd ~/.ssh
$ ls
```

If you do not see the config file, create one:

```
$ touch ~/.ssh/config
```

Since you have a config file now, you want to open the file and modify its contents:

```
$ open ~/.ssh/config
```

Make sure the config file reads:

>Host *
>  AddKeysToAgent yes
>  UseKeychain yes
>  IdentityFile ~/.ssh/id_XXX

If it does not, copy and paste these lines into your file, ensuring that the lines under `Host *` are indented. Furthermore, if you chose not to add a passphrase to your key, you can choose to omit the UseKeychain line. Be sure to replace id_XXX in the command with the name of your private key file generated earlier.

You may also choose to add additional lines such that your config file reads:

>Host *
>  ServerAliveInterval 60
>  ServerAliveCountMax 30
>  AddKeysToAgent yes
>  UseKeychain yes
>  IdentityFile ~/.ssh/id_XXX
  
These two additional lines refresh your connection to github (when uploading files) in case you have a large amount of data to upload or else the link to github will be automatically broken after a short amount of time.

If you see an error later on, such as:

> /Users/USER/.ssh/config: line 16: Bad configuration option: usekeychain

Then you want to add the following line to your config file indented under `Host *`:

```
IgnoreUnknown AddKeysToAgent,UseKeychain
```

For me, that line was necessary for Longleaf, but was not necessary for setting up Github on my local computer. Now, add your SSH private key to the ssh-agent and store your passphrase in the keychain. Be sure to replace id_XXX in the command with the name of your private key file.
  
```
$ ssh-add -K ~/.ssh/id_XXX
```

The -K option is Apple's standard version of ssh-add, which stores the passphrase in your keychain for you when you add an SSH key to the ssh-agent. If you chose not to add a passphrase to your key, run the command without the `-K` option. In MacOS Monterey (12.0), the -K and -A flags are deprecated and have been replaced by the --apple-use-keychain and --apple-load-keychain flags, respectively.

Finally, now that you have the ssh-key fully set up on your local computer or on the cluster (Longleaf), you want to copy your ssh-keycode so that you can put it in your Github profile. 

Next, is where Longleaf deviates from a local computer. On your local computer, copy the contents of the file to your clipboard that you saved your ssh-keycode in by inputting the following into your terminal (for a mac):
```
$ pbcopy < ~/.ssh/id_XXX.pub
```
If you want to set up your Github profile on Longleaf, simply download the file that you saved your ssh-keycode in to your local computer:

```
$ scp username@longleaf.unc.edu:~/.ssh/id_XXX.pub /~/Desktop/
```

Then, copy the contents of this file to your clipboard:

```
$ pbcopy < ~/Desktop/id_XXX.pub
```

With your ssh-key in your clipboard, navigate and login to Github. In the upper-right corner of any page, click your profile photo, then click Settings. In the user settings sidebar, click SSH and GPG keys. Click New SSH key or Add SSH key. In the "Title" field, add a descriptive label for the new key. For example, if you're using a personal Mac, you might call this key "Personal MacBook Air". Paste your key into the "Key" field. Click Add SSH key. Now you can clone and upload to that repository.

After doing the above steps for setting up Github, you can download this current github by navigate to your home directory then entering the command below.
```
$ cd ~
$ git clone https://github.com/njlauersdorf/klotsa.git
```

Now you're ready to run some code! 

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
