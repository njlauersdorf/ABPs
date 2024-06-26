# Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [HOOMD-Blue Installation Instructions](#HOOMD-Blue-Installation-Instructions)
    - [Creating a Longleaf account](#creating-a-longleaf-account)
    - [Install Prerequisite Software and Tools](#Install-Prerequisite-Software-and-Tools)
    - [Install HOOMD-Blue V3.0](#Install-HOOMD-Blue-V3.0)
        - [Local Install via Conda](#local-install-via-conda)
        - [Cluster Install via Conda](#cluster-install-via-conda)
    - [Install HOOMD-Blue V2.9](#Install-HOOMD-Blue-V2.9)
      - [Local install via conda](#local-install-via-conda)
      - [Local install via source](#local-install-via-source)
      - [Cluster install via source](#cluster-install-via-source)
5. [ABPs Code Package](#abps-code-package)
    - [Organization](#organization)
    - [Running Simulations](#Running-Simulations)
    - [Longleaf Simulations and SLURM](#Longleaf-Simulations-and-SLURM)
    - [Submitting post-processing](#Submitting-Post-Processing)
    - [Downloading data from Longleaf](#Downloading-data-from-Longleaf)
6. [Collaboration](#collaboration)

# General Info
This github project consists of a collection of software and packages for simulating systems of active and passive Brownian particles in HOOMD-Blue using molecular dynamics. Simulation submission bash files read in the user's desired system (particle activities, particle softness, system density, particle fraction, particle size, and population size) and initial conditions (box dimensions, box shape, initial positions, whether it's a randomized gas phase or an instantiated liquid-gas phase separated system, and initial orientation, whether it's randomized or biased with local alignment, i.e. the liquid-gas interface) creates an initial hoomd snapshot/frame using these specified conditions, and runs and saves a simulation for the desired time at the desired stepsize. In addition, this software enables the post-processing of the outputted system over time with focus placed on steady-state measurements and characterization (see Lauersdorf, et al. (2021) for derivation of the theory these measurements led to). 

# Technologies

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

# HOOMD-Blue Installation Instructions

It is highly recommended one install's both HOOMD-Blue and this github repository in their user's home directory due to the pathing in the analysis files. This can be modified line-by-line otherwise. Furthermore, this installation instruction and github repository is intended for use with HOOMD v2.9.7. This is not the most recent version of HOOMD. HOOMD-Blue v3+ drastically changes the methodology for running simulations, hence, the simulation submission scripts must be modified to use with HOOMD v3+. However, post-processing files can still be used normally. One should reference the guide for the newest version (https://hoomd-blue.readthedocs.io/en/latest/) to determine how these submission files need to be modified. It is recommended one installs HOOMD-Blue on either a Mac or Linux OS or problems may arise with use of an emulat. It is strongly suggested one sticks to Mac or Linux when running simulations. Post-processing scripts of simulations have been written and tested for analysis on both Mac, Linux, and Windows.

## Creating a Longleaf account

To create a Longleaf account, follow the instructions on this link: [https://help.rc.unc.edu/request-a-cluster-account/](https://help.rc.unc.edu/request-a-cluster-account/). In a relevant prompt, be sure to mention that you "need access to the GPUs on Longleaf" and that you "need access to the shared /proj space for Daphne Klotsa's lab (/proj/dklotsalab/)". You may need to provide approval from your advisor or cc her in an email for the latter request. If you get an account and they don't fulfill the latter two requests, email them at research@unc.edu to explicitly ask for it.

## Install Prerequisite Software and Tools

### Homebrew (mac exclusive)
For mac users, Homebrew (https://brew.sh/) is a package manager that will help you to install and manage the required software packages easily. If you're familiar with Linux systems, think of Homebrew as apt-get on Debian-based systems or dnf/yum on RedHat-based systems.

Go to https://brew.sh and run the installation command shown in Terminal.

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

This will install Homebrew package manager as well as the Command Line Tools that are used to carry out the following guide.

If an error ensues related to Xcode, you can install the full Xcode suite via the app store. You can use this as an IDE if you'd like. This should take a couple hours to install. Once Xcode installation is complete, open Xcode and agree to the license agreement so that Homebrew can help identify the Xcode command line tools. Alternatively, in a Terminal window, you can run: 

```
$ sudo xcodebuild -license
```

Once homebrew finishes its install, install a few prerequisites with it:

```
$ brew install gnu-sed
```

### git

Git will be installed with the following commands.

```
brew install git
```

Note that macOS's default shell has changed to zsh from bash. There should not be much significant differences but if you do prefer to use bash, install bash via homebrew brew install bash and then change the default shell with chsh -s /usr/local/bin/bash.

#### Setting up your Github

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

#### Adding your SSH key to the ssh-agent

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

### Miniconda

Go to https://docs.conda.io/en/latest/miniconda.html to download and install Miniconda for Python 3.9 or run the code shown below to install Miniconda 4.11.0. I have confirmed that updating to version 4.12.0 works without any problem.

```
curl -sO https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-MacOSX-x86_64.sh
/bin/bash ./Miniconda*.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash zsh
source $HOME/.bash_profile $HOME/.zshrc
rm $HOME/Miniconda*.sh
```

You can confirm the Miniconda installation by running conda -V in your terminal. The output should resemble the following:

conda 4.11.0

Alternatively to miniconda, you can download the full Anaconda suite. Open the Anaconda-Navigator and select 'create' under the listed environments. Enter a name for your environment and select Python 3.8 for your Python version (which is the same as on Longleaf). 

#### Create Conda Environment

Conda is capable of creating environments with specified versions of python and packages. You can switch between the environments by activating them, providing you with a customized, separate environment for each specific task. In this guide, we will separate our environments into one specificically for HOOMD-blue simulation package and another for post-processing of the simulation data. By default, the environments are contained within $HOME/miniconda3/envs/, but this guide will contain all the environments within $HOME/pyEnvs/. If desired, changing the environment directory should not affect workflow as long as the paths in your simulation scripts are changed accordingly.

First, we start with creating an environment for HOOMD-blue simulation package.

```
mkdir $HOME/pyEnvs
conda create --prefix $HOME/pyEnvs/rekt -y python=3.8
```

Note that I am using python 3.8 here, but HOOMD-blue V3.0 requires any python above 3.6. Change this according to your needs.

Activate the created conda environment.
```
conda activate $HOME/pyEnvs/hoomd
Run conda list to ensure the specified python version is installed.
```

After your environment is created either with miniconda or full, open a terminal and download the HOOMD pre-requisites into this environment:

```
$ conda install -c conda-forge sphinx git numpy cmake clang openmpi gsd numpy matplotlib yaml llvm ipython gsd pybind11 eigen ffmpeg
$ conda install -c anaconda conda-package-handling
$ python3 -m pip install gsd
$ python3 -m pip install freud-analysis
$ python3 -m pip install shapely
```

## Install HOOMD-Blue V3.0

### Local Install via Conda

```
conda install -c conda-forge eigen numpy pybind11 tbb tbb-devel gsd cereal "hoomd=3.0.0=*cpu*"
```

Note the quotation marks as zsh may interpret the asterisk for something else and not pass it to conda.

If you see an output regarding

failed with initial frozen solve,

don't worry about it. It is most likely will resolve itself.

As far as I know, there is no support from HOOMD-blue for AMD GPUs on Mac (more specifically no support for ROCm for Mac). Thus, you shouldn't need any GPU related builds on your Mac. I doubt this will ever be the case, but if you do have an eGPU, just search for instructions on CUDA setup on Macs and then install the GPU build instead of CPU.

This sets up the conda environment with HOOMD-blue simulation package on your machine. Confirm that the simulation package is working by starting up a python session with python. Then, run the following to confirm the installation.

```
import hoomd
integrator = hoomd.md.Integrator(dt=0.005)
integrator.dt
```

This should output the following.

0.005

### Cluster Install via Conda

This guide will walk you through the installation process for HOOMD-blue(http://glotzerlab.engin.umich.edu/hoomd-blue/) molecular dynamics simulation package and some commonly used data processing tools on the UNC-CH's Longleaf Research Computing system. Prior experience with UNIX CLI would be helpful for you to understand this guide better, but is not required. This installation guide assumes you have established an account on the Longleaf Cluster (https://its.unc.edu/research-computing/request-a-cluster-account/). Should you choose or need to run any HOOMD-blue simulations on the Dogwood Cluster, this installation guide can be followed without any alterations. The following steps assume you have bash as your shell, which is the default setup for longleaf. If you chose to use other shells such as tcsh, change the commands appropriately. There may be some syntax differences that require you to modify the commands accordingly.

#### Load Modules

Modules are pre-built software that are on the cluster for you to easily use. Load and save the following modules

git
gcc/9.1.0
cuda/11.4
cmake
geos

```
module load git gcc/9.1.0 cuda/11.4 cmake geos
module save
```

#### Install miniconda

Type the following commands to download and install miniconda3 with Python version 3.9 and conda version 4.11.

```
curl -sO https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
bash ./Miniconda*.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source $HOME/.bashrc
rm $HOME/Miniconda*.sh
```

You can confirm the Miniconda installation by running conda -V in your terminal. The output should resemble the following:

conda 4.11.0

Not necessary, but you can update your conda install with conda install -n base -c defaults conda.

#### Activate Conda Environment

Conda is capable of creating environments with specified versions of python and packages. You can switch between the environments by activating them, providing you with a customized, separate environment for each specific task. In this guide, we will separate our environments into one specificically for HOOMD-blue simulation package and another for post-processing of the simulation data.

First, we start with creating an environment for HOOMD-blue simulation package.

```
conda create -n hoomd -y python=3.8
```

Note that I am using python 3.8 here, but HOOMD-blue V3 requires any python above 3.6. Change this according to your needs.

Activate the created conda environment.

```
conda activate rekt
```

Run conda list to ensure the specified python version is installed.

Install the prerequisite conda packages needed for building HOOMD-blue.

```
conda install -c conda-forge eigen numpy pybind11 tbb tbb-devel gsd cereal
```

Confirm the installation by running conda list again.

#### Build and Install HOOMD-BLue V3.0

Start by cloning the github repository.

```
git clone --recursive https://github.com/glotzerlab/hoomd-blue
```

Configure build settings with CUDA and TBB enabled.

```
CC=gcc CXX=g++ cmake -B build/hoomd -S hoomd-blue -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_GPU=ON -DENABLE_TBB=ON
```

Once the configuration has finished, build and install HOOMD-blue.

```
cmake --build build/hoomd -j8
cmake --install build/hoomd
```

To confirm the installation, start an interactive python session with python and run the following.

```
import hoomd
integrator = hoomd.md.Integrator(dt=0.005)
integrator.dt
```

This should output the following.

0.005

## Install HOOMD-Blue V2.9

### Local install via conda

This only works on your local computer. One downside to this is that if you need to compile anything special (i.e. the compile/cmake commands), you can't modify them.  However, this is the easiest method as we'll mainly just be doing local testing before running on the cluster, so I suggest using this method of installing HOOMD.

```
$ conda install -c conda-forge hoomd
```

All done! That was easy! 

### Local install via source

First, download HOOMD-Blue version 2.9.7:

```
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```

Proceed with these instructions by untarring the downloaded folder:

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

Finally, install HOOMD-Blue into your Python environment:

```
$ cmake --install .
```

Before running HOOMD-Blue, be sure you always have `source ~/virtual_envs/[virtual environment name]/bin/activate` included at the beginning of any bash scripts.

### Cluster Install Via Source

This guide will walk you through the installation process for HOOMD-blue([http://glotzerlab.engin.umich.edu/hoomd-blue/](http://glotzerlab.engin.umich.edu/hoomd-blue/)) molecular dynamics simulation package and some commonly used data processing tools on the UNC-CH's Longleaf Research Computing system. Prior experience with UNIX CLI would be helpful for you to understand this guide better, but is not required. This installation guide assumes you have established an account on the Longleaf Cluster ([https://its.unc.edu/research-computing/request-a-cluster-account/](https://help.rc.unc.edu/request-a-cluster-account/)). Should you choose or need to run any HOOMD-blue simulations on the Dogwood Cluster, this installation guide can be followed without any alterations. The following steps assume you have bash as your shell, which is the default setup for longleaf. If you chose to use other shells such as tcsh, change the commands appropriately. There may be some syntax differences that require you to modify the commands accordingly.

Once you get access to UNC's Longleaf supercomputer, email: research@unc.edu. Ask for "access to the GPUs" along with "read, write, and execute permissions for the contents of /proj/dklotsalab directory"

Login to Longleaf with SSH using your ONYEN as your username:

```
$ ssh username@longleaf.unc.edu
```

If this is your first time logging in from this computer, enter 'yes' to the prompt of remoting into a new computer. Then, type your password (current password for your ONYEN) and press enter. You will log into a log-in node located in your user's folder within the `/nas` directory. This is your home directory (`cd ~`). 

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

#### Load Modules

Install prerequisite modules and save them for future logins on cluster. These are the versions I have currently installed, but you can install the most recent versions as well.

```
$ module avail
$ module add git/2.33.0 cmake/3.18.0 clang/6.0 ffmpeg/3.4 geos/3.8.1 cuda/10.1 gcc/6.3.0 python/3.5.1
$ module save
$ module list
```

#### Create a Virtual Environment

Create a python virtual environment to load everything into with 'install' command later. Make sure it is the same name as your local virtual environment for uniformity between your local HOOMD and the Longleaf HOOMD codes.

```
$ cd ~
$ mkdir virtual_envs
$ python3 -m venv virtual_envs/rekt
$ source ~/virtual_envs/rekt/bin/activate
```

Install some useful tools for HOOMD post-processing:

```
$ python3 -m pip install gsd
$ python3 -m pip install freud-analysis
$ python3 -m pip install shapely
```

Next, download HOOMD-Blue version 2.9.7:

```
$ cd ~
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz
```

Proceed by untarring the downloaded folder:

```
$ tar -xzvf hoomd-v2.9.7.tar.gz
```


or you can download via Github:

```
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
$ cd hoomd-blue
$ git fetch --all --tags
$ git checkout tags/v2.9.7
$ python3 ./install-prereq-headers.py
```

Configure HOOMD-Blue. First, activate a compile node for more memory. After you get the compile node, re-activate your virtual environment and ensure the python3 is pathed to your virtual environment's python3 before compiling:

```
$ sinteractivevolta
$ source ~/virtual_envs/rekt/bin/activate
$ which python3
```
> ~/virtual_envs/[virtual environment name]/bin/python3


Next, compile the build. When configuring on cluster, be sure `-DENABLE_CUDA=ON` and `-DENABLE_CUDA=ON` in the `cmake` tags. The former enables use of CUDA-enabled GPUs for very quick simulations. The latter enables use of MPI, allowing for use of a message passing interface for parallel programming. The `CC=gcc CXX=g++` prefix specify your compilers. 

```
$ cd hoomd-v2.9.7
$ mkdir build
$ cd build
$ CC=gcc CXX=g++ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF
```

You can enter the following to enter a GUI to better see and verify that HOOMD was compiled properly. Namely, be sure MPI and CUDA were both identified and that the CMAKE_INSTALL_PREFIX is equal to the path to your virtual environment's python3. 

```
$ ccmake .
```

If everything looks good, you can either press 't' to enter advanced mode or 'q' to leave the GUI. If you need to modify something, you can enter your changes then proceed to press 'c', which outputs text. Once that's done, you can press 'e' then 'g'.

Next, build HOOMD: 

```
$ cmake --build ./ -j4
```

To test your build use the built-in test command.

```
$ ctest
```

I had a 51% pass rate since MPI is active in our build, so a similar pass rate should be fine. Since we have CUDA enabled (but not MPI), more tests should pass than our local build.

Finally, install HOOMD-Blue into your Python environment:

```
$ cmake --install .
```

Before running HOOMD-Blue, be sure you always have `source ~/virtual_envs/[virtual environment name]/bin/activate` included at the beginning of any bash scripts.

## ABPs Code Package
Here we will discuss some basics of SLURM which is Longleaf's method of submitting and managing running of code. Longleaf uses BASH, similar to your Mac now, so you can easily submit code using the `sh` prefix before a bash script for defining variables and submitting individual runs for each specified set of initial conditions. This github project utilizes bash scripts to read in user's desired measurement/simulation type, select the desired python file to run (either for a simulation or post-processing) based on user input, and to read in the specified initial/system conditions into a template python file for a) post-processing of each .gsd file (simulation file) within the current directory or b) create a python file for instantiating a system and running that file to simulate each possible configuration of initial conditions inputted, which, in turn, outputs a .gsd file detailing the simulation at each time step.  

### Organization
This package is organized into 7 main directories:
1. deprecated: contains deprecated files for post-processing and running simulations whose structure were updated to fit with the new standard of running post-processing or simulation files. All functionality of these files is included in the updated files in post_proc detailed later.
2. images: contains images used in Github readme.md.
3. papers: contains LaTeX files for our published works
4. post_proc: contains post-processing files that are 1) deprecated but need to be updated contained in /post_proc/to_update directory, current, updated files for running simulations and post-processing conforming with the proper structure in /post_proc/lib directory, post-processing files for cleaning, wrangling, and analyzing wrangled data in Jupyter Notebooks in /post_proc/Jupyter directory, and template and shell scripts for running post-processing and simulations in parent directory: /post_proc/
5. sample_post_proc_outputs: contains sample image, video, and txt outputs from all post-processing files
6. sample_sim: contains sample simulation file for analysis of post-processing files to verify functionality
7. theory: contains python files that apply theory to make predictions. Does not analyze simulation files. Can be run on their own.

### Running Simulations
The bash file used to submit a simulation is /klotsa/ABPs/runPeloopBinaryCluster.sh. Before submitting the bash file, one should manually input all desired possible physical conditions of each system to run as lists in each variable's location at the top of the page. The bash file will loop through all possible pairings of these variables and submit create and submit individual python files for each possible configuration based on template python files. Whether running on a cluster (i.e. Longleaf) or locally, one should submit this file as:

```
$ sh ~/ABPs/runPeloopBinaryCluster.sh
```

If running on Longleaf, be sure you are running these simulations in the /proj/ (group workspace) or /pine/ (personal workspace) (note that /pine/ workspace has very limited storage compared to the group workspace). Upon submitting this bash file, a folder named /MM_DD_YYYY_parent, where MM is the month, DD is the day, and YYYY is the year, will be created where each python file for every run is created and saved. To determine which template python file to use, the user is prompted to answer a few questions that describe the initial conditions of the system and where the simulation is being run.

> Are you running on Longleaf (y/n)?

If true, answer `y` otherwise answer `n`. Answering this correctly will automatically specify the pathing to hoomd for you and proper slurm terminology for submitting a run to separate GPU nodes (as opposed to CPU nodes if ran locally). Your input is then followed by four yes-or-no questions in a row to specify the initial conditions

|----------------------------------------------------------------------|
|      Possible simulation options and corresponding user inputs       |
|          **************************************************          |
|          -------------Random initial conditions-------------         |
| Random gas: random_init                                              |
|          ----------Near steady-state MIPS clusters----------         |
| Homogeneous cluster: homogeneous_cluster                             |
| 100% slow bulk, 100% fast interface cluster: slow_bulk_cluster       |
| 100% fast bulk, 100% slow interface cluster: fast_bulk_cluster       |
| Half-slow, half-fast cluster: half_cluster                           |
|             ----------Bulk only of MIPS cluster----------            |
| Constant pressure (changing volume): constant_pressure               |
|          -----------Elongated planar membranes--------------         |
| Slow planar membrane: slow_membrane                                  |
| Immobile planar membrane: immobile_membrane                          |
| Immobile oriented planar membrane: immobile_orient_membrane          |
| Slow constrained planar membrane: slow_constrained_membrane          |
| Slow adsorb constrained membrane: slow_adsorb_constrained_membrane   |
| Slow interior constrained membrane: slow_int_constrained_membrane    |
| Diffusive mixture membrane: slow_int_constrained_membrane_dif_temp   |
|----------------------------------------------------------------------|

> What initial conditions do you want (see above for options)?

For whichever initial condition you desire (see left of colon for brief description of each option and more detailed description within called functions themselves), input the exact wording right of the colon (e.g. for a "Random gas" initial condition, you'd want to input "random_init" without the quotation marks. If your input does not match any possible options, the submission will abort. Future initial conditions are planned to be added soon. 

At the end of that bash script, a second bash script, either run_local.sh or run_gpu.sh depending on if you selected to run on your local CPU or Longleaf's GPU respectively, will be submitted for each .gsd file in the current directory where this information is passed to. These shell scripts feed the specified initial conditions into a template python file, run_sim_sample.py, and saves that new python file in the parent directory based on the current date, then submits the python file to Longleaf (using SLURM) or your local computer to run. This python file can be referenced later to verify initial conditions. 

### Longleaf Simulations and SLURM

Once the python file is submitted on Longleaf, a slurm-XXXXXXXXXXX.out file will be created that documents the progress of the run and which submission number it is (if ran on Longleaf). Be sure to check this to file to be sure the submission will take less than 11 days or else the simulation will be automatically aborted per Longleaf's maximum run length. Similarly, the slurm file is where to go to see if and what type of error occurred. There are two common errors: 1) a particle exits the simulation box, which occurs due to too great of particle overlap from too small of a time step (therefore, increase the time step and re-submit to fix this) and 2) the simulation finds a faulty GPU node. If the latter occurs, the simulation never starts and the slurm file remains empty despite the simulation queue (squeue -u <Longleaf username, i.e. Onyen>) saying the simulation is still running (and it will continue to take up a GPU node until it his the maximum time limit of 11:00:00 days unless cancelled beforehand (scancel <submission number>). If running locally, the estimated simulation time will be output to the terminal at a regular interval. In addition only the first error commonly occurs when you have too small of a time step. Testing a run locally (using the most active and hard spheres desired of your input systems) is a good method to find a good starting time step to use. Sometimes, a particle will exit the box later in the simulation, however, this is less common with the initial conditions being the main culprit for most errors.
   
You can check the progress of your simulations using some basic slurm commands:
   
```
squeue -u [ONYEN]   
```

If you need to cancel a simulation, you can run:
   
```
$ scancel [SLURM RUN ID]   
```
   
where SLURM RUN ID is the XXXXXXX simulation number in your slurm_XXXXXXX.out file or when you input `squeue`.
   
### Submitting post-processing
The bash file used to submit a simulation is /klotsa/ABPs/post_proc_binary.sh. While HOOMD-Blue can only be run on MacOS or Linux based systems, post-processing of simulation files can be run on Windows as well. This file will submit the specified post processing routine for every simulation file (.gsd) in the current directory. Submit this bash file similarly to running a simulation:
   
```
$ sh ~/ABPs/post_proc_binary.sh
```
   
Upon submitting this bash file, three folders will be created: /MM_DD_YYYY_txt_files, /MM_DD_YYYY_pic_files, and /MM_DD_YYYY_vid_files, where they store the outputted .txt, .png, and .mp4 files respectively for all post-processing scripts started on that date (MM/DD/YYYY). In addition, there will be prompts to specify a few analytical details, such as:

> What is your operating system?
> 
> What is your bin size?
> 
> What is your step size?
> 
> What do you want to analyze?
   
   
where it seeks the length of the bin for meshing the system (typically 5), the time step size to be analyzed (always 1 unless we care about velocity), and our post-processing method. Right before the prompt, the optional inputs are given for each, similar to the initial conditions in running simulations in the previous section. A second bash script will be submitted for each .gsd file in the current directory where this information is passed to. While running each python post-processing for each .gsd file on at a time on your local CPU, each will be submitted separately and run together on separate CPUs on Longleaf using SLURM. These post processing files typically output a series of .png files for each frame (optional input) and a .txt file (located in the /MM_DD_YYYY_txt_files_folder) compiling certain information at each time step. Once the python script finishes running, the second bash file will compile each series of .png files (located in the /MM_DD_YYYY_pic_files folder) into a .mp4 (located in the /MM_DD_YYYY_vid_files folder). The bash script will proceed to delete all .png files that were created by that post-processing script. Therefore, what we're left with will always be .mp4 files. or.txt files unless an error occurs. These .txt files are typically read into certain jupyter notebook scripts to clean, wrangle, and analyze them further (i.e. averaged over time and to identify trends over our input parameters).

### Downloading data from Longleaf

If you'd like to download a file from Longleaf to your local computer, you can input:
   
```
scp [ONYEN]@longleaf.unc.edu:/path/to/file ~/Desktop/
```
   
which will download a single file based on its path from Longleaf to your local Desktop. If you want to download a whole folder, you can modify it as below:
   
```
$ scp -r [ONYEN]@longleaf.unc.edu:/path/to/folder ~/Desktop/
```

where the -r signifies a recursive method since there are multiple files in the folder.
   
## Collaboration

To collaborate or ask questions, please contact me at: njlauers@live.unc.edu. If you wish to use any part of this package in your research, please supply my name in the acknowledgments or cite any of my publications.

   
