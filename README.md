# klotsa
***

## Introduction
***
Collection of software and packages for simulating systems of active and passive Brownian particles in HOOMD-Blue using molecular dynamics.

## Technologies
***
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

## Installation
***
It is highly recommended one install's both HOOMD-Blue and this github repository in their user's home directory due to the pathing in the analysis files. 

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
Configure HOOMD-Blue. When configuring locally, be sure `-DENABLE_CUDA=OFF` in the `cmake` tags. When configuring locally, you installed Open MPI, let `-DENABLE_CUDA=ON` in the `cmake` tags.
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

**For installing klotsa group github repository:**
```
$ cd ~
$ git clone https://github.com/njlauersdorf/klotsa.git
```

