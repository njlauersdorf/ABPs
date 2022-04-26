#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:50:02 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Read in .gsd file of particle positions
2.) Mesh the space
3.) Loop through tsteps and ...
3a.) Place all particles in appropriate mesh grid
3b.) Calculates location of steady state cluster's center of mass
3c.) Translate particles positions such that origin (0,0) is cluster's center of mass
3b.) Loop through all bins ...
3b.i.) Compute number density and average orientation per bin
3c.) Determine and label each phase (bulk dense phase, gas phase, and gas-bulk interface)
3d.) Calculate and output parameters to be used to calculate area fraction of each or all phase(s)
3e.) For frames with clusters, plot particle positions color-coded by phase it's a part of
4) Generate movie from frames
'''

import sys
import os

# Run locally
hoomdPath=str(sys.argv[10])
outPath2=str(sys.argv[11])
outPath=str(sys.argv[12])

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage

import numpy as np
import matplotlib

if hoomdPath == '/nas/longleaf/home/njlauers/hoomd-blue/build':
    matplotlib.use('Agg')
else:
    pass

import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
import matplotlib.ticker as ticker

from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit


# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''

# Define base file name for outputs
outF = inFile[:-4]

#Read input file
f = hoomd.open(name=inFile, mode='rb')

#Label simulation parameters from command line
peA = float(sys.argv[2])                        #Activity (Pe) for species A
peB = float(sys.argv[3])                        #Activity (Pe) for species B
parFrac_orig = float(sys.argv[4])               #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig

if parFrac==100.0:
    parFrac_orig=0.5
    parFrac=50.0

peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

eps = float(sys.argv[5])                        #Softness, coefficient of interparticle repulsion (epsilon)

#Set system area fraction (phi)
try:
    phi = float(sys.argv[6])
    intPhi = int(phi)
    phi /= 100.
except:
    phi = 0.6
    intPhi = 60

#Get simulation time step
try:
    dtau = float(sys.argv[7])
except:
    dtau = 0.000001

# Set some constants
r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

# Define Fourier series for fit
def fourier(x, *a):
    ret = a[1]
    for deg in range(1, int(len(a)/2)):
        ret += ((a[(deg*2)] * np.cos((deg) * ((x-a[0])*np.pi/180))) + (a[2*deg+1] * np.sin((deg) * ((x-a[0])*np.pi/180))))
    return ret

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * (x-f)) + bi * sin(i * (x-f))
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def avgCollisionForce(peNet):
    '''
    Purpose: Average compressive force experienced by a reference particle in the
    bulk dense phase due to neighboring active forces computed from the integral
    of possible orientations

    Inputs: Net activity of system

    Output: Average magnitude of compressive forces experienced by a bulk particle
    '''

    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)

    return (magnitude * peNet) / (np.pi)

def ljForce(r, eps, sigma=1.):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
    and separation distance of 2 particles to compute magnitude of lennard-jones force experienced
    by each

    Inputs:
        r: Separation distance in simulation units
        epsilon: magnitude of lennard-jones potential
        sigma: particle diameter (default=1.0)

    Output: lennard jones force (dU)
    '''

    #Dimensionless distance unit
    div = (sigma/r)

    dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
    return dU

# Lennard-Jones pressure
def ljPress(r, pe, eps, sigma=1.):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
    activity (pe), and separation distance (r) of 2 particles to compute pressure from
    avg compressive active forces from neighbors

    Inputs:
        r: Separation distance in simulation units
        epsilon: magnitude of lennard-jones potential
        pe: activity (peclet number)
        sigma: particle diameter (default=1.0)

    Output: Analytical virial pressure (see monodisperse paper for derivation)
    '''
    #Area fraction at HCP
    phiCP = np.pi / (2. * np.sqrt(3.))

    # LJ force
    ljF = avgCollisionForce(pe)

    return (2. *np.sqrt(3) * ljF / r)


# Calculate cluster radius
def conForRClust(pe, eps):
    '''
    Purpose: Compute analytical radius of the custer given activity and softness

    Inputs:
        pe: net activity (peclet number)
        eps: softness (magnitude of repulsive interparticle force)

    Output: cluster radius (simulation distance units)
    '''
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out

def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))

def symlog_arr(x):
    """ Returns the symmetric log10 value """
    out_arr = np.zeros(np.shape(x))
    for d in range(0, len(x)):
        for f in range(0, len(x)):
            if x[d][f]!=0:
                out_arr[d][f]=np.sign(x[d][f]) * np.log10(np.abs(x[d][f]))
    return out_arr
# Calculate dense phase area fraction from lattice spacing
def latToPhi(latIn):
    '''
    Purpose: Compute analytical area fraction of the dense phase given the lattice
    spacing.

    Inputs:
        latIn: lattice spacing

    Output: dense phase area fraction
    '''
    phiCP = np.pi / (2. * np.sqrt(3.))
    return phiCP / (latIn**2)


#Calculate gas phase area fraction
def compPhiG(pe, a, kap=4.5, sig=1.):
    '''
    Purpose: Compute analytical area fraction of the gas phase at steady state
    given activity and lattice spacing

    Inputs:
        pe: net activity (peclet number)
        a: lattice spacing
        kap: fitting parameter (default=4.5, shown by Redner)
        sig: particle diameter (default=1.0)

    Output: Area fraction of the gas phase at steady state
    '''
    num = 3. * (np.pi**2) * kap * sig
    den = 4. * pe * a
    return num / den

#Calculate analytical values
lat_theory = conForRClust(peNet, eps)
curPLJ = ljPress(lat_theory, peNet, eps)
phi_theory = latToPhi(lat_theory)
phi_g_theory = compPhiG(peNet, lat_theory)
steady_state_once = 'False'

def quatToAngle(quat):
    '''
    Purpose: Take quaternion orientation vector of particle as given by hoomd-blue
    simulations and output angle between [-pi, pi]

    Inputs: Quaternion orientation vector of particle

    Output: angle between [-pi, pi]
    '''

    r = quat[0]         #magnitude
    x = quat[1]         #x-direction
    y = quat[2]         #y-direction
    z = quat[3]         #z-direction
    rad = math.atan2(y, x)

    return rad

def computeTauLJ(epsilon):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force) and compute lennard-jones
    time unit of simulation

    Inputs: epsilon

    Output: lennard-jones time unit
    '''
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ

def getLat(peNet, eps):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force) and net activity to
    compute lattice spacing as derived analytically (force balance of repulsive LJ force
    and compressive active force)

    Inputs:
        peNet: net activity of system
        epsilon: magnitude of lennard-jones potential

    Output: average lattice spacing of system
    '''

    #If system is passive, output cut-off radius
    if peNet == 0:
        return 2.**(1./6.)
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

    #Loop through to find separation distance (r) where lennard-jones force (jForce)
    #approximately equals compressive active force (avgCollisionForce)
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(peNet):
            r -= j
        r += j

    return r
def computeFLJ(r, x1, y1, x2, y2, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (x2 - x1) / r
    fy = f * (y2 - y1) / r
    return fx, fy
#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)
tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size
n_len = 21
n_arr = np.linspace(0, n_len-1, n_len)      #Fourier modes
popt_sum = np.zeros(n_len)                  #Fourier Coefficients
num_frames = 0

#Import modules
import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster
import itertools

import numpy as np
import math
import random
from scipy import stats

#Set plotting parameters
matplotlib.rc('font', serif='Helvetica Neue')
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5

def computeTauPerTstep(epsilon, mindt=0.000001):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), and output the amount
    of Brownian time units per time step in LJ units

    Inputs:
        epsilon: magnitude of lennard-jones potential
        mindt: time step in LJ units (default=0.000001)

    Output: lennard jones force (dU)
    '''

    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

def roundUp(n, decimals=0):
    '''
    Purpose: Round up number of bins to account for floating point inaccuracy

    Inputs:
        n: number of bins along length of box
        decimals: exponent of multiplier for rounding (default=0)
    Output: number of bins along box length rounded up
    '''

    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def getNBins(length, minSz=(2**(1./6.))):
    '''
    Purpose: Given box size, return number of bins

    Inputs:
        length: length of box
        minSz: set minimum bin length to LJ cut-off distance
    Output: number of bins along box length rounded up
    '''

    initGuess = int(length) + 1
    nBins = initGuess
    # This loop only exits on function return
    while True:
        if length / nBins > minSz:
            return nBins
        else:
            nBins -= 1

#Set plot colors
fastCol = '#e31a1c'
slowCol = '#081d58'

#Open input simulation file
f = hoomd.open(name=inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

#Get particle number from initial frame
snap = f[0]
typ = snap.particles.typeid
partNum = len(typ)

#Set output file names
bin_width = float(sys.argv[8])
time_step = float(sys.argv[9])
outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile

outTxt_msd = 'msd_' + outfile + '.txt'

lat_theory_arr = np.array([])

#.txt file for saving overall phase composition data
g = open(outPath2+outTxt_msd, 'w+') # write file headings
g.write('tst'.center(15) + ' ' +\
                        'bin_size'.center(15) + ' ' +\
                        'clust_size'.center(15) + ' ' +\
                        'tot_msd'.center(15) + ' ' +\
                        'tot_msd_std'.center(15) + ' ' +\
                        'slow_msd'.center(15) + ' ' +\
                        'slow_msd_std'.center(15) + ' ' +\
                        'fast_msd'.center(15) + ' ' +\
                        'fast_msd_std'.center(15) + '\n')
g.close()

"""
def ljForce(r, eps, sigma=1.):
    div = (sigma/r)
    dU = (24. * eps / sigma) * ((2*(div**13)) - (div)**7)
    return dU

def avgCollisionForce(pe, power=1.):
    '''Computed from the integral of possible angles'''
    peCritical = 40.
    if pe < peCritical:
        pe = 0
    else:
        pe -= peCritical
    magnitude = 6.
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
#     return (magnitude * (pe**power)) / (np.pi)
#     return (pe * (1. + (8./(np.pi**2.))))
    coeff = 1.92#3.0#1.92#2.03#3.5#2.03
    #coeff= 0.4053
    return (pe * coeff)

def conForRClust(pe, eps):
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out
"""

#Bulk phase
fit_A = 0.03
fit_B = 1.3603
fit_C = 0.4684

#Gas phase
fit_A2 = 0.403
fit_B2 = 0.633
fit_C2 = 0.101

#Interface phase
fit_A3 = 0.011
fit_B3 = 1.106
fit_C3 = 0.49

if peA<=peB:
    if peA>=100:
        ss_chi_f = fit_A * ((peA/peB) ** (-fit_B)) + fit_C
        ss_chi_f_gas = fit_A2 * ((peA/peB) ** (fit_B2)) + fit_C2
        ss_chi_f_int = fit_A3 * ((peA/peB) ** (-fit_B3)) + fit_C3
        steady_state = 'True'
    else:
        steady_state = 'False'
else:
    if peB>=100:
        ss_chi_f = fit_A * ((peB/peA) ** (-fit_B)) + fit_C
        ss_chi_f_gas = fit_A2 * ((peB/peA) ** (fit_B2)) + fit_C2
        ss_chi_f_int = fit_A3 * ((peB/peA) ** (-fit_B3)) + fit_C3
        steady_state = 'True'
    else:
        steady_state = 'False'

#Calculate analytical values

if peA<=peB:
    peNet_gas = ss_chi_f_gas*peB + (1.0-ss_chi_f_gas)*peA
    peNet_int = ss_chi_f_int*peB + (1.0-ss_chi_f_int)*peA
else:
    peNet_gas = ss_chi_f_gas*peA + (1.0-ss_chi_f_gas)*peB
    peNet_int = ss_chi_f_int*peA + (1.0-ss_chi_f_int)*peB

lat_theory_int = conForRClust(peNet_int-50, eps)
phi_d = latToPhi(lat_theory_int)
phi_g = compPhiG(peNet_gas, lat_theory_int)
clust_theory = partNum * (((phi_g-phi)*phi_d)/(phi*(phi_g-phi_d)))

first_steady_state_frame = 'False'

bulk_msd_arr = np.array([])
slow_bulk_msd_arr = np.array([])
fast_bulk_msd_arr = np.array([])
bulk_msd_std_arr = np.array([])
slow_bulk_msd_std_arr = np.array([])
fast_bulk_msd_std_arr = np.array([])
ss_time_arr = np.array([])
clust_size_arr = np.array([])
with hoomd.open(name=inFile, mode='rb') as t:

    r = np.linspace(0.0,  5.0, 100)             # Define radius for x-axis of plot later

    start = int(0/time_step)#205                                             # first frame to process
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = int(dumps/time_step)#int(dumps/time_step)-1                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step

    # Get box dimensions
    box_data = snap.configuration.box
    l_box = box_data[0]                                     #box length
    h_box = l_box / 2.0                                     #half box length

    #2D binning of system
    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / NBins), 6)
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)

    time_arr=np.zeros(dumps)                                  #time step array

    for p in range(start, end):
        #if clust_size_arr[p] = np
        j=int(p*time_step)
        print('j')
        print(j)

        #Arrays of particle data
        snap_current = t[j]                                 #Take current frame

        pos_current = snap_current.particles.position               # position
        pos_current[:,-1] = 0.0                             # 2D system

        snap = t[j]                                 #Take current frame

        #Arrays of particle data
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0                             # 2D system
        xy = np.delete(pos, 2, 1)

        ori = snap.particles.orientation            #Orientation (quaternions)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]

        typ = snap.particles.typeid                 # Particle type
        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        print(tst)
        time_arr[j]=tst

        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes


        min_size=int(partNum/8)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size



        #If a single cluster is greater than minimum size, determine CoM of largest cluster
        if len(large_clust_ind_all[0])>0:

            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box

            com_tmp_posX_temp = query_points[0]
            com_tmp_posY_temp = query_points[1]
        else:

            com_tmp_posX = h_box
            com_tmp_posY = h_box

            com_tmp_posX_temp = 0
            com_tmp_posY_temp = 0
        if ((np.amax(clust_size)>=0.85*clust_theory) & (np.amax(clust_size)<=1.15*clust_theory) & (steady_state == 'True')) | (steady_state == 'False'):
            print('true!')
            print(steady_state)
            #shift reference frame to center of mass of cluster
            fsize=10
            pos[:,0]= pos[:,0]-com_tmp_posX_temp
            pos[:,1]= pos[:,1]-com_tmp_posY_temp

            #Ensure particles are within simulation box (periodic boundary conditions)
            for i in range(0, partNum):
                    if pos[i,0]>h_box:
                        pos[i,0]=pos[i,0]-l_box
                    elif pos[i,0]<-h_box:
                        pos[i,0]=pos[i,0]+l_box

                    if pos[i,1]>h_box:
                        pos[i,1]=pos[i,1]-l_box
                    elif pos[i,1]<-h_box:
                        pos[i,1]=pos[i,1]+l_box

            #Bin system to calculate orientation and alignment that will be used in vector plots
            NBins = getNBins(l_box, bin_width)
            sizeBin = roundUp(((l_box) / NBins), 6)

            # Initialize empty arrays
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]           #Binned IDs of particles
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]           #Binned types of particles
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]            #Bins specifying if particles occupy bin (1) or not (0)
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]

            pos_box_x_plot = [[0 for b in range(NBins)] for a in range(NBins)]
            pos_box_y_plot = [[0 for b in range(NBins)] for a in range(NBins)]

            p_all_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_all_y = [[0 for b in range(NBins)] for a in range(NBins)]


            p_plot_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_plot_y = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_num = [[0 for b in range(NBins)] for a in range(NBins)]

            align_tot_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot_y = [[0 for b in range(NBins)] for a in range(NBins)]

            num_dens3 = [[0 for b in range(NBins)] for a in range(NBins)]

            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
                #posParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]

            phaseBin = [[0 for b in range(NBins)] for a in range(NBins)]

            pos_box_x_new = [[0 for b in range(NBins)] for a in range(NBins)]
            pos_box_y_new = [[0 for b in range(NBins)] for a in range(NBins)]

            partTyp=np.zeros(partNum)
            partPhase=np.zeros(partNum)
            edgePhase=np.zeros(partNum)
            bulkPhase=np.zeros(partNum)

            #Calculate binned alignment/number density for plots at end
            for k in range(0, len(ids)):

                    # Convert position to be > 0 to place in list mesh
                    tmp_posX = pos[k][0] + h_box
                    tmp_posY = pos[k][1] + h_box

                    x_ind = int(tmp_posX / sizeBin)
                    y_ind = int(tmp_posY / sizeBin)

                    # Append all particles to appropriate bin
                    binParts[x_ind][y_ind].append(k)
                    typParts[x_ind][y_ind].append(typ[k])

                    if clust_size[ids[k]] >= min_size:
                        occParts[x_ind][y_ind] = 1

            pos_box_start=np.array([])
            for ix in range(0, len(occParts)):
                    pos_box_start = np.append(pos_box_start, ix*sizeBin)
                    for iy in range(0, len(occParts)):

                        #Label position of midpoint of bin
                        pos_box_x_plot[ix][iy] = ((ix+0.5)*sizeBin)
                        pos_box_y_plot[ix][iy] = ((iy+0.5)*sizeBin)

                        #Label position of lower left vertex of bin
                        pos_box_x_new[ix][iy] = ((ix)*sizeBin)
                        pos_box_y_new[ix][iy] = ((iy)*sizeBin)

                        #If particles in bin, loop through particles
                        if len(binParts[ix][iy]) != 0:
                            for h in range(0, len(binParts[ix][iy])):

                                #Calculate x,y position of particle
                                x_pos=pos[binParts[ix][iy]][h][0]+h_box

                                y_pos=pos[binParts[ix][iy]][h][1]+h_box

                                #Calculate x-distance from CoM
                                difx=x_pos-com_tmp_posX

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        difx += l_box
                                    else:
                                        difx -= l_box

                                #Calculate y-distance from CoM
                                dify=y_pos-com_tmp_posY

                                #Enforce periodic boundary conditions
                                dify_abs = np.abs(dify)
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        dify += l_box
                                    else:
                                        dify -= l_box

                                #Calculate total distance from CoM
                                difr=(difx**2+dify**2)**0.5

                                #Calculate x and y orientation of active force
                                px = np.sin(ang[binParts[ix][iy][h]])
                                py = -np.cos(ang[binParts[ix][iy][h]])


                                #Calculate alignment towards CoM
                                r_dot_p = (-difx * px) + (-dify * py)

                                #Sum x,y orientation over each bin
                                p_all_x[ix][iy]+=px
                                p_all_y[ix][iy]+=py

                            #Calculate number density per bin
                            num_dens3[ix][iy] = (len(binParts[ix][iy])/(sizeBin**2))*(math.pi/4)

                            #Calculate average orientation per bin
                            p_plot_x[ix][iy] = p_all_x[ix][iy]/len(binParts[ix][iy])
                            p_plot_y[ix][iy] = p_all_y[ix][iy]/len(binParts[ix][iy])


            #Define colors for plots
            yellow = ("#fdfd96")
            green = ("#77dd77")
            red = ("#ff6961")
            purple = ("#cab2d6")

            #Re-create bins for true measurement (txt file output)
            NBins = getNBins(l_box, bin_width)
            sizeBin = roundUp(((l_box) / NBins), 6)

            #Initialize arrays to save to

            pos_box_x = [[0 for b in range(NBins)] for a in range(NBins)]
            pos_box_y = [[0 for b in range(NBins)] for a in range(NBins)]

            p_all_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_all_y = [[0 for b in range(NBins)] for a in range(NBins)]

            p_all_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            p_all_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            p_all_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            p_all_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            p_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]

            p_avg_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            p_avg_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            p_avg_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            p_avg_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            v_all_x = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_y = [[0 for b in range(NBins)] for a in range(NBins)]

            v_all_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            v_all_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]

            align_norm_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_norm_y = [[0 for b in range(NBins)] for a in range(NBins)]

            align_norm_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            align_norm_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            align_norm_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            align_norm_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_xDif = [[0 for b in range(NBins)] for a in range(NBins)]
            align_avg_yDif = [[0 for b in range(NBins)] for a in range(NBins)]

            align_avg_num = [[0 for b in range(NBins)] for a in range(NBins)]

            align_tot_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot_y = [[0 for b in range(NBins)] for a in range(NBins)]

            align_tot_xA = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot_yA = [[0 for b in range(NBins)] for a in range(NBins)]

            align_tot_xB = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot_yB = [[0 for b in range(NBins)] for a in range(NBins)]

            num_dens3 = [[0 for b in range(NBins)] for a in range(NBins)]
            num_densDif = [[0 for b in range(NBins)] for a in range(NBins)]
            num_dens3A = [[0 for b in range(NBins)] for a in range(NBins)]
            num_dens3B = [[0 for b in range(NBins)] for a in range(NBins)]

            fa_all_tot = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_all_x_tot = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_all_y_tot = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_fast_tot = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_slow_tot = [[0 for b in range(NBins)] for a in range(NBins)]

            fa_all_num = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_fast_num = [[0 for b in range(NBins)] for a in range(NBins)]
            fa_slow_num = [[0 for b in range(NBins)] for a in range(NBins)]

            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            Binpe = [[0 for b in range(NBins)] for a in range(NBins)]

            phaseBin = [[0 for b in range(NBins)] for a in range(NBins)]            #Label phase of each bin

            new_green = '#39FF14'
            new_brown = '#b15928'

            pos_box_x_new = [[0 for b in range(NBins)] for a in range(NBins)]
            pos_box_y_new = [[0 for b in range(NBins)] for a in range(NBins)]

            partTyp=np.zeros(partNum)
            partPhase=np.zeros(partNum)
            extedgePhase=np.zeros(partNum)
            intedgePhase=np.zeros(partNum)

            #Bin particles
            for k in range(0, len(ids)):

                    # Convert position to be > 0 to place in list mesh
                    tmp_posX = pos[k][0] + h_box
                    tmp_posY = pos[k][1] + h_box
                    x_ind = int(tmp_posX / sizeBin)
                    y_ind = int(tmp_posY / sizeBin)

                    # Append all particles to appropriate bin
                    binParts[x_ind][y_ind].append(k)
                    typParts[x_ind][y_ind].append(typ[k])

                    #Label if bin is part of largest cluster
                    if clust_size[ids[k]] >= min_size:
                        occParts[x_ind][y_ind] = 1

            pos_box_start=np.array([])

            #Calculate alignment/number density to be used for determining interface

            #Loop over system bins
            for ix in range(0, len(occParts)):
                    pos_box_start = np.append(pos_box_start, ix*sizeBin)
                    for iy in range(0, len(occParts)):
                        typ0_temp=0
                        typ1_temp=0

                        #Calculate center of bin (for plotting)
                        pos_box_x[ix][iy] = ((ix+0.5)*sizeBin)
                        pos_box_y[ix][iy] = ((iy+0.5)*sizeBin)

                        #Calculate location of bin (bottom left corner) for calculations
                        pos_box_x_new[ix][iy] = ((ix)*sizeBin)
                        pos_box_y_new[ix][iy] = ((iy)*sizeBin)

                        #If particles in bin, proceed
                        if len(binParts[ix][iy]) != 0:

                            #Loop over particles per bin
                            for h in range(0, len(binParts[ix][iy])):

                                #(x,y) position of particle
                                x_pos=pos[binParts[ix][iy]][h][0]+h_box
                                y_pos=pos[binParts[ix][iy]][h][1]+h_box

                                #x-distance of particle from CoM
                                difx=x_pos-com_tmp_posX
                                difx_abs = np.abs(difx)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        difx += l_box
                                    else:
                                        difx -= l_box

                                #y-distance of particle from CoM
                                dify=y_pos-com_tmp_posY
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        dify += l_box
                                    else:
                                        dify -= l_box

                                #Separation distance from CoM
                                difr=(difx**2+dify**2)**0.5

                                #x,y particle orientation
                                px = np.sin(ang[binParts[ix][iy][h]])
                                py = -np.cos(ang[binParts[ix][iy][h]])

                                #Alignment towards CoM
                                r_dot_p = (-difx * px) + (-dify * py)

                                #Summed orientation of particles per bin
                                p_all_x[ix][iy]+=px
                                p_all_y[ix][iy]+=py

                                #Perform measurements for type A particles only
                                if typ[binParts[ix][iy][h]]==0:
                                    typ0_temp +=1               #Number of type A particles per bin
                                    p_all_xA[ix][iy]+=px        #Summed x-orientation of type B particles
                                    p_all_yA[ix][iy]+=py        #Summed y-orientation of type B particles

                                #Perform measurements for type B particles only
                                elif typ[binParts[ix][iy][h]]==1:
                                    typ1_temp +=1               #Number of type B particles per bin
                                    p_all_xB[ix][iy]+=px        #Summed x-orientation of type B particles
                                    p_all_yB[ix][iy]+=py        #Summed y-orientation of type B particles

                            #number density of bin
                            num_dens3[ix][iy] = (len(binParts[ix][iy])/(sizeBin**2))*(math.pi/4)        #Total number density
                            num_dens3A[ix][iy] = (typ0_temp/(sizeBin**2))*(math.pi/4)                   #Number density of type A particles
                            num_dens3B[ix][iy] = (typ1_temp/(sizeBin**2))*(math.pi/4)                   #Number density of type B particles

                            if peB >= peA:
                                num_densDif[ix][iy]=num_dens3B[ix][iy]-num_dens3A[ix][iy]                   #Difference in number density
                            else:
                                num_densDif[ix][iy]=num_dens3A[ix][iy]-num_dens3B[ix][iy]

                            #average x,y orientation per bin
                            p_avg_x[ix][iy] = p_all_x[ix][iy]/len(binParts[ix][iy])
                            p_avg_y[ix][iy] = p_all_y[ix][iy]/len(binParts[ix][iy])


                            #average x,y orientation per bin for A type particles
                            if typ0_temp>0:
                                p_avg_xA[ix][iy] = p_all_xA[ix][iy]/typ0_temp
                                p_avg_yA[ix][iy] = p_all_yA[ix][iy]/typ0_temp

                            else:
                                p_avg_xA[ix][iy] = 0.0
                                p_avg_yA[ix][iy] = 0.0


                            #average x,y orientation per bin for B type particles
                            if typ1_temp>0:
                                p_avg_xB[ix][iy] = p_all_xB[ix][iy]/typ1_temp
                                p_avg_yB[ix][iy] = p_all_yB[ix][iy]/typ1_temp

                            else:
                                p_avg_xB[ix][iy] = 0.0
                                p_avg_yB[ix][iy] = 0.0


            # Search 2 bins around each bin to average alignment (reduce noise)
            for ix in range(0, NBins):

                    #Based on x-index (ix), find neighboring x-indices to loop through

                    if (ix + 2) == NBins:
                        lookx = [ix-1, ix-1, ix, ix+1, 0]
                    elif (ix + 1) == NBins:
                        lookx = [ix-2, ix-1, ix, 0, 1]
                    elif ix==0:
                        lookx=[NBins-2, NBins-1, ix, ix+1, ix+2]
                    elif ix==1:
                        lookx=[NBins-1, ix-1, ix, ix+1, ix+2]
                    else:
                        lookx = [ix-2, ix-1, ix, ix+1, ix+2]

                    for iy in range(0, NBins):

                        #Based on y-index (iy), find neighboring y-indices to loop through

                        if (iy + 2) == NBins:
                            looky = [iy-1, iy-1, iy, iy+1, 0]
                        elif (iy + 1) == NBins:
                            looky = [iy-2, iy-1, iy, 0, 1]
                        elif iy==0:
                            looky=[NBins-2, NBins-1, iy, iy+1, iy+2]
                        elif iy==1:
                            looky=[NBins-1, iy-1, iy, iy+1, iy+2]
                        else:
                            looky = [iy-2, iy-1, iy, iy+1, iy+2]


                        # Loop through surrounding x-index
                        for indx in lookx:

                            # Loop through surrounding y-index
                            for indy in looky:

                                        #Summed average orientation of surrounding bins
                                        align_tot_x[ix][iy] += p_avg_x[indx][indy]
                                        align_tot_y[ix][iy] += p_avg_y[indx][indy]

                                        #Number of terms summed
                                        align_avg_num[ix][iy] += 1

                                        #Summed average orientation of surrounding bins for type A particles
                                        align_tot_xA[ix][iy] += p_avg_xA[indx][indy]
                                        align_tot_yA[ix][iy] += p_avg_yA[indx][indy]

                                        #Summed average orientation of surrounding bins for type B particles
                                        align_tot_xB[ix][iy] += p_avg_xB[indx][indy]
                                        align_tot_yB[ix][iy] += p_avg_yB[indx][indy]

                        #If particles in bin, continue...
                        if align_avg_num[ix][iy]>0:

                            #Average x,y orientation of particles per bin
                            align_avg_x[ix][iy]=align_tot_x[ix][iy]/align_avg_num[ix][iy]
                            align_avg_y[ix][iy] = align_tot_y[ix][iy]/align_avg_num[ix][iy]

                            #Average x,y orientation of type A particles per bin
                            align_avg_xA[ix][iy]=align_tot_xA[ix][iy]/align_avg_num[ix][iy]
                            align_avg_yA[ix][iy] = align_tot_yA[ix][iy]/align_avg_num[ix][iy]

                            #Average x,y orientation of type B particles per bin
                            align_avg_xB[ix][iy]=align_tot_xB[ix][iy]/align_avg_num[ix][iy]
                            align_avg_yB[ix][iy] = align_tot_yB[ix][iy]/align_avg_num[ix][iy]

                        #Otherwise, set each array value to zero
                        else:
                            align_avg_x[ix][iy]=0
                            align_avg_y[ix][iy]=0

                            align_avg_xA[ix][iy]=0
                            align_avg_yA[ix][iy]=0

                            align_avg_xB[ix][iy]=0
                            align_avg_yB[ix][iy]=0



            align_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
            align_combinedA = np.zeros((len(v_avg_x), len(v_avg_y),2))
            align_combinedB = np.zeros((len(v_avg_x), len(v_avg_y),2))
            align_combinedDif = np.zeros((len(v_avg_x), len(v_avg_y),2))

            pos_box_combined_align = np.zeros((len(v_avg_x), len(v_avg_y),2))

            for ix in range(0, len(align_avg_x)):
                for iy in range(0, len(align_avg_y)):

                        align_combined[ix][iy][0]=align_avg_x[ix][iy]
                        align_combined[ix][iy][1]=align_avg_y[ix][iy]

                        align_combinedA[ix][iy][0]=align_avg_xA[ix][iy]
                        align_combinedA[ix][iy][1]=align_avg_yA[ix][iy]

                        align_combinedB[ix][iy][0]=align_avg_xB[ix][iy]
                        align_combinedB[ix][iy][1]=align_avg_yB[ix][iy]

                        align_combinedDif[ix][iy][0]=align_avg_xB[ix][iy] - align_avg_xA[ix][iy]
                        align_combinedDif[ix][iy][1]=align_avg_yB[ix][iy] - align_avg_yA[ix][iy]

                        pos_box_combined_align[ix][iy][0]=pos_box_x[ix][iy]
                        pos_box_combined_align[ix][iy][1]=pos_box_y[ix][iy]

            alignx_grad = np.gradient(align_combined, axis=0)
            aligny_grad = np.gradient(align_combined, axis=1)

            alignx_gradA = np.gradient(align_combinedA, axis=0)
            aligny_gradA = np.gradient(align_combinedA, axis=1)

            alignx_gradB = np.gradient(align_combinedB, axis=0)
            aligny_gradB = np.gradient(align_combinedB, axis=1)

            alignx_gradDif = np.gradient(align_combinedDif, axis=0)
            aligny_gradDif = np.gradient(align_combinedDif, axis=1)

            num_densx_grad = np.gradient(num_dens3, axis=0)
            num_densy_grad = np.gradient(num_dens3, axis=1)

            align_gradx_x = alignx_grad[:,:,0]
            align_gradx_y = alignx_grad[:,:,1]
            align_grady_x = aligny_grad[:,:,0]
            align_grady_y = aligny_grad[:,:,1]

            align_gradx_xA = alignx_gradA[:,:,0]
            align_gradx_yA = alignx_gradA[:,:,1]
            align_grady_xA = aligny_gradA[:,:,0]
            align_grady_yA = aligny_gradA[:,:,1]

            align_gradx_xB = alignx_gradB[:,:,0]
            align_gradx_yB = alignx_gradB[:,:,1]
            align_grady_xB = aligny_gradB[:,:,0]
            align_grady_yB = aligny_gradB[:,:,1]

            align_gradx_xDif = alignx_gradDif[:,:,0]
            align_gradx_yDif = alignx_gradDif[:,:,1]
            align_grady_xDif = aligny_gradDif[:,:,0]
            align_grady_yDif = aligny_gradDif[:,:,1]

            div_align = align_gradx_x + align_grady_y
            curl_align = -align_grady_x + align_gradx_y

            div_alignA = align_gradx_xA + align_grady_yA
            curl_alignA = -align_grady_xA + align_gradx_yA

            div_alignB = align_gradx_xB + align_grady_yB
            curl_alignB = -align_grady_xB + align_gradx_yB

            div_alignDif = align_gradx_xDif + align_grady_yDif
            curl_alignDif = -align_grady_xDif + align_gradx_yDif

            div_num_dens = num_densx_grad + num_densy_grad
            div_num_dens2 = np.gradient(num_dens3)

            #Calculate density limits for phases (gas, interface, bulk)
            vmax_eps = phi_theory * 1.4
            phi_dense_theory_max=phi_theory*1.3
            phi_dense_theory_min=phi_theory*0.95

            phi_gas_theory_max= phi_g_theory*4.0
            phi_gas_theory_min=0.0

            #Time frame for plots
            pad = str(j).zfill(4)

            #Calculate average activity per bin
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        pe_sum=0
                        if len(binParts[ix][iy])>0:
                            for h in range(0, len(binParts[ix][iy])):
                                if typ[binParts[ix][iy][h]]==0:
                                    pe_sum += peA
                                else:
                                    pe_sum += peB
                            Binpe[ix][iy] = pe_sum/len(binParts[ix][iy])




            #Initialize arrays
            press_int = [[0 for b in range(NBins)] for a in range(NBins)]
            align_mag = [[0 for b in range(NBins)] for a in range(NBins)]
            align_magA = [[0 for b in range(NBins)] for a in range(NBins)]
            align_magB = [[0 for b in range(NBins)] for a in range(NBins)]
            align_magDif = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_mag = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magA = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magB = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magDif = [[0 for b in range(NBins)] for a in range(NBins)]
            press_bin = [[0 for b in range(NBins)] for a in range(NBins)]
            press_binA = [[0 for b in range(NBins)] for a in range(NBins)]
            press_binB = [[0 for b in range(NBins)] for a in range(NBins)]
            press_binDif = [[0 for b in range(NBins)] for a in range(NBins)]
            #Calculate weighted alignment/num density product for determining highly aligned interface
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        press_int[ix][iy] = num_dens3[ix][iy]*(align_avg_x[ix][iy]**2+align_avg_y[ix][iy]**2)**0.5
                        align_mag[ix][iy] = (align_avg_x[ix][iy]**2+align_avg_y[ix][iy]**2)**0.5
                        align_magA[ix][iy] = (align_avg_xA[ix][iy]**2+align_avg_yA[ix][iy]**2)**0.5
                        align_magB[ix][iy] = (align_avg_xB[ix][iy]**2+align_avg_yB[ix][iy]**2)**0.5
                        align_magDif[ix][iy] = (align_magB[ix][iy]-align_magA[ix][iy])#(align_avg_xDif[ix][iy]**2+align_avg_yDif[ix][iy]**2)**0.5
                        if j>(start*time_step):
                            vel_mag[ix][iy] = ((v_avg_x[ix][iy]**2+v_avg_y[ix][iy]**2)**0.5)#/peB
                            vel_magA[ix][iy] = ((v_avg_xA[ix][iy]**2+v_avg_yA[ix][iy]**2)**0.5)#/peA
                            vel_magB[ix][iy] = ((v_avg_xB[ix][iy]**2+v_avg_yB[ix][iy]**2)**0.5)#/peB
                            vel_magDif[ix][iy] = (vel_magB[ix][iy]*peB-vel_magA[ix][iy]*peA)#(align_avg_xDif[ix][iy]**2+align_avg_yDif[ix][iy]**2)**0.5
                        press_bin[ix][iy] = num_dens3[ix][iy]*align_mag[ix][iy]#*Binpe[ix][iy]
                        press_binA[ix][iy] = num_dens3A[ix][iy]*align_magA[ix][iy]#*peA
                        press_binB[ix][iy] = num_dens3B[ix][iy]*align_magB[ix][iy]#*peB
                        press_binDif[ix][iy] = (num_dens3B[ix][iy]*align_magB[ix][iy])-(num_dens3A[ix][iy]*align_magA[ix][iy])

                        if align_mag[ix][iy]>0:
                            align_norm_x[ix][iy] = align_avg_x[ix][iy] / align_mag[ix][iy]
                            align_norm_y[ix][iy] = align_avg_y[ix][iy] / align_mag[ix][iy]

                        if align_magA[ix][iy]>0:
                            align_norm_xA[ix][iy] = align_avg_xA[ix][iy] / align_magA[ix][iy]
                            align_norm_yA[ix][iy] = align_avg_yA[ix][iy] / align_magA[ix][iy]

                        if align_magB[ix][iy]>0:
                            align_norm_xB[ix][iy] = align_avg_xB[ix][iy] / align_magB[ix][iy]
                            align_norm_yB[ix][iy] = align_avg_yB[ix][iy] / align_magB[ix][iy]

            #Gradient of orientation
            aligngrad = np.gradient(align_mag)

            #Gradient of number density
            numdensgrad = np.gradient(num_dens3)
            numdensgradA = np.gradient(num_dens3A)
            numdensgradB = np.gradient(num_dens3B)
            #Gradient of pressure
            pgrad = np.gradient(press_int)

            #Product of gradients of number density and orientation
            comb_grad = np.multiply(numdensgrad, aligngrad)

            #Magnitude of pressure gradient
            fulgrad = np.sqrt(pgrad[0]**2 + pgrad[1]**2)

            #Magnitude of pressure gradient
            numdensegrad2 = np.sqrt(numdensgrad[0]**2 + numdensgrad[1]**2)
            numdensegrad2A = np.sqrt(numdensgradA[0]**2 + numdensgradA[1]**2)
            numdensegrad2B = np.sqrt(numdensgradB[0]**2 + numdensgradB[1]**2)

            #Magnitude of number_density * orientation gradient
            fulgrad2 = np.sqrt(comb_grad[0]**2 + comb_grad[1]**2)

            #Weighted criterion for determining interface (more weighted to alignment than number density)
            criterion = align_mag*fulgrad

            #Ranges for determining interface
            fulgrad_min = 0.05*np.max(criterion)
            fulgrad_max = np.max(criterion)

            #Initialize count of bins for each phase
            gasBin_num=0
            edgeBin_num=0
            bulkBin_num=0

            #Label phase of bin per above criterion in number density and alignment
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                            #Criterion for interface or gas
                            if (criterion[ix][iy]<fulgrad_min) & (num_dens3[ix][iy] < phi_dense_theory_min):

                                #Criterion for gas
                                if num_dens3[ix][iy]<phi_gas_theory_max:
                                    phaseBin[ix][iy]=2
                                    gasBin_num+=1

                                #Criterion for interface
                                else:
                                    phaseBin[ix][iy]=1
                                    edgeBin_num+=1

                            #Criterion for interface
                            elif (criterion[ix][iy]>fulgrad_min) | (num_dens3[ix][iy] < phi_dense_theory_min):
                                phaseBin[ix][iy]=1
                                edgeBin_num+=1

                            #Otherwise, label it as bulk
                            else:
                                phaseBin[ix][iy]=0
                                bulkBin_num+=1

                            #Label each particle with same phase
                            for h in range(0, len(binParts[ix][iy])):
                                partPhase[binParts[ix][iy][h]]=phaseBin[ix][iy]
                                partTyp[binParts[ix][iy][h]]=typ[binParts[ix][iy][h]]
            # Blur interface (twice/two loops) identification to remove noise.
            #Check neighbors to be sure correctly identified phase. If not, average
            #with neighbors. If so, leave.



            for f in range(0,2):

                for ix in range(0, len(occParts)):

                        #Identify neighboring bin indices in x-direction
                        if (ix + 1) == NBins:
                            lookx = [ix-1, ix, 0]
                        elif ix==0:
                            lookx=[NBins-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, ix+1]

                        # Loop through y index of mesh
                        for iy in range(0, NBins):

                            #Identify neighboring bin indices in y-direction
                            if (iy + 1) == NBins:
                                looky = [iy-1, iy, 0]
                            elif iy==0:
                                looky=[NBins-1, iy, iy+1]
                            else:
                                looky = [iy-1, iy, iy+1]

                            #Count phases of surrounding bins
                            gas_bin=0
                            edge_bin=0
                            bulk_bin=0
                            ref_phase = phaseBin[ix][iy]            #reference bin phase

                            #Loop through surrounding x-index
                            for indx in lookx:

                                # Loop through surrounding y-index
                                for indy in looky:

                                    #If not reference bin, continue
                                    if (indx!=ix) or (indy!=iy):

                                        #If bulk, label it
                                        if phaseBin[indx][indy]==0:
                                            bulk_bin+=1

                                        #If interface, label it
                                        elif phaseBin[indx][indy]==1:
                                            edge_bin+=1

                                        #If gas, label it
                                        else:
                                            gas_bin+=1
                            #If reference bin is a gas bin, continue
                            if ref_phase==2:

                                #If 2 or fewer surrounding gas bins, change it to
                                #edge or bulk (whichever is more abundant)
                                if gas_bin<=2:
                                    if edge_bin>=bulk_bin:
                                        phaseBin[ix][iy]=1
                                    else:
                                        phaseBin[ix][iy]=0

                            #If reference bin is a bulk bin, continue
                            elif ref_phase==0:

                                #If 2 or fewer surrounding bulk bins, change it to
                                #edge or gas (whichever is more abundant)
                                if bulk_bin<=2:
                                    if edge_bin>=gas_bin:
                                        phaseBin[ix][iy]=1
                                    else:
                                        phaseBin[ix][iy]=2

                            #If reference bin is a edge bin, continue
                            elif ref_phase==1:

                                #If 2 or fewer surrounding edge bins, change it to
                                #bulk or gas (whichever is more abundant)
                                if edge_bin<=2:
                                    if bulk_bin>=gas_bin:
                                        phaseBin[ix][iy]=0
                                    else:
                                        phaseBin[ix][iy]=2

            #Label individual particle phases from identified bin phases
            edge_num_bin=0
            bulk_num_bin=0
            gas_num_bin=0
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if phaseBin[ix][iy]==1:
                            edge_num_bin+=1
                        elif phaseBin[ix][iy]==0:
                            bulk_num_bin+=1
                        elif phaseBin[ix][iy]==2:
                            gas_num_bin+=1
                        for h in range(0, len(binParts[ix][iy])):
                            partPhase[binParts[ix][iy][h]]=phaseBin[ix][iy]
                            partTyp[binParts[ix][iy][h]]=typ[binParts[ix][iy][h]]






            edge_id=np.zeros((len(occParts), len(occParts)), dtype=int)            #Label separate interfaces
            ext_edge_id=np.zeros((len(occParts), len(occParts)), dtype=int)        #Label exterior edges of interfaces
            int_edge_id=np.zeros((len(occParts), len(occParts)), dtype=int)        #Label interior edges of interfaces

            #initiate ix, iy bin id's to while-loop over


            rerun_edge_num_bin=0


            com_x_ind = int(h_box / sizeBin)

            com_y_ind = int(h_box / sizeBin)

            bulk_id2=np.zeros((len(occParts), len(occParts)), dtype=int)            #Label separate interfaces

            rerun_bulk_num_bin=0

            if phaseBin[com_x_ind][com_y_ind]==0:
                com_bulk_indx = com_x_ind
                com_bulk_indy = com_y_ind
            elif len(np.where(partPhase==0)[0])>0:
                shortest_r = 10000
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if phaseBin[ix][iy]==0:


                            difx = (ix * sizeBin - h_box)

                            #Enforce periodic boundary conditions
                            difx_abs = np.abs(difx)
                            if difx_abs>=h_box:
                                if difx < -h_box:
                                    difx += l_box
                                else:
                                    difx -= l_box

                            dify = (iy * sizeBin - h_box)

                            #Enforce periodic boundary conditions
                            dify_abs = np.abs(dify)
                            if dify_abs>=h_box:
                                if dify < -h_box:
                                    dify += l_box
                                else:
                                    dify -= l_box

                            r_dist = (difx**2 + dify**2)**0.5
                            if r_dist < shortest_r:
                                shortest_r = r_dist
                                com_bulk_indx = ix
                                com_bulk_indy = iy
            else:
                com_bulk_indx = com_x_ind
                com_bulk_indy = com_y_ind
            ix = 0
            iy = 0
            end_test2=0
            while rerun_bulk_num_bin!=bulk_num_bin:
                    #If bin is an interface, continue
                    if phaseBin[ix][iy]==0:

                            #If bin hadn't been assigned an interface id yet, continue
                        if bulk_id2[ix][iy]==0:

                                    end_test2+=1         #Increase interface index

                                    #Append ID of bulk ID
                                    bulk_id_list=[]
                                    bulk_id_list.append([ix,iy])


                                    single_num_bin=0

                                    #Count surrounding bin phases
                                    gas_count=0
                                    bulk_count=0

                                    #loop over identified interface bins
                                    for ix2,iy2 in bulk_id_list:

                                            #identify neighboring bins
                                            if (ix2 + 1) == NBins:
                                                lookx = [ix2-1, ix2, 0]
                                            elif ix2==0:
                                                lookx=[NBins-1, ix2, ix2+1]
                                            else:
                                                lookx = [ix2-1, ix2, ix2+1]
                                            if (iy2 + 1) == NBins:
                                                    looky = [iy2-1, iy2, 0]
                                            elif iy2==0:
                                                    looky=[NBins-1, iy2, iy2+1]
                                            else:
                                                    looky = [iy2-1, iy2, iy2+1]

                                            #loop over surrounding x-index bins
                                            for indx in lookx:
                                            # Loop through surrounding y-index bins
                                                for indy in looky:

                                                    #If bin is a bulk, continue
                                                    if phaseBin[indx][indy]==0:

                                                        #If bin wasn't assigned an interface id, continue
                                                        if bulk_id2[indx][indy]==0:

                                                            #append ids to looped list
                                                            bulk_id_list.append([indx, indy])
                                                            rerun_bulk_num_bin+=1

                                                            #Append interface id
                                                            bulk_id2[indx][indy]=end_test2
                                                            single_num_bin+=1

                            #If bin has been identified as an interface, look at different reference bin
                        else:
                                if (ix==(NBins-1)) & (iy==(NBins-1)):
                                    break
                                if ix!=(NBins-1):
                                    ix+=1
                                else:
                                    ix=0
                                    iy+=1
                    #If bin is not an interface, go to different reference bin
                    else:
                        if (ix==(NBins-1)) & (iy==(NBins-1)):
                            break
                        if ix!=(NBins-1):
                            ix+=1
                        else:
                            ix=0
                            iy+=1

            ix=0
            iy=0
            end_test=0
            big_bulk_id = bulk_id2[com_bulk_indx][com_bulk_indy]

            possible_interface_ids = []
            # Individually label each interface until all edge bins identified using flood fill algorithm
            while rerun_edge_num_bin!=edge_num_bin:

                    #If bin is an interface, continue
                    if phaseBin[ix][iy]==1:

                            #If bin hadn't been assigned an interface id yet, continue
                            if edge_id[ix][iy]==0:

                                    end_test+=1         #Increase interface index

                                    #Append ID of interface ID
                                    edge_id_list=[]
                                    edge_id_list.append([ix,iy])


                                    single_num_bin=0

                                    #Count surrounding bin phases
                                    gas_count=0
                                    bulk_count=0

                                    #loop over identified interface bins
                                    for ix2,iy2 in edge_id_list:

                                            #identify neighboring bins
                                            if (ix2 + 1) == NBins:
                                                lookx = [ix2-1, ix2, 0]
                                            elif ix2==0:
                                                lookx=[NBins-1, ix2, ix2+1]
                                            else:
                                                lookx = [ix2-1, ix2, ix2+1]
                                            if (iy2 + 1) == NBins:
                                                    looky = [iy2-1, iy2, 0]
                                            elif iy2==0:
                                                    looky=[NBins-1, iy2, iy2+1]
                                            else:
                                                    looky = [iy2-1, iy2, iy2+1]

                                            #loop over surrounding x-index bins
                                            for indx in lookx:
                                            # Loop through surrounding y-index bins
                                                for indy in looky:

                                                    #If bin is an interface, continue
                                                    if phaseBin[indx][indy]==1:

                                                        #If bin wasn't assigned an interface id, continue
                                                        if edge_id[indx][indy]==0:

                                                            #append ids to looped list
                                                            edge_id_list.append([indx, indy])
                                                            rerun_edge_num_bin+=1

                                                            #Append interface id
                                                            edge_id[indx][indy]=end_test
                                                            single_num_bin+=1

                                                    #If bin is a gas, count it
                                                    elif phaseBin[indx][indy]==2:

                                                        gas_count+=1

                                                    #else bin is counted as bulk
                                                    else:
                                                        if bulk_id2[indx][indy]==big_bulk_id:
                                                            if end_test not in possible_interface_ids:
                                                                possible_interface_ids.append(end_test)
                                                        bulk_count+=1

                                    #If fewer than or equal to 4 neighboring interfaces, re-label phase as bulk or gas
                                    if single_num_bin<=4:

                                        #If more neighboring gas bins, reference bin is truly a gas bin
                                        if gas_count>bulk_count:
                                            for ix3 in range(0, len(occParts)):
                                                for iy3 in range(0, len(occParts)):
                                                    if edge_id[ix3][iy3]==end_test:
                                                        edge_id[ix3][iy3]=0
                                                        phaseBin[ix3][iy3]=2


                                        #Else if more neighboring bulk bins, reference bin is truly a bulk bin
                                        else:
                                            for ix3 in range(0, len(occParts)):
                                                for iy3 in range(0, len(occParts)):
                                                    if edge_id[ix3][iy3]==end_test:
                                                        edge_id[ix3][iy3]=0
                                                        phaseBin[ix3][iy3]=0


                            #If bin has been identified as an interface, look at different reference bin
                            else:
                                if (ix==(NBins-1)) & (iy==(NBins-1)):
                                    break
                                if ix!=(NBins-1):
                                    ix+=1
                                else:
                                    ix=0
                                    iy+=1
                    #If bin is not an interface, go to different reference bin
                    else:
                        if (ix==(NBins-1)) & (iy==(NBins-1)):
                            break
                        if ix!=(NBins-1):
                            ix+=1
                        else:
                            ix=0
                            iy+=1


            #Label which interface each particle belongs to
            for ix in range(0, len(edge_id)):
                    for iy in range(0, len(edge_id)):
                        if edge_id[ix][iy] != 0:
                            if len(binParts[ix][iy])>0:
                                for h in range(0, len(binParts[ix][iy])):
                                    edgePhase[binParts[ix][iy][h]]=edge_id[ix][iy]
                                    partPhase[binParts[ix][iy][h]]=phaseBin[ix][iy]
                                    partTyp[binParts[ix][iy][h]]=typ[binParts[ix][iy][h]]
                        elif (edge_id[ix][iy] == 0) & (bulk_id2[ix][iy]==0):
                            bulk_bin=0
                            gas_bin=0
                            if (ix + 2) == NBins:
                                lookx = [ix-1, ix-1, ix, ix+1, 0]
                            elif (ix + 1) == NBins:
                                lookx = [ix-2, ix-1, ix, 0, 1]
                            elif ix==0:
                                lookx=[NBins-2, NBins-1, ix, ix+1, ix+2]
                            elif ix==1:
                                lookx=[NBins-1, ix-1, ix, ix+1, ix+2]
                            else:
                                lookx = [ix-2, ix-1, ix, ix+1, ix+2]

                            #Based on y-index (iy), find neighboring y-indices to loop through
                            if (iy + 2) == NBins:
                                looky = [iy-1, iy-1, iy, iy+1, 0]
                            elif (iy + 1) == NBins:
                                looky = [iy-2, iy-1, iy, 0, 1]
                            elif iy==0:
                                looky=[NBins-2, NBins-1, iy, iy+1, iy+2]
                            elif iy==1:
                                looky=[NBins-1, iy-1, iy, iy+1, iy+2]
                            else:
                                looky = [iy-2, iy-1, iy, iy+1, iy+2]

                            for indx in lookx:

                                for indy in looky:

                                    if phaseBin[indx][indy]==0:
                                        bulk_bin+=1
                                    elif phaseBin[indx][indy]==2:
                                        gas_bin+=1
                            if bulk_bin>=gas_bin:
                                phaseBin[ix][iy]=0
                            else:
                                phaseBin[ix][iy]=2
                            for h in range(0, len(binParts[ix][iy])):
                                    bulkPhase[binParts[ix][iy][h]]=bulk_id2[ix][iy]
                                    partPhase[binParts[ix][iy][h]]=phaseBin[ix][iy]
                                    partTyp[binParts[ix][iy][h]]=typ[binParts[ix][iy][h]]
                        elif bulk_id2[ix][iy]!=0:
                            if len(binParts[ix][iy])>0:
                                for h in range(0, len(binParts[ix][iy])):
                                    bulkPhase[binParts[ix][iy][h]]=bulk_id2[ix][iy]
                                    partPhase[binParts[ix][iy][h]]=phaseBin[ix][iy]
                                    partTyp[binParts[ix][iy][h]]=typ[binParts[ix][iy][h]]



            bub_id = []

            bub_fast_comp = np.array([])
            bub_slow_comp = np.array([])
            bub_total_comp = np.array([])

            dis_bub=0
            bub_large=0
            bub_large_ids=np.array([])
            if_bub=[]

            #Determine which grouping of particles (phases or different interfaces) are large enough to perform measurements on or if noise
            for m in range(0, end_test+1):

                    num_bubs_bins=0
                    #Find which particles belong to group 'm'
                    bub_temp = np.where(edgePhase==m)[0]
                    for ix in range(0, len(edge_id)):
                        for iy in range(0, len(edge_id)):
                            if edge_id[ix][iy]==m:
                                num_bubs_bins +=1
                    #If fewer than 100 particles belong to group 'm', then it is most likely noise and we should remove it
                    if (len(bub_temp)<=100) or (num_bubs_bins<10):
                        dis_bub+=1
                        edgePhase[bub_temp]=0

                        for ix in range(0, len(edge_id)):
                            for iy in range(0, len(edge_id)):
                                gasBin_temp=0
                                bulkBin_temp=0
                                if edge_id[ix][iy]==m:
                                    if m in possible_interface_ids:
                                        possible_interface_ids.remove(m)
                                    if (ix + 2) == NBins:
                                        lookx = [ix-1, ix-1, ix, ix+1, 0]
                                    elif (ix + 1) == NBins:
                                        lookx = [ix-2, ix-1, ix, 0, 1]
                                    elif ix==0:
                                        lookx=[NBins-2, NBins-1, ix, ix+1, ix+2]
                                    elif ix==1:
                                        lookx=[NBins-1, ix-1, ix, ix+1, ix+2]
                                    else:
                                        lookx = [ix-2, ix-1, ix, ix+1, ix+2]

                                    #Based on y-index (iy), find neighboring y-indices to loop through
                                    if (iy + 2) == NBins:
                                        looky = [iy-1, iy-1, iy, iy+1, 0]
                                    elif (iy + 1) == NBins:
                                        looky = [iy-2, iy-1, iy, 0, 1]
                                    elif iy==0:
                                        looky=[NBins-2, NBins-1, iy, iy+1, iy+2]
                                    elif iy==1:
                                        looky=[NBins-1, iy-1, iy, iy+1, iy+2]
                                    else:
                                        looky = [iy-2, iy-1, iy, iy+1, iy+2]

                                    for indx in lookx:
                                        for indy in looky:
                                            if phaseBin[indx][indy]==0:
                                                bulkBin_temp+=1
                                            elif phaseBin[indx][indy]==2:
                                                gasBin_temp+=1
                                    edge_id[ix][iy]=0
                                    if gasBin_temp>bulkBin_temp:
                                        phaseBin[ix][iy]=2
                                        if len(binParts[ix][iy])>0:
                                            for h in range(0, len(binParts[ix][iy])):
                                                partPhase[binParts[ix][iy][h]]=2
                                    else:
                                        phaseBin[ix][iy]=0
                                        if len(binParts[ix][iy])>0:
                                            for h in range(0, len(binParts[ix][iy])):
                                                partPhase[binParts[ix][iy][h]]=0


                    #If more than 100 particles belong to group 'm', then it is most likely significant and we should perform calculations
                    else:




                        #Label if structure is bulk/gas or interface
                        if len(np.where(partPhase[bub_temp]==0)[0])==0:

                            #Calculate composition of particles in each structure
                            bub_slow_comp = np.append(bub_slow_comp, len(np.where((edgePhase==m) & (partTyp==0))[0]))
                            bub_fast_comp = np.append(bub_fast_comp, len(np.where((edgePhase==m) & (partTyp==1))[0]))
                            bub_total_comp = np.append(bub_total_comp, len(np.where((edgePhase==m) & (partTyp==1))[0])+len(np.where((edgePhase==m) & (partTyp==0))[0]))
                            if_bub.append(1)
                            #Label significant structure IDs
                            bub_large_ids = np.append(bub_large_ids, m)

                            #Count number of significant structures
                            bub_large+=1

            #Initiate empty arrays
            bulk_fast_comp = np.array([])
            bulk_slow_comp = np.array([])
            bulk_total_comp = np.array([])
            if_bulk = []
            bulk_large=0
            bulk_large_ids = np.array([])

            #Calculate composition of each bulk phase structure
            for m in range(0, end_test2+1):

                    num_bulk_bins=0
                    #Find which particles belong to group 'm'
                    bulk_temp = np.where(bulkPhase==m)[0]
                    for ix in range(0, len(bulk_id2)):
                        for iy in range(0, len(bulk_id2)):
                            if bulk_id2[ix][iy]==m:
                                num_bulk_bins +=1



                    #Label if structure is bulk/gas or interface
                    if len(np.where(partPhase[bulk_temp]==0)[0])>0:

                        if_bulk.append(1)
                        #Calculate composition of particles in each structure
                        bulk_slow_comp = np.append(bulk_slow_comp, len(np.where((bulkPhase==m) & (partTyp==0))[0]))
                        bulk_fast_comp = np.append(bulk_fast_comp, len(np.where((bulkPhase==m) & (partTyp==1))[0]))
                        bulk_total_comp = np.append(bulk_total_comp, len(np.where((bulkPhase==m) & (partTyp==1))[0])+len(np.where((bulkPhase==m) & (partTyp==0))[0]))
                        #Label significant structure IDs
                        bulk_large_ids = np.append(bulk_large_ids, m)

                        bulk_large+=1

            #Identify which of the largest bubbles is a possible gas-dense interface
            int_poss_ids = []
            for k in range(0, len(possible_interface_ids)):

                if possible_interface_ids[k] in bub_large_ids:

                    int_poss_ids.append(np.where(bub_large_ids==possible_interface_ids[k])[0][0])

            # Determine order of interfaces based on size (largest=dense + gas phases, second largest = gas/dense interface, etc.)

            #Initiate empty arrays
            bub_id_arr = np.array([], dtype=int)
            bub_size_id_arr = np.array([], dtype=int)
            bub_fast_arr = np.array([], dtype=int)
            bub_slow_arr = np.array([], dtype=int)
            if_bub_id_arr = np.array([], dtype=int)

            #If 5 or more interface structures, continue...
            if bub_large>=5:
                    if len(int_poss_ids)>0:
                        first=np.max(bub_total_comp[int_poss_ids])
                    else:
                        first=np.max(bub_total_comp)

                    bub_first_id = np.where(bub_total_comp==first)[0]

                    for k in range(0, len(bub_first_id)):
                        if len(bub_id_arr)<5:
                            bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                            bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                            if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                            bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                            bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                    if len(bub_id_arr)<5:
                        second_arr = np.where(bub_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bub_total_comp[second_arr])
                            bub_second_id = np.where(bub_total_comp==second)[0]
                            for k in range(0, len(bub_second_id)):
                                if len(bub_id_arr)<5:
                                    bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bub_second_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bub_total_comp[third_arr])
                            bub_third_id = np.where(bub_total_comp==third)[0]
                            for k in range(0, len(bub_third_id)):
                                if len(bub_id_arr)<5:
                                    bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bub_third_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fourth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third))[0]
                        if len(fourth_arr)>0:
                            fourth = np.max(bub_total_comp[fourth_arr])
                            bub_fourth_id = np.where(bub_total_comp==fourth)[0]
                            for k in range(0, len(bub_fourth_id)):
                                if len(bub_id_arr)<5:
                                    bub_id_arr = np.append(bub_id_arr, bub_fourth_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fourth_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fourth_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fourth_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fourth_id[k]])
                        else:
                            fourth_arr = 0
                            fourth = 0
                            bub_fourth_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fifth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third) & (bub_total_comp!=fourth))[0]
                        if len(fifth_arr)>0:
                            fifth = np.max(bub_total_comp[fifth_arr])
                            bub_fifth_id = np.where(bub_total_comp==fifth)[0]
                            for k in range(0, len(bub_fifth_id)):
                                if len(bub_id_arr)<5:
                                    bub_id_arr = np.append(bub_id_arr, bub_fifth_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fifth_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fifth_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fifth_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fifth_id[k]])
                        else:
                            fifth_arr = 0
                            fifth = 0
                            bub_fifth_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)
                    clust_true = 1
            #If 4 interface structures...
            elif bub_large==4:
                    if len(int_poss_ids)>0:
                        first=np.max(bub_total_comp[int_poss_ids])
                    else:
                        first=np.max(bub_total_comp)

                    bub_first_id = np.where(bub_total_comp==first)[0]

                    for k in range(0, len(bub_first_id)):
                        if len(bub_id_arr)<4:
                            bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                            bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                            if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                            bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                            bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                    if len(bub_id_arr)<4:
                        second_arr = np.where(bub_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bub_total_comp[second_arr])
                            bub_second_id = np.where(bub_total_comp==second)[0]
                            for k in range(0, len(bub_second_id)):
                                if len(bub_id_arr)<4:
                                    bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bub_second_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<4:
                        third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bub_total_comp[third_arr])
                            bub_third_id = np.where(bub_total_comp==third)[0]
                            for k in range(0, len(bub_third_id)):
                                if len(bub_id_arr)<4:
                                    bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bub_third_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<4:
                        fourth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third))[0]
                        if len(fourth_arr)>0:
                            fourth = np.max(bub_total_comp[fourth_arr])
                            bub_fourth_id = np.where(bub_total_comp==fourth)[0]
                            for k in range(0, len(bub_fourth_id)):
                                if len(bub_id_arr)<4:
                                    bub_id_arr = np.append(bub_id_arr, bub_fourth_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fourth_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fourth_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fourth_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fourth_id[k]])
                        else:
                            fourth_arr = 0
                            fourth = 0
                            bub_fourth_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)
                    if len(bub_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)


                    clust_true = 1
            #If 3 interface structures...
            elif bub_large==3:
                    if len(int_poss_ids)>0:
                        first=np.max(bub_total_comp[int_poss_ids])
                    else:
                        first=np.max(bub_total_comp)

                    bub_first_id = np.where(bub_total_comp==first)[0]

                    for k in range(0, len(bub_first_id)):
                        if len(bub_id_arr)<3:
                            bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                            bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                            if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                            bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                            bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                    if len(bub_id_arr)<3:
                        second_arr = np.where(bub_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bub_total_comp[second_arr])
                            bub_second_id = np.where(bub_total_comp==second)[0]
                            for k in range(0, len(bub_second_id)):
                                if len(bub_id_arr)<3:
                                    bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bub_second_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<3:
                        third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bub_total_comp[third_arr])
                            bub_third_id = np.where(bub_total_comp==third)[0]
                            for k in range(0, len(bub_third_id)):
                                if len(bub_id_arr)<3:
                                    bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bub_third_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)
                    if len(bub_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)


                    clust_true = 1

            #If 2 interface structures...
            elif bub_large==2:
                    if len(int_poss_ids)>0:
                        first=np.max(bub_total_comp[int_poss_ids])
                    else:
                        first=np.max(bub_total_comp)

                    bub_first_id = np.where(bub_total_comp==first)[0]

                    for k in range(0, len(bub_first_id)):
                        if len(bub_id_arr)<2:
                            bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                            bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                            if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                            bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                            bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                    if len(bub_id_arr)<2:
                        second_arr = np.where(bub_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bub_total_comp[second_arr])
                            bub_second_id = np.where(bub_total_comp==second)[0]
                            for k in range(0, len(bub_second_id)):
                                if len(bub_id_arr)<2:
                                    bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                    bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                    if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                    bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                    bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bub_second_id = 0
                            bub_id_arr = np.append(bub_id_arr, 999)
                            bub_size_id_arr = np.append(bub_size_id_arr, 0)
                            if_bub_id_arr = np.append(if_bub_id_arr, 0)
                            bub_fast_arr = np.append(bub_fast_arr, 0)
                            bub_slow_arr = np.append(bub_slow_arr, 0)
                    if len(bub_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)


                    clust_true = 1

            #If 1 interface structure...
            elif bub_large==1:
                    if len(int_poss_ids)>0:
                        first=np.max(bub_total_comp[int_poss_ids])
                    else:
                        first=np.max(bub_total_comp)

                    bub_first_id = np.where(bub_total_comp==first)[0]

                    for k in range(0, len(bub_first_id)):
                        if len(bub_id_arr)<1:
                            bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                            bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                            if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                            bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                            bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                    if len(bub_id_arr)<5:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)


                    clust_true = 1

            #If no interface structures (this is an error)...
            else:

                    if len(bub_id_arr)<5:
                        first_arr = 0
                        first = 0
                        bub_first_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                    if len(bub_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)


                    clust_true = 1

            #Initiate empty arrays
            bulk_id_arr = np.array([], dtype=int)
            bulk_size_id_arr = np.array([], dtype=int)
            bulk_fast_arr = np.array([], dtype=int)
            bulk_slow_arr = np.array([], dtype=int)
            if_bulk_id_arr = np.array([], dtype=int)

            #If 5 or more bulk phase structure...
            if bulk_large>=5:

                    first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

                    bulk_first_id = np.where(bulk_total_comp==first)[0]

                    for k in range(0, len(bulk_first_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

                    if len(bulk_id_arr)<5:
                        second_arr = np.where(bulk_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bulk_total_comp[second_arr])
                            bulk_second_id = np.where(bulk_total_comp==second)[0]
                            for k in range(0, len(bulk_second_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bulk_second_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bulk_total_comp[third_arr])
                            bulk_third_id = np.where(bulk_total_comp==third)[0]
                            for k in range(0, len(bulk_third_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bulk_third_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)
                    if len(bulk_id_arr)<5:
                        fourth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third))[0]
                        if len(fourth_arr)>0:
                            fourth = np.max(bulk_total_comp[fourth_arr])
                            bulk_fourth_id = np.where(bulk_total_comp==fourth)[0]
                            for k in range(0, len(bulk_fourth_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_fourth_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fourth_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fourth_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fourth_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fourth_id[k]])
                        else:
                            fourth_arr = 0
                            fourth = 0
                            bulk_fourth_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third) & (bulk_total_comp!=fourth))[0]
                        if len(fifth_arr)>0:
                            fifth = np.max(bulk_total_comp[fifth_arr])
                            bulk_fifth_id = np.where(bulk_total_comp==fifth)[0]
                            for k in range(0, len(bulk_fifth_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_fifth_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fifth_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fifth_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fifth_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fifth_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bulk_second_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)
                    clust_true = 1
            #If 4 bulk phase structures...
            elif bulk_large==4:
                    first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

                    bulk_first_id = np.where(bulk_total_comp==first)[0]

                    for k in range(0, len(bulk_first_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

                    if len(bulk_id_arr)<5:
                        second_arr = np.where(bulk_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bulk_total_comp[second_arr])
                            bulk_second_id = np.where(bulk_total_comp==second)[0]
                            for k in range(0, len(bulk_second_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bulk_second_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bulk_total_comp[third_arr])
                            bulk_third_id = np.where(bulk_total_comp==third)[0]
                            for k in range(0, len(bulk_third_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bulk_third_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fourth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third))[0]
                        if len(fourth_arr)>0:
                            fourth = np.max(bulk_total_comp[fourth_arr])
                            bulk_fourth_id = np.where(bulk_total_comp==fourth)[0]
                            for k in range(0, len(bulk_fourth_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_fourth_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fourth_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fourth_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fourth_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fourth_id[k]])
                        else:
                            fourth_arr = 0
                            fourth = 0
                            bulk_fourth_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bulk_fifth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)
            #If 3 bulk phase structures...
            elif bulk_large==3:
                    first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

                    bulk_first_id = np.where(bulk_total_comp==first)[0]

                    for k in range(0, len(bulk_first_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

                    if len(bulk_id_arr)<5:
                        second_arr = np.where(bulk_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bulk_total_comp[second_arr])
                            bulk_second_id = np.where(bulk_total_comp==second)[0]
                            for k in range(0, len(bulk_second_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bulk_second_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                        if len(third_arr)>0:
                            third = np.max(bulk_total_comp[third_arr])
                            bulk_third_id = np.where(bulk_total_comp==third)[0]
                            for k in range(0, len(bulk_third_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                        else:
                            third_arr = 0
                            third = 0
                            bulk_third_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bulk_fourth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bulk_fifth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    clust_true = 1

            #If 2 bulk phase structures...
            elif bulk_large==2:
                    first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

                    bulk_first_id = np.where(bulk_total_comp==first)[0]

                    for k in range(0, len(bulk_first_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

                    if len(bulk_id_arr)<5:
                        second_arr = np.where(bulk_total_comp != first)[0]
                        if len(second_arr)>0:
                            second = np.max(bulk_total_comp[second_arr])
                            bulk_second_id = np.where(bulk_total_comp==second)[0]
                            for k in range(0, len(bulk_second_id)):
                                if len(bulk_id_arr)<5:
                                    bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                        else:
                            second_arr = 0
                            second = 0
                            bulk_second_id = 0
                            bulk_id_arr = np.append(bulk_id_arr, 999)
                            bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                            if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                            bulk_fast_arr = np.append(bulk_fast_arr, 0)
                            bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bulk_third_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bulk_fourth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bulk_fifth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    clust_true = 1

            #If 1 bulk phase structures...
            elif bulk_large==1:
                    first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

                    bulk_first_id = np.where(bulk_total_comp==first)[0]

                    for k in range(0, len(bulk_first_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

                    if len(bulk_id_arr)<5:
                        second_arr = 0
                        second = 0
                        bulk_second_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bulk_third_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bulk_fourth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bulk_fifth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    clust_true = 1

            #If 0 bulk phase structures...
            elif bulk_large==0:
                    if len(bulk_id_arr)<5:
                        first_arr = 0
                        first = 0
                        bulk_first_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        second_arr = 0
                        second = 0
                        bulk_second_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        third_arr = 0
                        third = 0
                        bulk_third_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fourth_arr = 0
                        fourth = 0
                        bulk_fourth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    if len(bulk_id_arr)<5:
                        fifth_arr = 0
                        fifth = 0
                        bulk_fifth_id = 0
                        bulk_id_arr = np.append(bulk_id_arr, 999)
                        bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                        if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                        bulk_fast_arr = np.append(bulk_fast_arr, 0)
                        bulk_slow_arr = np.append(bulk_slow_arr, 0)

                    clust_true = 1

            #Identify which structures are bubbles
            bub_ids = np.where(if_bub_id_arr==1)[0]


            #Identify which structures are bulk/gas phase
            bulk_ids = np.where(if_bub_id_arr==0)[0]

            #If bubbles exist, calculate the structure ID for the interface
            if len(bub_ids)>0:
                interface_id = bub_size_id_arr[np.min(np.where(if_bub_id_arr==1)[0])]
            #If bulk/gas exist, calculate the structure ID for the gas/bulk
            if len(bulk_ids)>0:
                bulk_id = bub_size_id_arr[np.min(np.where(if_bub_id_arr==0)[0])]

            # Individually label each interface until all edge bins identified using flood fill algorithm
            if len(bub_ids)>0:
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):

                        #If bin is an interface, continue
                        if phaseBin[ix][iy]==1:

                                        #Count surrounding bin phases
                                        gas_count=0
                                        bulk_count=0

                                        #identify neighboring bins
                                        if (ix + 1) == NBins:
                                                    lookx = [ix-1, ix, 0]
                                        elif ix==0:
                                                    lookx=[NBins-1, ix, ix+1]
                                        else:
                                                    lookx = [ix-1, ix, ix+1]
                                        if (iy + 1) == NBins:
                                                        looky = [iy-1, iy, 0]
                                        elif iy==0:
                                                        looky=[NBins-1, iy, iy+1]
                                        else:
                                                        looky = [iy-1, iy, iy+1]
                                        if int(edge_id[ix][iy])==interface_id:

                                            #loop over surrounding x-index bins
                                            for indx in lookx:

                                                # Loop through surrounding y-index bins
                                                for indy in looky:


                                                    #If bin hadn't been assigned an interface id yet, continue

                                                    #If bin is a gas, continue
                                                    if phaseBin[indx][indy]==2:

                                                            #count number of gas bins
                                                            gas_count+=1

                                                    elif phaseBin[indx][indy]==0:

                                                            bulk_count+=1

                                                    #If more than interface bins surround, identify if interior or exterior edge
                                                    if (gas_count>0) or (bulk_count>0):
                                                        #If more neighboring gas bins around reference bin, then it's an exterior edge
                                                        if gas_count>=bulk_count:
                                                            ext_edge_id[ix][iy]=1

                                                        #Otherwise, it's an interior edge
                                                        else:
                                                            int_edge_id[ix][iy]=1

                                        elif int(edge_id[ix][iy])!=0:
                                            #loop over surrounding x-index bins
                                            for indx in lookx:

                                                # Loop through surrounding y-index bins
                                                for indy in looky:

                                                    #If bin is a gas, count it
                                                    if phaseBin[indx][indy]==2:
                                                            gas_count+=1

                                                    #If bin is a bulk, count it
                                                    elif phaseBin[indx][indy]==0:
                                                            bulk_count+=1

                                                    #If surrounding bins aren't all interface, continue...
                                                    if (gas_count>0) or (bulk_count>0):

                                                        #If more bulk than gas, the bin is an external edge
                                                        if gas_count<=bulk_count:
                                                            ext_edge_id[ix][iy]=1

                                                        #If more gas than bulk, the bin is an internal edge
                                                        else:
                                                            int_edge_id[ix][iy]=1

            #Label phase of each particle
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if len(binParts[ix][iy])>0:
                            for h in range(0, len(binParts[ix][iy])):

                                extedgePhase[binParts[ix][iy][h]]=ext_edge_id[ix][iy]
                                intedgePhase[binParts[ix][iy][h]]=int_edge_id[ix][iy]


            #Save positions of external and internal edges
            ext_pos_box_x_arr=np.array([])
            ext_pos_box_y_arr=np.array([])
            int_pos_box_x_arr=np.array([])
            int_pos_box_y_arr=np.array([])
            clust_true = 0
            int_vert_x=np.array([])
            int_vert_y=np.array([])
            ext_vert_x=np.array([])
            ext_vert_y=np.array([])
            int_codes=[]
            ext_codes = []

            #Save positions of interior and exterior edge bins
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                                if ext_edge_id[ix][iy]==1:
                                    ext_pos_box_x_arr=np.append(ext_pos_box_x_arr, (ix+0.5)*sizeBin)
                                    ext_pos_box_y_arr=np.append(ext_pos_box_y_arr, (iy+0.5)*sizeBin)
                                    #if len(ext_vert_x)==0:
                                        #ext_vert_x = np.append(ext_vert_x, (ix+0.5)*sizeBin)
                                        #ext_vert_y = np.append(ext_vert_y, (iy+0.5)*sizeBin)


                                    #ext_codes = np.append(ext_codes, Path.LINETO)
                                elif int_edge_id[ix][iy]==1:
                                    int_pos_box_x_arr=np.append(int_pos_box_x_arr, (ix+0.5)*sizeBin)
                                    int_pos_box_y_arr=np.append(int_pos_box_y_arr, (iy+0.5)*sizeBin)
                                    #int_vert_x = np.append(int_vert_x, (ix+0.5)*sizeBin)
                                    #int_vert_y = np.append(int_vert_y, (iy+0.5)*sizeBin)
                                    #int_codes = np.append(int_codes, Path.LINETO)

            #ext_vert_x = np.append(ext_vert_x, ext_pos_box_x_arr[0])
            #ext_vert_y = np.append(ext_vert_y, ext_pos_box_y_arr[0])

            #ext_pos_box_x_arr = np.delete(ext_pos_box_x_arr, 0)
            #ext_pos_box_y_arr = np.delete(ext_pos_box_y_arr, 0)

            if j>(start*time_step):
                while len(ext_pos_box_x_arr)>0:
                    if len(ext_vert_x)==0:
                        ext_vert_x = np.append(ext_vert_x, ext_pos_box_x_arr[0])
                        ext_vert_y = np.append(ext_vert_y, ext_pos_box_y_arr[0])

                        ext_pos_box_x_arr = np.delete(ext_pos_box_x_arr, 0)
                        ext_pos_box_y_arr = np.delete(ext_pos_box_y_arr, 0)

                        ext_codes = np.append(ext_codes, Path.MOVETO)
                    else:
                        shortest_length = 100000
                        for iy in range(0, len(ext_pos_box_y_arr)):
                            difx = ext_vert_x[-1]-ext_pos_box_x_arr[iy]
                            dify = ext_vert_y[-1]-ext_pos_box_y_arr[iy]

                            #Enforce periodic boundary conditions
                            difx_abs = np.abs(difx)
                            if difx_abs>=h_box:
                                    if difx < -h_box:
                                        difx += l_box
                                    else:
                                        difx -= l_box

                            #Enforce periodic boundary conditions
                            dify_abs = np.abs(dify)
                            if dify_abs>=h_box:
                                    if dify < -h_box:
                                        dify += l_box
                                    else:
                                        dify -= l_box

                            difr = (difx**2 + dify**2)**0.5
                            if difr < shortest_length:
                                shortest_length = difr
                                shortest_xlength = difx
                                shortest_ylength = dify
                                shortest_id = iy
                            elif difr == shortest_length:
                                if (difx<0) or (dify<0):
                                    if (shortest_xlength <0) or (shortest_ylength<0):
                                        shortest_length = difr
                                        shortest_xlength = difx
                                        shortest_ylength = dify
                                        shortest_id = iy
                                    else:
                                        pass
                                else:
                                    pass

                        ext_vert_x = np.append(ext_vert_x, ext_pos_box_x_arr[shortest_id])
                        ext_vert_y = np.append(ext_vert_y, ext_pos_box_y_arr[shortest_id])

                        ext_pos_box_x_arr = np.delete(ext_pos_box_x_arr, shortest_id)
                        ext_pos_box_y_arr = np.delete(ext_pos_box_y_arr, shortest_id)

                while len(int_pos_box_x_arr)>0:
                    if len(int_vert_x)==0:
                        int_vert_x = np.append(int_vert_x, int_pos_box_x_arr[0])
                        int_vert_y = np.append(int_vert_y, int_pos_box_y_arr[0])

                        int_pos_box_x_arr = np.delete(int_pos_box_x_arr, 0)
                        int_pos_box_y_arr = np.delete(int_pos_box_y_arr, 0)

                        int_codes = np.append(int_codes, Path.MOVETO)
                    else:
                        shortest_length = 100000
                        for iy in range(0, len(int_pos_box_y_arr)):
                            difx = int_vert_x[-1]-int_pos_box_x_arr[iy]
                            dify = int_vert_y[-1]-int_pos_box_y_arr[iy]

                            #Enforce periodic boundary conditions
                            difx_abs = np.abs(difx)
                            if difx_abs>=h_box:
                                    if difx < -h_box:
                                        difx += l_box
                                    else:
                                        difx -= l_box

                            #Enforce periodic boundary conditions
                            dify_abs = np.abs(dify)
                            if dify_abs>=h_box:
                                    if dify < -h_box:
                                        dify += l_box
                                    else:
                                        dify -= l_box

                            difr = (difx**2 + dify**2)**0.5
                            if difr < shortest_length:
                                shortest_length = difr
                                shortest_xlength = difx
                                shortest_ylength = dify
                                shortest_id = iy

                            elif difr == shortest_length:
                                if (difx<0) or (dify<0):
                                    if (shortest_xlength <0) or (shortest_ylength<0):
                                        shortest_length = difr
                                        shortest_xlength = difx
                                        shortest_ylength = dify
                                        shortest_id = iy
                                    else:
                                        pass
                                else:
                                    pass

                        int_vert_x = np.append(int_vert_x, int_pos_box_x_arr[shortest_id])
                        int_vert_y = np.append(int_vert_y, int_pos_box_y_arr[shortest_id])
                        int_pos_box_x_arr = np.delete(int_pos_box_x_arr, shortest_id)
                        int_pos_box_y_arr = np.delete(int_pos_box_y_arr, shortest_id)




                #np.reshape(int_vert )

                #int_codes[0]=Path.MOVETO
                #ext_codes[0]=Path.MOVETO



            #If there is an interface (bubble), find the mid-point of the cluster's edges
            #Constant density in bulk phase, so approximately center of mass
            if len(bub_ids) > 0:
                    edge_num_bin=0
                    x_box_pos=0
                    y_box_pos=0

                    #Sum positions of external edges of interface
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if len(ext_pos_box_x_arr)>0:
                                if (edge_id[ix][iy]==interface_id) & (ext_edge_id[ix][iy]==1):
                                    x_box_pos += (ix+0.5)*sizeBin
                                    y_box_pos += (iy+0.5)*sizeBin
                                    edge_num_bin +=1
                            elif len(int_pos_box_x_arr)>0:
                                if (edge_id[ix][iy]==interface_id) & (int_edge_id[ix][iy]==1):
                                    x_box_pos += (ix+0.5)*sizeBin
                                    y_box_pos += (iy+0.5)*sizeBin
                                    edge_num_bin +=1

                    #Determine mean location (CoM) of external edges of interface
                    if edge_num_bin>0:
                        box_com_x = x_box_pos/edge_num_bin
                        box_com_y = y_box_pos/edge_num_bin
                    else:
                        box_com_x=0
                        box_com_y=0


                    #Initialize empty arrays for calculation
                    theta_id = np.array([])
                    radius_id = np.array([])
                    x_id = np.array([], dtype=int)
                    y_id = np.array([], dtype=int)

                    #Calculate distance from CoM to external edge bin and angle from CoM
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):

                            # If bin is interface and external edge, continue...
                            if (edge_id[ix][iy]==interface_id) & (ext_edge_id[ix][iy]==1):

                                #Reference bin location
                                x_box_pos = (ix+0.5)*sizeBin
                                y_box_pos = (iy+0.5)*sizeBin

                                #Calculate x-distance from CoM
                                difx=x_box_pos-box_com_x

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                if difx_abs>=h_box:
                                        if difx < -h_box:
                                            difx += l_box
                                        else:
                                            difx -= l_box

                                #Calculate y-distance from CoM
                                dify=y_box_pos-box_com_y

                                #Enforce periodic boundary conditions
                                dify_abs = np.abs(dify)
                                if dify_abs>=h_box:
                                        if dify < -h_box:
                                            dify += l_box
                                        else:
                                            dify -= l_box

                                #Calculate angle from CoM and x-axis
                                theta_val = np.arctan2(np.abs(dify), np.abs(difx))*(180/math.pi)

                                #Enforce correct quadrant for particle
                                if (difx>0) & (dify>0):
                                    pass
                                elif (difx<0) & (dify>0):
                                    theta_val = 180-theta_val
                                elif (difx<0) & (dify<0):
                                    theta_val = theta_val+180
                                elif (difx>0) & (dify<0):
                                    theta_val = 360-theta_val

                                #Save calculated angle from CoM and x-axis
                                theta_id = np.append(theta_id, theta_val)

                                #Save id of bin of calculation
                                x_id = np.append(x_id, int(ix))
                                y_id = np.append(y_id, int(iy))

                                #Save radius from CoM of bin
                                radius_id = np.append(radius_id, (difx**2 + dify**2)**0.5)
            '''
            #Measures radius and width of each interface
            edge_width = []
            bub_width_ext = []
            bub_width_int = []
            bub_width = []

            id_step = 0

            #Loop over interfaces
            for m in range(0, len(bub_id_arr)):

                #Always true
                if if_bub_id_arr[m]==1:

                    #Find which particles belong to mth interface structure
                    edge_parts = np.where((edgePhase==bub_size_id_arr[m]))[0]

                    #If particles belong to mth interface structure, continue...
                    if len(edge_parts)>0:

                        #Initiate empty arrays
                        shortest_r=np.array([])
                        bub_rad_int=np.array([])
                        bub_rad_ext=np.array([])

                        #Find interior and exterior particles of interface
                        int_bub_id_tmp = np.where((edgePhase==bub_size_id_arr[m]) & (intedgePhase==1))[0]
                        ext_bub_id_tmp = np.where((edgePhase==bub_size_id_arr[m]) & (extedgePhase==1))[0]

                        #Calculate (x,y) center of mass of interface
                        x_com_bub = np.mean(pos[edge_parts,0]+h_box)
                        y_com_bub = np.mean(pos[edge_parts,1]+h_box)

                        #Loop over bins in system
                        for ix in range(0, len(occParts)):
                                    for iy in range(0, len(occParts)):

                                        #If bin belongs to mth interface structure, continue...
                                        if edge_id[ix][iy]==bub_size_id_arr[m]:

                                            #If bin is an exterior particle of mth interface structure, continue...
                                            if ext_edge_id[ix][iy]==1:

                                                #Calculate (x,y) position of bin
                                                pos_box_x1 = (ix+0.5)*sizeBin
                                                pos_box_y1 = (iy+0.5)*sizeBin

                                                #Calculate x distance from mth interface structure's center of mass
                                                bub_rad_tmp_x = (pos_box_x1-x_com_bub)
                                                bub_rad_tmp_x_abs = np.abs(pos_box_x1-x_com_bub)

                                                #Enforce periodic boundary conditions
                                                if bub_rad_tmp_x_abs>=h_box:
                                                    if bub_rad_tmp_x < -h_box:
                                                        bub_rad_tmp_x += l_box
                                                    else:
                                                        bub_rad_tmp_x -= l_box

                                                #Calculate y distance from mth interface structure's center of mass
                                                bub_rad_tmp_y = (pos_box_y1-y_com_bub)
                                                bub_rad_tmp_y_abs = np.abs(pos_box_y1-y_com_bub)

                                                #Enforce periodic boundary conditions
                                                if bub_rad_tmp_y_abs>=h_box:
                                                    if bub_rad_tmp_y < -h_box:
                                                        bub_rad_tmp_y += l_box
                                                    else:
                                                        bub_rad_tmp_y -= l_box

                                                #Calculate magnitude of distance from center of mass of mth interface structure
                                                bub_rad_tmp = (bub_rad_tmp_x**2 + bub_rad_tmp_y**2)**0.5

                                                #Save this interface's radius to array
                                                bub_rad_ext = np.append(bub_rad_ext, bub_rad_tmp+(sizeBin/2))

                                            #If bin is interior particle of mth interface structure, continue
                                            if int_edge_id[ix][iy]==1:

                                                #Calculate (x,y) position of bin
                                                pos_box_x1 = (ix+0.5)*sizeBin
                                                pos_box_y1 = (iy+0.5)*sizeBin

                                                #Calculate x distance from mth interface structure's center of mass
                                                bub_rad_tmp_x = (pos_box_x1-x_com_bub)
                                                bub_rad_tmp_x_abs = np.abs(pos_box_x1-x_com_bub)

                                                #Enforce periodic boundary conditions
                                                if bub_rad_tmp_x_abs>=h_box:
                                                    if bub_rad_tmp_x < -h_box:
                                                        bub_rad_tmp_x += l_box
                                                    else:
                                                        bub_rad_tmp_x -= l_box

                                                #Calculate y distance from mth interface structure's center of mass
                                                bub_rad_tmp_y = (pos_box_y1-y_com_bub)
                                                bub_rad_tmp_y_abs = np.abs(pos_box_y1-y_com_bub)

                                                #Enforce periodic boundary conditions
                                                if bub_rad_tmp_y_abs>=h_box:
                                                    if bub_rad_tmp_y < -h_box:
                                                        bub_rad_tmp_y += l_box
                                                    else:
                                                        bub_rad_tmp_y -= l_box

                                                #Calculate magnitude of distance to mth interface structure's center of mass
                                                bub_rad_tmp = (bub_rad_tmp_x**2 + bub_rad_tmp_y**2)**0.5

                                                #Save this interface's interior radius to array
                                                bub_rad_int = np.append(bub_rad_int, bub_rad_tmp+(sizeBin/2))
                        #if there were interior bins found, calculate the average interior radius of mth interface structure
                        if len(bub_rad_int)>0:
                            bub_width_int.append(np.mean(bub_rad_int)+sizeBin/2)
                        else:
                            bub_width_int.append(0)

                        #if there were exterior bins found, calculate the average exterior radius of mth interface structure
                        if len(bub_rad_ext)>0:
                            bub_width_ext.append(np.mean(bub_rad_ext)+sizeBin/2)
                        else:
                            bub_width_ext.append(0)

                        #Use whichever is larger to calculate the true radius of the mth interface structure
                        if bub_width_ext[id_step]>bub_width_int[id_step]:
                            bub_width.append(bub_width_ext[id_step])
                        else:
                            bub_width.append(bub_width_int[id_step])

                        #If both interior and exterior particles were identified, continue...
                        if (len(int_bub_id_tmp)>0) & (len(ext_bub_id_tmp)>0):

                                #Loop over bins in system
                                for ix in range(0, len(occParts)):
                                    for iy in range(0, len(occParts)):

                                        #If bin is part of mth interface structure, continue...
                                        if edge_id[ix][iy]==bub_size_id_arr[m]:

                                            #If bin is an exterior bin of mth interface structure, continue...
                                            if ext_edge_id[ix][iy]==1:

                                                #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                                                difr_short=10000000.

                                                #Calculate position of exterior edge bin
                                                pos_box_x1 = (ix+0.5)*sizeBin
                                                pos_box_y1 = (iy+0.5)*sizeBin

                                                #Loop over bins of system
                                                for ix2 in range(0, len(occParts)):
                                                    for iy2 in range(0, len(occParts)):

                                                        #If bin belongs to mth interface structure, continue...
                                                        if edge_id[ix2][iy2]==bub_size_id_arr[m]:

                                                            #If bin is an interior edge bin for mth interface structure, continue...
                                                            if int_edge_id[ix2][iy2]==1:

                                                                    #Calculate position of interior edge bin
                                                                    pos_box_x2 = (ix2+0.5)*sizeBin
                                                                    pos_box_y2 = (iy2+0.5)*sizeBin

                                                                    #Calculate distance from interior edge bin to exterior edge bin
                                                                    difr = ( (pos_box_x1-pos_box_x2)**2 + (pos_box_y1-pos_box_y2)**2)**0.5

                                                                    #If this distance is the shortest calculated thus far, replace the value with it
                                                                    if difr<difr_short:
                                                                        difr_short=difr
                                                #Save each shortest distance to an interior edge bin calculated for each exterior edge bin
                                                shortest_r = np.append(shortest_r, difr_short)

                                #Calculate and save the average shortest-distance between each interior edge and exterior edge bins for the mth interface structure
                                edge_width.append(np.mean(shortest_r)+sizeBin)

                        #If both an interior and exterior edge were not identified, save the cluster radius instead for the edge width
                        else:
                            edge_width.append(bub_width[id_step])

                        #Step for number of bins with identified edge width
                        id_step +=1

                    #If no particles in interface, save zeros for radius and width
                    else:
                        edge_width.append(0)
                        bub_width.append(0)

                #Never true
                else:
                    edge_width.append(0)
                    bub_width.append(0)
            '''

            #Initiate counts of phases/structures
            bulkBin=0
            gasBin=0
            intBin=0
            firstbubBin=0
            secondbubBin=0
            thirdbubBin=0
            fourthbubBin=0
            bubBin2=0
            bubBin=np.zeros(len(bub_id_arr))
            bulkBin_arr=np.zeros(len(bub_id_arr))

            #Measure number of bins belong to each phase
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if phaseBin[ix][iy]==0:
                        bulkBin+=1
                    elif phaseBin[ix][iy]==2:
                        gasBin+=1
                    elif phaseBin[ix][iy]==1:
                        bubBin2+=1
                    if edge_id[ix][iy]==interface_id:
                        intBin+=1

            #Count number of bins belonging to each interface structure
            for m in range(0, len(bub_id_arr)):
                if if_bub_id_arr[m]!=0:
                    #if (bub_fast_arr[m]!=0) or (bub_slow_arr[m]!=0):
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if edge_id[ix][iy] == bub_size_id_arr[m]:
                                bubBin[m] +=1

            #Count number of bins belonging to each bulk phase structure
            for m in range(0, len(bulk_id_arr)):
                #if (bulk_fast_arr[m]!=0) or (bulk_slow_arr[m]!=0):
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if bulk_id2[ix][iy] == bulk_size_id_arr[m]:
                            bulkBin_arr[m] +=1

            #Initiate empty arrays for velocity outputs
            v_all_x_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_y_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            v_all_xA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_yA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            v_all_xB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_yB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_x_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_y_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_xA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_yA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            v_avg_xB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_yB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            #Initiate empty arrays
            vel_mag = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magA = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magB = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magDif = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normx = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normy = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normxA = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normyA = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normxB = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normyB = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normx_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normy_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normxA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normyA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normxB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_normyB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_normDif = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_mag_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magA_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magB_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_magDif_no_gas = [[0 for b in range(NBins)] for a in range(NBins)]

            vel_grad_x = [[0 for b in range(NBins)] for a in range(NBins)]
            vel_grad_y = [[0 for b in range(NBins)] for a in range(NBins)]

            div = [[0 for b in range(NBins)] for a in range(NBins)]

            align_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))

            pos_box_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
            if j>(start*time_step):

                for ix in range(0, len(v_avg_x)):
                    for iy in range(0, len(v_avg_y)):

                        align_combined[ix][iy][0]=align_avg_x[ix][iy]
                        align_combined[ix][iy][1]=align_avg_y[ix][iy]

                        pos_box_combined[ix][iy][0]=pos_box_x[ix][iy]
                        pos_box_combined[ix][iy][1]=pos_box_y[ix][iy]

                #v_grad = np.gradient
                #vx_grad = np.gradient(v_avg_x_no_gas, pos_box_x)   #central diff. du_dx
                #vy_grad = np.gradient(v_avg_y_no_gas, pos_box_y)  #central diff. dv_dy


            #Slow/fast composition of bulk phase
            slow_bulk_num = len(np.where((partPhase==0) & (partTyp==0))[0])
            fast_bulk_num = len(np.where((partPhase==0) & (partTyp==1))[0])

            #Slow/fast composition of gas phase
            slow_gas_num = len(np.where((partPhase==2) & (partTyp==0))[0])
            fast_gas_num = len(np.where((partPhase==2) & (partTyp==1))[0])

            #Slow/fast composition of main interface
            slow_int_num = len(np.where((edgePhase==interface_id) & (partTyp==0))[0])
            fast_int_num = len(np.where((edgePhase==interface_id) & (partTyp==1))[0])

            #Slow/fast composition of all interface
            slow_bub_num = len(np.where((partPhase==1) & (partTyp==0))[0]) - slow_int_num
            fast_bub_num = len(np.where((partPhase==1) & (partTyp==1))[0]) - fast_int_num

            #Colors for plotting each phase
            yellow = ("#fdfd96")        #Largest gas-dense interface
            green = ("#77dd77")         #Bulk phase
            red = ("#ff6961")           #Gas phase
            purple = ("#cab2d6")        #Bubble or small gas-dense interfaces


            #If bulk/gas exist, calculate the structure ID for the gas/bulk

            slow_bulk_id_plot = np.where((partPhase==0) & (typ==0))[0]        #Bulk phase structure(s)
            fast_bulk_id_plot = np.where((partPhase==0) & (typ==1))[0]        #Bulk phase structure(s)

            bulk_id_plot = np.where(partPhase==0)[0]        #Bulk phase structure(s)

            slow_int_id_plot = np.where((partPhase==1) & (typ==0))[0]        #Bulk phase structure(s)
            fast_int_id_plot = np.where((partPhase==1) & (typ==1))[0]        #Bulk phase structure(s)
            edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
            int_id_plot = np.where(partPhase==1)[0]         #All interfaces

            slow_bulk_int_id_plot = np.where((partPhase!=2) & (typ==0))[0]        #Bulk phase structure(s)
            fast_bulk_int_id_plot = np.where((partPhase!=2) & (typ==1))[0]        #Bulk phase structure(s)
            bulk_int_id_plot = np.where(partPhase!=2)[0]

            slow_gas_int_id_plot = np.where((partPhase!=0) & (typ==0))[0]        #Bulk phase structure(s)
            fast_gas_int_id_plot = np.where((partPhase!=0) & (typ==1))[0]        #Bulk phase structure(s)
            gas_int_id_plot = np.where(partPhase!=0)[0]

            slow_gas_id_plot = np.where((partPhase==2) & (typ==0))[0]        #Bulk phase structure(s)
            fast_gas_id_plot = np.where((partPhase==2) & (typ==1))[0]        #Bulk phase structure(s)
            gas_id_plot = np.where(partPhase==2)[0]         #All interfaces

            #Find ids of which particles belong to each phase
            bulk_id_plot = np.where(partPhase==0)[0]        #Bulk phase structure(s)
            edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
            int_id_plot = np.where(partPhase==1)[0]         #All interfaces
            bulk_int_id_plot = np.where((partPhase!=2) | (edgePhase==interface_id))[0]

            if len(bulk_ids)>0:
                bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
            else:
                bub_id_plot = []
            gas_id = np.where(partPhase==2)[0]



            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos0_bulk = pos[slow_bulk_id_plot]
            pos0_int = pos[slow_int_id_plot]
            pos0_gas = pos[slow_gas_id_plot]
            pos0_bulk_int = pos[slow_bulk_int_id_plot]
            pos0_gas_int = pos[slow_gas_int_id_plot]

            pos1=pos[typ1ind]
            pos1_bulk = pos[fast_bulk_id_plot]
            pos1_int = pos[fast_int_id_plot]
            pos1_gas = pos[fast_gas_id_plot]
            pos1_bulk_int = pos[fast_bulk_int_id_plot]
            pos1_gas_int = pos[fast_gas_int_id_plot]

            pos_bulk = pos[bulk_id_plot]
            pos_int = pos[int_id_plot]
            pos_gas = pos[gas_id_plot]
            pos_bulk_int = pos[bulk_int_id_plot]
            pos_gas_int = pos[gas_int_id_plot]


            '''
            # Get positions of particles in the largest cluster
            rdf_all_bulk = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAA_bulk = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAB_bulk = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfBB_bulk = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)

            rdf_all_bulk_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAA_bulk_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAB_bulk_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfBB_bulk_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)

            rdf_all_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAA_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAB_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfBB_int = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)

            rdf_all_gas = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAA_gas = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfAB_gas = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)
            rdfBB_gas = freud.density.RDF(bins=nBins, r_max=rstop, r_min=0.1, normalize=True)

            print('bulk')

            rdf_all_bulk.compute(system=system_all_bulk, query_points=f_box.wrap(pos_bulk_int))
            rdfAA_bulk.compute(system=system_A_bulk, query_points=f_box.wrap(pos0_bulk_int))
            rdfAB_bulk.compute(system=system_A_bulk, query_points=f_box.wrap(pos1_bulk_int))
            rdfBB_bulk.compute(system=system_B_bulk, query_points=f_box.wrap(pos1_bulk_int))

            print('dense')

            rdf_all_bulk_int.compute(system=system_all_bulk_int, query_points=f_box.wrap(pos))
            rdfAA_bulk_int.compute(system=system_A_bulk_int, query_points=f_box.wrap(pos0)
            rdfAB_bulk_int.compute(system=system_A_bulk_int, query_points=f_box.wrap(pos1))
            rdfBB_bulk_int.compute(system=system_B_bulk_int, query_points=f_box.wrap(pos1))

            print('int')

            rdf_all_int.compute(system=system_all_int, query_points=f_box.wrap(pos))
            rdfAA_int.compute(system=system_A_int, query_points=f_box.wrap(pos0)
            rdfAB_int.compute(system=system_A_int, query_points=f_box.wrap(pos1))
            rdfBB_int.compute(system=system_B_int, query_points=f_box.wrap(pos1))

            print('gas')

            rdf_all_gas.compute(system=system_all_gas, query_points=f_box.wrap(pos_gas_int))
            rdfAA_gas.compute(system=system_A_gas, query_points=f_box.wrap(pos0_gas_int)
            rdfAB_gas.compute(system=system_A_gas, query_points=f_box.wrap(pos1_gas_int))
            rdfBB_gas.compute(system=system_B_gas, query_points=f_box.wrap(pos1_gas_int))

            print('done')

            plt.plot(r,rdf_bulk.rdf,label='total')
            plt.plot(r,rdfAA_bulk.rdf,label='AA')
            plt.plot(r,rdfAB_bulk.rdf,label='AB')
            plt.plot(r,rdfBA_bulk.rdf,label='BA')
            plt.plot(r,rdfBB_bulk.rdf,label='BB')
            plt.xlabel('r')
            plt.legend()
            plt.xlim(0.1, 10.)
            plt.xlabel(r'$r/a$')
            plt.ylabel(r'$g(r)$')
            plt.tight_layout()
            plt.show()
            '''

            #radialDFall = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
            #radialDFA = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
            #radialDFB = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a

            #query_points=clp_all.centers[large_clust_ind_all]
            #radialDFall.compute(system=system_all_cut,reset=False)               #Calculate radial density function
            '''
            radialDFA.compute(system=system_AA_cut, reset=False)               #Calculate radial density function
            radialDFB.compute(system=system_BB_cut, reset=False)               #Calculate radial density function

            #plt.plot(r, radialDFall.rdf)
            plt.plot(r, radialDFA.rdf)
            plt.plot(r, radialDFB.rdf)
            plt.show()
            stop

            stop
            '''
            if len(bulk_ids)>0:
                bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
            else:
                bub_id_plot = []
            gas_id = np.where(partPhase==2)[0]              #Gas phase structure(s)

            #label previous positions for velocity calculation



            #shift reference frame to center of mass of cluster
            #pos[:,0]= pos[:,0]-com_tmp_posX_temp
            #pos[:,1]= pos[:,1]-com_tmp_posY_temp

            #Ensure particles are within simulation box (periodic boundary conditions)
            #for i in range(0, partNum):
            #        if pos[i,0]>h_box:
            #            pos[i,0]=pos[i,0]-l_box
            #        elif pos[i,0]<-h_box:
            #            pos[i,0]=pos[i,0]+l_box
            #
            #        if pos[i,1]>h_box:
            #            pos[i,1]=pos[i,1]-l_box
            #        elif pos[i,1]<-h_box:
            #            pos[i,1]=pos[i,1]+l_box
            #Measures radius and width of each interface

            new_align = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_x = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_y = [[0 for b in range(NBins)] for a in range(NBins)]
            difr_short_ext = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_int = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_int = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_dif = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad_x = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad_y = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_trad = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad_int = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_trad_int = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_trad = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_trad_x = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_trad_y = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_trad2 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num2 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align1 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num1 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_num_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg_dif_trad = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg0 = [[0 for b in range(NBins)] for a in range(NBins)]
            new_align_avg1 = [[0 for b in range(NBins)] for a in range(NBins)]
            id_step = 0


            #Bin system to calculate orientation and alignment that will be used in vector plots
            NBins = getNBins(l_box, bin_width)
            sizeBin = roundUp(((l_box) / NBins), 6)
            interior_bin=0
            exterior_bin=0
            if bub_large >= 1:

                # Initialize empty arrays
                int_x = np.array([], dtype=int)
                int_y = np.array([], dtype=int)
                ext_x = np.array([], dtype=int)
                ext_y = np.array([], dtype=int)

                int_x_pos = np.array([], dtype=int)
                int_y_pos = np.array([], dtype=int)
                ext_x_pos = np.array([], dtype=int)
                ext_y_pos = np.array([], dtype=int)

                int_bin_unorder_x = np.array([], dtype=int)
                int_bin_unorder_y = np.array([], dtype=int)
                int_bin_unorder_x2 = np.array([], dtype=float)
                int_bin_unorder_y2 = np.array([], dtype=float)

                ext_bin_unorder_x = np.array([], dtype=int)
                ext_bin_unorder_y = np.array([], dtype=int)
                ext_bin_unorder_x2 = np.array([], dtype=float)
                ext_bin_unorder_y2 = np.array([], dtype=float)

                int_bin_unorder_x_copy = np.array([], dtype=int)
                int_bin_unorder_y_copy = np.array([], dtype=int)
                int_bin_unorder_x2_copy = np.array([], dtype=float)
                int_bin_unorder_y2_copy = np.array([], dtype=float)

                ext_bin_unorder_x_copy = np.array([], dtype=int)
                ext_bin_unorder_y_copy = np.array([], dtype=int)
                ext_bin_unorder_x2_copy = np.array([], dtype=float)
                ext_bin_unorder_y2_copy = np.array([], dtype=float)



                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if int_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[0]:
                                interior_bin +=1
                                int_bin_unorder_x_copy = np.append(int_bin_unorder_x_copy, ix)
                                int_bin_unorder_y_copy = np.append(int_bin_unorder_y_copy, iy)
                                int_bin_unorder_x2_copy = np.append(int_bin_unorder_x2_copy, float(ix))
                                int_bin_unorder_y2_copy = np.append(int_bin_unorder_y2_copy, float(iy))
                        if ext_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[0]:
                                exterior_bin +=1
                                ext_bin_unorder_x_copy = np.append(ext_bin_unorder_x_copy, ix)
                                ext_bin_unorder_y_copy = np.append(ext_bin_unorder_y_copy, iy)
                                ext_bin_unorder_x2_copy = np.append(ext_bin_unorder_x2_copy, float(ix))
                                ext_bin_unorder_y2_copy = np.append(ext_bin_unorder_y2_copy, float(iy))


                if interior_bin > 0:
                    if exterior_bin>0:
                        if interior_bin>exterior_bin:
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                        else:
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, int_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, int_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                    else:
                        for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                elif exterior_bin > 0:
                    for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                interior_bin = len(int_bin_unorder_x)
                exterior_bin = len(ext_bin_unorder_x)
                if interior_bin > 0:
                    for ix in range(0, len(int_bin_unorder_x)):
                            int_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=1
                            ext_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=0
                if exterior_bin >0:
                    for ix in range(0, len(ext_bin_unorder_x)):
                            int_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=0
                            ext_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=1
                if interior_bin>0:
                    int_x = np.append(int_x, int_bin_unorder_x[0])
                    int_y = np.append(int_y, int_bin_unorder_y[0])
                if exterior_bin>0:
                    ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                    ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                if interior_bin > 0:
                    ix=int(int_x[0])
                    iy=int(int_y[0])
                    shortest_idx_arr = np.array([])
                    shortest_idy_arr = np.array([])
                    int_bin_unorder_x = np.delete(int_bin_unorder_x, 0)
                    int_bin_unorder_y = np.delete(int_bin_unorder_y, 0)
                    fail=0
                    if len(int_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if int_edge_id[right][iy]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][up]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][down]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][iy]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][up]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][down]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][up]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][down]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                            ix=int_x[1]
                            iy=int_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0
                            while len(int_bin_unorder_x)>0:
                                    current_size = len(int_bin_unorder_x)

                                    if past_size == current_size:
                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if int_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[0]:
                                                            loc_id = np.where((int_bin_unorder_x == ix6) & (int_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if int_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[0]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            int_x = np.append(int_x, ix)
                                            int_y = np.append(int_y, iy)

                                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[0]:
                                        if int_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((int_bin_unorder_x == ix2) & (int_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if int_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[0]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if int_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[0]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size

                        # append the starting x,y coordinates
                        #int_x = np.r_[int_x, int_x[0]]
                        #int_y = np.r_[int_y, int_y[0]]






                    for m in range(0, len(int_x)):
                        int_x_pos = np.append(int_x_pos, int_x[m] * sizeBin)
                        int_y_pos = np.append(int_y_pos, int_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, int_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[0])
                    adjacent_y = np.append(adjacent_y, int_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[0])
                    if len(int_x)>1:
                        for m in range(1, len(int_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, int_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                adjacent_y = np.append(adjacent_y, int_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                            else:
                                difx = int_x_pos[m]-int_x_pos[m-1]
                                dify = int_y_pos[m]-int_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        int_x_pos[m:-1] += l_box
                                        int_x[m:-1] += NBins
                                    else:
                                        int_x_pos[m:-1] -= l_box
                                        int_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        int_y_pos[m:-1] += l_box
                                        int_y[m:-1] += NBins
                                    else:
                                        int_y_pos[m:-1] -= l_box
                                        int_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, int_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, int_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                                    if (m==len(int_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])

                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])


                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        int_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        int_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(int_x)==3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, k=2, per=True)
                        elif len(int_x)>3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, per=True)
                        if len(int_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                            jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                            smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                            xn, yn = xi[:-1], yi[:-1]
                            xn = xn[(jump > 0) & (smooth_jump < limit)]
                            yn = yn[(jump > 0) & (smooth_jump < limit)]

                            xn_pos = np.copy(xn)
                            yn_pos = np.copy(yn)
                            xn_pos_non_per = np.copy(xn)
                            yn_pos_non_per = np.copy(yn)


                            for m in range(0, len(xn)):
                                xn_pos[m] = xn[m] * sizeBin
                                yn_pos[m] = yn[m] * sizeBin
                                xn_pos_non_per[m] = xn[m] * sizeBin
                                yn_pos_non_per[m] = yn[m] * sizeBin

                                if xn[m] < 0:
                                    xn[m]+=NBins
                                if xn[m]>=NBins:
                                    xn[m]-=NBins

                                if yn[m] < 0:
                                    yn[m]+=NBins
                                if yn[m]>=NBins:
                                    yn[m]-=NBins

                                if xn_pos[m] < 0:
                                    xn_pos[m]+=l_box
                                if xn_pos[m]>=l_box:
                                    xn_pos[m]-=l_box

                                if yn_pos[m] < 0:
                                    yn_pos[m]+=l_box
                                if yn_pos[m]>=l_box:
                                    yn_pos[m]-=l_box
                        else:
                            xn = np.zeros(1)
                            yn = np.zeros(1)
                            xn_pos = np.zeros(1)
                            yn_pos = np.zeros(1)
                            xn_pos_non_per = np.zeros(1)
                            yn_pos_non_per = np.zeros(1)

                            xn_pos[0] = int_x[0]
                            yn_pos[0] = int_y[0]
                            xn_pos[0] = int_x[0] * sizeBin
                            yn_pos[0] = int_y[0] * sizeBin
                            xn_pos_non_per[0] = int_x[0] * sizeBin
                            yn_pos_non_per[0] = int_y[0] * sizeBin
                            if xn[0] < 0:
                                xn[0]+=NBins
                            if xn[0]>=NBins:
                                xn[0]-=NBins

                            if yn[0] < 0:
                                yn[0]+=NBins
                            if yn[0]>=NBins:
                                yn[0]-=NBins

                            if xn_pos[0] < 0:
                                xn_pos[0]+=l_box
                            if xn_pos[0]>=l_box:
                                xn_pos[0]-=l_box

                            if yn_pos[0] < 0:
                                yn_pos[0]+=l_box
                            if yn_pos[0]>=l_box:
                                yn_pos[0]-=l_box

                    else:
                        xn=np.array([int_x[0]])
                        yn=np.array([int_y[0]])
                        xn_pos = np.copy(xn)
                        yn_pos = np.copy(yn)
                        xn_pos_non_per = np.copy(xn)
                        yn_pos_non_per = np.copy(yn)
                        for m in range(0, len(xn)):
                            xn_pos[m] = xn[m] * sizeBin
                            yn_pos[m] = yn[m] * sizeBin
                            xn_pos_non_per[m] = xn[m] * sizeBin
                            yn_pos_non_per[m] = yn[m] * sizeBin

                            if xn[m] < 0:
                                xn[m]+=NBins
                            if xn[m]>=NBins:
                                xn[m]-=NBins

                            if yn[m] < 0:
                                yn[m]+=NBins
                            if yn[m]>=NBins:
                                yn[m]-=NBins

                            if xn_pos[m] < 0:
                                xn_pos[m]+=l_box
                            if xn_pos[m]>=l_box:
                                xn_pos[m]-=l_box

                            if yn_pos[m] < 0:
                                yn_pos[m]+=l_box
                            if yn_pos[m]>=l_box:
                                yn_pos[m]-=l_box

                if exterior_bin > 0:
                    ix=int(ext_x[0])
                    iy=int(ext_y[0])

                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                    fail=0
                    if len(ext_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if ext_edge_id[right][iy]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][up]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][down]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][iy]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][up]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][down]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][up]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][down]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:

                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=ext_x[1]
                            iy=ext_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0

                            while len(ext_bin_unorder_x)>0:
                                    current_size = len(ext_bin_unorder_x)

                                    if past_size == current_size:

                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if ext_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[0]:
                                                            loc_id = np.where((ext_bin_unorder_x == ix6) & (ext_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if ext_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[0]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]
                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]


                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            ext_x = np.append(ext_x, ix)
                                            ext_y = np.append(ext_y, iy)

                                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[0]:
                                        if ext_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((ext_bin_unorder_x == ix2) & (ext_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if ext_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[0]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if ext_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[0]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size


                    # append the starting x,y coordinates


                    # append the starting x,y coordinates
                    #int_x = np.r_[int_x, int_x[0]]
                    #int_y = np.r_[int_y, int_y[0]]
                    for m in range(0, len(ext_x)):
                        ext_x_pos = np.append(ext_x_pos, ext_x[m] * sizeBin)
                        ext_y_pos = np.append(ext_y_pos, ext_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, ext_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[0])
                    adjacent_y = np.append(adjacent_y, ext_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[0])
                    if len(ext_x)>1:
                        for m in range(1, len(ext_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, ext_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                adjacent_y = np.append(adjacent_y, ext_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                            else:
                                difx = ext_x_pos[m]-ext_x_pos[m-1]
                                dify = ext_y_pos[m]-ext_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        ext_x_pos[m:-1] += l_box
                                        ext_x[m:-1] += NBins
                                    else:
                                        ext_x_pos[m:-1] -= l_box
                                        ext_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        ext_y_pos[m:-1] += l_box
                                        ext_y[m:-1] += NBins
                                    else:
                                        ext_y_pos[m:-1] -= l_box
                                        ext_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, ext_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, ext_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                                    if (m==len(ext_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])

                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        ext_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        ext_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(ext_x)==3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, k=2, per=True)
                        elif len(ext_x)>3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, per=True)
                        if len(ext_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi2, yi2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                            jump2 = np.sqrt(np.diff(xi2)**2 + np.diff(yi2)**2)
                            smooth_jump2 = ndimage.gaussian_filter1d(jump2, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit2 = 2*np.median(smooth_jump2)    # factor 2 is arbitrary
                            xn2, yn2 = xi2[:-1], yi2[:-1]
                            xn2 = xn2[(jump2 > 0) & (smooth_jump2 < limit2)]
                            yn2 = yn2[(jump2 > 0) & (smooth_jump2 < limit2)]

                            xn2_pos = np.copy(xn2)
                            yn2_pos = np.copy(yn2)
                            xn2_pos_non_per = np.copy(xn2)
                            yn2_pos_non_per = np.copy(yn2)


                            for m in range(0, len(xn2)):
                                xn2_pos[m] = xn2[m] * sizeBin
                                yn2_pos[m] = yn2[m] * sizeBin
                                xn2_pos_non_per[m] = xn2[m] * sizeBin
                                yn2_pos_non_per[m] = yn2[m] * sizeBin

                                if xn2[m] < 0:
                                    xn2[m]+=NBins
                                if xn2[m]>=NBins:
                                    xn2[m]-=NBins

                                if yn2[m] < 0:
                                    yn2[m]+=NBins
                                if yn2[m]>=NBins:
                                    yn2[m]-=NBins

                                if xn2_pos[m] < 0:
                                    xn2_pos[m]+=l_box
                                if xn2_pos[m]>=l_box:
                                    xn2_pos[m]-=l_box

                                if yn2_pos[m] < 0:
                                    yn2_pos[m]+=l_box
                                if yn2_pos[m]>=l_box:
                                    yn2_pos[m]-=l_box
                        else:
                            xn2 = np.zeros(1)
                            yn2 = np.zeros(1)
                            xn2_pos = np.zeros(1)
                            yn2_pos = np.zeros(1)
                            xn2_pos_non_per = np.zeros(1)
                            yn2_pos_non_per = np.zeros(1)

                            xn2_pos[0] = ext_x[0]
                            yn2_pos[0] = ext_y[0]
                            xn2_pos[0] = ext_x[0] * sizeBin
                            yn2_pos[0] = ext_y[0] * sizeBin
                            xn2_pos_non_per[0] = ext_x[0] * sizeBin
                            yn2_pos_non_per[0] = ext_y[0] * sizeBin
                            if xn2[0] < 0:
                                xn2[0]+=NBins
                            if xn2[0]>=NBins:
                                xn2[0]-=NBins

                            if yn2[0] < 0:
                                yn2[0]+=NBins
                            if yn2[0]>=NBins:
                                yn2[0]-=NBins

                            if xn2_pos[0] < 0:
                                xn2_pos[0]+=l_box
                            if xn2_pos[0]>=l_box:
                                xn2_pos[0]-=l_box

                            if yn2_pos[0] < 0:
                                yn2_pos[0]+=l_box
                            if yn2_pos[0]>=l_box:
                                yn2_pos[0]-=l_box
                    else:
                        xn2=np.array([ext_x[0]])
                        yn2=np.array([ext_y[0]])
                        xn2_pos = np.copy(xn2)
                        yn2_pos = np.copy(yn2)
                        xn2_pos_non_per = np.copy(xn2)
                        yn2_pos_non_per = np.copy(yn2)
                        for m in range(0, len(xn2)):
                            xn2_pos[m] = xn2[m] * sizeBin
                            yn2_pos[m] = yn2[m] * sizeBin
                            xn2_pos_non_per[m] = xn2[m] * sizeBin
                            yn2_pos_non_per[m] = yn2[m] * sizeBin

                            if xn2[m] < 0:
                                xn2[m]+=NBins
                            if xn2[m]>=NBins:
                                xn2[m]-=NBins

                            if yn2[m] < 0:
                                yn2[m]+=NBins
                            if yn2[m]>=NBins:
                                yn2[m]-=NBins

                            if xn2_pos[m] < 0:
                                xn2_pos[m]+=l_box
                            if xn2_pos[m]>=l_box:
                                xn2_pos[m]-=l_box

                            if yn2_pos[m] < 0:
                                yn2_pos[m]+=l_box
                            if yn2_pos[m]>=l_box:
                                yn2_pos[m]-=l_box

            interior_bin_bub1=0
            exterior_bin_bub1=0
            if bub_large >= 2:
                # Initialize empty arrays
                int_x = np.array([], dtype=int)
                int_y = np.array([], dtype=int)
                ext_x = np.array([], dtype=int)
                ext_y = np.array([], dtype=int)

                int_x_pos = np.array([], dtype=int)
                int_y_pos = np.array([], dtype=int)
                ext_x_pos = np.array([], dtype=int)
                ext_y_pos = np.array([], dtype=int)

                int_bin_unorder_x_copy = np.array([], dtype=int)
                int_bin_unorder_y_copy = np.array([], dtype=int)
                int_bin_unorder_x2_copy = np.array([], dtype=float)
                int_bin_unorder_y2_copy = np.array([], dtype=float)

                ext_bin_unorder_x_copy = np.array([], dtype=int)
                ext_bin_unorder_y_copy = np.array([], dtype=int)
                ext_bin_unorder_x2_copy = np.array([], dtype=float)
                ext_bin_unorder_y2_copy = np.array([], dtype=float)

                int_bin_unorder_x = np.array([], dtype=int)
                int_bin_unorder_y = np.array([], dtype=int)
                int_bin_unorder_x2 = np.array([], dtype=float)
                int_bin_unorder_y2 = np.array([], dtype=float)

                ext_bin_unorder_x = np.array([], dtype=int)
                ext_bin_unorder_y = np.array([], dtype=int)
                ext_bin_unorder_x2 = np.array([], dtype=float)
                ext_bin_unorder_y2 = np.array([], dtype=float)



                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if int_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[1]:
                                interior_bin_bub1 +=1
                                int_bin_unorder_x_copy = np.append(int_bin_unorder_x_copy, ix)
                                int_bin_unorder_y_copy = np.append(int_bin_unorder_y_copy, iy)
                                int_bin_unorder_x2_copy = np.append(int_bin_unorder_x2_copy, float(ix))
                                int_bin_unorder_y2_copy = np.append(int_bin_unorder_y2_copy, float(iy))
                        if ext_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[1]:
                                exterior_bin_bub1 +=1
                                ext_bin_unorder_x_copy = np.append(ext_bin_unorder_x_copy, ix)
                                ext_bin_unorder_y_copy = np.append(ext_bin_unorder_y_copy, iy)
                                ext_bin_unorder_x2_copy = np.append(ext_bin_unorder_x2_copy, float(ix))
                                ext_bin_unorder_y2_copy = np.append(ext_bin_unorder_y2_copy, float(iy))

                if interior_bin_bub1 > 0:
                    if exterior_bin_bub1>0:
                        if interior_bin_bub1>exterior_bin_bub1:
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                        else:
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, int_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, int_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                    else:
                        for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                elif exterior_bin_bub1 > 0:
                    for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                interior_bin_bub1 = len(int_bin_unorder_x)
                exterior_bin_bub1 = len(ext_bin_unorder_x)
                for ix in range(0, len(int_bin_unorder_x)):
                        int_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=1
                        ext_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=0
                for ix in range(0, len(ext_bin_unorder_x)):
                        int_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=0
                        ext_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=1
                if interior_bin_bub1>0:
                    int_x = np.append(int_x, int_bin_unorder_x[0])
                    int_y = np.append(int_y, int_bin_unorder_y[0])
                if exterior_bin_bub1>0:
                    ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                    ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                if interior_bin_bub1>0:
                    ix=int(int_x[0])
                    iy=int(int_y[0])
                    int_bin_unorder_x = np.delete(int_bin_unorder_x, 0)
                    int_bin_unorder_y = np.delete(int_bin_unorder_y, 0)
                    fail=0
                    if len(int_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if int_edge_id[right][iy]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][up]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][down]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][iy]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][up]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][down]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][up]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][down]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=int_x[1]
                            iy=int_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0
                            while len(int_bin_unorder_x)>0:
                                    current_size = len(int_bin_unorder_x)

                                    if past_size == current_size:
                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if int_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[1]:
                                                            loc_id = np.where((int_bin_unorder_x == ix6) & (int_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if int_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[1]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            int_x = np.append(int_x, ix)
                                            int_y = np.append(int_y, iy)

                                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[1]:
                                        if int_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((int_bin_unorder_x == ix2) & (int_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if int_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[1]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if int_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[1]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size


                    # append the starting x,y coordinates
                    #int_x = np.r_[int_x, int_x[0]]
                    #int_y = np.r_[int_y, int_y[0]]
                    for m in range(0, len(int_x)):
                        int_x_pos = np.append(int_x_pos, int_x[m] * sizeBin)
                        int_y_pos = np.append(int_y_pos, int_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, int_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[0])
                    adjacent_y = np.append(adjacent_y, int_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[0])
                    if len(int_x)>1:
                        for m in range(1, len(int_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, int_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                adjacent_y = np.append(adjacent_y, int_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                            else:
                                difx = int_x_pos[m]-int_x_pos[m-1]
                                dify = int_y_pos[m]-int_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        int_x_pos[m:-1] += l_box
                                        int_x[m:-1] += NBins
                                    else:
                                        int_x_pos[m:-1] -= l_box
                                        int_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        int_y_pos[m:-1] += l_box
                                        int_y[m:-1] += NBins
                                    else:
                                        int_y_pos[m:-1] -= l_box
                                        int_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, int_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, int_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                                    if (m==len(int_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])


                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        int_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        int_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]


                        if len(int_x)==3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, k=2, per=True)
                        elif len(int_x)>3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, per=True)
                        if len(int_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                            jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                            smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                            xn_bub2, yn_bub2 = xi[:-1], yi[:-1]
                            xn_bub2 = xn_bub2[(jump > 0) & (smooth_jump < limit)]
                            yn_bub2 = yn_bub2[(jump > 0) & (smooth_jump < limit)]
                            xn_bub2_pos = np.copy(xn_bub2)
                            yn_bub2_pos = np.copy(yn_bub2)
                            xn_bub2_pos_non_per = np.copy(xn_bub2)
                            yn_bub2_pos_non_per = np.copy(yn_bub2)



                            for m in range(0, len(xn_bub2)):
                                xn_bub2_pos[m] = xn_bub2[m] * sizeBin
                                yn_bub2_pos[m] = yn_bub2[m] * sizeBin
                                xn_bub2_pos_non_per[m] = xn_bub2[m] * sizeBin
                                yn_bub2_pos_non_per[m] = yn_bub2[m] * sizeBin

                                if xn_bub2[m] < 0:
                                    xn_bub2[m]+=NBins
                                if xn_bub2[m]>=NBins:
                                    xn_bub2[m]-=NBins

                                if yn_bub2[m] < 0:
                                    yn_bub2[m]+=NBins
                                if yn_bub2[m]>=NBins:
                                    yn_bub2[m]-=NBins

                                if xn_bub2_pos[m] < 0:
                                    xn_bub2_pos[m]+=l_box
                                if xn_bub2_pos[m]>=l_box:
                                    xn_bub2_pos[m]-=l_box

                                if yn_bub2_pos[m] < 0:
                                    yn_bub2_pos[m]+=l_box
                                if yn_bub2_pos[m]>=l_box:
                                    yn_bub2_pos[m]-=l_box
                        else:
                            xn_bub2 = np.zeros(1)
                            yn_bub2 = np.zeros(1)
                            xn_bub2_pos = np.zeros(1)
                            yn_bub2_pos = np.zeros(1)
                            xn_bub2_pos_non_per = np.zeros(1)
                            yn_bub2_pos_non_per = np.zeros(1)

                            xn_bub2_pos[0] = int_x[0]
                            yn_bub2_pos[0] = int_y[0]
                            xn_bub2_pos[0] = int_x[0] * sizeBin
                            yn_bub2_pos[0] = int_y[0] * sizeBin
                            xn_bub2_pos_non_per[0] = int_x[0] * sizeBin
                            yn_bub2_pos_non_per[0] = int_y[0] * sizeBin
                            if xn_bub2[0] < 0:
                                xn_bub2[0]+=NBins
                            if xn_bub2[0]>=NBins:
                                xn_bub2[0]-=NBins

                            if yn_bub2[0] < 0:
                                yn_bub2[0]+=NBins
                            if yn_bub2[0]>=NBins:
                                yn_bub2[0]-=NBins

                            if xn_bub2_pos[0] < 0:
                                xn_bub2_pos[0]+=l_box
                            if xn_bub2_pos[0]>=l_box:
                                xn_bub2_pos[0]-=l_box

                            if yn_bub2_pos[0] < 0:
                                yn_bub2_pos[0]+=l_box
                            if yn_bub2_pos[0]>=l_box:
                                yn_bub2_pos[0]-=l_box
                    else:
                        xn_bub2 = np.array([int_x[0]])
                        yn_bub2 = np.array([int_y[0]])
                        xn_bub2_pos = np.copy(xn_bub2)
                        yn_bub2_pos = np.copy(yn_bub2)
                        xn_bub2_pos_non_per = np.copy(xn_bub2)
                        yn_bub2_pos_non_per = np.copy(yn_bub2)



                        for m in range(0, len(xn_bub2)):
                            xn_bub2_pos[m] = xn_bub2[m] * sizeBin
                            yn_bub2_pos[m] = yn_bub2[m] * sizeBin
                            xn_bub2_pos_non_per[m] = xn_bub2[m] * sizeBin
                            yn_bub2_pos_non_per[m] = yn_bub2[m] * sizeBin

                            if xn_bub2[m] < 0:
                                xn_bub2[m]+=NBins
                            if xn_bub2[m]>=NBins:
                                xn_bub2[m]-=NBins

                            if yn_bub2[m] < 0:
                                yn_bub2[m]+=NBins
                            if yn_bub2[m]>=NBins:
                                yn_bub2[m]-=NBins

                            if xn_bub2_pos[m] < 0:
                                xn_bub2_pos[m]+=l_box
                            if xn_bub2_pos[m]>=l_box:
                                xn_bub2_pos[m]-=l_box

                            if yn_bub2_pos[m] < 0:
                                yn_bub2_pos[m]+=l_box
                            if yn_bub2_pos[m]>=l_box:
                                yn_bub2_pos[m]-=l_box



                if exterior_bin_bub1>0:
                    ix=int(ext_x[0])
                    iy=int(ext_y[0])

                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                    fail=0
                    if len(ext_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if ext_edge_id[right][iy]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][up]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][down]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][iy]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][up]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][down]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][up]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][down]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=ext_x[1]
                            iy=ext_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0

                            while len(ext_bin_unorder_x)>0:
                                    current_size = len(ext_bin_unorder_x)

                                    if past_size == current_size:

                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if ext_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[1]:
                                                            loc_id = np.where((ext_bin_unorder_x == ix6) & (ext_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if ext_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[1]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]
                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]


                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            ext_x = np.append(ext_x, ix)
                                            ext_y = np.append(ext_y, iy)

                                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[1]:
                                        if ext_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((ext_bin_unorder_x == ix2) & (ext_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if ext_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[1]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if ext_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[1]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size




                    for m in range(0, len(ext_x)):
                        ext_x_pos = np.append(ext_x_pos, ext_x[m] * sizeBin)
                        ext_y_pos = np.append(ext_y_pos, ext_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = []#np.array([])
                    adjacent_x_arr_pos = []#np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = []#np.array([])
                    adjacent_y_arr_pos = []#np.array([])
                    adjacent_x = np.append(adjacent_x, ext_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[0])
                    adjacent_y = np.append(adjacent_y, ext_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[0])
                    if len(ext_x)>1:
                        for m in range(1, len(ext_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, ext_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                adjacent_y = np.append(adjacent_y, ext_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                            else:
                                difx = ext_x_pos[m]-ext_x_pos[m-1]
                                dify = ext_y_pos[m]-ext_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        ext_x_pos[m:-1] += l_box
                                        ext_x[m:-1] += NBins
                                    else:
                                        ext_x_pos[m:-1] -= l_box
                                        ext_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        ext_y_pos[m:-1] += l_box
                                        ext_y[m:-1] += NBins
                                    else:
                                        ext_y_pos[m:-1] -= l_box
                                        ext_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):

                                    adjacent_x_arr.append(adjacent_x)
                                    adjacent_x_arr_pos.append(adjacent_x_pos)
                                    adjacent_y_arr.append(adjacent_y)
                                    adjacent_y_arr_pos.append(adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])

                                else:
                                    adjacent_x = np.append(adjacent_x, ext_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, ext_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                                    if (m==len(ext_x)-1):
                                        adjacent_x_arr.append(adjacent_x)
                                        adjacent_x_arr_pos.append(adjacent_x_pos)
                                        adjacent_y_arr.append(adjacent_y)
                                        adjacent_y_arr_pos.append(adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])
                        # append the starting x,y coordinates
                        #int_x = np.r_[int_x, int_x[0]]
                        #int_y = np.r_[int_y, int_y[0]]
                        xn2_bub2_arr = []
                        yn2_bub2_arr = []
                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        ext_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        ext_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]
                        print(ext_x)
                        print(ext_y)


                        if len(ext_x)==3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, k=2, per=True)
                        elif len(ext_x)>3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, per=True)
                        if len(ext_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi2, yi2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                            jump2 = np.sqrt(np.diff(xi2)**2 + np.diff(yi2)**2)
                            smooth_jump2 = ndimage.gaussian_filter1d(jump2, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit2 = 2*np.median(smooth_jump2)    # factor 2 is arbitrary
                            xn2_bub2, yn2_bub2 = xi2[:-1], yi2[:-1]
                            xn2_bub2 = xn2_bub2[(jump2 > 0) & (smooth_jump2 < limit2)]
                            yn2_bub2 = yn2_bub2[(jump2 > 0) & (smooth_jump2 < limit2)]
                            xn2_bub2_pos = np.copy(xn2_bub2)
                            yn2_bub2_pos = np.copy(yn2_bub2)
                            xn2_bub2_pos_non_per = np.copy(xn2_bub2)
                            yn2_bub2_pos_non_per = np.copy(yn2_bub2)



                            for m in range(0, len(xn2_bub2)):
                                xn2_bub2_pos[m] = xn2_bub2[m] * sizeBin
                                yn2_bub2_pos[m] = yn2_bub2[m] * sizeBin
                                xn2_bub2_pos_non_per[m] = xn2_bub2[m] * sizeBin
                                yn2_bub2_pos_non_per[m] = yn2_bub2[m] * sizeBin

                                if xn2_bub2[m] < 0:
                                    xn2_bub2[m]+=NBins
                                if xn2_bub2[m]>=NBins:
                                    xn2_bub2[m]-=NBins

                                if yn2_bub2[m] < 0:
                                    yn2_bub2[m]+=NBins
                                if yn2_bub2[m]>=NBins:
                                    yn2_bub2[m]-=NBins

                                if xn2_bub2_pos[m] < 0:
                                    xn2_bub2_pos[m]+=l_box
                                if xn2_bub2_pos[m]>=l_box:
                                    xn2_bub2_pos[m]-=l_box

                                if yn2_bub2_pos[m] < 0:
                                    yn2_bub2_pos[m]+=l_box
                                if yn2_bub2_pos[m]>=l_box:
                                    yn2_bub2_pos[m]-=l_box
                        else:
                            xn2_bub2 = np.zeros(1)
                            yn2_bub2 = np.zeros(1)
                            xn2_bub2_pos = np.zeros(1)
                            yn2_bub2_pos = np.zeros(1)
                            xn2_bub2_pos_non_per = np.zeros(1)
                            yn2_bub2_pos_non_per = np.zeros(1)

                            xn2_bub2_pos[0] = ext_x[0]
                            yn2_bub2_pos[0] = ext_y[0]
                            xn2_bub2_pos[0] = ext_x[0] * sizeBin
                            yn2_bub2_pos[0] = ext_y[0] * sizeBin
                            xn2_bub2_pos_non_per[0] = ext_x[0] * sizeBin
                            yn2_bub2_pos_non_per[0] = ext_y[0] * sizeBin
                            if xn2_bub2[0] < 0:
                                xn2_bub2[0]+=NBins
                            if xn2_bub2[0]>=NBins:
                                xn2_bub2[0]-=NBins

                            if yn2_bub2[0] < 0:
                                yn2_bub2[0]+=NBins
                            if yn2_bub2[0]>=NBins:
                                yn2_bub2[0]-=NBins

                            if xn2_bub2_pos[0] < 0:
                                xn2_bub2_pos[0]+=l_box
                            if xn2_bub2_pos[0]>=l_box:
                                xn2_bub2_pos[0]-=l_box

                            if yn2_bub2_pos[0] < 0:
                                yn2_bub2_pos[0]+=l_box
                            if yn2_bub2_pos[0]>=l_box:
                                yn2_bub2_pos[0]-=l_box


                    else:
                        xn2_bub2 = np.array([ext_x[0]])
                        yn2_bub2 = np.array([ext_y[0]])
                        xn2_bub2_pos = np.copy(xn2_bub2)
                        yn2_bub2_pos = np.copy(yn2_bub2)
                        xn2_bub2_pos_non_per = np.copy(xn2_bub2)
                        yn2_bub2_pos_non_per = np.copy(yn2_bub2)



                        for m in range(0, len(xn2_bub2)):
                            xn2_bub2_pos[m] = xn2_bub2[m] * sizeBin
                            yn2_bub2_pos[m] = yn2_bub2[m] * sizeBin
                            xn2_bub2_pos_non_per[m] = xn2_bub2[m] * sizeBin
                            yn2_bub2_pos_non_per[m] = yn2_bub2[m] * sizeBin

                            if xn2_bub2[m] < 0:
                                xn2_bub2[m]+=NBins
                            if xn2_bub2[m]>=NBins:
                                xn2_bub2[m]-=NBins

                            if yn2_bub2[m] < 0:
                                yn2_bub2[m]+=NBins
                            if yn2_bub2[m]>=NBins:
                                yn2_bub2[m]-=NBins

                            if xn2_bub2_pos[m] < 0:
                                xn2_bub2_pos[m]+=l_box
                            if xn2_bub2_pos[m]>=l_box:
                                xn2_bub2_pos[m]-=l_box

                            if yn2_bub2_pos[m] < 0:
                                yn2_bub2_pos[m]+=l_box
                            if yn2_bub2_pos[m]>=l_box:
                                yn2_bub2_pos[m]-=l_box
            interior_bin_bub2=0
            exterior_bin_bub2=0
            if bub_large >=3:
                # Initialize empty arrays
                int_x = np.array([], dtype=int)
                int_y = np.array([], dtype=int)
                ext_x = np.array([], dtype=int)
                ext_y = np.array([], dtype=int)

                int_x_pos = np.array([], dtype=int)
                int_y_pos = np.array([], dtype=int)
                ext_x_pos = np.array([], dtype=int)
                ext_y_pos = np.array([], dtype=int)

                int_bin_unorder_x = np.array([], dtype=int)
                int_bin_unorder_y = np.array([], dtype=int)
                int_bin_unorder_x2 = np.array([], dtype=float)
                int_bin_unorder_y2 = np.array([], dtype=float)

                ext_bin_unorder_x = np.array([], dtype=int)
                ext_bin_unorder_y = np.array([], dtype=int)
                ext_bin_unorder_x2 = np.array([], dtype=float)
                ext_bin_unorder_y2 = np.array([], dtype=float)

                int_bin_unorder_x_copy = np.array([], dtype=int)
                int_bin_unorder_y_copy = np.array([], dtype=int)
                int_bin_unorder_x2_copy = np.array([], dtype=float)
                int_bin_unorder_y2_copy = np.array([], dtype=float)

                ext_bin_unorder_x_copy = np.array([], dtype=int)
                ext_bin_unorder_y_copy = np.array([], dtype=int)
                ext_bin_unorder_x2_copy = np.array([], dtype=float)
                ext_bin_unorder_y2_copy = np.array([], dtype=float)



                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if int_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[2]:
                                interior_bin_bub2 +=1
                                int_bin_unorder_x_copy = np.append(int_bin_unorder_x_copy, ix)
                                int_bin_unorder_y_copy = np.append(int_bin_unorder_y_copy, iy)
                                int_bin_unorder_x2_copy = np.append(int_bin_unorder_x2_copy, float(ix))
                                int_bin_unorder_y2_copy = np.append(int_bin_unorder_y2_copy, float(iy))
                        if ext_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[2]:
                                exterior_bin_bub2 +=1
                                ext_bin_unorder_x_copy = np.append(ext_bin_unorder_x_copy, ix)
                                ext_bin_unorder_y_copy = np.append(ext_bin_unorder_y_copy, iy)
                                ext_bin_unorder_x2_copy = np.append(ext_bin_unorder_x2_copy, float(ix))
                                ext_bin_unorder_y2_copy = np.append(ext_bin_unorder_y2_copy, float(iy))


                if interior_bin_bub2 > 0:
                    if exterior_bin_bub2>0:
                        if interior_bin_bub2>exterior_bin_bub2:
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                        else:
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, int_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, int_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                    else:
                        for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                elif exterior_bin_bub2 > 0:
                    for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                interior_bin_bub2 = len(int_bin_unorder_x)
                exterior_bin_bub2 = len(ext_bin_unorder_x)
                for ix in range(0, len(int_bin_unorder_x)):
                    for iy in range(0, len(int_bin_unorder_y)):
                        int_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=1
                        ext_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=0
                for ix in range(0, len(ext_bin_unorder_x)):
                    for iy in range(0, len(ext_bin_unorder_y)):
                        int_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=0
                        ext_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=1
                if interior_bin_bub2>0:
                    int_x = np.append(int_x, int_bin_unorder_x[0])
                    int_y = np.append(int_y, int_bin_unorder_y[0])
                if exterior_bin_bub2>0:
                    ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                    ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                if interior_bin_bub2>0:
                    ix=int(int_x[0])
                    iy=int(int_y[0])

                    int_bin_unorder_x = np.delete(int_bin_unorder_x, 0)
                    int_bin_unorder_y = np.delete(int_bin_unorder_y, 0)
                    fail=0
                    if len(int_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if int_edge_id[right][iy]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][up]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][down]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][iy]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][up]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][down]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][up]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][down]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=int_x[1]
                            iy=int_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0
                            while len(int_bin_unorder_x)>0:
                                    current_size = len(int_bin_unorder_x)

                                    if past_size == current_size:
                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if int_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[2]:
                                                            loc_id = np.where((int_bin_unorder_x == ix6) & (int_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if int_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[2]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            int_x = np.append(int_x, ix)
                                            int_y = np.append(int_y, iy)

                                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[2]:
                                        if int_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((int_bin_unorder_x == ix2) & (int_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if int_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[2]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if int_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[2]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size
                        # append the starting x,y coordinates
                        #int_x = np.r_[int_x, int_x[0]]
                        #int_y = np.r_[int_y, int_y[0]]

                    for m in range(0, len(int_x)):
                        int_x_pos = np.append(int_x_pos, int_x[m] * sizeBin)
                        int_y_pos = np.append(int_y_pos, int_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, int_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[0])
                    adjacent_y = np.append(adjacent_y, int_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[0])
                    if len(int_x)>1:
                        for m in range(1, len(int_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, int_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                adjacent_y = np.append(adjacent_y, int_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                            else:
                                difx = int_x_pos[m]-int_x_pos[m-1]
                                dify = int_y_pos[m]-int_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        int_x_pos[m:-1] += l_box
                                        int_x[m:-1] += NBins
                                    else:
                                        int_x_pos[m:-1] -= l_box
                                        int_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        int_y_pos[m:-1] += l_box
                                        int_y[m:-1] += NBins
                                    else:
                                        int_y_pos[m:-1] -= l_box
                                        int_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, int_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, int_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                                    if (m==len(int_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])


                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        int_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        int_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(int_x)==3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, k=2, per=True)
                        elif len(int_x)>3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, per=True)

                        if len(int_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                            jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                            smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                            xn_bub3, yn_bub3 = xi[:-1], yi[:-1]
                            xn_bub3 = xn_bub3[(jump > 0) & (smooth_jump < limit)]
                            yn_bub3 = yn_bub3[(jump > 0) & (smooth_jump < limit)]

                            xn_bub3_pos = np.copy(xn_bub3)
                            yn_bub3_pos = np.copy(yn_bub3)
                            xn_bub3_pos_non_per = np.copy(xn_bub3)
                            yn_bub3_pos_non_per = np.copy(yn_bub3)


                            for m in range(0, len(xn_bub3)):
                                xn_bub3_pos[m] = xn_bub3[m] * sizeBin
                                yn_bub3_pos[m] = yn_bub3[m] * sizeBin
                                xn_bub3_pos_non_per[m] = xn_bub3[m] * sizeBin
                                yn_bub3_pos_non_per[m] = yn_bub3[m] * sizeBin

                                if xn_bub3[m] < 0:
                                    xn_bub3[m]+=NBins
                                if xn_bub3[m]>=NBins:
                                    xn_bub3[m]-=NBins

                                if yn_bub3[m] < 0:
                                    yn_bub3[m]+=NBins
                                if yn_bub3[m]>=NBins:
                                    yn_bub3[m]-=NBins

                                if xn_bub3_pos[m] < 0:
                                    xn_bub3_pos[m]+=l_box
                                if xn_bub3_pos[m]>=l_box:
                                    xn_bub3_pos[m]-=l_box

                                if yn_bub3_pos[m] < 0:
                                    yn_bub3_pos[m]+=l_box
                                if yn_bub3_pos[m]>=l_box:
                                    yn_bub3_pos[m]-=l_box
                        else:
                            xn_bub3 = np.zeros(1)
                            yn_bub3 = np.zeros(1)
                            xn_bub3_pos = np.zeros(1)
                            yn_bub3_pos = np.zeros(1)
                            xn_bub3_pos_non_per = np.zeros(1)
                            yn_bub3_pos_non_per = np.zeros(1)

                            xn_bub3_pos[0] = int_x[0]
                            yn_bub3_pos[0] = int_y[0]
                            xn_bub3_pos[0] = int_x[0] * sizeBin
                            yn_bub3_pos[0] = int_y[0] * sizeBin
                            xn_bub3_pos_non_per[0] = int_x[0] * sizeBin
                            yn_bub3_pos_non_per[0] = int_y[0] * sizeBin

                            if xn_bub3[0] < 0:
                                xn_bub3[0]+=NBins
                            if xn_bub3[0]>=NBins:
                                xn_bub3[0]-=NBins

                            if yn_bub3[0] < 0:
                                yn_bub3[0]+=NBins
                            if yn_bub3[0]>=NBins:
                                yn_bub3[0]-=NBins

                            if xn_bub3_pos[0] < 0:
                                xn_bub3_pos[0]+=l_box
                            if xn_bub3_pos[0]>=l_box:
                                xn_bub3_pos[0]-=l_box

                            if yn_bub3_pos[0] < 0:
                                yn_bub3_pos[0]+=l_box
                            if yn_bub3_pos[0]>=l_box:
                                yn_bub3_pos[0]-=l_box
                    else:

                        xn_bub3=np.array([int_x[0]])
                        yn_bub3=np.array([int_y[0]])
                        xn_bub3_pos = np.copy(xn_bub3)
                        yn_bub3_pos = np.copy(yn_bub3)
                        xn_bub3_pos_non_per = np.copy(xn_bub3)
                        yn_bub3_pos_non_per = np.copy(yn_bub3)


                        for m in range(0, len(xn_bub3)):
                            xn_bub3_pos[m] = xn_bub3[m] * sizeBin
                            yn_bub3_pos[m] = yn_bub3[m] * sizeBin
                            xn_bub3_pos_non_per[m] = xn_bub3[m] * sizeBin
                            yn_bub3_pos_non_per[m] = yn_bub3[m] * sizeBin

                            if xn_bub3[m] < 0:
                                xn_bub3[m]+=NBins
                            if xn_bub3[m]>=NBins:
                                xn_bub3[m]-=NBins

                            if yn_bub3[m] < 0:
                                yn_bub3[m]+=NBins
                            if yn_bub3[m]>=NBins:
                                yn_bub3[m]-=NBins

                            if xn_bub3_pos[m] < 0:
                                xn_bub3_pos[m]+=l_box
                            if xn_bub3_pos[m]>=l_box:
                                xn_bub3_pos[m]-=l_box

                            if yn_bub3_pos[m] < 0:
                                yn_bub3_pos[m]+=l_box
                            if yn_bub3_pos[m]>=l_box:
                                yn_bub3_pos[m]-=l_box


                if exterior_bin_bub2>0:
                    ix=int(ext_x[0])
                    iy=int(ext_y[0])

                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                    fail=0
                    if len(ext_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if ext_edge_id[right][iy]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][up]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][down]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][iy]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][up]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][down]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][up]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][down]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=ext_x[1]
                            iy=ext_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0

                            while len(ext_bin_unorder_x)>0:
                                    current_size = len(ext_bin_unorder_x)

                                    if past_size == current_size:

                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if ext_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[2]:
                                                            loc_id = np.where((ext_bin_unorder_x == ix6) & (ext_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if ext_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[2]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]
                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]


                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            ext_x = np.append(ext_x, ix)
                                            ext_y = np.append(ext_y, iy)

                                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[2]:
                                        if ext_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((ext_bin_unorder_x == ix2) & (ext_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if ext_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[2]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if ext_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[2]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size


                    for m in range(0, len(ext_x)):
                        ext_x_pos = np.append(ext_x_pos, ext_x[m] * sizeBin)
                        ext_y_pos = np.append(ext_y_pos, ext_y[m] * sizeBin)


                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, ext_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[0])
                    adjacent_y = np.append(adjacent_y, ext_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[0])

                    if len(ext_x)>1:
                        for m in range(1, len(ext_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, ext_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                adjacent_y = np.append(adjacent_y, ext_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                            else:
                                difx = ext_x_pos[m]-ext_x_pos[m-1]
                                dify = ext_y_pos[m]-ext_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        ext_x_pos[m:-1] += l_box
                                        ext_x[m:-1] += NBins
                                    else:
                                        ext_x_pos[m:-1] -= l_box
                                        ext_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        ext_y_pos[m:-1] += l_box
                                        ext_y[m:-1] += NBins
                                    else:
                                        ext_y_pos[m:-1] -= l_box
                                        ext_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, ext_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, ext_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                                    if (m==len(ext_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])





                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        ext_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        ext_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(ext_x)==3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, k=2, per=True)
                        elif len(ext_x)>3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, per=True)

                        if len(ext_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi2, yi2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                            jump2 = np.sqrt(np.diff(xi2)**2 + np.diff(yi2)**2)
                            smooth_jump2 = ndimage.gaussian_filter1d(jump2, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit2 = 2*np.median(smooth_jump2)    # factor 2 is arbitrary
                            xn2_bub3, yn2_bub3 = xi2[:-1], yi2[:-1]
                            xn2_bub3 = xn2_bub3[(jump2 > 0) & (smooth_jump2 < limit2)]
                            yn2_bub3 = yn2_bub3[(jump2 > 0) & (smooth_jump2 < limit2)]

                            xn2_bub3_pos = np.copy(xn2_bub3)
                            yn2_bub3_pos = np.copy(yn2_bub3)
                            xn2_bub3_pos_non_per = np.copy(xn2_bub3)
                            yn2_bub3_pos_non_per = np.copy(yn2_bub3)


                            for m in range(0, len(xn2_bub3)):
                                xn2_bub3_pos[m] = xn2_bub3[m] * sizeBin
                                yn2_bub3_pos[m] = yn2_bub3[m] * sizeBin
                                xn2_bub3_pos_non_per[m] = xn2_bub3[m] * sizeBin
                                yn2_bub3_pos_non_per[m] = yn2_bub3[m] * sizeBin

                                if xn2_bub3[m] < 0:
                                    xn2_bub3[m]+=NBins
                                if xn2_bub3[m]>=NBins:
                                    xn2_bub3[m]-=NBins

                                if yn2_bub3[m] < 0:
                                    yn2_bub3[m]+=NBins
                                if yn2_bub3[m]>=NBins:
                                    yn2_bub3[m]-=NBins

                                if xn2_bub3_pos[m] < 0:
                                    xn2_bub3_pos[m]+=l_box
                                if xn2_bub3_pos[m]>=l_box:
                                    xn2_bub3_pos[m]-=l_box

                                if yn2_bub3_pos[m] < 0:
                                    yn2_bub3_pos[m]+=l_box
                                if yn2_bub3_pos[m]>=l_box:
                                    yn2_bub3_pos[m]-=l_box
                        else:
                            xn2_bub3 = np.zeros(1)
                            yn2_bub3 = np.zeros(1)
                            xn2_bub3_pos = np.zeros(1)
                            yn2_bub3_pos = np.zeros(1)
                            xn2_bub3_pos_non_per = np.zeros(1)
                            yn2_bub3_pos_non_per = np.zeros(1)

                            xn2_bub3_pos[0] = ext_x[0]
                            yn2_bub3_pos[0] = ext_y[0]
                            xn2_bub3_pos[0] = ext_x[0] * sizeBin
                            yn2_bub3_pos[0] = ext_y[0] * sizeBin
                            xn2_bub3_pos_non_per[0] = ext_x[0] * sizeBin
                            yn2_bub3_pos_non_per[0] = ext_y[0] * sizeBin
                            if xn2_bub3[0] < 0:
                                xn2_bub3[0]+=NBins
                            if xn2_bub3[0]>=NBins:
                                xn2_bub3[0]-=NBins

                            if yn2_bub3[0] < 0:
                                yn2_bub3[0]+=NBins
                            if yn2_bub3[0]>=NBins:
                                yn2_bub3[0]-=NBins

                            if xn2_bub3_pos[0] < 0:
                                xn2_bub3_pos[0]+=l_box
                            if xn2_bub3_pos[0]>=l_box:
                                xn2_bub3_pos[0]-=l_box

                            if yn2_bub3_pos[0] < 0:
                                yn2_bub3_pos[0]+=l_box
                            if yn2_bub3_pos[0]>=l_box:
                                yn2_bub3_pos[0]-=l_box
                    else:

                        xn2_bub3=np.array([ext_x[0]])
                        yn2_bub3=np.array([ext_y[0]])
                        xn2_bub3_pos = np.copy(xn2_bub3)
                        yn2_bub3_pos = np.copy(yn2_bub3)
                        xn2_bub3_pos_non_per = np.copy(xn2_bub3)
                        yn2_bub3_pos_non_per = np.copy(yn2_bub3)


                        for m in range(0, len(xn2_bub3)):
                            xn2_bub3_pos[m] = xn2_bub3[m] * sizeBin
                            yn2_bub3_pos[m] = yn2_bub3[m] * sizeBin
                            xn2_bub3_pos_non_per[m] = xn2_bub3[m] * sizeBin
                            yn2_bub3_pos_non_per[m] = yn2_bub3[m] * sizeBin

                            if xn2_bub3[m] < 0:
                                xn2_bub3[m]+=NBins
                            if xn2_bub3[m]>=NBins:
                                xn2_bub3[m]-=NBins

                            if yn2_bub3[m] < 0:
                                yn2_bub3[m]+=NBins
                            if yn2_bub3[m]>=NBins:
                                yn2_bub3[m]-=NBins

                            if xn2_bub3_pos[m] < 0:
                                xn2_bub3_pos[m]+=l_box
                            if xn2_bub3_pos[m]>=l_box:
                                xn2_bub3_pos[m]-=l_box

                            if yn2_bub3_pos[m] < 0:
                                yn2_bub3_pos[m]+=l_box
                            if yn2_bub3_pos[m]>=l_box:
                                yn2_bub3_pos[m]-=l_box

            interior_bin_bub3=0
            exterior_bin_bub3=0
            if bub_large >=4:
                # Initialize empty arrays
                int_x = np.array([], dtype=int)
                int_y = np.array([], dtype=int)
                ext_x = np.array([], dtype=int)
                ext_y = np.array([], dtype=int)

                int_x_pos = np.array([], dtype=int)
                int_y_pos = np.array([], dtype=int)
                ext_x_pos = np.array([], dtype=int)
                ext_y_pos = np.array([], dtype=int) #IM HERE. NEED TO CALC POS ARRAY HERE

                int_bin_unorder_x = np.array([], dtype=int)
                int_bin_unorder_y = np.array([], dtype=int)
                int_bin_unorder_x2 = np.array([], dtype=float)
                int_bin_unorder_y2 = np.array([], dtype=float)

                ext_bin_unorder_x = np.array([], dtype=int)
                ext_bin_unorder_y = np.array([], dtype=int)
                ext_bin_unorder_x2 = np.array([], dtype=float)
                ext_bin_unorder_y2 = np.array([], dtype=float)

                int_bin_unorder_x_copy = np.array([], dtype=int)
                int_bin_unorder_y_copy = np.array([], dtype=int)
                int_bin_unorder_x2_copy = np.array([], dtype=float)
                int_bin_unorder_y2_copy = np.array([], dtype=float)

                ext_bin_unorder_x_copy = np.array([], dtype=int)
                ext_bin_unorder_y_copy = np.array([], dtype=int)
                ext_bin_unorder_x2_copy = np.array([], dtype=float)
                ext_bin_unorder_y2_copy = np.array([], dtype=float)



                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if int_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[3]:
                                interior_bin_bub3 +=1
                                int_bin_unorder_x_copy = np.append(int_bin_unorder_x_copy, ix)
                                int_bin_unorder_y_copy = np.append(int_bin_unorder_y_copy, iy)
                                int_bin_unorder_x2_copy = np.append(int_bin_unorder_x2_copy, float(ix))
                                int_bin_unorder_y2_copy = np.append(int_bin_unorder_y2_copy, float(iy))
                        if ext_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[3]:
                                exterior_bin_bub3 +=1
                                ext_bin_unorder_x_copy = np.append(ext_bin_unorder_x_copy, ix)
                                ext_bin_unorder_y_copy = np.append(ext_bin_unorder_y_copy, iy)
                                ext_bin_unorder_x2_copy = np.append(ext_bin_unorder_x2_copy, float(ix))
                                ext_bin_unorder_y2_copy = np.append(ext_bin_unorder_y2_copy, float(iy))


                if interior_bin_bub3 > 0:
                    if exterior_bin_bub3>0:
                        if interior_bin_bub3>exterior_bin_bub3:
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                        else:
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, int_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, int_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                    else:
                        for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                elif exterior_bin_bub3 > 0:
                    for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                interior_bin_bub3 = len(int_bin_unorder_x)
                exterior_bin_bub3 = len(ext_bin_unorder_x)

                for ix in range(0, len(int_bin_unorder_x)):
                    for iy in range(0, len(int_bin_unorder_y)):
                        int_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=1
                        ext_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=0
                for ix in range(0, len(ext_bin_unorder_x)):
                    for iy in range(0, len(ext_bin_unorder_y)):
                        int_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=0
                        ext_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=1

                if interior_bin_bub3>0:
                    int_x = np.append(int_x, int_bin_unorder_x[0])
                    int_y = np.append(int_y, int_bin_unorder_y[0])
                if exterior_bin_bub3>0:
                    ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                    ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                if interior_bin_bub3>0:
                    ix=int(int_x[0])
                    iy=int(int_y[0])

                    int_bin_unorder_x = np.delete(int_bin_unorder_x, 0)
                    int_bin_unorder_y = np.delete(int_bin_unorder_y, 0)
                    fail=0
                    if len(int_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if int_edge_id[right][iy]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][up]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][down]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][iy]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][up]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][down]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][up]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][down]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=int_x[1]
                            iy=int_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0
                            while len(int_bin_unorder_x)>0:
                                    current_size = len(int_bin_unorder_x)


                                    if past_size == current_size:
                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if int_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[3]:
                                                            loc_id = np.where((int_bin_unorder_x == ix6) & (int_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if int_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[3]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            int_x = np.append(int_x, ix)
                                            int_y = np.append(int_y, iy)

                                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[3]:
                                        if int_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((int_bin_unorder_x == ix2) & (int_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if int_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[3]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if int_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[3]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size




                    for m in range(0, len(int_x)):
                        int_x_pos = np.append(int_x_pos, int_x[m] * sizeBin)
                        int_y_pos = np.append(int_y_pos, int_y[m] * sizeBin)


                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, int_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[0])
                    adjacent_y = np.append(adjacent_y, int_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[0])

                    if len(int_x)>1:
                        for m in range(1, len(int_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, int_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                adjacent_y = np.append(adjacent_y, int_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                            else:
                                difx = int_x_pos[m]-int_x_pos[m-1]
                                dify = int_y_pos[m]-int_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        int_x_pos[m:-1] += l_box
                                        int_x[m:-1] += NBins
                                    else:
                                        int_x_pos[m:-1] -= l_box
                                        int_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        int_y_pos[m:-1] += l_box
                                        int_y[m:-1] += NBins
                                    else:
                                        int_y_pos[m:-1] -= l_box
                                        int_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, int_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, int_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                                    if (m==len(int_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])


                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        int_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        int_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(int_x)==3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, k=2, per=True)
                        elif len(int_x)>3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, per=True)

                        if len(int_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                            jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                            smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                            xn_bub4, yn_bub4 = xi[:-1], yi[:-1]
                            xn_bub4 = xn_bub4[(jump > 0) & (smooth_jump < limit)]
                            yn_bub4 = yn_bub4[(jump > 0) & (smooth_jump < limit)]

                            xn_bub4_pos = np.copy(xn_bub4)
                            yn_bub4_pos = np.copy(yn_bub4)
                            xn_bub4_pos_non_per = np.copy(xn_bub4)
                            yn_bub4_pos_non_per = np.copy(yn_bub4)


                            for m in range(0, len(xn_bub4)):
                                xn_bub4_pos[m] = xn_bub4[m] * sizeBin
                                yn_bub4_pos[m] = yn_bub4[m] * sizeBin
                                xn_bub4_pos_non_per[m] = xn_bub4[m] * sizeBin
                                yn_bub4_pos_non_per[m] = yn_bub4[m] * sizeBin

                                if xn_bub4[m] < 0:
                                    xn_bub4[m]+=NBins
                                if xn_bub4[m]>=NBins:
                                    xn_bub4[m]-=NBins

                                if yn_bub4[m] < 0:
                                    yn_bub4[m]+=NBins
                                if yn_bub4[m]>=NBins:
                                    yn_bub4[m]-=NBins

                                if xn_bub4_pos[m] < 0:
                                    xn_bub4_pos[m]+=l_box
                                if xn_bub4_pos[m]>=l_box:
                                    xn_bub4_pos[m]-=l_box

                                if yn_bub4_pos[m] < 0:
                                    yn_bub4_pos[m]+=l_box
                                if yn_bub4_pos[m]>=l_box:
                                    yn_bub4_pos[m]-=l_box
                        else:
                            xn_bub4 = np.zeros(1)
                            yn_bub4 = np.zeros(1)
                            xn_bub4_pos = np.zeros(1)
                            yn_bub4_pos = np.zeros(1)
                            xn_bub4_pos_non_per = np.zeros(1)
                            yn_bub4_pos_non_per = np.zeros(1)

                            xn_bub4_pos[0] = int_x[0]
                            yn_bub4_pos[0] = int_y[0]
                            xn_bub4_pos[0] = int_x[0] * sizeBin
                            yn_bub4_pos[0] = int_y[0] * sizeBin
                            xn_bub4_pos_non_per[0] = int_x[0] * sizeBin
                            yn_bub4_pos_non_per[0] = int_y[0] * sizeBin
                            if xn_bub4[0] < 0:
                                xn_bub4[0]+=NBins
                            if xn_bub4[0]>=NBins:
                                xn_bub4[0]-=NBins

                            if yn_bub4[0] < 0:
                                yn_bub4[0]+=NBins
                            if yn_bub4[0]>=NBins:
                                yn_bub4[0]-=NBins

                            if xn_bub4_pos[0] < 0:
                                xn_bub4_pos[0]+=l_box
                            if xn_bub4_pos[0]>=l_box:
                                xn_bub4_pos[0]-=l_box

                            if yn_bub4_pos[0] < 0:
                                yn_bub4_pos[0]+=l_box
                            if yn_bub4_pos[0]>=l_box:
                                yn_bub4_pos[0]-=l_box
                    else:

                        xn_bub4 = np.array([int_x[0]])
                        yn_bub4 = np.array([int_y[0]])
                        xn_bub4_pos = np.copy(xn_bub4)
                        yn_bub4_pos = np.copy(yn_bub4)
                        xn_bub4_pos_non_per = np.copy(xn_bub4)
                        yn_bub4_pos_non_per = np.copy(yn_bub4)


                        for m in range(0, len(xn_bub4)):
                            xn_bub4_pos[m] = xn_bub4[m] * sizeBin
                            yn_bub4_pos[m] = yn_bub4[m] * sizeBin
                            xn_bub4_pos_non_per[m] = xn_bub4[m] * sizeBin
                            yn_bub4_pos_non_per[m] = yn_bub4[m] * sizeBin

                            if xn_bub4[m] < 0:
                                xn_bub4[m]+=NBins
                            if xn_bub4[m]>=NBins:
                                xn_bub4[m]-=NBins

                            if yn_bub4[m] < 0:
                                yn_bub4[m]+=NBins
                            if yn_bub4[m]>=NBins:
                                yn_bub4[m]-=NBins

                            if xn_bub4_pos[m] < 0:
                                xn_bub4_pos[m]+=l_box
                            if xn_bub4_pos[m]>=l_box:
                                xn_bub4_pos[m]-=l_box

                            if yn_bub4_pos[m] < 0:
                                yn_bub4_pos[m]+=l_box
                            if yn_bub4_pos[m]>=l_box:
                                yn_bub4_pos[m]-=l_box


                if exterior_bin_bub3>0:
                    ix=int(ext_x[0])
                    iy=int(ext_y[0])

                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                    fail=0
                    if len(ext_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if ext_edge_id[right][iy]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][up]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][down]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][iy]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][up]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][down]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][up]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][down]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=ext_x[1]
                            iy=ext_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0

                            while len(ext_bin_unorder_x)>0:
                                    current_size = len(ext_bin_unorder_x)

                                    if past_size == current_size:

                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if ext_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[3]:
                                                            loc_id = np.where((ext_bin_unorder_x == ix6) & (ext_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if ext_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[3]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]
                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]


                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            ext_x = np.append(ext_x, ix)
                                            ext_y = np.append(ext_y, iy)

                                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[3]:
                                        if ext_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((ext_bin_unorder_x == ix2) & (ext_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if ext_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[3]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if ext_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[3]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size




                    for m in range(0, len(ext_x)):
                        ext_x_pos = np.append(ext_x_pos, ext_x[m] * sizeBin)
                        ext_y_pos = np.append(ext_y_pos, ext_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, ext_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[0])
                    adjacent_y = np.append(adjacent_y, ext_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[0])
                    if len(ext_x)>1:
                        for m in range(1, len(ext_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, ext_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                adjacent_y = np.append(adjacent_y, ext_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                            else:
                                difx = ext_x_pos[m]-ext_x_pos[m-1]
                                dify = ext_y_pos[m]-ext_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        ext_x_pos[m:-1] += l_box
                                        ext_x[m:-1] += NBins
                                    else:
                                        ext_x_pos[m:-1] -= l_box
                                        ext_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        ext_y_pos[m:-1] += l_box
                                        ext_y[m:-1] += NBins
                                    else:
                                        ext_y_pos[m:-1] -= l_box
                                        ext_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, ext_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, ext_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                                    if (m==len(ext_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])





                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)

                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        ext_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        ext_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(ext_x)==3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, k=2, per=True)
                        elif len(ext_x)>3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, per=True)

                        if len(ext_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi2, yi2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                            jump2 = np.sqrt(np.diff(xi2)**2 + np.diff(yi2)**2)
                            smooth_jump2 = ndimage.gaussian_filter1d(jump2, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit2 = 2*np.median(smooth_jump2)    # factor 2 is arbitrary
                            xn2_bub4, yn2_bub4 = xi2[:-1], yi2[:-1]
                            xn2_bub4 = xn2_bub4[(jump2 > 0) & (smooth_jump2 < limit2)]
                            yn2_bub4 = yn2_bub4[(jump2 > 0) & (smooth_jump2 < limit2)]

                            xn2_bub4_pos = np.copy(xn2_bub4)
                            yn2_bub4_pos = np.copy(yn2_bub4)
                            xn2_bub4_pos_non_per = np.copy(xn2_bub4)
                            yn2_bub4_pos_non_per = np.copy(yn2_bub4)
                            for m in range(0, len(xn2_bub4)):
                                xn2_bub4_pos[m] = xn2_bub4[m] * sizeBin
                                yn2_bub4_pos[m] = yn2_bub4[m] * sizeBin
                                xn2_bub4_pos_non_per[m] = xn2_bub4[m] * sizeBin
                                yn2_bub4_pos_non_per[m] = yn2_bub4[m] * sizeBin

                                if xn2_bub4[m] < 0:
                                    xn2_bub4[m]+=NBins
                                if xn2_bub4[m]>=NBins:
                                    xn2_bub4[m]-=NBins

                                if yn2_bub4[m] < 0:
                                    yn2_bub4[m]+=NBins
                                if yn2_bub4[m]>=NBins:
                                    yn2_bub4[m]-=NBins

                                if xn2_bub4_pos[m] < 0:
                                    xn2_bub4_pos[m]+=l_box
                                if xn2_bub4_pos[m]>=l_box:
                                    xn2_bub4_pos[m]-=l_box

                                if yn2_bub4_pos[m] < 0:
                                    yn2_bub4_pos[m]+=l_box
                                if yn2_bub4_pos[m]>=l_box:
                                    yn2_bub4_pos[m]-=l_box
                        else:
                            xn2_bub4 = np.zeros(1)
                            yn2_bub4 = np.zeros(1)
                            xn2_bub4_pos = np.zeros(1)
                            yn2_bub4_pos = np.zeros(1)
                            xn2_bub4_pos_non_per = np.zeros(1)
                            yn2_bub4_pos_non_per = np.zeros(1)

                            xn2_bub4_pos[0] = ext_x[0]
                            yn2_bub4_pos[0] = ext_y[0]
                            xn2_bub4_pos[0] = ext_x[0] * sizeBin
                            yn2_bub4_pos[0] = ext_y[0] * sizeBin
                            xn2_bub4_pos_non_per[0] = ext_x[0] * sizeBin
                            yn2_bub4_pos_non_per[0] = ext_y[0] * sizeBin
                            if xn2_bub4[0] < 0:
                                xn2_bub4[0]+=NBins
                            if xn2_bub4[0]>=NBins:
                                xn2_bub4[0]-=NBins

                            if yn2_bub4[0] < 0:
                                yn2_bub4[0]+=NBins
                            if yn2_bub4[0]>=NBins:
                                yn2_bub4[0]-=NBins

                            if xn2_bub4_pos[0] < 0:
                                xn2_bub4_pos[0]+=l_box
                            if xn2_bub4_pos[0]>=l_box:
                                xn2_bub4_pos[0]-=l_box

                            if yn2_bub4_pos[0] < 0:
                                yn2_bub4_pos[0]+=l_box
                            if yn2_bub4_pos[0]>=l_box:
                                yn2_bub4_pos[0]-=l_box
                    else:
                        xn2_bub4 = np.array(ext_x[0])
                        yn2_bub4 = np.array(ext_y[0])
                        xn2_bub4_pos = np.copy(xn2_bub4)
                        yn2_bub4_pos = np.copy(yn2_bub4)
                        xn2_bub4_pos_non_per = np.copy(xn2_bub4)
                        yn2_bub4_pos_non_per = np.copy(yn2_bub4)
                        for m in range(0, len(xn2_bub4)):
                            xn2_bub4_pos[m] = xn2_bub4[m] * sizeBin
                            yn2_bub4_pos[m] = yn2_bub4[m] * sizeBin
                            xn2_bub4_pos_non_per[m] = xn2_bub4[m] * sizeBin
                            yn2_bub4_pos_non_per[m] = yn2_bub4[m] * sizeBin

                            if xn2_bub4[m] < 0:
                                xn2_bub4[m]+=NBins
                            if xn2_bub4[m]>=NBins:
                                xn2_bub4[m]-=NBins

                            if yn2_bub4[m] < 0:
                                yn2_bub4[m]+=NBins
                            if yn2_bub4[m]>=NBins:
                                yn2_bub4[m]-=NBins

                            if xn2_bub4_pos[m] < 0:
                                xn2_bub4_pos[m]+=l_box
                            if xn2_bub4_pos[m]>=l_box:
                                xn2_bub4_pos[m]-=l_box

                            if yn2_bub4_pos[m] < 0:
                                yn2_bub4_pos[m]+=l_box
                            if yn2_bub4_pos[m]>=l_box:
                                yn2_bub4_pos[m]-=l_box
            interior_bin_bub4=0
            exterior_bin_bub4=0
            if bub_large ==5:
                # Initialize empty arrays
                int_x = np.array([], dtype=int)
                int_y = np.array([], dtype=int)
                ext_x = np.array([], dtype=int)
                ext_y = np.array([], dtype=int)

                int_bin_unorder_x = np.array([], dtype=int)
                int_bin_unorder_y = np.array([], dtype=int)
                int_bin_unorder_x2 = np.array([], dtype=float)
                int_bin_unorder_y2 = np.array([], dtype=float)

                ext_bin_unorder_x = np.array([], dtype=int)
                ext_bin_unorder_y = np.array([], dtype=int)
                ext_bin_unorder_x2 = np.array([], dtype=float)
                ext_bin_unorder_y2 = np.array([], dtype=float)

                int_bin_unorder_x_copy = np.array([], dtype=int)
                int_bin_unorder_y_copy = np.array([], dtype=int)
                int_bin_unorder_x2_copy = np.array([], dtype=float)
                int_bin_unorder_y2_copy = np.array([], dtype=float)

                ext_bin_unorder_x_copy = np.array([], dtype=int)
                ext_bin_unorder_y_copy = np.array([], dtype=int)
                ext_bin_unorder_x2_copy = np.array([], dtype=float)
                ext_bin_unorder_y2_copy = np.array([], dtype=float)

                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if int_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[4]:
                                interior_bin_bub4 +=1
                                int_bin_unorder_x_copy = np.append(int_bin_unorder_x_copy, ix)
                                int_bin_unorder_y_copy = np.append(int_bin_unorder_y_copy, iy)
                                int_bin_unorder_x2_copy = np.append(int_bin_unorder_x2_copy, float(ix))
                                int_bin_unorder_y2_copy = np.append(int_bin_unorder_y2_copy, float(iy))
                        if ext_edge_id[ix][iy]==1:
                            if edge_id[ix][iy]==bub_size_id_arr[4]:
                                exterior_bin_bub4 +=1
                                ext_bin_unorder_x_copy = np.append(ext_bin_unorder_x_copy, ix)
                                ext_bin_unorder_y_copy = np.append(ext_bin_unorder_y_copy, iy)
                                ext_bin_unorder_x2_copy = np.append(ext_bin_unorder_x2_copy, float(ix))
                                ext_bin_unorder_y2_copy = np.append(ext_bin_unorder_y2_copy, float(iy))


                if interior_bin_bub4 > 0:
                    if exterior_bin_bub4>0:
                        if interior_bin_bub4>exterior_bin_bub4:
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                        else:
                            for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                            for v in range(0, len(int_bin_unorder_x_copy)):
                                int_bin_unorder_x = np.append(int_bin_unorder_x, int_bin_unorder_x_copy[v])
                                int_bin_unorder_y = np.append(int_bin_unorder_y, int_bin_unorder_y_copy[v])
                                int_bin_unorder_x2 = np.append(int_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                int_bin_unorder_y2 = np.append(int_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                    else:
                        for v in range(0, len(int_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, int_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, int_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, int_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, int_bin_unorder_y2_copy[v])
                elif exterior_bin_bub4 > 0:
                    for v in range(0, len(ext_bin_unorder_x_copy)):
                                ext_bin_unorder_x = np.append(ext_bin_unorder_x, ext_bin_unorder_x_copy[v])
                                ext_bin_unorder_y = np.append(ext_bin_unorder_y, ext_bin_unorder_y_copy[v])
                                ext_bin_unorder_x2 = np.append(ext_bin_unorder_x2, ext_bin_unorder_x2_copy[v])
                                ext_bin_unorder_y2 = np.append(ext_bin_unorder_y2, ext_bin_unorder_y2_copy[v])
                interior_bin_bub4 = len(int_bin_unorder_x)
                exterior_bin_bub4 = len(ext_bin_unorder_x)

                for ix in range(0, len(int_bin_unorder_x)):
                    for iy in range(0, len(int_bin_unorder_y)):
                        int_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=1
                        ext_edge_id[int_bin_unorder_x[ix]][int_bin_unorder_y[ix]]=0
                for ix in range(0, len(ext_bin_unorder_x)):
                    for iy in range(0, len(ext_bin_unorder_y)):
                        int_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=0
                        ext_edge_id[ext_bin_unorder_x[ix]][ext_bin_unorder_y[ix]]=1

                if interior_bin_bub4>0:
                    int_x = np.append(int_x, int_bin_unorder_x[0])
                    int_y = np.append(int_y, int_bin_unorder_y[0])
                if exterior_bin_bub4>0:
                    ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                    ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                if interior_bin_bub4>0:
                    ix=int(int_x[0])
                    iy=int(int_y[0])

                    int_bin_unorder_x = np.delete(int_bin_unorder_x, 0)
                    int_bin_unorder_y = np.delete(int_bin_unorder_y, 0)
                    fail=0
                    if len(int_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if int_edge_id[right][iy]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][up]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[ix][down]==1:
                            int_x = np.append(int_x, ix)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][iy]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, iy)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == iy))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][up]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[left][down]==1:
                            int_x = np.append(int_x, left)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == left) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][up]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, up)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == up))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        elif int_edge_id[right][down]==1:
                            int_x = np.append(int_x, right)
                            int_y = np.append(int_y, down)

                            loc_id = np.where((int_bin_unorder_x == right) & (int_bin_unorder_y == down))[0]

                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=int_x[1]
                            iy=int_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0
                            while len(int_bin_unorder_x)>0:
                                    current_size = len(int_bin_unorder_x)

                                    if past_size == current_size:
                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if int_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[4]:
                                                            loc_id = np.where((int_bin_unorder_x == ix6) & (int_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if int_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[4]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            int_x = np.append(int_x, ix)
                                            int_y = np.append(int_y, iy)

                                            loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                            int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                            int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[4]:
                                        if int_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((int_bin_unorder_x == ix2) & (int_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if int_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[4]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((int_bin_unorder_x == ix4) & (int_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if int_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[4]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    int_x = np.append(int_x, ix)
                                                    int_y = np.append(int_y, iy)

                                                    loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                    int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                    int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                int_x = np.append(int_x, ix)
                                                int_y = np.append(int_y, iy)

                                                loc_id = np.where((int_bin_unorder_x == ix) & (int_bin_unorder_y == iy))[0]

                                                int_bin_unorder_x = np.delete(int_bin_unorder_x, loc_id)
                                                int_bin_unorder_y = np.delete(int_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size

                    # append the starting x,y coordinates
                    #int_x = np.r_[int_x, int_x[0]]
                    #int_y = np.r_[int_y, int_y[0]]


                    for m in range(0, len(int_x)):
                        int_x_pos = np.append(int_x_pos, int_x[m] * sizeBin)
                        int_y_pos = np.append(int_y_pos, int_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, int_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[0])
                    adjacent_y = np.append(adjacent_y, int_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[0])
                    if len(int_x)>1:
                        for m in range(1, len(int_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, int_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                adjacent_y = np.append(adjacent_y, int_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                            else:
                                difx = int_x_pos[m]-int_x_pos[m-1]
                                dify = int_y_pos[m]-int_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        int_x_pos[m:-1] += l_box
                                        int_x[m:-1] += NBins
                                    else:
                                        int_x_pos[m:-1] -= l_box
                                        int_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        int_y_pos[m:-1] += l_box
                                        int_y[m:-1] += NBins
                                    else:
                                        int_y_pos[m:-1] -= l_box
                                        int_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, int_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, int_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, int_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, int_y_pos[m])
                                    if (m==len(int_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])


                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        int_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        int_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(int_x)==3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, k=2, per=True)
                        elif len(int_x)>3:
                            tck, u = interpolate.splprep([int_x, int_y], s=0, per=True)

                        if len(int_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                            jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                            smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary

                            xn_bub5, yn_bub5 = xi[:-1], yi[:-1]
                            xn_bub5 = xn_bub5[(jump > 0) & (smooth_jump < limit)]
                            yn_bub5 = yn_bub5[(jump > 0) & (smooth_jump < limit)]

                            xn_bub5_pos = np.copy(xn_bub5)
                            yn_bub5_pos = np.copy(yn_bub5)
                            xn_bub5_pos_non_per = np.copy(xn_bub5)
                            yn_bub5_pos_non_per = np.copy(yn_bub5)



                            for m in range(0, len(xn_bub5)):
                                xn_bub5_pos[m] = xn_bub5[m] * sizeBin
                                yn_bub5_pos[m] = yn_bub5[m] * sizeBin
                                xn_bub5_pos_non_per[m] = xn_bub5[m] * sizeBin
                                yn_bub5_pos_non_per[m] = yn_bub5[m] * sizeBin

                                if xn_bub5[m] < 0:
                                    xn_bub5[m]+=NBins
                                if xn_bub5[m]>=NBins:
                                    xn_bub5[m]-=NBins

                                if yn_bub5[m] < 0:
                                    yn_bub5[m]+=NBins
                                if yn_bub5[m]>=NBins:
                                    yn_bub5[m]-=NBins

                                if xn_bub5_pos[m] < 0:
                                    xn_bub5_pos[m]+=l_box
                                if xn_bub5_pos[m]>=l_box:
                                    xn_bub5_pos[m]-=l_box

                                if yn_bub5_pos[m] < 0:
                                    yn_bub5_pos[m]+=l_box
                                if yn_bub5_pos[m]>=l_box:
                                    yn_bub5_pos[m]-=l_box
                        else:
                            xn_bub5 = np.zeros(1)
                            yn_bub5 = np.zeros(1)
                            xn_bub5_pos = np.zeros(1)
                            yn_bub5_pos = np.zeros(1)
                            xn_bub5_pos_non_per = np.zeros(1)
                            yn_bub5_pos_non_per = np.zeros(1)

                            xn_bub5_pos[0] = int_x[0]
                            yn_bub5_pos[0] = int_y[0]
                            xn_bub5_pos[0] = int_x[0] * sizeBin
                            yn_bub5_pos[0] = int_y[0] * sizeBin
                            xn_bub5_pos_non_per[0] = int_x[0] * sizeBin
                            yn_bub5_pos_non_per[0] = int_y[0] * sizeBin
                            if xn_bub5[0] < 0:
                                xn_bub5[0]+=NBins
                            if xn_bub5[0]>=NBins:
                                xn_bub5[0]-=NBins

                            if yn_bub5[0] < 0:
                                yn_bub5[0]+=NBins
                            if yn_bub5[0]>=NBins:
                                yn_bub5[0]-=NBins

                            if xn_bub5_pos[0] < 0:
                                xn_bub5_pos[0]+=l_box
                            if xn_bub5_pos[0]>=l_box:
                                xn_bub5_pos[0]-=l_box

                            if yn_bub5_pos[0] < 0:
                                yn_bub5_pos[0]+=l_box
                            if yn_bub5_pos[0]>=l_box:
                                yn_bub5_pos[0]-=l_box
                    else:
                        xn_bub5 = np.array([int_x[0]])
                        yn_bub5 = np.array([int_y[0]])
                        xn_bub5_pos = np.copy(xn_bub5)
                        yn_bub5_pos = np.copy(yn_bub5)
                        xn_bub5_pos_non_per = np.copy(xn_bub5)
                        yn_bub5_pos_non_per = np.copy(yn_bub5)



                        for m in range(0, len(xn_bub5)):
                            xn_bub5_pos[m] = xn_bub5[m] * sizeBin
                            yn_bub5_pos[m] = yn_bub5[m] * sizeBin
                            xn_bub5_pos_non_per[m] = xn_bub5[m] * sizeBin
                            yn_bub5_pos_non_per[m] = yn_bub5[m] * sizeBin

                            if xn_bub5[m] < 0:
                                xn_bub5[m]+=NBins
                            if xn_bub5[m]>=NBins:
                                xn_bub5[m]-=NBins

                            if yn_bub5[m] < 0:
                                yn_bub5[m]+=NBins
                            if yn_bub5[m]>=NBins:
                                yn_bub5[m]-=NBins

                            if xn_bub5_pos[m] < 0:
                                xn_bub5_pos[m]+=l_box
                            if xn_bub5_pos[m]>=l_box:
                                xn_bub5_pos[m]-=l_box

                            if yn_bub5_pos[m] < 0:
                                yn_bub5_pos[m]+=l_box
                            if yn_bub5_pos[m]>=l_box:
                                yn_bub5_pos[m]-=l_box


                if exterior_bin_bub4>0:
                    ix=int(ext_x[0])
                    iy=int(ext_y[0])

                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                    fail=0
                    if len(ext_bin_unorder_x)>0:
                        if ix < (NBins-1):
                            right = int(ix+1)
                        else:
                            right= int(0)

                        if ix > 0:
                            left = int(ix-1)
                        else:
                            left=int(NBins-1)

                        if iy < (NBins-1):
                            up = int(iy+1)
                        else:
                            up= int(0)

                        if iy > 0:
                            down = int(iy-1)
                        else:
                            down= int(NBins-1)

                        if ext_edge_id[right][iy]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][up]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[ix][down]==1:
                            ext_x = np.append(ext_x, ix)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][iy]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, iy)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == iy))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][up]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[left][down]==1:
                            ext_x = np.append(ext_x, left)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == left) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][up]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, up)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == up))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        elif ext_edge_id[right][down]==1:
                            ext_x = np.append(ext_x, right)
                            ext_y = np.append(ext_y, down)

                            loc_id = np.where((ext_bin_unorder_x == right) & (ext_bin_unorder_y == down))[0]

                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                        else:
                            fail=1
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])
                        if fail==0:
                            #ext_x = np.append(ext_x, ext_bin_unorder_x[0])
                            #ext_y = np.append(ext_y, ext_bin_unorder_y[0])

                            ix=ext_x[1]
                            iy=ext_y[1]

                            #ext_bin_unorder_x = np.delete(ext_bin_unorder_x, 0)
                            #ext_bin_unorder_y = np.delete(ext_bin_unorder_y, 0)
                            past_size=0

                            while len(ext_bin_unorder_x)>0:
                                    current_size = len(ext_bin_unorder_x)

                                    if past_size == current_size:

                                        shortest_length = 100000.
                                        for ix6 in range(0, len(occParts)):
                                            for iy6 in range(0, len(occParts)):
                                                if (ix6!=ix) | (iy6!=iy):
                                                    if ext_edge_id[ix6][iy6]==1:
                                                        if edge_id[ix6][iy6]==bub_size_id_arr[4]:
                                                            loc_id = np.where((ext_bin_unorder_x == ix6) & (ext_bin_unorder_y == iy6))[0]
                                                            if len(loc_id)>0:
                                                                difx = (ix+0.5)*sizeBin-(ix6+0.5)*sizeBin
                                                                dify = (iy+0.5)*sizeBin-(iy6+0.5)*sizeBin

                                                                #Enforce periodic boundary conditions
                                                                difx_abs = np.abs(difx)
                                                                if difx_abs>=h_box:
                                                                    if difx < -h_box:
                                                                        difx += l_box
                                                                    else:
                                                                        difx -= l_box

                                                                #Enforce periodic boundary conditions
                                                                dify_abs = np.abs(dify)
                                                                if dify_abs>=h_box:
                                                                    if dify < -h_box:
                                                                        dify += l_box
                                                                    else:
                                                                        dify -= l_box

                                                                difr = (difx**2 + dify**2)**0.5

                                                                if difr < shortest_length:
                                                                    shortest_length = difr
                                                                    shortest_idx_arr = np.array([ix6])
                                                                    shortest_idy_arr = np.array([iy6])
                                                                elif difr == shortest_length:
                                                                    shortest_idx_arr = np.append(shortest_idx_arr, ix6)
                                                                    shortest_idy_arr = np.append(shortest_idy_arr, iy6)
                                        if shortest_length > h_box/10:
                                            break
                                        if len(shortest_idx_arr) > 1:
                                            num_neigh = np.zeros(len(shortest_idx_arr))

                                            for ind3 in range(0, len(shortest_idx_arr)):
                                                ix3 = shortest_idx_arr[ind3]
                                                iy3 = shortest_idy_arr[ind3]
                                                if (ix3 + 1) == NBins:
                                                    lookx = [ix3-1, ix3, 0]
                                                elif ix3==0:
                                                    lookx=[NBins-1, ix3, ix3+1]
                                                else:
                                                    lookx = [ix3-1, ix3, ix3+1]
                                                #Identify neighboring bin indices in y-direction
                                                if (iy3 + 1) == NBins:
                                                    looky = [iy3-1, iy3, 0]
                                                elif iy3==0:
                                                    looky=[NBins-1, iy3, iy3+1]
                                                else:
                                                    looky = [iy3-1, iy3, iy3+1]
                                                for ix4 in lookx:
                                                    for iy4 in looky:
                                                        if (ix4 != ix3) | (iy4 != iy3):
                                                            loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                            if len(loc_id)>0:
                                                                if ext_edge_id[ix4][iy4]==1:
                                                                    if edge_id[ix][iy] == bub_size_id_arr[4]:
                                                                        num_neigh[ind3]+=1
                                            min_inds = np.min(num_neigh)
                                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                                            if len(loc_min_inds)==1:

                                                ix = shortest_idx_arr[loc_min_inds][0]
                                                iy = shortest_idy_arr[loc_min_inds][0]
                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                iy = shortest_idy_arr[np.min(loc_min_inds)]


                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        elif len(shortest_idx_arr)==1:

                                            ix = shortest_idx_arr[0]
                                            iy = shortest_idy_arr[0]

                                            ext_x = np.append(ext_x, ix)
                                            ext_y = np.append(ext_y, iy)

                                            loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                            ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                            ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                        else:
                                            break
                                    if edge_id[ix][iy] == bub_size_id_arr[4]:
                                        if ext_edge_id[ix][iy]==1:
                                            #Identify neighboring bin indices in x-direction
                                            if (ix + 1) == NBins:
                                                lookx = [ix-1, ix, 0]
                                            elif ix==0:
                                                lookx=[NBins-1, ix, ix+1]
                                            else:
                                                lookx = [ix-1, ix, ix+1]
                                            #Identify neighboring bin indices in y-direction
                                            if (iy + 1) == NBins:
                                                looky = [iy-1, iy, 0]
                                            elif iy==0:
                                                looky=[NBins-1, iy, iy+1]
                                            else:
                                                looky = [iy-1, iy, iy+1]
                                            shortest_length = 100000.
                                            for ix2 in lookx:
                                                for iy2 in looky:
                                                    if (ix2!=ix) | (iy2!=iy):
                                                        loc_id = np.where((ext_bin_unorder_x == ix2) & (ext_bin_unorder_y == iy2))[0]
                                                        if len(loc_id)>0:
                                                            if ext_edge_id[ix2][iy2]==1:
                                                                if edge_id[ix][iy] == bub_size_id_arr[4]:

                                                                    difx = (ix2+0.5)*sizeBin-(ix+0.5)*sizeBin
                                                                    dify = (iy2+0.5)*sizeBin-(iy+0.5)*sizeBin

                                                                    #Enforce periodic boundary conditions
                                                                    difx_abs = np.abs(difx)
                                                                    if difx_abs>=h_box:
                                                                        if difx < -h_box:
                                                                            difx += l_box
                                                                        else:
                                                                            difx -= l_box

                                                                    #Enforce periodic boundary conditions
                                                                    dify_abs = np.abs(dify)
                                                                    if dify_abs>=h_box:
                                                                        if dify < -h_box:
                                                                            dify += l_box
                                                                        else:
                                                                            dify -= l_box

                                                                    difr = (difx**2 + dify**2)**0.5

                                                                    if difr < shortest_length:
                                                                        shortest_length = difr
                                                                        shortest_idx_arr = np.array([ix2])
                                                                        shortest_idy_arr = np.array([iy2])

                                                                    elif difr == shortest_length:
                                                                        shortest_idx_arr = np.append(shortest_idx_arr, ix2)
                                                                        shortest_idy_arr = np.append(shortest_idy_arr, iy2)

                                            if len(shortest_idx_arr) > 1:
                                                num_neigh = np.zeros(len(shortest_idx_arr))

                                                for ind3 in range(0, len(shortest_idx_arr)):
                                                    ix3 = shortest_idx_arr[ind3]
                                                    iy3 = shortest_idy_arr[ind3]
                                                    if (ix3 + 1) == NBins:
                                                        lookx = [ix3-1, ix3, 0]
                                                    elif ix3==0:
                                                        lookx=[NBins-1, ix3, ix3+1]
                                                    else:
                                                        lookx = [ix3-1, ix3, ix3+1]
                                                    #Identify neighboring bin indices in y-direction
                                                    if (iy3 + 1) == NBins:
                                                        looky = [iy3-1, iy3, 0]
                                                    elif iy3==0:
                                                        looky=[NBins-1, iy3, iy3+1]
                                                    else:
                                                        looky = [iy3-1, iy3, iy3+1]
                                                    for ix4 in lookx:
                                                        for iy4 in looky:
                                                            if (ix4 != ix3) | (iy4 != iy3):
                                                                loc_id = np.where((ext_bin_unorder_x == ix4) & (ext_bin_unorder_y == iy4))[0]
                                                                if len(loc_id)>0:
                                                                    if ext_edge_id[ix4][iy4]==1:
                                                                        if edge_id[ix][iy] == bub_size_id_arr[4]:
                                                                            num_neigh[ind3]+=1
                                                min_inds = np.min(num_neigh)
                                                loc_min_inds = np.where(num_neigh == min_inds)[0]

                                                if len(loc_min_inds)==1:

                                                    ix = shortest_idx_arr[loc_min_inds][0]
                                                    iy = shortest_idy_arr[loc_min_inds][0]

                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                                else:
                                                    ix = shortest_idx_arr[np.min(loc_min_inds)]
                                                    iy = shortest_idy_arr[np.min(loc_min_inds)]
                                                    ext_x = np.append(ext_x, ix)
                                                    ext_y = np.append(ext_y, iy)

                                                    loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                    ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                    ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            elif len(shortest_idx_arr)==1:

                                                ix = shortest_idx_arr[0]
                                                iy = shortest_idy_arr[0]

                                                ext_x = np.append(ext_x, ix)
                                                ext_y = np.append(ext_y, iy)

                                                loc_id = np.where((ext_bin_unorder_x == ix) & (ext_bin_unorder_y == iy))[0]

                                                ext_bin_unorder_x = np.delete(ext_bin_unorder_x, loc_id)
                                                ext_bin_unorder_y = np.delete(ext_bin_unorder_y, loc_id)
                                            else:
                                                break
                                    past_size = current_size

                    for m in range(0, len(ext_x)):
                        ext_x_pos = np.append(ext_x_pos, ext_x[m] * sizeBin)
                        ext_y_pos = np.append(ext_y_pos, ext_y[m] * sizeBin)

                    adjacent_x = np.array([])
                    adjacent_x_pos = np.array([])
                    adjacent_x_arr = np.array([])
                    adjacent_x_arr_pos = np.array([])
                    adjacent_y = np.array([])
                    adjacent_y_pos = np.array([])
                    adjacent_y_arr = np.array([])
                    adjacent_y_arr_pos = np.array([])

                    adjacent_x = np.append(adjacent_x, ext_x[0])
                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[0])
                    adjacent_y = np.append(adjacent_y, ext_y[0])
                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[0])

                    if len(ext_x)>1:
                        for m in range(1, len(ext_x)):
                            if len(adjacent_x) == 0:
                                adjacent_x = np.append(adjacent_x, ext_x[m])
                                adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                adjacent_y = np.append(adjacent_y, ext_y[m])
                                adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                            else:
                                difx = ext_x_pos[m]-ext_x_pos[m-1]
                                dify = ext_y_pos[m]-ext_y_pos[m-1]

                                #Enforce periodic boundary conditions
                                difx_abs = np.abs(difx)
                                dify_abs = np.abs(dify)

                                #Enforce periodic boundary conditions
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        ext_x_pos[m:-1] += l_box
                                        ext_x[m:-1] += NBins
                                    else:
                                        ext_x_pos[m:-1] -= l_box
                                        ext_x[m:-1] -= NBins

                                #Enforce periodic boundary conditions
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        ext_y_pos[m:-1] += l_box
                                        ext_y[m:-1] += NBins
                                    else:
                                        ext_y_pos[m:-1] -= l_box
                                        ext_y[m:-1] -= NBins

                                if (difx_abs>=h_box) or (dify_abs>=h_box):
                                    adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                    adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                    adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                    adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                    adjacent_x = np.array([])
                                    adjacent_x_pos = np.array([])
                                    adjacent_y = np.array([])
                                    adjacent_y_pos = np.array([])
                                else:
                                    adjacent_x = np.append(adjacent_x, ext_x[m])
                                    adjacent_x_pos = np.append(adjacent_x_pos, ext_x_pos[m])
                                    adjacent_y = np.append(adjacent_y, ext_y[m])
                                    adjacent_y_pos = np.append(adjacent_y_pos, ext_y_pos[m])
                                    if (m==len(ext_x)-1):
                                        adjacent_x_arr = np.append(adjacent_x_arr, adjacent_x)
                                        adjacent_x_arr_pos = np.append(adjacent_x_arr_pos, adjacent_x_pos)
                                        adjacent_y_arr = np.append(adjacent_y_arr, adjacent_y)
                                        adjacent_y_arr_pos = np.append(adjacent_y_arr_pos, adjacent_y_pos)
                                        adjacent_x = np.array([])
                                        adjacent_x_pos = np.array([])
                                        adjacent_y = np.array([])
                                        adjacent_y_pos = np.array([])





                        adjacent_x_arr_pos_new = np.array([])
                        adjacent_y_arr_pos_new = np.array([])
                        adjacent_x_arr_new = np.array([])
                        adjacent_y_arr_new = np.array([])
                        for m in range(0, len(adjacent_x_arr_pos)):
                            adjacent_x_arr_pos_new = np.append(adjacent_x_arr_pos_new, adjacent_x_arr_pos[m])
                            adjacent_y_arr_pos_new = np.append(adjacent_y_arr_pos_new, adjacent_y_arr_pos[m])
                            adjacent_x_arr_new = np.append(adjacent_x_arr_new, adjacent_x_arr[m])
                            adjacent_y_arr_new = np.append(adjacent_y_arr_new, adjacent_y_arr[m])

                        int_x_copy = np.copy(adjacent_x_arr_new)
                        int_y_copy = np.copy(adjacent_y_arr_new)
                        if len(adjacent_x_arr_new) >= 3:
                            for m in range(0, len(adjacent_x_arr_new)):

                                if m==0:
                                    adjacent_x_arr_new[m] = (int_x_copy[-1] + int_x_copy[0] + int_x_copy[1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[-1] + int_y_copy[0] + int_y_copy[1])/3
                                elif m==len(adjacent_x_arr_new)-1:
                                    adjacent_x_arr_new[m]= (int_x_copy[m] + int_x_copy[0] + int_x_copy[m-1])/3
                                    adjacent_y_arr_new[m]= (int_y_copy[m] + int_y_copy[0] + int_y_copy[m-1])/3
                                else:
                                    adjacent_x_arr_new[m] = (int_x_copy[m-1] + int_x_copy[m] + int_x_copy[m+1])/3
                                    adjacent_y_arr_new[m] = (int_y_copy[m-1] + int_y_copy[m] + int_y_copy[m+1])/3
                        else:
                            for m in range(0, len(adjacent_x_arr_new)):

                                adjacent_x_arr_new[m] = np.mean(int_x_copy)
                                adjacent_y_arr_new[m] = np.mean(int_y_copy)
                        okay = np.where(np.abs(np.diff(adjacent_x_arr_new)) + np.abs(np.diff(adjacent_y_arr_new)) > 0)
                        ext_x = np.r_[adjacent_x_arr_new[okay], adjacent_x_arr_new[-1], adjacent_x_arr_new[0]]
                        ext_y = np.r_[adjacent_y_arr_new[okay], adjacent_y_arr_new[-1], adjacent_y_arr_new[0]]

                        if len(ext_x)==3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, k=2, per=True)
                        elif len(ext_x)>3:
                            tck2, u2 = interpolate.splprep([ext_x, ext_y], s=0, per=True)

                        if len(ext_x)>=3:
                            # evaluate the spline fits for 1000 evenly spaced distance values
                            xi2, yi2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                            jump2 = np.sqrt(np.diff(xi2)**2 + np.diff(yi2)**2)
                            smooth_jump2 = ndimage.gaussian_filter1d(jump2, 5, mode='wrap')  # window of size 5 is arbitrary
                            limit2 = 2*np.median(smooth_jump2)    # factor 2 is arbitrary
                            xn2_bub5, yn2_bub5 = xi2[:-1], yi2[:-1]
                            xn2_bub5 = xn2_bub5[(jump2 > 0) & (smooth_jump2 < limit2)]
                            yn2_bub5 = yn2_bub5[(jump2 > 0) & (smooth_jump2 < limit2)]

                            xn2_bub5_pos = np.copy(xn2_bub5)
                            yn2_bub5_pos = np.copy(yn2_bub5)
                            xn2_bub5_pos_non_per = np.copy(xn2_bub5)
                            yn2_bub5_pos_non_per = np.copy(yn2_bub5)
                            for m in range(0, len(xn2_bub5)):
                                xn2_bub5_pos[m] = xn2_bub5[m] * sizeBin
                                yn2_bub5_pos[m] = yn2_bub5[m] * sizeBin
                                xn2_bub5_pos_non_per[m] = xn2_bub5[m] * sizeBin
                                yn2_bub5_pos_non_per[m] = yn2_bub5[m] * sizeBin
                                if xn2_bub5[m] < 0:
                                    xn2_bub5[m]+=NBins
                                if xn2_bub5[m]>=NBins:
                                    xn2_bub5[m]-=NBins

                                if yn2_bub5[m] < 0:
                                    yn2_bub5[m]+=NBins
                                if yn2_bub5[m]>=NBins:
                                    yn2_bub5[m]-=NBins

                                if xn2_bub5_pos[m] < 0:
                                    xn2_bub5_pos[m]+=l_box
                                if xn2_bub5_pos[m]>=l_box:
                                    xn2_bub5_pos[m]-=l_box

                                if yn2_bub5_pos[m] < 0:
                                    yn2_bub5_pos[m]+=l_box
                                if yn2_bub5_pos[m]>=l_box:
                                    yn2_bub5_pos[m]-=l_box
                        else:
                            xn2_bub5 = np.zeros(1)
                            yn2_bub5 = np.zeros(1)
                            xn2_bub5_pos = np.zeros(1)
                            yn2_bub5_pos = np.zeros(1)
                            xn2_bub5_pos_non_per = np.zeros(1)
                            yn2_bub5_pos_non_per = np.zeros(1)

                            xn2_bub5_pos[0] = ext_x[0]
                            yn2_bub5_pos[0] = ext_y[0]
                            xn2_bub5_pos[0] = ext_x[0] * sizeBin
                            yn2_bub5_pos[0] = ext_y[0] * sizeBin
                            xn2_bub5_pos_non_per[0] = ext_x[0] * sizeBin
                            yn2_bub5_pos_non_per[0] = ext_y[0] * sizeBin
                            if xn2_bub5[0] < 0:
                                xn2_bub5[0]+=NBins
                            if xn2_bub5[0]>=NBins:
                                xn2_bub5[0]-=NBins

                            if yn2_bub5[0] < 0:
                                yn2_bub5[0]+=NBins
                            if yn2_bub5[0]>=NBins:
                                yn2_bub5[0]-=NBins

                            if xn2_bub5_pos[0] < 0:
                                xn2_bub5_pos[0]+=l_box
                            if xn2_bub5_pos[0]>=l_box:
                                xn2_bub5_pos[0]-=l_box

                            if yn2_bub5_pos[0] < 0:
                                yn2_bub5_pos[0]+=l_box
                            if yn2_bub5_pos[0]>=l_box:
                                yn2_bub5_pos[0]-=l_box
                    else:
                        xn2_bub5 = np.array([ext_x[0]])
                        yn2_bub5 = np.array([ext_y[0]])
                        xn2_bub5_pos = np.copy(xn2_bub5)
                        yn2_bub5_pos = np.copy(yn2_bub5)
                        xn2_bub5_pos_non_per = np.copy(xn2_bub5)
                        yn2_bub5_pos_non_per = np.copy(yn2_bub5)
                        for m in range(0, len(xn2_bub5)):
                            xn2_bub5_pos[m] = xn2_bub5[m] * sizeBin
                            yn2_bub5_pos[m] = yn2_bub5[m] * sizeBin
                            xn2_bub5_pos_non_per[m] = xn2_bub5[m] * sizeBin
                            yn2_bub5_pos_non_per[m] = yn2_bub5[m] * sizeBin
                            if xn2_bub5[m] < 0:
                                xn2_bub5[m]+=NBins
                            if xn2_bub5[m]>=NBins:
                                xn2_bub5[m]-=NBins

                            if yn2_bub5[m] < 0:
                                yn2_bub5[m]+=NBins
                            if yn2_bub5[m]>=NBins:
                                yn2_bub5[m]-=NBins

                            if xn2_bub5_pos[m] < 0:
                                xn2_bub5_pos[m]+=l_box
                            if xn2_bub5_pos[m]>=l_box:
                                xn2_bub5_pos[m]-=l_box

                            if yn2_bub5_pos[m] < 0:
                                yn2_bub5_pos[m]+=l_box
                            if yn2_bub5_pos[m]>=l_box:
                                yn2_bub5_pos[m]-=l_box


            binArea = sizeBin**2

            #IDs of particles participating in each phase
            edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
            int_id_plot = np.where(partPhase==1)[0]         #All interfaces
            bulk_int_id_plot = np.where(partPhase!=2)[0]

            if len(bulk_ids)>0:
                bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
            else:
                bub_id_plot = []
            gas_id = np.where(partPhase==2)[0]

            #Determine positions of particles in each phase
            bulk_int_pos = pos[bulk_int_id_plot]
            bulk_pos = pos[bulk_id_plot]
            int_pos = pos[edge_id_plot]


            #Colors for plotting each phase
            yellow = ("#fdfd96")        #Largest gas-dense interface
            green = ("#77dd77")         #Bulk phase
            red = ("#ff6961")           #Gas phase
            purple = ("#cab2d6")        #Bubble or small gas-dense interfaces


            #IDs of particles participating in each phase
            edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
            int_id_plot = np.where(partPhase==1)[0]         #All interfaces
            bulk_int_id_plot = np.where(partPhase!=2)[0]

            if len(bulk_ids)>0:
                bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
            else:
                bub_id_plot = []
            gas_id = np.where(partPhase==2)[0]

            #If bulk/gas exist, calculate the structure ID for the gas/bulk
            if peA<=peB:
                slow_bulk_id_plot = np.where((partPhase==0) & (typ==0))[0]        #Bulk phase structure(s)
                fast_bulk_id_plot = np.where((partPhase==0) & (typ==1))[0]        #Bulk phase structure(s)
            else:
                slow_bulk_id_plot = np.where((partPhase==0) & (typ==1))[0]        #Bulk phase structure(s)
                fast_bulk_id_plot = np.where((partPhase==0) & (typ==0))[0]        #Bulk phase structure(s)
            bulk_id_plot = np.where(partPhase==0)[0]        #Bulk phase structure(s)
            if peA<=peB:
                slow_int_id_plot = np.where((partPhase==1) & (typ==0))[0]        #Bulk phase structure(s)
                fast_int_id_plot = np.where((partPhase==1) & (typ==1))[0]        #Bulk phase structure(s)
            else:
                slow_int_id_plot = np.where((partPhase==1) & (typ==1))[0]        #Bulk phase structure(s)
                fast_int_id_plot = np.where((partPhase==1) & (typ==0))[0]        #Bulk phase structure(s)
            edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
            int_id_plot = np.where(partPhase==1)[0]         #All interfaces
            if peA<=peB:
                slow_bulk_int_id_plot = np.where((partPhase!=2) & (typ==0))[0]        #Bulk phase structure(s)
                fast_bulk_int_id_plot = np.where((partPhase!=2) & (typ==1))[0]        #Bulk phase structure(s)
            else:
                slow_bulk_int_id_plot = np.where((partPhase!=2) & (typ==1))[0]        #Bulk phase structure(s)
                fast_bulk_int_id_plot = np.where((partPhase!=2) & (typ==0))[0]        #Bulk phase structure(s)
            bulk_int_id_plot = np.where(partPhase!=2)[0]
            if peA<=peB:
                slow_gas_int_id_plot = np.where((partPhase!=0) & (typ==0))[0]        #Bulk phase structure(s)
                fast_gas_int_id_plot = np.where((partPhase!=0) & (typ==1))[0]        #Bulk phase structure(s)
            else:
                slow_gas_int_id_plot = np.where((partPhase!=0) & (typ==1))[0]        #Bulk phase structure(s)
                fast_gas_int_id_plot = np.where((partPhase!=0) & (typ==0))[0]        #Bulk phase structure(s)
            gas_int_id_plot = np.where(partPhase!=0)[0]
            if peA<=peB:
                slow_gas_id_plot = np.where((partPhase==2) & (typ==0))[0]        #Bulk phase structure(s)
                fast_gas_id_plot = np.where((partPhase==2) & (typ==1))[0]        #Bulk phase structure(s)
            else:
                slow_gas_id_plot = np.where((partPhase==2) & (typ==1))[0]        #Bulk phase structure(s)
                fast_gas_id_plot = np.where((partPhase==2) & (typ==0))[0]        #Bulk phase structure(s)
            gas_id_plot = np.where(partPhase==2)[0]         #All interfaces
            print('percent_dif')
            print(((len(fast_bulk_id_plot)/len(bulk_id_plot) - ss_chi_f)/ss_chi_f)*100)

            if len(fast_bulk_id_plot)>0:

                bulk_ratio = len(fast_bulk_id_plot)/len(bulk_id_plot)
                if steady_state == 'True':

                    if ((bulk_ratio <= 1.01 * ss_chi_f) &  (0.99 * ss_chi_f <= bulk_ratio)):
                        steady_state_once = 'True'
                        if first_steady_state_frame == 'False':
                            print('analyze!')
                            #Arrays of particle data
                            snap_first = t[j]                                 #Take current frame

                            pos_first = snap_first.particles.position               # position
                            pos_first[:,-1] = 0.0

                            first_steady_state_frame = 'True'
                            bulk_id_plot_first = np.copy(bulk_id_plot)
                            slow_bulk_id_plot_first = np.copy(slow_bulk_id_plot)
                            fast_bulk_id_plot_first = np.copy(fast_bulk_id_plot)

                        print('rdf!')


                        pos0=pos[typ0ind]                               # Find positions of type 0 particles
                        pos0_bulk = pos[slow_bulk_id_plot]
                        pos0_int = pos[slow_int_id_plot]
                        pos0_gas = pos[slow_gas_id_plot]
                        pos0_bulk_int = pos[slow_bulk_int_id_plot]
                        pos0_gas_int = pos[slow_gas_int_id_plot]

                        pos1=pos[typ1ind]
                        pos1_bulk = pos[fast_bulk_id_plot]
                        pos1_int = pos[fast_int_id_plot]
                        pos1_gas = pos[fast_gas_id_plot]
                        pos1_bulk_int = pos[fast_bulk_int_id_plot]
                        pos1_gas_int = pos[fast_gas_int_id_plot]

                        pos_bulk = pos[bulk_id_plot]
                        pos_int = pos[int_id_plot]
                        pos_gas = pos[gas_id_plot]
                        pos_bulk_int = pos[bulk_int_id_plot]
                        pos_gas_int = pos[gas_int_id_plot]

                        pe_tot_int = 0
                        pe_num_int = 0
                        for i in range(0, len(int_id_plot)):

                            if typ[int_id_plot[i]]==0:
                                pe_tot_int += peA
                                pe_num_int += 1
                            else:
                                pe_tot_int += peB
                                pe_num_int += 1
                        pe_net_int = pe_tot_int / pe_num_int

                        lat_theory2 = conForRClust(pe_net_int-50, eps)
                        lat_theory_arr = np.append(lat_theory_arr, lat_theory2)

                        #a = system_all.points[cl_all.cluster_keys[large_clust_ind_all[0]]]
                        AA_in_large = np.array([])
                        BB_in_large=np.array([])
                        mark=0
                        r = np.linspace(0.0,  3.0, 20)             # Define radius for x-axis of plot later

                        # Width, in distance units, of bin
                        wBins = 0.02

                        # Distance to compute RDF for
                        rstop = 10.

                        # Number of bins given this distance
                        nBins = rstop / wBins

                        wbinsTrue=(rstop)/(nBins-1)

                        r=np.arange(0.0,rstop+wbinsTrue,wbinsTrue)
                        query_args = dict(mode='ball', r_min = 0.1, r_max=rstop)

                        common_bulk_id = np.intersect1d(bulk_id_plot, bulk_id_plot_first)
                        common_fast_bulk_id = np.intersect1d(fast_bulk_id_plot, fast_bulk_id_plot_first)
                        common_slow_bulk_id = np.intersect1d(slow_bulk_id_plot, slow_bulk_id_plot_first)


                        bulk_msd_dif_x = pos_current[common_bulk_id,0] - pos_first[common_bulk_id,0]

                        bulk_msd_dif_y = pos_current[common_bulk_id,1] - pos_first[common_bulk_id,1]

                        x_lim0 = np.where(bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            bulk_msd_dif_x[x_lim0] = bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            bulk_msd_dif_x[x_lim1] = bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            bulk_msd_dif_y[y_lim0] = bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            bulk_msd_dif_y[y_lim1] = bulk_msd_dif_y[y_lim1] + l_box


                        bulk_tot_disp = ( bulk_msd_dif_x ** 2 + bulk_msd_dif_y ** 2 ) ** 0.5
                        bulk_msd = np.mean(bulk_tot_disp ** 2)

                        bulk_msd_arr = np.append(bulk_msd_arr, bulk_msd)
                        msd_std = 0
                        msd_std_num = 0
                        for n in range(0, len(bulk_tot_disp)):
                            msd_std += ((bulk_tot_disp[n]**2)-bulk_msd)**2
                            msd_std_num +=1

                        bulk_msd_std_arr = np.append(bulk_msd_std_arr, (msd_std / msd_std_num)**0.5)


                        slow_bulk_msd_dif_x = pos_current[common_slow_bulk_id,0] - pos_first[common_slow_bulk_id,0]

                        slow_bulk_msd_dif_y = pos_current[common_slow_bulk_id,1] - pos_first[common_slow_bulk_id,1]

                        x_lim0 = np.where(slow_bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            slow_bulk_msd_dif_x[x_lim0] = slow_bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(slow_bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            slow_bulk_msd_dif_x[x_lim1] = slow_bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(slow_bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            slow_bulk_msd_dif_y[y_lim0] = slow_bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(slow_bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            slow_bulk_msd_dif_y[y_lim1] = slow_bulk_msd_dif_y[y_lim1] + l_box


                        slow_bulk_tot_disp = ( slow_bulk_msd_dif_x ** 2 + slow_bulk_msd_dif_y ** 2 ) ** 0.5
                        slow_bulk_msd = np.mean(slow_bulk_tot_disp ** 2)

                        slow_bulk_msd_arr = np.append(slow_bulk_msd_arr, slow_bulk_msd)
                        slow_msd_std = 0
                        slow_msd_std_num = 0
                        for n in range(0, len(slow_bulk_tot_disp)):
                            slow_msd_std += ((slow_bulk_tot_disp[n]**2)-slow_bulk_msd)**2
                            slow_msd_std_num +=1

                        slow_bulk_msd_std_arr = np.append(slow_bulk_msd_std_arr, (slow_msd_std / slow_msd_std_num)**0.5)


                        fast_bulk_msd_dif_x = pos_current[common_fast_bulk_id,0] - pos_first[common_fast_bulk_id,0]

                        fast_bulk_msd_dif_y = pos_current[common_fast_bulk_id,1] - pos_first[common_fast_bulk_id,1]

                        x_lim0 = np.where(fast_bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            fast_bulk_msd_dif_x[x_lim0] = fast_bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(fast_bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            fast_bulk_msd_dif_x[x_lim1] = fast_bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(fast_bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            fast_bulk_msd_dif_y[y_lim0] = fast_bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(fast_bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            fast_bulk_msd_dif_y[y_lim1] = fast_bulk_msd_dif_y[y_lim1] + l_box


                        fast_bulk_tot_disp = ( fast_bulk_msd_dif_x ** 2 + fast_bulk_msd_dif_y ** 2 ) ** 0.5
                        fast_bulk_msd = np.mean(fast_bulk_tot_disp ** 2)

                        fast_bulk_msd_arr = np.append(fast_bulk_msd_arr, fast_bulk_msd)
                        fast_msd_std = 0
                        fast_msd_std_num = 0
                        for n in range(0, len(fast_bulk_tot_disp)):
                            fast_msd_std += ((fast_bulk_tot_disp[n]**2)-fast_bulk_msd)**2
                            fast_msd_std_num +=1

                        fast_bulk_msd_std_arr = np.append(fast_bulk_msd_std_arr, (fast_msd_std / fast_msd_std_num)**0.5)


                        ss_time_arr = np.append(ss_time_arr, tst)
                        clust_size_arr = np.append(clust_size_arr, np.amax(clust_size))
                else:
                    if j >= int(dumps/2):
                        steady_state_once = 'True'
                        if first_steady_state_frame == 'False':
                            #Arrays of particle data
                            snap_first = t[0]                                 #Take current frame

                            pos_first = snap_first.particles.position               # position
                            pos_first[:,-1] = 0.0

                            snap_current = t[j]                                 #Take current frame
                            first_steady_state_frame = 'True'

                        print('rdf!2')
                        pos0=pos[typ0ind]                               # Find positions of type 0 particles
                        pos0_bulk = pos[slow_bulk_id_plot]
                        pos0_int = pos[slow_int_id_plot]
                        pos0_gas = pos[slow_gas_id_plot]
                        pos0_bulk_int = pos[slow_bulk_int_id_plot]
                        pos0_gas_int = pos[slow_gas_int_id_plot]

                        pos1=pos[typ1ind]
                        pos1_bulk = pos[fast_bulk_id_plot]
                        pos1_int = pos[fast_int_id_plot]
                        pos1_gas = pos[fast_gas_id_plot]
                        pos1_bulk_int = pos[fast_bulk_int_id_plot]
                        pos1_gas_int = pos[fast_gas_int_id_plot]

                        pos_bulk = pos[bulk_id_plot]
                        pos_int = pos[int_id_plot]
                        pos_gas = pos[gas_id_plot]
                        pos_bulk_int = pos[bulk_int_id_plot]
                        pos_gas_int = pos[gas_int_id_plot]

                        pe_tot_int = 0
                        pe_num_int = 0
                        for i in range(0, len(int_id_plot)):

                            if typ[int_id_plot[i]]==0:
                                pe_tot_int += peA
                                pe_num_int += 1
                            else:
                                pe_tot_int += peB
                                pe_num_int += 1
                        pe_net_int = pe_tot_int / pe_num_int

                        lat_theory2 = conForRClust(pe_net_int-50, eps)
                        lat_theory_arr = np.append(lat_theory_arr, lat_theory2)

                        #a = system_all.points[cl_all.cluster_keys[large_clust_ind_all[0]]]
                        AA_in_large = np.array([])
                        BB_in_large=np.array([])
                        mark=0
                        r = np.linspace(0.0,  3.0, 20)             # Define radius for x-axis of plot later

                        # Width, in distance units, of bin
                        wBins = 0.02

                        # Distance to compute RDF for
                        rstop = 10.

                        # Number of bins given this distance
                        nBins = rstop / wBins

                        wbinsTrue=(rstop)/(nBins-1)

                        r=np.arange(0.0,rstop+wbinsTrue,wbinsTrue)
                        query_args = dict(mode='ball', r_min = 0.1, r_max=rstop)

                        #MSD STUFF
                        common_bulk_id = np.intersect1d(bulk_id_plot, bulk_id_plot_first)
                        common_fast_bulk_id = np.intersect1d(fast_bulk_id_plot, fast_bulk_id_plot_first)
                        common_slow_bulk_id = np.intersect1d(slow_bulk_id_plot, slow_bulk_id_plot_first)


                        bulk_msd_dif_x = pos_current[common_bulk_id,0] - pos_first[common_bulk_id,0]

                        bulk_msd_dif_y = pos_current[common_bulk_id,1] - pos_first[common_bulk_id,1]

                        x_lim0 = np.where(bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            bulk_msd_dif_x[x_lim0] = bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            bulk_msd_dif_x[x_lim1] = bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            bulk_msd_dif_y[y_lim0] = bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            bulk_msd_dif_y[y_lim1] = bulk_msd_dif_y[y_lim1] + l_box


                        bulk_tot_disp = ( bulk_msd_dif_x ** 2 + bulk_msd_dif_y ** 2 ) ** 0.5
                        bulk_msd = np.mean(bulk_tot_disp ** 2)

                        bulk_msd_arr = np.append(bulk_msd_arr, bulk_msd)
                        msd_std = 0
                        msd_std_num = 0
                        for n in range(0, len(bulk_tot_disp)):
                            msd_std += ((bulk_tot_disp[n]**2)-bulk_msd)**2
                            msd_std_num +=1

                        bulk_msd_std_arr = np.append(bulk_msd_std_arr, (msd_std / msd_std_num)**0.5)

                        slow_bulk_msd_dif_x = pos_current[common_slow_bulk_id,0] - pos_first[common_slow_bulk_id,0]

                        slow_bulk_msd_dif_y = pos_current[common_slow_bulk_id,1] - pos_first[common_slow_bulk_id,1]

                        x_lim0 = np.where(slow_bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            slow_bulk_msd_dif_x[x_lim0] = slow_bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(slow_bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            slow_bulk_msd_dif_x[x_lim1] = slow_bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(slow_bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            slow_bulk_msd_dif_y[y_lim0] = slow_bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(slow_bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            slow_bulk_msd_dif_y[y_lim1] = slow_bulk_msd_dif_y[y_lim1] + l_box


                        slow_bulk_tot_disp = ( slow_bulk_msd_dif_x ** 2 + slow_bulk_msd_dif_y ** 2 ) ** 0.5
                        slow_bulk_msd = np.mean(slow_bulk_tot_disp ** 2)

                        slow_bulk_msd_arr = np.append(slow_bulk_msd_arr, slow_bulk_msd)
                        slow_msd_std = 0
                        slow_msd_std_num = 0
                        for n in range(0, len(slow_bulk_tot_disp)):
                            slow_msd_std += ((slow_bulk_tot_disp[n]**2)-slow_bulk_msd)**2
                            slow_msd_std_num +=1

                        slow_bulk_msd_std_arr = np.append(slow_bulk_msd_std_arr, (slow_msd_std / slow_msd_std_num)**0.5)

                        fast_bulk_msd_dif_x = pos_current[common_fast_bulk_id,0] - pos_first[common_fast_bulk_id,0]

                        fast_bulk_msd_dif_y = pos_current[common_fast_bulk_id,1] - pos_first[common_fast_bulk_id,1]

                        x_lim0 = np.where(fast_bulk_msd_dif_x > h_box)[0]
                        if len(x_lim0)>0:
                            fast_bulk_msd_dif_x[x_lim0] = fast_bulk_msd_dif_x[x_lim0] - l_box

                        x_lim1 = np.where(fast_bulk_msd_dif_x < -h_box)[0]
                        if len(x_lim1)>0:
                            fast_bulk_msd_dif_x[x_lim1] = fast_bulk_msd_dif_x[x_lim1] + l_box

                        y_lim0 = np.where(fast_bulk_msd_dif_y > h_box)[0]
                        if len(y_lim0)>0:
                            fast_bulk_msd_dif_y[y_lim0] = fast_bulk_msd_dif_y[y_lim0] - l_box

                        y_lim1 = np.where(fast_bulk_msd_dif_y < -h_box)[0]
                        if len(y_lim1)>0:
                            fast_bulk_msd_dif_y[y_lim1] = fast_bulk_msd_dif_y[y_lim1] + l_box


                        fast_bulk_tot_disp = ( fast_bulk_msd_dif_x ** 2 + fast_bulk_msd_dif_y ** 2 ) ** 0.5
                        fast_bulk_msd = np.mean(fast_bulk_tot_disp ** 2)

                        fast_bulk_msd_arr = np.append(fast_bulk_msd_arr, fast_bulk_msd)
                        fast_msd_std = 0
                        fast_msd_std_num = 0
                        for n in range(0, len(fast_bulk_tot_disp)):
                            fast_msd_std += ((fast_bulk_tot_disp[n]**2)-fast_bulk_msd)**2
                            fast_msd_std_num +=1

                        fast_bulk_msd_std_arr = np.append(fast_bulk_msd_std_arr, (fast_msd_std / fast_msd_std_num)**0.5)

                        ss_time_arr = np.append(ss_time_arr, tst)
                        clust_size_arr = np.append(clust_size_arr, np.amax(clust_size))


if steady_state_once == 'True':

    #ss_inds = np.where(r>=0.75*rstop)[0]
    fsize=10

    fig, ax1 = plt.subplots(figsize=(12,6))

    bulk_msd_max = np.max(bulk_msd_arr)
    slow_bulk_msd_max = np.max(slow_bulk_msd_arr)
    fast_bulk_msd_max = np.max(fast_bulk_msd_arr)

    if (bulk_msd_max >= slow_bulk_msd_max) & (bulk_msd_max >= fast_bulk_msd_max):
        plot_max = 1.05 * np.max(bulk_msd_max)
    elif (slow_bulk_msd_max >= bulk_msd_max) & (slow_bulk_msd_max >= fast_bulk_msd_max):
        plot_max = 1.05 * np.max(slow_bulk_msd_max)
    elif (fast_bulk_msd_max >= slow_bulk_msd_max) & (fast_bulk_msd_max >= bulk_msd_max):
        plot_max = 1.05 * np.max(fast_bulk_msd_max)


    plot_min = 0.0

    step = np.round(np.abs(plot_max - plot_min)/6,2)
    step_x = np.round(np.abs(np.max(ss_time_arr)-ss_time_arr[0])/10,2)
    fastCol = '#e31a1c'
    slowCol = '#081d58'
    purple = ("#756bb1")
    pink = ("#dd1c77")
    if peA <= peB:
        #plt.plot(r, rdf_all_bulk_rdf, label=r'All',
        #               c='black', lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], bulk_msd_arr, label=r'All',
                       c=purple, lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], slow_bulk_msd_arr, label=r'Slow',
                       c=slowCol, lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], fast_bulk_msd_arr, label=r'Fast',
                       c=fastCol, lw=1.8*1.2, ls='--')
    else:
        #plt.plot(r, rdf_all_bulk_rdf, label=r'All',
        #               c='black', lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], bulk_msd_arr, label=r'All',
                    c='black', lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], fast_bulk_msd_arr, label=r'Slow',
                    c=slowCol, lw=1.8*1.2, ls='--')
        plt.plot(ss_time_arr-ss_time_arr[0], slow_bulk_msd_arr, label=r'Fast',
                    c=fastCol, lw=1.8*1.2, ls='--')

    ax1.set_ylim(plot_min, plot_max)


    ax1.set_xlabel(r'Time ($\tau_\mathrm{B}$)', fontsize=fsize*2.8)



    ax1.set_ylabel(r'MSD', fontsize=fsize*2.8)

    # Set all the x ticks for radial plots
    loc = ticker.MultipleLocator(base=step_x)
    ax1.xaxis.set_major_locator(loc)
    loc = ticker.MultipleLocator(base=round(step_x/2,3))
    ax1.xaxis.set_minor_locator(loc)
    ax1.set_xlim(0, np.max(ss_time_arr)-ss_time_arr[0])
    plt.legend(loc='upper left', fontsize=fsize*2.6)

    # Set y ticks
    print(step)
    loc = ticker.MultipleLocator(base=step)
    ax1.yaxis.set_major_locator(loc)
    loc = ticker.MultipleLocator(base=round(step/2,3))
    ax1.yaxis.set_minor_locator(loc)
    # Left middle plot
    ax1.tick_params(axis='x', labelsize=fsize*2.5)
    ax1.tick_params(axis='y', labelsize=fsize*2.5)
    #plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(outPath + 'msd_' + out + ".png", dpi=300)
    plt.close()


    g = open(outPath2+outTxt_msd, 'a')
    for i in range(0, len(ss_time_arr)):
        g.write('{0:.2f}'.format(ss_time_arr[i]).center(15) + ' ')
        g.write('{0:.6f}'.format(sizeBin).center(15) + ' ')
        g.write('{0:.0f}'.format(clust_size_arr[i]).center(15) + ' ')
        g.write('{0:.6f}'.format(bulk_msd_arr[i]).center(15) + ' ')
        g.write('{0:.6f}'.format(bulk_msd_std_arr[i]).center(15) + ' ')
        if peA<=peB:
            g.write('{0:.6f}'.format(slow_bulk_msd_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(slow_bulk_msd_std_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(fast_bulk_msd_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(fast_bulk_msd_std_arr[i]).center(15) + '\n')
        else:
            g.write('{0:.6f}'.format(fast_bulk_msd_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(fast_bulk_msd_std_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(slow_bulk_msd_arr[i]).center(15) + ' ')
            g.write('{0:.6f}'.format(slow_bulk_msd_std_arr[i]).center(15) + '\n')
    g.close()
