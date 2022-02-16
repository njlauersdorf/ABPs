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
import time

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

from contextlib import closing
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
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
import multiprocessing as mp


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
    
    #Difference in x,y positions
    difx = x2-x1
    dify = y2-y1
    
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
                            
    fx = f * (difx) / r
    fy = f * (dify) / r
    return fx, fy

#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)
tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size
n_len = 21
n_arr = np.linspace(0, n_len-1, n_len)      #Fourier modes
popt_sum = np.zeros(n_len)                  #Fourier Coefficients


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
out = outfile + "_frame_"  

outTxt_lat = 'lat_' + outfile + '.txt'

g = open(outPath2+outTxt_lat, 'w+') # write file headings
g.write('tauB'.center(20) + ' ' +\
                        'sizeBin'.center(20) + ' ' +\
                        'clust_size'.center(20) + ' ' +\
                        'lat_mean_bulk'.center(20) + ' ' +\
                        'lat_mean_int'.center(20) + ' ' +\
                        'lat_mean_bub'.center(20) + ' ' +\
                        'lat_mean_all'.center(20) + ' ' +\
                        'lat_std_bulk'.center(20) + ' ' +\
                        'lat_std_int'.center(20) + ' ' +\
                        'lat_std_bub'.center(20) + ' ' +\
                        'lat_std_all'.center(20) + '\n')
g.close()

with hoomd.open(name=inFile, mode='rb') as t:
    
    start = int(0/time_step)#205                                             # first frame to process
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = int(dumps/time_step)-1                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step
    
    box_data = snap.configuration.box # Get box dimensions
    l_box = box_data[0]                                     #box length
    h_box = l_box / 2.0     #Take current frame

    #2D binning of system
    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / NBins), 6)
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    
    time_arr=np.zeros(dumps)                                  #time step array
tSteps=np.linspace(0,dumps,dumps+1).astype(int)

def lattice(p):
    j=int(p*time_step)
    print('j')
    print(j)
    with hoomd.open(name=inFile, mode='rb') as t:
        snap = t[j]
    
    
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
    
    #shift reference frame to center of mass of cluster
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
                    
        
    #Colors for plotting each phase
    yellow = ("#fdfd96")        #Largest gas-dense interface
    green = ("#77dd77")         #Bulk phase
    red = ("#ff6961")           #Gas phase
    purple = ("#cab2d6")        #Bubble or small gas-dense interfaces
           
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
                    num_densDif[ix][iy]=num_dens3B[ix][iy]-num_dens3A[ix][iy]                   #Difference in number density

                    #average x,y orientation per bin
                    p_avg_x[ix][iy] = p_all_x[ix][iy]/len(binParts[ix][iy])
                    p_avg_y[ix][iy] = p_all_y[ix][iy]/len(binParts[ix][iy])
                    
                    if j>(start*time_step):
                        v_avg_x[ix][iy] = v_all_x[ix][iy]/len(binParts[ix][iy])
                        v_avg_y[ix][iy] = v_all_y[ix][iy]/len(binParts[ix][iy])
                    #average x,y orientation per bin for A type particles
                    if typ0_temp>0:
                        p_avg_xA[ix][iy] = p_all_xA[ix][iy]/typ0_temp
                        p_avg_yA[ix][iy] = p_all_yA[ix][iy]/typ0_temp
                        if j>(start*time_step):
                            v_avg_xA[ix][iy] = v_all_xA[ix][iy]/typ0_temp
                            v_avg_yA[ix][iy] = v_all_yA[ix][iy]/typ0_temp
                    else:
                        p_avg_xA[ix][iy] = 0.0
                        p_avg_yA[ix][iy] = 0.0
                        if j>(start*time_step):
                            v_avg_xA[ix][iy] = 0.0
                            v_avg_yA[ix][iy] = 0.0
                    
                    #average x,y orientation per bin for B type particles
                    if typ1_temp>0:
                        p_avg_xB[ix][iy] = p_all_xB[ix][iy]/typ1_temp
                        p_avg_yB[ix][iy] = p_all_yB[ix][iy]/typ1_temp
                        if j>(start*time_step):
                            v_avg_xB[ix][iy] = v_all_xB[ix][iy]/typ1_temp
                            v_avg_yB[ix][iy] = v_all_yB[ix][iy]/typ1_temp
                    else:
                        p_avg_xB[ix][iy] = 0.0
                        p_avg_yB[ix][iy] = 0.0
                        if j>(start*time_step):
                            v_avg_xB[ix][iy] = 0.0
                            v_avg_yB[ix][iy] = 0.0
                        
                    
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
                    
    #Initiate empty arrays
    align_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
    align_combinedA = np.zeros((len(v_avg_x), len(v_avg_y),2))
    align_combinedB = np.zeros((len(v_avg_x), len(v_avg_y),2))
    align_combinedDif = np.zeros((len(v_avg_x), len(v_avg_y),2))
    pos_box_combined_align = np.zeros((len(v_avg_x), len(v_avg_y),2))
    
    #Loop over bins and save calculated alignment to a (:,:,2) array instead of (:,:)
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

    #Calculate gradient of alignment over x (axis=0) and y (axis=1) directions
    alignx_grad = np.gradient(align_combined, axis=0)
    aligny_grad = np.gradient(align_combined, axis=1)
    
    #Calculate gradient of type A alignment over x (axis=0) and y (axis=1) directions
    alignx_gradA = np.gradient(align_combinedA, axis=0)
    aligny_gradA = np.gradient(align_combinedA, axis=1)
    
    #Calculate gradient of type B alignment over x (axis=0) and y (axis=1) directions
    alignx_gradB = np.gradient(align_combinedB, axis=0)
    aligny_gradB = np.gradient(align_combinedB, axis=1)
    
    #Calculate gradient of alignment difference over x (axis=0) and y (axis=1) directions
    alignx_gradDif = np.gradient(align_combinedDif, axis=0)
    aligny_gradDif = np.gradient(align_combinedDif, axis=1)
    
    #Calculate gradient of number density over x (axis=0) and y (axis=1) directions
    num_densx_grad = np.gradient(num_dens3, axis=0)
    num_densy_grad = np.gradient(num_dens3, axis=1)
    
    align_gradx_x = alignx_grad[:,:,0]      #Calculate gradient of x-alignment over x direction
    align_gradx_y = alignx_grad[:,:,1]      #Calculate gradient of y-alignment over x direction
    align_grady_x = aligny_grad[:,:,0]      #Calculate gradient of x-alignment over y direction
    align_grady_y = aligny_grad[:,:,1]      #Calculate gradient of y-alignment over y direction
    
    align_gradx_xA = alignx_gradA[:,:,0]    #Calculate gradient of x-alignment of type A particles over x direction
    align_gradx_yA = alignx_gradA[:,:,1]    #Calculate gradient of y-alignment of type A particles over x direction
    align_grady_xA = aligny_gradA[:,:,0]    #Calculate gradient of x-alignment of type A particles over y direction
    align_grady_yA = aligny_gradA[:,:,1]    #Calculate gradient of y-alignment of type A particles over y direction
    
    align_gradx_xB = alignx_gradB[:,:,0]    #Calculate gradient of x-alignment of type B particles over x direction
    align_gradx_yB = alignx_gradB[:,:,1]    #Calculate gradient of y-alignment of type B particles over x direction
    align_grady_xB = aligny_gradB[:,:,0]    #Calculate gradient of x-alignment of type B particles over y direction
    align_grady_yB = aligny_gradB[:,:,1]    #Calculate gradient of y-alignment of type B particles over y direction
    
    
    align_gradx_xDif = alignx_gradDif[:,:,0]    #Calculate gradient of x-alignment difference over x direction
    align_gradx_yDif = alignx_gradDif[:,:,1]    #Calculate gradient of y-alignment difference over x direction
    align_grady_xDif = aligny_gradDif[:,:,0]    #Calculate gradient of x-alignment difference over y direction
    align_grady_yDif = aligny_gradDif[:,:,1]    #Calculate gradient of y-alignment difference over y direction
               
    #Calculate divergence of all alignment
    div_align = align_gradx_x + align_grady_y
    curl_align = -align_grady_x + align_gradx_y
    
    #Calculate divergence of type A alignment
    div_alignA = align_gradx_xA + align_grady_yA
    curl_alignA = -align_grady_xA + align_gradx_yA
    
    #Calculate divergence of type B alignment
    div_alignB = align_gradx_xB + align_grady_yB
    curl_alignB = -align_grady_xB + align_gradx_yB
    
    #Calculate divergence of alignment difference between type B and A particles
    div_alignDif = align_gradx_xDif + align_grady_yDif
    curl_alignDif = -align_grady_xDif + align_gradx_yDif
    
    #Calculate divergence of number density
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
                
                #Pressure integrand (number density times alignment) per bin
                press_int[ix][iy] = num_dens3[ix][iy]*(align_avg_x[ix][iy]**2+align_avg_y[ix][iy]**2)**0.5
                
                #Alignment of all particles per bin
                align_mag[ix][iy] = (align_avg_x[ix][iy]**2+align_avg_y[ix][iy]**2)**0.5
                
                #Alignment of type A particles per bin
                align_magA[ix][iy] = (align_avg_xA[ix][iy]**2+align_avg_yA[ix][iy]**2)**0.5
                
                #Alignment of type B particles per bin
                align_magB[ix][iy] = (align_avg_xB[ix][iy]**2+align_avg_yB[ix][iy]**2)**0.5
                
                #Difference in alignment of type B to type A particles per bin
                align_magDif[ix][iy] = (align_magB[ix][iy]-align_magA[ix][iy])#(align_avg_xDif[ix][iy]**2+align_avg_yDif[ix][iy]**2)**0.5
                
                #If 2nd time step or higher, continue...
                if j>(start*time_step):
                    
                    #Velocity of all particles per bin
                    vel_mag[ix][iy] = ((v_avg_x[ix][iy]**2+v_avg_y[ix][iy]**2)**0.5)#/peB
                    
                    #Velocity of type A particles per bin
                    vel_magA[ix][iy] = ((v_avg_xA[ix][iy]**2+v_avg_yA[ix][iy]**2)**0.5)#/peA
                    
                    #Velocity of type B particles per bin
                    vel_magB[ix][iy] = ((v_avg_xB[ix][iy]**2+v_avg_yB[ix][iy]**2)**0.5)#/peB
                    
                    #Difference in velocity of type B to type A particles per bin
                    vel_magDif[ix][iy] = (vel_magB[ix][iy]*peB-vel_magA[ix][iy]*peA)#(align_avg_xDif[ix][iy]**2+align_avg_yDif[ix][iy]**2)**0.5
                
                #Pressure integrand (number density times alignment) of all particles per bin
                press_bin[ix][iy] = num_dens3[ix][iy]*align_mag[ix][iy]#*Binpe[ix][iy]
                
                #Pressure integrand (number density times alignment) of type A particles per bin
                press_binA[ix][iy] = num_dens3A[ix][iy]*align_magA[ix][iy]#*peA
                
                #Pressure integrand (number density times alignment) of type B particles per bin
                press_binB[ix][iy] = num_dens3B[ix][iy]*align_magB[ix][iy]#*peB
                
                #Difference in pressure integrand of type B to type A particles per bin
                press_binDif[ix][iy] = (num_dens3B[ix][iy]*align_magB[ix][iy])-(num_dens3A[ix][iy]*align_magA[ix][iy])

                #Calculate x,y-components of normalized alignment (0 to 1) with surface normal of all particles per bin
                if align_mag[ix][iy]>0:
                    align_norm_x[ix][iy] = align_avg_x[ix][iy] / align_mag[ix][iy]
                    align_norm_y[ix][iy] = align_avg_y[ix][iy] / align_mag[ix][iy]
            
                #Calculate x,y-components normalized alignment (0 to 1) of type A particles per bin
                if align_magA[ix][iy]>0:
                    align_norm_xA[ix][iy] = align_avg_xA[ix][iy] / align_magA[ix][iy]
                    align_norm_yA[ix][iy] = align_avg_yA[ix][iy] / align_magA[ix][iy]
                
                #Calculate x,y-components normalized alignment (0 to 1) of type B particles per bin
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
                    
    
    #Initiate empty arrays
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
    '''
    bulkSigXX = np.zeros(5)
    bulkSigYY = np.zeros(5)
    bulkSigXY = np.zeros(5)
    bulkSigYX = np.zeros(5)
    bulk_area_arr = np.zeros(5)
    binArea = sizeBin * sizeBin
    if len(bulk_large_ids)>0:
        for m in range(0, len(bulk_large_ids)):
            for ix in range(0, len(bulk_id2)):
                for iy in range(0, len(bulk_id2)):
                    if bulk_id2[ix][iy]==bulk_large_ids[m]:
                        bulk_area_arr[m]+=1
            bulk_area_arr[m] = bulk_area_arr[m] * binArea
    
    if len(bulk_large_ids)>0:
        for m in range(0, len(bulk_large_ids)):
            
            bulk_temp = np.where(bulkPhase==bulk_large_ids[m])[0]
            
            bulk_pos = pos[bulk_temp,:]
            #Compute neighbor list for 6-nearest neighbors given particle positions
            
            system_all = freud.AABBQuery(f_box, f_box.wrap(bulk_pos))
            nlist2 = system_all.query(f_box.wrap(bulk_pos), dict(r_max=r_cut, exclude_ii=True))
            
            #Set empty arrays
            point_ind_arr = np.array([])
            point_query_arr = np.array([])
            difr = np.array([])
            
            #Save neighbor indices and distances from neighbor list to array
            for bond in nlist2:
                
                    
                #Difference in x,y positions
                difx = pos[bond[1],0]-pos[bond[0],0]
                dify = pos[bond[1],1]-pos[bond[0],1]
                
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
                
                #distance between points
                difr = (difx**2 + dify**2)**0.5
                
                if 0.1 < difr <= r_cut:
                    
                    # Compute the x and y components of force
                    fx, fy = computeFLJ(difr, pos[bond[0],0], pos[bond[0],1], pos[bond[1],0], pos[bond[1],0], eps)
                    # This will go into the bulk pressure
                    bulkSigXX[m] += (fx * (difx))
                    bulkSigYY[m] += (fy * (dify))
                    bulkSigXY[m] += (fx * (dify))
                    bulkSigYX[m] += (fy * (difx))
    bulkTrace = ((bulkSigXX + bulkSigYY)/2)
    bulkPress = np.zeros(5)
    for k in range(0, len(bulkTrace)):
        if bulk_area_arr[k]>0:
            bulkPress[m] = (bulkTrace[m]/bulk_area_arr[m])/2
    print(bulkPress)
    stop
    '''
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
    
    #Sort interface structures by size with largest cluster corresponding to first bulk phase and decending in size until, at most, the fifth largest bulk is labeled

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
    
    #Sort bulk structures by size with largest cluster corresponding to first bulk phase and decending in size until, at most, the fifth largest bulk is labeled
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
                        
    
    #Initiate empty arrays
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
                
            elif int_edge_id[ix][iy]==1:
                int_pos_box_x_arr=np.append(int_pos_box_x_arr, (ix+0.5)*sizeBin)
                int_pos_box_y_arr=np.append(int_pos_box_y_arr, (iy+0.5)*sizeBin)
                            
    #Sort arrays of points defining exterior and interior surfaces of interface so adjacent points are next to eachother in array
    
    #if 2nd time step or higher, continue...
    if j>(start*time_step):
        
        #Sort points defining exterior surface of interface
        while len(ext_pos_box_x_arr)>0:
            
            #If no particles in the sorted array, append the first point of unsorted array (this will be starting point to determine adjacent points)
            if len(ext_vert_x)==0:
                ext_vert_x = np.append(ext_vert_x, ext_pos_box_x_arr[0])
                ext_vert_y = np.append(ext_vert_y, ext_pos_box_y_arr[0])
                
                ext_pos_box_x_arr = np.delete(ext_pos_box_x_arr, 0)
                ext_pos_box_y_arr = np.delete(ext_pos_box_y_arr, 0)
                
                ext_codes = np.append(ext_codes, Path.MOVETO)
            
            #If at least one point in sorted array, find next nearest point
            else:
                shortest_length = 100000
                
                #Loop over all points in exterior surface to find next closest point to most recently appended sorted point
                for iy in range(0, len(ext_pos_box_y_arr)):
                    
                    #Difference in x,y positions
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
                    
                    #distance between points
                    difr = (difx**2 + dify**2)**0.5
                    
                    #If distance between points lesser than previously determined shortest distance, replace these values
                    if difr < shortest_length:
                        shortest_length = difr
                        shortest_xlength = difx
                        shortest_ylength = dify
                        shortest_id = iy
                        
                    #If the distance is equal, favor points to right (greater x or y values) of reference point
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
                        
                #Save nearest point of exterior surface of interface determined by loop
                ext_vert_x = np.append(ext_vert_x, ext_pos_box_x_arr[shortest_id])
                ext_vert_y = np.append(ext_vert_y, ext_pos_box_y_arr[shortest_id])
                
                ext_pos_box_x_arr = np.delete(ext_pos_box_x_arr, shortest_id)
                ext_pos_box_y_arr = np.delete(ext_pos_box_y_arr, shortest_id)
       
        #Sort points defining interior surface of interface
        while len(int_pos_box_x_arr)>0:
            
            #If no particles in the sorted array, append the first point of unsorted array (this will be starting point to determine adjacent points)
            if len(int_vert_x)==0:
                int_vert_x = np.append(int_vert_x, int_pos_box_x_arr[0])
                int_vert_y = np.append(int_vert_y, int_pos_box_y_arr[0])
                
                int_pos_box_x_arr = np.delete(int_pos_box_x_arr, 0)
                int_pos_box_y_arr = np.delete(int_pos_box_y_arr, 0)
                
                int_codes = np.append(int_codes, Path.MOVETO)
            
            #If at least one point in sorted array, find next nearest point
            else:
                shortest_length = 100000
                
                #Loop over all points in interior surface to find next closest point to most recently appended sorted point
                for iy in range(0, len(int_pos_box_y_arr)):
                    
                    #Difference in x,y positions
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
                                
                    #distance between points
                    difr = (difx**2 + dify**2)**0.5
                    
                    #If distance between points lesser than previously determined shortest distance, replace these values
                    if difr < shortest_length:
                        shortest_length = difr
                        shortest_xlength = difx
                        shortest_ylength = dify
                        shortest_id = iy
                    
                    #If the distance is equal, favor points to right (greater x or y values) of reference point
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
                 
                #Save nearest point of interior surface of interface determined by loop
                int_vert_x = np.append(int_vert_x, int_pos_box_x_arr[shortest_id])
                int_vert_y = np.append(int_vert_y, int_pos_box_y_arr[shortest_id])
                int_pos_box_x_arr = np.delete(int_pos_box_x_arr, shortest_id)
                int_pos_box_y_arr = np.delete(int_pos_box_y_arr, shortest_id)
        
    
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
    
    v_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))

    align_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))

    pos_box_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
    
    #if currently 2nd time step or higher, continue
    if j>(start*time_step):
        
        #Combine previously calculated arrays to a higher dimension matrix (:,:,2) instead of (:,:)
        for ix in range(0, len(v_avg_x)):
            for iy in range(0, len(v_avg_y)):
                align_combined[ix][iy][0]=align_avg_x[ix][iy]
                align_combined[ix][iy][1]=align_avg_y[ix][iy]
                
                pos_box_combined[ix][iy][0]=pos_box_x[ix][iy]
                pos_box_combined[ix][iy][1]=pos_box_y[ix][iy]
                


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
    bub1_parts = np.array([])
    bub2_parts = np.array([])
    bub3_parts = np.array([])
    bub4_parts = np.array([])
    bub5_parts = np.array([])
    for m in range(0, len(bub_large_ids)):
        if m==0:
            bub1_parts = np.where(edgePhase==bub_large_ids[m])[0]
        elif m==1:
            bub2_parts = np.where(edgePhase==bub_large_ids[m])[0]
        elif m==2:
            bub3_parts = np.where(edgePhase==bub_large_ids[m])[0]
        elif m==3:
            bub4_parts = np.where(edgePhase==bub_large_ids[m])[0]
        elif m==4:
            bub5_parts = np.where(edgePhase==bub_large_ids[m])[0]
            
    #Colors for plotting each phase
    yellow = ("#fdfd96")        #Largest gas-dense interface
    green = ("#77dd77")         #Bulk phase
    red = ("#ff6961")           #Gas phase
    purple = ("#cab2d6")        #Bubble or small gas-dense interfaces

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
    
    #label previous positions for velocity calculation
    
    
    #Positions of particles in each phase
    bulk_int_pos = pos[bulk_int_id_plot]
    bulk_pos = pos[bulk_id_plot]
    int_pos = pos[edge_id_plot]
    
    #Compute neighbor list for 6-nearest neighbors given particle positions
    system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
    nlist2 = system_all.query(f_box.wrap(bulk_int_pos), dict(num_neighbors=7))
    
    #Set empty arrays
    point_ind_arr = np.array([], dtype=int)
    point_query_arr = np.array([], dtype=int)
    difr = np.array([])
    
    #Save neighbor indices and distances from neighbor list to array
    for bond in nlist2:
        if bond[2]>0:
            point_ind_arr = np.append(point_ind_arr, bond[0])
            point_query_arr = np.append(point_query_arr, bond[1])
            difr = np.append(difr, bond[2])
        
    
    #Set empty arrays
    lat_mean_indiv_arr_bulk = np.array([])
    lat_mean_indiv_arr_gas = np.array([])
    lat_mean_indiv_arr_int = np.array([])
    lat_mean_indiv_arr_bub = np.array([])
    lat_mean_indiv_arr = np.array([])
    pos_x_bulk = np.array([])
    pos_y_bulk = np.array([])
    pos_x_int = np.array([])
    pos_y_int = np.array([])
    pos_x_gas = np.array([])
    pos_y_gas = np.array([])
    pos_x_bub = np.array([])
    pos_y_bub = np.array([])
    lat_mean_arr = np.array([])
    lat_std_arr_bulk = np.array([])
    lat_std_arr_int = np.array([])
    lat_std_arr_gas = np.array([])
    
    for i in range(0, len(bulk_int_id_plot)):
        pair_ids = np.where(point_ind_arr==i)[0]
        
        #If reference bulk or interface particle identified in neighbor list, continue
        if len(pair_ids)>0:
            
            #Distance to six nearest neighbor from reference bulk or interface particle
            difr_pair_ids = difr[pair_ids]
            
            #Calculate and save mean lattice spacing
            lat_mean_val = np.mean(difr_pair_ids)
            lat_mean_indiv_arr = np.append(lat_mean_indiv_arr, lat_mean_val)
            
            #If bulk particle, save lattice spacing and positions to array
            if partPhase[bulk_int_id_plot[i]]==0:
                lat_mean_indiv_arr_bulk = np.append(lat_mean_indiv_arr_bulk, lat_mean_val)
                pos_x_bulk = np.append(pos_x_bulk, pos[bulk_int_id_plot[i],0])
                pos_y_bulk = np.append(pos_y_bulk, pos[bulk_int_id_plot[i],1])

            #If interface particle, save lattice spacing and positions to array
            elif partPhase[bulk_int_id_plot[i]]==1:
                if edgePhase[bulk_int_id_plot[i]]==interface_id:
                    lat_mean_indiv_arr_int = np.append(lat_mean_indiv_arr_int, lat_mean_val)
                    pos_x_int = np.append(pos_x_int, pos[bulk_int_id_plot[i],0])
                    pos_y_int = np.append(pos_y_int, pos[bulk_int_id_plot[i],1])
                else:
                    lat_mean_indiv_arr_bub = np.append(lat_mean_indiv_arr_bub, lat_mean_val)
                    pos_x_bub = np.append(pos_x_bub, pos[bulk_int_id_plot[i],0])
                    pos_y_bub = np.append(pos_y_bub, pos[bulk_int_id_plot[i],1])

            #If gas particle (should not be currently), save lattice spacing and positions to array
            elif partPhase[bulk_int_id_plot[i]]==2:
                lat_mean_indiv_arr_gas = np.append(lat_mean_indiv_arr_gas, lat_mean_val)
                pos_x_gas = np.append(pos_x_gas, pos[bulk_int_id_plot[i],0])
                pos_y_gas = np.append(pos_y_gas, pos[bulk_int_id_plot[i],1])

        #If reference particle is not in neighbor list, save 0 lattice spacing
        else:
            
            lat_mean_indiv_arr_bulk = np.append(lat_mean_indiv_arr_bulk, 0)
            lat_mean_indiv_arr_int = np.append(lat_mean_indiv_arr_int, 0)
            lat_mean_indiv_arr_gas = np.append(lat_mean_indiv_arr_gas, 0)
            lat_mean_indiv_arr_bub = np.append(lat_mean_indiv_arr_bub, 0)
            lat_mean_indiv_arr = np.append(lat_mean_indiv_arr, 0)
    
    #Calculate standard deviation of bulk lattice spacings
    if len(lat_mean_indiv_arr_bulk) >0:
        lat_std_num = 0
        lat_std_val = 0
        lat_mean_bulk = np.mean(lat_mean_indiv_arr_bulk)
        for k in range(0, len(lat_mean_indiv_arr_bulk)):
            lat_std_val += (lat_mean_indiv_arr_bulk[k]-lat_mean_bulk)**2
            lat_std_num += 1
        std_dev_bulk = (lat_std_val / lat_std_num)**0.5
    else:
        lat_mean_bulk=0
        std_dev_bulk=0
    
    #Calculate standard deviation of interface lattice spacings
    if len(lat_mean_indiv_arr_int) >0:
        lat_std_num = 0
        lat_std_val = 0
        lat_mean_int = np.mean(lat_mean_indiv_arr_int)
        for k in range(0, len(lat_mean_indiv_arr_int)):
            lat_std_val += (lat_mean_indiv_arr_int[k]-lat_mean_int)**2
            lat_std_num += 1
        std_dev_int = (lat_std_val / lat_std_num)**0.5
    else:
        lat_mean_int=0
        std_dev_int=0
        
    #Calculate standard deviation of bubble lattice spacings
    if len(lat_mean_indiv_arr_bub) >0:
        lat_std_num = 0
        lat_std_val = 0
        lat_mean_bub = np.mean(lat_mean_indiv_arr_bub)
        for k in range(0, len(lat_mean_indiv_arr_bub)):
            lat_std_val += (lat_mean_indiv_arr_bub[k]-lat_mean_bub)**2
            lat_std_num += 1
        std_dev_bub = (lat_std_val / lat_std_num)**0.5
    else:
        lat_mean_bub=0
        std_dev_bub=0
    
    #Calculate standard deviation of all lattice spacings
    if len(lat_mean_indiv_arr) >0:
        lat_std_num = 0
        lat_std_val = 0
        lat_mean_all = np.mean(lat_mean_indiv_arr)
        for k in range(0, len(lat_mean_indiv_arr)):
            lat_std_val += (lat_mean_indiv_arr[k]-lat_mean_all)**2
            lat_std_num += 1
        std_dev_all = (lat_std_val / lat_std_num)**0.5
    else:
        lat_mean_all=0
        std_dev_all=0
    '''
    #Save lattice spacing means and standard deviations of each phase to txt file
    print(outPath2+outTxt_lat)
    stop
    g = open(outPath2+outTxt_lat, 'a')
    g.write('{0:.2f}'.format(tst).center(20) + ' ')
    g.write('{0:.6f}'.format(sizeBin).center(20) + ' ')
    g.write('{0:.0f}'.format(np.amax(clust_size)).center(20) + ' ')
    g.write('{0:.6f}'.format(lat_mean_bulk).center(20) + ' ')
    g.write('{0:.6f}'.format(lat_mean_int).center(20) + ' ')
    g.write('{0:.6f}'.format(lat_mean_bub).center(20) + ' ')
    g.write('{0:.6f}'.format(lat_mean_all).center(20) + ' ')
    g.write('{0:.6f}'.format(std_dev_bulk).center(20) + ' ')
    g.write('{0:.6f}'.format(std_dev_int).center(20) + ' ')
    g.write('{0:.6f}'.format(std_dev_bub).center(20) + ' ')
    g.write('{0:.6f}'.format(std_dev_all).center(20) + '\n')
    g.close()
    '''
    
    
    #Plot scatter plot of bulk and interface particles color-coded by lattice spacing
    
    #If bulk or interface particles identified, continue
    if (len(lat_mean_indiv_arr_bulk)>0) or (len(lat_mean_indiv_arr_int)>0):
        pad = str(j).zfill(4)
        x_pos_plot = np.append(pos_x_bulk+h_box, pos_x_int+h_box)
        x_pos_final = np.append(x_pos_plot, pos_x_bub+h_box)
        y_pos_plot = np.append(pos_y_bulk+h_box, pos_y_int+h_box)
        y_pos_final = np.append(y_pos_plot, pos_y_bub+h_box)
        lat_mean_plot = np.append(lat_mean_indiv_arr_bulk, lat_mean_indiv_arr_int)
        lat_mean_final = np.append(lat_mean_plot, lat_mean_indiv_arr_bub)
        
        vmin_num = np.min(lat_mean_indiv_arr_bulk)
        if np.max(lat_mean_indiv_arr_bulk)>r_cut:
            vmax_num = r_cut
        else:
            vmax_num = np.max(lat_mean_indiv_arr_bulk)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(x_pos_final, y_pos_final, s=0.7, c=lat_mean_final, vmin=vmin_num, vmax=vmax_num, cmap='viridis')

        norm= matplotlib.colors.Normalize(vmin=vmin_num, vmax=vmax_num)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(vmin_num, vmax_num+vmax_num/10, (vmax_num-vmin_num)/10)
        clb = fig.colorbar(sm, ticks=tick_lev)#ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ax=ax2)
        
        clb.ax.set_title(r'$a$', fontsize=15)

        
        plt.xlim(0, l_box)
        plt.ylim(0, l_box)
        
        plt.text(0.77, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=200)
        plt.close()


        #Plot histogram of the average lattice spacing of bulk (green) and interface (yellow) particle and their six nearest neighbors
        
        #continue if any lattice spacings measured
    if (len(lat_mean_indiv_arr_int)>0) or (len(lat_mean_indiv_arr_bulk)>0):
    
        xmin = 0.5
        xmax = 1.0
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        #Remove bulk particles that are outside plot's xrange
        if (len(lat_mean_indiv_arr_bulk)>0):
            bulk_id = np.where((lat_mean_indiv_arr_bulk > xmax) | (lat_mean_indiv_arr_bulk < xmin))[0]
            lat_mean_indiv_arr_bulk = np.delete(lat_mean_indiv_arr_bulk, bulk_id)
        
            plt.hist(lat_mean_indiv_arr_bulk, alpha = 1.0, bins=50, color=green)
        
        #If interface particle measured, continue
        if (len(lat_mean_indiv_arr_int)>0):
            int_id = np.where((lat_mean_indiv_arr_int > xmax) | (lat_mean_indiv_arr_int < xmin))[0]
            lat_mean_indiv_arr_int = np.delete(lat_mean_indiv_arr_int, int_id)

            plt.hist(lat_mean_indiv_arr_int, alpha = 0.8, bins=50, color=yellow)
        
        if (len(lat_mean_indiv_arr_bub)>0):
            bub_id = np.where((lat_mean_indiv_arr_bub > xmax) | (lat_mean_indiv_arr_bub < xmin))[0]
            lat_mean_indiv_arr_int = np.delete(lat_mean_indiv_arr_bub, bub_id)

            plt.hist(lat_mean_indiv_arr_bub, alpha = 0.4, bins=50, color=purple)
        
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=12, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'lattice spacing ($a$)', fontsize=20)
        plt.ylabel('Number of particles', fontsize=20)
        plt.xlim([xmin,xmax])
        
        plt.text(0.77, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.tight_layout()
        plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        plt.close()
        
    fig = plt.figure(figsize=(8.5,8))
    ax = fig.add_subplot(111)

    myEps = [1., 0.1, 0.01, 0.001, 0.0001]
    plt.scatter(pos[bulk_id_plot,0]+h_box, pos[bulk_id_plot,1]+h_box, s=0.75, marker='.', c=green)
    plt.scatter(pos[gas_id,0]+h_box, pos[gas_id,1]+h_box, s=0.75, marker='.', c=red)
    plt.scatter(pos[edge_id_plot,0]+h_box, pos[edge_id_plot,1]+h_box, s=0.75, marker='.', c=yellow)
    
    if len(bub_id_plot)>0:
        plt.scatter(pos[bub_id_plot,0]+h_box, pos[bub_id_plot,1]+h_box, s=0.75, marker='.', c=purple)
    '''
    if len(bub1_parts)>0:
        plt.scatter(pos[bub1_parts,0]+h_box, pos[bub1_parts,1]+h_box, s=0.75, marker='.', c=purple)
    if len(bub2_parts)>0:
        plt.scatter(pos[bub2_parts,0]+h_box, pos[bub2_parts,1]+h_box, s=0.75, marker='.', c=purple)
    if len(bub3_parts)>0:
        plt.scatter(pos[bub3_parts,0]+h_box, pos[bub3_parts,1]+h_box, s=0.75, marker='.', c=purple)
    if len(bub4_parts)>0:
        plt.scatter(pos[bub4_parts,0]+h_box, pos[bub4_parts,1]+h_box, s=0.75, marker='.', c=purple)
    if len(bub5_parts)>0:
        plt.scatter(pos[bub5_parts,0]+h_box, pos[bub5_parts,1]+h_box, s=0.75, marker='.', c=purple)
    '''

    plt.quiver(pos_box_x_plot, pos_box_y_plot, p_plot_x, p_plot_y)
    plt.xticks(pos_box_start)
    plt.yticks(pos_box_start)
    plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False,
                            labelleft=False)
    plt.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            right=False,      # ticks along the bottom edge are off
                            left=False,         # ticks along the top edge are off
                            labelbottom=False,
                            labelleft=False)

    plt.ylim((0, l_box))
    plt.xlim((0, l_box))
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
           
    plt.text(0.77, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

    eps_leg=[]
    mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
    msz=40
    red_patch = mpatches.Patch(color=red, label='Dilute')
    green_patch = mpatches.Patch(color=green, label='Bulk')
    yellow_patch = mpatches.Patch(color=yellow, label='Interface')
    purple_patch = mpatches.Patch(color=purple, label='Bubble')
    plt.legend(handles=[green_patch, yellow_patch, red_patch, purple_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=16, loc='upper left',labelspacing=0.1, handletextpad=0.1)
    plt.tight_layout()
    plt.savefig(outPath + 'interface_acc_' + out + pad + ".png", dpi=100)
    plt.close()
    
    return_arr = [tst, sizeBin, np.amax(clust_size), lat_mean_bulk, lat_mean_int, lat_mean_bub, lat_mean_all, std_dev_bulk, std_dev_int, std_dev_bub, std_dev_all]
    return return_arr
    
def worker(arg, q):
    '''stupidly simulates long running process'''
    s = 'this is a test'
    txt = s
    return_arr = lattice(arg)
    with open(outPath2+outTxt_lat, 'rb') as f:
        size = len(f.read())
    res = return_arr[0], return_arr[1], return_arr[2], return_arr[3], return_arr[4], return_arr[5], return_arr[6], return_arr[7], return_arr[8], return_arr[9], return_arr[10]
    q.put(res)
    return res
"""
def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open(outPath2+outTxt_lat, 'a') as f:
        while 1:
            m = q.get()
            print(m)
            if m == 'kill':
                f.write('killed')
                break
            f.write('{0:.2f}'.format(m[0]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[1]).center(20) + ' ')
            f.write('{0:.0f}'.format(m[2]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[3]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[4]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[5]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[6]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[7]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[8]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[9]).center(20) + ' ')
            f.write('{0:.6f}'.format(m[10]).center(20) + '\n')
            f.flush()
"""
def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open(outPath2+outTxt_lat, 'a') as f:
        for i in range(0, len(q)):
            f.write('{0:.2f}'.format(q[i][0]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][1]).center(20) + ' ')
            f.write('{0:.0f}'.format(q[i][2]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][3]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][4]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][5]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][6]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][7]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][8]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][9]).center(20) + ' ')
            f.write('{0:.6f}'.format(q[i][10]).center(20) + '\n')
            f.flush()
tSteps=np.linspace(0,60,60+1).astype(int)
def main():
    #must use Manager queue here, or will not work
    #manager = mp.Manager()


    #q = manager.Queue()
    #with closing(mp.Pool(processes = 8, maxtasksperchild=1)) as pool:
        #mp.cpu_count()
        #put listener to work first
        pool = mp.Pool(processes = mp.cpu_count(), maxtasksperchild=1)
        
        pool.map_async(lattice, tSteps, callback=listener)
        #watcher.wait()
        pool.close()
        pool.join()
        #watcher = pool.apply_async(listener, (q,))

        #fire off workers
        #jobs = []
        #for i in tSteps:
        #    job = pool.apply_async(worker, (i, q))
        #    jobs.append(job)
        # collect results from the workers through the pool result queue
        #for job in jobs:
        #    job.get()

        #now we are done, kill the listener
        #q.put('kill')

'''
def main():
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    
    
    q = manager.Queue()
    print('cpu')
    print(mp.cpu_count())
    pool = mp.Pool(processes = mp.cpu_count(), maxtasksperchild=1)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for i in tSteps:
        job = pool.apply_async(worker, (i, q))
        jobs.append(job)
    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
'''
if __name__ == "__main__":
    main()

