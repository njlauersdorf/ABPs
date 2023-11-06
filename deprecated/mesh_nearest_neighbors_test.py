#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:05:25 2021

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
What does this file do?
1.) Read in .gsd file of particle positions
2.) Mesh the space
3.) Loop through tsteps and ...
3a.) Place all particles in appropriate mesh grid
3b.) Loop through all particles ...
3b.i.) Compute distance to every particle in adjacent grids
3.b.ii.) If distance is equal to particle diameter, increment neighbor count
3c.) Plot particle position colored by nearest neighbors (0-6)
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
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick

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

def computeDist(x1, y1, x2, y2):
    '''Compute distance between two points'''
    return np.sqrt( ((x2-x1)**2) + ((y2 - y1)**2) )

def computeFLJ(r, x1, y1, x2, y2, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (x2 - x1) / r
    fy = f * (y2 - y1) / r
    return fx, fy

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
def getDistance(point1, point2x, point2y):
    """Find the distance between two points"""
    distance = np.sqrt((point2x - point1[0])**2 + (point2y - point1[1])**2)
    return distance
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
from freud import locality
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

with hoomd.open(name=inFile, mode='rb') as t:
    
    #Initial time step data
    start = int(0/time_step)#205   
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = int(dumps/time_step)-1                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step
                                              # first frame to process
    #Get box data from initial time step              
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.0
    a_box = l_box * l_box
    
    # Make the mesh
    r_cut = lat_theory             #Maximum distance to find neighbors and for bin length equal to lattice spacing
    
    # Make size of bin divisible by l_box:
    divBox = round(l_box, 4)    # round l_box
    if divBox - l_box < 0:
        divBox += 0.0001        # make sure rounded > actual
    
    # Adjust sizeBin so that it divides into divBox:
    convert = 100000.0
    intBinSize = int(r_cut * convert)
    intDivBox = int(divBox * convert)
    while (intDivBox % intBinSize) != 0:
        intBinSize += 1
    sizeBin = (intBinSize / convert)    # divisible bin size

    nBins = int(divBox / sizeBin)       # must be an integer
    
    # Enlarge the box to include the periodic images
    buff = float(int(r_cut * 2.0) + 1)
    
    # Image rendering options
    drawBins = False
    myCols = 'viridis'
    for j in range(start, end):
        snap = t[j]                               # snapshot of frame
        typ = snap.particles.typeid          # get types
        pos = snap.particles.position    # get positions
        tst = snap.configuration.step    # get timestep
    
                # Get number of each type of particle
        partNum = len(typ)
        part_A = int(partNum * (parFrac/100))
        part_B = partNum - part_A
        # Mesh array
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        
        imageParts = []                                 # append (x, y) tuples
        # Duplicate replicate positions
        for k in range(0, partNum):
            
            # x-coordinate image creation
            if pos[k][0] + h_box < buff:                # against left box edge
                imageParts.append((pos[k][0] + l_box, pos[k][1]))
                # Create image of corners of periodic box
                if pos[k][1] + h_box < buff: # bottom left corner
                    imageParts.append((pos[k][0] + l_box, pos[k][1] + l_box))
                if (pos[k][1] + h_box - l_box) > -buff: # top left corner
                    imageParts.append((pos[k][0] + l_box, pos[k][1] - l_box))
        
            if (pos[k][0] + h_box - l_box) > -buff:     # against right box edge
                imageParts.append((pos[k][0] - l_box, pos[k][1]))
                # Create image of corners of periodic box
                if pos[k][1] + h_box < buff: # bottom right corner
                    imageParts.append((pos[k][0] - l_box, pos[k][1] + l_box))
                if (pos[k][1] + h_box - l_box) > -buff: # top right corner
                    imageParts.append((pos[k][0] - l_box, pos[k][1] - l_box))
            
            # y-coordinate image creation
            if pos[k][1] + h_box < buff:                # against bottom box edge
                imageParts.append((pos[k][0], pos[k][1] + l_box))
            
            if (pos[k][1] + h_box - l_box) > -buff:     # against top box edge
                imageParts.append((pos[k][0], pos[k][1] - l_box))
    
        # Put particles in their respective bins
        for k in range(0, partNum):
            # Get mesh indices
            tmp_posX = pos[k][0] + h_box
            tmp_posY = pos[k][1] + h_box
            x_ind = int(tmp_posX / sizeBin)
            y_ind = int(tmp_posY / sizeBin)
            # Append particle id to appropriate bin
            binParts[x_ind][y_ind].append(k)
        
        # Make an array that will hold the number of nearest neighbors
        near_neigh = [0] * partNum
        A_neigh = [0] * partNum
        B_neigh = [0] * partNum
    
        # Compute distance, each pair will be counted twice
        for k in range(0, partNum):
            # Get mesh indices
            tmp_posX = pos[k][0] + h_box
            tmp_posY = pos[k][1] + h_box
            x_ind = int(tmp_posX / sizeBin)
            y_ind = int(tmp_posY / sizeBin)
            # Get index of surrounding bins
            l_bin = x_ind - 1  # index of left bins
            r_bin = x_ind + 1  # index of right bins
            b_bin = y_ind - 1  # index of bottom bins
            t_bin = y_ind + 1  # index of top bins
            if r_bin == nBins:
                r_bin -= nBins  # adjust if wrapped
            if t_bin == nBins:
                t_bin -= nBins  # adjust if wrapped
            h_list = [l_bin, x_ind, r_bin]  # list of horizontal bin indices
            v_list = [b_bin, y_ind, t_bin]  # list of vertical bin indices
    
            # Loop through all bins
            for h in range(0, len(h_list)):
                for v in range(0, len(v_list)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and h_list[h] == -1:
                        wrapX -= l_box
                    if h == 2 and h_list[h] == 0:
                        wrapX += l_box
                    if v == 0 and v_list[v] == -1:
                        wrapY -= l_box
                    if v == 2 and v_list[v] == 0:
                        wrapY += l_box
                    # Compute distance between particles
                    for b in range(0, len(binParts[h_list[h]][v_list[v]])):
                        ref = binParts[h_list[h]][v_list[v]][b]
                        
                        #Difference in x,y positions
                        difx = pos[k][0]-pos[ref][0]
                        dify = pos[k][1]-pos[ref][1]
                        
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
            
                        r = (difx ** 2 + dify ** 2) ** 0.5
                        #r = getDistance(pos[k],
                        #                pos[ref][0] + wrapX,
                        #                pos[ref][1] + wrapY)
                        #r = round(r, 4)  # round value to 4 decimal places
    
                        # If LJ potential is on, store into a list (omit self)
                        if 0.1 < r <= r_cut:
                            near_neigh[k] += 1
                            if typ[ref] == 0:   # neighbor is A particle
                                A_neigh[k] += 1
                            else:               # neighbor is B particle
                                B_neigh[k] += 1
    
        # Plot position colored by neighbor number
        sz = 1.0
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        
        scatter = plt.scatter(pos[:,0], pos[:,1],
                              c=near_neigh[:], cmap=myCols,
                              s=sz, edgecolors='none')
        # I can plot this in a simpler way: subtract from all particles without looping
        xImages, yImages = zip(*imageParts)
        periodicIm = plt.scatter(xImages, yImages, c='#DCDCDC', s=sz, edgecolors='none')
        # Get colorbar
        #plt.clim(0, 6)  # limit the colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('# of Neighbors', rotation=270, labelpad=35, fontsize=32)
        cbar.ax.tick_params(labelsize=26)
        # Limits and ticks
        viewBuff = buff / 2.0
        plt.xlim(-h_box - viewBuff, h_box + viewBuff)
        plt.ylim(-h_box - viewBuff, h_box + viewBuff)
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                        
        pad = str(j).zfill(4)
        
        if drawBins:
            # Add the bins as vertical and horizontal lines:
            for binInd in range(0, nBins):
                coord = (sizeBin * binInd) - h_box
                plt.axvline(x=coord, c='k', lw=1.0, zorder=0)
                plt.axhline(y=coord, c='k', lw=1.0, zorder=0)
        
        plt.tight_layout()
        plt.savefig(outPath + out + pad + '.png', dpi=150)
        plt.close()
        