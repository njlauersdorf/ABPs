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
out = outfile + "_frame_"

outTxt_lat = 'lat_' + outfile + '.txt'


#.txt file for saving lattice spacing data
g = open(outPath2+outTxt_lat, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
                        'sizeBin'.center(15) + ' ' +\
                        'clust_size'.center(15) + ' ' +\
                        'lat_theory'.center(15) + ' ' +\
                        'bulk_mean'.center(15) + ' ' +\
                        'bulk_std'.center(15) + ' ' +\
                        'int_mean'.center(15) + ' ' +\
                        'int_std'.center(15) + ' ' +\
                        'dense_mean'.center(15) + ' ' +\
                        'dense_std'.center(15) + '\n')
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

with hoomd.open(name=inFile, mode='rb') as t:

    r = np.linspace(0.0,  5.0, 100)             # Define radius for x-axis of plot later

    start = int(720/time_step)#205                                             # first frame to process
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = int(dumps/time_step)-1                                             # final frame to process
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
    com_velocity = np.array([])
    com_tmp_posX_arr = np.array([])
    com_tmp_posY_arr = np.array([])
    for p in range(start, end):
        j=int(p*time_step)
        print('j')
        print(j)

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

            com_tmp_posX_arr = np.append(com_tmp_posX_arr, com_tmp_posX)
            com_tmp_posY_arr = np.append(com_tmp_posY_arr, com_tmp_posY)

            if len(com_tmp_posX_arr)>=2:
                difx = com_tmp_posX_arr[-1] - com_tmp_posX_arr[-2]

                difx_abs = np.abs(difx)
                if difx_abs>=h_box:
                    if difx < -h_box:
                        difx += l_box
                    else:
                        difx -= l_box

                difx = com_tmp_posY_arr[-1] - com_tmp_posY_arr[-2]

                #Enforce periodic boundary conditions
                dify_abs = np.abs(dify)
                if dify_abs>=h_box:
                    if dify < -h_box:
                        dify += l_box
                    else:
                        dify -= l_box

                difr = (difx**2 + dify**2)**0.5
                com_velocity = np.append(com_velocity, difr/(time_arr[j]-time_arr[j-1]))

        else:
            com_tmp_posX_arr = np.append(com_tmp_posX_arr, 0)
            com_tmp_posY_arr = np.append(com_tmp_posY_arr, 0)
            com_velocity = np.append(com_velocity, 0)

        #shift reference frame to center of mass of cluster
        pos[:,0]= pos[:,0]
        pos[:,1]= pos[:,1]


        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(pos_bulk_int_x_lat+h_box, pos_bulk_int_y_lat+h_box, c=bulk_int_lat_arr, s=0.7, vmin=0.97*bulk_lat_mean, vmax=1.03*bulk_lat_mean)


        min_n = 0.97*bulk_lat_mean
        max_n = 1.03*bulk_lat_mean
        if bulk_lat_mean != 0.0:
            tick_lev = np.arange(min_n, max_n+(max_n-min_n)/6, (max_n - min_n)/6)
            #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
            #sm.set_array([])
            clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        else:
            clb = plt.colorbar(orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        plt.text(0.75, 0.92, s=r'$\overline{a}$' + ' = ' + '{:.3f}'.format(dense_lat_mean),
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.quiver(pos_box_x, pos_box_y, velocity_x_A_bin_plot, velocity_y_A_bin_plot, scale=20.0, color='black', alpha=0.8)

        clb.ax.tick_params(labelsize=16)
        clb.set_label('a', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=100)
        plt.close()

    xmin = 0.5
    xmax = 1.0

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    #Remove bulk particles that are outside plot's xrange
    if (len(bulk_lats)>0):
        bulk_id = np.where((bulk_lats > xmax) | (bulk_lats < xmin))[0]
        bulk_lats = np.delete(bulk_lats, bulk_id)

        plt.hist(bulk_lats, alpha = 1.0, bins=50, color=green)

    #If interface particle measured, continue
    if (len(int_lats)>0):
        int_id = np.where((int_lats > xmax) | (int_lats < xmin))[0]
        int_lats = np.delete(int_lats, int_id)

        plt.hist(int_lats, alpha = 0.8, bins=50, color=yellow)

    green_patch = mpatches.Patch(color=green, label='Bulk')
    yellow_patch = mpatches.Patch(color=yellow, label='Interface')
    plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=12, loc='upper right',labelspacing=0.1, handletextpad=0.1)

    plt.xlabel(r'lattice spacing ($a$)', fontsize=20)
    plt.ylabel('Number of particles', fontsize=20)
    plt.xlim([xmin,xmax])

    plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(tst) + ' ' + r'$\tau_\mathrm{B}$',
        fontsize=18,transform = ax.transAxes,
        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

    plt.tight_layout()
    plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
    plt.close()
