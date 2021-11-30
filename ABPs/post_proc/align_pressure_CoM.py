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

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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

outTxt_phase_info = 'PhaseComp_' + outfile + '.txt'
outTxt_theta_ext = 'Theta_vs_radii_ext_' + outfile
outTxt_theta_int = 'Theta_vs_radii_int_' + outfile
outTxt_coeff = 'Coeff_' + outfile + '.txt'
outTxt_bub_info = 'BubComp_' + outfile + '.txt'

#.txt file for saving overall phase composition data  

outTxt_align_press = 'Align_press_CoM_' + outfile + '.txt'

g = open(outPath2+outTxt_align_press, 'w+') # write file headings
g.write('tauB'.center(20) + ' ' +\
                        'sizeBin'.center(20) + ' ' +\
                        'clust_size'.center(20) + ' ' +\
                        'press_align' + '\n')
g.close()

outTxt_radial = 'Radial_int_CoM_' + outfile + '.txt'

g = open(outPath2+outTxt_radial, 'w+') # write file headings
g.write('tauB'.center(20) + ' ' +\
                        'sizeBin'.center(20) + ' ' +\
                        'clust_size'.center(20) + ' ' +\
                        'interface_id'.center(20) + ' ' +\
                        'bub_id'.center(20) + ' ' +\
                        'r_min'.center(20) + ' ' +\
                        'r_max'.center(20) + ' ' +\
                        'align'.center(20) + ' ' +\
                        'num_dens'.center(20) + ' ' +\
                        'press'.center(20) + '\n')
g.close()

with hoomd.open(name=inFile, mode='rb') as t:
    
    start = int(0/time_step)#205                                             # first frame to process
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
                        
        fa_all_tot_bub1 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub1 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_tot_bub1_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub1_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        activity_all_bub1=np.array([])
        partid_all_bub1=np.array([])
        
        fa_all_tot_bub2 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub2 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_tot_bub2_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub2_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        activity_all_bub2=np.array([])
        partid_all_bub2=np.array([])
        
        fa_all_tot_bub3 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub3 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_tot_bub3_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub3_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        activity_all_bub3=np.array([])
        partid_all_bub3=np.array([])
        
        fa_all_tot_bub4 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub4 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_tot_bub4_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub4_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        activity_all_bub4=np.array([])
        partid_all_bub4=np.array([])
        
        fa_all_tot_bub5 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub5 = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_tot_bub5_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        r_dist_tot_bub5_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        activity_all_bub5=np.array([])
        partid_all_bub5=np.array([])
        
        fa_all_tot_trad = np.array([])#[[0 for b in range(NBins)] for a in range(NBins)]
        align_all_tot_trad = np.array([])
        r_dist_tot_trad = np.array([])
        fa_all_tot_test = np.array([])
        r_dist_tot_test = np.array([])
        
        binx_all=np.array([])
        biny_all=np.array([])
        #partid_all=np.array([])
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
                            if j>int(start*time_step):
                                vx = (pos_prev[binParts[ix][iy][h],0]-pos[binParts[ix][iy][h],0])
                                
                                #Enforce periodic boundary conditions
                                vx_abs = np.abs(vx)
                                if vx_abs>=h_box:
                                    if vx < -h_box:
                                        vx += l_box
                                    else:
                                        vx -= l_box
                                
                                vx=vx/(time_arr[j]-time_arr[j-1])
                                vy = (pos_prev[binParts[ix][iy][h],1]-pos[binParts[ix][iy][h],1])
                                
                                
                                #Enforce periodic boundary conditions
                                vy_abs = np.abs(vy)
                                if vy_abs>=h_box:
                                    if vy < -h_box:
                                        vy += l_box
                                    else:
                                        vy -= l_box
                                        
                                vy=vy/(time_arr[j]-time_arr[j-1])
                            #Alignment towards CoM
                            r_dot_p = (-difx * px) + (-dify * py)
                            
                            #Summed orientation of particles per bin
                            p_all_x[ix][iy]+=px
                            p_all_y[ix][iy]+=py
                            
                            if j>(start*time_step):
                                v_all_x[ix][iy]+=vx
                                v_all_y[ix][iy]+=vy
                            
                            #Perform measurements for type A particles only
                            if typ[binParts[ix][iy][h]]==0:
                                typ0_temp +=1               #Number of type A particles per bin
                                p_all_xA[ix][iy]+=px        #Summed x-orientation of type B particles
                                p_all_yA[ix][iy]+=py        #Summed y-orientation of type B particles
                                if j>(start*time_step):
                                    v_all_xA[ix][iy]+=vx
                                    v_all_yA[ix][iy]+=vy
                            #Perform measurements for type B particles only
                            elif typ[binParts[ix][iy][h]]==1:
                                typ1_temp +=1               #Number of type B particles per bin
                                p_all_xB[ix][iy]+=px        #Summed x-orientation of type B particles
                                p_all_yB[ix][iy]+=py        #Summed y-orientation of type B particles
                                if j>(start*time_step):
                                    v_all_xB[ix][iy]+=vx
                                    v_all_yB[ix][iy]+=vy
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
                        
                        
        '''    
        #Calculate average alignment in each bin or set to zero if bin empty
        for ix in range(0, NBins):
                for iy in range(0, NBins):
                        
                        #Average x,y orientation of particles per bin
                        align_avg_x[ix][iy]= p_avg_x[ix][iy]#align_tot_x[ix][iy]/align_avg_num[ix][iy]
                        align_avg_y[ix][iy] = p_avg_y[ix][iy]#align_tot_y[ix][iy]/align_avg_num[ix][iy]
                        
                        #Average x,y orientation of type A particles per bin
                        align_avg_xA[ix][iy]= p_avg_xA[ix][iy]#align_tot_xA[ix][iy]/align_avg_num[ix][iy]
                        align_avg_yA[ix][iy] = p_avg_yA[ix][iy]#align_tot_yA[ix][iy]/align_avg_num[ix][iy]
                        
                        #Average x,y orientation of type B particles per bin
                        align_avg_xB[ix][iy]=p_avg_xB[ix][iy]#align_tot_xB[ix][iy]/align_avg_num[ix][iy]
                        align_avg_yB[ix][iy] = p_avg_yB[ix][iy]#align_tot_yB[ix][iy]/align_avg_num[ix][iy]
                        
                        #Average difference in x,y orientation between type A and B particles
                        align_avg_xDif[ix][iy] = p_avg_xB[ix][iy]-p_avg_xA[ix][iy]#(align_tot_xB[ix][iy]-align_tot_xA[ix][iy])/align_avg_num[ix][iy]
                        align_avg_yDif[ix][iy] = p_avg_yB[ix][iy]-p_avg_yA[ix][iy]#(align_tot_yB[ix][iy]-align_tot_yA[ix][iy])/align_avg_num[ix][iy]
        '''
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
        
        #Loop over system bins
        for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    
                    #If bin is not gas, continue...
                    #if phaseBin[ix][iy]!=2:
                        
                        #Initiate count of particle type per bin
                        typ0_temp=0
                        typ1_temp=0
                        
                        #If particles in bin, proceed
                        if len(binParts[ix][iy]) != 0:
                            
                            #Loop over particles per bin
                            for h in range(0, len(binParts[ix][iy])):
                                
                                #If at least one time-frame measured before this step, continue...
                                if j>(start*time_step):
                                    
                                    #x displacement of particle
                                    vx = (pos_prev[binParts[ix][iy][h],0]-pos[binParts[ix][iy][h],0])
                                    
                                    #Enforce periodic boundary conditions
                                    vx_abs = np.abs(vx)
                                    if vx_abs>=h_box:
                                        if vx < -h_box:
                                            vx += l_box
                                        else:
                                            vx -= l_box
                                    
                                    #x velocity of particle
                                    vx=vx/(time_arr[j]-time_arr[j-1])
                                    
                                    #y displacement of particle
                                    vy = (pos_prev[binParts[ix][iy][h],1]-pos[binParts[ix][iy][h],1])
                                    
                                    
                                    #Enforce periodic boundary conditions
                                    vy_abs = np.abs(vy)
                                    if vy_abs>=h_box:
                                        if vy < -h_box:
                                            vy += l_box
                                        else:
                                            vy -= l_box
                                    
                                    #y velocity of particle
                                    vy=vy/(time_arr[j]-time_arr[j-1])
                                
                                if j>(start*time_step):
                                    v_all_x_no_gas[ix][iy]+=vx
                                    v_all_y_no_gas[ix][iy]+=vy
                                
                                #Add velocity and count of type A particles to bin
                                if typ[binParts[ix][iy][h]]==0:
                                    typ0_temp +=1               #Number of type A particles per bin
                                    if j>(start*time_step):
                                        v_all_xA_no_gas[ix][iy]+=vx
                                        v_all_yA_no_gas[ix][iy]+=vy
                                        
                                #Add velocity and count of type B particles to bin
                                elif typ[binParts[ix][iy][h]]==1:
                                    typ1_temp +=1               #Number of type B particles per bin
                                    if j>(start*time_step):
                                        v_all_xB_no_gas[ix][iy]+=vx
                                        v_all_yB_no_gas[ix][iy]+=vy
                            
                            #average x,y velocity per bin for A and B type particles (excluding gas phase)
                            if j>(start * time_step):
                                v_avg_x_no_gas[ix][iy] = v_all_x_no_gas[ix][iy]/len(binParts[ix][iy])
                                v_avg_y_no_gas[ix][iy] = v_all_y_no_gas[ix][iy]/len(binParts[ix][iy])
                                
                            #average x,y velocity per bin for A type particles (excluding gas phase)
                            if typ0_temp>0:
                                if j>(start*time_step):
                                    v_avg_xA_no_gas[ix][iy] = v_all_xA_no_gas[ix][iy]/typ0_temp
                                    v_avg_yA_no_gas[ix][iy] = v_all_yA_no_gas[ix][iy]/typ0_temp
                            else:
                                if j>(start*time_step):
                                    v_avg_xA_no_gas[ix][iy] = 0.0
                                    v_avg_yA_no_gas[ix][iy] = 0.0
                            
                            #average x,y velocity per bin for B type particles (excluding gas phase)
                            if typ1_temp>0:
                                if j>(start*time_step):
                                    v_avg_xB_no_gas[ix][iy] = v_all_xB_no_gas[ix][iy]/typ1_temp
                                    v_avg_yB_no_gas[ix][iy] = v_all_yB_no_gas[ix][iy]/typ1_temp
                            else:
                                if j>(start*time_step):
                                    v_avg_xB_no_gas[ix][iy] = 0.0
                                    v_avg_yB_no_gas[ix][iy] = 0.0
                                    
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
        if j>(start*time_step):
            
            for ix in range(0, len(v_avg_x)):
                for iy in range(0, len(v_avg_y)):
                    v_combined[ix][iy][0]=v_avg_x[ix][iy]
                    v_combined[ix][iy][1]=v_avg_y[ix][iy]
                    
                    align_combined[ix][iy][0]=align_avg_x[ix][iy]
                    align_combined[ix][iy][1]=align_avg_y[ix][iy]
                    
                    pos_box_combined[ix][iy][0]=pos_box_x[ix][iy]
                    pos_box_combined[ix][iy][1]=pos_box_y[ix][iy]
            vx_grad = np.gradient(v_combined, axis=0)
            vy_grad = np.gradient(v_combined, axis=1)

            #v_grad = np.gradient
            #vx_grad = np.gradient(v_avg_x_no_gas, pos_box_x)   #central diff. du_dx
            #vy_grad = np.gradient(v_avg_y_no_gas, pos_box_y)  #central diff. dv_dy
            
            vel_gradx_x = vx_grad[:,:,0]
            vel_gradx_y = vx_grad[:,:,1]
            vel_grady_x = vy_grad[:,:,0]
            vel_grady_y = vy_grad[:,:,1]
                        
            div = vel_gradx_x + vel_grady_y
            curl = -vel_grady_x + vel_gradx_y
            
        #Calculate average velocity per bin
        if j>(start*time_step):
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                    
                        vel_mag[ix][iy] = ((v_avg_x[ix][iy]**2+v_avg_y[ix][iy]**2)**0.5)    #Average velocity per bin of all particles relative to largest preferred velocity (peB)
                        vel_magA[ix][iy] = ((v_avg_xA[ix][iy]**2+v_avg_yA[ix][iy]**2)**0.5) #Average velocity per bin of type A particles relative to preferred velocity (peA)
                        vel_magB[ix][iy] = ((v_avg_xB[ix][iy]**2+v_avg_yB[ix][iy]**2)**0.5) #Average velocity per bin of type B particles relative to preferred velocity (peB)
                        vel_magDif[ix][iy] = (vel_magB[ix][iy]/peB-vel_magA[ix][iy])        #Difference in magnitude of average velocity per bin between type B and A particles
                        
                        vel_mag_no_gas[ix][iy] = ((v_avg_x_no_gas[ix][iy]**2+v_avg_y_no_gas[ix][iy]**2)**0.5)   #Average velocity per bin of all particles relative to largest preferred velocity (peB)
                        vel_magA_no_gas[ix][iy] = ((v_avg_xA_no_gas[ix][iy]**2+v_avg_yA_no_gas[ix][iy]**2)**0.5) #Average velocity per bin of type A particles relative to preferred velocity (peA)
                        vel_magB_no_gas[ix][iy] = ((v_avg_xB_no_gas[ix][iy]**2+v_avg_yB_no_gas[ix][iy]**2)**0.5) #Average velocity per bin of type B particles relative to preferred velocity (peB)
                        vel_magDif_no_gas[ix][iy] = (vel_magB_no_gas[ix][iy]-vel_magA_no_gas[ix][iy])        #Difference in magnitude of average velocity per bin between type B and A particles
                        
            

        #Counts number of different particles belonging to each phase
        for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if j>(start*time_step):
                        if vel_mag[ix][iy]>0:
                            vel_normx[ix][iy] = v_avg_x[ix][iy] / vel_mag[ix][iy]
                            vel_normy[ix][iy] = v_avg_y[ix][iy] / vel_mag[ix][iy]
                        else:
                            vel_normx[ix][iy]=0
                            vel_normy[ix][iy]=0
                        if vel_magA[ix][iy]>0:
                            vel_normxA[ix][iy] = v_avg_xA[ix][iy] / vel_magA[ix][iy]
                            vel_normyA[ix][iy] = v_avg_yA[ix][iy] / vel_magA[ix][iy]
                        else:
                            vel_normxA[ix][iy] = 0
                            vel_normyA[ix][iy] = 0
                        
                        if vel_magB[ix][iy]>0:
                            vel_normxB[ix][iy] = v_avg_xB[ix][iy] / vel_magB[ix][iy]
                            vel_normyB[ix][iy] = v_avg_yB[ix][iy] / vel_magB[ix][iy]
                        else:
                            vel_normxB[ix][iy] = 0
                            vel_normyB[ix][iy] = 0
                            
                        if vel_mag_no_gas[ix][iy]>0:
                            vel_normx_no_gas[ix][iy] = v_avg_x_no_gas[ix][iy] / vel_mag_no_gas[ix][iy]
                            vel_normy_no_gas[ix][iy] = v_avg_y_no_gas[ix][iy] / vel_mag_no_gas[ix][iy]
                        else:
                            vel_normx_no_gas[ix][iy]=0
                            vel_normy_no_gas[ix][iy]=0
                        if vel_magA_no_gas[ix][iy]>0:
                            vel_normxA_no_gas[ix][iy] = v_avg_xA_no_gas[ix][iy] / vel_magA_no_gas[ix][iy]
                            vel_normyA_no_gas[ix][iy] = v_avg_yA_no_gas[ix][iy] / vel_magA_no_gas[ix][iy]
                        else:
                            vel_normxA_no_gas[ix][iy] = 0
                            vel_normyA_no_gas[ix][iy] = 0
                        
                        if vel_magB_no_gas[ix][iy]>0:
                            vel_normxB_no_gas[ix][iy] = v_avg_xB_no_gas[ix][iy] / vel_magB_no_gas[ix][iy]
                            vel_normyB_no_gas[ix][iy] = v_avg_yB_no_gas[ix][iy] / vel_magB_no_gas[ix][iy]
                        else:
                            vel_normxB_no_gas[ix][iy] = 0
                            vel_normyB_no_gas[ix][iy] = 0
        
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
        
        bulk_id_plot = np.where(partPhase==0)[0]        #Bulk phase structure(s)
        edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
        int_id_plot = np.where(partPhase==1)[0]         #All interfaces
        bulk_int_id_plot = np.where(partPhase!=2)[0]

        if len(bulk_ids)>0:
            bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
        else:
            bub_id_plot = []
        gas_id = np.where(partPhase==2)[0]              #Gas phase structure(s)
        
        #label previous positions for velocity calculation
        pos_prev = pos.copy()
        
        new_align_trad = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_trad_int = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad_int = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_avg_trad = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad2 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_avg_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_avg_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]
        
        
        #Bin system to calculate orientation and alignment that will be used in vector plots
        NBins = getNBins(l_box, bin_width)
        sizeBin = roundUp(((l_box) / NBins), 6)

        #Initiate empty arrays
        new_align_num_trad = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad0 = [[0 for b in range(NBins)] for a in range(NBins)]
        new_align_num_trad1 = [[0 for b in range(NBins)] for a in range(NBins)]

                    
        #Loop over all bins
        for ix in range(0, len(occParts)):
            for iy in range(0, len(occParts)): 
                
                #If average alignment not defined yet (bulk bin)
                if new_align_avg_trad[ix][iy]==0:
                    
                    #Calculate position of exterior edge bin
                    pos_box_x1 = (ix+0.5)*sizeBin
                    pos_box_y1 = (iy+0.5)*sizeBin
                                            
                    #Calculate difference in bin's position with CoM at h_box
                    difx_trad = pos_box_x1 - h_box
                    
                    #Enforce periodic boundary conditions
                    difx_trad_abs = np.abs(difx_trad)
                    if difx_trad_abs>=h_box:
                        if difx_trad < -h_box:
                            difx_trad += l_box
                        else:
                            difx_trad -= l_box
                    
                    #Enforce periodic boundary conditions
                    dify_trad = pos_box_y1 - h_box
                    dify_trad_abs = np.abs(dify_trad)
                    if dify_trad_abs>=h_box:
                        if dify_trad < -h_box:
                            dify_trad += l_box
                        else:
                            dify_trad -= l_box
                            
                    #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                    difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                    
                    x_norm_unitv = difx_trad / difr_trad
                    y_norm_unitv = dify_trad / difr_trad
                                                
                    x_norm_unitv_trad = (difx_trad) / difr_trad
                    y_norm_unitv_trad = (dify_trad) / difr_trad
                    
                    #If particles in bin, continue...
                    if len(binParts[ix][iy])>0:
                        
                        #Loop over all particles in bin
                        for h in range(0, len(binParts[ix][iy])):
                            
                            #Calculate x and y orientation of active force
                            px = np.sin(ang[binParts[ix][iy][h]])
                            py = -np.cos(ang[binParts[ix][iy][h]])

                            #Calculate alignment towards CoM
                            r_dot_p_trad = (-x_norm_unitv_trad * px) + (-y_norm_unitv_trad * py)
                            
                            #Add alignment with CoM for average calculation of all particles
                            new_align_trad[ix][iy] += r_dot_p_trad
                            new_align_num_trad[ix][iy]+= 1
                            align_all_tot_trad=np.append(align_all_tot_trad, r_dot_p_trad)

                            #If particle is of type A, add alignment with nearest surface's normal for average calculation
                            if typ[binParts[ix][iy][h]]==0:
                                fa_all_tot_trad=np.append(fa_all_tot_trad, r_dot_p_trad*peA)
                                r_dist_tot_trad = np.append(r_dist_tot_trad, difr_trad)
                                new_align_trad0[ix][iy] += r_dot_p_trad
                                new_align_num_trad0[ix][iy]+= 1
                                
                            #If particle is of type B, add alignment with nearest surface's normal for average calculation
                            elif typ[binParts[ix][iy][h]]==1:
                                fa_all_tot_trad=np.append(fa_all_tot_trad, r_dot_p_trad*peB)
                                r_dist_tot_trad = np.append(r_dist_tot_trad, difr_trad)
                                new_align_trad1[ix][iy] += r_dot_p_trad
                                new_align_num_trad1[ix][iy]+= 1
        
        #Loop over bins in system
        for ix in range(0, len(occParts)):
            for iy in range(0, len(occParts)): 
                
                #If the average alignment hasn't been calculated before (gas phase)
                if new_align_avg_trad[ix][iy]==0:
                
                    #Calculate average alignment with CoM per bin
                    if new_align_num_trad[ix][iy]>0:
                            new_align_avg_trad[ix][iy] = new_align_trad[ix][iy] / new_align_num_trad[ix][iy]
                            if new_align_num_trad0[ix][iy]>0:
                                new_align_avg_trad0[ix][iy] = new_align_trad0[ix][iy] / new_align_num_trad0[ix][iy]
                            if new_align_num_trad1[ix][iy]>0:
                                new_align_avg_trad1[ix][iy] = new_align_trad1[ix][iy] / new_align_num_trad1[ix][iy]
        
        #Area per bin
        binArea = sizeBin**2
        
        #Particle IDs for each phase
        int_id_plot = np.where(partPhase==1)[0]         #All interfaces
        bulk_int_id_plot = np.where(partPhase!=2)[0]
        gas_id = np.where(partPhase==2)[0]  

               
        #Initiate empty values of summed number density
        bulk_num_dens_sum = 0
        bulk_num_dens_num = 0
        int_num_dens_sum = 0
        int_num_dens_num = 0
        gas_num_dens_sum = 0
        gas_num_dens_num = 0
        
        #Sum total number density per bin to average
        for ix in range(0, len(phaseBin)):
            for iy in range(0, len(phaseBin)):
                if phaseBin[ix][iy]==0:
                    bulk_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    bulk_num_dens_num +=1  
                elif phaseBin[ix][iy]==1:
                    int_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    int_num_dens_num +=1
                elif phaseBin[ix][iy]==2:
                    gas_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    gas_num_dens_num +=1
        
        
        #Average number density in each phase
        if bulk_num_dens_num > 0:
            bulk_avg = (bulk_num_dens_sum/bulk_num_dens_num) 
        else:
            bulk_avg = 0
        if int_num_dens_num > 0:
            int_avg = (int_num_dens_sum/int_num_dens_num)  
        else:
            int_avg = 0
        if gas_num_dens_num > 0:
            gas_avg = (gas_num_dens_sum/gas_num_dens_num) 
        else:
            gas_avg = 0
        
        
        #Area of each phase
        bulk_area = bulk_num_dens_num * sizeBin**2
        int_area = int_num_dens_num * sizeBin**2
        gas_area = gas_num_dens_num * sizeBin**2  
        
        #Initiate empty values for integral of pressures across interfaces
        int_sum_trad = 0
              

        #X locations across interface for integration
        xint = np.linspace(0, np.ceil(np.max(r_dist_tot_trad)), num=int((int((int(np.ceil(np.ceil(np.max(r_dist_tot_trad))))+1)))/3))
        
        #Pressure integrand components for each value of X
        yfinal = np.zeros(len(xint))
        alignfinal = np.zeros(len(xint))
        densfinal = np.zeros(len(xint))
        
        #If exterior and interior surfaces defined, continue...
                        
        area_prev = 0
        #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
        for i in range(1, len(xint)):
            binx_nodup=np.array([])
            biny_nodup=np.array([])
            
            #Min and max location across interface of current step
            min_range = xint[i-1]
            max_range = xint[i]   
            
            #Calculate area of rectangle for current step
            area = np.pi * (max_range ** 2) - area_prev
            
            #Save total area of previous step sizes
            area_prev = np.pi * (max_range ** 2)
             
            #Find particles that are housed within current slice
            points = np.where((min_range<=r_dist_tot_trad) & (r_dist_tot_trad<=max_range))[0]  
            
            #If at least 1 particle in slice, continue...
            if len(points)>0:
                
                #If the force is defined, continue...
                points2 = np.logical_not(np.isnan(fa_all_tot_trad[points]))
                if len(points2)>0:
                    
                    #Calculate total active force normal to interface in slice
                    yfinal[i-1] = np.sum(fa_all_tot_trad[points][points2])
                else:
                    yfinal[i-1]=0
                    
                alignfinal[i-1] = np.mean(align_all_tot_trad[points])
                densfinal[i-1] = len(points)
                    
                    #If area of slice is non-zero, calculate the pressure [F/A]
                if area != 0:
                    yfinal[i-1] = yfinal[i-1]/area
                    densfinal[i-1] = densfinal[i-1]/area
                else:
                    yfinal[i-1] = 0
                    densfinal[i-1] = 0
                
                    
            else:
                yfinal[i-1]=0
                densfinal[i-1]=0
                alignfinal[i-1]=0
        
        #Integrate force across interface using trapezoidal rule
        for o in range(1, len(xint)):
                int_sum_trad += ((xint[o]-xint[o-1])/2)*(yfinal[o]+yfinal[o-1])
        g = open(outPath2+outTxt_radial, 'a')
        for p in range(0, len(yfinal)-1):
            g.write('{0:.2f}'.format(tst).center(20) + ' ')
            g.write('{0:.6f}'.format(sizeBin).center(20) + ' ')
            g.write('{0:.0f}'.format(clust_size[m]).center(20) + ' ')
            g.write('{0:.0f}'.format(interface_id).center(20) + ' ')
            g.write('{0:.0f}'.format(bub_size_id_arr[m]).center(20) + ' ')
            g.write('{0:.6f}'.format(xint[p]).center(20) + ' ')
            g.write('{0:.6f}'.format(xint[p+1]).center(20) + ' ')
            g.write('{0:.6f}'.format(alignfinal[p]).center(20) + ' ')
            g.write('{0:.6f}'.format(densfinal[p]).center(20) + ' ')
            g.write('{0:.6f}'.format(yfinal[p]).center(20) + '\n')
        g.close()
                #Save active pressure of each interface
        g = open(outPath2+outTxt_align_press, 'a')
        g.write('{0:.2f}'.format(tst).center(20) + ' ')
        g.write('{0:.6f}'.format(sizeBin).center(20) + ' ')
        g.write('{0:.0f}'.format(np.amax(clust_size)).center(20) + ' ')
        g.write('{0:.6f}'.format(int_sum_trad).center(20) + '\n')
        g.close()
        