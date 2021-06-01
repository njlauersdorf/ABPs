#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:11:55 2021

@author: nicklauersdorf

Purpose: Perform radial measurement (lattice spacing, virial pressure, number density,
and alignment) over various conical surfaces of dense phase.

Input file:
    HOOMD .GSD file
        
        
Output file(s):
    CoM_ang:
        
        tauB: time step in Brownian time units
        clust_size: largest cluster size
        min_ang: smallest angle of conical surface as defined from x-axis WRT CoM
        max_ang: largest angle of conical surface as defined from x-axis WRT CoM
        radius: radial bin in maximum distance from CoM
        press_vp: virial pressure of bin corresponding to conical surface's (min_ang, max_ang) radial bin (radius)
        lat: average lattice spacing of bin corresponding to conical surface's (min_ang, max_ang) radial bin (radius)
        num_dens: number density of bin corresponding to conical surface's (min_ang, max_ang) radial bin (radius)
        align: alignment of bin toward CoM corresponding to conical surface's (min_ang, max_ang) radial bin (radius)  
"""

                
# Import modules
import sys
from gsd import hoomd
import freud
import numpy as np
import math

# Run on Cluster
hoomdPath='/nas/home/njlauers/hoomd-blue/build/'
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/rad_vp_ang3/'

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'


sys.path.insert(0,hoomdPath)

# Get infile (.gsd) and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''

outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')

#Label simulation parameters
peA = float(sys.argv[2])                        #Activity (Pe) for species A
peB = float(sys.argv[3])                        #Activity (Pe) for species B 
parFrac_orig = float(sys.argv[4])               #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:                        
    parFrac=parFrac_orig*100.
else:           
    parFrac=parFrac_orig
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

eps = float(sys.argv[5])                        #Softness, coefficient of interparticle repulsion (epsilon)

#System area fraction
try:
    phi = float(sys.argv[6])
    intPhi = int(phi)
    phi /= 100.
except:
    phi = 0.6
    intPhi = 60

#Simulation time step
try:
    dtau = float(sys.argv[7])
except:
    dtau = 0.000001

# Set some constants
r_cut=2**(1/6)                  # Cut-off radius for LJ potential
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

#Functions for computing analytical/theoretical values from inputs
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

#Calculate activity-softness dependent analytical values
lat=getLat(peNet,eps)                    # average lattice spacing of dense phase
tauLJ=computeTauLJ(eps)                  # simulation time in LJ time-units
dt = dtau * tauLJ                        # timestep size

#Import modules
import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster

import math
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib import collections  as mc
from matplotlib import lines
    
#Set plotting parameters
matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5
       
     
def computeDist(x1, y1, x2, y2):
    '''Compute distance between point 1 (x1, y1) and point 2 (x2, y2)'''
    return np.sqrt( ((x2-x1)**2) + ((y2 - y1)**2) )
    
def computeFLJ(r, dx, dy, eps):
    '''
    Purpose: Compute directional LJ-force in both x and y directions between 2 particles
    
    Inputs: 
        r: Separation distance in simulation units
        dx: separation distance in x-direction
        dy: separation distance in y-direction
        epsilon: magnitude of lennard-jones potential
    
    Output: lennard jones force magnitude in x-direction (fx) and y-direction (fy)
    '''
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (dx) / r
    fy = f * (dy) / r
    return fx, fy

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
            
def distComps(point1, point2x, point2y):
    '''
    Purpose: Compute distance between two points
    
    Inputs: 
        point1: vector (x,y) of point 1 location
        point2x: position of point 2 along x-axis
        point2y: position of point 2 along y-axis
    Outputs: 
        dx: separation distance between points along x-axis
        dy: separation distance between points along y-axis
        r: magnitude of separation distance between points
    '''
    
    dx = point2x - point1[0]
    dy = point2y - point1[1]
    
    r = np.sqrt((dx**2) + (dy**2))
    
    return dx, dy, r

f = hoomd.open(name=inFile, mode='rb')


# Set particle colors based on slow/fast activities for plot
slowCol = '#d8b365'
fastCol = '#5ab4ac'


box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

                        
outTxt = 'CoM_ang_' + outF + '.txt'
                
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
                        'clust_size'.center(15) + ' ' +\
                        'min_ang'.center(15) + ' ' +\
                        'max_ang'.center(15) + ' ' +\
                        'radius'.center(15) + ' ' +\
                        'rad_bin'.center(15) + ' ' +\
                        'press_vp'.center(15) + ' ' +\
                        'lat'.center(15) + ' ' +\
                        'num_dens'.center(15) + ' ' +\
                        'align'.center(15) + '\n')
g.close()
                
# Access file frames
with hoomd.open(name=inFile, mode='rb') as t:

    
    start = 0                                               # first frame to process
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = dumps                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step
    
    # Get box dimensions
    box_data = snap.configuration.box                       
    l_box = box_data[0]                                     #box length
    h_box = l_box / 2.0                                     #half box length
    a_box = (l_box * l_box)                                 #box area
    
    #Radial bins per conical surface for performing measurement
    radius=np.arange(0,h_box+3.0, 3.0)
    
    #2D binning of system
    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / NBins), 6)
    
    
    partNum = len(snap.particles.typeid)                    #Number of particles
    pos = snap.particles.position                           #(x,y,z) positions of particles
    
    # Initialize freud to identify clusters in each time step based on neighbor density
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    my_clust = cluster.Cluster()
    c_props = cluster.ClusterProperties()
    density = freud.density.LocalDensity(r_max=10., diameter=1.0)
    
    # Loop through time frames
    for j in range(start, end):
    
        print('j')
        print(j)
        
        # Get the current snapshot
        snap = t[j]
        
        # Easier accessors
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0                             # 2D system
        xy = np.delete(pos, 2, 1)                   # Remove Z coordinates
        
        typ = snap.particles.typeid                 # type
        
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        
        ori = snap.particles.orientation            # orientation
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        
        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes
        
        
        min_size=int(partNum/5)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size

        # Only look at clusters if the size is at least 20% of the system
        if len(large_clust_ind_all[0])>0:
            
            #Identify center of mass of largest cluster
            query_points=clp_all.centers[lcID]
            
            #Shift center of mass to binned position
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box
            
            #Identify bin indices of CoM
            com_x_ind = int(com_tmp_posX / sizeBin)
            com_y_ind = int(com_tmp_posY / sizeBin)
            
            #Initialize arrays
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]   #Houses indices identifying particles in each bin
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]   #Houses type of each particle in bin
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]    #Labels whether bin is part of largest cluster
            
            
            
            #Assigns particle indices and types to bins
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
        
            #Define conical surfaces where measurements will be performed
            num_sights_len=20
            num_sights=np.arange(0, 360+int(360/num_sights_len),int(360/num_sights_len))
            
            
            area_slice=np.zeros(len(radius)-1)
            
            #Calculate area of each radial bin of conical surfaces
            for f in range(0,len(radius)-1):
                area_slice[f]=((num_sights[1]-num_sights[0])/360)*math.pi*(radius[f+1]**2-radius[f]**2)
            
            #Loop over conical surfaces
            for k in range(1,len(num_sights)):
                pos_new_x=np.array([])
                pos_new_y=np.array([])
                losBin = [[0 for b in range(NBins)] for a in range(NBins)]
                
                #Loop over spatial (x,y) bins in system to see if in conical surface
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        #x and y ranges of bin's edges
                        x_min_ref=ix*sizeBin
                        x_max_ref=(ix+1)*sizeBin
                        y_min_ref=iy*sizeBin
                        y_max_ref=(iy+1)*sizeBin
                        
                        #Calculate minimum x-distance from CoM
                        dif_x_min = (x_min_ref-com_tmp_posX)
                        difx_min_abs = np.abs(dif_x_min)
                        
                        #Enforce periodic boundary conditions
                        if difx_min_abs>=h_box:
                            if dif_x_min < -h_box:
                                dif_x_min += l_box
                            else:
                                dif_x_min -= l_box
                                
                        #Calculate maximum x-distance from CoM
                        dif_x_max = (x_max_ref-com_tmp_posX)
                        difx_max_abs = np.abs(dif_x_max)
                        
                        #Enforce periodic boundary conditions
                        if difx_max_abs>=h_box:
                            if dif_x_max < -h_box:
                                dif_x_max += l_box
                            else:
                                dif_x_max -= l_box
                                
                        #Calculate minimum y-distance from CoM
                        dif_y_min = (y_min_ref-com_tmp_posY)
                        dify_min_abs = np.abs(dif_y_min)
                        
                        #Enforce periodic boundary conditions
                        if dify_min_abs>=h_box:
                            if dif_y_min < -h_box:
                                dif_y_min += l_box
                            else:
                                dif_y_min -= l_box
                                
                        #Calculate maximum y-distance from CoM
                        dif_y_max = (y_max_ref-com_tmp_posY)
                        dify_max_abs = np.abs(dif_y_max)
                        
                        #Enforce periodic boundary conditions
                        if dify_max_abs>=h_box:
                            if dif_y_max < -h_box:
                                dif_y_max += l_box
                            else:
                                dif_y_max -= l_box
                        
                        #If reference bin is not CoM bin, determine quadrant of system with respect to CoM
                        if ((ix!=com_x_ind) or (iy!=com_y_ind)):
                            if ((dif_x_min>=0) and (dif_x_max>=0)):
                                if ((dif_y_min>=0) and (dif_y_max>=0)):
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=1
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=1
                                    max_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    min_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(min_quad-1)*90
                                elif ((dif_y_min<=0) and (dif_y_max<=0)):
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=4
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=4
                                    min_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                                elif (((dif_y_min<=0) and (dif_y_max>=0)) or ((dif_y_min>=0) and (dif_y_max<=0))):
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=4
                                    max_ref=np.array([x_min_ref, y_max_ref])
                                    max_quad=1

                                    max_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(max_quad-1)*90
                            elif ((dif_x_min<0) and (dif_x_max<0)):
                                if ((dif_y_min>0) and (dif_y_max>0)):
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=2
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=2
                                    max_angle=((np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*(180/np.pi))+(min_quad-1)*90
                                    min_angle=((np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*(180/np.pi))+(max_quad-1)*90
                                elif ((dif_y_min<0) and (dif_y_max<0)):
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=3
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(max_quad-1)*90
                                elif (((dif_y_min<0) and (dif_y_max>0)) or ((dif_y_min>0) and (dif_y_max<0))):
                                    min_ref=np.array([x_max_ref, y_min_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=2
                                    max_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(min_quad-1)*90
                                    min_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                            elif (((dif_x_min<0) and (dif_x_max>0)) or ((dif_x_min>0) and (dif_x_max<0))):
                                if ((dif_y_min>0) and (dif_y_max>0)):
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=2
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=1
                                    max_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    min_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(max_quad-1)*90
                                elif ((dif_y_min<0) and (dif_y_max<0)):
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=4
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                                elif (((dif_y_min<0) and (dif_y_max>0)) or ((dif_y_min>0) and (dif_y_max<0))):

                                    max_angle=45.1
                                    min_angle=44.9

                            # Checks to see if reference bin in conical surface based on angle from CoM. If it is, set losBin=1
                            if min_angle<=90 and max_angle>=270:
                                if 0<=num_sights[k-1]<=min_angle:
                                    losBin[ix][iy]=1
                                elif 0<=num_sights[k]<=min_angle:
                                    losBin[ix][iy]=1
                                elif num_sights[k-1]<=min_angle<=num_sights[k-1]:
                                    losBin[ix][iy]=1
                                elif max_angle<=num_sights[k-1]<=360:
                                    losBin[ix][iy]=1
                                elif max_angle<=num_sights[k]<=360:
                                    losBin[ix][iy]=1
                                elif num_sights[k-1]<=max_angle<=num_sights[k-1]:
                                    losBin[ix][iy]=1
                            elif min_angle<=num_sights[k-1]<=max_angle:
                                losBin[ix][iy]=1
                            elif min_angle<=num_sights[k]<=max_angle:
                                losBin[ix][iy]=1
                            elif num_sights[k-1]<=min_angle<=num_sights[k]:
                                losBin[ix][iy]=1
                            elif num_sights[k-1]<=max_angle<=num_sights[k]:
                                losBin[ix][iy]=1
                                    
                        #If reference bin is CoM bin, always include it in measurement for conical surface    
                        elif ((ix==com_x_ind) and (iy==com_y_ind)):
                            losBin[ix][iy]=1
                            
                #Initialize output arrays
                rad_bin=np.zeros(len(radius)-1)             #Particles per bin
                align_rad=np.zeros(len(radius)-1)           #Summed alignment per bin
                pressure_vp = np.zeros(len(radius)-1)       #Summed virial pressure per bin
                press_num = np.zeros(len(radius)-1)         #Number of values summed to each array
                lat_space = np.zeros(len(radius)-1)         #Summed lattice spacing per bin
                
                #Loop over system (x,y) bins
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        
                        #If bin is in conical surface, proceed
                        if losBin[ix][iy]==1:
                            
                            #If particles in bin, proceed
                            if len(binParts[ix][iy])!=0:
                                
                                #loop over particles in bin to determine if particle in conical surface
                                for h in range(0,len(binParts[ix][iy])):
                                    x_pos=pos[binParts[ix][iy]][h][0]+h_box
                                        
                                    y_pos=pos[binParts[ix][iy]][h][1]+h_box
                                    
                                    #x-distance from CoM
                                    difx=x_pos-com_tmp_posX
                                    
                                    #Enforce periodic boundary conditions
                                    difx_abs = np.abs(difx)
                                    if difx_abs>=h_box:
                                        if difx < -h_box:
                                            difx += l_box
                                        else:
                                            difx -= l_box
                                            
                                    #y-distance from CoM
                                    dify=y_pos-com_tmp_posY
                                    
                                    #Enforce periodic boundary conditions
                                    dify_abs = np.abs(dify)
                                    if dify_abs>=h_box:
                                        if dify < -h_box:
                                            dify += l_box
                                        else:
                                            dify -= l_box
                                            
                                    #Determine quadrant and angle from CoM with positive x-axis being theta=0
                                    if (difx)>0:
                                        if (dify)>0:
                                            part_quad=1
                                            part_angle=(np.arctan(np.abs(dify)/np.abs(difx)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)<0:
                                            part_quad=4
                                            part_angle=(np.arctan(np.abs(difx)/np.abs(dify)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)==0:
                                            part_angle=0
                                    elif (difx)<0:
                                        if (dify)>0:
                                            part_quad=2
                                            part_angle=(np.arctan(np.abs(difx)/np.abs(dify)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)<0:
                                            part_quad=3
                                            part_angle=(np.arctan(np.abs(dify)/np.abs(difx)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)==0:
                                            part_angle=180
                                    elif (difx)==0:
                                        if (dify)>0:
                                            part_angle=90
                                        elif (dify)<0:
                                            part_angle=270
                                        elif (dify)==0:
                                            part_angle=(num_sights[k]+num_sights[k-1])/2
                                    
                                    #If angle between particle and x-axis WRT CoM is within conical surface, perform measurements
                                    if num_sights[k-1]<=part_angle<=num_sights[k]:

                                        #Save position of particle
                                        pos_new_x=np.append(pos_new_x, x_pos)
                                        pos_new_y=np.append(pos_new_y, y_pos)
                                        
                                        #Distance between CoM and particle
                                        difr=(difx**2+dify**2)**0.5
                                        
                                        #Loop through radial bin of conical surface to see which it belongs to
                                        for l in range(1,len(radius)):
                                            
                                            #If particle belongs to radial bin, proceed
                                            if radius[l-1]<=difr<=radius[l]:
                                                
                                                #Add 1 to count of particles per bin
                                                rad_bin[l-1]+=1
                                                
                                                #Define search range of bins around reference bin
                                                if ix==0:
                                                    ix_new_range = [len(occParts)-1, 0, 1]
                                                elif ix==len(occParts)-1:
                                                    ix_new_range = [len(occParts)-2, len(occParts)-1, 0]
                                                else:
                                                    ix_new_range = [ix-1, ix, ix+1]
                                                
                                                if iy==0:
                                                    iy_new_range = [len(occParts)-1, 0, 1]
                                                elif iy==len(occParts)-1:
                                                    iy_new_range = [len(occParts)-2, len(occParts)-1, 0]
                                                else:
                                                    iy_new_range = [iy-1, iy, iy+1]
                                                
                                                #Loop over surrounding bins
                                                for ix2 in ix_new_range:
                                                    for iy2 in iy_new_range:
                                                        
                                                        #If particles in bin, proceed
                                                        if len(binParts[ix2][iy2])!=0:
                                                            
                                                            #Loop over particles in bin
                                                            for h2 in range(0,len(binParts[ix2][iy2])):
                                                                
                                                                #If not reference bin, proceed
                                                                if binParts[ix2][iy2][h2] != binParts[ix][iy][h]:
                                                                    
                                                                    #x, y positions of second particle
                                                                    x_pos_new=pos[binParts[ix2][iy2]][h2][0]+h_box
                                            
                                                                    y_pos_new=pos[binParts[ix2][iy2]][h2][1]+h_box
                                                                    
                                                                    #Difference in x-position between particle 1 of reference bin and particle 2 of surrounding bin
                                                                    difx2=x_pos-x_pos_new
                                                                    difx_abs2 = np.abs(difx2)
                                                                    
                                                                    #Enforce periodic boundary conditions
                                                                    if difx_abs2>=h_box:
                                                                        if difx2 < -h_box:
                                                                            difx2 += l_box
                                                                        else:
                                                                            difx2 -= l_box
                                                                            
                                                                    #Difference in y-position between particle 1 of reference bin and particle 2 of surrounding bin
                                                                    dify2=y_pos-y_pos_new
                                                                    dify_abs2 = np.abs(dify2)
                                                                    
                                                                    #Enforce periodic boundary conditions
                                                                    if dify_abs2>=h_box:
                                                                        if dify2 < -h_box:
                                                                            dify2 += l_box
                                                                        else:
                                                                            dify2 -= l_box
                                                                    
                                                                    #Magnitude of distance between particle 1 and 2
                                                                    difr2=(difx2**2+dify2**2)**0.5
                                                                    
                                                                    #If separation distance is within LJ potential cut-off radius, proceed with measurement
                                                                    if 0.1<=difr2<=r_cut:
                                                                        
                                                                        #Compute LJ force between particle 1 and 2
                                                                        fx, fy = computeFLJ(difr2, difx2, dify2, eps)
                                                                        
                                                                        # Compute the x-component of stress (the x force times x distance)
                                                                        sigx = fx * (difx2)
                                                                        
                                                                        # Compute the x-component of stress (the x force times x distance)
                                                                        sigy = fy * (dify2)
                                                                        
                                                                        #Add 1 to number of terms summed
                                                                        press_num[l-1] += 1
                                                                        
                                                                        #Add virial pressure for pair of particles to that bin for pressure
                                                                        pressure_vp[l-1] += ((sigx + sigy) / 2.)
                                                                        
                                                                        #Add separation distance of particles to that bin for lattice spacing
                                                                        lat_space[l-1] += difr2
                                                                
                                                                
                                                                
                                                #Calculate x and y orientations of particles              
                                                px = np.sin(ang[binParts[ix][iy][h]])
                                                py = -np.cos(ang[binParts[ix][iy][h]])
                                                
                                                
                                                #Calculate pair force alignment
                                                r_dot_p = (-difx * px) + (-dify * py)
                                                
                                                #Normalize alignment
                                                align=r_dot_p/difr
                                                
                                                #Sum alignment
                                                align_rad[l-1]+=align
                                
                #Initialize time-averaged arrays                                
                num_dens=np.zeros(len(radius)-1)
                align_tot=np.zeros(len(radius)-1)
                lat_space_avg=np.zeros(len(radius)-1)
                pressure_vp_avg=np.zeros(len(radius)-1)
                
                #Calculate alignment per bin averaged per particle
                for f in range(0, len(align_rad)):
                    if rad_bin[f]!=0:
                        align_tot[f]=(align_rad[f]/rad_bin[f])
                        
                #Calculate number density per bin averaged per particle
                for f in range(0, len(radius)-1):
                    num_dens[f]=rad_bin[f]/area_slice[f]
                    
                #Calculate lattice spacing per bin averaged per particle
                for f in range(0, len(lat_space_avg)):
                    if rad_bin[f]!=0:
                        lat_space_avg[f]=(lat_space[f]/rad_bin[f]) 
                        
                #Calculate virial pressure per bin averaged per particle
                for f in range(0, len(pressure_vp)):
                    pressure_vp_avg[f]=(pressure_vp[f]/area_slice[f]) 

                #Output data to .txt file
                g = open(outPath+outTxt, 'a')
                for h in range(0, len(rad_bin)):
                    g.write('{0:.2f}'.format(tst).center(15) + ' ')
                    g.write('{0:.0f}'.format(np.amax(clust_size)).center(15) + ' ')
                    g.write('{0:.1f}'.format(num_sights[k-1]).center(15) + ' ')
                    g.write('{0:.1f}'.format(num_sights[k]).center(15) + ' ')
                    g.write('{0:.6f}'.format(radius[h+1]).center(15) + ' ')
                    g.write('{0:.1f}'.format(rad_bin[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(pressure_vp_avg[h]/2).center(15) + ' ')
                    g.write('{0:.6f}'.format(lat_space_avg[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(num_dens[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(align_tot[h]).center(15) + '\n')
                g.close()
            
