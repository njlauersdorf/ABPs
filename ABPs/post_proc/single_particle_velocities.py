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
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches


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


import hoomd
from hoomd import md
from hoomd import deprecated

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster

# Command line arguments
infile = str(sys.argv[1])                               # Get infile (.gsd)
peA = float(sys.argv[2])                                #Activity (Pe) for species A
peB = float(sys.argv[3]) 

parFrac_orig = float(sys.argv[4])                       #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig

if parFrac==100.0:
    parFrac_orig=0.5
    parFrac=50.0
    
eps = float(sys.argv[5])                                #Softness, coefficient of interparticle repulsion (epsilon)

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

#Set plotting parameters
matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5

def computeR(part1, part2):
    """Computes distance"""
    return np.sqrt(((part2[0]-part1[0])**2)+((part2[1]-part1[1])**2))

def computeA(diameter):
    """Computes area of circle"""
    radius = diameter / 2.0
    return np.pi * (radius**2)

def getDistance(point1, point2x, point2y):
    """Find the distance between two points"""
    distance = np.sqrt((point2x - point1[0])**2 + (point2y - point1[1])**2)
    return distance
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
def slowSort(array):
    """Sort an array the slow (but certain) way"""
    cpy = np.copy(array)
    ind = np.arange(0, len(array))
    for i in range(0, len(cpy)):
        for j in range(0, len(cpy)):
            if cpy[i] > cpy[j] and i < j:
                # Swap the copy array values
                tmp = cpy[i]
                cpy[i] = cpy[j]
                cpy[j] = tmp
                # Swap the corresponding indices
                tmp = ind[i]
                ind[i] = ind[j]
                ind[j] = tmp
    return ind

def indSort(arr1, arr2):
    """Take sorted index array, use to sort array"""
    # arr1 is array to sort
    # arr2 is index array
    cpy = np.copy(arr1)
    for i in range(0, len(arr1)):
        arr1[i] = cpy[arr2[i]]

def chkSort(array):
    """Make sure sort actually did its job"""
    for i in range(0, len(array)-2):
        if array[i] > array[i+1]:
            print("{} is not greater than {} for indices=({},{})").format(array[i+1], array[i], i, i+1)
            return False
    return True

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
def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))
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
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

lat_theory = conForRClust(peNet, eps)           #Analytical lattice spacing

#Open input simulation file
f = hoomd.open(name=infile, mode='rb')
                
#Get particle number from initial frame
snap = f[0]
typ = snap.particles.typeid
partNum = len(typ)

# Create outfile name from infile name
#Set output file names
bin_width = float(sys.argv[8])
time_step = float(sys.argv[9])
outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = 'single_v_' + outfile + "_frame_"
out2 = 'single_vA_' + outfile + "_frame_"
out3 = 'single_vB_' + outfile + "_frame_"

f = hoomd.open(name=infile, mode='rb')  # open gsd file with hoomd

start = int(0/time_step)#205                                             # first frame to process
dumps = int(f.__len__())                                # get number of timesteps dumped
end = int(dumps/time_step)-1   

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
            
with hoomd.open(name=infile, mode='rb') as t:
    
    
    #Initial time step data
    snap = t[0]           
    typ = snap.particles.typeid
    partNum = len(typ)
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
    
    #2D binning of system
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    
    # Enlarge the box to include the periodic images
    buff = float(int(r_cut * 2.0) + 1)
    
    # Image rendering options
    drawBins = False
    myCols = 'viridis'
    first_tstep = snap.configuration.step                   # First time step
    time_arr=np.zeros(dumps) 
    vx_arr=np.zeros(partNum, dtype=float)
    vy_arr=np.zeros(partNum, dtype=float)
    vmag_arr=np.zeros(partNum, dtype=float)
    for p in range(start, end):
        j=int(p*time_step)
        print('j')
        print(j)
        snap = t[j]                               # snapshot of frame
        typ = snap.particles.typeid          # get types
        pos = snap.particles.position    # get positions
        tst = snap.configuration.step    # get timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau    
                # Get number of each type of particle
        part_A = int(partNum * (parFrac/100))
        part_B = partNum - part_A
        
        # Duplicate replicate positions
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
        
        #Minimum cluster size to shift reference frame to cluster's CoM
        min_size=int(partNum/8)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
        
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
        
        
        
            
        #Bin system to calculate orientation and alignment that will be used in vector plots
        NBins = getNBins(l_box, 6.0)
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
                        
        binParts = [[[] for b in range(NBins)] for a in range(NBins)]
        typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
        occParts = [[0 for b in range(NBins)] for a in range(NBins)]
        edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
        Binpe = [[0 for b in range(NBins)] for a in range(NBins)]
        
        phaseBin = [[0 for b in range(NBins)] for a in range(NBins)]            #Label phase of each bin
            
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
                                
        #Loop over particles per bin
        if j>(start*time_step):
            for k in range(0, partNum):
            
             
                                
            #If at least one time-frame measured before this step, continue...
            
                
                #x displacement of particle
                vx = (pos[k,0]-pos_prev[k,0])
                
                #Enforce periodic boundary conditions
                vx_abs = np.abs(vx)
                if vx_abs>=h_box:
                    if vx < -h_box:
                        vx += l_box
                    else:
                        vx -= l_box
                
                #x velocity of particle
                vx_arr[k]=vx/(time_arr[j]-time_arr[j-1])
                
                #y displacement of particle
                vy = (pos[k,1]-pos_prev[k,1])
                
                
                #Enforce periodic boundary conditions
                vy_abs = np.abs(vy)
                if vy_abs>=h_box:
                    if vy < -h_box:
                        vy += l_box
                    else:
                        vy -= l_box
                
                #y velocity of particle
                vy_arr[k]=vy/(time_arr[j]-time_arr[j-1])
                vmag_arr[k]=(vy_arr[k]**2 + vx_arr[k]**2)**0.5
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
        v_A_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
        v_B_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
        v_Dif_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
        
        pos_box_combined = np.zeros((len(v_avg_x), len(v_avg_y),2))
        if j>(start*time_step):
            
            for ix in range(0, len(v_avg_x)):
                for iy in range(0, len(v_avg_y)):
                    v_combined[ix][iy][0]=v_avg_x[ix][iy]
                    v_combined[ix][iy][1]=v_avg_y[ix][iy]
                    v_A_combined[ix][iy][0]=v_avg_xA[ix][iy]
                    v_A_combined[ix][iy][1]=v_avg_yA[ix][iy]
                    v_B_combined[ix][iy][0]=v_avg_xB[ix][iy]
                    v_B_combined[ix][iy][1]=v_avg_yB[ix][iy]
                    
                    v_Dif_combined[ix][iy][0]=v_avg_xB[ix][iy]-v_avg_xA[ix][iy]
                    v_Dif_combined[ix][iy][1]=v_avg_yB[ix][iy]-v_avg_yA[ix][iy]
                    
                    pos_box_combined[ix][iy][0]=pos_box_x[ix][iy]
                    pos_box_combined[ix][iy][1]=pos_box_y[ix][iy]
                    
            vx_grad = np.gradient(v_combined, axis=0)
            vy_grad = np.gradient(v_combined, axis=1)
            
            vx_grad_A = np.gradient(v_A_combined, axis=0)
            vy_grad_A = np.gradient(v_A_combined, axis=1)
            
            vx_grad_B = np.gradient(v_B_combined, axis=0)
            vy_grad_B = np.gradient(v_B_combined, axis=1)
            
            vx_grad_Dif = np.gradient(v_Dif_combined, axis=0)
            vy_grad_Dif = np.gradient(v_Dif_combined, axis=1)

            #v_grad = np.gradient
            #vx_grad = np.gradient(v_avg_x_no_gas, pos_box_x)   #central diff. du_dx
            #vy_grad = np.gradient(v_avg_y_no_gas, pos_box_y)  #central diff. dv_dy
            
            vel_gradx_x = vx_grad[:,:,0]
            vel_gradx_y = vx_grad[:,:,1]
            vel_grady_x = vy_grad[:,:,0]
            vel_grady_y = vy_grad[:,:,1]
            
            vel_gradx_xA = vx_grad_A[:,:,0]
            vel_gradx_yA = vx_grad_A[:,:,1]
            vel_grady_xA = vy_grad_A[:,:,0]
            vel_grady_yA = vy_grad_A[:,:,1]
            
            vel_gradx_xB = vx_grad_B[:,:,0]
            vel_gradx_yB = vx_grad_B[:,:,1]
            vel_grady_xB = vy_grad_B[:,:,0]
            vel_grady_yB = vy_grad_B[:,:,1]
            
            vel_gradx_xDif = vx_grad_Dif[:,:,0]
            vel_gradx_yDif = vx_grad_Dif[:,:,1]
            vel_grady_xDif = vy_grad_Dif[:,:,0]
            vel_grady_yDif = vy_grad_Dif[:,:,1]
                        
            div = vel_gradx_x + vel_grady_y
            divA = vel_gradx_xA + vel_grady_yA
            divB = vel_gradx_xB + vel_grady_yB
            divDif = vel_gradx_xDif + vel_grady_yDif
            
            curl = -vel_grady_x + vel_gradx_y
            curlA = -vel_grady_xA + vel_gradx_yA
            curlB = -vel_grady_xB + vel_gradx_yB
            curlDif = -vel_grady_xDif + vel_gradx_yDif
            
        #Calculate average velocity per bin
        if j>(start*time_step):
            for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                    
                        vel_mag[ix][iy] = ((v_avg_x[ix][iy]**2+v_avg_y[ix][iy]**2)**0.5)    #Average velocity per bin of all particles relative to largest preferred velocity (peB)
                        vel_magA[ix][iy] = ((v_avg_xA[ix][iy]**2+v_avg_yA[ix][iy]**2)**0.5) #Average velocity per bin of type A particles relative to preferred velocity (peA)
                        vel_magB[ix][iy] = ((v_avg_xB[ix][iy]**2+v_avg_yB[ix][iy]**2)**0.5) #Average velocity per bin of type B particles relative to preferred velocity (peB)
                        vel_magDif[ix][iy] = (vel_magB[ix][iy]-vel_magA[ix][iy])        #Difference in magnitude of average velocity per bin between type B and A particles
            

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
                          
        # Convert position to be > 0 to place in list mesh
        com_x_ind_orig = int(com_tmp_posX / sizeBin)
        com_y_ind_orig = int(com_tmp_posY / sizeBin)
        
        offset_x = h_box - com_tmp_posX
        offset_y = h_box - com_tmp_posY
        
        com_tmp_posX_new = com_tmp_posX + offset_x
        com_tmp_posY_new = com_tmp_posY + offset_y
        
        com_x_ind = int(com_tmp_posX_new / sizeBin)
        com_y_ind = int(com_tmp_posY_new / sizeBin)
        
        com_tmp_posX_new = pos_box_x[com_x_ind][com_y_ind]
        com_tmp_posY_new = pos_box_y[com_x_ind][com_y_ind]
        
        com_x_ind_offset = com_x_ind - com_x_ind_orig
        com_y_ind_offset = com_y_ind - com_y_ind_orig
        
        if j>int(start*time_step): 
            
            
        
            pos_box_x_arr = np.zeros(np.shape(pos_box_x))
            pos_box_y_arr = np.zeros(np.shape(pos_box_x))
            
            velnorm_box_x_arr = np.zeros(np.shape(pos_box_x))
            velnorm_box_y_arr = np.zeros(np.shape(pos_box_x))
            
            vmag_box_arr = np.zeros(np.shape(pos_box_x))
            vmagA_box_arr = np.zeros(np.shape(pos_box_x))
            vmagB_box_arr = np.zeros(np.shape(pos_box_x))
            vmagDif_box_arr = np.zeros(np.shape(pos_box_x))
            
            div_box_arr = np.zeros(np.shape(pos_box_x))
            divA_box_arr = np.zeros(np.shape(pos_box_x))
            divB_box_arr = np.zeros(np.shape(pos_box_x))
            divDif_box_arr = np.zeros(np.shape(pos_box_x))
            
            curl_box_arr = np.zeros(np.shape(pos_box_x))
            curlA_box_arr = np.zeros(np.shape(pos_box_x))
            curlB_box_arr = np.zeros(np.shape(pos_box_x))
            curlDif_box_arr = np.zeros(np.shape(pos_box_x))
            
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    new_box_xpos = ix + com_x_ind_offset
                    new_box_ypos = iy + com_y_ind_offset
                    
                    if new_box_xpos>=NBins:
                        new_box_xpos=new_box_xpos-NBins
                    elif new_box_xpos < 0:
                        new_box_xpos=new_box_xpos+NBins
                        
                    if new_box_ypos>=NBins:
                        new_box_ypos=new_box_ypos-NBins
                    elif new_box_ypos < 0:
                        new_box_ypos=new_box_ypos+NBins
                        
                    velnorm_box_x_arr[new_box_xpos][new_box_ypos] = vel_normx[ix][iy]
                    velnorm_box_y_arr[new_box_xpos][new_box_ypos] = vel_normy[ix][iy]
                    
                    vmag_box_arr[new_box_xpos][new_box_ypos] = vel_mag[ix][iy]
                    vmagA_box_arr[new_box_xpos][new_box_ypos] = vel_magA[ix][iy]
                    vmagB_box_arr[new_box_xpos][new_box_ypos] = vel_magB[ix][iy]
                    vmagDif_box_arr[new_box_xpos][new_box_ypos] = vel_magDif[ix][iy]
                    
                    div_box_arr[new_box_xpos][new_box_ypos] = div[ix][iy]
                    divA_box_arr[new_box_xpos][new_box_ypos] = divA[ix][iy]
                    divB_box_arr[new_box_xpos][new_box_ypos] = divB[ix][iy]
                    divDif_box_arr[new_box_xpos][new_box_ypos] = divDif[ix][iy]
                    
                    curl_box_arr[new_box_xpos][new_box_ypos] = curl[ix][iy]
                    curlA_box_arr[new_box_xpos][new_box_ypos] = curlA[ix][iy]
                    curlB_box_arr[new_box_xpos][new_box_ypos] = curlB[ix][iy]
                    curlDif_box_arr[new_box_xpos][new_box_ypos] = curlDif[ix][iy]
            for h in range(0, partNum):
                pos[h,0]=pos[h,0]+offset_x
                
                #Enforce periodic boundary conditions
                pos_abs = np.abs(pos[h,0])
                if pos_abs>=h_box:
                    if pos[h,0] < -h_box:
                        pos[h,0] += l_box
                    else:
                        pos[h,0] -= l_box
                        
                pos[h,1]=pos[h,1]+offset_y
                
                #Enforce periodic boundary conditions
                pos_abs = np.abs(pos[h,1])
                if pos_abs>=h_box:
                    if pos[h,1] < -h_box:
                        pos[h,1] += l_box
                    else:
                        pos[h,1] -= l_box
                
            pos[:,0]+h_box
            new_green = '#39FF14'
            div_min = -2
            div_max = 2
            curl_min = -2
            curl_max = 2
            Cmag_max=10**2
            Cmag_min=10**0
            
            Cmag_max2=4
            Cmag_min2=-4
            mag_max=2
            mag_min=-1
                        
            sz = 0.75
            
            print(np.min(vmag_arr))
            print(np.min(np.abs(vx_arr)))
            print(np.min(np.abs(vy_arr)))
            typ0_arr = np.where(typ==0)[0]
            typ1_arr = np.where(typ==1)[0]
            
            fig = plt.figure(figsize=(8,6)) 
            ax = fig.add_subplot(111)           
            scatter = plt.scatter(pos[:,0]+h_box, pos[:,1]+h_box,
                                  c=symlog(vmag_arr), cmap=myCols,
                                  s=sz, edgecolors='none')
            plt.quiver(pos_box_x, pos_box_y, velnorm_box_x_arr, velnorm_box_y_arr, color='black')

            # I can plot this in a simpler way: subtract from all particles without looping
            #xImages, yImages = zip(*imageParts)
            #periodicIm = plt.scatter(xImages, yImages, c='#DCDCDC', s=sz, edgecolors='none')
            # Get colorbar
            #plt.clim(0, 6)  # limit the colorbar
            
            norm= matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
    
            sm = plt.cm.ScalarMappable(norm=norm, cmap = scatter.cmap)
            sm.set_array([])
            #tick_lev = np.arange(0.0, 7.0, 1.0)
            tick_lev = np.arange(-2.0, 2.0+2.0/10, 2.0/10)
            #plt.clim(-2.0, 2.0)
            clb = fig.colorbar(sm, ax=ax, ticks=tick_lev, extend='both')
            #cbar = plt.colorbar(scatter)
            clb.ax.tick_params(labelsize=18)  
            clb.set_label(r'$\mathrm{log}_{10}|\mathbf{v}|$', rotation=270, labelpad=25, fontsize=20)
        
            # Limits and ticks
            viewBuff = buff / 2.0
            plt.xlim(0 - viewBuff, l_box + viewBuff)
            plt.ylim(0 - viewBuff, l_box + viewBuff)
            
            pad = str(j).zfill(4)
            
            plt.tick_params(axis='both', which='both',
                            bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                
            
            if drawBins:
                # Add the bins as vertical and horizontal lines:
                for binInd in range(0, nBins):
                    coord = (sizeBin * binInd) - h_box
                    plt.axvline(x=coord, c='k', lw=1.0, zorder=0)
                    plt.axhline(y=coord, c='k', lw=1.0, zorder=0)
            
            plt.text(0.74, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst) + ' ' + r'$\tau_\mathrm{r}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
            plt.tight_layout()
            plt.savefig(outPath + out + pad + '.png', dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(8,6)) 
            ax = fig.add_subplot(111)           
            scatter = plt.scatter(pos[typ1_arr,0]+h_box, pos[typ1_arr,1]+h_box,
                                  c=symlog(vmag_arr[typ1_arr]), cmap=myCols,
                                  s=sz, edgecolors='none')
            plt.quiver(pos_box_x, pos_box_y, velnorm_box_x_arr, velnorm_box_y_arr, color='black')

            # I can plot this in a simpler way: subtract from all particles without looping
            #xImages, yImages = zip(*imageParts)
            #periodicIm = plt.scatter(xImages, yImages, c='#DCDCDC', s=sz, edgecolors='none')
            # Get colorbar
            #plt.clim(0, 6)  # limit the colorbar
            
            norm= matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
    
            sm = plt.cm.ScalarMappable(norm=norm, cmap = scatter.cmap)
            sm.set_array([])
            #tick_lev = np.arange(0.0, 7.0, 1.0)
            tick_lev = np.arange(-2.0, 2.0+2.0/10, 2.0/10)
            #plt.clim(-2.0, 2.0)
            clb = fig.colorbar(sm, ax=ax, ticks=tick_lev, extend='both')
            #cbar = plt.colorbar(scatter)
            clb.ax.tick_params(labelsize=18)  
            clb.set_label(r'$\mathrm{log}_{10}|\mathbf{v_\mathrm{B}}|$', rotation=270, labelpad=25, fontsize=20)
        
            # Limits and ticks
            viewBuff = buff / 2.0
            plt.xlim(0 - viewBuff, l_box + viewBuff)
            plt.ylim(0 - viewBuff, l_box + viewBuff)
            
            pad = str(j).zfill(4)
            
            plt.tick_params(axis='both', which='both',
                            bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                
            
            if drawBins:
                # Add the bins as vertical and horizontal lines:
                for binInd in range(0, nBins):
                    coord = (sizeBin * binInd) - h_box
                    plt.axvline(x=coord, c='k', lw=1.0, zorder=0)
                    plt.axhline(y=coord, c='k', lw=1.0, zorder=0)
            
            plt.text(0.74, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst) + ' ' + r'$\tau_\mathrm{r}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
            plt.tight_layout()
            plt.savefig(outPath + out3 + pad + '.png', dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(8,6)) 
            ax = fig.add_subplot(111)           
            scatter = plt.scatter(pos[typ0_arr,0]+h_box, pos[typ0_arr,1]+h_box,
                                  c=symlog(vmag_arr[typ0_arr]), cmap=myCols,
                                  s=sz, edgecolors='none')
            plt.quiver(pos_box_x, pos_box_y, velnorm_box_x_arr, velnorm_box_y_arr, color='black')

            # I can plot this in a simpler way: subtract from all particles without looping
            #xImages, yImages = zip(*imageParts)
            #periodicIm = plt.scatter(xImages, yImages, c='#DCDCDC', s=sz, edgecolors='none')
            # Get colorbar
            #plt.clim(0, 6)  # limit the colorbar
            
            norm= matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
    
            sm = plt.cm.ScalarMappable(norm=norm, cmap = scatter.cmap)
            sm.set_array([])
            #tick_lev = np.arange(0.0, 7.0, 1.0)
            tick_lev = np.arange(-2.0, 2.0+2.0/10, 2.0/10)
            #plt.clim(-2.0, 2.0)
            clb = fig.colorbar(sm, ax=ax, ticks=tick_lev, extend='both')
            #cbar = plt.colorbar(scatter)
            clb.ax.tick_params(labelsize=18)  
            clb.set_label(r'$\mathrm{log}_{10}|\mathbf{v_\mathrm{B}}|$', rotation=270, labelpad=25, fontsize=20)
        
            # Limits and ticks
            viewBuff = buff / 2.0
            plt.xlim(0 - viewBuff, l_box + viewBuff)
            plt.ylim(0 - viewBuff, l_box + viewBuff)
            
            pad = str(j).zfill(4)
            
            plt.tick_params(axis='both', which='both',
                            bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                
            
            if drawBins:
                # Add the bins as vertical and horizontal lines:
                for binInd in range(0, nBins):
                    coord = (sizeBin * binInd) - h_box
                    plt.axvline(x=coord, c='k', lw=1.0, zorder=0)
                    plt.axhline(y=coord, c='k', lw=1.0, zorder=0)
            
            plt.text(0.74, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst) + ' ' + r'$\tau_\mathrm{r}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
            plt.tight_layout()
            plt.savefig(outPath + out2 + pad + '.png', dpi=300)
            plt.close()
            
            
        #label previous positions for velocity calculation
        pos_prev = pos.copy()
        
        