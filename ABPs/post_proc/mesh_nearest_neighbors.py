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
hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#
outPath='/Volumes/External/test_video_mono/'

#Run on Cluster
#hoomdPath='/nas/home/njlauers/hoomd-blue/build/'
#outPath='/proj/dklotsalab/users/ABPs/videos/sim_frames/'

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

# Command line arguments
infile = str(sys.argv[1])                               # Get infile (.gsd)
peA = float(sys.argv[2])                                #Activity (Pe) for species A
peB = float(sys.argv[3])                                #Activity (Pe) for species B

parFrac_orig = float(sys.argv[4])                       #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
    
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

#import modules
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

import numpy as np
from scipy import stats

import matplotlib
#matplotlib.use('Agg') #Uncomment if running on cluster
import matplotlib.pyplot as plt

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

# Create outfile name from infile name
file_name = os.path.basename(infile)
outfile, file_extension = os.path.splitext(file_name)   # get base name
out = 'defects_' + outfile + "_frame_"

f = hoomd.open(name=infile, mode='rb')  # open gsd file with hoomd

dumps = int(f.__len__())                # get number of timesteps dumped
start = 462                       # gives first frame to read
end = dumps                     # gives last frame to read

with hoomd.open(name=infile, mode='rb') as t:
    
    #Initial time step data
    snap = t[0]           

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
                        r = getDistance(pos[k],
                                        pos[ref][0] + wrapX,
                                        pos[ref][1] + wrapY)
                        r = round(r, 4)  # round value to 4 decimal places
    
                        # If LJ potential is on, store into a list (omit self)
                        if 0.1 < r <= r_cut:
                            near_neigh[k] += 1
                            if typ[ref] == 0:   # neighbor is A particle
                                A_neigh[k] += 1
                            else:               # neighbor is B particle
                                B_neigh[k] += 1
    
        # Plot position colored by neighbor number
        sz = 0.75
        scatter = plt.scatter(pos[:,0], pos[:,1],
                              c=near_neigh[:], cmap=myCols,
                              s=sz, edgecolors='none')
        # I can plot this in a simpler way: subtract from all particles without looping
        xImages, yImages = zip(*imageParts)
        periodicIm = plt.scatter(xImages, yImages, c='#DCDCDC', s=sz, edgecolors='none')
        # Get colorbar
        plt.clim(0, 6)  # limit the colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('# of Neighbors', rotation=270, labelpad=15)
    
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
        plt.savefig(outPath + out + pad + '.png', dpi=500)
        plt.close()
        