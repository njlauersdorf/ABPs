#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:17:46 2020

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
3.b.ii.) If distance is less than LJ cutoff, store as effective diameter
3c.) Plot particle position colored by effective diameter
'''

import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math

hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
#imgPath='/pine/scr/n/j/njlauers/scm_tmpdir/phase_comp_new_edge_video/'
#imgPath2='/pine/scr/n/j/njlauers/scm_tmpdir/phase_comp_new_edge_txt/'
r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
#inFile='pa150_pb500_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'pa150_pb300_xa50_ep1_phi60_pNum10000.gsd'
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/updated_radial_phase_dens_new_edge/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
#outPath='/Volumes/External/TestRun/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')
# Inside and outside activity from command line

#Label simulation parameters
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac = float(sys.argv[4])
eps = float(sys.argv[5])
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))

#Determine which activity is the slow activity or if system is monodisperse
if peA<peB:
    typS=0
elif peA>peB:
    typS=1
else:
    typS=2
  
try:
    phi = float(sys.argv[6])
    intPhi = int(phi)
    phi /= 100.
except:
    phi = 0.6
    intPhi = 60

try:
    dtau = float(sys.argv[7])
except:
    dtau = 0.000001

# Set some constants
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster

import numpy as np
import math
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections

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

def roundUp(n, decimals=0):
    '''Round up size of bins to account for floating point inaccuracy'''
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def getNBins(length, minSz=(2**(1./6.))):
    "Given box size, return number of bins"
    initGuess = int(length) + 1
    nBins = initGuess
    # This loop only exits on function return
    while True:
        if length / nBins > minSz:
            return nBins
        else:
            nBins -= 1
def computeTauPerTstep(epsilon, mindt=0.000001):
    '''Read in epsilon, output tauBrownian per timestep'''
#    if epsilon != 1.:
#        mindt=0.00001
    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau
def avgCollisionForce(peNet):
    '''Computed from the integral of possible angles'''
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    return (magnitude * peNet) / (np.pi) 
def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ
def ljForce(r, eps, sigma=1.):
    '''Compute the Lennard-Jones force'''
    div = (sigma/r)
    dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
    return dU
def getLat(peNet, eps):
    '''Get the lattice spacing for any pe'''
    if peNet == 0:
        return 2.**(1./6.)
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(peNet):
            r -= j
        r += j
    return r  
def maximum(a, b, c, d): 
  
    if (a >= b) and (a >= c) and (a >= d): 
        largest = a 
    elif (b >= a) and (b >= c) and (b >= d): 
        largest = b 
    elif (c >= a) and (c >= b) and (c >= d):
        largest = c
    else: 
        largest = d
          
    return largest 
def minimum(a, b, c, d): 
  
    if (a <= b) and (a <= c) and (a <= d): 
        smallest = a 
    elif (b <= a) and (b <= c) and (b <= d): 
        smallest = b 
    elif (c <= a) and (c <= b) and (c <= d):
        smallest = c
    else: 
        smallest = d
          
    return smallest 

lat=getLat(peNet,eps)

tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size

#    f = hoomd.open(name=gsd_file, mode='rb')  # open gsd file with hoomd
out_file, file_extension = os.path.splitext(inFile)   # get base name
out_file = "spatial_delta_" + out_file + "_frame"

f = hoomd.open(name=inFile, mode='rb')
box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep
dumps = int(f.__len__())                # get number of timesteps dumped

start = 0                       # gives first frame to read
end = dumps                     # gives last frame to read
start = dumps - 2

positions = np.zeros((end), dtype=np.ndarray)       # array of positions
types = np.zeros((end), dtype=np.ndarray)           # particle types
box_data = np.zeros((1), dtype=np.ndarray)          # box dimensions
timesteps = np.zeros((end), dtype=np.float64)       # timesteps

# Get relevant data from long.gsd file
with hoomd.open(name=inFile, mode='rb') as t:
    snap = t[0]
    box_data = snap.configuration.box
    for iii in range(start, end):
        snap = t[iii]                               # snapshot of frame
        types[iii] = snap.particles.typeid          # get types
        positions[iii] = snap.particles.position    # get positions
        timesteps[iii] = snap.configuration.step    # get timestep

# Get index order for chronological timestep sorting
newInd = slowSort(timesteps)
# Use these indexes to reorder other arrays
indSort(timesteps, newInd)
indSort(positions, newInd)
indSort(types, newInd)

if chkSort(timesteps):
    print("Array succesfully sorted")
else:
    print("Array not sorted")
timesteps -= timesteps[0]

# Get number of each type of particle
partNum = len(types[start])
part_A = int(partNum * parFrac)
part_B = partNum - part_A

# Feed data into freud analysis software
l_box = box_data[0]
h_box = l_box / 2.0
a_box = l_box * l_box

# Make the mesh
r_cut = 2**(1./6.)
nBins = getNBins(l_box, r_cut)
sizeBin = roundUp((l_box / nBins), 6)

# Enlarge the box to include the periodic images
buff = float(int(r_cut * 2.0) + 1)

# Image rendering options
drawBins = False
#myCols = plt.cm.viridis
#myCols = plt.cm.jet
myCols = plt.cm.jet_r

outTxt = 'effective_diameter_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'peNet'.center(15) + ' ' +\
        'avg_effSigma'.center(15) + ' ' +\
        'stdDev_Sigma'.center(15) + ' ' +\
        'uncty_Sigma'.center(15) + ' ' +\
        'count_A_edge'.center(15) + ' ' +\
        'count_B_edge'.center(15) + ' ' +\
        'count_A_gas'.center(15) + ' ' +\
        'count_B_gas'.center(15) + ' ' +\
        'count_A_bulk'.center(15) + ' ' +\
        'count_B_bulk'.center(15) + '\n')
g.close()


with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0                   # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.
    typ = snap.particles.typeid
    partNum = len(typ)
    # Set up cluster computation using box
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    #my_clust = cluster.Cluster()
    #c_props = cluster.ClusterProperties()
    # Compute each mesh
    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / NBins), 6)
    # Loop through each timestep
    tot_count_gas=np.zeros((end,2))
    tot_count_dense=np.zeros((end,2))
    tot_count_edge=np.zeros((end,2))
    rad_count_gas=np.zeros((end,int(NBins/2)-1,2))
    rad_count_dense=np.zeros((end,int(NBins/2)-1,2))
    rad_count_edge=np.zeros((end,int(NBins/2)-1,2))
    time_arr=np.zeros(end)
    size_arr = np.zeros(int(end))
    clust_edge_arr=np.zeros(end)
    
    average_clust_radius=0
    count_of_rads=0
    gas_area_arr = np.zeros(end)
    dense_area_arr = np.zeros(end)
    edge_area_arr = np.zeros(end)
    dense_area_big_arr = np.zeros(end)
    edge_area_big_arr = np.zeros(end)
    count_avg=0
    for j in range(start, int(end)):
        print('j')
        print(j)
        r=np.arange(1,h_box,1)
        # Outfile to write data to
        imgbase = add + 'pressure_pa' + str(peA) +\
       '_pb' + str(peB) +\
       '_xa' + str(parFrac) +\
       '_phi' + str(intPhi) +\
       '_ep' + '{0:.3f}'.format(eps)+'_frame'+str(j)
        imgFile = imgbase + '.png'
        snap = t[j]
        # Easier accessors
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0
        typ = snap.particles.typeid                 # type

        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
        
        xy = np.delete(pos, 2, 1)
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        time_arr[j]=tst
        # Compute clusters for this timestep
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        
        cl_all=freud.cluster.Cluster()                      #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()         #Define cluster properties
        ids = cl_all.cluster_idx              # get id of each cluster
        clp_all.compute(system_all, ids)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        min_size=int(partNum/5)
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]
        large_clust_ind_all=np.where(clust_size>min_size)
        

        if len(large_clust_ind_all[0])>0:
            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box
            com_x_ind = int(com_tmp_posX / sizeBin)
            com_y_ind = int(com_tmp_posY / sizeBin)
            
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            
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
            # If sufficient neighbor bins are empty, we have an edge
            thresh = 1.5

            #Determines edge bins
            # Loop through x index of mesh
            for ix in range(0, len(occParts)):
        
                # If at right edge, wrap to left
                if (ix + 1) != NBins:
                    lookx = [ix-1, ix, ix+1]
                else:
                    lookx = [ix-1, ix, 0]
                
                # Loop through y index of mesh

                for iy in range(0, len(occParts[ix])):
                    # Reset neighbor counter
                    count = 0
                    # If the bin is not occupied, skip it
                    if occParts[ix][iy] == 0:
                        continue
                    # If at top edge, wrap to bottom
                    if (iy + 1) != NBins:
                        looky = [iy-1, iy, iy+1]
                    else:
                        looky = [iy-1, iy, 0]
                    # Loop through surrounding x-index
                    for indx in lookx:
                        # Loop through surrounding y-index
                        for indy in looky:
                    
                            # If neighbor bin is NOT occupied
                            if occParts[indx][indy] == 0:
                                # If neighbor bin shares a vertex
                                if indx != ix and indy != iy:
                                    count += 0.5
                                # If neighbor bin shares a side
                                else:
                                    count += 1

                    # If sufficient neighbors are empty, we found an edge
                    if count >= thresh:
                        edgeBin[indx][indy] = 1
            PhaseParts=np.zeros(len(pos))
            PhaseParts2=np.zeros(len(pos))
            PhasePartsarea=np.zeros(len(pos))
            gasBins = 0
            bulkBins = 0
            edgeBins=0
            edgeBinsbig = 0
            bulkBinsbig = 0
            testIDs = [[0 for b in range(NBins)] for a in range(NBins)]
            testIDs_area = [[0 for b in range(NBins)] for a in range(NBins)]

            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    
                    # Is the bin an edge?
                    
                    if edgeBin[ix][iy] == 1:
                        testIDs[ix][iy] = 0
                        testIDs_area[ix][iy] = 0
                        edgeBins+=1
                        continue
                    # Does the bin belong to the dense phase?

                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] >=min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            testIDs_area[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
                    testIDs_area[ix][iy] = 0
              
                
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if testIDs[ix][iy] == 0:
                        for h in range(0,len(binParts[ix][iy])):
                            PhaseParts[binParts[ix][iy][h]]=0
                            PhaseParts2[binParts[ix][iy][h]]=0
                            PhasePartsarea[binParts[ix][iy][h]]=0
                        edgeBins+=1
                        continue
                    # Does the bin belong to the dense phase?

                    if testIDs[ix][iy]==1:
                        bulkBins += 1
                        for h in range(0,len(binParts[ix][iy])):
                            PhaseParts[binParts[ix][iy][h]]=1
                            PhaseParts2[binParts[ix][iy][h]]=1
                            PhasePartsarea[binParts[ix][iy][h]]=1
                        continue
                    gasBins += 1
                    for h in range(0,len(binParts[ix][iy])):
                        PhaseParts[binParts[ix][iy][h]]=2
                        PhaseParts2[binParts[ix][iy][h]]=2
                        PhasePartsarea[binParts[ix][iy][h]]=2
                        
                        
            count_A_edge=0
            count_B_edge=0
            gas_r_lim=2.0*lat
            bulk_particle_range=3.0
            end_loop=0
            steps=0
            print(typS)
            if typS==0:
                while count_B_edge<=count_A_edge:
                    bulk_r_lim=bulk_particle_range*lat
                    search_range=int((bulk_particle_range+1)/sizeBin)
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if edgeBin[ix][iy]==1:
                                testIDs[ix][iy]=0
                                x_min_ref=ix*sizeBin
                                x_max_ref=(ix+1)*sizeBin
                                y_min_ref=iy*sizeBin
                                y_max_ref=(iy+1)*sizeBin
                        #print(ix)
                        #print(search_range)
                        #ix=NBins-3
                                if search_range<=ix<=(NBins-(search_range+1)):
                                    x_min=ix-search_range
                                    x_max=ix+search_range+1
                                    x_tot=np.arange(x_min,x_max,1)
                                elif ix<search_range:
                                    x_max=ix+search_range+1
                                    x_lim=ix-search_range
                                    x_min=(NBins+x_lim)
                                    x_range_1=np.arange(x_min,NBins)
                                    x_range_2=np.arange(0,x_max)
                                    x_tot=np.append(x_range_1,x_range_2)
                                elif ix>(NBins-(search_range+1)):
                                    x_min=ix-search_range
                                    x_lim=ix+search_range
                                    x_max=x_lim-NBins+1
                                    x_range_1=np.arange(x_min,NBins)
                                    x_range_2=np.arange(0,x_max)
                                    x_tot=np.append(x_range_1,x_range_2)

                                if search_range<=iy<=(NBins-(search_range+1)):
                                    y_min=iy-search_range
                                    y_max=iy+search_range+1
                                    y_tot=np.arange(y_min,y_max,1)
                                elif iy<search_range:
                                    y_max=iy+search_range+1
                                    y_lim=iy-search_range
                                    y_min=(NBins+y_lim)
                                    y_range_1=np.arange(y_min,NBins)
                                    y_range_2=np.arange(0,y_max)
                                    y_tot=np.append(y_range_1,y_range_2)
                                elif iy>(NBins-(search_range+1)):
                                    y_min=iy-search_range
                                    y_lim=iy+search_range
                                    y_max=y_lim-NBins+1
                                    y_range_1=np.arange(y_min,NBins)
                                    y_range_2=np.arange(0,y_max)
                                    y_tot=np.append(y_range_1,y_range_2)
                                for ix2 in x_tot:
                                    for iy2 in y_tot:
                                
                                        if testIDs[ix2][iy2] == 0:
                                            pass
                                        elif testIDs[ix2][iy2] == 1:
                                            x_min_check=ix2*sizeBin
                                            x_max_check=(ix2+1)*sizeBin
                                            y_min_check=iy2*sizeBin
                                            y_max_check=(iy2+1)*sizeBin
                                            x1=np.abs(x_min_check-x_min_ref)
                                            x2=np.abs(x_max_check-x_min_ref)
                                            x3=np.abs(x_min_check-x_max_ref)
                                            x4=np.abs(x_max_check-x_max_ref)
                                            y1=np.abs(y_min_check-y_min_ref)
                                            y2=np.abs(y_max_check-y_min_ref)
                                            y3=np.abs(y_min_check-y_max_ref)
                                            y4=np.abs(y_max_check-y_max_ref)
                                            x_max=maximum(x1,x2,x3,x4)
                                            y_max=maximum(y1,y2,y3,y4)
                                            x_min=minimum(x1,x2,x3,x4)
                                            y_min=minimum(y1,y2,y3,y4)
                                            r_min=(y_min**2+x_min**2)**0.5
                                            r_max=(y_max**2+x_max**2)**0.5
                                            if r_min<=bulk_r_lim:
                                                if r_max<=bulk_r_lim:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    r_cut_new=np.abs(r_max-bulk_r_lim)
                                                    if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                        testIDs[ix2][iy2]=0
                                                    else:
                                                        testIDs[ix2][iy2]=1
                                            else:
                                                testIDs[ix2][iy2]=1
                                        elif testIDs[ix2][iy2] == 2:
                                            x_min_check=ix2*sizeBin
                                            x_max_check=(ix2+1)*sizeBin
                                            y_min_check=iy2*sizeBin
                                            y_max_check=(iy2+1)*sizeBin
                                            x1=np.abs(x_min_check-x_min_ref)
                                            x2=np.abs(x_max_check-x_min_ref)
                                            x3=np.abs(x_min_check-x_max_ref)
                                            x4=np.abs(x_max_check-x_max_ref)
                                            y1=np.abs(y_min_check-y_min_ref)
                                            y2=np.abs(y_max_check-y_min_ref)
                                            y3=np.abs(y_min_check-y_max_ref)
                                            y4=np.abs(y_max_check-y_max_ref)
                                            x_max=maximum(x1,x2,x3,x4)
                                            y_max=maximum(y1,y2,y3,y4)
                                            x_min=minimum(x1,x2,x3,x4)
                                            y_min=minimum(y1,y2,y3,y4)
                                            r_min=(y_min**2+x_min**2)**0.5
                                            r_max=(y_max**2+x_max**2)**0.5
                                            if r_min<=gas_r_lim:
                                                if r_max<=gas_r_lim:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    r_cut_new=r_max-gas_r_lim
                                                    if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                        testIDs[ix2][iy2]=0
                                                    else:
                                                        testIDs[ix2][iy2]=2
                                            else:
                                                testIDs[ix2][iy2]=2
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if testIDs[ix][iy] == 0:
                                for h in range(0,len(binParts[ix][iy])):
                                    PhaseParts[binParts[ix][iy][h]]=0
                                    PhaseParts2[binParts[ix][iy][h]]=0
                                edgeBins+=1
                                continue
                    # Does the bin belong to the dense phase?

                            if testIDs[ix][iy]==1:
                                bulkBins += 1
                                for h in range(0,len(binParts[ix][iy])):
                                    PhaseParts[binParts[ix][iy][h]]=1
                                    PhaseParts2[binParts[ix][iy][h]]=1
                                continue
                            gasBins += 1
                            for h in range(0,len(binParts[ix][iy])):
                                PhaseParts[binParts[ix][iy][h]]=2
                                PhaseParts2[binParts[ix][iy][h]]=2
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            count=0
                            phase=0
                            for f in range(0,len(binParts[ix][iy])):
                                count+=1
                                phase+=PhaseParts2[binParts[ix][iy][f]]
                            if count>0:
                                avg_phase=int(phase/count)
                            else:
                                avg_phase=2
                            if avg_phase==0:
                                testIDs[ix][iy]=0
                                edgeBins+=1
                            elif avg_phase==1:
                                testIDs[ix][iy]=1
                                bulkBins+=1
                            else:
                                testIDs[ix][iy]=2
                                gasBins+=1
                    
                    edgeParts_num=np.where(PhaseParts2==0)
                    bulkParts_num=np.where(PhaseParts2==1)
                    gasParts_num=np.where(PhaseParts2==2)
                    gas_type=typ[gasParts_num[0]]
                    bulk_type=typ[bulkParts_num[0]]
                    edge_type=typ[edgeParts_num[0]]
                    gas_pos=pos[gasParts_num[0]]
                    bulk_pos=pos[bulkParts_num[0]]
                    edge_pos=pos[edgeParts_num[0]]
                    count_A_gas=len(np.where(gas_type==0)[0])
                    count_B_gas=len(np.where(gas_type==1)[0])
                    count_A_dense=len(np.where(bulk_type==0)[0])
                    count_B_dense=len(np.where(bulk_type==1)[0])
                    count_A_edge=len(np.where(edge_type==0)[0])
                    count_B_edge=len(np.where(edge_type==1)[0]) 
                    
                    if count_B_edge>count_A_edge:
                        break
                    
                    if bulk_r_lim>=(35.0*lat):
                        bulk_particle_range=9.0
                        end_loop=1
                
                    gas_pos=pos[gasParts_num[0]]
                    bulk_pos=pos[bulkParts_num[0]]
                    edge_pos=pos[edgeParts_num[0]]
                    
                    binArea = sizeBin * sizeBin
                    # Area of each phase
                    gasArea = binArea * gasBins
            
                    bulkArea = binArea * bulkBins
                    edgeArea = binArea * edgeBins
                    count_A_edge_final=count_A_edge
                    count_B_edge_final=count_B_edge
                    count_A_dense_final=count_A_dense
                    count_B_dense_final=count_B_dense
                    count_A_gas_final=count_A_gas
                    count_B_gas_final=count_B_gas
                    edgeParts_final=edgeParts_num
                    bulkParts_final=bulkParts_num
                    gasParts_final=gasParts_num
                    PhaseParts_final=PhaseParts2
                    bulk_particle_range_final=bulk_particle_range
                    bulk_particle_range+=1
                    steps+=1
                    bulk_r_lim_final=bulk_r_lim
                    if end_loop==1:
                        end_loop+=1
                    elif end_loop==2:
                        break
            elif typS==1:
                while count_B_edge>=count_A_edge:
                    bulk_r_lim=bulk_particle_range*lat
                    search_range=int((bulk_particle_range+1)/sizeBin)
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if edgeBin[ix][iy]==1:
                                testIDs[ix][iy]=0
                                x_min_ref=ix*sizeBin
                                x_max_ref=(ix+1)*sizeBin
                                y_min_ref=iy*sizeBin
                                y_max_ref=(iy+1)*sizeBin
                        #print(ix)
                        #print(search_range)
                        #ix=NBins-3
                                if search_range<=ix<=(NBins-(search_range+1)):
                                    x_min=ix-search_range
                                    x_max=ix+search_range+1
                                    x_tot=np.arange(x_min,x_max,1)
                                elif ix<search_range:
                                    x_max=ix+search_range+1
                                    x_lim=ix-search_range
                                    x_min=(NBins+x_lim)
                                    x_range_1=np.arange(x_min,NBins)
                                    x_range_2=np.arange(0,x_max)
                                    x_tot=np.append(x_range_1,x_range_2)
                                elif ix>(NBins-(search_range+1)):
                                    x_min=ix-search_range
                                    x_lim=ix+search_range
                                    x_max=x_lim-NBins+1
                                    x_range_1=np.arange(x_min,NBins)
                                    x_range_2=np.arange(0,x_max)
                                    x_tot=np.append(x_range_1,x_range_2)

                                if search_range<=iy<=(NBins-(search_range+1)):
                                    y_min=iy-search_range
                                    y_max=iy+search_range+1
                                    y_tot=np.arange(y_min,y_max,1)
                                elif iy<search_range:
                                    y_max=iy+search_range+1
                                    y_lim=iy-search_range
                                    y_min=(NBins+y_lim)
                                    y_range_1=np.arange(y_min,NBins)
                                    y_range_2=np.arange(0,y_max)
                                    y_tot=np.append(y_range_1,y_range_2)
                                elif iy>(NBins-(search_range+1)):
                                    y_min=iy-search_range
                                    y_lim=iy+search_range
                                    y_max=y_lim-NBins+1
                                    y_range_1=np.arange(y_min,NBins)
                                    y_range_2=np.arange(0,y_max)
                                    y_tot=np.append(y_range_1,y_range_2)
                                for ix2 in x_tot:
                                    for iy2 in y_tot:
                                
                                        if testIDs[ix2][iy2] == 0:
                                            pass
                                        elif testIDs[ix2][iy2] == 1:
                                            x_min_check=ix2*sizeBin
                                            x_max_check=(ix2+1)*sizeBin
                                            y_min_check=iy2*sizeBin
                                            y_max_check=(iy2+1)*sizeBin
                                            x1=np.abs(x_min_check-x_min_ref)
                                            x2=np.abs(x_max_check-x_min_ref)
                                            x3=np.abs(x_min_check-x_max_ref)
                                            x4=np.abs(x_max_check-x_max_ref)
                                            y1=np.abs(y_min_check-y_min_ref)
                                            y2=np.abs(y_max_check-y_min_ref)
                                            y3=np.abs(y_min_check-y_max_ref)
                                            y4=np.abs(y_max_check-y_max_ref)
                                            x_max=maximum(x1,x2,x3,x4)
                                            y_max=maximum(y1,y2,y3,y4)
                                            x_min=minimum(x1,x2,x3,x4)
                                            y_min=minimum(y1,y2,y3,y4)
                                            r_min=(y_min**2+x_min**2)**0.5
                                            r_max=(y_max**2+x_max**2)**0.5
                                            if r_min<=bulk_r_lim:
                                                if r_max<=bulk_r_lim:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    r_cut_new=np.abs(r_max-bulk_r_lim)
                                                    if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                        testIDs[ix2][iy2]=0
                                                    else:
                                                        testIDs[ix2][iy2]=1
                                            else:
                                                testIDs[ix2][iy2]=1
                                        elif testIDs[ix2][iy2] == 2:
                                            x_min_check=ix2*sizeBin
                                            x_max_check=(ix2+1)*sizeBin
                                            y_min_check=iy2*sizeBin
                                            y_max_check=(iy2+1)*sizeBin
                                            x1=np.abs(x_min_check-x_min_ref)
                                            x2=np.abs(x_max_check-x_min_ref)
                                            x3=np.abs(x_min_check-x_max_ref)
                                            x4=np.abs(x_max_check-x_max_ref)
                                            y1=np.abs(y_min_check-y_min_ref)
                                            y2=np.abs(y_max_check-y_min_ref)
                                            y3=np.abs(y_min_check-y_max_ref)
                                            y4=np.abs(y_max_check-y_max_ref)
                                            x_max=maximum(x1,x2,x3,x4)
                                            y_max=maximum(y1,y2,y3,y4)
                                            x_min=minimum(x1,x2,x3,x4)
                                            y_min=minimum(y1,y2,y3,y4)
                                            r_min=(y_min**2+x_min**2)**0.5
                                            r_max=(y_max**2+x_max**2)**0.5
                                            if r_min<=gas_r_lim:
                                                if r_max<=gas_r_lim:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    r_cut_new=r_max-gas_r_lim
                                                    if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                        testIDs[ix2][iy2]=0
                                                    else:
                                                        testIDs[ix2][iy2]=2
                                            else:
                                                testIDs[ix2][iy2]=2
                    for ix in range(0, len(occParts)):
                        for iy in range(0, len(occParts)):
                            if testIDs[ix][iy] == 0:
                                for h in range(0,len(binParts[ix][iy])):
                                    PhaseParts[binParts[ix][iy][h]]=0
                                    PhaseParts2[binParts[ix][iy][h]]=0
                                edgeBins+=1
                                continue
                    # Does the bin belong to the dense phase?

                            if testIDs[ix][iy]==1:
                                bulkBins += 1
                                for h in range(0,len(binParts[ix][iy])):
                                    PhaseParts[binParts[ix][iy][h]]=1
                                    PhaseParts2[binParts[ix][iy][h]]=1
                                continue
                            gasBins += 1
                            for h in range(0,len(binParts[ix][iy])):
                                PhaseParts[binParts[ix][iy][h]]=2
                                PhaseParts2[binParts[ix][iy][h]]=2

                    edgeParts_num=np.where(PhaseParts2==0)
                    bulkParts_num=np.where(PhaseParts2==1)
                    gasParts_num=np.where(PhaseParts2==2)

                    gas_type=typ[gasParts_num[0]]
                    bulk_type=typ[bulkParts_num[0]]
                    edge_type=typ[edgeParts_num[0]]
                    count_A_gas=len(np.where(gas_type==0)[0])
                    count_B_gas=len(np.where(gas_type==1)[0])
                    count_A_dense=len(np.where(bulk_type==0)[0])
                    count_B_dense=len(np.where(bulk_type==1)[0])
                    count_A_edge=len(np.where(edge_type==0)[0])
                    count_B_edge=len(np.where(edge_type==1)[0]) 
                    if count_B_edge>count_A_edge:
                        break
                    if bulk_r_lim>=(35.0*lat):
                        bulk_particle_range=9.0
                        end_loop=1
                    edgeParts_final=edgeParts_num
                    steps+=1
                    bulkParts_final=bulkParts_num
                    gasParts_final=gasParts_num
                    gas_pos=pos[gasParts_num[0]]
                    bulk_pos=pos[bulkParts_num[0]]
                    edge_pos=pos[edgeParts_num[0]]
                    binArea = sizeBin * sizeBin
                    # Area of each phase
                    gasArea = binArea * gasBins
            
                    bulkArea = binArea * bulkBins
                    edgeArea = binArea * edgeBins
                    count_A_edge_final=count_A_edge
                    count_B_edge_final=count_B_edge
                    count_A_dense_final=count_A_dense
                    count_B_dense_final=count_B_dense
                    count_A_gas_final=count_A_gas
                    count_B_gas_final=count_B_gas
                    bulk_particle_range_final=bulk_particle_range

                    bulk_particle_range+=1
                    bulk_r_lim_final=bulk_r_lim
                    PhaseParts_final=PhaseParts2

                    if end_loop==1:
                        end_loop+=1
                    elif end_loop==2:
                        break
            elif typS==2:
                bulk_particle_range=10.0
                bulk_r_lim=bulk_particle_range*lat
                search_range=int((bulk_particle_range+1)/sizeBin)
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if edgeBin[ix][iy]==1:
                            testIDs[ix][iy]=0
                            x_min_ref=ix*sizeBin
                            x_max_ref=(ix+1)*sizeBin
                            y_min_ref=iy*sizeBin
                            y_max_ref=(iy+1)*sizeBin
                        #print(ix)
                        #print(search_range)
                        #ix=NBins-3
                            if search_range<=ix<=(NBins-(search_range+1)):
                                x_min=ix-search_range
                                x_max=ix+search_range+1
                                x_tot=np.arange(x_min,x_max,1)
                            elif ix<search_range:
                                x_max=ix+search_range+1
                                x_lim=ix-search_range
                                x_min=(NBins+x_lim)
                                x_range_1=np.arange(x_min,NBins)
                                x_range_2=np.arange(0,x_max)
                                x_tot=np.append(x_range_1,x_range_2)
                            elif ix>(NBins-(search_range+1)):
                                x_min=ix-search_range
                                x_lim=ix+search_range
                                x_max=x_lim-NBins+1
                                x_range_1=np.arange(x_min,NBins)
                                x_range_2=np.arange(0,x_max)
                                x_tot=np.append(x_range_1,x_range_2)

                            if search_range<=iy<=(NBins-(search_range+1)):
                                y_min=iy-search_range
                                y_max=iy+search_range+1
                                y_tot=np.arange(y_min,y_max,1)
                            elif iy<search_range:
                                y_max=iy+search_range+1
                                y_lim=iy-search_range
                                y_min=(NBins+y_lim)
                                y_range_1=np.arange(y_min,NBins)
                                y_range_2=np.arange(0,y_max)
                                y_tot=np.append(y_range_1,y_range_2)
                            elif iy>(NBins-(search_range+1)):
                                y_min=iy-search_range
                                y_lim=iy+search_range
                                y_max=y_lim-NBins+1
                                y_range_1=np.arange(y_min,NBins)
                                y_range_2=np.arange(0,y_max)
                                y_tot=np.append(y_range_1,y_range_2)
                            for ix2 in x_tot:
                                for iy2 in y_tot:
                                
                                    if testIDs[ix2][iy2] == 0:
                                        pass
                                    elif testIDs[ix2][iy2] == 1:
                                        x_min_check=ix2*sizeBin
                                        x_max_check=(ix2+1)*sizeBin
                                        y_min_check=iy2*sizeBin
                                        y_max_check=(iy2+1)*sizeBin
                                        x1=np.abs(x_min_check-x_min_ref)
                                        x2=np.abs(x_max_check-x_min_ref)
                                        x3=np.abs(x_min_check-x_max_ref)
                                        x4=np.abs(x_max_check-x_max_ref)
                                        y1=np.abs(y_min_check-y_min_ref)
                                        y2=np.abs(y_max_check-y_min_ref)
                                        y3=np.abs(y_min_check-y_max_ref)
                                        y4=np.abs(y_max_check-y_max_ref)
                                        x_max=maximum(x1,x2,x3,x4)
                                        y_max=maximum(y1,y2,y3,y4)
                                        x_min=minimum(x1,x2,x3,x4)
                                        y_min=minimum(y1,y2,y3,y4)
                                        r_min=(y_min**2+x_min**2)**0.5
                                        r_max=(y_max**2+x_max**2)**0.5
                                        if r_min<=bulk_r_lim:
                                            if r_max<=bulk_r_lim:
                                                testIDs[ix2][iy2]=0
                                            else:
                                                r_cut_new=np.abs(r_max-bulk_r_lim)
                                                if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    testIDs[ix2][iy2]=1
                                        else:
                                            testIDs[ix2][iy2]=1
                                    elif testIDs[ix2][iy2] == 2:
                                        x_min_check=ix2*sizeBin
                                        x_max_check=(ix2+1)*sizeBin
                                        y_min_check=iy2*sizeBin
                                        y_max_check=(iy2+1)*sizeBin
                                        x1=np.abs(x_min_check-x_min_ref)
                                        x2=np.abs(x_max_check-x_min_ref)
                                        x3=np.abs(x_min_check-x_max_ref)
                                        x4=np.abs(x_max_check-x_max_ref)
                                        y1=np.abs(y_min_check-y_min_ref)
                                        y2=np.abs(y_max_check-y_min_ref)
                                        y3=np.abs(y_min_check-y_max_ref)
                                        y4=np.abs(y_max_check-y_max_ref)
                                        x_max=maximum(x1,x2,x3,x4)
                                        y_max=maximum(y1,y2,y3,y4)
                                        x_min=minimum(x1,x2,x3,x4)
                                        y_min=minimum(y1,y2,y3,y4)
                                        r_min=(y_min**2+x_min**2)**0.5
                                        r_max=(y_max**2+x_max**2)**0.5
                                        if r_min<=gas_r_lim:
                                            if r_max<=gas_r_lim:
                                                testIDs[ix2][iy2]=0
                                            else:
                                                r_cut_new=r_max-gas_r_lim
                                                if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                    testIDs[ix2][iy2]=0
                                                else:
                                                    testIDs[ix2][iy2]=2
                                        else:
                                            testIDs[ix2][iy2]=2
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if testIDs[ix][iy] == 0:
                            for h in range(0,len(binParts[ix][iy])):
                                PhaseParts[binParts[ix][iy][h]]=0
                                PhaseParts2[binParts[ix][iy][h]]=0
                            edgeBins+=1
                            continue
                    # Does the bin belong to the dense phase?

                        if testIDs[ix][iy]==1:
                            bulkBins += 1
                            for h in range(0,len(binParts[ix][iy])):
                                PhaseParts[binParts[ix][iy][h]]=1
                                PhaseParts2[binParts[ix][iy][h]]=1
                            continue
                        gasBins += 1
                        for h in range(0,len(binParts[ix][iy])):
                            PhaseParts[binParts[ix][iy][h]]=2
                            PhaseParts2[binParts[ix][iy][h]]=2
                                
            
                edgeParts_num=np.where(PhaseParts2==0)
                bulkParts_num=np.where(PhaseParts2==1)
                gasParts_num=np.where(PhaseParts2==2)
                gas_type=typ[gasParts_num[0]]
                bulk_type=typ[bulkParts_num[0]]
                edge_type=typ[edgeParts_num[0]]

                edgeParts_final=edgeParts_num
                bulkParts_final=bulkParts_num
                gasParts_final=gasParts_num
                count_A_gas=len(np.where(gas_type==0)[0])
                count_B_gas=len(np.where(gas_type==1)[0])
                count_A_dense=len(np.where(bulk_type==0)[0])
                count_B_dense=len(np.where(bulk_type==1)[0])
                count_A_edge=len(np.where(edge_type==0)[0])
                count_B_edge=len(np.where(edge_type==1)[0]) 
                binArea = sizeBin * sizeBin
                    # Area of each phase
                gasArea = binArea * gasBins
                gas_pos=pos[gasParts_num[0]]
                bulk_pos=pos[bulkParts_num[0]]
                edge_pos=pos[edgeParts_num[0]]
                bulkArea = binArea * bulkBins
                edgeArea = binArea * edgeBins
                count_A_edge_final=count_A_edge
                count_B_edge_final=count_B_edge
                count_A_dense_final=count_A_dense
                count_B_dense_final=count_B_dense
                count_A_gas_final=count_A_gas
                steps=1
                count_B_gas_final=count_B_gas
                bulk_particle_range_final=bulk_particle_range
                bulk_r_lim_final=bulk_r_lim
                PhaseParts_final=PhaseParts2

            if steps>0: 
                count_bulk=0
                effSigma = np.ones(count_A_dense_final+count_B_dense_final)
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        for k in binParts[ix][iy]:
                            if PhaseParts_final[k]==1:
                                # Get mesh indices
                                x_ind = ix
                                y_ind = iy
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
                                                if r < effSigma[count_bulk]:
                                                    effSigma[count_bulk] = r
                                count_bulk+=1
                            else:
                                pass
                avg_effSigma=(np.sum(effSigma)/len(effSigma))
                stdDev=0
                for i in range(0,len(effSigma)):
                    stdDev+=(effSigma[i]-avg_effSigma)**2
                total_stdDev=(stdDev/(len(effSigma)-1))**0.5
                uncertainty_mean=total_stdDev/(len(effSigma))**0.5
                g = open(outPath+outTxt, 'a')
                g.write('{0:.1f}'.format(tst).center(15) + ' ')
                g.write('{0:.1f}'.format(peNet).center(15) + ' ')
                g.write('{0:.6f}'.format(avg_effSigma).center(15) + ' ')
                g.write('{0:.6f}'.format(total_stdDev).center(15) + ' ')
                g.write('{0:.6f}'.format(uncertainty_mean).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_edge_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_edge_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_gas_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_gas_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_dense_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_dense_final).center(15) + '\n')
                g.close()
                
                
                

    