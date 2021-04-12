#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:41:30 2020

@author: nicklauersdorf
"""

import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
from shapely.geometry import MultiPoint


#from descartes.patch import PolygonPatch
# Run locally
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
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/phase_dens_edge_width/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def computeVel(activity):
    "Given particle activity, output intrinsic swim speed"
    # This gives:
    # v_0 = Pe * sigma / tau_B = Pe * sigma / 3 * tau_R
    velocity = (activity * sigma) / (3 * (1/D_r))
    return velocity

def computeActiveForce(velocity):
    "Given particle activity, output repulsion well depth"
    # This is multiplied by Brownian time and gives:
    #          Pe = 3 * v_0 * tau_R / sigma
    # the conventional description of the Peclet number
    activeForce = velocity * threeEtaPiSigma
    return activeForce

def computeEps(alpha, activeForce):
    "Given particle activity, output repulsion well depth"
    # Here is where we will be testing the ratio we use (via alpha)
    epsilon = (alpha * activeForce * sigma / 24.0) + 1.0
    # Add 1 because of integer rounding
    epsilon = int(epsilon) + 1
    return epsilon
def avgCollisionForce(peNet):
    '''Computed from the integral of possible angles'''
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    return (magnitude * peNet) / (np.pi) 

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

#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)

tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size

# Get filenames for various file types

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle


matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5
            
def computeDist(x1, y1, x2, y2):
    '''Compute distance between two points'''
    return np.sqrt( ((x2-x1)**2) + ((y2 - y1)**2) )
    
def computeFLJ(r, x1, y1, x2, y2, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (x2 - x1) / r
    fy = f * (y2 - y1) / r
    return fx, fy

def computeTauPerTstep(epsilon, mindt=0.000001):
    '''Read in epsilon, output tauBrownian per timestep'''
#    if epsilon != 1.:
#        mindt=0.00001
    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

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

def findBins(lookN, currentInd, maxInds):
    '''Get the surrounding bin indices'''
    maxInds -= 1
    left = currentInd - lookN
    right = currentInd + lookN
    binsList = []
    for i in range(left, right):
        ind = i
        if i > maxInds:
            ind -= maxInds
        binsList.append(ind)
    return binsList
f = hoomd.open(name=inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep
            
outTxt = 'total_phase_dens_edge_width_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'count_A_edge'.center(15) + ' ' +\
        'count_B_edge'.center(15) + ' ' +\
        'count_A_gas'.center(15) + ' ' +\
        'count_B_gas'.center(15) + ' ' +\
        'count_A_bulk'.center(15) + ' ' +\
        'count_B_bulk'.center(15) + ' ' +\
        'edge_bulk_parts'.center(15) + ' ' +\
        'edge_gas_parts'.center(15) + ' ' +\
        'lat'.center(15) + ' ' +\
        'edge_area'.center(15) + ' ' +\
        'bulk_area'.center(15) + ' ' +\
        'gas_area'.center(15) + '\n')
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
    print(sizeBin)
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
        r=np.arange(1,h_box,1)
        print('j')
        print(j)
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
        min_size=int(partNum/3)
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
            gas_particle_range=2.0
            gas_r_lim=gas_particle_range*lat
            bulk_particle_range=1.0
            end_loop=0
            steps=0
            
            
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
                    steps+=1
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
                count_B_gas_final=count_B_gas
                steps+=1

                bulk_particle_range_final=bulk_particle_range
                bulk_r_lim_final=bulk_r_lim
                PhaseParts_final=PhaseParts2
            if steps>0:
                bulk_particle_range=bulk_particle_range_final-2

                search_range=int((bulk_particle_range+1)/sizeBin)
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if edgeBin[ix][iy]==1:
                            testIDs_area[ix][iy]=0
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
                                
                                    if testIDs_area[ix2][iy2] == 0:
                                        pass
                                    elif testIDs_area[ix2][iy2] == 1:
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
                                                testIDs_area[ix2][iy2]=0
                                            else:
                                                r_cut_new=np.abs(r_max-bulk_r_lim)
                                                if r_cut_new<=((sizeBin**2+sizeBin**2)**0.5)/2:
                                                    testIDs_area[ix2][iy2]=0
                                                else:
                                                    testIDs_area[ix2][iy2]=1
                                        else:
                                            testIDs_area[ix2][iy2]=1
                                    
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if testIDs_area[ix][iy] == 0:
                            for h in range(0,len(binParts[ix][iy])):
                                PhasePartsarea[binParts[ix][iy][h]]=0
                            continue
                    # Does the bin belong to the dense phase?

                        if testIDs_area[ix][iy]==1:
                            for h in range(0,len(binParts[ix][iy])):
                                PhasePartsarea[binParts[ix][iy][h]]=1
                            continue
                        for h in range(0,len(binParts[ix][iy])):
                            PhasePartsarea[binParts[ix][iy][h]]=2
                                
            
                edgeParts_num=np.where(PhaseParts2==0)
                bulkParts_num=np.where(PhaseParts2==1)
                gasParts_num=np.where(PhaseParts2==2)
                gas_type_area=typ[gasParts_num[0]]
                bulk_type_area=typ[bulkParts_num[0]]
                edge_type_area=typ[edgeParts_num[0]]

                edgeParts_area=edgeParts_num
                bulkParts_area=bulkParts_num
                gasParts_area=gasParts_num
                count_A_gas=len(np.where(gas_type==0)[0])
                count_B_gas=len(np.where(gas_type==1)[0])
                count_A_dense=len(np.where(bulk_type==0)[0])
                count_B_dense=len(np.where(bulk_type==1)[0])
                count_A_edge=len(np.where(edge_type==0)[0])
                count_B_edge=len(np.where(edge_type==1)[0]) 
                gas_pos_area=pos[gasParts_num[0]]
                bulk_pos_area=pos[bulkParts_num[0]]
                edge_pos_area=pos[edgeParts_num[0]]
                count_A_edge_area=count_A_edge
                count_B_edge_area=count_B_edge
                count_A_dense_area=count_A_dense
                count_B_dense_area=count_B_dense
                count_A_gas_area=count_A_gas
                count_B_gas_area=count_B_gas
                bulk_particle_range_area=bulk_particle_range
                bulk_r_lim_area=bulk_r_lim
                PhaseParts_area=PhaseParts2
                
                com_x=(query_points[0])
                com_y=(query_points[1]) 
                tmp_pos_edge=np.zeros((len(edge_pos_area),2))
                tmp_pos_bulk=np.zeros((len(bulk_pos),2))
            
                tmp_pos_edge[:,0] = edge_pos_area[:,0] - com_x
                tmp_pos_edge[:,1] = edge_pos_area[:,1] - com_y
                tmp_pos_bulk[:,0] = bulk_pos[:,0] - com_x
                tmp_pos_bulk[:,1] = bulk_pos[:,1] - com_y
                for x in range(0,len(tmp_pos_edge)):
                    if tmp_pos_edge[x,0]>h_box:
                        tmp_pos_edge[x,0]=tmp_pos_edge[x,0]-l_box
                    elif tmp_pos_edge[x,0]<(-h_box):
                        tmp_pos_edge[x,0]=tmp_pos_edge[x,0]+l_box
                    if tmp_pos_edge[x,1]>h_box:
                        tmp_pos_edge[x,1]=tmp_pos_edge[x,1]-l_box
                    elif tmp_pos_edge[x,1]<(-h_box):
                        tmp_pos_edge[x,1]=tmp_pos_edge[x,1]+l_box
                    
                for x in range(0,len(tmp_pos_bulk)):
                    if tmp_pos_bulk[x,0]>h_box:
                        tmp_pos_bulk[x,0]=tmp_pos_bulk[x,0]-l_box
                    elif tmp_pos_bulk[x,0]<(-h_box):
                        tmp_pos_bulk[x,0]=tmp_pos_bulk[x,0]+l_box
                    if tmp_pos_bulk[x,1]>h_box:
                        tmp_pos_bulk[x,1]=tmp_pos_bulk[x,1]-l_box
                    elif tmp_pos_bulk[x,1]<(-h_box):
                        tmp_pos_bulk[x,1]=tmp_pos_bulk[x,1]+l_box
                    
                xypos_tuple=tuple(map(tuple, tmp_pos_edge))
                sampleCircle=MultiPoint(points=xypos_tuple).buffer(2*lat)#Point(xypos_tuple).buffer(1)            
                edge_area=sampleCircle.area
                edge_dens=((count_A_edge_final+count_B_edge_final)/edge_area)
            #BLUE = '#6699cc'
            #fig = plt.figure() 
            #ax = fig.gca() 
            #ax.add_patch(PolygonPatch(sampleCircle, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
            #ax.axis('scaled')
            #plt.show()
            
                xypos_tuple=tuple(map(tuple, tmp_pos_bulk))
                sampleCircle=MultiPoint(points=xypos_tuple).buffer(1*lat)#Point(xypos_tuple).buffer(1)            
            
            
                bulk_area=sampleCircle.area
                bulk_dens=((count_A_dense_final+count_B_dense_final)/bulk_area)
            #BLUE = '#6699cc'
            #fig = plt.figure() 
            #ax = fig.gca() 
            #ax.add_patch(PolygonPatch(sampleCircle, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
            #ax.axis('scaled')
            #plt.show()
            
                gas_area=(l_box*l_box)-(bulk_area+edge_area)
                gas_dens=(count_A_gas_final+count_B_gas_final)/gas_area
            
                g = open(outPath+outTxt, 'a')
                g.write('{0:.1f}'.format(tst).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_edge_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_edge_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_gas_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_gas_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_A_dense_final).center(15) + ' ')
                g.write('{0:.6f}'.format(count_B_dense_final).center(15) + ' ')
                g.write('{0:.6f}'.format(bulk_particle_range_final).center(15) + ' ')
                g.write('{0:.6f}'.format(gas_particle_range).center(15) + ' ')
                g.write('{0:.6f}'.format(lat).center(15) + ' ')
                g.write('{0:.6f}'.format(edge_area).center(15) + ' ')
                g.write('{0:.6f}'.format(gas_area).center(15) + ' ')
                g.write('{0:.6f}'.format(bulk_area).center(15) + '\n')
                g.close()
            
