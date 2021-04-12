#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:50:02 2020

@author: nicklauersdorf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:41:30 2020

@author: nicklauersdorf
"""
from matplotlib.lines import Line2D
import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams["font.family"] = "Times New Roman"

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#

#Run on Cluster
hoomdPath='/nas/home/njlauers/hoomd-blue/build/'

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

#Cut off interaction radius (Per LJ Potential)
r_cut=2**(1/6)

# Run locally
#outPath='/Volumes/External/test_video/'

#Run on Cluster
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/lat_video/'

r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

#outPath='/Volumes/External/test_video/'

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
#inFile='pa300_pb300_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'cluster_pa400_pb350_phi60_eps0.1_xa0.8_align3_dtau1.0e-06.gsd'
#inFile='pa400_pb500_xa20_ep1.0_phi60_pNum100000.gsd'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/alignment_sparse/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')
# Inside and outside activity from command line

#Label simulation parameters
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac_orig = float(sys.argv[4])
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
eps = float(sys.argv[5])
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))
#Determine which activity is the slow activity or if system is monodisperse

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


# Define functions for calculating analytical predictions

def ljForce(r, eps, sigma=1.):
    # Calculates LJ force between 2 particles given by separation distance (r) 
    #and softness (eps) with diameter (sigma=1.0)
    div = (sigma/r)
    dU = (24. * eps / sigma) * ((2*(div**13)) - (div)**7)
    return dU
    
def avgCollisionForce(pe, power=1.):
    # Calculate average force experienced by a particle with its six nearest
    # neighbors within the bulk of the dense phase.  
    '''Computed from the integral of possible angles'''
    #Account for critical activity of onset of MIPS
    peCritical = 40.
    if pe < peCritical:
        pe = 0
    else:
        pe -= peCritical
    coeff = 1.92 #obtained from fitted experimental data
    return (pe * coeff)

def conForRClust(pe, eps):
    #Calculates the lattice spacing between a pair of particles given the
    # balance of compressive (avgCollisionForce) and repulsive (ljForce) forces
    out = []
    
    # cut-off radius of LJ potential
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out

# Calculates analtyical lattice spacing and LJ pressure
lat_theory = conForRClust(peNet, eps)

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
def quatToAngle(quat):
    "Take vector, output angle between [-pi, pi]"
    #print(quat)
    r = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    #print(2*math.atan2(x,r))
    rad = math.atan2(y, x)#(2*math.acos(r))#math.atan2(y, x)#
    
    return rad
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
'''             
outTxt = 'alignment_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'clust_size'.center(15) + ' ' +\
        'min_ang'.center(15) + ' ' +\
        'max_ang'.center(15) + ' ' +\
        'radius'.center(15) + ' ' +\
        'rad_bin'.center(15) + ' ' +\
        'num_dens'.center(15) + ' ' +\
        'align'.center(15) + '\n')
g.close()
'''  

file_name = os.path.basename(inFile)
outfile, file_extension = os.path.splitext(file_name)   # get base name
out = outfile + "_frame_"  

with hoomd.open(name=inFile, mode='rb') as t:
    
    #990                # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    start = 0#int((dumps-1)*(2/3))
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.
    radius=np.arange(0,h_box+3.0, 3.0)

    typ = snap.particles.typeid
    partNum = len(typ)
    # Set up cluster computation using box
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)

    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp(((l_box) / NBins), 6)
    #my_clust = cluster.Cluster()
    #c_props = cluster.ClusterProperties()
    # Compute each mesh
    
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
    align_avg=np.zeros(len(radius))
    align_num=0
    test_run=0
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
        test_run+=1
        if test_run>1:
            pos_prev = pos
            tst_prev = tst
        else:
            pass
        
        pos = snap.particles.position               # position
        #snap_future = t[j+1]
        #pos_future=snap_future.particles.position
        #print(snap.particles.velocity)
        #print(np.arctan(snap.particles.velocity[:,1], snap.particles.velocity[:,0]))
        pos[:,-1] = 0.0
        ori = snap.particles.orientation 
        #print(ori)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        #print(ang)
        #print(np.multiply(np.sin(ang/2),(snap.particles.velocity[:,0])/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        #print(np.multiply(np.sin(ang/2),snap.particles.velocity[:,1]/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        
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
            rad_bins=np.zeros(len(radius))
            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box

            com_x_ind = int(com_tmp_posX / sizeBin)
            com_y_ind = int(com_tmp_posY / sizeBin)
            # Loop through each timestep
            tot_count_gas=np.zeros((end,2))
            tot_count_dense=np.zeros((end,2))
            tot_count_edge=np.zeros((end,2))
            rad_count_gas=np.zeros((end,int(NBins/2)-1,2))
            rad_count_dense=np.zeros((end,int(NBins/2)-1,2))
            rad_count_edge=np.zeros((end,int(NBins/2)-1,2))
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            #posParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            
            
            
            #Assigns particle indices and types to bins
            '''
            for k in range(0, len(ids)):

                # Convert position to be > 0 to place in list mesh
                tmp_posX = pos[k][0] + h_box
                tmp_posY = pos[k][1] + h_box
                x_ind = int(tmp_posX / sizeBin)
                y_ind = int(tmp_posY / sizeBin)
                # Append all particles to appropriate bin
                binParts[x_ind][y_ind].append(k)
                typParts[x_ind][y_ind].append(typ[k])
                #posParts[x_ind][y_ind].append(pos[k])
                
                if clust_size[ids[k]] >= min_size:
                    occParts[x_ind][y_ind] = 1
                    
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
            #int_width = 20.0
            search_range = 14.0
            search_range2 = 7.0
            edge_id = np.where(PhaseParts2==0)[0]
            bulk_id = np.where(PhaseParts2==1)[0]
            gas_id = np.where(PhaseParts2==2)[0]
            
            #fig = plt.figure(figsize=(6, 6))
                #widths = [1.0,1.0]#, [1,1,1,1],[1,1,1,1]]#[1,1.205, 1.205, 1.205]
                #gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.2)
                #ax0 = fig.add_subplot(gs[0, 0])
            myEps = [1., 0.1, 0.01, 0.001, 0.0001]
            '''
            '''    
            for ix in range(0, len(binParts)):
                print(ix)
                for iy in range(0, len(binParts)):
                    if edgeBin[ix][iy]==1:
                        x_ref=(ix+0.5)*sizeBin
                        y_ref=(iy+0.5)*sizeBin
                        for ix2 in range(0, len(binParts)):
                            for iy2 in range(0, len(binParts)):
                                if testIDs[ix2][iy2]==1:
                                                    x_ref2=(ix2+0.5)*sizeBin
                                                    y_ref2=(iy2+0.5)*sizeBin
                                                    
                                                    dx1 = np.abs(x_ref2 - x_ref)
                                                    
                                                    difx_abs = np.abs(dx1)
                                                    if difx_abs>=h_box:
                                                        if dx1 < -h_box:
                                                            dx1 += l_box
                                                        else:
                                                            dx1 -= l_box
                                                            
                                                    dy1 = np.abs(y_ref2 - y_ref)
                                                    
                                                    difx_abs = np.abs(dy1)
                                                    if difx_abs>=h_box:
                                                        if dy1 < -h_box:
                                                            dy1 += l_box
                                                        else:
                                                            dy1 -= l_box

                                                    r_min=(dx1**2+dy1**2)**0.5
                                                    if r_min<=search_range:
                                                        
                                                        for j in range(0, len(binParts[ix2][iy2])):
                                                            PhaseParts2[binParts[ix2][iy2][j]]=0
                                elif testIDs[ix2][iy2]==2:
                                                    x_ref2=(ix2+0.5)*sizeBin
                                                    y_ref2=(iy2+0.5)*sizeBin
                                                    
                                                    dx1 = np.abs(x_ref2 - x_ref)
                                                    
                                                    difx_abs = np.abs(dx1)
                                                    if difx_abs>=h_box:
                                                        if dx1 < -h_box:
                                                            dx1 += l_box
                                                        else:
                                                            dx1 -= l_box
                                                            
                                                    dy1 = np.abs(y_ref2 - y_ref)
                                                    
                                                    difx_abs = np.abs(dy1)
                                                    if difx_abs>=h_box:
                                                        if dy1 < -h_box:
                                                            dy1 += l_box
                                                        else:
                                                            dy1 -= l_box

                                                    r_min=(dx1**2+dy1**2)**0.5
                                                    if r_min<=search_range2:
                                                        
                                                        for j in range(0, len(binParts[ix2][iy2])):
                                                            PhaseParts2[binParts[ix2][iy2][j]]=0
            '''                                            
                            
            #edge_id = np.where(PhaseParts2==0)[0]
            #bulk_id = np.where(PhaseParts2==1)[0]
            #gas_id = np.where(PhaseParts2==2)[0]
            
                #widths = [1.0,1.0]#, [1,1,1,1],[1,1,1,1]]#[1,1.205, 1.205, 1.205]
                #gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.2)
                #ax0 = fig.add_subplot(gs[0, 0])
            yellow = ("#fdfd96")
            green = ("#77dd77")
            red = ("#ff6961")
            NBins = getNBins(l_box, 2.0)
            sizeBin = roundUp(((l_box) / NBins), 6)
         
            rad_bins=np.zeros(len(radius))
            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box

            com_x_ind = int(com_tmp_posX / sizeBin)
            com_y_ind = int(com_tmp_posY / sizeBin)
            # Loop through each timestep
            tot_count_gas=np.zeros((end,2))
            tot_count_dense=np.zeros((end,2))
            tot_count_edge=np.zeros((end,2))
            rad_count_gas=np.zeros((end,int(NBins/2)-1,2))
            rad_count_dense=np.zeros((end,int(NBins/2)-1,2))
            rad_count_edge=np.zeros((end,int(NBins/2)-1,2))
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            #posParts=  [[[] for b in range(NBins)] for a in range(NBins)]
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
                #posParts[x_ind][y_ind].append(pos[k])
                
                if clust_size[ids[k]] >= min_size:
                    occParts[x_ind][y_ind] = 1
                    
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
                        
                                                
                                                
            align_x = [[0 for b in range(NBins)] for a in range(NBins)]
            align_y = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot = [[0 for b in range(NBins)] for a in range(NBins)]
            
            pos_box_x = [[0 for b in range(NBins)] for a in range(NBins)]
            pos_box_y = [[0 for b in range(NBins)] for a in range(NBins)]
            
            v_all_x = [[0 for b in range(NBins)] for a in range(NBins)]
            v_all_y = [[0 for b in range(NBins)] for a in range(NBins)]
            
            p_all_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_all_y = [[0 for b in range(NBins)] for a in range(NBins)]
            
            
            
            
            
            align_x_avg = [[0 for b in range(NBins)] for a in range(NBins)]
            align_y_avg = [[0 for b in range(NBins)] for a in range(NBins)]
            align_tot_avg = [[0 for b in range(NBins)] for a in range(NBins)]
            
            v_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            v_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]
            
            p_avg_x = [[0 for b in range(NBins)] for a in range(NBins)]
            p_avg_y = [[0 for b in range(NBins)] for a in range(NBins)]
            
            num_dens = [[0 for b in range(NBins)] for a in range(NBins)]
            
            
            pos_box_start=np.array([])
            for ix in range(0, len(occParts)):
                pos_box_start = np.append(pos_box_start, ix*sizeBin)
                for iy in range(0, len(occParts)):
                    pos_box_x[ix][iy] = ((ix+0.5)*sizeBin)
                    pos_box_y[ix][iy] = ((iy+0.5)*sizeBin)
                    '''
                    if len(binParts[ix][iy]) != 0:
                        for h in range(0, len(binParts[ix][iy])):
                            x_pos=pos[binParts[ix][iy]][h][0]+h_box
                                        
                            y_pos=pos[binParts[ix][iy]][h][1]+h_box
                                        
                            difx=x_pos-com_tmp_posX
                            difx_abs = np.abs(difx)
                            if difx_abs>=h_box:
                                if difx < -h_box:
                                    difx += l_box
                                else:
                                    difx -= l_box
                            dify=y_pos-com_tmp_posY
                            dify_abs = np.abs(dify)
                            if dify_abs>=h_box:
                                if dify < -h_box:
                                    dify += l_box
                                else:
                                    dify -= l_box
                            
                                        
                            difr=(difx**2+dify**2)**0.5
                            px = np.sin(ang[binParts[ix][iy][h]])
                            py = -np.cos(ang[binParts[ix][iy][h]])
                                                
                                                
                                                
                            r_dot_p = (-difx * px) + (-dify * py)
                            
                            p_all_x[ix][iy]+=px#*np.abs(difx)
                            p_all_y[ix][iy]+=py#*np.abs(dify)
                            
                            if test_run>1:
                                #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                dif_x = pos_future[binParts[ix][iy][h]][0]-pos[binParts[ix][iy][h]][0]

                                dif_x_abs = np.abs(dif_x)
                                if dif_x_abs>=h_box:
                                    if dif_x < -h_box:
                                        dif_x += l_box
                                    else:
                                        dif_x -= l_box
                                #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                dif_y = pos_future[binParts[ix][iy][h]][1]-pos[binParts[ix][iy][h]][1]

                                dif_y_abs = np.abs(dif_y)
                                if dif_y_abs>=h_box:
                                    if dif_y < -h_box:
                                        dif_y += l_box
                                    else:
                                        dif_y -= l_box
                                    
                                    
                                dif_t = tst_prev-tst
                                
                                v_all_x[ix][iy]+= dif_x/dif_t
                                v_all_y[ix][iy]+= dif_y/dif_t
                        num_dens[ix][iy] = (len(binParts[ix][iy])/(sizeBin**2))*(math.pi/4)
                        p_avg_x[ix][iy] = p_all_x[ix][iy]/len(binParts[ix][iy])
                        p_avg_y[ix][iy] = p_all_y[ix][iy]/len(binParts[ix][iy])

                        if test_run>1:
                            v_avg_x[ix][iy] = v_all_x[ix][iy]/len(binParts[ix][iy])
                            v_avg_y[ix][iy] = v_all_y[ix][iy]/len(binParts[ix][iy])
                    '''
            #pos_box_start=np.array([])
            
            
            min_dist = [[0 for b in range(NBins)] for a in range(NBins)]
            min_dist_vals = [[0 for b in range(NBins)] for a in range(NBins)]
            min_dist_avg_part= np.zeros((len(pos), 6))
            min_dist_part = np.zeros(len(pos))
            for ix in range(0, len(occParts)):
                print('ix')
                print(ix)
                #pos_box_start = np.append(pos_box_start, ix*sizeBin)
                for iy in range(0, len(occParts)):
                    print('iy')
                    print(iy)
                    if len(binParts[ix][iy]) != 0:
                        for h in range(0, len(binParts[ix][iy])):
                            part_ind=0
                            min_dist_val = 1000.
                            if ix ==0:
                                search_range_x = [len(occParts)-1, ix, ix+1]
                            elif ix == (len(binParts)-1):
                                search_range_x = [ix-1, ix, 0]
                            else:
                                search_range_x = [ix-1, ix, ix+1]
                            if iy == 0:
                                search_range_y = [len(occParts)-1, iy, iy+1]
                            elif iy == (len(binParts)-1):
                                search_range_y = [iy-1, iy, 0]
                            else:
                                search_range_y = [iy-1, iy, iy+1]
                            for ix2 in search_range_x:
                                for iy2 in search_range_y:
                                    if len(binParts[ix2][iy2]) != 0:
                                        for j in range(0, len(binParts[ix2][iy2])):
                                            if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
        
                                                dif_x_abs = np.abs(dif_x)
                                                if dif_x_abs>=h_box:
                                                    if dif_x < -h_box:
                                                        dif_x += l_box
                                                    else:
                                                        dif_x -= l_box
                                                #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
        
                                                dif_y_abs = np.abs(dif_y)
                                                if dif_y_abs>=h_box:
                                                    if dif_y < -h_box:
                                                        dif_y += l_box
                                                    else:
                                                        dif_y -= l_box
                                                difr=(dif_x**2+dif_y**2)**0.5
                                                if part_ind==6:
                                                    max_val = np.amax(min_dist_avg_part[binParts[ix][iy][h]])
                                                    if difr<max_val:
                                                        max_val_ind = np.where(min_dist_avg_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                        min_dist_avg_part[binParts[ix][iy][h]][max_val_ind] = difr
                                                else:
                                                    min_dist_avg_part[binParts[ix][iy][h]][part_ind]=difr
                                                    part_ind+=1
                                                
                            if  part_ind!= 6:
                                if ix ==0:
                                    search_range_x = [len(occParts)-2, len(occParts)-1, ix, ix+1, ix+2]
                                elif ix ==1:
                                    search_range_x = [len(occParts)-1, ix-1, ix, ix+1, ix+2]
                                elif ix == (len(binParts)-2):
                                    search_range_x = [ix-2, ix-1, ix, ix+1, 0]
                                elif ix == (len(binParts)-1):
                                    search_range_x = [ix-2, ix-1, ix, 0, 1]
                                else:
                                    search_range_x = [ix-2, ix-1, ix, ix+1, ix+2]
                                if iy ==0:
                                    search_range_y = [len(occParts)-2, len(occParts)-1, iy, iy+1, iy+2]
                                elif iy ==1:
                                    search_range_y = [len(occParts)-1, iy-1, iy, iy+1, iy+2]
                                elif iy == (len(binParts)-2):
                                    search_range_y = [iy-2, iy-1, iy, iy+1, 1]
                                elif iy == (len(binParts)-1):
                                    search_range_y = [iy-2, iy-1, iy, 0, 1]
                                else:
                                    search_range_y = [iy-2, iy-1, iy, iy+1, iy+2]
                                for ix2 in search_range_x:
                                    for iy2 in search_range_y:
                                        if len(binParts[ix2][iy2]) != 0:
                                            for j in range(0, len(binParts[ix2][iy2])):
                                                if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                    #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                    dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
            
                                                    dif_x_abs = np.abs(dif_x)
                                                    if dif_x_abs>=h_box:
                                                        if dif_x < -h_box:
                                                            dif_x += l_box
                                                        else:
                                                            dif_x -= l_box
                                                    #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                    dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
            
                                                    dif_y_abs = np.abs(dif_y)
                                                    if dif_y_abs>=h_box:
                                                        if dif_y < -h_box:
                                                            dif_y += l_box
                                                        else:
                                                            dif_y -= l_box
                                                    difr=(dif_x**2+dif_y**2)**0.5
                                                    
                                                    if part_ind==6:
                                                        max_val = np.amax(min_dist_avg_part[binParts[ix][iy][h]])
                                                        if difr<max_val:
                                                            max_val_ind = np.where(min_dist_avg_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                            min_dist_avg_part[binParts[ix][iy][h]][max_val_ind] = difr
                                                    else:
                                                        
                                                        min_dist_avg_part[binParts[ix][iy][h]][part_ind]=difr
                                                        part_ind+=1
                                                    
                                if  part_ind!= 6:
                                    if ix ==0:
                                        search_range_x = [len(occParts)-3, len(occParts)-2, len(occParts)-1, ix, ix+1, ix+2, ix+3]
                                    elif ix ==1:
                                        search_range_x = [len(occParts)-2, len(occParts)-1, ix-1, ix, ix+1, ix+2, ix+3]
                                    elif ix ==2:
                                        search_range_x = [len(occParts)-1, ix-2, ix-1, ix, ix+1, ix+2, ix+3]
                                    elif ix == (len(binParts)-3):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, ix+2, 0]
                                    elif ix == (len(binParts)-2):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, 0, 1]
                                    elif ix == (len(binParts)-1):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, 0, 1, 2]
                                    else:
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, ix+2, ix+3]
                                    if iy ==0:
                                        search_range_y = [len(occParts)-3, len(occParts)-2, len(occParts)-1, iy, iy+1, iy+2, iy+3]
                                    elif iy ==1:
                                        search_range_y = [len(occParts)-2, len(occParts)-1, iy-1, iy, iy+1, iy+2, iy+3]
                                    elif iy ==2:
                                        search_range_y = [len(occParts)-1, iy-2, iy-1, iy, iy+1, iy+2, iy+3]
                                    elif iy == (len(binParts)-3):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, iy+2, 0]
                                    elif iy == (len(binParts)-2):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, 0, 1]
                                    elif iy == (len(binParts)-1):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, 0, 1, 2]
                                    else:
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, iy+2, iy+3]
                                    for ix2 in search_range_x:
                                        for iy2 in search_range_y:
                                            if len(binParts[ix2][iy2]) != 0:
                                                for j in range(0, len(binParts[ix2][iy2])):
                                                    if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                        #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                        dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
                
                                                        dif_x_abs = np.abs(dif_x)
                                                        if dif_x_abs>=h_box:
                                                            if dif_x < -h_box:
                                                                dif_x += l_box
                                                            else:
                                                                dif_x -= l_box
                                                        #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                        dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
                
                                                        dif_y_abs = np.abs(dif_y)
                                                        if dif_y_abs>=h_box:
                                                            if dif_y < -h_box:
                                                                dif_y += l_box
                                                            else:
                                                                dif_y -= l_box
                                                        difr=(dif_x**2+dif_y**2)**0.5
                                                        if part_ind==6:
                                                            max_val = np.amax(min_dist_avg_part[binParts[ix][iy][h]])
                                                            if difr<max_val:
                                                                max_val_ind = np.where(min_dist_avg_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                                min_dist_avg_part[binParts[ix][iy][h]][max_val_ind] = difr
                                                        else:
                                                            
                                                            min_dist_avg_part[binParts[ix][iy][h]][part_ind]=difr
                                                            part_ind+=1
                                    
                                    
                                        
            '''
            for ix in range(0, len(occParts)):
                print('ix')
                print(ix)
                #pos_box_start = np.append(pos_box_start, ix*sizeBin)
                for iy in range(0, len(occParts)):
                    print('iy')
                    print(iy)
                    if len(binParts[ix][iy]) != 0:
                        for h in range(0, len(binParts[ix][iy])):
                            min_dist_val = 1000.
                            if ix ==0:
                                search_range_x = [len(occParts)-1, ix, ix+1]
                            elif ix == (len(binParts)-1):
                                search_range_x = [ix-1, ix, 0]
                            else:
                                search_range_x = [ix-1, ix, ix+1]
                            if iy == 0:
                                search_range_y = [len(occParts)-1, iy, iy+1]
                            elif iy == (len(binParts)-1):
                                search_range_y = [iy-1, iy, 0]
                            else:
                                search_range_y = [iy-1, iy, iy+1]
                            print(search_range_x)
                            print(search_range_y)
                            for ix2 in search_range_x:
                                for iy2 in search_range_y:
                                    if len(binParts[ix2][iy2]) != 0:
                                        for j in range(0, len(binParts[ix2][iy2])):
                                            if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
        
                                                dif_x_abs = np.abs(dif_x)
                                                if dif_x_abs>=h_box:
                                                    if dif_x < -h_box:
                                                        dif_x += l_box
                                                    else:
                                                        dif_x -= l_box
                                                #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
        
                                                dif_y_abs = np.abs(dif_y)
                                                if dif_y_abs>=h_box:
                                                    if dif_y < -h_box:
                                                        dif_y += l_box
                                                    else:
                                                        dif_y -= l_box
                                                difr=(dif_x**2+dif_y**2)**0.5
                                                if len(min_dist_avg_part[binParts[ix][iy][h]])==6:
                                                    max_val = np.maximum(min_dist_part[binParts[ix][iy][h]])
                                                    if difr<max_val:
                                                        max_val_ind = np.where(min_dist_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                        min_dist_avg_part[binParts[ix][iy][h][max_val_ind]] = difr
                                                else:
                                                    min_dist_avg_part[binParts[ix][iy][h]]=np.append(min_dist_avg_part[binParts[ix][iy][h]], difr)
                                                
                                                if 0.0<difr<min_dist_val:
                                                    min_dist_val = difr
                                                    min_dist_part[binParts[ix][iy][h]]=min_dist_val
                            if min_dist_val != 1000.0:
                                min_dist[ix][iy]+=min_dist_val
                                
                                min_dist_vals[ix][iy]+=1
                            else:
                                if ix ==0:
                                    search_range_x = [len(occParts)-2, len(occParts)-1, ix, ix+1, ix+2]
                                elif ix ==1:
                                    search_range_x = [len(occParts)-1, ix-1, ix, ix+1, ix+2]
                                elif ix == (len(binParts)-2):
                                    search_range_x = [ix-2, ix-1, ix, ix+1, 0]
                                elif ix == (len(binParts)-1):
                                    search_range_x = [ix-2, ix-1, ix, 0, 1]
                                else:
                                    search_range_x = [ix-2, ix-1, ix, ix+1, ix+2]
                                if iy ==0:
                                    search_range_y = [len(occParts)-2, len(occParts)-1, iy, iy+1, iy+2]
                                elif iy ==1:
                                    search_range_y = [len(occParts)-1, iy-1, iy, iy+1, iy+2]
                                elif iy == (len(binParts)-2):
                                    search_range_y = [iy-2, iy-1, iy, iy+1, 1]
                                elif iy == (len(binParts)-1):
                                    search_range_y = [iy-2, iy-1, iy, 0, 1]
                                else:
                                    search_range_y = [iy-2, iy-1, iy, iy+1, iy+2]
                                for ix2 in search_range_x:
                                    for iy2 in search_range_y:
                                        if len(binParts[ix2][iy2]) != 0:
                                            for j in range(0, len(binParts[ix2][iy2])):
                                                if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                    #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                    dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
            
                                                    dif_x_abs = np.abs(dif_x)
                                                    if dif_x_abs>=h_box:
                                                        if dif_x < -h_box:
                                                            dif_x += l_box
                                                        else:
                                                            dif_x -= l_box
                                                    #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                    dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
            
                                                    dif_y_abs = np.abs(dif_y)
                                                    if dif_y_abs>=h_box:
                                                        if dif_y < -h_box:
                                                            dif_y += l_box
                                                        else:
                                                            dif_y -= l_box
                                                    difr=(dif_x**2+dif_y**2)**0.5
                                                    
                                                    if len(min_dist_avg_part[binParts[ix][iy][h]])==6:
                                                        max_val = np.maximum(min_dist_part[binParts[ix][iy][h]])
                                                        if difr<max_val:
                                                            max_val_ind = np.where(min_dist_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                            min_dist_avg_part[binParts[ix][iy][h][max_val_ind]] = difr
                                                    else:
                                                        min_dist_avg_part[binParts[ix][iy][h]]=np.append(min_dist_avg_part[binParts[ix][iy][h]], difr)
                                                    
                                                    if 0.0<difr<min_dist_val:
                                                        min_dist_val = difr
                                                        min_dist_part[binParts[ix][iy][h]]=min_dist_val
                                if min_dist_val != 1000.0:
                                    min_dist[ix][iy]+=min_dist_val
                                    min_dist_vals[ix][iy]+=1
                                else:
                                    if ix ==0:
                                        search_range_x = [len(occParts)-3, len(occParts)-2, len(occParts)-1, ix, ix+1, ix+2, ix+3]
                                    elif ix ==1:
                                        search_range_x = [len(occParts)-2, len(occParts)-1, ix-1, ix, ix+1, ix+2, ix+3]
                                    elif ix ==2:
                                        search_range_x = [len(occParts)-1, ix-2, ix-1, ix, ix+1, ix+2, ix+3]
                                    elif ix == (len(binParts)-3):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, ix+2, 0]
                                    elif ix == (len(binParts)-2):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, 0, 1]
                                    elif ix == (len(binParts)-1):
                                        search_range_x = [ix-3, ix-2, ix-1, ix, 0, 1, 2]
                                    else:
                                        search_range_x = [ix-3, ix-2, ix-1, ix, ix+1, ix+2, ix+3]
                                    if iy ==0:
                                        search_range_y = [len(occParts)-3, len(occParts)-2, len(occParts)-1, iy, iy+1, iy+2, iy+3]
                                    elif iy ==1:
                                        search_range_y = [len(occParts)-2, len(occParts)-1, iy-1, iy, iy+1, iy+2, iy+3]
                                    elif iy ==2:
                                        search_range_y = [len(occParts)-1, iy-2, iy-1, iy, iy+1, iy+2, iy+3]
                                    elif iy == (len(binParts)-3):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, iy+2, 0]
                                    elif iy == (len(binParts)-2):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, 0, 1]
                                    elif iy == (len(binParts)-1):
                                        search_range_y = [iy-3, iy-2, iy-1, iy, 0, 1, 2]
                                    else:
                                        search_range_y = [iy-3, iy-2, iy-1, iy, iy+1, iy+2, iy+3]
                                    for ix2 in search_range_x:
                                        for iy2 in search_range_y:
                                            if len(binParts[ix2][iy2]) != 0:
                                                for j in range(0, len(binParts[ix2][iy2])):
                                                    if binParts[ix][iy][h]!=binParts[ix2][iy2][j]:
                                                        #dif_x = pos[binParts[ix][iy][h]][0]-pos_prev[binParts[ix][iy][h]][0]
                                                        dif_x = pos[binParts[ix][iy][h]][0]-pos[binParts[ix2][iy2][j]][0]
                
                                                        dif_x_abs = np.abs(dif_x)
                                                        if dif_x_abs>=h_box:
                                                            if dif_x < -h_box:
                                                                dif_x += l_box
                                                            else:
                                                                dif_x -= l_box
                                                        #dif_y = pos[binParts[ix][iy][h]][1]-pos_prev[binParts[ix][iy][h]][1]
                                                        dif_y = pos[binParts[ix][iy][h]][1]-pos[binParts[ix2][iy2][j]][1]
                
                                                        dif_y_abs = np.abs(dif_y)
                                                        if dif_y_abs>=h_box:
                                                            if dif_y < -h_box:
                                                                dif_y += l_box
                                                            else:
                                                                dif_y -= l_box
                                                        difr=(dif_x**2+dif_y**2)**0.5
                                                        if len(min_dist_avg_part[binParts[ix][iy][h]])==6:
                                                            max_val = np.maximum(min_dist_part[binParts[ix][iy][h]])
                                                            if difr<max_val:
                                                                max_val_ind = np.where(min_dist_part[binParts[ix][iy][h]]==max_val)[0][0]
                                                                min_dist_avg_part[binParts[ix][iy][h][max_val_ind]] = difr
                                                        else:
                                                            min_dist_avg_part[binParts[ix][iy][h]]=np.append(min_dist_avg_part[binParts[ix][iy][h]], difr)
                                                    
                                                        if 0.0<difr<min_dist_val:
                                                            min_dist_val = difr
                                                            min_dist_part[binParts[ix][iy][h]]=min_dist_val
                                    if min_dist_val != 1000.0:
                                        min_dist[ix][iy]+=min_dist_val
                                        min_dist_vals[ix][iy]+=1
            '''
            
            min_dist_avg_part_tot= np.zeros(len(pos))
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    for h in range(0, len(binParts[ix][iy])):
                        min_dist_avg_part_tot[binParts[ix][iy][h]] =np.sum(min_dist_avg_part[binParts[ix][iy][h]])/6
            min_dist_avg_part_bin = [[0 for b in range(NBins)] for a in range(NBins)]
            min_dist_avg_part_bin_num = [[0 for b in range(NBins)] for a in range(NBins)]
            
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if len(binParts[ix][iy])>0:
                        for h in range(0, len(binParts[ix][iy])):
                            min_dist_avg_part_bin[ix][iy] +=min_dist_avg_part_tot[binParts[ix][iy][h]]
                            min_dist_avg_part_bin_num[ix][iy]+=1
                        min_dist_avg_part_bin[ix][iy]=min_dist_avg_part_bin[ix][iy]/min_dist_avg_part_bin_num[ix][iy]
                    else:
                        min_dist_avg_part_bin[ix][iy]=6.0
                      
            pad = str(j).zfill(4)
                   
            vmax_p = lat_theory*3.0
            vmin_p = lat_theory/1.25
            
            fig = plt.figure(figsize=(8,6))
            
            im = plt.contourf(pos_box_x, pos_box_y, min_dist_avg_part_bin, 600, vmin=vmin_p, vmax=vmax_p, cmap='ocean')
            norm= matplotlib.colors.Normalize(vmin=vmin_p, vmax=vmax_p)
            #norm= matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
            #im.cvalues.max()
            sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
            sm.set_array(min_dist_avg_part_bin)
            tick_lev = np.arange(vmin_p, vmax_p+vmax_p/10, (vmax_p-vmin_p)/10)
            clb = fig.colorbar(sm, ticks=tick_lev)#ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ax=ax2)
            #im.levels
            
            clb.ax.set_title(r'$|r_i|$', fontsize=15)
            #fig.colorbar(im, ax=ax2)


            plt.xlim(0, l_box)
            plt.ylim(0, l_box)
            
            plt.text(270.0, 20., s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #ax2.axis('off')
            plt.axis('off')
            plt.title(r'$\mathrm{Pe}$' + ' = ' + str(int(peA)) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))), fontsize=17)
            plt.tight_layout()
            plt.savefig(outPath + 'lat_wide_' + out + pad + ".png", dpi=200)
            plt.close()
            
            vmax_p = lat_theory*1.25  
            
            fig = plt.figure(figsize=(8,6))
            
            im = plt.contourf(pos_box_x, pos_box_y, min_dist_avg_part_bin, 600, vmin=vmin_p, vmax=vmax_p, cmap='ocean')
            norm= matplotlib.colors.Normalize(vmin=vmin_p, vmax=vmax_p)
            #norm= matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
            #im.cvalues.max()
            sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
            sm.set_array(min_dist_avg_part_bin)
            tick_lev = np.arange(vmin_p, vmax_p+vmax_p/10, (vmax_p-vmin_p)/10)
            clb = fig.colorbar(sm, ticks=tick_lev)#ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ax=ax2)
            #im.levels
            
            clb.ax.set_title(r'$|r_i|$', fontsize=15)
            #fig.colorbar(im, ax=ax2)


            plt.xlim(0, l_box)
            plt.ylim(0, l_box)
            
            plt.text(270.0, 20., s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #ax2.axis('off')
            plt.axis('off')
            plt.title(r'$\mathrm{Pe}$' + ' = ' + str(int(peA)) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))), fontsize=17)
            plt.tight_layout()
            plt.savefig(outPath + 'lat_thin_' + out + pad + ".png", dpi=200)
            plt.close()