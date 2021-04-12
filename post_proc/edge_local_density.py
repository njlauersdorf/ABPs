import sys
import os
import gsd
from gsd import hoomd
import freud
import numpy as np
import math
import time
# Run locally
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'##'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
imgPath='/pine/scr/n/j/njlauers/scm_tmpdir/edge_local_density6/'
r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])
if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
#FilePath='/Volumes/External/TestRun/'
#inFile = 'pa150_pb300_xa50_ep1_phi60_pNum10000.gsd'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/edge_dens/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')
# Inside and outside activity from command line
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac = float(sys.argv[4])
eps = float(sys.argv[5])

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

def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ

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

outTxt2 = 'Edge_local_dens_' + outF + '.txt'
g = open(imgPath+outTxt2, 'w+') # write file headings
g.write('tauB'.center(30) + ' ' +\
        'search_radius'.center(30) + ' ' +\
        'density_mean'.center(30) + ' ' +\
        'density_std_dev'.center(30) + ' ' +\
        'edge_count'.center(30) + '\n')
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
    edge_density=freud.density.LocalDensity(r_max=3, diameter=0)
    for j in range(start, int(end)):
        r=np.arange(1,h_box,1)
        j=j
        print('j',flush=True)
        print(j,flush=True)
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
            #print('cluster!',flush=True)
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

            #Blurs edge bins
            blurBin = [[0 for b in range(NBins)] for a in range(NBins)]
            Nedges = 0
            Phaseparts=np.zeros(len(pos))
            # Blur the edge bins a bit
            
                    
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    Nedges += edgeBin[ix][iy]
                    if edgeBin[ix][iy] == 1:
                        # If at right edge, wrap to left
                        if (ix + 1) != NBins:
                            lookx = [ix-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, 0]
                        # If at top edge, wrap to bottom
                        if (iy + 1) != NBins:
                            looky = [iy-1, iy, iy+1]
                        else:
                            looky = [iy-1, iy, 0]
                        # Loop through surrounding x-index
                        for indx in lookx:
                            # Loop through surrounding y-index
                            for indy in looky:
                                # Make all surrounding bins 'edge' bins
                                blurBin[indx][indy] = 1
            com_x=query_points[0]
            com_y=query_points[1]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if edgeBin[ix][iy]==1:  
                        if len(binParts[ix][iy])>0:
                            for i in range(0,len(binParts[ix][iy])):
                                Phaseparts[binParts[ix][iy][i]]=1
                    Nedges += edgeBin[ix][iy]
                    if edgeBin[ix][iy] == 1:
                        # If at right edge, wrap to left
                        if (ix + 1) != NBins:
                            lookx = [ix-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, 0]
                        # If at top edge, wrap to bottom
                        if (iy + 1) != NBins:
                            looky = [iy-1, iy, iy+1]
                        else:
                            looky = [iy-1, iy, 0]
                        # Loop through surrounding x-index
                        for indx in lookx:
                            # Loop through surrounding y-index
                            for indy in looky:
                                # Make all surrounding bins 'edge' bins
                                blurBin[indx][indy] = 1
            # Now let's get the area of each phase (by summing bin areas)
            gasBins = 0
            bulkBins = 0
            edgeBins=0
            edgeBinsbig = 0
            bulkBinsbig = 0
            testIDs = [[0 for b in range(NBins)] for a in range(NBins)]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    
                    # Is the bin an edge?
                    
                    if blurBin[ix][iy] == 1:
                        if len(binParts[ix][iy])>0:
                            if ids[binParts[ix][iy][0]]==lcID:
                                edgeBinsbig+=1
                        testIDs[ix][iy] = 0
                        edgeBins+=1
                        continue
                    # Does the bin belong to the dense phase?

                    if len(binParts[ix][iy]) != 0:
                        if ids[binParts[ix][iy][0]]==lcID:
                            bulkBinsbig+=1
                        if clust_size[ids[binParts[ix][iy][0]]] >=min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    
            #Checks for which bins are part of largest cluster
            check_clust=[[0 for b in range(NBins)] for a in range(NBins)]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if len(binParts[ix][iy])>0:
                        if ids[binParts[ix][iy][0]] ==lcID:
                            check_clust[ix][iy]=1
                    
            #tot_count_edge[j,0]=count_A_edge
            #tot_count_edge[j,1]=count_B_edge
            #tot_count_dense[j,0]=count_A_dense
            #tot_count_dense[j,1]=count_B_dense
            #tot_count_gas[j,0]=count_A_gas
            #tot_count_gas[j,1]=count_B_gas
            edge_loc=np.array([])
            #print(edgeBinsbig)
            edge_bins_largest=0
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if blurBin[ix][iy]==1:
                        if len(binParts[ix][iy])>0:
                            for h in range(0,len(binParts[ix][iy])):
                                if ids[binParts[ix][iy][h]]==lcID:
                                    edge_bins_largest+=1
            edge_loc=np.zeros((edge_bins_largest,3))
            f=0
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if blurBin[ix][iy]==1:
                        if len(binParts[ix][iy])>0:
                            for h in range(0,len(binParts[ix][iy])):
                                if ids[binParts[ix][iy][h]]==lcID:
                                    edge_loc[f,:]=pos[binParts[ix][iy][h]]
                                    f+=1
                                    
                                    
            edge_density.compute(system=system_all, query_points=f_box.wrap(edge_loc))
            edge_density_final=edge_density.density
            mean=np.mean(edge_density_final)
            test=0
            val=0
            for i in range(0,len(edge_density_final)):
                test+=(edge_density_final[i]-mean)**2
                val+=1
            print(mean)
            standard_dev=((test/(val-1))**0.5)
            edge_particles=val
            #print(np.shape(edge_density_final))
            #print(type(edge_density))
            #print(dir(edge_density))
            #print(edge_density.r_max)
            g = open(imgPath+outTxt2, 'a')
            #for k in range(0, len(edge_density_final)):
            #    g.write('{0:.2f}'.format(tst).center(30) + ' ')
            #    g.write('{0:.1f}'.format(edge_density.r_max).center(30) + ' ')
            #    g.write('{0:.6f}'.format(edge_density_final[k]).center(30) + '\n')
            g.write('{0:.2f}'.format(tst).center(30) + ' ')
            g.write('{0:.1f}'.format(edge_density.r_max).center(30) + ' ')
            g.write('{0:.6f}'.format(mean).center(30) + ' ')
            g.write('{0:.6f}'.format(standard_dev).center(30) + ' ')
            g.write('{0:.6f}'.format(edge_particles).center(30) + '\n')            
            g.close()
    
