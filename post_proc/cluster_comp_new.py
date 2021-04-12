import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
import time
# Run locally
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
imgPath='/pine/scr/n/j/njlauers/scm_tmpdir/cluster_comp/'
r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
inFile = 'pa150_pb300_xa50_ep1_phi60_pNum10000.gsd'
outPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=gsdPath+inFile, mode='rb')
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
dt = 0.000001 * tauLJ                        # timestep size

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

f = hoomd.open(name=gsdPath+inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

outTxt = 'edge_comp_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(30) + ' ' +\
        'gas A particles'.center(30) + ' ' +\
        'gas B particles'.center(30) + ' ' +\
        'edge A particles'.center(30) + ' ' +\
        'edge B particles'.center(30) + ' ' +\
        'bulk A particles'.center(30) + ' ' +\
        'bulk B particles'.center(30) + ' '+\
        'gas area'.center(30) + ' ' +\
        'edge area'.center(30) + ' ' +\
        'bulk area'.center(30) + '\n')
g.close()

outTxt2 = 'Radial_dens_' + outF + '.txt'
g = open(outPath+outTxt2, 'w+') # write file headings
g.write('tauB'.center(30) + ' ' +\
        'search radius'.center(30) + ' ' +\
        'gas A density'.center(30) + ' ' +\
        'gas B density'.center(30) + ' ' +\
        'edge A density'.center(30) + ' ' +\
        'edge B density'.center(30) + ' ' +\
        'bulk A density'.center(30) + ' ' +\
        'bulk B density'.center(30) + ' '+\
        'total A density'.center(30) + ' ' +\
        'total B density'.center(30) + '\n')
g.close()

with hoomd.open(name=gsdPath+inFile, mode='rb') as t:
    
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
    clust_edge_arr=np.zeros(end)
    
    average_clust_radius=0
    count_of_rads=0
    gas_area_arr = np.zeros(end)
    dense_area_arr = np.zeros(end)
    edge_area_arr = np.zeros(end)
    dense_area_big_arr = np.zeros(end)
    edge_area_big_arr = np.zeros(end)
    count_avg=0
    for j in range(start, int(end/100)):
        r=np.arange(1,h_box,1)
        j=j*100
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
        min_size=int(partNum/20)
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

            #Blurs edge bins
            blurBin = [[0 for b in range(NBins)] for a in range(NBins)]
            Nedges = 0

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
                    
            # Area of a bin
            binArea = sizeBin * sizeBin
            # Area of each phase
            gasArea = binArea * gasBins
            bulkArea = binArea * bulkBins
            edgeArea = binArea * edgeBins
            bulkAreabig= binArea * bulkBinsbig
            edgeAreabig= binArea * edgeBinsbig

            dense_area_arr[j] = bulkArea
            gas_area_arr[j] = gasArea
            edge_area_arr[j] = edgeArea
            dense_area_big_arr[j]=bulkAreabig
            edge_area_big_arr[j] = edgeAreabig
            
            count_A_edge=0
            count_B_edge=0
            count_A_dense=0
            count_B_dense=0
            count_A_edge_big=0
            count_B_edge_big=0
            count_A_dense_big=0
            count_B_dense_big=0
            count_A_gas=0
            count_B_gas=0
            
            #Checks for which bins are part of largest cluster
            check_clust=[[0 for b in range(NBins)] for a in range(NBins)]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if len(binParts[ix][iy])>0:
                        if ids[binParts[ix][iy][0]] ==lcID:
                            check_clust[ix][iy]=1
            
            #counts number of particles of either edge, bulk, or gas of both type A and B total over all bins
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if check_clust[ix][iy]==1:
                        if testIDs[ix][iy]==0:
                            if len(typParts[ix][iy])>0:
                                for h in range(0,len(typParts[ix][iy])):
                                    if typParts[ix][iy][h]==0:
                                        count_A_edge_big+=1
                                    elif typParts[ix][iy][h]==1:
                                        count_B_edge_big+=1
                        elif testIDs[ix][iy]==1:
                            if len(typParts[ix][iy])>0:
                                for h in range(0,len(typParts[ix][iy])):
                                    if typParts[ix][iy][h]==0:
                                        count_A_dense_big+=1
                                    elif typParts[ix][iy][h]==1:
                                        count_B_dense_big+=1
                    if testIDs[ix][iy]==0:
                            if len(typParts[ix][iy])>0:
                                for h in range(0,len(typParts[ix][iy])):
                                    if typParts[ix][iy][h]==0:
                                        count_A_edge+=1
                                    elif typParts[ix][iy][h]==1:
                                        count_B_edge+=1
                    elif testIDs[ix][iy]==1:
                            if len(typParts[ix][iy])>0:
                                for h in range(0,len(typParts[ix][iy])):
                                    if typParts[ix][iy][h]==0:
                                        count_A_dense+=1
                                    elif typParts[ix][iy][h]==1:
                                        count_B_dense+=1
                    elif testIDs[ix][iy]==2:
                        if len(typParts[ix][iy])>0:
                            for h in range(0,len(typParts[ix][iy])):
                                if typParts[ix][iy][h]==0:
                                    count_A_gas+=1
                                elif typParts[ix][iy][h]==1:
                                    count_B_gas+=1
                    else:
                        pass
                    
            #tot_count_edge[j,0]=count_A_edge
            #tot_count_edge[j,1]=count_B_edge
            #tot_count_dense[j,0]=count_A_dense
            #tot_count_dense[j,1]=count_B_dense
            #tot_count_gas[j,0]=count_A_gas
            #tot_count_gas[j,1]=count_B_gas
            
            rad=np.arange(1,int(NBins/2),1)

            tot_count_A_edge=np.zeros(len(r))
            tot_count_B_edge=np.zeros(len(r))
            tot_count_A_dense=np.zeros(len(r))
            tot_count_B_dense=np.zeros(len(r))
            tot_count_A_gas=np.zeros(len(r))
            tot_count_B_gas=np.zeros(len(r))
            tot_count_A=np.zeros(len(r))
            tot_count_B=np.zeros(len(r))
            
            rad_look_prev=0
            x_arr=np.array([])
            y_arr=np.array([])
            edge_bins=0
            edge_dist = 0
            com_x=(query_points[0]+h_box)
            com_y=(query_points[1]+h_box) 
            '''
            #Calculates average edge distance from CoM
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if check_clust[ix][iy]==1:
                        if testIDs[ix][iy]==0:
                            bin_x_loc_min=(ix)*sizeBin
                            bin_x_loc_max=(ix+1)*sizeBin
                            bin_y_loc_min = (iy)*sizeBin
                            bin_y_loc_max = (iy+1)*sizeBin
                            dif_loc_x_min = np.abs(bin_x_loc_max-com_x)
                            dif_loc_x_max = np.abs(bin_x_loc_min-com_x)
                            dif_loc_y_min = np.abs(bin_y_loc_max-com_y)
                            dif_loc_y_max = np.abs(bin_y_loc_min-com_y)
                            if dif_loc_x_max>h_box:
                                dif_loc_x_max=dif_loc_x_max-l_box
                            if dif_loc_x_min>h_box:
                                dif_loc_x_min=dif_loc_x_min-l_box
                            if dif_loc_y_max>h_box:
                                dif_loc_y_max=dif_loc_y_max-l_box
                            if dif_loc_y_min>h_box:
                                dif_loc_y_min=dif_loc_y_min-l_box
                            dif_loc_x=(dif_loc_x_min+dif_loc_x_max)/2
                            dif_loc_y=(dif_loc_y_min+dif_loc_y_max)/2
                            dif_loc=(dif_loc_x**2+dif_loc_y**2)**0.5
                            edge_dist+=dif_loc
                            edge_bins+=1
            
            average_clust_radius+=(edge_dist/edge_bins)
            count_of_rads+=1
            '''
            for i in range(0,len(r)):
                
                rad_look = r[i]
                print(rad_look)
                count_A_edge2 = 0
                count_B_edge2 = 0
                count_A_dense2 = 0
                count_B_dense2 = 0
                count_A_gas2 = 0
                count_B_gas2 = 0
                
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        bin_x_loc_min=(ix)*sizeBin
                        bin_x_loc_max=(ix+1)*sizeBin
                        bin_y_loc_min = (iy)*sizeBin
                        bin_y_loc_max = (iy+1)*sizeBin
                        dif_loc_x_min = np.abs(bin_x_loc_max-com_x)
                        dif_loc_x_max = np.abs(bin_x_loc_min-com_x)
                        dif_loc_y_min = np.abs(bin_y_loc_max-com_y)
                        dif_loc_y_max = np.abs(bin_y_loc_min-com_y)
                        if dif_loc_x_max>h_box:
                            dif_loc_x_max=dif_loc_x_max-l_box
                        if dif_loc_x_min>h_box:
                            dif_loc_x_min=dif_loc_x_min-l_box
                        if dif_loc_y_max>h_box:
                            dif_loc_y_max=dif_loc_y_max-l_box
                        if dif_loc_y_min>h_box:
                            dif_loc_y_min=dif_loc_y_min-l_box
                        
                        dif_loc_corner_1=(dif_loc_x_min**2+dif_loc_y_min**2)**0.5
                        dif_loc_corner_2=(dif_loc_x_max**2+dif_loc_y_min**2)**0.5
                        dif_loc_corner_3=(dif_loc_x_min**2+dif_loc_y_max**2)**0.5
                        dif_loc_corner_4=(dif_loc_x_max**2+dif_loc_y_max**2)**0.5
                        dif_locs=np.array([dif_loc_corner_1,dif_loc_corner_2,dif_loc_corner_3,dif_loc_corner_4])
                        
                        dif_loc_min=np.min(dif_locs)
                        dif_loc_max=np.max(dif_locs)
                        
                        if (dif_loc_min)<=rad_look<=dif_loc_max or (dif_loc_min)<=(rad_look-1)<=dif_loc_max:
                            #if check_clust[ix][iy]==1:
                            if testIDs[ix][iy]==0:
                                if len(binParts[ix][iy])>0:
                                        for h in range(0,len(binParts[ix][iy])):
                                            part_pos=pos[binParts[ix][iy][h]]
                                            part_pos_x=part_pos[0]+h_box
                                            part_pos_y=part_pos[1]+h_box
                                            
                                            dif_part_pos_x = np.abs(part_pos_x-com_x)
                                            dif_part_pos_y = np.abs(part_pos_y-com_y)
                                            if dif_part_pos_x>h_box:
                                                dif_part_pos_x=dif_part_pos_x-l_box
                                            if dif_part_pos_y>h_box:
                                                dif_part_pos_y=dif_part_pos_y-l_box

                                            rad_dif_part_pos = (dif_part_pos_x**2 + dif_part_pos_y**2)**0.5
                                            
                                            if (rad_look-1)<=rad_dif_part_pos<=rad_look:
                                                x_arr=np.append(x_arr,part_pos_x)
                                                y_arr=np.append(y_arr,part_pos_y)
                                                if typParts[ix][iy][h]==0:
                                                    count_A_edge2+=1
                                                elif typParts[ix][iy][h]==1:
                                                    count_B_edge2+=1
                            elif testIDs[ix][iy]==1:
                                if len(binParts[ix][iy])>0:
                                        for h in range(0,len(binParts[ix][iy])):
                                            part_pos=pos[binParts[ix][iy][h]]
                                            part_pos_x=part_pos[0]+h_box
                                            part_pos_y=part_pos[1]+h_box
                                            
                                            dif_part_pos_x = np.abs(part_pos_x-com_x)
                                            dif_part_pos_y = np.abs(part_pos_y-com_y)
                                            if dif_part_pos_x>h_box:
                                                dif_part_pos_x=dif_part_pos_x-l_box
                                            if dif_part_pos_y>h_box:
                                                dif_part_pos_y=dif_part_pos_y-l_box
                                            rad_dif_part_pos = (dif_part_pos_x**2 + dif_part_pos_y**2)**0.5
                                            
                                            if (rad_look-1)<=rad_dif_part_pos<=rad_look:

                                                x_arr=np.append(x_arr,part_pos_x)
                                                y_arr=np.append(y_arr,part_pos_y)

                                                if typParts[ix][iy][h]==0:
                                                    count_A_dense2+=1
                                                elif typParts[ix][iy][h]==1:
                                                    count_B_dense2+=1

                            elif testIDs[ix][iy]==2:
                                if len(binParts[ix][iy])>0:
                                        for h in range(0,len(binParts[ix][iy])):
                                            part_pos=pos[binParts[ix][iy][h]]
                                            
                                            part_pos_x=part_pos[0]+h_box
                                            part_pos_y=part_pos[1]+h_box
                                            
                                            
                                            dif_part_pos_x = np.abs(part_pos_x-com_x)
                                            dif_part_pos_y = np.abs(part_pos_y-com_y)
                                            if dif_part_pos_x>h_box:
                                                dif_part_pos_x=dif_part_pos_x-l_box
                                            if dif_part_pos_y>h_box:
                                                dif_part_pos_y=dif_part_pos_y-l_box
                                            rad_dif_part_pos = (dif_part_pos_x**2 + dif_part_pos_y**2)**0.5

                                            if (rad_look-1)<=rad_dif_part_pos<=rad_look:
                                                x_arr=np.append(x_arr,part_pos_x)
                                                y_arr=np.append(y_arr,part_pos_y)
                                                if typParts[ix][iy][h]==0:
                                                    count_A_gas2+=1
                                                elif typParts[ix][iy][h]==1:
                                                    count_B_gas2+=1
                #plt.plot(x_arr,y_arr,'.')
                #plt.plot(com_x,com_y,'.',c='r')
                #plt.show()
                
                tot_count_A_edge[i]=count_A_edge2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_B_edge[i]=count_B_edge2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_A_dense[i]=count_A_dense2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_B_dense[i]=count_B_dense2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_A_gas[i]=count_A_gas2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_B_gas[i]=count_B_gas2/(math.pi*((rad_look)**2-(rad_look-1)**2))
                tot_count_A[i]=tot_count_A_edge[i]+tot_count_A_dense[i]+tot_count_A_gas[i]
                tot_count_B[i]=tot_count_B_edge[i]+tot_count_B_dense[i]+tot_count_B_gas[i]
        else:
            count_A_edge=0
            count_B_edge=0
            count_A_dense=0
            count_B_dense=0
            count_A_edge_big=0
            count_B_edge_big=0
            count_A_dense_big=0
            count_B_dense_big=0
            count_A_gas=0
            count_B_gas=0
            gasArea=0
            edgeArea=0
            bulkArea=0
            r=[0]
            tot_count_A_gas=[0]
            tot_count_B_gas=[0]
            tot_count_A_edge=[0]
            tot_count_B_edge=[0]
            tot_count_A_dense=[0]
            tot_count_B_dense=[0]
            tot_count_A=[0]
            tot_count_B=[0]
        g = open(outPath+outTxt, 'a')
        g.write('{0:.3f}'.format(tst).center(30) + ' ')
        g.write('{0:.3f}'.format(count_A_gas).center(30) + ' ')
        g.write('{0:.3f}'.format(count_B_gas).center(30) + ' ')
        g.write('{0:.3f}'.format(count_A_edge).center(30) + ' ')
        g.write('{0:.3f}'.format(count_B_edge).center(30) + ' ')
        g.write('{0:.3f}'.format(count_A_dense).center(30) + ' ')
        g.write('{0:.3f}'.format(count_B_dense).center(30) + ' ')
        g.write('{0:.3f}'.format(gasArea).center(30) + ' ')
        g.write('{0:.3f}'.format(edgeArea).center(30) + ' ')
        g.write('{0:.3f}'.format(bulkArea).center(30) + '\n')
        g.close()
        
        g = open(outPath+outTxt2, 'a')
        for k in range(0, len(r)):
            g.write('{0:.3f}'.format(tst).center(30) + ' ')
            g.write('{0:.3f}'.format(r[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_A_gas[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_B_gas[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_A_edge[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_B_edge[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_A_dense[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_B_dense[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_A[k]).center(30) + ' ')
            g.write('{0:.3f}'.format(tot_count_B[k]).center(30) + '\n')
        g.close()
    
    #r_norm=r            
    #pad = "pa" + str(peA) + "_pb" + str(peB) + "_xa" + str(parFrac) 
    #img_files='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/proj/dklotsalab/users/ABPs/binary_soft/random_init/composition_analysis2/'
    '''
    #clust_avg=average_clust_radius/count_of_rads
    plt.plot(r_norm,np.mean(tot_count_A_edge,axis=0),label='Pe='+str(peA)+' edge')  
    plt.plot(r_norm,np.mean(tot_count_B_edge,axis=0),label='Pe='+str(peB)+' edge')      
    plt.plot(r_norm,np.mean(tot_count_A_dense,axis=0),label='Pe='+str(peA)+' bulk')      
    plt.plot(r_norm,np.mean(tot_count_B_dense,axis=0),label='Pe='+str(peB)+' bulk')      
    plt.plot(r_norm,np.mean(tot_count_A_gas,axis=0),label='Pe='+str(peA)+' gas')      
    plt.plot(r_norm,np.mean(tot_count_B_gas,axis=0),label='Pe='+str(peB)+' gas') 
    #plt.axvline(x=clust_avg, label='average cluster radius',c='black')
    #plt.plot(c,b,c='r', label='edge ending')
    #plt.plot(r_edge_arr, b,c='g',label='edge beginning')
    plt.ylabel('density')
    plt.xlabel('radius from CoM')
    plt.legend()
    plt.savefig(img_files+"edge_comp_" + pad + "radial_dense.png", dpi=150)
    plt.close()
    plt.show()
            
    plt.plot(r_norm,np.mean(tot_count_A,axis=0), label='Pe='+str(peA))  
    plt.plot(r_norm,np.mean(tot_count_B,axis=0), label='Pe='+str(peB))  
    #plt.axvline(x=clust_avg, label='average cluster radius',c='black')

    plt.ylabel('density')
    plt.xlabel('radius from CoM')
    plt.legend()
    #plt.show()
    plt.savefig(img_files+"edge_comp_" + pad + "radial_dense_avg.png", dpi=150)
    plt.close()
    #plt.show()
    '''
    '''
    tot_count_edge[0,0]=0
    tot_count_edge[0,1]=0
    tot_count_edge[1,0]=0
    tot_count_edge[1,1]=0
    tot_count_edge[2,0]=0
    tot_count_edge[2,1]=0
    tot_count_dense[0,0]=0
    tot_count_dense[0,1]=0
    tot_count_dense[1,0]=0
    tot_count_dense[1,1]=0
    tot_count_dense[2,0]=0
    tot_count_dense[2,1]=0
    
    
    for i in range(3,end):
        if tot_count_edge[i,0]!=0:
            if tot_count_edge[i-1,0]==0 and tot_count_edge[i+1,0]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
            elif tot_count_edge[i-2,0]==0 and tot_count_edge[i+2,0]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
            elif tot_count_edge[i-3,0]==0 and tot_count_edge[i+3,0]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
        elif tot_count_edge[i,1]!=0:
            if tot_count_edge[i-1,1]==0 and tot_count_edge[i+1,1]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
            elif tot_count_edge[i-2,1]==0 and tot_count_edge[i+2,1]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
            elif tot_count_edge[i-3,1]==0 and tot_count_edge[i+3,1]==0:
                tot_count_edge[i,0]=0
                tot_count_edge[i,1]=0
        if tot_count_dense[i,0]!=0:
            if tot_count_dense[i-1,0]==0 and tot_count_dense[i+1,0]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
            elif tot_count_dense[i-2,0]==0 and tot_count_dense[i+2,0]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
            elif tot_count_dense[i-3,0]==0 and tot_count_dense[i+3,0]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
        elif tot_count_dense[i,1]!=0:
            if tot_count_dense[i-1,1]==0 and tot_count_dense[i+1,1]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
            elif tot_count_dense[i-2,1]==0 and tot_count_dense[i+2,1]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
            elif tot_count_dense[i-3,1]==0 and tot_count_dense[i+3,1]==0:
                tot_count_dense[i,0]=0
                tot_count_dense[i,1]=0
        if tot_count_gas[i,0]!=0:
            if tot_count_gas[i-1,0]==0 and tot_count_gas[i+1,0]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
            elif tot_count_gas[i-2,0]==0 and tot_count_gas[i+2,0]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
            elif tot_count_gas[i-3,0]==0 and tot_count_gas[i+3,0]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
        elif tot_count_dense[i,1]!=0:
            if tot_count_gas[i-1,1]==0 and tot_count_gas[i+1,1]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
            elif tot_count_gas[i-2,1]==0 and tot_count_gas[i+2,1]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
            elif tot_count_gas[i-3,1]==0 and tot_count_gas[i+3,1]==0:
                tot_count_gas[i,0]=0
                tot_count_gas[i,1]=0
                
    a = np.min(np.where(tot_count_edge[:,0]!=0)[0])
    b = np.min(np.where(tot_count_edge[:,1]!=0)[0])
    '''
    #img_files='/proj/dklotsalab/users/ABPs/binary_soft/random_init/composition_analysis2/'
    #pad = "pa" + str(peA) + "_pb" + str(peB) + "_xa" + str(parFrac) + "_eps" + str(eps) 
    '''
    if a<b:
        c=b
    else:
        c=a
    
    if peA>peB:
        ratio_count_edge = tot_count_edge[c:,1]/tot_count_edge[c:,0]
        ratio_count_dense = tot_count_dense[c:,1]/tot_count_dense[c:,0]
        ratio_count_gas = tot_count_gas[c:,1]/tot_count_gas[c:,0]
        plt.plot(time_arr[c:],ratio_count_edge, label=str(peB)+':'+str(peA)+' edge')
        plt.plot(time_arr[c:],ratio_count_dense, label=str(peB)+':'+str(peA)+' dense')
        plt.plot(time_arr[c:],ratio_count_gas, label=str(peB)+':'+str(peA)+' gas')

    else:
        ratio_count_edge = tot_count_edge[c:,0]/tot_count_edge[c:,1]
        ratio_count_dense = tot_count_dense[c:,0]/tot_count_dense[c:,1]
        ratio_count_gas = tot_count_gas[c:,0]/tot_count_gas[c:,1]
        plt.plot(time_arr[c:],ratio_count_edge, label=str(peA)+':'+str(peB)+' edge')
        plt.plot(time_arr[c:],ratio_count_dense, label=str(peA)+':'+str(peB)+' dense')
        plt.plot(time_arr[c:],ratio_count_gas, label=str(peA)+':'+str(peB)+' gas')
    '''
    '''
    plt.plot(time_arr,tot_count_edge[:,0]/edge_area_arr,label='PeA='+str(peA)+' edge')
    plt.plot(time_arr,tot_count_edge[:,1]/edge_area_arr,label='PeB='+str(peB)+' edge')
    plt.plot(time_arr,tot_count_dense[:,0]/dense_area_arr,label='PeA='+str(peA)+' dense')
    plt.plot(time_arr,tot_count_dense[:,1]/dense_area_arr,label='PeB='+str(peB)+' dense')
    plt.plot(time_arr,tot_count_gas[:,0]/gas_area_arr,label='PeA='+str(peA)+' gas')
    plt.plot(time_arr,tot_count_gas[:,1]/gas_area_arr,label='PeB='+str(peB)+' gas')
    plt.legend()
    plt.ylabel('Concentration')
    plt.xlabel('Brownian Time Units')
    plt.title('PeA:peB for xA='+str(parFrac))
    plt.savefig(img_files+"edge_comp_" + pad + "tot_count.png", dpi=150)
    plt.close()
    '''
    #ratio_avg_gas = np.mean(ratio_count_gas)
    #ratio_avg_dense = np.mean(ratio_count_dense)
    #ratio_avg_edge = np.mean(ratio_count_edge)

    '''
    with open(img_files+"edge_comp_" + pad + ".txt","a+") as f:
        f.write('PeA'+str(int(peA))+'PeB'+str(int(peB))+"\n")
        f.write('time_array\n')
        np.savetxt(f,time_arr)
        f.write('total_counts edge (A then B)\n')
        np.savetxt(f,tot_count_edge)
        f.write('total_counts gas (A then B)\n')
        np.savetxt(f,tot_count_gas)
        f.write('total_counts bulk (A then B)\n')
        np.savetxt(f,tot_count_dense)
            #plt.savefig(imgPath+imgFile, bbox_inches='tight', pad_inches=0., dpi=250)
            #plt.close()
    '''

