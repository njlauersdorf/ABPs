'''
#                           This is an 80 character line                       #
Compute the length of the cluster edge:
-Use Freud to find the complete system neighborlist
-Grab the largest cluster
-Mesh the system
-Compute which bins have largest cluster particles
-If adjacent bins are empty, the reference bin is an edge
-Multiply by bin size to get length
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections

import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
import matplotlib.pyplot as plt
import time
# Run locally
#hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build/'
#gsdPath='/proj/dklotsalab/users/ABPs/binary_soft/random_init/'
# Run locally
#sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)

hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
gsdPath='/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)

r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
inFile = 'pa150_pb500_xa50_ep1_phi60_pNum10000.gsd'

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
'''
name = "pa" + str(peA) +\
"_pb" + str(peB) +\
"_xa" + str(partPercA) +\
"_ep" + str(epsAB)+\
"_pNum" + str(partNum)
'''
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
import random
from scipy import stats

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
'''
# Get infile and open
inFile = str(sys.argv[1])
if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
''' 
f = hoomd.open(name=gsdPath+inFile, mode='rb')

# Outfile to write data to
base = add + 'pressure_pa' + str(peA) +\
       '_pb' + str(peB) +\
       '_xa' + str(parFrac) +\
       '_phi' + str(intPhi) +\
       '_ep' + '{0:.3f}'.format(eps)
outFile = base + '.txt'
imgFile = base + '.png'

g = open(outFile, 'w') # write file headings
g.write('Timestep'.center(10) + ' ' +\
        'gasArea'.center(20) + ' ' +\
        'gasTrace'.center(20) + ' ' +\
        'gasPress'.center(20) + ' ' +\
        'bulkArea'.center(20) + ' ' +\
        'bulkTrace'.center(20) + ' ' +\
        'bulkPress'.center(20) + ' ' +\
        'SurfaceTense'.center(20) + ' ' +\
        'Length'.center(20) + '\n')
g.close()


box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

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
    gas_area_arr = np.zeros(end)
    dense_area_arr = np.zeros(end)
    edge_area_arr = np.zeros(end)
    time_arr=np.zeros(end)
    for j in range(start, end):
        j=600
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
        print(j)
        clp_all.compute(system_all, ids)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        min_size=int(partNum/20)
        large_clust_ind_all=np.where(clust_size>min_size)
        lcID = np.where(clust_size == np.amax(clust_size))
        lcIDcomps=np.where(ids==lcID[0][0])

        

        if len(large_clust_ind_all[0])>0:
            
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            liqPos = []
            gasPos = []
            stop
            for k in range(0, len(ids)):
                
                # Convert position to be > 0 to place in list mesh
                tmp_posX = pos[k][0] + h_box
                tmp_posY = pos[k][1] + h_box
                x_ind = int(tmp_posX / sizeBin)
                y_ind = int(tmp_posY / sizeBin)
                # Append all particles to appropriate bin
                binParts[x_ind][y_ind].append(k)
            
                # Get sufficient cluster mesh as well
                if clust_size[ids[k]] >= min_size:
                    liqPos.append(pos[k])
                    occParts[x_ind][y_ind] = 1
                # Get a gas particle list as well
                elif clust_size[ids[k]] <= 100:
                    gasPos.append(pos[k])
            
            # If sufficient neighbor bins are empty, we have an edge
            thresh = 1.5
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

            # Now let's compute the pressure
            bulkSigXX = 0
            bulkSigYY = 0
            
            gasSigXX = 0
            gasSigYY = 0
            reftype=np.zeros(len(ids))
            for k in range(0, len(ids)):
                # Convert position to be > 0 to place in list mesh
                tmp_posX = pos[k][0] + h_box
                tmp_posY = pos[k][1] + h_box
                x_ind = int(tmp_posX / sizeBin)
                y_ind = int(tmp_posY / sizeBin)
                # Only compute pressure for non-edge bins
                #if blurBin[x_ind][y_ind] == 1:
                    # If at right edge, wrap to left
                if (x_ind + 1) != NBins:
                    lookx = [x_ind-1, x_ind, x_ind+1]
                else:
                    lookx = [x_ind-1, x_ind, 0]
                # If at top edge, wrap to bottom
                if (y_ind + 1) != NBins:
                    looky = [y_ind-1, y_ind, y_ind+1]
                else:
                    looky = [y_ind-1, y_ind, 0]
                # Reference particle position
                refx = pos[k][0]
                refy = pos[k][1]
                # Loop through surrounding x-index
                for indx in lookx:
                    # Loop through surrounding y-index
                    for indy in looky:
                       # Loop through all particles in that bin
                        for comp in binParts[indx][indy]:
                            typParts[x_ind][y_ind]=typ[binParts[indx][indy]]
                            # Compute the distance
            # Now let's get the area of each phase (by summing bin areas)
            gasBins = 0
            bulkBins = 0
            edgeBins = 0
            testIDs = [[0 for b in range(NBins)] for a in range(NBins)]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    # Is the bin an edge?
                    if blurBin[ix][iy] == 1:
                        testIDs[ix][iy] = 0
                        edgeBins+=1
                        continue
                    # Does the bin belong to the dense phase?

                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] > min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
            count_A_edge=0
            count_B_edge=0
            count_A_gas=0
            count_B_gas=0
            count_A_dense=0
            count_B_dense=0
            
            check_clust=[[0 for b in range(NBins)] for a in range(NBins)]
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] > min_size:
                            check_clust[ix][iy]=1
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if check_clust[ix][iy]==1:
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
            print('j')
            print(j)
            print(count_A_gas,flush=True)
            print(count_B_gas,flush=True)
            print('dense',flush=True)
            print(count_A_dense,flush=True)
            print(count_B_dense,flush=True)
            print('edge', flush=True)
            print(count_A_edge, flush=True)
            print(count_B_edge,flush=True)
            tot_count_edge[j,0]=count_A_edge
            tot_count_edge[j,1]=count_B_edge
            tot_count_dense[j,0]=count_A_dense
            tot_count_dense[j,1]=count_B_dense
            tot_count_gas[j,0]=count_A_gas
            tot_count_gas[j,1]=count_B_gas
            
            #print(count_A_dense,flush=True)
            #print(count_B_dense,flush=True)
            #print(count_A_gas,flush=True)
            #print(count_B_gas,flush=True)
            
            # The edge length of sufficiently large clusters
            lEdge = Nedges * sizeBin
            # Divide by two because each pair is counted twice
            bulkTrace = (bulkSigXX + bulkSigYY)/2.
            gasTrace = (gasSigXX + gasSigYY)/2.
            # Area of a bin
            binArea = sizeBin * sizeBin
            # Area of each phase
            gasArea = binArea * gasBins
            bulkArea = binArea * bulkBins
            edgeArea = binArea * edgeBins
            dense_area_arr[j] = bulkArea
            gas_area_arr[j] = gasArea
            edge_area_arr[j] = edgeArea

            '''
            # Pressure of each phase
            gasPress = gasTrace / gasArea
            bulkPress = bulkTrace / bulkArea
            # Surface tension
            surfaceTense = (bulkPress - gasPress) * lEdge
        
            print("Number of gas bins: ", gasBins)
            print("Gas phase area: ", gasArea)
            print("Number of bulk bins: ", bulkBins)
            print("Bulk phase area: ", bulkArea)
            print("Trace of gas stress tensor: ", gasTrace)
            print("Trace of bulk stress tensor: ", bulkTrace)
            print("Gas phase pressure: ", gasPress)
            print("Bulk phase pressure: ", bulkPress)
            print("Surface tension: ", surfaceTense)
            '''
#           # A sanity check on a perfect hcp circle
#           print(Nedges)
#           print(Nedges * sizeBin)
#           x = list(list(zip(*liqPos))[0])
#           y = list(list(zip(*liqPos))[1])
#           diam = max(x) - min(x)
#           circ = diam * np.pi
#           print(circ)
#           print(Nedges * sizeBin / circ)

            # Write this to a textfile with the timestep
            '''
            g = open(outFile, 'a')
            g.write('{0:.3f}'.format(tst).center(10) + ' ')
            g.write('{0:.3f}'.format(gasArea).center(20) + ' ')
            g.write('{0:.3f}'.format(gasTrace).center(20) + ' ')
            g.write('{0:.3f}'.format(gasPress).center(20) + ' ')
            g.write('{0:.3f}'.format(bulkArea).center(20) + ' ')
            g.write('{0:.3f}'.format(bulkTrace).center(20) + ' ')
            g.write('{0:.3f}'.format(bulkPress).center(20) + ' ')
            g.write('{0:.3f}'.format(surfaceTense).center(20) + ' ')
            g.write('{0:.1f}'.format(lEdge).center(20) + '\n')
            g.close()
            '''
            '''
            # Output an image of the frame we're computing
            imgPath='/Users/nicklauersdorf/hoomd-blue/build/img_files2/'

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow([*zip(*testIDs)], extent=[0, l_box, 0, l_box], origin='lower', aspect='auto')
            ax[0].set_aspect('equal')
            ax[0].axes.set_xticks([])
            ax[0].axes.set_yticks([])
            # Plot gas particles
            x = list(list(zip(*gasPos))[0])
            y = list(list(zip(*gasPos))[1])
            ax[1].scatter(x, y, edgecolor='none', s=0.05, c='b')
            # Plot liquid particles
            x = list(list(zip(*liqPos))[0])
            y = list(list(zip(*liqPos))[1])
            ax[1].scatter(x, y, edgecolor='none', s=0.05, c='g')
            ax[1].set_xlim(-h_box, h_box)
            ax[1].set_ylim(-h_box, h_box)
            ax[1].axes.set_xticks([])
            ax[1].axes.set_yticks([])
            ax[1].set_aspect('equal')
            plt.tight_layout(w_pad=0.1)
            plt.show()
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
    '''
    print('edge!',flush=True)
    print(tot_count_edge[:,0],flush=True)
    print(tot_count_edge[:,1],flush=True)            
    a = np.min(np.where(tot_count_edge[:,0]!=0)[0])
    b = np.min(np.where(tot_count_edge[:,1]!=0)[0])

    img_files='/proj/dklotsalab/users/ABPs/binary_soft/random_init/composition_analysis2/'

    pad = "pa" + str(int(peA)) + "_pb" + str(int(peB)) + "_xa" + str(int(parFrac))+'_phi'+str(intPhi)+'_pNum'+str(partNum) 
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
        print('opt1',flush=True)
    else:
        print('opt2',flush=True)
        print('peB then peA', flush=True)
        print(peB,flush=True)
        print(peA,flush=True)
        print(parFrac,flush=True)
        ratio_count_edge = tot_count_edge[c:,0]/tot_count_edge[c:,1]
        ratio_count_dense = tot_count_dense[c:,0]/tot_count_dense[c:,1]
        ratio_count_gas = tot_count_gas[c:,0]/tot_count_gas[c:,1]
        plt.plot(time_arr[c:],ratio_count_edge, label=str(peA)+':'+str(peB)+' edge')
        plt.plot(time_arr[c:],ratio_count_dense, label=str(peA)+':'+str(peB)+' dense')
        plt.plot(time_arr[c:],ratio_count_gas, label=str(peA)+':'+str(peB)+' gas')
    plt.legend()
    plt.ylabel('ratio of slow:fast concentration')
    plt.xlabel('Brownian Time Units')
    plt.title('PeA:PeB for xA='+str(int(parFrac)))
    plt.savefig(img_files+"edge_comp_" + pad + ".png", dpi=150)
    plt.close()
    '''
    #ratio_avg_gas = np.mean(ratio_count_gas)
    #ratio_avg_dense=np.mean(ratio_count_dense)
    #ratio_avg_edge = np.mean(ratio_count_edge)
    print('AREA!', flush=True)
    open(img_files+"mean_data_edge_comp_"+pad+"time.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"edge_parts.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"gas_parts.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"bulk_parts.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"edge_area.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"gas_area.txt","w+")
    open(img_files+"mean_data_edge_comp_"+pad+"bulk_area.txt","w+")
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"time.txt",time_arr,delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"gas_parts.txt",tot_count_gas,delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"edge_parts.txt",tot_count_edge,delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"bulk_parts.txt",tot_count_dense,delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"edge_area.txt",edge_area_arr,delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"gas_area.txt",gas_area_arr, delimiter=',')
    np.savetxt(img_files+"mean_data_edge_comp_"+pad+"bulk_area.txt",dense_area_arr,delimiter=',')
    '''
    with open(img_files+"mean_data_edge_comp_" + pad + ".txt","a+") as f:
        print('yes this works', flush=True)
        f.write('PeA'+str(int(peA))+'PeB'+str(int(peB))+"\n")
        f.write('time_array\n')
        np.savetxt(f,time_arr)
        f.write('total_counts edge (A then B)\n')
        np.savetxt(f,tot_count_edge)
        f.write('total_counts gas (A then B)\n')
        np.savetxt(f,tot_count_gas)
        f.write('total_counts bulk (A then B)\n')
        np.savetxt(f,tot_count_dense)
        f.write('edge Area\n')
        np.savetxt(f,edge_area_arr)
        f.write('gas Area\n')
        np.savetxt(f,gas_area_arr)
        f.write('bulk Area\n')
        np.savetxt(f,dense_area_arr)
        #plt.savefig(imgPath+imgFile, bbox_inches='tight', pad_inches=0., dpi=250)
        #plt.close()
    '''     
