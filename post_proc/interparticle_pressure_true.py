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
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build/'
#gsdPath='/proj/dklotsalab/users/ABPs/binary_soft/random_init/'
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/interpart_press/'
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
dt = dtau * tauLJ    

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
f = hoomd.open(name=inFile, mode='rb')

# Outfile to write data to
outF = inFile[:-4]
outTxt = 'interpart_press_' + outF + '.txt'

g = open(outPath+outTxt, 'w') # write file headings
g.write('Timestep'.center(15) + ' ' +\
        'gasArea'.center(15) + ' ' +\
        'gasSigXX'.center(15) + ' ' +\
        'gasSigXY'.center(15) + ' ' +\
        'gasSigYX'.center(15) + ' ' +\
        'gasSigYY'.center(15) + ' ' +\
        'gasTrace'.center(15) + ' ' +\
        'bulkArea'.center(15) + ' ' +\
        'bulkSigXX'.center(15) + ' ' +\
        'bulkSigXY'.center(15) + ' ' +\
        'bulkSigYX'.center(15) + ' ' +\
        'bulkSigYY'.center(15) + ' ' +\
        'bulkTrace'.center(15) + ' ' +\
        'Length'.center(15) + ' ' +\
        'NDense'.center(15) + '\n')
g.close()

            
box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

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
    gas_area_arr = np.zeros(end)
    dense_area_arr = np.zeros(end)
    edge_area_arr = np.zeros(end)
    time_arr=np.zeros(end)
    for j in range(start, end):
        # Outfile to write data to
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
        
        
        

        if len(large_clust_ind_all[0])>0:
            
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            liqPos = []
            gasPos = []
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
            bulkSigXY=0
            bulkSigYX=0
            gasSigXX = 0
            gasSigYY = 0
            gasSigXY=0
            gasSigYX=0
            
            reftype=np.zeros(len(ids))
            for k in range(0, len(ids)):
                # Convert position to be > 0 to place in list mesh
                tmp_posX = pos[k][0] + h_box
                tmp_posY = pos[k][1] + h_box
                x_ind = int(tmp_posX / sizeBin)
                y_ind = int(tmp_posY / sizeBin)
                # Only compute pressure for non-edge bins
                if blurBin[x_ind][y_ind] != 1:
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
                                dist = computeDist(refx, refy, pos[comp][0], pos[comp][1])
                                # If potential is on ...
                                if 0.1 < dist <= r_cut:
                                    # Compute the x and y components of force
                                    fx, fy = computeFLJ(dist, refx, refy, pos[comp][0], pos[comp][1], eps)
                                    # This will go into the bulk pressure
                                    if clust_size[ids[k]] >= min_size:
                                        bulkSigXX += (fx * (pos[comp][0] - refx))
                                        bulkSigYY += (fy * (pos[comp][1] - refy))
                                        bulkSigXY += (fx * (pos[comp][1] - refy))
                                        bulkSigYX += (fy * (pos[comp][0] - refx))
                                    # This goes into the gas pressure
                                    else:
                                        gasSigXX += (fx * (pos[comp][0] - refx))
                                        gasSigYY += (fy * (pos[comp][1] - refy))
                                        gasSigXY += (fx * (pos[comp][1] - refy))
                                        gasSigYX += (fy * (pos[comp][0] - refx))
                                    
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

            # The edge length of sufficiently large clusters
            lEdge = Nedges * sizeBin
            # Divide by two because each pair is counted twice
            bulkTrace = (bulkSigXX + bulkSigYY + bulkSigXY + bulkSigYX)/2.
            gasTrace = (gasSigXX + gasSigYY + gasSigXY + gasSigYX)/2.
            # Area of a bin
            binArea = sizeBin * sizeBin
            # Area of each phase
            gasArea = binArea * gasBins
            bulkArea = binArea * bulkBins
            edgeArea = binArea * edgeBins
            dense_area_arr[j] = bulkArea
            gas_area_arr[j] = gasArea
            edge_area_arr[j] = edgeArea
            
            ndense = max(clust_size)
            
            g = open(outPath+outTxt, 'a')
            g.write('{0:.3f}'.format(tst).center(15) + ' ')
            g.write('{0:.3f}'.format(gasArea).center(15) + ' ')
            g.write('{0:.3f}'.format(gasSigXX).center(15) + ' ')
            g.write('{0:.3f}'.format(gasSigXY).center(15) + ' ')
            g.write('{0:.3f}'.format(gasSigYX).center(15) + ' ')
            g.write('{0:.3f}'.format(gasSigYY).center(15) + ' ')
            g.write('{0:.3f}'.format(gasTrace).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkArea).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkSigXX).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkSigXY).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkSigYX).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkSigYY).center(15) + ' ')
            g.write('{0:.3f}'.format(bulkTrace).center(15) + ' ')
            g.write('{0:.1f}'.format(lEdge).center(15) + ' ')
            g.write('{0:.0f}'.format(ndense).center(15) + '\n')
            g.close()
