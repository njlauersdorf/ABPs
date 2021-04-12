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

#from descartes.patch import PolygonPatch
# Run locally
hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#/nas/home/njlauers/hoomd-blue/build/'#Users/nicklauersdorf/hoomd-blue/build/'
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
outPath='/Volumes/External/mono_test/'
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
#matplotlib.use('Agg')
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


outTxt = 'absorption_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'clust_size'.center(15) + ' ' +\
        'min_ang'.center(15) + ' ' +\
        'max_ang'.center(15) + ' ' +\
        'ang_desorb'.center(15) + ' ' +\
        'ang_absorb'.center(15) + ' ' +\
        'net_flux'.center(15) + ' ' +\
        'tot_desorb'.center(15) + ' ' +\
        'tot_absorb'.center(15) + '\n')
g.close()

run=0
with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 920                # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
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

        
        pos = snap.particles.position               # position
        snap_future = t[j+1]
        pos_future=snap_future.particles.position
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
            testIDs_final = [[0 for b in range(NBins)] for a in range(NBins)]

            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    
                    # Is the bin an edge?
                    
                    if edgeBin[ix][iy] == 1:
                        testIDs[ix][iy] = 0
                        testIDs_final[ix][iy] = 0
                        edgeBins+=1
                        continue
                    # Does the bin belong to the dense phase?

                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] >=min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            testIDs_final[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
                    testIDs_final[ix][iy] = 2
              
                
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
                        
            
            search_range = 14.0
            search_range2 = 6.0
            
            
            for ix in range(0, len(binParts)):
                print(ix)
                for iy in range(0, len(binParts)):
                    if edgeBin[ix][iy]==1:
                        x_ref=(ix+0.5)*sizeBin
                        y_ref=(iy+0.5)*sizeBin
                        ixy_range = int(search_range/sizeBin)+1
                        new_ix_max = ix+ixy_range
                        new_ix_min = ix-ixy_range
                        new_iy_max = iy+ixy_range
                        new_iy_min = iy-ixy_range
                        
                        if new_ix_max>len(binParts):
                            new_ix_max = len(binParts)
                        if new_ix_min<0:
                            new_ix_min = 0
                        if new_iy_max>len(binParts):
                            new_iy_max = len(binParts)
                        if new_iy_min>len(binParts):
                            new_iy_min = 0
                        for ix2 in range(new_ix_min, new_ix_max):
                            for iy2 in range(new_iy_min, new_iy_max):
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
            
            
            edge_id = np.where(PhaseParts2==0)[0]
            bulk_id = np.where(PhaseParts2==1)[0]
            gas_id = np.where(PhaseParts2==2)[0]
            
            
            edge_parts_tot =  np.array([])
            bulk_parts_tot =  np.array([])
            gas_parts_tot =  np.array([])
            
            for ix in range(0, len(binParts)):
                for iy in range(0, len(binParts)):
                    for h in range(0,len(binParts[ix][iy])):
                            if PhaseParts2[binParts[ix][iy][h]]==0:
                                edge_parts_tot = np.append(edge_parts_tot, binParts[ix][iy][h])
                            elif PhaseParts2[binParts[ix][iy][h]]==1:
                                bulk_parts_tot = np.append(bulk_parts_tot, binParts[ix][iy][h])
                            elif PhaseParts2[binParts[ix][iy][h]]==2:
                                gas_parts_tot = np.append(gas_parts_tot, binParts[ix][iy][h])       
            #widths = [1.0,1.0]#, [1,1,1,1],[1,1,1,1]]#[1,1.205, 1.205, 1.205]
                #gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.2)
                #ax0 = fig.add_subplot(gs[0, 0])
            myEps = [1., 0.1, 0.01, 0.001, 0.0001]
            
            count_A_edge=0
            count_B_edge=0
            gas_particle_range=2.0
            gas_r_lim=gas_particle_range*lat
            bulk_particle_range=5.0
            end_loop=0
            steps=0
            num_sights_len=20
            
            num_sights=np.arange(0, 360+int(360/num_sights_len),int(360/num_sights_len))
            bulk_parts_los=  [[] for c in range(num_sights_len)]
            edge_parts_los=  [[] for c in range(num_sights_len)]
            gas_parts_los=  [[] for c in range(num_sights_len)]



            area_slice=np.zeros(len(radius)-1)
            for f in range(0,len(radius)-1):
                area_slice[f]=((num_sights[1]-num_sights[0])/360)*math.pi*(radius[f+1]**2-radius[f]**2)
            for k in range(1,len(num_sights)):
                pos_new_x=np.array([])
                pos_new_y=np.array([])
                losBin = [[0 for b in range(NBins)] for a in range(NBins)]

                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        x_min_ref=ix*sizeBin#+com_tmp_posX
                        x_max_ref=(ix+1)*sizeBin#+com_tmp_posX
                        y_min_ref=iy*sizeBin# +com_tmp_posY
                        y_max_ref=(iy+1)*sizeBin#+com_tmp_posY

                        dif_x_min = (x_min_ref-com_tmp_posX)
                        difx_min_abs = np.abs(dif_x_min)

                        if difx_min_abs>=h_box:
                            if dif_x_min < -h_box:
                                dif_x_min += l_box
                            else:
                                dif_x_min -= l_box
                        dif_x_max = (x_max_ref-com_tmp_posX)

                        difx_max_abs = np.abs(dif_x_max)
                        if difx_max_abs>=h_box:
                            if dif_x_max < -h_box:
                                dif_x_max += l_box
                            else:
                                dif_x_max -= l_box
                        dif_y_min = (y_min_ref-com_tmp_posY)

                        dify_min_abs = np.abs(dif_y_min)
                        if dify_min_abs>=h_box:
                            if dif_y_min < -h_box:
                                dif_y_min += l_box
                            else:
                                dif_y_min -= l_box
                        dif_y_max = (y_max_ref-com_tmp_posY)

                        dify_max_abs = np.abs(dif_y_max)
                        if dify_max_abs>=h_box:
                            if dif_y_max < -h_box:
                                dif_y_max += l_box
                            else:
                                dif_y_max -= l_box
                            
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

                            #if min_angle<=
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
                                    
                            
                        elif ((ix==com_x_ind) and (iy==com_y_ind)):
                            losBin[ix][iy]=1
                for ix in range(0, len(losBin)):
                    for iy in range(0, len(losBin)):
                        if losBin[ix][iy]==1:
                            for h in range(0,len(binParts[ix][iy])):
                                if PhaseParts2[binParts[ix][iy][h]]==0:
                                    edge_parts_los[k-1] = np.append(edge_parts_los[k-1], binParts[ix][iy][h])
                                elif PhaseParts2[binParts[ix][iy][h]]==1:
                                    bulk_parts_los[k-1] = np.append(bulk_parts_los[k-1], binParts[ix][iy][h])
                                elif PhaseParts2[binParts[ix][iy][h]]==2:
                                    gas_parts_los[k-1] = np.append(gas_parts_los[k-1], binParts[ix][iy][h]) 
            
            
                
            
                                 
            
            
            
            
            
            if test_run>0:
                desorb_parts_tot = np.intersect1d(gas_parts_tot, edge_parts_prev)
                absorb_parts_tot = np.intersect1d(gas_parts_prev, edge_parts_tot)
                
                num_desorb_tot = len(desorb_parts_tot)
                num_absorb_tot = len(absorb_parts_tot)
                
                desorb_parts_los = [[] for c in range(num_sights_len)]
                absorb_parts_los = [[] for c in range(num_sights_len)]
                
                num_desorb_los = [0 for c in range(num_sights_len)]
                num_absorb_los = [0 for c in range(num_sights_len)]
                net_flux = [0 for c in range(num_sights_len)]
                for k in range(1, len(num_sights)):
                    desorb_parts_los[k-1] = np.intersect1d(gas_parts_los[k-1], edge_parts_los_prev[k-1])
                    if k==1:
                        desorb_parts_los[k-1] = np.append(desorb_parts_los[k-1], np.intersect1d(gas_parts_los[len(num_sights)-2], edge_parts_los_prev[k-1]))
                    else:
                        desorb_parts_los[k-1] = np.append(desorb_parts_los[k-1], np.intersect1d(gas_parts_los[k-2], edge_parts_los_prev[k-1]))
                    if k==(len(num_sights)-1):
                        desorb_parts_los[k-1] = np.append(desorb_parts_los[k-1], np.intersect1d(gas_parts_los[0], edge_parts_los_prev[k-1]))
                    else:
                        desorb_parts_los[k-1] = np.append(desorb_parts_los[k-1], np.intersect1d(gas_parts_los[k], edge_parts_los_prev[k-1]))
                    
                    
                    
                    absorb_parts_los[k-1] = np.intersect1d(gas_parts_los_prev[k-1], edge_parts_los[k-1])
                    if k==1:
                        absorb_parts_los[k-1] = np.append(absorb_parts_los[k-1], np.intersect1d(gas_parts_los_prev[len(num_sights)-2], edge_parts_los[k-1]))
                    else:
                        absorb_parts_los[k-1] = np.append(absorb_parts_los[k-1], np.intersect1d(gas_parts_los_prev[k-2], edge_parts_los[k-1]))
                    if k==(len(num_sights)-1):
                        absorb_parts_los[k-1] = np.append(absorb_parts_los[k-1], np.intersect1d(gas_parts_los_prev[0], edge_parts_los[k-1]))
                    else:
                        absorb_parts_los[k-1] = np.append(absorb_parts_los[k-1], np.intersect1d(gas_parts_los_prev[k], edge_parts_los[k-1]))
                    
                    num_desorb_los[k-1] = len(desorb_parts_los[k-1])
                    num_absorb_los[k-1] = len(absorb_parts_los[k-1])
                    net_flux[k-1] = num_absorb_los[k-1] - num_desorb_los[k-1]

                yellow = ("#fdfd96")
                green = ("#77dd77")
                red = ("#ff6961")
                       
                plt.figure()
                plt.scatter(pos[gas_id,0]+h_box, pos[gas_id,1]+h_box, s=0.6, c=red)                
                plt.scatter(pos[bulk_id,0]+h_box, pos[bulk_id,1]+h_box, s=0.6, c=green)
                plt.scatter(pos[edge_id,0]+h_box, pos[edge_id,1]+h_box, s=0.6, c=yellow)
                plt.scatter(com_tmp_posX, com_tmp_posY, s=20.0, c=red)
                
                #tot_mag = np.max(np.abs(net_flux))
                
                for k in range(1, len(num_sights)):
                    
                    arrow_ang = (num_sights[k]-0.5*(num_sights[k]-num_sights[k-1]))*np.pi/180
                    #magnitude = (np.abs(net_flux[k-1])/tot_mag) * h_box/2
                    magnitude = np.abs(net_flux[k-1])
                    offset = 40.
                    headwidth = 9.0
                    headlength = 9.0
                    
                    if 0<arrow_ang<np.pi/2:
                        arrow_ang_calc = arrow_ang
                        
                        magnitude_x = np.cos(arrow_ang_calc)*magnitude
                        magnitude_y = np.sin(arrow_ang_calc)*magnitude
                        shift_x = np.cos(arrow_ang_calc)*offset
                        shift_y = np.sin(arrow_ang_calc)*offset
                        
                        head_shift_x = np.cos(arrow_ang_calc) * headlength
                        head_shift_y = np.sin(arrow_ang_calc) * headlength
                        
                    elif np.pi/2<arrow_ang<np.pi:
                        arrow_ang_calc = arrow_ang-(np.pi/2)
                        magnitude_x = -np.sin(arrow_ang_calc)*magnitude
                        magnitude_y = np.cos(arrow_ang_calc)*magnitude
                        shift_x = -np.sin(arrow_ang_calc)*offset
                        shift_y = np.cos(arrow_ang_calc)*offset
                        
                        head_shift_x = -np.sin(arrow_ang_calc)*headlength
                        head_shift_y = np.cos(arrow_ang_calc)*headlength
                        
                    elif np.pi<arrow_ang<(3*np.pi/2):
                        arrow_ang_calc = arrow_ang-np.pi
                        magnitude_x = -np.cos(arrow_ang_calc)*magnitude
                        magnitude_y = -np.sin(arrow_ang_calc)*magnitude
                        shift_x = -np.cos(arrow_ang_calc)*offset
                        shift_y = -np.sin(arrow_ang_calc)*offset
                        
                        head_shift_x = -np.cos(arrow_ang_calc)*headlength
                        head_shift_y = -np.sin(arrow_ang_calc)*headlength
                    elif (3*np.pi/2)<arrow_ang<(2*np.pi):
                        arrow_ang_calc = arrow_ang-(3*np.pi/2)
                        magnitude_x = np.sin(arrow_ang_calc)*magnitude
                        magnitude_y = -np.cos(arrow_ang_calc)*magnitude
                        shift_x = np.sin(arrow_ang_calc)*offset
                        shift_y = -np.cos(arrow_ang_calc)*offset
                        
                        head_shift_x = np.sin(arrow_ang_calc)*headlength
                        head_shift_y = -np.cos(arrow_ang_calc)*headlength
                        
                    elif (arrow_ang==0) or (arrow_ang==np.pi/2) or (arrow_ang==3*np.pi/2) or (arrow_ang==2*np.pi):
                        arrow_ang_calc = arrow_ang
                        magnitude_x = np.cos(arrow_ang_calc)*magnitude
                        magnitude_y = np.sin(arrow_ang_calc)*magnitude
                        shift_x = np.cos(arrow_ang_calc)*offset
                        shift_y = np.sin(arrow_ang_calc)*offset
                        
                        head_shift_x = np.cos(arrow_ang_calc)*headlength
                        head_shift_y = np.sin(arrow_ang_calc)*headlength
                        
                    magnitude_x_map = magnitude
                    if net_flux[k-1]>0:
                        plt.arrow(com_tmp_posX+shift_x+magnitude_x+head_shift_x, com_tmp_posY+shift_y+magnitude_y+head_shift_y, -magnitude_x, -magnitude_y, length_includes_head=False,
                                  head_width=headwidth, head_length=headlength)
                    else:
                        plt.arrow(com_tmp_posX+shift_x, com_tmp_posY+shift_y, magnitude_x, magnitude_y,length_includes_head=False,
                                  head_width=headwidth, head_length=headlength)
                    
                    

                    
                run+=1
                out = "pa" + "{:.0f}".format(peA) +\
                    "_phi" + "{:.0f}".format(intPhi) +\
                    "_ep" + "{:.5f}".format(eps) +\
                    "_fm"
                plt.savefig(outPath2+out+str(j)+'.png')
                g = open(outPath+outTxt, 'a')
                for h in range(1, len(num_sights)):
                        g.write('{0:.2f}'.format(tst).center(15) + ' ')
                        g.write('{0:.0f}'.format(np.amax(clust_size)).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_sights[h-1]).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_sights[h]).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_desorb_los[h-1]).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_absorb_los[h-1]).center(15) + ' ')
                        g.write('{0:.1f}'.format(net_flux[h-1]).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_desorb_tot).center(15) + ' ')
                        g.write('{0:.1f}'.format(num_absorb_tot).center(15)  + '\n')
                g.close()
            
                            
                
            test_run += 1
            edge_parts_prev = np.copy(edge_parts_tot)
            #bulk_parts_prev = np.copy(bulk_parts)
            gas_parts_prev = np.copy(gas_parts_tot)
            
            edge_parts_los_prev = np.copy(edge_parts_los)
            #bulk_parts_prev = np.copy(bulk_parts)
            gas_parts_los_prev = np.copy(gas_parts_los)
            
            
                

                #plt.show()
                
                
                
            
            
