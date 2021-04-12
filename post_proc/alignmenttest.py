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

import sys
from gsd import hoomd
import freud
import numpy as np
import math

#from descartes.patch import PolygonPatch
# Run locally
hoomdPath='/nas/home/njlauers/hoomd-blue/build/'#Users/nicklauersdorf/hoomd-blue/build/'
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
outPath='/pine/scr/n/j/njlauers/alignment/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
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
                
outTxt = 'alignment_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'clust_size'.center(15) + ' ' +\
        'min_ang'.center(15) + ' ' +\
        'max_ang'.center(15) + ' ' +\
        'rad_pop'.center(15) + ' ' +\
        'radius'.center(15) + ' ' +\
        'num_dens'.center(15) + ' ' +\
        'align'.center(15) + '\n')
g.close()
   
with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0#600                  # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.
    radius=np.arange(0,h_box, 1.0*lat)
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
        print(snap.particles)
        print(type(snap.particles))
        print(vars(snap.particles))
        pos = snap.particles.position               # position
        
        #print(snap.particles.velocity)
        #print(np.arctan(snap.particles.velocity[:,1], snap.particles.velocity[:,0]))
        pos[:,-1] = 0.0
        ori = snap.particles.orientation 
        #print(ori)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        #print(ang)
        #print(np.multiply(np.sin(ang/2),(snap.particles.velocity[:,0])/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        #print(np.multiply(np.sin(ang/2),snap.particles.velocity[:,1]/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        #stop
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
            # If sufficient neighbor bins are empty, we have an edge
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
                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] >=min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            testIDs_area[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
                    testIDs_area[ix][iy] = 2
            
            
                        
                        
            count_A_edge=0
            count_B_edge=0
            gas_particle_range=2.0
            gas_r_lim=gas_particle_range*lat
            bulk_particle_range=5.0
            end_loop=0
            steps=0
            num_sights_len=20
            num_sights=np.arange(0, 360+int(360/num_sights_len),int(360/num_sights_len))
            
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
                        if ix!=com_x_ind and iy!=com_y_ind:
                            if dif_x_min>0 and dif_x_max>0:
                                if dif_y_min>0 and dif_y_max>0:
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=1
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=1
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(min_quad-1)*90
                                elif dif_y_min<0 and dif_y_max<0:
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=4
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=4
                                    min_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                                elif (dif_y_min<0 and dif_y_max>0) or (dif_y_min>0 and dif_y_max<0):
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=4
                                    max_ref=np.array([x_min_ref, y_max_ref])
                                    max_quad=1
                                    min_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(max_quad-1)*90
                            elif dif_x_min<0 and dif_x_max<0:
                                if (dif_y_min)>0 and (dif_y_max)>0:
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=2
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=2
                                    min_angle=((np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*(180/np.pi))+(min_quad-1)*90
                                    max_angle=((np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*(180/np.pi))+(max_quad-1)*90
                                    
                                    
                                elif dif_y_min<0 and dif_y_max<0:
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=3
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(max_quad-1)*90
                                elif (dif_y_min<0 and dif_y_max>0) or (dif_y_min>0 and dif_y_max<0):
                                    min_ref=np.array([x_max_ref, y_min_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=2
                                    min_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90

                            elif (dif_x_min<0 and dif_x_max>0) or (dif_x_min>0 and dif_x_max<0):
                                if dif_y_min>0 and dif_y_max>0:
                                    min_ref=np.array([x_min_ref, y_min_ref])
                                    min_quad=2
                                    max_ref=np.array([x_max_ref, y_min_ref])
                                    max_quad=1
                                    min_angle=(np.arctan(np.abs(dif_x_min)/np.abs(dif_y_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_y_min)/np.abs(dif_x_max)))*180/np.pi+(max_quad-1)*90
                                elif dif_y_min<0 and dif_y_max<0:
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=3
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=4
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                                elif (dif_y_min<0 and dif_y_max>0) or (dif_y_min>0 and dif_y_max<0):
                                    min_ref=np.array([x_min_ref, y_max_ref])
                                    min_quad=1
                                    max_ref=np.array([x_max_ref, y_max_ref])
                                    max_quad=1
                                    min_angle=(np.arctan(np.abs(dif_y_max)/np.abs(dif_x_min)))*180/np.pi+(min_quad-1)*90
                                    max_angle=(np.arctan(np.abs(dif_x_max)/np.abs(dif_y_max)))*180/np.pi+(max_quad-1)*90
                            if num_sights[k-1]<=min_angle:
                                if min_angle<=num_sights[k]:
                                    losBin[ix][iy]=1
                            if num_sights[k-1]<=max_angle:
                                if max_angle<=num_sights[k]:
                                    losBin[ix][iy]=1
                            
                        elif ix==com_x_ind and iy==com_y_ind:
                            losBin[ix][iy]=1

                rad_bin=np.zeros(len(radius)-1)
                
                align_rad=np.zeros(len(radius)-1)
                for ix in range(0, len(occParts)):
                    for iy in range(0, len(occParts)):
                        if losBin[ix][iy]==1:
                            if len(binParts[ix][iy])!=0:
                                
                                for h in range(0,len(binParts[ix][iy])):
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

                                    if (difx)>0:
                                        if (dify)>0:
                                            part_quad=1
                                            part_angle=(np.arctan(np.abs(dify)/np.abs(difx)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)<0:
                                            part_quad=4
                                            part_angle=(np.arctan(np.abs(difx)/np.abs(dify)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)==0:
                                            part_angle=0
                                    elif (difx)<0:
                                        if (dify)>0:
                                            part_quad=2
                                            part_angle=(np.arctan(np.abs(difx)/np.abs(dify)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)<0:
                                            part_quad=3
                                            part_angle=(np.arctan(np.abs(dify)/np.abs(difx)))*180/np.pi+(part_quad-1)*90
                                        elif (dify)==0:
                                            part_angle=180
                                    elif (difx)==0:
                                        if (dify)>0:
                                            part_angle=90
                                        elif (dify)<0:
                                            part_angle=270
                                        elif (dify)==0:
                                            part_angle=(num_sights[k]+num_sights[k-1])/2
                                    if num_sights[k-1]<=part_angle<=num_sights[k]:

                                        pos_new_x=np.append(pos_new_x, x_pos)
                                        pos_new_y=np.append(pos_new_y, y_pos)
                                        
                                        difr=(difx**2+dify**2)**0.5
                                        for l in range(1,len(radius)):
                                            if radius[l-1]<=difr<=radius[l]:
                                                rad_bin[l-1]+=1
                                                #rad_bin_ind[l-1].append(typParts[ix][iy][h])
                                                px = np.sin(ang[binParts[ix][iy][h]])
                                                px = np.sin(ang[binParts[ix][iy][h]])
                                                py = -np.cos(ang[binParts[ix][iy][h]])
                                                
                                                
                                                
                                                r_dot_p = (-difx * px) + (-dify * py)
                                                align=r_dot_p/difr
                                                align_rad[l-1]+=align
                                                
                
                num_dens=np.zeros(len(radius)-1)
                align_tot=np.zeros(len(radius)-1)
                for f in range(0, len(align_rad)):
                    if rad_bin[f]!=0:
                        align_tot[f]=(align_rad[f]/rad_bin[f])
                for f in range(0, len(radius)-1):
                    if rad_bin[f]!=0:
                        num_dens[f]=rad_bin[f]/area_slice[f]
                        
                g = open(outPath+outTxt, 'a')
                for h in range(0, len(rad_bin)):
                    g.write('{0:.3f}'.format(tst).center(15) + ' ')
                    g.write('{0:.3f}'.format(np.amax(clust_size)).center(15) + ' ')
                    g.write('{0:.2f}'.format(num_sights[k-1]).center(15) + ' ')
                    g.write('{0:.2f}'.format(num_sights[k]).center(15) + ' ')
                    g.write('{0:.6f}'.format(rad_bin[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(radius[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(num_dens[h]).center(15) + ' ')
                    g.write('{0:.6f}'.format(align_tot[h]).center(15) + '\n')
                g.close()