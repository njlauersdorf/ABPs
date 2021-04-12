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
from scipy.interpolate import interp1d

#from descartes.patch import PolygonPatch
# Run locally
hoomdPath='Users/nicklauersdorf/hoomd-blue/build/'#'/nas/home/njlauers/hoomd-blue/build/'#Users/nicklauersdorf/hoomd-blue/build/'
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
outPath='/Volumes/External/whingdingdilly-master/ipython/clusters_soft/alignment_binary/'#'/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/alignment_sparse/rerun2/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
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
                    
outTxt = 'alignment_' + outF + '.txt'
g = open(outPath+outTxt, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'clust_size'.center(15) + ' ' +\
        'min_ang'.center(15) + ' ' +\
        'max_ang'.center(15) + ' ' +\
        'radius'.center(15) + ' ' +\
        'rad_bin'.center(15) + ' ' +\
        'rad_bin_0'.center(15) + ' ' +\
        'rad_bin_1'.center(15) + ' ' +\
        'num_dens'.center(15) + ' ' +\
        'num_dens_0'.center(15) + ' ' +\
        'num_dens_1'.center(15) + ' ' +\
        'align'.center(15) + ' ' +\
        'align_0'.center(15) + ' ' +\
        'align_1'.center(15) + '\n')
g.close()
   
#g.close()
int_len=3.0
fracd=3/int_len
fracd=1.0

if eps==0.0001:
        new_cut_off=int(4*fracd)
elif eps==0.001:
        new_cut_off=int(6*fracd)
elif eps==0.01:
        new_cut_off=int(8*fracd)
elif eps==0.1:
        new_cut_off=int(10*fracd)
elif eps==1.0:
        new_cut_off=int(12*fracd)
        
with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0                  # first frame to process
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

                rad_bin=np.zeros(len(radius)-1)
                rad_0_bin=np.zeros(len(radius)-1)
                rad_1_bin=np.zeros(len(radius)-1)
                align_rad=np.zeros(len(radius)-1)
                align_0_rad=np.zeros(len(radius)-1)
                align_1_rad=np.zeros(len(radius)-1)
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
                                                px = np.sin(ang[binParts[ix][iy][h]])
                                                px = np.sin(ang[binParts[ix][iy][h]])
                                                py = -np.cos(ang[binParts[ix][iy][h]])
                                                
                                                
                                                
                                                r_dot_p = (-difx * px) + (-dify * py)
                                                align=r_dot_p/difr

                                                align_rad[l-1]+=align
                                                if typ[binParts[ix][iy][h]]==0:
                                                    align_0_rad[l-1]+=align
                                                    rad_0_bin[l-1]+=1
                                                elif typ[binParts[ix][iy][h]]==1:
                                                    align_1_rad[l-1]+=align
                                                    rad_1_bin[l-1]+=1
                                #plt.figure()          
                #plt.scatter(pos_new_x-com_tmp_posX, pos_new_y-com_tmp_posY, c='b', s=1)   
                #plt.scatter(0, 0, c='r', s=5)  
                #plt.ylim((-h_box, 10.0))
                #plt.xlim((-10.0, 10.0))
                #plt.draw()
                #plt.pause(0.1)
                #stop
                #plt.pause(0.1)
                                #plt.close()
                radial_steps=radius[new_cut_off:]
                radial_steps_very_orig=radius      
                                                
                num_dens=np.zeros(len(radius)-1)
                num_0_dens=np.zeros(len(radius)-1)
                num_1_dens=np.zeros(len(radius)-1)
                align_tot=np.zeros(len(radius)-1)
                align_0_tot=np.zeros(len(radius)-1)
                align_1_tot=np.zeros(len(radius)-1)
                for f in range(0, len(align_rad)):
                    if rad_bin[f]!=0:
                        align_tot[f]=(align_rad[f]/rad_bin[f])
                        align_0_tot[f]=(align_0_rad[f]/rad_0_bin[f])
                        align_1_tot[f]=(align_1_rad[f]/rad_1_bin[f])
                for f in range(0, len(radius)-1):
                    if rad_bin[f]!=0:
                        num_dens[f]=rad_bin[f]/area_slice[f]
                        num_0_dens[f]=rad_0_bin[f]/area_slice[f]
                        num_1_dens[f]=rad_1_bin[f]/area_slice[f]
                
                
                
                num_dens_very_orig=num_dens
                num_dens_0_very_orig=num_0_dens
                num_dens_1_very_orig=num_1_dens
                
                num_dens_tot=smooth(num_dens[new_cut_off:],3)
                num_dens_0=smooth(num_0_dens[new_cut_off:],3)
                num_dens_1=smooth(num_1_dens[new_cut_off:],3)

                if num_dens[10]>1.0:
                    
                    align_very_orig=align_tot
                    align_0_very_orig=align_0_tot
                    align_1_very_orig=align_1_tot
                    
                    
                    align_orig=align_tot[new_cut_off:]
                    align_0_orig=align_0_tot[new_cut_off:]
                    align_1_orig=align_1_tot[new_cut_off:]
                    
                    align_orig=smooth(align_orig,3)
                    align_0_orig=smooth(align_0_orig,3)
                    align_1_orig=smooth(align_1_orig,3)
                    
                    alpha_orig=align_orig*num_dens_tot
                    alpha_0_orig=align_0_orig*num_dens_0
                    alpha_1_orig=align_1_orig*num_dens_1
                    
                    alpha_very_orig=align_very_orig*num_dens_very_orig
                    alpha_0_very_orig=align_0_very_orig*num_dens_0_very_orig
                    alpha_1_very_orig=align_1_very_orig*num_dens_1_very_orig
                
                    rad_end_new=np.where(alpha_orig==np.max(alpha_orig))[0][0]
                    
                    if np.max(alpha_orig)>=0.2:
                        c=edge_end_funct(alpha_orig, radial_steps)
                        if c!=0:
                            d=edge_begin_funct(alpha_orig, radial_steps)
                            if d!=0:
                                end_rad=radial_steps[c]
                                begin_rad=radial_steps[d]
                                
                                end_rad_orig_ind = find_nearest(radial_steps_very_orig, end_rad)
                                end_rad_point = radial_steps_very_orig[end_rad_orig_ind]
                                
                                begin_rad_orig_ind = find_nearest(radial_steps_very_orig, begin_rad)
                                begin_rad_point = radial_steps_very_orig[begin_rad_orig_ind]
                                if begin_rad_point<end_rad_point:
                                    norm_rad_very_orig=radial_steps_very_orig/end_rad_point
                                    norm_rad_very_orig_app = np.append(norm_rad_very_orig, 0.0)
                                    num_dens_very_orig_app = np.append(num_dens_very_orig, num_dens_very_orig[0])
                                    num_dens_0_very_orig_app = np.append(num_dens_0_very_orig, num_dens_0_very_orig[0])
                                    num_dens_1_very_orig_app = np.append(num_dens_1_very_orig, num_dens_1_very_orig[0])
                                    
                                    align_very_orig_app = np.append(align_very_orig, 0.0)
                                    align_0_very_orig_app = np.append(align_0_very_orig, 0.0)
                                    align_1_very_orig_app = np.append(align_1_very_orig, 0.0)
                                    
                                    alpha_very_orig_app = align_very_orig_app * num_dens_very_orig_app
                                    alpha_0_very_orig_app = align_0_very_orig_app * num_dens_0_very_orig_app
                                    alpha_1_very_orig_app = align_1_very_orig_app * num_dens_1_very_orig_app
                                    
                                    #norm_rad_end=np.where(norm_rad_very_orig_app==1.0)[0][0]
                                    
                                    norm_rad=norm_rad_very_orig_app#[:norm_rad_end+1]
                                    if np.max(norm_rad)>=1.0:
                                        
                                        int_len2=1.0/10000.0
                                        
                                        rad_renew=np.arange(0.0, 1.0+int_len2, int_len2)
                
                                        alpha_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_very_orig_app, kind='cubic')
                                        alpha_0_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_0_very_orig_app, kind='cubic')
                                        alpha_1_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_1_very_orig_app, kind='cubic')
                                        
                                        
                                        align_int_renew = interp1d(norm_rad_very_orig_app, align_very_orig_app, kind='cubic')
                                        align_0_int_renew = interp1d(norm_rad_very_orig_app, align_0_very_orig_app, kind='cubic')
                                        align_1_int_renew = interp1d(norm_rad_very_orig_app, align_1_very_orig_app, kind='cubic')
                                        
                                        num_dens_int_renew = interp1d(norm_rad_very_orig_app, num_dens_very_orig_app, kind='cubic')
                                        num_dens_0_int_renew = interp1d(norm_rad_very_orig_app, num_dens_0_very_orig_app, kind='cubic')
                                        num_dens_1_int_renew = interp1d(norm_rad_very_orig_app, num_dens_1_very_orig_app, kind='cubic')
                                        
                                        alpha_n_int_rerenew=alpha_n_int_renew(rad_renew)
                                        alpha_0_n_int_rerenew=alpha_0_n_int_renew(rad_renew)
                                        alpha_1_n_int_rerenew=alpha_1_n_int_renew(rad_renew)
                                        
                                        align_int_rerenew=align_int_renew(rad_renew)
                                        align_0_int_rerenew=align_0_int_renew(rad_renew)
                                        align_1_int_rerenew=align_1_int_renew(rad_renew)
                                        
                                        num_dens_int_rerenew=num_dens_int_renew(rad_renew)
                                        num_dens_0_int_rerenew=num_dens_0_int_renew(rad_renew)
                                        num_dens_1_int_rerenew=num_dens_1_int_renew(rad_renew)
                                        
                                        
                                        alpha_n_int_num+=1
                                        
                                        align_tot+=align_int_rerenew
                                        align_0_tot+=align_0_int_rerenew
                                        align_1_tot+=align_1_int_rerenew
                                        
                                        num_dens_tot+=num_dens_int_rerenew
                                        num_dens_0_tot+=num_dens_0_int_rerenew
                                        num_dens_1_tot+=num_dens_1_int_rerenew
                                        
                                        int_end_avg+=end_rad_point
                                        int_begin_avg+=begin_rad_point
                                        
                                        rad_min=radial_steps_very_orig-begin_rad_point
                                        rad_end_min = end_rad_point - begin_rad_point
                                        norm_rad_very_orig2 = rad_min / rad_end_min
                                        norm_rad_begin2 = np.where(norm_rad_very_orig2==0.0)[0][0]
                                        norm_rad_end2 = np.where(norm_rad_very_orig2==1.0)[0][0]
            
                                        norm_rad3=norm_rad_very_orig2[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        align_int_rerenew2 = align_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        align_0_int_rerenew2 = align_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        align_1_int_rerenew2 = align_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        num_dens_int_rerenew2 = num_dens_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        num_dens_0_int_rerenew2 = num_dens_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        num_dens_1_int_rerenew2 = num_dens_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        alpha_n_int_rerenew2 = alpha_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        alpha_0_n_int_rerenew2 = alpha_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        alpha_1_n_int_rerenew2 = alpha_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                               
                                        int_len3=norm_rad3.max()/10000.0
                                        
                                        rad_renew3=np.arange(0.0, 1.0+int_len3, int_len3)
                
                                        alpha_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_very_orig, kind='cubic')
                                        alpha_0_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_0_very_orig, kind='cubic')
                                        alpha_1_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_1_very_orig, kind='cubic')
                                        
                                        align_int_renew2 = interp1d(norm_rad_very_orig2, align_very_orig, kind='cubic')
                                        align_0_int_renew2 = interp1d(norm_rad_very_orig2, align_0_very_orig, kind='cubic')
                                        align_1_int_renew2 = interp1d(norm_rad_very_orig2, align_1_very_orig, kind='cubic')
                                        
                                        num_dens_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_very_orig, kind='cubic')
                                        num_dens_0_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_0_very_orig, kind='cubic')
                                        num_dens_1_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_1_very_orig, kind='cubic')
                                        
                                        alpha_n_int_rerenew2=alpha_n_int_renew2(rad_renew3)
                                        alpha_0_n_int_rerenew2=alpha_0_n_int_renew2(rad_renew3)
                                        alpha_1_n_int_rerenew2=alpha_1_n_int_renew2(rad_renew3)
                                        
                                        align_int_rerenew2=align_int_renew2(rad_renew3)
                                        align_0_int_rerenew2=align_0_int_renew2(rad_renew3)
                                        align_1_int_rerenew2=align_1_int_renew2(rad_renew3)
                                        
                                        num_dens_int_rerenew2=num_dens_int_renew2(rad_renew3)
                                        num_dens_0_int_rerenew2=num_dens_0_int_renew2(rad_renew3)
                                        num_dens_1_int_rerenew2=num_dens_1_int_renew2(rad_renew3)
                                        
                                        
                                        
                                        
                                        align_tot2+=align_int_rerenew2
                                        align_0_tot2+=align_0_int_rerenew2
                                        align_1_tot2+=align_1_int_rerenew2
                                        
                                        num_dens_tot2+=num_dens_int_rerenew2
                                        num_dens_0_tot2+=num_dens_0_int_rerenew2
                                        num_dens_1_tot2+=num_dens_1_int_rerenew2
                                    
                                    else:
                                        pass
                                    
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass

                    else:
                        pass
                
                
                
                

    
    time_list = []
    print(params['peA'][i])
    print(params['phi'][i])
    min_ang_list=[]
    max_ang_list=[]
    ang_dif=(all_dens[i]['max_ang'][0]-all_dens[i]['min_ang'][0])
    angs=np.arange(0,360+ang_dif,ang_dif)
    for j in range(0,len(all_dens[i]['tauB'])):
        if all_dens[i]['tauB'][j] in time_list:
            pass
        else:
            time_list.append(all_dens[i]['tauB'][j])
    times=len(time_list)
    times_one=len(np.where((all_dens[i]['tauB']==all_dens[i]['tauB'][0]))[0])
    data=len(all_dens)
    rad_len=np.where((all_dens[i]['tauB']==all_dens[i]['tauB'][0]) & (all_dens[i]['min_ang']==all_dens[i]['min_ang'][0]))[0]
    radial_steps=all_dens[i]['radius'][new_cut_off:(len(rad_len))]
    radial_steps_very_orig=all_dens[i]['radius'][:(len(rad_len))]
    a=[]

    for j in range(0,times):
        a.append(all_dens[i]['clust_size'][times_one*j])

    radial_steps=pd.Series.to_numpy(radial_steps)
    radial_steps_very_orig=pd.Series.to_numpy(radial_steps_very_orig)
    #radial_steps=np.insert(radial_steps,0, 0) 
    alpha_tot=0
    alpha_0_tot=0
    alpha_1_tot=0
    
    align_tot_orig=0
    align_0_tot_orig=0
    align_1_tot_orig=0
    
    num_dens_tot_orig=0
    num_dens_0_tot_orig=0
    num_dens_1_tot_orig=0
    
    align_tot=0
    align_0_tot=0
    align_1_tot=0
    
    align_tot2=0
    align_0_tot2=0
    align_1_tot2=0
    
    num_dens_tot2=0
    num_dens_0_tot2=0
    num_dens_1_tot2=0
    
    num_dens_tot=0
    num_dens_0_tot=0
    num_dens_1_tot=0
    
    alpha_n_int_num=0
    clust_rad=0
    int_width_avg=0
    int_begin_avg=0
    int_end_avg=0
    for j in range(0,times-1):
        for k in range(0,len(angs)):

                
                
                num_dens_very_orig=pd.Series.to_numpy(all_dens[i]['num_dens'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                num_dens_0_very_orig=pd.Series.to_numpy(all_dens[i]['num_dens_0'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                num_dens_1_very_orig=pd.Series.to_numpy(all_dens[i]['num_dens_1'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                if num_dens[10]>1.0:
                    #if j==10:
                    #    if k==7:
                    #        print(all_dens[i]['radius'][7+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))-5])
                    #num_dens=np.insert(num_dens,0, num_dens[0], axis=0) 
                    
                    align_very_orig=pd.Series.to_numpy(all_dens[i]['align'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    align_0_very_orig=pd.Series.to_numpy(all_dens[i]['align_0'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    align_1_very_orig=pd.Series.to_numpy(all_dens[i]['align_1'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    
                    align_orig=pd.Series.to_numpy(all_dens[i]['align'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    align_0_orig=pd.Series.to_numpy(all_dens[i]['align_0'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    align_1_orig=pd.Series.to_numpy(all_dens[i]['align_1'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    
                    
                    
                    #align_orig=np.insert(align_orig,0, align_orig[0], axis=0) 
                    align_orig=smooth(align_orig,3)
                    align_0_orig=smooth(align_0_orig,3)
                    align_1_orig=smooth(align_1_orig,3)

                    alpha_orig=align_orig*num_dens
                    alpha_0_orig=align_0_orig*num_dens_0
                    alpha_1_orig=align_1_orig*num_dens_1
                    
                    alpha_very_orig=align_very_orig*num_dens_very_orig
                    alpha_0_very_orig=align_0_very_orig*num_dens_0_very_orig
                    alpha_1_very_orig=align_1_very_orig*num_dens_1_very_orig
                    
                    #alpha_n_int = interp1d(radial_steps, alpha_orig, kind='cubic')
                    #alpha_n_int_new=alpha_n_int(rad_new)
                    
                    rad_end_new=np.where(alpha_orig==np.max(alpha_orig))[0][0]
                    if np.max(alpha_orig)>=0.2:
                        c=edge_end_funct(alpha_orig, radial_steps)
                        if c!=0:
                            d=edge_begin_funct(alpha_orig, radial_steps)
                            if d!=0:
                                end_rad=radial_steps[c]
                                begin_rad=radial_steps[d]
                                
                                end_rad_orig_ind = find_nearest(radial_steps_very_orig, end_rad)
                                end_rad_point = radial_steps_very_orig[end_rad_orig_ind]
                                
                                begin_rad_orig_ind = find_nearest(radial_steps_very_orig, begin_rad)
                                begin_rad_point = radial_steps_very_orig[begin_rad_orig_ind]
                                if begin_rad_point<end_rad_point:
                                    '''
                                    plt.plot(radial_steps_very_orig[new_cut_off:], alpha_very_orig[new_cut_off:])
                                    plt.plot(radial_steps_very_orig[begin_rad_orig_ind:end_rad_orig_ind+1], alpha_very_orig[begin_rad_orig_ind:end_rad_orig_ind+1], label='interface')
                                    file_name_final='align_plot_pe_'+str(params['pe'][i])+'_phi_'+str(params['phi'][i])+'_eps_'+str(params['eps'][i])+'_time_'+str(time_list[j])+'_ang_'+str(k*18)+'.png'
                                    plt.ylabel(r'$\alpha(r)n(r)')
                                    plt.xlabel('r')
                                    plt.legend()
                                    plt.show()
                                    '''
                                    
                                    #plt.savefig('/Volumes/External/whingdingdilly-master/ipython/clusters_soft/align_data/'+file_name_final)
                                    
                                    #plt.close()
                                    '''
                                    norm_rad_very_orig=radial_steps_very_orig/end_rad_point
                                    norm_rad_very_orig_app = np.append(norm_rad_very_orig, 0.0)
                                    num_dens_very_orig_app = np.append(num_dens_very_orig, num_dens_very_orig[0])
                                    align_very_orig_app = np.append(align_very_orig, 0.0)
                                    alpha_very_orig_app = align_very_orig_app * num_dens_very_orig_app
                                    
                                    norm_rad_end=np.where(norm_rad_very_orig_app==1.0)[0][0]
                                    
                                    norm_rad=norm_rad_very_orig_app[:norm_rad_end+1]
                                                
                                    int_len2=norm_rad.max()/10000.0
                                    
                                    rad_renew=np.arange(0.0, 1.0+int_len2, int_len2)
            
                                    alpha_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_very_orig_app, kind='cubic')
                                    align_int_renew = interp1d(norm_rad_very_orig_app, align_very_orig_app, kind='cubic')
                                    num_dens_int_renew = interp1d(norm_rad_very_orig_app, num_dens_very_orig_app, kind='cubic')
                                    alpha_n_int_rerenew=alpha_n_int_renew(rad_renew)
                                    align_int_rerenew=align_int_renew(rad_renew)
                                    num_dens_int_rerenew=num_dens_int_renew(rad_renew)
                                    '''
                                    
                                    
                                    norm_rad_very_orig=radial_steps_very_orig/end_rad_point
                                    norm_rad_very_orig_app = np.append(norm_rad_very_orig, 0.0)
                                    num_dens_very_orig_app = np.append(num_dens_very_orig, num_dens_very_orig[0])
                                    num_dens_0_very_orig_app = np.append(num_dens_0_very_orig, num_dens_0_very_orig[0])
                                    num_dens_1_very_orig_app = np.append(num_dens_1_very_orig, num_dens_1_very_orig[0])
                                    
                                    align_very_orig_app = np.append(align_very_orig, 0.0)
                                    align_0_very_orig_app = np.append(align_0_very_orig, 0.0)
                                    align_1_very_orig_app = np.append(align_1_very_orig, 0.0)
                                    
                                    alpha_very_orig_app = align_very_orig_app * num_dens_very_orig_app
                                    alpha_0_very_orig_app = align_0_very_orig_app * num_dens_0_very_orig_app
                                    alpha_1_very_orig_app = align_1_very_orig_app * num_dens_1_very_orig_app
                                    
                                    #norm_rad_end=np.where(norm_rad_very_orig_app==1.0)[0][0]
                                    
                                    norm_rad=norm_rad_very_orig_app#[:norm_rad_end+1]
                                    if np.max(norm_rad)>=1.0:
                                        int_len2=1.0/10000.0
                                        
                                        rad_renew=np.arange(0.0, 1.0+int_len2, int_len2)
                
                                        alpha_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_very_orig_app, kind='cubic')
                                        alpha_0_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_0_very_orig_app, kind='cubic')
                                        alpha_1_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_1_very_orig_app, kind='cubic')
                                        
                                        
                                        align_int_renew = interp1d(norm_rad_very_orig_app, align_very_orig_app, kind='cubic')
                                        align_0_int_renew = interp1d(norm_rad_very_orig_app, align_0_very_orig_app, kind='cubic')
                                        align_1_int_renew = interp1d(norm_rad_very_orig_app, align_1_very_orig_app, kind='cubic')
                                        
                                        num_dens_int_renew = interp1d(norm_rad_very_orig_app, num_dens_very_orig_app, kind='cubic')
                                        num_dens_0_int_renew = interp1d(norm_rad_very_orig_app, num_dens_0_very_orig_app, kind='cubic')
                                        num_dens_1_int_renew = interp1d(norm_rad_very_orig_app, num_dens_1_very_orig_app, kind='cubic')
                                        
                                        alpha_n_int_rerenew=alpha_n_int_renew(rad_renew)
                                        alpha_0_n_int_rerenew=alpha_0_n_int_renew(rad_renew)
                                        alpha_1_n_int_rerenew=alpha_1_n_int_renew(rad_renew)
                                        
                                        align_int_rerenew=align_int_renew(rad_renew)
                                        align_0_int_rerenew=align_0_int_renew(rad_renew)
                                        align_1_int_rerenew=align_1_int_renew(rad_renew)
                                        
                                        num_dens_int_rerenew=num_dens_int_renew(rad_renew)
                                        num_dens_0_int_rerenew=num_dens_0_int_renew(rad_renew)
                                        num_dens_1_int_rerenew=num_dens_1_int_renew(rad_renew)
                                        
                                        
                                        alpha_n_int_num+=1
                                        
                                        align_tot+=align_int_rerenew
                                        align_0_tot+=align_0_int_rerenew
                                        align_1_tot+=align_1_int_rerenew
                                        
                                        num_dens_tot+=num_dens_int_rerenew
                                        num_dens_0_tot+=num_dens_0_int_rerenew
                                        num_dens_1_tot+=num_dens_1_int_rerenew
                                        
                                        int_end_avg+=end_rad_point
                                        int_begin_avg+=begin_rad_point
                                        
                                        rad_min=radial_steps_very_orig-begin_rad_point
                                        rad_end_min = end_rad_point - begin_rad_point
                                        norm_rad_very_orig2 = rad_min / rad_end_min
                                        norm_rad_begin2 = np.where(norm_rad_very_orig2==0.0)[0][0]
                                        norm_rad_end2 = np.where(norm_rad_very_orig2==1.0)[0][0]
            
                                        norm_rad3=norm_rad_very_orig2[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        align_int_rerenew2 = align_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        align_0_int_rerenew2 = align_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        align_1_int_rerenew2 = align_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        num_dens_int_rerenew2 = num_dens_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        num_dens_0_int_rerenew2 = num_dens_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        num_dens_1_int_rerenew2 = num_dens_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                        alpha_n_int_rerenew2 = alpha_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        alpha_0_n_int_rerenew2 = alpha_0_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        alpha_1_n_int_rerenew2 = alpha_1_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                               
                                        int_len3=norm_rad3.max()/10000.0
                                        
                                        rad_renew3=np.arange(0.0, 1.0+int_len3, int_len3)
                
                                        alpha_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_very_orig, kind='cubic')
                                        alpha_0_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_0_very_orig, kind='cubic')
                                        alpha_1_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_1_very_orig, kind='cubic')
                                        
                                        align_int_renew2 = interp1d(norm_rad_very_orig2, align_very_orig, kind='cubic')
                                        align_0_int_renew2 = interp1d(norm_rad_very_orig2, align_0_very_orig, kind='cubic')
                                        align_1_int_renew2 = interp1d(norm_rad_very_orig2, align_1_very_orig, kind='cubic')
                                        
                                        num_dens_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_very_orig, kind='cubic')
                                        num_dens_0_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_0_very_orig, kind='cubic')
                                        num_dens_1_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_1_very_orig, kind='cubic')
                                        
                                        alpha_n_int_rerenew2=alpha_n_int_renew2(rad_renew3)
                                        alpha_0_n_int_rerenew2=alpha_0_n_int_renew2(rad_renew3)
                                        alpha_1_n_int_rerenew2=alpha_1_n_int_renew2(rad_renew3)
                                        
                                        align_int_rerenew2=align_int_renew2(rad_renew3)
                                        align_0_int_rerenew2=align_0_int_renew2(rad_renew3)
                                        align_1_int_rerenew2=align_1_int_renew2(rad_renew3)
                                        
                                        num_dens_int_rerenew2=num_dens_int_renew2(rad_renew3)
                                        num_dens_0_int_rerenew2=num_dens_0_int_renew2(rad_renew3)
                                        num_dens_1_int_rerenew2=num_dens_1_int_renew2(rad_renew3)
                                        
                                        
                                        
                                        
                                        align_tot2+=align_int_rerenew2
                                        align_0_tot2+=align_0_int_rerenew2
                                        align_1_tot2+=align_1_int_rerenew2
                                        
                                        num_dens_tot2+=num_dens_int_rerenew2
                                        num_dens_0_tot2+=num_dens_0_int_rerenew2
                                        num_dens_1_tot2+=num_dens_1_int_rerenew2
                                    
                                    else:
                                        pass
                                    
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass

                    else:
                        pass
    pe_net = params['peA'][i]*(params['xA'][i]/100) + params['peB'][i]*(1.0-(params['xA'][i]/100))
    lat=getLat(pe_net,params['eps'][i])
    '''
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.plot(rad_renew, num_dens_tot/alpha_n_int_num, label='n(r)')
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('all_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('align_alpha_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('align_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('alpha_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, num_dens_tot/alpha_n_int_num, label='n(r)')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('num_dens_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    '''
    #g = open(outTxt, 'a')
    #g.write('{0:.0f}'.format(params['pe'][i]).center(15) + ' ')
    #g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ')
    #g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
    #g.write('{0:.6f}'.format(lat).center(15) + ' ')
    #g.write('{0:.6f}'.format(int_width_tot).center(15) + '\n')
    #g.close()
    g = open(outTxt2, 'a') # write file headings
    for m in range(0,len(num_dens_tot)):
        g.write('{0:.0f}'.format(params['peA'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['peB'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['xA'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ') 
        g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
        g.write('{0:.6f}'.format(rad_renew[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_0_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_1_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_0_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_1_tot[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(alpha_n_int_num).center(15) + ' ')
        g.write('{0:.6f}'.format(lat).center(15) + ' ')
        g.write('{0:.6f}'.format(int_begin_avg).center(15) + ' ')
        g.write('{0:.6f}'.format(int_end_avg).center(15) + '\n')
    g.close()
    
    g = open(outTxt3, 'a') # write file headings
    for m in range(0,len(num_dens_tot)):
        g.write('{0:.0f}'.format(params['peA'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['peB'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['xA'][i]).center(15) + ' ')
        g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ') 
        g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
        g.write('{0:.6f}'.format(rad_renew3[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_0_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(num_dens_1_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_0_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(align_1_tot2[m]).center(15) + ' ')
        g.write('{0:.6f}'.format(alpha_n_int_num).center(15) + ' ')
        g.write('{0:.6f}'.format(lat).center(15) + ' ')
        g.write('{0:.6f}'.format(int_begin_avg).center(15) + ' ')
        g.write('{0:.6f}'.format(int_end_avg).center(15) + '\n')
    g.close()
    
    
            #plt.plot(rad_new[:len(rad_new)-1], a['deriv'], label='derivative')
            #plt.plot(rad_new, alpha_n_int_new, label='not deriv')
            
            #plt.plot(rad_new[a['begin']:a['end']+1], a['deriv'][a['begin']:a['end']+1], label='derivative')
            #plt.plot(rad_new[a['begin']:a['end']+1], alpha_n_int_new[a['begin']:a['end']+1], label='not deriv')
            #plt.legend()
            #plt.show()
            #plt.plot(radial_steps, align_orig*num_dens)
            #plt.plot(radial_steps[:edge_end_int+1]/radial_steps[:edge_end_int], num_dens[edge_begin_int:edge_end_int+1]*align_orig[edge_begin_int:edge_end_int+1])

            #plt.show()
            #print(norm_rad)
            #stop
    '''
            rad_clust=rad_new[:a['end']]
            plt.plot(rad_new[:len(rad_new)-1], a['deriv'], label='derivative')
            plt.plot(rad_new, alpha_n_int_new, label='not deriv')
            plt.legend()
            plt.show()
            
            
            stop
            print(rad_new[alpha_n_deriv_min_ind+alpha_n_deriv_min_ind_done+1])
            print(rad_new[alpha_n_deriv_min_ind_done])
            print(rad_new[alpha_n_deriv_max_ind])
            print(rad_new[alpha_n_deriv_max_ind_done+alpha_n_deriv_max_ind])

            plt.plot(rad_new, alpha_n_int_new)
            plt.plot(rad_new[:len(rad_new)-1], smooth(alpha_n_deriv,4))
            plt.show()
            stop
            stop
            print(align_deriv_max_ind[0])
            stop
            begin_int_1=np.max(align_deriv_max_ind[0])
            
            #Define end_int as the last location of radius array
            
            #Only use negative slopes
            align_deriv_ind=np.where(align_deriv<0)
            
            #Find negative slopes whose magnitudes are greater than 10% of the maximum slope
            align_deriv_max_ind=np.where(np.abs(align_deriv[align_deriv_ind])>0.05*np.max(align_deriv))
            
            
            #Find the latest occurance of slope that meets above 2 criteria
            print(align_deriv_max_ind[0])
            begin_int_1=np.max(align_deriv_max_ind[0])
    '''
print('done!')
#    print(len(all_dens[i]['tauB']))
#    print(len(all_dens[i]['clust_size']))
#    print(len(all_dens[i]['min_ang']))
#    print(len(all_dens[i]['max_ang']))
#    print(len(all_dens[i]['radius']))
#    print(len(all_dens[i]['num_dens']))
#    print(len(all_dens[i]['align']))
#    print(all_dens[0][headers[1]][0])
#print(all_dens[0][headers[2]][0])
#print(headers)
            