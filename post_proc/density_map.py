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
import os
from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib import cm

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
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/dens_video/'

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




def ljForce(r, eps, sigma=1.):
    div = (sigma/r)
    dU = (24. * eps / sigma) * ((2*(div**13)) - (div)**7)
    return dU

# # Lennard-Jones pressure
# def ljPress(r, eps, sigma=1.):
#     phiCP = np.pi / (2. * np.sqrt(3.))
#     div = (sigma/r)
#     dU = (24. * eps / r) * ((2.*(div**12.)) - (div)**6.)
#     # This is just pressure divided by the area of a particle
# #     return (12. * dU / (np.pi * r))
#     return (12. * dU / (np.pi * r * phiCP))

def ljPress(r, pe, eps, sigma=1.):
    phiCP = np.pi / (2. * np.sqrt(3.))
    # This is off by a factor of 1.2...
    ljF = avgCollisionForce(pe)
    return (2. *np.sqrt(3) * ljF / r)
    
def avgCollisionForce(pe, power=1.):
    '''Computed from the integral of possible angles'''
    peCritical = 40.
    if pe < peCritical:
        pe = 0
    else:
        pe -= peCritical
    magnitude = 6.
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
#     return (magnitude * (pe**power)) / (np.pi)
#     return (pe * (1. + (8./(np.pi**2.))))
    coeff = 1.92#2.03#3.5#2.03
    #coeff= 0.4053
    return (pe * coeff)

def conForRClust(pe, eps):
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out
def latToPhi(latIn):
    '''Read in lattice spacing, output phi'''
    phiCP = np.pi / (2. * np.sqrt(3.))
    return phiCP / (latIn**2)

lat_theory = conForRClust(peNet, eps)
curPLJ = ljPress(lat_theory, peNet, eps)
phi_theory = latToPhi(lat_theory)

def edge_begin_funct(val_arr, rad_arr):
    
    #Calculate slope of alpha(x) between sparse array values
    deriv=np.zeros(len(val_arr)-1)
    for j in range(0,len(val_arr)-1):
        deriv[j]=(val_arr[j+1]-val_arr[j])#/(rcom_new[j+1]-rcom_new[j])
        
    #Find slopes that are greater than 10% of the maximum slope
    deriv_max_ind=np.where(deriv==np.max(deriv))[0][0]
    #Find slopes that are greater than 10% of the maximum slope
    val_max_ind=np.where(val_arr==np.max(val_arr))[0][0]

    skip=0
    if (val_max_ind-5)<=deriv_max_ind<=(val_max_ind):
        max_slope=deriv[deriv_max_ind]
    elif val_max_ind == 0:
        skip = 1
    elif 0<=val_max_ind<=4:
        max_slope=np.max(deriv[0:val_max_ind])
        deriv_max_ind=np.where(deriv==max_slope)[0][0]
    else:
    #    print('3')
        
        max_slope=np.max(deriv[val_max_ind-5:val_max_ind])
        deriv_max_ind=np.where(deriv==max_slope)[0][0]
        
    #start_range=deriv[:deriv_max_ind]
    if skip==0:
        j=deriv_max_ind
        while_j=0
    
        while (((((deriv[j]))>(0.2*max_slope))) or (val_arr[j]>(0.2*np.max(val_arr))) ):
    
            if ((deriv[j+1]<0.0) and (deriv[j]<0.0)):
                if (val_max_ind-j)>1:
                    j+=1
                    while_j=0
                    break
                else:
                    j-=1
                    while_j=1
            elif val_arr[j]<=0.0:
                while_j=0
                break
            elif ((deriv[j]<0.0) and (val_arr[j]<0.2*np.max(val_arr))):
                if (val_max_ind-j)>1:
                    j+=1
                    while_j=0
                    break
                else:
                    j-=1
                    while_j=1
            else:
                
                j-=1
                while_j=1
            if j <= 0.0:
                while_j=0
                break
        if while_j==1:
            j+=1
    elif skip==1:
        j=0
    
    #plt.plot(rad_arr, val_arr)
    #plt.plot(rad_arr[j:], val_arr[j:])
    #plt.plot(rad_arr[:len(rad_arr)-1], deriv)
    #plt.plot(rad_arr[j:len(rad_arr)-1], deriv[j:])
    #plt.show()
    return j


def edge_end_funct(val_arr, rad_arr):
    
    #Calculate slope of alpha(x) between sparse array values
    skip=0
    deriv=np.zeros(len(val_arr)-1)
    for j in range(0,len(val_arr)-1):
        deriv[j]=(val_arr[j+1]-val_arr[j])#/(rcom_new[j+1]-rcom_new[j])
    #Find slopes that are greater than 10% of the maximum slope
    deriv_min_ind=np.where(deriv==np.min(deriv))[0][0]
    
    val_max_ind=np.where(val_arr==np.max(val_arr))[0][0]
    
    if (val_max_ind)<=deriv_min_ind<=(val_max_ind+6):
        min_slope=deriv[deriv_min_ind]
    elif (len(val_arr)-4)<=val_max_ind<(len(val_arr)-1):
        min_slope=np.max(deriv[val_max_ind:len(val_arr)])
        deriv_min_ind=np.where(deriv==min_slope)[0][0]
    elif val_max_ind>=(len(val_arr)-1):
        skip=1
    else:
    #    print('3')
        
        min_slope=np.min(deriv[val_max_ind:val_max_ind+6])
        deriv_min_ind=np.where(deriv==min_slope)[0][0]

    #if ((len(deriv)-10)<=deriv_min_ind<=len(deriv)) or ((len(val_arr)-5)<=val_max_ind<=len(val_arr)):
    #    print('1')
    #    j=0.0
    #    skip=1
    #elif (val_max_ind)<deriv_min_ind<(val_max_ind+5):
    #    print('2')
    #    min_slope=deriv[deriv_min_ind]
    #elif deriv_min_ind==val_max_ind:
    #    deriv_min_ind=deriv_min_ind+1
        #min_slope=deriv[deriv_min_ind]
    
    #else:
    #    print('3')
        
    #    min_slope=np.max(deriv[val_max_ind+1:val_max_ind+5])
    #    deriv_min_ind=np.where(deriv==min_slope)[0][0]
    if skip==0:
        j=deriv_min_ind
    
        while (((((deriv[j]))<(0.2*min_slope))) or (val_arr[j]>(0.2*np.max(val_arr))) ):
    
            if ((deriv[j-1]>0.0) and (deriv[j]>0.0)):
                if (j-val_max_ind)>1:
                    j-=1
                    while_j=0
                    break
                else:
                    j+=1
                    while_j=1
            elif ((deriv[j]>0.0) and (val_arr[j]<0.2*np.max(val_arr))):
                if (j-val_max_ind)>1:
                    j-=1
                    while_j=0
                    break
                else:
                    j+=1
                    while_j=1
            elif val_arr[j]<=0.0:
                while_j=0
                break
            else:
                
                j+=1
                while_j=1
            if j >= len(deriv):
                while_j=0
                break
    else:
        j=0
    return j


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
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
                

file_name = os.path.basename(inFile)
outfile, file_extension = os.path.splitext(file_name)   # get base name
out = outfile + "_frame_"  

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
        NBins = getNBins(l_box, r_cut)
        sizeBin = roundUp(((l_box) / NBins), 6)
        
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
            
            
            
            
            yellow = ("#fdfd96")
            green = ("#77dd77")
            red = ("#ff6961")
            NBins = getNBins(l_box, 6.0)
            sizeBin = roundUp(((l_box) / NBins), 6)
                                 
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
            
            num_dens3 = [[0 for b in range(NBins)] for a in range(NBins)]
            
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            #posParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
                      
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
                    
            
            pos_box_start=np.array([])
            for ix in range(0, len(occParts)):
                pos_box_start = np.append(pos_box_start, ix*sizeBin)
                for iy in range(0, len(occParts)):
                    pos_box_x[ix][iy] = ((ix+0.5)*sizeBin)
                    pos_box_y[ix][iy] = ((iy+0.5)*sizeBin)
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
                            
                            
                        num_dens3[ix][iy] = (len(binParts[ix][iy])/(sizeBin**2))*(math.pi/4)
                        p_avg_x[ix][iy] = p_all_x[ix][iy]/len(binParts[ix][iy])
                        p_avg_y[ix][iy] = p_all_y[ix][iy]/len(binParts[ix][iy])

                            
                    
                        #widths = [1.0,1.0]#, [1,1,1,1],[1,1,1,1]]#[1,1.205, 1.205, 1.205]
                        #gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.2)
                        #ax0 = fig.add_subplot(gs[0, 0])
            '''
            fig = plt.figure(figsize=(6, 6))
                        #widths = [1.0,1.0]#, [1,1,1,1],[1,1,1,1]]#[1,1.205, 1.205, 1.205]
                        #gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.2)
                        #ax0 = fig.add_subplot(gs[0, 0])
            myEps = [1., 0.1, 0.01, 0.001, 0.0001]
            plt.scatter(pos[bulk_id,0]+h_box, pos[bulk_id,1]+h_box, s=1, marker='.', c=green)
            plt.scatter(pos[edge_id,0]+h_box, pos[edge_id,1]+h_box, s=1, marker='.', c=yellow)
            plt.scatter(pos[gas_id,0]+h_box, pos[gas_id,1]+h_box, s=1, marker='.', c=red)
        
                        
                        #plt.scatter(pos[:,0]+h_box, pos[:,1]+h_box, s=1, marker='.', c=plt.cm.jet(float(1)/ (len(myEps)-1) ))
            plt.quiver(pos_box_x, pos_box_y, p_avg_x, p_avg_y)
            plt.xticks(pos_box_start)
            plt.yticks(pos_box_start)
            plt.tick_params(
                                axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelleft=False)
            plt.tick_params(
                                axis='y',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                right=False,      # ticks along the bottom edge are off
                                left=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelleft=False)
                        #plt.grid()
                        #ax0.set_title('avg orientation')
            plt.ylim((0, l_box))
            plt.xlim((0, l_box))
                        #plt.show()
            plt.text(-40, 330, '(a)',
                             fontsize=22)
                    #plt.title('time='+str(round(tst,1)))
                    #plt.tight_layout()
            eps_leg=[]
            mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
            msz=40
            red_patch = mpatches.Patch(color=red, label='Gas')
            green_patch = mpatches.Patch(color=green, label='Bulk')
            yellow_patch = mpatches.Patch(color=yellow, label='Interface')
            plt.legend(handles=[green_patch, yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=17, loc='upper left')

            plt.show()
            '''
            print(j)

            vmax_eps = phi_theory * 1.3
            pad = str(j).zfill(4)
            fig = plt.figure(figsize=(8,6))

            #ax2 = fig.add_subplot()


            im = plt.contourf(pos_box_x, pos_box_y, num_dens3, vmin=0.0, vmax=vmax_eps)
            
#            norm= matplotlib.colors.Normalize(vmin=im.cvalues.min(), vmax=im.cvalues.max())
            norm= matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_eps)

            sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
            sm.set_array([])
            tick_lev = np.arange(0, vmax_eps+vmax_eps/10, vmax_eps/10)
            clb = fig.colorbar(sm, ticks=tick_lev)
            clb.ax.set_title(r'$\phi$')
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
            plt.savefig(outPath + 'density_' + out + pad + ".png", dpi=200)
            plt.close()
