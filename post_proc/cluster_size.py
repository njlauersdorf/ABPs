#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:01:19 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Get center of mass
'''

import sys
import os

# Run locally
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
outDPI = 500
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/rdf_data/'#'/Volumes/External/04_01_20_parent/gsd/'


import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster

import math
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib import collections  as mc
from matplotlib import lines
    
# Grab files
slowCol = '#d8b365'
fastCol = '#5ab4ac'

# Command line arguments
infile = str(sys.argv[1])                               # gsd file
#infile = 'pa450_pb500_xa50_ep1.0_phi60_pNum100000.gsd'
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac = float(sys.argv[4])
eps = float(sys.argv[5])
print('peA',flush=True)
print(peA,flush=True)
print('peB',flush=True)
print(peB,flush=True)
print('xA',flush=True)
print(parFrac,flush=True)
print('eps',flush=True)
print(eps,flush=True)

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
    
out = "final_pe" + "{:.0f}".format(peA) +\
      "_phi" + "{:.0f}".format(intPhi) +\
      "_eps" + "{:.5f}".format(eps) +\
      "_fm"
    
# Create outfile name from infile name
file_name = os.path.basename(infile)
outfile, file_extension = os.path.splitext(file_name)   # get base name
out = outfile

# Get dumps to output
f = hoomd.open(name=infile, mode='rb')  # open gsd file with hoomd
dumps = int(f.__len__())                # get number of timesteps dumped
start = 0
#start = dumps - 1                       # gives first frame to read
end = dumps                             # gives last frame to read
#end = int(0.9*dumps)
#start = end-1
#end = 20
#start = dumps - 100
print('start',flush=True)
print(start,flush=True)
print('end',flush=True)
print(end,flush=True)

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

# Round up size of bins to account for floating point inaccuracy
def roundUp(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
    
def quatToAngle(quat):
    "Take vector, output angle between [-pi, pi]"
    x = quat[1]
    y = quat[2]
    rad = math.atan2(y, x)
    return rad
    
def distComps(point1, point2x, point2y):
    '''Given points output x, y and r'''
    dx = point2x - point1[0]
    dy = point2y - point1[1]
    r = np.sqrt((dx**2) + (dy**2))
    return dx, dy, r
    
def computeFLJ(r, dx, dy, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (dx) / r
    fy = f * (dy) / r
    return fx, fy

# Compute mesh
r_cut = 2**(1./6.)

outTxt2 = 'Cluster_size_' + out + '.txt'
g = open(outPath+outTxt2, 'w') # write file headings
g.write('tst'.center(25) + ' ' +\
        'ClusterSize'.center(25) + ' ' +\
        'SteadyStateSize'.center(25) + ' ' +\
        'SteadyStateTime'.center(25) + '\n')
g.close()

# Access file frames
with hoomd.open(name=infile, mode='rb') as t:

    # Take first snap for box
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    # Get box dimensions
    l_box = box_data[0]
    h_box = l_box / 2.
    a_box = l_box * l_box
    nBins = (getNBins(l_box, r_cut))
    sizeBin = roundUp((l_box / nBins), 6)
    partNum = len(snap.particles.typeid)
    pos = snap.particles.position
    print('l_box',flush=True)
    print(l_box,flush=True)    
    # Get the largest x-position in the largest cluster
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    my_clust = cluster.Cluster()
    c_props = cluster.ClusterProperties()

    # You need a list of length nBins to hold the sum
    phi_sum = [ 0. for i in range(0, nBins) ]
    phi_sum0 = [ 0. for i in range(0, nBins) ]
    phi_sum1 = [ 0. for i in range(0, nBins) ]
    p_sum = [ 0. for i in range(0, nBins) ]
    pswim_sum = [ 0. for i in range(0, nBins) ]
    pint_sum = [ 0. for i in range(0, nBins) ]
    # You need one array of length nBins to hold the counts
    num = [ 0. for i in range(0, nBins) ]
    # List to hold the average
    phi_avg = []
    p_avg = []
    pint_avg = []
    pswim_avg = []
    # You should store the max distance of each bin as well
    r_bins = np.arange(sizeBin, sizeBin * nBins, sizeBin)
    size_arr = np.zeros(int(end))
    time_arr = np.zeros(int(end))
    std_arr=np.zeros(int(end))
    std_dif_arr=np.zeros(end-1)
    std_dif_arr2=np.zeros(end-1)
    # Loop through snapshots
    for j in range(start, int(end)):
        print(j)
        snap = t[j]
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau      
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0                             # convert to Brownian time
        system = freud.AABBQuery(f_box, f_box.wrap(pos))
        my_clust=freud.cluster.Cluster()                      #Define cluster
        # Compute neighbor list for only largest cluster
        my_clust.compute(system, neighbors={'r_max': 1.0})
        c_props = freud.cluster.ClusterProperties()         #Define cluster properties

        ids = my_clust.cluster_idx              # get id of each cluster
        c_props.compute(system, ids)            # find cluster properties

        clust_size = c_props.sizes              # find cluster sizes
                           # convert to Brownian time
        time_arr[j]=tst
        print(np.amax(clust_size))
        size_arr[j]=np.amax(clust_size)
    mean_size=np.mean(size_arr)
    for j in range(start,int(end)):
        print('j')
        print(j)
        if j==0:
            pass
        elif 0<j<51:
            mean_size=np.mean(size_arr[:j+1])
            mean_size_span=np.mean(size_arr[:2*j+1])
            test = (np.abs(mean_size-mean_size_span)/mean_size_span)*100
            if test<5.0:
                equilib_id=j
                break
        elif j>=51:
            mean_size=np.mean(size_arr[j-51:j+1])
            mean_size_span=np.mean(size_arr[j-51:j+51])
            test = (np.abs(mean_size-mean_size_span)/mean_size_span)*100
            if test<5.0:
                equilib_id=j
                break
    snap = t[equilib_id]
    tst = snap.configuration.step 
    steadystatetime=tst[equilib_id]
    steadystatesize=np.mean(size_arr[equilib_id:])
    for j in range(equilib_id, int(end)):
        print('j')
        print(j)
        # Get the current snapshot
        snap = t[j]
        # Easier accessors
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0
        xy = np.delete(pos, 2, 1)
        typ = snap.particles.typeid                 # type
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time

        g = open(outPath+outTxt2, 'a')
        g.write('{0:.3f}'.format(tst).center(25) + ' ')
        g.write('{0:.6f}'.format(size_arr[j]).center(25) + ' ')
        g.write('{0:.0f}'.format(steadystatesize).center(25) + ' ')
        g.write('{0:.6f}'.format(steadystatetime).center(25) + '\n')
        g.close()


## Compute the average in each bin
#for k in range(0, len(phi_sum)):
#    if num[k] > 0:
#        phi_avg.append(phi_sum[k] / num[k])
#        p_avg.append(p_sum[k] / num[k])
#        pint_avg.append(pint_sum[k] / num[k])
#        pswim_avg.append(pswim_sum[k] / num[k])
#    else:
#        phi_avg.append(0.)
#        p_avg.append(0.)
#        pint_avg.append(0.)
#        pswim_avg.append(0.)

# Write textfile
# Append data to file

## Plot scatter of phi_loc vs r_com
#outDPI = 500
#plt.plot(r_bins, pswim_avg, lw=1.5, c='r', zorder=0)
#plt.scatter(r_bins, pswim_avg, s=5, c='k', zorder=1)
#plt.xlim(0,)
#plt.tight_layout()
#plt.savefig("pswim" + out + ".png", dpi=outDPI)
#plt.close()
