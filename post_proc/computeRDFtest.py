#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:55:06 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
First: obtain the pair correlation function (using Freud)
Then: take fourier transform of it (obtain structure factor)
And then: take inverse first moment (obtain coarsening length)
Finally: plot all that shit
'''

# Imports and loading the .gsd file
import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math

hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)

inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
#inFile='pa300_pb300_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'cluster_pa400_pb350_phi60_eps0.1_xa0.8_align3_dtau1.0e-06.gsd'
#inFile='pa400_pb500_xa20_ep1.0_phi60_pNum100000.gsd'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
outPath='/Volumes/External/n100000test/final/'
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

import hoomd
from hoomd import md
from hoomd import deprecated

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster

import numpy as np
from scipy.fftpack import fft, ifft

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

import math

# File to read from
#in_file = 'pa150_pb500_xa50_ep1_phi60_pNum10000.gsd'


out = "pa"+str(peA)+\
"_pb"+str(peB)+\
"_xa"+str(parFrac)+\
"_ep"+str(eps)

f = hoomd.open(name=inFile, mode='rb') # open gsd file with hoomd
dumps = f.__len__()                     # get number of timesteps dumped

start = 0       # gives first frame to read
end = dumps     # gives last frame to read

positions = np.zeros((end), dtype=np.ndarray)       # array of positions
types = np.zeros((end), dtype=np.ndarray)           # particle types
box_data = np.zeros((1), dtype=np.ndarray)          # box dimensions
timesteps = np.zeros((end), dtype=np.float64)       # timesteps
orient = np.zeros((end), dtype=np.ndarray)          # orientations

# Get relevant data from .gsd file
with hoomd.open(name=inFile, mode='rb') as t:
    snap = t[0]
    box_data = snap.configuration.box

timesteps -= timesteps[0]       # get rid of brownian run time

# Feed data into freud analysis software
l_box = box_data[0]
h_box = l_box / 2.0
a_box = l_box * l_box
f_box = box.Box(Lx = l_box, Ly = l_box, is2D = True)    # make freud box

#widthBin = 0.05
#nBins=l_box/widthBin
rstop=5.0
nBins = 1000
widthBin = 0.5
searchRange = nBins * widthBin
r=np.arange(0.,rstop,widthBin)


#r = np.arange(0.0, searchRange, widthBin)
#k = np.arange(0.0, )

N = int(nBins)                     # number of samples
T = widthBin                    # spacing between samples
r = np.linspace(1.0, rstop, N)    # 0 through searchRange with Nbins
# // is floor division, adjusts to left in number line
k = np.linspace(0.0, 1.0/(2.0*T), N//2)

with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0#600                  # first frame to process
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
    # Loop through each timestep
    for j in range(start, int(end/100)):
        j=j*100
        print('j')
        print(j)
        snap = t[j]
        # Easier accessors
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0
        typ = snap.particles.typeid                 # type
    
        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind] 

        # Compute RDF for all particles
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        
        cl_all=freud.cluster.Cluster()                      #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()         #Define cluster properties
        ids = cl_all.cluster_idx              # get id of each cluster
        clp_all.compute(system_all, ids)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        min_size=int(partNum/10)
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]
        large_clust_ind_all=np.where(clust_size>min_size)
    
        print('1')
        if len(large_clust_ind_all[0])>0:
            print('2')
    #system_all_cut = freud.AABBQuery(f_box, f_box.wrap(pos_all_in_clust))    #Calculate neighbor list
            system_A_cut = freud.AABBQuery(f_box, f_box.wrap(pos0))    #Calculate neighbor list
            system_B_cut = freud.AABBQuery(f_box, f_box.wrap(pos1))    #Calculate neighbor list
            print('3')

            radialDFall = freud.density.RDF(bins=nBins, r_max=10,r_min=1.0)#h_box-1)    #Set density per bin with a
            radialDFAA = freud.density.RDF(bins=nBins, r_max=10,r_min=1.0)#h_box-1)    #Set density per bin with a
            print('4')
            radialDFAB = freud.density.RDF(bins=nBins, r_max=10,r_min=1.0)#h_box-1)    #Set density per bin with a
            radialDFBA = freud.density.RDF(bins=nBins, r_max=10,r_min=1.0)#h_box-1)    #Set density per bin with a
            radialDFBB = freud.density.RDF(bins=nBins, r_max=10,r_min=1.0)#h_box-1)    #Set density per bin with a
            print('5')
            radialDFAA.compute(system=system_A_cut, query_points=pos0 , reset=False)               #Calculate radial density function
            radialDFAB.compute(system=system_B_cut, query_points=pos0 , reset=False) 
            print('6')
            radialDFBA.compute(system=system_A_cut, query_points=pos1 , reset=False)               #Calculate radial density function
            radialDFBB.compute(system=system_B_cut, query_points=pos1 , reset=False) 
            print('7')
plt.plot(r, radialDFAA.rdf, label='A-A')
plt.plot(r, radialDFAB.rdf, label='A-B')
plt.plot(r, radialDFBA.rdf, label='B-A')
plt.plot(r, radialDFBB.rdf, label='B-B')
#plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.show()
stop
