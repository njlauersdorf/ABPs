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

pe_a = sys.argv[1]                    # activity A
pe_b = sys.argv[2]                    # activity B
part_perc_a = sys.argv[3]              # percentage A particles
part_frac_a = float(part_perc_a) / 100.0    # fraction A particles
hoomd_path = str(sys.argv[4])               # local path to hoomd-blue
gsd_path = str(sys.argv[5])                 # local path to gsd
eps = sys.argv[6]

sys.path.insert(0,hoomd_path)
sys.path.insert(0,gsd_path)
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
in_file = 'pa150_pb500_xa50_ep1_phi60_pNum10000.gsd'


out = "pa"+str(pe_a)+\
"_pb"+str(pe_b)+\
"_xa"+str(part_perc_a)+\
"_ep"+str(eps)

f = hoomd.open(name=gsd_path+in_file, mode='rb') # open gsd file with hoomd
dumps = f.__len__()                     # get number of timesteps dumped

start = 0       # gives first frame to read
end = dumps     # gives last frame to read

positions = np.zeros((end), dtype=np.ndarray)       # array of positions
types = np.zeros((end), dtype=np.ndarray)           # particle types
box_data = np.zeros((1), dtype=np.ndarray)          # box dimensions
timesteps = np.zeros((end), dtype=np.float64)       # timesteps
orient = np.zeros((end), dtype=np.ndarray)          # orientations

# Get relevant data from .gsd file
with hoomd.open(name=gsd_path+in_file, mode='rb') as t:
    snap = t[0]
    box_data = snap.configuration.box
    for iii in range(start, end):
        snap = t[iii]                               # snapshot of frame
        types[iii] = snap.particles.typeid          # get types
        positions[iii] = snap.particles.position    # get positions
        orient[iii] = snap.particles.orientation    # get orientation
        timesteps[iii] = snap.configuration.step    # get timestep

timesteps -= timesteps[0]       # get rid of brownian run time

# Get number of each type of particle
part_num = len(types[start])
part_A = int(part_num * part_frac_a)
part_B = part_num - part_A

# Feed data into freud analysis software
l_box = box_data[0]
h_box = l_box / 2.0
a_box = l_box * l_box
f_box = box.Box(Lx = l_box, Ly = l_box, is2D = True)    # make freud box

nBins = 1000
widthBin = 0.005
searchRange = nBins * widthBin
radialDF = freud.density.RDF(searchRange, widthBin)

#r = np.arange(0.0, searchRange, widthBin)
#k = np.arange(0.0, )

N = nBins                       # number of samples
T = widthBin                    # spacing between samples
r = np.linspace(0.0, N*T, N)    # 0 through searchRange with Nbins
# // is floor division, adjusts to left in number line
k = np.linspace(0.0, 1.0/(2.0*T), N//2)

for iii in range(start, int(end/100)):
    iii=iii*100
    # Easier accessors
    pos = positions[iii]
    typ = types[iii]
    direct = orient[iii]
    
    typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
    typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
    pos0=pos[typ0ind]                               # Find positions of type 0 particles
    pos1=pos[typ1ind] 

    # Compute RDF for all particles
    system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
    radialDF.compute(system=system_all)
            
    #system_all_cut = freud.AABBQuery(f_box, f_box.wrap(pos_all_in_clust))    #Calculate neighbor list
    system_A_cut = freud.AABBQuery(f_box, f_box.wrap(pos0))    #Calculate neighbor list
    system_B_cut = freud.AABBQuery(f_box, f_box.wrap(pos1))    #Calculate neighbor list


    radialDFall = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
    radialDFAA = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
    radialDFAB = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
    radialDFBA = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
    radialDFBB = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
        
    radialDFAA.compute(system=system_A_cut, query_points=pos0 , reset=False)               #Calculate radial density function
    radialDFAB.compute(system=system_B_cut, query_points=pos0 , reset=False) 
    radialDFBA.compute(system=system_A_cut, query_points=pos1 , reset=False)               #Calculate radial density function
    radialDFBB.compute(system=system_B_cut, query_points=pos1 , reset=False) 
    
plt.plot(r, radialDFAA.rdf, label='A-A')
plt.plot(r, radialDFAB.rdf, label='A-B')
plt.plot(r, radialDFBA.rdf, label='B-A')
plt.plot(r, radialDFBB.rdf, label='B-B')
plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.show()
stop

plt.figure(figsize=(12, 3))
plt.subplot(141)   
plt.plot(r, radialDFAA.rdf, label='A-A')
plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.subplot(142)   
plt.plot(r, radialDFAB.rdf, label='A-B')
plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.subplot(143)   
plt.plot(r, radialDFBA.rdf, label='B-A')
plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.subplot(144)   
plt.plot(r, radialDFBB.rdf, label='B-B')
plt.ylim([0,10])
plt.xlabel(r'r $(\sigma)$')
plt.ylabel(r'g(r)')
plt.legend()
plt.show()
stop
#plt.savefig('RDF_' + out + '_fm' + str(iii) + '.png', dpi=1000)
#plt.close()