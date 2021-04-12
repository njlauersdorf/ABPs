#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Read in .gsd file of particle positions
2.) Mesh the space
3.) Loop through tsteps and ...
3a.) Place all particles in appropriate mesh grid
3b.) Loop through all particles ...
3b.i.) Compute distance to every particle in adjacent grids
3.b.ii.) If distance is less than LJ cutoff, store as effective diameter
3c.) Plot particle with effective diameter as patch
4.) Generate movie from frames
'''

import sys
import os

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#

#Run on Cluster
hoomdPath='/nas/home/njlauers/hoomd-blue/build/'

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

#Cut off interaction radius (Per LJ Potential)
r_cut=2**(1/6)

# Run locally
#outPath='/Volumes/External/test_video_mono/'

#Run on Cluster
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/sim_video/'

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
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
from matplotlib import lines

def computeR(part1, part2):
    """Computes distance"""
    return np.sqrt(((part2[0]-part1[0])**2)+((part2[1]-part1[1])**2))

def computeA(diameter):
    """Computes area of circle"""
    radius = diameter / 2.0
    return np.pi * (radius**2)

def getDistance(point1, point2x, point2y):
    """Find the distance between two points"""
    distance = np.sqrt((point2x - point1[0])**2 + (point2y - point1[1])**2)
    return distance
    
# Grab files
fastCol = '#d8b365'
slowCol = '#5ab4ac'

# Command line arguments
infile = str(sys.argv[1])                               # gsd file
peA = float(sys.argv[2])
peB = float(sys.argv[3])#float(sys.argv[3])
parFrac_orig = float(sys.argv[4])
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
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
    
out = "pa" + "{:.0f}".format(peA) +\
      "pb" + "{:.0f}".format(peB) +\
      "_phi" + "{:.0f}".format(intPhi) +\
      "_ep" + "{:.5f}".format(eps) +\
      "_fm"
    
# Create outfile name from infile name

file_name = os.path.basename(infile)
outfile, file_extension = os.path.splitext(file_name)   # get base name
out = outfile + "_frame_"

# Get dumps to output

f = hoomd.open(name=infile, mode='rb')  # open gsd file with hoomd
dumps = int(f.__len__())                # get number of timesteps dumped
start = 0
#start = dumps - 1                       # gives first frame to read
end = dumps                             # gives last frame to read
#end = 20

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

# Compute mesh
r_cut = 2**(1./6.)

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

    # Loop through snapshots
    for j in range(start, end):
    
        # Get the current snapshot
        snap = t[j]
        # Easier accessors
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0
  
        typ0ind=np.where(snap.particles.typeid==0)[0]      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)[0]      # Calculate which particles are type 1
        #typ0ind=np.linspace(0,(partNum/2)-1, int(partNum/2+1))
        #typ1ind=np.linspace(partNum/2, partNum-1, int(partNum/2+1))
        #typ0ind=typ0ind.astype(int)
        #typ1ind=typ1ind.astype(int)
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind]
        
        xy = np.delete(pos, 2, 1)

        typ = snap.particles.typeid                 # type
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        
        # Create frame pad for images
        pad = str(j).zfill(4)
        
        # Plot the figure
        #fig,
        fig, ax = plt.subplots(1, 1)
        #fig = plt.figure(figsize=(10, 10))
        # Axes to hold each plot
        #ax = []
        # Plots
        #ax.append(fig.add_subplot())
        #ax.scatter(pos0[:,0], pos0[:,1], c=slowCol, edgecolor='none', s=7., label='PeA: '+str(peA))
        #ax.scatter(pos1[:,0], pos1[:,1], c=fastCol, edgecolor='none', s=7., label='PeB: '+str(peB))
        ells0 = [Ellipse(xy=pos0[i,:],
                width=1.0, height=1.0, label='PeA: '+str(peA))
        for i in range(0,len(typ0ind))]
        ells1 = [Ellipse(xy=pos1[i,:],
                width=1.0, height=1.0, label='PeB: '+str(peB))
        for i in range(0,len(typ1ind))]

        for e in ells0:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set(label='PeA: '+str(peA))
            e.set_facecolor(slowCol)
        #if parFrac<100.0:
        for e in ells1:
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set(label='PeB: '+str(peB))
                e.set_facecolor(fastCol)
        ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        ax.set_xlim(-h_box, h_box)
        ax.set_ylim(-h_box, h_box)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')
        if parFrac<100.0:
            ax.legend(handles=[ells0[0], ells1[1]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(peA),r'$\mathrm{Pe}_\mathrm{B} = $'+str(peB)], loc='upper right', prop={'size': 15}, markerscale=8.0)
        else:
            pass#ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(peA)], loc='upper right', prop={'size': 15}, markerscale=8.0)

        #plt.subplots_adjust(0,0,1,1)
        #plt.title('PeA:PeB='+str(int(parFrac))+':'+str(int(100-parFrac)))
        if parFrac<100.0:
            plt.title(r'$\mathrm{N}$' + ' = ' + str(int(partNum)) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))), fontsize=17)

        else:
            plt.title(r'$\mathrm{Pe}$' + ' = ' + str(int(peA)) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))), fontsize=17)

        plt.tight_layout()
        plt.savefig(outPath+out + pad + ".png", dpi=200)
        plt.close()
