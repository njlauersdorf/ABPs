#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Read in .gsd file of particle positions
3.) Loop through tsteps and ...
3a.) Determine largest cluster's center of mass
3b.) Translate particles positions such that origin (0,0) is cluster's center of mass
3c.) Plot each particle's location in x-y simulation space color-coded by activity
3d.) Generate figure for current time step
4) Generate movie from frames
'''


import sys
import os

# Run locally
hoomdPath=str(sys.argv[10])
outPath2=str(sys.argv[11])
outPath=str(sys.argv[12])

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches


import numpy as np
import matplotlib
if hoomdPath == '/nas/longleaf/home/njlauers/hoomd-blue/build':
    matplotlib.use('Agg')
else:
    pass
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
from matplotlib import lines


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




# Command line arguments
infile = str(sys.argv[1])                               # Get infile (.gsd)
peA = float(sys.argv[2])                                #Activity (Pe) for species A
peB = float(sys.argv[3])                                #Activity (Pe) for species B


#Set plot colors
fastCol = '#e31a1c'
slowCol = '#081d58'
    
parFrac_orig = float(sys.argv[4])                       #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
 
if (parFrac == 100.):
    mono=1
    mono_activity=peA
    mono_type = 0
elif (parFrac == 0.):
    mono = 1
    mono_activity=peB
    mono_type = 1
elif peA==peB:
    mono=1
    mono_activity=peA
    mono_type = 2
else:
    mono=0
    
eps = float(sys.argv[5])                                #Softness, coefficient of interparticle repulsion (epsilon)

#Set system area fraction (phi)
try:
    phi = float(sys.argv[6])
    intPhi = int(phi)
    phi /= 100.
except:
    phi = 0.6
    intPhi = 60
    
#Get simulation time step
try:
    dtau = float(sys.argv[7])
except:
    dtau = 0.000001
    
#Open input simulation file
f = hoomd.open(name=infile, mode='rb')
                
#Get particle number from initial frame
snap = f[0]
typ = snap.particles.typeid
partNum = len(typ)

#Set output file names
bin_width = float(sys.argv[8])
time_step = float(sys.argv[9])

# Create outfile name from infile name
outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) 
out = 'sim_frame_' + outfile + "_frame_"

# Get timesteps to output
dumps = int(f.__len__())                # get number of timesteps dumped
start = 500                               # gives first frame to read
end = dumps                             # gives last frame to read

def getNBins(length, minSz=(2**(1./6.))):
    '''
    Purpose: Given box size, return number of bins
    
    Inputs: 
        length: length of box
        minSz: set minimum bin length to LJ cut-off distance
    Output: number of bins along box length rounded up
    '''
    
    initGuess = int(length) + 1
    nBins = initGuess
    # This loop only exits on function return
    while True:
        if length / nBins > minSz:
            return nBins
        else:
            nBins -= 1

def roundUp(n, decimals=0):
    '''
    Purpose: Round up number of bins to account for floating point inaccuracy
    
    Inputs: 
        n: number of bins along length of box
        decimals: exponent of multiplier for rounding (default=0)
    Output: number of bins along box length rounded up
    '''
    
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

#Cut off interaction radius (Per LJ Potential)
r_cut=2**(1/6)

# Access file frames
with hoomd.open(name=infile, mode='rb') as t:

    # Take first snap for box
    snap = t[0]
    first_tstep = snap.configuration.step

    # Get box dimensions
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.
    a_box = l_box * l_box
    
    #2D binning of system
    nBins = (getNBins(l_box, r_cut))
    sizeBin = roundUp((l_box / nBins), 6)
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)    
    
    #Find particle identifier information
    partNum = len(snap.particles.typeid)                #Get particle number from initial frame
    typ0ind=np.where(snap.particles.typeid==0)[0]       # Calculate which particles are type 0
    typ1ind=np.where(snap.particles.typeid==1)[0]       # Calculate which particles are type 1
    
    '''
    #If you want to colorize monodisperse systems to visualize motion, uncomment
    if (len(typ0ind)>0) and (len(typ1ind)>0):
        pass      
    else:
            typ0ind = random.sample(range(int(partNum)), int(partNum/2))
            typ1ind = np.array([], dtype=int)
            for i in range(0, partNum):
                if i in typ0ind:
                    pass
                else:
                    typ1ind = np.append(typ1ind, int(i))
    '''
                    
    # Loop through snapshots
    for j in range(start, end):
        
        
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
        
        #Identify clusters
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))        #Make neighbor list of particles
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        
        clust_size = clp_all.sizes                                  # find cluster sizes
        min_size=int(partNum/5)                                     #Define minimum size of largest cluster to perform measurement
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Find ID of largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Find IDs of clusters with size larger than minimum limit
        
        query_points=clp_all.centers[lcID]                          #Find CoM of largest cluster
        #com_tmp_posX = query_points[0]# + h_box                     #X position of largest cluster's CoM
        #com_tmp_posY = query_points[1]# + h_box                     #Y position of largest cluster's CoM
        com_tmp_posX = np.mean(pos[:,0])
        com_tmp_posY = np.mean(pos[:,1])
        #Shift origin (0,0) to cluster's CoM
        pos[:,0]= pos[:,0]-com_tmp_posX                             
        pos[:,1]= pos[:,1]-com_tmp_posY
        
        #Enforce periodic boundary conditions
        for i in range(0, partNum):
            if pos[i,0]>h_box:
                pos[i,0]=pos[i,0]-l_box
            elif pos[i,0]<-h_box:
                pos[i,0]=pos[i,0]+l_box
                
            if pos[i,1]>h_box:
                pos[i,1]=pos[i,1]-l_box
            elif pos[i,1]<-h_box:
                pos[i,1]=pos[i,1]+l_box
        print('mono')
        print(mono)
        if mono==0:
            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]
            
            # Create frame pad for images
            pad = str(j).zfill(4)
            
            #Plot each particle as a point color-coded by activity and labeled by their activity
            fig = plt.figure(figsize=(6.5,6)) 
            ax = fig.add_subplot(111)  
            
            
                
                
            sz = 0.75
            #Assign type 0 particles to plot
            
            ells0 = [Ellipse(xy=pos0[i,:],
                    width=sz, height=sz, label='PeA: '+str(peA))
            for i in range(0,len(typ0ind))]
            
            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=pos1[i,:],
                    width=sz, height=sz, label='PeB: '+str(peB))
            for i in range(0,len(typ1ind))]
            
            # Plot position colored by neighbor number
            if peA <= peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            
            #Label time step
            ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=18,
                    bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
            #Set axes parameters
            ax.set_xlim(-h_box, h_box)
            ax.set_ylim(-h_box, h_box)
            ax.axes.set_xticks([])
            ax.axes.set_yticks([])
            ax.axes.set_xticklabels([])
            ax.axes.set_yticks([])
            ax.set_aspect('equal')
            
            #Create legend for binary system
            if parFrac<100.0:
                leg = ax.legend(handles=[ells0[0], ells1[1]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(peA)), r'$\mathrm{Pe}_\mathrm{B} = $'+str(int(peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                if peA <= peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peA)), r'$\mathrm{Pe} = $'+str(int(peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
            plt.tight_layout()
            plt.savefig(outPath+out + pad + ".png", dpi=150, transparent=False)
            plt.close()
            
        elif mono == 1:
            if mono_type == 0:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                
                # Create frame pad for images
                pad = str(j).zfill(4)
                
                #Plot each particle as a point color-coded by activity and labeled by their activity
                fig = plt.figure(figsize=(6.5,6)) 
                ax = fig.add_subplot(111)  
                
                
                    
                    
                sz = 0.75
                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=pos0[i,:],
                        width=sz, height=sz, label='Pe: '+str(peA))
                for i in range(0,len(typ0ind))]
                
                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                
                #Label time step
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                
                #Set axes parameters
                ax.set_xlim(-h_box, h_box)
                ax.set_ylim(-h_box, h_box)
                ax.axes.set_xticks([])
                ax.axes.set_yticks([])
                ax.axes.set_xticklabels([])
                ax.axes.set_yticks([])
                ax.set_aspect('equal')
                

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
                plt.tight_layout()
                plt.savefig(outPath+out + pad + ".png", dpi=150, transparent=False)
                plt.close()
            elif mono_type == 1:
                #Local each particle's positions
                pos0=pos[typ1ind]                               # Find positions of type 0 particles
                
                # Create frame pad for images
                pad = str(j).zfill(4)
                
                #Plot each particle as a point color-coded by activity and labeled by their activity
                fig = plt.figure(figsize=(6.5,6)) 
                ax = fig.add_subplot(111)  
                
                
                    
                    
                sz = 0.75
                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=pos0[i,:],
                        width=sz, height=sz, label='Pe: '+str(peB))
                for i in range(0,len(typ0ind))]
                
                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                
                #Label time step
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                
                #Set axes parameters
                ax.set_xlim(-h_box, h_box)
                ax.set_ylim(-h_box, h_box)
                ax.axes.set_xticks([])
                ax.axes.set_yticks([])
                ax.axes.set_xticklabels([])
                ax.axes.set_yticks([])
                ax.set_aspect('equal')
                

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
                plt.tight_layout()
                plt.savefig(outPath+out + pad + ".png", dpi=150, transparent=False)
                plt.close()
            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]  
                # Create frame pad for images
                pad = str(j).zfill(4)
                
                #Plot each particle as a point color-coded by activity and labeled by their activity
                fig = plt.figure(figsize=(6.5,6)) 
                ax = fig.add_subplot(111)  
                
                
                    
                    
                sz = 0.75
                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=pos0[i,:],
                        width=sz, height=sz, label='Pe: '+str(peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=pos1[i,:],
                        width=sz, height=sz, label='Pe: '+str(peB))
                for i in range(0,len(typ1ind))]
                
                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)
                
                #Label time step
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                
                #Set axes parameters
                ax.set_xlim(-h_box, h_box)
                ax.set_ylim(-h_box, h_box)
                ax.axes.set_xticks([])
                ax.axes.set_yticks([])
                ax.axes.set_xticklabels([])
                ax.axes.set_yticks([])
                ax.set_aspect('equal')
                

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
                plt.tight_layout()
                plt.savefig(outPath+out + pad + ".png", dpi=150, transparent=False)
                plt.close()
