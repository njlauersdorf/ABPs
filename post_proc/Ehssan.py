#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:10:34 2020

@author: nicklauersdorf
"""
import matplotlib.pyplot as plt
import numpy as np
#import vtk
#from mayavi.scripts import mayavi2

gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_26_20_parent/'

# need to extract values from filename (pa, pb, xa) for naming
part_perc_a = int(50)
part_frac_a = float(part_perc_a) / 100.0
pe_a = int(150)
pe_b = int(150)
ep = int(1)
partNum=int(10)

#sys.path.append(hoomd_path)

import hoomd
from hoomd import md
from hoomd import deprecated

#initialize system randomly, can specify GPU execution here

################################################################################
############################# Begin Data Analysis ##############################
################################################################################

#sys.path.append(gsd_path)
import gsd
from gsd import hoomd
from gsd import pygsd
import numpy as np
import math

def quat_to_theta(quat):
    
    x = quat[1]
    y = quat[2]
    rad = math.atan2(y,x)/np.pi # gives values from [-1,1]

    return rad

myfile = "pa" + str(pe_a) + "_pb" + str(pe_b) + "_xa" + str(part_perc_a) + "_ep" + str(ep) + "_pNum" + str(partNum)+".gsd"
print(myfile)

f = hoomd.open(name=gsdPath+myfile, mode='rb')
dumps = f.__len__()
size_min = 10                                           # minimum size of cluster

position_array = np.zeros((dumps), dtype=np.ndarray)    # array of position arrays
type_array = np.zeros((dumps), dtype=np.ndarray)        # particle types
box_data = np.zeros((1), dtype=np.ndarray)              # box dimensions
timesteps = np.zeros((dumps), dtype=np.float64)         # timesteps
orientations = np.zeros((dumps), dtype=np.ndarray)      # orientations
velocities = np.zeros((dumps), dtype=np.ndarray)

start = 0
stop = dumps

with hoomd.open(name=gsdPath+myfile, mode='rb') as t:           # open for reading
    snap = t[0]                                         # snap 0th snapshot
    box_data = snap.configuration.box                   # get box dimensions
    for i in range(start, stop):
        snap = t[i]                                     # take snap of each dump
        type_array[i] = snap.particles.typeid
        position_array[i] = snap.particles.position     # store all particle positions
        timesteps[i] = snap.configuration.step          # store tstep for plotting purposes
        orientations[i] = snap.particles.orientation    # store the orientation of all particles
        velocities[i] = snap.particles.velocity         #store the velocity of all particles
pos=position_array[0]
x=pos[:,0]
x.sort()
y=pos[:,1]
y.sort()
box_l=box_data[0]/2
box_w=box_data[1]/2
stop
x_vals = np.zeros(len(x)*11+1)
y_vals = np.zeros(len(y)*11+1)
print(len(y_vals))
stop
x_vals[0:11]=np.linspace(box_w*-1,x[0],11)
x_vals[len(y_vals)-11:len(y_vals)]=np.linspace(x[len(x)-1],box_w,11)
y_vals[0:11]=np.linspace(box_l*-1,y[0],11)
y_vals[len(y_vals)-11:len(y_vals)]=np.linspace(y[len(y)-1],box_l,11)
print(x_vals)
for i in range(0,len(x)-1):
    x_vals[(i+1)*10:(i+2)*10+1]=np.linspace(x[i],x[i+1],11)
    y_vals[(i+1)*10:(i+2)*10+1]=np.linspace(y[i],y[i+1],11)
print(x_vals)
print()
print(y)


stop
#inFile='phase_density_pa'+str(pa)+'_pb'+str(pb)+'_xa'+str(xa)+'_phi'+str(phi)+'_ep1.000.txt'
#f= open(gsdPath+inFile, "rb")