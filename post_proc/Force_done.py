#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:25:05 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
This is going to be SUPER easy
    -look at particle type
    -get orientation
    -vectorize active force
    -sum force in each bin (this accounts for orientation)
    -take magnitude of summed quantity
    
    We'll plot this along with some other contributing heatmaps
    1. Original sim
    1. Binned orientation
    3. Binned active force magnitudes
    4. Magnitude of binned vector forces
'''

# Imports and loading the .gsd file
import sys
import pyculib
import finufftpy

gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_26_20_parent/'
hoomdPath='/Users/nicklauersdorf/hoomd-blue/build'

# need to extract values from filename (pa, pb, xa) for naming

part_perc_a = int(50)
part_frac_a = float(part_perc_a) / 100.0
pe_a = int(150)
pe_b = int(150)
ep = int(1)
partNum=int(10)

myfile = "pa" + str(pe_a) + "_pb" + str(pe_b) + "_xa" + str(part_perc_a) + "_ep" + str(ep) + "_pNum" + str(partNum)+".gsd"
print(myfile)               # local path to gsd

#sys.path.insert(0,hoomdPath)     # ensure hoomd is in your python path
#sys.path.insert(0,gsdPath)       # ensure gsd is in your python path

import hoomd
from hoomd import md
from hoomd import deprecated

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import box
from freud import density
from freud import Box


import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import colors

import math

def quatToVector(quat, type):
    "Takes quaternion, returns orientation vector"
    if type == 0:
        mag = pe_a
    else:
        mag = pe_b
    print(quat)
    
    angle = 2 * math.acos(quat[0])
    x = (quat[1] / math.sqrt(1.-quat[0]**2))*mag
    y = (quat[2] / math.sqrt(1.-quat[0]**2))*mag
    act_vec = (x, y)
    return act_vec

def getMagnitude(vecF):
    "Take force vector, output magnitude"
    x = vecF[0]
    y = vecF[1]
    magF = np.sqrt((x**2)+(y**2))
    return magF

def quatToAngle(quat):
    "Take quaternion, output angle between [0, 2pi]"
    x = (quat[1] / math.sqrt(1.-quat[0]**2))#quat[1]
    y = (quat[2] / math.sqrt(1.-quat[0]**2))#quat[2]
    theta = math.atan2(y, x)
    #theta += np.pi
    print(theta)
    stop
    return theta

def vecToAngle(vec):
    "Take vector, output angle between [-pi, pi]"
    x = vec[0]
    y = vec[1]
    theta = math.atan2(y, x)
    return theta

# Make particle colormap
#colorsList = [(255,0,255),(50,205,50)]
colorsList = ['#FF00FF','#39FF14']
my_cmap = colors.ListedColormap(colorsList)

# File to read from
in_file = "pa"+str(pe_a)+\
"_pb"+str(pe_b)+\
"_xa"+str(part_perc_a)+\
".gsd"

fil = hoomd.open(name=gsdPath+myfile, mode='rb') # open gsd file with hoomd
dumps = fil.__len__()                     # get number of timesteps dumped

start = 0       # gives first frame to read
end = dumps     # gives last frame to read
#start = 154
#end = 234
dt=0.000001
positions = np.zeros((end), dtype=np.ndarray)       # array of positions
types = np.zeros((end), dtype=np.ndarray)           # particle types
box_data = np.zeros((1), dtype=np.ndarray)          # box dimensions
timesteps = np.zeros((end), dtype=np.float64)       # timesteps
orient = np.zeros((end), dtype=np.ndarray)          # orientations
time_dif=np.zeros((end-1), dtype=np.ndarray) 
radial_pos = np.zeros((end), dtype=np.ndarray) 
radial_pos_dif=np.zeros((end-1), dtype=np.ndarray) 

# Get relevant data from .gsd file
with hoomd.open(name=gsdPath+myfile, mode='rb') as t:
    snap = t[0]
    box_data = snap.configuration.box
    xy_array = np.zeros((end,partNum,2))
    tst_array = np.zeros(end)
    for iii in range(start, end):
        snap = t[iii]                               # snapshot of frame
        types[iii] = snap.particles.typeid          # get types
        pos=snap.particles.position
        positions[iii] = pos    # get positions
        orient[iii] = snap.particles.orientation    # get orientation
        timesteps[iii] = snap.configuration.step    # get timestep
        xy = np.delete(pos, 2, 1)                   # Deletes z-position
        
        xy_array[iii]=xy 
    #    radial_pos[iii]=(snap.particles[iii+1,:,0]**2+snap.particles[iii+1,:,1]**2)

    #for j in range(start+1,end):
    #    del_xy[j-1]=xy_array[j]-xy_array[j-1]
    #    for i in range(start+1,partNum):
    #        if abs(del_xy[j-1,i-1,0])>h_box: 
            
    #            del_xy[j-1,i-1,0]=l_box-abs(del_xy[j-1,i-1,0])
                
    #        if abs(del_xy[j-1,i-1,1])>h_box:
    #            del_xy[j-1,i-1,1]=l_box-abs(del_xy[j-1,i-1,1])
            
    #for j in range(start+1,end-1):
    #    del_r[j-1]=(del_xy[j,:,0]**2+del_xy[j,:,1]**2)**0.5-(del_xy[j-1,:,0]**2+del_xy[j-1,:,1]**2)**0.5


    #for i in range(start,end-1):
    #    time_dif[i]=(timesteps[i+1]-timesteps[i])*dt
        #radial_pos_dif[i]=(radial_pos[i+1]-radial
for i in range(0,100):
    timesteps -= timesteps[0]       # get rid of brownian run time
    tst_array=timesteps*dt
    del_xy=np.zeros((end-1,partNum,2), dtype=np.float64)
    del_tst=np.zeros((end-1), dtype=np.float64)
    del_r = np.zeros((end-1,partNum), dtype=np.float64)
    v_arr_r = np.zeros((end-1,partNum), dtype=np.float64)
    v_arr_x = np.zeros((end-1,partNum), dtype=np.float64)
    v_arr_y = np.zeros((end-1,partNum), dtype=np.float64)

    # Get number of each type of particle
    part_num = len(types[start])
    part_A = int(partNum * part_frac_a)
    part_B = partNum - part_A

    # Feed data into freud analysis software
    l_box = box_data[0]
    h_box = l_box / 2.0
    a_box = l_box * l_box
    f_box = box.Box(Lx = l_box, Ly = l_box, is2D = True)    # make freud box

#Calculate net displacement between each time step in x- and y- direction
#Accounts for periodic box
    for j in range(start+1,end):
        del_xy[j-1]=xy_array[j]-xy_array[j-1]
        for i in range(start+1,partNum):
            if abs(del_xy[j-1,i-1,0])>h_box: 
            
                del_xy[j-1,i-1,0]=l_box-abs(del_xy[j-1,i-1,0])
                
            if abs(del_xy[j-1,i-1,1])>h_box:
                del_xy[j-1,i-1,1]=l_box-abs(del_xy[j-1,i-1,1])

#Calculate difference between time steps and the radial displacement 
    for j in range(start+1,end):
        del_tst[j-1]=tst_array[j]-tst_array[j-1]
        del_r[j-1]=(abs(del_xy[j-1,:,0])**2+abs(del_xy[j-1,:,1])**2)**0.5
    
#Calculate radial velocity
    for j in range(0,partNum):
        v_arr_r[:,j]=del_r[:,j]/del_tst
        v_arr_x[:,j]=del_xy[:,j,0]/del_tst
        v_arr_y[:,j]=del_xy[:,j,1]/del_tst


'''
a= f_box.get_box_vector(0)[0:2]
b= f_box.get_box_vector(1)[0:2]
real_lat_vec = np.array([a,b])

recip_lat_vec=(2*math.pi/(a[0]*b[1]-b[0]*a[1]))*np.array([[b[1], -a[1]],[-b[0], a[0]]])
recip_a = recip_lat_vec[:,0]
recip_b = recip_lat_vec[:,1]

# Make the mesh
nBins = 100
sizeBin = l_box / nBins

pos = positions[1]
pos = np.delete(pos, 2, 1)
'''

    typ = types[1]
    direct = orient[1]


#Define array for x and y activity vectors
    act_vec = np.zeros((part_num, 2), dtype=np.complex128)
    
#Define mesh for locations of x and y activity vectors
#mesh = np.zeros((nBins, nBins, 2), dtype=np.complex128)
    
#Define array for positions
#pos_arr = np.zeros((nBins, nBins, 2), dtype=np.float64)
    
#x_ft_array = np.zeros((nBins,nBins))
#y_ft_array = np.zeros((nBins,nBins))
    
#Define x and y positions of mesh
#x_pos=np.linspace(0,l_box,nBins)
#y_pos=np.linspace(0,l_box,nBins)
    
#Empty arrays for Forward Fourier Transform from Real to Reciprocal Space
    fft_x_1=np.zeros(part_num, dtype=np.complex128)
    fft_y_1=np.zeros(part_num, dtype=np.complex128)

#Empty arrays for Inverse Fourier Transform from Reciprocal to Real Space
    fft_x_2=np.zeros(part_num, dtype=np.complex128)
    fft_y_2=np.zeros(part_num, dtype=np.complex128)

#Assign empty arrays
    pos_new=np.zeros((part_num,2),dtype=np.float64)
    pos_x=np.zeros((part_num,1),dtype=np.float64)
    pos_y=np.zeros((part_num,1),dtype=np.float64)
    act_vec_x=np.zeros((part_num,1), dtype=np.complex128)
    act_vec_y=np.zeros((part_num,1),dtype=np.complex128)

#Calculate activity vector
    for jjj in range(0, part_num):
        act_vec[jjj] = quatToVector(direct[jjj], typ[jjj])
   #Shifts position to range from [-pi,pi] for NUFFT calculation
        pos_x[jjj]=(pos[jjj,0])#/h_box)*math.pi
        pos_y[jjj]=(pos[jjj,1])#/h_box)*math.pi
        act_vec_x[jjj]=act_vec[jjj,0]
        act_vec_y[jjj]=act_vec[jjj,1]

    finufftpy.nufft1d1(x=pos_x,c=act_vec_x,isign=-1,eps=float(1*10**-15), ms=part_num, f=fft_x_1)
    finufftpy.nufft1d1(x=pos_y,c=act_vec_y,isign=-1,eps=float(1*10**-15), ms=part_num, f=fft_y_1)
    fft_total = np.zeros((part_num,2), dtype=np.complex128)
    fft_total[:,0]=fft_x_1
    fft_total[:,1]=fft_y_1


'''
k_vals = np.zeros((part_num,1), dtype=np.float64)
k0=(-12/2)
for i in range(0,part_num):
    k_vals[i]= k0+1
    k0=k_vals[i]
k_vec_x=k_vals[:,0]*orient[0][:,0]
k_vec_y=k_vals[:,0]*orient[0][:,1]
k_vec = np.zeros((2,part_num))
k_vec[0]=k_vec_x
k_vec[1]=k_vec_y
'''

#Calculate fraction of displacement in either x or y direction

    print(np.shape(v_arr_x))
    a=np.abs(v_arr_x[0,:])/np.abs(v_arr_y[0,:])


#Calculate wavevector
    freq=(2*math.pi)/del_tst

    k_val = np.zeros((end-1,partNum), dtype=np.float64)
    k_val_x = np.zeros((end-1,partNum), dtype=np.float64)
    k_val_y = np.zeros((end-1,partNum), dtype=np.float64)

    for i in range(0, part_num):
        k_val[:,i]=freq/v_arr_r[:,i]
    #k_val_x[:,i]=2*math.pi*freq/v_arr_x[:,i]
    #k_val_y[:,i]=2*math.pi*freq/v_arr_y[:,i]

#k_vect_trans=(((k_vect/np.max(k_vect))-0.5)*2*math.pi)

#Calculate wavevector in x and y direction since k proportional to displacement
    k_val_y=(k_val**2/(1+a**2))**0.5
    k_val_x=(k_val**2-k_val_y**2)**0.5

#print(k_vect_y)
#k_vect_x=(k_val**2-k_vect_y**2)**0.5
#print(k_vect_x)

#Calculate actual k vector
#k_vect=np.array([k_vect_x,k_vect_y], dtype=np.float64)

#Make positions in k-space range from [-pi,pi] due to NUFFT
#k_vect_trans=(((k_vect/np.max(k_vect))-0.5)*2*math.pi)
#k_val_trans = (k_vect_trans[0,:,:]**2+k_vect_trans[1,:,:]**2)**0.5

#Assign separate matrices for k vector in x and y direction
#k_vect_x=k_vect_trans[0,0,:]
#k_vect_y=k_vect_trans[1,0,:]

    v_r=np.zeros((part_num,2), dtype=np.complex128)
    v_r_real = np.zeros((part_num,2))
#k_vect_x = k_vect_x.astype('float64') 
#k_vect_y = k_vect_y.astype('float64') 
    k_vect = np.array([k_val_x,k_val_y])

    for i in range(0,part_num):
        k_dot_F=(np.dot(fft_total[i,:],k_vect[:,0,i]))/(k_val[0,i]**2)
        v_r[i]=(1/(k_val[0,i]**2))*(fft_total[i]-k_dot_F*k_vect[:,0,i])
 
#Adjoint Fourier Transform (Adjoint is the inverse for FT)
    finufftpy.nufft1d2(x=k_val_x[0,:],c=fft_x_2,isign=1,eps=float(1*10**-15), f=v_r[:,0])
    finufftpy.nufft1d2(x=k_val_y[0,:],c=fft_y_2,isign=1,eps=float(1*10**-15), f=v_r[:,1])


    v_r_real[:,0]=fft_x_2
    v_r_real[:,1]=fft_y_2

    fig1, ax1 = plt.subplots()
    ax1.set_title('Vector Field Real Space')

    ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width', label='fluid velocity')
    ax1.quiver(pos_x[:,0], pos_y[:,0], act_vec_x[:,0], act_vec_y[:,0], units='width', color='blue', label='active force vector')
    ax1.quiver(pos_x[:,0], pos_y[:,0], v_arr_x[0,:], v_arr_y[0,:], units='width', color='green', label='simulation velocity')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
    ax1.plot(pos_x[:,0],pos_y[:,0],'.',c='r')
    plt.legend()
    plt.show()
stop
stop
#for iii in range(start, end):
    
#    fft_x_1*
for iii in range(start, end):

    # Array to hold the magnitude of the binned force
    binnedF = np.zeros((nBins, nBins), dtype=np.float32)
    # And binned orientation
    binnedO = np.zeros((nBins, nBins), dtype=np.float32)
    # And binned force magnitude
    binnedM = np.zeros((nBins, nBins), dtype=np.float32)
    # And occupancy of each index
    binnedD = np.zeros((nBins, nBins), dtype=np.float32)
    
    # Define particle positions
    pos = positions[iii]
    # Remove z-locations
    pos = np.delete(pos, 2, 1)
    
    
    typ = types[iii]
    direct = orient[iii]
    
    #Define array for x and y activity vectors
    act_vec = np.zeros((part_num, 2), dtype=np.complex128)
    
    #Define mesh for locations of x and y activity vectors
    mesh = np.zeros((nBins, nBins, 2), dtype=np.complex128)
    
    #Define array for positions
    pos_arr = np.zeros((nBins, nBins, 2), dtype=np.float64)
    
    x_ft_array = np.zeros((nBins,nBins))
    y_ft_array = np.zeros((nBins,nBins))
    
    #Define x and y positions of mesh
    x_pos=np.linspace(0,l_box,nBins)
    y_pos=np.linspace(0,l_box,nBins)
    
    #Empty arrays for Forward Fourier Transform
    fft_x=np.zeros(np.shape(act_vec), dtype=np.complex128)
    fft_y=np.zeros(np.shape(act_vec), dtype=np.complex128)
    
    fft_x_1=np.empty(np.shape(act_vec[:,0]), dtype=np.complex128)
    fft_y_1=np.empty(np.shape(act_vec[:,1]), dtype=np.complex128)
    
    #Empty arrays for Inverse Fourier Transform
    
    #for i in range(0,nBins): 
    #    pos_arr[:,i,1]=y_pos
    #    pos_arr[i,:,0]=x_pos
    #pos_shift=(((pos+h_box)/l_box)*2*math.pi)

    pos_new=np.zeros((part_num,2),dtype=np.float64)
    pos_x=np.zeros((part_num,1),dtype=np.float64)
    pos_y=np.zeros((part_num,1),dtype=np.float64)
    act_vec_x=np.zeros((part_num,1), dtype=np.complex128)
    act_vec_y=np.zeros((part_num,1),dtype=np.complex128)
    # Computes the x and y components of each particles' active force
    for jjj in range(0, part_num):
        act_vec[jjj] = quatToVector(direct[jjj], typ[jjj])
        #pos_new[jjj]=pos_shift[jjj]
        pos_x[jjj]=pos[jjj,0]#(((pos[jjj,0])/h_box)*math.pi)
        pos_y[jjj]=pos[jjj,1]#(((pos[jjj,1])/h_box)*math.pi)
        act_vec_x[jjj]=act_vec[jjj,0]
        act_vec_y[jjj]=act_vec[jjj,1]
    #pos_y=pos_new[:,1]
    #for jjj
    #pos_x=pos_new[:,0]
    #pos_y=pos_new[:,1]

    finufftpy.nufft1d2(x=pos_x,c=act_vec_x,isign=-1,eps=float(1*10**-15),f=fft_x_1)
    finufftpy.nufft1d2(x=pos_y,c=act_vec_y,isign=-1,eps=float(1*10**-15),f=fft_y_1)
    print(fft_x_1)
    print(fft_y_1)
    fft_x_1
    finufftpy.nufft1d2(x=pos_x,c=act_vec_x,isign=-1,eps=float(1*10**-15),f=fft_x_1)
    finufftpy.nufft1d2(x=pos_y,c=act_vec_y,isign=-1,eps=float(1*10**-15),f=fft_y_1)
    
    stop
    finufftpy.nufft1d2(x=pos_x,c=act_vec_x,isign=1,eps=float(1*10**-15),f=fft_x_1)
    finufftpy.nufft1d2(x=pos_y,c=act_vec_y,isign=1,eps=float(1*10**-15),f=fft_y_1)
    print(fft_x_1)
    print(ff_y_1)
    stop
    finufftpy.nufft2d2(x=pos_new[:,0], y=pos_new[:,1],c=act_vec[:,0],isign=1,eps=float(1*10**-15),f=fft_x)
    finufftpy.nufft2d2(x=pos_new[:,0], y=pos_new[:,1],c=act_vec[:,1],isign=1,eps=float(1*10**-15),f=fft_y)
    stop

    # Take vector sum in each bin (will have x and y components)
    for jjj in range(0, part_num):
        # Get mesh indices
        tmp_posX = pos[jjj][0] + h_box
        tmp_posY = pos[jjj][1] + h_box
        
        x_ind = int(tmp_posX / sizeBin)
        y_ind = int(tmp_posY / sizeBin)
        
        # Sum vector active force
        mesh[x_ind][y_ind][0] += act_vec[jjj][0]
        mesh[x_ind][y_ind][1] += act_vec[jjj][1]
        
        # Sum magnitude of each force
        if typ[jjj] == 0:
            binnedM[x_ind][y_ind] += pe_a
        else:
            binnedM[x_ind][y_ind] += pe_b
        # Get occupancy of each index
        binnedD[x_ind][y_ind] += 1

        #finufftpy.nufft1d1(pos[jjj][1],act_vec[jjj][1])
    # Take magnitude of each bin (removes directional component)
    #finufftpy.nufft1d1(x=pos_arr[:,:,0], y=pos_arr[:,:,1],c=mesh[:,:,0],isign=1,eps=float(1*10**-16),ms=1,mt=1,f=a)
    print('blah')
    finufftpy.nufft2d2(x=pos_arr[:,:,0], y=pos_arr[:,:,1],c=mesh[:,:,0],isign=1,eps=float(1*10**-15),f=fft_x)
    finufftpy.nufft2d2(x=pos_arr[:,:,0], y=pos_arr[:,:,1],c=mesh[:,:,1],isign=1,eps=float(1*10**-15),f=fft_y)
    print(fft_x)
    print(fft_y)
    stop
    
    print(x_pos)
    for jjj in range(0, nBins):
        for mmm in range(0, nBins):
            binnedF[jjj][mmm] += getMagnitude(mesh[jjj][mmm])
            binnedO[jjj][mmm] = vecToAngle(mesh[jjj][mmm])
    print(pos[0][0])
    stop
    
    #cufftPlan2D()
    #cufftExecD2Z(mesh[:,:,0])
    #cufftPlan2D()
    #cufftExecD2Z(mesh[:,:,1])
    print(np.shape(mesh))
    stop
    pyculib.fft.fft(1)
    stop
    '''
    fig = plt.figure()
    # Plot the original simulation
    ax = fig.add_subplot(221)
    ax.scatter(pos[:,0], pos[:,1], c=typ, cmap=my_cmap, s=0.05, edgecolors='none')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(-h_box, h_box)
    plt.ylim(-h_box, h_box)
    ax.set_aspect('equal')
    plt.title('Original') 

    # Plot binned orientation
    ax = fig.add_subplot(222)
    plt.imshow(binnedO.T,
               extent=(0,nBins,0,nBins),
               origin='lower',
               cmap=plt.get_cmap('hsv'))
    plt.xticks(())
    plt.yticks(())
    cb = plt.colorbar()
#    cb.set_ticks([])
    ax.set_aspect('equal')
    plt.title('Orientation')

    # Plot binned summed force magnitudes
    ax = fig.add_subplot(223)
    plt.imshow(binnedM.T,
               extent=(0,nBins,0,nBins),
               origin='lower')
    plt.xticks(())
    plt.yticks(())
    cb = plt.colorbar()
#    cb.set_ticks([])
    ax.set_aspect('equal')
    plt.title('Summed ||Force||')

    # Plot binned data using imshow
    ax = fig.add_subplot(224)
    plt.imshow(binnedF.T,
               extent=(0,nBins,0,nBins),
               origin='lower',
               clim=(0,10000))
    plt.xticks(())
    plt.yticks(())
    cb = plt.colorbar()
#    cb.set_ticks([])
    ax.set_aspect('equal')
    plt.title('Summed Vector Force')
    plt.show()
    # Figure name
#    plt.show()
    #plt.savefig('nBins' + str(nBins) +
    #            '_pa'+ str(pe_a) +
    #            '_pb'+ str(pe_b) +
    #            '_xa'+ str(part_perc_a) +
    #            '_step_'+ str(iii) +
    #            '.png', dpi=1000)
    #plt.close()
    '''