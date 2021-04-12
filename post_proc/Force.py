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
partNum=4
end=1

xy_array=np.zeros((1,2,2))

timesteps -= timesteps[0]       # get rid of brownian run time
tst_array=timesteps*dt
del_xy=np.zeros((end-1,partNum,2), dtype=np.float64)
del_tst=np.zeros((end-1), dtype=np.float64)
del_r = np.zeros((end-1,partNum), dtype=np.float64)
v_arr_r = np.zeros((end-1,partNum), dtype=np.float64)
v_arr_x = np.zeros((end-1,partNum), dtype=np.float64)
v_arr_y = np.zeros((end-1,partNum), dtype=np.float64)

# Get number of each type of particle
part_num = 6#len(types[start])
part_A = int(partNum * part_frac_a)
part_B = partNum - part_A

# Feed data into freud analysis software
l_box = box_data[0]
h_box = l_box / 2.0
a_box = l_box * l_box
f_box = box.Box(Lx = l_box, Ly = l_box, is2D = True)    # make freud box

#Calculate net displacement between each time step in x- and y- direction
#Accounts for periodic box
'''
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
#Assign time interval used in fluid velocity calculations
typ = types[1]
direct = orient[1]

#Define array for x and y activity vectors
act_vec = np.zeros((part_num, 2), dtype=np.complex128)
  
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
n=6
n=2
Nk = 0.5*n 
Nk = int(2*np.ceil(Nk/2))
xp=np.linspace(-math.pi, math.pi, n, dtype=np.float64)
yp=np.zeros(n, dtype=np.float64)
#yp=np.linspace(-math.pi, math.pi, n, dtype=np.float64)



act_vec_x=np.zeros(n,dtype=np.complex128)
act_vec_x[0]=150

act_vec_y=np.zeros(n,dtype=np.complex128)
#act_vec_y[1]=150
'''
if n%2==1:
    act_vec[int((n-1)/2)]=500
else:
    act_vec[int((n)/2)]=500
'''
#fft_x = np.zeros(Nk, dtype=np.complex128)
#fft_y = np.zeros(n, dtype=np.complex128)

fft_2x = np.zeros((Nk,Nk,2), dtype=np.complex128)
fft_2y = np.zeros((n,n), dtype=np.complex128)

source_x=np.zeros(n,dtype=np.complex128)
source_y=np.zeros(n,dtype=np.complex128)
#source_x[3]=-150
source_y[0]=150
source = np.zeros((n,2),dtype=np.complex128)
source[:,0]=source_x
source[:,1]=source_y
finufftpy.nufft2d1many(x=xp,y=yp,c=source,isign=1,ms=Nk,mt=Nk,f=fft_2x,eps=float(1*10**-16), modeord=1)
#finufftpy.nufft2d1(x=xp,y=yp,c=source_y,isign=1,ms=n,mt=n,f=fft_2y,eps=float(1*10**-16))

ms=np.zeros(Nk)

if Nk%2==1:
    ms0=-(Nk-1)/2
else:
    ms0=-Nk/2
    
ms0=0
for i in range(0,int(Nk/2)):
    ms[i]=ms0
    ms0=ms0+1
if Nk%2==1:
    ms0=-(Nk-1)/2
    ind0=-(Nk-1)/2
else:
    ms0=-Nk/2
    ind0=-Nk/2

for i in range(0,int(Nk/2)):
    ind=int(ind0+i)
    ms[ind]=ms0
    ms0=ms0+1

mt=np.zeros(Nk)

mt0=0
for i in range(0,int(Nk/2)):
    mt[i]=mt0
    mt0=mt0+1
if Nk%2==1:
    mt0=-(Nk-1)/2
    ind0=-(Nk-1)/2
else:
    mt0=-Nk/2
    ind0=-Nk/2

for i in range(0,int(Nk/2)):
    ind=int(ind0+i)
    mt[ind]=mt0
    mt0=mt0+1
    
kx=ms
ky=mt
kx_arr=np.zeros((Nk,Nk))

ky_arr=np.zeros((Nk,Nk))
for i in range(0,Nk):
    kx_arr[:,i]=ms
    ky_arr[i,:]=mt

fft_x=fft_2x[:,:,0]
fft_y=fft_2x[:,:,1]
k_dot_F_x=np.dot(kx_arr,fft_x)
k_dot_F_y=np.dot(ky_arr,fft_y)
print(np.shape(k_dot_F_x))
inv_kmag=1/np.sqrt(np.square(kx_arr)+np.square(ky_arr))
check = np.isinf(inv_kmag)
for i in range(0,Nk):
    for j in range(0,Nk):
        if check[i,j]==True:
            inv_kmag[i,j]=0

k_F_x=(np.square(inv_kmag))*kx_arr*k_dot_F_x
k_F_y=(np.square(inv_kmag))*ky_arr*k_dot_F_y
v_r_x=(np.square(inv_kmag))*(fft_x-1j*k_F_x)
v_r_y=(np.square(inv_kmag))*(fft_y-1j*k_F_y)
v_recip=np.zeros((Nk,Nk,2))
v_recip[:,:,0]=v_r_x
v_recip[:,:,1]=v_r_y
v_real=np.zeros((n,2), dtype=np.complex128)

finufftpy.nufft2d2many(x=xp,y=yp,c=v_real,isign=-1,f=v_recip,eps=float(1*10**-16), modeord=1)

'''
stop

k_F=np.array([])
stop
k_dot_F_x=np.dot(kx_arr,fft_2x)
print(np.shape(k_dot_F_x))
stop
print(np.shape(k_dot_F_x))
stop
k_dot_F_y=np.dot(kx,fft_2y)
fft_Total=np.array([fft_x,fft_y])
k_dot_F_total=np.array([k_dot_F_x, k_dot_F_y])
K_F=np.array([k_dot_F_total[0]*k,k_dot_F_total[1]*k])
v_r=kmag*(fft_Total-1j*K_F)
'''
fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')

#ax1.quiver(xp, yp, v_r[0,:], v_r[1,:], units='width', label='fluid velocity from t=1 to 2')
ax1.quiver(xp, yp, np.real(v_real[:,0]), np.real(v_real[:,1]), units='width', label='velocity field')
ax1.quiver(xp, yp, source_x, source_y, units='width', label='activity', color='g')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
ax1.plot(xp,yp,'.',c='r')
ax1.set_xlabel('x position')
ax1.set_ylabel('y position')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
plt.legend()
plt.show()
print(mt)
print(ms)

stop

for j in range(0,n):
    act_vec_x=np.zeros(n,dtype=np.complex128)
    j=11
    act_vec_x[j]=-150
    act_vec_y=np.zeros(n,dtype=np.complex128)
    finufftpy.nufft1d1(x=xp,c=act_vec_x,isign=1,eps=float(1*10**-16), ms=n, f=fft_x)
    finufftpy.nufft1d1(x=yp,c=act_vec_y,isign=1,eps=float(1*10**-16), ms=n, f=fft_y)
    kvect=np.zeros((n))
    kmag=np.zeros((n))
    k_ind=0

stop
stop

#finufftpy.nufft1d1(x=xp,c=act_vec_x,isign=-1,eps=float(1*10**-16), ms=n, f=fft_x)
#finufftpy.nufft1d1(x=yp,c=act_vec_y,isign=-1,eps=float(1*10**-16), ms=n, f=fft_y)

#finufftpy.nufft2d1(x=xp,y=yp, c=act_vec,isign=-1,eps=float(1*10**-16), ms=int(n/2), mt=int(n/2), f=fft)


k=np.zeros(n)
if n%2==1:
    k0=-(n-1)/2
else:
    k0=-n/2
for i in range(0,n):
    k[i]=k0
    k0=k0+1
'''
kmag=1/(k**2+k**2)

if n%2==1:
    kmag[int((n-1)/2)]=1
else:
    kmag[int(n/2)]=1
'''

for j in range(0,n):
    act_vec_x=np.zeros(n,dtype=np.complex128)
    j=11
    act_vec_x[j]=-150
    act_vec_y=np.zeros(n,dtype=np.complex128)
    finufftpy.nufft1d1(x=xp,c=act_vec_x,isign=1,eps=float(1*10**-16), ms=n, f=fft_x)
    finufftpy.nufft1d1(x=yp,c=act_vec_y,isign=1,eps=float(1*10**-16), ms=n, f=fft_y)
    kvect=np.zeros((n))
    kmag=np.zeros((n))
    k_ind=0
    #kvect=np.array([k,k])
    '''
    kmag=1/(k**2+k**2)**0.5
    if n%2==1:
        kmag[int((n-1)/2)]=1
    else:
        kmag[int(n/2)]=1
    ident=np.eye(n)
    khat=k*kmag
    print(khat)
    stop
    print(np.shape(ident-khat*khat))
    stop
    print(kmag)
    stop
    '''
    for i in range(0,n):
        kvect[k_ind]=k[i]-k[j]
        k_ind+=1
    kmag=1/(kvect**2)
    kmag[j]=1
    #print()
    
    k_dot_F_x=np.dot(kvect,fft_x)
    k_dot_F_x_total=k_dot_F_x*kmag*kvect*1j
    k_dot_F_y=np.dot(kvect,fft_y)
    k_dot_F_y_total=k_dot_F_y*kmag*kvect*1j
    tot_x=kmag*(fft_x-k_dot_F_x)
    tot_y=kmag*(fft_y-k_dot_F_y)
    fft_2_x = np.zeros(n, dtype=np.complex128)
    fft_2_y = np.zeros(n, dtype=np.complex128)
    finufftpy.nufft1d2(x=xp,c=fft_2_x,isign=1,eps=float(1*10**-15), f=tot_x)
    finufftpy.nufft1d2(x=xp,c=fft_2_y,isign=1,eps=float(1*10**-15), f=tot_y)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Vector Field Real Space')

#ax1.quiver(xp, yp, v_r[0,:], v_r[1,:], units='width', label='fluid velocity from t=1 to 2')
    ax1.quiver(xp, yp, np.real(fft_2_x), np.real(fft_2_y), units='width', label='fluid velocity from t=1 to 2')
    ax1.quiver(xp, yp, act_vec_x, act_vec_y, units='width', label='fluid velocity from t=1 to 2', color='g')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
    ax1.plot(xp,yp,'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
    plt.legend()
    plt.show()
    stop
    stop
    

    
    #*kmag
    print(kmag)
    stop
'''
for j in range(0,n):
    k_ind=0
    kvect=np.zeros((n-1))
    kmag=np.zeros((n-1))
    for i in range(0,n):
        if i==j:
            pass
        else:
            kvect[k_ind]=k[i]-k[j]
            k_ind+=1
    kmag=1/kvect**2
    force=np.append(fft[0:j],fft[j+1:])

    k_dot_F=np.dot(kvect,force)*kmag
    v_r=kmag*(force-1j*k_dot_F*kvect)
    fft_2 = np.zeros(21, dtype=np.complex128)
    finufftpy.nufft1d2(x=xp,c=fft_2,isign=1,eps=float(1*10**-15), f=v_r)
'''
viscosity=1/(3*math.pi)

k_dot_F_x=np.dot(k,fft_x)
k_dot_F_y=np.dot(k,fft_y)
fft_Total=np.array([fft_x,fft_y])
k_dot_F_total=np.array([k_dot_F_x, k_dot_F_y])
K_F=np.array([k_dot_F_total[0]*k,k_dot_F_total[1]*k])
v_r=kmag*(fft_Total-1j*K_F)

fft_2_x = np.zeros(n, dtype=np.complex128)
fft_2_y = np.zeros(n, dtype=np.complex128)

finufftpy.nufft1d2(x=xp,c=fft_2_x,isign=1,eps=float(1*10**-15), f=v_r[0,:])
finufftpy.nufft1d2(x=yp,c=fft_2_y,isign=1,eps=float(1*10**-15), f=v_r[1,:])
fft_2_x=(1/(8*math.pi**3*viscosity))*fft_2_x
fft_2_y=(1/(8*math.pi**3*viscosity))*fft_2_y

fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')

#ax1.quiver(xp, yp, v_r[0,:], v_r[1,:], units='width', label='fluid velocity from t=1 to 2')
ax1.quiver(xp, yp, fft_2_y, fft_2_x, units='width', label='fluid velocity from t=1 to 2')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
ax1.plot(pos_x[:,0],pos_y[:,0],'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
plt.legend()
plt.show()
stop

stop
 
k=np.zeros(n)
if n%2==1:
    k0=-(n-1)/2
else:
    k0=-n/2
for i in range(0,n):
    k[i]=k0
    k0=k0+1


    #if i
#k = [0:Nk/2-1 -Nk/2:-1];

ky=np.zeros(n)
#Calculates vector field in reciprocal space
v_r_prev=np.zeros((part_num,part_num,2))


for j in range(0,n):
    k_ind=0
    kvect=np.zeros((n-1))
    kmag=np.zeros((n-1))
    for i in range(0,n):
        if i==j:
            pass
        else:
            kvect[k_ind]=k[i]-k[j]
            k_ind+=1
    kmag=1/kvect**2
    force=np.append(fft[0:j],fft[j+1:])
    print(force)
    stop
    k_dot_F=np.dot(kvect,force)*kmag
    v_r=kmag*(force-1j*k_dot_F*kvect)
    fft_2 = np.zeros(21, dtype=np.complex128)
    finufftpy.nufft1d2(x=xp,c=fft_2,isign=1,eps=float(1*10**-15), f=v_r)
    print(np.real(fft_2))

    print(v_r)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Vector Field Real Space')

    ax1.quiver(xp, yp, v_r, ky, units='width', label='fluid velocity from t=1 to 2')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
    ax1.plot(xp,yp,'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
    plt.legend()
    plt.show()
    stop

#(1/((1/(3*math.pi))))*
v_r=(1/((1/(3*math.pi))))*kmag*(fft-1j*k_dot_F*k)
plt.plot(k,v_r)
plt.show()
print(v_r)
stop
fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')

ax1.quiver(xp, yp, v_r, ky, units='width', label='fluid velocity from t=1 to 2')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
ax1.plot(xp,yp,'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
plt.legend()
plt.show()
stop
            #k_dot_F=(np.dot(fft_total[i,:],k_vect[:,i]))/(kmag[i]**2)
            #v_r[i]=(1/((1/(3*math.pi))*kmag[i]**2))*(fft_total[i]-k_dot_F*k_vect[:,i])
fft_2 = np.zeros(21, dtype=np.complex128)
print(v_r)
print(xp)
#stop
finufftpy.nufft1d2(x=xp,c=fft_2,isign=1,eps=float(1*10**-15), f=v_r)
stop
fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')

ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width', label='fluid velocity from t=1 to 2')
ax1.quiver(pos_x[:,0], pos_y[:,0], act_vec_x[:,0], act_vec_y[:,0], units='width', color='blue', label='active force vector at t=1')
ax1.quiver(pos_x[:,0], pos_y[:,0], fft_x_1, fft_y_1, units='width', color='green', label='simulation velocity from t=0 to t=1')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
ax1.plot(pos_x[:,0],pos_y[:,0],'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
plt.legend()
plt.show()
stop
finufftpy.nufft1d2(x=pos_y,c=fft_y_2,isign=-1,eps=float(1*10**-15), f=v_r[:,1])


print(kmag)
print(fft)
stop


#Calculate activity vector
for jjj in range(0, part_num):
    #Calculate active force vector given orientation and total activity
   act_vec[jjj] = quatToVector(direct[jjj], typ[jjj])
   
   #Shifts position to range from [-pi,pi] for NUFFT calculation
   pos_x[jjj]=(pos[jjj,0])#/h_box)*math.pi
   pos_y[jjj]=(pos[jjj,1])#/h_box)*math.pi
   
   #Store x- and y- active vector components
   act_vec_x[jjj]=act_vec[jjj,0]
   act_vec_y[jjj]=act_vec[jjj,1]

act_vec_x[0]=150
act_vec_y[0]=0
act_vec_x[1]=0
act_vec_y[1]=0
act_vec_x[2]=0
act_vec_y[2]=0
act_vec_x[3]=0
act_vec_y[3]=0
act_vec_x[4]=0
act_vec_y[4]=0
act_vec_x[5]=0
act_vec_y[5]=0
activity = np.zeros((part_num,1), dtype=np.complex128)
for i in range(part_num):
    activity[i]=150
#act_vec_x=np.zeros((part_num,1), dtype=np.complex128)
#act_vec_y=np.zeros((part_num,1),dtype=np.complex128)
#act_vec_x[2]=5000
#Calculate active force in reciprocal space in x- and y- direction
Nk=int(part_num*0.5)
#Nk=2*np.ceil(Nk/2)
act_vec=np.array([act_vec_x,act_vec_y])
act_vec=np.reshape(act_vec[:,:,0],(part_num,2))

pos_x = (pos_x/h_box)*math.pi
pos_y=(pos_y/h_box)*math.pi

pos_x[0]=0.0
pos_y[0]=0.0
pos_x[1]=0.1
pos_y[1]=0.00000001
pos_x[2]=-0.1
pos_y[2]=-0.00000001
pos_x[3]=0.00000002
pos_y[3]=0.1
pos_x[4]=-0.0000002
pos_y[4]=-0.1

activity_mat=np.zeros((part_num, part_num),dtype=np.complex128)

x2=np.zeros(2,dtype=np.float64)
y2=np.zeros(2,dtype=np.float64)
c2=np.zeros(2,dtype=np.complex128)

finufftpy.nufft1d1(x=pos_x,c=act_vec_x,isign=1,eps=float(1*10**-15), ms=part_num, f=fft_x_1)
finufftpy.nufft1d1(x=pos_y,c=act_vec_y,isign=1,eps=float(1*10**-15), ms=part_num, f=fft_y_1)

k=np.zeros(part_num)
k0=-part_num/2
for i in range(0,part_num):
    k[i]=k0
    k0=k0+1
kmag=1/(2*k**2)
kmag[int(part_num/2)]=1

#Initiate empty array for active force in reciprocal space
fft_total = np.zeros((part_num,2), dtype=np.complex128)

#Store active force in reciprocal space in x- and y- direction
fft_total[:,0]=fft_x_1
fft_total[:,1]=fft_y_1

k_vect=np.array([k, k])

#[kx ky] = ndgrid(k,k);
#kfilter = 1/(kx.^2+ky.^2); % inverse -Laplacian in k-space (as above)
#kfilter(1,1) = 0; kfilter(Nk/2+1,:) = 0; kfilter(:,Nk/2+1) = 0;
    
#Initiate reciprocal (v_r) and real-space (v_r_real) fluid vector field
v_r=np.zeros((part_num,2), dtype=np.complex128)
v_r_real = np.zeros((part_num,2))

#Calculates vector field in reciprocal space
v_r_prev=np.zeros((part_num,part_num,2))
for i in range(0,part_num):
    print('i')
    print(i)
    for j in range(0,part_num):
        if i==j:
            v_r_prev[i,j]=0
        else:
            k_v = k_vect[:,j]-k_vect[:,i]
            kvectmag=(k_v[0]**2+k_v[1]**2)**0.5
            k_v=np.reshape(k_v, (1,2))
            force = np.reshape(fft_total[i,:], (2,1))
            k_dot_F=(np.dot(k_v, force))/(kvectmag**2)
            v_r_prev[i,j,:]=(1/((1/(3*math.pi))*kvectmag**2))*(fft_total[i,:]-k_dot_F*k_v)
            print(v_r_prev)
        
    v_r=np.sum(v_r_prev,1)

            #k_dot_F=(np.dot(fft_total[i,:],k_vect[:,i]))/(kmag[i]**2)
            #v_r[i]=(1/((1/(3*math.pi))*kmag[i]**2))*(fft_total[i]-k_dot_F*k_vect[:,i])
finufftpy.nufft1d2(x=pos_x,c=fft_x_2,isign=-1,eps=float(1*10**-15), f=v_r[:,0])
finufftpy.nufft1d2(x=pos_y,c=fft_y_2,isign=-1,eps=float(1*10**-15), f=v_r[:,1])
print(fft_x_2)
print(fft_y_2)
v_r_real[:,0]=(1/(8*math.pi**3))*fft_x_2
v_r_real[:,1]=(1/(8*math.pi**3))*fft_y_2

fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')
print(np.shape(fft_x_1))
print(np.shape(fft_y_1))

ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width', label='fluid velocity from t=1 to 2')
ax1.quiver(pos_x[:,0], pos_y[:,0], act_vec_x[:,0], act_vec_y[:,0], units='width', color='blue', label='active force vector at t=1')
ax1.quiver(pos_x[:,0], pos_y[:,0], fft_x_1, fft_y_1, units='width', color='green', label='simulation velocity from t=0 to t=1')

#Q = ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width')

#qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
ax1.plot(pos_x[:,0],pos_y[:,0],'.',c='r')
#plt.xlim([-h_box,h_box])
#plt.ylim([-h_box,h_box])
plt.legend()
plt.show()
stop
stop

#Calculate fraction of displacement in x- relative to y- direction
a=np.abs(v_arr_x[0,:])/np.abs(v_arr_y[0,:])

#Calculate angular frequency
freq=(2*math.pi)/del_tst

#Initiate empty wave vectors
k_val = np.zeros((end-1,partNum), dtype=np.float64)
k_val_x = np.zeros((end-1,partNum), dtype=np.float64)
k_val_y = np.zeros((end-1,partNum), dtype=np.float64)

#Calculate total wave vector
for i in range(0, part_num):
    k_val[:,i]=freq/v_arr_r[:,i]
    #k_val_x[:,i]=2*math.pi*freq/v_arr_x[:,i]
    #k_val_y[:,i]=2*math.pi*freq/v_arr_y[:,i]
#Calculate wavevector in x and y direction since k proportional to displacement
k_val_y=(k_val**2/(1+a**2))**0.5
k_val_x=(k_val**2-k_val_y**2)**0.5

#Define x-,y- wave vector
k_vect = np.array([k_val_x,k_val_y])

#Initiate reciprocal (v_r) and real-space (v_r_real) fluid vector field
v_r=np.zeros((part_num,2), dtype=np.complex128)
v_r_real = np.zeros((part_num,2))

#Calculates vector field in reciprocal space
for i in range(0,part_num):
    k_dot_F=(np.dot(fft_total[i,:],k_vect[:,0,i]))/(k_val[0,i]**2)
    v_r[i]=(1/((1/(3*math.pi))*k_val[0,i]**2))*(fft_total[i]-k_dot_F*k_vect[:,0,i])
 
#Adjoint Fourier Transform (Adjoint is the inverse for FT)
finufftpy.nufft1d2(x=k_val_x[0,:],c=fft_x_2,isign=1,eps=float(1*10**-15), f=v_r[:,0])
finufftpy.nufft1d2(x=k_val_y[0,:],c=fft_y_2,isign=1,eps=float(1*10**-15), f=v_r[:,1])


v_r_real[:,0]=fft_x_2
v_r_real[:,1]=fft_y_2

pos_x=(pos_x/math.pi)*h_box
pos_y=(pos_y/math.pi)*h_box

fig1, ax1 = plt.subplots()
ax1.set_title('Vector Field Real Space')

ax1.quiver(pos_x[:,0], pos_y[:,0], v_r_real[:,0], v_r_real[:,1], units='width', label='fluid velocity from t=1 to 2')
ax1.quiver(pos_x[:,0], pos_y[:,0], act_vec_x[:,0], act_vec_y[:,0], units='width', color='blue', label='active force vector at t=1')
ax1.quiver(pos_x[:,0], pos_y[:,0], v_arr_x[0,:], v_arr_y[0,:], units='width', color='green', label='simulation velocity from t=0 to t=1')

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