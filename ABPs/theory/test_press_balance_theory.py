#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:20:51 2021

@author: nicklauersdorf
"""

import sys
import os

import math
import numpy as np
import random
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick

#Constants
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)


#Functions
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

def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ


def compPeNet(xf, pes, pef):
    "Given each species activity (pes and pef) and particle fraction (xf), compute net activity (peNet)"
    peNet = (pes * (1.-xf)) + (pef * xf)
    return peNet
def avgCollisionForce(peNet):
    '''Computed from the integral of possible angles'''
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    return (magnitude * peNet) / (np.pi)  
def ljForce(r, eps, sigma=1.):
    '''Compute the Lennard-Jones force'''
    div = (sigma/r)
    dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
    return dU

# Lennard-Jones pressure
def ljPress(r, pe, eps, sigma=1.):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
    activity (pe), and separation distance (r) of 2 particles to compute pressure from
    avg compressive active forces from neighbors
    
    Inputs: 
        r: Separation distance in simulation units
        epsilon: magnitude of lennard-jones potential
        pe: activity (peclet number)
        sigma: particle diameter (default=1.0)
    
    Output: Analytical virial pressure (see monodisperse paper for derivation)
    '''
    #Area fraction at HCP
    phiCP = np.pi / (2. * np.sqrt(3.))
    
    # LJ force
    ljF = avgCollisionForce(pe)
    
    return (2. *np.sqrt(3) * ljF / r)

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
def latToPhi(latIn):
    '''Read in lattice spacing, output phi'''
    phiCP = np.pi / (2. * np.sqrt(3.))
    return phiCP / (latIn**2)
def clustFrac(phi, phiG, aF, aS, xF, sig=1.):
    '''Compute the fraction of particles in the cluster'''
    if xF == 0.:
        phiLS = latToPhi(aS)
        phiLF = 1.
    elif xF == 1.:
        phiLS = 1.
        phiLF = latToPhi(aF)
    else:
        phiLS = latToPhi(aS)
        phiLF = latToPhi(aF)
    coeff = (phiG - phi) / phi
    num = phiLF * phiLS
    den = ( phiG * ((phiLS*xF) + (phiLF*(1.-xF))) ) - (phiLF * phiLS)
    ans = coeff * num / den
    return ans
#Calculate gas phase area fraction
def compPhiG(pe, a, kap=4.5, sig=1.):
    '''
    Purpose: Compute analytical area fraction of the gas phase at steady state
    given activity and lattice spacing
    
    Inputs: 
        pe: net activity (peclet number)
        a: lattice spacing 
        kap: fitting parameter (default=4.5, shown by Redner)
        sig: particle diameter (default=1.0)
    
    Output: Area fraction of the gas phase at steady state
    '''
    num = 3. * (np.pi**2) * kap * sig
    den = 4. * pe * a
    if den>0:
        return num / den
    else:
        return 0
# Calculate dense phase area fraction from lattice spacing
def latToPhi(latIn):
    '''
    Purpose: Compute analytical area fraction of the dense phase given the lattice
    spacing.
    
    Inputs: 
        latIn: lattice spacing
    
    Output: dense phase area fraction
    '''
    phiCP = np.pi / (2. * np.sqrt(3.))
    return phiCP / (latIn**2)
def areaType(Nx, latx):
    Ax = Nx * np.pi * 0.25 * (latx**2)
    return Ax

#Slow activities interested in
pe_a = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
#Fast activities interested in
pe_b = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

#Particle fraction of slow activities
xA = 50./100.0

#Particle fraction of fast activities
xF = 50./100.0

#Particle fraction of slow activities
xS=1.0-xF

partNum = 50000          # total number of particles
partNumS=xS*partNum      # total number of slow particles
partNumF=xF*partNum      # total number of fast particles

intPhi = 60               # integer system area fraction
phi = float(intPhi)/100.0 # system area fraction
eps = 1.0                 # particle softness

#initialize empty arrays for appending to
press_fast_int = np.array([])
press_slow_int = np.array([])
press_fast_dense = np.array([])
press_slow_dense = np.array([])
pa_pair = np.array([])
pb_pair = np.array([])
pnet_pair = np.array([])
int_width_slow = np.array([])
int_width_fast = np.array([])
int_width_theory = np.array([])
int_width_predict_fastd = np.array([])
int_width_predict_slowd = np.array([])
int_width_fast_int_arr = np.array([])
int_width_slow_int_arr = np.array([])
#Loop over slow activities
for i in range(0, len(pe_a)): 
    
    #Loop over fast activities
    for j in range(0, len(pe_b)):
        
        #Be sure slow activity is less than or equal to fast activity
        if pe_a[i]<=pe_b[j]:
            
            #Compute net activity
            peNet = compPeNet(xF, pe_a[i], pe_b[j])
            
            # Compute lattice spacing based on each activity
            latS = getLat(pe_a[i], eps)
            latF = getLat(pe_b[j], eps)
            latNet = getLat(peNet, eps)
            
            Ns = partNum * (1. - xF)          # number of slow particles
            Nf = partNum - Ns                 # number of fast particles
            
            phiG = compPhiG(peNet, latNet)              # area fraction of gas phase
            NGas = (phiG / phi) * partNum    # number of gas particles
            
            #Compute area fraction of dense phase based on each lattice spacing
            phi_theory = latToPhi(latNet)
            phiS_theory = latToPhi(latS)
            phiF_theory = latToPhi(latF)
            
            #Compute number density of slow and fast particles
            nS_theory = phi_theory /(np.pi/4)
            nF_theory = phi_theory /(np.pi/4)
            
            #Compute number of particles in liquid phase
            Nl = int(round(partNum * ((phi_theory * (phiG - phi)) / (phi * (phiG - phi_theory)))))
            
            #Compute number of particles in gas phase
            NGas = partNum - Nl
            
            # Critical packing fraction in HCP lattice
            phiCP = np.pi / (2. * np.sqrt(3))
            
            # The area is the sum of the particle areas (normalized by close packing density of spheres)
            Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
            As = (Ns * np.pi * (latNet)**2) / (4*phiCP)
            if As > Al:
                As = Al
            Af = (Nf * np.pi * (latNet)**2) / (4*phiCP)
            if Af > Al:
                Af = Al
            
            # The area for instantiated liquid cluster
            Al_real=Al
            
            # The cluster radius is the square root of liquid area divided by pi
            Rl = np.sqrt(Al_real / np.pi)
            Rs = np.sqrt(As / np.pi)
            Rf = np.sqrt(Af / np.pi)
            
            #The interface width is the difference between the 100% bulk species radius and the total cluster radius
            int_width_fast_int = Rl-Rs
            int_width_slow_int = Rl-Rf
            int_width_fast_int_arr = np.append(int_width_fast_int_arr, int_width_fast_int)
            int_width_slow_int_arr = np.append(int_width_slow_int_arr, int_width_slow_int)
            #Calculate interface aligned active pressure from theory
            press_fast_int = np.append(press_fast_int, int_width_fast_int * nF_theory * 1.0 * pe_b[j])
            press_slow_int = np.append(press_slow_int, int_width_slow_int * nS_theory * 1.0 * pe_a[i])
            
            #Calculate bulk interparticle pressure from theory
            press_slow_dense_val = 4.0 * np.sqrt(3) * pe_a[i] / latNet
            press_fast_dense_val = 4.0 * np.sqrt(3) * pe_b[j] / latNet
            
            #Append pressures to array
            press_slow_dense = np.append(press_slow_dense, press_slow_dense_val)
            press_fast_dense = np.append(press_fast_dense, press_fast_dense_val)
            int_width_predict_slowd = np.append(int_width_predict_slowd, press_slow_dense_val/(pe_b[j]*nF_theory))
            int_width_predict_fastd = np.append(int_width_predict_fastd, press_fast_dense_val/(pe_a[i]*nS_theory))
            
            #Append activities to array
            pa_pair = np.append(pa_pair, pe_a[i])
            pb_pair = np.append(pb_pair, pe_b[j])
            pnet_pair = np.append(pnet_pair, pe_b[j]*0.5 + pe_a[i]*0.5)
            
            #Append interface widths to array
            int_width_fast = np.append(int_width_fast, int_width_fast_int)
            int_width_slow = np.append(int_width_slow, int_width_slow_int)
            
            #Calculate dense phase pressure again and constants for predicting what interface width is for balancing pressure
            curPLJ = ljPress(latNet, peNet, eps)

            alpha_max = 0.5
            I_arr = 3.0
            int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/peNet) * (latNet **2) * I_arr
            int_width_theory = np.append(int_width_theory, int_width)

plt.scatter(pnet_pair, press_slow_dense, c='red')
plt.scatter(pnet_pair, press_fast_dense, c='blue')
plt.show()
#Plot difference in fast bulk phase pressure and slow interface pressure as a function of fast and slow activities
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = -20000
max_n = 20000
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pa_pair, pb_pair, c=press_fast_dense-press_slow_int, linewidths=1.0, edgecolor='black', s=60.0, vmin=min_n, vmax=max_n, cmap='seismic')
norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
sm.set_array([])
tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

clb.ax.tick_params(labelsize=16)
clb.set_label(r'$\Pi_\mathrm{dense}^\mathrm{F}-\Pi_\mathrm{int}^\mathrm{S}$', labelpad=20, y=0.5, rotation=-90, fontsize=18)
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Slow}$', fontsize=20)
plt.ylabel(r'$\mathrm{Pe}_\mathrm{Fast}$', fontsize=20)              
plt.tight_layout()

plt.show()

#Plot difference in slow bulk pressure and fast interface pressure as a function of fast and slow activity
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = -20000
max_n = 20000
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pa_pair, pb_pair, c=press_slow_dense-press_fast_int, linewidths=1.0, edgecolor='black', s=60.0, vmin=min_n, vmax=max_n, cmap='seismic')
norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
sm.set_array([])
tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

clb.ax.tick_params(labelsize=16)
clb.set_label(r'$\Pi_\mathrm{dense}^\mathrm{S}-\Pi_\mathrm{int}^\mathrm{F}$', labelpad=20, y=0.5, rotation=-90, fontsize=18)
               
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Slow}$', fontsize=20)
plt.ylabel(r'$\mathrm{Pe}_\mathrm{Fast}$', fontsize=20)
plt.tight_layout()
plt.show()


#Plot interface width with slow bulk as a function of fast and slow activities
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = 0
max_n = 8
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pa_pair, pb_pair, c=int_width_predict_slowd, s=60.0, linewidths = 1.0, edgecolors='black', vmin=min_n, vmax=max_n, cmap='cool')
norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
sm.set_array([])
tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

clb.ax.tick_params(labelsize=16)
clb.set_label(r'$h^\mathrm{S}$', labelpad=20, y=0.5, rotation=-90, fontsize=18)
               
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Slow}$', fontsize=20)
plt.ylabel(r'$\mathrm{Pe}_\mathrm{Fast}$', fontsize=20)
plt.tight_layout()
plt.show()

#Plot interface width with fast bulk as a function of fast and slow activities
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = 0
max_n = 20
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pa_pair, pb_pair, c=int_width_predict_fastd, linewidths=1.0, edgecolor='black', s=60.0, vmin=min_n, vmax=max_n, cmap='cool')
norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
sm.set_array([])
tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

clb.ax.tick_params(labelsize=16)
clb.set_label(r'$h^\mathrm{F}$', labelpad=20, y=0.5, rotation=-90, fontsize=18)
               
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Slow}$', fontsize=20)
plt.ylabel(r'$\mathrm{Pe}_\mathrm{Fast}$', fontsize=20)
plt.tight_layout()
plt.show()


#Plot interface width with fast bulk as a function of fast and slow activities
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = 0
max_n = 20
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pnet_pair, int_width_predict_fastd, linewidths=1.0, edgecolor='black', facecolor='red', s=60.0)
im = plt.scatter(pnet_pair, int_width_slow_int_arr, linewidths=1.0, edgecolor='black', facecolor='blue', s=60.0)
         
yellow_patch = mpatches.Patch(color='blue', label='Actual')
red_patch = mpatches.Patch(color='red', label='Pressure Balance')
plt.legend(handles=[yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=12, loc='upper left',labelspacing=0.1, handletextpad=0.1)

    
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Net}$', fontsize=20)
plt.ylabel(r'$h^\mathrm{S}$', fontsize=20)
plt.tight_layout()
plt.show()

#Plot interface width with fast bulk as a function of fast and slow activities
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
div_min = -3
min_n = 0
max_n = 20
levels_text=40
level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
tick_locs   = [0.0,np.pi/6,np.pi/3]
tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

im = plt.scatter(pnet_pair, int_width_predict_slowd, linewidths=1.0, edgecolor='black', facecolor='red', s=60.0)
im = plt.scatter(pnet_pair, int_width_fast_int_arr, linewidths=1.0, edgecolor='black', facecolor='blue', s=60.0)
         
yellow_patch = mpatches.Patch(color='blue', label='Actual')
red_patch = mpatches.Patch(color='red', label='Pressure Balance')
plt.legend(handles=[yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=12, loc='upper left',labelspacing=0.1, handletextpad=0.1)

    
plt.xlabel(r'$\mathrm{Pe}_\mathrm{Net}$', fontsize=20)
plt.ylabel(r'$h^\mathrm{F}$', fontsize=20)
plt.tight_layout()
plt.show()