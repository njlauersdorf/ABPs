#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'

# Run on cluster
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build/'

sys.path.insert(0,hoomdPath)

import hoomd
from hoomd import md
from hoomd import deprecated

import math
import numpy as np
import random
import matplotlib.pyplot as plt

kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

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
    return num / den
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
### And here's an example of how to use these ###
# Read in bash arguments
runFor = 100              # simulation length (in tauLJ)
dumpPerBrownian = 20000  # how often to dump data
pa=51
pb=500
if pa<=pb:
    peS=pa
    peF=pb
else:   
    peS = pb                  # activity of A particles
    peF = pa      
xA = 50/100.0
xF = 50/100.0
xS=1.0-xF

partNum = 50000           # total number of particles
partNumS=xS*partNum
partNumF=xF*partNum
intPhi = 60                 # system area fraction
phi = float(intPhi)/100.0
eps = 1.0  



# Compute parameters from activities
tauLJ = computeTauLJ(eps)

#epsA = (epsA if (epsA >= epsB) else epsB)   # use the larger epsilon
#epsB = epsA                                 # make sure all use this
#epsAB = epsA                                # make sure all use this
dt = 0.000001 * tauLJ                        # timestep size
simLength = runFor * tauBrown               # how long to run (in tauBrown)
simTauLJ = simLength / tauLJ                # how long to run (in tauLJ)
totTsteps = int(simLength / dt)             # how many tsteps to run
numDumps = float(simLength / 0.1)           # dump data every 0.5 tauBrown
dumpFreq = float(totTsteps / numDumps)      # normalized dump frequency
dumpFreq = int(dumpFreq)                    # ensure this is an integer


num_F=xF*partNum
num_S=xS*partNum
# Random seeds
seed1 = 60024                # integrator seed
seed2 = 60024                # orientation seed
seed3 = 60024                # activity seed

peNet = compPeNet(xF, peS, peF)

# Compute lattice spacing based on each activity
latS = getLat(peS, eps)
latF = getLat(peF, eps)
latNet = getLat(peNet, eps)

#N = 10000.                 # number of particles
Ns = partNum * (1. - xF)          # number of slow particles
Nf = partNum - Ns                 # number of fast particles
phiG = compPhiG(peNet, latNet)              # area fraction of gas phase
#phiG = phig/100.  # number of gas particles

# Compute gas phase density, phiG
#phiG = compPhiG(peNet, latNet)
NGas = (phiG / phi) * partNum    # number of gas particles

phi_theory = latToPhi(latNet)
phiS_theory = latToPhi(latS)
phiF_theory = latToPhi(latF)

Nl = int(round(partNum * ((phi_theory * (phiG - phi)) / (phi * (phiG - phi_theory)))))
NGas = partNum - Nl

#Nls = int(Nl * (1. - xF))
#Nlf = Nl - Nls

def areaType(Nx, latx):
    Ax = Nx * np.pi * 0.25 * (latx**2)
    return Ax
   
areas = areaType(Ns - NGas, latNet)
areaf = areaType(Nf, latNet)
areatot = areaf + areas
rtot = np.sqrt(areatot / np.pi)
# This depends on choice of interior species
rIn = np.sqrt(areaf / np.pi)

# Now you need to convert this to a cluster radius
phiCP = np.pi / (2. * np.sqrt(3))

# The area is the sum of the particle areas (normalized by close packing density of spheres)
Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
As = (Ns * np.pi * (latNet)**2) / (4*phiCP)
Af = (Nf * np.pi * (latNet)**2) / (4*phiCP)
curPLJ = ljPress(latNet, peNet, eps)

# The area for seed
Al_real=Al

# The cluster radius is the square root of liquid area divided by pi
Rl = np.sqrt(Al_real / np.pi)
Rs = np.sqrt(As / np.pi)
Rf = np.sqrt(Af / np.pi)

alpha_max = 0.5
I_arr = 3.0
int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/peNet) * (latNet **2) * I_arr

if int_width >= Rl:
    int_width = Rl-1.0
# Remember!!! This is a prediction of the EQUILIBRIUM size, reduce this to seed a cluster
# MAKE SURE that the composition of the seed has the same composition of the system
# e.g. for xF = 0.3 the initial seed should be 30% fast 70% slow


#print(int_width)
#stop

# Use latNet to space your particles
def computeDistance(x, y):
    return np.sqrt((x**2) + (y**2))
 
def interDist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  
def orientToOrigin(x, y, act):
    "Using similar triangles to find sides"
    x *= -1.
    y *= -1.
    hypRatio = act / np.sqrt(x**2 + y**2)
    xAct = hypRatio * x
    yAct = hypRatio * y
    return xAct, yAct

def densProbability(r, activity_net, activity_slow):
    "Using similar triangles to find sides"
    gas_dense_dif_phi = 0.866 * np.log10(activity_net - 46.993) - 0.443
    rate_decay = -6.151 * np.log10(activity_net-49.921) - 4.392
    mid_point = 0.044 * np.log10(activity_slow-49.893) + 0.836
    gas_phi = -0.26 * np.log10(activity_slow-41.742)+0.783
    
    num_dens_r = ((gas_dense_dif_phi / (1+np.exp(-rate_decay * (r-mid_point)))) + gas_phi)
    
    return num_dens_r

def alignProbability(r, activity_net, activity_slow):
    "Using similar triangles to find sides"
    max_align = 0.2019 * np.log10(activity_net - 41.2803) - 0.1885
    mid_point = 0.0492 * np.log10(activity_slow - 47.0061) + 0.8220
    std_dev = 0.1057
    align_r = max_align * np.exp(-(r-mid_point)**2/(2*std_dev**2))
    
    return align_r

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

def activityProbability(r, r_swap = [], probA = []):
    "Using similar triangles to find sides"
    if len(r_swap)>0:
        prob_rA = np.zeros(len(r))
        prob_rB = np.zeros(len(r))
        for i in range(1, len(r_swap)):
            r_min = find_nearest(r, r_swap[i-1])
            r_max = find_nearest(r, r_swap[i])
            
            prob_rA[r_min:r_max+1]=probA[i]
            prob_rB[r_min:r_max+1]=1.0-probA[i]
    else:
        prob_rA = np.ones(len(r)) * 0.5
        prob_rB = np.ones(len(r)) - prob_rA
    
    return prob_rA, prob_rB
# List of activities
peList = [ peF, peS ]
# List of ring radii
#rList = [ 0, rIn, rtot ]
rList = [ 0, Rf, Rl ]
# Depth of alignment
#rAlign = 3.

def ljForce(r, eps, sigma=1.):
    div = (sigma/r)
    dU = (24. * eps / sigma) * ((2*(div**13)) - (div)**7)
    return dU

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
    coeff = 1.874#3.0#1.92#2.03#3.5#2.03
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


r = np.linspace(0, 2, num=400)
dens = densProbability(r, peNet, peS)
align = alignProbability(r, peNet, peS)
activity_distrib = np.zeros(len(r))


prob_arr_A, prob_arr_B = activityProbability(r, r_swap = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], probA = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])#probA = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
plt.plot(r, prob_arr_A, linewidth=1.8*1.8, color='black')
plt.show()
prod=align * dens
prod = np.abs(np.gradient(align * dens)) * align

min_ind = np.min(np.where(prod>0.05*np.max(prod))[0])
max_ind = np.max(np.where(prod>0.05*np.max(prod))[0])
tot_int_pe = np.zeros(len(dens[min_ind:max_ind]))
area_loc = np.zeros(len(dens[min_ind:max_ind]))
tot_parts = np.zeros(len(dens[min_ind:max_ind]))
for i in range(0, len(dens[min_ind:max_ind])): 
    area_loc[i] = (np.pi * ((r[min_ind:max_ind+1][i+1]*rList[-1])**2 - (r[min_ind:max_ind][i]*rList[-1])**2))
    tot_parts[i] =  area_loc[i] * dens[min_ind:max_ind][i]# * (prob_arr_A[np.min(np.where(prod>0.02)[0]):np.max(np.where(prod>0.02)[0])][i] * pa + prob_arr_B[np.min(np.where(prod>0.02)[0]):np.max(np.where(prod>0.02)[0])][i] * pb)
    tot_int_pe[i] = tot_parts[i] * (prob_arr_A[min_ind:max_ind][i] * pa + prob_arr_B[min_ind:max_ind][i] * pb)
peNet_int = np.sum(tot_int_pe) / np.sum(tot_parts)
print(peNet_int)
#print(net_int_pe)
x = np.array([r[min_ind], r[min_ind]+0.00000001])
y = np.array([0, 1.0])
x2 = np.array([r[max_ind], r[max_ind]+0.00000001])

fig, ax1 = plt.subplots(figsize=(12,5))
plt.plot(x, y, linestyle='--', linewidth=1.2, color='black')
plt.plot(x2, y, linestyle='--', linewidth=1.2, color='black')
#plt.plot(r, (align*dens)/np.max(align*dens), linestyle='-', linewidth=1.8*1.8, color='black')
plt.plot(r, prod/np.max(prod), linestyle='-', linewidth=1.8*1.8, color='red')
plt.xlabel(r'Distance from CoM ($r$)')
plt.ylabel(r'$\nabla [\alpha (r)n(r)] \times \alpha (r)$')
plt.show()

'''
#press = dens[min_ind:max_ind] * align[min_ind:max_ind] * (prob_arr_A[min_ind:max_ind] * pa + prob_arr_B[min_ind:max_ind] * pb)
press2 = dens[min_ind:max_ind] * align[min_ind:max_ind] * (peNet_int-50)
#press2 = dens * align * (peNet_int-50)

#press_int = 0
press_int2=0
latInt = conForRClust(peNet_int-50, eps)
for i in range(1, len(press2)):
    #press_int += ((press[i-1]+press[i])/2)*(r[min_ind:max_ind][i]-r[min_ind:max_ind][i-1]) * Rl
    press_int2 += ((press2[i-1]+press2[i])/2)*(r[min_ind:max_ind][i]-r[min_ind:max_ind][i-1]) * Rl
    #press_int2 += ((press2[i-1]+press2[i])/2)*(r[i]-r[i-1]) * Rl
press_interpart = 2 * 1.874 * np.sqrt(3) * (peNet_int-50) / latInt
print(press_interpart)
print(press_int2)
stop
'''
peNet_int = np.linspace(51, 500, num=100)
#peNet = np.linspace(51, 500, num=500)
peA_arr = np.linspace(51, 500, num=100)
peB_arr = np.linspace(51, 500, num=100)
press_interpart = np.zeros(200)
#press_int2 = np.zeros(200)
#for k in range(0,len(peNet)):
press_int2 = np.array([])
press_interpart = np.array([])
penet_arr = np.array([])
penet_arr2 = np.array([])

pdif_arr = np.array([])
lat_arr = np.array([])
for k in range(0,len(peB_arr)):
    for j in range(0,len(peA_arr)):
        if peA_arr[j]<=peB_arr[k]:
            peS=peA_arr[j]
            
            peNet = peA_arr[j] * 0.5 + peB_arr[k] * 0.5
            r = np.linspace(0, 2, num=400)
            dens = densProbability(r, peNet, peS)
            align = alignProbability(r, peNet, peS)        
            
            prob_arr_A, prob_arr_B = activityProbability(r)#, r_swap = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], probA = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])#probA = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])##probA = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            
            prod = np.abs(np.gradient(align * dens)) * align
    
            min_ind = np.min(np.where(prod>0.05*np.max(prod))[0])
            max_ind = np.max(np.where(prod>0.05*np.max(prod))[0])
            tot_int_pe = np.zeros(len(dens[min_ind:max_ind]))
            area_loc = np.zeros(len(dens[min_ind:max_ind]))
            tot_parts = np.zeros(len(dens[min_ind:max_ind]))
            for i in range(0, len(dens[min_ind:max_ind])): 
                area_loc[i] = (np.pi * ((r[min_ind:max_ind+1][i+1]*rList[-1])**2 - (r[min_ind:max_ind][i]*rList[-1])**2))
                tot_parts[i] =  area_loc[i] * dens[min_ind:max_ind][i]# * (prob_arr_A[np.min(np.where(prod>0.02)[0]):np.max(np.where(prod>0.02)[0])][i] * pa + prob_arr_B[np.min(np.where(prod>0.02)[0]):np.max(np.where(prod>0.02)[0])][i] * pb)
                tot_int_pe[i] = tot_parts[i] * (prob_arr_A[min_ind:max_ind][i] * peA_arr[j] + prob_arr_B[min_ind:max_ind][i] * peB_arr[k])
            peNet_int = np.sum(tot_int_pe) / np.sum(tot_parts)
            
            #press = dens[min_ind:max_ind] * align[min_ind:max_ind] * (prob_arr_A[min_ind:max_ind] * pa + prob_arr_B[min_ind:max_ind] * pb)
            #press2 = dens[min_ind:max_ind] * align[min_ind:max_ind] * (peNet_int-50)
            press2 = dens * align * (peNet_int-50)
            press_int2 = np.append(press_int2, 0)
            latInt = conForRClust(peNet_int-50, eps)
            for i in range(1, len(press2)):
                #press_int += ((press[i-1]+press[i])/2)*(r[min_ind:max_ind][i]-r[min_ind:max_ind][i-1]) * Rl
                #press_int2[-1] += ((press2[i-1]+press2[i])/2)*(r[min_ind:max_ind][i]-r[min_ind:max_ind][i-1]) * Rl
                press_int2[-1] += ((press2[i-1]+press2[i])/2)*(r[i]-r[i-1]) * Rl
            press_interpart = np.append(press_interpart, 2 * 1.874 * np.sqrt(3) * (peNet_int-50) / latInt)
            penet_arr = np.append(penet_arr, peNet)
            penet_arr2 = np.append(penet_arr2, peNet_int)
            pdif_arr = np.append(pdif_arr, np.abs(peA_arr[j]-peB_arr[k]))
            lat_arr = np.append(lat_arr, latInt)

fig, ax1 = plt.subplots(figsize=(6,5))
#plt.plot(peNet_int, press_int2, linestyle='-', linewidth=1.8*1.8, color=yellow, label='Interface')
plt.scatter(penet_arr, lat_arr, c=pdif_arr, s=1.0)
plt.colorbar()
plt.ylabel(r'Lattice spacing ($a$)')
plt.xlabel(r'Net Activity($\mathrm{Pe}_\mathrm{Net}$)')
#plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=red)
plt.show()
stop
yellow = ("#fdfd96")
green = ("#77dd77")
red = ("#ff6961")
purple = ("#cab2d6")
fig, ax1 = plt.subplots(figsize=(6,5))
#plt.plot(peNet_int, press_int2, linestyle='-', linewidth=1.8*1.8, color=yellow, label='Interface')
plt.scatter(penet_arr, press_interpart, c=pdif_arr, s=1.0)
plt.colorbar()
plt.ylabel(r'Bulk Pressure ($\Pi_\mathrm{d}$)')
plt.xlabel(r'Net Activity($\mathrm{Pe}_\mathrm{Net}$)')
#plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=red)
plt.show()

fig, ax1 = plt.subplots(figsize=(6,5))
#plt.plot(peNet_int, press_int2, linestyle='-', linewidth=1.8*1.8, color=yellow, label='Interface')
plt.scatter(penet_arr, penet_arr2, c=pdif_arr, s=1.0)
plt.colorbar()
plt.ylabel(r'Interface Net Activity$)')
plt.xlabel(r'Net Activity of Interface ($\mathrm{Pe}_\mathrm{Net}^\mathrm{i}$)')
#plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=red)
plt.show()

yellow = ("#fdfd96")
green = ("#77dd77")
red = ("#ff6961")
purple = ("#cab2d6")
fig, ax1 = plt.subplots(figsize=(6,5))
#plt.plot(peNet_int, press_int2, linestyle='-', linewidth=1.8*1.8, color=yellow, label='Interface')
plt.scatter(penet_arr, press_int2, c=pdif_arr, s=1.0)
plt.colorbar()
plt.ylabel(r'Interface Pressure ($\Pi_\mathrm{i}$)')
plt.xlabel(r'Net Activity ($\mathrm{Pe}_\mathrm{Net}$)')
#plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=red)
plt.show()

plt.scatter(penet_arr, press_int2-press_interpart, c=pdif_arr, s=1.0)
plt.colorbar()
plt.ylabel(r'Pressure Difference ($\Pi_\mathrm{int}-\Pi_\mathrm{bulk}$)')
plt.xlabel(r'Net Activity($\mathrm{Pe}_\mathrm{Net}$)')
plt.xlim([0, 550])
plt.ylim([-500, 500])
plt.show()
stop


print(press_interpart)
print(press_int2)
yellow = ("#fdfd96")
green = ("#77dd77")
red = ("#ff6961")
purple = ("#cab2d6")
fig, ax1 = plt.subplots(figsize=(6,5))
plt.plot(peNet_int, press_int2, linestyle='-', linewidth=1.8*1.8, color=yellow, label='Interface')
plt.plot(peNet_int, press_interpart, linestyle='-', linewidth=1.8*1.8, color=green, label='Bulk')
plt.ylabel(r'Pressure ($\Pi$)')
plt.xlabel(r'Net Activity of Interface ($\mathrm{Pe}_\mathrm{Net}^\mathrm{i}$)')
plt.legend(loc='upper left')
#plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=red)
plt.show()
x_arr = np.array([0, 500])
y_arr = np.array([0,0])
y_arr2 = np.array([-1000, 1000])
x_arr2 = np.array([pa,pa])
fig, ax1 = plt.subplots(figsize=(6,5))
plt.plot(peNet_int, press_int2 - press_interpart, linestyle='-', linewidth=1.8*1.8, color=purple)
plt.plot(x_arr, y_arr, linestyle='--', linewidth=1.2, color='black')
plt.plot(x_arr2, y_arr2, linestyle='dotted', linewidth=1.2, color='black')

plt.ylabel(r'Pressure Difference ($\Pi_\mathrm{int}-\Pi_\mathrm{bulk}$)')
plt.xlabel(r'Net Activity of Interface ($\mathrm{Pe}_\mathrm{Net}^\mathrm{i}$)')
plt.xlim([0, 500])
plt.ylim([-300, 300])
plt.show()
stop

rAlign = int_width#*(2/3)#3.#int_width
# List to store particle positions and types
pos = []
typ = []
rOrient = []
# z-value for simulation initialization
z = 0.5

for i in range(0,len(peList)):
    rMin = rList[i]             # starting distance for particle placement
    rMax = rList[i + 1]         # maximum distance for particle placement
    ver = np.sin(60*np.pi/180)*latNet#np.sqrt(0.75) * latNet   # vertical shift between lattice rows
    hor = latNet / 2.0             # horizontal shift between lattice rows
    x = 0
    y = 0
    shift = 0
    while y < rMax:
        r = computeDistance(x, y)
        # Check if x-position is large enough
        if r <rMin: # <(rMin + (latNet / 2.)):
            x += latNet
            continue
            
        # Check if x-position is too large
        if r >rMax:#>= (rMax - (latNet/2.)):
            y += ver
            shift += 1
            if shift % 2:
                x = hor
            else:
                x = 0
            continue
        
        # Whether or not particle is oriented
        if r > (rList[1]):
            # Aligned
            rOrient.append(1)
        else:
            # Random
            rOrient.append(0)
        
        # If the loop makes it this far, append
        pos.append((x, y, z))
        typ.append(i)
        if x != 0 and y != 0:
            # Mirror positions, alignment and type
            pos.append((-x, y, z))
            pos.append((-x, -y, z))
            pos.append((x, -y, z))
            rOrient.append(rOrient[-1])
            rOrient.append(rOrient[-1])
            rOrient.append(rOrient[-1])
            typ.append(i)
            typ.append(i)
            typ.append(i)
        # y must be zero
        elif x != 0:
            pos.append((-x, y, z))
            rOrient.append(rOrient[-1])
            typ.append(i)
        # x must be zero
        elif y!= 0:
            pos.append((x, -y, z))
            rOrient.append(rOrient[-1])
            typ.append(i)
         
        # Increment counter
        x += latNet

NLiq = len(pos)
NGas = partNum - NLiq
# Set this according to phiTotal
areaParts = partNum * np.pi * (0.25)
abox = (areaParts / 0.6)
lbox = np.sqrt(abox)
hbox = lbox / 2.
tooClose = 0.9

# Make a mesh for random particle placement
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
nBins = (getNBins(lbox, r_cut))
sizeBin = roundUp((lbox / nBins), 6)

# Place particles in gas phase
count = 0
gaspos = []
binParts = [[[] for b in range(nBins)] for a in range(nBins)]

while count < NGas:

    place = 1
    # Generate random position
    gasx = (np.random.rand() - 0.5) * lbox
    gasy = (np.random.rand() - 0.5) * lbox
    r = computeDistance(gasx, gasy)
    
    # Is this an HCP bin?
    if r <= (rList[-1] + (tooClose / 2.)):
        continue
    
    # Are any gas particles too close?
    tmpx = gasx + hbox
    tmpy = gasy + hbox
    indx = int(tmpx / sizeBin)
    indy = int(tmpy / sizeBin)
    # Get index of surrounding bins
    lbin = indx - 1  # index of left bins
    rbin = indx + 1  # index of right bins
    bbin = indy - 1  # index of bottom bins
    tbin = indy + 1  # index of top bins
    if rbin == nBins:
        rbin -= nBins  # adjust if wrapped
    if tbin == nBins:
        tbin -= nBins  # adjust if wrapped
    hlist = [lbin, indx, rbin]  # list of horizontal bin indices
    vlist = [bbin, indy, tbin]  # list of vertical bin indices
    # Loop through all bins
    for h in range(0, len(hlist)):
        for v in range(0, len(vlist)):
            # Take care of periodic wrapping for position
            wrapX = 0.0
            wrapY = 0.0
            if h == 0 and hlist[h] == -1:
                wrapX -= lbox
            if h == 2 and hlist[h] == 0:
                wrapX += lbox
            if v == 0 and vlist[v] == -1:
                wrapY -= lbox
            if v == 2 and vlist[v] == 0:
                wrapY += lbox
            # Compute distance between particles
            if binParts[hlist[h]][vlist[v]]:
                for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                    # Get index of nearby particle
                    ref = binParts[hlist[h]][vlist[v]][b]
                    r = interDist(gasx, gasy,
                                  gaspos[ref][0] + wrapX,
                                  gaspos[ref][1] + wrapY)
                    # Round to 4 decimal places
                    r = round(r, 4)
                    # If too close, generate new position
                    if r <= tooClose:
                        place = 0
                        break
            if place == 0:
                break
        if place == 0:
            break
            
    # Is it safe to append the particle?
    if place == 1:
        binParts[indx][indy].append(count)
        gaspos.append((gasx, gasy, z))
        rOrient.append(0)       # not oriented
        typ.append(1)           # final particle type, same as outer ring
        count += 1              # increment count
print('3')
# Get each coordinate in a list
pos = pos + gaspos
x, y, z = zip(*pos)

## Plot as scatter
#cs = np.divide(typ, float(len(peList)))
##cs = rOrient
#plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
#ax = plt.gca()
#ax.set_aspect('equal')
#plt.show()

partNum = len(pos)

# Get the number of types
uniqueTyp = []
for i in typ:
    if i not in uniqueTyp:
        uniqueTyp.append(i)
# Get the number of each type
particles = [ 0 for x in range(0,len(uniqueTyp)) ]
for i in range(0,len(uniqueTyp)):
    for j in typ:
        if uniqueTyp[i] == j:
            particles[i] += 1
# Convert types to letter values
unique_char_types = []
for i in uniqueTyp:
    unique_char_types.append( chr(ord('@') + i+1) )
char_types = []
for i in typ:
    char_types.append( chr(ord('@') + i+1) )

# Get a list of activities for all particles
pe = []
for i in typ:
    pe.append(peList[i])

# Now we make the system in hoomd
hoomd.context.initialize()
# A small shift to help with the periodic box
snap = hoomd.data.make_snapshot(N = partNum,
                                box = hoomd.data.boxdim(Lx=lbox,
                                                        Ly=lbox,
                                                        dimensions=2),
                                particle_types = unique_char_types)

# Set positions/types for all particles
snap.particles.position[:] = pos[:]
snap.particles.typeid[:] = typ[:]
snap.particles.types[:] = char_types[:]

# Initialize the system
system = hoomd.init.read_snapshot(snap)
all = hoomd.group.all()
groups = []
for i in unique_char_types:
    groups.append(hoomd.group.type(type=i))

# Set particle potentials
nl = hoomd.md.nlist.cell()
lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)
lj.set_params(mode='shift')
for i in range(0,len(unique_char_types)):
    for j in range(i, len(unique_char_types)):
        lj.pair_coeff.set(unique_char_types[i],
                          unique_char_types[j],
                          epsilon=eps, sigma=sigma)

# Brownian integration
print('run equil!')
brownEquil = 10000
hoomd.md.integrate.mode_standard(dt=dt)
bd = hoomd.md.integrate.brownian(group=all, kT=kT, seed=seed1)
hoomd.run(brownEquil)
print('done equil!')
# Set activity of each group
np.random.seed(seed2)                           # seed for random orientations
angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
activity = []
for i in range(0,partNum):
    if rOrient[i] == 0:
        x = (np.cos(angle[i])) * pe[i]
        y = (np.sin(angle[i])) * pe[i]
    else:
        x, y = orientToOrigin(pos[i][0], pos[i][1], pe[i])
    z = 0.
    tuple = (x, y, z)
    activity.append(tuple)
# Implement the activities in hoomd
hoomd.md.force.active(group=all,
                      seed=seed3,
                      f_lst=activity,
                      rotation_diff=D_r,
                      orientation_link=False,
                      orientation_reverse_link=True)

# Name the file from parameters
out = "slow_out_pa" + str(int(pb))
out += "_pb" + str(int(pa))
out += "_phi" + str(intPhi)
out += "_eps" + str(eps)
out += "_xa" + str(xA)
out += "_pNum" + str(partNum)
out += "_dtau" + "{:.1e}".format(dt)
out += ".gsd"

# Write dump
hoomd.dump.gsd(out,
               period=dumpFreq,
               group=all,
               overwrite=False,
               phase=-1,
               dynamic=['attribute', 'property', 'momentum'])
print('real run!')
# Run
hoomd.run(totTsteps)
