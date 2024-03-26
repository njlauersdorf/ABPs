#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'

# Run on cluster
hoomdPath = "${hoomd_path}" # path to where you installed hoomd-blue '/.../hoomd-blue/build/'

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
runFor = ${runfor}              # simulation length (in tauLJ)
dumpPerBrownian = ${dump_freq}  # how often to dump data
pa=${pe_a}
pb=${pe_b}
if pa<=pb:
    peS=pa
    peF=pb
else:   
    peS = pb                  # activity of A particles
    peF = pa      
xA = ${part_frac_a}/100.0
xF = ${part_frac_a}/100.0
xS=1.0-xF

partNum = ${part_num}           # total number of particles
partNumS=xS*partNum
partNumF=xF*partNum
intPhi = ${phi}                 # system area fraction
phi = float(intPhi)/100.0
eps = ${ep}  

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
seed1 = ${seed1}                # integrator seed
seed2 = ${seed2}                # orientation seed
seed3 = ${seed3}                # activity seed

peNet = compPeNet(xF, peS, peF)

# Compute lattice spacing based on each activity
latS = getLat(peS, eps)
latF = getLat(peF, eps)
latNet = getLat(peNet, eps)
latF=latNet
latS=latNet

# Compute gas phase density, phiG
phiG = compPhiG(peNet, latNet)
phi_theory = latToPhi(latNet)

Nl = int(round(partNum * ((phi_theory * (phiG - phi)) / (phi * (phiG - phi_theory)))))


#Nls = int(Nl * (1. - xF))
#Nlf = Nl - Nls


  
# Now you need to convert this to a cluster radius
phiCP = np.pi / (2. * np.sqrt(3))

# The area is the sum of the particle areas (normalized by close packing density of spheres)
Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)

curPLJ = ljPress(latNet, peNet, eps)

# The area for seed
Al_real=Al

# The cluster radius is the square root of liquid area divided by pi
Rl = np.sqrt(Al_real / np.pi)

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

# List of activities
peList = [ peS ]
# List of ring radii
rList = [ 0., Rl ]
# Depth of alignment
#rAlign = 3.

rAlign = int_width#*(2/3)#3.#int_width
# List to store particle positions and types
pos = []
typ = []
rOrient = []
# z-value for simulation initialization
z = 0.5
for i in range(0, len(peList)):
    rMin = rList[0]             # starting distance for particle placement
    rMax = rList[1]         # maximum distance for particle placement
    ver = np.sqrt(0.75) * latNet   # vertical shift between lattice rows
    hor = latNet / 2.0             # horizontal shift between lattice rows
    
    x = 0.
    y = 0.
    shift = 0.
    while y <= rMax:
        r = computeDistance(x, y)
        # Check if x-position is too large
        if r > (rMax + (latNet/2.)):
            y += ver
            shift += 1
            if shift % 2:
                x = hor
            else:
                x = 0.
            continue
        
        # Whether or not particle is oriented
        if r > (rMax - rAlign):
            # Aligned
            rOrient.append(1)
        else:
            # Random
            rOrient.append(0)
            
        # If the loop makes it this far, append
        pos.append((x, y, z))
        typ.append(i)
        if x != 0. and y != 0.:
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
        elif x != 0.:
            pos.append((-x, y, z))
            rOrient.append(rOrient[-1])
            typ.append(i)
        # x must be zero
        elif y!= 0.:
            pos.append((x, -y, z))
            rOrient.append(rOrient[-1])
            typ.append(i)
            
        # Increment counter
        x += latNet
        
# Update number of particles in gas and dense phase    

NLiq = len(pos)
NGas = partNum - NLiq
typ_S=0
typ_F=0

for i in range(0,len(typ)):
    if pos[i][0]<=0:
        typ[i]=0
        typ_F+=1
    else:
        typ[i]=1
        typ_S+=1
gas_F=partNumF-typ_F
gas_S=partNumS-typ_S

# Set this according to phiTotal
areaParts = partNum * np.pi * (0.25)
abox = (areaParts / phi)
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
    if r <= (rList[-1] + (latNet/2.) + (tooClose / 2.)):
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
        typ.append(0)           # final particle type, same as outer ring
        count += 1              # increment count


## Get each coordinate in a list
#print("N_liq: {}").format(len(pos))
#print("Intended N_liq: {}").format(NLiq)
#print("N_gas: {}").format(len(gaspos))
#print("Intended N_gas: {}").format(NGas)
#print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
#print("Intended N: {}").format(partNum)
pos = pos + gaspos

NGas_shift=NGas
for i in range(0,NGas):
    j=NLiq+i
    rand_val=random.random()
    xF_gas=gas_F/NGas_shift
    if rand_val<=xF_gas:
        typ[j]=0
        typ_F+=1
        gas_F-=1
        NGas_shift-=1
    else:
        typ[j]=1
        typ_S+=1
        NGas_shift-=1  
typ_arr=np.array(typ)
id0=np.where(typ_arr==0)
id1=np.where(typ_arr==1)

x, y, z = zip(*pos)
## Plot as scatter
#cs = np.divide(typ, float(len(peList)))
#cs = rOrient
#plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
#ax = plt.gca()
#ax.set_aspect('equal')
partNum = len(pos)
# Get the number of types
uniqueTyp = []
for i in typ:
    if i not in uniqueTyp:
        uniqueTyp.append(i)
# Get the number of each type
particles = [ 0 for x in range(0, len(uniqueTyp)) ]
for i in range(0, len(uniqueTyp)):
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
peList = [pa, pb]
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
for i in range(0, len(unique_char_types)):
    for j in range(i, len(unique_char_types)):
        lj.pair_coeff.set(unique_char_types[i],
                          unique_char_types[j],
                          epsilon=eps, sigma=sigma)

# Brownian integration
brownEquil = 10000

hoomd.md.integrate.mode_standard(dt=dt)
bd = hoomd.md.integrate.brownian(group=all, kT=kT, seed=seed1)
hoomd.run(brownEquil)

# Set activity of each group
np.random.seed(seed2)                           # seed for random orientations
angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
activity = []
for i in range(0, partNum):
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
#out = "cluster_pe"
#for i in peList:
#    out += str(int(i))
#    out += "_"
#out += "r"
#for i in range(1, len(rList)):
#    out += str(int(rList[i]))
#    out += "_"
#out += "rAlign_" + str(rAlign) + ".gsd"
out = "half_pa" + str(int(pa))
out += "_pb" + str(int(pb))
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
               overwrite=True,
               phase=-1,
               dynamic=['attribute', 'property', 'momentum'])

# Run

hoomd.run(totTsteps)
