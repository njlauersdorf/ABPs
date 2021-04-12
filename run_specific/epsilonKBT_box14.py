'''
#                           This is an 80 character line                       #
    This is intended to investigate how hard our particles need to be. We want
    to maintain a ratio of active force to LJ well-depth:
                    epsilon = alpha * F_A * sigma / 24.0
    This code will investigate alpha, in order to find the smallest value that
    will maintain a "hard-sphere" potential. (This will optimize computation
    time while keeping our model realistic)
'''
# Initial imports
import sys
import os

# Read in bash arguments
hoomdPath = "${hoomd_path}" # path to where you installed hoomd-blue '/.../hoomd-blue/build/'
gsdPath = "${gsd_path}"         # path to where you want to save your gsd output file
runFor = ${runfor}              # simulation length (in tauLJ)
dumpFreq = ${dump_freq}         # how often to dump data
partPercA = ${part_frac_a}      # percentage of A particles
partFracA = float(partPercA) / 100.0  # fraction of particles that are A
peA = ${pe_a}                   # activity of A particles
peB = ${pe_b}                   # activity of B particles
partNum = ${part_num}           # Number of particles in system
intPhi = ${phi}                 # system area fraction (integer, i.e. 45, 65, etc.)
phi = float(intPhi) / 100.0     # system area fraction (decimal, i.e. 0.45, 0.65, etc.)
eps = ${ep}                     # epsilon (potential well depth for LJ potential)

#aspect ratio
base = 4
height = 1

seed1 = ${seed1}                # seed for position
seed2 = ${seed2}                # seed for bd equilibration
seed3 = ${seed3}                # seed for initial orientations
seed4 = ${seed4}                # seed for A activity
seed5 = ${seed5}                # seed for B activity

# Remaining imports
sys.path.insert(0,hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

import hoomd                    # import hoomd functions based on path
from hoomd import md
from hoomd import deprecated
import numpy as np


# Set some constants
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

# Compute parameters from activities
if peA != 0:                        # A particles are NOT Brownian
    vA = computeVel(peA)
    FpA = computeActiveForce(vA)
    #epsA = computeEps(alpha, FpA)
    epsA = eps
    tauA = computeTauLJ(epsA)
else:                               # A particles are Brownian
    vA = 0.0
    FpA = 0.0
    epsA = eps#kT
    tauA = computeTauLJ(epsA)

if peB != 0:                        # B particles are NOT Brownian
    vB = computeVel(peB)
    FpB = computeActiveForce(vB)
    epsB=eps
    #epsB = computeEps(alpha, FpB)
    #epsB = kT
    tauB = computeTauLJ(epsB)
else:                               # B particles are Brownian
    vB = 0.0                        
    FpB = 0.0
    epsB = eps#kT
    tauB = computeTauLJ(epsB)

#epsAB = (epsA + epsB + 1) / 2.0             # AB interaction well depth
epsAB=epsA                                  # assign AB interaction well depth to same as A and B
tauLJ = (tauA if (tauA <= tauB) else tauB)  # use the smaller tauLJ.  Doesn't matter since these are the same
epsA = (epsA if (epsA >= epsB) else epsB)   # use the larger epsilon. Doesn't matter since these are the same

dt = 0.000001 * tauLJ                        # timestep size.  I use 0.000001 for dt=tauLJ* (eps/10^6) generally
simLength = runFor * tauBrown               # how long to run (in tauBrown)
simTauLJ = simLength / tauLJ                # how long to run (in tauLJ)
totTsteps = int(simLength / dt)             # how many tsteps to run
numDumps = float(simLength / 0.1)           # dump data every 0.1 tauBrown.  
dumpFreq = float(totTsteps / numDumps)      # normalized dump frequency.  
dumpFreq = int(dumpFreq)                    # ensure this is an integer

print("Brownian tau in use:"+str(tauBrown))
print("Lennard-Jones tau in use:"+str(tauLJ))
print("Timestep in use:"+str(dt))
print("A-epsilon in use:"+str(epsA))
print("B-epsilon in use:"+str(epsB))
print("AB-epsilon in use:"+str(epsAB))
print("Total number of timesteps:"+str(totTsteps))
print("Total number of output frames:"+str(numDumps))
print("File dump frequency:"+str(dumpFreq))



area_ratio = base*height

box_area = partNum/phi
box_length = (box_area/area_ratio)**0.5
lx = base*box_length
ly = height * box_length
set_box = hoomd.data.boxdim(Lx=lx, Ly=ly, Lz=0, dimensions=2)

# Initialize system
hoomd.context.initialize()

#Randomly distrubte N particles with specified density phi_p (dictates simulation box size) 
#of particle type name with minimum separation distance min_dist with random seed in 2 dimensions
#system = hoomd.deprecated.init.create_random(N = partNum,
#                                             phi_p = phi,
#                                             name = 'A',
#                                             min_dist = 0.70,
#                                             seed = seed1,
#                                             dimensions = 2)

system = hoomd.deprecated.init.create_random(N = partNum,
                                             name = 'A',
                                             min_dist = 0.70,
                                             seed = seed1,
                                             box = set_box)

# Add B-type particles
system.particles.types.add('B')

#Save current time step of system
snapshot = system.take_snapshot()

partA = partNum * partFracA         # get the total number of A particles
partA = int(partA)                  # make sure it is an integer
partB = partNum - partA             # get the total number of B particles
partB = int(partB)                  # make sure it is an integer
mid = int(partA)                    # starting index to assign B particles

# Assign particles B type within the snapshot
if partPercA == 0:                      # take care of all b case
    mid = 0
    for i in range(mid, partNum):
        system.particles[i].type = 'B'
elif partPercA != 100:                  # mix of each or all A
    for i in range(mid, partNum):
        system.particles[i].type = 'B'

# Assigning groups and lengths to particles
all = hoomd.group.all()
gA = hoomd.group.type(type = 'A', update=True)
gB = hoomd.group.type(type = 'B', update=True)

# Number of particles of each group
N = len(all)
Na = len(gA)
Nb = len(gB)

# Define potential between pairs
nl = hoomd.md.nlist.cell()

#Can change potential between particles here with hoomd.md.pair...
lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)

#Set parameters of pair force dependent on type of interaction
lj.set_params(mode='shift')
lj.pair_coeff.set('A', 'A', epsilon=epsA, sigma=1.0)
lj.pair_coeff.set('A', 'B', epsilon=epsAB, sigma=1.0)
lj.pair_coeff.set('B', 'B', epsilon=epsB, sigma=1.0)

# General integration parameters

#Equilibration number of time steps
brownEquil = 100000

# Each time step corresponds to time step size of dt
hoomd.md.integrate.mode_standard(dt=dt)

# Overdamped Langevin equations without activity at temperature kT.  Seed2 specifies translational diffusion.
hoomd.md.integrate.brownian(group=all, kT=kT, seed=seed2)

#Run hoomd over brownEquil time steps
hoomd.run(brownEquil)

#set the activity of each type
np.random.seed(seed3)                           # seed for random orientations
angle = np.random.rand(partNum) * 2 * np.pi    # random particle orientation

# Case 1: Mixture
if partPercA != 0 and partPercA != 100:
    # First assign A-type active force vectors (w/ peA)
    activity_a = []
    for i in range(0,mid):
        x = (np.cos(angle[i])) * peA    # x active force vector
        y = (np.sin(angle[i])) * peA    # y active force vector
        z = 0                           # z active force vector
        tuple = (x, y, z)               # made into a tuple
        activity_a.append(tuple)        # add to activity A list

    # Now assign B-type active force vectors (w/ peB)
    activity_b = []
    for i in range(mid,partNum):
        x = (np.cos(angle[i])) * peB
        y = (np.sin(angle[i])) * peB
        z = 0
        tuple = (x, y, z)
        activity_b.append(tuple)
    # Set A-type activity in hoomd
    hoomd.md.force.active(group=gA,
                          seed=seed4,
                          f_lst=activity_a,
                          rotation_diff=D_r,
                          orientation_link=False,
                          orientation_reverse_link=True)
    # Set B-type activity in hoomd
    hoomd.md.force.active(group=gB,
                          seed=seed5,
                          f_lst=activity_b,
                          rotation_diff=D_r,
                          orientation_link=False,
                          orientation_reverse_link=True)
else:
    # Case 2: All B system
    if partPercA == 0:
        activity_b = []
        for i in range(0,partNum):
            x = (np.cos(angle[i])) * peB
            y = (np.sin(angle[i])) * peB
            z = 0
            tuple = (x, y, z)
            activity_b.append(tuple)
        hoomd.md.force.active(group=gB,
                              seed=seed5,
                              f_lst=activity_b,
                              rotation_diff=D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)
    # Case 3: All A system
    else:
        activity_a = []
        for i in range(0,partNum):
            x = (np.cos(angle[i])) * peA
            y = (np.sin(angle[i])) * peA
            z = 0
            tuple = (x, y, z)
            activity_a.append(tuple)
        hoomd.md.force.active(group=gA,
                              seed=seed4,
                              f_lst=activity_a,
                              rotation_diff=D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)

# base file name for output (specify variables that will be changed or that you care about)
name = "pa" + str(peA) +\
"_pb" + str(peB) +\
"_xa" + str(partPercA) +\
"_ep" + str(epsAB)+\
"_phi"+str(intPhi)+\
"_pNum" + str(partNum)

# Actual gsd file name for output
gsdName = name + ".gsd"

# Remove .gsd files if they exist

try:
    os.remove(gsdName)
except OSError:
    pass

#Specify how often and what to save
    
# Options for what to save (dynamic)
# 'attribute' important quantities: total particle number (N), types (types), which particles are each type (typeid),
#  diameter of particle (diameter)
#'property' important quantities: particle position (position), particle orientation in quaternions (orientation)
#'momentum': I save this but it's unimportant.  It saves velocity, angular momentum, and image.  I looked into this and the v
# velocity calculation is incorrect in my current version of hoomd.  THe newer version seems to have fixed this I believe. They mis-calculated
# the quaternions to angles.
#'topology' is another option for working with molecules.
    
hoomd.dump.gsd(gsdName,
               period=dumpFreq,
               group=all,
               overwrite=False,
               phase=-1,
               dynamic=['attribute', 'property', 'momentum'])

#Number of time steps to run simulation for.
hoomd.run(totTsteps)
