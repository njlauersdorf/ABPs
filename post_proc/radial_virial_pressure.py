'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Get center of mass
'''

import sys
from gsd import hoomd
import freud
import numpy as np
import math

#from descartes.patch import PolygonPatch
# Run locally
hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#'/nas/home/njlauers/hoomd-blue/build/'#Users/nicklauersdorf/hoomd-blue/build/'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)

r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
#inFile='pa300_pb300_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'cluster_pa400_pb350_phi60_eps0.1_xa0.8_align3_dtau1.0e-06.gsd'
#inFile='pa400_pb500_xa20_ep1.0_phi60_pNum100000.gsd'
outPath='/Volumes/External/mono_test/'#'/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/alignment_sparse/rerun2/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')

#Label simulation parameters
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac_orig = float(sys.argv[4])
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
eps = float(sys.argv[5])
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))
#Determine which activity is the slow activity or if system is monodisperse

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

# Set some constants
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
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

def avgCollisionForce(peNet):
    '''Computed from the integral of possible angles'''
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    return (magnitude * peNet) / (np.pi) 

def maximum(a, b, c, d): 
  
    if (a >= b) and (a >= c) and (a >= d): 
        largest = a 
    elif (b >= a) and (b >= c) and (b >= d): 
        largest = b 
    elif (c >= a) and (c >= b) and (c >= d):
        largest = c
    else: 
        largest = d
          
    return largest 

def minimum(a, b, c, d): 
  
    if (a <= b) and (a <= c) and (a <= d): 
        smallest = a 
    elif (b <= a) and (b <= c) and (b <= d): 
        smallest = b 
    elif (c <= a) and (c <= b) and (c <= d):
        smallest = c
    else: 
        smallest = d
          
    return smallest 
def quatToAngle(quat):
    "Take vector, output angle between [-pi, pi]"
    #print(quat)
    r = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    #print(2*math.atan2(x,r))
    rad = math.atan2(y, x)#(2*math.acos(r))#math.atan2(y, x)#
    
    return rad
def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ
def ljForce(r, eps, sigma=1.):
    '''Compute the Lennard-Jones force'''
    div = (sigma/r)
    dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
    return dU
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
#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)

tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size


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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib import collections  as mc
from matplotlib import lines
    
matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5
            
def computeDist(x1, y1, x2, y2):
    '''Compute distance between two points'''
    return np.sqrt( ((x2-x1)**2) + ((y2 - y1)**2) )
    
def computeFLJ(r, dx, dy, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (dx) / r
    fy = f * (dy) / r
    return fx, fy

def computeTauPerTstep(epsilon, mindt=0.000001):
    '''Read in epsilon, output tauBrownian per timestep'''
#    if epsilon != 1.:
#        mindt=0.00001
    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

def roundUp(n, decimals=0):
    '''Round up size of bins to account for floating point inaccuracy'''
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
    
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
def distComps(point1, point2x, point2y):
    '''Given points output x, y and r'''
    dx = point2x - point1[0]
    dy = point2y - point1[1]
    r = np.sqrt((dx**2) + (dy**2))
    return dx, dy, r
def findBins(lookN, currentInd, maxInds):
    '''Get the surrounding bin indices'''
    maxInds -= 1
    left = currentInd - lookN
    right = currentInd + lookN
    binsList = []
    for i in range(left, right):
        ind = i
        if i > maxInds:
            ind -= maxInds
        binsList.append(ind)
    return binsList
f = hoomd.open(name=inFile, mode='rb')


# Grab files
slowCol = '#d8b365'
fastCol = '#5ab4ac'


box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep


outTxt = 'CoM_' + outF + '.txt'
g = open(outTxt, 'w+') # write file headings
g.write('tau'.center(25) + ' ' +\
                    'rCoM'.center(25) + ' ' +\
                    'NinBin'.center(25) + ' ' +\
                    'phiLoc'.center(25) + ' ' +\
                    'align'.center(25) + ' ' +\
                    'pInt'.center(25) + ' ' +\
                    'pSwim'.center(25) + '\n')
g.close()
# Access file frames
with hoomd.open(name=inFile, mode='rb') as t:

    # Take first snap for box
    start = 600                  # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    # Get box dimensions
    l_box = box_data[0]
    h_box = l_box / 2.
    a_box = l_box * l_box
    
    radius=np.arange(0,h_box+3.0, 3.0)
    
    nBins = (getNBins(l_box, r_cut))
    sizeBin = roundUp((l_box / nBins), 6)
    partNum = len(snap.particles.typeid)
    pos = snap.particles.position
    
    # Get the largest x-position in the largest cluster
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    my_clust = cluster.Cluster()
    c_props = cluster.ClusterProperties()
    density = freud.density.LocalDensity(r_max=10., diameter=1.0)
    
    # You need a list of length nBins to hold the sum
    phi_sum = [ 0. for i in range(0, nBins) ]
    p_sum = [ 0. for i in range(0, nBins) ]
    pswim_sum = [ 0. for i in range(0, nBins) ]
    pint_sum = [ 0. for i in range(0, nBins) ]
    # You need one array of length nBins to hold the counts
    num = [ 0. for i in range(0, nBins) ]
    # You should store the max distance of each bin as well
    r_bins = np.arange(sizeBin, sizeBin * nBins, sizeBin)

    # Loop through snapshots
    for j in range(start, end):
    
        print('j')
        print(j)
        
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
        ori = snap.particles.orientation            # orientation
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        
        # Compute the center of mass
        # Compute neighbor list for only largest cluster
        
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        
        cl_all=freud.cluster.Cluster()                      #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()         #Define cluster properties
        ids = cl_all.cluster_idx              # get id of each cluster
        clp_all.compute(system_all, ids)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        min_size=int(partNum/5)
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]
        large_clust_ind_all=np.where(clust_size>min_size)

        com = clp_all.centers[0]                # find cluster CoM

        # Only look at clusters if the size is at least 10% of the system
        if len(large_clust_ind_all[0])>0:
            

            # Keep only positions of largest cluster (faster computations)
            lc_pos = []
            r_com = []
            lc_ids = []
            aligns = []
            pswim = []
            for k in range(0, partNum):
                if ids[k] == lcID:
                    lc_pos.append(pos[k])
                    lc_ids.append(k)
                    # See if particle should be wrapped
                    rrx = lc_pos[-1][0] - com[0]
                    rx = np.abs(rrx)
                    if rx >= h_box:
                        rx -= l_box
                        rx = np.abs(rx)
                        # How should rrx be adjusted
                        if rrx < -h_box:
                            rrx += l_box
                        else:
                            rrx -= l_box
                    rry = lc_pos[-1][1] - com[1]
                    ry = np.abs(rry)
                    if ry >= h_box:
                        print("Wrapping")
                        ry -= l_box
                        ry = np.abs(ry)
                        # How should rrx be adjusted
                        if rry < -h_box:
                            rry += l_box
                        else:
                            rry -= l_box
                    # We want the vector from ref to CoM
                    rrx = -rrx
                    rry = -rry
                    mag = np.sqrt(rx**2 + ry**2)
                    r_com.append(mag)
                    # Now let's get the x and y components of the body axis
                    px = np.sin(ang[k])
                    py = -np.cos(ang[k])
                    # Now compute the dot product
                    r_dot_p = (rrx * px) + (rry * py)
                    # A value of 1 is perfectly aligned
                    r_dot_p /= mag
                    # We don't need to normalize by p, it's magnitude is 1
                    aligns.append(r_dot_p)
                    
                    # Compute the swim pressure
                    swim_dot = (lc_pos[-1][0] * px) + (lc_pos[-1][0] * py)
                    pswim.append(swim_dot)
            
            # Compute interparticle pressure
            pressure = [0. for i in range(0, len(lc_pos))]
            # Create the neighborlist of the system
            lc = freud.locality.AABBQuery(box=f_box, points=f_box.wrap(lc_pos))
            nlist = lc.query(lc_pos, dict(r_min=0.1, r_max=r_cut)).toNeighborList()
            
            # Loop through and compute pressure
            pairs = set()
            for (m, n) in nlist:
                # Never compute things twice
                if (m, n) in pairs or (n, m) in pairs:
                    continue
                # So we know we've computed it
                pairs.add( (m, n) )
                # Loops through each j neighbor of reference particle i
                xx, yy, rr = distComps(lc_pos[m], lc_pos[n][0], lc_pos[n][1])
                fx, fy = computeFLJ(rr, xx, yy, eps)
                # Compute the x force times x distance
                sigx = fx * (xx)
                # Likewise for y
                sigy = fy * (yy)
                pressure[m] += ((sigx + sigy) / 2.)
                pressure[n] += ((sigx + sigy) / 2.)

            # Compute density around largest cluster points
            phi_locs = density.compute(system_all, query_points=lc_pos)
            phi_loc = phi_locs.density * np.pi * 0.25
            
            # Add/increment each particle in appropriate index
            for k in range(0, len(lc_ids)):
                # Convert r to appropriate bin
                tmp_r = int(r_com[k] / sizeBin)
                p_sum[tmp_r] += aligns[k]
                phi_sum[tmp_r] += phi_loc[k]
                pint_sum[tmp_r] += pressure[k]
                num[tmp_r] += 1

            # Write textfile
            
            
            # Append data to file
            g = open(outTxt, 'a')
            for j in range(0, len(r_bins)):
                g.write('{0:.6f}'.format(tst).center(25) + ' ')
                g.write('{0:.6f}'.format(r_bins[j]).center(25) + ' ')
                g.write('{0:.0f}'.format(num[j]).center(25) + ' ')
                g.write('{0:.6f}'.format(phi_sum[j]).center(25) + ' ')
                g.write('{0:.6f}'.format(p_sum[j]).center(25) + ' ')
                g.write('{0:.1f}'.format(pint_sum[j]).center(25) + ' ')
                g.write('{0:.1f}'.format(pswim_sum[j]).center(25) + '\n')
            g.close()