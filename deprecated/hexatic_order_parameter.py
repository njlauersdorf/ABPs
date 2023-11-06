#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:50:02 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Read in .gsd file of particle positions
2.) Mesh the space
3.) Loop through tsteps and ...
3a.) Place all particles in appropriate mesh grid
3b.) Calculates location of steady state cluster's center of mass
3c.) Translate particles positions such that origin (0,0) is cluster's center of mass
3b.) Loop through all bins ...
3b.i.) Compute number density and average orientation per bin
3c.) Determine and label each phase (bulk dense phase, gas phase, and gas-bulk interface)
3d.) Calculate and output parameters to be used to calculate area fraction of each or all phase(s)
3e.) For frames with clusters, plot particle positions color-coded by phase it's a part of
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
from scipy import interpolate
from scipy import ndimage

import numpy as np
import matplotlib

if hoomdPath == '/nas/longleaf/home/njlauers/hoomd-blue/build':
    matplotlib.use('Agg')
else:
    pass

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


from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit


# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
    
# Define base file name for outputs
outF = inFile[:-4]

#Read input file
f = hoomd.open(name=inFile, mode='rb')

#Label simulation parameters from command line
peA = float(sys.argv[2])                        #Activity (Pe) for species A
peB = float(sys.argv[3])                        #Activity (Pe) for species B 
parFrac_orig = float(sys.argv[4])               #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig

if parFrac==100.0:
    parFrac_orig=0.5
    parFrac=50.0
    
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

eps = float(sys.argv[5])                        #Softness, coefficient of interparticle repulsion (epsilon)

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

# Set some constants
r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

# Define Fourier series for fit
def fourier(x, *a):
    ret = a[1]
    for deg in range(1, int(len(a)/2)):
        ret += ((a[(deg*2)] * np.cos((deg) * ((x-a[0])*np.pi/180))) + (a[2*deg+1] * np.sin((deg) * ((x-a[0])*np.pi/180))))
    return ret

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * (x-f)) + bi * sin(i * (x-f))
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def avgCollisionForce(peNet):
    '''
    Purpose: Average compressive force experienced by a reference particle in the 
    bulk dense phase due to neighboring active forces computed from the integral
    of possible orientations
    
    Inputs: Net activity of system
    
    Output: Average magnitude of compressive forces experienced by a bulk particle
    '''
    
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    
    return (magnitude * peNet) / (np.pi) 

def ljForce(r, eps, sigma=1.):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
    and separation distance of 2 particles to compute magnitude of lennard-jones force experienced
    by each
    
    Inputs: 
        r: Separation distance in simulation units
        epsilon: magnitude of lennard-jones potential
        sigma: particle diameter (default=1.0)
    
    Output: lennard jones force (dU)
    '''
    
    #Dimensionless distance unit
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


# Calculate cluster radius
def conForRClust(pe, eps):
    '''
    Purpose: Compute analytical radius of the custer given activity and softness
    
    Inputs: 
        pe: net activity (peclet number)
        eps: softness (magnitude of repulsive interparticle force)
    
    Output: cluster radius (simulation distance units)
    '''
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out

def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))

def symlog_arr(x):
    """ Returns the symmetric log10 value """
    out_arr = np.zeros(np.shape(x))
    for d in range(0, len(x)):
        for f in range(0, len(x)):
            if x[d][f]!=0:
                out_arr[d][f]=np.sign(x[d][f]) * np.log10(np.abs(x[d][f]))
    return out_arr
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

#Calculate analytical values
lat_theory = conForRClust(peNet, eps)
curPLJ = ljPress(lat_theory, peNet, eps)
phi_theory = latToPhi(lat_theory)
phi_g_theory = compPhiG(peNet, lat_theory)

def quatToAngle(quat):
    '''
    Purpose: Take quaternion orientation vector of particle as given by hoomd-blue
    simulations and output angle between [-pi, pi]
    
    Inputs: Quaternion orientation vector of particle
    
    Output: angle between [-pi, pi]
    '''
    
    r = quat[0]         #magnitude
    x = quat[1]         #x-direction
    y = quat[2]         #y-direction
    z = quat[3]         #z-direction
    rad = math.atan2(y, x)
    
    return rad

def computeTauLJ(epsilon):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force) and compute lennard-jones
    time unit of simulation
    
    Inputs: epsilon
    
    Output: lennard-jones time unit
    '''
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ

def getLat(peNet, eps):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force) and net activity to
    compute lattice spacing as derived analytically (force balance of repulsive LJ force
    and compressive active force)
    
    Inputs: 
        peNet: net activity of system
        epsilon: magnitude of lennard-jones potential
    
    Output: average lattice spacing of system
    '''
    
    #If system is passive, output cut-off radius
    if peNet == 0:
        return 2.**(1./6.)
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    
    #Loop through to find separation distance (r) where lennard-jones force (jForce)
    #approximately equals compressive active force (avgCollisionForce)
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(peNet):
            r -= j
        r += j
        
    return r 

#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)
tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size
n_len = 21
n_arr = np.linspace(0, n_len-1, n_len)      #Fourier modes
popt_sum = np.zeros(n_len)                  #Fourier Coefficients


#Import modules
import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster
from freud import order
import itertools

import bokeh.io
import bokeh.plotting

import freud.util
bokeh.io.output_notebook()


import numpy as np
import math
import random
from scipy import stats

#Set plotting parameters
matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5
    
def computeTauPerTstep(epsilon, mindt=0.000001):
    '''
    Purpose: Take epsilon (magnitude of lennard-jones force), and output the amount
    of Brownian time units per time step in LJ units
    
    Inputs: 
        epsilon: magnitude of lennard-jones potential
        mindt: time step in LJ units (default=0.000001)
    
    Output: lennard jones force (dU)
    '''

    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

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
def make_polygon(sides, radius=1):
    thetas = np.linspace(0, 2*np.pi, sides+1)[:sides]
    vertices = np.array([[radius*np.sin(theta), radius*np.cos(theta)]
                         for theta in thetas])
    return vertices

def default_bokeh(plot):
    """Wrapper which takes the default bokeh outputs and changes them to more
    sensible values
    """
    plot.title.text_font_size = "18pt"
    plot.title.align = "center"

    plot.xaxis.axis_label_text_font_size = "14pt"
    plot.yaxis.axis_label_text_font_size = "14pt"

    plot.xaxis.major_tick_in = 10
    plot.xaxis.major_tick_out = 0
    plot.xaxis.minor_tick_in = 5
    plot.xaxis.minor_tick_out = 0

    plot.yaxis.major_tick_in = 10
    plot.yaxis.major_tick_out = 0
    plot.yaxis.minor_tick_in = 5
    plot.yaxis.minor_tick_out = 0

    plot.xaxis.major_label_text_font_size = "12pt"
    plot.yaxis.major_label_text_font_size = "12pt"            
def cubeellipse(theta, lam=0.5, gamma=0.6, s=4.0, r=1., h=1.):
    """Create an RGB colormap from an input angle theta. Takes lam (a list of
    intensity values, from 0 to 1), gamma (a nonlinear weighting power),
    s (starting angle), r (number of revolutions around the circle), and
    h (a hue factor)."""
    lam = lam**gamma

    a = h*lam*(1 - lam)
    v = np.array([[-.14861, 1.78277], [-.29227, -.90649], [1.97294, 0.]],
                    dtype=np.float32)
    ctarray = np.array([np.cos(theta*r + s), np.sin(theta*r + s)],
                          dtype=np.float32)
    # convert to 255 rgb
    ctarray = 255*(lam + a*v.dot(ctarray)).T
    ctarray = np.clip(ctarray.astype(dtype=np.int32), 0, 255)
    return "#{0:02x}{1:02x}{2:02x}".format(*ctarray)

def local_to_global(verts, positions, orientations):
    """
    Take a list of shape vertices, positions, and orientations and create
    a list of vertices in the "global coordinate system" for plotting
    in bokeh
    """
    verts = np.asarray(verts)
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    # create array of rotation matrices
    rot_mats = np.array([[[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]]
                        for theta in orientations])
    # rotate vertices
    r_verts = np.swapaxes(rot_mats @ verts.T, 1, 2)
    # now translate to global coordinates
    output_array = np.add(r_verts, np.tile(positions[:, np.newaxis, :],
                                           reps=(len(verts), 1)))
    return output_array            
#Open input simulation file
f = hoomd.open(name=inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep
                
#Get particle number from initial frame
snap = f[0]
typ = snap.particles.typeid
partNum = len(typ)

#Set output file names
bin_width = float(sys.argv[8])
time_step = float(sys.argv[9])
outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile + "_frame_"  

outTxt_phase_info = 'PhaseComp_' + outfile + '.txt'
outTxt_theta_ext = 'Theta_vs_radii_ext_' + outfile
outTxt_theta_int = 'Theta_vs_radii_int_' + outfile
outTxt_coeff = 'Coeff_' + outfile + '.txt'
outTxt_bub_info = 'BubComp_' + outfile + '.txt'

#.txt file for saving overall phase composition data  
g = open(outPath2+outTxt_phase_info, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
                        'sizeBin'.center(15) + ' ' +\
                        'clust_size'.center(15) + ' ' +\
                        'Na_bulk'.center(15) + ' ' +\
                        'Nb_bulk'.center(15) + ' ' +\
                        'NBin_bulk'.center(15) + ' ' +\
                        'Na_gas'.center(15) + ' ' +\
                        'Nb_gas'.center(15) + ' ' +\
                        'NBin_gas'.center(15) + ' ' +\
                        'Na_int'.center(15) + ' ' +\
                        'Nb_int'.center(15) + ' ' +\
                        'NBin_int'.center(15) + ' ' +\
                        'Na_bub'.center(15) + ' ' +\
                        'Nb_bub'.center(15) + ' ' +\
                        'NBin_bub'.center(15) + '\n')
g.close()

g = open(outPath2+outTxt_theta_ext +'.txt', 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
            'clust_size'.center(15) + ' ' +\
            'int_id'.center(15) + ' ' +\
            'bub_id'.center(15) + ' ' +\
            'coeff_num'.center(15) + ' ' +\
            'coeff'.center(15) + '\n')
g.close()

g = open(outPath2+outTxt_theta_int+'.txt', 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
            'clust_size'.center(15) + ' ' +\
            'int_id'.center(15) + ' ' +\
            'bub_id'.center(15) + ' ' +\
            'coeff_num'.center(15) + ' ' +\
            'coeff'.center(15) + '\n')
g.close()

#.txt file for saving interface phase composition data  
g = open(outPath2+outTxt_bub_info, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
                        'sizeBin'.center(15) + ' ' +\
                        'clust_size'.center(15) + ' ' +\
                        'int_id'.center(15) + ' ' +\
                        'bub_id'.center(15) + ' ' +\
                        'Na'.center(15) + ' ' +\
                        'Nb'.center(15) + ' ' +\
                        'radius'.center(15) + ' ' +\
                        'radius_err'.center(15) + ' ' +\
                        'sa_ext'.center(15) + ' ' +\
                        'sa_int'.center(15) + ' ' +\
                        'edge_width'.center(15) + ' ' +\
                        'edge_width_err'.center(15) + ' ' +\
                        'NBin'.center(15) + '\n')
g.close()

with hoomd.open(name=inFile, mode='rb') as t:
    
    start = int(420/time_step)#205                                             # first frame to process
    dumps = int(t.__len__())                                # get number of timesteps dumped
    end = int(dumps/time_step)-1                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step
    
    # Get box dimensions
    box_data = snap.configuration.box                       
    l_box = box_data[0]                                     #box length
    h_box = l_box / 2.0                                     #half box length
    
    #2D binning of system
    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / NBins), 6)
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)
    
    time_arr=np.zeros(dumps)                                  #time step array
    
    for p in range(start, end):     
        j=int(p*time_step)
        print('j')
        print(j)

        snap = t[j]                                 #Take current frame

        #Arrays of particle data
        pos = snap.particles.position               # position
        pos[:,-1] = 0.0                             # 2D system
        xy = np.delete(pos, 2, 1)
        
        ori = snap.particles.orientation            #Orientation (quaternions)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        
        typ = snap.particles.typeid                 # Particle type
        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
        
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        time_arr[j]=tst
        
        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        nlist = system_all.query(f_box.wrap(pos), {'r_max': lat}).toNeighborList()
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes
        
        
        min_size=int(partNum/8)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
        
        
        
        
        #If a single cluster is greater than minimum size, determine CoM of largest cluster
        if len(large_clust_ind_all[0])>0:
            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box
            
            com_tmp_posX_temp = query_points[0] 
            com_tmp_posY_temp = query_points[1] 
        else:
            
            com_tmp_posX = h_box
            com_tmp_posY = h_box
            
            com_tmp_posX_temp = 0
            com_tmp_posY_temp = 0
        
        #shift reference frame to center of mass of cluster   
        pos[:,0]= pos[:,0]-com_tmp_posX_temp    
        pos[:,1]= pos[:,1]-com_tmp_posY_temp
        
        #Ensure particles are within simulation box (periodic boundary conditions)
        for i in range(0, partNum):
                if pos[i,0]>h_box:
                    pos[i,0]=pos[i,0]-l_box
                elif pos[i,0]<-h_box:
                    pos[i,0]=pos[i,0]+l_box
                    
                if pos[i,1]>h_box:
                    pos[i,1]=pos[i,1]-l_box
                elif pos[i,1]<-h_box:
                    pos[i,1]=pos[i,1]+l_box
        
        #
        hex_order = freud.order.Hexatic(k=6)

        # Compute hexatic order for 6 nearest neighbors
        hex_order.compute(system=(f_box, pos))#, neighbors=nlist)
        psi_k = hex_order.particle_order
        
        #Average hexatic order parameter
        avg_psi_k = np.mean(psi_k)
        
        # Create an array of angles relative to the average
        order_param = np.abs(psi_k)
        
        
        
        #Plot particles colorized by hexatic order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(order_param)
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']


        
        im = plt.scatter(pos[:,0], pos[:,1], c=order_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Psi$', labelpad=-55, y=1.04, rotation=0, fontsize=18)
                
        plt.xlim(-h_box, h_box)
        plt.ylim(-h_box, h_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                       
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        pad = str(j).zfill(4)
        plt.savefig(outPath + 'hexatic_order_' + out + pad + ".png", dpi=100)
        plt.close()
        
        #Compute translational order parameter
        trans_order = freud.order.Translational(k=6)
        trans_order.compute(system=(f_box, pos))#, neighbors=nlist)
        trans_param = np.abs(trans_order.particle_order)
        
        #Plot particles colorized by translational order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(order_param)
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']


        
        im = plt.scatter(pos[:,0], pos[:,1], c=trans_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Psi$', labelpad=-55, y=1.04, rotation=0, fontsize=18)
                
        plt.xlim(-h_box, h_box)
        plt.ylim(-h_box, h_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                       
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        pad = str(j).zfill(4)
        plt.savefig(outPath + 'translational_order_' + out + pad + ".png", dpi=100)
        plt.close()

        #Compute Steinhardt order parameter
        ql = freud.order.Steinhardt(l=6)
        ql.compute(system=(f_box, pos), neighbors=nlist)
        output = np.abs(ql.particle_order)
        
        #Plot particles colorized by Steinhardt order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(order_param)
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']


        
        im = plt.scatter(pos[:,0], pos[:,1], c=output, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Psi$', labelpad=-55, y=1.04, rotation=0, fontsize=18)
        
        
        plt.xlim(-h_box, h_box)
        plt.ylim(-h_box, h_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                       
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        #plt.gcf().subplots_adjust(top=-0.15)
        pad = str(j).zfill(4)
        plt.savefig(outPath + 'steinhardt_order_' + out + pad + ".png", dpi=100)
        plt.close()
        
        #Calculate relative bond orientation of crystal domains
        relative_angles = np.angle(psi_k)
        
        #Since hexagonal domains, translate angle to 0 to pi/3 radians
        for g in range(0, len(relative_angles)):
            if relative_angles[g]<(-2*np.pi/3):
                relative_angles[g]+=(np.pi)
            elif (-2*np.pi/3)<=relative_angles[g]<(-np.pi/3):
                relative_angles[g]+=(2*np.pi/3)
            elif (-np.pi/3)<=relative_angles[g]<0:
                relative_angles[g]+=(np.pi/3)
            elif np.pi/3<relative_angles[g]<=(2*np.pi/3):
                relative_angles[g]-=(np.pi/3)
            elif (2*np.pi/3) < relative_angles[g]:
                relative_angles[g]-=(2*np.pi/3) 
                
        #Plot particles colorized by bond orientation angle
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(relative_angles)
        max_n = np.max(relative_angles)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.scatter(pos[:,0], pos[:,1], c=relative_angles, s=0.7, vmin=0.0, vmax=np.pi/3, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=0.0, vmax=np.pi/3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm)
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\theta$', labelpad=-38, y=1.05, rotation=0, fontsize=18)
        clb.locator     = matplotlib.ticker.FixedLocator(tick_locs)
        clb.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
        clb.update_ticks()
        
        plt.xlim(-h_box, h_box)
        plt.ylim(-h_box, h_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                       
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        pad = str(j).zfill(4)
        plt.savefig(outPath + 'relative_angle_' + out + pad + ".png", dpi=100)
        plt.close()
        
        '''
        #NEMATIC ORDER IN-PROGRESS
        #Calculate nematic order parameter
        nop = freud.order.Nematic([1, 0, 0])
        nop.compute(ori)
        print(nop.order)
        nematic_order = np.abs(nop.order)
        
        print(np.shape(nematic_order))
        #Plot particles colorized by nematic order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(order_param)
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']
        

        
        im = plt.scatter(pos[:,0], pos[:,1], c=nematic_order, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Psi$', labelpad=-55, y=1.04, rotation=0, fontsize=18)
        
        
        plt.xlim(-h_box, h_box)
        plt.ylim(-h_box, h_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                       
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        #plt.gcf().subplots_adjust(top=-0.15)
        pad = str(j).zfill(4)
        plt.savefig(outPath + 'nematic_order_' + out + pad + ".png", dpi=100)
        plt.close()
        '''