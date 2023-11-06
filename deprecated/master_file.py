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
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage
from scipy import stats
from scipy.optimize import curve_fit

import random

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster



import matplotlib
import matplotlib.collections
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib import collections  as mc
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick

#Set plotting parameters
matplotlib.rc('font', serif='Helvetica Neue')
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5


import itertools


class sim_analysis
    def __init__(self):

        # Input pathing

        inFile = str(sys.argv[1])
        if inFile[0:7] == "cluster":
            add = 'cluster_'
        else:
            add = ''

        #Set particle activities
        try:
            peA = float(sys.argv[2])                        #Activity (Pe) for species A
            peB = float(sys.argv[3])                        #Activity (Pe) for species B
        except:
            try:
                peA = float(sys.argv[2])                        #Activity (Pe) for species A
                peB = 0
            except:
                try:
                    peA = 0
                    peB = float(sys.argv[3])                        #Activity (Pe) for species B
                except:
                    peA=100.
                    peB=100.

        #Set particle fraction (fraction of system consisting of species B)
        try:
            parFrac_orig = float(sys.argv[4])               #Fraction of system consists of species A (chi_A)
            if parFrac_orig<1.0:
                parFrac=parFrac_orig*100.
            else:
                parFrac=parFrac_orig
        except:
            parFrac_orig = 0.5               #Fraction of system consists of species A (chi_A)
            if parFrac_orig<1.0:
                parFrac=parFrac_orig*100.
            else:
                parFrac=parFrac_orig

        #Set particle softness (epsilon)
        try:
            eps = float(sys.argv[5])                        #Softness, coefficient of interparticle repulsion (epsilon)
        except:
            eps = 1.0

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

        #Set bin width for binning and spatially averaging
        try:
            bin_width = float(sys.argv[8])
        except:
            bin_width = 5.0

        #Set number of time steps to average over
        try:
            time_step = float(sys.argv[9])
        except:
            time_step = 1.0

        try:
            hoomdPath = str(sys.argv[10])  #hoomd model path
        except:
            hoomdPath = '/nas/longleaf/home/njlauers/hoomd-blue/build'

        try:
            outPath2=str(sys.argv[11])   #
        except:

        outPath=str(sys.argv[12])    #

        if hoomdPath == '/nas/longleaf/home/njlauers/hoomd-blue/build':
            matplotlib.use('Agg')
        else:
            pass

        peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net/average activity of system

        # Define physical constants
        r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)
        kT = 1.0                        # temperature
        threeEtaPiSigma = 1.0           # drag coefficient
        sigma = 1.0                     # particle diameter
        D_t = kT / threeEtaPiSigma      # translational diffusion constant
        D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
        tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

        #Calculate average compressive force from interface
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

        # Calculate interparticle force from Lennard Jones potential
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

        #Calculate lattice spacing based on force balance
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

        # Calculate LJ time step size
        def computeTauLJ(epsilon):
            '''
            Purpose: Take epsilon (magnitude of lennard-jones force) and compute lennard-jones
            time unit of simulation

            Inputs: epsilon

            Output: lennard-jones time unit
            '''
            tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
            return tauLJ

        #Calculate analytical values
        lat_theory = conForRClust(peNet, eps)
        curPLJ = ljPress(lat_theory, peNet, eps)
        phi_theory = latToPhi(lat_theory)
        phi_g_theory = compPhiG(peNet, lat_theory)

        #Calculate activity-softness dependent variables
        lat=getLat(peNet,eps)
        tauLJ=computeTauLJ(eps)
        dt = dtau * tauLJ                        # timestep size
        n_len = 21
        n_arr = np.linspace(0, n_len-1, n_len)      #Fourier modes
        popt_sum = np.zeros(n_len)                  #Fourier Coefficients

        #Convert HOOMD's orientation (Quaternion) to an angle from the +x-axis
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
# Get infile and open


# Define base file name for outputs
outF = inFile[:-4]

#Read input file
f = hoomd.open(name=inFile, mode='rb')

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












def computeFLJ(r, x1, y1, x2, y2, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (x2 - x1) / r
    fy = f * (y2 - y1) / r
    return fx, fy







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

#Set plot colors
fastCol = '#e31a1c'
slowCol = '#081d58'

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

outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile + "_frame_"
