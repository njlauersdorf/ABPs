#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
#                           This is an 80 character line                       #
    This is intended to investigate how hard our particles need to be. We want
    to maintain a ratio of active force to LJ well-depth:
                    epsilon = alpha * F_A * sigma / 24.0
    This code will investigate alpha, in order to find the smallest value that
    will maintain a "hard-sphere" potential. (This will optimize computation
    time while keeping our model realistic)
'''

import os, sys
# Read in bash arguments
hoomdPath = "${hoomd_path}" # path to where you installed hoomd-blue '/.../hoomd-blue/build/'
init_cond = "${init_cond}"

runFor = ${runfor}              # simulation length (in tauLJ)
dumpFreq = ${dump_freq}         # how often to dump data
partPercA = ${part_frac_a}      # percentage of A particles
peA = ${pe_a}                   # activity of A particles
peB = ${pe_b}                   # activity of B particles
partNum = ${part_num}           # Number of particles in system
intPhi = ${phi}                 # system area fraction (integer, i.e. 45, 65, etc.)
eps = ${ep}                     # epsilon (potential well depth for LJ potential)
aspect_ratio = "${aspect_ratio}"

seed1 = ${seed1}                # seed for position
seed2 = ${seed2}                # seed for bd equilibration
seed3 = ${seed3}                # seed for initial orientations
seed4 = ${seed4}                # seed for A activity
seed5 = ${seed5}                # seed for B activity

sys.path.append(os.path.expanduser('~') + '/ABPs/src/run_sim')
#sys.path.append(os.path.expanduser('~') + '/hoomd-blue/build/run_test/lib')

import run_sim
import theory

# Set some constants
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
r_cut = 2**(1./6.)

theory_functs = theory.theory()

# Compute parameters from activities
if peA != 0:                        # A particles are NOT Brownian
    epsA = eps
    tauA = theory_functs.computeTauLJ(epsA)
else:                               # A particles are Brownian
    epsA = eps#kT
    tauA = theory_functs.computeTauLJ(epsA)

if peB != 0:                        # B particles are NOT Brownian
    epsB=eps
    tauB = theory_functs.computeTauLJ(epsB)
else:                               # B particles are Brownian
    epsB = eps#kT
    tauB = theory_functs.computeTauLJ(epsB)
    
tauLJ = (tauA if (tauA <= tauB) else tauB)  # use the smaller tauLJ.  Doesn't matter since these are the same
epsA = (epsA if (epsA >= epsB) else epsB)   # use the larger epsilon. Doesn't matter since these are the same


eps = ( ( ( 4 * (peA * (partPercA/100) + peB * (1.0 - (partPercA/100 ) ) ) ) / 24) + 10 )
epsA = ( ( ( 4 * (peA * (partPercA/100) + peB * (1.0 - (partPercA/100 ) ) ) ) / 24) + 10 )
epsB = ( ( ( 4 * (peA * (partPercA/100) + peB * (1.0 - (partPercA/100 ) ) ) ) / 24) + 10 )

tauLJ = theory_functs.computeTauLJ(eps)
tauLJ = theory_functs.computeTauLJ(1.0)

dt = 0.000001 * tauLJ                        # timestep size.  I use 0.000001 for dt=tauLJ* (eps/10^6) generally

sim_functs = run_sim.run_sim(hoomdPath, runFor, dumpFreq, partPercA, peA, peB, partNum, intPhi, eps, aspect_ratio, seed1, seed2, seed3, seed4, seed5, kT, threeEtaPiSigma, sigma, r_cut, tauLJ, epsA, epsB, dt)

if init_cond == 'random_init':
    sim_functs.random_init()
elif init_cond == 'random_init_fine':
    sim_functs.random_init_fine()
elif init_cond == 'homogeneous_cluster':
    sim_functs.homogeneous_cluster()
elif init_cond == 'homogeneous_cluster_fine':
    sim_functs.homogeneous_cluster_fine()
elif init_cond == 'fast_bulk_cluster':
    sim_functs.fast_bulk_cluster()
elif init_cond == 'slow_bulk_cluster':
    sim_functs.slow_bulk_cluster()
elif init_cond == 'half_cluster':
    sim_functs.half_cluster()
elif init_cond == 'constant_pressure':
    sim_functs.constant_pressure()
elif init_cond == 'slow_membrane':
    sim_functs.fast_penetrate_slow_membrane()
elif init_cond == 'stationary_membrane':
    sim_functs.fast_penetrate_stationary_membrane()
elif init_cond == 'immobile_membrane':
    sim_functs.fast_penetrate_immobile_membrane()
elif init_cond == 'immobile_orient_membrane':
    sim_functs.fast_orient_penetrate_immobile_membrane()
elif init_cond == 'slow_constrained_membrane':
    sim_functs.fast_penetrate_slow_constrained_membrane()
elif init_cond == 'slow_int_constrained_membrane':
    sim_functs.fast_interior_slow_constrained_membrane()
elif init_cond == 'slow_adsorb_constrained_membrane':
    sim_functs.fast_adsorb_slow_constrained_membrane()
elif init_cond == 'hard_sphere':
    sim_functs.hard_sphere()