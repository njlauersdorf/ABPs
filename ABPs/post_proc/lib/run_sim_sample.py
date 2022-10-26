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

sys.path.append(os.path.join(os.path.expanduser('~') + '/klotsa/ABPs/post_proc', 'lib'))

import run_sim

sim_functs = run_sim.run_sim(hoomdPath, runFor, dumpFreq, partPercA, peA, peB, partNum, intPhi, eps, aspect_ratio, seed1, seed2, seed3, seed4, seed5)

if init_cond == 'random_init':
    sim_functs.random_init()
elif init_cond == 'homogeneous_cluster':
    sim_functs.homogeneous_cluster()
elif init_cond == 'fast_bulk_cluster':
    sim_functs.fast_bulk_cluster()
elif init_cond == 'slow_bulk_cluster':
    sim_functs.slow_bulk_cluster()
elif init_cond == 'half_cluster':
    sim_functs.half_cluster()
elif init_cond == 'slow_membrane':
    sim_functs.fast_penetrate_slow_membrane()
