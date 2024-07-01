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

# Import modules
import sys
import os

first_time = 1

# Path to hoomd and output data
hoomdPath=str(sys.argv[2])
outPath = str(sys.argv[3])

# Import modules
from gsd import hoomd
import freud
import numpy as np
import math
import scipy
from scipy import stats
import matplotlib

# If running on Longleaf, use matplotlib from desktop
if hoomdPath[:4] == '/nas':
    matplotlib.use('Agg')
else:
    pass

# Import modules
import matplotlib.pyplot as plt


# Add path to post-processing library
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

#Import post-processing library
import plotting_utility
import binning
import phase_identification
import interface
import theory
import plotting
import utility
import measurement
import kinetics
import stress_and_pressure
import data_output
import particles
import csv

theory_functs = theory.theory()

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
peA = float(sys.argv[4])                        #Activity (Pe) for species A
peB = float(sys.argv[5])                        #Activity (Pe) for species B
parFrac_orig = float(sys.argv[6])               #Fraction of system consists of species A (chi_A)

#Convert particle fraction to a percent
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig

if parFrac==100.0:
    parFrac_orig=0.5
    parFrac=50.0

peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

eps = float(sys.argv[7])                        #Softness, coefficient of interparticle repulsion (epsilon)

#Set system area fraction (phi)
try:
    
    phi = float(sys.argv[8])
    if phi <= 1.0:
        intPhi = int(phi * 100)
    else:
        intPhi = int(phi)
except:
    phi = 0.6
    intPhi = 60

#Get simulation time step
try:
    dtau = float(sys.argv[9])
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

#Define colors for plots
yellow = ("#fdfd96")
green = ("#77dd77")
red = ("#ff6961")
purple = ("#cab2d6")
new_green = ("#39FF14")
new_brown = ("#b15928")

#Calculate analytical values
lat_theory = theory_functs.conForRClust(peNet, eps)
curPLJ = theory_functs.ljPress(lat_theory, peNet, eps)
phi_theory = theory_functs.latToPhi(lat_theory)
if peNet > 0:
    phi_g_theory = theory_functs.compPhiG(peNet, lat_theory)
else:
    phi_g_theory = phi


#Calculate activity-softness dependent variables
lat=theory_functs.getLat(peNet,eps)
tauLJ=theory_functs.computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size

start_dict = None
#Import modules
import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster
import itertools
import random

#Set plot colors
fastCol = '#e31a1c'
slowCol = '#081d58'

#Open input simulation file
f = hoomd.open(name=inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
tauPerDT = theory_functs.computeTauPerTstep(epsilon=eps)  # brownian time per timestep

#Get particle number from initial frame
snap = f[0]
typ = snap.particles.typeid
partNum = len(typ)
lat_sum = 0
action_arr = np.array([])

#Set output file names
bin_width = float(sys.argv[10])
time_step = float(sys.argv[11])
measurement_method = str(sys.argv[12])

plot = str(sys.argv[13])
start_frame = str(sys.argv[14])
end_frame = str(sys.argv[15])
optional_method = str(sys.argv[16])

outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(intPhi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile + "_frame_"

dataPath = outPath + '_txt_files/'
picPath = outPath + '_pic_files/'
averagesPath = outPath[:-8] + 'averages/'
timePath = outPath[:-8] + 'heterogeneity_time/'

# Instantiate empty outputs for time-looped functions
partPhase_time = np.array([])
partPhase_time_arr = np.array([])
clust_size_arr = np.array([])

# Instantiate radial distribution function outputs
avg_num_dens_dict = {}
avg_radial_df_dict = {}
sum_num = 0

# Instantiate penetration y-shift and displacement
vertical_shift = 0
dify_long = 0

# Instantiate MSD outputs
com_x_msd = np.array([0])                       # Cluster CoM x-MSD
com_y_msd = np.array([0])                       # Cluster CoM y-MSD
com_r_msd = np.array([0])                       # Cluster CoM total MSD

# Instantiate part-velocity outputs
time_velA_mag = np.array([])
time_velB_mag = np.array([])

time_clust_all_size = np.array([])
time_clust_A_size = np.array([])
time_clust_B_size = np.array([])

tracer_ids = np.array([])

# Optional input parameters for plotting data
com_option = False
mono_option = False
zoom_option = False
orientation_option = False
interface_option = False
banner_option = False
presentation_option = False
large_arrows_option = False
mono_slow_option = False
mono_fast_option = False
swap_col_option = False

# Check whether optional input parameters given in user input
if optional_method != 'none':
    optional_options = optional_method.split('_')
    for i in range(0, len(optional_options)):
        if optional_options[i] == 'com':
            com_option = True
        elif optional_options[i] == 'mono':
            mono_option = True
        elif optional_options[i] == 'zoom':
            zoom_option = True
        elif optional_options[i] == 'orient':
            orientation_option = True
        elif optional_options[i] == 'interface':
            interface_option = True
        elif optional_options[i] == 'banner':
            banner_option = True
        elif optional_options[i] == 'presentation':
            presentation_option = True
        elif optional_options[i] == 'large':
            large_arrows_option = True
        elif optional_options[i] == 'mono-slow':
            mono_slow_option = True
        elif optional_options[i] == 'mono-fast':
            mono_fast_option = True
        elif optional_options[i] == 'swap':
            swap_col_option = True

time_prob_AA_bulk = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
time_prob_AB_bulk = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
time_prob_BA_bulk = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
time_prob_BB_bulk = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
time_prob_num = 0 

neigh_dict = {}


    

with hoomd.open(name=inFile, mode='rb') as t:
    
    dumps = int(t.__len__())

    try:
        start = int(int(start_frame)/time_step)
    except:
        start = int(0/time_step)                                                 # first frame to process

    try:
        end = int(int(end_frame)/time_step)
    except:
        end = int(dumps/time_step)-1                                             # final frame to process

    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step

    snap = t[1]                                             # Take first snap for box
    second_tstep = snap.configuration.step                   # First time step
    second_tstep -= first_tstep                          # normalize by first timestep
    dt_step = second_tstep * dtau                                 # convert to Brownian time
    
    # Get box dimensions
    box_data = snap.configuration.box

    lx_box = box_data[0]                                     #box length
    ly_box = box_data[1]
    hx_box = lx_box / 2.0                                     #half box length
    hy_box = ly_box / 2.0                                     #half box length
    utility_functs = utility.utility(lx_box, ly_box)

    #2D binning of system
    NBins_x = utility_functs.getNBins(lx_box, r_cut)
    NBins_y = utility_functs.getNBins(ly_box, r_cut)
    sizeBin_x = utility_functs.roundUp((lx_box / NBins_x), 6)
    sizeBin_y = utility_functs.roundUp((ly_box / NBins_y), 6)

    f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

    time_arr=np.zeros(dumps)                                  #time step array

    plotting_utility_functs = plotting_utility.plotting_utility(lx_box, ly_box, partNum, typ)

    # Tracks if steady state occurs once
    steady_state_once = 'False'

    if measurement_method=='crop-gsd':
        utility_functs.crop_gsd(inFile, start, end)

    elif measurement_method == 'circular-wall':
        outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
            
        # If time-averaged fluctuations file does not exist, calculate it...
        if os.path.isfile(averagesPath + "radial_avgs_fa_avg_" + outfile+ '.csv')==0:
            sum_num = 0
            
            # Time range to calculate time-average at steady state
            end_avg = int(dumps/time_step)-1
            start_avg = int(end_avg/3)

            # Loop over time
            for p in range(start_avg, end_avg):

                # Current time step
                j=int(p*time_step)

                print('j')
                print(j)
                
                snap = t[j]                                 #Take current frame

                #Arrays of particle data
                pos = snap.particles.position               # current positions
                pos[:,-1] = 0.0                             # 2D system
                xy = np.delete(pos, 2, 1)

                ori = snap.particles.orientation            #current orientation (quaternions)
                ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]
                x_orient_arr = np.array(list(map(utility_functs.quatToXOrient, ori))) # convert to [-pi, pi]
                y_orient_arr = np.array(list(map(utility_functs.quatToYOrient, ori))) # convert to [-pi, pi]

                typ = snap.particles.typeid                 # Particle type
                typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
                typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

                tst = snap.configuration.step               # timestep
                tst -= first_tstep                          # normalize by first timestep
                tst *= dtau                                 # convert to Brownian time
                time_arr[j]=tst

                #Bin system to calculate orientation and alignment that will be used in vector plots
                NBins_x = utility_functs.getNBins(lx_box, bin_width)
                NBins_y = utility_functs.getNBins(ly_box, bin_width)

                # Calculate size of bins
                sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
                sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

                # Instantiate particle properties module
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                particle_prop_functs.radial_heterogeneity_circle()


    else:
        if measurement_method == 'radial-heterogeneity':
            
            outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
            
            # If time-averaged fluctuations file does not exist, calculate it...
            if os.path.isfile(averagesPath + "radial_avgs_fa_avg_" + outfile+ '.csv')==0:
                sum_num = 0
                
                # Time range to calculate time-average at steady state
                end_avg = int(dumps/time_step)-1
                start_avg = int(end_avg/3)

                # Loop over time
                for p in range(start_avg, end_avg):

                    # Current time step
                    j=int(p*time_step)

                    print('j')
                    print(j)
                    
                    snap = t[j]                                 #Take current frame

                    #Arrays of particle data
                    pos = snap.particles.position               # current positions
                    pos[:,-1] = 0.0                             # 2D system
                    xy = np.delete(pos, 2, 1)

                    ori = snap.particles.orientation            #current orientation (quaternions)
                    ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]
                    x_orient_arr = np.array(list(map(utility_functs.quatToXOrient, ori))) # convert to [-pi, pi]
                    y_orient_arr = np.array(list(map(utility_functs.quatToYOrient, ori))) # convert to [-pi, pi]

                    typ = snap.particles.typeid                 # Particle type
                    typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
                    typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

                    tst = snap.configuration.step               # timestep
                    tst -= first_tstep                          # normalize by first timestep
                    tst *= dtau                                 # convert to Brownian time
                    time_arr[j]=tst
                    

                    #Compute cluster parameters using neighbor list of all particles within LJ cut-off distance
                    system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
                    cl_all=freud.cluster.Cluster() 

                    cl_all.compute(system_all, neighbors={'r_max': 1.3})        # Calculate clusters given neighbor list, positions,
                                                                            # and maximal radial interaction distance
                    clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
                    ids = cl_all.cluster_idx                                    # get id of each cluster
                    clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
                    clust_size = clp_all.sizes                                  # find cluster sizes

                    min_size=int(partNum/10)                                     #Minimum cluster size for measurements to happen
                    lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
                    large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
                    clust_large = np.amax(clust_size)
                    

                    # Instantiate particle properties module
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                    # If elongated simulation box...
                    if lx_box != ly_box:

                        clust_large = 0
                    
                    # Instantiate empty phase identification arrays
                    partTyp=np.zeros(partNum)
                    partPhase=np.zeros(partNum)
                    edgePhase=np.zeros(partNum)
                    bulkPhase=np.zeros(partNum)

                    # Calculate cluster CoM
                    com_dict = plotting_utility_functs.com_view(pos, clp_all)
                    
                    # If CoM option given, convert to CoM view
                    if com_option == True:
                        pos = com_dict['pos']
                    #else:
                    #    pos[:,0] = pos[:,0]
                    #    out = np.where(pos[:,0]<-hx_box)[0]
                    #    pos[out,0] = pos[out,0] + lx_box


                    #Bin system to calculate orientation and alignment that will be used in vector plots
                    NBins_x = utility_functs.getNBins(lx_box, bin_width)
                    NBins_y = utility_functs.getNBins(ly_box, bin_width)

                    # Calculate size of bins
                    sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
                    sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

                    # Instantiate binning functions module
                    binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)
                    
                    # Calculate bin positions
                    pos_dict = binning_functs.create_bins()

                    # Assign particles to bins
                    part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                    # Calculate average orientation per bin
                    orient_dict = binning_functs.bin_orient(part_dict, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                    # Calculate area fraction per bin
                    area_frac_dict = binning_functs.bin_area_frac(part_dict)

                    # Calculate average activity per bin
                    activ_dict = binning_functs.bin_activity(part_dict)

                    # Define output file name
                    outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
                    out = outfile + "_frame_"
                    pad = str(j).zfill(5)
                    outFile = out + pad

                    
                    min_size = 0
                    # If cluster sufficiently large
                    if clust_large >= min_size:
                        
                        # Instantiate empty binning arrays
                        clust_size_arr = np.append(clust_size_arr, clust_large)
                        fa_all_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_x_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_y_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        fa_all_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        # Bin average alignment toward cluster's CoM
                        align_dict = binning_functs.bin_align(orient_dict)

                        

                        #Time frame for plots
                        pad = str(j).zfill(5)

                        # Bin average aligned active force pressure
                        press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)

                        # Bin average active force normal to cluster CoM
                        normal_fa_dict = binning_functs.bin_normal_active_fa(align_dict, area_frac_dict, activ_dict)

                        # Find curl and divergence of binned average alignment toward cluster CoM
                        align_grad_dict = binning_functs.curl_and_div(align_dict)

                        # Instantiate plotting functions module
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)

                        # Instantiate phase identification functions module
                        phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ)

                        # Identify phases of system
                        phase_dict = phase_ident_functs.phase_ident()
                        #phase_dict = phase_ident_functs.phase_ident_planar()

                        # Find IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        # Blur phases to mitigate noise
                        phase_dict = phase_ident_functs.phase_blur(phase_dict)

                        # Update phases of each particle ID
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Find updated IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        # Count number of particles per phase
                        count_dict = particle_prop_functs.phase_count(phase_dict)

                        # Find CoM of bulk phase
                        bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

                        # Separate non-connecting bulk phases
                        bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

                        # Separate non-connecting interfaces
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

                        # Update phase identification array
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Reduce mis-identification of gas
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

                        # Find interface composition
                        phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

                        # Find bulk composition
                        bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

                        # Sort bulk by largest to smallest
                        bulk_comp_dict = phase_ident_functs.phase_sort(bulk_comp_dict)

                        # Sort interface by largest to smallest
                        int_comp_dict = phase_ident_functs.phase_sort(int_comp_dict)

                        # Instantiate interface functions module
                        interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, x_orient_arr, y_orient_arr, pos)
                        
                        # Identify interior and exterior surface bin points
                        surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

                        #planar_surface_dict = interface_functs.det_planar_surface_points(phase_dict, int_dict, int_comp_dict)
                        
                        #Save positions of external and internal edges
                        clust_true = 0

                        # Sort surface points for both interior and exterior surfaces of each interface
                        surface2_pos_dict = interface_functs.surface_sort(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'])
                        surface1_pos_dict = interface_functs.surface_sort(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'])

                        # Find CoM of each surface
                        surface_com_dict = interface_functs.surface_com(int_dict, int_comp_dict, surface_dict)

                        # Find radius of surface
                        surface_radius_bin = interface_functs.surface_radius_bins(int_dict, int_comp_dict, surface_dict, surface_com_dict)

                        # Count bins per phase
                        bin_count_dict = phase_ident_functs.phase_bin_count(phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict)

                        # Bin average active force
                        active_fa_dict = binning_functs.bin_active_fa(orient_dict, part_dict, phase_dict['bin'])


                        # Calculate larger bins for orientation visualizations (note this is only used for visualizations)
                        bin_width2 = 15

                        #Bin system to calculate orientation and alignment that will be used in vector plots
                        NBins_x2 = utility_functs.getNBins(lx_box, bin_width2)
                        NBins_y2 = utility_functs.getNBins(ly_box, bin_width2)

                        # Calculate size of bins
                        sizeBin_x2 = utility_functs.roundUp(((lx_box) / NBins_x2), 6)
                        sizeBin_y2 = utility_functs.roundUp(((ly_box) / NBins_y2), 6)

                        # Instantiate binning functions module
                        binning_functs2 = binning.binning(lx_box, ly_box, partNum, NBins_x2, NBins_y2, peA, peB, typ, eps)
                            
                        # Calculate bin positions
                        pos_dict2 = binning_functs2.create_bins()

                        # Assign particles to bins
                        part_dict2 = binning_functs2.bin_parts(pos, ids, clust_size)

                        # Calculate average orientation per bin
                        orient_dict2 = binning_functs2.bin_orient(part_dict2, pos, x_orient_arr, y_orient_arr, com_dict['com'])
                        
                        #Slow/fast composition of bulk phase
                        part_count_dict, part_id_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)

                        # Instantiate data output functions module
                        data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)
                        
                        #Instantiate dictionaries to save data to
                        all_surface_curves = {}
                        all_surface_measurements = {}
                        
                        # Separate individual interfaces/surfaces
                        sep_surface_dict = interface_functs.separate_surfaces(surface_dict, int_dict, int_comp_dict)

                        # Loop over every surface
                        for m in range(0, len(sep_surface_dict)):

                            # Instantiate dictionaries to save data to
                            averaged_data_arr = {}

                            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                            
                            all_surface_curves[key] = {}
                            all_surface_measurements[key] = {}
                            
                            # Save composition data of interface
                            if (int_comp_dict['ids'][m]!=999):
                                averaged_data_arr['int_id'] = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])
                                averaged_data_arr['bub_id'] = int(int_comp_dict['ids'][m])
                                averaged_data_arr['Na'] = int(int_comp_dict['comp']['A'][m])
                                averaged_data_arr['Nb'] = int(int_comp_dict['comp']['B'][m])
                                averaged_data_arr['Nbin'] = int(bin_count_dict['ids']['int'][m])

                            # If sufficient interior interface points, take measurements
                            if sep_surface_dict[key]['interior']['num']>0:

                                # Sort surface points to curve
                                sort_interior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['interior'])

                                # Prepare surface curve for interpolation
                                sort_interior_ids = interface_functs.surface_curve_prep(sort_interior_ids, int_type = 'interior')

                                # Interpolate surface curve
                                all_surface_curves[key]['interior'] = interface_functs.surface_curve_interp(sort_interior_ids)

                                # Find surface curve CoM
                                com_pov_interior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['interior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['interior'] = interface_functs.surface_radius(com_pov_interior_pos['pos'])

                                # Measure surface curve area
                                all_surface_measurements[key]['interior']['surface area'] = interface_functs.surface_area(com_pov_interior_pos['pos'])

                                # Save surface curve CoM 
                                all_surface_measurements[key]['interior']['com'] = com_pov_interior_pos['com']

                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['interior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['interior'])

                                # Save radial measurements
                                averaged_data_arr['int_mean_rad'] = all_surface_measurements[key]['interior']['mean radius']
                                averaged_data_arr['int_std_rad'] = all_surface_measurements[key]['interior']['std radius']
                                averaged_data_arr['int_sa'] = all_surface_measurements[key]['interior']['surface area']

                            else:
                                averaged_data_arr['int_mean_rad'] = 0
                                averaged_data_arr['int_std_rad'] = 0
                                averaged_data_arr['int_sa'] = 0

                            # If sufficient exterior interface points, take measurements
                            if sep_surface_dict[key]['exterior']['num']>0:

                                # Sort surface points to curve
                                sort_exterior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['exterior'])

                                # Prepare surface curve for interpolation
                                sort_exterior_ids = interface_functs.surface_curve_prep(sort_exterior_ids, int_type = 'exterior')
                                
                                # Interpolate surface curve
                                all_surface_curves[key]['exterior'] = interface_functs.surface_curve_interp(sort_exterior_ids)

                                # Find surface curve CoM
                                com_pov_exterior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['exterior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['exterior'] = interface_functs.surface_radius(com_pov_exterior_pos['pos'])
                                
                                # Measure surface curve area
                                all_surface_measurements[key]['exterior']['surface area'] = interface_functs.surface_area(com_pov_exterior_pos['pos'])
                                
                                # Save surface curve CoM 
                                all_surface_measurements[key]['exterior']['com'] = com_pov_exterior_pos['com']
                                
                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['exterior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['exterior'])
                                
                                # Save radial measurements
                                averaged_data_arr['ext_mean_rad'] = all_surface_measurements[key]['exterior']['mean radius']
                                averaged_data_arr['ext_std_rad'] = all_surface_measurements[key]['exterior']['std radius']
                                averaged_data_arr['ext_sa'] = all_surface_measurements[key]['exterior']['surface area']
                            else:
                                averaged_data_arr['ext_mean_rad'] = 0
                                averaged_data_arr['ext_std_rad'] = 0
                                averaged_data_arr['ext_sa'] = 0

                            # If sufficient exterior and interior interface points, measure interface width
                            if (sep_surface_dict[key]['exterior']['num']>0) & sep_surface_dict[key]['interior']['num']>0:
                                all_surface_measurements[key]['exterior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                all_surface_measurements[key]['interior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                averaged_data_arr['width'] = all_surface_measurements[key]['exterior']['surface width']['width']
                            else:
                                averaged_data_arr['width'] = 0
                            
                            # If measurement method specified, save interface data
                            if measurement_method == 'interface-props':
                                data_output_functs.write_to_txt(averaged_data_arr, dataPath + 'BubComp_' + outfile + '.txt')
                        

                        
                        # If cluster has been initially formed, 
                        if steady_state_once == 'False':

                            # Instantiate array for saving largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_arr = np.zeros(partNum)
                            in_clust_arr[clust_id_time]=1

                            # Instantiate array for saving particle positions over time
                            pos_x_arr_time = pos[:,0]
                            pos_y_arr_time = pos[:,1]

                            # Instantiate array for saving surface CoM over time
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            # Instantiate array for saving cluster CoM over time
                            com_x_parts_arr_time = np.array([com_dict['com']['x']])
                            com_y_parts_arr_time = np.array([com_dict['com']['y']])

                            # Instantiate array for saving phase information over time
                            partPhase_time = phase_dict['part']

                            time_entered_bulk = np.ones(partNum) * tst
                            time_entered_gas = np.ones(partNum) * tst

                            # Instantiate array for saving time step
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Change to True since steady state has been reached
                            steady_state_once = 'True'

                            start_dict = {'bulk': {'time': time_entered_bulk}, 'gas': {'time': time_entered_gas} }
                            lifetime_dict = {}
                            msd_bulk_dict = {}
                            lifetime_stat_dict = {}

                        # If cluster has been formed previously
                        else:

                            # Save largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_temp = np.zeros(partNum)
                            in_clust_temp[clust_id_time]=1
                            in_clust_arr = np.vstack((in_clust_arr, in_clust_temp))

                            # Save particle positions over time
                            pos_x_arr_time = np.vstack((pos_x_arr_time, pos[:,0]))
                            pos_y_arr_time = np.vstack((pos_y_arr_time, pos[:,1]))
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Save phase information over time
                            partPhase_time = np.vstack((partPhase_time, phase_dict['part']))

                            # Save surface CoM over time
                            try:
                                com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box)
                                com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box)
                            except:
                                com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box)
                                com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box)

                            # Save cluster CoM over time
                            com_x_parts_arr_time = np.append(com_x_parts_arr_time, com_dict['com']['x'])
                            com_y_parts_arr_time = np.append(com_y_parts_arr_time, com_dict['com']['y'])

                        # Calculate alignment of interface with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
                        
                        # Calculate alignment of bulk with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)

                        # Calculate alignment of gas with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)

                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        try:
                            
                            # Calculate average binned interface properties for current time frame
                            single_time_dict, int_single_time_dict, plot_dict, plot_bin_dict = particle_prop_functs.radial_heterogeneity_avgs(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict, phase_dict)

                            # Find unique radial positions of bins
                            unique_rad = np.unique(single_time_dict['rad'])

                            # Find unique angles of bins
                            unique_theta = np.unique(single_time_dict['theta'])

                            # Instantiate empty arrays for radial integration
                            int_fa_dens_time_theta_avg = np.zeros(len(unique_theta))        # Radially integrated (theta) aligned body force density for all particles for difference between current time step and steady state at each bin
                            int_faA_dens_time_theta_avg = np.zeros(len(unique_theta))       # Radially integrated (theta) aligned body force density for A particles for difference between current time step and steady state at each bin
                            int_faB_dens_time_theta_avg = np.zeros(len(unique_theta))       # Radially integrated (theta) aligned body force density for B particles for difference between current time step and steady state at each bin

                            int_fa_avg_real_time_theta_avg = np.zeros(len(unique_theta))    # Radially integrated (theta) average body force magnitude for all particles for difference between current time step and steady state at each bin

                            int_fa_sum_time_theta_avg = np.zeros(len(unique_theta))         # Radially integrated (theta) total aligned body force magnitude for all particles for difference between current time step and steady state at each bin
                            int_faA_sum_time_theta_avg = np.zeros(len(unique_theta))        # Radially integrated (theta) total aligned body force magnitude for A particles for difference between current time step and steady state at each bin
                            int_faB_sum_time_theta_avg = np.zeros(len(unique_theta))        # Radially integrated (theta) total aligned body force magnitude for B particles for difference between current time step and steady state at each bin

                            int_fa_avg_time_theta_avg = np.zeros(len(unique_theta))         # Radially integrated (theta) average aligned body force magnitude for all particles for difference between current time step and steady state at each bin
                            int_faA_avg_time_theta_avg = np.zeros(len(unique_theta))        # Radially integrated (theta) average aligned body force magnitude for A particles for difference between current time step and steady state at each bin
                            int_faB_avg_time_theta_avg = np.zeros(len(unique_theta))        # Radially integrated (theta) average aligned body force magnitude for B particles for difference between current time step and steady state at each bin

                            int_num_dens_time_theta_avg = np.zeros(len(unique_theta))       # Radially integrated (theta) average number density for all particles for difference between current time step and steady state at each bin
                            int_num_densA_time_theta_avg = np.zeros(len(unique_theta))      # Radially integrated (theta) average number density for A particles for difference between current time step and steady state at each bin
                            int_num_densB_time_theta_avg = np.zeros(len(unique_theta))      # Radially integrated (theta) average number density for B particles for difference between current time step and steady state at each bin

                            int_part_fracA_time_theta_avg = np.zeros(len(unique_theta))      # Radially integrated (theta) average number density for A particles for difference between current time step and steady state at each bin
                            int_part_fracB_time_theta_avg = np.zeros(len(unique_theta))      # Radially integrated (theta) average number density for B particles for difference between current time step and steady state at each bin

                            int_align_time_theta_avg = np.zeros(len(unique_theta))          # Radially integrated (theta) average alignment for all particles for difference between current time step and steady state at each bin
                            int_alignA_time_theta_avg = np.zeros(len(unique_theta))         # Radially integrated (theta) average alignment for A particles for difference between current time step and steady state at each bin
                            int_alignB_time_theta_avg = np.zeros(len(unique_theta))         # Radially integrated (theta) average alignment for B particles for difference between current time step and steady state at each bin

                            # Find radii within 1/3 and 1.1 cluster radii from center of mass
                            temp_id_new = np.where((unique_rad>=0.3) & (unique_rad<=1.1))

                            # Loop over radial bin locations within desired range
                            for j in range(1, len(temp_id_new)):

                                # Find bins that match current radial location
                                test_id = np.where(single_time_dict['rad']==unique_rad[temp_id_new[j]])[0]

                                # If a match is found...
                                if len(test_id)>0:

                                    # Radial step for trapezoidal rule
                                    rad_step = (unique_rad[temp_id_new[j]]-unique_rad[temp_id_new[j-1]])*single_time_dict['radius_ang'][temp_id_new[j]]

                                    # Radially integrate via trapezoidal rule the average aligned active force density for all, A, and B particles 
                                    int_fa_dens_time_theta_avg += (rad_step/2) * (single_time_dict['fa_dens']['all'][temp_id_new[j],:]+single_time_dict['fa_dens']['all'][temp_id_new[j-1],:])
                                    int_faA_dens_time_theta_avg += (rad_step/2) * (single_time_dict['fa_dens']['A'][temp_id_new[j],:]+single_time_dict['fa_dens']['A'][temp_id_new[j-1],:])
                                    int_faB_dens_time_theta_avg += (rad_step/2) * (single_time_dict['fa_dens']['B'][temp_id_new[j],:]+single_time_dict['fa_dens']['B'][temp_id_new[j-1],:])

                                    # Radially integrate via trapezoidal rule the average active force magnitude for all, A, and B particles 
                                    int_fa_avg_real_time_theta_avg += (rad_step/2) * (single_time_dict['fa_avg_real']['all'][temp_id_new[j],:]+single_time_dict['fa_avg_real']['all'][temp_id_new[j-1],:])

                                    # Radially integrate via trapezoidal rule the average aligned active force magnitude for all, A, and B particles 
                                    int_fa_avg_time_theta_avg += (rad_step/2) * (single_time_dict['fa_avg']['all'][temp_id_new[j],:]+single_time_dict['fa_avg']['all'][temp_id_new[j-1],:])
                                    int_faA_avg_time_theta_avg += (rad_step/2) * (single_time_dict['fa_avg']['A'][temp_id_new[j],:]+single_time_dict['fa_avg']['A'][temp_id_new[j-1],:])
                                    int_faB_avg_time_theta_avg += (rad_step/2) * (single_time_dict['fa_avg']['B'][temp_id_new[j],:]+single_time_dict['fa_avg']['B'][temp_id_new[j-1],:])

                                    # Radially integrate via trapezoidal rule the total aligned active force magnitude for all, A, and B particles 
                                    int_fa_sum_time_theta_avg += (rad_step/2) * (single_time_dict['fa_sum']['all'][temp_id_new[j],:]+single_time_dict['fa_sum']['all'][temp_id_new[j-1],:])
                                    int_faA_sum_time_theta_avg += (rad_step/2) * (single_time_dict['fa_sum']['A'][temp_id_new[j],:]+single_time_dict['fa_sum']['A'][temp_id_new[j-1],:])
                                    int_faB_sum_time_theta_avg += (rad_step/2) * (single_time_dict['fa_sum']['B'][temp_id_new[j],:]+single_time_dict['fa_sum']['B'][temp_id_new[j-1],:])

                                    # Radially integrate via trapezoidal rule the number density for all, A, and B particles 
                                    int_num_dens_time_theta_avg += (rad_step/2) * (single_time_dict['num_dens']['all'][temp_id_new[j],:]+single_time_dict['num_dens']['all'][temp_id_new[j-1],:])
                                    int_num_densA_time_theta_avg += (rad_step/2) * (single_time_dict['num_dens']['A'][temp_id_new[j],:]+single_time_dict['num_dens']['A'][temp_id_new[j-1],:])
                                    int_num_densB_time_theta_avg += (rad_step/2) * (single_time_dict['num_dens']['B'][temp_id_new[j],:]+single_time_dict['num_dens']['B'][temp_id_new[j-1],:])

                                    int_part_fracA_time_theta_avg += (rad_step/2) * (single_time_dict['part_frac']['A'][temp_id_new[j],:]+single_time_dict['part_frac']['A'][temp_id_new[j-1],:])
                                    int_part_fracB_time_theta_avg += (rad_step/2) * (single_time_dict['part_frac']['B'][temp_id_new[j],:]+single_time_dict['part_frac']['B'][temp_id_new[j-1],:])


                                    # Radially integrate via trapezoidal rule the average alignment for all, A, and B particles 
                                    int_align_time_theta_avg += (rad_step/2) * (single_time_dict['align']['all'][temp_id_new[j],:]+single_time_dict['align']['all'][temp_id_new[j-1],:])
                                    int_alignA_time_theta_avg += (rad_step/2) * (single_time_dict['align']['A'][temp_id_new[j],:]+single_time_dict['align']['A'][temp_id_new[j-1],:])
                                    int_alignB_time_theta_avg += (rad_step/2) * (single_time_dict['align']['B'][temp_id_new[j],:]+single_time_dict['align']['B'][temp_id_new[j-1],:])

                        except:
                            pass

                        # If average active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_real_r_int += int_fa_avg_real_time_theta_avg
                        except: 
                            sum_fa_avg_real_r_int = int_fa_avg_real_time_theta_avg

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_sum_r_int += int_fa_sum_time_theta_avg
                        except: 
                            sum_fa_sum_r_int = int_fa_sum_time_theta_avg

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_sum_r_int += int_faA_sum_time_theta_avg
                        except: 
                            sum_faA_sum_r_int = int_faA_sum_time_theta_avg

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_sum_r_int += int_faB_sum_time_theta_avg
                        except: 
                            sum_faB_sum_r_int = int_faB_sum_time_theta_avg

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_r_int += int_fa_avg_time_theta_avg
                        except: 
                            sum_fa_avg_r_int = int_fa_avg_time_theta_avg

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_avg_r_int += int_faA_avg_time_theta_avg
                        except: 
                            sum_faA_avg_r_int = int_faA_avg_time_theta_avg

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_avg_r_int += int_faB_avg_time_theta_avg
                        except: 
                            sum_faB_avg_r_int = int_faB_avg_time_theta_avg

                        # If average body force density calculated before, add to sum or else start sum
                        try:
                            sum_fa_dens_r_int += int_fa_dens_time_theta_avg
                        except: 
                            sum_fa_dens_r_int = int_fa_dens_time_theta_avg

                        # If average body force density for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_dens_r_int += int_faA_dens_time_theta_avg
                        except: 
                            sum_faA_dens_r_int = int_faA_dens_time_theta_avg

                        # If average body force density for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_dens_r_int += int_faB_dens_time_theta_avg
                        except: 
                            sum_faB_dens_r_int = int_faB_dens_time_theta_avg

                        # If average alignment calculated before, add to sum or else start sum
                        try:
                            sum_align_r_int += int_align_time_theta_avg
                        except: 
                            sum_align_r_int = int_align_time_theta_avg

                        # If average alignment for A particles calculated before, add to sum or else start sum
                        try:
                            sum_alignA_r_int += int_alignA_time_theta_avg
                        except: 
                            sum_alignA_r_int = int_alignA_time_theta_avg

                        # If average alignment for B particles calculated before, add to sum or else start sum
                        try:
                            sum_alignB_r_int += int_alignB_time_theta_avg
                        except: 
                            sum_alignB_r_int = int_alignB_time_theta_avg

                        # If average number density calculated before, add to sum or else start sum
                        try:
                            sum_num_dens_r_int += int_num_dens_time_theta_avg
                        except: 
                            sum_num_dens_r_int = int_num_dens_time_theta_avg

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densA_r_int += int_num_densA_time_theta_avg
                        except: 
                            sum_num_densA_r_int = int_num_densA_time_theta_avg

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densB_r_int += int_num_densB_time_theta_avg
                        except: 
                            sum_num_densB_r_int = int_num_densB_time_theta_avg

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_part_fracA_r_int += int_part_fracA_time_theta_avg
                        except: 
                            sum_part_fracA_r_int = int_part_fracA_time_theta_avg

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_part_fracB_r_int += int_part_fracB_time_theta_avg
                        except: 
                            sum_part_fracB_r_int = int_part_fracB_time_theta_avg



                        # If average active force magnitude calculated before, add to sum or else start sum
                        try:
                            integrated_sum_fa_avg_real_r += int_single_time_dict['fa_avg_real']['all']
                        except: 
                            integrated_sum_fa_avg_real_r = int_single_time_dict['fa_avg_real']['all']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            integrated_sum_fa_sum_r += int_single_time_dict['fa_sum']['all']
                        except: 
                            integrated_sum_fa_sum_r = int_single_time_dict['fa_sum']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            integrated_sum_faA_sum_r += int_single_time_dict['fa_sum']['A']
                        except: 
                            integrated_sum_faA_sum_r = int_single_time_dict['fa_sum']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_faB_sum_r += int_single_time_dict['fa_sum']['B']
                        except: 
                            integrated_sum_faB_sum_r = int_single_time_dict['fa_sum']['B']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            integrated_sum_fa_avg_r += int_single_time_dict['fa_avg']['all']
                        except: 
                            integrated_sum_fa_avg_r = int_single_time_dict['fa_avg']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            integrated_sum_faA_avg_r += int_single_time_dict['fa_avg']['A']
                        except: 
                            integrated_sum_faA_avg_r = int_single_time_dict['fa_avg']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_faB_avg_r += int_single_time_dict['fa_avg']['B']
                        except: 
                            integrated_sum_faB_avg_r = int_single_time_dict['fa_avg']['B']

                        # If average body force density calculated before, add to sum or else start sum
                        try:
                            integrated_sum_fa_dens_r += int_single_time_dict['fa_dens']['all']
                        except: 
                            integrated_sum_fa_dens_r = int_single_time_dict['fa_dens']['all']

                        # If average body force density for A particles calculated before, add to sum or else start sum
                        try:  
                            integrated_sum_faA_dens_r += int_single_time_dict['fa_dens']['A']
                        except: 
                            integrated_sum_faA_dens_r = int_single_time_dict['fa_dens']['A']

                        # If average body force density for B particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_faB_dens_r += int_single_time_dict['fa_dens']['B']
                        except: 
                            integrated_sum_faB_dens_r = int_single_time_dict['fa_dens']['B']

                        # If average alignment calculated before, add to sum or else start sum
                        try:
                            integrated_sum_align_r += int_single_time_dict['align']['all']
                        except: 
                            integrated_sum_align_r = int_single_time_dict['align']['all']

                        # If average alignment for A particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_alignA_r += int_single_time_dict['align']['A']
                        except: 
                            integrated_sum_alignA_r = int_single_time_dict['align']['A']

                        # If average alignment for B particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_alignB_r += int_single_time_dict['align']['B']
                        except: 
                            integrated_sum_alignB_r = int_single_time_dict['align']['B']

                        # If average number density calculated before, add to sum or else start sum
                        try:
                            integrated_sum_num_dens_r += int_single_time_dict['num_dens']['all']
                        except: 
                            integrated_sum_num_dens_r = int_single_time_dict['num_dens']['all']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_num_densA_r += int_single_time_dict['num_dens']['A']
                        except: 
                            integrated_sum_num_densA_r = int_single_time_dict['num_dens']['A']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_num_densB_r += int_single_time_dict['num_dens']['B']
                        except: 
                            integrated_sum_num_densB_r = int_single_time_dict['num_dens']['B']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_part_fracA_r += int_single_time_dict['part_frac']['A']
                        except: 
                            integrated_sum_part_fracA_r = int_single_time_dict['part_frac']['A']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            integrated_sum_part_fracB_r += int_single_time_dict['part_frac']['B']
                        except: 
                            integrated_sum_part_fracB_r = int_single_time_dict['part_frac']['B']

                        # If average cluster radius calculated before, add to sum or else start sum
                        try:
                            integrated_sum_rad += int_single_time_dict['radius']
                        except: 
                            integrated_sum_rad = int_single_time_dict['radius']





                        # If average active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_real_r += single_time_dict['fa_avg_real']['all']
                        except: 
                            sum_fa_avg_real_r = single_time_dict['fa_avg_real']['all']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_sum_r += single_time_dict['fa_sum']['all']
                        except: 
                            sum_fa_sum_r = single_time_dict['fa_sum']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_sum_r += single_time_dict['fa_sum']['A']
                        except: 
                            sum_faA_sum_r = single_time_dict['fa_sum']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_sum_r += single_time_dict['fa_sum']['B']
                        except: 
                            sum_faB_sum_r = single_time_dict['fa_sum']['B']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_r += single_time_dict['fa_avg']['all']
                        except: 
                            sum_fa_avg_r = single_time_dict['fa_avg']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_avg_r += single_time_dict['fa_avg']['A']
                        except: 
                            sum_faA_avg_r = single_time_dict['fa_avg']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_avg_r += single_time_dict['fa_avg']['B']
                        except: 
                            sum_faB_avg_r = single_time_dict['fa_avg']['B']

                        # If average body force density calculated before, add to sum or else start sum
                        try:
                            sum_fa_dens_r += single_time_dict['fa_dens']['all']
                        except: 
                            sum_fa_dens_r = single_time_dict['fa_dens']['all']

                        # If average body force density for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_dens_r += single_time_dict['fa_dens']['A']
                        except: 
                            sum_faA_dens_r = single_time_dict['fa_dens']['A']

                        # If average body force density for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_dens_r += single_time_dict['fa_dens']['B']
                        except: 
                            sum_faB_dens_r = single_time_dict['fa_dens']['B']

                        # If average alignment calculated before, add to sum or else start sum
                        try:
                            sum_align_r += single_time_dict['align']['all']
                        except: 
                            sum_align_r = single_time_dict['align']['all']

                        # If average alignment for A particles calculated before, add to sum or else start sum
                        try:
                            sum_alignA_r += single_time_dict['align']['A']
                        except: 
                            sum_alignA_r = single_time_dict['align']['A']

                        # If average alignment for B particles calculated before, add to sum or else start sum
                        try:
                            sum_alignB_r += single_time_dict['align']['B']
                        except: 
                            sum_alignB_r = single_time_dict['align']['B']

                        # If average number density calculated before, add to sum or else start sum
                        try:
                            sum_num_dens_r += single_time_dict['num_dens']['all']
                        except: 
                            sum_num_dens_r = single_time_dict['num_dens']['all']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densA_r += single_time_dict['num_dens']['A']
                        except: 
                            sum_num_densA_r = single_time_dict['num_dens']['A']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densB_r += single_time_dict['num_dens']['B']
                        except: 
                            sum_num_densB_r = single_time_dict['num_dens']['B']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_part_fracA_r += single_time_dict['part_frac']['A']
                        except: 
                            sum_part_fracA_r = single_time_dict['part_frac']['A']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_part_fracB_r += single_time_dict['part_frac']['B']
                        except: 
                            sum_part_fracB_r = single_time_dict['part_frac']['B']

                        # If average cluster radius calculated before, add to sum or else start sum
                        try:
                            sum_rad += single_time_dict['radius']
                        except: 
                            sum_rad = single_time_dict['radius']

                        # If average x-position of cluster's center of mass calculated before, add to sum or else start sum
                        try:
                            sum_com_x += single_time_dict['com']['x']
                        except: 
                            sum_com_x = single_time_dict['com']['x']

                        # If average y-position of cluster's center of mass calculated before, add to sum or else start sum 
                        try:
                            sum_com_y += single_time_dict['com']['y']
                        except:
                            sum_com_y = single_time_dict['com']['y']

                        # Number of time steps summed over
                        sum_num += 1
    
                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_avg_r_int = sum_fa_avg_r_int / sum_num
                avg_faA_avg_r_int = sum_faA_avg_r_int / sum_num
                avg_faB_avg_r_int = sum_faB_avg_r_int / sum_num

                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_sum_r_int = sum_fa_sum_r_int / sum_num
                avg_faA_sum_r_int = sum_faA_sum_r_int / sum_num
                avg_faB_sum_r_int = sum_faB_sum_r_int / sum_num

                # Calculate time-averaged active force magnitude averaged over all, A, and B particles
                avg_fa_avg_real_r_int = sum_fa_avg_real_r_int / sum_num

                # Calculate time-averaged body force density averaged over all, A, and B particles
                avg_fa_dens_r_int = sum_fa_dens_r_int / sum_num
                avg_faA_dens_r_int = sum_faA_dens_r_int / sum_num
                avg_faB_dens_r_int = sum_faB_dens_r_int / sum_num

                # Calculate time-averaged alignment averaged over all, A, and B particles
                avg_align_r_int = sum_align_r_int / sum_num
                avg_alignA_r_int = sum_alignA_r_int / sum_num
                avg_alignB_r_int = sum_alignB_r_int / sum_num

                # Calculate time-averaged number density for all, A, and B particles
                avg_num_dens_r_int = sum_num_dens_r_int / sum_num
                avg_num_densA_r_int = sum_num_densA_r_int / sum_num
                avg_num_densB_r_int = sum_num_densB_r_int / sum_num

                avg_part_fracA_r_int = sum_part_fracA_r_int / sum_num
                avg_part_fracB_r_int = sum_part_fracB_r_int / sum_num





                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_avg_r = sum_fa_avg_r / sum_num
                avg_faA_avg_r = sum_faA_avg_r / sum_num
                avg_faB_avg_r = sum_faB_avg_r / sum_num

                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_sum_r = sum_fa_sum_r / sum_num
                avg_faA_sum_r = sum_faA_sum_r / sum_num
                avg_faB_sum_r = sum_faB_sum_r / sum_num

                # Calculate time-averaged active force magnitude averaged over all, A, and B particles
                avg_fa_avg_real_r = sum_fa_avg_real_r / sum_num

                # Calculate time-averaged body force density averaged over all, A, and B particles
                avg_fa_dens_r = sum_fa_dens_r / sum_num
                avg_faA_dens_r = sum_faA_dens_r / sum_num
                avg_faB_dens_r = sum_faB_dens_r / sum_num

                # Calculate time-averaged alignment averaged over all, A, and B particles
                avg_align_r = sum_align_r / sum_num
                avg_alignA_r = sum_alignA_r / sum_num
                avg_alignB_r = sum_alignB_r / sum_num

                # Calculate time-averaged number density for all, A, and B particles
                avg_num_dens_r = sum_num_dens_r / sum_num
                avg_num_densA_r = sum_num_densA_r / sum_num
                avg_num_densB_r = sum_num_densB_r / sum_num

                avg_part_fracA_r = sum_part_fracA_r / sum_num
                avg_part_fracB_r = sum_part_fracB_r / sum_num

                # Calculate time-averaged radius and x- and y- position of cluster's center of mass
                avg_rad_val = sum_rad / sum_num
                avg_com_x = sum_com_x / sum_num
                avg_com_y = sum_com_y / sum_num




                
                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                integrated_avg_fa_avg_r = integrated_sum_fa_avg_r / sum_num
                integrated_avg_faA_avg_r = integrated_sum_faA_avg_r / sum_num
                integrated_avg_faB_avg_r = integrated_sum_faB_avg_r / sum_num

                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                integrated_avg_fa_sum_r = integrated_sum_fa_sum_r / sum_num
                integrated_avg_faA_sum_r = integrated_sum_faA_sum_r / sum_num
                integrated_avg_faB_sum_r = integrated_sum_faB_sum_r / sum_num

                # Calculate time-averaged active force magnitude averaged over all, A, and B particles
                integrated_avg_fa_avg_real_r = integrated_sum_fa_avg_real_r / sum_num

                # Calculate time-averaged body force density averaged over all, A, and B particles
                integrated_avg_fa_dens_r = integrated_sum_fa_dens_r / sum_num
                integrated_avg_faA_dens_r = integrated_sum_faA_dens_r / sum_num
                integrated_avg_faB_dens_r = integrated_sum_faB_dens_r / sum_num

                # Calculate time-averaged alignment averaged over all, A, and B particles
                integrated_avg_align_r = integrated_sum_align_r / sum_num
                integrated_avg_alignA_r = integrated_sum_alignA_r / sum_num
                integrated_avg_alignB_r = integrated_sum_alignB_r / sum_num

                # Calculate time-averaged number density for all, A, and B particles
                integrated_avg_num_dens_r = integrated_sum_num_dens_r / sum_num
                integrated_avg_num_densA_r = integrated_sum_num_densA_r / sum_num
                integrated_avg_num_densB_r = integrated_sum_num_densB_r / sum_num

                integrated_avg_part_fracA_r = integrated_sum_part_fracA_r / sum_num
                integrated_avg_part_fracB_r = integrated_sum_part_fracB_r / sum_num

                # Calculate time-averaged radius and x- and y- position of cluster's center of mass
                integrated_avg_rad_val = integrated_sum_rad / sum_num

                # Incorporate all time-averaged properties into a dictionary for saving
                avg_rad_dict_int = {'theta': unique_theta, 'radius': avg_rad_val, 'com': {'x': avg_com_x, 'y': avg_com_y}, 'fa_avg_real': {'all': avg_fa_avg_real_r_int}, 'fa_sum': {'all': avg_fa_sum_r_int, 'A': avg_faA_sum_r_int, 'B': avg_faB_sum_r_int}, 'fa_avg': {'all': avg_fa_avg_r_int, 'A': avg_faA_avg_r_int, 'B': avg_faB_avg_r_int}, 'fa_dens': {'all': avg_fa_dens_r_int, 'A': avg_faA_dens_r_int, 'B': avg_faB_dens_r_int}, 'align': {'all': avg_align_r_int, 'A': avg_alignA_r_int, 'B': avg_alignB_r_int}, 'num_dens': {'all': avg_num_dens_r_int, 'A': avg_num_densA_r_int, 'B': avg_num_densB_r_int}, 'part_frac': {'A': avg_part_fracA_r_int, 'B': avg_part_fracB_r_int}}
                
                # Incorporate all time-averaged properties into a dictionary for saving
                avg_rad_dict = {'rad': single_time_dict['rad'], 'theta': single_time_dict['theta'], 'radius': avg_rad_val, 'com': {'x': avg_com_x, 'y': avg_com_y}, 'fa_avg_real': {'all': avg_fa_avg_real_r}, 'fa_sum': {'all': avg_fa_sum_r, 'A': avg_faA_sum_r, 'B': avg_faB_sum_r}, 'fa_avg': {'all': avg_fa_avg_r, 'A': avg_faA_avg_r, 'B': avg_faB_avg_r}, 'fa_dens': {'all': avg_fa_dens_r, 'A': avg_faA_dens_r, 'B': avg_faB_dens_r}, 'align': {'all': avg_align_r, 'A': avg_alignA_r, 'B': avg_alignB_r}, 'num_dens': {'all': avg_num_dens_r, 'A': avg_num_densA_r, 'B': avg_num_densB_r}, 'part_frac': {'A': avg_part_fracA_r, 'B': avg_part_fracB_r}}
                
                # Incorporate all time-averaged properties into a dictionary for saving
                integrated_avg_rad_dict = {'theta': int_single_time_dict['theta'], 'radius': integrated_avg_rad_val, 'fa_avg_real': {'all': integrated_avg_fa_avg_real_r}, 'fa_sum': {'all': integrated_avg_fa_sum_r, 'A': integrated_avg_faA_sum_r, 'B': integrated_avg_faB_sum_r}, 'fa_avg': {'all': integrated_avg_fa_avg_r, 'A': integrated_avg_faA_avg_r, 'B': integrated_avg_faB_avg_r}, 'fa_dens': {'all': integrated_avg_fa_dens_r, 'A': integrated_avg_faA_dens_r, 'B': integrated_avg_faB_dens_r}, 'align': {'all': integrated_avg_align_r, 'A': integrated_avg_alignA_r, 'B': integrated_avg_alignB_r}, 'num_dens': {'all': integrated_avg_num_dens_r, 'A': integrated_avg_num_densA_r, 'B': integrated_avg_num_densB_r}, 'part_frac': {'A': integrated_avg_part_fracA_r, 'B': integrated_avg_part_fracB_r}}
                
                # If plot defined, plot time-averaged properties
                if plot == 'y':
                    plotting_functs.plot_avg_radial_heterogeneity(avg_rad_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all')                

                # Save time-averaged properties (theta, r/r_c) to separate files   
                np.savetxt(averagesPath + "radial_avgs_fa_avg_int_" + outfile+ '.csv', avg_fa_avg_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_avg_int_" + outfile+ '.csv', avg_faA_avg_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_avg_int_" + outfile+ '.csv', avg_faB_avg_r_int, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_sum_int_" + outfile+ '.csv', avg_fa_sum_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_sum_int_" + outfile+ '.csv', avg_faA_sum_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_sum_int_" + outfile+ '.csv', avg_faB_sum_r_int, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_avg_real_int_" + outfile+ '.csv', avg_fa_avg_real_r_int, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_dens_int_" + outfile+ '.csv', avg_fa_dens_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_dens_int_" + outfile+ '.csv', avg_faA_dens_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_dens_int_" + outfile+ '.csv', avg_faB_dens_r_int, delimiter=",")
                
                np.savetxt(averagesPath + "radial_avgs_align_int_" + outfile+ '.csv', avg_align_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_alignA_int_" + outfile+ '.csv', avg_alignA_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_alignB_int_" + outfile+ '.csv', avg_alignB_r_int, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_num_dens_int_" + outfile+ '.csv', avg_num_dens_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_num_densA_int_" + outfile+ '.csv', avg_num_densA_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_num_densB_int_" + outfile+ '.csv', avg_num_densB_r_int, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_part_fracA_int_" + outfile+ '.csv', avg_part_fracA_r_int, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_part_fracB_int_" + outfile+ '.csv', avg_part_fracB_r_int, delimiter=",")


                # Save time-averaged properties (theta, r/r_c) to separate files   
                np.savetxt(averagesPath + "radial_avgs_fa_avg_" + outfile+ '.csv', avg_fa_avg_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_avg_" + outfile+ '.csv', avg_faA_avg_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_avg_" + outfile+ '.csv', avg_faB_avg_r, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_sum_" + outfile+ '.csv', avg_fa_sum_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_sum_" + outfile+ '.csv', avg_faA_sum_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_sum_" + outfile+ '.csv', avg_faB_sum_r, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_avg_real_" + outfile+ '.csv', avg_fa_avg_real_r, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_fa_dens_" + outfile+ '.csv', avg_fa_dens_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faA_dens_" + outfile+ '.csv', avg_faA_dens_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_faB_dens_" + outfile+ '.csv', avg_faB_dens_r, delimiter=",")
                
                np.savetxt(averagesPath + "radial_avgs_align_" + outfile+ '.csv', avg_align_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_alignA_" + outfile+ '.csv', avg_alignA_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_alignB_" + outfile+ '.csv', avg_alignB_r, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_num_dens_" + outfile+ '.csv', avg_num_dens_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_num_densA_" + outfile+ '.csv', avg_num_densA_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_num_densB_" + outfile+ '.csv', avg_num_densB_r, delimiter=",")

                np.savetxt(averagesPath + "radial_avgs_part_fracA_" + outfile+ '.csv', avg_part_fracA_r, delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_part_fracB_" + outfile+ '.csv', avg_part_fracB_r, delimiter=",")
                

                # Save time-averaged properties (theta, r/r_c) to separate files   
                np.savetxt(averagesPath + "integrated_radial_avgs_fa_avg_" + outfile+ '.csv', integrated_avg_fa_avg_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faA_avg_" + outfile+ '.csv', integrated_avg_faA_avg_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faB_avg_" + outfile+ '.csv', integrated_avg_faB_avg_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_fa_sum_" + outfile+ '.csv', integrated_avg_fa_sum_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faA_sum_" + outfile+ '.csv', integrated_avg_faA_sum_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faB_sum_" + outfile+ '.csv', integrated_avg_faB_sum_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_fa_avg_real_" + outfile+ '.csv', integrated_avg_fa_avg_real_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_fa_dens_" + outfile+ '.csv', integrated_avg_fa_dens_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faA_dens_" + outfile+ '.csv', integrated_avg_faA_dens_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_faB_dens_" + outfile+ '.csv', integrated_avg_faB_dens_r, delimiter=",")
                
                np.savetxt(averagesPath + "integrated_radial_avgs_align_" + outfile+ '.csv', integrated_avg_align_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_alignA_" + outfile+ '.csv', integrated_avg_alignA_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_alignB_" + outfile+ '.csv', integrated_avg_alignB_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_num_dens_" + outfile+ '.csv', integrated_avg_num_dens_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_num_densA_" + outfile+ '.csv', integrated_avg_num_densA_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_num_densB_" + outfile+ '.csv', integrated_avg_num_densB_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_part_fracA_" + outfile+ '.csv', integrated_avg_part_fracA_r, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_part_fracB_" + outfile+ '.csv', integrated_avg_part_fracB_r, delimiter=",")

                np.savetxt(averagesPath + "integrated_radial_avgs_radius_" + outfile+ '.csv', integrated_avg_rad_val, delimiter=",")


                # Save non-position dependent time-averaged properties to combined file
                field_names = ['com_x', 'com_y', 'radius']
                indiv_vals = {'com_x': avg_com_x, 'com_y': avg_com_y, 'radius': avg_rad_val}
                
                with open(averagesPath + "radial_avgs_indiv_vals_" + outfile+ '.csv', 'w') as csvfile:  
                    writer = csv.writer(csvfile)
                    for key, value in indiv_vals.items():
                        writer.writerow([key, value])

                # Save position bins (theta, r/r_c) to separate files
                np.savetxt(averagesPath + "radial_avgs_rad_" + outfile+ '.csv', single_time_dict['rad'], delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_theta_" + outfile+ '.csv', single_time_dict['theta'], delimiter=",")
                np.savetxt(averagesPath + "radial_avgs_theta_int_" + outfile+ '.csv', unique_theta, delimiter=",")
                np.savetxt(averagesPath + "integrated_radial_avgs_theta_" + outfile+ '.csv', int_single_time_dict['theta'], delimiter=",")


                

                # if calculated this run, say we don't need to load the files in
                load_save = 0
            else:
                
                # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "integrated_radial_avgs_fa_avg_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_fa_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faA_avg_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faA_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faB_avg_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faB_avg_r = np.array(list(csv.reader(csvfile)))

                    # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "integrated_radial_avgs_fa_sum_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_fa_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faA_sum_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faA_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faB_sum_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faB_sum_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged active force magnitude array for all particles
                with open(averagesPath + "integrated_radial_avgs_fa_avg_real_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_fa_avg_real_r = np.array(list(csv.reader(csvfile)))
                
                # Load in time-averaged body force density array for all, A, and B particles
                with open(averagesPath + "integrated_radial_avgs_fa_dens_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_fa_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faA_dens_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faA_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_faB_dens_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_faB_dens_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged alignment array for all, A, and B particles
                with open(averagesPath + "integrated_radial_avgs_align_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_align_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_alignA_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_alignA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_alignB_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_alignB_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged numbder density array for all, A, and B particles
                with open(averagesPath + "integrated_radial_avgs_num_dens_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_num_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_num_densA_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_num_densA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_num_densB_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_num_densB_r = np.array(list(csv.reader(csvfile)))

                with open(averagesPath + "integrated_radial_avgs_part_fracA_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_part_fracA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_part_fracB_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_part_fracB_r = np.array(list(csv.reader(csvfile)))





                # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "radial_avgs_fa_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faA_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faB_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_avg_r = np.array(list(csv.reader(csvfile)))

                    # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "radial_avgs_fa_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faA_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faB_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_sum_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged active force magnitude array for all particles
                with open(averagesPath + "radial_avgs_fa_avg_real_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_avg_real_r = np.array(list(csv.reader(csvfile)))
                
                # Load in time-averaged body force density array for all, A, and B particles
                with open(averagesPath + "radial_avgs_fa_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faA_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_faB_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_dens_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged alignment array for all, A, and B particles
                with open(averagesPath + "radial_avgs_align_" + outfile+ '.csv', newline='') as csvfile:
                    avg_align_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_alignA_" + outfile+ '.csv', newline='') as csvfile:
                    avg_alignA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_alignB_" + outfile+ '.csv', newline='') as csvfile:
                    avg_alignB_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged numbder density array for all, A, and B particles
                with open(averagesPath + "radial_avgs_num_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_num_densA_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_densA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_num_densB_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_densB_r = np.array(list(csv.reader(csvfile)))

                with open(averagesPath + "radial_avgs_part_fracA_" + outfile+ '.csv', newline='') as csvfile:
                    avg_part_fracA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_part_fracB_" + outfile+ '.csv', newline='') as csvfile:
                    avg_part_fracB_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged cluster radius and x- and y-positions of cluster's center of mass
                with open(averagesPath + "radial_avgs_indiv_vals_" + outfile+ '.csv', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    avg_indiv_vals = dict(reader)
                avg_rad_val = avg_indiv_vals['radius']
                avg_com_x = avg_indiv_vals['com_x']
                avg_com_y = avg_indiv_vals['com_y']
                
                # Load position bins (theta, r/r_c) 
                with open(averagesPath + "radial_avgs_rad_" + outfile+ '.csv', newline='') as csvfile:
                    avg_rad = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "radial_avgs_theta_" + outfile+ '.csv', newline='') as csvfile:
                    avg_theta = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_theta_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_theta = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "integrated_radial_avgs_radius_" + outfile+ '.csv', newline='') as csvfile:
                    integrated_avg_rad_val = np.array(list(csv.reader(csvfile)))

                
                #Define dictionary for time-averaged properties
                avg_rad_dict = {'rad': avg_rad, 'theta': avg_theta, 'radius': avg_rad_val, 'com': {'x': avg_com_x, 'y': avg_com_y}, 'fa_avg_real': {'all': avg_fa_avg_real_r}, 'fa_sum': {'all': avg_fa_sum_r, 'A': avg_faA_sum_r, 'B': avg_faB_sum_r}, 'fa_avg': {'all': avg_fa_avg_r, 'A': avg_faA_avg_r, 'B': avg_faB_avg_r}, 'fa_dens': {'all': avg_fa_dens_r, 'A': avg_faA_dens_r, 'B': avg_faB_dens_r}, 'align': {'all': avg_align_r, 'A': avg_alignA_r, 'B': avg_alignB_r}, 'num_dens': {'all': avg_num_dens_r, 'A': avg_num_densA_r, 'B': avg_num_densB_r}, 'part_frac': {'A': avg_part_fracA_r, 'B': avg_part_fracB_r}}
            
                # Incorporate all time-averaged properties into a dictionary for saving
                integrated_avg_rad_dict = {'theta': integrated_avg_theta, 'radius': integrated_avg_rad_val, 'fa_avg_real': {'all': integrated_avg_fa_avg_real_r}, 'fa_sum': {'all': integrated_avg_fa_sum_r, 'A': integrated_avg_faA_sum_r, 'B': integrated_avg_faB_sum_r}, 'fa_avg': {'all': integrated_avg_fa_avg_r, 'A': integrated_avg_faA_avg_r, 'B': integrated_avg_faB_avg_r}, 'fa_dens': {'all': integrated_avg_fa_dens_r, 'A': integrated_avg_faA_dens_r, 'B': integrated_avg_faB_dens_r}, 'align': {'all': integrated_avg_align_r, 'A': integrated_avg_alignA_r, 'B': integrated_avg_alignB_r}, 'num_dens': {'all': integrated_avg_num_dens_r, 'A': integrated_avg_num_densA_r, 'B': integrated_avg_num_densB_r}, 'part_frac': {'A': integrated_avg_part_fracA_r, 'B': integrated_avg_part_fracB_r}}
                
                # Say we loaded these from files
                load_save = 1
        elif measurement_method == 'bulk-heterogeneity':
            
            outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
            
            # If time-averaged fluctuations file does not exist, calculate it...
            if os.path.isfile(averagesPath + "bulk_avgs_fa_avg_" + outfile+ '.csv')==0:
                sum_num = 0
                
                # Time range to calculate time-average at steady state
                end_avg = int(dumps/time_step)-1
                start_avg = int(end_avg/3)

                start_avg = 500
                end_avg = 501
                # Loop over time
                for p in range(start_avg, end_avg):

                    # Current time step
                    j=int(p*time_step)

                    print('j')
                    print(j)
                    
                    snap = t[j]                                 #Take current frame

                    #Arrays of particle data
                    pos = snap.particles.position               # current positions
                    pos[:,-1] = 0.0                             # 2D system
                    xy = np.delete(pos, 2, 1)

                    ori = snap.particles.orientation            #current orientation (quaternions)
                    ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]
                    x_orient_arr = np.array(list(map(utility_functs.quatToXOrient, ori))) # convert to x unit vector orientation
                    y_orient_arr = np.array(list(map(utility_functs.quatToYOrient, ori))) # convert to y unit vector orientation
                    
                    typ = snap.particles.typeid                 # Particle type
                    typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
                    typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

                    tst = snap.configuration.step               # timestep
                    tst -= first_tstep                          # normalize by first timestep
                    tst *= dtau                                 # convert to Brownian time
                    time_arr[j]=tst

                    #Compute cluster parameters using neighbor list of all particles within LJ cut-off distance
                    system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
                    cl_all=freud.cluster.Cluster() 

                    cl_all.compute(system_all, neighbors={'r_max': 1.3})        # Calculate clusters given neighbor list, positions,
                                                                            # and maximal radial interaction distance
                    clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
                    ids = cl_all.cluster_idx                                    # get id of each cluster
                    clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
                    clust_size = clp_all.sizes                                  # find cluster sizes

                    min_size=int(partNum/10)                                     #Minimum cluster size for measurements to happen
                    lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
                    large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
                    clust_large = np.amax(clust_size)
                    

                    # Instantiate particle properties module
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                    # If elongated simulation box...
                    if lx_box != ly_box:

                        clust_large = 0
                    
                    # Instantiate empty phase identification arrays
                    partTyp=np.zeros(partNum)
                    partPhase=np.zeros(partNum)
                    edgePhase=np.zeros(partNum)
                    bulkPhase=np.zeros(partNum)

                    # Calculate cluster CoM
                    com_dict = plotting_utility_functs.com_view(pos, clp_all)
                    
                    # If CoM option given, convert to CoM view
                    if com_option == True:
                        pos = com_dict['pos']
                    #else:
                    #    pos[:,0] = pos[:,0]
                    #    out = np.where(pos[:,0]<-hx_box)[0]
                    #    pos[out,0] = pos[out,0] + lx_box


                    #Bin system to calculate orientation and alignment that will be used in vector plots
                    NBins_x = utility_functs.getNBins(lx_box, bin_width)
                    NBins_y = utility_functs.getNBins(ly_box, bin_width)

                    # Calculate size of bins
                    sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
                    sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

                    # Instantiate binning functions module
                    binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)
                    
                    # Calculate bin positions
                    pos_dict = binning_functs.create_bins()

                    # Assign particles to bins
                    part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                    # Calculate average orientation per bin
                    orient_dict = binning_functs.bin_orient(part_dict, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                    # Calculate area fraction per bin
                    area_frac_dict = binning_functs.bin_area_frac(part_dict)

                    # Calculate average activity per bin
                    activ_dict = binning_functs.bin_activity(part_dict)

                    # Define output file name
                    outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
                    out = outfile + "_frame_"
                    pad = str(j).zfill(5)
                    outFile = out + pad

                    

                    # If cluster sufficiently large
                    if clust_large >= min_size:
                        
                        # Instantiate empty binning arrays
                        clust_size_arr = np.append(clust_size_arr, clust_large)
                        fa_all_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_x_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_y_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        fa_all_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        # Bin average alignment toward cluster's CoM
                        align_dict = binning_functs.bin_align(orient_dict)

                        

                        #Time frame for plots
                        pad = str(j).zfill(5)

                        # Bin average aligned active force pressure
                        press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)

                        # Bin average active force normal to cluster CoM
                        normal_fa_dict = binning_functs.bin_normal_active_fa(align_dict, area_frac_dict, activ_dict)

                        # Find curl and divergence of binned average alignment toward cluster CoM
                        align_grad_dict = binning_functs.curl_and_div(align_dict)

                        # Instantiate plotting functions module
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)

                        # Instantiate phase identification functions module
                        phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ)

                        # Identify phases of system
                        phase_dict = phase_ident_functs.phase_ident()
                        #phase_dict = phase_ident_functs.phase_ident_planar()

                        # Find IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        # Blur phases to mitigate noise
                        phase_dict = phase_ident_functs.phase_blur(phase_dict)

                        # Update phases of each particle ID
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Find updated IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        # Count number of particles per phase
                        count_dict = particle_prop_functs.phase_count(phase_dict)

                        # Find CoM of bulk phase
                        bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

                        # Separate non-connecting bulk phases
                        bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

                        # Separate non-connecting interfaces
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

                        # Update phase identification array
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Reduce mis-identification of gas
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

                        # Find interface composition
                        phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

                        # Find bulk composition
                        bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

                        # Sort bulk by largest to smallest
                        bulk_comp_dict = phase_ident_functs.phase_sort(bulk_comp_dict)

                        # Sort interface by largest to smallest
                        int_comp_dict = phase_ident_functs.phase_sort(int_comp_dict)

                        # Instantiate interface functions module
                        interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, x_orient_arr, y_orient_arr, pos)
                        
                        # Identify interior and exterior surface bin points
                        surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

                        #planar_surface_dict = interface_functs.det_planar_surface_points(phase_dict, int_dict, int_comp_dict)
                        
                        #Save positions of external and internal edges
                        clust_true = 0

                        # Sort surface points for both interior and exterior surfaces of each interface
                        surface2_pos_dict = interface_functs.surface_sort(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'])
                        surface1_pos_dict = interface_functs.surface_sort(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'])

                        # Find CoM of each surface
                        surface_com_dict = interface_functs.surface_com(int_dict, int_comp_dict, surface_dict)

                        # Find radius of surface
                        surface_radius_bin = interface_functs.surface_radius_bins(int_dict, int_comp_dict, surface_dict, surface_com_dict)

                        # Count bins per phase
                        bin_count_dict = phase_ident_functs.phase_bin_count(phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict)

                        # Bin average active force
                        active_fa_dict = binning_functs.bin_active_fa(orient_dict, part_dict, phase_dict['bin'])


                        # Calculate larger bins for orientation visualizations (note this is only used for visualizations)
                        bin_width2 = 15

                        #Bin system to calculate orientation and alignment that will be used in vector plots
                        NBins_x2 = utility_functs.getNBins(lx_box, bin_width2)
                        NBins_y2 = utility_functs.getNBins(ly_box, bin_width2)

                        # Calculate size of bins
                        sizeBin_x2 = utility_functs.roundUp(((lx_box) / NBins_x2), 6)
                        sizeBin_y2 = utility_functs.roundUp(((ly_box) / NBins_y2), 6)

                        # Instantiate binning functions module
                        binning_functs2 = binning.binning(lx_box, ly_box, partNum, NBins_x2, NBins_y2, peA, peB, typ, eps)
                            
                        # Calculate bin positions
                        pos_dict2 = binning_functs2.create_bins()

                        # Assign particles to bins
                        part_dict2 = binning_functs2.bin_parts(pos, ids, clust_size)

                        # Calculate average orientation per bin
                        orient_dict2 = binning_functs2.bin_orient(part_dict2, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                        #Slow/fast composition of bulk phase
                        part_count_dict, part_id_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)

                        # Instantiate data output functions module
                        data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)
                        
                        #Instantiate dictionaries to save data to
                        all_surface_curves = {}
                        all_surface_measurements = {}
                        
                        # Separate individual interfaces/surfaces
                        sep_surface_dict = interface_functs.separate_surfaces(surface_dict, int_dict, int_comp_dict)

                        # Loop over every surface
                        for m in range(0, len(sep_surface_dict)):

                            # Instantiate dictionaries to save data to
                            averaged_data_arr = {}

                            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                            
                            all_surface_curves[key] = {}
                            all_surface_measurements[key] = {}
                            
                            # Save composition data of interface
                            if (int_comp_dict['ids'][m]!=999):
                                averaged_data_arr['int_id'] = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])
                                averaged_data_arr['bub_id'] = int(int_comp_dict['ids'][m])
                                averaged_data_arr['Na'] = int(int_comp_dict['comp']['A'][m])
                                averaged_data_arr['Nb'] = int(int_comp_dict['comp']['B'][m])
                                averaged_data_arr['Nbin'] = int(bin_count_dict['ids']['int'][m])

                            # If sufficient interior interface points, take measurements
                            if sep_surface_dict[key]['interior']['num']>0:

                                # Sort surface points to curve
                                sort_interior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['interior'])

                                # Prepare surface curve for interpolation
                                sort_interior_ids = interface_functs.surface_curve_prep(sort_interior_ids, int_type = 'interior')

                                # Interpolate surface curve
                                all_surface_curves[key]['interior'] = interface_functs.surface_curve_interp(sort_interior_ids)

                                # Find surface curve CoM
                                com_pov_interior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['interior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['interior'] = interface_functs.surface_radius(com_pov_interior_pos['pos'])

                                # Measure surface curve area
                                all_surface_measurements[key]['interior']['surface area'] = interface_functs.surface_area(com_pov_interior_pos['pos'])

                                # Save surface curve CoM 
                                all_surface_measurements[key]['interior']['com'] = com_pov_interior_pos['com']

                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['interior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['interior'])

                                # Save radial measurements
                                averaged_data_arr['int_mean_rad'] = all_surface_measurements[key]['interior']['mean radius']
                                averaged_data_arr['int_std_rad'] = all_surface_measurements[key]['interior']['std radius']
                                averaged_data_arr['int_sa'] = all_surface_measurements[key]['interior']['surface area']

                            else:
                                averaged_data_arr['int_mean_rad'] = 0
                                averaged_data_arr['int_std_rad'] = 0
                                averaged_data_arr['int_sa'] = 0

                            # If sufficient exterior interface points, take measurements
                            if sep_surface_dict[key]['exterior']['num']>0:

                                # Sort surface points to curve
                                sort_exterior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['exterior'])

                                # Prepare surface curve for interpolation
                                sort_exterior_ids = interface_functs.surface_curve_prep(sort_exterior_ids, int_type = 'exterior')
                                
                                # Interpolate surface curve
                                all_surface_curves[key]['exterior'] = interface_functs.surface_curve_interp(sort_exterior_ids)

                                # Find surface curve CoM
                                com_pov_exterior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['exterior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['exterior'] = interface_functs.surface_radius(com_pov_exterior_pos['pos'])
                                
                                # Measure surface curve area
                                all_surface_measurements[key]['exterior']['surface area'] = interface_functs.surface_area(com_pov_exterior_pos['pos'])
                                
                                # Save surface curve CoM 
                                all_surface_measurements[key]['exterior']['com'] = com_pov_exterior_pos['com']
                                
                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['exterior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['exterior'])
                                
                                # Save radial measurements
                                averaged_data_arr['ext_mean_rad'] = all_surface_measurements[key]['exterior']['mean radius']
                                averaged_data_arr['ext_std_rad'] = all_surface_measurements[key]['exterior']['std radius']
                                averaged_data_arr['ext_sa'] = all_surface_measurements[key]['exterior']['surface area']
                            else:
                                averaged_data_arr['ext_mean_rad'] = 0
                                averaged_data_arr['ext_std_rad'] = 0
                                averaged_data_arr['ext_sa'] = 0

                            # If sufficient exterior and interior interface points, measure interface width
                            if (sep_surface_dict[key]['exterior']['num']>0) & sep_surface_dict[key]['interior']['num']>0:
                                all_surface_measurements[key]['exterior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                all_surface_measurements[key]['interior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                averaged_data_arr['width'] = all_surface_measurements[key]['exterior']['surface width']['width']
                            else:
                                averaged_data_arr['width'] = 0
                            
                            # If measurement method specified, save interface data
                            if measurement_method == 'interface-props':
                                data_output_functs.write_to_txt(averaged_data_arr, dataPath + 'BubComp_' + outfile + '.txt')
                        

                        
                        # If cluster has been initially formed, 
                        if steady_state_once == 'False':

                            # Instantiate array for saving largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_arr = np.zeros(partNum)
                            in_clust_arr[clust_id_time]=1

                            # Instantiate array for saving particle positions over time
                            pos_x_arr_time = pos[:,0]
                            pos_y_arr_time = pos[:,1]

                            # Instantiate array for saving surface CoM over time
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            # Instantiate array for saving cluster CoM over time
                            com_x_parts_arr_time = np.array([com_dict['com']['x']])
                            com_y_parts_arr_time = np.array([com_dict['com']['y']])

                            # Instantiate array for saving phase information over time
                            partPhase_time = phase_dict['part']

                            time_entered_bulk = np.ones(partNum) * tst
                            time_entered_gas = np.ones(partNum) * tst

                            # Instantiate array for saving time step
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Change to True since steady state has been reached
                            steady_state_once = 'True'

                            start_dict = {'bulk': {'time': time_entered_bulk}, 'gas': {'time': time_entered_gas} }
                            lifetime_dict = {}
                            msd_bulk_dict = {}
                            lifetime_stat_dict = {}

                        # If cluster has been formed previously
                        else:

                            # Save largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_temp = np.zeros(partNum)
                            in_clust_temp[clust_id_time]=1
                            in_clust_arr = np.vstack((in_clust_arr, in_clust_temp))

                            # Save particle positions over time
                            pos_x_arr_time = np.vstack((pos_x_arr_time, pos[:,0]))
                            pos_y_arr_time = np.vstack((pos_y_arr_time, pos[:,1]))
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Save phase information over time
                            partPhase_time = np.vstack((partPhase_time, phase_dict['part']))

                            # Save surface CoM over time
                            try:
                                com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box)
                                com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box)
                            except:
                                com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box)
                                com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box)

                            # Save cluster CoM over time
                            com_x_parts_arr_time = np.append(com_x_parts_arr_time, com_dict['com']['x'])
                            com_y_parts_arr_time = np.append(com_y_parts_arr_time, com_dict['com']['y'])

                        # Calculate alignment of interface with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
                        
                        # Calculate alignment of bulk with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)

                        # Calculate alignment of gas with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)

                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        try:
                            # Calculate average binned interface properties for current time frame
                            single_time_dict, plot_bin_dict = particle_prop_functs.bulk_heterogeneity_avgs(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict, phase_dict)
                        except:
                            pass
                        
                        # If average active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_real_r += single_time_dict['fa_avg_real']['all']
                        except: 
                            sum_fa_avg_real_r = single_time_dict['fa_avg_real']['all']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_sum_r += single_time_dict['fa_sum']['all']
                        except: 
                            sum_fa_sum_r = single_time_dict['fa_sum']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_sum_r += single_time_dict['fa_sum']['A']
                        except: 
                            sum_faA_sum_r = single_time_dict['fa_sum']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_sum_r += single_time_dict['fa_sum']['B']
                        except: 
                            sum_faB_sum_r = single_time_dict['fa_sum']['B']

                        # If average aligned active force magnitude calculated before, add to sum or else start sum
                        try:
                            sum_fa_avg_r += single_time_dict['fa_avg']['all']
                        except: 
                            sum_fa_avg_r = single_time_dict['fa_avg']['all']

                        # If average aligned active force magnitude for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_avg_r += single_time_dict['fa_avg']['A']
                        except: 
                            sum_faA_avg_r = single_time_dict['fa_avg']['A']

                        # If average aligned active force magnitude for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_avg_r += single_time_dict['fa_avg']['B']
                        except: 
                            sum_faB_avg_r = single_time_dict['fa_avg']['B']

                        # If average body force density calculated before, add to sum or else start sum
                        try:
                            sum_fa_dens_r += single_time_dict['fa_dens']['all']
                        except: 
                            sum_fa_dens_r = single_time_dict['fa_dens']['all']

                        # If average body force density for A particles calculated before, add to sum or else start sum
                        try:  
                            sum_faA_dens_r += single_time_dict['fa_dens']['A']
                        except: 
                            sum_faA_dens_r = single_time_dict['fa_dens']['A']

                        # If average body force density for B particles calculated before, add to sum or else start sum
                        try:
                            sum_faB_dens_r += single_time_dict['fa_dens']['B']
                        except: 
                            sum_faB_dens_r = single_time_dict['fa_dens']['B']

                        # If average alignment calculated before, add to sum or else start sum
                        try:
                            sum_align_r += single_time_dict['align']['all']
                        except: 
                            sum_align_r = single_time_dict['align']['all']

                        # If average alignment for A particles calculated before, add to sum or else start sum
                        try:
                            sum_alignA_r += single_time_dict['align']['A']
                        except: 
                            sum_alignA_r = single_time_dict['align']['A']

                        # If average alignment for B particles calculated before, add to sum or else start sum
                        try:
                            sum_alignB_r += single_time_dict['align']['B']
                        except: 
                            sum_alignB_r = single_time_dict['align']['B']

                        # If average number density calculated before, add to sum or else start sum
                        try:
                            sum_num_dens_r += single_time_dict['num_dens']['all']
                        except: 
                            sum_num_dens_r = single_time_dict['num_dens']['all']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densA_r += single_time_dict['num_dens']['A']
                        except: 
                            sum_num_densA_r = single_time_dict['num_dens']['A']

                        # If average number density for A particles calculated before, add to sum or else start sum
                        try:
                            sum_num_densB_r += single_time_dict['num_dens']['B']
                        except: 
                            sum_num_densB_r = single_time_dict['num_dens']['B']

                        # Number of time steps summed over
                        sum_num += 1
    
                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_avg_r = sum_fa_avg_r / sum_num
                avg_faA_avg_r = sum_faA_avg_r / sum_num
                avg_faB_avg_r = sum_faB_avg_r / sum_num

                # Calculate time-averaged aligned active force magnitude averaged over all, A, and B particles
                avg_fa_sum_r = sum_fa_sum_r / sum_num
                avg_faA_sum_r = sum_faA_sum_r / sum_num
                avg_faB_sum_r = sum_faB_sum_r / sum_num

                # Calculate time-averaged active force magnitude averaged over all, A, and B particles
                avg_fa_avg_real_r = sum_fa_avg_real_r / sum_num

                # Calculate time-averaged body force density averaged over all, A, and B particles
                avg_fa_dens_r = sum_fa_dens_r / sum_num
                avg_faA_dens_r = sum_faA_dens_r / sum_num
                avg_faB_dens_r = sum_faB_dens_r / sum_num

                # Calculate time-averaged alignment averaged over all, A, and B particles
                avg_align_r = sum_align_r / sum_num
                avg_alignA_r = sum_alignA_r / sum_num
                avg_alignB_r = sum_alignB_r / sum_num

                # Calculate time-averaged number density for all, A, and B particles
                avg_num_dens_r = sum_num_dens_r / sum_num
                avg_num_densA_r = sum_num_densA_r / sum_num
                avg_num_densB_r = sum_num_densB_r / sum_num

                # Incorporate all time-averaged properties into a dictionary for saving
                avg_rad_dict = {'x': single_time_dict['x'], 'y': single_time_dict['y'], 'fa_avg_real': {'all': avg_fa_avg_real_r}, 'fa_sum': {'all': avg_fa_sum_r, 'A': avg_faA_sum_r, 'B': avg_faB_sum_r}, 'fa_avg': {'all': avg_fa_avg_r, 'A': avg_faA_avg_r, 'B': avg_faB_avg_r}, 'fa_dens': {'all': avg_fa_dens_r, 'A': avg_faA_dens_r, 'B': avg_faB_dens_r}, 'align': {'all': avg_align_r, 'A': avg_alignA_r, 'B': avg_alignB_r}, 'num_dens': {'all': avg_num_dens_r, 'A': avg_num_densA_r, 'B': avg_num_densB_r}}

                # Save time-averaged properties (theta, r/r_c) to separate files   
                np.savetxt(averagesPath + "bulk_avgs_fa_avg_" + outfile+ '.csv', avg_fa_avg_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faA_avg_" + outfile+ '.csv', avg_faA_avg_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faB_avg_" + outfile+ '.csv', avg_faB_avg_r, delimiter=",")

                np.savetxt(averagesPath + "bulk_avgs_fa_sum_" + outfile+ '.csv', avg_fa_sum_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faA_sum_" + outfile+ '.csv', avg_faA_sum_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faB_sum_" + outfile+ '.csv', avg_faB_sum_r, delimiter=",")

                np.savetxt(averagesPath + "bulk_avgs_fa_avg_real_" + outfile+ '.csv', avg_fa_avg_real_r, delimiter=",")

                np.savetxt(averagesPath + "bulk_avgs_fa_dens_" + outfile+ '.csv', avg_fa_dens_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faA_dens_" + outfile+ '.csv', avg_faA_dens_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_faB_dens_" + outfile+ '.csv', avg_faB_dens_r, delimiter=",")
                
                np.savetxt(averagesPath + "bulk_avgs_align_" + outfile+ '.csv', avg_align_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_alignA_" + outfile+ '.csv', avg_alignA_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_alignB_" + outfile+ '.csv', avg_alignB_r, delimiter=",")

                np.savetxt(averagesPath + "bulk_avgs_num_dens_" + outfile+ '.csv', avg_num_dens_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_num_densA_" + outfile+ '.csv', avg_num_densA_r, delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_num_densB_" + outfile+ '.csv', avg_num_densB_r, delimiter=",")

                # Save position bins (theta, r/r_c) to separate files
                np.savetxt(averagesPath + "bulk_avgs_x_" + outfile+ '.csv', single_time_dict['x'], delimiter=",")
                np.savetxt(averagesPath + "bulk_avgs_y_" + outfile+ '.csv', single_time_dict['y'], delimiter=",")

                # if calculated this run, say we don't need to load the files in
                load_save = 0
            else:
                
                # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "bulk_avgs_fa_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faA_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_avg_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faB_avg_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_avg_r = np.array(list(csv.reader(csvfile)))

                    # Load in time-averaged aligned active force magnitude array for all, A, and B particles
                with open(averagesPath + "bulk_avgs_fa_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faA_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_sum_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faB_sum_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_sum_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged active force magnitude array for all particles
                with open(averagesPath + "bulk_avgs_fa_avg_real_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_avg_real_r = np.array(list(csv.reader(csvfile)))
                
                # Load in time-averaged body force density array for all, A, and B particles
                with open(averagesPath + "bulk_avgs_fa_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_fa_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faA_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faA_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_faB_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_faB_dens_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged alignment array for all, A, and B particles
                with open(averagesPath + "bulk_avgs_align_" + outfile+ '.csv', newline='') as csvfile:
                    avg_align_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_alignA_" + outfile+ '.csv', newline='') as csvfile:
                    avg_alignA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_alignB_" + outfile+ '.csv', newline='') as csvfile:
                    avg_alignB_r = np.array(list(csv.reader(csvfile)))

                # Load in time-averaged numbder density array for all, A, and B particles
                with open(averagesPath + "bulk_avgs_num_dens_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_dens_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_num_densA_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_densA_r = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_num_densB_" + outfile+ '.csv', newline='') as csvfile:
                    avg_num_densB_r = np.array(list(csv.reader(csvfile)))

                # Load position bins (theta, r/r_c) 
                with open(averagesPath + "bulk_avgs_x_" + outfile+ '.csv', newline='') as csvfile:
                    avg_x = np.array(list(csv.reader(csvfile)))
                with open(averagesPath + "bulk_avgs_y_" + outfile+ '.csv', newline='') as csvfile:
                    avg_y = np.array(list(csv.reader(csvfile)))

                #Define dictionary for time-averaged properties
                avg_rad_dict = {'x': avg_x, 'y': avg_y, 'fa_avg_real': {'all': avg_fa_avg_real_r}, 'fa_sum': {'all': avg_fa_sum_r, 'A': avg_faA_sum_r, 'B': avg_faB_sum_r}, 'fa_avg': {'all': avg_fa_avg_r, 'A': avg_faA_avg_r, 'B': avg_faB_avg_r}, 'fa_dens': {'all': avg_fa_dens_r, 'A': avg_faA_dens_r, 'B': avg_faB_dens_r}, 'align': {'all': avg_align_r, 'A': avg_alignA_r, 'B': avg_alignB_r}, 'num_dens': {'all': avg_num_dens_r, 'A': avg_num_densA_r, 'B': avg_num_densB_r}}
            
                # Say we loaded these from files
                load_save = 1

        # Loop over time for non-time averaged measurements
        for p in range(start, end):
            j=int(p*time_step)

            print('j')
            print(j)
            
            snap = t[j]                                 #Take current frame

            #Arrays of particle data
            pos = snap.particles.position               # current positions
            pos[:,-1] = 0.0                             # 2D system
            xy = np.delete(pos, 2, 1)

            ori = snap.particles.orientation            #current orientation (quaternions)
            ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]
            x_orient_arr = np.array(list(map(utility_functs.quatToXOrient, ori))) # convert to [-pi, pi]
            y_orient_arr = np.array(list(map(utility_functs.quatToYOrient, ori))) # convert to [-pi, pi]


            typ = snap.particles.typeid                 # Particle type
            typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
            typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

            print(x_orient_arr[typ0ind])
            print(y_orient_arr[typ0ind])

            tst = snap.configuration.step               # timestep
            tst -= first_tstep                          # normalize by first timestep
            tst *= dtau                                 # convert to Brownian time
            time_arr[j]=tst

            if measurement_method=='gsd-to-csv':
                utility_functs.gsd_to_csv(dataPath, outfile, pos, x_orient_arr, y_orient_arr, typ, tst, first_time=first_time)
                first_time = 0
            elif measurement_method == 'k-means':

                import tensorflow as tf

                def initializeCentroids(samples, numClusters, numSamples):

                    randomIndices     = tf.random.shuffle(tf.range(0, numSamples))

                    centroidIndices   = tf.slice(randomIndices, begin = [0, ], size = [numClusters, ])

                    initialCentroids  = tf.gather(samples, centroidIndices)
                    
                    return initialCentroids

                
                def assign2NearestCentroid(samples, centroids):

                    expandedSamples   = tf.expand_dims(samples, 0)

                    expandedCentroids = tf.expand_dims(centroids, 1)

                    distances         = tf.reduce_sum(tf.square(tf.subtract(expandedSamples, expandedCentroids)), 2)
                    print(np.shape(tf.subtract(expandedSamples, expandedCentroids)))
                    stop
                    print(np.shape(distances))
                    nearestIndices    = tf.argmin(distances, 0)
                    #for i in range(0, 2):
                    #    np.where(distances[:,i]>)
                        

                    return nearestIndices

                def updateCentroids(samples, nearestIndices, numClusters):
                    nearestIndices  = tf.cast(nearestIndices, tf.int32)

                    partitions      = tf.dynamic_partition(samples, nearestIndices, numClusters)

                    newCentroids    = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
                    
                    return newCentroids

                
                def plotClusters(samples, labels, centroids):

                    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

                    for i, centroid in enumerate(centroids):

                        plt.scatter(samples[:, 0], samples[:, 1], c = colour[labels], s=0.7)

                        plt.plot(centroids[i, 0], centroids[i, 1], markersize = 35, marker = "x", color = 'k', mew = 10)
                        plt.plot(centroids[i, 0], centroids[i, 1], markersize = 30, marker = "x", color = 'm', mew = 5)
                    plt.xlim([-hx_box, hx_box])
                    plt.ylim([-hy_box, hy_box])
                    plt.show()


                import sklearn.datasets as skd 

                numFeatures           = 2
                numClusters           = 2
                numSamples            = len(xy)
                numIter               = 200

                centroids = tf.cast([[3, 5], [5, 6]], tf.float64)

                centroids = tf.convert_to_tensor(centroids)
                centroids = tf.cast(centroids, tf.float32)

                samples   = tf.convert_to_tensor(xy)

                updateNorm = float('inf')

                oldCentroids = initializeCentroids(samples, numClusters, numSamples)
                oldCentroids = tf.cast(oldCentroids, tf.float32)

                func = np.empty((1, ), dtype = np.float32)
                func[0] = np.infty

                ind_clust = 0

                while (updateNorm > 1e-4):

                    nearestIndices = assign2NearestCentroid(samples, oldCentroids)
                    newCentroids   = updateCentroids(samples, nearestIndices, numClusters)
                    newCentroids = tf.cast(newCentroids, tf.float32)

                    updateNorm     = tf.sqrt(tf.reduce_sum(tf.square(newCentroids - oldCentroids)) / tf.reduce_sum(tf.square(centroids)))
                    
                    oldCentroids   = newCentroids
                    oldCentroids = tf.cast(oldCentroids, tf.float32)

                    if ind_clust % 50:
                        plotClusters(samples, nearestIndices, newCentroids)
                    
                    updateNorm = np.reshape(updateNorm.numpy(), (1, ))
                    
                    func = np.append(func, updateNorm, axis = 0)

                    ind_clust += 1

                plt.plot(func)
                plt.show()

            elif measurement_method=='phases-random':

                system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
                cl_all=freud.cluster.Cluster()                              #Define cluster
                
                cl_all.compute(system_all, neighbors={'r_max': 0.9})        # Calculate clusters given neighbor list, positions,
                                                                            # and maximal radial interaction distance
                clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
                ids = cl_all.cluster_idx                                    # get id of each cluster
                clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
                clust_size = clp_all.sizes                                  # find cluster sizes

                min_size=int(partNum/10)                                     #Minimum cluster size for measurements to happen
                lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
                clust_large = np.amax(clust_size)

                num_all = len(np.where(ids==lcID)[0])

                num_A = len(np.where((ids==lcID) & (typ==0))[0])
                num_B = len(np.where((ids==lcID) & (typ==1))[0])

                in_clust_dict = {'all': num_all, 'A': num_A, 'B': num_B}
                
                data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

            
                data_output_functs.write_to_txt(in_clust_dict, dataPath + 'in_clust_' + outfile + '.txt')



                # Instantiate particle properties module
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
            else:
                    
                #Compute cluster parameters using neighbor list of all particles within LJ cut-off distance
                system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
                cl_all=freud.cluster.Cluster()                              #Define cluster
                """
                if clustering_method == 'k-means':
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                    points_n = 100
                    clusters_n = 2
                    
                    points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))

                    #points = np.linspace(0, partNum, partNum)
                    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

                    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
                    assignments = tf.argmin(distances, 0)

                    means = []
                    for c in range(clusters_n):
                        means.append(tf.reduce_mean(tf.gather(points, tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),reduction_indices=[1]))
                        new_centroids = tf.concat(means, 0)
                        update_centroids = tf.assign(centroids, new_centroids)

                    with tf.Session() as sess:
                        sess.run(init)
                        for step in xrange(iteration_n):
                            [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

                    print("centroids", centroid_values)
                    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
                    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
                    plt.show()
                """
                cl_all.compute(system_all, neighbors={'r_max': 0.9})        # Calculate clusters given neighbor list, positions,
                                                                            # and maximal radial interaction distance
                clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
                ids = cl_all.cluster_idx                                    # get id of each cluster
                clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
                clust_size = clp_all.sizes                                  # find cluster sizes

                min_size=int(partNum/10)                                     #Minimum cluster size for measurements to happen
                lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
                large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
                clust_large = np.amax(clust_size)
                

                # Instantiate particle properties module
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                # If elongated simulation box...
                if lx_box != ly_box:

                    clust_large = 0

                    """
                    # Array of cluster sizes
                    clust_size = clp_all.sizes                                  # find cluster sizes

                    # Minimum cluster size to consider
                    min_size=int(partNum/8)                                     #Minimum cluster size for measurements to happen

                    # ID of largest cluster
                    lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster

                    # IDs of all clusters of sufficiently large size
                    large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size

                    #If at least one cluster is sufficiently large, determine CoM of largest cluster
                    if len(large_clust_ind_all[0])>0:

                        # Largest cluster's CoM position
                        query_points=clp_all.centers[lcID]
                        
                        # Largest cluster's CoM in natural box size of x=[-hx_box, hx_box] and y=[-hy_box, hy_box]
                        com_tmp_posX_temp = 70
                        com_tmp_posY_temp = 50

                    new_pos = pos.copy()
                    #shift reference frame positions such that CoM of largest cluster is at mid-point of simulation box
                    pos[:,0]= pos[:,0]+com_tmp_posX_temp
                    pos[:,1]= pos[:,1]-com_tmp_posY_temp

                    #Loop over all particles to ensure particles are within simulation box (periodic boundary conditions)
                    for i in range(0, partNum):
                        if pos[i,0]>hx_box:
                            pos[i,0]=pos[i,0]-lx_box
                        elif pos[i,0]<-hx_box:
                            pos[i,0]=pos[i,0]+lx_box

                        if pos[i,1]>hy_box:
                            pos[i,1]=pos[i,1]-ly_box
                        elif pos[i,1]<-hy_box:
                            pos[i,1]=pos[i,1]+ly_box
                    
                    #Compute cluster parameters using system_all neighbor list
                    system_all_temp = freud.AABBQuery(f_box, f_box.wrap(pos))
                    cl_all_temp=freud.cluster.Cluster()                              #Define cluster
                    cl_all_temp.compute(system_all_temp, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                                # and maximal radial interaction distance
                    clp_all_temp = freud.cluster.ClusterProperties()                 #Define cluster properties
                    ids_temp = cl_all_temp.cluster_idx                                    # get id of each cluster
                    clp_all_temp.compute(system_all_temp, ids_temp)                            # Calculate cluster properties given cluster IDs

                    clust_size_temp = clp_all_temp.sizes  
                    lcID_temp = np.where(clust_size_temp == np.amax(clust_size_temp))[0][0]    #Identify largest cluster

                    # Largest cluster's CoM position
                    query_points_temp=clp_all_temp.centers[lcID]
                    
                    com_opt = {'x': [169.9191, 161.86292, 157.76503, 136.27533, 108.46028], 'y': [165.99588, 159.7088, 152.05992, 144.84607, 130.15036]}
                    #com_opt = {'x': [query_points_temp[0] + hx_box], 'y': [query_points_temp[1] + hy_box]}

                    """
                # Instantiate empty phase identification arrays
                partTyp=np.zeros(partNum)
                partPhase=np.zeros(partNum)
                edgePhase=np.zeros(partNum)
                bulkPhase=np.zeros(partNum)

                if measurement_method=='spatial-heterogeneity':
                    
                    # Calculate cluster CoM
                    com_dict = plotting_utility_functs.com_view(pos, clp_all)
                    
                    # If CoM option given, convert to CoM view
                    if com_option == True:
                        pos = com_dict['pos']

                    NBins_x = utility_functs.getNBins(lx_box, bin_width)
                    NBins_y = utility_functs.getNBins(ly_box, bin_width)

                    # Calculate size of bins
                    sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
                    sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

                    # Instantiate binning functions module
                    binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)
                    
                    # Calculate bin positions
                    pos_dict = binning_functs.create_bins()

                    # Assign particles to bins
                    part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                    # Calculate average orientation per bin
                    orient_dict = binning_functs.bin_orient(part_dict, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                    # Calculate area fraction per bin
                    area_frac_dict = binning_functs.bin_area_frac(part_dict)

                    # Calculate average activity per bin
                    activ_dict = binning_functs.bin_activity(part_dict)

                    # Define output file name
                    outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
                    out = outfile + "_frame_"
                    pad = str(j).zfill(5)
                    outFile = out + pad

                    # If cluster sufficiently large
                    if clust_large >= min_size:
                        
                        # Instantiate empty binning arrays
                        clust_size_arr = np.append(clust_size_arr, clust_large)
                        fa_all_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_x_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_y_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        fa_all_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        # Bin average alignment toward cluster's CoM
                        align_dict = binning_functs.bin_align(orient_dict)

                        #Time frame for plots
                        pad = str(j).zfill(5)

                        # Bin average aligned active force pressure
                        press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)

                        # Bin average active force normal to cluster CoM
                        normal_fa_dict = binning_functs.bin_normal_active_fa(align_dict, area_frac_dict, activ_dict)

                        # Find curl and divergence of binned average alignment toward cluster CoM
                        align_grad_dict = binning_functs.curl_and_div(align_dict)

                        if measurement_method == 'random-forest':
                            from sklern.ensemble import RandomForestClassifier
                            from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
                            from sklearn.model_selection import RandomizedSearchCV, train_test_split
                            from scipy.stats import randint

                            dataset 


                        # Instantiate plotting functions module
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)

                        # Instantiate phase identification functions module
                        phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ)

                        # Identify phases of system
                        phase_dict = phase_ident_functs.phase_ident()
                        #phase_dict = phase_ident_functs.phase_ident_planar()

                        # Find IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        # Blur phases to mitigate noise
                        phase_dict = phase_ident_functs.phase_blur(phase_dict)

                        # Update phases of each particle ID
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Find IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        # Count number of particles per phase
                        count_dict = particle_prop_functs.phase_count(phase_dict)

                        # Find CoM of bulk phase
                        bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

                        # Separate non-connecting bulk phases
                        bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

                        # Separate non-connecting interfaces
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

                        # Update phase identification array
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Reduce mis-identification of gas
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

                        # Find interface composition
                        phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

                        # Find bulk composition
                        bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

                        # Sort bulk by largest to smallest
                        bulk_comp_dict = phase_ident_functs.phase_sort(bulk_comp_dict)

                        # Sort interface by largest to smallest
                        int_comp_dict = phase_ident_functs.phase_sort(int_comp_dict)

                        # Instantiate interface functions module
                        interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, x_orient_arr, y_orient_arr, pos)
                        
                        # Identify interior and exterior surface bin points
                        surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

                        #planar_surface_dict = interface_functs.det_planar_surface_points(phase_dict, int_dict, int_comp_dict)
                        
                        #Save positions of external and internal edges
                        clust_true = 0

                        # Bin average velocity, angular velocity, and velocity curl and divergence
                        if j>(start*time_step):
                            vel_dict = binning_functs.bin_vel(pos, prev_pos, part_dict, dt_step)

                            part_vel_dict, vel_phase_plot_dict = particle_prop_functs.velocity(vel_dict['part']['mag'], phase_dict['part'])

                            vel_plot_dict, vel_stat_dict = particle_prop_functs.part_velocity(prev_pos, prev_ang, ori)
                            
                            velocity_dict_avgs = {'all': {'bulk': part_vel_dict['bulk']['all']['avg'], 'int': part_vel_dict['int']['all']['avg'], 'gas': part_vel_dict['gas']['all']['avg']}, 'A': {'bulk': part_vel_dict['bulk']['A']['avg'], 'int': part_vel_dict['int']['A']['avg'], 'gas': part_vel_dict['gas']['A']['avg']}, 'B': {'bulk': part_vel_dict['bulk']['B']['avg'], 'int': part_vel_dict['int']['B']['avg'], 'gas': part_vel_dict['gas']['B']['avg']} }

                        # Sort surface points for both interior and exterior surfaces of each interface
                        surface2_pos_dict = interface_functs.surface_sort(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'])
                        surface1_pos_dict = interface_functs.surface_sort(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'])

                        # Find CoM of each surface
                        surface_com_dict = interface_functs.surface_com(int_dict, int_comp_dict, surface_dict)

                        # Find radius of surface
                        surface_radius_bin = interface_functs.surface_radius_bins(int_dict, int_comp_dict, surface_dict, surface_com_dict)

                        # Count bins per phase
                        bin_count_dict = phase_ident_functs.phase_bin_count(phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict)

                        # Bin average active force
                        active_fa_dict = binning_functs.bin_active_fa(orient_dict, part_dict, phase_dict['bin'])

                        #Slow/fast composition of bulk phase
                        part_count_dict, part_id_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)
                        
                        # Initialize lattice structure functions
                        lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                        # Calculate interparticle stresses and pressures
                        stress_stat_dict, press_stat_dict, press_stat_indiv_dict, press_plot_dict, stress_plot_dict, press_plot_indiv_dict, press_hetero_dict = stress_and_pressure_functs.interparticle_pressure_nlist_phases()
                        
                        bin_width_arr = np.linspace(1, 49, 49, dtype=float)
                        bin_width_arr = np.linspace(1, 20, 20, dtype=float)
                        #bin_width_arr2 = np.linspace(50, int(lx_box), int((lx_box-50)/10), dtype=float)
                        #bin_width_arr = np.concatenate((bin_width_arr, bin_width_arr2), axis=0)

                        # Heterogeneity
                        heterogeneity_typ_system = np.array([])

                        heterogeneity_activity_system = np.array([])

                        heterogeneity_typ_bulk = np.array([])
                        heterogeneity_typ_int = np.array([])
                        heterogeneity_typ_gas = np.array([])

                        heterogeneity_activity_bulk = np.array([])
                        heterogeneity_activity_int = np.array([])
                        heterogeneity_activity_gas = np.array([])

                        heterogeneity_area_frac_system_all = np.array([])
                        heterogeneity_area_frac_system_A = np.array([])
                        heterogeneity_area_frac_system_B = np.array([])

                        heterogeneity_area_frac_bulk_all = np.array([])
                        heterogeneity_area_frac_bulk_A = np.array([])
                        heterogeneity_area_frac_bulk_B = np.array([])

                        heterogeneity_area_frac_int_all = np.array([])
                        heterogeneity_area_frac_int_A = np.array([])
                        heterogeneity_area_frac_int_B = np.array([])

                        heterogeneity_area_frac_gas_all = np.array([])
                        heterogeneity_area_frac_gas_A = np.array([])
                        heterogeneity_area_frac_gas_B = np.array([])

                        heterogeneity_press_system_all = np.array([])
                        heterogeneity_press_system_A = np.array([])
                        heterogeneity_press_system_B = np.array([])

                        press_mean_dict = {'all': {'bulk': press_stat_dict['all-all']['bulk'], 'int': press_stat_dict['all-all']['int'], 'gas': press_stat_dict['all-all']['gas'] }, 'A': {'bulk': press_stat_dict['all-A']['bulk'], 'int': press_stat_dict['all-A']['int'], 'gas': press_stat_dict['all-A']['gas'] }, 'B': {'bulk': press_stat_dict['all-B']['bulk'], 'int': press_stat_dict['all-B']['int'], 'gas': press_stat_dict['all-B']['gas'] } }

                        heterogeneity_press_bulk_all = np.array([])
                        heterogeneity_press_bulk_A = np.array([])
                        heterogeneity_press_bulk_B = np.array([])

                        heterogeneity_press_int_all = np.array([])
                        heterogeneity_press_int_A = np.array([])
                        heterogeneity_press_int_B = np.array([])

                        heterogeneity_press_gas_all = np.array([])
                        heterogeneity_press_gas_A = np.array([])
                        heterogeneity_press_gas_B = np.array([])

                        heterogeneity_velocity_system_all = np.array([])
                        heterogeneity_velocity_system_A = np.array([])
                        heterogeneity_velocity_system_B = np.array([])

                        heterogeneity_velocity_bulk_all = np.array([])
                        heterogeneity_velocity_bulk_A = np.array([])
                        heterogeneity_velocity_bulk_B = np.array([])

                        heterogeneity_velocity_int_all = np.array([])
                        heterogeneity_velocity_int_A = np.array([])
                        heterogeneity_velocity_int_B = np.array([])

                        heterogeneity_velocity_gas_all = np.array([])
                        heterogeneity_velocity_gas_A = np.array([])
                        heterogeneity_velocity_gas_B = np.array([])

                        phi_dict = {'all': {'bulk': (part_count_dict['bulk']['all'] * (np.pi/4)) / (bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)), 'int': (part_count_dict['int']['all'] * (np.pi/4)) / (bin_count_dict['bin']['int'] * (sizeBin_x * sizeBin_y)), 'gas': (part_count_dict['gas']['all'] * (np.pi/4)) / (bin_count_dict['bin']['gas'] * (sizeBin_x * sizeBin_y)) }, 'A': {'bulk': (part_count_dict['bulk']['A'] * (np.pi/4)) / (bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)), 'int': (part_count_dict['int']['A'] * (np.pi/4)) / (bin_count_dict['bin']['int'] * (sizeBin_x * sizeBin_y)), 'gas': (part_count_dict['gas']['A'] * (np.pi/4)) / (bin_count_dict['bin']['gas'] * (sizeBin_x * sizeBin_y)) }, 'B': {'bulk': (part_count_dict['bulk']['B'] * (np.pi/4)) / (bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)), 'int': (part_count_dict['int']['B'] * (np.pi/4)) / (bin_count_dict['bin']['int'] * (sizeBin_x * sizeBin_y)), 'gas': (part_count_dict['gas']['B'] * (np.pi/4)) / (bin_count_dict['bin']['gas'] * (sizeBin_x * sizeBin_y)) } }
                        press_dict_avgs = {'all': {'bulk': press_stat_dict['all-all']['bulk']['press'], 'int': press_stat_dict['all-all']['int']['press'], 'gas': press_stat_dict['all-all']['gas']['press'] }, 'A': {'bulk': press_stat_dict['all-A']['bulk']['press'], 'int': press_stat_dict['all-A']['bulk']['press'], 'gas': press_stat_dict['all-A']['bulk']['press'] }, 'B': {'bulk': press_stat_dict['all-B']['bulk']['press'], 'int': press_stat_dict['all-B']['int']['press'], 'gas': press_stat_dict['all-B']['gas']['press'] } }
                        typ_dict = {'bulk': part_count_dict['bulk']['B']/part_count_dict['bulk']['all'], 'int': part_count_dict['int']['B']/part_count_dict['int']['all'], 'gas': part_count_dict['gas']['B']/part_count_dict['gas']['all']}
                        activity_dict = {'bulk': (part_count_dict['bulk']['A']/part_count_dict['bulk']['all']) * peA + (part_count_dict['bulk']['B']/part_count_dict['bulk']['all']) * peB, 'int': (part_count_dict['int']['A']/part_count_dict['int']['all'])*peA + (part_count_dict['int']['B']/part_count_dict['int']['all'])*peB, 'gas': (part_count_dict['gas']['A']/part_count_dict['gas']['all']) * peA + (part_count_dict['gas']['B']/part_count_dict['gas']['all']) * peB}
                        # I NEED TO MAKE A FUNCTION THAT IDENTIFIES PHASE OF BIN BASED ON NUMBER OF PARTICLES IN IT!
                        for q in range(0, len(bin_width_arr)):
                            """
                            #Bin system to calculate orientation and alignment that will be used in vector plots
                            NBins_x_tmp = utility_functs.getNBins(lx_box, bin_width_arr[q])
                            NBins_y_tmp = utility_functs.getNBins(ly_box, bin_width_arr[q])

                            # Calculate size of bins
                            sizeBin_x_tmp = utility_functs.roundUp(((lx_box) / NBins_x_tmp), 6)
                            sizeBin_y_tmp = utility_functs.roundUp(((ly_box) / NBins_y_tmp), 6)

                            # Instantiate binning functions module
                            binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x_tmp, NBins_y_tmp, peA, peB, typ, eps)
                            
                            # Calculate bin positions
                            pos_dict = binning_functs.create_bins()

                            # Assign particles to bins
                            part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                            heterogeneity_typ_all_tmp = binning_functs.bin_heterogeneity_binned_system(part_dict['typ_mean'], (1.0-parFrac/100))
                            id_heterogeneity_typ_system_all = np.append(id_heterogeneity_typ_system_all, heterogeneity_typ_all_tmp)

                            heterogeneity_activity_all_tmp = binning_functs.bin_heterogeneity_binned_system(part_dict['act_mean'], peA * (parFrac/100) + peB * (1.0-parFrac/100))
                            id_heterogeneity_activity_system_all = np.append(id_heterogeneity_activity_system_all, heterogeneity_activity_all_tmp)

                                # Calculate area fraction per bin
                            area_frac_dict = binning_functs.bin_area_frac(part_dict)

                            # Heterogeneity
                            heterogeneity_area_frac_all = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['all'], (intPhi/100))
                            heterogeneity_area_frac_A = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['A'], (intPhi/100)*parFrac/100)
                            heterogeneity_area_frac_B = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['B'], (intPhi/100)*(1.0-parFrac/100))

                            heterogeneity_area_frac_system_all = np.append(heterogeneity_area_frac_system_all, heterogeneity_area_frac_all)
                            heterogeneity_area_frac_system_A = np.append(heterogeneity_area_frac_system_A, heterogeneity_area_frac_A)
                            heterogeneity_area_frac_system_B = np.append(heterogeneity_area_frac_system_B, heterogeneity_area_frac_B)


                            # Heterogeneity
                            heterogeneity_typ = binning_functs.bin_heterogeneity_system(part_dict['typ'])
                            heterogeneity_typ_system = np.append(heterogeneity_typ_system, heterogeneity_typ)

                            heterogeneity_activity = binning_functs.bin_heterogeneity_system(part_dict['typ'])
                            heterogeneity_activity_system = np.append(heterogeneity_activity_system, heterogeneity_activity)
                                        
                            binned_vel, binned_vel_mean = binning_functs.bin_part_velocity(part_dict['id'], vel_plot_dict)

                            heterogeneity_velocity_all_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['all'], vel_stat_dict['all']['mean'])
                            heterogeneity_velocity_A_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['A'], vel_stat_dict['A']['mean'])
                            heterogeneity_velocity_B_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['B'], vel_stat_dict['B']['mean'])
                            
                            id_heterogeneity_velocity_system_all = np.append(id_heterogeneity_velocity_system_all, heterogeneity_velocity_all_tmp)
                            id_heterogeneity_velocity_system_A = np.append(id_heterogeneity_velocity_system_A, heterogeneity_velocity_A_tmp)
                            id_heterogeneity_velocity_system_B = np.append(id_heterogeneity_velocity_system_B, heterogeneity_velocity_B_tmp)


                            heterogeneity_part_velocity_all_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['all'], vel_plot_dict['all']['mag'])
                            heterogeneity_part_velocity_A_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['A'], vel_plot_dict['A']['mag'])
                            heterogeneity_part_velocity_B_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['B'], vel_plot_dict['B']['mag'])

                            heterogeneity_part_velocity_all = np.append(heterogeneity_part_velocity_all, heterogeneity_part_velocity_all_tmp)
                            heterogeneity_part_velocity_A = np.append(heterogeneity_part_velocity_A, heterogeneity_part_velocity_A_tmp)
                            heterogeneity_part_velocity_B = np.append(heterogeneity_part_velocity_B, heterogeneity_part_velocity_B_tmp)
                        
                            binned_press, binned_press_mean = binning_functs.bin_part_press(part_dict['id'], press_plot_dict)
                            
                            heterogeneity_press_all_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['all'], press_stat_dict['all-all']['press'])
                            heterogeneity_press_A_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['A'], press_stat_dict['all-A']['press'])
                            heterogeneity_press_B_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['B'], press_stat_dict['all-B']['press'])
                            
                            id_heterogeneity_press_system_all = np.append(id_heterogeneity_press_system_all, heterogeneity_press_all_tmp)
                            id_heterogeneity_press_system_A = np.append(id_heterogeneity_press_system_A, heterogeneity_press_A_tmp)
                            id_heterogeneity_press_system_B = np.append(id_heterogeneity_press_system_B, heterogeneity_press_B_tmp)

                            heterogeneity_part_press_all_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['all'], press_plot_dict['all-all']['press'])
                            heterogeneity_part_press_A_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['A'], press_plot_dict['all-A']['press'])
                            heterogeneity_part_press_B_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['B'], press_plot_dict['all-B']['press'])

                            heterogeneity_part_press_all = np.append(heterogeneity_part_press_all, heterogeneity_part_press_all_tmp)
                            heterogeneity_part_press_A = np.append(heterogeneity_part_press_A, heterogeneity_part_press_A_tmp)
                            heterogeneity_part_press_B = np.append(heterogeneity_part_press_B, heterogeneity_part_press_B_tmp)
                            """

                            #Bin system to calculate orientation and alignment that will be used in vector plots
                            NBins_x_tmp = utility_functs.getNBins(lx_box, bin_width_arr[q])
                            NBins_y_tmp = utility_functs.getNBins(ly_box, bin_width_arr[q])

                            # Calculate size of bins
                            sizeBin_x_tmp = utility_functs.roundUp(((lx_box) / NBins_x_tmp), 6)
                            sizeBin_y_tmp = utility_functs.roundUp(((ly_box) / NBins_y_tmp), 6)

                            # Instantiate binning functions module
                            binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x_tmp, NBins_y_tmp, peA, peB, typ, eps)

                            # Calculate bin positions
                            pos_dict_tmp = binning_functs.create_bins()

                            # Assign particles to bins
                            part_dict_tmp = binning_functs.bin_parts(pos, ids, clust_size)

                            phase_dict_tmp = phase_ident_functs.rebin_phases(part_dict_tmp, phase_dict)

                            # Calculate area fraction per bin
                            area_frac_dict_tmp = binning_functs.bin_area_frac(part_dict_tmp)

                            # Heterogeneity
                            heterogeneity_typ = binning_functs.bin_heterogeneity_system(part_dict_tmp['typ'])
                            heterogeneity_typ_system = np.append(heterogeneity_typ_system, heterogeneity_typ)

                            heterogeneity_activity = binning_functs.bin_heterogeneity_system(part_dict_tmp['act'])
                            heterogeneity_activity_system = np.append(heterogeneity_activity_system, heterogeneity_activity)
                            
                            heterogeneity_typ_phases = binning_functs.bin_heterogeneity_binned_phases(part_dict_tmp['typ_mean'], phase_dict_tmp, typ_dict)
                            heterogeneity_typ_bulk = np.append(heterogeneity_typ_bulk, heterogeneity_typ_phases['bulk'])
                            heterogeneity_typ_int = np.append(heterogeneity_typ_int, heterogeneity_typ_phases['int'])
                            heterogeneity_typ_gas = np.append(heterogeneity_typ_gas, heterogeneity_typ_phases['gas'])

                            heterogeneity_activity_phases = binning_functs.bin_heterogeneity_binned_phases(part_dict_tmp['act_mean'], phase_dict_tmp, activity_dict)
                            heterogeneity_activity_bulk = np.append(heterogeneity_activity_bulk, heterogeneity_activity_phases['bulk'])
                            heterogeneity_activity_int = np.append(heterogeneity_activity_int, heterogeneity_activity_phases['int'])
                            heterogeneity_activity_gas = np.append(heterogeneity_activity_gas, heterogeneity_activity_phases['gas'])

                            # Heterogeneity
                            heterogeneity_area_frac_all = binning_functs.bin_heterogeneity_binned_system(area_frac_dict_tmp['bin']['all'], (intPhi/100))
                            heterogeneity_area_frac_A = binning_functs.bin_heterogeneity_binned_system(area_frac_dict_tmp['bin']['A'], (intPhi/100)*parFrac/100)
                            heterogeneity_area_frac_B = binning_functs.bin_heterogeneity_binned_system(area_frac_dict_tmp['bin']['B'], (intPhi/100)*(1.0-parFrac/100))

                            heterogeneity_area_frac_system_all = np.append(heterogeneity_area_frac_system_all, heterogeneity_area_frac_all)
                            heterogeneity_area_frac_system_A = np.append(heterogeneity_area_frac_system_A, heterogeneity_area_frac_A)
                            heterogeneity_area_frac_system_B = np.append(heterogeneity_area_frac_system_B, heterogeneity_area_frac_B)

                            heterogeneity_area_frac_phases_all = binning_functs.bin_heterogeneity_binned_phases(area_frac_dict_tmp['bin']['all'], phase_dict_tmp, phi_dict['all'])
                            heterogeneity_area_frac_phases_A = binning_functs.bin_heterogeneity_binned_phases(area_frac_dict_tmp['bin']['A'], phase_dict_tmp, phi_dict['A'])
                            heterogeneity_area_frac_phases_B = binning_functs.bin_heterogeneity_binned_phases(area_frac_dict_tmp['bin']['B'], phase_dict_tmp, phi_dict['B'])
                            
                            heterogeneity_area_frac_bulk_all = np.append(heterogeneity_area_frac_bulk_all, heterogeneity_area_frac_phases_all['bulk'])
                            heterogeneity_area_frac_bulk_A = np.append(heterogeneity_area_frac_bulk_A, heterogeneity_area_frac_phases_A['bulk'])
                            heterogeneity_area_frac_bulk_B = np.append(heterogeneity_area_frac_bulk_B, heterogeneity_area_frac_phases_B['bulk'])

                            heterogeneity_area_frac_int_all = np.append(heterogeneity_area_frac_int_all, heterogeneity_area_frac_phases_all['int'])
                            heterogeneity_area_frac_int_A = np.append(heterogeneity_area_frac_int_A, heterogeneity_area_frac_phases_A['int'])
                            heterogeneity_area_frac_int_B = np.append(heterogeneity_area_frac_int_B, heterogeneity_area_frac_phases_B['int'])

                            heterogeneity_area_frac_gas_all = np.append(heterogeneity_area_frac_gas_all, heterogeneity_area_frac_phases_all['gas'])
                            heterogeneity_area_frac_gas_A = np.append(heterogeneity_area_frac_gas_A, heterogeneity_area_frac_phases_A['gas'])
                            heterogeneity_area_frac_gas_B = np.append(heterogeneity_area_frac_gas_B, heterogeneity_area_frac_phases_B['gas'])

                            binned_press, binned_press_mean = binning_functs.bin_part_press_phases(part_dict_tmp['id'], press_plot_indiv_dict)

                            heterogeneity_press_all_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['all'], press_stat_dict['all-all']['system']['press'])
                            heterogeneity_press_A_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['A'], press_stat_dict['all-A']['system']['press'])
                            heterogeneity_press_B_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['B'], press_stat_dict['all-B']['system']['press'])

                            heterogeneity_press_system_all = np.append(heterogeneity_press_system_all, heterogeneity_press_all_tmp)
                            heterogeneity_press_system_A = np.append(heterogeneity_press_system_A, heterogeneity_press_A_tmp)
                            heterogeneity_press_system_B = np.append(heterogeneity_press_system_B, heterogeneity_press_B_tmp)
                            
                            #binPress = binning_functs.bin_parts_from_interpart_press(part_dict_tmp, press_hetero_dict['all'])
                            #print(binPress)
                            #stop
                            heterogeneity_press_phases_all = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['all'], phase_dict_tmp, press_dict_avgs['all'])
                            heterogeneity_press_phases_A = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['A'], phase_dict_tmp, press_dict_avgs['A'])
                            heterogeneity_press_phases_B = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['B'], phase_dict_tmp, press_dict_avgs['B'])

                            heterogeneity_press_bulk_all = np.append(heterogeneity_press_bulk_all, heterogeneity_press_phases_all['bulk'])
                            heterogeneity_press_bulk_A = np.append(heterogeneity_press_bulk_A, heterogeneity_press_phases_A['bulk'])
                            heterogeneity_press_bulk_B = np.append(heterogeneity_press_bulk_B, heterogeneity_press_phases_B['bulk'])

                            heterogeneity_press_int_all = np.append(heterogeneity_press_int_all, heterogeneity_press_phases_all['int'])
                            heterogeneity_press_int_A = np.append(heterogeneity_press_int_A, heterogeneity_press_phases_A['int'])
                            heterogeneity_press_int_B = np.append(heterogeneity_press_int_B, heterogeneity_press_phases_B['int'])

                            heterogeneity_press_gas_all = np.append(heterogeneity_press_gas_all, heterogeneity_press_phases_all['gas'])
                            heterogeneity_press_gas_A = np.append(heterogeneity_press_gas_A, heterogeneity_press_phases_A['gas'])
                            heterogeneity_press_gas_B = np.append(heterogeneity_press_gas_B, heterogeneity_press_phases_B['gas'])

                            # Bin average velocity, angular velocity, and velocity curl and divergence
                            if j>(start*time_step):
                                vel_dict_tmp = binning_functs.bin_vel(pos, prev_pos, part_dict_tmp, dt_step)

                                heterogeneity_velocity_all_tmp = binning_functs.bin_heterogeneity_binned_system(vel_dict_tmp['bin']['all']['mag'], vel_stat_dict['all']['mean'])
                                heterogeneity_velocity_A_tmp = binning_functs.bin_heterogeneity_binned_system(vel_dict_tmp['bin']['A']['mag'], vel_stat_dict['A']['mean'])
                                heterogeneity_velocity_B_tmp = binning_functs.bin_heterogeneity_binned_system(vel_dict_tmp['bin']['B']['mag'], vel_stat_dict['B']['mean'])

                                heterogeneity_velocity_system_all = np.append(heterogeneity_velocity_system_all, heterogeneity_velocity_all_tmp)
                                heterogeneity_velocity_system_A = np.append(heterogeneity_velocity_system_A, heterogeneity_velocity_A_tmp)
                                heterogeneity_velocity_system_B = np.append(heterogeneity_velocity_system_B, heterogeneity_velocity_B_tmp)
                                
                                #binPress = binning_functs.bin_parts_from_interpart_press(part_dict_tmp, press_hetero_dict['all'])
                                #print(binPress)
                                #stop
                                heterogeneity_velocity_phases_all = binning_functs.bin_heterogeneity_binned_phases(vel_dict_tmp['bin']['all']['mag'], phase_dict_tmp, velocity_dict_avgs['all'])
                                heterogeneity_velocity_phases_A = binning_functs.bin_heterogeneity_binned_phases(vel_dict_tmp['bin']['A']['mag'], phase_dict_tmp, velocity_dict_avgs['A'])
                                heterogeneity_velocity_phases_B = binning_functs.bin_heterogeneity_binned_phases(vel_dict_tmp['bin']['B']['mag'], phase_dict_tmp, velocity_dict_avgs['B'])

                                heterogeneity_velocity_bulk_all = np.append(heterogeneity_velocity_bulk_all, heterogeneity_velocity_phases_all['bulk'])
                                heterogeneity_velocity_bulk_A = np.append(heterogeneity_velocity_bulk_A, heterogeneity_velocity_phases_A['bulk'])
                                heterogeneity_velocity_bulk_B = np.append(heterogeneity_velocity_bulk_B, heterogeneity_velocity_phases_B['bulk'])

                                heterogeneity_velocity_int_all = np.append(heterogeneity_velocity_int_all, heterogeneity_velocity_phases_all['int'])
                                heterogeneity_velocity_int_A = np.append(heterogeneity_velocity_int_A, heterogeneity_velocity_phases_A['int'])
                                heterogeneity_velocity_int_B = np.append(heterogeneity_velocity_int_B, heterogeneity_velocity_phases_B['int'])

                                heterogeneity_velocity_gas_all = np.append(heterogeneity_velocity_gas_all, heterogeneity_velocity_phases_all['gas'])
                                heterogeneity_velocity_gas_A = np.append(heterogeneity_velocity_gas_A, heterogeneity_velocity_phases_A['gas'])
                                heterogeneity_velocity_gas_B = np.append(heterogeneity_velocity_gas_B, heterogeneity_velocity_phases_B['gas'])

                    data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                    heterogeneity_typ_dict = {'bin': bin_width_arr, 'bulk': heterogeneity_typ_bulk.tolist(), 'int': heterogeneity_typ_int.tolist(), 'gas': heterogeneity_typ_gas.tolist(), 'system': heterogeneity_typ_system.tolist()}
                    heterogeneity_activity_dict = {'bin': bin_width_arr, 'bulk': heterogeneity_activity_bulk.tolist(), 'int': heterogeneity_activity_int.tolist(), 'gas': heterogeneity_activity_gas.tolist(), 'system': heterogeneity_activity_system.tolist()}
                    if j>(start*time_step):
                        heterogeneity_velocity_dict = {'bin': bin_width_arr, 'bulk': {'all': heterogeneity_velocity_bulk_all.tolist(), 'A': heterogeneity_velocity_bulk_A.tolist(), 'B': heterogeneity_velocity_bulk_B.tolist()}, 'int': {'all': heterogeneity_velocity_int_all.tolist(), 'A': heterogeneity_velocity_int_A.tolist(), 'B': heterogeneity_velocity_int_B.tolist()}, 'gas': {'all': heterogeneity_velocity_gas_all.tolist(), 'A': heterogeneity_velocity_gas_A.tolist(), 'B': heterogeneity_velocity_gas_B.tolist()}, 'system': {'all': heterogeneity_velocity_system_all.tolist(), 'A': heterogeneity_velocity_system_A.tolist(), 'B': heterogeneity_velocity_system_B.tolist()} }
                        data_output_functs.write_to_txt(heterogeneity_velocity_dict, dataPath + 'heterogeneity_phases_velocity_' + outfile + '.txt')
                    heterogeneity_phi_dict = {'bin': bin_width_arr, 'bulk': {'all': heterogeneity_area_frac_bulk_all.tolist(), 'A': heterogeneity_area_frac_bulk_A.tolist(), 'B': heterogeneity_area_frac_bulk_B.tolist()}, 'int': {'all': heterogeneity_area_frac_int_all.tolist(), 'A': heterogeneity_area_frac_int_A.tolist(), 'B': heterogeneity_area_frac_int_B.tolist()}, 'gas': {'all': heterogeneity_area_frac_gas_all.tolist(), 'A': heterogeneity_area_frac_gas_A.tolist(), 'B': heterogeneity_area_frac_gas_B.tolist()}, 'system': {'all': heterogeneity_area_frac_system_all.tolist(), 'A': heterogeneity_area_frac_system_A.tolist(), 'B': heterogeneity_area_frac_system_B.tolist()} }
                    heterogeneity_press_dict = {'bin': bin_width_arr, 'bulk': {'all': heterogeneity_press_bulk_all.tolist(), 'A': heterogeneity_press_bulk_A.tolist(), 'B': heterogeneity_press_bulk_B.tolist()}, 'int': {'all': heterogeneity_press_int_all.tolist(), 'A': heterogeneity_press_int_A.tolist(), 'B': heterogeneity_press_int_B.tolist()}, 'gas': {'all': heterogeneity_press_gas_all.tolist(), 'A': heterogeneity_press_gas_A.tolist(), 'B': heterogeneity_press_gas_B.tolist()}, 'system': {'all': heterogeneity_press_system_all.tolist(), 'A': heterogeneity_press_system_A.tolist(), 'B': heterogeneity_press_system_B.tolist()} }
                    
                    # Instantiate data output functions module

                    # Write neighbor data to output file
                    data_output_functs.write_to_txt(heterogeneity_typ_dict, dataPath + 'heterogeneity_phases_typ_' + outfile + '.txt')
                    data_output_functs.write_to_txt(heterogeneity_activity_dict, dataPath + 'heterogeneity_phases_activity_' + outfile + '.txt')
                    data_output_functs.write_to_txt(heterogeneity_phi_dict, dataPath + 'heterogeneity_phases_phi_' + outfile + '.txt')
                    data_output_functs.write_to_txt(heterogeneity_press_dict, dataPath + 'heterogeneity_phases_press_' + outfile + '.txt')

                    prev_pos = pos.copy()
                    prev_ang = ang.copy()
                else:

                    # Calculate cluster CoM
                    com_dict = plotting_utility_functs.com_view(pos, clp_all)
                    
                    # If CoM option given, convert to CoM view
                    if com_option == True:
                        pos = com_dict['pos']
                    #else:
                    #    pos[:,0] = pos[:,0]
                    #    out = np.where(pos[:,0]<-hx_box)[0]
                    #    pos[out,0] = pos[out,0] + lx_box


                    #Bin system to calculate orientation and alignment that will be used in vector plots
                    NBins_x = utility_functs.getNBins(lx_box, bin_width)
                    NBins_y = utility_functs.getNBins(ly_box, bin_width)

                    # Calculate size of bins
                    sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
                    sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

                    # Instantiate binning functions module
                    binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)
                    
                    # Calculate bin positions
                    pos_dict = binning_functs.create_bins()

                    # Assign particles to bins
                    part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                    # Calculate average orientation per bin
                    orient_dict = binning_functs.bin_orient(part_dict, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                    # Calculate area fraction per bin
                    area_frac_dict = binning_functs.bin_area_frac(part_dict)

                    # Calculate average activity per bin
                    activ_dict = binning_functs.bin_activity(part_dict)

                    # Define output file name
                    outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
                    out = outfile + "_frame_"
                    pad = str(j).zfill(5)
                    outFile = out + pad
                    min_size = 0
                    # If cluster sufficiently large
                    if clust_large >= min_size:
                        
                        # Instantiate empty binning arrays
                        clust_size_arr = np.append(clust_size_arr, clust_large)
                        fa_all_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_x_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_all_y_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        fa_all_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_fast_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
                        fa_slow_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

                        # Bin average alignment toward cluster's CoM
                        align_dict = binning_functs.bin_align(orient_dict)

                        

                        #Time frame for plots
                        pad = str(j).zfill(5)

                        # Bin average aligned active force pressure
                        press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)

                        # Bin average active force normal to cluster CoM
                        normal_fa_dict = binning_functs.bin_normal_active_fa(align_dict, area_frac_dict, activ_dict)

                        # Find curl and divergence of binned average alignment toward cluster CoM
                        align_grad_dict = binning_functs.curl_and_div(align_dict)

                        # Instantiate plotting functions module
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)

                        # Instantiate phase identification functions module
                        phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ)

                        # Identify phases of system
                        phase_dict = phase_ident_functs.phase_ident()
                        #phase_dict = phase_ident_functs.phase_ident_planar()

                        # Find IDs of particles in each phase
                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        # Blur phases to mitigate noise
                        phase_dict = phase_ident_functs.phase_blur(phase_dict)

                        # Update phases of each particle ID
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        bulk_id = np.where(phase_dict['part']==0)[0]
                        int_id = np.where(phase_dict['part']==1)[0]
                        gas_id = np.where(phase_dict['part']==2)[0]
                        
                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        # Count number of particles per phase
                        count_dict = particle_prop_functs.phase_count(phase_dict)

                        # Find CoM of bulk phase
                        bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

                        # Separate non-connecting bulk phases
                        bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

                        # Separate non-connecting interfaces
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

                        # Update phase identification array
                        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

                        # Reduce mis-identification of gas
                        phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

                        # Find interface composition
                        phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

                        # Find bulk composition
                        bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

                        # Sort bulk by largest to smallest
                        bulk_comp_dict = phase_ident_functs.phase_sort(bulk_comp_dict)

                        # Sort interface by largest to smallest
                        int_comp_dict = phase_ident_functs.phase_sort(int_comp_dict)

                        # Instantiate interface functions module
                        interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, x_orient_arr, y_orient_arr, pos)
                        
                        # Identify interior and exterior surface bin points
                        surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

                        #planar_surface_dict = interface_functs.det_planar_surface_points(phase_dict, int_dict, int_comp_dict)
                        
                        #Save positions of external and internal edges
                        clust_true = 0

                        # Bin average velocity, angular velocity, and velocity curl and divergence
                        if j>(start*time_step):
                            vel_dict = binning_functs.bin_vel(pos, prev_pos, part_dict, dt_step)
                            ang_vel_dict = binning_functs.bin_ang_vel(ang, prev_ang, part_dict, dt_step)
                            vel_grad = binning_functs.curl_and_div(vel_dict)

                        # Sort surface points for both interior and exterior surfaces of each interface
                        surface2_pos_dict = interface_functs.surface_sort(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'])
                        surface1_pos_dict = interface_functs.surface_sort(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'])

                        # Find CoM of each surface
                        surface_com_dict = interface_functs.surface_com(int_dict, int_comp_dict, surface_dict)

                        # Find radius of surface
                        surface_radius_bin = interface_functs.surface_radius_bins(int_dict, int_comp_dict, surface_dict, surface_com_dict)

                        # Count bins per phase
                        bin_count_dict = phase_ident_functs.phase_bin_count(phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict)

                        # Bin average active force
                        active_fa_dict = binning_functs.bin_active_fa(orient_dict, part_dict, phase_dict['bin'])


                        bin_width2 = 15
                        #Bin system to calculate orientation and alignment that will be used in vector plots
                        NBins_x2 = utility_functs.getNBins(lx_box, bin_width2)
                        NBins_y2 = utility_functs.getNBins(ly_box, bin_width2)

                        # Calculate size of bins
                        sizeBin_x2 = utility_functs.roundUp(((lx_box) / NBins_x2), 6)
                        sizeBin_y2 = utility_functs.roundUp(((ly_box) / NBins_y2), 6)

                        # Instantiate binning functions module
                        binning_functs2 = binning.binning(lx_box, ly_box, partNum, NBins_x2, NBins_y2, peA, peB, typ, eps)
                            
                        # Calculate bin positions
                        pos_dict2 = binning_functs2.create_bins()

                        # Assign particles to bins
                        part_dict2 = binning_functs2.bin_parts(pos, ids, clust_size)

                        # Calculate average orientation per bin
                        orient_dict2 = binning_functs2.bin_orient(part_dict2, pos, x_orient_arr, y_orient_arr, com_dict['com'])

                        #Slow/fast composition of bulk phase
                        part_count_dict, part_id_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)

                        # Instantiate data output functions module
                        data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)
                        
                        #Instantiate dictionaries to save data to
                        all_surface_curves = {}
                        all_surface_measurements = {}
                        
                        # Separate individual interfaces/surfaces
                        sep_surface_dict = interface_functs.separate_surfaces(surface_dict, int_dict, int_comp_dict)

                        # Loop over every surface
                        for m in range(0, len(sep_surface_dict)):

                            # Instantiate dictionaries to save data to
                            averaged_data_arr = {}

                            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                            
                            all_surface_curves[key] = {}
                            all_surface_measurements[key] = {}
                            
                            # Save composition data of interface
                            if (int_comp_dict['ids'][m]!=999):
                                averaged_data_arr['int_id'] = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])
                                averaged_data_arr['bub_id'] = int(int_comp_dict['ids'][m])
                                averaged_data_arr['Na'] = int(int_comp_dict['comp']['A'][m])
                                averaged_data_arr['Nb'] = int(int_comp_dict['comp']['B'][m])
                                averaged_data_arr['Nbin'] = int(bin_count_dict['ids']['int'][m])

                            # If sufficient interior interface points, take measurements
                            if sep_surface_dict[key]['interior']['num']>0:

                                # Sort surface points to curve
                                sort_interior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['interior'])

                                # Prepare surface curve for interpolation
                                sort_interior_ids = interface_functs.surface_curve_prep(sort_interior_ids, int_type = 'interior')

                                # Interpolate surface curve
                                all_surface_curves[key]['interior'] = interface_functs.surface_curve_interp(sort_interior_ids)

                                # Find surface curve CoM
                                com_pov_interior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['interior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['interior'] = interface_functs.surface_radius(com_pov_interior_pos['pos'])

                                # Measure surface curve area
                                all_surface_measurements[key]['interior']['surface area'] = interface_functs.surface_area(com_pov_interior_pos['pos'])

                                # Save surface curve CoM 
                                all_surface_measurements[key]['interior']['com'] = com_pov_interior_pos['com']

                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['interior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['interior'])

                                # Save radial measurements
                                averaged_data_arr['int_mean_rad'] = all_surface_measurements[key]['interior']['mean radius']
                                averaged_data_arr['int_std_rad'] = all_surface_measurements[key]['interior']['std radius']
                                averaged_data_arr['int_sa'] = all_surface_measurements[key]['interior']['surface area']

                            else:
                                averaged_data_arr['int_mean_rad'] = 0
                                averaged_data_arr['int_std_rad'] = 0
                                averaged_data_arr['int_sa'] = 0

                            # If sufficient exterior interface points, take measurements
                            if sep_surface_dict[key]['exterior']['num']>0:

                                # Sort surface points to curve
                                sort_exterior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['exterior'])

                                # Prepare surface curve for interpolation
                                sort_exterior_ids = interface_functs.surface_curve_prep(sort_exterior_ids, int_type = 'exterior')
                                
                                # Interpolate surface curve
                                all_surface_curves[key]['exterior'] = interface_functs.surface_curve_interp(sort_exterior_ids)

                                # Find surface curve CoM
                                com_pov_exterior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['exterior']['pos'])
                                
                                # Measure average surface curve radius
                                all_surface_measurements[key]['exterior'] = interface_functs.surface_radius(com_pov_exterior_pos['pos'])
                                
                                # Measure surface curve area
                                all_surface_measurements[key]['exterior']['surface area'] = interface_functs.surface_area(com_pov_exterior_pos['pos'])
                                
                                # Save surface curve CoM 
                                all_surface_measurements[key]['exterior']['com'] = com_pov_exterior_pos['com']
                                
                                # Perform Fourier analysis on surface curve
                                #all_surface_measurements[key]['exterior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['exterior'])
                                
                                # Save radial measurements
                                averaged_data_arr['ext_mean_rad'] = all_surface_measurements[key]['exterior']['mean radius']
                                averaged_data_arr['ext_std_rad'] = all_surface_measurements[key]['exterior']['std radius']
                                averaged_data_arr['ext_sa'] = all_surface_measurements[key]['exterior']['surface area']
                            else:
                                averaged_data_arr['ext_mean_rad'] = 0
                                averaged_data_arr['ext_std_rad'] = 0
                                averaged_data_arr['ext_sa'] = 0

                            # If sufficient exterior and interior interface points, measure interface width
                            if (sep_surface_dict[key]['exterior']['num']>0) & sep_surface_dict[key]['interior']['num']>0:
                                all_surface_measurements[key]['exterior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                all_surface_measurements[key]['interior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                                averaged_data_arr['width'] = all_surface_measurements[key]['exterior']['surface width']['width']
                            else:
                                averaged_data_arr['width'] = 0
                            
                            # If measurement method specified, save interface data
                            if measurement_method == 'interface-props':
                                data_output_functs.write_to_txt(averaged_data_arr, dataPath + 'BubComp_' + outfile + '.txt')
                        

                        
                        # If cluster has been initially formed, 
                        if steady_state_once == 'False':

                            # Instantiate array for saving largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_arr = np.zeros(partNum)
                            in_clust_arr[clust_id_time]=1

                            # Instantiate array for saving particle positions over time
                            pos_x_arr_time = pos[:,0]
                            pos_y_arr_time = pos[:,1]

                            # Instantiate array for saving surface CoM over time
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            try:
                                com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box])
                                com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box])
                            except:
                                com_x_arr_time = np.array([])
                                com_y_arr_time = np.array([])
                            # Instantiate array for saving cluster CoM over time
                            com_x_parts_arr_time = np.array([com_dict['com']['x']])
                            com_y_parts_arr_time = np.array([com_dict['com']['y']])

                            # Instantiate array for saving phase information over time
                            partPhase_time = phase_dict['part']

                            time_entered_bulk = np.ones(partNum) * tst
                            time_entered_gas = np.ones(partNum) * tst

                            # Instantiate array for saving time step
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Change to True since steady state has been reached
                            steady_state_once = 'True'

                            start_dict = {'bulk': {'time': time_entered_bulk}, 'gas': {'time': time_entered_gas} }
                            lifetime_dict = {}
                            msd_bulk_dict = {}
                            lifetime_stat_dict = {}

                        # If cluster has been formed previously
                        else:

                            # Save largest cluster ID over time
                            clust_id_time = np.where(ids==lcID)[0]
                            in_clust_temp = np.zeros(partNum)
                            in_clust_temp[clust_id_time]=1
                            in_clust_arr = np.vstack((in_clust_arr, in_clust_temp))

                            # Save particle positions over time
                            pos_x_arr_time = np.vstack((pos_x_arr_time, pos[:,0]))
                            pos_y_arr_time = np.vstack((pos_y_arr_time, pos[:,1]))
                            partPhase_time_arr = np.append(partPhase_time_arr, tst)

                            # Save phase information over time
                            partPhase_time = np.vstack((partPhase_time, phase_dict['part']))

                            # Save surface CoM over time
                            try:
                                com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box)
                                com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box)
                            except:
                                try:
                                    com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box)
                                    com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box)
                                except:
                                    com_x_arr_time = np.append(com_x_arr_time, 0)
                                    com_y_arr_time = np.append(com_y_arr_time, 0)

                            # Save cluster CoM over time
                            com_x_parts_arr_time = np.append(com_x_parts_arr_time, com_dict['com']['x'])
                            com_y_parts_arr_time = np.append(com_y_parts_arr_time, com_dict['com']['y'])

                        # Calculate alignment of interface with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
                        
                        # Calculate alignment of bulk with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)

                        # Calculate alignment of gas with nearest surface normal
                        method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)

                        
                        if measurement_method == 'vorticity':
                            #DONE!
                            if j>(start*time_step):

                                if plot == 'y':

                                    plotting_functs.plot_vorticity(vel_dict['bin'], vel_grad['curl'], phase_dict, all_surface_curves, int_comp_dict, active_fa_dict, species='all', interface_id = interface_option)
                        elif measurement_method == 'velocity-corr':
                            if j>(start * time_step):
                                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, pos, x_orient_arr, y_orient_arr)
                                
                                try:
                                    part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)
                                except:
                                    displace_dict = {'A': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])}, 'B': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])} }
                                    part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)

                                vel_plot_dict, corr_dict, vel_stat_dict = particle_prop_functs.velocity_corr(vel_dict['part'], prev_pos, prev_ang, ori)
                                data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'velocity_corr_' + outfile + '.txt')
                        
                        elif measurement_method == 'adsorption':                
                            #DONE!

                            kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                            # Calculate the rate of adsorption to and desorption from cluster
                            kinetics_dict = kinetic_functs.adsorption_nlist()
                            
                            # Save kinetics data between gas and cluster
                            data_output_functs.write_to_txt(kinetics_dict, dataPath + 'kinetics_' + outfile + '.txt')
                        
                        elif measurement_method == 'collision':
                            #DONE!
                            collision_stat_dict, collision_plot_dict = particle_prop_functs.collision_rate()
                            
                            time_clust_all_size = np.append(time_clust_all_size, collision_plot_dict['all'])
                            time_clust_A_size = np.append(time_clust_A_size, collision_plot_dict['A'])
                            time_clust_B_size = np.append(time_clust_B_size, collision_plot_dict['B'])

                            data_output_functs.write_to_txt(collision_stat_dict, dataPath + 'collision_' + outfile + '.txt')       
                            

                        elif measurement_method == 'phase-velocity':
                            #DONE!
                            if j>(start*time_step):

                                part_ang_vel_dict = particle_prop_functs.angular_velocity(ang_vel_dict['part'], phase_dict['part'])
                                part_vel_dict, vel_phase_plot_dict = particle_prop_functs.velocity(vel_dict['part']['mag'], phase_dict['part'])

                                data_output_functs.write_to_txt(part_ang_vel_dict, dataPath + 'angular_velocity_' + outfile + '.txt')
                                data_output_functs.write_to_txt(part_vel_dict, dataPath + 'velocity_' + outfile + '.txt')

                                #if plot == 'y':

                                # Plot histograms of angular velocities in each phase
                                #plotting_functs.ang_vel_histogram(ang_vel_dict['part'], phase_dict['part'])
                                #plotting_functs.ang_vel_bulk_sf_histogram(ang_vel_dict['part'], phase_dict['part'])
                                #plotting_functs.ang_vel_int_sf_histogram(ang_vel_dict['part'], phase_dict['part'])
                    
                            #elif measurement_method == 'voronoi':
                            #    if plot == 'y':
                            #        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst)
                            #        plotting_functs.plot_voronoi(pos)

                        elif (measurement_method == 'activity'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity(pos, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, mono_slow_id = mono_slow_option, mono_fast_id = mono_fast_option, swap_col_id = swap_col_option)
                        elif (measurement_method == 'activity-blank'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_blank(pos, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, mono_slow_id = mono_slow_option, mono_fast_id = mono_fast_option, swap_col_id = swap_col_option)
                        elif (measurement_method == 'activity-blank-video'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_blank_video(pos, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, mono_slow_id = mono_slow_option, mono_fast_id = mono_fast_option, swap_col_id = swap_col_option)
                        
                        elif (measurement_method == 'activity-paper'):
                            #DONE!
                            if plot == 'y':  
                                plotting_functs.plot_part_activity_paper(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)
                        elif (measurement_method == 'activity-seg'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                
                                plotting_functs.plot_part_activity_zoom_seg_new(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)
                        elif (measurement_method == 'activity-wide-adsorb'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_adsorb(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)
                        elif (measurement_method == 'activity-wide-desorb'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_desorb(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)

                        elif (measurement_method == 'activity-wide-adsorb-orient'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_adsorb_orient(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)
                        elif (measurement_method == 'activity-wide-desorb-orient'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_desorb_orient(pos, x_orient_arr, y_orient_arr, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)


                        elif measurement_method == 'activity-com':
                            
                            if plot == 'y':
                                
                                plotting_functs.plot_part_activity_com_plotted(pos, part_id_dict, all_surface_curves, int_comp_dict, com_opt)

                        elif measurement_method == 'phases':
                            #DONE!
                            # Save number of particles per phase data
                            data_output_functs.write_to_txt(part_count_dict, dataPath + 'PhaseComp_' + outfile + '.txt')
                            
                            # Save number of bins per phase data
                            data_output_functs.write_to_txt(bin_count_dict['bin'], dataPath + 'PhaseComp_bins_' + outfile + '.txt')
                            
                            if plot == 'y':

                                # Plot particles color-coded by phase
                                #active_fa_dict2

                                if large_arrows_option==True:
                                    plotting_functs.plot_phases(pos, part_id_dict, phase_dict, all_surface_curves, int_comp_dict, orient_dict2, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, large_arrows_id=large_arrows_option, active_fa_dict_new = active_fa_dict)
                                else:
                                    plotting_functs.plot_phases(pos, part_id_dict, phase_dict, all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, large_arrows_id=large_arrows_option)
                        elif measurement_method == 'radial-heterogeneity':

                            particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                            
                            from csv import writer
                            
                            # If surface is properly defined, calculate and save difference in current time step from previous for interface properties
                            try:
                                radial_heterogeneity_dict, dif_radial_heterogeneity_dict, dif_avg_radial_heterogeneity_dict, plot_heterogeneity_dict, plot_bin_dict = particle_prop_functs.radial_heterogeneity(method2_align_dict, avg_rad_dict, integrated_avg_rad_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict, phase_dict, load_save=load_save)
                                
                                unique_rad = np.unique(radial_heterogeneity_dict['rad'])
                                unique_theta = np.unique(radial_heterogeneity_dict['theta'])

                                int_fa_dens_time_theta = np.zeros(len(unique_theta))
                                int_faA_dens_time_theta = np.zeros(len(unique_theta))
                                int_faB_dens_time_theta = np.zeros(len(unique_theta))

                                int_fa_avg_real_time_theta = np.zeros(len(unique_theta))

                                int_fa_avg_time_theta = np.zeros(len(unique_theta))
                                int_faA_avg_time_theta = np.zeros(len(unique_theta))
                                int_faB_avg_time_theta = np.zeros(len(unique_theta))

                                int_fa_sum_time_theta = np.zeros(len(unique_theta))
                                int_faA_sum_time_theta = np.zeros(len(unique_theta))
                                int_faB_sum_time_theta = np.zeros(len(unique_theta))

                                int_num_dens_time_theta = np.zeros(len(unique_theta))
                                int_num_densA_time_theta = np.zeros(len(unique_theta))
                                int_num_densB_time_theta = np.zeros(len(unique_theta))

                                int_part_fracA_time_theta = np.zeros(len(unique_theta))
                                int_part_fracB_time_theta = np.zeros(len(unique_theta))

                                int_align_time_theta = np.zeros(len(unique_theta))
                                int_alignA_time_theta = np.zeros(len(unique_theta))
                                int_alignB_time_theta = np.zeros(len(unique_theta))
                                
                                temp_id_new = np.where((unique_rad>=0.3) & (unique_rad<=1.1))[0]
                                
                                for j in range(1, len(temp_id_new)):
                                    test_id = np.where(radial_heterogeneity_dict['rad']==unique_rad[temp_id_new[j]])[0]
                                    if len(test_id)>0:

                                        rad_step = (unique_rad[temp_id_new[j]]-unique_rad[temp_id_new[j-1]])*radial_heterogeneity_dict['radius_ang']

                                        int_fa_dens_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_dens']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_dens']['all'][temp_id_new[j-1],:])
                                        int_faA_dens_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_dens']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_dens']['A'][temp_id_new[j-1],:])
                                        int_faB_dens_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_dens']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_dens']['B'][temp_id_new[j-1],:])

                                        int_fa_avg_real_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_avg_real']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_avg_real']['all'][temp_id_new[j-1],:])

                                        int_fa_avg_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_avg']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_avg']['all'][temp_id_new[j-1],:])
                                        int_faA_avg_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_avg']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_avg']['A'][temp_id_new[j-1],:])
                                        int_faB_avg_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_avg']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_avg']['B'][temp_id_new[j-1],:])

                                        int_fa_sum_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_sum']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_sum']['all'][temp_id_new[j-1],:])
                                        int_faA_sum_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_sum']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_sum']['A'][temp_id_new[j-1],:])
                                        int_faB_sum_time_theta += (rad_step/2) * (radial_heterogeneity_dict['fa_sum']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['fa_sum']['B'][temp_id_new[j-1],:])

                                        int_num_dens_time_theta += (rad_step/2) * (radial_heterogeneity_dict['num_dens']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['num_dens']['all'][temp_id_new[j-1],:])
                                        int_num_densA_time_theta += (rad_step/2) * (radial_heterogeneity_dict['num_dens']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['num_dens']['A'][temp_id_new[j-1],:])
                                        int_num_densB_time_theta += (rad_step/2) * (radial_heterogeneity_dict['num_dens']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['num_dens']['B'][temp_id_new[j-1],:])

                                        int_part_fracA_time_theta += (rad_step/2) * (radial_heterogeneity_dict['part_frac']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['part_frac']['A'][temp_id_new[j-1],:])
                                        int_part_fracB_time_theta += (rad_step/2) * (radial_heterogeneity_dict['part_frac']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['part_frac']['B'][temp_id_new[j-1],:])

                                        int_align_time_theta += (rad_step/2) * (radial_heterogeneity_dict['align']['all'][temp_id_new[j],:]+radial_heterogeneity_dict['align']['all'][temp_id_new[j-1],:])
                                        int_alignA_time_theta += (rad_step/2) * (radial_heterogeneity_dict['align']['A'][temp_id_new[j],:]+radial_heterogeneity_dict['align']['A'][temp_id_new[j-1],:])
                                        int_alignB_time_theta += (rad_step/2) * (radial_heterogeneity_dict['align']['B'][temp_id_new[j],:]+radial_heterogeneity_dict['align']['B'][temp_id_new[j-1],:])

                                
                                # Flatten position arrays (theta, r/r_c)
                                rad_arr = radial_heterogeneity_dict['rad'].flatten()
                                theta_arr = radial_heterogeneity_dict['theta'].flatten()

                                # Define arrays for cluster radius and x- and y-positions of cluster's center of mass
                                radius_arr = np.ones(len(theta_arr)) * radial_heterogeneity_dict['radius']
                                com_x_arr = np.ones(len(theta_arr)) * radial_heterogeneity_dict['com']['x']
                                com_y_arr = np.ones(len(theta_arr)) * radial_heterogeneity_dict['com']['y']

                                # Flatten array for average active force magnitude
                                fa_avg_real_arr = radial_heterogeneity_dict['fa_avg_real']['all'].flatten()

                                # Flatten array for average aligned active force magnitude
                                fa_avg_arr = radial_heterogeneity_dict['fa_avg']['all'].flatten()
                                faA_avg_arr = radial_heterogeneity_dict['fa_avg']['A'].flatten()
                                faB_avg_arr = radial_heterogeneity_dict['fa_avg']['B'].flatten()

                                # Flatten array for average aligned active force magnitude
                                fa_sum_arr = radial_heterogeneity_dict['fa_sum']['all'].flatten()
                                faA_sum_arr = radial_heterogeneity_dict['fa_sum']['A'].flatten()
                                faB_sum_arr = radial_heterogeneity_dict['fa_sum']['B'].flatten()

                                # Flatten array for average aligned body force density
                                fa_dens_arr = radial_heterogeneity_dict['fa_dens']['all'].flatten()
                                faA_dens_arr = radial_heterogeneity_dict['fa_dens']['A'].flatten()
                                faB_dens_arr = radial_heterogeneity_dict['fa_dens']['B'].flatten()

                                # Flatten array for average alignment
                                align_arr = radial_heterogeneity_dict['align']['all'].flatten()
                                alignA_arr = radial_heterogeneity_dict['align']['A'].flatten()
                                alignB_arr = radial_heterogeneity_dict['align']['B'].flatten()

                                # Flatten array for average number density
                                num_dens_arr = radial_heterogeneity_dict['num_dens']['all'].flatten()
                                num_densA_arr = radial_heterogeneity_dict['num_dens']['A'].flatten()
                                num_densB_arr = radial_heterogeneity_dict['num_dens']['B'].flatten()

                                part_fracA_arr = radial_heterogeneity_dict['part_frac']['A'].flatten()
                                part_fracB_arr = radial_heterogeneity_dict['part_frac']['B'].flatten()
                                

                                # Loop over radial locations to save each theta to output file for that radial location
                                #for m in range(0, len(unique_theta)):
                                #    radial_heterogeneity_save_dict_int_theta = {'theta': unique_theta[m], 'fa_avg_real': {'all': radial_heterogeneity_dict_theta['fa_avg_real']['all'][m]}, 'fa_avg': {'all': radial_heterogeneity_dict_theta['fa_sum']['all'][m], 'A': radial_heterogeneity_dict_theta['fa_sum']['A'][m], 'B': radial_heterogeneity_dict_theta['fa_sum']['B'][m]}, 'fa_avg': {'all': radial_heterogeneity_dict_theta['fa_avg']['all'][m], 'A': radial_heterogeneity_dict_theta['fa_avg']['A'][m], 'B': radial_heterogeneity_dict_theta['fa_avg']['B'][m]}, 'fa_dens': {'all': radial_heterogeneity_dict_theta['fa_dens']['all'][m], 'A': radial_heterogeneity_dict_theta['fa_dens']['A'][m], 'B': radial_heterogeneity_dict_theta['fa_dens']['B'][m]}, 'align': {'all': radial_heterogeneity_dict_theta['align']['all'][m], 'A': radial_heterogeneity_dict_theta['align']['A'][m], 'B': radial_heterogeneity_dict_theta['align']['B'][m]}, 'num_dens': {'all': radial_heterogeneity_dict_theta['num_dens']['all'][m], 'A': radial_heterogeneity_dict_theta['num_dens']['A'][m], 'B': radial_heterogeneity_dict_theta['num_dens']['B'][m]}} 
                                #    data_output_functs.write_to_txt(radial_heterogeneity_save_dict_int_theta, dataPath + 'Radial_heterogeneity_int_theta_' + outfile + '.txt')

                                for m in range(0, len(unique_theta)):
                                    radial_heterogeneity_save_dict_int = {'theta': unique_theta[m], 'fa_avg_real': {'all': int_fa_avg_real_time_theta[m]}, 'fa_avg': {'all': int_fa_avg_time_theta[m], 'A': int_faA_avg_time_theta[m], 'B': int_faB_avg_time_theta[m]}, 'fa_sum': {'all': int_fa_sum_time_theta[m], 'A': int_faA_sum_time_theta[m], 'B': int_faB_sum_time_theta[m]}, 'fa_dens': {'all': int_fa_dens_time_theta[m], 'A': int_faA_dens_time_theta[m], 'B': int_faB_dens_time_theta[m]}, 'align': {'all': int_align_time_theta[m], 'A': int_alignA_time_theta[m], 'B': int_alignB_time_theta[m]}, 'num_dens': {'all': int_num_dens_time_theta[m], 'A': int_num_densA_time_theta[m], 'B': int_num_densB_time_theta[m]}, 'part_frac': {'A': int_part_fracA_time_theta[m], 'B': int_part_fracB_time_theta[m]}} 
                                    data_output_functs.write_to_txt(radial_heterogeneity_save_dict_int, dataPath + 'Radial_heterogeneity_int_' + outfile + '.txt')
                                
                                for m in range(0, len(unique_theta)):
                                    dif_radial_heterogeneity_save_dict = {'theta': dif_radial_heterogeneity_dict['theta'][m], 'fa_avg_real': {'all': dif_radial_heterogeneity_dict['fa_avg_real']['all'][m]}, 'fa_avg': {'all': dif_radial_heterogeneity_dict['fa_avg']['all'][m], 'A': dif_radial_heterogeneity_dict['fa_avg']['A'][m], 'B': dif_radial_heterogeneity_dict['fa_avg']['B'][m]}, 'fa_sum': {'all': dif_radial_heterogeneity_dict['fa_sum']['all'][m], 'A': dif_radial_heterogeneity_dict['fa_sum']['A'][m], 'B': dif_radial_heterogeneity_dict['fa_sum']['B'][m]}, 'fa_dens': {'all': dif_radial_heterogeneity_dict['fa_dens']['all'][m], 'A': dif_radial_heterogeneity_dict['fa_dens']['A'][m], 'B': dif_radial_heterogeneity_dict['fa_dens']['B'][m]}, 'align': {'all': dif_radial_heterogeneity_dict['align']['all'][m], 'A': dif_radial_heterogeneity_dict['align']['A'][m], 'B': dif_radial_heterogeneity_dict['align']['B'][m]}, 'num_dens': {'all': dif_radial_heterogeneity_dict['num_dens']['all'][m], 'A': dif_radial_heterogeneity_dict['num_dens']['A'][m], 'B': dif_radial_heterogeneity_dict['num_dens']['B'][m]}, 'part_frac': {'A': dif_radial_heterogeneity_dict['part_frac']['A'][m], 'B': dif_radial_heterogeneity_dict['part_frac']['B'][m]}} 
                                    data_output_functs.write_to_txt(dif_radial_heterogeneity_save_dict, dataPath + 'dif_Radial_heterogeneity_' + outfile + '.txt')
                                
                                for m in range(0, len(unique_theta)):
                                    dif_avg_radial_heterogeneity_save_dict = {'theta': dif_avg_radial_heterogeneity_dict['theta'][m], 'fa_avg_real': {'all': dif_avg_radial_heterogeneity_dict['fa_avg_real']['all'][m]}, 'fa_avg': {'all': dif_avg_radial_heterogeneity_dict['fa_avg']['all'][m], 'A': dif_avg_radial_heterogeneity_dict['fa_avg']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['fa_avg']['B'][m]}, 'fa_sum': {'all': dif_avg_radial_heterogeneity_dict['fa_sum']['all'][m], 'A': dif_avg_radial_heterogeneity_dict['fa_sum']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['fa_sum']['B'][m]}, 'fa_dens': {'all': dif_avg_radial_heterogeneity_dict['fa_dens']['all'][m], 'A': dif_avg_radial_heterogeneity_dict['fa_dens']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['fa_dens']['B'][m]}, 'align': {'all': dif_avg_radial_heterogeneity_dict['align']['all'][m], 'A': dif_avg_radial_heterogeneity_dict['align']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['align']['B'][m]}, 'num_dens': {'all': dif_avg_radial_heterogeneity_dict['num_dens']['all'][m], 'A': dif_avg_radial_heterogeneity_dict['num_dens']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['num_dens']['B'][m]}, 'part_frac': {'A': dif_avg_radial_heterogeneity_dict['part_frac']['A'][m], 'B': dif_avg_radial_heterogeneity_dict['part_frac']['B'][m]}} 
                                    data_output_functs.write_to_txt(dif_avg_radial_heterogeneity_save_dict, dataPath + 'dif_avg_Radial_heterogeneity_' + outfile + '.txt')
                                
                                # Loop over radial locations to save each theta to output file for that radial location
                                for m in range(0, len(rad_arr)):
                                    rad_arr = np.ones(len(theta_arr)) * radial_heterogeneity_dict['rad'][m]
                                    radial_heterogeneity_save_dict = {'rad': rad_arr.tolist(), 'theta': theta_arr.tolist(), 'radius': radius_arr.tolist(), 'com_x': com_x_arr.tolist(), 'com_y': com_y_arr.tolist(), 'fa_avg_real': {'all': radial_heterogeneity_dict['fa_avg_real']['all'][m,:].tolist()}, 'fa_sum': {'all': radial_heterogeneity_dict['fa_sum']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['fa_sum']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['fa_sum']['B'][m,:].tolist()}, 'fa_avg': {'all': radial_heterogeneity_dict['fa_avg']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['fa_avg']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['fa_avg']['B'][m,:].tolist()}, 'fa_dens': {'all': radial_heterogeneity_dict['fa_dens']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['fa_dens']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['fa_dens']['B'][m,:].tolist()}, 'align': {'all': radial_heterogeneity_dict['align']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['align']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['align']['B'][m,:].tolist()}, 'num_dens': {'all': radial_heterogeneity_dict['num_dens']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['num_dens']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['num_dens']['B'][m,:].tolist()}, 'part_frac': {'A': radial_heterogeneity_dict['part_frac']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['part_frac']['B'][m,:].tolist()}} 
                                    data_output_functs.write_to_txt(radial_heterogeneity_save_dict, dataPath + 'Radial_heterogeneity_' + outfile + '.txt')

                            except:
                                pass  
                            

                            # If plot defined, create radial heterogeneity visualizations
                            if plot == 'y':

                                # Plot difference in interface properties for current time step from time-averaged properties with radial positions normalized by cluster radius (transformed to circle)
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 1, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 1, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 1, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 1, component = 'dif')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 1, component = 'dif')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 1, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 1, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 1, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='part_frac', types='A', circle_opt = 1, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='part_frac', types='B', circle_opt = 1, component = 'dif')  
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 1, component = 'dif')    
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 1, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 1, component = 'dif')      

                                # Plot difference in interface properties for current time step from time-averaged properties with real positions
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'dif')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'dif')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'dif')    
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='part_frac', types='A', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='part_frac', types='B', circle_opt = 0, component = 'dif')   
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'dif')  
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'dif')  
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'dif')  
                                """ 
                                if os.path.isfile(picPath + 'avg_radial_heterogeneity_' + outFile + ".png") == 0:
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 1, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 1, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 1, component = 'avg')      

                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'avg')   
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'avg') 
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'avg') 
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'avg') 

                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 1, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 1, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 1, component = 'current')      

                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'current') 
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'current') 
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'current') 
                                """
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B')      
                        elif measurement_method == 'bulk-heterogeneity':

                            particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                            
                            from csv import writer
                            
                            try:
                                radial_heterogeneity_dict, plot_heterogeneity_dict, plot_bin_dict = particle_prop_functs.bulk_heterogeneity(method2_align_dict, avg_rad_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict, phase_dict, load_save=load_save)

                                # Loop over radial locations to save each theta to output file for that radial location
                                for m in range(0, len(radial_heterogeneity_dict['x'])):
                                    x_arr = np.ones(len(radial_heterogeneity_dict['y'])) * radial_heterogeneity_dict['x'][m]
                                    radial_heterogeneity_save_dict = {'x': x_arr.tolist(), 'y': radial_heterogeneity_dict['y'].tolist(), 'fa_avg_real': {'all': radial_heterogeneity_dict['fa_avg_real']['all'][m,:].tolist()}, 'align': {'all': radial_heterogeneity_dict['align']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['align']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['align']['B'][m,:].tolist()}, 'num_dens': {'all': radial_heterogeneity_dict['num_dens']['all'][m,:].tolist(), 'A': radial_heterogeneity_dict['num_dens']['A'][m,:].tolist(), 'B': radial_heterogeneity_dict['num_dens']['B'][m,:].tolist()}} 
                                    data_output_functs.write_to_txt(radial_heterogeneity_save_dict, dataPath + 'Bulk_heterogeneity_' + outfile + '.txt')

                            except:
                                pass  
                            

                            # If plot defined, create radial heterogeneity visualizations
                            if plot == 'y':

                                # Plot difference in interface properties for current time step from time-averaged properties with real positions
                                #plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'dif')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'dif')     
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'dif')     
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'dif')      
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'dif')  
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'dif')  
                                plotting_functs.plot_bulk_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'dif')  
                                """ 
                                if os.path.isfile(picPath + 'avg_radial_heterogeneity_' + outFile + ".png") == 0:
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 1, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 1, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 1, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 1, component = 'avg')      

                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'avg')   
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'avg')     
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'avg')      
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'avg') 
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'avg') 
                                    plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'avg') 

                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 1, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 1, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 1, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 1, component = 'current')      

                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='all', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='A', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_dens', types='B', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg', types='all', circle_opt = 0, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='fa_avg_real', types='all', circle_opt = 0, component = 'current')     
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='all', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='A', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='num_dens', types='B', circle_opt = 0, component = 'current')      
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='all', circle_opt = 0, component = 'current') 
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A', circle_opt = 0, component = 'current') 
                                plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B', circle_opt = 0, component = 'current') 
                                """
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='A')      
                                #plotting_functs.plot_radial_heterogeneity(averagesPath, inFile, pos, radial_heterogeneity_dict, plot_bin_dict, plot_heterogeneity_dict, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, measure='align', types='B')      

                        
                        elif measurement_method== 'bubble-body-forces':

                            
                            #DONE!
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)
                            
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate interparticle stresses and pressures
                            stress_stat_dict, press_stat_dict, press_stat_indiv_dict, press_plot_dict, stress_plot_dict, press_plot_indiv_dict, press_hetero_dict = stress_and_pressure_functs.interparticle_pressure_nlist_phases(phase_dict)

                            
                            # Measure radial interparticle pressure
                            radial_int_press_dict = particle_prop_functs.radial_int_press(stress_plot_dict)

                            radial_fa_dict = particle_prop_functs.radial_surface_normal_fa_bubble2(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict)

                            #stress_stat_dict, press_stat_dict, press_plot_dict, stress_plot_dict = lattice_structure_functs.interparticle_pressure_nlist()

                            #radial_int_press_dict = stress_and_pressure_functs.radial_int_press_bubble2(stress_plot_dict, all_surface_curves, int_comp_dict, all_surface_measurements)

                            #com_radial_dict_bubble, com_radial_dict_fa_bubble = particle_prop_functs.radial_measurements2(radial_int_press_dict, radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)
                            #com_radial_dict_fa_bubble = particle_prop_functs.radial_measurements3(radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)
                            
                            com_radial_dict_fa_bubble = particle_prop_functs.radial_ang_measurements(radial_fa_dict, radial_int_press_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)
                            for m in range(0, len(sep_surface_dict)):
                                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                                data_output_functs.write_to_txt(com_radial_dict_fa_bubble[key], dataPath + 'bubble_com_active_pressure_radial_' + outfile + '.txt')
                            
                            radial_fa_dict = particle_prop_functs.radial_surface_normal_fa_bubble3(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict)

                            com_radial_dict_fa_bubble = particle_prop_functs.radial_measurements3(radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)
                            for m in range(0, len(sep_surface_dict)):
                                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                                data_output_functs.write_to_txt(com_radial_dict_fa_bubble[key], dataPath + 'bubble_com_active_pressure_radial2_' + outfile + '.txt')
                            

                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            act_press_dict_bubble = stress_and_pressure_functs.total_active_pressure_bubble(com_radial_dict_fa_bubble, all_surface_measurements, int_comp_dict, all_surface_measurements)
                            
                            """
                            bin_width_arr = np.linspace(1, 6, 6, dtype=float)
                            
                            # Heterogeneity
                            bulk_id = np.where(phase_dict['part']==0)[0]
                            bulk_A_id = np.where((phase_dict['part']==0) & (typ==0))[0]
                            bulk_B_id = np.where((phase_dict['part']==0) & (typ==1))[0]

                            int_id = np.where(phase_dict['part']==1)[0]
                            int_A_id = np.where((phase_dict['part']==1) & (typ==0))[0]
                            int_B_id = np.where((phase_dict['part']==1) & (typ==1))[0]

                            gas_id = np.where(phase_dict['part']==2)[0]
                            gas_A_id = np.where((phase_dict['part']==2) & (typ==0))[0]
                            gas_B_id = np.where((phase_dict['part']==2) & (typ==1))[0]

                            A_id = np.where((typ==0))[0]
                            B_id = np.where((typ==1))[0]
                            
                            act_press_mean_dict = {'all': {'bulk': np.mean(method2_align_dict['part']['align_fa'][bulk_id]), 'int': np.mean(method2_align_dict['part']['align_fa'][int_id]), 'gas': np.mean(method2_align_dict['part']['align_fa'][gas_id]), 'system': np.mean(method2_align_dict['part']['align_fa'])}, 'A': {'bulk': np.mean(method2_align_dict['part']['align_fa'][bulk_A_id]), 'int': np.mean(method2_align_dict['part']['align_fa'][int_A_id]), 'gas': np.mean(method2_align_dict['part']['align_fa'][gas_A_id]), 'system': np.mean(method2_align_dict['part']['align_fa'][A_id]) }, 'B': {'bulk': np.mean(method2_align_dict['part']['align_fa'][bulk_B_id]), 'int': np.mean(method2_align_dict['part']['align_fa'][int_B_id]), 'gas': np.mean(method2_align_dict['part']['align_fa'][gas_B_id]), 'system': np.mean(method2_align_dict['part']['align_fa'][B_id]) } }
                            
                            align_mean_dict = {'all': {'bulk': np.mean(method2_align_dict['part']['align'][bulk_id]), 'int': np.mean(method2_align_dict['part']['align'][int_id]), 'gas': np.mean(method2_align_dict['part']['align'][gas_id]), 'system': np.mean(method2_align_dict['part']['align'])}, 'A': {'bulk': np.mean(method2_align_dict['part']['align'][bulk_A_id]), 'int': np.mean(method2_align_dict['part']['align'][int_A_id]), 'gas': np.mean(method2_align_dict['part']['align'][gas_A_id]), 'system': np.mean(method2_align_dict['part']['align'][A_id]) }, 'B': {'bulk': np.mean(method2_align_dict['part']['align'][bulk_B_id]), 'int': np.mean(method2_align_dict['part']['align'][int_B_id]), 'gas': np.mean(method2_align_dict['part']['align'][gas_B_id]), 'system': np.mean(method2_align_dict['part']['align'][B_id]) } }
                            
                            bulk_area = bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)
                            int_area = bin_count_dict['bin']['int'] * (sizeBin_x * sizeBin_y)
                            gas_area = bin_count_dict['bin']['gas'] * (sizeBin_x * sizeBin_y)
                            
                            num_dens_mean_dict = {'all': {'bulk': part_count_dict['bulk']['all']/bulk_area, 'int': part_count_dict['int']['all']/int_area, 'gas': part_count_dict['gas']['all']/gas_area, 'system': (intPhi/100)/(np.pi/4)}, 'A': {'bulk': part_count_dict['bulk']['A']/bulk_area, 'int': part_count_dict['int']['A']/int_area, 'gas': part_count_dict['gas']['A']/gas_area, 'system': (parFrac/100)*(intPhi/100)/(np.pi/4) }, 'B': {'bulk': part_count_dict['bulk']['B']/bulk_area, 'int': part_count_dict['int']['B']/int_area, 'gas': part_count_dict['gas']['B']/gas_area, 'system': (1.0-parFrac/100)*(intPhi/100)/(np.pi/4) } }

                            heterogeneity_act_press_bulk_all = np.array([])
                            heterogeneity_act_press_bulk_A = np.array([])
                            heterogeneity_act_press_bulk_B = np.array([])

                            heterogeneity_act_press_int_all = np.array([])
                            heterogeneity_act_press_int_A = np.array([])
                            heterogeneity_act_press_int_B = np.array([])

                            heterogeneity_act_press_gas_all = np.array([])
                            heterogeneity_act_press_gas_A = np.array([])
                            heterogeneity_act_press_gas_B = np.array([])

                            heterogeneity_act_press_bulk_all_system = np.array([])
                            heterogeneity_act_press_bulk_A_system = np.array([])
                            heterogeneity_act_press_bulk_B_system = np.array([])

                            heterogeneity_act_press_int_all_system = np.array([])
                            heterogeneity_act_press_int_A_system = np.array([])
                            heterogeneity_act_press_int_B_system = np.array([])

                            heterogeneity_act_press_gas_all_system = np.array([])
                            heterogeneity_act_press_gas_A_system = np.array([])
                            heterogeneity_act_press_gas_B_system = np.array([])

                            heterogeneity_act_press_bulk_all_non_norm = np.array([])
                            heterogeneity_act_press_bulk_A_non_norm = np.array([])
                            heterogeneity_act_press_bulk_B_non_norm = np.array([])

                            heterogeneity_act_press_int_all_non_norm = np.array([])
                            heterogeneity_act_press_int_A_non_norm = np.array([])
                            heterogeneity_act_press_int_B_non_norm = np.array([])

                            heterogeneity_act_press_gas_all_non_norm = np.array([])
                            heterogeneity_act_press_gas_A_non_norm = np.array([])
                            heterogeneity_act_press_gas_B_non_norm = np.array([])

                            heterogeneity_act_press_bulk_all_mean = np.array([])
                            heterogeneity_act_press_bulk_A_mean = np.array([])
                            heterogeneity_act_press_bulk_B_mean = np.array([])

                            heterogeneity_act_press_int_all_mean = np.array([])
                            heterogeneity_act_press_int_A_mean = np.array([])
                            heterogeneity_act_press_int_B_mean = np.array([])

                            heterogeneity_act_press_gas_all_mean = np.array([])
                            heterogeneity_act_press_gas_A_mean = np.array([])
                            heterogeneity_act_press_gas_B_mean = np.array([])




                            heterogeneity_align_bulk_all = np.array([])
                            heterogeneity_align_bulk_A = np.array([])
                            heterogeneity_align_bulk_B = np.array([])

                            heterogeneity_align_int_all = np.array([])
                            heterogeneity_align_int_A = np.array([])
                            heterogeneity_align_int_B = np.array([])

                            heterogeneity_align_gas_all = np.array([])
                            heterogeneity_align_gas_A = np.array([])
                            heterogeneity_align_gas_B = np.array([])

                            heterogeneity_align_bulk_all_system = np.array([])
                            heterogeneity_align_bulk_A_system = np.array([])
                            heterogeneity_align_bulk_B_system = np.array([])

                            heterogeneity_align_int_all_system = np.array([])
                            heterogeneity_align_int_A_system = np.array([])
                            heterogeneity_align_int_B_system = np.array([])

                            heterogeneity_align_gas_all_system = np.array([])
                            heterogeneity_align_gas_A_system = np.array([])
                            heterogeneity_align_gas_B_system = np.array([])

                            heterogeneity_align_bulk_all_non_norm = np.array([])
                            heterogeneity_align_bulk_A_non_norm = np.array([])
                            heterogeneity_align_bulk_B_non_norm = np.array([])

                            heterogeneity_align_int_all_non_norm = np.array([])
                            heterogeneity_align_int_A_non_norm = np.array([])
                            heterogeneity_align_int_B_non_norm = np.array([])

                            heterogeneity_align_gas_all_non_norm = np.array([])
                            heterogeneity_align_gas_A_non_norm = np.array([])
                            heterogeneity_align_gas_B_non_norm = np.array([])

                            heterogeneity_align_bulk_all_mean = np.array([])
                            heterogeneity_align_bulk_A_mean = np.array([])
                            heterogeneity_align_bulk_B_mean = np.array([])

                            heterogeneity_align_int_all_mean = np.array([])
                            heterogeneity_align_int_A_mean = np.array([])
                            heterogeneity_align_int_B_mean = np.array([])

                            heterogeneity_align_gas_all_mean = np.array([])
                            heterogeneity_align_gas_A_mean = np.array([])
                            heterogeneity_align_gas_B_mean = np.array([])



                            heterogeneity_num_dens_bulk_all = np.array([])
                            heterogeneity_num_dens_bulk_A = np.array([])
                            heterogeneity_num_dens_bulk_B = np.array([])

                            heterogeneity_num_dens_int_all = np.array([])
                            heterogeneity_num_dens_int_A = np.array([])
                            heterogeneity_num_dens_int_B = np.array([])

                            heterogeneity_num_dens_gas_all = np.array([])
                            heterogeneity_num_dens_gas_A = np.array([])
                            heterogeneity_num_dens_gas_B = np.array([])

                            heterogeneity_num_dens_bulk_all_system = np.array([])
                            heterogeneity_num_dens_bulk_A_system = np.array([])
                            heterogeneity_num_dens_bulk_B_system = np.array([])

                            heterogeneity_num_dens_int_all_system = np.array([])
                            heterogeneity_num_dens_int_A_system = np.array([])
                            heterogeneity_num_dens_int_B_system = np.array([])

                            heterogeneity_num_dens_gas_all_system = np.array([])
                            heterogeneity_num_dens_gas_A_system = np.array([])
                            heterogeneity_num_dens_gas_B_system = np.array([])

                            heterogeneity_num_dens_bulk_all_non_norm = np.array([])
                            heterogeneity_num_dens_bulk_A_non_norm = np.array([])
                            heterogeneity_num_dens_bulk_B_non_norm = np.array([])

                            heterogeneity_num_dens_int_all_non_norm = np.array([])
                            heterogeneity_num_dens_int_A_non_norm = np.array([])
                            heterogeneity_num_dens_int_B_non_norm = np.array([])

                            heterogeneity_num_dens_gas_all_non_norm = np.array([])
                            heterogeneity_num_dens_gas_A_non_norm = np.array([])
                            heterogeneity_num_dens_gas_B_non_norm = np.array([])

                            heterogeneity_num_dens_bulk_all_mean = np.array([])
                            heterogeneity_num_dens_bulk_A_mean = np.array([])
                            heterogeneity_num_dens_bulk_B_mean = np.array([])

                            heterogeneity_num_dens_int_all_mean = np.array([])
                            heterogeneity_num_dens_int_A_mean = np.array([])
                            heterogeneity_num_dens_int_B_mean = np.array([])

                            heterogeneity_num_dens_gas_all_mean = np.array([])
                            heterogeneity_num_dens_gas_A_mean = np.array([])
                            heterogeneity_num_dens_gas_B_mean = np.array([])

                            radial_fa_dict = particle_prop_functs.radial_surface_normal_fa_bubble2(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict)
                            
                            particle_prop_functs.radial_heterogeneity(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict, phase_dict)

                            

                            


                            
                            stop
                            print(all_surface_curves[key]['exterior']['pos'])
                            print(all_surface_measurements[key]['exterior']['com'])
                            stop
                            com_radial_dict_fa_bubble = particle_prop_functs.radial_ang_active_measurements(radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)

                            stop

                            #hetero_plot_dict = particle_prop_functs.heterogeneity_single_particles(method2_align_dict['part'], phase_dict, act_press_mean_dict, type_m='phases')   
                            #hetero_plot_system_dict = particle_prop_functs.heterogeneity_single_particles(method2_align_dict['part'], phase_dict, act_press_mean_dict, type_m='system')                       

                            #plotting_functs.plot_heterogeneity(hetero_plot_dict, type_m='phases', interface_id = interface_option, orientation_id = orientation_option)
                            #plotting_functs.plot_heterogeneity(hetero_plot_dict, type_m='system', interface_id = interface_option, orientation_id = orientation_option)
                            # I NEED TO MAKE A FUNCTION THAT IDENTIFIES PHASE OF BIN BASED ON NUMBER OF PARTICLES IN IT!
                            for q in range(4, len(bin_width_arr)):
                                
                                #Bin system to calculate orientation and alignment that will be used in vector plots
                                NBins_x_tmp = utility_functs.getNBins(lx_box, bin_width_arr[q])
                                NBins_y_tmp = utility_functs.getNBins(ly_box, bin_width_arr[q])

                                # Calculate size of bins
                                sizeBin_x_tmp = utility_functs.roundUp(((lx_box) / NBins_x_tmp), 6)
                                sizeBin_y_tmp = utility_functs.roundUp(((ly_box) / NBins_y_tmp), 6)

                                # Instantiate binning functions module
                                binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x_tmp, NBins_y_tmp, peA, peB, typ, eps)

                                # Calculate bin positions
                                pos_dict_tmp = binning_functs.create_bins()

                                # Assign particles to bins
                                part_dict_tmp = binning_functs.bin_parts(pos, ids, clust_size)

                                phase_dict_tmp = phase_ident_functs.rebin_phases(part_dict_tmp, phase_dict)

                                # Calculate area fraction per bin
                                area_frac_dict_tmp = binning_functs.bin_area_frac(part_dict_tmp)
                                
                                #print(len(phase_dict_tmp['part']))
                                #print(phase_dict_tmp['part'][np.where(phase_dict_tmp['part']==19584)[0]])
                                
                                bulk_id = np.where(phase_dict_tmp['part']==0)[0]
                                bulk_A_id = np.where((phase_dict_tmp['part']==0) & (typ==0))[0]
                                bulk_B_id = np.where((phase_dict_tmp['part']==0) & (typ==1))[0]

                                int_id = np.where(phase_dict_tmp['part']==1)[0]
                                int_A_id = np.where((phase_dict_tmp['part']==1) & (typ==0))[0]
                                int_B_id = np.where((phase_dict_tmp['part']==1) & (typ==1))[0]

                                gas_id = np.where(phase_dict_tmp['part']==2)[0]
                                gas_A_id = np.where((phase_dict_tmp['part']==2) & (typ==0))[0]
                                gas_B_id = np.where((phase_dict_tmp['part']==2) & (typ==1))[0]

                                binned_press, binned_press_mean = binning_functs.bin_act_press_phases(part_dict_tmp['id'], method2_align_dict['part'])
                                
                                heterogeneity_act_press_phases_all, std_binned_all = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align_fa']['all'], phase_dict_tmp, act_press_mean_dict['all'])
                                heterogeneity_act_press_phases_A, std_binned_A = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align_fa']['A'], phase_dict_tmp, act_press_mean_dict['A'])
                                heterogeneity_act_press_phases_B, std_binned_B = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align_fa']['B'], phase_dict_tmp, act_press_mean_dict['B'])
                                
                                heterogeneity_act_press_bulk_all = np.append(heterogeneity_act_press_bulk_all, heterogeneity_act_press_phases_all['bulk']['norm'])
                                heterogeneity_act_press_bulk_A = np.append(heterogeneity_act_press_bulk_A, heterogeneity_act_press_phases_A['bulk']['norm'])
                                heterogeneity_act_press_bulk_B = np.append(heterogeneity_act_press_bulk_B, heterogeneity_act_press_phases_B['bulk']['norm'])

                                heterogeneity_act_press_int_all = np.append(heterogeneity_act_press_int_all, heterogeneity_act_press_phases_all['int']['norm'])
                                heterogeneity_act_press_int_A = np.append(heterogeneity_act_press_int_A, heterogeneity_act_press_phases_A['int']['norm'])
                                heterogeneity_act_press_int_B = np.append(heterogeneity_act_press_int_B, heterogeneity_act_press_phases_B['int']['norm'])

                                heterogeneity_act_press_gas_all = np.append(heterogeneity_act_press_gas_all, heterogeneity_act_press_phases_all['gas']['norm'])
                                heterogeneity_act_press_gas_A = np.append(heterogeneity_act_press_gas_A, heterogeneity_act_press_phases_A['gas']['norm'])
                                heterogeneity_act_press_gas_B = np.append(heterogeneity_act_press_gas_B, heterogeneity_act_press_phases_B['gas']['norm'])

                                #heterogeneity_act_press_phases_all_system, std_binned_all_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['all'], phase_dict_tmp, act_press_mean_dict['all'])
                                #heterogeneity_act_press_phases_A_system, std_binned_A_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['A'], phase_dict_tmp, act_press_mean_dict['A'])
                                #heterogeneity_act_press_phases_B_system, std_binned_B_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['B'], phase_dict_tmp, act_press_mean_dict['B'])

                                heterogeneity_act_press_bulk_all_system = np.append(heterogeneity_act_press_bulk_all_system, heterogeneity_act_press_phases_all['bulk']['sys_norm'])
                                heterogeneity_act_press_bulk_A_system = np.append(heterogeneity_act_press_bulk_A_system, heterogeneity_act_press_phases_A['bulk']['sys_norm'])
                                heterogeneity_act_press_bulk_B_system = np.append(heterogeneity_act_press_bulk_B_system, heterogeneity_act_press_phases_B['bulk']['sys_norm'])

                                heterogeneity_act_press_int_all_system = np.append(heterogeneity_act_press_int_all_system, heterogeneity_act_press_phases_all['int']['sys_norm'])
                                heterogeneity_act_press_int_A_system = np.append(heterogeneity_act_press_int_A_system, heterogeneity_act_press_phases_A['int']['sys_norm'])
                                heterogeneity_act_press_int_B_system = np.append(heterogeneity_act_press_int_B_system, heterogeneity_act_press_phases_B['int']['sys_norm'])

                                heterogeneity_act_press_gas_all_system = np.append(heterogeneity_act_press_gas_all_system, heterogeneity_act_press_phases_all['gas']['sys_norm'])
                                heterogeneity_act_press_gas_A_system = np.append(heterogeneity_act_press_gas_A_system, heterogeneity_act_press_phases_A['gas']['sys_norm'])
                                heterogeneity_act_press_gas_B_system = np.append(heterogeneity_act_press_gas_B_system, heterogeneity_act_press_phases_B['gas']['sys_norm'])

                                heterogeneity_act_press_bulk_all_non_norm = np.append(heterogeneity_act_press_bulk_all_non_norm, heterogeneity_act_press_phases_all['bulk']['non_norm'])
                                heterogeneity_act_press_bulk_A_non_norm = np.append(heterogeneity_act_press_bulk_A_non_norm, heterogeneity_act_press_phases_A['bulk']['non_norm'])
                                heterogeneity_act_press_bulk_B_non_norm = np.append(heterogeneity_act_press_bulk_B_non_norm, heterogeneity_act_press_phases_B['bulk']['non_norm'])

                                heterogeneity_act_press_int_all_non_norm = np.append(heterogeneity_act_press_int_all_non_norm, heterogeneity_act_press_phases_all['int']['non_norm'])
                                heterogeneity_act_press_int_A_non_norm = np.append(heterogeneity_act_press_int_A_non_norm, heterogeneity_act_press_phases_A['int']['non_norm'])
                                heterogeneity_act_press_int_B_non_norm = np.append(heterogeneity_act_press_int_B_non_norm, heterogeneity_act_press_phases_B['int']['non_norm'])

                                heterogeneity_act_press_gas_all_non_norm = np.append(heterogeneity_act_press_gas_all_non_norm, heterogeneity_act_press_phases_all['gas']['non_norm'])
                                heterogeneity_act_press_gas_A_non_norm = np.append(heterogeneity_act_press_gas_A_non_norm, heterogeneity_act_press_phases_A['gas']['non_norm'])
                                heterogeneity_act_press_gas_B_non_norm = np.append(heterogeneity_act_press_gas_B_non_norm, heterogeneity_act_press_phases_B['gas']['non_norm'])

                                heterogeneity_act_press_bulk_all_mean = np.append(heterogeneity_act_press_bulk_all_mean, heterogeneity_act_press_phases_all['bulk']['mean'])
                                heterogeneity_act_press_bulk_A_mean = np.append(heterogeneity_act_press_bulk_A_mean, heterogeneity_act_press_phases_A['bulk']['mean'])
                                heterogeneity_act_press_bulk_B_mean = np.append(heterogeneity_act_press_bulk_B_mean, heterogeneity_act_press_phases_B['bulk']['mean'])

                                heterogeneity_act_press_int_all_mean = np.append(heterogeneity_act_press_int_all_mean, heterogeneity_act_press_phases_all['int']['mean'])
                                heterogeneity_act_press_int_A_mean = np.append(heterogeneity_act_press_int_A_mean, heterogeneity_act_press_phases_A['int']['mean'])
                                heterogeneity_act_press_int_B_mean = np.append(heterogeneity_act_press_int_B_mean, heterogeneity_act_press_phases_B['int']['mean'])

                                heterogeneity_act_press_gas_all_mean = np.append(heterogeneity_act_press_gas_all_mean, heterogeneity_act_press_phases_all['gas']['mean'])
                                heterogeneity_act_press_gas_A_mean = np.append(heterogeneity_act_press_gas_A_mean, heterogeneity_act_press_phases_A['gas']['mean'])
                                heterogeneity_act_press_gas_B_mean = np.append(heterogeneity_act_press_gas_B_mean, heterogeneity_act_press_phases_B['gas']['mean'])

                                heterogeneity_align_phases_all, std_binned_all_align = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align']['all'], phase_dict_tmp, align_mean_dict['all'])
                                heterogeneity_align_phases_A, std_binned_A_align = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align']['A'], phase_dict_tmp, align_mean_dict['A'])
                                heterogeneity_align_phases_B, std_binned_B_align = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['align']['B'], phase_dict_tmp, align_mean_dict['B'])

                                heterogeneity_align_bulk_all = np.append(heterogeneity_align_bulk_all, heterogeneity_align_phases_all['bulk']['norm'])
                                heterogeneity_align_bulk_A = np.append(heterogeneity_align_bulk_A, heterogeneity_align_phases_A['bulk']['norm'])
                                heterogeneity_align_bulk_B = np.append(heterogeneity_align_bulk_B, heterogeneity_align_phases_B['bulk']['norm'])

                                heterogeneity_align_int_all = np.append(heterogeneity_align_int_all, heterogeneity_align_phases_all['int']['norm'])
                                heterogeneity_align_int_A = np.append(heterogeneity_align_int_A, heterogeneity_align_phases_A['int']['norm'])
                                heterogeneity_align_int_B = np.append(heterogeneity_align_int_B, heterogeneity_align_phases_B['int']['norm'])

                                heterogeneity_align_gas_all = np.append(heterogeneity_align_gas_all, heterogeneity_align_phases_all['gas']['norm'])
                                heterogeneity_align_gas_A = np.append(heterogeneity_align_gas_A, heterogeneity_align_phases_A['gas']['norm'])
                                heterogeneity_align_gas_B = np.append(heterogeneity_align_gas_B, heterogeneity_align_phases_B['gas']['norm'])

                                #heterogeneity_act_press_phases_all_system, std_binned_all_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['all'], phase_dict_tmp, act_press_mean_dict['all'])
                                #heterogeneity_act_press_phases_A_system, std_binned_A_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['A'], phase_dict_tmp, act_press_mean_dict['A'])
                                #heterogeneity_act_press_phases_B_system, std_binned_B_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['B'], phase_dict_tmp, act_press_mean_dict['B'])

                                heterogeneity_align_bulk_all_system = np.append(heterogeneity_align_bulk_all_system, heterogeneity_align_phases_all['bulk']['sys_norm'])
                                heterogeneity_align_bulk_A_system = np.append(heterogeneity_align_bulk_A_system, heterogeneity_align_phases_A['bulk']['sys_norm'])
                                heterogeneity_align_bulk_B_system = np.append(heterogeneity_align_bulk_B_system, heterogeneity_align_phases_B['bulk']['sys_norm'])

                                heterogeneity_align_int_all_system = np.append(heterogeneity_align_int_all_system, heterogeneity_align_phases_all['int']['sys_norm'])
                                heterogeneity_align_int_A_system = np.append(heterogeneity_align_int_A_system, heterogeneity_align_phases_A['int']['sys_norm'])
                                heterogeneity_align_int_B_system = np.append(heterogeneity_align_int_B_system, heterogeneity_align_phases_B['int']['sys_norm'])

                                heterogeneity_align_gas_all_system = np.append(heterogeneity_align_gas_all_system, heterogeneity_align_phases_all['gas']['sys_norm'])
                                heterogeneity_align_gas_A_system = np.append(heterogeneity_align_gas_A_system, heterogeneity_align_phases_A['gas']['sys_norm'])
                                heterogeneity_align_gas_B_system = np.append(heterogeneity_align_gas_B_system, heterogeneity_align_phases_B['gas']['sys_norm'])

                                heterogeneity_align_bulk_all_non_norm = np.append(heterogeneity_align_bulk_all_non_norm, heterogeneity_align_phases_all['bulk']['non_norm'])
                                heterogeneity_align_bulk_A_non_norm = np.append(heterogeneity_align_bulk_A_non_norm, heterogeneity_align_phases_A['bulk']['non_norm'])
                                heterogeneity_align_bulk_B_non_norm = np.append(heterogeneity_align_bulk_B_non_norm, heterogeneity_align_phases_B['bulk']['non_norm'])

                                heterogeneity_align_int_all_non_norm = np.append(heterogeneity_align_int_all_non_norm, heterogeneity_align_phases_all['int']['non_norm'])
                                heterogeneity_align_int_A_non_norm = np.append(heterogeneity_align_int_A_non_norm, heterogeneity_align_phases_A['int']['non_norm'])
                                heterogeneity_align_int_B_non_norm = np.append(heterogeneity_align_int_B_non_norm, heterogeneity_align_phases_B['int']['non_norm'])

                                heterogeneity_align_gas_all_non_norm = np.append(heterogeneity_align_gas_all_non_norm, heterogeneity_align_phases_all['gas']['non_norm'])
                                heterogeneity_align_gas_A_non_norm = np.append(heterogeneity_align_gas_A_non_norm, heterogeneity_align_phases_A['gas']['non_norm'])
                                heterogeneity_align_gas_B_non_norm = np.append(heterogeneity_align_gas_B_non_norm, heterogeneity_align_phases_B['gas']['non_norm'])

                                heterogeneity_align_bulk_all_mean = np.append(heterogeneity_align_bulk_all_mean, heterogeneity_align_phases_all['bulk']['mean'])
                                heterogeneity_align_bulk_A_mean = np.append(heterogeneity_align_bulk_A_mean, heterogeneity_align_phases_A['bulk']['mean'])
                                heterogeneity_align_bulk_B_mean = np.append(heterogeneity_align_bulk_B_mean, heterogeneity_align_phases_B['bulk']['mean'])

                                heterogeneity_align_int_all_mean = np.append(heterogeneity_align_int_all_mean, heterogeneity_align_phases_all['int']['mean'])
                                heterogeneity_align_int_A_mean = np.append(heterogeneity_align_int_A_mean, heterogeneity_align_phases_A['int']['mean'])
                                heterogeneity_align_int_B_mean = np.append(heterogeneity_align_int_B_mean, heterogeneity_align_phases_B['int']['mean'])

                                heterogeneity_align_gas_all_mean = np.append(heterogeneity_align_gas_all_mean, heterogeneity_align_phases_all['gas']['mean'])
                                heterogeneity_align_gas_A_mean = np.append(heterogeneity_align_gas_A_mean, heterogeneity_align_phases_A['gas']['mean'])
                                heterogeneity_align_gas_B_mean = np.append(heterogeneity_align_gas_B_mean, heterogeneity_align_phases_B['gas']['mean'])

                                heterogeneity_num_dens_phases_all, std_binned_all_num_dens = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['num_dens']['all'], phase_dict_tmp, num_dens_mean_dict['all'])
                                heterogeneity_num_dens_phases_A, std_binned_A_num_dens = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['num_dens']['A'], phase_dict_tmp, num_dens_mean_dict['A'])
                                heterogeneity_num_dens_phases_B, std_binned_B_num_dens = binning_functs.bin_heterogeneity_binned_phases(binned_press_mean['num_dens']['B'], phase_dict_tmp, num_dens_mean_dict['B'])

                                heterogeneity_num_dens_bulk_all = np.append(heterogeneity_num_dens_bulk_all, heterogeneity_num_dens_phases_all['bulk']['norm'])
                                heterogeneity_num_dens_bulk_A = np.append(heterogeneity_num_dens_bulk_A, heterogeneity_num_dens_phases_A['bulk']['norm'])
                                heterogeneity_num_dens_bulk_B = np.append(heterogeneity_num_dens_bulk_B, heterogeneity_num_dens_phases_B['bulk']['norm'])

                                heterogeneity_num_dens_int_all = np.append(heterogeneity_num_dens_int_all, heterogeneity_num_dens_phases_all['int']['norm'])
                                heterogeneity_num_dens_int_A = np.append(heterogeneity_num_dens_int_A, heterogeneity_num_dens_phases_A['int']['norm'])
                                heterogeneity_num_dens_int_B = np.append(heterogeneity_num_dens_int_B, heterogeneity_num_dens_phases_B['int']['norm'])

                                heterogeneity_num_dens_gas_all = np.append(heterogeneity_num_dens_gas_all, heterogeneity_num_dens_phases_all['gas']['norm'])
                                heterogeneity_num_dens_gas_A = np.append(heterogeneity_num_dens_gas_A, heterogeneity_num_dens_phases_A['gas']['norm'])
                                heterogeneity_num_dens_gas_B = np.append(heterogeneity_num_dens_gas_B, heterogeneity_num_dens_phases_B['gas']['norm'])

                                #heterogeneity_act_press_phases_all_system, std_binned_all_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['all'], phase_dict_tmp, act_press_mean_dict['all'])
                                #heterogeneity_act_press_phases_A_system, std_binned_A_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['A'], phase_dict_tmp, act_press_mean_dict['A'])
                                #heterogeneity_act_press_phases_B_system, std_binned_B_system = binning_functs.bin_heterogeneity_binned_phases_system(binned_press_mean['B'], phase_dict_tmp, act_press_mean_dict['B'])

                                heterogeneity_num_dens_bulk_all_system = np.append(heterogeneity_num_dens_bulk_all_system, heterogeneity_num_dens_phases_all['bulk']['sys_norm'])
                                heterogeneity_num_dens_bulk_A_system = np.append(heterogeneity_num_dens_bulk_A_system, heterogeneity_num_dens_phases_A['bulk']['sys_norm'])
                                heterogeneity_num_dens_bulk_B_system = np.append(heterogeneity_num_dens_bulk_B_system, heterogeneity_num_dens_phases_B['bulk']['sys_norm'])

                                heterogeneity_num_dens_int_all_system = np.append(heterogeneity_num_dens_int_all_system, heterogeneity_num_dens_phases_all['int']['sys_norm'])
                                heterogeneity_num_dens_int_A_system = np.append(heterogeneity_num_dens_int_A_system, heterogeneity_num_dens_phases_A['int']['sys_norm'])
                                heterogeneity_num_dens_int_B_system = np.append(heterogeneity_num_dens_int_B_system, heterogeneity_num_dens_phases_B['int']['sys_norm'])

                                heterogeneity_num_dens_gas_all_system = np.append(heterogeneity_num_dens_gas_all_system, heterogeneity_num_dens_phases_all['gas']['sys_norm'])
                                heterogeneity_num_dens_gas_A_system = np.append(heterogeneity_num_dens_gas_A_system, heterogeneity_num_dens_phases_A['gas']['sys_norm'])
                                heterogeneity_num_dens_gas_B_system = np.append(heterogeneity_num_dens_gas_B_system, heterogeneity_num_dens_phases_B['gas']['sys_norm'])

                                heterogeneity_num_dens_bulk_all_non_norm = np.append(heterogeneity_num_dens_bulk_all_non_norm, heterogeneity_num_dens_phases_all['bulk']['non_norm'])
                                heterogeneity_num_dens_bulk_A_non_norm = np.append(heterogeneity_num_dens_bulk_A_non_norm, heterogeneity_num_dens_phases_A['bulk']['non_norm'])
                                heterogeneity_num_dens_bulk_B_non_norm = np.append(heterogeneity_num_dens_bulk_B_non_norm, heterogeneity_num_dens_phases_B['bulk']['non_norm'])

                                heterogeneity_num_dens_int_all_non_norm = np.append(heterogeneity_num_dens_int_all_non_norm, heterogeneity_num_dens_phases_all['int']['non_norm'])
                                heterogeneity_num_dens_int_A_non_norm = np.append(heterogeneity_num_dens_int_A_non_norm, heterogeneity_num_dens_phases_A['int']['non_norm'])
                                heterogeneity_num_dens_int_B_non_norm = np.append(heterogeneity_num_dens_int_B_non_norm, heterogeneity_num_dens_phases_B['int']['non_norm'])

                                heterogeneity_num_dens_gas_all_non_norm = np.append(heterogeneity_num_dens_gas_all_non_norm, heterogeneity_num_dens_phases_all['gas']['non_norm'])
                                heterogeneity_num_dens_gas_A_non_norm = np.append(heterogeneity_num_dens_gas_A_non_norm, heterogeneity_num_dens_phases_A['gas']['non_norm'])
                                heterogeneity_num_dens_gas_B_non_norm = np.append(heterogeneity_num_dens_gas_B_non_norm, heterogeneity_num_dens_phases_B['gas']['non_norm'])

                                heterogeneity_num_dens_bulk_all_mean = np.append(heterogeneity_num_dens_bulk_all_mean, heterogeneity_num_dens_phases_all['bulk']['mean'])
                                heterogeneity_num_dens_bulk_A_mean = np.append(heterogeneity_num_dens_bulk_A_mean, heterogeneity_num_dens_phases_A['bulk']['mean'])
                                heterogeneity_num_dens_bulk_B_mean = np.append(heterogeneity_num_dens_bulk_B_mean, heterogeneity_num_dens_phases_B['bulk']['mean'])

                                heterogeneity_num_dens_int_all_mean = np.append(heterogeneity_num_dens_int_all_mean, heterogeneity_num_dens_phases_all['int']['mean'])
                                heterogeneity_num_dens_int_A_mean = np.append(heterogeneity_num_dens_int_A_mean, heterogeneity_num_dens_phases_A['int']['mean'])
                                heterogeneity_num_dens_int_B_mean = np.append(heterogeneity_num_dens_int_B_mean, heterogeneity_num_dens_phases_B['int']['mean'])

                                heterogeneity_num_dens_gas_all_mean = np.append(heterogeneity_num_dens_gas_all_mean, heterogeneity_num_dens_phases_all['gas']['mean'])
                                heterogeneity_num_dens_gas_A_mean = np.append(heterogeneity_num_dens_gas_A_mean, heterogeneity_num_dens_phases_A['gas']['mean'])
                                heterogeneity_num_dens_gas_B_mean = np.append(heterogeneity_num_dens_gas_B_mean, heterogeneity_num_dens_phases_B['gas']['mean'])
                                
                                #DO ALIGNMENT, ACTIVITY, AND NUMBER DENSITY HETEROGENEITY
                                if q == 4:
                                    heterogeneity_plot_dict = {'x': pos_dict_tmp['bottom left']['x'], 'y': pos_dict_tmp['bottom left']['y'], 'all': std_binned_all, 'A': std_binned_A, 'B': std_binned_B}
                                    heterogeneity_plot_system_dict = {'x': pos_dict_tmp['bottom left']['x'], 'y': pos_dict_tmp['bottom left']['y'], 'all': std_binned_all, 'A': std_binned_A, 'B': std_binned_B}
                            heterogeneity_act_press_dict = {'bin': bin_width_arr, 'bulk_press': {'all': heterogeneity_act_press_bulk_all.tolist(), 'A': heterogeneity_act_press_bulk_A.tolist(), 'B': heterogeneity_act_press_bulk_B.tolist()}, 'bulk_sys_press': {'all': heterogeneity_act_press_bulk_all_system.tolist(), 'A': heterogeneity_act_press_bulk_A_system.tolist(), 'B': heterogeneity_act_press_bulk_B_system.tolist()}, 'bulk_non_press': {'all': heterogeneity_act_press_bulk_all_non_norm.tolist(), 'A': heterogeneity_act_press_bulk_A_non_norm.tolist(), 'B': heterogeneity_act_press_bulk_B_non_norm.tolist()}, 'bulk_press_mean': {'all': heterogeneity_act_press_bulk_all_mean.tolist(), 'A': heterogeneity_act_press_bulk_A_mean.tolist(), 'B': heterogeneity_act_press_bulk_B_mean.tolist()}, 'int_press': {'all': heterogeneity_act_press_int_all.tolist(), 'A': heterogeneity_act_press_int_A.tolist(), 'B': heterogeneity_act_press_int_B.tolist()}, 'int_sys_press': {'all': heterogeneity_act_press_int_all_system.tolist(), 'A': heterogeneity_act_press_int_A_system.tolist(), 'B': heterogeneity_act_press_int_B_system.tolist()}, 'int_non_press': {'all': heterogeneity_act_press_int_all_non_norm.tolist(), 'A': heterogeneity_act_press_int_A_non_norm.tolist(), 'B': heterogeneity_act_press_int_B_non_norm.tolist()}, 'int_mean_press': {'all': heterogeneity_act_press_int_all_mean.tolist(), 'A': heterogeneity_act_press_int_A_mean.tolist(), 'B': heterogeneity_act_press_int_B_mean.tolist()}, 'gas_norm_press': {'all': heterogeneity_act_press_gas_all.tolist(), 'A': heterogeneity_act_press_gas_A.tolist(), 'B': heterogeneity_act_press_gas_B.tolist()}, 'gas_sys_press': {'all': heterogeneity_act_press_gas_all_system.tolist(), 'A': heterogeneity_act_press_gas_A_system.tolist(), 'B': heterogeneity_act_press_gas_B_system.tolist()}, 'gas_non_press': {'all': heterogeneity_act_press_gas_all_non_norm.tolist(), 'A': heterogeneity_act_press_gas_A_non_norm.tolist(), 'B': heterogeneity_act_press_gas_B_non_norm.tolist()}, 'gas_mean_press': {'all': heterogeneity_act_press_gas_all_mean.tolist(), 'A': heterogeneity_act_press_gas_A_mean.tolist(), 'B': heterogeneity_act_press_gas_B_mean.tolist()}, 'bulk_align': {'all': heterogeneity_align_bulk_all.tolist(), 'A': heterogeneity_align_bulk_A.tolist(), 'B': heterogeneity_align_bulk_B.tolist()}, 'bulk_sys_align': {'all': heterogeneity_align_bulk_all_system.tolist(), 'A': heterogeneity_align_bulk_A_system.tolist(), 'B': heterogeneity_align_bulk_B_system.tolist()}, 'bulk_non_align': {'all': heterogeneity_align_bulk_all_non_norm.tolist(), 'A': heterogeneity_align_bulk_A_non_norm.tolist(), 'B': heterogeneity_align_bulk_B_non_norm.tolist()}, 'bulk_align_mean': {'all': heterogeneity_align_bulk_all_mean.tolist(), 'A': heterogeneity_align_bulk_A_mean.tolist(), 'B': heterogeneity_align_bulk_B_mean.tolist()}, 'int_align': {'all': heterogeneity_align_int_all.tolist(), 'A': heterogeneity_align_int_A.tolist(), 'B': heterogeneity_align_int_B.tolist()}, 'int_sys_align': {'all': heterogeneity_align_int_all_system.tolist(), 'A': heterogeneity_align_int_A_system.tolist(), 'B': heterogeneity_align_int_B_system.tolist()}, 'int_non_align': {'all': heterogeneity_align_int_all_non_norm.tolist(), 'A': heterogeneity_align_int_A_non_norm.tolist(), 'B': heterogeneity_align_int_B_non_norm.tolist()}, 'int_mean_align': {'all': heterogeneity_align_int_all_mean.tolist(), 'A': heterogeneity_align_int_A_mean.tolist(), 'B': heterogeneity_align_int_B_mean.tolist()}, 'gas_norm_align': {'all': heterogeneity_align_gas_all.tolist(), 'A': heterogeneity_align_gas_A.tolist(), 'B': heterogeneity_align_gas_B.tolist()}, 'gas_sys_align': {'all': heterogeneity_align_gas_all_system.tolist(), 'A': heterogeneity_align_gas_A_system.tolist(), 'B': heterogeneity_align_gas_B_system.tolist()}, 'gas_non_align': {'all': heterogeneity_align_gas_all_non_norm.tolist(), 'A': heterogeneity_align_gas_A_non_norm.tolist(), 'B': heterogeneity_align_gas_B_non_norm.tolist()}, 'gas_mean_align': {'all': heterogeneity_align_gas_all_mean.tolist(), 'A': heterogeneity_align_gas_A_mean.tolist(), 'B': heterogeneity_align_gas_B_mean.tolist()}, 'bulk_dens': {'all': heterogeneity_num_dens_bulk_all.tolist(), 'A': heterogeneity_num_dens_bulk_A.tolist(), 'B': heterogeneity_num_dens_bulk_B.tolist()}, 'bulk_sys_dens': {'all': heterogeneity_num_dens_bulk_all_system.tolist(), 'A': heterogeneity_num_dens_bulk_A_system.tolist(), 'B': heterogeneity_num_dens_bulk_B_system.tolist()}, 'bulk_non_dens': {'all': heterogeneity_num_dens_bulk_all_non_norm.tolist(), 'A': heterogeneity_num_dens_bulk_A_non_norm.tolist(), 'B': heterogeneity_num_dens_bulk_B_non_norm.tolist()}, 'bulk_dens_mean': {'all': heterogeneity_num_dens_bulk_all_mean.tolist(), 'A': heterogeneity_num_dens_bulk_A_mean.tolist(), 'B': heterogeneity_num_dens_bulk_B_mean.tolist()}, 'int_dens': {'all': heterogeneity_num_dens_int_all.tolist(), 'A': heterogeneity_num_dens_int_A.tolist(), 'B': heterogeneity_num_dens_int_B.tolist()}, 'int_sys_dens': {'all': heterogeneity_num_dens_int_all_system.tolist(), 'A': heterogeneity_num_dens_int_A_system.tolist(), 'B': heterogeneity_num_dens_int_B_system.tolist()}, 'int_non_dens': {'all': heterogeneity_num_dens_int_all_non_norm.tolist(), 'A': heterogeneity_num_dens_int_A_non_norm.tolist(), 'B': heterogeneity_num_dens_int_B_non_norm.tolist()}, 'int_mean_dens': {'all': heterogeneity_num_dens_int_all_mean.tolist(), 'A': heterogeneity_num_dens_int_A_mean.tolist(), 'B': heterogeneity_num_dens_int_B_mean.tolist()}, 'gas_norm_dens': {'all': heterogeneity_num_dens_gas_all.tolist(), 'A': heterogeneity_num_dens_gas_A.tolist(), 'B': heterogeneity_num_dens_gas_B.tolist()}, 'gas_sys_dens': {'all': heterogeneity_num_dens_gas_all_system.tolist(), 'A': heterogeneity_num_dens_gas_A_system.tolist(), 'B': heterogeneity_num_dens_gas_B_system.tolist()}, 'gas_non_dens': {'all': heterogeneity_num_dens_gas_all_non_norm.tolist(), 'A': heterogeneity_num_dens_gas_A_non_norm.tolist(), 'B': heterogeneity_num_dens_gas_B_non_norm.tolist()}, 'gas_mean_dens': {'all': heterogeneity_num_dens_gas_all_mean.tolist(), 'A': heterogeneity_num_dens_gas_A_mean.tolist(), 'B': heterogeneity_num_dens_gas_B_mean.tolist()}}
                            data_output_functs.write_to_txt(heterogeneity_act_press_dict, dataPath + 'heterogeneity_active_pressure_' + outfile + '.txt')
                            #stop
                            #plotting_functs.plot_normal_fa_heterogeneity_map(heterogeneity_plot_dict, type_m='phases', interface_id = interface_option, orientation_id = orientation_option)
                            #plotting_functs.plot_normal_fa_heterogeneity_map(heterogeneity_plot_system_dict, type_m='system', interface_id = interface_option, orientation_id = orientation_option)


                            

                            #plt.contourf(pos_dict_tmp[])

                            #stop
                            """
                            
                            """
                            for m in range(0, len(sep_surface_dict)):

                                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                                data_output_functs.write_to_txt(act_press_dict_bubble[key], dataPath + 'bubble_com_active_pressure_' + outfile + '.txt')
                            """
                            """
                            radial_fa_dict = particle_prop_functs.radial_surface_normal_fa(method2_align_dict)

                            stress_stat_dict, press_stat_dict, press_plot_dict, stress_plot_dict = lattice_structure_functs.interparticle_pressure_nlist()

                            radial_int_press_dict = particle_prop_functs.radial_int_press(stress_plot_dict)

                            com_radial_dict, com_radial_dict_fa = particle_prop_functs.radial_measurements(radial_int_press_dict, radial_fa_dict)

                            data_output_functs.write_to_txt(com_radial_dict, dataPath + 'cluster_com_bubble_interparticle_pressure_radial_' + outfile + '.txt')
                            data_output_functs.write_to_txt(com_radial_dict_fa, dataPath + 'cluster_com_bubble_active_pressure_radial_' + outfile + '.txt')
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            act_press_dict = stress_and_pressure_functs.total_active_pressure_interface(com_radial_dict_fa, all_surface_measurements, int_comp_dict)
                            act_press_dict_bubble = stress_and_pressure_functs.total_active_pressure_bubble(com_radial_dict_fa, all_surface_measurements, int_comp_dict)
                            """
                            #DONE!

                            
                            if plot == 'y':

                                # Plot contour map of degree of alignment of particle's active forces with nearest normal to cluster surface
                                plotting_functs.plot_alignment_force(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='all', method='surface') 
                        

                        elif measurement_method == 'density':
                            #DONE!
                            num_dens_dict = binning_functs.phase_number_density(bin_count_dict, part_count_dict)
                            data_output_functs.write_to_txt(num_dens_dict, dataPath + 'Num_dens_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot contour map of binned area fraction for all, A, and B particles and difference in A and B area fraction
                                plotting_functs.plot_area_fraction(area_frac_dict, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_area_fraction(area_frac_dict, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_area_fraction(area_frac_dict, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_area_fraction(area_frac_dict, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='dif', interface_id = interface_option, orientation_id = orientation_option)

                                # Plot contour map of binned fast particle fraction
                                plotting_functs.plot_particle_fraction(area_frac_dict, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='B', interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'com-align':
                            #DONE!
                            if plot == 'y':

                                # Plot contour map of degree of alignment of particle's active forces with direction toward cluster CoM
                                plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='all', method='com')
                                plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='A', method='com')
                                plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='B', method='com')
                                plotting_functs.plot_normal_fa_map(normal_fa_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)
                            
                        elif measurement_method == 'surface-align':
                            #DONE!
                            if plot == 'y':

                                # Plot contour map of degree of alignment of particle's active forces with nearest normal to cluster surface
                                plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='all', method='surface') 
                                plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='A', method='surface')
                                plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='B', method='surface')       
                        
                        elif measurement_method == 'fluctuations':
                            #DONE! (need to add .txt to sample github)
                            if plot == 'y':

                                # Plot particles color-coded by activity with sub-plot of change in cluster size over time
                                plotting_functs.plot_clust_fluctuations(pos, outfile, all_surface_curves, int_comp_dict)
                        
                        elif measurement_method == 'cluster-msd':
                            if j>(start * time_step):

                                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                                cluster_msd_dict = particle_prop_functs.cluster_msd(com_x_msd, com_y_msd, com_r_msd, com_x_parts_arr_time, com_y_parts_arr_time)

                                com_x_msd = cluster_msd_dict['x']
                                com_y_msd = cluster_msd_dict['y']
                                com_r_msd = cluster_msd_dict['r']

                                cluster_msd_dict = {'x': com_x_msd[-1], 'y': com_y_msd[-1], 'r': com_r_msd[-1]}

                                data_output_functs.write_to_txt(cluster_msd_dict, dataPath + 'cluster_msd_' + outfile + '.txt')

                        elif measurement_method == 'compressibility':
                            #DONE!

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate radial distribution functions
                            radial_df_dict = lattice_structure_functs.radial_df()

                            # Calculate compressibility
                            compress_dict = lattice_structure_functs.compressibility(radial_df_dict)

                            # Save compressibility data
                            data_output_functs.write_to_txt(compress_dict, dataPath + 'compressibility_' + outfile + '.txt')
                        
                        elif measurement_method == 'structure-factor-sep':
                            #DONE!

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate compressibility, structure factor, and 1st-wave vector structure factor using alternative hard-coded method
                            compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor2()

                            data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor2_' + outfile + '.txt')

                            data_output_functs.write_to_txt(compress_dict, dataPath + 'sf_compressibility2_' + outfile + '.txt')

                        
                        elif measurement_method == 'structure-factor-freud':
                            #DONE!
                            #DEPRECATED DO TO SPEED, USE measurement_method='structure_factor'
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate compressibility, structure factor, and 1st-wave vector structure factor using Freud
                            compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor_freud()

                            #Save structure factor data
                            data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor_freud_' + outfile + '.txt')

                            # Save compressibility data
                            data_output_functs.write_to_txt(compress_dict, dataPath + 'sf_compressibility_freud_' + outfile + '.txt')

                        elif measurement_method == 'structure-factor-rdf':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate radial distribution functions
                            radial_df_dict = lattice_structure_functs.radial_df()

                            # Calculate compressibility, structure factor, and 1st-wave vector structure factor using hard-coded function
                            compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor(radial_df_dict)

                            # Save structure factor data
                            data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor_' + outfile + '.txt')

                            # Save compressibility data
                            data_output_functs.write_to_txt(compress_dict, dataPath + 'sf_compressibility_' + outfile + '.txt')

                        elif measurement_method == 'int-press':
                            #DONE!
                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)
                            
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate interparticle stresses and pressures
                            stress_stat_dict, press_stat_dict, press_stat_indiv_dict, press_plot_dict, stress_plot_dict, press_plot_indiv_dict, press_hetero_dict = stress_and_pressure_functs.interparticle_pressure_nlist_phases(phase_dict)

                            # Save stress and pressure data
                            data_output_functs.write_to_txt(stress_stat_dict, dataPath + 'interparticle_stress_' + outfile + '.txt')
                            data_output_functs.write_to_txt(press_stat_dict, dataPath + 'interparticle_press_' + outfile + '.txt')
                            data_output_functs.write_to_txt(press_stat_indiv_dict, dataPath + 'interparticle_press_indiv_' + outfile + '.txt')

                            binned_press = binning_functs.bin_measurement(part_dict['id'], phase_dict['bin'], press_hetero_dict)
                            """
                            heterogeneity_press_phases = binning_functs.bin_heterogeneity_press_phases(binned_press, press_hetero_dict, phase_dict, part_dict)
                            print(heterogeneity_press_phases)
                            stop
                            
                            heterogeneity_press_gas = np.append(heterogeneity_gas, heterogeneity_press_phases['gas'])
                            heterogeneity_press_int = np.append(heterogeneity_int, heterogeneity_press_phases['int'])
                            heterogeneity_press_bulk = np.append(heterogeneity_bulk, heterogeneity_press_phases['bulk'])
                            
                            print(heterogeneity_press_bulk)
                            print(heterogeneity_press_int)
                            print(heterogeneity_press_gas)
                            """
                            #heterogeneity_press_dict = {'bulk': heterogeneity_press_bulk.tolist(), 'int': heterogeneity_press_int.tolist(), 'gas': heterogeneity_press_gas.tolist(), 'system': heterogeneity_press_system.tolist()}
                    
                            #print(heterogeneity_dict)
                            """
                            # Heterogeneity
                            heterogeneity = binning_functs.bin_heterogeneity_system(part_dict['typ'], np.var(typ))
                            heterogeneity_system = np.append(heterogeneity_system, heterogeneity)

                            heterogeneity_phases = binning_functs.bin_heterogeneity_phases(part_dict['typ'], typ, phase_dict)
                            heterogeneity_gas = np.append(heterogeneity_gas, heterogeneity_phases['gas'])
                            heterogeneity_int = np.append(heterogeneity_int, heterogeneity_phases['int'])
                            heterogeneity_bulk = np.append(heterogeneity_bulk, heterogeneity_phases['bulk'])
                            
                            heterogeneity_dict = {'bulk': heterogeneity_bulk.tolist(), 'int': heterogeneity_int.tolist(), 'gas': heterogeneity_gas.tolist(), 'system': heterogeneity_system.tolist()}
                            """

                            # Measure radial interparticle pressure
                            radial_int_press_dict = stress_and_pressure_functs.radial_int_press(stress_plot_dict)

                            # Measure radial interparticle pressure
                            com_radial_int_press_dict = stress_and_pressure_functs.radial_com_interparticle_pressure(radial_int_press_dict)

                            # Save radial stress and pressure data
                            data_output_functs.write_to_txt(com_radial_int_press_dict, dataPath + 'com_interparticle_pressure_radial_' + outfile + '.txt')
                            
                            if plot == 'y':
                                plotting_functs.interpart_press_map(stress_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)                             
                        
                        elif measurement_method == 'cluster-velocity': 
                            #DONE!
                            if j>(start * time_step):
                                #Initialize particle-based location measurements
                                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                                
                                # Measure cluster displacement
                                cluster_velocity_dict = particle_prop_functs.cluster_velocity(prev_pos, dt_step)

                                # Save cluster displacement data
                                data_output_functs.write_to_txt(cluster_velocity_dict, dataPath + 'cluster_velocity_' + outfile + '.txt')

                        elif measurement_method == 'centrosymmetry':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate centrosymmetry properties
                            csp_stat_dict, csp_plot_dict = lattice_structure_functs.centrosymmetry()

                            # Save centrosymmetry data
                            data_output_functs.write_to_txt(csp_stat_dict, dataPath + 'centrosymmetry_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot particles color-coded by centrosymmetry parameter
                                plotting_functs.plot_csp(csp_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all')
                            
                        elif measurement_method == 'lattice-spacing':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Measure cluster lattice spacing
                            lat_stat_dict, lat_plot_dict = lattice_structure_functs.lattice_spacing()

                            # Save cluster lattice spacing data
                            data_output_functs.write_to_txt(lat_stat_dict, dataPath + 'lattice_spacing_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot histogram of lattice spacings within cluster
                                plotting_functs.lat_histogram(lat_plot_dict)

                                # Plot particles color-coded by average lattice spacing
                                plotting_functs.lat_map(lat_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)
                                
                        elif measurement_method == 'penetration':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            
                            if j>(start*time_step):

                                # Calculate degree of penetration
                                penetration_dict, start_dict = lattice_structure_functs.penetration_depth(start_dict, prev_pos)

                                # Save degree of penetration data
                                data_output_functs.write_to_txt(penetration_dict, dataPath + 'penetration_depth_' + outfile + '.txt')


                        elif measurement_method == 'radial-df':
                            #DONE

                            # Calculate average number densities
                            if len(avg_num_dens_dict)==0:
                                avg_num_dens_dict['all'] = (part_count_dict['bulk']['all']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['A'] = (part_count_dict['bulk']['A']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['B'] = (part_count_dict['bulk']['B']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['count'] = 1
                            else:
                                avg_num_dens_dict['all'] = (avg_num_dens_dict['all'] + part_count_dict['bulk']['all']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['A'] = (avg_num_dens_dict['A'] + part_count_dict['bulk']['A']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['B'] = (avg_num_dens_dict['B'] + part_count_dict['bulk']['B']/(bin_count_dict['bin']['bulk'] * (sizeBin_x * sizeBin_y)))
                                avg_num_dens_dict['count'] = avg_num_dens_dict['count'] + 1

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Measure cluster lattice spacing
                            lat_stat_dict, lat_plot_dict = lattice_structure_functs.lattice_spacing()

                            # Measure radial distribution function
                            radial_df_dict = lattice_structure_functs.radial_df()

                            # Calculate average radial distribution function
                            for key, value in radial_df_dict.items():
                                try:
                                    if key != 'r':
                                        avg_radial_df_dict[key] = np.array(avg_radial_df_dict[key]) + np.array(value)
                                    else:
                                        avg_radial_df_dict[key] = value
                                except:
                                    avg_radial_df_dict[key] = value
                            
                            # Calculate average lattice spacing
                            lat_sum += lat_stat_dict['bulk']['all']['mean']
                            sum_num += 1
                            
                            # Save radial distribution function data
                            data_output_functs.write_to_txt(radial_df_dict, dataPath + 'radial_df_' + outfile + '.txt')

                            # Measure Wasserstein metric (disparity in radial distribution functions)
                            wasserstein_dict = lattice_structure_functs.wasserstein_distance(radial_df_dict, lat_stat_dict['bulk']['all']['mean'])

                            # Save Wasserstein metric data
                            data_output_functs.write_to_txt(wasserstein_dict, dataPath + 'wasserstein_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot general and all partial radial distribution functions for current time step
                                plotting_functs.plot_general_rdf(radial_df_dict)
                                plotting_functs.plot_all_rdfs(radial_df_dict)

                        elif measurement_method == 'angular-df':

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Measure angular distribution function
                            angular_df_dict = lattice_structure_functs.angular_df()

                            # Save angular distribution data
                            data_output_functs.write_to_txt(angular_df_dict, dataPath + 'angular_df_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot general and all partial angular distribution functions for current time step
                                plotting_functs.plot_general_adf(angular_df_dict)
                                plotting_functs.plot_all_adfs(angular_df_dict)

                        elif measurement_method == 'domain-size':

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Measure domain size
                            domain_stat_dict = lattice_structure_functs.domain_size()

                            # Save domain size data
                            data_output_functs.write_to_txt(domain_stat_dict, dataPath + 'domain_' + outfile + '.txt')

                        elif measurement_method == 'clustering-coefficient':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            
                            # Measure clustering coefficient
                            clust_plot_dict, clust_stat_dict, prob_plot_dict, prob_stat_dict = lattice_structure_functs.clustering_coefficient()
                            
                            time_prob_AA_bulk += np.array(prob_stat_dict['bulk']['AA'])
                            time_prob_AB_bulk += np.array(prob_stat_dict['bulk']['AB'])
                            time_prob_BA_bulk += np.array(prob_stat_dict['bulk']['BA'])
                            time_prob_BB_bulk += np.array(prob_stat_dict['bulk']['BB'])
                            time_prob_num += 1

                            #if plot == 'y':
                                #plotting_functs.prob_histogram(prob_plot_dict, prob_stat_dict)
                                #plotting_functs.prob_histogram2(prob_plot_dict, prob_stat_dict)

                                #stop

                                # Plot all, A, and B particles and color-code by clustering coefficient
                                #plotting_functs.plot_clustering(clust_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)
                                #plotting_functs.plot_clustering(clust_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='A', interface_id = interface_option, orientation_id = orientation_option)
                                #plotting_functs.plot_clustering(clust_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, type='B', interface_id = interface_option, orientation_id = orientation_option)
                        elif measurement_method == 'gas-tracer':
                            #DONE
                            if plot == 'y':

                                # Plot partices and color-code by activity
                                plotting_functs.plot_tracers(pos, part_id_dict, phase_dict, all_surface_curves, int_comp_dict, orient_dict2, interface_id = interface_option, orientation_id = orientation_option, presentation_id = presentation_option, tracer_ids = tracer_ids)

                        elif measurement_method == 'local-gas-density':

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            
                            # Calculate local density of gas phase
                            if lx_box == ly_box:
                                local_dens_stat_dict, local_dens_plot_dict = lattice_structure_functs.local_gas_density()

                            # Save local density of gas phase data
                            data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_gas_density_' + outfile + '.txt')
                            
                            
                            if plot == 'y':

                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)


                                # Plot particles color-coded by local density for all-all, all-A, all-B, A-all, B-all, A-A, A-B, B-A, and B-B nearest neighbor pairs in gas
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'local-density':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            
                            # Calculate local density of dense phase
                            if lx_box == ly_box:
                                local_dens_stat_dict, local_dens_plot_dict = lattice_structure_functs.local_density()

                            # Save local density of dense phase data
                            data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_density_' + outfile + '.txt')
                            
                            if plot == 'y':

                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)

                                # Plot particles color-coded by local density for all-all, all-A, all-B, A-all, B-all, A-A, A-B, B-A, and B-B nearest neighbor pairs in cluster
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'neighbors':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                            
                            # Calculate nearest shell of neighbor data
                            if lx_box == ly_box:
                                neigh_stat_dict, ori_stat_dict, neigh_plot_dict = lattice_structure_functs.nearest_neighbors()
                            else:
                                neigh_stat_dict, ori_stat_dict, neigh_plot_dict = lattice_structure_functs.nearest_neighbors_penetrate()                
                            
                            # Save nearest neighbor data
                            data_output_functs.write_to_txt(neigh_stat_dict, dataPath + 'nearest_neighbors_' + outfile + '.txt')
                            
                            # Save nearest neighbor orientation data
                            data_output_functs.write_to_txt(ori_stat_dict, dataPath + 'nearest_ori_' + outfile + '.txt')
                            
                            if plot == 'y':
                                
                                # Plot particles color-coded by number of neighbors with all-all, all-A, all-B, A-all, B-all, A-A, A-B, B-A, and B-B nearest neighbor pairs
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='all-all', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='all-A', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='all-B', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='A-all', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='B-all', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='A-A', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='A-B', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='B-A', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='B-B', sep_surface_dict=all_surface_curves, int_comp_dict=int_comp_dict, active_fa_dict=active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)

                                # Plot particles color-coded by orientational order with all-all, all-A, all-B, A-A, A-B, B-A, and B-B nearest neighbor pairs
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                
                        elif measurement_method == 'orientation':
                            #DONE!
                            if plot == 'y':
                                # Plot all, A, and B particles and color-code by active force orientation
                                plotting_functs.plot_ang(ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='all', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_ang(ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_ang(ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                
                        elif measurement_method == 'int-press-dep':
                            #DONE!
                            #DEPRECATED! Use measurement_method = 'int_press'
                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)

                            # Calculate interparticle stress
                            stress_plot_dict, stress_stat_dict = stress_and_pressure_functs.interparticle_stress()

                            # Save interparticle stress
                            data_output_functs.write_to_txt(stress_stat_dict,  dataPath + 'interparticle_stress_' + outfile + '.txt')

                            # Calculate virial interparticle pressure
                            press_dict = stress_and_pressure_functs.virial_pressure(stress_stat_dict)

                            # Save interparticle pressure
                            data_output_functs.write_to_txt(press_dict,  dataPath + 'virial_pressure_' + outfile + '.txt')

                            # Calculate shear stress
                            shear_dict = stress_and_pressure_functs.shear_stress(stress_stat_dict)

                            # Save shear stress
                            data_output_functs.write_to_txt(shear_dict, dataPath + 'shear_stress_' + outfile + '.txt')

                            if plot == 'y':

                                # Calculate binned interparticle pressure
                                vp_bin_arr = stress_and_pressure_functs.virial_pressure_binned(stress_plot_dict)

                                # Calculate interparticle pressure per particle
                                vp_part_arr = stress_and_pressure_functs.virial_pressure_part(stress_plot_dict)

                                # Plot binned interparticle pressure
                                plotting_functs.plot_interpart_press_binned(vp_bin_arr, all_surface_curves, int_comp_dict, active_fa_dict, type='all', interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'int-press-nlist':
                            
                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)
                            stress_plot_dict, stress_stat_dict = stress_and_pressure_functs.interparticle_stress_nlist(phase_dict['part'])

                        elif measurement_method == 'com-body-forces':
                            #DONE!
                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)

                            #Initialize particle-based location measurements
                            particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                            # Calculate radially active forces toward cluster CoM
                            radial_fa_dict = stress_and_pressure_functs.radial_normal_fa()

                            # Calculate radial active force pressure toward cluster CoM
                            com_radial_dict = stress_and_pressure_functs.radial_com_active_force_pressure(radial_fa_dict)

                            # Save radial active force pressure toward cluster CoM
                            data_output_functs.write_to_txt(com_radial_dict, dataPath + 'com_interface_pressure_radial_' + outfile + '.txt')

                            # Calculate total active force pressure toward cluster CoM
                            act_press_dict = stress_and_pressure_functs.total_active_pressure(com_radial_dict)

                            # Save total active force pressure toward cluster CoM
                            data_output_functs.write_to_txt(act_press_dict, dataPath + 'com_interface_pressure_' + outfile + '.txt')
                        
                        elif measurement_method == 'surface-body-forces':
                            #DONE!
                            # Initialize stress and pressure functions
                            stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)

                            #Initialize particle-based location measurements
                            particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                            
                            # Calculate radially active forces normal to cluster surface
                            radial_fa_dict = particle_prop_functs.radial_surface_normal_fa(method2_align_dict)

                            # Calculate radial active force pressure normal to cluster surface
                            surface_radial_dict = stress_and_pressure_functs.radial_com_active_force_pressure(radial_fa_dict)

                            # Save radial active force pressure normal to cluster surface
                            data_output_functs.write_to_txt(surface_radial_dict, dataPath + 'surface_interface_pressure_radial_' + outfile + '.txt')

                            # Calculate total active force pressure normal to cluster surface
                            act_press_dict = stress_and_pressure_functs.total_active_pressure(surface_radial_dict)

                            # Save total active force pressure normal to cluster surface
                            data_output_functs.write_to_txt(act_press_dict, dataPath + 'surface_interface_pressure_' + outfile + '.txt')
                            
                        elif measurement_method == 'hexatic-order':
                            # DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate hexatic order of particles with nearest neighbors
                            hexatic_order_dict= lattice_structure_functs.hexatic_order()

                            #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot particles and color-code by hexatic order parameter
                                plotting_functs.plot_hexatic_order(pos, hexatic_order_dict['order'], all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)

                                # Plot particles and color-code by domain angle from hexatic order parameter
                                plotting_functs.plot_domain_angle(pos, hexatic_order_dict['theta'], all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                        
                        elif measurement_method == 'voronoi':

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate voronoi tesselation of particles
                            voronoi_dict= lattice_structure_functs.voronoi()
                            
                            #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot particles and color-code by hexatic order parameter
                                plotting_functs.plot_hexatic_order(pos, hexatic_order_dict['order'], all_surface_curves, int_comp_dict)

                                # Plot particles and color-code by domain angle from hexatic order parameter
                                plotting_functs.plot_domain_angle(pos, hexatic_order_dict['theta'], all_surface_curves, int_comp_dict)

                        elif measurement_method == 'translational-order':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate translational order parameter
                            trans_order_param= lattice_structure_functs.translational_order()

                            #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                            if plot == 'y':

                                # Plot particles and color-code by translational order parameter
                                plotting_functs.plot_trans_order(pos, trans_order_param, all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'steinhardt-order':
                            #DONE!
                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate steinhardt order parameter
                            stein_order_param= lattice_structure_functs.steinhardt_order()

                            #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                            if plot == 'y':
                                
                                # Plot particles and color-code by Steinhardt order parameter
                                plotting_functs.plot_stein_order(pos, stein_order_param, all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
                        
                        elif measurement_method == 'nematic-order':

                            # Initialize lattice structure functions
                            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                            # Calculate nematic order parameter
                            nematic_order_param= lattice_structure_functs.nematic_order()

                            #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                            if plot == 'y':
                                plotting_functs.plot_stein_order(pos, nematic_order_param, all_surface_curves, int_comp_dict)
                        
                        elif measurement_method == 'kinetic-motion':
                            #DONE!
                            if len(partPhase_time_arr)>1:
                            
                                if steady_state_once == 'True':
                                    kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                                    clust_motion_dict, adsorption_dict = kinetic_functs.particle_flux(partPhase_time, in_clust_arr, partPhase_time_arr, clust_size_arr, pos_x_arr_time, pos_y_arr_time, com_x_arr_time, com_y_arr_time, com_x_parts_arr_time, com_y_parts_arr_time)

                                    data_output_functs.write_to_txt(adsorption_dict, dataPath + 'not_adsorption_final_' + outfile + '.txt')
                                    data_output_functs.write_to_txt(clust_motion_dict, dataPath + 'not_clust_motion_final_' + outfile + '.txt')
                        elif measurement_method == 'lifetime':
                            #DONE!
                            if len(partPhase_time_arr)>1:
                            
                                if steady_state_once == 'True':
                                    kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                                    start_dict, lifetime_dict, lifetime_stat_dict,  = kinetic_functs.cluster_lifetime(partPhase_time, start_dict, lifetime_dict, lifetime_stat_dict, in_clust_arr, partPhase_time_arr, msd_bulk_dict, pos, prev_pos)

                                    data_output_functs.write_to_txt(lifetime_stat_dict, dataPath + 'lifetime_' + outfile + '.txt')
                        elif measurement_method == 'bulk-msd':
                            #DONE!
                            if len(partPhase_time_arr)>1:
                            
                                if steady_state_once == 'True':
                                    kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                                    start_dict, lifetime_dict, lifetime_stat_dict,  = kinetic_functs.bulk_msd(partPhase_time, start_dict, msd_bulk_dict, pos, prev_pos)

                                    #data_output_functs.write_to_txt(lifetime_stat_dict, dataPath + 'lifetime_' + outfile + '.txt')
                    else:
                        
                        # Instantiate plotting functions module
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)
                        
                        # Instantiate data output module
                        data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                        # Instantiate particle properties module
                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)
                        
                        if j>(start*time_step):

                            # Bin average velocity, angular velocity, and curl and divergence of velocity
                            vel_dict = binning_functs.bin_vel(pos, prev_pos, part_dict, dt_step)
                            ang_vel_dict = binning_functs.bin_ang_vel(ang, prev_ang, part_dict, dt_step)
                            vel_grad = binning_functs.curl_and_div(vel_dict)

                        

                        if measurement_method == 'activity':

                            #DONE
                            if plot == 'y':

                                # Plot partices and color-code by activity
                                plotting_functs.plot_part_activity(pos, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option, mono_slow_id = mono_slow_option, mono_fast_id = mono_fast_option, swap_col_id = swap_col_option)

                        elif measurement_method == 'gas-radial-df':
                            #DONE

                            # Measure radial distribution function
                            radial_df_dict = particle_prop_functs.gas_radial_df()

                            # Calculate average radial distribution function
                            for key, value in radial_df_dict.items():
                                try:
                                    if key != 'r':
                                        avg_radial_df_dict[key] = np.array(avg_radial_df_dict[key]) + np.array(value)
                                    else:
                                        avg_radial_df_dict[key] = value
                                except:
                                    avg_radial_df_dict[key] = value
                            
                            # Calculate average lattice spacing
                            sum_num += 1
                            
                            # Save radial distribution function data
                            data_output_functs.write_to_txt(radial_df_dict, dataPath + 'gas_radial_df_' + outfile + '.txt')
                            
                            if plot == 'y':

                                # Plot general and all partial radial distribution functions for current time step
                                plotting_functs.plot_general_rdf(radial_df_dict)
                                plotting_functs.plot_all_rdfs(radial_df_dict)

                        elif (measurement_method == 'activity-wide-adsorb'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_adsorb(pos, x_orient_arr, y_orient_arr, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)
                        elif (measurement_method == 'activity-wide-desorb'):
                            #DONE!
                            if plot == 'y':
                                
                                # Plot particles color-coded by activity
                                plotting_functs.plot_part_activity_wide_desorb(pos, x_orient_arr, y_orient_arr, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)

                        elif measurement_method == 'activity-wide':

                            #DONE
                            if plot == 'y':

                                # Plot partices and color-code by activity
                                plotting_functs.plot_part_activity_wide(pos, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)

                        elif measurement_method == 'local-system-density':

                            # Calculate local density of gas phase
                            if lx_box == ly_box:
                                local_dens_stat_dict, local_homo_stat_dict, local_dens_plot_dict = particle_prop_functs.local_system_density()

                            # Save local density of gas phase data
                            data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_system_density_' + outfile + '.txt')
                            data_output_functs.write_to_txt(local_homo_stat_dict, dataPath + 'local_system_homogeneity_' + outfile + '.txt')

                            if plot == 'y':

                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)


                                # Plot particles color-coded by local density for all-all, all-A, all-B, A-all, B-all, A-A, A-B, B-A, and B-B nearest neighbor pairs in gas
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)

                        elif measurement_method == 'neighbors':
                            #DONE
                            # Calculate nearest shell of neighbor data for membrane penetration
                            neigh_stat_dict, ori_stat_dict, neigh_plot_dict = particle_prop_functs.nearest_neighbors_penetrate()                
                            
                            # Write neighbor data to output file
                            data_output_functs.write_to_txt(neigh_stat_dict, dataPath + 'nearest_neighbors_' + outfile + '.txt')
                            
                            if plot == 'y':

                                # Plot particles and color-code by number of nearest neighbors
                                plotting_functs.plot_neighbors(neigh_plot_dict, x_orient_arr, y_orient_arr, pos, pair='all-all')

                        elif measurement_method == 'penetration':
                            #DONE~

                            if j>(start*time_step):
                                penetration_dict, start_dict, vertical_shift, dify_long = particle_prop_functs.penetration_depth(start_dict, prev_pos, vertical_shift, dify_long)

                                data_output_functs.write_to_txt(penetration_dict, dataPath + 'penetration_depth_' + outfile + '.txt')

                                action_arr = np.append(action_arr, penetration_dict['action'])
                        elif measurement_method == 'part-velocity':
                            if j>(start * time_step):
                                
                                vel_plot_dict, vel_stat_dict = particle_prop_functs.part_velocity(prev_pos, prev_ang, ori, dt_step)
                                data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'velocity_' + outfile + '.txt')

                                # Initialize stress and pressure functions
                                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac)
                                
                                stress_stat_dict, press_stat_dict, press_plot_dict = stress_and_pressure_functs.interparticle_pressure_nlist_system()

                                bin_width_arr = np.linspace(1, 49, 49, dtype=float)
                                bin_width_arr2 = np.linspace(50, int(lx_box), int((lx_box-50)/10), dtype=float)
                                bin_width_arr = np.concatenate((bin_width_arr, bin_width_arr2), axis=0)
                                
                                heterogeneity_part_velocity_all = np.array([])
                                heterogeneity_part_velocity_A = np.array([])
                                heterogeneity_part_velocity_B = np.array([])

                                id_heterogeneity_velocity_system_all = np.array([])
                                id_heterogeneity_velocity_system_A = np.array([])
                                id_heterogeneity_velocity_system_B = np.array([])


                                heterogeneity_activity_system = np.array([])
                                heterogeneity_typ_system = np.array([])

                                heterogeneity_part_press_all = np.array([])
                                heterogeneity_part_press_A = np.array([])
                                heterogeneity_part_press_B = np.array([])

                                id_heterogeneity_press_system_all = np.array([])
                                id_heterogeneity_press_system_A = np.array([])
                                id_heterogeneity_press_system_B = np.array([])

                                heterogeneity_area_frac_system_all = np.array([])
                                heterogeneity_area_frac_system_A = np.array([])
                                heterogeneity_area_frac_system_B = np.array([])

                                id_heterogeneity_typ_system_all = np.array([])
                                id_heterogeneity_activity_system_all = np.array([])

                                for q in range(0, len(bin_width_arr)):
                                    #Bin system to calculate orientation and alignment that will be used in vector plots
                                    NBins_x_tmp = utility_functs.getNBins(lx_box, bin_width_arr[q])
                                    NBins_y_tmp = utility_functs.getNBins(ly_box, bin_width_arr[q])

                                    # Calculate size of bins
                                    sizeBin_x_tmp = utility_functs.roundUp(((lx_box) / NBins_x_tmp), 6)
                                    sizeBin_y_tmp = utility_functs.roundUp(((ly_box) / NBins_y_tmp), 6)

                                    # Instantiate binning functions module
                                    binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x_tmp, NBins_y_tmp, peA, peB, typ, eps)
                                    
                                    # Calculate bin positions
                                    pos_dict = binning_functs.create_bins()

                                    # Assign particles to bins
                                    part_dict = binning_functs.bin_parts(pos, ids, clust_size)

                                    heterogeneity_typ_all_tmp = binning_functs.bin_heterogeneity_binned_system(part_dict['typ_mean'], (1.0-parFrac/100))
                                    id_heterogeneity_typ_system_all = np.append(id_heterogeneity_typ_system_all, heterogeneity_typ_all_tmp)

                                    heterogeneity_activity_all_tmp = binning_functs.bin_heterogeneity_binned_system(part_dict['act_mean'], peA * (parFrac/100) + peB * (1.0-parFrac/100))
                                    id_heterogeneity_activity_system_all = np.append(id_heterogeneity_activity_system_all, heterogeneity_activity_all_tmp)

                                    # Calculate area fraction per bin
                                    area_frac_dict = binning_functs.bin_area_frac(part_dict)

                                    # Heterogeneity
                                    heterogeneity_area_frac_all = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['all'], (intPhi/100))
                                    heterogeneity_area_frac_A = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['A'], (intPhi/100)*parFrac/100)
                                    heterogeneity_area_frac_B = binning_functs.bin_heterogeneity_binned_system(area_frac_dict['bin']['B'], (intPhi/100)*(1.0-parFrac/100))

                                    heterogeneity_area_frac_system_all = np.append(heterogeneity_area_frac_system_all, heterogeneity_area_frac_all)
                                    heterogeneity_area_frac_system_A = np.append(heterogeneity_area_frac_system_A, heterogeneity_area_frac_A)
                                    heterogeneity_area_frac_system_B = np.append(heterogeneity_area_frac_system_B, heterogeneity_area_frac_B)


                                    # Heterogeneity
                                    heterogeneity_typ = binning_functs.bin_heterogeneity_system(part_dict['typ'])
                                    heterogeneity_typ_system = np.append(heterogeneity_typ_system, heterogeneity_typ)

                                    heterogeneity_activity = binning_functs.bin_heterogeneity_system(part_dict['typ'])
                                    heterogeneity_activity_system = np.append(heterogeneity_activity_system, heterogeneity_activity)
                                                
                                    binned_vel, binned_vel_mean = binning_functs.bin_part_velocity(part_dict['id'], vel_plot_dict)

                                    heterogeneity_velocity_all_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['all'], vel_stat_dict['all']['mean'])
                                    heterogeneity_velocity_A_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['A'], vel_stat_dict['A']['mean'])
                                    heterogeneity_velocity_B_tmp = binning_functs.bin_heterogeneity_binned_system(binned_vel_mean['B'], vel_stat_dict['B']['mean'])
                                    
                                    id_heterogeneity_velocity_system_all = np.append(id_heterogeneity_velocity_system_all, heterogeneity_velocity_all_tmp)
                                    id_heterogeneity_velocity_system_A = np.append(id_heterogeneity_velocity_system_A, heterogeneity_velocity_A_tmp)
                                    id_heterogeneity_velocity_system_B = np.append(id_heterogeneity_velocity_system_B, heterogeneity_velocity_B_tmp)


                                    heterogeneity_part_velocity_all_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['all'], vel_plot_dict['all']['mag'])
                                    heterogeneity_part_velocity_A_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['A'], vel_plot_dict['A']['mag'])
                                    heterogeneity_part_velocity_B_tmp = binning_functs.bin_heterogeneity_part_vel_system(binned_vel['B'], vel_plot_dict['B']['mag'])

                                    heterogeneity_part_velocity_all = np.append(heterogeneity_part_velocity_all, heterogeneity_part_velocity_all_tmp)
                                    heterogeneity_part_velocity_A = np.append(heterogeneity_part_velocity_A, heterogeneity_part_velocity_A_tmp)
                                    heterogeneity_part_velocity_B = np.append(heterogeneity_part_velocity_B, heterogeneity_part_velocity_B_tmp)
                                
                                    binned_press, binned_press_mean = binning_functs.bin_part_press(part_dict['id'], press_plot_dict)
                                    
                                    heterogeneity_press_all_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['all'], press_stat_dict['all-all']['press'])
                                    heterogeneity_press_A_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['A'], press_stat_dict['all-A']['press'])
                                    heterogeneity_press_B_tmp = binning_functs.bin_heterogeneity_binned_system(binned_press_mean['B'], press_stat_dict['all-B']['press'])
                                    
                                    id_heterogeneity_press_system_all = np.append(id_heterogeneity_press_system_all, heterogeneity_press_all_tmp)
                                    id_heterogeneity_press_system_A = np.append(id_heterogeneity_press_system_A, heterogeneity_press_A_tmp)
                                    id_heterogeneity_press_system_B = np.append(id_heterogeneity_press_system_B, heterogeneity_press_B_tmp)

                                    heterogeneity_part_press_all_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['all'], press_plot_dict['all-all']['press'])
                                    heterogeneity_part_press_A_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['A'], press_plot_dict['all-A']['press'])
                                    heterogeneity_part_press_B_tmp = binning_functs.bin_heterogeneity_part_press_system(binned_press['B'], press_plot_dict['all-B']['press'])

                                    heterogeneity_part_press_all = np.append(heterogeneity_part_press_all, heterogeneity_part_press_all_tmp)
                                    heterogeneity_part_press_A = np.append(heterogeneity_part_press_A, heterogeneity_part_press_A_tmp)
                                    heterogeneity_part_press_B = np.append(heterogeneity_part_press_B, heterogeneity_part_press_B_tmp)

                                heterogeneity_dict = {'bin_width': bin_width_arr.tolist(), 'id_typ': id_heterogeneity_typ_system_all.tolist(), 'q_typ': heterogeneity_typ_system.tolist(), 'id_activity': id_heterogeneity_activity_system_all.tolist(), 'q_activity': heterogeneity_activity_system.tolist(), 'id_area_frac': {'all': heterogeneity_area_frac_system_all.tolist(), 'A': heterogeneity_area_frac_system_A.tolist(), 'B': heterogeneity_area_frac_system_B.tolist()}, 'id_velocity': {'all': id_heterogeneity_velocity_system_all.tolist(), 'A': id_heterogeneity_velocity_system_A.tolist(), 'B': id_heterogeneity_velocity_system_B.tolist()}, 'q_velocity': {'all': heterogeneity_part_velocity_all.tolist(), 'A': heterogeneity_part_velocity_A.tolist(), 'B': heterogeneity_part_velocity_B.tolist()}, 'id_press': {'all': id_heterogeneity_press_system_all.tolist(), 'A': id_heterogeneity_press_system_A.tolist(), 'B': id_heterogeneity_press_system_B.tolist()}, 'q_press': {'all': heterogeneity_part_press_all.tolist(), 'A': heterogeneity_part_press_A.tolist(), 'B': heterogeneity_part_press_B.tolist()}}

                                # Write neighbor data to output file
                                data_output_functs.write_to_txt(heterogeneity_dict, dataPath + 'gas_heterogeneity_' + outfile + '.txt')
                                
                                time_velA_mag = np.append(time_velA_mag, vel_plot_dict['A']['mag'])
                                time_velB_mag = np.append(time_velB_mag, vel_plot_dict['B']['mag'])

                                #elif measurement_options[0] == 'part-velocity':
                                #    if j>(start * time_step):

                                #        vel_plot_dict, vel_stat_dict = particle_prop_functs.part_velocity_phases(prev_pos, prev_ang, ori)
                                #        data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'velocity_' + outfile + '.txt')

                                #        time_velA_mag = np.append(time_velA_mag, vel_plot_dict['A']['mag'])
                                #        time_velB_mag = np.append(time_velB_mag, vel_plot_dict['B']['mag'])
                                #if plot == 'y':
                                #plotting_functs.vel_histogram(vel_plot_dict, dt_step)
                        elif measurement_method == 'velocity-corr':
                            if j>(start * time_step):

                                #try:
                                #    part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)
                                #except:
                                    #displace_dict = {'A': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])}, 'B': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])} }
                                    #part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)

                                vel_plot_dict, corr_dict, vel_stat_dict = particle_prop_functs.velocity_corr(vel_dict['part'], prev_pos, prev_ang, ori)
                                data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'velocity_corr_' + outfile + '.txt')
                        elif measurement_method == 'collision':
                            #DONE!
                            if j>(start * time_step):

                                vel_plot_dict, vel_stat_dict = particle_prop_functs.part_velocity(prev_pos, prev_ang, ori)

                                collision_stat_dict, collision_plot_dict, neigh_dict = particle_prop_functs.collision_rate(vel_plot_dict, neigh_dict)
                            
                                time_clust_all_size = np.append(time_clust_all_size, collision_plot_dict['all'])
                                time_clust_A_size = np.append(time_clust_A_size, collision_plot_dict['A'])
                                time_clust_B_size = np.append(time_clust_B_size, collision_plot_dict['B'])

                                data_output_functs.write_to_txt(collision_stat_dict, dataPath + 'collision_' + outfile + '.txt')       
                        elif measurement_method == 'local-gas-density':

                            # Calculate local density of gas phase
                            if lx_box == ly_box:
                                local_dens_stat_dict, local_dens_plot_dict = particle_prop_functs.local_gas_density()

                            # Save local density of gas phase data
                            data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_gas_density_' + outfile + '.txt')
                            
                            
                            if plot == 'y':

                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_homogeneity(local_dens_plot_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)


                                # Plot particles color-coded by local density for all-all, all-A, all-B, A-all, B-all, A-A, A-B, B-A, and B-B nearest neighbor pairs in gas
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='all-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='all-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='all-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='A-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='B-all', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='A-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='A-B', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='B-A', interface_id = interface_option, orientation_id = orientation_option)
                                plotting_functs.plot_local_density(local_dens_plot_dict, pair='B-B', interface_id = interface_option, orientation_id = orientation_option)

                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                    utility_functs = utility.utility(lx_box, ly_box)

                    prev_pos = pos.copy()
                    prev_ang = ang.copy()

                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                    utility_functs = utility.utility(lx_box, ly_box)

        # Perform measurements after all time steps looped through
        if measurement_method == 'adsorption-final':
            if len(partPhase_time_arr)>1:
                if steady_state_once == 'True':
                    kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                    adsorption_dict = kinetic_functs.particle_flux_final(partPhase_time, time_entered_bulk, time_entered_gas, in_clust_arr, partPhase_time_arr, clust_size_arr, pos_x_arr_time, pos_y_arr_time, com_x_arr_time, com_y_arr_time, com_x_parts_arr_time, com_y_parts_arr_time)

                    data_output_functs.write_all_time_to_txt(adsorption_dict, dataPath + 'adsorption_final_' + outfile + '.txt')
        elif measurement_method == 'penetration':
            
            x = 1
            num_frames = 0
            bulk_press_num = np.zeros(len(typ0ind))
            bulk_press_sum = 0

            while x == 1:
                snap_temp = t[num_frames]                                 #Take current frame
                #Arrays of particle data
                pos_temp = snap_temp.particles.position               # position
                pos_temp[:,-1] = 0.0                             # 2D system

                typ = snap_temp.particles.typeid                 # Particle type
                typ0ind=np.where(typ==0)      # Calculate which particles are type 0
                typ1ind=np.where(typ==1)      # Calculate which particles are type 1
                

                if (action_arr[num_frames]=='gas') & (np.abs(pos_temp[typ1ind,0])>(np.amax(pos_temp[typ0ind,0])+r_cut)):
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                    stress_stat_dict, press_stat_dict, press_plot_dict = stress_and_pressure_functs.interparticle_pressure_nlist_system()
                    num_frames += 1
                    bulk_press_sum += press_plot_dict['all-A']['press']
                    bulk_press_num += 1
                else:
                    x = 0

            if x == 0:
                bulk_ss_press = np.mean(bulk_press_sum / bulk_press_num)
                bulk_ss_press_std = np.std(bulk_press_sum / bulk_press_num)

                vertical_shift = 0
                dify_long = 0
                start_dict = None
                for p in range(start, end):
                    j=int(p*time_step)

                    snap = t[j]                                 #Take current frame

                    #Arrays of particle data
                    pos = snap.particles.position               # position
                    pos[:,-1] = 0.0                             # 2D system
                    xy = np.delete(pos, 2, 1)

                    ori = snap.particles.orientation            #Orientation (quaternions)
                    ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]
                    x_orient_arr = np.array(list(map(utility_functs.quatToXOrient, ori))) # convert to [-pi, pi]
                    y_orient_arr = np.array(list(map(utility_functs.quatToYOrient, ori))) # convert to [-pi, pi]


                    typ = snap.particles.typeid                 # Particle type
                    typ0ind=np.where(typ==0)[0]      # Calculate which particles are type 0
                    typ1ind=np.where(typ==1)[0]      # Calculate which particles are type 1

                    tst = snap.configuration.step               # timestep
                    tst -= first_tstep                          # normalize by first timestep
                    tst *= dtau                                 # convert to Brownian time
                    
                    outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
                    out = outfile + "_frame_"
                    pad = str(j).zfill(5)
                    outFile = out + pad

                    if j>(start*time_step):
                        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, x_orient_arr, y_orient_arr)

                        penetration_dict, start_dict, vertical_shift, dify_long = particle_prop_functs.penetration_depth(start_dict, prev_pos, vertical_shift, dify_long)
                        stress_stat_dict, press_stat_dict, press_plot_dict = stress_and_pressure_functs.interparticle_pressure_nlist_system()
                        adjusted_press = np.array([])
                        if (penetration_dict['action']=='gas') & (np.abs(pos[typ1ind,0])>(np.amax(pos[typ0ind,0])+r_cut)):
                            press_dif = press_plot_dict['all-A']['press'] - bulk_ss_press
                            adjusted_press = np.append(press_dif, press_plot_dict['all-B']['press'])
                        else:
                            adjusted_press = press_plot_dict['all-all']['press'] - bulk_ss_press
                        press_plot_dict['all-A']['press'] = press_plot_dict['all-A']['press'] - bulk_ss_press
                        press_plot_dict['all-all']['press'] = adjusted_press
                        data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                        data_output_functs.write_to_txt(stress_stat_dict, dataPath + 'interparticle_stress_' + outfile + '.txt')
                        data_output_functs.write_to_txt(press_stat_dict, dataPath + 'interparticle_press_' + outfile + '.txt')
                        if plot == 'y':

                            #vp_part_arr = particle_prop_functs.virial_pressure_part(stress_stat_dict)

                            plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile, intPhi)


                            plotting_functs.interpart_press_map2(press_plot_dict, pos, prev_pos, ang)     

                    prev_pos = pos.copy()
                    prev_ang = ang.copy()
        
        elif measurement_method == 'radial-df':
            # Done!
            
            for key, value in radial_df_dict.items():
                if key != 'r':
                    avg_radial_df_dict[key] = (np.array(avg_radial_df_dict[key])/sum_num).tolist()
            lat_avg = lat_sum / sum_num

            lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, x_orient_arr, y_orient_arr, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

            avg_compress_dict = lattice_structure_functs.compressibility(avg_radial_df_dict, avg_num_dens = avg_num_dens_dict)

            avg_wasserstein_dict = lattice_structure_functs.wasserstein_distance(avg_radial_df_dict, lat_avg)

            data_output_functs.write_to_txt(avg_compress_dict, dataPath + 'avg_compressibility_' + outfile + '.txt')

            data_output_functs.write_to_txt(avg_wasserstein_dict, dataPath + 'avg_wasserstein_' + outfile + '.txt')

            data_output_functs.write_to_txt(avg_radial_df_dict, dataPath + 'avg_radial_df_' + outfile + '.txt')

            if plot == 'y':

                        plotting_functs.plot_general_rdf(radial_df_dict)
                        plotting_functs.plot_all_rdfs(radial_df_dict)

        elif measurement_method == 'gas-radial-df':
            # Done!
            
            for key, value in radial_df_dict.items():
                if key != 'r':
                    avg_radial_df_dict[key] = (np.array(avg_radial_df_dict[key])/sum_num).tolist()

            data_output_functs.write_to_txt(avg_radial_df_dict, dataPath + 'avg_gas_radial_df_' + outfile + '.txt')

            if plot == 'y':

                        plotting_functs.plot_general_rdf(avg_radial_df_dict)
                        plotting_functs.plot_all_rdfs(avg_radial_df_dict)

        elif measurement_method == 'part-velocity':
            if j>(start * time_step):
                
                vel_plot_dict = {'A': {'mag': time_velA_mag}, 'B': {'mag': time_velB_mag}}
                if plot == 'y':
                    vel_histo_dict, vel_histo_stat_dict = plotting_functs.vel_histogram(vel_plot_dict, dt_step, avg='True')
                    data_output_functs.write_to_txt(vel_histo_stat_dict, dataPath + 'velocity_histo_stat_' + outfile + '.txt')
                    np.savetxt(dataPath + 'velocity_histo_xA_' + outfile + '.csv', vel_histo_dict['A']['x'], delimiter=",")
                    np.savetxt(dataPath + 'velocity_histo_yA_' + outfile + '.csv', vel_histo_dict['A']['y'], delimiter=",")
                    np.savetxt(dataPath + 'velocity_histo_xB_' + outfile + '.csv', vel_histo_dict['B']['x'], delimiter=",")
                    np.savetxt(dataPath + 'velocity_histo_yB_' + outfile + '.csv', vel_histo_dict['B']['y'], delimiter=",")

        elif measurement_method == 'phase-velocity':
                    
            if j>(start*time_step):

                part_ang_vel_dict = particle_prop_functs.angular_velocity(ang_vel_dict['part'], phase_dict['part'])
                part_vel_dict, vel_phase_plot_dict = particle_prop_functs.velocity(vel_dict['part']['mag'], phase_dict['part'])

                data_output_functs.write_to_txt(part_ang_vel_dict, dataPath + 'angular_velocity_' + outfile + '.txt')
                data_output_functs.write_to_txt(part_vel_dict, dataPath + 'velocity_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.vel_histogram(vel_phase_plot_dict, dt_step, avg='True')

        
        elif measurement_method == 'collision':
            #DONE!

            # Calculate the rate of collisions between particles
            collision_plot_dict = {'all': time_clust_all_size, 'A': time_clust_A_size, 'B': time_clust_B_size}
            
            if plot == 'y':
                plotting_functs.cluster_size_histogram(collision_plot_dict, avg='True')
        elif measurement_method == 'clustering-coefficient':

            prob_stat_dict = {'neigh': prob_stat_dict['neigh'], 'bulk': {'AA': time_prob_AA_bulk/time_prob_num, 'BA': time_prob_BA_bulk/time_prob_num, 'AB': time_prob_AB_bulk/time_prob_num, 'BB': time_prob_BB_bulk/time_prob_num}}

            if plot == 'y':
                #plotting_functs.prob_histogram(prob_plot_dict, prob_stat_dict)
                plotting_functs.prob_histogram2(prob_plot_dict, prob_stat_dict, avg=True)