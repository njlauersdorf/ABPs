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
#sys.path.insert(0,hoomdPath)

from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage

import numpy as np
import matplotlib

print(hoomdPath)
if hoomdPath == '/nas/longleaf/home/njlauers/hoomd-blue/build':
    matplotlib.use('Agg')
else:
    pass

import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick

from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
#from symfit import parameters, variables, sin, cos, Fit
import plotting_utility
import binning
import phase_identification
import interface
import theory
import plotting
import utility
import measurement
import stress_and_pressure
import data_output
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

#Define colors for plots
yellow = ("#fdfd96")
green = ("#77dd77")
red = ("#ff6961")
purple = ("#cab2d6")
new_green = ("#39FF14")
new_brown = ("#b15928")

# Define Fourier series for fit


#Calculate analytical values
lat_theory = theory_functs.conForRClust(peNet, eps)
curPLJ = theory_functs.ljPress(lat_theory, peNet, eps)
phi_theory = theory_functs.latToPhi(lat_theory)
phi_g_theory = theory_functs.compPhiG(peNet, lat_theory)


#Calculate activity-softness dependent variables
lat=theory_functs.getLat(peNet,eps)
tauLJ=theory_functs.computeTauLJ(eps)
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
import itertools

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
                        'edge_begin'.center(15) + ' ' +\
                        'edge_end'.center(15) + ' ' +\
                        'NBin'.center(15) + '\n')
g.close()

outTxt_num_dens = 'Num_dens_' + outfile + '.txt'

g = open(outPath2+outTxt_num_dens, 'w+') # write file headings
g.write('tauB'.center(20) + ' ' +\
                        'sizeBin'.center(20) + ' ' +\
                        'clust_size'.center(20) + ' ' +\
                        'dense_area'.center(20) + ' ' +\
                        'dense_ndens'.center(20) + ' ' +\
                        'dense_ndens_std'.center(20) + ' ' +\
                        'bulk_area'.center(20) + ' ' +\
                        'bulk_ndens'.center(20) + ' ' +\
                        'bulk_ndens_std'.center(20) + ' ' +\
                        'int_area'.center(20) + ' ' +\
                        'int_ndens'.center(20) + ' ' +\
                        'int_ndens_std'.center(20) + ' ' +\
                        'bub_area'.center(20) + ' ' +\
                        'bub_ndens'.center(20) + ' ' +\
                        'bub_ndens_std'.center(20) + ' ' +\
                        'gas_area'.center(20) + ' ' +\
                        'gas_ndens'.center(20) + ' ' +\
                        'gas_ndens_std'.center(20) + '\n')
g.close()

with hoomd.open(name=inFile, mode='rb') as t:

    dumps = int(t.__len__())
    start = int(200/time_step)#205                                             # first frame to process
                                # get number of timesteps dumped
    end = int(dumps/time_step)-1                                             # final frame to process
    snap = t[0]                                             # Take first snap for box
    first_tstep = snap.configuration.step                   # First time step

    snap = t[1]                                             # Take first snap for box
    second_tstep = snap.configuration.step                   # First time step
    second_tstep -= first_tstep                          # normalize by first timestep
    dt_step = second_tstep * dtau                                 # convert to Brownian time

    # Get box dimensions
    box_data = snap.configuration.box

    l_box = box_data[0]                                     #box length
    h_box = l_box / 2.0                                     #half box length
    utility_functs = utility.utility(l_box)

    #2D binning of system
    NBins = utility_functs.getNBins(l_box, r_cut)
    sizeBin = utility_functs.roundUp((l_box / NBins), 6)
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)

    time_arr=np.zeros(dumps)                                  #time step array

    plotting_utility_functs = plotting_utility.plotting_utility(l_box, partNum, typ)


    dt=0

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
        ang = np.array(list(map(utility_functs.quatToAngle, ori))) # convert to [-pi, pi]

        typ = snap.particles.typeid                 # Particle type
        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1

        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        time_arr[j]=tst

        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
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
        clust_large = np.amax(clust_size)

        partTyp=np.zeros(partNum)
        partPhase=np.zeros(partNum)
        edgePhase=np.zeros(partNum)
        bulkPhase=np.zeros(partNum)

        com_dict = plotting_utility_functs.com_view(pos, clp_all)
        pos = com_dict['pos']

        #Bin system to calculate orientation and alignment that will be used in vector plots
        NBins = utility_functs.getNBins(l_box, bin_width)

        sizeBin = utility_functs.roundUp(((l_box) / NBins), 6)

        binning_functs = binning.binning(l_box, partNum, NBins, peA, peB, typ)

        pos_dict = binning_functs.create_bins()

        part_dict = binning_functs.bin_parts(pos, ids, clust_size)

        orient_dict = binning_functs.bin_orient(part_dict, pos, ang, com_dict['com'])
        area_frac_dict = binning_functs.bin_area_frac(part_dict)
        activ_dict = binning_functs.bin_activity(part_dict)

        fa_all_tot = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_x_tot = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_all_y_tot = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_fast_tot = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_slow_tot = [[0 for b in range(NBins)] for a in range(NBins)]

        fa_all_num = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_fast_num = [[0 for b in range(NBins)] for a in range(NBins)]
        fa_slow_num = [[0 for b in range(NBins)] for a in range(NBins)]

        align_dict = binning_functs.bin_align(orient_dict)

        #Time frame for plots
        pad = str(j).zfill(4)

        press_dict = binning_functs.bin_press(align_dict, area_frac_dict)

        #radial_density_function_analysis_binary_updates
        align_grad_dict = binning_functs.curl_and_div(align_dict)

        phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, l_box, partNum, NBins, peA, peB, parFrac, eps, typ)

        phase_dict = phase_ident_functs.phase_ident()

        phase_dict = phase_ident_functs.phase_blur(phase_dict)

        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

        count_dict = phase_ident_functs.phase_count(phase_dict)

        bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

        bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

        phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

        phase_dict = phase_ident_functs.update_phasePart(phase_dict)

        phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

        phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

        bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

        bulk_comp_dict = phase_ident_functs.bulk_sort2(bulk_comp_dict)

        int_comp_dict = phase_ident_functs.int_sort2(int_comp_dict)

        interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, l_box, partNum, NBins, peA, peB, parFrac, eps, typ, ang)

        surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

        #Save positions of external and internal edges
        clust_true = 0

        if j>(start*time_step):
            vel_dict = binning_functs.bin_vel(pos, prev_pos, part_dict, dt_step)
            ang_vel_dict = binning_functs.bin_ang_vel(ang, prev_ang, part_dict, dt_step)
            vel_grad = binning_functs.curl_and_div(vel_dict)

        surface2_pos_dict = interface_functs.surface_sort(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'])
        surface1_pos_dict = interface_functs.surface_sort(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'])


        surface_com_dict = interface_functs.surface_com(int_dict, int_comp_dict, surface_dict)

        surface_radius_bin = interface_functs.surface_radius_bins(int_dict, int_comp_dict, surface_dict, surface_com_dict)

        '''
        #Measures radius and width of each interface
        edge_width = []
        bub_width_ext = []
        bub_width_int = []
        bub_width = []

        id_step = 0

        #Loop over interfaces
        for m in range(0, len(bub_id_arr)):

            #Always true
            if if_bub_id_arr[m]==1:

                #Find which particles belong to mth interface structure
                edge_parts = np.where((edgePhase==bub_size_id_arr[m]))[0]

                #If particles belong to mth interface structure, continue...
                if len(edge_parts)>0:

                    #Initiate empty arrays
                    shortest_r=np.array([])
                    bub_rad_int=np.array([])
                    bub_rad_ext=np.array([])

                    #Find interior and exterior particles of interface
                    int_bub_id_tmp = np.where((edgePhase==bub_size_id_arr[m]) & (intedgePhase==1))[0]
                    ext_bub_id_tmp = np.where((edgePhase==bub_size_id_arr[m]) & (extedgePhase==1))[0]

                    #Calculate (x,y) center of mass of interface
                    x_com_bub = np.mean(pos[edge_parts,0]+h_box)
                    y_com_bub = np.mean(pos[edge_parts,1]+h_box)

                    #Loop over bins in system
                    for ix in range(0, len(occParts)):
                                for iy in range(0, len(occParts)):

                                    #If bin belongs to mth interface structure, continue...
                                    if edge_id[ix][iy]==bub_size_id_arr[m]:

                                        #If bin is an exterior particle of mth interface structure, continue...
                                        if ext_edge_id[ix][iy]==1:

                                            #Calculate (x,y) position of bin
                                            pos_box_x1 = (ix+0.5)*sizeBin
                                            pos_box_y1 = (iy+0.5)*sizeBin

                                            #Calculate x distance from mth interface structure's center of mass
                                            bub_rad_tmp_x = (pos_box_x1-x_com_bub)
                                            bub_rad_tmp_x_abs = np.abs(pos_box_x1-x_com_bub)

                                            #Enforce periodic boundary conditions
                                            if bub_rad_tmp_x_abs>=h_box:
                                                if bub_rad_tmp_x < -h_box:
                                                    bub_rad_tmp_x += l_box
                                                else:
                                                    bub_rad_tmp_x -= l_box

                                            #Calculate y distance from mth interface structure's center of mass
                                            bub_rad_tmp_y = (pos_box_y1-y_com_bub)
                                            bub_rad_tmp_y_abs = np.abs(pos_box_y1-y_com_bub)

                                            #Enforce periodic boundary conditions
                                            if bub_rad_tmp_y_abs>=h_box:
                                                if bub_rad_tmp_y < -h_box:
                                                    bub_rad_tmp_y += l_box
                                                else:
                                                    bub_rad_tmp_y -= l_box

                                            #Calculate magnitude of distance from center of mass of mth interface structure
                                            bub_rad_tmp = (bub_rad_tmp_x**2 + bub_rad_tmp_y**2)**0.5

                                            #Save this interface's radius to array
                                            bub_rad_ext = np.append(bub_rad_ext, bub_rad_tmp+(sizeBin/2))

                                        #If bin is interior particle of mth interface structure, continue
                                        if int_edge_id[ix][iy]==1:

                                            #Calculate (x,y) position of bin
                                            pos_box_x1 = (ix+0.5)*sizeBin
                                            pos_box_y1 = (iy+0.5)*sizeBin

                                            #Calculate x distance from mth interface structure's center of mass
                                            bub_rad_tmp_x = (pos_box_x1-x_com_bub)
                                            bub_rad_tmp_x_abs = np.abs(pos_box_x1-x_com_bub)

                                            #Enforce periodic boundary conditions
                                            if bub_rad_tmp_x_abs>=h_box:
                                                if bub_rad_tmp_x < -h_box:
                                                    bub_rad_tmp_x += l_box
                                                else:
                                                    bub_rad_tmp_x -= l_box

                                            #Calculate y distance from mth interface structure's center of mass
                                            bub_rad_tmp_y = (pos_box_y1-y_com_bub)
                                            bub_rad_tmp_y_abs = np.abs(pos_box_y1-y_com_bub)

                                            #Enforce periodic boundary conditions
                                            if bub_rad_tmp_y_abs>=h_box:
                                                if bub_rad_tmp_y < -h_box:
                                                    bub_rad_tmp_y += l_box
                                                else:
                                                    bub_rad_tmp_y -= l_box

                                            #Calculate magnitude of distance to mth interface structure's center of mass
                                            bub_rad_tmp = (bub_rad_tmp_x**2 + bub_rad_tmp_y**2)**0.5

                                            #Save this interface's interior radius to array
                                            bub_rad_int = np.append(bub_rad_int, bub_rad_tmp+(sizeBin/2))
                    #if there were interior bins found, calculate the average interior radius of mth interface structure
                    if len(bub_rad_int)>0:
                        bub_width_int.append(np.mean(bub_rad_int)+sizeBin/2)
                    else:
                        bub_width_int.append(0)

                    #if there were exterior bins found, calculate the average exterior radius of mth interface structure
                    if len(bub_rad_ext)>0:
                        bub_width_ext.append(np.mean(bub_rad_ext)+sizeBin/2)
                    else:
                        bub_width_ext.append(0)

                    #Use whichever is larger to calculate the true radius of the mth interface structure
                    if bub_width_ext[id_step]>bub_width_int[id_step]:
                        bub_width.append(bub_width_ext[id_step])
                    else:
                        bub_width.append(bub_width_int[id_step])

                    #If both interior and exterior particles were identified, continue...
                    if (len(int_bub_id_tmp)>0) & (len(ext_bub_id_tmp)>0):

                            #Loop over bins in system
                            for ix in range(0, len(occParts)):
                                for iy in range(0, len(occParts)):

                                    #If bin is part of mth interface structure, continue...
                                    if edge_id[ix][iy]==bub_size_id_arr[m]:

                                        #If bin is an exterior bin of mth interface structure, continue...
                                        if ext_edge_id[ix][iy]==1:

                                            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                                            difr_short=10000000.

                                            #Calculate position of exterior edge bin
                                            pos_box_x1 = (ix+0.5)*sizeBin
                                            pos_box_y1 = (iy+0.5)*sizeBin

                                            #Loop over bins of system
                                            for ix2 in range(0, len(occParts)):
                                                for iy2 in range(0, len(occParts)):

                                                    #If bin belongs to mth interface structure, continue...
                                                    if edge_id[ix2][iy2]==bub_size_id_arr[m]:

                                                        #If bin is an interior edge bin for mth interface structure, continue...
                                                        if int_edge_id[ix2][iy2]==1:

                                                                #Calculate position of interior edge bin
                                                                pos_box_x2 = (ix2+0.5)*sizeBin
                                                                pos_box_y2 = (iy2+0.5)*sizeBin

                                                                #Calculate distance from interior edge bin to exterior edge bin
                                                                difr = ( (pos_box_x1-pos_box_x2)**2 + (pos_box_y1-pos_box_y2)**2)**0.5

                                                                #If this distance is the shortest calculated thus far, replace the value with it
                                                                if difr<difr_short:
                                                                    difr_short=difr
                                            #Save each shortest distance to an interior edge bin calculated for each exterior edge bin
                                            shortest_r = np.append(shortest_r, difr_short)

                            #Calculate and save the average shortest-distance between each interior edge and exterior edge bins for the mth interface structure
                            edge_width.append(np.mean(shortest_r)+sizeBin)

                    #If both an interior and exterior edge were not identified, save the cluster radius instead for the edge width
                    else:
                        edge_width.append(bub_width[id_step])

                    #Step for number of bins with identified edge width
                    id_step +=1

                #If no particles in interface, save zeros for radius and width
                else:
                    edge_width.append(0)
                    bub_width.append(0)

            #Never true
            else:
                edge_width.append(0)
                bub_width.append(0)
        '''

        bin_count_dict = phase_ident_functs.phase_bin_count(phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict)


        #interface = interface.interface()

        #Slow/fast composition of bulk phase

        part_count_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)

        #Colors for plotting each phase
        yellow = ("#fdfd96")        #Largest gas-dense interface
        green = ("#77dd77")         #Bulk phase
        red = ("#ff6961")           #Gas phase
        purple = ("#cab2d6")        #Bubble or small gas-dense interfaces

        #label previous positions for velocity calculation
        prev_pos = pos.copy()
        prev_ang = ang.copy()
        '''
        #Output general calculations/information for each phase
        g = open(outPath2+outTxt_phase_info, 'a')
        g.write('{0:.2f}'.format(tst).center(15) + ' ')
        g.write('{0:.6f}'.format(sizeBin).center(15) + ' ')
        g.write('{0:.0f}'.format(np.amax(clust_size)).center(15) + ' ')
        g.write('{0:.0f}'.format(slow_bulk_num).center(15) + ' ')
        g.write('{0:.0f}'.format(fast_bulk_num).center(15) + ' ')
        g.write('{0:.0f}'.format(bulkBin).center(15) + ' ')
        g.write('{0:.0f}'.format(slow_gas_num).center(15) + ' ')
        g.write('{0:.0f}'.format(fast_gas_num).center(15) + ' ')
        g.write('{0:.0f}'.format(gasBin).center(15) + ' ')
        g.write('{0:.0f}'.format(slow_int_num).center(15) + ' ')
        g.write('{0:.0f}'.format(fast_int_num).center(15) + ' ')
        g.write('{0:.0f}'.format(intBin).center(15) + ' ')
        g.write('{0:.0f}'.format(slow_bub_num).center(15) + ' ')
        g.write('{0:.0f}'.format(fast_bub_num).center(15) + ' ')
        g.write('{0:.0f}'.format(bubBin2-intBin).center(15) + '\n')
        g.close()
        '''

        #Bin system to calculate orientation and alignment that will be used in vector plots
        all_surface_curves = {}
        sep_surface_dict = interface_functs.separate_surfaces(surface_dict, int_dict, int_comp_dict)
        all_surface_measurements = {}

        for m in range(0, len(sep_surface_dict)):

            key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))
            all_surface_curves[key] = {}
            all_surface_measurements[key] = {}
            print(key)
            if sep_surface_dict[key]['interior']['num']>0:
                sort_interior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['interior'])
                all_surface_curves[key]['interior'] = interface_functs.surface_curve_interp(sort_interior_ids)

                com_pov_interior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['interior']['pos'])

                all_surface_measurements[key]['interior'] = interface_functs.surface_radius(com_pov_interior_pos['pos'])
                all_surface_measurements[key]['interior']['surface area'] = interface_functs.surface_area(com_pov_interior_pos['pos'])
                all_surface_measurements[key]['interior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['interior'])

            if sep_surface_dict[key]['exterior']['num']>0:
                sort_exterior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['exterior'])
                all_surface_curves[key]['exterior'] = interface_functs.surface_curve_interp(sort_exterior_ids)

                com_pov_exterior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['exterior']['pos'])
                all_surface_measurements[key]['exterior'] = interface_functs.surface_radius(com_pov_exterior_pos['pos'])
                all_surface_measurements[key]['exterior']['surface area'] = interface_functs.surface_area(com_pov_exterior_pos['pos'])
                all_surface_measurements[key]['exterior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['exterior'])

            if (sep_surface_dict[key]['exterior']['num']>0) & sep_surface_dict[key]['interior']['num']>0:
                all_surface_measurements[key]['exterior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                all_surface_measurements[key]['interior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])


        method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
        method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)
        method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)

        plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)
        #plotting.plot_phases(pos, part_count_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_all_density(area_frac_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_type_A_density(area_frac_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_type_B_density(area_frac_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_dif_density(area_frac_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_type_B_frac(area_frac_dict, all_surface_curves, int_comp_dict)
        #plotting.plot_dif_frac(area_frac_dict, all_surface_curves, int_comp_dict)

        lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

        lat_stat_dict, lat_plot_dict = lattice_structure_functs.lattice_spacing(bulk_dict)
        data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
        data_output_functs.write_to_txt(lat_stat_dict, '/Volumes/EXTERNAL2/test44.txt')
        stop
        #plotting_functs.lat_histogram(lat_plot_dict)
        #radial_df_dict = lattice_structure_functs.radial_df()

        neigh_stat_dict, neigh_plot_dict = lattice_structure_functs.nearest_neighbors()
        plotting_functs.plot_all_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
        plotting_functs.plot_all_neighbors_of_A_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
        plotting_functs.plot_all_neighbors_of_B_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
        plotting_functs.plot_A_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
        plotting_functs.plot_B_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
        stop
        #plotting_functs.plot_general_rdf(radial_df_dict)
        #plotting_functs.plot_all_rdfs(radial_df_dict)
        angular_df_dict = lattice_structure_functs.angular_df()
        stop
        data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
        data_output_functs.write_data(lat_stat_dict, '/Volumes/External/test44.csv')
        #stop
        #stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

        #stress_dict = stress_and_pressure_functs.interparticle_stress()

        #press_dict = stress_and_pressure_functs.virial_pressure(stress_dict)
        #shear_dict = stress_and_pressure.shear_stress(stress_dict)

        #part_force_dict = measurement_functs.particle_active_forces()
        #com_radial_dict = stress_and_pressure_functs.radial_active_force_pressure(part_force_dict)
        #act_press_dict = stress_and_pressure_functs.total_active_pressure(com_radial_dict)


        """
        plotting.plot_all_align(align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_all_align(method1_align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_all_align(method2_align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_A_align(align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_A_align(method1_align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_A_align(method2_align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_B_align(align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_B_align(method1_align_dict, all_surface_curves, int_comp_dict)
        plotting.plot_type_B_align(method2_align_dict, all_surface_curves, int_comp_dict)

        stop


        #Output calculations/information for each interface structure
        g = open(outPath2+outTxt_bub_info, 'a')
        for m in range(0, len(bub_size_id_arr)):
            if if_bub_id_arr[m]!=0:
                g.write('{0:.2f}'.format(tst).center(15) + ' ')
                g.write('{0:.6f}'.format(sizeBin).center(15) + ' ')
                g.write('{0:.0f}'.format(np.amax(clust_size)).center(15) + ' ')
                g.write('{0:.0f}'.format(interface_id).center(15) + ' ')
                g.write('{0:.0f}'.format(bub_size_id_arr[m]).center(15) + ' ')
                g.write('{0:.0f}'.format(bub_slow_arr[m]).center(15) + ' ')
                g.write('{0:.0f}'.format(bub_fast_arr[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(bub_width[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(bub_width_sd[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(surface_area_ext[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(surface_area_int[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(edge_width_arr[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(edge_width_arr_sd[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(edge_width_begin_arr[m]).center(15) + ' ')
                g.write('{0:.6f}'.format(edge_width_end_arr[m]).center(15) + ' ')
                g.write('{0:.0f}'.format(bubBin[m]).center(15) + '\n')
        g.close()

        binArea = sizeBin**2

        #IDs of particles participating in each phase
        edge_id_plot = np.where(edgePhase==interface_id)[0]     #Largest gas-dense interface
        int_id_plot = np.where(partPhase==1)[0]         #All interfaces
        bulk_int_id_plot = np.where(partPhase!=2)[0]

        if len(bulk_ids)>0:
            bub_id_plot = np.where((edgePhase!=interface_id) & (edgePhase!=bulk_id))[0]     #All interfaces excluding the largest gas-dense interface
        else:
            bub_id_plot = []
        gas_id = np.where(partPhase==2)[0]

        #Determine positions of particles in each phase
        bulk_int_pos = pos[bulk_int_id_plot]
        bulk_pos = pos[bulk_id_plot]
        int_pos = pos[edge_id_plot]

        #Initiate zero values
        bulk_num_dens_sum = 0
        bulk_num_dens_num = 0
        int_num_dens_sum = 0
        int_num_dens_num = 0
        bub_num_dens_sum = 0
        bub_num_dens_num = 0
        gas_num_dens_sum = 0
        gas_num_dens_num = 0
        dense_num_dens_sum = 0
        dense_num_dens_num = 0

        #Calculate components for mean of number density per phase
        for ix in range(0, len(phaseBin)):
            for iy in range(0, len(phaseBin)):
                if phaseBin[ix][iy]==0:
                    bulk_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    bulk_num_dens_num +=1
                elif edge_id[ix][iy]==interface_id:
                    int_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    int_num_dens_num +=1
                elif (edge_id[ix][iy]!=bulk_id) & (edge_id[ix][iy]!=interface_id):
                    bub_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    bub_num_dens_num +=1
                elif phaseBin[ix][iy]==2:
                    gas_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    gas_num_dens_num +=1
                elif phaseBin[ix][iy]!=2:
                    dense_num_dens_sum += len(binParts[ix][iy])/sizeBin**2
                    dense_num_dens_num +=1

        #Calculate mean number density per phase
        if bulk_num_dens_num > 0:
            bulk_avg = (bulk_num_dens_sum/bulk_num_dens_num)
        else:
            bulk_avg = 0
        if int_num_dens_num > 0:
            int_avg = (int_num_dens_sum/int_num_dens_num)
        else:
            int_avg = 0
        if bub_num_dens_num > 0:
            bub_avg = (bub_num_dens_sum/bub_num_dens_num)
        else:
            bub_avg = 0
        if gas_num_dens_num > 0:
            gas_avg = (gas_num_dens_sum/gas_num_dens_num)
        else:
            gas_avg = 0
        if dense_num_dens_num > 0:
            dense_avg = (dense_num_dens_sum/dense_num_dens_num)
        else:
            dense_avg = 0

        #Calculate area of each phase
        bulk_area = bulk_num_dens_num * sizeBin**2
        int_area = int_num_dens_num * sizeBin**2
        bub_area = bub_num_dens_num * sizeBin**2
        gas_area = gas_num_dens_num * sizeBin**2
        dense_area = dense_num_dens_num * sizeBin**2

        #Initiate empty values for standard deviation calculation
        bulk_num_dens_std_sum = 0
        bulk_num_dens_std_num = 0
        int_num_dens_std_sum = 0
        int_num_dens_std_num = 0
        bub_num_dens_std_sum = 0
        bub_num_dens_std_num = 0
        gas_num_dens_std_sum = 0
        gas_num_dens_std_num = 0
        dense_num_dens_std_sum = 0
        dense_num_dens_std_num = 0

        #Calculate components for standard deviations of number density per phase
        for ix in range(0, len(phaseBin)):
            for iy in range(0, len(phaseBin)):
                if phaseBin[ix][iy]==0:
                    bulk_num_dens_std_sum += ((len(binParts[ix][iy])/sizeBin**2) - (bulk_avg))**2
                    bulk_num_dens_std_num +=1
                elif edge_id[ix][iy]==interface_id:
                    int_num_dens_std_sum += ((len(binParts[ix][iy])/sizeBin**2) - (int_avg))**2
                    int_num_dens_std_num +=1
                elif (edge_id[ix][iy]!=bulk_id) & (edge_id[ix][iy]!=interface_id):
                    bub_num_dens_std_sum += ((len(binParts[ix][iy])/sizeBin**2) - (bub_avg))**2
                    bub_num_dens_std_num +=1
                elif phaseBin[ix][iy]==2:
                    gas_num_dens_std_sum += ((len(binParts[ix][iy])/sizeBin**2) - (gas_avg))**2
                    gas_num_dens_std_num +=1

                if phaseBin[ix][iy]!=2:
                    dense_num_dens_std_sum += ((len(binParts[ix][iy])/sizeBin**2) - (dense_avg))**2
                    dense_num_dens_std_num +=1


        #Calculate standard deviation of number density for each phase
        if bulk_num_dens_std_num > 0:
            bulk_std = (bulk_num_dens_std_sum/bulk_num_dens_std_num)**0.5
        else:
            bulk_std = 0
        if int_num_dens_std_num > 0:
            int_std = (int_num_dens_std_sum/int_num_dens_std_num)**0.5
        else:
            int_std = 0
        if bub_num_dens_std_num > 0:
            bub_std = (bub_num_dens_std_sum/bub_num_dens_std_num)**0.5
        else:
            bub_std = 0
        if gas_num_dens_std_num > 0:
            gas_std = (gas_num_dens_std_sum/gas_num_dens_std_num)**0.5
        else:
            gas_std = 0
        if dense_num_dens_std_num > 0:
            dense_std = (dense_num_dens_std_sum/dense_num_dens_std_num)**0.5
        else:
            dense_std = 0

        #Output means and standard deviations of number density for each phase
        g = open(outPath2+outTxt_num_dens, 'a')
        g.write('{0:.2f}'.format(tst).center(20) + ' ')
        g.write('{0:.6f}'.format(sizeBin).center(20) + ' ')
        g.write('{0:.0f}'.format(np.amax(clust_size)).center(20) + ' ')
        g.write('{0:.6f}'.format(dense_area).center(20) + ' ')
        g.write('{0:.6f}'.format(dense_avg).center(20) + ' ')
        g.write('{0:.6f}'.format(dense_std).center(20) + ' ')
        g.write('{0:.6f}'.format(bulk_area).center(20) + ' ')
        g.write('{0:.6f}'.format(bulk_avg).center(20) + ' ')
        g.write('{0:.6f}'.format(bulk_std).center(20) + ' ')
        g.write('{0:.6f}'.format(int_area).center(20) + ' ')
        g.write('{0:.6f}'.format(int_avg).center(20) + ' ')
        g.write('{0:.6f}'.format(int_std).center(20) + ' ')
        g.write('{0:.6f}'.format(bub_area).center(20) + ' ')
        g.write('{0:.6f}'.format(bub_avg).center(20) + ' ')
        g.write('{0:.6f}'.format(bub_std).center(20) + ' ')
        g.write('{0:.6f}'.format(gas_area).center(20) + ' ')
        g.write('{0:.6f}'.format(gas_avg).center(20) + ' ')
        g.write('{0:.6f}'.format(gas_std).center(20) + '\n')
        g.close()


        #Contour plot of the number density of all particles per bin
        '''
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.0
        max_n = np.max(num_dens3)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(pos_box_x, pos_box_y, num_dens3, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$n$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        plt.close()
        '''


        div_max = 3
        div_min_n = -2
        div_max_n = 2
        min_n2 = 0
        max_n2 = 0.75


        min_press_grad = 0.0
        max_press_grad = 0.2
        min_press = 0.0
        max_press = 0.5

        curl_min = -3
        curl_max = 3
        Cmag_max=10**2
        Cmag_min=10**0

        Cmag_max2=4
        Cmag_min2=-4
        mag_max=2
        mag_min=0
        new_green = '#39FF14'
        '''


        #Contour plot of the number density of type A particles per bin
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(pos_box_x, pos_box_y, num_dens3A, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$n_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')

        plt.tight_layout()
        plt.savefig(outPath + 'num_densA_' + out + pad + ".png", dpi=100)
        plt.close()

        #Contour plot of the number density of type B particles per bin
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(pos_box_x, pos_box_y, num_dens3B, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$n_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'num_densB_' + out + pad + ".png", dpi=100)
        plt.close()
        '''


        min_n = 0.0#np.min(fast_frac_arr)
        max_n = 1.0#np.max(fast_frac_arr)

        empt_arr = np.ones(np.shape(empt_arr))
        import random
        for ix in range(0, len(empt_arr)):
            for iy in range(0, len(empt_arr)):
                test = random.randint(1,20)
                if test ==1:
                    empt_arr[ix][iy] = 1
                else:
                    empt_arr[ix][iy] = 0


        filtered_cmap = view_colormap('inferno')
        '''
        #Contour plot of the difference in number density of type B to type A per bin
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)

        im = plt.contourf(pos_box_x, pos_box_y, fast_frac_arr, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        #im = plt.contourf(pos_box_x, pos_box_y, fast_frac_arr, level_boundaries, vmin=min_n, vmax=max_n, cmap=my_cmap, extend='both')

        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        #im = plt.contourf(pos_box_x, pos_box_y, fast_frac_arr, level_boundaries, vmin=min_n, vmax=max_n, cmap='inferno', extend='both')
        plt.contourf(pos_box_x, pos_box_y, empt_arr, alpha=0.3, cmap='Greys', vmin=0.0, vmax=1.0, extend='both')


        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm = plt.cm.ScalarMappable(norm=norm, cmap = filtered_cmap)

        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        clb.set_label(r'$\chi_\mathrm{F}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'fast_frac_' + out + pad + ".png", dpi=100)
        plt.close()
        '''

        min_n = 0.3#np.min(fast_frac_arr)
        max_n = 0.7#np.max(fast_frac_arr)

        fig = plt.figure(figsize=(8.5,8))
        ax = fig.add_subplot(111)

        myEps = [1., 0.1, 0.01, 0.001, 0.0001]
        plt.scatter(pos[bulk_id_plot,0]+h_box, pos[bulk_id_plot,1]+h_box, s=0.75, marker='.', c=green)
        plt.scatter(pos[gas_id,0]+h_box, pos[gas_id,1]+h_box, s=0.75, marker='.', c=red)
        plt.scatter(pos[edge_id_plot,0]+h_box, pos[edge_id_plot,1]+h_box, s=0.75, marker='.', c=yellow)

        if len(bub_id_plot)>0:
            plt.scatter(pos[bub_id_plot,0]+h_box, pos[bub_id_plot,1]+h_box, s=0.75, marker='.', c=purple)
        '''
        if len(bub1_parts)>0:
            plt.scatter(pos[bub1_parts,0]+h_box, pos[bub1_parts,1]+h_box, s=0.75, marker='.', c=purple)
        if len(bub2_parts)>0:
            plt.scatter(pos[bub2_parts,0]+h_box, pos[bub2_parts,1]+h_box, s=0.75, marker='.', c=purple)
        if len(bub3_parts)>0:
            plt.scatter(pos[bub3_parts,0]+h_box, pos[bub3_parts,1]+h_box, s=0.75, marker='.', c=purple)
        if len(bub4_parts)>0:
            plt.scatter(pos[bub4_parts,0]+h_box, pos[bub4_parts,1]+h_box, s=0.75, marker='.', c=purple)
        if len(bub5_parts)>0:
            plt.scatter(pos[bub5_parts,0]+h_box, pos[bub5_parts,1]+h_box, s=0.75, marker='.', c=purple)
        '''

        plt.quiver(pos_box_x_plot, pos_box_y_plot, p_plot_x, p_plot_y)
        plt.xticks(pos_box_start)
        plt.yticks(pos_box_start)
        plt.tick_params(
                                axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelleft=False)
        plt.tick_params(
                                axis='y',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                right=False,      # ticks along the bottom edge are off
                                left=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelleft=False)

        plt.ylim((0, l_box))
        plt.xlim((0, l_box))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.77, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18,transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        eps_leg=[]
        mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
        msz=40
        red_patch = mpatches.Patch(color=red, label='Dilute')
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        purple_patch = mpatches.Patch(color=purple, label='Bubble')
        plt.legend(handles=[green_patch, yellow_patch, red_patch, purple_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=16, loc='upper left',labelspacing=0.1, handletextpad=0.1)
        plt.tight_layout()
        plt.savefig(outPath + 'interface_acc_' + out + pad + ".png", dpi=100)
        plt.close()



        '''
        if np.abs(np.max(num_densDif)) > np.abs(np.min(num_densDif)):
            min_n = -np.abs(np.max(num_densDif))
            max_n = np.abs(np.max(num_densDif))
        else:
            min_n = -np.abs(np.min(num_densDif))
            max_n = np.abs(np.min(num_densDif))

        #Contour plot of the difference in number density of type B to type A per bin
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)

        im = plt.contourf(pos_box_x, pos_box_y, num_densDif, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        if peB >= peA:
            clb.set_label(r'$n_\mathrm{B}-n_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)
        else:
            clb.set_label(r'$n_\mathrm{A}-n_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'num_densDif_' + out + pad + ".png", dpi=100)
        plt.close()

        #Plot the magnitude of the net active force normal to nearest interfacial surface (positive is toward, negative is away)
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -np.max(np.abs(fa_all_tot))
        max_n = np.max(np.abs(fa_all_tot))
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(pos_box_x, pos_box_y, fa_all_tot, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        #Plot interior and exterior interface surfaces
        if bub_large >=1:
            if interior_bin>0:
                plt.scatter(xn_pos, yn_pos, c='black', s=3.0)
            if exterior_bin>0:
                plt.scatter(xn2_pos, yn2_pos, c='black', s=3.0)

        if bub_large >=2:
            if interior_bin_bub1>0:
                plt.scatter(xn_bub2_pos, yn_bub2_pos, c='black', s=3.0)
            if exterior_bin_bub1>0:
                plt.scatter(xn2_bub2_pos, yn2_bub2_pos, c='black', s=3.0)
        if bub_large >=3:
            if interior_bin_bub2>0:
                plt.scatter(xn_bub3_pos, yn_bub3_pos, c='black', s=3.0)
            if exterior_bin_bub2>0:
                plt.scatter(xn2_bub3_pos, yn2_bub3_pos, c='black', s=3.0)

        if bub_large >=4:
            if interior_bin_bub3>0:
                plt.scatter(xn_bub4_pos, yn_bub4_pos, c='black', s=3.0)
            if exterior_bin_bub3>0:
                plt.scatter(xn2_bub4_pos, yn2_bub4_pos, c='black', s=3.0)
        if bub_large >=5:
            if interior_bin_bub4>0:
                plt.scatter(xn_bub5_pos, yn_bub5_pos, c='black', s=3.0)
            if exterior_bin_bub4>0:
                plt.scatter(xn2_bub5_pos, yn2_bub5_pos, c='black', s=3.0)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$F^\mathrm{a}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, l_box)
        plt.ylim(0, l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(outPath + 'fa_' + out + pad + ".png", dpi=100)
        plt.close()

        sz = 0.75

        typ0ind=np.where(snap.particles.typeid==0)[0]       # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)[0]       # Calculate which particles are type 1
        #Local each particle's positions
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind]

        # Create frame pad for images
        pad = str(j).zfill(4)

        #Plot each particle as a point color-coded by activity and labeled by their activity
        fig = plt.figure(figsize=(6.5,6))
        ax = fig.add_subplot(111)




        sz = 0.75
        num_range = int(l_box / sizeBin)
        x_span = np.zeros(NBins+1)
        val = -h_box
        for p in range(0, len(x_span)):
            x_span[p] = val + (p * sizeBin)

        #x_span = np.linspace(-h_box, h_box, num=NBins, dtype=float)

        #Assign type 0 particles to plot
        if peA!=peB:

            ells0 = [Ellipse(xy=pos0[i,:],
                    width=sz, height=sz, label='PeA: '+str(peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=pos1[i,:],
                    width=sz, height=sz, label='PeB: '+str(peB))
            for i in range(0,len(typ1ind))]

            # Plot position colored by neighbor number
            if peA <= peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            #Label time step
            ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst) + ' ' + r'$\tau_\mathrm{r}$',
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=18,
                    bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #Set axes parameters
            ax.set_xlim(-h_box, h_box)
            ax.set_ylim(-h_box, h_box)
            ax.axes.set_xticks([])
            ax.axes.set_yticks([])
            ax.axes.set_xticklabels([])
            ax.axes.set_yticks([])
            ax.set_aspect('equal')
            for y in range(0, len(x_span)):
                plt.plot(np.array([x_span[y], x_span[y]]), np.array([-h_box, h_box]), color='black', linestyle='-')
                plt.plot(np.array([-h_box, h_box]), np.array([x_span[y], x_span[y]]), color='black', linestyle='-')

            #Create legend for binary system
            if parFrac<100.0:
                leg = ax.legend(handles=[ells0[0], ells1[1]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(peA)), r'$\mathrm{Pe}_\mathrm{B} = $'+str(int(peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                if peA <= peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peA)), r'$\mathrm{Pe} = $'+str(int(peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
            plt.tight_layout()
            plt.savefig(outPath+ 'sim_frame_arrows_' + out + pad + ".png", dpi=150, transparent=False)
            plt.close()

        else:

            # Create frame pad for images
            pad = str(j).zfill(4)

            #Plot each particle as a point color-coded by activity and labeled by their activity
            fig = plt.figure(figsize=(6.5,6))
            ax = fig.add_subplot(111)




            sz = 0.75
            #Assign type 0 particles to plot
            ells0 = [Ellipse(xy=pos[i,:],
                    width=sz, height=sz, label='Pe: '+str(peA))
            for i in range(0,len(pos))]

            # Plot position colored by neighbor number
            slowGroup = mc.PatchCollection(ells0, facecolors=fastCol)
            ax.add_collection(slowGroup)

            #Label time step
            ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*tst) + ' ' + r'$\tau_\mathrm{r}$',
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=18,
                    bbox=dict(facecolor=(1,1,1,0.8), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #Set axes parameters
            ax.set_xlim(-h_box, h_box)
            ax.set_ylim(-h_box, h_box)
            ax.axes.set_xticks([])
            ax.axes.set_yticks([])
            ax.axes.set_xticklabels([])
            ax.axes.set_yticks([])
            ax.set_aspect('equal')
            plt.quiver(bin_pos_x, bin_pos_y, align_avg_x, align_avg_y)
            for y in range(0, len(x_span)):
                plt.plot(np.array([x_span[y], x_span[y]]), np.array([-h_box, h_box]), color='black', linestyle='-')
                plt.plot(np.array([-h_box, h_box]), np.array([x_span[y], x_span[y]]), color='black', linestyle='-')


            leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
            leg.legendHandles[0].set_color(fastCol)
            plt.tight_layout()
            plt.savefig(outPath+ 'sim_frame_arrows_' + out + pad + ".png", dpi=150, transparent=False)
            plt.close()
        '''
        """
