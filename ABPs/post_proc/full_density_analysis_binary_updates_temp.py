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
hoomdPath=str(sys.argv[2])
outPath = str(sys.argv[3])

#outPath2=str(sys.argv[11])
#outPath=str(sys.argv[12])

# Add hoomd location to Path
#sys.path.insert(0,hoomdPath)

from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage
import datetime

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
import particles
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
    intPhi = int(phi)
    phi /= 100.
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
bin_width = float(sys.argv[10])
time_step = float(sys.argv[11])
measurement_method = str(sys.argv[12])
plot = str(sys.argv[13])

outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(int(intPhi))+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile + "_frame_"

dataPath = outPath + '_txt_files/'
picPath = outPath + '_pic_files/'

import time
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

        if clust_large >= min_size:
            start_time = time.time()
            partTyp=np.zeros(partNum)
            partPhase=np.zeros(partNum)
            edgePhase=np.zeros(partNum)
            bulkPhase=np.zeros(partNum)

            com_dict = plotting_utility_functs.com_view(pos, clp_all)
            pos = com_dict['pos']

            #Bin system to calculate orientation and alignment that will be used in vector plots
            NBins = utility_functs.getNBins(l_box, bin_width)

            sizeBin = utility_functs.roundUp(((l_box) / NBins), 6)

            binning_functs = binning.binning(l_box, partNum, NBins, peA, peB, typ, eps)

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

            press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)

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
                    print(sort_interior_ids)

                    all_surface_curves[key]['interior'] = interface_functs.surface_curve_interp(sort_interior_ids)

                    com_pov_interior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['interior']['pos'])

                    all_surface_measurements[key]['interior'] = interface_functs.surface_radius(com_pov_interior_pos['pos'])
                    all_surface_measurements[key]['interior']['surface area'] = interface_functs.surface_area(com_pov_interior_pos['pos'])
                    #all_surface_measurements[key]['interior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['interior'])

                if sep_surface_dict[key]['exterior']['num']>0:
                    sort_exterior_ids = interface_functs.sort_surface_points(sep_surface_dict[key]['exterior'])

                    all_surface_curves[key]['exterior'] = interface_functs.surface_curve_interp(sort_exterior_ids)

                    com_pov_exterior_pos = interface_functs.surface_com_pov(all_surface_curves[key]['exterior']['pos'])
                    all_surface_measurements[key]['exterior'] = interface_functs.surface_radius(com_pov_exterior_pos['pos'])
                    all_surface_measurements[key]['exterior']['surface area'] = interface_functs.surface_area(com_pov_exterior_pos['pos'])
                    #all_surface_measurements[key]['exterior']['fourier'] = interface_functs.fourier_analysis(all_surface_measurements[key]['exterior'])

                if (sep_surface_dict[key]['exterior']['num']>0) & sep_surface_dict[key]['interior']['num']>0:
                    all_surface_measurements[key]['exterior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])
                    all_surface_measurements[key]['interior']['surface width'] = interface_functs.surface_width(all_surface_measurements[key]['interior']['mean radius'], all_surface_measurements[key]['exterior']['mean radius'])


            method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
            method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)
            method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)

            if measurement_method == 'activity':

                plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                if plot == 'y':
                    plotting_functs.plot_part_activity(pos, all_surface_curves, int_comp_dict)

            if measurement_method == 'phases':

                plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                if plot == 'y':
                    plotting_functs.plot_phases(pos, part_count_dict, all_surface_curves, int_comp_dict)

            elif measurement_method == 'number_density':

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_all_density(area_frac_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_A_density(area_frac_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_B_density(area_frac_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_dif_density(area_frac_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_B_frac(area_frac_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_dif_frac(area_frac_dict, all_surface_curves, int_comp_dict)

            elif measurement_method == 'com_alignment':

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_all_align(method1_align_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_A_align(method1_align_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_B_align(method1_align_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_dif_align(method1_align_dict, all_surface_curves, int_comp_dict)
            elif measurement_method == 'surface_alignment':

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_all_align(method2_align_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_A_align(method2_align_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_type_B_align(method2_align_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_dif_align(method1_align_dict, all_surface_curves, int_comp_dict)


            elif measurement_method == 'lattice_spacing':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                lat_stat_dict, lat_plot_dict = lattice_structure_functs.lattice_spacing(bulk_dict)

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)

                data_output_functs.write_to_txt(lat_stat_dict, dataPath + 'lattice_spacing_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.lat_histogram(lat_plot_dict)

                    if j>(start*time_step):
                        plotting_functs.lat_map(lat_plot_dict, all_surface_curves, int_comp_dict, velocity_dict = vel_dict)
                    else:
                        plotting_functs.lat_map(lat_plot_dict, all_surface_curves, int_comp_dict)

            elif measurement_method == 'radial_df':
                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                radial_df_dict = lattice_structure_functs.radial_df()

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(radial_df_dict, dataPath + 'radial_df_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_general_rdf(radial_df_dict)
                    plotting_functs.plot_all_rdfs(radial_df_dict)

            elif measurement_method == 'angular_df':
                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                angular_df_dict = lattice_structure_functs.angular_df()

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(angular_df_dict, dataPath + 'angular_df_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_general_adf(angular_df_dict)
                    plotting_functs.plot_all_adfs(angular_df_dict)

            elif measurement_method == 'neighbors':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                neigh_stat_dict, neigh_plot_dict = lattice_structure_functs.nearest_neighbors()

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(neigh_stat_dict, dataPath + 'nearest_neighbors_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_all_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_all_neighbors_of_A_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_all_neighbors_of_B_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_A_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    plotting_functs.plot_B_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)

            elif measurement_method == 'interparticle_pressure':
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                stress_plot_dict, stress_stat_dict = stress_and_pressure_functs.interparticle_stress()

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(stress_stat_dict,  dataPath + 'interparticle_stress_' + outfile + '.txt')

                press_dict = stress_and_pressure_functs.virial_pressure(stress_stat_dict)

                data_output_functs.write_to_txt(press_dict,  dataPath + 'virial_pressure_' + outfile + '.txt')

                shear_dict = stress_and_pressure_functs.shear_stress(stress_stat_dict)


                data_output_functs.write_to_txt(shear_dict, dataPath + 'shear_stress_' + outfile + '.txt')

                if plot == 'y':
                    vp_bin_arr = stress_and_pressure_functs.virial_pressure_binned(stress_plot_dict)
                    vp_part_arr = stress_and_pressure_functs.virial_pressure_part(stress_plot_dict)

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_interpart_press_binned(vp_bin_arr, all_surface_curves, int_comp_dict)
                    plotting_functs.interpart_press_map(pos, vp_part_arr, all_surface_curves, int_comp_dict)

            elif measurement_method == 'com_interface_pressure':
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                particle_prop_functs = particles.particle_props(l_box, partNum, NBins, peA, peB, typ, pos, ang)

                radial_fa_dict = particle_prop_functs.radial_normal_fa()

                com_radial_dict = stress_and_pressure_functs.radial_com_active_force_pressure(radial_fa_dict)

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(com_radial_dict, dataPath + 'com_radial_' + outfile + '.txt')

                act_press_dict = stress_and_pressure_functs.total_active_pressure(com_radial_dict)

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(act_press_dict, dataPath + 'com_interface_pressure_' + outfile + '.txt')
            elif measurement_method == 'surface_interface_pressure':
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                particle_prop_functs = particles.particle_props(l_box, partNum, NBins, peA, peB, typ, pos, ang)

                radial_fa_dict = particle_prop_functs.radial_surface_normal_fa(method2_align_dict)

                surface_radial_dict = stress_and_pressure_functs.radial_surface_active_force_pressure(radial_fa_dict, all_surface_measurements, all_surface_curves)

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(surface_radial_dict, dataPath + 'surface_radial_' + outfile + '.txt')

                act_press_dict = stress_and_pressure_functs.total_active_pressure(com_radial_dict)

                data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                data_output_functs.write_to_txt(act_press_dict, dataPath + 'surface_interface_pressure_' + outfile + '.txt')
            elif measurement_method == 'hexatic_order':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                hexatic_order_dict= lattice_structure_functs.hexatic_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_hexatic_order(pos, hexatic_order_dict['order'], all_surface_curves, int_comp_dict)

                    plotting_functs.plot_domain_angle(pos, hexatic_order_dict['theta'], all_surface_curves, int_comp_dict)

            elif measurement_method == 'translational_order':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                trans_order_param= lattice_structure_functs.translational_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_trans_order(pos, trans_order_param, all_surface_curves, int_comp_dict)

            elif measurement_method == 'steinhardt_order':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                stein_order_param= lattice_structure_functs.steinhardt_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_stein_order(pos, stein_order_param, all_surface_curves, int_comp_dict)

            elif measurement_method == 'nematic_order':

                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                nematic_order_param= lattice_structure_functs.nematic_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_stein_order(pos, nematic_order_param, all_surface_curves, int_comp_dict)


            stop

        prev_pos = pos.copy()
        prev_ang = ang.copy()