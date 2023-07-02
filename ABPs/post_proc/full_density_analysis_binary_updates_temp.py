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
import scipy

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
import kinetics
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

# Define Fourier series for fit


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
n_len = 21
n_arr = np.linspace(0, n_len-1, n_len)      #Fourier modes
popt_sum = np.zeros(n_len)                  #Fourier Coefficients

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
lat_sum = 0
action_arr = np.array([])

#Set output file names
bin_width = float(sys.argv[10])
time_step = float(sys.argv[11])
measurement_method = str(sys.argv[12])
plot = str(sys.argv[13])

outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(intPhi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
out = outfile + "_frame_"

dataPath = outPath + '_txt_files/'
picPath = outPath + '_pic_files/'

partPhase_time = np.array([])
partPhase_time_arr = np.array([])
clust_size_arr = np.array([])
avg_num_dens_dict = {}

vertical_shift = 0
avg_radial_df_dict = {}
sum_num = 0
dify_long = 0

com_option = False
mono_option = False
zoom_option = False
orientation_option = False
interface_option = False
banner_option = False

measurement_options = measurement_method.split('_')
for i in range(0, len(measurement_options)):
    if measurement_options[i] == 'com':
        com_option = True
    elif measurement_options[i] == 'mono':
        mono_option = True
    elif measurement_options[i] == 'zoom':
        zoom_option = True
    elif measurement_options[i] == 'orient':
        orientation_option = True
    elif measurement_options[i] == 'interface':
        interface_option = True
    elif measurement_options[i] == 'banner':
        banner_option = True
    elif measurement_options[i] == 'presentation':
        presentation_option = True
import time
with hoomd.open(name=inFile, mode='rb') as t:
    
    dumps = int(t.__len__())

    start = 500#0#int(0/time_step)#205                                             # first frame to process
    
                                # get number of timesteps dumped
    
    end = 502#int(dumps/time_step)-1                                             # final frame to process
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


    dt=0

    steady_state_once = 'False'

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

        min_size=int(partNum/10)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
        clust_large = np.amax(clust_size)
        


        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)


        

        
        

        if lx_box != ly_box:

            
            clust_large = 0

            start_time = time.time()
            partTyp=np.zeros(partNum)
            partPhase=np.zeros(partNum)
            edgePhase=np.zeros(partNum)
            bulkPhase=np.zeros(partNum)
            com_dict = plotting_utility_functs.com_view(pos, clp_all)

            
            
            if com_option == True:
                pos = com_dict['pos']

            #com_dict['com']['x']=0
            #com_dict['com']['y']=0
            

            #Bin system to calculate orientation and alignment that will be used in vector plots
            NBins_x = utility_functs.getNBins(lx_box, bin_width)
            NBins_y = utility_functs.getNBins(ly_box, bin_width)

            sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
            sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

            binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)

            pos_dict = binning_functs.create_bins()

            part_dict = binning_functs.bin_parts(pos, ids, clust_size)

            orient_dict = binning_functs.bin_orient(part_dict, pos, ang, com_dict['com'])
            area_frac_dict = binning_functs.bin_area_frac(part_dict)
            activ_dict = binning_functs.bin_activity(part_dict)
            
        else:
            start_time = time.time()
            partTyp=np.zeros(partNum)
            partPhase=np.zeros(partNum)
            edgePhase=np.zeros(partNum)
            bulkPhase=np.zeros(partNum)

            com_dict = plotting_utility_functs.com_view(pos, clp_all)
            
            

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
            if com_option == True:
                pos = com_dict['pos']

            #Bin system to calculate orientation and alignment that will be used in vector plots
            NBins_x = utility_functs.getNBins(lx_box, bin_width)
            NBins_y = utility_functs.getNBins(ly_box, bin_width)

            sizeBin_x = utility_functs.roundUp(((lx_box) / NBins_x), 6)
            sizeBin_y = utility_functs.roundUp(((ly_box) / NBins_y), 6)

            binning_functs = binning.binning(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps)

            pos_dict = binning_functs.create_bins()

            part_dict = binning_functs.bin_parts(pos, ids, clust_size)

            orient_dict = binning_functs.bin_orient(part_dict, pos, ang, com_dict['com'])
            area_frac_dict = binning_functs.bin_area_frac(part_dict)
            activ_dict = binning_functs.bin_activity(part_dict)
        
        outfile = 'pa'+str(int(peA))+'_pb'+str(int(peB))+'_xa'+str(int(parFrac))+'_eps'+str(eps)+'_phi'+str(phi)+'_pNum' + str(int(partNum)) + '_bin' + str(int(bin_width)) + '_time' + str(int(time_step))
        out = outfile + "_frame_"
        pad = str(j).zfill(5)
        outFile = out + pad

        if clust_large >= min_size:

            clust_size_arr = np.append(clust_size_arr, clust_large)
            fa_all_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_all_x_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_all_y_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_fast_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_slow_tot = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

            fa_all_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_fast_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]
            fa_slow_num = [[0 for b in range(NBins_y)] for a in range(NBins_x)]

            align_dict = binning_functs.bin_align(orient_dict)

            #Time frame for plots
            pad = str(j).zfill(5)

            press_dict = binning_functs.bin_active_press(align_dict, area_frac_dict)
            #plt.contourf(pos_dict['mid point']['x'], pos_dict['mid point']['y'], press_dict['bin']['all'])
            #plt.show()
            #stop
            normal_fa_dict = binning_functs.bin_normal_active_fa(align_dict, area_frac_dict, activ_dict)

            #radial_density_function_analysis_binary_updates
            align_grad_dict = binning_functs.curl_and_div(align_dict)

            
            plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile)

            phase_ident_functs = phase_identification.phase_identification(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ)

            phase_dict = phase_ident_functs.phase_ident()
            #phase_dict = phase_ident_functs.phase_ident_planar()
            bulk_id = np.where(phase_dict['part']==0)[0]
            int_id = np.where(phase_dict['part']==1)[0]
            gas_id = np.where(phase_dict['part']==2)[0]
            

            

            phase_dict = phase_ident_functs.phase_blur(phase_dict)

            phase_dict = phase_ident_functs.update_phasePart(phase_dict)

            bulk_id = np.where(phase_dict['part']==0)[0]
            int_id = np.where(phase_dict['part']==1)[0]
            gas_id = np.where(phase_dict['part']==2)[0]
            
            count_dict = phase_ident_functs.phase_count(phase_dict)

            bulk_com_dict = phase_ident_functs.com_bulk(phase_dict, count_dict)

            bulk_dict = phase_ident_functs.separate_bulks(phase_dict, count_dict, bulk_com_dict)

            phase_dict, bulk_dict, int_dict = phase_ident_functs.separate_ints(phase_dict, count_dict, bulk_dict)

            phase_dict = phase_ident_functs.update_phasePart(phase_dict)

            phase_dict, bulk_dict, int_dict = phase_ident_functs.reduce_gas_noise(phase_dict, bulk_dict, int_dict)

            phase_dict, bulk_dict, int_dict, int_comp_dict = phase_ident_functs.int_comp(part_dict, phase_dict, bulk_dict, int_dict)

            bulk_comp_dict = phase_ident_functs.bulk_comp(part_dict, phase_dict, bulk_dict)

            bulk_comp_dict = phase_ident_functs.phase_sort(bulk_comp_dict)

            int_comp_dict = phase_ident_functs.phase_sort(int_comp_dict)

            interface_functs = interface.interface(area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, ang)
            
            surface_dict = interface_functs.det_surface_points(phase_dict, int_dict, int_comp_dict)

            #planar_surface_dict = interface_functs.det_planar_surface_points(phase_dict, int_dict, int_comp_dict)
            #plt.scatter(surface_dict['surface 1']['pos']['x'], surface_dict['surface 1']['pos']['y'], s=10.0, c='red')
            #plt.scatter(surface_dict['surface 2']['pos']['x'], surface_dict['surface 2']['pos']['y'], s=10.0, c='blue')
            #plt.scatter(pos[:,0]+hx_box, pos[:,1]+hy_box, s=3.0, c='black')
            #plt.xlim(0, lx_box)
            #plt.ylim(0, ly_box)
            #plt.show()
            
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

            active_fa_dict = binning_functs.bin_active_fa(orient_dict, part_dict, phase_dict['bin'])

            #interface = interface.interface()

            #Slow/fast composition of bulk phase
            part_count_dict, part_id_dict = phase_ident_functs.phase_part_count(phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ)

            #Colors for plotting each phase
            yellow = ("#fdfd96")        #Largest gas-dense interface
            green = ("#77dd77")         #Bulk phase
            red = ("#ff6961")           #Gas phase
            purple = ("#cab2d6")        #Bubble or small gas-dense interfaces

            #label previous positions for velocity calculation

            
            data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)
            
            #Instantiate dictionaries to save data to
            all_surface_curves = {}

            all_surface_measurements = {}
            
            # Separate individual interfaces/surfaces
            sep_surface_dict = interface_functs.separate_surfaces(surface_dict, int_dict, int_comp_dict)

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
                if measurement_method == 'interface_props':
                    data_output_functs.write_to_txt(averaged_data_arr, dataPath + 'BubComp_' + outfile + '.txt')
            
            # If cluster has been formed, 
            if steady_state_once == 'False':
                clust_id_time = np.where(ids==lcID)[0]
                in_clust_arr = np.zeros(partNum)
                in_clust_arr[clust_id_time]=1
                #in_clust_arr = np.zeros(partNum)
                pos_x_arr_time = pos[:,0]
                pos_y_arr_time = pos[:,1]
                try:
                    com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box])
                    com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box])
                except:
                    com_x_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box])
                    com_y_arr_time = np.array([all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box])
                com_x_parts_arr_time = np.array([com_dict['com']['x']])
                com_y_parts_arr_time = np.array([com_dict['com']['y']])
                partPhase_time = phase_dict['part']
                partPhase_time_arr = np.append(partPhase_time_arr, tst)
                steady_state_once = 'True'
            else:
                clust_id_time = np.where(ids==lcID)[0]
                in_clust_temp = np.zeros(partNum)
                in_clust_temp[clust_id_time]=1
                in_clust_arr = np.vstack((in_clust_arr, in_clust_temp))
                pos_x_arr_time = np.vstack((pos_x_arr_time, pos[:,0]))
                pos_y_arr_time = np.vstack((pos_y_arr_time, pos[:,1]))
                partPhase_time_arr = np.append(partPhase_time_arr, tst)
                partPhase_time = np.vstack((partPhase_time, phase_dict['part']))
                try:
                    com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['x']-hx_box)
                    com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['exterior']['com']['y']-hy_box)
                except:
                    com_x_arr_time = np.append(com_x_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['x']-hx_box)
                    com_y_arr_time = np.append(com_y_arr_time, all_surface_measurements['surface id ' + str(int(int_comp_dict['ids'][0]))]['interior']['com']['y']-hy_box)

                com_x_parts_arr_time = np.append(com_x_parts_arr_time, com_dict['com']['x'])
                com_y_parts_arr_time = np.append(com_y_parts_arr_time, com_dict['com']['y'])

            method1_align_dict, method2_align_dict = interface_functs.surface_alignment(all_surface_measurements, all_surface_curves, sep_surface_dict, int_dict, int_comp_dict)
            
            
            method1_align_dict, method2_align_dict = interface_functs.bulk_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict)

            method1_align_dict, method2_align_dict = interface_functs.gas_alignment(method1_align_dict, method2_align_dict, all_surface_measurements, all_surface_curves, sep_surface_dict, int_comp_dict)


            if measurement_options[0] == 'vorticity':

                #DONE!
                if j>(start*time_step):

                    if plot == 'y':

                        plotting_functs.plot_vorticity(vel_dict['bin'], vel_grad['curl'], phase_dict, all_surface_curves, int_comp_dict, active_fa_dict, species='all', interface_id = interface_option)
            elif measurement_method == 'single_velocity':
                if j>(start * time_step):
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, pos, ang)
                    
                    try:
                        part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)
                    except:
                        displace_dict = {'A': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])}, 'B': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])} }
                        part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)

                    vel_plot_dict, corr_dict, vel_stat_dict = particle_prop_functs.single_velocity(vel_dict['part'], prev_pos, prev_ang, ori)
                    data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'collision_' + outfile + '.txt')
            elif measurement_method == 'adsorption':
                
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
                
                
                kinetics_dict = particle_prop_functs.adsorption_nlist()
                collision_dict = particle_prop_functs.collision_rate()
                
                
                data_output_functs.write_to_txt(kinetics_dict, dataPath + 'kinetics_' + outfile + '.txt')
                data_output_functs.write_to_txt(collision_dict, dataPath + 'collision_' + outfile + '.txt')


            elif measurement_method == 'velocity':
                if j>(start*time_step):
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, pos, ang)

                    part_ang_vel_dict = particle_prop_functs.angular_velocity(ang_vel_dict['part'], phase_dict['part'])
                    part_vel_dict = particle_prop_functs.velocity(vel_dict['part']['mag'], phase_dict['part'])

                    data_output_functs.write_to_txt(part_ang_vel_dict, dataPath + 'angular_velocity_' + outfile + '.txt')
                    data_output_functs.write_to_txt(part_vel_dict, dataPath + 'velocity_' + outfile + '.txt')

                    if plot == 'y':
                        plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)
                        plotting_functs.ang_vel_histogram(ang_vel_dict['part'], phase_dict['part'])
                        plotting_functs.ang_vel_bulk_sf_histogram(ang_vel_dict['part'], phase_dict['part'])
                        plotting_functs.ang_vel_int_sf_histogram(ang_vel_dict['part'], phase_dict['part'])
                        plotting_functs.ang_vel_int_sf_histogram(ang_vel_dict['part'], phase_dict['part'])
                    stop
            #elif measurement_method == 'voronoi':
            #    if plot == 'y':
            #        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst)
            #        plotting_functs.plot_voronoi(pos)

            elif (measurement_options[0] == 'activity'):
                
                #DONE!
                if plot == 'y':
                    
                    
                    plotting_functs.plot_part_activity(pos, all_surface_curves, int_comp_dict, active_fa_dict, mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option, banner_id = banner_option, presentation_id = presentation_option)

            elif measurement_method == 'activity_com':

                if plot == 'y':
                    
                    plotting_functs.plot_part_activity_com_plotted(pos, part_id_dict, all_surface_curves, int_comp_dict, com_opt)

            elif measurement_options[0] == 'phases':
                #DONE!
                data_output_functs.write_to_txt(part_count_dict, dataPath + 'PhaseComp_' + outfile + '.txt')
                data_output_functs.write_to_txt(bin_count_dict['bin'], dataPath + 'PhaseComp_bins_' + outfile + '.txt')

                # Plot particles color-coded by phase
                if plot == 'y':
                    plotting_functs.plot_phases(pos, part_id_dict, all_surface_curves, int_comp_dict, active_fa_dict, interface_id = interface_option, orientation_id = orientation_option)
            elif measurement_method== 'bubble_interface_pressure':

                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                radial_fa_dict = particle_prop_functs.radial_surface_normal_fa_bubble2(method2_align_dict, all_surface_curves, int_comp_dict, all_surface_measurements, int_dict)

                #stress_stat_dict, press_stat_dict, press_plot_dict, stress_plot_dict = lattice_structure_functs.interparticle_pressure_nlist()

                #radial_int_press_dict = particle_prop_functs.radial_int_press_bubble2(stress_plot_dict, all_surface_curves, int_comp_dict, all_surface_measurements)

                #com_radial_dict_bubble, com_radial_dict_fa_bubble = particle_prop_functs.radial_measurements2(radial_int_press_dict, radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)
                com_radial_dict_fa_bubble = particle_prop_functs.radial_measurements3(radial_fa_dict, surface_dict, all_surface_curves, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict)

                #for m in range(0, len(com_radial_dict_fa_bubble)):

                key = 'surface id ' + str(averaged_data_arr['int_id'])
                data_output_functs.write_to_txt(com_radial_dict_fa_bubble[key], dataPath + 'bubble_com_active_pressure_radial_' + outfile + '.txt')
                
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                act_press_dict_bubble = stress_and_pressure_functs.total_active_pressure_bubble(com_radial_dict_fa_bubble, all_surface_measurements, int_comp_dict, all_surface_measurements)
                
                for m in range(0, len(sep_surface_dict)):

                    key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                    data_output_functs.write_to_txt(act_press_dict_bubble[key], dataPath + 'bubble_com_active_pressure_' + outfile + '.txt')


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
                """
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['num_dens']['A'], color='blue')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['num_dens']['B'], color='red')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['num_dens']['all'], color='black')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['num_dens']['A'], color='blue', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['num_dens']['B'], color='red', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['num_dens']['all'], color='black', linestyle='dashed')
                plt.show()

                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['align']['A'], color='blue')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['align']['B'], color='red')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['align']['all'], color='black')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['align']['A'], color='blue', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['align']['B'], color='red', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['align']['all'], color='black', linestyle='dashed')
                plt.show()

                plt.plot(com_radial_dict['r'], com_radial_dict['A']['press'], color='blue')
                plt.plot(com_radial_dict['r'], com_radial_dict['B']['press'], color='red')
                plt.plot(com_radial_dict['r'], com_radial_dict['all']['press'], color='black')
                plt.plot(com_radial_dict_bubble['r'], com_radial_dict_bubble['A']['press'], color='blue', linestyle='dashed')
                plt.plot(com_radial_dict_bubble['r'], com_radial_dict_bubble['B']['press'], color='red', linestyle='dashed')
                plt.plot(com_radial_dict_bubble['r'], com_radial_dict_bubble['all']['press'], color='black', linestyle='dashed')
                plt.show()

                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['fa_press']['A'], color='blue')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['fa_press']['B'], color='red')
                plt.plot(com_radial_dict_fa['r'], com_radial_dict_fa['fa_press']['all'], color='black')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['fa_press']['A'], color='blue', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['fa_press']['B'], color='red', linestyle='dashed')
                plt.plot(com_radial_dict_fa_bubble['r'], com_radial_dict_fa_bubble['fa_press']['all'], color='black', linestyle='dashed')
                plt.show()

                stop
                """


                


            elif measurement_options[0] == 'density':

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

            elif measurement_options[0] == 'com-align':

                if plot == 'y':

                    # Plot contour map of degree of alignment of particle's active forces with direction toward cluster CoM
                    plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='all')
                    plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='A')
                    plotting_functs.plot_alignment(method1_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='B')
                    plotting_functs.plot_normal_fa_map(normal_fa_dict, all_surface_curves, int_comp_dict)
                    
            elif measurement_options[0] == 'surface-alignment':

                if plot == 'y':

                    # Plot contour map of degree of alignment of particle's active forces with nearest normal to cluster surface
                    plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='all') 
                    plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='A')
                    plotting_functs.plot_alignment(method2_align_dict, all_surface_curves, int_comp_dict, pos, interface_id = interface_option, type='B')       
             
                stop
            elif measurement_method == 'fluctuations':

                if plot == 'y':

                    # Plot particles color-coded by activity with sub-plot of change in cluster size over time
                    plotting_functs.plot_clust_fluctuations(pos, outfile, all_surface_curves, int_comp_dict)

            elif measurement_method == 'compressibility':
                #DONE!

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate radial distribution functions
                radial_df_dict = lattice_structure_functs.radial_df()

                # Calculate compressibility
                compress_dict = lattice_structure_functs.compressibility(radial_df_dict)

                # Save compressibility data
                data_output_functs.write_to_txt(compress_dict, dataPath + 'compressibility_' + outfile + '.txt')
            
            elif measurement_method == 'structure_factor2':
                
                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate compressibility, structure factor, and 1st-wave vector structure factor using alternative hard-coded method
                compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor2()

                #data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor_' + outfile + '.txt')

                #data_output_functs.write_to_txt(compress_dict, dataPath + 'compressibility2_' + outfile + '.txt')

            
            elif measurement_method == 'structure_factor_freud':
                
                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate compressibility, structure factor, and 1st-wave vector structure factor using Freud
                compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor_freud()

                #data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor_' + outfile + '.txt')

                #data_output_functs.write_to_txt(compress_dict, dataPath + 'compressibility2_' + outfile + '.txt')

            elif measurement_method == 'structure_factor':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate radial distribution functions
                radial_df_dict = lattice_structure_functs.radial_df()

                # Calculate compressibility, structure factor, and 1st-wave vector structure factor using hard-coded function
                compress_dict, structure_factor_dict, k0_dict = lattice_structure_functs.structure_factor(radial_df_dict, part_count_dict)

                # Save structure factor data
                data_output_functs.write_to_txt(k0_dict, dataPath + 'structure_factor_' + outfile + '.txt')

                # Save compressibility data
                data_output_functs.write_to_txt(compress_dict, dataPath + 'compressibility2_' + outfile + '.txt')

            elif measurement_method == 'int_press':

                # Initialize stress and pressure functions
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate interparticle stresses and pressures
                stress_stat_dict, press_stat_dict, press_plot_dict, stress_plot_dict = lattice_structure_functs.interparticle_pressure_nlist()

                # Save stress and pressure data
                data_output_functs.write_to_txt(stress_stat_dict, dataPath + 'interparticle_stress_' + outfile + '.txt')
                data_output_functs.write_to_txt(press_stat_dict, dataPath + 'interparticle_press_' + outfile + '.txt')
                 
                # Measure radial interparticle pressure
                radial_int_press_dict = particle_prop_functs.radial_int_press(stress_plot_dict)

                # Measure radial interparticle pressure
                com_radial_int_press_dict = stress_and_pressure_functs.radial_com_interparticle_pressure(radial_int_press_dict)

                # Save radial stress and pressure data
                data_output_functs.write_to_txt(com_radial_int_press_dict, dataPath + 'com_interparticle_pressure_radial_' + outfile + '.txt')
                
                if plot == 'y':

                    #plotting_functs.plot_interpart_press_binned(vp_bin_arr, all_surface_curves, int_comp_dict)
                    plotting_functs.interpart_press_map_final(stress_plot_dict, all_surface_curves, int_comp_dict)                               
            
            elif measurement_method == 'cluster_velocity': 
                if j>(start * time_step):
                    
                    #Initialize particle-based location measurements
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
                    
                    # Measure cluster displacement
                    cluster_velocity_dict = particle_prop_functs.cluster_velocity(prev_pos, dt_step)

                    # Save cluster displacement data
                    data_output_functs.write_to_txt(cluster_velocity_dict, dataPath + 'cluster_velocity_' + outfile + '.txt')

            elif measurement_method == 'centrosymmetry':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate centrosymmetry properties
                csp_stat_dict, csp_plot_dict = lattice_structure_functs.centrosymmetry()

                # Save centrosymmetry data
                data_output_functs.write_to_txt(csp_stat_dict, dataPath + 'centrosymmetry_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs.plot_csp(csp_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all')
                

            elif measurement_method == 'lattice_spacing':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Measure cluster lattice spacing
                lat_stat_dict, lat_plot_dict = lattice_structure_functs.lattice_spacing()

                # Save cluster lattice spacing data
                data_output_functs.write_to_txt(lat_stat_dict, dataPath + 'lattice_spacing_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.lat_histogram(lat_plot_dict)

                    if j>(start*time_step):
                        plotting_functs.lat_map(lat_plot_dict, all_surface_curves, int_comp_dict, velocity_dict = vel_dict)
                    else:
                        plotting_functs.lat_map(lat_plot_dict, all_surface_curves, int_comp_dict)
            elif measurement_method == 'penetration':
                
                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
                if j>(start*time_step):

                    # Calculate degree of penetration
                    penetration_dict, start_dict = lattice_structure_functs.penetration_depth(start_dict, prev_pos)

                    # Save degree of penetration data
                    data_output_functs.write_to_txt(penetration_dict, dataPath + 'penetration_depth_' + outfile + '.txt')

            elif measurement_method == 'radial_df':

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
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

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

            elif measurement_method == 'angular_df':

                #DONE

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Measure angular distribution function
                angular_df_dict = lattice_structure_functs.angular_df()

                # Save angular distribution data
                data_output_functs.write_to_txt(angular_df_dict, dataPath + 'angular_df_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.plot_general_adf(angular_df_dict)
                    plotting_functs.plot_all_adfs(angular_df_dict)
                stop
            elif measurement_method == 'domain_size':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Measure domain size
                domain_stat_dict = lattice_structure_functs.domain_size()

                # Save domain size data
                data_output_functs.write_to_txt(domain_stat_dict, dataPath + 'domain_' + outfile + '.txt')

            elif measurement_method == 'clustering':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
                # Measure clustering coefficient
                neigh_plot_dict = lattice_structure_functs.clustering_coefficient()
                
                if plot == 'y':
                    plotting_functs.plot_clustering(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all')
                    plotting_functs.plot_clustering(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='A')
                    plotting_functs.plot_clustering(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='B')
                    
                stop
                #data_output_functs.write_to_txt(neigh_stat_dict, dataPath + 'nearest_neighbors_' + outfile + '.txt')
                #data_output_functs.write_to_txt(ori_stat_dict, dataPath + 'nearest_ori_' + outfile + '.txt')
            elif measurement_method == 'local_gas_density':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
                # Calculate local density of gas phase
                if lx_box == ly_box:
                    local_dens_stat_dict, local_dens_plot_dict = lattice_structure_functs.local_gas_density()

                # Save local density of gas phase data
                data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_gas_density_' + outfile + '.txt')
                
                
                if plot == 'y':
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-B')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-B')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-B')
            
            elif measurement_method == 'local_density':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
                # Calculate local density of dense phase
                if lx_box == ly_box:
                    local_dens_stat_dict, local_dens_plot_dict = lattice_structure_functs.local_density()

                # Save local density of dense phase data
                data_output_functs.write_to_txt(local_dens_stat_dict, dataPath + 'local_density_' + outfile + '.txt')
                
                if plot == 'y':
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-all')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='all-B')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-A')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='A-B')
                    plotting_functs.plot_local_density(local_dens_plot_dict, all_surface_curves, int_comp_dict, pair='B-B')
            
            elif measurement_options[0] == 'neighbors':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                
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

                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all-all')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all-A')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all-B')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='A-all')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='B-all')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='A-A')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='A-B')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='B-A')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='B-B')
                    

                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-all', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='all-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='A_A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='A-B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)
                    plotting_functs.plot_neighbors_ori(neigh_plot_dict, ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, pair='B-A', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)

                    #.plot_neighbors_ori(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all-A')
                    #plotting_functs.plot_neighbors_ori(neigh_plot_dict, all_surface_curves, int_comp_dict, ang, pos, pair='all-B')
                    #plotting_functs.plot_particle_orientations(neigh_plot_dict, all_surface_curves, int_comp_dict, pair='all-all')
                    #stop
                    #plotting_functs.local_orientational_order_map(neigh_plot_dict, all_surface_curves, int_comp_dict, type='all-all')
                    #stop
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, pair='all-all')
                    #stop
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, pair='all-A')
                    #plotting_functs.plot_neighbors(neigh_plot_dict, all_surface_curves, int_comp_dict, pair='all-B')

                    #plotting_functs.plot_A_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_B_neighbors_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)

                    #plotting_functs.plot_all_ori_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_all_ori_of_A_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_all_ori_of_B_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_A_ori_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
                    #plotting_functs.plot_B_ori_of_all_parts(neigh_plot_dict, all_surface_curves, int_comp_dict)
            elif measurement_options[0] == 'orientation':
                #DONE
                if plot == 'y':

                    plotting_functs.plot_ang(ang, pos, all_surface_curves, int_comp_dict, active_fa_dict, type='B', mono_id = mono_option, zoom_id = zoom_option, interface_id = interface_option, orientation_id = orientation_option)

            elif measurement_method == 'interparticle_pressure':

                # Initialize stress and pressure functions
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

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
                    vp_bin_arr = stress_and_pressure_functs.virial_pressure_binned(stress_plot_dict)
                    vp_part_arr = stress_and_pressure_functs.virial_pressure_part(stress_plot_dict)

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_interpart_press_binned(vp_bin_arr, all_surface_curves, int_comp_dict)
                    plotting_functs.interpart_press_map(pos, vp_part_arr, all_surface_curves, int_comp_dict)
            elif measurement_method == 'interparticle_pressure_nlist':
                
                # Initialize stress and pressure functions
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)
                stress_plot_dict, stress_stat_dict = stress_and_pressure_functs.interparticle_stress_nlist(phase_dict['part'])

            elif measurement_method == 'com_interface_pressure':
                
                # Initialize stress and pressure functions
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                #Initialize particle-based location measurements
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                # Calculate radially active forces toward cluster CoM
                radial_fa_dict = particle_prop_functs.radial_normal_fa()

                # Calculate radial active force pressure toward cluster CoM
                com_radial_dict = stress_and_pressure_functs.radial_com_active_force_pressure(radial_fa_dict)

                # Save radial active force pressure toward cluster CoM
                data_output_functs.write_to_txt(com_radial_dict, dataPath + 'com_interface_pressure_radial_' + outfile + '.txt')

                # Calculate total active force pressure toward cluster CoM
                act_press_dict = stress_and_pressure_functs.total_active_pressure(com_radial_dict)

                # Save total active force pressure toward cluster CoM
                data_output_functs.write_to_txt(act_press_dict, dataPath + 'com_interface_pressure_' + outfile + '.txt')
            
            elif measurement_method == 'surface_interface_pressure':
                
                # Initialize stress and pressure functions
                stress_and_pressure_functs = stress_and_pressure.stress_and_pressure(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                #Initialize particle-based location measurements
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
                
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
                
            elif measurement_method == 'hexatic_order':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate hexatic order of particles with nearest neighbors
                hexatic_order_dict= lattice_structure_functs.hexatic_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.plot_hexatic_order(pos, hexatic_order_dict['order'], all_surface_curves, int_comp_dict)

                    plotting_functs.plot_domain_angle(pos, hexatic_order_dict['theta'], all_surface_curves, int_comp_dict)
            
            elif measurement_method == 'voronoi':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate voronoi tesselation of particles
                voronoi_dict= lattice_structure_functs.voronoi()
                
                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.plot_hexatic_order(pos, hexatic_order_dict['order'], all_surface_curves, int_comp_dict)

                    plotting_functs.plot_domain_angle(pos, hexatic_order_dict['theta'], all_surface_curves, int_comp_dict)

            elif measurement_method == 'translational_order':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate translational order parameter
                trans_order_param= lattice_structure_functs.translational_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.plot_trans_order(pos, trans_order_param, all_surface_curves, int_comp_dict)

            elif measurement_method == 'steinhardt_order':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate steinhardt order parameter
                stein_order_param= lattice_structure_functs.steinhardt_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':

                    plotting_functs.plot_stein_order(pos, stein_order_param, all_surface_curves, int_comp_dict)
                stop
            elif measurement_method == 'nematic_order':

                # Initialize lattice structure functions
                lattice_structure_functs = measurement.measurement(l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

                # Calculate nematic order parameter
                nematic_order_param= lattice_structure_functs.nematic_order()

                #data_output_functs = data_output.data_output(l_box, sizeBin, tst, clust_large, dt_step)
                #data_output_functs.write_to_txt(hexatic_order_dict, dataPath + 'hexatic_order_' + outfile + '.txt')

                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst)

                    plotting_functs.plot_stein_order(pos, nematic_order_param, all_surface_curves, int_comp_dict)
            elif measurement_method == 'adsorption_final':

                if len(partPhase_time_arr)>1:
                
                    if steady_state_once == 'True':
                        kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                        clust_motion_dict, adsorption_dict = kinetic_functs.particle_flux(partPhase_time, in_clust_arr, partPhase_time_arr, clust_size_arr, pos_x_arr_time, pos_y_arr_time, com_x_arr_time, com_y_arr_time, com_x_parts_arr_time, com_y_parts_arr_time)

                        data_output_functs.write_to_txt(adsorption_dict, dataPath + 'not_adsorption_final_' + outfile + '.txt')
                        data_output_functs.write_to_txt(clust_motion_dict, dataPath + 'not_clust_motion_final_' + outfile + '.txt')
        else:

            if j>(start*time_step):
                vel_dict = binning_functs.bin_vel(pos, prev_pos, part_dict, dt_step)
                ang_vel_dict = binning_functs.bin_ang_vel(ang, prev_ang, part_dict, dt_step)
                vel_grad = binning_functs.curl_and_div(vel_dict)

            if measurement_method == 'activity':

                #DONE
                if plot == 'y':
                    plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile)
                    plotting_functs.plot_part_activity(pos, ang)
                    #if j>(start*time_step):
                    #    plotting_functs.plot_part_activity2(pos, prev_pos, ang)
                    #else:
                    #    plotting_functs.plot_part_activity(pos, ang)
            elif measurement_method == 'neighbors':
                #DONE
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
                
                neigh_stat_dict, ori_stat_dict, neigh_plot_dict = particle_prop_functs.nearest_neighbors_penetrate()                
                
                data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                data_output_functs.write_to_txt(neigh_stat_dict, dataPath + 'nearest_neighbors_' + outfile + '.txt')
                if plot == 'y':

                    plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile)

                    #if j>(start*time_step):
                    #    plotting_functs.plot_neighbors2(neigh_plot_dict, ang, pos, prev_pos, pair='all-all')
                    #else:
                    plotting_functs.plot_neighbors(neigh_plot_dict, ang, pos, pair='all-all')

            elif measurement_method == 'penetration':
                #DONE
                
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
                if j>(start*time_step):
                    penetration_dict, start_dict, vertical_shift, dify_long = particle_prop_functs.penetration_depth(start_dict, prev_pos, vertical_shift, dify_long)

                    data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                    data_output_functs.write_to_txt(penetration_dict, dataPath + 'penetration_depth_' + outfile + '.txt')

                    action_arr = np.append(action_arr, penetration_dict['action'])
            elif measurement_method == 'part_velocity':
                if j>(start * time_step):
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                    #try:
                    #    part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)
                    #except:
                        #displace_dict = {'A': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])}, 'B': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])} }
                        #part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)

                    data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                    vel_plot_dict, vel_stat_dict = particle_prop_functs.part_velocity(prev_pos, prev_ang, ori)
                    data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'velocity_' + outfile + '.txt')
                    #data_output_functs.write_to_txt(neigh_dict, dataPath + 'collision_' + outfile + '.txt')

            elif measurement_method == 'single_velocity':
                if j>(start * time_step):
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                    #try:
                    #    part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)
                    #except:
                        #displace_dict = {'A': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])}, 'B': {'x': np.array([]), 'y': np.array([]), 'mag': np.array([])} }
                        #part_msd_dict = particle_prop_functs.single_msd(prev_pos, displace_dict)

                    data_output_functs = data_output.data_output(lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step)

                    vel_plot_dict, corr_dict, vel_stat_dict = particle_prop_functs.single_velocity(vel_dict['part'], prev_pos, prev_ang, ori)
                    data_output_functs.write_to_txt(vel_stat_dict, dataPath + 'collision_' + outfile + '.txt')
                    
        #if j == start:
        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

        utility_functs = utility.utility(lx_box, ly_box)

        prev_pos = pos.copy()
        prev_ang = ang.copy()

        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

        utility_functs = utility.utility(lx_box, ly_box)

    if measurement_method == 'adsorption_final':
        if len(partPhase_time_arr)>1:
            if steady_state_once == 'True':
                kinetic_functs = kinetics.kinetic_props(lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac)

                adsorption_dict, clust_motion_dict = kinetic_functs.particle_flux_final(partPhase_time, in_clust_arr, partPhase_time_arr, clust_size_arr, pos_x_arr_time, pos_y_arr_time, com_x_arr_time, com_y_arr_time, com_x_parts_arr_time, com_y_parts_arr_time)

                data_output_functs.write_all_time_to_txt(adsorption_dict, dataPath + 'adsorption_final_' + outfile + '.txt')
                data_output_functs.write_all_time_to_txt(clust_motion_dict, dataPath + 'clust_motion_final_' + outfile + '.txt')
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
                particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                stress_stat_dict, press_stat_dict, press_plot_dict = particle_prop_functs.interparticle_pressure_nlist()
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
                    particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)

                    penetration_dict, start_dict, vertical_shift, dify_long = particle_prop_functs.penetration_depth(start_dict, prev_pos, vertical_shift, dify_long)
                    stress_stat_dict, press_stat_dict, press_plot_dict = particle_prop_functs.interparticle_pressure_nlist()
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

                        plotting_functs = plotting.plotting(orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, picPath, outFile)


                        plotting_functs.interpart_press_map2(press_plot_dict, pos, prev_pos, ang)     

                prev_pos = pos.copy()
                prev_ang = ang.copy()
    elif measurement_method == 'radial_df':
        # Done but inaccurate in planar system
        
        for key, value in radial_df_dict.items():
            if key != 'r':
                avg_radial_df_dict[key] = (np.array(avg_radial_df_dict[key])/sum_num).tolist()
        lat_avg = lat_sum / sum_num

        print(sum_num)
        lattice_structure_functs = measurement.measurement(lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict)

        avg_compress_dict = lattice_structure_functs.compressibility(avg_radial_df_dict, avg_num_dens = avg_num_dens_dict)

        avg_wasserstein_dict = lattice_structure_functs.wasserstein_distance(avg_radial_df_dict, lat_avg)

        data_output_functs.write_to_txt(avg_compress_dict, dataPath + 'avg_compressibility_' + outfile + '.txt')

        data_output_functs.write_to_txt(avg_wasserstein_dict, dataPath + 'avg_wasserstein_' + outfile + '.txt')

        data_output_functs.write_to_txt(avg_radial_df_dict, dataPath + 'avg_radial_df_' + outfile + '.txt')

        if plot == 'y':

                    plotting_functs.plot_general_rdf(radial_df_dict)
                    plotting_functs.plot_all_rdfs(radial_df_dict)
    elif measurement_method == 'collision':
                
        particle_prop_functs = particles.particle_props(lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang)
        
        collision_dict = particle_prop_functs.collision_rate()
        
        data_output_functs.write_to_txt(collision_dict, dataPath + 'collision_' + outfile + '.txt')
