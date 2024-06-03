#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python class contains various functions for easily plotting physical
quantities from HOOMD-Blue simulations (.gsd). Basic simulation inputs and
quantities measured via other python classes in this module (i.e.
via measurement.py, binning.py, etc.) for a single time frame are input into
the plotting functions and that data, as specified by the plotting function's
name, is plotted for that frame.
"""

import sys
import os

from gsd import hoomd
import freud
from freud import box
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage

import numpy as np
import matplotlib

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
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.collections

import statistics
from statistics import mode

from scipy.optimize import curve_fit

#from symfit import parameters, variables, sin, cos, Fit

# Add github's library of post-processing functions to system path to import
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility

# Plot parameters
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.latex.preview'] = True


class plotting:
    """
    Purpose: 
    This class contains a series of basic functions for visualizing properties calculated in other classes through scatter plots, heat maps, and histograms 
    """
    def __init__(self, orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, outPath, outFile, phi):

        # Values from theory
        self.phiCP = np.pi / (2. * np.sqrt(3.))                                     # critical packing fraction of HCP lattice
        theory_functs = theory.theory()

        # Simulation Inputs

        self.lx_box = lx_box                                #x dimension box length
        self.hx_box = self.lx_box/2
        self.ly_box = ly_box                                #y dimension box length
        self.hy_box = self.ly_box/2
        utility_functs = utility.utility(self.lx_box, self.ly_box)

        self.partNum = partNum                                                      # System size
        self.peA = peA                                                              # Type A (slow) particle activity (Peclet number)
        self.peB = peB                                                              # Type B (fast) particle activity (Peclet number)
        self.peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))                          # Net (average) activity of system
        self.parFrac = parFrac                                                      # Slow particle fraction
        self.eps = eps                                                              # Potential magnitude (softness, units of kT)
        self.phi = phi

        # Binning properties
        try:
            self.NBins_x = int(NBins_x)                                             # Number of bins in x-dimension
            self.NBins_y = int(NBins_y)                                             # Number of bins in y-dimension
        except:
            print('NBins must be either a float or an integer')

        self.sizeBin_x = utility_functs.roundUp((self.lx_box / self.NBins_x), 6)    # y-length of bin (simulation units)
        self.sizeBin_y = utility_functs.roundUp((self.ly_box / self.NBins_y), 6)    # x-length of bin (simulation units)

        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        #Frame-specific information
        self.tst = tst                                                              # Current time step (Brownian units)

        self.orient_x = orient_dict['bin']['all']['x']                              # 1D array (partNum) of X-orientations of each particle
        self.orient_y = orient_dict['bin']['all']['y']                              # 1D array (partNum) of Y-orientations of each particle
        self.orient_mag = orient_dict['bin']['all']['mag']                          # 1D array (partNum) of orientation magnitude of each particle

        self.typ = typ                                                              # 1D Boolean array (partNum) of each particles type of either type A (0) or B (1)

        self.pos_x = pos_dict['mid point']['x']                                     # 2D array (NBins_x, NBins_y) of y mid-point of each bin
        self.pos_y = pos_dict['mid point']['y']                                     # 2D array (NBins_x, NBins_y)of x mid-point of each bin

        self.beta_A = 1.0
        self.beta_B = 2.3

        self.outPath = outPath
        self.outFile = outFile

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

    def plot_phases_heterogeneity(self, pos, plot_heterogeneity_dict, phase_ids_dict, phase_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False, presentation_id=False):
        #DONE!
        """
        This function plots each particle's position and color-codes each
        particle by the phase it is a part of, i.e. bulk=green,
        interface=purple, and gas=red.

        Inputs:
        pos: array (partNum, 3) of each particles x,y,z positions
        phase_ids_dict: dictionary (output from various phase_identification
        functions) containing information on the composition of each phase

        phase_ids_dict: dictionary (output from phase_part_count() in
        phase_identification.py) that contains the id of each particle specifying
        the phase of bulk (0), interface (1), or gas (2).

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with each particle's position plotted and color coded by the
        phase it's a part of and the interface surfaces plotted in black
        """

        yellow = ("#7570b3")
        yellow = ("#ffd700")
        red = ("#fd8d3c")
        red = ("#dd571c")
        red = ("#ff7700")
        red = ("#cc5500")
        red = ("#f66e8d")
        green = ("#77dd77")
        green = ("#4a3728")
        green = ("#3c4142")
        green = ("#878787")
        green = ("#71797E")

        green = ("#cc5500")
        #green = ("#b3de69")
        #red = ("#ff6961")
        #red = ("#000000")

        bulk_part_ids = phase_ids_dict['bulk']['all']
        gas_part_ids = phase_ids_dict['gas']['all']
        int_part_ids = phase_ids_dict['int']['all']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            if presentation_id == True:
                x_dim = scaling+0.18
            else:
                x_dim = int(scaling + 0.5)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        average_theta_path = '/Volumes/External/n100000test/pa0_pb500/averages/radial_avgs_theta_pa0_pb500_xa50_eps1.0_phi60.0_pNum50000_bin5_time1.csv'
        average_rad_path = '/Volumes/External/n100000test/pa0_pb500/averages/radial_avgs_rad_pa0_pb500_xa50_eps1.0_phi60.0_pNum50000_bin5_time1.csv'

        import csv 

        averages_file =  open(average_theta_path, newline='')
        avg_theta_r = np.array(list(csv.reader(averages_file)))

        avg_theta_r_flatten = (avg_theta_r.flatten()).astype(np.float) 
        avg_theta_r_flatten = np.append(avg_theta_r_flatten, avg_theta_r_flatten[0])
        averages_file =  open(average_rad_path, newline='')
        avg_rad_r = np.array(list(csv.reader(averages_file)))

        avg_rad_r_flatten = (avg_rad_r.flatten()).astype(np.float) 
        
        average_indiv_vals_path = '/Volumes/External/n100000test/pa0_pb500/averages/radial_avgs_indiv_vals_pa0_pb500_xa50_eps1.0_phi60.0_pNum50000_bin5_time1.csv'

        averages_file =  open(average_indiv_vals_path, newline='')
        reader = csv.reader(averages_file)
        avg_indiv_vals = dict(reader)
        avg_radius_r_flatten= float(avg_indiv_vals['radius'])

        
        for i in range(0, len(avg_theta_r_flatten)-1):   
            x_coords = self.hx_box + avg_rad_r_flatten * avg_radius_r_flatten * np.cos(avg_theta_r_flatten[i]*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten * avg_radius_r_flatten * np.sin(avg_theta_r_flatten[i]*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.65)

        for i in range(0, len(avg_rad_r_flatten)):   
            x_coords = self.hx_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.65)
        


        # Set plotted particle size
        sz = 0.755
        """
        # Plot position colored by phase
        if len(bulk_part_ids)>0:
            ells_bulk = [Ellipse(xy=np.array([pos[bulk_part_ids[i],0]+self.hx_box,pos[bulk_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(bulk_part_ids))]
            bulkGroup = mc.PatchCollection(ells_bulk, facecolor=green)
            ax.add_collection(bulkGroup)
        
        if len(gas_part_ids)>0:
            ells_gas = [Ellipse(xy=np.array([pos[gas_part_ids[i],0]+self.hx_box,pos[gas_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(gas_part_ids))]
            gasGroup = mc.PatchCollection(ells_gas, facecolor=red)
            ax.add_collection(gasGroup)
        """

        x_coords = self.hx_box + plot_heterogeneity_dict['rad']['all'] * avg_radius_r_flatten * np.cos(plot_heterogeneity_dict['theta']['all']*(np.pi/180))
        y_coords = self.hy_box + plot_heterogeneity_dict['rad']['all'] * avg_radius_r_flatten * np.sin(plot_heterogeneity_dict['theta']['all']*(np.pi/180))
        
        if len(x_coords)>0:
            ells_int = [Ellipse(xy=np.array([x_coords[i],y_coords[i]]),
                width=sz, height=sz)
        for i in range(0,len(x_coords))]

            intGroup = mc.PatchCollection(ells_int, facecolor=yellow)
            ax.add_collection(intGroup)
        """
        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        """
        x_coords = self.hx_box + 1.0 * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
        y_coords = self.hy_box + 1.0 * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))
    
        plt.plot(x_coords, y_coords, c='black', linewidth=3.0) 
        #active_r = ( active_fa_dict['bin']['x'] ** 2 + active_fa_dict['bin']['y'] ** 2 ) ** 0.5
        active_r = ( np.array(active_fa_dict['bin']['all']['x']) ** 2 + np.array(active_fa_dict['bin']['all']['y']) ** 2 ) ** 0.5
        nonzero_id = np.nonzero(active_r)
        
        nonzero_x_id = nonzero_id[0]
        nonzero_y_id = nonzero_id[1]
        # Plot averaged, binned orientation of particles
        """
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    for i in range(0, len(nonzero_x_id)):
                        #plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                        plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['all']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['all']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
            except:
                pass
        
        """
        new_orient_dict_x = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
        new_orient_dict_y = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))

        pos_x_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
        pos_y_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))

        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    for k in range(0, len(active_fa_dict['bin']['all']['x'])):
                        for l in range(0, len(active_fa_dict['bin']['all']['x'])):
                            count = 0
                            noncount = 0
                            for i in range(0, len(phase_dict['bin'])):
                                for j in range(0, len(phase_dict['bin'])):
                                    if (self.pos_x[i][j] <= self.pos_x[k+1][l] * 3) & (self.pos_x[i][j] >= self.pos_x[k][l] * 3) & (self.pos_y[i][j] <= self.pos_y[k][l+1] * 3) & (self.pos_y[i][j] >= self.pos_y[k][l] * 3):
                                        if phase_dict['bin'][i][j]==1:
                                            count += 1
                                        else:
                                            noncount += 1
                            if count > 0:
                                #print(k)
                                #print(l)
                                if (active_fa_dict['bin']['all']['x'][k][l] ** 2 + active_fa_dict['bin']['all']['y'][k][l] ** 2) ** 0.5 >=0.20:
                                    new_orient_dict_x[k][l]=active_fa_dict['bin']['all']['x'][k][l]
                                    new_orient_dict_y[k][l]=active_fa_dict['bin']['all']['y'][k][l]
                                    pos_x_new[k][l] = self.pos_x[k][l]*3
                                    pos_y_new[k][l] = self.pos_y[k][l]*3


                    active_r = ( np.array(new_orient_dict_x) ** 2 + np.array(new_orient_dict_y) ** 2 ) ** 0.5

                    nonzero_id = np.nonzero(active_r)
                    
                    nonzero_x_id = nonzero_id[0]
                    nonzero_y_id = nonzero_id[1]
                    
                    for i in range(0, len(nonzero_x_id)):
                        #plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                        plt.quiver(pos_x_new[nonzero_x_id[i]][nonzero_y_id[i]], pos_y_new[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_x[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_y[nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                    
            except:
                pass
        
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        # Label simulation time
        if presentation_id == True:
            if self.lx_box == self.ly_box:
                plt.text(0.647, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=28, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        else:
            if self.lx_box == self.ly_box:
                plt.text(0.68, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=28, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        

        eps_leg=[]
        mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
        msz=40

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)
        plt.xlim(25, self.lx_box-25)
        plt.ylim(25, self.ly_box-25)
        plt.scatter(self.hx_box, self.hy_box, c='black', s=70)

        # Add legend of phases
        
        if presentation_id == True:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=32)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.2, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.003, 1.025], handlelength=1.5, columnspacing=0.5, fontsize=36, ncol=3, facecolor='white', edgecolor='black', framealpha=0.6)
            ax.add_artist(one_leg)
        else:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=36)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.06, 1.17], handlelength=1.5, columnspacing=0.5, fontsize=40, ncol=3, facecolor='None', edgecolor='None')
            ax.add_artist(one_leg)
        plt.tight_layout()
        ax.set_facecolor('#F2f2f2')
        plt.savefig(self.outPath + 'phases_heterogeneity_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
        plt.close()  

    def plot_phases(self, pos, phase_ids_dict, phase_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False, banner_id = False, presentation_id=False, large_arrows_id=False):
        #DONE!
        """
        This function plots each particle's position and color-codes each
        particle by the phase it is a part of, i.e. bulk=green,
        interface=purple, and gas=red.

        Inputs:
        pos: array (partNum, 3) of each particles x,y,z positions
        phase_ids_dict: dictionary (output from various phase_identification
        functions) containing information on the composition of each phase

        phase_ids_dict: dictionary (output from phase_part_count() in
        phase_identification.py) that contains the id of each particle specifying
        the phase of bulk (0), interface (1), or gas (2).

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with each particle's position plotted and color coded by the
        phase it's a part of and the interface surfaces plotted in black
        """

        yellow = ("#7570b3")
        yellow = ("#ffd700")
        red = ("#fd8d3c")
        red = ("#dd571c")
        red = ("#ff7700")
        red = ("#cc5500")
        red = ("#f66e8d")
        green = ("#77dd77")
        green = ("#4a3728")
        green = ("#3c4142")
        green = ("#878787")
        green = ("#71797E")

        

        green = ("#cc5500")
        #green = ("#b3de69")
        #red = ("#ff6961")
        #red = ("#000000")

        green = ("#4d9221")
        red = ("#cc5500")
        yellow = ("#71797E")

        bulk_part_ids = phase_ids_dict['bulk']['all']
        gas_part_ids = phase_ids_dict['gas']['all']
        int_part_ids = phase_ids_dict['int']['all']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            if presentation_id == True:
                x_dim = scaling+0.18
            else:
                x_dim = scaling + 0.3
            y_dim = scaling

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755
        
        # Plot position colored by phase
        if len(bulk_part_ids)>0:
            ells_bulk = [Ellipse(xy=np.array([pos[bulk_part_ids[i],0]+self.hx_box,pos[bulk_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(bulk_part_ids))]
            bulkGroup = mc.PatchCollection(ells_bulk, facecolor=green)
            ax.add_collection(bulkGroup)
        
        if len(gas_part_ids)>0:
            ells_gas = [Ellipse(xy=np.array([pos[gas_part_ids[i],0]+self.hx_box,pos[gas_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(gas_part_ids))]
            gasGroup = mc.PatchCollection(ells_gas, facecolor=red)
            ax.add_collection(gasGroup)
        
        if len(int_part_ids)>0:
            ells_int = [Ellipse(xy=np.array([pos[int_part_ids[i],0]+self.hx_box,pos[int_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(int_part_ids))]

            intGroup = mc.PatchCollection(ells_int, facecolor=yellow)
            ax.add_collection(intGroup)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        
        # Plot averaged, binned orientation of particles
        
        
        if orientation_id == True:
            if large_arrows_id == True:

                new_orient_dict_x = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
                new_orient_dict_y = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))

                pos_x_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
                pos_y_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
                try:
                    if active_fa_dict!=None:
                        for k in range(0, len(active_fa_dict['bin']['all']['x'])):
                            for l in range(0, len(active_fa_dict['bin']['all']['y'])):
                                count = 0
                                noncount = 0
                                '''
                                for i in range(0, len(phase_dict['bin'])):
                                    for j in range(0, len(phase_dict['bin'])):
                                        if (self.pos_x[i][j] <= self.pos_x[k+1][l] * 3) & (self.pos_x[i][j] >= self.pos_x[k][l] * 3) & (self.pos_y[i][j] <= self.pos_y[k][l+1] * 3) & (self.pos_y[i][j] >= self.pos_y[k][l] * 3):
                                            if phase_dict['bin'][i][j]==1:
                                                count += 1
                                            else:
                                                noncount += 1
                                '''
                                #if count > noncount/4:
                                    #print(k)
                                    #print(l)
                                    #if ((active_fa_dict['bin']['all']['x'][k][l] ** 2 + active_fa_dict['bin']['all']['y'][k][l] ** 2) ** 0.5 >=0.20) | (count > noncount):
                                new_orient_dict_x[k][l]=active_fa_dict['bin']['all']['x'][k][l]
                                new_orient_dict_y[k][l]=active_fa_dict['bin']['all']['y'][k][l]
                                pos_x_new[k][l] = self.pos_x[k][l]*3
                                pos_y_new[k][l] = self.pos_y[k][l]*3


                        active_r = ( np.array(new_orient_dict_x) ** 2 + np.array(new_orient_dict_y) ** 2 ) ** 0.5

                        nonzero_id = np.nonzero(active_r)
                        
                        nonzero_x_id = nonzero_id[0]
                        nonzero_y_id = nonzero_id[1]
                        
                        for i in range(0, len(nonzero_x_id)):
                            #plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                            plt.quiver(pos_x_new[nonzero_x_id[i]][nonzero_y_id[i]], pos_y_new[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_x[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_y[nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                        
                except:
                    pass
            else:
                try:
                    if active_fa_dict!=None:
                        active_r = ( active_fa_dict['bin']['x'] ** 2 + active_fa_dict['bin']['y'] ** 2 ) ** 0.5

                        nonzero_id = np.nonzero(active_r)
        
                        nonzero_x_id = nonzero_id[0]
                        nonzero_y_id = nonzero_id[1]

                        plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8, width=0.004)
                except:
                    pass
        
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        # Label simulation time
        """
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

                if self.tst<10:
                    plt.text(0.68, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif (self.tst>=10) & (self.tst<100):
                    plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif (self.tst>=100):
                    plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        """

        eps_leg=[]
        mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
        msz=40

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)
        
        #plt.scatter(self.hx_box, self.hy_box, c='black', s=70)

        # Add legend of phases
        
        if presentation_id == True:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=32)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.2, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.003, 1.025], handlelength=1.5, columnspacing=0.5, fontsize=36, ncol=3, facecolor='white', edgecolor='black', framealpha=0.6)
            ax.add_artist(one_leg)
        else:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=36)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.04, 1.16], handlelength=1.5, columnspacing=0.5, fontsize=40, ncol=3, facecolor='None', edgecolor='None')
            ax.add_artist(one_leg)
        
        plt.tight_layout()
        #ax.set_facecolor('#F2f2f2')
        plt.savefig(self.outPath + 'phases_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
        plt.close()  
    
    def plot_tracers(self, pos, phase_ids_dict, phase_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False, presentation_id=False, tracer_ids = []):
        #DONE!
        """
        This function plots each particle's position and color-codes each
        particle by the phase it is a part of, i.e. bulk=green,
        interface=purple, and gas=red.

        Inputs:
        pos: array (partNum, 3) of each particles x,y,z positions
        phase_ids_dict: dictionary (output from various phase_identification
        functions) containing information on the composition of each phase

        phase_ids_dict: dictionary (output from phase_part_count() in
        phase_identification.py) that contains the id of each particle specifying
        the phase of bulk (0), interface (1), or gas (2).

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with each particle's position plotted and color coded by the
        phase it's a part of and the interface surfaces plotted in black
        """

        import random

        gray = ("#d9d9d9")

        bulk_part_ids = phase_ids_dict['bulk']['all']
        gas_part_ids = phase_ids_dict['gas']['all']
        int_part_ids = phase_ids_dict['int']['all']
        print(gas_part_ids)
        print(type(gas_part_ids))
        stop

        if len(tracer_ids) == 0:
            tracer_ids = random.sample(gas_part_ids, (len(gas_part_ids)/10))
        
        print(tracer_ids)
        stop

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            if presentation_id == True:
                x_dim = scaling+0.18
            else:
                x_dim = int(scaling + 0.5)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Plot position colored by phase
        if len(bulk_part_ids)>0:
            ells_bulk = [Ellipse(xy=np.array([pos[bulk_part_ids[i],0]+self.hx_box,pos[bulk_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(bulk_part_ids))]
            bulkGroup = mc.PatchCollection(ells_bulk, facecolor=green)
            ax.add_collection(bulkGroup)

        if len(gas_part_ids)>0:
            ells_gas = [Ellipse(xy=np.array([pos[gas_part_ids[i],0]+self.hx_box,pos[gas_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(gas_part_ids))]
            gasGroup = mc.PatchCollection(ells_gas, facecolor=red)
            ax.add_collection(gasGroup)

        if len(int_part_ids)>0:
            ells_int = [Ellipse(xy=np.array([pos[int_part_ids[i],0]+self.hx_box,pos[int_part_ids[i],1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(int_part_ids))]

            intGroup = mc.PatchCollection(ells_int, facecolor=yellow)
            ax.add_collection(intGroup)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        #active_r = ( active_fa_dict['bin']['x'] ** 2 + active_fa_dict['bin']['y'] ** 2 ) ** 0.5
        active_r = ( np.array(active_fa_dict['bin']['all']['x']) ** 2 + np.array(active_fa_dict['bin']['all']['y']) ** 2 ) ** 0.5
        nonzero_id = np.nonzero(active_r)
        
        nonzero_x_id = nonzero_id[0]
        nonzero_y_id = nonzero_id[1]
        # Plot averaged, binned orientation of particles
        """
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    for i in range(0, len(nonzero_x_id)):
                        #plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                        plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['all']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['all']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
            except:
                pass
        
        """
        new_orient_dict_x = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
        new_orient_dict_y = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))

        pos_x_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))
        pos_y_new = np.zeros(np.shape(active_fa_dict['bin']['all']['x']))

        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    for k in range(0, len(active_fa_dict['bin']['all']['x'])):
                        for l in range(0, len(active_fa_dict['bin']['all']['x'])):
                            count = 0
                            noncount = 0
                            for i in range(0, len(phase_dict['bin'])):
                                for j in range(0, len(phase_dict['bin'])):
                                    if (self.pos_x[i][j] <= self.pos_x[k+1][l] * 3) & (self.pos_x[i][j] >= self.pos_x[k][l] * 3) & (self.pos_y[i][j] <= self.pos_y[k][l+1] * 3) & (self.pos_y[i][j] >= self.pos_y[k][l] * 3):
                                        if phase_dict['bin'][i][j]==1:
                                            count += 1
                                        else:
                                            noncount += 1
                            if count > 0:
                                #print(k)
                                #print(l)
                                if (active_fa_dict['bin']['all']['x'][k][l] ** 2 + active_fa_dict['bin']['all']['y'][k][l] ** 2) ** 0.5 >=0.20:
                                    new_orient_dict_x[k][l]=active_fa_dict['bin']['all']['x'][k][l]
                                    new_orient_dict_y[k][l]=active_fa_dict['bin']['all']['y'][k][l]
                                    pos_x_new[k][l] = self.pos_x[k][l]*3
                                    pos_y_new[k][l] = self.pos_y[k][l]*3


                    active_r = ( np.array(new_orient_dict_x) ** 2 + np.array(new_orient_dict_y) ** 2 ) ** 0.5

                    nonzero_id = np.nonzero(active_r)
                    
                    nonzero_x_id = nonzero_id[0]
                    nonzero_y_id = nonzero_id[1]
                    
                    for i in range(0, len(nonzero_x_id)):
                        #plt.quiver(self.pos_x[nonzero_x_id[i]][nonzero_y_id[i]]*3, self.pos_y[nonzero_x_id[i]][nonzero_y_id[i]]*3, active_fa_dict['bin']['x'][nonzero_x_id[i]][nonzero_y_id[i]], active_fa_dict['bin']['y'][nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                        plt.quiver(pos_x_new[nonzero_x_id[i]][nonzero_y_id[i]], pos_y_new[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_x[nonzero_x_id[i]][nonzero_y_id[i]], new_orient_dict_y[nonzero_x_id[i]][nonzero_y_id[i]], scale=10.0, width=0.01, color='black', alpha=0.8)
                    
            except:
                pass
        
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        # Label simulation time
        if presentation_id == True:
            if self.lx_box == self.ly_box:
                plt.text(0.647, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=28, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        else:
            if self.lx_box == self.ly_box:
                plt.text(0.68, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=28, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        

        eps_leg=[]
        mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
        msz=40

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        # Add legend of phases
        
        if presentation_id == True:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=32)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.2, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.003, 1.025], handlelength=1.5, columnspacing=0.5, fontsize=36, ncol=3, facecolor='white', edgecolor='black', framealpha=0.6)
            ax.add_artist(one_leg)
        else:
            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=green, label='Bulk', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=yellow, label='Interface', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=red, label='Gas', markersize=36)]

            one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.06, 1.17], handlelength=1.5, columnspacing=0.5, fontsize=40, ncol=3, facecolor='None', edgecolor='None')
            ax.add_artist(one_leg)
        plt.tight_layout()
        ax.set_facecolor('#F2f2f2')
        plt.savefig(self.outPath + 'gas_tracers_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
        plt.close()  

    def plot_area_fraction(self, area_frac_dict, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):#, int_comp_dict):#sep_surface_dict, int_comp_dict):
        #DONE!
        """
        This function plots the binned average area fraction at each location
        in space.

        Inputs:
        area_frac_dict: dictionary (output from various bin_area_frac() in
        binning.py) containing information on the average area fraction of all,
        type A, type B, and the difference of type B and type A particles  in
        addition to the fraction of type A particles

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with binned average area fraction at each location
        in space plotted as a heat map with color bar
        """

        # Area fraction as specified by 'type' input
        if type == 'all':
            area_frac = area_frac_dict['bin']['all']
        elif type == 'A':
            area_frac = area_frac_dict['bin']['A']
        elif type == 'B':
            area_frac = area_frac_dict['bin']['B']
        elif type == 'dif':
            area_frac = area_frac_dict['bin']['dif']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Contour plot properties
        if type == 'dif':
            min_area_frac = -np.amax(area_frac)
            max_area_frac = np.amax(area_frac)
        else:
            min_area_frac = 0.0
            max_area_frac = np.max(area_frac)

        levels_text=40
        level_boundaries = np.linspace(min_area_frac, max_area_frac, levels_text + 1)

        # Contour plot of respective area fraction
        if type == 'dif':
            im = plt.contourf(self.pos_x, self.pos_y, area_frac, level_boundaries, vmin=min_area_frac, vmax=max_area_frac, cmap='seismic', extend='both')
        else:
            im = plt.contourf(self.pos_x, self.pos_y, area_frac, level_boundaries, vmin=min_area_frac, vmax=max_area_frac, cmap='Blues', extend='both')


        # modify color bar properties
        norm= matplotlib.colors.Normalize(vmin=min_area_frac, vmax=max_area_frac)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])

        tick_lev = np.arange(min_area_frac, max_area_frac+max_area_frac/10, (max_area_frac-min_area_frac)/10)

        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}-\phi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}-\phi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'density_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def plot_normal_fa_map(self, normal_fa_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):
        
        """
        This function plots the average alignment (toward cluster surface
        normal) of particles of a given type per bin at each location in space.

        Inputs:
        align_dict: dictionary (output from various bin_align() in
        binning.py) containing information on the average alignment of particles'
        active force orientation with the surface normal, where a positive value
        points more towards the surface and a negative value points more away
        from a surface.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pos (partNum, 3): array of (x,y,z) positions for each particle

        interface_id ( default value = True): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        Outputs:
        .png file with binned average alignment of respective type at each
        location in space plotted as a heat map with color bar and a quiver plot
        showing the average direction (magnitude normalized) of orientation per
        bin.
        """

        # Area fraction as specified by 'type' input
        if type == 'all':
            normal_fa = normal_fa_dict['bin']['all']
        elif type == 'A':
            normal_fa = normal_fa_dict['bin']['A']
        elif type == 'B':
            normal_fa = normal_fa_dict['bin']['B']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.25)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Define colorbar limits
        div_min = -3
        min_n = 0.0
        max_n = np.max(normal_fa)

        # Define colorbar ticks
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)

        # Plot normal force of given species
        im = plt.contourf(self.pos_x, self.pos_y, normal_fa, level_boundaries, vmin=min_n, vmax=max_n, cmap='Reds', extend='both')
        
        # Set colorbar properties
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)

        # Plot colorbar 
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{all} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{all} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{A} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{A} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{B} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{B} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Remove plot ticks
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'com_normal_fa_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def plot_normal_fa_heterogeneity_map(self, normal_fa_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False, type_m = 'phases'):
        
        """
        This function plots the average alignment (toward cluster surface
        normal) of particles of a given type per bin at each location in space.

        Inputs:
        align_dict: dictionary (output from various bin_align() in
        binning.py) containing information on the average alignment of particles'
        active force orientation with the surface normal, where a positive value
        points more towards the surface and a negative value points more away
        from a surface.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pos (partNum, 3): array of (x,y,z) positions for each particle

        interface_id ( default value = True): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        Outputs:
        .png file with binned average alignment of respective type at each
        location in space plotted as a heat map with color bar and a quiver plot
        showing the average direction (magnitude normalized) of orientation per
        bin.
        """

        # Area fraction as specified by 'type' input
        if type == 'all':
            normal_fa = normal_fa_dict['all']
        elif type == 'A':
            normal_fa = normal_fa_dict['A']
        elif type == 'B':
            normal_fa = normal_fa_dict['B']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.25)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Define colorbar limits
        div_min = -3
        min_n = np.min(normal_fa)
        max_n = np.max(normal_fa)

        # Define colorbar ticks
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)

        # Plot normal force of given species
        im = plt.contourf(normal_fa_dict['x'], normal_fa_dict['y'], normal_fa, level_boundaries, vmin=min_n, vmax=max_n, cmap='Reds', extend='both')
        
        # Set colorbar properties
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)

        # Plot colorbar 
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{all} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{all} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{A} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{A} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{B} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\langle F_\mathrm{B} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Remove plot ticks
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.52, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        ax.axis('off')
        plt.tight_layout()
        if type_m == 'phases':
            plt.savefig(self.outPath + 'com_normal_fa_heterogeneity_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif type_m == 'system':
            plt.savefig(self.outPath + 'com_normal_fa_system_heterogeneity_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def plot_particle_fraction(self, num_dens_dict, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='A', interface_id = False, orientation_id = False):
        #DONE!
        """
        This function plots the fraction of particles of a given type per bin
        at each location in space.

        Inputs:
        num_dens_dict: dictionary (output from various bin_area_frac() in
        binning.py) containing information on the average area fraction of all,
        type A, type B, and the difference of type B and type A particles in
        addition to the fraction of type A particles

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with binned particle fraction of respective type at each
        location in space plotted as a heat map with color bar
        """

        if type == 'A':
            part_frac = num_dens_dict['bin']['fast frac']
        elif type == 'B':
            part_frac = np.ones(np.shape(num_dens_dict['bin']['fast frac'])) - num_dens_dict['bin']['fast frac']
        elif type == 'dif':
            part_frac = num_dens_dict['bin']['fast frac'] - (np.ones(np.shape(num_dens_dict['bin']['fast frac'])) - num_dens_dict['bin']['fast frac'])

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Contour plot properties
        if type == 'dif':
            min_part_frac = -1.0
            max_part_frac = 1.0
        else:
            min_part_frac = 0.0
            max_part_frac = 1.0

        levels_text=40
        level_boundaries = np.linspace(min_part_frac, max_part_frac, levels_text + 1)

        # Contour plot of respective particle fraction
        im = plt.contourf(self.pos_x, self.pos_y, part_frac, level_boundaries, vmin=min_part_frac, vmax=max_part_frac, cmap='seismic', extend='both')

        # Modify color bar properties
        norm= matplotlib.colors.Normalize(vmin=min_part_frac, vmax=max_part_frac)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_part_frac, max_part_frac+max_part_frac/10, (max_part_frac-min_part_frac)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}-\chi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}-\chi_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass

        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label simulation time
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.61, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.84, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_frac_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def plot_alignment(self, align_dict, sep_surface_dict, int_comp_dict, pos, interface_id = True, type='all', method='com'):
        #DONE!
        """
        This function plots the average alignment (toward cluster surface
        normal) of particles of a given type per bin at each location in space.

        Inputs:
        align_dict: dictionary (output from various bin_align() in
        binning.py) containing information on the average alignment of particles'
        active force orientation with the surface normal, where a positive value
        points more towards the surface and a negative value points more away
        from a surface.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pos (partNum, 3): array of (x,y,z) positions for each particle

        interface_id ( default value = True): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        method (optional): string specifying whether the alignment input is calculated
        using the cluster's center of mass ('com') or the cluster's surface normal ('surface).

        Outputs:
        .png file with binned average alignment of respective type at each
        location in space plotted as a heat map with color bar and a quiver plot
        showing the average direction (magnitude normalized) of orientation per
        bin.
        """

        # Assign alignment based on particle type specified
        if type == 'all':
            align = align_dict['bin']['all']['mag']
        elif type == 'A':
            align = align_dict['bin']['A']['mag']
        elif type == 'B':
            align = align_dict['bin']['B']['mag']
        elif type == 'dif':
            align = align_dict['bin']['B']['mag'] - align_dict['bin']['A']['mag']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Contour plot properties
        if type == 'dif':
            min_align = -np.amax(align_dif)
            max_align = np.amax(align_dif)
        else:
            min_align = -1.0
            max_align = 1.0

        levels_text=40
        level_boundaries = np.linspace(min_align, max_align, levels_text + 1)

        # Plot particle binned properties
        im = plt.contourf(self.pos_x, self.pos_y, align, level_boundaries, vmin=min_align, vmax=max_align, cmap='seismic', extend='both')
        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)

        # Modify color bar properties
        norm= matplotlib.colors.Normalize(vmin=min_align, vmax=max_align)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_align, max_align+max_align/10, (max_align-min_align)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
                
        # Plot cluster interior and exterior surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        

        # Label simulation time
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Modify plot parameters

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        if method == 'com':
            plt.savefig(self.outPath + 'com_alignment_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        else:
            plt.savefig(self.outPath + 'surface_alignment_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()
    
    def plot_alignment_force(self, align_dict, sep_surface_dict, int_comp_dict, pos, interface_id = True, type='all', method='com'):
        #DONE!
        """
        This function plots the average alignment (toward cluster surface
        normal) of particles of a given type per bin at each location in space.

        Inputs:
        align_dict: dictionary (output from various bin_align() in
        binning.py) containing information on the average alignment of particles'
        active force orientation with the surface normal, where a positive value
        points more towards the surface and a negative value points more away
        from a surface.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pos (partNum, 3): array of (x,y,z) positions for each particle

        interface_id ( default value = True): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        method (optional): string specifying whether the alignment input is calculated
        using the cluster's center of mass ('com') or the cluster's surface normal ('surface).

        Outputs:
        .png file with binned average alignment of respective type at each
        location in space plotted as a heat map with color bar and a quiver plot
        showing the average direction (magnitude normalized) of orientation per
        bin.
        """

        # Assign alignment based on particle type specified
        if type == 'all':
            align = align_dict['bin']['all']['fa_mag']
        elif type == 'A':
            align = align_dict['bin']['A']['fa_mag']
        elif type == 'B':
            align = align_dict['bin']['B']['fa_mag']
        elif type == 'dif':
            align = align_dict['bin']['B']['fa_mag'] - align_dict['bin']['A']['fa_mag']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Contour plot properties
        if type == 'dif':
            min_align = -np.amax(align_dif)
            max_align = np.amax(align_dif)
        else:
            min_align = np.min(align)
            max_align = np.max(align)

        levels_text=40
        level_boundaries = np.linspace(min_align, max_align, levels_text + 1)
        print(level_boundaries)
        print(min_align)
        print(max_align)
        # Plot particle binned properties
        im = plt.contourf(self.pos_x, self.pos_y, align, level_boundaries, vmin=min_align, vmax=max_align, cmap='seismic', extend='both')
        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)

        # Modify color bar properties
        norm= matplotlib.colors.Normalize(vmin=min_align, vmax=max_align)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_align, max_align+max_align/10, (max_align-min_align)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
                
        # Plot cluster interior and exterior surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        

        # Label simulation time
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Modify plot parameters

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        if method == 'com':
            plt.savefig(self.outPath + 'com_alignment_force_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        else:
            plt.savefig(self.outPath + 'surface_alignment_force_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def lat_histogram(self, lat_plot_dict):
        """
        This function plots a histogram of the lattice spacings of bulk (green)
        and interface (purple) particles

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        Outputs:
        .png file with a histogram of all bulk (green) and interface (purple)
        particle lattice spacings color coded.
        """

        bulk_lats = lat_plot_dict['bulk']['all']['vals']
        int_lats = lat_plot_dict['int']['all']['vals']

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")

        


        #Remove bulk particles that are outside plot's xrange
        bulk_lat_mean = np.mean(lat_plot_dict['bulk']['all']['vals'])

        xmin = 0.85*bulk_lat_mean
        xmax = 1.15*bulk_lat_mean

        if (len(bulk_lats)>0):
            bulk_id = np.where((bulk_lats > xmax) | (bulk_lats < xmin))[0]
            bulk_lats = np.delete(bulk_lats, bulk_id)

            plt.hist(bulk_lats, alpha = 1.0, bins=100, color=green)

        #If interface particle measured, include it in histogram
        if (len(int_lats)>0):
            int_id = np.where((int_lats > xmax) | (int_lats < xmin))[0]
            int_lats = np.delete(int_lats, int_id)

            plt.hist(int_lats, alpha = 0.8, bins=100, color=yellow)

        # Create legend of phases
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        # Label current time step
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Modify plot parameters
        plt.xlabel(r'lattice spacing ($a$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.tight_layout()
        plt.savefig(self.outPath + 'lat_histo_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()

    def prob_histogram(self, prob_plot_dict, prob_stat_dict):
        """
        This function plots a histogram of the lattice spacings of bulk (green)
        and interface (purple) particles

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        Outputs:
        .png file with a histogram of all bulk (green) and interface (purple)
        particle lattice spacings color coded.
        """

        bulk_lats = prob_plot_dict['bulk']['AA']
        int_lats = prob_plot_dict['int']['AA']

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")

        xmin = -0.5
        xmax=7.5

        bins_1 = [-0.4, 0.4, 0.6, 1.4, 1.6, 2.4, 2.6, 3.4, 3.6, 4.4, 4.6, 5.4, 5.6, 6.4, 6.6, 7.4]
        bins_2 = [-0.3, 0.3, 0.7, 1.3, 1.7, 2.3, 2.7, 3.3, 3.7, 4.3, 4.7, 5.3, 5.7, 6.3, 6.7, 7.3]
        bins_3 = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2]
        bins_4 = [-0.1, 0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9, 5.1, 5.9, 6.1, 6.9, 7.1]

        if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
            
            plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_1, color='blue', rwidth=1.0, density=False, edgecolor='black', linewidth=1.5)
            if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_2, color='purple', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_2, color='teal', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_2, color='red', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
        elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
            plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_1, color='purple', rwidth=1.0, density=False, edgecolor='black', linewidth=1.5)
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_2, color='blue', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_2, color='teal', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_2, color='red', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)
        elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
            plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_1, color='teal', rwidth=1.0, density=False, edgecolor='black', linewidth=1.5)
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_2, color='blue', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_2, color='purple', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_4, color='red', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_3, color='red', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_2, color='red', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)
        elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
            plt.hist(prob_plot_dict['bulk']['BB'], alpha = 0.3, bins=bins_1, color='red', rwidth=1.0, density=False, edgecolor='black', linewidth=1.5)
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_2, color='blue', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_2, color='purple', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_4, color='teal', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_3, color='teal', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                plt.hist(prob_plot_dict['bulk']['BA'], alpha = 0.3, bins=bins_2, color='teal', rwidth=1.0, density=False)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_3, color='blue', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_4, color='purple', rwidth=1.0, density=False)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.hist(prob_plot_dict['bulk']['AB'], alpha = 0.3, bins=bins_3, color='purple', rwidth=1.0, density=False)
                    plt.hist(prob_plot_dict['bulk']['AA'], alpha = 0.3, bins=bins_4, color='blue', rwidth=1.0, density=False)

        # Create legend of phases
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        # Label current time step
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Modify plot parameters
        plt.xlabel(r'lattice spacing ($a$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.tight_layout()
        plt.savefig(self.outPath + 'plot_prob_histo_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()

    def prob_histogram2(self, prob_plot_dict, prob_stat_dict, avg=False):
        """
        This function plots a histogram of the lattice spacings of bulk (green)
        and interface (purple) particles

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        Outputs:
        .png file with a histogram of all bulk (green) and interface (purple)
        particle lattice spacings color coded.
        """

        bulk_lats = prob_plot_dict['bulk']['AA']
        int_lats = prob_plot_dict['int']['AA']

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")

        xmin = -0.5
        xmax=7.5

        slowslowCol = '#045a8d'
        fastslowCol = '#a6bddb'
        fastfastCol = '#b30000'
        slowfastCol = '#fdb884'

        bins_1 = [-0.4, 0.4, 0.6, 1.4, 1.6, 2.4, 2.6, 3.4, 3.6, 4.4, 4.6, 5.4, 5.6, 6.4, 6.6, 7.4]
        bins_2 = [-0.3, 0.3, 0.7, 1.3, 1.7, 2.3, 2.7, 3.3, 3.7, 4.3, 4.7, 5.3, 5.7, 6.3, 6.7, 7.3]
        bins_3 = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2]
        bins_4 = [-0.1, 0.1, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9, 5.1, 5.9, 6.1, 6.9, 7.1]

        if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
            
            plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.9, edgecolor='black', linewidth=1.5)
            
            if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            
        elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
            plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.9, edgecolor='black', linewidth=1.5)
            
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            
        elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['BB'])):
            plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.9, edgecolor='black', linewidth=1.5)
            
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            
        elif (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['BB']) >= np.max(prob_stat_dict['bulk']['BA'])):
            plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BB'], alpha = 1.0, color=fastfastCol, width=0.9, edgecolor='black', linewidth=1.5)
            
            if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])) & (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['BA'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['BA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            elif (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AA'])) & (np.max(prob_stat_dict['bulk']['BA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['BA'], alpha = 1.0, color=fastslowCol, width=0.7, edgecolor='black', linewidth=1.5)
                if (np.max(prob_stat_dict['bulk']['AA']) >= np.max(prob_stat_dict['bulk']['AB'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.3, edgecolor='black', linewidth=1.5)
                elif (np.max(prob_stat_dict['bulk']['AB']) >= np.max(prob_stat_dict['bulk']['AA'])):
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AB'], alpha = 1.0, color=slowfastCol, width=0.5, edgecolor='black', linewidth=1.5)
                    plt.bar(prob_stat_dict['neigh'], prob_stat_dict['bulk']['AA'], alpha = 1.0, color=slowslowCol, width=0.3, edgecolor='black', linewidth=1.5)
            
        # Create legend of phases
        slowslow_patch = mpatches.Patch(color=slowslowCol, label='Slow-Slow')
        fastslow_patch = mpatches.Patch(color=fastslowCol, label='Fast-Slow')
        fastfast_patch = mpatches.Patch(color=fastfastCol, label='Fast-Fast')
        slowfast_patch = mpatches.Patch(color=slowfastCol, label='Slow-Fast')
        plt.legend(handles=[slowslow_patch, fastslow_patch, fastfast_patch, slowfast_patch], fancybox=False, ncol=4, fontsize=21, bbox_transform=ax.transAxes, bbox_to_anchor=[0.1, 1.13], labelspacing=0.1, handletextpad=0.1, facecolor='None', edgecolor='None')

        # Label current time step
        if avg == False:
            plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=21,transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Modify plot parameters
        plt.xlabel(r'Number of Neighbors', fontsize=21)
        plt.ylabel('Probability', fontsize=21)
        plt.xlim([xmin,xmax])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        plt.savefig(self.outPath + 'plot_prob_histo_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()
    
    def vel_histogram(self, vel_plot_dict, dt_step, avg='False'):
        """
        This function plots a histogram of the lattice spacings of bulk (green)
        and interface (purple) particles

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        Outputs:
        .png file with a histogram of all bulk (green) and interface (purple)
        particle lattice spacings color coded.
        """

        A_vel = vel_plot_dict['A']['mag']
        B_vel = vel_plot_dict['B']['mag']

        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111)

        #Define colors for plots
        fastCol = '#e31a1c'
        slowCol = '#081d58'

        xmin = np.min(A_vel)-5
        xmax = np.max(B_vel)+5

        yA, xA, _ = plt.hist(A_vel, bins=100, color=slowCol, range=[xmin, xmax], alpha=0.2, density=True)
        yB, xB, _ = plt.hist(B_vel, bins=100, color=fastCol, range=[xmin, xmax], alpha=0.2, density=True)

        if np.max(yB) >= np.max(yA):
            y_max = 1.1 * np.max(yB)      
        else:
            y_max = 1.1 * np.max(yA)

        ax.plot([self.peA, self.peA], [0, y_max], color=slowCol, linestyle='solid', linewidth=3.0)
        ax.plot([self.peB, self.peB], [0, y_max], color=fastCol, linestyle='solid', linewidth=3.0)
        ax.plot([np.mean(A_vel), np.mean(A_vel)], [0, y_max], color=slowCol, linestyle='dashed', linewidth=3.0)
        ax.plot([np.median(A_vel), np.median(A_vel)], [0, y_max], color=slowCol, linestyle='dotted', linewidth=3.0)
        ax.plot([np.mean(B_vel), np.mean(B_vel)], [0, y_max], color=fastCol, linestyle='dashed', linewidth=3.0)
        ax.plot([np.median(B_vel), np.median(B_vel)], [0, y_max], color=fastCol, linestyle='dotted', linewidth=3.0)
        
        fast_leg = []
        fast_leg.append(Line2D([0], [0], linestyle='solid', lw = 3.0, color=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=25))
        fast_leg.append(Line2D([0], [0], linestyle='solid', lw = 3.0, color=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=25))
        fast_leg.append(Line2D([0], [0], linestyle='dashed', lw = 3.0, color=slowCol, label=r'$\overline{v}_\mathrm{S} = $'+str(int(np.mean(A_vel))), markersize=25))
        fast_leg.append(Line2D([0], [0], linestyle='dashed', lw = 3.0, color=fastCol, label=r'$\overline{v}_\mathrm{F} = $'+str(int(np.mean(B_vel))), markersize=25))
        fast_leg.append(Line2D([0], [0], linestyle='dotted', lw = 3.0, color=slowCol, label=r'$\widetilde{v}_\mathrm{S} = $'+str(int(np.median(A_vel))), markersize=25))
        fast_leg.append(Line2D([0], [0], linestyle='dotted', lw = 3.0, color=fastCol, label=r'$\widetilde{v}_\mathrm{F} = $'+str(int(np.median(B_vel))), markersize=25))
        one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.2, handletextpad=0.3, bbox_transform=ax.transAxes, bbox_to_anchor=[0.75, 1.23], handlelength=1.5, columnspacing=0.9, fontsize=25, ncol=3, facecolor='None', edgecolor='None')
        ax.add_artist(one_leg)

        # Create legend of phases
        blue_patch = mpatches.Patch(color=slowCol, label='A')
        red_patch = mpatches.Patch(color=fastCol, label='B')
        #plt.legend(handles=[blue_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        
        # Modify plot parameters
        ax.set_xlabel(r'Effective Velocity ($v$)', fontsize=25)
        ax.set_ylabel('Density', fontsize=25)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([0,y_max])
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        if avg=='False':
            # Label current time step
            plt.text(0.735, 0.92, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=25,transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.tight_layout()
        
            plt.savefig(self.outPath + 'plot_vel_histo_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
        
            plt.savefig(self.outPath + 'plot_avg_vel_histo_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
            plt.close()

        vel_histo_dict = {'A': {'x': xA, 'y': yA}, 'B': {'x': xB, 'y': yB}}
        vel_histo_stat_dict = {'A': {'mean': np.mean(A_vel), 'median': np.median(A_vel), 'std': np.std(A_vel)}, 'B': {'mean': np.mean(B_vel), 'median': np.median(B_vel), 'std': np.std(B_vel)}}

        return vel_histo_dict, vel_histo_stat_dict
    def cluster_size_histogram(self, collision_plot_dict, avg='False'):
        """
        This function plots a histogram of the lattice spacings of bulk (green)
        and interface (purple) particles

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        Outputs:
        .png file with a histogram of all bulk (green) and interface (purple)
        particle lattice spacings color coded.
        """

        A_size = collision_plot_dict['A']
        B_size = collision_plot_dict['B']

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Define colors for plots
        fastCol = '#e31a1c'
        slowCol = '#081d58'

        xmin = 1
        xmax = np.max(B_size)+1

        yA, xA, _ = plt.hist(A_size, bins=int(xmax-1), color=slowCol, range=[xmin, xmax], alpha=0.2)
        yB, xB, _ = plt.hist(B_size, bins=int(xmax-1), color=fastCol, range=[xmin, xmax], alpha=0.2)

        if np.max(yB) >= np.max(yA):
            y_max = 1.1 * np.max(yB)      
        else:
            y_max = 1.1 * np.max(yA)

        #ax.plot([self.peA, self.peA], [0, y_max], color=slowCol, linestyle='solid', linewidth=3.0)
        #ax.plot([self.peB, self.peB], [0, y_max], color=fastCol, linestyle='solid', linewidth=3.0)
        #ax.plot([np.mean(A_vel), np.mean(A_vel)], [0, y_max], color=slowCol, linestyle='dashed', linewidth=3.0)
        #ax.plot([np.median(A_vel), np.median(A_vel)], [0, y_max], color=slowCol, linestyle='dotted', linewidth=3.0)
        #ax.plot([np.mean(B_vel), np.mean(B_vel)], [0, y_max], color=fastCol, linestyle='dashed', linewidth=3.0)
        #ax.plot([np.median(B_vel), np.median(B_vel)], [0, y_max], color=fastCol, linestyle='dotted', linewidth=3.0)
        
        #fast_leg = []
        #fast_leg.append(Line2D([0], [0], linestyle='solid', lw = 3.0, color=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=25))
        #fast_leg.append(Line2D([0], [0], linestyle='solid', lw = 3.0, color=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=25))
        #fast_leg.append(Line2D([0], [0], linestyle='dashed', lw = 3.0, color=slowCol, label=r'$\overline{v}_\mathrm{S} = $'+str(int(np.mean(A_vel))), markersize=25))
        #fast_leg.append(Line2D([0], [0], linestyle='dashed', lw = 3.0, color=fastCol, label=r'$\overline{v}_\mathrm{F} = $'+str(int(np.mean(B_vel))), markersize=25))
        #fast_leg.append(Line2D([0], [0], linestyle='dotted', lw = 3.0, color=slowCol, label=r'$\widetilde{v}_\mathrm{S} = $'+str(int(np.median(A_vel))), markersize=25))
        #fast_leg.append(Line2D([0], [0], linestyle='dotted', lw = 3.0, color=fastCol, label=r'$\widetilde{v}_\mathrm{F} = $'+str(int(np.median(B_vel))), markersize=25))
        #one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.2, handletextpad=0.3, bbox_transform=ax.transAxes, bbox_to_anchor=[1.03, 1.23], handlelength=1.5, columnspacing=0.9, fontsize=25, ncol=3, facecolor='None', edgecolor='None')
        #ax.add_artist(one_leg)

        # Create legend of phases
        blue_patch = mpatches.Patch(color=slowCol, label='A')
        red_patch = mpatches.Patch(color=fastCol, label='B')
        #plt.legend(handles=[blue_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        
        # Modify plot parameters
        ax.set_xlabel(r'Cluster Size ($N_\mathrm{c}$)', fontsize=25)
        ax.set_ylabel('Density', fontsize=25)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([0,y_max])
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        if avg=='False':
            # Label current time step
            plt.text(0.735, 0.92, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=25,transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.tight_layout()
        
            plt.savefig(self.outPath + 'plot_clust_size_histo_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
        
            plt.savefig(self.outPath + 'plot_avg_clust_size_histo_' + self.outFile + ".png", dpi=150, transparent=False, bbox_inches='tight')
            plt.close()

        

    def lat_map(self, lat_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):
        """
        This function plots the lattice spacings of all dense phase particles
        at each location in space.

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the lattice spacing with color bar.
        """


        bulk_lat_mean = np.mean(lat_plot_dict['bulk']['all']['vals'])
        dense_lats = lat_plot_dict['dense']['all']

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(dense_lats['x']+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(dense_lats['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Find min/max orientation
        dense_lats = lat_plot_dict['dense']['all']
        min_lat = 0.85*bulk_lat_mean
        max_lat = 1.15*bulk_lat_mean

        # Set plotted particle size
        sz = 0.755
        
        ells = [Ellipse(xy=np.array([dense_lats['x'][i]+self.hx_box,dense_lats['y'][i]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(dense_lats['x']))]

        # Plot position colored by neighbor number
        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(dense_lats['vals']))
        
        minClb = min_lat
        maxClb = max_lat
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_lat, max_lat+1, (max_lat - min_lat)/5)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        #clb.set_label(r'$g_6(r)$', labelpad=-55, y=1.04, rotation=0, fontsize=18)

        clb.set_label(r'Lattice Spacing ($a$)', labelpad=25, y=0.5, rotation=270, fontsize=30)
        plt.title('All Reference Particles', fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.92, s=r'$\overline{a}$' + ' = ' + '{:.3f}'.format(bulk_lat_mean),
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.85, 0.92, s=r'$\overline{a}$' + ' = ' + '{:.3f}'.format(bulk_lat_mean),
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()

        plt.savefig(self.outPath + 'lat_map_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()

    def plot_general_rdf(self, radial_df_dict):
        #DONE!
        """
        This function plots the general radial distribution function of all particles
         within the bulk at current time step.

        Inputs:
        radial_df_dict: dictionary (output from radial_df() in
        measurement.py) containing information on the radial distribution
        function of combinations of particle types or all particles within the bulk.
        
        Outputs:
        .png file with general radial distribution function of all particles within
        the bulk irrespective of particle type at current time step.
        """

        fsize=10

        # Initialize rdf data
        rdf_allall = np.array(radial_df_dict['all-all'])
        rdf_aa = np.array(radial_df_dict['A-A'])
        rdf_ab = np.array(radial_df_dict['A-B'])
        rdf_bb = np.array(radial_df_dict['B-B'])

        fig, ax1 = plt.subplots(figsize=(12,6))

        # Define limits
        plot_max = 1.05 * np.max(radial_df_dict['all-all'])

        plot_min = -1.0

        # Find x-ticks
        rstop = 10.

        step = round(np.abs(plot_max - plot_min)/6,2)
       

        # Plot y=1.0
        x_arr = np.array([0.0,15.0])
        y_arr = np.array([1.0, 1.0])

        plt.plot(x_arr, y_arr, c='black', lw=1.0, ls='--')

        # Plot general rdf
        plt.plot(radial_df_dict['r'], radial_df_dict['all-all'],
                    c='black', lw=9.0, ls='-', alpha=1, label='All-All')

        # Set limits
        ax1.set_ylim(plot_min, plot_max)
        ax1.set_xlim(0, rstop)
        
        # Label axes
        ax1.set_xlabel(r'Separation Distance ($r$)', fontsize=fsize*2.8)
        ax1.set_ylabel(r'$g(r)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        x_tick_val = radial_df_dict['r'][np.where(rdf_allall==np.max(rdf_allall))[0][0]]
        loc = ticker.MultipleLocator(base=(x_tick_val*2))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(x_tick_val))
        ax1.xaxis.set_minor_locator(loc)
        ax1.tick_params(axis='x', labelsize=fsize*2.5)

        
        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)

        # Set legend
        plt.legend(loc='upper right', fontsize=fsize*2.6)
        
        plt.tight_layout()
        plt.savefig(self.outPath + 'plot_general_rdf_' + self.outFile + ".png", dpi=300)
        plt.close()

    def plot_all_rdfs(self, radial_df_dict):
        #DONE!
        """
        This function plots the partial radial distribution function (rdf) of each particle
        type pair within the bulk at current time step.

        Inputs:
        radial_df_dict: dictionary (output from radial_df() in
        measurement.py) containing information on the radial distribution
        function of combinations of particle types or all particles within the bulk.
        
        Outputs:
        .png file with partial radial distribution function (rdf) of each particle type 
        within the bulk at current time step.
        """
        fsize=10

        # Initialize rdf data
        rdf_allall = np.array(radial_df_dict['all-all'])
        rdf_aa = np.array(radial_df_dict['A-A'])
        rdf_ab = np.array(radial_df_dict['A-B'])
        rdf_bb = np.array(radial_df_dict['B-B'])

        plt.plot(radial_df_dict['r'], radial_df_dict['A-A'], c='blue')
        plt.plot(radial_df_dict['r'], radial_df_dict['B-B'], c='red')
        plt.show()
        stop

        #Find order for plotting overlapping data
        AA_bulk_max = np.max(rdf_aa)
        AB_bulk_max = np.max(rdf_ab)
        BB_bulk_max = np.max(rdf_bb)

        if (AA_bulk_max >= AB_bulk_max):
            if (AA_bulk_max >= BB_bulk_max):
                AA_bulk_order = 1
                if (BB_bulk_max >= AB_bulk_max):
                    BB_bulk_order = 2
                    AB_bulk_order = 3
                else:
                    AB_bulk_order = 2
                    BB_bulk_order = 3
            else:
                BB_bulk_order = 1
                AA_bulk_order = 2
                AB_bulk_order = 3
        else:
            if (AA_bulk_max >= BB_bulk_max):
                AB_bulk_order = 1
                AA_bulk_order = 2
                BB_bulk_order = 3
            else:
                AA_bulk_order = 3
                if (BB_bulk_max >= AB_bulk_max):
                    BB_bulk_order = 1
                    AB_bulk_order = 2
                else:
                    AB_bulk_order = 1
                    BB_bulk_order = 2
        
        # Find y-limits
        if AA_bulk_order == 1:
            plot_max = 1.05 * AA_bulk_max
        elif AB_bulk_order == 1:
            plot_max = 1.05 * AB_bulk_max
        elif BB_bulk_order == 1:
            plot_max = 1.05 * BB_bulk_max

        plot_min = -1.0

        # Calculate x-ticks
        rstop=10.0

        step = round(np.abs(plot_max - plot_min)/6,2)
        step_x = 1.5
        
        #Define plot colors
        purple = ("#00441b")
        orange = ("#ff7f00")
        green = ("#377EB8")

        fig, ax1 = plt.subplots(figsize=(12,6))
        
        first_width = 15.0
        second_width = 8.0
        third_width = 2.5

        # Plot partial rdf's in order of largest 1st-peak (widest line) to
        # shortest 1st-peak (thinnest line)
        if AA_bulk_order == 1:
            plt.plot(radial_df_dict['r'], rdf_aa, label=r'A-A',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AB_bulk_order == 2:
                plt.plot(radial_df_dict['r'], rdf_ab, label=r'A-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_bb, label=r'B-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(radial_df_dict['r'], rdf_bb, label=r'B-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_ab, label=r'A-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
        elif AB_bulk_order == 1:
            plt.plot(radial_df_dict['r'], rdf_ab, label=r'A-B',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AA_bulk_order == 2:
                plt.plot(radial_df_dict['r'], rdf_aa, label=r'A-A',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_bb, label=r'B-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(radial_df_dict['r'], rdf_bb, label=r'B-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_aa, label=r'A-A',
                            c=purple, lw=third_width, ls='-', alpha=1)
        elif BB_bulk_order == 1:
            plt.plot(radial_df_dict['r'], rdf_bb, label=r'B-B',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AA_bulk_order == 1:
                plt.plot(radial_df_dict['r'], rdf_aa, label=r'A-A',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_ab, label=r'A-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(radial_df_dict['r'], rdf_ab, label=r'A-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(radial_df_dict['r'], rdf_aa, label=r'A-A',
                            c=purple, lw=third_width, ls='-', alpha=1.0)

        # Plot y=1.0 line

        x_arr = np.array([0.0,10.0])
        y_arr = np.array([1.0, 1.0])

        plt.plot(x_arr, y_arr, c='black', lw=1.5, ls='--')

        # Set plot parameters
        ax1.set_ylim(plot_min, plot_max)
        ax1.set_xlim(0, rstop)

        ax1.set_xlabel(r'Separation Distance ($r$)', fontsize=fsize*2.8)

        ax1.set_ylabel(r'$g(r)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        x_tick_val = radial_df_dict['r'][np.where(rdf_allall==np.max(rdf_allall))[0][0]]
        loc = ticker.MultipleLocator(base=(x_tick_val*2))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(x_tick_val))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)

        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)

        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)

        # Set legend
        leg = [Line2D([0], [0], lw=8.0, c=green, label='A-A', linestyle='solid'), Line2D([0], [0], lw=8.0, c=orange, label='A-B', linestyle='solid'), Line2D([0], [0], lw=8.0, c=purple, label='B-B', linestyle='solid')]

        legend = ax1.legend(handles=leg, loc='upper right', columnspacing=1., handletextpad=0.3, bbox_transform=ax1.transAxes, bbox_to_anchor=[0.98, 1.03], fontsize=fsize*2.8, frameon=False, ncol=1)
        
        plt.tight_layout()
        plt.savefig(self.outPath + 'plot_partial_rdf_' + self.outFile+ ".png", dpi=300)
        plt.close()

    def plot_general_adf(self, angular_df_dict):
        """
        This function plots the general angular distribution function (adf) of all particles
         within the bulk at current time step.

        Inputs:
        angular_df_dict: dictionary (output from angular_df() in
        measurement.py) containing information on the angular distribution
        function of combinations of particle types or all particles within the bulk.
        
        Outputs:
        .png file with general angular distribution function (adf) of each particle type 
        within the bulk at current time step.
        """
        fsize=10

        # Initialize adf data
        adf_allall = np.array(angular_df_dict['all-all'])
        adf_aa = np.array(angular_df_dict['A-A'])
        adf_ab = np.array(angular_df_dict['A-B'])
        adf_bb = np.array(angular_df_dict['B-B'])

        fig, ax1 = plt.subplots(figsize=(12,6))

        # Define limits
        if 1.2 * np.max(angular_df_dict['all-all']) > 0:
            plot_max = 1.2 * np.max(angular_df_dict['all-all'])
        else:
            plot_max = 0.8 * np.max(angular_df_dict['all-all'])

        if np.min(angular_df_dict['all-all']) < 0:
            plot_min = 1.2 * np.min(angular_df_dict['all-all'])
        else:
            plot_min = 0.8 * np.min(angular_df_dict['all-all'])

        # Find x-ticks
        rstop=2*np.pi
        step = np.abs(plot_max - plot_min)/6

        # Plot general adf
        plt.plot(angular_df_dict['theta'], angular_df_dict['all-all'],
                    c='black', lw=9.0, ls='-', alpha=1, label='All-All')

        # Set limits
        ax1.set_ylim(plot_min, plot_max)
        ax1.set_xlim(0, rstop)

        # Label axes
        ax1.set_xlabel(r'Interparticle separation angle from $+\hat{\mathbf{x}}$ ($\theta$)', fontsize=fsize*2.8)
        ax1.set_ylabel(r'$g(\theta)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        loc = ticker.MultipleLocator(base=(np.pi/3))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(np.pi/6))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)

        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)

        # Set legend
        plt.legend(loc='upper right', fontsize=fsize*2.6)

        plt.tight_layout()
        plt.savefig(self.outPath + 'general_adf_' + self.outFile + ".png", dpi=300)
        plt.close()

    def plot_all_adfs(self, angular_df_dict):
        """
        This function plots the partial angular distribution function (adf) of each particle
        type pair within the bulk at current time step.

        Inputs:
        angular_df_dict: dictionary (output from angular_df() in
        measurement.py) containing information on the angular distribution
        function of combinations of particle types or all particles within the bulk.
        
        Outputs:
        .png file with general angular distribution function (adf) of each particle type 
        within the bulk at current time step.
        """

        fsize=10

        # Initialize adf data
        adf_allall = np.array(angular_df_dict['all-all'])
        adf_aa = np.array(angular_df_dict['A-A'])
        adf_ab = np.array(angular_df_dict['A-B'])
        adf_bb = np.array(angular_df_dict['B-B'])

        #Find order for plotting overlapping data
        AA_bulk_max = np.amax(adf_aa)
        AB_bulk_max = np.amax(adf_ab)
        BB_bulk_max = np.amax(adf_bb)

        if (AA_bulk_max >= AB_bulk_max):
            if (AA_bulk_max >= BB_bulk_max):
                AA_bulk_order = 1
                if (BB_bulk_max >= AB_bulk_max):
                    BB_bulk_order = 2
                    AB_bulk_order = 3
                else:
                    AB_bulk_order = 2
                    BB_bulk_order = 3
            else:
                BB_bulk_order = 1
                AA_bulk_order = 2
                AB_bulk_order = 3
        else:
            if (AA_bulk_max >= BB_bulk_max):
                AB_bulk_order = 1
                AA_bulk_order = 2
                BB_bulk_order = 3
            else:
                AA_bulk_order = 3
                if (BB_bulk_max >= AB_bulk_max):
                    BB_bulk_order = 1
                    AB_bulk_order = 2
                else:
                    AB_bulk_order = 1
                    BB_bulk_order = 2

        # Find y-limits
        if AA_bulk_order == 1:
            if 1.2 * np.max(angular_df_dict['A-A']) > 0:
                plot_max = 1.2 * np.max(angular_df_dict['A-A'])
            else:
                plot_max = 0.8 * np.max(angular_df_dict['A-A'])

            if np.min(angular_df_dict['A-A']) < 0:
                plot_min = 1.2 * np.min(angular_df_dict['A-A'])
            else:
                plot_min = 0.8 * np.min(angular_df_dict['A-A'])
        elif AB_bulk_order == 1:
            if 1.2 * np.max(angular_df_dict['A-B']) > 0:
                plot_max = 1.2 * np.max(angular_df_dict['A-B'])
            else:
                plot_max = 0.8 * np.max(angular_df_dict['A-B'])

            if np.min(angular_df_dict['A-B']) < 0:
                plot_min = 1.2 * np.min(angular_df_dict['A-B'])
            else:
                plot_min = 0.8 * np.min(angular_df_dict['A-B'])
        elif BB_bulk_order == 1:
            if 1.2 * np.max(angular_df_dict['B-B']) > 0:
                plot_max = 1.2 * np.max(angular_df_dict['B-B'])
            else:
                plot_max = 0.8 * np.max(angular_df_dict['B-B'])

            if np.min(angular_df_dict['B-B']) < 0:
                plot_min = 1.2 * np.min(angular_df_dict['B-B'])
            else:
                plot_min = 0.8 * np.min(angular_df_dict['B-B'])

        rstop=2*np.pi

        # Calculate x-ticks
        step = np.abs(plot_max - plot_min)/6
        step_x = 1.5

        #Define plot colors
        purple = ("#00441b")
        orange = ("#ff7f00")
        green = ("#377EB8")

        fig, ax1 = plt.subplots(figsize=(12,6))

        # Plot partial adf's in order of largest 1st-peak (widest line) to
        # shortest 1st-peak (thinnest line)
        first_width = 6.0
        second_width = 4.0
        third_width = 2.0
        if AA_bulk_order == 1:
            plt.plot(angular_df_dict['theta'], adf_aa, label=r'A-A',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AB_bulk_order == 2:
                plt.plot(angular_df_dict['theta'], adf_ab, label=r'A-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_bb, label=r'B-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(angular_df_dict['theta'], adf_bb, label=r'B-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_ab, label=r'A-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
        elif AB_bulk_order == 1:
            plt.plot(angular_df_dict['theta'], adf_ab, label=r'A-B',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AA_bulk_order == 2:
                plt.plot(angular_df_dict['theta'], adf_aa, label=r'A-A',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_bb, label=r'B-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(angular_df_dict['theta'], adf_bb, label=r'B-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_aa, label=r'A-A',
                            c=purple, lw=third_width, ls='-', alpha=1)
        elif BB_bulk_order == 1:
            plt.plot(angular_df_dict['theta'], adf_bb, label=r'B-B',
                        c=green, lw=first_width, ls='-', alpha=1)
            if AA_bulk_order == 1:
                plt.plot(angular_df_dict['theta'], adf_aa, label=r'A-A',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_ab, label=r'A-B',
                            c=purple, lw=third_width, ls='-', alpha=1)
            else:
                plt.plot(angular_df_dict['theta'], adf_ab, label=r'A-B',
                            c=orange, lw=second_width, ls='-', alpha=1)
                plt.plot(angular_df_dict['theta'], adf_aa, label=r'A-A',
                            c=purple, lw=third_width, ls='-', alpha=1.0)

        # Set limits
        ax1.set_ylim(plot_min, plot_max)
        ax1.set_xlim(0, rstop)

        # Label axes
        ax1.set_xlabel(r'Interparticle separation angle from $+\hat{\mathbf{x}}$ ($\theta$)', fontsize=fsize*2.8)
        ax1.set_ylabel(r'$g(\theta)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        loc = ticker.MultipleLocator(base=(np.pi/3))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(np.pi/6))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)
       
        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)

        # Set legend
        leg = [Line2D([0], [0], lw=8.0, c=green, label='A-A', linestyle='solid'), Line2D([0], [0], lw=8.0, c=orange, label='A-B', linestyle='solid'), Line2D([0], [0], lw=8.0, c=purple, label='B-B', linestyle='solid')]
        legend = ax1.legend(handles=leg, loc='upper right', columnspacing=1., handletextpad=0.3, bbox_transform=ax1.transAxes, bbox_to_anchor=[0.98, 1.03], fontsize=fsize*2.8, frameon=False, ncol=3)
        
        plt.tight_layout()
        plt.savefig(self.outPath + 'partial_adf_' + self.outFile + ".png", dpi=300)
        plt.close()
    def plot_csp(self, neigh_plot_dict, ang, pos, sep_surface_dict=None, int_comp_dict=None, pair='all-all', interface_id = False, orientation_id = False):
        """
        This function plots the positions of each particle and
        color-codes them by their centrosymmetry value

        Inputs:
        neigh_plot_dict: dictionary (output from nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        ang (partNum): array of angles (radians) of active force orientations of
        each particle

        pos (partNum, 3): array of (x,y,z) positions of each particle

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the number of nearest neighbors
        of reference particles of type all, A, or B should be plotted with the nearest
        neighbors to be counted of type all, A, or B (i.e. pair='all-A' is all
        neighbors of A reference particles are counted and averaged over the
        number of A reference particles).

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the centrosymmetry with color bar.
        """

        # Find particle type indices
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.25)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Define arrays to plot
        pos_x_arr = np.append(neigh_plot_dict['all']['x']+self.hx_box, neigh_plot_dict['all']['x']+self.hx_box)
        pos_x_arr = np.append(pos_x_arr, neigh_plot_dict['all']['x']+self.hx_box)

        pos_y_arr = np.append(neigh_plot_dict['all']['y']+self.hy_box, neigh_plot_dict['all']['y']+self.hy_box+self.ly_box)
        pos_y_arr = np.append(pos_y_arr, neigh_plot_dict['all']['y']+self.hy_box-self.ly_box)

        num_neigh_arr = np.append(neigh_plot_dict['all']['csp'], neigh_plot_dict['all']['csp'])
        num_neigh_arr = np.append(num_neigh_arr, neigh_plot_dict['all']['csp'])

        if pair == 'all':
            
            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = 0.4

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([pos_x_arr[i],pos_y_arr[i]]),
                    width=sz, height=sz)
            for i in range(0,len(pos_x_arr))]

            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells, cmap='Reds')
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(num_neigh_arr))

        elif pair == 'A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A']['x'][i]+self.hx_box,neigh_plot_dict['A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A']['csp']))

        elif pair == 'B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B']['csp'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B']['x'][i]+self.hx_box,neigh_plot_dict['B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B']['csp']))

        # Define color bar min and max
        minClb = min_neigh
        maxClb = max_neigh

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(min_neigh, max_neigh, 10)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        clb.ax.tick_params(labelsize=24)

        clb.set_label('Centrosymmetry Parameter', labelpad=25, y=0.5, rotation=270, fontsize=30)

        # Label respective reference and neighbor particle types
        if pair == 'all':
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'A':
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'B':
            plt.title('B Reference Particles', fontsize=20)

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass

        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Set plotting limits
        if self.lx_box > self.ly_box:
            plt.xlim(-(dense_x_width)+self.hx_box, (dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.outPath + 'csp_' + pair + '_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close() 
    
    def plot_neighbors(self, neigh_plot_dict, px, py, pos, pair='all-all', sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False):
        """
        This function plots the number of neighbors of all dense phase particles
        at each location in space.

        Inputs:
        neigh_plot_dict: dictionary (output from various nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        ang (partNum): array of angles (degrees) for each particle

        pos (partNum, 3): array of (x,y,z) positions for each particle

        pair (optional): string specifying whether the number of nearest neighbors
        of reference particles of type all, A, or B should be plotted with the nearest
        neighbors to be counted of type all, A, or B (i.e. pair='all-A' is all
        neighbors of A reference particles are counted and averaged over the
        number of A reference particles).

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the number of neighbors with color bar.
        """
        
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755
        pos_x_arr = np.append(neigh_plot_dict['all-all']['x']+self.hx_box, neigh_plot_dict['all-all']['x']+self.hx_box)
        pos_x_arr = np.append(pos_x_arr, neigh_plot_dict['all-all']['x']+self.hx_box)

        pos_y_arr = np.append(neigh_plot_dict['all-all']['y']+self.hy_box, neigh_plot_dict['all-all']['y']+self.hy_box+self.ly_box)
        pos_y_arr = np.append(pos_y_arr, neigh_plot_dict['all-all']['y']+self.hy_box-self.ly_box)

        num_neigh_arr = np.append(neigh_plot_dict['all-all']['neigh'], neigh_plot_dict['all-all']['neigh'])
        num_neigh_arr = np.append(num_neigh_arr, neigh_plot_dict['all-all']['neigh'])

        if pair == 'all-all':
            
            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-all']['neigh'])
            max_neigh = 7
            ells = [Ellipse(xy=np.array([pos_x_arr[i],pos_y_arr[i]]),
                    width=sz, height=sz)
            for i in range(0,len(pos_x_arr))]

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(num_neigh_arr))


            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)
        elif pair == 'all-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-A']['x'][i]+self.hx_box,neigh_plot_dict['all-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-A']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-A']['neigh']))

        elif pair == 'all-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-B']['x'][i]+self.hx_box,neigh_plot_dict['all-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-B']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-B']['neigh']))

        elif pair == 'A-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-all']['neigh'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-all']['x'][i]+self.hx_box,neigh_plot_dict['A-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-all']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-all']['neigh']))

        elif pair == 'B-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-all']['neigh'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-all']['x'][i]+self.hx_box,neigh_plot_dict['B-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-all']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-all']['neigh']))

        elif pair == 'A-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-A']['x'][i]+self.hx_box,neigh_plot_dict['A-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-A']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-A']['neigh']))

        elif pair == 'A-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-B']['x'][i]+self.hx_box,neigh_plot_dict['A-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-B']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-B']['neigh']))

        elif pair == 'B-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-A']['x'][i]+self.hx_box,neigh_plot_dict['B-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-A']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-A']['neigh']))

        elif pair == 'B-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-B']['x'][i]+self.hx_box,neigh_plot_dict['B-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-B']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-B']['neigh']))

        # Define color bar min and max
        minClb = min_neigh-0.5
        maxClb = int(max_neigh) + 1

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.arange(min_neigh, int(max_neigh)+1, 1)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.arange(min_neigh-0.5, int(max_neigh) + 1.5, 1)

        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types

        if pair == 'all-all':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('All Reference Particles', fontsize=30)
        elif pair == 'all-A':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('A Reference Particles', fontsize=30)
        elif pair == 'all-B':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('B Reference Particles', fontsize=30)
        elif pair == 'A-all':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('All Reference Particles', fontsize=30)
        elif pair == 'B-all':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('All Reference Particles', fontsize=30)
        elif pair == 'A-A':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('A Reference Particles', fontsize=30)
        elif pair == 'A-B':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('B Reference Particles', fontsize=30)
        elif pair == 'B-A':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('A Reference Particles', fontsize=30)
        elif pair == 'B-B':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=30)
            plt.title('B Reference Particles', fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'neigh_' + pair + '_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()        
    def plot_heterogeneity(self, hetero_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, type_m='phases'):
        """
        This function plots the number of neighbors of all dense phase particles
        at each location in space.

        Inputs:
        neigh_plot_dict: dictionary (output from various nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the number of nearest neighbors
        of reference particles of type all, A, or B should be plotted with the nearest
        neighbors to be counted of type all, A, or B (i.e. pair='all-A' is all
        neighbors of A reference particles are counted and averaged over the
        number of A reference particles).

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the lattice spacing with color bar.
        """
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Find min/max number of neighbors
        min_neigh = np.min(hetero_plot_dict['hetero'])
        max_neigh = np.max(hetero_plot_dict['hetero'])

        # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
        ells = [Ellipse(xy=np.array([hetero_plot_dict['x'][i]+self.hx_box,hetero_plot_dict['y'][i]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(hetero_plot_dict['x']))]

        # Plot position colored by number of A neighbors
        neighborGroup = mc.PatchCollection(ells, cmap='Reds')#facecolors=slowCol)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(hetero_plot_dict['hetero']))

        plt.title('All Reference Particles', fontsize=20)

        
        # Define color bar min and max
        minClb = np.min(hetero_plot_dict['hetero'])
        maxClb = np.max(hetero_plot_dict['hetero'])

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(minClb, maxClb, 6)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.linspace(minClb, maxClb, 1000)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types

        clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        
        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)


        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        if type_m=='phases':
            plt.savefig(self.outPath + 'hetero_' + self.outFile + ".png", dpi=250, transparent=False)
        elif type_m=='system':
            plt.savefig(self.outPath + 'hetero_system_' + self.outFile + ".png", dpi=250, transparent=False)

        plt.close() 

    def plot_neighbors_ori(self, neigh_plot_dict, ang, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, pair='all', mono_id=False, interface_id = False, orientation_id = False, zoom_id = False):
        """
        This function plots the number of neighbors of all dense phase particles
        at each location in space.

        Inputs:
        neigh_plot_dict: dictionary (output from various nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the number of nearest neighbors
        of reference particles of type all, A, or B should be plotted with the nearest
        neighbors to be counted of type all, A, or B (i.e. pair='all-A' is all
        neighbors of A reference particles are counted and averaged over the
        number of A reference particles).

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the lattice spacing with color bar.
        """
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755
        if pair == 'all-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-all']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-all']['x'][i]+self.hx_box,neigh_plot_dict['all-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-all']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-all']['ori']))

            plt.title('All Reference Particles', fontsize=20)

        elif pair == 'all-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-A']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-A']['x'][i]+self.hx_box,neigh_plot_dict['all-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-A']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-A']['ori']))
            
            plt.title('A Reference Particles', fontsize=20)

        elif pair == 'all-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-B']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-B']['x'][i]+self.hx_box,neigh_plot_dict['all-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-B']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-B']['ori']))
            
            plt.title('B Reference Particles', fontsize=20)
        
        elif pair == 'A-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-all']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-all']['x'][i]+self.hx_box,neigh_plot_dict['A-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-all']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-all']['ori']))

            
            plt.title('all Reference Particles', fontsize=20)

        elif pair == 'B-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-all']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-all']['x'][i]+self.hx_box,neigh_plot_dict['B-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-all']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-all']['ori']))
            
            plt.title('all Reference Particles', fontsize=20)

        elif pair == 'A-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-A']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-A']['x'][i]+self.hx_box,neigh_plot_dict['A-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-A']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-A']['ori']))
            
            plt.title('A Reference Particles', fontsize=20)

        elif pair == 'B-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-A']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-A']['x'][i]+self.hx_box,neigh_plot_dict['B-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-A']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-A']['ori']))
            
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'A-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-B']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-B']['x'][i]+self.hx_box,neigh_plot_dict['A-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-B']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-B']['ori']))
            
            plt.title('B Reference Particles', fontsize=20)

        elif pair == 'B-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-B']['ori'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-B']['x'][i]+self.hx_box,neigh_plot_dict['B-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-B']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap='seismic')#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-B']['ori']))
            
            plt.title('B Reference Particles', fontsize=20)
        #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
        #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-mean_dify, px, py, color='black', width=0.003)
        #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box-mean_dify, px, py, color='black', width=0.003)
        #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box-mean_dify, px, py, color='black', width=0.003)
    

        
        # Define color bar min and max
        minClb = -1.0
        maxClb = 1.0

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(minClb, maxClb, 6)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.linspace(minClb, maxClb, 1000)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types

        clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        
        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)


        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + pair + '_local_ori_' + self.outFile + ".png", dpi=250, transparent=False)
        plt.close()    

    def plot_ang(self, ang, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', mono_id=False, interface_id = False, orientation_id = False, zoom_id = False):
        """
        This function plots the orientation (radians) of all particles
        at each location in space.

        Inputs:
        ang (partNum): array of angles (degrees) for each particle

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the active force orientation with color bar.
        """
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Find min/max number of neighbors
        min_neigh = np.min(ang)
        max_neigh = np.max(ang)

        # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 0)[0]
        
        if type == 'all':
            ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(pos))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('hsv', 1000))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(ang))

            plt.title('All Reference Particles', fontsize=20)
        
        elif type == 'A':
            typ0ind = np.where(self.typ == 0)[0]

            ells = [Ellipse(xy=np.array([pos[typ0ind[i],0]+self.hx_box,pos[typ0ind[i],1]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(typ0ind))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('hsv', 1000))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(ang[typ0ind]))

            plt.title('A Reference Particles', fontsize=20)
        
        elif type == 'B':
            typ1ind = np.where(self.typ == 1)[0]

            ells = [Ellipse(xy=np.array([pos[typ1ind[i],0]+self.hx_box,pos[typ1ind[i],1]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(typ1ind))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('hsv', 1000))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(ang[typ1ind]))

            plt.title('B Reference Particles', fontsize=20)

        

        # Define color bar min and max
        minClb = np.min(ang)
        maxClb = np.max(ang)
        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(minClb, maxClb, 10)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.linspace(minClb, maxClb, 1000)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types

        clb.set_label(r'$\theta$', labelpad=25, y=0.5, rotation=270, fontsize=30)
        

        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label simulation time
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        



        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:

            if zoom_id == True:
                plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                plt.xlim(self.hy_box-25-2, self.hy_box+25+2)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.outPath + 'plot_ang_' + type + '_' + self.outFile + ".png", dpi=250, transparent=False)
        plt.close()

    def plot_local_density(self, local_dens_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, pair='all-all', interface_id = False, orientation_id = False):
        """
        This function plots the binned average area fraction at each location
        in space.

        Inputs:
        local_dens_plot_dict: dictionary (output from local_density() in
        measurement.py) containing information on the local density of all, type A, 
        or type B particles around all, type A, or type B particles in the dense phase.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the local density of all
        (pair='all-X'), type A (pair='A-X'), type B (pair='B-X') around all 
        (pair='X-all), type A (pair='X-A), or type B (pair='X-B) particles

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with position of each dense phase particle plotted and color coded by the 
        local area fraction around it with color bar
        """
        
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Find min/max number of neighbors
        min_neigh = np.min(local_dens_plot_dict[pair]['dens'])
        max_neigh = np.max(local_dens_plot_dict[pair]['dens'])

        # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
        ells = [Ellipse(xy=np.array([local_dens_plot_dict[pair]['pos_x'][i]+self.hx_box,local_dens_plot_dict[pair]['pos_y'][i]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(local_dens_plot_dict[pair]['pos_x']))]

        # Plot position colored by number of A neighbors
        neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('cool', 1000))#facecolors=slowCol)
        coll = ax.add_collection(neighborGroup)

        # Plot title
        coll.set_array(np.ravel(local_dens_plot_dict[pair]['dens']))
        neigh_pair = pair.split('-')
        plt.title(neigh_pair[1].capitalize() + ' Reference Particles', fontsize=30)

        # Define color bar min and max
        minClb = np.min(local_dens_plot_dict[pair]['dens'])
        maxClb = np.max(local_dens_plot_dict[pair]['dens'])

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(minClb, maxClb, 10)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.linspace(minClb, maxClb, 1000)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types
        if neigh_pair[0] == 'A':
            clb.set_label(r'$\phi^\mathrm{S}_\mathrm{local}$', labelpad=35, y=0.5, rotation=270, fontsize=30)
        elif neigh_pair[0] == 'B':
            clb.set_label(r'$\phi^\mathrm{F}_\mathrm{local}$', labelpad=35, y=0.5, rotation=270, fontsize=30)
        else:
            clb.set_label(r'$\phi_\mathrm{local}$', labelpad=35, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.outPath + 'local_density_' + str(pair) + '_' + self.outFile + ".png", dpi=250, transparent=False)
        plt.close()    

    def plot_homogeneity(self, local_dens_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, pair='all-all', interface_id = False, orientation_id = False):
        """
        This function plots the binned average area fraction at each location
        in space.

        Inputs:
        local_dens_plot_dict: dictionary (output from local_density() in
        measurement.py) containing information on the local density of all, type A, 
        or type B particles around all, type A, or type B particles in the dense phase.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the local density of all
        (pair='all-X'), type A (pair='A-X'), type B (pair='B-X') around all 
        (pair='X-all), type A (pair='X-A), or type B (pair='X-B) particles

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with position of each dense phase particle plotted and color coded by the 
        local area fraction around it with color bar
        """
        
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Find min/max number of neighbors
        min_neigh = -1.2#-np.min(local_dens_plot_dict[pair]['homo'])
        max_neigh = 1.2#np.max(local_dens_plot_dict[pair]['homo'])

        # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
        ells = [Ellipse(xy=np.array([local_dens_plot_dict[pair]['pos_x'][i]+self.hx_box,local_dens_plot_dict[pair]['pos_y'][i]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(local_dens_plot_dict[pair]['pos_x']))]

        # Plot position colored by number of A neighbors
        neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('cool', 1000))#facecolors=slowCol)
        coll = ax.add_collection(neighborGroup)

        # Plot title
        coll.set_array(np.ravel(local_dens_plot_dict[pair]['homo']))
        neigh_pair = pair.split('-')
        plt.title(neigh_pair[1].capitalize() + ' Reference Particles', fontsize=30)

        # Define color bar min and max
        minClb = -1.2#np.min(local_dens_plot_dict[pair]['homo'])
        maxClb = 1.2#np.max(local_dens_plot_dict[pair]['homo'])

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.linspace(minClb, maxClb, 10)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.linspace(minClb, maxClb, 1000)
        
        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.4f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)

        # Label respective reference and neighbor particle types
        if neigh_pair[0] == 'A':
            clb.set_label(r'$(\phi_\mathrm{S}(x)-\langle \phi_\mathrm{S} \rangle )^2$', labelpad=35, y=0.5, rotation=270, fontsize=30)
        elif neigh_pair[0] == 'B':
            clb.set_label(r'$(\phi_\mathrm{F}(x)-\langle \phi_\mathrm{F} \rangle )^2$', labelpad=35, y=0.5, rotation=270, fontsize=30)
        else:
            clb.set_label(r'$(\phi(x)-\langle \phi \rangle )^2$', labelpad=35, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.outPath + 'local_homogeneity_' + str(pair) + '_' + self.outFile + ".png", dpi=250, transparent=False)
        plt.close() 

    def plot_clustering(self, neigh_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):
        """
        This function plots the clustering coefficient of all dense phase particles
        at each location in space.

        Inputs:
        neigh_plot_dict: dictionary (output from various nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the clustering coefficient with color bar.
        """
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            #y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.755

        # Define colorbar limits
        min_neigh = 0
        max_neigh = np.amax(neigh_plot_dict[type]['clust'])

        # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
        ells = [Ellipse(xy=np.array([neigh_plot_dict[type]['pos']['x'][i]+self.hx_box,neigh_plot_dict[type]['pos']['y'][i]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(neigh_plot_dict[type]['pos']['x']))]

        # Plot position of particles colored by clustering coefficient
        neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', 10))#facecolors=slowCol)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(neigh_plot_dict[type]['clust']))

        # Plot mean clustering coefficient
        plt.text(0.62, 0.92, s=r'$\mu$' + ' = ' + '{:.4f}'.format(np.mean(neigh_plot_dict[type]['clust'])),
            fontsize=30, transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Plot graph title
        plt.title(type.capitalize() + ' Reference Particles', fontsize=30)

        # Set color bar properties
        minClb = 0.0
        maxClb = 1.0
        coll.set_clim([minClb, maxClb])
        tick_lev = np.linspace(minClb, maxClb, 10)
        level_boundaries = np.linspace(minClb, maxClb, 10)
        
        # Plot colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=24)
        clb.set_label('Clustering Coefficient', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'clustering_'  + type + '_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()  
        
    def plot_neighbors2(self, neigh_plot_dict, px, py, pos, prev_pos, sep_surface_dict=None, int_comp_dict=None, pair='all-all'):
        """
        This function plots the number of neighbors of all dense phase particles
        at each location in space.

        Inputs:
        neigh_plot_dict: dictionary (output from various nearest_neighbors() in
        measurement.py) containing information on the nearest neighbors of each
        respective type ('all', 'A', or 'B') within the potential
        cut-off radius for reference particles of each respective
        type ('all', 'A', or 'B') for the dense phase, labeled as specific activity
        pairings, i.e. 'all-A' means all neighbors of A reference particles.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        pair (optional): string specifying whether the number of nearest neighbors
        of reference particles of type all, A, or B should be plotted with the nearest
        neighbors to be counted of type all, A, or B (i.e. pair='all-A' is all
        neighbors of A reference particles are counted and averaged over the
        number of A reference particles).

        Outputs:
        .png file with the position of each particle plotted and color coded
        by the lattice spacing with color bar.
        """
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]
        
        # Calculate interparticle distance between type A reference particle and type A neighbor
        difx, dify, difr = self.utility_functs.sep_dist_arr(pos[typ0ind], prev_pos[typ0ind], difxy=True)
        mean_dify = np.mean(dify)
        mean_dify = 0.
        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(neigh_plot_dict['all-all']['x'][typ0ind]) * 1.5 #(area_dense / self.ly_box)
            #dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim# * 3
        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(neigh_plot_dict['all-all']['y']+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.75

        pos_x_arr = np.append(neigh_plot_dict['all-all']['x']+self.hx_box, neigh_plot_dict['all-all']['x']+self.hx_box)
        pos_x_arr = np.append(pos_x_arr, neigh_plot_dict['all-all']['x']+self.hx_box)

        pos_y_arr = np.append(neigh_plot_dict['all-all']['y']+self.hy_box-mean_dify, neigh_plot_dict['all-all']['y']+self.hy_box+self.ly_box-mean_dify)
        pos_y_arr = np.append(pos_y_arr, neigh_plot_dict['all-all']['y']+self.hy_box-self.ly_box-mean_dify)

        num_neigh_arr = np.append(neigh_plot_dict['all-all']['neigh'], neigh_plot_dict['all-all']['neigh'])
        num_neigh_arr = np.append(num_neigh_arr, neigh_plot_dict['all-all']['neigh'])

        if pair == 'all-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = 8#np.amax(neigh_plot_dict['all-all']['neigh'])

            ells = [Ellipse(xy=np.array([pos_x_arr[i],pos_y_arr[i]]),
                    width=sz, height=sz)
            for i in range(0,len(pos_x_arr))]

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size


            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-all']['neigh']))

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-mean_dify, px, py, color='black', width=0.003)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box-mean_dify, px, py, color='black', width=0.003)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box-mean_dify, px, py, color='black', width=0.003)
        
        elif pair == 'all-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-A']['x'][i]+self.hx_box,neigh_plot_dict['all-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-A']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-A']['neigh']))

        elif pair == 'all-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-B']['x'][i]+self.hx_box,neigh_plot_dict['all-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-B']['x']))]

            # Plot position colored by number of all neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-B']['neigh']))

        elif pair == 'A-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-all']['neigh'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-all']['x'][i]+self.hx_box,neigh_plot_dict['A-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-all']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-all']['neigh']))

        elif pair == 'B-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-all']['neigh'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-all']['x'][i]+self.hx_box,neigh_plot_dict['B-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-all']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-all']['neigh']))

        elif pair == 'A-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-A']['x'][i]+self.hx_box,neigh_plot_dict['A-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-A']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-A']['neigh']))

        elif pair == 'A-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['A-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-B']['x'][i]+self.hx_box,neigh_plot_dict['A-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-B']['x']))]

            # Plot position colored by number of A neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-B']['neigh']))

        elif pair == 'B-A':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-A']['neigh'])

            # Generate list of ellipses for A particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-A']['x'][i]+self.hx_box,neigh_plot_dict['B-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-A']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-A']['neigh']))

        elif pair == 'B-B':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['B-B']['neigh'])

            # Generate list of ellipses for B particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-B']['x'][i]+self.hx_box,neigh_plot_dict['B-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-B']['x']))]

            # Plot position colored by number of B neighbors
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-B']['neigh']))

        # Define color bar min and max
        minClb = min_neigh-0.5
        maxClb = 9#int(max_neigh) + 1

        # Set color bar range
        coll.set_clim([minClb, maxClb])

        # Set tick levels
        tick_lev = np.arange(min_neigh, int(max_neigh)+1, 1)

        # Define boundaries of colors (such that ticks at midpoints)
        level_boundaries = np.arange(min_neigh-0.5, int(max_neigh) + 1.5, 1)

        # Define colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'), boundaries=level_boundaries)
        clb.ax.tick_params(labelsize=16)

        # Label respective reference and neighbor particle types

        if pair == 'all-all':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'all-A':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'all-B':
            clb.set_label('# Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)
        elif pair == 'A-all':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'B-all':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'A-A':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'A-B':
            clb.set_label('# A Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)
        elif pair == 'B-A':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'B-B':
            clb.set_label('# B Neighbors', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.8, 0.04, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        """
        # Plot interpolated inner and outer interface surface curves
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            try:
                pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
            except:
                pass

            try:
                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
            except:
                pass
        """



        if self.lx_box > self.ly_box:
            plt.xlim(-(dense_x_width)+self.hx_box, (dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.outPath + 'all_all_neigh_' + self.outFile + ".png", dpi=75, transparent=False)
        plt.close()  

    def plot_hexatic_order(self, pos, hexatic_order_param, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False):

        """
        This function plots the hexatic order parameter of each particle in space

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        hexatic_order_param (partNum): array (output from various hexatic_order() in
        measurement.py, hexatic_order_dict['order']) containing information on the 
        hexatic order parameter of all, type A, and type B particles

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with hexatic order parameter of each particle plotted with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)
        levels_text=40
        tick_locs   = [0.0,np.pi/12, np.pi/6,3*np.pi/12, np.pi/3]
        tick_labels = ['0',r'$\pi/12$', r'$\pi/6$', r'$3\pi/12$', r'$\pi/3$']

        # Find min/max orientation
        min_hex = 0.0
        max_hex = 1.0

        # Set plotted particle size
        sz = 0.755
        
        # Plot position colored by neighbor number
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(hexatic_order_param))
        
        # Set colorbar properties
        minClb = min_hex
        maxClb = max_hex
        coll.set_clim([minClb, maxClb])

        tick_lev = np.arange(min_hex, max_hex+1, (max_hex - min_hex)/5)

        # Plot colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        clb.set_label(r'$g_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=30)
        
        # Set plot title
        plt.title('All Reference Particles', fontsize=30)
        
        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)


        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'hexatic_order_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()

    def plot_domain_angle(self, pos, relative_angles, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False):

        """
        This function plots the domain angle of each particle in space

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        relative_angles(partNum): array (output from various hexatic_order() in
        measurement.py, hexatic_order_dict['theta']) containing information on the 
        domain angle of all, type A, and type B particles

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with domain angle of each particle plotted with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Find min/max orientation
        min_theta = 0.0
        max_theta = np.pi/3

        # Set plotted particle size
        sz = 0.755
        
        # Plot position colored by neighbor number
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        neighborGroup = mc.PatchCollection(ells, cmap='hsv')
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(relative_angles))

        # Set colorbar properties
        minClb = min_theta
        maxClb = max_theta
        coll.set_clim([minClb, maxClb])
        tick_lev = np.arange(min_theta, max_theta+1, (max_theta - min_theta)/6)

        # Plot colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        clb.set_label(r'$\theta_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=30)

        # Set plot title
        plt.title('All Reference Particles', fontsize=30)

        # Set colorbar ticks
        tick_locs   = [0.0,np.pi/12, np.pi/6,3*np.pi/12, np.pi/3]
        tick_labels = ['0',r'$\pi/12$', r'$\pi/6$', r'$3\pi/12$', r'$\pi/3$']
        clb.locator     = matplotlib.ticker.FixedLocator(tick_locs)
        clb.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
        clb.update_ticks()

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)



        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'relative_angle_' + self.outFile + ".png", dpi=300, transparent=False)
        plt.close()

    def plot_trans_order(self, pos, trans_param, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False):

        """
        This function plots the translational order of each particle in space

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        trans_param(partNum): array (output from translational_order() in
        measurement.py) containing information on the 
        translational order of all, type A, and type B particles

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with translational order of each particle plotted with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Find min/max orientation
        min_tran = np.min(trans_param)
        max_tran = 1.0

        # Set plotted particle size
        sz = 0.755
        
        # Plot position colored by neighbor number
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(trans_param))
        
        # Set colorbar properties
        minClb = min_tran
        maxClb = max_tran
        coll.set_clim([minClb, maxClb])
        tick_lev = np.arange(min_tran, max_tran+1, (max_tran - min_tran)/6)

        # Plot colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        clb.set_label(r'$\Psi_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=30)

        # Set plot title
        plt.title('All Reference Particles', fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'translational_order_' + self.outFile + ".png", dpi=300, transparent=False)

    def plot_stein_order(self, pos, stein_param, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, interface_id = False, orientation_id = False):

        """
        This function plots the steinhardt order of each particle in space

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        stein_param(partNum): array (output from steinhardt_order() in
        measurement.py) containing information on the 
        steinhardt order of all, type A, and type B particles

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with steinhardt order of each particle plotted with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.0)
            y_dim = int(scaling)
        
        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Define colorbar limits
        min_stein = np.min(stein_param)
        max_stein = 1.0

        levels_text=40
        level_boundaries = np.linspace(min_stein, max_stein, levels_text + 1)

        # Set plotted particle size
        sz = 0.755
        
        # Plot position colored by steinhardt order parameter
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(stein_param))
        
        # Modify colorbar properties
        minClb = min_stein
        maxClb = max_stein
        coll.set_clim([minClb, maxClb])

        tick_lev = np.arange(min_stein, max_stein+1, (max_stein - min_stein)/6)

        # Plot colorbar
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)

        clb.set_label(r'$q_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=30)

        # Plot title
        plt.title('All Reference Particles', fontsize=30)


        
        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'steinhardt_order_' + self.outFile + ".png", dpi=300, transparent=False)
        plt.close()

    def plot_nematic_order(self, pos, nematic_param, sep_surface_dict, int_comp_dict):

        #Plot particles colorized by translational order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.mean(nematic_param)
        max_n = np.max(nematic_param)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']



        im = plt.scatter(pos[:,0]+self.hx_box, pos[:,1]+self.hy_box, c=nematic_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))

        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Psi$', labelpad=-55, y=1.04, rotation=0, fontsize=18)

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))
            try:
                pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
            except:
                pass

            try:
                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
            except:
                pass

        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'translational_order_' + out + pad + ".png", dpi=100)
        #plt.close()



    def plot_interpart_press_binned2(self, interpart_press_binned, sep_surface_dict, int_comp_dict):

        vmax_p = np.max(interpart_press_binned)
        vmin_p = np.min(interpart_press_binned)


        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        #ax2 = fig.add_subplot()


        im = plt.contourf(self.pos_x, self.pos_y, interpart_press_binned, vmin = vmin_p, vmax=vmax_p)

        norm= matplotlib.colors.Normalize(vmin=vmin_p, vmax=vmax_p)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(vmin_p, vmax_p+vmax_p/10, (vmax_p-vmin_p)/10)
        clb = fig.colorbar(sm, ticks=tick_lev)
        clb.ax.tick_params(labelsize=16)

        clb.ax.set_title(r'$\Pi^\mathrm{P}$', fontsize=20)
        #fig.colorbar(im, ax=ax2)

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))
            try:
                pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
            except:
                pass

            try:
                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
            except:
                pass

        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
            fontsize=18, transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #ax2.axis('off')
        plt.axis('off')
        #plt.title(r'$\mathrm{Pe}$' + ' = ' + str(int(peA)) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))), fontsize=17)
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'interpart_press_' + out + pad + ".png", dpi=200)
        #plt.close()

    def plot_interpart_press_binned(self, interpart_press_binned, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):
        """
        This function plots the binned interparticle pressure

        Inputs:
        interpart_press_binned (NBins_x, NBins_y): array (output from various virial_pressure_binned() in
        stress_and_pressure.py) containing information on the stress and positions of all,
        type A, and type B particles.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with binned interparticle pressure plotted with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.75)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Define color bar limits
        div_min = -3
        min_n = np.min(interpart_press_binned)
        max_n = np.max(interpart_press_binned)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)

        # Plot binned interparticle pressure
        im = plt.contourf(self.pos_x, self.pos_y, interpart_press_binned, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        # Plot binned particle orientation
        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)

        # Define color bar properties
        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)

        # Plot colorbar
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=24)
        clb.set_label(r'$\Pi^\mathrm{P}$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'interparticle_pressure_binned_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

    def interpart_press_map(self, stress_plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, type='all', interface_id = False, orientation_id = False):

        """
        This function plots the positions of all dense phase particles color-coded by 
        nearest neighbor pressure for each particle.

        Inputs:
        stress_plot_dict: dictionary (output from various interparticle_pressure_nlist() in
        measurement.py) containing information on the stress and positions of all,
        type A, and type B particles.

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        Outputs:
        .png file with particles plotted and color-coded by nearest-neighbor pressure with color bar
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 2.25)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)
        
        # Set positions and stress based on type specified
        if type == 'all':
            stress_xx = np.append(stress_plot_dict['dense']['all-all']['XX'], stress_plot_dict['gas']['all-all']['XX'])
            stress_yy = np.append(stress_plot_dict['dense']['all-all']['YY'], stress_plot_dict['gas']['all-all']['YY'])

            pos_x = np.append(stress_plot_dict['dense']['pos']['all']['x'], stress_plot_dict['gas']['pos']['all']['x'])
            pos_y = np.append(stress_plot_dict['dense']['pos']['all']['y'], stress_plot_dict['gas']['pos']['all']['y'])
        elif type == 'A':
            stress_xx = np.append(stress_plot_dict['dense']['all-A']['XX'], stress_plot_dict['gas']['all-A']['XX'])
            stress_yy = np.append(stress_plot_dict['dense']['all-A']['YY'], stress_plot_dict['gas']['all-A']['YY'])

            pos_x = np.append(stress_plot_dict['dense']['pos']['A']['x'], stress_plot_dict['gas']['pos']['A']['x'])
            pos_y = np.append(stress_plot_dict['dense']['pos']['A']['y'], stress_plot_dict['gas']['pos']['A']['y'])
        elif type == 'B':
            stress_xx = np.append(stress_plot_dict['dense']['all-B']['XX'], stress_plot_dict['gas']['all-B']['XX'])
            stress_yy = np.append(stress_plot_dict['dense']['all-B']['YY'], stress_plot_dict['gas']['all-B']['YY'])

            pos_x = np.append(stress_plot_dict['dense']['pos']['B']['x'], stress_plot_dict['gas']['pos']['B']['x'])
            pos_y = np.append(stress_plot_dict['dense']['pos']['B']['y'], stress_plot_dict['gas']['pos']['B']['y'])

        # Define nearest-neighbor pressure
        press = (stress_xx + stress_yy ) / (2 * np.pi * self.r_cut**2)
        
        # Define colorbar limits
        bulk_lat_mean = np.mean(press)

        min_n = np.min(press)
        max_n = np.max(press)

        # Particle size
        sz = 0.755

        # Plot color-coded particles
        ells = [Ellipse(xy=np.array([pos_x[i]+self.hx_box,pos_y[i]+self.hy_box]),
            width=sz, height=sz) for i in range(0,len(pos_x))]
        bulkGroup = mc.PatchCollection(ells, cmap='Reds')
        bulkGroup.set_array(press)
        ax.add_collection(bulkGroup)

        # Set colorbar parameters
        if bulk_lat_mean != 0.0:
            tick_lev = np.arange(min_n, max_n+(max_n-min_n)/6, (max_n - min_n)/6)
            clb = plt.colorbar(bulkGroup, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.1f'))
        else:
            clb = plt.colorbar(bulkGroup, orientation="vertical", format=tick.FormatStrFormatter('%.1f'))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        clb.ax.tick_params(labelsize=24)

        # Label color bar based on particle type specified 
        if self.lx_box == self.ly_box:
            clb.set_label(r'$\hat{\Pi}_\mathrm{d}^\mathrm{P}$', labelpad=30, y=0.5, rotation=270, fontsize=30)
        elif self.lx_box > self.ly_box:
            clb.set_label(r'$\hat{\Pi}_\mathrm{d}^\mathrm{P}$', labelpad=30, y=0.5, rotation=270, fontsize=30)

        # Set plot title
        if type == 'all':
            plt.title('All Reference Particles', fontsize=30)
        elif type == 'A':
            plt.title('A Reference Particles', fontsize=30)
        elif type == 'B':
            plt.title('B Reference Particles', fontsize=30)

        # Plot interpolated inner and outer interface surface curves
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass
        
        # Plot averaged, binned orientation of particles
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        # Label time step
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

       # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outPath + 'int_press_' + type + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()
    
    def interpart_press_map2(self, interpart_press_part, pos, prev_pos, ang):
        
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]
        
        # Position and orientation arrays of type A particles in respective phase
        pos_A=pos[typ0ind]                               # Find positions of type 0 particles

        # Position and orientation arrays of type B particles in respective phase
        pos_B=pos[typ1ind]

        # Calculate interparticle distance between type A reference particle and type A neighbor
        difx, dify, difr = self.utility_functs.sep_dist_arr(pos[typ0ind], prev_pos[typ0ind], difxy=True)
        
        mean_dify = np.mean(dify)
        mean_dify = 0
        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        pos_x_arr = np.append(interpart_press_part['all-all']['x']+self.hx_box, interpart_press_part['all-all']['x']+self.hx_box)
        pos_x_arr = np.append(pos_x_arr, interpart_press_part['all-all']['x']+self.hx_box)

        pos_y_arr = np.append(interpart_press_part['all-all']['y']+self.hy_box-mean_dify, interpart_press_part['all-all']['y']+self.hy_box+self.ly_box-mean_dify)
        pos_y_arr = np.append(pos_y_arr, interpart_press_part['all-all']['y']+self.hy_box-self.ly_box-mean_dify)

        num_neigh_arr = np.append(interpart_press_part['all-all']['press'], interpart_press_part['all-all']['press'])
        num_neigh_arr = np.append(num_neigh_arr, interpart_press_part['all-all']['press'])
        
        # Instantiated simulation box
        f_box = box.Box(Lx=self.lx_box, Ly=self.ly_box, is2D=True)

        query_args = dict(mode='ball', r_min = 0.1, r_max=5.0)
        system_A = freud.AABBQuery(f_box, f_box.wrap(pos_A))
        AB_nlist = system_A.query(f_box.wrap(pos_B), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type B bulk particles
        AB_neigh_ind = np.array([], dtype=int)
        AB_num_neigh = np.array([])


        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        #fy_arr = np.zeros()
        for i in range(0, len(pos_A)):
            if i in AB_nlist.point_indices:
                if i not in AB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    # Save nearest neighbor information for i reference particle
                    AB_neigh_ind = np.append(AB_neigh_ind, int(i))
        
        filter_id = np.where(interpart_press_part['all-all']['press'][typ0ind[AB_neigh_ind]]>=100)[0]
        pos_A_ref = pos_A[AB_neigh_ind[filter_id]]
        if len(filter_id)>0:

            

            query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)
            system = freud.AABBQuery(f_box, f_box.wrap(pos))
            allall_nlist = system_A.query(f_box.wrap(pos_A_ref), query_args).toNeighborList()

            allall_neigh_ind = np.array([], dtype=int)
            #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
            fx_arr = np.zeros(len(pos_A_ref))
            fy_arr = np.zeros(len(pos_A_ref))

            for i in range(0, len(pos_A_ref)):
                if i in allall_nlist.query_point_indices:
                    if i not in allall_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(allall_nlist.query_point_indices==i)[0]
                        if len(loc)>1:
                            
                            # Array of reference particle location
                            pos_temp = np.ones((len(loc), 3))* pos_A_ref[i]

                            # Calculate interparticle separation distances
                            difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos[allall_nlist.point_indices[loc]], difxy=True)

                            # Calculate interparticle forces
                            fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                            fx_arr[i]+=np.sum(fx)
                            fy_arr[i]+=np.sum(fy)
                        # Save nearest neighbor information for i reference particle
                        allall_neigh_ind = np.append(allall_neigh_ind, int(i))

            
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)


        # Set plotted particle size
        sz = 0.75

        

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size


            # Plot position colored by neighbor number
            


        bulk_lat_mean = np.mean(interpart_press_part['all-all']['press'])

        min_n = 0.0#np.min(interpart_press_part['all-A']['press'])
        if self.peB > self.peA:
            max_n = self.peB*(2/3)#np.max(interpart_press_part['all-A']['press'])
        else:
            max_n = self.peA*(2/3)
        ells = [Ellipse(xy=np.array([pos_x_arr[i],pos_y_arr[i]]),
                    width=sz, height=sz)
            for i in range(0,len(pos_x_arr))]

        neighborGroup = mc.PatchCollection(ells, cmap='Reds')#facecolors=slowCol)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(num_neigh_arr))

        #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
        plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-mean_dify, px, py, color='black', width=0.002)
        plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box-mean_dify, px, py, color='black', width=0.002)
        plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box-mean_dify, px, py, color='black', width=0.002)

        #if len(pos_A_ref)>0:
        #    plt.quiver(pos_A_ref[:,0]+self.hx_box, pos_A_ref[:,1]+self.hy_box-mean_dify, fx_arr, fy_arr, color='green', width=0.003, linestyle='dashed')
        #    plt.quiver(pos_A_ref[:,0]+self.hx_box, pos_A_ref[:,1]+self.hy_box+self.ly_box-mean_dify, fx_arr, fy_arr, color='green', width=0.003, linestyle='dashed')
        #    plt.quiver(pos_A_ref[:,0]+self.hx_box, pos_A_ref[:,1]+self.hy_box-self.ly_box-mean_dify, fx_arr, fy_arr, color='green', width=0.003, linestyle='dashed')

        # Define color bar min and max


        # Set color bar range
        coll.set_clim([min_n, max_n])

        # Set tick levels
        #tick_lev = np.arange(min_neigh, int(max_neigh)+1, 1) ticks=tick_lev, 

        # Define colorbar
        clb = plt.colorbar(coll, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))
        clb.ax.tick_params(labelsize=16)

        # Label respective reference and neighbor particle types

        clb.set_label('Interparticle Stress', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)


        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.8, 0.04, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        if self.lx_box > self.ly_box:
            plt.xlim(-(dense_x_width)+self.hx_box, (dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        

        # Modify plot parameters
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.outPath + 'force_lines_' + self.outFile + ".png", dpi=75, transparent=False)
        plt.close()  

        #plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=100)
        #plt.close()
    

    def plot_clust_fluctuations(self, pos, outfile_name, ang, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None):

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        slowCol = '#081d58'

        

        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            mono=1
            mono_activity=self.peA
            mono_type = 2
            mono=0
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)

        #fig = plt.figure(figsize=(x_dim,y_dim*2))
        #ax = fig.add_subplot(121, gridspec_kw={'height_ratios': [2, 1]})
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios': [2, 1]})

        sz = 0.755


        if mono==0:
            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot
            """
            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(typ1ind))]
            """

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax0.add_collection(slowGroup)
            ax0.add_collection(fastGroup)
            """
            #Create legend for binary system
            if (len(typ0ind)!=0) & (len(typ1ind)!=0):
                leg = ax0.legend(handles=[ells0[0], ells1[0]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{B} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                if self.peA <= self.peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax0.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)
            """


            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #ax0.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #ax0.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #ax0.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            if (self.peA==0):
                slowCol = '#081d58'
            else:
                slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax0.add_collection(slowGroup)

                #leg = ax0.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax0.add_collection(slowGroup)

                #leg = ax0.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax0.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax0.add_collection(fastGroup)

                #leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                #leg = ax.legend(handles=[ells0[0]], labels=[r'$F^\mathrm{a} = $'+str(int(self.peB))], loc='upper right', prop={'size': 28}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)
        """
        try:
            if sep_surface_dict!=None:
                for m in range(0, len(sep_surface_dict)):
                    key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

                    try:
                        pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                        pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                        plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                    except:
                        pass

                    try:
                        pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                        pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                        plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                    except:
                        pass
        except:
            pass
        """
        try:
            if active_fa_dict!=None:
                ax0.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
        except:
            pass
        #Label time step
        #ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
        #        horizontalalignment='right', verticalalignment='bottom',
        #        transform=ax.transAxes,
        #        fontsize=18,
        #        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax0.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax0.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax0.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax0.set_xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            ax0.set_ylim(0, self.ly_box)
            ax0.set_xlim(0, self.lx_box)

        # Label simulation time
        if self.lx_box == self.ly_box:
            ax0.text(0.55, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=24, transform = ax0.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            ax0.text(0.65, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=24, transform = ax0.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax0.axes.set_xticks([])
        ax0.axes.set_yticks([])
        ax0.axes.set_xticklabels([])
        ax0.axes.set_yticks([])
        ax0.set_aspect('equal')

        # Create frame pad for images
        #pad = str(j).zfill(4)

        import pandas as pd

        

        txtFile1 = 'surface_interface_pressure_' + outfile_name + '.txt'
        df1 = pd.read_csv('/Volumes/EXTERNAL2/temp_files_new/surface_interface_pressure/' + txtFile1, sep='\s+', header=0)

        time_arr = df1['tauB']
        clust_size_arr = df1['clust_size']
        press_all_arr = df1['all']
        press_A_arr = df1['A']
        press_B_arr = df1['B']
        # Strips the newline character

        txtFile2 = 'interparticle_press_' + outfile_name + '.txt'
        df2 = pd.read_csv('/Volumes/EXTERNAL2/temp_files_new/interparticle_press/' + txtFile2, sep='\s+', header=0)

        time_arr2 = df2['tauB']
        clust_size_arr2 = df2['clust_size']
        press_all_arr2 = df2['all-all_bulk_press']
        press_A_arr2 = df2['all-A_bulk_press']
        press_B_arr2 = df2['all-B_bulk_press']

        # Strips the newline character

        rstop = 10.
        if np.max(press_all_arr) >= np.max(press_all_arr2):
            max_n = np.max(press_all_arr)
        else:
            max_n = np.max(press_all_arr2)

        x_arr = np.array([self.tst,self.tst])
        y_arr = np.array([0.0, max_n])

        if np.mean(press_all_arr) >= np.mean(press_all_arr2):
            max_n = np.mean(press_all_arr)
            min_n = np.mean(press_all_arr2)
        else:
            max_n = np.mean(press_all_arr2)
            min_n = np.mean(press_all_arr)

        #plt.plot(x_arr, y_arr, c='black', lw=1.0, ls='--')

        #plt.plot(time_arr, press_all_arr,
        #            c='purple', lw=3.0, ls='-', alpha=1, label='Interface Pressure')
        #plt.plot(time_arr2, press_all_arr2,
        #            c='green', lw=3.0, ls='-', alpha=1, label='Bulk Pressure')
        
        
        """
        plt.plot(time_arr2, 100*clust_size_arr2,
                    c='#2ca25f', lw=3.0, ls='-', alpha=1, label='Cluster Size')

        plt.plot(time_arr2, 100*clust_size_A_arr3,
                    c='blue', lw=3.0, ls='-', alpha=1, label='Cluster Size')
        plt.plot(time_arr2, 100*clust_size_B_arr3,
                    c='red', lw=3.0, ls='-', alpha=1, label='Cluster Size')
        """
        fsize=5


        #ax1.set_ylim(np.mean(press_all_arr)-0.5*np.mean(press_all_arr), np.mean(press_all_arr)+0.5*np.mean(press_all_arr))


        ax1.set_xlabel(r'Time ($\tau_\mathrm{B}$)', fontsize=24)

        ax1.set_ylabel(r'% Change Cluster Size', fontsize=24)

        #lat_theory = np.mean(lat_theory_arr)
        # Set all the x ticks for radial plots
        ax1.set_ylim(-5, 5)
        x_arr = np.array([self.tst, self.tst])
        y_arr = np.array([-100, 50.0])

        x_arr2 = np.array([0, 600])
        y_arr2 = np.array([0.0, 0.0])
        
        plt.plot(time_arr2, 100*clust_size_arr2/np.mean(clust_size_arr2[300:])-100,
                    c='#2ca25f', lw=2.0, ls='-', alpha=1, label='Cluster Size')
        plt.plot(x_arr2, y_arr2, linestyle='dashed', color='black', linewidth=2.0)
        plt.plot(x_arr, y_arr, linestyle='solid', color='black', linewidth=2.0, alpha=0.5)
        
        ax1.set_xlim(300, 600)
        #ax1.legend(loc='upper right', fontsize=fsize*2.6)
        #step = 2.0
        # Set y ticks

        # Left middle plot
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)


        plt.tight_layout()
        plt.savefig(self.outPath + 'clust_fluctuations_' + self.outFile + ".png", dpi=150, transparent=False)
        plt.close()   
    def plot_part_activity_wide_desorb(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        if self.peA == 0:
            pos[:,1] = pos[:,1] - 40
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 100:
            pos[:,1] = pos[:,1] - 10
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 50:
            pos[:,1] = pos[:,1] + 20
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 150:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 250:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0

        out = np.where(pos[:,0]<-self.hx_box)[0]
        pos[out,0] = pos[out,0] + self.lx_box

        out = np.where(pos[:,0]>self.hx_box)[0]
        pos[out,0] = pos[out,0] - self.lx_box

        out = np.where(pos[:,1]<-self.hy_box)[0]
        pos[out,1] = pos[out,1] + self.ly_box

        out = np.where(pos[:,1]>self.hy_box)[0]
        pos[out,1] = pos[out,1] - self.ly_box

        if self.peA==100:
            pos_old = np.copy(pos)
            pos[:,0]=pos_old[:,1]
            pos[:,1]= -1 * pos_old[:,0]
        elif self.peA==50:
            pos_old = np.copy(pos)
            pos[:,0]= np.cos(np.pi/4) * pos_old[:,0] - np.sin(np.pi/4) * pos_old[:,1]
            pos[:,1]= np.sin(np.pi/4) * pos_old[:,0] + np.cos(np.pi/4) * pos_old[:,1]

        sz = 0.755

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            

            #Assign type 0 particles to plot
            if self.peA==50:
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)


                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                

                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

            
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]-self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]-self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]-self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)


            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            plt.quiver((pos[:,0]+self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, headwidth=6, color='black', alpha=0.8)

            if self.peA == 50:
                
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

                
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+3*self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+3*self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+3*self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)

            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.99, 1.19], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
            

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peA)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.13], handlelength=1.5, columnspacing=0.4, fontsize=32, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass

        
        
        if orientation_id == True:
            try:
                if active_fa_dict!=None:

                    
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                #plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                #plt.xlim(self.hy_box-25-2, self.hy_box+25+2)
                if self.peA==0:
                    plt.ylim(self.hy_box-15, self.hy_box+35)
                    plt.xlim(self.hx_box-70, self.lx_box-120)
                elif self.peA==100:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hy_box-25, self.hy_box+25)
                    plt.xlim(self.hx_box-100, self.lx_box-150)
                    #plt.ylim(0, self.ly_box)
                    #plt.xlim(0, self.lx_box)
                elif self.peA==50:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hx_box-40, self.hy_box+10)
                    plt.xlim(self.hx_box-80, self.lx_box-130)

                elif self.peA==150:
                    plt.ylim(self.hy_box-25, self.hy_box+25)
                    plt.xlim(self.hx_box-115, self.lx_box-165)
                elif self.peA==250:
                    plt.ylim(self.hy_box-25, self.hy_box+25)
                    plt.xlim(self.hx_box-115, self.lx_box-165)
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(55, self.lx_box+60)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                plt.text(0.66, 0.06, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_desorb_' + self.outFile + ".png", dpi=200, transparent=True, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_wide_desorb_orient(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        if self.peA == 0:
            pos[:,1] = pos[:,1] - 40
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 100:
            pos[:,1] = pos[:,1] - 10
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 50:
            pos[:,1] = pos[:,1] + 20
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 150:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 250:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0

        out = np.where(pos[:,0]<-self.hx_box)[0]
        pos[out,0] = pos[out,0] + self.lx_box

        out = np.where(pos[:,0]>self.hx_box)[0]
        pos[out,0] = pos[out,0] - self.lx_box

        out = np.where(pos[:,1]<-self.hy_box)[0]
        pos[out,1] = pos[out,1] + self.ly_box

        out = np.where(pos[:,1]>self.hy_box)[0]
        pos[out,1] = pos[out,1] - self.ly_box

        if self.peA==100:
            pos_old = np.copy(pos)
            pos[:,0]=pos_old[:,1]
            pos[:,1]= -1 * pos_old[:,0]
        elif self.peA==50:
            pos_old = np.copy(pos)
            pos[:,0]= np.cos(np.pi/4) * pos_old[:,0] - np.sin(np.pi/4) * pos_old[:,1]
            pos[:,1]= np.sin(np.pi/4) * pos_old[:,0] + np.cos(np.pi/4) * pos_old[:,1]

        sz = 0.755

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            

            #Assign type 0 particles to plot
            if self.peA==50:
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)


                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

            
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]-self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]-self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]-self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)


            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            plt.quiver((pos[:,0]+self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=180.0, headwidth=10, color='black', alpha=0.8)

            if self.peA == 50:
                
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

                
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+3*self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+3*self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+3*self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)

            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.99, 1.19], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
            

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peA)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.13], handlelength=1.5, columnspacing=0.4, fontsize=32, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass

        
        
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                #plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                #plt.xlim(self.hy_box-25-2, self.hy_box+25+2)
                if self.peA==0:
                    plt.ylim(self.hy_box-5, self.hy_box+25)
                    plt.xlim(self.hx_box-65, self.lx_box-145)
                elif self.peA==100:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hy_box-15, self.hy_box+15)
                    plt.xlim(self.hx_box-95, self.lx_box-175)
                    #plt.ylim(0, self.ly_box)
                    #plt.xlim(0, self.lx_box)
                elif self.peA==50:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hx_box-30, self.hy_box)
                    plt.xlim(self.hx_box-70, self.lx_box-150)

                elif self.peA==150:
                    plt.ylim(self.hy_box-15, self.hy_box+15)
                    plt.xlim(self.hx_box-95, self.lx_box-175)
                elif self.peA==250:
                    plt.ylim(self.hy_box-15, self.hy_box+15)
                    plt.xlim(self.hx_box-100, self.lx_box-180)
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(55, self.lx_box+60)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                plt.text(0.66, 0.06, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_desorb_orient_' + self.outFile + ".png", dpi=200, transparent=True, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_wide_adsorb(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        if self.peA == 0:
            pos[:,1] = pos[:,1] - 40
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 50:
            pos[:,1] = pos[:,1] + 20
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 100:
            pos[:,1] = pos[:,1] - 10
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 150:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 250:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0

        out = np.where(pos[:,0]<-self.hx_box)[0]
        pos[out,0] = pos[out,0] + self.lx_box

        out = np.where(pos[:,0]>self.hx_box)[0]
        pos[out,0] = pos[out,0] - self.lx_box

        out = np.where(pos[:,1]<-self.hy_box)[0]
        pos[out,1] = pos[out,1] + self.ly_box

        out = np.where(pos[:,1]>self.hy_box)[0]
        pos[out,1] = pos[out,1] - self.ly_box
        
        if self.peA==100:
            pos_old = np.copy(pos)
            pos[:,0]=pos_old[:,1]
            pos[:,1]= -1 * pos_old[:,0]
        elif self.peA==50:
            pos_old = np.copy(pos)
            pos[:,0]= np.cos(np.pi/4) * pos_old[:,0] - np.sin(np.pi/4) * pos_old[:,1]
            pos[:,1]= np.sin(np.pi/4) * pos_old[:,0] + np.cos(np.pi/4) * pos_old[:,1]
        
        

        sz = 0.755

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            

            #Assign type 0 particles to plot
            if self.peA==50:
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)


                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)
            
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]-self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]-self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            if self.peA == 50:
                
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)
                
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+3*self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+3*self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)
            
            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.99, 1.19], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peA)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.13], handlelength=1.5, columnspacing=0.4, fontsize=32, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                #plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                #plt.xlim(self.hy_box-25-2, self.hy_box+25+2)
                
                if self.peA==0:
                    plt.ylim(self.hy_box-15, self.hy_box+35)
                    plt.xlim(self.hx_box+50, self.lx_box)
                elif self.peA==50:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hx_box-40, self.hy_box+10)
                    plt.xlim(self.hx_box+20, self.lx_box-30)
                elif self.peA==100:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hy_box-25, self.hy_box+25)
                    plt.xlim(self.hx_box+40, self.lx_box-10)
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(55, self.lx_box+60)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                plt.text(0.66, 0.06, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_adsorb_' + self.outFile + ".png", dpi=200, transparent=True, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_wide_adsorb_orient(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        if self.peA == 0:
            pos[:,1] = pos[:,1] - 40
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 50:
            pos[:,1] = pos[:,1] + 20
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 100:
            pos[:,1] = pos[:,1] - 10
            pos[:,0] = pos[:,0] - 50
        elif self.peA == 150:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0
        elif self.peA == 250:
            pos[:,1] = pos[:,1] - 0
            pos[:,0] = pos[:,0] - 0

        out = np.where(pos[:,0]<-self.hx_box)[0]
        pos[out,0] = pos[out,0] + self.lx_box

        out = np.where(pos[:,0]>self.hx_box)[0]
        pos[out,0] = pos[out,0] - self.lx_box

        out = np.where(pos[:,1]<-self.hy_box)[0]
        pos[out,1] = pos[out,1] + self.ly_box

        out = np.where(pos[:,1]>self.hy_box)[0]
        pos[out,1] = pos[out,1] - self.ly_box
        
        if self.peA==100:
            pos_old = np.copy(pos)
            pos[:,0]=pos_old[:,1]
            pos[:,1]= -1 * pos_old[:,0]
        elif self.peA==50:
            pos_old = np.copy(pos)
            pos[:,0]= np.cos(np.pi/4) * pos_old[:,0] - np.sin(np.pi/4) * pos_old[:,1]
            pos[:,1]= np.sin(np.pi/4) * pos_old[:,0] + np.cos(np.pi/4) * pos_old[:,1]
        
        sz = 0.755

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            

            #Assign type 0 particles to plot
            if self.peA==50:
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)


                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)-(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

            
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]-self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]-self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]-self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)


            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            plt.quiver((pos[:,0]+self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=180.0, headwidth=10, color='black', alpha=0.8)

            if self.peA == 50:
                
                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                ells0 = [Ellipse(xy=np.array([(pos0[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos0[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([(pos1[i,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos1[i,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)-(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)
                plt.quiver((pos[:,0]+self.hx_box)+(np.sqrt(2)/2)*self.lx_box, (pos[:,1]+self.hy_box)+(np.sqrt(2)/2)*self.ly_box, px, py, scale=200.0, color='black', alpha=0.8)

                
            else:
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+3*self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(pos0))]

                #Assign type 1 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+3*self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(pos1))]

                # Plot position colored by neighbor number
                if self.peA <= self.peB:
                    slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
                else:
                    slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                    fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
                ax.add_collection(slowGroup)
                ax.add_collection(fastGroup)

                plt.quiver((pos[:,0]+3*self.hx_box), (pos[:,1]+self.hy_box), px, py, scale=200.0, color='black', alpha=0.8)

            
            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.99, 1.19], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peA)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.13], handlelength=1.5, columnspacing=0.4, fontsize=32, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                #plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                #plt.xlim(self.hy_box-25-2, self.hy_box+25+2)
                
                if self.peA==0:
                    plt.ylim(self.hy_box-5, self.hy_box+25)
                    plt.xlim(self.hx_box+80, self.lx_box)

                elif self.peA==50:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hx_box-35, self.hy_box-5)
                    plt.xlim(self.hx_box+30, self.lx_box-50)
                elif self.peA==100:
                    #plt.ylim(self.hy_box-25, self.hy_box+25)
                    #plt.xlim(self.hx_box-115, self.lx_box-165)
                    plt.ylim(self.hy_box-15, self.hy_box+15)
                    plt.xlim(self.hx_box+55, self.lx_box-25)
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(55, self.lx_box+60)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                plt.text(0.66, 0.06, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_adsorb_orient_' + self.outFile + ".png", dpi=200, transparent=True, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_paper(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(1,2, figsize=(x_dim,2*y_dim), facecolor='white')

        sz = 0.755
        """
        pos[:,0]=pos[:,0]+50
        test_id = np.where(pos[:,0]>self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<-self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        pos[:,1]=pos[:,1]-80
        test_id = np.where(pos[:,1]>self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<-self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box
        """
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax[0].add_collection(slowGroup)
            ax[0].add_collection(fastGroup)


            ells02 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells12 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup2 = mc.PatchCollection(ells02, facecolors=slowCol)
                fastGroup2 = mc.PatchCollection(ells12,facecolors=fastCol)
            else:
                slowGroup2 = mc.PatchCollection(ells12, facecolors=slowCol)
                fastGroup2 = mc.PatchCollection(ells02,facecolors=fastCol)
            ax[1].add_collection(slowGroup2)
            ax[1].add_collection(fastGroup2)

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            ax[1].quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px[typ1ind], py[typ1ind], color='black', scale=60, headwidth=10, headaxislength = 10, headlength=10)# width=0.02)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

            ax[0].set_ylim(0, self.ly_box)
            ax[0].set_xlim(0, self.lx_box)
            
            print(self.tst)
            if self.tst == 9.12:
                x_min = self.hx_box-65
                y_min = self.hy_box+22.5
                x_max = self.hx_box-45
                y_max = self.hy_box+42.5
                
            else:
                x_min = 0
                y_min = 0
                x_max = self.lx_box
                y_max = self.ly_box
            ax[1].set_ylim(y_min, y_max)
            ax[1].set_xlim(x_min, x_max)
            #plt.ylim(self.hy_box-90, self.hy_box+90)
            #plt.xlim(self.hx_box-80, self.hx_box+100)

        ax[0].plot([x_min, x_min], [y_min, y_max], linewidth = 1.0, linestyle='dashed', c='black')
        ax[0].plot([x_max, x_max], [y_min, y_max], linewidth = 1.0, linestyle='dashed', c='black')
        ax[0].plot([x_min, x_max], [y_min, y_min], linewidth = 1.0, linestyle='dashed', c='black')
        ax[0].plot([x_min, x_max], [y_max, y_max], linewidth = 1.0, linestyle='dashed', c='black')
        
        ax[0].plot([x_min, self.lx_box+10], [y_min, 0], linewidth = 1.0, linestyle='dashed', c='black', clip_on=False)
        ax[0].plot([x_min, self.lx_box + 10], [y_max, self.ly_box], linewidth = 1.0, linestyle='dashed', c='black', clip_on=False)

        ax[0].axes.set_xticks([])
        ax[0].axes.set_yticks([])
        ax[0].axes.set_xticklabels([])
        ax[0].axes.set_yticks([])
        ax[0].set_aspect('equal')

        ax[1].axes.set_xticks([])
        ax[1].axes.set_yticks([])
        ax[1].axes.set_xticklabels([])
        ax[1].axes.set_yticks([])
        ax[1].set_aspect('equal')

        #fig.subplots_adjust(hspace=-5)
        fig.subplots_adjust(wspace=-5)

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=300, transparent=True, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close() 

    def plot_part_activity(self, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, mono_slow_id = False, mono_fast_id = False, swap_col_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#b2182b'
        slowCol = '#4393c3'

        if mono_slow_id == True:
            fastCol = '#4393c3'
        elif mono_fast_id == True:
            slowCol = '#b2182b'
        elif swap_col_id == True:
            fastCol = '#4393c3'
            slowCol = '#b2182b'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        sz = 0.755
        """
        pos[:,0]=pos[:,0]+50
        test_id = np.where(pos[:,0]>self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<-self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        pos[:,1]=pos[:,1]-80
        test_id = np.where(pos[:,1]>self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<-self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box
        """
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            
            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                if (self.peA>=100):
                    one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.02, 1.15], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                    ax.add_artist(one_leg)
                elif (self.peA>=10) & (self.peA<100):
                    one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.97, 1.15], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                    ax.add_artist(one_leg)
                elif (self.peA<10):
                    one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.94, 1.15], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                    ax.add_artist(one_leg)

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peA)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.13], handlelength=1.5, columnspacing=0.4, fontsize=32, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8, width=0.004)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                plt.xlim(self.hy_box-25-2, self.hy_box+25+2)

                
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(0, self.lx_box)
                #plt.ylim(self.hy_box-90, self.hy_box+90)
                #plt.xlim(self.hx_box-80, self.hx_box+100)
        longleaf_opt = True
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                if longleaf_opt == True:
                    if self.tst<10:
                        plt.text(0.68-0.06, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    elif (self.tst>=10) & (self.tst<100):
                        plt.text(0.65-0.07, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    elif (self.tst>=100):
                        plt.text(0.62-0.08, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                else:
                    if self.tst<10:
                        plt.text(0.68, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    elif (self.tst>=10) & (self.tst<100):
                        plt.text(0.65, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    elif (self.tst>=100):
                        plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                            fontsize=30, transform = ax.transAxes,
                            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=120, transparent=False, bbox_inches='tight')
        plt.close() 

    def plot_part_activity_blank(self, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, mono_slow_id = False, mono_fast_id = False, swap_col_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#b2182b'
        slowCol = '#4393c3'

        if mono_slow_id == True:
            fastCol = '#4393c3'
        elif mono_fast_id == True:
            slowCol = '#b2182b'
        elif swap_col_id == True:
            fastCol = '#4393c3'
            slowCol = '#b2182b'


        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        sz = 0.755
        """
        pos[:,0]=pos[:,0]+50
        test_id = np.where(pos[:,0]>self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<-self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        pos[:,1]=pos[:,1]-80
        test_id = np.where(pos[:,1]>self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<-self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box
        """
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            
            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8, width=0.004)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                plt.xlim(self.hy_box-25-2, self.hy_box+25+2)

                
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(0, self.lx_box)
                #plt.ylim(self.hy_box-90, self.hy_box+90)
                #plt.xlim(self.hx_box-80, self.hx_box+100)
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()

        if mono_slow_id == True:
            plt.savefig(self.outPath + 'part_activity_blank_mono_slow_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif mono_fast_id == True:
            plt.savefig(self.outPath + 'part_activity_blank_mono_fast_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif swap_col_id == True:
            plt.savefig(self.outPath + 'part_activity_blank_swap_col_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        else:
            plt.savefig(self.outPath + 'part_activity_blank_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()  
    
    def plot_part_activity_blank_video(self, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, mono_slow_id = False, mono_fast_id = False, swap_col_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#b2182b'
        slowCol = '#4393c3'

        if mono_slow_id == True:
            fastCol = '#4393c3'
        elif mono_fast_id == True:
            slowCol = '#b2182b'
        elif swap_col_id == True:
            fastCol = '#4393c3'
            slowCol = '#b2182b'


        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        sz = 0.755
        """
        pos[:,0]=pos[:,0]+50
        test_id = np.where(pos[:,0]>self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<-self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        pos[:,1]=pos[:,1]-80
        test_id = np.where(pos[:,1]>self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<-self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box
        """
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            
            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8, width=0.004)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                plt.ylim(self.hy_box-25-2, self.hy_box+25+2)
                plt.xlim(self.hy_box-25-2, self.hy_box+25+2)

                
            elif banner_id == True: 
                plt.ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                plt.xlim(0, self.lx_box)
            else:
                plt.ylim(0, self.ly_box)
                plt.xlim(0, self.lx_box)
                #plt.ylim(self.hy_box-90, self.hy_box+90)
                #plt.xlim(self.hx_box-80, self.hx_box+100)
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

                if self.tst<10:
                    plt.text(0.68, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif (self.tst>=10) & (self.tst<100):
                    plt.text(0.65, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif (self.tst>=100):
                    plt.text(0.62, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.tight_layout()


        plt.savefig(self.outPath + 'part_activity_blank_video_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close() 

    def plot_avg_radial_heterogeneity(self, single_time_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, measure="fa", types="all"):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')
        #ax = fig.add_subplot(111)

        sz = 0.755
        
        x_coords = single_time_dict['com']['x'] + np.einsum('i,j', single_time_dict['rad'] * single_time_dict['radius'], np.cos(single_time_dict['theta']*(np.pi/180)))
        y_coords = single_time_dict['com']['y'] + np.einsum('i,j', single_time_dict['rad'] * single_time_dict['radius'], np.sin(single_time_dict['theta']*(np.pi/180)))
        
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        
        vals = single_time_dict[measure][types].flatten()

        plt.tricontourf(x_coords, y_coords, vals, cmap='Reds')
        plt.colorbar()


        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax.set_xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            ax.set_ylim(0, self.ly_box)
            ax.set_xlim(0, self.lx_box)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                #    fontsize=24, transform = ax.transAxes,
                #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                ax.text(0.03, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                ax.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        #ax.axes.set_xticks([])
        #ax.axes.set_yticks([])
        #ax.axes.set_xticklabels([])
        #ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        plt.savefig(self.outPath + 'avg_radial_heterogeneity_' + self.outFile + ".png", dpi=900, transparent=False, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close() 

    

    def plot_radial_heterogeneity(self, averagesPath, inFile, pos, single_time_dict, plot_bin_dict, plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, measure="fa", types="all", circle_opt=0, component = 'dif'):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]
        
        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        # Minimum dimension length (in inches)
        scaling =7.0

        # X and Y-dimension lengths (in inches)
        x_dim = int(scaling + 1.0)
        y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')
        #ax = fig.add_subplot(111)

        sz = 0.755


        #Local each particle's positions
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind]
        """
        #Assign type 0 particles to plot

        ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                width=sz, height=sz, label='PeA: '+str(self.peA))
        for i in range(0,len(pos0))]

        #Assign type 1 particles to plot
        ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                width=sz, height=sz, label='PeB: '+str(self.peB))
        for i in range(0,len(pos1))]

        # Plot position colored by neighbor number
        if self.peA <= self.peB:
            slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
            fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
        else:
            slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
            fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
        ax.add_collection(slowGroup)
        ax.add_collection(fastGroup)
        """

        average_theta_path = averagesPath + 'radial_avgs_theta_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        average_rad_path = averagesPath + 'radial_avgs_rad_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        import csv 

        averages_file =  open(average_theta_path, newline='')
        avg_theta_r = np.array(list(csv.reader(averages_file)))

        avg_theta_r_flatten = (avg_theta_r.flatten()).astype(np.float) 
        avg_theta_r_flatten = np.append(avg_theta_r_flatten, avg_theta_r_flatten[0])
        averages_file =  open(average_rad_path, newline='')
        avg_rad_r = np.array(list(csv.reader(averages_file)))

        avg_rad_r_flatten = (avg_rad_r.flatten()).astype(np.float) 
        
        average_indiv_vals_path = averagesPath + 'radial_avgs_indiv_vals_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        averages_file =  open(average_indiv_vals_path, newline='')
        reader = csv.reader(averages_file)
        avg_indiv_vals = dict(reader)
        avg_radius_r_flatten= float(avg_indiv_vals['radius'])

        if component != 'avg':
            avg_theta_r_new = np.zeros(np.shape(single_time_dict[measure][types]))

            for i in range(0, len(single_time_dict['rad'])):
                avg_theta_r_new[i,:] = single_time_dict['theta']

            avg_rad_r_new = np.zeros(np.shape(single_time_dict[measure][types]))

            for i in range(0, len(single_time_dict['theta'])):
                avg_rad_r_new[:,i] = single_time_dict['rad']

            avg_rad_r_flatten = avg_rad_r_new.flatten()
            avg_theta_r_flatten = avg_theta_r_new.flatten()
        else:
            avg_theta_r_new = np.zeros((len(avg_rad_r_flatten), len(avg_theta_r_flatten)-1))
            for i in range(0, len(avg_rad_r_flatten)):
                avg_theta_r_new[i,:] = avg_theta_r_flatten[:-1]

            avg_rad_r_new = np.zeros((len(avg_rad_r_flatten), len(avg_theta_r_flatten)-1))

            for i in range(0, len(avg_theta_r_flatten)-1):
                avg_rad_r_new[:,i] = avg_rad_r_flatten

            avg_rad_r_flatten = avg_rad_r_new.flatten()
            avg_theta_r_flatten = avg_theta_r_new.flatten()

        avg_radius_r_flatten_new = (plot_bin_dict['rad_bin']['all'].flatten()).astype(np.float)
        
        if circle_opt == 1:
            x_coords = self.hx_box + avg_rad_r_flatten * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))
        else:
            
            if component == 'avg':
                x_coords = float(self.hx_box) + avg_rad_r_flatten * avg_radius_r_flatten_new * np.cos(avg_theta_r_flatten*(np.pi/180))
                y_coords = float(self.hy_box) + avg_rad_r_flatten * avg_radius_r_flatten_new * np.sin(avg_theta_r_flatten*(np.pi/180))
            else:
                x_coords = float(single_time_dict['com']['x']) + avg_rad_r_flatten * avg_radius_r_flatten_new * np.cos(avg_theta_r_flatten*(np.pi/180))
                y_coords = float(single_time_dict['com']['y']) + avg_rad_r_flatten * avg_radius_r_flatten_new * np.sin(avg_theta_r_flatten*(np.pi/180))

        #x_coords = x_coords.flatten()
        #y_coords = y_coords.flatten()



        
        average_theta_path = averagesPath + 'radial_avgs_theta_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        average_rad_path = averagesPath + 'radial_avgs_rad_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        if measure == 'fa_avg':
            average_val_path = averagesPath + 'radial_avgs_fa_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'radial_avgs_faA_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_faB_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'fa_dens':
            average_val_path = averagesPath + 'radial_avgs_fa_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'radial_avgs_faA_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_faB_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'fa_avg_real':
            average_val_path = averagesPath + 'radial_avgs_fa_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'radial_avgs_faA_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_faB_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'num_dens':
            average_val_path = averagesPath + 'radial_avgs_num_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'radial_avgs_num_densA_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_num_densB_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'part_frac':
            average_valA_path = averagesPath + 'radial_avgs_part_fracA_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_part_fracB_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        
        elif measure == 'align':
            average_val_path = averagesPath + 'radial_avgs_align_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'radial_avgs_alignA_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'radial_avgs_alignB_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        average_indiv_vals_path = averagesPath + 'radial_avgs_indiv_vals_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        import csv 

        if types == 'all':
            averages_file =  open(average_val_path, newline='')
            avg_val_r = np.array(list(csv.reader(averages_file)))

            avg_val_r_flatten = (avg_val_r.flatten()).astype(np.float) 

        elif types == 'A':
            averages_file =  open(average_valA_path, newline='')
            avg_valA_r = np.array(list(csv.reader(averages_file)))

            avg_valA_r_flatten = (avg_valA_r.flatten()).astype(np.float) 

        elif types == 'B':
            averages_file =  open(average_valB_path, newline='')
            avg_valB_r = np.array(list(csv.reader(averages_file)))

            avg_valB_r_flatten = (avg_valB_r.flatten()).astype(np.float) 

        averages_file =  open(average_indiv_vals_path, newline='')
        reader = csv.reader(averages_file)
        avg_indiv_vals = dict(reader)
        avg_radius_r_flatten= float(avg_indiv_vals['radius'])

        avg_com_x_r_flatten = float(avg_indiv_vals['com_x'])
        avg_com_y_r_flatten = float(avg_indiv_vals['com_y'])



        
        #vals = avg_num_dens_r_flatten#single_time_dict[measure][types].flatten()
        #vals = (single_time_dict[measure][types].flatten()*(avg_num_dens_r_flatten**2))**0.5+avg_num_dens_r_flatten
        
        if component == 'dif':
            if types == 'all':
                vals = plot_bin_dict[measure][types].flatten() - avg_val_r_flatten #/ avg_num_dens_r_flatten**2
            elif types == 'A':
                vals = plot_bin_dict[measure][types].flatten() - avg_valA_r_flatten# 
            elif types == 'B':
                vals = plot_bin_dict[measure][types].flatten() - avg_valB_r_flatten #
        elif component == 'current':
            if types == 'all':
                vals = plot_bin_dict[measure][types].flatten()
            elif types == 'A':
                vals = plot_bin_dict[measure][types].flatten()
            elif types == 'B':
                vals = plot_bin_dict[measure][types].flatten()
        elif component == 'avg':
            if types == 'all':
                vals = avg_val_r_flatten #/ avg_num_dens_r_flatten**2
            elif types == 'A':
                vals = avg_valA_r_flatten# 
            elif types == 'B':
                vals = avg_valB_r_flatten #
        #test_id = np.where(avg_num_dens_r_flatten>0)[0]
        #vals[test_id] = vals[test_id] / avg_num_dens_r_flatten[test_id]**2
        #test_id = np.where(vals>3.0)[0]
        #vals[test_id]=3.0
        #if component == 'dif':
        if component == 'dif':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 2.0
                elif (types =='B'):
                    limit_vals = 2.0
                elif (types =='A'):
                    limit_vals = 1.25
            elif measure == 'part_frac':
                if (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 1.0 * 2.0 * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB * 1.0 * 2.0
                elif (types =='A'):
                    limit_vals = self.peA * 1.0 * 2.0
            elif measure == 'align':
                if (types =='all'):
                    limit_vals = 1.0
                elif (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
        elif component == 'current':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 2.5
                elif (types =='B'):
                    limit_vals = 2.5
                elif (types =='A'):
                    limit_vals = 1.25
            elif measure == 'part_frac':
                if (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 1.0 * 2.5
                elif (types =='B'):
                    limit_vals = self.peB * 1.0 * 2.5
                elif (types =='A'):
                    limit_vals = self.peA * 1.0 * 2.0
            elif measure == 'align':
                if (types == 'all'):
                    limit_vals = 1.0
                elif (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
        elif component == 'avg':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 1.6
                elif (types =='B'):
                    limit_vals = 1.1
                elif (types =='A'):
                    limit_vals = 0.6
            elif measure == 'part_frac':
                if (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3) * 0.3
                elif (types =='B'):
                    limit_vals = self.peB * 0.4
                elif (types =='A'):
                    limit_vals = self.peA * 0.2
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB * (0.55)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 0.3 * 1.6 * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB * 0.4 * 1.1
                elif (types =='A'):
                    limit_vals = self.peA * 0.2 * 0.6

            elif measure == 'align':
                if (types == 'all'):
                    limit_vals = 0.2
                elif (types =='B'):
                    limit_vals = 0.3
                elif (types =='A'):
                    limit_vals = 0.1
                
        """
        else:
            if measure == 'num_dens':
                limit_vals = 1.5
            elif measure == 'fa_avg':
                limit_vals = 500
            elif measure == 'fa_avg_real':
                limit_vals = 500
            elif measure == 'fa_dens':
                limit_vals = 750
            elif measure == 'align':
                limit_vals = 0.75
        """
        if component == 'dif':
            im = plt.tricontourf(x_coords, y_coords, vals, cmap='seismic', vmin=-limit_vals, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())
        
        else:
            if (measure == 'num_dens') | (measure == 'fa_avg_real') | (measure == 'part_frac'):
                im = plt.tricontourf(x_coords, y_coords, vals, cmap='Greens', vmin=0, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())
            else:
                im = plt.tricontourf(x_coords, y_coords, vals, cmap='seismic', vmin=-limit_vals, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())

        if circle_opt == 1:
            x_coords = self.hx_box + 1.0 * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
            y_coords = self.hy_box + 1.0 * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))

            plt.plot(x_coords, y_coords, c='black', linewidth=3.0) 

        else:
            
            if interface_id == True:
                try:

                    if sep_surface_dict!=None:
                        
                        for m in range(0, len(sep_surface_dict)):
                            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                            print(key)

                            
                            try:
                                pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                                pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                                plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                                plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                                
                            
                            except:
                                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                            
                            try:
                                
                                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                                plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                                
                            except:
                                pass
                            
                except:
                    pass
        

        sm = plt.cm.ScalarMappable(norm=im.norm, cmap = im.cmap)
        sm.set_array([])
        clb = fig.colorbar(sm)#, ticks=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])#, ax=ax2)
        #clb.ax.set_title(r'$\mathrm{Pe}_\mathrm{F}$', fontsize=23)
        if (component == 'current'):
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$F_a \alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$n F_a \alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$n$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'part_frac':
                if types == 'A':
                    clb.set_label(r'$\chi^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\chi^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$F_a$', fontsize=23, rotation=270, labelpad=28)
        elif (component == 'avg'):
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$\langle F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$\langle n F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\langle \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$\langle n \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle n^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle n^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'part_frac':
                if types == 'A':
                    clb.set_label(r'$\langle \chi^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle \chi^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$\langle F_a \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
        elif component == 'dif':
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$F_a \alpha - \langle F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{S} - \langle F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{F} - \langle F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$n F_a \alpha - \langle n F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} - \langle n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F} - \langle n^\mathrm{F} F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\alpha - \langle \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\alpha^\mathrm{S} - \langle \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\alpha^\mathrm{F} - \langle \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$n - \langle n \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} - \langle n^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} - \langle n^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'part_frac':
                if types == 'A':
                    clb.set_label(r'$\chi^\mathrm{S} - \langle \chi^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\chi^\mathrm{F} - \langle \chi^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$F_a - \langle F_a \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)

        clb.ax.tick_params(labelsize=20)
        
        
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass
        """
        

        
        for i in range(0, len(avg_theta_r_flatten)-1):   
            x_coords = self.hx_box + avg_rad_r_flatten * avg_radius_r_flatten * np.cos(avg_theta_r_flatten[i]*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten * avg_radius_r_flatten * np.sin(avg_theta_r_flatten[i]*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.5)

        for i in range(0, len(avg_rad_r_flatten)):   
            x_coords = self.hx_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.5)
        """
        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax.set_xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            ax.set_ylim(0, self.ly_box)
            ax.set_xlim(0, self.lx_box)
            ax.set_ylim(25, self.ly_box-25)
            ax.set_xlim(25, self.lx_box-25)
        
        
        # Label simulation time
        if component != 'avg':
            if banner_id == False:
                if self.lx_box == self.ly_box:
                    #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    #    fontsize=24, transform = ax.transAxes,
                    #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    ax.text(0.03, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif self.lx_box > self.ly_box:
                    ax.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=18, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')
        
        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        if component == 'dif':
            if circle_opt == 1:
                plt.savefig(self.outPath + 'dif_circle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
            else:
                plt.savefig(self.outPath + 'dif_noncircle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif component == 'current':
            if circle_opt == 1:
                plt.savefig(self.outPath + 'current_circle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
            else:
                plt.savefig(self.outPath + 'current_noncircle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif component == 'avg':
            if circle_opt == 1:
                plt.savefig(self.outPath + 'avg_circle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
            else:
                plt.savefig(self.outPath + 'avg_noncircle_radial_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close() 


    


    def plot_bulk_heterogeneity(self, averagesPath, inFile, pos, single_time_dict, plot_bin_dict, plot_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False, measure="fa", types="all", circle_opt=0, component = 'dif'):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space.

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]
        
        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        # Minimum dimension length (in inches)
        scaling =7.0

        # X and Y-dimension lengths (in inches)
        x_dim = int(scaling + 1.0)
        y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')
        #ax = fig.add_subplot(111)

        sz = 0.755

        plt.scatter(pos[:,0]+self.hx_box, pos[:,1]+self.hy_box, s=0.7, c='black')
        #Local each particle's positions
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind]
        """
        #Assign type 0 particles to plot

        ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                width=sz, height=sz, label='PeA: '+str(self.peA))
        for i in range(0,len(pos0))]

        #Assign type 1 particles to plot
        ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                width=sz, height=sz, label='PeB: '+str(self.peB))
        for i in range(0,len(pos1))]

        # Plot position colored by neighbor number
        if self.peA <= self.peB:
            slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
            fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
        else:
            slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
            fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
        ax.add_collection(slowGroup)
        ax.add_collection(fastGroup)
        """

        average_x_path = averagesPath + 'bulk_avgs_x_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        average_y_path = averagesPath + 'bulk_avgs_y_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        import csv 

        averages_file =  open(average_x_path, newline='')
        avg_x = np.array(list(csv.reader(averages_file)))

        avg_x_flatten = (avg_x.flatten()).astype(np.float) 
        avg_x_flatten = np.append(avg_x_flatten, avg_x_flatten[0])
        averages_file =  open(average_y_path, newline='')
        avg_y = np.array(list(csv.reader(averages_file)))

        avg_y_flatten = (avg_y.flatten()).astype(np.float) 

        if component != 'avg':
            avg_y_new = np.zeros(np.shape(single_time_dict[measure][types]))

            for i in range(0, len(single_time_dict['x'])):
                avg_y_new[i,:] = single_time_dict['y']

            avg_x_new = np.zeros(np.shape(single_time_dict[measure][types]))

            for i in range(0, len(single_time_dict['y'])):
                avg_x_new[:,i] = single_time_dict['x']

            avg_x_flatten = avg_x_new.flatten()
            avg_y_flatten = avg_y_new.flatten()
        else:
            avg_y_new = np.zeros((len(avg_x_flatten), len(avg_y_flatten)-1))
            for i in range(0, len(avg_x_flatten)):
                avg_y_new[i,:] = avg_y_flatten[:-1]

            avg_x_new = np.zeros((len(avg_x_flatten), len(avg_y_flatten)-1))

            for i in range(0, len(avg_y_flatten)-1):
                avg_x_new[:,i] = avg_x_flatten

            avg_x_flatten = avg_x_new.flatten()
            avg_y_flatten = avg_y_new.flatten()
        
        if circle_opt == 1:
            x_coords = avg_x_flatten
            y_coords = avg_y_flatten
        else:
            
            if component == 'avg':
                x_coords = avg_x_flatten
                y_coords = avg_y_flatten
            else:
                x_coords = avg_x_flatten
                y_coords = avg_y_flatten

        #x_coords = x_coords.flatten()
        #y_coords = y_coords.flatten()



        
        average_y_path = averagesPath + 'bulk_avgs_y_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        average_x_path = averagesPath + 'bulk_avgs_x_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        if measure == 'fa_avg':
            average_val_path = averagesPath + 'bulk_avgs_fa_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'bulk_avgs_faA_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'bulk_avgs_faB_avg_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'fa_dens':
            average_val_path = averagesPath + 'bulk_avgs_fa_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'bulk_avgs_faA_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'bulk_avgs_faB_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'fa_avg_real':
            average_val_path = averagesPath + 'bulk_avgs_fa_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'bulk_avgs_faA_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'bulk_avgs_faB_avg_real_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'num_dens':
            average_val_path = averagesPath + 'bulk_avgs_num_dens_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'bulk_avgs_num_densA_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'bulk_avgs_num_densB_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
        elif measure == 'align':
            average_val_path = averagesPath + 'bulk_avgs_align_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valA_path = averagesPath + 'bulk_avgs_alignA_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'
            average_valB_path = averagesPath + 'bulk_avgs_alignB_pa' + str(int(self.peA)) + '_pb' + str(int(self.peB))  + '_xa' + str(int(self.parFrac)) + '_eps' + str(self.eps) + '_phi' + str(float(self.phi)) + '_pNum' + str(self.partNum) +  '_bin5_time1.csv'

        import csv 

        if types == 'all':
            averages_file =  open(average_val_path, newline='')
            avg_val_r = np.array(list(csv.reader(averages_file)))

            avg_val_r_flatten = (avg_val_r.flatten()).astype(np.float) 

        elif types == 'A':
            averages_file =  open(average_valA_path, newline='')
            avg_valA_r = np.array(list(csv.reader(averages_file)))

            avg_valA_r_flatten = (avg_valA_r.flatten()).astype(np.float) 

        elif types == 'B':
            averages_file =  open(average_valB_path, newline='')
            avg_valB_r = np.array(list(csv.reader(averages_file)))

            avg_valB_r_flatten = (avg_valB_r.flatten()).astype(np.float) 

        #vals = avg_num_dens_r_flatten#single_time_dict[measure][types].flatten()
        #vals = (single_time_dict[measure][types].flatten()*(avg_num_dens_r_flatten**2))**0.5+avg_num_dens_r_flatten
        
        if component == 'dif':
            if types == 'all':
                vals = plot_bin_dict[measure][types].flatten() - avg_val_r_flatten #/ avg_num_dens_r_flatten**2
            elif types == 'A':
                vals = plot_bin_dict[measure][types].flatten() - avg_valA_r_flatten# 
            elif types == 'B':
                vals = plot_bin_dict[measure][types].flatten() - avg_valB_r_flatten #
        elif component == 'current':
            if types == 'all':
                vals = plot_bin_dict[measure][types].flatten()
            elif types == 'A':
                vals = plot_bin_dict[measure][types].flatten()
            elif types == 'B':
                vals = plot_bin_dict[measure][types].flatten()
        elif component == 'avg':
            if types == 'all':
                vals = avg_val_r_flatten #/ avg_num_dens_r_flatten**2
            elif types == 'A':
                vals = avg_valA_r_flatten# 
            elif types == 'B':
                vals = avg_valB_r_flatten #
        #test_id = np.where(avg_num_dens_r_flatten>0)[0]
        #vals[test_id] = vals[test_id] / avg_num_dens_r_flatten[test_id]**2
        #test_id = np.where(vals>3.0)[0]
        #vals[test_id]=3.0
        #if component == 'dif':
        if component == 'dif':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 2.0
                elif (types =='B'):
                    limit_vals = 2.0
                elif (types =='A'):
                    limit_vals = 1.25
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 1.0 * 2.0 * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB * 1.0 * 2.0
                elif (types =='A'):
                    limit_vals = self.peA * 1.0 * 2.0
            elif measure == 'align':
                if (types =='all'):
                    limit_vals = 1.0
                elif (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0

                    
        elif component == 'current':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 2.5
                elif (types =='B'):
                    limit_vals = 2.5
                elif (types =='A'):
                    limit_vals = 1.25
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 1.0 * 2.5
                elif (types =='B'):
                    limit_vals = self.peB * 1.0 * 2.5
                elif (types =='A'):
                    limit_vals = self.peA * 1.0 * 2.0
            elif measure == 'align':
                if (types == 'all'):
                    limit_vals = 1.0
                elif (types =='B'):
                    limit_vals = 1.0
                elif (types =='A'):
                    limit_vals = 1.0
        elif component == 'avg':
            if measure == 'num_dens':
                if (types =='all'):
                    limit_vals = 1.6
                elif (types =='B'):
                    limit_vals = 1.1
                elif (types =='A'):
                    limit_vals = 0.6
            elif measure == 'fa_avg':
                if (types =='all'):
                    limit_vals = self.peB * (2/3) * 0.3
                elif (types =='B'):
                    limit_vals = self.peB * 0.4
                elif (types =='A'):
                    limit_vals = self.peA * 0.2
            elif measure == 'fa_avg_real':
                if (types =='all'):
                    limit_vals = self.peB * (0.55)
                elif (types =='B'):
                    limit_vals = self.peB
                elif (types =='A'):
                    limit_vals = self.peA
            elif measure == 'fa_dens':
                if (types == 'all'):
                    limit_vals = self.peB * 0.3 * 1.6 * (2/3)
                elif (types =='B'):
                    limit_vals = self.peB * 0.4 * 1.1
                elif (types =='A'):
                    limit_vals = self.peA * 0.2 * 0.6

            elif measure == 'align':
                if (types == 'all'):
                    limit_vals = 0.2
                elif (types =='B'):
                    limit_vals = 0.3
                elif (types =='A'):
                    limit_vals = 0.1
        limit_vals =  np.amax(vals)
        """
        else:
            if measure == 'num_dens':
                limit_vals = 1.5
            elif measure == 'fa_avg':
                limit_vals = 500
            elif measure == 'fa_avg_real':
                limit_vals = 500
            elif measure == 'fa_dens':
                limit_vals = 750
            elif measure == 'align':
                limit_vals = 0.75
        """
        if component == 'dif':
            im = plt.tricontourf(x_coords, y_coords, vals, cmap='seismic', vmin=-limit_vals, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())
        
        else:
            if (measure == 'num_dens') | (measure == 'fa_avg_real'):
                im = plt.tricontourf(x_coords, y_coords, vals, cmap='Greens', vmin=0, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())
            else:
                im = plt.tricontourf(x_coords, y_coords, vals, cmap='seismic', vmin=-limit_vals, vmax=limit_vals)#, norm=matplotlib.colors.LogNorm())#, vmin=-2.5, vmax=2.5, cmap='seismic')#,alpha=0.65, norm=matplotlib.colors.LogNorm())


        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                            
                        
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                            
                        except:
                            pass
                        
            except:
                pass
    

        sm = plt.cm.ScalarMappable(norm=im.norm, cmap = im.cmap)
        sm.set_array([])
        clb = fig.colorbar(sm)#, ticks=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])#, ax=ax2)
        #clb.ax.set_title(r'$\mathrm{Pe}_\mathrm{F}$', fontsize=23)
        if (component == 'current'):
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$F_a \alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$n F_a \alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\alpha$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\alpha^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\alpha^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$n$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$F_a$', fontsize=23, rotation=270, labelpad=28)
        elif (component == 'avg'):
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$\langle F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$\langle n F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\langle \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$\langle n \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\langle n^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\langle n^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$\langle F_a \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
        elif component == 'dif':
            if measure == 'fa_avg':
                if types == 'all':
                    clb.set_label(r'$F_a \alpha - \langle F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{S} - \langle F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$F_a^\mathrm{S} \alpha^\mathrm{F} - \langle F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_dens':
                if types == 'all':
                    clb.set_label(r'$n F_a \alpha - \langle n F_a \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} - \langle n^\mathrm{S} F_a^\mathrm{S} \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} F_a^\mathrm{F} \alpha^\mathrm{F} - \langle n^\mathrm{F} F_a^\mathrm{S} \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'align':
                if types == 'all':
                    clb.set_label(r'$\alpha - \langle \alpha \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$\alpha^\mathrm{S} - \langle \alpha^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$\alpha^\mathrm{F} - \langle \alpha^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'num_dens':
                if types == 'all':
                    clb.set_label(r'$n - \langle n \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'A':
                    clb.set_label(r'$n^\mathrm{S} - \langle n^\mathrm{S} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
                elif types == 'B':
                    clb.set_label(r'$n^\mathrm{F} - \langle n^\mathrm{F} \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)
            elif measure == 'fa_avg_real':
                if types == 'all':
                    clb.set_label(r'$F_a - \langle F_a \rangle_\mathrm{t}$', fontsize=23, rotation=270, labelpad=28)

        clb.ax.tick_params(labelsize=20)
        
        
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass
        """
        

        
        for i in range(0, len(avg_theta_r_flatten)-1):   
            x_coords = self.hx_box + avg_rad_r_flatten * avg_radius_r_flatten * np.cos(avg_theta_r_flatten[i]*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten * avg_radius_r_flatten * np.sin(avg_theta_r_flatten[i]*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.5)

        for i in range(0, len(avg_rad_r_flatten)):   
            x_coords = self.hx_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
            y_coords = self.hy_box + avg_rad_r_flatten[i] * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))
            plt.plot(x_coords, y_coords, c='black', linewidth=0.5, alpha=0.5)
        """
        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax.set_xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            ax.set_ylim(0, self.ly_box)
            ax.set_xlim(0, self.lx_box)
            ax.set_ylim(25, self.ly_box-25)
            ax.set_xlim(25, self.lx_box-25)
        
        
        # Label simulation time
        if component != 'avg':
            if banner_id == False:
                if self.lx_box == self.ly_box:
                    #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    #    fontsize=24, transform = ax.transAxes,
                    #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                    ax.text(0.03, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=30, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
                elif self.lx_box > self.ly_box:
                    ax.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                        fontsize=18, transform = ax.transAxes,
                        bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')
        
        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website
        if component == 'dif':
            plt.savefig(self.outPath + 'dif_bulk_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif component == 'current':
            plt.savefig(self.outPath + 'current_bulk_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        elif component == 'avg':
            plt.savefig(self.outPath + 'avg_bulk_heterogeneity_' + measure + '_' + types +'_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close() 


    def plot_part_activity_zoom_seg_new(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space for a specific Pe_S=0 and Pe_F=500 system to track a domain
        in that system (with zoomed in inset) to show the cycle of avalanche/herding process

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        px (partNum): x-orientation unit vector of active force

        py (partNum): y-orientation unit vector of active force

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        # Define y-dimension of figure
        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        # Check if monodisperse or binary system
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1

        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')

        import math

        # Define function to rotate particles
        def rotate(x,y): 
            #rotate x,y around xo,yo by theta (rad)
            theta=0
            xr=math.cos(theta)*(x)-math.sin(theta)*(y)
            yr=math.sin(theta)*(x)+math.cos(theta)*(y)
            return xr,yr

        # Rotated positions of particles
        pos_x_rot, pos_y_rot = rotate(pos[:,0], pos[:,1])

        # Shift particle positions to proper box size
        pos[:,0] = pos_x_rot + self.hx_box
        pos[:,1] = pos_y_rot + self.hy_box

        # Shift positions so domain doesn't move
        if (round(self.tst,1)==165.3):
            shift_x = 10
            pos[:,0]=pos[:,0]+shift_x
        else:
            shift_x = 0

        # Enforce periodic boundary conditions
        test_id = np.where(pos[:,0]>self.lx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<0)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        # Shift positions so domain doesn't move
        if (round(self.tst,1)==165.3):
            shift_y = 20
            pos[:,1]=pos[:,1]+shift_y #913
        else:
            shift_y = 0
        
        # Enforce periodic boundary conditions
        test_id = np.where(pos[:,1]>self.ly_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<0)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box

        # Define particle diameter
        sz = 0.755
        sz=0.77
        
        # Define inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Plot particles
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0], pos0[i,1]]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0], pos1[i,1]]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)

            # Place particles in main plot
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            # Plot legend
            """
            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.05, 1.17], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
            """

        # Plot inner and outer interfaces
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x + shift_x, pos_interior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box + shift_x, pos_interior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box + shift_x, pos_interior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + shift_x, pos_interior_surface_y+self.ly_box + shift_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + shift_x, pos_interior_surface_y-self.ly_box + shift_y, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y+self.ly_box + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y-self.ly_box + shift_y, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box + shift_x, pos_exterior_surface_y + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y+self.ly_box + shift_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + shift_x, pos_exterior_surface_y-self.ly_box + shift_y, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        
        # Plot average active force orientations
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax.set_xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                ax.set_ylim(self.hy_box-25-2, self.hy_box+25+2)
                ax.set_xlim(self.hy_box-25-2, self.hy_box+25+2)

                
            elif banner_id == True: 
                ax.set_ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                ax.set_xlim(0, self.lx_box)
            else:
                ax.set_ylim(0, self.ly_box)
                ax.set_xlim(0, self.lx_box)
                ax.set_ylim(self.hy_box-105, self.hy_box+105)
                ax.set_xlim(self.hx_box-105, self.hx_box+105)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                ax.text(0.03, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                ax.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        # Remove main plot axes labels
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        for axis in ['top', 'bottom', 'left', 'right']:

            ax.spines[axis].set_linewidth(2.0)

        print(round(self.tst,1))
        # Define inset location based on time step
        if (round(self.tst,1)==162.6):
            # 950
            x_min = self.hx_box-35
            x_max = self.hx_box-10
            y_min = self.hy_box-80
            y_max = self.hy_box-55
        if (round(self.tst,1)==154.5):
            # 950
            x_min = self.hx_box-20
            x_max = self.hx_box+5
            y_min = self.hy_box-15
            y_max = self.hy_box+10
            
        if (round(self.tst,1)==165.3):
            # 950
            #x_min = self.hx_box-65
            #x_max = self.hx_box+5
            #y_min = self.hy_box-80
            #y_max = self.hy_box-20
            x_min = self.hx_box-35
            x_max = self.hx_box-10
            y_min = self.hy_box-80
            y_max = self.hy_box-55
        if (round(self.tst,1)==143.7):
            # 950
            x_min = self.hx_box-55
            x_max = self.hx_box+30
            y_min = self.hy_box+45
            y_max = self.hy_box+100
        if (round(self.tst,1)==144.0):
            # 950
            x_min = self.hx_box-25
            x_max = self.hx_box+30
            y_min = self.hy_box+55
            y_max = self.hy_box+100

        #if (round(self.tst,1)<=144.6):
            # 950
        #    x_min = self.hx_box-55
        #    x_max = self.hx_box
        #    y_min = self.hy_box-85
        #    y_max = self.hy_box-30
        #if (round(self.tst,1)<=158.7):
            # 950
        #    x_min = self.hx_box-55
        #    x_max = self.hx_box
        #    y_min = self.hy_box-85
        #    y_max = self.hy_box-30
        
        # Plot location of inset on main plot
        ax.plot([x_min, x_max], [y_min, y_min], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_max, x_max], [y_min, y_max], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_min, x_max], [y_max, y_max], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_min, x_min], [y_min, y_max], linestyle='dashed', linewidth=1.5, color='black')

        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website

        # Save image
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=900, transparent=False, bbox_inches='tight')
        plt.close()  


    def plot_part_activity_zoom_seg(self, pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, mono_id=False, interface_id = False, orientation_id = False, zoom_id = False, banner_id = False, presentation_id = False):

        """
        This function plots the particle positions and color codes each particle with its intrinsic
        activity at each location in space for a specific Pe_S=0 and Pe_F=500 system to track a domain
        in that system (with zoomed in inset) to show the cycle of avalanche/herding process

        Inputs:

        pos (partNum, 3): array of (x,y,z) positions for each particle

        px (partNum): x-orientation unit vector of active force

        py (partNum): y-orientation unit vector of active force

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        mono_id (default value = False): True/False value that specifies whether system should be treated
        as a monodisperse system (True) or binary system (False)

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        orientation_id ( default value = False): True/False value that specifies whether
        the average, binned orientation of particles should be plotted

        zoom_id ( default value = False ): True/False value that specifies whether the bulk of the cluster
        should be zoomed into for a close-up

        banner_id ( default value = False ): True/False value that specifies whether the system should be
        elongated along the x-axis when plotted

        presentation_id ( default value = False ): True/False value that specifies whether the formatting should
        be made for a PowerPoint presentation

        Outputs:
        .png file with each particle color-coded by intrinsic activity
        """

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        fastCol = '#a50f15'
        fastCol = '#b2182b'
        #fastCol = '#d6604d'
        slowCol = '#081d58'
        slowCol = '#2b8cbe'
        slowCol = '#2166ac'
        slowCol = '#4393c3'

        # Define y-dimension of figure
        if banner_id == True:
            y_dim = y_dim * (3/5)
        
        # Check if monodisperse or binary system
        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            
            mono_activity=self.peA
            mono_type = 2

            if mono_id == False:
                mono=0
            else:
                mono=1

        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')

        import math

        # Define function to rotate particles
        def rotate(x,y): 
            #rotate x,y around xo,yo by theta (rad)
            theta=0
            xr=math.cos(theta)*(x)-math.sin(theta)*(y)
            yr=math.sin(theta)*(x)+math.cos(theta)*(y)
            return xr,yr
        
        # Shift positions so domain doesn't move
        if (self.tst==279.3):
            pos[:,0]=pos[:,0]+45
        if (self.tst==274.8):
            pos[:,0]=pos[:,0]+38
        if (self.tst==282.9) | (self.tst==283.2) | (self.tst==283.5) | (self.tst==283.8):
            pos[:,0]=pos[:,0]+22
        if (self.tst==521.1):
            pos[:,0]=pos[:,0]+0
        if (self.tst==523.5):
            pos[:,0]=pos[:,0]+0
        if (round(self.tst,1)==106.2):
            pos[:,0]=pos[:,0]-75
        if (round(self.tst,1)==115.2):
            pos[:,0]=pos[:,0]+20
        if (round(self.tst,1)==127.2):
            pos[:,0]=pos[:,0]+63
        if (round(self.tst,1)>=124.8) & (round(self.tst,1)<=126):
            pos[:,0]=pos[:,0]+70

        # Enforce periodic boundary conditions
        test_id = np.where(pos[:,0]>self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]-self.lx_box
        test_id = np.where(pos[:,0]<-self.hx_box)[0]
        pos[test_id,0]=pos[test_id,0]+self.lx_box

        # Shift positions so domain doesn't move
        if (self.tst==273.9) | (self.tst==274.5) | (self.tst==274.8):
            pos[:,1]=pos[:,1]-20 #913
        #pos[:,1]=pos[:,1]-30 #930
        if (self.tst==279.3):
            pos[:,1]=pos[:,1]-30
        if (self.tst==282.9) | (self.tst==283.2) | (self.tst==283.5) | (self.tst==283.8):
            pos[:,1]=pos[:,1]-30
        if (self.tst==521.1):
            pos[:,1]=pos[:,1]-100
        if (self.tst==528):
            pos[:,1]=pos[:,1]-100
        if (round(self.tst,1)==106.2):
            pos[:,1]=pos[:,1]-50
        if (round(self.tst,1)==115.2):
            pos[:,1]=pos[:,1]-95
        if (round(self.tst,1)==127.2):
            pos[:,1]=pos[:,1]-55
        if (round(self.tst,1)>=124.8) & (round(self.tst,1)<=126):
            pos[:,1]=pos[:,1]-52
        
        # Enforce periodic boundary conditions
        test_id = np.where(pos[:,1]>self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]-self.ly_box
        test_id = np.where(pos[:,1]<-self.hy_box)[0]
        pos[test_id,1]=pos[test_id,1]+self.ly_box

        # Rotated positions of particles
        pos_x_rot, pos_y_rot = rotate(pos[:,0], pos[:,1])

        # Shift particle positions to proper box size
        pos[:,0] = pos_x_rot + self.hx_box
        pos[:,1] = pos_y_rot + self.hy_box

        # Define particle diameter
        sz = 0.755
        sz=0.77
        
        # Define inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        l, b, h, w = 0.4, 0.0, 0.6, 0.2
        ax6 = inset_axes(ax, loc=4, width=2.5, height=2.5)

        # Plot particles
        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=np.array([pos0[i,0], pos0[i,1]]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0], pos1[i,1]]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)

            # Place particles in main plot
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            
            #if (self.tst==274.8):
            #    sz = 1.0
            ells0_2 = [Ellipse(xy=np.array([pos0[i,0], pos0[i,1]]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1_2 = [Ellipse(xy=np.array([pos1[i,0], pos1[i,1]]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup_2 = mc.PatchCollection(ells0_2, facecolors=slowCol)
                fastGroup_2 = mc.PatchCollection(ells1_2,facecolors=fastCol)
            else:
                slowGroup_2 = mc.PatchCollection(ells1_2, facecolors=slowCol)
                fastGroup_2 = mc.PatchCollection(ells0_2,facecolors=fastCol)

            # Place particles in inset
            ax6.add_collection(slowGroup_2)
            ax6.add_collection(fastGroup_2)

            # Plot legend
            """
            if presentation_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
            
            elif banner_id == True:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=20), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=20)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=20, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                if self.peA <= self.peB:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]
                else:
                    fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peB)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA)), markersize=32)]
                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.05, 1.17], handlelength=1.5, columnspacing=0.4, fontsize=36, ncol=2, facecolor='none', edgecolor='none')
                ax.add_artist(one_leg)
            """

        # Plot inner and outer interfaces
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        
                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        
                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
                        
            except:
                pass
        
        # Plot average active force orientations
        if orientation_id == True:
            try:
                if active_fa_dict!=None:
                    plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
            except:
                pass

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            ax.set_xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            ax.set_ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            ax.set_ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            ax.set_xlim(0.0, self.lx_box)
        # Plot entire system            
        else:

            if zoom_id == True:
                ax.set_ylim(self.hy_box-25-2, self.hy_box+25+2)
                ax.set_xlim(self.hy_box-25-2, self.hy_box+25+2)

                
            elif banner_id == True: 
                ax.set_ylim(1.5*self.ly_box/5, 3.5*self.ly_box/5)
                ax.set_xlim(0, self.lx_box)
            else:
                ax.set_ylim(0, self.ly_box)
                ax.set_xlim(0, self.lx_box)
                ax.set_ylim(self.hy_box-128, self.hy_box+122)
                ax.set_xlim(self.hx_box-122, self.hx_box+128)
        
        # Label simulation time
        if banner_id == False:
            if self.lx_box == self.ly_box:
                ax.text(0.03, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=30, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            elif self.lx_box > self.ly_box:
                ax.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.3f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                    fontsize=18, transform = ax.transAxes,
                    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        # Remove main plot axes labels
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Remove inset axes labels
        ax6.axes.set_xticks([])
        ax6.axes.set_yticks([])
        ax6.axes.set_xticklabels([])
        ax6.axes.set_yticks([])
        ax6.set_aspect('equal')

        # Plot particle orientations
        ax6.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px[typ1ind], py[typ1ind], color='black', width=1.0, minlength=0.0, headwidth=2000,headlength=900, headaxislength=900, scale=70)

        # Define simulation and inset box linewidth
        for axis in ['top', 'bottom', 'left', 'right']:

            ax6.spines[axis].set_linewidth(1.5)

        for axis in ['top', 'bottom', 'left', 'right']:

            ax.spines[axis].set_linewidth(2.0)

        # Define inset axes limits
        ax6.set_ylim(self.hy_box-128, self.hy_box+122)
        ax6.set_xlim(self.hx_box-122, self.hx_box+128)

        # Define inset location based on time step
        if self.tst==285:
            # 950
            x_min = self.hx_box-20
            x_max = self.hx_box
            y_min = self.hy_box-20
            y_max = self.hy_box
        elif (self.tst==273.9) | (self.tst==274.5):
            # 913
            x_min = self.hx_box-10
            x_max = self.hx_box+10
            y_min = self.hy_box-10
            y_max = self.hy_box+10
        if (self.tst==274.8):
            x_min = self.hx_box-37
            x_max = self.hx_box-15
            y_min = self.hy_box-10
            y_max = self.hy_box+12
        if (self.tst==279.3):
            x_min = self.hx_box-37
            x_max = self.hx_box-15
            y_min = self.hy_box-10
            y_max = self.hy_box+12
        if (self.tst==282.9) | (self.tst==283.2) | (self.tst==283.5) | (self.tst==283.8):
            x_min = self.hx_box-37
            x_max = self.hx_box-15
            y_min = self.hy_box-10
            y_max = self.hy_box+12
        if (self.tst==521.1):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
        if (self.tst==528):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
        if (round(self.tst,1)==127.2):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
            x_min = 0
            x_max = self.lx_box
            y_min = 0
            y_max = self.ly_box
        if (round(self.tst,1)==106.2):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
            x_min = 0
            x_max = self.lx_box
            y_min = 0
            y_max = self.ly_box
        if (round(self.tst,1)==115.2):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
            x_min = 0
            x_max = self.lx_box
            y_min = 0
            y_max = self.ly_box
        if (round(self.tst,1)>=124.8) & (round(self.tst,1)<=126):
            x_min = self.hx_box-25
            x_max = self.hx_box+25
            y_min = self.hy_box-25
            y_max = self.hy_box+25
            x_min = 0
            x_max = self.lx_box
            y_min = 0
            y_max = self.ly_box
        
        # Define inset axes limits
        ax6.set_ylim(y_min, y_max)
        ax6.set_xlim(x_min, x_max)

        # Plot location of inset on main plot
        ax.plot([x_min, x_max], [y_min, y_min], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_max, x_max], [y_min, y_max], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_min, x_max], [y_max, y_max], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_min, x_min], [y_min, y_max], linestyle='dashed', linewidth=1.5, color='black')
        
        # Connect corners of inset to location of main plot
        ax.plot([x_max, self.lx_box-4], [y_max, self.hy_box-8.8], linestyle='dashed', linewidth=1.5, color='black')
        ax.plot([x_min, self.hx_box+8.8], [y_min, 3], linestyle='dashed', linewidth=1.5, color='black')
        
        # Create frame images
        #ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4') .  # For website

        # Save image
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=900, transparent=False, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_com_plotted(self, pos, phase_ids_dict, sep_surface_dict=None, int_comp_dict=None, com_opt=None):
        #print(com_opt)
        
        
        bulk_part_ids = phase_ids_dict['bulk']['all']
        gas_part_ids = phase_ids_dict['gas']['all']
        int_part_ids = phase_ids_dict['int']['all']
        dense_part_ids = np.append(bulk_part_ids, int_part_ids)

        

        #com_opt = {'x': [np.mean(pos[bulk_part_ids,0])]+self.hx_box, 'y': [np.mean(pos[bulk_part_ids,1]+self.hy_box)]}
        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        #slowCol = '#e31a1c'
        slowCol = '#081d58'

        

        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            mono=1
            mono_activity=self.peA
            mono_type = 2
            mono=0
            #mono=1
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        sz = 0.85#9
        """
        pos0_x_arr = np.append(pos[typ0ind,0]+self.hx_box, pos[typ0ind,0]+self.hx_box)
        pos0_x_arr = np.append(pos0_x_arr, pos[typ0ind,0]+self.hx_box)

        pos0_y_arr = np.append(pos[typ0ind,1]+self.hy_box, pos[typ0ind,1]+self.hy_box+self.ly_box)
        pos0_y_arr = np.append(pos0_y_arr, pos[typ0ind,1]+self.hy_box-self.ly_box)

        pos1_x_arr = np.append(pos[typ1ind,0]+self.hx_box, pos[typ1ind,0]+self.hx_box)
        pos1_x_arr = np.append(pos1_x_arr, pos[typ1ind,0]+self.hx_box)

        pos1_y_arr = np.append(pos[typ1ind,1]+self.hy_box, pos[typ1ind,1]+self.hy_box+self.ly_box)
        pos1_y_arr = np.append(pos1_y_arr, pos[typ1ind,1]+self.hy_box-self.ly_box)
        """

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]
            """
            print(len(pos0))
            len_slow = int(len(pos)/2)
            import random

            rows_id = random.sample(range(0, 
                              pos.shape[0]-1), len_slow)

            pos1_id = np.array([], dtype=int)
            for i in range(0, len(pos)):
                if i not in rows_id:

                    pos1_id = np.append(pos1_id, int(i))
            pos0 = pos[rows_id]
            pos1 = pos[pos1_id]
            """


            #Assign type 0 particles to plot
            """
            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(typ1ind))]
            """

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            """
            if self.peA != self.peB:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=36, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Active', markersize=28)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=32, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            """

            #fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=36)]

            #one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
            #one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.97, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
            #ax.add_artist(one_leg)

            #Create legend for binary system
            """
            if (len(typ0ind)!=0) & (len(typ1ind)!=0):
                leg = ax.legend(handles=[ells0[0], ells1[0]], labels=[r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB))], loc='upper right', prop={'size': 22}, markerscale=8.0, edgecolor='black', linewidth=2.0)
                if self.peA <= self.peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA))], loc='upper right', prop={'size': 22}, markerscale=8.0, edgecolor='black', linewidth=2.0)
                leg.legendHandles[0].set_color(slowCol)
            """

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                #fast_leg.append(Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe} = $'+str(int(self.peB)), markersize=25))

                #leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 24}, markerscale=8.0, edgecolor='black')
                #leg = ax.legend(handles=[ells0[0]], labels=[r'$F^\mathrm{a} = $'+str(int(self.peB))], loc='upper right', prop={'size': 28}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)
                #one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.2, handletextpad=0.0, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=25, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                #ax.add_artist(one_leg)

                #leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 22}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)
        """
        try:
            if sep_surface_dict!=None:
                for m in range(0, len(sep_surface_dict)):
                    key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

                    try:
                        pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                        pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                        plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                    except:
                        pass

                    try:
                        pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                        pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                        plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                    except:
                        pass
        except:
            pass
        """
        try:
            if active_fa_dict!=None:
                plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
        except:
            pass
        #Label time step
        #ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
        #        horizontalalignment='right', verticalalignment='bottom',
        #        transform=ax.transAxes,
        #        fontsize=18,
        #        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)
            #plt.ylim(self.hy_box-30-2, self.hy_box+30-2)
            #plt.xlim(self.hy_box-30-2, self.hy_box+30-2)


        # Label simulation time
        
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.66, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        #Assign type 0 particles to plot
        plt.scatter(com_opt['x'][0], com_opt['y'][0], facecolor='black', edgecolor='black', s=300, marker='o')
        #plt.scatter(com_opt['x'][0:3], com_opt['y'][0:3], edgecolor='black', facecolor='None', linewidth=3, s=200, marker='o')
        #plt.plot(com_opt['x'][0:2], com_opt['y'][0:2], color='black', linewidth=5, linestyle='solid')

        # Create frame pad for images
        #pad = str(j).zfill(4)
        ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4')
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()  

    def plot_part_activity_empower_article(self, pos, ang, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None):

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        slowCol = '#081d58'
        #fastCol = '#081d58'

        

        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            mono=1
            mono_activity=self.peA
            mono_type = 2
            mono=0
            #fastCol = '#081d58'
            #slowCol = '#e31a1c'
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim), facecolor='white')
        ax = fig.add_subplot(111)

        sz = 0.77#9
        #sz = 0.9
        """
        pos0_x_arr = np.append(pos[typ0ind,0]+self.hx_box, pos[typ0ind,0]+self.hx_box)
        pos0_x_arr = np.append(pos0_x_arr, pos[typ0ind,0]+self.hx_box)

        pos0_y_arr = np.append(pos[typ0ind,1]+self.hy_box, pos[typ0ind,1]+self.hy_box+self.ly_box)
        pos0_y_arr = np.append(pos0_y_arr, pos[typ0ind,1]+self.hy_box-self.ly_box)

        pos1_x_arr = np.append(pos[typ1ind,0]+self.hx_box, pos[typ1ind,0]+self.hx_box)
        pos1_x_arr = np.append(pos1_x_arr, pos[typ1ind,0]+self.hx_box)

        pos1_y_arr = np.append(pos[typ1ind,1]+self.hy_box, pos[typ1ind,1]+self.hy_box+self.ly_box)
        pos1_y_arr = np.append(pos1_y_arr, pos[typ1ind,1]+self.hy_box-self.ly_box)
        """

        if mono==0:

            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot
            """
            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(typ1ind))]
            """

            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)
            """
            if self.peA != self.peB:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label=r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), markersize=32), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label=r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB)), markersize=32)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, labelspacing=0.4, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=36, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            else:
                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Active', markersize=28)]

                one_leg = ax.legend(handles=fast_leg, loc='upper right', borderpad=0.3, handletextpad=0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.0], handlelength=1.5, columnspacing=0.0, fontsize=32, ncol=1, fancybox=True, framealpha=0.5, facecolor='white', edgecolor='black')
                ax.add_artist(one_leg)
            """

            fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Slow', markersize=36), Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=fastCol, label='Fast', markersize=36)]

            #one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
            one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.83, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
            ax.add_artist(one_leg)

            #Create legend for binary system
            """
            if (len(typ0ind)!=0) & (len(typ1ind)!=0):
                leg = ax.legend(handles=[ells0[0], ells1[0]], labels=[r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peB))], loc='upper right', prop={'size': 22}, markerscale=8.0, edgecolor='black', linewidth=2.0)
                if self.peA <= self.peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe}_\mathrm{S} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{F} = $'+str(int(self.peA))], loc='upper right', prop={'size': 22}, markerscale=8.0, edgecolor='black', linewidth=2.0)
                leg.legendHandles[0].set_color(slowCol)
            """

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box, px, py, color='black', width=0.003)
            #plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box, px, py, color='black', width=0.003)

        elif mono == 1:

            #if (self.peA==0):
            #    slowCol = '#081d58'
            #else:
            #    slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                fast_leg = []

                fast_leg = [Line2D([0], [0], lw=0, marker='o', markeredgewidth=1.8*1.2, markeredgecolor='None', markerfacecolor=slowCol, label='Active', markersize=36)]

                #one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[1.0, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                one_leg = ax.legend(handles=fast_leg, borderpad=0.3, labelspacing=0.4, handletextpad=-0.2, bbox_transform=ax.transAxes, bbox_to_anchor=[0.7, 1.15], handlelength=1.5, columnspacing=1.0, fontsize=36, ncol=2, facecolor='None', edgecolor='None')
                ax.add_artist(one_leg)
        """
        print(int_comp_dict['ids'])
        try:
            if sep_surface_dict!=None:
                
                for m in range(0, len(sep_surface_dict)):
                    key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                    print(key)

                    try:
                        pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                        print(pos_interior_surface_x)
                        pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                        plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                    except:
                        pass

                    try:
                        pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                        pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                        plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                    except:
                        pass
        except:
            pass
        """
        try:
            if active_fa_dict!=None:
                plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
        except:
            pass
        #Label time step
        #ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
        #        horizontalalignment='right', verticalalignment='bottom',
        #        transform=ax.transAxes,
        #        fontsize=18,
        #        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)
            #plt.ylim(self.hy_box-30-2, self.hy_box+30-2)
            #plt.xlim(self.hy_box-30-2, self.hy_box+30-2)


        # Label simulation time
        """
        if self.lx_box == self.ly_box:
            #plt.text(0.69, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            #    fontsize=24, transform = ax.transAxes,
            #    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
            plt.text(0.66, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=30, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        """
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame pad for images
        #pad = str(j).zfill(4)
        ax.set_facecolor('white')
        #ax.set_facecolor('#F4F4F4')
        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=300, transparent=False, bbox_inches='tight')
        #plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".eps", format='eps', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_part_activity2(self, pos, prev_pos, px, py, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None):

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]
        
        # Calculate interparticle distance between type A reference particle and type A neighbor
        difx, dify, difr = self.utility_functs.sep_dist_arr(pos[typ0ind], prev_pos[typ0ind], difxy=True)
        
        mean_dify = np.mean(dify)
        mean_dify = 0
        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (1.0 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)
            dense_x_width = 42.23 * 1.5
            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))
            y_dim = y_dim * 3

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        #Set plot colors
        fastCol = '#e31a1c'
        slowCol = '#081d58'

        pos0_x_arr = np.append(pos[typ0ind,0]+self.hx_box, pos[typ0ind,0]+self.hx_box)
        pos0_x_arr = np.append(pos0_x_arr, pos[typ0ind,0]+self.hx_box)

        pos0_y_arr = np.append(pos[typ0ind,1]+self.hy_box-mean_dify, pos[typ0ind,1]+self.hy_box+self.ly_box-mean_dify)
        pos0_y_arr = np.append(pos0_y_arr, pos[typ0ind,1]+self.hy_box-self.ly_box-mean_dify)

        pos1_x_arr = np.append(pos[typ1ind,0]+self.hx_box, pos[typ1ind,0]+self.hx_box)
        pos1_x_arr = np.append(pos1_x_arr, pos[typ1ind,0]+self.hx_box)

        pos1_y_arr = np.append(pos[typ1ind,1]+self.hy_box-mean_dify, pos[typ1ind,1]+self.hy_box+self.ly_box-mean_dify)
        pos1_y_arr = np.append(pos1_y_arr, pos[typ1ind,1]+self.hy_box-self.ly_box-mean_dify)

        if (len(typ1ind)==0):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (len(typ0ind)==0):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            mono=1
            mono_activity=self.peA
            mono_type = 2
        else:
            mono=0

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        sz = 0.75
              
        if mono==0:
            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot
            
            ells0 = [Ellipse(xy=np.array([pos0_x_arr[i], pos0_y_arr[i]]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(pos0_x_arr))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1_x_arr[i], pos1_y_arr[i]]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(pos1_x_arr))]

            # Plot position colored by neighbor number
            if self.peA <= self.peB:
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells1,facecolors=fastCol)
            else:
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                fastGroup = mc.PatchCollection(ells0,facecolors=fastCol)
            ax.add_collection(slowGroup)
            ax.add_collection(fastGroup)

            #Create legend for binary system
            if (len(typ0ind)!=0) & (len(typ1ind)!=0):
                leg = ax.legend(handles=[ells0[0], ells1[0]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{B} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                if self.peA <= self.peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-mean_dify, px, py, color='black', width=0.003)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box+self.ly_box-mean_dify, px, py, color='black', width=0.003)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box-self.ly_box-mean_dify, px, py, color='black', width=0.003)

        elif mono == 1:

            if (self.peA==0):
                slowCol = '#081d58'
            else:
                slowCol = '#e31a1c'

            if mono_type == 0:

                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 1:

                #Local each particle's positions
                pos1=pos[typ1ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(slowGroup)

                leg = ax.legend(handles=[ells1[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

            elif mono_type == 2:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles
                pos1=pos[typ1ind]

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                #leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                #leg = ax.legend(handles=[ells0[0]], labels=[r'$F^\mathrm{a} = $'+str(int(self.peB))], loc='upper right', prop={'size': 28}, markerscale=8.0)
                #leg.legendHandles[0].set_color(slowCol)
        """
        try:
            if sep_surface_dict!=None:
                for m in range(0, len(sep_surface_dict)):
                    key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

                    try:
                        pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                        pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                        plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                    except:
                        pass

                    try:
                        pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                        pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                        plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                    except:
                        pass
        except:
            pass
        """
        try:
            if active_fa_dict!=None:
                plt.quiver(self.pos_x, self.pos_y, active_fa_dict['bin']['x'], active_fa_dict['bin']['y'], scale=20.0, color='black', alpha=0.8)
        except:
            pass
        #Label time step
        #ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
        #        horizontalalignment='right', verticalalignment='bottom',
        #        transform=ax.transAxes,
        #        fontsize=18,
        #        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #Set axes parameters
        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(-(0.5*dense_x_width)+self.hx_box, (0.5*dense_x_width)+self.hx_box)
            plt.ylim(0.0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width), dense_y_mid+(dense_y_width))
            plt.xlim(0.0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.8, 0.04, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.4f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame pad for images
        #pad = str(j).zfill(4)

        plt.tight_layout()
        plt.savefig(self.outPath + 'part_activity_' + self.outFile + ".png", dpi=75, transparent=False)
        plt.close()   
        
        
    def ang_vel_histogram(self, ang_vel, phasePart):

        xmin = np.min(ang_vel)
        xmax = np.max(ang_vel)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

        bulk_ang_vel = ang_vel[np.where(phasePart==0)[0]]
        int_ang_vel = ang_vel[np.where(phasePart==1)[0]]
        gas_ang_vel = ang_vel[np.where(phasePart==2)[0]]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Remove bulk particles that are outside plot's xrange
        if (len(bulk_ang_vel)>0):
            bulk_id = np.where((bulk_ang_vel > xmax) | (bulk_ang_vel < xmin))[0]
            bulk_ang_vel = np.delete(bulk_ang_vel, bulk_id)

            plt.hist(bulk_ang_vel, alpha = 1.0, bins=60, color=green)

        #If interface particle measured, continue
        if (len(int_ang_vel)>0):
            int_id = np.where((int_ang_vel > xmax) | (int_ang_vel < xmin))[0]
            int_ang_vel = np.delete(int_ang_vel, int_id)

            plt.hist(int_ang_vel, alpha = 0.8, bins=75, color=yellow)

        if (len(gas_ang_vel)>0):
            gas_id = np.where((gas_ang_vel > xmax) | (gas_ang_vel < xmin))[0]
            gas_ang_vel = np.delete(gas_ang_vel, gas_id)

            plt.hist(gas_ang_vel, alpha = 0.8, bins=75, color=red)

        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        red_patch = mpatches.Patch(color=red, label='Gas')
        plt.legend(handles=[green_patch, yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'angular velocity ($\phi$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        #plt.close()

    def ang_vel_bulk_sf_histogram(self, ang_vel, phasePart):

        xmin = np.min(ang_vel)
        xmax = np.max(ang_vel)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

        fastCol = '#e31a1c'
        slowCol = '#081d58'

        bulk_A_ang_vel = ang_vel[np.where((phasePart==0) & (self.typ==0))[0]]
        bulk_B_ang_vel = ang_vel[np.where((phasePart==0) & (self.typ==1))[0]]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        if (len(bulk_A_ang_vel)>0):
            plt.hist(bulk_A_ang_vel, alpha = 1.0, bins=60, color=slowCol)
        if (len(bulk_B_ang_vel)>0):
            plt.hist(bulk_B_ang_vel, alpha = 0.5, bins=60, color=fastCol)

        green_patch = mpatches.Patch(color=slowCol, label='A Bulk')
        yellow_patch = mpatches.Patch(color=fastCol, label='B bulk')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'angular velocity ($\phi$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        #plt.close()
    def ang_vel_int_sf_histogram(self, ang_vel, phasePart):

        xmin = np.min(ang_vel)
        xmax = np.max(ang_vel)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

        fastCol = '#e31a1c'
        slowCol = '#081d58'

        int_A_ang_vel = ang_vel[np.where((phasePart==1) & (self.typ==0))[0]]
        int_B_ang_vel = ang_vel[np.where((phasePart==1) & (self.typ==1))[0]]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        if (len(int_A_ang_vel)>0):
            plt.hist(int_A_ang_vel, alpha = 1.0, bins=60, color=slowCol)
        if (len(int_B_ang_vel)>0):
            plt.hist(int_B_ang_vel, alpha = 0.5, bins=60, color=fastCol)

        green_patch = mpatches.Patch(color=slowCol, label='A Bulk')
        yellow_patch = mpatches.Patch(color=fastCol, label='B bulk')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'angular velocity ($\phi$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        #plt.close()

    def ang_vel_gas_sf_histogram(self, ang_vel, phasePart):

        xmin = np.min(ang_vel)
        xmax = np.max(ang_vel)

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

        fastCol = '#e31a1c'
        slowCol = '#081d58'

        gas_A_ang_vel = ang_vel[np.where((phasePart==2) & (self.typ==0))[0]]
        gas_B_ang_vel = ang_vel[np.where((phasePart==2) & (self.typ==1))[0]]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        if (len(gas_A_ang_vel)>0):
            plt.hist(gas_A_ang_vel, alpha = 1.0, bins=60, color=slowCol)
        if (len(gas_B_ang_vel)>0):
            plt.hist(gas_B_ang_vel, alpha = 0.5, bins=60, color=fastCol)

        green_patch = mpatches.Patch(color=slowCol, label='A Bulk')
        yellow_patch = mpatches.Patch(color=fastCol, label='B bulk')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'angular velocity ($\phi$)', fontsize=18)
        plt.ylabel('Number of particles', fontsize=18)
        plt.xlim([xmin,xmax])

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(0.03, 0.94, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
            fontsize=18,transform = ax.transAxes,
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        #plt.close()

    def plot_vorticity(self, velocity, vorticity, phase_dict, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None, species='all', interface_id = False):
        #DONE!
        """
        This function bins each particle's position and plots the average
        vorticity within each bin.

        Inputs:

        velocity: dictionary of each particle types velocity at each binned
        location.
        
        vorticity: dictionary of each particle types vorticity at each binned
        location

        phase_dict: dictionary (output from phase_ident() in
        phase_identification.py) that contains the id of each bin specifying
        the phase of bulk (0), interface (1), or gas (2).

        sep_surface_dict (default value = None): dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict (default value = None): dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        active_fa_dict (default value = None): dictionary (output from bin_active_fa() in binning.py)
        that contains information on the binned average active force magnitude and orientation over
        space

        species (default value = all): String specifying the particle species' ('all', 'A', or 'B') 
        binned vorticity to plot

        interface_id ( default value = False): True/False value that specifies whether
        the interface interior and exterior surfaces should be plotted

        Outputs:
        .png file with each bin's vorticity plotted
        """

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(pos[:,0]+self.hx_box)

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = (area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling/ (dense_x_width / self.ly_box))

        # If box is rectangular with long dimension of y-axis
        elif self.lx_box < self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (y)
            mid_point = np.mean(pos[:,1]+self.hy_box)

            # estimated shorted dimension length of dense phase (x)
            dense_x_width = (area_dense / self.lx_box)

            # Set maximum dimension length (y) of simulation box to be 13 inches
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int((scaling/(dense_x_width / self.lx_box)) + 1.0)
            y_dim = int(scaling)

        # If box is square
        else:

            # Minimum dimension length (in inches)
            scaling =7.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        

        
        # Define color bar limits and ticks
        min_n_temp = np.abs(np.min(vorticity['all']))
        max_n_temp = np.abs(np.max(vorticity['all']))

        if min_n_temp > max_n_temp:
            max_n = min_n_temp
            min_n = -min_n_temp
        else:
            min_n = -max_n_temp
            max_n = max_n_temp

        levels_text=20
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        
        # Plot vorticity of given species
        im = plt.contourf(self.pos_x, self.pos_y, vorticity[species], level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        
        # Color bar parameters
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])

        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        
        # Plot colorbar
        clb = plt.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        
        clb.ax.tick_params(labelsize=16)

        if species == 'all':
            clb.set_label(r'$\nabla \times v$', labelpad=-40, y=1.07, rotation=0, fontsize=20)
        elif species == 'A':
            clb.set_label(r'$\nabla \times v_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)
        elif species == 'B':
            clb.set_label(r'$\nabla \times v_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        # Remove ticks
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        # Plot interface surfaces
        if interface_id == True:
            try:

                if sep_surface_dict!=None:
                    
                    for m in range(0, len(sep_surface_dict)):
                        key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                        print(key)

                        try:
                            pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                            pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x + self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x - self.lx_box, pos_interior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_interior_surface_x, pos_interior_surface_y-self.ly_box, c='black', s=3.0)
                        except:
                            pass

                        try:
                            
                            pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                            pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x + self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x - self.lx_box, pos_exterior_surface_y, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y+self.ly_box, c='black', s=3.0)
                            plt.scatter(pos_exterior_surface_x, pos_exterior_surface_y-self.ly_box, c='black', s=3.0) 
                        except:
                            pass
            except:
                pass

        # If rectangular box, reduce system size plotted
        if self.lx_box > self.ly_box:
            plt.xlim(dense_x_mid-(dense_x_width/2), dense_x_mid+(dense_x_width/2))
            plt.ylim(0, self.ly_box)
        elif self.lx_box < self.ly_box:
            plt.ylim(dense_y_mid-(dense_y_width/2), dense_y_mid+(dense_y_width/2))
            plt.xlim(0, self.lx_box)
        # Plot entire system
        else:
            plt.ylim(0, self.ly_box)
            plt.xlim(0, self.lx_box)

        # Plot binned velocity of given species
        for ix in range(0, len(self.pos_x)):
            for iy in range(0, len(self.pos_y)):
                if phase_dict['bin'][ix][iy] == 0:
                    plt.quiver(self.pos_x[ix][iy], self.pos_y[ix][iy], velocity[species]['x'][ix][iy], velocity[species]['y'][ix][iy], scale=250, color='black', width=0.002, alpha=0.8)

        ax.axis('off')

        plt.tight_layout()


        plt.savefig(self.outPath + 'vorticity_' + species + '_' + self.outFile + ".png", dpi=200, transparent=False, bbox_inches='tight')
        plt.close()  

    def plot_voronoi(self, pos):
        import freud
        from freud import box
        f_box = box.Box(Lx=self.lx_box, Ly=self.ly_box, is2D=True)
        voro = freud.locality.Voronoi()
        voro.compute((f_box, pos)).polytopes

        plt.figure()
        ax = plt.gca()
        voro.plot(ax=ax, cmap="RdBu")
        plt.xlim([-50, 50])
        #ax.scatter(pos[:, 0], pos[:, 1], s=2, c="k")
        plt.show()

        stop
