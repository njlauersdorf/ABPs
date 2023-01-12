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
    def __init__(self, orient_dict, pos_dict, lx_box, ly_box, NBins_x, NBins_y, sizeBin_x, sizeBin_y, peA, peB, parFrac, eps, typ, tst, partNum, outPath, outFile):

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

        # Binning properties
        try:
            self.NBins_x = int(NBins_x)                                             # Number of bins in x-dimension
            self.NBins_y = int(NBins_y)                                             # Number of bins in y-dimension
        except:
            print('NBins must be either a float or an integer')

        self.sizeBin_x = utility_functs.roundUp((self.lx_box / self.NBins_x), 6)    # y-length of bin (simulation units)
        self.sizeBin_y = utility_functs.roundUp((self.ly_box / self.NBins_y), 6)    # x-length of bin (simulation units)



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

    def plot_phases(self, pos, phase_ids_dict, sep_surface_dict, int_comp_dict):
        """
        This function plots each particle's position and color-codes each
        particle by the phase it is a part of, i.e. bulk=green,
        interface=purple, and gas=red.

        Inputs:
        pos: array (partNum, 3) of each particles x,y,z positions
        phase_ids_dict: dictionary (output from various phase_identification
        functions) containing information on the composition of each phase

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        Outputs:
        .png file with each particle's position plotted and color coded by the
        phase it's a part of and the interface surfaces plotted in black
        """

        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

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
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Set plotted particle size
        sz = 0.75

        # Plot position colored by neighbor number
        
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

        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)

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




        pos_x_ticks = np.linspace(0, self.NBins_x, self.NBins_x + 1) * self.sizeBin_x
        pos_y_ticks = np.linspace(0, self.NBins_y, self.NBins_y + 1) * self.sizeBin_y

        plt.xticks(pos_x_ticks)
        plt.yticks(pos_y_ticks)

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

        #plt.ylim((0, self.l_box))
        #plt.xlim((0, self.l_box))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
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

        red_patch = mpatches.Patch(color=red, label='Dilute')
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        #purple_patch = mpatches.Patch(color=purple, label='Bubble')
        plt.legend(handles=[green_patch, yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=16, loc='upper left',labelspacing=0.1, handletextpad=0.1)
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'plot_phases_' + out + pad + ".png", dpi=100)
        #plt.close()
    def plot_area_fraction(self, area_frac_dict, sep_surface_dict, int_comp_dict, pos, type='all'):#, int_comp_dict):#sep_surface_dict, int_comp_dict):
        """
        This function plots the binned average area fraction at each location
        in space.

        Inputs:
        num_dens_dict: dictionary (output from various bin_area_frac() in
        binning.py) containing information on the average area fraction of all,
        type A, type B, and the difference of type B and type A particles  in
        addition to the fraction of type A particles

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

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
            x_dim = int(scaling + 1.0)
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

        clb.ax.tick_params(labelsize=16)
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{A}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{A}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}-\phi_\mathrm{A}$', labelpad=-30, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\phi_\mathrm{B}-\phi_\mathrm{A}$', labelpad=-25, y=1.15, rotation=0, fontsize=20)
                
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


        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
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
        plt.show()
        #plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_normal_fa_map(self, normal_fa_dict, sep_surface_dict, int_comp_dict):

        num_dens_B = normal_fa_dict['bin']['all']
        print(normal_fa_dict)
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.0
        max_n = np.max(num_dens_B)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, num_dens_B, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

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

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\phi_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_particle_fraction(self, num_dens_dict, sep_surface_dict, int_comp_dict, pos, type='A'):
        """
        This function plots the fraction of particles of a given type per bin
        at each location in space.

        Inputs:
        num_dens_dict: dictionary (output from various bin_area_frac() in
        binning.py) containing information on the average area fraction of all,
        type A, type B, and the difference of type B and type A particles in
        addition to the fraction of type A particles

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

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
            x_dim = int(scaling + 1.0)
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
        clb.ax.tick_params(labelsize=16)

        if type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{A}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{A}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}-\chi_\mathrm{A}$', labelpad=-30, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\chi_\mathrm{B}-\chi_\mathrm{A}$', labelpad=-25, y=1.15, rotation=0, fontsize=20)

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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
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
        plt.show()
        #plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_alignment(self, align_dict, sep_surface_dict, int_comp_dict, pos, type='all'):

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

        type (optional): string specifying whether the area fraction of all
        (type='all'), type A (type='A'), type B (type='B'), or the difference
        of types B and A ('dif') should be plotted.

        Outputs:
        .png file with binned average alignment of respective type at each
        location in space plotted as a heat map with color bar and a quiver plot
        showing the average direction (magnitude normalized) of orientation per
        bin.
        """

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
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Contour plot properties
        if type == 'dif':
            min_align = -np.amax(align_dif)
            max_align = np.amax(align_dif)
        else:
            min_align = 0.0
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
        clb.ax.tick_params(labelsize=16)
        print('test')
        print(type)
        if type == 'all':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'A':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'B':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=-40, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=-30, y=1.15, rotation=0, fontsize=20)
        elif type == 'dif':
            if self.lx_box == self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=-30, y=1.05, rotation=0, fontsize=20)
            elif self.lx_box > self.ly_box:
                clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=-25, y=1.15, rotation=0, fontsize=20)

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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
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
        plt.show()
        #plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        #plt.close()

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
        plt.show()
        #plt.savefig(outPath + 'lat_histo_' + out + pad + ".png", dpi=150)
        #plt.close()
    def lat_map(self, lat_plot_dict, sep_surface_dict, int_comp_dict, velocity_dict=None):
        """
        This function plots the lattice spacings of all dense phase particles
        at each location in space.

        Inputs:
        lat_plot_dict: dictionary (output from lattice_spacing() in
        measurement.py) containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface particle.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        velocity_dict (optional): dictionary (output from () in .py) that
        contains average velocity per bin of all, type A, and type B particles

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
            x_dim = int(scaling + 1.0)
            y_dim = int(scaling)

        # Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
        fig = plt.figure(figsize=(x_dim,y_dim))
        ax = fig.add_subplot(111)

        # Find min/max orientation
        dense_lats = lat_plot_dict['dense']['all']
        min_lat = 0.85*bulk_lat_mean
        max_lat = 1.15*bulk_lat_mean

        # Set plotted particle size
        sz = 0.75
        
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
        clb.ax.tick_params(labelsize=16)

        #clb.set_label(r'$g_6(r)$', labelpad=-55, y=1.04, rotation=0, fontsize=18)

        clb.set_label(r'$a$', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)

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

        plt.text(0.8, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        plt.text(0.8, 0.92, s=r'$\overline{a}$' + ' = ' + '{:.3f}'.format(bulk_lat_mean),
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
        plt.show()
        stop
        #plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=100)
        #plt.close()
    def plot_general_rdf(self, radial_df_dict):

        fsize=10

        rdf_allall = np.array(radial_df_dict['all-all'])
        rdf_aa = np.array(radial_df_dict['A-A'])
        rdf_ab = np.array(radial_df_dict['A-B'])
        rdf_bb = np.array(radial_df_dict['B-B'])

        fig, ax1 = plt.subplots(figsize=(12,6))

        plot_max = 1.05 * np.max(radial_df_dict['all-all'])

        plot_min = -1.0

        rstop = 10.

        step = int(np.abs(plot_max - plot_min)/6)

        x_arr = np.array([0.0,15.0])
        y_arr = np.array([1.0, 1.0])

        plt.plot(x_arr, y_arr, c='black', lw=1.0, ls='--')

        plt.plot(radial_df_dict['r'], radial_df_dict['all-all'],
                    c='black', lw=9.0, ls='-', alpha=1, label='All-All')


        ax1.set_ylim(plot_min, plot_max)
        #ax1.set_ylim(0, 2)


        ax1.set_xlabel(r'Separation Distance ($r$)', fontsize=fsize*2.8)

        ax1.set_ylabel(r'$g(r)$', fontsize=fsize*2.8)

        #lat_theory = np.mean(lat_theory_arr)
        # Set all the x ticks for radial plots

        x_tick_val = radial_df_dict['r'][np.where(rdf_allall==np.max(rdf_allall))[0][0]]
        loc = ticker.MultipleLocator(base=(x_tick_val*2))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(x_tick_val))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)
        plt.legend(loc='upper right', fontsize=fsize*2.6)
        #step = 2.0
        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        # Left middle plot
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)
        #plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'rdf_all_' + out + ".png", dpi=300)
        #plt.close()

    def plot_all_rdfs(self, radial_df_dict):

        fsize=10

        rdf_allall = np.array(radial_df_dict['all-all'])
        rdf_aa = np.array(radial_df_dict['A-A'])
        rdf_ab = np.array(radial_df_dict['A-B'])
        rdf_bb = np.array(radial_df_dict['B-B'])

        #all_bulk_max = np.max(rdf_all_bulk_rdf)
        AA_bulk_max = np.max(rdf_aa)
        AB_bulk_max = np.max(rdf_ab)
        #BA_bulk_max = np.max(rdf_BA_bulk_rdf)
        BB_bulk_max = np.max(rdf_bb)
        #all_bulk_max = np.max(rdf_all_bulk_rdf)
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

        if AA_bulk_order == 1:
            plot_max = 1.05 * AA_bulk_max
        elif AB_bulk_order == 1:
            plot_max = 1.05 * AB_bulk_max
        elif BB_bulk_order == 1:
            plot_max = 1.05 * BB_bulk_max

        plot_min = -1.0

        rstop=10.0

        step = int(np.abs(plot_max - plot_min)/6)
        step_x = 1.5
        fastCol = '#e31a1c'
        slowCol = '#081d58'
        purple = ("#00441b")
        orange = ("#ff7f00")
        green = ("#377EB8")

        x_arr = np.array([0.0,10.0])
        y_arr = np.array([1.0, 1.0])

        fig, ax1 = plt.subplots(figsize=(12,6))

        first_width = 15.0
        second_width = 8.0
        third_width = 2.5
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

        plt.plot(x_arr, y_arr, c='black', lw=1.5, ls='--')

        ax1.set_ylim(plot_min, plot_max)
        #ax1.set_xlim(0, rstop)


        ax1.set_xlabel(r'Separation Distance ($r$)', fontsize=fsize*2.8)



        ax1.set_ylabel(r'$g(r)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        x_tick_val = radial_df_dict['r'][np.where(rdf_allall==np.max(rdf_allall))[0][0]]
        loc = ticker.MultipleLocator(base=(x_tick_val*2))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(x_tick_val))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)
        #plt.legend(loc='upper right', fontsize=fsize*2.6)
        #step = 2.0
        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        # Left middle plot
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)
        #plt.legend(loc='upper right')

        leg = [Line2D([0], [0], lw=8.0, c=green, label='A-A', linestyle='solid'), Line2D([0], [0], lw=8.0, c=orange, label='A-B', linestyle='solid'), Line2D([0], [0], lw=8.0, c=purple, label='B-B', linestyle='solid')]

        legend = ax1.legend(handles=leg, loc='upper right', columnspacing=1., handletextpad=0.3, bbox_transform=ax1.transAxes, bbox_to_anchor=[0.98, 1.03], fontsize=fsize*2.8, frameon=False, ncol=1)
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'rdf_' + out + ".png", dpi=300)
        #plt.close()

    def plot_general_adf(self, angular_df_dict):

        fsize=10

        adf_allall = np.array(angular_df_dict['all-all'])
        adf_aa = np.array(angular_df_dict['A-A'])
        adf_ab = np.array(angular_df_dict['A-B'])
        adf_bb = np.array(angular_df_dict['B-B'])

        fig, ax1 = plt.subplots(figsize=(12,6))

        plot_max = 1.2 * np.max(angular_df_dict['all-all'])

        plot_min = -0.02

        rstop=2*np.pi
        step = np.abs(plot_max - plot_min)/6

        plt.plot(angular_df_dict['theta'], angular_df_dict['all-all'],
                    c='black', lw=9.0, ls='-', alpha=1, label='All-All')


        ax1.set_ylim(plot_min, plot_max)
        #ax1.set_ylim(0, 2)


        ax1.set_xlabel(r'Interparticle separation angle from $+\hat{\mathbf{x}}$ ($\theta$)', fontsize=fsize*2.8)

        ax1.set_ylabel(r'$g(\theta)$', fontsize=fsize*2.8)

        #lat_theory = np.mean(lat_theory_arr)
        # Set all the x ticks for radial plots

        loc = ticker.MultipleLocator(base=(np.pi/3))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(np.pi/6))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)
        plt.legend(loc='upper right', fontsize=fsize*2.6)
        #step = 2.0
        # Set y ticks

        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        # Left middle plot
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)
        #plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'rdf_all_' + out + ".png", dpi=300)
        #plt.close()

    def plot_all_adfs(self, angular_df_dict):

        fsize=10

        adf_allall = np.array(angular_df_dict['all-all'])
        adf_aa = np.array(angular_df_dict['A-A'])
        adf_ab = np.array(angular_df_dict['A-B'])
        adf_bb = np.array(angular_df_dict['B-B'])

        #all_bulk_max = np.max(rdf_all_bulk_rdf)
        AA_bulk_max = np.max(adf_aa)
        AB_bulk_max = np.max(adf_ab)
        #BA_bulk_max = np.max(rdf_BA_bulk_rdf)
        BB_bulk_max = np.max(adf_bb)
        #all_bulk_max = np.max(rdf_all_bulk_rdf)
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

        if AA_bulk_order == 1:
            plot_max = 1.2 * AA_bulk_max
        elif AB_bulk_order == 1:
            plot_max = 1.2 * AB_bulk_max
        elif BB_bulk_order == 1:
            plot_max = 1.2 * BB_bulk_max

        plot_min = -0.02

        rstop=2*np.pi

        step = np.abs(plot_max - plot_min)/6
        step_x = 1.5
        fastCol = '#e31a1c'
        slowCol = '#081d58'
        purple = ("#00441b")
        orange = ("#ff7f00")
        green = ("#377EB8")

        fig, ax1 = plt.subplots(figsize=(12,6))

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

        ax1.set_ylim(plot_min, plot_max)
        #ax1.set_xlim(0, rstop)

        ax1.set_xlabel(r'Interparticle separation angle from $+\hat{\mathbf{x}}$ ($\theta$)', fontsize=fsize*2.8)

        ax1.set_ylabel(r'$g(\theta)$', fontsize=fsize*2.8)

        # Set all the x ticks for radial plots
        loc = ticker.MultipleLocator(base=(np.pi/3))
        ax1.xaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=(np.pi/6))
        ax1.xaxis.set_minor_locator(loc)
        ax1.set_xlim(0, rstop)
        #plt.legend(loc='upper right', fontsize=fsize*2.6)
        #step = 2.0
        # Set y ticks
        loc = ticker.MultipleLocator(base=step)
        ax1.yaxis.set_major_locator(loc)
        loc = ticker.MultipleLocator(base=step/2)
        ax1.yaxis.set_minor_locator(loc)
        # Left middle plot
        ax1.tick_params(axis='x', labelsize=fsize*2.5)
        ax1.tick_params(axis='y', labelsize=fsize*2.5)
        #plt.legend(loc='upper right')

        leg = [Line2D([0], [0], lw=8.0, c=green, label='A-A', linestyle='solid'), Line2D([0], [0], lw=8.0, c=orange, label='A-B', linestyle='solid'), Line2D([0], [0], lw=8.0, c=purple, label='B-B', linestyle='solid')]

        legend = ax1.legend(handles=leg, loc='upper right', columnspacing=1., handletextpad=0.3, bbox_transform=ax1.transAxes, bbox_to_anchor=[0.98, 1.03], fontsize=fsize*2.8, frameon=False, ncol=3)
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'rdf_' + out + ".png", dpi=300)
        #plt.close()
    def plot_neighbors(self, neigh_plot_dict, ang, pos, sep_surface_dict=None, int_comp_dict=None, pair='all-all'):
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

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))

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

        if pair == 'all-all':

            # Find min/max number of neighbors
            min_neigh = 0
            max_neigh = np.amax(neigh_plot_dict['all-all']['neigh'])

            # Generate list of ellipses for all particles to plot containing position (x,y) and point size that automatically scales with figure size
            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-all']['x'][i]+self.hx_box,neigh_plot_dict['all-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-all']['x']))]

            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells, cmap=plt.cm.get_cmap('tab10', int(max_neigh) + 1))#facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-all']['neigh']))

            px = np.sin(ang[typ1ind])
            py = -np.cos(ang[typ1ind])

            #plt.scatter(neigh_plot_dict['all-all']['x'][typ1ind]+self.hx_box, neigh_plot_dict['all-all']['y'][typ1ind]+self.hy_box, c='black', s=sz)
            plt.quiver(pos[typ1ind,0]+self.hx_box, pos[typ1ind,1]+self.hy_box, px, py, color='black', width=0.003)
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


    def plot_particle_orientations(self, neigh_plot_dict, sep_surface_dict, int_comp_dict, pair='all-all'):


        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = np.mean(neigh_plot_dict['all-all']['x']+self.hx_box)

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
        if pair == 'all-all':

            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['all-all']['ori'])
            max_ori = np.amax(neigh_plot_dict['all-all']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-all']['x'][i]+self.hx_box,neigh_plot_dict['all-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-all']['x']))]
            
            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-all']['ori']))
            
        elif pair == 'all-A':

            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['all-A']['ori'])
            max_ori = np.amax(neigh_plot_dict['all-A']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-A']['x'][i]+self.hx_box,neigh_plot_dict['all-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-A']['x']))]

            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-A']['ori']))

        elif pair == 'all-B':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['all-B']['ori'])
            max_ori = np.amax(neigh_plot_dict['all-B']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['all-B']['x'][i]+self.hx_box,neigh_plot_dict['all-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['all-B']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['all-B']['ori']))
        elif pair == 'A-all':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['A-all']['ori'])
            max_ori = np.amax(neigh_plot_dict['A-all']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-all']['x'][i]+self.hx_box,neigh_plot_dict['A-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-all']['x']))]

            # Plot position colored by neighbor number
            neighborGroup = mc.PatchCollection(ells)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-all']['ori']))
        elif pair == 'B-all':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['B-all']['ori'])
            max_ori = np.amax(neigh_plot_dict['B-all']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-all']['x'][i]+self.hx_box,neigh_plot_dict['B-all']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-all']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells, facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-all']['ori']))
        elif pair == 'A-A':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['A-A']['ori'])
            max_ori = np.amax(neigh_plot_dict['A-A']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-A']['x'][i]+self.hx_box,neigh_plot_dict['A-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-A']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells, facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-A']['ori']))
        elif pair == 'A-B':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['A-B']['ori'])
            max_ori = np.amax(neigh_plot_dict['A-B']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['A-B']['x'][i]+self.hx_box,neigh_plot_dict['A-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['A-B']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells, facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['A-B']['ori']))
        elif pair == 'B-A':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['B-A']['ori'])
            max_ori = np.amax(neigh_plot_dict['B-A']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-A']['x'][i]+self.hx_box,neigh_plot_dict['B-A']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-A']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells, facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-A']['ori']))
        elif pair == 'B-B':
            # Find min/max orientation
            min_ori = np.amin(neigh_plot_dict['B-B']['ori'])
            max_ori = np.amax(neigh_plot_dict['B-B']['ori'])

            ells = [Ellipse(xy=np.array([neigh_plot_dict['B-B']['x'][i]+self.hx_box,neigh_plot_dict['B-B']['y'][i]+self.hy_box]),
                    width=sz, height=sz)
            for i in range(0,len(neigh_plot_dict['B-B']['x']))]

            # Plot position colored by neighbor number
            slowCol='red'
            neighborGroup = mc.PatchCollection(ells, facecolors=slowCol)
            coll = ax.add_collection(neighborGroup)
            coll.set_array(np.ravel(neigh_plot_dict['B-B']['ori']))

        minClb = min_ori
        maxClb = max_ori
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_ori, max_ori+1, (max_ori - min_ori)/6)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.8, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.9, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #plt.quiver(self.pos_x, self.pos_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)

        if pair == 'all-all':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'all-A':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'all-B':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)
        elif pair == 'A-all':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'B-all':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('All Reference Particles', fontsize=20)
        elif pair == 'A-A':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'A-B':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)
        elif pair == 'B-A':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('A Reference Particles', fontsize=20)
        elif pair == 'B-B':
            clb.set_label(r'$\langle \mathbf{\hat{p}}_\mathrm{ref} \cdot \mathbf{\hat{p}}_\mathrm{neigh} \rangle$', labelpad=25, y=0.5, rotation=270, fontsize=20)
            plt.title('B Reference Particles', fontsize=20)

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
        plt.show()
        stop
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_hexatic_order(self, pos, hexatic_order_param, sep_surface_dict, int_comp_dict):

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
        levels_text=40
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

        # Find min/max orientation
        min_hex = 0.0
        max_hex = 1.0

        # Set plotted particle size
        sz = 0.75
        
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        # Plot position colored by neighbor number
        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(hexatic_order_param))
        
        minClb = min_hex
        maxClb = max_hex
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_hex, max_hex+1, (max_hex - min_hex)/5)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        #clb.set_label(r'$g_6(r)$', labelpad=-55, y=1.04, rotation=0, fontsize=18)

        clb.set_label(r'$g_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)
        
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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'hexatic_order_' + out + pad + ".png", dpi=100)
        #plt.close()
        plt.show()
    def plot_domain_angle(self, pos, relative_angles, sep_surface_dict, int_comp_dict):

        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

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

        # Find min/max orientation
        min_theta = 0.0
        max_theta = np.pi/3

        # Set plotted particle size
        sz = 0.75
        
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        # Plot position colored by neighbor number
        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(relative_angles))
        
        minClb = min_theta
        maxClb = max_theta
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_theta, max_theta+1, (max_theta - min_theta)/6)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        clb.set_label(r'$\theta_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)

        clb.locator     = matplotlib.ticker.FixedLocator(tick_locs)
        clb.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
        clb.update_ticks()

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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'relative_angle_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_trans_order(self, pos, trans_param, sep_surface_dict, int_comp_dict):

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

        # Find min/max orientation
        min_tran = np.min(trans_param)
        max_tran = 1.0

        # Set plotted particle size
        sz = 0.75
        
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        # Plot position colored by neighbor number
        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(trans_param))
        
        minClb = min_tran
        maxClb = max_tran
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_tran, max_tran+1, (max_tran - min_tran)/6)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        clb.set_label(r'$\Psi_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)

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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'translational_order_' + out + pad + ".png", dpi=100)
        #plt.close()
        stop
    def plot_stein_order(self, pos, stein_param, sep_surface_dict, int_comp_dict):

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




        # Find min/max orientation
        min_stein = np.min(stein_param)
        max_stein = 1.0


        levels_text=40
        level_boundaries = np.linspace(min_stein, max_stein, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

        # Set plotted particle size
        sz = 0.75
        
        ells = [Ellipse(xy=np.array([pos[i,0]+self.hx_box,pos[i,1]+self.hy_box]),
                width=sz, height=sz)
        for i in range(0,len(pos))]

        # Plot position colored by neighbor number
        neighborGroup = mc.PatchCollection(ells)
        coll = ax.add_collection(neighborGroup)
        coll.set_array(np.ravel(stein_param))
        
        minClb = min_stein
        maxClb = max_stein
        coll.set_clim([minClb, maxClb])

        # Modify colorbar properties
        tick_lev = np.arange(min_stein, max_stein+1, (max_stein - min_stein)/6)

        #level_boundaries = np.arange(min_ori-0.5, int(max_ori) + 1.5, 1)
        clb = plt.colorbar(coll, ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)

        clb.set_label(r'$q_6(a)$', labelpad=25, y=0.5, rotation=270, fontsize=20)
        plt.title('All Reference Particles', fontsize=20)


        
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

        # Label simulation time
        if self.lx_box == self.ly_box:
            plt.text(0.75, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        elif self.lx_box > self.ly_box:
            plt.text(0.85, 0.1, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'translational_order_' + out + pad + ".png", dpi=100)
        #plt.close()

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

    def plot_interpart_press_binned(self, interpart_press_binned, sep_surface_dict, int_comp_dict):

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(interpart_press_binned)
        max_n = np.max(interpart_press_binned)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, interpart_press_binned, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

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

        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)


        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])
        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = fig.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, format=tick.FormatStrFormatter('%.2f'))
        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\Pi^\mathrm{P}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.1f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'num_dens_' + out + pad + ".png", dpi=100)
        #plt.close()

    def local_orientational_order_map(self, neigh_plot_dict, sep_surface_dict, int_comp_dict, type='all-all'):

        pos_x = neigh_plot_dict['all-all']['x']
        pos_y = neigh_plot_dict['all-all']['y']
        ori = neigh_plot_dict['all-all']['ori']

        min_n = np.min(ori)
        max_n = np.max(ori)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(pos_x+self.hx_box, pos_y+self.hy_box, c=ori, s=0.7, vmin=min_n, vmax=max_n)



        tick_lev = np.arange(min_n, max_n+(max_n-min_n)/6, (max_n - min_n)/6)
        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\hat{\Pi}_\mathrm{d}^\mathrm{P}$', labelpad=-90, y=1.08, rotation=0, fontsize=20)

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

        plt.xlim(0, self.lx_box)
        plt.ylim(0, self.ly_box)

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        stop
        #plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=100)
        #plt.close()

    def interpart_press_map(self, pos, interpart_press_part, sep_surface_dict, int_comp_dict):

        bulk_lat_mean = np.mean(interpart_press_part)

        min_n = np.min(interpart_press_part)
        max_n = np.max(interpart_press_part)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(pos[:,0]+self.hx_box, pos[:,1]+self.hy_box, c=interpart_press_part, s=0.7, vmin=min_n, vmax=max_n)



        if bulk_lat_mean != 0.0:
            tick_lev = np.arange(min_n, max_n+(max_n-min_n)/6, (max_n - min_n)/6)
            #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
            #sm.set_array([])
            clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        else:
            clb = plt.colorbar(orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        clb.ax.tick_params(labelsize=16)
        clb.set_label(r'$\hat{\Pi}_\mathrm{d}^\mathrm{P}$', labelpad=-90, y=1.08, rotation=0, fontsize=20)

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

        ax.axis('off')
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'lat_map_' + out + pad + ".png", dpi=100)
        #plt.close()
    def plot_part_activity(self, pos, sep_surface_dict=None, int_comp_dict=None, active_fa_dict=None):

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        # If box is rectangular with long dimension of x-axis
        if self.lx_box > self.ly_box:

            # Estimated area of dense phase
            area_dense = (0.8 * self.partNum * (np.pi/4) / self.phiCP)

            # Mid point of dense phase across longest box dimension (x)
            dense_x_mid = self.hx_box

            # estimated shortest dimension length of dense phase (y)
            dense_x_width = np.amax(pos[typ0ind,0]) * 1.5 #(area_dense / self.ly_box)

            # Set maximum dimension length (x) of simulation box to be 12 inches (plus 1 inch color bar)
            scaling = 13.0

            # X and Y-dimension lengths (in inches)
            x_dim = int(scaling)
            y_dim = int(scaling/ (2*dense_x_width / self.ly_box))

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
            
            ells0 = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(typ1ind))]

            if self.lx_box > self.ly_box:
                ells0_top = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box+self.ly_box]),
                    width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells0_bottom = [Ellipse(xy=np.array([pos0[i,0]+self.hx_box, pos0[i,1]+self.hy_box-self.ly_box]),
                        width=sz, height=sz, label='PeA: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1_top = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box + self.ly_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(typ1ind))]
                ells1_bottom = [Ellipse(xy=np.array([pos1[i,0]+self.hx_box, pos1[i,1]+self.hy_box - self.ly_box]),
                    width=sz, height=sz, label='PeB: '+str(self.peB))
                for i in range(0,len(typ1ind))]
                if self.peA <= self.peB:
                    slowGroup_bottom = mc.PatchCollection(ells0_bottom, facecolors=slowCol)
                    slowGroup_top = mc.PatchCollection(ells0_top, facecolors=slowCol)
                    fastGroup_bottom = mc.PatchCollection(ells1_bottom,facecolors=fastCol)
                    fastGroup_top = mc.PatchCollection(ells1_top,facecolors=fastCol)
                    ax.add_collection(slowGroup_bottom)
                    ax.add_collection(slowGroup_top)
                    ax.add_collection(fastGroup_bottom)
                    ax.add_collection(fastGroup_top)

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
            plt.xlim(-(dense_x_width)+self.hx_box, (dense_x_width)+self.hx_box)
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

    def plot_vorticity(self, velocity, vorticity, sep_surface_dict, int_comp_dict, species='all'):

        min_n_temp = np.abs(np.min(vorticity))
        max_n_temp = np.abs(np.max(vorticity))

        if min_n_temp > max_n_temp:
            max_n = min_n_temp
            min_n = -min_n_temp
        else:
            min_n = -max_n_temp
            max_n = max_n_temp

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)

        levels_text=20
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, vorticity, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
        norm= matplotlib.colors.Normalize(vmin=min_n, vmax=max_n)

        sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        sm.set_array([])

        tick_lev = np.arange(min_n, max_n+max_n/10, (max_n-min_n)/10)
        clb = plt.colorbar(sm, ticks=tick_lev, boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, orientation="vertical", format=tick.FormatStrFormatter('%.3f'))
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

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

        clb.ax.tick_params(labelsize=16)
        if species == 'all':
            clb.set_label(r'$\nabla \times v$', labelpad=-40, y=1.07, rotation=0, fontsize=20)
        elif species == 'A':
            clb.set_label(r'$\nabla \times v_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)
        elif species == 'B':
            clb.set_label(r'$\nabla \times v_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

        plt.quiver(self.pos_x, self.pos_y, velocity['x'], velocity['y'], scale=20.0, color='black', alpha=0.8)


        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)

        ax.axis('off')
        plt.tight_layout()
        if species == 'all':
            plt.savefig(outPath + 'curl_map_' + out + pad + ".png", dpi=100)
        elif species == 'A':
            plt.savefig(outPath + 'curl_A_map_' + out + pad + ".png", dpi=100)
        elif species == 'B':
            plt.savefig(outPath + 'curl_B_map_' + out + pad + ".png", dpi=100)

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
