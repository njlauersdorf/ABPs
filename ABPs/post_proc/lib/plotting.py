
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

import statistics
from statistics import mode

#from symfit import parameters, variables, sin, cos, Fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility

from scipy.optimize import curve_fit

class plotting:
    def __init__(self, orient_dict, pos_dict, l_box, NBins, sizeBin, peA, peB, parFrac, eps, typ, tst):

        self.orient_x = orient_dict['bin']['all']['x']
        self.orient_y = orient_dict['bin']['all']['y']
        self.orient_mag = orient_dict['bin']['all']['mag']

        self.pos_x = pos_dict['mid point']['x']
        self.pos_y = pos_dict['mid point']['y']

        self.l_box = l_box
        self.h_box = self.l_box/2
        utility_functs = utility.utility(self.l_box)

        self.peA = peA
        self.peB = peB
        self.parFrac = parFrac

        theory_functs = theory.theory()

        try:
            self.NBins = int(NBins)
        except:
            print('NBins must be either a float or an integer')

        self.sizeBin = utility_functs.roundUp((self.l_box / self.NBins), 6)

        self.peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

        self.eps = eps

        self.typ = typ

        self.tst = tst

    def plot_phases(self, pos, phase_ids_dict, sep_surface_dict, int_comp_dict):

        yellow = ("#7570b3")
        green = ("#77dd77")
        red = ("#ff6961")

        bulk_part_ids = phase_ids_dict['ids']['bulk']['all']
        gas_part_ids = phase_ids_dict['ids']['gas']['all']
        int_part_ids = phase_ids_dict['ids']['int']['all']

        fig = plt.figure(figsize=(8.5,8))
        ax = fig.add_subplot(111)

        if len(bulk_part_ids)>0:
            plt.scatter(pos[bulk_part_ids,0]+self.h_box, pos[bulk_part_ids,1]+self.h_box, s=0.75, marker='.', c=green)
        if len(gas_part_ids)>0:
            plt.scatter(pos[gas_part_ids,0]+self.h_box, pos[gas_part_ids,1]+self.h_box, s=0.75, marker='.', c=red)
        if len(int_part_ids)>0:
            plt.scatter(pos[int_part_ids,0]+self.h_box, pos[int_part_ids,1]+self.h_box, s=0.75, marker='.', c=yellow)

        plt.quiver(self.pos_x, self.pos_y, self.orient_x, self.orient_y)

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




        pos_ticks = np.linspace(0, self.NBins, self.NBins + 1) * self.sizeBin

        plt.xticks(pos_ticks)
        plt.yticks(pos_ticks)

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

        plt.text(0.77, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{B}$',
                fontsize=18,transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        eps_leg=[]
        mkSz = [0.1, 0.1, 0.15, 0.1, 0.1]
        msz=40

        plt.xlim(0, self.l_box)
        plt.ylim(0, self.l_box)
        red_patch = mpatches.Patch(color=red, label='Dilute')
        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        #purple_patch = mpatches.Patch(color=purple, label='Bubble')
        plt.legend(handles=[green_patch, yellow_patch, red_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=16, loc='upper left',labelspacing=0.1, handletextpad=0.1)
        plt.tight_layout()
        plt.show()
        #plt.savefig(outPath + 'plot_phases_' + out + pad + ".png", dpi=100)
        #plt.close()
    def plot_all_density(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        num_dens = num_dens_dict['bin']['all']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.0
        max_n = np.max(num_dens)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, num_dens, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
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
        clb.set_label(r'$\phi$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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
    def plot_type_A_density(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        num_dens_A = num_dens_dict['bin']['A']

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.0
        max_n = np.max(num_dens_A)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, num_dens_A, level_boundaries, vmin=min_n, vmax=max_n, cmap='Blues', extend='both')
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
        clb.set_label(r'$\phi_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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

    def plot_type_B_density(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        num_dens_B = num_dens_dict['bin']['B']
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

    def plot_dif_density(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        num_dens_dif = num_dens_dict['bin']['dif']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -np.max(np.abs(num_dens_dif))
        max_n = np.max(np.abs(num_dens_dif))
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, num_dens_dif, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\phi_\mathrm{B}-\phi_\mathrm{A}$', labelpad=-30, y=1.07, rotation=0, fontsize=20)

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
    def plot_type_B_frac(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        fast_frac = num_dens_dict['bin']['fast frac']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.0
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, fast_frac, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\chi_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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
    def plot_dif_frac(self, num_dens_dict, sep_surface_dict, int_comp_dict):

        fast_frac = num_dens_dict['bin']['fast frac']
        slow_frac = np.ones(np.shape(fast_frac)) - fast_frac

        frac_dif = fast_frac - slow_frac
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -1.0
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, frac_dif, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\chi_\mathrm{B}-\chi_\mathrm{A}$', labelpad=-30, y=1.07, rotation=0, fontsize=20)

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

    def plot_all_align(self, align_dict, sep_surface_dict, int_comp_dict):

        align = align_dict['bin']['all']['mag']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -1.0#0.0
        max_n = 1.0#np.max(align)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, align, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\alpha$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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

    def plot_type_A_align(self, align_dict, sep_surface_dict, int_comp_dict):

        align_A = align_dict['bin']['A']['mag']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -1.0
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, align_A, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\alpha_\mathrm{A}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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

    def plot_type_B_align(self, align_dict, sep_surface_dict, int_comp_dict):

        align_B = align_dict['bin']['B']['mag']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -1.0
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, align_B, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\alpha_\mathrm{B}$', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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

    def plot_dif_align(self, align_dict, sep_surface_dict, int_comp_dict):

        align_dif = align_dict['bin']['dif']['mag']
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = -np.max(np.abs(align_dif))
        max_n = np.max(np.abs(align_dif))
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.contourf(self.pos_x, self.pos_y, align_dif, level_boundaries, vmin=min_n, vmax=max_n, cmap='seismic', extend='both')
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
        clb.set_label(r'$\alpha_\mathrm{B}-\alpha_\mathrm{A}$', labelpad=-30, y=1.07, rotation=0, fontsize=20)

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

    def lat_histogram(self, lat_plot_dict):

        dense_lat_mean = np.mean(lat_plot_dict['bulk']['all']['vals'])

        xmin = 0.85*dense_lat_mean
        xmax = 1.1*dense_lat_mean

        #Define colors for plots
        yellow = ("#7570b3")
        green = ("#77dd77")

        bulk_lats = lat_plot_dict['bulk']['all']['vals']
        int_lats = lat_plot_dict['int']['all']['vals']

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        #Remove bulk particles that are outside plot's xrange
        if (len(bulk_lats)>0):
            bulk_id = np.where((bulk_lats > xmax) | (bulk_lats < xmin))[0]
            bulk_lats = np.delete(bulk_lats, bulk_id)

            plt.hist(bulk_lats, alpha = 1.0, bins=100, color=green)

        #If interface particle measured, continue
        if (len(int_lats)>0):
            int_id = np.where((int_lats > xmax) | (int_lats < xmin))[0]
            int_lats = np.delete(int_lats, int_id)

            plt.hist(int_lats, alpha = 0.8, bins=100, color=yellow)

        green_patch = mpatches.Patch(color=green, label='Bulk')
        yellow_patch = mpatches.Patch(color=yellow, label='Interface')
        plt.legend(handles=[green_patch, yellow_patch], fancybox=True, framealpha=0.75, ncol=1, fontsize=18, loc='upper right',labelspacing=0.1, handletextpad=0.1)

        plt.xlabel(r'lattice spacing ($a$)', fontsize=18)
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
    def lat_map(self, lat_plot_dict, sep_surface_dict, int_comp_dict, velocity_dict=None):

        bulk_lat_mean = np.mean(lat_plot_dict['bulk']['all']['vals'])

        dense_lats = lat_plot_dict['dense']['all']
        min_n = 0.97*bulk_lat_mean
        max_n = 1.03*bulk_lat_mean

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(lat_plot_dict['dense']['all']['x']+self.h_box, lat_plot_dict['dense']['all']['y']+self.h_box, c=lat_plot_dict['dense']['all']['vals'], s=0.7, vmin=min_n, vmax=max_n)



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
        plt.text(0.75, 0.92, s=r'$\overline{a}$' + ' = ' + '{:.3f}'.format(bulk_lat_mean),
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        if velocity_dict!=None:
            plt.quiver(self.pos_x, self.pos_y, velocity_dict['bin']['all']['x'], velocity_dict['bin']['all']['y'], scale=20.0, color='black', alpha=0.8)

        clb.ax.tick_params(labelsize=16)
        clb.set_label('a', labelpad=-40, y=1.07, rotation=0, fontsize=20)

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

    def plot_all_neighbors_of_all_parts(self, neigh_plot_dict, sep_surface_dict, int_comp_dict):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(neigh_plot_dict['all-all']['x']+self.h_box, neigh_plot_dict['all-all']['y']+self.h_box, c=neigh_plot_dict['all-all']['neigh'], s=0.7)


        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        min_n = 0.0
        max_n = np.amax(neigh_plot_dict['all-all']['neigh'])

        tick_lev = np.arange(min_n, max_n+1, 1)
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        #plt.quiver(pos_box_x, pos_box_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)
        clb.ax.tick_params(labelsize=16)
        clb.set_label('# neighbors for all particles', labelpad=25, y=0.5, rotation=270, fontsize=20)

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
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_A_neighbors_of_all_parts(self, neigh_plot_dict, sep_surface_dict, int_comp_dict):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(neigh_plot_dict['A-all']['x']+self.h_box, neigh_plot_dict['A-all']['y']+self.h_box, c=neigh_plot_dict['A-all']['neigh'], s=0.7)


        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        min_n = 0.0
        max_n = np.amax(neigh_plot_dict['A-all']['neigh'])

        tick_lev = np.arange(min_n, max_n+1, 1)
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        #plt.quiver(pos_box_x, pos_box_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)
        clb.ax.tick_params(labelsize=16)
        clb.set_label('# A neighbors for all particles', labelpad=25, y=0.5, rotation=270, fontsize=20)

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
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_B_neighbors_of_all_parts(self, neigh_plot_dict, sep_surface_dict, int_comp_dict):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(neigh_plot_dict['B-all']['x']+self.h_box, neigh_plot_dict['B-all']['y']+self.h_box, c=neigh_plot_dict['B-all']['neigh'], s=0.7)


        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        min_n = 0.0
        max_n = np.amax(neigh_plot_dict['B-all']['neigh'])

        tick_lev = np.arange(min_n, max_n+1, 1)
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        #plt.quiver(pos_box_x, pos_box_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)
        clb.ax.tick_params(labelsize=16)
        clb.set_label('# B neighbors for all particles', labelpad=25, y=0.5, rotation=270, fontsize=20)

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
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_all_neighbors_of_A_parts(self, neigh_plot_dict, sep_surface_dict, int_comp_dict):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(neigh_plot_dict['all-A']['x']+self.h_box, neigh_plot_dict['all-A']['y']+self.h_box, c=neigh_plot_dict['all-A']['neigh'], s=0.7)


        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        min_n = 0.0
        max_n = np.amax(neigh_plot_dict['all-A']['neigh'])

        tick_lev = np.arange(min_n, max_n+1, 1)
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        #plt.quiver(pos_box_x, pos_box_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)
        clb.ax.tick_params(labelsize=16)
        clb.set_label('# neighbors for A particles', labelpad=25, y=0.5, rotation=270, fontsize=20)

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
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_all_neighbors_of_B_parts(self, neigh_plot_dict, sep_surface_dict, int_comp_dict):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(neigh_plot_dict['all-B']['x']+self.h_box, neigh_plot_dict['all-B']['y']+self.h_box, c=neigh_plot_dict['all-B']['neigh'], s=0.7)


        #sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
        #sm.set_array([])
        min_n = 0.0
        max_n = np.amax(neigh_plot_dict['all-B']['neigh'])

        tick_lev = np.arange(min_n, max_n+1, 1)
        clb = plt.colorbar(ticks=tick_lev, orientation="vertical", format=tick.FormatStrFormatter('%.0f'))

        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.text(0.663, 0.04, s=r'$\tau$' + ' = ' + '{:.2f}'.format(self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                fontsize=18, transform = ax.transAxes,
                bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))
        #plt.quiver(pos_box_x, pos_box_y, velocity_x_bin_plot, velocity_y_bin_plot, scale=20.0, color='black', alpha=0.8)
        clb.ax.tick_params(labelsize=16)
        clb.set_label('# neighbors for B particles', labelpad=25, y=0.5, rotation=270, fontsize=20)

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
        #plt.savefig(outPath + 'num_neigh_' + out + pad + ".png", dpi=100)
        #plt.close()
    def plot_hexatic_order(self, pos, hexatic_order_param, sep_surface_dict, int_comp_dict):

        #Plot particles colorized by hexatic order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = 0.9
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=hexatic_order_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
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
        #pad = str(j).zfill(4)
        #plt.savefig(outPath + 'hexatic_order_' + out + pad + ".png", dpi=100)
        #plt.close()
        plt.show()
    def plot_domain_angle(self, pos, relative_angles, sep_surface_dict, int_comp_dict):

        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']

        #Plot particles colorized by bond orientation angle
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(relative_angles)
        max_n = np.max(relative_angles)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=relative_angles, s=0.7, vmin=0.0, vmax=np.pi/3, cmap='viridis')
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
        #plt.savefig(outPath + 'relative_angle_' + out + pad + ".png", dpi=100)
        #plt.close()

    def plot_trans_order(self, pos, trans_param, sep_surface_dict, int_comp_dict):

        #Plot particles colorized by translational order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.min(trans_param)
        max_n = 1.0
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']



        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=trans_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
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

    def plot_stein_order(self, pos, stein_param, sep_surface_dict, int_comp_dict):

        #Plot particles colorized by translational order parameter
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        div_min = -3
        min_n = np.mean(stein_param)
        max_n = np.max(stein_param)
        levels_text=40
        level_boundaries = np.linspace(min_n, max_n, levels_text + 1)
        tick_locs   = [0.0,np.pi/6,np.pi/3]
        tick_labels = ['0',r'$\pi/6$',r'$\pi/3$']



        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=stein_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
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



        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=nematic_param, s=0.7, vmin=min_n, vmax=max_n, cmap='viridis')
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

    def interpart_press_map(self, pos, interpart_press_part, sep_surface_dict, int_comp_dict):

        bulk_lat_mean = np.mean(interpart_press_part)

        min_n = np.min(interpart_press_part)
        max_n = np.max(interpart_press_part)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        im = plt.scatter(pos[:,0]+self.h_box, pos[:,1]+self.h_box, c=interpart_press_part, s=0.7, vmin=min_n, vmax=max_n)



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
    def plot_part_activity(self, pos, sep_surface_dict, int_comp_dict):

        #Set plot colors
        fastCol = '#e31a1c'
        slowCol = '#081d58'

        typ0ind = np.where(self.typ == 0)[0]
        typ1ind = np.where(self.typ == 1)[0]

        if (self.parFrac == 100.):
            mono=1
            mono_activity=self.peA
            mono_type = 0
        elif (self.parFrac == 0.):
            mono = 1
            mono_activity=self.peB
            mono_type = 1
        elif self.peA==self.peB:
            mono=1
            mono_activity=self.peA
            mono_type = 2
        else:
            mono=0

        #Plot each particle as a point color-coded by activity and labeled by their activity
        fig = plt.figure(figsize=(6.5,6))
        ax = fig.add_subplot(111)

        sz = 0.75

        if mono==0:
            #Local each particle's positions
            pos0=pos[typ0ind]                               # Find positions of type 0 particles
            pos1=pos[typ1ind]

            #Assign type 0 particles to plot

            ells0 = [Ellipse(xy=pos0[i,:]+self.h_box,
                    width=sz, height=sz, label='PeA: '+str(self.peA))
            for i in range(0,len(typ0ind))]

            #Assign type 1 particles to plot
            ells1 = [Ellipse(xy=pos1[i,:]+self.h_box,
                    width=sz, height=sz, label='PeB: '+str(self.peB))
            for i in range(0,len(typ1ind))]

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
            if self.parFrac<100.0:
                leg = ax.legend(handles=[ells0[0], ells1[1]], labels=[r'$\mathrm{Pe}_\mathrm{A} = $'+str(int(self.peA)), r'$\mathrm{Pe}_\mathrm{B} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                if self.peA <= self.peB:
                    leg.legendHandles[0].set_color(slowCol)
                    leg.legendHandles[1].set_color(fastCol)
                else:
                    leg.legendHandles[0].set_color(fastCol)
                    leg.legendHandles[1].set_color(slowCol)
            #Create legend for monodisperse system
            else:
                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peA)), r'$\mathrm{Pe} = $'+str(int(self.peA))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

        elif mono == 1:
            if mono_type == 0:
                #Local each particle's positions
                pos0=pos[typ0ind]                               # Find positions of type 0 particles

                #Assign type 0 particles to plot
                ells0 = [Ellipse(xy=pos0[i,:]+self.h_box,
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
                ells1 = [Ellipse(xy=pos1[i,:]+self.h_box,
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
                ells0 = [Ellipse(xy=pos0[i,:]+self.h_box,
                        width=sz, height=sz, label='Pe: '+str(self.peA))
                for i in range(0,len(typ0ind))]
                ells1 = [Ellipse(xy=pos1[i,:]+self.h_box,
                        width=sz, height=sz, label='Pe: '+str(self.peB))
                for i in range(0,len(typ1ind))]

                # Plot position colored by neighbor number
                slowGroup = mc.PatchCollection(ells0, facecolors=slowCol)
                ax.add_collection(slowGroup)
                fastGroup = mc.PatchCollection(ells1, facecolors=slowCol)
                ax.add_collection(fastGroup)

                leg = ax.legend(handles=[ells0[0]], labels=[r'$\mathrm{Pe} = $'+str(int(self.peB))], loc='upper right', prop={'size': 15}, markerscale=8.0)
                leg.legendHandles[0].set_color(slowCol)

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

        #Label time step
        ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.2f}'.format(3*self.tst) + ' ' + r'$\tau_\mathrm{r}$',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

        #Set axes parameters
        ax.set_xlim(0, self.l_box)
        ax.set_ylim(0, self.l_box)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axes.set_xticklabels([])
        ax.axes.set_yticks([])
        ax.set_aspect('equal')

        # Create frame pad for images
        #pad = str(j).zfill(4)

        plt.tight_layout()
        #plt.savefig(outPath+out + pad + ".png", dpi=150, transparent=False)
        #plt.close()
        plt.show()
