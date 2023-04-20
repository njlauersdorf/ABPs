
import sys
import os

from gsd import hoomd
from freud import box
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage

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
from matplotlib.colors import LinearSegmentedColormap

import numpy as np


#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility
import plotting_utility
import phase_identification
import binning
import particles

class kinetic_props:
    def __init__(self, lx_box, ly_box, NBins_x, NBins_y, partNum, typ, eps, peA, peB, parFrac):

        import freud

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        # Instantiated simulation box
        self.f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

        try:
            # Total number of bins in x-direction
            self.NBins_x = int(NBins_x)

            # Total number of bins in y-direction
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        # Initialize utility functions for call back later
        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        # X-length of bin
        self.sizeBin_x = self.utility_functs.roundUp((lx_box / NBins_x), 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp((ly_box / NBins_y), 6)

        # Number of particles
        self.partNum = partNum

        # Array (partNum) of particle types
        self.typ = typ

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Fraction of particles of type A
        self.parFrac = parFrac

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

        self.plotting_utility_functs = plotting_utility.plotting_utility(self.lx_box, self.ly_box, self.partNum, self.typ)
    def particle_flux(self, partPhase_time, in_clust_arr, partPhase_time_arr, clust_size_arr, pos_x_arr_time, pos_y_arr_time, com_x_arr_time, com_y_arr_time, com_x_parts_arr_time, com_y_parts_arr_time):
        start_part_phase = partPhase_time[0:,]
        start_bulk_id = np.where(partPhase_time[0,:]==0)[0]
        start_gas_id = np.where(partPhase_time[0,:]==2)[0]
        start_int_id = np.where(partPhase_time[0,:]==1)[0]

        start_clust_id = np.where(in_clust_arr[0,:]==1)[0]
        start_gas2_id = np.where(in_clust_arr[0,:]==0)[0]

        start_bulk_id_with_int = np.where(partPhase_time[0,:]==0)[0]
        start_gas_id_with_int = np.where(partPhase_time[0,:]==2)[0]
        start_int_id_with_int = np.where(partPhase_time[0,:]==1)[0]

        num_clust_to_gas2 = np.array([])
        num_slow_clust_to_gas2 = np.array([])
        num_fast_clust_to_gas2 = np.array([])

        num_gas2_to_clust = np.array([])
        num_slow_gas2_to_clust = np.array([])
        num_fast_gas2_to_clust = np.array([])

        num_bulk_to_gas = np.array([])
        num_slow_bulk_to_gas = np.array([])
        num_fast_bulk_to_gas = np.array([])

        num_gas_to_bulk = np.array([])
        num_slow_gas_to_bulk = np.array([])
        num_fast_gas_to_bulk = np.array([])

        num_gas_to_int = np.array([])
        num_slow_gas_to_int = np.array([])
        num_fast_gas_to_int = np.array([])

        num_int_to_gas = np.array([])
        num_slow_int_to_gas = np.array([])
        num_fast_int_to_gas = np.array([])

        num_bulk_to_int = np.array([])
        num_slow_bulk_to_int = np.array([])
        num_fast_bulk_to_int = np.array([])

        num_int_to_bulk = np.array([])
        num_slow_int_to_bulk = np.array([])
        num_fast_int_to_bulk = np.array([])

        num_bulk_to_gas = np.array([])
        num_slow_bulk_to_gas = np.array([])
        num_fast_bulk_to_gas = np.array([])

        num_gas_to_bulk = np.array([])
        num_slow_gas_to_bulk = np.array([])
        num_fast_gas_to_bulk = np.array([])

        num_bulk_to_gas_no_int = np.array([])
        num_slow_bulk_to_gas_no_int = np.array([])
        num_fast_bulk_to_gas_no_int = np.array([])

        num_gas_to_bulk_no_int = np.array([])
        num_slow_gas_to_bulk_no_int = np.array([])
        num_fast_gas_to_bulk_no_int = np.array([])

        align_vect = np.array([])
        percent_change_vect = np.array([])

        difx_clust_arr = np.array([])
        dify_clust_arr = np.array([])
        difr_clust_arr = np.array([])

        difx_adsorb_desorb_arr = np.array([])
        dify_adsorb_desorb_arr = np.array([])
        difr_adsorb_desorb_arr = np.array([])

        difx_adsorb_arr = np.array([])
        dify_adsorb_arr = np.array([])
        difr_adsorb_arr = np.array([])
        

        for j in range(1, np.shape(partPhase_time)[0]):

            bulk_id = np.where(partPhase_time[j,:]==0)[0]
            gas_id = np.where(partPhase_time[j,:]==2)[0]
            int_id = np.where(partPhase_time[j,:]==1)[0]

            clust_id = np.where(in_clust_arr[j,:]==1)[0]
            gas2_id = np.where(in_clust_arr[j,:]==0)[0]
            gas2_id_prev = np.where(in_clust_arr[j-1,:]==0)[0]

            still_in_clust = np.intersect1d(start_clust_id, clust_id, return_indices=True)
            not_in_clust = np.delete(start_clust_id, still_in_clust[1])

            still_in_gas2 = np.intersect1d(start_gas2_id, gas2_id, return_indices=True)
            not_in_gas2 = np.delete(start_gas2_id, still_in_gas2[1])

            still_in_bulk_no_int = np.intersect1d(start_bulk_id, bulk_id, return_indices=True)
            not_in_bulk_no_int = np.delete(start_bulk_id, still_in_bulk_no_int[1])

            still_in_gas_no_int = np.intersect1d(start_gas_id, gas_id, return_indices=True)
            not_in_gas_no_int = np.delete(start_gas_id, still_in_gas_no_int[1])

            still_in_bulk = np.intersect1d(start_bulk_id_with_int, bulk_id, return_indices=True)
            not_in_bulk = np.delete(start_bulk_id_with_int, still_in_bulk[1])

            still_in_gas = np.intersect1d(start_gas_id_with_int, gas_id, return_indices=True)
            not_in_gas = np.delete(start_gas_id_with_int, still_in_gas[1])

            still_in_int = np.intersect1d(start_int_id_with_int, int_id, return_indices=True)
            not_in_int = np.delete(start_int_id_with_int, still_in_int[1])

            clust_now_in_gas2 = np.intersect1d(gas2_id, not_in_clust, return_indices=True)
            not_in_clust_ids = np.intersect1d(start_clust_id, clust_now_in_gas2[0], return_indices=True)

            gas2_now_in_clust = np.intersect1d(clust_id, not_in_gas2, return_indices=True)
            not_in_gas2_ids = np.intersect1d(start_gas2_id, gas2_now_in_clust[0], return_indices=True)

            bulk_now_in_gas_no_int = np.intersect1d(gas_id, not_in_bulk_no_int, return_indices=True)
            gas_now_in_bulk_no_int = np.intersect1d(bulk_id, not_in_gas_no_int, return_indices=True)
            not_in_bulk_ids_to_gas_no_int = np.intersect1d(start_bulk_id, bulk_now_in_gas_no_int[0], return_indices=True)
            not_in_gas_ids_to_bulk_no_int = np.intersect1d(start_gas_id, gas_now_in_bulk_no_int[0], return_indices=True)

            bulk_now_in_int = np.intersect1d(int_id, not_in_bulk, return_indices=True)
            int_now_in_bulk = np.intersect1d(bulk_id, not_in_int, return_indices=True)
            not_in_bulk_ids_to_int = np.intersect1d(start_bulk_id_with_int, bulk_now_in_int[0], return_indices=True)
            not_in_int_ids_to_bulk = np.intersect1d(start_int_id_with_int, int_now_in_bulk[0], return_indices=True)

            gas_now_in_int = np.intersect1d(int_id, not_in_gas, return_indices=True)
            int_now_in_gas = np.intersect1d(gas_id, not_in_int, return_indices=True)
            not_in_gas_ids_to_int = np.intersect1d(start_gas_id_with_int, gas_now_in_int[0], return_indices=True)
            not_in_int_ids_to_gas = np.intersect1d(start_int_id_with_int, int_now_in_gas[0], return_indices=True)

            gas_now_in_bulk = np.intersect1d(bulk_id, not_in_gas, return_indices=True)
            bulk_now_in_gas = np.intersect1d(gas_id, not_in_bulk, return_indices=True)

            not_in_gas_ids_to_bulk = np.intersect1d(start_gas_id_with_int, gas_now_in_bulk[0], return_indices=True)
            not_in_bulk_ids_to_gas = np.intersect1d(start_bulk_id_with_int, bulk_now_in_gas[0], return_indices=True)

            not_in_bulk_comb = np.append(not_in_bulk_ids_to_gas[1], not_in_bulk_ids_to_int[1])
            not_in_int_comb = np.append(not_in_int_ids_to_gas[1], not_in_int_ids_to_bulk[1])
            not_in_gas_comb = np.append(not_in_gas_ids_to_bulk[1], not_in_gas_ids_to_int[1])

            if len(clust_now_in_gas2)>0:
                num_clust_to_gas2 = np.append(num_clust_to_gas2, len(clust_now_in_gas2[0]))
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, len(np.where(self.typ[clust_now_in_gas2[0].astype('int')]==0)[0]))
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, len(np.where(self.typ[clust_now_in_gas2[0].astype('int')]==1)[0]))
            else:
                num_clust_to_gas2 = np.append(num_clust_to_gas2, 0)
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, 0)
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, 0)

            if len(gas2_now_in_clust)>0:
                num_gas2_to_clust = np.append(num_gas2_to_clust, len(gas2_now_in_clust[0]))
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, len(np.where(self.typ[gas2_now_in_clust[0].astype('int')]==0)[0]))
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, len(np.where(self.typ[gas2_now_in_clust[0].astype('int')]==1)[0]))
            else:
                num_gas2_to_clust = np.append(num_gas2_to_clust, 0)
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, 0)
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, 0)
            
            
            com_x_desorb = np.mean(pos_x_arr_time[j,gas2_now_in_clust[0]])
            com_y_desorb = np.mean(pos_y_arr_time[j,gas2_now_in_clust[0]])
            com_x_adsorb = np.mean(pos_x_arr_time[j-1,clust_now_in_gas2[0]])
            com_y_adsorb = np.mean(pos_y_arr_time[j-1,clust_now_in_gas2[0]])



            pos_x_arr_desorb_adsorb = np.append(pos_x_arr_time[j,gas2_now_in_clust[0]], pos_x_arr_time[j-1,clust_now_in_gas2[0]])
            pos_y_arr_desorb_adsorb = np.append(pos_y_arr_time[j,gas2_now_in_clust[0]], pos_y_arr_time[j-1,clust_now_in_gas2[0]])


            com_adsorb_desorb_dict = self.plotting_utility_functs.com_part_view(pos_x_arr_time[j, clust_id], pos_y_arr_time[j, clust_id], pos_x_arr_desorb_adsorb, pos_y_arr_desorb_adsorb, com_x_parts_arr_time[j]-self.hx_box, com_y_parts_arr_time[j]-self.hy_box)
            com_adsorb_dict = self.plotting_utility_functs.com_part_view(pos_x_arr_time[j, clust_id], pos_y_arr_time[j, clust_id], pos_x_arr_time[j,gas2_now_in_clust[0]], pos_y_arr_time[j,gas2_now_in_clust[0]], com_x_parts_arr_time[j]-self.hx_box, com_y_parts_arr_time[j]-self.hy_box)
            com_desorb_dict = self.plotting_utility_functs.com_part_view(pos_x_arr_time[j, clust_id], pos_y_arr_time[j, clust_id], pos_x_arr_time[j-1,clust_now_in_gas2[0]], pos_y_arr_time[j-1,clust_now_in_gas2[0]], com_x_parts_arr_time[j]-self.hx_box, com_y_parts_arr_time[j]-self.hy_box)

            #difx_adsorb_desorb = com_adsorb_dict['com']['x']-com_desorb_dict['com']['x']
            #dify_adsorb_desorb = com_adsorb_dict['com']['y']-com_desorb_dict['com']['y']

            difx_adsorb_desorb = self.utility_functs.sep_dist_x(com_adsorb_dict['com']['x'], com_desorb_dict['com']['x'])
            dify_adsorb_desorb = self.utility_functs.sep_dist_y(com_adsorb_dict['com']['y'], com_desorb_dict['com']['y'])

            difr_adsorb_desorb = ( difx_adsorb_desorb ** 2 + dify_adsorb_desorb ** 2 ) ** 0.5

            difx_clust = self.utility_functs.sep_dist_x(com_x_parts_arr_time[j], com_x_parts_arr_time[j-1])
            dify_clust = self.utility_functs.sep_dist_y(com_y_parts_arr_time[j], com_y_parts_arr_time[j-1])

            clust_id_prev = np.where(in_clust_arr[j-1,:]==1)[0]
            clust_now_in_gas_prev = np.intersect1d(clust_id_prev, gas2_id, return_indices=True)
            clust_desorb = np.intersect1d(clust_id_prev, clust_now_in_gas_prev[0], return_indices=True)
            clust_without_desorb = np.delete(clust_id_prev, clust_desorb[1])

            gas_now_in_clust_prev = np.intersect1d(gas2_id_prev, clust_id, return_indices=True)
            clust_with_adsorb = np.append(clust_without_desorb, gas_now_in_clust_prev[0])
            pos_x_with_adsorb = np.append(pos_x_arr_time[j-1, clust_without_desorb], pos_x_arr_time[j, clust_with_adsorb])
            pos_y_with_adsorb = np.append(pos_y_arr_time[j-1, clust_without_desorb], pos_y_arr_time[j, clust_with_adsorb])

            com_adsorb_desorb_dict2 = self.plotting_utility_functs.com_part_view(pos_x_with_adsorb, pos_y_with_adsorb, pos_x_with_adsorb, pos_y_with_adsorb, com_x_parts_arr_time[j-1]-self.hx_box, com_y_parts_arr_time[j-1]-self.hy_box)

            difx_adsorb = self.utility_functs.sep_dist_x(com_adsorb_desorb_dict2['com']['x'], com_x_parts_arr_time[j-1]-self.hx_box)
            dify_adsorb = self.utility_functs.sep_dist_y(com_adsorb_desorb_dict2['com']['y'], com_y_parts_arr_time[j-1]-self.hy_box)
            difr_adsorb = ( difx_adsorb ** 2 + dify_adsorb ** 2 ) ** 0.5

            #plt.scatter(pos_x_with_adsorb, pos_y_with_adsorb, s=0.7, color='black')
            #plt.scatter(com_adsorb_desorb_dict2['com']['x'], com_adsorb_desorb_dict2['com']['y'], s=25, color='red')
            #plt.scatter(com_x_parts_arr_time[j]-self.hx_box, com_y_parts_arr_time[j]-self.hy_box, s=25, color='blue')
            #plt.scatter(com_x_parts_arr_time[j-1]-self.hx_box, com_y_parts_arr_time[j-1]-self.hy_box, s=25, color='purple')
            #plt.show()
            difr_clust = ( difx_clust ** 2 + dify_clust ** 2 ) ** 0.5

            difx_clust_norm = difx_clust / difr_clust
            dify_clust_norm = dify_clust / difr_clust

            difx_adsorb_desorb_norm = difx_adsorb_desorb / difr_adsorb_desorb
            dify_adsorb_desorb_norm = dify_adsorb_desorb / difr_adsorb_desorb

            align_vect = np.append(align_vect, (difx_clust_norm * difx_adsorb_desorb_norm) + (dify_clust_norm * dify_adsorb_desorb_norm))
            percent_change_vect = np.append(percent_change_vect, (difr_adsorb_desorb - difr_clust) / difr_clust)

            difx_clust_arr = np.append(difx_clust_arr, difx_clust)
            dify_clust_arr = np.append(dify_clust_arr, dify_clust)
            difr_clust_arr = np.append(difr_clust_arr, difr_clust)

            difx_adsorb_desorb_arr = np.append(difx_adsorb_desorb_arr, difx_adsorb_desorb)
            dify_adsorb_desorb_arr = np.append(dify_adsorb_desorb_arr, dify_adsorb_desorb)
            difr_adsorb_desorb_arr = np.append(difr_adsorb_desorb_arr, difr_adsorb_desorb)

            difx_adsorb_arr = np.append(difx_adsorb_arr, difx_adsorb)
            dify_adsorb_arr = np.append(dify_adsorb_arr, dify_adsorb)
            difr_adsorb_arr = np.append(difr_adsorb_arr, difr_adsorb)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, len(bulk_now_in_gas_no_int[0]))
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, len(np.where(self.typ[bulk_now_in_gas_no_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, len(np.where(self.typ[bulk_now_in_gas_no_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, 0)
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, 0)
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, len(gas_now_in_bulk_no_int[0]))
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, len(np.where(self.typ[gas_now_in_bulk_no_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, len(np.where(self.typ[gas_now_in_bulk_no_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, 0)
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, 0)
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, 0)

            if len(bulk_now_in_int)>0:
                num_bulk_to_int = np.append(num_bulk_to_int, len(bulk_now_in_int[0]))

                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, len(np.where(self.typ[bulk_now_in_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, len(np.where(self.typ[bulk_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_int = np.append(num_bulk_to_int, 0)
                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, 0)
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, 0)

            if len(int_now_in_bulk)>0:
                num_int_to_bulk = np.append(num_int_to_bulk, len(int_now_in_bulk[0]))
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, len(np.where(self.typ[int_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, len(np.where(self.typ[int_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_int_to_bulk = np.append(num_int_to_bulk, 0)
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, 0)
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, 0)

            if len(gas_now_in_int)>0:
                num_gas_to_int = np.append(num_gas_to_int, len(gas_now_in_int[0]))
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, len(np.where(self.typ[gas_now_in_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, len(np.where(self.typ[gas_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_int = np.append(num_gas_to_int, 0)
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, 0)
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, 0)

            if len(int_now_in_gas)>0:
                num_int_to_gas = np.append(num_int_to_gas, len(int_now_in_gas[0]))
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, len(np.where(self.typ[int_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, len(np.where(self.typ[int_now_in_gas[0].astype('int')]==1)[0]))
            else:
                num_int_to_gas = np.append(num_int_to_gas, 0)
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, 0)
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk = np.append(num_gas_to_bulk, len(gas_now_in_bulk[0]))
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, len(np.where(self.typ[gas_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, len(np.where(self.typ[gas_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk = np.append(num_gas_to_bulk, 0)
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, 0)
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, 0)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas = np.append(num_bulk_to_gas, len(bulk_now_in_gas[0]))
                num_slow_bulk_to_gas = np.append(num_slow_bulk_to_gas, len(np.where(self.typ[bulk_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas = np.append(num_fast_bulk_to_gas, len(np.where(self.typ[bulk_now_in_gas[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_gas = np.append(num_bulk_to_gas, 0)
                num_slow_bulk_to_gas = np.append(num_slow_bulk_to_gas, 0)
                num_fast_bulk_to_gas = np.append(num_fast_bulk_to_gas, 0)


            now_in_bulk_comb = np.array([])
            now_in_gas_comb = np.array([])
            now_in_int_comb = np.array([])

            no_flux_bulk = 0

            if (len(bulk_now_in_int)>0) & (len(bulk_now_in_gas)>0):
                now_in_int_comb = np.append(now_in_int_comb, bulk_now_in_int[0])
                now_in_gas_comb = np.append(now_in_gas_comb, bulk_now_in_gas[0])
            elif (len(bulk_now_in_int)>0) & (len(bulk_now_in_gas)==0):
                now_in_int_comb = np.append(now_in_int_comb, bulk_now_in_int[0])
            elif (len(bulk_now_in_int)==0) & (len(bulk_now_in_gas)>0):
                now_in_gas_comb = np.append(now_in_gas_comb, bulk_now_in_gas[0])
            else:
                no_flux_bulk = 1


            no_flux_gas = 0

            if (len(gas_now_in_int)>0) & (len(gas_now_in_bulk)>0):
                now_in_int_comb = np.append(now_in_int_comb, gas_now_in_int[0])
                now_in_bulk_comb = np.append(now_in_bulk_comb, gas_now_in_bulk[0])
            elif (len(gas_now_in_int)>0) & (len(gas_now_in_bulk)==0):
                now_in_int_comb = np.append(now_in_int_comb, gas_now_in_int[0])
            elif (len(gas_now_in_int)==0) & (len(gas_now_in_bulk)>0):
                now_in_bulk_comb = np.append(now_in_bulk_comb, gas_now_in_bulk[0])
            else:
                no_flux_gas = 1


            no_flux_int = 0

            if (len(int_now_in_gas)>0) & (len(int_now_in_bulk)>0):
                now_in_gas_comb = np.append(now_in_gas_comb, int_now_in_gas[0])
                now_in_bulk_comb = np.append(now_in_bulk_comb, int_now_in_bulk[0])
            elif (len(int_now_in_gas)>0) & (len(int_now_in_bulk)==0):
                now_in_gas_comb = np.append(now_in_gas_comb, int_now_in_gas[0])
            elif (len(int_now_in_gas)==0) & (len(int_now_in_bulk)>0):
                now_in_bulk_comb = np.append(now_in_bulk_comb, int_now_in_bulk[0])
            else:
                no_flux_int = 1

            if no_flux_bulk == 0:
                start_bulk_id_with_int = np.delete(start_bulk_id_with_int, not_in_bulk_comb)

            if no_flux_gas == 0:
                start_gas_id_with_int = np.delete(start_gas_id_with_int, not_in_gas_comb)

            if no_flux_int == 0:
                start_int_id_with_int = np.delete(start_int_id_with_int, not_in_int_comb)


            start_bulk_id = np.delete(start_bulk_id, not_in_bulk_ids_to_gas_no_int[1])
            start_gas_id = np.delete(start_gas_id, not_in_gas_ids_to_bulk_no_int[1])

            start_gas_id = np.append(start_gas_id, bulk_now_in_gas_no_int[0])
            start_bulk_id = np.append(start_bulk_id, gas_now_in_bulk_no_int[0])

            start_clust_id = np.delete(start_clust_id, not_in_clust_ids[1])
            start_gas2_id = np.delete(start_gas2_id, not_in_gas2_ids[1])

            start_clust_id = np.append(start_clust_id, gas2_now_in_clust[0])
            start_gas2_id = np.append(start_gas2_id, clust_now_in_gas2[0])


            start_int_id_with_int = np.append(start_int_id_with_int, now_in_int_comb)
            start_gas_id_with_int = np.append(start_gas_id_with_int, now_in_gas_comb)
            start_bulk_id_with_int = np.append(start_bulk_id_with_int, now_in_bulk_comb)

        
        num_gas_to_dense = (num_gas_to_bulk + num_gas_to_int)
        num_slow_gas_to_dense = (num_slow_gas_to_bulk + num_slow_gas_to_int)
        num_fast_gas_to_dense = (num_fast_gas_to_bulk + num_fast_gas_to_int)

        num_dense_to_gas = (num_bulk_to_gas + num_int_to_gas)
        num_slow_dense_to_gas = (num_slow_bulk_to_gas + num_slow_int_to_gas)
        num_fast_dense_to_gas = (num_fast_bulk_to_gas + num_fast_int_to_gas)

        adsorption_dict = {'tauB': partPhase_time_arr[1:].tolist(), 'gas_to_clust': {'all': num_gas2_to_clust.tolist(), 'A': num_slow_gas2_to_clust.tolist(),'B': num_fast_gas2_to_clust.tolist()}, 'clust_to_gas': {'all': num_clust_to_gas2.tolist(), 'A': num_slow_clust_to_gas2.tolist(),'B': num_fast_clust_to_gas2.tolist()}, 'gas_to_dense': {'all': num_gas_to_dense.tolist(), 'A': num_slow_gas_to_dense.tolist(),'B': num_fast_gas_to_dense.tolist()}, 'dense_to_gas': {'all': num_dense_to_gas.tolist(), 'A': num_slow_dense_to_gas.tolist(),'B': num_fast_dense_to_gas.tolist()}, 'gas_to_bulk': {'all': num_gas_to_bulk.tolist(), 'A': num_slow_gas_to_bulk.tolist(),'B': num_fast_gas_to_bulk.tolist()}, 'bulk_to_gas': {'all': num_bulk_to_gas.tolist(), 'A': num_slow_bulk_to_gas.tolist(),'B': num_fast_bulk_to_gas.tolist()}, 'int_to_bulk': {'all': num_int_to_bulk.tolist(), 'A': num_slow_int_to_bulk.tolist(),'B': num_fast_int_to_bulk.tolist()}, 'bulk_to_int': {'all': num_bulk_to_int.tolist(), 'A': num_slow_bulk_to_int.tolist(),'B': num_fast_bulk_to_int.tolist()}, 'gas_to_int': {'all': num_gas_to_int.tolist(), 'A': num_slow_gas_to_int.tolist(),'B': num_fast_gas_to_int.tolist()}, 'int_to_gas': {'all': num_int_to_gas.tolist(), 'A': num_slow_int_to_gas.tolist(),'B': num_fast_int_to_gas.tolist()}, 'gas_to_bulk_no_int': {'all': num_gas_to_bulk_no_int.tolist(), 'A': num_slow_gas_to_bulk_no_int.tolist(),'B': num_fast_gas_to_bulk_no_int.tolist()}, 'bulk_to_gas_no_int': {'all': num_bulk_to_gas_no_int.tolist(), 'A': num_slow_bulk_to_gas_no_int.tolist(),'B': num_fast_bulk_to_gas_no_int.tolist()}}
        clust_motion_dict = {'tauB': partPhase_time_arr[1:].tolist(), 'align': align_vect.tolist(), 'magnitude': percent_change_vect.tolist(), 'net_flux': {'x': difx_adsorb_arr.tolist(), 'y': dify_adsorb_arr.tolist(), 'r': difr_adsorb_arr.tolist()}, 'total': {'x': difx_clust_arr.tolist(), 'y': dify_clust_arr.tolist(), 'r': difr_clust_arr.tolist()}, 'com_flux': {'x': difx_adsorb_desorb_arr.tolist(), 'y': dify_adsorb_desorb_arr.tolist(), 'r': difr_adsorb_desorb_arr.tolist()}}
        return adsorption_dict, clust_motion_dict