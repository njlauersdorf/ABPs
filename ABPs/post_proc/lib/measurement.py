
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

class measurement:
    def __init__(self, l_box, NBins, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict):

        import freud

        self.l_box = l_box
        self.h_box = self.l_box/2

        self.f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)

        try:
            self.NBins = int(NBins)
        except:
            print('NBins must be either a float or an integer')

        self.utility_functs = utility.utility(self.l_box)

        self.sizeBin = self.utility_functs.roundUp((l_box / NBins), 6)

        self.partNum = partNum

        self.phasePart = phase_dict['part']

        self.phaseBin = phase_dict['bin']

        self.phase_dict = phase_dict

        self.part_dict = part_dict

        self.pos = pos

        self.typ = typ

        self.ang = ang

        self.binParts = part_dict['id']

        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        self.eps = eps

        self.peA = peA

        self.peB = peB

        self.parFrac = parFrac

        self.align_dict = align_dict

        self.area_frac_dict = area_frac_dict

        self.press_dict = press_dict

        self.binning = binning.binning(self.l_box, self.partNum, self.NBins, self.peA, self.peB, self.typ, self.eps)

        self.plotting_utility = plotting_utility.plotting_utility(self.l_box, self.partNum, self.typ)

        self.phase_ident_functs = phase_identification.phase_identification(self.area_frac_dict, self.align_dict, self.part_dict, self.press_dict, self.l_box, self.partNum, self.NBins, self.peA, self.peB, self.parFrac, self.eps, self.typ)

        self.particle_prop_functs = particles.particle_props(self.l_box, self.partNum, self.NBins, self.peA, self.peB, self.typ, self.pos, self.ang)

        self.theory_functs = theory.theory()


    def average_activity(self, part_ids = None):
        pe_tot = 0
        pe_num = 0

        if part_ids is None:
            part_ids = self.binParts

        for i in part_ids:

            if self.typ[i]==0:
                pe_tot += self.peA
                pe_num += 1
            else:
                pe_tot += self.peB
                pe_num += 1

        if pe_num>0:
            peNet = pe_tot / pe_num
        else:
            peNet = 0

        return peNet
    def lattice_spacing(self, bulk_dict):

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]                               # Find positions of type 0 particles
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))

        A_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        B_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        bulk_A_lats = self.utility_functs.sep_dist_arr(pos_dense[A_bulk_nlist.point_indices], pos_A_bulk[A_bulk_nlist.query_point_indices])
        bulk_B_lats = self.utility_functs.sep_dist_arr(pos_dense[B_bulk_nlist.point_indices], pos_B_bulk[B_bulk_nlist.query_point_indices])

        bulk_lats = np.append(bulk_A_lats, bulk_B_lats)

        bulk_lat_mean = np.mean(bulk_lats)
        bulk_A_lat_mean = np.mean(bulk_A_lats)
        bulk_B_lat_mean = np.mean(bulk_B_lats)

        bulk_lat_std = np.std(bulk_lats)
        bulk_A_lat_std = np.std(bulk_A_lats)
        bulk_B_lat_std = np.std(bulk_B_lats)

        bulk_A_lat_ind = np.array([], dtype=int)
        bulk_A_lat_arr = np.array([])

        bulk_B_lat_ind = np.array([], dtype=int)
        bulk_B_lat_arr = np.array([])

        for i in A_bulk_nlist.point_indices:
            if i not in bulk_A_lat_ind:
                loc = np.where(A_bulk_nlist.point_indices==i)[0]
                bulk_A_lat_arr = np.append(bulk_A_lat_arr, np.mean(bulk_A_lats[loc]))
                bulk_A_lat_ind = np.append(bulk_A_lat_ind, int(i))

        for i in B_bulk_nlist.point_indices:
            if i not in bulk_B_lat_ind:
                loc = np.where(B_bulk_nlist.point_indices==i)[0]
                bulk_B_lat_arr = np.append(bulk_B_lat_arr, np.mean(bulk_B_lats[loc]))
                bulk_B_lat_ind = np.append(bulk_B_lat_ind, int(i))

        bulk_lat_arr = np.append(bulk_A_lat_arr, bulk_B_lat_arr)
        bulk_lat_ind = np.append(bulk_A_lat_ind, bulk_B_lat_ind)

        system_all_int = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))   #Calculate neighbor list

        A_int_nlist = system_all_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        B_int_nlist = system_all_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        int_A_lats = self.utility_functs.sep_dist_arr(self.pos[A_int_nlist.point_indices], pos_A_int[A_int_nlist.query_point_indices])
        int_B_lats = self.utility_functs.sep_dist_arr(self.pos[B_int_nlist.point_indices], pos_B_int[B_int_nlist.query_point_indices])

        int_lats = np.append(int_A_lats, int_B_lats)

        int_lat_mean = np.mean(int_lats)
        int_A_lat_mean = np.mean(int_A_lats)
        int_B_lat_mean = np.mean(int_B_lats)

        int_lat_std = np.std(int_lats)
        int_A_lat_std = np.std(int_A_lats)
        int_B_lat_std = np.std(int_B_lats)

        int_A_lat_ind = np.array([], dtype=int)
        int_A_lat_arr = np.array([])

        int_B_lat_ind = np.array([], dtype=int)
        int_B_lat_arr = np.array([])

        for i in A_int_nlist.point_indices:
            if i not in int_A_lat_ind:
                loc = np.where(A_int_nlist.point_indices==i)[0]
                int_A_lat_arr = np.append(int_A_lat_arr, np.mean(int_A_lats[loc]))
                int_A_lat_ind = np.append(int_A_lat_ind, int(i))

        for i in B_int_nlist.point_indices:
            if i not in int_B_lat_ind:
                loc = np.where(B_int_nlist.point_indices==i)[0]
                int_B_lat_arr = np.append(int_B_lat_arr, np.mean(int_B_lats[loc]))
                int_B_lat_ind = np.append(int_B_lat_ind, int(i))

        int_lat_arr = np.append(int_A_lat_arr, int_B_lat_arr)
        int_lat_ind = np.append(int_A_lat_ind, int_B_lat_ind)

        dense_lat_arr = np.append(bulk_lat_arr, int_lat_arr)
        dense_A_lat_arr = np.append(bulk_A_lat_arr, int_A_lat_arr)
        dense_B_lat_arr = np.append(bulk_B_lat_arr, int_B_lat_arr)

        dense_lats = np.append(bulk_lats, int_lats)
        dense_A_lats = np.append(bulk_A_lats, int_A_lats)
        dense_B_lats = np.append(bulk_B_lats, int_B_lats)

        dense_lat_mean = np.mean(dense_lats)
        dense_A_lat_mean = np.mean(dense_A_lats)
        dense_B_lat_mean = np.mean(dense_B_lats)

        dense_lat_std = np.std(dense_lats)
        dense_A_lat_std = np.std(dense_A_lats)
        dense_B_lat_std = np.std(dense_B_lats)

        pos_bulk_x_lat = pos_dense[bulk_lat_ind,0]
        pos_bulk_y_lat = pos_dense[bulk_lat_ind,1]

        pos_bulk_x_A_lat = pos_dense[bulk_A_lat_ind,0]
        pos_bulk_y_A_lat = pos_dense[bulk_A_lat_ind,1]

        pos_bulk_x_B_lat = pos_dense[bulk_B_lat_ind,0]
        pos_bulk_y_B_lat = pos_dense[bulk_B_lat_ind,1]

        pos_int_x_lat = self.pos[int_lat_ind,0]
        pos_int_y_lat = self.pos[int_lat_ind,1]

        pos_int_x_A_lat = self.pos[int_A_lat_ind,0]
        pos_int_y_A_lat = self.pos[int_A_lat_ind,1]

        pos_int_x_B_lat = self.pos[int_B_lat_ind,0]
        pos_int_y_B_lat = self.pos[int_B_lat_ind,1]

        pos_dense_x_lat = np.append(pos_dense[bulk_lat_ind,0], self.pos[int_lat_ind,0])
        pos_dense_y_lat = np.append(pos_dense[bulk_lat_ind,1], self.pos[int_lat_ind,1])

        pos_dense_x_A_lat = np.append(pos_dense[bulk_A_lat_ind,0], self.pos[int_A_lat_ind,0])
        pos_dense_y_A_lat = np.append(pos_dense[bulk_A_lat_ind,1], self.pos[int_A_lat_ind,1])

        pos_dense_x_B_lat = np.append(pos_dense[bulk_B_lat_ind,0], self.pos[int_B_lat_ind,0])
        pos_dense_y_B_lat = np.append(pos_dense[bulk_B_lat_ind,1], self.pos[int_B_lat_ind,1])

        dense_lat_arr = np.append(bulk_lat_arr, int_lat_arr)

        lat_stat_dict = {'bulk': {'all': {'mean': bulk_lat_mean, 'std': bulk_lat_std}, 'A': {'mean': bulk_A_lat_mean, 'std': bulk_A_lat_std}, 'B': {'mean': bulk_B_lat_mean, 'std': bulk_B_lat_std}}, 'int': {'all': {'mean': int_lat_mean, 'std': int_lat_std}, 'A': {'mean': int_A_lat_mean, 'std': int_A_lat_std}, 'B': {'mean': int_B_lat_mean, 'std': int_B_lat_std}}, 'dense': {'all': {'mean': dense_lat_mean, 'std': dense_lat_std }, 'A': {'mean': dense_A_lat_mean, 'std': dense_A_lat_std }, 'B': {'mean': dense_B_lat_mean, 'std': dense_B_lat_std } } }
        lat_plot_dict = {'dense': {'all': {'vals': dense_lat_arr, 'x': pos_dense_x_lat, 'y': pos_dense_y_lat}, 'A': {'vals': dense_A_lat_arr, 'x': pos_dense_x_A_lat, 'y': pos_dense_y_A_lat}, 'B': {'vals': dense_B_lat_arr, 'x': pos_dense_x_B_lat, 'y': pos_dense_x_B_lat }  }, 'bulk': {'all': {'vals': bulk_lat_arr, 'x': pos_bulk_x_lat, 'y': pos_bulk_y_lat}, 'A': {'vals': bulk_A_lat_arr, 'x': pos_bulk_x_A_lat, 'y': pos_bulk_y_A_lat}, 'B': {'vals': bulk_B_lat_arr, 'x': pos_bulk_x_B_lat, 'y': pos_bulk_x_B_lat }  }, 'int': {'all': {'vals': int_lat_arr, 'x': pos_int_x_lat, 'y': pos_int_y_lat}, 'A': {'vals': int_A_lat_arr, 'x': pos_int_x_A_lat, 'y': pos_int_y_A_lat}, 'B': {'vals': int_B_lat_arr, 'x': pos_int_x_B_lat, 'y': pos_int_x_B_lat }  } }
        return lat_stat_dict, lat_plot_dict


        """
        #Output means and standard deviations of number density for each phase
        g = open(outPath2+outTxt_lat, 'a')
        g.write('{0:.2f}'.format(tst).center(15) + ' ')
        g.write('{0:.6f}'.format(sizeBin).center(15) + ' ')
        g.write('{0:.0f}'.format(np.amax(clust_size)).center(15) + ' ')
        g.write('{0:.6f}'.format(lat_theory2).center(15) + ' ')
        g.write('{0:.6f}'.format(bulk_lat_mean).center(15) + ' ')
        g.write('{0:.6f}'.format(bulk_lat_std).center(15) + ' ')
        g.write('{0:.6f}'.format(int_lat_mean).center(15) + ' ')
        g.write('{0:.6f}'.format(int_lat_std).center(15) + ' ')
        g.write('{0:.6f}'.format(dense_lat_mean).center(15) + ' ')
        g.write('{0:.6f}'.format(dense_lat_std).center(15) + '\n')
        g.close()
        """

    def num_dens_mean(self, area_frac_dict):

        num_dens = area_frac_dict['bin']['all']
        num_dens_A = area_frac_dict['bin']['A']
        num_dens_B = area_frac_dict['bin']['B']

        num_dens_sum = 0
        num_dens_A_sum = 0
        num_dens_B_sum = 0

        num_dens_val = 0

        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if self.phaseBin[ix][iy]==0:

                    num_dens_sum += num_dens[ix][iy]/(math.pi/4)
                    num_dens_val += 1

                    num_dens_A_sum += num_dens_A[ix][iy]/(math.pi/4)

                    num_dens_B_sum += num_dens_B[ix][iy]/(math.pi/4)
        if num_dens_val > 0:
            num_dens_mean = num_dens_sum / num_dens_val
            num_dens_A_mean = num_dens_A_sum / num_dens_val
            num_dens_B_mean = num_dens_B_sum / num_dens_val
        else:
            num_dens_mean = 0
            num_dens_A_mean = 0
            num_dens_B_mean = 0

        return {'all': num_dens_mean, 'A': num_dens_A_mean, 'B': num_dens_B_mean}

    def radial_df(self):

        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        distances = np.array([])
        bulk_area_test = phase_count_dict['bulk'] * (self.sizeBin**2)


        num_dens_mean = len(phase_part_dict['bulk']['all'])/bulk_area_test
        num_dens_A_mean = len(phase_part_dict['bulk']['A'])/bulk_area_test
        num_dens_B_mean = len(phase_part_dict['bulk']['B'])/bulk_area_test

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Width, in distance units, of bin
        wBins = 0.02

        # Distance to compute RDF for
        rstop = 10.

        # Number of bins given this distance
        nBins = rstop / wBins

        wbinsTrue=(rstop)/(nBins-1)

        r=np.arange(0.0,rstop+wbinsTrue,wbinsTrue)

        query_args = dict(mode='ball', r_min = 0.1, r_max=rstop)

        print('bulk neighbors')
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        #Checks for 'A' type neighbors around type 'A' particles within bulk
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()

        #Checks for 'A' type neighbors around type 'B' particles within bulk
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        #Checks for 'B' type neighbors around type 'B' particles within bulk
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        difr_AA_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AA_bulk_nlist.point_indices], pos_A_bulk[AA_bulk_nlist.query_point_indices])

        difr_AB_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AB_bulk_nlist.point_indices], pos_B_bulk[AB_bulk_nlist.query_point_indices])

        difr_BB_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[BB_bulk_nlist.point_indices], pos_B_bulk[BB_bulk_nlist.query_point_indices])


        g_r_allall_bulk = []
        g_r_allA_bulk = []
        g_r_allB_bulk = []
        g_r_AA_bulk = []
        g_r_AB_bulk = []
        g_r_BB_bulk = []

        r_arr = []

        difr_allA_bulk = np.append(difr_AA_bulk, difr_AB_bulk)
        difr_allB_bulk = np.append(difr_BB_bulk, difr_AB_bulk)

        difr_allall_bulk = np.append(difr_allA_bulk, difr_allB_bulk)


        for m in range(1, len(r)-2):
            difr = r[m+1] - r[m]

            inds = np.where((difr_allall_bulk>=r[m]) & (difr_allall_bulk<r[m+1]))[0]
            rho_all = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_all = len(phase_part_dict['bulk']['all']) * num_dens_mean_dict['all']
            g_r_allall_bulk.append(rho_all / rho_tot_all)

            inds = np.where((difr_allA_bulk>=r[m]) & (difr_allA_bulk<r[m+1]))[0]
            rho_a = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_a = len(pos_A_bulk) * num_dens_mean_dict['all']
            g_r_allA_bulk.append(rho_a / rho_tot_a)

            inds = np.where((difr_allB_bulk>=r[m]) & (difr_allB_bulk<r[m+1]))[0]
            rho_b = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_b = len(pos_B_bulk) * num_dens_mean_dict['all']
            g_r_allB_bulk.append(rho_b / rho_tot_b)

            inds = np.where((difr_AA_bulk>=r[m]) & (difr_AA_bulk<r[m+1]))[0]
            rho_aa = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_aa = len(pos_A_bulk) * num_dens_mean_dict['A']
            g_r_AA_bulk.append(rho_aa / rho_tot_aa)

            inds = np.where((difr_AB_bulk>=r[m]) & (difr_AB_bulk<r[m+1]))[0]
            rho_ab = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_ab = len(pos_B_bulk) * (num_dens_mean_dict['A'])
            g_r_AB_bulk.append(rho_ab / rho_tot_ab)

            inds = np.where((difr_BB_bulk>=r[m]) & (difr_BB_bulk<r[m+1]))[0]
            rho_bb = (len(inds) / (2*math.pi * r[m] * difr) )
            rho_tot_bb = len(pos_B_bulk) * num_dens_mean_dict['B']
            g_r_BB_bulk.append(rho_bb / rho_tot_bb)

            r_arr.append(r[m])

        rad_df_dict = {'r': r_arr, 'all-all': g_r_allall_bulk, 'all-A': g_r_allA_bulk, 'all-B': g_r_allB_bulk, 'A-A': g_r_AA_bulk, 'A-B': g_r_AB_bulk, 'B-B': g_r_BB_bulk}
        return rad_df_dict
    def angular_df(self):

        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        distances = np.array([])
        bulk_area_test = phase_count_dict['bulk'] * (self.sizeBin**2)

        num_dens_mean = len(phase_part_dict['bulk']['all'])/bulk_area_test
        num_dens_A_mean = len(phase_part_dict['bulk']['A'])/bulk_area_test
        num_dens_B_mean = len(phase_part_dict['bulk']['B'])/bulk_area_test

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Width, in distance units, of bin
        wBins = 0.02

        # Distance to compute RDF for
        rstop = 10.

        # Number of bins given this distance
        nBins = rstop / wBins

        wbinsTrue=(rstop)/(nBins-1)

        r=np.arange(0.0,rstop+wbinsTrue,wbinsTrue)

        #Angular Distribution Function
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        #system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        #Aall_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_bulk), query_args).toNeighborList()
        #Ball_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_bulk), query_args).toNeighborList()

        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        #difx_Aall_bulk, dify_Aall_bulk, difr_Aall_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[Aall_bulk_nlist.point_indices], pos_bulk[Aall_bulk_nlist.query_point_indices], difxy=True)

        #difx_Ball_bulk, dify_Ball_bulk, difr_Ball_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[Ball_bulk_nlist.point_indices], pos_bulk[Ball_bulk_nlist.query_point_indices], difxy=True)

        difx_AA_bulk, dify_AA_bulk, difr_AA_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AA_bulk_nlist.point_indices], pos_A_bulk[AA_bulk_nlist.query_point_indices], difxy=True)

        difx_AB_bulk, dify_AB_bulk, difr_AB_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AB_bulk_nlist.point_indices], pos_B_bulk[AB_bulk_nlist.query_point_indices], difxy=True)

        difx_BB_bulk, dify_BB_bulk, difr_BB_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[BB_bulk_nlist.point_indices], pos_B_bulk[BB_bulk_nlist.query_point_indices], difxy=True)

        #ang_loc_Aall = self.utility_functs.shift_quadrants(difx_Aall_bulk, dify_Aall_bulk)

        #ang_loc_Ball = self.utility_functs.shift_quadrants(difx_Ball_bulk, dify_Ball_bulk)

        ang_loc_AA = self.utility_functs.shift_quadrants(difx_AA_bulk, dify_AA_bulk)

        ang_loc_AB = self.utility_functs.shift_quadrants(difx_AB_bulk, dify_AB_bulk)

        ang_loc_BB = self.utility_functs.shift_quadrants(difx_BB_bulk, dify_BB_bulk)

        theta=np.arange(0.0,2*np.pi,np.pi/180)

        fastCol = '#e31a1c'
        slowCol = '#081d58'

        g_theta_allall_bulk = np.array([])
        g_theta_AA_bulk = np.array([])
        g_theta_Aall_bulk = np.array([])
        g_theta_Ball_bulk = np.array([])
        g_theta_AB_bulk = np.array([])
        g_theta_BB_bulk = np.array([])

        ang_loc_Aall = np.append(ang_loc_AA, ang_loc_AB)
        ang_loc_Ball = np.append(ang_loc_BB, ang_loc_AB)

        ang_loc_allall = np.append(ang_loc_Aall, ang_loc_Ball)

        theta_arr = np.array([])
        for m in range(1, len(theta)-2):
            diftheta = theta[m+1] - theta[m]

            inds = np.where((ang_loc_allall>=theta[m]) & (ang_loc_allall<theta[m+1]))[0]

            rho_all = len(inds)
            rho_tot_all = len(pos_bulk) * num_dens_mean_dict['all']
            g_theta_allall_bulk = np.append(g_theta_allall_bulk, rho_all / rho_tot_all)

            inds = np.where((ang_loc_Aall>=theta[m]) & (ang_loc_Aall<theta[m+1]))[0]
            rho_a = (len(inds))
            rho_tot_a = len(pos_A_bulk) * num_dens_mean_dict['all']
            g_theta_Aall_bulk = np.append(g_theta_Aall_bulk, rho_a / rho_tot_a)

            inds = np.where((ang_loc_Ball>=theta[m]) & (ang_loc_Ball<theta[m+1]))[0]
            rho_b = (len(inds))
            rho_tot_b = len(pos_B_bulk) * num_dens_mean_dict['all']
            g_theta_Ball_bulk = np.append(g_theta_Ball_bulk, rho_b / rho_tot_b)

            inds = np.where((ang_loc_AA>=theta[m]) & (ang_loc_AA<theta[m+1]))[0]
            rho_aa = (len(inds))
            rho_tot_aa = len(pos_A_bulk) * num_dens_mean_dict['A']
            g_theta_AA_bulk = np.append(g_theta_AA_bulk, rho_aa / rho_tot_aa)

            inds = np.where((ang_loc_AB>=theta[m]) & (ang_loc_AB<theta[m+1]))[0]
            rho_ab = (len(inds))
            rho_tot_ab = len(pos_B_bulk) * num_dens_mean_dict['B']
            g_theta_AB_bulk = np.append(g_theta_AB_bulk, rho_ab / rho_tot_ab)

            inds = np.where((ang_loc_BB>=theta[m]) & (ang_loc_BB<theta[m+1]))[0]
            rho_bb = (len(inds))
            rho_tot_bb = len(pos_B_bulk) * num_dens_mean_dict['B']
            g_theta_BB_bulk = np.append(g_theta_BB_bulk, rho_bb / rho_tot_bb)

            theta_arr = np.append(theta_arr, theta[m])

        g_theta_allall_bulk=g_theta_allall_bulk/(-np.trapz(theta_arr, g_theta_allall_bulk))
        g_theta_Aall_bulk=g_theta_Aall_bulk/(-np.trapz(theta_arr, g_theta_Aall_bulk))
        g_theta_Ball_bulk=g_theta_Ball_bulk/(-np.trapz(theta_arr, g_theta_Ball_bulk))
        g_theta_AA_bulk=g_theta_AA_bulk/(-np.trapz(theta_arr, g_theta_AA_bulk))
        g_theta_AB_bulk=g_theta_AB_bulk/(-np.trapz(theta_arr, g_theta_AB_bulk))
        g_theta_BB_bulk=g_theta_BB_bulk/(-np.trapz(theta_arr, g_theta_BB_bulk))

        ang_df_dict = {'theta': np.ndarray.tolist(theta_arr), 'all-all': np.ndarray.tolist(g_theta_allall_bulk), 'A-all': np.ndarray.tolist(g_theta_Aall_bulk), 'B-all': np.ndarray.tolist(g_theta_Ball_bulk), 'A-A': np.ndarray.tolist(g_theta_AA_bulk), 'A-B': np.ndarray.tolist(g_theta_AB_bulk), 'B-B': np.ndarray.tolist(g_theta_BB_bulk)}
        return ang_df_dict

    def nearest_neighbors(self):

        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)
        distances = np.array([])
        bulk_area_test = phase_count_dict['bulk'] * (self.sizeBin**2)

        peNet_int = self.average_activity(part_ids = phase_part_dict['int']['all'])

        if bulk_area_test > 0:
            num_dens_mean = len(phase_part_dict['bulk']['all'])/bulk_area_test
            num_dens_A_mean = len(phase_part_dict['bulk']['A'])/bulk_area_test
            num_dens_B_mean = len(phase_part_dict['bulk']['B'])/bulk_area_test
        else:
            num_dens_mean = 0
            num_dens_A_mean = 0
            num_dens_B_mean = 0

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]


        query_args = dict(mode='ball', r_min = 0.1, r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])
        AA_bulk_dot = np.array([])

        for i in range(0, len(pos_A_bulk)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                    AA_bulk_dot = np.append(AA_bulk_dot, np.sum(np.cos(ang_A_bulk[i]-ang_A_dense[AA_bulk_nlist.point_indices[loc]])))
            else:
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                AA_bulk_dot = np.append(AA_bulk_dot, 0)

        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])
        BA_bulk_dot = np.array([])

        for i in range(0, len(pos_A_bulk)):
            if i in BA_bulk_nlist.query_point_indices:
                if i not in BA_bulk_neigh_ind:
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]
                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
                    BA_bulk_dot = np.append(BA_bulk_dot, np.sum(np.cos(ang_A_bulk[i]-ang_B_dense[BA_bulk_nlist.point_indices[loc]])))
            else:
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
                BA_bulk_dot = np.append(BA_bulk_dot, 0)

        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])
        AB_bulk_dot = np.array([])

        for i in range(0, len(pos_B_bulk)):
            if i in AB_bulk_nlist.query_point_indices:
                if i not in AB_bulk_neigh_ind:
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]
                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_dot = np.append(AB_bulk_dot, np.sum(np.cos(ang_B_bulk[i]-ang_A_dense[AB_bulk_nlist.point_indices[loc]])))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
                AB_bulk_dot = np.append(AB_bulk_dot, 0)

        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])
        BB_bulk_dot = np.array([])

        for i in range(0, len(pos_B_bulk)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                    BB_bulk_dot = np.append(BB_bulk_dot, np.sum(np.cos(ang_B_bulk[i]-ang_B_dense[BB_bulk_nlist.point_indices[loc]])))
            else:
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                BB_bulk_dot = np.append(BB_bulk_dot, 0)


        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])
        AA_int_dot = np.array([])

        for i in range(0, len(pos_A_int)):
            if i in AA_int_nlist.query_point_indices:
                if i not in AA_int_neigh_ind:
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]
                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
                    AA_int_dot = np.append(AA_int_dot, np.sum(np.cos(ang_A_int[i]-ang_A[AA_int_nlist.point_indices[loc]])))
            else:
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
                AA_int_dot = np.append(AA_int_dot, 0)

        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])
        AB_int_dot = np.array([])

        for i in range(0, len(pos_B_int)):
            if i in AB_int_nlist.query_point_indices:
                if i not in AB_int_neigh_ind:
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]
                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
                    AB_int_dot = np.append(AB_int_dot, np.sum(np.cos(ang_B_int[i]-ang_A[AB_int_nlist.point_indices[loc]])))
            else:
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
                AB_int_dot = np.append(AB_int_dot, 0)

        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])
        BA_int_dot = np.array([])

        for i in range(0, len(pos_A_int)):
            if i in BA_int_nlist.query_point_indices:
                if i not in BA_int_neigh_ind:
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]

                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
                    BA_int_dot = np.append(BA_int_dot, np.sum(np.cos(ang_A_int[i]-ang_B[BA_int_nlist.point_indices[loc]])))
            else:
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
                BA_int_dot = np.append(BA_int_dot, 0)

        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])
        BB_int_dot = np.array([])

        for i in range(0, len(pos_B_int)):
            if i in BB_int_nlist.query_point_indices:
                if i not in BB_int_neigh_ind:
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]
                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
                    BB_int_dot = np.append(BB_int_dot, np.sum(np.cos(ang_B_int[i]-ang_B[BB_int_nlist.point_indices[loc]])))
            else:
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
                BB_int_dot = np.append(BB_int_dot, 0)


        allB_int_dot = BB_int_dot + AB_int_dot
        allB_int_num_neigh = BB_int_num_neigh + AB_int_num_neigh

        Ball_int_dot = np.append(BA_int_dot, BB_int_dot)
        Ball_int_num_neigh = np.append(BA_int_num_neigh, BB_int_num_neigh)

        allA_int_dot = AA_int_dot + BA_int_dot
        allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh

        Aall_int_dot = np.append(AB_int_dot, AA_int_dot)
        Aall_int_num_neigh = np.append(AB_int_num_neigh, AA_int_num_neigh)



        allB_bulk_dot = BB_bulk_dot + AB_bulk_dot
        allB_bulk_num_neigh = BB_bulk_num_neigh + AB_bulk_num_neigh

        Ball_bulk_dot = np.append(BA_bulk_dot, BB_bulk_dot)
        Ball_bulk_num_neigh = np.append(BA_bulk_num_neigh, BB_bulk_num_neigh)

        allA_bulk_dot = AA_bulk_dot + BA_bulk_dot
        allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh

        Aall_bulk_dot = np.append(AB_bulk_dot, AA_bulk_dot)
        Aall_bulk_num_neigh = np.append(AB_bulk_num_neigh, AA_bulk_num_neigh)

        allall_int_dot = np.append(allA_int_dot, allB_int_dot)
        allall_int_num_neigh = np.append(allA_int_num_neigh, allB_int_num_neigh)

        allall_bulk_dot = np.append(allA_bulk_dot, allB_bulk_dot)
        allall_bulk_num_neigh = np.append(allA_bulk_num_neigh, allB_bulk_num_neigh)

        AA_dense_dot = np.append(AA_bulk_dot, AA_int_dot)
        AB_dense_dot = np.append(AB_bulk_dot, AB_int_dot)
        BA_dense_dot = np.append(BA_bulk_dot, BA_int_dot)
        BB_dense_dot = np.append(BB_bulk_dot, BB_int_dot)

        for i in range(0, len(allB_bulk_dot)):
            if allB_bulk_num_neigh[i]>0:
                allB_bulk_dot[i] = allB_bulk_dot[i]/allB_bulk_num_neigh[i]

        for i in range(0, len(allA_bulk_dot)):
            if allA_bulk_num_neigh[i]>0:
                allA_bulk_dot[i] = allA_bulk_dot[i]/allA_bulk_num_neigh[i]

        for i in range(0, len(allall_bulk_dot)):
            if allall_bulk_num_neigh[i]>0:
                allall_bulk_dot[i] = allall_bulk_dot[i]/allall_bulk_num_neigh[i]

        for i in range(0, len(Aall_bulk_dot)):
            if Aall_bulk_num_neigh[i]>0:
                Aall_bulk_dot[i] = Aall_bulk_dot[i]/Aall_bulk_num_neigh[i]

        for i in range(0, len(Ball_bulk_dot)):
            if Ball_bulk_num_neigh[i]>0:
                Ball_bulk_dot[i] = Ball_bulk_dot[i]/Ball_bulk_num_neigh[i]

        for i in range(0, len(allB_int_dot)):
            if allB_int_num_neigh[i]>0:
                allB_int_dot[i] = allB_int_dot[i]/allB_int_num_neigh[i]

        for i in range(0, len(allA_int_dot)):
            if allA_int_num_neigh[i]>0:
                allA_int_dot[i] = allA_int_dot[i]/allA_int_num_neigh[i]

        for i in range(0, len(allall_int_dot)):
            if allall_int_num_neigh[i]>0:
                allall_int_dot[i] = allall_int_dot[i]/allall_int_num_neigh[i]

        for i in range(0, len(Aall_int_dot)):
            if Aall_int_num_neigh[i]>0:
                Aall_int_dot[i] = Aall_int_dot[i]/Aall_int_num_neigh[i]

        for i in range(0, len(Ball_int_dot)):
            if Ball_int_num_neigh[i]>0:
                Ball_int_dot[i] = Ball_int_dot[i]/Ball_int_num_neigh[i]

        print(len(allB_bulk_dot))
        print(len(allA_bulk_dot))
        print(len(pos_A_bulk))
        print(len(pos_B_bulk))

        print(np.mean(allA_int_dot))
        print(np.mean(allB_int_dot))
        print(np.mean(allall_int_dot))
        print(np.mean(allA_bulk_dot))
        print(np.mean(allB_bulk_dot))
        print(np.mean(allall_bulk_dot))

        allB_bulk_num_neigh = AB_bulk_num_neigh + BB_bulk_num_neigh
        Ball_bulk_num_neigh = np.append(BA_bulk_num_neigh, BB_bulk_num_neigh)
        allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh
        Aall_bulk_num_neigh = np.append(AB_bulk_num_neigh, AA_bulk_num_neigh)

        allB_int_num_neigh = AB_int_num_neigh + BB_int_num_neigh
        Ball_int_num_neigh = np.append(BA_int_num_neigh, BB_int_num_neigh)
        allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh
        Aall_int_num_neigh = np.append(AB_int_num_neigh, AA_int_num_neigh)

        allall_bulk_num_neigh = np.append(allA_bulk_num_neigh, allB_bulk_num_neigh)
        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

        allall_int_num_neigh = np.append(allA_int_num_neigh, allB_int_num_neigh)
        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

        AA_dense_num_neigh = np.append(AA_bulk_num_neigh, AA_int_num_neigh)
        AB_dense_num_neigh = np.append(AB_bulk_num_neigh, AB_int_num_neigh)
        BA_dense_num_neigh = np.append(BA_bulk_num_neigh, BA_int_num_neigh)
        BB_dense_num_neigh = np.append(BB_bulk_num_neigh, BB_int_num_neigh)

        Aall_dense_dot = np.append(Aall_bulk_dot, Aall_int_dot)
        Ball_dense_dot = np.append(Ball_bulk_dot, Ball_int_dot)

        Aall_dense_num_neigh = np.append(Aall_bulk_num_neigh, Aall_int_num_neigh)
        Aall_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Aall_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])
        Ball_dense_num_neigh = np.append(Ball_bulk_num_neigh, Ball_int_num_neigh)
        Ball_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Ball_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        allA_dense_num_neigh = np.append(allA_bulk_num_neigh, allA_int_num_neigh)
        allA_dense_dot = np.append(allA_bulk_dot, allA_int_dot)
        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

        allB_dense_num_neigh = np.append(allB_bulk_num_neigh, allB_int_num_neigh)
        allB_dense_dot = np.append(allB_bulk_dot, allB_int_dot)
        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        allall_dense_num_neigh = np.append(allall_bulk_num_neigh, allall_int_num_neigh)
        allall_dense_dot = np.append(allall_bulk_dot, allall_int_dot)
        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

        neigh_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_num_neigh), 'std': np.std(allall_bulk_num_neigh)}, 'all-A': {'mean': np.mean(allA_bulk_num_neigh), 'std': np.std(allA_bulk_num_neigh)}, 'all-B': {'mean': np.mean(allB_bulk_num_neigh), 'std': np.std(allB_bulk_num_neigh)}, 'A-A': {'mean': np.mean(AA_bulk_num_neigh), 'std': np.std(AA_bulk_num_neigh)}, 'A-B': {'mean': np.mean(AB_bulk_num_neigh), 'std': np.std(AB_bulk_num_neigh)}, 'B-B': {'mean': np.mean(BB_bulk_num_neigh), 'std': np.std(BB_bulk_num_neigh)}}, 'int': {'all-all': {'mean': np.mean(allall_int_num_neigh), 'std': np.std(allall_int_num_neigh)}, 'all-A': {'mean': np.mean(allA_int_num_neigh), 'std': np.std(allA_int_num_neigh)}, 'all-B': {'mean': np.mean(allB_int_num_neigh), 'std': np.std(allB_int_num_neigh)}, 'A-A': {'mean': np.mean(AA_int_num_neigh), 'std': np.std(AA_int_num_neigh)}, 'A-B': {'mean': np.mean(AB_int_num_neigh), 'std': np.std(AB_int_num_neigh)}, 'B-B': {'mean': np.mean(BB_int_num_neigh), 'std': np.std(BB_int_num_neigh)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_num_neigh), 'std': np.std(allall_dense_num_neigh)}, 'all-A': {'mean': np.mean(allA_dense_num_neigh), 'std': np.std(allA_dense_num_neigh)}, 'all-B': {'mean': np.mean(allB_dense_num_neigh), 'std': np.std(allB_dense_num_neigh)}, 'A-A': {'mean': np.mean(AA_dense_num_neigh), 'std': np.std(AA_dense_num_neigh)}, 'A-B': {'mean': np.mean(AB_dense_num_neigh), 'std': np.std(AB_dense_num_neigh)}, 'B-B': {'mean': np.mean(BB_dense_num_neigh), 'std': np.std(BB_dense_num_neigh)}}}

        ori_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_dot), 'std': np.std(allall_bulk_dot)}, 'all-A': {'mean': np.mean(allA_bulk_dot), 'std': np.std(allA_bulk_dot)}, 'all-B': {'mean': np.mean(allB_bulk_dot), 'std': np.std(allB_bulk_dot)}, 'A-A': {'mean': np.mean(AA_bulk_dot), 'std': np.std(AA_bulk_dot)}, 'A-B': {'mean': np.mean(AB_bulk_dot), 'std': np.std(AB_bulk_dot)}, 'B-B': {'mean': np.mean(BB_bulk_dot), 'std': np.std(BB_bulk_dot)}}, 'int': {'all-all': {'mean': np.mean(allall_int_dot), 'std': np.std(allall_int_dot)}, 'all-A': {'mean': np.mean(allA_int_dot), 'std': np.std(allA_int_dot)}, 'all-B': {'mean': np.mean(allB_int_dot), 'std': np.std(allB_int_dot)}, 'A-A': {'mean': np.mean(AA_int_dot), 'std': np.std(AA_int_dot)}, 'A-B': {'mean': np.mean(AB_int_dot), 'std': np.std(AB_int_dot)}, 'B-B': {'mean': np.mean(BB_int_dot), 'std': np.std(BB_int_dot)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_dot), 'std': np.std(allall_dense_dot)}, 'all-A': {'mean': np.mean(allA_dense_dot), 'std': np.std(allA_dense_dot)}, 'all-B': {'mean': np.mean(allB_dense_dot), 'std': np.std(allB_dense_dot)}, 'A-A': {'mean': np.mean(AA_dense_dot), 'std': np.std(AA_dense_dot)}, 'A-B': {'mean': np.mean(AB_dense_dot), 'std': np.std(AB_dense_dot)}, 'B-B': {'mean': np.mean(BB_dense_dot), 'std': np.std(BB_dense_dot)}}}


        neigh_plot_dict = {'all-all': {'neigh': allall_dense_num_neigh, 'ori': allall_dense_dot, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}, 'all-A': {'neigh': allA_dense_num_neigh, 'ori': allA_dense_dot, 'x': allA_dense_pos_x, 'y': allA_dense_pos_y}, 'all-B': {'neigh': allB_dense_num_neigh, 'ori': allB_dense_dot, 'x': allB_dense_pos_x, 'y': allB_dense_pos_y}, 'A-all': {'neigh': Aall_dense_num_neigh, 'ori': Aall_dense_dot, 'x': Aall_dense_pos_x, 'y': Aall_dense_pos_y}, 'B-all': {'neigh': Ball_dense_num_neigh, 'ori': Ball_dense_dot, 'x': Ball_dense_pos_x, 'y': Ball_dense_pos_y}}

        return neigh_stat_dict, ori_stat_dict, neigh_plot_dict
    def interparticle_pressure_nlist(self):


        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)
        distances = np.array([])
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        bulk_area = phase_count_dict['bulk'] * (self.sizeBin**2)
        int_area = phase_count_dict['int'] * (self.sizeBin**2)
        gas_area = phase_count_dict['gas'] * (self.sizeBin**2)
        dense_area = bulk_area + int_area

        peNet_int = self.average_activity(part_ids = phase_part_dict['int']['all'])

        if bulk_area > 0:
            num_dens_mean = len(phase_part_dict['bulk']['all'])/bulk_area
            num_dens_A_mean = len(phase_part_dict['bulk']['A'])/bulk_area
            num_dens_B_mean = len(phase_part_dict['bulk']['B'])/bulk_area
        else:
            num_dens_mean = 0
            num_dens_A_mean = 0
            num_dens_B_mean = 0

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_gas_int = self.pos[phase_part_dict['gas_int']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_gas_int = self.pos[phase_part_dict['gas_int']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas_int = self.pos[phase_part_dict['gas_int']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]


        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))
        #system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))

        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        #allall_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_bulk), query_args).toNeighborList()

        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])

        SigXX_AA_bulk_part=np.zeros(len(pos_A_bulk))
        SigXY_AA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYX_AA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYY_AA_bulk_part=np.zeros(len(pos_A_bulk))

        SigXX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigXY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))


        for i in range(0, len(pos_A_bulk)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        difx_test = pos_A_bulk[i,0]-pos_A_dense[AA_bulk_nlist.point_indices[loc]][:,0]
                        dify_test = pos_A_bulk[i,1]-pos_A_dense[AA_bulk_nlist.point_indices[loc]][:,1]
                        difr_test = (difx_test ** 2 + dify_test ** 2 ) **0.5

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AA_bulk_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_bulk_part[i] += np.sum(SigXX)
                        SigYY_AA_bulk_part[i] += np.sum(SigYY)
                        SigXY_AA_bulk_part[i] += np.sum(SigXY)
                        SigYX_AA_bulk_part[i] += np.sum(SigYX)

                        SigXX_AA_bulk_part_num[i] += len(SigXX)
                        SigYY_AA_bulk_part_num[i] += len(SigYY)
                        SigXY_AA_bulk_part_num[i] += len(SigXY)
                        SigYX_AA_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_bulk[i][0], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_bulk[i][1], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_bulk_part[i] = SigXX
                        SigYY_AA_bulk_part[i] = SigYY
                        SigXY_AA_bulk_part[i] = SigXY
                        SigYX_AA_bulk_part[i] = SigYX

                        SigXX_AA_bulk_part_num[i] += 1
                        SigYY_AA_bulk_part_num[i] += 1
                        SigXY_AA_bulk_part_num[i] += 1
                        SigYX_AA_bulk_part_num[i] += 1

                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
            else:
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))

        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])

        SigXX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part=np.zeros(len(pos_A_bulk))

        SigXX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))


        for i in range(0, len(pos_A_bulk)):
            if i in BA_bulk_nlist.query_point_indices:
                if i not in BA_bulk_neigh_ind:
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BA_bulk_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_bulk_part[i] += np.sum(SigXX)
                        SigYY_BA_bulk_part[i] += np.sum(SigYY)
                        SigXY_BA_bulk_part[i] += np.sum(SigXY)
                        SigYX_BA_bulk_part[i] += np.sum(SigYX)

                        SigXX_BA_bulk_part_num[i] += len(SigXX)
                        SigYY_BA_bulk_part_num[i] += len(SigYY)
                        SigXY_BA_bulk_part_num[i] += len(SigXY)
                        SigYX_BA_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_bulk[i][0], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_bulk[i][1], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_bulk_part[i] = SigXX
                        SigYY_BA_bulk_part[i] = SigYY
                        SigXY_BA_bulk_part[i] = SigXY
                        SigYX_BA_bulk_part[i] = SigYX

                        SigXX_BA_bulk_part_num[i] += 1
                        SigYY_BA_bulk_part_num[i] += 1
                        SigXY_BA_bulk_part_num[i] += 1
                        SigYX_BA_bulk_part_num[i] += 1

                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
            else:
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))

        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])

        SigXX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))


        for i in range(0, len(pos_B_bulk)):
            if i in AB_bulk_nlist.query_point_indices:
                if i not in AB_bulk_neigh_ind:
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AB_bulk_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_bulk_part[i] += np.sum(SigXX)
                        SigYY_AB_bulk_part[i] += np.sum(SigYY)
                        SigXY_AB_bulk_part[i] += np.sum(SigXY)
                        SigYX_AB_bulk_part[i] += np.sum(SigYX)

                        SigXX_AB_bulk_part_num[i] += len(SigXX)
                        SigYY_AB_bulk_part_num[i] += len(SigYY)
                        SigXY_AB_bulk_part_num[i] += len(SigXY)
                        SigYX_AB_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_bulk[i][0], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_bulk[i][1], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_bulk_part[i] += SigXX
                        SigYY_AB_bulk_part[i] += SigYY
                        SigXY_AB_bulk_part[i] += SigXY
                        SigYX_AB_bulk_part[i] += SigYX

                        SigXX_AB_bulk_part_num[i] += 1
                        SigYY_AB_bulk_part_num[i] += 1
                        SigXY_AB_bulk_part_num[i] += 1
                        SigYX_AB_bulk_part_num[i] += 1

                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))

        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])

        SigXX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))


        for i in range(0, len(pos_B_bulk)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BB_bulk_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_bulk_part[i] += np.sum(SigXX)
                        SigYY_BB_bulk_part[i] += np.sum(SigYY)
                        SigXY_BB_bulk_part[i] += np.sum(SigXY)
                        SigYX_BB_bulk_part[i] += np.sum(SigYX)

                        SigXX_BB_bulk_part_num[i] += len(SigXX)
                        SigYY_BB_bulk_part_num[i] += len(SigYY)
                        SigXY_BB_bulk_part_num[i] += len(SigXY)
                        SigYX_BB_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_bulk[i][0], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_bulk[i][1], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_bulk_part[i] += SigXX
                        SigYY_BB_bulk_part[i] += SigYY
                        SigXY_BB_bulk_part[i] += SigXY
                        SigYX_BB_bulk_part[i] += SigYX

                        SigXX_BB_bulk_part_num[i] += 1
                        SigYY_BB_bulk_part_num[i] += 1
                        SigXY_BB_bulk_part_num[i] += 1
                        SigYX_BB_bulk_part_num[i] += 1

                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
            else:
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
        """
        allall_bulk_neigh_ind = np.array([], dtype=int)
        allall_bulk_num_neigh = np.array([])

        SigXX_allall_bulk_part=np.zeros(len(pos_bulk))
        SigXY_allall_bulk_part=np.zeros(len(pos_bulk))
        SigYX_allall_bulk_part=np.zeros(len(pos_bulk))
        SigYY_allall_bulk_part=np.zeros(len(pos_bulk))

        SigXX_allall_bulk_part_num=np.zeros(len(pos_bulk))
        SigXY_allall_bulk_part_num=np.zeros(len(pos_bulk))
        SigYX_allall_bulk_part_num=np.zeros(len(pos_bulk))
        SigYY_allall_bulk_part_num=np.zeros(len(pos_bulk))


        for i in range(0, len(pos_bulk)):
            if i in allall_bulk_nlist.query_point_indices:
                if i not in allall_bulk_neigh_ind:
                    loc = np.where(allall_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_bulk[i]

                        difx_test = pos_bulk[i,0]-pos_dense[allall_bulk_nlist.point_indices[loc]][:,0]
                        dify_test = pos_bulk[i,1]-pos_dense[allall_bulk_nlist.point_indices[loc]][:,1]
                        difr_test = (difx_test ** 2 + dify_test ** 2 ) **0.5

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_dense[allall_bulk_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_allall_bulk_part[i] += np.sum(SigXX)
                        SigYY_allall_bulk_part[i] += np.sum(SigYY)
                        SigXY_allall_bulk_part[i] += np.sum(SigXY)
                        SigYX_allall_bulk_part[i] += np.sum(SigYX)

                        SigXX_allall_bulk_part_num[i] += len(SigXX)
                        SigYY_allall_bulk_part_num[i] += len(SigYY)
                        SigXY_allall_bulk_part_num[i] += len(SigXY)
                        SigYX_allall_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_bulk[i][0], pos_dense[allall_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_bulk[i][1], pos_dense[allall_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_allall_bulk_part[i] = SigXX
                        SigYY_allall_bulk_part[i] = SigYY
                        SigXY_allall_bulk_part[i] = SigXY
                        SigYX_allall_bulk_part[i] = SigYX

                        SigXX_allall_bulk_part_num[i] += 1
                        SigYY_allall_bulk_part_num[i] += 1
                        SigXY_allall_bulk_part_num[i] += 1
                        SigYX_allall_bulk_part_num[i] += 1

                    allall_bulk_num_neigh = np.append(allall_bulk_num_neigh, len(loc))
                    allall_bulk_neigh_ind = np.append(allall_bulk_neigh_ind, int(i))
            else:
                allall_bulk_num_neigh = np.append(allall_bulk_num_neigh, 0)
                allall_bulk_neigh_ind = np.append(allall_bulk_neigh_ind, int(i))
        """
        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])

        SigXX_AA_int_part=np.zeros(len(pos_A_int))
        SigXY_AA_int_part=np.zeros(len(pos_A_int))
        SigYX_AA_int_part=np.zeros(len(pos_A_int))
        SigYY_AA_int_part=np.zeros(len(pos_A_int))

        SigXX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_AA_int_part_num=np.zeros(len(pos_A_int))


        for i in range(0, len(pos_A_int)):
            if i in AA_int_nlist.query_point_indices:
                if i not in AA_int_neigh_ind:
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AA_int_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_int_part[i] += np.sum(SigXX)
                        SigYY_AA_int_part[i] += np.sum(SigYY)
                        SigXY_AA_int_part[i] += np.sum(SigXY)
                        SigYX_AA_int_part[i] += np.sum(SigYX)

                        SigXX_AA_int_part_num[i] += len(SigXX)
                        SigYY_AA_int_part_num[i] += len(SigYY)
                        SigXY_AA_int_part_num[i] += len(SigXY)
                        SigYX_AA_int_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_int[i][0], pos_A[AA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_int[i][1], pos_A[AA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_int_part[i] = SigXX
                        SigYY_AA_int_part[i] = SigYY
                        SigXY_AA_int_part[i] = SigXY
                        SigYX_AA_int_part[i] = SigYX

                        SigXX_AA_int_part_num[i] += 1
                        SigYY_AA_int_part_num[i] += 1
                        SigXY_AA_int_part_num[i] += 1
                        SigYX_AA_int_part_num[i] += 1

                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
            else:
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))

        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])

        SigXX_BA_int_part=np.zeros(len(pos_A_int))
        SigXY_BA_int_part=np.zeros(len(pos_A_int))
        SigYX_BA_int_part=np.zeros(len(pos_A_int))
        SigYY_BA_int_part=np.zeros(len(pos_A_int))

        SigXX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_BA_int_part_num=np.zeros(len(pos_A_int))


        for i in range(0, len(pos_A_int)):
            if i in BA_int_nlist.query_point_indices:
                if i not in BA_int_neigh_ind:
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BA_int_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_int_part[i] += np.sum(SigXX)
                        SigYY_BA_int_part[i] += np.sum(SigYY)
                        SigXY_BA_int_part[i] += np.sum(SigXY)
                        SigYX_BA_int_part[i] += np.sum(SigYX)

                        SigXX_BA_int_part_num[i] += len(SigXX)
                        SigYY_BA_int_part_num[i] += len(SigYY)
                        SigXY_BA_int_part_num[i] += len(SigXY)
                        SigYX_BA_int_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_int[i][0], pos_B[BA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_int[i][1], pos_B[BA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_int_part[i] = SigXX
                        SigYY_BA_int_part[i] = SigYY
                        SigXY_BA_int_part[i] = SigXY
                        SigYX_BA_int_part[i] = SigYX

                        SigXX_BA_int_part_num[i] += 1
                        SigYY_BA_int_part_num[i] += 1
                        SigXY_BA_int_part_num[i] += 1
                        SigYX_BA_int_part_num[i] += 1

                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
            else:
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))

        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])

        SigXX_AB_int_part=np.zeros(len(pos_B_int))
        SigXY_AB_int_part=np.zeros(len(pos_B_int))
        SigYX_AB_int_part=np.zeros(len(pos_B_int))
        SigYY_AB_int_part=np.zeros(len(pos_B_int))

        SigXX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_AB_int_part_num=np.zeros(len(pos_B_int))


        for i in range(0, len(pos_B_int)):
            if i in AB_int_nlist.query_point_indices:
                if i not in AB_int_neigh_ind:
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AB_int_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_int_part[i] += np.sum(SigXX)
                        SigYY_AB_int_part[i] += np.sum(SigYY)
                        SigXY_AB_int_part[i] += np.sum(SigXY)
                        SigYX_AB_int_part[i] += np.sum(SigYX)

                        SigXX_AB_int_part_num[i] += len(SigXX)
                        SigYY_AB_int_part_num[i] += len(SigYY)
                        SigXY_AB_int_part_num[i] += len(SigXY)
                        SigYX_AB_int_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_int[i][0], pos_A[AB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_int[i][1], pos_A[AB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_int_part[i] += SigXX
                        SigYY_AB_int_part[i] += SigYY
                        SigXY_AB_int_part[i] += SigXY
                        SigYX_AB_int_part[i] += SigYX

                        SigXX_AB_int_part_num[i] += 1
                        SigYY_AB_int_part_num[i] += 1
                        SigXY_AB_int_part_num[i] += 1
                        SigYX_AB_int_part_num[i] += 1

                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
            else:
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))

        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])

        SigXX_BB_int_part=np.zeros(len(pos_B_int))
        SigXY_BB_int_part=np.zeros(len(pos_B_int))
        SigYX_BB_int_part=np.zeros(len(pos_B_int))
        SigYY_BB_int_part=np.zeros(len(pos_B_int))

        SigXX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_BB_int_part_num=np.zeros(len(pos_B_int))


        for i in range(0, len(pos_B_int)):
            if i in BB_int_nlist.query_point_indices:
                if i not in BB_int_neigh_ind:
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BB_int_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_int_part[i] += np.sum(SigXX)
                        SigYY_BB_int_part[i] += np.sum(SigYY)
                        SigXY_BB_int_part[i] += np.sum(SigXY)
                        SigYX_BB_int_part[i] += np.sum(SigYX)

                        SigXX_BB_int_part_num[i] += len(SigXX)
                        SigYY_BB_int_part_num[i] += len(SigYY)
                        SigXY_BB_int_part_num[i] += len(SigXY)
                        SigYX_BB_int_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_int[i][0], pos_B[BB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_int[i][1], pos_B[BB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_int_part[i] += SigXX
                        SigYY_BB_int_part[i] += SigYY
                        SigXY_BB_int_part[i] += SigXY
                        SigYX_BB_int_part[i] += SigYX

                        SigXX_BB_int_part_num[i] += 1
                        SigYY_BB_int_part_num[i] += 1
                        SigXY_BB_int_part_num[i] += 1
                        SigYX_BB_int_part_num[i] += 1

                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
            else:
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))

        system_A_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_gas_int))
        system_B_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_gas_int))

        AA_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        AB_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()
        BA_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        BB_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()

        AA_gas_neigh_ind = np.array([], dtype=int)
        AA_gas_num_neigh = np.array([])

        SigXX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part_num=np.zeros(len(pos_A_gas))


        for i in range(0, len(pos_A_gas)):
            if i in AA_gas_nlist.query_point_indices:
                if i not in AA_gas_neigh_ind:
                    loc = np.where(AA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AA_gas_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_gas_part[i] += np.sum(SigXX)
                        SigYY_AA_gas_part[i] += np.sum(SigYY)
                        SigXY_AA_gas_part[i] += np.sum(SigXY)
                        SigYX_AA_gas_part[i] += np.sum(SigYX)

                        SigXX_AA_gas_part_num[i] += len(SigXX)
                        SigYY_AA_gas_part_num[i] += len(SigYY)
                        SigXY_AA_gas_part_num[i] += len(SigXY)
                        SigYX_AA_gas_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_gas[i][0], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_gas[i][1], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AA_gas_part[i] = SigXX
                        SigYY_AA_gas_part[i] = SigYY
                        SigXY_AA_gas_part[i] = SigXY
                        SigYX_AA_gas_part[i] = SigYX

                        SigXX_AA_gas_part_num[i] += 1
                        SigYY_AA_gas_part_num[i] += 1
                        SigXY_AA_gas_part_num[i] += 1
                        SigYX_AA_gas_part_num[i] += 1

                    AA_gas_num_neigh = np.append(AA_gas_num_neigh, len(loc))
                    AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))
            else:
                AA_gas_num_neigh = np.append(AA_gas_num_neigh, 0)
                AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))

        BA_gas_neigh_ind = np.array([], dtype=int)
        BA_gas_num_neigh = np.array([])

        SigXX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part_num=np.zeros(len(pos_A_gas))


        for i in range(0, len(pos_A_gas)):
            if i in BA_gas_nlist.query_point_indices:
                if i not in BA_gas_neigh_ind:
                    loc = np.where(BA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BA_gas_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_gas_part[i] += np.sum(SigXX)
                        SigYY_BA_gas_part[i] += np.sum(SigYY)
                        SigXY_BA_gas_part[i] += np.sum(SigXY)
                        SigYX_BA_gas_part[i] += np.sum(SigYX)

                        SigXX_BA_gas_part_num[i] += len(SigXX)
                        SigYY_BA_gas_part_num[i] += len(SigYY)
                        SigXY_BA_gas_part_num[i] += len(SigXY)
                        SigYX_BA_gas_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_A_gas[i][0], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_A_gas[i][1], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BA_gas_part[i] = SigXX
                        SigYY_BA_gas_part[i] = SigYY
                        SigXY_BA_gas_part[i] = SigXY
                        SigYX_BA_gas_part[i] = SigYX

                        SigXX_BA_gas_part_num[i] += 1
                        SigYY_BA_gas_part_num[i] += 1
                        SigXY_BA_gas_part_num[i] += 1
                        SigYX_BA_gas_part_num[i] += 1

                    BA_gas_num_neigh = np.append(BA_gas_num_neigh, len(loc))
                    BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))
            else:
                BA_gas_num_neigh = np.append(BA_gas_num_neigh, 0)
                BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))

        AB_gas_neigh_ind = np.array([], dtype=int)
        AB_gas_num_neigh = np.array([])

        SigXX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part_num=np.zeros(len(pos_B_gas))


        for i in range(0, len(pos_B_gas)):
            if i in AB_gas_nlist.query_point_indices:
                if i not in AB_gas_neigh_ind:
                    loc = np.where(AB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AB_gas_nlist.point_indices[loc]], difxy=True)

                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_gas_part[i] += np.sum(SigXX)
                        SigYY_AB_gas_part[i] += np.sum(SigYY)
                        SigXY_AB_gas_part[i] += np.sum(SigXY)
                        SigYX_AB_gas_part[i] += np.sum(SigYX)

                        SigXX_AB_gas_part_num[i] += len(SigXX)
                        SigYY_AB_gas_part_num[i] += len(SigYY)
                        SigXY_AB_gas_part_num[i] += len(SigXY)
                        SigYX_AB_gas_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_gas[i][0], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_gas[i][1], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_AB_gas_part[i] += SigXX
                        SigYY_AB_gas_part[i] += SigYY
                        SigXY_AB_gas_part[i] += SigXY
                        SigYX_AB_gas_part[i] += SigYX

                        SigXX_AB_gas_part_num[i] += 1
                        SigYY_AB_gas_part_num[i] += 1
                        SigXY_AB_gas_part_num[i] += 1
                        SigYX_AB_gas_part_num[i] += 1

                    AB_gas_num_neigh = np.append(AB_gas_num_neigh, len(loc))
                    AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))
            else:
                AB_gas_num_neigh = np.append(AB_gas_num_neigh, 0)
                AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))

        BB_gas_neigh_ind = np.array([], dtype=int)
        BB_gas_num_neigh = np.array([])

        SigXX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part_num=np.zeros(len(pos_B_gas))


        for i in range(0, len(pos_B_gas)):
            if i in BB_gas_nlist.query_point_indices:
                if i not in BB_gas_neigh_ind:
                    loc = np.where(BB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BB_gas_nlist.point_indices[loc]], difxy=True)


                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_gas_part[i] += np.sum(SigXX)
                        SigYY_BB_gas_part[i] += np.sum(SigYY)
                        SigXY_BB_gas_part[i] += np.sum(SigXY)
                        SigYX_BB_gas_part[i] += np.sum(SigYX)

                        SigXX_BB_gas_part_num[i] += len(SigXX)
                        SigYY_BB_gas_part_num[i] += len(SigYY)
                        SigXY_BB_gas_part_num[i] += len(SigXY)
                        SigYX_BB_gas_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist(pos_B_gas[i][0], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist(pos_B_gas[i][1], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        SigXX_BB_gas_part[i] += SigXX
                        SigYY_BB_gas_part[i] += SigYY
                        SigXY_BB_gas_part[i] += SigXY
                        SigYX_BB_gas_part[i] += SigYX

                        SigXX_BB_gas_part_num[i] += 1
                        SigYY_BB_gas_part_num[i] += 1
                        SigXY_BB_gas_part_num[i] += 1
                        SigYX_BB_gas_part_num[i] += 1

                    BB_gas_num_neigh = np.append(BB_gas_num_neigh, len(loc))
                    BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))
            else:
                BB_gas_num_neigh = np.append(BB_gas_num_neigh, 0)
                BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))


        #bulk

        #xx
        allB_bulk_SigXX_part = SigXX_BB_bulk_part + SigXX_AB_bulk_part
        allB_bulk_SigXX_part_num = SigXX_BB_bulk_part_num + SigXX_AB_bulk_part_num

        Ball_bulk_SigXX_part = np.append(SigXX_BA_bulk_part, SigXX_BB_bulk_part)
        Ball_bulk_SigXX_part_num = np.append(SigXX_BA_bulk_part_num, SigXX_BB_bulk_part_num)

        allA_bulk_SigXX_part = SigXX_AA_bulk_part + SigXX_BA_bulk_part
        allA_bulk_SigXX_part_num = SigXX_AA_bulk_part_num + SigXX_BA_bulk_part_num

        Aall_bulk_SigXX_part = np.append(SigXX_AB_bulk_part, SigXX_AA_bulk_part)
        Aall_bulk_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_bulk_part_num)

        allall_bulk_SigXX_part = np.append(allA_bulk_SigXX_part, allB_bulk_SigXX_part)
        allall_bulk_SigXX_part_num = np.append(allA_bulk_SigXX_part_num, allB_bulk_SigXX_part_num)

        #xy
        allB_bulk_SigXY_part = SigXY_BB_bulk_part + SigXY_AB_bulk_part
        allB_bulk_SigXY_part_num = SigXY_BB_bulk_part_num + SigXY_AB_bulk_part_num

        Ball_bulk_SigXY_part = np.append(SigXY_BA_bulk_part, SigXY_BB_bulk_part)
        Ball_bulk_SigXY_part_num = np.append(SigXY_BA_bulk_part_num, SigXY_BB_bulk_part_num)

        allA_bulk_SigXY_part = SigXY_AA_bulk_part + SigXY_BA_bulk_part
        allA_bulk_SigXY_part_num = SigXY_AA_bulk_part_num + SigXY_BA_bulk_part_num

        Aall_bulk_SigXY_part = np.append(SigXY_AB_bulk_part, SigXY_AA_bulk_part)
        Aall_bulk_SigXY_part_num = np.append(SigXY_AB_bulk_part_num, SigXY_AA_bulk_part_num)

        allall_bulk_SigXY_part = np.append(allA_bulk_SigXY_part, allB_bulk_SigXY_part)
        allall_bulk_SigXY_part_num = np.append(allA_bulk_SigXY_part_num, allB_bulk_SigXY_part_num)

        #yx
        allB_bulk_SigYX_part = SigYX_BB_bulk_part + SigYX_AB_bulk_part
        allB_bulk_SigYX_part_num = SigYX_BB_bulk_part_num + SigYX_AB_bulk_part_num

        Ball_bulk_SigYX_part = np.append(SigYX_BA_bulk_part, SigYX_BB_bulk_part)
        Ball_bulk_SigYX_part_num = np.append(SigYX_BA_bulk_part_num, SigYX_BB_bulk_part_num)

        allA_bulk_SigYX_part = SigYX_AA_bulk_part + SigYX_BA_bulk_part
        allA_bulk_SigYX_part_num = SigYX_AA_bulk_part_num + SigYX_BA_bulk_part_num

        Aall_bulk_SigYX_part = np.append(SigYX_AB_bulk_part, SigYX_AA_bulk_part)
        Aall_bulk_SigYX_part_num = np.append(SigYX_AB_bulk_part_num, SigYX_AA_bulk_part_num)

        allall_bulk_SigYX_part = np.append(allA_bulk_SigYX_part, allB_bulk_SigYX_part)
        allall_bulk_SigYX_part_num = np.append(allA_bulk_SigYX_part_num, allB_bulk_SigYX_part_num)

        #yy
        allB_bulk_SigYY_part = SigYY_BB_bulk_part + SigYY_AB_bulk_part
        allB_bulk_SigYY_part_num = SigYY_BB_bulk_part_num + SigYY_AB_bulk_part_num

        Ball_bulk_SigYY_part = np.append(SigYY_BA_bulk_part, SigYY_BB_bulk_part)
        Ball_bulk_SigYY_part_num = np.append(SigYY_BA_bulk_part_num, SigYY_BB_bulk_part_num)

        allA_bulk_SigYY_part = SigYY_AA_bulk_part + SigYY_BA_bulk_part
        allA_bulk_SigYY_part_num = SigYY_AA_bulk_part_num + SigYY_BA_bulk_part_num

        Aall_bulk_SigYY_part = np.append(SigYY_AB_bulk_part, SigYY_AA_bulk_part)
        Aall_bulk_SigYY_part_num = np.append(SigYY_AB_bulk_part_num, SigYY_AA_bulk_part_num)

        allall_bulk_SigYY_part = np.append(allA_bulk_SigYY_part, allB_bulk_SigYY_part)
        allall_bulk_SigYY_part_num = np.append(allA_bulk_SigYY_part_num, allB_bulk_SigYY_part_num)

        #interface

        #xx
        allB_int_SigXX_part = SigXX_BB_int_part + SigXX_AB_int_part
        allB_int_SigXX_part_num = SigXX_BB_int_part_num + SigXX_AB_int_part_num

        Ball_int_SigXX_part = np.append(SigXX_BA_int_part, SigXX_BB_int_part)
        Ball_int_SigXX_part_num = np.append(SigXX_BA_int_part_num, SigXX_BB_int_part_num)

        allA_int_SigXX_part = SigXX_AA_int_part + SigXX_BA_int_part
        allA_int_SigXX_part_num = SigXX_AA_int_part_num + SigXX_BA_int_part_num

        Aall_int_SigXX_part = np.append(SigXX_AB_int_part, SigXX_AA_int_part)
        Aall_int_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_int_part_num)

        allall_int_SigXX_part = np.append(allA_int_SigXX_part, allB_int_SigXX_part)
        allall_int_SigXX_part_num = np.append(allA_int_SigXX_part_num, allB_int_SigXX_part_num)

        #xy
        allB_int_SigXY_part = SigXY_BB_int_part + SigXY_AB_int_part
        allB_int_SigXY_part_num = SigXY_BB_int_part_num + SigXY_AB_int_part_num

        Ball_int_SigXY_part = np.append(SigXY_BA_int_part, SigXY_BB_int_part)
        Ball_int_SigXY_part_num = np.append(SigXY_BA_int_part_num, SigXY_BB_int_part_num)

        allA_int_SigXY_part = SigXY_AA_int_part + SigXY_BA_int_part
        allA_int_SigXY_part_num = SigXY_AA_int_part_num + SigXY_BA_int_part_num

        Aall_int_SigXY_part = np.append(SigXY_AB_int_part, SigXY_AA_int_part)
        Aall_int_SigXY_part_num = np.append(SigXY_AB_int_part_num, SigXY_AA_int_part_num)

        allall_int_SigXY_part = np.append(allA_int_SigXY_part, allB_int_SigXY_part)
        allall_int_SigXY_part_num = np.append(allA_int_SigXY_part_num, allB_int_SigXY_part_num)

        #yx
        allB_int_SigYX_part = SigYX_BB_int_part + SigYX_AB_int_part
        allB_int_SigYX_part_num = SigYX_BB_int_part_num + SigYX_AB_int_part_num

        Ball_int_SigYX_part = np.append(SigYX_BA_int_part, SigYX_BB_int_part)
        Ball_int_SigYX_part_num = np.append(SigYX_BA_int_part_num, SigYX_BB_int_part_num)

        allA_int_SigYX_part = SigYX_AA_int_part + SigYX_BA_int_part
        allA_int_SigYX_part_num = SigYX_AA_int_part_num + SigYX_BA_int_part_num

        Aall_int_SigYX_part = np.append(SigYX_AB_int_part, SigYX_AA_int_part)
        Aall_int_SigYX_part_num = np.append(SigYX_AB_int_part_num, SigYX_AA_int_part_num)

        allall_int_SigYX_part = np.append(allA_int_SigYX_part, allB_int_SigYX_part)
        allall_int_SigYX_part_num = np.append(allA_int_SigYX_part_num, allB_int_SigYX_part_num)

        #yy
        allB_int_SigYY_part = SigYY_BB_int_part + SigYY_AB_int_part
        allB_int_SigYY_part_num = SigYY_BB_int_part_num + SigYY_AB_int_part_num

        Ball_int_SigYY_part = np.append(SigYY_BA_int_part, SigYY_BB_int_part)
        Ball_int_SigYY_part_num = np.append(SigYY_BA_int_part_num, SigYY_BB_int_part_num)

        allA_int_SigYY_part = SigYY_AA_int_part + SigYY_BA_int_part
        allA_int_SigYY_part_num = SigYY_AA_int_part_num + SigYY_BA_int_part_num

        Aall_int_SigYY_part = np.append(SigYY_AB_int_part, SigYY_AA_int_part)
        Aall_int_SigYY_part_num = np.append(SigYY_AB_int_part_num, SigYY_AA_int_part_num)

        allall_int_SigYY_part = np.append(allA_int_SigYY_part, allB_int_SigYY_part)
        allall_int_SigYY_part_num = np.append(allA_int_SigYY_part_num, allB_int_SigYY_part_num)

        #gas

        #xx
        allB_gas_SigXX_part = SigXX_BB_gas_part + SigXX_AB_gas_part
        allB_gas_SigXX_part_num = SigXX_BB_gas_part_num + SigXX_AB_gas_part_num

        Ball_gas_SigXX_part = np.append(SigXX_BA_gas_part, SigXX_BB_gas_part)
        Ball_gas_SigXX_part_num = np.append(SigXX_BA_gas_part_num, SigXX_BB_gas_part_num)

        allA_gas_SigXX_part = SigXX_AA_gas_part + SigXX_BA_gas_part
        allA_gas_SigXX_part_num = SigXX_AA_gas_part_num + SigXX_BA_gas_part_num

        Aall_gas_SigXX_part = np.append(SigXX_AB_gas_part, SigXX_AA_gas_part)
        Aall_gas_SigXX_part_num = np.append(SigXX_AB_gas_part_num, SigXX_AA_gas_part_num)

        allall_gas_SigXX_part = np.append(allA_gas_SigXX_part, allB_gas_SigXX_part)
        allall_gas_SigXX_part_num = np.append(allA_gas_SigXX_part_num, allB_gas_SigXX_part_num)

        #xy
        allB_gas_SigXY_part = SigXY_BB_gas_part + SigXY_AB_gas_part
        allB_gas_SigXY_part_num = SigXY_BB_gas_part_num + SigXY_AB_gas_part_num

        Ball_gas_SigXY_part = np.append(SigXY_BA_gas_part, SigXY_BB_gas_part)
        Ball_gas_SigXY_part_num = np.append(SigXY_BA_gas_part_num, SigXY_BB_gas_part_num)

        allA_gas_SigXY_part = SigXY_AA_gas_part + SigXY_BA_gas_part
        allA_gas_SigXY_part_num = SigXY_AA_gas_part_num + SigXY_BA_gas_part_num

        Aall_gas_SigXY_part = np.append(SigXY_AB_gas_part, SigXY_AA_gas_part)
        Aall_gas_SigXY_part_num = np.append(SigXY_AB_gas_part_num, SigXY_AA_gas_part_num)

        allall_gas_SigXY_part = np.append(allA_gas_SigXY_part, allB_gas_SigXY_part)
        allall_gas_SigXY_part_num = np.append(allA_gas_SigXY_part_num, allB_gas_SigXY_part_num)

        #yx
        allB_gas_SigYX_part = SigYX_BB_gas_part + SigYX_AB_gas_part
        allB_gas_SigYX_part_num = SigYX_BB_gas_part_num + SigYX_AB_gas_part_num

        Ball_gas_SigYX_part = np.append(SigYX_BA_gas_part, SigYX_BB_gas_part)
        Ball_gas_SigYX_part_num = np.append(SigYX_BA_gas_part_num, SigYX_BB_gas_part_num)

        allA_gas_SigYX_part = SigYX_AA_gas_part + SigYX_BA_gas_part
        allA_gas_SigYX_part_num = SigYX_AA_gas_part_num + SigYX_BA_gas_part_num

        Aall_gas_SigYX_part = np.append(SigYX_AB_gas_part, SigYX_AA_gas_part)
        Aall_gas_SigYX_part_num = np.append(SigYX_AB_gas_part_num, SigYX_AA_gas_part_num)

        allall_gas_SigYX_part = np.append(allA_gas_SigYX_part, allB_gas_SigYX_part)
        allall_gas_SigYX_part_num = np.append(allA_gas_SigYX_part_num, allB_gas_SigYX_part_num)

        #yy
        allB_gas_SigYY_part = SigYY_BB_gas_part + SigYY_AB_gas_part
        allB_gas_SigYY_part_num = SigYY_BB_gas_part_num + SigYY_AB_gas_part_num

        Ball_gas_SigYY_part = np.append(SigYY_BA_gas_part, SigYY_BB_gas_part)
        Ball_gas_SigYY_part_num = np.append(SigYY_BA_gas_part_num, SigYY_BB_gas_part_num)

        allA_gas_SigYY_part = SigYY_AA_gas_part + SigYY_BA_gas_part
        allA_gas_SigYY_part_num = SigYY_AA_gas_part_num + SigYY_BA_gas_part_num

        Aall_gas_SigYY_part = np.append(SigYY_AB_gas_part, SigYY_AA_gas_part)
        Aall_gas_SigYY_part_num = np.append(SigYY_AB_gas_part_num, SigYY_AA_gas_part_num)

        allall_gas_SigYY_part = np.append(allA_gas_SigYY_part, allB_gas_SigYY_part)
        allall_gas_SigYY_part_num = np.append(allA_gas_SigYY_part_num, allB_gas_SigYY_part_num)

        allall_dense_SigXX_part = np.append(allall_bulk_SigXX_part, allall_int_SigXX_part)
        allall_dense_SigYX_part = np.append(allall_bulk_SigYX_part, allall_int_SigYX_part)
        allall_dense_SigXY_part = np.append(allall_bulk_SigXY_part, allall_int_SigXY_part)
        allall_dense_SigYY_part = np.append(allall_bulk_SigYY_part, allall_int_SigYY_part)

        allA_dense_SigXX_part = np.append(allA_bulk_SigXX_part, allA_int_SigXX_part)
        allA_dense_SigYX_part = np.append(allA_bulk_SigYX_part, allA_int_SigYX_part)
        allA_dense_SigXY_part = np.append(allA_bulk_SigXY_part, allA_int_SigXY_part)
        allA_dense_SigYY_part = np.append(allA_bulk_SigYY_part, allA_int_SigYY_part)

        allB_dense_SigXX_part = np.append(allB_bulk_SigXX_part, allB_int_SigXX_part)
        allB_dense_SigYX_part = np.append(allB_bulk_SigYX_part, allB_int_SigYX_part)
        allB_dense_SigXY_part = np.append(allB_bulk_SigXY_part, allB_int_SigXY_part)
        allB_dense_SigYY_part = np.append(allB_bulk_SigYY_part, allB_int_SigYY_part)

        SigXX_AA_dense_part = np.append(SigXX_AA_bulk_part, SigXX_AA_int_part)
        SigYX_AA_dense_part = np.append(SigYX_AA_bulk_part, SigYX_AA_int_part)
        SigXY_AA_dense_part = np.append(SigXY_AA_bulk_part, SigXY_AA_int_part)
        SigYY_AA_dense_part = np.append(SigYY_AA_bulk_part, SigYY_AA_int_part)

        SigXX_AB_dense_part = np.append(SigXX_AB_bulk_part, SigXX_AB_int_part)
        SigYX_AB_dense_part = np.append(SigYX_AB_bulk_part, SigYX_AB_int_part)
        SigXY_AB_dense_part = np.append(SigXY_AB_bulk_part, SigXY_AB_int_part)
        SigYY_AB_dense_part = np.append(SigYY_AB_bulk_part, SigYY_AB_int_part)

        SigXX_BB_dense_part = np.append(SigXX_BB_bulk_part, SigXX_BB_int_part)
        SigYX_BB_dense_part = np.append(SigYX_BB_bulk_part, SigYX_BB_int_part)
        SigXY_BB_dense_part = np.append(SigXY_BB_bulk_part, SigXY_BB_int_part)
        SigYY_BB_dense_part = np.append(SigYY_BB_bulk_part, SigYY_BB_int_part)


        print(bulk_area)
        print(gas_area)
        print(int_area)
        allall_bulk_int_press = np.sum(allall_bulk_SigXX_part + allall_bulk_SigYY_part)/(2*bulk_area)
        allall_gas_int_press = np.sum(allall_gas_SigXX_part + allall_gas_SigYY_part)/(2*gas_area)
        allall_int_int_press = np.sum(allall_int_SigXX_part + allall_int_SigYY_part)/(2*int_area)
        allall_dense_int_press = np.sum(allall_dense_SigXX_part + allall_dense_SigYY_part)/(2*dense_area)

        allA_bulk_int_press = np.sum(allA_bulk_SigXX_part + allA_bulk_SigYY_part)/(2*bulk_area)
        allA_gas_int_press = np.sum(allA_gas_SigXX_part + allA_gas_SigYY_part)/(2*gas_area)
        allA_int_int_press = np.sum(allA_int_SigXX_part + allA_int_SigYY_part)/(2*int_area)
        allA_dense_int_press = np.sum(allA_dense_SigXX_part + allA_dense_SigYY_part)/(2*dense_area)

        allB_bulk_int_press = np.sum(allB_bulk_SigXX_part + allB_bulk_SigYY_part)/(2*bulk_area)
        allB_gas_int_press = np.sum(allB_gas_SigXX_part + allB_gas_SigYY_part)/(2*gas_area)
        allB_int_int_press = np.sum(allB_int_SigXX_part + allB_int_SigYY_part)/(2*int_area)
        allB_dense_int_press = np.sum(allB_dense_SigXX_part + allB_dense_SigYY_part)/(2*dense_area)

        AA_bulk_int_press = np.sum(SigXX_AA_bulk_part + SigYY_AA_bulk_part)/(2*bulk_area)
        AA_gas_int_press = np.sum(SigXX_AA_gas_part + SigYY_AA_gas_part)/(2*gas_area)
        AA_int_int_press = np.sum(SigXX_AA_int_part + SigYY_AA_int_part)/(2*int_area)
        AA_dense_int_press = np.sum(SigXX_AA_dense_part + SigYY_AA_dense_part)/(2*dense_area)

        AB_bulk_int_press = np.sum(SigXX_AB_bulk_part + SigYY_AB_bulk_part)/(2*bulk_area)
        AB_gas_int_press = np.sum(SigXX_AB_gas_part + SigYY_AB_gas_part)/(2*gas_area)
        AB_int_int_press = np.sum(SigXX_AB_int_part + SigYY_AB_int_part)/(2*int_area)
        AB_dense_int_press = np.sum(SigXX_AB_dense_part + SigYY_AB_dense_part)/(2*dense_area)

        BB_bulk_int_press = np.sum(SigXX_BB_bulk_part + SigYY_BB_bulk_part)/(2*bulk_area)
        BB_gas_int_press = np.sum(SigXX_BB_gas_part + SigYY_BB_gas_part)/(2*gas_area)
        BB_int_int_press = np.sum(SigXX_BB_int_part + SigYY_BB_int_part)/(2*int_area)
        BB_dense_int_press = np.sum(SigXX_BB_dense_part + SigYY_BB_dense_part)/(2*dense_area)

        allall_bulk_shear_stress = np.sum(allall_bulk_SigXY_part)/(bulk_area)
        allall_gas_shear_stress = np.sum(allall_gas_SigXY_part)/(gas_area)
        allall_int_shear_stress = np.sum(allall_int_SigXY_part)/(int_area)
        allall_dense_shear_stress = np.sum(allall_dense_SigXY_part)/(dense_area)

        allA_bulk_shear_stress = np.sum(allA_bulk_SigXY_part)/(bulk_area)
        allA_gas_shear_stress = np.sum(allA_gas_SigXY_part)/(gas_area)
        allA_int_shear_stress = np.sum(allA_int_SigXY_part)/(int_area)
        allA_dense_shear_stress = np.sum(allA_dense_SigXY_part)/(dense_area)

        allB_bulk_shear_stress = np.sum(allB_bulk_SigXY_part)/(bulk_area)
        allB_gas_shear_stress = np.sum(allB_gas_SigXY_part)/(gas_area)
        allB_int_shear_stress = np.sum(allB_int_SigXY_part)/(int_area)
        allB_dense_shear_stress = np.sum(allB_dense_SigXY_part)/(dense_area)

        AA_bulk_shear_stress = np.sum(SigXY_AA_bulk_part)/(bulk_area)
        AA_gas_shear_stress = np.sum(SigXY_AA_gas_part)/(gas_area)
        AA_int_shear_stress = np.sum(SigXY_AA_int_part)/(int_area)
        AA_dense_shear_stress = np.sum(SigXY_AA_dense_part)/(dense_area)

        AB_bulk_shear_stress = np.sum(SigXY_AB_bulk_part)/(bulk_area)
        AB_gas_shear_stress = np.sum(SigXY_AB_gas_part)/(gas_area)
        AB_int_shear_stress = np.sum(SigXY_AB_int_part)/(int_area)
        AB_dense_shear_stress = np.sum(SigXY_AB_dense_part)/(dense_area)

        BB_bulk_shear_stress = np.sum(SigXY_BB_bulk_part)/(bulk_area)
        BB_gas_shear_stress = np.sum(SigXY_BB_gas_part)/(gas_area)
        BB_int_shear_stress = np.sum(SigXY_BB_int_part)/(int_area)
        BB_dense_shear_stress = np.sum(SigXY_BB_dense_part)/(dense_area)

        allall_shear_stress = np.append(allall_bulk_shear_stress, allall_int_shear_stress)
        allall_shear_stress = np.append(allall_shear_stress, allall_gas_shear_stress)

        allall_int_press = np.append(allall_bulk_int_press, allall_int_int_press)
        allall_int_press = np.append(allall_int_press, allall_gas_int_press)


        print(np.sum(allall_dense_SigXX_part + allall_dense_SigYY_part)/(2*dense_area))

        print(np.sum(allall_gas_SigXX_part + allall_gas_SigYY_part)/(2*gas_area))
        #print(np.sum(SigXX_allall_bulk_part + SigYY_allall_bulk_part)/(2*bulk_area))

        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

        allall_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
        allall_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])

        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

        Aall_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Aall_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        Ball_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Ball_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

        allall_pos_x = np.append(allall_dense_pos_x, allall_gas_pos_x)
        allall_pos_y = np.append(allall_dense_pos_y, allall_gas_pos_y)

        stress_stat_dict = {'bulk': {'all-all': {'XX': np.sum(allall_bulk_SigXX_part), 'XY': np.sum(allall_bulk_SigXY_part), 'YX': np.sum(allall_bulk_SigYX_part), 'YY': np.sum(allall_bulk_SigYY_part)}, 'all-A': {'XX': np.sum(allA_bulk_SigXX_part), 'XY': np.sum(allA_bulk_SigXY_part), 'YX': np.sum(allA_bulk_SigYX_part), 'YY': np.sum(allA_bulk_SigYY_part)}, 'all-B': {'XX': np.sum(allB_bulk_SigXX_part), 'XY': np.sum(allB_bulk_SigXY_part), 'YX': np.sum(allB_bulk_SigYX_part), 'YY': np.sum(allB_bulk_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_bulk_part), 'XY': np.sum(SigXY_AA_bulk_part), 'YX': np.sum(SigYX_AA_bulk_part), 'YY': np.sum(SigYY_AA_bulk_part)}, 'A-B': {'XX': np.sum(SigXX_AB_bulk_part), 'XY': np.sum(SigXY_AB_bulk_part), 'YX': np.sum(SigYX_AB_bulk_part), 'YY': np.sum(SigYY_AB_bulk_part)}, 'B-B': {'XX': np.sum(SigXX_BB_bulk_part), 'XY': np.sum(SigXY_BB_bulk_part), 'YX': np.sum(SigYX_BB_bulk_part), 'YY': np.sum(SigYY_BB_bulk_part)}}, 'gas': {'all-all': {'XX': np.sum(allall_gas_SigXX_part), 'XY': np.sum(allall_gas_SigXY_part), 'YX': np.sum(allall_gas_SigYX_part), 'YY': np.sum(allall_gas_SigYY_part)}, 'all-A': {'XX': np.sum(allA_gas_SigXX_part), 'XY': np.sum(allA_gas_SigXY_part), 'YX': np.sum(allA_gas_SigYX_part), 'YY': np.sum(allA_gas_SigYY_part)}, 'all-B': {'XX': np.sum(allB_gas_SigXX_part), 'XY': np.sum(allB_gas_SigXY_part), 'YX': np.sum(allB_gas_SigYX_part), 'YY': np.sum(allB_gas_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_gas_part), 'XY': np.sum(SigXY_AA_gas_part), 'YX': np.sum(SigYX_AA_gas_part), 'YY': np.sum(SigYY_AA_gas_part)}, 'A-B': {'XX': np.sum(SigXX_AB_gas_part), 'XY': np.sum(SigXY_AB_gas_part), 'YX': np.sum(SigYX_AB_gas_part), 'YY': np.sum(SigYY_AB_gas_part)}, 'B-B': {'XX': np.sum(SigXX_BB_gas_part), 'XY': np.sum(SigXY_BB_gas_part), 'YX': np.sum(SigYX_BB_gas_part), 'YY': np.sum(SigYY_BB_gas_part)}}, 'dense': {'all-all': {'XX': np.sum(allall_dense_SigXX_part), 'XY': np.sum(allall_dense_SigXY_part), 'YX': np.sum(allall_dense_SigYX_part), 'YY': np.sum(allall_dense_SigYY_part)}, 'all-A': {'XX': np.sum(allA_dense_SigXX_part), 'XY': np.sum(allA_dense_SigXY_part), 'YX': np.sum(allA_dense_SigYX_part), 'YY': np.sum(allA_dense_SigYY_part)}, 'all-B': {'XX': np.sum(allB_dense_SigXX_part), 'XY': np.sum(allB_dense_SigXY_part), 'YX': np.sum(allB_dense_SigYX_part), 'YY': np.sum(allB_dense_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_dense_part), 'XY': np.sum(SigXY_AA_dense_part), 'YX': np.sum(SigYX_AA_dense_part), 'YY': np.sum(SigYY_AA_dense_part)}, 'A-B': {'XX': np.sum(SigXX_AB_dense_part), 'XY': np.sum(SigXY_AB_dense_part), 'YX': np.sum(SigYX_AB_dense_part), 'YY': np.sum(SigYY_AB_dense_part)}, 'B-B': {'XX': np.sum(SigXX_BB_dense_part), 'XY': np.sum(SigXY_BB_dense_part), 'YX': np.sum(SigYX_BB_dense_part), 'YY': np.sum(SigYY_BB_dense_part)}}}

        #ori_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_dot), 'std': np.std(allall_bulk_dot)}, 'all-A': {'mean': np.mean(allA_bulk_dot), 'std': np.std(allA_bulk_dot)}, 'all-B': {'mean': np.mean(allB_bulk_dot), 'std': np.std(allB_bulk_dot)}, 'A-A': {'mean': np.mean(AA_bulk_dot), 'std': np.std(AA_bulk_dot)}, 'A-B': {'mean': np.mean(AB_bulk_dot), 'std': np.std(AB_bulk_dot)}, 'B-B': {'mean': np.mean(BB_bulk_dot), 'std': np.std(BB_bulk_dot)}}, 'int': {'all-all': {'mean': np.mean(allall_int_dot), 'std': np.std(allall_int_dot)}, 'all-A': {'mean': np.mean(allA_int_dot), 'std': np.std(allA_int_dot)}, 'all-B': {'mean': np.mean(allB_int_dot), 'std': np.std(allB_int_dot)}, 'A-A': {'mean': np.mean(AA_int_dot), 'std': np.std(AA_int_dot)}, 'A-B': {'mean': np.mean(AB_int_dot), 'std': np.std(AB_int_dot)}, 'B-B': {'mean': np.mean(BB_int_dot), 'std': np.std(BB_int_dot)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_dot), 'std': np.std(allall_dense_dot)}, 'all-A': {'mean': np.mean(allA_dense_dot), 'std': np.std(allA_dense_dot)}, 'all-B': {'mean': np.mean(allB_dense_dot), 'std': np.std(allB_dense_dot)}, 'A-A': {'mean': np.mean(AA_dense_dot), 'std': np.std(AA_dense_dot)}, 'A-B': {'mean': np.mean(AB_dense_dot), 'std': np.std(AB_dense_dot)}, 'B-B': {'mean': np.mean(BB_dense_dot), 'std': np.std(BB_dense_dot)}}}
        press_stat_dict = {'all-all': {'bulk': {'press': allall_bulk_int_press, 'shear': allall_bulk_shear_stress}, 'int': {'press': allall_int_int_press, 'shear': allall_int_shear_stress}, 'gas': {'press': allall_gas_int_press, 'shear': allall_gas_shear_stress}, 'dense': {'press': allall_dense_int_press, 'shear': allall_dense_shear_stress}}, 'all-A': {'bulk': {'press': allA_bulk_int_press, 'shear': allA_bulk_shear_stress}, 'int': {'press': allA_int_int_press, 'shear': allA_int_shear_stress}, 'gas': {'press': allA_gas_int_press, 'shear': allA_gas_shear_stress}, 'dense': {'press': allA_dense_int_press, 'shear': allA_dense_shear_stress}}, 'all-B': {'bulk': {'press': allB_bulk_int_press, 'shear': allB_bulk_shear_stress}, 'int': {'press': allB_int_int_press, 'shear': allB_int_shear_stress}, 'gas': {'press': allB_gas_int_press, 'shear': allB_gas_shear_stress}, 'dense': {'press': allB_dense_int_press, 'shear': allB_dense_shear_stress}}, 'A-A': {'bulk': {'press': AA_bulk_int_press, 'shear': AA_bulk_shear_stress}, 'int': {'press': AA_int_int_press, 'shear': AA_int_shear_stress}, 'gas': {'press': AA_gas_int_press, 'shear': AA_gas_shear_stress}, 'dense': {'press': AA_dense_int_press, 'shear': AA_dense_shear_stress}}, 'A-B': {'bulk': {'press': AB_bulk_int_press, 'shear': AB_bulk_shear_stress}, 'int': {'press': AB_int_int_press, 'shear': AB_int_shear_stress}, 'gas': {'press': AB_gas_int_press, 'shear': AB_gas_shear_stress}, 'dense': {'press': AB_dense_int_press, 'shear': AB_dense_shear_stress}}, 'B-B': {'bulk': {'press': BB_bulk_int_press, 'shear': BB_bulk_shear_stress}, 'int': {'press': BB_int_int_press, 'shear': BB_int_shear_stress}, 'gas': {'press': BB_gas_int_press, 'shear': BB_gas_shear_stress}, 'dense': {'press': BB_dense_int_press, 'shear': BB_dense_shear_stress}}}

        press_plot_dict = {'all-all': {'press': allall_int_press, 'shear': allall_shear_stress, 'x': allall_pos_x, 'y': allall_pos_y}}

        return stress_stat_dict, press_stat_dict, press_plot_dict

    def domain_size(self):

        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        system_A_dense = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_dense = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        #Compute cluster parameters using system_all neighbor list
        cl_A=freud.cluster.Cluster()                              #Define cluster
        cl_B=freud.cluster.Cluster()                              #Define cluster

        cl_A.compute(system_A_dense, neighbors={'r_max': self.r_cut, 'r_min': 0.1})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        cl_B.compute(system_B_dense, neighbors={'r_max': self.r_cut, 'r_min': 0.1})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance


        clp_A = freud.cluster.ClusterProperties()                 #Define cluster properties
        clp_B = freud.cluster.ClusterProperties()                 #Define cluster properties

        ids_A = cl_A.cluster_idx                                    # get id of each cluster
        ids_B = cl_B.cluster_idx                                    # get id of each cluster

        clp_A.compute(system_A_dense, ids_A)                            # Calculate cluster properties given cluster IDs
        clp_B.compute(system_B_dense, ids_B)                            # Calculate cluster properties given cluster IDs

        clust_size_A = clp_A.sizes                                  # find cluster sizes
        clust_size_B = clp_B.sizes                                  # find cluster sizes

        lcID_A = np.where(clust_size_A >= 1)[0]    #Identify largest cluster
        lcID_B = np.where(clust_size_B >= 1)[0]    #Identify largest cluster

        print(clust_size_A[lcID_A])
        print(clust_size_B[lcID_B])
        print(np.mean(clust_size_A[lcID_A]))
        print(np.mean(clust_size_B[lcID_B]))
        print(np.median(clust_size_A[lcID_A]))
        print(np.median(clust_size_B[lcID_B]))
        print(np.std(clust_size_A[lcID_A]))
        print(np.std(clust_size_B[lcID_B]))
        print(len(clust_size_A[lcID_A]))
        print(len(clust_size_B[lcID_B]))
        print(np.max(clust_size_A[lcID_A]))
        print(np.max(clust_size_B[lcID_B]))
        print(np.sum(clust_size_A[lcID_A])/len(pos_A_dense))
        print(np.sum(clust_size_B[lcID_B])/len(pos_B_dense))

        print(1-(len(clust_size_A[lcID_A])/len(pos_A_dense)))
        print(1-(len(clust_size_B[lcID_B])/len(pos_B_dense)))


        stop
        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])
        AA_bulk_dot = np.array([])

        for i in range(0, len(pos_A_dense)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
            else:
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))

        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])
        BB_bulk_dot = np.array([])

        for i in range(0, len(pos_B_dense)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                    BB_bulk_dot = np.append(BB_bulk_dot, np.sum(np.cos(ang_B_bulk[i]-ang_B_dense[BB_bulk_nlist.point_indices[loc]])))
            else:
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                BB_bulk_dot = np.append(BB_bulk_dot, 0)


        allB_bulk_num_neigh = AB_bulk_num_neigh + BB_bulk_num_neigh
        Ball_bulk_num_neigh = np.append(BA_bulk_num_neigh, BB_bulk_num_neigh)
        allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh
        Aall_bulk_num_neigh = np.append(AB_bulk_num_neigh, AA_bulk_num_neigh)

        allB_int_num_neigh = AB_int_num_neigh + BB_int_num_neigh
        Ball_int_num_neigh = np.append(BA_int_num_neigh, BB_int_num_neigh)
        allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh
        Aall_int_num_neigh = np.append(AB_int_num_neigh, AA_int_num_neigh)

        allall_bulk_num_neigh = np.append(allA_bulk_num_neigh, allB_bulk_num_neigh)
        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

        allall_int_num_neigh = np.append(allA_int_num_neigh, allB_int_num_neigh)
        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

        AA_dense_num_neigh = np.append(AA_bulk_num_neigh, AA_int_num_neigh)
        AB_dense_num_neigh = np.append(AB_bulk_num_neigh, AB_int_num_neigh)
        BA_dense_num_neigh = np.append(BA_bulk_num_neigh, BA_int_num_neigh)
        BB_dense_num_neigh = np.append(BB_bulk_num_neigh, BB_int_num_neigh)

        Aall_dense_num_neigh = np.append(Aall_bulk_num_neigh, Aall_int_num_neigh)
        Ball_dense_num_neigh = np.append(Ball_bulk_num_neigh, Ball_int_num_neigh)

        allA_dense_num_neigh = np.append(allA_bulk_num_neigh, allA_int_num_neigh)
        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

        allB_dense_num_neigh = np.append(allB_bulk_num_neigh, allB_int_num_neigh)
        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        allall_dense_num_neigh = np.append(allall_bulk_num_neigh, allall_int_num_neigh)
        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

        neigh_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_num_neigh), 'std': np.std(allall_bulk_num_neigh)}, 'all-A': {'mean': np.mean(allA_bulk_num_neigh), 'std': np.std(allA_bulk_num_neigh)}, 'all-B': {'mean': np.mean(allB_bulk_num_neigh), 'std': np.std(allB_bulk_num_neigh)}, 'A-A': {'mean': np.mean(AA_bulk_num_neigh), 'std': np.std(AA_bulk_num_neigh)}, 'A-B': {'mean': np.mean(AB_bulk_num_neigh), 'std': np.std(AB_bulk_num_neigh)}, 'B-B': {'mean': np.mean(BB_bulk_num_neigh), 'std': np.std(BB_bulk_num_neigh)}}, 'int': {'all-all': {'mean': np.mean(allall_int_num_neigh), 'std': np.std(allall_int_num_neigh)}, 'all-A': {'mean': np.mean(allA_int_num_neigh), 'std': np.std(allA_int_num_neigh)}, 'all-B': {'mean': np.mean(allB_int_num_neigh), 'std': np.std(allB_int_num_neigh)}, 'A-A': {'mean': np.mean(AA_int_num_neigh), 'std': np.std(AA_int_num_neigh)}, 'A-B': {'mean': np.mean(AB_int_num_neigh), 'std': np.std(AB_int_num_neigh)}, 'B-B': {'mean': np.mean(BB_int_num_neigh), 'std': np.std(BB_int_num_neigh)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_num_neigh), 'std': np.std(allall_dense_num_neigh)}, 'all-A': {'mean': np.mean(allA_dense_num_neigh), 'std': np.std(allA_dense_num_neigh)}, 'all-B': {'mean': np.mean(allB_dense_num_neigh), 'std': np.std(allB_dense_num_neigh)}, 'A-A': {'mean': np.mean(AA_dense_num_neigh), 'std': np.std(AA_dense_num_neigh)}, 'A-B': {'mean': np.mean(AB_dense_num_neigh), 'std': np.std(AB_dense_num_neigh)}, 'B-B': {'mean': np.mean(BB_dense_num_neigh), 'std': np.std(BB_dense_num_neigh)}}}

        neigh_plot_dict = {'all-all': {'neigh': allall_dense_num_neigh, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}, 'all-A': {'neigh': allA_dense_num_neigh, 'x': allA_dense_pos_x, 'y': allA_dense_pos_y}, 'all-B': {'neigh': allB_dense_num_neigh, 'x': allB_dense_pos_x, 'y': allB_dense_pos_y}, 'A-all': {'neigh': Aall_dense_num_neigh, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}, 'B-all': {'neigh': Ball_dense_num_neigh, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}}

        return neigh_stat_dict, neigh_plot_dict

    def hexatic_order(self):

        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        hex_order = freud.order.Hexatic(k=6)

        # Compute hexatic order for 6 nearest neighbors
        hex_order.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)
        #hex_order.compute(system=(f_box, pos_A), neighbors=allA_bulk_nlist)

        psi_k = hex_order.particle_order

        #Average hexatic order parameter
        avg_psi_k = np.mean(psi_k)

        # Create an array of angles relative to the average
        order_param = np.abs(psi_k)

        #Calculate relative bond orientation of crystal domains
        relative_angles = np.angle(psi_k)

        #Since hexagonal domains, translate angle to 0 to pi/3 radians
        for g in range(0, len(relative_angles)):
            if relative_angles[g]<(-2*np.pi/3):
                relative_angles[g]+=(np.pi)
            elif (-2*np.pi/3)<=relative_angles[g]<(-np.pi/3):
                relative_angles[g]+=(2*np.pi/3)
            elif (-np.pi/3)<=relative_angles[g]<0:
                relative_angles[g]+=(np.pi/3)
            elif np.pi/3<relative_angles[g]<=(2*np.pi/3):
                relative_angles[g]-=(np.pi/3)
            elif (2*np.pi/3) < relative_angles[g]:
                relative_angles[g]-=(2*np.pi/3)

        hexatic_order_dict = {'order': order_param, 'theta': relative_angles}
        return hexatic_order_dict

    def translational_order(self):

        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        #Compute translational order parameter
        trans_order = freud.order.Translational(k=6)

        trans_order.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)

        trans_param = np.abs(trans_order.particle_order)

        return trans_param

    def steinhardt_order(self):

        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        #Compute Steinhardt order parameter
        ql = freud.order.Steinhardt(l=6)

        ql.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)

        stein_param = np.abs(ql.particle_order)

        return stein_param

    def nematic_order(self):

        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        #Compute Steinhardt order parameter
        nop = freud.order.Nematic([1, 0, 0])

        nop.compute(ori)

        nematic_param = np.abs(nop.order)

        return nematic_param
