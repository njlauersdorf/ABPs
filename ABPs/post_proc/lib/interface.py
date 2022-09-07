
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


#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility


class interface:
    def __init__(self, area_frac_dict, align_dict, part_dict, press_dict, l_box, partNum, NBins, peA, peB, parFrac, eps, typ, ang):

        self.align_x = align_dict['bin']['all']['x']
        self.align_y = align_dict['bin']['all']['y']
        self.align_mag = align_dict['bin']['all']['mag']

        self.align_x_A = align_dict['bin']['A']['x']
        self.align_y_A = align_dict['bin']['A']['y']
        self.align_mag_A = align_dict['bin']['A']['mag']

        self.align_x_B = align_dict['bin']['B']['x']
        self.align_y_B = align_dict['bin']['B']['y']
        self.align_mag_B = align_dict['bin']['B']['mag']

        self.area_frac = area_frac_dict['bin']['all']
        self.area_frac_A = area_frac_dict['bin']['A']
        self.area_frac_B = area_frac_dict['bin']['B']

        self.occParts = part_dict['clust']
        self.binParts = part_dict['id']
        self.typParts = part_dict['typ']

        self.press = press_dict['bin']['all']
        self.press_A = press_dict['bin']['A']
        self.press_B = press_dict['bin']['B']

        self.theory_functs = theory.theory()

        self.l_box = l_box
        self.h_box = self.l_box/2

        self.utility_functs = utility.utility(self.l_box)

        self.partNum = partNum
        self.min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen

        try:
            self.NBins = int(NBins)
        except:
            print('NBins must be either a float or an integer')

        self.sizeBin = self.utility_functs.roundUp((self.l_box / self.NBins), 6)

        self.peA = peA
        self.peB = peB
        self.parFrac = parFrac

        self.peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

        self.eps = eps

        lat_theory = self.theory_functs.conForRClust(self.peNet, eps)
        curPLJ = self.theory_functs.ljPress(lat_theory, self.peNet, eps)
        self.phi_theory = self.theory_functs.latToPhi(lat_theory)
        self.phi_g_theory = self.theory_functs.compPhiG(self.peNet, lat_theory)

        self.typ = typ

        self.ang = ang

    def det_surface_points(self, phase_dict, int_dict, int_comp_dict):

        phaseBin = phase_dict['bin']

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']

        surface1_id=np.zeros((self.NBins, self.NBins), dtype=int)        #Label exterior edges of interfaces
        surface2_id=np.zeros((self.NBins, self.NBins), dtype=int)        #Label interior edges of interfaces

        surface1_phaseInt=np.zeros(self.partNum)
        surface2_phaseInt=np.zeros(self.partNum)

        # Individually label each interface until all edge bins identified using flood fill algorithm
        if len(int_large_ids)>0:
            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):

                    #If bin is an interface, continue
                    if phaseBin[ix][iy]==1:

                        #Count surrounding bin phases
                        gas_neigh_num=0
                        bulk_neigh_num=0

                        #identify neighboring bins
                        if (ix + 1) == self.NBins:
                            lookx = [ix-1, ix, 0]
                        elif ix==0:
                            lookx=[self.NBins-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, ix+1]

                        if (iy + 1) == self.NBins:
                            looky = [iy-1, iy, 0]
                        elif iy==0:
                            looky=[self.NBins-1, iy, iy+1]
                        else:
                            looky = [iy-1, iy, iy+1]


                        #loop over surrounding x-index bins
                        for indx in lookx:

                            # Loop through surrounding y-index bins
                            for indy in looky:


                                #If bin hadn't been assigned an interface id yet, continue

                                #If bin is a gas, continue
                                if phaseBin[indx][indy]==2:

                                    #count number of gas bins
                                    gas_neigh_num+=1

                                elif phaseBin[indx][indy]==0:

                                    bulk_neigh_num+=1

                                #If more than interface bins surround, identify if interior or exterior edge
                                if (gas_neigh_num>0) or (bulk_neigh_num>0):
                                    #If more neighboring gas bins around reference bin, then it's an exterior edge
                                    if int(int_id[ix][iy])==int_large_ids[0]:

                                        if gas_neigh_num>=bulk_neigh_num:
                                            surface2_id[ix][iy]=1

                                        #Otherwise, it's an interior edge
                                        else:
                                            surface1_id[ix][iy]=1

                                    elif int(int_id[ix][iy])!=0:

                                        #If more bulk than gas, the bin is an external edge
                                        if gas_neigh_num<=bulk_neigh_num:
                                            surface2_id[ix][iy]=1

                                        #If more gas than bulk, the bin is an internal edge
                                        else:
                                            surface1_id[ix][iy]=1

        #Label phase of each particle
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if len(self.binParts[ix][iy])>0:
                    for h in range(0, len(self.binParts[ix][iy])):

                        surface2_phaseInt[self.binParts[ix][iy][h]]=surface2_id[ix][iy]
                        surface1_phaseInt[self.binParts[ix][iy][h]]=surface1_id[ix][iy]

        #Save positions of external and internal edges
        surface2_pos_x=np.array([])
        surface2_pos_y=np.array([])
        surface1_pos_x=np.array([])
        surface1_pos_y=np.array([])

        #Save positions of interior and exterior edge bins
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if surface2_id[ix][iy]==1:
                    surface2_pos_x=np.append(surface2_pos_x, (ix+0.5)*self.sizeBin)
                    surface2_pos_y=np.append(surface2_pos_y, (iy+0.5)*self.sizeBin)

                elif surface1_id[ix][iy]==1:
                    surface1_pos_x=np.append(surface1_pos_x, (ix+0.5)*self.sizeBin)
                    surface1_pos_y=np.append(surface1_pos_y, (iy+0.5)*self.sizeBin)

        surface_dict = {'surface 1': {'pos': {'x': surface1_pos_x, 'y': surface1_pos_y}, 'id': {'bin': surface1_id, 'part': surface1_phaseInt}}, 'surface 2': {'pos': {'x': surface2_pos_x, 'y': surface2_pos_y}, 'id': {'bin': surface2_id, 'part': surface2_phaseInt}}}
        return surface_dict

    def surface_sort(self, surface_pos_x, surface_pos_y):

        #Save positions of external and internal edges
        clust_true = 0
        surface_pos_x_sorted=np.array([])
        surface_pos_y_sorted=np.array([])

        while len(surface_pos_x)>0:
            if len(surface_pos_x_sorted)==0:
                surface_pos_x_sorted = np.append(surface_pos_x_sorted, surface_pos_x[0])
                surface_pos_y_sorted = np.append(surface_pos_y_sorted, surface_pos_y[0])

                surface_pos_x = np.delete(surface_pos_x, 0)
                surface_pos_y = np.delete(surface_pos_y, 0)

            else:
                shortest_length = 100000
                for i in range(0, len(surface_pos_y)):

                    difx = self.utility_functs.sep_dist(surface_pos_x_sorted[-1], surface_pos_x[i])
                    dify = self.utility_functs.sep_dist(surface_pos_y_sorted[-1], surface_pos_y[i])

                    difr = (difx**2 + dify**2)**0.5

                    if difr < shortest_length:
                        shortest_length = difr
                        shortest_xlength = difx
                        shortest_ylength = dify
                        shortest_id = i

                    elif difr == shortest_length:
                        if (difx<0) or (dify<0):
                            if (shortest_xlength <0) or (shortest_ylength<0):
                                shortest_length = difr
                                shortest_xlength = difx
                                shortest_ylength = dify
                                shortest_id = i
                            else:
                                pass
                        else:
                            pass

                surface_pos_x_sorted = np.append(surface_pos_x_sorted, surface_pos_x[shortest_id])
                surface_pos_y_sorted = np.append(surface_pos_y_sorted, surface_pos_y[shortest_id])

                surface_pos_x = np.delete(surface_pos_x, shortest_id)
                surface_pos_y = np.delete(surface_pos_y, shortest_id)
        surface_pos_sort_dict = {'x': surface_pos_x_sorted, 'y': surface_pos_y_sorted}
        return surface_pos_sort_dict

    def surface_com(self, int_dict, int_comp_dict, surface_dict):

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']

        surface1_id = surface_dict['surface 1']['id']['bin']
        surface2_id = surface_dict['surface 2']['id']['bin']

        surface1_pos_x = surface_dict['surface 1']['pos']['x']
        surface2_pos_x = surface_dict['surface 2']['pos']['x']

        #If there is an interface (bubble), find the mid-point of the cluster's edges
        #Constant density in bulk phase, so approximately center of mass
        if len(int_large_ids) > 0:
            surface_num=0
            x_box_pos=0
            y_box_pos=0

            hor_wrap_left = 0
            hor_wrap_right = 0
            vert_wrap_bot = 0
            vert_wrap_top = 0

            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):
                    if (int_id[ix][iy]==int_large_ids[0]):
                        if (ix == 0):
                            hor_wrap_left = 1
                        elif (ix == self.NBins-1):
                            hor_wrap_right = 1
                        if (iy == 0):
                            vert_wrap_bot = 1
                        elif (iy == self.NBins-1):
                            vert_wrap_top = 1

            #Sum positions of external edges of interface
            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):
                    if len(surface2_pos_x)>0:
                        if (int_id[ix][iy]==int_large_ids[0]) & (surface2_id[ix][iy]==1):
                            x_box_pos_temp = (ix+0.5)*self.sizeBin
                            if (hor_wrap_right==1) & (hor_wrap_left==1) & (x_box_pos_temp<self.h_box):
                                x_box_pos_temp += self.h_box

                            y_box_pos_temp = (iy+0.5)*self.sizeBin
                            if (vert_wrap_bot==1) & (vert_wrap_top==1) & (y_box_pos_temp<self.h_box):
                                y_box_pos_temp += self.h_box

                            x_box_pos += x_box_pos_temp
                            y_box_pos += y_box_pos_temp
                            surface_num +=1
                    elif len(surface1_pos_x)>0:
                        if (int_id[ix][iy]==int_large_ids[0]) & (surface1_id[ix][iy]==1):
                            x_box_pos_temp = (ix+0.5)*self.sizeBin
                            if (hor_wrap_right==1) & (hor_wrap_left==1) & (x_box_pos_temp<self.h_box):
                                x_box_pos_temp += self.h_box

                            y_box_pos_temp = (iy+0.5)*self.sizeBin
                            if (vert_wrap_bot==1) & (vert_wrap_top==1) & (y_box_pos_temp<self.h_box):
                                y_box_pos_temp += self.h_box

                            x_box_pos += x_box_pos_temp
                            y_box_pos += y_box_pos_temp
                            surface_num +=1

            #Determine mean location (CoM) of external edges of interface
            if surface_num>0:
                box_com_x = x_box_pos/surface_num
                box_com_y = y_box_pos/surface_num

                box_com_x_abs = np.abs(box_com_x)
                if box_com_x_abs>=self.l_box:
                    if box_com_x < -self.h_box:
                        box_com_x += self.l_box
                    else:
                        box_com_x -= self.l_box

                box_com_y_abs = np.abs(box_com_y)
                if box_com_y_abs>=self.l_box:
                    if box_com_y < -self.h_box:
                        box_com_y += self.l_box
                    else:
                        box_com_y -= self.l_box
            else:
                box_com_x=0
                box_com_y=0

        surface_com_dict = {'x': box_com_x, 'y': box_com_y}
        return surface_com_dict

    def surface_radius_bins(self, int_dict, int_comp_dict, surface_dict, surface_com_dict):

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']

        surface1_id = surface_dict['surface 1']['id']['bin']
        surface2_id = surface_dict['surface 2']['id']['bin']

        surface1_pos_x = surface_dict['surface 1']['pos']['x']
        surface2_pos_x = surface_dict['surface 2']['pos']['x']

        box_com_x = surface_com_dict['x']
        box_com_y = surface_com_dict['y']

        if len(int_large_ids) > 0:
            #Initialize empty arrays for calculation
            thetas = np.array([])
            radii = np.array([])

            x_id = np.array([], dtype=int)
            y_id = np.array([], dtype=int)

            #Calculate distance from CoM to external edge bin and angle from CoM
            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):

                    # If bin is interface and external edge, continue...
                    if (int_id[ix][iy]==int_large_ids[0]) & (surface2_id[ix][iy]==1):

                        #Reference bin location
                        x_box_pos = (ix+0.5)*self.sizeBin
                        y_box_pos = (iy+0.5)*self.sizeBin

                        #Calculate x-distance from CoM
                        difx=x_box_pos-box_com_x

                        #Enforce periodic boundary conditions
                        difx_abs = np.abs(difx)
                        if difx_abs>=self.h_box:
                            if difx < -self.h_box:
                                difx += self.l_box
                            else:
                                difx -= self.l_box

                        #Calculate y-distance from CoM
                        dify=y_box_pos-box_com_y

                        #Enforce periodic boundary conditions
                        dify_abs = np.abs(dify)
                        if dify_abs>=self.h_box:
                            if dify < -self.h_box:
                                dify += self.l_box
                            else:
                                dify -= self.l_box

                        #Calculate angle from CoM and x-axis
                        theta_val = np.arctan2(np.abs(dify), np.abs(difx))*(180/math.pi)

                        #Enforce correct quadrant for particle
                        if (difx>0) & (dify>0):
                            pass
                        elif (difx<0) & (dify>0):
                            theta_val = 180-theta_val
                        elif (difx<0) & (dify<0):
                            theta_val = theta_val+180
                        elif (difx>0) & (dify<0):
                            theta_val = 360-theta_val

                        #Save calculated angle from CoM and x-axis
                        thetas = np.append(thetas, theta_val)

                        #Save id of bin of calculation
                        x_id = np.append(x_id, int(ix))
                        y_id = np.append(y_id, int(iy))

                        #Save radius from CoM of bin
                        radii = np.append(radii, (difx**2 + dify**2)**0.5)
        radius_dict = {'radius': {'vals': radii, 'mean': np.mean(radii)}, 'theta': {'vals': thetas, 'mean': np.mean(thetas)}}
        return radius_dict

    def separate_surfaces(self, surface_dict, int_dict, int_comp_dict):
        # Initialize empty arrays
        surface1_id = surface_dict['surface 1']['id']['bin']
        surface1_x = surface_dict['surface 1']['pos']['x']
        surface1_y = surface_dict['surface 1']['pos']['y']

        surface2_id = surface_dict['surface 2']['id']['bin']
        surface2_x = surface_dict['surface 2']['pos']['x']
        surface2_y = surface_dict['surface 2']['pos']['y']

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']

        int_surface_x = np.array([], dtype=int)
        int_surface_y = np.array([], dtype=int)

        ext_surface_x = np.array([], dtype=int)
        ext_surface_y = np.array([], dtype=int)

        surface1_x_id = np.array([], dtype=int)
        surface1_y_id = np.array([], dtype=int)

        surface2_x_id = np.array([], dtype=int)
        surface2_y_id = np.array([], dtype=int)

        int_surface_id=np.zeros((self.NBins, self.NBins), dtype=int)
        ext_surface_id=np.zeros((self.NBins, self.NBins), dtype=int)

        sep_surface_dict = {}

        surface1_num = 0
        surface2_num = 0
        for m in range(0, len(int_large_ids)):
            if int_large_ids[m]!=999:
                for ix in range(0, self.NBins):
                    for iy in range(0, self.NBins):
                        if int_id[ix][iy]==int_large_ids[m]:
                            if surface1_id[ix][iy]==1:
                                surface1_num +=1
                                surface1_x_id = np.append(surface1_x_id, ix)
                                surface1_y_id = np.append(surface1_y_id, iy)
                            if surface2_id[ix][iy]==1:
                                surface2_num +=1
                                surface2_x_id = np.append(surface2_x_id, ix)
                                surface2_y_id = np.append(surface2_y_id, iy)

                if (surface1_num > 0) & (surface2_num > 0):
                    if surface1_num>surface2_num:
                        for v in range(0, len(surface1_x_id)):
                            ext_surface_x = np.append(ext_surface_x, surface1_x_id[v])
                            ext_surface_y = np.append(ext_surface_y, surface1_y_id[v])
                        for v in range(0, len(surface2_x_id)):
                            int_surface_x = np.append(int_surface_x, surface2_x_id[v])
                            int_surface_y = np.append(int_surface_y, surface2_y_id[v])
                    else:
                        for v in range(0, len(surface2_x_id)):
                            ext_surface_x = np.append(ext_surface_x, surface2_x_id[v])
                            ext_surface_y = np.append(ext_surface_y, surface2_y_id[v])
                        for v in range(0, len(surface1_x_id)):
                            int_surface_x = np.append(int_surface_x, surface1_x_id[v])
                            int_surface_y = np.append(int_surface_y, surface1_y_id[v])
                elif (surface1_num > 0) & (surface2_num == 0):
                    for v in range(0, len(surface1_x_id)):
                        ext_surface_x = np.append(ext_x, surface1_x_id[v])
                        ext_surface_y = np.append(ext_y, surface1_y_id[v])

                elif (surface1_num == 0) & (surface2_num > 0):
                    for v in range(0, len(surface2_x_id)):
                        ext_surface_x = np.append(ext_x, surface2_x_id[v])
                        ext_surface_y = np.append(ext_y, surface2_y_id[v])

                int_surface_num = len(int_surface_x)
                ext_surface_num = len(ext_surface_x)

                if int_surface_num > 0:
                    for ix in range(0, len(int_surface_x)):
                        int_surface_id[int_surface_x[ix]][int_surface_y[ix]]=1
                        ext_surface_id[int_surface_x[ix]][int_surface_y[ix]]=0
                if ext_surface_num >0:
                    for ix in range(0, len(ext_surface_x)):
                        int_surface_id[ext_surface_x[ix]][ext_surface_y[ix]]=0
                        ext_surface_id[ext_surface_x[ix]][ext_surface_y[ix]]=1

                int_surface_pos_x = int_surface_x * self.sizeBin
                int_surface_pos_y = int_surface_y * self.sizeBin

                ext_surface_pos_x = ext_surface_x * self.sizeBin
                ext_surface_pos_y = ext_surface_y * self.sizeBin
                indiv_surface_dict = {'interior': {'x bin': int_surface_x, 'y bin': int_surface_y, 'ids': int_surface_id, 'num': int_surface_num}, 'exterior': {'x bin': ext_surface_x, 'y bin': ext_surface_y, 'ids': ext_surface_id, 'num': ext_surface_num}}
                key_temp = 'surface id ' + str(int(int_large_ids[m]))
                sep_surface_dict[key_temp] = indiv_surface_dict
        return sep_surface_dict

    def sort_surface_points(self, surface_dict):

        surface_x = surface_dict['x bin']
        surface_y = surface_dict['y bin']
        surface_id = surface_dict['ids']

        surface_x_sort = np.array([surface_x[0]])
        surface_y_sort = np.array([surface_y[0]])

        ix=int(surface_x_sort[0])
        iy=int(surface_y_sort[0])

        shortest_idx = np.array([])
        shortest_idy = np.array([])

        surface_x = np.delete(surface_x, 0)
        surface_y = np.delete(surface_y, 0)
        fail=0

        #Determine if first interior surface bin of interface has at least 1
        #neighbor that is an interior surface bin of interface
        if len(surface_x)>0:

            if ix < (self.NBins-1):
                right = int(ix+1)
            else:
                right= int(0)

            if ix > 0:
                left = int(ix-1)
            else:
                left=int(self.NBins-1)

            if iy < (self.NBins-1):
                up = int(iy+1)
            else:
                up= int(0)

            if iy > 0:
                down = int(iy-1)
            else:
                down= int(self.NBins-1)

            if surface_id[right][iy]==1:
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, iy)

                loc_id = np.where((surface_x == right) & (surface_y == iy))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[ix][up]==1:
                surface_x_sort = np.append(surface_x_sort, ix)
                surface_y_sort = np.append(surface_y_sort, up)

                loc_id = np.where((surface_x == ix) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[ix][down]==1:
                surface_x_sort = np.append(surface_x_sort, ix)
                surface_y_sort = np.append(surface_y_sort, down)

                loc_id = np.where((surface_x == ix) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[left][iy]==1:
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, iy)

                loc_id = np.where((surface_x == left) & (surface_y == iy))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[left][up]==1:
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, up)

                loc_id = np.where((surface_x == left) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[left][down]==1:
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, down)

                loc_id = np.where((surface_x == left) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surfacer_y = np.delete(surface_y, loc_id)
            elif surface_id[right][up]==1:
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, up)

                loc_id = np.where((surface_x == right) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            elif surface_id[right][down]==1:
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, down)

                loc_id = np.where((surface_x == right) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)
            else:
                fail=1

            #If found at least 1 interior surface bin neighbor
            if fail==0:
                ix_ref=surface_x_sort[1]
                iy_ref=surface_y_sort[1]

                past_size=0
                while len(surface_x)>0:
                    current_size = len(surface_x)

                    if past_size == current_size:
                        shortest_length = 100000.
                        for ix in range(0, self.NBins):
                            for iy in range(0, self.NBins):
                                if (ix!=ix_ref) | (iy!=iy_ref):
                                    if surface_id[ix][iy]==1:
                                        loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                        if len(loc_id)>0:

                                            difx = self.utility_functs.sep_dist((ix_ref+0.5)*self.sizeBin, (ix+0.5)*self.sizeBin)
                                            dify = self.utility_functs.sep_dist((iy_ref+0.5)*self.sizeBin, (iy+0.5)*self.sizeBin)

                                            difr = (difx**2 + dify**2)**0.5

                                            if difr < shortest_length:
                                                shortest_length = difr
                                                shortest_idx = np.array([ix])
                                                shortest_idy = np.array([iy])

                                            elif difr == shortest_length:
                                                shortest_idx = np.append(shortest_idx, ix)
                                                shortest_idy = np.append(shortest_idy, iy)

                        if shortest_length > self.h_box/10:
                            break

                        if len(shortest_idx) > 1:
                            num_neigh = np.zeros(len(shortest_idx))

                            for ind in range(0, len(shortest_idx)):

                                ix_ind = shortest_idx[ind]
                                iy_ind = shortest_idy[ind]

                                if (ix_ind + 1) == self.NBins:
                                    lookx = [ix_ind-1, ix_ind, 0]
                                elif ix_ind==0:
                                    lookx=[self.NBins-1, ix_ind, ix_ind+1]
                                else:
                                    lookx = [ix_ind-1, ix_ind, ix_ind+1]

                                #Identify neighboring bin indices in y-direction
                                if (iy_ind + 1) == self.NBins:
                                    looky = [iy_ind-1, iy_ind, 0]
                                elif iy_ind==0:
                                    looky=[self.NBins-1, iy_ind, iy_ind+1]
                                else:
                                    looky = [iy_ind-1, iy_ind, iy_ind+1]
                                for ix in lookx:
                                    for iy in looky:
                                        if (ix != ix_ind) | (iy != iy_ind):
                                            loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                            if len(loc_id)>0:
                                                if surface_id[ix][iy]==1:
                                                    num_neigh[ind]+=1
                            min_inds = np.min(num_neigh)
                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                            if len(loc_min_inds)==1:

                                ix_ref = shortest_idx[loc_min_inds][0]
                                iy_ref = shortest_idy[loc_min_inds][0]

                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)
                            else:
                                ix_ref = shortest_idx[np.min(loc_min_inds)]
                                iy_ref = shortest_idy[np.min(loc_min_inds)]
                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)
                        elif len(shortest_idx)==1:

                            ix_ref = shortest_idx[0]
                            iy_ref = shortest_idy[0]

                            surface_x_sort = np.append(surface_x_sort, ix_ref)
                            surface_y_sort = np.append(surface_y_sort, iy_ref)

                            loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                            surface_x = np.delete(surface_x, loc_id)
                            surface_y = np.delete(surface_y, loc_id)
                        else:
                            break

                    if surface_id[ix_ref][iy_ref]==1:

                        #Identify neighboring bin indices in x-direction
                        if (ix_ref + 1) == self.NBins:
                            lookx = [ix_ref-1, ix_ref, 0]
                        elif ix_ref==0:
                            lookx=[self.NBins-1, ix_ref, ix_ref+1]
                        else:
                            lookx = [ix_ref-1, ix_ref, ix_ref+1]

                        #Identify neighboring bin indices in y-direction
                        if (iy_ref + 1) == self.NBins:
                            looky = [iy_ref-1, iy_ref, 0]
                        elif iy_ref==0:
                            looky=[self.NBins-1, iy_ref, iy_ref+1]
                        else:
                            looky = [iy_ref-1, iy_ref, iy_ref+1]

                        shortest_length = 100000.
                        for ix in lookx:
                            for iy in looky:
                                if (ix!=ix_ref) | (iy!=iy_ref):
                                    loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                    if len(loc_id)>0:
                                        if surface_id[ix][iy]==1:
                                            difx = self.utility_functs.sep_dist((ix+0.5)*self.sizeBin, (ix_ref+0.5)*self.sizeBin)
                                            dify = self.utility_functs.sep_dist((iy+0.5)*self.sizeBin, (iy_ref+0.5)*self.sizeBin)

                                            difr = (difx**2 + dify**2)**0.5

                                            if difr < shortest_length:
                                                shortest_length = difr
                                                shortest_idx = np.array([ix])
                                                shortest_idy = np.array([iy])

                                            elif difr == shortest_length:
                                                shortest_idx = np.append(shortest_idx, ix)
                                                shortest_idy = np.append(shortest_idy, iy)

                        if len(shortest_idx) > 1:
                            num_neigh = np.zeros(len(shortest_idx))

                            for ind in range(0, len(shortest_idx)):
                                ix_ind = shortest_idx[ind]
                                iy_ind = shortest_idy[ind]

                                if (ix_ind + 1) == self.NBins:
                                    lookx = [ix_ind-1, ix_ind, 0]
                                elif ix_ind==0:
                                    lookx=[self.NBins-1, ix_ind, ix_ind+1]
                                else:
                                    lookx = [ix_ind-1, ix_ind, ix_ind+1]

                                #Identify neighboring bin indices in y-direction
                                if (iy_ind + 1) == self.NBins:
                                    looky = [iy_ind-1, iy_ind, 0]
                                elif iy_ind==0:
                                    looky=[self.NBins-1, iy_ind, iy_ind+1]
                                else:
                                    looky = [iy_ind-1, iy_ind, iy_ind+1]

                                for ix in lookx:
                                    for iy in looky:
                                        if (ix != ix_ind) | (iy != iy_ind):
                                            loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                            if len(loc_id)>0:
                                                if surface_id[ix][iy]==1:
                                                    num_neigh[ind]+=1

                            min_inds = np.min(num_neigh)
                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                            if len(loc_min_inds)==1:

                                ix_ref = shortest_idx[loc_min_inds][0]
                                iy_ref = shortest_idy[loc_min_inds][0]

                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)
                            else:
                                ix_ref = shortest_idx[np.min(loc_min_inds)]
                                iy_ref = shortest_idy[np.min(loc_min_inds)]
                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)

                        elif len(shortest_idx)==1:

                            ix_ref = shortest_idx[0]
                            iy_ref = shortest_idy[0]

                            surface_x_sort = np.append(surface_x_sort, ix_ref)
                            surface_y_sort = np.append(surface_y_sort, iy_ref)

                            loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                            surface_x = np.delete(surface_x, loc_id)
                            surface_y = np.delete(surface_y, loc_id)
                        else:
                            break

                    past_size = current_size
        sort_surface_ids = {'x': surface_x_sort, 'y': surface_y_sort}
        return sort_surface_ids

    def surface_curve_interp(self, sort_surface_ids):

        surface_x_sort = sort_surface_ids['x']
        surface_y_sort = sort_surface_ids['y']

        surface_x_sort_pos = surface_x_sort * self.sizeBin
        surface_y_sort_pos = surface_y_sort * self.sizeBin


        adjacent_x = np.array([surface_x_sort[0]])
        adjacent_x_pos = np.array([surface_x_sort_pos[0]])
        adjacent_y = np.array([surface_y_sort[0]])
        adjacent_y_pos = np.array([surface_y_sort_pos[0]])

        adjacent_x_discont = np.array([])
        adjacent_x_discont_pos = np.array([])
        adjacent_y_discont = np.array([])
        adjacent_y_discont_pos = np.array([])

        if len(surface_x_sort)>1:
            for m in range(1, len(surface_x_sort)):
                if len(adjacent_x) == 0:
                    adjacent_x = np.append(adjacent_x, surface_x_sort[m])
                    adjacent_x_pos = np.append(adjacent_x_pos, surface_x_sort_pos[m])
                    adjacent_y = np.append(adjacent_y, surface_y_sort[m])
                    adjacent_y_pos = np.append(adjacent_y_pos, surface_y_sort_pos[m])
                else:

                    difx = surface_x_sort_pos[m]-surface_x_sort_pos[m-1]
                    dify = surface_y_sort_pos[m]-surface_y_sort_pos[m-1]

                    #Enforce periodic boundary conditions
                    difx_abs = np.abs(difx)
                    dify_abs = np.abs(dify)

                    #Enforce periodic boundary conditions
                    if difx_abs>=self.h_box:
                        if difx < -self.h_box:
                            surface_x_sort_pos[m:-1] += self.l_box
                            surface_x_sort[m:-1] += self.NBins
                        else:
                            surface_x_sort_pos[m:-1] -= self.l_box
                            surface_x_sort[m:-1] -= self.NBins

                    #Enforce periodic boundary conditions
                    if dify_abs>=self.h_box:
                        if dify < -self.h_box:
                            surface_y_sort_pos[m:-1] += self.l_box
                            surface_y_sort[m:-1] += self.NBins
                        else:
                            surface_y_sort_pos[m:-1] -= self.l_box
                            surface_y_sort[m:-1] -= self.NBins

                    if (difx_abs>=self.h_box) or (dify_abs>=self.h_box):
                        adjacent_x_discont = np.append(adjacent_x_discont, adjacent_x)
                        adjacent_x_discont_pos = np.append(adjacent_x_discont_pos, adjacent_x_pos)
                        adjacent_y_discont = np.append(adjacent_y_discont, adjacent_y)
                        adjacent_y_discont_pos = np.append(adjacent_y_discont_pos, adjacent_y_pos)

                        adjacent_x = np.array([])
                        adjacent_x_pos = np.array([])
                        adjacent_y = np.array([])
                        adjacent_y_pos = np.array([])
                    else:
                        adjacent_x = np.append(adjacent_x, surface_x_sort[m])
                        adjacent_x_pos = np.append(adjacent_x_pos, surface_x_sort_pos[m])
                        adjacent_y = np.append(adjacent_y, surface_y_sort[m])
                        adjacent_y_pos = np.append(adjacent_y_pos, surface_y_sort_pos[m])

                        if (m==len(surface_x_sort)-1):
                            adjacent_x_discont = np.append(adjacent_x_discont, adjacent_x)
                            adjacent_x_discont_pos = np.append(adjacent_x_discont_pos, adjacent_x_pos)
                            adjacent_y_discont = np.append(adjacent_y_discont, adjacent_y)
                            adjacent_y_discont_pos = np.append(adjacent_y_discont_pos, adjacent_y_pos)

                            adjacent_x = np.array([])
                            adjacent_x_pos = np.array([])
                            adjacent_y = np.array([])
                            adjacent_y_pos = np.array([])

            adjacent_x_discont_pos_smooth = np.array([])
            adjacent_y_discont_pos_smooth = np.array([])
            adjacent_x_discont_smooth = np.array([])
            adjacent_y_discont_smooth = np.array([])

            for m in range(0, len(adjacent_x_discont_pos)):
                adjacent_x_discont_pos_smooth = np.append(adjacent_x_discont_pos_smooth, adjacent_x_discont_pos[m])
                adjacent_y_discont_pos_smooth = np.append(adjacent_y_discont_pos_smooth, adjacent_y_discont_pos[m])
                adjacent_x_discont_smooth = np.append(adjacent_x_discont_smooth, adjacent_x_discont[m])
                adjacent_y_discont_smooth = np.append(adjacent_y_discont_smooth, adjacent_y_discont[m])

            adjacent_x_discont_smooth_copy = np.copy(adjacent_x_discont_smooth)
            adjacent_y_discont_smooth_copy = np.copy(adjacent_y_discont_smooth)

            if len(adjacent_x_discont_smooth) >= 3:
                for m in range(0, len(adjacent_x_discont_smooth)):

                    if m==0:
                        adjacent_x_discont_smooth[m] = (adjacent_x_discont_smooth_copy[-1] + adjacent_x_discont_smooth_copy[0] + adjacent_x_discont_smooth_copy[1])/3
                        adjacent_y_discont_smooth[m] = (adjacent_y_discont_smooth_copy[-1] + adjacent_y_discont_smooth_copy[0] + adjacent_y_discont_smooth_copy[1])/3
                    elif m==len(adjacent_x_discont_smooth_copy)-1:
                        adjacent_x_discont_smooth[m]= (adjacent_x_discont_smooth_copy[m] + adjacent_x_discont_smooth_copy[0] + adjacent_x_discont_smooth_copy[m-1])/3
                        adjacent_y_discont_smooth[m]= (adjacent_y_discont_smooth_copy[m] + adjacent_y_discont_smooth_copy[0] + adjacent_y_discont_smooth_copy[m-1])/3
                    else:
                        adjacent_x_discont_smooth[m] = (adjacent_x_discont_smooth_copy[m-1] + adjacent_x_discont_smooth_copy[m] + adjacent_x_discont_smooth_copy[m+1])/3
                        adjacent_y_discont_smooth[m] = (adjacent_y_discont_smooth_copy[m-1] + adjacent_y_discont_smooth_copy[m] + adjacent_y_discont_smooth_copy[m+1])/3
            else:
                for m in range(0, len(adjacent_x_discont_smooth)):

                    adjacent_x_discont_smooth[m] = np.mean(adjacent_x_discont_smooth_copy)
                    adjacent_y_discont_smooth[m] = np.mean(adjacent_y_discont_smooth_copy)
            okay = np.where(np.abs(np.diff(adjacent_x_discont_smooth)) + np.abs(np.diff(adjacent_y_discont_smooth)) > 0)
            surface_x_interp = np.r_[adjacent_x_discont_smooth[okay], adjacent_x_discont_smooth[-1], adjacent_x_discont_smooth[0]]
            surface_y_interp = np.r_[adjacent_y_discont_smooth[okay], adjacent_y_discont_smooth[-1], adjacent_y_discont_smooth[0]]

            if len(surface_x_interp)==3:
                tck, u = interpolate.splprep([surface_x_interp, surface_y_interp], s=0, k=2, per=True)
            elif len(surface_x_interp)>3:
                tck, u = interpolate.splprep([surface_x_interp, surface_y_interp], s=0, per=True)
            if len(surface_x_interp)>=3:
                # evaluate the spline fits for 1000 evenly spaced distance values
                xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

                jump = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2)
                smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
                limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
                xn, yn = xi[:-1], yi[:-1]
                xn = xn[(jump > 0) & (smooth_jump < limit)]
                yn = yn[(jump > 0) & (smooth_jump < limit)]

                xn_pos = np.copy(xn)
                yn_pos = np.copy(yn)
                xn_pos_non_per = np.copy(xn)
                yn_pos_non_per = np.copy(yn)


                for m in range(0, len(xn)):
                    xn_pos[m] = xn[m] * self.sizeBin
                    yn_pos[m] = yn[m] * self.sizeBin
                    xn_pos_non_per[m] = xn[m] * self.sizeBin
                    yn_pos_non_per[m] = yn[m] * self.sizeBin

                    if xn[m] < 0:
                        xn[m]+=self.NBins
                    if xn[m]>=self.NBins:
                        xn[m]-=self.NBins

                    if yn[m] < 0:
                        yn[m]+=self.NBins
                    if yn[m]>=self.NBins:
                        yn[m]-=self.NBins

                    if xn_pos[m] < 0:
                        xn_pos[m]+=self.l_box
                    if xn_pos[m]>=self.l_box:
                        xn_pos[m]-=self.l_box

                    if yn_pos[m] < 0:
                        yn_pos[m]+=self.l_box
                    if yn_pos[m]>=self.l_box:
                        yn_pos[m]-=self.l_box



            else:
                xn = np.zeros(1)
                yn = np.zeros(1)
                xn_pos = np.zeros(1)
                yn_pos = np.zeros(1)
                xn_pos_non_per = np.zeros(1)
                yn_pos_non_per = np.zeros(1)

                xn_pos[0] = int_x[0]
                yn_pos[0] = int_y[0]
                xn_pos[0] = int_x[0] * self.sizeBin
                yn_pos[0] = int_y[0] * self.sizeBin
                xn_pos_non_per[0] = int_x[0] * self.sizeBin
                yn_pos_non_per[0] = int_y[0] * self.sizeBin
                if xn[0] < 0:
                    xn[0]+=self.NBins
                if xn[0]>=self.NBins:
                    xn[0]-=self.NBins

                if yn[0] < 0:
                    yn[0]+=self.NBins
                if yn[0]>=self.NBins:
                    yn[0]-=self.NBins

                if xn_pos[0] < 0:
                    xn_pos[0]+=self.l_box
                if xn_pos[0]>=self.l_box:
                    xn_pos[0]-=self.l_box

                if yn_pos[0] < 0:
                    yn_pos[0]+=self.l_box
                if yn_pos[0]>=self.l_box:
                    yn_pos[0]-=self.l_box

        else:
            xn=np.array([int_x[0]])
            yn=np.array([int_y[0]])
            xn_pos = np.copy(xn)
            yn_pos = np.copy(yn)
            xn_pos_non_per = np.copy(xn)
            yn_pos_non_per = np.copy(yn)
            for m in range(0, len(xn)):
                xn_pos[m] = xn[m] * self.sizeBin
                yn_pos[m] = yn[m] * self.sizeBin
                xn_pos_non_per[m] = xn[m] * self.sizeBin
                yn_pos_non_per[m] = yn[m] * self.sizeBin

                if xn[m] < 0:
                    xn[m]+=self.NBins
                if xn[m]>=self.NBins:
                    xn[m]-=self.NBins

                if yn[m] < 0:
                    yn[m]+=self.NBins
                if yn[m]>=self.NBins:
                    yn[m]-=self.NBins

                if xn_pos[m] < 0:
                    xn_pos[m]+=self.l_box
                if xn_pos[m]>=self.l_box:
                    xn_pos[m]-=self.l_box

                if yn_pos[m] < 0:
                    yn_pos[m]+=self.l_box
                if yn_pos[m]>=self.l_box:
                    yn_pos[m]-=self.l_box

        surface_curve_dict = {'id': {'x': xn, 'y': yn}, 'pos': {'x': xn_pos, 'y': yn_pos}}
        return surface_curve_dict


    def surface_area(self, surface_curve_pos_dict):

        surface_curve_xpos = surface_curve_pos_dict['x']
        surface_curve_ypos = surface_curve_pos_dict['y']

        surface_area = 0
        for id2 in range(1, len(surface_curve_xpos)):

            #Calculate position of interior edge bin
            difx = self.utility_functs.sep_dist(surface_curve_xpos[id2-1], surface_curve_xpos[id2])
            dify = self.utility_functs.sep_dist(surface_curve_ypos[id2-1], surface_curve_ypos[id2])

            #Calculate distance from interior edge bin to exterior edge bin
            difr = ( (difx)**2 + (dify)**2)**0.5

            #If this distance is the shortest calculated thus far, replace the value with it
            surface_area += difr

        return surface_area

    def surface_boundary_jumps(self, surface_curve_pos_dict):
        surface_curve_xpos = surface_curve_pos_dict['x']
        surface_curve_ypos = surface_curve_pos_dict['y']

        from_right_to_left = []
        from_left_to_right = []
        from_bot_to_top = []
        from_top_to_bot = []
        boundary_jump = []

        for i in range(1, len(surface_curve_xpos)):

            #Calculate x distance from mth interface structure's center of mass
            difx = (surface_curve_xpos[i]-surface_curve_xpos[i-1])
            difx_abs = np.abs(difx)

            #Calculate y distance from mth interface structure's center of mass
            dify = (surface_curve_ypos[i]-surface_curve_ypos[i-1])
            dify_abs = np.abs(dify)

            #Enforce periodic boundary conditions
            if (difx_abs>=self.h_box) | (dify_abs>=self.h_box):

                if (difx_abs>=self.h_box):
                    if (difx > 0):
                        from_right_to_left.append(0)
                        from_left_to_right.append(1)
                        boundary_jump.append(1)

                    elif (difx < 0):
                        from_right_to_left.append(1)
                        from_left_to_right.append(0)
                        boundary_jump.append(1)
                else:
                    from_right_to_left.append(0)
                    from_left_to_right.append(0)

                if (dify_abs>=self.h_box):
                    if (dify > 0):
                        from_top_to_bot.append(0)
                        from_bot_to_top.append(1)
                        boundary_jump.append(1)

                    elif (dify < 0):
                        from_top_to_bot.append(1)
                        from_bot_to_top.append(0)
                        boundary_jump.append(1)
                else:
                    from_top_to_bot.append(0)
                    from_bot_to_top.append(0)

            else:
                from_right_to_left.append(0)
                from_left_to_right.append(0)
                from_top_to_bot.append(0)
                from_bot_to_top.append(0)
                boundary_jump.append(0)


        boundary_dict = {'jump': boundary_jump, 'l_to_r':from_left_to_right, 'r_to_l': from_right_to_left, 'b_to_t': from_bot_to_top, 't_to_b': from_top_to_bot}
        return boundary_dict

    def surface_com_pov(self, surface_curve_pos_dict):
        surface_curve_xpos = surface_curve_pos_dict['x']
        surface_curve_ypos = surface_curve_pos_dict['y']

        for i in range(1, len(surface_curve_xpos)):
            #Calculate x distance from mth interface structure's center of mass
            difx = (surface_curve_xpos[i]-surface_curve_xpos[i-1])
            difx_abs = np.abs(difx)

            #Calculate y distance from mth interface structure's center of mass
            dify = (surface_curve_ypos[i]-surface_curve_ypos[i-1])
            dify_abs = np.abs(dify)

            #Enforce periodic boundary conditions
            if (difx_abs>=self.h_box) | (dify_abs>=self.h_box):

                if (difx_abs>=self.h_box):
                    if (difx > 0):
                        surface_curve_xpos[i:] -= self.l_box
                    elif (difx < 0):
                        surface_curve_xpos[i:] += self.l_box

                if (dify_abs>=self.h_box):
                    if (dify > 0):
                        surface_curve_ypos[i:] -= self.l_box

                    elif (dify < 0):
                        surface_curve_ypos[i:] += self.l_box

        x_com = np.mean(surface_curve_xpos)
        if x_com > self.l_box:
            x_com -=self.l_box
            surface_curve_xpos -= self.l_box
        elif x_com < 0:
            x_com += self.l_box
            surface_curve_xpos += self.l_box

        y_com = np.mean(surface_curve_ypos)
        if y_com > self.l_box:
            y_com -=self.l_box
            surface_curve_ypos -= self.l_box
        elif y_com < 0:
            y_com += self.l_box
            surface_curve_ypos += self.l_box

        com_pos_dict = {'pos': {'x': surface_curve_xpos, 'y': surface_curve_ypos}, 'com': {'x': x_com, 'y': y_com}}

        return com_pos_dict

    def surface_com3(self, surface_curve_pos_dict):
        surface_curve_xpos = surface_curve_pos_dict['x']
        surface_curve_ypos = surface_curve_pos_dict['y']

        boundary_dict = surface_boundary_jumps(surface_curve_pos_dict)

        hor_wrap_left = []
        hor_wrap_right = []
        vert_wrap_bot = []
        vert_wrap_top = []

        for i in range(1, len(surface_curve_xpos)):

            #Calculate x distance from mth interface structure's center of mass
            difx = (surface_curve_xpos[i]-surface_curve_xpos[i-1])
            difx_abs = np.abs(difx)

            #Calculate y distance from mth interface structure's center of mass
            dify = (surface_curve_ypos[i]-surface_curve_ypos[i-1])
            dify_abs = np.abs(dify)

            #Enforce periodic boundary conditions
            if (difx_abs>=self.h_box) | (dify_abs>=self.h_box):

                if (difx_abs>=self.h_box):
                    if (difx > 0):
                        from_right_to_left.append(0)
                        from_left_to_right.append(1)
                        boundary_jump.append(1)

                    elif (difx < 0):
                        from_right_to_left.append(1)
                        from_left_to_right.append(0)
                        boundary_jump.append(1)
                else:
                    from_right_to_left.append(0)
                    from_left_to_right.append(0)

                if (dify_abs>=self.h_box):
                    if (dify > 0):
                        from_top_to_bot.append(0)
                        from_bot_to_top.append(1)
                        boundary_jump.append(1)

                    elif (dify < 0):
                        from_top_to_bot.append(1)
                        from_bot_to_top.append(0)
                        boundary_jump.append(1)
                else:
                    from_top_to_bot.append(0)
                    from_bot_to_top.append(0)

            else:
                from_right_to_left.append(0)
                from_left_to_right.append(0)
                from_top_to_bot.append(0)
                from_bot_to_top.append(0)
                boundary_jump.append(0)

                x_arr.append(surface_curve_xpos[i])
                y_arr.append(surface_curve_ypos[i])

            #np.where(boundary_jump==1)
            #for j in range(0, len(x_com)):
            #    if from_top_to_bot[j]==1:

            #Calculate magnitude of distance from center of mass of mth interface structure
            radius_val = (difx**2 + dify**2)**0.5

            #Save radius from CoM of bin
            radii.append(radius_val)

            #Calculate angle from CoM and x-axis
            theta_val = np.arctan2(dify, difx)*(180/math.pi)

            '''
            #Enforce correct quadrant for particle
            if (bub_rad_tmp_x>0) & (bub_rad_tmp_y>0):
                pass
            elif (bub_rad_tmp_x<0) & (bub_rad_tmp_y>0):
                theta_val = 180-theta_val
            elif (bub_rad_tmp_x<0) & (bub_rad_tmp_y<0):
                theta_val = theta_val+180
            elif (bub_rad_tmp_x>0) & (bub_rad_tmp_y<0):
                theta_val = 360-theta_val
            '''

            if theta_val < 0:
                theta_val = 360 + theta_val

            #Save calculated angle from CoM and x-axis
            thetas.append(theta_val)

        radius_dict = {'radius': {'vals': radii, 'mean': np.mean(radii)}, 'theta': {'vals': thetas, 'mean': np.mean(thetas)}}
        return radius_dict

    def surface_radius(self, surface_curve_pos_dict):

        surface_com_pos_dict = self.surface_com_pov(surface_curve_pos_dict)

        surface_com_xpos = surface_com_pos_dict['pos']['x']
        surface_com_ypos = surface_com_pos_dict['pos']['y']

        int_com_x = surface_com_pos_dict['com']['x']
        int_com_y = surface_com_pos_dict['com']['y']

        radii = []
        thetas = []

        for i in range(0, len(surface_com_xpos)):

            difx = self.utility_functs.sep_dist(surface_com_xpos[i], int_com_x)
            dify = self.utility_functs.sep_dist(surface_com_ypos[i], int_com_y)

            #Calculate magnitude of distance from center of mass of mth interface structure
            radius_val = (difx**2 + dify**2)**0.5

            #Save radius from CoM of bin
            radii.append(radius_val)

            #Calculate angle from CoM and x-axis
            theta_val = np.arctan2(dify, difx)#*(180/math.pi)


            #if theta_val < 0:
            #    theta_val = np.pi * 2 + theta_val

            #Save calculated angle from CoM and x-axis
            thetas.append(theta_val)
        thetas = np.array(thetas)
        radii = np.array(radii)

        mean_radius = np.mean(radii)

        #if there were interior bins found, calculate the average interior radius of mth interface structure
        std = 0
        for z in range(0, len(radii)):
            std+=(radii[z]-mean_radius)**2
        std_radius = (std/len(radii))**0.5

        radius_dict = {'values': {'radius': np.array(radii), 'theta': np.array(thetas)}, 'mean radius': np.mean(radii), 'std radius': std_radius}
        return radius_dict

    def surface_width(self, interior_surface_rad, exterior_surface_rad):


        #if there were exterior bins found, calculate the average exterior radius of mth interface structure

        surface_width = np.abs(exterior_surface_rad-interior_surface_rad)

        width_dict = {'width': surface_width}
        return width_dict

    def surface_alignment(self, surface_measurements, surface_curve, sep_surface_dict, int_dict, int_comp_dict):

        new_align = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_trad = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_trad_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_avg_trad_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_num = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_num0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_num1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_dif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        part_align = np.zeros(self.partNum)
        part_difr = np.zeros(self.partNum)

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

            interior_radius = surface_measurements[key]['interior']['mean radius']
            exterior_radius = surface_measurements[key]['exterior']['mean radius']

            interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
            exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])

            interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
            interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']

            interior_int_com_x = interior_surface_com_pos_dict['com']['x']
            interior_int_com_y = interior_surface_com_pos_dict['com']['y']

            exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
            exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']

            exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
            exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']

            ext_surface_id = sep_surface_dict[key]['exterior']['ids']
            int_surface_id = sep_surface_dict[key]['interior']['ids']

            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):

                    #If bin is part of mth interface structure, continue...
                    if int_id[ix][iy]==int_large_ids[m]:

                        if ext_surface_id[ix][iy]==0:

                            #Calculate position of exterior edge bin

                            xpos_ref = (ix+0.5)*self.sizeBin
                            ypos_ref = (iy+0.5)*self.sizeBin

                            difx_trad = self.utility_functs.sep_dist(xpos_ref, self.h_box)

                            dify_trad = self.utility_functs.sep_dist(ypos_ref, self.h_box)

                            difx_bub = self.utility_functs.sep_dist(xpos_ref, exterior_int_com_x)

                            dify_bub = self.utility_functs.sep_dist(ypos_ref, exterior_int_com_y)

                            x_norm_unitv = 0
                            y_norm_unitv = 0

                            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                            difr_short= ( (difx_bub )**2 + (dify_bub)**2)**0.5#( (difx_bub )**2 + (dify_bub)**2)**0.5#10000000.
                            difr_bub = ( (difx_bub )**2 + (dify_bub)**2)**0.5
                            difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5

                            difx_short = np.abs(difx_bub)
                            dify_short = np.abs(dify_bub)

                            x_norm_unitv_trad = (difx_trad) / difr_trad
                            y_norm_unitv_trad = (dify_trad) / difr_trad

                            x_norm_unitv = (difx_bub) / difr_bub
                            y_norm_unitv = (dify_bub) / difr_bub
                            #Loop over bins of system

                            for id in range(0, len(exterior_surface_com_xpos)):

                                difx_width = self.utility_functs.sep_dist(xpos_ref, exterior_surface_com_xpos[id])

                                dify_width = self.utility_functs.sep_dist(ypos_ref, exterior_surface_com_ypos[id])

                                #Calculate distance from interior edge bin to exterior edge bin
                                difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                #If this distance is the shortest calculated thus far, replace the value with it
                                if difr<difr_short:
                                    difr_short=difr
                                    difx_short = np.abs(difx_width)
                                    dify_short = np.abs(dify_width)
                                    x_norm_unitv = difx_width / difr
                                    y_norm_unitv = dify_width / difr

                            if len(self.binParts[ix][iy])>0:

                                for h in range(0, len(self.binParts[ix][iy])):
                                    #Calculate x and y orientation of active force
                                    px = np.sin(self.ang[self.binParts[ix][iy][h]])
                                    py = -np.cos(self.ang[self.binParts[ix][iy][h]])

                                    #Calculate alignment towards CoM
                                    if (difr_short == difr_bub) | ( exterior_radius<=interior_radius):
                                        x_dot_p = (-x_norm_unitv * px)
                                        y_dot_p = (-y_norm_unitv * py)
                                        r_dot_p = x_dot_p + y_dot_p
                                    else:
                                        x_dot_p = (x_norm_unitv * px)
                                        y_dot_p = (y_norm_unitv * py)
                                        r_dot_p = x_dot_p + y_dot_p

                                    x_dot_p_trad = (-x_norm_unitv_trad * px)
                                    y_dot_p_trad = (-y_norm_unitv_trad * py)
                                    r_dot_p_trad = x_dot_p_trad + y_dot_p_trad
                                    #Sum x,y orientation over each bin

                                    new_align[ix][iy] += r_dot_p
                                    new_align_x[ix][iy] += px
                                    new_align_y[ix][iy] += py
                                    new_align_num[ix][iy]+= 1
                                    new_align_trad[ix][iy] += r_dot_p_trad
                                    new_align_trad_x[ix][iy] += x_dot_p
                                    new_align_trad_y[ix][iy] += y_dot_p
                                    part_align[self.binParts[ix][iy][h]] = r_dot_p

                                    if self.typ[self.binParts[ix][iy][h]]==0:
                                        new_align0[ix][iy] += r_dot_p
                                        new_align_num0[ix][iy]+= 1
                                        new_align_trad0[ix][iy] += r_dot_p_trad
                                    elif self.typ[self.binParts[ix][iy][h]]==1:
                                        new_align1[ix][iy] += r_dot_p
                                        new_align_num1[ix][iy]+= 1
                                        new_align_trad1[ix][iy] += r_dot_p_trad
                        #if ext_edge_id[ix][iy]==0:


                        if ext_surface_id[ix][iy]==1:


                            #Calculate position of exterior edge bin
                            xpos_ref = (ix+0.5)*self.sizeBin
                            ypos_ref = (iy+0.5)*self.sizeBin

                            difx_trad = self.utility_functs.sep_dist(xpos_ref, self.h_box)

                            dify_trad = self.utility_functs.sep_dist(ypos_ref, self.h_box)

                            difx_bub = self.utility_functs.sep_dist(xpos_ref, exterior_int_com_x)

                            dify_bub = self.utility_functs.sep_dist(ypos_ref, exterior_int_com_y)

                            x_norm_unitv = 0
                            y_norm_unitv = 0

                            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                            difr_short= ( (difx_bub )**2 + (dify_bub)**2)**0.5#( (difx_bub )**2 + (dify_bub)**2)**0.5#10000000.
                            difr_bub = ( (difx_bub )**2 + (dify_bub)**2)**0.5
                            difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                            difx_short = np.abs(difx_bub)
                            dify_short = np.abs(dify_bub)
                            x_norm_unitv_trad = (difx_trad) / difr_trad
                            y_norm_unitv_trad = (dify_trad) / difr_trad

                            x_norm_unitv = (difx_bub) / difr_bub
                            y_norm_unitv = (dify_bub) / difr_bub
                            #Loop over bins of system

                            for id in range(0, len(interior_surface_com_xpos)):

                                #Calculate position of interior edge bin

                                difx_width = self.utility_functs.sep_dist(xpos_ref, interior_surface_com_xpos[id])

                                dify_width = self.utility_functs.sep_dist(ypos_ref, interior_surface_com_ypos[id])

                                #Calculate distance from interior edge bin to exterior edge bin
                                difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                #If this distance is the shortest calculated thus far, replace the value with it
                                if difr<difr_short:
                                    difr_short=difr
                                    difx_short = np.abs(difx_width)
                                    dify_short = np.abs(dify_width)
                                    x_norm_unitv = difx_width / difr
                                    y_norm_unitv = dify_width / difr




                            if len(self.binParts[ix][iy])>0:
                                for h in range(0, len(self.binParts[ix][iy])):

                                    px = np.sin(self.ang[self.binParts[ix][iy][h]])
                                    py = -np.cos(self.ang[self.binParts[ix][iy][h]])
                                    #Calculate alignment towards CoM

                                    if (difr_short == difr_bub) | ( exterior_radius>interior_radius):
                                        x_dot_p = (-x_norm_unitv * px)
                                        y_dot_p = (-y_norm_unitv * py)
                                        r_dot_p = x_dot_p + y_dot_p

                                    else:
                                        x_dot_p = (x_norm_unitv * px)
                                        y_dot_p = (y_norm_unitv * py)
                                        r_dot_p = x_dot_p + y_dot_p

                                    x_dot_p_trad = (-x_norm_unitv_trad * px)
                                    y_dot_p_trad = (-y_norm_unitv_trad * py)
                                    r_dot_p_trad = x_dot_p_trad + y_dot_p_trad


                                    #Sum x,y orientation over each bin
                                    new_align[ix][iy] += r_dot_p
                                    new_align_x[ix][iy] += x_dot_p
                                    new_align_y[ix][iy] += y_dot_p
                                    new_align_x[ix][iy] += px
                                    new_align_y[ix][iy] += py
                                    new_align_num[ix][iy]+= 1
                                    new_align_trad[ix][iy] += r_dot_p_trad
                                    new_align_trad_x[ix][iy] += x_dot_p_trad
                                    new_align_trad_y[ix][iy] += y_dot_p_trad
                                    part_align[self.binParts[ix][iy][h]] = r_dot_p
                                    part_difr[self.binParts[ix][iy][h]] = difr_short
                                    if self.typ[self.binParts[ix][iy][h]]==0:
                                        new_align0[ix][iy] += r_dot_p
                                        new_align_x0[ix][iy] += x_dot_p
                                        new_align_y0[ix][iy] += y_dot_p
                                        new_align_num0[ix][iy]+= 1
                                        new_align_trad0[ix][iy] += r_dot_p_trad
                                        new_align_trad_x0[ix][iy] += x_dot_p_trad
                                        new_align_trad_y0[ix][iy] += y_dot_p_trad
                                    elif self.typ[self.binParts[ix][iy][h]]==1:
                                        new_align1[ix][iy] += r_dot_p
                                        new_align_x1[ix][iy] += x_dot_p
                                        new_align_y1[ix][iy] += y_dot_p
                                        new_align_num1[ix][iy]+= 1
                                        new_align_trad1[ix][iy] += r_dot_p_trad
                                        new_align_trad_x1[ix][iy] += x_dot_p_trad
                                        new_align_trad_y1[ix][iy] += y_dot_p_trad

                    #If bin is an exterior bin of mth interface structure, continue...
                    if new_align_num[ix][iy]>0:
                    #    if new_align_avg[ix][iy]==0:
                        new_align_avg[ix][iy] = new_align[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_x[ix][iy] = new_align_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_y[ix][iy] = new_align_y[ix][iy] / new_align_num[ix][iy]

                        new_align_avg_trad[ix][iy] = new_align_trad[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_x[ix][iy] = new_align_trad_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_y[ix][iy] = new_align_trad_y[ix][iy] / new_align_num[ix][iy]

                        if new_align_num0[ix][iy]>0:
                            new_align_avg0[ix][iy] = new_align0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_x0[ix][iy] = new_align_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_y0[ix][iy] = new_align_y0[ix][iy] / new_align_num0[ix][iy]

                            new_align_avg_trad0[ix][iy] = new_align_trad0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_x0[ix][iy] = new_align_trad_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_y0[ix][iy] = new_align_trad_y0[ix][iy] / new_align_num0[ix][iy]

                        if new_align_num1[ix][iy]>0:
                            new_align_avg1[ix][iy] = new_align1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_x1[ix][iy] = new_align_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_y1[ix][iy] = new_align_y1[ix][iy] / new_align_num1[ix][iy]

                            new_align_avg_trad1[ix][iy] = new_align_trad1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_x1[ix][iy] = new_align_trad_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_y1[ix][iy] = new_align_trad_y1[ix][iy] / new_align_num1[ix][iy]

                        if new_align_num1[ix][iy]>0:
                            if new_align_num0[ix][iy]>0:
                                new_align_avg_dif[ix][iy] = np.abs(new_align_avg1[ix][iy]) - np.abs(new_align_avg0[ix][iy])



        method1_align_dict = {'bin': {'all': {'x': new_align_avg_trad_x, 'y': new_align_avg_trad_y, 'mag': new_align_avg_trad, 'num': new_align_num}, 'A': {'x': new_align_avg_trad_x0, 'y': new_align_avg_trad_y0, 'mag': new_align_avg_trad0, 'num': new_align_num0}, 'B': {'x': new_align_avg_trad_x1, 'y': new_align_avg_trad_y1, 'mag': new_align_avg_trad1, 'num': new_align_num1}}, 'part': part_align}
        method2_align_dict = {'bin': {'all': {'x': new_align_avg_x, 'y': new_align_avg_y, 'mag': new_align_avg, 'num': new_align_num}, 'A': {'x': new_align_avg_x0, 'y': new_align_avg_y0, 'mag': new_align_avg0, 'num': new_align_num0}, 'B': {'x': new_align_avg_x1, 'y': new_align_avg_y1, 'mag': new_align_avg1, 'num': new_align_num1}}, 'part': {'align': part_align, 'difr': part_difr}}
        return  method1_align_dict, method2_align_dict
    def bulk_alignment(self, method1_align_dict, method2_align_dict, surface_measurements, surface_curve, sep_surface_dict, bulk_dict, bulk_comp_dict, int_comp_dict):
        #Calculate alignment of bulk particles
        #Loop over all bulk bins identified

        new_align = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_dif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        bulk_large_ids = bulk_comp_dict['ids']['bulk id']

        bulk_id = bulk_dict['bin']

        new_align_avg_trad = method1_align_dict['bin']['all']['mag']
        new_align_avg_trad0 = method1_align_dict['bin']['A']['mag']
        new_align_avg_trad1 = method1_align_dict['bin']['B']['mag']

        new_align_avg_trad_x = method1_align_dict['bin']['all']['x']
        new_align_avg_trad_x0 = method1_align_dict['bin']['A']['x']
        new_align_avg_trad_x1 = method1_align_dict['bin']['B']['x']

        new_align_avg_trad_y = method1_align_dict['bin']['all']['y']
        new_align_avg_trad_y0 = method1_align_dict['bin']['A']['y']
        new_align_avg_trad_y1 = method1_align_dict['bin']['B']['y']

        new_align_avg = method2_align_dict['bin']['all']['mag']
        new_align_avg0 = method2_align_dict['bin']['A']['mag']
        new_align_avg1 = method2_align_dict['bin']['B']['mag']

        new_align_avg_x = method2_align_dict['bin']['all']['x']
        new_align_avg_x0 = method2_align_dict['bin']['A']['x']
        new_align_avg_x1 = method2_align_dict['bin']['B']['x']

        new_align_avg_y = method2_align_dict['bin']['all']['y']
        new_align_avg_y0 = method2_align_dict['bin']['A']['y']
        new_align_avg_y1 = method2_align_dict['bin']['B']['y']

        new_align_num = method1_align_dict['bin']['all']['num']
        new_align_num0 = method1_align_dict['bin']['A']['num']
        new_align_num1 = method1_align_dict['bin']['B']['num']

        part_align = method2_align_dict['part']['align']
        part_difr = method2_align_dict['part']['difr']

        for m in range(0, len(bulk_large_ids)):

            for ix in range(0, self.NBins):
                for iy in range(0, self.NBins):
                    if bulk_id[ix][iy] == bulk_large_ids[m]:
                        #If bin is part of mth interface structure, continue...
                        if new_align_num[ix][iy]==0:
                            #Calculate position of exterior edge bin
                            difr_short = 100000
                            xpos_ref = (ix+0.5)*self.sizeBin
                            ypos_ref = (iy+0.5)*self.sizeBin

                            difx_trad = self.utility_functs.sep_dist(xpos_ref, self.h_box)

                            dify_trad = self.utility_functs.sep_dist(ypos_ref, self.h_box)

                            difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                            difr_bub= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                            difr_short= 1000000
                            x_norm_unitv = difx_trad / difr_trad
                            y_norm_unitv = dify_trad / difr_trad

                            x_norm_unitv_trad = (difx_trad) / difr_trad
                            y_norm_unitv_trad = (dify_trad) / difr_trad

                            for m in range(0, len(sep_surface_dict)):
                                key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

                                interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                                exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])

                                interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
                                interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']

                                interior_int_com_x = interior_surface_com_pos_dict['com']['x']
                                interior_int_com_y = interior_surface_com_pos_dict['com']['y']

                                exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
                                exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']

                                exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
                                exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']


                                difx_ext = self.utility_functs.sep_dist(xpos_ref, exterior_int_com_x)

                                dify_ext = self.utility_functs.sep_dist(ypos_ref, exterior_int_com_y)

                                difr_ext= ( (difx_ext )**2 + (dify_ext)**2)**0.5
                                if difr_ext < difr_short:
                                    difr_short= ( (difx_ext )**2 + (dify_ext)**2)**0.5
                                    x_norm_unitv = difx_ext / difr_short
                                    y_norm_unitv = dify_ext / difr_short


                                difx_int = self.utility_functs.sep_dist(xpos_ref, interior_int_com_x)

                                dify_int = self.utility_functs.sep_dist(ypos_ref, interior_int_com_y)

                                difr_int= ( (difx_int )**2 + (dify_int)**2)**0.5
                                if difr_int < difr_short:
                                    difr_short= ( (difx_int )**2 + (dify_int)**2)**0.5
                                    x_norm_unitv = difx_int / difr_short
                                    y_norm_unitv = dify_int / difr_short



                                #Loop over bins of system
                                for id in range(0, len(exterior_surface_com_xpos)):
                                    #If bin is an interior edge bin for mth interface structure, continue...

                                    difx_width = self.utility_functs.sep_dist(xpos_ref, exterior_surface_com_xpos[id])

                                    dify_width = self.utility_functs.sep_dist(ypos_ref, exterior_surface_com_ypos[id])

                                    #Calculate distance from interior edge bin to exterior edge bin
                                    difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                    #If this distance is the shortest calculated thus far, replace the value with it
                                    if difr<difr_short:
                                        difr_short=difr
                                        x_norm_unitv = difx_width / difr
                                        y_norm_unitv = dify_width / difr
                                        interior_bin_short = 0
                                        exterior_bin_short = 1
                                for id in range(0, len(interior_surface_com_xpos)):
                                    #If bin is an interior edge bin for mth interface structure, continue...

                                    #Calculate position of interior edge bin
                                    difx_width = self.utility_functs.sep_dist(xpos_ref, interior_surface_com_xpos[id])

                                    dify_width = self.utility_functs.sep_dist(ypos_ref, interior_surface_com_ypos[id])


                                    #Calculate distance from interior edge bin to exterior edge bin
                                    difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                    #If this distance is the shortest calculated thus far, replace the value with it
                                    if difr<difr_short:
                                        difr_short=difr
                                        x_norm_unitv = difx_width / difr
                                        y_norm_unitv = dify_width / difr
                                        interior_bin_short = 1
                                        exterior_bin_short = 0
                            #If particles in bin, continue...
                            if len(self.binParts[ix][iy])>0:
                                #Loop over all particles in bin
                                for h in range(0, len(self.binParts[ix][iy])):
                                    #Calculate x and y orientation of active force
                                    px = np.sin(self.ang[self.binParts[ix][iy][h]])
                                    py = -np.cos(self.ang[self.binParts[ix][iy][h]])

                                    #Calculate alignment of single particle with nearest surface

                                    if (interior_bin_short == 1) | (difr_short == difr_bub):
                                        x_dot_p = (-x_norm_unitv * px)
                                        y_dot_p = (-y_norm_unitv * py)
                                    elif exterior_bin_short == 1:
                                        x_dot_p = (x_norm_unitv * px)
                                        y_dot_p = (y_norm_unitv * py)

                                    r_dot_p = x_dot_p + y_dot_p

                                    #Calculate alignment of single particle with cluster's Center of mass
                                    x_dot_p_trad = (-x_norm_unitv_trad * px)
                                    y_dot_p_trad =  (-y_norm_unitv_trad * py)
                                    r_dot_p_trad = x_dot_p_trad + y_dot_p_trad

                                    #Save alignment of each particle
                                    part_align[self.binParts[ix][iy][h]] = r_dot_p
                                    part_difr[self.binParts[ix][iy][h]] = difr_short
                                    #Calculate alignment of all particles per bin
                                    new_align[ix][iy] += r_dot_p
                                    new_align_x[ix][iy] += x_dot_p
                                    new_align_y[ix][iy] += y_dot_p
                                    new_align_num[ix][iy]+= 1
                                    new_align_trad[ix][iy] += r_dot_p_trad
                                    new_align_trad_x[ix][iy] += x_dot_p_trad
                                    new_align_trad_y[ix][iy] += y_dot_p_trad

                                    #if particle type is B, add to total alignment
                                    if self.typ[self.binParts[ix][iy][h]]==0:
                                        new_align0[ix][iy] += r_dot_p
                                        new_align_x0[ix][iy] += x_dot_p
                                        new_align_y0[ix][iy] += y_dot_p
                                        new_align_num0[ix][iy]+= 1
                                        new_align_trad0[ix][iy] += r_dot_p_trad
                                        new_align_trad_x0[ix][iy] += x_dot_p_trad
                                        new_align_trad_y0[ix][iy] += y_dot_p_trad

                                    #if particle type is B, add to total alignment
                                    elif self.typ[self.binParts[ix][iy][h]]==1:
                                        new_align1[ix][iy] += r_dot_p
                                        new_align_x1[ix][iy] += x_dot_p
                                        new_align_y1[ix][iy] += y_dot_p
                                        new_align_num1[ix][iy]+= 1
                                        new_align_trad1[ix][iy] += r_dot_p_trad
                                        new_align_trad_x1[ix][iy] += x_dot_p_trad
                                        new_align_trad_y1[ix][iy] += y_dot_p_trad


        #Calculate average alignment of bulk bins
        #Loop over bins in system
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                #If particle is notpart of an interface, continue
                if bulk_id[ix][iy]>0:
                    #If summed alignment with nearest surface greater than zero (non-gas), continue...
                    if new_align_num[ix][iy]>0:
                        new_align_avg[ix][iy] = new_align[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_x[ix][iy] = new_align_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_y[ix][iy] = new_align_y[ix][iy] / new_align_num[ix][iy]

                        new_align_avg_trad[ix][iy] = new_align_trad[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_x[ix][iy] = new_align_trad_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_y[ix][iy] = new_align_trad_y[ix][iy] / new_align_num[ix][iy]

                        if new_align_num0[ix][iy]>0:
                            new_align_avg0[ix][iy] = new_align0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_x0[ix][iy] = new_align_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_y0[ix][iy] = new_align_y0[ix][iy] / new_align_num0[ix][iy]

                            new_align_avg_trad0[ix][iy] = new_align_trad0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_x0[ix][iy] = new_align_trad_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_y0[ix][iy] = new_align_trad_y0[ix][iy] / new_align_num0[ix][iy]

                        if new_align_num1[ix][iy]>0:
                            new_align_avg1[ix][iy] = new_align1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_x1[ix][iy] = new_align_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_y1[ix][iy] = new_align_y1[ix][iy] / new_align_num1[ix][iy]

                            new_align_avg_trad1[ix][iy] = new_align_trad1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_x1[ix][iy] = new_align_trad_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_y1[ix][iy] = new_align_trad_y1[ix][iy] / new_align_num1[ix][iy]

        method1_align_dict = {'bin': {'all': {'x': new_align_avg_trad_x, 'y': new_align_avg_trad_y, 'mag': new_align_avg_trad, 'num': new_align_num}, 'A': {'x': new_align_avg_trad_x0, 'y': new_align_avg_trad_y0, 'mag': new_align_avg_trad0, 'num': new_align_num0}, 'B': {'x': new_align_avg_trad_x1, 'y': new_align_avg_trad_y1, 'mag': new_align_avg_trad1, 'num': new_align_num1}}, 'part': part_align}
        method2_align_dict = {'bin': {'all': {'x': new_align_avg_x, 'y': new_align_avg_y, 'mag': new_align_avg, 'num': new_align_num}, 'A': {'x': new_align_avg_x0, 'y': new_align_avg_y0, 'mag': new_align_avg0, 'num': new_align_num0}, 'B': {'x': new_align_avg_x1, 'y': new_align_avg_y1, 'mag': new_align_avg1, 'num': new_align_num1}}, 'part': {'align': part_align, 'difr': part_difr}}
        return  method1_align_dict, method2_align_dict
    def gas_alignment(self, method1_align_dict, method2_align_dict, surface_measurements, surface_curve, sep_surface_dict, int_comp_dict):

        new_align = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_trad_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_x1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y0 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        new_align_trad_y1 = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        new_align_avg_trad = method1_align_dict['bin']['all']['mag']
        new_align_avg_trad0 = method1_align_dict['bin']['A']['mag']
        new_align_avg_trad1 = method1_align_dict['bin']['B']['mag']

        new_align_avg_trad_x = method1_align_dict['bin']['all']['x']
        new_align_avg_trad_x0 = method1_align_dict['bin']['A']['x']
        new_align_avg_trad_x1 = method1_align_dict['bin']['B']['x']

        new_align_avg_trad_y = method1_align_dict['bin']['all']['y']
        new_align_avg_trad_y0 = method1_align_dict['bin']['A']['y']
        new_align_avg_trad_y1 = method1_align_dict['bin']['B']['y']

        new_align_avg = method2_align_dict['bin']['all']['mag']
        new_align_avg0 = method2_align_dict['bin']['A']['mag']
        new_align_avg1 = method2_align_dict['bin']['B']['mag']

        new_align_avg_x = method2_align_dict['bin']['all']['x']
        new_align_avg_x0 = method2_align_dict['bin']['A']['x']
        new_align_avg_x1 = method2_align_dict['bin']['B']['x']

        new_align_avg_y = method2_align_dict['bin']['all']['y']
        new_align_avg_y0 = method2_align_dict['bin']['A']['y']
        new_align_avg_y1 = method2_align_dict['bin']['B']['y']

        new_align_num = method1_align_dict['bin']['all']['num']
        new_align_num0 = method1_align_dict['bin']['A']['num']
        new_align_num1 = method1_align_dict['bin']['B']['num']

        part_align = method2_align_dict['part']['align']
        part_difr = method2_align_dict['part']['difr']

        #Calculate alignment of gas bins
        #Loop over bins in system
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if new_align_avg[ix][iy]==0:
                    #Calculate position of exterior edge bin
                    xpos_ref = (ix+0.5)*self.sizeBin
                    ypos_ref = (iy+0.5)*self.sizeBin


                    difr_short= 100000
                    #Loop over bins of system
                    if len(sep_surface_dict) >= 0:


                        difx_trad = self.utility_functs.sep_dist(xpos_ref, self.h_box)

                        dify_trad = self.utility_functs.sep_dist(ypos_ref, self.h_box)

                        #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                        difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                        difr_bub= ( (difx_trad )**2 + (dify_trad)**2)**0.5#10000000.

                        x_norm_unitv = difx_trad / difr_trad
                        y_norm_unitv = dify_trad / difr_trad

                        x_norm_unitv_trad = (difx_trad) / difr_trad
                        y_norm_unitv_trad = (dify_trad) / difr_trad
                    if len(sep_surface_dict) >= 1:
                        for m in range(0, len(sep_surface_dict)):
                            key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))

                            interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                            exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])

                            interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
                            interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']

                            interior_int_com_x = interior_surface_com_pos_dict['com']['x']
                            interior_int_com_y = interior_surface_com_pos_dict['com']['y']

                            exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
                            exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']

                            exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
                            exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']
                            for id in range(0, len(exterior_surface_com_xpos)):
                                #If bin is an interior edge bin for mth interface structure, continue...

                                #Calculate position of interior edge bin
                                difx_width = self.utility_functs.sep_dist(xpos_ref, exterior_surface_com_xpos[id])

                                dify_width = self.utility_functs.sep_dist(ypos_ref, exterior_surface_com_ypos[id])

                                #Calculate distance from interior edge bin to exterior edge bin
                                difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                #If this distance is the shortest calculated thus far, replace the value with it
                                if difr<difr_short:
                                    difr_short=difr
                                    x_norm_unitv = difx_width / difr
                                    y_norm_unitv = dify_width / difr
                                    interior_bin_short = 0
                                    exterior_bin_short = 1
                            for id in range(0, len(interior_surface_com_xpos)):
                                #If bin is an interior edge bin for mth interface structure, continue...

                                #Calculate position of interior edge bin
                                difx_width = self.utility_functs.sep_dist(xpos_ref, exterior_surface_com_xpos[id])

                                dify_width = self.utility_functs.sep_dist(ypos_ref, exterior_surface_com_ypos[id])

                                #Calculate distance from interior edge bin to exterior edge bin
                                difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                #If this distance is the shortest calculated thus far, replace the value with it
                                if difr<difr_short:
                                    difr_short=difr
                                    x_norm_unitv = difx_width / difr
                                    y_norm_unitv = dify_width / difr
                                    interior_bin_short = 1
                                    exterior_bin_short = 0

                    #If particles in bin, continue...
                    if len(self.binParts[ix][iy])>0:
                        #Loop over particles in bin
                        for h in range(0, len(self.binParts[ix][iy])):

                            #Calculate x and y orientation of reference particle's active force
                            px = np.sin(self.ang[self.binParts[ix][iy][h]])
                            py = -np.cos(self.ang[self.binParts[ix][iy][h]])

                            #If nearest surface is exterior surface, calculate alignment with that surface
                            if exterior_bin_short == 1:
                                x_dot_p = (-x_norm_unitv * px)
                                y_dot_p = (-y_norm_unitv * py)

                            #If nearest surface is interior surface, calculate alignment with that surface
                            elif interior_bin_short == 1:
                                x_dot_p = (x_norm_unitv * px)
                                y_dot_p = (y_norm_unitv * px)

                            r_dot_p = x_dot_p + y_dot_p

                            #Calculate alignment towards CoM
                            x_dot_p_trad = (-x_norm_unitv_trad * px)
                            y_dot_p_trad = (-y_norm_unitv_trad * py)
                            r_dot_p_trad = x_dot_p_trad + y_dot_p_trad

                            #Calculate total alignment and number of particles per bin for all particles

                            part_align[self.binParts[ix][iy][h]] = r_dot_p
                            part_difr[self.binParts[ix][iy][h]] = difr_short
                            new_align[ix][iy] += r_dot_p
                            new_align_x[ix][iy] += x_dot_p
                            new_align_y[ix][iy] += y_dot_p
                            new_align_num[ix][iy]+= 1
                            new_align_trad[ix][iy] += r_dot_p_trad
                            new_align_trad_x[ix][iy] += x_dot_p_trad
                            new_align_trad_y[ix][iy] += y_dot_p_trad

                            #Calculate total alignment and number of particles per bin for type A particles
                            if self.typ[self.binParts[ix][iy][h]]==0:
                                new_align0[ix][iy] += r_dot_p
                                new_align_x0[ix][iy] += x_dot_p
                                new_align_y0[ix][iy] += y_dot_p
                                new_align_num0[ix][iy]+= 1
                                new_align_trad0[ix][iy] += r_dot_p_trad
                                new_align_trad_x0[ix][iy] += x_dot_p_trad
                                new_align_trad_y0[ix][iy] += y_dot_p_trad

                            #Calculate total alignment and number of particles per bin for type B particles
                            elif self.typ[self.binParts[ix][iy][h]]==1:
                                new_align1[ix][iy] += r_dot_p
                                new_align_x1[ix][iy] += x_dot_p
                                new_align_y1[ix][iy] += y_dot_p
                                new_align_num1[ix][iy]+= 1
                                new_align_trad1[ix][iy] += r_dot_p_trad
                                new_align_trad_x1[ix][iy] += x_dot_p_trad
                                new_align_trad_y1[ix][iy] += y_dot_p_trad


        #Calculate average alignment per bin
        #Loop over bins in system
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if new_align_avg[ix][iy]==0:

                    #Calculate alignment with nearest interfacial surface
                    #If denominator is non-zero, continue...
                    if new_align_num[ix][iy]>0:
                        new_align_avg[ix][iy] = new_align[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_x[ix][iy] = new_align_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_y[ix][iy] = new_align_y[ix][iy] / new_align_num[ix][iy]

                        new_align_avg_trad[ix][iy] = new_align_trad[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_x[ix][iy] = new_align_trad_x[ix][iy] / new_align_num[ix][iy]
                        new_align_avg_trad_y[ix][iy] = new_align_trad_y[ix][iy] / new_align_num[ix][iy]
                        if new_align_num0[ix][iy]>0:
                            new_align_avg0[ix][iy] = new_align0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_x0[ix][iy] = new_align_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_y0[ix][iy] = new_align_y0[ix][iy] / new_align_num0[ix][iy]

                            new_align_avg_trad0[ix][iy] = new_align_trad0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_x0[ix][iy] = new_align_trad_x0[ix][iy] / new_align_num0[ix][iy]
                            new_align_avg_trad_y0[ix][iy] = new_align_trad_y0[ix][iy] / new_align_num0[ix][iy]
                        if new_align_num1[ix][iy]>0:
                            new_align_avg1[ix][iy] = new_align1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_x1[ix][iy] = new_align_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_y1[ix][iy] = new_align_y1[ix][iy] / new_align_num1[ix][iy]

                            new_align_avg_trad1[ix][iy] = new_align_trad1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_x1[ix][iy] = new_align_trad_x1[ix][iy] / new_align_num1[ix][iy]
                            new_align_avg_trad_y1[ix][iy] = new_align_trad_y1[ix][iy] / new_align_num1[ix][iy]

        method1_align_dict = {'bin': {'all': {'x': new_align_avg_trad_x, 'y': new_align_avg_trad_y, 'mag': new_align_avg_trad, 'num': new_align_num}, 'A': {'x': new_align_avg_trad_x0, 'y': new_align_avg_trad_y0, 'mag': new_align_avg_trad0, 'num': new_align_num0}, 'B': {'x': new_align_avg_trad_x1, 'y': new_align_avg_trad_y1, 'mag': new_align_avg_trad1, 'num': new_align_num1}}, 'part': part_align}
        method2_align_dict = {'bin': {'all': {'x': new_align_avg_x, 'y': new_align_avg_y, 'mag': new_align_avg, 'num': new_align_num}, 'A': {'x': new_align_avg_x0, 'y': new_align_avg_y0, 'mag': new_align_avg0, 'num': new_align_num0}, 'B': {'x': new_align_avg_x1, 'y': new_align_avg_y1, 'mag': new_align_avg1, 'num': new_align_num1}}, 'part': {'align': part_align, 'difr': part_difr}}
        return  method1_align_dict, method2_align_dict
    def fourier_analysis(self, radius_dict, n_len = 21):

        def fourier(x, *a):
            ret = a[1]
            for deg in range(1, int(len(a)/2)):
                ret += (a[(deg*2)] * np.cos(deg * (x-a[0]))) + (a[2*deg+1] * np.sin(deg * (x-a[0])))
            return ret

        from symfit import parameters, variables, sin, cos, Fit
        import numpy as np
        import matplotlib.pyplot as plt

        def fourier_series(x, f, n=0):
            """
            Returns a symbolic fourier series of order `n`.

            :param n: Order of the fourier series.
            :param x: Independent variable
            :param f: Frequency of the fourier series
            """
            # Make the parameter objects for all the terms
            a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
            sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
            # Construct the series
            series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                             for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
            return series

        x, y = variables('x, y')
        w, = parameters('w')

        num_coeff=11

        model_dict = {y: fourier_series(x, f=w, n=num_coeff)}

        radius = radius_dict['values']['radius']
        theta = radius_dict['values']['theta']

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=theta, y=radius)
        fit_result = fit.execute()

        coeffs = np.array([fit_result.params['a0']])
        n_arr = np.array([0])
        for j in range(1, num_coeff+1):
            a_key = 'a' + str(j)
            b_key = 'b' + str(j)
            n_arr = np.append(n_arr, j)
            coeffs = np.append(coeffs, (fit_result.params[a_key]**2 + fit_result.params[b_key]**2)**0.5)

        fourier_dict = {'modes': n_arr, 'coeffs': coeffs}
        return fourier_dict
