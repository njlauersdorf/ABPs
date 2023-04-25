
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

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility

# Class of interface identification functions
class interface:
    def __init__(self, area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ, ang):

        # Array (NBins_x, NBins_y) of average alignment of all particles per bin in either direction ('x', 'y', or 'mag'nitude)
        self.align_x = align_dict['bin']['all']['x']
        self.align_y = align_dict['bin']['all']['y']
        self.align_mag = align_dict['bin']['all']['mag']

        # Array (NBins_x, NBins_y) of average alignment of A particles per bin in either direction ('x', 'y', or 'mag'nitude)
        self.align_x_A = align_dict['bin']['A']['x']
        self.align_y_A = align_dict['bin']['A']['y']
        self.align_mag_A = align_dict['bin']['A']['mag']

        # Array (NBins_x, NBins_y) of average alignment of B particles per bin in either direction ('x', 'y', or 'mag'nitude)
        self.align_x_B = align_dict['bin']['B']['x']
        self.align_y_B = align_dict['bin']['B']['y']
        self.align_mag_B = align_dict['bin']['B']['mag']

        # Array (NBins_x, NBins_y) of average area fraction of each particle type ('all', 'A', or 'B') per bin
        self.area_frac = area_frac_dict['bin']['all']
        self.area_frac_A = area_frac_dict['bin']['A']
        self.area_frac_B = area_frac_dict['bin']['B']

        # Array (NBins_x, NBins_y) of whether bin is part of a cluster (1) or not (0)
        self.occParts = part_dict['clust']

        # Array (NBins_x, NBins_y) of binned particle ids
        self.binParts = part_dict['id']

        # Array (NBins_x, NBins_y) of binned particle types
        self.typParts = part_dict['typ']

        # Array (NBins_x, NBins_y) of average aligned active force pressure of each particle type ('all', 'A', or 'B') per bin
        self.press = press_dict['bin']['all']
        self.press_A = press_dict['bin']['A']
        self.press_B = press_dict['bin']['B']

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        # Initialize utility functions for call back later
        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        # Number of particles
        self.partNum = partNum

        # Minimum cluster size
        self.min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen

        try:
            # Total number of bins in x-direction
            self.NBins_x = int(NBins_x)

            # Total number of bins in y-direction
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        # X-length of bin
        self.sizeBin_x = self.utility_functs.roundUp((self.lx_box / self.NBins_x), 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp((self.ly_box / self.NBins_y), 6)
        
        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Fraction of A type particles in system
        self.parFrac = parFrac

        # Net (average) particle activity
        self.peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # Calculate lattice spacing from theory
        lat_theory = self.theory_functs.conForRClust(self.peNet, eps)

        # Calculate interparticle pressure from theory
        curPLJ = self.theory_functs.ljPress(lat_theory, self.peNet, eps)

        # Calculate dense phase area fraction from theory
        self.phi_theory = self.theory_functs.latToPhi(lat_theory)

        # Calculate gas phase area fraction from theory
        self.phi_g_theory = self.theory_functs.compPhiG(self.peNet, lat_theory)

        # Array (partNum) of particle types
        self.typ = typ

        # Array (partNum) of particle orientations [-pi,pi]
        self.ang = ang

    def det_surface_points(self, phase_dict, int_dict, int_comp_dict):
        '''
        Purpose: Takes the phase and interface ids of each bin and computes the interior and exterior
        surface bin ids and locations for that interface

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        int_dict: dictionary of arrays labeling interface id of each bin and particle

        int_comp_dict: dictionary of arrays containing composition information of each interface

        Outputs:
        surface_dict: dictionary containing x and y positions and ids of bins belonging to exterior (1) or interior surface (2)
        '''

        # Array (NBins_x, NBins_y) identifying whether bin is part of bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (NBins_x, NBins_y) identifying ids of which interface that bin is a part of
        int_id = int_dict['bin']

        # ID (int) of largest interface per number of particles
        int_large_ids = int_comp_dict['ids']

        # Instantiate empty array (NBins_x, NBins_y) that labels whether bin is part of exterior surface (1) or not (0)
        surface1_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)        #Label exterior edges of interfaces

        # Instantiate empty array (NBins_x, NBins_y) that labels whether bin is part of interior surface (1) or not (0)
        surface2_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)        #Label interior edges of interfaces

        # Instantiate empty array (partNum) that labels whether particle is part of exterior surface bins (1) or not (0)
        surface1_phaseInt=np.zeros(self.partNum)

        # Instantiate empty array (partNum) that labels whether particle is part of interior surface bins (1) or not (0)
        surface2_phaseInt=np.zeros(self.partNum)

        # Individually label each interface until all edge bins identified using flood fill algorithm
        if len(int_large_ids)>0:

            # Loop over all bins
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):

                    #If bin is an interface, continue
                    if phaseBin[ix][iy]==1:

                        #Count surrounding bin phases
                        gas_neigh_num=0
                        bulk_neigh_num=0

                        #identify neighboring x-bins
                        if (ix + 1) == self.NBins_x:
                            lookx = [ix-1, ix, 0]
                        elif ix==0:
                            lookx=[self.NBins_x-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, ix+1]

                        #identify neighboring y-bins
                        if (iy + 1) == self.NBins_y:
                            looky = [iy-1, iy, 0]
                        elif iy==0:
                            looky=[self.NBins_y-1, iy, iy+1]
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
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
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
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if surface2_id[ix][iy]==1:
                    surface2_pos_x=np.append(surface2_pos_x, (ix+0.5)*self.sizeBin_x)
                    surface2_pos_y=np.append(surface2_pos_y, (iy+0.5)*self.sizeBin_y)

                elif surface1_id[ix][iy]==1:
                    surface1_pos_x=np.append(surface1_pos_x, (ix+0.5)*self.sizeBin_x)
                    surface1_pos_y=np.append(surface1_pos_y, (iy+0.5)*self.sizeBin_y)

        # Dictionary containing x and y positions and ids of bins belonging to exterior (1) or interior surface (2)
        surface_dict = {'surface 1': {'pos': {'x': surface1_pos_x, 'y': surface1_pos_y}, 'id': {'bin': surface1_id, 'part': surface1_phaseInt}}, 'surface 2': {'pos': {'x': surface2_pos_x, 'y': surface2_pos_y}, 'id': {'bin': surface2_id, 'part': surface2_phaseInt}}}

        return surface_dict

    def surface_sort(self, surface_pos_x, surface_pos_y):
        '''
        Purpose: Takes the interior or exterior surface point positions and organizes
        the surface points in order of adjacency from the first surface ID such that
        the curve can be accurately plotted

        Inputs:
        surface_pos_x: array of x-positions of bins belonging to respective interface surface

        surface_pos_y: array of y-positions of bins belonging to respective interface surface

        Outputs:
        surface_pos_sort_dict: dictionary containing x and y positions bins belonging to exterior (1) or interior surface (2)
        '''

        #Save positions of external and internal edges
        clust_true = 0

        # Instantiate empty arrays for sorted x- and y- positions of surface points
        surface_pos_x_sorted=np.array([])
        surface_pos_y_sorted=np.array([])

        # Loop over all surface points until all sorted
        while len(surface_pos_x)>0:

            # If no surface points have been sorted, add the first point
            if len(surface_pos_x_sorted)==0:

                # Save sorted surface positions
                surface_pos_x_sorted = np.append(surface_pos_x_sorted, surface_pos_x[0])
                surface_pos_y_sorted = np.append(surface_pos_y_sorted, surface_pos_y[0])

                # Remove already sorted surface positions
                surface_pos_x = np.delete(surface_pos_x, 0)
                surface_pos_y = np.delete(surface_pos_y, 0)

            else:

                # Set initial shortest length measured to unrealistically high number that will be immediately replaced by second surface bin
                shortest_length = 100000
                for i in range(0, len(surface_pos_y)):

                    # Distance of reference surface bin from previously sorted surface bin
                    difx = self.utility_functs.sep_dist_x(surface_pos_x_sorted[-1], surface_pos_x[i])
                    dify = self.utility_functs.sep_dist_y(surface_pos_y_sorted[-1], surface_pos_y[i])

                    difr = (difx**2 + dify**2)**0.5

                    # If separation distance shorter than all previous bins looked at, then replace it as currently nearest neighbor
                    if difr < shortest_length:
                        shortest_length = difr
                        shortest_xlength = difx
                        shortest_ylength = dify
                        shortest_id = i

                    # If separation distance equal to previous bins looked at, then choose the bin that is to the left or below as nearest neighbor
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
                # Save identified nearest neighbor bin to sorted surface points
                surface_pos_x_sorted = np.append(surface_pos_x_sorted, surface_pos_x[shortest_id])
                surface_pos_y_sorted = np.append(surface_pos_y_sorted, surface_pos_y[shortest_id])

                # Remove identified nearest neighbor bin to from unsorted surface points
                surface_pos_x = np.delete(surface_pos_x, shortest_id)
                surface_pos_y = np.delete(surface_pos_y, shortest_id)

        # Dictionary containing the x- and y-positions of the sorted surface bin points.
        surface_pos_sort_dict = {'x': surface_pos_x_sorted, 'y': surface_pos_y_sorted}

        return surface_pos_sort_dict

    def surface_com(self, int_dict, int_comp_dict, surface_dict):
        '''
        Purpose: Takes the interior and exterior surface point positions and identifies
        the center of mass of those surface points

        Inputs:
        int_dict: dictionary of arrays labeling interface id of each bin and particle

        int_comp_dict: dictionary of arrays containing composition information of each interface

        surface_dict: dictionary containing x and y positions and ids of bins belonging to exterior (1) or interior surface (2)

        Outputs:
        surface_com_dict: dictionary containing x and y positions of surface's center of mass
        '''

        # Array (NBins_x, NBins_y) identifying ids of which interface that bin is a part of
        int_id = int_dict['bin']

        # ID (int) of largest interface per number of particles
        int_large_ids = int_comp_dict['ids']

        # Arrays (NBins_x, NBins_y) that labels whether bin is part of exterior surface (1) or not (0)
        surface1_id = surface_dict['surface 1']['id']['bin']

        # Arrays (NBins_x, NBins_y) that labels whether bin is part of interior surface (1) or not (0)
        surface2_id = surface_dict['surface 2']['id']['bin']

        # Arrays (NBins_x, NBins_y) that lists exterior surface bin x-positions
        surface1_pos_x = surface_dict['surface 1']['pos']['x']

        # Arrays (NBins_x, NBins_y) that lists interior surface bin x-positions
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

            # Loop over all bins
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):

                    # If interface is largest interface...
                    if (int_id[ix][iy]==int_large_ids[0]):

                        # If reference bin is at system's periodic boundary, label wrapping is needed for CoM calculation
                        if (ix == 0):
                            hor_wrap_left = 1
                        elif (ix == self.NBins_x-1):
                            hor_wrap_right = 1
                        if (iy == 0):
                            vert_wrap_bot = 1
                        elif (iy == self.NBins_y-1):
                            vert_wrap_top = 1

            #Sum positions of external edges of interface

            # Loop over all bins
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):

                    # If interior surface bins identified, prioritize those for CoM calculation
                    if len(surface2_pos_x)>0:

                        # If reference bin is part of interior surface of largest interface...
                        if (int_id[ix][iy]==int_large_ids[0]) & (surface2_id[ix][iy]==1):

                            # Identify mid-point position of reference bin and wrap across boundary if needed
                            x_box_pos_temp = (ix+0.5)*self.sizeBin_x
                            if (hor_wrap_right==1) & (hor_wrap_left==1) & (x_box_pos_temp<self.hx_box):
                                x_box_pos_temp += self.hx_box

                            y_box_pos_temp = (iy+0.5)*self.sizeBin_y
                            if (vert_wrap_bot==1) & (vert_wrap_top==1) & (y_box_pos_temp<self.hy_box):
                                y_box_pos_temp += self.hy_box

                            # Sum wrapped position and number of surface bins for averaging later
                            x_box_pos += x_box_pos_temp
                            y_box_pos += y_box_pos_temp
                            surface_num +=1

                    # Otherwise, if reference bin is part of exterior surface of largest interface...
                    elif len(surface1_pos_x)>0:

                        # If reference bin is part of exterior surface of largest interface...
                        if (int_id[ix][iy]==int_large_ids[0]) & (surface1_id[ix][iy]==1):

                            # Identify mid-point position of reference bin and wrap across boundary if needed
                            x_box_pos_temp = (ix+0.5)*self.sizeBin_x
                            if (hor_wrap_right==1) & (hor_wrap_left==1) & (x_box_pos_temp<self.hx_box):
                                x_box_pos_temp += self.hx_box

                            y_box_pos_temp = (iy+0.5)*self.sizeBin_y
                            if (vert_wrap_bot==1) & (vert_wrap_top==1) & (y_box_pos_temp<self.hy_box):
                                y_box_pos_temp += self.hy_box

                            # Sum wrapped position and number of surface bins for averaging later
                            x_box_pos += x_box_pos_temp
                            y_box_pos += y_box_pos_temp
                            surface_num +=1

            #Determine mean location (CoM) of external edges of interface
            if surface_num>0:

                # Average CoM location
                box_com_x = x_box_pos/surface_num
                box_com_y = y_box_pos/surface_num

                # If CoM outside of simulation box, shift to simulation box using periodic boundary conditions
                box_com_x_abs = np.abs(box_com_x)
                if box_com_x_abs>=self.lx_box:
                    if box_com_x < -self.hx_box:
                        box_com_x += self.lx_box
                    else:
                        box_com_x -= self.lx_box

                box_com_y_abs = np.abs(box_com_y)
                if box_com_y_abs>=self.ly_box:
                    if box_com_y < -self.hy_box:
                        box_com_y += self.ly_box
                    else:
                        box_com_y -= self.ly_box
            else:
                box_com_x=0
                box_com_y=0

        # Dictionary containing the x- and y-positions of the CoM of the largest interface
        surface_com_dict = {'x': box_com_x, 'y': box_com_y}

        return surface_com_dict

    def surface_radius_bins(self, int_dict, int_comp_dict, surface_dict, surface_com_dict):
        '''
        Purpose: Takes the interior and exterior surface point positions and identifies
        the center of mass of those surface points

        Inputs:
        int_dict: dictionary of arrays labeling interface id of each bin and particle

        int_comp_dict: dictionary of arrays containing composition information of each interface

        surface_dict: dictionary containing x and y positions and ids of bins belonging to exterior (1) or interior surface (2)

        surface_com_dict: dictionary containing x and y positions of surface's center of mass

        Outputs:
        radius_dict: dictionary containing arrays of all distances and angle sof surface points from
        largest interface's CoM in addition to the mean radius and angle
        '''

        # Array (NBins_x, NBins_y) identifying ids of which interface that bin is a part of
        int_id = int_dict['bin']

        # ID (int) of largest interface per number of particles
        int_large_ids = int_comp_dict['ids']

        # Arrays (NBins_x, NBins_y) that labels whether bin is part of exterior surface (1) or not (0)
        surface1_id = surface_dict['surface 1']['id']['bin']

        # Arrays (NBins_x, NBins_y) that labels whether bin is part of interior surface (1) or not (0)
        surface2_id = surface_dict['surface 2']['id']['bin']

        # Arrays (NBins_x, NBins_y) that lists exterior surface bin x-positions
        surface1_pos_x = surface_dict['surface 1']['pos']['x']

        # Arrays (NBins_x, NBins_y) that lists interior surface bin x-positions
        surface2_pos_x = surface_dict['surface 2']['pos']['x']

        # X-position of largest interface's CoM
        box_com_x = surface_com_dict['x']

        # y-position of largest interface's CoM
        box_com_y = surface_com_dict['y']

        # If there is an identified interface...
        if len(int_large_ids) > 0:

            #Initialize empty arrays for angle and distance from CoM of surface points and respective IDs
            thetas = np.array([])
            radii = np.array([])

            x_id = np.array([], dtype=int)
            y_id = np.array([], dtype=int)

            #Calculate distance from CoM to external edge bin and angle from CoM

            # Loop over all bins
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):

                    # If bin is interface and external edge, continue...
                    if (int_id[ix][iy]==int_large_ids[0]) & (surface2_id[ix][iy]==1):

                        #Reference bin location
                        x_box_pos = (ix+0.5)*self.sizeBin_x
                        y_box_pos = (iy+0.5)*self.sizeBin_y

                        #Calculate x-distance from CoM
                        difx=x_box_pos-box_com_x

                        #Enforce periodic boundary conditions
                        difx_abs = np.abs(difx)
                        if difx_abs>=self.hx_box:
                            if difx < -self.hx_box:
                                difx += self.lx_box
                            else:
                                difx -= self.lx_box

                        #Calculate y-distance from CoM
                        dify=y_box_pos-box_com_y

                        #Enforce periodic boundary conditions
                        dify_abs = np.abs(dify)
                        if dify_abs>=self.hy_box:
                            if dify < -self.hy_box:
                                dify += self.ly_box
                            else:
                                dify -= self.ly_box

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

        # Dictionary containing an array of all distances/thetas and the mean distance/theta from the CoM for the largest interface
        radius_dict = {'radius': {'vals': radii, 'mean': np.mean(radii)}, 'theta': {'vals': thetas, 'mean': np.mean(thetas)}}

        return radius_dict

    def separate_surfaces(self, surface_dict, int_dict, int_comp_dict):
        '''
        Purpose: Takes the location and ids of each surface point (either interior or
        exterior) for each interface and identifies which is the interior surface and
        which is the exterior surface

        Inputs:
        surface_dict: dictionary containing x and y positions and ids of bins belonging to exterior (1) or interior surface (2)

        int_dict: dictionary of arrays labeling interface id of each bin and particle

        int_comp_dict: dictionary of arrays containing composition information of each interface

        Outputs:
        sep_surface_dict: dictionary that stores interior and exterior surface information for each respective interface of sufficient size
        '''

        # Arrays that labels whether bin is part of exterior surface (1) or not (0)
        surface1_id = surface_dict['surface 1']['id']['bin']

        # Arrays that lists exterior surface bin x-positions
        surface1_x = surface_dict['surface 1']['pos']['x']

        # Arrays that lists exterior surface bin y-positions
        surface1_y = surface_dict['surface 1']['pos']['y']

        # Arrays that labels whether bin is part of interior surface (1) or not (0)
        surface2_id = surface_dict['surface 2']['id']['bin']

        # Arrays that lists interior surface bin x-positions
        surface2_x = surface_dict['surface 2']['pos']['x']

        # Arrays that lists interior surface bin y-positions
        surface2_y = surface_dict['surface 2']['pos']['y']

        # Array (NBins_x, NBins_y) identifying ids of which interface that bin is a part of
        int_id = int_dict['bin']

        # ID (int) of largest interface per number of particles
        int_large_ids = int_comp_dict['ids']

        

        # Dictionary that stores interior and exterior surface information for each respective interface of sufficient size
        sep_surface_dict = {}        

        # Loop over all interfaces
        for m in range(0, len(int_large_ids)):

            # Instantiate empty array that contains x- and y- indices for interior surface
            int_surface_x = np.array([], dtype=int)
            int_surface_y = np.array([], dtype=int)

            # Instantiate empty array that contains x- and y- indices for exterior surface
            ext_surface_x = np.array([], dtype=int)
            ext_surface_y = np.array([], dtype=int)

            # Instantiate empty array that contains x- and y- indices for either (1st or 2nd) interface surface
            surface1_x_id = np.array([], dtype=int)
            surface1_y_id = np.array([], dtype=int)

            surface2_x_id = np.array([], dtype=int)
            surface2_y_id = np.array([], dtype=int)

            # Start count of number of bins belonging to either interface surface (1 or 2)
            surface1_num = 0
            surface2_num = 0

            # Instantiate empty array (NBins_x, NBins_y) that labels whether bin is interior surface bin (1) or not (0)
            int_surface_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)

            # Instantiate empty array (NBins_x, NBins_y) that labels whether bin is exterior surface bin (1) or not (0)
            ext_surface_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)
        

            # If interface sufficiently large...
            if int_large_ids[m]!=999:

                # Loop over all bins
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):

                        # If bin belongs to reference interface
                        if int_id[ix][iy]==int_large_ids[m]:

                            # Save indices of current bin if part of either interface's surfaces
                            if surface1_id[ix][iy]==1:
                                surface1_num +=1
                                surface1_x_id = np.append(surface1_x_id, ix)
                                surface1_y_id = np.append(surface1_y_id, iy)
                            if surface2_id[ix][iy]==1:
                                surface2_num +=1
                                surface2_x_id = np.append(surface2_x_id, ix)
                                surface2_y_id = np.append(surface2_y_id, iy)

                # If two surfaces of reference interface are defined (one interior and one exterior)
                # then find which is the interior and which is the exterior surface
                if (surface1_num > 0) & (surface2_num > 0):

                    # If surface 1 has more bins than surface 2, surface 1 is the exterior and surface 2 is the interior
                    if surface1_num>surface2_num:
                        for v in range(0, len(surface1_x_id)):
                            ext_surface_x = np.append(ext_surface_x, surface1_x_id[v])
                            ext_surface_y = np.append(ext_surface_y, surface1_y_id[v])
                        for v in range(0, len(surface2_x_id)):
                            int_surface_x = np.append(int_surface_x, surface2_x_id[v])
                            int_surface_y = np.append(int_surface_y, surface2_y_id[v])

                    # If surface 2 has more bins than surface 1, surface 2 is the exterior and surface 1 is the interior
                    else:
                        for v in range(0, len(surface2_x_id)):
                            ext_surface_x = np.append(ext_surface_x, surface2_x_id[v])
                            ext_surface_y = np.append(ext_surface_y, surface2_y_id[v])
                        for v in range(0, len(surface1_x_id)):
                            int_surface_x = np.append(int_surface_x, surface1_x_id[v])
                            int_surface_y = np.append(int_surface_y, surface1_y_id[v])

                # If only one surface of reference interface is defined (one interior or one exterior)
                # then label the only surface as an exterior surface
                elif (surface1_num > 0) & (surface2_num == 0):
                    for v in range(0, len(surface1_x_id)):
                        ext_surface_x = np.append(ext_surface_x, surface1_x_id[v])
                        ext_surface_y = np.append(ext_surface_y, surface1_y_id[v])
                elif (surface1_num == 0) & (surface2_num > 0):
                    for v in range(0, len(surface2_x_id)):
                        ext_surface_x = np.append(ext_surface_x, surface2_x_id[v])
                        ext_surface_y = np.append(ext_surface_y, surface2_y_id[v])

                # Number of bins belonging to interior and exterior surfaces
                int_surface_num = len(int_surface_x)
                ext_surface_num = len(ext_surface_x)

                # If interior surface defined, label each bin as belonging to exterior/interior surface (1) or not (0)
                if int_surface_num > 0:
                    for ix in range(0, len(int_surface_x)):
                        int_surface_id[int_surface_x[ix]][int_surface_y[ix]]=1
                        ext_surface_id[int_surface_x[ix]][int_surface_y[ix]]=0
                if ext_surface_num >0:
                    for ix in range(0, len(ext_surface_x)):
                        int_surface_id[ext_surface_x[ix]][ext_surface_y[ix]]=0
                        ext_surface_id[ext_surface_x[ix]][ext_surface_y[ix]]=1

                # Find x- and y- positions of each interior and exterior surface point
                int_surface_pos_x = int_surface_x * self.sizeBin_x
                int_surface_pos_y = int_surface_y * self.sizeBin_y

                ext_surface_pos_x = ext_surface_x * self.sizeBin_x
                ext_surface_pos_y = ext_surface_y * self.sizeBin_y
                
                # Dictionary containing information on exterior and interior surfaces for reference interface
                indiv_surface_dict = {'interior': {'x bin': int_surface_x, 'y bin': int_surface_y, 'ids': int_surface_id, 'num': int_surface_num}, 'exterior': {'x bin': ext_surface_x, 'y bin': ext_surface_y, 'ids': ext_surface_id, 'num': ext_surface_num}}

                # Dictionary key labeling reference interface
                key_temp = 'surface id ' + str(int(int_large_ids[m]))
                
                # Reference interface surface dictionary saved to dictionary that contains exterior and interior surface information for all interfaces
                sep_surface_dict[key_temp] = indiv_surface_dict
        return sep_surface_dict

    def sort_surface_points(self, surface_dict):
        '''
        Purpose: Takes the location and ids of each surface point (either interior or
        exterior) for a given interface and identifies which sorts the points in terms of
        adjacency such that a curve can be plotted

        Inputs:
        surface_dict: dictionary containing x and y positions and ids of bin for interior or
        exterior surface depending on input dictionary, i.e. sep_surface_dict['surface id 1']['interior']
        or sep_surface_dict['surface id 1']['exterior']

        Outputs:
        sep_surface_dict: dictionary that stores interior and exterior surface information for each respective interface of sufficient size
        '''

        # Arrays that lists surface bin x- and y-positions
        surface_x = surface_dict['x bin']
        surface_y = surface_dict['y bin']

        # Array (NBins_x, NBins_y) that identifies whether bin is of the given interface surface (1) or not (0)
        surface_id = surface_dict['ids']

        # Instantiate array with starting surface point to sort from in terms of adjacency
        surface_x_sort = np.array([surface_x[0]])
        surface_y_sort = np.array([surface_y[0]])

        # Reference surface point IDs
        ix=int(surface_x_sort[0])
        iy=int(surface_y_sort[0])

        # Nearest neighboring bins IDs
        shortest_idx = np.array([])
        shortest_idy = np.array([])

        # Unsorted x- and y- IDs of given surface
        surface_x = np.delete(surface_x, 0)
        surface_y = np.delete(surface_y, 0)

        #
        fail=0

        #Determine if first interior surface bin of interface has at least 1
        #neighbor that is an interior surface bin of interface
        if len(surface_x)>0:

            # Identify x ID of bin to right
            if ix < (self.NBins_x-1):
                right = int(ix+1)
            else:
                right= int(0)

            # Identify x ID of bin to left
            if ix > 0:
                left = int(ix-1)
            else:
                left=int(self.NBins_x-1)

            # Identify y ID of bin above
            if iy < (self.NBins_y-1):
                up = int(iy+1)
            else:
                up= int(0)

            # Identify y ID of bin below
            if iy > 0:
                down = int(iy-1)
            else:
                down= int(self.NBins_y-1)

            # If bin to right of reference bin is part of the same surface, save it to sorted array
            if surface_id[right][iy]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, iy)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == right) & (surface_y == iy))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin above reference bin is part of the same surface, save it to sorted array
            elif surface_id[ix][up]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, ix)
                surface_y_sort = np.append(surface_y_sort, up)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == ix) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin below reference bin is part of the same surface, save it to sorted array
            elif surface_id[ix][down]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, ix)
                surface_y_sort = np.append(surface_y_sort, down)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == ix) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin left of reference bin is part of the same surface, save it to sorted array
            elif surface_id[left][iy]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, iy)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == left) & (surface_y == iy))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin upper left of reference bin is part of the same surface, save it to sorted array
            elif surface_id[left][up]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, up)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == left) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin lower left of reference bin is part of the same surface, save it to sorted array
            elif surface_id[left][down]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, left)
                surface_y_sort = np.append(surface_y_sort, down)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == left) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surfacer_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin upper right of reference bin is part of the same surface, save it to sorted array
            elif surface_id[right][up]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, up)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == right) & (surface_y == up))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # Otherwise, if bin lower right of reference bin is part of the same surface, save it to sorted array
            elif surface_id[right][down]==1:

                # Save neighbor bin to sorted array of surface IDs
                surface_x_sort = np.append(surface_x_sort, right)
                surface_y_sort = np.append(surface_y_sort, down)

                # Remove neighbor bin from unsorted array of surface IDs
                loc_id = np.where((surface_x == right) & (surface_y == down))[0]

                surface_x = np.delete(surface_x, loc_id)
                surface_y = np.delete(surface_y, loc_id)

            # If none are of the same surface, fail the surface sorting mechanism
            else:
                fail=1

            #If found at least 1 surface bin neighbor
            if fail==0:

                # Reference surface bin ID
                ix_ref=surface_x_sort[1]
                iy_ref=surface_y_sort[1]

                # Previous number of unsorted surface IDs
                past_size=0

                # While unsorted surface IDs, sort surface IDs
                while len(surface_x)>0:

                    # Current number onsorted surface IDs
                    current_size = len(surface_x)

                    # If no surface ID was sorted in previous loop...
                    if past_size == current_size:

                        # Shortest length to nearest, neighboring surface bin
                        shortest_length = 100000.

                        # Loop over all bins
                        for ix in range(0, self.NBins_x):
                            for iy in range(0, self.NBins_y):

                                # If not the same bin...
                                if (ix!=ix_ref) | (iy!=iy_ref):

                                    # If bin is of the reference surface
                                    if surface_id[ix][iy]==1:

                                        # Find unsorted surface bin location
                                        loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                        if len(loc_id)>0:

                                            # Find separation distance from reference bin
                                            difx = self.utility_functs.sep_dist_x((ix_ref+0.5)*self.sizeBin_x, (ix+0.5)*self.sizeBin_x)
                                            dify = self.utility_functs.sep_dist_y((iy_ref+0.5)*self.sizeBin_y, (iy+0.5)*self.sizeBin_y)

                                            difr = (difx**2 + dify**2)**0.5

                                            # If separation distance from reference bin shorter than all prior surface bins, save it
                                            if difr < shortest_length:
                                                shortest_length = difr
                                                shortest_idx = np.array([ix])
                                                shortest_idy = np.array([iy])

                                            # If separation distance from reference bin same as at least one prior surface bins, save both
                                            elif difr == shortest_length:
                                                shortest_idx = np.append(shortest_idx, ix)
                                                shortest_idy = np.append(shortest_idy, iy)

                        # If shortest length is too far, break the sorting loop
                        if shortest_length > (self.hx_box+self.hy_box)/20:
                            break

                        # If multiple surface bins of same distance from reference bin found, find which to prioritize
                        if len(shortest_idx) > 1:
                            num_neigh = np.zeros(len(shortest_idx))

                            # Loop over nearest surface bins
                            for ind in range(0, len(shortest_idx)):

                                # Nearest surface bin IDs
                                ix_ind = shortest_idx[ind]
                                iy_ind = shortest_idy[ind]

                                # Nearest shell of neighboring x-bin IDs to loop over
                                if (ix_ind + 1) == self.NBins_x:
                                    lookx = [ix_ind-1, ix_ind, 0]
                                elif ix_ind==0:
                                    lookx=[self.NBins_x-1, ix_ind, ix_ind+1]
                                else:
                                    lookx = [ix_ind-1, ix_ind, ix_ind+1]

                                # Nearest shell of neighboring y-bin IDs to loop over
                                if (iy_ind + 1) == self.NBins_y:
                                    looky = [iy_ind-1, iy_ind, 0]
                                elif iy_ind==0:
                                    looky=[self.NBins_y-1, iy_ind, iy_ind+1]
                                else:
                                    looky = [iy_ind-1, iy_ind, iy_ind+1]

                                # Loop over neighboring bins
                                for ix in lookx:
                                    for iy in looky:

                                        # If neighboring bin is different than nearest surface bin...
                                        if (ix != ix_ind) | (iy != iy_ind):
                                            loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]
                                            if len(loc_id)>0:

                                                # If neighboring bin is of the reference surface, count it
                                                if surface_id[ix][iy]==1:
                                                    num_neigh[ind]+=1

                            # Find nearest surface bin with fewest number of neighboring surface bins
                            min_inds = np.min(num_neigh)
                            loc_min_inds = np.where(num_neigh == min_inds)[0]

                            # Prioritize sorting nearest surface bin with fewest surface bin neighbors

                            # If only one nearest surface bin with fewest surface bin neighbors...
                            if len(loc_min_inds)==1:

                                # Set nearest surface bin IDs with fewest surface bin neighbors to new reference bin
                                ix_ref = shortest_idx[loc_min_inds][0]
                                iy_ref = shortest_idy[loc_min_inds][0]

                                # Save nearest surface bin with fewest surface bin neighbors to sorted list
                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                # Remove nearest surface bin with fewest surface bin neighbors from unsorted list
                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)

                            # If multiple nearest surface bin with fewest surface bin neighbors...
                            else:

                                # Set first identified nearest surface bin IDs with fewest surface bin neighbors to new reference bin
                                ix_ref = shortest_idx[np.min(loc_min_inds)]
                                iy_ref = shortest_idy[np.min(loc_min_inds)]

                                # Save nearest surface bin with fewest surface bin neighbors to sorted list
                                surface_x_sort = np.append(surface_x_sort, ix_ref)
                                surface_y_sort = np.append(surface_y_sort, iy_ref)

                                # Remove nearest surface bin with fewest surface bin neighbors from unsorted list
                                loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                                surface_x = np.delete(surface_x, loc_id)
                                surface_y = np.delete(surface_y, loc_id)

                        # If one surface bin of same distance from reference bin found, sort it
                        elif len(shortest_idx)==1:

                            # Set nearest surface bin to new reference bin
                            ix_ref = shortest_idx[0]
                            iy_ref = shortest_idy[0]

                            # Save nearest surface bin to sorted list
                            surface_x_sort = np.append(surface_x_sort, ix_ref)
                            surface_y_sort = np.append(surface_y_sort, iy_ref)

                            # Remove nearest surface bin from unsorted list
                            loc_id = np.where((surface_x == ix_ref) & (surface_y == iy_ref))[0]

                            surface_x = np.delete(surface_x, loc_id)
                            surface_y = np.delete(surface_y, loc_id)

                        # If no surface bins found, break the loop
                        else:
                            break

                    # If reference bin is of the reference surface...
                    if surface_id[ix_ref][iy_ref]==1:

                        # Nearest shell of neighboring x-bin IDs to loop over
                        if (ix_ref + 1) == self.NBins_x:
                            lookx = [ix_ref-1, ix_ref, 0]
                        elif ix_ref==0:
                            lookx=[self.NBins_x-1, ix_ref, ix_ref+1]
                        else:
                            lookx = [ix_ref-1, ix_ref, ix_ref+1]

                        # Nearest shell of neighboring y-bin IDs to loop over
                        if (iy_ref + 1) == self.NBins_y:
                            looky = [iy_ref-1, iy_ref, 0]
                        elif iy_ref==0:
                            looky=[self.NBins_y-1, iy_ref, iy_ref+1]
                        else:
                            looky = [iy_ref-1, iy_ref, iy_ref+1]

                        # Set nearest neighboring surface bin distance to unrealistically large value
                        shortest_length = 100000.

                        # Loop over neighboring bins
                        for ix in lookx:
                            for iy in looky:

                                # If neighboring bin is not reference bin...
                                if (ix!=ix_ref) | (iy!=iy_ref):

                                    # ID of surface bin in unsorted surface bin array
                                    loc_id = np.where((surface_x == ix) & (surface_y == iy))[0]

                                    # If neighboring bin a member of unsorted surface bin array...
                                    if len(loc_id)>0:

                                        # If neighboring bin is a member of reference surface...
                                        if surface_id[ix][iy]==1:

                                            # Find separation distance of neighboring bin from reference bin
                                            difx = self.utility_functs.sep_dist_x((ix+0.5)*self.sizeBin_x, (ix_ref+0.5)*self.sizeBin_x)
                                            dify = self.utility_functs.sep_dist_y((iy+0.5)*self.sizeBin_y, (iy_ref+0.5)*self.sizeBin_y)

                                            difr = (difx**2 + dify**2)**0.5

                                            # If separation distance less than all previous surface bins, save it
                                            if difr < shortest_length:
                                                shortest_length = difr
                                                shortest_idx = np.array([ix])
                                                shortest_idy = np.array([iy])

                                            # If separation distance is equal to at least one previous surface bin, save both
                                            elif difr == shortest_length:
                                                shortest_idx = np.append(shortest_idx, ix)
                                                shortest_idy = np.append(shortest_idy, iy)

                        # If more than one neighboring surface bin nearest to reference bin...
                        if len(shortest_idx) > 1:

                            # Instantiate empty array to count number of surface bins neighboring each nearest surface bin
                            num_neigh = np.zeros(len(shortest_idx))

                            # Loop over neighboring bin nearest to reference bin
                            for ind in range(0, len(shortest_idx)):
                                ix_ind = shortest_idx[ind]
                                iy_ind = shortest_idy[ind]

                                # Nearest shell of neighboring x-bin IDs to loop over
                                if (ix_ind + 1) == self.NBins_x:
                                    lookx = [ix_ind-1, ix_ind, 0]
                                elif ix_ind==0:
                                    lookx=[self.NBins_x-1, ix_ind, ix_ind+1]
                                else:
                                    lookx = [ix_ind-1, ix_ind, ix_ind+1]

                                # Nearest shell of neighboring y-bin IDs to loop over
                                if (iy_ind + 1) == self.NBins_y:
                                    looky = [iy_ind-1, iy_ind, 0]
                                elif iy_ind==0:
                                    looky=[self.NBins_y-1, iy_ind, iy_ind+1]
                                else:
                                    looky = [iy_ind-1, iy_ind, iy_ind+1]

                                # Loop over neighboring bins
                                for ix in lookx:
                                    for iy in looky:

                                        # If
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
    def surface_curve_prep(self, sort_surface_ids, int_type='None'):
        shift_sort_surface_ids={'x': sort_surface_ids['x'], 'y': sort_surface_ids['y']}
        surface_curve_pos_dict_temp={'x': sort_surface_ids['x'] * self.sizeBin_x, 'y': sort_surface_ids['y'] * self.sizeBin_y}

        com_pov_int_surface = self.surface_com_pov(surface_curve_pos_dict_temp)

        if int_type == 'interior':
            for m in range(0, len(sort_surface_ids['x'])):

                if sort_surface_ids['x'][m] * self.sizeBin_x < com_pov_int_surface['com']['x']:

                    shift_sort_surface_ids['x'][m] = sort_surface_ids['x'][m] + 1.
                if sort_surface_ids['y'][m] * self.sizeBin_y < com_pov_int_surface['com']['y']:
                        
                    shift_sort_surface_ids['y'][m] =  sort_surface_ids['y'][m] + 1.
        elif int_type == 'exterior':
            for m in range(0, len(sort_surface_ids['x'])):

                if sort_surface_ids['x'][m] * self.sizeBin_x > com_pov_int_surface['com']['x']:

                    shift_sort_surface_ids['x'][m] = sort_surface_ids['x'][m] + 1.
                
                if sort_surface_ids['y'][m] * self.sizeBin_y > com_pov_int_surface['com']['y']:
                        
                    shift_sort_surface_ids['y'][m] =  sort_surface_ids['y'][m] + 1.

        return shift_sort_surface_ids

    def surface_curve_interp(self, sort_surface_ids):

        surface_x_sort = sort_surface_ids['x']
        surface_y_sort = sort_surface_ids['y']

        surface_x_sort_pos = surface_x_sort * self.sizeBin_x
        surface_y_sort_pos = surface_y_sort * self.sizeBin_y

        pos_arr = {'x': surface_x_sort_pos, 'y': surface_y_sort_pos}
        com_pov_int_surface = self.surface_com_pov(pos_arr)
        shift_x_pos = self.hx_box - com_pov_int_surface['com']['x']
        shift_y_pos = self.hy_box - com_pov_int_surface['com']['y']


        surface_x_sort_pos = surface_x_sort_pos + shift_x_pos
        surface_y_sort_pos = surface_y_sort_pos + shift_y_pos

        periodic = np.where(surface_x_sort_pos > self.lx_box)[0]
        surface_x_sort_pos[periodic] = surface_x_sort_pos[periodic] - self.lx_box
        periodic = np.where(surface_x_sort_pos < 0)[0]
        surface_x_sort_pos[periodic] = surface_x_sort_pos[periodic] + self.lx_box

        periodic = np.where(surface_y_sort_pos > self.ly_box)[0]
        surface_y_sort_pos[periodic] = surface_y_sort_pos[periodic] - self.ly_box
        periodic = np.where(surface_y_sort_pos < 0)[0]
        surface_y_sort_pos[periodic] = surface_y_sort_pos[periodic] + self.ly_box

        shift_x_id = round( (self.hx_box - com_pov_int_surface['com']['x']) / self.sizeBin_x)
        shift_y_id = round( (self.hy_box - com_pov_int_surface['com']['y']) / self.sizeBin_y)
        surface_x_sort = surface_x_sort + shift_x_id
        surface_y_sort = surface_y_sort + shift_y_id

        periodic = np.where(surface_x_sort > self.NBins_x)[0]
        surface_x_sort[periodic] = surface_x_sort[periodic] - self.NBins_x
        periodic = np.where(surface_x_sort < 0)[0]
        surface_x_sort[periodic] = surface_x_sort[periodic] + self.NBins_x

        periodic = np.where(surface_y_sort > self.NBins_y)[0]
        surface_y_sort[periodic] = surface_y_sort[periodic] - self.NBins_y
        periodic = np.where(surface_y_sort < 0)[0]
        surface_y_sort[periodic] = surface_y_sort[periodic] + self.NBins_y

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
                    if difx_abs>=self.hx_box:
                        if difx < -self.hx_box:
                            surface_x_sort_pos[m:-1] += self.lx_box
                            surface_x_sort[m:-1] += self.NBins_x
                        else:
                            surface_x_sort_pos[m:-1] -= self.lx_box
                            surface_x_sort[m:-1] -= self.NBins_x

                    #Enforce periodic boundary conditions
                    if dify_abs>=self.hy_box:
                        if dify < -self.hy_box:
                            surface_y_sort_pos[m:-1] += self.ly_box
                            surface_y_sort[m:-1] += self.NBins_y
                        else:
                            surface_y_sort_pos[m:-1] -= self.ly_box
                            surface_y_sort[m:-1] -= self.NBins_y

                    if (difx_abs>=self.hx_box) or (dify_abs>=self.hy_box):
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
            
            
            #pos_arr = {'x': sort_surface_ids['x'] * self.sizeBin_x, 'y': sort_surface_ids['y'] * self.sizeBin_y}
            #print(pos_arr)
            #print(adjacent_x_discont)
            #stop
            #com_pov_int_surface = self.surface_com_pov(pos_arr)
            #plt.scatter(sort_surface_ids['x']*self.sizeBin_x, sort_surface_ids['y']*self.sizeBin_y, c='black')
            
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
                    xn_pos[m] = xn[m] * self.sizeBin_x
                    yn_pos[m] = yn[m] * self.sizeBin_y
                    xn_pos_non_per[m] = xn[m] * self.sizeBin_x
                    yn_pos_non_per[m] = yn[m] * self.sizeBin_y

                    if xn[m] < 0:
                        xn[m]+=self.NBins_x
                    if xn[m]>=self.NBins_x:
                        xn[m]-=self.NBins_x

                    if yn[m] < 0:
                        yn[m]+=self.NBins_y
                    if yn[m]>=self.NBins_y:
                        yn[m]-=self.NBins_y

                    if xn_pos[m] < 0:
                        xn_pos[m]+=self.lx_box
                    if xn_pos[m]>=self.lx_box:
                        xn_pos[m]-=self.lx_box

                    if yn_pos[m] < 0:
                        yn_pos[m]+=self.ly_box
                    if yn_pos[m]>=self.ly_box:
                        yn_pos[m]-=self.ly_box



            else:

                xn = np.zeros(1)
                yn = np.zeros(1)
                xn_pos = np.zeros(1)
                yn_pos = np.zeros(1)
                xn_pos_non_per = np.zeros(1)
                yn_pos_non_per = np.zeros(1)

                xn_pos[0] = surface_x_interp[0]
                yn_pos[0] = surface_y_interp[0]
                xn_pos[0] = surface_x_interp[0] * self.sizeBin_x
                yn_pos[0] = surface_y_interp[0] * self.sizeBin_y
                xn_pos_non_per[0] = surface_x_interp[0] * self.sizeBin_x
                yn_pos_non_per[0] = surface_y_interp[0] * self.sizeBin_y
                if xn[0] < 0:
                    xn[0]+=self.NBins_x
                if xn[0]>=self.NBins_x:
                    xn[0]-=self.NBins_x

                if yn[0] < 0:
                    yn[0]+=self.NBins_y
                if yn[0]>=self.NBins_y:
                    yn[0]-=self.NBins_y

                if xn_pos[0] < 0:
                    xn_pos[0]+=self.lx_box
                if xn_pos[0]>=self.lx_box:
                    xn_pos[0]-=self.lx_box

                if yn_pos[0] < 0:
                    yn_pos[0]+=self.ly_box
                if yn_pos[0]>=self.ly_box:
                    yn_pos[0]-=self.ly_box

        else:

            xn=np.array([surface_x_sort[0]])
            yn=np.array([surface_y_sort[0]])
            xn_pos = np.copy(xn)
            yn_pos = np.copy(yn)
            xn_pos_non_per = np.copy(xn)
            yn_pos_non_per = np.copy(yn)
            for m in range(0, len(xn)):
                xn_pos[m] = xn[m] * self.sizeBin_x
                yn_pos[m] = yn[m] * self.sizeBin_y
                xn_pos_non_per[m] = xn[m] * self.sizeBin_x
                yn_pos_non_per[m] = yn[m] * self.sizeBin_y

                if xn[m] < 0:
                    xn[m]+=self.NBins_x
                if xn[m]>=self.NBins_x:
                    xn[m]-=self.NBins_x

                if yn[m] < 0:
                    yn[m]+=self.NBins_y
                if yn[m]>=self.NBins_y:
                    yn[m]-=self.NBins_y

                if xn_pos[m] < 0:
                    xn_pos[m]+=self.lx_box
                if xn_pos[m]>=self.lx_box:
                    xn_pos[m]-=self.lx_box

                if yn_pos[m] < 0:
                    yn_pos[m]+=self.ly_box
                if yn_pos[m]>=self.ly_box:
                    yn_pos[m]-=self.ly_box
        
        xn = xn + shift_x_id
        yn = yn + shift_y_id

        xn_pos = xn_pos - shift_x_pos
        yn_pos = yn_pos - shift_y_pos

        periodic = np.where(xn_pos > self.lx_box)[0]
        xn_pos[periodic] = xn_pos[periodic] - self.lx_box
        periodic = np.where(xn_pos < 0)[0]
        xn_pos[periodic] = xn_pos[periodic] + self.lx_box

        periodic = np.where(yn_pos > self.ly_box)[0]
        yn_pos[periodic] = yn_pos[periodic] - self.ly_box
        periodic = np.where(yn_pos < 0)[0]
        yn_pos[periodic] = yn_pos[periodic] + self.ly_box

        xn = xn - shift_x_id
        yn = yn - shift_y_id

        periodic = np.where(xn > self.NBins_x)[0]
        xn[periodic] = xn[periodic] - self.NBins_x
        periodic = np.where(xn < 0)[0]
        xn[periodic] = xn[periodic] + self.NBins_x

        periodic = np.where(yn > self.NBins_y)[0]
        yn[periodic] = yn[periodic] - self.NBins_y
        periodic = np.where(yn < 0)[0]
        yn[periodic] = yn[periodic] + self.NBins_y

        surface_curve_dict = {'id': {'x': xn, 'y': yn}, 'pos': {'x': xn_pos, 'y': yn_pos}}
        
        return surface_curve_dict


    def surface_area(self, surface_curve_pos_dict):

        # X- and Y- positions of reference surface
        surface_curve_xpos = surface_curve_pos_dict['x']
        surface_curve_ypos = surface_curve_pos_dict['y']

        surface_area = 0
        for id2 in range(1, len(surface_curve_xpos)):

            #Calculate position of interior edge bin
            difx = self.utility_functs.sep_dist_x(surface_curve_xpos[id2-1], surface_curve_xpos[id2])
            dify = self.utility_functs.sep_dist_y(surface_curve_ypos[id2-1], surface_curve_ypos[id2])

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
            if (difx_abs>=self.hx_box) | (dify_abs>=self.hy_box):

                if (difx_abs>=self.hx_box):
                    if (difx > 0):
                        surface_curve_xpos[i:] -= self.lx_box
                    elif (difx < 0):
                        surface_curve_xpos[i:] += self.lx_box

                if (dify_abs>=self.hy_box):
                    if (dify > 0):
                        surface_curve_ypos[i:] -= self.ly_box

                    elif (dify < 0):
                        surface_curve_ypos[i:] += self.ly_box

        x_com = np.mean(surface_curve_xpos)
        if x_com > self.lx_box:
            x_com -=self.lx_box
            surface_curve_xpos -= self.lx_box
        elif x_com < 0:
            x_com += self.lx_box
            surface_curve_xpos += self.lx_box

        y_com = np.mean(surface_curve_ypos)
        if y_com > self.ly_box:
            y_com -=self.ly_box
            surface_curve_ypos -= self.ly_box
        elif y_com < 0:
            y_com += self.ly_box
            surface_curve_ypos += self.ly_box

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
            if (difx_abs>=self.hx_box) | (dify_abs>=self.hy_box):

                if (difx_abs>=self.hx_box):
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

                if (dify_abs>=self.hy_box):
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

            difx = self.utility_functs.sep_dist_x(surface_com_xpos[i], int_com_x)
            dify = self.utility_functs.sep_dist_y(surface_com_ypos[i], int_com_y)

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

        new_align = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_trad = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_trad_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg_trad = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg_trad_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_avg_trad_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_num = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_num0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_num1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg_dif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        part_align = np.zeros(self.partNum)
        part_difr = np.zeros(self.partNum)

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

            try:
                
                interior_radius = surface_measurements[key]['interior']['mean radius']
                interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
                interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']
                interior_int_com_x = interior_surface_com_pos_dict['com']['x']
                interior_int_com_y = interior_surface_com_pos_dict['com']['y']
                int_surface_id = sep_surface_dict[key]['interior']['ids']
                interior_exist = 1
            except:
                interior_radius = None
                interior_surface_com_xpos = None
                interior_surface_com_ypos = None
                interior_int_com_x = None
                interior_int_com_y = None
                int_surface_id = None
                interior_exist = 0

            try:
                exterior_radius = surface_measurements[key]['exterior']['mean radius']
                exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])
                exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
                exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']
                exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
                exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']
                ext_surface_id = sep_surface_dict[key]['exterior']['ids']
                exterior_exist = 1
            except:
                exterior_exist = 0
                exterior_radius = None
                exterior_surface_com_xpos = None
                exterior_surface_com_ypos = None
                exterior_int_com_x = None
                exterior_int_com_y = None
                ext_surface_id = None

            if (interior_exist == 1) | (exterior_exist == 1):

                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):

                        #If bin is part of mth interface structure, continue...
                        if int_id[ix][iy]==int_large_ids[m]:

                            if ext_surface_id[ix][iy]==0:

                                #Calculate position of exterior edge bin

                                xpos_ref = (ix+0.5)*self.sizeBin_x
                                ypos_ref = (iy+0.5)*self.sizeBin_y

                                difx_trad = self.utility_functs.sep_dist_x(xpos_ref, self.hx_box)

                                dify_trad = self.utility_functs.sep_dist_y(ypos_ref, self.hy_box)

                                difx_bub = self.utility_functs.sep_dist_x(xpos_ref, exterior_int_com_x)

                                dify_bub = self.utility_functs.sep_dist_y(ypos_ref, exterior_int_com_y)

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

                                    difx_width = self.utility_functs.sep_dist_x(xpos_ref, exterior_surface_com_xpos[id])

                                    dify_width = self.utility_functs.sep_dist_y(ypos_ref, exterior_surface_com_ypos[id])

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
                                        if interior_exist == 1:
                                            if (difr_short == difr_bub) | ( exterior_radius<=interior_radius):
                                                x_dot_p = (-x_norm_unitv * px)
                                                y_dot_p = (-y_norm_unitv * py)
                                                r_dot_p = x_dot_p + y_dot_p
                                            else:
                                                x_dot_p = (x_norm_unitv * px)
                                                y_dot_p = (y_norm_unitv * py)
                                                r_dot_p = x_dot_p + y_dot_p
                                        else:
                                            if (difr_short == difr_bub):
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


                            elif ext_surface_id[ix][iy]==1:


                                #Calculate position of exterior edge bin
                                xpos_ref = (ix+0.5)*self.sizeBin_x
                                ypos_ref = (iy+0.5)*self.sizeBin_y

                                difx_trad = self.utility_functs.sep_dist_x(xpos_ref, self.hx_box)

                                dify_trad = self.utility_functs.sep_dist_y(ypos_ref, self.hy_box)

                                difx_bub = self.utility_functs.sep_dist_x(xpos_ref, exterior_int_com_x)

                                dify_bub = self.utility_functs.sep_dist_y(ypos_ref, exterior_int_com_y)

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
                                if interior_exist == 1:
                                    for id in range(0, len(interior_surface_com_xpos)):

                                        #Calculate position of interior edge bin

                                        difx_width = self.utility_functs.sep_dist_x(xpos_ref, interior_surface_com_xpos[id])

                                        dify_width = self.utility_functs.sep_dist_y(ypos_ref, interior_surface_com_ypos[id])

                                        #Calculate distance from interior edge bin to exterior edge bin
                                        difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                        #If this distance is the shortest calculated thus far, replace the value with it
                                        if difr<difr_short:
                                            difr_short=difr
                                            difx_short = np.abs(difx_width)
                                            dify_short = np.abs(dify_width)
                                            x_norm_unitv = difx_width / difr
                                            y_norm_unitv = dify_width / difr
                                else:
                                    #Calculate position of interior edge bin

                                    difx_width = self.utility_functs.sep_dist_x(xpos_ref, exterior_int_com_x)

                                    dify_width = self.utility_functs.sep_dist_y(ypos_ref, exterior_int_com_x)

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

                                        if interior_exist == 1:
                                            if (difr_short == difr_bub) | ( exterior_radius>interior_radius):
                                                x_dot_p = (-x_norm_unitv * px)
                                                y_dot_p = (-y_norm_unitv * py)
                                                r_dot_p = x_dot_p + y_dot_p
                                            else:
                                                x_dot_p = (x_norm_unitv * px)
                                                y_dot_p = (y_norm_unitv * py)
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

        new_align = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_trad = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_trad_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_x0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_x1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y0 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        new_align_trad_y1 = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        new_align_avg_dif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        bulk_large_ids = bulk_comp_dict['ids']

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

            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if bulk_id[ix][iy] == bulk_large_ids[m]:
                        #If bin is part of mth interface structure, continue...
                        if new_align_num[ix][iy]==0:
                            #Calculate position of exterior edge bin
                            difr_short = 100000
                            xpos_ref = (ix+0.5)*self.sizeBin_x
                            ypos_ref = (iy+0.5)*self.sizeBin_y

                            difx_trad = self.utility_functs.sep_dist_x(xpos_ref, self.hx_box)

                            dify_trad = self.utility_functs.sep_dist_y(ypos_ref, self.hy_box)

                            difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                            difr_bub= ( (difx_trad )**2 + (dify_trad)**2)**0.5
                            difr_short= 1000000
                            x_norm_unitv = difx_trad / difr_trad
                            y_norm_unitv = dify_trad / difr_trad

                            x_norm_unitv_trad = (difx_trad) / difr_trad
                            y_norm_unitv_trad = (dify_trad) / difr_trad

                            for m in range(0, len(sep_surface_dict)):
                                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                                try:
                                    interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                                    interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
                                    interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']
                                    interior_int_com_x = interior_surface_com_pos_dict['com']['x']
                                    interior_int_com_y = interior_surface_com_pos_dict['com']['y']
                                    interior_exist = 1
                                except:
                                    interior_surface_com_xpos = None
                                    interior_surface_com_ypos = None
                                    interior_int_com_x = None
                                    interior_int_com_y = None
                                    interior_exist = 0

                                try:
                                    exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])
                                    exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
                                    exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']
                                    exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
                                    exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']
                                    exterior_exist = 1
                                except:
                                    exterior_exist = 0
                                    exterior_surface_com_xpos = None
                                    exterior_surface_com_ypos = None
                                    exterior_int_com_x = None
                                    exterior_int_com_y = None


                                if (exterior_exist == 1):
                                    difx_ext = self.utility_functs.sep_dist_x(xpos_ref, exterior_int_com_x)

                                    dify_ext = self.utility_functs.sep_dist_y(ypos_ref, exterior_int_com_y)

                                    difr_ext= ( (difx_ext )**2 + (dify_ext)**2)**0.5

                                    if difr_ext < difr_short:
                                        difr_short= ( (difx_ext )**2 + (dify_ext)**2)**0.5
                                        x_norm_unitv = difx_ext / difr_short
                                        y_norm_unitv = dify_ext / difr_short
                                        interior_bin_short = 0
                                        exterior_bin_short = 0

                                if (interior_exist == 1):
                                    difx_int = self.utility_functs.sep_dist_x(xpos_ref, interior_int_com_x)

                                    dify_int = self.utility_functs.sep_dist_y(ypos_ref, interior_int_com_y)

                                    difr_int= ( (difx_int )**2 + (dify_int)**2)**0.5
                                    if difr_int < difr_short:
                                        difr_short= ( (difx_int )**2 + (dify_int)**2)**0.5
                                        x_norm_unitv = difx_int / difr_short
                                        y_norm_unitv = dify_int / difr_short
                                        interior_bin_short = 0
                                        exterior_bin_short = 0



                                #Loop over bins of system
                                if (exterior_exist == 1):
                                    for id in range(0, len(exterior_surface_com_xpos)):
                                        #If bin is an interior edge bin for mth interface structure, continue...

                                        difx_width = self.utility_functs.sep_dist_x(xpos_ref, exterior_surface_com_xpos[id])

                                        dify_width = self.utility_functs.sep_dist_y(ypos_ref, exterior_surface_com_ypos[id])

                                        #Calculate distance from interior edge bin to exterior edge bin
                                        difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                        #If this distance is the shortest calculated thus far, replace the value with it
                                        if difr<difr_short:
                                            difr_short=difr
                                            x_norm_unitv = difx_width / difr
                                            y_norm_unitv = dify_width / difr
                                            interior_bin_short = 0
                                            exterior_bin_short = 1
                                if (interior_exist == 1):
                                    for id in range(0, len(interior_surface_com_xpos)):
                                        #If bin is an interior edge bin for mth interface structure, continue...

                                        #Calculate position of interior edge bin
                                        difx_width = self.utility_functs.sep_dist_x(xpos_ref, interior_surface_com_xpos[id])

                                        dify_width = self.utility_functs.sep_dist_y(ypos_ref, interior_surface_com_ypos[id])


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
                                    else:
                                        x_dot_p = (-x_norm_unitv * px)
                                        y_dot_p = (-y_norm_unitv * py)


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
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
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

        # Instantiate empty arrays for calculating average alignment

        # Total alignment with nearest surface normal for each particle type ('all', 'A', or 'B') per bin
        Surface_align = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Total x-dimension alignment with nearest surface normal in x-dimension for each particle type ('all', 'A', or 'B') per bin
        Surface_align_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_x_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_x_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Total y-dimension alignment with nearest surface normal in y-dimension for each particle type ('all', 'A', or 'B') per bin
        Surface_align_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_y_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        Surface_align_y_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Total alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Total x-dimension alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_x_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_x_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Total y-dimension alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_y_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        CoM_align_y_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Average alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align_avg = method1_align_dict['bin']['all']['mag']
        CoM_align_avg_A = method1_align_dict['bin']['A']['mag']
        CoM_align_avg_B = method1_align_dict['bin']['B']['mag']

        # Average x-dimension alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align_avg_x = method1_align_dict['bin']['all']['x']
        CoM_align_avg_x_A = method1_align_dict['bin']['A']['x']
        CoM_align_avg_x_B = method1_align_dict['bin']['B']['x']

        # Average y-dimension alignment with largest cluster's CoM for each particle type ('all', 'A', or 'B') per bin
        CoM_align_avg_y = method1_align_dict['bin']['all']['y']
        CoM_align_avg_y_A = method1_align_dict['bin']['A']['y']
        CoM_align_avg_y_B = method1_align_dict['bin']['B']['y']

        # Average alignment with nearest surface normal for each particle type ('all', 'A', or 'B') per bin
        Surface_align_avg = method2_align_dict['bin']['all']['mag']
        Surface_align_avg_A = method2_align_dict['bin']['A']['mag']
        Surface_align_avg_B = method2_align_dict['bin']['B']['mag']

        # Average x-dimension alignment with nearest surface normal in x-dimension for each particle type ('all', 'A', or 'B') per bin
        Surface_align_avg_x = method2_align_dict['bin']['all']['x']
        Surface_align_avg_x_A = method2_align_dict['bin']['A']['x']
        Surface_align_avg_x_B = method2_align_dict['bin']['B']['x']

        # Average y-dimension alignment with nearest surface normal in y-dimension for each particle type ('all', 'A', or 'B') per bin
        Surface_align_avg_y = method2_align_dict['bin']['all']['y']
        Surface_align_avg_y_A = method2_align_dict['bin']['A']['y']
        Surface_align_avg_y_B = method2_align_dict['bin']['B']['y']

        # Number of particles in each bin of respective particle type ('all', 'A', or 'B')
        Align_num = method1_align_dict['bin']['all']['num']
        Align_num_A = method1_align_dict['bin']['A']['num']
        Align_num_B = method1_align_dict['bin']['B']['num']

        # Alignment with nearest surface normal per particle
        part_align = method2_align_dict['part']['align']

        # Separation distance with nearest surface normal per particle
        part_difr = method2_align_dict['part']['difr']

        #Calculate alignment of gas bins

        #Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # If average alignment has not been calculated previously (gas)
                if Surface_align_avg[ix][iy]==0:

                    #Calculate reference bin position
                    xpos_ref = (ix+0.5)*self.sizeBin_x
                    ypos_ref = (iy+0.5)*self.sizeBin_y


                    difr_short= 100000

                    # If at least no surfaces have been defined...
                    if len(sep_surface_dict) >= 0:

                        # Separation distance of reference bin from middle of simulation box
                        difx_trad = self.utility_functs.sep_dist_x(xpos_ref, self.hx_box)
                        dify_trad = self.utility_functs.sep_dist_y(ypos_ref, self.hy_box)
                        difr_trad= ( (difx_trad )**2 + (dify_trad)**2)**0.5

                        #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
                        difr_bub= ( (difx_trad )**2 + (dify_trad)**2)**0.5

                        # Separation from nearest surface normal unit vectors
                        Surface_x_norm_unitv = difx_trad / difr_trad
                        Surface_y_norm_unitv = dify_trad / difr_trad

                        # Separation from CoM unit vectors
                        CoM_x_norm_unitv = (difx_trad) / difr_trad
                        CoM_y_norm_unitv = (dify_trad) / difr_trad

                    # If at least one surface has been defined...
                    if len(sep_surface_dict) >= 1:

                        # Loop over all surfaces
                        for m in range(0, len(sep_surface_dict)):

                            # Define 'm' surface id's dictionary key
                            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

                            # Get 'm' surface id's interior surface positional information
                            try:
                                interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                                interior_surface_com_xpos = interior_surface_com_pos_dict['pos']['x']
                                interior_surface_com_ypos = interior_surface_com_pos_dict['pos']['y']
                                interior_int_com_x = interior_surface_com_pos_dict['com']['x']
                                interior_int_com_y = interior_surface_com_pos_dict['com']['y']
                                interior_exist = 1
                            except:
                                interior_surface_com_xpos = None
                                interior_surface_com_ypos = None
                                interior_int_com_x = None
                                interior_int_com_y = None
                                interior_exist = 0

                            # Get 'm' surface id's exterior surface positional information
                            try:
                                exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])
                                exterior_surface_com_xpos = exterior_surface_com_pos_dict['pos']['x']
                                exterior_surface_com_ypos = exterior_surface_com_pos_dict['pos']['y']
                                exterior_int_com_x = exterior_surface_com_pos_dict['com']['x']
                                exterior_int_com_y = exterior_surface_com_pos_dict['com']['y']
                                exterior_exist = 1
                            except:
                                exterior_exist = 0
                                exterior_surface_com_xpos = None
                                exterior_surface_com_ypos = None
                                exterior_int_com_x = None
                                exterior_int_com_y = None

                            # If an exterior surface exists for given surface id (m), find separation distance from nearest surface
                            if (exterior_exist == 1):

                                # Loop over exterior surface id's positions
                                for id in range(0, len(exterior_surface_com_xpos)):
                                    #If bin is an interior edge bin for mth interface structure, continue...

                                    # Calculate separation distance of reference bin from exterior surface point
                                    difx_width = self.utility_functs.sep_dist_x(xpos_ref, exterior_surface_com_xpos[id])
                                    dify_width = self.utility_functs.sep_dist_y(ypos_ref, exterior_surface_com_ypos[id])
                                    difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                    #If this distance is the shortest separation distance thus far, save it as nearest surface point
                                    if difr<difr_short:
                                        difr_short=difr                         # Total separation distance from nearest surface point
                                        Surface_x_norm_unitv = difx_width / difr        # X-separation unit vector from nearest surface point
                                        Surface_y_norm_unitv = dify_width / difr        # Y-separation unit vector from nearest surface point
                                        interior_bin_short = 0                  # If true (1), nearest surface is interior surface point
                                        exterior_bin_short = 1                  # If true (1), nearest surface is exterior surface point

                            # If an interior surface exists for given surface id (m), find separation distance from nearest surface
                            if (interior_exist == 1):

                                # Loop over interior surface id's positions
                                for id in range(0, len(interior_surface_com_xpos)):

                                    # Calculate separation distance of reference bin from interior surface point
                                    difx_width = self.utility_functs.sep_dist_x(xpos_ref, interior_surface_com_xpos[id])
                                    dify_width = self.utility_functs.sep_dist_y(ypos_ref, interior_surface_com_ypos[id])
                                    difr = ( (difx_width)**2 + (dify_width)**2)**0.5

                                    #If this distance is the shortest separation distance thus far, save it as nearest surface point
                                    if difr<difr_short:
                                        difr_short=difr                         # Total separation distance from nearest surface point
                                        Surface_x_norm_unitv = difx_width / difr        # X-separation unit vector from nearest surface point
                                        Surface_y_norm_unitv = dify_width / difr        # Y-separation unit vector from nearest surface point
                                        interior_bin_short = 1                  # If true (1), nearest surface is interior surface point
                                        exterior_bin_short = 0                  # If true (1), nearest surface is exterior surface point

                    #If particles in reference bin, continue...
                    if len(self.binParts[ix][iy])>0:

                        #Loop over particles in reference bin
                        for h in range(0, len(self.binParts[ix][iy])):

                            #Calculate x and y orientation of reference particle's active force
                            px = np.sin(self.ang[self.binParts[ix][iy][h]])
                            py = -np.cos(self.ang[self.binParts[ix][iy][h]])

                            #If nearest surface is exterior surface, calculate alignment with that surface
                            if exterior_bin_short == 1:
                                Surface_x_dot_p = (-Surface_x_norm_unitv * px)
                                Surface_y_dot_p = (-Surface_y_norm_unitv * py)

                            #If nearest surface is interior surface, calculate alignment with that surface
                            elif interior_bin_short == 1:
                                Surface_x_dot_p = (Surface_x_norm_unitv * px)
                                Surface_y_dot_p = (Surface_y_norm_unitv * px)

                            # Total alignment with nearest surface
                            Surface_r_dot_p = Surface_x_dot_p + Surface_y_dot_p

                            #Calculate alignment towards largest cluster's CoM (or middle of simulation box if no cluster present)
                            CoM_x_dot_p = (-CoM_x_norm_unitv * px)
                            CoM_y_dot_p = (-CoM_y_norm_unitv * py)
                            CoM_r_dot_p= CoM_x_dot_p + CoM_y_dot_p

                            # Save alignment of particle with nearest surface
                            part_align[self.binParts[ix][iy][h]] = Surface_r_dot_p

                            # Save alignment of particle with nearest surface
                            part_difr[self.binParts[ix][iy][h]] = difr_short

                            # Total alignment of all particles per dimension with nearest surface in reference bin
                            Surface_align[ix][iy] += Surface_r_dot_p
                            Surface_align_x[ix][iy] += Surface_x_dot_p
                            Surface_align_y[ix][iy] += Surface_y_dot_p
                            Align_num[ix][iy]+= 1

                            # Total alignment of all particles per dimension with largest cluster's CoM in reference bin
                            CoM_align[ix][iy] += CoM_r_dot_p
                            CoM_align_x[ix][iy] += CoM_x_dot_p
                            CoM_align_y[ix][iy] += CoM_y_dot_p

                            #Calculate total alignment and number of particles per bin for type A particles
                            if self.typ[self.binParts[ix][iy][h]]==0:
                                Surface_align_A[ix][iy] += Surface_r_dot_p
                                Surface_align_x_A[ix][iy] += Surface_x_dot_p
                                Surface_align_y_A[ix][iy] += Surface_y_dot_p
                                Align_num_A[ix][iy]+= 1
                                CoM_align_A[ix][iy] += CoM_r_dot_p
                                CoM_align_x_A[ix][iy] += CoM_x_dot_p
                                CoM_align_y_A[ix][iy] += CoM_y_dot_p

                            #Calculate total alignment and number of particles per bin for type B particles
                            elif self.typ[self.binParts[ix][iy][h]]==1:
                                Surface_align_B[ix][iy] += Surface_r_dot_p
                                Surface_align_x_B[ix][iy] += Surface_x_dot_p
                                Surface_align_y_B[ix][iy] += Surface_y_dot_p
                                Align_num_B[ix][iy]+= 1
                                CoM_align_B[ix][iy] += CoM_r_dot_p
                                CoM_align_x_B[ix][iy] += CoM_x_dot_p
                                CoM_align_y_B[ix][iy] += CoM_y_dot_p


        #Calculate average alignment per bin
        #Loop over bins in system
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if Surface_align_avg[ix][iy]==0:

                    #Calculate alignment with nearest interfacial surface
                    #If denominator is non-zero, continue...
                    if Align_num[ix][iy]>0:
                        Surface_align_avg[ix][iy] = Surface_align[ix][iy] / Align_num[ix][iy]
                        Surface_align_avg_x[ix][iy] = Surface_align_x[ix][iy] / Align_num[ix][iy]
                        Surface_align_avg_y[ix][iy] = Surface_align_y[ix][iy] / Align_num[ix][iy]

                        CoM_align_avg[ix][iy] = CoM_align[ix][iy] / Align_num[ix][iy]
                        CoM_align_avg_x[ix][iy] = CoM_align_x[ix][iy] / Align_num[ix][iy]
                        CoM_align_avg_y[ix][iy] = CoM_align_y[ix][iy] / Align_num[ix][iy]
                        if Align_num_A[ix][iy]>0:
                            Surface_align_avg_A[ix][iy] = Surface_align_A[ix][iy] / Align_num_A[ix][iy]
                            Surface_align_avg_x_A[ix][iy] = Surface_align_x_A[ix][iy] / Align_num_A[ix][iy]
                            Surface_align_avg_y_A[ix][iy] = Surface_align_y_A[ix][iy] / Align_num_A[ix][iy]

                            CoM_align_avg_A[ix][iy] = CoM_align_A[ix][iy] / Align_num_A[ix][iy]
                            CoM_align_avg_x_A[ix][iy] = CoM_align_x_A[ix][iy] / Align_num_A[ix][iy]
                            CoM_align_avg_y_A[ix][iy] = CoM_align_y_A[ix][iy] / Align_num_A[ix][iy]
                        if Align_num_B[ix][iy]>0:
                            Surface_align_avg_B[ix][iy] = Surface_align_B[ix][iy] / Align_num_B[ix][iy]
                            Surface_align_avg_x_B[ix][iy] = Surface_align_x_B[ix][iy] / Align_num_B[ix][iy]
                            Surface_align_avg_y_B[ix][iy] = Surface_align_y_B[ix][iy] / Align_num_B[ix][iy]

                            CoM_align_avg_B[ix][iy] = CoM_align_B[ix][iy] / Align_num_B[ix][iy]
                            CoM_align_avg_x_B[ix][iy] = CoM_align_x_B[ix][iy] / Align_num_B[ix][iy]
                            CoM_align_avg_y_B[ix][iy] = CoM_align_y_B[ix][iy] / Align_num_B[ix][iy]

        CoM_align_dict = {'bin': {'all': {'x': CoM_align_avg_x, 'y': CoM_align_avg_y, 'mag': CoM_align_avg, 'num': Align_num}, 'A': {'x': CoM_align_avg_x_A, 'y': CoM_align_avg_y_A, 'mag': CoM_align_avg_A, 'num': Align_num_A}, 'B': {'x': CoM_align_avg_x_B, 'y': CoM_align_avg_y_B, 'mag': CoM_align_avg_B, 'num': Align_num_B}}, 'part': part_align}
        Surface_align_dict = {'bin': {'all': {'x': Surface_align_avg_x, 'y': Surface_align_avg_y, 'mag': Surface_align_avg, 'num': Align_num}, 'A': {'x': Surface_align_avg_x_A, 'y': Surface_align_avg_y_A, 'mag': Surface_align_avg_A, 'num': Align_num_A}, 'B': {'x': Surface_align_avg_x_B, 'y': Surface_align_avg_y_B, 'mag': Surface_align_avg_B, 'num': Align_num_B}}, 'part': {'align': part_align, 'difr': part_difr}}
        return  CoM_align_dict, Surface_align_dict
    def det_planar_surface_points(self):

        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': self.r_cut})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster

        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes

        in_clust = np.where(clust_size == np.amax(clust_size) )[0][0]
        not_in_clust = np.where(clust_size != np.amax(clust_size) )[0][0]

        slow_clust_ids = np.where( (ids==in_clust) & (self.typ==0) )[0]
        fast_clust_ids = np.where( (ids==in_clust) & (self.typ==1) )[0]
        slow_not_clust_ids = np.where( (ids!=in_clust) & (self.typ==0) )[0]
        fast_not_clust_ids = np.where( (ids!=in_clust) & (self.typ==1) )[0]


        #if self.lx_box >= self.ly_box:
        #    mem_max = np.max(self.pos[typ0ind,0])
        #    mem_min = np.min(self.pos[typ0ind,0])
        #else:
        #    mem_max = np.max(self.pos[typ0ind,1])
        #    mem_min = np.min(self.pos[typ0ind,1])

        #print(mem_min)
        #print(mem_max)

        #right_ads = np.where((self.pos[typ1ind,0]<(mem_max + 10.0)) & (self.pos[typ1ind,0]>0))
        #left_ads = np.where((self.pos[typ1ind,0]>(mem_min - 10.0)) & (self.pos[typ1ind,0]<0))
        
        kinetics_dict = {'in_clust': {'all': len(slow_clust_ids) + len(fast_clust_ids), 'A': len(slow_clust_ids), 'B': len(fast_clust_ids)}, 'out_clust': {'all': len(slow_not_clust_ids) + len(fast_not_clust_ids), 'A': len(slow_not_clust_ids), 'B': len(fast_not_clust_ids)}}
        
        return kinetics_dict
    """
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
            '''
            Returns a symbolic fourier series of order `n`.

            :param n: Order of the fourier series.
            :param x: Independent variable
            :param f: Frequency of the fourier series
            '''
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
    """
