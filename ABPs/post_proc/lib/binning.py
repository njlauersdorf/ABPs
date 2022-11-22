import sys, os
from gsd import hoomd
import freud, numpy as np, math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage
import numpy as np, matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
from scipy.optimize import curve_fit

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory, utility

# Class of spatial binning functions
class binning:

    def __init__(self, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps):
        self.theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box / 2
        
        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box / 2

        # Number of particles
        self.partNum = partNum

        # Minimum cluster size
        self.min_size = int(self.partNum / 8)

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
        self.sizeBin_x = self.utility_functs.roundUp(self.lx_box / self.NBins_x, 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp(self.ly_box / self.NBins_y, 6)

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Array (partNum) of particle types
        self.typ = typ

    def create_bins(self):
        '''
        Purpose: Takes the number and size of bins to calculate an array of bin positions

        Outputs:
        pos_dict: dictionary containing arrays of positions of the midpoint and bottom
        left corner of each bin
        '''
        # Initialize empty arrays to find mid-point of bins
        pos_box_x_mid = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        pos_box_y_mid = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Initialize empty arrays to find bottom left corner of bins
        pos_box_x_left = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        pos_box_y_bot = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins to find bin positions
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                pos_box_x_mid[ix][iy] = (ix + 0.5) * self.sizeBin_x
                pos_box_y_mid[ix][iy] = (iy + 0.5) * self.sizeBin_y
                pos_box_x_left[ix][iy] = ix * self.sizeBin_x
                pos_box_y_bot[ix][iy] = iy * self.sizeBin_y

        # Dictionary containing arrays of spatial locations of bottom left corner and midpoint of bins
        pos_dict = {'bottom left':{'x':pos_box_x_left,
          'y':pos_box_y_bot},
         'mid point':{'x':pos_box_x_mid,  'y':pos_box_y_mid}}

        return pos_dict

    def bin_parts(self, pos, ids, clust_size):
        '''
        Purpose: Takes the number and size of bins to calculate an array of bin positions

        Inputs:
        pos: array (partNum) of positions (x,y,z) of each particle

        ids: array (partNum) of cluster ids (int) that each particle is a part of

        clust_size: array of cluster sizes (int) in terms of number of particles for each cluster id

        Outputs:
        part_dict: dictionary containing arrays (NBins_x, NBins_y) whose elements contains information
        on whether each particle within the bin a) is part of a cluster ('occParts'), b) on which type
        of particles are in the bin ('typ'), and c) on what each particle's id is ('id')
        '''

        # Instantiate empty array that contains list of each particle's id within the bin
        binParts = [[[] for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty array that contains list of each particle's type within the bin
        typParts = [[[] for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty array that tells whether each bin is part of a cluster (1) or not (0)
        occParts = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all cluster IDs
        for k in range(0, len(ids)):

            # Location of cluster center of mass shifted to mid-point of box
            tmp_posX = pos[k][0] + self.hx_box
            tmp_posY = pos[k][1] + self.hy_box

            # Bin location of cluster center of mass
            x_ind = int(tmp_posX / self.sizeBin_x)
            y_ind = int(tmp_posY / self.sizeBin_y)

            # Append cluster id
            binParts[x_ind][y_ind].append(k)

            # Appent particle type
            typParts[x_ind][y_ind].append(self.typ[k])

            # Label whether particle (and bin) is part of cluster (1) or not (0)
            if clust_size[ids[k]] >= self.min_size:
                occParts[x_ind][y_ind] = 1

        # Dictionary of arrays (NBins_x, NBins_y) of binned cluster ids, particle types, and particle ids
        part_dict = {'clust':occParts,  'typ':typParts,  'id':binParts}

        return part_dict

    def bin_align(self, orient_dict):
        '''
        Purpose: Takes the orientation of each particle and calculates the average over it and the neighboring bins.
        Reduces noise for interface detection.

        Inputs:
        orient_dict: dictionary containing the average orientation within each bin for each particle type ('all', 'A', or 'B') and in each direction ('x' or 'y')

        Outputs:
        align_dict: dictionary containing the orientation averaged over its neighboring 2 shells of bins for each particle type ('all', 'A', or 'B') and in each direction ('x' or 'y')
        '''

        # Red in average orientation of each bin for each type and direction
        p_avg_x = orient_dict['bin']['all']['x']
        p_avg_y = orient_dict['bin']['all']['y']
        p_avg_xA = orient_dict['bin']['A']['x']
        p_avg_yA = orient_dict['bin']['A']['y']
        p_avg_xB = orient_dict['bin']['B']['x']
        p_avg_yB = orient_dict['bin']['B']['y']

        #Instantiate empty arrays for calculating
        align_avg_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_mag = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_xA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_yA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_xB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_norm_yB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_xA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_yA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_magA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_xB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_yB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_magB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_magDif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_avg_num = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_xA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_yA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_xB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        align_tot_yB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all x bin indices
        for ix in range(0, self.NBins_x):

            # Find neighboring 2 shells of x-bin ids
            if ix + 2 == self.NBins_x:
                lookx = [
                 ix - 1, ix - 1, ix, ix + 1, 0]
            else:
                if ix + 1 == self.NBins_x:
                    lookx = [
                     ix - 2, ix - 1, ix, 0, 1]
                else:
                    if ix == 0:
                        lookx = [
                         self.NBins_x - 2, self.NBins_x - 1, ix, ix + 1, ix + 2]
                    else:
                        if ix == 1:
                            lookx = [
                             self.NBins_x - 1, ix - 1, ix, ix + 1, ix + 2]
                        else:
                            lookx = [
                             ix - 2, ix - 1, ix, ix + 1, ix + 2]

            # Loop over all y bin indices
            for iy in range(0, self.NBins_y):

                # Find neighboring 2 shells of x-bin ids
                if iy + 2 == self.NBins_y:
                    looky = [
                     iy - 1, iy - 1, iy, iy + 1, 0]
                else:
                    if iy + 1 == self.NBins_y:
                        looky = [
                         iy - 2, iy - 1, iy, 0, 1]
                    else:
                        if iy == 0:
                            looky = [
                             self.NBins_y - 2, self.NBins_y - 1, iy, iy + 1, iy + 2]
                        else:
                            if iy == 1:
                                looky = [
                                 self.NBins_y - 1, iy - 1, iy, iy + 1, iy + 2]
                            else:
                                looky = [
                                 iy - 2, iy - 1, iy, iy + 1, iy + 2]

                # Loop over neighboring bin indices
                for indx in lookx:
                    for indy in looky:

                        # Sum orientation of each particle type ('all', 'A', or 'B') in each direction ('x' or 'y')
                        align_tot_x[ix][iy] += p_avg_x[indx][indy]
                        align_tot_y[ix][iy] += p_avg_y[indx][indy]

                        align_tot_xA[ix][iy] += p_avg_xA[indx][indy]
                        align_tot_yA[ix][iy] += p_avg_yA[indx][indy]

                        align_tot_xB[ix][iy] += p_avg_xB[indx][indy]
                        align_tot_yB[ix][iy] += p_avg_yB[indx][indy]

                        # Sum number of bins summed over
                        align_avg_num[ix][iy] += 1

                    #Average orientation over number of bins
                    if align_avg_num[ix][iy] > 0:
                        align_avg_x[ix][iy] = align_tot_x[ix][iy] / align_avg_num[ix][iy]
                        align_avg_y[ix][iy] = align_tot_y[ix][iy] / align_avg_num[ix][iy]

                        align_avg_xA[ix][iy] = align_tot_xA[ix][iy] / align_avg_num[ix][iy]
                        align_avg_yA[ix][iy] = align_tot_yA[ix][iy] / align_avg_num[ix][iy]

                        align_avg_xB[ix][iy] = align_tot_xB[ix][iy] / align_avg_num[ix][iy]
                        align_avg_yB[ix][iy] = align_tot_yB[ix][iy] / align_avg_num[ix][iy]

                        align_avg_mag[ix][iy] = (align_avg_x[ix][iy] ** 2 + align_avg_y[ix][iy] ** 2) ** 0.5
                        align_avg_magA[ix][iy] = (align_avg_xA[ix][iy] ** 2 + align_avg_yA[ix][iy] ** 2) ** 0.5
                        align_avg_magB[ix][iy] = (align_avg_xB[ix][iy] ** 2 + align_avg_yB[ix][iy] ** 2) ** 0.5
                        align_avg_magDif[ix][iy] = align_avg_magB[ix][iy] - align_avg_magA[ix][iy]

        # Dictionary of arrays (NBins_x, NBins_y) of binned orienations in x-direction, y-direction, and magnitude for each particle type ('all', 'A', or 'B')
        align_dict = {'bin': {'all':{'x':align_avg_x, 'y':align_avg_y, 'mag':align_avg_mag},
         'A':{'x':align_avg_xA, 'y':align_avg_yA, 'mag':align_avg_magA},  'B':{'x':align_avg_xB, 'y':align_avg_yB, 'mag':align_avg_magB},  'avg dif':{'mag': align_avg_magDif}}}

        return align_dict

    def bin_vel(self, pos, prev_pos, part_dict, dt):
        '''
        Purpose: Takes the current and previous positions of each particle and calculates the velocity over
        the previous time step, then averages and bins it

        Inputs:
        pos: array (partNum) of current positions (x,y,z) of each particle

        prev_pos: array (partNum) of previous positions (x,y,z) of each particle

        part_dict: dictionary of binned particle ids and cluster information

        dt: time step (float) in Brownian units between the previous and current position data

        Outputs:
        vel_dict: dictionary containing the average velocity of each bin for each particle type ('all', 'A', or 'B')
        and in each direction ('x' or 'y')
        '''

        # Binned particle ids and cluster information
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        # Instantiate empty arrays for calculating total x and y direction velocities for each particle type ('all', 'A', or 'B')
        v_all_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_all_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_A_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_A_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_B_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_B_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for calculating average x direction, y direction, and magnitude of velocities for each particle type ('all', 'A', or 'B')
        v_avg_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_xA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_yA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_xB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_yB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_mag = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_magA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        v_avg_magB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for calculating total x and y direction velocities for each particle
        v_x_part = np.zeros(self.partNum)
        v_y_part = np.zeros(self.partNum)
        v_mag_part = np.zeros(self.partNum)

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Start sum of particle types summed over
                typ0_temp = 0
                typ1_temp = 0

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Loop over particles in bin
                    for h in binParts[ix][iy]:

                        # Calculate displacement between time steps
                        difx = self.utility_functs.sep_dist_x(pos[h,0], prev_pos[h,0])

                        dify = self.utility_functs.sep_dist_y(pos[h,1], prev_pos[h,1])

                        # Calculate velocity between time steps
                        vx = difx/dt
                        vy = dify/dt

                        # Sum velocities to bins
                        v_all_x[ix][iy] += vx
                        v_all_y[ix][iy] += vy

                        if self.typ[h] == 0:
                            typ0_temp += 1
                            v_A_x[ix][iy] += vx
                            v_A_y[ix][iy] += vy
                        elif self.typ[h] == 1:
                            typ1_temp += 1
                            v_B_x[ix][iy] += vx
                            v_B_y[ix][iy] += vy

                        # Save velocities of respective particle
                        v_x_part[h] = vx
                        v_y_part[h] = vy
                        v_mag_part[h] = (vx**2 + vy**2 )**0.5

                # Average over velocities in bins
                if len(binParts[ix][iy])>0:
                    v_avg_x[ix][iy] = v_all_x[ix][iy] / len(binParts[ix][iy])
                    v_avg_y[ix][iy] = v_all_y[ix][iy] / len(binParts[ix][iy])

                    if typ0_temp > 0:
                        v_avg_xA[ix][iy] = v_A_x[ix][iy] / typ0_temp
                        v_avg_yA[ix][iy] = v_A_y[ix][iy] / typ0_temp

                    if typ1_temp > 0:
                        v_avg_xB[ix][iy] = v_B_x[ix][iy] / typ1_temp
                        v_avg_yB[ix][iy] = v_B_y[ix][iy] / typ1_temp

                # Calculate average velocity magnitude
                v_avg_mag[ix][iy] = (v_avg_x[ix][iy] ** 2 + v_avg_y[ix][iy] ** 2) ** 0.5
                v_avg_magA[ix][iy] = (v_avg_xA[ix][iy] ** 2 + v_avg_yA[ix][iy] ** 2) ** 0.5
                v_avg_magB[ix][iy] = (v_avg_xB[ix][iy] ** 2 + v_avg_yB[ix][iy] ** 2) ** 0.5

        # Dictionary of arrays of binned (NBins_x, NBins_y) and particle-based (partNum) velocities in x-direction, y-direction, and magnitude for each particle type ('all', 'A', or 'B')
        vel_dict = {'bin': {'all':{'x':v_avg_x,
          'y':v_avg_y,  'mag':v_avg_mag},
         'A':{'x':v_avg_xA,  'y':v_avg_yA,  'mag':v_avg_magA},  'B':{'x':v_avg_xB,  'y':v_avg_yB,  'mag':v_avg_magB}}, 'part': {'x': v_x_part, 'y': v_y_part, 'mag': v_mag_part}}

        return vel_dict

    def bin_ang_vel(self, ang, prev_ang, part_dict, dt):
        '''
        Purpose: Takes the current and previous orientations of each particle and calculates the angular velocity over
        the previous time step, then averages and bins it

        Inputs:
        ang: array (partNum) of current orientations [-pi, pi] of each particle

        prev_ang: array (partNum) of previous orientations [-pi, pi] of each particle

        part_dict: dictionary of binned particle ids and cluster information

        dt: time step (float) in Brownian units between the previous and current position data

        Outputs:
        ang_vel_dict: dictionary containing the average angular velocity of each bin for each particle type ('all', 'A', or 'B')
        and in each direction ('x' or 'y')
        '''

        # Binned particle ids and cluster information
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        # Instantiate empty arrays for calculating total angular velocities for each particle type ('all', 'A', or 'B')
        ang_v_all = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        ang_v_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        ang_v_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for calculating average angular velocities for each particle type ('all', 'A', or 'B')
        ang_v_avg = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        ang_v_avg_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        ang_v_avg_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for calculating total angular velocities for each particle
        ang_v_part = np.zeros(self.partNum)

        # Loop over all particles
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Start sum of particle types summed over
                typ0_temp = 0
                typ1_temp = 0

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Loop over particles in bin
                    for h in binParts[ix][iy]:

                        # Calculate angular displacement between time steps
                        ang_v = ang[h] - prev_ang[h]
                        ang_v_abs = np.abs(ang_v)

                        # Enforce periodic boundary conditions
                        if ang_v_abs >= np.pi:
                            if ang_v < -np.pi:
                                ang_v += 2*np.pi
                            else:
                                ang_v -= 2*np.pi

                        # Calculate angular velocity between time steps
                        ang_v_all[ix][iy] += ang_v/dt
                        if self.typ[h] == 0:
                            typ0_temp += 1
                            ang_v_A[ix][iy] += ang_v/dt
                        elif self.typ[h] == 1:
                            typ1_temp += 1
                            ang_v_B[ix][iy] += ang_v/dt

                        # Convert angular velocity from radians to degree
                        ang_v_part[h] = (ang_v/dt)* (180/np.pi)
                # Average angular velocity for each bin
                if len(binParts[ix][iy])>0:
                    ang_v_avg[ix][iy] = (ang_v_all[ix][iy] / len(binParts[ix][iy]))

                    if typ0_temp > 0:
                        ang_v_avg_A[ix][iy] = (ang_v_A[ix][iy] / typ0_temp)

                    if typ1_temp > 0:
                        ang_v_avg_B[ix][iy] = (ang_v_B[ix][iy] / typ1_temp)

        # Dictionary of arrays of binned (NBins_x, NBins_y) and particle-based (partNum) angular velocities for each particle type ('all', 'A', or 'B')
        ang_vel_dict = {'bin': {'all': ang_v_avg,
         'A': ang_v_avg_A,  'B': ang_v_avg_B}, 'part': ang_v_part}

        return ang_vel_dict

    def bin_activity(self, part_dict):
        '''
        Purpose: Takes the particle ids, finds the activity, bins the particles,
        and calculates the average activity of each bin

        Inputs:
        part_dict: dictionary of binned particle ids and cluster information

        Outputs:
        activ_dict: dictionary containing the average activity of each bin
        '''

        # Binned particle ids and cluster information
        occParts = part_dict['clust']
        binParts = part_dict['id']
        typParts = part_dict['typ']

        # Instantiate empty arrays for calculating average activity per bin
        pe_avg = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Sum number of particles of each type per bin
                typ0_temp = 0
                typ1_temp = 0

                # Sum total activity per bin
                pe_sum = 0

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Loop over particles in bin
                    for h in range(0, len(binParts[ix][iy])):

                        # Sum activities in bin
                        if typParts[ix][iy][h] == 0:
                            pe_sum += self.peA
                        else:
                            pe_sum += self.peB

                    # Calculate average activity per bin
                    pe_avg[ix][iy] = pe_sum / len(binParts[ix][iy])
        # Dictionary containing array (NBins_x, NBins_y) of average activity per bin
        activ_dict = {'avg': pe_avg}

        return activ_dict

    def bin_area_frac(self, part_dict):
        '''
        Purpose: Takes the particle ids, finds the activity, bins the particles,
        and calculates the average area fraction of each bin

        Inputs:
        part_dict: dictionary of binned particle ids and cluster information

        Outputs:
        area_frac_dict: dictionary containing the average area fraction of each bin
        '''

        # Binned particle ids and cluster information
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        # Instantiate empty arrays for calculating various area fractions per bin
        area_frac = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        area_frac_dif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        fast_frac_arr = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        area_frac_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        area_frac_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Sum number of particles per type in bin
                typ0_temp = 0
                typ1_temp = 0

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Loop over particles in bin
                    for h in range(0, len(binParts[ix][iy])):

                        # Sum number of particles of each type
                        if typParts[ix][iy][h] == 0:
                            typ0_temp += 1
                        elif typParts[ix][iy][h] == 1:
                            typ1_temp += 1

                    # Calculate area fraction of all particles in bin
                    area_frac[ix][iy] = (len(binParts[ix][iy]) / (self.sizeBin_x * self.sizeBin_y)) * (math.pi / 4)

                    # Calculate area fraction of A particles in bin
                    area_frac_A[ix][iy] = (typ0_temp / (self.sizeBin_x * self.sizeBin_y)) * (math.pi / 4)

                    # Calculate area fraction of B particles in bin
                    area_frac_B[ix][iy] = (typ1_temp / (self.sizeBin_x * self.sizeBin_y)) * (math.pi / 4)

                    # Calculate fast particle fraction in bin
                    fast_frac_arr[ix][iy] = area_frac_B[ix][iy] / area_frac[ix][iy]

                    # Calculate difference in area fraction of fast species and slow species in bin
                    if self.peB >= self.peA:
                        area_frac_dif[ix][iy] = area_frac_B[ix][iy] - area_frac_A[ix][iy]
                    else:
                        area_frac_dif[ix][iy] = area_frac_A[ix][iy] - area_frac_B[ix][iy]

        # Dictionary containing arrays (NBins_x, NBins_y) of various area fractions per bin
        area_frac_dict = {'bin': {'all':area_frac,
         'A':area_frac_A,  'B':area_frac_B,  'dif':area_frac_dif,  'fast frac':fast_frac_arr}}

        return area_frac_dict

    def bin_orient(self, part_dict, pos, ang, com):
        '''
        Purpose: Takes the particle ids and orientation, bins the particles,
        and calculates the average orientation of each bin

        Inputs:
        part_dict: dictionary of binned particle ids and cluster information

        pos: array (partNum) of current positions (x,y,z) of each particle

        ang: array (partNum) of current orientations [-pi, pi] of each particle

        com: array (x,y) of largest cluster's CoM position

        Outputs:
        orient_dict: dictionary containing the average orientation of each bin
        '''

        # Binned particle ids and cluster information
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        # Largest cluster CoM position
        com_tmp_posX = com['x']
        com_tmp_posY = com['y']

        # Instantiate empty arrays for total orientation in x- and y-directions per bin for respective particle types ('all', 'A', or 'B')
        p_all_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_all_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_A_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_A_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_B_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_B_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for average orientation in x- and y-directions per bin for respective particle types ('all', 'A', or 'B')
        p_avg_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_xA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_yA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_xB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_yB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for average orientation magnitude per bin for respective particle types ('all', 'A', or 'B')
        p_avg_mag = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_magA = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        p_avg_magB = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for calculating x- and y-direction orientation for each particle
        bin_part_x = np.zeros(self.partNum)
        bin_part_y = np.zeros(self.partNum)

        # Loop over all particles
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Sum number of particles per type in bin
                typ0_temp = 0
                typ1_temp = 0

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Loop over all particles
                    for h in range(0, len(binParts[ix][iy])):

                        # Shifted x- and y-position of particle
                        x_pos = pos[binParts[ix][iy]][h][0] + self.hx_box
                        y_pos = pos[binParts[ix][iy]][h][1] + self.hy_box

                        # Calculate separation distance from largest cluster's CoM
                        if self.lx_box==self.ly_box:
                            difx = self.utility_functs.sep_dist_x(x_pos, com_tmp_posX)
                            dify = self.utility_functs.sep_dist_y(y_pos, com_tmp_posY)
                        elif self.lx_box>self.ly_box:
                            difx = self.utility_functs.sep_dist_x(x_pos, x_pos)
                            dify = self.utility_functs.sep_dist_y(y_pos, com_tmp_posY)
                        elif self.lx_box<self.ly_box:
                            difx = self.utility_functs.sep_dist_x(x_pos, com_tmp_posX)
                            dify = self.utility_functs.sep_dist_y(y_pos, y_pos)

                        difr = (difx ** 2 + dify ** 2) ** 0.5

                        # x- and y-direction orientation unit vectors
                        px = np.sin(ang[binParts[ix][iy][h]])
                        py = -np.cos(ang[binParts[ix][iy][h]])

                        # Alignment toward largest cluster's CoM
                        r_dot_p = -difx * px + -dify * py

                        # Sum orientation unit vectors in respective bin
                        p_all_x[ix][iy] += px
                        p_all_y[ix][iy] += py

                        if typParts[ix][iy][h] == 0:
                            typ0_temp += 1
                            p_A_x[ix][iy] += px
                            p_A_y[ix][iy] += py

                        elif typParts[ix][iy][h] == 1:
                            typ1_temp += 1
                            p_B_x[ix][iy] += px
                            p_B_y[ix][iy] += py

                        # Save particle orientation unit vectors
                        bin_part_x[binParts[ix][iy][h]]=px
                        bin_part_y[binParts[ix][iy][h]]=py

                    # Calculate average orientation unit vector per bin for each particle type ('all', 'A', or 'B')
                    p_avg_x[ix][iy] = p_all_x[ix][iy] / len(binParts[ix][iy])
                    p_avg_y[ix][iy] = p_all_y[ix][iy] / len(binParts[ix][iy])
                    if typ0_temp > 0:
                        p_avg_xA[ix][iy] = p_A_x[ix][iy] / typ0_temp
                        p_avg_yA[ix][iy] = p_A_y[ix][iy] / typ0_temp
                    else:
                        p_avg_xA[ix][iy] = 0.0
                        p_avg_yA[ix][iy] = 0.0
                    if typ1_temp > 0:
                        p_avg_xB[ix][iy] = p_B_x[ix][iy] / typ1_temp
                        p_avg_yB[ix][iy] = p_B_y[ix][iy] / typ1_temp
                    else:
                        p_avg_xB[ix][iy] = 0.0
                        p_avg_yB[ix][iy] = 0.0

                    # Calculate average orientation magnitude per bin for each particle type ('all', 'A', or 'B')
                    p_avg_mag[ix][iy] = (p_avg_x[ix][iy] ** 2 + p_avg_y[ix][iy] ** 2) ** 0.5
                    p_avg_magA[ix][iy] = (p_avg_xA[ix][iy] ** 2 + p_avg_yA[ix][iy] ** 2) ** 0.5
                    p_avg_magB[ix][iy] = (p_avg_xB[ix][iy] ** 2 + p_avg_yB[ix][iy] ** 2) ** 0.5

        # Dictionary containing arrays of orientation for each bin (NBins_x, NBins_y) and per particle (partNum) of various area fractions per bin
        orient_dict = {'bin': {'all':{'x':p_avg_x,
          'y':p_avg_y,  'mag':p_avg_mag},
         'A':{'x':p_avg_xA,  'y':p_avg_yA,  'mag':p_avg_magA},  'B':{'x':p_avg_xB,  'y':p_avg_yB,  'mag':p_avg_magB}}, 'part': {'x': bin_part_x, 'y': bin_part_y}}

        return orient_dict

    def bin_active_press(self, align_dict, area_frac_dict):
        '''
        Purpose: Takes the average alignment and area fraction per bin and calculates
        the pressure arising from aligned active forces.

        Inputs:
        align_dict: dictionary of arrays (NBins_x, NBins_y) of average particle aligned per bin

        area_frac_dict: align_dict: dictionary of arrays (NBins_x, NBins_y) of area fractions

        Outputs:
        press_dict: dictionary containing the average active force pressures per bin
        '''

        # Binned particle alignment
        align_mag = align_dict['bin']['all']['mag']
        align_A_mag = align_dict['bin']['A']['mag']
        align_B_mag = align_dict['bin']['B']['mag']

        # Binned particle area fraction
        area_frac = area_frac_dict['bin']['all']
        area_frac_A = area_frac_dict['bin']['A']
        area_frac_B = area_frac_dict['bin']['B']

        # Instantiate empty arrays for average active force pressure per bin for respective particle types ('all', 'A', or 'B')
        press = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_dif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all particles
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Calculate active force pressure per bin
                press[ix][iy] = area_frac[ix][iy] * align_mag[ix][iy]
                press_A[ix][iy] = area_frac_A[ix][iy] * align_A_mag[ix][iy]
                press_B[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy]
                press_dif[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] - area_frac_A[ix][iy] * align_A_mag[ix][iy]

        # Dictionary containing arrays of active force pressure for each bin (NBins_x, NBins_y)
        press_dict = {'bin':{'all':press,
         'A':press_A,  'B':press_B,  'dif':press_dif}}
        return press_dict

    def bin_active_fa(self, orient_dict, part_dict, phaseBin):
        '''
        Purpose: Takes the average orientation and phase of each bin and calculates
        the x- and y- direction active forces in the interface.

        Inputs:
        orient_dict: dictionary of arrays (NBins_x, NBins_y) of average particle orientatino per bin

        part_dict: dictionary of binned particle ids and cluster information

        phaseBin: array (NBins_x, NBins_y) identifying whether bin is part of bulk (0), interface (1), or gas (2)

        Outputs:
        act_force_dict: dictionary containing the average x- and y- direction active force per bin
        '''

        # Binned particle orientation in x- and y- direction
        orient_x = orient_dict['part']['x']
        orient_y = orient_dict['part']['y']
        binParts = part_dict['id']

        # Instantiate empty arrays for average active force per bin in either direction ('x', 'y', and 'mag', magnitude)
        fa_x = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        fa_y = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        fa_mag = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # If interface particle...
                if phaseBin[ix][iy]==1:

                    # Loop over particles in bin
                    for h in binParts[ix][iy]:

                        # Calculate active force in x- and y- direction and magnitude for respective particle type
                        if self.typ[h]==0:
                            fa_x[ix][iy] += orient_x[h] * self.peA
                            fa_y[ix][iy] += orient_y[h] * self.peA
                        else:
                            fa_x[ix][iy] += orient_x[h] * self.peB
                            fa_y[ix][iy] += orient_y[h] * self.peB
                    fa_mag[ix][iy] = ( (fa_x[ix][iy]) ** 2 + (fa_y[ix][iy]) ** 2 ) ** 0.5

        # Normalize active force in either direction by total magnitude
        fa_x = fa_x / np.max(fa_mag)
        fa_y = fa_y / np.max(fa_mag)

        # Dictionary containing arrays of active forces for each interface bin (NBins_x, NBins_y) in x- and y- directions
        act_force_dict = {'bin':{'x': fa_x, 'y': fa_y}}

        return act_force_dict

    def bin_normal_active_fa(self, align_dict, area_frac_dict, activ_dict):
        '''
        Purpose: Takes the average alignment and area fraction per bin and calculates
        the pressure arising from aligned active forces.

        Inputs:
        align_dict: dictionary of arrays (NBins_x, NBins_y) of average particle aligned per bin

        area_frac_dict: dictionary of arrays (NBins_x, NBins_y) of area fractions

        Outputs:
        press_dict: dictionary containing the average active force pressures per bin
        '''

        # Binned particle alignment
        align_mag = align_dict['bin']['all']['mag']
        align_A_mag = align_dict['bin']['A']['mag']
        align_B_mag = align_dict['bin']['B']['mag']

        # Binned area fraction
        area_frac = area_frac_dict['bin']['all']
        area_frac_A = area_frac_dict['bin']['A']
        area_frac_B = area_frac_dict['bin']['B']

        # Average activity per bin
        fa = activ_dict['avg']

        # Instantiate empty arrays for average active pressure per bin for respective particle types ('all', 'A', or 'B', or 'dif'ference between B and A)
        press = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_A = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_B = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]
        press_dif = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Average active pressure per bin
                press[ix][iy] = area_frac[ix][iy] * align_mag[ix][iy] * fa[ix][iy]
                press_A[ix][iy] = area_frac_A[ix][iy] * align_A_mag[ix][iy] * self.peA
                press_B[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] * self.peB
                press_dif[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] * self.peB - area_frac_A[ix][iy] * align_A_mag[ix][iy] * self.peA

        # Dictionary containing arrays of active force pressure for each bin (NBins_x, NBins_y)
        act_press_dict = {'bin':{'all':press,
         'A':press_A,  'B':press_B,  'dif':press_dif}}
        return act_press_dict

    def bin_interpart_press(self, pos, part_dict):
        '''
        Purpose: Takes each particle's position and phase and calculates
        the pressure arising from particle overlap/interparticle forces

        Inputs:
        pos: array (partNum) of current positions (x,y,z) of each particle

        part_dict: dictionary of binned particle ids and cluster information

        Outputs:
        press_dict: dictionary containing arrays (NBins_x, NBins_y) of the interparticle pressures per bin
        '''

        #
        binParts = part_dict['id']

        # Instantiate empty arrays for total interparticle pressure per bin
        pressure_vp = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for number of reference-neighbor particle pairs summed over
        press_num = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Instantiate empty arrays for average interparticle pressure per bin
        pressure_vp_avg = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # If particles in bin...
                if len(binParts[ix][iy]) != 0:

                    # Find nearest shell of neighboring bins, including itself
                    if ix==0:
                        ix_new_range = [self.NBins_x-1, 0, 1]
                    elif ix==self.NBins_x-1:
                        ix_new_range = [self.NBins_x-2, self.NBins_x-1, 0]
                    else:
                        ix_new_range = [ix-1, ix, ix+1]

                    if iy==0:
                        iy_new_range = [self.NBins_y-1, 0, 1]
                    elif iy==self.NBins_y-1:
                        iy_new_range = [self.NBins_y-2, self.NBins_y-1, 0]
                    else:
                        iy_new_range = [iy-1, iy, iy+1]

                    # Loop over reference particles
                    for h in range(0, len(binParts[ix][iy])):
                        # Loop over nearest shell of neighboring bins, including itself
                        for ix2 in ix_new_range:
                            for iy2 in iy_new_range:

                                # If particles in neighbor bin
                                if len(binParts[ix2][iy2])!=0:

                                    # Loop over neighbor bin particles
                                    for h2 in range(0,len(binParts[ix2][iy2])):

                                        # If particles are not identical ID...
                                        if binParts[ix2][iy2][h2] != binParts[ix][iy][h]:

                                            # Calculate interparticle separation distance
                                            difx = self.utility_functs.sep_dist_x(pos[binParts[ix][iy]][h][0], pos[binParts[ix2][iy2]][h2][0])

                                            dify = self.utility_functs.sep_dist_y(pos[binParts[ix][iy]][h][1], pos[binParts[ix2][iy2]][h2][1])

                                            difr=(difx**2+dify**2)**0.5

                                            # If particles are within LJ potential cut off radius
                                            if 0.1<=difr<=self.r_cut:

                                                # Calculate interparticle forces
                                                fx, fy = self.theory_functs.computeFLJ(difr, pos[binParts[ix][iy]][h][0], pos[binParts[ix][iy]][h][1], pos[binParts[ix2][iy2]][h2][0], pos[binParts[ix2][iy2]][h2][1], self.eps)

                                                # Calculate the interparticle stress
                                                sigx = fx * (difx)
                                                sigy = fy * (dify)

                                                # Save the interparticle pressure and number of reference-neighbor pairs summed over
                                                press_num[ix][iy] += 1
                                                pressure_vp[ix][iy] += ((sigx + sigy) / 2.)

        # Loop over all bins to average interparticle pressure within bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                pressure_vp_avg[ix][iy]=pressure_vp[ix][iy]/(2*self.sizeBin_x * self.sizeBin_y)

        # Dictionary containing arrays of interparticle pressures for each bin (NBins_x, NBins_y)
        press_dict = {'tot': pressure_vp, 'num': press_num, 'avg': pressure_vp_avg}
        return press_dict

    def curl_and_div(self, input_dict):
        '''
        Purpose: Takes some dictionary of binned measurements and calculates
        the binned curl and divergence across space for that input dictionary

        Inputs:
        input_dict: arbitrary dictionary of binned values that follow the output dictionary structure in binning.py

        Outputs:
        grad_dict: dictionary containing arrays (NBins_x, NBins_y) of the divergence of curl of the input dictionary
        for each particle type ('all', 'A', and 'B').
        '''

        # X-direction properties of each particle type ('all', 'A', or 'B') for input array (NBins_x, NBins_y)
        all_input_x = input_dict['bin']['all']['x']
        A_input_x = input_dict['bin']['A']['x']
        B_input_x = input_dict['bin']['B']['x']

        # Y-direction properties of each particle type ('all', 'A', or 'B') for input array (NBins_x, NBins_y)
        all_input_y = input_dict['bin']['all']['y']
        A_input_y = input_dict['bin']['A']['y']
        B_input_y = input_dict['bin']['B']['y']

        # Instantiate empty arrays for combining x,y dimensions into one array (NBins_x, NBins_y, 2) for each particle type ('all', 'A', or 'B')
        tot_combined = np.zeros((self.NBins_x, self.NBins_y, 2))
        tot_A_combined = np.zeros((self.NBins_x, self.NBins_y, 2))
        tot_B_combined = np.zeros((self.NBins_x, self.NBins_y, 2))

        # Loop over all bins to save input values to reshaped array
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                tot_combined[ix][iy][0] = all_input_x[ix][iy]
                tot_combined[ix][iy][1] = all_input_y[ix][iy]

                tot_A_combined[ix][iy][0] = A_input_x[ix][iy]
                tot_A_combined[ix][iy][1] = A_input_y[ix][iy]

                tot_B_combined[ix][iy][0] = B_input_x[ix][iy]
                tot_B_combined[ix][iy][1] = B_input_y[ix][iy]

        # Calculate gradient across each dimension for combined, input array
        totx_grad = np.gradient(tot_combined, axis=0)
        toty_grad = np.gradient(tot_combined, axis=1)

        totx_A_grad = np.gradient(tot_A_combined, axis=0)
        toty_A_grad = np.gradient(tot_A_combined, axis=1)

        totx_B_grad = np.gradient(tot_B_combined, axis=0)
        toty_B_grad = np.gradient(tot_B_combined, axis=1)

        # Calculate components of divergence and curl for each particle type ('all', 'A', or 'B')
        dx_over_dx = totx_grad[:, :, 0]
        dx_over_dy = totx_grad[:, :, 1]
        dy_over_dx = toty_grad[:, :, 0]
        dy_over_dy = toty_grad[:, :, 1]

        dx_over_dx_A = totx_A_grad[:, :, 0]
        dx_over_dy_A = totx_A_grad[:, :, 1]
        dy_over_dx_A = toty_A_grad[:, :, 0]
        dy_over_dy_A = toty_A_grad[:, :, 1]

        dx_over_dx_B = totx_B_grad[:, :, 0]
        dx_over_dy_B = totx_B_grad[:, :, 1]
        dy_over_dx_B = toty_B_grad[:, :, 0]
        dy_over_dy_B = toty_B_grad[:, :, 1]

        # Caculate divergence of each particle type ('all', 'A', or 'B')
        div = dx_over_dx + dy_over_dy
        div_A = dx_over_dx_A + dy_over_dy_A
        div_B = dx_over_dx_B + dy_over_dy_B

        # Caculate curl of each particle type ('all', 'A', or 'B')
        curl = -dy_over_dx + dx_over_dy
        curl_A = -dy_over_dx_A + dx_over_dy_A
        curl_B = -dy_over_dx_B + dx_over_dy_B

        # Dictionary containing arrays of divergence and curl of input arrays pressures for each particle type ('all', 'A', 'B')
        grad_dict = {'div': {'all': div, 'A': div_A, 'B': div_B}, 'curl': {'all': curl, 'A': curl_A, 'B': curl_B} }
        return grad_dict

    def decrease_bin_size(self, phaseBin, phasePart, binParts, pos, typ, factor = 4):
        '''
        Purpose: Takes some dictionary of binned measurements and calculates
        the binned curl and divergence across space for that input dictionary

        Inputs:
        input_dict: arbitrary dictionary of binned values that follow the output dictionary structure in binning.py

        Outputs:
        grad_dict: dictionary containing arrays (NBins_x, NBins_y) of the divergence of curl of the input dictionary
        for each particle type ('all', 'A', and 'B').
        '''

        # New bin width after size decrease
        end_size_x = self.sizeBin_x / factor
        end_size_y = self.sizeBin_y / factor

        # Ensure new bin size is large enough (more than WCA potential cut off radius)
        if end_size_x < self.r_cut:
            raise ValueError('input factor must be large enough so x-dimension end_size is at least equal to the LJ cut off distance (r_cut)')
        if end_size_y < self.r_cut:
            raise ValueError('input factor must be large enough so y-dimension end_size is at least equal to the LJ cut off distance (r_cut)')

        # Total number of new bins per dimension
        NBins_x = self.NBins_x * factor
        NBins_y = self.NBins_y * factor

        # Instantiate empty arrays (NBins_x, NBins_y) for saving new binned particle information at reduced bin size
        phaseBin_new = [[0 for b in range(NBins)] for a in range(NBins)]
        binParts_new = [[[] for b in range(NBins)] for a in range(NBins)]
        typParts_new = [[[] for b in range(NBins)] for a in range(NBins)]

        pos_box_x_mid = np.array([])
        pos_box_y_mid = np.array([])

        id_box_x = np.array([], dtype=int)
        id_box_y = np.array([], dtype=int)

        # Loop over new bins
        for ix in range(0, NBins):
            for iy in range(0, NBins):

                # Calculate and save new bin mid point
                pos_box_x_mid = np.append(pos_box_x_mid, (ix + 0.5)* end_size)
                pos_box_y_mid = np.append(pos_box_y_mid, (iy * 0.5) * end_size)

                # Save new bin id
                id_box_x = np.append(id_box_x, ix)
                id_box_y = np.append(id_box_y, iy)

        # Loop over old bins
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):

                # Calculate x-range of old bin
                pos_box_x_min = (ix) * self.sizeBin
                pos_box_x_max = (ix + 1) * self.sizeBin

                # Calculate y-range of old bin
                pos_box_y_min = (iy) * self.sizeBin
                pos_box_y_max = (iy + 1) * self.sizeBin

                # Find new bins within x- and y-range of old bin
                bin_loc = np.where(((pos_box_x_mid>=pos_box_x_min) & (pos_box_x_mid<=pos_box_x_max)) & ((pos_box_y_mid>=pos_box_y_min) & (pos_box_y_mid<=pos_box_y_max)))[0]

                # Save new bin phase info
                for id in bin_loc:
                    phaseBin_new[id_box_x[int(id)]][id_box_y[int(id)]] = phaseBin[ix][iy]

                # Loop over particle in old bin
                for h in binParts[ix][iy]:

                    # Shift position so [0, lx_box]
                    tmp_posX = pos[h][0] + self.hx_box

                    # Shift position so [0, ly_box]
                    tmp_posY = pos[h][1] + self.hy_box

                    # Find new bin id of particle
                    x_ind = int(tmp_posX / end_size_x)
                    y_ind = int(tmp_posY / end_size_y)

                    # Save new bins information
                    binParts_new[x_ind][y_ind].append(h)
                    typParts_new[x_ind][y_ind].append(typ[h])
        """
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                pos_box_x_left = np.append(pos_box_x_left, ix * sizeBin)
                pos_box_y_bot = np.append(pos_box_y_bot, iy * sizeBin)

                id_box_x = np.append(pos_box_x_mid, ix)
                id_box_y = np.append(pos_box_y_mid, iy)

        for ix in range(0, NBins):
            for iy in range(0, NBins):
                pos_box_x_mid_val = (ix + 0.5) * sizeBin
                pos_box_y_mid_val = (iy + 0.5) * sizeBin

                bin_loc = np.where(((pos_dict['bottom left']['x']<=pos_box_x_mid_val) & (pos_box_x_mid_val<=pos_dict['bottom left']['x']+self.sizeBin)) & ((pos_dict['bottom left']['y']<=pos_box_y_mid_val) & (pos_box_y_mid_val<=pos_dict['bottom left']['y']+self.sizeBin)))[0]

                phaseBin_new[ix][iy] = phaseBin[id_box_x[bin_loc]][id_box_y[bin_loc]]
                for h in range(0, len(binParts[id_box_x[bin_loc]][id_box_y[bin_loc]])):
                    tmp_posX = self.pos[h][0] + self.h_box
                    tmp_posY = self.pos[h][1] + self.h_box

                    x_ind = int(tmp_posX / end_size)
                    y_ind = int(tmp_posY / end_size)

                    binParts_new[x_ind][y_ind].append(binParts[ix][iy][h])
                    typParts_new[x_ind][y_ind].append(typ[binParts[ix][iy][k])

        """
        # Dictionary containing type and id information of each particle
        part_dict = {'typ':typParts_new,  'id':binParts_new}

        # Dictionary containing phase information of each new bin and particle
        phase_dict = {'bin': phaseBin_new, 'part': phasePart}

        return phase_dict, part_dict
    def phase_number_density(self, bin_count_dict, part_count_dict):
        '''
        Purpose: Takes dictionaries that have the total number of particles of each type ('all', 'A', or 'B')
        within each phase and the total number of bins of each phase and calculates the average number density per phase

        Inputs:
        bin_count_dict: dictionary containing number of bins within each phase

        part_count_dict: dictionary containing number of each type of particle ('all', 'A', or 'B') within each phase

        Outputs:
        num_dens_dict: dictionary containing average number density of each phase for each type of particle ('all', 'A', or 'B').
        '''

        # Average interface number density of each particle type ('all', 'A', or 'B')
        all_int_num_dens = part_count_dict['num']['int']['all']/(bin_count_dict['bin']['all int'] * self.sizeBin**2)
        A_int_num_dens = part_count_dict['num']['int']['A']/(bin_count_dict['bin']['all int'] * self.sizeBin**2)
        B_int_num_dens = part_count_dict['num']['int']['B']/(bin_count_dict['bin']['all int'] * self.sizeBin**2)

        # Average bulk number density of each particle type ('all', 'A', or 'B')
        all_bulk_num_dens = part_count_dict['num']['bulk']['all']/(bin_count_dict['bin']['bulk'] * self.sizeBin**2)
        A_bulk_num_dens = part_count_dict['num']['bulk']['A']/(bin_count_dict['bin']['bulk'] * self.sizeBin**2)
        B_bulk_num_dens = part_count_dict['num']['bulk']['B']/(bin_count_dict['bin']['bulk'] * self.sizeBin**2)

        # Average gas number density of each particle type ('all', 'A', or 'B')
        all_gas_num_dens = part_count_dict['num']['gas']['all']/(bin_count_dict['bin']['gas'] * self.sizeBin**2)
        A_gas_num_dens = part_count_dict['num']['gas']['A']/(bin_count_dict['bin']['gas'] * self.sizeBin**2)
        B_gas_num_dens = part_count_dict['num']['gas']['B']/(bin_count_dict['bin']['gas'] * self.sizeBin**2)

        # Average dense number density of each particle type ('all', 'A', or 'B')
        all_dense_num_dens = (part_count_dict['num']['bulk']['all']+part_count_dict['num']['int']['all'])/((bin_count_dict['bin']['bulk']+bin_count_dict['bin']['all int']) * self.sizeBin**2)
        A_dense_num_dens = (part_count_dict['num']['bulk']['A']+part_count_dict['num']['int']['A'])/((bin_count_dict['bin']['bulk']+bin_count_dict['bin']['all int']) * self.sizeBin**2)
        B_dense_num_dens = (part_count_dict['num']['bulk']['B']+part_count_dict['num']['int']['B'])/((bin_count_dict['bin']['bulk']+bin_count_dict['bin']['all int']) * self.sizeBin**2)

        # Dictionary containing average number density of each phase for each particle type ('all', 'A', or 'B')
        num_dens_dict = {'bulk': {'all': all_bulk_num_dens, 'A': A_bulk_num_dens, 'B': B_bulk_num_dens}, 'int': {'all': all_int_num_dens, 'A': A_int_num_dens, 'B': B_int_num_dens}, 'gas': {'all': all_gas_num_dens, 'A': A_gas_num_dens, 'B': B_gas_num_dens}, 'dense': {'all': all_dense_num_dens, 'A': A_dense_num_dens, 'B': B_dense_num_dens}}

        return num_dens_dict
