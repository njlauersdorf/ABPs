import sys, os
from gsd import hoomd
from freud import box
import freud
import numpy as np
import math
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

# Class of individual particle property measurements
class particle_props:

    def __init__(self, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, pos, ang):

        # Initialize theory functions for call back later
        theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box / 2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box / 2

        # Instantiated simulation box
        self.f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

        # Initialize utility functions for call back later
        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        # Number of particles
        self.partNum = partNum

        # Set minimum cluster size to analyze
        self.min_size = int(self.partNum / 8)

        try:
            # Total number of bins in x-direction
            self.NBins_x = int(NBins_x)

            # Total number of bins in y-direction
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        # X-length of bin
        self.sizeBin_x = self.utility_functs.roundUp(self.lx_box / self.NBins_x, 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp(self.ly_box / self.NBins_y, 6)

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Array (partNum) of particle types
        self.typ = typ

        # Array (partNum) of particle positions
        self.pos = pos

        # Array (partNum) of particle orientations
        self.ang = ang

    def particle_phase_ids(self, phasePart):
        '''
        Purpose: Takes the phase that each particle belongs to and returns arrays of
        IDs of all particles of the respective type belonging to each phase

        Inputs:
        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        phase_part_dict: dictionary containing arrays that give the id of each particle
        of each type ('all', 'A', or 'B') belonging to each respective phase
        '''

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to bulk phase
        A_bulk_id = np.where((phasePart==0) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_bulk_id = np.where((phasePart==0) & (self.typ==1))[0]        #Bulk phase structure(s)
        bulk_id = np.where(phasePart==0)[0]        #Bulk phase structure(s)

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to interface
        A_int_id = np.where((phasePart==1) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_int_id = np.where((phasePart==1) & (self.typ==1))[0]        #Bulk phase structure(s)
        int_id = np.where(phasePart==1)[0]         #All interfaces

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to gas phase
        A_gas_id = np.where((phasePart==2) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_gas_id = np.where((phasePart==2) & (self.typ==1))[0]        #Bulk phase structure(s)
        gas_id = np.where(phasePart==2)[0]         #All interfaces

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to dense phase (either bulk or interface)
        A_dense_id = np.where((phasePart!=2) & (self.typ==0))[0]
        B_dense_id = np.where((phasePart!=2) & (self.typ==1))[0]
        dense_id = np.where(phasePart!=2)[0]

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to either gas or interface
        A_gas_int_id = np.where((phasePart!=0) & (self.typ==0))[0]
        B_gas_int_id = np.where((phasePart!=0) & (self.typ==1))[0]
        gas_int_id = np.where(phasePart!=0)[0]

        # Dictionary containing arrays that give the id of each particle
        # of each type ('all', 'A', or 'B') belonging to each respective phase
        phase_part_dict = {'bulk': {'all': bulk_id, 'A': A_bulk_id, 'B': B_bulk_id}, 'int': {'all': int_id, 'A': A_int_id, 'B': B_int_id}, 'gas': {'all': gas_id, 'A': A_gas_id, 'B': B_gas_id}, 'dense': {'all': dense_id, 'A': A_dense_id, 'B': B_dense_id}, 'gas_int': {'all': gas_int_id, 'A': A_gas_int_id, 'B': B_gas_int_id}}

        return phase_part_dict

    def particle_normal_fa(self):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude pointing towards largest cluster's CoM

        Output:
        part_normal_fa: array (partNum) of each particle's active force magnitude
        pointing toward largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the active force component
        # pointing towards the largest cluster's CoM
        part_normal_fa = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in particle's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Normalize x- and y- separation distance to make unit vectors
            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #Save active force magnitude pointing toward largest cluster's CoM
            if self.typ[h]==0:
                part_normal_fa[h]=r_dot_p*self.peA
            elif self.typ[h]==1:
                part_normal_fa[h]=r_dot_p*self.peB

        return part_normal_fa

    def particle_align(self):
        '''
        Purpose: Takes the orientation and position of each particle
        to calculate the active force alignment towards the largest cluster's CoM

        Output:
        part_align: array (partNum) of each particle's active force alignment
         toward largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the active force alignment
        # towards the largest cluster's CoM
        part_align = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in particle's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Normalize x- and y- separation distance to make unit vectors
            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #Save alignment toward largest cluster's CoM
            part_align[h]=r_dot_p

        return part_align

    def particle_sep_dist(self):
        '''
        Purpose: Takes the position of each particle
        to calculate the separation distance from the largest cluster's CoM

        Output:
        part_difr: array (partNum) of each particle's separation distance from
         the largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the separation distance
        # from the largest cluster's CoM
        part_difr = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in bin's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            #Save separation distance from largest cluster's CoM
            part_difr[h] = difr

        return part_difr

    def radial_normal_fa(self):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from largest cluster's CoM

        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force toward
        largest cluster's CoM as a function of separation distance from largest custer's CoM
        '''

        # Instantiate empty array (partNum) containing the average active force alignment
        # towards the largest cluster's CoM
        align_norm = np.array([])

        # Instantiate empty array (partNum) containing the average active force magnitude
        # towards the largest cluster's CoM
        fa_norm = np.array([])

        # Instantiate empty array (partNum) containing the distance from largest cluster's CoM
        r_dist_norm = np.array([])

        # Loop over all particles
        for h in range(0, len(self.pos)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = self.pos[h,0] - 0
            dify = self.pos[h,1] - 0

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Normalize x- and y- separation distance to make unit vectors
            x_norm_unitv = (difx) / difr
            y_norm_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_norm_unitv * px) + (-y_norm_unitv * py)

            # Save alignment with largest cluster's CoM
            align_norm=np.append(align_norm, r_dot_p)

            # Save active force magnitude toward largest cluster's CoM
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, r_dot_p*self.peA)
            else:
                fa_norm=np.append(fa_norm, r_dot_p*self.peB)

            # Save separation distance from largest cluster's CoM
            r_dist_norm = np.append(r_dist_norm, difr)

        # Dictionary containing each particle's alignment and aligned active force toward
        # largest cluster's CoM as a function of separation distance from largest custer's CoM
        radial_fa_dict = {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}

        return radial_fa_dict

    def radial_surface_normal_fa(self, method2_align_dict):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from the nearest interface surface normal

        Input:
        method2_align_dict: dictionary containing the alignment
        of each particle's active force with the nearest interface's surface normal,
        per and averaged per bin
        
        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force with
        the nearest interface surface normal as a function of separation distance from largest custer's CoM
        '''

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty array (partNum) containing the average active force magnitude
        # toward the nearest interface surface normal
        fa_norm = np.array([])

        # Instantiate empty array (partNum) containing the distance from the nearest interface surface
        r_dist_norm = np.array([])

        # Loop over all particles
        for h in range(0, len(self.pos)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = self.pos[h,0] - 0
            dify = self.pos[h,1] - 0

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Save active force magnitude toward the nearest interface surface normal
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, part_align[h]*self.peA)
            else:
                fa_norm=np.append(fa_norm, part_align[h]*self.peB)

            # Save separation distance from the nearest interface surface
            r_dist_norm = np.append(r_dist_norm, difr)

        # Dictionary containing each particle's alignment and aligned active force toward
        # the nearest interface surface normal as a function of separation distance from
        # largest custer's CoM
        radial_fa_dict = {'r': r_dist_norm, 'fa': fa_norm, 'align': part_align}

        return radial_fa_dict

    def angular_velocity(self, ang_vel, phasePart):
        '''
        Purpose: Takes the angular velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        ang_vel: array (partNum) of angular velocities of each particle

        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        ang_vel_phase_dict: dictionary containing the average and standard deviation of angular velocity
        of each phase and each respective type ('all', 'A', or 'B')
        '''

        # Total angular velocity of bulk particles of respective type ('all', 'A', or 'B')
        bulk_ang_vel = 0
        bulk_A_ang_vel = 0
        bulk_B_ang_vel = 0

        # Total number of bulk particles of respective type ('all', 'A', or 'B') summed over
        bulk_ang_num = 0
        bulk_A_ang_num = 0
        bulk_B_ang_num = 0

        # Total angular velocity of interface particles of respective type ('all', 'A', or 'B')
        int_ang_vel = 0
        int_A_ang_vel = 0
        int_B_ang_vel = 0

        # Total number of interface particles of respective type ('all', 'A', or 'B') summed over
        int_ang_num = 0
        int_A_ang_num = 0
        int_B_ang_num = 0

        # Total angular velocity of gas particles of respective type ('all', 'A', or 'B')
        gas_ang_vel = 0
        gas_A_ang_vel = 0
        gas_B_ang_vel = 0

        # Total number of gas particles of respective type ('all', 'A', or 'B') summed over
        gas_ang_num = 0
        gas_A_ang_num = 0
        gas_B_ang_num = 0

        # Loop over all particles to calculate mean angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If reference particle is from bulk...
            if phasePart[h] == 0:

                # Add to total bulk phase angular velocity
                bulk_ang_vel += np.abs(ang_vel[h])
                bulk_ang_num += 1

                # Add to total A bulk phase angular velocity
                if self.typ[h] == 0:
                    bulk_A_ang_vel += np.abs(ang_vel[h])
                    bulk_A_ang_num += 1

                # Add to total B bulk phase angular velocity
                elif self.typ[h] == 1:
                    bulk_B_ang_vel += np.abs(ang_vel[h])
                    bulk_B_ang_num += 1

            # If reference particle is from interface
            elif phasePart[h] == 1:

                # Add to total interface angular velocity
                int_ang_vel += np.abs(ang_vel[h])
                int_ang_num += 1

                # Add to total A interface angular velocity
                if self.typ[h] == 0:
                    int_A_ang_vel += np.abs(ang_vel[h])
                    int_A_ang_num += 1

                # Add to total B interface angular velocity
                elif self.typ[h] == 1:
                    int_B_ang_vel += np.abs(ang_vel[h])
                    int_B_ang_num += 1

            # If reference particle is from gas
            elif phasePart[h] == 2:

                # Add to total gas phase angular velocity
                gas_ang_vel += np.abs(ang_vel[h])
                gas_ang_num += 1

                # Add to total A gas phase angular velocity
                if self.typ[h] == 0:
                    gas_A_ang_vel += np.abs(ang_vel[h])
                    gas_A_ang_num += 1

                # Add to total B gas phase angular velocity
                elif self.typ[h] == 1:
                    gas_B_ang_vel += np.abs(ang_vel[h])
                    gas_B_ang_num += 1

        # Average total bulk angular velocities over total bulk particles of respective type ('all', 'A', or 'B')
        if bulk_ang_num > 0:
            bulk_ang_vel_avg = bulk_ang_vel/bulk_ang_num
            if bulk_A_ang_num > 0:
                bulk_A_ang_vel_avg = bulk_A_ang_vel/bulk_A_ang_num
            else:
                bulk_A_ang_vel_avg = 0

            if bulk_B_ang_num > 0:
                bulk_B_ang_vel_avg = bulk_B_ang_vel/bulk_B_ang_num
            else:
                bulk_B_ang_vel_avg = 0
        else:
            bulk_ang_vel_avg = 0
            bulk_A_ang_vel_avg = 0
            bulk_B_ang_vel_avg = 0

        # Average total interface angular velocities over total interface particles of respective type ('all', 'A', or 'B')
        if int_ang_num > 0:
            int_ang_vel_avg = int_ang_vel/int_ang_num

            if int_A_ang_num > 0:
                int_A_ang_vel_avg = int_A_ang_vel/int_A_ang_num
            else:
                int_A_ang_vel_avg = 0

            if int_B_ang_num > 0:
                int_B_ang_vel_avg = int_B_ang_vel/int_B_ang_num
            else:
                int_B_ang_vel_avg = 0
        else:
            int_ang_vel_avg = 0
            int_A_ang_vel_avg = 0
            int_B_ang_vel_avg = 0

        # Average total gas phase angular velocities over total gas particles of respective type ('all', 'A', or 'B')
        if gas_ang_num > 0:
            gas_ang_vel_avg = gas_ang_vel/gas_ang_num
            if gas_A_ang_num > 0:
                gas_A_ang_vel_avg = gas_A_ang_vel/gas_A_ang_num
            else:
                gas_A_ang_vel_avg = 0

            if gas_B_ang_num > 0:
                gas_B_ang_vel_avg = gas_B_ang_vel/gas_B_ang_num
            else:
                gas_B_ang_vel_avg = 0
        else:
            gas_ang_vel_avg = 0
            gas_A_ang_vel_avg = 0
            gas_B_ang_vel_avg = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for bulk phase and particle type ('all', 'A', or 'B')
        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for interface and particle type ('all', 'A', or 'B')
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for gas phase and particle type ('all', 'A', or 'B')
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        # Loop over all particles to calculate standard deviation angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If bulk, sum to bulk deviations from mean of respective type ('all', 'A', or 'B')
            if phasePart[h] == 0:

                bulk_all_val += (np.abs(ang_vel[h])-bulk_ang_vel_avg)**2
                if self.typ[h]==0:
                    bulk_A_val += (np.abs(ang_vel[h])-bulk_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_B_val += (np.abs(ang_vel[h])-bulk_B_ang_vel_avg)**2

            # If interface, sum to interface deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 1:

                int_all_val += (np.abs(ang_vel[h])-int_ang_vel_avg)**2
                if self.typ[h]==0:
                    int_A_val += (np.abs(ang_vel[h])-int_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    int_B_val += (np.abs(ang_vel[h])-int_B_ang_vel_avg)**2

            # If gas, sum to gas deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 2:

                gas_all_val += (np.abs(ang_vel[h])-gas_ang_vel_avg)**2
                if self.typ[h]==0:
                    gas_A_val += (np.abs(ang_vel[h])-gas_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    gas_B_val += (np.abs(ang_vel[h])-gas_B_ang_vel_avg)**2

        # Average sum of bulk deviations divided by number of bulk particles of respective type ('all', 'A', or 'B')
        if bulk_ang_num > 0:

            bulk_ang_vel_std = (bulk_all_val/bulk_ang_num)**0.5

            if bulk_A_ang_num > 0:
                bulk_A_ang_vel_std = (bulk_A_val/bulk_A_ang_num)**0.5
            else:
                bulk_A_ang_vel_std = 0

            if bulk_B_ang_num > 0:
                bulk_B_ang_vel_std = (bulk_B_val/bulk_B_ang_num)**0.5
            else:
                bulk_B_ang_vel_std = 0

        else:
            bulk_ang_vel_std = 0
            bulk_A_ang_vel_std = 0
            bulk_B_ang_vel_std = 0

        # Average sum of interface deviations divided by number of interface particles of respective type ('all', 'A', or 'B')
        if int_ang_num > 0:
            int_ang_vel_std = (int_all_val/int_ang_num)**0.5
            if int_A_ang_num > 0:
                int_A_ang_vel_std = (int_A_val/int_A_ang_num)**0.5
            else:
                int_A_ang_vel_std = 0
            if int_B_ang_num > 0:
                int_B_ang_vel_std = (int_B_val/int_B_ang_num)**0.5
            else:
                int_B_ang_vel_std = 0
        else:
            int_ang_vel_std = 0
            int_A_ang_vel_std = 0
            int_B_ang_vel_std = 0

        # Average sum of gas deviations divided by number of gas particles of respective type ('all', 'A', or 'B')
        if gas_ang_num > 0:
            gas_ang_vel_std = (gas_all_val/gas_ang_num)**0.5
            if gas_A_ang_num > 0:
                gas_A_ang_vel_std = (gas_A_val/gas_A_ang_num)**0.5
            else:
                gas_A_ang_vel_std = 0
            if gas_B_ang_num > 0:
                gas_B_ang_vel_std = (gas_B_val/gas_B_ang_num)**0.5
            else:
                gas_B_ang_vel_std = 0
        else:
            gas_ang_vel_std = 0
            gas_A_ang_vel_std = 0
            gas_B_ang_vel_std = 0

        # Dictionary containing the average and standard deviation of angular velocity
        # of each phase and each respective type ('all', 'A', or 'B')
        ang_vel_phase_dict = {'bulk': {'all': {'avg': bulk_ang_vel_avg, 'std': bulk_ang_vel_std}, 'A': {'avg': bulk_A_ang_vel_avg, 'std': bulk_A_ang_vel_std}, 'B': {'avg': bulk_B_ang_vel_avg, 'std': bulk_B_ang_vel_std} }, 'int': {'all': {'avg': int_ang_vel_avg, 'std': int_ang_vel_std}, 'A': {'avg': int_A_ang_vel_avg, 'std': int_A_ang_vel_std}, 'B': {'avg': int_B_ang_vel_avg, 'std': int_B_ang_vel_std}}, 'gas': {'all': {'avg': gas_ang_vel_avg, 'std': gas_ang_vel_std}, 'A': {'avg': gas_A_ang_vel_avg, 'std': gas_A_ang_vel_std}, 'B': {'avg': gas_B_ang_vel_avg, 'std': gas_B_ang_vel_std}}}

        return ang_vel_phase_dict

    def velocity(self, vel, phasePart):
        '''
        Purpose: Takes the velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        vel: array (partNum) of velocities of each particle

        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        vel_phase_dict: dictionary containing the average and standard deviation of velocity
        of each phase and each respective type ('all', 'A', or 'B')
        '''

        # Total velocity of bulk particles of respective type ('all', 'A', or 'B')
        bulk_vel = 0
        bulk_A_vel = 0
        bulk_B_vel = 0

        # Total number of bulk particles of respective type ('all', 'A', or 'B') summed over
        bulk_num = 0
        bulk_A_num = 0
        bulk_B_num = 0

        # Total velocity of interface particles of respective type ('all', 'A', or 'B')
        int_vel = 0
        int_A_vel = 0
        int_B_vel = 0

        # Total number of interface particles of respective type ('all', 'A', or 'B') summed over
        int_num = 0
        int_A_num = 0
        int_B_num = 0

        # Total velocity of gas particles of respective type ('all', 'A', or 'B')
        gas_vel = 0
        gas_A_vel = 0
        gas_B_vel = 0

        # Total number of gas particles of respective type ('all', 'A', or 'B') summed over
        gas_num = 0
        gas_A_num = 0
        gas_B_num = 0

        # Loop over all particles to calculate mean angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If reference particle is from bulk
            if phasePart[h] == 0:

                # Add to total bulk velocity
                bulk_vel += vel[h]
                bulk_num += 1

                # Add to total A bulk velocity
                if self.typ[h] == 0:
                    bulk_A_vel += vel[h]
                    bulk_A_num += 1

                # Add to total B bulk velocity
                elif self.typ[h] == 1:
                    bulk_B_vel += vel[h]
                    bulk_B_num += 1

            # If reference particle is from interface
            elif phasePart[h] == 1:

                # Add to total interface velocity
                int_vel += vel[h]
                int_num += 1

                # Add to total A interface velocity
                if self.typ[h] == 0:
                    int_A_vel += vel[h]
                    int_A_num += 1

                # Add to total B interface velocity
                elif self.typ[h] == 1:
                    int_B_vel += vel[h]
                    int_B_num += 1

            # If reference particle is from gas
            elif phasePart[h] == 2:

                # Add to total gas phase velocity
                gas_vel += vel[h]
                gas_num += 1

                # Add to total A gas phase velocity
                if self.typ[h] == 0:
                    gas_A_vel += vel[h]
                    gas_A_num += 1

                # Add to total B gas phase velocity
                elif self.typ[h] == 1:
                    gas_B_vel += vel[h]
                    gas_B_num += 1

        # Average total bulk phase angular velocities over total bulk particles of respective type ('all', 'A', or 'B')
        if bulk_num > 0:
            bulk_vel_avg = bulk_vel/bulk_num
            if bulk_A_num > 0:
                bulk_A_vel_avg = bulk_A_vel/bulk_A_num
            else:
                bulk_A_vel_avg = 0

            if bulk_B_num > 0:
                bulk_B_vel_avg = bulk_B_vel/bulk_B_num
            else:
                bulk_B_vel_avg = 0
        else:
            bulk_vel_avg = 0
            bulk_A_vel_avg = 0
            bulk_B_vel_avg = 0

        # Average total interface phase velocities over total interface particles of respective type ('all', 'A', or 'B')
        if int_num > 0:
            int_vel_avg = int_vel/int_num

            if int_A_num > 0:
                int_A_vel_avg = int_A_vel/int_A_num
            else:
                int_A_vel_avg = 0

            if int_B_num > 0:
                int_B_vel_avg = int_B_vel/int_B_num
            else:
                int_B_vel_avg = 0
        else:
            int_vel_avg = 0
            int_A_vel_avg = 0
            int_B_vel_avg = 0

        # Average total gas phase velocities over total gas particles of respective type ('all', 'A', or 'B')
        if gas_num > 0:
            gas_vel_avg = gas_vel/gas_num
            if gas_A_num > 0:
                gas_A_vel_avg = gas_A_vel/gas_A_num
            else:
                gas_A_vel_avg = 0

            if gas_B_num > 0:
                gas_B_vel_avg = gas_B_vel/gas_B_num
            else:
                gas_B_vel_avg = 0
        else:
            gas_vel_avg = 0
            gas_A_vel_avg = 0
            gas_B_vel_avg = 0

        # Instantiate sums of all particles' deviations from mean velocity for bulk phase and particle type ('all', 'A', or 'B')
        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0

        # Instantiate sums of all particles' deviations from mean velocity for interface and particle type ('all', 'A', or 'B')
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0

        # Instantiate sums of all particles' deviations from mean velocity for gas phase and particle type ('all', 'A', or 'B')
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        # Loop over all particles to calculate standard deviation velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If bulk, sum to bulk deviations from mean of respective type ('all', 'A', or 'B')
            if phasePart[h] == 0:

                bulk_all_val += (np.abs(vel[h])-bulk_vel_avg)**2

                if self.typ[h]==0:
                    bulk_A_val += (np.abs(vel[h])-bulk_A_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_A_val += (np.abs(vel[h])-bulk_B_vel_avg)**2

            # If interface, sum to interface deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 1:

                int_all_val += (np.abs(vel[h])-int_vel_avg)**2

                if self.typ[h]==0:
                    int_A_val += (np.abs(vel[h])-int_A_vel_avg)**2
                elif self.typ[h]==1:
                    int_A_val += (np.abs(vel[h])-int_B_vel_avg)**2

            # If gas, sum to gas deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 2:

                gas_all_val += (np.abs(vel[h])-gas_vel_avg)**2

                if self.typ[h]==0:
                    gas_A_val += (np.abs(vel[h])-gas_A_vel_avg)**2
                elif self.typ[h]==1:
                    gas_A_val += (np.abs(vel[h])-gas_B_vel_avg)**2

        # Average sum of bulk deviations divided by number of bulk particles of respective type ('all', 'A', or 'B')
        if bulk_num > 0:
            bulk_vel_std = (bulk_all_val/bulk_num)**0.5
            if bulk_A_num > 0:
                bulk_A_vel_std = (bulk_A_val/bulk_A_num)**0.5
            else:
                bulk_A_vel_std = 0
            if bulk_B_num > 0:
                bulk_B_vel_std = (bulk_B_val/bulk_B_num)**0.5
            else:
                bulk_B_vel_std = 0
        else:
            bulk_vel_std = 0
            bulk_A_vel_std = 0
            bulk_B_vel_std = 0

        # Average sum of interface deviations divided by number of interface particles of respective type ('all', 'A', or 'B')
        if int_num > 0:
            int_vel_std = (int_all_val/int_num)**0.5
            if int_A_num > 0:
                int_A_vel_std = (int_A_val/int_A_num)**0.5
            else:
                int_A_vel_std = 0
            if int_B_num > 0:
                int_B_vel_std = (int_B_val/int_B_num)**0.5
            else:
                int_B_vel_std = 0
        else:
            int_vel_std = 0
            int_A_vel_std = 0
            int_B_vel_std = 0

        # Average sum of gas deviations divided by number of gas particles of respective type ('all', 'A', or 'B')
        if gas_num > 0:
            gas_vel_std = (gas_all_val/gas_num)**0.5
            if gas_A_num > 0:
                gas_A_vel_std = (gas_A_val/gas_A_num)**0.5
            else:
                gas_A_vel_std = 0
            if gas_B_num > 0:
                gas_B_vel_std = (gas_B_val/gas_B_num)**0.5
            else:
                gas_B_vel_std = 0
        else:
            gas_vel_std = 0
            gas_A_vel_std = 0
            gas_B_vel_std = 0

        # Dictionary containing the average and standard deviation of velocity
        # of each phase and each respective type ('all', 'A', or 'B')
        vel_phase_dict = {'bulk': {'all': {'avg': bulk_vel_avg, 'std': bulk_vel_std}, 'A': {'avg': bulk_A_vel_avg, 'std': bulk_A_vel_std}, 'B': {'avg': bulk_B_vel_avg, 'std': bulk_B_vel_std} }, 'int': {'all': {'avg': int_vel_avg, 'std': int_vel_std}, 'A': {'avg': int_A_vel_avg, 'std': int_A_vel_std}, 'B': {'avg': int_B_vel_avg, 'std': int_B_vel_std}}, 'gas': {'all': {'avg': gas_vel_avg, 'std': gas_vel_std}, 'A': {'avg': gas_A_vel_avg, 'std': gas_A_vel_std}, 'B': {'avg': gas_B_vel_avg, 'std': gas_B_vel_std}}}

        return vel_phase_dict
    
    def single_velocity(self, vel, prev_pos, prev_ang, ori):
        '''
        Purpose: Takes the velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        vel: array (partNum) of velocities of each particle

        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        vel_phase_dict: dictionary containing the average and standard deviation of velocity
        of each phase and each respective type ('all', 'A', or 'B')
        '''
        
        dx, dy, dr = self.utility_functs.sep_dist_arr(self.pos, prev_pos, difxy=True)
    
        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        if len(typ1ind)>0:

            # Add to total bulk velocity
            typ0_vel_mag = vel['mag'][typ0ind]
            typ0_vel_x = vel['x'][typ0ind]
            typ0_vel_y = vel['y'][typ0ind]

            typ1_vel_mag = vel['mag'][typ1ind]
            typ1_vel_x = vel['x'][typ1ind]
            typ1_vel_y = vel['y'][typ1ind]

            typ0_avg = np.mean(typ1_vel_mag)

            r = np.arange(self.r_cut, 10*self.r_cut, self.r_cut)
            
            for r_dist in r:

                # Neighbor list query arguments to find interacting particles
                if r_dist == self.r_cut:
                    r_start = 0.1
                else:
                    r_start = self.r_cut+0.000001
                
                query_args = dict(mode='ball', r_min = r_start, r_max = r_dist)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)
                
                # Locate potential neighbor particles by type in the dense phase
                system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(prev_pos[typ0ind]))
                system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(prev_pos[typ1ind]))

                # Generate neighbor list of dense phase particles (per query args) of respective type A neighboring bulk phase reference particles of type B
                AB_nlist = system_A.query(self.f_box.wrap(prev_pos[typ1ind]), query_args).toNeighborList()                
                if len(AB_nlist.query_point_indices) > 0:
                    
                    # Find neighbors list IDs where 0 is reference particle
                    loc = np.where(AB_nlist.query_point_indices==0)[0]

                    fast_dx, fast_dy, fast_dr = self.utility_functs.sep_dist_arr(self.pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ0ind][AB_nlist.point_indices[loc]], difxy=True)

                    near_dr = dr[typ1ind][AB_nlist.query_point_indices[loc]]
                    near_dx = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                    near_dy = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                    
                    difx, dify, difr = self.utility_functs.sep_dist_arr(prev_pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ1ind][AB_nlist.query_point_indices[loc]], difxy=True)

                    vel_x_corr_loc = (difx/difr) * (near_dx/near_dr)
                    vel_y_corr_loc = (dify/difr) * (near_dy/near_dr)
                    #vel_r_corr_loc = ( vel_x_corr_loc ** 2 + vel_y_corr_loc ** 2 ) ** 0.5         

                    theta = np.arctan(dify, difx)
                    print(theta)
                    print(np.shape(ori))
                    
                    fx = ori[typ1ind,1]
                    fy = ori[typ1ind,2]
                    print(fx)
                    print(fy)
                    print(dx[typ1ind])
                    print(dy[typ1ind])
                    print(fx * dx[typ1ind]/dr[typ1ind])
                    print(fy * dx[typ1ind]/dr[typ1ind])
                    stop
                    print(self.ang[typ1ind])
                    print(prev_ang[typ1ind])


                    stop
                    difx2, dify2, difr2 = self.utility_functs.sep_dist_arr(self.pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ0ind][AB_nlist.point_indices[loc]], difxy=True)
                    
                    vel_x_corr_disp = (difx2/difr2) * (near_dx/near_dr)
                    vel_y_corr_disp = (dify2/difr2) * (near_dy/near_dr)
                    #vel_r_corr_disp = ( vel_x_corr_disp ** 2 + vel_y_corr_disp ** 2 ) ** 0.5

                    #near_vel_corr = 
                    print(vel_x_corr_loc)
                    print(vel_y_corr_loc)

                    print(vel_x_corr_disp)
                    print(vel_y_corr_disp)
                    stop
                    #Save nearest neighbor information to array


            # Dictionary containing the average and standard deviation of velocity
            # of each phase and each respective type ('all', 'A', or 'B')
            vel_dict = {'A': {'x': typ0_vel_x, 'y': typ0_vel_y, 'mag': typ0_vel_mag}, 'B': {'x': typ1_vel_x, 'y': typ1_vel_y, 'mag': typ1_vel_mag} }

            return vel_dict
    def single_msd(self, pos_prev, displace_dict):

        displace_x_typ0 = displace_dict['A']['x']
        displace_y_typ0 = displace_dict['A']['y']
        displace_r_typ0 = displace_dict['A']['mag']

        displace_x_typ1 = displace_dict['B']['x']
        displace_y_typ1 = displace_dict['B']['y']
        displace_r_typ1 = displace_dict['B']['mag']

        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]
        
        difx_typ0, dify_typ0, difr_typ0 = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)
        
        periodic_right = np.where(difx_typ0>self.hx_box)[0]
        periodic_left = np.where(difx_typ0<-self.hx_box)[0]

        periodic_top = np.where(dify_typ0>self.hy_box)[0]
        periodic_bot = np.where(dify_typ0<-self.hy_box)[0]
        print(np.min(self.pos[:,0]))
        print(np.max(self.pos[:,0]))
        stop
        #periodic_right * h_box
        difx_typ0_sim, dify_typ0_sim, difr_typ0_sim = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)

        difx_typ0_bulk, dify_typ0_bulk, difr_typ0_bulk = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)
        
        difx_typ0_bulk
        difx_typ1_sim, dify_typ1_sim, difr_typ1_sim = self.utility_functs.sep_dist_arr(self.pos[typ1ind], pos_prev[typ1ind], difxy=True)
        difx_typ1_bulk, dify_typ1_bulk, difr_typ1_bulk = self.utility_functs.sep_dist_arr(self.pos[typ1ind], pos_prev[typ1ind], difxy=True)
        
        displace_x_typ0 = np.append(displace_x_typ0, np.mean(difx_typ0_bulk))
        displace_y_typ0 = np.append(displace_y_typ0, np.mean(dify_typ0_bulk))
        displace_r_typ0 = np.append(displace_r_typ0, np.mean(difr_typ0_bulk))

        displace_x_typ1 = np.append(displace_x_typ1, np.mean(difx_typ1_bulk))
        displace_y_typ1 = np.append(displace_y_typ1, np.mean(dify_typ1_bulk))
        displace_r_typ1 = np.append(displace_r_typ1, np.mean(difr_typ1_bulk))

        x_max_typ0 = np.max(self.pos[typ0ind,0])
        y_max_typ0 = np.max(self.pos[typ0ind,1])

        x_max_typ1 = np.max(self.pos[typ1ind,0])
        y_max_typ1 = np.max(self.pos[typ1ind,1])

        msd_x_typ0 = np.zeros(len(typ0ind))
        msd_y_typ0 = np.zeros(len(typ0ind))
        msd_r_typ0 = np.zeros(len(typ0ind))

        for i in range(0, len(displace_x_typ0)):
            msd_x_typ0 = np.append(msd_x_typ0, np.sum(displace_x_typ0[i:i+1]) )
            msd_y_typ0 = np.append(msd_y_typ0, np.sum(displace_y_typ0[i:i+1]) )
            msd_r_typ0 = np.append(msd_r_typ0, np.sum(displace_r_typ0[i:i+1]) )

        for i in range(0, len(displace_x_typ1)):
            msd_x_typ1 = np.append(msd_x_typ1, np.sum(displace_x_typ1[i:i+1]) )
            msd_y_typ1 = np.append(msd_y_typ1, np.sum(displace_y_typ1[i:i+1]) )
            msd_r_typ1 = np.append(msd_r_typ1, np.sum(displace_r_typ1[i:i+1]) )

        displace_dict = {'A': {'x': displace_x_typ0, 'y': displace_y_typ0, 'mag': displace_r_typ0}, 'B': {'x': displace_x_typ1, 'y': displace_y_typ1, 'mag': displace_r_typ1} }
        msd_dict = {'A': {'x': msd_x_typ0, 'y': msd_y_typ0, 'mag': msd_r_typ0}, 'B': {'x': msd_x_typ1, 'y': msd_y_typ1, 'mag': msd_r_typ1} }
        return displace_dict, msd_dict