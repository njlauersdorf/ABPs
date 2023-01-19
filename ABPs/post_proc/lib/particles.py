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

    def __init__(self, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, ang):

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

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Array (partNum) of particle types
        self.typ = typ

        # Array (partNum) of particle positions
        self.pos = pos

        # Array (partNum) of particle orientations
        self.ang = ang

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

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
    
    def single_velocity(self, prev_pos, prev_ang, ori):
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


        # Add to total bulk velocity
        typ0_vel_x = dx[typ0ind]
        typ0_vel_y = dy[typ0ind]
        typ0_vel_mag = (typ0_vel_x**2 + typ0_vel_y**2) ** 0.5
        pos0 = self.pos[typ0ind]

        typ1_vel_x = dx[typ1ind]
        typ1_vel_y = dy[typ1ind]
        typ1_vel_mag = (typ1_vel_x**2 + typ1_vel_y**2) ** 0.5
        pos1 = self.pos[typ1ind]

        typ0_avg = np.mean(typ0_vel_mag)
        typ1_avg = np.mean(typ1_vel_mag)

        r = np.arange(self.r_cut, 7*self.r_cut, self.r_cut)
        
        for r_dist in r:

            # Neighbor list query arguments to find interacting particles
            if r_dist == self.r_cut:
                r_start = 0.1
            else:
                r_start = self.r_cut+0.000001
            
            query_args = dict(mode='ball', r_min = r_start, r_max = r_dist)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)
            
            # Locate potential neighbor particles by type in the dense phase
            system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(prev_pos[typ0ind]))

            # Generate neighbor list of dense phase particles (per query args) of respective type A neighboring bulk phase reference particles of type B
            AB_nlist = system_A.query(self.f_box.wrap(prev_pos[typ1ind]), query_args).toNeighborList()                

            if len(AB_nlist.query_point_indices) > 0:
                
                # Find neighbors list IDs where 0 is reference particle
                loc = np.where(AB_nlist.query_point_indices==0)[0]

                slow_displace_dx, slow_displace_dy, slow_displace_dr = self.utility_functs.sep_dist_arr(self.pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ0ind][AB_nlist.point_indices[loc]], difxy=True)

                fast_displace_dr = dr[typ1ind][AB_nlist.query_point_indices[loc]]
                fast_displace_dx = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                fast_displace_dy = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                
                sep_difx, sep_dify, sep_difr = self.utility_functs.sep_dist_arr(prev_pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ1ind][AB_nlist.query_point_indices[loc]], difxy=True)


                vel_x_corr_loc = (slow_displace_dx/slow_displace_dr) * (fast_displace_dx/fast_displace_dr)
                vel_y_corr_loc = (slow_displace_dy/slow_displace_dr) * (fast_displace_dy/fast_displace_dr)
                vel_r_corr_loc = ( vel_x_corr_loc ** 2 + vel_y_corr_loc ** 2 ) ** 0.5         

                theta = np.arctan(dify, difx)

                fx = ori[typ1ind,1]
                fy = ori[typ1ind,2]

                f_x_corr_loc = (slow_displace_dx/slow_displace_dr) * (fx)
                f_y_corr_loc = (slow_displace_dy/slow_displace_dr) * (fy)
                f_r_corr_loc = (f_x_corr_loc ** 2 + f_y_corr_loc ** 2) ** 0.5



            # Dictionary containing the average and standard deviation of velocity
            # of each phase and each respective type ('all', 'A', or 'B')
            vel_plot_dict = {'A': {'x': typ0_vel_x, 'y': typ0_vel_y, 'mag': typ0_vel_mag, 'pos': pos0}, 'B': {'x': typ1_vel_x, 'y': typ1_vel_y, 'mag': typ1_vel_mag, 'pos': pos1} }
            corr_dict = {'f': {'x': f_x_corr_loc, 'y': f_y_corr_loc, 'r': f_r_corr_loc}, 'v': {'x': vel_x_corr_loc, 'y': vel_y_corr_loc, 'r': vel_r_corr_loc}}
            vel_stat_dict = {'A': {'mag': typ0_avg}, 'B': {'mag': typ1_avg} }

            return vel_plot_dict, corr_dict, vel_stat_dict
    def adsorption(self):

        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        if self.lx_box >= self.ly_box:
            mem_max = np.max(self.pos[typ0ind,0])
            mem_min = np.min(self.pos[typ0ind,0])
        else:
            mem_max = np.max(self.pos[typ0ind,1])
            mem_min = np.min(self.pos[typ0ind,1])

        print(mem_min)
        print(mem_max)

        right_ads = np.where((self.pos[typ1ind,0]<(mem_max + 10.0)) & (self.pos[typ1ind,0]>0))
        left_ads = np.where((self.pos[typ1ind,0]>(mem_min - 10.0)) & (self.pos[typ1ind,0]<0))
        print(self.hx_box)
        print(right_ads)
        print(left_ads)
        return mem_min
    def adsorption3(partPhase_time, clust_time):
        start_part_phase = partPhase_time[:,0]
        start_bulk_id = np.where(partPhase_time[0,:]==0)[0]
        start_gas_id = np.where(partPhase_time[0,:]==2)[0]
        start_int_id = np.where(partPhase_time[0,:]==1)[0]

        start_clust_id = np.where(clust_time[0,:]==1)[0]
        start_gas2_id = np.where(clust_time[0,:]==0)[0]

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

        for j in range(1, np.shape(partPhase_time)[0]):

            bulk_id = np.where(partPhase_time[j,:]==0)[0]
            gas_id = np.where(partPhase_time[j,:]==2)[0]
            int_id = np.where(partPhase_time[j,:]==1)[0]

            clust_id = np.where(in_clust_arr[j,:]==1)[0]
            gas2_id = np.where(in_clust_arr[j,:]==0)[0]

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
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, len(np.where(typ[clust_now_in_gas2[0].astype('int')]==0)[0]))
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, len(np.where(typ[clust_now_in_gas2[0].astype('int')]==1)[0]))
            else:
                num_clust_to_gas2 = np.append(num_clust_to_gas2, 0)
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, 0)
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, 0)

            if len(gas2_now_in_clust)>0:
                num_gas2_to_clust = np.append(num_gas2_to_clust, len(gas2_now_in_clust[0]))
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, len(np.where(typ[gas2_now_in_clust[0].astype('int')]==0)[0]))
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, len(np.where(typ[gas2_now_in_clust[0].astype('int')]==1)[0]))
            else:
                num_gas2_to_clust = np.append(num_gas2_to_clust, 0)
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, 0)
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, 0)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, len(bulk_now_in_gas_no_int[0]))
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, len(np.where(typ[bulk_now_in_gas_no_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, len(np.where(typ[bulk_now_in_gas_no_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, 0)
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, 0)
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, len(gas_now_in_bulk_no_int[0]))
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, len(np.where(typ[gas_now_in_bulk_no_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, len(np.where(typ[gas_now_in_bulk_no_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, 0)
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, 0)
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, 0)

            if len(bulk_now_in_int)>0:
                num_bulk_to_int = np.append(num_bulk_to_int, len(bulk_now_in_int[0]))

                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, len(np.where(typ[bulk_now_in_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, len(np.where(typ[bulk_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_int = np.append(num_bulk_to_int, 0)
                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, 0)
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, 0)

            if len(int_now_in_bulk)>0:
                num_int_to_bulk = np.append(num_int_to_bulk, len(int_now_in_bulk[0]))
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, len(np.where(typ[int_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, len(np.where(typ[int_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_int_to_bulk = np.append(num_int_to_bulk, 0)
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, 0)
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, 0)

            if len(gas_now_in_int)>0:
                num_gas_to_int = np.append(num_gas_to_int, len(gas_now_in_int[0]))
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, len(np.where(typ[gas_now_in_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, len(np.where(typ[gas_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_int = np.append(num_gas_to_int, 0)
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, 0)
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, 0)

            if len(int_now_in_gas)>0:
                num_int_to_gas = np.append(num_int_to_gas, len(int_now_in_gas[0]))
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, len(np.where(typ[int_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, len(np.where(typ[int_now_in_gas[0].astype('int')]==1)[0]))
            else:
                num_int_to_gas = np.append(num_int_to_gas, 0)
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, 0)
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk = np.append(num_gas_to_bulk, len(gas_now_in_bulk[0]))
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, len(np.where(typ[gas_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, len(np.where(typ[gas_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk = np.append(num_gas_to_bulk, 0)
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, 0)
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, 0)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas = np.append(num_bulk_to_gas, len(bulk_now_in_gas[0]))
                num_slow_bulk_to_gas = np.append(num_slow_bulk_to_gas, len(np.where(typ[bulk_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas = np.append(num_fast_bulk_to_gas, len(np.where(typ[bulk_now_in_gas[0].astype('int')]==1)[0]))
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
    def collision_rate(self):
        
        
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

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        #Compute cluster parameters using system_all neighbor list
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos[fast_not_clust_ids]))

        BB_nlist = system_B.query(self.f_box.wrap(self.pos[fast_not_clust_ids]), query_args).toNeighborList()
        BB_collision_num = len(BB_nlist) / 2.0
        """
        #Compute cluster parameters using system_all neighbor list
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos[slow_not_clust_ids]))

        AB_nlist = system_A.query(self.f_box.wrap(self.pos[fast_not_clust_ids]), query_args).toNeighborList()
        AB_collision_num = len(AB_nlist) / 2.0

        AA_nlist = system_A.query(self.f_box.wrap(self.pos[slow_not_clust_ids]), query_args).toNeighborList()
        AA_collision_num = len(AA_nlist) / 2.0
        """

        AA_collision_num = 0
        AB_collision_num = 0

        return {'AA': AA_collision_num, 'AB': AB_collision_num, 'BB': BB_collision_num}

    def adsorption_nlist(self):

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
    def nearest_neighbors_penetrate(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the number of neighbors of each
        type for each particle and averaged over all particles of each phase.

        Outputs:
        neigh_stat_dict: dictionary containing the mean and standard deviation of the
        number of neighbors of each type ('all', 'A', or 'B') for a reference particle of
        a given type ('all', 'A', or 'B'), averaged over all particles in each phase.

        ori_stat_dict: dictionary containing the mean and standard deviation of the
        orientational correlation between a reference particle of
        a given type ('all', 'A', or 'B') and neighbors of each type ('all', 'A', or 'B'),
        averaged over all particles in each phase.

        neigh_plot_dict: dictionary containing information on the number of nearest
        neighbors of each bulk and interface reference particle of each type ('all', 'A', or 'B').
        '''

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max = self.r_cut)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        # Locate potential neighbor particles by type in the dense phase
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))
        
        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_nlist = system_A.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        AB_nlist = system_A.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        BA_nlist = system_B.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        BB_nlist = system_B.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_neigh_ind = np.array([], dtype=int)
        AA_num_neigh = np.array([])
        AA_dot = np.array([])
        
        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in AA_nlist.query_point_indices:
                if i not in AA_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_num_neigh = np.append(AA_num_neigh, len(loc))
                    AA_neigh_ind = np.append(AA_neigh_ind, int(i))
                    AA_dot = np.append(AA_dot, np.sum(np.cos(ang_A[i]-ang_A[AA_nlist.point_indices[loc]])))
            else:
                #Save nearest neighbor information to array
                AA_num_neigh = np.append(AA_num_neigh, 0)
                AA_neigh_ind = np.append(AA_neigh_ind, int(i))
                AA_dot = np.append(AA_dot, 0)
        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_neigh_ind = np.array([], dtype=int)
        BA_num_neigh = np.array([])
        BA_dot = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in BA_nlist.query_point_indices:
                if i not in BA_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_num_neigh = np.append(BA_num_neigh, len(loc))
                    BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                    BA_dot = np.append(BA_dot, np.sum(np.cos(ang_A[i]-ang_B[BA_nlist.point_indices[loc]])))
            else:
                #Save nearest neighbor information to array
                BA_num_neigh = np.append(BA_num_neigh, 0)
                BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                BA_dot = np.append(BA_dot, 0)

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_neigh_ind = np.array([], dtype=int)
        AB_num_neigh = np.array([])
        AB_dot = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in AB_nlist.query_point_indices:
                if i not in AB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_num_neigh = np.append(AB_num_neigh, len(loc))
                    AB_dot = np.append(AB_dot, np.sum(np.cos(ang_B[i]-ang_A[AB_nlist.point_indices[loc]])))
                    AB_neigh_ind = np.append(AB_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AB_num_neigh = np.append(AB_num_neigh, 0)
                AB_neigh_ind = np.append(AB_neigh_ind, int(i))
                AB_dot = np.append(AB_dot, 0)

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_neigh_ind = np.array([], dtype=int)
        BB_num_neigh = np.array([])
        BB_dot = np.array([])

        #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in BB_nlist.query_point_indices:
                if i not in BB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BB_num_neigh = np.append(BB_num_neigh, len(loc))
                    BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                    BB_dot = np.append(BB_dot, np.sum(np.cos(ang_B[i]-ang_B[BB_nlist.point_indices[loc]])))
            else:
                #Save nearest neighbor information to array
                BB_num_neigh = np.append(BB_num_neigh, 0)
                BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                BB_dot = np.append(BB_dot, 0)

        
        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
        allB_num_neigh = AB_num_neigh + BB_num_neigh
        allB_dot = BB_dot + AB_dot

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
        allA_num_neigh = AA_num_neigh + BA_num_neigh
        allA_dot = AA_dot + BA_dot

        # Save neighbor and local orientational order to arrays for all B reference particles of the respective phase with all nearest neighbors
        Ball_num_neigh = np.append(BA_num_neigh, BB_num_neigh)
        Ball_dot = np.append(BA_dot, BB_dot)

        # Save neighbor and local orientational order to arrays for all A reference particles of the respective phase with all nearest neighbors
        Aall_num_neigh = np.append(AB_num_neigh, AA_num_neigh)
        Aall_dot = np.append(AB_dot, AA_dot)

        # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
        allall_num_neigh = np.append(allA_num_neigh, allB_num_neigh)
        allall_dot = np.append(allA_dot, allB_dot)
        allall_pos_x = np.append(pos_A[:,0], pos_B[:,0])
        allall_pos_y = np.append(pos_A[:,1], pos_B[:,1])

        # Average orientational order over all neighbors for each particle
        for i in range(0, len(allB_dot)):
            if allB_num_neigh[i]>0:
                allB_dot[i] = allB_dot[i]/allB_num_neigh[i]

        for i in range(0, len(allA_dot)):
            if allA_num_neigh[i]>0:
                allA_dot[i] = allA_dot[i]/allA_num_neigh[i]

        for i in range(0, len(allall_dot)):
            if allall_num_neigh[i]>0:
                allall_dot[i] = allall_dot[i]/allall_num_neigh[i]

        for i in range(0, len(Aall_dot)):
            if Aall_num_neigh[i]>0:
                Aall_dot[i] = Aall_dot[i]/Aall_num_neigh[i]

        for i in range(0, len(Ball_dot)):
            if Ball_num_neigh[i]>0:
                Ball_dot[i] = Ball_dot[i]/Ball_num_neigh[i]

        # Create output dictionary for statistical averages of total nearest neighbor numbers on each particle per phase/activity pairing
        neigh_stat_dict = {'all-all': {'mean': np.mean(allall_num_neigh), 'std': np.std(allall_num_neigh)}, 'all-A': {'mean': np.mean(allA_num_neigh), 'std': np.std(allA_num_neigh)}, 'all-B': {'mean': np.mean(allB_num_neigh), 'std': np.std(allB_num_neigh)}, 'A-A': {'mean': np.mean(AA_num_neigh), 'std': np.std(AA_num_neigh)}, 'A-B': {'mean': np.mean(AB_num_neigh), 'std': np.std(AB_num_neigh)}, 'B-B': {'mean': np.mean(BB_num_neigh), 'std': np.std(BB_num_neigh)}}


        # Create output dictionary for statistical averages of total nearest neighbor orientational correlation on each particle per phase/activity pairing
        ori_stat_dict = {'all-all': {'mean': np.mean(allall_dot), 'std': np.std(allall_dot)}, 'all-A': {'mean': np.mean(allA_dot), 'std': np.std(allA_dot)}, 'all-B': {'mean': np.mean(allB_dot), 'std': np.std(allB_dot)}, 'A-A': {'mean': np.mean(AA_dot), 'std': np.std(AA_dot)}, 'A-B': {'mean': np.mean(AB_dot), 'std': np.std(AB_dot)}, 'B-B': {'mean': np.mean(BB_dot), 'std': np.std(BB_dot)}}

        # Create output dictionary for plotting of nearest neighbor information of each particle per phase/activity pairing and their respective x-y locations
        neigh_plot_dict = {'all-all': {'neigh': allall_num_neigh, 'ori': allall_dot, 'x': allall_pos_x, 'y': allall_pos_y}, 'all-A': {'neigh': allA_num_neigh, 'ori': allA_dot, 'x': pos_A[:,0], 'y': pos_A[:,1]}, 'all-B': {'neigh': allB_num_neigh, 'ori': allB_dot, 'x': pos_B[:,0], 'y': pos_B[:,1]}, 'A-all': {'neigh': Aall_num_neigh, 'ori': Aall_dot, 'x': self.pos[:,0], 'y': self.pos[:,1]}, 'B-all': {'neigh': Ball_num_neigh, 'ori': Ball_dot, 'x': self.pos[:,0], 'y': self.pos[:,1]}}

        
        return neigh_stat_dict, ori_stat_dict, neigh_plot_dict

    def penetration_depth(self, start_dict, pos_prev, vertical_shift, dify_long):

        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        wall_x = np.amax(self.pos[typ0ind,0])

        if ((self.pos[typ1ind,0] <= (wall_x - 1.0)) & (self.pos[typ1ind,0] >= -(wall_x - 1.0))) | ((pos_prev[typ1ind,0] <= (wall_x - 1.0)) & (pos_prev[typ1ind,0] >= -(wall_x - 1.0))):
            if ((pos_prev[typ1ind,0] > (wall_x - 1.0)) | (pos_prev[typ1ind,0] < -(wall_x - 1.0))) & ((self.pos[typ1ind,0] <= (wall_x - 1.0)) & (self.pos[typ1ind,0] >= -(wall_x - 1.0))):
                action = 'enter'
                start_x = self.pos[typ1ind,0]
                start_y = self.pos[typ1ind,1]

            elif (pos_prev[typ1ind,0] <= (wall_x - 1.0)) & (pos_prev[typ1ind,0] >= -(wall_x - 1.0)) & ((self.pos[typ1ind,0] > (wall_x - 1.0)) & (self.pos[typ1ind,0] < -(wall_x - 1.0))): 
                action = 'exit'
                start_x = start_dict['x']
                start_y = start_dict['y']
            else:
                action = 'bulk'  
                start_x = start_dict['x']
                start_y = start_dict['y']              

            if np.abs(self.pos[typ1ind,1]-pos_prev[typ1ind,1])>self.hy_box:
                if (self.pos[typ1ind,1]<0) & (pos_prev[typ1ind,1]>0):
                    vertical_shift = dify_long + self.hy_box-start_y
                    dify_long = vertical_shift
                    start_y = -self.hy_box
                elif (self.pos[typ1ind,1]>0) & (pos_prev[typ1ind,1]<0):
                    vertical_shift = dify_long + -self.hy_box-start_y
                    dify_long = vertical_shift
                    start_y = self.hy_box
                             
            if start_x < 0:
                penetration_depth = wall_x+self.pos[typ1ind,0]
                difx = wall_x + self.pos[typ1ind,0]
            else:
                penetration_depth = wall_x-self.pos[typ1ind,0]
                difx = self.pos[typ1ind,0]  - wall_x

            #dify =  (self.pos[typ1ind,1] + vertical_shift) - start_y
            dify =  dify_long + (self.pos[typ1ind,1]- start_y)
            
            difr = (difx ** 2 + dify **2 ) ** 0.5 
            

            difx_prev = self.utility_functs.sep_dist_x(self.pos[typ1ind,0], pos_prev[typ1ind,0])
            dify_prev = self.utility_functs.sep_dist_y(self.pos[typ1ind,1], pos_prev[typ1ind,1])
            difr_prev = (difx_prev ** 2 + dify_prev ** 2 ) ** 0.5
            
            
            MSD = ( difx ** 2 + dify ** 2 )
        else:
            action = 'gas'
            MSD = 0
            dify = 0
            difx_prev = 0
            dify_prev = 0
            difr_prev = 0
            difx = 0
            difr = 0
            penetration_depth = 0
            start_x = 0
            start_y = 0
            vertical_shift = 0
            dify_long = 0

            difx_prev = self.utility_functs.sep_dist_x(self.pos[typ1ind,0], pos_prev[typ1ind,0])
            dify_prev = self.utility_functs.sep_dist_y(self.pos[typ1ind,1], pos_prev[typ1ind,1])
            difr_prev = (difx_prev ** 2 + dify_prev ** 2 ) ** 0.5

        penetration_dict = {'wall_x': wall_x, 'displace': {'x': difx_prev, 'y': dify_prev, 'r': difr_prev}, 'total_displace':{'x': difx, 'y': dify, 'r': difr, 'MSD': MSD, 'depth': penetration_depth}, 'action': action}
        start_dict = {'x': start_x, 'y': start_y}

        return penetration_dict, start_dict, vertical_shift, dify_long
    def interparticle_pressure_nlist(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the total interparticle stress all,
        interacting neighbors acting on each reference particle.

        Outputs:
        stress_stat_dict: dictionary containing the total interparticle stress
        between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B')
        in each direction ('XX', 'XY', 'YX', 'YY'), averaged over all reference particles in each phase.

        press_stat_dict: dictionary containing the average interparticle pressure and
        shear stress between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B'),
        averaged over all reference particles in each phase.

        press_plot_dict: dictionary containing information on the interparticle stress
        and pressure of each bulk and interface reference particle of each type
        ('all', 'A', or 'B').
        '''

        
        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]

        # Calculate area of each phase
        if self.lx_box > self.ly_box:
            bulk_area = 2 * np.amax(pos_A[:,0]) * self.ly_box
        else:
            bulk_area = 2 * np.amax(pos_A[:,1]) * self.lx_box

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        # Locate potential neighbor particles by type in the dense phase
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_nlist = system_A.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        AB_nlist = system_A.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        BA_nlist = system_B.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        BB_nlist = system_B.query(self.f_box.wrap(pos_B), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type A bulk particles
        AA_neigh_ind = np.array([], dtype=int)
        AA_num_neigh = np.array([])

        SigXX_AA_part=np.zeros(len(pos_A))        #Sum of normal stress in x direction
        SigXY_AA_part=np.zeros(len(pos_A))        #Sum of tangential stress in x-y direction
        SigYX_AA_part=np.zeros(len(pos_A))        #Sum of tangential stress in y-x direction
        SigYY_AA_part=np.zeros(len(pos_A))        #Sum of normal stress in y direction

        SigXX_AA_part_num=np.zeros(len(pos_A))    #Number of interparticle stresses summed in normal x direction
        SigXY_AA_part_num=np.zeros(len(pos_A))    #Number of interparticle stresses summed in tangential x-y direction
        SigYX_AA_part_num=np.zeros(len(pos_A))    #Number of interparticle stresses summed in tangential y-x direction
        SigYY_AA_part_num=np.zeros(len(pos_A))    #Number of interparticle stresses summed in normal y direction

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A)):
            if i in AA_nlist.query_point_indices:
                if i not in AA_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AA_nlist.point_indices[loc]], difxy=True)

                        #Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_part[i] += np.sum(SigXX)
                        SigYY_AA_part[i] += np.sum(SigYY)
                        SigXY_AA_part[i] += np.sum(SigXY)
                        SigYX_AA_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_part_num[i] += len(SigXX)
                        SigYY_AA_part_num[i] += len(SigYY)
                        SigXY_AA_part_num[i] += len(SigXY)
                        SigYX_AA_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A[i][0], pos_A[AA_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A[i][1], pos_A[AA_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle force
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_part[i] = SigXX
                        SigYY_AA_part[i] = SigYY
                        SigXY_AA_part[i] = SigXY
                        SigYX_AA_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_part_num[i] += 1
                        SigYY_AA_part_num[i] += 1
                        SigXY_AA_part_num[i] += 1
                        SigYX_AA_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_num_neigh = np.append(AA_num_neigh, len(loc))
                    AA_neigh_ind = np.append(AA_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                AA_num_neigh = np.append(AA_num_neigh, 0)
                AA_neigh_ind = np.append(AA_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type A bulk particles
        BA_neigh_ind = np.array([], dtype=int)
        BA_num_neigh = np.array([])

        SigXX_BA_part=np.zeros(len(pos_A))
        SigXY_BA_part=np.zeros(len(pos_A))
        SigYX_BA_part=np.zeros(len(pos_A))
        SigYY_BA_part=np.zeros(len(pos_A))

        SigXX_BA_part_num=np.zeros(len(pos_A))
        SigXY_BA_part_num=np.zeros(len(pos_A))
        SigYX_BA_part_num=np.zeros(len(pos_A))
        SigYY_BA_part_num=np.zeros(len(pos_A))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A)):
            if i in BA_nlist.query_point_indices:
                if i not in BA_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BA_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_part[i] += np.sum(SigXX)
                        SigYY_BA_part[i] += np.sum(SigYY)
                        SigXY_BA_part[i] += np.sum(SigXY)
                        SigYX_BA_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_part_num[i] += len(SigXX)
                        SigYY_BA_part_num[i] += len(SigYY)
                        SigXY_BA_part_num[i] += len(SigXY)
                        SigYX_BA_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A[i][0], pos_B[BA_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A[i][1], pos_B[BA_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_part[i] = SigXX
                        SigYY_BA_part[i] = SigYY
                        SigXY_BA_part[i] = SigXY
                        SigYX_BA_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_part_num[i] += 1
                        SigYY_BA_part_num[i] += 1
                        SigXY_BA_part_num[i] += 1
                        SigYX_BA_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_num_neigh = np.append(BA_num_neigh, len(loc))
                    BA_neigh_ind = np.append(BA_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BA_num_neigh = np.append(BA_num_neigh, 0)
                BA_neigh_ind = np.append(BA_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type B bulk particles
        AB_neigh_ind = np.array([], dtype=int)
        AB_num_neigh = np.array([])

        SigXX_AB_part=np.zeros(len(pos_B))
        SigXY_AB_part=np.zeros(len(pos_B))
        SigYX_AB_part=np.zeros(len(pos_B))
        SigYY_AB_part=np.zeros(len(pos_B))

        SigXX_AB_part_num=np.zeros(len(pos_B))
        SigXY_AB_part_num=np.zeros(len(pos_B))
        SigYX_AB_part_num=np.zeros(len(pos_B))
        SigYY_AB_part_num=np.zeros(len(pos_B))


        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B)):
            if i in AB_nlist.query_point_indices:
                if i not in AB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AB_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_part[i] += np.sum(SigXX)
                        SigYY_AB_part[i] += np.sum(SigYY)
                        SigXY_AB_part[i] += np.sum(SigXY)
                        SigYX_AB_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_part_num[i] += len(SigXX)
                        SigYY_AB_part_num[i] += len(SigYY)
                        SigXY_AB_part_num[i] += len(SigXY)
                        SigYX_AB_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B[i][0], pos_A[AB_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B[i][1], pos_A[AB_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_part[i] += SigXX
                        SigYY_AB_part[i] += SigYY
                        SigXY_AB_part[i] += SigXY
                        SigYX_AB_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_part_num[i] += 1
                        SigYY_AB_part_num[i] += 1
                        SigXY_AB_part_num[i] += 1
                        SigYX_AB_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_num_neigh = np.append(AB_num_neigh, len(loc))
                    AB_neigh_ind = np.append(AB_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_num_neigh = np.append(AB_num_neigh, 0)
                AB_neigh_ind = np.append(AB_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type B bulk particles
        BB_neigh_ind = np.array([], dtype=int)
        BB_num_neigh = np.array([])

        SigXX_BB_part=np.zeros(len(pos_B))
        SigXY_BB_part=np.zeros(len(pos_B))
        SigYX_BB_part=np.zeros(len(pos_B))
        SigYY_BB_part=np.zeros(len(pos_B))

        SigXX_BB_part_num=np.zeros(len(pos_B))
        SigXY_BB_part_num=np.zeros(len(pos_B))
        SigYX_BB_part_num=np.zeros(len(pos_B))
        SigYY_BB_part_num=np.zeros(len(pos_B))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B)):
            if i in BB_nlist.query_point_indices:
                if i not in BB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BB_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_part[i] += np.sum(SigXX)
                        SigYY_BB_part[i] += np.sum(SigYY)
                        SigXY_BB_part[i] += np.sum(SigXY)
                        SigYX_BB_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_part_num[i] += len(SigXX)
                        SigYY_BB_part_num[i] += len(SigYY)
                        SigXY_BB_part_num[i] += len(SigXY)
                        SigYX_BB_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist_x(pos_B[i][0], pos_B[BB_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B[i][1], pos_B[BB_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_part[i] += SigXX
                        SigYY_BB_part[i] += SigYY
                        SigXY_BB_part[i] += SigXY
                        SigYX_BB_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_part_num[i] += 1
                        SigYY_BB_part_num[i] += 1
                        SigXY_BB_part_num[i] += 1
                        SigYX_BB_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_num_neigh = np.append(BB_num_neigh, len(loc))
                    BB_neigh_ind = np.append(BB_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BB_num_neigh = np.append(BB_num_neigh, 0)
                BB_neigh_ind = np.append(BB_neigh_ind, int(i))

        ###Bulk stress

        # Calculate total stress and number of neighbor pairs summed over for B bulk reference particles and all dense neighbors
        allB_SigXX_part = SigXX_BB_part + SigXX_AB_part
        allB_SigXX_part_num = SigXX_BB_part_num + SigXX_AB_part_num
        allB_SigXY_part = SigXY_BB_part + SigXY_AB_part
        allB_SigXY_part_num = SigXY_BB_part_num + SigXY_AB_part_num
        allB_SigYX_part = SigYX_BB_part + SigYX_AB_part
        allB_SigYX_part_num = SigYX_BB_part_num + SigYX_AB_part_num
        allB_SigYY_part = SigYY_BB_part + SigYY_AB_part
        allB_SigYY_part_num = SigYY_BB_part_num + SigYY_AB_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and B dense neighbors
        Ball_SigXX_part = np.append(SigXX_BA_part, SigXX_BB_part)
        Ball_SigXX_part_num = np.append(SigXX_BA_part_num, SigXX_BB_part_num)
        Ball_SigXY_part = np.append(SigXY_BA_part, SigXY_BB_part)
        Ball_SigXY_part_num = np.append(SigXY_BA_part_num, SigXY_BB_part_num)
        Ball_SigYX_part = np.append(SigYX_BA_part, SigYX_BB_part)
        Ball_SigYX_part_num = np.append(SigYX_BA_part_num, SigYX_BB_part_num)
        Ball_SigYY_part = np.append(SigYY_BA_part, SigYY_BB_part)
        Ball_SigYY_part_num = np.append(SigYY_BA_part_num, SigYY_BB_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A bulk reference particles and all dense neighbors
        allA_SigXX_part = SigXX_AA_part + SigXX_BA_part
        allA_SigXX_part_num = SigXX_AA_part_num + SigXX_BA_part_num
        allA_SigXY_part = SigXY_AA_part + SigXY_BA_part
        allA_SigXY_part_num = SigXY_AA_part_num + SigXY_BA_part_num
        allA_SigYX_part = SigYX_AA_part + SigYX_BA_part
        allA_SigYX_part_num = SigYX_AA_part_num + SigYX_BA_part_num
        allA_SigYY_part = SigYY_AA_part + SigYY_BA_part
        allA_SigYY_part_num = SigYY_AA_part_num + SigYY_BA_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and A dense neighbors
        Aall_SigXX_part = np.append(SigXX_AB_part, SigXX_AA_part)
        Aall_SigXX_part_num = np.append(SigXX_AB_part_num, SigXX_AA_part_num)
        Aall_SigXY_part = np.append(SigXY_AB_part, SigXY_AA_part)
        Aall_SigXY_part_num = np.append(SigXY_AB_part_num, SigXY_AA_part_num)
        Aall_SigYX_part = np.append(SigYX_AB_part, SigYX_AA_part)
        Aall_SigYX_part_num = np.append(SigYX_AB_part_num, SigYX_AA_part_num)
        Aall_SigYY_part = np.append(SigYY_AB_part, SigYY_AA_part)
        Aall_SigYY_part_num = np.append(SigYY_AB_part_num, SigYY_AA_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and all dense neighbors
        allall_SigXX_part = np.append(allA_SigXX_part, allB_SigXX_part)
        allall_SigXX_part_num = np.append(allA_SigXX_part_num, allB_SigXX_part_num)
        allall_SigXY_part = np.append(allA_SigXY_part, allB_SigXY_part)
        allall_SigXY_part_num = np.append(allA_SigXY_part_num, allB_SigXY_part_num)
        allall_SigYX_part = np.append(allA_SigYX_part, allB_SigYX_part)
        allall_SigYX_part_num = np.append(allA_SigYX_part_num, allB_SigYX_part_num)
        allall_SigYY_part = np.append(allA_SigYY_part, allB_SigYY_part)
        allall_SigYY_part_num = np.append(allA_SigYY_part_num, allB_SigYY_part_num)

        ###Interparticle pressure

        # Calculate total interparticle pressure experienced by all particles in each phase
        allall_int_press = np.sum(allall_SigXX_part + allall_SigYY_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all particles in each phase from all A particles
        allA_int_press = np.sum(allA_SigXX_part + allA_SigYY_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all A particles in each phase
        Aall_int_press = np.sum(Aall_SigXX_part + Aall_SigYY_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all particles in each phase from all B particles
        allB_int_press = np.sum(allB_SigXX_part + allB_SigYY_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all B particles in each phase
        Ball_int_press = np.sum(Ball_SigXX_part + Ball_SigYY_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all A particles
        AA_int_press = np.sum(SigXX_AA_part + SigYY_AA_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all B particles
        AB_int_press = np.sum(SigXX_AB_part + SigYY_AB_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all A particles
        BA_int_press = np.sum(SigXX_BA_part + SigYY_BA_part)/(2*bulk_area)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all B particles
        BB_int_press = np.sum(SigXX_BB_part + SigYY_BB_part)/(2*bulk_area)

        # Calculate total shear stress experienced by all particles in each phase from all particles
        allall_shear_stress = np.sum(allall_SigXY_part)/(bulk_area)

        # Calculate total shear stress experienced by all particles in each phase from A particles
        allA_shear_stress = np.sum(allA_SigXY_part)/(bulk_area)

        # Calculate total shear stress experienced by all A particles in each phase from all particles
        Aall_shear_stress = np.sum(Aall_SigXY_part)/(bulk_area)

        # Calculate total shear stress experienced by all particles in each phase from B particles
        allB_shear_stress = np.sum(allB_SigXY_part)/(bulk_area)

        # Calculate total shear stress experienced by all B particles in each phase from all particles
        Ball_shear_stress = np.sum(Ball_SigXY_part)/(bulk_area)

        # Calculate total shear stress experienced by all A particles in each phase from all A particles
        AA_shear_stress = np.sum(SigXY_AA_part)/(bulk_area)

        # Calculate total shear stress experienced by all A particles in each phase from all B particles
        AB_shear_stress = np.sum(SigXY_AB_part)/(bulk_area)

        # Calculate total shear stress experienced by all B particles in each phase from all A particles
        BA_shear_stress = np.sum(SigXY_BA_part)/(bulk_area)

        # Calculate total shear stress experienced by all B particles in each phase from all B particles
        BB_shear_stress = np.sum(SigXY_BB_part)/(bulk_area)

        # Make position arrays for plotting total stress on each particle for various activity pairings and phases
        allall_pos_x = np.append(pos_A[:,0], pos_B[:,0])
        allall_pos_y = np.append(pos_A[:,1], pos_B[:,1])

        allall_part_int_press = (allall_SigXX_part + allall_SigYY_part)/(2)
        allall_part_shear_stress = (allall_SigXY_part)

        allA_part_int_press = (allA_SigXX_part + allA_SigYY_part)/(2)
        allA_part_shear_stress = (allA_SigXY_part)

        allB_part_int_press = (allB_SigXX_part + allB_SigYY_part)/(2)
        allB_part_shear_stress = (allB_SigXY_part)

        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        stress_stat_dict = {'all-all': {'XX': np.sum(allall_SigXX_part), 'XY': np.sum(allall_SigXY_part), 'YX': np.sum(allall_SigYX_part), 'YY': np.sum(allall_SigYY_part)}, 'all-A': {'XX': np.sum(allA_SigXX_part), 'XY': np.sum(allA_SigXY_part), 'YX': np.sum(allA_SigYX_part), 'YY': np.sum(allA_SigYY_part)}, 'all-B': {'XX': np.sum(allB_SigXX_part), 'XY': np.sum(allB_SigXY_part), 'YX': np.sum(allB_SigYX_part), 'YY': np.sum(allB_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_part), 'XY': np.sum(SigXY_AA_part), 'YX': np.sum(SigYX_AA_part), 'YY': np.sum(SigYY_AA_part)}, 'A-B': {'XX': np.sum(SigXX_AB_part), 'XY': np.sum(SigXY_AB_part), 'YX': np.sum(SigYX_AB_part), 'YY': np.sum(SigYY_AB_part)}, 'B-B': {'XX': np.sum(SigXX_BB_part), 'XY': np.sum(SigXY_BB_part), 'YX': np.sum(SigYX_BB_part), 'YY': np.sum(SigYY_BB_part)}}

        # Create output dictionary for statistical averages of total pressure and shear stress on each particle per phase/activity pairing
        press_stat_dict = {'all-all': {'press': allall_int_press, 'shear': allall_shear_stress}, 'all-A': {'press': allA_int_press, 'shear': allA_shear_stress}, 'all-B': {'press': allB_int_press, 'shear': allB_shear_stress}, 'A-A': {'press': AA_int_press, 'shear': AA_shear_stress}, 'A-B': {'press': AB_int_press, 'shear': AB_shear_stress}, 'B-B': {'press': BB_int_press, 'shear': BB_shear_stress}}

        # Create output dictionary for plotting of total stress/pressure on each particle per phase/activity pairing and their respective x-y locations
        press_plot_dict = {'all-all': {'press': allall_part_int_press, 'shear': allall_part_shear_stress, 'x': allall_pos_x, 'y': allall_pos_y}, 'all-A': {'press': allA_part_int_press, 'shear': allA_part_shear_stress, 'x': pos_A[:,0], 'y': pos_A[:,1]}, 'all-B': {'press': allB_part_int_press, 'shear': allB_part_shear_stress, 'x': pos_B[:,0], 'y': pos_B[:,1]}}

        return stress_stat_dict, press_stat_dict, press_plot_dict
