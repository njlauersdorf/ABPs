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
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory, utility

class particle_props:

    def __init__(self, l_box, partNum, NBins, peA, peB, typ, pos, ang):
        theory_functs = theory.theory()
        self.l_box = l_box
        self.h_box = self.l_box / 2
        utility_functs = utility.utility(self.l_box)
        self.partNum = partNum
        self.min_size = int(self.partNum / 8)
        try:
            self.NBins = int(NBins)
        except:
            print('NBins must be either a float or an integer')
        self.sizeBin = utility_functs.roundUp(self.l_box / self.NBins, 6)
        self.peA = peA
        self.peB = peB
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        self.typ = typ

        self.utility_functs = utility.utility(self.l_box)

        self.pos = pos
        self.ang = ang

    def particle_phase_ids(self, phasePart):
        A_bulk_id = np.where((phasePart==0) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_bulk_id = np.where((phasePart==0) & (self.typ==1))[0]        #Bulk phase structure(s)
        bulk_id = np.where(phasePart==0)[0]        #Bulk phase structure(s)

        A_int_id = np.where((phasePart==1) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_int_id = np.where((phasePart==1) & (self.typ==1))[0]        #Bulk phase structure(s)
        int_id = np.where(phasePart==1)[0]         #All interfaces

        A_gas_id = np.where((phasePart==2) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_gas_id = np.where((phasePart==2) & (self.typ==1))[0]        #Bulk phase structure(s)
        gas_id = np.where(phasePart==2)[0]         #All interfaces

        A_dense_id = np.where((phasePart!=2) & (self.typ==0))[0]
        B_dense_id = np.where((phasePart!=2) & (self.typ==1))[0]
        dense_id = np.where(phasePart!=2)[0]

        A_gas_int_id = np.where((phasePart!=0) & (self.typ==0))[0]
        B_gas_int_id = np.where((phasePart!=0) & (self.typ==1))[0]
        gas_int_id = np.where(phasePart!=0)[0]

        phase_part_dict = {'bulk': {'all': bulk_id, 'A': A_bulk_id, 'B': B_bulk_id}, 'int': {'all': int_id, 'A': A_int_id, 'B': B_int_id}, 'gas': {'all': gas_id, 'A': A_gas_id, 'B': B_gas_id}, 'dense': {'all': dense_id, 'A': A_dense_id, 'B': B_dense_id}, 'gas_int': {'all': gas_int_id, 'A': A_gas_int_id, 'B': B_gas_int_id}}
        return phase_part_dict

    def particle_normal_fa(self):

        part_normal_fa = np.zeros(self.partNum)

        for h in range(0, len(self.pos)):

            #Calculate position of exterior edge bin
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in bin's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
            difr= ( (difx )**2 + (dify)**2)**0.5

            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #If particle is of type A, add alignment with nearest surface's normal for average calculation
            if self.typ[h]==0:
                fa_r[h]=r_dot_p*self.peA

            #If particle is of type B, add alignment with nearest surface's normal for average calculation
            elif self.typ[h]==1:
                part_normal_fa[h]=r_dot_p*self.peB

        return part_normal_fa

    def particle_align(self):

        part_align = np.zeros(self.partNum)

        for h in range(0, len(self.pos)):

            #Calculate position of exterior edge bin
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in bin's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
            difr= ( (difx )**2 + (dify)**2)**0.5

            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #Add alignment with CoM for average calculation of all particles
            part_align[h]=r_dot_p

        part_align_dict = part_align
        return part_align_dict

    def particle_sep_dist(self):

        part_difr = np.zeros(self.partNum)

        for h in range(0, len(self.pos)):

            #Calculate position of exterior edge bin
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in bin's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
            difr= ( (difx )**2 + (dify)**2)**0.5

            part_difr[h] = difr

        return part_difr
    def radial_normal_fa(self):

        align_norm = np.array([])
        fa_norm = np.array([])
        r_dist_norm = np.array([])
        for h in range(0, len(self.pos)):

            #x-distance from CoM
            difx = self.pos[h,0] - 0

            #y-distance from CoM
            dify = self.pos[h,1] - 0

            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
            difr= ( (difx )**2 + (dify)**2)**0.5

            x_norm_unitv = (difx) / difr
            y_norm_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = np.sin(self.ang[h])
            py = -np.cos(self.ang[h])

            #Calculate alignment towards CoM
            r_dot_p = (-x_norm_unitv * px) + (-y_norm_unitv * py)

            #Add alignment with CoM for average calculation of all particles
            align_norm=np.append(align_norm, r_dot_p)

            #If particle is of type A, add alignment with nearest surface's normal for average calculation
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, r_dot_p*self.peA)
            else:
                fa_norm=np.append(fa_norm, r_dot_p*self.peB)
            r_dist_norm = np.append(r_dist_norm, difr)

        radial_fa_dict = {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}
        return radial_fa_dict

    def radial_surface_normal_fa(self, method2_align_dict):

        part_align = method2_align_dict['part']['align']

        fa_norm = np.array([])
        r_dist_norm = np.array([])
        for h in range(0, len(self.pos)):

            #x-distance from CoM
            difx = self.pos[h,0] - 0

            #y-distance from CoM
            dify = self.pos[h,1] - 0

            #Very large initial distance to calculate closest interior edge bin to this exterior edge bin
            difr= ( (difx )**2 + (dify)**2)**0.5

            #If particle is of type A, add alignment with nearest surface's normal for average calculation
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, part_align[h]*self.peA)
            else:
                fa_norm=np.append(fa_norm, part_align[h]*self.peB)

            r_dist_norm = np.append(r_dist_norm, difr)
        radial_fa_dict = {'r': r_dist_norm, 'fa': fa_norm, 'align': part_align}
        return radial_fa_dict

    def angular_velocity(self, ang_vel, phasePart):

        bulk_ang_vel = 0
        bulk_A_ang_vel = 0
        bulk_B_ang_vel = 0
        bulk_ang_num = 0
        bulk_A_ang_num = 0
        bulk_B_ang_num = 0

        int_ang_vel = 0
        int_A_ang_vel = 0
        int_B_ang_vel = 0
        int_ang_num = 0
        int_A_ang_num = 0
        int_B_ang_num = 0

        gas_ang_vel = 0
        gas_A_ang_vel = 0
        gas_B_ang_vel = 0
        gas_ang_num = 0
        gas_A_ang_num = 0
        gas_B_ang_num = 0

        for h in range(0, self.partNum):

            if phasePart[h] == 0:
                bulk_ang_vel += np.abs(ang_vel[h])
                bulk_ang_num += 1

                if self.typ[h] == 0:
                    bulk_A_ang_vel += np.abs(ang_vel[h])
                    bulk_A_ang_num += 1

                elif self.typ[h] == 1:
                    bulk_B_ang_vel += np.abs(ang_vel[h])
                    bulk_B_ang_num += 1

            elif phasePart[h] == 1:
                int_ang_vel += np.abs(ang_vel[h])
                int_ang_num += 1

                if self.typ[h] == 0:
                    int_A_ang_vel += np.abs(ang_vel[h])
                    int_A_ang_num += 1

                elif self.typ[h] == 1:
                    int_B_ang_vel += np.abs(ang_vel[h])
                    int_B_ang_num += 1

            elif phasePart[h] == 2:
                gas_ang_vel += np.abs(ang_vel[h])
                gas_ang_num += 1

                if self.typ[h] == 0:
                    gas_A_ang_vel += np.abs(ang_vel[h])
                    gas_A_ang_num += 1

                elif self.typ[h] == 1:
                    gas_B_ang_vel += np.abs(ang_vel[h])
                    gas_B_ang_num += 1

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

        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        for h in range(0, self.partNum):

            if phasePart[h] == 0:

                bulk_all_val += (np.abs(ang_vel[h])-bulk_ang_vel_avg)**2

                if self.typ[h]==0:
                    bulk_A_val += (np.abs(ang_vel[h])-bulk_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_A_val += (np.abs(ang_vel[h])-bulk_B_ang_vel_avg)**2

            elif phasePart[h] == 1:

                int_all_val += (np.abs(ang_vel[h])-int_ang_vel_avg)**2

                if self.typ[h]==0:
                    int_A_val += (np.abs(ang_vel[h])-int_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    int_A_val += (np.abs(ang_vel[h])-int_B_ang_vel_avg)**2

            elif phasePart[h] == 2:

                gas_all_val += (np.abs(ang_vel[h])-gas_ang_vel_avg)**2

                if self.typ[h]==0:
                    gas_A_val += (np.abs(ang_vel[h])-gas_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    gas_A_val += (np.abs(ang_vel[h])-gas_B_ang_vel_avg)**2

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

        ang_vel_phase_dict = {'bulk': {'all': {'avg': bulk_ang_vel_avg, 'std': bulk_ang_vel_std}, 'A': {'avg': bulk_A_ang_vel_avg, 'std': bulk_A_ang_vel_std}, 'B': {'avg': bulk_B_ang_vel_avg, 'std': bulk_B_ang_vel_std} }, 'int': {'all': {'avg': int_ang_vel_avg, 'std': int_ang_vel_std}, 'A': {'avg': int_A_ang_vel_avg, 'std': int_A_ang_vel_std}, 'B': {'avg': int_B_ang_vel_avg, 'std': int_B_ang_vel_std}}, 'gas': {'all': {'avg': gas_ang_vel_avg, 'std': gas_ang_vel_std}, 'A': {'avg': gas_A_ang_vel_avg, 'std': gas_A_ang_vel_std}, 'B': {'avg': gas_B_ang_vel_avg, 'std': gas_B_ang_vel_std}}}

        return ang_vel_phase_dict

    def velocity(self, vel, phasePart):

        bulk_vel = 0
        bulk_A_vel = 0
        bulk_B_vel = 0
        bulk_num = 0
        bulk_A_num = 0
        bulk_B_num = 0

        int_vel = 0
        int_A_vel = 0
        int_B_vel = 0
        int_num = 0
        int_A_num = 0
        int_B_num = 0

        gas_vel = 0
        gas_A_vel = 0
        gas_B_vel = 0
        gas_num = 0
        gas_A_num = 0
        gas_B_num = 0

        for h in range(0, self.partNum):

            if phasePart[h] == 0:
                bulk_vel += vel[h]
                bulk_num += 1

                if self.typ[h] == 0:
                    bulk_A_vel += vel[h]
                    bulk_A_num += 1

                elif self.typ[h] == 1:
                    bulk_B_vel += vel[h]
                    bulk_B_num += 1

            elif phasePart[h] == 1:
                int_vel += vel[h]
                int_num += 1

                if self.typ[h] == 0:
                    int_A_vel += vel[h]
                    int_A_num += 1

                elif self.typ[h] == 1:
                    int_B_vel += vel[h]
                    int_B_num += 1

            elif phasePart[h] == 2:
                gas_vel += vel[h]
                gas_num += 1

                if self.typ[h] == 0:
                    gas_A_vel += vel[h]
                    gas_A_num += 1

                elif self.typ[h] == 1:
                    gas_B_vel += vel[h]
                    gas_B_num += 1

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

        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        for h in range(0, self.partNum):

            if phasePart[h] == 0:

                bulk_all_val += (np.abs(vel[h])-bulk_vel_avg)**2

                if self.typ[h]==0:
                    bulk_A_val += (np.abs(vel[h])-bulk_A_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_A_val += (np.abs(vel[h])-bulk_B_vel_avg)**2

            elif phasePart[h] == 1:

                int_all_val += (np.abs(vel[h])-int_vel_avg)**2

                if self.typ[h]==0:
                    int_A_val += (np.abs(vel[h])-int_A_vel_avg)**2
                elif self.typ[h]==1:
                    int_A_val += (np.abs(vel[h])-int_B_vel_avg)**2

            elif phasePart[h] == 2:

                gas_all_val += (np.abs(vel[h])-gas_vel_avg)**2

                if self.typ[h]==0:
                    gas_A_val += (np.abs(vel[h])-gas_A_vel_avg)**2
                elif self.typ[h]==1:
                    gas_A_val += (np.abs(vel[h])-gas_B_vel_avg)**2

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

        vel_phase_dict = {'bulk': {'all': {'avg': bulk_vel_avg, 'std': bulk_vel_std}, 'A': {'avg': bulk_A_vel_avg, 'std': bulk_A_vel_std}, 'B': {'avg': bulk_B_vel_avg, 'std': bulk_B_vel_std} }, 'int': {'all': {'avg': int_vel_avg, 'std': int_vel_std}, 'A': {'avg': int_A_vel_avg, 'std': int_A_vel_std}, 'B': {'avg': int_B_vel_avg, 'std': int_B_vel_std}}, 'gas': {'all': {'avg': gas_vel_avg, 'std': gas_vel_std}, 'A': {'avg': gas_A_vel_avg, 'std': gas_A_vel_std}, 'B': {'avg': gas_B_vel_avg, 'std': gas_B_vel_std}}}

        return vel_phase_dict
