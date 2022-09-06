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

    def __init__(self, l_box, partNum, NBins, peA, peB, typ):
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

        phase_part_dict = {'bulk': {'all': bulk_id, 'A': A_bulk_id, 'B': B_bulk_id}, 'int': {'all': int_id, 'A': A_int_id, 'B': B_int_id}, 'gas': {'all': gas_id, 'A': A_gas_id, 'B': B_gas_id}, 'dense': {'all': dense_id, 'A': A_dense_id, 'B': B_dense_id}}
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
