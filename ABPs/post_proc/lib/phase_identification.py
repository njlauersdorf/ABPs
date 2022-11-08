
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

import statistics
from statistics import mode

#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility

class phase_identification:
    def __init__(self, area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ):

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

        theory_functs = theory.theory()

        self.lx_box = lx_box
        self.hx_box = self.lx_box/2
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        utility_functs = utility.utility(self.lx_box, self.ly_box)

        self.partNum = partNum
        self.min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen

        try:
            self.NBins_x = int(NBins_x)
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        self.sizeBin_x = utility_functs.roundUp((self.lx_box / self.NBins_x), 6)
        self.sizeBin_y = utility_functs.roundUp((self.ly_box / self.NBins_y), 6)

        self.peA = peA
        self.peB = peB
        self.parFrac = parFrac

        self.peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))   #Net activity of system

        self.eps = eps

        lat_theory = theory_functs.conForRClust(self.peNet, eps)
        curPLJ = theory_functs.ljPress(lat_theory, self.peNet, eps)
        self.phi_theory = theory_functs.latToPhi(lat_theory)
        self.phi_g_theory = theory_functs.compPhiG(self.peNet, lat_theory)

        self.typ = typ

    def phase_ident(self):

        phaseBin = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]            #Label phase of each bin

        phasePart=np.zeros(self.partNum)

        #Calculate density limits for phases (gas, interface, bulk)
        #Calculate analytical values

        print(self.phi_theory)

        vmax_eps = self.phi_theory * 1.4
        phi_dense_theory_max=self.phi_theory*1.3
        phi_dense_theory_min=self.phi_theory*0.95

        phi_gas_theory_max= self.phi_g_theory*4.0
        phi_gas_theory_min=0.0

        #Gradient of orientation
        align_grad = np.gradient(self.align_mag)

        #Gradient of number density
        area_frac_grad = np.gradient(self.area_frac)
        area_frac_A_grad = np.gradient(self.area_frac_A)
        area_frac_B_grad = np.gradient(self.area_frac_B)

        #Gradient of pressure
        press_grad = np.gradient(self.press)

        #Product of gradients of number density and orientation
        prod_grad = np.multiply(area_frac_grad, align_grad)

        #Magnitude of pressure gradient
        press_grad_mag = np.sqrt(press_grad[0]**2 + press_grad[1]**2)

        #Magnitude of number density gradient
        area_frac_grad_mag = np.sqrt(area_frac_grad[0]**2 + area_frac_grad[1]**2)
        area_frac_A_grad_mag = np.sqrt(area_frac_A_grad[0]**2 + area_frac_A_grad[1]**2)
        area_frac_B_grad_mag = np.sqrt(area_frac_B_grad[0]**2 + area_frac_B_grad[1]**2)

        #Magnitude of number_density * orientation gradient
        prod_grad_mag = np.sqrt(prod_grad[0]**2 + prod_grad[1]**2)

        #Weighted criterion for determining interface (more weighted to alignment than number density)
        criterion = self.align_mag*press_grad_mag

        #Ranges for determining interface
        criterion_min = 0.05*np.max(criterion)
        criterion_max = np.max(criterion)

        #Initialize count of bins for each phase
        gasBin_num=0
        edgeBin_num=0
        bulkBin_num=0

        #Label phase of bin per above criterion in number density and alignment
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                #Criterion for interface or gas
                if (criterion[ix][iy]<criterion_min) & (self.area_frac[ix][iy] < phi_dense_theory_min):

                    #Criterion for gas
                    if self.area_frac[ix][iy]<phi_gas_theory_max:
                        phaseBin[ix][iy]=2
                        gasBin_num+=1

                    #Criterion for interface
                    else:
                        phaseBin[ix][iy]=1
                        edgeBin_num+=1

                #Criterion for interface
                elif (criterion[ix][iy]>criterion_min) | (self.area_frac[ix][iy] < phi_dense_theory_min):
                    phaseBin[ix][iy]=1
                    edgeBin_num+=1

                #Otherwise, label it as bulk
                else:
                    phaseBin[ix][iy]=0
                    bulkBin_num+=1

                #Label each particle with same phase
                for h in range(0, len(self.binParts[ix][iy])):
                    phasePart[self.binParts[ix][iy][h]]=phaseBin[ix][iy]

        phase_dict = {'bin': phaseBin, 'part': phasePart}
        return phase_dict

    def phase_blur(self, phase_dict):
        # Blur interface (twice/two loops) identification to remove noise.
        #Check neighbors to be sure correctly identified phase. If not, average
        #with neighbors. If so, leave.
        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        for f in range(0,2):

            for ix in range(0, self.NBins_x):

                #Identify neighboring bin indices in x-direction
                if (ix + 1) == self.NBins_x:
                    lookx = [ix-1, ix, 0]
                elif ix==0:
                    lookx=[self.NBins_x-1, ix, ix+1]
                else:
                    lookx = [ix-1, ix, ix+1]

                # Loop through y index of mesh
                for iy in range(0, self.NBins_y):

                    #Identify neighboring bin indices in y-direction
                    if (iy + 1) == self.NBins_y:
                        looky = [iy-1, iy, 0]
                    elif iy==0:
                        looky=[self.NBins_y-1, iy, iy+1]
                    else:
                        looky = [iy-1, iy, iy+1]

                    #Count phases of surrounding bins
                    gas_bin=0
                    int_bin=0
                    bulk_bin=0
                    ref_phase = phaseBin[ix][iy]            #reference bin phase

                    #Loop through surrounding x-index
                    for indx in lookx:

                        # Loop through surrounding y-index
                        for indy in looky:

                            #If not reference bin, continue
                            if (indx!=ix) or (indy!=iy):

                                #If bulk, label it
                                if phaseBin[indx][indy]==0:
                                    bulk_bin+=1

                                #If interface, label it
                                elif phaseBin[indx][indy]==1:
                                    int_bin+=1

                                #If gas, label it
                                else:
                                    gas_bin+=1
                    #If reference bin is a gas bin, continue
                    if ref_phase==2:

                        #If 2 or fewer surrounding gas bins, change it to
                        #edge or bulk (whichever is more abundant)
                        if gas_bin<=2:
                            if int_bin>=bulk_bin:
                                phaseBin[ix][iy]=1
                            else:
                                phaseBin[ix][iy]=0

                    #If reference bin is a bulk bin, continue
                    elif ref_phase==0:

                        #If 2 or fewer surrounding bulk bins, change it to
                        #edge or gas (whichever is more abundant)
                        if bulk_bin<=2:
                            if int_bin>=gas_bin:
                                phaseBin[ix][iy]=1
                            else:
                                phaseBin[ix][iy]=2

                    #If reference bin is a edge bin, continue
                    elif ref_phase==1:

                        #If 2 or fewer surrounding edge bins, change it to
                        #bulk or gas (whichever is more abundant)
                        if int_bin<=2:
                            if bulk_bin>=gas_bin:
                                phaseBin[ix][iy]=0
                            else:
                                phaseBin[ix][iy]=2

        phase_dict = {'bin': phaseBin, 'part': phasePart}
        return phase_dict

    def update_phasePart(self, phase_dict):

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        #Label individual particle phases from identified bin phases
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                for h in range(0, len(self.binParts[ix][iy])):
                    phasePart[self.binParts[ix][iy][h]]=phaseBin[ix][iy]
        phase_dict = {'bin': phaseBin, 'part': phasePart}
        return phase_dict

    def phase_count(self, phase_dict):

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        #Label individual particle phases from identified bin phases
        int_num=0
        bulk_num=0
        gas_num=0
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if phaseBin[ix][iy]==1:
                    int_num+=1
                elif phaseBin[ix][iy]==0:
                    bulk_num+=1
                elif phaseBin[ix][iy]==2:
                    gas_num+=1

        count_dict = {'bulk': bulk_num, 'int': int_num, 'gas': gas_num}
        return count_dict

    def com_bulk(self, phase_dict, count_dict):
        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        bulk_num = count_dict['bulk']

        com_x_ind = int(self.hx_box / self.sizeBin_x)

        com_y_ind = int(self.hy_box / self.sizeBin_y)

        if phaseBin[com_x_ind][com_y_ind]==0:
            com_bulk_indx = com_x_ind
            com_bulk_indy = com_y_ind
        elif bulk_num>0:
            shortest_r = 10000
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if phaseBin[ix][iy]==0:
                        pos_x = ix * self.sizeBin_x
                        pos_y = iy * self.sizeBin_y
                        difx = (pos_x - self.hx_box)

                        #Enforce periodic boundary conditions
                        difx_abs = np.abs(difx)
                        if difx_abs>=self.hx_box:
                            if difx < -self.hx_box:
                                difx += self.lx_box
                            else:
                                difx -= self.lx_box

                        dify = (pos_y - self.hy_box)

                        #Enforce periodic boundary conditions
                        dify_abs = np.abs(dify)
                        if dify_abs>=self.hy_box:
                            if dify < -self.hy_box:
                                dify += self.ly_box
                            else:
                                dify -= self.ly_box

                        difr = (difx**2 + dify**2)**0.5

                        if difr < shortest_r:
                            shortest_r = difr
                            com_bulk_indx = ix
                            com_bulk_indy = iy
        else:
            com_bulk_indx = com_x_ind
            com_bulk_indy = com_y_ind
        bulk_com_dict = {'x': com_bulk_indx, 'y': com_bulk_indy}
        return bulk_com_dict

    def separate_bulks(self, phase_dict, count_dict, bulk_com_dict):

        phaseBulk=np.zeros(self.partNum)

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        bulk_num = count_dict['bulk']

        com_bulk_indx = bulk_com_dict['x']
        com_bulk_indy = bulk_com_dict['y']

        #initiate ix, iy bin id's to while-loop over

        bulk_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)            #Label separate interfaces

        bulk_num_current=0
        ix_ref = 0
        iy_ref = 0
        bulk_id_current=0

        while bulk_num_current!=bulk_num:
            #If bin is an interface, continue
            if phaseBin[ix_ref][iy_ref]==0:

                    #If bin hadn't been assigned an interface id yet, continue
                if bulk_id[ix_ref][iy_ref]==0:

                    bulk_id_current+=1         #Increase interface index

                    #Append ID of bulk ID
                    bulk_id_list=[]
                    bulk_id_list.append([ix_ref,iy_ref])

                    #Count surrounding bin phases
                    gas_count=0
                    bulk_count=0

                    #loop over identified interface bins
                    for ix,iy in bulk_id_list:

                        #identify neighboring bins
                        if (ix + 1) == self.NBins_x:
                            lookx = [ix-1, ix, 0]
                        elif ix==0:
                            lookx=[self.NBins_x-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, ix+1]

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

                                #If bin is a bulk, continue
                                if phaseBin[indx][indy]==0:

                                    #If bin wasn't assigned an interface id, continue
                                    if bulk_id[indx][indy]==0:

                                        #append ids to looped list
                                        bulk_id_list.append([indx, indy])
                                        bulk_num_current+=1

                                        #Append interface id
                                        bulk_id[indx][indy]=bulk_id_current

                                        for h in range(0, len(self.binParts[indx][indy])):
                                            phaseBulk[self.binParts[indx][indy][h]]=bulk_id_current


                    #If bin has been identified as an interface, look at different reference bin
                else:
                        if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                            break
                        if ix_ref!=(self.NBins_x-1):
                            ix_ref+=1
                        else:
                            ix_ref=0
                            iy_ref+=1
            #If bin is not an interface, go to different reference bin
            else:
                if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                    break
                if ix_ref!=(self.NBins_x-1):
                    ix_ref+=1
                else:
                    ix_ref=0
                    iy_ref+=1

        big_bulk_id = bulk_id[com_bulk_indx][com_bulk_indy]

        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}
        return bulk_dict

    def separate_ints(self, phase_dict, count_dict, bulk_dict):

        phaseInt=np.zeros(self.partNum)

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        int_num = count_dict['int']

        bulk_id = bulk_dict['bin']
        phaseBulk = bulk_dict['part']
        big_bulk_id = bulk_dict['largest id']

        int_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)            #Label separate interfaces

        #initiate ix, iy bin id's to while-loop over
        int_num_current = 0

        ix_ref=0
        iy_ref=0
        int_id_current=0

        possible_int_ids = []
        # Individually label each interface until all edge bins identified using flood fill algorithm
        while int_num_current!=int_num:

            #If bin is an interface, continue
            if phaseBin[ix_ref][iy_ref]==1:

                #If bin hadn't been assigned an interface id yet, continue
                if int_id[ix_ref][iy_ref]==0:

                    int_id_current+=1         #Increase interface index

                    #Append ID of interface ID
                    int_id_list=[]
                    bulk_id_list=[]
                    int_id_list.append([ix_ref,iy_ref])


                    num_neigh_int=0

                    #Count surrounding bin phases
                    gas_num=0
                    bulk_num=0

                    #loop over identified interface bins
                    for ix,iy in int_id_list:

                        #identify neighboring bins
                        if (ix + 1) == self.NBins_x:
                            lookx = [ix-1, ix, 0]
                        elif ix==0:
                            lookx=[self.NBins_x-1, ix, ix+1]
                        else:
                            lookx = [ix-1, ix, ix+1]
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

                                #If bin is an interface, continue
                                if phaseBin[indx][indy]==1:

                                    #If bin wasn't assigned an interface id, continue
                                    if int_id[indx][indy]==0:

                                        #append ids to looped list
                                        int_id_list.append([indx, indy])
                                        int_num_current+=1

                                        #Append interface id
                                        int_id[indx][indy]=int_id_current
                                        for h in range(0, len(self.binParts[indx][indy])):
                                            phaseInt[self.binParts[indx][indy][h]]=int_id_current
                                        num_neigh_int+=1

                                #If bin is a gas, count it
                                elif phaseBin[indx][indy]==2:

                                    gas_num+=1

                                #else bin is counted as bulk
                                else:
                                    if bulk_id[indx][indy]==big_bulk_id:
                                        if int_id_current not in possible_int_ids:
                                            possible_int_ids.append(int_id_current)
                                    bulk_num+=1
                                    bulk_id_list.append(bulk_id[indx, indy])

                    #If fewer than or equal to 4 neighboring interfaces, re-label phase as bulk or gas
                    if num_neigh_int<=4:

                        #If more neighboring gas bins, reference bin is truly a gas bin
                        if gas_num>bulk_num:
                            for ix in range(0, self.NBins_x):
                                for iy in range(0, self.NBins_y):
                                    if int_id[ix][iy]==int_id_current:
                                        int_id[ix][iy]=0
                                        phaseBin[ix][iy]=2
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseInt[self.binParts[ix][iy][h]]=0
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseBulk[self.binParts[ix][iy][h]]=0

                        #Else if more neighboring bulk bins, reference bin is truly a bulk bin
                        else:
                            for ix in range(0, self.NBins_x):
                                for iy in range(0, self.NBins_y):
                                    if int_id[ix][iy]==int_id_current:
                                        int_id[ix][iy]=0
                                        phaseBin[ix][iy]=0
                                        bulk_id[ix][iy]=mode(bulk_id_list)
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseBulk[self.binParts[ix][iy][h]]=mode(bulk_id_list)
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseInt[self.binParts[ix][iy][h]]=0


                #If bin has been identified as an interface, look at different reference bin
                else:
                    if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                        break
                    if ix_ref!=(self.NBins_x-1):
                        ix_ref+=1
                    else:
                        ix_ref=0
                        iy_ref+=1
            #If bin is not an interface, go to different reference bin
            else:
                if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                    break
                if ix_ref!=(self.NBins_x-1):
                    ix_ref+=1
                else:
                    ix_ref=0
                    iy_ref+=1

        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        phase_dict = {'bin': phaseBin, 'part': phasePart}
        return phase_dict, bulk_dict, int_dict

    def reduce_gas_noise(self, phase_dict, bulk_dict, int_dict):

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        bulk_id = bulk_dict['bin']
        phaseBulk = bulk_dict['part']
        big_bulk_id = bulk_dict['largest id']

        int_id = int_dict['bin']
        phaseInt = int_dict['part']
        possible_int_ids = int_dict['largest ids']

        #Label which interface each particle belongs to
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if (int_id[ix][iy] == 0) & (bulk_id[ix][iy]==0):
                    bulk_num=0
                    gas_num=0
                    bulk_id_list=[]
                    if (ix + 2) == self.NBins_x:
                        lookx = [ix-1, ix-1, ix, ix+1, 0]
                    elif (ix + 1) == self.NBins_x:
                        lookx = [ix-2, ix-1, ix, 0, 1]
                    elif ix==0:
                        lookx=[self.NBins_x-2, self.NBins_x-1, ix, ix+1, ix+2]
                    elif ix==1:
                        lookx=[self.NBins_x-1, ix-1, ix, ix+1, ix+2]
                    else:
                        lookx = [ix-2, ix-1, ix, ix+1, ix+2]

                    #Based on y-index (iy), find neighboring y-indices to loop through
                    if (iy + 2) == self.NBins_y:
                        looky = [iy-1, iy-1, iy, iy+1, 0]
                    elif (iy + 1) == self.NBins_y:
                        looky = [iy-2, iy-1, iy, 0, 1]
                    elif iy==0:
                        looky=[self.NBins_y-2, self.NBins_y-1, iy, iy+1, iy+2]
                    elif iy==1:
                        looky=[self.NBins_y-1, iy-1, iy, iy+1, iy+2]
                    else:
                        looky = [iy-2, iy-1, iy, iy+1, iy+2]

                    for indx in lookx:

                        for indy in looky:

                            if phaseBin[indx][indy]==0:
                                bulk_num+=1
                                bulk_id_list.append(bulk_id[indx, indy])
                            elif phaseBin[indx][indy]==2:
                                gas_num+=1
                    if bulk_num>=gas_num:
                        phaseBin[ix][iy]=0
                        bulk_id[ix][iy]=mode(bulk_id_list)
                    else:
                        phaseBin[ix][iy]=2
                    for h in range(0, len(self.binParts[ix][iy])):
                        phaseBulk[self.binParts[ix][iy][h]]=bulk_id[ix][iy]
                        phasePart[self.binParts[ix][iy][h]]=phaseBin[ix][iy]

        phase_dict = {'bin': phaseBin, 'part': phasePart}

        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        return phase_dict, bulk_dict, int_dict

    def int_comp(self, part_dict, phase_dict, bulk_dict, int_dict):

        partTyp = self.typ

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        bulk_id = bulk_dict['bin']
        phaseBulk = bulk_dict['part']
        big_bulk_id = bulk_dict['largest id']

        int_id = int_dict['bin']
        phaseInt = int_dict['part']
        possible_int_ids = int_dict['largest ids']

        bub_id = []

        int_A_comp = np.array([])
        int_B_comp = np.array([])
        int_comp = np.array([])

        int_small_num=0
        int_large_num=0

        int_large_ids=np.array([])
        if_large_int=[]

        max_int_id = int(np.max(phaseInt))

        #Determine which grouping of particles (phases or different interfaces) are large enough to perform measurements on or if noise
        for m in range(1, max_int_id+1):


            #Find which particles belong to group 'm'
            int_id_part = np.where(phaseInt==m)[0]
            int_id_part_num = len(int_id_part)

            int_id_bin_num=0

            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if int_id[ix][iy]==m:
                        int_id_bin_num +=1

            #If fewer than 100 particles belong to group 'm', then it is most likely noise and we should remove it
            if (int_id_part_num<=100) or (int_id_bin_num<10):
                int_small_num+=1
                phaseInt[int_id_part]=0

                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):

                        bulk_id_list = []
                        gas_num=0
                        bulk_num=0

                        if int_id[ix][iy]==m:
                            if m in possible_int_ids:
                                possible_int_ids.remove(m)
                            if (ix + 2) == self.NBins_x:
                                lookx = [ix-1, ix-1, ix, ix+1, 0]
                            elif (ix + 1) == self.NBins_x:
                                lookx = [ix-2, ix-1, ix, 0, 1]
                            elif ix==0:
                                lookx=[self.NBins_x-2, self.NBins_x-1, ix, ix+1, ix+2]
                            elif ix==1:
                                lookx=[self.NBins_x-1, ix-1, ix, ix+1, ix+2]
                            else:
                                lookx = [ix-2, ix-1, ix, ix+1, ix+2]

                            #Based on y-index (iy), find neighboring y-indices to loop through
                            if (iy + 2) == self.NBins_y:
                                looky = [iy-1, iy-1, iy, iy+1, 0]
                            elif (iy + 1) == self.NBins_y:
                                looky = [iy-2, iy-1, iy, 0, 1]
                            elif iy==0:
                                looky=[self.NBins_y-2, self.NBins_y-1, iy, iy+1, iy+2]
                            elif iy==1:
                                looky=[self.NBins_y-1, iy-1, iy, iy+1, iy+2]
                            else:
                                looky = [iy-2, iy-1, iy, iy+1, iy+2]

                            for indx in lookx:
                                for indy in looky:
                                    if phaseBin[indx][indy]==0:
                                        bulk_num+=1
                                        bulk_id_list.append(bulk_id[indx, indy])
                                    elif phaseBin[indx][indy]==2:
                                        gas_num+=1
                            int_id[ix][iy]=0
                            for h in range(0, len(self.binParts[ix][iy])):
                                phaseInt[self.binParts[ix][iy][h]]=0

                            if gas_num>bulk_num:
                                phaseBin[ix][iy]=2
                                if len(self.binParts[ix][iy])>0:
                                    for h in range(0, len(self.binParts[ix][iy])):
                                        phasePart[self.binParts[ix][iy][h]]=2
                            else:
                                phaseBin[ix][iy]=0
                                if len(self.binParts[ix][iy])>0:
                                    for h in range(0, len(self.binParts[ix][iy])):
                                        phasePart[self.binParts[ix][iy][h]]=0
                                        phaseBulk[self.binParts[ix][iy][h]]=mode(bulk_id_list)

            #If more than 100 particles belong to group 'm', then it is most likely significant and we should perform calculations
            else:

                #Label if structure is bulk/gas or interface
                if len(np.where(phasePart[int_id_part]==0)[0])==0:

                    #Calculate composition of particles in each structure
                    int_A_comp = np.append(int_A_comp, len(np.where((phaseInt==m) & (partTyp==0))[0]))
                    int_B_comp = np.append(int_B_comp, len(np.where((phaseInt==m) & (partTyp==1))[0]))
                    int_comp = np.append(int_comp, len(np.where((phaseInt==m) & (partTyp==1))[0])+len(np.where((phaseInt==m) & (partTyp==0))[0]))

                    if_large_int.append(1)

                    #Label significant structure IDs
                    int_large_ids = np.append(int_large_ids, m)

                    #Count number of significant structures
                    int_large_num+=1

        phase_dict = {'bin': phaseBin, 'part': phasePart}

        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        int_comp_dict = {'ids': {'int id': int_large_ids, 'if int': if_large_int}, 'comp': {'all': int_comp, 'A': int_A_comp, 'B': int_B_comp}}

        return phase_dict, bulk_dict, int_dict, int_comp_dict

    def bulk_comp(self, part_dict, phase_dict, bulk_dict):

        partTyp = self.typ

        phaseBin = phase_dict['bin']
        phasePart = phase_dict['part']

        bulk_id = bulk_dict['bin']
        phaseBulk = bulk_dict['part']
        big_bulk_id = bulk_dict['largest id']

        #Initiate empty arrays
        bulk_A_comp = np.array([])
        bulk_B_comp = np.array([])
        bulk_comp = np.array([])
        if_large_bulk = []
        bulk_large_num=0
        bulk_large_ids = np.array([])

        max_bulk_id = int(np.max(phaseBulk))

        #Calculate composition of each bulk phase structure
        for m in range(1, max_bulk_id+1):

            #Find which particles belong to group 'm'
            bulk_id_part = np.where(phaseBulk==m)[0]
            bulk_id_part_num = len(bulk_id_part)

            bulk_id_bin_num=0
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if bulk_id[ix][iy]==m:
                        bulk_id_bin_num +=1



            #Label if structure is bulk/gas or interface
            if bulk_id_part_num>0:

                if_large_bulk.append(1)
                #Calculate composition of particles in each structure
                bulk_A_comp = np.append(bulk_A_comp, len(np.where((phaseBulk==m) & (partTyp==0))[0]))
                bulk_B_comp = np.append(bulk_B_comp, len(np.where((phaseBulk==m) & (partTyp==1))[0]))
                bulk_comp = np.append(bulk_comp, len(np.where((phaseBulk==m) & (partTyp==1))[0])+len(np.where((phaseBulk==m) & (partTyp==0))[0]))
                #Label significant structure IDs
                bulk_large_ids = np.append(bulk_large_ids, m)

                bulk_large_num+=1

        bulk_comp_dict = {'ids': {'bulk id': bulk_large_ids, 'if bulk': if_large_bulk}, 'comp': {'all': bulk_comp, 'A': bulk_A_comp, 'B': bulk_B_comp}}

        return bulk_comp_dict

    def bulk_sort2(self, bulk_comp_dict):

        bulk_large_ids = bulk_comp_dict['ids']['bulk id']
        if_large_bulk = bulk_comp_dict['ids']['if bulk']

        bulk_comp = bulk_comp_dict['comp']['all']
        bulk_A_comp = bulk_comp_dict['comp']['A']
        bulk_B_comp = bulk_comp_dict['comp']['B']


        bulk_large_ids = [x for _,x in sorted(zip(bulk_comp, bulk_large_ids))]
        if_large_bulk = [x for _,x in sorted(zip(bulk_comp, if_large_bulk))]

        bulk_A_comp = [x for _,x in sorted(zip(bulk_comp, bulk_A_comp))]
        bulk_B_comp = [x for _,x in sorted(zip(bulk_comp, bulk_B_comp))]

        bulk_comp = sorted(bulk_comp)

        if len(bulk_comp)>5:
            bulk_comp = bulk_comp[:5]
            bulk_A_comp = bulk_A_comp[:5]
            bulk_B_comp = bulk_B_comp[:5]

            bulk_large_ids = bulk_large_ids[:5]
            if_large_bulk = if_large_bulk[:5]
        elif len(bulk_comp)<5:
            dif_len = int(5 - len(bulk_comp))
            for i in range(0, dif_len):
                bulk_comp.append(0)
                bulk_A_comp.append(0)
                bulk_B_comp.append(0)
                if_large_bulk.append(0)
                bulk_large_ids.append(999)

        bulk_comp_dict = {'ids': {'bulk id': bulk_large_ids, 'if bulk': if_large_bulk}, 'comp': {'all': bulk_comp, 'A': bulk_A_comp, 'B': bulk_B_comp}}

        return bulk_comp_dict

    def int_sort2(self, int_comp_dict):

        int_large_ids = int_comp_dict['ids']['int id']
        if_large_int = int_comp_dict['ids']['if int']

        int_comp = int_comp_dict['comp']['all']
        int_A_comp = int_comp_dict['comp']['A']
        int_B_comp = int_comp_dict['comp']['B']


        int_large_ids = [x for _,x in sorted(zip(int_comp, int_large_ids))]
        if_large_int = [x for _,x in sorted(zip(int_comp, if_large_int))]

        int_A_comp = [x for _,x in sorted(zip(int_comp, int_A_comp))]
        int_B_comp = [x for _,x in sorted(zip(int_comp, int_B_comp))]

        int_comp = sorted(int_comp)

        if len(int_comp)>5:
            int_comp = int_comp[:5]
            int_A_comp = int_A_comp[:5]
            int_B_comp = int_B_comp[:5]

            int_large_ids = int_large_ids[:5]
            if_large_int = if_large_int[:5]
        elif len(int_comp)<5:
            dif_len = int(5 - len(int_comp))

            for i in range(0, dif_len):
                int_comp.append(0)
                int_A_comp.append(0)
                int_B_comp.append(0)
                if_large_int.append(0)
                int_large_ids.append(999)

        int_comp_dict = {'ids': {'int id': int_large_ids, 'if int': if_large_int}, 'comp': {'all': int_comp, 'A': int_A_comp, 'B': int_B_comp}}

        return int_comp_dict
    def phase_part_count(self, phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ):

        phasePart = phase_dict['part']

        phaseBulk = bulk_dict['part']
        bulk_large_ids = bulk_comp_dict['ids']['bulk id']

        phaseInt = int_dict['part']
        int_large_ids = int_comp_dict['ids']['int id']

        bulk_num = len(np.where((phasePart==0))[0])
        bulk_A_num = len(np.where((phasePart==0) & (typ==0))[0])
        bulk_B_num = len(np.where((phasePart==0) & (typ==1))[0])

        largest_bulk_num = len(np.where((phaseBulk==bulk_large_ids[0]))[0])
        largest_bulk_A_num = len(np.where((phaseBulk==bulk_large_ids[0]) & (typ==0))[0])
        largest_bulk_B_num = len(np.where((phaseBulk==bulk_large_ids[0]) & (typ==1))[0])

        #Slow/fast composition of all interface
        int_num = len(np.where((phasePart==1))[0])
        int_A_num = len(np.where((phasePart==1) & (typ==0))[0])
        int_B_num = len(np.where((phasePart==1) & (typ==1))[0])

        #Slow/fast composition of main interface
        largest_int_num = len(np.where((phaseInt==int_large_ids[0]))[0])
        largest_int_A_num = len(np.where((phaseInt==int_large_ids[0]) & (typ==0))[0])
        largest_int_B_num = len(np.where((phaseInt==int_large_ids[0]) & (typ==1))[0])

        #Slow/fast composition of gas phase
        gas_num = len(np.where((phasePart==2))[0])
        gas_A_num = len(np.where((phasePart==2) & (typ==0))[0])
        gas_B_num = len(np.where((phasePart==2) & (typ==1))[0])

        #If bulk/gas exist, calculate the structure ID for the gas/bulk
        bulk_part_ids = np.where(phasePart==0)[0]        #Bulk phase structure(s)
        bulk_A_part_ids = np.where((phasePart==0) & (typ==0))[0]        #Bulk phase structure(s)
        bulk_B_part_ids = np.where((phasePart==0) & (typ==1))[0]        #Bulk phase structure(s)

        int_part_ids = np.where(phasePart==1)[0]     #Largest gas-dense interface
        int_A_part_ids = np.where((phasePart==1) & (typ==0))[0]        #Bulk phase structure(s)
        int_B_part_ids = np.where((phasePart==1) & (typ==1))[0]        #Bulk phase structure(s)

        gas_part_ids = np.where(phasePart==2)[0]              #Gas phase structure(s)
        gas_A_part_ids = np.where((phasePart==2) & (typ==0))[0]        #Bulk phase structure(s)
        gas_B_part_ids = np.where((phasePart==2) & (typ==1))[0]        #Bulk phase structure(s)

        largest_int_part_ids = np.where(phaseInt==int_large_ids[0])[0]     #Largest gas-dense interface
        largest_int_A_part_ids = np.where((phaseInt==int_large_ids[0]) & (typ==0))[0]     #Largest gas-dense interface
        largest_int_B_part_ids = np.where((phaseInt==int_large_ids[0]) & (typ==1))[0]     #Largest gas-dense interface


        largest_bulk_part_ids = np.where(phaseBulk==bulk_large_ids[0])[0]     #Largest gas-dense interface
        largest_bulk_A_part_ids = np.where((phaseBulk==bulk_large_ids[0]) & (typ==0))[0]     #Largest gas-dense interface
        largest_bulk_B_part_ids = np.where((phaseBulk==bulk_large_ids[0]) & (typ==1))[0]     #Largest gas-dense interface

        small_int_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0))[0]
        small_int_A_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0) & (typ==0))[0]
        small_int_B_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0) & (typ==1))[0]


        small_bulk_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0))[0]
        small_bulk_A_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0) & (typ==0))[0]     #Largest gas-dense interface
        small_bulk_B_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0) & (typ==1))[0]     #Largest gas-dense interface

        part_count_dict = {'bulk': {'all': bulk_num, 'A': bulk_A_num, 'B': bulk_B_num}, 'largest bulk': {'all': largest_bulk_num, 'A': largest_bulk_A_num, 'B': largest_bulk_B_num}, 'int': {'all': int_num, 'A': int_A_num, 'B': int_B_num}, 'largest int': {'all': largest_int_num, 'A': largest_int_A_num, 'B': largest_int_B_num}, 'gas': {'all': gas_num, 'A': gas_A_num, 'B': gas_B_num}}
        part_id_dict = {'bulk': {'all': bulk_part_ids, 'A': bulk_A_part_ids, 'B': bulk_B_part_ids}, 'largest bulk': {'all': largest_bulk_part_ids, 'A': largest_bulk_A_part_ids, 'B': largest_bulk_B_part_ids}, 'small bulk': {'all': small_bulk_part_ids, 'A': small_bulk_A_part_ids, 'B': small_bulk_B_part_ids}, 'int': {'all': int_part_ids, 'A': int_A_part_ids, 'B': int_B_part_ids}, 'largest int': {'all': largest_int_part_ids, 'A': largest_int_A_part_ids, 'B': largest_int_B_part_ids}, 'small int': {'all': small_int_part_ids, 'A': small_int_A_part_ids, 'B': small_int_B_part_ids}, 'gas': {'all': gas_part_ids, 'A': gas_A_part_ids, 'B': gas_B_part_ids}}

        part_count_dict = {'num': part_count_dict, 'ids': part_id_dict}
        return part_count_dict
    def phase_bin_count(self, phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict):

        phaseBin = phase_dict['bin']

        bulk_id = bulk_dict['bin']

        bulk_large_ids = bulk_comp_dict['ids']['bulk id']
        if_large_bulk = bulk_comp_dict['ids']['if bulk']

        int_id = int_dict['bin']

        int_large_ids = int_comp_dict['ids']['int id']
        if_large_int = int_comp_dict['ids']['if int']

        #Initiate counts of phases/structures
        bulk_num=0
        gas_num=0
        all_int_num=0
        largest_int_num=0

        int_num_arr = np.zeros(len(int_large_ids))
        bulk_num_arr = np.zeros(len(bulk_large_ids))

        #Measure number of bins belong to each phase
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if phaseBin[ix][iy]==0:
                    bulk_num+=1
                elif phaseBin[ix][iy]==2:
                    gas_num+=1
                elif phaseBin[ix][iy]==1:
                    all_int_num+=1
                if int_id[ix][iy]==int_large_ids[0]:
                    largest_int_num+=1

        #Count number of bins belonging to each interface structure
        for m in range(0, len(int_large_ids)):
            if if_large_int[m]!=0:
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):
                        if int_id[ix][iy] == int_large_ids[m]:
                            int_num_arr[m] +=1

        #Count number of bins belonging to each bulk phase structure
        for m in range(0, len(bulk_large_ids)):
            if if_large_bulk[m]!=0:
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):
                        if bulk_id[ix][iy] == bulk_large_ids[m]:
                            bulk_num_arr[m] +=1

        bin_count_dict = {'bin': {'bulk': bulk_num, 'largest int': largest_int_num, 'gas': gas_num, 'all int': all_int_num}, 'ids': {'int': int_num_arr, 'bulk': bulk_num_arr}}
        return bin_count_dict

    def bulk_sort():
        #Initiate empty arrays
        bulk_id_arr = np.array([], dtype=int)
        bulk_size_id_arr = np.array([], dtype=int)
        bulk_fast_arr = np.array([], dtype=int)
        bulk_slow_arr = np.array([], dtype=int)
        if_bulk_id_arr = np.array([], dtype=int)

        #If 5 or more bulk phase structure...
        if bulk_large>=5:

            first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

            bulk_first_id = np.where(bulk_total_comp==first)[0]

            for k in range(0, len(bulk_first_id)):
                if len(bulk_id_arr)<5:
                    bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

            if len(bulk_id_arr)<5:
                second_arr = np.where(bulk_total_comp != first)[0]
                if len(second_arr)>0:
                    second = np.max(bulk_total_comp[second_arr])
                    bulk_second_id = np.where(bulk_total_comp==second)[0]
                    for k in range(0, len(bulk_second_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                else:
                    second_arr = 0
                    second = 0
                    bulk_second_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                if len(third_arr)>0:
                    third = np.max(bulk_total_comp[third_arr])
                    bulk_third_id = np.where(bulk_total_comp==third)[0]
                    for k in range(0, len(bulk_third_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                else:
                    third_arr = 0
                    third = 0
                    bulk_third_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)
            if len(bulk_id_arr)<5:
                fourth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third))[0]
                if len(fourth_arr)>0:
                    fourth = np.max(bulk_total_comp[fourth_arr])
                    bulk_fourth_id = np.where(bulk_total_comp==fourth)[0]
                    for k in range(0, len(bulk_fourth_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_fourth_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fourth_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fourth_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fourth_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fourth_id[k]])
                else:
                    fourth_arr = 0
                    fourth = 0
                    bulk_fourth_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third) & (bulk_total_comp!=fourth))[0]
                if len(fifth_arr)>0:
                    fifth = np.max(bulk_total_comp[fifth_arr])
                    bulk_fifth_id = np.where(bulk_total_comp==fifth)[0]
                    for k in range(0, len(bulk_fifth_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_fifth_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fifth_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fifth_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fifth_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fifth_id[k]])
                else:
                    second_arr = 0
                    second = 0
                    bulk_second_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)
            clust_true = 1
        #If 4 bulk phase structures...
        elif bulk_large==4:
            first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

            bulk_first_id = np.where(bulk_total_comp==first)[0]

            for k in range(0, len(bulk_first_id)):
                if len(bulk_id_arr)<5:
                    bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

            if len(bulk_id_arr)<5:
                second_arr = np.where(bulk_total_comp != first)[0]
                if len(second_arr)>0:
                    second = np.max(bulk_total_comp[second_arr])
                    bulk_second_id = np.where(bulk_total_comp==second)[0]
                    for k in range(0, len(bulk_second_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                else:
                    second_arr = 0
                    second = 0
                    bulk_second_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                if len(third_arr)>0:
                    third = np.max(bulk_total_comp[third_arr])
                    bulk_third_id = np.where(bulk_total_comp==third)[0]
                    for k in range(0, len(bulk_third_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                else:
                    third_arr = 0
                    third = 0
                    bulk_third_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fourth_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first) & (bulk_total_comp!=third))[0]
                if len(fourth_arr)>0:
                    fourth = np.max(bulk_total_comp[fourth_arr])
                    bulk_fourth_id = np.where(bulk_total_comp==fourth)[0]
                    for k in range(0, len(bulk_fourth_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_fourth_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_fourth_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_fourth_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_fourth_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_fourth_id[k]])
                else:
                    fourth_arr = 0
                    fourth = 0
                    bulk_fourth_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = 0
                fifth = 0
                bulk_fifth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)
        #If 3 bulk phase structures...
        elif bulk_large==3:
            first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

            bulk_first_id = np.where(bulk_total_comp==first)[0]

            for k in range(0, len(bulk_first_id)):
                if len(bulk_id_arr)<5:
                    bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

            if len(bulk_id_arr)<5:
                second_arr = np.where(bulk_total_comp != first)[0]
                if len(second_arr)>0:
                    second = np.max(bulk_total_comp[second_arr])
                    bulk_second_id = np.where(bulk_total_comp==second)[0]
                    for k in range(0, len(bulk_second_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                else:
                    second_arr = 0
                    second = 0
                    bulk_second_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = np.where((bulk_total_comp!=second) & (bulk_total_comp!=first))[0]
                if len(third_arr)>0:
                    third = np.max(bulk_total_comp[third_arr])
                    bulk_third_id = np.where(bulk_total_comp==third)[0]
                    for k in range(0, len(bulk_third_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_third_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_third_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_third_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_third_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_third_id[k]])
                else:
                    third_arr = 0
                    third = 0
                    bulk_third_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fourth_arr = 0
                fourth = 0
                bulk_fourth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = 0
                fifth = 0
                bulk_fifth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            clust_true = 1

        #If 2 bulk phase structures...
        elif bulk_large==2:
            first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

            bulk_first_id = np.where(bulk_total_comp==first)[0]

            for k in range(0, len(bulk_first_id)):
                if len(bulk_id_arr)<5:
                    bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

            if len(bulk_id_arr)<5:
                second_arr = np.where(bulk_total_comp != first)[0]
                if len(second_arr)>0:
                    second = np.max(bulk_total_comp[second_arr])
                    bulk_second_id = np.where(bulk_total_comp==second)[0]
                    for k in range(0, len(bulk_second_id)):
                        if len(bulk_id_arr)<5:
                            bulk_id_arr = np.append(bulk_id_arr, bulk_second_id[k])
                            bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_second_id[k]])
                            if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_second_id[k]])
                            bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_second_id[k]])
                            bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_second_id[k]])
                else:
                    second_arr = 0
                    second = 0
                    bulk_second_id = 0
                    bulk_id_arr = np.append(bulk_id_arr, 999)
                    bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                    if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                    bulk_fast_arr = np.append(bulk_fast_arr, 0)
                    bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = 0
                third = 0
                bulk_third_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fourth_arr = 0
                fourth = 0
                bulk_fourth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = 0
                fifth = 0
                bulk_fifth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            clust_true = 1

        #If 1 bulk phase structures...
        elif bulk_large==1:
            first=np.max(bulk_total_comp[np.where(bulk_large_ids==big_bulk_id)[0]])

            bulk_first_id = np.where(bulk_total_comp==first)[0]

            for k in range(0, len(bulk_first_id)):
                if len(bulk_id_arr)<5:
                    bulk_id_arr = np.append(bulk_id_arr, bulk_first_id[k])
                    bulk_size_id_arr = np.append(bulk_size_id_arr, bulk_large_ids[bulk_first_id[k]])
                    if_bulk_id_arr = np.append(if_bulk_id_arr, if_bulk[bulk_first_id[k]])
                    bulk_fast_arr = np.append(bulk_fast_arr, bulk_fast_comp[bulk_first_id[k]])
                    bulk_slow_arr = np.append(bulk_slow_arr, bulk_slow_comp[bulk_first_id[k]])

            if len(bulk_id_arr)<5:
                second_arr = 0
                second = 0
                bulk_second_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = 0
                third = 0
                bulk_third_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fourth_arr = 0
                fourth = 0
                bulk_fourth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = 0
                fifth = 0
                bulk_fifth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            clust_true = 1

        #If 0 bulk phase structures...
        elif bulk_large==0:
            if len(bulk_id_arr)<5:
                first_arr = 0
                first = 0
                bulk_first_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                second_arr = 0
                second = 0
                bulk_second_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                third_arr = 0
                third = 0
                bulk_third_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fourth_arr = 0
                fourth = 0
                bulk_fourth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            if len(bulk_id_arr)<5:
                fifth_arr = 0
                fifth = 0
                bulk_fifth_id = 0
                bulk_id_arr = np.append(bulk_id_arr, 999)
                bulk_size_id_arr = np.append(bulk_size_id_arr, 0)
                if_bulk_id_arr = np.append(if_bulk_id_arr, 0)
                bulk_fast_arr = np.append(bulk_fast_arr, 0)
                bulk_slow_arr = np.append(bulk_slow_arr, 0)

            clust_true = 1

        #Identify which structures are bulk/gas phase
        bulk_ids = np.where(if_bub_id_arr==0)[0]

        #If bulk/gas exist, calculate the structure ID for the gas/bulk
        if len(bulk_ids)>0:
            bulk_id = bub_size_id_arr[np.min(np.where(if_bub_id_arr==0)[0])]

        return

    def int_sort():
        #Identify which of the largest bubbles is a possible gas-dense interface
        int_poss_ids = []
        for k in range(0, len(possible_interface_ids)):

            if possible_interface_ids[k] in bub_large_ids:

                int_poss_ids.append(np.where(bub_large_ids==possible_interface_ids[k])[0][0])

        # Determine order of interfaces based on size (largest=dense + gas phases, second largest = gas/dense interface, etc.)

        #Initiate empty arrays
        bub_id_arr = np.array([], dtype=int)
        bub_size_id_arr = np.array([], dtype=int)
        bub_fast_arr = np.array([], dtype=int)
        bub_slow_arr = np.array([], dtype=int)
        if_bub_id_arr = np.array([], dtype=int)

        #If 5 or more interface structures, continue...
        if bub_large>=5:
                if len(int_poss_ids)>0:
                    first=np.max(bub_total_comp[int_poss_ids])
                else:
                    first=np.max(bub_total_comp)

                bub_first_id = np.where(bub_total_comp==first)[0]

                for k in range(0, len(bub_first_id)):
                    if len(bub_id_arr)<5:
                        bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                        bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                        if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                        bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                        bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                if len(bub_id_arr)<5:
                    second_arr = np.where(bub_total_comp != first)[0]
                    if len(second_arr)>0:
                        second = np.max(bub_total_comp[second_arr])
                        bub_second_id = np.where(bub_total_comp==second)[0]
                        for k in range(0, len(bub_second_id)):
                            if len(bub_id_arr)<5:
                                bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                    else:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                    if len(third_arr)>0:
                        third = np.max(bub_total_comp[third_arr])
                        bub_third_id = np.where(bub_total_comp==third)[0]
                        for k in range(0, len(bub_third_id)):
                            if len(bub_id_arr)<5:
                                bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                    else:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fourth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third))[0]
                    if len(fourth_arr)>0:
                        fourth = np.max(bub_total_comp[fourth_arr])
                        bub_fourth_id = np.where(bub_total_comp==fourth)[0]
                        for k in range(0, len(bub_fourth_id)):
                            if len(bub_id_arr)<5:
                                bub_id_arr = np.append(bub_id_arr, bub_fourth_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fourth_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fourth_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fourth_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fourth_id[k]])
                    else:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fifth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third) & (bub_total_comp!=fourth))[0]
                    if len(fifth_arr)>0:
                        fifth = np.max(bub_total_comp[fifth_arr])
                        bub_fifth_id = np.where(bub_total_comp==fifth)[0]
                        for k in range(0, len(bub_fifth_id)):
                            if len(bub_id_arr)<5:
                                bub_id_arr = np.append(bub_id_arr, bub_fifth_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fifth_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fifth_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fifth_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fifth_id[k]])
                    else:
                        fifth_arr = 0
                        fifth = 0
                        bub_fifth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)
                clust_true = 1
        #If 4 interface structures...
        elif bub_large==4:
                if len(int_poss_ids)>0:
                    first=np.max(bub_total_comp[int_poss_ids])
                else:
                    first=np.max(bub_total_comp)

                bub_first_id = np.where(bub_total_comp==first)[0]

                for k in range(0, len(bub_first_id)):
                    if len(bub_id_arr)<4:
                        bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                        bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                        if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                        bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                        bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                if len(bub_id_arr)<4:
                    second_arr = np.where(bub_total_comp != first)[0]
                    if len(second_arr)>0:
                        second = np.max(bub_total_comp[second_arr])
                        bub_second_id = np.where(bub_total_comp==second)[0]
                        for k in range(0, len(bub_second_id)):
                            if len(bub_id_arr)<4:
                                bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                    else:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<4:
                    third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                    if len(third_arr)>0:
                        third = np.max(bub_total_comp[third_arr])
                        bub_third_id = np.where(bub_total_comp==third)[0]
                        for k in range(0, len(bub_third_id)):
                            if len(bub_id_arr)<4:
                                bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                    else:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<4:
                    fourth_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first) & (bub_total_comp!=third))[0]
                    if len(fourth_arr)>0:
                        fourth = np.max(bub_total_comp[fourth_arr])
                        bub_fourth_id = np.where(bub_total_comp==fourth)[0]
                        for k in range(0, len(bub_fourth_id)):
                            if len(bub_id_arr)<4:
                                bub_id_arr = np.append(bub_id_arr, bub_fourth_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_fourth_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_fourth_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_fourth_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_fourth_id[k]])
                    else:
                        fourth_arr = 0
                        fourth = 0
                        bub_fourth_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)
                if len(bub_id_arr)<5:
                    fifth_arr = 0
                    fifth = 0
                    bub_fifth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)


                clust_true = 1
        #If 3 interface structures...
        elif bub_large==3:
                if len(int_poss_ids)>0:
                    first=np.max(bub_total_comp[int_poss_ids])
                else:
                    first=np.max(bub_total_comp)

                bub_first_id = np.where(bub_total_comp==first)[0]

                for k in range(0, len(bub_first_id)):
                    if len(bub_id_arr)<3:
                        bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                        bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                        if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                        bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                        bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                if len(bub_id_arr)<3:
                    second_arr = np.where(bub_total_comp != first)[0]
                    if len(second_arr)>0:
                        second = np.max(bub_total_comp[second_arr])
                        bub_second_id = np.where(bub_total_comp==second)[0]
                        for k in range(0, len(bub_second_id)):
                            if len(bub_id_arr)<3:
                                bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                    else:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<3:
                    third_arr = np.where((bub_total_comp!=second) & (bub_total_comp!=first))[0]
                    if len(third_arr)>0:
                        third = np.max(bub_total_comp[third_arr])
                        bub_third_id = np.where(bub_total_comp==third)[0]
                        for k in range(0, len(bub_third_id)):
                            if len(bub_id_arr)<3:
                                bub_id_arr = np.append(bub_id_arr, bub_third_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_third_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_third_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_third_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_third_id[k]])
                    else:
                        third_arr = 0
                        third = 0
                        bub_third_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)
                if len(bub_id_arr)<5:
                    fourth_arr = 0
                    fourth = 0
                    bub_fourth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fifth_arr = 0
                    fifth = 0
                    bub_fifth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)


                clust_true = 1

        #If 2 interface structures...
        elif bub_large==2:
                if len(int_poss_ids)>0:
                    first=np.max(bub_total_comp[int_poss_ids])
                else:
                    first=np.max(bub_total_comp)

                bub_first_id = np.where(bub_total_comp==first)[0]

                for k in range(0, len(bub_first_id)):
                    if len(bub_id_arr)<2:
                        bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                        bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                        if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                        bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                        bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                if len(bub_id_arr)<2:
                    second_arr = np.where(bub_total_comp != first)[0]
                    if len(second_arr)>0:
                        second = np.max(bub_total_comp[second_arr])
                        bub_second_id = np.where(bub_total_comp==second)[0]
                        for k in range(0, len(bub_second_id)):
                            if len(bub_id_arr)<2:
                                bub_id_arr = np.append(bub_id_arr, bub_second_id[k])
                                bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_second_id[k]])
                                if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_second_id[k]])
                                bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_second_id[k]])
                                bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_second_id[k]])
                    else:
                        second_arr = 0
                        second = 0
                        bub_second_id = 0
                        bub_id_arr = np.append(bub_id_arr, 999)
                        bub_size_id_arr = np.append(bub_size_id_arr, 0)
                        if_bub_id_arr = np.append(if_bub_id_arr, 0)
                        bub_fast_arr = np.append(bub_fast_arr, 0)
                        bub_slow_arr = np.append(bub_slow_arr, 0)
                if len(bub_id_arr)<5:
                    third_arr = 0
                    third = 0
                    bub_third_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fourth_arr = 0
                    fourth = 0
                    bub_fourth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fifth_arr = 0
                    fifth = 0
                    bub_fifth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)


                clust_true = 1

        #If 1 interface structure...
        elif bub_large==1:
                if len(int_poss_ids)>0:
                    first=np.max(bub_total_comp[int_poss_ids])
                else:
                    first=np.max(bub_total_comp)

                bub_first_id = np.where(bub_total_comp==first)[0]

                for k in range(0, len(bub_first_id)):
                    if len(bub_id_arr)<1:
                        bub_id_arr = np.append(bub_id_arr, bub_first_id[k])
                        bub_size_id_arr = np.append(bub_size_id_arr, bub_large_ids[bub_first_id[k]])
                        if_bub_id_arr = np.append(if_bub_id_arr, if_bub[bub_first_id[k]])
                        bub_fast_arr = np.append(bub_fast_arr, bub_fast_comp[bub_first_id[k]])
                        bub_slow_arr = np.append(bub_slow_arr, bub_slow_comp[bub_first_id[k]])

                if len(bub_id_arr)<5:
                    second_arr = 0
                    second = 0
                    bub_second_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    third_arr = 0
                    third = 0
                    bub_third_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fourth_arr = 0
                    fourth = 0
                    bub_fourth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fifth_arr = 0
                    fifth = 0
                    bub_fifth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)


                clust_true = 1

        #If no interface structures (this is an error)...
        else:

                if len(bub_id_arr)<5:
                    first_arr = 0
                    first = 0
                    bub_first_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    second_arr = 0
                    second = 0
                    bub_second_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    third_arr = 0
                    third = 0
                    bub_third_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fourth_arr = 0
                    fourth = 0
                    bub_fourth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)

                if len(bub_id_arr)<5:
                    fifth_arr = 0
                    fifth = 0
                    bub_fifth_id = 0
                    bub_id_arr = np.append(bub_id_arr, 999)
                    bub_size_id_arr = np.append(bub_size_id_arr, 0)
                    if_bub_id_arr = np.append(if_bub_id_arr, 0)
                    bub_fast_arr = np.append(bub_fast_arr, 0)
                    bub_slow_arr = np.append(bub_slow_arr, 0)


                clust_true = 1
        #Identify which structures are bubbles
        bub_ids = np.where(if_bub_id_arr==1)[0]

        #If bubbles exist, calculate the structure ID for the interface
        if len(bub_ids)>0:
            interface_id = bub_size_id_arr[np.min(np.where(if_bub_id_arr==1)[0])]

        return
