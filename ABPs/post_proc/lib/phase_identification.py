
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

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility

# Class of phase identification functions
class phase_identification:
    def __init__(self, area_frac_dict, align_dict, part_dict, press_dict, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, parFrac, eps, typ):

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
        theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        # Initialize utility functions for call back later
        utility_functs = utility.utility(self.lx_box, self.ly_box)

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
        self.sizeBin_x = utility_functs.roundUp((self.lx_box / self.NBins_x), 6)

        # Y-length of bin
        self.sizeBin_y = utility_functs.roundUp((self.ly_box / self.NBins_y), 6)

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
        lat_theory = theory_functs.conForRClust(self.peNet, eps)

        # Calculate interparticle pressure from theory
        curPLJ = theory_functs.ljPress(lat_theory, self.peNet, eps)

        # Calculate dense phase area fraction from theory
        self.phi_theory = theory_functs.latToPhi(lat_theory)

        # Calculate gas phase area fraction from theory
        self.phi_g_theory = theory_functs.compPhiG(self.peNet, lat_theory)

        # Array (partNum) of particle types
        self.typ = typ

    def phase_ident(self):
        '''
        Purpose: Takes the average orientation, area fraction, and pressure of each bin
        and determines whether the bins and belong to the bulk (0), interface (1), or gas (2)
        phase

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin and each particle.
        Bulk=0, Interface=1, Gas=2.
        '''
        # Instantiate empty array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = [[0 for b in range(self.NBins_y)] for a in range(self.NBins_x)]            #Label phase of each bin

        # Instantiate empty array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart=np.zeros(self.partNum)

        #Calculate density ranges for phase identification (gas, interface, bulk) based on theory
        phi_dense_theory_max=self.phi_theory*1.3
        phi_dense_theory_min=self.phi_theory*0.95

        phi_gas_theory_max= self.phi_g_theory*4.0
        phi_gas_theory_min=0.0

        #Gradient of pressure
        press_grad = np.gradient(self.press)
        press_grad_mag = np.sqrt(press_grad[0]**2 + press_grad[1]**2)

        #Weighted criterion for determining interface (more weighted to alignment than number density)
        criterion = self.align_mag*press_grad_mag

        # Criterion ranges for differentiating interface
        criterion_min = 0.05*np.max(criterion)
        criterion_max = np.max(criterion)

        #Initialize count of bins for each phase
        gasBin_num=0
        edgeBin_num=0
        bulkBin_num=0

        #Label phase of bin per above criterion in number density and alignment

        # Loop over all bins
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

        # Dictionary containing arrays for the phase of each bin and each particle
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        return phase_dict

    def phase_blur(self, phase_dict):
        '''
        Purpose: Takes the phase ids of each bin and blurs the neighbor bins to
        reduce mis-identifications and noise in phase determination method

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin and each particle
        after blurring. Bulk=0, Interface=1, Gas=2.
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Blur interface (twice/two loops) identification to remove noise.
        #Check neighbors to be sure correctly identified phase. If not, average
        #with neighbors. If so, leave.

        # Perform blur twice
        for f in range(0,2):

            # Loop over all x bin indices
            for ix in range(0, self.NBins_x):

                #Identify neighboring bin indices in x-direction
                if (ix + 1) == self.NBins_x:
                    lookx = [ix-1, ix, 0]
                elif ix==0:
                    lookx=[self.NBins_x-1, ix, ix+1]
                else:
                    lookx = [ix-1, ix, ix+1]

                # Loop over all y bin indices
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

                    #reference bin phase
                    ref_phase = phaseBin[ix][iy]

                    #Loop through neighboring bins, including reference bin
                    for indx in lookx:
                        for indy in looky:

                            #If not reference bin, continue
                            if (indx!=ix) or (indy!=iy):

                                #Count number of neighboring bins of each phase
                                if phaseBin[indx][indy]==0:
                                    bulk_bin+=1
                                elif phaseBin[indx][indy]==1:
                                    int_bin+=1
                                else:
                                    gas_bin+=1

                    #If reference bin is gas...
                    if ref_phase==2:

                        # If 2 or fewer surrounding gas bins, change reference bin
                        # phase to more abundant neighboring phase [bulk (0) or interface (1)]
                        if gas_bin<=2:
                            if int_bin>=bulk_bin:
                                phaseBin[ix][iy]=1
                            else:
                                phaseBin[ix][iy]=0

                    #If reference bin is a bulk bin...
                    elif ref_phase==0:

                        #If 2 or fewer surrounding bulk bins, change it to
                        # more abundant neighboring phase [interface (1) or gas (2)]
                        if bulk_bin<=2:
                            if int_bin>=gas_bin:
                                phaseBin[ix][iy]=1
                            else:
                                phaseBin[ix][iy]=2

                    #If reference bin is a edge bin...
                    elif ref_phase==1:

                        #If 2 or fewer surrounding bulk bins, change it to
                        # more abundant neighboring phase [bulk (0) or gas (2)]
                        if int_bin<=2:
                            if bulk_bin>=gas_bin:
                                phaseBin[ix][iy]=0
                            else:
                                phaseBin[ix][iy]=2

        # Dictionary containing arrays for the phase of each bin and each particle after blurring
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        return phase_dict

    def update_phasePart(self, phase_dict):
        '''
        Purpose: Takes the blurred phase ids of each bin and updates the array of
        identified particle phases.

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin and each particle
        after blurring. Bulk=0, Interface=1, Gas=2.
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        #Label individual particle phases from identified bin phases

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Loop over all particles in reference bin
                for h in range(0, len(self.binParts[ix][iy])):

                    # Update particle phase using bin phase
                    phasePart[self.binParts[ix][iy][h]]=phaseBin[ix][iy]

        # Dictionary containing arrays for the phase of each bin and each particle after blurring
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        return phase_dict

    def phase_count(self, phase_dict):
        '''
        Purpose: Takes the phase ids of each bin and counts the number of bins of each phase

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        Outputs:
        count_dict: dictionary containing the number of bins of each phase ['bulk' (0), 'int' (1), or 'gas' (2)]
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        #Label individual particle phases from identified bin phases
        int_num=0
        bulk_num=0
        gas_num=0

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Count number of each bin
                if phaseBin[ix][iy]==1:
                    int_num+=1
                elif phaseBin[ix][iy]==0:
                    bulk_num+=1
                elif phaseBin[ix][iy]==2:
                    gas_num+=1

        # Dictionary containing the number of bins of each phase
        count_dict = {'bulk': bulk_num, 'int': int_num, 'gas': gas_num}

        return count_dict

    def com_bulk(self, phase_dict, count_dict):
        '''
        Purpose: Takes the phase ids of each bin and finds the CoM bin index of the bulk phase bins

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        count_dict: dictionary containing the number of bins of each phase ['bulk' (0), 'int' (1), or 'gas' (2)]

        Outputs:
        bulk_com_dict: dictionary containing CoM index of the bulk phases
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Number of bulk bins
        bulk_num = count_dict['bulk']

        # Mid-point bin index of simulation box
        com_x_ind = int(self.hx_box / self.sizeBin_x)
        com_y_ind = int(self.hy_box / self.sizeBin_y)

        # If the mid-point bin (which should be the CoM of the typical cluster if only 1 cluster present)
        if phaseBin[com_x_ind][com_y_ind]==0:
            com_bulk_indx = com_x_ind
            com_bulk_indy = com_y_ind

        # Otherwise, if a bulk phase is present, find the actual CoM
        elif bulk_num>0:

            # Shortest distance from simulation box midpoint (initialized as something unrealistically large)
            shortest_r = 10000

            # Loop over all bins
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):

                    # If bulk bin...
                    if phaseBin[ix][iy]==0:

                        # Bottom-left corner position of reference bin
                        pos_x = ix * self.sizeBin_x
                        pos_y = iy * self.sizeBin_y

                        # x-separation distance from simulation box mid-point
                        difx = (pos_x - self.hx_box)

                        #Enforce periodic boundary conditions
                        difx_abs = np.abs(difx)
                        if difx_abs>=self.hx_box:
                            if difx < -self.hx_box:
                                difx += self.lx_box
                            else:
                                difx -= self.lx_box

                        # y-separation distance from simulation box mid-point
                        dify = (pos_y - self.hy_box)

                        #Enforce periodic boundary conditions
                        dify_abs = np.abs(dify)
                        if dify_abs>=self.hy_box:
                            if dify < -self.hy_box:
                                dify += self.ly_box
                            else:
                                dify -= self.ly_box

                        # Total distance of bottom-left corner of reference bin from
                        # simulation box midpoint
                        difr = (difx**2 + dify**2)**0.5

                        # If total distance is less than all previous bins, save its info
                        if difr < shortest_r:
                            shortest_r = difr
                            com_bulk_indx = ix
                            com_bulk_indy = iy

        # If no bulk bins, set CoM of bulk as simulation box midpoint
        else:
            com_bulk_indx = com_x_ind
            com_bulk_indy = com_y_ind

        # Dictionary containing CoM index of the bulk phases
        bulk_com_dict = {'x': com_bulk_indx, 'y': com_bulk_indy}

        return bulk_com_dict

    def separate_bulks(self, phase_dict, count_dict, bulk_com_dict):
        '''
        Purpose: Takes the phase ids of each bin/particle and separates the bulk bin into
        individual, isolated bulk phases (separate clusters)

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        count_dict: dictionary containing the number of bins of each phase ['bulk' (0), 'int' (1), or 'gas' (2)]

        bulk_com_dict: dictionary containing CoM index of the bulk phases

        Outputs:
        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Number of bulk bins
        bulk_num = count_dict['bulk']

        # Mid-point bin index of bulk phase
        com_bulk_indx = bulk_com_dict['x']
        com_bulk_indy = bulk_com_dict['y']

        # Instantiate empty array (partNum) that identifies which separate bulk phase (of multiple
        # clusters) each bulk particle belongs to
        phaseBulk=np.zeros(self.partNum)

        # Instantiate empty array (NBins_x, NBins_y) that identifies which separate bulk phase (of multiple
        # clusters) each bulk bin belongs to
        bulk_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)            #Label separate interfaces

        # Reference bin IDs
        ix_ref = 0
        iy_ref = 0

        # Current number of bulk particles with current bulk ID
        bulk_num_current=0

        # Current bulk ID
        bulk_id_current=0

        # Individually label each bulk phase until all bulk bins identified using flood fill algorithm

        # Loop over all bulk bins until labeled
        while bulk_num_current!=bulk_num:

            #If bin is an bulk, continue
            if phaseBin[ix_ref][iy_ref]==0:

                #If bin hadn't been assigned a bulk id yet, continue
                if bulk_id[ix_ref][iy_ref]==0:

                    #Increase bulk phase id
                    bulk_id_current+=1

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

                        #loop over neighboring bins, including itself
                        for indx in lookx:
                            for indy in looky:

                                #If bin is a bulk...
                                if phaseBin[indx][indy]==0:

                                    #If bin wasn't assigned a bulk id...
                                    if bulk_id[indx][indy]==0:

                                        # Save neighboring bulk bin IDs
                                        bulk_id_list.append([indx, indy])

                                        # Count number of bulk bins in this individual bulk phase
                                        bulk_num_current+=1

                                        # Assign current bulk ID to neighboring bulk bin, including itself
                                        bulk_id[indx][indy]=bulk_id_current

                                        # Loop over all particles in neighboring bin and label each particle's bulk ID
                                        for h in range(0, len(self.binParts[indx][indy])):
                                            phaseBulk[self.binParts[indx][indy][h]]=bulk_id_current


                #If bulk ID been identified of reference bin, look at different reference bin
                else:

                    if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                        break
                    if ix_ref!=(self.NBins_x-1):
                        ix_ref+=1
                    else:
                        ix_ref=0
                        iy_ref+=1

            #If bin is not a bulk bin, go to different reference bin
            else:

                if (ix_ref==(self.NBins_x-1)) & (iy_ref==(self.NBins_y-1)):
                    break
                if ix_ref!=(self.NBins_x-1):
                    ix_ref+=1
                else:
                    ix_ref=0
                    iy_ref+=1

        # Bulk ID of largest bulk phase
        big_bulk_id = bulk_id[com_bulk_indx][com_bulk_indy]

        # Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest bulk
        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        return bulk_dict

    def separate_ints(self, phase_dict, count_dict, bulk_dict):
        '''
        Purpose: Takes the phase ids of each bin/particle and separates the interface bin into
        individual, isolated interfaces (separate clusters)

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        count_dict: dictionary containing the number of bins of each phase ['bulk' (0), 'int' (1), or 'gas' (2)]

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin
        (NBins_x, NBins_y) and each particle (partNum) after blurring. Bulk=0, Interface=1, Gas=2.

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Array (partNum) that identifies which bulk ID each particle belongs to
        phaseBulk = bulk_dict['part']

        # Array (NBins_x, NBins_y) that identifies which bulk ID each bin belongs to
        bulk_id = bulk_dict['bin']

        # Largest bulk phase bulk ID
        big_bulk_id = bulk_dict['largest id']

        # Number of interface bins
        int_num = count_dict['int']

        # Instantiate empty array (partNum) that identifies which separate bulk phase (of multiple
        # clusters) each bulk particle belongs to
        phaseInt=np.zeros(self.partNum)

        # Instantiate empty array (NBins_x, NBins_y) that identifies which separate bulk phase (of multiple
        # clusters) each bulk bin belongs to
        int_id=np.zeros((self.NBins_x, self.NBins_y), dtype=int)            #Label separate interfaces

        # Reference bin IDs
        ix_ref=0
        iy_ref=0

        # Current number of interface particles with current interface ID
        int_num_current = 0

        # Current interface ID
        int_id_current=0

        # List of possible interface IDs belonging to largest cluster
        possible_int_ids = []

        # Individually label each interface until all interface bins identified using flood fill algorithm

        # Loop over all interface bins until labeled
        while int_num_current!=int_num:

            #If bin is an interface, continue
            if phaseBin[ix_ref][iy_ref]==1:

                #If bin hadn't been assigned an interface id yet, continue
                if int_id[ix_ref][iy_ref]==0:

                    #Increase interface phase id
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

                        #loop through neighbor bins, including itself
                        for indx in lookx:
                            for indy in looky:

                                #If bin is an interface...
                                if phaseBin[indx][indy]==1:

                                    #If bin wasn't assigned an interface id...
                                    if int_id[indx][indy]==0:

                                        # Save neighboring interface bin IDs
                                        int_id_list.append([indx, indy])

                                        #Increase interface phase id
                                        int_num_current+=1

                                        # Assign current interface ID to neighboring interface bin, including itself
                                        int_id[indx][indy]=int_id_current

                                        # Loop over all particles in neighboring bin and label each particle's interface ID
                                        for h in range(0, len(self.binParts[indx][indy])):
                                            phaseInt[self.binParts[indx][indy][h]]=int_id_current

                                        # Count number of neighboring interface bins
                                        num_neigh_int+=1

                                #If bin is a gas, count it
                                elif phaseBin[indx][indy]==2:

                                    gas_num+=1

                                #else bin is counted as bulk
                                else:

                                    # If neighboring bin belongs to largest bulk...
                                    if bulk_id[indx][indy]==big_bulk_id:

                                        # Save current interface ID to be checked for largest interface later
                                        if int_id_current not in possible_int_ids:
                                            possible_int_ids.append(int_id_current)

                                    bulk_num+=1

                                    # Save neighboring bin's bulk ID
                                    bulk_id_list.append(bulk_id[indx, indy])

                    #If fewer than or equal to 4 neighboring interfaces, re-label phase as bulk or gas
                    if num_neigh_int<=4:

                        #If more neighboring gas bins, reference bin is truly a gas bin
                        if gas_num>bulk_num:

                            # Loop over all bins
                            for ix in range(0, self.NBins_x):
                                for iy in range(0, self.NBins_y):

                                    # If bin belongs to current interface ID...
                                    if int_id[ix][iy]==int_id_current:

                                        # Remove interface ID (0)
                                        int_id[ix][iy]=0

                                        # Label bin as gas phase
                                        phaseBin[ix][iy]=2

                                        # Loop over all particles in bin and remove interface ID (0)
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseInt[self.binParts[ix][iy][h]]=0

                                        # Loop over all particles in bin and remove bulk ID (0)
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseBulk[self.binParts[ix][iy][h]]=0

                        #Else if more neighboring bulk bins, reference bin is truly a bulk bin
                        else:

                            # Loop over all bins
                            for ix in range(0, self.NBins_x):
                                for iy in range(0, self.NBins_y):

                                    # If bin belongs to current interface ID...
                                    if int_id[ix][iy]==int_id_current:

                                        # Remove interface ID (0)
                                        int_id[ix][iy]=0

                                        # Label bin as bulk phase
                                        phaseBin[ix][iy]=0

                                        # Label bin as most common bulk ID of neighboring bins
                                        bulk_id[ix][iy]=mode(bulk_id_list)

                                        # Loop over all particles in neighboring bin and label as most common neighboring
                                        # bins' bulk ID
                                        for h in range(0, len(self.binParts[ix][iy])):
                                            phaseBulk[self.binParts[ix][iy][h]]=mode(bulk_id_list)

                                        # Loop over all particles in neighboring bin and remove interface ID (0)
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

        # Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest interface
        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        # Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest bulk
        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        # Dictionary containing arrays that identify the phase of each bin
        # (NBins_x, NBins_y) and each particle (partNum) after blurring.
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        return phase_dict, bulk_dict, int_dict

    def reduce_gas_noise(self, phase_dict, bulk_dict, int_dict):
        '''
        Purpose: Takes the phase ids and bulk/interface ids of each bin/particle
        and blurs the gas phase bins/particles that have bulk phase neighbors to
        be bulk bins/particles (since interface must separate the two if identified correctly)

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin
        (NBins_x, NBins_y) and each particle (partNum) after blurring. Bulk=0, Interface=1, Gas=2.

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Array (NBins_x, NBins_y) that identifies which bulk ID each bin belongs to
        bulk_id = bulk_dict['bin']

        # Array (partNum) that identifies which bulk ID each particle belongs to
        phaseBulk = bulk_dict['part']

        # Largest bulk phase bulk ID
        big_bulk_id = bulk_dict['largest id']

        # Instantiate empty array (NBins_x, NBins_y) that identifies which separate bulk phase (of multiple
        # clusters) each bulk bin belongs to
        int_id = int_dict['bin']

        # Instantiate empty array (partNum) that identifies which separate bulk phase (of multiple
        # clusters) each bulk particle belongs to
        phaseInt = int_dict['part']

        # List of possible interface IDs belonging to largest cluster
        possible_int_ids = int_dict['largest ids']

        #Label which interface each particle belongs to

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # If bin is gas...
                if (int_id[ix][iy] == 0) & (bulk_id[ix][iy]==0):

                    # Number of neighboring bins of each phase
                    bulk_num=0
                    gas_num=0

                    # List of neighboring bulk IDs
                    bulk_id_list=[]

                    # Find neighboring 2 shells of bins' x-indices
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

                    # Find neighboring 2 shells of bins' y-indices
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

                    # Loop over all neighboring bins
                    for indx in lookx:
                        for indy in looky:

                            # If neighboring bin is bulk, count it and save its bulk ID
                            if phaseBin[indx][indy]==0:
                                bulk_num+=1
                                bulk_id_list.append(bulk_id[indx, indy])

                            # If neighboring bin is gas, count it
                            elif phaseBin[indx][indy]==2:
                                gas_num+=1

                    # If more bulk neighbors than gas, make reference bin bulk with most common bulk ID of neighbors
                    if bulk_num>=gas_num:
                        phaseBin[ix][iy]=0
                        bulk_id[ix][iy]=mode(bulk_id_list)

                    # Otherwise, keep it gas
                    else:
                        phaseBin[ix][iy]=2

                    # Loop over all particles in bin and update bulk ID and phase
                    for h in range(0, len(self.binParts[ix][iy])):
                        phaseBulk[self.binParts[ix][iy][h]]=bulk_id[ix][iy]
                        phasePart[self.binParts[ix][iy][h]]=phaseBin[ix][iy]

        # Dictionary containing arrays that identify the phase of each bin
        # (NBins_x, NBins_y) and each particle (partNum) after blurring.
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        # Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest bulk
        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        # Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest interface
        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        return phase_dict, bulk_dict, int_dict

    def int_comp(self, part_dict, phase_dict, bulk_dict, int_dict):
        '''
        Purpose: Takes the phase, bulk, and interface ids and identifies the sufficiently
        large interfaces and calculates the composition of each particle type
        ('all', 'A', or 'B') for each interface ID

        Inputs:
        part_dict: dictionary of binned particle ids and cluster information

        phase_dict: dictionary of arrays labeling phase of each bin and particle

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface

        Outputs:
        phase_dict: dictionary containing arrays that identify the phase of each bin
        (NBins_x, NBins_y) and each particle (partNum) after blurring. Bulk=0, Interface=1, Gas=2.

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface

        int_comp_dict: Dictionary containing information of sigificantly large interface IDs and their
        composition of each particle type ('all', 'A', or 'B')
        '''

        # Array (partNum) that identifies the particle types ('A'=0, 'B'=1)
        partTyp = self.typ

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Array (NBins_x, NBins_y) that identifies which bulk ID each bin belongs to
        bulk_id = bulk_dict['bin']

        # Array (partNum) that identifies which bulk ID each particle belongs to
        phaseBulk = bulk_dict['part']

        # Largest bulk phase bulk ID
        big_bulk_id = bulk_dict['largest id']

        # Instantiate empty array (NBins_x, NBins_y) that identifies which separate bulk phase (of multiple
        # clusters) each bulk bin belongs to
        int_id = int_dict['bin']

        # Instantiate empty array (partNum) that identifies which separate bulk phase (of multiple
        # clusters) each bulk particle belongs to
        phaseInt = int_dict['part']

        # List of possible interface IDs belonging to largest cluster
        possible_int_ids = int_dict['largest ids']

        # Initiate empty array of number of particles of each type ('all', 'A', 'B') for each interface id
        int_A_comp = np.array([])
        int_B_comp = np.array([])
        int_comp = np.array([])

        # Number of too small and sufficiently large interfaces
        int_small_num=0
        int_large_num=0

        # Initiate empty array containing interface IDs corresponding to interfaces of sufficient size
        int_large_ids=np.array([])

        # Initiate empty boolean array containing identifying whether interface is of sufficient size (1) or not (0)
        if_large_int=[]

        # Largest interface ID to iterate over
        max_int_id = int(np.max(phaseInt))

        #Determine which grouping of particles (phases or different interfaces) are large enough to perform measurements on or if noise

        # Loop over all interface IDs
        for m in range(1, max_int_id+1):

            #Find which particles belong to interface ID of 'm'
            int_id_part = np.where(phaseInt==m)[0]

            # Number of particles of interface ID
            int_id_part_num = len(int_id_part)

            # Number of bins of interface ID
            int_id_bin_num=0

            # Loop over all bins to count number of bins of interface ID of 'm'
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if int_id[ix][iy]==m:
                        int_id_bin_num +=1

            #If fewer than 100 particles belong to group 'm', then it is most likely noise and we should remove it
            if (int_id_part_num<=100) or (int_id_bin_num<10):

                # Increase number of too small of interfaces
                int_small_num+=1

                # Remove interface ID for all particles with 'm' interface ID
                phaseInt[int_id_part]=0

                # Loop over all bins
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):

                        # Number of neighboring bins of each phase and their respective bulk ID
                        bulk_id_list = []
                        gas_num=0
                        bulk_num=0

                        # If reference bin is of interface ID of 'm'
                        if int_id[ix][iy]==m:

                            # If 'm' is a possible largest interface ID, remove it since it's a small interface
                            if m in possible_int_ids:
                                possible_int_ids.remove(m)

                            # Find neighboring 2 shells of bins' x-indices
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

                            # Find neighboring 2 shells of bins' y-indices
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

                            # Loop over neighboring bins
                            for indx in lookx:
                                for indy in looky:

                                    # If neighboring bin is bulk, count it and save bulk ID
                                    if phaseBin[indx][indy]==0:
                                        bulk_num+=1
                                        bulk_id_list.append(bulk_id[indx, indy])

                                    # If neighboring bin is gas, count it
                                    elif phaseBin[indx][indy]==2:
                                        gas_num+=1

                            # Remove interface ID of reference bin
                            int_id[ix][iy]=0

                            # Loop over particles in reference bin and remove interface ID
                            for h in range(0, len(self.binParts[ix][iy])):
                                phaseInt[self.binParts[ix][iy][h]]=0

                            # If more gas neighboring bins than bulk, label reference bin and its particles as gas
                            if gas_num>bulk_num:
                                phaseBin[ix][iy]=2
                                if len(self.binParts[ix][iy])>0:
                                    for h in range(0, len(self.binParts[ix][iy])):
                                        phasePart[self.binParts[ix][iy][h]]=2

                            # If more bulk neighboring bins than gas, label reference bin and its particles
                            # as bulk with most common bulk ID among its neighboring bins

                            else:
                                phaseBin[ix][iy]=0
                                if len(self.binParts[ix][iy])>0:
                                    for h in range(0, len(self.binParts[ix][iy])):
                                        phasePart[self.binParts[ix][iy][h]]=0
                                        phaseBulk[self.binParts[ix][iy][h]]=mode(bulk_id_list)

            #If more than 100 particles belong to interface ID of 'm', then it is most likely significant and we should account for it
            else:

                #Label if structure is bulk/gas or interface
                if len(np.where(phasePart[int_id_part]==0)[0])==0:

                    #Calculate composition of each type of particles ('all', 'A', or 'B') in interface ID of 'm'
                    int_A_comp = np.append(int_A_comp, len(np.where((phaseInt==m) & (partTyp==0))[0]))
                    int_B_comp = np.append(int_B_comp, len(np.where((phaseInt==m) & (partTyp==1))[0]))
                    int_comp = np.append(int_comp, len(np.where((phaseInt==m) & (partTyp==1))[0])+len(np.where((phaseInt==m) & (partTyp==0))[0]))

                    # Label interface as sufficiently large (1)
                    if_large_int.append(1)

                    # Save ID of sufficiently large interface of ID 'm'
                    int_large_ids = np.append(int_large_ids, m)

                    #Count number of significant structures
                    int_large_num+=1

        # Dictionary containing arrays that identify the phase of each bin
        # (NBins_x, NBins_y) and each particle (partNum) after blurring.
        phase_dict = {'bin': phaseBin, 'part': phasePart}

        # Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest bulk
        bulk_dict = {'bin': bulk_id, 'part': phaseBulk, 'largest id': big_bulk_id}

        # Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        # each particle (partNum) along with the ID of the largest interface
        int_dict = {'bin': int_id, 'part': phaseInt, 'largest ids': possible_int_ids}

        # Dictionary containing information of sigificantly large interface IDs and their
        # composition of each particle type ('all', 'A', or 'B')
        int_comp_dict = {'ids': int_large_ids, 'if large': if_large_int, 'comp': {'all': int_comp, 'A': int_A_comp, 'B': int_B_comp}}

        return phase_dict, bulk_dict, int_dict, int_comp_dict

    def bulk_comp(self, part_dict, phase_dict, bulk_dict):
        '''
        Purpose: Takes the phase, bulk, and interface ids and identifies the sufficiently
        large bulk phases and calculates the composition of each particle type
        ('all', 'A', or 'B') for each bulk ID

        Inputs:
        part_dict: dictionary of binned particle ids and cluster information

        phase_dict: dictionary of arrays labeling phase of each bin and particle

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        Outputs:
        bulk_comp_dict: Dictionary containing information of sigificantly large bulk IDs and their
        composition of each particle type ('all', 'A', or 'B')
        '''

        # Array (partNum) that identifies the particle types ('A'=0, 'B'=1)
        partTyp = self.typ

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Array (NBins_x, NBins_y) that identifies which bulk ID each bin belongs to
        bulk_id = bulk_dict['bin']

        # Array (partNum) that identifies which bulk ID each particle belongs to
        phaseBulk = bulk_dict['part']

        # Largest bulk phase bulk ID
        big_bulk_id = bulk_dict['largest id']

        # Initiate empty array of number of particles of each type ('all', 'A', 'B') for each bulk id
        bulk_A_comp = np.array([])
        bulk_B_comp = np.array([])
        bulk_comp = np.array([])

        # Number of sufficiently large bulk
        bulk_large_num=0

        # Initiate empty array containing bulk IDs corresponding to bulk phases of sufficient size
        bulk_large_ids = np.array([])

        # Initiate empty boolean array containing identifying whether bulk phase is of sufficient size (1) or not (0)
        if_large_bulk = []

        # Largest bulk ID to iterate over
        max_bulk_id = int(np.max(phaseBulk))

        #Calculate composition of each bulk phase structure

        # Loop over all bulk IDs
        for m in range(1, max_bulk_id+1):

            #Find which particles belong to bulk ID of 'm'
            bulk_id_part = np.where(phaseBulk==m)[0]

            # Number of particles of bulk ID
            bulk_id_part_num = len(bulk_id_part)

            # Number of bins of bulk ID
            bulk_id_bin_num=0

            # Loop over all bins to count number of bins of bulk ID of 'm'
            for ix in range(0, self.NBins_x):
                for iy in range(0, self.NBins_y):
                    if bulk_id[ix][iy]==m:
                        bulk_id_bin_num +=1

            # If at least 1 particles belong to bulk ID of 'm', then it is most
            # likely a bulk due to our previous noise mitigation techniques
            if bulk_id_part_num>0:

                # Label bulk ID as sufficiently large
                if_large_bulk.append(1)

                #Calculate composition of particles in each bulk phase for each particle type ('all', 'A', or 'B')
                bulk_A_comp = np.append(bulk_A_comp, len(np.where((phaseBulk==m) & (partTyp==0))[0]))
                bulk_B_comp = np.append(bulk_B_comp, len(np.where((phaseBulk==m) & (partTyp==1))[0]))
                bulk_comp = np.append(bulk_comp, len(np.where((phaseBulk==m) & (partTyp==1))[0])+len(np.where((phaseBulk==m) & (partTyp==0))[0]))

                # Label sufficiently large bulk IDs
                bulk_large_ids = np.append(bulk_large_ids, m)

                # Count number of sufficiently large bulk phases
                bulk_large_num+=1

        # Dictionary containing information of sigificantly large bulk IDs and their
        # composition of each particle type ('all', 'A', or 'B')
        bulk_comp_dict = {'ids': bulk_large_ids, 'if large': if_large_bulk, 'comp': {'all': bulk_comp, 'A': bulk_A_comp, 'B': bulk_B_comp}}

        return bulk_comp_dict

    def bulk_sort2(self, bulk_comp_dict):
        '''
        Purpose: Takes the composition and other identifying information of
        significantly large bulk IDs and sorts them according size from largest
        bulk phase to smallest bulk phase

        Inputs:
        bulk_comp_dict: Dictionary containing information of sigificantly large bulk IDs and their
        composition of each particle type ('all', 'A', or 'B')

        Outputs:
        bulk_comp_dict: Dictionary containing information of sigificantly large bulk IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest bulk phase to smallest
        bulk phase
        '''

        # Array containing bulk IDs corresponding to bulk phases of sufficient size
        bulk_large_ids = bulk_comp_dict['ids']

        # Boolean array containing identifying whether bulk phase is of sufficient size (1) or not (0)
        if_large_bulk = bulk_comp_dict['if large']

        #Composition of particles in each bulk phase for each particle type ('all', 'A', or 'B')
        bulk_comp = bulk_comp_dict['comp']['all']
        bulk_A_comp = bulk_comp_dict['comp']['A']
        bulk_B_comp = bulk_comp_dict['comp']['B']

        # Sorted array containing bulk IDs corresponding to bulk phases of sufficient size from largest to smallest in size
        bulk_large_ids = [x for _,x in sorted(zip(bulk_comp, bulk_large_ids))]

        # Sorted boolean array containing identifying whether bulk phase is of sufficient size (1) or not (0) from largest to smallest in size
        if_large_bulk = [x for _,x in sorted(zip(bulk_comp, if_large_bulk))]

        #Sorted composition of particles in each bulk phase for each particle type ('all', 'A', or 'B') from largest to smallest in size
        bulk_A_comp = [x for _,x in sorted(zip(bulk_comp, bulk_A_comp))]
        bulk_B_comp = [x for _,x in sorted(zip(bulk_comp, bulk_B_comp))]
        bulk_comp = sorted(bulk_comp)

        # If 5 bulk phases identified, then use all 5 for output arrays
        if len(bulk_comp)>5:
            bulk_comp = bulk_comp[:5]
            bulk_A_comp = bulk_A_comp[:5]
            bulk_B_comp = bulk_B_comp[:5]

            bulk_large_ids = bulk_large_ids[:5]
            if_large_bulk = if_large_bulk[:5]

        # Otherwise, also append empty values so final output arrays are of length 5
        elif len(bulk_comp)<5:
            dif_len = int(5 - len(bulk_comp))
            for i in range(0, dif_len):
                bulk_comp.append(0)
                bulk_A_comp.append(0)
                bulk_B_comp.append(0)
                if_large_bulk.append(0)
                bulk_large_ids.append(999)

        # Dictionary containing information of sigificantly large bulk IDs and their
        # composition of each particle type ('all', 'A', or 'B') sorted from largest
        # to smallest bulk phase
        bulk_comp_dict = {'ids': bulk_large_ids, 'if large': if_large_bulk, 'comp': {'all': bulk_comp, 'A': bulk_A_comp, 'B': bulk_B_comp}}

        return bulk_comp_dict

    def int_sort2(self, int_comp_dict):
        '''
        Purpose: Takes the composition and other identifying information of
        significantly large interface IDs and sorts them according size from largest
        interface to smallest interface

        Inputs:
        int_comp_dict: Dictionary containing information of sigificantly large interface IDs and their
        composition of each particle type ('all', 'A', or 'B')

        Outputs:
        int_comp_dict: Dictionary containing information of sigificantly large interface IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest interface to smallest
        interface
        '''

        # Array containing interface IDs corresponding to interfaces of sufficient size
        int_large_ids = int_comp_dict['ids']

        # Boolean array containing identifying whether interface is of sufficient size (1) or not (0)
        if_large_int = int_comp_dict['if large']

        #Composition of particles in each interface for each particle type ('all', 'A', or 'B')
        int_comp = int_comp_dict['comp']['all']
        int_A_comp = int_comp_dict['comp']['A']
        int_B_comp = int_comp_dict['comp']['B']

        # Sorted array containing interface IDs corresponding to interfaces of sufficient size from largest to smallest in size
        int_large_ids = [x for _,x in sorted(zip(int_comp, int_large_ids))]

        # Sorted boolean array containing identifying whether interfaces is of sufficient size (1) or not (0) from largest to smallest in size
        if_large_int = [x for _,x in sorted(zip(int_comp, if_large_int))]

        #Sorted composition of particles in each interface for each particle type ('all', 'A', or 'B') from largest to smallest in size
        int_A_comp = [x for _,x in sorted(zip(int_comp, int_A_comp))]
        int_B_comp = [x for _,x in sorted(zip(int_comp, int_B_comp))]
        int_comp = sorted(int_comp)

        # If 5 interfaces identified, then use all 5 for output arrays
        if len(int_comp)>5:
            int_comp = int_comp[:5]
            int_A_comp = int_A_comp[:5]
            int_B_comp = int_B_comp[:5]

            int_large_ids = int_large_ids[:5]
            if_large_int = if_large_int[:5]

        # Otherwise, also append empty values so final output arrays are of length 5
        elif len(int_comp)<5:
            dif_len = int(5 - len(int_comp))

            for i in range(0, dif_len):
                int_comp.append(0)
                int_A_comp.append(0)
                int_B_comp.append(0)
                if_large_int.append(0)
                int_large_ids.append(999)

        # Dictionary containing information of sigificantly large interface IDs and their
        # composition of each particle type ('all', 'A', or 'B') sorted from largest
        # to smallest interface
        int_comp_dict = {'ids': int_large_ids, 'if large': if_large_int, 'comp': {'all': int_comp, 'A': int_A_comp, 'B': int_B_comp}}

        return int_comp_dict

    def phase_sort(self, comp_dict):
        '''
        Purpose: Takes the composition and other identifying information of
        significantly large phase IDs and sorts them according size from largest
        phase to smallest phase according to input dictionary

        Inputs:
        phase_comp_dict: Dictionary containing information of sigificantly large phase IDs and their
        composition of each particle type ('all', 'A', or 'B') for either bulk or interface

        Outputs:
        phase_comp_dict: Dictionary containing information of sigificantly large phase IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest phase to smallest
        phase for either bulk or interface depending on input phase
        '''

        # Array containing interface IDs corresponding to phase of sufficient size
        large_ids = comp_dict['ids']

        # Boolean array containing identifying whether phase is of sufficient size (1) or not (0)
        if_large = comp_dict['if large']

        #Composition of particles in each phase for each particle type ('all', 'A', or 'B')
        comp = comp_dict['comp']['all']
        A_comp = comp_dict['comp']['A']
        B_comp = comp_dict['comp']['B']

        # Sorted array containing interface IDs corresponding to phase of sufficient size from largest to smallest in size
        large_ids = [x for _,x in sorted(zip(comp, large_ids))]

        # Sorted boolean array containing identifying whether phase is of sufficient size (1) or not (0) from largest to smallest in size
        if_large = [x for _,x in sorted(zip(comp, if_large))]

        #Sorted composition of particles in each phase for each particle type ('all', 'A', or 'B') from largest to smallest in size
        A_comp = [x for _,x in sorted(zip(comp, A_comp))]
        B_comp = [x for _,x in sorted(zip(comp, B_comp))]
        comp = sorted(comp)

        # If 5 phases identified, then use all 5 for output arrays
        if len(comp)>5:
            comp = comp[:5]
            A_comp = A_comp[:5]
            B_comp = B_comp[:5]

            large_ids = large_ids[:5]
            if_large = if_large[:5]

        # Otherwise, also append empty values so final output arrays are of length 5
        elif len(comp)<5:
            dif_len = int(5 - len(comp))

            for i in range(0, dif_len):
                comp.append(0)
                A_comp.append(0)
                B_comp.append(0)
                if_large.append(0)
                large_ids.append(999)

        # Dictionary containing information of sigificantly large phase IDs and their
        # composition of each particle type ('all', 'A', or 'B') sorted from largest
        # to smallest interface
        comp_dict = {'ids': large_ids, 'if large': if_large, 'comp': {'all': comp, 'A': A_comp, 'B': B_comp}}

        return comp_dict

    def phase_part_count(self, phase_dict, int_dict, int_comp_dict, bulk_dict, bulk_comp_dict, typ):
        '''
        Purpose: Takes the phase, bulk, and interface ids and counts the number of each particle type
        ('all', 'A', or 'B') for each total phase, the largest phase ID of the bulk and interface, and
        all other, smaller phase IDs of the bulk and interface

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface

        int_comp_dict: Dictionary containing information of sigificantly large interface IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest interface to smallest
        interface

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        bulk_comp_dict: Dictionary containing information of sigificantly large bulk IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest bulk phase to smallest
        bulk phase or unsorted

        Outputs:
        part_count_dict: Dictionary containing the number of particles (int) of each type
        ('all', 'A', or 'B') within each phase

        part_id_dict: Dictionary containing the arrays of particle ids of each type
        ('all', 'A', or 'B') within each phase
        '''

        # Array (partNum) that identifies whether particle is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        # Array (partNum) that identifies which bulk ID each particle belongs to
        phaseBulk = bulk_dict['part']

        # Array containing bulk IDs corresponding to bulk phases of sufficient size
        bulk_large_ids = bulk_comp_dict['ids']

        # Instantiate empty array (partNum) that identifies which separate bulk phase (of multiple
        # clusters) each bulk particle belongs to
        phaseInt = int_dict['part']

        # Array containing interface IDs corresponding to interfaces of sufficient size
        int_large_ids = int_comp_dict['ids']

        #Number of particles in all bulk phases for each particle type ('all', 'A', or 'B')
        bulk_num = len(np.where((phasePart==0))[0])
        bulk_A_num = len(np.where((phasePart==0) & (typ==0))[0])
        bulk_B_num = len(np.where((phasePart==0) & (typ==1))[0])

        #Number of particles in the largest bulk phases for each particle type ('all', 'A', or 'B')
        largest_bulk_num = len(np.where((phaseBulk==bulk_large_ids[0]))[0])
        largest_bulk_A_num = len(np.where((phaseBulk==bulk_large_ids[0]) & (typ==0))[0])
        largest_bulk_B_num = len(np.where((phaseBulk==bulk_large_ids[0]) & (typ==1))[0])

        #Number of particles in all interfaces for each particle type ('all', 'A', or 'B')
        int_num = len(np.where((phasePart==1))[0])
        int_A_num = len(np.where((phasePart==1) & (typ==0))[0])
        int_B_num = len(np.where((phasePart==1) & (typ==1))[0])

        #Number of particles in the largest interface for each particle type ('all', 'A', or 'B')
        largest_int_num = len(np.where((phaseInt==int_large_ids[0]))[0])
        largest_int_A_num = len(np.where((phaseInt==int_large_ids[0]) & (typ==0))[0])
        largest_int_B_num = len(np.where((phaseInt==int_large_ids[0]) & (typ==1))[0])

        #Number of particles in all gas phases for each particle type ('all', 'A', or 'B')
        gas_num = len(np.where((phasePart==2))[0])
        gas_A_num = len(np.where((phasePart==2) & (typ==0))[0])
        gas_B_num = len(np.where((phasePart==2) & (typ==1))[0])

        #IDs of particles in all bulk phases for each particle type ('all', 'A', or 'B')
        bulk_part_ids = np.where(phasePart==0)[0]        #Bulk phase structure(s)
        bulk_A_part_ids = np.where((phasePart==0) & (typ==0))[0]        #Bulk phase structure(s)
        bulk_B_part_ids = np.where((phasePart==0) & (typ==1))[0]        #Bulk phase structure(s)

        #IDs of particles in all interfaces for each particle type ('all', 'A', or 'B')
        int_part_ids = np.where(phasePart==1)[0]     #Largest gas-dense interface
        int_A_part_ids = np.where((phasePart==1) & (typ==0))[0]        #Bulk phase structure(s)
        int_B_part_ids = np.where((phasePart==1) & (typ==1))[0]        #Bulk phase structure(s)

        #IDs of particles in all gas phases for each particle type ('all', 'A', or 'B')
        gas_part_ids = np.where(phasePart==2)[0]              #Gas phase structure(s)
        gas_A_part_ids = np.where((phasePart==2) & (typ==0))[0]        #Bulk phase structure(s)
        gas_B_part_ids = np.where((phasePart==2) & (typ==1))[0]        #Bulk phase structure(s)

        #ID of particles in largest interface for each particle type ('all', 'A', or 'B')
        largest_int_part_ids = np.where(phaseInt==int_large_ids[0])[0]     #Largest gas-dense interface
        largest_int_A_part_ids = np.where((phaseInt==int_large_ids[0]) & (typ==0))[0]     #Largest gas-dense interface
        largest_int_B_part_ids = np.where((phaseInt==int_large_ids[0]) & (typ==1))[0]     #Largest gas-dense interface

        #IDs of particles in largest bulk phase for each particle type ('all', 'A', or 'B')
        largest_bulk_part_ids = np.where(phaseBulk==bulk_large_ids[0])[0]     #Largest gas-dense interface
        largest_bulk_A_part_ids = np.where((phaseBulk==bulk_large_ids[0]) & (typ==0))[0]     #Largest gas-dense interface
        largest_bulk_B_part_ids = np.where((phaseBulk==bulk_large_ids[0]) & (typ==1))[0]     #Largest gas-dense interface

        #IDs of particles in all interfaces besides the largest for each particle type ('all', 'A', or 'B')
        small_int_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0))[0]
        small_int_A_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0) & (typ==0))[0]
        small_int_B_part_ids = np.where((phaseInt!=int_large_ids[0]) & (phaseInt!=0) & (typ==1))[0]

        #IDs of particles in all bulk phases besides the largest for each particle type ('all', 'A', or 'B')
        small_bulk_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0))[0]
        small_bulk_A_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0) & (typ==0))[0]     #Largest gas-dense interface
        small_bulk_B_part_ids = np.where((phaseBulk!=bulk_large_ids[0]) & (phaseBulk!=0) & (typ==1))[0]     #Largest gas-dense interface

        # Dictionary containing the number of particles (int) of each type ('all', 'A', or 'B') within each phase
        part_count_dict = {'bulk': {'all': bulk_num, 'A': bulk_A_num, 'B': bulk_B_num}, 'largest bulk': {'all': largest_bulk_num, 'A': largest_bulk_A_num, 'B': largest_bulk_B_num}, 'int': {'all': int_num, 'A': int_A_num, 'B': int_B_num}, 'largest int': {'all': largest_int_num, 'A': largest_int_A_num, 'B': largest_int_B_num}, 'gas': {'all': gas_num, 'A': gas_A_num, 'B': gas_B_num}}

        # Dictionary containing the arrays of particle ids of each type ('all', 'A', or 'B') within each phase
        part_id_dict = {'bulk': {'all': bulk_part_ids, 'A': bulk_A_part_ids, 'B': bulk_B_part_ids}, 'largest bulk': {'all': largest_bulk_part_ids, 'A': largest_bulk_A_part_ids, 'B': largest_bulk_B_part_ids}, 'small bulk': {'all': small_bulk_part_ids, 'A': small_bulk_A_part_ids, 'B': small_bulk_B_part_ids}, 'int': {'all': int_part_ids, 'A': int_A_part_ids, 'B': int_B_part_ids}, 'largest int': {'all': largest_int_part_ids, 'A': largest_int_A_part_ids, 'B': largest_int_B_part_ids}, 'small int': {'all': small_int_part_ids, 'A': small_int_A_part_ids, 'B': small_int_B_part_ids}, 'gas': {'all': gas_part_ids, 'A': gas_A_part_ids, 'B': gas_B_part_ids}}

        return part_count_dict, part_id_dict

    def phase_bin_count(self, phase_dict, bulk_dict, int_dict, bulk_comp_dict, int_comp_dict):
        '''
        Purpose: Takes the phase, bulk, and interface ids and counts the number of bins of each total phase,
        the largest phase ID of the bulk and interface, and all other, smaller phase IDs of the bulk and interface

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        bulk_dict: Dictionary containing arrays of bulk IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest bulk

        int_dict: Dictionary containing arrays of interface IDs for each bin (NBins_x, NBins_y) and
        each particle (partNum) along with the ID of the largest interface

        bulk_comp_dict: Dictionary containing information of sigificantly large bulk IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest bulk phase to smallest
        bulk phase or unsorted

        int_comp_dict: Dictionary containing information of sigificantly large interface IDs and their
        composition of each particle type ('all', 'A', or 'B') sorted from largest interface to smallest
        interface

        Outputs:
        bin_count_dict: Dictionary containing the number of bins (int) for each phase and each individual bulk
        and interface
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (NBins_x, NBins_y) that identifies which bulk ID each bin belongs to
        bulk_id = bulk_dict['bin']

        # Array containing bulk IDs corresponding to bulk phases of sufficient size
        bulk_large_ids = bulk_comp_dict['ids']

        # Array containing bulk IDs corresponding to bulk phases of sufficient size
        if_large_bulk = bulk_comp_dict['if large']

        # Array (NBins_x, NBins_y) that identifies which interface ID each bin belongs to
        int_id = int_dict['bin']

        # Array containing bulk IDs corresponding to interfaces of sufficient size
        int_large_ids = int_comp_dict['ids']

        # Array containing interface IDs corresponding to interfaces of sufficient size
        if_large_int = int_comp_dict['if large']

        #Initiate counts of number of bins of each phase
        bulk_num=0
        gas_num=0
        all_int_num=0
        largest_int_num=0

        # Initialize empty arrays for saving number of bins of each individual interface and bulk phase
        int_num_arr = np.zeros(len(int_large_ids))
        bulk_num_arr = np.zeros(len(bulk_large_ids))

        #Count number of bins that belong to each phase
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

        #Count number of bins belonging to each separate interface
        for m in range(0, len(int_large_ids)):
            if if_large_int[m]!=0:
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):
                        if int_id[ix][iy] == int_large_ids[m]:
                            int_num_arr[m] +=1

        #Count number of bins belonging to each separate bulk phase
        for m in range(0, len(bulk_large_ids)):
            if if_large_bulk[m]!=0:
                for ix in range(0, self.NBins_x):
                    for iy in range(0, self.NBins_y):
                        if bulk_id[ix][iy] == bulk_large_ids[m]:
                            bulk_num_arr[m] +=1

        # Dictionary containing the number of bins for each phase and each individual bulk and interface
        bin_count_dict = {'bin': {'bulk': bulk_num, 'largest int': largest_int_num, 'gas': gas_num, 'all int': all_int_num}, 'ids': {'int': int_num_arr, 'bulk': bulk_num_arr}}

        return bin_count_dict
    """
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
    """
