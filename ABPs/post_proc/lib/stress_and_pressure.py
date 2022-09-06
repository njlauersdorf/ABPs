
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

class stress_and_pressure:
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

        self.binning = binning.binning(self.l_box, self.partNum, self.NBins, self.peA, self.peB, self.ang)

        self.plotting_utility = plotting_utility.plotting_utility(self.l_box, self.partNum, self.typ)

        self.phase_ident = phase_identification.phase_identification(self.area_frac_dict, self.align_dict, self.part_dict, self.press_dict, self.l_box, self.partNum, self.NBins, self.peA, self.peB, self.parFrac, self.eps, self.typ)

    def interparticle_stress(self):

        phase_dict, part_dict = self.binning.decrease_bin_size(self.phase_dict['bin'], self.phase_dict['part'], self.part_dict['id'], self.pos, self.typ)

        NBins = len(phase_dict['bin'])
        binParts = part_dict['id']
        phaseBin = phase_dict['bin']

        SigXX_part = np.zeros(self.partNum)
        SigYY_part = np.zeros(self.partNum)
        SigXY_part = np.zeros(self.partNum)
        SigYX_part = np.zeros(self.partNum)

        SigXX_bin = np.zeros((NBins, NBins))
        SigYY_bin = np.zeros((NBins, NBins))
        SigXY_bin = np.zeros((NBins, NBins))
        SigYX_bin = np.zeros((NBins, NBins))

        bulkSigXX_all = 0
        bulkSigYY_all = 0
        bulkSigXY_all = 0
        bulkSigYX_all = 0

        bulkSigXX_A = 0
        bulkSigYY_A = 0
        bulkSigXY_A = 0
        bulkSigYX_A = 0

        bulkSigXX_B = 0
        bulkSigYY_B = 0
        bulkSigXY_B = 0
        bulkSigYX_B = 0

        intSigXX_all = 0
        intSigYY_all = 0
        intSigXY_all = 0
        intSigYX_all = 0

        intSigXX_A = 0
        intSigYY_A = 0
        intSigXY_A = 0
        intSigYX_A = 0

        intSigXX_B = 0
        intSigYY_B = 0
        intSigXY_B = 0
        intSigYX_B = 0

        gasSigXX_all = 0
        gasSigYY_all = 0
        gasSigXY_all = 0
        gasSigYX_all = 0

        gasSigXX_A = 0
        gasSigYY_A = 0
        gasSigXY_A = 0
        gasSigYX_A = 0

        gasSigXX_B = 0
        gasSigYY_B = 0
        gasSigXY_B = 0
        gasSigYX_B = 0

        theory_functs = theory.theory()
        for ix in range(0, NBins):
            for iy in range(0, NBins):

                # If at right edge, wrap to left
                if (ix + 1) == NBins:
                    lookx = [ix-1, ix, 0]
                elif (ix) == 0:
                    lookx = [NBins-1, ix, ix+1]
                else:
                    lookx = [ix-1, ix, ix+1]

                # If at right edge, wrap to left
                if (iy + 1) == NBins:
                    looky = [iy-1, iy, 0]
                elif (iy) == 0:
                    looky = [NBins-1, iy, iy+1]
                else:
                    looky = [iy-1, iy, iy+1]

                # Loop through surrounding x-index
                for indx in lookx:
                    # Loop through surrounding y-index
                    for indy in looky:
                        if len(binParts[ix][iy])>0:
                            # Loop through all reference particles in ix,iy bin
                            for h in binParts[ix][iy]:
                                # Loop through all neighboring particles in indx,indy bin
                                for j in binParts[ix][iy]:

                                    difx = self.utility_functs.sep_dist(self.pos[h][0], self.pos[j][0])

                                    dify = self.utility_functs.sep_dist(self.pos[h][1], self.pos[j][1])

                                    difr = ( (difx)**2 + (dify)**2)**0.5

                                    # If potential is on ...
                                    if 0.1 < difr <= self.r_cut:
                                        # Compute the x and y components of force
                                        fx, fy = theory_functs.computeFLJ(difr, self.pos[h][0], self.pos[h][1], self.pos[j][0], self.pos[j][1], self.eps)

                                        SigXX = (fx * difr)
                                        SigYY = (fy * dify)
                                        SigXY = (fx * dify)
                                        SigYX = (fy * difx)

                                        SigXX_part[h] += SigXX
                                        SigYY_part[h] += SigYY
                                        SigXY_part[h] += SigXY
                                        SigYX_part[h] += SigYX

                                        SigXX_bin[ix][iy] += SigXX
                                        SigYY_bin[ix][iy] += SigYY
                                        SigXY_bin[ix][iy] += SigXY
                                        SigYX_bin[ix][iy] += SigYX

                                        if phaseBin[ix][iy] == 0:

                                            bulkSigXX_all += SigXX
                                            bulkSigYY_all += SigYY
                                            bulkSigXY_all += SigXY
                                            bulkSigYX_all += SigYX

                                            if self.typ[j] == 0:
                                                bulkSigXX_A += SigXX
                                                bulkSigYY_A += SigYY
                                                bulkSigXY_A += SigXY
                                                bulkSigYX_A += SigYX
                                            else:
                                                bulkSigXX_B += SigXX
                                                bulkSigYY_B += SigYY
                                                bulkSigXY_B += SigXY
                                                bulkSigYX_B += SigYX

                                        elif phaseBin[ix][iy] == 1:

                                            intSigXX_all += SigXX
                                            intSigYY_all += SigYY
                                            intSigXY_all += SigXY
                                            intSigYX_all += SigYX

                                            if self.typ[j] == 0:
                                                intSigXX_A += SigXX
                                                intSigYY_A += SigYY
                                                intSigXY_A += SigXY
                                                intSigYX_A += SigYX
                                            else:
                                                intSigXX_B += SigXX
                                                intSigYY_B += SigYY
                                                intSigXY_B += SigXY
                                                intSigYX_B += SigYX

                                        else:

                                            gasSigXX_all += SigXX
                                            gasSigYY_all += SigYY
                                            gasSigXY_all += SigXY
                                            gasSigYX_all += SigYX

                                            if self.typ[j] == 0:
                                                gasSigXX_A += SigXX
                                                gasSigYY_A += SigYY
                                                gasSigXY_A += SigXY
                                                gasSigYX_A += SigYX
                                            else:
                                                gasSigXX_B += SigXX
                                                gasSigYY_B += SigYY
                                                gasSigXY_B += SigXY
                                                gasSigYX_B += SigYX

        stress_dict = {'bin': {'XX': SigXX_bin, 'XY': SigXY_bin, 'YX': SigYX_bin, 'YY': SigYY_bin}, 'part': {'XX': SigXX_part, 'XY': SigXY_part, 'YX': SigYX_bin, 'YY': SigYY_bin}, 'tot': {'bulk': {'all': {'XX': bulkSigXX_all, 'XY': bulkSigXY_all, 'YX': bulkSigYX_all, 'YY': bulkSigYY_all}, 'A': {'XX': bulkSigXX_A, 'XY': bulkSigXY_A, 'YX': bulkSigYX_A, 'YY': bulkSigYY_A}, 'B': {'XX': bulkSigXX_B, 'XY': bulkSigXY_B, 'YX': bulkSigYX_B, 'YY': bulkSigYY_B}}, 'int': {'all': {'XX': intSigXX_all, 'XY': intSigXY_all, 'YX': intSigYX_all, 'YY': intSigYY_all}, 'A': {'XX': intSigXX_A, 'XY': intSigXY_A, 'YX': intSigYX_A, 'YY': intSigYY_A}, 'B': {'XX': intSigXX_B, 'XY': intSigXY_B, 'YX': intSigYX_B, 'YY': intSigYY_B}}, 'gas': {'all': {'XX': gasSigXX_all, 'XY': gasSigXY_all, 'YX': gasSigYX_all, 'YY': gasSigYY_all}, 'A': {'XX': gasSigXX_A, 'XY': gasSigXY_A, 'YX': gasSigYX_A, 'YY': gasSigYY_A}, 'B': {'XX': gasSigXX_B, 'XY': gasSigXY_B, 'YX': gasSigYX_B, 'YY': gasSigYY_B}}}}
        return stress_dict

    def shear_stress(self, stress_dict):

        count_dict = self.phase_ident.phase_count(self.phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin
        int_area = count_dict['int'] * self.sizeBin
        gas_area = count_dict['gas'] * self.sizeBin

        bulk_shear_stress =  (stress_dict['tot']['bulk']['all']['XY']) / bulk_area
        bulkA_shear_stress =  (stress_dict['tot']['bulk']['A']['XY']) / bulk_area
        bulkB_shear_stress =  (stress_dict['tot']['bulk']['B']['XY']) / bulk_area

        int_shear_stress =  (stress_dict['tot']['int']['all']['XY']) / int_area
        intA_shear_stress =  (stress_dict['tot']['int']['A']['XY']) / int_area
        intB_shear_stress =  (stress_dict['tot']['int']['B']['XY']) / int_area

        gas_shear_stress =  (stress_dict['tot']['gas']['all']['XY']) / gas_area
        gasA_shear_stress =  (stress_dict['tot']['gas']['A']['XY']) / gas_area
        gasB_shear_stress =  (stress_dict['tot']['gas']['B']['XY']) / gas_area

        shear_dict = {'bulk': {'all': bulk_shear_stress, 'A': bulkA_shear_stress, 'B': bulkB_shear_stress}, 'int': {'all': int_shear_stress, 'A': intA_shear_stress, 'B': intB_shear_stress}, 'gas': {'all': gas_shear_stress, 'A': gasA_shear_stress, 'B': gasB_shear_stress}}

    def virial_pressure(self, stress_dict):

        bulkTrace = (stress_dict['tot']['bulk']['all']['XX'] + stress_dict['tot']['bulk']['all']['YY'])/2.
        intTrace = (stress_dict['tot']['int']['all']['XX'] + stress_dict['tot']['int']['all']['YY'])/2.
        gasTrace = (stress_dict['tot']['gas']['all']['XX'] + stress_dict['tot']['gas']['all']['YY'])/2.

        bulkTrace_A = (stress_dict['tot']['bulk']['A']['XX'] + stress_dict['tot']['bulk']['A']['YY'])/2.
        intTrace_A = (stress_dict['tot']['int']['A']['XX'] + stress_dict['tot']['int']['A']['YY'])/2.
        gasTrace_A = (stress_dict['tot']['gas']['A']['XX'] + stress_dict['tot']['gas']['A']['YY'])/2.

        bulkTrace_B = (stress_dict['tot']['bulk']['B']['XX'] + stress_dict['tot']['bulk']['B']['YY'])/2.
        intTrace_B = (stress_dict['tot']['int']['B']['XX'] + stress_dict['tot']['int']['B']['YY'])/2.
        gasTrace_B = (stress_dict['tot']['gas']['B']['XX'] + stress_dict['tot']['gas']['B']['YY'])/2.

        count_dict = self.phase_ident.phase_count(self.phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin
        int_area = count_dict['int'] * self.sizeBin
        gas_area = count_dict['gas'] * self.sizeBin

        bulk_press = bulkTrace / bulk_area
        bulkA_press = bulkTrace_A / bulk_area
        bulkB_press = bulkTrace_B / bulk_area

        int_press = intTrace / int_area
        intA_press = intTrace_A / int_area
        intB_press = intTrace_B / int_area

        gas_press = gasTrace / gas_area
        gasA_press = gasTrace_A / gas_area
        gasB_press = gasTrace_B / gas_area

        press_dict = {'bulk': {'all': bulk_press, 'A': bulkA_press, 'B': bulkB_press}, 'int': {'all': int_press, 'A': intA_press, 'B': intB_press}, 'gas': {'all': gas_press, 'A': gasA_press, 'B': gasB_press}}
        return press_dict

    def radial_active_force_pressure(self, part_force_dict):

        #X locations across interface for integration
        r = np.linspace(0, self.h_box, num=int((np.ceil(self.h_box)+1)/3))

        #Pressure integrand components for each value of X
        act_press_r = np.zeros(len(r)-1)
        align_r = np.zeros(len(r)-1)
        num_dens_r = np.zeros(len(r)-1)

        act_pressA_r = np.zeros(len(r)-1)
        alignA_r = np.zeros(len(r)-1)
        num_densA_r = np.zeros(len(r)-1)

        act_pressB_r = np.zeros(len(r)-1)
        alignB_r = np.zeros(len(r)-1)
        num_densB_r = np.zeros(len(r)-1)

        #If exterior and interior surfaces defined, continue...

        area_prev = 0
        #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
        for i in range(1, len(r)):

            #Min and max location across interface of current step
            min_r = r[i-1]
            max_r = r[i]

            #Calculate area of rectangle for current step
            area = np.pi * (max_r ** 2) - area_prev

            #Save total area of previous step sizes
            area_prev = np.pi * (max_r ** 2)


            #Find particles that are housed within current slice
            parts_inrange = np.where((min_r<=part_force_dict['r']) & (part_force_dict['r']<=max_r))[0]
            partsA_inrange = np.where((min_r<=part_force_dict['r']) & (part_force_dict['r']<=max_r) & (self.typ==0))[0]
            partsB_inrange = np.where((min_r<=part_force_dict['r']) & (part_force_dict['r']<=max_r) & (self.typ==1))[0]

            #If at least 1 particle in slice, continue...
            if len(parts_inrange)>0:

                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(part_force_dict['fa'][parts_inrange]))
                partsA_defined = np.logical_not(np.isnan(part_force_dict['fa'][partsA_inrange]))
                partsB_defined = np.logical_not(np.isnan(part_force_dict['fa'][partsB_inrange]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    act_press_r[i-1] = np.sum(part_force_dict['fa'][parts_inrange][parts_defined])
                    act_pressA_r[i-1] = np.sum(part_force_dict['fa'][partsA_inrange][partsA_defined])
                    act_pressB_r[i-1] = np.sum(part_force_dict['fa'][partsB_inrange][partsB_defined])

                #Calculate average alignment
                align_r[i-1] = np.mean(part_force_dict['align'][parts_inrange][parts_defined])
                alignA_r[i-1] = np.mean(part_force_dict['align'][partsA_inrange][partsA_defined])
                alignB_r[i-1] = np.mean(part_force_dict['align'][partsB_inrange][partsB_defined])

                #Calculate density
                num_dens_r[i-1] = len(parts_defined)
                num_densA_r[i-1] = len([partsA_defined])
                num_densB_r[i-1] = len([partsB_defined])

                    #If area of slice is non-zero, calculate the pressure [F/A]
                if area > 0:
                    act_press_r[i-1] = act_press_r[i-1]/area
                    act_pressA_r[i-1] = act_pressA_r[i-1]/area
                    act_pressB_r[i-1] = act_pressB_r[i-1]/area
                    num_dens_r[i-1] = num_dens_r[i-1]/area
                    num_densA_r[i-1] = num_densA_r[i-1]/area
                    num_densB_r[i-1] = num_densB_r[i-1]/area

        com_radial_dict = {'r': r, 'active pressure': {'all': act_press_r, 'A': act_pressA_r, 'B': act_pressB_r}, 'align': {'all': align_r, 'A': alignA_r, 'B': alignB_r}, 'num dens': {'all': num_dens_r, 'A': num_densA_r, 'B': num_densB_r}}
        return com_radial_dict

    def total_active_pressure(self, com_radial_dict):

        #Initiate empty values for integral of pressures across interfaces
        act_press = 0
        act_pressA = 0
        act_pressB = 0

        #Integrate force across interface using trapezoidal rule
        for i in range(1, len(com_radial_dict['r'])-1):
            act_press += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['active pressure']['all'][i]+com_radial_dict['active pressure']['all'][i-1])
            act_pressA += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['active pressure']['A'][i]+com_radial_dict['active pressure']['A'][i-1])
            act_pressB += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['active pressure']['B'][i]+com_radial_dict['active pressure']['B'][i-1])

        act_press_dict = {'all': act_press, 'A': act_pressA, 'B': act_pressB}
        return act_press_dict
