
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
import particles
class stress_and_pressure:
    def __init__(self, lx_box, ly_box, NBins_x, NBins_y, partNum, pos, typ, px, py, part_dict, eps, peA, peB, parFrac):

        import freud

        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        self.f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

        try:
            self.NBins_x = int(NBins_x)
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        self.sizeBin_x = self.utility_functs.roundUp((lx_box / NBins_x), 6)
        self.sizeBin_y = self.utility_functs.roundUp((ly_box / NBins_y), 6)

        self.partNum = partNum

        self.part_dict = part_dict

        self.pos = pos

        self.typ = typ

        self.px = px
        self.py = py

        self.binParts = part_dict['id']

        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        self.eps = eps

        self.peA = peA

        self.peB = peB

        self.parFrac = parFrac

        self.binning = binning.binning(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.typ, self.eps)

        self.plotting_utility = plotting_utility.plotting_utility(self.lx_box, self.ly_box, self.partNum, self.typ)

        self.particle_prop_functs = particles.particle_props(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.eps, self.typ, self.pos, self.px, self.py)

        self.theory_functs = theory.theory()

    def interparticle_stress(self, phase_dict):
        '''
        Purpose: Calculates total interparticle stress between particles in each phase

        Output:
        stress_plot_dict: dictionary containing interparticle stresses experienced by individual particles

        stress_stat_dict: dictionary containing the statistical values of interparticle stresses by all particles in each phase
        '''
        phase_dict, part_dict = self.binning.decrease_bin_size(phase_dict['bin'], phase_dict['part'], self.part_dict['id'], self.pos, self.typ)

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
                                for j in binParts[indx][indy]:

                                    difx = self.utility_functs.sep_dist_x(self.pos[h][0], self.pos[j][0])

                                    dify = self.utility_functs.sep_dist_y(self.pos[h][1], self.pos[j][1])

                                    difr = ( (difx)**2 + (dify)**2)**0.5


                                    # If potential is on ...
                                    if 0.1 < difr <= self.r_cut:
                                        # Compute the x and y components of force
                                        fx, fy = theory_functs.computeFLJ(difr, difx, dify, self.eps)

                                        SigXX = (fx * difx)
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

        stress_plot_dict = {'bin': {'XX': SigXX_bin, 'XY': SigXY_bin, 'YX': SigYX_bin, 'YY': SigYY_bin}, 'part': {'XX': SigXX_part, 'XY': SigXY_part, 'YX': SigYX_part, 'YY': SigYY_part}}
        stress_stat_dict = {'bulk': {'all': {'XX': bulkSigXX_all, 'XY': bulkSigXY_all, 'YX': bulkSigYX_all, 'YY': bulkSigYY_all}, 'A': {'XX': bulkSigXX_A, 'XY': bulkSigXY_A, 'YX': bulkSigYX_A, 'YY': bulkSigYY_A}, 'B': {'XX': bulkSigXX_B, 'XY': bulkSigXY_B, 'YX': bulkSigYX_B, 'YY': bulkSigYY_B}}, 'int': {'all': {'XX': intSigXX_all, 'XY': intSigXY_all, 'YX': intSigYX_all, 'YY': intSigYY_all}, 'A': {'XX': intSigXX_A, 'XY': intSigXY_A, 'YX': intSigYX_A, 'YY': intSigYY_A}, 'B': {'XX': intSigXX_B, 'XY': intSigXY_B, 'YX': intSigYX_B, 'YY': intSigYY_B}}, 'gas': {'all': {'XX': gasSigXX_all, 'XY': gasSigXY_all, 'YX': gasSigYX_all, 'YY': gasSigYY_all}, 'A': {'XX': gasSigXX_A, 'XY': gasSigXY_A, 'YX': gasSigYX_A, 'YY': gasSigYY_A}, 'B': {'XX': gasSigXX_B, 'XY': gasSigXY_B, 'YX': gasSigYX_B, 'YY': gasSigYY_B}}}
        return stress_plot_dict, stress_stat_dict

    def shear_stress(self, stress_stat_dict, phase_dict):
        '''
        Purpose: Calculates total shear stress between particles in each phase

        Output:
        stress_plot_dict: dictionary containing interparticle stresses experienced by individual particles

        shear_dict: dictionary containing the average shear pressure in each phase
        '''
        count_dict = self.particle_prop_functs.phase_count(phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin_x * self.sizeBin_y
        int_area = count_dict['int'] * self.sizeBin_x * self.sizeBin_y
        gas_area = count_dict['gas'] * self.sizeBin_x * self.sizeBin_y

        bulk_shear_stress =  (stress_stat_dict['bulk']['all']['XY']) / bulk_area
        bulkA_shear_stress =  (stress_stat_dict['bulk']['A']['XY']) / bulk_area
        bulkB_shear_stress =  (stress_stat_dict['bulk']['B']['XY']) / bulk_area

        int_shear_stress =  (stress_stat_dict['int']['all']['XY']) / int_area
        intA_shear_stress =  (stress_stat_dict['int']['A']['XY']) / int_area
        intB_shear_stress =  (stress_stat_dict['int']['B']['XY']) / int_area

        gas_shear_stress =  (stress_stat_dict['gas']['all']['XY']) / gas_area
        gasA_shear_stress =  (stress_stat_dict['gas']['A']['XY']) / gas_area
        gasB_shear_stress =  (stress_stat_dict['gas']['B']['XY']) / gas_area

        shear_dict = {'bulk': {'all': bulk_shear_stress, 'A': bulkA_shear_stress, 'B': bulkB_shear_stress}, 'int': {'all': int_shear_stress, 'A': intA_shear_stress, 'B': intB_shear_stress}, 'gas': {'all': gas_shear_stress, 'A': gasA_shear_stress, 'B': gasB_shear_stress}}

        return shear_dict

    def virial_pressure_binned(self, stress_plot_dict):
        '''
        Purpose: Calculates and bins the average interparticle pressure

        Output:
        press_arr: array (NBins_x, NBins_y) containing the total Virial formulation of interparticle pressure in each bin
        '''

        press_arr = np.zeros((self.NBins_x, self.NBins_y))

        for ix in range(0, self.NBins_x):

            for iy in range(0, self.NBins_y):

                press_arr[ix][iy] = (stress_plot_dict['bin']['XX'][ix][iy] + stress_plot_dict['bin']['YY'][ix][iy])/(2 * self.sizeBin_x * self.sizeBin_y)

        return press_arr

    def virial_pressure_part(self, stress_plot_dict):
        '''
        Purpose: Calculates the average interparticle pressure experienced by each particle

        Inputs:
        stress_plot_dict: dictionary containing interparticle stresses experienced by individual particles

        Output:
        press_arr: array (partNum) containing the total Virial formulation of interparticle pressure for each particle
        '''
        press_arr = np.zeros(self.partNum)

        for h in range(0, self.partNum):

            press_arr[h] = (stress_plot_dict['part']['XX'][h] + stress_plot_dict['part']['YY'][h])/(2)

        return press_arr

    def virial_pressure(self, stress_stat_dict, phase_dict):
        '''
        Purpose: Calculates and bins the average interparticle pressure

        Inputs:
        stress_stat_dict: dictionary containing the statistical values of interparticle stresses by all particles in each phase

        Output:
        press_dict: array (NBins_x, NBins_y) containing the total Virial formulation of interparticle pressure in each bin
        '''
        bulkTrace = (stress_stat_dict['bulk']['all']['XX'] + stress_stat_dict['bulk']['all']['YY'])/2.
        intTrace = (stress_stat_dict['int']['all']['XX'] + stress_stat_dict['int']['all']['YY'])/2.
        gasTrace = (stress_stat_dict['gas']['all']['XX'] + stress_stat_dict['gas']['all']['YY'])/2.

        bulkTrace_A = (stress_stat_dict['bulk']['A']['XX'] + stress_stat_dict['bulk']['A']['YY'])/2.
        intTrace_A = (stress_stat_dict['int']['A']['XX'] + stress_stat_dict['int']['A']['YY'])/2.
        gasTrace_A = (stress_stat_dict['gas']['A']['XX'] + stress_stat_dict['gas']['A']['YY'])/2.

        bulkTrace_B = (stress_stat_dict['bulk']['B']['XX'] + stress_stat_dict['bulk']['B']['YY'])/2.
        intTrace_B = (stress_stat_dict['int']['B']['XX'] + stress_stat_dict['int']['B']['YY'])/2.
        gasTrace_B = (stress_stat_dict['gas']['B']['XX'] + stress_stat_dict['gas']['B']['YY'])/2.

        count_dict = self.particle_prop_functs.phase_count(phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin_x * self.sizeBin_y
        int_area = count_dict['int'] * self.sizeBin_x * self.sizeBin_y
        gas_area = count_dict['gas'] * self.sizeBin_x * self.sizeBin_y

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

    def radial_com_interparticle_pressure(self, radial_stress_dict):

        #X locations across interface for integration
        if self.hx_box<self.hy_box:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))
        elif self.hy_box<self.hx_box:
            r = np.linspace(0, self.hy_box, num=int((np.ceil(self.hy_box)+1)/3))
        else:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))

        #Pressure integrand components for each value of X
        int_stress_XX_r = np.zeros((len(r)-1))
        int_stress_YY_r = np.zeros((len(r)-1))
        int_stress_XY_r = np.zeros((len(r)-1))
        int_stress_YX_r = np.zeros((len(r)-1))
        int_press_r = np.zeros((len(r)-1))
        #act_fa_r = []
        #lat_r = []
        num_dens_r = np.zeros((len(r)-1))

        int_stressA_XX_r = np.zeros((len(r)-1))
        int_stressA_YY_r = np.zeros((len(r)-1))
        int_stressA_XY_r = np.zeros((len(r)-1))
        int_stressA_YX_r = np.zeros((len(r)-1))
        int_pressA_r = np.zeros((len(r)-1))

        #act_faA_r = []
        #latA_r = []
        num_densA_r = np.zeros((len(r)-1))

        int_stressB_XX_r = np.zeros((len(r)-1))
        int_stressB_YY_r = np.zeros((len(r)-1))
        int_stressB_XY_r = np.zeros((len(r)-1))
        int_stressB_YX_r = np.zeros((len(r)-1))
        int_pressB_r = np.zeros((len(r)-1))
        #act_faB_r = []
        #latB_r = []
        num_densB_r = np.zeros((len(r)-1))

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
            parts_inrange = np.where((min_r<=radial_stress_dict['all']['r']) & (radial_stress_dict['all']['r']<=max_r))[0]
            partsA_inrange = np.where((min_r<=radial_stress_dict['A']['r']) & (radial_stress_dict['A']['r']<=max_r))[0]
            partsB_inrange = np.where((min_r<=radial_stress_dict['B']['r']) & (radial_stress_dict['B']['r']<=max_r))[0]

            #If at least 1 particle in slice, continue...
            if len(parts_inrange)>0:

                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(radial_stress_dict['all']['XX'][parts_inrange]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    int_stress_XX_r[i-1] = np.sum((radial_stress_dict['all']['XX'][parts_inrange][parts_defined]))
                    int_stress_YY_r[i-1] = np.sum((radial_stress_dict['all']['YY'][parts_inrange][parts_defined]))
                    int_stress_XY_r[i-1] = np.sum((radial_stress_dict['all']['XY'][parts_inrange][parts_defined]))
                    int_stress_YX_r[i-1] = np.sum((radial_stress_dict['all']['YX'][parts_inrange][parts_defined]))
                    int_press_r[i-1] = np.sum((radial_stress_dict['all']['XX'][parts_inrange][parts_defined] + radial_stress_dict['all']['YY'][parts_inrange][parts_defined])/2)
                    #Calculate density
                    num_dens_r[i-1] = len(parts_defined)
                    #If area of slice is non-zero, calculate the pressure [F/A]
                    if area > 0:
                        int_press_r[i-1] = int_press_r[i-1]/area
                        num_dens_r[i-1] = num_dens_r[i-1]/area

                    partsA_defined = np.logical_not(np.isnan(radial_stress_dict['A']['XX'][partsA_inrange]))

                    if len(partsA_defined)>0:
                        int_stressA_XX_r[i-1] = np.sum((radial_stress_dict['A']['XX'][partsA_inrange][partsA_defined]))
                        int_stressA_YY_r[i-1] = np.sum((radial_stress_dict['A']['YY'][partsA_inrange][partsA_defined]))
                        int_stressA_XY_r[i-1] = np.sum((radial_stress_dict['A']['XY'][partsA_inrange][partsA_defined]))
                        int_stressA_YX_r[i-1] = np.sum((radial_stress_dict['A']['YX'][partsA_inrange][partsA_defined]))
                        int_pressA_r[i-1] = np.sum((radial_stress_dict['A']['XX'][partsA_inrange][partsA_defined] + radial_stress_dict['A']['YY'][partsA_inrange][partsA_defined])/2)
                        #Calculate density
                        num_densA_r[i-1] = len(partsA_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            int_pressA_r[i-1] = int_pressA_r[i-1]/area
                            num_densA_r[i-1] = num_densA_r[i-1]/area

                    partsB_defined = np.logical_not(np.isnan(radial_stress_dict['B']['XX'][partsB_inrange]))
                    if len(partsB_defined)>0:
                        int_stressB_XX_r[i-1] = np.sum((radial_stress_dict['B']['XX'][partsB_inrange][partsB_defined]))
                        int_stressB_YY_r[i-1] = np.sum((radial_stress_dict['B']['YY'][partsB_inrange][partsB_defined]))
                        int_stressB_XY_r[i-1] = np.sum((radial_stress_dict['B']['XY'][partsB_inrange][partsB_defined]))
                        int_stressB_YX_r[i-1] = np.sum((radial_stress_dict['B']['YX'][partsB_inrange][partsB_defined]))
                        int_pressB_r[i-1] = np.sum((radial_stress_dict['B']['XX'][partsB_inrange][partsB_defined] + radial_stress_dict['B']['YY'][partsB_inrange][partsB_defined])/2)
                        #Calculate density
                        num_densB_r[i-1] = len(partsB_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            int_pressB_r[i-1] = int_pressB_r[i-1]/area
                            num_densB_r[i-1] = num_densB_r[i-1]/area
                
        com_radial_dict = {'r': r[1:].tolist(), 'all': {'XX': int_stress_XX_r.tolist(), 'YY': int_stress_YY_r.tolist(), 'XY': int_stress_XY_r.tolist(), 'YX': int_stress_YX_r.tolist(), 'press': int_press_r.tolist(), 'num_dens': num_dens_r.tolist()}, 'A': {'XX': int_stressA_XX_r.tolist(), 'YY': int_stressA_YY_r.tolist(), 'XY': int_stressA_XY_r.tolist(), 'YX': int_stressA_YX_r.tolist(), 'press': int_pressA_r.tolist(), 'num_dens': num_densA_r.tolist()}, 'B': {'XX': int_stressB_XX_r.tolist(), 'YY': int_stressB_YY_r.tolist(), 'XY': int_stressB_XY_r.tolist(), 'YX': int_stressB_YX_r.tolist(), 'press': int_pressB_r.tolist(), 'num_dens': num_densB_r.tolist()}}
        return com_radial_dict

    def radial_com_active_force_pressure(self, radial_fa_dict):

        #X locations across interface for integration
        if self.hx_box<self.hy_box:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))
        elif self.hy_box<self.hx_box:
            r = np.linspace(0, self.hy_box, num=int((np.ceil(self.hy_box)+1)/3))
        else:
             r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))

        #Pressure integrand components for each value of X
        act_press_r = np.zeros((len(r)-1))
        act_fa_r = np.zeros((len(r)-1))
        align_r = np.zeros((len(r)-1))
        num_dens_r = np.zeros((len(r)-1))

        act_pressA_r = np.zeros((len(r)-1))
        act_faA_r = np.zeros((len(r)-1))
        alignA_r = np.zeros((len(r)-1))
        num_densA_r = np.zeros((len(r)-1))

        act_pressB_r = np.zeros((len(r)-1))
        act_faB_r = np.zeros((len(r)-1))
        alignB_r = np.zeros((len(r)-1))
        num_densB_r = np.zeros((len(r)-1))

        rad_arr = np.zeros((len(r)-1))
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
            parts_inrange = np.where((min_r<=radial_fa_dict['all']['r']) & (radial_fa_dict['all']['r']<=max_r))[0]
            partsA_inrange = np.where((min_r<=radial_fa_dict['A']['r']) & (radial_fa_dict['A']['r']<=max_r))[0]
            partsB_inrange = np.where((min_r<=radial_fa_dict['B']['r']) & (radial_fa_dict['B']['r']<=max_r))[0]

            #If at least 1 particle in slice, continue...
            if len(parts_inrange)>0:
                
                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(radial_fa_dict['all']['fa'][parts_inrange]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    act_press_r[i-1] = np.sum(radial_fa_dict['all']['fa'][parts_inrange][parts_defined])
                    act_fa_r[i-1] = np.mean(radial_fa_dict['all']['fa'][parts_inrange][parts_defined])
                    align_r[i-1] = np.mean(radial_fa_dict['all']['align'][parts_inrange][parts_defined])
                    num_dens_r[i-1] = len(parts_defined)
                    #If area of slice is non-zero, calculate the pressure [F/A]
                    if area > 0:
                        act_press_r[i-1] = act_press_r[i-1]/area
                        num_dens_r[i-1] = num_dens_r[i-1]/area

                    partsA_defined = np.logical_not(np.isnan(radial_fa_dict['A']['fa'][partsA_inrange]))
                    if len(partsA_defined)>0:
                        act_pressA_r[i-1] = np.sum(radial_fa_dict['A']['fa'][partsA_inrange][partsA_defined])
                        act_faA_r[i-1] = np.mean(radial_fa_dict['A']['fa'][partsA_inrange][partsA_defined])
                        alignA_r[i-1] = np.mean(radial_fa_dict['A']['align'][partsA_inrange][partsA_defined])
                        num_densA_r[i-1] = len(partsA_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressA_r[i-1] = act_pressA_r[i-1]/area
                            num_densA_r[i-1] = num_densA_r[i-1]/area

                    partsB_defined = np.logical_not(np.isnan(radial_fa_dict['B']['fa'][partsB_inrange]))
                    if len(partsB_defined)>0:
                        act_pressB_r[i-1] = np.sum(radial_fa_dict['B']['fa'][partsB_inrange][partsB_defined])
                        act_faB_r[i-1] = np.mean(radial_fa_dict['B']['fa'][partsB_inrange][partsB_defined])
                        alignB_r[i-1] = np.mean(radial_fa_dict['B']['align'][partsB_inrange][partsB_defined])
                        #Calculate density
                        num_densB_r[i-1] = len(partsB_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressB_r[i-1] = act_pressB_r[i-1]/area
                            num_densB_r[i-1] = num_densB_r[i-1]/area

        com_radial_dict = {'r': r[1:].tolist(), 'fa_press': {'all': act_press_r.tolist(), 'A': act_pressA_r.tolist(), 'B': act_pressB_r.tolist()}, 'fa': {'all': act_fa_r.tolist(), 'A': act_faA_r.tolist(), 'B': act_faB_r.tolist()}, 'align': {'all': align_r.tolist(), 'A': alignA_r.tolist(), 'B': alignB_r.tolist()}, 'num_dens': {'all': num_dens_r.tolist(), 'A': num_densA_r.tolist(), 'B': num_densB_r.tolist()}}
        return com_radial_dict

    def total_active_pressure(self, com_radial_dict):

        #Initiate empty values for integral of pressures across interfaces
        act_press = 0
        act_pressA = 0
        act_pressB = 0

        #Integrate force across interface using trapezoidal rule
        for i in range(1, len(com_radial_dict['r'])-1):
            act_press += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['all'][i]+com_radial_dict['fa_press']['all'][i-1])
            act_pressA += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['A'][i]+com_radial_dict['fa_press']['A'][i-1])
            act_pressB += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['B'][i]+com_radial_dict['fa_press']['B'][i-1])

        act_press_dict = {'all': act_press, 'A': act_pressA, 'B': act_pressB}
        return act_press_dict

    def total_active_pressure_bubble(self, com_radial_dict, sep_surface_dict, int_comp_dict, all_surface_measurements):

        int_id = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])

        act_press_dict = {}
        for m in range(0, len(sep_surface_dict)):

            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

            try: 
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']

            #Initiate empty values for integral of pressures across interfaces
            act_press = 0
            act_pressA = 0
            act_pressB = 0

            num_dens = 0
            num_densA = 0
            num_densB = 0

            align = 0
            alignA = 0
            alignB = 0

            act_force = 0
            act_forceA = 0
            act_forceB = 0

            sum_tot = 0

            area = 0

            num = 0
            numA = 0
            numB = 0

            #Integrate force across interface using trapezoidal rule
            for i in range(1, len(com_radial_dict[key]['r'])-1):
                act_press += ((com_radial_dict[key]['r'][i]-com_radial_dict[key]['r'][i-1])/2)*(com_radial_dict[key]['fa_press']['all'][i]+com_radial_dict[key]['fa_press']['all'][i-1])
                act_pressA += ((com_radial_dict[key]['r'][i]-com_radial_dict[key]['r'][i-1])/2)*(com_radial_dict[key]['fa_press']['A'][i]+com_radial_dict[key]['fa_press']['A'][i-1])
                act_pressB += ((com_radial_dict[key]['r'][i]-com_radial_dict[key]['r'][i-1])/2)*(com_radial_dict[key]['fa_press']['B'][i]+com_radial_dict[key]['fa_press']['B'][i-1])

                area += com_radial_dict[key]['area'][i]

                num += com_radial_dict[key]['num']['all'][i]
                numA += com_radial_dict[key]['num']['A'][i]
                numB += com_radial_dict[key]['num']['B'][i]

                sum_tot += 1

                num_dens += com_radial_dict[key]['num_dens']['all'][i]
                num_densA += com_radial_dict[key]['num_dens']['A'][i]
                num_densB += com_radial_dict[key]['num_dens']['B'][i]

                align += com_radial_dict[key]['align']['all'][i]
                alignA += com_radial_dict[key]['align']['A'][i]
                alignB += com_radial_dict[key]['align']['B'][i]

                act_force += com_radial_dict[key]['fa']['all'][i]
                act_forceA += com_radial_dict[key]['fa']['A'][i]
                act_forceB += com_radial_dict[key]['fa']['B'][i]

            avg_fa = act_force / sum_tot
            avg_faA = act_forceA / sum_tot
            avg_faB = act_forceB / sum_tot

            avg_align = align / sum_tot
            avg_alignA = alignA / sum_tot
            avg_alignB = alignB / sum_tot

            avg_num_dens = num_dens / sum_tot
            avg_num_densA = num_densA / sum_tot
            avg_num_densB = num_densB / sum_tot

            act_press_dict[key] = {'int_id': int_id, 'current_id': int(int_comp_dict['ids'][m]), 'com_x': com_x, 'com_y': com_y, 'tot_fa': act_press, 'tot_faA': act_pressA, 'tot_faB': act_pressB, 'avg_fa': avg_fa, 'avg_faA': avg_faA, 'avg_faB': avg_faB, 'avg_align': avg_align, 'avg_alignA': avg_alignA, 'avg_alignB': avg_alignB, 'avg_num_dens': avg_num_dens, 'avg_num_densA': avg_num_densA, 'avg_num_densB': avg_num_densB, 'area': area, 'num': int(num), 'numA': int(numA), 'numB': int(numB)}
        return act_press_dict

    def total_active_pressure_interface(self, com_radial_dict, sep_surface_dict, int_comp_dict):

        exterior_surface_arr = np.array([])
        interior_surface_arr = np.array([])

        int_id = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])

        for m in range(0, len(sep_surface_dict)):
            if int(int_comp_dict['ids'][m]) == int_id:

                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                try:
                    exterior_surface_arr = np.append(exterior_surface_arr, sep_surface_dict[key]['exterior']['mean radius'])
                except:
                    exterior_surface_arr = np.append(exterior_surface_arr, 0)
                try:
                    interior_surface_arr = np.append(interior_surface_arr, sep_surface_dict[key]['interior']['mean radius'])
                except:
                    interior_surface_arr = np.append(interior_surface_arr, 0)

        #Initiate empty values for integral of pressures across interfaces
        act_press = 0
        act_pressA = 0
        act_pressB = 0

        #Integrate force across interface using trapezoidal rule
        for i in range(1, len(com_radial_dict['r'])-1):
            act_press += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['all'][i]+com_radial_dict['fa_press']['all'][i-1])
            act_pressA += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['A'][i]+com_radial_dict['fa_press']['A'][i-1])
            act_pressB += ((com_radial_dict['r'][i]-com_radial_dict['r'][i-1])/2)*(com_radial_dict['fa_press']['B'][i]+com_radial_dict['fa_press']['B'][i-1])

        act_press_dict = {'all': act_press, 'A': act_pressA, 'B': act_pressB}
        return act_press_dict

    

    def interparticle_stress_nlist(self, phasePart):

        phase_part_dict = self.particle_prop_functs.particle_phase_ids(phasePart)
        
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_gas_int = [[] for i in range(3)]
        for i in range(0, np.shape(pos_A_gas)[0]):
            pos_A_gas_int[0].append(pos_A_gas[i,0])
            pos_A_gas_int[1].append(pos_A_gas[i,1])
            pos_A_gas_int[2].append(pos_A_gas[i,2])
        for i in range(0, np.shape(pos_A_int)[0]):
            pos_A_gas_int[0].append(pos_A_int[i,0])
            pos_A_gas_int[1].append(pos_A_int[i,1])
            pos_A_gas_int[2].append(pos_A_int[i,2])
        pos_A_gas_int = np.array(pos_A_gas_int).reshape((np.shape(pos_A_gas_int)[1], np.shape(pos_A_gas_int)[0]))
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]                               # Find positions of type 0 particles
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_gas_int = [[] for i in range(3)]
        for i in range(0, np.shape(pos_B_gas)[0]):
            pos_B_gas_int[0].append(pos_B_gas[i,0])
            pos_B_gas_int[1].append(pos_B_gas[i,1])
            pos_B_gas_int[2].append(pos_B_gas[i,2])
        for i in range(0, np.shape(pos_B_int)[0]):
            pos_B_gas_int[0].append(pos_B_int[i,0])
            pos_B_gas_int[1].append(pos_B_int[i,1])
            pos_B_gas_int[2].append(pos_B_int[i,2])
        pos_B_gas_int = np.array(pos_B_gas_int).reshape((np.shape(pos_B_gas_int)[1], np.shape(pos_B_gas_int)[0]))
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_gas_int = [[] for i in range(3)]
        for i in range(0, np.shape(pos_gas)[0]):
            pos_gas_int[0].append(pos_gas[i,0])
            pos_gas_int[1].append(pos_gas[i,1])
            pos_gas_int[2].append(pos_gas[i,2])
        for i in range(0, np.shape(pos_int)[0]):
            pos_gas_int[0].append(pos_int[i,0])
            pos_gas_int[1].append(pos_int[i,1])
            pos_gas_int[2].append(pos_int[i,2])
        pos_gas_int = np.array(pos_gas_int).reshape((np.shape(pos_gas_int)[1], np.shape(pos_gas_int)[0]))
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))

        A_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        B_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        difx_A, dify_A, difr_A = self.utility_functs.sep_dist_arr(pos_dense[A_bulk_nlist.point_indices], pos_A_bulk[A_bulk_nlist.query_point_indices], difxy=True)
        difx_B, dify_B, difr_B = self.utility_functs.sep_dist_arr(pos_dense[B_bulk_nlist.point_indices], pos_B_bulk[B_bulk_nlist.query_point_indices], difxy=True)

        difx = np.append(difx_A, difx_B)
        dify = np.append(dify_A, dify_B)
        difr = np.append(difr_A, difr_B)

        fx = np.array([])
        fy = np.array([])

        bulk_A_ind = np.array([], dtype=int)
        bulk_SigXX_A = np.array([])
        bulk_SigYY_A = np.array([])
        bulk_SigXY_A = np.array([])
        bulk_SigYX_A = np.array([])

        bulk_B_ind = np.array([], dtype=int)
        bulk_SigXX_B = np.array([])
        bulk_SigYY_B = np.array([])
        bulk_SigXY_B = np.array([])
        bulk_SigYX_B = np.array([])

        for i in A_bulk_nlist.point_indices:
            if i not in bulk_A_ind:
                loc = np.where(A_bulk_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                bulk_SigXX_A = np.append(bulk_SigXX_A, np.sum(fx_arr * difx[loc]))
                bulk_SigYY_A = np.append(bulk_SigYY_A, np.sum(fy_arr * dify[loc]))
                bulk_SigXY_A = np.append(bulk_SigXY_A, np.sum(fx_arr * dify[loc]))
                bulk_SigYX_A = np.append(bulk_SigYX_A, np.sum(fy_arr * difx[loc]))
                bulk_A_ind = np.append(bulk_A_ind, int(i))

        bulk_SigXX_A_mean = np.mean(bulk_SigXX_A)
        bulk_SigYY_A_mean = np.mean(bulk_SigYY_A)
        bulk_SigXY_A_mean = np.mean(bulk_SigXY_A)
        bulk_SigYX_A_mean = np.mean(bulk_SigYX_A)

        bulk_SigXX_A_std = np.std(bulk_SigXX_A)
        bulk_SigYY_A_std = np.std(bulk_SigYY_A)
        bulk_SigXY_A_std = np.std(bulk_SigXY_A)
        bulk_SigYX_A_std = np.std(bulk_SigYX_A)

        for i in B_bulk_nlist.point_indices:
            if i not in bulk_B_ind:
                loc = np.where(B_bulk_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                bulk_SigXX_B = np.append(bulk_SigXX_B, np.sum(fx_arr * difx[loc]))
                bulk_SigYY_B = np.append(bulk_SigYY_B, np.sum(fy_arr * dify[loc]))
                bulk_SigXY_B = np.append(bulk_SigXY_B, np.sum(fx_arr * dify[loc]))
                bulk_SigYX_B = np.append(bulk_SigYX_B, np.sum(fy_arr * difx[loc]))
                bulk_B_ind = np.append(bulk_B_ind, int(i))

        bulk_SigXX_B_mean = np.mean(bulk_SigXX_B)
        bulk_SigYY_B_mean = np.mean(bulk_SigYY_B)
        bulk_SigXY_B_mean = np.mean(bulk_SigXY_B)
        bulk_SigYX_B_mean = np.mean(bulk_SigYX_B)

        bulk_SigXX_B_std = np.std(bulk_SigXX_B)
        bulk_SigYY_B_std = np.std(bulk_SigYY_B)
        bulk_SigXY_B_std = np.std(bulk_SigXY_B)
        bulk_SigYX_B_std = np.std(bulk_SigYX_B)

        bulk_SigXX = np.append(bulk_SigXX_A, bulk_SigXX_B)
        bulk_SigYY = np.append(bulk_SigYY_A, bulk_SigYY_B)
        bulk_SigXY = np.append(bulk_SigXY_A, bulk_SigXY_B)
        bulk_SigYX = np.append(bulk_SigYX_A, bulk_SigYX_B)
        bulk_ind = np.append(bulk_A_ind, bulk_B_ind)

        bulk_SigXX_mean = np.mean(bulk_SigXX)
        bulk_SigYY_mean = np.mean(bulk_SigYY)
        bulk_SigXY_mean = np.mean(bulk_SigXY)
        bulk_SigYX_mean = np.mean(bulk_SigYX)

        bulk_SigXX_std = np.std(bulk_SigXX)
        bulk_SigYY_std = np.std(bulk_SigYY)
        bulk_SigXY_std = np.std(bulk_SigXY)
        bulk_SigYX_std = np.std(bulk_SigYX)

        system_all_int = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))   #Calculate neighbor list

        A_int_nlist = system_all_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        B_int_nlist = system_all_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        difx_A, dify_A, difr_A = self.utility_functs.sep_dist_arr(self.pos[A_int_nlist.point_indices], pos_A_int[A_int_nlist.query_point_indices], difxy=True)
        difx_B, dify_B, difr_B = self.utility_functs.sep_dist_arr(self.pos[B_int_nlist.point_indices], pos_B_int[B_int_nlist.query_point_indices], difxy=True)

        difx = np.append(difx_A, difx_B)
        dify = np.append(dify_A, dify_B)
        difr = np.append(difr_A, difr_B)

        fx = np.array([])
        fy = np.array([])

        int_A_ind = np.array([], dtype=int)
        int_SigXX_A = np.array([])
        int_SigYY_A = np.array([])
        int_SigXY_A = np.array([])
        int_SigYX_A = np.array([])

        int_B_ind = np.array([], dtype=int)
        int_SigXX_B = np.array([])
        int_SigYY_B = np.array([])
        int_SigXY_B = np.array([])
        int_SigYX_B = np.array([])

        for i in A_int_nlist.point_indices:
            if i not in int_A_ind:
                loc = np.where(A_int_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                int_SigXX_A = np.append(int_SigXX_A, np.sum(fx_arr * difx[loc]))
                int_SigYY_A = np.append(int_SigYY_A, np.sum(fy_arr * dify[loc]))
                int_SigXY_A = np.append(int_SigXY_A, np.sum(fx_arr * dify[loc]))
                int_SigYX_A = np.append(int_SigYX_A, np.sum(fy_arr * difx[loc]))
                int_A_ind = np.append(int_A_ind, int(i))

        int_SigXX_A_mean = np.mean(int_SigXX_A)
        int_SigYY_A_mean = np.mean(int_SigYY_A)
        int_SigXY_A_mean = np.mean(int_SigXY_A)
        int_SigYX_A_mean = np.mean(int_SigYX_A)

        int_SigXX_A_std = np.std(int_SigXX_A)
        int_SigYY_A_std = np.std(int_SigYY_A)
        int_SigXY_A_std = np.std(int_SigXY_A)
        int_SigYX_A_std = np.std(int_SigYX_A)

        for i in B_int_nlist.point_indices:
            if i not in int_B_ind:
                loc = np.where(B_int_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                int_SigXX_B = np.append(int_SigXX_B, np.sum(fx_arr * difx[loc]))
                int_SigYY_B = np.append(int_SigYY_B, np.sum(fy_arr * dify[loc]))
                int_SigXY_B = np.append(int_SigXY_B, np.sum(fx_arr * dify[loc]))
                int_SigYX_B = np.append(int_SigYX_B, np.sum(fy_arr * difx[loc]))
                int_B_ind = np.append(int_B_ind, int(i))

        int_SigXX_B_mean = np.mean(int_SigXX_B)
        int_SigYY_B_mean = np.mean(int_SigYY_B)
        int_SigXY_B_mean = np.mean(int_SigXY_B)
        int_SigYX_B_mean = np.mean(int_SigYX_B)

        int_SigXX_B_std = np.std(int_SigXX_B)
        int_SigYY_B_std = np.std(int_SigYY_B)
        int_SigXY_B_std = np.std(int_SigXY_B)
        int_SigYX_B_std = np.std(int_SigYX_B)

        int_SigXX = np.append(int_SigXX_A, int_SigXX_B)
        int_SigYY = np.append(int_SigYY_A, int_SigYY_B)
        int_SigXY = np.append(int_SigXY_A, int_SigXY_B)
        int_SigYX = np.append(int_SigYX_A, int_SigYX_B)
        int_ind = np.append(int_A_ind, int_B_ind)

        int_SigXX_mean = np.mean(int_SigXX)
        int_SigYY_mean = np.mean(int_SigYY)
        int_SigXY_mean = np.mean(int_SigXY)
        int_SigYX_mean = np.mean(int_SigYX)

        int_SigXX_std = np.std(int_SigXX)
        int_SigYY_std = np.std(int_SigYY)
        int_SigXY_std = np.std(int_SigXY)
        int_SigYX_std = np.std(int_SigYX)

        system_all_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_gas_int))   #Calculate neighbor list

        A_gas_nlist = system_all_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        B_gas_nlist = system_all_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()

        difx_A, dify_A, difr_A = self.utility_functs.sep_dist_arr(pos_gas_int[A_gas_nlist.point_indices], pos_A_gas[A_gas_nlist.query_point_indices], difxy=True)
        difx_B, dify_B, difr_B = self.utility_functs.sep_dist_arr(pos_gas_int[B_gas_nlist.point_indices], pos_B_gas[B_gas_nlist.query_point_indices], difxy=True)

        difx = np.append(difx_A, difx_B)
        dify = np.append(dify_A, dify_B)
        difr = np.append(difr_A, difr_B)

        fx = np.array([])
        fy = np.array([])

        gas_A_ind = np.array([], dtype=int)
        gas_SigXX_A = np.array([])
        gas_SigYY_A = np.array([])
        gas_SigXY_A = np.array([])
        gas_SigYX_A = np.array([])

        gas_B_ind = np.array([], dtype=int)
        gas_SigXX_B = np.array([])
        gas_SigYY_B = np.array([])
        gas_SigXY_B = np.array([])
        gas_SigYX_B = np.array([])

        for i in A_gas_nlist.point_indices:
            if i not in gas_A_ind:
                loc = np.where(A_gas_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                gas_SigXX_A = np.append(gas_SigXX_A, np.sum(fx_arr * difx[loc]))
                gas_SigYY_A = np.append(gas_SigYY_A, np.sum(fy_arr * dify[loc]))
                gas_SigXY_A = np.append(gas_SigXY_A, np.sum(fx_arr * dify[loc]))
                gas_SigYX_A = np.append(gas_SigYX_A, np.sum(fy_arr * difx[loc]))
                gas_A_ind = np.append(gas_A_ind, int(i))

        gas_SigXX_A_mean = np.mean(gas_SigXX_A)
        gas_SigYY_A_mean = np.mean(gas_SigYY_A)
        gas_SigXY_A_mean = np.mean(gas_SigXY_A)
        gas_SigYX_A_mean = np.mean(gas_SigYX_A)

        gas_SigXX_A_std = np.std(gas_SigXX_A)
        gas_SigYY_A_std = np.std(gas_SigYY_A)
        gas_SigXY_A_std = np.std(gas_SigXY_A)
        gas_SigYX_A_std = np.std(gas_SigYX_A)

        for i in B_gas_nlist.point_indices:
            if i not in gas_B_ind:
                loc = np.where(B_gas_nlist.point_indices==i)[0]
                fx_arr, fy_arr = self.theory_functs.computeFLJ_arr(difr[loc], difx[loc], dify[loc], self.eps)
                gas_SigXX_B = np.append(gas_SigXX_B, np.sum(fx_arr * difx[loc]))
                gas_SigYY_B = np.append(gas_SigYY_B, np.sum(fy_arr * dify[loc]))
                gas_SigXY_B = np.append(gas_SigXY_B, np.sum(fx_arr * dify[loc]))
                gas_SigYX_B = np.append(gas_SigYX_B, np.sum(fy_arr * difx[loc]))
                gas_B_ind = np.append(gas_B_ind, int(i))

        gas_SigXX_B_mean = np.mean(gas_SigXX_B)
        gas_SigYY_B_mean = np.mean(gas_SigYY_B)
        gas_SigXY_B_mean = np.mean(gas_SigXY_B)
        gas_SigYX_B_mean = np.mean(gas_SigYX_B)

        gas_SigXX_B_std = np.std(gas_SigXX_B)
        gas_SigYY_B_std = np.std(gas_SigYY_B)
        gas_SigXY_B_std = np.std(gas_SigXY_B)
        gas_SigYX_B_std = np.std(gas_SigYX_B)

        gas_SigXX = np.append(gas_SigXX_A, gas_SigXX_B)
        gas_SigYY = np.append(gas_SigYY_A, gas_SigYY_B)
        gas_SigXY = np.append(gas_SigXY_A, gas_SigXY_B)
        gas_SigYX = np.append(gas_SigYX_A, gas_SigYX_B)
        gas_ind = np.append(gas_A_ind, gas_B_ind)

        gas_SigXX_mean = np.mean(gas_SigXX)
        gas_SigYY_mean = np.mean(gas_SigYY)
        gas_SigXY_mean = np.mean(gas_SigXY)
        gas_SigYX_mean = np.mean(gas_SigYX)

        gas_SigXX_std = np.std(gas_SigXX)
        gas_SigYY_std = np.std(gas_SigYY)
        gas_SigXY_std = np.std(gas_SigXY)
        gas_SigYX_std = np.std(gas_SigYX)

        dense_ind = np.append(bulk_ind, int_ind)
        all_ind = np.append(dense_ind, gas_ind)

        pos_x_all = self.pos[all_ind, 0]
        pos_y_all = self.pos[all_ind, 1]

        dense_SigXX = np.append(bulk_SigXX, int_SigXX)
        all_SigXX = np.append(dense_SigXX, gas_SigXX)

        dense_SigYY = np.append(bulk_SigYY, int_SigYY)
        all_SigYY = np.append(dense_SigYY, gas_SigYY)

        dense_SigXY = np.append(bulk_SigXY, int_SigXY)
        all_SigXY = np.append(dense_SigXY, gas_SigXY)

        dense_SigYX = np.append(bulk_SigYX, int_SigYX)
        all_SigYX = np.append(dense_SigYX, gas_SigYX)

        stress_stat_dict = {'bulk': {'all': {'XX': {'mean': bulk_SigXX_mean, 'std': bulk_SigXX_std}, 'YY': {'mean': bulk_SigYY_mean, 'std': bulk_SigYY_std}, 'XY': {'mean': bulk_SigXY_mean, 'std': bulk_SigXY_std}, 'YX': {'mean': bulk_SigYX_mean, 'std': bulk_SigYX_std}}, 'A': {'XX': {'mean': bulk_SigXX_A_mean, 'std': bulk_SigXX_A_std}, 'YY': {'mean': bulk_SigYY_A_mean, 'std': bulk_SigYY_A_std}, 'XY': {'mean': bulk_SigXY_A_mean, 'std': bulk_SigXY_A_std}, 'YX': {'mean': bulk_SigYX_A_mean, 'std': bulk_SigYX_A_std}}, 'B': {'XX': {'mean': bulk_SigXX_B_mean, 'std': bulk_SigXX_B_std}, 'YY': {'mean': bulk_SigYY_B_mean, 'std': bulk_SigYY_B_std}, 'XY': {'mean': bulk_SigXY_B_mean, 'std': bulk_SigXY_B_std}, 'YX': {'mean': bulk_SigYX_B_mean, 'std': bulk_SigYX_B_std}}}, 'int': {'all': {'XX': {'mean': int_SigXX_mean, 'std': int_SigXX_std}, 'YY': {'mean': int_SigYY_mean, 'std': int_SigYY_std}, 'XY': {'mean': int_SigXY_mean, 'std': int_SigXY_std}, 'YX': {'mean': int_SigYX_mean, 'std': int_SigYX_std}}, 'A': {'XX': {'mean': int_SigXX_A_mean, 'std': int_SigXX_A_std}, 'YY': {'mean': int_SigYY_A_mean, 'std': int_SigYY_A_std}, 'XY': {'mean': int_SigXY_A_mean, 'std': int_SigXY_A_std}, 'YX': {'mean': int_SigYX_A_mean, 'std': int_SigYX_A_std}}, 'B': {'XX': {'mean': int_SigXX_B_mean, 'std': int_SigXX_B_std}, 'YY': {'mean': int_SigYY_B_mean, 'std': int_SigYY_B_std}, 'XY': {'mean': int_SigXY_B_mean, 'std': int_SigXY_B_std}, 'YX': {'mean': int_SigYX_B_mean, 'std': int_SigYX_B_std}}}, 'gas': {'all': {'XX': {'mean': gas_SigXX_mean, 'std': gas_SigXX_std}, 'YY': {'mean': gas_SigYY_mean, 'std': gas_SigYY_std}, 'XY': {'mean': gas_SigXY_mean, 'std': gas_SigXY_std}, 'YX': {'mean': gas_SigYX_mean, 'std': gas_SigYX_std}}, 'A': {'XX': {'mean': gas_SigXX_A_mean, 'std': gas_SigXX_A_std}, 'YY': {'mean': gas_SigYY_A_mean, 'std': gas_SigYY_A_std}, 'XY': {'mean': gas_SigXY_A_mean, 'std': gas_SigXY_A_std}, 'YX': {'mean': gas_SigYX_A_mean, 'std': gas_SigYX_A_std}}, 'B': {'XX': {'mean': gas_SigXX_B_mean, 'std': gas_SigXX_B_std}, 'YY': {'mean': gas_SigYY_B_mean, 'std': gas_SigYY_B_std}, 'XY': {'mean': gas_SigXY_B_mean, 'std': gas_SigXY_B_std}, 'YX': {'mean': gas_SigYX_B_mean, 'std': gas_SigYX_B_std}}}}

        stress_plot_dict = {'part': {'XX': all_SigXX, 'YY': all_SigYY, 'XY': all_SigXY, 'YX': all_SigYX, 'x': pos_x_all, 'y': pos_y_all} }
        return stress_stat_dict, stress_plot_dict
    def interparticle_pressure_nlist_system(self):
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
            bulk_area = 2 * self.ly_box * self.lx_box

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


    def interparticle_pressure_nlist_phases(self, phase_dict):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the total interparticle stress all,
        interacting neighbors acting on each reference particle.

        Outputs:
        stress_stat_dict: dictionary containing the total interparticle stress
        between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B')
        in each direction ('XX', 'XY', 'YX', 'YY'), averaged over all reference particles in each phase.

        press_stat_indiv_dict: dictionary containing the average interparticle pressure and
        shear stress between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B'),
        averaged over all reference particles in each phase.

        press_stat_dict: dictionary containing the total interparticle pressure and
        shear stress between reference particles of a given type ('all', 'A', or 'B')
        and their nearest, interacting neighbors of each type ('all', 'A', or 'B'),
         over each phase.

        stress_plot_dict: dictionary containing information on the interparticle pressure
        of each bulk and interface reference particle of each type ('all', 'A', or 'B').

        press_plot_dict: dictionary containing information on the interparticle stress
        and pressure of each bulk and interface reference particle of each type
        ('all', 'A', or 'B').

        press_plot_indiv_dict: dictionary containing information on the interparticle
         pressure for reference particle of each type
        ('all', 'A', or 'B') in system.

        press_hetero_dict: dictionary containing information on the interparticle
         pressure of each reference particle of each type in each phase
        ('all', 'A', or 'B').
        '''

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(phase_dict['part'])

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(phase_dict)

        # Calculate area of each phase
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)
        int_area = phase_count_dict['int'] * (self.sizeBin_x * self.sizeBin_y)
        gas_area = phase_count_dict['gas'] * (self.sizeBin_x * self.sizeBin_y)
        dense_area = bulk_area + int_area
        system_area = bulk_area + int_area + gas_area

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_gas_int = self.pos[phase_part_dict['gas_int']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_gas_int = self.pos[phase_part_dict['gas_int']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_gas_int = self.pos[phase_part_dict['gas_int']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]
        id_system = np.append(phase_part_dict['dense']['all'], phase_part_dict['gas']['all'])

        # Position and orientation arrays of all particles in respective phase
        typ_bulk = self.typ[phase_part_dict['bulk']['all']]
        typ_int = self.typ[phase_part_dict['int']['all']]
        typ_gas = self.typ[phase_part_dict['gas']['all']]
        typ_gas_int = self.typ[phase_part_dict['gas_int']['all']]
        typ_dense = self.typ[phase_part_dict['dense']['all']]

        

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type A bulk particles
        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])

        SigXX_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of normal stress in x direction
        SigXY_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of tangential stress in x-y direction
        SigYX_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of tangential stress in y-x direction
        SigYY_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of normal stress in y direction

        SigXX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in normal x direction
        SigXY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in tangential x-y direction
        SigYX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in tangential y-x direction
        SigYY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in normal y direction

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_bulk)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AA_bulk_nlist.point_indices[loc]], difxy=True)

                        #Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_bulk_part[i] += np.sum(SigXX)
                        SigYY_AA_bulk_part[i] += np.sum(SigYY)
                        SigXY_AA_bulk_part[i] += np.sum(SigXY)
                        SigYX_AA_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_bulk_part_num[i] += len(SigXX)
                        SigYY_AA_bulk_part_num[i] += len(SigYY)
                        SigXY_AA_bulk_part_num[i] += len(SigXY)
                        SigYX_AA_bulk_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_bulk[i][0], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_bulk[i][1], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle force
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_bulk_part[i] = SigXX
                        SigYY_AA_bulk_part[i] = SigYY
                        SigXY_AA_bulk_part[i] = SigXY
                        SigYX_AA_bulk_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_bulk_part_num[i] += 1
                        SigYY_AA_bulk_part_num[i] += 1
                        SigXY_AA_bulk_part_num[i] += 1
                        SigYX_AA_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type A bulk particles
        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])

        SigXX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part=np.zeros(len(pos_A_bulk))

        SigXX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_bulk)):
            if i in BA_bulk_nlist.query_point_indices:
                if i not in BA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BA_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_bulk_part[i] += np.sum(SigXX)
                        SigYY_BA_bulk_part[i] += np.sum(SigYY)
                        SigXY_BA_bulk_part[i] += np.sum(SigXY)
                        SigYX_BA_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_bulk_part_num[i] += len(SigXX)
                        SigYY_BA_bulk_part_num[i] += len(SigYY)
                        SigXY_BA_bulk_part_num[i] += len(SigXY)
                        SigYX_BA_bulk_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_bulk[i][0], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_bulk[i][1], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_bulk_part[i] = SigXX
                        SigYY_BA_bulk_part[i] = SigYY
                        SigXY_BA_bulk_part[i] = SigXY
                        SigYX_BA_bulk_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_bulk_part_num[i] += 1
                        SigYY_BA_bulk_part_num[i] += 1
                        SigXY_BA_bulk_part_num[i] += 1
                        SigYX_BA_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type B bulk particles
        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])

        SigXX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))


        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_bulk)):
            if i in AB_bulk_nlist.query_point_indices:
                if i not in AB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AB_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_bulk_part[i] += np.sum(SigXX)
                        SigYY_AB_bulk_part[i] += np.sum(SigYY)
                        SigXY_AB_bulk_part[i] += np.sum(SigXY)
                        SigYX_AB_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_bulk_part_num[i] += len(SigXX)
                        SigYY_AB_bulk_part_num[i] += len(SigYY)
                        SigXY_AB_bulk_part_num[i] += len(SigXY)
                        SigYX_AB_bulk_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_bulk[i][0], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_bulk[i][1], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_bulk_part[i] += SigXX
                        SigYY_AB_bulk_part[i] += SigYY
                        SigXY_AB_bulk_part[i] += SigXY
                        SigYX_AB_bulk_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_bulk_part_num[i] += 1
                        SigYY_AB_bulk_part_num[i] += 1
                        SigXY_AB_bulk_part_num[i] += 1
                        SigYX_AB_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type B bulk particles
        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])
            
        SigXX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_bulk)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BB_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_bulk_part[i] += np.sum(SigXX)
                        SigYY_BB_bulk_part[i] += np.sum(SigYY)
                        SigXY_BB_bulk_part[i] += np.sum(SigXY)
                        SigYX_BB_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_bulk_part_num[i] += len(SigXX)
                        SigYY_BB_bulk_part_num[i] += len(SigYY)
                        SigXY_BB_bulk_part_num[i] += len(SigXY)
                        SigYX_BB_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist_x(pos_B_bulk[i][0], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_bulk[i][1], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_bulk_part[i] += SigXX
                        SigYY_BB_bulk_part[i] += SigYY
                        SigXY_BB_bulk_part[i] += SigXY
                        SigYX_BB_bulk_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_bulk_part_num[i] += 1
                        SigYY_BB_bulk_part_num[i] += 1
                        SigXY_BB_bulk_part_num[i] += 1
                        SigYX_BB_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))

        # Locate potential neighbor particles by type in the entire system
        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        # Generate neighbor list of any phase particles (per query args) of respective type (A or B) neighboring interface phase reference particles of respective type (A or B)
        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A particles acting on type A interface particles
        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])

        SigXX_AA_int_part=np.zeros(len(pos_A_int))
        SigXY_AA_int_part=np.zeros(len(pos_A_int))
        SigYX_AA_int_part=np.zeros(len(pos_A_int))
        SigYY_AA_int_part=np.zeros(len(pos_A_int))

        SigXX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_AA_int_part_num=np.zeros(len(pos_A_int))

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_int)):
            if i in AA_int_nlist.query_point_indices:
                if i not in AA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AA_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_int_part[i] += np.sum(SigXX)
                        SigYY_AA_int_part[i] += np.sum(SigYY)
                        SigXY_AA_int_part[i] += np.sum(SigXY)
                        SigYX_AA_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_int_part_num[i] += len(SigXX)
                        SigYY_AA_int_part_num[i] += len(SigYY)
                        SigXY_AA_int_part_num[i] += len(SigXY)
                        SigYX_AA_int_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_int[i][0], pos_A[AA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_int[i][1], pos_A[AA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_int_part[i] = SigXX
                        SigYY_AA_int_part[i] = SigYY
                        SigXY_AA_int_part[i] = SigXY
                        SigYX_AA_int_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_int_part_num[i] += 1
                        SigYY_AA_int_part_num[i] += 1
                        SigXY_AA_int_part_num[i] += 1
                        SigYX_AA_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type A interface particles
        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])

        SigXX_BA_int_part=np.zeros(len(pos_A_int))
        SigXY_BA_int_part=np.zeros(len(pos_A_int))
        SigYX_BA_int_part=np.zeros(len(pos_A_int))
        SigYY_BA_int_part=np.zeros(len(pos_A_int))

        SigXX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_BA_int_part_num=np.zeros(len(pos_A_int))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_int)):
            if i in BA_int_nlist.query_point_indices:
                if i not in BA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BA_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_int_part[i] += np.sum(SigXX)
                        SigYY_BA_int_part[i] += np.sum(SigYY)
                        SigXY_BA_int_part[i] += np.sum(SigXY)
                        SigYX_BA_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_int_part_num[i] += len(SigXX)
                        SigYY_BA_int_part_num[i] += len(SigYY)
                        SigXY_BA_int_part_num[i] += len(SigXY)
                        SigYX_BA_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_int[i][0], pos_B[BA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_int[i][1], pos_B[BA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_int_part[i] = SigXX
                        SigYY_BA_int_part[i] = SigYY
                        SigXY_BA_int_part[i] = SigXY
                        SigYX_BA_int_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_int_part_num[i] += 1
                        SigYY_BA_int_part_num[i] += 1
                        SigXY_BA_int_part_num[i] += 1
                        SigYX_BA_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type A interface particles
        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])

        SigXX_AB_int_part=np.zeros(len(pos_B_int))
        SigXY_AB_int_part=np.zeros(len(pos_B_int))
        SigYX_AB_int_part=np.zeros(len(pos_B_int))
        SigYY_AB_int_part=np.zeros(len(pos_B_int))

        SigXX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_AB_int_part_num=np.zeros(len(pos_B_int))

        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_int)):
            if i in AB_int_nlist.query_point_indices:
                if i not in AB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AB_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_int_part[i] += np.sum(SigXX)
                        SigYY_AB_int_part[i] += np.sum(SigYY)
                        SigXY_AB_int_part[i] += np.sum(SigXY)
                        SigYX_AB_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_int_part_num[i] += len(SigXX)
                        SigYY_AB_int_part_num[i] += len(SigYY)
                        SigXY_AB_int_part_num[i] += len(SigXY)
                        SigYX_AB_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_int[i][0], pos_A[AB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_int[i][1], pos_A[AB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_int_part[i] += SigXX
                        SigYY_AB_int_part[i] += SigYY
                        SigXY_AB_int_part[i] += SigXY
                        SigYX_AB_int_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_int_part_num[i] += 1
                        SigYY_AB_int_part_num[i] += 1
                        SigXY_AB_int_part_num[i] += 1
                        SigYX_AB_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type B interface particles
        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])

        SigXX_BB_int_part=np.zeros(len(pos_B_int))
        SigXY_BB_int_part=np.zeros(len(pos_B_int))
        SigYX_BB_int_part=np.zeros(len(pos_B_int))
        SigYY_BB_int_part=np.zeros(len(pos_B_int))

        SigXX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_BB_int_part_num=np.zeros(len(pos_B_int))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_int)):
            if i in BB_int_nlist.query_point_indices:
                if i not in BB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BB_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_int_part[i] += np.sum(SigXX)
                        SigYY_BB_int_part[i] += np.sum(SigYY)
                        SigXY_BB_int_part[i] += np.sum(SigXY)
                        SigYX_BB_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_int_part_num[i] += len(SigXX)
                        SigYY_BB_int_part_num[i] += len(SigYY)
                        SigXY_BB_int_part_num[i] += len(SigXY)
                        SigYX_BB_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_int[i][0], pos_B[BB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_int[i][1], pos_B[BB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_int_part[i] += SigXX
                        SigYY_BB_int_part[i] += SigYY
                        SigXY_BB_int_part[i] += SigXY
                        SigYX_BB_int_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_int_part_num[i] += 1
                        SigYY_BB_int_part_num[i] += 1
                        SigXY_BB_int_part_num[i] += 1
                        SigYX_BB_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))

        # Locate potential neighbor particles by type in the gas and interface phases
        system_A_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_gas_int))
        system_B_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_gas_int))

        # Generate neighbor list of gas and interface phase particles (per query args) of respective type (A or B) neighboring gas phase reference particles of respective type (A or B)
        AA_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        AB_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()
        BA_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        BB_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A gas and interface particles acting on type A gas particles
        AA_gas_neigh_ind = np.array([], dtype=int)
        AA_gas_num_neigh = np.array([])

        SigXX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part_num=np.zeros(len(pos_A_gas))

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_gas)):
            if i in AA_gas_nlist.query_point_indices:
                if i not in AA_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AA_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_gas_part[i] += np.sum(SigXX)
                        SigYY_AA_gas_part[i] += np.sum(SigYY)
                        SigXY_AA_gas_part[i] += np.sum(SigXY)
                        SigYX_AA_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_gas_part_num[i] += len(SigXX)
                        SigYY_AA_gas_part_num[i] += len(SigYY)
                        SigXY_AA_gas_part_num[i] += len(SigXY)
                        SigYX_AA_gas_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_gas[i][0], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_gas[i][1], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_gas_part[i] = SigXX
                        SigYY_AA_gas_part[i] = SigYY
                        SigXY_AA_gas_part[i] = SigXY
                        SigYX_AA_gas_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_gas_part_num[i] += 1
                        SigYY_AA_gas_part_num[i] += 1
                        SigXY_AA_gas_part_num[i] += 1
                        SigYX_AA_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_gas_num_neigh = np.append(AA_gas_num_neigh, len(loc))
                    AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AA_gas_num_neigh = np.append(AA_gas_num_neigh, 0)
                AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B gas and interface particles acting on type A gas particles
        BA_gas_neigh_ind = np.array([], dtype=int)
        BA_gas_num_neigh = np.array([])

        SigXX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part_num=np.zeros(len(pos_A_gas))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_gas)):
            if i in BA_gas_nlist.query_point_indices:
                if i not in BA_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BA_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_gas_part[i] += np.sum(SigXX)
                        SigYY_BA_gas_part[i] += np.sum(SigYY)
                        SigXY_BA_gas_part[i] += np.sum(SigXY)
                        SigYX_BA_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_gas_part_num[i] += len(SigXX)
                        SigYY_BA_gas_part_num[i] += len(SigYY)
                        SigXY_BA_gas_part_num[i] += len(SigXY)
                        SigYX_BA_gas_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_gas[i][0], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_gas[i][1], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_gas_part[i] = SigXX
                        SigYY_BA_gas_part[i] = SigYY
                        SigXY_BA_gas_part[i] = SigXY
                        SigYX_BA_gas_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_gas_part_num[i] += 1
                        SigYY_BA_gas_part_num[i] += 1
                        SigXY_BA_gas_part_num[i] += 1
                        SigYX_BA_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_gas_num_neigh = np.append(BA_gas_num_neigh, len(loc))
                    BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BA_gas_num_neigh = np.append(BA_gas_num_neigh, 0)
                BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type A gas and interface particles acting on type B gas particles
        AB_gas_neigh_ind = np.array([], dtype=int)
        AB_gas_num_neigh = np.array([])

        SigXX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part_num=np.zeros(len(pos_B_gas))

        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_gas)):
            if i in AB_gas_nlist.query_point_indices:
                if i not in AB_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AB_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_gas_part[i] += np.sum(SigXX)
                        SigYY_AB_gas_part[i] += np.sum(SigYY)
                        SigXY_AB_gas_part[i] += np.sum(SigXY)
                        SigYX_AB_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_gas_part_num[i] += len(SigXX)
                        SigYY_AB_gas_part_num[i] += len(SigYY)
                        SigXY_AB_gas_part_num[i] += len(SigXY)
                        SigYX_AB_gas_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_gas[i][0], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_gas[i][1], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_gas_part[i] += SigXX
                        SigYY_AB_gas_part[i] += SigYY
                        SigXY_AB_gas_part[i] += SigXY
                        SigYX_AB_gas_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_gas_part_num[i] += 1
                        SigYY_AB_gas_part_num[i] += 1
                        SigXY_AB_gas_part_num[i] += 1
                        SigYX_AB_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_gas_num_neigh = np.append(AB_gas_num_neigh, len(loc))
                    AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_gas_num_neigh = np.append(AB_gas_num_neigh, 0)
                AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B gas and interface particles acting on type B gas particles
        BB_gas_neigh_ind = np.array([], dtype=int)
        BB_gas_num_neigh = np.array([])

        SigXX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part_num=np.zeros(len(pos_B_gas))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_gas)):
            if i in BB_gas_nlist.query_point_indices:
                if i not in BB_gas_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BB_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_gas_part[i] += np.sum(SigXX)
                        SigYY_BB_gas_part[i] += np.sum(SigYY)
                        SigXY_BB_gas_part[i] += np.sum(SigXY)
                        SigYX_BB_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_gas_part_num[i] += len(SigXX)
                        SigYY_BB_gas_part_num[i] += len(SigYY)
                        SigXY_BB_gas_part_num[i] += len(SigXY)
                        SigYX_BB_gas_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_gas[i][0], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_gas[i][1], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_gas_part[i] += SigXX
                        SigYY_BB_gas_part[i] += SigYY
                        SigXY_BB_gas_part[i] += SigXY
                        SigYX_BB_gas_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_gas_part_num[i] += 1
                        SigYY_BB_gas_part_num[i] += 1
                        SigXY_BB_gas_part_num[i] += 1
                        SigYX_BB_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_gas_num_neigh = np.append(BB_gas_num_neigh, len(loc))
                    BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BB_gas_num_neigh = np.append(BB_gas_num_neigh, 0)
                BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))


        ###Bulk stress

        # Calculate total stress and number of neighbor pairs summed over for B bulk reference particles and all dense neighbors
        allB_bulk_SigXX_part = SigXX_BB_bulk_part + SigXX_AB_bulk_part
        allB_bulk_SigXX_part_num = SigXX_BB_bulk_part_num + SigXX_AB_bulk_part_num
        allB_bulk_SigXY_part = SigXY_BB_bulk_part + SigXY_AB_bulk_part
        allB_bulk_SigXY_part_num = SigXY_BB_bulk_part_num + SigXY_AB_bulk_part_num
        allB_bulk_SigYX_part = SigYX_BB_bulk_part + SigYX_AB_bulk_part
        allB_bulk_SigYX_part_num = SigYX_BB_bulk_part_num + SigYX_AB_bulk_part_num
        allB_bulk_SigYY_part = SigYY_BB_bulk_part + SigYY_AB_bulk_part
        allB_bulk_SigYY_part_num = SigYY_BB_bulk_part_num + SigYY_AB_bulk_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and B dense neighbors
        Ball_bulk_SigXX_part = np.append(SigXX_BA_bulk_part, SigXX_BB_bulk_part)
        Ball_bulk_SigXX_part_num = np.append(SigXX_BA_bulk_part_num, SigXX_BB_bulk_part_num)
        Ball_bulk_SigXY_part = np.append(SigXY_BA_bulk_part, SigXY_BB_bulk_part)
        Ball_bulk_SigXY_part_num = np.append(SigXY_BA_bulk_part_num, SigXY_BB_bulk_part_num)
        Ball_bulk_SigYX_part = np.append(SigYX_BA_bulk_part, SigYX_BB_bulk_part)
        Ball_bulk_SigYX_part_num = np.append(SigYX_BA_bulk_part_num, SigYX_BB_bulk_part_num)
        Ball_bulk_SigYY_part = np.append(SigYY_BA_bulk_part, SigYY_BB_bulk_part)
        Ball_bulk_SigYY_part_num = np.append(SigYY_BA_bulk_part_num, SigYY_BB_bulk_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A bulk reference particles and all dense neighbors
        allA_bulk_SigXX_part = SigXX_AA_bulk_part + SigXX_BA_bulk_part
        allA_bulk_SigXX_part_num = SigXX_AA_bulk_part_num + SigXX_BA_bulk_part_num
        allA_bulk_SigXY_part = SigXY_AA_bulk_part + SigXY_BA_bulk_part
        allA_bulk_SigXY_part_num = SigXY_AA_bulk_part_num + SigXY_BA_bulk_part_num
        allA_bulk_SigYX_part = SigYX_AA_bulk_part + SigYX_BA_bulk_part
        allA_bulk_SigYX_part_num = SigYX_AA_bulk_part_num + SigYX_BA_bulk_part_num
        allA_bulk_SigYY_part = SigYY_AA_bulk_part + SigYY_BA_bulk_part
        allA_bulk_SigYY_part_num = SigYY_AA_bulk_part_num + SigYY_BA_bulk_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and A dense neighbors
        Aall_bulk_SigXX_part = np.append(SigXX_AB_bulk_part, SigXX_AA_bulk_part)
        Aall_bulk_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_bulk_part_num)
        Aall_bulk_SigXY_part = np.append(SigXY_AB_bulk_part, SigXY_AA_bulk_part)
        Aall_bulk_SigXY_part_num = np.append(SigXY_AB_bulk_part_num, SigXY_AA_bulk_part_num)
        Aall_bulk_SigYX_part = np.append(SigYX_AB_bulk_part, SigYX_AA_bulk_part)
        Aall_bulk_SigYX_part_num = np.append(SigYX_AB_bulk_part_num, SigYX_AA_bulk_part_num)
        Aall_bulk_SigYY_part = np.append(SigYY_AB_bulk_part, SigYY_AA_bulk_part)
        Aall_bulk_SigYY_part_num = np.append(SigYY_AB_bulk_part_num, SigYY_AA_bulk_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and all dense neighbors
        allall_bulk_SigXX_part = np.append(allA_bulk_SigXX_part, allB_bulk_SigXX_part)
        allall_bulk_SigXX_part_num = np.append(allA_bulk_SigXX_part_num, allB_bulk_SigXX_part_num)
        allall_bulk_SigXY_part = np.append(allA_bulk_SigXY_part, allB_bulk_SigXY_part)
        allall_bulk_SigXY_part_num = np.append(allA_bulk_SigXY_part_num, allB_bulk_SigXY_part_num)
        allall_bulk_SigYX_part = np.append(allA_bulk_SigYX_part, allB_bulk_SigYX_part)
        allall_bulk_SigYX_part_num = np.append(allA_bulk_SigYX_part_num, allB_bulk_SigYX_part_num)
        allall_bulk_SigYY_part = np.append(allA_bulk_SigYY_part, allB_bulk_SigYY_part)
        allall_bulk_SigYY_part_num = np.append(allA_bulk_SigYY_part_num, allB_bulk_SigYY_part_num)

        ###Interface stress

        # Calculate total stress and number of neighbor pairs summed over for B interface reference particles and all neighbors
        allB_int_SigXX_part = SigXX_BB_int_part + SigXX_AB_int_part
        allB_int_SigXX_part_num = SigXX_BB_int_part_num + SigXX_AB_int_part_num
        allB_int_SigXY_part = SigXY_BB_int_part + SigXY_AB_int_part
        allB_int_SigXY_part_num = SigXY_BB_int_part_num + SigXY_AB_int_part_num
        allB_int_SigYX_part = SigYX_BB_int_part + SigYX_AB_int_part
        allB_int_SigYX_part_num = SigYX_BB_int_part_num + SigYX_AB_int_part_num
        allB_int_SigYY_part = SigYY_BB_int_part + SigYY_AB_int_part
        allB_int_SigYY_part_num = SigYY_BB_int_part_num + SigYY_AB_int_part_num

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and B neighbors
        Ball_int_SigXX_part = np.append(SigXX_BA_int_part, SigXX_BB_int_part)
        Ball_int_SigXX_part_num = np.append(SigXX_BA_int_part_num, SigXX_BB_int_part_num)
        Ball_int_SigXY_part = np.append(SigXY_BA_int_part, SigXY_BB_int_part)
        Ball_int_SigXY_part_num = np.append(SigXY_BA_int_part_num, SigXY_BB_int_part_num)
        Ball_int_SigYX_part = np.append(SigYX_BA_int_part, SigYX_BB_int_part)
        Ball_int_SigYX_part_num = np.append(SigYX_BA_int_part_num, SigYX_BB_int_part_num)
        Ball_int_SigYY_part = np.append(SigYY_BA_int_part, SigYY_BB_int_part)
        Ball_int_SigYY_part_num = np.append(SigYY_BA_int_part_num, SigYY_BB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A interface reference particles and all neighbors
        allA_int_SigXX_part = SigXX_AA_int_part + SigXX_BA_int_part
        allA_int_SigXX_part_num = SigXX_AA_int_part_num + SigXX_BA_int_part_num
        allA_int_SigXY_part = SigXY_AA_int_part + SigXY_BA_int_part
        allA_int_SigXY_part_num = SigXY_AA_int_part_num + SigXY_BA_int_part_num
        allA_int_SigYX_part = SigYX_AA_int_part + SigYX_BA_int_part
        allA_int_SigYX_part_num = SigYX_AA_int_part_num + SigYX_BA_int_part_num
        allA_int_SigYY_part = SigYY_AA_int_part + SigYY_BA_int_part
        allA_int_SigYY_part_num = SigYY_AA_int_part_num + SigYY_BA_int_part_num

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and A neighbors
        Aall_int_SigXX_part = np.append(SigXX_AB_int_part, SigXX_AA_int_part)
        Aall_int_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_int_part_num)
        Aall_int_SigXY_part = np.append(SigXY_AB_int_part, SigXY_AA_int_part)
        Aall_int_SigXY_part_num = np.append(SigXY_AB_int_part_num, SigXY_AA_int_part_num)
        Aall_int_SigYX_part = np.append(SigYX_AB_int_part, SigYX_AA_int_part)
        Aall_int_SigYX_part_num = np.append(SigYX_AB_int_part_num, SigYX_AA_int_part_num)
        Aall_int_SigYY_part = np.append(SigYY_AB_int_part, SigYY_AA_int_part)
        Aall_int_SigYY_part_num = np.append(SigYY_AB_int_part_num, SigYY_AA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and all neighbors
        allall_int_SigXX_part = np.append(allA_int_SigXX_part, allB_int_SigXX_part)
        allall_int_SigXX_part_num = np.append(allA_int_SigXX_part_num, allB_int_SigXX_part_num)
        allall_int_SigXY_part = np.append(allA_int_SigXY_part, allB_int_SigXY_part)
        allall_int_SigXY_part_num = np.append(allA_int_SigXY_part_num, allB_int_SigXY_part_num)
        allall_int_SigYX_part = np.append(allA_int_SigYX_part, allB_int_SigYX_part)
        allall_int_SigYX_part_num = np.append(allA_int_SigYX_part_num, allB_int_SigYX_part_num)
        allall_int_SigYY_part = np.append(allA_int_SigYY_part, allB_int_SigYY_part)
        allall_int_SigYY_part_num = np.append(allA_int_SigYY_part_num, allB_int_SigYY_part_num)

        ###Gas stress

        # Calculate total stress and number of neighbor pairs summed over for B gas reference particles and all gas and interface neighbors
        allB_gas_SigXX_part = SigXX_BB_gas_part + SigXX_AB_gas_part
        allB_gas_SigXX_part_num = SigXX_BB_gas_part_num + SigXX_AB_gas_part_num
        allB_gas_SigXY_part = SigXY_BB_gas_part + SigXY_AB_gas_part
        allB_gas_SigXY_part_num = SigXY_BB_gas_part_num + SigXY_AB_gas_part_num
        allB_gas_SigYX_part = SigYX_BB_gas_part + SigYX_AB_gas_part
        allB_gas_SigYX_part_num = SigYX_BB_gas_part_num + SigYX_AB_gas_part_num
        allB_gas_SigYY_part = SigYY_BB_gas_part + SigYY_AB_gas_part
        allB_gas_SigYY_part_num = SigYY_BB_gas_part_num + SigYY_AB_gas_part_num

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and B gas and interface neighbors
        Ball_gas_SigXX_part = np.append(SigXX_BA_gas_part, SigXX_BB_gas_part)
        Ball_gas_SigXX_part_num = np.append(SigXX_BA_gas_part_num, SigXX_BB_gas_part_num)
        Ball_gas_SigXY_part = np.append(SigXY_BA_gas_part, SigXY_BB_gas_part)
        Ball_gas_SigXY_part_num = np.append(SigXY_BA_gas_part_num, SigXY_BB_gas_part_num)
        Ball_gas_SigYX_part = np.append(SigYX_BA_gas_part, SigYX_BB_gas_part)
        Ball_gas_SigYX_part_num = np.append(SigYX_BA_gas_part_num, SigYX_BB_gas_part_num)
        Ball_gas_SigYY_part = np.append(SigYY_BA_gas_part, SigYY_BB_gas_part)
        Ball_gas_SigYY_part_num = np.append(SigYY_BA_gas_part_num, SigYY_BB_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A gas reference particles and all gas and interface neighbors
        allA_gas_SigXX_part = SigXX_AA_gas_part + SigXX_BA_gas_part
        allA_gas_SigXX_part_num = SigXX_AA_gas_part_num + SigXX_BA_gas_part_num
        allA_gas_SigXY_part = SigXY_AA_gas_part + SigXY_BA_gas_part
        allA_gas_SigXY_part_num = SigXY_AA_gas_part_num + SigXY_BA_gas_part_num
        allA_gas_SigYX_part = SigYX_AA_gas_part + SigYX_BA_gas_part
        allA_gas_SigYX_part_num = SigYX_AA_gas_part_num + SigYX_BA_gas_part_num
        allA_gas_SigYY_part = SigYY_AA_gas_part + SigYY_BA_gas_part
        allA_gas_SigYY_part_num = SigYY_AA_gas_part_num + SigYY_BA_gas_part_num

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and A gas and interface neighbors
        Aall_gas_SigXX_part = np.append(SigXX_AB_gas_part, SigXX_AA_gas_part)
        Aall_gas_SigXX_part_num = np.append(SigXX_AB_gas_part_num, SigXX_AA_gas_part_num)
        Aall_gas_SigXY_part = np.append(SigXY_AB_gas_part, SigXY_AA_gas_part)
        Aall_gas_SigXY_part_num = np.append(SigXY_AB_gas_part_num, SigXY_AA_gas_part_num)
        Aall_gas_SigYX_part = np.append(SigYX_AB_gas_part, SigYX_AA_gas_part)
        Aall_gas_SigYX_part_num = np.append(SigYX_AB_gas_part_num, SigYX_AA_gas_part_num)
        Aall_gas_SigYY_part = np.append(SigYY_AB_gas_part, SigYY_AA_gas_part)
        Aall_gas_SigYY_part_num = np.append(SigYY_AB_gas_part_num, SigYY_AA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and all gas and interface neighbors
        allall_gas_SigXX_part = np.append(allA_gas_SigXX_part, allB_gas_SigXX_part)
        allall_gas_SigXX_part_num = np.append(allA_gas_SigXX_part_num, allB_gas_SigXX_part_num)
        allall_gas_SigXY_part = np.append(allA_gas_SigXY_part, allB_gas_SigXY_part)
        allall_gas_SigXY_part_num = np.append(allA_gas_SigXY_part_num, allB_gas_SigXY_part_num)
        allall_gas_SigYX_part = np.append(allA_gas_SigYX_part, allB_gas_SigYX_part)
        allall_gas_SigYX_part_num = np.append(allA_gas_SigYX_part_num, allB_gas_SigYX_part_num)
        allall_gas_SigYY_part = np.append(allA_gas_SigYY_part, allB_gas_SigYY_part)
        allall_gas_SigYY_part_num = np.append(allA_gas_SigYY_part_num, allB_gas_SigYY_part_num)

        ###Dense stress

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and all neighbors
        allB_dense_SigXX_part = np.append(allB_bulk_SigXX_part, allB_int_SigXX_part)
        allB_dense_SigXX_part_num = np.append(allB_bulk_SigXX_part_num, allB_int_SigXX_part_num)
        allB_dense_SigYX_part = np.append(allB_bulk_SigYX_part, allB_int_SigYX_part)
        allB_dense_SigYX_part_num = np.append(allB_bulk_SigYX_part_num, allB_int_SigYX_part_num)
        allB_dense_SigXY_part = np.append(allB_bulk_SigXY_part, allB_int_SigXY_part)
        allB_dense_SigXY_part_num = np.append(allB_bulk_SigXY_part_num, allB_int_SigXY_part_num)
        allB_dense_SigYY_part = np.append(allB_bulk_SigYY_part, allB_int_SigYY_part)
        allB_dense_SigYY_part_num = np.append(allB_bulk_SigYY_part_num, allB_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and B neighbors
        Ball_dense_SigXX_part = np.append(Ball_bulk_SigXX_part, Ball_int_SigXX_part)
        Ball_dense_SigXX_part_num = np.append(Ball_bulk_SigXX_part_num, Ball_int_SigXX_part_num)
        Ball_dense_SigYX_part = np.append(Ball_bulk_SigYX_part, Ball_int_SigYX_part)
        Ball_dense_SigYX_part_num = np.append(Ball_bulk_SigYX_part_num, Ball_int_SigYX_part_num)
        Ball_dense_SigXY_part = np.append(Ball_bulk_SigXY_part, Ball_int_SigXY_part)
        Ball_dense_SigXY_part_num = np.append(Ball_bulk_SigXY_part_num, Ball_int_SigXY_part_num)
        Ball_dense_SigYY_part = np.append(Ball_bulk_SigYY_part, Ball_int_SigYY_part)
        Ball_dense_SigYY_part_num = np.append(Ball_bulk_SigYY_part_num, Ball_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and all neighbors
        allA_dense_SigXX_part = np.append(allA_bulk_SigXX_part, allA_int_SigXX_part)
        allA_dense_SigXX_part_num = np.append(allA_bulk_SigXX_part_num, allA_int_SigXX_part_num)
        allA_dense_SigYX_part = np.append(allA_bulk_SigYX_part, allA_int_SigYX_part)
        allA_dense_SigYX_part_num = np.append(allA_bulk_SigYX_part_num, allA_int_SigYX_part_num)
        allA_dense_SigXY_part = np.append(allA_bulk_SigXY_part, allA_int_SigXY_part)
        allA_dense_SigXY_part_num = np.append(allA_bulk_SigXY_part_num, allA_int_SigXY_part_num)
        allA_dense_SigYY_part = np.append(allA_bulk_SigYY_part, allA_int_SigYY_part)
        allA_dense_SigYY_part_num = np.append(allA_bulk_SigYY_part_num, allA_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and A neighbors
        Aall_dense_SigXX_part = np.append(Aall_bulk_SigXX_part, Aall_int_SigXX_part)
        Aall_dense_SigXX_part_num = np.append(Aall_bulk_SigXX_part_num, Aall_int_SigXX_part_num)
        Aall_dense_SigYX_part = np.append(Aall_bulk_SigYX_part, Aall_int_SigYX_part)
        Aall_dense_SigYX_part_num = np.append(Aall_bulk_SigYX_part_num, Aall_int_SigYX_part_num)
        Aall_dense_SigXY_part = np.append(Aall_bulk_SigXY_part, Aall_int_SigXY_part)
        Aall_dense_SigXY_part_num = np.append(Aall_bulk_SigXY_part_num, Aall_int_SigXY_part_num)
        Aall_dense_SigYY_part = np.append(Aall_bulk_SigYY_part, Aall_int_SigYY_part)
        Aall_dense_SigYY_part_num = np.append(Aall_bulk_SigYY_part_num, Aall_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and all neighbors
        allall_dense_SigXX_part = np.append(allall_bulk_SigXX_part, allall_int_SigXX_part)
        allall_dense_SigXX_part_num = np.append(allall_bulk_SigXX_part_num, allall_int_SigXX_part_num)
        allall_dense_SigYX_part = np.append(allall_bulk_SigYX_part, allall_int_SigYX_part)
        allall_dense_SigYX_part_num = np.append(allall_bulk_SigYX_part_num, allall_int_SigYX_part_num)
        allall_dense_SigXY_part = np.append(allall_bulk_SigXY_part, allall_int_SigXY_part)
        allall_dense_SigXY_part_num = np.append(allall_bulk_SigXY_part_num, allall_int_SigXY_part_num)
        allall_dense_SigYY_part = np.append(allall_bulk_SigYY_part, allall_int_SigYY_part)
        allall_dense_SigYY_part_num = np.append(allall_bulk_SigYY_part_num, allall_int_SigYY_part_num)


        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and A neighbors
        SigXX_AA_dense_part = np.append(SigXX_AA_bulk_part, SigXX_AA_int_part)
        SigXX_AA_dense_part_num = np.append(SigXX_AA_bulk_part_num, SigXX_AA_int_part_num)
        SigYX_AA_dense_part = np.append(SigYX_AA_bulk_part, SigYX_AA_int_part)
        SigYX_AA_dense_part_num = np.append(SigYX_AA_bulk_part_num, SigYX_AA_int_part_num)
        SigXY_AA_dense_part = np.append(SigXY_AA_bulk_part, SigXY_AA_int_part)
        SigXY_AA_dense_part_num = np.append(SigXY_AA_bulk_part_num, SigXY_AA_int_part_num)
        SigYY_AA_dense_part = np.append(SigYY_AA_bulk_part, SigYY_AA_int_part)
        SigYY_AA_dense_part_num = np.append(SigYY_AA_bulk_part_num, SigYY_AA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and A neighbors
        SigXX_AB_dense_part = np.append(SigXX_AB_bulk_part, SigXX_AB_int_part)
        SigXX_AB_dense_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AB_int_part_num)
        SigYX_AB_dense_part = np.append(SigYX_AB_bulk_part, SigYX_AB_int_part)
        SigYX_AB_dense_part_num = np.append(SigYX_AB_bulk_part_num, SigYX_AB_int_part_num)
        SigXY_AB_dense_part = np.append(SigXY_AB_bulk_part, SigXY_AB_int_part)
        SigXY_AB_dense_part_num = np.append(SigXY_AB_bulk_part_num, SigXY_AB_int_part_num)
        SigYY_AB_dense_part = np.append(SigYY_AB_bulk_part, SigYY_AB_int_part)
        SigYY_AB_dense_part_num = np.append(SigYY_AB_bulk_part_num, SigYY_AB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and B neighbors
        SigXX_BA_dense_part = np.append(SigXX_BA_bulk_part, SigXX_BA_int_part)
        SigXX_BA_dense_part_num = np.append(SigXX_BA_bulk_part_num, SigXX_BA_int_part_num)
        SigYX_BA_dense_part = np.append(SigYX_BA_bulk_part, SigYX_BA_int_part)
        SigYX_BA_dense_part_num = np.append(SigYX_BA_bulk_part_num, SigYX_BA_int_part_num)
        SigXY_BA_dense_part = np.append(SigXY_BA_bulk_part, SigXY_BA_int_part)
        SigXY_BA_dense_part_num = np.append(SigXY_BA_bulk_part_num, SigXY_BA_int_part_num)
        SigYY_BA_dense_part = np.append(SigYY_BA_bulk_part, SigYY_BA_int_part)
        SigYY_BA_dense_part_num = np.append(SigYY_BA_bulk_part_num, SigYY_BA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and B neighbors
        SigXX_BB_dense_part = np.append(SigXX_BB_bulk_part, SigXX_BB_int_part)
        SigXX_BB_dense_part_num = np.append(SigXX_BB_bulk_part_num, SigXX_BB_int_part_num)
        SigYX_BB_dense_part = np.append(SigYX_BB_bulk_part, SigYX_BB_int_part)
        SigYX_BB_dense_part_num = np.append(SigYX_BB_bulk_part_num, SigYX_BB_int_part_num)
        SigXY_BB_dense_part = np.append(SigXY_BB_bulk_part, SigXY_BB_int_part)
        SigXY_BB_dense_part_num = np.append(SigXY_BB_bulk_part_num, SigXY_BB_int_part_num)
        SigYY_BB_dense_part = np.append(SigYY_BB_bulk_part, SigYY_BB_int_part)
        SigYY_BB_dense_part_num = np.append(SigYY_BB_bulk_part_num, SigYY_BB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and all neighbors
        allB_system_SigXX_part = np.append(allB_dense_SigXX_part, allB_gas_SigXX_part)
        allB_system_SigXX_part_num = np.append(allB_dense_SigXX_part_num, allB_gas_SigXX_part_num)
        allB_system_SigYX_part = np.append(allB_dense_SigYX_part, allB_gas_SigYX_part)
        allB_system_SigYX_part_num = np.append(allB_dense_SigYX_part_num, allB_gas_SigYX_part_num)
        allB_system_SigXY_part = np.append(allB_dense_SigXY_part, allB_gas_SigXY_part)
        allB_system_SigXY_part_num = np.append(allB_dense_SigXY_part_num, allB_gas_SigXY_part_num)
        allB_system_SigYY_part = np.append(allB_dense_SigYY_part, allB_gas_SigYY_part)
        allB_system_SigYY_part_num = np.append(allB_dense_SigYY_part_num, allB_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and B neighbors
        Ball_system_SigXX_part = np.append(Ball_dense_SigXX_part, Ball_gas_SigXX_part)
        Ball_system_SigXX_part_num = np.append(Ball_dense_SigXX_part_num, Ball_gas_SigXX_part_num)
        Ball_system_SigYX_part = np.append(Ball_dense_SigYX_part, Ball_gas_SigYX_part)
        Ball_system_SigYX_part_num = np.append(Ball_dense_SigYX_part_num, Ball_gas_SigYX_part_num)
        Ball_system_SigXY_part = np.append(Ball_dense_SigXY_part, Ball_gas_SigXY_part)
        Ball_system_SigXY_part_num = np.append(Ball_dense_SigXY_part_num, Ball_gas_SigXY_part_num)
        Ball_system_SigYY_part = np.append(Ball_dense_SigYY_part, Ball_gas_SigYY_part)
        Ball_system_SigYY_part_num = np.append(Ball_dense_SigYY_part_num, Ball_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and all neighbors
        allA_system_SigXX_part = np.append(allA_dense_SigXX_part, allA_gas_SigXX_part)
        allA_system_SigXX_part_num = np.append(allA_dense_SigXX_part_num, allA_gas_SigXX_part_num)
        allA_system_SigYX_part = np.append(allA_dense_SigYX_part, allA_gas_SigYX_part)
        allA_system_SigYX_part_num = np.append(allA_dense_SigYX_part_num, allA_gas_SigYX_part_num)
        allA_system_SigXY_part = np.append(allA_dense_SigXY_part, allA_gas_SigXY_part)
        allA_system_SigXY_part_num = np.append(allA_dense_SigXY_part_num, allA_gas_SigXY_part_num)
        allA_system_SigYY_part = np.append(allA_dense_SigYY_part, allA_gas_SigYY_part)
        allA_system_SigYY_part_num = np.append(allA_dense_SigYY_part_num, allA_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and A neighbors
        Aall_system_SigXX_part = np.append(Aall_dense_SigXX_part, Aall_gas_SigXX_part)
        Aall_system_SigXX_part_num = np.append(Aall_dense_SigXX_part_num, Aall_gas_SigXX_part_num)
        Aall_system_SigYX_part = np.append(Aall_dense_SigYX_part, Aall_gas_SigYX_part)
        Aall_system_SigYX_part_num = np.append(Aall_dense_SigYX_part_num, Aall_gas_SigYX_part_num)
        Aall_system_SigXY_part = np.append(Aall_dense_SigXY_part, Aall_gas_SigXY_part)
        Aall_system_SigXY_part_num = np.append(Aall_dense_SigXY_part_num, Aall_gas_SigXY_part_num)
        Aall_system_SigYY_part = np.append(Aall_dense_SigYY_part, Aall_gas_SigYY_part)
        Aall_system_SigYY_part_num = np.append(Aall_dense_SigYY_part_num, Aall_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and all neighbors
        allall_system_SigXX_part = np.append(allall_dense_SigXX_part, allall_gas_SigXX_part)
        allall_system_SigXX_part_num = np.append(allall_dense_SigXX_part_num, allall_gas_SigXX_part_num)
        allall_system_SigYX_part = np.append(allall_dense_SigYX_part, allall_gas_SigYX_part)
        allall_system_SigYX_part_num = np.append(allall_dense_SigYX_part_num, allall_gas_SigYX_part_num)
        allall_system_SigXY_part = np.append(allall_dense_SigXY_part, allall_gas_SigXY_part)
        allall_system_SigXY_part_num = np.append(allall_dense_SigXY_part_num, allall_gas_SigXY_part_num)
        allall_system_SigYY_part = np.append(allall_dense_SigYY_part, allall_gas_SigYY_part)
        allall_system_SigYY_part_num = np.append(allall_dense_SigYY_part_num, allall_gas_SigYY_part_num)


        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and A neighbors
        SigXX_AA_system_part = np.append(SigXX_AA_dense_part, SigXX_AA_gas_part)
        SigXX_AA_system_part_num = np.append(SigXX_AA_dense_part_num, SigXX_AA_gas_part_num)
        SigYX_AA_system_part = np.append(SigYX_AA_dense_part, SigYX_AA_gas_part)
        SigYX_AA_system_part_num = np.append(SigYX_AA_dense_part_num, SigYX_AA_gas_part_num)
        SigXY_AA_system_part = np.append(SigXY_AA_dense_part, SigXY_AA_gas_part)
        SigXY_AA_system_part_num = np.append(SigXY_AA_dense_part_num, SigXY_AA_gas_part_num)
        SigYY_AA_system_part = np.append(SigYY_AA_dense_part, SigYY_AA_gas_part)
        SigYY_AA_system_part_num = np.append(SigYY_AA_dense_part_num, SigYY_AA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and A neighbors
        SigXX_AB_system_part = np.append(SigXX_AB_dense_part, SigXX_AB_gas_part)
        SigXX_AB_system_part_num = np.append(SigXX_AB_dense_part_num, SigXX_AB_gas_part_num)
        SigYX_AB_system_part = np.append(SigYX_AB_dense_part, SigYX_AB_gas_part)
        SigYX_AB_system_part_num = np.append(SigYX_AB_dense_part_num, SigYX_AB_gas_part_num)
        SigXY_AB_system_part = np.append(SigXY_AB_dense_part, SigXY_AB_gas_part)
        SigXY_AB_system_part_num = np.append(SigXY_AB_dense_part_num, SigXY_AB_gas_part_num)
        SigYY_AB_system_part = np.append(SigYY_AB_dense_part, SigYY_AB_gas_part)
        SigYY_AB_system_part_num = np.append(SigYY_AB_dense_part_num, SigYY_AB_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and B neighbors
        SigXX_BA_system_part = np.append(SigXX_BA_dense_part, SigXX_BA_gas_part)
        SigXX_BA_system_part_num = np.append(SigXX_BA_dense_part_num, SigXX_BA_gas_part_num)
        SigYX_BA_system_part = np.append(SigYX_BA_dense_part, SigYX_BA_gas_part)
        SigYX_BA_system_part_num = np.append(SigYX_BA_dense_part_num, SigYX_BA_gas_part_num)
        SigXY_BA_system_part = np.append(SigXY_BA_dense_part, SigXY_BA_gas_part)
        SigXY_BA_system_part_num = np.append(SigXY_BA_dense_part_num, SigXY_BA_gas_part_num)
        SigYY_BA_system_part = np.append(SigYY_BA_dense_part, SigYY_BA_gas_part)
        SigYY_BA_system_part_num = np.append(SigYY_BA_dense_part_num, SigYY_BA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and B neighbors
        SigXX_BB_system_part = np.append(SigXX_BB_dense_part, SigXX_BB_gas_part)
        SigXX_BB_system_part_num = np.append(SigXX_BB_dense_part_num, SigXX_BB_gas_part_num)
        SigYX_BB_system_part = np.append(SigYX_BB_dense_part, SigYX_BB_gas_part)
        SigYX_BB_system_part_num = np.append(SigYX_BB_dense_part_num, SigYX_BB_gas_part_num)
        SigXY_BB_system_part = np.append(SigXY_BB_dense_part, SigXY_BB_gas_part)
        SigXY_BB_system_part_num = np.append(SigXY_BB_dense_part_num, SigXY_BB_gas_part_num)
        SigYY_BB_system_part = np.append(SigYY_BB_dense_part, SigYY_BB_gas_part)
        SigYY_BB_system_part_num = np.append(SigYY_BB_dense_part_num, SigYY_BB_gas_part_num)

        typ_system = np.append(typ_dense, typ_gas)
        ###Interparticle pressure

        # Calculate total interparticle pressure experienced by all particles in each phase
        allall_bulk_int_press = np.sum(allall_bulk_SigXX_part + allall_bulk_SigYY_part)/(4*bulk_area)
        allall_gas_int_press = np.sum(allall_gas_SigXX_part + allall_gas_SigYY_part)/(4*gas_area)
        allall_int_int_press = np.sum(allall_int_SigXX_part + allall_int_SigYY_part)/(4*int_area)
        allall_dense_int_press = np.sum(allall_dense_SigXX_part + allall_dense_SigYY_part)/(4*dense_area)
        allall_system_int_press = np.sum(allall_system_SigXX_part + allall_system_SigYY_part)/(4*system_area)
        allall_int_press = np.append(allall_bulk_int_press, allall_int_int_press)
        allall_int_press = np.append(allall_int_press, allall_gas_int_press)
        allall_dense_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        allall_system_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        allall_system_part_ids = np.append(allall_system_part_ids, phase_part_dict['gas']['all'])


        # Calculate total interparticle pressure experienced by each particles in each phase
        allall_bulk_int_press_indiv = (allall_bulk_SigXX_part + allall_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_gas_int_press_indiv = (allall_gas_SigXX_part + allall_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_int_int_press_indiv = (allall_int_SigXX_part + allall_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_dense_int_press_indiv = (allall_dense_SigXX_part + allall_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_system_int_press_indiv = (allall_system_SigXX_part + allall_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_int_press_indiv = np.append(allall_bulk_int_press, allall_int_int_press)
        allall_int_press_indiv = np.append(allall_int_press, allall_gas_int_press)

        # Calculate total interparticle pressure experienced by all particles in each phase from all A particles
        allA_bulk_int_press = np.sum(allA_bulk_SigXX_part + allA_bulk_SigYY_part)/(4*bulk_area)
        allA_gas_int_press = np.sum(allA_gas_SigXX_part + allA_gas_SigYY_part)/(4*gas_area)
        allA_int_int_press = np.sum(allA_int_SigXX_part + allA_int_SigYY_part)/(4*int_area)
        allA_dense_int_press = np.sum(allA_dense_SigXX_part + allA_dense_SigYY_part)/(4*dense_area)
        allA_system_int_press = np.sum(allA_system_SigXX_part + allA_system_SigYY_part)/(4*system_area)
        allA_dense_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        allA_system_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        allA_system_part_ids = np.append(allA_system_part_ids, phase_part_dict['gas']['A'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        allA_bulk_int_press_indiv = (allA_bulk_SigXX_part + allA_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_gas_int_press_indiv = (allA_gas_SigXX_part + allA_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_int_int_press_indiv = (allA_int_SigXX_part + allA_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_dense_int_press_indiv = (allA_dense_SigXX_part + allA_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_system_int_press_indiv = (allA_system_SigXX_part + allA_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        
        # Calculate total interparticle pressure experienced by all A particles in each phase
        Aall_bulk_int_press = np.sum(Aall_bulk_SigXX_part + Aall_bulk_SigYY_part)/(4*bulk_area)
        Aall_gas_int_press = np.sum(Aall_gas_SigXX_part + Aall_gas_SigYY_part)/(4*gas_area)
        Aall_int_int_press = np.sum(Aall_int_SigXX_part + Aall_int_SigYY_part)/(4*int_area)
        Aall_dense_int_press = np.sum(Aall_dense_SigXX_part + Aall_dense_SigYY_part)/(4*dense_area)
        Aall_system_int_press = np.sum(Aall_system_SigXX_part + Aall_system_SigYY_part)/(4*system_area)
        Aall_dense_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        Aall_system_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        Aall_system_part_ids = np.append(Aall_system_part_ids, phase_part_dict['gas']['all'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        Aall_bulk_int_press_indiv = (Aall_bulk_SigXX_part + Aall_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_gas_int_press_indiv = (Aall_gas_SigXX_part + Aall_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_int_int_press_indiv = (Aall_int_SigXX_part + Aall_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_dense_int_press_indiv = (Aall_dense_SigXX_part + Aall_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_system_int_press_indiv = (Aall_system_SigXX_part + Aall_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all particles in each phase from all B particles
        allB_bulk_int_press = np.sum(allB_bulk_SigXX_part + allB_bulk_SigYY_part)/(4*bulk_area)
        allB_gas_int_press = np.sum(allB_gas_SigXX_part + allB_gas_SigYY_part)/(4*gas_area)
        allB_int_int_press = np.sum(allB_int_SigXX_part + allB_int_SigYY_part)/(4*int_area)
        allB_dense_int_press = np.sum(allB_dense_SigXX_part + allB_dense_SigYY_part)/(4*dense_area)
        allB_system_int_press = np.sum(allB_system_SigXX_part + allB_system_SigYY_part)/(4*system_area)
        allB_dense_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        allB_system_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        allB_system_part_ids = np.append(allB_system_part_ids, phase_part_dict['gas']['B'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        allB_bulk_int_press_indiv = (allB_bulk_SigXX_part + allB_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_gas_int_press_indiv = (allB_gas_SigXX_part + allB_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_int_int_press_indiv = (allB_int_SigXX_part + allB_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_dense_int_press_indiv = (allB_dense_SigXX_part + allB_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_system_int_press_indiv = (allB_system_SigXX_part + allB_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase
        Ball_bulk_int_press = np.sum(Ball_bulk_SigXX_part + Ball_bulk_SigYY_part)/(4*bulk_area)
        Ball_gas_int_press = np.sum(Ball_gas_SigXX_part + Ball_gas_SigYY_part)/(4*gas_area)
        Ball_int_int_press = np.sum(Ball_int_SigXX_part + Ball_int_SigYY_part)/(4*int_area)
        Ball_dense_int_press = np.sum(Ball_dense_SigXX_part + Ball_dense_SigYY_part)/(4*dense_area)
        Ball_system_int_press = np.sum(Ball_system_SigXX_part + Ball_system_SigYY_part)/(4*system_area)
        Ball_dense_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        Ball_system_part_ids = np.append(phase_part_dict['bulk']['all'], phase_part_dict['int']['all'])
        Ball_system_part_ids = np.append(Ball_system_part_ids, phase_part_dict['gas']['all'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        Ball_bulk_int_press_indiv = (Ball_bulk_SigXX_part + Ball_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_gas_int_press_indiv = (Ball_gas_SigXX_part + Ball_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_int_int_press_indiv = (Ball_int_SigXX_part + Ball_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_dense_int_press_indiv = (Ball_dense_SigXX_part + Ball_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_system_int_press_indiv = (Ball_system_SigXX_part + Ball_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all A particles
        AA_bulk_int_press = np.sum(SigXX_AA_bulk_part + SigYY_AA_bulk_part)/(4*bulk_area)
        AA_gas_int_press = np.sum(SigXX_AA_gas_part + SigYY_AA_gas_part)/(4*gas_area)
        AA_int_int_press = np.sum(SigXX_AA_int_part + SigYY_AA_int_part)/(4*int_area)
        AA_dense_int_press = np.sum(SigXX_AA_dense_part + SigYY_AA_dense_part)/(4*dense_area)
        AA_system_int_press = np.sum(SigXX_AA_system_part + SigYY_AA_system_part)/(4*system_area)
        AA_dense_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        AA_system_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        AA_system_part_ids = np.append(AA_system_part_ids, phase_part_dict['gas']['A'])

                # Calculate total interparticle pressure experienced by each particles in each phase
        AA_bulk_int_press_indiv = (SigXX_AA_bulk_part + SigYY_AA_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_gas_int_press_indiv = (SigXX_AA_gas_part + SigYY_AA_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_int_int_press_indiv = (SigXX_AA_int_part + SigYY_AA_int_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_dense_int_press_indiv = (SigXX_AA_dense_part + SigYY_AA_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_system_int_press_indiv = (SigXX_AA_system_part + SigYY_AA_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all B particles
        AB_bulk_int_press = np.sum(SigXX_AB_bulk_part + SigYY_AB_bulk_part)/(4*bulk_area)
        AB_gas_int_press = np.sum(SigXX_AB_gas_part + SigYY_AB_gas_part)/(4*gas_area)
        AB_int_int_press = np.sum(SigXX_AB_int_part + SigYY_AB_int_part)/(4*int_area)
        AB_dense_int_press = np.sum(SigXX_AB_dense_part + SigYY_AB_dense_part)/(4*dense_area)
        AB_system_int_press = np.sum(SigXX_AB_system_part + SigYY_AB_system_part)/(4*system_area)
        AB_dense_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        AB_system_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        AB_system_part_ids = np.append(AB_system_part_ids, phase_part_dict['gas']['B'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        AB_bulk_int_press_indiv = (SigXX_AB_bulk_part + SigYY_AB_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_gas_int_press_indiv = (SigXX_AB_gas_part + SigYY_AB_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_int_int_press_indiv = (SigXX_AB_int_part + SigYY_AB_int_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_dense_int_press_indiv = (SigXX_AB_dense_part + SigYY_AB_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_system_int_press_indiv = (SigXX_AB_system_part + SigYY_AB_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all A particles
        BA_bulk_int_press = np.sum(SigXX_BA_bulk_part + SigYY_BA_bulk_part)/(4*bulk_area)
        BA_gas_int_press = np.sum(SigXX_BA_gas_part + SigYY_BA_gas_part)/(4*gas_area)
        BA_int_int_press = np.sum(SigXX_BA_int_part + SigYY_BA_int_part)/(4*int_area)
        BA_dense_int_press = np.sum(SigXX_BA_dense_part + SigYY_BA_dense_part)/(4*dense_area)
        BA_system_int_press = np.sum(SigXX_BA_system_part + SigYY_BA_system_part)/(4*system_area)
        BA_dense_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        BA_system_part_ids = np.append(phase_part_dict['bulk']['A'], phase_part_dict['int']['A'])
        BA_system_part_ids = np.append(BA_system_part_ids, phase_part_dict['gas']['A'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        BA_bulk_int_press_indiv = (SigXX_BA_bulk_part + SigYY_BA_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_gas_int_press_indiv = (SigXX_BA_gas_part + SigYY_BA_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_int_int_press_indiv = (SigXX_BA_int_part + SigYY_BA_int_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_dense_int_press_indiv = (SigXX_BA_dense_part + SigYY_BA_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_system_int_press_indiv = (SigXX_BA_system_part + SigYY_BA_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all B particles
        BB_bulk_int_press = np.sum(SigXX_BB_bulk_part + SigYY_BB_bulk_part)/(4*bulk_area)
        BB_gas_int_press = np.sum(SigXX_BB_gas_part + SigYY_BB_gas_part)/(4*gas_area)
        BB_int_int_press = np.sum(SigXX_BB_int_part + SigYY_BB_int_part)/(4*int_area)
        BB_dense_int_press = np.sum(SigXX_BB_dense_part + SigYY_BB_dense_part)/(4*dense_area)
        BB_system_int_press = np.sum(SigXX_BB_system_part + SigYY_BB_system_part)/(4*system_area)
        BB_dense_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        BB_system_part_ids = np.append(phase_part_dict['bulk']['B'], phase_part_dict['int']['B'])
        BB_system_part_ids = np.append(BB_system_part_ids, phase_part_dict['gas']['B'])

        # Calculate total interparticle pressure experienced by each particles in each phase
        BB_bulk_int_press_indiv = (SigXX_BB_bulk_part + SigYY_BB_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_gas_int_press_indiv = (SigXX_BB_gas_part + SigYY_BB_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_int_int_press_indiv = (SigXX_BB_int_part + SigYY_BB_int_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_dense_int_press_indiv = (SigXX_BB_dense_part + SigYY_BB_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_system_int_press_indiv = (SigXX_BB_system_part + SigYY_BB_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total shear stress experienced by all particles in each phase from all particles
        allall_bulk_shear_stress = np.sum(allall_bulk_SigXY_part)/(bulk_area)
        allall_gas_shear_stress = np.sum(allall_gas_SigXY_part)/(gas_area)
        allall_int_shear_stress = np.sum(allall_int_SigXY_part)/(int_area)
        allall_dense_shear_stress = np.sum(allall_dense_SigXY_part)/(dense_area)
        allall_system_shear_stress = np.sum(allall_system_SigXY_part)/(system_area)
        allall_shear_stress = np.append(allall_bulk_shear_stress, allall_int_shear_stress)
        allall_shear_stress = np.append(allall_shear_stress, allall_gas_shear_stress)

        # Calculate total shear stress experienced by all particles in each phase from A particles
        allA_bulk_shear_stress = np.sum(allA_bulk_SigXY_part)/(bulk_area)
        allA_gas_shear_stress = np.sum(allA_gas_SigXY_part)/(gas_area)
        allA_int_shear_stress = np.sum(allA_int_SigXY_part)/(int_area)
        allA_dense_shear_stress = np.sum(allA_dense_SigXY_part)/(dense_area)
        allA_system_shear_stress = np.sum(allA_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all particles
        Aall_bulk_shear_stress = np.sum(Aall_bulk_SigXY_part)/(bulk_area)
        Aall_gas_shear_stress = np.sum(Aall_gas_SigXY_part)/(gas_area)
        Aall_int_shear_stress = np.sum(Aall_int_SigXY_part)/(int_area)
        Aall_dense_shear_stress = np.sum(Aall_dense_SigXY_part)/(dense_area)
        Aall_system_shear_stress = np.sum(Aall_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all particles in each phase from B particles
        allB_bulk_shear_stress = np.sum(allB_bulk_SigXY_part)/(bulk_area)
        allB_gas_shear_stress = np.sum(allB_gas_SigXY_part)/(gas_area)
        allB_int_shear_stress = np.sum(allB_int_SigXY_part)/(int_area)
        allB_dense_shear_stress = np.sum(allB_dense_SigXY_part)/(dense_area)
        allB_system_shear_stress = np.sum(allB_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all particles
        Ball_bulk_shear_stress = np.sum(Ball_bulk_SigXY_part)/(bulk_area)
        Ball_gas_shear_stress = np.sum(Ball_gas_SigXY_part)/(gas_area)
        Ball_int_shear_stress = np.sum(Ball_int_SigXY_part)/(int_area)
        Ball_dense_shear_stress = np.sum(Ball_dense_SigXY_part)/(dense_area)
        Ball_system_shear_stress = np.sum(Ball_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all A particles
        AA_bulk_shear_stress = np.sum(SigXY_AA_bulk_part)/(bulk_area)
        AA_gas_shear_stress = np.sum(SigXY_AA_gas_part)/(gas_area)
        AA_int_shear_stress = np.sum(SigXY_AA_int_part)/(int_area)
        AA_dense_shear_stress = np.sum(SigXY_AA_dense_part)/(dense_area)
        AA_system_shear_stress = np.sum(SigXY_AA_system_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all B particles
        AB_bulk_shear_stress = np.sum(SigXY_AB_bulk_part)/(bulk_area)
        AB_gas_shear_stress = np.sum(SigXY_AB_gas_part)/(gas_area)
        AB_int_shear_stress = np.sum(SigXY_AB_int_part)/(int_area)
        AB_dense_shear_stress = np.sum(SigXY_AB_dense_part)/(dense_area)
        AB_system_shear_stress = np.sum(SigXY_AB_system_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all A particles
        BA_bulk_shear_stress = np.sum(SigXY_BA_bulk_part)/(bulk_area)
        BA_gas_shear_stress = np.sum(SigXY_BA_gas_part)/(gas_area)
        BA_int_shear_stress = np.sum(SigXY_BA_int_part)/(int_area)
        BA_dense_shear_stress = np.sum(SigXY_BA_dense_part)/(dense_area)
        BA_system_shear_stress = np.sum(SigXY_BA_system_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all B particles
        BB_bulk_shear_stress = np.sum(SigXY_BB_bulk_part)/(bulk_area)
        BB_gas_shear_stress = np.sum(SigXY_BB_gas_part)/(gas_area)
        BB_int_shear_stress = np.sum(SigXY_BB_int_part)/(int_area)
        BB_dense_shear_stress = np.sum(SigXY_BB_dense_part)/(dense_area)
        BB_system_shear_stress = np.sum(SigXY_BB_system_part)/(system_area)


        # Make position arrays for plotting total stress on each particle for various activity pairings and phases
        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])
        allall_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
        allall_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])
        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])
        Aall_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Aall_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])
        Ball_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Ball_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])
        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])
        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])
        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)
        allall_pos_x = np.append(allall_dense_pos_x, allall_gas_pos_x)
        allall_pos_y = np.append(allall_dense_pos_y, allall_gas_pos_y)

        pos_dense_x = np.append(pos_bulk[:,0], pos_int[:,0])
        pos_dense_y = np.append(pos_bulk[:,1], pos_int[:,1])
        pos_A_dense_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        pos_A_dense_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])
        pos_B_dense_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        pos_B_dense_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        stress_stat_dict = {'bulk': {'all-all': {'XX': np.sum(allall_bulk_SigXX_part), 'XY': np.sum(allall_bulk_SigXY_part), 'YX': np.sum(allall_bulk_SigYX_part), 'YY': np.sum(allall_bulk_SigYY_part)}, 'all-A': {'XX': np.sum(allA_bulk_SigXX_part), 'XY': np.sum(allA_bulk_SigXY_part), 'YX': np.sum(allA_bulk_SigYX_part), 'YY': np.sum(allA_bulk_SigYY_part)}, 'all-B': {'XX': np.sum(allB_bulk_SigXX_part), 'XY': np.sum(allB_bulk_SigXY_part), 'YX': np.sum(allB_bulk_SigYX_part), 'YY': np.sum(allB_bulk_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_bulk_part), 'XY': np.sum(SigXY_AA_bulk_part), 'YX': np.sum(SigYX_AA_bulk_part), 'YY': np.sum(SigYY_AA_bulk_part)}, 'A-B': {'XX': np.sum(SigXX_AB_bulk_part), 'XY': np.sum(SigXY_AB_bulk_part), 'YX': np.sum(SigYX_AB_bulk_part), 'YY': np.sum(SigYY_AB_bulk_part)}, 'B-B': {'XX': np.sum(SigXX_BB_bulk_part), 'XY': np.sum(SigXY_BB_bulk_part), 'YX': np.sum(SigYX_BB_bulk_part), 'YY': np.sum(SigYY_BB_bulk_part)}}, 'gas': {'all-all': {'XX': np.sum(allall_gas_SigXX_part), 'XY': np.sum(allall_gas_SigXY_part), 'YX': np.sum(allall_gas_SigYX_part), 'YY': np.sum(allall_gas_SigYY_part)}, 'all-A': {'XX': np.sum(allA_gas_SigXX_part), 'XY': np.sum(allA_gas_SigXY_part), 'YX': np.sum(allA_gas_SigYX_part), 'YY': np.sum(allA_gas_SigYY_part)}, 'all-B': {'XX': np.sum(allB_gas_SigXX_part), 'XY': np.sum(allB_gas_SigXY_part), 'YX': np.sum(allB_gas_SigYX_part), 'YY': np.sum(allB_gas_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_gas_part), 'XY': np.sum(SigXY_AA_gas_part), 'YX': np.sum(SigYX_AA_gas_part), 'YY': np.sum(SigYY_AA_gas_part)}, 'A-B': {'XX': np.sum(SigXX_AB_gas_part), 'XY': np.sum(SigXY_AB_gas_part), 'YX': np.sum(SigYX_AB_gas_part), 'YY': np.sum(SigYY_AB_gas_part)}, 'B-B': {'XX': np.sum(SigXX_BB_gas_part), 'XY': np.sum(SigXY_BB_gas_part), 'YX': np.sum(SigYX_BB_gas_part), 'YY': np.sum(SigYY_BB_gas_part)}}, 'dense': {'all-all': {'XX': np.sum(allall_dense_SigXX_part), 'XY': np.sum(allall_dense_SigXY_part), 'YX': np.sum(allall_dense_SigYX_part), 'YY': np.sum(allall_dense_SigYY_part)}, 'all-A': {'XX': np.sum(allA_dense_SigXX_part), 'XY': np.sum(allA_dense_SigXY_part), 'YX': np.sum(allA_dense_SigYX_part), 'YY': np.sum(allA_dense_SigYY_part)}, 'all-B': {'XX': np.sum(allB_dense_SigXX_part), 'XY': np.sum(allB_dense_SigXY_part), 'YX': np.sum(allB_dense_SigYX_part), 'YY': np.sum(allB_dense_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_dense_part), 'XY': np.sum(SigXY_AA_dense_part), 'YX': np.sum(SigYX_AA_dense_part), 'YY': np.sum(SigYY_AA_dense_part)}, 'A-B': {'XX': np.sum(SigXX_AB_dense_part), 'XY': np.sum(SigXY_AB_dense_part), 'YX': np.sum(SigYX_AB_dense_part), 'YY': np.sum(SigYY_AB_dense_part)}, 'B-B': {'XX': np.sum(SigXX_BB_dense_part), 'XY': np.sum(SigXY_BB_dense_part), 'YX': np.sum(SigYX_BB_dense_part), 'YY': np.sum(SigYY_BB_dense_part)}}, 'int': {'all-all': {'XX': np.sum(allall_int_SigXX_part), 'XY': np.sum(allall_int_SigXY_part), 'YX': np.sum(allall_int_SigYX_part), 'YY': np.sum(allall_int_SigYY_part)}, 'all-A': {'XX': np.sum(allA_int_SigXX_part), 'XY': np.sum(allA_int_SigXY_part), 'YX': np.sum(allA_int_SigYX_part), 'YY': np.sum(allA_int_SigYY_part)}, 'all-B': {'XX': np.sum(allB_int_SigXX_part), 'XY': np.sum(allB_int_SigXY_part), 'YX': np.sum(allB_int_SigYX_part), 'YY': np.sum(allB_int_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_int_part), 'XY': np.sum(SigXY_AA_int_part), 'YX': np.sum(SigYX_AA_int_part), 'YY': np.sum(SigYY_AA_int_part)}, 'A-B': {'XX': np.sum(SigXX_AB_int_part), 'XY': np.sum(SigXY_AB_int_part), 'YX': np.sum(SigYX_AB_int_part), 'YY': np.sum(SigYY_AB_int_part)}, 'B-B': {'XX': np.sum(SigXX_BB_int_part), 'XY': np.sum(SigXY_BB_int_part), 'YX': np.sum(SigYX_BB_int_part), 'YY': np.sum(SigYY_BB_int_part)}}}

        # Create output dictionary for statistical averages of total pressure and shear stress on each particle per phase/activity pairing
        press_stat_dict = {'all-all': {'bulk': {'press': allall_bulk_int_press, 'shear': allall_bulk_shear_stress}, 'int': {'press': allall_int_int_press, 'shear': allall_int_shear_stress}, 'gas': {'press': allall_gas_int_press, 'shear': allall_gas_shear_stress}, 'dense': {'press': allall_dense_int_press, 'shear': allall_dense_shear_stress}, 'system': {'press': allall_system_int_press, 'shear': allall_system_shear_stress}}, 'all-A': {'bulk': {'press': allA_bulk_int_press, 'shear': allA_bulk_shear_stress}, 'int': {'press': allA_int_int_press, 'shear': allA_int_shear_stress}, 'gas': {'press': allA_gas_int_press, 'shear': allA_gas_shear_stress}, 'dense': {'press': allA_dense_int_press, 'shear': allA_dense_shear_stress}, 'system': {'press': allA_system_int_press, 'shear': allA_system_shear_stress}}, 'all-B': {'bulk': {'press': allB_bulk_int_press, 'shear': allB_bulk_shear_stress}, 'int': {'press': allB_int_int_press, 'shear': allB_int_shear_stress}, 'gas': {'press': allB_gas_int_press, 'shear': allB_gas_shear_stress}, 'dense': {'press': allB_dense_int_press, 'shear': allB_dense_shear_stress}, 'system': {'press': allB_system_int_press, 'shear': allB_system_shear_stress}}, 'A-A': {'bulk': {'press': AA_bulk_int_press, 'shear': AA_bulk_shear_stress}, 'int': {'press': AA_int_int_press, 'shear': AA_int_shear_stress}, 'gas': {'press': AA_gas_int_press, 'shear': AA_gas_shear_stress}, 'dense': {'press': AA_dense_int_press, 'shear': AA_dense_shear_stress}, 'system': {'press': AA_system_int_press, 'shear': AA_system_shear_stress}}, 'A-B': {'bulk': {'press': AB_bulk_int_press, 'shear': AB_bulk_shear_stress}, 'int': {'press': AB_int_int_press, 'shear': AB_int_shear_stress}, 'gas': {'press': AB_gas_int_press, 'shear': AB_gas_shear_stress}, 'dense': {'press': AB_dense_int_press, 'shear': AB_dense_shear_stress}, 'system': {'press': AB_system_int_press, 'shear': AB_system_shear_stress}}, 'B-B': {'bulk': {'press': BB_bulk_int_press, 'shear': BB_bulk_shear_stress}, 'int': {'press': BB_int_int_press, 'shear': BB_int_shear_stress}, 'gas': {'press': BB_gas_int_press, 'shear': BB_gas_shear_stress}, 'dense': {'press': BB_dense_int_press, 'shear': BB_dense_shear_stress}, 'system': {'press': BB_system_int_press, 'shear': BB_system_shear_stress}}}

        # Create output dictionary for statistical averages of total pressure and shear stress on each particle per phase/activity pairing
        press_stat_indiv_dict = {'all-all': {'bulk': {'mean': np.mean(allall_bulk_int_press_indiv), 'std': np.std(allall_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allall_int_int_press_indiv), 'std': np.std(allall_int_int_press_indiv)}, 'gas': {'mean': np.mean(allall_gas_int_press_indiv), 'std': np.std(allall_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allall_dense_int_press_indiv), 'std': np.std(allall_dense_int_press_indiv)}, 'system': {'mean': np.mean(allall_system_int_press_indiv), 'std': np.std(allall_system_int_press_indiv)}}, 'all-A': {'bulk': {'mean': np.mean(allA_bulk_int_press_indiv), 'std': np.std(allA_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allA_int_int_press_indiv), 'std': np.std(allA_int_int_press_indiv)}, 'gas': {'mean': np.mean(allA_gas_int_press_indiv), 'std': np.std(allA_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allA_dense_int_press_indiv), 'std': np.std(allA_dense_int_press_indiv)}, 'system': {'mean': np.mean(allA_system_int_press_indiv), 'std': np.std(allA_system_int_press_indiv)}}, 'all-B': {'bulk': {'mean': np.mean(allB_bulk_int_press_indiv), 'std': np.std(allB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allB_int_int_press_indiv), 'std': np.std(allB_int_int_press_indiv)}, 'gas': {'mean': np.mean(allB_gas_int_press_indiv), 'std': np.std(allB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allB_dense_int_press_indiv), 'std': np.std(allB_dense_int_press_indiv)}, 'system': {'mean': np.mean(allB_system_int_press_indiv), 'std': np.std(allB_system_int_press_indiv)}}, 'A-A': {'bulk': {'mean': np.mean(AA_bulk_int_press_indiv), 'std': np.std(AA_bulk_int_press_indiv)}, 'int': {'mean': np.mean(AA_int_int_press_indiv), 'std': np.std(AA_int_int_press_indiv)}, 'gas': {'mean': np.mean(AA_gas_int_press_indiv), 'std': np.std(AA_gas_int_press_indiv)}, 'dense': {'mean': np.mean(AA_dense_int_press_indiv), 'std': np.std(AA_dense_int_press_indiv)}, 'system': {'mean': np.mean(AA_system_int_press_indiv), 'std': np.std(AA_system_int_press_indiv)}}, 'A-B': {'bulk': {'mean': np.mean(AB_bulk_int_press_indiv), 'std': np.std(AB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(AB_int_int_press_indiv), 'std': np.std(AB_int_int_press_indiv)}, 'gas': {'mean': np.mean(AB_gas_int_press_indiv), 'std': np.std(AB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(AB_dense_int_press_indiv), 'std': np.std(AB_dense_int_press_indiv)}, 'system': {'mean': np.mean(AB_system_int_press_indiv), 'std': np.std(AB_system_int_press_indiv)}}, 'B-B': {'bulk': {'mean': np.mean(BB_bulk_int_press_indiv), 'std': np.std(BB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(BB_int_int_press_indiv), 'std': np.std(BB_int_int_press_indiv)}, 'gas': {'mean': np.mean(BB_gas_int_press_indiv), 'std': np.std(BB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(BB_dense_int_press_indiv), 'std': np.std(BB_dense_int_press_indiv)}, 'system': {'mean': np.mean(BB_system_int_press_indiv), 'std': np.std(BB_system_int_press_indiv)}}}

        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        stress_plot_dict = {'bulk': {'all-all': {'XX': allall_bulk_SigXX_part, 'XY': allall_bulk_SigXY_part, 'YX': allall_bulk_SigYX_part, 'YY': allall_bulk_SigYY_part}, 'all-A': {'XX': allA_bulk_SigXX_part, 'XY': allA_bulk_SigXY_part, 'YX': allA_bulk_SigYX_part, 'YY': allA_bulk_SigYY_part}, 'all-B': {'XX': allB_bulk_SigXX_part, 'XY': allB_bulk_SigXY_part, 'YX': allB_bulk_SigYX_part, 'YY': allB_bulk_SigYY_part}, 'A-A': {'XX': SigXX_AA_bulk_part, 'XY': SigXY_AA_bulk_part, 'YX': SigYX_AA_bulk_part, 'YY': SigYY_AA_bulk_part}, 'A-B': {'XX': SigXX_AB_bulk_part, 'XY': SigXY_AB_bulk_part, 'YX': SigYX_AB_bulk_part, 'YY': SigYY_AB_bulk_part}, 'B-B': {'XX': SigXX_BB_bulk_part, 'XY': SigXY_BB_bulk_part, 'YX': SigYX_BB_bulk_part, 'YY': SigYY_BB_bulk_part}, 'pos': {'all': {'x': pos_bulk[:,0], 'y': pos_bulk[:,1]}, 'A': {'x': pos_A_bulk[:,0], 'y': pos_A_bulk[:,1]}, 'B': {'x': pos_B_bulk[:,0], 'y': pos_B_bulk[:,1]}}, 'typ': typ_bulk}, 'gas': {'all-all': {'XX': allall_gas_SigXX_part, 'XY': allall_gas_SigXY_part, 'YX': allall_gas_SigYX_part, 'YY': allall_gas_SigYY_part}, 'all-A': {'XX': allA_gas_SigXX_part, 'XY': allA_gas_SigXY_part, 'YX': allA_gas_SigYX_part, 'YY': allA_gas_SigYY_part}, 'all-B': {'XX': allB_gas_SigXX_part, 'XY': allB_gas_SigXY_part, 'YX': allB_gas_SigYX_part, 'YY': allB_gas_SigYY_part}, 'A-A': {'XX': SigXX_AA_gas_part, 'XY': SigXY_AA_gas_part, 'YX': SigYX_AA_gas_part, 'YY': SigYY_AA_gas_part}, 'A-B': {'XX': SigXX_AB_gas_part, 'XY': SigXY_AB_gas_part, 'YX': SigYX_AB_gas_part, 'YY': SigYY_AB_gas_part}, 'B-B': {'XX': SigXX_BB_gas_part, 'XY': SigXY_BB_gas_part, 'YX': SigYX_BB_gas_part, 'YY': SigYY_BB_gas_part}, 'pos': {'all': {'x': pos_gas[:,0], 'y': pos_gas[:,1]}, 'A': {'x': pos_A_gas[:,0], 'y': pos_A_gas[:,1]}, 'B': {'x': pos_B_gas[:,0], 'y': pos_B_gas[:,1]}}, 'typ': typ_gas}, 'dense': {'all-all': {'XX': allall_dense_SigXX_part, 'XY': allall_dense_SigXY_part, 'YX': allall_dense_SigYX_part, 'YY': allall_dense_SigYY_part}, 'all-A': {'XX': allA_dense_SigXX_part, 'XY': allA_dense_SigXY_part, 'YX': allA_dense_SigYX_part, 'YY': allA_dense_SigYY_part}, 'all-B': {'XX': allB_dense_SigXX_part, 'XY': allB_dense_SigXY_part, 'YX': allB_dense_SigYX_part, 'YY': allB_dense_SigYY_part}, 'A-A': {'XX': SigXX_AA_dense_part, 'XY': SigXY_AA_dense_part, 'YX': SigYX_AA_dense_part, 'YY': SigYY_AA_dense_part}, 'A-B': {'XX': SigXX_AB_dense_part, 'XY': SigXY_AB_dense_part, 'YX': SigYX_AB_dense_part, 'YY': SigYY_AB_dense_part}, 'B-B': {'XX': SigXX_BB_dense_part, 'XY': SigXY_BB_dense_part, 'YX': SigYX_BB_dense_part, 'YY': SigYY_BB_dense_part}, 'pos': {'all': {'x': pos_dense_x, 'y': pos_dense_y}, 'A': {'x': pos_A_dense_x, 'y': pos_A_dense_y}, 'B': {'x': pos_B_dense_x, 'y': pos_B_dense_y}}, 'typ': typ_dense}}

        # Create output dictionary for plotting of total stress/pressure on each particle per phase/activity pairing and their respective x-y locations
        press_plot_dict = {'all-all': {'press': allall_int_press, 'shear': allall_shear_stress, 'x': allall_pos_x, 'y': allall_pos_y}}
        
        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        press_plot_indiv_dict = {'all-all': {'press': allall_system_int_press_indiv}, 'all-A': {'press': allA_system_int_press_indiv}, 'all-B': {'press': allB_system_int_press_indiv}, 'A-A': {'press': AA_system_int_press_indiv}, 'A-B': {'press': AB_system_int_press_indiv}, 'B-B': {'press': BB_system_int_press_indiv}, 'id': id_system, 'typ': typ_system}

        # Create output dictionary for plotting of total stress/pressure on each particle per phase/activity pairing and their respective x-y locations
        press_hetero_dict = {'system': {'all-all': {'press': allall_system_int_press_indiv, 'id': allall_system_part_ids}, 'all-A': {'press': allA_system_int_press_indiv, 'id': allA_system_part_ids}, 'A-all': {'press': Aall_system_int_press_indiv, 'id': Aall_system_part_ids}, 'all-B': {'press': allB_system_int_press_indiv, 'id': allB_system_part_ids}, 'B-all': {'press': Ball_system_int_press_indiv, 'id': Ball_system_part_ids}, 'A-A': {'press': AA_system_int_press_indiv, 'id': AA_system_part_ids}, 'A-B': {'press': AB_system_int_press_indiv, 'id': AB_system_part_ids}, 'B-A': {'press': BA_system_int_press_indiv, 'id': BA_system_part_ids}, 'B-B': {'press': BB_system_int_press_indiv, 'id': BB_system_part_ids} }, 'dense': {'all-all': {'press': allall_dense_int_press_indiv, 'id': allall_dense_part_ids}, 'all-A': {'press': allA_dense_int_press_indiv, 'id': allA_dense_part_ids}, 'A-all': {'press': Aall_dense_int_press_indiv, 'id': Aall_dense_part_ids}, 'all-B': {'press': allB_dense_int_press_indiv, 'id': allB_dense_part_ids}, 'B-all': {'press': Ball_dense_int_press_indiv, 'id': Ball_dense_part_ids}, 'A-A': {'press': AA_dense_int_press_indiv, 'id': AA_dense_part_ids}, 'A-B': {'press': AB_dense_int_press_indiv, 'id': AB_dense_part_ids}, 'B-A': {'press': BA_dense_int_press_indiv, 'id': BA_dense_part_ids}, 'B-B': {'press': BB_dense_int_press_indiv, 'id': BB_dense_part_ids} }, 'bulk': {'all-all': {'press': allall_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['all']}, 'all-A': {'press': allA_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['A']}, 'A-all': {'press': Aall_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['all']}, 'all-B': {'press': allB_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['B']}, 'B-all': {'press': Ball_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['all']}, 'A-A': {'press': AA_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['A']}, 'A-B': {'press': AB_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['B']}, 'B-A': {'press': BA_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['A']}, 'B-B': {'press': BB_bulk_int_press_indiv, 'id': phase_part_dict['bulk']['B']} }, 'int': {'all-all': {'press': allall_int_int_press_indiv, 'id': phase_part_dict['int']['all']}, 'all-A': {'press': allA_int_int_press_indiv, 'id': phase_part_dict['int']['A']}, 'A-all': {'press': Aall_int_int_press_indiv, 'id': phase_part_dict['int']['all']}, 'all-B': {'press': allB_int_int_press_indiv, 'id': phase_part_dict['int']['B']}, 'B-all': {'press': Ball_int_int_press_indiv, 'id': phase_part_dict['int']['all']}, 'A-A': {'press': AA_int_int_press_indiv, 'id': phase_part_dict['int']['A']}, 'A-B': {'press': AB_int_int_press_indiv, 'id': phase_part_dict['int']['B']}, 'B-A': {'press': BA_int_int_press_indiv, 'id': phase_part_dict['int']['A']}, 'B-B': {'press': BB_int_int_press_indiv, 'id': phase_part_dict['int']['B']} }, 'gas': {'all-all': {'press': allall_gas_int_press_indiv, 'id': phase_part_dict['gas']['all']}, 'all-A': {'press': allA_gas_int_press_indiv, 'id': phase_part_dict['gas']['A']}, 'A-all': {'press': Aall_gas_int_press_indiv, 'id': phase_part_dict['gas']['all']}, 'all-B': {'press': allB_gas_int_press_indiv, 'id': phase_part_dict['gas']['B']}, 'B-all': {'press': Ball_gas_int_press_indiv, 'id': phase_part_dict['gas']['all']}, 'A-A': {'press': AA_gas_int_press_indiv, 'id': phase_part_dict['gas']['A']}, 'A-B': {'press': AB_gas_int_press_indiv, 'id': phase_part_dict['gas']['B']}, 'B-A': {'press': BA_gas_int_press_indiv, 'id': phase_part_dict['gas']['A']}, 'B-B': {'press': BB_gas_int_press_indiv, 'id': phase_part_dict['gas']['B']} }  }
        
        return stress_stat_dict, press_stat_dict, press_stat_indiv_dict, press_plot_dict, stress_plot_dict, press_plot_indiv_dict, press_hetero_dict

    def radial_int_press(self, stress_plot_dict):
        '''
        Purpose: Takes the stress of each particle and bins particles in terms of separation distance
        and angle around cluster CoM to calculate total stress in each bin

        Input:
        stress_plot_dict: dictionary containing stresses in each direction acting on each particle of each type in each phase
        
        Output:
        radial_stress_dict: dictionary containing each particle's stress in each direction acting on each particle as a 
        function of separation distance and angle around cluster's CoM
        '''

        # Stress in each direction of all particles
        stress_xx = np.append(stress_plot_dict['dense']['all-all']['XX'], stress_plot_dict['gas']['all-all']['XX'])
        stress_yy = np.append(stress_plot_dict['dense']['all-all']['YY'], stress_plot_dict['gas']['all-all']['YY'])
        stress_xy = np.append(stress_plot_dict['dense']['all-all']['XY'], stress_plot_dict['gas']['all-all']['XY'])
        stress_yx = np.append(stress_plot_dict['dense']['all-all']['YX'], stress_plot_dict['gas']['all-all']['YX'])

        # X- and Y- Positions of each particle in same order as above
        pos_x = np.append(stress_plot_dict['dense']['pos']['all']['x'], stress_plot_dict['gas']['pos']['all']['x'])
        pos_y = np.append(stress_plot_dict['dense']['pos']['all']['y'], stress_plot_dict['gas']['pos']['all']['y'])
        
        # Particle type in same order as above
        typ = np.append(stress_plot_dict['dense']['typ'], stress_plot_dict['gas']['typ'])

        # Instantiate empty array (partNum) containing the average active force magnitude
        # towards the largest cluster's CoM
        stress_xx_arr = np.array([])
        stress_yy_arr = np.array([])
        stress_xy_arr = np.array([])
        stress_yx_arr = np.array([])

        stress_xx_A_arr = np.array([])
        stress_yy_A_arr = np.array([])
        stress_xy_A_arr = np.array([])
        stress_yx_A_arr = np.array([])

        stress_xx_B_arr = np.array([])
        stress_yy_B_arr = np.array([])
        stress_xy_B_arr = np.array([])
        stress_yx_B_arr = np.array([])

        # Instantiate empty array (partNum) containing the distance from largest cluster's CoM
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])

        theta_dist_norm = np.array([])
        thetaA_dist_norm = np.array([])
        thetaB_dist_norm = np.array([])

        # Loop over all particles
        for h in range(0, len(pos_x)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = pos_x[h] - 0
            dify = pos_y[h] - 0

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Angle around cluster's CoM
            thetar = np.arctan2(dify, difx)*(180/np.pi)            
            
            # Distance from cluster CoM (normalized by cluster radius) to bin particles by
            r = np.linspace(0, 1.5, num=75)

            # Angle around cluster CoM to bin particles by
            theta = np.linspace(0, 360, num=45)


            # Save stress of each particle in each direction to array in addition to separation and angle around CoM by type
            if typ[h] == 0:
                stress_xx_A_arr =np.append(stress_xx_A_arr, stress_xx[h])
                stress_yy_A_arr =np.append(stress_yy_A_arr, stress_yy[h])
                stress_xy_A_arr =np.append(stress_xy_A_arr, stress_xy[h])
                stress_yx_A_arr =np.append(stress_yx_A_arr, stress_yx[h])
                rA_dist_norm = np.append(rA_dist_norm, difr)
                thetaA_dist_norm = np.append(thetaA_dist_norm, thetar)
            else:
                stress_xx_B_arr =np.append(stress_xx_B_arr, stress_xx[h])
                stress_yy_B_arr =np.append(stress_yy_B_arr, stress_yy[h])
                stress_xy_B_arr =np.append(stress_xy_B_arr, stress_xy[h])
                stress_yx_B_arr =np.append(stress_yx_B_arr, stress_yx[h])
                rB_dist_norm = np.append(rB_dist_norm, difr)
                thetaB_dist_norm = np.append(thetaB_dist_norm, thetar)
            
            # Save stress of each particle in each direction to array in addition to separation and angle around CoM for all types
            stress_xx_arr =np.append(stress_xx_arr, stress_xx[h])
            stress_yy_arr =np.append(stress_yy_arr, stress_yy[h])
            stress_xy_arr =np.append(stress_xy_arr, stress_xy[h])
            stress_yx_arr =np.append(stress_yx_arr, stress_yx[h])
            r_dist_norm = np.append(r_dist_norm, difr)
            theta_dist_norm = np.append(theta_dist_norm, thetar)

        # Dictionary containing each particle's stress in each direction and separation and angle around cluster CoM
        radial_stress_dict = {'all': {'XX': stress_xx_arr, 'YY': stress_yy_arr, 'XY': stress_xy_arr, 'YX': stress_yx_arr, 'r': r_dist_norm, 'theta': theta_dist_norm}, 'A': {'XX': stress_xx_A_arr, 'YY': stress_yy_A_arr, 'XY': stress_xy_A_arr, 'YX': stress_yx_A_arr, 'r': rA_dist_norm, 'theta': thetaA_dist_norm}, 'B': {'XX': stress_xx_B_arr, 'YY': stress_yy_B_arr, 'XY': stress_xy_B_arr, 'YX': stress_yx_B_arr, 'r': rB_dist_norm, 'theta': thetaB_dist_norm}, 'typ': typ}

        return radial_stress_dict

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
        alignA_norm = np.array([])
        alignB_norm = np.array([])

        # Instantiate empty array (partNum) containing the average active force magnitude
        # towards the largest cluster's CoM
        fa_norm = np.array([])
        faA_norm = np.array([])
        faB_norm = np.array([])

        # Instantiate empty array (partNum) containing the distance from largest cluster's CoM
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])

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
            px = self.px[h]
            py = self.py[h]

            #Calculate alignment towards CoM
            r_dot_p = (-x_norm_unitv * px) + (-y_norm_unitv * py)

            # Save alignment with largest cluster's CoM
            align_norm=np.append(align_norm, r_dot_p)

            # Save active force magnitude toward largest cluster's CoM
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, r_dot_p*self.peA)
                faA_norm=np.append(faA_norm, r_dot_p*self.peA)
                alignA_norm=np.append(alignA_norm, r_dot_p)
                rA_dist_norm = np.append(rA_dist_norm, difr)
            else:
                fa_norm=np.append(fa_norm, r_dot_p*self.peB)
                faB_norm=np.append(faB_norm, r_dot_p*self.peB)
                alignB_norm=np.append(alignB_norm, r_dot_p)
                rB_dist_norm = np.append(rB_dist_norm, difr)

            # Save separation distance from largest cluster's CoM
            r_dist_norm = np.append(r_dist_norm, difr)

        # Dictionary containing each particle's alignment and aligned active force toward
        # largest cluster's CoM as a function of separation distance from largest custer's CoM
        radial_fa_dict = {'all': {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}, 'A': {'r': rA_dist_norm, 'fa': faA_norm, 'align': alignA_norm}, 'B': {'r': rB_dist_norm, 'fa': faB_norm, 'align': alignB_norm}}

        return radial_fa_dict

    def radial_int_press_bubble2(self, stress_plot_dict, sep_surface_dict, int_comp_dict, all_surface_measurements):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from largest cluster's CoM

        stress_plot_dict: dictionary (output from various interparticle_pressure_nlist() in
        measurement.py) containing information on the stress and positions of all,
        type A, and type B particles.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        all_surface_measurements: dictionary that contains information on measured properties
        of each identified interface surface

        Output:
        radial_stress_dict: dictionary containing each particle's stress in each direction
        as a function of separation distance from each interface's CoM
        '''

        # Stress in all directions of all particles in order of dense first then gas particles
        stress_xx = np.append(stress_plot_dict['dense']['all-all']['XX'], stress_plot_dict['gas']['all-all']['XX'])
        stress_yy = np.append(stress_plot_dict['dense']['all-all']['YY'], stress_plot_dict['gas']['all-all']['YY'])
        stress_xy = np.append(stress_plot_dict['dense']['all-all']['XY'], stress_plot_dict['gas']['all-all']['XY'])
        stress_yx = np.append(stress_plot_dict['dense']['all-all']['YX'], stress_plot_dict['gas']['all-all']['YX'])

        # All particle positions in same order as above
        pos_x = np.append(stress_plot_dict['dense']['pos']['all']['x'], stress_plot_dict['gas']['pos']['all']['x'])
        pos_y = np.append(stress_plot_dict['dense']['pos']['all']['y'], stress_plot_dict['gas']['pos']['all']['y'])
        
        # All particle types in same order as above
        typ = np.append(stress_plot_dict['dense']['typ'], stress_plot_dict['gas']['typ'])

        radial_stress_dict = {}

        # Loop over all interface surfaces
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

            # Current interface's CoM
            try:
                
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']

            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']
        
            # Instantiate empty array (partNum) containing the stress in each direction for particles of each type ('all', 'A', or 'B)
            stress_xx_arr = np.array([])
            stress_yy_arr = np.array([])
            stress_xy_arr = np.array([])
            stress_yx_arr = np.array([])

            stress_xx_A_arr = np.array([])
            stress_yy_A_arr = np.array([])
            stress_xy_A_arr = np.array([])
            stress_yx_A_arr = np.array([])

            stress_xx_B_arr = np.array([])
            stress_yy_B_arr = np.array([])
            stress_xy_B_arr = np.array([])
            stress_yx_B_arr = np.array([])

            # Instantiate empty array (partNum) containing the distance from current interface's CoM
            r_dist_norm = np.array([])
            rA_dist_norm = np.array([])
            rB_dist_norm = np.array([])
        
            # Loop over all particles
            for h in range(0, len(pos_x)):

                # Separation distance from current interface's CoM (shifted from middle of box)
                difx = pos_x[h] - (com_x-self.hx_box)
                dify = pos_y[h] - (com_y-self.hy_box)

                difr= ( (difx )**2 + (dify)**2)**0.5

                if typ[h] == 0:

                    # Save stress of each A particle in each direction
                    stress_xx_A_arr =np.append(stress_xx_A_arr, stress_xx[h])
                    stress_yy_A_arr =np.append(stress_yy_A_arr, stress_yy[h])
                    stress_xy_A_arr =np.append(stress_xy_A_arr, stress_xy[h])
                    stress_yx_A_arr =np.append(stress_yx_A_arr, stress_yx[h])

                    # Save separation distance of each A particle from current interface's CoM
                    rA_dist_norm = np.append(rA_dist_norm, difr)
                else:

                    # Save stress of each B particle in each direction
                    stress_xx_B_arr =np.append(stress_xx_B_arr, stress_xx[h])
                    stress_yy_B_arr =np.append(stress_yy_B_arr, stress_yy[h])
                    stress_xy_B_arr =np.append(stress_xy_B_arr, stress_xy[h])
                    stress_yx_B_arr =np.append(stress_yx_B_arr, stress_yx[h])
                    
                    # Save separation distance of each B particle from current interface's CoM
                    rB_dist_norm = np.append(rB_dist_norm, difr)
                
                # Save stress of each particle in each direction
                stress_xx_arr =np.append(stress_xx_arr, stress_xx[h])
                stress_yy_arr =np.append(stress_yy_arr, stress_yy[h])
                stress_xy_arr =np.append(stress_xy_arr, stress_xy[h])
                stress_yx_arr =np.append(stress_yx_arr, stress_yx[h])

                
                # Save separation distance of each particle from current interface's CoM
                r_dist_norm = np.append(r_dist_norm, difr)

            # Dictionary containing each particle's stress in each direction for all particle types ('all', 'A', or 'B')
            # as a function of separation distance from each interface's CoM

            radial_stress_dict[key] = {'all': {'XX': stress_xx_arr, 'YY': stress_yy_arr, 'XY': stress_xy_arr, 'YX': stress_yx_arr, 'r': r_dist_norm}, 'A': {'XX': stress_xx_A_arr, 'YY': stress_yy_A_arr, 'XY': stress_xy_A_arr, 'YX': stress_yx_A_arr, 'r': rA_dist_norm}, 'B': {'XX': stress_xx_B_arr, 'YY': stress_yy_B_arr, 'XY': stress_xy_B_arr, 'YX': stress_yx_B_arr, 'r': rB_dist_norm}, 'typ': typ}

        return radial_stress_dict
    """
    def radial_surface_active_force_pressure(self, int_comp_dict, all_surface_measurements, all_surface_curves, radial_fa_dict):

        int_large_ids = int_comp_dict['ids']['int id']

        print(radial_fa_dict)
        stop

        for m in range(0, len(int_large_ids)):
            if int_large_ids[m]!=999:
                key = 'surface id ' + str(int(int_comp_dict['ids']['int id'][m]))
                interior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['interior']['pos'])
                exterior_surface_com_pos_dict = self.surface_com_pov(surface_curve[key]['exterior']['pos'])

                #Calculate location of interior surface from CoM
                interior_rad = all_surface_measurements[key]['interior']['radius']

                #X locations across interface for integration
                xint = np.linspace(0, np.ceil(np.max(radial_fa_dict['r'])), num=int((int((int(np.ceil(np.ceil(np.max(radial_fa_dict['r']))))+1)))/3))

                #Pressure integrand components for each value of X
                yfinal = np.zeros(len(xint))
                alignfinal = np.zeros(len(xint))
                densfinal = np.zeros(len(xint))

                #Flatten exterior and interior surfaces and line 2 ends up. Theta defines angle from y axis to overshot of exterior surface of interior surface
                tan_theta = (surface_area_ext[0]-surface_area_int[0])/edge_width_arr[0]

                #Can simplify geometric area into two components: rectangle which is defined as the product of interior surface length and the current step size across the interface (x)
                #and triangle which accounts for the overlap of the exterior surface with the interior surface, which is defined as the product of the overlap length (exterior length - interior length) and half of the current step size across the interface (x)
                area_rect_prev = 0
                area_tri_prev = 0

                #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
                for i in range(1, len(xint)):
                    binx_nodup=np.array([])
                    biny_nodup=np.array([])

                    #Min and max location across interface of current step
                    min_range = xint[i-1]
                    max_range = xint[i]

                    #Calculate area of rectangle for current step
                    area_rect = (all_surface_measurements[key]['interior']['surface area'] * xint[i]) - area_rect_prev

                    #Save total area of previous step sizes
                    area_rect_prev = (all_surface_measurements[key]['interior']['radius'] * xint[i])

                    #Calculate the overshot length of current step
                    dif_sa = tan_theta * xint[i]

                    #Calculate area of triangle from overshot of exterior surface with interior surface
                    area_tri = (dif_sa * xint[i])/2

                    #Subtract total area of triangle from previous steps
                    area_poly = area_tri - area_tri_prev

                    #Save total area of previous step sizes
                    area_tri_prev = (dif_sa * xint[i])/2

                    #Total area of slice we're calculating pressure over
                    area_slice = area_poly + area_rect

                    #Find particles that are housed within current slice
                    points = np.where((min_range<=r_dist_tot_bub1) & (r_dist_tot_bub1<=max_range))[0]

                    #If at least 1 particle in slice, continue...
                    if len(points)>0:

                        #If the force is defined, continue...
                        points2 = np.logical_not(np.isnan(fa_all_tot_bub1[points]))
                        if len(points2)>0:

                            #Calculate total active force normal to interface in slice
                            yfinal[i-1] = np.sum(fa_all_tot_bub1[points][points2])
                            alignfinal[i-1] = np.mean(align_all_tot_bub1[points][points2])
                            densfinal[i-1] = len(fa_all_tot_bub1[points][points2])
                            #If either exterior or interior surface area failed to calculate, use the radii
                            if len(surface_area_ext)==0:
                                area_slice = np.pi * ((all_surface_measurements[key]['interior']['radius'] +xint[i])**2-(all_surface_measurements[key]['interior']['radius'] + xint[i-1])**2)
                            if len(surface_area_int)==0:
                                area_slice = np.pi * ((all_surface_measurements[key]['interior']['radius'] +xint[i])**2-(all_surface_measurements[key]['interior']['radius'] + xint[i-1])**2)

                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area_slice != 0:
                                yfinal[i-1] = yfinal[i-1]/area_slice
                                densfinal[i-1] = densfinal[i-1]/area_slice
                            else:
                                yfinal[i-1] = 0
                                alignfinal[i-1] = 0
                                densfinal[i-1] = 0
                        else:
                            yfinal[i-1]=0
                            alignfinal[i-1] = 0
                            densfinal[i-1] = 0
                    else:
                        yfinal[i-1]=0
                        alignfinal[i-1] = 0
                        densfinal[i-1] = 0

                #Renaming variables
                yint = yfinal
                xint_final = xint
                yint_final = yint

                #Integrate force across interface using trapezoidal rule
                for o in range(1, len(xint_final)):
                    int_sum_bub1 += ((xint_final[o]-xint_final[o-1])/2)*(yint_final[o]+yint_final[o-1])
                #plt.plot(xint, yfinal, linestyle='-', label='b1 pres', color='black')
                g = open(outPath2+outTxt_radial, 'a')
                for p in range(0, len(yfinal)-1):
                    g.write('{0:.2f}'.format(tst).center(20) + ' ')
                    g.write('{0:.6f}'.format(sizeBin).center(20) + ' ')
                    g.write('{0:.0f}'.format(np.amax(clust_size)).center(20) + ' ')
                    g.write('{0:.0f}'.format(interface_id).center(20) + ' ')
                    g.write('{0:.0f}'.format(bub_size_id_arr[m]).center(20) + ' ')
                    g.write('{0:.6f}'.format(xint[p]).center(20) + ' ')
                    g.write('{0:.6f}'.format(xint[p+1]).center(20) + ' ')
                    g.write('{0:.6f}'.format(alignfinal[p]).center(20) + ' ')
                    g.write('{0:.6f}'.format(densfinal[p]).center(20) + ' ')
                    g.write('{0:.6f}'.format(yfinal[p]).center(20) + '\n')
                g.close()
    """
