
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
    def __init__(self, lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict):

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

        self.binning = binning.binning(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.ang, self.eps)

        self.plotting_utility = plotting_utility.plotting_utility(self.lx_box, self.ly_box, self.partNum, self.typ)

        self.phase_ident = phase_identification.phase_identification(self.area_frac_dict, self.align_dict, self.part_dict, self.press_dict, self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.parFrac, self.eps, self.typ)

        self.particle_prop_functs = particles.particle_props(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.eps, self.typ, self.pos, self.ang)

        self.theory_functs = theory.theory()

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
                                for j in binParts[indx][indy]:

                                    difx = self.utility_functs.sep_dist(self.pos[h][0], self.pos[j][0])

                                    dify = self.utility_functs.sep_dist(self.pos[h][1], self.pos[j][1])

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

    def shear_stress(self, stress_stat_dict):

        count_dict = self.phase_ident.phase_count(self.phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin
        int_area = count_dict['int'] * self.sizeBin
        gas_area = count_dict['gas'] * self.sizeBin

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

        press_arr = np.zeros((self.NBins, self.NBins))

        for ix in range(0, self.NBins):

            for iy in range(0, self.NBins):

                press_arr[ix][iy] = (stress_plot_dict['bin']['XX'][ix][iy] + stress_plot_dict['bin']['YY'][ix][iy])/(2 * self.sizeBin**2)

        return press_arr

    def virial_pressure_part(self, stress_plot_dict):

        press_arr = np.zeros(self.partNum)

        for h in range(0, self.partNum):

            press_arr[h] = (stress_plot_dict['part']['XX'][h] + stress_plot_dict['part']['YY'][h])/(2)

        return press_arr

    def virial_pressure(self, stress_stat_dict):

        bulkTrace = (stress_stat_dict['bulk']['all']['XX'] + stress_stat_dict['bulk']['all']['YY'])/2.
        intTrace = (stress_stat_dict['int']['all']['XX'] + stress_stat_dict['int']['all']['YY'])/2.
        gasTrace = (stress_stat_dict['gas']['all']['XX'] + stress_stat_dict['gas']['all']['YY'])/2.

        bulkTrace_A = (stress_stat_dict['bulk']['A']['XX'] + stress_stat_dict['bulk']['A']['YY'])/2.
        intTrace_A = (stress_stat_dict['int']['A']['XX'] + stress_stat_dict['int']['A']['YY'])/2.
        gasTrace_A = (stress_stat_dict['gas']['A']['XX'] + stress_stat_dict['gas']['A']['YY'])/2.

        bulkTrace_B = (stress_stat_dict['bulk']['B']['XX'] + stress_stat_dict['bulk']['B']['YY'])/2.
        intTrace_B = (stress_stat_dict['int']['B']['XX'] + stress_stat_dict['int']['B']['YY'])/2.
        gasTrace_B = (stress_stat_dict['gas']['B']['XX'] + stress_stat_dict['gas']['B']['YY'])/2.

        count_dict = self.phase_ident.phase_count(self.phase_dict)

        bulk_area = count_dict['bulk'] * self.sizeBin**2
        int_area = count_dict['int'] * self.sizeBin**2
        gas_area = count_dict['gas'] * self.sizeBin**2

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
                        num_densA_r[i-1] = len([partsA_defined])
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
                        num_densB_r[i-1] = len([partsB_defined])
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

    def total_active_pressure_bubble(self, com_radial_dict, sep_surface_dict, int_comp_dict):
        
        exterior_surface_arr = np.array([])
        interior_surface_arr = np.array([])

        int_id = int(int_comp_dict['ids'][np.where(int_comp_dict['comp']['all']==np.max(int_comp_dict['comp']['all']))[0][0]])

        print(sep_surface_dict)

        for m in range(0, len(sep_surface_dict)):
            if int(int_comp_dict['ids'][m]) != int_id:

                key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
                try:
                    exterior_surface_arr = np.append(exterior_surface_arr, sep_surface_dict[key]['exterior']['mean radius'])
                except:
                    exterior_surface_arr = np.append(exterior_surface_arr, 0)
                try:
                    interior_surface_arr = np.append(interior_surface_arr, sep_surface_dict[key]['interior']['mean radius'])
                except:
                    interior_surface_arr = np.append(interior_surface_arr, 0)

        print(exterior_surface_arr)
        print(interior_surface_arr)
        stop
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

        print(exterior_surface_arr)
        print(interior_surface_arr)
        stop

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
