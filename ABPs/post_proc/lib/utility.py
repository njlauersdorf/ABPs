
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

#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit


class utility:
    def __init__(self, lx_box, ly_box):

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total x-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

    def sep_dist_x(self, pos1, pos2):
        '''
        Purpose: Calculates separation distance (accounting for periodic boundary conditions)
        in x direction given two points

        Inputs:
        pos1: x-location of a point

        pos2: x-location of a point

        Output:
        difr: separation distance in x-direction
        '''
        dif = pos1 - pos2
        dif_abs = np.abs(dif)
        if dif_abs>=self.hx_box:
            if dif < -self.hx_box:
                dif += self.lx_box
            else:
                dif -= self.lx_box

        return dif

    def sep_dist_y(self, pos1, pos2):
        '''
        Purpose: Calculates separation distance (accounting for periodic boundary conditions)
        in y direction given two points

        Inputs:
        pos1: y-location of a point

        pos2: y-location of a point

        Output:
        difr: separation distance in y-direction
        '''
        dif = pos1 - pos2
        dif_abs = np.abs(dif)
        if dif_abs>=self.hy_box:
            if dif < -self.hy_box:
                dif += self.ly_box
            else:
                dif -= self.ly_box

        return dif

    def sep_dist_arr(self, pos1, pos2, difxy=False):
        '''
        Purpose: Calculates separation distance (accounting for periodic boundary conditions)
        of each dimension between pairs of points

        Inputs:
        pos1: array of locations of points (x,y,z)

        pos2: array of locations of points (x,y,z)

        difxy (optional): if True, returns separation distance in x- and y- directions

        Output:
        difr_mag: array of separation distance magnitudes

        difx (optional): array of separation distances in x direction

        dify (optional): array of separation distances in y direction
        '''

        difr = (pos1 - pos2)

        difx_out = np.where(difr[:,0]>self.hx_box)[0]
        difr[difx_out,0] = difr[difx_out,0]-self.lx_box

        difx_out = np.where(difr[:,0]<-self.hx_box)[0]
        difr[difx_out,0] = difr[difx_out,0]+self.lx_box

        dify_out = np.where(difr[:,1]>self.hy_box)[0]
        difr[dify_out,1] = difr[dify_out,1]-self.ly_box

        dify_out = np.where(difr[:,1]<-self.hy_box)[0]
        difr[dify_out,1] = difr[dify_out,1]+self.ly_box

        difr_mag = (difr[:,0]**2 + difr[:,1]**2)**0.5

        if difxy == True:
            return difr[:,0], difr[:,1], difr_mag
        else:
            return difr_mag

    def shift_quadrants(self, difx, dify):
        '''
        Purpose: Calculates angle between X-axis and a given location (neighbor particle)
        from some origin (reference particle)

        Inputs:
        difx: array of interparticle separation distances in x direction

        dify: array of interparticle separation distances in y direction

        Output:
        ang_loc: array of angles between x-axis and a given location (i.e. neighbor particle)
        from some origin (i.e. reference particle) in terms of radians [-pi, pi]
        '''

        quad1 = np.where((difx > 0) & (dify >= 0))[0]
        quad2 = np.where((difx <= 0) & (dify > 0))[0]
        quad3 = np.where((difx < 0) & (dify <= 0))[0]
        quad4 = np.where((difx >= 0) & (dify < 0))[0]

        ang_loc = np.zeros(len(difx))

        ang_loc[quad1] = np.arctan(dify[quad1]/difx[quad1])
        ang_loc[quad2] = (np.pi/2) + np.arctan(-difx[quad2]/dify[quad2])
        ang_loc[quad3] = (np.pi) + np.arctan(dify[quad3]/difx[quad3])
        ang_loc[quad4] = (3*np.pi/2) + np.arctan(-difx[quad4]/dify[quad4])

        return ang_loc
    def roundUp(self, n, decimals=0):
        '''
        Purpose: Round up number of bins to account for floating point inaccuracy

        Inputs:
        n: number of bins along a given length of box

        decimals (optional): exponent of multiplier for rounding (default=0)

        Output:
        num_bins: number of bins along respective box length rounded up
        '''

        multiplier = 10 ** decimals
        num_bins = math.ceil(n * multiplier) / multiplier
        return num_bins

    def getNBins(self, length, minSz=(2**(1./6.))):
        '''
        Purpose: Given box size, return number of bins

        Inputs:
        length: length of box in a given dimension

        minSz (optional): minimum bin length (default set to LJ cut-off distance)

        Output: number of bins along respective box length rounded up
        '''

        initGuess = int(length) + 1
        nBins = initGuess
        # This loop only exits on function return
        while True:
            if length / nBins > minSz:
                return nBins
            else:
                nBins -= 1

    def quatToAngle(self, quat):
        '''
        Purpose: Take quaternion orientation vector of particle as given by hoomd-blue
        simulations and output angle between [-pi, pi]

        Inputs:
        quat: Quaternion orientation vector of particle

        Output:
        rad: angle between [-pi, pi]
        '''

        r = quat[0]         #magnitude
        x = quat[1]         #x-direction
        y = quat[2]         #y-direction
        z = quat[3]         #z-direction
        rad = math.atan2(y, x)

        return rad

    def symlog(self, x):
        """ Returns the symmetric log10 value """
        return np.sign(x) * np.log10(np.abs(x))

    def symlog_arr(self, x):
        """ Returns the symmetric log10 value of an array """
        out_arr = np.zeros(np.shape(x))
        for d in range(0, len(x)):
            for f in range(0, len(x)):
                if x[d][f]!=0:
                    out_arr[d][f]=np.sign(x[d][f]) * np.log10(np.abs(x[d][f]))
        return out_arr
