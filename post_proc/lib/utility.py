
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

    def crop_sim(self, start, end):
        import gsd.hoomd
        pre_crop = gsd.hoomd.open("file_name", mode='r')
        post_crop = "new_file_name"
        with gsd.hoomd.open(post_crop, mode='wb') as f:
            for i in range(start, end):
                f.append(pre_crop[i])

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
    def quaternion_rotation_matrix(Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix
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

        rot_matrix = utility.quaternion_rotation_matrix(quat)
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        orientation_vector = np.matmul(rot_matrix, z_axis)
        
        rad = math.atan2(y, x)
        rad2 = math.atan2(r, np.sqrt(x**2+y**2+z**2))
        alpha = 2*np.arccos(r)
        beta_x = np.arccos(x/np.sin(alpha/2))
        beta_y = np.arccos(y/np.sin(alpha/2))
        beta_z = np.arccos(z/np.sin(alpha/2))
        phi = np.arctan2(2*(r*x + y*z),1-2*(x**2+y**2))
        theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(r*y-x*z)), np.sqrt(1-2*(r*y-x*z)))
        psi = np.arctan2(2*(r*z + x*y),1-2*(y**2+z**2))
        
        x_vect = orientation_vector[0]
        y_vect = orientation_vector[1]
        #sto
        
        
        #print(rad)
        ##print(alpha)
        #print(beta_x)
        #print(beta_y)
        #print(beta_z)
        theta = np.arctan2(orientation_vector[1][0], orientation_vector[0][0])
        #print((orientation_vector[1]**2 + orientation_vector[0]**2)**0.5 * np.cos(np.arctan2(orientation_vector[1], orientation_vector[0])))
        #stop
        

        return theta#rad

    def quatToXOrient(self, quat):
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

        rot_matrix = utility.quaternion_rotation_matrix(quat)
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        orientation_vector = np.matmul(rot_matrix, z_axis)

        x_vect = orientation_vector[0]
        y_vect = orientation_vector[1]
        

        return x_vect

    def quatToYOrient(self, quat):
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

        rot_matrix = utility.quaternion_rotation_matrix(quat)
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        orientation_vector = np.matmul(rot_matrix, z_axis)

        x_vect = orientation_vector[0]
        y_vect = orientation_vector[1]
        

        return y_vect

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
