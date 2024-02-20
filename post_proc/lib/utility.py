
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

from scipy.optimize import curve_fit


class utility:
    """
    Purpose: 
    This class contains a series of basic functions for making analysis of hoomd files easier, including
    cropping of simulation files, calculating separation distances, finding ideal bin size/number,
    and converting quaternions to x- and y-unit vector orientations or angles from the x-axis
    """
    def __init__(self, lx_box, ly_box):

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total x-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

    def crop_sim(self, start, end):
        import gsd.hoomd

        # Open old .gsd file
        pre_crop = gsd.hoomd.open("file_name", mode='r')

        # Name new .gsd file
        post_crop = "new_file_name"

        # Crop old .gsd file in time and save to new .gsd file
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

        # Total separation distance in x (with periodic boundary errors)
        dif = pos1 - pos2

        # Enforce periodic boundary conditions in x
        dif_abs = np.abs(dif)
        if dif_abs>=self.hx_box:
            if dif < -self.hx_box:
                dif += self.lx_box
            else:
                dif -= self.lx_box

        # Return separation distance in x
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

        # Total separation distance in y (with periodic boundary errors)
        dif = pos1 - pos2

        # Enforce periodic boundary conditions in y
        dif_abs = np.abs(dif)
        if dif_abs>=self.hy_box:
            if dif < -self.hy_box:
                dif += self.ly_box
            else:
                dif -= self.ly_box

        # Return separation distance in y
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

        # Separation distance in x and y (with periodic boundary errors)
        difr = (pos1 - pos2)

        # Enforce periodic boundary conditions in x
        difx_out = np.where(difr[:,0]>self.hx_box)[0]
        difr[difx_out,0] = difr[difx_out,0]-self.lx_box

        difx_out = np.where(difr[:,0]<-self.hx_box)[0]
        difr[difx_out,0] = difr[difx_out,0]+self.lx_box

        #Enforce periodic boundary conditions in y
        dify_out = np.where(difr[:,1]>self.hy_box)[0]
        difr[dify_out,1] = difr[dify_out,1]-self.ly_box

        dify_out = np.where(difr[:,1]<-self.hy_box)[0]
        difr[dify_out,1] = difr[dify_out,1]+self.ly_box

        # Total actual separation distance
        difr_mag = (difr[:,0]**2 + difr[:,1]**2)**0.5

        # If you want both x, y, and r returned or just r
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

        # Find particles in each quadrant of graph
        quad1 = np.where((difx > 0) & (dify >= 0))[0]
        quad2 = np.where((difx <= 0) & (dify > 0))[0]
        quad3 = np.where((difx < 0) & (dify <= 0))[0]
        quad4 = np.where((difx >= 0) & (dify < 0))[0]
        
        # Initialize empty array
        ang_loc = np.zeros(len(difx))

        #Calculate each particle's angular location around particle given interparticle separation distances and quadrant
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

        # Round up number of bins to nearest integer
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

        # Initial number of bins in one dimension you want
        initGuess = int(length) + 1
        nBins = initGuess

        # This loop only exits on function return and checks if number of bins correct to return correct bin number 
        while True:
            if length / nBins > minSz:
                return nBins
            else:
                nBins -= 1
    def gsd_to_csv(self, dataPath, outFile, pos, x_orient_arr, y_orient_arr, typ, time_step, first_time = 1):
        '''
        Purpose: Convert important information from gsd to csv format

        Inputs: 
        dataPath: output path for csv files

        outFile: base name of ouput files with important simulation parameters

        pos (partNum, 3): array of particle positions

        x_orient_arr (partNum): array of x-orientation unit vectors of particles

        y_orient_arr (partNum): array of y-orientation unit vectors of particles

        typ (partNum): array of particle types, either slow (0) or fast (1)

        time_step (float): current time step

        first_time (optional): if true and this is first time data is saved, save headers to output files
        '''
        
        import csv

        # Creat output csv files to save simulation information to
        with open(dataPath + outFile + '_positions_x.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_pos_x:
            csv_writer_pos_x = csv.writer(open_csv_pos_x)
            with open(dataPath + outFile + '_positions_y.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_pos_y:
                csv_writer_pos_y = csv.writer(open_csv_pos_y)
                with open(dataPath + outFile + '_orientations_x.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_orient_x:
                    csv_writer_orient_x = csv.writer(open_csv_orient_x)
                    with open(dataPath + outFile + '_orientations_y.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_orient_y:
                        csv_writer_orient_y = csv.writer(open_csv_orient_y)
                        with open(dataPath + outFile + '_time_and_box_width.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_time_and_box:
                            csv_writer_time_and_box = csv.writer(open_csv_time_and_box)
                            with open(dataPath + outFile + '_type.csv', 'a+', encoding = 'UTF8', newline = '') as open_csv_typ:
                                csv_writer_typ = csv.writer(open_csv_typ)

                                # Particle IDs as headers for saved arrays
                                part_ids = np.linspace(0, len(pos)-1, num=len(pos), dtype=int)

                                # If first save, save headers
                                if first_time == 1:
                                    csv_writer_pos_x.writerow(part_ids)
                                    csv_writer_pos_y.writerow(part_ids)
                                    csv_writer_typ.writerow(part_ids)
                                    csv_writer_orient_x.writerow(part_ids)
                                    csv_writer_orient_y.writerow(part_ids)
                                    csv_writer_time_and_box.writerow(['box_width', 'time_step'])
                                
                                # Save x-positions of particles
                                csv_writer_pos_x.writerow(pos[:,0])
                            
                                # Save y-positions of particles
                                csv_writer_pos_y.writerow(pos[:,1])

                                # Save time and box information of current time step
                                csv_writer_time_and_box.writerow([self.lx_box, time_step])

                                # Save activity of particles
                                csv_writer_typ.writerow(typ)

                                # Save x-orientation unit vector of particles
                                csv_writer_orient_x.writerow(x_orient_arr)

                                # Save y-orientation unit vector of particles
                                csv_writer_orient_y.writerow(y_orient_arr)
                                

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
        theta: angle between [-pi, pi]
        '''
        
        r = quat[0]         #magnitude
        x = quat[1]         #x-direction
        y = quat[2]         #y-direction
        z = quat[3]         #z-direction

        # Define rotation matrix
        rot_matrix = utility.quaternion_rotation_matrix(quat)
        
        # Define axis to rotate around
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        # Find orientation vector through matrix multiplication
        orientation_vector = np.matmul(rot_matrix, z_axis)
        
        # Define orientation in radians
        rad = math.atan2(y, x)
        rad2 = math.atan2(r, np.sqrt(x**2+y**2+z**2))

        # Define some terms used to calculate orientation
        alpha = 2*np.arccos(r)
        beta_x = np.arccos(x/np.sin(alpha/2))
        beta_y = np.arccos(y/np.sin(alpha/2))
        beta_z = np.arccos(z/np.sin(alpha/2))

        # Calculate different rotation angles around axes 
        phi = np.arctan2(2*(r*x + y*z),1-2*(x**2+y**2))
        theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(r*y-x*z)), np.sqrt(1-2*(r*y-x*z)))
        psi = np.arctan2(2*(r*z + x*y),1-2*(y**2+z**2))
        
        # Define x and y orientation
        x_vect = orientation_vector[0][0]
        y_vect = orientation_vector[1][0]

        # Define orientation in terms of angle from x-axis
        theta = np.arctan2(orientation_vector[1][0], orientation_vector[0][0])
 
        return theta#rad

    def quatToXOrient(self, quat):
        '''
        Purpose: Take quaternion orientation vector of particle as given by hoomd-blue
        simulations and output x direction unit vector

        Inputs:
        quat: Quaternion orientation vector of particle

        Output:
        x_vect: x-orientation unit vector
        '''
    
        r = quat[0]         #magnitude
        x = quat[1]         #x-direction
        y = quat[2]         #y-direction
        z = quat[3]         #z-direction

        rot_matrix = utility.quaternion_rotation_matrix(quat)
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        orientation_vector = np.matmul(rot_matrix, z_axis)

        x_vect = orientation_vector[0][0]
        y_vect = orientation_vector[1][0]
        

        return x_vect

    def quatToYOrient(self, quat):
        '''
        Purpose: Take quaternion orientation quaternion of particle as given by hoomd-blue
        simulations and output y direction unit vector

        Inputs:
        quat: Quaternion orientation vector of particle

        Output:
        y_vect: y-orientation unit vector
        '''

        r = quat[0]         #magnitude
        x = quat[1]         #x-direction
        y = quat[2]         #y-direction
        z = quat[3]         #z-direction

        # Define rotation matrix
        rot_matrix = utility.quaternion_rotation_matrix(quat)
        
        # Define axis to rotate around
        z_axis = np.array([0,0,1])
        z_axis = np.reshape(z_axis, (3,1))

        # Find orientation vector through matrix multiplication
        orientation_vector = np.matmul(rot_matrix, z_axis)

        # Define x and y orientation
        x_vect = orientation_vector[0][0]
        y_vect = orientation_vector[1][0]
        
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
