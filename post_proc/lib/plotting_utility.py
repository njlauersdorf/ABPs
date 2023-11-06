
import sys
import os

from gsd import hoomd
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

# Class of plotting utility functions
class plotting_utility:
    def __init__(self, lx_box, ly_box, partNum, typ):

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        # Number of particles
        self.partNum = partNum

        # Array (partNum) of particle types
        self.typ = typ

    def normalize(self, input_dict):
        '''
        Purpose: Takes some input dictionary of binned vectors in the x- and y-
        directions and normalizes them by the total vector's magnitude

        Input:
        input_dict: input dictionary of arrays (NBins_x, NBins_y) of binned vectors
        in the x- and y-directions

        Output:
        norm_dict: dictionary of arrays (NBins_x, NBins_y) of binned unit vectors of
        the normalized x- and y-vectors in the input dictionary
        '''

        # Binned x-direction vectors of each respective type ('all', 'A', or 'B')
        all_input_x = input_dict['avg']['x']
        A_input_x = input_dict['avg']['x']
        B_input_x = input_dict['avg']['x']

        # Binned y-direction vectors of each respective type ('all', 'A', or 'B')
        all_input_y = input_dict['avg A']['y']
        A_input_y = input_dict['avg A']['y']
        B_input_y = input_dict['avg A']['y']

        # Binned vector magnitudes of each respective type ('all', 'A', or 'B')
        all_input_mag = input_dict['avg B']['mag']
        A_input_mag = input_dict['avg B']['mag']
        B_input_mag = input_dict['avg B']['mag']

        # Loops over all bins to normalize and find unit vectors
        for ix in range(0, len(all_input_x)):
            for iy in range(0, len(all_input_y)):

                # Calculates x- and y-direction unit vectors of all particles
                if all_input_mag[ix][iy]>0:
                    all_input_x_norm[ix][iy] = all_input_x[ix][iy] / all_input_mag[ix][iy]
                    all_input_y_norm[ix][iy] = all_input_y[ix][iy] / all_input_mag[ix][iy]
                else:
                    all_input_x_norm[ix][iy]=0
                    all_input_y_norm[ix][iy]=0

                # Calculates x- and y-direction unit vectors of A particles
                if A_input_mag[ix][iy]>0:
                    A_input_x_norm[ix][iy] = A_input_x[ix][iy] / A_input_mag[ix][iy]
                    A_input_y_norm[ix][iy] = A_input_y[ix][iy] / A_input_mag[ix][iy]
                else:
                    A_input_x_norm[ix][iy] = 0
                    A_input_y_norm[ix][iy] = 0

                # Calculates x- and y-direction unit vectors of B particles
                if B_input_mag[ix][iy]>0:
                    B_input_x_norm[ix][iy] = B_input_x[ix][iy] / B_input_mag[ix][iy]
                    B_input_y_norm[ix][iy] = B_input_y[ix][iy] / B_input_mag[ix][iy]
                else:
                    B_input_x_norm[ix][iy] = 0
                    B_input_y_norm[ix][iy] = 0

        # Dictionary of arrays (NBins_x, NBins_y) of binned unit vectors of
        # the normalized x- and y-vectors in the input dictionary
        norm_dict = {'avg': {'x': all_input_x_norm, 'y': all_input_y_norm}, 'avg A': {'x': A_input_x_norm, 'y': A_input_y_norm}, 'avg B': {'x': B_input_x_norm, 'y': B_input_y_norm}}
        return norm_dict

    def com_view(self, pos, clp_all):
        '''
        Purpose: Takes the position of particles and the largest cluster's CoM and
        shifts the position of all particles so the largest cluster's CoM is at the
        center of the system

        Input:
        pos: array (partNum) of positions (x,y,z) of each particle

        clp_all: cluster properties defined from Freud neighbor list and cluster calculation

        Output:
        com_dict: dictionary containing the shifted position of every particle such that
        the largest cluster's CoM is at the middle of the box (hx_box, hy_box) in addition to the
        unshifted largest cluster's CoM position
        '''

        # Array of cluster sizes
        clust_size = clp_all.sizes                                  # find cluster sizes

        # Minimum cluster size to consider
        min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen

        # ID of largest cluster
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster

        # IDs of all clusters of sufficiently large size
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size

        #If at least one cluster is sufficiently large, determine CoM of largest cluster
        if len(large_clust_ind_all[0])>0:

            # Largest cluster's CoM position
            query_points=clp_all.centers[lcID]

            # Largest cluster's CoM shifted to box size of x=[0, lx_box] and y=[0, ly_box]
            com_tmp_posX = query_points[0] + self.hx_box
            com_tmp_posY = query_points[1] + self.hy_box

            # Largest cluster's CoM in natural box size of x=[-hx_box, hx_box] and y=[-hy_box, hy_box]
            com_tmp_posX_temp = query_points[0]
            com_tmp_posY_temp = query_points[1]

        # If no sufficiently large clusters, set CoM position to middle of box (for plotting purposes)
        else:

            # Middle of box of x=[0, lx_box] and y=[0, ly_box]
            com_tmp_posX = self.hx_box
            com_tmp_posY = self.hy_box

            # Middle of box of x=[-hx_box, hx_box] and y=[-hy_box, hy_box]
            com_tmp_posX_temp = 0
            com_tmp_posY_temp = 0
        

        new_pos = pos.copy()
        #shift reference frame positions such that CoM of largest cluster is at mid-point of simulation box
        new_pos[:,0]= pos[:,0]-com_tmp_posX_temp
        new_pos[:,1]= pos[:,1]-com_tmp_posY_temp

        #Loop over all particles to ensure particles are within simulation box (periodic boundary conditions)
        for i in range(0, self.partNum):
            if new_pos[i,0]>self.hx_box:
                new_pos[i,0]=new_pos[i,0]-self.lx_box
            elif new_pos[i,0]<-self.hx_box:
                new_pos[i,0]=new_pos[i,0]+self.lx_box

            if new_pos[i,1]>self.hy_box:
                new_pos[i,1]=new_pos[i,1]-self.ly_box
            elif new_pos[i,1]<-self.hy_box:
                new_pos[i,1]=new_pos[i,1]+self.ly_box

        # Dictionary containing the shifted position of every particle such that
        # the largest cluster's CoM is at the middle of the box (hx_box, hy_box) in addition to the
        # unshifted largest cluster's CoM position

        com_dict = {'pos': new_pos, 'com': {'x': com_tmp_posX-self.hx_box, 'y': com_tmp_posY-self.hy_box}}

        return com_dict

    def com_part_view(self, pos_x_all, pos_y_all, pos_x, pos_y, com_x_parts_arr_time, com_y_parts_arr_time):
        '''
        Purpose: Takes the position of particles and the largest cluster's CoM and
        shifts the position of all particles so the largest cluster's CoM is at the
        center of the system

        Input:
        pos: array (partNum) of positions (x,y,z) of each particle

        clp_all: cluster properties defined from Freud neighbor list and cluster calculation

        Output:
        com_dict: dictionary containing the shifted position of every particle such that
        the largest cluster's CoM is at the middle of the box (hx_box, hy_box) in addition to the
        unshifted largest cluster's CoM position
        '''

        #plt.scatter(pos_x, pos_y, s=0.7)
        #plt.scatter(com_x_parts_arr_time, com_y_parts_arr_time)
        #plt.xlim(-self.hx_box, self.hx_box)
        #plt.ylim(-self.hy_box, self.hy_box)
        #plt.show()

        # Largest cluster's CoM shifted to box size of x=[0, lx_box] and y=[0, ly_box]
        com_tmp_posX = 0 - com_x_parts_arr_time
        com_tmp_posY = 0 - com_y_parts_arr_time

        #shift reference frame positions such that CoM of largest cluster is at mid-point of simulation box
        new_pos_x= pos_x+com_tmp_posX
        new_pos_y= pos_y+com_tmp_posY

        #plt.scatter(new_pos_x, new_pos_y, s=0.7)
        #plt.xlim(-self.hx_box, self.hx_box)
        #plt.ylim(-self.hy_box, self.hy_box)
        #plt.show()
        #stop

        #Loop over all particles to ensure particles are within simulation box (periodic boundary conditions)
        for i in range(0, len(new_pos_x)):
            if new_pos_x[i]>self.hx_box:
                new_pos_x[i]=new_pos_x[i]-self.lx_box
            elif new_pos_x[i]<-self.hx_box:
                new_pos_x[i]=new_pos_x[i]+self.lx_box

            if new_pos_y[i]>self.hy_box:
                new_pos_y[i]=new_pos_y[i]-self.ly_box
            elif new_pos_y[i]<-self.hy_box:
                new_pos_y[i]=new_pos_y[i]+self.ly_box
        
        

        if com_tmp_posX>self.hx_box:
            com_tmp_posX -= self.lx_box
        elif com_tmp_posX<-self.hx_box:
            com_tmp_posX += self.lx_box

        if com_tmp_posY>self.hy_box:
            com_tmp_posY -= self.ly_box
        elif com_tmp_posY<-self.hy_box:
            com_tmp_posY += self.ly_box      


        com_tmp_posX_new = np.mean(new_pos_x) - com_tmp_posX
        com_tmp_posY_new = np.mean(new_pos_y) - com_tmp_posY

        # Dictionary containing the shifted position of every particle such that
        # the largest cluster's CoM is at the middle of the box (hx_box, hy_box) in addition to the
        # unshifted largest cluster's CoM position

        com_dict = {'pos': {'x': new_pos_x, 'y': new_pos_y}, 'com': {'x': com_tmp_posX_new, 'y': com_tmp_posY_new}}

        return com_dict

    def grayscale_cmap(cmap):
        '''
        Purpose: Takes a colormap and returns the grayscale version of it

        Input:
        cmap: a colorized cmap object from matplotlib.pyplot.colormap

        Output:
        gray_cmap: a gray-scale cmap object from matplotlib.pyplot.colormap
        '''

        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        gray_cmap = LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

        return gray_cmap

    def view_colormap(cmap):
        '''
        Purpose: Takes a colormap and plots a colormap with its grayscale equivalent version

        Input:
        cmap: a colorized cmap object from matplotlib.pyplot.colormap

        Output:
        gray_cmap: a gray-scale cmap object from matplotlib.pyplot.colormap
        '''

        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        cmap2 = plt.cm.get_cmap('Greys')
        colors2 = cmap2(np.arange(cmap2.N))
        colors2[:,:-1] = [1., 1., 1.]

        cmap3 = plt.cm.get_cmap('BuPu')
        my_cmap = cmap3(np.arange(cmap3.N))
        alphas = np.ones(cmap.N) * 0.3

        for i in range(cmap.N):

            my_cmap[i,:-1] = colors2[i,:-1] * alphas[i] + colors[i,:-1] * (1.0 - alphas[i])


        #cmap = grayscale_cmap(cmap)
        #grayscale = cmap(np.arange(cmap.N))

        return my_cmap
