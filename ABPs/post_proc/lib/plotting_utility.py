
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

class plotting_utility:
    def __init__(self, lx_box, ly_box, partNum, typ):
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2
        self.partNum = partNum
        self.typ = typ
    def normalize(self, input_dict):

        all_input_x = input_dict['avg']['x']
        A_input_x = input_dict['avg']['x']
        B_input_x = input_dict['avg']['x']

        all_input_y = input_dict['avg A']['y']
        A_input_y = input_dict['avg A']['y']
        B_input_y = input_dict['avg A']['y']

        all_input_mag = input_dict['avg B']['mag']
        A_input_mag = input_dict['avg B']['mag']
        B_input_mag = input_dict['avg B']['mag']

        #Counts number of different particles belonging to each phase
        for ix in range(0, len(all_input_x)):
            for iy in range(0, len(all_input_y)):
                if all_input_mag[ix][iy]>0:
                    all_input_x_norm[ix][iy] = all_input_x[ix][iy] / all_input_mag[ix][iy]
                    all_input_y_norm[ix][iy] = all_input_y[ix][iy] / all_input_mag[ix][iy]
                else:
                    all_input_x_norm[ix][iy]=0
                    all_input_y_norm[ix][iy]=0

                if A_input_mag[ix][iy]>0:
                    A_input_x_norm[ix][iy] = A_input_x[ix][iy] / A_input_mag[ix][iy]
                    A_input_y_norm[ix][iy] = A_input_y[ix][iy] / A_input_mag[ix][iy]
                else:
                    A_input_x_norm[ix][iy] = 0
                    A_input_y_norm[ix][iy] = 0

                if B_input_mag[ix][iy]>0:
                    B_input_x_norm[ix][iy] = B_input_x[ix][iy] / B_input_mag[ix][iy]
                    B_input_y_norm[ix][iy] = B_input_y[ix][iy] / B_input_mag[ix][iy]
                else:
                    B_input_x_norm[ix][iy] = 0
                    B_input_y_norm[ix][iy] = 0

        norm_dict = {'avg': {'x': all_input_x_norm, 'y': all_input_y_norm}, 'avg A': {'x': A_input_x_norm, 'y': A_input_y_norm}, 'avg B': {'x': B_input_x_norm, 'y': B_input_y_norm}}
        return norm_dict

    def com_view(self, pos, clp_all):

        clust_size = clp_all.sizes                                  # find cluster sizes

        min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen

        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size

        #If a single cluster is greater than minimum size, determine CoM of largest cluster
        if len(large_clust_ind_all[0])>0:
            query_points=clp_all.centers[lcID]

            com_tmp_posX = query_points[0] + self.hx_box
            com_tmp_posY = query_points[1] + self.hy_box

            com_tmp_posX_temp = query_points[0]
            com_tmp_posY_temp = query_points[1]
        else:

            com_tmp_posX = self.hx_box
            com_tmp_posY = self.hy_box

            com_tmp_posX_temp = 0
            com_tmp_posY_temp = 0


        #shift reference frame to center of mass of cluster
        pos[:,0]= pos[:,0]-com_tmp_posX_temp
        pos[:,1]= pos[:,1]-com_tmp_posY_temp

        #Ensure particles are within simulation box (periodic boundary conditions)
        for i in range(0, self.partNum):
                if pos[i,0]>self.hx_box:
                    pos[i,0]=pos[i,0]-self.lx_box
                elif pos[i,0]<-self.hx_box:
                    pos[i,0]=pos[i,0]+self.lx_box

                if pos[i,1]>self.hy_box:
                    pos[i,1]=pos[i,1]-self.ly_box
                elif pos[i,1]<-self.hy_box:
                    pos[i,1]=pos[i,1]+self.ly_box

        com_dict = {'pos': pos, 'com': {'x': com_tmp_posX, 'y': com_tmp_posY}}

        return com_dict
    def grayscale_cmap(cmap):
        """Return a grayscale version of the given colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    def view_colormap(cmap):
        """Plot a colormap with its grayscale equivalent"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        cmap2 = plt.cm.get_cmap('Greys')
        colors2 = cmap2(np.arange(cmap2.N))
        print(cmap2(np.arange(cmap2.N))[:,:-1])
        colors2[:,:-1] = [1., 1., 1.]

        cmap3 = plt.cm.get_cmap('BuPu')
        my_cmap = cmap3(np.arange(cmap3.N))
        alphas = np.ones(cmap.N) * 0.3

        for i in range(cmap.N):

            my_cmap[i,:-1] = colors2[i,:-1] * alphas[i] + colors[i,:-1] * (1.0 - alphas[i])


        #cmap = grayscale_cmap(cmap)
        #grayscale = cmap(np.arange(cmap.N))

        return my_cmap
