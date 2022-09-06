
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


#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit


class utility:
    def __init__(self, l_box):

        self.l_box = l_box
        self.h_box = self.l_box/2

    def sep_dist(self, pos1, pos2):
          dif = pos1 - pos2
          dif_abs = np.abs(dif)
          if dif_abs>=self.h_box:
              if dif < -self.h_box:
                  dif += self.l_box
              else:
                  dif -= self.l_box

          return dif
    def sep_dist_arr(self, pos1, pos2, difxy=False):

        difr = (pos1 - pos2)

        difx_out = np.where(difr[:,0]>self.h_box)[0]
        difr[difx_out,0] = difr[difx_out,0]-self.l_box

        difx_out = np.where(difr[:,0]<-self.h_box)[0]
        difr[difx_out,0] = difr[difx_out,0]+self.l_box

        dify_out = np.where(difr[:,1]>self.h_box)[0]
        difr[dify_out,1] = difr[dify_out,1]-self.l_box

        dify_out = np.where(difr[:,1]<-self.h_box)[0]
        difr[dify_out,1] = difr[dify_out,1]+self.l_box

        difr_mag = (difr[:,0]**2 + difr[:,1]**2)**0.5

        if difxy == True:
            return difr[:,0], difr[:,1], difr_mag
        else:
            return difr_mag
    def shift_quadrants(self, difx, dify):
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
            n: number of bins along length of box
            decimals: exponent of multiplier for rounding (default=0)
        Output: number of bins along box length rounded up
        '''

        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

    def getNBins(self, length, minSz=(2**(1./6.))):
        '''
        Purpose: Given box size, return number of bins

        Inputs:
            length: length of box
            minSz: set minimum bin length to LJ cut-off distance
        Output: number of bins along box length rounded up
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

        Inputs: Quaternion orientation vector of particle

        Output: angle between [-pi, pi]
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
        """ Returns the symmetric log10 value """
        out_arr = np.zeros(np.shape(x))
        for d in range(0, len(x)):
            for f in range(0, len(x)):
                if x[d][f]!=0:
                    out_arr[d][f]=np.sign(x[d][f]) * np.log10(np.abs(x[d][f]))
        return out_arr
    """
    class Node:
       def __init__(self, key, val):
          self.key = key
          self.val = val
          self.next = None
    class LinkedList:
       def __init__(self):
          self.prehead = Node(None, None)
       def search(self, key):
          p = self.prehead.next
          while p:
             if p.key == key:
                return p
             p = p.next
          return None
       def add(self, key, val):
          p = self.search(key)
          if p:
             p.val = val
          else:
             node = Node(key, val)
             self.prehead.next, node.next = node, self.prehead.next
       def get(self, key):
          p = self.search(key)
          if p:
             return p.val
          else:
             return None
       def remove(self, key):
          prev = self.prehead
          cur = prev.next
          while cur:
             if cur.key == key:
                break
             prev, cur = cur, cur.next
          if cur:
             prev.next = cur.next
       def serialize(self):
          p = self.prehead.next
          ret = []
          while p:
             ret.append([p.key, p.val])
             p = p.next
          return ret
    class MyHashMap:
       def __init__(self):
          self.size = 1033
          self.arr = [LinkedList() for _ in range(self.size)]
       def _hash(self, key):
          return key % self.size
       def put(self, key, value):
          h = self._hash(key)
          self.arr[h].add(key, value)
       def get(self, key):
          h = self._hash(key)
          ret = self.arr[h].get(key)
          if ret is not None:
             return ret
          else:
             return -1
       def remove(self, key):
          h = self._hash(key)
          self.arr[h].remove(key)
    ob = utility_functs.utility.MyHashMap()
    key = ix * len(self.binParts) + iy
    ix = int(key / len(self.binParts));
    j = int(key % len(self.binParts));
    ob.put(1, 1)
    ob.put(2, 2)
    print(ob.get(1))
    print(ob.get(3))
    ob.put(2, 1)
    print(ob.get(2))
    ob.remove(2)
    print(ob.get(2))
    """
