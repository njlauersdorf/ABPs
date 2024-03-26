
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

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility
import plotting_utility
import phase_identification
import binning

# Class of data output functions
class data_output:

    def __init__(self, lx_box, ly_box, sizeBin_x, sizeBin_y, tst, clust_large, dt_step):

        # Total x-length of box
        self.lx_box = lx_box

        # Total y-length of box
        self.ly_box = ly_box

        # X-length of bin
        self.sizeBin_x = sizeBin_x

        # Y-length of bin
        self.sizeBin_y = sizeBin_y

        # Current time step in Brownian time units
        self.tst = tst

        # Dumped/saved time step size in Brownian time units
        self.dt_step = dt_step

        # Largest cluster size
        self.clust_size = clust_large

    def write_to_csv(self, input_dict, outPath):

        import csv

        #Output values for radial measurements from CoM

        headers = ['tauB', 'sizeBin_x', 'sizeBin_y', 'clust_size']
        data = [self.tst, self.sizeBin_x, self.sizeBin_y, self.clust_size]

        is_file = os.path.isfile(outPath)

        for key, value in input_dict.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():

                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            key_new = key + '_' + key2 + '_' + key3
                            headers.append(key_new)
                            data.append(value3)

                    else:
                        key_new = key + '_' + key2
                        headers.append(key_new)
                        data.append(value2)
            else:
                headers.append(key)
                data.append(value)

        if is_file == 0:
            with open(outPath, 'w+') as f:
                writer=csv.writer(f)

                writer.writerow(headers)

            with open(outPath, 'a') as f:
                writer=csv.writer(f)

                writer.writerow(data)

        else:
            with open(outPath, 'r',) as f:
                #csvReader = csv.reader(f)
                final_line = f.readlines()[-1]

            tst_ind = final_line.find(',')
            prev_tst = float(final_line[:tst_ind])

            if round(self.tst - prev_tst, 1)==round(self.dt_step, 1):

                with open(outPath, 'a') as f:
                    writer=csv.writer(f)

                    writer.writerow(data)
            else:
                raise ValueError('System has already been run. Delete previous save file if you wish to proceed.')

    def write_to_txt(self, input_dict, outPath):

        #Output values for radial measurements from CoM
        headers = ['tauB', 'sizeBin_x', 'sizeBin_y', 'clust_size']
        data = [self.tst, self.sizeBin_x, self.sizeBin_y, self.clust_size]

        is_file = os.path.isfile(outPath)

        for key, value in input_dict.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():

                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            key_new = key + '_' + key2 + '_' + key3
                            headers.append(key_new)
                            data.append(value3)

                    else:
                        key_new = key + '_' + key2
                        headers.append(key_new)
                        data.append(value2)
            else:
                headers.append(key)
                data.append(value)

        if is_file == 0:
            header_string = ''
            with open(outPath, 'w+') as f:
                for i in range(0, len(headers)):
                    if i == len(headers)-1:
                        header_string += headers[i].center(20) + '\n'
                    else:
                        header_string += headers[i].center(20) + ' '
                f.write(header_string)

            with open(outPath, 'a') as f:
                for i in range(0, len(data)):
                    if i == len(data)-1:
                        if type(data[i])==int:
                            f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                        else:
                            f.write('{0:.9f}'.format(data[i]).center(20) + '\n')
                    else:
                        if type(data[i])==int:
                            f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                        else:
                            f.write('{0:.9f}'.format(data[i]).center(20) + ' ')


        else:
            with open(outPath, 'r',) as f:
                #csvReader = csv.reader(f)
                final_line = f.readlines()[-1].strip()

            tst_ind = final_line.find(' ')
            prev_tst = float(final_line[:tst_ind])
            if round(self.tst - prev_tst, 1)==round(self.dt_step, 1):

                with open(outPath, 'a') as f:
                    for i in range(0, len(data)):
                        if i == len(data)-1:
                            if type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                            else:
                                f.write('{0:.9f}'.format(data[i]).center(20) + '\n')
                        else:
                            if type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                            else:
                                f.write('{0:.9f}'.format(data[i]).center(20) + ' ')
            else:
                raise ValueError('System has already been run. Delete previous save file if you wish to proceed.')
    def write_to_txt(self, input_dict, outPath):

        #Output values for radial measurements from CoM
        headers = ['tauB', 'sizeBin_x', 'sizeBin_y', 'clust_size']
        data = [self.tst, self.sizeBin_x, self.sizeBin_y, self.clust_size]

        is_file = os.path.isfile(outPath)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():

                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            key_new = key + '_' + key2 + '_' + key3
                            headers.append(key_new)
                            data.append(value3)

                    else:
                        key_new = key + '_' + key2
                        headers.append(key_new)
                        data.append(value2)
            else:
                headers.append(key)
                data.append(value)
        
        if is_file == 0:
            header_string = ''
            with open(outPath, 'w+') as f:
                for i in range(0, len(headers)):
                    if i == len(headers)-1:
                        header_string += headers[i].center(20) + '\n'
                    else:
                        header_string += headers[i].center(20) + ' '
                f.write(header_string)

            arr_ind = 0
            arr_len = 1

            for i in range(0, len(data)):
                if type(data[i])==list:
                    arr_len_temp = len(data[i])
                    if arr_len_temp > arr_len:
                        arr_len = arr_len_temp
            
            with open(outPath, 'a') as f:
                while arr_ind < arr_len:
                    #print(arr_ind)
                    for i in range(0, len(data)):
                        if i == len(data)-1:
                            if type(data[i])==list:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1
                            elif (type(data[i])==int) | (type(data[i])==np.uint32):
                                f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif (type(data[i])==float) | (type(data[i])==np.float64) | (type(data[i])==np.float32):
                                f.write('{0:.9f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==str:
                                f.write('{}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            else:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1

                        else:
                            if type(data[i])==list:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + ' ')
                            elif (type(data[i])==int) | (type(data[i])==np.uint32):
                                f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                            elif (type(data[i])==float) | (type(data[i])==np.float64) | (type(data[i])==np.float32):
                                f.write('{0:.9f}'.format(data[i]).center(20) + ' ')
                            elif type(data[i])==str:
                                f.write('{}'.format(data[i]).center(20) + ' ')
                            else:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')


        else:

            arr_ind = 0
            arr_len = 1

            for i in range(0, len(data)):
                if type(data[i])==list:
                    arr_len_temp = len(data[i])
                    if arr_len_temp > arr_len:
                        arr_len = arr_len_temp

            with open(outPath, 'a') as f:
                while arr_ind < arr_len:
                    for i in range(0, len(data)):
                        
                        if i == len(data)-1:
                            
                            if type(data[i])==list:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1
                            elif (type(data[i])==int) | (type(data[i])==np.uint32):
                                f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif (type(data[i])==float) | (type(data[i])==np.float64) | (type(data[i])==np.float32):
                                f.write('{0:.9f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==str:
                                f.write('{}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            else:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1

                        else:
                            if type(data[i])==list:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + ' ')
                            elif (type(data[i])==int) | (type(data[i])==np.uint32):
                                f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                            elif (type(data[i])==float) | (type(data[i])==np.float64) | (type(data[i])==np.float32):
                                f.write('{0:.9f}'.format(data[i]).center(20) + ' ')
                            elif type(data[i])==str:
                                f.write('{}'.format(data[i]).center(20) + ' ')
                            else:
                                f.write('{0:.9f}'.format(data[i][arr_ind]).center(20) + ' ')

    def write_to_txt2(self, input_dict, outPath):

        #Output values for radial measurements from CoM
        headers = ['tauB', 'sizeBin_x', 'sizeBin_y', 'clust_size']
        data = [self.tst, self.sizeBin_x, self.sizeBin_y, self.clust_size]

        is_file = os.path.isfile(outPath)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():

                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            key_new = key + '_' + key2 + '_' + key3
                            headers.append(key_new)
                            data.append(value3)

                    else:
                        key_new = key + '_' + key2
                        headers.append(key_new)
                        data.append(value2)
            else:
                headers.append(key)
                data.append(value)

        if is_file == 0:
            header_string = ''
            with open(outPath, 'w+') as f:
                for i in range(0, len(headers)):
                    if i == len(headers)-1:
                        header_string += headers[i].center(20) + '\n'
                    else:
                        header_string += headers[i].center(20) + ' '
                f.write(header_string)

            arr_ind = 0
            arr_len = 1

            for i in range(0, len(data)):
                if type(data[i])==list:
                    arr_len_temp = len(data[i])
                    if arr_len_temp > arr_len:
                        arr_len = arr_len_temp

            with open(outPath, 'a') as f:
                while arr_ind < arr_len:
                    #print(arr_ind)
                    for i in range(0, len(data)):
                        if i == len(data)-1:
                            if type(data[i])==list:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==float:
                                f.write('{0:.6f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            else:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1

                        else:
                            if type(data[i])==list:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')
                            elif type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                            elif type(data[i]==float):
                                f.write('{0:.6f}'.format(data[i]).center(20) + ' ')
                            else:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')


        else:
            with open(outPath, 'r',) as f:
                #csvReader = csv.reader(f)
                final_line = f.readlines()[-1].strip()

            tst_ind = final_line.find(' ')
            prev_tst = float(final_line[:tst_ind])
            if round(self.tst - prev_tst, 1)==round(self.dt_step, 1):

                arr_ind = 0
                arr_len = 1

                for i in range(0, len(data)):
                    if type(data[i])==list:
                        arr_len_temp = len(data[i])
                        if arr_len_temp > arr_len:
                            arr_len = arr_len_temp

                with open(outPath, 'a') as f:
                    while arr_ind < arr_len:
                        for i in range(0, len(data)):
                            if i == len(data)-1:

                                if type(data[i])==list:
                                    f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                    arr_ind += 1
                                elif type(data[i])==int:
                                    f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                                    arr_ind += 1
                                elif type(data[i])==float:
                                    f.write('{0:.6f}'.format(data[i]).center(20) + '\n')
                                    arr_ind += 1
                                else:
                                    f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                    arr_ind += 1

                            else:
                                if type(data[i])==list:
                                    f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')
                                elif type(data[i])==int:
                                    f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                                elif type(data[i])==float:
                                    f.write('{0:.6f}'.format(data[i]).center(20) + ' ')
                                else:
                                    f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')
            else:
                raise ValueError('System has already been run. Delete previous save file if you wish to proceed.')
    def write_all_time_to_txt(self, input_dict, outPath):
        
        #Output values for radial measurements from CoM
        headers = ['sizeBin_x', 'sizeBin_y']
        data = [self.sizeBin_x, self.sizeBin_y]

        is_file = os.path.isfile(outPath)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():

                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            key_new = key + '_' + key2 + '_' + key3
                            headers.append(key_new)
                            data.append(value3)

                    else:
                        key_new = key + '_' + key2
                        headers.append(key_new)
                        data.append(value2)
            else:
                headers.append(key)
                data.append(value)

        if is_file == 0:
            header_string = ''
            with open(outPath, 'w+') as f:
                for i in range(0, len(headers)):
                    if i == len(headers)-1:
                        header_string += headers[i].center(20) + '\n'
                    else:
                        header_string += headers[i].center(20) + ' '
                f.write(header_string)

            arr_ind = 0
            arr_len = 1

            for i in range(0, len(data)):
                if (type(data[i])==list) | (type(data[i])==np.ndarray):
                    arr_len_temp = len(data[i])
                    if arr_len_temp > arr_len:
                        arr_len = arr_len_temp
            
            with open(outPath, 'a') as f:
                while arr_ind < arr_len:
                    #print(arr_ind)
                    
                    for i in range(0, len(data)):
                        if i == len(data)-1:
                            if type(data[i])==list:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            elif type(data[i])==float:
                                f.write('{0:.6f}'.format(data[i]).center(20) + '\n')
                                arr_ind += 1
                            else:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + '\n')
                                arr_ind += 1

                        else:
                            if type(data[i])==list:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')
                            elif type(data[i])==int:
                                f.write('{0:.0f}'.format(data[i]).center(20) + ' ')
                            elif type(data[i])==float:
                                f.write('{0:.6f}'.format(data[i]).center(20) + ' ')
                            else:
                                f.write('{0:.6f}'.format(data[i][arr_ind]).center(20) + ' ')