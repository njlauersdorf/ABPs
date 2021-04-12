#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:46:17 2020

@author: nicklauersdorf
"""

# This is the import cell
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.ticker as ticker
from collections import OrderedDict
from scipy import stats
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d


first = True
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 10)

# Here are my rc parameters for matplotlibf
fsize = 9
mpl.rc('font', serif='Helvetica Neue') 
mpl.rcParams.update({'font.size': fsize})
mpl.rcParams['figure.figsize'] = 3.2, 2.8
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5
# Set x tick params
mpl.rcParams['xtick.major.size'] = 4.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 3.
mpl.rcParams['xtick.minor.width'] = 1.25
# Set y tick params
mpl.rcParams['ytick.major.size'] = 4.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 3.
mpl.rcParams['ytick.minor.width'] = 1.25
# Load LaTeX and amsmath
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# Get the current path
if first:
    parent = os.getcwd()
os.chdir(parent)

# Grab file names from data folder
dens = os.listdir('../rad_vp_eps_12')
try:
    data.remove('.DS_Store')
except:
    print(".DS_Store not in directory")

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Some functions to get the relevant data from the filenames
def checkFile(fname, string):
    for i in range(len(fname)):
        if fname[i] == string[0]:
#             print"{} matches {}".format(fname[i], string[0])
            for j in range(1, len(string)):
                if (i + j) > (len(fname) - 1):
                    break
                elif fname[i + j] == string[j]:
#                     print"{} matches {}".format(fname[i+j], string[j])
                    if j == (len(string) - 1):
#                         print"Final match!"
                        return True
                else:
                    break
    return False
    
def txtValue(fname, string):
    out = ""
    index = 0
    for i in range(len(fname)):
        if fname[i] == string[0]:
            for j in range(1, len(string)):
                if (i + j) > (len(fname) - 1):
                    break
                elif fname[i + j] == string[j]:
                    if j == (len(string) - 1):
                        # Last index of search string
                        index = i + j
                else:
                    break
                        
    # First index of value
    index += 1
    mybool = True
    while mybool:
        if fname[index].isdigit():
            out = out + fname[index]
            index += 1
        elif fname[index] == ".":    
            if fname[index+1].isdigit():
                out = out + fname[index]
                index += 1
            else:
                mybool = False
        else:
            mybool = False
    return float(out)

# Sorting functions
def multiSort(arr1, arr2, arr3, arr4):
    """Sort an array the slow (but certain) way, returns original indices in sorted order"""
    # Doing this for PeR, PeS, xS in this case
    cpy1 = np.copy(arr1)
    cpy2 = np.copy(arr2)
    cpy3 = np.copy(arr3)
    cpy4 = np.copy(arr4)
    ind = np.arange(0, len(arr1))
    for i in range(len(cpy1)):
        for j in range(len(cpy1)):
            # Sort by first variable
            if cpy1[i] > cpy1[j] and i < j:
                # Swap copy array values
                cpy1[i], cpy1[j] = cpy1[j], cpy1[i]
                cpy2[i], cpy2[j] = cpy2[j], cpy2[i]
                cpy3[i], cpy3[j] = cpy3[j], cpy3[i]
                cpy4[i], cpy4[j] = cpy4[j], cpy4[i]
                # Swap the corresponding indices
                ind[i], ind[j] = ind[j], ind[i]
                
            # If first variable is equal, resort to second variable
            elif cpy1[i] == cpy1[j] and cpy2[i] > cpy2[j] and i < j:
                # Swap copy array values
                cpy1[i], cpy1[j] = cpy1[j], cpy1[i]
                cpy2[i], cpy2[j] = cpy2[j], cpy2[i]
                cpy3[i], cpy3[j] = cpy3[j], cpy3[i]
                cpy4[i], cpy4[j] = cpy4[j], cpy4[i]
                # Swap the corresponding indices
                ind[i], ind[j] = ind[j], ind[i]
                
            elif cpy1[i] == cpy1[j] and cpy2[i] == cpy2[j] and cpy3[i] > cpy3[j] and i < j:
                # Swap copy array values
                cpy1[i], cpy1[j] = cpy1[j], cpy1[i]
                cpy2[i], cpy2[j] = cpy2[j], cpy2[i]
                cpy3[i], cpy3[j] = cpy3[j], cpy3[i]
                cpy4[i], cpy4[j] = cpy4[j], cpy4[i]
                # Swap the corresponding indices
                ind[i], ind[j] = ind[j], ind[i]
            elif cpy1[i] == cpy1[j] and cpy2[i] == cpy2[j] and cpy3[i] == cpy3[j] and cpy4[i] > cpy4[j] and i < j:
                # Swap copy array values
                cpy1[i], cpy1[j] = cpy1[j], cpy1[i]
                cpy2[i], cpy2[j] = cpy2[j], cpy2[i]
                cpy3[i], cpy3[j] = cpy3[j], cpy3[i]
                cpy4[i], cpy4[j] = cpy4[j], cpy4[i]
                # Swap the corresponding indices
                ind[i], ind[j] = ind[j], ind[i]
    return ind

def indSort(arr1, arr2):
    """Take sorted index array, use to sort array"""
    # arr1 is array to sort
    # arr2 is index array
    cpy = np.copy(arr1)
    for i in range(len(arr1)):
        arr1[i] = cpy[arr2[i]]


def edge_width(val_arr):
    
    #Calculate slope of alpha(x) between sparse array values
    deriv=np.zeros(len(val_arr)-1)
    for j in range(0,len(val_arr)-1):
        deriv[j]=(val_arr[j+1]-val_arr[j])#/(rcom_new[j+1]-rcom_new[j])
        
    #Find slopes that are greater than 10% of the maximum slope
    deriv_max_ind=np.where(deriv==np.max(deriv))[0][0]
    val_max_ind=np.where(val_arr==np.max(val_arr))[0][0]
    val_max=val_arr[val_max_ind]
    #Find slopes that are greater than 10% of the maximum slope
    deriv_min_ind=np.where(deriv==np.min(deriv))[0][0]

    max_slope=deriv[deriv_max_ind]
    min_slope=deriv[deriv_min_ind]
    #start_range=deriv[:deriv_max_ind]
    i=deriv_max_ind-1
    while_i=0
    while (deriv[i]>(0.1*max_slope)):
        i-=1
    #    while_i=1
    #if while_i==1:
    #    i=i+1
    #i=i-1
    j=deriv_min_ind+1
    while_j=0
    while (deriv[j]<(0.1*min_slope)):
        j+=1
    #j=j+1
    #    while_j=0
    #if while_j==1:
    #    j=j-1
    return {'begin': i, 'end': j, 'deriv':deriv}

def edge_begin_funct(val_arr, rad_arr):
    
    #Calculate slope of alpha(x) between sparse array values
    deriv=np.zeros(len(val_arr)-1)
    for j in range(0,len(val_arr)-1):
        deriv[j]=(val_arr[j+1]-val_arr[j])#/(rcom_new[j+1]-rcom_new[j])
        
    #Find slopes that are greater than 10% of the maximum slope
    deriv_max_ind=np.where(deriv==np.max(deriv))[0][0]
    #Find slopes that are greater than 10% of the maximum slope
    val_max_ind=np.where(val_arr==np.max(val_arr))[0][0]

    skip=0
    if (val_max_ind-5)<=deriv_max_ind<=(val_max_ind):
        max_slope=deriv[deriv_max_ind]
    elif val_max_ind == 0:
        skip = 1
    elif 0<=val_max_ind<=4:
        max_slope=np.max(deriv[0:val_max_ind])
        deriv_max_ind=np.where(deriv==max_slope)[0][0]
    else:
    #    print('3')
        
        max_slope=np.max(deriv[val_max_ind-5:val_max_ind])
        deriv_max_ind=np.where(deriv==max_slope)[0][0]
        
    #start_range=deriv[:deriv_max_ind]
    if skip==0:
        j=deriv_max_ind
        while_j=0
    
        while (((((deriv[j]))>(0.2*max_slope))) or (val_arr[j]>(0.2*np.max(val_arr))) ):
    
            if ((deriv[j+1]<0.0) and (deriv[j]<0.0)):
                if (val_max_ind-j)>1:
                    j+=1
                    while_j=0
                    break
                else:
                    j-=1
                    while_j=1
            elif val_arr[j]<=0.0:
                while_j=0
                break
            elif ((deriv[j]<0.0) and (val_arr[j]<0.2*np.max(val_arr))):
                if (val_max_ind-j)>1:
                    j+=1
                    while_j=0
                    break
                else:
                    j-=1
                    while_j=1
            else:
                
                j-=1
                while_j=1
            if j <= 0.0:
                while_j=0
                break
        if while_j==1:
            j+=1
    elif skip==1:
        j=0
    
    #plt.plot(rad_arr, val_arr)
    #plt.plot(rad_arr[j:], val_arr[j:])
    #plt.plot(rad_arr[:len(rad_arr)-1], deriv)
    #plt.plot(rad_arr[j:len(rad_arr)-1], deriv[j:])
    #plt.show()
    return j


def edge_end_funct(val_arr, rad_arr):
    
    #Calculate slope of alpha(x) between sparse array values
    skip=0
    deriv=np.zeros(len(val_arr)-1)
    for j in range(0,len(val_arr)-1):
        deriv[j]=(val_arr[j+1]-val_arr[j])#/(rcom_new[j+1]-rcom_new[j])
    #Find slopes that are greater than 10% of the maximum slope
    deriv_min_ind=np.where(deriv==np.min(deriv))[0][0]
    
    val_max_ind=np.where(val_arr==np.max(val_arr))[0][0]
    
    if (val_max_ind)<=deriv_min_ind<=(val_max_ind+6):
        min_slope=deriv[deriv_min_ind]
    elif (len(val_arr)-4)<=val_max_ind<(len(val_arr)-1):
        min_slope=np.max(deriv[val_max_ind:len(val_arr)])
        deriv_min_ind=np.where(deriv==min_slope)[0][0]
    elif val_max_ind>=(len(val_arr)-1):
        skip=1
    else:
    #    print('3')
        
        min_slope=np.min(deriv[val_max_ind:val_max_ind+6])
        deriv_min_ind=np.where(deriv==min_slope)[0][0]

    #if ((len(deriv)-10)<=deriv_min_ind<=len(deriv)) or ((len(val_arr)-5)<=val_max_ind<=len(val_arr)):
    #    print('1')
    #    j=0.0
    #    skip=1
    #elif (val_max_ind)<deriv_min_ind<(val_max_ind+5):
    #    print('2')
    #    min_slope=deriv[deriv_min_ind]
    #elif deriv_min_ind==val_max_ind:
    #    deriv_min_ind=deriv_min_ind+1
        #min_slope=deriv[deriv_min_ind]
    
    #else:
    #    print('3')
        
    #    min_slope=np.max(deriv[val_max_ind+1:val_max_ind+5])
    #    deriv_min_ind=np.where(deriv==min_slope)[0][0]
    if skip==0:
        j=deriv_min_ind
    
        while (((((deriv[j]))<(0.2*min_slope))) or (val_arr[j]>(0.2*np.max(val_arr))) ):
    
            if ((deriv[j-1]>0.0) and (deriv[j]>0.0)):
                if (j-val_max_ind)>1:
                    j-=1
                    while_j=0
                    break
                else:
                    j+=1
                    while_j=1
            elif ((deriv[j]>0.0) and (val_arr[j]<0.2*np.max(val_arr))):
                if (j-val_max_ind)>1:
                    j-=1
                    while_j=0
                    break
                else:
                    j+=1
                    while_j=1
            elif val_arr[j]<=0.0:
                while_j=0
                break
            else:
                
                j+=1
                while_j=1
            if j >= len(deriv):
                while_j=0
                break
    else:
        j=0
    return j


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    #deriv_max_ind_done=np.min(np.where(>0.05*max_slope)[0])
    #deriv_min_ind_done=np.max(np.where(np.abs(alpha_n_deriv[alpha_n_deriv_min_ind+1:])>0.05*np.abs(min_slope))[0])
    #        print(alpha_n_deriv_min_ind)
    #        print(alpha_n_deriv_min_ind_done)
    #        print(alpha_n_deriv_max_ind)
    #        print(alpha_n_deriv_max_ind_done)
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def computeVel(activity):
    "Given particle activity, output intrinsic swim speed"
    # This gives:
    # v_0 = Pe * sigma / tau_B = Pe * sigma / 3 * tau_R
    velocity = (activity * sigma) / (3 * (1/D_r))
    return velocity

def computeActiveForce(velocity):
    "Given particle activity, output repulsion well depth"
    # This is multiplied by Brownian time and gives:
    #          Pe = 3 * v_0 * tau_R / sigma
    # the conventional description of the Peclet number
    activeForce = velocity * threeEtaPiSigma
    return activeForce

def computeEps(alpha, activeForce):
    "Given particle activity, output repulsion well depth"
    # Here is where we will be testing the ratio we use (via alpha)
    epsilon = (alpha * activeForce * sigma / 24.0) + 1.0
    # Add 1 because of integer rounding
    epsilon = int(epsilon) + 1
    return epsilon

def avgCollisionForce(peNet):
    '''Computed from the integral of possible angles'''
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
    return (magnitude * peNet) / (np.pi) 

def maximum(a, b, c, d): 
  
    if (a >= b) and (a >= c) and (a >= d): 
        largest = a 
    elif (b >= a) and (b >= c) and (b >= d): 
        largest = b 
    elif (c >= a) and (c >= b) and (c >= d):
        largest = c
    else: 
        largest = d
          
    return largest 

def minimum(a, b, c, d): 
  
    if (a <= b) and (a <= c) and (a <= d): 
        smallest = a 
    elif (b <= a) and (b <= c) and (b <= d): 
        smallest = b 
    elif (c <= a) and (c <= b) and (c <= d):
        smallest = c
    else: 
        smallest = d
          
    return smallest 
def quatToAngle(quat):
    "Take vector, output angle between [-pi, pi]"
    #print(quat)
    r = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    #print(2*math.atan2(x,r))
    rad = math.atan2(y, x)#(2*math.acos(r))#math.atan2(y, x)#
    
    return rad
def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ
def ljForce(r, eps, sigma=1.):
    '''Compute the Lennard-Jones force'''
    div = (sigma/r)
    dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
    return dU
def getLat(peNet, eps):
    '''Get the lattice spacing for any pe'''
    if peNet == 0:
        return 2.**(1./6.)
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(peNet):
            r -= j
        r += j
    return r  
# Grab parameters, sort them
chkStrings = ["pe", "pa", "pb", "xa", "eps", "phi", "cluster", "dtau"]
default = [0., 0., 0., 100., 1., 60., 0, 0.000001]
storeVals = [[] for i in chkStrings]
for i in dens:
    for j in range(0, len(chkStrings)):
        if chkStrings[j] != "cluster":
            if checkFile(i, chkStrings[j]):
                storeVals[j].append(txtValue(i, chkStrings[j]))
            else:
                storeVals[j].append(default[j])  
        else:
            if checkFile(i, chkStrings[j]):
                storeVals[j].append(1)
            else:
                storeVals[j].append(default[j]) 
                
# Issue with epsilon in file output 0 -> 0.0001
for i in range(0, len(storeVals[4])):
    if storeVals[4][i] == 0.0:
        storeVals[4][i] = 0.0001

# Sort the arrays
if len(storeVals[0]) > 1:
    # Sort them!
#     print("Sorting... ")
    # Sort by: pe, phi, epsilon, cluster
    indArr = multiSort(storeVals[chkStrings.index("pa")],
                       storeVals[chkStrings.index("phi")],
                       storeVals[chkStrings.index("eps")],
                       storeVals[chkStrings.index("cluster")])
    indSort(dens, indArr)
    for i in storeVals:
        indSort(i, indArr)

# Now that the data is sorted, read it into a dataframe
all_dens = []
os.chdir(parent)
os.chdir('../rad_vp_eps_12')
for i in dens:
#     print(i)
    df = pd.read_csv(i, sep='\s+', header=0)
    all_dens.append(df)
os.chdir(parent)

# Grab the parameters from each file, store in a dataframe
headers = ['pe', 'peA', 'peB', 'xA', 'eps', 'phi', 'tauPer_dt']
params = pd.DataFrame(columns=headers)
for i in range(0, len(all_dens)):
    pe = int(storeVals[chkStrings.index("pe")][i])
    pa = int(storeVals[chkStrings.index("pa")][i])
    pb = int(storeVals[chkStrings.index("pb")][i])
    xa = float(storeVals[chkStrings.index("xa")][i])
    ep = float(storeVals[chkStrings.index("eps")][i])
    phi = float(storeVals[chkStrings.index("phi")][i])
    dtau = float(storeVals[chkStrings.index("dtau")][i])
    df = pd.DataFrame([[pe, pa, pb, xa, ep, phi, dtau]], columns=headers)
    params = params.append(df, ignore_index = True)
# This is how you access the data at different levels

# Let's add columns to the time-resolved simulation data
for i in range(len(all_dens)):
    # Ger rid of NaN in favor of 0
    all_dens[i].fillna(0, inplace=True)

headers=list(all_dens[0])

#outTxt = 'alignment_totals.txt'
#g = open(outTxt, 'w+') # write file headings
#g.write('pe'.center(15) + ' ' +\
#        'phi'.center(15) + ' ' +\
#        'eps'.center(15) + ' ' +\
#        'lat'.center(15) + ' ' +\
#        'gamma'.center(15) + '\n')
#g.close()
int_len=3.0
fracd=3/int_len
fracd=1.0

for i in range(0, len(all_dens)):
    print(params['eps'][i])
    if params['eps'][i]==0.0001:
        new_cut_off=int(4*fracd)
    elif params['eps'][i]==0.001:
        new_cut_off=int(6*fracd)
    elif params['eps'][i]==0.01:
        new_cut_off=int(8*fracd)
    elif params['eps'][i]==0.1:
        new_cut_off=int(10*fracd)
    elif params['eps'][i]==1.0:
        new_cut_off=int(12*fracd)
    outTxt2 = 'align_single_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.txt'
    outTxt3 = 'align_single_interface_only_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.txt'
    
    g = open(outTxt2, 'w+') # write file headings
    g.write('pe'.center(15) + ' ' +\
        'phi'.center(15) + ' ' +\
        'eps'.center(15) + ' ' +\
        'radius'.center(15) + ' ' +\
        'num_dens'.center(15) + ' ' +\
        'align'.center(15) + ' ' +\
        'press_vp'.center(15) + ' ' +\
        'timeframes'.center(15) + ' ' +\
        'lat'.center(15) + ' ' +\
        'int_begin'.center(15) + ' ' +\
        'int_end'.center(15) + '\n')
    g.close()
    
    g = open(outTxt3, 'w+') # write file headings
    g.write('pe'.center(15) + ' ' +\
        'phi'.center(15) + ' ' +\
        'eps'.center(15) + ' ' +\
        'radius'.center(15) + ' ' +\
        'num_dens'.center(15) + ' ' +\
        'align'.center(15) + ' ' +\
        'press_vp'.center(15) + ' ' +\
        'timeframes'.center(15) + ' ' +\
        'lat'.center(15) + ' ' +\
        'int_begin'.center(15) + ' ' +\
        'int_end'.center(15) + '\n')
    g.close()
    
    time_list = []
    print(params['pe'][i])
    print(params['phi'][i])
    min_ang_list=[]
    max_ang_list=[]
    ang_dif=(all_dens[i]['max_ang'][0]-all_dens[i]['min_ang'][0])
    angs=np.arange(0,360+ang_dif,ang_dif)
    for j in range(0,len(all_dens[i]['tauB'])):
        if all_dens[i]['tauB'][j] in time_list:
            pass
        else:
            time_list.append(all_dens[i]['tauB'][j])
    times=len(time_list)
    times_one=len(np.where((all_dens[i]['tauB']==all_dens[i]['tauB'][0]))[0])
    data=len(all_dens)
    rad_len=np.where((all_dens[i]['tauB']==all_dens[i]['tauB'][0]) & (all_dens[i]['min_ang']==all_dens[i]['min_ang'][0]))[0]
    radial_steps=all_dens[i]['radius'][new_cut_off:(len(rad_len))]
    radial_steps_very_orig=all_dens[i]['radius'][:(len(rad_len))]
    a=[]

    for j in range(0,times):
        a.append(all_dens[i]['clust_size'][times_one*j])

    radial_steps=pd.Series.to_numpy(radial_steps)
    radial_steps_very_orig=pd.Series.to_numpy(radial_steps_very_orig)
    #radial_steps=np.insert(radial_steps,0, 0) 
    alpha_tot=0
    align_tot_orig=0
    num_dens_tot_orig=0
    align_tot=0
    press_vp_tot=0
    align_tot2=0
    press_vp_tot2=0
    num_dens_tot2=0
    num_dens_tot=0
    alpha_n_int_num=0
    clust_rad=0
    int_width_avg=0
    int_begin_avg=0
    int_end_avg=0
    for j in range(0,times-1):
        
        if all_dens[i]['clust_size'][times_one*j]>=np.max(all_dens[i]['clust_size'])*0.95:
            for k in range(0,len(angs)):

                #print(all_dens[i]['radius'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                num_dens=pd.Series.to_numpy(all_dens[i]['num_dens'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                num_dens=smooth(num_dens,3)
                num_dens_very_orig=pd.Series.to_numpy(all_dens[i]['num_dens'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                if num_dens[10]>1.0:
                    #if j==10:
                    #    if k==7:
                    #        print(all_dens[i]['radius'][7+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))-5])
                    #num_dens=np.insert(num_dens,0, num_dens[0], axis=0) 
                    
                    align_very_orig=pd.Series.to_numpy(all_dens[i]['align'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    align_orig=pd.Series.to_numpy(all_dens[i]['align'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    #align_orig=np.insert(align_orig,0, align_orig[0], axis=0) 
                    align_orig=smooth(align_orig,3)
                    
                    press_vp_very_orig=pd.Series.to_numpy(all_dens[i]['press_vp'][(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    press_vp_orig=pd.Series.to_numpy(all_dens[i]['press_vp'][new_cut_off+(times_one*j+k*len(rad_len)):(times_one*j+(k+1)*len(rad_len))])
                    #align_orig=np.insert(align_orig,0, align_orig[0], axis=0) 
                    press_vp_orig=smooth(press_vp_orig,3)

                    alpha_orig=align_orig*num_dens
                    alpha_very_orig=align_very_orig*num_dens_very_orig
                    
                    #alpha_n_int = interp1d(radial_steps, alpha_orig, kind='cubic')
                    #alpha_n_int_new=alpha_n_int(rad_new)
                    
                    rad_end_new=np.where(alpha_orig==np.max(alpha_orig))[0][0]
                    if np.max(alpha_orig)>=0.2:
                        c=edge_end_funct(alpha_orig, radial_steps)
                        if c!=0:
                            d=edge_begin_funct(alpha_orig, radial_steps)
                            if d!=0:
                                end_rad=radial_steps[c]
                                begin_rad=radial_steps[d]
                                
                                end_rad_orig_ind = find_nearest(radial_steps_very_orig, end_rad)
                                end_rad_point = radial_steps_very_orig[end_rad_orig_ind]
                                
                                begin_rad_orig_ind = find_nearest(radial_steps_very_orig, begin_rad)
                                begin_rad_point = radial_steps_very_orig[begin_rad_orig_ind]
                                if begin_rad_point<end_rad_point:
                                    '''
                                    plt.plot(radial_steps_very_orig[new_cut_off:], alpha_very_orig[new_cut_off:])
                                    plt.plot(radial_steps_very_orig[begin_rad_orig_ind:end_rad_orig_ind+1], alpha_very_orig[begin_rad_orig_ind:end_rad_orig_ind+1], label='interface')
                                    file_name_final='align_plot_pe_'+str(params['pe'][i])+'_phi_'+str(params['phi'][i])+'_eps_'+str(params['eps'][i])+'_time_'+str(time_list[j])+'_ang_'+str(k*18)+'.png'
                                    plt.ylabel(r'$\alpha(r)n(r)')
                                    plt.xlabel('r')
                                    plt.legend()
                                    plt.show()
                                    '''
                                    
                                    #plt.savefig('/Volumes/External/whingdingdilly-master/ipython/clusters_soft/align_data/'+file_name_final)
                                    
                                    #plt.close()
                                    '''
                                    norm_rad_very_orig=radial_steps_very_orig/end_rad_point
                                    norm_rad_very_orig_app = np.append(norm_rad_very_orig, 0.0)
                                    num_dens_very_orig_app = np.append(num_dens_very_orig, num_dens_very_orig[0])
                                    align_very_orig_app = np.append(align_very_orig, 0.0)
                                    alpha_very_orig_app = align_very_orig_app * num_dens_very_orig_app
                                    
                                    norm_rad_end=np.where(norm_rad_very_orig_app==1.0)[0][0]
                                    
                                    norm_rad=norm_rad_very_orig_app[:norm_rad_end+1]
                                                
                                    int_len2=norm_rad.max()/10000.0
                                    
                                    rad_renew=np.arange(0.0, 1.0+int_len2, int_len2)
            
                                    alpha_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_very_orig_app, kind='cubic')
                                    align_int_renew = interp1d(norm_rad_very_orig_app, align_very_orig_app, kind='cubic')
                                    num_dens_int_renew = interp1d(norm_rad_very_orig_app, num_dens_very_orig_app, kind='cubic')
                                    alpha_n_int_rerenew=alpha_n_int_renew(rad_renew)
                                    align_int_rerenew=align_int_renew(rad_renew)
                                    num_dens_int_rerenew=num_dens_int_renew(rad_renew)
                                    '''
                                    
                                    
                                    norm_rad_very_orig=radial_steps_very_orig/end_rad_point
                                    norm_rad_very_orig_app = np.append(norm_rad_very_orig, 0.0)
                                    num_dens_very_orig_app = np.append(num_dens_very_orig, num_dens_very_orig[0])
                                    align_very_orig_app = np.append(align_very_orig, 0.0)
                                    press_vp_very_orig_app = np.append(press_vp_very_orig, press_vp_very_orig[0])

                                    alpha_very_orig_app = align_very_orig_app * num_dens_very_orig_app
                                    
                                    #norm_rad_end=np.where(norm_rad_very_orig_app==1.0)[0][0]
                                    
                                    norm_rad=norm_rad_very_orig_app#[:norm_rad_end+1]
                                    if np.max(norm_rad)>=1.0:
                                        int_len2=1.0/10000.0
                                        
                                        rad_renew=np.arange(0.0, 1.0+int_len2, int_len2)
                
                                        alpha_n_int_renew = interp1d(norm_rad_very_orig_app, alpha_very_orig_app, kind='cubic')
                                        align_int_renew = interp1d(norm_rad_very_orig_app, align_very_orig_app, kind='cubic')
                                        num_dens_int_renew = interp1d(norm_rad_very_orig_app, num_dens_very_orig_app, kind='cubic')
                                        press_vp_int_renew = interp1d(norm_rad_very_orig_app, press_vp_very_orig_app, kind='cubic')
                                        alpha_n_int_rerenew=alpha_n_int_renew(rad_renew)
                                        align_int_rerenew=align_int_renew(rad_renew)
                                        num_dens_int_rerenew=num_dens_int_renew(rad_renew)
                                        press_vp_int_rerenew=press_vp_int_renew(rad_renew)
                                        
                                        
                                        
                                
                                        alpha_n_int_num+=1
                                        align_tot+=align_int_rerenew
                                        press_vp_tot+=press_vp_int_rerenew
                                        num_dens_tot+=num_dens_int_rerenew
                                        int_end_avg+=end_rad_point
                                        int_begin_avg+=begin_rad_point
                                        
                                        rad_min=radial_steps_very_orig-begin_rad_point
                                        rad_end_min = end_rad_point - begin_rad_point
                                        norm_rad_very_orig2 = rad_min / rad_end_min
                                        norm_rad_begin2 = np.where(norm_rad_very_orig2==0.0)[0][0]
                                        norm_rad_end2 = np.where(norm_rad_very_orig2==1.0)[0][0]
            
                                        norm_rad3=norm_rad_very_orig2[norm_rad_begin2:norm_rad_end2+1]
                                        align_int_rerenew2 = align_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        num_dens_int_rerenew2 = num_dens_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        alpha_n_int_rerenew2 = alpha_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        press_vp_int_rerenew2 = press_vp_very_orig[norm_rad_begin2:norm_rad_end2+1]
                                        
                                               
                                        int_len3=norm_rad3.max()/10000.0
                                        
                                        rad_renew3=np.arange(0.0, 1.0+int_len3, int_len3)
                
                                        alpha_n_int_renew2 = interp1d(norm_rad_very_orig2, alpha_very_orig, kind='cubic')
                                        align_int_renew2 = interp1d(norm_rad_very_orig2, align_very_orig, kind='cubic')
                                        num_dens_int_renew2 = interp1d(norm_rad_very_orig2, num_dens_very_orig, kind='cubic')
                                        press_vp_int_renew2 = interp1d(norm_rad_very_orig2, press_vp_very_orig, kind='cubic')
                                        alpha_n_int_rerenew2=alpha_n_int_renew2(rad_renew3)
                                        align_int_rerenew2=align_int_renew2(rad_renew3)
                                        num_dens_int_rerenew2=num_dens_int_renew2(rad_renew3)
                                        press_vp_int_rerenew2=press_vp_int_renew2(rad_renew3)
                                        
                                        
                                        
                                        
                                        align_tot2+=align_int_rerenew2
                                        num_dens_tot2+=num_dens_int_rerenew2
                                        press_vp_tot2+=press_vp_int_rerenew2
                                    
                                    else:
                                        pass
                                    
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass

                    else:
                        pass
    lat=getLat(params['pe'][i],params['eps'][i])
    '''
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.plot(rad_renew, num_dens_tot/alpha_n_int_num, label='n(r)')
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('all_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('align_alpha_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, align_tot/alpha_n_int_num, label=r'$\alpha(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('align_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, alpha_tot/alpha_n_int_num, label=r'$\alpha(r)*n(r)$')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('alpha_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    plt.figure()
    plt.plot(rad_renew, num_dens_tot/alpha_n_int_num, label='n(r)')
    plt.title(r'$r_c=$'+str(round(clust_rad/alpha_n_int_num,2))+r'$,\delta=$'+str(round(int_width_tot,2))+',a='+str(round(lat,3))+r'$,\delta/a=$'+str(round(int_width_tot/lat,0)))
    plt.xlabel(r'$r/r_c$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('num_dens_pe'+str(params['pe'][i])+'_phi'+str(params['phi'][i])+'_eps'+str(params['eps'][i])+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    plt.close()
    '''
    #g = open(outTxt, 'a')
    #g.write('{0:.0f}'.format(params['pe'][i]).center(15) + ' ')
    #g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ')
    #g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
    #g.write('{0:.6f}'.format(lat).center(15) + ' ')
    #g.write('{0:.6f}'.format(int_width_tot).center(15) + '\n')
    #g.close()
    if isinstance(num_dens_tot, int)==0:
        g = open(outTxt2, 'a') # write file headings

        for m in range(0,len(num_dens_tot)):
            g.write('{0:.0f}'.format(params['pe'][i]).center(15) + ' ')
            g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ') 
            g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
            g.write('{0:.6f}'.format(rad_renew[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(num_dens_tot[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(align_tot[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(press_vp_tot[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(alpha_n_int_num).center(15) + ' ')
            g.write('{0:.6f}'.format(lat).center(15) + ' ')
            g.write('{0:.6f}'.format(int_begin_avg).center(15) + ' ')
            g.write('{0:.6f}'.format(int_end_avg).center(15) + '\n')
        g.close()
        
        g = open(outTxt3, 'a') # write file headings
        for m in range(0,len(num_dens_tot)):
            g.write('{0:.0f}'.format(params['pe'][i]).center(15) + ' ')
            g.write('{0:.0f}'.format(params['phi'][i]).center(15) + ' ') 
            g.write('{0:.4f}'.format(params['eps'][i]).center(15) + ' ')
            g.write('{0:.6f}'.format(rad_renew3[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(num_dens_tot2[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(align_tot2[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(press_vp_tot2[m]).center(15) + ' ')
            g.write('{0:.6f}'.format(alpha_n_int_num).center(15) + ' ')
            g.write('{0:.6f}'.format(lat).center(15) + ' ')
            g.write('{0:.6f}'.format(int_begin_avg).center(15) + ' ')
            g.write('{0:.6f}'.format(int_end_avg).center(15) + '\n')
        g.close()
    
    
            #plt.plot(rad_new[:len(rad_new)-1], a['deriv'], label='derivative')
            #plt.plot(rad_new, alpha_n_int_new, label='not deriv')
            
            #plt.plot(rad_new[a['begin']:a['end']+1], a['deriv'][a['begin']:a['end']+1], label='derivative')
            #plt.plot(rad_new[a['begin']:a['end']+1], alpha_n_int_new[a['begin']:a['end']+1], label='not deriv')
            #plt.legend()
            #plt.show()
            #plt.plot(radial_steps, align_orig*num_dens)
            #plt.plot(radial_steps[:edge_end_int+1]/radial_steps[:edge_end_int], num_dens[edge_begin_int:edge_end_int+1]*align_orig[edge_begin_int:edge_end_int+1])

            #plt.show()
            #print(norm_rad)
            #stop
    '''
            rad_clust=rad_new[:a['end']]
            plt.plot(rad_new[:len(rad_new)-1], a['deriv'], label='derivative')
            plt.plot(rad_new, alpha_n_int_new, label='not deriv')
            plt.legend()
            plt.show()
            
            
            stop
            print(rad_new[alpha_n_deriv_min_ind+alpha_n_deriv_min_ind_done+1])
            print(rad_new[alpha_n_deriv_min_ind_done])
            print(rad_new[alpha_n_deriv_max_ind])
            print(rad_new[alpha_n_deriv_max_ind_done+alpha_n_deriv_max_ind])

            plt.plot(rad_new, alpha_n_int_new)
            plt.plot(rad_new[:len(rad_new)-1], smooth(alpha_n_deriv,4))
            plt.show()
            stop
            stop
            print(align_deriv_max_ind[0])
            stop
            begin_int_1=np.max(align_deriv_max_ind[0])
            
            #Define end_int as the last location of radius array
            
            #Only use negative slopes
            align_deriv_ind=np.where(align_deriv<0)
            
            #Find negative slopes whose magnitudes are greater than 10% of the maximum slope
            align_deriv_max_ind=np.where(np.abs(align_deriv[align_deriv_ind])>0.05*np.max(align_deriv))
            
            
            #Find the latest occurance of slope that meets above 2 criteria
            print(align_deriv_max_ind[0])
            begin_int_1=np.max(align_deriv_max_ind[0])
    '''
print('done!')
#    print(len(all_dens[i]['tauB']))
#    print(len(all_dens[i]['clust_size']))
#    print(len(all_dens[i]['min_ang']))
#    print(len(all_dens[i]['max_ang']))
#    print(len(all_dens[i]['radius']))
#    print(len(all_dens[i]['num_dens']))
#    print(len(all_dens[i]['align']))
#    print(all_dens[0][headers[1]][0])
#print(all_dens[0][headers[2]][0])
#print(headers)