#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:48 2020

@author: nicklauersdorf
"""

# This is the import cell
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.ticker as ticker
first = True
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 10)

# Here are my rc parameters for matplotlib
mpl.rc('font', serif='Helvetica Neue') 
#mpl.rcParams.update({'font.size': 9})
#mpl.rcParams['figure.figsize'] = 3.2, 2.8
#mpl.rcParams['figure.dpi'] = 2000
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 1.5
# Get the current path
if first:
    parent = '/Users/nicklauersdorf/Jupyternb/'#radial_density_cluster/' #os.getcwd()
#print(parent)
#stop
os.chdir(parent)
# Grab file names from data folder
dens = os.listdir('rdf_data_Tom_eps_1.0')
try:
    dens.remove('.DS_Store')
except:
    print(".DS_Store not in directory")
    
# Some functions to get the relevant data from the filenames
def checkFile(fname, string):
    for i in range(0,len(fname)):
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
    for i in range(0,len(fname)):
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
    for i in range(0,len(cpy1)):
        for j in range(0,len(cpy1)):
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
    for i in range(0,len(arr1)):
        arr1[i] = cpy[arr2[i]]

# Grab parameters, sort them
chkStrings = ["pe", "pa", "pb", "xa", "ep", "phi", "cluster", "dtau"]
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
#    # Sort them!
    print("Sorting... ")
    # Sort by: pe, phi, epsilon, cluster
    indArr = multiSort(storeVals[chkStrings.index("pa")],
                       storeVals[chkStrings.index("phi")],
                       storeVals[chkStrings.index("ep")],
                       storeVals[chkStrings.index("cluster")])
    indSort(dens, indArr)
    for i in storeVals:
        indSort(i, indArr)

# Now that the data is sorted, read it into a dataframe
all_dens = []
os.chdir(parent)
os.chdir('rdf_data_Tom_eps_1.0')
print('poop')
for i in dens:
    print(i)
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
    ep = float(storeVals[chkStrings.index("ep")][i])
    phi = float(storeVals[chkStrings.index("phi")][i])
    dtau = float(storeVals[chkStrings.index("dtau")][i])
    df = pd.DataFrame([[pe, pa, pb, xa, ep, phi, dtau]], columns=headers)
    params = params.append(df, ignore_index = True)
    
    
# Let's add columns to the time-resolved simulation data
for i in range(0,len(all_dens)):
    # Ger rid of NaN in favor of 0
    all_dens[i].fillna(0, inplace=True)

headers=list(all_dens[0])
# This is how you access the data at different levels
#display(all_dens[0])

# Let's try and plot phiG and phiC vs peNet
print("Totals")
gas = "Gas-r=5.0"
liq = "Liq-r=5.0"

distEps = []
distPhi = []
params_num=len(params)
for i in range(0, len(params)):
    params_num+=1
    if params['eps'][i] not in distEps:
        distEps.append(params['eps'][i])
    if params['phi'][i] not in distPhi:
        distPhi.append(params['phi'][i])
distPhi.sort(reverse=True)
distEps.sort(reverse=True)

#print(distPhi)
#print(distEps)

cols = []
cnt = 0.
np.array([])
for i in range(0,len(all_dens)):
    times=np.array([])
    for j in range(0,len(all_dens[i]['tst'])):
        if all_dens[i]['tst'][j] in times:
            pass
        else:
            times=np.append(times,all_dens[i]['tst'][j])
    rad_length=len(all_dens[i]['tst'])/len(times)
    time_length=0
    phi_0_sum=0
    phi_1_sum=0
    phi_total_sum=0
    NinBin_sum=0
    
    #start=np.where(times==0.0)[0]
    #end=np.where(times==.0)[0]
    print(all_dens[0])
    stop
    for j in range(0,len(times)):
        rad_search=np.array(all_dens[i]['search_radius'])[int(j*rad_length):int((j+1)*rad_length)]
        phi_total=np.array(all_dens[i]['gas_A_dens'])[int(j*rad_length):int((j+1)*rad_length)]
        phi_0=np.array(all_dens[i]['gas_B_dens'])[int(j*rad_length):int((j+1)*rad_length)]
        #phi_1=np.array(all_dens[i]['phiLoc1'])[int(j*rad_length):int((j+1)*rad_length)]
        #NinBin=np.array(all_dens[i]['NinBin'])[int(j*rad_length):int((j+1)*rad_length)]
        print(len(rad_search))
        print(len(phi_total))
        print(len(phi_0))
        time_length+=1
        phi_0_sum+=phi_0
        #phi_1_sum+=gas_B_dens
        phi_total_sum+=phi_total
        #NinBin_sum+=NinBin
    print(np.shape(phi_total_sum))
    print(np.shape(phi_0_sum))
    print(np.shape(rad_search))
    plt.plot(rad_search,phi_0_sum)
    plt.plot(rad_search,phi_total_sum)
    plt.show()
stop
'''
    fig = plt.figure()
    ax = plt.subplot(111)
    line, = ax.plot(rad_search,phi_total_sum, label='total')
    line, = ax.plot(rad_search,phi_0_sum,label='slow')
    line, = ax.plot(rad_search,phi_1_sum,label='fast')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('$r_{CoM}$')
    ax.set_ylabel('Density')
    plt.title('Total radial density vs '+'$r_{CoM}$')
    plt.show()
    stop
    #plt.savefig('Total_radial_density_pa'+str(params['peA'][i])+'_pb'+str(params['peB'][i])+'_eps'+str(params['eps'][i])+ '_xa='+str(params['xA'][i])+'.png')
    #plt.close()
    
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    line, = ax.plot(rad_search,phi_total_sum/NinBin_sum, label='total')
    line, = ax.plot(rad_search,phi_0_sum/NinBin_sum,label='slow')
    line, = ax.plot(rad_search,phi_1_sum/NinBin_sum,label='fast')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('$r_{CoM}$')
    ax.set_ylabel('Normalized Density')
    plt.title('Normalized total radial density vs '+'$r_{CoM}$')
    plt.savefig('Normalized_radial_density_pa'+str(params['peA'][i])+'_pb'+str(params['peB'][i])+'_eps'+str(params['eps'][i])+ '_xa='+str(params['xA'][i])+'.png')
    plt.close()
    
    fig = plt.figure()
    ax = plt.subplot(111)
    line, = ax.plot(rad_search,phi_total_sum/time_length, label='total')
    line, = ax.plot(rad_search,phi_0_sum/time_length,label='slow')
    line, = ax.plot(rad_search,phi_1_sum/time_length,label='fast')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('$r_{CoM}$')
    ax.set_ylabel('Density')
    plt.title('Time averaged radial density vs '+'$r_{CoM}$')
    plt.savefig('Time Averaged total_radial_density_pa'+str(params['peA'][i])+'_pb'+str(params['peB'][i])+'_eps'+str(params['eps'][i])+ '_xa='+str(params['xA'][i])+'.png')
    plt.close()
    
    fig = plt.figure()
    ax = plt.subplot(111)
    line, = ax.plot(rad_search,(phi_total_sum/NinBin_sum)/time_length, label='total')
    line, = ax.plot(rad_search,(phi_0_sum/NinBin_sum)/time_length,label='slow')
    line, = ax.plot(rad_search,(phi_1_sum/NinBin_sum)/time_length,label='fast')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('$r_{CoM}$')
    ax.set_ylabel('Normalized Density')
    plt.title('Time Averaged, normalized radial density vs '+'$r_{CoM}$')
    plt.savefig('Time_averaged_normalized_radial_density_pa'+str(params['peA'][i])+'_pb'+str(params['peB'][i])+'_eps'+str(params['eps'][i])+ '_xa='+str(params['xA'][i])+'.png')
    plt.close()
print('done!')
'''