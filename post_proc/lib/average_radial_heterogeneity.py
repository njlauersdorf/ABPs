#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:50:02 2020

@author: nicklauersdorf
"""

'''
#                           This is an 80 character line                       #
What does this file do?
(Reads single argument, .gsd file name)
1.) Read in .gsd file of particle positions
2.) Mesh the space
3.) Loop through tsteps and ...
3a.) Place all particles in appropriate mesh grid
3b.) Calculates location of steady state cluster's center of mass
3c.) Translate particles positions such that origin (0,0) is cluster's center of mass
3b.) Loop through all bins ...
3b.i.) Compute number density and average orientation per bin
3c.) Determine and label each phase (bulk dense phase, gas phase, and gas-bulk interface)
3d.) Calculate and output parameters to be used to calculate area fraction of each or all phase(s)
3e.) For frames with clusters, plot particle positions color-coded by phase it's a part of
4) Generate movie from frames
'''

# Import modules
import sys
import os
import matplotlib.pyplot as plt
# Path to hoomd and output data
hoomdPath=str(sys.argv[2])
outPath = str(sys.argv[3])


# Import modules
import pandas as pd
import numpy as np

# Add path to post-processing library
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Get infile and open
inFile = str(sys.argv[1])

average_theta_path = outPath + 'averages/radial_avgs_theta' + inFile[20:-4] + '.csv'
average_rad_path = outPath + 'averages/radial_avgs_rad' + inFile[20:-4] + '.csv'


average_fa_path = outPath + 'averages/radial_avgs_fa' + inFile[20:-4] + '.csv'
average_faA_path = outPath + 'averages/radial_avgs_faA' + inFile[20:-4] + '.csv'
average_faB_path = outPath + 'averages/radial_avgs_faB' + inFile[20:-4] + '.csv'

average_num_dens_path = outPath + 'averages/radial_avgs_num_dens' + inFile[20:-4] + '.csv'
average_num_densA_path = outPath + 'averages/radial_avgs_num_densA' + inFile[20:-4] + '.csv'
average_num_densB_path = outPath + 'averages/radial_avgs_num_densB' + inFile[20:-4] + '.csv'

average_indiv_vals_path = outPath + 'averages/radial_avgs_indiv_vals' + inFile[20:-4] + '.csv'

file_path = outPath + inFile

#Read input file
start_val = 0
df = pd.read_csv(file_path, sep='\s+', header=0)

import csv 

averages_file =  open(average_theta_path, newline='')
avg_theta_r = np.array(list(csv.reader(averages_file)))

avg_theta_r_flatten = (avg_theta_r.flatten()).astype(np.float) 

averages_file =  open(average_rad_path, newline='')
avg_rad_r = np.array(list(csv.reader(averages_file)))

avg_rad_r_flatten = (avg_rad_r.flatten()).astype(np.float) 

averages_file =  open(average_fa_path, newline='')
avg_fa_r = np.array(list(csv.reader(averages_file)))

avg_fa_r_flatten = (avg_fa_r.flatten()).astype(np.float) 

averages_file =  open(average_faA_path, newline='')
avg_faA_r = np.array(list(csv.reader(averages_file)))

avg_faA_r_flatten = (avg_faA_r.flatten()).astype(np.float) 

averages_file =  open(average_faB_path, newline='')
avg_faB_r = np.array(list(csv.reader(averages_file)))

avg_faB_r_flatten = (avg_faB_r.flatten()).astype(np.float) 

averages_file =  open(average_num_dens_path, newline='')
avg_num_dens_r = np.array(list(csv.reader(averages_file)))

avg_num_dens_r_flatten = (avg_num_dens_r.flatten()).astype(np.float) 

averages_file =  open(average_num_densA_path, newline='')
avg_num_densA_r = np.array(list(csv.reader(averages_file)))

avg_num_densA_r_flatten = (avg_num_densA_r.flatten()).astype(np.float) 

averages_file =  open(average_num_densB_path, newline='')
avg_num_densB_r = np.array(list(csv.reader(averages_file)))

avg_num_densB_r_flatten = (avg_num_densB_r.flatten()).astype(np.float) 

averages_file =  open(average_indiv_vals_path, newline='')
reader = csv.reader(averages_file)
avg_indiv_vals = dict(reader)
avg_radius_r_flatten= float(avg_indiv_vals['radius'])

avg_com_x_r_flatten = float(avg_indiv_vals['com_x'])
avg_com_y_r_flatten = float(avg_indiv_vals['com_y'])



print(np.shape(avg_fa_r))
avg_theta_r_new = np.zeros(np.shape(avg_fa_r))
print(avg_theta_r_flatten)
print(len(avg_theta_r_flatten))
for i in range(0, len(avg_rad_r_flatten)):
    avg_theta_r_new[i,:] = avg_theta_r_flatten

avg_rad_r_new = np.zeros(np.shape(avg_fa_r))
for i in range(0, len(avg_theta_r_flatten)):
    avg_rad_r_new[:,i] = avg_rad_r_flatten

avg_rad_r_flatten = avg_rad_r_new.flatten()
avg_theta_r_flatten = avg_theta_r_new.flatten()

print(np.shape(avg_rad_r_flatten))
print(np.shape(avg_theta_r_flatten))
print(np.shape(avg_com_x_r_flatten))
print(np.shape(avg_com_y_r_flatten))
print(np.shape(avg_radius_r_flatten))

x_coords = avg_com_x_r_flatten + avg_rad_r_flatten * avg_radius_r_flatten * np.cos(avg_theta_r_flatten*(np.pi/180))
y_coords = avg_com_y_r_flatten + avg_rad_r_flatten * avg_radius_r_flatten * np.sin(avg_theta_r_flatten*(np.pi/180))


# Minimum dimension length (in inches)
scaling =7.0

# X and Y-dimension lengths (in inches)
x_dim = int(scaling + 1.0)
y_dim = int(scaling)

import matplotlib
# Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')
#ax = fig.add_subplot(111)

vals = avg_num_dens_r_flatten

im = plt.tricontourf(x_coords, y_coords, vals, cmap='Greys')#, norm=matplotlib.colors.LogNorm())

sm = plt.cm.ScalarMappable(norm=im.norm, cmap = im.cmap)
sm.set_array([])
clb = fig.colorbar(sm)#ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ax=ax2)
clb.ax.set_title(r'$n$', fontsize=23)
clb.ax.tick_params(labelsize=20)

ax.set_ylim(0, 250)
ax.set_xlim(0, 250)

ax.axes.set_xticks([])
ax.axes.set_yticks([])
ax.axes.set_xticklabels([])
ax.axes.set_yticks([])
ax.set_aspect('equal')

# Create frame images
#ax.set_facecolor('white')
#ax.set_facecolor('#F4F4F4') .  # For website
plt.show()
#plt.savefig



"""
with open(average_faA_path, newline='') as csvfile:
    avg_faA_r = np.array(list(csv.reader(csvfile)))

with open(average_faB_path, newline='') as csvfile:
    avg_faB_r = np.array(list(csv.reader(csvfile)))

with open(average_align_path, newline='') as csvfile:
    avg_align_r = np.array(list(csv.reader(csvfile)))

with open(average_alignA_path, newline='') as csvfile:
    avg_alignA_r = np.array(list(csv.reader(csvfile)))

with open(average_alignB_path, newline='') as csvfile:
    avg_alignB_r = np.array(list(csv.reader(csvfile)))

with open(average_num_path, newline='') as csvfile:
    avg_num_r = np.array(list(csv.reader(csvfile)))

with open(average_numA_path, newline='') as csvfile:
    avg_numA_r = np.array(list(csv.reader(csvfile)))

with open(average_numB_path, newline='') as csvfile:
    avg_numB_r = np.array(list(csv.reader(csvfile)))

with open(average_num_dens_path, newline='') as csvfile:
    avg_num_dens_r = np.array(list(csv.reader(csvfile)))

with open(average_num_densA_path, newline='') as csvfile:
    avg_num_densA_r = np.array(list(csv.reader(csvfile)))

with open(average_num_densB_path, newline='') as csvfile:
    avg_num_densB_r = np.array(list(csv.reader(csvfile)))
"""

headers = df.columns.tolist()

num_lines = len(df['tauB'].values)
num_time_steps = len(np.unique(df['tauB'].values))
num_data = int(num_lines/num_time_steps)

time_sum = np.zeros(num_data)
rad_sum = np.zeros(num_data)
theta_sum = np.zeros(num_data)

fa_sum = np.zeros(num_data)
faA_sum = np.zeros(num_data)
faB_sum = np.zeros(num_data)

align_sum = np.zeros(num_data)
alignA_sum = np.zeros(num_data)
alignB_sum = np.zeros(num_data)

num_sum = np.zeros(num_data)
numA_sum = np.zeros(num_data)
numB_sum = np.zeros(num_data)

num_dens_sum = np.zeros(num_data)
num_densA_sum = np.zeros(num_data)
num_densB_sum = np.zeros(num_data)

radius_sum = np.zeros(num_data)
com_x_sum = np.zeros(num_data)
com_y_sum = np.zeros(num_data)
count = 0

print(headers)
for i in range(0, num_time_steps):
    if df['tauB'].values[start_val]>=150:
        time_sum += (df['tauB'].values[start_val:num_data+start_val])
        rad_sum += (df['rad'].values[start_val:num_data+start_val])
        theta_sum += (df['theta'].values[start_val:num_data+start_val])

        #temp = (df['fa_all'].values[start_val:num_data+start_val]*avg_fa_r_flatten**2)
        #test_id = np.where(avg_fa_r_flatten>0.5)[0]
        #print(len(temp))
        #print(len(avg_fa_r))
        #print(len(test_id))
        #print(len(temp[test_id]))
        #print(len(avg_fa_r_flatten[test_id]))
        fa_sum += (df['fa_all'].values[start_val:num_data+start_val])#*avg_fa_r_flatten**2)**0.5+*avg_fa_r_flatten#temp[test_id] / avg_fa_r_flatten[test_id]**2
        faA_sum += (df['fa_A'].values[start_val:num_data+start_val])
        faB_sum += (df['fa_B'].values[start_val:num_data+start_val])

        align_sum += (df['align_all'].values[start_val:num_data+start_val])
        alignA_sum += (df['align_A'].values[start_val:num_data+start_val])
        alignB_sum += (df['align_B'].values[start_val:num_data+start_val])

        num_sum += (df['num_all'].values[start_val:num_data+start_val])
        numA_sum += (df['num_A'].values[start_val:num_data+start_val])
        numB_sum += (df['num_B'].values[start_val:num_data+start_val])

        num_dens_sum += (df['num_dens_all'].values[start_val:num_data+start_val]*avg_num_dens_r_flatten**2)**0.5#+avg_num_dens_r_flatten#*avg_num_dens_r_flatten**2)
        num_densA_sum += (df['num_dens_A'].values[start_val:num_data+start_val])#*avg_num_densA_r_flatten**2)
        num_densB_sum += (df['num_dens_B'].values[start_val:num_data+start_val])#*avg_num_densB_r_flatten**2)

        radius_sum += (df['radius'].values[start_val:num_data+start_val])
        com_x_sum += (df['com_x'].values[start_val:num_data+start_val])
        com_y_sum += (df['com_y'].values[start_val:num_data+start_val])
        count += 1
    start_val += num_data

time_time_avg = time_sum / count
rad_time_avg = rad_sum / count
theta_time_avg = theta_sum / count

fa_time_avg = fa_sum / count
faA_time_avg = faA_sum / count
faB_time_avg = faB_sum / count

align_time_avg = align_sum / count
alignA_time_avg = alignA_sum / count
alignB_time_avg = alignB_sum / count

num_time_avg = num_sum / count
numA_time_avg = numA_sum / count
numB_time_avg = numB_sum / count

num_dens_time_avg = num_dens_sum / count
num_densA_time_avg = num_densA_sum / count
num_densB_time_avg = num_densB_sum / count

radius_time_avg = radius_sum / count
com_x_time_avg = com_x_sum / count
com_y_time_avg = com_y_sum / count


# X and Y-dimension lengths (in inches)
x_dim = int(scaling + 1.0)
y_dim = int(scaling)

import matplotlib
# Generate figure of dimensions proportional to simulation box size (with added x-length for color bar)
fig, ax = plt.subplots(figsize=(x_dim,y_dim), facecolor='white')
#ax = fig.add_subplot(111)

vals = num_dens_time_avg.flatten()

im = plt.tricontourf(x_coords, y_coords, vals, cmap='Greys')#, norm=matplotlib.colors.LogNorm())

sm = plt.cm.ScalarMappable(norm=im.norm, cmap = im.cmap)
sm.set_array([])
clb = fig.colorbar(sm)#ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ax=ax2)
clb.ax.set_title(r'$n$', fontsize=23)
clb.ax.tick_params(labelsize=20)

ax.set_ylim(0, 250)
ax.set_xlim(0, 250)

ax.axes.set_xticks([])
ax.axes.set_yticks([])
ax.axes.set_xticklabels([])
ax.axes.set_yticks([])
ax.set_aspect('equal')

# Create frame images
#ax.set_facecolor('white')
#ax.set_facecolor('#F4F4F4') .  # For website
plt.show()
#plt.savefig

stop


unique_rad = np.unique(rad_time_avg)
unique_theta = np.unique(theta_time_avg)

time_time_theta_avg = np.array([])
rad_time_theta_avg = np.array([])
theta_time_theta_avg = np.array([])

fa_time_theta_avg = np.array([])
faA_time_theta_avg = np.array([])
faB_time_theta_avg = np.array([])

align_time_theta_avg = np.array([])
alignA_time_theta_avg = np.array([])
alignB_time_theta_avg = np.array([])

num_time_theta_avg = np.array([])
numA_time_theta_avg = np.array([])
numB_time_theta_avg = np.array([])

num_dens_time_theta_avg = np.array([])
num_densA_time_theta_avg = np.array([])
num_densB_time_theta_avg = np.array([])

radius_time_theta_avg = np.array([])
com_x_time_theta_avg = np.array([])
com_y_time_theta_avg = np.array([])


for i in range(0, len(unique_rad)):
    if (unique_rad[i]>=0.8) & (unique_rad[i]<=1.0):
        temp_id = np.where(rad_time_avg == unique_rad[i])

        time_time_theta_avg = np.append(time_time_theta_avg, np.mean(time_time_avg[temp_id]))
        rad_time_theta_avg = np.append(rad_time_theta_avg, np.mean(rad_time_avg[temp_id]))
        theta_time_theta_avg = np.append(theta_time_theta_avg, np.mean(theta_time_avg[temp_id]))

        fa_time_theta_avg = np.append(fa_time_theta_avg, np.mean(fa_time_avg[temp_id]))
        faA_time_theta_avg = np.append(faA_time_theta_avg, np.mean(faA_time_avg[temp_id]))
        faB_time_theta_avg = np.append(faB_time_theta_avg, np.mean(faB_time_avg[temp_id]))

        align_time_theta_avg = np.append(align_time_theta_avg, np.mean(align_time_avg[temp_id]))
        alignA_time_theta_avg = np.append(alignA_time_theta_avg, np.mean(alignA_time_avg[temp_id]))
        alignB_time_theta_avg = np.append(alignB_time_theta_avg, np.mean(alignB_time_avg[temp_id]))

        num_time_theta_avg = np.append(num_time_theta_avg, np.mean(num_time_avg[temp_id]))
        numA_time_theta_avg = np.append(num_time_theta_avg, np.mean(numA_time_avg[temp_id]))
        numB_time_theta_avg = np.append(num_time_theta_avg, np.mean(numB_time_avg[temp_id]))

        num_dens_time_theta_avg = np.append(num_dens_time_theta_avg, np.mean(num_dens_time_avg[temp_id]))
        num_densA_time_theta_avg = np.append(num_densA_time_theta_avg, np.mean(num_densA_time_avg[temp_id]))
        num_densB_time_theta_avg = np.append(num_densB_time_theta_avg, np.mean(num_densB_time_avg[temp_id]))

        radius_time_theta_avg = np.append(radius_time_theta_avg, np.mean(radius_time_avg[temp_id]))
        com_x_time_theta_avg = np.append(com_x_time_theta_avg, np.mean(com_x_time_avg[temp_id]))
        com_y_time_theta_avg = np.append(com_y_time_theta_avg, np.mean(com_y_time_avg[temp_id]))

plt.plot(rad_time_theta_avg, fa_time_theta_avg)
plt.ylabel('active force heterogeneity')
plt.xlabel('distance from CoM')
plt.show()

plt.plot(rad_time_theta_avg, num_dens_time_theta_avg, c='black')
plt.plot(rad_time_theta_avg, num_densA_time_theta_avg, c='blue')
plt.plot(rad_time_theta_avg, num_densB_time_theta_avg, c='red')
plt.ylabel('num dens heterogeneity')
plt.xlabel('distance from CoM')
plt.show()

time_time_rad_avg = np.zeros(len(unique_theta))
rad_time_rad_avg = np.zeros(len(unique_theta))
theta_time_rad_avg = np.zeros(len(unique_theta))

fa_time_rad_avg = np.zeros(len(unique_theta))
faA_time_rad_avg = np.zeros(len(unique_theta))
faB_time_rad_avg = np.zeros(len(unique_theta))

align_time_rad_avg = np.zeros(len(unique_theta))
alignA_time_rad_avg = np.zeros(len(unique_theta))
alignB_time_rad_avg = np.zeros(len(unique_theta))

num_time_rad_avg = np.zeros(len(unique_theta))
numA_time_rad_avg = np.zeros(len(unique_theta))
numB_time_rad_avg = np.zeros(len(unique_theta))

num_dens_time_rad_avg = np.zeros(len(unique_theta))
num_densA_time_rad_avg = np.zeros(len(unique_theta))
num_densB_time_rad_avg = np.zeros(len(unique_theta))

radius_time_rad_avg = np.zeros(len(unique_theta))
com_x_time_rad_avg = np.zeros(len(unique_theta))
com_y_time_rad_avg = np.zeros(len(unique_theta))


for i in range(0, len(unique_theta)):
    temp_id = np.where((theta_time_avg == unique_theta[i]) & (rad_time_avg >=0.8) & (rad_time_avg<=1.0))[0]

    time_time_rad_avg[i] = np.mean(time_time_avg[temp_id])
    rad_time_rad_avg[i] = np.mean(rad_time_avg[temp_id])
    theta_time_rad_avg[i] = np.mean(theta_time_avg[temp_id])

    fa_time_rad_avg[i] = np.mean(fa_time_avg[temp_id])
    faA_time_rad_avg[i] = np.mean(faA_time_avg[temp_id])
    faB_time_rad_avg[i] = np.mean(faB_time_avg[temp_id])

    align_time_rad_avg[i] = np.mean(align_time_avg[temp_id])
    alignA_time_rad_avg[i] = np.mean(alignA_time_avg[temp_id])
    alignB_time_rad_avg[i] = np.mean(alignB_time_avg[temp_id])

    num_time_rad_avg[i] = np.mean(num_time_avg[temp_id])
    numA_time_rad_avg[i] = np.mean(numA_time_avg[temp_id])
    numB_time_rad_avg[i] = np.mean(numB_time_avg[temp_id])

    num_dens_time_rad_avg[i] = np.mean(num_dens_time_avg[temp_id])
    num_densA_time_rad_avg[i] = np.mean(num_densA_time_avg[temp_id])
    num_densB_time_rad_avg[i] = np.mean(num_densB_time_avg[temp_id])

    radius_time_rad_avg[i] = np.mean(radius_time_avg[temp_id])
    com_x_time_rad_avg[i] = np.mean(com_x_time_avg[temp_id])
    com_y_time_rad_avg[i] = np.mean(com_y_time_avg[temp_id])

plt.plot(theta_time_rad_avg, fa_time_rad_avg)
plt.ylabel('active force heterogeneity')
plt.xlabel('Angle around CoM')
plt.show()

plt.plot(theta_time_rad_avg, num_dens_time_rad_avg, c='black')
plt.plot(theta_time_rad_avg, num_densA_time_rad_avg, c='blue')
plt.plot(theta_time_rad_avg, num_densB_time_rad_avg, c='red')
plt.ylabel('num dens heterogeneity')
plt.xlabel('distance from CoM')
plt.show()
stop

stop
