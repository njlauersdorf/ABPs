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
import csv 


current_files = os.listdir()

current_path = os.getcwd()

outPath = current_path + '/output_files/'

# Import modules
import pandas as pd
import numpy as np

for i in range(0, len(current_files)):

    if current_files[i][0:6] == 'dif_av':

        

        current_file = current_files[i]
        
        inFile = current_file[28:-4]
        #inFile = current_file[24:-4]

        average_theta_path = current_path + '/averages/integrated_radial_avgs_theta' + inFile + '.csv'

        average_fa_dens_path = current_path + '/averages/integrated_radial_avgs_fa_dens' + inFile + '.csv'
        average_faA_dens_path = current_path + '/averages/integrated_radial_avgs_faA_dens' + inFile + '.csv'
        average_faB_dens_path = current_path + '/averages/integrated_radial_avgs_faB_dens' + inFile + '.csv'

        average_fa_sum_path = current_path + '/averages/integrated_radial_avgs_fa_sum' + inFile + '.csv'
        average_faA_sum_path = current_path + '/averages/integrated_radial_avgs_faA_sum' + inFile + '.csv'
        average_faB_sum_path = current_path + '/averages/integrated_radial_avgs_faB_sum' + inFile + '.csv'

        average_fa_avg_path = current_path + '/averages/integrated_radial_avgs_fa_avg' + inFile + '.csv'
        average_faA_avg_path = current_path + '/averages/integrated_radial_avgs_faA_avg' + inFile + '.csv'
        average_faB_avg_path = current_path + '/averages/integrated_radial_avgs_faB_avg' + inFile + '.csv'

        average_fa_avg_real_path = current_path + '/averages/integrated_radial_avgs_fa_avg_real' + inFile + '.csv'

        average_align_path = current_path + '/averages/integrated_radial_avgs_align' + inFile + '.csv'
        average_alignA_path = current_path + '/averages/integrated_radial_avgs_alignA' + inFile + '.csv'
        average_alignB_path = current_path + '/averages/integrated_radial_avgs_alignB' + inFile + '.csv'

        average_num_dens_path = current_path + '/averages/integrated_radial_avgs_num_dens' + inFile + '.csv'
        average_num_densA_path = current_path + '/averages/integrated_radial_avgs_num_densA' + inFile + '.csv'
        average_num_densB_path = current_path + '/averages/integrated_radial_avgs_num_densB' + inFile + '.csv'

        average_xA_path = current_path + '/averages/integrated_radial_avgs_part_fracA' + inFile + '.csv'
        average_xB_path = current_path + '/averages/integrated_radial_avgs_part_fracB' + inFile + '.csv'

        file_path = current_path + '/' + current_file

        #with pd.read_csv(file_path, sep='\s+', header=0) as df:
        df = pd.read_csv(file_path, sep='\s+', header=0)
        headers = df.columns.tolist()

        num_lines = len(df['tauB'].values)
        num_time_steps = len(np.unique(df['tauB'].values))
        test_id_new = np.where(np.unique(df['tauB'].values)>=150)[0]
        if len(test_id_new)>20:
            num_data = int(num_lines/num_time_steps)

            #Read input file
            start_val = 0
            

            averages_file =  open(average_theta_path, newline='')
            avg_theta_r = np.array(list(csv.reader(averages_file)))

            avg_theta_r_flatten = (avg_theta_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_fa_dens_path, newline='')
            avg_fa_dens_r = np.array(list(csv.reader(averages_file)))

            avg_fa_dens_r_flatten = (avg_fa_dens_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faA_dens_path, newline='')
            avg_faA_dens_r = np.array(list(csv.reader(averages_file)))

            avg_faA_dens_r_flatten = (avg_faA_dens_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faB_dens_path, newline='')
            avg_faB_dens_r = np.array(list(csv.reader(averages_file)))

            avg_faB_dens_r_flatten = (avg_faB_dens_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_fa_sum_path, newline='')
            avg_fa_sum_r = np.array(list(csv.reader(averages_file)))

            avg_fa_sum_r_flatten = (avg_fa_sum_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faA_sum_path, newline='')
            avg_faA_sum_r = np.array(list(csv.reader(averages_file)))

            avg_faA_sum_r_flatten = (avg_faA_sum_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faB_sum_path, newline='')
            avg_faB_sum_r = np.array(list(csv.reader(averages_file)))

            avg_faB_sum_r_flatten = (avg_faB_sum_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_fa_avg_path, newline='')
            avg_fa_avg_r = np.array(list(csv.reader(averages_file)))

            avg_fa_avg_r_flatten = (avg_fa_avg_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faA_avg_path, newline='')
            avg_faA_avg_r = np.array(list(csv.reader(averages_file)))

            avg_faA_avg_r_flatten = (avg_faA_avg_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_faB_avg_path, newline='')
            avg_faB_avg_r = np.array(list(csv.reader(averages_file)))

            avg_faB_avg_r_flatten = (avg_faB_avg_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_fa_avg_real_path, newline='')
            avg_fa_avg_real_r = np.array(list(csv.reader(averages_file)))

            avg_fa_avg_real_r_flatten = (avg_fa_avg_real_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_num_dens_path, newline='')
            avg_num_dens_r = np.array(list(csv.reader(averages_file)))

            avg_num_dens_r_flatten = (avg_num_dens_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_num_densA_path, newline='')
            avg_num_densA_r = np.array(list(csv.reader(averages_file)))

            avg_num_densA_r_flatten = (avg_num_densA_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_num_densB_path, newline='')
            avg_num_densB_r = np.array(list(csv.reader(averages_file)))

            avg_num_densB_r_flatten = (avg_num_densB_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_align_path, newline='')

            avg_align_r = np.array(list(csv.reader(averages_file)))

            avg_align_r_flatten = (avg_align_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_alignA_path, newline='')
            avg_alignA_r = np.array(list(csv.reader(averages_file)))

            avg_alignA_r_flatten = (avg_alignA_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_alignB_path, newline='')
            avg_alignB_r = np.array(list(csv.reader(averages_file)))

            avg_alignB_r_flatten = (avg_alignB_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_xA_path, newline='')
            avg_xA_r = np.array(list(csv.reader(averages_file)))

            avg_xA_avg_r_flatten = (avg_xA_r.flatten()).astype(np.float) 

            averages_file.close()

            averages_file =  open(average_xB_path, newline='')
            avg_xB_r = np.array(list(csv.reader(averages_file)))

            avg_xB_avg_r_flatten = (avg_xB_r.flatten()).astype(np.float) 

            averages_file.close()

            avg_theta_r_new = np.zeros(np.shape(avg_fa_dens_r))

            time_sum = np.zeros(num_data)
            rad_sum = np.zeros(num_data)
            theta_sum = np.zeros(num_data)

            fa_dens_sum = np.zeros(num_data)
            faA_dens_sum = np.zeros(num_data)
            faB_dens_sum = np.zeros(num_data)

            fa_avg_real_sum = np.zeros(num_data)

            fa_avg_sum = np.zeros(num_data)
            faA_avg_sum = np.zeros(num_data)
            faB_avg_sum = np.zeros(num_data)

            fa_sum_sum = np.zeros(num_data)
            faA_sum_sum = np.zeros(num_data)
            faB_sum_sum = np.zeros(num_data)

            xA_avg_sum = np.zeros(num_data)
            xB_avg_sum = np.zeros(num_data)

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

            for j in range(0, num_time_steps):
                if df['tauB'].values[start_val]>=150:
                    theta_sum += (df['theta'].values[start_val:num_data+start_val])

                    fa_dens_sum += (df['fa_dens_all'].values[start_val:num_data+start_val])**2#*avg_fa_r_flatten**2)**0.5+*avg_fa_r_flatten#temp[test_id] / avg_fa_r_flatten[test_id]**2
                    faA_dens_sum += (df['fa_dens_A'].values[start_val:num_data+start_val])**2
                    faB_dens_sum += (df['fa_dens_B'].values[start_val:num_data+start_val])**2

                    fa_avg_real_sum += (df['fa_avg_real_all'].values[start_val:num_data+start_val])**2#*avg_fa_r_flatten**2)**0.5+*avg_fa_r_flatten#temp[test_id] / avg_fa_r_flatten[test_id]**2

                    fa_avg_sum += (df['fa_avg_all'].values[start_val:num_data+start_val])**2#*avg_fa_r_flatten**2)**0.5+*avg_fa_r_flatten#temp[test_id] / avg_fa_r_flatten[test_id]**2
                    faA_avg_sum += (df['fa_avg_A'].values[start_val:num_data+start_val])**2
                    faB_avg_sum += (df['fa_avg_B'].values[start_val:num_data+start_val])**2

                    fa_sum_sum += (df['fa_sum_all'].values[start_val:num_data+start_val])**2#*avg_fa_r_flatten**2)**0.5+*avg_fa_r_flatten#temp[test_id] / avg_fa_r_flatten[test_id]**2
                    faA_sum_sum += (df['fa_sum_A'].values[start_val:num_data+start_val])**2
                    faB_sum_sum += (df['fa_sum_B'].values[start_val:num_data+start_val])**2

                    xA_avg_sum += (df['fa_sum_A'].values[start_val:num_data+start_val])**2
                    xB_avg_sum += (df['fa_sum_B'].values[start_val:num_data+start_val])**2

                    align_sum += (df['align_all'].values[start_val:num_data+start_val])**2
                    alignA_sum += (df['align_A'].values[start_val:num_data+start_val])**2
                    alignB_sum += (df['align_B'].values[start_val:num_data+start_val])**2

                    num_dens_sum += (df['num_dens_all'].values[start_val:num_data+start_val])**2#+avg_num_dens_r_flatten#*avg_num_dens_r_flatten**2)
                    num_densA_sum += (df['num_dens_A'].values[start_val:num_data+start_val])**2#*avg_num_densA_r_flatten**2)
                    num_densB_sum += (df['num_dens_B'].values[start_val:num_data+start_val])**2#*avg_num_densB_r_flatten**2)

                    count += 1
                start_val += num_data

            theta_time_avg = theta_sum / count

            fa_dens_time_avg = fa_dens_sum / count
            faA_dens_time_avg = faA_dens_sum / count
            faB_dens_time_avg = faB_dens_sum / count

            fa_avg_time_avg = fa_avg_sum / count
            faA_avg_time_avg = faA_avg_sum / count
            faB_avg_time_avg = faB_avg_sum / count

            fa_sum_time_avg = fa_sum_sum / count
            faA_sum_time_avg = faA_sum_sum / count
            faB_sum_time_avg = faB_sum_sum / count

            fa_avg_real_time_avg = fa_avg_real_sum / count

            align_time_avg = align_sum / count
            alignA_time_avg = alignA_sum / count
            alignB_time_avg = alignB_sum / count

            num_dens_time_avg = num_dens_sum / count
            num_densA_time_avg = num_densA_sum / count
            num_densB_time_avg = num_densB_sum / count

            xA_avg_time_avg = xA_avg_sum / count
            xB_avg_time_avg = xB_avg_sum / count

            fa_dens_time_avg_norm = np.mean(fa_dens_time_avg) / np.mean(avg_fa_dens_r_flatten**2)
            faA_dens_time_avg_norm = np.mean(faA_dens_time_avg) / np.mean(avg_faA_dens_r_flatten**2)
            faB_dens_time_avg_norm = np.mean(faB_dens_time_avg) / np.mean(avg_faB_dens_r_flatten**2)

            fa_avg_time_avg_norm = np.mean(fa_avg_time_avg) / np.mean(avg_fa_avg_r_flatten**2)
            faA_avg_time_avg_norm = np.mean(faA_avg_time_avg) / np.mean(avg_faA_avg_r_flatten**2)
            faB_avg_time_avg_norm = np.mean(faB_avg_time_avg) / np.mean(avg_faB_avg_r_flatten**2)

            fa_sum_time_avg_norm = np.mean(fa_sum_time_avg) / np.mean(avg_fa_sum_r_flatten**2)
            faA_sum_time_avg_norm = np.mean(faA_sum_time_avg) / np.mean(avg_faA_sum_r_flatten**2)
            faB_sum_time_avg_norm = np.mean(faB_sum_time_avg) / np.mean(avg_faB_sum_r_flatten**2)

            fa_avg_real_time_avg_norm = np.mean(fa_avg_real_time_avg) / np.mean(avg_fa_avg_real_r_flatten**2)

            align_time_avg_norm = np.mean(align_time_avg) / np.mean(avg_align_r_flatten**2)
            alignA_time_avg_norm = np.mean(alignA_time_avg) / np.mean(avg_alignA_r_flatten**2)
            alignB_time_avg_norm = np.mean(alignB_time_avg) / np.mean(avg_alignB_r_flatten**2)

            num_dens_time_avg_norm = np.mean(num_dens_time_avg) / np.mean(avg_num_dens_r_flatten**2)
            num_densA_time_avg_norm = np.mean(num_densA_time_avg) / np.mean(avg_num_densA_r_flatten**2)
            num_densB_time_avg_norm = np.mean(num_densB_time_avg) / np.mean(avg_num_densB_r_flatten**2)

            xA_avg_time_avg_norm = np.mean(xA_avg_time_avg) / np.mean(avg_xA_avg_r_flatten**2)
            xB_avg_time_avg_norm = np.mean(xB_avg_time_avg) / np.mean(avg_xB_avg_r_flatten**2)
            """
            fa_dens_time_theta_avg = np.mean(fa_dens_time_avg_norm)
            faA_dens_time_theta_avg = np.mean(faA_dens_time_avg_norm)
            faB_dens_time_theta_avg = np.mean(faB_dens_time_avg_norm)

            fa_avg_real_time_theta_avg = np.mean(fa_avg_real_time_avg_norm)

            fa_avg_time_theta_avg = np.mean(fa_avg_time_avg_norm)
            faA_avg_time_theta_avg = np.mean(faA_avg_time_avg_norm)
            faB_avg_time_theta_avg = np.mean(faB_avg_time_avg_norm)

            align_time_theta_avg = np.mean(align_time_avg_norm)
            alignA_time_theta_avg = np.mean(alignA_time_avg_norm)
            alignB_time_theta_avg = np.mean(alignB_time_avg_norm)

            num_dens_time_theta_avg = np.mean(num_dens_time_avg_norm)
            num_densA_time_theta_avg = np.mean(num_densA_time_avg_norm)
            num_densB_time_theta_avg = np.mean(num_densB_time_avg_norm)
            """

            pa_id = inFile.find('pa')
            pb_id = inFile.find('pb')
            xa_id = inFile.find('xa')

            #averaged_data_arr = {'fa_dens': {'all': fa_dens_time_theta_avg, 'A': faA_dens_time_theta_avg, 'B': faB_dens_time_theta_avg}, 'fa_avg_real': {'all': fa_avg_real_time_theta_avg}, 'fa_avg': {'all': fa_avg_time_theta_avg, 'A': faA_avg_time_theta_avg, 'B': faB_avg_time_theta_avg}, 'num_dens': {'all': num_dens_time_theta_avg, 'A': num_densA_time_theta_avg, 'B': num_densB_time_theta_avg}, 'align': {'all': align_time_theta_avg, 'A': alignA_time_theta_avg, 'B': alignB_time_theta_avg}}
            averaged_data_arr = {'fa_dens': {'all': fa_dens_time_avg_norm, 'A': faA_dens_time_avg_norm, 'B': faB_dens_time_avg_norm}, 'fa_sum': {'all': fa_sum_time_avg_norm, 'A': faA_sum_time_avg_norm, 'B': faB_sum_time_avg_norm}, 'fa_avg_real': {'all': fa_avg_real_time_avg_norm}, 'fa_avg': {'all': fa_avg_time_avg_norm, 'A': faA_avg_time_avg_norm, 'B': faB_avg_time_avg_norm}, 'num_dens': {'all': num_dens_time_avg_norm, 'A': num_densA_time_avg_norm, 'B': num_densB_time_avg_norm}, 'align': {'all': align_time_avg_norm, 'A': alignA_time_avg_norm, 'B': alignB_time_avg_norm}, 'part_frac': {'A': xA_avg_time_avg_norm, 'B': xB_avg_time_avg_norm}}
            #averaged_data_arr_new = {'fa_dens': {'all': np.mean(avg_fa_dens_r_flatten**2), 'A': np.mean(avg_faA_dens_r_flatten**2), 'B': np.mean(avg_faB_dens_r_flatten**2)}, 'fa_avg_real': {'all': np.mean(avg_fa_avg_real_r_flatten**2)}, 'fa_avg': {'all': np.mean(avg_fa_avg_r_flatten**2), 'A': np.mean(avg_faA_avg_r_flatten**2), 'B': np.mean(avg_faB_avg_r_flatten**2)}, 'num_dens': {'all': np.mean(avg_num_dens_r_flatten**2), 'A': np.mean(avg_num_densA_r_flatten**2), 'B': np.mean(avg_num_densB_r_flatten**2)}, 'align': {'all': np.mean(avg_align_r_flatten**2), 'A': np.mean(avg_alignA_r_flatten**2), 'B': np.mean(avg_alignB_r_flatten**2)}}

            def write_to_txt(input_dict, outPath):

                #Output values for radial measurements from CoM
                headers = ['pa', 'pb']
                data = [inFile[pa_id+2:pb_id-1], inFile[pb_id+2:xa_id-1]]

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

            write_to_txt(averaged_data_arr, outPath + 'compiled_heterogeneity_dif_av.txt')
            #write_to_txt(averaged_data_arr_new, outPath + 'compiled_heterogeneity_final.txt')
