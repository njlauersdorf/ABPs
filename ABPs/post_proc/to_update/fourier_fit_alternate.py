#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:11:55 2021

@author: nicklauersdorf

Purpose: Fit cluster radius as a function of theta to a Fourier series. Quantifies 
magnitude of surface fluctuations and shape stability of the dense phase cluster. 

Method: Uses alternate method (symfit) for verification of primary method
(scipy.optimize.curve_fit) for fitting simulation data. Symfit runs for 
significantly longer than scipy.optimize.curve_fit and provides comparable 
results; therefore, this is used to confirm accuracy.

Inputs:
    .txt file including:
        Time step: Brownian time units
        Cluster size: Particles in dense phase
        Cluster radius: simulation time units
        Theta: Angle from center of mass (degrees)
        
        
Outputs
    .txt file(s):
        Time_coeff:
            Time step: Brownian time units
            Fourier mode: integer Fourier modes (n)
            Fourier coefficient: float Fourier coefficients from fit of R(theta) at each time step
        
        Avg_coeff:
            Fourier mode: integer Fourier modes(n)
            Fourier coefficient: Time-averaged Fourier coefficients from each time step
        
"""


# Import modules
import sys
import numpy as np
import pandas as pd

from symfit import parameters, variables, sin, cos, Fit


# Define Fourier series for fit
def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * (x-f)) + bi * sin(i * (x-f))
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

#Define input File
inFile = str(sys.argv[1])

# Save input file data to list
all_pres = []

# When running on Longleaf
df = pd.read_csv('/proj/dklotsalab/users/ABPs/monodisperse_soft/theta_V_radius3/'+inFile, sep='\s+', header=0)

# When running locally
df = pd.read_csv('/Volumes/External/whingdingdilly-master/ipython/clusters_soft/theta_V_radius_test/'+inFile, sep='\s+', header=0)

all_pres.append(df)

# Ger rid of NaN in favor of 0
all_pres[0].fillna(0, inplace=True)

#Define number of Fourier modes [n_len-1 Fourier modes]
n_len=21

#Define parameters for fit using Symfit
x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=n_len)}

#If running on cluster
outPath2='/proj/dklotsalab/users/ABPs/monodisperse_soft/fourier_coeffs_final/'

#If running locally
#outPath2='/Volumes/External/whingdingdilly-master/ipython/clusters_soft/fourier_coeff_clust/'

#Angular step size for Fourier series fit
theta_step=7.5

#Number of steady state frames found in loop
ss_num=0

# Output file base name
outF = inFile[:-4]

#Time-averaged Fourier fit information
outTxt_coeff = 'Avg_coeff_' + outF +'.txt' 
g = open(outPath2+outTxt_coeff, 'w+') # write file headings
g.write('n'.center(15) + ' ' +\
        'coeff'.center(15) + '\n')
g.close()

#Fourier fit information every time step
outTxt_coeff2 = 'Time_coeff_' + outF +'.txt' 
g = open(outPath2+outTxt_coeff2, 'w+') # write file headings
g.write('tauB'.center(15) + ' ' +\
        'n'.center(15) + ' ' +\
        'coeff'.center(15) + '\n')
g.close()

#Frame to start at
row_num_init = np.min(np.where(all_pres[0]['tauB']>=10.0)[0])

#Keeps track of current frame in loop
row_num = np.min(np.where(all_pres[0]['tauB']>=10.0)[0])


#Initial time step
time_step=all_pres[0]['tauB'][row_num]

#Previous time step
time_step_prev=-1#all_pres[i]['tauB'][row_num-1]

# empty arrays for output information
ang=np.array([])
rad=np.array([])
popt_sum = np.zeros(n_len)
n_arr = np.linspace(0, n_len-1, n_len)

#While row number of file in length of file
while (row_num<(len(all_pres[0]['tauB'])-1)):
    
    #First time step
    if row_num==row_num_init:
            #If cluster at steady state
            if all_pres[0]['clust_size'][row_num]>(0.95*np.max(all_pres[0]['clust_size'][row_num_init:])):
                
                # Add current angle and radius to array for Fourier series fit
                ang = np.append(ang, all_pres[0]['theta'][row_num])
                rad = np.append(rad, all_pres[0]['radius'][row_num])
                
                #Look at next row number
                row_num+=1
                
                #Time step of new row number
                time_step = all_pres[0]['tauB'][row_num]
                
                #Time step of previous row number
                time_step_prev = all_pres[0]['tauB'][row_num-1]
            
            #If cluster not at steady state
            else:
                #Output nothing and look at next row number
                row_num+=1
                
                #Time step of new row number
                time_step = all_pres[0]['tauB'][row_num]
                
                #Time step of previous row number
                time_step_prev = all_pres[0]['tauB'][row_num-1]                                
    #If not initial row number
    else:
            #If row belongs to previous time step still
            if time_step==time_step_prev:
                
                #If cluster at steady state
                if all_pres[0]['clust_size'][row_num]>(0.95*np.max(all_pres[0]['clust_size'][row_num_init:])):
                    
                    #Add current angle and radius to array for Fourier series fit
                    ang = np.append(ang, all_pres[0]['theta'][row_num])
                    rad = np.append(rad, all_pres[0]['radius'][row_num])
                    
                    #Look at next row number
                    row_num+=1
                    
                    #Time step of new row number
                    time_step = all_pres[0]['tauB'][row_num]
                    
                    #Time step of previous row number
                    time_step_prev = all_pres[0]['tauB'][row_num-1]
                    
                #If cluster not at steady state
                else:
                    #Output nothing and look at next row number
                    row_num+=1
                    
                    #Time step of new row number
                    time_step = all_pres[0]['tauB'][row_num]
                    
                    #Time step of previous row number
                    time_step_prev = all_pres[0]['tauB'][row_num-1]
            
            #If row belongs to a higher time step, we want to fit previous time's data
            else:
                print('time')
                print(time_step)
                
                #If there are angles/radii to fit, fit all angles/radii of previous time step to Fourier series
                if len(ang)>0:
                    
                    #Desired angles around entire circle for fit defined by theta step size
                    des_ang = np.linspace(0, 360, int(360/theta_step)+1)
                    
                    #Empty array to house average radii theta_step/2 around desired angle (des_ang[k])
                    des_rad = np.zeros(int(360/theta_step)+1)
                    
                    #Angular range around des_ang[k] for averaging radii
                    dtheta=theta_step/2
                    
                    #Loop over all desired angles to calculate each average radius
                    for k in range(0, len(des_ang)):
                        
                            #If des_ang[k] not near bounds (0 and 360)
                            if dtheta<=des_ang[k]<=(360-dtheta):
                                new_ids = np.where( ((des_ang[k]-dtheta)<=ang) & (ang<=(des_ang[k]+dtheta))   )[0]
                            
                            #If des_ang[k] within dtheta of 0 bound
                            elif (des_ang[k]-dtheta)<0:
                                new_ids = np.where( ((360+des_ang[k]-dtheta)<=ang) | (ang<=dtheta+des_ang[k])   )[0]
                            
                            #If des_ang[k] within dtheta of 360 bound
                            elif (des_ang[k]-dtheta)>0:
                                new_ids = np.where( ((des_ang[k]-dtheta)<=ang) | (ang<=dtheta+des_ang[k]-360)   )[0]
                            
                            #If found radii within range to average, set des_rad[k] to average radius
                            if len(new_ids)>0:
                                des_rad[k] = np.mean(rad[new_ids])
                            
                            #Otherwise, set des_rad[k] to previous averaged radius
                            else:
                                des_rad[k]=des_rad[k-1]
                    
                    #Fit des_ang and des_rad to Fourier series defined by Fourier_series()
                    fit = Fit(model_dict, x=des_ang*np.pi/180, y=des_rad)
                    fit_result = fit.execute()
                    
                    #Add 1 steady-state frame to averaged count
                    ss_num +=1
                    
                    #Sum each normalized (by n=0 amplitude at that time step) Fourier mode's amplitudes to previous time steps
                    popt_sum[0]+=fit_result.params['a0']/fit_result.params['a0']#(popt[1])#/popt[1])
                    for k in range(1, len(popt_sum)):
                        popt_sum[k]+=(((fit_result.params['a'+str(k)]**2 + fit_result.params['b'+str(k)]**2)**0.5))/fit_result.params['a0']
                    
                    #Output Time step, Fourier mode, and Fourier mode amplitudes at current time step to file
                    g = open(outPath2+outTxt_coeff2, 'a')
                    for m in range(0, n_len-1):
                        g.write('{0:.6f}'.format(time_step_prev).center(15) + ' ')
                        g.write('{0:.0f}'.format(n_arr[m]).center(15) + ' ')
                        g.write('{0:.6f}'.format(popt_sum[m]).center(15) + '\n')
                    g.close()   
                    
                    #Clear arrays for angles and radii to consider in fit for next time step
                    ang=np.array([])
                    rad=np.array([])
                    
                #For new time step, if at steady state, append data for fit
                if all_pres[0]['clust_size'][row_num]>(0.95*np.max(all_pres[0]['clust_size'][row_num_init:])):
                    
                        #Add current angle and radius to array for Fourier series fit
                        ang = np.append(ang, all_pres[0]['theta'][row_num])
                        rad = np.append(rad, all_pres[0]['radius'][row_num])
                        
                        #Look at next row number
                        row_num+=1
                        
                        #Time step of new row number
                        time_step = all_pres[0]['tauB'][row_num]
                        
                        #Time step of previous row number
                        time_step_prev = all_pres[0]['tauB'][row_num-1]
                        
                #For new time step, if not at steady state, go to next row
                else:
                    
                        #Output nothing and look at next row number
                        row_num+=1
                        
                        #Time step of new row number
                        time_step = all_pres[0]['tauB'][row_num]
                        
                        #Time step of previous row number
                        time_step_prev = all_pres[0]['tauB'][row_num-1]   

#Calculate time-averaged Fourier modes                           
popt_avg = popt_sum/ss_num

#Output time-averaged Fourier mode and amplitudes
g = open(outPath2+outTxt_coeff, 'a')
for m in range(0, n_len-1):
    g.write('{0:.0f}'.format(n_arr[m]).center(15) + ' ')
    g.write('{0:.6f}'.format(popt_avg[m]).center(15) + '\n')
g.close()
                 
        
