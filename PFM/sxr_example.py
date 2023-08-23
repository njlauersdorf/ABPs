'''
Created on Feb 16, 2017

@author: lauersdorf
'''
from SXR_Model import sxr_x_emission
from time import time
from numpy import array, zeros, cos, squeeze
from sxr_lib import np_array, add_float
import matplotlib.pyplot as plt
import scipy.io
import sys
import os
import datetime


mu_sav = scipy.io.readsav('SAV/mu.sav')
bessy2015_sav = scipy.io.readsav('SAV/bessy2015.sav')
Al_Charge_State_Frac_sav = scipy.io.readsav('SAV/Al_Charge_State_Frac.sav')
Al_PEC_info_sav = scipy.io.readsav('SAV/Al_PEC_info.sav')
Al_PEC_info_NEW_sav = scipy.io.readsav('SAV/Al_PEC_info.sav')
Aluminum_charge_state_frac_sav = scipy.io.readsav('SAV/Aluminum_charge_state_frac.sav')
ions_info_sav = scipy.io.readsav('SAV/ions_info.sav')

ADAS = Al_Charge_State_Frac_sav['adas']
Al_lines = Al_PEC_info_NEW_sav['al_lines']
ion_i = ions_info_sav['ion_i']
Al_lines_old = Al_PEC_info_sav['al_lines']
bessy = bessy2015_sav['bessy']

radius = 0.52 # MST minor radius, m

delta_a = 0.075 # m
delta_h = 0.000 # m

si_thick = 35.0 # microns

# Te profile

te    =   1485.0 # eV  
alpha_te    =8.0           
beta_te  =15.0           
    

# Te island 

del_te    =  0.0 # eV
r0    =0.0 # eV
th0   = 0.0 # norm.
del_r   =  0.0 # norm.
del_th    =  0.0 # rad

# ne profile


dens       =1.13 # 1e19m-3
alpha_ne  =4.2 
beta_ne  =4.0     

# electron density ring

del_dens  =0.05 # x 1e19 m-3
del_dr    = 0.10 # norm
dr    = 0.25 / radius # norm  

# electron density island

del_ne    = [5.0, 4.0, 3.0] # x 1e19 m-3
ne_r0   = [5.0, 4.0, 3.0] # norm.
ne_del_r  = [5.0, 4.0, 3.0] # norm.
ne_th0    =  [5.0, 4.0, 3.0] # rad
ne_del_th =  [5.0, 4.0, 3.0] # rad

# impurity ions to be included in the simulation

zz      = [  6.0,  8.0,  5.0,  7.0,  10.0,   11.0,   12.0,   13.0]#,   18.0  ]
imp_str = ['c+6', 'o+8','b+5','n+7','al+10', 'al+11','al+12','al+13']#, 'ar+18' ]
n_imp = len(imp_str)
n_zz    = len(zz)

# ratios of densities of Al+10, Al+11, Al+12, and Al+13 from ADAS modeling
AlDen = array([4.66244e-30, 3.41797e-25, 4.86619e-21, 1.80713e-17, 2.66453e-14,\
                   1.68429e-11, 5.17369e-09, 8.39320e-07, 7.47659e-05, 0.00342171, 0.0737855, 0.582899,\
                   0.290577, 0.0492406])
    
perc_Al10 = AlDen[10] / AlDen[11]
perc_Al11 = AlDen[11] / AlDen[11]
perc_Al12 = AlDen[12] / AlDen[11]
perc_Al13 = AlDen[13] / AlDen[11]

# core density is based on C density scaled by empirical ratios for O,B,N, and
# based on Al+11 scaled by ADAS for other states of Al
#
# core Ar density is its own fitting parameter

nC0      = 7.5e-3 # x 1e19 m-3 
nAl0     = 3.0e-3 # x 1e19 m-3
#nAr0     = 1.0e-7 # x 1e19 m-3

# impurity ion density profiles

ni_dens = [nC0, nC0*0.9, nC0*0.3, nC0*0.3, nAl0*perc_Al10, nAl0, nAl0*perc_Al12, nAl0*perc_Al13] # x 1e19 m-3
         #[C+6, O+8    , B+5    , N+7    , Al+10         , Al+11, Al+12        , Al+13         , Ar+18]
    
# all profiles have same alpha and beta values

alpha_nC = 15.0
beta_nC = 2.0

alpha_ni = zeros(n_zz) + alpha_nC
beta_ni  = zeros(n_zz) + beta_nC

#impurity ion density rings
# all profiles have ring located at same radius and have same width

rn0    = 0.33   # m
del_rC = 0.0425 # m

ni_dr     = zeros(n_zz) + rn0 / radius    # norm
del_ni_dr = zeros(n_zz) + del_rC / radius # norm

# density ring amplitude is based on C ring amplitude scaled by empirical ratios
# for O,B,N, and based on Al+11 scaled by ADAS results for other charge states of Al
#
# Ar has no ring (amplitude=0)

del_nC = 0.01    # x 1e19 m-3
del_Al = 1.25e-3 # x 1e19 m-3

del_ni_dens = [ del_nC, del_nC*0.75, del_nC*0.3, del_nC*0.3, del_Al*perc_Al10, del_Al, del_Al*perc_Al12, del_Al*perc_Al13, 0.0] 

# impurtiy island 

is_del_ni= 0.0#[0.00,0.0]
is_ni_dr=0.0#[0.00,0.0]
is_dr= 0.0#[0.0,0.0]
ni_th_dr=0.0#[0.0,0.0]
ni_del_th= 0.0#[0.0,0.0]

li = ''

be_thick = array([857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 421.00, 421.00, 421.00, 421.00, 421.00,
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00])
    
bright_p = array([0.455010,     0.422147,     0.379831,     0.327601,     0.266417,     0.198905,\
    0.128902,    0.0604414,   0.00326962,    0.0603420,     0.175870,     0.120760, 0.0576637,\
    0.0120282,    0.0854655,     0.158671,     0.227481,     0.288657, 0.340528,     0.382937,\
    0.191081,     0.125095,    0.0530729,    0.0213637,    0.0939824,     0.161066,     0.220241,\
    0.270641,     0.312562,     0.346947,     0.147935,    0.0803607,   0.00746861,    0.0669813,\
    0.138755,     0.204269, 0.261366,     0.309410,     0.348884,     0.380865,     0.455010,\
    0.422147,     0.379831,     0.327601,     0.266417,     0.198905,     0.128902,    0.0604414,\
    0.00326962,    0.0603420,     0.175870,     0.120760,    0.0576637,    0.0120282, 0.0854655,
    0.158671,     0.227481,     0.288657,     0.382937,     0.191081,     0.125095,    0.0530729,\
    0.0213637,    0.0939824,     0.161066,     0.220241,     0.270641,     0.312562,     0.346947,\
    0.147935,    0.0803607,   0.00746861, 0.0669813,     0.138755,     0.204269,     0.261366,\
    0.309410,     0.348884,     0.380865])
    
bright_phi = array([0.169488,    0.0629166,      6.22716,      6.09748,      5.96037,      5.82060,      5.68349,      5.55381,      2.29328,      2.18671,     0.169488,    0.0629166,      6.22716,      2.95589,      2.81878,      2.67901,      2.54190,      2.41222,
      2.29328,      2.18671,      1.51466,      1.38497,      1.24786,      4.24968,      4.11258,      3.98290,      3.86397,      3.75741,      3.66343,      3.58131,      4.65663,      4.52695,      4.38983,      1.10845,     0.971309,     0.841593,
     0.722619,     0.616017,     0.521998,     0.439842,     0.169488,    0.0629166,      6.22716,      6.09748,      5.96037,      5.82060,      5.68349,      5.55381,      2.29328,      2.18671,     0.169488,    0.0629166,      6.22716,      2.95589,
      2.81878,      2.67901,      2.54190,      2.41222,      2.18671,      1.51466,      1.38497,      1.24786,      4.24968,      4.11258,      3.98290,      3.86397,      3.75741,      3.66343,      3.58131,      4.65663,      4.52695,      4.38983,
      1.10845,     0.971309,     0.841593,     0.722619,     0.616017,     0.521998,     0.439842])
    
bright_alfa = array([
            0.562187, 0.455616, 0.3366750, 0.206992, 0.0698860, 0.0698860,
            0.206992, 0.336675, 0.4556160, 0.562187, 0.5621870, 0.4556160, 
            0.336675, 0.206992, 0.0698860, 0.069886, 0.2069920, 0.3366750, 
            0.455616, 0.562187, 0.3365650, 0.206874, 0.0697632, 0.0700091, 
            0.207111, 0.336785, 0.4557160, 0.562276, 0.6562570, 0.7383810, 
            0.336942, 0.207259, 0.0701416, 0.069652, 0.2067880, 0.3365040, 
            0.455478, 0.562080, 0.6560990, 0.738255, 0.5621870, 0.4556160, 
            0.336675, 0.206992,0.06988600, 0.069886, 0.2069920, 0.3366750,
            0.455616, 0.562187, 0.5621870, 0.455616, 0.3366750, 0.2069920, 
            0.0698860,0.0698860, 0.206992, 0.336675, 0.562187, 
            0.336565, 0.206874,0.0697632, 0.0700091, 0.207111, 0.336785, 
            0.455716, 0.562276, 0.656257, 0.738381, 0.336942, 0.207259, 
            0.0701416, 0.0696520, 0.206788, 0.336504,0.455478, 0.562080, 
            0.656099, 0.738255])
    
meas = array([0.0341914,    0.0588694,     0.676304,      3.60283,      6.32861,      7.83843,      8.61106,      8.85919,      8.90548,      8.55441,      8.12827,      8.64937,      8.80557,      8.48702,      7.57199,      6.41089,      3.18115,     0.569319,
    0.120082,    0.0698749,      6.84370,      7.96648,      8.59326,      8.27841,      7.66085,      6.54788,      4.30654,      1.25152,  -0.00737862,     0.441739,      8.00824,      8.13176,      8.47297,      8.24714,      8.15334,      6.89304,
    5.85517,      3.85869,      1.19441,     0.307292,    0.0834942,     0.244043,      1.57579,      8.85026,      15.0638,      18.1562,      19.8733,      20.7685,      19.8594,      19.3658,      18.9356,      21.0858,      21.1731,      19.7654,
    18.2176,      14.0432,      7.10720,      1.24874,    0.0393101,      15.1707,      17.5717,      18.4612,      18.4284,      17.0555,      14.7991,      9.69892,      3.10073,     0.503314,    -0.451082,      17.2662,      18.8467,      19.4291,
    18.9366,      18.0835,      15.4406,      12.3074,      8.12172,      2.43435,     0.852094])
    
# Initialize the model

time1 = datetime.datetime.now()                
st = sxr_x_emission(te=te, alpha_te=alpha_te, beta_te=beta_te, 
               r0=r0, th0=th0, del_r=del_r, del_th=del_th, del_te=del_te,
               
               dens=dens, alpha_ne=alpha_ne, beta_ne=beta_ne,
               ne_r0=ne_r0, ne_del_r=ne_del_r, ne_th0=ne_th0, ne_del_th=ne_del_th, del_ne=del_ne,
               del_dens=del_dens, del_dr=del_dr, dr=dr,
               
               ni_dens=ni_dens, alpha_ni=alpha_ni, beta_ni=beta_ni, 
               ni_dr=ni_dr, del_ni_dr=del_ni_dr, del_ni_dens=del_ni_dens,
               is_del_ni=is_del_ni, is_ni_dr=is_ni_dr, is_dr=is_dr, ni_th_dr=ni_th_dr, ni_del_th=ni_del_th,
               imp_str=imp_str, n_imp=n_imp,
               
               delta_a=delta_a, delta_h=delta_h,
               radius=radius,
               
                si_thick = np_array(si_thick/cos(bright_alfa)), be_thick = be_thick,
                
                p=bright_p, phi=bright_phi)

line_int = st.sxr_line_integral()
print line_int
#values1 = st.sxr_spectrum([1, 2], [3, 4])['en_int']
#print values1

time2 = datetime.datetime.now()
time3 = time2 - time1
print time3
stop

'''THE FOLLOWING CODE DOESN'T WORK AS INTENDED SINCE VARIABLE NAMES HAVE CHANGED'''
script_dir = os.path.dirname(__file__)
rel_file_path = os.path.join(script_dir, 'SAV')

#EXAMPLE OF LOADING A .SAV FILE FROM THE FOLDER
abs_file_path = os.path.join(rel_file_path, 'Al_Charge_State_Frac.sav')
#Al_Charge_State_Frac = scipy.io.readsav(abs_file_path)
#ADAS = Al_Charge_State_Frac['ADAS']
#print ADAS.dtype.names
#print ADAS['some name in ADAS.dtype.names']


b = add_float(line_int, meas)
yr=array([0, 1.05])*b.max()
plt.plot(line_int, label = 'model', marker = '^', linestyle = '-')
plt.plot(meas, color = 'r', label = 'measurement')
plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc=3, ncol =2, mode = "expand", borderaxespad=0.)
plt.ylabel('W/m^3')
plt.xlabel('diode index')
plt.xlim([0, 80])
print 'close graph to continue...'
plt.show()

#select a line of sight

i_p = 47
st = sxr_x_emission(te=te, alpha_te=alpha_te, beta_te=beta_te, 
               r0=r0, th0=th0, del_r=del_r, del_th=del_th, del_te=del_te,
               
               dens=dens, alpha_ne=alpha_ne, beta_ne=beta_ne,
               ne_r0=ne_r0, ne_del_r=ne_del_r, ne_th0=ne_th0, ne_del_th=ne_del_th, del_ne=del_ne,
               del_dens=del_dens, del_dr=del_dr, dr=dr,
               
               ni_dens=ni_dens, alpha_ni=alpha_ni, beta_ni=beta_ni, 
               ni_dr=ni_dr, del_ni_dr=del_ni_dr, del_ni_dens=del_ni_dens,
               is_del_ni=is_del_ni, is_ni_dr=is_ni_dr, is_dr=is_dr, ni_th_dr=ni_th_dr, ni_del_th=ni_del_th,
               imp_str=imp_str, n_imp=n_imp,
               
               delta_a=delta_a, delta_h=delta_h,
               radius=radius,
               
                si_thick = np_array(35.0/cos(bright_alfa[i_p])), be_thick = be_thick[i_p])
values1 = st.sxr_spectrum([0, 2, 3], [0, 3, 4])['en_int']
print values1

rd = array([ 0.1, 0.5, 0.7, 0.8, 0.9, 0.95]) # RHO, norm.
rd = rd.reshape((6,1))
tze = rd * 0.0 # rad
values = st.sxr_spectrum(tze, rd, arr = '')


y = squeeze(values['sp_brems'][:,0,0]) + squeeze(values['sp_recomb'][:,0,0])
E = st.en / 1.0e3
print values['sp_r_imp'].shape

stop
plt.plot(E, y, linestyle = '-')
plt.plot(E, squeeze(values['sp_brems'][:,0,0]), color ='r')
plt.plot(E, squeeze(values['sp_recomb'][:,0,0]), color = 'c')
plt.plot(E, squeeze(values['sp_r_imp'][0,:,0,0]), color = 'g')
plt.plot(E, squeeze(values['sp_r_imp'][1,:,0,0]), color = 'y')
plt.plot(E, squeeze(values['sp_r_imp'][2,:,0,0]), color = 'm')
plt.xlim([0, 10])
plt.ylabel('SXR spectrum')
plt.xlabel('E (keV)')

print 'close graph to continue...'
plt.show()
    
plt.plot(E, squeeze(values['sp_b_BeSi'][:,0,0]) + squeeze(values['sp_r_BeSi'][:,0,0]))
plt.plot(E, squeeze(values['sp_b_BeSi'][:,0,0]), color = 'r')
plt.plot(E, squeeze(values['sp_r_BeSi'][:,0,0]), color = 'c')
plt.plot(E, squeeze(values['sp_r_i_BeSi'][0,:,0,0]), color = 'g')
plt.plot(E, squeeze(values['sp_r_i_BeSi'][1,:,0,0]), color = 'y')
plt.plot(E, squeeze(values['sp_r_i_BeSi'][2,:,0,0]), color = 'm')

plt.ylabel('SXR spectrum')
plt.xlabel('E (keV)')
plt.title(str(be_thick[i_p]) + ' um')
plt.xlim([0,10]) 
print 'close graph to continue...'
plt.show()
