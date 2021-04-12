#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:20:44 2020

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
import matplotlib.cm as cm

def avgCollisionForce(pe, power=1.):
    '''Computed from the integral of possible angles'''
    peCritical = 40.
    if pe < peCritical:
        pe = 0
    else:
        pe -= peCritical
    magnitude = 6.
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
#     return (magnitude * (pe**power)) / (np.pi)
#     return (pe * (1. + (8./(np.pi**2.))))
    coeff = 2.03
    #coeff= 0.4053
    return (pe * coeff)

def ljPress(r, pe, eps, sigma=1.):
    phiCP = np.pi / (2. * np.sqrt(3.))
    # This is off by a factor of 1.2...
    ljF = avgCollisionForce(pe)
    return (2. *np.sqrt(3) * ljF / r)

# All data is loaded, now compute analytical aspects
r_cut = (2.**(1./6.))

# Get lattice spacing for particle size
def ljForce(r, eps, sigma=1.):
    div = (sigma/r)
    dU = (24. * eps / sigma) * ((2*(div**13)) - (div)**7)
    return dU

# # Lennard-Jones pressure
# def ljPress(r, eps, sigma=1.):
#     phiCP = np.pi / (2. * np.sqrt(3.))
#     div = (sigma/r)
#     dU = (24. * eps / r) * ((2.*(div**12.)) - (div)**6.)
#     # This is just pressure divided by the area of a particle
# #     return (12. * dU / (np.pi * r))
#     return (12. * dU / (np.pi * r * phiCP))

def ljPress(r, pe, eps, sigma=1.):
    phiCP = np.pi / (2. * np.sqrt(3.))
    # This is off by a factor of 1.2...
    ljF = avgCollisionForce(pe)
    return (2. *np.sqrt(3) * ljF / r)
    
def avgCollisionForce(pe, power=1.):
    '''Computed from the integral of possible angles'''
    peCritical = 40.
    if pe < peCritical:
        pe = 0
    else:
        pe -= peCritical
    magnitude = 6.
    # A vector sum of the six nearest neighbors
    magnitude = np.sqrt(28)
#     return (magnitude * (pe**power)) / (np.pi)
#     return (pe * (1. + (8./(np.pi**2.))))
    coeff = 2.03
    #coeff= 0.4053
    return (pe * coeff)

def fStar(pe, epsilon, sigma=1.):
    out = (avgCollisionForce(pe) * sigma) / (24.*epsilon)
    return out
    
def conForRClust(pe, eps):
    out = []
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while ljForce(r, eps) < avgCollisionForce(pe):
            r -= j
        r += j
    out = r
    return out

def nonDimFLJ(r, sigma=1.):
    div = (sigma/r)
    dU = ((2*(div**13)) - (div)**7)
    return dU

def latForFStar(fstar):
    r = 1.112
    skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for j in skip:
        while nonDimFLJ(r) < fstar:
            r -= j
        r += j
    out = r
    return out
    
def latToPhi(latIn):
    '''Read in lattice spacing, output phi'''
    phiCP = np.pi / (2. * np.sqrt(3.))
    return phiCP / (latIn**2)

# From area fraction, get lattice spacing
def phiToLat(phiIn):
    '''Read in phi, output the lattice spacing'''
    phiCP = np.pi / (2. * np.sqrt(3.))
    latCP = 1.
    return np.sqrt(phiCP / phiIn)
    
def compPhiG(pe, a, kap=4.05, sig=1.):
    num = 3. * (np.pi**2) * kap * sig
    den = 4. * pe * a
    return num / den
    
def clustFrac(phi, phiG, a, sig=1.):
    phiL = latToPhi(a)
    ApL = np.pi * (sig**2) / 4.
    Ap = np.pi * (sig**2) / 4.
    num = (phiL*phiG) - (phiL*phi)
    den = ((ApL/Ap)*phi*phiG) - (phi*phiL)
    ans = num / den
    return ans

def radCurve(area):
    # From area of circle get curvature
    return np.sqrt(area/np.pi)

def radCirc(circ):
    return circ / (2. * np.pi)


f = open('/Volumes/External/whingdingdilly-master/ipython/clusters_soft/alignment_correct/alignment_totals.txt', 'r')

lines=f.readlines()
result=[]
gamma=np.array([])
pe=np.array([])
phi=np.array([])
eps=np.array([])
lat=np.array([])


for i in range(0,len(lines)):
    if i>0:
        if float(lines[i].split()[4]) >0.0:
            pe=np.append(pe, float(lines[i].split()[0]))
            phi=np.append(phi, float(lines[i].split()[1]))
            eps=np.append(eps, float(lines[i].split()[2]))
            lat=np.append(lat, float(lines[i].split()[3]))
            gamma= np.append(gamma, float(lines[i].split()[4]))
            
mkEps = ['<', 'd', '*', 's', 'o']              
mk='s'
epsRange = [1., 0.1, 0.01, 0.001, 0.0001]
constPes = np.arange(100, 550, 50) 
phicol = [45.0, 55.0, 65.0]
phiVal = [0.45, 0.55, 0.65]
fig = plt.figure(figsize=(10, 7))
med = 1.5
msz = 80.
myEps = [1., 0.1, 0.01, 0.001, 0.0001]
# myEps = [0.0001, 0.001, 0.01, 0.1, 1.]
mkEps = ['<', 'd', '*', 's', 'o']
bsSz = 9.
mkSz = [1., 1., 1.25, 1., 1.]
eps_leg=[]
for i in range(0, len(myEps)):
#     eps_leg.append(Line2D([0], [0], lw=0., marker=mk, markeredgewidth=med,
#                           markeredgecolor=plt.cm.jet((np.log10(myEps[i]) + 4)/ (len(myEps)-1) ),
#                           markerfacecolor=plt.cm.jet((np.log10(myEps[i]) + 4)/ (len(myEps)-1) ),
#                           label=r'$10^{{{}}}$'.format(-i), markersize=msz))
    eps_leg.append(Line2D([0], [0], lw=0., marker='.', markeredgewidth=med,
                          markeredgecolor=plt.cm.jet(float(i)/ (len(myEps)-1) ),
                          markerfacecolor=plt.cm.jet(float(i)/ (len(myEps)-1) ),
                          label=r'$10^{{{}}}$'.format(-i), markersize=(bsSz * mkSz[i])))
def getEpsInd(eps):
    return myEps.index(eps)
# Parent gs
phiVal = [0.45, 0.55, 0.65]
phicol = [45.0, 55.0, 65.0]
alphaPhi=[0.25, 0.5, 0.75]
phiLS = ['--', ':', '-.']
phi_leg = []
avgR=[]
for i in range(0, len(phiVal)):
    phi_leg.append(Line2D([0], [0], lw=0, c='k', marker='.', 
                          markeredgecolor='black', label=phiVal[i], 
                          markerfacecolor='black', markersize=(bsSz * mkSz[i]), alpha=alphaPhi[i]))   
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)

# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0,len(epsRange)):
    for j in range(0,len(constPes)):
        for k in range(0,len(phiVal)):
            thislat = conForRClust(constPes[j], epsRange[i])
            phicp2=math.pi/(2*3**0.5)
            index=np.where((pe==constPes[j]) & (phi==phicol[k]) & (eps==epsRange[i]))[0]
            if (len(index))>0:
                delta=gamma[index][0]
                rcom_theory = np.linspace(0, delta, 10000) 
                integrandtheory=((4*phicp2*constPes[j])/(math.pi**2*thislat**2))*(rcom_theory/delta)*(1-(rcom_theory/delta))
                theorypressure=np.trapz(integrandtheory, x=rcom_theory)
                #plt.plot(rcom_theory, integrandtheory)
                #plt.show()
                plt.scatter(constPes[j], theorypressure, zorder=1,
                    facecolors=plt.cm.jet(float(epsRange.index(epsRange[i]))/(len(epsRange)-1)), edgecolors=plt.cm.jet(float(epsRange.index(epsRange[i]))/(len(epsRange)-1)),
                   s=msz,alpha= ((phicol.index(phicol[k])+1)/float(len(phiVal))),
                   marker=mkEps[getEpsInd(epsRange[i])]) 

plt.ylabel('Interparticle Pressure ($\Pi^P$)')
plt.xlabel('Activity (Pe)')
ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

plt.tight_layout()
plt.show()

first = True
# Get the current path
if first:
    parent = os.getcwd()
os.chdir(parent)

# Grab file names from data folder
dens = os.listdir('../alignment_per')
try:
    data.remove('.DS_Store')
except:
    print(".DS_Store not in directory")
    
    
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
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
# Grab parameters, sort them
pe=np.array([])
phi=np.array([])
eps=np.array([])
integral3=np.array([])
integral_int=np.array([])
from scipy.stats import norm
from astropy import modeling
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

int_leg=[]

int_leg.append(Line2D([0], [0], lw=0., marker='.', markeredgewidth=med,
                          markeredgecolor='black',
                          markerfacecolor='black',
                          label='interface start', markersize=(bsSz * mkSz[0])))
int_leg.append(Line2D([0], [0], lw=0., marker='*', markeredgewidth=med,
                          markeredgecolor='black',
                          markerfacecolor='black',
                          label='interface end', markersize=(bsSz * mkSz[0])))
    
theory_leg=[]

theory_leg.append(Line2D([0], [0], lw=0., marker='.', markeredgewidth=med,
                          markeredgecolor='black',
                          markerfacecolor='black',
                          label='simulation', markersize=(bsSz * mkSz[0])))
theory_leg.append(Line2D([0], [0], lw=0., marker='*', markeredgewidth=med,
                          markeredgecolor='black',
                          markerfacecolor='black',
                          label='theory', markersize=(bsSz * mkSz[0])))

def gaus_comps(x,a,x0,sigma):
    
    return {'gaus': a*exp(-(x-x0)**2/(2*sigma**2)), 'a':a, 'x0':x0, 'sigma': sigma}

def gaus(x,a,x0,sigma):
    
    return a*exp(-(x-x0)**2/(2*sigma**2))
int_begin=np.array([])
clust_rad=np.array([])

int_width=np.array([])
rad_start=np.array([])
int_end=np.array([])
int_max=np.array([])
int_max_loc=np.array([])
gamma=np.array([])
#timeframes=np.array([])
lat=np.array([])
theory_press=np.array([])
fwhm_arr=np.array([])
integral_fit=np.array([])
for j in dens:
    f = open(j, 'r')

    lines=f.readlines()
    result=[]
    
    radius=np.array([])
    num_dens=np.array([])
    align=np.array([])
    

    for i in range(0,len(lines)):
        if i>0:
            if i==1:
                pe_val=float(lines[i].split()[0])
                pe=np.append(pe, pe_val)
                phi_val=float(lines[i].split()[1])
                phi=np.append(phi, phi_val)
                eps_val=float(lines[i].split()[2])
                eps=np.append(eps, eps_val)
                rad_start_val=float(lines[i].split()[4])
                timeframes=float(lines[i].split()[7])
                clust_rad_val=float(lines[i].split()[8])
                lat_val=float(lines[i].split()[9])
                int_begin_val= float(lines[i].split()[10])
                int_end_val= float(lines[i].split()[11])
                int_width_val= float(lines[i].split()[12])
                
                
            radius=np.append(radius, float(lines[i].split()[3]))
                
            num_dens= np.append(num_dens, float(lines[i].split()[5]))
            align=np.append(align, float(lines[i].split()[6]))
    int_width=np.append(int_width, int_width_val/timeframes)
    int_end=np.append(int_end, int_end_val/timeframes)
    int_begin=np.append(int_begin, int_begin_val/timeframes)
    clust_rad=np.append(clust_rad, clust_rad_val/timeframes)
    align=align/timeframes
    num_dens=num_dens/timeframes


    radius_corr=(radius*(int_end_val/timeframes-rad_start_val))+rad_start_val
    rad_areas = []
    rad_areas=np.array([])
    
    num_dens_corr=np.insert(num_dens, 0, num_dens[0])
    align_corr=np.insert(align, 0, align[0])
    radius_corr=np.insert(radius_corr, 0, 0)
    for k in range(0, len(radius_corr)):
        cur_area = (np.pi * (radius_corr[k]**2))*(18/360)
        if k > 0:
            prev_area = np.pi * (radius_corr[k-1]**2)*(18/360)
        else:
            prev_area=0
        rad_areas=np.append(rad_areas, cur_area - prev_area)
    
    integrand_corr=align_corr*num_dens_corr#*pe_val
    max_val=np.max(integrand_corr)
    int_max = np.append(int_max, max_val)
    max_loc_val=radius_corr[np.where(integrand_corr==np.max(integrand_corr))[0]][0]
    max_loc_ind=np.where(integrand_corr==max_val)[0][0]

    int_max_loc=np.append(int_max_loc, max_loc_val)
    #mean,std=norm.fit(integrand_corr)
    #plt.plot(radius_corr, integrand_corr)
    #fitter = modeling.fitting.LevMarLSQFitter()
    #model = modeling.models.Gaussian1D()
    #fitted_model = fitter(model, radius_corr, integrand_corr)
    #plt.plot(radius_corr, fitted_model(radius_corr))
    #plt.show()
    
    mean = sum(radius_corr*integrand_corr)/len(integrand_corr)
    sigma = sum(integrand_corr*(radius_corr-mean)**2)/len(integrand_corr)
    #+0.004*radius_corr

    #popt,pcov = curve_fit(gaus,radius_corr,integrand_corr,p0=[1,mean,sigma])
    #print(gaus_comps(radius_corr, 1, mean, sigma))
    #stop
    #plt.plot(radius_corr,integrand_corr,color='black',label='data')
    #plt.plot(radius_corr,gaus(radius_corr,*popt),color='red',label='fit')
    #plt.show()
    #stop    

    integrand_new=np.zeros(len(radius_corr))
    for e in range(0, len(integrand_new)):
        integrand_new[e]=pe_val*max_val*(math.pi/2)**0.5*(sigma)*math.erf((max_loc_val-radius_corr[e])/(2**0.5*sigma))
    
    theory_press=np.append(theory_press, np.trapz(integrand_new, x=radius_corr))

    sigma=(1/((2*math.pi)**0.5*max_val))*12#*(int_width_val/timeframes)
    t = max_loc_ind
    while integrand_corr[t]>0.5*max_val:
        t+=1
    q = max_loc_ind
    while integrand_corr[q]>0.5*max_val:
        q-=1
    fwhm=radius_corr[t]-radius_corr[q]
    sigma=fwhm/2.355
    fwhm_arr=np.append(fwhm_arr, fwhm)
    #+np.mean(integrand_corr[:int(max_loc_ind/2)]
    fit=(max_val)*np.exp(-(radius_corr-max_loc_val)**2/(2*sigma**2))
    int_begin_ind=find_nearest(radius_corr, int_begin_val/timeframes)
    int_end_ind=find_nearest(radius_corr, int_end_val/timeframes)
    #plt.plot(radius_corr, fit, label='fit')
    phicp = math.pi/(2*3**0.5)
    range_side=np.arange(0, int_width_val/timeframes, int_width_val/timeframes/len(radius_corr[int_begin_ind:int_end_ind+1]))

    #plt.plot(radius_corr, align_corr)
    #plt.plot([radius_corr[int_begin_ind], radius_corr[int_begin_ind]+0.000000000001], [0, np.max(align_corr)], color='black')
    #plt.plot([radius_corr[int_end_ind], radius_corr[int_end_ind]+0.000000000001], [0, np.max(align_corr)], color='black')
    #plt.ylabel(r'$\alpha(r)$')
    #plt.xlabel('r')

    #plt.show()
    #stop
    
    
    #plt.plot(radius_corr, integrand_corr, label='data')
    #plt.plot([radius_corr[int_begin_ind], radius_corr[int_begin_ind]+0.000000000001], [0, np.max(fit)], color='black')
    #plt.plot([radius_corr[int_end_ind], radius_corr[int_end_ind]+0.000000000001], [0, np.max(fit)], color='black')
    #plt.plot(radius_corr[int_begin_ind:int_end_ind+1], ((3.5*4*phicp)/(math.pi**2*lat_val**2))*(range_side/(int_width_val/timeframes))*(1-range_side/(int_width_val/timeframes)))
    #plt.ylabel(r'$\alpha(r)n(r)$')
    #plt.xlabel('r')
    #plt.legend()
    #plt.show()
    #stop
    #plt.savefig('fit_pe'+str(pe_val)+'_phi'+str(phi_val)+'_eps'+str(eps_val)+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    #plt.close()

    
    integral_fit=np.append(integral_fit, np.trapz((fit*pe_val)[int_begin_ind:int_end_ind+1], x=radius_corr[int_begin_ind:int_end_ind+1]))
    integrand_corr=integrand_corr*pe_val
    
    integral3=np.append(integral3, np.trapz(integrand_corr, x=radius_corr))
    integral_rad=np.zeros(len(radius_corr))
    for m in range(0, len(radius_corr)):
        integral_rad[m]=np.trapz(integrand_corr[m:], x=radius_corr[m:])
    
    integral_int=np.append(integral_int, np.trapz(integrand_corr[int_begin_ind:int_end_ind+1], x=radius_corr[int_begin_ind:int_end_ind+1]))
    
    #plt.plot(radius_corr, integrand_corr)
    #plt.plot(radius_corr, (max_val-np.mean(integrand_corr[:int(max_loc_ind/2)]))*exp(-(radius_corr-max_loc_val)**2/(2*sigma**2))+np.mean(integrand_corr[:int(max_loc_ind/2)]))
    #plt.show()    
    #stop
    #plt.plot(radius_corr, integral_rad)
    #plt.plot([radius_corr[int_begin_ind], radius_corr[int_begin_ind]+0.000000000001], [0, np.max(integral3)], color='black')
    #plt.plot([radius_corr[int_end_ind], radius_corr[int_end_ind]+0.000000000001], [0, np.max(integral3)], color='black')
    #plt.ylabel('Interparticle Pressure ($\Pi^P$)')
    #plt.xlabel('r')
    #plt.title('Pe='+str(pe_val)+', Phi='+str(phi_val)+', Eps='+str(eps_val))
    #plt.savefig('radial_dens_pe'+str(pe_val)+'_phi'+str(phi_val)+'_eps'+str(eps_val)+'.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
    #plt.close()
plt.scatter(pe, clust_rad)
plt.show()
stop
#theorypress=np.zeros(len(pe_val))
plt.scatter(pe, integral_fit, marker='.')
plt.show()

fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)
# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], integral_fit[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='*') 
    plt.scatter(pe[i], integral_int[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel(r'Interparticle Pressure ($\Pi^P$)')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)
theorylegend = ax[0].legend(handles=theory_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.37, 0.8],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=1)
ax[0].add_artist(theorylegend)
plt.tight_layout()
plt.savefig('interparticle_pressure_theory.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()

fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)
# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], int_max_loc[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel(r'$r_{peak}$')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

plt.tight_layout()
plt.savefig('peak_location.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()

fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)
# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], fwhm_arr[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel(r'$FWHM$')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

plt.tight_layout()
plt.savefig('fwhm.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()

fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)

# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], int_max[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel(r'$\alpha(r_{peak})n(r_{peak})$')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

plt.tight_layout()
plt.savefig('peak_height.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()
   
fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)

# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], int_begin[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
    plt.scatter(pe[i], int_end[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='*')
plt.ylabel('r')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

intlegend = ax[0].legend(handles=int_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.37, 0.8],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=1)
ax[0].add_artist(intlegend)

plt.tight_layout()
plt.savefig('interface_begin_end.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()

fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)

# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    print((((-1)*np.log10(eps[i])+1)/(len(epsRange)-1)))
    
    plt.scatter(pe[i], integral3[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel('Interparticle Pressure ($\Pi^P$)')
plt.xlabel('Activity (Pe)')

ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)

plt.tight_layout()
plt.savefig('total_interparticle_pressure_entire_cluster.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()
    
    
    
fig=plt.figure(figsize=(10,7))
fsize = 20
pgs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.21, hspace=0.1)

# Axes to hold each plot
ax = []
# Plots
ax.append(fig.add_subplot(pgs[0]))
for i in range(0, len(integral3)):
    plt.scatter(pe[i], integral_int[i], zorder=1,
                    facecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))), edgecolors=plt.cm.jet((((-1)*np.log10(eps[i]))/(len(epsRange)-1))),
                   s=msz,alpha= (((phi[i]-np.min(phi))+10)/30)*(3/4),
                   marker='.') 
plt.ylabel('Interparticle Pressure ($\Pi^P$)')
plt.xlabel('Activity (Pe)')
ax[0].text(0.18, 1.02, r'$\epsilon=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
one_leg = ax[0].legend(handles=eps_leg, loc='center',
             columnspacing=0.1, handletextpad=-0.1,
             bbox_transform=ax[0].transAxes, bbox_to_anchor=[0.6, 1.03],
             fontsize=fsize, frameon=False, ncol=5)
ax[0].add_artist(one_leg)

ax[0].text(0.35, 1.09, r'$\phi=$',
           transform=ax[0].transAxes,
           fontsize=fsize)
philegend = ax[0].legend(handles=phi_leg, loc='lower right', 
              columnspacing=1., handletextpad=0.1,
              bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.02, 1.03],
              fontsize=fsize, frameon=False, handlelength=2.5, ncol=3)
ax[0].add_artist(philegend)
plt.tight_layout()
plt.savefig('interparticle_pressure_from_r_to_rc.png', bbox_inches='tight', pad_inches=0.02, dpi=1000)
plt.close()
stop
alpha= ((phi-np.min(phi))+10)/30







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
                       storeVals[chkStrings.index("ep")],
                       storeVals[chkStrings.index("cluster")])
    indSort(pres, indArr)
    for i in storeVals:
        indSort(i, indArr)
    
# Now that the data is sorted, read it into a dataframe
all_pres = []
os.chdir(parent)
os.chdir('../pressures')
for i in pres:
#     print(i)
    f = open(i, 'r')

    lines=f.readlines()
    result=[]
    gamma=np.array([])
    pe=np.array([])
    phi=np.array([])
    eps=np.array([])
    lat=np.array([])


for i in range(0,len(lines)):
    if i>0:
        if float(lines[i].split()[4]) >0.0:
            pe=np.append(pe, float(lines[i].split()[0]))
            phi=np.append(phi, float(lines[i].split()[1]))
            eps=np.append(eps, float(lines[i].split()[2]))
            lat=np.append(lat, float(lines[i].split()[3]))
            gamma= np.append(gamma, float(lines[i].split()[4]))
    df = pd.read_csv(i, sep='\s+', header=0)
    all_pres.append(df)
os.chdir(parent)

# This is how you access the data at different levels
display(all_pres[-1])
    
    
stop
#g = open('/Volumes/External/whingdingdilly-master/ipython/clusters_soft/alignment_correct/alignment_single_data.txt', 'r')

lines2=g.readlines()
result=[]
pe=np.array([])
phi=np.array([])
eps=np.array([])
radius=np.array([])
num_dens=np.array([])
align=np.array([])
timeframes=np.array([])
clust_rad=np.array([])
lat=np.array([])
int_width=np.array([])


for i in range(0,len(lines2)):
    if i>0:
    #    if float(lines[i].split()[4]) >0.0:
            pe=np.append(pe, float(lines2[i].split()[0]))
            phi=np.append(phi, float(lines2[i].split()[1]))
            eps=np.append(eps, float(lines2[i].split()[2]))
            radius=np.append(radius, float(lines2[i].split()[3]))
            num_dens= np.append(num_dens, float(lines2[i].split()[4]))
            align=np.append(align, float(lines2[i].split()[5]))
            timeframes=np.append(timeframes, float(lines2[i].split()[6]))
            clust_rad=np.append(clust_rad, float(lines2[i].split()[7]))
            lat= np.append(lat, float(lines2[i].split()[8]))
            int_width= np.append(int_width, float(lines2[i].split()[9]))

stop
print(phi)
print(pe)
print(lat)
print(gamma)
print(eps)
a=np.where(phi==45.0)[0]
b=np.where(phi==55.0)[0]
c=np.where(phi==65.0)[0]

fig = plt.figure()
ax = plt.subplot(111)


scatter = ax.scatter(pe[a],gamma[a]/lat[a],marker='1',label=r'$\phi=45$', c=np.log10(eps[a]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[b],gamma[b]/lat[b],marker='.',label=r'$\phi=55$', c=np.log10(eps[b]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[c],gamma[c]/lat[c],marker='*',label=r'$\phi=65$', c=np.log10(eps[c]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)


#scatter = ax.scatter(netpe_array_final,dens_array_final,marker='<',label='Dense Phase', c=xf_array_final, cmap=cm.plasma)


#scatter = ax.scatter(netpe_array_final[clust_large],gas_parts_array_final[clust_large],marker='.',label='gas phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],bulk_parts_array_final[clust_large],marker='v',label='bulk phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],edge_parts_array_final[clust_large],marker='v',label='edge phase', c=eps_array_final[clust_large], cmap=cm.plasma)

colorbar = fig.colorbar(scatter, ax=ax, ticks=[-4, -3, -2, -1, 0])
colorbar.set_label('$log(\epsilon$)', labelpad=-15, y=1.1, rotation=0)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)
ax.set_xlabel('$Pe$')
ax.set_ylabel(r'$\delta/a$')
ax.set_ylim(0,)
#plt.title('Particle Fraction of Total Phase')
#plt.savefig('Gas_Bulk_Edge_ParticleFraction2.png')
#plt.close()


#plt.scatter(pe, gamma/lat)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)


scatter = ax.scatter(pe[a],gamma[a],marker='1',label=r'$\phi=45$', c=np.log10(eps[a]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[b],gamma[b],marker='.',label=r'$\phi=55$', c=np.log10(eps[b]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[c],gamma[c],marker='*',label=r'$\phi=65$', c=np.log10(eps[c]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)


#scatter = ax.scatter(netpe_array_final,dens_array_final,marker='<',label='Dense Phase', c=xf_array_final, cmap=cm.plasma)


#scatter = ax.scatter(netpe_array_final[clust_large],gas_parts_array_final[clust_large],marker='.',label='gas phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],bulk_parts_array_final[clust_large],marker='v',label='bulk phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],edge_parts_array_final[clust_large],marker='v',label='edge phase', c=eps_array_final[clust_large], cmap=cm.plasma)

colorbar = fig.colorbar(scatter, ax=ax, ticks=[-4, -3, -2, -1, 0])
colorbar.set_label('$log(\epsilon$)', labelpad=-15, y=1.1, rotation=0)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)
ax.set_xlabel('$Pe$')
ax.set_ylabel(r'$\delta$')
ax.set_ylim(0,)
#plt.title('Particle Fraction of Total Phase')
#plt.savefig('Gas_Bulk_Edge_ParticleFraction2.png')
#plt.close()


#plt.scatter(pe, gamma/lat)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)


scatter = ax.scatter(pe[a],lat[a],marker='1',label=r'$\phi=45$', c=np.log10(eps[a]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[b],lat[b],marker='.',label=r'$\phi=55$', c=np.log10(eps[b]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)
scatter = ax.scatter(pe[c],lat[c],marker='*',label=r'$\phi=65$', c=np.log10(eps[c]), cmap=cm.viridis_r, vmin=-4.0, vmax=0.0)


#scatter = ax.scatter(netpe_array_final,dens_array_final,marker='<',label='Dense Phase', c=xf_array_final, cmap=cm.plasma)


#scatter = ax.scatter(netpe_array_final[clust_large],gas_parts_array_final[clust_large],marker='.',label='gas phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],bulk_parts_array_final[clust_large],marker='v',label='bulk phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],edge_parts_array_final[clust_large],marker='v',label='edge phase', c=eps_array_final[clust_large], cmap=cm.plasma)

colorbar = fig.colorbar(scatter, ax=ax, ticks=[-4, -3, -2, -1, 0])
colorbar.set_label('$log(\epsilon$)', labelpad=-15, y=1.1, rotation=0)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3)
ax.set_xlabel('$Pe$')
ax.set_ylabel(r'$a$')
ax.set_ylim(0,)
#plt.title('Particle Fraction of Total Phase')
#plt.savefig('Gas_Bulk_Edge_ParticleFraction2.png')
#plt.close()


#plt.scatter(pe, gamma/lat)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)


scatter = ax.scatter(pe,gamma/lat,marker='.',label=r'$\delta/a$', c=np.log(phi), cmap=cm.autumn_r)
#scatter = ax.scatter(netpe_array_final,dens_array_final,marker='<',label='Dense Phase', c=xf_array_final, cmap=cm.plasma)


#scatter = ax.scatter(netpe_array_final[clust_large],gas_parts_array_final[clust_large],marker='.',label='gas phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],bulk_parts_array_final[clust_large],marker='v',label='bulk phase', c=eps_array_final[clust_large], cmap=cm.plasma)
#scatter = ax.scatter(netpe_array_final[clust_large],edge_parts_array_final[clust_large],marker='v',label='edge phase', c=eps_array_final[clust_large], cmap=cm.plasma)

colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('$log(\epsilon$)', labelpad=-15, y=1.1, rotation=0)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.15,
                 box.width, box.height * 0.85])
ax.set_xlabel('$Pe$')
ax.set_ylabel(r'$\delta/a$')
ax.set_ylim(0,)
#plt.title('Particle Fraction of Total Phase')
#plt.savefig('Gas_Bulk_Edge_ParticleFraction2.png')
#plt.close()


#plt.scatter(pe, gamma/lat)
plt.show()

stop

    #result.append(x.split(' ')[1])
#print(a[0])
stop
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