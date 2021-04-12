#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:50:02 2020

@author: nicklauersdorf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:41:30 2020

@author: nicklauersdorf
"""

import sys
from gsd import hoomd
import freud
import numpy as np
import math
from scipy import interpolate

#from descartes.patch import PolygonPatch
# Run locally
hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#/nas/home/njlauers/hoomd-blue/build/'#Users/nicklauersdorf/hoomd-blue/build/'
#gsdPath='/Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
#imgPath='/pine/scr/n/j/njlauers/scm_tmpdir/phase_comp_new_edge_video/'
#imgPath2='/pine/scr/n/j/njlauers/scm_tmpdir/phase_comp_new_edge_txt/'
r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
#inFile='pa300_pb300_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'cluster_pa400_pb350_phi60_eps0.1_xa0.8_align3_dtau1.0e-06.gsd'
#inFile='pa400_pb500_xa20_ep1.0_phi60_pNum100000.gsd'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
outPath='/pine/scr/n/j/njlauers/scm_tmpdir/alignment/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')
# Inside and outside activity from command line

#Label simulation parameters
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac_orig = float(sys.argv[4])
if parFrac_orig<1.0:
    parFrac=parFrac_orig*100.
else:
    parFrac=parFrac_orig
    
#Check to see if system is monodisperse (mono_true=1) or binary (mono_true=0)
#and if monodisperse activity is either peA (a_activity=1) or peB (a_activity=0)    
if (parFrac == 0.0):
    mono_true = 1
    a_activity=0
    
elif  (parFrac == 100.0):
    mono_ture = 1
    a_activity=1
    
else:
    mono_true = 0
    
eps = float(sys.argv[5])

#Calculate Net activity
peNet=peA*(parFrac/100)+peB*(1-(parFrac/100))

try:
    phi = float(sys.argv[6])
    intPhi = int(phi)
    phi /= 100.
except:
    phi = 0.6
    intPhi = 60

try:
    dtau = float(sys.argv[7])
except:
    dtau = 0.000001

# Set some constants
kT = 1.0                        # temperature
threeEtaPiSigma = 1.0           # drag coefficient
sigma = 1.0                     # particle diameter
D_t = kT / threeEtaPiSigma      # translational diffusion constant
D_r = (3.0 * D_t) / (sigma**2)  # rotational diffusion constant
tauBrown = (sigma**2) / D_t     # brownian time scale (invariant)

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
#Calculate activity-softness dependent variables
lat=getLat(peNet,eps)

tauLJ=computeTauLJ(eps)
dt = dtau * tauLJ                        # timestep size

# Get filenames for various file types

import gsd
from gsd import hoomd
from gsd import pygsd

import freud
from freud import parallel
from freud import box
from freud import density
from freud import cluster
import itertools

import numpy as np
import math
import random
from scipy import stats

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle


matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5
            
def computeDist(x1, y1, x2, y2):
    '''Compute distance between two points'''
    return np.sqrt( ((x2-x1)**2) + ((y2 - y1)**2) )
    
def computeFLJ(r, x1, y1, x2, y2, eps):
    sig = 1.
    f = (24. * eps / r) * ( (2*((sig/r)**12)) - ((sig/r)**6) )
    fx = f * (x2 - x1) / r
    fy = f * (y2 - y1) / r
    return fx, fy

def computeTauPerTstep(epsilon, mindt=0.000001):
    '''Read in epsilon, output tauBrownian per timestep'''
#    if epsilon != 1.:
#        mindt=0.00001
    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

def roundUp(n, decimals=0):
    '''Round up size of bins to account for floating point inaccuracy'''
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
    
def getNBins(length, minSz=(2**(1./6.))):
    "Given box size, return number of bins"
    initGuess = int(length) + 1
    nBins = initGuess
    # This loop only exits on function return
    while True:
        if length / nBins > minSz:
            return nBins
        else:
            nBins -= 1

def findBins(lookN, currentInd, maxInds):
    '''Get the surrounding bin indices'''
    maxInds -= 1
    left = currentInd - lookN
    right = currentInd + lookN
    binsList = []
    for i in range(left, right):
        ind = i
        if i > maxInds:
            ind -= maxInds
        binsList.append(ind)
    return binsList
f = hoomd.open(name=inFile, mode='rb')

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep
                
#outTxt = 'alignment_' + outF + '.txt'
#g = open(outPath+outTxt, 'w+') # write file headings
#g.write('tauB'.center(15) + ' ' +\
#        'clust_size'.center(15) + ' ' +\
#        'min_ang'.center(15) + ' ' +\
#        'max_ang'.center(15) + ' ' +\
#        'radius'.center(15) + ' ' +\
#        'num_dens'.center(15) + ' ' +\
#        'align'.center(15) + '\n')
#g.close()
   
with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0#600                  # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    l_box = box_data[0]
    h_box = l_box / 2.
    radius=np.arange(0,h_box, 1.0*lat)
    typ = snap.particles.typeid
    partNum = len(typ)
    # Set up cluster computation using box
    f_box = box.Box(Lx=l_box, Ly=l_box, is2D=True)

    NBins = getNBins(l_box, r_cut)
    sizeBin = roundUp(((l_box) / NBins), 6)
    #my_clust = cluster.Cluster()
    #c_props = cluster.ClusterProperties()
    # Compute each mesh
    
    time_arr=np.zeros(end)
    size_arr = np.zeros(int(end))
    clust_edge_arr=np.zeros(end)
    
    average_clust_radius=0
    count_of_rads=0
    gas_area_arr = np.zeros(end)
    dense_area_arr = np.zeros(end)
    edge_area_arr = np.zeros(end)
    dense_area_big_arr = np.zeros(end)
    edge_area_big_arr = np.zeros(end)
    count_avg=0
    align_avg=np.zeros(len(radius))
    align_num=0
             
    for j in range(start, int(end)):
        r=np.arange(1,h_box,1)
        j=600
        print('j')
        print(j)
        # Outfile to write data to
        imgbase = add + 'pressure_pa' + str(peA) +\
       '_pb' + str(peB) +\
       '_xa' + str(parFrac) +\
       '_phi' + str(intPhi) +\
       '_ep' + '{0:.3f}'.format(eps)+'_frame'+str(j)
        imgFile = imgbase + '.png'
        snap = t[j]
        # Easier accessors

        pos = snap.particles.position               # position
        
        #print(snap.particles.velocity)
        #print(np.arctan(snap.particles.velocity[:,1], snap.particles.velocity[:,0]))
        pos[:,-1] = 0.0
        ori = snap.particles.orientation 
        #print(ori)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
        #print(ang)
        #print(np.multiply(np.sin(ang/2),(snap.particles.velocity[:,0])/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        #print(np.multiply(np.sin(ang/2),snap.particles.velocity[:,1]/(snap.particles.velocity[:,0]**2+snap.particles.velocity[:,1]**2)**0.5))
        #stop
        typ = snap.particles.typeid                 # type

        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
        
        xy = np.delete(pos, 2, 1)
        tst = snap.configuration.step               # timestep
        tst -= first_tstep                          # normalize by first timestep
        tst *= dtau                                 # convert to Brownian time
        time_arr[j]=tst
        # Compute clusters for this timestep
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))
        
        cl_all=freud.cluster.Cluster()                      #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()         #Define cluster properties
        ids = cl_all.cluster_idx              # get id of each cluster
        clp_all.compute(system_all, ids)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        min_size=int(partNum/3)
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]
        large_clust_ind_all=np.where(clust_size>min_size)
        

        if len(large_clust_ind_all[0])>0:
            rad_bins=np.zeros(len(radius))
            query_points=clp_all.centers[lcID]
            com_tmp_posX = query_points[0] + h_box
            com_tmp_posY = query_points[1] + h_box

            com_x_ind = int(com_tmp_posX / sizeBin)
            com_y_ind = int(com_tmp_posY / sizeBin)
            # Loop through each timestep
            tot_count_gas=np.zeros((end,2))
            tot_count_dense=np.zeros((end,2))
            tot_count_edge=np.zeros((end,2))
            rad_count_gas=np.zeros((end,int(NBins/2)-1,2))
            rad_count_dense=np.zeros((end,int(NBins/2)-1,2))
            rad_count_edge=np.zeros((end,int(NBins/2)-1,2))
            # Get the positions of all particles in LC
            binParts = [[[] for b in range(NBins)] for a in range(NBins)]
            typParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            #posParts=  [[[] for b in range(NBins)] for a in range(NBins)]
            occParts = [[0 for b in range(NBins)] for a in range(NBins)]
            edgeBin = [[0 for b in range(NBins)] for a in range(NBins)]
            
            
            
            #Assigns particle indices and types to bins
            for k in range(0, len(ids)):

                # Convert position to be > 0 to place in list mesh
                tmp_posX = pos[k][0] + h_box
                tmp_posY = pos[k][1] + h_box
                x_ind = int(tmp_posX / sizeBin)
                y_ind = int(tmp_posY / sizeBin)
                # Append all particles to appropriate bin
                binParts[x_ind][y_ind].append(k)
                typParts[x_ind][y_ind].append(typ[k])
                #posParts[x_ind][y_ind].append(pos[k])
                
                if clust_size[ids[k]] >= min_size:
                    occParts[x_ind][y_ind] = 1
            # If sufficient neighbor bins are empty, we have an edge
            PhaseParts=np.zeros(len(pos))
            PhaseParts2=np.zeros(len(pos))
            PhasePartsarea=np.zeros(len(pos))
            gasBins = 0
            bulkBins = 0
            edgeBins=0
            edgeBinsbig = 0
            bulkBinsbig = 0
            testIDs = [[0 for b in range(NBins)] for a in range(NBins)]
            testIDs_area = [[0 for b in range(NBins)] for a in range(NBins)]

            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if len(binParts[ix][iy]) != 0:
                        if clust_size[ids[binParts[ix][iy][0]]] >=min_size:
                            bulkBins += 1
                            testIDs[ix][iy] = 1
                            testIDs_area[ix][iy] = 1
                            continue
                    gasBins += 1
                    testIDs[ix][iy] = 2
                    testIDs_area[ix][iy] = 2
            
            
                        
                        
            count_A_edge=0
            count_B_edge=0
            gas_particle_range=2.0
            gas_r_lim=gas_particle_range*lat
            bulk_particle_range=5.0
            end_loop=0
            steps=0            
            rad_bin=np.zeros(len(radius)-1)
            align_tot = np.zeros((len(occParts), len(occParts)))
            align_tot_x = np.zeros((len(occParts), len(occParts)))
            align_tot_y = np.zeros((len(occParts), len(occParts)))
            align_avg = np.zeros((len(occParts), len(occParts)))
            align_avg_x = np.zeros((len(occParts), len(occParts)))
            align_avg_y = np.zeros((len(occParts), len(occParts)))
            parts_tot = np.zeros((len(occParts), len(occParts)))
            bin_pos_x = np.zeros((len(occParts), len(occParts)))
            bin_pos_y = np.zeros((len(occParts), len(occParts)))
            align_all=np.array([])
            pos_all_x=np.array([])
            pos_all_y=np.array([])
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):

                        if len(binParts[ix][iy])!=0:
                                
                            for h in range(0,len(binParts[ix][iy])):
                                x_pos=pos[binParts[ix][iy]][h][0]+h_box
                                        
                                y_pos=pos[binParts[ix][iy]][h][1]+h_box
                                        
                                difx=x_pos-com_tmp_posX
                                difx_abs = np.abs(difx)
                                if difx_abs>=h_box:
                                    if difx < -h_box:
                                        difx += l_box
                                    else:
                                        difx -= l_box
                                dify=y_pos-com_tmp_posY
                                dify_abs = np.abs(dify)
                                if dify_abs>=h_box:
                                    if dify < -h_box:
                                        dify += l_box
                                    else:
                                        dify -= l_box

                                #pos_new_x=np.append(pos_new_x, x_pos)
                                #pos_new_y=np.append(pos_new_y, y_pos)
                                        
                                difr=(difx**2+dify**2)**0.5
                                px = np.sin(ang[binParts[ix][iy][h]])
                                py = -np.cos(ang[binParts[ix][iy][h]])
                                                
                                r_dot_p = (-difx * px) + (-dify * py)
                                align=r_dot_p/difr
                                
                                align_tot[ix][iy]+=align
                                align_tot_x[ix][iy]+=px
                                align_tot_y[ix][iy]+=py
                                parts_tot[ix][iy]+=1
                
            for ix in range(0, len(occParts)):
                for iy in range(0, len(occParts)):
                    if parts_tot[ix][iy]!=0:
                        align_avg_x[ix][iy]=(align_tot_x[ix][iy]/parts_tot[ix][iy])
                        align_avg_y[ix][iy]=(align_tot_y[ix][iy]/parts_tot[ix][iy])
                        align_avg[ix][iy]=(align_tot[ix][iy]/parts_tot[ix][iy])
                        
                    bin_pos_x[ix][iy]=(ix+0.5)*sizeBin
                    bin_pos_y[ix][iy]=(iy+0.5)*sizeBin
            #align_all=np.array([align_all, align_all])
            #print(np.shape(align_all))
            #plt.contour(pos_all_x, pos_all_y, align_all)
            #plt.show()
           # print('begin')
            #f=interpolate.interp2d(bin_pos_x, bin_pos_y, align_avg, kind='cubic')
            #print('1')
            #x_new=np.arange(0, l_box, l_box/100)
            #print('2')
            znew=f(x_new, x_new)
            #print('3')
            plt.contourf(x_new, x_new, align_avg)
            plt.xlabel('x')
            plt.ylabel('y')
            cbar=plt.colorbar()
            cbar.set_label(r'$\alpha$', rotation=270)
            plt.show()
            stop
            '''
            f=interpolate.interp2d(bin_pos_x, bin_pos_y, align_avg, kind='cubic')
            x_new=np.arange(0, l_box, l_box/200)
            print('1')
            znew=f(x_new, x_new)
            print('2')
            clev = np.arange(align_avg.min(),align_avg.max(),0.01)
            print('3')
            plt.contourf(x_new, x_new, znew, clev, inline=true)
            print('4')
            plt.xlabel('x')
            plt.ylabel('y')
            cbar=plt.colorbar()
            cbar.set_label(r'$\alpha$', rotation=270)
            
            plt.show()
            '''
            #levels=[-1.0, -0.6666666, -0.333333, 0.0, 0.333333, 0.6666666, 1.0]
            '''
            clev = np.arange(align_avg.min(),align_avg.max(),0.01)
            plt.contourf(bin_pos_x, bin_pos_y, align_avg, clev)
            plt.xlabel('x')
            plt.ylabel('y')
            cbar=plt.colorbar()
            cbar.set_label(r'$\alpha$', rotation=270)
            
            plt.show()
            '''
            '''
            bin_x_pos_skip=np.array([])
            bin_y_pos_skip=np.array([])
            align_avg_x_skip=np.array([])
            align_avg_y_skip=np.array([])
            align_avg_local_x=np.zeros((int(len(align_avg)/3), int(len(align_avg)/3)))
            align_avg_local_y=np.zeros((int(len(align_avg)/3), int(len(align_avg)/3)))

            for ix in range(0, len(occParts)-4):
                for iy in range(0, len(occParts)-4):
                    if ix%3==0:
                        if iy%3==0:
                            bin_x_pos_skip = np.append(bin_x_pos_skip, bin_pos_x[ix])
                            bin_y_pos_skip = np.append(bin_y_pos_skip, bin_pos_y[iy])
                            align_avg_local_x = np.append(align_avg_local_x, (align_avg_x[ix-1][iy]+align_avg_x[ix][iy]+align_avg_x[ix+1][iy]+align_avg_x[ix-1][iy-1]+align_avg_x[ix][iy-1]+align_avg_x[ix+1][iy-1]+align_avg_x[ix-1][iy+1]+align_avg_x[ix][iy+1]+align_avg_x[ix+1][iy+1])/9)
                            align_avg_local_y = np.append(align_avg_local_y, (align_avg_y[ix-1][iy]+align_avg_y[ix][iy]+align_avg_y[ix+1][iy]+align_avg_y[ix-1][iy-1]+align_avg_y[ix][iy-1]+align_avg_x[ix+1][iy-1]+align_avg_x[ix-1][iy+1]+align_avg_x[ix][iy+1]+align_avg_x[ix+1][iy+1])/9)
            print(np.shape(bin_x_pos_skip))
            print(align_avg_local_x
            '''
            #plt.quiver(bin_x_pos_skip, bin_y_pos_skip, align_avg_local_x, align_avg_local_y)
            #plt.xlabel('x')
            #plt.ylabel('y')
            #cbar=plt.colorbar()
            #cbar.set_label(r'$\alpha$', rotation=270)
            
            #plt.show()
            
            
            
            stop
            '''      
            g = open(outPath+outTxt, 'a')
            for h in range(0, len(rad_bin)):
                g.write('{0:.2f}'.format(tst).center(15) + ' ')
                g.write('{0:.1f}'.format(np.amax(clust_size)).center(15) + ' ')
                g.write('{0:.2f}'.format(num_sights[k-1]).center(15) + ' ')
                g.write('{0:.2f}'.format(num_sights[k]).center(15) + ' ')
                g.write('{0:.6f}'.format(rad_bin[h]).center(15) + ' ')
                g.write('{0:.6f}'.format(num_dens[h]).center(15) + ' ')
                g.write('{0:.6f}'.format(align_tot[h]).center(15) + '\n')
            g.close()
            '''