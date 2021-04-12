import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle


import scipy.spatial as spatial

import seaborn as sns

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



matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['agg.path.chunksize'] = 999999999999999999999.
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 1.5

# Run locally
#hoomdPath='/Users/nicklauersdorf/hoomd-blue/build/'#

#Run on Cluster
hoomdPath='/nas/home/njlauers/hoomd-blue/build/'

# Add hoomd location to Path
sys.path.insert(0,hoomdPath)

#Cut off interaction radius (Per LJ Potential)
r_cut=2**(1/6)

# Run locally
#outPath='/Volumes/External/test_video/'

#Run on Cluster
outPath='/proj/dklotsalab/users/ABPs/binary_soft/random_init/neighbor_video/'

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
#inFile='pa300_pb300_xa50_ep1.0_phi60_pNum100000.gsd'
#inFile = 'cluster_pa400_pb350_phi60_eps0.1_xa0.8_align3_dtau1.0e-06.gsd'
#inFile='pa400_pb500_xa20_ep1.0_phi60_pNum100000.gsd'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/total_phase_dens_updated/'#'Users/nicklauersdorf/hoomd-blue/build/04_01_20_parent/'#'/Volumes/External/04_01_20_parent/gsd/'
#outPath='/pine/scr/n/j/njlauers/scm_tmpdir/alignment_sparse/'#Users/nicklauersdorf/hoomd-blue/build/test4/'#pine/scr/n/j/njlauers/scm_tmpdir/surfacetens/'
#outPath='/Users/nicklauersdorf/hoomd-blue/build/gsd/'
outF = inFile[:-4]

f = hoomd.open(name=inFile, mode='rb')
dumps = f.__len__()
# Inside and outside activity from command line

#Label simulation parameters
peA = float(sys.argv[2])
peB = float(sys.argv[3])
part_frac_a = float(sys.argv[4])
if part_frac_a>1.0:
    if part_frac_a==100.0:
        part_frac_a=50.0
    part_perc_a = part_frac_a
    part_frac_a=part_frac_a/100.
else:
    if part_frac_a==1.0:
        part_frac_a=0.5
    part_perc_a = part_frac_a*100.
eps = float(sys.argv[5])
peNet=peA*(part_frac_a/100)+peB*(1-(part_frac_a/100))
#Determine which activity is the slow activity or if system is monodisperse

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


#initialize system randomly, can specify GPU execution here

################################################################################
############################# Begin Data Analysis ##############################
################################################################################

import gsd
from gsd import hoomd
from gsd import pygsd

import scipy.spatial as spatial

import seaborn as sns
sns.set(color_codes=True)


parallel.NumThreads(1)                               # don't run multiple threads




size_clusters = np.zeros((dumps), dtype=np.ndarray)

size_min = 1000

# arrays to hold the avg neighbor data
avg_aa = np.zeros(dumps, dtype=np.float64)
avg_ab = np.zeros(dumps, dtype=np.float64)
avg_ba = np.zeros(dumps, dtype=np.float64)
avg_bb = np.zeros(dumps, dtype=np.float64)
avg_all_ab = np.zeros(dumps, dtype=np.float64)

with hoomd.open(name=inFile, mode='rb') as t:
    
    start = 0                  # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process
    snap = t[0]
    first_tstep = snap.configuration.step
    box_data = snap.configuration.box
    l_box = box_data[0]
    left = -(l_box/2)
    right = (l_box/2)
    h_box = l_box / 2.
    radius=np.arange(0,h_box+3.0, 3.0)

    typ = snap.particles.typeid
    partNum = len(typ)
    
    part_a = partNum * part_frac_a         # get the total number of A particles
    part_a = int(part_a)
    part_b = partNum - part_a              # get the total number of B particles
    part_b = int(part_b)
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
    
    # analyze all particles
    for j in range(0, dumps):
        
        snap = t[j]
            
        # Easier accessors
    
        pos = snap.particles.position               # position
            
        pos[:,-1] = 0.0
        ori = snap.particles.orientation 
            #print(ori)
        ang = np.array(list(map(quatToAngle, ori))) # convert to [-pi, pi]
    
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
        how_many = cl_all.num_clusters
        
        sort_id = np.sort(ids)                              # array of IDs sorted small to large
        q_clust = np.zeros((how_many), dtype=np.ndarray)    # my binary 'is it clustered?' array
        index = 0                                           # index of the sorted array to look at
        for a in range(0,len(q_clust)):
            add_clust = 0
            while 1:
                add_clust += 1
                if index == partNum:                       # break if index is too large
                    break
                if sort_id[index] != a:                     # break if ID changes
                    break
                if add_clust == 1:                          # all particles appear once
                    q_clust[a] = 0
                if add_clust > size_min:                    # only multiple ids appear twice
                    q_clust[a] = 1
                index += 1                                  # increment index
    
        # This will get me the length of each array
        Aliq_count = 0
        Bliq_count = 0
        all_count = 0
        
        for c in range(0, partNum):
            
            
            if q_clust[ids[c]] == 1:
                all_count += 1
                if typ[c] == 0:
                    Aliq_count += 1
                else:
                    Bliq_count += 1
    
    #    if all_count != 0:
    #        loop_count = 0
    #        all_pos = np.zeros((all_count, 2), dtype=np.float64)
    #        for c in range(0, part_num):
    #            if q_clust[ids[c]] == 1:
    #                all_pos[loop_count][0] = l_pos[c][0]
    #                all_pos[loop_count][1] = l_pos[c][1]
    #                loop_count += 1
        
        if Aliq_count != 0:
            loop_count = 0
            Aliq_pos = np.zeros((Aliq_count, 2), dtype=np.float64)
            for c in range(0, partNum):
                if q_clust[ids[c]] == 1:
                    if typ[c] == 0:
                        Aliq_pos[loop_count, 0] = pos[c][0]
                        Aliq_pos[loop_count, 1] = pos[c][1]
                        loop_count += 1
        
        if Bliq_count != 0:
            loop_count = 0
            Bliq_pos = np.zeros((Bliq_count, 2), dtype=np.float64)
            for c in range(0, partNum):
                if q_clust[ids[c]] == 1:
                    if typ[c] == 1:
                        Bliq_pos[loop_count, 0] = pos[c][0]
                        Bliq_pos[loop_count, 1] = pos[c][1]
                        loop_count += 1
    
        if all_count!=0:
            loop_count = 0
            ABliq_pos = np.zeros((all_count, 2), dtype=np.float64)
            for c in range(0, partNum):
                if q_clust[ids[c]] == 1:
                    ABliq_pos[loop_count, 0] = pos[c][0]
                    ABliq_pos[loop_count, 1] = pos[c][1]
                    loop_count += 1
                
        
        if Aliq_count!=0:
            a_tree = spatial.KDTree(Aliq_pos)                      # tree of A-type particles
        if Bliq_count!=0:
            b_tree = spatial.KDTree(Bliq_pos)                      # tree of B-type particles
        if all_count!=0:
            ab_tree = spatial.KDTree(ABliq_pos)
        radius = 1.0
        
        if all_count!=0:
            all_ab = ab_tree.query_ball_tree(ab_tree, radius)
            num_all_ab = np.array([])
            
            for i in range(0, len(all_ab)):
                num_all_ab = np.append(num_all_ab, float(len(all_ab[i])))
                    
            #for i in range
            num_all_ab -= 1.0                                       # can't reference itself
            if len(num_all_ab) != 0:
                avg_all_ab[j] = (np.sum(num_all_ab)/len(num_all_ab))
            
        # Let's look at the dense phase:
        # -how many A neighbors the avg A particle has and,
        if Aliq_count !=0:
            aa = a_tree.query_ball_tree(a_tree, radius)
            num_aa = np.array([])
            for i in range(0, len(aa)):
                num_aa = np.append(num_aa, float(len(aa[i])))
                    
            #for i in range
            num_aa -= 1.0                                       # can't reference itself
            if len(num_aa) != 0:
                avg_aa[j] = (np.sum(num_aa)/len(num_aa))
            
            if Bliq_count !=0:
            # -how many B neighbors the avg A particle has and,
            
                ab = a_tree.query_ball_tree(b_tree, radius)
                num_ab = np.array([])
                for i in range(0, len(ab)):
                    num_ab = np.append(num_ab, float(len(ab[i])))
                    
                if len(num_ab) != 0:
                    avg_ab[j] = (np.sum(num_ab)/len(num_ab))
            
                # -how many A neighbors the avg B particle has and,
                
                ba = b_tree.query_ball_tree(a_tree, radius)
                num_ba = np.array([])
                for i in range(0, len(ba)):
                    num_ba = np.append(num_ba, float(len(ba[i])))
                    
                if len(num_ba) != 0:
                    avg_ba[j] = (np.sum(num_ba)/len(num_ba))
            
                # -how many B neighbors the avg B particle has.
                
                bb = b_tree.query_ball_tree(b_tree, radius)
                num_bb = np.array([])
                for i in range(0, len(bb)):
                    num_bb = np.append(num_bb, float(len(bb[i])))
                    
                num_bb -= 1.0                                       # can't reference itself
                if len(num_bb) != 0:
                    avg_bb[j] = (np.sum(num_bb)/len(num_bb))

################################################################################
#################### Plot the individual and total data ########################
################################################################################
        pad = str(j).zfill(4)
        plt_name  = "pa" + str(peA) + "_pb" + str(peB) + "_xa" + str(part_perc_a) + "_eps" + str(eps)
        plt_name1 = "pa" + str(peA) + "_pb" + str(peB) + "_xa" + str(part_perc_a) + "_phi" + str(phi) + "_eps" + str(eps) + "_pNum" +str(partNum) + '_frame_'+pad
        plt_name2 = "pa" + str(peA) + "_pb" + str(peB) + "_xa" + str(part_perc_a) + "_eps" + str(eps)

    # plot some junk
    
        #plt.plot(avg_aa, color="g")
        #plt.plot(avg_ab, color="r")
        #plt.plot(avg_ba, color="b")
        #plt.plot(avg_bb, color="k")
        #plt.savefig('avg_neighs_'+ plt_name + '.png', dpi=1000)
        #plt.close()
        #plt.show()
        
    #    # This is just for checking the cluster algorithm with a visual
    #    fig, ax = plt.subplots()
    #    fig.set_facecolor('black')
    #    plt.subplots_adjust(top = 0.99, bottom = 0.01, right = 0.995, left = 0.005)
    #    x = all_pos[:, 0]
    #    y = all_pos[:, 1]
    #    plt.scatter(x, y, s=1.5, c='b')
    #    ax.set_xlim([left, right])
    #    ax.set_ylim([left, right])
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)
    #    plt.savefig('all_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=1000, box_inches = 'tight', edgecolor='none')
    #    plt.close()
        if all_count !=0:
            fig, ax = plt.subplots()
            #fig.set_facecolor('black')
            plt.subplots_adjust(top = 0.93, bottom = 0.02, right = 0.96, left = 0.02)
            x = ABliq_pos[:, 0]
            y = ABliq_pos[:, 1]
            z_a = num_all_ab
            plt.scatter(x, y, c=z_a, s=1.0, cmap='plasma')
            ax.set_xlim([left, right])
            ax.set_ylim([left, right])
            ax.spines['bottom'].set_color('0.0')
            ax.spines['top'].set_color('0.0')
            ax.spines['right'].set_color('0.0')
            ax.spines['left'].set_color('0.0')
            ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            plt.colorbar()
            plt.clim(0.0, 6.0)
            plt.xticks([])
            plt.yticks([])
            plt.text(0.53, 1.07, s=r'$\mathrm{Pe}_\mathrm{A}$' + ' = ' + str(int(peA)) + ', ' + r'$\mathrm{Pe}_\mathrm{B}$' + ' = ' + str(int(peB)) + ', ' +r'$\chi_\mathrm{A}$' + ' = ' + str(part_frac_a) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))),
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes,
                fontsize=17)
            #plt.tight_layout()
            plt.savefig(outPath+'all_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150, box_inches = 'tight', edgecolor='black', linewidth=1.)
            plt.close()
        #plt.show()
        
        if Aliq_count !=0:
            fig, ax = plt.subplots()
            #fig.set_facecolor('black')
            plt.subplots_adjust(top = 0.93, bottom = 0.02, right = 0.96, left = 0.02)
            x = Aliq_pos[:, 0]
            y = Aliq_pos[:, 1]
            z_a = num_aa
            plt.scatter(x, y, c=z_a, s=1.0, cmap='plasma')
            ax.set_xlim([left, right])
            ax.set_ylim([left, right])
            ax.spines['bottom'].set_color('0.0')
            ax.spines['top'].set_color('0.0')
            ax.spines['right'].set_color('0.0')
            ax.spines['left'].set_color('0.0')
            ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=18,
                bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            plt.colorbar()
            plt.clim(0.0, 6.0)
            plt.xticks([])
            plt.yticks([])
            plt.text(0.53, 1.07, s=r'$\mathrm{Pe}_\mathrm{A}$' + ' = ' + str(int(peA)) + ', ' + r'$\mathrm{Pe}_\mathrm{B}$' + ' = ' + str(int(peB)) + ', ' +r'$\chi_\mathrm{A}$' + ' = ' + str(part_frac_a) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))),
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes,
                fontsize=17)
            plt.savefig(outPath+ 'aa_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150, box_inches = 'tight', edgecolor='black', linewidth=1.)
            plt.close()
            #plt.show()
            if Bliq_count !=0:
                fig, ax = plt.subplots()
                #fig.set_facecolor('black')
                plt.subplots_adjust(top = 0.93, bottom = 0.02, right = 0.96, left = 0.02)
                x = Aliq_pos[:, 0]
                y = Aliq_pos[:, 1]
                z_a = num_ab
                plt.scatter(x, y, c=z_a, s=1.0, cmap='plasma')
                ax.set_xlim([left, right])
                ax.set_ylim([left, right])
                ax.spines['bottom'].set_color('0.0')
                ax.spines['top'].set_color('0.0')
                ax.spines['right'].set_color('0.0')
                ax.spines['left'].set_color('0.0')
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                plt.colorbar()
                plt.clim(0.0, 6.0)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.53, 1.07, s=r'$\mathrm{Pe}_\mathrm{A}$' + ' = ' + str(int(peA)) + ', ' + r'$\mathrm{Pe}_\mathrm{B}$' + ' = ' + str(int(peB)) + ', ' +r'$\chi_\mathrm{A}$' + ' = ' + str(part_frac_a) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))),
                         horizontalalignment='center', verticalalignment='top',
                         transform=ax.transAxes,
                         fontsize=17)
                plt.savefig(outPath+'ab_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150, box_inches = 'tight', edgecolor='black', linewidth=1.)
                plt.close()
                #plt.show()
            
                fig, ax = plt.subplots()
                #fig.set_facecolor('black')
                plt.subplots_adjust(top = 0.93, bottom = 0.02, right = 0.96, left = 0.02)
                x = Bliq_pos[:, 0]
                y = Bliq_pos[:, 1]
                z_a = num_ba
                plt.scatter(x, y, c=z_a, s=1.0, cmap='plasma')
                ax.set_xlim([left, right])
                ax.set_ylim([left, right])
                ax.spines['bottom'].set_color('0.0')
                ax.spines['top'].set_color('0.0')
                ax.spines['right'].set_color('0.0')
                ax.spines['left'].set_color('0.0')
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                plt.colorbar()
                plt.clim(0.0, 6.0)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.53, 1.07, s=r'$\mathrm{Pe}_\mathrm{A}$' + ' = ' + str(int(peA)) + ', ' + r'$\mathrm{Pe}_\mathrm{B}$' + ' = ' + str(int(peB)) + ', ' +r'$\chi_\mathrm{A}$' + ' = ' + str(part_frac_a) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))),
                         horizontalalignment='center', verticalalignment='top',
                         transform=ax.transAxes,
                         fontsize=17)
                plt.savefig(outPath+'ba_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150, box_inches = 'tight', edgecolor='black', linewidth=1.)
                plt.close()
                #plt.show()
            
                fig, ax = plt.subplots()
                #fig.set_facecolor('black')
                plt.subplots_adjust(top = 0.93, bottom = 0.02, right = 0.96, left = 0.02)
                x = Bliq_pos[:, 0]
                y = Bliq_pos[:, 1]
                z_a = num_bb
                plt.scatter(x, y, c=z_a, s=1.0, cmap='plasma')
            #        ax.get_xlim()
            #        ax.get_ylim()
                ax.set_xlim([left, right])
                ax.set_ylim([left, right])
                ax.spines['bottom'].set_color('0.0')
                ax.spines['top'].set_color('0.0')
                ax.spines['right'].set_color('0.0')
                ax.spines['left'].set_color('0.0')
                ax.text(0.95, 0.025, s=r'$\tau$' + ' = ' + '{:.1f}'.format(tst*3) + ' ' + r'$\tau_\mathrm{r}$',
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=18,
                        bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                plt.colorbar()
                plt.clim(0.0, 6.0)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.53, 1.07, s=r'$\mathrm{Pe}_\mathrm{A}$' + ' = ' + str(int(peA)) + ', ' + r'$\mathrm{Pe}_\mathrm{B}$' + ' = ' + str(int(peB)) + ', ' +r'$\chi_\mathrm{A}$' + ' = ' + str(part_frac_a) + ', ' + r'$\phi$' + ' = ' + str(phi) + ', ' + r'$\epsilon$' + ' = ' + r'$10^{{{}}}$'.format(int(np.log10(eps))),
                         horizontalalignment='center', verticalalignment='top',
                         transform=ax.transAxes,
                         fontsize=17)
                plt.savefig(outPath+'bb_' + plt_name1 + '.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150, box_inches = 'tight', edgecolor='black', linewidth=1.)
                plt.close()
                #plt.show()
                #stop
