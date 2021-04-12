#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 07:32:51 2020

@author: nicklauersdorf
"""
import sys
import os
from gsd import hoomd
import freud
import numpy as np
import math
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Run locally
hoomdPath='/nas/longleaf/home/njlauers/hoomd-blue/build'
gsdPath='/Volumes/External/04_01_20_parent/gsd/'
# Run locally
sys.path.insert(0,hoomdPath)
#sys.path.insert(0,gsdPath)
imgPath='/pine/scr/n/j/njlauers/scm_tmpdir/cluster_comp/'
r_cut=2**(1/6)
# Get infile and open
inFile = str(sys.argv[1])

if inFile[0:7] == "cluster":
    add = 'cluster_'
else:
    add = ''
inFile = 'pa150_pb500_xa50_ep1_phi60_pNum10000.gsd'
f = hoomd.open(name=gsdPath+inFile, mode='rb')
# Inside and outside activity from command line
peA = float(sys.argv[2])
peB = float(sys.argv[3])
parFrac = float(sys.argv[4])
eps = float(sys.argv[5])

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

# Remaining imports
import numpy as np

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
def computeTauPerTstep(epsilon, mindt=0.000001):
    '''Read in epsilon, output tauBrownian per timestep'''
#    if epsilon != 1.:
#        mindt=0.00001
    kBT = 1.0
    tstepPerTau = float(epsilon / (kBT * mindt))
    return 1. / tstepPerTau

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
def roundUp(n, decimals=0):
    '''Round up size of bins to account for floating point inaccuracy'''
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
    
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

def computeTauLJ(epsilon):
    "Given epsilon, compute lennard-jones time unit"
    tauLJ = ((sigma**2) * threeEtaPiSigma) / epsilon
    return tauLJ

box_data = np.zeros((1), dtype=np.ndarray)  # box dimension holder
r_cut = 2**(1./6.)                          # potential cutoff
tauPerDT = computeTauPerTstep(epsilon=eps)  # brownian time per timestep

# Get number of timesteps in 1 tauBrownian
tauPerDT = computeTauPerTstep(epsilon=eps)
# Little loop to give the desired values
count = 0

f = hoomd.open(name=gsdPath+inFile, mode='rb')

#Define number of time frames simulated
start = 0                   # first frame to process
dumps = int(f.__len__())    # get number of timesteps dumped
end = dumps                 # final frame to process

#xy_array = np.zeros((end,partNum,2))
#r=np.zeros((end,partNum))

#for j in range(start, end):
        #snap = f[j]
        # Easier accessors
        #pos = snap.particles.position              # position in xyz
        #xy = np.delete(pos, 2, 1)                   # Deletes z-position
        
        #xy_array[j]=xy                              # Saves xy position to array

#del_xy_tot=np.zeros((end-1,partNum,2))


#for j in range(start+1,end):
#    del_xy[j-1]=xy_array[j]-xy_array[j-1]
#    for i in range(start+1,partNum):
#        if abs(del_xy[j-1,i-1,0])>h_box: 
#            
#            del_xy[j-1,i-1,0]=l_box-abs(del_xy[j-1,i-1,0])
#                
#        if abs(del_xy[j-1,i-1,1])>h_box:
#            del_xy[j-1,i-1,1]=l_box-abs(del_xy[j-1,i-1,1])
            
#for j in range(start, end):

with hoomd.open(name=gsdPath+inFile, mode='rb') as t:
    start = 0                   # first frame to process
    dumps = int(t.__len__())    # get number of timesteps dumped
    end = dumps                 # final frame to process

    snap = t[0]
    box_data = snap.configuration.box               # Obtain simulated box data
    l_box = box_data[0]                             # Define side length of box
    h_box = l_box / 2.0                             # Define half side length of box
    f_box = freud.box.Box(Lx = l_box, Ly = l_box, is2D = True)    # make freud box
    #maximal interaction distance of h_box-1 (approximately the entire box if x,y=0,0)
    nBins = getNBins(l_box, r_cut)
    sizeBin = roundUp((l_box / nBins), 6) 
    #print(nBins)
    #widthBin = 0.1                                  # Define bin width (can change to alter results)
    #nBins=int(l_box/widthBin)                       # Define number of bins based on box length and bin width
    #print(nBins)
    #stop
    r = np.linspace(0.0,  h_box-1, nBins)             # Define radius for x-axis of plot later
    #rdf_all = np.zeros((end,nBins))           # RDFs
    #rdf_A = np.zeros((end,nBins))            # RDFs
    #rdf_B = np.zeros((end,nBins))            # RDFs
    
    mark=0
    n_r=101                                      # Maximum radius for density calculations
    end_r=40
    total_rdf_all=0
    total_rdf_A=0
    total_rdf_B=0
    

    for iii in range(start,int(end/100)):                   # Run over all saved times
        
        # Easier accessorss
        iii=iii*100
        print('blah')
        print(iii)
        snap = t[iii]                         # Current snapshot of frame
        #snap=t[400]
        typ = snap.particles.typeid                 # get types
        pos = snap.particles.position

        typ0ind=np.where(snap.particles.typeid==0)      # Calculate which particles are type 0
        typ1ind=np.where(snap.particles.typeid==1)      # Calculate which particles are type 1
        pos0=pos[typ0ind]                               # Find positions of type 0 particles
        pos1=pos[typ1ind]     
        partNum = len(typ)

               # get positions
        orient = snap.particles.orientation         # get orientation
        timesteps = snap.configuration.step         # get timestep

        pos[:,-1] = 0.0                             # set z=0
        
        system_all = freud.AABBQuery(f_box, f_box.wrap(pos))    #Calculate neighbor list
        #system_AA = freud.AABBQuery(f_box, f_box.wrap(pos0))    #Calculate neighbor list
        #system_BB = freud.AABBQuery(f_box, f_box.wrap(pos1))    #Calculate neighbor list
        #print(dir(system_all))
        #stop
        cl_all=freud.cluster.Cluster()                      #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()         #Define cluster properties
        clp_all.compute(system_all, cl_all.cluster_idx)             # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes              # find cluster sizes
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]
 
        large_clust_ind_all=np.where(clp_all.sizes>partNum/20)
        
        #cl_AA=freud.cluster.Cluster()                      #Define cluster
        #cl_AA.compute(system_AA, neighbors={'r_max': 1.7})    # Calculate clusters given neighbor list, positions,
         # and maximal radial interaction distance
        #clp_AA = freud.cluster.ClusterProperties()         #Define cluster properties
        #clp_AA.compute(system_AA, cl_AA.cluster_idx)             # Calculate cluster properties given cluster IDs
        #large_clust_ind_AA=np.where(clp_AA.sizes>partNum/1000)
        
        #cl_BB=freud.cluster.Cluster()                      #Define cluster
        #cl_BB.compute(system_BB, neighbors={'r_max': 1.7})    # Calculate clusters given neighbor list, positions,
                                                        # and maximal radial interaction distance
        #clp_BB = freud.cluster.ClusterProperties()         #Define cluster properties
        #clp_BB.compute(system_BB, cl_BB.cluster_idx)             # Calculate cluster properties given cluster IDs
        #large_clust_ind_BB=np.where(clp_BB.sizes>partNum/1000)  # Find positions of type 1 particles


        #a = system_all.points[cl_all.cluster_keys[large_clust_ind_all[0]]]
        AA_in_large = np.array([])
        BB_in_large=np.array([])
        
        if len(large_clust_ind_all[0])>0:
            mark+=1
            for i in range(len(typ0ind[0])):                
                if typ0ind[0][i] in cl_all.cluster_keys[large_clust_ind_all[0][0]]:
                    AA_in_large=np.append(AA_in_large, int(typ0ind[0][i]))
            for i in range(len(typ1ind[0])):
                if typ1ind[0][i] in cl_all.cluster_keys[large_clust_ind_all[0][0]]:
                    BB_in_large=np.append(BB_in_large, int(typ1ind[0][i]))
            A_loc_in_large = np.zeros((len(AA_in_large),3))
            B_loc_in_large = np.zeros((len(BB_in_large),3))
            for i in range(len(AA_in_large)):
                j=int(AA_in_large[i])
                A_loc_in_large[i,:] = pos[j]
            for i in range(len(BB_in_large)):
                j=int(BB_in_large[i])
                B_loc_in_large[i,:] = pos[j]
            pos_all_in_clust=pos[cl_all.cluster_keys[large_clust_ind_all[0][0]]]
        
            system_all_cut = freud.AABBQuery(f_box, f_box.wrap(pos_all_in_clust))    #Calculate neighbor list
            system_AA_cut = freud.AABBQuery(f_box, f_box.wrap(A_loc_in_large))    #Calculate neighbor list
            system_BB_cut = freud.AABBQuery(f_box, f_box.wrap(B_loc_in_large))    #Calculate neighbor list


            radialDFall = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
            radialDFA = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
            radialDFB = freud.density.RDF(nBins, h_box-1)    #Set density per bin with a
        
            #query_points=clp_all.centers[large_clust_ind_all]
            radialDFall.compute(system=system_all_cut,reset=False)               #Calculate radial density function
            radialDFA.compute(system=system_AA_cut, reset=False)               #Calculate radial density function
            radialDFB.compute(system=system_BB_cut, reset=False)               #Calculate radial density function
           
            #radialDF.compute(system=(f_box, f_box.wrap(A_loc_in_large)), query_points=clp_all.centers[large_clust_ind_all], reset=True)               #Calculate radial density function
            #radialDFall.compute(system=system_all_cut, query_points=clp_all.centers[large_clust_ind_all],reset=False)               #Calculate radial density function
            #radialDFA.compute(system=system_AA_cut,query_points=clp_all.centers[large_clust_ind_all], reset=False)               #Calculate radial density function
            #radialDFB.compute(system=system_BB_cut,query_points=clp_all.centers[large_clust_ind_all], reset=False)               #Calculate radial density function
            #,query_points=f_box.wrap(clp_all.centers[large_clust_ind_all])
            #,query_points=f_box.wrap(clp_all.centers[large_clust_ind_all])
            #,query_points=f_box.wrap(clp_all.centers[large_clust_ind_all])
            #radialDF.compute(system=system_all,reset=True)               #Calculate radial density function
            
            #total_rdf_all=total_rdf_all+radialDFall.rdf
            #total_rdf_A=total_rdf_A+radialDFA.rdf
            #total_rdf_B=total_rdf_B+radialDFB.rdf
            #print(iii)
            
            #rdf_all=np.reshape(rdf_all,(1,np.size(rdf_all)))
            #print(rdf_all)
            #stop
        else:
            pass
    plt.plot(r, radialDFall.rdf)
    plt.plot(r, radialDFA.rdf)
    plt.plot(r, radialDFB.rdf)
    plt.show()
    stop
    print(radialDFall.rdf)
    print(radialDFA.rdf)

    #rdf_all_avg=np.sum(rdf_all,0)/mark
    #rdf_B_avg=np.sum(rdf_B,0)/mark
    #rdf_A_avg=np.sum(rdf_A,0)/mark
    rdf_all_avg=total_rdf_all/mark
    rdf_B_avg=total_rdf_B/mark
    rdf_A_avg=total_rdf_A/mark
    img_path='/Users/nicklauersdorf/hoomd-blue/build/img_files/'
    base = img_path + 'pa' + str(peA) +\
       '_pb' + str(peB) +\
       '_xa' + str(partPercA) +\
       '_phi' + str(intPhi)
    imgFile = inFile+name+'.png'
    #rdf_all_avg=np.sum(rdf_all,0)/end
    #rdf_A_avg=np.sum(rdf_A,0)/end
    #rdf_B_avg=np.sum(rdf_B,0)/end

    #plt.plot(r,rdf_all_avg, label = 'All')

    plt.plot(r,rdf_A_avg, label='Pe_A=150')
    plt.plot(r,rdf_B_avg, label='Pe_B=500')
    plt.plot(r,rdf_all_avg, label='both')

    plt.legend()
    #plt.ylim((0,0.08))
    plt.xlabel('r')
    plt.ylabel('RDF')
    plt.tight_layout(w_pad=0.1)
    plt.show()
    stop
    plt.savefig(imgFile, bbox_inches='tight', pad_inches=0., dpi=250)
    plt.close()
    #plt.plot(r,rdf_A_avg)
    #plt.plot(r,rdf_B_avg)
        
    '''
        if len(large_clust_ind_all[0])>0:
            dens0_array_all = np.zeros((end,n_r, len(large_clust_ind_all[0])))  #Define arrays used in following calculations
            dens1_array_all = np.zeros((end,n_r, len(large_clust_ind_all[0])))
            numpart0_array_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            numpart1_array_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            dif_dens0_all=np.zeros((end,n_r-1, len(large_clust_ind_all[0])))
            dif_dens1_all=np.zeros((end,n_r-1, len(large_clust_ind_all[0])))
            ratio_dens_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            dif_parts0_all=np.zeros((end,n_r-1, len(large_clust_ind_all[0])))
            dif_parts1_all=np.zeros((end,n_r-1, len(large_clust_ind_all[0])))

            y=np.linspace(end_r/n_r,end_r,n_r)  # Define array of integers for density radii
            diam=0.5
            for i in range(1,n_r+1):
            
                iacc=1+i*(end_r/n_r)
                
                density0 = freud.density.LocalDensity(diameter=diam, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 0)
                density1 = freud.density.LocalDensity(diameter=diam, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 1)
            
                density0.compute(system=(f_box, f_box.wrap(A_loc_in_large)), query_points=clp_all.centers[large_clust_ind_all]) #Compute local density of type 0 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 0 particle positions, and box info
                density1.compute(system=(f_box, f_box.wrap(B_loc_in_large)), query_points=clp_all.centers[large_clust_ind_all]) #Compute local density of type 1 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 1 particle positions, and box info
                dens0_array_all[iii,i-1] = density0.density #Set type 0 density array value
                dens1_array_all[iii,i-1] = density1.density #Set type 1 density array value
                
                numpart0_array_all[iii,i-1] = density0.density*((diam/2)**2)*math.pi*(iacc**2)*math.pi #Calculate number of type 0 particles in density region of radius r_max
                numpart1_array_all[iii,i-1] = density1.density*((diam/2)**2)*math.pi*(iacc**2)*math.pi #Calculate number of type 1 particles in density region of radius r_max
            for i in range(2,n_r):
                dif_parts0_all[iii,i-2]=numpart0_array_all[iii,i-1]-numpart0_array_all[iii,i-2] #Calculate difference in number of type 0 particles betwen regions of r=i and r=i-1
                dif_parts1_all[iii,i-2]=numpart1_array_all[iii,i-1]-numpart1_array_all[iii,i-2] #Calculate difference in number of type 1 particles betwen regions of r=i and r=i-1
                dif_dens1_all[iii,i-2]=dif_parts1_all[iii,i-2]/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
                dif_dens0_all[iii,i-2]=dif_parts0_all[iii,i-2]/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_all[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y[1:],dif_dens0_all[iii,:,i],label="Number of fast parts: "+str(np.size(AA_in_large)),linestyle='solid', color=color)

                ax.plot(y[1:],dif_dens1_all[iii,:,i],label="Number of slow parts: "+str(np.size(BB_in_large)),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_all[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y,dens0_array_all[iii,:,i],label="Number of fast parts: "+str(np.size(AA_in_large)),linestyle='solid', color=color)

                ax.plot(y,dens1_array_all[iii,:,i],label="Number of slow parts: "+str(np.size(BB_in_large)),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_all[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y,numpart0_array_all[iii,:,i],label="Number of fast parts: "+str(np.size(AA_in_large)),linestyle='solid', color=color)

                ax.plot(y,numpart1_array_all[iii,:,i],label="Number of slow parts: "+str(np.size(BB_in_large)),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_all[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y[1:],dif_parts0_all[iii,:,i],label="Number of fast parts: "+str(np.size(AA_in_large)),linestyle='solid', color=color)

                ax.plot(y[1:],dif_parts1_all[iii,:,i],label="Number of slow parts: "+str(np.size(BB_in_large)),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
    '''
    '''
        plt.scatter(A_loc_in_large[:,0],A_loc_in_large[:,1], s=0.5, c='r', label='A')
        plt.scatter(B_loc_in_large[:,0],B_loc_in_large[:,1], s=0.5, c='g', label='B')
        plt.ylim((-h_box,h_box))
        plt.xlim((-h_box,h_box))
        plt.legend()
        plt.show()
        stop
    '''
    '''
        points = len(large_clust_ind_all[0])        
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        for i in range(points):
            j=large_clust_ind_all[0][i]
            cluster_system_all = freud.AABBQuery(system_all.box, system_all.points[cl_all.cluster_keys[j]])
            cluster_system_all.plot(ax=ax, s=10, label="Cluster {}".format(i))

        plt.title('Center of mass for each cluster', fontsize=20)
        plt.legend(loc='best', fontsize=14)
        plt.gca().tick_params(axis='both', which='both', labelsize=14, size=8)
        plt.gca().set_aspect('equal')
        plt.show()
        
        points = len(large_clust_ind_AA[0])        
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        print(points)
        for i in range(points):
            j=large_clust_ind_AA[0][i]
            cluster_system_AA = freud.AABBQuery(system_AA.box, system_AA.points[cl_AA.cluster_keys[j]])
            cluster_system_AA.plot(ax=ax, s=10, label="Cluster {}".format(i))

        plt.title('Center of mass for each cluster', fontsize=20)
        plt.legend(loc='best', fontsize=14)
        plt.gca().tick_params(axis='both', which='both', labelsize=14, size=8)
        plt.gca().set_aspect('equal')
        plt.show()
        
        points = len(large_clust_ind_BB[0])  
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        print(points)
        for i in range(points):
            j=large_clust_ind_BB[0][i]
            cluster_system_BB = freud.AABBQuery(system_BB.box, system_BB.points[cl_BB.cluster_keys[j]])
            cluster_system_BB.plot(ax=ax, s=10, label="Cluster {}".format(i))

        plt.title('Center of mass for each cluster', fontsize=20)
        plt.legend(loc='best', fontsize=14)
        plt.gca().tick_params(axis='both', which='both', labelsize=14, size=8)
        plt.gca().set_aspect('equal')
        plt.show()
        stop
    '''
        #system.query(clp.centers,dict(num_neighbors>1))
        #stop
    '''
        x_all=np.linspace(1,len(clp_all.centers), num=len(clp_all.centers)*100) #Define array of integers for labeling of clusters
        x_AA=np.linspace(1,len(clp_AA.centers), num=len(clp_AA.centers)*100) #Define array of integers for labeling of clusters
        x_BB=np.linspace(1,len(clp_BB.centers), num=len(clp_BB.centers)*100) #Define array of integers for labeling of clusters

        y=np.linspace(end_r/n_r,end_r,n_r)  # Define array of integers for density radii
        
        if len(large_clust_ind_all[0])>0:
            dens0_array_all = np.zeros((end,n_r, len(large_clust_ind_all[0])))  #Define arrays used in following calculations
            dens1_array_all = np.zeros((end,n_r, len(large_clust_ind_all[0])))
            numpart0_array_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            numpart1_array_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            dif_dens0_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            dif_dens1_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            ratio_dens_all=np.zeros((end,n_r, len(large_clust_ind_all[0])))
            
            print(np.shape(pos0))
            print(np.shape(B_loc_in_large))
            stop
            for i in range(1,n_r+1):
            
                iacc=i*(end_r/n_r)
                
                density0 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 0)
                density1 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 1)
            
                density0.compute(system=(f_box, f_box.wrap(pos0)), query_points=clp_all.centers[large_clust_ind_all]) #Compute local density of type 0 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 0 particle positions, and box info
                density1.compute(system=(f_box, f_box.wrap(pos1)), query_points=clp_all.centers[large_clust_ind_all]) #Compute local density of type 1 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 1 particle positions, and box info
                
                dens0_array_all[iii,i-1] = density0.density #Set type 0 density array value
                dens1_array_all[iii,i-1] = density1.density #Set type 1 density array value
                
                numpart0_array_all[iii,i-1] = density0.density*(iacc**2)*math.pi #Calculate number of type 0 particles in density region of radius r_max
                numpart1_array_all[iii,i-1] = density1.density*(iacc**2)*math.pi #Calculate number of type 1 particles in density region of radius r_max

            for i in range(2,n_r):
                dif_parts0_all=numpart0_array_all[iii,i]-numpart0_array_all[iii,i-1] #Calculate difference in number of type 0 particles betwen regions of r=i and r=i-1
                dif_parts1_all=numpart1_array_all[iii,i]-numpart1_array_all[iii,i-1] #Calculate difference in number of type 1 particles betwen regions of r=i and r=i-1
                dif_dens1_all[iii,i-2]=dif_parts1_all/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
                dif_dens0_all[iii,i-2]=dif_parts0_all/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1

            #ratio_dens_all[iii] = dens0_array_all[iii]/dens1_array_all[iii]#dif_dens0[iii]/dif_dens1[iii]

        if len(large_clust_ind_AA[0])>0:
            dens0_array_AA = np.zeros((end,n_r, len(large_clust_ind_AA[0])))  #Define arrays used in following calculations
            dens1_array_AA = np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            numpart0_array_AA=np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            numpart1_array_AA=np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            dif_dens0_AA=np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            dif_dens1_AA=np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            ratio_dens_AA=np.zeros((end,n_r, len(large_clust_ind_AA[0])))
            for i in range(1,n_r+1):
                iacc=i*(end_r/n_r)
                
                density0 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 0)
                density1 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 1)
                
                density0.compute(system=(f_box, f_box.wrap(pos0)), query_points=clp_AA.centers[large_clust_ind_AA]) #Compute local density of type 0 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 0 particle positions, and box info
                density1.compute(system=(f_box, f_box.wrap(pos1)), query_points=clp_AA.centers[large_clust_ind_AA]) #Compute local density of type 1 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 1 particle positions, and box info
            
                dens0_array_AA[iii,i-1] = density0.density #Set type 0 density array value
                dens1_array_AA[iii,i-1] = density1.density #Set type 1 density array value
                
                numpart0_array_AA[iii,i-1] = density0.density*(iacc**2)*math.pi #Calculate number of type 0 particles in density region of radius r_max
                numpart1_array_AA[iii,i-1] = density1.density*(iacc**2)*math.pi #Calculate number of type 1 particles in density region of radius r_max
                
            for i in range(2,n_r):
                dif_parts0_AA=numpart0_array_AA[iii,i]-numpart0_array_AA[iii,i-1] #Calculate difference in number of type 0 particles betwen regions of r=i and r=i-1
                dif_parts1_AA=numpart1_array_AA[iii,i]-numpart1_array_AA[iii,i-1] #Calculate difference in number of type 1 particles betwen regions of r=i and r=i-1
                dif_dens1_AA[iii,i-2]=dif_parts1_AA/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
                dif_dens0_AA[iii,i-2]=dif_parts0_AA/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
            
            #ratio_dens_AA[iii] = dens0_array_AA[iii]/dens1_array_AA[iii]#dif_dens0[iii]/dif_dens1[iii]
            
        if len(large_clust_ind_BB[0])>0:
            
            dens0_array_BB = np.zeros((end,n_r, len(large_clust_ind_BB[0])))  #Define arrays used in following calculations
            dens1_array_BB = np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            numpart0_array_BB=np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            numpart1_array_BB=np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            dif_dens0_BB=np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            dif_dens1_BB=np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            ratio_dens_BB=np.zeros((end,n_r, len(large_clust_ind_BB[0])))
            
            for i in range(1,n_r+1):
            
                iacc=i*(end_r/n_r)
                
                density0 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 0)
                density1 = freud.density.LocalDensity(diameter=1.0, r_max=iacc)    # Define local density calculation parameters 
            # (particle diameters and maximum radius of area for density calculation for particle type 1)
                
                density0.compute(system=(f_box, f_box.wrap(pos0)), query_points=clp_BB.centers[large_clust_ind_BB]) #Compute local density of type 0 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 0 particle positions, and box info
                density1.compute(system=(f_box, f_box.wrap(pos1)), query_points=clp_BB.centers[large_clust_ind_BB]) #Compute local density of type 1 particles
            # from a radial distance, r_max, from center of mass clusters given center of mass location, type 1 particle positions, and box info
            
                dens0_array_BB[iii,i-1] = density0.density #Set type 0 density array value
                dens1_array_BB[iii,i-1] = density1.density #Set type 1 density array value

                numpart0_array_BB[iii,i-1] = density0.density*(iacc**2)*math.pi #Calculate number of type 0 particles in density region of radius r_max
                numpart1_array_BB[iii,i-1] = density1.density*(iacc**2)*math.pi #Calculate number of type 1 particles in density region of radius r_max
            
            
            for i in range(2,n_r):
                dif_parts0_BB=numpart0_array_BB[iii,i]-numpart0_array_BB[iii,i-1] #Calculate difference in number of type 0 particles betwen regions of r=i and r=i-1
                dif_parts1_BB=numpart1_array_BB[iii,i]-numpart1_array_BB[iii,i-1] #Calculate difference in number of type 1 particles betwen regions of r=i and r=i-1
                dif_dens1_BB[iii,i-2]=dif_parts1_BB/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
                dif_dens0_BB[iii,i-2]=dif_parts0_BB/((math.pi*(i*(end_r/n_r))**2)-(math.pi*((i-1)*(end_r/n_r))**2)) #Calculate difference in density of type 1 particles in region between r=i and r=i-1
            
            #ratio_dens_BB[iii] = dens0_array_BB[iii]/dens1_array_BB[iii]#dif_dens0[iii]/dif_dens1[iii]

            
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_all[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y,dif_dens0_all[iii,:,i],label="Number of fast parts: "+str(clp_all.sizes[large_clust_ind_all[0][i]]),linestyle='solid', color=color)

                ax.plot(y,dif_dens1_all[iii,:,i],label="Number of slow parts: "+str(clp_all.sizes[large_clust_ind_all[0][i]]),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_AA[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y,dif_dens0_AA[iii,:,i],label="Number of fast parts: "+str(clp_AA.sizes[large_clust_ind_AA[0][i]]),linestyle='solid', color=color)

                ax.plot(y,dif_dens1_AA[iii,:,i],label="Number of slow parts: "+str(clp_AA.sizes[large_clust_ind_AA[0][i]]),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()

            fig, ax = plt.subplots()

            for i in range(0,len(large_clust_ind_BB[0])):
                color=next(ax._get_lines.prop_cycler)['color']
                ax.plot(y,dif_dens0_BB[iii,:,i],label="Number of fast parts: "+str(clp_BB.sizes[large_clust_ind_BB[0][i]]),linestyle='solid', color=color)

                ax.plot(y,dif_dens1_BB[iii,:,i],label="Number of slow parts: "+str(clp_BB.sizes[large_clust_ind_BB[0][i]]),linestyle='dashed',color=color)

                #ax.plot(y,ratio_dens[iii,:,i],label="Number of parts: "+str(clp.sizes[large_clust_ind[0][i]]))
            plt.legend(loc='best', fontsize=14)
            #legend_without_duplicate_labels(ax)
            ax.set_xlabel('radius')
            ax.set_ylabel('density')
            plt.show()
            
        stop
            #print(np.shape(clp.centers))
            #stop
            #plt.show()

#
#            for i, data in enumerate([density.num_neighbors, density.density]):
#                #poly = np.poly1d(np.polyfit(cov_basis, data, 1))
#                axes[i].tick_params(axis="both", which="both", labelsize=14)
#                #axes[i].scatter(cov_basis, data)
#                x = np.linspace(*axes[i].get_xlim(), 30)
#                #axes[i].plot(x, poly(x), label="Best fit")
#                axes[i].set_xlabel("Covariance", fontsize=16)
#
#            axes[0].set_ylabel("Number of neighbors", fontsize=16);
#            axes[1].set_ylabel("Density", fontsize=16);
#            plt.show()
#            stop
    '''
    '''
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        for i in range(cl.num_clusters):
            cluster_system = freud.AABBQuery(system.box, system.points[cl.cluster_keys[i]])
            cluster_system.plot(ax=ax, s=10, label="Cluster {}".format(i))

        for i, c in enumerate(clp.centers):
            ax.scatter(c[0], c[1], s=len(cl.cluster_keys[i]),
               label="Cluster {} Center".format(i))

        plt.title('Center of mass for each cluster', fontsize=20)
        #plt.legend(loc='best', fontsize=14)
        plt.gca().tick_params(axis='both', which='both', labelsize=14, size=8)
        plt.gca().set_aspect('equal')
        plt.show()
    '''
    '''
        #print(dir(system.points))
        #stop
        a=system.query(pos, dict(mode='ball', r_max=r_cut, exclude_ii=True))
        #print(dir(a.toNeighborList))
        #stop
        cl.compute(system, neighbors=a.toNeighborList)#snap.bonds.group)
        print(cl.cluster_idx)
        stop
        #radialDF.compute(system, reset=False)               #Calculate radial density function
        #rdf[iii]=radialDF.rdf                       # Assign RDF to array for callback later
    '''
#plt.plot(r, rdf[10], label='All')                  # Plot rdf vs r for 
#plt.ylabel("RDF")
#plt.xlabel('r')
#plt.show()

