
import sys
import os

from gsd import hoomd
from freud import box
import freud
from freud import diffraction
import numpy as np
import math
import scipy
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage


import matplotlib

import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import curve_fit

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory
import utility
import plotting_utility
import phase_identification
import binning
import particles

# Class of lattice structure measurements
class measurement:
    """
    Purpose: 
    This class contains a series of basic functions for analyzing systems that require 
    the differentiation of phases, including lattice spacing, order parameters, and 
    neighbor information of each phase
    """
    def __init__(self, lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, px, py, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict):

        import freud

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box/2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box/2

        # Instantiated simulation box
        self.f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

        try:
            # Total number of bins in x-direction
            self.NBins_x = int(NBins_x)

            # Total number of bins in y-direction
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        # Initialize utility functions for call back later
        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        # X-length of bin
        self.sizeBin_x = self.utility_functs.roundUp((lx_box / NBins_x), 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp((ly_box / NBins_y), 6)

        # Number of particles
        self.partNum = partNum

        # Dictionary containing phase information per particle
        self.phasePart = phase_dict['part']

        # Dictionary containing binned phase information
        self.phaseBin = phase_dict['bin']

        # Dictionary containing binned and per-particle phase information
        self.phase_dict = phase_dict

        # Dictionary containing particle ids
        self.part_dict = part_dict

        # Array (partNum) of particle positions
        self.pos = pos

        # Array (partNum) of particle types
        self.typ = typ

        # Array (partNum) of particle orientations
        self.px = px
        self.py = py

        # Array (NBins, NBins) of particle ids located per bin
        self.binParts = part_dict['id']

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Fraction of particles of type A
        self.parFrac = parFrac

        # Dictionary containing binned particle alignment information
        self.align_dict = align_dict

        # Dictionary containing binned area fraction information
        self.area_frac_dict = area_frac_dict

        # Dictionary containing binned interparticle pressure information
        self.press_dict = press_dict

        # Initialize binning functions for call back later
        self.binning = binning.binning(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.typ, self.eps)

        # Initialize plotting functions for call back later
        self.plotting_utility = plotting_utility.plotting_utility(self.lx_box, self.ly_box, self.partNum, self.typ)

        # Initialize particle property functions for call back later
        self.particle_prop_functs = particles.particle_props(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.eps, self.typ, self.pos, self.px, self.py)

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

        self.sf_box = freud.box.Box(Lx=lx_box, Ly=ly_box, Lz=lx_box * 3, is2D=False)


    def average_activity(self, part_ids = None):
        '''
        Purpose: Takes the composition of the system and each particle's activity and calculates
        the average activity within the system

        Inputs:
        part_ids (optional): option to provide updated array of partice types

        Output:
        peNet: average activity
        '''

        # Instantiate empty sums
        pe_tot = 0
        pe_num = 0

        # Set to default particle IDs if undefined
        if part_ids is None:
            part_ids = self.binParts

        # Loop over particles
        for i in part_ids:
            
            # Sum activity of particles given
            if self.typ[i]==0:
                pe_tot += self.peA
                pe_num += 1
            else:
                pe_tot += self.peB
                pe_num += 1

        # Calculate average activity of particles given
        if pe_num>0:
            peNet = pe_tot / pe_num
        else:
            peNet = 0

        return peNet

    def lattice_spacing(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to compute the average
        interparticle separation distance (lattice spacing) between each particle and their nearest,
        interacting neighbors for plotting and simplified, statistical outputs

        Outputs:
        lat_stat_dict: dictionary containing the mean and standard deviation of the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, for each phase.

        lat_plot_dict: dictionary containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface reference particle type ('all', 'A', or 'B').
        '''
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]                               # Find positions of type 0 particles
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        # Locate potential neighbor particles of all types in the dense phase
        system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))

        #Checks for all neighbors around type 'A' particles within bulk
        A_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()

        #Checks for all neighbors around type 'B' particles within bulk
        B_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        # Calculate interparticle separation distances between A reference particles and all neighbors within bulk
        bulk_A_lats = self.utility_functs.sep_dist_arr(pos_dense[A_bulk_nlist.point_indices], pos_A_bulk[A_bulk_nlist.query_point_indices])

        # Calculate interparticle separation distances between B reference particles and all neighbors within bulk
        bulk_B_lats = self.utility_functs.sep_dist_arr(pos_dense[B_bulk_nlist.point_indices], pos_B_bulk[B_bulk_nlist.query_point_indices])

        # Calculate interparticle separation distances between all reference particles and all neighbors within bulk
        bulk_lats = np.append(bulk_A_lats, bulk_B_lats)

        # Mean lattice spacing by type
        bulk_lat_mean = np.mean(bulk_lats)
        bulk_A_lat_mean = np.mean(bulk_A_lats)
        bulk_B_lat_mean = np.mean(bulk_B_lats)

        # Standard deviation lattice spacing by type
        bulk_lat_std = np.std(bulk_lats)
        bulk_A_lat_std = np.std(bulk_A_lats)
        bulk_B_lat_std = np.std(bulk_B_lats)

        # Initialize empty arrays to find mean lattice spacing per reference particle for plotting
        bulk_A_lat_ind = np.array([], dtype=int)
        bulk_A_lat_arr = np.array([])

        bulk_B_lat_ind = np.array([], dtype=int)
        bulk_B_lat_arr = np.array([])

        #Loop over all A-all neighbor list pairings
        for i in A_bulk_nlist.point_indices:
            if i not in bulk_A_lat_ind:
                # Find IDs of neighbor list with i as reference particle
                loc = np.where(A_bulk_nlist.point_indices==i)[0]

                #Calculate mean and STD of i reference particle's lattice spacing
                bulk_A_lat_arr = np.append(bulk_A_lat_arr, np.mean(bulk_A_lats[loc]))
                bulk_A_lat_ind = np.append(bulk_A_lat_ind, int(i))

        #Loop over all B-all neighbor list pairings
        for i in B_bulk_nlist.point_indices:
            if i not in bulk_B_lat_ind:
                # Find IDs of neighbor list with i as reference particle
                loc = np.where(B_bulk_nlist.point_indices==i)[0]

                #Calculate mean and STD of i reference particle's lattice spacing
                bulk_B_lat_arr = np.append(bulk_B_lat_arr, np.mean(bulk_B_lats[loc]))
                bulk_B_lat_ind = np.append(bulk_B_lat_ind, int(i))

        # Array of mean lattice spacings for all reference particles
        bulk_lat_arr = np.append(bulk_A_lat_arr, bulk_B_lat_arr)

        # Array of particle ids corresponding to the mean lattice spacing array for all reference particles
        bulk_lat_ind = np.append(bulk_A_lat_ind, bulk_B_lat_ind)

        # Locate potential neighbor particles of all types in the system
        system_all_int = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))   #Calculate neighbor list

        #Checks for all neighbors around type 'A' particles within interface
        A_int_nlist = system_all_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()

        #Checks for all neighbors around type 'B' particles within bulk
        B_int_nlist = system_all_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        # Calculate interparticle separation distances between A reference particles and all neighbors within interface
        int_A_lats = self.utility_functs.sep_dist_arr(self.pos[A_int_nlist.point_indices], pos_A_int[A_int_nlist.query_point_indices])

        # Calculate interparticle separation distances between B reference particles and all neighbors within interface
        int_B_lats = self.utility_functs.sep_dist_arr(self.pos[B_int_nlist.point_indices], pos_B_int[B_int_nlist.query_point_indices])

        # Calculate interparticle separation distances between all reference particles and all neighbors within interface
        int_lats = np.append(int_A_lats, int_B_lats)

        # Mean lattice spacing by type
        int_lat_mean = np.mean(int_lats)
        int_A_lat_mean = np.mean(int_A_lats)
        int_B_lat_mean = np.mean(int_B_lats)

        # Standard deviation lattice spacing by type
        int_lat_std = np.std(int_lats)
        int_A_lat_std = np.std(int_A_lats)
        int_B_lat_std = np.std(int_B_lats)

        # Initialize empty arrays to find mean lattice spacing per reference particle for plotting
        int_A_lat_ind = np.array([], dtype=int)
        int_A_lat_arr = np.array([])

        int_B_lat_ind = np.array([], dtype=int)
        int_B_lat_arr = np.array([])

        #Loop over all A-all neighbor list pairings
        for i in A_int_nlist.point_indices:
            if i not in int_A_lat_ind:
                # Find IDs of neighbor list with i as reference particle
                loc = np.where(A_int_nlist.point_indices==i)[0]

                #Calculate mean and STD of i reference particle's lattice spacing
                int_A_lat_arr = np.append(int_A_lat_arr, np.mean(int_A_lats[loc]))
                int_A_lat_ind = np.append(int_A_lat_ind, int(i))

        #Loop over all B-all neighbor list pairings
        for i in B_int_nlist.point_indices:
            if i not in int_B_lat_ind:
                # Find IDs of neighbor list with i as reference particle
                loc = np.where(B_int_nlist.point_indices==i)[0]

                #Calculate mean and STD of i reference particle's lattice spacing
                int_B_lat_arr = np.append(int_B_lat_arr, np.mean(int_B_lats[loc]))
                int_B_lat_ind = np.append(int_B_lat_ind, int(i))

        # Array of mean lattice spacings for all reference particles
        int_lat_arr = np.append(int_A_lat_arr, int_B_lat_arr)

        # Array of particle ids corresponding to the mean lattice spacing array for all reference particles
        int_lat_ind = np.append(int_A_lat_ind, int_B_lat_ind)

        # Array of mean lattice spacings for reference particles of respective type in dense phase
        dense_lat_arr = np.append(bulk_lat_arr, int_lat_arr)
        dense_A_lat_arr = np.append(bulk_A_lat_arr, int_A_lat_arr)
        dense_B_lat_arr = np.append(bulk_B_lat_arr, int_B_lat_arr)

        # Array of mean interparticle separation distances for reference particles of respective type in dense phase
        dense_lats = np.append(bulk_lats, int_lats)
        dense_A_lats = np.append(bulk_A_lats, int_A_lats)
        dense_B_lats = np.append(bulk_B_lats, int_B_lats)

        # Mean lattice spacing by type
        dense_lat_mean = np.mean(dense_lats)
        dense_A_lat_mean = np.mean(dense_A_lats)
        dense_B_lat_mean = np.mean(dense_B_lats)

        # Standard deviation lattice spacing by type
        dense_lat_std = np.std(dense_lats)
        dense_A_lat_std = np.std(dense_A_lats)
        dense_B_lat_std = np.std(dense_B_lats)

        # Save x-y position to arrays for all reference particles of the respective type and phases for plotting purposes
        pos_bulk_x_lat = pos_dense[bulk_lat_ind,0]
        pos_bulk_y_lat = pos_dense[bulk_lat_ind,1]
        pos_bulk_x_A_lat = pos_dense[bulk_A_lat_ind,0]
        pos_bulk_y_A_lat = pos_dense[bulk_A_lat_ind,1]
        pos_bulk_x_B_lat = pos_dense[bulk_B_lat_ind,0]
        pos_bulk_y_B_lat = pos_dense[bulk_B_lat_ind,1]

        pos_int_x_lat = self.pos[int_lat_ind,0]
        pos_int_y_lat = self.pos[int_lat_ind,1]
        pos_int_x_A_lat = self.pos[int_A_lat_ind,0]
        pos_int_y_A_lat = self.pos[int_A_lat_ind,1]
        pos_int_x_B_lat = self.pos[int_B_lat_ind,0]
        pos_int_y_B_lat = self.pos[int_B_lat_ind,1]

        pos_dense_x_lat = np.append(pos_dense[bulk_lat_ind,0], self.pos[int_lat_ind,0])
        pos_dense_y_lat = np.append(pos_dense[bulk_lat_ind,1], self.pos[int_lat_ind,1])
        pos_dense_x_A_lat = np.append(pos_dense[bulk_A_lat_ind,0], self.pos[int_A_lat_ind,0])
        pos_dense_y_A_lat = np.append(pos_dense[bulk_A_lat_ind,1], self.pos[int_A_lat_ind,1])
        pos_dense_x_B_lat = np.append(pos_dense[bulk_B_lat_ind,0], self.pos[int_B_lat_ind,0])
        pos_dense_y_B_lat = np.append(pos_dense[bulk_B_lat_ind,1], self.pos[int_B_lat_ind,1])

        # Create output dictionary for statistical averages of lattice spacing of each particle per phase/activity pairing
        lat_stat_dict = {'bulk': {'all': {'mean': bulk_lat_mean, 'std': bulk_lat_std}, 'A': {'mean': bulk_A_lat_mean, 'std': bulk_A_lat_std}, 'B': {'mean': bulk_B_lat_mean, 'std': bulk_B_lat_std}}, 'int': {'all': {'mean': int_lat_mean, 'std': int_lat_std}, 'A': {'mean': int_A_lat_mean, 'std': int_A_lat_std}, 'B': {'mean': int_B_lat_mean, 'std': int_B_lat_std}}, 'dense': {'all': {'mean': dense_lat_mean, 'std': dense_lat_std }, 'A': {'mean': dense_A_lat_mean, 'std': dense_A_lat_std }, 'B': {'mean': dense_B_lat_mean, 'std': dense_B_lat_std } } }

        # Create output dictionary for plotting of lattice spacing of each particle per phase/activity pairing and their respective x-y locations
        lat_plot_dict = {'dense': {'all': {'vals': dense_lat_arr, 'x': pos_dense_x_lat, 'y': pos_dense_y_lat}, 'A': {'vals': dense_A_lat_arr, 'x': pos_dense_x_A_lat, 'y': pos_dense_y_A_lat}, 'B': {'vals': dense_B_lat_arr, 'x': pos_dense_x_B_lat, 'y': pos_dense_x_B_lat }  }, 'bulk': {'all': {'vals': bulk_lat_arr, 'x': pos_bulk_x_lat, 'y': pos_bulk_y_lat}, 'A': {'vals': bulk_A_lat_arr, 'x': pos_bulk_x_A_lat, 'y': pos_bulk_y_A_lat}, 'B': {'vals': bulk_B_lat_arr, 'x': pos_bulk_x_B_lat, 'y': pos_bulk_x_B_lat }  }, 'int': {'all': {'vals': int_lat_arr, 'x': pos_int_x_lat, 'y': pos_int_y_lat}, 'A': {'vals': int_A_lat_arr, 'x': pos_int_x_A_lat, 'y': pos_int_y_A_lat}, 'B': {'vals': int_B_lat_arr, 'x': pos_int_x_B_lat, 'y': pos_int_x_B_lat }  } }

        return lat_stat_dict, lat_plot_dict

    def num_dens_mean(self, area_frac_dict):
        '''
        Purpose: Takes the binned area fraction of each species and computes the
        average number density of each species within the bulk

        Outputs:
        bulk_num_dens_dict: dictionary containing the mean number density of each species
        within the bulk
        '''

        # Read in binned number density array (NBins, NBins)
        num_dens = area_frac_dict['bin']['all']
        num_dens_A = area_frac_dict['bin']['A']
        num_dens_B = area_frac_dict['bin']['B']

        # Running sum of bulk number density of respective type
        num_dens_sum = 0
        num_dens_A_sum = 0
        num_dens_B_sum = 0

        # Number of bins summing over
        num_dens_val = 0

        #Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):
                if self.phaseBin[ix][iy]==0:

                    # Sum bulk number density of bin for respective type
                    num_dens_sum += num_dens[ix][iy]/(math.pi/4)
                    num_dens_A_sum += num_dens_A[ix][iy]/(math.pi/4)
                    num_dens_B_sum += num_dens_B[ix][iy]/(math.pi/4)
                    num_dens_val += 1

        # Average over number of bins
        if num_dens_val > 0:
            num_dens_mean = num_dens_sum / num_dens_val
            num_dens_A_mean = num_dens_A_sum / num_dens_val
            num_dens_B_mean = num_dens_B_sum / num_dens_val
        else:
            num_dens_mean = 0
            num_dens_A_mean = 0
            num_dens_B_mean = 0

        # Create output dictionary for average number density of bulk
        bulk_num_dens_dict = {'all': num_dens_mean, 'A': num_dens_A_mean, 'B': num_dens_B_mean}

        return bulk_num_dens_dict

    def radial_df(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to compute the
        interparticle separation distance between each pair of reference particle
        and one of their nearest, interacting neighbor and averages over the system to
        provide the probability of finding a neighbor of a given species at each
        separation distance from a reference particle of a given species (i.e. radial
        distribution function, RDF)

        Outputs:
        rad_df_dict: dictionary containing arrays of the probability distribution
        function of finding a particle of a given species ('all', 'A', or 'B') at
        a given radial distance ('r') from a reference particle of a given species
        ('all', 'A', or 'B') for a given reference-neighbor pair within the bulk phase (i.e. all-A means
        all reference particles with A neighbors).
        '''

        # Calculates mean bulk number density
        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Width, in distance units, of bin
        wBins = 0.02

        # Maximum distance to compute RDF for
        rstop = 10.

        # Number of bins given total distance desire (rstop) and step size (wBins)
        nBins = rstop / wBins

        # Actual bin width
        wbinsTrue=(rstop)/(nBins-1)

        # Array of radial steps for RDF
        r=np.arange(0.0,rstop+wbinsTrue,wbinsTrue)

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=rstop)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        #Checks for 'A' type neighbors around type 'A' particles within bulk
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()

        #Checks for 'A' type neighbors around type 'B' particles within bulk
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        #Checks for 'B' type neighbors around type 'A' particles within bulk
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()

        #Checks for 'B' type neighbors around type 'B' particles within bulk
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        # Calculate interparticle distance between type A reference particle and type A neighbor
        difr_AA_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AA_bulk_nlist.point_indices], pos_A_bulk[AA_bulk_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type A neighbor
        difr_AB_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AB_bulk_nlist.point_indices], pos_B_bulk[AB_bulk_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type A neighbor
        difr_BA_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[BA_bulk_nlist.point_indices], pos_A_bulk[BA_bulk_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type B neighbor
        difr_BB_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[BB_bulk_nlist.point_indices], pos_B_bulk[BB_bulk_nlist.query_point_indices])

        # Calculate interparticle distance between all reference particle and type A neighbor
        difr_allA_bulk = np.append(difr_AA_bulk, difr_AB_bulk)

        # Calculate interparticle distance between all reference particle and type B neighbor
        difr_allB_bulk = np.append(difr_BB_bulk, difr_AB_bulk)

        # Calculate interparticle distance between all reference particle and all neighbors
        difr_allall_bulk = np.append(difr_allA_bulk, difr_allB_bulk)

        #Initiate empty arrays for RDF of each respective activity pairing
        g_r_allall_bulk = []
        g_r_allA_bulk = []
        g_r_allB_bulk = []
        g_r_AA_bulk = []
        g_r_AB_bulk = []
        g_r_BA_bulk = []
        g_r_BB_bulk = []
        r_arr = []

        # Loop over radial slices
        for m in range(1, len(r)-2):
            difr = r[m+1] - r[m]

            #Save minimum interparticle separation from separation range
            r_arr.append(r[m])

            # Locate all neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allall_bulk>=r[m]) & (difr_allall_bulk<r[m+1]))[0]

            # Total number of all-all particle pairs divided by volume of radial slice
            rho_all = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-all pairs divided by volume of the bulk phase
            rho_tot_all = len(phase_part_dict['bulk']['all']) * num_dens_mean_dict['all']

            # Save all-all RDF at respective distance range
            g_r_allall_bulk.append(rho_all / rho_tot_all)

            # Locate A neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allA_bulk>=r[m]) & (difr_allA_bulk<r[m+1]))[0]

            # Total number of all-A particle pairs divided by volume of radial slice
            rho_a = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-A pairs divided by volume of the bulk phase
            rho_tot_a = len(pos_A_bulk) * num_dens_mean_dict['all']

            # Save all-A RDF at respective distance range
            g_r_allA_bulk.append(rho_a / rho_tot_a)

            # Locate B neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allB_bulk>=r[m]) & (difr_allB_bulk<r[m+1]))[0]

            # Total number of all-B particle pairs divided by volume of radial slice
            rho_b = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-B pairs divided by volume of the bulk phase
            rho_tot_b = len(pos_B_bulk) * num_dens_mean_dict['all']

            # Save all-B RDF at respective distance range
            g_r_allB_bulk.append(rho_b / rho_tot_b)

            # Locate A neighboring particles within given distance range from A reference particle
            inds = np.where((difr_AA_bulk>=r[m]) & (difr_AA_bulk<r[m+1]))[0]

            # Total number of A-A particle pairs divided by volume of radial slice
            rho_aa = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible A-A pairs divided by volume of the bulk phase
            rho_tot_aa = len(pos_A_bulk) * num_dens_mean_dict['A']

            # Save A-A RDF at respective distance range
            g_r_AA_bulk.append(rho_aa / rho_tot_aa)

            # Locate B neighboring particles within given distance range from A reference particle
            inds = np.where((difr_AB_bulk>=r[m]) & (difr_AB_bulk<r[m+1]))[0]

            # Total number of A-B particle pairs divided by volume of radial slice
            rho_ab = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible A-B pairs divided by volume of the bulk phase
            rho_tot_ab = len(pos_B_bulk) * (num_dens_mean_dict['A'])

            # Save A-B RDF at respective distance range
            g_r_AB_bulk.append(rho_ab / rho_tot_ab)

            # Locate A neighboring particles within given distance range from B reference particle
            inds = np.where((difr_BA_bulk>=r[m]) & (difr_BA_bulk<r[m+1]))[0]

            # Total number of B-A particle pairs divided by volume of radial slice
            rho_ba = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible B-A pairs divided by volume of the bulk phase
            rho_tot_ba = len(pos_A_bulk) * (num_dens_mean_dict['B'])

            # Save B-A RDF at respective distance range
            g_r_BA_bulk.append(rho_ba / rho_tot_ba)


            # Locate B neighboring particles within given distance range from B reference particle
            inds = np.where((difr_BB_bulk>=r[m]) & (difr_BB_bulk<r[m+1]))[0]

            # Total number of B-B particle pairs divided by volume of radial slice (numerator of RDF)
            rho_bb = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible B-B pairs divided by volume of the bulk phase
            rho_tot_bb = len(pos_B_bulk) * num_dens_mean_dict['B']

            # Save B-B RDF at respective distance range
            g_r_BB_bulk.append(rho_bb / rho_tot_bb)



        # Create output dictionary for plotting of RDF vs separation distance
        rad_df_dict = {'r': r_arr, 'all-all': g_r_allall_bulk, 'all-A': g_r_allA_bulk, 'all-B': g_r_allB_bulk, 'A-A': g_r_AA_bulk, 'A-B': g_r_AB_bulk, 'B-A': g_r_BA_bulk, 'B-B': g_r_BB_bulk}
        return rad_df_dict

    def wasserstein_distance(self, rad_df_dict, lattice_spacing):
        '''
        Purpose: Takes the composition of each phase and uses already calculated radial
        distribution functions to compute the Wasserstein or earth-mover's distance (a distance
        metric to calculate the 'work' needed to convert one probability distribution into another)
        as a means of quantifying segregation via g(r)

        Inputs:
        rad_df_dict: dictionary containing arrays of the probability distribution function of finding
        a neighbor particle of a given species ('all', 'A', or 'B') at a given distance ('r') from
        the reference particle of a given species ('all', 'A', or 'B') for a given reference-neighbor 
        pair (i.e. all-A means all reference particles with A neighbors).

        lattice_spacing: mean bulk lattice spacing for normalizing distances to compare between systems

        Outputs:
        wasserstein_dict: dictionary containing the average work required to convert each g(r) of a
        reference particle type to another within the bulk
        '''

        # Define input radial density function
        r_arr = rad_df_dict['r']/lattice_spacing
        g_r_allall_bulk = rad_df_dict['all-all']
        g_r_AA_bulk = rad_df_dict['A-A']
        g_r_AB_bulk = rad_df_dict['A-B']
        g_r_BB_bulk = rad_df_dict['B-B']
        g_r_allA_bulk = rad_df_dict['all-A']
        g_r_allB_bulk = rad_df_dict['all-B']

        # Calculate wasserstein distance for A-A g(r) to B-B g(r)
        AA_BB_wasserstein = scipy.stats.wasserstein_distance(g_r_AA_bulk, g_r_BB_bulk)

        # Calculate wasserstein distance for A-B g(r) to B-B g(r)
        AB_BB_wasserstein = scipy.stats.wasserstein_distance(g_r_AB_bulk, g_r_BB_bulk)

        # Calculate wasserstein distance for all-A g(r) to all-B g(r)
        allA_allB_wasserstein = scipy.stats.wasserstein_distance(g_r_allA_bulk, g_r_allB_bulk)

        # Calculate wasserstein distance for all-B g(r) to B-B g(r)
        allB_BB_wasserstein = scipy.stats.wasserstein_distance(g_r_allB_bulk, g_r_BB_bulk)

        # Calculate wasserstein distance for all-A g(r) to A-A g(r)
        allA_AA_wasserstein = scipy.stats.wasserstein_distance(g_r_allA_bulk, g_r_AA_bulk)

        # Create output dictionary for wasserstein distance for each reference-neighbor activity pairing in bulk
        wasserstein_dict = {'AA-BB': AA_BB_wasserstein, 'AB-BB': AB_BB_wasserstein, 'allA-allB': allA_allB_wasserstein, 'allA-AA': allA_AA_wasserstein, 'allB-BB': allB_BB_wasserstein}

        return wasserstein_dict

    def structure_factor2(self):
        #IN PROGRESS
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to compute the average
        interparticle separation distance (lattice spacing) between each particle and their nearest,
        interacting neighbors for plotting and simplified, statistical outputs

        Outputs:
        lat_stat_dict: dictionary containing the mean and standard deviation of the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, for each phase.

        lat_plot_dict: dictionary containing information on the lattice spacing, averaged
        over all neighbors within the potential cut-off radius, of each bulk and
        interface reference particle type ('all', 'A', or 'B').
        '''
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]                               # Find positions of type 0 particles
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Neighbor list query arguments to find interacting particles
        #query_args = dict(mode='ball', r_min = 0.1, r_max=45)
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=len(pos_bulk)-1)

        # Locate potential neighbor particles of all types in the dense phase
        system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_bulk))
        import time

        t = time.time()
        #Checks for all neighbors around type 'A' particles within bulk
        allall_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_bulk), query_args).toNeighborList()
        elapsed = time.time() - t

        # Calculate interparticle separation distances between A reference particles and all neighbors within bulk
        bulk_lats = self.utility_functs.sep_dist_arr(pos_bulk[allall_bulk_nlist.point_indices], pos_bulk[allall_bulk_nlist.query_point_indices])
        elapsed = time.time() - t
        

        k_arr = np.linspace(0, 1, num=10)

        ssf_all = np.zeros(len(k_arr))
        
        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        allall_bulk_neigh_ind = np.array([], dtype=int)
        allall_bulk_num_neigh = np.array([])
        allall_bulk_dot = np.array([])

        ssf_all_sum = np.zeros(len(k_arr))
        #bulk_lats = np.reshape(bulk_lats, (1, len(bulk_lats)))
        #k_arr = np.reshape(k_arr, (len(k_arr), 1))
        #coeff = np.matmul(k_arr, bulk_lats)

        #ssf_all_sum_final = (np.sum(np.cos(coeff))**2 + np.sum(np.sin(-k_arr[k] * bulk_lats))**2)**0.5

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        """
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        allall_bulk_neigh_ind = np.array([], dtype=int)
        allall_bulk_num_neigh = np.array([])
        
        ssf_all_sum2 = np.zeros((len(k_arr), len(pos_bulk)))
        ssf_all2 = np.zeros((len(k_arr), len(pos_bulk)))

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_bulk)):
            if i in allall_bulk_nlist.query_point_indices:
                if i not in allall_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(allall_bulk_nlist.query_point_indices==i)[0]
                    for k in range(0, len(k_arr)):
                        ssf_all_sum2[k,i] = (np.sum(np.cos(-k_arr[k] * bulk_lats[loc]))**2 + np.sum(np.sin(-k_arr[k] * bulk_lats[loc]))**2)**0.5
                        ssf_all2[k,i] = np.sum(np.exp(k_arr[k] * bulk_lats[loc] * 1j))
                    #Save nearest neighbor information to array
                    allall_bulk_num_neigh = np.append(allall_bulk_num_neigh, len(loc))
                    allall_bulk_neigh_ind = np.append(allall_bulk_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                allall_bulk_num_neigh = np.append(allall_bulk_num_neigh, 0)
                allall_bulk_neigh_ind = np.append(allall_bulk_neigh_ind, int(i))
        stop
        """
        for k in range(0, len(k_arr)):
            ssf_all_sum[k] = (np.sum(np.cos(-k_arr[k] * bulk_lats))**2 + np.sum(np.sin(-k_arr[k] * bulk_lats))**2)**0.5
            ssf_all[k] = np.sum(np.exp(k_arr[k] * bulk_lats * 1j))

        for k in range(0, len(k_arr)):

            ssf_all[k] = np.sum(np.cos(k_arr[k] * bulk_lats))**2 + np.sum(np.sin(k_arr[k] * bulk_lats))**2
            ssf_all[k] = np.abs(np.sum(np.exp(k_arr[k] * bulk_lats * 1j)))**2

        np.einsum('i,k->k', k_arr, bulk_lats)
        
        # Calculate interparticle separation distances between all reference particles and all neighbors within bulk
        

    def structure_factor_freud(self):

        #IN PROGRESS
        '''
        Note: Deprecated! This methodology is far too slow to use for large system sizes.

        Purpose: Takes the composition of each phase to compute the structure factor and average compressibility 
        of the bulk phase given the structural definition of compressibility with Freud-analysis method.

        Outputs:
        compress_dict: dictionary containing the average compressibility of each reference particle type
        ('all', 'A', or 'B') of bulk particles

        structure_factor_dict: dictionary containing the structure factor of each reference particle type
        ('all', 'A', or 'B') of bulk particles for each wavenumber

        k0_dict: dictionary containing the structure factor of each reference particle type
        ('all', 'A', or 'B') of bulk particles for k=0 wavenumber. Useful for calculating domain sizes.
        '''

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Define structure factor function
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=10, k_min=0
        )

        
        # Calculates structure factor with Freud-analysis method
        sf.compute(
            (self.sf_box, self.sf_box.wrap(pos_bulk[0:500])),
            query_points=pos_bulk[0:500],
            N_total=len(pos_bulk[0:500])
        )

    def structure_factor(self, rad_df_dict):
        '''
        Purpose: Takes the composition of each phase and uses already calculated radial
        distribution functions to compute the structure factor and average compressibility 
        of the bulk phase given the structural definition of compressibility.

        Inputs:
        rad_df_dict: dictionary containing arrays of the probability distribution function of finding
        a neighbor particle of a given species ('all', 'A', or 'B') at a given distance ('r') from
        the reference particle of a given species ('all', 'A', or 'B') for a given reference-neighbor 
        pair (i.e. all-A means all reference particles with A neighbors).

        Outputs:
        compress_dict: dictionary containing the average compressibility of each reference particle type
        ('all', 'A', or 'B') of bulk particles

        structure_factor_dict: dictionary containing the structure factor of each reference particle type
        ('all', 'A', or 'B') of bulk particles for each wavenumber

        k0_dict: dictionary containing the structure factor of each reference particle type
        ('all', 'A', or 'B') of bulk particles for k=0 wavenumber. Useful for calculating domain sizes.
        '''

        # Average number density of each species in bulk
        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)
        

        # Define input radial density function
        r_arr = rad_df_dict['r']
        g_r_allall_bulk = rad_df_dict['all-all']
        g_r_AA_bulk = rad_df_dict['A-A']
        g_r_AB_bulk = rad_df_dict['A-B']
        g_r_BA_bulk = rad_df_dict['B-A']
        g_r_BB_bulk = rad_df_dict['B-B']
        g_r_allA_bulk = rad_df_dict['all-A']
        g_r_allB_bulk = rad_df_dict['all-B']

        # Calculate partial structure factor of A neighbor particles with A reference particles in bulk using fourier transforms of g(r)
        partial_ssf_AA = (num_dens_mean_dict['A']/num_dens_mean_dict['all'])+num_dens_mean_dict['all']*(num_dens_mean_dict['A']/num_dens_mean_dict['all'])*(num_dens_mean_dict['A']/num_dens_mean_dict['all'])*np.fft.fft(np.array(g_r_AA_bulk) - 1.0)
        
        # Calculate partial structure factor of A neighbor particles with B reference particles in bulk using fourier transforms of g(r)
        partial_ssf_AB = num_dens_mean_dict['all']*(num_dens_mean_dict['B']/num_dens_mean_dict['all'])*(num_dens_mean_dict['A']/num_dens_mean_dict['all'])*np.fft.fft(np.array(g_r_AB_bulk) - 1.0)
        
        # Calculate partial structure factor of B neighbor particles with B reference particles in bulk using fourier transforms of g(r)
        partial_ssf_BB = (num_dens_mean_dict['B']/num_dens_mean_dict['all'])+num_dens_mean_dict['all']*(num_dens_mean_dict['B']/num_dens_mean_dict['all'])*(num_dens_mean_dict['B']/num_dens_mean_dict['all'])*np.fft.fft(np.array(g_r_BB_bulk) - 1.0)
        

        # Initiate empty arrays for calculating partial structure factor of each neighbor-reference activity pair in bulk
        partial_ssf_allall_arr = np.array([])
        partial_ssf_AA_arr = np.array([])
        partial_ssf_AB_arr = np.array([])
        partial_ssf_BA_arr = np.array([])
        partial_ssf_BB_arr = np.array([])

        # define wave numbers for fourier transform
        k_arr = np.linspace(0, 10, num=1000)
        
        # separation distance in distance units
        difr = r_arr[1]-r_arr[0]
        
        # Loop over all wave numbers
        for i in range(1, len(k_arr)):

            # Calculate partial structure factor of all neighbors with all reference particles in bulk
            partial_ssf_allall = np.trapz(np.array(r_arr) * (np.array(g_r_allall_bulk)-1) * np.sin(k_arr[i] * np.array(r_arr))/(k_arr[i]), x=r_arr)
            
            # Calculate partial structure factor of A neighbors with A reference particles in bulk
            partial_ssf_AA = np.trapz(np.array(r_arr) * (np.array(g_r_AA_bulk)-1) * np.sin(k_arr[i] * np.array(r_arr))/(k_arr[i]), x=r_arr)
            
            # Calculate partial structure factor of A neighbors with B reference particles in bulk
            partial_ssf_AB = np.trapz(np.array(r_arr) * (np.array(g_r_AB_bulk)-1) * np.sin(k_arr[i] * np.array(r_arr))/(k_arr[i]), x=r_arr)
            
            # Calculate partial structure factor of B neighbors with A reference particles in bulk
            partial_ssf_BA = np.trapz(np.array(r_arr) * (np.array(g_r_BA_bulk)-1) * np.sin(k_arr[i] * np.array(r_arr))/(k_arr[i]), x=r_arr)
            
            # Calculate partial structure factor of B neighbors with B reference particles in bulk
            partial_ssf_BB = np.trapz(np.array(r_arr) * (np.array(g_r_BB_bulk)-1) * np.sin(k_arr[i] * np.array(r_arr))/(k_arr[i]), x=r_arr)

            # Save and normalize partial structure factors
            partial_ssf_allall_arr=np.append( partial_ssf_allall_arr, 1 + 4 * np.pi * num_dens_mean_dict['all'] * partial_ssf_allall)
            partial_ssf_AA_arr=np.append( partial_ssf_AA_arr, 1 + 4 * np.pi * num_dens_mean_dict['all'] * partial_ssf_AA)
            partial_ssf_AB_arr=np.append( partial_ssf_AB_arr, 1 + 4 * np.pi * num_dens_mean_dict['all'] * partial_ssf_AB)
            partial_ssf_BA_arr=np.append( partial_ssf_BA_arr, 1 + 4 * np.pi * num_dens_mean_dict['all'] * partial_ssf_AB)
            partial_ssf_BB_arr=np.append( partial_ssf_BB_arr, 1 + 4 * np.pi * num_dens_mean_dict['all'] * partial_ssf_BB)

        # Calculate compressibilities of each particle type given radial density function
        compressibility_allall = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allall_bulk)-1), x=r_arr)) / (num_dens_mean_dict['all'] * ((self.peA**2)/6))
        #compressibility_AA = (1 + num_dens_mean_dict['A'] * np.trapz((np.array(g_r_AA_bulk)-1), x=r_arr)) / (num_dens_mean_dict['A'] * ((self.peA**2)/6))
        #compressibility_AB = (1 + num_dens_mean_dict['A'] * np.trapz((np.array(g_r_AB_bulk)-1), x=r_arr)) / (num_dens_mean_dict['B'] * ((self.peB**2)/6))
        #compressibility_BA = (1 + num_dens_mean_dict['B'] * np.trapz((np.array(g_r_AB_bulk)-1), x=r_arr)) / (num_dens_mean_dict['A'] * ((self.peA**2)/6))
        #compressibility_BB = (1 + num_dens_mean_dict['B'] * np.trapz((np.array(g_r_BB_bulk)-1), x=r_arr)) / (num_dens_mean_dict['B'] * ((self.peB**2)/6))
        compressibility_allA = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allA_bulk)-1), x=r_arr)) / (num_dens_mean_dict['A'] * ((self.peA**2)/6))
        compressibility_allB = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allB_bulk)-1), x=r_arr)) / (num_dens_mean_dict['B'] * ((self.peB**2)/6))

        # Create output dictionary for average compressibility of each particle type in bulk
        compress_dict = {'all': compressibility_allall, 'A': compressibility_allA, 'B': compressibility_allB}
        
        # Create output dictionary for structure factor of each reference-neighbor activity pairing in bulk
        structure_factor_dict = {'k': k_arr[1:], 'all-all': partial_ssf_allall_arr, 'all-all2': partial_ssf_allall_num,'A-A': partial_ssf_AA_arr, 'A-B': partial_ssf_AB_arr, 'B-A': partial_ssf_BA_arr, 'B-B': partial_ssf_BB_arr}
        
        # Create output dictionary for structure factor at k=0 for each reference-neighbor activity pairing in bulk
        k0_dict = {'all-all': partial_ssf_allall_arr[0], 'all-all2': partial_ssf_allall_num[0],'A-A': partial_ssf_AA_arr[0], 'A-B': partial_ssf_AB_arr[0], 'B-A': partial_ssf_BA_arr[0], 'B-B': partial_ssf_BB_arr[0]}
        
        return compress_dict, structure_factor_dict, k0_dict

    def compressibility(self, rad_df_dict, avg_num_dens = 999):
        '''
        Purpose: Takes the composition of each phase and uses already calculated radial
        distribution functions to compute the average compressibility of the bulk phase given
        the structural definition of compressibility.

        Inputs:
        rad_df_dict: dictionary containing arrays of the probability distribution function of finding
        a neighbor particle of a given species ('all', 'A', or 'B') at a given distance ('r') from
        the reference particle of a given species ('all', 'A', or 'B') for a given reference-neighbor 
        pair (i.e. all-A means all reference particles with A neighbors).

        avg_num_dens (optional): optional dictionary input that provides the average number density of 
        each species ('all', 'A', or 'B') within the bulk. If not provided, will calculate it

        Outputs:
        compress_dict: dictionary containing the average compressibility of each reference particle type
        ('all', 'A', or 'B') of bulk particles
        '''

        # If no input dictionary, calculate average number density of each species in the bulk
        if avg_num_dens==999:
            num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        # Otherwise, use input dictionary
        else:
            num_dens_mean_dict = {}
            num_dens_mean_dict['all'] = avg_num_dens['all']/avg_num_dens['count']
            num_dens_mean_dict['A'] = avg_num_dens['A']/avg_num_dens['count']
            num_dens_mean_dict['B'] = avg_num_dens['B']/avg_num_dens['count']
        

        # Define input radial density function
        r_arr = rad_df_dict['r']
        g_r_allall_bulk = rad_df_dict['all-all']
        g_r_allA_bulk = rad_df_dict['all-A']
        g_r_allB_bulk = rad_df_dict['all-B']

        # Calculate compressibility of all, A, and B particles within the bulk given g(r)
        compressibility_allall = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allall_bulk)-1), x=r_arr)) / (num_dens_mean_dict['all'] * ((self.peA**2)/6))
        compressibility_allA = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allA_bulk)-1), x=r_arr)) / (num_dens_mean_dict['A'] * ((self.peA**2)/6))
        compressibility_allB = (1 + num_dens_mean_dict['all'] * np.trapz((np.array(g_r_allB_bulk)-1), x=r_arr)) / (num_dens_mean_dict['B'] * ((self.peB**2)/6))

        # Create output dictionary for average compressibility of each particle type in bulk
        compress_dict = {'compress': {'all-all': compressibility_allall, 'all-A': compressibility_allA, 'all-B': compressibility_allB}}
        
        return compress_dict

    def angular_df(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to compute the
        interparticle separation angle between the interparticle separation vector of a given
        reference particle and one of their nearest, interacting neighbors with the x-axis
        and averages over the system to provide the probability of finding a neighbor of a given species at each
        separation distance from a reference particle of a given species within the bulk phase (i.e. radial
        distribution function, RDF)

        Outputs:
        ang_df_dict: dictionary containing arrays of the probability distribution
        function of finding a particle of a given species ('all', 'A', or 'B') at
        a given angle ('theta') formed by the x-axis and the reference particle of a given species
        ('all', 'A', or 'B') for a given reference-neighbor pair (i.e. all-A means
        all reference particles with A neighbors).
        '''
        # Calculate bulk average number density
        
        num_dens_mean_dict = self.num_dens_mean(self.area_frac_dict)

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        # Calculate interparticle distance between type A reference particle and type A neighbor
        difx_AA_bulk, dify_AA_bulk, difr_AA_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AA_bulk_nlist.point_indices], pos_A_bulk[AA_bulk_nlist.query_point_indices], difxy=True)

        # Calculate interparticle distance between type B reference particle and type A neighbor
        difx_AB_bulk, dify_AB_bulk, difr_AB_bulk = self.utility_functs.sep_dist_arr(pos_A_dense[AB_bulk_nlist.point_indices], pos_B_bulk[AB_bulk_nlist.query_point_indices], difxy=True)

        # Calculate interparticle distance between type B reference particle and type B neighbor
        difx_BB_bulk, dify_BB_bulk, difr_BB_bulk = self.utility_functs.sep_dist_arr(pos_B_dense[BB_bulk_nlist.point_indices], pos_B_bulk[BB_bulk_nlist.query_point_indices], difxy=True)

        # get angle from x-axis to type A neighboring particle given x-y distance from type A reference particle
        ang_loc_AA = self.utility_functs.shift_quadrants(difx_AA_bulk, dify_AA_bulk)

        # get angle from x-axis to type A neighboring particle given x-y distance from type B reference particle
        ang_loc_AB = self.utility_functs.shift_quadrants(difx_AB_bulk, dify_AB_bulk)

        # get angle from x-axis to type B neighboring particle given x-y distance from type B reference particle
        ang_loc_BB = self.utility_functs.shift_quadrants(difx_BB_bulk, dify_BB_bulk)

        # get angle from x-axis to type A neighboring particle given x-y distance from all reference particle
        ang_loc_Aall = np.append(ang_loc_AA, ang_loc_AB)

        # get angle from x-axis to type B neighboring particle given x-y distance from all reference particle
        ang_loc_Ball = np.append(ang_loc_BB, ang_loc_AB)

        # get angle from x-axis to all neighboring particle given x-y distance from all reference particle
        ang_loc_allall = np.append(ang_loc_Aall, ang_loc_Ball)

        # Array of angular steps for ADF
        theta=np.arange(0.0,2*np.pi,np.pi/180)

        #Initiate empty arrays for ADF of each respective activity pairing
        g_theta_allall_bulk = np.array([])
        g_theta_AA_bulk = np.array([])
        g_theta_Aall_bulk = np.array([])
        g_theta_Ball_bulk = np.array([])
        g_theta_AB_bulk = np.array([])
        g_theta_BB_bulk = np.array([])
        theta_arr = np.array([])

        # Loop over angular slices
        for m in range(1, len(theta)-2):
            diftheta = theta[m+1] - theta[m]

            # Save minimum theta from theta range
            theta_arr = np.append(theta_arr, theta[m])

            ### all-all activity pairs

            # Locate all neighboring particles within given theta range from all reference particle
            inds = np.where((ang_loc_allall>=theta[m]) & (ang_loc_allall<theta[m+1]))[0]

            # Number of all-all neighbor pairs within theta range
            rho_all = len(inds)

            # Total number of all-all particle pairs divided by bulk volume
            rho_tot_all = len(pos_bulk) * num_dens_mean_dict['all']

            # Save all-all ADF at respective theta range
            g_theta_allall_bulk = np.append(g_theta_allall_bulk, rho_all / rho_tot_all)

            ### A-all activity pairs

            # Locate A neighboring particles within given theta range from all reference particle
            inds = np.where((ang_loc_Aall>=theta[m]) & (ang_loc_Aall<theta[m+1]))[0]

            # Number of A-all neighbor pairs within theta range
            rho_a = (len(inds))

            # Total number of A-all particle pairs divided by bulk volume
            rho_tot_a = len(pos_A_bulk) * num_dens_mean_dict['all']

            # Save A-all ADF at respective theta range
            g_theta_Aall_bulk = np.append(g_theta_Aall_bulk, rho_a / rho_tot_a)

            ### B-all activity pairs

            # Locate B neighboring particles within given theta range from all reference particle
            inds = np.where((ang_loc_Ball>=theta[m]) & (ang_loc_Ball<theta[m+1]))[0]

            # Number of B-all neighbor pairs within theta range
            rho_b = (len(inds))

            # Total number of B-all particle pairs divided by bulk volume
            rho_tot_b = len(pos_B_bulk) * num_dens_mean_dict['all']

            # Save B-all ADF at respective theta range
            g_theta_Ball_bulk = np.append(g_theta_Ball_bulk, rho_b / rho_tot_b)

            ### A-A activity pairs

            # Locate A neighboring particles within given theta range from A reference particle
            inds = np.where((ang_loc_AA>=theta[m]) & (ang_loc_AA<theta[m+1]))[0]

            # Number of A-A neighbor pairs within theta range
            rho_aa = (len(inds))

            # Total number of A-A particle pairs divided by bulk volume
            rho_tot_aa = len(pos_A_bulk) * num_dens_mean_dict['A']

            # Save A-A ADF at respective theta range
            g_theta_AA_bulk = np.append(g_theta_AA_bulk, rho_aa / rho_tot_aa)

            ### A-B activity pairs

            # Locate A neighboring particles within given theta range from B reference particle
            inds = np.where((ang_loc_AB>=theta[m]) & (ang_loc_AB<theta[m+1]))[0]

            # Number of A-B neighbor pairs within theta range
            rho_ab = (len(inds))

            # Total number of A-B particle pairs divided by bulk volume
            rho_tot_ab = len(pos_B_bulk) * num_dens_mean_dict['B']

            # Save A-B ADF at respective theta range
            g_theta_AB_bulk = np.append(g_theta_AB_bulk, rho_ab / rho_tot_ab)

            ### B-B activity pairs

            # Locate B neighboring particles within given theta range from B reference particle
            inds = np.where((ang_loc_BB>=theta[m]) & (ang_loc_BB<theta[m+1]))[0]

            # Number of B-B neighbor pairs within theta range
            rho_bb = (len(inds))

            # Total number of B-B particle pairs divided by bulk volume
            rho_tot_bb = len(pos_B_bulk) * num_dens_mean_dict['B']

            # Save B-B ADF at respective theta range
            g_theta_BB_bulk = np.append(g_theta_BB_bulk, rho_bb / rho_tot_bb)

        # Normalize ADFs by total area under curve (so integrates to 1)
        g_theta_allall_bulk=g_theta_allall_bulk/(-np.trapz(theta_arr, g_theta_allall_bulk))
        g_theta_Aall_bulk=g_theta_Aall_bulk/(-np.trapz(theta_arr, g_theta_Aall_bulk))
        g_theta_Ball_bulk=g_theta_Ball_bulk/(-np.trapz(theta_arr, g_theta_Ball_bulk))
        g_theta_AA_bulk=g_theta_AA_bulk/(-np.trapz(theta_arr, g_theta_AA_bulk))
        g_theta_AB_bulk=g_theta_AB_bulk/(-np.trapz(theta_arr, g_theta_AB_bulk))
        g_theta_BB_bulk=g_theta_BB_bulk/(-np.trapz(theta_arr, g_theta_BB_bulk))

        # Create output dictionary for plotting of ADF vs theta
        ang_df_dict = {'theta': np.ndarray.tolist(theta_arr), 'all-all': np.ndarray.tolist(g_theta_allall_bulk), 'A-all': np.ndarray.tolist(g_theta_Aall_bulk), 'B-all': np.ndarray.tolist(g_theta_Ball_bulk), 'A-A': np.ndarray.tolist(g_theta_AA_bulk), 'A-B': np.ndarray.tolist(g_theta_AB_bulk), 'B-B': np.ndarray.tolist(g_theta_BB_bulk)}
        return ang_df_dict

    def centrosymmetry(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the central symmetry parameter of each
        type for each particle and averaged over all particles of each phase. Useful in
        identifying defects.

        Outputs:
        csp_stat_dict: dictionary containing the particle-based central symmetry parameter 
        for a reference particle of a given type ('all', 'A', or 'B'), averaged over all 
        particles in each phase.

        csp_plot_dict: dictionary containing information on the central symmetry parameter
        of each bulk and interface reference particle of each type ('all', 'A', or 'B').
        '''
        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]
        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max = self.r_cut)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        # Locate potential neighbor particles by type in the dense phase
        system_all_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_dense))
        
        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        allA_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        allB_bulk_nlist = system_all_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        # Calculate interparticle separation distances between A reference particles and all neighbors within bulk
        allA_difx, allA_dify, allA_difr = self.utility_functs.sep_dist_arr(pos_dense[allA_bulk_nlist.point_indices], pos_A_bulk[allA_bulk_nlist.query_point_indices], difxy=True)

        # Calculate interparticle separation distances between B reference particles and all neighbors within bulk
        allB_difx, allB_dify, allB_difr = self.utility_functs.sep_dist_arr(pos_dense[allB_bulk_nlist.point_indices], pos_B_bulk[allB_bulk_nlist.query_point_indices], difxy=True)

        #Initiate empty arrays for finding central symmetry parameter of all neighboring dense particles surrounding type A bulk particles
        A_bulk_neigh_ind = np.array([], dtype=int)
        csp_A_bulk = np.array([])

        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_bulk)):

            # If reference ID has neighbors...
            if i in allA_bulk_nlist.query_point_indices:

                # If reference ID hasn't been considered yet...
                if i not in A_bulk_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(allA_bulk_nlist.query_point_indices==i)[0]

                    # Make array of x- and y- separation distances for nearest neighbor IDs
                    allA_difx_mat_1 = allA_difx[loc] * np.ones((len(loc), len(loc)))
                    allA_dify_mat_1 = allA_dify[loc] * np.ones((len(loc), len(loc)))

                    # Make inverse array of x- and y- separation distances for nearest neighbor IDs
                    allA_difx_mat_2 = np.reshape(allA_difx[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    allA_dify_mat_2 = np.reshape(allA_dify[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    
                    # Sum normal and inverse arrays
                    allA_difx_comb = allA_difx_mat_1 + allA_difx_mat_2
                    allA_dify_comb = allA_dify_mat_1 + allA_dify_mat_2

                    # Calculate r-separation distances for combined array
                    allA_difr = (allA_difx_comb ** 2 + allA_dify_comb ** 2 ) ** 0.5

                    # Calculate central symmetry parameter for reference particle
                    csp_A_bulk = np.append(csp_A_bulk, np.sum(np.sort(allA_difr)[:,0]))

                    # Save reference particle ID 
                    A_bulk_neigh_ind = np.append(A_bulk_neigh_ind, int(i))

            else:
                #Save central symmetry parameter information to array
                csp_A_bulk = np.append(csp_A_bulk, 0)
                A_bulk_neigh_ind = np.append(A_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding central symmetry parameter of all neighboring dense particles surrounding type B bulk particles
        B_bulk_neigh_ind = np.array([], dtype=int)
        csp_B_bulk = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_bulk)):

            # If reference ID has neighbors...
            if i in allB_bulk_nlist.query_point_indices:

                # If reference ID hasn't been considered yet...
                if i not in B_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(allB_bulk_nlist.query_point_indices==i)[0]

                    # Make array of x- and y- separation distances for nearest neighbor IDs
                    allB_difx_mat_1 = allB_difx[loc] * np.ones((len(loc), len(loc)))
                    allB_dify_mat_1 = allB_dify[loc] * np.ones((len(loc), len(loc)))

                    # Make inverse array of x- and y- separation distances for nearest neighbor IDs
                    allB_difx_mat_2 = np.reshape(allB_difx[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    allB_dify_mat_2 = np.reshape(allB_dify[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    
                    # Sum normal and inverse arrays
                    allB_difx_comb = allB_difx_mat_1 + allB_difx_mat_2
                    allB_dify_comb = allB_dify_mat_1 + allB_dify_mat_2

                    # Calculate r-separation distances for combined array
                    allB_difr = (allB_difx_comb ** 2 + allB_dify_comb ** 2 ) ** 0.5

                    # Calculate central symmetry parameter for reference particle
                    csp_B_bulk = np.append(csp_B_bulk, np.sum(np.sort(allB_difr)[:,0]))

                    # Save reference particle ID 
                    B_bulk_neigh_ind = np.append(B_bulk_neigh_ind, int(i))

            else:
                #Save central symmetry parameter information to array
                csp_B_bulk = np.append(csp_B_bulk, 0)

                B_bulk_neigh_ind = np.append(B_bulk_neigh_ind, int(i))

        # Locate potential neighbor particles by type in the entire system
        system_all_int = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        allA_int_nlist = system_all_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        allB_int_nlist = system_all_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        # Calculate interparticle separation distances between A reference particles and all neighbors within bulk
        allA_difx, allA_dify, allA_difr = self.utility_functs.sep_dist_arr(self.pos[allA_int_nlist.point_indices], pos_A_int[allA_int_nlist.query_point_indices], difxy=True)

        # Calculate interparticle separation distances between B reference particles and all neighbors within bulk
        allB_difx, allB_dify, allB_difr = self.utility_functs.sep_dist_arr(self.pos[allB_int_nlist.point_indices], pos_B_int[allB_int_nlist.query_point_indices], difxy=True)

        #Initiate empty arrays for finding central symmetry parameter of all neighboring dense particles surrounding type A interface particles
        A_int_neigh_ind = np.array([], dtype=int)
        csp_A_int = np.array([])

        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_int)):

            # If reference ID has neighbors...
            if i in allA_int_nlist.query_point_indices:

                # If reference ID hasn't been considered yet...
                if i not in A_int_neigh_ind:
                    
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(allA_int_nlist.query_point_indices==i)[0]

                    # Make array of x- and y- separation distances for nearest neighbor IDs
                    allA_difx_mat_1 = allA_difx[loc] * np.ones((len(loc), len(loc)))
                    allA_dify_mat_1 = allA_dify[loc] * np.ones((len(loc), len(loc)))

                    # Make inverse array of x- and y- separation distances for nearest neighbor IDs
                    allA_difx_mat_2 = np.reshape(allA_difx[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    allA_dify_mat_2 = np.reshape(allA_dify[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))

                    # Sum normal and inverse arrays
                    allA_difx_comb = allA_difx_mat_1 + allA_difx_mat_2
                    allA_dify_comb = allA_dify_mat_1 + allA_dify_mat_2

                    # Calculate r-separation distances for combined array
                    allA_difr = (allA_difx_comb ** 2 + allA_dify_comb ** 2 ) ** 0.5

                    # Calculate central symmetry parameter for reference particle
                    csp_A_int = np.append(csp_A_int, np.sum(np.sort(allA_difr)[:,0]))

                    # Save reference particle ID 
                    A_int_neigh_ind = np.append(A_int_neigh_ind, int(i))
            else:
                #Save central symmetry parameter information to array
                csp_A_int = np.append(csp_A_int, 0)

                A_int_neigh_ind = np.append(A_int_neigh_ind, int(i))

        #Initiate empty arrays for finding central symmetry parameter of all neighboring dense particles surrounding type B interface particles
        B_int_neigh_ind = np.array([], dtype=int)
        csp_B_int = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_int)):

            # If reference ID has neighbors...
            if i in allB_int_nlist.query_point_indices:

                # If reference ID hasn't been considered yet...
                if i not in B_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(allB_int_nlist.query_point_indices==i)[0]

                    # Make array of x- and y- separation distances for nearest neighbor IDs
                    allB_difx_mat_1 = allB_difx[loc] * np.ones((len(loc), len(loc)))
                    allB_dify_mat_1 = allB_dify[loc] * np.ones((len(loc), len(loc)))

                    # Make inverse array of x- and y- separation distances for nearest neighbor IDs
                    allB_difx_mat_2 = np.reshape(allB_difx[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    allB_dify_mat_2 = np.reshape(allB_dify[loc], (len(loc), 1)) * np.ones((len(loc), len(loc)))
                    
                    # Sum normal and inverse arrays
                    allB_difx_comb = allB_difx_mat_1 + allB_difx_mat_2
                    allB_dify_comb = allB_dify_mat_1 + allB_dify_mat_2

                    # Calculate r-separation distances for combined array
                    allB_difr = (allB_difx_comb ** 2 + allB_dify_comb ** 2 ) ** 0.5

                    # Calculate central symmetry parameter for reference particle
                    csp_B_int = np.append(csp_B_int, np.sum(np.sort(allB_difr)[:,0]))

                    # Save reference particle ID 
                    B_int_neigh_ind = np.append(B_int_neigh_ind, int(i))
            else:
                #Save central symmetry parameter information to array
                csp_B_int = np.append(csp_B_int, 0)

                B_int_neigh_ind = np.append(B_int_neigh_ind, int(i))

        # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
        csp_all_bulk = np.append(csp_A_bulk, csp_B_bulk)
        all_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        all_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

        # Save neighbor, local orientational order, and position to arrays for all interface reference particles with all nearest neighbors
        csp_all_int = np.append(csp_A_int, csp_B_int)
        all_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        all_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

        # Save local orientational order for the respective activity dense phase reference particles with the respective activity nearest neighbors
        csp_A_dense = np.append(csp_A_bulk, csp_A_int)
        csp_B_dense = np.append(csp_B_bulk, csp_B_int)
        csp_all_dense = np.append(csp_all_bulk, csp_all_int)

        # Save neighbor, local orientational order, and position to arrays for A dense phase reference particles with all nearest neighbors
        all_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        all_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        A_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        A_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

        B_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        B_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        # Create output dictionary for statistical averages of total nearest neighbor numbers on each particle per phase/activity pairing
        csp_stat_dict = {'bulk': {'all': {'mean': np.mean(csp_all_bulk), 'std': np.std(csp_all_bulk)}, 'A': {'mean': np.mean(csp_A_bulk), 'std': np.std(csp_A_bulk)}, 'B': {'mean': np.mean(csp_B_bulk), 'std': np.std(csp_B_bulk)}}, 'int': {'all': {'mean': np.mean(csp_all_int), 'std': np.std(csp_all_int)}, 'A': {'mean': np.mean(csp_A_int), 'std': np.std(csp_A_int)}, 'B': {'mean': np.mean(csp_B_int), 'std': np.std(csp_B_int)}}, 'dense': {'all': {'mean': np.mean(csp_all_dense), 'std': np.std(csp_all_dense)}, 'all-A': {'mean': np.mean(csp_A_dense), 'std': np.std(csp_A_dense)}, 'all-B': {'mean': np.mean(csp_B_dense), 'std': np.std(csp_B_dense)}}}

        # Create output dictionary for plotting of nearest neighbor information of each particle per phase/activity pairing and their respective x-y locations
        csp_plot_dict = {'all': {'csp': csp_all_dense, 'x': all_dense_pos_x, 'y': all_dense_pos_y}, 'A': {'csp': csp_A_dense, 'x': A_dense_pos_x, 'y': A_dense_pos_y}, 'B': {'csp': csp_B_dense, 'x': B_dense_pos_x, 'y': B_dense_pos_y}}

        return csp_stat_dict, csp_plot_dict
        
    def nearest_neighbors(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the number of neighbors of each
        type for each particle and averaged over all particles of each phase.

        Outputs:
        neigh_stat_dict: dictionary containing the mean and standard deviation of the
        number of neighbors of each type ('all', 'A', or 'B') for a reference particle of
        a given type ('all', 'A', or 'B'), averaged over all particles in each phase.

        ori_stat_dict: dictionary containing the mean and standard deviation of the
        orientational correlation between a reference particle of
        a given type ('all', 'A', or 'B') and neighbors of each type ('all', 'A', or 'B'),
        averaged over all particles in each phase.

        neigh_plot_dict: dictionary containing information on the number of nearest
        neighbors of each bulk and interface reference particle of each type ('all', 'A', or 'B').
        '''
        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        px_A=self.px[typ0ind]
        px_A_bulk = self.px[phase_part_dict['bulk']['A']]
        px_A_int = self.px[phase_part_dict['int']['A']]
        px_A_gas = self.px[phase_part_dict['gas']['A']]
        px_A_dense = self.px[phase_part_dict['dense']['A']]

        py_A=self.py[typ0ind]
        py_A_bulk = self.py[phase_part_dict['bulk']['A']]
        py_A_int = self.py[phase_part_dict['int']['A']]
        py_A_gas = self.py[phase_part_dict['gas']['A']]
        py_A_dense = self.py[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        px_B=self.px[typ1ind]
        px_B_bulk = self.px[phase_part_dict['bulk']['B']]
        px_B_int = self.px[phase_part_dict['int']['B']]
        px_B_gas = self.px[phase_part_dict['gas']['B']]
        px_B_dense = self.px[phase_part_dict['dense']['B']]

        py_B=self.py[typ1ind]
        py_B_bulk = self.py[phase_part_dict['bulk']['B']]
        py_B_int = self.py[phase_part_dict['int']['B']]
        py_B_gas = self.py[phase_part_dict['gas']['B']]
        py_B_dense = self.py[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        px_bulk = self.px[phase_part_dict['bulk']['all']]
        px_int = self.px[phase_part_dict['int']['all']]
        px_gas = self.px[phase_part_dict['gas']['all']]
        px_dense = self.px[phase_part_dict['dense']['all']]

        py_bulk = self.py[phase_part_dict['bulk']['all']]
        py_int = self.py[phase_part_dict['int']['all']]
        py_gas = self.py[phase_part_dict['gas']['all']]
        py_dense = self.py[phase_part_dict['dense']['all']]
        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max = self.r_cut)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))
        
        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
    
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])
        AA_bulk_dot = np.array([])
        
        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_bulk)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                    AA_bulk_dot = np.append(AA_bulk_dot, np.sum((px_A_bulk[i]*px_A_dense[AA_bulk_nlist.point_indices[loc]]+py_A_bulk[i]*py_A_dense[AA_bulk_nlist.point_indices[loc]])/(((px_A_bulk[i]**2+py_A_bulk[i]**2)**0.5)*((px_A_dense[AA_bulk_nlist.point_indices[loc]]**2+py_A_dense[AA_bulk_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                AA_bulk_dot = np.append(AA_bulk_dot, 0)
                
        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])
        BA_bulk_dot = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_bulk)):
            if i in BA_bulk_nlist.query_point_indices:
                if i not in BA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
                    BA_bulk_dot = np.append(BA_bulk_dot, np.sum((px_A_bulk[i]*px_B_dense[BA_bulk_nlist.point_indices[loc]]+py_A_bulk[i]*py_B_dense[BA_bulk_nlist.point_indices[loc]])/(((px_A_bulk[i]**2+py_A_bulk[i]**2)**0.5)*((px_B_dense[BA_bulk_nlist.point_indices[loc]]**2+py_B_dense[BA_bulk_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
                BA_bulk_dot = np.append(BA_bulk_dot, 0)

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])
        AB_bulk_dot = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_bulk)):
            if i in AB_bulk_nlist.query_point_indices:
                if i not in AB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_dot = np.append(AB_bulk_dot, np.sum((px_B_bulk[i]*px_A_dense[AB_bulk_nlist.point_indices[loc]]+py_B_bulk[i]*py_A_dense[AB_bulk_nlist.point_indices[loc]])/(((px_B_bulk[i]**2+py_B_bulk[i]**2)**0.5)*((px_A_dense[AB_bulk_nlist.point_indices[loc]]**2+py_A_dense[AB_bulk_nlist.point_indices[loc]]**2)**0.5))))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
                AB_bulk_dot = np.append(AB_bulk_dot, 0)

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])
        BB_bulk_dot = np.array([])

        #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_bulk)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                    BB_bulk_dot = np.append(BB_bulk_dot, np.sum((px_B_bulk[i]*px_B_dense[BB_bulk_nlist.point_indices[loc]]+py_B_bulk[i]*py_B_dense[BB_bulk_nlist.point_indices[loc]])/(((px_B_bulk[i]**2+py_B_bulk[i]**2)**0.5)*((px_B_dense[BB_bulk_nlist.point_indices[loc]]**2+py_B_dense[BB_bulk_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                BB_bulk_dot = np.append(BB_bulk_dot, 0)
        

        # Locate potential neighbor particles by type in the entire system
        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        # Generate neighbor list of all particles (per query args) of respective type (A or B) neighboring interface phase reference particles of respective type (A or B)
        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type A interface particles
        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])
        AA_int_dot = np.array([])

        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_int)):
            if i in AA_int_nlist.query_point_indices:
                if i not in AA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
                    AA_int_dot = np.append(AA_int_dot, np.sum((px_A_int[i]*px_A[AA_int_nlist.point_indices[loc]]+py_A_int[i]*py_A[AA_int_nlist.point_indices[loc]])/(((px_A_int[i]**2+py_A_int[i]**2)**0.5)*((px_A[AA_int_nlist.point_indices[loc]]**2+py_A[AA_int_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
                AA_int_dot = np.append(AA_int_dot, 0)

        #Initiate empty arrays for finding nearest B neighboring particles surrounding type A interface particles
        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])
        AB_int_dot = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_int)):
            if i in AB_int_nlist.query_point_indices:
                if i not in AB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
                    AB_int_dot = np.append(AB_int_dot, np.sum((px_B_int[i]*px_A[AB_int_nlist.point_indices[loc]]+py_B_int[i]*py_A[AB_int_nlist.point_indices[loc]])/(((px_B_int[i]**2+py_B_int[i]**2)**0.5)*((px_A[AB_int_nlist.point_indices[loc]]**2+py_A[AB_int_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
                AB_int_dot = np.append(AB_int_dot, 0)

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type B interface particles
        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])
        BA_int_dot = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_int)):
            if i in BA_int_nlist.query_point_indices:
                if i not in BA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
                    BA_int_dot = np.append(BA_int_dot, np.sum((px_A_int[i]*px_B[BA_int_nlist.point_indices[loc]]+py_A_int[i]*py_B[BA_int_nlist.point_indices[loc]])/(((px_A_int[i]**2+py_A_int[i]**2)**0.5)*((px_B[BA_int_nlist.point_indices[loc]]**2+py_B[BA_int_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
                BA_int_dot = np.append(BA_int_dot, 0)

        # Initiate empty arrays for finding nearest B neighboring particles surrounding type B interface particles
        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])
        BB_int_dot = np.array([])

        # Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_int)):
            if i in BB_int_nlist.query_point_indices:
                if i not in BB_int_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]

                    # Save nearest neighbor information to array
                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
                    BB_int_dot = np.append(BB_int_dot, np.sum((px_B_int[i]*px_B[BB_int_nlist.point_indices[loc]]+py_B_int[i]*py_B[BB_int_nlist.point_indices[loc]])/(((px_B_int[i]**2+py_B_int[i]**2)**0.5)*((px_B[BB_int_nlist.point_indices[loc]]**2+py_B[BB_int_nlist.point_indices[loc]]**2)**0.5))))
            else:
                # Save nearest neighbor information to array
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
                BB_int_dot = np.append(BB_int_dot, 0)

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
        allB_bulk_num_neigh = AB_bulk_num_neigh + BB_bulk_num_neigh
        allB_bulk_dot = BB_bulk_dot + AB_bulk_dot
        allB_int_num_neigh = AB_int_num_neigh + BB_int_num_neigh
        allB_int_dot = BB_int_dot + AB_int_dot

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
        allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh
        allA_bulk_dot = AA_bulk_dot + BA_bulk_dot
        allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh
        allA_int_dot = AA_int_dot + BA_int_dot

        # Save neighbor and local orientational order to arrays for all B reference particles of the respective phase with all nearest neighbors
        Ball_bulk_num_neigh = np.append(BA_bulk_num_neigh, BB_bulk_num_neigh)
        Ball_bulk_dot = np.append(BA_bulk_dot, BB_bulk_dot)
        Ball_int_num_neigh = np.append(BA_int_num_neigh, BB_int_num_neigh)
        Ball_int_dot = np.append(BA_int_dot, BB_int_dot)

        # Save neighbor and local orientational order to arrays for all A reference particles of the respective phase with all nearest neighbors
        Aall_bulk_num_neigh = np.append(AB_bulk_num_neigh, AA_bulk_num_neigh)
        Aall_bulk_dot = np.append(AB_bulk_dot, AA_bulk_dot)
        Aall_int_num_neigh = np.append(AB_int_num_neigh, AA_int_num_neigh)
        Aall_int_dot = np.append(AB_int_dot, AA_int_dot)

        # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
        allall_bulk_num_neigh = np.append(allA_bulk_num_neigh, allB_bulk_num_neigh)
        allall_bulk_dot = np.append(allA_bulk_dot, allB_bulk_dot)
        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

        # Save neighbor, local orientational order, and position to arrays for all interface reference particles with all nearest neighbors
        allall_int_num_neigh = np.append(allA_int_num_neigh, allB_int_num_neigh)
        allall_int_dot = np.append(allA_int_dot, allB_int_dot)
        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

        # Average local orientational order over all neighbors for each B bulk reference particle
        for i in range(0, len(allB_bulk_dot)):
            if allB_bulk_num_neigh[i]>0:
                allB_bulk_dot[i] = allB_bulk_dot[i]/allB_bulk_num_neigh[i]

        # Average local orientational order over all neighbors for each A bulk reference particle
        for i in range(0, len(allA_bulk_dot)):
            if allA_bulk_num_neigh[i]>0:
                allA_bulk_dot[i] = allA_bulk_dot[i]/allA_bulk_num_neigh[i]

        # Average local orientational order over all neighbors for each all bulk reference particle
        for i in range(0, len(allall_bulk_dot)):
            if allall_bulk_num_neigh[i]>0:
                allall_bulk_dot[i] = allall_bulk_dot[i]/allall_bulk_num_neigh[i]

        # Average local orientational order over aAll neighbors for each all bulk reference particle
        for i in range(0, len(Aall_bulk_dot)):
            if Aall_bulk_num_neigh[i]>0:
                Aall_bulk_dot[i] = Aall_bulk_dot[i]/Aall_bulk_num_neigh[i]

        # Average local orientational order over B neighbors for each all bulk reference particle
        for i in range(0, len(Ball_bulk_dot)):
            if Ball_bulk_num_neigh[i]>0:
                Ball_bulk_dot[i] = Ball_bulk_dot[i]/Ball_bulk_num_neigh[i]

        # Average local orientational order over B neighbors for each B bulk reference particle
        for i in range(0, len(BB_bulk_dot)):
            if BB_bulk_num_neigh[i]>0:
                BB_bulk_dot[i] = BB_bulk_dot[i]/BB_bulk_num_neigh[i]

        # Average local orientational order over A neighbors for each A bulk reference particle
        for i in range(0, len(AA_bulk_dot)):
            if AA_bulk_num_neigh[i]>0:
                AA_bulk_dot[i] = AA_bulk_dot[i]/AA_bulk_num_neigh[i]

        # Average local orientational order over A neighbors for each B bulk reference particle
        for i in range(0, len(AB_bulk_dot)):
            if AB_bulk_num_neigh[i]>0:
                AB_bulk_dot[i] = AB_bulk_dot[i]/AB_bulk_num_neigh[i]

        # Average local orientational order over B neighbors for each A bulk reference particle
        for i in range(0, len(BA_bulk_dot)):
            if BA_bulk_num_neigh[i]>0:
                BA_bulk_dot[i] = BA_bulk_dot[i]/BA_bulk_num_neigh[i]

        # Average local orientational order over all neighbors for each B interface reference particle
        for i in range(0, len(allB_int_dot)):
            if allB_int_num_neigh[i]>0:
                allB_int_dot[i] = allB_int_dot[i]/allB_int_num_neigh[i]

        # Average local orientational order over all neighbors for each A interface reference particle
        for i in range(0, len(allA_int_dot)):
            if allA_int_num_neigh[i]>0:
                allA_int_dot[i] = allA_int_dot[i]/allA_int_num_neigh[i]

        # Average local orientational order over all neighbors for each all interface reference particle
        for i in range(0, len(allall_int_dot)):
            if allall_int_num_neigh[i]>0:
                allall_int_dot[i] = allall_int_dot[i]/allall_int_num_neigh[i]

        # Average local orientational order over A neighbors for each all interface reference particle
        for i in range(0, len(Aall_int_dot)):
            if Aall_int_num_neigh[i]>0:
                Aall_int_dot[i] = Aall_int_dot[i]/Aall_int_num_neigh[i]

        # Average local orientational order over B neighbors for each all interface reference particle
        for i in range(0, len(Ball_int_dot)):
            if Ball_int_num_neigh[i]>0:
                Ball_int_dot[i] = Ball_int_dot[i]/Ball_int_num_neigh[i]

        # Average local orientational order over B neighbors for each B interface reference particle
        for i in range(0, len(BB_int_dot)):
            if BB_int_num_neigh[i]>0:
                BB_int_dot[i] = BB_int_dot[i]/BB_int_num_neigh[i]

        # Average local orientational order over A neighbors for each A interface reference particle
        for i in range(0, len(AA_int_dot)):
            if AA_int_num_neigh[i]>0:
                AA_int_dot[i] = AA_int_dot[i]/AA_int_num_neigh[i]

        # Average local orientational order over A neighbors for each B interface reference particle
        for i in range(0, len(AB_int_dot)):
            if AB_int_num_neigh[i]>0:
                AB_int_dot[i] = AB_int_dot[i]/AB_int_num_neigh[i]

        # Average local orientational order over B neighbors for each A interface reference particle
        for i in range(0, len(BA_int_dot)):
            if BA_int_num_neigh[i]>0:
                BA_int_dot[i] = BA_int_dot[i]/BA_int_num_neigh[i]

        # Save local orientational order for the respective activity dense phase reference particles with the respective activity nearest neighbors
        AA_dense_num_neigh = np.append(AA_bulk_num_neigh, AA_int_num_neigh)
        AB_dense_num_neigh = np.append(AB_bulk_num_neigh, AB_int_num_neigh)
        BA_dense_num_neigh = np.append(BA_bulk_num_neigh, BA_int_num_neigh)
        BB_dense_num_neigh = np.append(BB_bulk_num_neigh, BB_int_num_neigh)

        # Save number of nearest neighbors for the respective activity dense phase reference particles with the respective activity nearest neighbors
        AA_dense_dot = np.append(AA_bulk_dot, AA_int_dot)
        AB_dense_dot = np.append(AB_bulk_dot, AB_int_dot)
        BA_dense_dot = np.append(BA_bulk_dot, BA_int_dot)
        BB_dense_dot = np.append(BB_bulk_dot, BB_int_dot)

        # Save neighbor, local orientational order, and position to arrays for A dense phase reference particles with all nearest neighbors
        Aall_dense_num_neigh = np.append(Aall_bulk_num_neigh, Aall_int_num_neigh)
        Aall_dense_dot = np.append(Aall_bulk_dot, Aall_int_dot)
        Aall_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Aall_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        # Save neighbor, local orientational order, and position to arrays for B dense phase reference particles with all nearest neighbors
        Ball_dense_num_neigh = np.append(Ball_bulk_num_neigh, Ball_int_num_neigh)
        Ball_dense_dot = np.append(Ball_bulk_dot, Ball_int_dot)
        Ball_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Ball_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])

        # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with A nearest neighbors
        allA_dense_num_neigh = np.append(allA_bulk_num_neigh, allA_int_num_neigh)
        allA_dense_dot = np.append(allA_bulk_dot, allA_int_dot)
        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

        # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with B nearest neighbors
        allB_dense_num_neigh = np.append(allB_bulk_num_neigh, allB_int_num_neigh)
        allB_dense_dot = np.append(allB_bulk_dot, allB_int_dot)
        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with all nearest neighbors
        allall_dense_num_neigh = np.append(allall_bulk_num_neigh, allall_int_num_neigh)
        allall_dense_dot = np.append(allall_bulk_dot, allall_int_dot)
        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

        A_def_id = np.where(allA_bulk_num_neigh>0)[0]
        B_def_id = np.where(allB_bulk_num_neigh>0)[0]

        AA_prob_bulk=AA_bulk_num_neigh[A_def_id]/allA_bulk_num_neigh[A_def_id]
        AB_prob_bulk=AB_bulk_num_neigh[B_def_id]/allB_bulk_num_neigh[B_def_id]
        BA_prob_bulk=BA_bulk_num_neigh[A_def_id]/allA_bulk_num_neigh[A_def_id]
        BB_prob_bulk=BB_bulk_num_neigh[B_def_id]/allB_bulk_num_neigh[B_def_id]

        A_def_id = np.where(allA_int_num_neigh>0)[0]
        B_def_id = np.where(allB_int_num_neigh>0)[0]

        AA_prob_int=AA_int_num_neigh[A_def_id]/allA_int_num_neigh[A_def_id]
        AB_prob_int=AB_int_num_neigh[B_def_id]/allB_int_num_neigh[B_def_id]
        BA_prob_int=BA_int_num_neigh[A_def_id]/allA_int_num_neigh[A_def_id]
        BB_prob_int=BB_int_num_neigh[B_def_id]/allB_int_num_neigh[B_def_id]

        A_def_id = np.where(allA_dense_num_neigh>0)[0]
        B_def_id = np.where(allB_dense_num_neigh>0)[0]

        AA_prob_dense=AA_dense_num_neigh[A_def_id]/allA_dense_num_neigh[A_def_id]
        AB_prob_dense=AB_dense_num_neigh[B_def_id]/allB_dense_num_neigh[B_def_id]
        BA_prob_dense=BA_dense_num_neigh[A_def_id]/allA_dense_num_neigh[A_def_id]
        BB_prob_dense=BB_dense_num_neigh[B_def_id]/allB_dense_num_neigh[B_def_id]

        chi_s_bulk = len(pos_A_bulk)/len(pos_bulk)
        chi_f_bulk = len(pos_B_bulk)/len(pos_bulk)

        chi_s_int = len(pos_A_int)/len(pos_int)
        chi_f_int = len(pos_B_int)/len(pos_int)

        chi_s_dense = len(pos_A_dense)/len(pos_dense)
        chi_f_dense = len(pos_B_dense)/len(pos_dense)
        
        if chi_s_bulk>0:
            deg_seg_AA_bulk = AA_prob_bulk/chi_s_bulk
            deg_seg_AB_bulk = AB_prob_bulk/chi_s_bulk
        else:
            deg_seg_AA_bulk = 0
            deg_seg_AB_bulk = 0
        if chi_f_bulk>0:    
            deg_seg_BA_bulk = BA_prob_bulk/chi_f_bulk
            deg_seg_BB_bulk = BB_prob_bulk/chi_f_bulk
        else:
            deg_seg_BA_bulk = 0
            deg_seg_BB_bulk = 0

        if chi_s_int>0:
            deg_seg_AA_int = AA_prob_int/chi_s_int
            deg_seg_AB_int = AB_prob_int/chi_s_int
        else:
            deg_seg_AA_int = 0
            deg_seg_AB_int = 0
        if chi_f_int>0:
            deg_seg_BA_int = BA_prob_int/chi_f_int
            deg_seg_BB_int = BB_prob_int/chi_f_int
        else:
            deg_seg_BA_int = 0
            deg_seg_BB_int = 0

        if chi_s_dense>0:
            deg_seg_AA_dense = AA_prob_dense/chi_s_dense
            deg_seg_AB_dense = AB_prob_dense/chi_s_dense
        else:
            deg_seg_AA_dense = 0
            deg_seg_AB_dense = 0
        if chi_f_dense>0:
            deg_seg_BA_dense = BA_prob_dense/chi_f_dense
            deg_seg_BB_dense = BB_prob_dense/chi_f_dense
        else:
            deg_seg_BA_dense = 0
            deg_seg_BB_dense = 0

        # Create output dictionary for statistical averages of total nearest neighbor numbers on each particle per phase/activity pairing
        neigh_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_num_neigh), 'std': np.std(allall_bulk_num_neigh)}, 'all-A': {'mean': np.mean(allA_bulk_num_neigh), 'std': np.std(allA_bulk_num_neigh)}, 'all-B': {'mean': np.mean(allB_bulk_num_neigh), 'std': np.std(allB_bulk_num_neigh)}, 'A-A': {'mean': np.mean(AA_bulk_num_neigh), 'std': np.std(AA_bulk_num_neigh), 'prob_mean': np.mean(AA_prob_bulk), 'prob_std': np.std(AA_prob_bulk), 'deg_seg_mean': np.mean(deg_seg_AA_bulk), 'deg_seg_std': np.std(deg_seg_AA_bulk)}, 'A-B': {'mean': np.mean(AB_bulk_num_neigh), 'std': np.std(AB_bulk_num_neigh), 'prob_mean': np.mean(AB_prob_bulk), 'prob_std': np.std(AB_prob_bulk), 'deg_seg_mean': np.mean(deg_seg_AB_bulk), 'deg_seg_std': np.std(deg_seg_AB_bulk)}, 'B-A': {'mean': np.mean(BA_bulk_num_neigh), 'std': np.std(BA_bulk_num_neigh), 'prob_mean': np.mean(BA_prob_bulk), 'prob_std': np.std(BA_prob_bulk), 'deg_seg_mean': np.mean(deg_seg_BA_bulk), 'deg_seg_std': np.std(deg_seg_BA_bulk)}, 'B-B': {'mean': np.mean(BB_bulk_num_neigh), 'std': np.std(BB_bulk_num_neigh), 'prob_mean': np.mean(BB_prob_bulk), 'prob_std': np.std(BB_prob_bulk), 'deg_seg_mean': np.mean(deg_seg_BB_bulk), 'deg_seg_std': np.std(deg_seg_BB_bulk)}}, 'int': {'all-all': {'mean': np.mean(allall_int_num_neigh), 'std': np.std(allall_int_num_neigh)}, 'all-A': {'mean': np.mean(allA_int_num_neigh), 'std': np.std(allA_int_num_neigh)}, 'all-B': {'mean': np.mean(allB_int_num_neigh), 'std': np.std(allB_int_num_neigh)}, 'A-A': {'mean': np.mean(AA_int_num_neigh), 'std': np.std(AA_int_num_neigh), 'prob_mean': np.mean(AA_prob_int), 'prob_std': np.std(AA_prob_int), 'deg_seg_mean': np.mean(deg_seg_AA_int), 'deg_seg_std': np.std(deg_seg_AA_int)}, 'A-B': {'mean': np.mean(AB_int_num_neigh), 'std': np.std(AB_int_num_neigh), 'prob_mean': np.mean(AB_prob_int), 'prob_std': np.std(AB_prob_int), 'deg_seg_mean': np.mean(deg_seg_AB_int), 'deg_seg_std': np.std(deg_seg_AB_int)}, 'B-A': {'mean': np.mean(BA_int_num_neigh), 'std': np.std(BA_int_num_neigh), 'prob_mean': np.mean(BA_prob_int), 'prob_std': np.std(BA_prob_int), 'deg_seg_mean': np.mean(deg_seg_BA_int), 'deg_seg_std': np.std(deg_seg_BA_int)}, 'B-B': {'mean': np.mean(BB_int_num_neigh), 'std': np.std(BB_int_num_neigh), 'prob_mean': np.mean(BB_prob_int), 'prob_std': np.std(BB_prob_int), 'deg_seg_mean': np.mean(deg_seg_BB_int), 'deg_seg_std': np.std(deg_seg_BB_int)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_num_neigh), 'std': np.std(allall_dense_num_neigh)}, 'all-A': {'mean': np.mean(allA_dense_num_neigh), 'std': np.std(allA_dense_num_neigh)}, 'all-B': {'mean': np.mean(allB_dense_num_neigh), 'std': np.std(allB_dense_num_neigh)}, 'A-A': {'mean': np.mean(AA_dense_num_neigh), 'std': np.std(AA_dense_num_neigh), 'prob_mean': np.mean(AA_prob_dense), 'prob_std': np.std(AA_prob_dense), 'deg_seg_mean': np.mean(deg_seg_AA_dense), 'deg_seg_std': np.std(deg_seg_AA_dense)}, 'A-B': {'mean': np.mean(AB_dense_num_neigh), 'std': np.std(AB_dense_num_neigh), 'prob_mean': np.mean(AB_prob_dense), 'prob_std': np.std(AB_prob_dense), 'deg_seg_mean': np.mean(deg_seg_AB_dense), 'deg_seg_std': np.std(deg_seg_AB_dense)}, 'B-A': {'mean': np.mean(BA_dense_num_neigh), 'std': np.std(BA_dense_num_neigh), 'prob_mean': np.mean(BA_prob_dense), 'prob_std': np.std(BA_prob_dense), 'deg_seg_mean': np.mean(deg_seg_BA_dense), 'deg_seg_std': np.std(deg_seg_BA_dense)}, 'B-B': {'mean': np.mean(BB_dense_num_neigh), 'std': np.std(BB_dense_num_neigh), 'prob_mean': np.mean(BB_prob_dense), 'prob_std': np.std(BB_prob_dense), 'deg_seg_mean': np.mean(deg_seg_BB_dense), 'deg_seg_std': np.std(deg_seg_BB_dense)}}}

        # Create output dictionary for statistical averages of total nearest neighbor orientational correlation on each particle per phase/activity pairing
        ori_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_dot), 'std': np.std(allall_bulk_dot)}, 'all-A': {'mean': np.mean(allA_bulk_dot), 'std': np.std(allA_bulk_dot)}, 'all-B': {'mean': np.mean(allB_bulk_dot), 'std': np.std(allB_bulk_dot)}, 'A-A': {'mean': np.mean(AA_bulk_dot), 'std': np.std(AA_bulk_dot)}, 'A-B': {'mean': np.mean(AB_bulk_dot), 'std': np.std(AB_bulk_dot)}, 'B-B': {'mean': np.mean(BB_bulk_dot), 'std': np.std(BB_bulk_dot)}}, 'int': {'all-all': {'mean': np.mean(allall_int_dot), 'std': np.std(allall_int_dot)}, 'all-A': {'mean': np.mean(allA_int_dot), 'std': np.std(allA_int_dot)}, 'all-B': {'mean': np.mean(allB_int_dot), 'std': np.std(allB_int_dot)}, 'A-A': {'mean': np.mean(AA_int_dot), 'std': np.std(AA_int_dot)}, 'A-B': {'mean': np.mean(AB_int_dot), 'std': np.std(AB_int_dot)}, 'B-B': {'mean': np.mean(BB_int_dot), 'std': np.std(BB_int_dot)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_dot), 'std': np.std(allall_dense_dot)}, 'all-A': {'mean': np.mean(allA_dense_dot), 'std': np.std(allA_dense_dot)}, 'all-B': {'mean': np.mean(allB_dense_dot), 'std': np.std(allB_dense_dot)}, 'A-A': {'mean': np.mean(AA_dense_dot), 'std': np.std(AA_dense_dot)}, 'A-B': {'mean': np.mean(AB_dense_dot), 'std': np.std(AB_dense_dot)}, 'B-B': {'mean': np.mean(BB_dense_dot), 'std': np.std(BB_dense_dot)}}}

        # Create output dictionary for plotting of nearest neighbor information of each particle per phase/activity pairing and their respective x-y locations
        neigh_plot_dict = {'all-all': {'neigh': allall_dense_num_neigh, 'ori': allall_dense_dot, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}, 'all-A': {'neigh': allA_dense_num_neigh, 'ori': allA_dense_dot, 'x': allA_dense_pos_x, 'y': allA_dense_pos_y}, 'all-B': {'neigh': allB_dense_num_neigh, 'ori': allB_dense_dot, 'x': allB_dense_pos_x, 'y': allB_dense_pos_y}, 'A-all': {'neigh': Aall_dense_num_neigh, 'ori': Aall_dense_dot, 'x': Aall_dense_pos_x, 'y': Aall_dense_pos_y}, 'B-all': {'neigh': Ball_dense_num_neigh, 'ori': Ball_dense_dot, 'x': Ball_dense_pos_x, 'y': Ball_dense_pos_y}, 'A-A': {'neigh': AA_dense_num_neigh, 'ori': AA_dense_dot, 'x': pos_A_dense[:,0], 'y': pos_A_dense[:,1]}, 'A-B': {'neigh': AB_dense_num_neigh, 'ori': AB_dense_dot, 'x': pos_B_dense[:,0], 'y': pos_B_dense[:,1]}, 'B-A': {'neigh': BA_dense_num_neigh, 'ori': BA_dense_dot, 'x': pos_A_dense[:,0], 'y': pos_A_dense[:,1]}, 'B-B': {'neigh': BB_dense_num_neigh, 'ori': BB_dense_dot, 'x': pos_B_dense[:,0], 'y': pos_B_dense[:,1]}}

        return neigh_stat_dict, ori_stat_dict, neigh_plot_dict
    def local_gas_density(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the local density for various search
        distances of each type for each particle and averaged over all particles in the gas

        Outputs:
        local_gas_dens_stat_dict: dictionary containing the local density for various search
        distances of of each type ('all', 'A', or 'B') for a reference particle of a given 
        type ('all', 'A', or 'B'), averaged over all particles in the gas.

        local_gas_dens_plot_dict: dictionary containing the local density of each particle 
        for various search distances of of each type ('all', 'A', or 'B') for a reference 
        particle of a given type ('all', 'A', or 'B') in the gas.
        '''

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        gas_area = phase_count_dict['gas'] * (self.sizeBin_x * self.sizeBin_y)
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)
        int_area = phase_count_dict['int'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        pos_gas_int_x = np.append(pos_gas[:,0], pos_int[:,0])
        pos_gas_int_y = np.append(pos_gas[:,1], pos_int[:,1])
        pos_gas_int_z = np.append(pos_gas[:,2], pos_int[:,2])
        pos_gas_int = np.array([pos_gas_int_x, pos_gas_int_y, pos_gas_int_z])
        pos_gas_int = np.reshape(pos_gas_int, (np.shape(pos_gas_int)[1], np.shape(pos_gas_int)[0]))


        pos_A_gas_int_x = np.append(pos_A_gas[:,0], pos_A_int[:,0])
        pos_A_gas_int_y = np.append(pos_A_gas[:,1], pos_A_int[:,1])
        pos_A_gas_int_z = np.append(pos_A_gas[:,2], pos_A_int[:,2])
        pos_A_gas_int = np.array([pos_A_gas_int_x, pos_A_gas_int_y, pos_A_gas_int_z])
        pos_A_gas_int = np.reshape(pos_A_gas_int, (np.shape(pos_A_gas_int)[1], np.shape(pos_A_gas_int)[0]))

        pos_B_gas_int_x = np.append(pos_B_gas[:,0], pos_B_int[:,0])
        pos_B_gas_int_y = np.append(pos_B_gas[:,1], pos_B_int[:,1])
        pos_B_gas_int_z = np.append(pos_B_gas[:,2], pos_B_int[:,2])
        pos_B_gas_int = np.array([pos_B_gas_int_x, pos_B_gas_int_y, pos_B_gas_int_z])
        pos_B_gas_int = np.reshape(pos_B_gas_int, (np.shape(pos_B_gas_int)[1], np.shape(pos_B_gas_int)[0]))
        

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around A reference particles
        allA_local_dens_mean_arr = []
        allA_local_dens_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around B reference particles
        allB_local_dens_mean_arr = []
        allB_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around all reference particles
        Aall_local_dens_mean_arr = []
        Aall_local_dens_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around all reference particles
        Ball_local_dens_mean_arr = []
        Ball_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around all reference particles
        allall_local_dens_mean_arr = []
        allall_local_dens_std_arr = []
        

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around A reference particles
        AA_local_dens_mean_arr = []
        AA_local_dens_std_arr = []
        AA_prob_std_arr = []
        AA_prob_mean_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around B reference particles
        AB_local_dens_mean_arr = []
        AB_local_dens_std_arr = []
        AB_prob_std_arr = []
        AB_prob_mean_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around A reference particles
        BA_local_dens_mean_arr = []
        BA_local_dens_std_arr = []
        BA_prob_std_arr = []
        BA_prob_mean_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around B reference particles
        BB_local_dens_mean_arr = []
        BB_local_dens_std_arr = []
        BB_prob_std_arr = []
        BB_prob_mean_arr = []


        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_gas_num_neigh = np.zeros(len(pos_A_gas))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_gas_num_neigh = np.zeros(len(pos_B_gas))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_gas_num_neigh = np.zeros(len(pos_A_gas))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_gas_num_neigh = np.zeros(len(pos_B_gas))

        # Search distance for neighbors in local density calculation
        rad_dist = [0, self.r_cut, 2*self.r_cut, 3*self.r_cut, 4*self.r_cut, 5*self.r_cut]
        
        # Loop over search distances
        for j in range(1, len(rad_dist)):

            #Initiate empty arrays for tracking which particles have been analyzed when finding local neighbors
            AA_gas_neigh_ind = np.array([], dtype=int)
            AB_gas_neigh_ind = np.array([], dtype=int)
            BA_gas_neigh_ind = np.array([], dtype=int)
            BB_gas_neigh_ind = np.array([], dtype=int)

            #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
            AA_gas_neigh_ind = np.zeros(len(pos_A_gas))

            #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
            AB_gas_neigh_ind = np.zeros(len(pos_B_gas))

            #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
            BA_gas_neigh_ind = np.zeros(len(pos_A_gas))

            #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
            BB_gas_neigh_ind = np.zeros(len(pos_B_gas))

            # List of query arguments for neighbor list caculation
            query_args = dict(mode='ball', r_min = rad_dist[j-1]+0.001, r_max = rad_dist[j])#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

            # Locate potential neighbor particles by type in the dense phase
            system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_gas_int))
            system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_gas_int))
            
            # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
            AA_gas_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
            AB_gas_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()
            BA_gas_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
            BB_gas_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()
        
            #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_gas)):
                if i in AA_gas_nlist.query_point_indices:
                    if i not in AA_gas_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AA_gas_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AA_gas_num_neigh[i] += len(loc)
                        AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))

            #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_gas)):
                if i in BA_gas_nlist.query_point_indices:
                    if i not in BA_gas_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BA_gas_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BA_gas_num_neigh[i] += len(loc)
                        BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))

            #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_gas)):
                if i in AB_gas_nlist.query_point_indices:
                    if i not in AB_gas_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AB_gas_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AB_gas_num_neigh[i] += len(loc)
                        AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))

            #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_gas)):
                if i in BB_gas_nlist.query_point_indices:
                    if i not in BB_gas_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BB_gas_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BB_gas_num_neigh[i] += len(loc)
                        BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))

            slice_area = (np.pi*rad_dist[j]**2)
            # Local density of A neighbor particles around A reference particles in bulk
            AA_gas_local_dens = AA_gas_num_neigh / slice_area

            # Local density of A neighbor particles around B reference particles in bulk
            AB_gas_local_dens = AB_gas_num_neigh / slice_area

            # Local density of B neighbor particles around A reference particles in bulk
            BA_gas_local_dens = BA_gas_num_neigh / slice_area

            # Local density of B neighbor particles around B reference particles in bulk
            BB_gas_local_dens = BB_gas_num_neigh / slice_area
            
            # Save neighbor and local orientational order to arrays for all B reference particles of the respective phase with all nearest neighbors
            Ball_gas_local_dens= np.append(BA_gas_local_dens, BB_gas_local_dens)

            # Save neighbor and local orientational order to arrays for all A reference particles of the respective phase with all nearest neighbors
            Aall_gas_local_dens = np.append(AA_gas_local_dens, AB_gas_local_dens)

            # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
            allB_gas_num_neigh = AB_gas_num_neigh + BB_gas_num_neigh
            allB_gas_local_dens = allB_gas_num_neigh / slice_area

            # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
            allA_gas_num_neigh = AA_gas_num_neigh + BA_gas_num_neigh
            allA_gas_local_dens = allA_gas_num_neigh / slice_area

            # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
            allall_gas_num_neigh = np.append(allA_gas_num_neigh, allB_gas_num_neigh)
            allall_gas_local_dens = allall_gas_num_neigh / slice_area

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around A reference particles in dense phase
            AA_local_dens_mean_arr.append(np.mean(AA_gas_local_dens))
            AA_local_dens_std_arr.append(np.std(AA_gas_local_dens))
            AA_prob_std_arr.append(np.std(AA_gas_local_dens/allA_gas_local_dens))
            AA_prob_mean_arr.append(np.mean(AA_gas_local_dens/allA_gas_local_dens))
            AA_gas_local_dens_inhomog = (AA_gas_local_dens - AA_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around B reference particles in dense phase
            AB_local_dens_mean_arr.append(np.mean(AB_gas_local_dens))
            AB_local_dens_std_arr.append(np.std(AB_gas_local_dens))
            AB_prob_std_arr.append(np.std(AB_gas_local_dens/allB_gas_local_dens))
            AB_prob_mean_arr.append(np.mean(AB_gas_local_dens/allB_gas_local_dens))
            AB_gas_local_dens_inhomog = (AB_gas_local_dens - AB_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around A reference particles in dense phase
            BA_local_dens_mean_arr.append(np.mean(BA_gas_local_dens))
            BA_local_dens_std_arr.append(np.std(BA_gas_local_dens))
            
            BA_gas_local_dens_inhomog = (BA_gas_local_dens - BA_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around B reference particles in dense phase
            BB_local_dens_mean_arr.append(np.mean(BB_gas_local_dens))
            BB_local_dens_std_arr.append(np.std(BB_gas_local_dens))
            BB_prob_std_arr.append(np.std(BB_gas_local_dens/allB_gas_local_dens))
            BB_prob_mean_arr.append(np.mean(BB_gas_local_dens/allB_gas_local_dens))
            BB_gas_local_dens_inhomog = (BB_gas_local_dens - BB_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles in dense phase
            allA_local_dens_mean_arr.append(np.mean(allA_gas_local_dens))
            allA_local_dens_std_arr.append(np.std(allA_gas_local_dens))
            allA_gas_local_dens_inhomog = (allA_gas_local_dens - allA_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles in dense phase
            allB_local_dens_mean_arr.append(np.mean(allB_gas_local_dens))
            allB_local_dens_std_arr.append(np.std(allB_gas_local_dens))
            allB_gas_local_dens_inhomog = (allB_gas_local_dens - allB_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles in dense phase
            Aall_local_dens_mean_arr.append(np.mean(Aall_gas_local_dens))
            Aall_local_dens_std_arr.append(np.std(Aall_gas_local_dens))
            Aall_gas_local_dens_inhomog = (Aall_gas_local_dens - Aall_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles in dense phase
            Ball_local_dens_mean_arr.append(np.mean(Ball_gas_local_dens))
            Ball_local_dens_std_arr.append(np.std(Ball_gas_local_dens))
            Ball_gas_local_dens_inhomog = (Ball_gas_local_dens - Ball_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around all reference particles in dense phase
            allall_local_dens_mean_arr.append(np.mean(allall_gas_local_dens))
            allall_local_dens_std_arr.append(np.std(allall_gas_local_dens))
            allall_gas_local_dens_inhomog = (allall_gas_local_dens - allall_local_dens_mean_arr[-1])**2

            # If search distance given, then prepare data for plotting!
            if rad_dist[j]==2*self.r_cut:

                # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
                allall_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
                allall_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])

                # Save neighbor, local orientational order, and position to arrays for A dense phase reference particles with all nearest neighbors
                Aall_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
                Aall_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])

                # Save neighbor, local orientational order, and position to arrays for B dense phase reference particles with all nearest neighbors
                Ball_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
                Ball_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with A nearest neighbors
                allA_gas_pos_x = pos_A_gas[:,0]
                allA_gas_pos_y = pos_A_gas[:,1]

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with B nearest neighbors
                allB_gas_pos_x = pos_B_gas[:,0]
                allB_gas_pos_y = pos_B_gas[:,1]

                # Create output dictionary for single particle values of local density per phase/activity pairing for plotting
                local_gas_dens_plot_dict = {'all-all': {'dens': allall_gas_local_dens, 'homo': allall_gas_local_dens_inhomog, 'pos_x': allall_gas_pos_x, 'pos_y': allall_gas_pos_y}, 'all-A': {'dens': allA_gas_local_dens, 'homo': allA_gas_local_dens_inhomog, 'pos_x': allA_gas_pos_x, 'pos_y': allA_gas_pos_y}, 'all-B': {'dens': allB_gas_local_dens, 'homo': allB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'A-all': {'dens': Aall_gas_local_dens, 'homo': Aall_gas_local_dens_inhomog, 'pos_x': Aall_gas_pos_x, 'pos_y': Aall_gas_pos_y}, 'B-all': {'dens': Ball_gas_local_dens, 'homo': Ball_gas_local_dens_inhomog, 'pos_x': Ball_gas_pos_x, 'pos_y': Ball_gas_pos_y}, 'A-A': {'dens': AA_gas_local_dens, 'homo': AA_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'A-B': {'dens': AB_gas_local_dens, 'homo': AB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'B-A': {'dens': BA_gas_local_dens, 'homo': BA_gas_local_dens_inhomog, 'pos_x': allA_gas_pos_x, 'pos_y': allA_gas_pos_y}, 'B-B': {'dens': BB_gas_local_dens, 'homo': BB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}}

        # Create output dictionary for statistical averages of local density per phase/activity pairing
        local_gas_dens_stat_dict = {'radius': rad_dist[1:], 'allall_mean': allall_local_dens_mean_arr, 'allall_std': allall_local_dens_std_arr, 'allA_mean': allA_local_dens_mean_arr, 'allA_std': allA_local_dens_std_arr, 'allB_mean': allB_local_dens_mean_arr, 'allB_std': allB_local_dens_std_arr, 'AA_mean': AA_local_dens_mean_arr, 'AA_std': AA_local_dens_std_arr, 'AB_mean': AB_local_dens_mean_arr, 'AB_std': AB_local_dens_std_arr, 'BA_mean': BA_local_dens_mean_arr, 'BA_std': BA_local_dens_std_arr, 'BB_mean': BB_local_dens_mean_arr, 'BB_std': BB_local_dens_std_arr}
    
        return local_gas_dens_stat_dict, local_gas_dens_plot_dict

    def local_density(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the local density for various search
        distances of each type for each particle and averaged over all particles of each phase.

        Outputs:
        local_dens_stat_dict: dictionary containing the local density for various search
        distances of of each type ('all', 'A', or 'B') for a reference particle of a given 
        type ('all', 'A', or 'B'), averaged over all particles in each phase.
        '''
        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around all reference particles
        Aall_local_dens_mean_arr = []
        Aall_dense_local_dens_mean_arr = []
        Aall_local_dens_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around all reference particles
        Ball_local_dens_mean_arr = []
        Ball_dense_local_dens_mean_arr = []
        Ball_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around A reference particles
        allA_local_dens_mean_arr = []
        allA_dense_local_dens_mean_arr = []
        allA_local_dens_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around B reference particles
        allB_local_dens_mean_arr = []
        allB_dense_local_dens_mean_arr = []
        allB_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around all reference particles
        allall_local_dens_mean_arr = []
        allall_dense_local_dens_mean_arr = []
        allall_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around A reference particles
        AA_local_dens_mean_arr = []
        AA_dense_local_dens_mean_arr = []
        AA_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around B reference particles
        AB_local_dens_mean_arr = []
        AB_dense_local_dens_mean_arr = []
        AB_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around A reference particles
        BA_local_dens_mean_arr = []
        BA_dense_local_dens_mean_arr = []
        BA_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around B reference particles
        BB_local_dens_mean_arr = []
        BB_dense_local_dens_mean_arr = []
        BB_local_dens_std_arr = []
        
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_bulk_num_neigh = np.zeros(len(pos_A_bulk))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_bulk_num_neigh = np.zeros(len(pos_B_bulk))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_bulk_num_neigh = np.zeros(len(pos_A_bulk))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_bulk_num_neigh = np.zeros(len(pos_B_bulk))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_int_num_neigh = np.zeros(len(pos_A_int))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_int_num_neigh = np.zeros(len(pos_B_int))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_int_num_neigh = np.zeros(len(pos_A_int))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_int_num_neigh = np.zeros(len(pos_B_int))

        # Search distance for neighbors in local density calculation
        rad_dist = [0, self.r_cut, 2* self.r_cut, 3*self.r_cut, 4*self.r_cut, 5*self.r_cut]
        
        # Loop over search distances
        for j in range(1, len(rad_dist)):

            #Initiate empty arrays for tracking which particles have been analyzed when finding local neighbors
            AA_bulk_neigh_ind = np.array([], dtype=int)
            AB_bulk_neigh_ind = np.array([], dtype=int)
            BA_bulk_neigh_ind = np.array([], dtype=int)
            BB_bulk_neigh_ind = np.array([], dtype=int)
            AA_int_neigh_ind = np.array([], dtype=int)
            AB_int_neigh_ind = np.array([], dtype=int)
            BA_int_neigh_ind = np.array([], dtype=int)
            BB_int_neigh_ind = np.array([], dtype=int)

            # List of query arguments for neighbor list caculation
            query_args = dict(mode='ball', r_min = rad_dist[j-1]+0.001, r_max = rad_dist[j])#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

            # Locate potential neighbor particles by type in the dense phase
            system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
            system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))
            
            # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
            AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
            AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
            BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
            BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        
            #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_bulk)):
                if i in AA_bulk_nlist.query_point_indices:
                    if i not in AA_bulk_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AA_bulk_num_neigh[i] += len(loc)
                        AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))

            #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_bulk)):
                if i in BA_bulk_nlist.query_point_indices:
                    if i not in BA_bulk_neigh_ind:
                        
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BA_bulk_num_neigh[i] += len(loc)
                        BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))

            #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_bulk)):
                if i in AB_bulk_nlist.query_point_indices:
                    if i not in AB_bulk_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AB_bulk_num_neigh[i] += len(loc)
                        AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))

            #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_bulk)):
                if i in BB_bulk_nlist.query_point_indices:
                    if i not in BB_bulk_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BB_bulk_num_neigh[i] += len(loc)
                        BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))

            # Locate potential neighbor particles by type in the dense phase
            system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
            system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))
            
            # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
            AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
            AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
            BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
            BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        
            #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_int)):
                if i in AA_int_nlist.query_point_indices:
                    if i not in AA_int_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AA_int_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AA_int_num_neigh[i] += len(loc)
                        AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))

            #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A_int)):
                if i in BA_int_nlist.query_point_indices:
                    if i not in BA_int_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BA_int_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BA_int_num_neigh[i] += len(loc)
                        BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))

            #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_int)):
                if i in AB_int_nlist.query_point_indices:
                    if i not in AB_int_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AB_int_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AB_int_num_neigh[i] += len(loc)
                        AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))

            #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B_int)):
                if i in BB_int_nlist.query_point_indices:
                    if i not in BB_int_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BB_int_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BB_int_num_neigh[i] += len(loc)
                        BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))

            # Local density of A neighbor particles around A reference particles in bulk
            AA_bulk_local_dens = AA_bulk_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of A neighbor particles around B reference particles in bulk
            AB_bulk_local_dens = AB_bulk_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of B neighbor particles around A reference particles in bulk
            BA_bulk_local_dens = BA_bulk_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of B neighbor particles around B reference particles in bulk
            BB_bulk_local_dens = BB_bulk_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of A neighbor particles around A reference particles in interface
            AA_int_local_dens = AA_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of A neighbor particles around B reference particles in interface
            AB_int_local_dens = AB_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of B neighbor particles around A reference particles in interface
            BA_int_local_dens = BA_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of B neighbor particles around B reference particles in interface
            BB_int_local_dens = BB_int_num_neigh / (np.pi*rad_dist[j]**2)
            
            # Save neighbor and local orientational order to arrays for all B reference particles of the respective phase with all nearest neighbors
            Ball_bulk_local_dens= np.append(BA_bulk_local_dens, BB_bulk_local_dens)
            Ball_int_local_dens = np.append(BA_int_local_dens, BB_int_local_dens)

            # Save neighbor and local orientational order to arrays for all A reference particles of the respective phase with all nearest neighbors
            Aall_bulk_local_dens = np.append(AA_bulk_local_dens, AB_bulk_local_dens)
            Aall_int_local_dens = np.append(AA_int_local_dens, AB_int_local_dens)

            # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
            allB_bulk_num_neigh = AB_bulk_num_neigh + BB_bulk_num_neigh
            allB_bulk_local_dens = allB_bulk_num_neigh / (np.pi*rad_dist[j]**2)
            allB_int_num_neigh = AB_int_num_neigh + BB_int_num_neigh
            allB_int_local_dens = allB_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
            allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh
            allA_bulk_local_dens = allA_bulk_num_neigh / (np.pi*rad_dist[j]**2)
            allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh
            allA_int_local_dens = allA_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
            allall_bulk_num_neigh = np.append(allA_bulk_num_neigh, allB_bulk_num_neigh)
            allall_bulk_local_dens = allall_bulk_num_neigh / (np.pi*rad_dist[j]**2)
            allall_int_num_neigh = np.append(allA_int_num_neigh, allB_int_num_neigh)
            allall_int_local_dens = allall_int_num_neigh / (np.pi*rad_dist[j]**2)

            # Local density of all neighbor particles around all reference particles in dense phase
            allall_dense_local_dens = np.append(allall_bulk_local_dens, allall_int_local_dens)
            allall_dense_local_dens_mean_arr.append(np.mean(allall_dense_local_dens))
            allall_dense_local_dens_inhomog = (allall_dense_local_dens - allall_dense_local_dens_mean_arr[-1])

            # Local density of B neighbor particles around all reference particles in dense phase
            Ball_dense_local_dens= np.append(Ball_bulk_local_dens, Ball_int_local_dens)
            Ball_dense_local_dens_mean_arr.append(np.mean(Ball_dense_local_dens))
            Ball_dense_local_dens_inhomog = (Ball_dense_local_dens - Ball_dense_local_dens_mean_arr[-1])

            # Local density of A neighbor particles around all reference particles in dense phase
            Aall_dense_local_dens= np.append(Aall_bulk_local_dens, Aall_int_local_dens)
            Aall_dense_local_dens_mean_arr.append(np.mean(Aall_dense_local_dens))
            Aall_dense_local_dens_inhomog = (Aall_dense_local_dens - Aall_dense_local_dens_mean_arr[-1])

            # Local density of all neighbor particles around A reference particles in dense phase
            allA_dense_local_dens = np.append(allA_bulk_local_dens, allA_int_local_dens)
            allA_dense_local_dens_mean_arr.append(np.mean(allA_dense_local_dens))
            allA_dense_local_dens_inhomog = (allA_dense_local_dens - allA_dense_local_dens_mean_arr[-1])

            # Local density of all neighbor particles around B reference particles in dense phase
            allB_dense_local_dens = np.append(allB_bulk_local_dens, allB_int_local_dens)
            allB_dense_local_dens_mean_arr.append(np.mean(allB_dense_local_dens))
            allB_dense_local_dens_inhomog = (allB_dense_local_dens - allB_dense_local_dens_mean_arr[-1])

            # Local density of A neighbor particles around A reference particles in dense phase
            AA_dense_local_dens = np.append(AA_bulk_local_dens, AA_int_local_dens)
            AA_dense_local_dens_mean_arr.append(np.mean(AA_dense_local_dens))
            AA_dense_local_dens_inhomog = (AA_dense_local_dens - AA_dense_local_dens_mean_arr[-1])

            # Local density of A neighbor particles around B reference particles in dense phase
            AB_dense_local_dens = np.append(AB_bulk_local_dens, AB_int_local_dens)
            AB_dense_local_dens_mean_arr.append(np.mean(AB_dense_local_dens))
            AB_dense_local_dens_inhomog = (AB_dense_local_dens - AB_dense_local_dens_mean_arr[-1])

            # Local density of B neighbor particles around A reference particles in dense phase
            BA_dense_local_dens = np.append(BA_bulk_local_dens, BA_int_local_dens)
            BA_dense_local_dens_mean_arr.append(np.mean(BA_dense_local_dens))
            BA_dense_local_dens_inhomog = (BA_dense_local_dens - BA_dense_local_dens_mean_arr[-1])

            # Local density of B neighbor particles around B reference particles in dense phase
            BB_dense_local_dens = np.append(BB_bulk_local_dens, BB_int_local_dens)
            BB_dense_local_dens_mean_arr.append(np.mean(BB_dense_local_dens))
            BB_dense_local_dens_inhomog = (BB_dense_local_dens - BB_dense_local_dens_mean_arr[-1])

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around A reference particles in dense phase
            AA_local_dens_mean_arr.append(np.mean(AA_bulk_local_dens))
            AA_local_dens_inhomog = (AA_bulk_local_dens - AA_local_dens_mean_arr[-1])
            AA_local_dens_std_arr.append(np.std(AA_bulk_local_dens))

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around B reference particles in dense phase
            AB_local_dens_mean_arr.append(np.mean(AB_bulk_local_dens))
            AB_local_dens_inhomog = (AB_bulk_local_dens - AB_local_dens_mean_arr[-1])
            AB_local_dens_std_arr.append(np.std(AB_bulk_local_dens))

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around A reference particles in dense phase
            BA_local_dens_mean_arr.append(np.mean(BA_bulk_local_dens))
            BA_local_dens_inhomog = (BA_bulk_local_dens - BA_local_dens_mean_arr[-1])
            BA_local_dens_std_arr.append(np.std(BA_bulk_local_dens))

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around B reference particles in dense phase
            BB_local_dens_mean_arr.append(np.mean(BB_bulk_local_dens))
            BB_local_dens_inhomog = (BB_bulk_local_dens - BB_local_dens_mean_arr[-1])
            BB_local_dens_std_arr.append(np.std(BB_bulk_local_dens))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles in dense phase
            allA_local_dens_mean_arr.append(np.mean(allA_bulk_local_dens))
            allA_local_dens_inhomog = (allA_bulk_local_dens - allA_local_dens_mean_arr[-1])
            allA_local_dens_std_arr.append(np.std(allA_bulk_local_dens))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles in dense phase
            allB_local_dens_mean_arr.append(np.mean(allB_bulk_local_dens))
            allB_local_dens_inhomog = (allB_bulk_local_dens - allB_local_dens_mean_arr[-1])
            allB_local_dens_std_arr.append(np.std(allB_bulk_local_dens))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles in dense phase
            Aall_local_dens_mean_arr.append(np.mean(Aall_bulk_local_dens))
            Aall_local_dens_inhomog = (Aall_bulk_local_dens - Aall_local_dens_mean_arr[-1])
            Aall_local_dens_std_arr.append(np.std(Aall_bulk_local_dens))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles in dense phase
            Ball_local_dens_mean_arr.append(np.mean(Ball_bulk_local_dens))
            Ball_local_dens_inhomog = (Ball_bulk_local_dens - Ball_local_dens_mean_arr[-1])
            Ball_local_dens_std_arr.append(np.std(Ball_bulk_local_dens))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around all reference particles in dense phase
            allall_local_dens_mean_arr.append(np.mean(allall_bulk_local_dens))
            allall_local_dens_std_arr.append(np.std(allall_bulk_local_dens))

            # If search distance given, then prepare data for plotting!
            if rad_dist[j]==5*self.r_cut:

                # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
                allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
                allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])

                # Save neighbor, local orientational order, and position to arrays for all interface reference particles with all nearest neighbors
                allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
                allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])

                # Save neighbor, local orientational order, and position to arrays for A dense phase reference particles with all nearest neighbors
                Aall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
                Aall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

                # Save neighbor, local orientational order, and position to arrays for B dense phase reference particles with all nearest neighbors
                Ball_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
                Ball_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with A nearest neighbors
                allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
                allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with B nearest neighbors
                allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
                allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with all nearest neighbors
                allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
                allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)

                # Create output dictionary for single particle values of local density per phase/activity pairing for plotting
                local_dens_plot_dict = {'all-all': {'dens': allall_dense_local_dens, 'homo': allall_dense_local_dens_inhomog, 'pos_x': allall_dense_pos_x, 'pos_y': allall_dense_pos_y}, 'all-A': {'dens': allA_dense_local_dens, 'homo': allA_dense_local_dens_inhomog, 'pos_x': allA_dense_pos_x, 'pos_y': allA_dense_pos_y}, 'all-B': {'dens': allB_dense_local_dens, 'homo': allB_dense_local_dens_inhomog, 'pos_x': allB_dense_pos_x, 'pos_y': allB_dense_pos_y}, 'A-all': {'dens': Aall_dense_local_dens, 'homo': Aall_dense_local_dens_inhomog, 'pos_x': Aall_dense_pos_x, 'pos_y': Aall_dense_pos_y}, 'B-all': {'dens': Ball_dense_local_dens, 'homo': Ball_dense_local_dens_inhomog, 'pos_x': Ball_dense_pos_x, 'pos_y': Ball_dense_pos_y}, 'A-A': {'dens': AA_dense_local_dens, 'homo': AA_dense_local_dens_inhomog, 'pos_x': allA_dense_pos_x, 'pos_y': allA_dense_pos_y}, 'A-B': {'dens': AB_dense_local_dens, 'homo': AB_dense_local_dens_inhomog, 'pos_x': allB_dense_pos_x, 'pos_y': allB_dense_pos_y}, 'B-A': {'dens': BA_dense_local_dens, 'homo': BA_dense_local_dens_inhomog, 'pos_x': allA_dense_pos_x, 'pos_y': allA_dense_pos_y}, 'B-B': {'dens': BB_dense_local_dens, 'homo': BB_dense_local_dens_inhomog, 'pos_x': allB_dense_pos_x, 'pos_y': allB_dense_pos_y}}

        # Create output dictionary for statistical averages of local density per phase/activity pairing
        local_dens_stat_dict = {'radius': rad_dist[1:], 'allA_mean': allA_local_dens_mean_arr, 'allA_std': allA_local_dens_std_arr, 'allB_mean': allB_local_dens_mean_arr, 'allB_std': allB_local_dens_std_arr, 'AA_mean': AA_local_dens_mean_arr, 'AA_std': AA_local_dens_std_arr, 'AB_mean': AB_local_dens_mean_arr, 'AB_std': AB_local_dens_std_arr, 'BA_mean': BA_local_dens_mean_arr, 'BA_std': BA_local_dens_std_arr, 'BB_mean': BB_local_dens_mean_arr, 'BB_std': BB_local_dens_std_arr}
    
        return local_dens_stat_dict, local_dens_plot_dict

    
    def clustering_coefficient(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the clustering coefficient of each
        type for each particle and averaged over all particles of each phase. Useful for
        visualizing segregated domains.

        Outputs:
        clust_plot_dict: dictionary containing information on the number of cluster coefficient
         of each bulk and interface reference particle of each type ('all', 'A', or 'B').

        clust_stat_dict: dictionary containing information on the average cluster coefficient
         of each bulk and interface reference particle of each type ('all', 'A', or 'B').

        prob_plot_dict: dictionary containing information on the number of neighbors
         of each bulk and interface reference particle of each type ('all', 'A', or 'B') with each type.

        prob_stat_dict: dictionary containing information on fraction of bulk and interface 
        reference particle of each type ('all', 'A', or 'B') having a given number of neighbors.
        '''

        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]
        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max = self.r_cut)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))
        
        # Python3 Program to print BFS traversal
        # from a given source vertex. BFS(int s)
        # traverses vertices reachable from s.
        from collections import defaultdict
        
        # This class represents a directed graph
        # using adjacency list representation
        class Graph:
        
            # Constructor
            def __init__(self):
        
                # default dictionary to store graph
                self.graph = defaultdict(list)
        
            # function to add an edge to graph
        # Make a list visited[] to check if a node is already visited or not
            def addEdge(self,u,v):
                self.graph[u].append(v)
                self.visited=[]
        
            # Function to print a BFS of graph
            def BFS(self, s):
        
                # Create a queue for BFS
                queue = []
                path=[]
        
                # Add the source node in
                # visited and enqueue it
                queue.append(s)
                path.append(s)
                
                self.visited.append(s)
        
                while queue:        
                    # Dequeue a vertex from
                    # queue and print it
                    s = queue.pop(0)
        
                    # Get all adjacent vertices of the
                    # dequeued vertex s. If a adjacent
                    # has not been visited, then add it
                    # in visited and enqueue it
                    for i in self.graph[s]:
                        if i not in self.visited:
                            queue.append(i)
                            self.visited.append(s)
                            path.append(i)
                return path
        
        # Driver code

        # Create a graph given in
        # the above diagram
        g = Graph()

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
    
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])
        AA_path_length = np.array([])
        
        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_bulk)):

            # If reference particle has neighbors...
            if i in AA_bulk_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in AA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                    for j in loc:
                        g.addEdge(i, j)
                    AA_path_length = np.append(AA_path_length, len(g.BFS(i)))
            else:
                #Save nearest neighbor information to array
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
                path_length = np.append(AA_path_length, 0)
        
        print ("Following is Breadth First Traversal"
                        " (starting from vertex 2)")
        #g.BFS(11249)

        #Find fraction of A bulk particles with a given number of A neighbors
        p0_AA_bulk = len(np.where(AA_bulk_num_neigh==0)[0])/len(pos_A_bulk)
        p1_AA_bulk = len(np.where(AA_bulk_num_neigh==1)[0])/len(pos_A_bulk)
        p2_AA_bulk = len(np.where(AA_bulk_num_neigh==2)[0])/len(pos_A_bulk)
        p3_AA_bulk = len(np.where(AA_bulk_num_neigh==3)[0])/len(pos_A_bulk)
        p4_AA_bulk = len(np.where(AA_bulk_num_neigh==4)[0])/len(pos_A_bulk)
        p5_AA_bulk = len(np.where(AA_bulk_num_neigh==5)[0])/len(pos_A_bulk)
        p6_AA_bulk = len(np.where(AA_bulk_num_neigh==6)[0])/len(pos_A_bulk)
        p7_AA_bulk = len(np.where(AA_bulk_num_neigh==7)[0])/len(pos_A_bulk)
        p8_AA_bulk = len(np.where(AA_bulk_num_neigh==8)[0])/len(pos_A_bulk)
        
        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_bulk)):

            # If reference particle has neighbors...
            if i in BA_bulk_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in BA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
        
        #Find fraction of A bulk particles with a given number of B neighbors
        p0_BA_bulk = len(np.where(BA_bulk_num_neigh==0)[0])/len(pos_A_bulk)
        p1_BA_bulk = len(np.where(BA_bulk_num_neigh==1)[0])/len(pos_A_bulk)
        p2_BA_bulk = len(np.where(BA_bulk_num_neigh==2)[0])/len(pos_A_bulk)
        p3_BA_bulk = len(np.where(BA_bulk_num_neigh==3)[0])/len(pos_A_bulk)
        p4_BA_bulk = len(np.where(BA_bulk_num_neigh==4)[0])/len(pos_A_bulk)
        p5_BA_bulk = len(np.where(BA_bulk_num_neigh==5)[0])/len(pos_A_bulk)
        p6_BA_bulk = len(np.where(BA_bulk_num_neigh==6)[0])/len(pos_A_bulk)
        p7_BA_bulk = len(np.where(BA_bulk_num_neigh==7)[0])/len(pos_A_bulk)
        p8_BA_bulk = len(np.where(BA_bulk_num_neigh==8)[0])/len(pos_A_bulk)

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_bulk)):

            # If reference particle has neighbors...
            if i in AB_bulk_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in AB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))

        #Find fraction of B bulk particles with a given number of A neighbors
        p0_AB_bulk = len(np.where(AB_bulk_num_neigh==0)[0])/len(pos_B_bulk)
        p1_AB_bulk = len(np.where(AB_bulk_num_neigh==1)[0])/len(pos_B_bulk)
        p2_AB_bulk = len(np.where(AB_bulk_num_neigh==2)[0])/len(pos_B_bulk)
        p3_AB_bulk = len(np.where(AB_bulk_num_neigh==3)[0])/len(pos_B_bulk)
        p4_AB_bulk = len(np.where(AB_bulk_num_neigh==4)[0])/len(pos_B_bulk)
        p5_AB_bulk = len(np.where(AB_bulk_num_neigh==5)[0])/len(pos_B_bulk)
        p6_AB_bulk = len(np.where(AB_bulk_num_neigh==6)[0])/len(pos_B_bulk)
        p7_AB_bulk = len(np.where(AB_bulk_num_neigh==7)[0])/len(pos_B_bulk)
        p8_AB_bulk = len(np.where(AB_bulk_num_neigh==8)[0])/len(pos_B_bulk)

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])

        #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_bulk)):

            # If reference particle has neighbors...
            if i in BB_bulk_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in BB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))

        #Find fraction of B bulk particles with a given number of B neighbors
        p0_BB_bulk = len(np.where(BB_bulk_num_neigh==0)[0])/len(pos_B_bulk)
        p1_BB_bulk = len(np.where(BB_bulk_num_neigh==1)[0])/len(pos_B_bulk)
        p2_BB_bulk = len(np.where(BB_bulk_num_neigh==2)[0])/len(pos_B_bulk)
        p3_BB_bulk = len(np.where(BB_bulk_num_neigh==3)[0])/len(pos_B_bulk)
        p4_BB_bulk = len(np.where(BB_bulk_num_neigh==4)[0])/len(pos_B_bulk)
        p5_BB_bulk = len(np.where(BB_bulk_num_neigh==5)[0])/len(pos_B_bulk)
        p6_BB_bulk = len(np.where(BB_bulk_num_neigh==6)[0])/len(pos_B_bulk)
        p7_BB_bulk = len(np.where(BB_bulk_num_neigh==7)[0])/len(pos_B_bulk)
        p8_BB_bulk = len(np.where(BB_bulk_num_neigh==8)[0])/len(pos_B_bulk)

        # Locate potential neighbor particles by type in the entire system
        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        # Generate neighbor list of all particles (per query args) of respective type (A or B) neighboring interface phase reference particles of respective type (A or B)
        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type A interface particles
        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])

        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_int)):

            # If reference particle has neighbors...
            if i in AA_int_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in AA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))

        #Find fraction of A interface particles with a given number of B neighbors
        p0_AA_int = len(np.where(AA_int_num_neigh==0)[0])/len(pos_A_int)
        p1_AA_int = len(np.where(AA_int_num_neigh==1)[0])/len(pos_A_int)
        p2_AA_int = len(np.where(AA_int_num_neigh==2)[0])/len(pos_A_int)
        p3_AA_int = len(np.where(AA_int_num_neigh==3)[0])/len(pos_A_int)
        p4_AA_int = len(np.where(AA_int_num_neigh==4)[0])/len(pos_A_int)
        p5_AA_int = len(np.where(AA_int_num_neigh==5)[0])/len(pos_A_int)
        p6_AA_int = len(np.where(AA_int_num_neigh==6)[0])/len(pos_A_int)
        p7_AA_int = len(np.where(AA_int_num_neigh==7)[0])/len(pos_A_int)
        p8_AA_int = len(np.where(AA_int_num_neigh==8)[0])/len(pos_A_int)

        #Initiate empty arrays for finding nearest B neighboring particles surrounding type A interface particles
        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_int)):

            # If reference particle has neighbors...
            if i in AB_int_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in AB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))

        #Find fraction of B interface particles with a given number of A neighbors
        p0_AB_int = len(np.where(AB_int_num_neigh==0)[0])/len(pos_B_int)
        p1_AB_int = len(np.where(AB_int_num_neigh==1)[0])/len(pos_B_int)
        p2_AB_int = len(np.where(AB_int_num_neigh==2)[0])/len(pos_B_int)
        p3_AB_int = len(np.where(AB_int_num_neigh==3)[0])/len(pos_B_int)
        p4_AB_int = len(np.where(AB_int_num_neigh==4)[0])/len(pos_B_int)
        p5_AB_int = len(np.where(AB_int_num_neigh==5)[0])/len(pos_B_int)
        p6_AB_int = len(np.where(AB_int_num_neigh==6)[0])/len(pos_B_int)
        p7_AB_int = len(np.where(AB_int_num_neigh==7)[0])/len(pos_B_int)
        p8_AB_int = len(np.where(AB_int_num_neigh==8)[0])/len(pos_B_int)

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type B interface particles
        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A_int)):

            # If reference particle has neighbors...
            if i in BA_int_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in BA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))

        #Find fraction of A interface particles with a given number of B neighbors
        p0_BA_int = len(np.where(BA_int_num_neigh==0)[0])/len(pos_A_int)
        p1_BA_int = len(np.where(BA_int_num_neigh==1)[0])/len(pos_A_int)
        p2_BA_int = len(np.where(BA_int_num_neigh==2)[0])/len(pos_A_int)
        p3_BA_int = len(np.where(BA_int_num_neigh==3)[0])/len(pos_A_int)
        p4_BA_int = len(np.where(BA_int_num_neigh==4)[0])/len(pos_A_int)
        p5_BA_int = len(np.where(BA_int_num_neigh==5)[0])/len(pos_A_int)
        p6_BA_int = len(np.where(BA_int_num_neigh==6)[0])/len(pos_A_int)
        p7_BA_int = len(np.where(BA_int_num_neigh==7)[0])/len(pos_A_int)
        p8_BA_int = len(np.where(BA_int_num_neigh==8)[0])/len(pos_A_int)

        # Initiate empty arrays for finding nearest B neighboring particles surrounding type B interface particles
        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])

        # Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B_int)):

            # If reference particle has neighbors...
            if i in BB_int_nlist.query_point_indices:

                # If reference particle hasn't been considered already...
                if i not in BB_int_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]

                    # Save nearest neighbor information to array
                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information to array
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
        
        #Find fraction of B interface particles with a given number of B neighbors
        p0_BB_int = len(np.where(BB_int_num_neigh==0)[0])/len(pos_B_int)
        p1_BB_int = len(np.where(BB_int_num_neigh==1)[0])/len(pos_B_int)
        p2_BB_int = len(np.where(BB_int_num_neigh==2)[0])/len(pos_B_int)
        p3_BB_int = len(np.where(BB_int_num_neigh==3)[0])/len(pos_B_int)
        p4_BB_int = len(np.where(BB_int_num_neigh==4)[0])/len(pos_B_int)
        p5_BB_int = len(np.where(BB_int_num_neigh==5)[0])/len(pos_B_int)
        p6_BB_int = len(np.where(BB_int_num_neigh==6)[0])/len(pos_B_int)
        p7_BB_int = len(np.where(BB_int_num_neigh==7)[0])/len(pos_B_int)
        p8_BB_int = len(np.where(BB_int_num_neigh==8)[0])/len(pos_B_int)

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
        allB_bulk_num_neigh = AB_bulk_num_neigh + BB_bulk_num_neigh

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
        allA_bulk_num_neigh = AA_bulk_num_neigh + BA_bulk_num_neigh

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
        allB_int_num_neigh = AB_int_num_neigh + BB_int_num_neigh

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
        allA_int_num_neigh = AA_int_num_neigh + BA_int_num_neigh

        # Initiate empty arrays for calculating cluster coefficients
        A_bulk_clust_coeff = np.zeros(len(allA_bulk_num_neigh))
        B_bulk_clust_coeff = np.zeros(len(allB_bulk_num_neigh))
        all_bulk_clust_coeff = np.append(A_bulk_clust_coeff, B_bulk_clust_coeff)

        A_int_clust_coeff = np.zeros(len(allA_int_num_neigh))
        B_int_clust_coeff = np.zeros(len(allB_int_num_neigh))
        all_int_clust_coeff = np.append(A_int_clust_coeff, B_int_clust_coeff)

        # Cluster coefficient for each B reference particle in bulk
        for i in range(0, len(allB_bulk_num_neigh)):
            if allB_bulk_num_neigh[i]>0:
                B_bulk_clust_coeff[i] = BB_bulk_num_neigh[i]/allB_bulk_num_neigh[i]

        # Cluster coefficient for each A reference particle in bulk
        for i in range(0, len(allA_bulk_num_neigh)):
            if allA_bulk_num_neigh[i]>0:
                A_bulk_clust_coeff[i] = AA_bulk_num_neigh[i]/allA_bulk_num_neigh[i]

        # Cluster coefficient for each B reference particle in interface
        for i in range(0, len(allB_int_num_neigh)):
            if allB_int_num_neigh[i]>0:
                B_int_clust_coeff[i] = BB_int_num_neigh[i]/allB_int_num_neigh[i]

        # Cluster coefficient for each A reference particle in interface
        for i in range(0, len(allA_int_num_neigh)):
            if allA_int_num_neigh[i]>0:
                A_int_clust_coeff[i] = AA_int_num_neigh[i]/allA_int_num_neigh[i]
        
        # Save number of neighbors for the respective activity dense phase reference particles with the respective activity nearest neighbors
        AA_dense_num_neigh = np.append(AA_bulk_num_neigh, AA_int_num_neigh)
        AB_dense_num_neigh = np.append(AB_bulk_num_neigh, AB_int_num_neigh)
        BA_dense_num_neigh = np.append(BA_bulk_num_neigh, BA_int_num_neigh)
        BB_dense_num_neigh = np.append(BB_bulk_num_neigh, BB_int_num_neigh)

        # Save cluster coefficient for the respective activity dense phase reference particles with the any activity nearest neighbors
        A_dense_clust_coeff = np.append(A_bulk_clust_coeff, A_int_clust_coeff)
        B_dense_clust_coeff = np.append(B_bulk_clust_coeff, B_int_clust_coeff)
        all_dense_clust_coeff = np.append(A_dense_clust_coeff, B_dense_clust_coeff)

        # Save x- and y- positions of A, B, or all types of particles in bulk
        A_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        A_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])
        B_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        B_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])
        all_dense_pos_x = np.append(A_dense_pos_x, B_dense_pos_x)
        all_dense_pos_y = np.append(A_dense_pos_y, B_dense_pos_y)

        # Number of neighbor array
        num_neigh_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Create output dictionary for plotting of cluster coefficient information of each particle per phase/activity pairing and their respective x-y locations
        clust_plot_dict = {'A': {'pos': {'x': A_dense_pos_x, 'y': A_dense_pos_y}, 'clust': A_dense_clust_coeff}, 'B': {'pos': {'x': B_dense_pos_x, 'y': B_dense_pos_y}, 'clust': B_dense_clust_coeff}, 'all': {'pos': {'x': all_dense_pos_x, 'y': all_dense_pos_y}, 'clust': all_dense_clust_coeff}}
        clust_stat_dict = {'dense': {'clust_coeff': {'all': np.mean(all_dense_clust_coeff), 'A': np.mean(A_dense_clust_coeff), 'B': np.mean(B_dense_clust_coeff)}}, 'bulk': {'clust_coeff': {'all': np.mean(all_bulk_clust_coeff), 'A': np.mean(A_bulk_clust_coeff), 'B': np.mean(B_bulk_clust_coeff)}}, 'int': {'clust_coeff': {'all': np.mean(all_int_clust_coeff), 'A': np.mean(A_int_clust_coeff), 'B': np.mean(B_int_clust_coeff)}}}
        prob_plot_dict = {'bulk': {'AA': AA_bulk_num_neigh, 'AB': AB_bulk_num_neigh, 'BA': BA_bulk_num_neigh, 'BB': BB_bulk_num_neigh}, 'int': {'AA': AA_int_num_neigh, 'AB': AB_int_num_neigh, 'BA': BA_int_num_neigh, 'BB': BB_int_num_neigh}, 'dense': {'AA': AA_dense_num_neigh, 'AB': AB_dense_num_neigh, 'BA': BA_dense_num_neigh, 'BB': BB_dense_num_neigh}}
        prob_stat_dict = {'neigh': num_neigh_arr, 'bulk': {'AA': [p0_AA_bulk, p1_AA_bulk, p2_AA_bulk, p3_AA_bulk, p4_AA_bulk, p5_AA_bulk, p6_AA_bulk, p7_AA_bulk, p8_AA_bulk], 'BA': [p0_BA_bulk, p1_BA_bulk, p2_BA_bulk, p3_BA_bulk, p4_BA_bulk, p5_BA_bulk, p6_BA_bulk, p7_BA_bulk, p8_BA_bulk], 'AB': [p0_AB_bulk, p1_AB_bulk, p2_AB_bulk, p3_AB_bulk, p4_AB_bulk, p5_AB_bulk, p6_AB_bulk, p7_AB_bulk, p8_AB_bulk], 'BB': [p0_BB_bulk, p1_BB_bulk, p2_BB_bulk, p3_BB_bulk, p4_BB_bulk, p5_BB_bulk, p6_BB_bulk, p7_BB_bulk, p8_BB_bulk]}, 'bulk': {'AA': [p0_AA_int, p1_AA_int, p2_AA_int, p3_AA_int, p4_AA_int, p5_AA_int, p6_AA_int, p7_AA_int, p8_AA_int], 'BA': [p0_BA_int, p1_BA_int, p2_BA_int, p3_BA_int, p4_BA_int, p5_BA_int, p6_BA_int, p7_BA_int, p8_BA_int], 'AB': [p0_AB_int, p1_AB_int, p2_AB_int, p3_AB_int, p4_AB_int, p5_AB_int, p6_AB_int, p7_AB_int, p8_AB_int], 'BB': [p0_BB_int, p1_BB_int, p2_BB_int, p3_BB_int, p4_BB_int, p5_BB_int, p6_BB_int, p7_BB_int, p8_BB_int]}}
        
        return clust_plot_dict, clust_stat_dict, prob_plot_dict, prob_stat_dict

    def domain_size(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the domain size of each cluster of like
        species within the bulk phase. Useful for quantifying segregated domains.

        Outputs:
        clust_plot_dict: dictionary containing information on the number of cluster coefficient
         of each bulk and interface reference particle of each type ('all', 'A', or 'B').
        '''
        # Count total number of bins in each phase
        phase_count_dict = self.particle_prop_functs.phase_count(self.phase_dict)

        # Count particles per phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        # Generate neighbor list of dense phase particles of respective type as reference particles
        system_A_dense = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_dense = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        #Define A domains
        cl_A=freud.cluster.Cluster()  

        #Define A domains
        cl_B=freud.cluster.Cluster()                             

        # Calculate A domains given neighbor list, positions, and maximal radial interaction distance
        cl_A.compute(system_A_dense, neighbors={'r_max': self.r_cut, 'r_min': 0.1})        
        
        # Calculate B domains given neighbor list, positions, and maximal radial interaction distance
        cl_B.compute(system_B_dense, neighbors={'r_max': self.r_cut, 'r_min': 0.1})        
        
        #Define A domain properties
        clp_A = freud.cluster.ClusterProperties()                 
        
        #Define B domain properties
        clp_B = freud.cluster.ClusterProperties()                 
        
        # get id of each A domain
        ids_A = cl_A.cluster_idx   

        # get id of each B domain
        ids_B = cl_B.cluster_idx                                    

        # Calculate A domain properties given domain IDs
        clp_A.compute(system_A_dense, ids_A)                            

        # Calculate B domain properties given domain IDs
        clp_B.compute(system_B_dense, ids_B)                            

        # find A domain sizes
        clust_size_A = clp_A.sizes                          

        # find B domain sizes
        clust_size_B = clp_B.sizes                                 

        #Identify largest A or B domain
        lcID_A = np.where(clust_size_A >= 1)[0]    
        lcID_B = np.where(clust_size_B >= 1)[0]

        #Identify largest A or B domain
        lcID_A2 = np.where(clust_size_A >= 2)[0]    
        lcID_B2 = np.where(clust_size_B >= 2)[0]
        
        #Identify smallest A or B domain
        lcID_A_smallest = len(np.where(clust_size_A == 1)[0])
        lcID_B_smallest = len(np.where(clust_size_B == 1)[0])

        #Identify second smallest A or B domain
        lcID_A_second_smallest = len(np.where(clust_size_A == 2)[0])
        lcID_B_second_smallest = len(np.where(clust_size_B == 2)[0])

        #Identify third smallest A or B domain
        lcID_A_third_smallest = len(np.where(clust_size_A == 3)[0])
        lcID_B_third_smallest = len(np.where(clust_size_B == 3)[0])

        #Identify fourth smallest A or B domain
        lcID_A_fourth_smallest = len(np.where(clust_size_A == 4)[0])
        lcID_B_fourth_smallest = len(np.where(clust_size_B == 4)[0])

        # Calculate largest A or B domain size
        max_A = np.max(clust_size_A[lcID_A])
        max_B = np.max(clust_size_B[lcID_B])

        # Find largest overall domain size
        if max_A >= max_B:
            max_val = max_A
        else:
            max_val = max_B
        
        # Find size of largest A domain
        first_max_clust_A = np.max(clust_size_A[lcID_A])

        # Remove largest A domain
        first_max_id = np.where(clust_size_A[lcID_A]==first_max_clust_A)[0]
        clust_A_first_temp = np.delete(clust_size_A[lcID_A], first_max_id)

        # Find size of second largest A domain
        second_max_clust_A = np.max(clust_A_first_temp)

        # Remove second largest A domain
        second_max_id = np.where(clust_A_first_temp==second_max_clust_A)[0]
        clust_A_second_temp = np.delete(clust_A_first_temp, second_max_id)

        # Find size of third largest A domain
        third_max_clust_A = np.max(clust_A_second_temp)

        # Remove third largest A domain
        third_max_id = np.where(clust_A_second_temp==third_max_clust_A)[0]
        clust_A_third_temp = np.delete(clust_A_second_temp, third_max_id)

        # Find size of fourth largest A domain
        fourth_max_clust_A = np.max(clust_A_third_temp)

        # Remove fourth largest A domain
        fourth_max_id = np.where(clust_A_third_temp==fourth_max_clust_A)[0]
        clust_A_fourth_temp = np.delete(clust_A_third_temp, fourth_max_id)

        # Find ID and size of fifth largest A domain
        fifth_max_clust_A = np.max(clust_A_fourth_temp)

        # Find size of largest B domain
        first_max_clust_B = np.max(clust_size_B[lcID_B])

        # Remove largest B domain
        first_max_id = np.where(clust_size_B[lcID_B]==first_max_clust_B)[0]
        clust_B_first_temp = np.delete(clust_size_B[lcID_B], first_max_id)

        # Find size of second largest B domain
        second_max_clust_B = np.max(clust_B_first_temp)

        # Remove second largest B domain
        second_max_id = np.where(clust_B_first_temp==second_max_clust_B)[0]
        clust_B_second_temp = np.delete(clust_B_first_temp, second_max_id)

        # Find size of third largest B domain
        third_max_clust_B = np.max(clust_B_second_temp)

        # Remove third largest B domain
        third_max_id = np.where(clust_B_second_temp==third_max_clust_B)[0]
        clust_B_third_temp = np.delete(clust_B_second_temp, third_max_id)

        # Find size of fourth largest B domain
        fourth_max_clust_B = np.max(clust_B_third_temp)

        # Remove fourth largest B domain
        fourth_max_id = np.where(clust_B_third_temp==fourth_max_clust_B)[0]
        clust_B_fourth_temp = np.delete(clust_B_third_temp, fourth_max_id)

        # Find size of fifth largest B domain
        fifth_max_clust_B = np.max(clust_B_fourth_temp)
        
        # Create output dictionary for statistical information on various domains sizes for type A and B particles in bulk
        domain_stat_dict = {'A': {'pop': len(pos_A_dense), 'avg_size': np.mean(clust_size_A[lcID_A]), 'std_size': np.std(clust_size_A[lcID_A]), 'num': len(clust_size_A[lcID_A]), 'avg_size2': np.mean(clust_size_A[lcID_A2]), 'std_size2': np.std(clust_size_A[lcID_A2]), 'num2': len(clust_size_A[lcID_A2]), 'first_size': first_max_clust_A, 'second_size': second_max_clust_A, 'third_size': third_max_clust_A, 'fourth_size': fourth_max_clust_A, 'fifth_size': fifth_max_clust_A, 'one_num': lcID_A_smallest, 'two_num': lcID_A_second_smallest, 'three_num': lcID_A_third_smallest, 'fourth_num': lcID_A_fourth_smallest}, 'B': {'pop': len(pos_B_dense), 'avg_size': np.mean(clust_size_B[lcID_B]), 'std_size': np.std(clust_size_B[lcID_B]), 'num': len(clust_size_B[lcID_B]), 'avg_size2': np.mean(clust_size_B[lcID_B2]), 'std_size2': np.std(clust_size_B[lcID_B2]), 'num2': len(clust_size_B[lcID_B2]), 'first': first_max_clust_B, 'second': second_max_clust_B, 'third': third_max_clust_B, 'fourth': fourth_max_clust_B, 'fifth': fifth_max_clust_B, 'one_num': lcID_B_smallest, 'two_num': lcID_B_second_smallest, 'three_num': lcID_B_third_smallest, 'fourth_num': lcID_B_fourth_smallest}}
        
        return domain_stat_dict

    def hexatic_order(self):
        '''
        Purpose: Takes the position of all particles in the system and uses neighbor lists to find the
        nearest, interacting neighbors of each reference particle and calculates the
        local hexatic order parameter of each reference particle.

        Outputs:
        hexatic_order_dict: dictionary containing the hexatic order parameter, which
        measures the degree of the local structure matching an ideal HCP lattice, and crystal
        domain angle (relative to the x-axis) for each particle in the system
        '''
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)
        
        # Locate potential neighbor particles in the system
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        # Generate neighbor list of all particles (per query args) neighboring all reference particles
        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        # Set ideal number of neighbors to 6 for 2D HCP lattice
        hex_order = freud.order.Hexatic(k=6)

        # Compute hexatic order for 6 nearest neighbors
        hex_order.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)

        #hex_order.compute(system=(f_box, pos_A), neighbors=allA_bulk_nlist)

        psi_k = hex_order.particle_order

        #Average hexatic order parameter
        avg_psi_k = np.mean(psi_k)

        # Create an array of angles relative to the average
        order_param = np.abs(psi_k)

        #Calculate relative bond orientation of crystal domains
        relative_angles = np.angle(psi_k)

        #Since hexagonal domains, translate angle to 0 to pi/3 radians
        for g in range(0, len(relative_angles)):
            if relative_angles[g]<(-2*np.pi/3):
                relative_angles[g]+=(np.pi)
            elif (-2*np.pi/3)<=relative_angles[g]<(-np.pi/3):
                relative_angles[g]+=(2*np.pi/3)
            elif (-np.pi/3)<=relative_angles[g]<0:
                relative_angles[g]+=(np.pi/3)
            elif np.pi/3<relative_angles[g]<=(2*np.pi/3):
                relative_angles[g]-=(np.pi/3)
            elif (2*np.pi/3) < relative_angles[g]:
                relative_angles[g]-=(2*np.pi/3)

        # Create output dictionary for particle-based information on hexatic order of all particles
        hexatic_order_dict = {'order': order_param, 'theta': relative_angles}
        return hexatic_order_dict
    
    def voronoi(self):
        #IN PROGRESS
        '''
        Purpose: Takes the position of all particles in the system and uses neighbor lists to find the
        nearest, interacting neighbors of each reference particle and calculates the
        voronoi tesselation of each reference particle.

        Outputs:
        voronoi_dict: dictionary containing the voronoi tesselation, which
        quantifies the number of nearest neighbors and their local geometry.
        '''

        # Count particles per phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Set ideal number of neighbors to 6 for 2D HCP lattice
        voronoi = freud.locality.Voronoi()

        # Create temporary simulation box that only encompasses the dense phase
        dense_box = box.Box(Lx=np.abs(np.max(np.max(pos_dense[:,0]))-np.min(np.max(pos_dense[:,0])))+1.0, Ly=np.abs(np.max(np.max(pos_dense[:,1]))-np.min(np.max(pos_dense[:,1])))+1.0, is2D=True)

        # Compute voronoi tesselation for nearest neighbors
        voronoi.compute((self.f_box, pos_dense))

        #voronoi_polytopes = voronoi.particle_order

        """
        plt.figure(figsize=(8,7))
        ax = plt.gca()
        ax.set_xlim(np.min(pos_dense[:,0]), np.max(pos_dense[:,0]))
        ax.set_ylim(np.min(pos_dense[:,1]), np.max(pos_dense[:,1]))
        voronoi.plot(ax=ax)
        plt.show()
        stop
        """

        # Create output dictionary for particle-based information on voronoi tesselation of all particles
        voronoi_dict = {'order': voronoi_polytopes}
        return voronoi_dict

    def translational_order(self):
        '''
        Purpose: Takes the position of all particles in the system and uses neighbor lists to find the
        nearest, interacting neighbors of each reference particle and calculates the
        local translational order parameter of each reference particle.

        Outputs:
        trans_param: returns an array of translational order parameters calculated
        for each particle
        '''
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        # Locate potential neighbor particles in the system
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        # Generate neighbor list of all particles (per query args) neighboring all reference particles
        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        # Set ideal number of neighbors to 6 for 2D HCP lattice
        trans_order = freud.order.Translational(k=6)

        #Compute translational order parameter
        trans_order.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)

        trans_param = np.abs(trans_order.particle_order)

        return trans_param

    def steinhardt_order(self):
        '''
        Purpose: Takes the position of all particles in the system and uses neighbor lists to find the
        nearest, interacting neighbors of each reference particle and calculates the
        local Steinhardt order parameter of each reference particle.

        Outputs:
        stein_param: returns an array of Steinhardt order parameters calculated
        for each particle, which measures the degree of local crystallinity
        '''
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        # Locate potential neighbor particles in the system
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        # Generate neighbor list of all particles (per query args) neighboring all reference particles
        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        # Set ideal number of neighbors to 6 for 2D HCP lattice
        ql = freud.order.Steinhardt(l=6)

        #Compute Steinhardt order parameter
        ql.compute(system=(self.f_box, self.pos), neighbors=allall_bulk_nlist)
        stein_param = np.abs(ql.particle_order)

        return stein_param

    def nematic_order(self):
        '''
        Purpose: Takes the position of all particles in the system and uses neighbor lists to find the
        nearest, interacting neighbors of each reference particle and calculates the
        local nematic order parameter of each reference particle.

        Outputs:
        nematic_param: returns an array of nematic order parameters calculated
        for each particle, which measures the degree of local alignment
        '''
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='nearest', r_min = 0.1, num_neighbors=6)

        # Locate potential neighbor particles in the system
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))

        # Generate neighbor list of all particles (per query args) neighboring all reference particles
        allall_bulk_nlist = system_all.query(self.f_box.wrap(self.pos), query_args).toNeighborList()

        #Set x-axis to reference axis for calculating nematic order parameter
        nop = freud.order.Nematic([1, 0, 0])

        #Compute Nematic order parameter
        nop.compute(ori)
        nematic_param = np.abs(nop.order)

        return nematic_param

    

