
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
    def __init__(self, lx_box, ly_box, NBins_x, NBins_y, partNum, phase_dict, pos, typ, ang, part_dict, eps, peA, peB, parFrac, align_dict, area_frac_dict, press_dict):

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
        self.ang = ang

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

        # Initialize phase identification functions for call back later
        self.phase_ident_functs = phase_identification.phase_identification(self.area_frac_dict, self.align_dict, self.part_dict, self.press_dict, self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.parFrac, self.eps, self.typ)

        # Initialize particle property functions for call back later
        self.particle_prop_functs = particles.particle_props(self.lx_box, self.ly_box, self.partNum, self.NBins_x, self.NBins_y, self.peA, self.peB, self.eps, self.typ, self.pos, self.ang)

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
        pe_tot = 0
        pe_num = 0

        if part_ids is None:
            part_ids = self.binParts

        for i in part_ids:

            if self.typ[i]==0:
                pe_tot += self.peA
                pe_num += 1
            else:
                pe_tot += self.peB
                pe_num += 1

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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

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

        print(elapsed)
        # Calculate interparticle separation distances between A reference particles and all neighbors within bulk
        bulk_lats = self.utility_functs.sep_dist_arr(pos_bulk[allall_bulk_nlist.point_indices], pos_bulk[allall_bulk_nlist.query_point_indices])
        elapsed = time.time() - t
        

        print(elapsed)
        k_arr = np.linspace(0, 1, num=10)
        print(len(bulk_lats))
        print(type(len(bulk_lats)))
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

        print(ssf_all_sum / len(pos_bulk))
        print(ssf_all / len(pos_bulk))
        print(len(pos_bulk))
        stop
        for k in range(0, len(k_arr)):

            ssf_all[k] = np.sum(np.cos(k_arr[k] * bulk_lats))**2 + np.sum(np.sin(k_arr[k] * bulk_lats))**2
            ssf_all[k] = np.abs(np.sum(np.exp(k_arr[k] * bulk_lats * 1j)))**2


        print( (1/len(pos_bulk)) * ssf_all)

        np.einsum('i,k->k', k_arr, bulk_lats)
        stop
        
        # Calculate interparticle separation distances between all reference particles and all neighbors within bulk
        

    def structure_factor_freud(self):
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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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
                    AA_bulk_dot = np.append(AA_bulk_dot, np.sum(np.cos(ang_A_bulk[i]-ang_A_dense[AA_bulk_nlist.point_indices[loc]])))
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
                    BA_bulk_dot = np.append(BA_bulk_dot, np.sum(np.cos(ang_A_bulk[i]-ang_B_dense[BA_bulk_nlist.point_indices[loc]])))
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
                    AB_bulk_dot = np.append(AB_bulk_dot, np.sum(np.cos(ang_B_bulk[i]-ang_A_dense[AB_bulk_nlist.point_indices[loc]])))
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
                    BB_bulk_dot = np.append(BB_bulk_dot, np.sum(np.cos(ang_B_bulk[i]-ang_B_dense[BB_bulk_nlist.point_indices[loc]])))
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
                    AA_int_dot = np.append(AA_int_dot, np.sum(np.cos(ang_A_int[i]-ang_A[AA_int_nlist.point_indices[loc]])))
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
                    AB_int_dot = np.append(AB_int_dot, np.sum(np.cos(ang_B_int[i]-ang_A[AB_int_nlist.point_indices[loc]])))
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
                    BA_int_dot = np.append(BA_int_dot, np.sum(np.cos(ang_A_int[i]-ang_B[BA_int_nlist.point_indices[loc]])))
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
                    BB_int_dot = np.append(BB_int_dot, np.sum(np.cos(ang_B_int[i]-ang_B[BB_int_nlist.point_indices[loc]])))
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

        # Create output dictionary for statistical averages of total nearest neighbor numbers on each particle per phase/activity pairing
        neigh_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_num_neigh), 'std': np.std(allall_bulk_num_neigh)}, 'all-A': {'mean': np.mean(allA_bulk_num_neigh), 'std': np.std(allA_bulk_num_neigh)}, 'all-B': {'mean': np.mean(allB_bulk_num_neigh), 'std': np.std(allB_bulk_num_neigh)}, 'A-A': {'mean': np.mean(AA_bulk_num_neigh), 'std': np.std(AA_bulk_num_neigh)}, 'A-B': {'mean': np.mean(AB_bulk_num_neigh), 'std': np.std(AB_bulk_num_neigh)}, 'B-B': {'mean': np.mean(BB_bulk_num_neigh), 'std': np.std(BB_bulk_num_neigh)}}, 'int': {'all-all': {'mean': np.mean(allall_int_num_neigh), 'std': np.std(allall_int_num_neigh)}, 'all-A': {'mean': np.mean(allA_int_num_neigh), 'std': np.std(allA_int_num_neigh)}, 'all-B': {'mean': np.mean(allB_int_num_neigh), 'std': np.std(allB_int_num_neigh)}, 'A-A': {'mean': np.mean(AA_int_num_neigh), 'std': np.std(AA_int_num_neigh)}, 'A-B': {'mean': np.mean(AB_int_num_neigh), 'std': np.std(AB_int_num_neigh)}, 'B-B': {'mean': np.mean(BB_int_num_neigh), 'std': np.std(BB_int_num_neigh)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_num_neigh), 'std': np.std(allall_dense_num_neigh)}, 'all-A': {'mean': np.mean(allA_dense_num_neigh), 'std': np.std(allA_dense_num_neigh)}, 'all-B': {'mean': np.mean(allB_dense_num_neigh), 'std': np.std(allB_dense_num_neigh)}, 'A-A': {'mean': np.mean(AA_dense_num_neigh), 'std': np.std(AA_dense_num_neigh)}, 'A-B': {'mean': np.mean(AB_dense_num_neigh), 'std': np.std(AB_dense_num_neigh)}, 'B-B': {'mean': np.mean(BB_dense_num_neigh), 'std': np.std(BB_dense_num_neigh)}}}


        # Create output dictionary for statistical averages of total nearest neighbor orientational correlation on each particle per phase/activity pairing
        ori_stat_dict = {'bulk': {'all-all': {'mean': np.mean(allall_bulk_dot), 'std': np.std(allall_bulk_dot)}, 'all-A': {'mean': np.mean(allA_bulk_dot), 'std': np.std(allA_bulk_dot)}, 'all-B': {'mean': np.mean(allB_bulk_dot), 'std': np.std(allB_bulk_dot)}, 'A-A': {'mean': np.mean(AA_bulk_dot), 'std': np.std(AA_bulk_dot)}, 'A-B': {'mean': np.mean(AB_bulk_dot), 'std': np.std(AB_bulk_dot)}, 'B-B': {'mean': np.mean(BB_bulk_dot), 'std': np.std(BB_bulk_dot)}}, 'int': {'all-all': {'mean': np.mean(allall_int_dot), 'std': np.std(allall_int_dot)}, 'all-A': {'mean': np.mean(allA_int_dot), 'std': np.std(allA_int_dot)}, 'all-B': {'mean': np.mean(allB_int_dot), 'std': np.std(allB_int_dot)}, 'A-A': {'mean': np.mean(AA_int_dot), 'std': np.std(AA_int_dot)}, 'A-B': {'mean': np.mean(AB_int_dot), 'std': np.std(AB_int_dot)}, 'B-B': {'mean': np.mean(BB_int_dot), 'std': np.std(BB_int_dot)}}, 'dense': {'all-all': {'mean': np.mean(allall_dense_dot), 'std': np.std(allall_dense_dot)}, 'all-A': {'mean': np.mean(allA_dense_dot), 'std': np.std(allA_dense_dot)}, 'all-B': {'mean': np.mean(allB_dense_dot), 'std': np.std(allB_dense_dot)}, 'A-A': {'mean': np.mean(AA_dense_dot), 'std': np.std(AA_dense_dot)}, 'A-B': {'mean': np.mean(AB_dense_dot), 'std': np.std(AB_dense_dot)}, 'B-B': {'mean': np.mean(BB_dense_dot), 'std': np.std(BB_dense_dot)}}}

        # Create output dictionary for plotting of nearest neighbor information of each particle per phase/activity pairing and their respective x-y locations
        neigh_plot_dict = {'all-all': {'neigh': allall_dense_num_neigh, 'ori': allall_dense_dot, 'x': allall_dense_pos_x, 'y': allall_dense_pos_y}, 'all-A': {'neigh': allA_dense_num_neigh, 'ori': allA_dense_dot, 'x': allA_dense_pos_x, 'y': allA_dense_pos_y}, 'all-B': {'neigh': allB_dense_num_neigh, 'ori': allB_dense_dot, 'x': allB_dense_pos_x, 'y': allB_dense_pos_y}, 'A-all': {'neigh': Aall_dense_num_neigh, 'ori': Aall_dense_dot, 'x': Aall_dense_pos_x, 'y': Aall_dense_pos_y}, 'B-all': {'neigh': Ball_dense_num_neigh, 'ori': Ball_dense_dot, 'x': Ball_dense_pos_x, 'y': Ball_dense_pos_y}, 'A-A': {'neigh': AA_dense_num_neigh, 'ori': AA_dense_dot, 'x': pos_A_dense[:,0], 'y': pos_A_dense[:,1]}, 'A-B': {'neigh': AB_dense_num_neigh, 'ori': AB_dense_dot, 'x': pos_B_dense[:,0], 'y': pos_B_dense[:,1]}, 'B-A': {'neigh': BA_dense_num_neigh, 'ori': BA_dense_dot, 'x': pos_A_dense[:,0], 'y': pos_A_dense[:,1]}, 'B-B': {'neigh': BB_dense_num_neigh, 'ori': BB_dense_dot, 'x': pos_B_dense[:,0], 'y': pos_B_dense[:,1]}}

        
        return neigh_stat_dict, ori_stat_dict, neigh_plot_dict
    def local_gas_density(self):
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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        gas_area = phase_count_dict['gas'] * (self.sizeBin_x * self.sizeBin_y)
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)
        int_area = phase_count_dict['int'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around B reference particles
        AB_local_dens_mean_arr = []
        AB_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around A reference particles
        BA_local_dens_mean_arr = []
        BA_local_dens_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around B reference particles
        BB_local_dens_mean_arr = []
        BB_local_dens_std_arr = []

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_gas_num_neigh = np.zeros(len(pos_A_gas))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_gas_num_neigh = np.zeros(len(pos_B_gas))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_gas_num_neigh = np.zeros(len(pos_A_gas))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_gas_num_neigh = np.zeros(len(pos_B_gas))

        print('total_density')
        print(len(pos_gas)/gas_area)
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
            AA_gas_local_dens_inhomog = (AA_gas_local_dens - AA_local_dens_mean_arr[-1])**2

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around B reference particles in dense phase
            AB_local_dens_mean_arr.append(np.mean(AB_gas_local_dens))
            AB_local_dens_std_arr.append(np.std(AB_gas_local_dens))
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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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

    def interparticle_pressure_nlist(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the total interparticle stress all,
        interacting neighbors acting on each reference particle.

        Outputs:
        stress_stat_dict: dictionary containing the total interparticle stress
        between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B')
        in each direction ('XX', 'XY', 'YX', 'YY'), averaged over all reference particles in each phase.

        press_stat_dict: dictionary containing the average interparticle pressure and
        shear stress between a reference particle of a given type ('all', 'A', or 'B')
        and its nearest, interacting neighbors of each type ('all', 'A', or 'B'),
        averaged over all reference particles in each phase.

        stress_plot_dict: dictionary containing information on the interparticle pressure
        of each bulk and interface reference particle of each type ('all', 'A', or 'B').

        press_plot_dict: dictionary containing information on the interparticle stress
        and pressure of each bulk and interface reference particle of each type
        ('all', 'A', or 'B').
        '''
        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Count total number of bins in each phase
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Calculate area of each phase
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)
        int_area = phase_count_dict['int'] * (self.sizeBin_x * self.sizeBin_y)
        gas_area = phase_count_dict['gas'] * (self.sizeBin_x * self.sizeBin_y)
        dense_area = bulk_area + int_area
        system_area = bulk_area + int_area + gas_area

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_gas_int = self.pos[phase_part_dict['gas_int']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_gas_int = self.pos[phase_part_dict['gas_int']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]

        # Position and orientation arrays of all particles in respective phase
        pos_bulk = self.pos[phase_part_dict['bulk']['all']]
        pos_int = self.pos[phase_part_dict['int']['all']]
        pos_gas = self.pos[phase_part_dict['gas']['all']]
        pos_gas_int = self.pos[phase_part_dict['gas_int']['all']]
        pos_dense = self.pos[phase_part_dict['dense']['all']]

        # Position and orientation arrays of all particles in respective phase
        typ_bulk = self.typ[phase_part_dict['bulk']['all']]
        typ_int = self.typ[phase_part_dict['int']['all']]
        typ_gas = self.typ[phase_part_dict['gas']['all']]
        typ_gas_int = self.typ[phase_part_dict['gas_int']['all']]
        typ_dense = self.typ[phase_part_dict['dense']['all']]

        

        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        # Locate potential neighbor particles by type in the dense phase
        system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_dense))
        system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_dense))

        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        AB_bulk_nlist = system_A_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()
        BA_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_A_bulk), query_args).toNeighborList()
        BB_bulk_nlist = system_B_bulk.query(self.f_box.wrap(pos_B_bulk), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type A bulk particles
        AA_bulk_neigh_ind = np.array([], dtype=int)
        AA_bulk_num_neigh = np.array([])

        SigXX_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of normal stress in x direction
        SigXY_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of tangential stress in x-y direction
        SigYX_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of tangential stress in y-x direction
        SigYY_AA_bulk_part=np.zeros(len(pos_A_bulk))        #Sum of normal stress in y direction

        SigXX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in normal x direction
        SigXY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in tangential x-y direction
        SigYX_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in tangential y-x direction
        SigYY_AA_bulk_part_num=np.zeros(len(pos_A_bulk))    #Number of interparticle stresses summed in normal y direction

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_bulk)):
            if i in AA_bulk_nlist.query_point_indices:
                if i not in AA_bulk_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AA_bulk_nlist.point_indices[loc]], difxy=True)

                        #Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_bulk_part[i] += np.sum(SigXX)
                        SigYY_AA_bulk_part[i] += np.sum(SigYY)
                        SigXY_AA_bulk_part[i] += np.sum(SigXY)
                        SigYX_AA_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_bulk_part_num[i] += len(SigXX)
                        SigYY_AA_bulk_part_num[i] += len(SigYY)
                        SigXY_AA_bulk_part_num[i] += len(SigXY)
                        SigYX_AA_bulk_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_bulk[i][0], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_bulk[i][1], pos_A_dense[AA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle force
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_bulk_part[i] = SigXX
                        SigYY_AA_bulk_part[i] = SigYY
                        SigXY_AA_bulk_part[i] = SigXY
                        SigYX_AA_bulk_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_bulk_part_num[i] += 1
                        SigYY_AA_bulk_part_num[i] += 1
                        SigXY_AA_bulk_part_num[i] += 1
                        SigYX_AA_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, len(loc))
                    AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                AA_bulk_num_neigh = np.append(AA_bulk_num_neigh, 0)
                AA_bulk_neigh_ind = np.append(AA_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type A bulk particles
        BA_bulk_neigh_ind = np.array([], dtype=int)
        BA_bulk_num_neigh = np.array([])

        SigXX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part=np.zeros(len(pos_A_bulk))

        SigXX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigXY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYX_BA_bulk_part_num=np.zeros(len(pos_A_bulk))
        SigYY_BA_bulk_part_num=np.zeros(len(pos_A_bulk))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_bulk)):
            if i in BA_bulk_nlist.query_point_indices:
                if i not in BA_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BA_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_bulk_part[i] += np.sum(SigXX)
                        SigYY_BA_bulk_part[i] += np.sum(SigYY)
                        SigXY_BA_bulk_part[i] += np.sum(SigXY)
                        SigYX_BA_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_bulk_part_num[i] += len(SigXX)
                        SigYY_BA_bulk_part_num[i] += len(SigYY)
                        SigXY_BA_bulk_part_num[i] += len(SigXY)
                        SigYX_BA_bulk_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_bulk[i][0], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_bulk[i][1], pos_B_dense[BA_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_bulk_part[i] = SigXX
                        SigYY_BA_bulk_part[i] = SigYY
                        SigXY_BA_bulk_part[i] = SigXY
                        SigYX_BA_bulk_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_bulk_part_num[i] += 1
                        SigYY_BA_bulk_part_num[i] += 1
                        SigXY_BA_bulk_part_num[i] += 1
                        SigYX_BA_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, len(loc))
                    BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BA_bulk_num_neigh = np.append(BA_bulk_num_neigh, 0)
                BA_bulk_neigh_ind = np.append(BA_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type A dense particles acting on type B bulk particles
        AB_bulk_neigh_ind = np.array([], dtype=int)
        AB_bulk_num_neigh = np.array([])

        SigXX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_AB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_AB_bulk_part_num=np.zeros(len(pos_B_bulk))


        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_bulk)):
            if i in AB_bulk_nlist.query_point_indices:
                if i not in AB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_dense[AB_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_bulk_part[i] += np.sum(SigXX)
                        SigYY_AB_bulk_part[i] += np.sum(SigYY)
                        SigXY_AB_bulk_part[i] += np.sum(SigXY)
                        SigYX_AB_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_bulk_part_num[i] += len(SigXX)
                        SigYY_AB_bulk_part_num[i] += len(SigYY)
                        SigXY_AB_bulk_part_num[i] += len(SigXY)
                        SigYX_AB_bulk_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_bulk[i][0], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_bulk[i][1], pos_A_dense[AB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_bulk_part[i] += SigXX
                        SigYY_AB_bulk_part[i] += SigYY
                        SigXY_AB_bulk_part[i] += SigXY
                        SigYX_AB_bulk_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_bulk_part_num[i] += 1
                        SigYY_AB_bulk_part_num[i] += 1
                        SigXY_AB_bulk_part_num[i] += 1
                        SigYX_AB_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, len(loc))
                    AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_bulk_num_neigh = np.append(AB_bulk_num_neigh, 0)
                AB_bulk_neigh_ind = np.append(AB_bulk_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B dense particles acting on type B bulk particles
        BB_bulk_neigh_ind = np.array([], dtype=int)
        BB_bulk_num_neigh = np.array([])

        SigXX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part=np.zeros(len(pos_B_bulk))

        SigXX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigXY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYX_BB_bulk_part_num=np.zeros(len(pos_B_bulk))
        SigYY_BB_bulk_part_num=np.zeros(len(pos_B_bulk))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_bulk)):
            if i in BB_bulk_nlist.query_point_indices:
                if i not in BB_bulk_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_bulk_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_bulk[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_dense[BB_bulk_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_bulk_part[i] += np.sum(SigXX)
                        SigYY_BB_bulk_part[i] += np.sum(SigYY)
                        SigXY_BB_bulk_part[i] += np.sum(SigXY)
                        SigYX_BB_bulk_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_bulk_part_num[i] += len(SigXX)
                        SigYY_BB_bulk_part_num[i] += len(SigYY)
                        SigXY_BB_bulk_part_num[i] += len(SigXY)
                        SigYX_BB_bulk_part_num[i] += len(SigYX)

                    else:
                        difx = self.utility_functs.sep_dist_x(pos_B_bulk[i][0], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_bulk[i][1], pos_B_dense[BB_bulk_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_bulk_part[i] += SigXX
                        SigYY_BB_bulk_part[i] += SigYY
                        SigXY_BB_bulk_part[i] += SigXY
                        SigYX_BB_bulk_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_bulk_part_num[i] += 1
                        SigYY_BB_bulk_part_num[i] += 1
                        SigXY_BB_bulk_part_num[i] += 1
                        SigYX_BB_bulk_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, len(loc))
                    BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BB_bulk_num_neigh = np.append(BB_bulk_num_neigh, 0)
                BB_bulk_neigh_ind = np.append(BB_bulk_neigh_ind, int(i))

        # Locate potential neighbor particles by type in the entire system
        system_A_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B_int = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        # Generate neighbor list of any phase particles (per query args) of respective type (A or B) neighboring interface phase reference particles of respective type (A or B)
        AA_int_nlist = system_A_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        AB_int_nlist = system_A_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()
        BA_int_nlist = system_B_int.query(self.f_box.wrap(pos_A_int), query_args).toNeighborList()
        BB_int_nlist = system_B_int.query(self.f_box.wrap(pos_B_int), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A particles acting on type A interface particles
        AA_int_neigh_ind = np.array([], dtype=int)
        AA_int_num_neigh = np.array([])

        SigXX_AA_int_part=np.zeros(len(pos_A_int))
        SigXY_AA_int_part=np.zeros(len(pos_A_int))
        SigYX_AA_int_part=np.zeros(len(pos_A_int))
        SigYY_AA_int_part=np.zeros(len(pos_A_int))

        SigXX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_AA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_AA_int_part_num=np.zeros(len(pos_A_int))

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_int)):
            if i in AA_int_nlist.query_point_indices:
                if i not in AA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AA_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_int_part[i] += np.sum(SigXX)
                        SigYY_AA_int_part[i] += np.sum(SigYY)
                        SigXY_AA_int_part[i] += np.sum(SigXY)
                        SigYX_AA_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_int_part_num[i] += len(SigXX)
                        SigYY_AA_int_part_num[i] += len(SigYY)
                        SigXY_AA_int_part_num[i] += len(SigXY)
                        SigYX_AA_int_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_int[i][0], pos_A[AA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_int[i][1], pos_A[AA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_int_part[i] = SigXX
                        SigYY_AA_int_part[i] = SigYY
                        SigXY_AA_int_part[i] = SigXY
                        SigYX_AA_int_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_int_part_num[i] += 1
                        SigYY_AA_int_part_num[i] += 1
                        SigXY_AA_int_part_num[i] += 1
                        SigYX_AA_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_int_num_neigh = np.append(AA_int_num_neigh, len(loc))
                    AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AA_int_num_neigh = np.append(AA_int_num_neigh, 0)
                AA_int_neigh_ind = np.append(AA_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type A interface particles
        BA_int_neigh_ind = np.array([], dtype=int)
        BA_int_num_neigh = np.array([])

        SigXX_BA_int_part=np.zeros(len(pos_A_int))
        SigXY_BA_int_part=np.zeros(len(pos_A_int))
        SigYX_BA_int_part=np.zeros(len(pos_A_int))
        SigYY_BA_int_part=np.zeros(len(pos_A_int))

        SigXX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigXY_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYX_BA_int_part_num=np.zeros(len(pos_A_int))
        SigYY_BA_int_part_num=np.zeros(len(pos_A_int))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_int)):
            if i in BA_int_nlist.query_point_indices:
                if i not in BA_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BA_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_int_part[i] += np.sum(SigXX)
                        SigYY_BA_int_part[i] += np.sum(SigYY)
                        SigXY_BA_int_part[i] += np.sum(SigXY)
                        SigYX_BA_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_int_part_num[i] += len(SigXX)
                        SigYY_BA_int_part_num[i] += len(SigYY)
                        SigXY_BA_int_part_num[i] += len(SigXY)
                        SigYX_BA_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_int[i][0], pos_B[BA_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_int[i][1], pos_B[BA_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_int_part[i] = SigXX
                        SigYY_BA_int_part[i] = SigYY
                        SigXY_BA_int_part[i] = SigXY
                        SigYX_BA_int_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_int_part_num[i] += 1
                        SigYY_BA_int_part_num[i] += 1
                        SigXY_BA_int_part_num[i] += 1
                        SigYX_BA_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_int_num_neigh = np.append(BA_int_num_neigh, len(loc))
                    BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BA_int_num_neigh = np.append(BA_int_num_neigh, 0)
                BA_int_neigh_ind = np.append(BA_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type A interface particles
        AB_int_neigh_ind = np.array([], dtype=int)
        AB_int_num_neigh = np.array([])

        SigXX_AB_int_part=np.zeros(len(pos_B_int))
        SigXY_AB_int_part=np.zeros(len(pos_B_int))
        SigYX_AB_int_part=np.zeros(len(pos_B_int))
        SigYY_AB_int_part=np.zeros(len(pos_B_int))

        SigXX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_AB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_AB_int_part_num=np.zeros(len(pos_B_int))

        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_int)):
            if i in AB_int_nlist.query_point_indices:
                if i not in AB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A[AB_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_int_part[i] += np.sum(SigXX)
                        SigYY_AB_int_part[i] += np.sum(SigYY)
                        SigXY_AB_int_part[i] += np.sum(SigXY)
                        SigYX_AB_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_int_part_num[i] += len(SigXX)
                        SigYY_AB_int_part_num[i] += len(SigYY)
                        SigXY_AB_int_part_num[i] += len(SigXY)
                        SigYX_AB_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_int[i][0], pos_A[AB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_int[i][1], pos_A[AB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_int_part[i] += SigXX
                        SigYY_AB_int_part[i] += SigYY
                        SigXY_AB_int_part[i] += SigXY
                        SigYX_AB_int_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_int_part_num[i] += 1
                        SigYY_AB_int_part_num[i] += 1
                        SigXY_AB_int_part_num[i] += 1
                        SigYX_AB_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_int_num_neigh = np.append(AB_int_num_neigh, len(loc))
                    AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_int_num_neigh = np.append(AB_int_num_neigh, 0)
                AB_int_neigh_ind = np.append(AB_int_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B particles acting on type B interface particles
        BB_int_neigh_ind = np.array([], dtype=int)
        BB_int_num_neigh = np.array([])

        SigXX_BB_int_part=np.zeros(len(pos_B_int))
        SigXY_BB_int_part=np.zeros(len(pos_B_int))
        SigYX_BB_int_part=np.zeros(len(pos_B_int))
        SigYY_BB_int_part=np.zeros(len(pos_B_int))

        SigXX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigXY_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYX_BB_int_part_num=np.zeros(len(pos_B_int))
        SigYY_BB_int_part_num=np.zeros(len(pos_B_int))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_int)):
            if i in BB_int_nlist.query_point_indices:
                if i not in BB_int_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_int_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_int[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B[BB_int_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_int_part[i] += np.sum(SigXX)
                        SigYY_BB_int_part[i] += np.sum(SigYY)
                        SigXY_BB_int_part[i] += np.sum(SigXY)
                        SigYX_BB_int_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_int_part_num[i] += len(SigXX)
                        SigYY_BB_int_part_num[i] += len(SigYY)
                        SigXY_BB_int_part_num[i] += len(SigXY)
                        SigYX_BB_int_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_int[i][0], pos_B[BB_int_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_int[i][1], pos_B[BB_int_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_int_part[i] += SigXX
                        SigYY_BB_int_part[i] += SigYY
                        SigXY_BB_int_part[i] += SigXY
                        SigYX_BB_int_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_int_part_num[i] += 1
                        SigYY_BB_int_part_num[i] += 1
                        SigXY_BB_int_part_num[i] += 1
                        SigYX_BB_int_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_int_num_neigh = np.append(BB_int_num_neigh, len(loc))
                    BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BB_int_num_neigh = np.append(BB_int_num_neigh, 0)
                BB_int_neigh_ind = np.append(BB_int_neigh_ind, int(i))

        # Locate potential neighbor particles by type in the gas and interface phases
        system_A_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A_gas_int))
        system_B_gas = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B_gas_int))

        # Generate neighbor list of gas and interface phase particles (per query args) of respective type (A or B) neighboring gas phase reference particles of respective type (A or B)
        AA_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        AB_gas_nlist = system_A_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()
        BA_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_A_gas), query_args).toNeighborList()
        BB_gas_nlist = system_B_gas.query(self.f_box.wrap(pos_B_gas), query_args).toNeighborList()

        #Initiate empty arrays for finding stress in given direction from type A gas and interface particles acting on type A gas particles
        AA_gas_neigh_ind = np.array([], dtype=int)
        AA_gas_num_neigh = np.array([])

        SigXX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_AA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_AA_gas_part_num=np.zeros(len(pos_A_gas))

        #Loop over neighbor pairings of A-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_gas)):
            if i in AA_gas_nlist.query_point_indices:
                if i not in AA_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AA_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_gas_part[i] += np.sum(SigXX)
                        SigYY_AA_gas_part[i] += np.sum(SigYY)
                        SigXY_AA_gas_part[i] += np.sum(SigXY)
                        SigYX_AA_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_gas_part_num[i] += len(SigXX)
                        SigYY_AA_gas_part_num[i] += len(SigYY)
                        SigXY_AA_gas_part_num[i] += len(SigXY)
                        SigYX_AA_gas_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_gas[i][0], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_gas[i][1], pos_A_gas_int[AA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AA_gas_part[i] = SigXX
                        SigYY_AA_gas_part[i] = SigYY
                        SigXY_AA_gas_part[i] = SigXY
                        SigYX_AA_gas_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AA_gas_part_num[i] += 1
                        SigYY_AA_gas_part_num[i] += 1
                        SigXY_AA_gas_part_num[i] += 1
                        SigYX_AA_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AA_gas_num_neigh = np.append(AA_gas_num_neigh, len(loc))
                    AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AA_gas_num_neigh = np.append(AA_gas_num_neigh, 0)
                AA_gas_neigh_ind = np.append(AA_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B gas and interface particles acting on type A gas particles
        BA_gas_neigh_ind = np.array([], dtype=int)
        BA_gas_num_neigh = np.array([])

        SigXX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part=np.zeros(len(pos_A_gas))

        SigXX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigXY_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYX_BA_gas_part_num=np.zeros(len(pos_A_gas))
        SigYY_BA_gas_part_num=np.zeros(len(pos_A_gas))

        #Loop over neighbor pairings of B-A neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_A_gas)):
            if i in BA_gas_nlist.query_point_indices:
                if i not in BA_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_A_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BA_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_gas_part[i] += np.sum(SigXX)
                        SigYY_BA_gas_part[i] += np.sum(SigYY)
                        SigXY_BA_gas_part[i] += np.sum(SigXY)
                        SigYX_BA_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_gas_part_num[i] += len(SigXX)
                        SigYY_BA_gas_part_num[i] += len(SigYY)
                        SigXY_BA_gas_part_num[i] += len(SigXY)
                        SigYX_BA_gas_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_A_gas[i][0], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_A_gas[i][1], pos_B_gas_int[BA_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BA_gas_part[i] = SigXX
                        SigYY_BA_gas_part[i] = SigYY
                        SigXY_BA_gas_part[i] = SigXY
                        SigYX_BA_gas_part[i] = SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BA_gas_part_num[i] += 1
                        SigYY_BA_gas_part_num[i] += 1
                        SigXY_BA_gas_part_num[i] += 1
                        SigYX_BA_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BA_gas_num_neigh = np.append(BA_gas_num_neigh, len(loc))
                    BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                BA_gas_num_neigh = np.append(BA_gas_num_neigh, 0)
                BA_gas_neigh_ind = np.append(BA_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type A gas and interface particles acting on type B gas particles
        AB_gas_neigh_ind = np.array([], dtype=int)
        AB_gas_num_neigh = np.array([])

        SigXX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_AB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_AB_gas_part_num=np.zeros(len(pos_B_gas))

        #Loop over neighbor pairings of A-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_gas)):
            if i in AB_gas_nlist.query_point_indices:
                if i not in AB_gas_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_A_gas_int[AB_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_gas_part[i] += np.sum(SigXX)
                        SigYY_AB_gas_part[i] += np.sum(SigYY)
                        SigXY_AB_gas_part[i] += np.sum(SigXY)
                        SigYX_AB_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_gas_part_num[i] += len(SigXX)
                        SigYY_AB_gas_part_num[i] += len(SigYY)
                        SigXY_AB_gas_part_num[i] += len(SigXY)
                        SigYX_AB_gas_part_num[i] += len(SigYX)

                    else:
                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_gas[i][0], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_gas[i][1], pos_A_gas_int[AB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_AB_gas_part[i] += SigXX
                        SigYY_AB_gas_part[i] += SigYY
                        SigXY_AB_gas_part[i] += SigXY
                        SigYX_AB_gas_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_AB_gas_part_num[i] += 1
                        SigYY_AB_gas_part_num[i] += 1
                        SigXY_AB_gas_part_num[i] += 1
                        SigYX_AB_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    AB_gas_num_neigh = np.append(AB_gas_num_neigh, len(loc))
                    AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))
            else:
                # Save nearest neighbor information for i reference particle
                AB_gas_num_neigh = np.append(AB_gas_num_neigh, 0)
                AB_gas_neigh_ind = np.append(AB_gas_neigh_ind, int(i))

        #Initiate empty arrays for finding stress in given direction from type B gas and interface particles acting on type B gas particles
        BB_gas_neigh_ind = np.array([], dtype=int)
        BB_gas_num_neigh = np.array([])

        SigXX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part=np.zeros(len(pos_B_gas))

        SigXX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigXY_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYX_BB_gas_part_num=np.zeros(len(pos_B_gas))
        SigYY_BB_gas_part_num=np.zeros(len(pos_B_gas))

        #Loop over neighbor pairings of B-B neighbor pairs to calculate interparticle stress in each direction
        for i in range(0, len(pos_B_gas)):
            if i in BB_gas_nlist.query_point_indices:
                if i not in BB_gas_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_gas_nlist.query_point_indices==i)[0]
                    if len(loc)>1:

                        # Array of reference particle location
                        pos_temp = np.ones((len(loc), 3))* pos_B_gas[i]

                        # Calculate interparticle separation distances
                        difx, dify, difr = self.utility_functs.sep_dist_arr(pos_temp, pos_B_gas_int[BB_gas_nlist.point_indices[loc]], difxy=True)

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ_arr(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_gas_part[i] += np.sum(SigXX)
                        SigYY_BB_gas_part[i] += np.sum(SigYY)
                        SigXY_BB_gas_part[i] += np.sum(SigXY)
                        SigYX_BB_gas_part[i] += np.sum(SigYX)

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_gas_part_num[i] += len(SigXX)
                        SigYY_BB_gas_part_num[i] += len(SigYY)
                        SigXY_BB_gas_part_num[i] += len(SigXY)
                        SigYX_BB_gas_part_num[i] += len(SigYX)

                    else:

                        # Calculate interparticle separation distances
                        difx = self.utility_functs.sep_dist_x(pos_B_gas[i][0], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][0])
                        dify = self.utility_functs.sep_dist_y(pos_B_gas[i][1], pos_B_gas_int[BB_gas_nlist.point_indices[loc]][0][1])
                        difr = ( (difx)**2 + (dify)**2)**0.5

                        # Calculate interparticle separation forces
                        fx, fy = self.theory_functs.computeFLJ(difr, difx, dify, self.eps)

                        # Calculate array of stresses for each neighbor list pairing acting on i reference particle
                        SigXX = (fx * difx)
                        SigYY = (fy * dify)
                        SigXY = (fx * dify)
                        SigYX = (fy * difx)

                        # Calculate total stress acting on i reference particle
                        SigXX_BB_gas_part[i] += SigXX
                        SigYY_BB_gas_part[i] += SigYY
                        SigXY_BB_gas_part[i] += SigXY
                        SigYX_BB_gas_part[i] += SigYX

                        # Calculate number of neighbor pairs summed over
                        SigXX_BB_gas_part_num[i] += 1
                        SigYY_BB_gas_part_num[i] += 1
                        SigXY_BB_gas_part_num[i] += 1
                        SigYX_BB_gas_part_num[i] += 1

                    # Save nearest neighbor information for i reference particle
                    BB_gas_num_neigh = np.append(BB_gas_num_neigh, len(loc))
                    BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))
            else:

                # Save nearest neighbor information for i reference particle
                BB_gas_num_neigh = np.append(BB_gas_num_neigh, 0)
                BB_gas_neigh_ind = np.append(BB_gas_neigh_ind, int(i))


        ###Bulk stress

        # Calculate total stress and number of neighbor pairs summed over for B bulk reference particles and all dense neighbors
        allB_bulk_SigXX_part = SigXX_BB_bulk_part + SigXX_AB_bulk_part
        allB_bulk_SigXX_part_num = SigXX_BB_bulk_part_num + SigXX_AB_bulk_part_num
        allB_bulk_SigXY_part = SigXY_BB_bulk_part + SigXY_AB_bulk_part
        allB_bulk_SigXY_part_num = SigXY_BB_bulk_part_num + SigXY_AB_bulk_part_num
        allB_bulk_SigYX_part = SigYX_BB_bulk_part + SigYX_AB_bulk_part
        allB_bulk_SigYX_part_num = SigYX_BB_bulk_part_num + SigYX_AB_bulk_part_num
        allB_bulk_SigYY_part = SigYY_BB_bulk_part + SigYY_AB_bulk_part
        allB_bulk_SigYY_part_num = SigYY_BB_bulk_part_num + SigYY_AB_bulk_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and B dense neighbors
        Ball_bulk_SigXX_part = np.append(SigXX_BA_bulk_part, SigXX_BB_bulk_part)
        Ball_bulk_SigXX_part_num = np.append(SigXX_BA_bulk_part_num, SigXX_BB_bulk_part_num)
        Ball_bulk_SigXY_part = np.append(SigXY_BA_bulk_part, SigXY_BB_bulk_part)
        Ball_bulk_SigXY_part_num = np.append(SigXY_BA_bulk_part_num, SigXY_BB_bulk_part_num)
        Ball_bulk_SigYX_part = np.append(SigYX_BA_bulk_part, SigYX_BB_bulk_part)
        Ball_bulk_SigYX_part_num = np.append(SigYX_BA_bulk_part_num, SigYX_BB_bulk_part_num)
        Ball_bulk_SigYY_part = np.append(SigYY_BA_bulk_part, SigYY_BB_bulk_part)
        Ball_bulk_SigYY_part_num = np.append(SigYY_BA_bulk_part_num, SigYY_BB_bulk_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A bulk reference particles and all dense neighbors
        allA_bulk_SigXX_part = SigXX_AA_bulk_part + SigXX_BA_bulk_part
        allA_bulk_SigXX_part_num = SigXX_AA_bulk_part_num + SigXX_BA_bulk_part_num
        allA_bulk_SigXY_part = SigXY_AA_bulk_part + SigXY_BA_bulk_part
        allA_bulk_SigXY_part_num = SigXY_AA_bulk_part_num + SigXY_BA_bulk_part_num
        allA_bulk_SigYX_part = SigYX_AA_bulk_part + SigYX_BA_bulk_part
        allA_bulk_SigYX_part_num = SigYX_AA_bulk_part_num + SigYX_BA_bulk_part_num
        allA_bulk_SigYY_part = SigYY_AA_bulk_part + SigYY_BA_bulk_part
        allA_bulk_SigYY_part_num = SigYY_AA_bulk_part_num + SigYY_BA_bulk_part_num

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and A dense neighbors
        Aall_bulk_SigXX_part = np.append(SigXX_AB_bulk_part, SigXX_AA_bulk_part)
        Aall_bulk_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_bulk_part_num)
        Aall_bulk_SigXY_part = np.append(SigXY_AB_bulk_part, SigXY_AA_bulk_part)
        Aall_bulk_SigXY_part_num = np.append(SigXY_AB_bulk_part_num, SigXY_AA_bulk_part_num)
        Aall_bulk_SigYX_part = np.append(SigYX_AB_bulk_part, SigYX_AA_bulk_part)
        Aall_bulk_SigYX_part_num = np.append(SigYX_AB_bulk_part_num, SigYX_AA_bulk_part_num)
        Aall_bulk_SigYY_part = np.append(SigYY_AB_bulk_part, SigYY_AA_bulk_part)
        Aall_bulk_SigYY_part_num = np.append(SigYY_AB_bulk_part_num, SigYY_AA_bulk_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all bulk reference particles and all dense neighbors
        allall_bulk_SigXX_part = np.append(allA_bulk_SigXX_part, allB_bulk_SigXX_part)
        allall_bulk_SigXX_part_num = np.append(allA_bulk_SigXX_part_num, allB_bulk_SigXX_part_num)
        allall_bulk_SigXY_part = np.append(allA_bulk_SigXY_part, allB_bulk_SigXY_part)
        allall_bulk_SigXY_part_num = np.append(allA_bulk_SigXY_part_num, allB_bulk_SigXY_part_num)
        allall_bulk_SigYX_part = np.append(allA_bulk_SigYX_part, allB_bulk_SigYX_part)
        allall_bulk_SigYX_part_num = np.append(allA_bulk_SigYX_part_num, allB_bulk_SigYX_part_num)
        allall_bulk_SigYY_part = np.append(allA_bulk_SigYY_part, allB_bulk_SigYY_part)
        allall_bulk_SigYY_part_num = np.append(allA_bulk_SigYY_part_num, allB_bulk_SigYY_part_num)

        ###Interface stress

        # Calculate total stress and number of neighbor pairs summed over for B interface reference particles and all neighbors
        allB_int_SigXX_part = SigXX_BB_int_part + SigXX_AB_int_part
        allB_int_SigXX_part_num = SigXX_BB_int_part_num + SigXX_AB_int_part_num
        allB_int_SigXY_part = SigXY_BB_int_part + SigXY_AB_int_part
        allB_int_SigXY_part_num = SigXY_BB_int_part_num + SigXY_AB_int_part_num
        allB_int_SigYX_part = SigYX_BB_int_part + SigYX_AB_int_part
        allB_int_SigYX_part_num = SigYX_BB_int_part_num + SigYX_AB_int_part_num
        allB_int_SigYY_part = SigYY_BB_int_part + SigYY_AB_int_part
        allB_int_SigYY_part_num = SigYY_BB_int_part_num + SigYY_AB_int_part_num

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and B neighbors
        Ball_int_SigXX_part = np.append(SigXX_BA_int_part, SigXX_BB_int_part)
        Ball_int_SigXX_part_num = np.append(SigXX_BA_int_part_num, SigXX_BB_int_part_num)
        Ball_int_SigXY_part = np.append(SigXY_BA_int_part, SigXY_BB_int_part)
        Ball_int_SigXY_part_num = np.append(SigXY_BA_int_part_num, SigXY_BB_int_part_num)
        Ball_int_SigYX_part = np.append(SigYX_BA_int_part, SigYX_BB_int_part)
        Ball_int_SigYX_part_num = np.append(SigYX_BA_int_part_num, SigYX_BB_int_part_num)
        Ball_int_SigYY_part = np.append(SigYY_BA_int_part, SigYY_BB_int_part)
        Ball_int_SigYY_part_num = np.append(SigYY_BA_int_part_num, SigYY_BB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A interface reference particles and all neighbors
        allA_int_SigXX_part = SigXX_AA_int_part + SigXX_BA_int_part
        allA_int_SigXX_part_num = SigXX_AA_int_part_num + SigXX_BA_int_part_num
        allA_int_SigXY_part = SigXY_AA_int_part + SigXY_BA_int_part
        allA_int_SigXY_part_num = SigXY_AA_int_part_num + SigXY_BA_int_part_num
        allA_int_SigYX_part = SigYX_AA_int_part + SigYX_BA_int_part
        allA_int_SigYX_part_num = SigYX_AA_int_part_num + SigYX_BA_int_part_num
        allA_int_SigYY_part = SigYY_AA_int_part + SigYY_BA_int_part
        allA_int_SigYY_part_num = SigYY_AA_int_part_num + SigYY_BA_int_part_num

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and A neighbors
        Aall_int_SigXX_part = np.append(SigXX_AB_int_part, SigXX_AA_int_part)
        Aall_int_SigXX_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AA_int_part_num)
        Aall_int_SigXY_part = np.append(SigXY_AB_int_part, SigXY_AA_int_part)
        Aall_int_SigXY_part_num = np.append(SigXY_AB_int_part_num, SigXY_AA_int_part_num)
        Aall_int_SigYX_part = np.append(SigYX_AB_int_part, SigYX_AA_int_part)
        Aall_int_SigYX_part_num = np.append(SigYX_AB_int_part_num, SigYX_AA_int_part_num)
        Aall_int_SigYY_part = np.append(SigYY_AB_int_part, SigYY_AA_int_part)
        Aall_int_SigYY_part_num = np.append(SigYY_AB_int_part_num, SigYY_AA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all interface reference particles and all neighbors
        allall_int_SigXX_part = np.append(allA_int_SigXX_part, allB_int_SigXX_part)
        allall_int_SigXX_part_num = np.append(allA_int_SigXX_part_num, allB_int_SigXX_part_num)
        allall_int_SigXY_part = np.append(allA_int_SigXY_part, allB_int_SigXY_part)
        allall_int_SigXY_part_num = np.append(allA_int_SigXY_part_num, allB_int_SigXY_part_num)
        allall_int_SigYX_part = np.append(allA_int_SigYX_part, allB_int_SigYX_part)
        allall_int_SigYX_part_num = np.append(allA_int_SigYX_part_num, allB_int_SigYX_part_num)
        allall_int_SigYY_part = np.append(allA_int_SigYY_part, allB_int_SigYY_part)
        allall_int_SigYY_part_num = np.append(allA_int_SigYY_part_num, allB_int_SigYY_part_num)

        ###Gas stress

        # Calculate total stress and number of neighbor pairs summed over for B gas reference particles and all gas and interface neighbors
        allB_gas_SigXX_part = SigXX_BB_gas_part + SigXX_AB_gas_part
        allB_gas_SigXX_part_num = SigXX_BB_gas_part_num + SigXX_AB_gas_part_num
        allB_gas_SigXY_part = SigXY_BB_gas_part + SigXY_AB_gas_part
        allB_gas_SigXY_part_num = SigXY_BB_gas_part_num + SigXY_AB_gas_part_num
        allB_gas_SigYX_part = SigYX_BB_gas_part + SigYX_AB_gas_part
        allB_gas_SigYX_part_num = SigYX_BB_gas_part_num + SigYX_AB_gas_part_num
        allB_gas_SigYY_part = SigYY_BB_gas_part + SigYY_AB_gas_part
        allB_gas_SigYY_part_num = SigYY_BB_gas_part_num + SigYY_AB_gas_part_num

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and B gas and interface neighbors
        Ball_gas_SigXX_part = np.append(SigXX_BA_gas_part, SigXX_BB_gas_part)
        Ball_gas_SigXX_part_num = np.append(SigXX_BA_gas_part_num, SigXX_BB_gas_part_num)
        Ball_gas_SigXY_part = np.append(SigXY_BA_gas_part, SigXY_BB_gas_part)
        Ball_gas_SigXY_part_num = np.append(SigXY_BA_gas_part_num, SigXY_BB_gas_part_num)
        Ball_gas_SigYX_part = np.append(SigYX_BA_gas_part, SigYX_BB_gas_part)
        Ball_gas_SigYX_part_num = np.append(SigYX_BA_gas_part_num, SigYX_BB_gas_part_num)
        Ball_gas_SigYY_part = np.append(SigYY_BA_gas_part, SigYY_BB_gas_part)
        Ball_gas_SigYY_part_num = np.append(SigYY_BA_gas_part_num, SigYY_BB_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A gas reference particles and all gas and interface neighbors
        allA_gas_SigXX_part = SigXX_AA_gas_part + SigXX_BA_gas_part
        allA_gas_SigXX_part_num = SigXX_AA_gas_part_num + SigXX_BA_gas_part_num
        allA_gas_SigXY_part = SigXY_AA_gas_part + SigXY_BA_gas_part
        allA_gas_SigXY_part_num = SigXY_AA_gas_part_num + SigXY_BA_gas_part_num
        allA_gas_SigYX_part = SigYX_AA_gas_part + SigYX_BA_gas_part
        allA_gas_SigYX_part_num = SigYX_AA_gas_part_num + SigYX_BA_gas_part_num
        allA_gas_SigYY_part = SigYY_AA_gas_part + SigYY_BA_gas_part
        allA_gas_SigYY_part_num = SigYY_AA_gas_part_num + SigYY_BA_gas_part_num

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and A gas and interface neighbors
        Aall_gas_SigXX_part = np.append(SigXX_AB_gas_part, SigXX_AA_gas_part)
        Aall_gas_SigXX_part_num = np.append(SigXX_AB_gas_part_num, SigXX_AA_gas_part_num)
        Aall_gas_SigXY_part = np.append(SigXY_AB_gas_part, SigXY_AA_gas_part)
        Aall_gas_SigXY_part_num = np.append(SigXY_AB_gas_part_num, SigXY_AA_gas_part_num)
        Aall_gas_SigYX_part = np.append(SigYX_AB_gas_part, SigYX_AA_gas_part)
        Aall_gas_SigYX_part_num = np.append(SigYX_AB_gas_part_num, SigYX_AA_gas_part_num)
        Aall_gas_SigYY_part = np.append(SigYY_AB_gas_part, SigYY_AA_gas_part)
        Aall_gas_SigYY_part_num = np.append(SigYY_AB_gas_part_num, SigYY_AA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all gas reference particles and all gas and interface neighbors
        allall_gas_SigXX_part = np.append(allA_gas_SigXX_part, allB_gas_SigXX_part)
        allall_gas_SigXX_part_num = np.append(allA_gas_SigXX_part_num, allB_gas_SigXX_part_num)
        allall_gas_SigXY_part = np.append(allA_gas_SigXY_part, allB_gas_SigXY_part)
        allall_gas_SigXY_part_num = np.append(allA_gas_SigXY_part_num, allB_gas_SigXY_part_num)
        allall_gas_SigYX_part = np.append(allA_gas_SigYX_part, allB_gas_SigYX_part)
        allall_gas_SigYX_part_num = np.append(allA_gas_SigYX_part_num, allB_gas_SigYX_part_num)
        allall_gas_SigYY_part = np.append(allA_gas_SigYY_part, allB_gas_SigYY_part)
        allall_gas_SigYY_part_num = np.append(allA_gas_SigYY_part_num, allB_gas_SigYY_part_num)

        ###Dense stress

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and all neighbors
        allB_dense_SigXX_part = np.append(allB_bulk_SigXX_part, allB_int_SigXX_part)
        allB_dense_SigXX_part_num = np.append(allB_bulk_SigXX_part_num, allB_int_SigXX_part_num)
        allB_dense_SigYX_part = np.append(allB_bulk_SigYX_part, allB_int_SigYX_part)
        allB_dense_SigYX_part_num = np.append(allB_bulk_SigYX_part_num, allB_int_SigYX_part_num)
        allB_dense_SigXY_part = np.append(allB_bulk_SigXY_part, allB_int_SigXY_part)
        allB_dense_SigXY_part_num = np.append(allB_bulk_SigXY_part_num, allB_int_SigXY_part_num)
        allB_dense_SigYY_part = np.append(allB_bulk_SigYY_part, allB_int_SigYY_part)
        allB_dense_SigYY_part_num = np.append(allB_bulk_SigYY_part_num, allB_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and B neighbors
        Ball_dense_SigXX_part = np.append(Ball_bulk_SigXX_part, Ball_int_SigXX_part)
        Ball_dense_SigXX_part_num = np.append(Ball_bulk_SigXX_part_num, Ball_int_SigXX_part_num)
        Ball_dense_SigYX_part = np.append(Ball_bulk_SigYX_part, Ball_int_SigYX_part)
        Ball_dense_SigYX_part_num = np.append(Ball_bulk_SigYX_part_num, Ball_int_SigYX_part_num)
        Ball_dense_SigXY_part = np.append(Ball_bulk_SigXY_part, Ball_int_SigXY_part)
        Ball_dense_SigXY_part_num = np.append(Ball_bulk_SigXY_part_num, Ball_int_SigXY_part_num)
        Ball_dense_SigYY_part = np.append(Ball_bulk_SigYY_part, Ball_int_SigYY_part)
        Ball_dense_SigYY_part_num = np.append(Ball_bulk_SigYY_part_num, Ball_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and all neighbors
        allA_dense_SigXX_part = np.append(allA_bulk_SigXX_part, allA_int_SigXX_part)
        allA_dense_SigXX_part_num = np.append(allA_bulk_SigXX_part_num, allA_int_SigXX_part_num)
        allA_dense_SigYX_part = np.append(allA_bulk_SigYX_part, allA_int_SigYX_part)
        allA_dense_SigYX_part_num = np.append(allA_bulk_SigYX_part_num, allA_int_SigYX_part_num)
        allA_dense_SigXY_part = np.append(allA_bulk_SigXY_part, allA_int_SigXY_part)
        allA_dense_SigXY_part_num = np.append(allA_bulk_SigXY_part_num, allA_int_SigXY_part_num)
        allA_dense_SigYY_part = np.append(allA_bulk_SigYY_part, allA_int_SigYY_part)
        allA_dense_SigYY_part_num = np.append(allA_bulk_SigYY_part_num, allA_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and A neighbors
        Aall_dense_SigXX_part = np.append(Aall_bulk_SigXX_part, Aall_int_SigXX_part)
        Aall_dense_SigXX_part_num = np.append(Aall_bulk_SigXX_part_num, Aall_int_SigXX_part_num)
        Aall_dense_SigYX_part = np.append(Aall_bulk_SigYX_part, Aall_int_SigYX_part)
        Aall_dense_SigYX_part_num = np.append(Aall_bulk_SigYX_part_num, Aall_int_SigYX_part_num)
        Aall_dense_SigXY_part = np.append(Aall_bulk_SigXY_part, Aall_int_SigXY_part)
        Aall_dense_SigXY_part_num = np.append(Aall_bulk_SigXY_part_num, Aall_int_SigXY_part_num)
        Aall_dense_SigYY_part = np.append(Aall_bulk_SigYY_part, Aall_int_SigYY_part)
        Aall_dense_SigYY_part_num = np.append(Aall_bulk_SigYY_part_num, Aall_int_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and all neighbors
        allall_dense_SigXX_part = np.append(allall_bulk_SigXX_part, allall_int_SigXX_part)
        allall_dense_SigXX_part_num = np.append(allall_bulk_SigXX_part_num, allall_int_SigXX_part_num)
        allall_dense_SigYX_part = np.append(allall_bulk_SigYX_part, allall_int_SigYX_part)
        allall_dense_SigYX_part_num = np.append(allall_bulk_SigYX_part_num, allall_int_SigYX_part_num)
        allall_dense_SigXY_part = np.append(allall_bulk_SigXY_part, allall_int_SigXY_part)
        allall_dense_SigXY_part_num = np.append(allall_bulk_SigXY_part_num, allall_int_SigXY_part_num)
        allall_dense_SigYY_part = np.append(allall_bulk_SigYY_part, allall_int_SigYY_part)
        allall_dense_SigYY_part_num = np.append(allall_bulk_SigYY_part_num, allall_int_SigYY_part_num)


        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and A neighbors
        SigXX_AA_dense_part = np.append(SigXX_AA_bulk_part, SigXX_AA_int_part)
        SigXX_AA_dense_part_num = np.append(SigXX_AA_bulk_part_num, SigXX_AA_int_part_num)
        SigYX_AA_dense_part = np.append(SigYX_AA_bulk_part, SigYX_AA_int_part)
        SigYX_AA_dense_part_num = np.append(SigYX_AA_bulk_part_num, SigYX_AA_int_part_num)
        SigXY_AA_dense_part = np.append(SigXY_AA_bulk_part, SigXY_AA_int_part)
        SigXY_AA_dense_part_num = np.append(SigXY_AA_bulk_part_num, SigXY_AA_int_part_num)
        SigYY_AA_dense_part = np.append(SigYY_AA_bulk_part, SigYY_AA_int_part)
        SigYY_AA_dense_part_num = np.append(SigYY_AA_bulk_part_num, SigYY_AA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and A neighbors
        SigXX_AB_dense_part = np.append(SigXX_AB_bulk_part, SigXX_AB_int_part)
        SigXX_AB_dense_part_num = np.append(SigXX_AB_bulk_part_num, SigXX_AB_int_part_num)
        SigYX_AB_dense_part = np.append(SigYX_AB_bulk_part, SigYX_AB_int_part)
        SigYX_AB_dense_part_num = np.append(SigYX_AB_bulk_part_num, SigYX_AB_int_part_num)
        SigXY_AB_dense_part = np.append(SigXY_AB_bulk_part, SigXY_AB_int_part)
        SigXY_AB_dense_part_num = np.append(SigXY_AB_bulk_part_num, SigXY_AB_int_part_num)
        SigYY_AB_dense_part = np.append(SigYY_AB_bulk_part, SigYY_AB_int_part)
        SigYY_AB_dense_part_num = np.append(SigYY_AB_bulk_part_num, SigYY_AB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and B neighbors
        SigXX_BA_dense_part = np.append(SigXX_BA_bulk_part, SigXX_BA_int_part)
        SigXX_BA_dense_part_num = np.append(SigXX_BA_bulk_part_num, SigXX_BA_int_part_num)
        SigYX_BA_dense_part = np.append(SigYX_BA_bulk_part, SigYX_BA_int_part)
        SigYX_BA_dense_part_num = np.append(SigYX_BA_bulk_part_num, SigYX_BA_int_part_num)
        SigXY_BA_dense_part = np.append(SigXY_BA_bulk_part, SigXY_BA_int_part)
        SigXY_BA_dense_part_num = np.append(SigXY_BA_bulk_part_num, SigXY_BA_int_part_num)
        SigYY_BA_dense_part = np.append(SigYY_BA_bulk_part, SigYY_BA_int_part)
        SigYY_BA_dense_part_num = np.append(SigYY_BA_bulk_part_num, SigYY_BA_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and B neighbors
        SigXX_BB_dense_part = np.append(SigXX_BB_bulk_part, SigXX_BB_int_part)
        SigXX_BB_dense_part_num = np.append(SigXX_BB_bulk_part_num, SigXX_BB_int_part_num)
        SigYX_BB_dense_part = np.append(SigYX_BB_bulk_part, SigYX_BB_int_part)
        SigYX_BB_dense_part_num = np.append(SigYX_BB_bulk_part_num, SigYX_BB_int_part_num)
        SigXY_BB_dense_part = np.append(SigXY_BB_bulk_part, SigXY_BB_int_part)
        SigXY_BB_dense_part_num = np.append(SigXY_BB_bulk_part_num, SigXY_BB_int_part_num)
        SigYY_BB_dense_part = np.append(SigYY_BB_bulk_part, SigYY_BB_int_part)
        SigYY_BB_dense_part_num = np.append(SigYY_BB_bulk_part_num, SigYY_BB_int_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and all neighbors
        allB_system_SigXX_part = np.append(allB_dense_SigXX_part, allB_gas_SigXX_part)
        allB_system_SigXX_part_num = np.append(allB_dense_SigXX_part_num, allB_gas_SigXX_part_num)
        allB_system_SigYX_part = np.append(allB_dense_SigYX_part, allB_gas_SigYX_part)
        allB_system_SigYX_part_num = np.append(allB_dense_SigYX_part_num, allB_gas_SigYX_part_num)
        allB_system_SigXY_part = np.append(allB_dense_SigXY_part, allB_gas_SigXY_part)
        allB_system_SigXY_part_num = np.append(allB_dense_SigXY_part_num, allB_gas_SigXY_part_num)
        allB_system_SigYY_part = np.append(allB_dense_SigYY_part, allB_gas_SigYY_part)
        allB_system_SigYY_part_num = np.append(allB_dense_SigYY_part_num, allB_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and B neighbors
        Ball_system_SigXX_part = np.append(Ball_dense_SigXX_part, Ball_gas_SigXX_part)
        Ball_system_SigXX_part_num = np.append(Ball_dense_SigXX_part_num, Ball_gas_SigXX_part_num)
        Ball_system_SigYX_part = np.append(Ball_dense_SigYX_part, Ball_gas_SigYX_part)
        Ball_system_SigYX_part_num = np.append(Ball_dense_SigYX_part_num, Ball_gas_SigYX_part_num)
        Ball_system_SigXY_part = np.append(Ball_dense_SigXY_part, Ball_gas_SigXY_part)
        Ball_system_SigXY_part_num = np.append(Ball_dense_SigXY_part_num, Ball_gas_SigXY_part_num)
        Ball_system_SigYY_part = np.append(Ball_dense_SigYY_part, Ball_gas_SigYY_part)
        Ball_system_SigYY_part_num = np.append(Ball_dense_SigYY_part_num, Ball_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and all neighbors
        allA_system_SigXX_part = np.append(allA_dense_SigXX_part, allA_gas_SigXX_part)
        allA_system_SigXX_part_num = np.append(allA_dense_SigXX_part_num, allA_gas_SigXX_part_num)
        allA_system_SigYX_part = np.append(allA_dense_SigYX_part, allA_gas_SigYX_part)
        allA_system_SigYX_part_num = np.append(allA_dense_SigYX_part_num, allA_gas_SigYX_part_num)
        allA_system_SigXY_part = np.append(allA_dense_SigXY_part, allA_gas_SigXY_part)
        allA_system_SigXY_part_num = np.append(allA_dense_SigXY_part_num, allA_gas_SigXY_part_num)
        allA_system_SigYY_part = np.append(allA_dense_SigYY_part, allA_gas_SigYY_part)
        allA_system_SigYY_part_num = np.append(allA_dense_SigYY_part_num, allA_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and A neighbors
        Aall_system_SigXX_part = np.append(Aall_dense_SigXX_part, Aall_gas_SigXX_part)
        Aall_system_SigXX_part_num = np.append(Aall_dense_SigXX_part_num, Aall_gas_SigXX_part_num)
        Aall_system_SigYX_part = np.append(Aall_dense_SigYX_part, Aall_gas_SigYX_part)
        Aall_system_SigYX_part_num = np.append(Aall_dense_SigYX_part_num, Aall_gas_SigYX_part_num)
        Aall_system_SigXY_part = np.append(Aall_dense_SigXY_part, Aall_gas_SigXY_part)
        Aall_system_SigXY_part_num = np.append(Aall_dense_SigXY_part_num, Aall_gas_SigXY_part_num)
        Aall_system_SigYY_part = np.append(Aall_dense_SigYY_part, Aall_gas_SigYY_part)
        Aall_system_SigYY_part_num = np.append(Aall_dense_SigYY_part_num, Aall_gas_SigYY_part_num)

        # Calculate total stress and number of neighbor pairs summed over for all dense reference particles and all neighbors
        allall_system_SigXX_part = np.append(allall_dense_SigXX_part, allall_gas_SigXX_part)
        allall_system_SigXX_part_num = np.append(allall_dense_SigXX_part_num, allall_gas_SigXX_part_num)
        allall_system_SigYX_part = np.append(allall_dense_SigYX_part, allall_gas_SigYX_part)
        allall_system_SigYX_part_num = np.append(allall_dense_SigYX_part_num, allall_gas_SigYX_part_num)
        allall_system_SigXY_part = np.append(allall_dense_SigXY_part, allall_gas_SigXY_part)
        allall_system_SigXY_part_num = np.append(allall_dense_SigXY_part_num, allall_gas_SigXY_part_num)
        allall_system_SigYY_part = np.append(allall_dense_SigYY_part, allall_gas_SigYY_part)
        allall_system_SigYY_part_num = np.append(allall_dense_SigYY_part_num, allall_gas_SigYY_part_num)


        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and A neighbors
        SigXX_AA_system_part = np.append(SigXX_AA_dense_part, SigXX_AA_gas_part)
        SigXX_AA_system_part_num = np.append(SigXX_AA_dense_part_num, SigXX_AA_gas_part_num)
        SigYX_AA_system_part = np.append(SigYX_AA_dense_part, SigYX_AA_gas_part)
        SigYX_AA_system_part_num = np.append(SigYX_AA_dense_part_num, SigYX_AA_gas_part_num)
        SigXY_AA_system_part = np.append(SigXY_AA_dense_part, SigXY_AA_gas_part)
        SigXY_AA_system_part_num = np.append(SigXY_AA_dense_part_num, SigXY_AA_gas_part_num)
        SigYY_AA_system_part = np.append(SigYY_AA_dense_part, SigYY_AA_gas_part)
        SigYY_AA_system_part_num = np.append(SigYY_AA_dense_part_num, SigYY_AA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and A neighbors
        SigXX_AB_system_part = np.append(SigXX_AB_dense_part, SigXX_AB_gas_part)
        SigXX_AB_system_part_num = np.append(SigXX_AB_dense_part_num, SigXX_AB_gas_part_num)
        SigYX_AB_system_part = np.append(SigYX_AB_dense_part, SigYX_AB_gas_part)
        SigYX_AB_system_part_num = np.append(SigYX_AB_dense_part_num, SigYX_AB_gas_part_num)
        SigXY_AB_system_part = np.append(SigXY_AB_dense_part, SigXY_AB_gas_part)
        SigXY_AB_system_part_num = np.append(SigXY_AB_dense_part_num, SigXY_AB_gas_part_num)
        SigYY_AB_system_part = np.append(SigYY_AB_dense_part, SigYY_AB_gas_part)
        SigYY_AB_system_part_num = np.append(SigYY_AB_dense_part_num, SigYY_AB_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for A dense reference particles and B neighbors
        SigXX_BA_system_part = np.append(SigXX_BA_dense_part, SigXX_BA_gas_part)
        SigXX_BA_system_part_num = np.append(SigXX_BA_dense_part_num, SigXX_BA_gas_part_num)
        SigYX_BA_system_part = np.append(SigYX_BA_dense_part, SigYX_BA_gas_part)
        SigYX_BA_system_part_num = np.append(SigYX_BA_dense_part_num, SigYX_BA_gas_part_num)
        SigXY_BA_system_part = np.append(SigXY_BA_dense_part, SigXY_BA_gas_part)
        SigXY_BA_system_part_num = np.append(SigXY_BA_dense_part_num, SigXY_BA_gas_part_num)
        SigYY_BA_system_part = np.append(SigYY_BA_dense_part, SigYY_BA_gas_part)
        SigYY_BA_system_part_num = np.append(SigYY_BA_dense_part_num, SigYY_BA_gas_part_num)

        # Calculate total stress and number of neighbor pairs summed over for B dense reference particles and B neighbors
        SigXX_BB_system_part = np.append(SigXX_BB_dense_part, SigXX_BB_gas_part)
        SigXX_BB_system_part_num = np.append(SigXX_BB_dense_part_num, SigXX_BB_gas_part_num)
        SigYX_BB_system_part = np.append(SigYX_BB_dense_part, SigYX_BB_gas_part)
        SigYX_BB_system_part_num = np.append(SigYX_BB_dense_part_num, SigYX_BB_gas_part_num)
        SigXY_BB_system_part = np.append(SigXY_BB_dense_part, SigXY_BB_gas_part)
        SigXY_BB_system_part_num = np.append(SigXY_BB_dense_part_num, SigXY_BB_gas_part_num)
        SigYY_BB_system_part = np.append(SigYY_BB_dense_part, SigYY_BB_gas_part)
        SigYY_BB_system_part_num = np.append(SigYY_BB_dense_part_num, SigYY_BB_gas_part_num)

        ###Interparticle pressure

        # Calculate total interparticle pressure experienced by all particles in each phase
        allall_bulk_int_press = np.sum(allall_bulk_SigXX_part + allall_bulk_SigYY_part)/(4*bulk_area)
        allall_gas_int_press = np.sum(allall_gas_SigXX_part + allall_gas_SigYY_part)/(4*gas_area)
        allall_int_int_press = np.sum(allall_int_SigXX_part + allall_int_SigYY_part)/(4*int_area)
        allall_dense_int_press = np.sum(allall_dense_SigXX_part + allall_dense_SigYY_part)/(4*dense_area)
        allall_system_int_press = np.sum(allall_system_SigXX_part + allall_system_SigYY_part)/(4*system_area)
        allall_int_press = np.append(allall_bulk_int_press, allall_int_int_press)
        allall_int_press = np.append(allall_int_press, allall_gas_int_press)

        # Calculate total interparticle pressure experienced by each particles in each phase
        allall_bulk_int_press_indiv = (allall_bulk_SigXX_part + allall_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_gas_int_press_indiv = (allall_gas_SigXX_part + allall_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_int_int_press_indiv = (allall_int_SigXX_part + allall_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_dense_int_press_indiv = (allall_dense_SigXX_part + allall_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_system_int_press_indiv = (allall_system_SigXX_part + allall_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allall_int_press_indiv = np.append(allall_bulk_int_press, allall_int_int_press)
        allall_int_press_indiv = np.append(allall_int_press, allall_gas_int_press)

        # Calculate total interparticle pressure experienced by all particles in each phase from all A particles
        allA_bulk_int_press = np.sum(allA_bulk_SigXX_part + allA_bulk_SigYY_part)/(4*bulk_area)
        allA_gas_int_press = np.sum(allA_gas_SigXX_part + allA_gas_SigYY_part)/(4*gas_area)
        allA_int_int_press = np.sum(allA_int_SigXX_part + allA_int_SigYY_part)/(4*int_area)
        allA_dense_int_press = np.sum(allA_dense_SigXX_part + allA_dense_SigYY_part)/(4*dense_area)
        allA_system_int_press = np.sum(allA_system_SigXX_part + allA_system_SigYY_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        allA_bulk_int_press_indiv = (allA_bulk_SigXX_part + allA_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_gas_int_press_indiv = (allA_gas_SigXX_part + allA_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_int_int_press_indiv = (allA_int_SigXX_part + allA_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_dense_int_press_indiv = (allA_dense_SigXX_part + allA_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allA_system_int_press_indiv = (allA_system_SigXX_part + allA_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all A particles in each phase
        Aall_bulk_int_press = np.sum(Aall_bulk_SigXX_part + Aall_bulk_SigYY_part)/(4*bulk_area)
        Aall_gas_int_press = np.sum(Aall_gas_SigXX_part + Aall_gas_SigYY_part)/(4*gas_area)
        Aall_int_int_press = np.sum(Aall_int_SigXX_part + Aall_int_SigYY_part)/(4*int_area)
        Aall_dense_int_press = np.sum(Aall_dense_SigXX_part + Aall_dense_SigYY_part)/(4*dense_area)
        Aall_system_int_press = np.sum(Aall_system_SigXX_part + Aall_system_SigYY_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        Aall_bulk_int_press_indiv = (Aall_bulk_SigXX_part + Aall_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_gas_int_press_indiv = (Aall_gas_SigXX_part + Aall_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_int_int_press_indiv = (Aall_int_SigXX_part + Aall_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_dense_int_press_indiv = (Aall_dense_SigXX_part + Aall_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Aall_system_int_press_indiv = (Aall_system_SigXX_part + Aall_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all particles in each phase from all B particles
        allB_bulk_int_press = np.sum(allB_bulk_SigXX_part + allB_bulk_SigYY_part)/(4*bulk_area)
        allB_gas_int_press = np.sum(allB_gas_SigXX_part + allB_gas_SigYY_part)/(4*gas_area)
        allB_int_int_press = np.sum(allB_int_SigXX_part + allB_int_SigYY_part)/(4*int_area)
        allB_dense_int_press = np.sum(allB_dense_SigXX_part + allB_dense_SigYY_part)/(4*dense_area)
        allB_system_int_press = np.sum(allB_system_SigXX_part + allB_system_SigYY_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        allB_bulk_int_press_indiv = (allB_bulk_SigXX_part + allB_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_gas_int_press_indiv = (allB_gas_SigXX_part + allB_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_int_int_press_indiv = (allB_int_SigXX_part + allB_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_dense_int_press_indiv = (allB_dense_SigXX_part + allB_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        allB_system_int_press_indiv = (allB_system_SigXX_part + allB_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase
        Ball_bulk_int_press = np.sum(Ball_bulk_SigXX_part + Ball_bulk_SigYY_part)/(4*bulk_area)
        Ball_gas_int_press = np.sum(Ball_gas_SigXX_part + Ball_gas_SigYY_part)/(4*gas_area)
        Ball_int_int_press = np.sum(Ball_int_SigXX_part + Ball_int_SigYY_part)/(4*int_area)
        Ball_dense_int_press = np.sum(Ball_dense_SigXX_part + Ball_dense_SigYY_part)/(4*dense_area)
        Ball_system_int_press = np.sum(Ball_system_SigXX_part + Ball_system_SigYY_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        Ball_bulk_int_press_indiv = (Ball_bulk_SigXX_part + Ball_bulk_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_gas_int_press_indiv = (Ball_gas_SigXX_part + Ball_gas_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_int_int_press_indiv = (Ball_int_SigXX_part + Ball_int_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_dense_int_press_indiv = (Ball_dense_SigXX_part + Ball_dense_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)
        Ball_system_int_press_indiv = (Ball_system_SigXX_part + Ball_system_SigYY_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all A particles
        AA_bulk_int_press = np.sum(SigXX_AA_bulk_part + SigYY_AA_bulk_part)/(4*bulk_area)
        AA_gas_int_press = np.sum(SigXX_AA_gas_part + SigYY_AA_gas_part)/(4*gas_area)
        AA_int_int_press = np.sum(SigXX_AA_int_part + SigYY_AA_int_part)/(4*int_area)
        AA_dense_int_press = np.sum(SigXX_AA_dense_part + SigYY_AA_dense_part)/(4*dense_area)
        AA_system_int_press = np.sum(SigXX_AA_system_part + SigYY_AA_system_part)/(4*system_area)

                # Calculate total interparticle pressure experienced by each particles in each phase
        AA_bulk_int_press_indiv = (SigXX_AA_bulk_part + SigYY_AA_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_gas_int_press_indiv = (SigXX_AA_gas_part + SigYY_AA_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_int_int_press_indiv = (SigXX_AA_int_part + SigYY_AA_int_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_dense_int_press_indiv = (SigXX_AA_dense_part + SigYY_AA_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        AA_system_int_press_indiv = (SigXX_AA_system_part + SigYY_AA_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all A particles in each phase from all B particles
        AB_bulk_int_press = np.sum(SigXX_AB_bulk_part + SigYY_AB_bulk_part)/(4*bulk_area)
        AB_gas_int_press = np.sum(SigXX_AB_gas_part + SigYY_AB_gas_part)/(4*gas_area)
        AB_int_int_press = np.sum(SigXX_AB_int_part + SigYY_AB_int_part)/(4*int_area)
        AB_dense_int_press = np.sum(SigXX_AB_dense_part + SigYY_AB_dense_part)/(4*dense_area)
        AB_system_int_press = np.sum(SigXX_AB_system_part + SigYY_AB_system_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        AB_bulk_int_press_indiv = (SigXX_AB_bulk_part + SigYY_AB_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_gas_int_press_indiv = (SigXX_AB_gas_part + SigYY_AB_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_int_int_press_indiv = (SigXX_AB_int_part + SigYY_AB_int_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_dense_int_press_indiv = (SigXX_AB_dense_part + SigYY_AB_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        AB_system_int_press_indiv = (SigXX_AB_system_part + SigYY_AB_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all A particles
        BA_bulk_int_press = np.sum(SigXX_BA_bulk_part + SigYY_BA_bulk_part)/(4*bulk_area)
        BA_gas_int_press = np.sum(SigXX_BA_gas_part + SigYY_BA_gas_part)/(4*gas_area)
        BA_int_int_press = np.sum(SigXX_BA_int_part + SigYY_BA_int_part)/(4*int_area)
        BA_dense_int_press = np.sum(SigXX_BA_dense_part + SigYY_BA_dense_part)/(4*dense_area)
        BA_system_int_press = np.sum(SigXX_BA_system_part + SigYY_BA_system_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        BA_bulk_int_press_indiv = (SigXX_BA_bulk_part + SigYY_BA_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_gas_int_press_indiv = (SigXX_BA_gas_part + SigYY_BA_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_int_int_press_indiv = (SigXX_BA_int_part + SigYY_BA_int_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_dense_int_press_indiv = (SigXX_BA_dense_part + SigYY_BA_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        BA_system_int_press_indiv = (SigXX_BA_system_part + SigYY_BA_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total interparticle pressure experienced by all B particles in each phase from all B particles
        BB_bulk_int_press = np.sum(SigXX_BB_bulk_part + SigYY_BB_bulk_part)/(4*bulk_area)
        BB_gas_int_press = np.sum(SigXX_BB_gas_part + SigYY_BB_gas_part)/(4*gas_area)
        BB_int_int_press = np.sum(SigXX_BB_int_part + SigYY_BB_int_part)/(4*int_area)
        BB_dense_int_press = np.sum(SigXX_BB_dense_part + SigYY_BB_dense_part)/(4*dense_area)
        BB_system_int_press = np.sum(SigXX_BB_system_part + SigYY_BB_system_part)/(4*system_area)

        # Calculate total interparticle pressure experienced by each particles in each phase
        BB_bulk_int_press_indiv = (SigXX_BB_bulk_part + SigYY_BB_bulk_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_gas_int_press_indiv = (SigXX_BB_gas_part + SigYY_BB_gas_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_int_int_press_indiv = (SigXX_BB_int_part + SigYY_BB_int_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_dense_int_press_indiv = (SigXX_BB_dense_part + SigYY_BB_dense_part)/(2*np.pi * (self.r_cut/2)**2)
        BB_system_int_press_indiv = (SigXX_BB_system_part + SigYY_BB_system_part)/(2*np.pi * (self.r_cut/2)**2)

        # Calculate total shear stress experienced by all particles in each phase from all particles
        allall_bulk_shear_stress = np.sum(allall_bulk_SigXY_part)/(bulk_area)
        allall_gas_shear_stress = np.sum(allall_gas_SigXY_part)/(gas_area)
        allall_int_shear_stress = np.sum(allall_int_SigXY_part)/(int_area)
        allall_dense_shear_stress = np.sum(allall_dense_SigXY_part)/(dense_area)
        allall_system_shear_stress = np.sum(allall_system_SigXY_part)/(system_area)
        allall_shear_stress = np.append(allall_bulk_shear_stress, allall_int_shear_stress)
        allall_shear_stress = np.append(allall_shear_stress, allall_gas_shear_stress)

        # Calculate total shear stress experienced by all particles in each phase from A particles
        allA_bulk_shear_stress = np.sum(allA_bulk_SigXY_part)/(bulk_area)
        allA_gas_shear_stress = np.sum(allA_gas_SigXY_part)/(gas_area)
        allA_int_shear_stress = np.sum(allA_int_SigXY_part)/(int_area)
        allA_dense_shear_stress = np.sum(allA_dense_SigXY_part)/(dense_area)
        allA_system_shear_stress = np.sum(allA_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all particles
        Aall_bulk_shear_stress = np.sum(Aall_bulk_SigXY_part)/(bulk_area)
        Aall_gas_shear_stress = np.sum(Aall_gas_SigXY_part)/(gas_area)
        Aall_int_shear_stress = np.sum(Aall_int_SigXY_part)/(int_area)
        Aall_dense_shear_stress = np.sum(Aall_dense_SigXY_part)/(dense_area)
        Aall_system_shear_stress = np.sum(Aall_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all particles in each phase from B particles
        allB_bulk_shear_stress = np.sum(allB_bulk_SigXY_part)/(bulk_area)
        allB_gas_shear_stress = np.sum(allB_gas_SigXY_part)/(gas_area)
        allB_int_shear_stress = np.sum(allB_int_SigXY_part)/(int_area)
        allB_dense_shear_stress = np.sum(allB_dense_SigXY_part)/(dense_area)
        allB_system_shear_stress = np.sum(allB_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all particles
        Ball_bulk_shear_stress = np.sum(Ball_bulk_SigXY_part)/(bulk_area)
        Ball_gas_shear_stress = np.sum(Ball_gas_SigXY_part)/(gas_area)
        Ball_int_shear_stress = np.sum(Ball_int_SigXY_part)/(int_area)
        Ball_dense_shear_stress = np.sum(Ball_dense_SigXY_part)/(dense_area)
        Ball_system_shear_stress = np.sum(Ball_system_SigXY_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all A particles
        AA_bulk_shear_stress = np.sum(SigXY_AA_bulk_part)/(bulk_area)
        AA_gas_shear_stress = np.sum(SigXY_AA_gas_part)/(gas_area)
        AA_int_shear_stress = np.sum(SigXY_AA_int_part)/(int_area)
        AA_dense_shear_stress = np.sum(SigXY_AA_dense_part)/(dense_area)
        AA_system_shear_stress = np.sum(SigXY_AA_system_part)/(system_area)

        # Calculate total shear stress experienced by all A particles in each phase from all B particles
        AB_bulk_shear_stress = np.sum(SigXY_AB_bulk_part)/(bulk_area)
        AB_gas_shear_stress = np.sum(SigXY_AB_gas_part)/(gas_area)
        AB_int_shear_stress = np.sum(SigXY_AB_int_part)/(int_area)
        AB_dense_shear_stress = np.sum(SigXY_AB_dense_part)/(dense_area)
        AB_system_shear_stress = np.sum(SigXY_AB_system_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all A particles
        BA_bulk_shear_stress = np.sum(SigXY_BA_bulk_part)/(bulk_area)
        BA_gas_shear_stress = np.sum(SigXY_BA_gas_part)/(gas_area)
        BA_int_shear_stress = np.sum(SigXY_BA_int_part)/(int_area)
        BA_dense_shear_stress = np.sum(SigXY_BA_dense_part)/(dense_area)
        BA_system_shear_stress = np.sum(SigXY_BA_system_part)/(system_area)

        # Calculate total shear stress experienced by all B particles in each phase from all B particles
        BB_bulk_shear_stress = np.sum(SigXY_BB_bulk_part)/(bulk_area)
        BB_gas_shear_stress = np.sum(SigXY_BB_gas_part)/(gas_area)
        BB_int_shear_stress = np.sum(SigXY_BB_int_part)/(int_area)
        BB_dense_shear_stress = np.sum(SigXY_BB_dense_part)/(dense_area)
        BB_system_shear_stress = np.sum(SigXY_BB_system_part)/(system_area)


        # Make position arrays for plotting total stress on each particle for various activity pairings and phases
        allall_bulk_pos_x = np.append(pos_A_bulk[:,0], pos_B_bulk[:,0])
        allall_bulk_pos_y = np.append(pos_A_bulk[:,1], pos_B_bulk[:,1])
        allall_gas_pos_x = np.append(pos_A_gas[:,0], pos_B_gas[:,0])
        allall_gas_pos_y = np.append(pos_A_gas[:,1], pos_B_gas[:,1])
        allall_int_pos_x = np.append(pos_A_int[:,0], pos_B_int[:,0])
        allall_int_pos_y = np.append(pos_A_int[:,1], pos_B_int[:,1])
        Aall_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Aall_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])
        Ball_dense_pos_x = np.append(pos_bulk[:,0], pos_int[:,0])
        Ball_dense_pos_y = np.append(pos_bulk[:,1], pos_int[:,1])
        allA_dense_pos_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        allA_dense_pos_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])
        allB_dense_pos_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        allB_dense_pos_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])
        allall_dense_pos_x = np.append(allall_bulk_pos_x, allall_int_pos_x)
        allall_dense_pos_y = np.append(allall_bulk_pos_y, allall_int_pos_y)
        allall_pos_x = np.append(allall_dense_pos_x, allall_gas_pos_x)
        allall_pos_y = np.append(allall_dense_pos_y, allall_gas_pos_y)

        pos_dense_x = np.append(pos_bulk[:,0], pos_int[:,0])
        pos_dense_y = np.append(pos_bulk[:,1], pos_int[:,1])
        pos_A_dense_x = np.append(pos_A_bulk[:,0], pos_A_int[:,0])
        pos_A_dense_y = np.append(pos_A_bulk[:,1], pos_A_int[:,1])
        pos_B_dense_x = np.append(pos_B_bulk[:,0], pos_B_int[:,0])
        pos_B_dense_y = np.append(pos_B_bulk[:,1], pos_B_int[:,1])

        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        stress_stat_dict = {'bulk': {'all-all': {'XX': np.sum(allall_bulk_SigXX_part), 'XY': np.sum(allall_bulk_SigXY_part), 'YX': np.sum(allall_bulk_SigYX_part), 'YY': np.sum(allall_bulk_SigYY_part)}, 'all-A': {'XX': np.sum(allA_bulk_SigXX_part), 'XY': np.sum(allA_bulk_SigXY_part), 'YX': np.sum(allA_bulk_SigYX_part), 'YY': np.sum(allA_bulk_SigYY_part)}, 'all-B': {'XX': np.sum(allB_bulk_SigXX_part), 'XY': np.sum(allB_bulk_SigXY_part), 'YX': np.sum(allB_bulk_SigYX_part), 'YY': np.sum(allB_bulk_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_bulk_part), 'XY': np.sum(SigXY_AA_bulk_part), 'YX': np.sum(SigYX_AA_bulk_part), 'YY': np.sum(SigYY_AA_bulk_part)}, 'A-B': {'XX': np.sum(SigXX_AB_bulk_part), 'XY': np.sum(SigXY_AB_bulk_part), 'YX': np.sum(SigYX_AB_bulk_part), 'YY': np.sum(SigYY_AB_bulk_part)}, 'B-B': {'XX': np.sum(SigXX_BB_bulk_part), 'XY': np.sum(SigXY_BB_bulk_part), 'YX': np.sum(SigYX_BB_bulk_part), 'YY': np.sum(SigYY_BB_bulk_part)}}, 'gas': {'all-all': {'XX': np.sum(allall_gas_SigXX_part), 'XY': np.sum(allall_gas_SigXY_part), 'YX': np.sum(allall_gas_SigYX_part), 'YY': np.sum(allall_gas_SigYY_part)}, 'all-A': {'XX': np.sum(allA_gas_SigXX_part), 'XY': np.sum(allA_gas_SigXY_part), 'YX': np.sum(allA_gas_SigYX_part), 'YY': np.sum(allA_gas_SigYY_part)}, 'all-B': {'XX': np.sum(allB_gas_SigXX_part), 'XY': np.sum(allB_gas_SigXY_part), 'YX': np.sum(allB_gas_SigYX_part), 'YY': np.sum(allB_gas_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_gas_part), 'XY': np.sum(SigXY_AA_gas_part), 'YX': np.sum(SigYX_AA_gas_part), 'YY': np.sum(SigYY_AA_gas_part)}, 'A-B': {'XX': np.sum(SigXX_AB_gas_part), 'XY': np.sum(SigXY_AB_gas_part), 'YX': np.sum(SigYX_AB_gas_part), 'YY': np.sum(SigYY_AB_gas_part)}, 'B-B': {'XX': np.sum(SigXX_BB_gas_part), 'XY': np.sum(SigXY_BB_gas_part), 'YX': np.sum(SigYX_BB_gas_part), 'YY': np.sum(SigYY_BB_gas_part)}}, 'dense': {'all-all': {'XX': np.sum(allall_dense_SigXX_part), 'XY': np.sum(allall_dense_SigXY_part), 'YX': np.sum(allall_dense_SigYX_part), 'YY': np.sum(allall_dense_SigYY_part)}, 'all-A': {'XX': np.sum(allA_dense_SigXX_part), 'XY': np.sum(allA_dense_SigXY_part), 'YX': np.sum(allA_dense_SigYX_part), 'YY': np.sum(allA_dense_SigYY_part)}, 'all-B': {'XX': np.sum(allB_dense_SigXX_part), 'XY': np.sum(allB_dense_SigXY_part), 'YX': np.sum(allB_dense_SigYX_part), 'YY': np.sum(allB_dense_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_dense_part), 'XY': np.sum(SigXY_AA_dense_part), 'YX': np.sum(SigYX_AA_dense_part), 'YY': np.sum(SigYY_AA_dense_part)}, 'A-B': {'XX': np.sum(SigXX_AB_dense_part), 'XY': np.sum(SigXY_AB_dense_part), 'YX': np.sum(SigYX_AB_dense_part), 'YY': np.sum(SigYY_AB_dense_part)}, 'B-B': {'XX': np.sum(SigXX_BB_dense_part), 'XY': np.sum(SigXY_BB_dense_part), 'YX': np.sum(SigYX_BB_dense_part), 'YY': np.sum(SigYY_BB_dense_part)}}, 'int': {'all-all': {'XX': np.sum(allall_int_SigXX_part), 'XY': np.sum(allall_int_SigXY_part), 'YX': np.sum(allall_int_SigYX_part), 'YY': np.sum(allall_int_SigYY_part)}, 'all-A': {'XX': np.sum(allA_int_SigXX_part), 'XY': np.sum(allA_int_SigXY_part), 'YX': np.sum(allA_int_SigYX_part), 'YY': np.sum(allA_int_SigYY_part)}, 'all-B': {'XX': np.sum(allB_int_SigXX_part), 'XY': np.sum(allB_int_SigXY_part), 'YX': np.sum(allB_int_SigYX_part), 'YY': np.sum(allB_int_SigYY_part)}, 'A-A': {'XX': np.sum(SigXX_AA_int_part), 'XY': np.sum(SigXY_AA_int_part), 'YX': np.sum(SigYX_AA_int_part), 'YY': np.sum(SigYY_AA_int_part)}, 'A-B': {'XX': np.sum(SigXX_AB_int_part), 'XY': np.sum(SigXY_AB_int_part), 'YX': np.sum(SigYX_AB_int_part), 'YY': np.sum(SigYY_AB_int_part)}, 'B-B': {'XX': np.sum(SigXX_BB_int_part), 'XY': np.sum(SigXY_BB_int_part), 'YX': np.sum(SigYX_BB_int_part), 'YY': np.sum(SigYY_BB_int_part)}}}

        # Create output dictionary for statistical averages of total pressure and shear stress on each particle per phase/activity pairing
        press_stat_dict = {'all-all': {'bulk': {'press': allall_bulk_int_press, 'shear': allall_bulk_shear_stress}, 'int': {'press': allall_int_int_press, 'shear': allall_int_shear_stress}, 'gas': {'press': allall_gas_int_press, 'shear': allall_gas_shear_stress}, 'dense': {'press': allall_dense_int_press, 'shear': allall_dense_shear_stress}, 'system': {'press': allall_system_int_press, 'shear': allall_system_shear_stress}}, 'all-A': {'bulk': {'press': allA_bulk_int_press, 'shear': allA_bulk_shear_stress}, 'int': {'press': allA_int_int_press, 'shear': allA_int_shear_stress}, 'gas': {'press': allA_gas_int_press, 'shear': allA_gas_shear_stress}, 'dense': {'press': allA_dense_int_press, 'shear': allA_dense_shear_stress}, 'system': {'press': allA_system_int_press, 'shear': allA_system_shear_stress}}, 'all-B': {'bulk': {'press': allB_bulk_int_press, 'shear': allB_bulk_shear_stress}, 'int': {'press': allB_int_int_press, 'shear': allB_int_shear_stress}, 'gas': {'press': allB_gas_int_press, 'shear': allB_gas_shear_stress}, 'dense': {'press': allB_dense_int_press, 'shear': allB_dense_shear_stress}, 'system': {'press': allB_system_int_press, 'shear': allB_system_shear_stress}}, 'A-A': {'bulk': {'press': AA_bulk_int_press, 'shear': AA_bulk_shear_stress}, 'int': {'press': AA_int_int_press, 'shear': AA_int_shear_stress}, 'gas': {'press': AA_gas_int_press, 'shear': AA_gas_shear_stress}, 'dense': {'press': AA_dense_int_press, 'shear': AA_dense_shear_stress}, 'system': {'press': AA_system_int_press, 'shear': AA_system_shear_stress}}, 'A-B': {'bulk': {'press': AB_bulk_int_press, 'shear': AB_bulk_shear_stress}, 'int': {'press': AB_int_int_press, 'shear': AB_int_shear_stress}, 'gas': {'press': AB_gas_int_press, 'shear': AB_gas_shear_stress}, 'dense': {'press': AB_dense_int_press, 'shear': AB_dense_shear_stress}, 'system': {'press': AB_system_int_press, 'shear': AB_system_shear_stress}}, 'B-B': {'bulk': {'press': BB_bulk_int_press, 'shear': BB_bulk_shear_stress}, 'int': {'press': BB_int_int_press, 'shear': BB_int_shear_stress}, 'gas': {'press': BB_gas_int_press, 'shear': BB_gas_shear_stress}, 'dense': {'press': BB_dense_int_press, 'shear': BB_dense_shear_stress}, 'system': {'press': BB_system_int_press, 'shear': BB_system_shear_stress}}}

        # Create output dictionary for statistical averages of total pressure and shear stress on each particle per phase/activity pairing
        press_stat_indiv_dict = {'all-all': {'bulk': {'mean': np.mean(allall_bulk_int_press_indiv), 'std': np.std(allall_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allall_int_int_press_indiv), 'std': np.std(allall_int_int_press_indiv)}, 'gas': {'mean': np.mean(allall_gas_int_press_indiv), 'std': np.std(allall_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allall_dense_int_press_indiv), 'std': np.std(allall_dense_int_press_indiv)}, 'system': {'mean': np.mean(allall_system_int_press_indiv), 'std': np.std(allall_system_int_press_indiv)}}, 'all-A': {'bulk': {'mean': np.mean(allA_bulk_int_press_indiv), 'std': np.std(allA_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allA_int_int_press_indiv), 'std': np.std(allA_int_int_press_indiv)}, 'gas': {'mean': np.mean(allA_gas_int_press_indiv), 'std': np.std(allA_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allA_dense_int_press_indiv), 'std': np.std(allA_dense_int_press_indiv)}, 'system': {'mean': np.mean(allA_system_int_press_indiv), 'std': np.std(allA_system_int_press_indiv)}}, 'all-B': {'bulk': {'mean': np.mean(allB_bulk_int_press_indiv), 'std': np.std(allB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(allB_int_int_press_indiv), 'std': np.std(allB_int_int_press_indiv)}, 'gas': {'mean': np.mean(allB_gas_int_press_indiv), 'std': np.std(allB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(allB_dense_int_press_indiv), 'std': np.std(allB_dense_int_press_indiv)}, 'system': {'mean': np.mean(allB_system_int_press_indiv), 'std': np.std(allB_system_int_press_indiv)}}, 'A-A': {'bulk': {'mean': np.mean(AA_bulk_int_press_indiv), 'std': np.std(AA_bulk_int_press_indiv)}, 'int': {'mean': np.mean(AA_int_int_press_indiv), 'std': np.std(AA_int_int_press_indiv)}, 'gas': {'mean': np.mean(AA_gas_int_press_indiv), 'std': np.std(AA_gas_int_press_indiv)}, 'dense': {'mean': np.mean(AA_dense_int_press_indiv), 'std': np.std(AA_dense_int_press_indiv)}, 'system': {'mean': np.mean(AA_system_int_press_indiv), 'std': np.std(AA_system_int_press_indiv)}}, 'A-B': {'bulk': {'mean': np.mean(AB_bulk_int_press_indiv), 'std': np.std(AB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(AB_int_int_press_indiv), 'std': np.std(AB_int_int_press_indiv)}, 'gas': {'mean': np.mean(AB_gas_int_press_indiv), 'std': np.std(AB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(AB_dense_int_press_indiv), 'std': np.std(AB_dense_int_press_indiv)}, 'system': {'mean': np.mean(AB_system_int_press_indiv), 'std': np.std(AB_system_int_press_indiv)}}, 'B-B': {'bulk': {'mean': np.mean(BB_bulk_int_press_indiv), 'std': np.std(BB_bulk_int_press_indiv)}, 'int': {'mean': np.mean(BB_int_int_press_indiv), 'std': np.std(BB_int_int_press_indiv)}, 'gas': {'mean': np.mean(BB_gas_int_press_indiv), 'std': np.std(BB_gas_int_press_indiv)}, 'dense': {'mean': np.mean(BB_dense_int_press_indiv), 'std': np.std(BB_dense_int_press_indiv)}, 'system': {'mean': np.mean(BB_system_int_press_indiv), 'std': np.std(BB_system_int_press_indiv)}}}

        # Create output dictionary for statistical averages of total stress on each particle per phase/activity pairing
        stress_plot_dict = {'bulk': {'all-all': {'XX': allall_bulk_SigXX_part, 'XY': allall_bulk_SigXY_part, 'YX': allall_bulk_SigYX_part, 'YY': allall_bulk_SigYY_part}, 'all-A': {'XX': allA_bulk_SigXX_part, 'XY': allA_bulk_SigXY_part, 'YX': allA_bulk_SigYX_part, 'YY': allA_bulk_SigYY_part}, 'all-B': {'XX': allB_bulk_SigXX_part, 'XY': allB_bulk_SigXY_part, 'YX': allB_bulk_SigYX_part, 'YY': allB_bulk_SigYY_part}, 'A-A': {'XX': SigXX_AA_bulk_part, 'XY': SigXY_AA_bulk_part, 'YX': SigYX_AA_bulk_part, 'YY': SigYY_AA_bulk_part}, 'A-B': {'XX': SigXX_AB_bulk_part, 'XY': SigXY_AB_bulk_part, 'YX': SigYX_AB_bulk_part, 'YY': SigYY_AB_bulk_part}, 'B-B': {'XX': SigXX_BB_bulk_part, 'XY': SigXY_BB_bulk_part, 'YX': SigYX_BB_bulk_part, 'YY': SigYY_BB_bulk_part}, 'pos': {'all': {'x': pos_bulk[:,0], 'y': pos_bulk[:,1]}, 'A': {'x': pos_A_bulk[:,0], 'y': pos_A_bulk[:,1]}, 'B': {'x': pos_B_bulk[:,0], 'y': pos_B_bulk[:,1]}}, 'typ': typ_bulk}, 'gas': {'all-all': {'XX': allall_gas_SigXX_part, 'XY': allall_gas_SigXY_part, 'YX': allall_gas_SigYX_part, 'YY': allall_gas_SigYY_part}, 'all-A': {'XX': allA_gas_SigXX_part, 'XY': allA_gas_SigXY_part, 'YX': allA_gas_SigYX_part, 'YY': allA_gas_SigYY_part}, 'all-B': {'XX': allB_gas_SigXX_part, 'XY': allB_gas_SigXY_part, 'YX': allB_gas_SigYX_part, 'YY': allB_gas_SigYY_part}, 'A-A': {'XX': SigXX_AA_gas_part, 'XY': SigXY_AA_gas_part, 'YX': SigYX_AA_gas_part, 'YY': SigYY_AA_gas_part}, 'A-B': {'XX': SigXX_AB_gas_part, 'XY': SigXY_AB_gas_part, 'YX': SigYX_AB_gas_part, 'YY': SigYY_AB_gas_part}, 'B-B': {'XX': SigXX_BB_gas_part, 'XY': SigXY_BB_gas_part, 'YX': SigYX_BB_gas_part, 'YY': SigYY_BB_gas_part}, 'pos': {'all': {'x': pos_gas[:,0], 'y': pos_gas[:,1]}, 'A': {'x': pos_A_gas[:,0], 'y': pos_A_gas[:,1]}, 'B': {'x': pos_B_gas[:,0], 'y': pos_B_gas[:,1]}}, 'typ': typ_gas}, 'dense': {'all-all': {'XX': allall_dense_SigXX_part, 'XY': allall_dense_SigXY_part, 'YX': allall_dense_SigYX_part, 'YY': allall_dense_SigYY_part}, 'all-A': {'XX': allA_dense_SigXX_part, 'XY': allA_dense_SigXY_part, 'YX': allA_dense_SigYX_part, 'YY': allA_dense_SigYY_part}, 'all-B': {'XX': allB_dense_SigXX_part, 'XY': allB_dense_SigXY_part, 'YX': allB_dense_SigYX_part, 'YY': allB_dense_SigYY_part}, 'A-A': {'XX': SigXX_AA_dense_part, 'XY': SigXY_AA_dense_part, 'YX': SigYX_AA_dense_part, 'YY': SigYY_AA_dense_part}, 'A-B': {'XX': SigXX_AB_dense_part, 'XY': SigXY_AB_dense_part, 'YX': SigYX_AB_dense_part, 'YY': SigYY_AB_dense_part}, 'B-B': {'XX': SigXX_BB_dense_part, 'XY': SigXY_BB_dense_part, 'YX': SigYX_BB_dense_part, 'YY': SigYY_BB_dense_part}, 'pos': {'all': {'x': pos_dense_x, 'y': pos_dense_y}, 'A': {'x': pos_A_dense_x, 'y': pos_A_dense_y}, 'B': {'x': pos_B_dense_x, 'y': pos_B_dense_y}}, 'typ': typ_dense}}

        # Create output dictionary for plotting of total stress/pressure on each particle per phase/activity pairing and their respective x-y locations
        press_plot_dict = {'all-all': {'press': allall_int_press, 'shear': allall_shear_stress, 'x': allall_pos_x, 'y': allall_pos_y}}

        return stress_stat_dict, press_stat_dict, press_stat_indiv_dict, press_plot_dict, stress_plot_dict
    def clustering_coefficient(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the clustering coefficient of each
        type for each particle and averaged over all particles of each phase. Useful for
        visualizing segregated domains.

        Outputs:
        clust_plot_dict: dictionary containing information on the number of cluster coefficient
         of each bulk and interface reference particle of each type ('all', 'A', or 'B').
        '''
        # Count total number of bins in each phase
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Get array of ids that give which particles of each type belong to each phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Calculate area of bulk
        bulk_area = phase_count_dict['bulk'] * (self.sizeBin_x * self.sizeBin_y)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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
        print(AA_path_length)
        print(np.mean(AA_path_length))
        print(np.min(AA_path_length))
        print(np.max(AA_path_length))

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
        phase_count_dict = self.phase_ident_functs.phase_count(self.phase_dict)

        # Count particles per phase
        phase_part_dict = self.particle_prop_functs.particle_phase_ids(self.phasePart)

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        ang_A=self.ang[typ0ind]
        pos_A_bulk = self.pos[phase_part_dict['bulk']['A']]
        ang_A_bulk = self.ang[phase_part_dict['bulk']['A']]
        pos_A_int = self.pos[phase_part_dict['int']['A']]
        ang_A_int = self.ang[phase_part_dict['int']['A']]
        pos_A_gas = self.pos[phase_part_dict['gas']['A']]
        pos_A_dense = self.pos[phase_part_dict['dense']['A']]
        ang_A_dense = self.ang[phase_part_dict['dense']['A']]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        ang_B=self.ang[typ1ind]
        pos_B_bulk = self.pos[phase_part_dict['bulk']['B']]
        ang_B_bulk = self.ang[phase_part_dict['bulk']['B']]
        pos_B_int = self.pos[phase_part_dict['int']['B']]
        ang_B_int = self.ang[phase_part_dict['int']['B']]
        pos_B_gas = self.pos[phase_part_dict['gas']['B']]
        pos_B_dense = self.pos[phase_part_dict['dense']['B']]
        ang_B_dense = self.ang[phase_part_dict['dense']['B']]

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
        domain_stat_dict = {'A': {'pop': len(pos_A_dense), 'avg_size': np.mean(clust_size_A[lcID_A]), 'std_size': np.std(clust_size_A[lcID_A]), 'num': len(clust_size_A[lcID_A]), 'first_size': first_max_clust_A, 'second_size': second_max_clust_A, 'third_size': third_max_clust_A, 'fourth_size': fourth_max_clust_A, 'fifth_size': fifth_max_clust_A, 'one_num': lcID_A_smallest, 'two_num': lcID_A_second_smallest, 'three_num': lcID_A_third_smallest, 'fourth_num': lcID_A_fourth_smallest}, 'B': {'pop': len(pos_B_dense), 'avg_size': np.mean(clust_size_B[lcID_B]), 'std_size': np.std(clust_size_B[lcID_B]), 'num': len(clust_size_B[lcID_B]), 'first': first_max_clust_B, 'second': second_max_clust_B, 'third': third_max_clust_B, 'fourth': fourth_max_clust_B, 'fifth': fifth_max_clust_B, 'one_num': lcID_B_smallest, 'two_num': lcID_B_second_smallest, 'three_num': lcID_B_third_smallest, 'fourth_num': lcID_B_fourth_smallest}}
        
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

    

