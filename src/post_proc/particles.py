import sys, os
from gsd import hoomd
from freud import box
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage
import numpy as np, matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
from scipy.optimize import curve_fit
from scipy import stats

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory, utility, plotting_utility

# Class of individual particle property measurements
class particle_props:
    """
    Purpose: 
    This class contains a series of basic functions for analyzing properties of systems 
    without the need for differentiating phases, including lattice spacing, order parameters,
    and neighbor information.
    """
    def __init__(self, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, eps, typ, pos, px, py):

        # Initialize theory functions for call back later
        theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box / 2

        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box / 2

        # Instantiated simulation box
        self.f_box = box.Box(Lx=lx_box, Ly=ly_box, is2D=True)

        # Initialize utility functions for call back later
        self.utility_functs = utility.utility(self.lx_box, self.ly_box)

        # Number of particles
        self.partNum = partNum

        # Set minimum cluster size to analyze
        self.min_size = int(self.partNum / 8)

        try:
            # Total number of bins in x-direction
            self.NBins_x = int(NBins_x)

            # Total number of bins in y-direction
            self.NBins_y = int(NBins_y)
        except:
            print('NBins must be either a float or an integer')

        # X-length of bin
        self.sizeBin_x = self.utility_functs.roundUp(self.lx_box / self.NBins_x, 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp(self.ly_box / self.NBins_y, 6)

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Array (partNum) of particle types
        self.typ = typ

        # Array (partNum) of particle positions
        self.pos = pos

        # Array (partNum) of particle orientations
        self.px = px
        self.py = py

        # Initialize theory functions for call back later
        self.theory_functs = theory.theory()

        self.plotting_utility_functs = plotting_utility.plotting_utility(self.lx_box, self.ly_box, self.partNum, self.typ)

    def particle_phase_ids(self, phasePart):
        #DONE!
        '''
        Purpose: Takes the phase that each particle belongs to and returns arrays of
        IDs of all particles of the respective type belonging to each phase

        Inputs:
        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        phase_part_dict: dictionary containing arrays that give the id of each particle
        of each type ('all', 'A', or 'B') belonging to each respective phase
        '''

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to bulk phase
        A_bulk_id = np.where((phasePart==0) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_bulk_id = np.where((phasePart==0) & (self.typ==1))[0]        #Bulk phase structure(s)
        bulk_id = np.where(phasePart==0)[0]        #Bulk phase structure(s)

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to interface
        A_int_id = np.where((phasePart==1) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_int_id = np.where((phasePart==1) & (self.typ==1))[0]        #Bulk phase structure(s)
        int_id = np.where(phasePart==1)[0]         #All interfaces

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to gas phase
        A_gas_id = np.where((phasePart==2) & (self.typ==0))[0]        #Bulk phase structure(s)
        B_gas_id = np.where((phasePart==2) & (self.typ==1))[0]        #Bulk phase structure(s)
        gas_id = np.where(phasePart==2)[0]         #All interfaces

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to dense phase (either bulk or interface)
        A_dense_id = np.where((phasePart!=2) & (self.typ==0))[0]
        B_dense_id = np.where((phasePart!=2) & (self.typ==1))[0]
        dense_id = np.where(phasePart!=2)[0]

        # IDs of each particle of respective type ('all', 'A', or 'B') belonging to either gas or interface
        A_gas_int_id = np.where((phasePart!=0) & (self.typ==0))[0]
        B_gas_int_id = np.where((phasePart!=0) & (self.typ==1))[0]
        gas_int_id = np.where(phasePart!=0)[0]

        # Dictionary containing arrays that give the id of each particle
        # of each type ('all', 'A', or 'B') belonging to each respective phase
        phase_part_dict = {'bulk': {'all': bulk_id, 'A': A_bulk_id, 'B': B_bulk_id}, 'int': {'all': int_id, 'A': A_int_id, 'B': B_int_id}, 'gas': {'all': gas_id, 'A': A_gas_id, 'B': B_gas_id}, 'dense': {'all': dense_id, 'A': A_dense_id, 'B': B_dense_id}, 'gas_int': {'all': gas_int_id, 'A': A_gas_int_id, 'B': B_gas_int_id}}

        return phase_part_dict

    def particle_normal_fa(self):
        #DONE!
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude pointing towards largest cluster's CoM

        Output:
        part_normal_fa: array (partNum) of each particle's active force magnitude
        pointing toward largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the active force component
        # pointing towards the largest cluster's CoM
        part_normal_fa = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in particle's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Normalize x- and y- separation distance to make unit vectors
            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = self.px[h]
            py = self.py[h]

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #Save active force magnitude pointing toward largest cluster's CoM
            if self.typ[h]==0:
                part_normal_fa[h]=r_dot_p*self.peA
            elif self.typ[h]==1:
                part_normal_fa[h]=r_dot_p*self.peB

        return part_normal_fa

    def particle_align(self):
        '''
        Purpose: Takes the orientation and position of each particle
        to calculate the active force alignment towards the largest cluster's CoM

        Output:
        part_align: array (partNum) of each particle's active force alignment
         toward largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the active force alignment
        # towards the largest cluster's CoM
        part_align = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in particle's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Normalize x- and y- separation distance to make unit vectors
            x_unitv = (difx) / difr
            y_unitv = (dify) / difr

            #Calculate x and y orientation of active force
            px = self.px[h]
            py = self.py[h]

            #Calculate alignment towards CoM
            r_dot_p = (-x_unitv * px) + (-y_unitv * py)

            #Save alignment toward largest cluster's CoM
            part_align[h]=r_dot_p

        return part_align

    def particle_sep_dist(self):
        '''
        Purpose: Takes the position of each particle
        to calculate the separation distance from the largest cluster's CoM

        Output:
        part_difr: array (partNum) of each particle's separation distance from
         the largest cluster's CoM
        '''

        # Instantiate empty array (partNum) containing the separation distance
        # from the largest cluster's CoM
        part_difr = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, len(self.pos)):

            #Calculate x- and y- positions of reference particle
            pos_x1 = self.pos[h,0]+self.h_box
            pos_y1 = self.pos[h,1]+self.h_box

            #Calculate difference in bin's position with CoM at h_box
            difx = self.utility_functs.sep_dist(pos_x1, self.h_box)
            dify = self.utility_functs.sep_dist(pos_y1, self.h_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            #Save separation distance from largest cluster's CoM
            part_difr[h] = difr

        return part_difr

    def radial_surface_normal_fa(self, method2_align_dict):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from the nearest interface surface normal

        Input:
        method2_align_dict: dictionary containing the alignment
        of each particle's active force with the nearest interface's surface normal,
        per and averaged per bin
        
        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force with
        the nearest interface surface normal as a function of separation distance from largest custer's CoM
        '''

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty array (partNum) containing the average active force magnitude
        # toward the nearest interface surface normal
        fa_norm = np.array([])
        faA_norm = np.array([])
        faB_norm = np.array([])

        # Instantiate empty array (partNum) containing the distance from the nearest interface surface
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])

        align_norm = np.array([])
        alignA_norm = np.array([])
        alignB_norm = np.array([])

        # Loop over all particles
        for h in range(0, len(self.pos)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = self.pos[h,0] - 0
            dify = self.pos[h,1] - 0

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Save particle properties as function of distance from cluster's CoM
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, part_align[h]*self.peA)  # Aligned active force magnitude of all particles
                faA_norm=np.append(faA_norm, part_align[h]*self.peA)  # Aligned active force magnitude of particle type A
                rA_dist_norm = np.append(rA_dist_norm, difr)  #Distance from custer CoM of particle type A
                alignA_norm=np.append(alignA_norm, part_align[h])  # Alignment of particle type A
            else:
                fa_norm=np.append(fa_norm, part_align[h]*self.peB)  # Aligned active force magnitude of all particles
                faB_norm=np.append(faB_norm, part_align[h]*self.peB)  # Aligned active force magnitude of particle type B
                alignB_norm=np.append(alignB_norm, part_align[h])  # Alignment of particle type B
                rB_dist_norm = np.append(rB_dist_norm, difr)  #Distance from custer CoM of particle type B

            # Save separation distance from the nearest interface surface
            r_dist_norm = np.append(r_dist_norm, difr)  #Distance from custer CoM of all particles
            align_norm=np.append(align_norm, part_align[h])  # Alignment of all particles

        # Dictionary containing each particle's alignment and aligned active force toward
        # the nearest interface surface normal as a function of separation distance from
        # largest custer's CoM
        radial_fa_dict = {'all': {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}, 'A': {'r': rA_dist_norm, 'fa': faA_norm, 'align': alignA_norm}, 'B': {'r': rB_dist_norm, 'fa': faB_norm, 'align': alignB_norm}}

        return radial_fa_dict

    def heterogeneity_single_particles(self, body_force_dict, phase_dict, avg_body_force_dict, type_m='phases'):
        
        bulk_id = np.where(phase_dict['part']==0)[0]
        int_id = np.where(phase_dict['part']==1)[0]
        gas_id = np.where(phase_dict['part']==2)[0]

        bulk_A_id = np.where((phase_dict['part']==0) & (self.typ==0))[0]
        bulk_B_id = np.where((phase_dict['part']==0) & (self.typ==1))[0]

        int_A_id = np.where((phase_dict['part']==1) & (self.typ==0))[0]
        int_B_id = np.where((phase_dict['part']==1) & (self.typ==1))[0]

        gas_A_id = np.where((phase_dict['part']==2) & (self.typ==0))[0]
        gas_B_id = np.where((phase_dict['part']==2) & (self.typ==1))[0]
        #if (avg_body_force_dict['all']['bulk'])

        if type_m == 'phases':
            hetero_bulk_all = ((body_force_dict['align_fa'][bulk_id]-np.mean(body_force_dict['align_fa'][bulk_id]))**2)
            hetero_bulk_A = ((body_force_dict['align_fa'][bulk_A_id]-np.mean(body_force_dict['align_fa'][bulk_A_id]))**2)
            hetero_bulk_B = ((body_force_dict['align_fa'][bulk_B_id]-np.mean(body_force_dict['align_fa'][bulk_B_id]))**2)

            hetero_int_all = ((body_force_dict['align_fa'][int_id]-np.mean(body_force_dict['align_fa'][int_id]))**2)
            hetero_int_A = ((body_force_dict['align_fa'][int_A_id]-np.mean(body_force_dict['align_fa'][int_A_id]))**2)
            hetero_int_B = ((body_force_dict['align_fa'][int_B_id]-np.mean(body_force_dict['align_fa'][int_B_id]))**2)

            hetero_gas_all = ((body_force_dict['align_fa'][gas_id]-np.mean(body_force_dict['align_fa'][gas_id]))**2)
            hetero_gas_A = ((body_force_dict['align_fa'][gas_A_id]-np.mean(body_force_dict['align_fa'][gas_A_id]))**2)
            hetero_gas_B = ((body_force_dict['align_fa'][gas_B_id]-np.mean(body_force_dict['align_fa'][gas_B_id]))**2)
        elif type_m == 'system':
            hetero_bulk_all = ((body_force_dict['align_fa'][bulk_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_bulk_A = ((body_force_dict['align_fa'][bulk_A_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_bulk_B = ((body_force_dict['align_fa'][bulk_B_id]-np.mean(body_force_dict['align_fa']))**2)

            hetero_int_all = ((body_force_dict['align_fa'][int_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_int_A = ((body_force_dict['align_fa'][int_A_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_int_B = ((body_force_dict['align_fa'][int_B_id]-np.mean(body_force_dict['align_fa']))**2)

            hetero_gas_all = ((body_force_dict['align_fa'][gas_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_gas_A = ((body_force_dict['align_fa'][gas_A_id]-np.mean(body_force_dict['align_fa']))**2)
            hetero_gas_B = ((body_force_dict['align_fa'][gas_B_id]-np.mean(body_force_dict['align_fa']))**2)

        hetero_dense_all = np.append(hetero_bulk_all, hetero_int_all)
        hetero_system_all = np.append(hetero_dense_all, hetero_gas_all)

        pos_dense_x = np.append(self.pos[bulk_id,0], self.pos[int_id,0])
        pos_system_x = np.append(pos_dense_x, self.pos[gas_id,0])

        pos_dense_y = np.append(self.pos[bulk_id,1], self.pos[int_id,1])
        pos_system_y = np.append(pos_dense_y, self.pos[gas_id,1])

        hetero_plot_dict = {'x': pos_system_x, 'y': pos_system_y, 'hetero': hetero_system_all}

        return hetero_plot_dict

    def gas_radial_df(self):
        #DONE!
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

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]

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
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))

        #Checks for 'A' type neighbors around type 'A' particles within bulk
        AA_nlist = system_A.query(self.f_box.wrap(pos_A), query_args).toNeighborList()

        #Checks for 'A' type neighbors around type 'B' particles within bulk
        AB_nlist = system_A.query(self.f_box.wrap(pos_B), query_args).toNeighborList()

        #Checks for 'B' type neighbors around type 'A' particles within bulk
        BA_nlist = system_B.query(self.f_box.wrap(pos_A), query_args).toNeighborList()

        #Checks for 'B' type neighbors around type 'B' particles within bulk
        BB_nlist = system_B.query(self.f_box.wrap(pos_B), query_args).toNeighborList()

        # Calculate interparticle distance between type A reference particle and type A neighbor
        difr_AA = self.utility_functs.sep_dist_arr(pos_A[AA_nlist.point_indices], pos_A[AA_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type A neighbor
        difr_AB = self.utility_functs.sep_dist_arr(pos_A[AB_nlist.point_indices], pos_B[AB_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type A neighbor
        difr_BA = self.utility_functs.sep_dist_arr(pos_B[BA_nlist.point_indices], pos_A[BA_nlist.query_point_indices])

        # Calculate interparticle distance between type B reference particle and type B neighbor
        difr_BB = self.utility_functs.sep_dist_arr(pos_B[BB_nlist.point_indices], pos_B[BB_nlist.query_point_indices])

        # Calculate interparticle distance between all reference particle and type A neighbor
        difr_allA = np.append(difr_AA, difr_AB)

        # Calculate interparticle distance between all reference particle and type B neighbor
        difr_allB = np.append(difr_BB, difr_AB)

        # Calculate interparticle distance between all reference particle and all neighbors
        difr_allall = np.append(difr_allA, difr_allB)

        #Initiate empty arrays for RDF of each respective activity pairing
        g_r_allall = []
        g_r_allA = []
        g_r_allB = []
        g_r_AA = []
        g_r_AB = []
        g_r_BA = []
        g_r_BB = []
        r_arr = []

        # Loop over radial slices
        for m in range(1, len(r)-2):
            difr = r[m+1] - r[m]

            #Save minimum interparticle separation from separation range
            r_arr.append(r[m])

            # Locate all neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allall>=r[m]) & (difr_allall<r[m+1]))[0]

            # Total number of all-all particle pairs divided by volume of radial slice
            rho_all = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-all pairs divided by volume of the bulk phase
            rho_tot_all = len(self.pos) * len(self.pos)/(self.lx_box * self.ly_box)

            # Save all-all RDF at respective distance range
            g_r_allall.append(rho_all / rho_tot_all)

            # Locate A neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allA>=r[m]) & (difr_allA<r[m+1]))[0]

            # Total number of all-A particle pairs divided by volume of radial slice
            rho_a = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-A pairs divided by volume of the bulk phase
            rho_tot_a = len(pos_A) * len(self.pos)/(self.lx_box * self.ly_box)

            # Save all-A RDF at respective distance range
            g_r_allA.append(rho_a / rho_tot_a)

            # Locate B neighboring particles within given distance range from all reference particle
            inds = np.where((difr_allB>=r[m]) & (difr_allB<r[m+1]))[0]

            # Total number of all-B particle pairs divided by volume of radial slice
            rho_b = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible all-B pairs divided by volume of the bulk phase
            rho_tot_b = len(pos_B) * len(self.pos)/(self.lx_box * self.ly_box)

            # Save all-B RDF at respective distance range
            g_r_allB.append(rho_b / rho_tot_b)

            # Locate A neighboring particles within given distance range from A reference particle
            inds = np.where((difr_AA>=r[m]) & (difr_AA<r[m+1]))[0]

            # Total number of A-A particle pairs divided by volume of radial slice
            rho_aa = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible A-A pairs divided by volume of the bulk phase
            rho_tot_aa = len(pos_A) * len(pos_A)/(self.lx_box * self.ly_box)

            # Save A-A RDF at respective distance range
            g_r_AA.append(rho_aa / rho_tot_aa)

            # Locate B neighboring particles within given distance range from A reference particle
            inds = np.where((difr_AB>=r[m]) & (difr_AB<r[m+1]))[0]

            # Total number of A-B particle pairs divided by volume of radial slice
            rho_ab = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible A-B pairs divided by volume of the bulk phase
            rho_tot_ab = len(pos_B) * len(pos_A)/(self.lx_box * self.ly_box)

            # Save A-B RDF at respective distance range
            g_r_AB.append(rho_ab / rho_tot_ab)

            # Locate A neighboring particles within given distance range from B reference particle
            inds = np.where((difr_BA>=r[m]) & (difr_BA<r[m+1]))[0]

            # Total number of B-A particle pairs divided by volume of radial slice
            rho_ba = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible B-A pairs divided by volume of the bulk phase
            rho_tot_ba = len(pos_A) * len(pos_B)/(self.lx_box * self.ly_box)

            # Save B-A RDF at respective distance range
            g_r_BA.append(rho_ba / rho_tot_ba)

            # Locate B neighboring particles within given distance range from B reference particle
            inds = np.where((difr_BB>=r[m]) & (difr_BB<r[m+1]))[0]

            # Total number of B-B particle pairs divided by volume of radial slice (numerator of RDF)
            rho_bb = (len(inds) / (2*math.pi * r[m] * difr) )

            # Total number of maximum possible B-B pairs divided by volume of the bulk phase
            rho_tot_bb = len(pos_B) * len(pos_B)/(self.lx_box * self.ly_box)

            # Save B-B RDF at respective distance range
            g_r_BB.append(rho_bb / rho_tot_bb)

        # Create output dictionary for plotting of RDF vs separation distance
        rad_df_dict = {'r': r_arr, 'all-all': g_r_allall, 'all-A': g_r_allA, 'all-B': g_r_allB, 'A-A': g_r_AA, 'A-B': g_r_AB, 'B-A': g_r_BA, 'B-B': g_r_BB}
        return rad_df_dict

    def angular_velocity(self, ang_vel, phasePart):
        '''
        Purpose: Takes the angular velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        ang_vel: array (partNum) of angular velocities of each particle

        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        ang_vel_phase_dict: dictionary containing the average and standard deviation of angular velocity
        of each phase and each respective type ('all', 'A', or 'B')
        '''

        # Total angular velocity of bulk particles of respective type ('all', 'A', or 'B')
        bulk_ang_vel = 0
        bulk_A_ang_vel = 0
        bulk_B_ang_vel = 0

        # Total number of bulk particles of respective type ('all', 'A', or 'B') summed over
        bulk_ang_num = 0
        bulk_A_ang_num = 0
        bulk_B_ang_num = 0

        # Total angular velocity of interface particles of respective type ('all', 'A', or 'B')
        int_ang_vel = 0
        int_A_ang_vel = 0
        int_B_ang_vel = 0

        # Total number of interface particles of respective type ('all', 'A', or 'B') summed over
        int_ang_num = 0
        int_A_ang_num = 0
        int_B_ang_num = 0

        # Total angular velocity of gas particles of respective type ('all', 'A', or 'B')
        gas_ang_vel = 0
        gas_A_ang_vel = 0
        gas_B_ang_vel = 0

        # Total number of gas particles of respective type ('all', 'A', or 'B') summed over
        gas_ang_num = 0
        gas_A_ang_num = 0
        gas_B_ang_num = 0

        # Loop over all particles to calculate mean angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If reference particle is from bulk...
            if phasePart[h] == 0:

                # Add to total bulk phase angular velocity
                bulk_ang_vel += np.abs(ang_vel[h])
                bulk_ang_num += 1

                # Add to total A bulk phase angular velocity
                if self.typ[h] == 0:
                    bulk_A_ang_vel += np.abs(ang_vel[h])
                    bulk_A_ang_num += 1

                # Add to total B bulk phase angular velocity
                elif self.typ[h] == 1:
                    bulk_B_ang_vel += np.abs(ang_vel[h])
                    bulk_B_ang_num += 1

            # If reference particle is from interface
            elif phasePart[h] == 1:

                # Add to total interface angular velocity
                int_ang_vel += np.abs(ang_vel[h])
                int_ang_num += 1

                # Add to total A interface angular velocity
                if self.typ[h] == 0:
                    int_A_ang_vel += np.abs(ang_vel[h])
                    int_A_ang_num += 1

                # Add to total B interface angular velocity
                elif self.typ[h] == 1:
                    int_B_ang_vel += np.abs(ang_vel[h])
                    int_B_ang_num += 1

            # If reference particle is from gas
            elif phasePart[h] == 2:

                # Add to total gas phase angular velocity
                gas_ang_vel += np.abs(ang_vel[h])
                gas_ang_num += 1

                # Add to total A gas phase angular velocity
                if self.typ[h] == 0:
                    gas_A_ang_vel += np.abs(ang_vel[h])
                    gas_A_ang_num += 1

                # Add to total B gas phase angular velocity
                elif self.typ[h] == 1:
                    gas_B_ang_vel += np.abs(ang_vel[h])
                    gas_B_ang_num += 1

        # Average total bulk angular velocities over total bulk particles of respective type ('all', 'A', or 'B')
        if bulk_ang_num > 0:
            bulk_ang_vel_avg = bulk_ang_vel/bulk_ang_num
            if bulk_A_ang_num > 0:
                bulk_A_ang_vel_avg = bulk_A_ang_vel/bulk_A_ang_num
            else:
                bulk_A_ang_vel_avg = 0

            if bulk_B_ang_num > 0:
                bulk_B_ang_vel_avg = bulk_B_ang_vel/bulk_B_ang_num
            else:
                bulk_B_ang_vel_avg = 0
        else:
            bulk_ang_vel_avg = 0
            bulk_A_ang_vel_avg = 0
            bulk_B_ang_vel_avg = 0

        # Average total interface angular velocities over total interface particles of respective type ('all', 'A', or 'B')
        if int_ang_num > 0:
            int_ang_vel_avg = int_ang_vel/int_ang_num

            if int_A_ang_num > 0:
                int_A_ang_vel_avg = int_A_ang_vel/int_A_ang_num
            else:
                int_A_ang_vel_avg = 0

            if int_B_ang_num > 0:
                int_B_ang_vel_avg = int_B_ang_vel/int_B_ang_num
            else:
                int_B_ang_vel_avg = 0
        else:
            int_ang_vel_avg = 0
            int_A_ang_vel_avg = 0
            int_B_ang_vel_avg = 0

        # Average total gas phase angular velocities over total gas particles of respective type ('all', 'A', or 'B')
        if gas_ang_num > 0:
            gas_ang_vel_avg = gas_ang_vel/gas_ang_num
            if gas_A_ang_num > 0:
                gas_A_ang_vel_avg = gas_A_ang_vel/gas_A_ang_num
            else:
                gas_A_ang_vel_avg = 0

            if gas_B_ang_num > 0:
                gas_B_ang_vel_avg = gas_B_ang_vel/gas_B_ang_num
            else:
                gas_B_ang_vel_avg = 0
        else:
            gas_ang_vel_avg = 0
            gas_A_ang_vel_avg = 0
            gas_B_ang_vel_avg = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for bulk phase and particle type ('all', 'A', or 'B')
        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for interface and particle type ('all', 'A', or 'B')
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0

        # Instantiate sums of all particles' deviations from mean angular velocity for gas phase and particle type ('all', 'A', or 'B')
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        # Loop over all particles to calculate standard deviation angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If bulk, sum to bulk deviations from mean of respective type ('all', 'A', or 'B')
            if phasePart[h] == 0:

                bulk_all_val += (np.abs(ang_vel[h])-bulk_ang_vel_avg)**2
                if self.typ[h]==0:
                    bulk_A_val += (np.abs(ang_vel[h])-bulk_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_B_val += (np.abs(ang_vel[h])-bulk_B_ang_vel_avg)**2

            # If interface, sum to interface deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 1:

                int_all_val += (np.abs(ang_vel[h])-int_ang_vel_avg)**2
                if self.typ[h]==0:
                    int_A_val += (np.abs(ang_vel[h])-int_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    int_B_val += (np.abs(ang_vel[h])-int_B_ang_vel_avg)**2

            # If gas, sum to gas deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 2:

                gas_all_val += (np.abs(ang_vel[h])-gas_ang_vel_avg)**2
                if self.typ[h]==0:
                    gas_A_val += (np.abs(ang_vel[h])-gas_A_ang_vel_avg)**2
                elif self.typ[h]==1:
                    gas_B_val += (np.abs(ang_vel[h])-gas_B_ang_vel_avg)**2

        # Average sum of bulk deviations divided by number of bulk particles of respective type ('all', 'A', or 'B')
        if bulk_ang_num > 0:

            bulk_ang_vel_std = (bulk_all_val/bulk_ang_num)**0.5

            if bulk_A_ang_num > 0:
                bulk_A_ang_vel_std = (bulk_A_val/bulk_A_ang_num)**0.5
            else:
                bulk_A_ang_vel_std = 0

            if bulk_B_ang_num > 0:
                bulk_B_ang_vel_std = (bulk_B_val/bulk_B_ang_num)**0.5
            else:
                bulk_B_ang_vel_std = 0

        else:
            bulk_ang_vel_std = 0
            bulk_A_ang_vel_std = 0
            bulk_B_ang_vel_std = 0

        # Average sum of interface deviations divided by number of interface particles of respective type ('all', 'A', or 'B')
        if int_ang_num > 0:
            int_ang_vel_std = (int_all_val/int_ang_num)**0.5
            if int_A_ang_num > 0:
                int_A_ang_vel_std = (int_A_val/int_A_ang_num)**0.5
            else:
                int_A_ang_vel_std = 0
            if int_B_ang_num > 0:
                int_B_ang_vel_std = (int_B_val/int_B_ang_num)**0.5
            else:
                int_B_ang_vel_std = 0
        else:
            int_ang_vel_std = 0
            int_A_ang_vel_std = 0
            int_B_ang_vel_std = 0

        # Average sum of gas deviations divided by number of gas particles of respective type ('all', 'A', or 'B')
        if gas_ang_num > 0:
            gas_ang_vel_std = (gas_all_val/gas_ang_num)**0.5
            if gas_A_ang_num > 0:
                gas_A_ang_vel_std = (gas_A_val/gas_A_ang_num)**0.5
            else:
                gas_A_ang_vel_std = 0
            if gas_B_ang_num > 0:
                gas_B_ang_vel_std = (gas_B_val/gas_B_ang_num)**0.5
            else:
                gas_B_ang_vel_std = 0
        else:
            gas_ang_vel_std = 0
            gas_A_ang_vel_std = 0
            gas_B_ang_vel_std = 0

        # Dictionary containing the average and standard deviation of angular velocity
        # of each phase and each respective type ('all', 'A', or 'B')
        ang_vel_phase_dict = {'bulk': {'all': {'avg': bulk_ang_vel_avg, 'std': bulk_ang_vel_std}, 'A': {'avg': bulk_A_ang_vel_avg, 'std': bulk_A_ang_vel_std}, 'B': {'avg': bulk_B_ang_vel_avg, 'std': bulk_B_ang_vel_std} }, 'int': {'all': {'avg': int_ang_vel_avg, 'std': int_ang_vel_std}, 'A': {'avg': int_A_ang_vel_avg, 'std': int_A_ang_vel_std}, 'B': {'avg': int_B_ang_vel_avg, 'std': int_B_ang_vel_std}}, 'gas': {'all': {'avg': gas_ang_vel_avg, 'std': gas_ang_vel_std}, 'A': {'avg': gas_A_ang_vel_avg, 'std': gas_A_ang_vel_std}, 'B': {'avg': gas_B_ang_vel_avg, 'std': gas_B_ang_vel_std}}}

        return ang_vel_phase_dict

    def velocity(self, vel, phasePart):
        '''
        Purpose: Takes the velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        vel: array (partNum) of velocities of each particle

        phasePart: array (partNum) labeling whether particle is a member of bulk (0),
        interface (1), or gas (2) phase

        Output:
        vel_phase_dict: dictionary containing the average and standard deviation of velocity
        of each phase and each respective type ('all', 'A', or 'B')

        vel_phase_plot_dict: dictionary containing the velocity and position
        of each particle for each phase and each respective type ('all', 'A', or 'B')
        '''

        # Total velocity of bulk particles of respective type ('all', 'A', or 'B')
        bulk_vel = 0
        bulk_A_vel = 0
        bulk_B_vel = 0

        # Total number of bulk particles of respective type ('all', 'A', or 'B') summed over
        bulk_num = 0
        bulk_A_num = 0
        bulk_B_num = 0

        # Total velocity of interface particles of respective type ('all', 'A', or 'B')
        int_vel = 0
        int_A_vel = 0
        int_B_vel = 0

        # Total number of interface particles of respective type ('all', 'A', or 'B') summed over
        int_num = 0
        int_A_num = 0
        int_B_num = 0

        # Total velocity of gas particles of respective type ('all', 'A', or 'B')
        gas_vel = 0
        gas_A_vel = 0
        gas_B_vel = 0

        # Total number of gas particles of respective type ('all', 'A', or 'B') summed over
        gas_num = 0
        gas_A_num = 0
        gas_B_num = 0

        # Loop over all particles to calculate mean angular velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If reference particle is from bulk
            if phasePart[h] == 0:

                # Add to total bulk velocity
                bulk_vel += vel[h]
                bulk_num += 1

                # Add to total A bulk velocity
                if self.typ[h] == 0:
                    bulk_A_vel += vel[h]
                    bulk_A_num += 1

                # Add to total B bulk velocity
                elif self.typ[h] == 1:
                    bulk_B_vel += vel[h]
                    bulk_B_num += 1

            # If reference particle is from interface
            elif phasePart[h] == 1:

                # Add to total interface velocity
                int_vel += vel[h]
                int_num += 1

                # Add to total A interface velocity
                if self.typ[h] == 0:
                    int_A_vel += vel[h]
                    int_A_num += 1

                # Add to total B interface velocity
                elif self.typ[h] == 1:
                    int_B_vel += vel[h]
                    int_B_num += 1

            # If reference particle is from gas
            elif phasePart[h] == 2:

                # Add to total gas phase velocity
                gas_vel += vel[h]
                gas_num += 1

                # Add to total A gas phase velocity
                if self.typ[h] == 0:
                    gas_A_vel += vel[h]
                    gas_A_num += 1

                # Add to total B gas phase velocity
                elif self.typ[h] == 1:
                    gas_B_vel += vel[h]
                    gas_B_num += 1

        # Average total bulk phase angular velocities over total bulk particles of respective type ('all', 'A', or 'B')
        if bulk_num > 0:
            bulk_vel_avg = bulk_vel/bulk_num
            if bulk_A_num > 0:
                bulk_A_vel_avg = bulk_A_vel/bulk_A_num
            else:
                bulk_A_vel_avg = 0

            if bulk_B_num > 0:
                bulk_B_vel_avg = bulk_B_vel/bulk_B_num
            else:
                bulk_B_vel_avg = 0
        else:
            bulk_vel_avg = 0
            bulk_A_vel_avg = 0
            bulk_B_vel_avg = 0

        # Average total interface phase velocities over total interface particles of respective type ('all', 'A', or 'B')
        if int_num > 0:
            int_vel_avg = int_vel/int_num

            if int_A_num > 0:
                int_A_vel_avg = int_A_vel/int_A_num
            else:
                int_A_vel_avg = 0

            if int_B_num > 0:
                int_B_vel_avg = int_B_vel/int_B_num
            else:
                int_B_vel_avg = 0
        else:
            int_vel_avg = 0
            int_A_vel_avg = 0
            int_B_vel_avg = 0

        # Average total gas phase velocities over total gas particles of respective type ('all', 'A', or 'B')
        if gas_num > 0:
            gas_vel_avg = gas_vel/gas_num
            if gas_A_num > 0:
                gas_A_vel_avg = gas_A_vel/gas_A_num
            else:
                gas_A_vel_avg = 0

            if gas_B_num > 0:
                gas_B_vel_avg = gas_B_vel/gas_B_num
            else:
                gas_B_vel_avg = 0
        else:
            gas_vel_avg = 0
            gas_A_vel_avg = 0
            gas_B_vel_avg = 0

        # Instantiate sums of all particles' deviations from mean velocity for bulk phase and particle type ('all', 'A', or 'B')
        bulk_all_val = 0
        bulk_A_val = 0
        bulk_B_val = 0

        # Instantiate sums of all particles' deviations from mean velocity for interface and particle type ('all', 'A', or 'B')
        int_all_val = 0
        int_A_val = 0
        int_B_val = 0

        # Instantiate sums of all particles' deviations from mean velocity for gas phase and particle type ('all', 'A', or 'B')
        gas_all_val = 0
        gas_A_val = 0
        gas_B_val = 0

        # Loop over all particles to calculate standard deviation velocities of each particle type and phase
        for h in range(0, self.partNum):

            # If bulk, sum to bulk deviations from mean of respective type ('all', 'A', or 'B')
            if phasePart[h] == 0:

                bulk_all_val += (np.abs(vel[h])-bulk_vel_avg)**2

                if self.typ[h]==0:
                    bulk_A_val += (np.abs(vel[h])-bulk_A_vel_avg)**2
                elif self.typ[h]==1:
                    bulk_B_val += (np.abs(vel[h])-bulk_B_vel_avg)**2

            # If interface, sum to interface deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 1:

                int_all_val += (np.abs(vel[h])-int_vel_avg)**2

                if self.typ[h]==0:
                    int_A_val += (np.abs(vel[h])-int_A_vel_avg)**2
                elif self.typ[h]==1:
                    int_B_val += (np.abs(vel[h])-int_B_vel_avg)**2

            # If gas, sum to gas deviations from mean of respective type ('all', 'A', or 'B')
            elif phasePart[h] == 2:

                gas_all_val += (np.abs(vel[h])-gas_vel_avg)**2

                if self.typ[h]==0:
                    gas_A_val += (np.abs(vel[h])-gas_A_vel_avg)**2
                elif self.typ[h]==1:
                    gas_B_val += (np.abs(vel[h])-gas_B_vel_avg)**2


        # Average sum of bulk deviations divided by number of bulk particles of respective type ('all', 'A', or 'B')
        if bulk_num > 0:
            bulk_vel_std = (bulk_all_val/bulk_num)**0.5
            if bulk_A_num > 0:
                bulk_A_vel_std = (bulk_A_val/bulk_A_num)**0.5
            else:
                bulk_A_vel_std = 0
            if bulk_B_num > 0:
                bulk_B_vel_std = (bulk_B_val/bulk_B_num)**0.5
            else:
                bulk_B_vel_std = 0
        else:
            bulk_vel_std = 0
            bulk_A_vel_std = 0
            bulk_B_vel_std = 0
        
        # Average sum of interface deviations divided by number of interface particles of respective type ('all', 'A', or 'B')
        if int_num > 0:
            int_vel_std = (int_all_val/int_num)**0.5
            if int_A_num > 0:
                int_A_vel_std = (int_A_val/int_A_num)**0.5
            else:
                int_A_vel_std = 0
            if int_B_num > 0:
                int_B_vel_std = (int_B_val/int_B_num)**0.5
            else:
                int_B_vel_std = 0
        else:
            int_vel_std = 0
            int_A_vel_std = 0
            int_B_vel_std = 0

        # Average sum of gas deviations divided by number of gas particles of respective type ('all', 'A', or 'B')
        if gas_num > 0:
            gas_vel_std = (gas_all_val/gas_num)**0.5
            if gas_A_num > 0:
                gas_A_vel_std = (gas_A_val/gas_A_num)**0.5
            else:
                gas_A_vel_std = 0
            if gas_B_num > 0:
                gas_B_vel_std = (gas_B_val/gas_B_num)**0.5
            else:
                gas_B_vel_std = 0
        else:
            gas_vel_std = 0
            gas_A_vel_std = 0
            gas_B_vel_std = 0


        # Dictionary containing the average and standard deviation of velocity
        # of each phase and each respective type ('all', 'A', or 'B')
        vel_phase_dict = {'bulk': {'all': {'avg': bulk_vel_avg, 'std': bulk_vel_std}, 'A': {'avg': bulk_A_vel_avg, 'std': bulk_A_vel_std}, 'B': {'avg': bulk_B_vel_avg, 'std': bulk_B_vel_std} }, 'int': {'all': {'avg': int_vel_avg, 'std': int_vel_std}, 'A': {'avg': int_A_vel_avg, 'std': int_A_vel_std}, 'B': {'avg': int_B_vel_avg, 'std': int_B_vel_std}}, 'gas': {'all': {'avg': gas_vel_avg, 'std': gas_vel_std}, 'A': {'avg': gas_A_vel_avg, 'std': gas_A_vel_std}, 'B': {'avg': gas_B_vel_avg, 'std': gas_B_vel_std}}}

        # Particle IDs of all, A, or B gas particles
        gas_id = np.where(phasePart == 2 )[0]
        gasA_id = np.where( (self.typ == 0 ) & ( phasePart == 2 ) )[0]
        gasB_id = np.where( (self.typ == 1 ) & ( phasePart == 2 ) )[0]

        vel_phase_plot_dict = {'all': {'mag':vel[gas_id]}, 'A': {'mag': vel[gasA_id]}, 'B': {'mag': vel[gasB_id]} }

        return vel_phase_dict, vel_phase_plot_dict

    def part_velocity(self, prev_pos, prev_ang, ori, dt_step):
        '''
        Purpose: Calculates the velocity of each particle type indescriminate of phases

        Inputs:
        prev_pos: array (partNum) of particle positions from previous time step

        prev_ang: array(partNum) of particle orientations (radians) from previous time step

        ori: array (partNum) of particle orientations (radians) from current time step

        Output:
        vel_plot_dict: dictionary containing the velocity
        of each respective type ('all', 'A', or 'B')

        vel_stat_dict: dictionary containing the statistical values of velocity
        of each respective type ('all', 'A', or 'B')
        '''
        
        # Calculate displacements
        dx, dy, dr = self.utility_functs.sep_dist_arr(self.pos, prev_pos, difxy=True)

        # Find IDs of each particle type
        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        typ01_vel_mag = ((dx**2 + dy**2) ** 0.5)/dt_step

        # Slow particle properties
        typ0_vel_x = dx[typ0ind]/dt_step
        typ0_vel_y = dy[typ0ind]/dt_step
        typ0_vel_mag = ((typ0_vel_x**2 + typ0_vel_y**2) ** 0.5)
        pos0 = self.pos[typ0ind]

        # Fast particle properties
        typ1_vel_x = dx[typ1ind]/dt_step
        typ1_vel_y = dy[typ1ind]/dt_step
        typ1_vel_mag = ((typ1_vel_x**2 + typ1_vel_y**2) ** 0.5)
        pos1 = self.pos[typ1ind]

        # Slow/fast particle average velocities
        typ01_avg = np.mean(typ01_vel_mag)
        typ0_avg = np.mean(typ0_vel_mag)
        typ1_avg = np.mean(typ1_vel_mag)

        # Slow/fast particle standard deviation velocities
        typ01_std = np.std(typ01_vel_mag)
        typ0_std = np.std(typ0_vel_mag)
        typ1_std = np.std(typ1_vel_mag)

        # Slow/fast particle mode velocities
        typ01_mode = stats.mode(typ01_vel_mag)[0][0]
        typ0_mode = stats.mode(typ0_vel_mag)[0][0]
        typ1_mode = stats.mode(typ1_vel_mag)[0][0]

        # Slow/fast particle median velocities
        typ01_median = np.median(typ01_vel_mag)
        typ0_median = np.median(typ0_vel_mag)
        typ1_median = np.median(typ1_vel_mag)

        # Dictionary containing the velocity of each particle type ('all', 'A', or 'B') for plotting
        vel_plot_dict = {'all': {'x': dx/dt_step, 'y': dy/dt_step, 'mag': dr/dt_step, 'pos': self.pos}, 'A': {'x': typ0_vel_x, 'y': typ0_vel_y, 'mag': typ0_vel_mag, 'pos': pos0}, 'B': {'x': typ1_vel_x, 'y': typ1_vel_y, 'mag': typ1_vel_mag, 'pos': pos1} }

        # Dictionary containing the average, mode, standard deviation, and median of velocity
        # of each particle type ('all', 'A', or 'B')
        vel_stat_dict = {'all': {'mean': typ01_avg, 'std': typ01_std, 'mode': typ01_mode, 'median': typ01_median}, 'A': {'mean': typ0_avg, 'std': typ0_std, 'mode': typ0_mode, 'median': typ0_median}, 'B': {'mean': typ1_avg, 'std': typ1_std, 'mode': typ1_mode, 'median': typ1_median} }

        return vel_plot_dict, vel_stat_dict

    def velocity_corr(self, part_vel, prev_pos, prev_ang, ori):
        '''
        Purpose: Takes the velocity and phase of each particle
        to calculate the mean and standard deviation of each phase for each
        respective particle type

        Inputs:
        part_vel: array (partNum) of velocities of each particle

        prev_pos: array (partNum) of particle positions from previous time step

        prev_ang: array(partNum) of particle orientations (radians) from previous time step

        ori: array (partNum) of particle orientations (radians) from current time step

        Output:
        vel_phase_dict: dictionary containing the average and standard deviation of velocity
        of each phase and each respective type ('all', 'A', or 'B')

        vel_plot_dict: dictionary containing the velocity
        of each respective type ('all', 'A', or 'B')

        corr_dict: dictionary containing the correlation of particle velocity with active force orientation
        '''
        
        # Calculate displacement
        dx, dy, dr = self.utility_functs.sep_dist_arr(self.pos, prev_pos, difxy=True)

        # Find IDs of each particle type
        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        # Slow particle properties
        typ0_vel_x = dx[typ0ind]
        typ0_vel_y = dy[typ0ind]
        typ0_vel_mag = (typ0_vel_x**2 + typ0_vel_y**2) ** 0.5
        pos0 = self.pos[typ0ind]

        # Fast particle properties
        typ1_vel_x = dx[typ1ind]
        typ1_vel_y = dy[typ1ind]
        typ1_vel_mag = (typ1_vel_x**2 + typ1_vel_y**2) ** 0.5
        pos1 = self.pos[typ1ind]

        # Slow/fast particle average velocities
        typ0_avg = np.mean(typ0_vel_mag)
        typ1_avg = np.mean(typ1_vel_mag)

        # Interparticle search distances
        r = np.arange(self.r_cut, 7*self.r_cut, self.r_cut)
        
        # Loop over interparticle search distances
        for r_dist in r:

            # Define minimum interparticle separation distance
            if r_dist == self.r_cut:
                r_start = 0.1
            else:
                r_start = self.r_cut+0.000001
            
            # Define neighbor list query arguments
            query_args = dict(mode='ball', r_min = r_start, r_max = r_dist)
            
            # Locate A-type particles
            system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(prev_pos[typ0ind]))

            # Generate neighbor list of respective type A neighboring reference particles of type B
            AB_nlist = system_A.query(self.f_box.wrap(prev_pos[typ1ind]), query_args).toNeighborList()                

            if len(AB_nlist.query_point_indices) > 0:
                
                # Find neighbors list IDs where 0 is reference particle
                loc = np.where(AB_nlist.query_point_indices==0)[0]

                # Slow particle displacement
                slow_displace_dx, slow_displace_dy, slow_displace_dr = self.utility_functs.sep_dist_arr(self.pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ0ind][AB_nlist.point_indices[loc]], difxy=True)

                # Fast particle displacement
                fast_displace_dr = dr[typ1ind][AB_nlist.query_point_indices[loc]]
                fast_displace_dx = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                fast_displace_dy = dy[typ1ind][AB_nlist.query_point_indices[loc]]
                
                # Previous interparticle separation distance
                sep_difx, sep_dify, sep_difr = self.utility_functs.sep_dist_arr(prev_pos[typ0ind][AB_nlist.point_indices[loc]], prev_pos[typ1ind][AB_nlist.query_point_indices[loc]], difxy=True)

                # Local correlation between slow and fast particle displacement
                vel_x_corr_loc = (slow_displace_dx/slow_displace_dr) * (fast_displace_dx/fast_displace_dr)
                vel_y_corr_loc = (slow_displace_dy/slow_displace_dr) * (fast_displace_dy/fast_displace_dr)
                vel_r_corr_loc = ( vel_x_corr_loc ** 2 + vel_y_corr_loc ** 2 ) ** 0.5         

                # displacement orientation
                theta = np.arctan(sep_dify, sep_difx)

                # Active force magnitude
                fx = ori[typ1ind,1]
                fy = ori[typ1ind,2]

                # Active force magnitude local correlation
                f_x_corr_loc = (slow_displace_dx/slow_displace_dr) * (fx)
                f_y_corr_loc = (slow_displace_dy/slow_displace_dr) * (fy)
                f_r_corr_loc = (f_x_corr_loc ** 2 + f_y_corr_loc ** 2) ** 0.5

            # Dictionary containing the velocity of each particle to plot
            vel_plot_dict = {'A': {'x': typ0_vel_x, 'y': typ0_vel_y, 'mag': typ0_vel_mag, 'pos': pos0}, 'B': {'x': typ1_vel_x, 'y': typ1_vel_y, 'mag': typ1_vel_mag, 'pos': pos1} }
            
            # Dictionary containing the correlation of active force and velocity between particles
            corr_dict = {'f': {'x': f_x_corr_loc, 'y': f_y_corr_loc, 'r': f_r_corr_loc}, 'v': {'x': vel_x_corr_loc, 'y': vel_y_corr_loc, 'r': vel_r_corr_loc}}
            
            # Dictionary containing the average and standard deviation of velocity
            # of each phase and each respective type ('all', 'A', or 'B')
            vel_stat_dict = {'A': {'mag': typ0_avg}, 'B': {'mag': typ1_avg} }

            return vel_plot_dict, corr_dict, vel_stat_dict
    """         
    def adsorption(self):
        #IN PROGRESS
        '''
        Purpose: Calculates the rate of adsorption to and desorption from the cluster

        Output:
        mem_min: 
        '''

        # Particle IDs of each type
        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        # Calculate minimum and maximum membrane position
        if self.lx_box >= self.ly_box:
            mem_max = np.max(self.pos[typ0ind,0])
            mem_min = np.min(self.pos[typ0ind,0])
        else:
            mem_max = np.max(self.pos[typ0ind,1])
            mem_min = np.min(self.pos[typ0ind,1])

        # Calculate number of particles adsorbing from gas to membrane on either side
        right_ads = np.where((self.pos[typ1ind,0]<(mem_max + 10.0)) & (self.pos[typ1ind,0]>0))
        left_ads = np.where((self.pos[typ1ind,0]>(mem_min - 10.0)) & (self.pos[typ1ind,0]<0))

        return mem_min

    def adsorption3(partPhase_time, clust_time):
        # IN PROGRESS
        start_part_phase = partPhase_time[:,0]
        start_bulk_id = np.where(partPhase_time[0,:]==0)[0]
        start_gas_id = np.where(partPhase_time[0,:]==2)[0]
        start_int_id = np.where(partPhase_time[0,:]==1)[0]

        start_clust_id = np.where(clust_time[0,:]==1)[0]
        start_gas2_id = np.where(clust_time[0,:]==0)[0]

        num_clust_to_gas2 = np.array([])
        num_slow_clust_to_gas2 = np.array([])
        num_fast_clust_to_gas2 = np.array([])

        num_gas2_to_clust = np.array([])
        num_slow_gas2_to_clust = np.array([])
        num_fast_gas2_to_clust = np.array([])

        num_bulk_to_gas = np.array([])
        num_slow_bulk_to_gas = np.array([])
        num_fast_bulk_to_gas = np.array([])

        num_gas_to_bulk = np.array([])
        num_slow_gas_to_bulk = np.array([])
        num_fast_gas_to_bulk = np.array([])

        num_gas_to_int = np.array([])
        num_slow_gas_to_int = np.array([])
        num_fast_gas_to_int = np.array([])

        num_int_to_gas = np.array([])
        num_slow_int_to_gas = np.array([])
        num_fast_int_to_gas = np.array([])

        num_bulk_to_int = np.array([])
        num_slow_bulk_to_int = np.array([])
        num_fast_bulk_to_int = np.array([])

        num_int_to_bulk = np.array([])
        num_slow_int_to_bulk = np.array([])
        num_fast_int_to_bulk = np.array([])

        num_bulk_to_gas = np.array([])
        num_slow_bulk_to_gas = np.array([])
        num_fast_bulk_to_gas = np.array([])

        num_gas_to_bulk = np.array([])
        num_slow_gas_to_bulk = np.array([])
        num_fast_gas_to_bulk = np.array([])

        num_bulk_to_gas_no_int = np.array([])
        num_slow_bulk_to_gas_no_int = np.array([])
        num_fast_bulk_to_gas_no_int = np.array([])

        num_gas_to_bulk_no_int = np.array([])
        num_slow_gas_to_bulk_no_int = np.array([])
        num_fast_gas_to_bulk_no_int = np.array([])

        for j in range(1, np.shape(partPhase_time)[0]):

            bulk_id = np.where(partPhase_time[j,:]==0)[0]
            gas_id = np.where(partPhase_time[j,:]==2)[0]
            int_id = np.where(partPhase_time[j,:]==1)[0]

            clust_id = np.where(in_clust_arr[j,:]==1)[0]
            gas2_id = np.where(in_clust_arr[j,:]==0)[0]

            still_in_clust = np.intersect1d(start_clust_id, clust_id, return_indices=True)
            not_in_clust = np.delete(start_clust_id, still_in_clust[1])

            still_in_gas2 = np.intersect1d(start_gas2_id, gas2_id, return_indices=True)
            not_in_gas2 = np.delete(start_gas2_id, still_in_gas2[1])

            still_in_bulk_no_int = np.intersect1d(start_bulk_id, bulk_id, return_indices=True)
            not_in_bulk_no_int = np.delete(start_bulk_id, still_in_bulk_no_int[1])

            still_in_gas_no_int = np.intersect1d(start_gas_id, gas_id, return_indices=True)
            not_in_gas_no_int = np.delete(start_gas_id, still_in_gas_no_int[1])

            still_in_bulk = np.intersect1d(start_bulk_id_with_int, bulk_id, return_indices=True)
            not_in_bulk = np.delete(start_bulk_id_with_int, still_in_bulk[1])

            still_in_gas = np.intersect1d(start_gas_id_with_int, gas_id, return_indices=True)
            not_in_gas = np.delete(start_gas_id_with_int, still_in_gas[1])

            still_in_int = np.intersect1d(start_int_id_with_int, int_id, return_indices=True)
            not_in_int = np.delete(start_int_id_with_int, still_in_int[1])

            clust_now_in_gas2 = np.intersect1d(gas2_id, not_in_clust, return_indices=True)
            not_in_clust_ids = np.intersect1d(start_clust_id, clust_now_in_gas2[0], return_indices=True)

            gas2_now_in_clust = np.intersect1d(clust_id, not_in_gas2, return_indices=True)
            not_in_gas2_ids = np.intersect1d(start_gas2_id, gas2_now_in_clust[0], return_indices=True)

            bulk_now_in_gas_no_int = np.intersect1d(gas_id, not_in_bulk_no_int, return_indices=True)
            gas_now_in_bulk_no_int = np.intersect1d(bulk_id, not_in_gas_no_int, return_indices=True)
            not_in_bulk_ids_to_gas_no_int = np.intersect1d(start_bulk_id, bulk_now_in_gas_no_int[0], return_indices=True)
            not_in_gas_ids_to_bulk_no_int = np.intersect1d(start_gas_id, gas_now_in_bulk_no_int[0], return_indices=True)

            bulk_now_in_int = np.intersect1d(int_id, not_in_bulk, return_indices=True)
            int_now_in_bulk = np.intersect1d(bulk_id, not_in_int, return_indices=True)
            not_in_bulk_ids_to_int = np.intersect1d(start_bulk_id_with_int, bulk_now_in_int[0], return_indices=True)
            not_in_int_ids_to_bulk = np.intersect1d(start_int_id_with_int, int_now_in_bulk[0], return_indices=True)

            gas_now_in_int = np.intersect1d(int_id, not_in_gas, return_indices=True)
            int_now_in_gas = np.intersect1d(gas_id, not_in_int, return_indices=True)
            not_in_gas_ids_to_int = np.intersect1d(start_gas_id_with_int, gas_now_in_int[0], return_indices=True)
            not_in_int_ids_to_gas = np.intersect1d(start_int_id_with_int, int_now_in_gas[0], return_indices=True)

            gas_now_in_bulk = np.intersect1d(bulk_id, not_in_gas, return_indices=True)
            bulk_now_in_gas = np.intersect1d(gas_id, not_in_bulk, return_indices=True)

            not_in_gas_ids_to_bulk = np.intersect1d(start_gas_id_with_int, gas_now_in_bulk[0], return_indices=True)
            not_in_bulk_ids_to_gas = np.intersect1d(start_bulk_id_with_int, bulk_now_in_gas[0], return_indices=True)

            not_in_bulk_comb = np.append(not_in_bulk_ids_to_gas[1], not_in_bulk_ids_to_int[1])
            not_in_int_comb = np.append(not_in_int_ids_to_gas[1], not_in_int_ids_to_bulk[1])
            not_in_gas_comb = np.append(not_in_gas_ids_to_bulk[1], not_in_gas_ids_to_int[1])

            if len(clust_now_in_gas2)>0:
                num_clust_to_gas2 = np.append(num_clust_to_gas2, len(clust_now_in_gas2[0]))
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, len(np.where(typ[clust_now_in_gas2[0].astype('int')]==0)[0]))
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, len(np.where(typ[clust_now_in_gas2[0].astype('int')]==1)[0]))
            else:
                num_clust_to_gas2 = np.append(num_clust_to_gas2, 0)
                num_slow_clust_to_gas2 = np.append(num_slow_clust_to_gas2, 0)
                num_fast_clust_to_gas2 = np.append(num_fast_clust_to_gas2, 0)

            if len(gas2_now_in_clust)>0:
                num_gas2_to_clust = np.append(num_gas2_to_clust, len(gas2_now_in_clust[0]))
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, len(np.where(typ[gas2_now_in_clust[0].astype('int')]==0)[0]))
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, len(np.where(typ[gas2_now_in_clust[0].astype('int')]==1)[0]))
            else:
                num_gas2_to_clust = np.append(num_gas2_to_clust, 0)
                num_slow_gas2_to_clust = np.append(num_slow_gas2_to_clust, 0)
                num_fast_gas2_to_clust = np.append(num_fast_gas2_to_clust, 0)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, len(bulk_now_in_gas_no_int[0]))
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, len(np.where(typ[bulk_now_in_gas_no_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, len(np.where(typ[bulk_now_in_gas_no_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_gas_no_int = np.append(num_bulk_to_gas_no_int, 0)
                num_slow_bulk_to_gas_no_int = np.append(num_slow_bulk_to_gas_no_int, 0)
                num_fast_bulk_to_gas_no_int = np.append(num_fast_bulk_to_gas_no_int, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, len(gas_now_in_bulk_no_int[0]))
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, len(np.where(typ[gas_now_in_bulk_no_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, len(np.where(typ[gas_now_in_bulk_no_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk_no_int = np.append(num_gas_to_bulk_no_int, 0)
                num_slow_gas_to_bulk_no_int = np.append(num_slow_gas_to_bulk_no_int, 0)
                num_fast_gas_to_bulk_no_int = np.append(num_fast_gas_to_bulk_no_int, 0)

            if len(bulk_now_in_int)>0:
                num_bulk_to_int = np.append(num_bulk_to_int, len(bulk_now_in_int[0]))

                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, len(np.where(typ[bulk_now_in_int[0].astype('int')]==0)[0]))
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, len(np.where(typ[bulk_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_int = np.append(num_bulk_to_int, 0)
                num_slow_bulk_to_int = np.append(num_slow_bulk_to_int, 0)
                num_fast_bulk_to_int = np.append(num_fast_bulk_to_int, 0)

            if len(int_now_in_bulk)>0:
                num_int_to_bulk = np.append(num_int_to_bulk, len(int_now_in_bulk[0]))
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, len(np.where(typ[int_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, len(np.where(typ[int_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_int_to_bulk = np.append(num_int_to_bulk, 0)
                num_slow_int_to_bulk = np.append(num_slow_int_to_bulk, 0)
                num_fast_int_to_bulk = np.append(num_fast_int_to_bulk, 0)

            if len(gas_now_in_int)>0:
                num_gas_to_int = np.append(num_gas_to_int, len(gas_now_in_int[0]))
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, len(np.where(typ[gas_now_in_int[0].astype('int')]==0)[0]))
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, len(np.where(typ[gas_now_in_int[0].astype('int')]==1)[0]))
            else:
                num_gas_to_int = np.append(num_gas_to_int, 0)
                num_slow_gas_to_int = np.append(num_slow_gas_to_int, 0)
                num_fast_gas_to_int = np.append(num_fast_gas_to_int, 0)

            if len(int_now_in_gas)>0:
                num_int_to_gas = np.append(num_int_to_gas, len(int_now_in_gas[0]))
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, len(np.where(typ[int_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, len(np.where(typ[int_now_in_gas[0].astype('int')]==1)[0]))
            else:
                num_int_to_gas = np.append(num_int_to_gas, 0)
                num_slow_int_to_gas = np.append(num_slow_int_to_gas, 0)
                num_fast_int_to_gas = np.append(num_fast_int_to_gas, 0)

            if len(gas_now_in_bulk)>0:
                num_gas_to_bulk = np.append(num_gas_to_bulk, len(gas_now_in_bulk[0]))
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, len(np.where(typ[gas_now_in_bulk[0].astype('int')]==0)[0]))
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, len(np.where(typ[gas_now_in_bulk[0].astype('int')]==1)[0]))
            else:
                num_gas_to_bulk = np.append(num_gas_to_bulk, 0)
                num_slow_gas_to_bulk = np.append(num_slow_gas_to_bulk, 0)
                num_fast_gas_to_bulk = np.append(num_fast_gas_to_bulk, 0)

            if len(bulk_now_in_gas)>0:
                num_bulk_to_gas = np.append(num_bulk_to_gas, len(bulk_now_in_gas[0]))
                num_slow_bulk_to_gas = np.append(num_slow_bulk_to_gas, len(np.where(typ[bulk_now_in_gas[0].astype('int')]==0)[0]))
                num_fast_bulk_to_gas = np.append(num_fast_bulk_to_gas, len(np.where(typ[bulk_now_in_gas[0].astype('int')]==1)[0]))
            else:
                num_bulk_to_gas = np.append(num_bulk_to_gas, 0)
                num_slow_bulk_to_gas = np.append(num_slow_bulk_to_gas, 0)
                num_fast_bulk_to_gas = np.append(num_fast_bulk_to_gas, 0)


            now_in_bulk_comb = np.array([])
            now_in_gas_comb = np.array([])
            now_in_int_comb = np.array([])

            no_flux_bulk = 0

            if (len(bulk_now_in_int)>0) & (len(bulk_now_in_gas)>0):
                now_in_int_comb = np.append(now_in_int_comb, bulk_now_in_int[0])
                now_in_gas_comb = np.append(now_in_gas_comb, bulk_now_in_gas[0])
            elif (len(bulk_now_in_int)>0) & (len(bulk_now_in_gas)==0):
                now_in_int_comb = np.append(now_in_int_comb, bulk_now_in_int[0])
            elif (len(bulk_now_in_int)==0) & (len(bulk_now_in_gas)>0):
                now_in_gas_comb = np.append(now_in_gas_comb, bulk_now_in_gas[0])
            else:
                no_flux_bulk = 1


            no_flux_gas = 0

            if (len(gas_now_in_int)>0) & (len(gas_now_in_bulk)>0):
                now_in_int_comb = np.append(now_in_int_comb, gas_now_in_int[0])
                now_in_bulk_comb = np.append(now_in_bulk_comb, gas_now_in_bulk[0])
            elif (len(gas_now_in_int)>0) & (len(gas_now_in_bulk)==0):
                now_in_int_comb = np.append(now_in_int_comb, gas_now_in_int[0])
            elif (len(gas_now_in_int)==0) & (len(gas_now_in_bulk)>0):
                now_in_bulk_comb = np.append(now_in_bulk_comb, gas_now_in_bulk[0])
            else:
                no_flux_gas = 1


            no_flux_int = 0

            if (len(int_now_in_gas)>0) & (len(int_now_in_bulk)>0):
                now_in_gas_comb = np.append(now_in_gas_comb, int_now_in_gas[0])
                now_in_bulk_comb = np.append(now_in_bulk_comb, int_now_in_bulk[0])
            elif (len(int_now_in_gas)>0) & (len(int_now_in_bulk)==0):
                now_in_gas_comb = np.append(now_in_gas_comb, int_now_in_gas[0])
            elif (len(int_now_in_gas)==0) & (len(int_now_in_bulk)>0):
                now_in_bulk_comb = np.append(now_in_bulk_comb, int_now_in_bulk[0])
            else:
                no_flux_int = 1

            if no_flux_bulk == 0:
                start_bulk_id_with_int = np.delete(start_bulk_id_with_int, not_in_bulk_comb)

            if no_flux_gas == 0:
                start_gas_id_with_int = np.delete(start_gas_id_with_int, not_in_gas_comb)

            if no_flux_int == 0:
                start_int_id_with_int = np.delete(start_int_id_with_int, not_in_int_comb)


            start_bulk_id = np.delete(start_bulk_id, not_in_bulk_ids_to_gas_no_int[1])
            start_gas_id = np.delete(start_gas_id, not_in_gas_ids_to_bulk_no_int[1])

            start_gas_id = np.append(start_gas_id, bulk_now_in_gas_no_int[0])
            start_bulk_id = np.append(start_bulk_id, gas_now_in_bulk_no_int[0])

            start_clust_id = np.delete(start_clust_id, not_in_clust_ids[1])
            start_gas2_id = np.delete(start_gas2_id, not_in_gas2_ids[1])

            start_clust_id = np.append(start_clust_id, gas2_now_in_clust[0])
            start_gas2_id = np.append(start_gas2_id, clust_now_in_gas2[0])


            start_int_id_with_int = np.append(start_int_id_with_int, now_in_int_comb)
            start_gas_id_with_int = np.append(start_gas_id_with_int, now_in_gas_comb)
            start_bulk_id_with_int = np.append(start_bulk_id_with_int, now_in_bulk_comb)
    """
    def collision_rate(self, vel_plot_dict, prev_neigh_dict):
        '''
        Purpose: Calculates the rate of collision between particles in the gas phase
        
        Input:
        vel_plot_dict: dictionary containing velocities of particles from previous time step to current

        prev_neigh_dict: dictionary containing neighbor information from previous time step

        Output:

        collision_stat_dict: dictionary containing the mean, median, mode, and standard deviation of number of colisions
        in gas for each respective type ('all', 'A', or 'B')

        collision_plot_dict: dictionary containing the number of colisions
        in gas for each respective type ('all', 'A', or 'B') for all particles

        neigh_dict: dictionary containing neighbor lists of each respective type in gas ('all', 'A', or 'B')
        '''

        # Dictionary containing statistics of rates of collisions
        if len(BB_rel_vel_stay)==0:
            collision_stat_dict = {'rel_vel_stay': {'AA': 0, 'AB': 0, 'BA': 0, 'BB': 0}, 'rel_vel_new':{'AA': 0, 'AB': 0, 'BA': 0, 'BB': 0}, 'rel_vel_all':{'AA': np.mean(AA_rel_vel_all), 'AB': np.mean(AB_rel_vel_all), 'BA': np.mean(BA_rel_vel_all), 'BB': np.mean(BB_rel_vel_all)}, 'num':{'AA': AA_collision_num, 'AB': AB_collision_num, 'BA': BA_collision_num, 'BB': BB_collision_num}, 'new_num':{'AA': AA_collision_num_unique, 'AB': AB_collision_num_unique, 'BA': BA_collision_num_unique, 'BB': BB_collision_num_unique}, 'size': {'all': {'large': clust_all_large, 'mean': clust_all_mean, 'mode': clust_all_mode, 'median': clust_all_median, 'std': clust_all_std}, 'A': {'large': clust_A_large, 'mean': clust_A_mean, 'mode': clust_A_mode, 'median': clust_A_median, 'std': clust_A_std}, 'B': {'large': clust_B_large, 'mean': clust_B_mean, 'mode': clust_B_mode, 'median': clust_B_median, 'std': clust_B_std}}}
        else:
            collision_stat_dict = {'rel_vel_stay': {'AA': np.mean(AA_rel_vel_stay), 'AB': np.mean(AB_rel_vel_stay), 'BA': np.mean(BA_rel_vel_stay), 'BB': np.mean(BB_rel_vel_stay)}, 'rel_vel_new':{'AA': np.mean(AA_rel_vel_new), 'AB': np.mean(AB_rel_vel_new), 'BA': np.mean(BA_rel_vel_new), 'BB': np.mean(BB_rel_vel_new)}, 'rel_vel_all':{'AA': np.mean(AA_rel_vel_all), 'AB': np.mean(AB_rel_vel_all), 'BA': np.mean(BA_rel_vel_all), 'BB': np.mean(BB_rel_vel_all)}, 'num':{'AA': AA_collision_num, 'AB': AB_collision_num, 'BA': BA_collision_num, 'BB': BB_collision_num}, 'new_num':{'AA': AA_collision_num_unique, 'AB': AB_collision_num_unique, 'BA': BA_collision_num_unique, 'BB': BB_collision_num_unique}, 'size': {'all': {'large': clust_all_large, 'mean': clust_all_mean, 'mode': clust_all_mode, 'median': clust_all_median, 'std': clust_all_std}, 'A': {'large': clust_A_large, 'mean': clust_A_mean, 'mode': clust_A_mode, 'median': clust_A_median, 'std': clust_A_std}, 'B': {'large': clust_B_large, 'mean': clust_B_mean, 'mode': clust_B_mode, 'median': clust_B_median, 'std': clust_B_std}}}
        
        # Dictionary containing cluster sizes
        collision_plot_dict = {'all': clust_all_size, 'A': clust_A_size, 'B': clust_B_size}
        
        # Dictionary containing neighbor lists
        neigh_dict = {'AA': AA_nlist, 'AB': AB_nlist, 'BA': BA_nlist, 'BB': BB_nlist}

        from statistics import mode

        # Find fast and slow particle IDs
        slow_ids = np.where( (self.typ==0) )[0]
        fast_ids = np.where( (self.typ==1) )[0]

        vel_A = vel_plot_dict['A']                             # Velocities of slow particles
        pos_A=self.pos[slow_ids]                               # Find positions of slow particles

        vel_B = vel_plot_dict['B']                             # Velocities of fast particles
        pos_B=self.pos[fast_ids]                               # Find positions of fast particles

        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max=self.r_cut)

        #Define reference particle list of all, slow (A), and fast (B) particles
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos[fast_ids]))
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos[slow_ids]))

        # Find Fast-Fast neighbor list
        BB_nlist = system_B.query(self.f_box.wrap(self.pos[fast_ids]), query_args).toNeighborList()

        # Find Fast-fast number of collisions
        BB_collision_num = len(BB_nlist) / 2.0

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_neigh_ind = np.array([], dtype=int)
        BB_num_neigh = np.array([])
        BB_dot = np.array([])
        BB_rel_vel_all = np.array([])
        BB_rel_vel_new = np.array([])
        BB_rel_vel_stay = np.array([])
        
        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in BB_nlist.query_point_indices:

                # If particle hadn't been considered before...
                if i not in BB_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_nlist.query_point_indices==i)[0]
                    
                    try:

                        neigh_num_temp = 0

                        # Find where reference particle in previous neighbor dictionary
                        loc_prev = np.where(prev_neigh_dict['BB'].query_point_indices==i)[0]

                        # Loop over neighboring particles
                        for j in range(0, len(loc)):

                            # Find where neighbor in previous neighbor dictionary
                            prev_neigh = np.where( prev_neigh_dict['BB'].point_indices[loc_prev]==BB_nlist.point_indices[loc[j]])[0]

                            # If not a previous neighbor
                            if len(prev_neigh)==0:

                                # Add to number of neighbors
                                neigh_num_temp += 1

                                # Calculate relative velocity
                                if (vel_B['mag'][i]>0) & ( vel_B['mag'][BB_nlist.point_indices[loc[j]]]>0):
                                    ang_rel = (vel_B['x'][i] * vel_B['x'][BB_nlist.point_indices[loc[j]]] + vel_B['y'][i] * vel_B['y'][BB_nlist.point_indices[loc[j]]])/(vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of new neighbors
                                BB_rel_vel_new = np.append(BB_rel_vel_new, np.sqrt(vel_B['mag'][i]**2 + vel_B['mag'][BB_nlist.point_indices[loc[j]]]**2 - 2 * vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc[j]]]*ang_rel) )

                            else:

                                # Calculate relative velocity
                                if (vel_B['mag'][i]>0) & ( vel_B['mag'][BB_nlist.point_indices[loc[j]]]>0):
                                    ang_rel = (vel_B['x'][i] * vel_B['x'][BB_nlist.point_indices[loc[j]]] + vel_B['y'][i] * vel_B['y'][BB_nlist.point_indices[loc[j]]])/(vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of previous neighbors
                                BB_rel_vel_stay = np.append(BB_rel_vel_stay, np.sqrt(vel_B['mag'][i]**2 + vel_B['mag'][BB_nlist.point_indices[loc[j]]]**2 - 2 * vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc[j]]]*ang_rel) )
                        
                        # Save number of current neighbors
                        BB_num_neigh = np.append(BB_num_neigh, neigh_num_temp)

                    except:

                        #Save number of current neighbors
                        BB_num_neigh = np.append(BB_num_neigh, len(loc))

                    # Save reference particle ID
                    BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                    
                    # Calculate dot product of orientations of neighboring particles
                    BB_dot = np.append(BB_dot, np.sum((px_B[i]*px_B[BB_nlist.point_indices[loc]]+py_A[i]*py_B[BB_nlist.point_indices[loc]])/(((px_B[i]**2+py_B[i]**2)**0.5)*((px_B[BB_nlist.point_indices[loc]]**2+py_B[BB_nlist.point_indices[loc]]**2)**0.5))))

                    # Calculate relative velocity
                    ang_rel = (vel_B['x'][i] * vel_B['x'][BB_nlist.point_indices[loc]] + vel_B['y'][i] * vel_B['y'][BB_nlist.point_indices[loc]])/(vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc]])
                    
                    # Calculate relative velocity of both current and previous neighbors
                    BB_rel_vel_all = np.append(BB_rel_vel_all, np.sqrt(vel_B['mag'][i]**2 + vel_B['mag'][BB_nlist.point_indices[loc]]**2 - 2 * vel_B['mag'][i] * vel_B['mag'][BB_nlist.point_indices[loc]]*ang_rel) )

            else:
                #Save nearest neighbor information to array
                BB_num_neigh = np.append(BB_num_neigh, 0)
                BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                BB_dot = np.append(BB_dot, 0)
        
        # Calculate number of unique collisions between two fast particles
        BB_collision_num_unique = np.sum(BB_num_neigh)/2

        # Find Slow-Fast neighbor list
        AB_nlist = system_A.query(self.f_box.wrap(self.pos[fast_ids]), query_args).toNeighborList()

        # Find Slow-fast number of collisions
        AB_collision_num = len(AB_nlist)

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_neigh_ind = np.array([], dtype=int)
        AB_num_neigh = np.array([])
        AB_dot = np.array([])
        AB_rel_vel_all = np.array([])
        AB_rel_vel_new = np.array([])
        AB_rel_vel_stay = np.array([])
        
        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in AB_nlist.query_point_indices:

                # If particle hadn't been considered before...
                if i not in AB_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_nlist.query_point_indices==i)[0]
                    
                    try:

                        neigh_num_temp = 0

                        # Find where reference particle in previous neighbor dictionary
                        loc_prev = np.where(prev_neigh_dict['AB'].query_point_indices==i)[0]

                        # Loop over neighboring particles
                        for j in range(0, len(loc)):

                            # Find where neighbor in previous neighbor dictionary
                            prev_neigh = np.where( prev_neigh_dict['AB'].point_indices[loc_prev]==AB_nlist.point_indices[loc[j]])[0]

                            # If not a previous neighbor
                            if len(prev_neigh)==0:

                                # Add to number of neighbors
                                neigh_num_temp += 1

                                # Calculate relative velocity
                                if (vel_B['mag'][i]>0) & ( vel_A['mag'][AB_nlist.point_indices[loc[j]]]>0):
                                    ang_rel = (vel_B['x'][i] * vel_A['x'][AB_nlist.point_indices[loc[j]]] + vel_B['y'][i] * vel_A['y'][AB_nlist.point_indices[loc[j]]])/(vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of new neighbors
                                AB_rel_vel_new = np.append(AB_rel_vel_new, np.sqrt(vel_B['mag'][i]**2 + vel_A['mag'][AB_nlist.point_indices[loc[j]]]**2 - 2 * vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc[j]]]*ang_rel) )
                        
                            else:

                                # Calculate relative velocity
                                if (vel_B['mag'][i]>0) & ( vel_A['mag'][AB_nlist.point_indices[loc[j]]]>0):
                                    ang_rel = (vel_B['x'][i] * vel_A['x'][AB_nlist.point_indices[loc[j]]] + vel_B['y'][i] * vel_A['y'][AB_nlist.point_indices[loc[j]]])/(vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of previous neighbors
                                AB_rel_vel_stay = np.append(AB_rel_vel_stay, np.sqrt(vel_B['mag'][i]**2 + vel_A['mag'][AB_nlist.point_indices[loc[j]]]**2 - 2 * vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc[j]]]*ang_rel) )
                        
                        # Save number of current neighbors
                        AB_num_neigh = np.append(AB_num_neigh, neigh_num_temp)

                    except:

                        #Save nearest neighbor information to array
                        AB_num_neigh = np.append(AB_num_neigh, len(loc))

                    # Save reference particle ID
                    AB_neigh_ind = np.append(AB_neigh_ind, int(i))

                    # Calculate dot product of orientations of neighboring particles
                    AB_dot = np.append(AB_dot, np.sum((px_B[i]*px_A[AB_nlist.point_indices[loc]]+py_B[i]*py_A[AB_nlist.point_indices[loc]])/(((px_B[i]**2+py_B[i]**2)**0.5)*((px_A[AB_nlist.point_indices[loc]]**2+py_A[AB_nlist.point_indices[loc]]**2)**0.5))))

                    # Calculate relative velocity
                    ang_rel = (vel_B['x'][i] * vel_A['x'][AB_nlist.point_indices[loc]] + vel_B['y'][i] * vel_A['y'][AB_nlist.point_indices[loc]])/(vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc]])

                    # Calculate relative velocity of both current and previous neighbors
                    AB_rel_vel_all = np.append(AB_rel_vel_all, np.sqrt(vel_B['mag'][i]**2 + vel_A['mag'][AB_nlist.point_indices[loc]]**2 - 2 * vel_B['mag'][i] * vel_A['mag'][AB_nlist.point_indices[loc]]*ang_rel) )

            else:
                #Save nearest neighbor information to array
                AB_num_neigh = np.append(AB_num_neigh, 0)
                AB_neigh_ind = np.append(AB_neigh_ind, int(i))
                AB_dot = np.append(AB_dot, 0)
        
        # Calculate number of unique collisions between slow and fast particles
        AB_collision_num_unique = np.sum(AB_num_neigh)

        # Find Fast-Slow neighbor list
        BA_nlist = system_B.query(self.f_box.wrap(self.pos[slow_ids]), query_args).toNeighborList()

        # Find Fast-Slow number of collisions
        BA_collision_num = len(BA_nlist)

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_neigh_ind = np.array([], dtype=int)
        BA_num_neigh = np.array([])
        BA_dot = np.array([])
        BA_rel_vel_all = np.array([])
        BA_rel_vel_new = np.array([])
        BA_rel_vel_stay = np.array([])
        
        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in BA_nlist.query_point_indices:

                # If particle hadn't been considered before...
                if i not in BA_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_nlist.query_point_indices==i)[0]
                    
                    try:

                        neigh_num_temp = 0

                        # Find where reference particle in previous neighbor dictionary
                        loc_prev = np.where(prev_neigh_dict['BA'].query_point_indices==i)[0]

                        # Loop over neighboring particles
                        for j in range(0, len(loc)):

                            # Find where neighbor in previous neighbor dictionary
                            prev_neigh = np.where( prev_neigh_dict['BA'].point_indices[loc_prev]==BA_nlist.point_indices[loc[j]])[0]

                            # If not a previous neighbor
                            if len(prev_neigh)==0:

                                # Add to number of neighbors
                                neigh_num_temp += 1

                                # Calculate relative velocity
                                if (vel_A['mag'][i]>0) & ( vel_B['mag'][BA_nlist.point_indices[loc[j]]]>0):
                                    ang_rel = (vel_A['x'][i] * vel_B['x'][BA_nlist.point_indices[loc[j]]] + vel_A['y'][i] * vel_B['y'][BA_nlist.point_indices[loc[j]]])/(vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of new neighbors
                                BA_rel_vel_new = np.append(BA_rel_vel_new, np.sqrt(vel_A['mag'][i]**2 + vel_B['mag'][BA_nlist.point_indices[loc[j]]]**2 - 2 * vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc[j]]]*ang_rel) )

                            else:

                                # Calculate relative velocity
                                if (vel_A['mag'][i]>0) & ( vel_B['mag'][BA_nlist.point_indices[loc[j]]]>0):

                                    ang_rel = (vel_A['x'][i] * vel_B['x'][BA_nlist.point_indices[loc[j]]] + vel_A['y'][i] * vel_B['y'][BA_nlist.point_indices[loc[j]]])/(vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of previous neighbors
                                BA_rel_vel_stay = np.append(BA_rel_vel_stay, np.sqrt(vel_A['mag'][i]**2 + vel_B['mag'][BA_nlist.point_indices[loc[j]]]**2 - 2 * vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc[j]]]*ang_rel) )

                        # Save number of current neighbors
                        BA_num_neigh = np.append(BA_num_neigh, neigh_num_temp)

                    except:

                        #Save nearest neighbor information to array
                        BA_num_neigh = np.append(BA_num_neigh, len(loc))

                    # Calculate relative velocity
                    ang_rel = (vel_A['x'][i] * vel_B['x'][BA_nlist.point_indices[loc]] + vel_A['y'][i] * vel_B['y'][BA_nlist.point_indices[loc]])/(vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc]])

                    # Calculate relative velocity of both current and previous neighbors
                    BA_rel_vel_all = np.append(BA_rel_vel_all, np.sqrt(vel_A['mag'][i]**2 + vel_B['mag'][BA_nlist.point_indices[loc]]**2 - 2 * vel_A['mag'][i] * vel_B['mag'][BA_nlist.point_indices[loc]]*ang_rel) )

                    # Save reference particle ID
                    BA_neigh_ind = np.append(BA_neigh_ind, int(i))

                    # Calculate dot product of orientations of neighboring particles
                    BA_dot = np.append(BA_dot, np.sum((px_A[i]*px_B[BA_nlist.point_indices[loc]]+py_A[i]*py_B[BA_nlist.point_indices[loc]])/(((px_A[i]**2+py_A[i]**2)**0.5)*((px_B[BA_nlist.point_indices[loc]]**2+py_B[BA_nlist.point_indices[loc]]**2)**0.5))))

            else:
                #Save nearest neighbor information to array
                BA_num_neigh = np.append(BA_num_neigh, 0)
                BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                BA_dot = np.append(BA_dot, 0)
        
        # Calculate number of unique collisions between fast and slow particles
        BA_collision_num_unique = np.sum(BA_num_neigh)

        # Find Slow-Slow neighbor list
        AA_nlist = system_A.query(self.f_box.wrap(self.pos[slow_ids]), query_args).toNeighborList()

        # Find Slow-Slow number of collisions
        AA_collision_num = len(AA_nlist) / 2.0

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_neigh_ind = np.array([], dtype=int)
        AA_num_neigh = np.array([])
        AA_dot = np.array([])
        AA_rel_vel_all = np.array([])
        AA_rel_vel_new = np.array([])
        AA_rel_vel_stay = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in AA_nlist.query_point_indices:

                # If particle hadn't been considered before...
                if i not in AA_neigh_ind:

                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_nlist.query_point_indices==i)[0]
                    
                    try:

                        neigh_num_temp = 0

                        # Find where reference particle in previous neighbor dictionary
                        loc_prev = np.where(prev_neigh_dict['AA'].query_point_indices==i)[0]

                        # Loop over neighboring particles
                        for j in range(0, len(loc)):

                            # Find where neighbor in previous neighbor dictionary
                            prev_neigh = np.where( prev_neigh_dict['AA'].point_indices[loc_prev]==AA_nlist.point_indices[loc[j]])[0]

                            # If not a previous neighbor
                            if len(prev_neigh)==0:

                                # Add to number of neighbors
                                neigh_num_temp += 1

                                # Calculate relative velocity
                                if (vel_A['mag'][i]>0) & ( vel_A['mag'][AA_nlist.point_indices[loc[j]]]>0):

                                    ang_rel = (vel_A['x'][i] * vel_A['x'][AA_nlist.point_indices[loc[j]]] + vel_A['y'][i] * vel_A['y'][AA_nlist.point_indices[loc[j]]])/(vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of new neighbors
                                AA_rel_vel_new = np.append(AA_rel_vel_new, np.sqrt(vel_A['mag'][i]**2 + vel_A['mag'][AA_nlist.point_indices[loc[j]]]**2 - 2 * vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc[j]]]*ang_rel) )

                            else:

                                # Calculate relative velocity
                                if (vel_A['mag'][i]>0) & ( vel_A['mag'][AA_nlist.point_indices[loc[j]]]>0):

                                    ang_rel = (vel_A['x'][i] * vel_A['x'][AA_nlist.point_indices[loc[j]]] + vel_A['y'][i] * vel_A['y'][AA_nlist.point_indices[loc[j]]])/(vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc[j]]])
                                else:
                                    ang_rel = 0

                                # Save relative velocity of previous neighbors
                                AA_rel_vel_stay = np.append(AA_rel_vel_stay, np.sqrt(vel_A['mag'][i]**2 + vel_A['mag'][AA_nlist.point_indices[loc[j]]]**2 - 2 * vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc[j]]]*ang_rel) )

                        #Save nearest neighbor information to array
                        AA_num_neigh = np.append(AA_num_neigh, neigh_num_temp)

                    except:

                        #Save nearest neighbor information to array
                        AA_num_neigh = np.append(AA_num_neigh, len(loc))

                    # Save reference particle ID
                    AA_neigh_ind = np.append(AA_neigh_ind, int(i))

                    # Calculate dot product of orientations of neighboring particles
                    AA_dot = np.append(AA_dot, np.sum((px_A[i]*px_A[AA_nlist.point_indices[loc]]+py_A[i]*py_A[AA_nlist.point_indices[loc]])/(((px_A[i]**2+py_A[i]**2)**0.5)*((px_A[AA_nlist.point_indices[loc]]**2+py_A[AA_nlist.point_indices[loc]]**2)**0.5))))

                    # Calculate relative velocity
                    ang_rel = (vel_A['x'][i] * vel_A['x'][AA_nlist.point_indices[loc]] + vel_A['y'][i] * vel_A['y'][AA_nlist.point_indices[loc]])/(vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc]])

                    # Calculate relative velocity of both current and previous neighbors
                    AA_rel_vel_all = np.append(AA_rel_vel_all, np.sqrt(vel_A['mag'][i]**2 + vel_A['mag'][AA_nlist.point_indices[loc]]**2 - 2 * vel_A['mag'][i] * vel_A['mag'][AA_nlist.point_indices[loc]]*ang_rel) )

            else:
                #Save nearest neighbor information to array
                AA_num_neigh = np.append(AA_num_neigh, 0)
                AA_neigh_ind = np.append(AA_neigh_ind, int(i))
                AA_dot = np.append(AA_dot, 0)

        # Calculate number of unique collisions between slow particles
        AA_collision_num_unique = np.sum(AA_num_neigh)/2

        #Compute cluster parameters using neighbor list of all particles within LJ cut-off distance
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': self.r_cut})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_all_size = clp_all.sizes                                  # find cluster sizes

        #Compute cluster parameters using neighbor list of A particles within LJ cut-off distance
        cl_A=freud.cluster.Cluster()                              #Define cluster
        cl_A.compute(system_A, neighbors={'r_max': self.r_cut})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_A = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_A.cluster_idx                                    # get id of each cluster
        clp_A.compute(system_A, ids)                            # Calculate cluster properties given cluster IDs
        clust_A_size = clp_A.sizes                                  # find cluster sizes

        #Compute cluster parameters using neighbor list of A particles within LJ cut-off distance
        cl_B=freud.cluster.Cluster()                              #Define cluster
        cl_B.compute(system_B, neighbors={'r_max': self.r_cut})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_B = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_B.cluster_idx                                    # get id of each cluster
        clp_B.compute(system_B, ids)                            # Calculate cluster properties given cluster IDs
        clust_B_size = clp_B.sizes        

        # Define maximum cluster size of all, A, and B particles
        clust_all_large = np.amax(clust_all_size)
        clust_A_large = np.amax(clust_A_size)
        clust_B_large = np.amax(clust_B_size)

        # Find clusters of all, A, and B particles larger than 1 particle
        lcID_all = np.where(clust_all_size >= 1)[0]    
        lcID_A = np.where(clust_A_size >= 1)[0]    
        lcID_B = np.where(clust_B_size >= 1)[0]

        # Find mean size of clusters of all, A, and B particles larger than 1 particle
        clust_all_mean = np.mean(clust_all_size[lcID_all])
        clust_A_mean = np.mean(clust_A_size[lcID_A])
        clust_B_mean = np.mean(clust_B_size[lcID_B])

        # Find mode size of clusters of all, A, and B particles larger than 1 particle
        clust_all_mode = mode(clust_all_size[lcID_all])
        clust_A_mode = mode(clust_A_size[lcID_A])
        clust_B_mode = mode(clust_B_size[lcID_B])

        # Find standard deviation size of clusters of all, A, and B particles larger than 1 particle
        clust_all_std = np.std(clust_all_size[lcID_all])
        clust_A_std = np.std(clust_A_size[lcID_A])
        clust_B_std = np.std(clust_B_size[lcID_B])

        # Find median size of clusters of all, A, and B particles larger than 1 particle
        clust_all_median = np.median(clust_all_size[lcID_all])
        clust_A_median = np.median(clust_A_size[lcID_A])
        clust_B_median = np.median(clust_B_size[lcID_B])
        
        # Dictionary containing statistics of rates of collisions
        if len(BB_rel_vel_stay)==0:
            collision_stat_dict = {'rel_vel_stay': {'AA': 0, 'AB': 0, 'BA': 0, 'BB': 0}, 'rel_vel_new':{'AA': 0, 'AB': 0, 'BA': 0, 'BB': 0}, 'rel_vel_all':{'AA': np.mean(AA_rel_vel_all), 'AB': np.mean(AB_rel_vel_all), 'BA': np.mean(BA_rel_vel_all), 'BB': np.mean(BB_rel_vel_all)}, 'num':{'AA': AA_collision_num, 'AB': AB_collision_num, 'BA': BA_collision_num, 'BB': BB_collision_num}, 'new_num':{'AA': AA_collision_num_unique, 'AB': AB_collision_num_unique, 'BA': BA_collision_num_unique, 'BB': BB_collision_num_unique}, 'size': {'all': {'large': clust_all_large, 'mean': clust_all_mean, 'mode': clust_all_mode, 'median': clust_all_median, 'std': clust_all_std}, 'A': {'large': clust_A_large, 'mean': clust_A_mean, 'mode': clust_A_mode, 'median': clust_A_median, 'std': clust_A_std}, 'B': {'large': clust_B_large, 'mean': clust_B_mean, 'mode': clust_B_mode, 'median': clust_B_median, 'std': clust_B_std}}}
        else:
            collision_stat_dict = {'rel_vel_stay': {'AA': np.mean(AA_rel_vel_stay), 'AB': np.mean(AB_rel_vel_stay), 'BA': np.mean(BA_rel_vel_stay), 'BB': np.mean(BB_rel_vel_stay)}, 'rel_vel_new':{'AA': np.mean(AA_rel_vel_new), 'AB': np.mean(AB_rel_vel_new), 'BA': np.mean(BA_rel_vel_new), 'BB': np.mean(BB_rel_vel_new)}, 'rel_vel_all':{'AA': np.mean(AA_rel_vel_all), 'AB': np.mean(AB_rel_vel_all), 'BA': np.mean(BA_rel_vel_all), 'BB': np.mean(BB_rel_vel_all)}, 'num':{'AA': AA_collision_num, 'AB': AB_collision_num, 'BA': BA_collision_num, 'BB': BB_collision_num}, 'new_num':{'AA': AA_collision_num_unique, 'AB': AB_collision_num_unique, 'BA': BA_collision_num_unique, 'BB': BB_collision_num_unique}, 'size': {'all': {'large': clust_all_large, 'mean': clust_all_mean, 'mode': clust_all_mode, 'median': clust_all_median, 'std': clust_all_std}, 'A': {'large': clust_A_large, 'mean': clust_A_mean, 'mode': clust_A_mode, 'median': clust_A_median, 'std': clust_A_std}, 'B': {'large': clust_B_large, 'mean': clust_B_mean, 'mode': clust_B_mode, 'median': clust_B_median, 'std': clust_B_std}}}
        
        # Dictionary containing cluster sizes
        collision_plot_dict = {'all': clust_all_size, 'A': clust_A_size, 'B': clust_B_size}
        
        # Dictionary containing neighbor lists
        neigh_dict = {'AA': AA_nlist, 'AB': AB_nlist, 'BA': BA_nlist, 'BB': BB_nlist}

        return collision_stat_dict, collision_plot_dict, neigh_dict
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

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]

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
        AA_gas_num_neigh = np.zeros(len(pos_A))

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_gas_num_neigh = np.zeros(len(pos_B))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_gas_num_neigh = np.zeros(len(pos_A))

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_gas_num_neigh = np.zeros(len(pos_B))

        # Search distance for neighbors in local density calculation
        rad_dist = [0, self.r_cut, 2*self.r_cut, 3*self.r_cut, 4*self.r_cut, 5*self.r_cut, 10*self.r_cut, 20*self.r_cut]
        
        # Loop over search distances
        for j in range(1, len(rad_dist)):

            #Initiate empty arrays for tracking which particles have been analyzed when finding local neighbors
            AA_gas_neigh_ind = np.array([], dtype=int)
            AB_gas_neigh_ind = np.array([], dtype=int)
            BA_gas_neigh_ind = np.array([], dtype=int)
            BB_gas_neigh_ind = np.array([], dtype=int)

            #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
            AA_gas_neigh_ind = np.zeros(len(pos_A))

            #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
            AB_gas_neigh_ind = np.zeros(len(pos_B))

            #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
            BA_gas_neigh_ind = np.zeros(len(pos_A))

            #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
            BB_gas_neigh_ind = np.zeros(len(pos_B))

            # List of query arguments for neighbor list caculation
            query_args = dict(mode='ball', r_min = rad_dist[j-1]+0.001, r_max = rad_dist[j])#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

            # Locate potential neighbor particles by type in the dense phase
            system_A_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
            system_B_bulk = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))
            
            # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
            AA_gas_nlist = system_A_bulk.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
            AB_gas_nlist = system_A_bulk.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
            BA_gas_nlist = system_B_bulk.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
            BB_gas_nlist = system_B_bulk.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        
            #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A)):
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
            for i in range(0, len(pos_A)):
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
            for i in range(0, len(pos_B)):
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
            for i in range(0, len(pos_B)):
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
            if rad_dist[j]==20*self.r_cut:

                # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
                allall_gas_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                allall_gas_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for A dense phase reference particles with all nearest neighbors
                Aall_gas_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                Aall_gas_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for B dense phase reference particles with all nearest neighbors
                Ball_gas_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                Ball_gas_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with A nearest neighbors
                allA_gas_pos_x = pos_A[:,0]
                allA_gas_pos_y = pos_A[:,1]

                # Save neighbor, local orientational order, and position to arrays for all dense phase reference particles with B nearest neighbors
                allB_gas_pos_x = pos_B[:,0]
                allB_gas_pos_y = pos_B[:,1]

                # Create output dictionary for single particle values of local density per phase/activity pairing for plotting
                local_gas_dens_plot_dict = {'all-all': {'dens': allall_gas_local_dens, 'homo': allall_gas_local_dens_inhomog, 'pos_x': allall_gas_pos_x, 'pos_y': allall_gas_pos_y}, 'all-A': {'dens': allA_gas_local_dens, 'homo': allA_gas_local_dens_inhomog, 'pos_x': allA_gas_pos_x, 'pos_y': allA_gas_pos_y}, 'all-B': {'dens': allB_gas_local_dens, 'homo': allB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'A-all': {'dens': Aall_gas_local_dens, 'homo': Aall_gas_local_dens_inhomog, 'pos_x': Aall_gas_pos_x, 'pos_y': Aall_gas_pos_y}, 'B-all': {'dens': Ball_gas_local_dens, 'homo': Ball_gas_local_dens_inhomog, 'pos_x': Ball_gas_pos_x, 'pos_y': Ball_gas_pos_y}, 'A-A': {'dens': AA_gas_local_dens, 'homo': AA_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'A-B': {'dens': AB_gas_local_dens, 'homo': AB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}, 'B-A': {'dens': BA_gas_local_dens, 'homo': BA_gas_local_dens_inhomog, 'pos_x': allA_gas_pos_x, 'pos_y': allA_gas_pos_y}, 'B-B': {'dens': BB_gas_local_dens, 'homo': BB_gas_local_dens_inhomog, 'pos_x': allB_gas_pos_x, 'pos_y': allB_gas_pos_y}}

        # Create output dictionary for statistical averages of local density per phase/activity pairing
        local_gas_dens_stat_dict = {'radius': rad_dist[1:], 'allall_mean': allall_local_dens_mean_arr, 'allall_std': allall_local_dens_std_arr, 'allA_mean': allA_local_dens_mean_arr, 'allA_std': allA_local_dens_std_arr, 'allB_mean': allB_local_dens_mean_arr, 'allB_std': allB_local_dens_std_arr, 'AA_mean': AA_local_dens_mean_arr, 'AA_std': AA_local_dens_std_arr, 'AB_mean': AB_local_dens_mean_arr, 'AB_std': AB_local_dens_std_arr, 'BA_mean': BA_local_dens_mean_arr, 'BA_std': BA_local_dens_std_arr, 'BB_mean': BB_local_dens_mean_arr, 'BB_std': BB_local_dens_std_arr}
    
        return local_gas_dens_stat_dict, local_gas_dens_plot_dict
    
    def local_system_density(self):
        '''
        Purpose: Takes the composition of each phase and uses neighbor lists to find the
        nearest, interacting neighbors and calculates the local density for various search
        distances of each type for each particle and averaged over all particles in system.

        Outputs:
        local_dens_stat_dict: dictionary containing the local density for various search
        distances of each type ('all', 'A', or 'B') for a reference particle of a given 
        type ('all', 'A', or 'B'), averaged over all particles in system.

        local_homo_stat_dict: dictionary containing fluctuations in the local density from the
        steady-state mean for various search distances of each type ('all', 'A', or 'B') for a reference
        particle of a given type ('all', 'A', or 'B'), averaged over all particles in system
        '''
        
        # Calculate area of system
        system_area = self.lx_box * self.ly_box

        # Position and orientation arrays of type A particles in system
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles

        # Position and orientation arrays of type B particles in system
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around A reference particles
        allA_local_dens_mean_arr = []
        allA_local_dens_std_arr = []
        allA_inhomog_mean_arr = []
        allA_inhomog_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around B reference particles
        allB_local_dens_mean_arr = []
        allB_local_dens_std_arr = []
        allB_inhomog_mean_arr = []
        allB_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around all reference particles
        Aall_local_dens_mean_arr = []
        Aall_local_dens_std_arr = []
        Aall_inhomog_mean_arr = []
        Aall_inhomog_std_arr = []
        
        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around all reference particles
        Ball_local_dens_mean_arr = []
        Ball_local_dens_std_arr = []
        Ball_inhomog_mean_arr = []
        Ball_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of all neighbor particles around all reference particles
        allall_local_dens_mean_arr = []
        allall_local_dens_std_arr = []
        allall_inhomog_mean_arr = []
        allall_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around A reference particles
        AA_local_dens_mean_arr = []
        AA_local_dens_std_arr = []
        AA_inhomog_mean_arr = []
        AA_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of A neighbor particles around B reference particles
        AB_local_dens_mean_arr = []
        AB_local_dens_std_arr = []
        AB_inhomog_mean_arr = []
        AB_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around A reference particles
        BA_local_dens_mean_arr = []
        BA_local_dens_std_arr = []
        BA_inhomog_mean_arr = []
        BA_inhomog_std_arr = []

        #Initiate empty arrays for calculating mean and standard deviation of local density
        #of B neighbor particles around B reference particles
        BB_local_dens_mean_arr = []
        BB_local_dens_std_arr = []
        BB_inhomog_mean_arr = []
        BB_inhomog_std_arr = []

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type A particles
        AA_num_neigh = np.zeros(len(pos_A))

        #Initiate empty arrays for finding nearest A neighboring particles surrounding type B particles
        AB_num_neigh = np.zeros(len(pos_B))

        #Initiate empty arrays for finding nearest B neighboring particles surrounding type A particles
        BA_num_neigh = np.zeros(len(pos_A))

        #Initiate empty arrays for finding nearest B neighboring particles surrounding type B particles
        BB_num_neigh = np.zeros(len(pos_B))

        # Search distance for neighbors in local density calculation
        rad_dist = [0, self.r_cut, 2*self.r_cut, 3*self.r_cut, 4*self.r_cut, 5*self.r_cut]
        
        # Loop over search distances
        for j in range(1, len(rad_dist)):

            #Initiate empty arrays for finding nearest A neighboring particles surrounding type A particles
            AA_neigh_ind = np.zeros(len(pos_A))

            #Initiate empty arrays for finding nearest A neighboring particles surrounding type B particles
            AB_neigh_ind = np.zeros(len(pos_B))

            #Initiate empty arrays for finding nearest B neighboring particles surrounding type A particles
            BA_neigh_ind = np.zeros(len(pos_A))

            #Initiate empty arrays for finding nearest B neighboring particles surrounding type B particles
            BB_neigh_ind = np.zeros(len(pos_B))

            # List of query arguments for neighbor list caculation
            query_args = dict(mode='ball', r_min = rad_dist[j-1]+0.001, r_max = rad_dist[j])#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

            # Locate potential neighbor particles by type in the phase
            system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
            system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))
            
            # Generate neighbor list of phase particles (per query args) of respective type (A or B) neighboring phase reference particles of respective type (A or B)
            AA_nlist = system_A.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
            AB_nlist = system_A.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
            BA_nlist = system_B.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
            BB_nlist = system_B.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        
            #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A)):
                if i in AA_nlist.query_point_indices:
                    if i not in AA_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AA_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AA_num_neigh[i] += len(loc)
                        AA_neigh_ind = np.append(AA_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AA_neigh_ind = np.append(AA_neigh_ind, int(i))

            #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_A)):
                if i in BA_nlist.query_point_indices:
                    if i not in BA_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BA_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BA_num_neigh[i] += len(loc)
                        BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BA_neigh_ind = np.append(BA_neigh_ind, int(i))

            #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B)):
                if i in AB_nlist.query_point_indices:
                    if i not in AB_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(AB_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        AB_num_neigh[i] += len(loc)
                        AB_neigh_ind = np.append(AB_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        AB_neigh_ind = np.append(AB_neigh_ind, int(i))

            #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
            for i in range(0, len(pos_B)):
                if i in BB_nlist.query_point_indices:
                    if i not in BB_neigh_ind:
                        # Find neighbors list IDs where i is reference particle
                        loc = np.where(BB_nlist.query_point_indices==i)[0]

                        #Save nearest neighbor information to array
                        BB_num_neigh[i] += len(loc)
                        BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                    else:
                        #Save nearest neighbor information to array
                        BB_neigh_ind = np.append(BB_neigh_ind, int(i))

            slice_area = (np.pi*rad_dist[j]**2)
            # Local density of A neighbor particles around A reference particles
            AA_local_dens = AA_num_neigh / slice_area

            # Local density of A neighbor particles around B reference particles
            AB_local_dens = AB_num_neigh / slice_area

            # Local density of B neighbor particles around A reference particles
            BA_local_dens = BA_num_neigh / slice_area

            # Local density of B neighbor particles around B reference particles
            BB_local_dens = BB_num_neigh / slice_area
            
            # Save neighbor and local orientational order to arrays for all B reference particles with all nearest neighbors
            Ball_local_dens= np.append(BA_local_dens, BB_local_dens)

            # Save neighbor and local orientational order to arrays for all A reference particles with all nearest neighbors
            Aall_local_dens = np.append(AA_local_dens, AB_local_dens)

            # Save neighbor and local orientational order to arrays for all reference particles with B nearest neighbors
            allB_num_neigh = AB_num_neigh + BB_num_neigh
            allB_local_dens = allB_num_neigh / slice_area

            # Save neighbor and local orientational order to arrays for all reference particles with A nearest neighbors
            allA_num_neigh = AA_num_neigh + BA_num_neigh
            allA_local_dens = allA_num_neigh / slice_area

            # Save neighbor, local orientational order, and position to arrays for all reference particles with all nearest neighbors
            allall_num_neigh = np.append(allA_num_neigh, allB_num_neigh)
            allall_local_dens = allall_num_neigh / slice_area

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around A reference particles
            AA_local_dens_mean_arr.append(np.mean(AA_local_dens))
            AA_local_dens_std_arr.append(np.std(AA_local_dens))
            AA_local_dens_inhomog = (AA_local_dens - AA_local_dens_mean_arr[-1])**2
            AA_inhomog_mean_arr.append(np.mean(AA_local_dens_inhomog))
            AA_inhomog_std_arr.append(np.std(AA_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of A neighbor particles 
            # around B reference particles
            AB_local_dens_mean_arr.append(np.mean(AB_local_dens))
            AB_local_dens_std_arr.append(np.std(AB_local_dens))
            AB_local_dens_inhomog = (AB_local_dens - AB_local_dens_mean_arr[-1])**2
            AB_inhomog_mean_arr.append(np.mean(AB_local_dens_inhomog))
            AB_inhomog_std_arr.append(np.std(AB_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around A reference particles
            BA_local_dens_mean_arr.append(np.mean(BA_local_dens))
            BA_local_dens_std_arr.append(np.std(BA_local_dens))
            BA_local_dens_inhomog = (BA_local_dens - BA_local_dens_mean_arr[-1])**2
            BA_inhomog_mean_arr.append(np.mean(BA_local_dens_inhomog))
            BA_inhomog_std_arr.append(np.std(BA_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of B neighbor particles 
            # around B reference particles
            BB_local_dens_mean_arr.append(np.mean(BB_local_dens))
            BB_local_dens_std_arr.append(np.std(BB_local_dens))
            BB_local_dens_inhomog = (BB_local_dens - BB_local_dens_mean_arr[-1])**2
            BB_inhomog_mean_arr.append(np.mean(BB_local_dens_inhomog))
            BB_inhomog_std_arr.append(np.std(BB_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles
            allA_local_dens_mean_arr.append(np.mean(allA_local_dens))
            allA_local_dens_std_arr.append(np.std(allA_local_dens))
            allA_local_dens_inhomog = (allA_local_dens - allA_local_dens_mean_arr[-1])**2
            allA_inhomog_mean_arr.append(np.mean(allA_local_dens_inhomog))
            allA_inhomog_std_arr.append(np.std(allA_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles
            allB_local_dens_mean_arr.append(np.mean(allB_local_dens))
            allB_local_dens_std_arr.append(np.std(allB_local_dens))
            allB_local_dens_inhomog = (allB_local_dens - allB_local_dens_mean_arr[-1])**2
            allB_inhomog_mean_arr.append(np.mean(allB_local_dens_inhomog))
            allB_inhomog_std_arr.append(np.std(allB_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around A reference particles
            Aall_local_dens_mean_arr.append(np.mean(Aall_local_dens))
            Aall_local_dens_std_arr.append(np.std(Aall_local_dens))
            Aall_local_dens_inhomog = (Aall_local_dens - Aall_local_dens_mean_arr[-1])**2
            Aall_inhomog_mean_arr.append(np.mean(Aall_local_dens_inhomog))
            Aall_inhomog_std_arr.append(np.std(Aall_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around B reference particles
            Ball_local_dens_mean_arr.append(np.mean(Ball_local_dens))
            Ball_local_dens_std_arr.append(np.std(Ball_local_dens))
            Ball_local_dens_inhomog = (Ball_local_dens - Ball_local_dens_mean_arr[-1])**2
            Ball_inhomog_mean_arr.append(np.mean(Ball_local_dens_inhomog))
            Ball_inhomog_std_arr.append(np.std(Ball_local_dens_inhomog))

            # Calculate mean and standard deviation of local density of all neighbor particles 
            # around all reference particles
            allall_local_dens_mean_arr.append(np.mean(allall_local_dens))
            allall_local_dens_std_arr.append(np.std(allall_local_dens))
            allall_local_dens_inhomog = (allall_local_dens - allall_local_dens_mean_arr[-1])**2
            allall_inhomog_mean_arr.append(np.mean(allall_local_dens_inhomog))
            allall_inhomog_std_arr.append(np.std(allall_local_dens_inhomog))

            # If search distance given, then prepare data for plotting!
            if rad_dist[j]==2*self.r_cut:

                # Save neighbor, local orientational order, and position to arrays for all reference particles with all nearest neighbors
                allall_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                allall_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for A reference particles with all nearest neighbors
                Aall_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                Aall_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for B reference particles with all nearest neighbors
                Ball_pos_x = np.append(pos_A[:,0], pos_B[:,0])
                Ball_pos_y = np.append(pos_A[:,1], pos_B[:,1])

                # Save neighbor, local orientational order, and position to arrays for all reference particles with A nearest neighbors
                allA_pos_x = pos_A[:,0]
                allA_pos_y = pos_A[:,1]

                # Save neighbor, local orientational order, and position to arrays for all reference particles with B nearest neighbors
                allB_pos_x = pos_B[:,0]
                allB_pos_y = pos_B[:,1]

                # Create output dictionary for single particle values of local density per activity pairing for plotting
                local_dens_plot_dict = {'all-all': {'dens': allall_local_dens, 'homo': allall_local_dens_inhomog, 'pos_x': allall_pos_x, 'pos_y': allall_pos_y}, 'all-A': {'dens': allA_local_dens, 'homo': allA_local_dens_inhomog, 'pos_x': allA_pos_x, 'pos_y': allA_pos_y}, 'all-B': {'dens': allB_local_dens, 'homo': allB_local_dens_inhomog, 'pos_x': allB_pos_x, 'pos_y': allB_pos_y}, 'A-all': {'dens': Aall_local_dens, 'homo': Aall_local_dens_inhomog, 'pos_x': Aall_pos_x, 'pos_y': Aall_pos_y}, 'B-all': {'dens': Ball_local_dens, 'homo': Ball_local_dens_inhomog, 'pos_x': Ball_pos_x, 'pos_y': Ball_pos_y}, 'A-A': {'dens': AA_local_dens, 'homo': AA_local_dens_inhomog, 'pos_x': allB_pos_x, 'pos_y': allB_pos_y}, 'A-B': {'dens': AB_local_dens, 'homo': AB_local_dens_inhomog, 'pos_x': allB_pos_x, 'pos_y': allB_pos_y}, 'B-A': {'dens': BA_local_dens, 'homo': BA_local_dens_inhomog, 'pos_x': allA_pos_x, 'pos_y': allA_pos_y}, 'B-B': {'dens': BB_local_dens, 'homo': BB_local_dens_inhomog, 'pos_x': allB_pos_x, 'pos_y': allB_pos_y}}

        # Create output dictionary for statistical averages of local density per activity pairing
        local_dens_stat_dict = {'radius': rad_dist[1:], 'allall_mean': allall_local_dens_mean_arr, 'allall_std': allall_local_dens_std_arr, 'allA_mean': allA_local_dens_mean_arr, 'allA_std': allA_local_dens_std_arr, 'allB_mean': allB_local_dens_mean_arr, 'allB_std': allB_local_dens_std_arr, 'AA_mean': AA_local_dens_mean_arr, 'AA_std': AA_local_dens_std_arr, 'AB_mean': AB_local_dens_mean_arr, 'AB_std': AB_local_dens_std_arr, 'BA_mean': BA_local_dens_mean_arr, 'BA_std': BA_local_dens_std_arr, 'BB_mean': BB_local_dens_mean_arr, 'BB_std': BB_local_dens_std_arr}
        local_homo_stat_dict = {'radius': rad_dist[1:], 'allall_mean': allall_inhomog_mean_arr, 'allall_std': allall_inhomog_std_arr, 'allA_mean': allA_inhomog_mean_arr, 'allA_std': allA_inhomog_std_arr, 'allB_mean': allB_inhomog_mean_arr, 'allB_std': allB_inhomog_std_arr, 'AA_mean': AA_inhomog_mean_arr, 'AA_std': AA_inhomog_std_arr, 'AB_mean': AB_inhomog_mean_arr, 'AB_std': AB_inhomog_std_arr, 'BA_mean': BA_inhomog_mean_arr, 'BA_std': BA_inhomog_std_arr, 'BB_mean': BB_inhomog_mean_arr, 'BB_std': BB_inhomog_std_arr}

        return local_dens_stat_dict, local_homo_stat_dict, local_dens_plot_dict

    def cluster_msd(self, com_x_msd, com_y_msd, com_r_msd, com_x_parts_arr_time, com_y_parts_arr_time):
        
        '''
        Purpose: Calculates the mean squared displacement of the cluster

        Inputs:
        com_x_msd: total x displacement from initial time point

        com_y_msd: total y displacement from initial time point

        com_r_msd: total men squared displacement from initial time point

        com_x_parts_arr_time: com x-location over time

        com_y_parts_arr_time: com y-location over time

        Output:
        cluster_msd_dict: dictionary containing the total x- and y- displacement  and mean squared displacement 
        from initial time point
        '''

        # X-displacement between time points
        difx = com_x_parts_arr_time[-1] - com_x_parts_arr_time[-2]
        
        #Enforce periodic boundary conditions
        difx_abs = np.abs(difx)
        if difx_abs>=self.hx_box:
            if difx < -self.hx_box:
                difx += self.lx_box
            else:
                difx -= self.lx_box

        # X-displacement over time
        com_x_msd = np.append(com_x_msd, com_x_msd[-1] + difx)

        # Y-displacement between time points
        dify = com_y_parts_arr_time[-1] - com_y_parts_arr_time[-2]

        #Enforce periodic boundary conditions
        dify_abs = np.abs(dify)
        if dify_abs>=self.hy_box:
            if dify < -self.hy_box:
                dify += self.ly_box
            else:
                dify -= self.ly_box
        
        # Y-displacement over time
        com_y_msd = np.append(com_y_msd, com_y_msd[-1] + dify)

        # Total displacement between time points
        difr = (difx**2 + dify**2)**0.5

        # MSD over time
        com_r_msd = np.append(com_r_msd, (com_x_msd[-1] ** 2 + com_y_msd[-1] ** 2) ** 0.5 )

        # Dictionary containing MSD
        cluster_msd_dict = {'x': com_x_msd, 'y': com_y_msd, 'r': com_r_msd}

        return cluster_msd_dict

    def particle_msd(self, msd_arr, prev_pos):
        # IN PROGRESS
        '''
        Purpose: Calculates the mean squared displacement of individual particles

        Inputs:
        msd_arr: total men squared displacement from initial time point for each particle

        com_x_parts_arr_time: com x-location over time

        com_y_parts_arr_time: com y-location over time

        Output:
        cluster_msd_dict: dictionary containing the total x- and y- displacement  and mean squared displacement 
        from initial time point
        '''

        # X-displacement between time points
        difx = com_x_parts_arr_time[-1] - com_x_parts_arr_time[-2]
        
        #Enforce periodic boundary conditions
        difx_abs = np.abs(difx)
        if difx_abs>=self.hx_box:
            if difx < -self.hx_box:
                difx += self.lx_box
            else:
                difx -= self.lx_box

        # X-displacement over time
        com_x_msd = np.append(com_x_msd, com_x_msd[-1] + difx)

        # Y-displacement between time points
        dify = com_y_parts_arr_time[-1] - com_y_parts_arr_time[-2]

        #Enforce periodic boundary conditions
        dify_abs = np.abs(dify)
        if dify_abs>=self.hy_box:
            if dify < -self.hy_box:
                dify += self.ly_box
            else:
                dify -= self.ly_box
        
        # Y-displacement over time
        com_y_msd = np.append(com_y_msd, com_y_msd[-1] + dify)

        # Total displacement between time points
        difr = (difx**2 + dify**2)**0.5

        # MSD over time
        com_r_msd = np.append(com_r_msd, (com_x_msd[-1] ** 2 + com_y_msd[-1] ** 2) ** 0.5 )

        # Dictionary containing MSD
        cluster_msd_dict = {'x': com_x_msd, 'y': com_y_msd, 'r': com_r_msd}

        return cluster_msd_dict

    
    """
    def radial_int_press_bubble(self, stress_plot_dict, sep_surface_dict, int_comp_dict, all_surface_measurements):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from largest cluster's CoM

        stress_plot_dict: dictionary (output from various interparticle_pressure_nlist() in
        measurement.py) containing information on the stress and positions of all,
        type A, and type B particles.

        sep_surface_dict: dictionary (output from surface_curve_interp() in
        interface.py) that contains the interpolated curve representing the
        inner and outer surfaces of each interface.

        int_comp_dict: dictionary (output from int_sort2() in
        phase_identification.py) that contains information on each
        isolated/individual interface.

        all_surface_measurements: dictionary that contains information on measured properties
        of each identified interface surface

        Output:
        radial_stress_dict: dictionary containing each particle's stress in each direction
        as a function of separation distance from largest custer's CoM
        '''

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))

            try:
                
                pos_exterior_surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                pos_exterior_surface_y = sep_surface_dict[key]['exterior']['pos']['y']

                com_x = np.mean(pos_exterior_surface_x)
                com_y = np.mean(pos_exterior_surface_y)
            except:
                pos_interior_surface_x = sep_surface_dict[key]['interior']['pos']['x']
                pos_interior_surface_y = sep_surface_dict[key]['interior']['pos']['y']

                com_x = np.mean(pos_interior_surface_x)
                com_y = np.mean(pos_interior_surface_y)

        # Stress in all directions of all particles in order of dense first then gas particles
        stress_xx = np.append(stress_plot_dict['dense']['all-all']['XX'], stress_plot_dict['gas']['all-all']['XX'])
        stress_yy = np.append(stress_plot_dict['dense']['all-all']['YY'], stress_plot_dict['gas']['all-all']['YY'])
        stress_xy = np.append(stress_plot_dict['dense']['all-all']['XY'], stress_plot_dict['gas']['all-all']['XY'])
        stress_yx = np.append(stress_plot_dict['dense']['all-all']['YX'], stress_plot_dict['gas']['all-all']['YX'])

        # All particle positions in same order as above
        pos_x = np.append(stress_plot_dict['dense']['pos']['all']['x'], stress_plot_dict['gas']['pos']['all']['x'])
        pos_y = np.append(stress_plot_dict['dense']['pos']['all']['y'], stress_plot_dict['gas']['pos']['all']['y'])
        
        # All particle types in same order as above
        typ = np.append(stress_plot_dict['dense']['typ'], stress_plot_dict['gas']['typ'])

        # Instantiate empty array (partNum) containing the stress in each direction that particles experience
        stress_xx_arr = np.array([])
        stress_yy_arr = np.array([])
        stress_xy_arr = np.array([])
        stress_yx_arr = np.array([])

        stress_xx_A_arr = np.array([])
        stress_yy_A_arr = np.array([])
        stress_xy_A_arr = np.array([])
        stress_yx_A_arr = np.array([])

        stress_xx_B_arr = np.array([])
        stress_yy_B_arr = np.array([])
        stress_xy_B_arr = np.array([])
        stress_yx_B_arr = np.array([])

        # Instantiate empty array (partNum) containing the distance from largest cluster's CoM
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])
        
        # Loop over all particles
        for h in range(0, len(pos_x)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = pos_x[h] - (com_x-self.hx_box)
            dify = pos_y[h] - (com_y-self.hy_box)
            
            difr= ( (difx )**2 + (dify)**2)**0.5

            if typ[h] == 0:

                # Save stress in each direction for A particles
                stress_xx_A_arr =np.append(stress_xx_A_arr, stress_xx[h])
                stress_yy_A_arr =np.append(stress_yy_A_arr, stress_yy[h])
                stress_xy_A_arr =np.append(stress_xy_A_arr, stress_xy[h])
                stress_yx_A_arr =np.append(stress_yx_A_arr, stress_yx[h])

                # Save separation distance from largest cluster's CoM for A particles
                rA_dist_norm = np.append(rA_dist_norm, difr)
            else:

                # Save stress in each direction for B particles
                stress_xx_B_arr =np.append(stress_xx_B_arr, stress_xx[h])
                stress_yy_B_arr =np.append(stress_yy_B_arr, stress_yy[h])
                stress_xy_B_arr =np.append(stress_xy_B_arr, stress_xy[h])
                stress_yx_B_arr =np.append(stress_yx_B_arr, stress_yx[h])

                # Save separation distance from largest cluster's CoM for B particles
                rB_dist_norm = np.append(rB_dist_norm, difr)
            
            # Save stress in each direction for all particles
            stress_xx_arr =np.append(stress_xx_arr, stress_xx[h])
            stress_yy_arr =np.append(stress_yy_arr, stress_yy[h])
            stress_xy_arr =np.append(stress_xy_arr, stress_xy[h])
            stress_yx_arr =np.append(stress_yx_arr, stress_yx[h])

            
            # Save separation distance from largest cluster's CoM
            r_dist_norm = np.append(r_dist_norm, difr)

        # Dictionary containing each particle's stress in each direction
        # as a function of separation distance from largest custer's CoM

        radial_stress_dict = {'all': {'XX': stress_xx_arr, 'YY': stress_yy_arr, 'XY': stress_xy_arr, 'YX': stress_yx_arr, 'r': r_dist_norm}, 'A': {'XX': stress_xx_A_arr, 'YY': stress_yy_A_arr, 'XY': stress_xy_A_arr, 'YX': stress_yx_A_arr, 'r': rA_dist_norm}, 'B': {'XX': stress_xx_B_arr, 'YY': stress_yy_B_arr, 'XY': stress_xy_B_arr, 'YX': stress_yx_B_arr, 'r': rB_dist_norm}, 'typ': typ}

        return radial_stress_dict
    """
    
    
    """
    def radial_surface_normal_fa_bubble(self, method2_align_dict, sep_surface_dict, int_comp_dict, all_surface_measurements):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from the nearest interface surface normal

        Input:
        method2_align_dict: dictionary containing the alignment
        of each particle's active force with the nearest interface's surface normal,
        per and averaged per bin
        
        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force with
        the nearest interface surface normal as a function of separation distance from largest custer's CoM
        '''

        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            try: 
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty array (partNum) containing the average active force magnitude
        # toward the nearest interface surface normal
        fa_norm = np.array([])
        faA_norm = np.array([])
        faB_norm = np.array([])

        # Instantiate empty array (partNum) containing the distance from the nearest interface surface
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])

        align_norm = np.array([])
        alignA_norm = np.array([])
        alignB_norm = np.array([])

        # Loop over all particles
        for h in range(0, len(self.pos)):

            # Separation distance from largest custer's CoM (middle of box)
            difx = self.pos[h,0] - (com_x-self.hx_box)
            dify = self.pos[h,1] - (com_y-self.hy_box)

            difr= ( (difx )**2 + (dify)**2)**0.5

            # Save active force magnitude toward the nearest interface surface normal
            if self.typ[h] == 0:
                fa_norm=np.append(fa_norm, part_align[h]*self.peA)
                faA_norm=np.append(faA_norm, part_align[h]*self.peA)
                rA_dist_norm = np.append(rA_dist_norm, difr)
                alignA_norm=np.append(alignA_norm, part_align[h])
            else:
                fa_norm=np.append(fa_norm, part_align[h]*self.peB)
                faB_norm=np.append(faB_norm, part_align[h]*self.peB)
                alignB_norm=np.append(alignB_norm, part_align[h])
                rB_dist_norm = np.append(rB_dist_norm, difr)

            # Save separation distance from the nearest interface surface
            r_dist_norm = np.append(r_dist_norm, difr)
            align_norm=np.append(align_norm, part_align[h])

        # Dictionary containing each particle's alignment and aligned active force toward
        # the nearest interface surface normal as a function of separation distance from
        # largest custer's CoM
        radial_fa_dict = {'all': {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}, 'A': {'r': rA_dist_norm, 'fa': faA_norm, 'align': alignA_norm}, 'B': {'r': rB_dist_norm, 'fa': faB_norm, 'align': alignB_norm}}

        return radial_fa_dict
    """ 
    def radial_surface_normal_fa_bubble2(self, method2_align_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, int_dict):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from the nearest interface surface normal

        Input:
        method2_align_dict: dictionary containing the alignment
        of each particle's active force with the nearest interface's surface normal,
        per and averaged per bin
        
        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force with
        the nearest interface surface normal as a function of separation distance from largest custer's CoM
        '''
        radial_fa_dict = {}
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            try: 
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']
        
            # Instantiate empty array (partNum) containing the average active force alignment
            # with the nearest interface surface normal
            part_align = method2_align_dict['part']['align']

            # Instantiate empty array (partNum) containing the average active force magnitude
            # toward the nearest interface surface normal
            fa_norm = np.array([])
            faA_norm = np.array([])
            faB_norm = np.array([])

            # Instantiate empty array (partNum) containing the distance from the nearest interface surface
            r_dist_norm = np.array([])
            rA_dist_norm = np.array([])
            rB_dist_norm = np.array([])

            align_norm = np.array([])
            alignA_norm = np.array([])
            alignB_norm = np.array([])

            theta_dist_norm = np.array([])
            thetaA_dist_norm = np.array([])
            thetaB_dist_norm = np.array([])
            
            # Loop over all particles
            for h in range(0, len(self.pos)):
                #if int_dict['part'][h] == int(int_comp_dict['ids'][m]):
                # Separation distance from largest custer's CoM (middle of box)
                difx = self.pos[h,0] - (com_x-self.hx_box)
                dify = self.pos[h,1] - (com_y-self.hy_box)

                difr= ( (difx )**2 + (dify)**2)**0.5


                # Save active force magnitude toward the nearest interface surface normal
                if self.typ[h] == 0:
                    fa_norm=np.append(fa_norm, part_align[h]*self.peA)
                    faA_norm=np.append(faA_norm, part_align[h]*self.peA)
                    rA_dist_norm = np.append(rA_dist_norm, difr)
                    alignA_norm=np.append(alignA_norm, part_align[h])
                    thetaA_dist_norm = np.append(thetaA_dist_norm, np.arctan2(dify, difx)*(180/np.pi))
                else:
                    fa_norm=np.append(fa_norm, part_align[h]*self.peB)
                    faB_norm=np.append(faB_norm, part_align[h]*self.peB)
                    alignB_norm=np.append(alignB_norm, part_align[h])
                    rB_dist_norm = np.append(rB_dist_norm, difr)
                    thetaB_dist_norm = np.append(thetaB_dist_norm, np.arctan2(dify, difx)*(180/np.pi))

                # Save separation distance from the nearest interface surface
                r_dist_norm = np.append(r_dist_norm, difr)
                align_norm=np.append(align_norm, part_align[h])
                theta_dist_norm = np.append(theta_dist_norm, np.arctan2(dify, difx)*(180/np.pi))
            # Dictionary containing each particle's alignment and aligned active force toward
            # the nearest interface surface normal as a function of separation distance from
            # largest custer's CoM
            radial_fa_dict[key] = {'all': {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm, 'theta': theta_dist_norm}, 'A': {'r': rA_dist_norm, 'fa': faA_norm, 'align': alignA_norm, 'theta': thetaA_dist_norm}, 'B': {'r': rB_dist_norm, 'fa': faB_norm, 'align': alignB_norm, 'theta': thetaB_dist_norm}}

        return radial_fa_dict
    
    def radial_surface_normal_fa_bubble3(self, method2_align_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, int_dict):
        '''
        Purpose: Takes the orientation, position, and active force of each particle
        to calculate the active force magnitude toward, alignment toward, and separation
        distance from the nearest interface surface normal

        Input:
        method2_align_dict: dictionary containing the alignment
        of each particle's active force with the nearest interface's surface normal,
        per and averaged per bin
        
        Output:
        radial_fa_dict: dictionary containing each particle's alignment and aligned active force with
        the nearest interface surface normal as a function of separation distance from largest custer's CoM
        '''
        radial_fa_dict = {}
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            try: 
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']
        
            # Instantiate empty array (partNum) containing the average active force alignment
            # with the nearest interface surface normal
            part_align = method2_align_dict['part']['align']

            # Instantiate empty array (partNum) containing the average active force magnitude
            # toward the nearest interface surface normal
            fa_norm = np.array([])
            faA_norm = np.array([])
            faB_norm = np.array([])

            # Instantiate empty array (partNum) containing the distance from the nearest interface surface
            r_dist_norm = np.array([])
            rA_dist_norm = np.array([])
            rB_dist_norm = np.array([])

            align_norm = np.array([])
            alignA_norm = np.array([])
            alignB_norm = np.array([])
            
            # Loop over all particles
            for h in range(0, len(self.pos)):
                if int_dict['part'][h] == int(int_comp_dict['ids'][m]):
                    # Separation distance from largest custer's CoM (middle of box)
                    difx = self.pos[h,0] - (com_x-self.hx_box)
                    dify = self.pos[h,1] - (com_y-self.hy_box)

                    difr= ( (difx )**2 + (dify)**2)**0.5

                    # Save active force magnitude toward the nearest interface surface normal
                    if self.typ[h] == 0:
                        fa_norm=np.append(fa_norm, part_align[h]*self.peA)
                        faA_norm=np.append(faA_norm, part_align[h]*self.peA)
                        rA_dist_norm = np.append(rA_dist_norm, difr)
                        alignA_norm=np.append(alignA_norm, part_align[h])
                    else:
                        fa_norm=np.append(fa_norm, part_align[h]*self.peB)
                        faB_norm=np.append(faB_norm, part_align[h]*self.peB)
                        alignB_norm=np.append(alignB_norm, part_align[h])
                        rB_dist_norm = np.append(rB_dist_norm, difr)

                    # Save separation distance from the nearest interface surface
                    r_dist_norm = np.append(r_dist_norm, difr)
                    align_norm=np.append(align_norm, part_align[h])
            # Dictionary containing each particle's alignment and aligned active force toward
            # the nearest interface surface normal as a function of separation distance from
            # largest custer's CoM
            radial_fa_dict[key] = {'all': {'r': r_dist_norm, 'fa': fa_norm, 'align': align_norm}, 'A': {'r': rA_dist_norm, 'fa': faA_norm, 'align': alignA_norm}, 'B': {'r': rB_dist_norm, 'fa': faB_norm, 'align': alignB_norm}}

        return radial_fa_dict

    def radial_measurements2(self, radial_fa_dict, surface_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict):
        int_id = averaged_data_arr['int_id']

        int_ids = int_dict['bin']
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            key2 = 'surface ' + str(int(int_comp_dict['ids'][m]))

            try: 
                exterior_radius = all_surface_measurements[key]['exterior']['mean radius']
                exterior = 1
            except:
                exterior = 0
                exterior_radius = 0

            try:
                interior_radius = all_surface_measurements[key]['interior']['mean radius']
                interior = 1
            except:
                interior = 0
                interior_radius = 0
                
            if exterior_radius >= interior_radius:
                radius = exterior_radius
            else:
                radius = interior_radius

            #X locations across interface for integration
            if radius * 1.5 <self.hx_box:
                try:
                    r = np.linspace(0, exterior_radius * 1.5, num=int((np.ceil(exterior_radius * 1.5 - interior_radius * 0.5)+1)/3))
                except:
                    r = np.linspace(0, radius * 1.5, num=int((np.ceil(radius + 40.0)+1)/3))
            else:
                try:
                    r = np.linspace(interior_radius * 0.5, self.hx_box, num=int((np.ceil(self.hx_box - interior_radius * 0.5)+1)/3))
                except:
                    r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))        

            #Pressure integrand components for each value of X
            int_stress_XX_r = np.zeros((len(r)-1))
            int_stress_YY_r = np.zeros((len(r)-1))
            int_stress_XY_r = np.zeros((len(r)-1))
            int_stress_YX_r = np.zeros((len(r)-1))
            int_press_r = np.zeros((len(r)-1))
            #act_fa_r = []
            #lat_r = []
            num_dens_r = np.zeros((len(r)-1))

            int_stressA_XX_r = np.zeros((len(r)-1))
            int_stressA_YY_r = np.zeros((len(r)-1))
            int_stressA_XY_r = np.zeros((len(r)-1))
            int_stressA_YX_r = np.zeros((len(r)-1))
            int_pressA_r = np.zeros((len(r)-1))

            #act_faA_r = []
            #latA_r = []
            num_densA_r = np.zeros((len(r)-1))

            int_stressB_XX_r = np.zeros((len(r)-1))
            int_stressB_YY_r = np.zeros((len(r)-1))
            int_stressB_XY_r = np.zeros((len(r)-1))
            int_stressB_YX_r = np.zeros((len(r)-1))
            int_pressB_r = np.zeros((len(r)-1))
            #act_faB_r = []
            #latB_r = []
            num_densB_r = np.zeros((len(r)-1))

            #If exterior and interior surfaces defined, continue...

            area_prev = 0

            #Pressure integrand components for each value of X
            act_press_r = np.zeros((len(r)-1))
            act_fa_r = np.zeros((len(r)-1))
            align_r = np.zeros((len(r)-1))
            num_dens_r = np.zeros((len(r)-1))

            act_pressA_r = np.zeros((len(r)-1))
            act_faA_r = np.zeros((len(r)-1))
            alignA_r = np.zeros((len(r)-1))
            num_densA_r = np.zeros((len(r)-1))

            act_pressB_r = np.zeros((len(r)-1))
            act_faB_r = np.zeros((len(r)-1))
            alignB_r = np.zeros((len(r)-1))
            num_densB_r = np.zeros((len(r)-1))

            rad_arr = np.zeros((len(r)-1))
            
            #If exterior and interior surfaces defined, continue...
        
            #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
            for i in range(1, len(r)):

                #Min and max location across interface of current step
                min_r = r[i-1]
                max_r = r[i]

                #Calculate area of rectangle for current step
                area = np.pi * (max_r ** 2) - area_prev

                #Save total area of previous step sizes
                area_prev = np.pi * (max_r ** 2)

                #Find particles that are housed within current slice
                parts_inrange = np.where((min_r<=radial_stress_dict[key]['all']['r']) & (radial_stress_dict[key]['all']['r']<=max_r))[0]
                partsA_inrange = np.where((min_r<=radial_stress_dict[key]['A']['r']) & (radial_stress_dict[key]['A']['r']<=max_r))[0]
                partsB_inrange = np.where((min_r<=radial_stress_dict[key]['B']['r']) & (radial_stress_dict[key]['B']['r']<=max_r))[0]
                #Find particles that are housed within current slice
                parts_inrange_fa = np.where((min_r<=radial_fa_dict[key]['all']['r']) & (radial_fa_dict[key]['all']['r']<=max_r))[0]
                partsA_inrange_fa = np.where((min_r<=radial_fa_dict[key]['A']['r']) & (radial_fa_dict[key]['A']['r']<=max_r))[0]
                partsB_inrange_fa = np.where((min_r<=radial_fa_dict[key]['B']['r']) & (radial_fa_dict[key]['B']['r']<=max_r))[0]


                #If at least 1 particle in slice, continue...
                if len(parts_inrange)>0:
                    
                    #If the force is defined, continue...
                    parts_defined = np.logical_not(np.isnan(radial_stress_dict[key]['all']['XX'][parts_inrange]))

                    if len(parts_defined)>0:
                        #Calculate total active force normal to interface in slice
                        int_stress_XX_r[i-1] = np.sum((radial_stress_dict[key]['all']['XX'][parts_inrange][parts_defined]))
                        int_stress_YY_r[i-1] = np.sum((radial_stress_dict[key]['all']['YY'][parts_inrange][parts_defined]))
                        int_stress_XY_r[i-1] = np.sum((radial_stress_dict[key]['all']['XY'][parts_inrange][parts_defined]))
                        int_stress_YX_r[i-1] = np.sum((radial_stress_dict[key]['all']['YX'][parts_inrange][parts_defined]))
                        int_press_r[i-1] = np.sum((radial_stress_dict[key]['all']['XX'][parts_inrange][parts_defined] + radial_stress_dict[key]['all']['YY'][parts_inrange][parts_defined])/2)
                        #Calculate density
                        num_dens_r[i-1] = len(parts_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            int_press_r[i-1] = int_press_r[i-1]/area
                            num_dens_r[i-1] = num_dens_r[i-1]/area

                        partsA_defined = np.logical_not(np.isnan(radial_stress_dict[key]['A']['XX'][partsA_inrange]))

                        if len(partsA_defined)>0:
                            int_stressA_XX_r[i-1] = np.sum((radial_stress_dict[key]['A']['XX'][partsA_inrange][partsA_defined]))
                            int_stressA_YY_r[i-1] = np.sum((radial_stress_dict[key]['A']['YY'][partsA_inrange][partsA_defined]))
                            int_stressA_XY_r[i-1] = np.sum((radial_stress_dict[key]['A']['XY'][partsA_inrange][partsA_defined]))
                            int_stressA_YX_r[i-1] = np.sum((radial_stress_dict[key]['A']['YX'][partsA_inrange][partsA_defined]))
                            int_pressA_r[i-1] = np.sum((radial_stress_dict[key]['A']['XX'][partsA_inrange][partsA_defined] + radial_stress_dict[key]['A']['YY'][partsA_inrange][partsA_defined])/2)
                            #Calculate density
                            num_densA_r[i-1] = len([partsA_defined])
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                int_pressA_r[i-1] = int_pressA_r[i-1]/area
                                num_densA_r[i-1] = num_densA_r[i-1]/area

                        partsB_defined = np.logical_not(np.isnan(radial_stress_dict[key]['B']['XX'][partsB_inrange]))
                        if len(partsB_defined)>0:
                            int_stressB_XX_r[i-1] = np.sum((radial_stress_dict[key]['B']['XX'][partsB_inrange][partsB_defined]))
                            int_stressB_YY_r[i-1] = np.sum((radial_stress_dict[key]['B']['YY'][partsB_inrange][partsB_defined]))
                            int_stressB_XY_r[i-1] = np.sum((radial_stress_dict[key]['B']['XY'][partsB_inrange][partsB_defined]))
                            int_stressB_YX_r[i-1] = np.sum((radial_stress_dict[key]['B']['YX'][partsB_inrange][partsB_defined]))
                            int_pressB_r[i-1] = np.sum((radial_stress_dict[key]['B']['XX'][partsB_inrange][partsB_defined] + radial_stress_dict[key]['B']['YY'][partsB_inrange][partsB_defined])/2)
                            #Calculate density
                            num_densB_r[i-1] = len([partsB_defined])
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                int_pressB_r[i-1] = int_pressB_r[i-1]/area
                                num_densB_r[i-1] = num_densB_r[i-1]/area
                    #If the force is defined, continue...
                    parts_defined = np.logical_not(np.isnan(radial_fa_dict[key]['all']['fa'][parts_inrange]))

                    if len(parts_defined)>0:
                        #Calculate total active force normal to interface in slice
                        act_press_r[i-1] = np.sum(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                        act_fa_r[i-1] = np.mean(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                        align_r[i-1] = np.mean(radial_fa_dict[key]['all']['align'][parts_inrange_fa][parts_defined])
                        num_dens_r[i-1] = len(parts_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_press_r[i-1] = act_press_r[i-1]/area
                            num_dens_r[i-1] = num_dens_r[i-1]/area

                        partsA_defined = np.logical_not(np.isnan(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa]))
                        if len(partsA_defined)>0:
                            act_pressA_r[i-1] = np.sum(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                            act_faA_r[i-1] = np.mean(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                            alignA_r[i-1] = np.mean(radial_fa_dict[key]['A']['align'][partsA_inrange_fa][partsA_defined])
                            num_densA_r[i-1] = len(partsA_defined)
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                act_pressA_r[i-1] = act_pressA_r[i-1]/area
                                num_densA_r[i-1] = num_densA_r[i-1]/area

                        partsB_defined = np.logical_not(np.isnan(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa]))
                        if len(partsB_defined)>0:
                            act_pressB_r[i-1] = np.sum(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                            act_faB_r[i-1] = np.mean(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                            alignB_r[i-1] = np.mean(radial_fa_dict[key]['B']['align'][partsB_inrange_fa][partsB_defined])
                            #Calculate density
                            num_densB_r[i-1] = len(partsB_defined)
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                act_pressB_r[i-1] = act_pressB_r[i-1]/area
                                num_densB_r[i-1] = num_densB_r[i-1]/area
            plt.plot(r[1:], act_press_r, color='purple')
            plt.plot(r[1:], act_pressA_r, color='blue')
            plt.plot(r[1:], act_pressB_r, color='red')

        plt.show()
        com_radial_dict_fa = {'r': r[1:].tolist(), 'fa_press': {'all': act_press_r.tolist(), 'A': act_pressA_r.tolist(), 'B': act_pressB_r.tolist()}, 'fa': {'all': act_fa_r.tolist(), 'A': act_faA_r.tolist(), 'B': act_faB_r.tolist()}, 'align': {'all': align_r.tolist(), 'A': alignA_r.tolist(), 'B': alignB_r.tolist()}, 'num_dens': {'all': num_dens_r.tolist(), 'A': num_densA_r.tolist(), 'B': num_densB_r.tolist()}}
        com_radial_dict = {'r': r[1:].tolist(), 'all': {'XX': int_stress_XX_r.tolist(), 'YY': int_stress_YY_r.tolist(), 'XY': int_stress_XY_r.tolist(), 'YX': int_stress_YX_r.tolist(), 'press': int_press_r.tolist(), 'num_dens': num_dens_r.tolist()}, 'A': {'XX': int_stressA_XX_r.tolist(), 'YY': int_stressA_YY_r.tolist(), 'XY': int_stressA_XY_r.tolist(), 'YX': int_stressA_YX_r.tolist(), 'press': int_pressA_r.tolist(), 'num_dens': num_densA_r.tolist()}, 'B': {'XX': int_stressB_XX_r.tolist(), 'YY': int_stressB_YY_r.tolist(), 'XY': int_stressB_XY_r.tolist(), 'YX': int_stressB_YX_r.tolist(), 'press': int_pressB_r.tolist(), 'num_dens': num_densB_r.tolist()}}
        return com_radial_dict, com_radial_dict_fa
    def radial_ang_measurements(self, radial_fa_dict, radial_int_press_dict, surface_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict):
        int_id = averaged_data_arr['int_id']
        com_radial_dict_fa = {}
        int_ids = int_dict['bin']
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            key2 = 'surface ' + str(int(int_comp_dict['ids'][m]))


            try: 
                surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                surface_x = sep_surface_dict[key]['interior']['pos']['x']
                surface_y = sep_surface_dict[key]['interior']['pos']['y']
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']

            try: 
                exterior_radius = all_surface_measurements[key]['exterior']['mean radius']
                exterior = 1
            except:
                exterior = 0
                exterior_radius = 0

            try:
                interior_radius = all_surface_measurements[key]['interior']['mean radius']
                interior = 1
            except:
                interior = 0
                interior_radius = 0
                
            if exterior_radius >= interior_radius:
                radius = exterior_radius
            else:
                radius = interior_radius
                
            theta_all = ((radial_fa_dict[key]['all']['theta'] + 360 ) % 360)
            theta_A = ((radial_fa_dict[key]['A']['theta'] + 360 ) % 360)
            theta_B = ((radial_fa_dict[key]['B']['theta'] + 360 ) % 360)

            theta_all_int_press = ((radial_int_press_dict['all']['theta'] + 360 ) % 360)
            theta_A_int_press = ((radial_int_press_dict['A']['theta'] + 360 ) % 360)
            theta_B_int_press = ((radial_int_press_dict['B']['theta'] + 360 ) % 360)
            
            
            difx = surface_x - (com_x)
            dify = surface_y - (com_y)

            difr= ( (difx )**2 + (dify)**2)**0.5
            thetar = np.arctan2(dify, difx)*(180/np.pi)
            
            thetar_ang = ((thetar + 360 ) % 360)

            #r = np.linspace(np.min(radial_fa_dict[key]['all']['r']), np.max(radial_fa_dict[key]['all']['r']), num=int((np.ceil(np.max(radial_fa_dict[key]['all']['r']) - np.min(radial_fa_dict[key]['all']['r']))+1)/2))
            
            
            r = np.linspace(0, 1.5, num=75)
            theta = np.linspace(0, 360, num=45)

            #Pressure integrand components for each value of X
            int_stress_XX_r = np.zeros((len(r)-1))
            int_stress_YY_r = np.zeros((len(r)-1))
            int_stress_XY_r = np.zeros((len(r)-1))
            int_stress_YX_r = np.zeros((len(r)-1))
            int_press_r = np.zeros((len(r)-1))
            #act_fa_r = []
            #lat_r = []
            num_dens_r = np.zeros((len(r)-1))

            int_stressA_XX_r = np.zeros((len(r)-1))
            int_stressA_YY_r = np.zeros((len(r)-1))
            int_stressA_XY_r = np.zeros((len(r)-1))
            int_stressA_YX_r = np.zeros((len(r)-1))
            int_pressA_r = np.zeros((len(r)-1))

            #act_faA_r = []
            #latA_r = []
            num_densA_r = np.zeros((len(r)-1))

            int_stressB_XX_r = np.zeros((len(r)-1))
            int_stressB_YY_r = np.zeros((len(r)-1))
            int_stressB_XY_r = np.zeros((len(r)-1))
            int_stressB_YX_r = np.zeros((len(r)-1))
            int_pressB_r = np.zeros((len(r)-1))
            #act_faB_r = []
            #latB_r = []
            num_densB_r = np.zeros((len(r)-1))

            #If exterior and interior surfaces defined, continue...

            

            #Pressure integrand components for each value of X
            act_press_r = np.zeros((len(r)-1))
            int_press_r = np.zeros((len(r)-1))
            act_fa_r = np.zeros((len(r)-1))
            align_r = np.zeros((len(r)-1))
            num_dens_r = np.zeros((len(r)-1))
            num_r = np.zeros((len(r)-1))
            area_r = np.zeros((len(r)-1))

            act_pressA_r = np.zeros((len(r)-1))
            int_pressA_r = np.zeros((len(r)-1))
            act_faA_r = np.zeros((len(r)-1))
            alignA_r = np.zeros((len(r)-1))
            num_densA_r = np.zeros((len(r)-1))
            numA_r = np.zeros((len(r)-1))

            act_pressB_r = np.zeros((len(r)-1))
            int_pressB_r = np.zeros((len(r)-1))
            act_faB_r = np.zeros((len(r)-1))
            alignB_r = np.zeros((len(r)-1))
            num_densB_r = np.zeros((len(r)-1))
            numB_r = np.zeros((len(r)-1))

            rad_arr = np.zeros((len(r)-1))

            sum_act_press_r = np.zeros((len(r)-1)) 
            sum_int_press_r = np.zeros((len(r)-1)) 
            sum_act_fa_r = np.zeros((len(r)-1))
            sum_align_r = np.zeros((len(r)-1))
            sum_num_dens_r = np.zeros((len(r)-1))
            sum_num_r = np.zeros((len(r)-1))
            sum_area_r = np.zeros((len(r)-1))

            sum_act_pressA_r = np.zeros((len(r)-1))
            sum_int_pressA_r = np.zeros((len(r)-1))
            sum_act_faA_r = np.zeros((len(r)-1))
            sum_alignA_r = np.zeros((len(r)-1))
            sum_num_densA_r = np.zeros((len(r)-1))
            sum_numA_r = np.zeros((len(r)-1))

            sum_act_pressB_r = np.zeros((len(r)-1))
            sum_int_pressB_r = np.zeros((len(r)-1))
            sum_act_faB_r = np.zeros((len(r)-1))
            sum_alignB_r = np.zeros((len(r)-1))
            sum_num_densB_r = np.zeros((len(r)-1))
            sum_numB_r = np.zeros((len(r)-1))

            num_theta = 0
            
            #If exterior and interior surfaces defined, continue...
        
            #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
            num_theta = 0
            for j in range(1, len(theta)):

                total_area_prev = 0
                total_area_prev_slice = 0

                min_theta = theta[j-1]
                max_theta = theta[j]

                dif_theta = (max_theta - min_theta)/360

                surface_inrange = np.where((min_theta<=thetar_ang) & (thetar_ang<=max_theta))[0]
                radius_temp = np.mean(difr[surface_inrange])

                for i in range(1, len(r)):

                    #Min and max location across interface of current step
                    min_r = r[i-1]
                    max_r = r[i]

                    #Calculate area of rectangle for current step
                    total_area = (np.pi * ((max_r*radius_temp) ** 2) - total_area_prev)
                    total_area_slice = total_area * dif_theta

                    #Save total area of previous step sizes
                    total_area_prev = np.pi * ((max_r*radius_temp) ** 2)
                    total_area_prev_slice = total_area_prev * dif_theta

                    if r[i] * radius_temp <= self.hx_box:

                        #Find particles that are housed within current slice
                        parts_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['all']['r']/radius_temp)) & ((radial_fa_dict[key]['all']['r']/radius_temp)<=max_r) & (min_theta<=theta_all) & (theta_all<=max_theta))[0]
                        partsA_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['A']['r']/radius_temp)) & ((radial_fa_dict[key]['A']['r']/radius_temp)<=max_r) & (min_theta<=theta_A) & (theta_A<=max_theta))[0]
                        partsB_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['B']['r']/radius_temp)) & ((radial_fa_dict[key]['B']['r']/radius_temp)<=max_r) & (min_theta<=theta_B) & (theta_B<=max_theta))[0]

                        parts_inrange_int = np.where((min_r<=(radial_int_press_dict['all']['r']/radius_temp)) & ((radial_int_press_dict['all']['r']/radius_temp)<=max_r) & (min_theta<=theta_all_int_press) & (theta_all_int_press<=max_theta))[0]
                        partsA_inrange_int = np.where((min_r<=(radial_int_press_dict['A']['r']/radius_temp)) & ((radial_int_press_dict['A']['r']/radius_temp)<=max_r) & (min_theta<=theta_A_int_press) & (theta_A_int_press<=max_theta))[0]
                        partsB_inrange_int = np.where((min_r<=(radial_int_press_dict['B']['r']/radius_temp)) & ((radial_int_press_dict['B']['r']/radius_temp)<=max_r) & (min_theta<=theta_B_int_press) & (theta_B_int_press<=max_theta))[0]
                        
                        if len(parts_inrange_int)>0:
                            int_press_r[i-1] = np.sum( (radial_int_press_dict['all']['XX'][parts_inrange_fa] + radial_int_press_dict['all']['YY'][parts_inrange_fa]))
                            area_r[i-1] = total_area_slice
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if total_area_slice > 0:
                                int_press_r[i-1] = int_press_r[i-1]/total_area_slice

                            if len(partsA_inrange_int)>0:
                                int_pressA_r[i-1] = np.sum( (radial_int_press_dict['A']['XX'][partsA_inrange_fa] + radial_int_press_dict['A']['YY'][partsA_inrange_fa]))
                                #If area of slice is non-zero, calculate the pressure [F/A]
                                if total_area_slice > 0:
                                    int_pressA_r[i-1] = int_pressA_r[i-1]/total_area_slice

                            if len(partsB_inrange_int)>0:
                                int_pressB_r[i-1] = np.sum( (radial_int_press_dict['B']['XX'][partsB_inrange_fa] + radial_int_press_dict['B']['YY'][partsB_inrange_fa]))
                                #If area of slice is non-zero, calculate the pressure [F/A]
                                if total_area_slice > 0:
                                    int_pressB_r[i-1] = int_pressB_r[i-1]/total_area_slice

                        #If at least 1 particle in slice, continue...
                        if len(parts_inrange_fa)>0:

                            #If the force is defined, continue...
                            parts_defined = np.logical_not(np.isnan(radial_fa_dict[key]['all']['fa'][parts_inrange_fa]))

                            if len(parts_defined)>0:
                                #Calculate total active force normal to interface in slice
                                act_press_r[i-1] = np.sum(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                                act_fa_r[i-1] = np.mean(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                                align_r[i-1] = np.mean(radial_fa_dict[key]['all']['align'][parts_inrange_fa][parts_defined])
                                num_dens_r[i-1] = len(parts_defined)
                                num_r[i-1] = len(parts_defined)
                                area_r[i-1] = total_area_slice
                                #If area of slice is non-zero, calculate the pressure [F/A]
                                if total_area_slice > 0:
                                    act_press_r[i-1] = act_press_r[i-1]/total_area_slice
                                    num_dens_r[i-1] = num_dens_r[i-1]/total_area_slice

                                partsA_defined = np.logical_not(np.isnan(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa]))
                                if len(partsA_defined)>0:
                                    act_pressA_r[i-1] = np.sum(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                                    act_faA_r[i-1] = np.mean(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                                    alignA_r[i-1] = np.mean(radial_fa_dict[key]['A']['align'][partsA_inrange_fa][partsA_defined])
                                    num_densA_r[i-1] = len(partsA_defined)
                                    numA_r[i-1] = len(partsA_defined)
                                    #If area of slice is non-zero, calculate the pressure [F/A]
                                    if total_area_slice > 0:
                                        act_pressA_r[i-1] = act_pressA_r[i-1]/total_area_slice
                                        num_densA_r[i-1] = num_densA_r[i-1]/total_area_slice

                                partsB_defined = np.logical_not(np.isnan(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa]))
                                if len(partsB_defined)>0:
                                    act_pressB_r[i-1] = np.sum(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                                    act_faB_r[i-1] = np.mean(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                                    alignB_r[i-1] = np.mean(radial_fa_dict[key]['B']['align'][partsB_inrange_fa][partsB_defined])
                                    #Calculate density
                                    num_densB_r[i-1] = len(partsB_defined)
                                    numB_r[i-1] = len(partsB_defined)
                                    #If area of slice is non-zero, calculate the pressure [F/A]
                                    if total_area_slice > 0:
                                        act_pressB_r[i-1] = act_pressB_r[i-1]/total_area_slice
                                        num_densB_r[i-1] = num_densB_r[i-1]/total_area_slice
                            
                                
                                
                    
                    else:

                        int_press_r[i-1] = 0
                        act_press_r[i-1] = 0
                        act_fa_r[i-1] = 0
                        align_r[i-1] = 0
                        num_dens_r[i-1] = 0
                        num_r[i-1] = 0

                        int_pressA_r[i-1] = 0
                        act_pressA_r[i-1] = 0
                        act_faA_r[i-1] = 0
                        alignA_r[i-1] = 0
                        num_densA_r[i-1] = 0
                        numA_r[i-1] = 0

                        int_pressB_r[i-1] = 0
                        act_pressB_r[i-1] = 0
                        act_faB_r[i-1] = 0
                        alignB_r[i-1] = 0
                        num_densB_r[i-1] = 0
                        numB_r[i-1] = 0

                #save this theta's measurement
                sum_act_press_r += act_press_r
                sum_int_press_r += int_press_r
                sum_act_fa_r += act_fa_r
                sum_align_r += align_r
                sum_num_dens_r += num_dens_r
                sum_num_r += num_r

                sum_act_pressA_r += act_pressA_r
                sum_int_pressA_r += int_pressA_r
                sum_act_faA_r += act_faA_r
                sum_alignA_r += alignA_r
                sum_num_densA_r += num_densA_r
                sum_numA_r += numA_r

                sum_act_pressB_r += act_pressB_r
                sum_int_pressB_r += int_pressB_r
                sum_act_faB_r += act_faB_r
                sum_alignB_r += alignB_r
                sum_num_densB_r += num_densB_r
                sum_numB_r += numB_r

                num_theta += 1

            avg_act_press_r = sum_act_press_r / num_theta
            avg_int_press_r = sum_int_press_r / num_theta
            avg_act_fa_r = sum_act_fa_r / num_theta
            avg_align_r = sum_align_r / num_theta
            avg_num_dens_r = sum_num_dens_r / num_theta
            avg_num_r = sum_num_r / num_theta

            avg_act_pressA_r = sum_act_pressA_r / num_theta
            avg_int_pressA_r = sum_int_pressA_r / num_theta
            avg_act_faA_r = sum_act_faA_r / num_theta
            avg_alignA_r = sum_alignA_r / num_theta
            avg_num_densA_r = sum_num_densA_r / num_theta
            avg_numA_r = sum_numA_r / num_theta

            avg_act_pressB_r = sum_act_pressB_r / num_theta
            avg_int_pressB_r = sum_int_pressB_r / num_theta
            avg_act_faB_r = sum_act_faB_r / num_theta
            avg_alignB_r = sum_alignB_r / num_theta
            avg_num_densB_r = sum_num_densB_r / num_theta
            avg_numB_r = sum_numB_r / num_theta


            com_radial_dict_fa[key] = {'int_id': int_id, 'current_id': int(int_comp_dict['ids'][m]), 'ext_rad': exterior_radius, 'int_rad': interior_radius, 'r': r[1:].tolist(), 'area': area_r.tolist(), 'com_x': com_x, 'com_y': com_y, 'int_press': {'all': avg_int_press_r.tolist(), 'A': avg_int_pressA_r.tolist(), 'B': avg_int_pressB_r.tolist()}, 'fa_press': {'all': avg_act_press_r.tolist(), 'A': avg_act_pressA_r.tolist(), 'B': avg_act_pressB_r.tolist()}, 'fa': {'all': avg_act_fa_r.tolist(), 'A': avg_act_faA_r.tolist(), 'B': avg_act_faB_r.tolist()}, 'align': {'all': avg_align_r.tolist(), 'A': avg_alignA_r.tolist(), 'B': avg_alignB_r.tolist()}, 'num_dens': {'all': avg_num_dens_r.tolist(), 'A': avg_num_densA_r.tolist(), 'B': avg_num_densB_r.tolist()}, 'num': {'all': avg_num_r.tolist(), 'A': avg_numA_r.tolist(), 'B': avg_numB_r.tolist()}}
        return com_radial_dict_fa

    def bulk_heterogeneity_avgs(self, method2_align_dict, surface_dict, int_comp_dict, all_surface_measurements, int_dict, phase_dict):
        
        bulk_id = np.where(phase_dict['part']==0)[0]
        bulk_A_id = np.where((phase_dict['part']==0) & (self.typ==0))[0]
        bulk_B_id = np.where((phase_dict['part']==0) & (self.typ==1))[0]

        int_id = np.where(phase_dict['part']==1)[0]
        int_A_id = np.where((phase_dict['part']==1) & (self.typ==0))[0]
        int_B_id = np.where((phase_dict['part']==1) & (self.typ==1))[0]

        gas_id = np.where(phase_dict['part']==2)[0]
        gas_A_id = np.where((phase_dict['part']==2) & (self.typ==0))[0]
        gas_B_id = np.where((phase_dict['part']==2) & (self.typ==1))[0]

        A_id = np.where((self.typ==0))[0]
        B_id = np.where((self.typ==1))[0]

        interface_id = int(np.where(int_comp_dict['comp']==np.max(int_comp_dict['comp']))[0])

        int_id = np.where(int_dict['part']==int_dict['largest ids'][0])[0]
        key = 'surface id ' + str(int(int_dict['largest ids'][0]))

        min_x = all_surface_measurements[key]['exterior']['com']['x']-25
        max_x = all_surface_measurements[key]['exterior']['com']['x']+25

        min_y = all_surface_measurements[key]['exterior']['com']['y']-25
        max_y = all_surface_measurements[key]['exterior']['com']['y']+25

        #min_x = self.hx_box-20
        #max_x = self.hx_box+20

        #min_y = self.hy_box-20
        #max_y = self.hy_box+20

        pos = np.copy(self.pos)
        pos[:,0] = pos[:,0] + +self.hx_box
        pos[:,1] = pos[:,1] + +self.hy_box

        
        test_id = np.where(pos[:,0]>self.lx_box)[0]
        if len(test_id)>0:
            pos[:,0] = pos[:,0]-self.lx_box
        test_id = np.where(pos[:,0]<0)[0]
        if len(test_id)>0:
            pos[:,0] = pos[:,0]+self.lx_box

        test_id = np.where(pos[:,1]>self.ly_box)[0]
        if len(test_id)>0:
            pos[:,1] = pos[:,1]-self.ly_box
        test_id = np.where(pos[:,1]<0)[0]
        if len(test_id)>0:
            pos[:,1] = pos[:,1]+self.ly_box
        
        test_id = np.where((pos[:,0]>=min_x) & (pos[:,0]<=max_x) & (pos[:,1]>=min_y) & (pos[:,1]<=max_y))[0]
        
        pos_wrap_x = pos[test_id,0]
        pos_wrap_y = pos[test_id,1]

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty array (partNum) containing the average active force magnitude
        # toward the nearest interface surface normal
        fa_norm = np.array([])
        faA_norm = np.array([])
        faB_norm = np.array([])

        fa_avg = np.array([])

        # Instantiate empty array (partNum) containing the distance from the nearest interface surface
        x_dist_norm = np.array([])
        xA_dist_norm = np.array([])
        xB_dist_norm = np.array([])

        align_norm = np.array([])
        alignA_norm = np.array([])
        alignB_norm = np.array([])

        y_dist_norm = np.array([])
        yA_dist_norm = np.array([])
        yB_dist_norm = np.array([])

        for m in range(0, len(pos_wrap_x)):

            # Save active force magnitude toward the nearest interface surface normal
            if self.typ[test_id[m]] == 0:
                fa_norm=np.append(fa_norm, part_align[test_id[m]]*self.peA)
                fa_avg=np.append(fa_avg, self.peA)
                faA_norm=np.append(faA_norm, part_align[test_id[m]]*self.peA)
                alignA_norm=np.append(alignA_norm, part_align[test_id[m]])
                xA_dist_norm = np.append(xA_dist_norm, pos_wrap_x[m])
                yA_dist_norm = np.append(yA_dist_norm, pos_wrap_y[m])
            else:
                fa_norm=np.append(fa_norm, part_align[test_id[m]]*self.peB)
                fa_avg=np.append(fa_avg, self.peB)
                faB_norm=np.append(faB_norm, part_align[test_id[m]]*self.peB)
                alignB_norm=np.append(alignB_norm, part_align[test_id[m]])
                xB_dist_norm = np.append(xB_dist_norm, pos_wrap_x[m])
                yB_dist_norm = np.append(yB_dist_norm, pos_wrap_y[m])

            # Save separation distance from the nearest interface surface
            #print(avg_rad)
            x_dist_norm = np.append(x_dist_norm, pos_wrap_x[m])
            y_dist_norm = np.append(y_dist_norm, pos_wrap_y[m])
            align_norm=np.append(align_norm, part_align[test_id[m]])

        import math

        def round_decimals_down(number:float, decimals:int=2):
            """
            Returns a value rounded down to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.floor(number)

            factor = 10 ** decimals
            return math.floor(number * factor) / factor

        def round_decimals_up(number:float, decimals:int=2):
            """
            Returns a value rounded up to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.ceil(number)

            factor = 10 ** decimals
            return math.ceil(number * factor) / factor

        y_final_bins = np.round(np.linspace(min_y, max_y, 20),3)
        x_final_bins = np.round(np.linspace(min_x, max_x, 20),3)

        x_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        y_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_avg_real_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        
        fa_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))


        align_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        alignA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        alignB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        num_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        num_densA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        num_densB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        num_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        numA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))
        numB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        
        fa_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faA_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        faB_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))

        fa_avg_real_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))

        align_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        alignA_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        alignB_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))

        num_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        num_densA_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))
        num_densB_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))

        num_final_binned_val = np.zeros((len(y_final_bins), len(x_final_bins)))
        numA_final_binned_val = np.zeros((len(y_final_bins), len(x_final_bins)))
        numB_final_binned_val = np.zeros((len(y_final_bins), len(x_final_bins)))

        sum_id = 0
        sum_id_theta = 0
        
        for xind in range(1, len(x_final_bins)):

            for yind in range(1, len(y_final_bins)):
                
                bin_ids = np.where((x_dist_norm <x_final_bins[xind]) & (x_dist_norm >= x_final_bins[xind-1]) & (y_dist_norm < y_final_bins[yind]) & (y_dist_norm >= y_final_bins[yind-1]))[0]
                binA_ids = np.where((xA_dist_norm <x_final_bins[xind]) & (xA_dist_norm >= x_final_bins[xind-1]) & (yA_dist_norm < y_final_bins[yind]) & (yA_dist_norm >= y_final_bins[yind-1]))[0]
                binB_ids = np.where((xB_dist_norm <x_final_bins[xind]) & (xB_dist_norm >= x_final_bins[xind-1]) & (yB_dist_norm < y_final_bins[yind]) & (yB_dist_norm >= y_final_bins[yind-1]))[0]
                
                if len(bin_ids)>0:

                    area_bin = (x_final_bins[xind]-x_final_bins[xind-1]) * (y_final_bins[yind]-y_final_bins[yind-1])

                    fa_avg_real_final_binned[xind-1, yind-1] = np.mean(fa_avg[bin_ids])
                    fa_sum_final_binned[xind-1, yind-1] = np.sum(fa_norm[bin_ids])
                    fa_avg_final_binned[xind-1, yind-1] = np.mean(fa_norm[bin_ids])

                    x_final_binned[xind-1, yind-1] = x_final_bins[xind-1]
                    y_final_binned[xind-1, yind-1] = y_final_bins[yind-1]

                    if len(binA_ids)>0:
                        faA_sum_final_binned[xind-1, yind-1] = np.sum(faA_norm[binA_ids])
                    if len(binB_ids)>0:
                        faB_sum_final_binned[xind-1, yind-1] = np.sum(faB_norm[binB_ids])

                    if len(binA_ids)>0:
                        faA_avg_final_binned[xind-1, yind-1] = np.mean(faA_norm[binA_ids])
                    if len(binB_ids)>0:
                        faB_avg_final_binned[xind-1, yind-1] = np.mean(faB_norm[binB_ids])

                    fa_dens_final_binned[xind-1, yind-1] = np.sum(fa_norm[bin_ids]) / area_bin
                    if len(binA_ids)>0:
                        faA_dens_final_binned[xind-1, yind-1] = np.sum(faA_norm[binA_ids]) / area_bin
                    if len(binB_ids)>0:
                        faB_dens_final_binned[xind-1, yind-1] = np.sum(faB_norm[binB_ids]) / area_bin

                    align_final_binned[xind-1, yind-1] = np.mean(align_norm[bin_ids])
                    if len(binA_ids)>0:
                        alignA_final_binned[xind-1, yind-1] = np.mean(alignA_norm[binA_ids])
                    if len(binB_ids)>0:
                        alignB_final_binned[xind-1, yind-1] = np.mean(alignB_norm[binB_ids])

                    num_dens_final_binned[xind-1, yind-1] = len(bin_ids) / area_bin
                    if len(binA_ids)>0:
                        num_densA_final_binned[xind-1, yind-1] = len(binA_ids) / area_bin
                    if len(binB_ids)>0:
                        num_densB_final_binned[xind-1, yind-1] = len(binB_ids) / area_bin

                    num_final_binned[xind-1, yind-1] = len(bin_ids)
                    if len(binA_ids)>0:
                        numA_final_binned[xind-1, yind-1] = len(binA_ids)
                    if len(binB_ids)>0:
                        numB_final_binned[xind-1, yind-1] = len(binB_ids)

                    sum_id += 1

        bulk_heterogeneity_dict = {'x': x_final_binned, 'y': x_final_binned, 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned}, 'fa_sum': {'all': fa_sum_final_binned, 'A': faA_sum_final_binned, 'B': faB_sum_final_binned}, 'fa_avg': {'all': fa_avg_final_binned, 'A': faA_avg_final_binned, 'B': faB_avg_final_binned}, 'fa_dens': {'all': fa_dens_final_binned, 'A': faA_dens_final_binned, 'B': faB_dens_final_binned}, 'align': {'all': align_final_binned, 'A': alignA_final_binned, 'B': alignB_final_binned}, 'num_dens': {'all': num_dens_final_binned, 'A': num_densA_final_binned, 'B': num_densB_final_binned}}
        #radial_heterogeneity_dict_theta = {'theta': theta_final_bins, 'fa_avg_real': {'all': fa_avg_real_final_binned_theta}, 'fa_sum': {'all': fa_sum_final_binned_theta, 'A': faA_sum_final_binned_theta, 'B': faB_sum_final_binned_theta}, 'fa_avg': {'all': fa_avg_final_binned_theta, 'A': faA_avg_final_binned_theta, 'B': faB_avg_final_binned_theta}, 'fa_dens': {'all': fa_dens_final_binned_theta, 'A': faA_dens_final_binned_theta, 'B': faB_dens_final_binned_theta}, 'align': {'all': align_final_binned_theta, 'A': alignA_final_binned_theta, 'B': alignB_final_binned_theta}, 'num_dens': {'all': num_dens_final_binned_theta, 'A': num_densA_final_binned_theta, 'B': num_densB_final_binned_theta}}
        plot_bin_dict = {'x': x_final_bins, 'y': y_final_bins, 'fa_sum': {'all': fa_sum_final_binned_val, 'A': faA_sum_final_binned_val, 'B': faB_sum_final_binned_val}, 'fa_avg': {'all': fa_avg_final_binned_val, 'A': faA_avg_final_binned_val, 'B': faB_avg_final_binned_val}, 'fa_dens': {'all': fa_dens_final_binned_val, 'A': faA_dens_final_binned_val, 'B': faB_dens_final_binned_val}, 'align': {'all': align_final_binned_val, 'A': alignA_final_binned_val, 'B': alignB_final_binned_val}, 'num_dens': {'all': num_dens_final_binned_val, 'A': num_densA_final_binned_val, 'B': num_densB_final_binned_val}}
            
        return bulk_heterogeneity_dict, plot_bin_dict

    def radial_heterogeneity_avgs(self, method2_align_dict, surface_dict, int_comp_dict, all_surface_measurements, int_dict, phase_dict):
        
        bulk_id = np.where(phase_dict['part']==0)[0]
        bulk_A_id = np.where((phase_dict['part']==0) & (self.typ==0))[0]
        bulk_B_id = np.where((phase_dict['part']==0) & (self.typ==1))[0]

        int_id = np.where(phase_dict['part']==1)[0]
        int_A_id = np.where((phase_dict['part']==1) & (self.typ==0))[0]
        int_B_id = np.where((phase_dict['part']==1) & (self.typ==1))[0]

        gas_id = np.where(phase_dict['part']==2)[0]
        gas_A_id = np.where((phase_dict['part']==2) & (self.typ==0))[0]
        gas_B_id = np.where((phase_dict['part']==2) & (self.typ==1))[0]

        A_id = np.where((self.typ==0))[0]
        B_id = np.where((self.typ==1))[0]

        try:
            int_id = np.where(int_dict['part']==int_dict['largest ids'][0])[0]
            key = 'surface id ' + str(int(int_dict['largest ids'][0]))
        except:
            interface_id = int(np.where(int_comp_dict['comp']==np.max(int_comp_dict['comp']))[0])
            interface_id = int_comp_dict['ids'][interface_id]
            int_id = np.where(int_dict['part']==interface_id)[0]
            key = 'surface id ' + str(int(interface_id))
        
        ext_rad = (surface_dict[key]['exterior']['pos']['x']**2 + surface_dict[key]['exterior']['pos']['y']**2)**0.5

        

        difx_ext_surface = (surface_dict[key]['exterior']['pos']['x'] - all_surface_measurements[key]['exterior']['com']['x'])
        dify_ext_surface = (surface_dict[key]['exterior']['pos']['y'] - all_surface_measurements[key]['exterior']['com']['y'])
        difr_ext_surface = (difx_ext_surface ** 2 + dify_ext_surface ** 2) ** 0.5

        ang_ext_surface = np.arctan2(dify_ext_surface, difx_ext_surface)*(180/np.pi)
        
        neg_ang = np.where(ang_ext_surface<=0)[0]
        if len(neg_ang)>0:
            ang_ext_surface[neg_ang] = 180 + (180+ang_ext_surface[neg_ang])
        
        pos_wrap_x = self.pos[int_id,0] + self.hx_box
        test_id = np.where(pos_wrap_x>self.lx_box)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]-self.lx_box
        test_id = np.where(pos_wrap_x<0)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]+self.lx_box

        pos_wrap_y = self.pos[int_id,1] + self.hy_box
        test_id = np.where(pos_wrap_y>self.ly_box)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]-self.ly_box
        test_id = np.where(pos_wrap_y<0)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]+self.ly_box

        difx_parts = (pos_wrap_x - all_surface_measurements[key]['exterior']['com']['x'])
        dify_parts = (pos_wrap_y - all_surface_measurements[key]['exterior']['com']['y'])
        difr_parts = (difx_parts ** 2 + dify_parts ** 2) ** 0.5

        ang_parts = np.arctan2(dify_parts, difx_parts)*(180/np.pi)
        neg_ang = np.where(ang_parts<=0)[0]
        if len(neg_ang)>0:
            ang_parts[neg_ang] = 180 + (180+ang_parts[neg_ang])

        difr_parts_norm = np.zeros(len(difr_parts))

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty array (partNum) containing the average active force magnitude
        # toward the nearest interface surface normal
        fa_norm = np.array([])
        faA_norm = np.array([])
        faB_norm = np.array([])

        fa_avg = np.array([])

        # Instantiate empty array (partNum) containing the distance from the nearest interface surface
        r_dist_norm = np.array([])
        rA_dist_norm = np.array([])
        rB_dist_norm = np.array([])

        align_norm = np.array([])
        alignA_norm = np.array([])
        alignB_norm = np.array([])

        theta_dist_norm = np.array([])
        thetaA_dist_norm = np.array([])
        thetaB_dist_norm = np.array([])

        r_dist = np.array([])
        rA_dist = np.array([])
        rB_dist = np.array([])
        
        nearest_surface_arr = np.array([])
        nearest_surfaceA_arr = np.array([])
        nearest_surfaceB_arr = np.array([])
        

        for m in range(0, len(pos_wrap_x)):

            near_surfaces = np.where( (ang_ext_surface>= ang_parts[m]-0.5)& (ang_ext_surface<= ang_parts[m]+0.5) )[0]

            if len(near_surfaces) > 0:
                difx_surface_part = (surface_dict[key]['exterior']['pos']['x'][near_surfaces] - pos_wrap_x[m])
                dify_surface_part  = (surface_dict[key]['exterior']['pos']['y'][near_surfaces] - pos_wrap_y[m])
                difr_surface_part  = np.mean((difx_surface_part ** 2 + dify_surface_part ** 2) ** 0.5)
                avg_rad = np.mean(difr_ext_surface[near_surfaces])
                difr_parts_norm[m] = difr_parts[m]/avg_rad
                nearest_surface_arr = np.append(nearest_surface_arr, avg_rad)

                # Save active force magnitude toward the nearest interface surface normal
                if self.typ[int_id[m]] == 0:
                    fa_norm=np.append(fa_norm, part_align[int_id[m]]*self.peA)
                    fa_avg=np.append(fa_avg, self.peA)
                    faA_norm=np.append(faA_norm, part_align[int_id[m]]*self.peA)
                    rA_dist_norm = np.append(rA_dist_norm, difr_parts[m]/avg_rad)
                    alignA_norm=np.append(alignA_norm, part_align[int_id[m]])
                    thetaA_dist_norm = np.append(thetaA_dist_norm, ang_parts[m])
                    rA_dist = np.append(rA_dist, avg_rad)
                    nearest_surfaceA_arr = np.append(nearest_surfaceA_arr, avg_rad)
                else:
                    fa_norm=np.append(fa_norm, part_align[int_id[m]]*self.peB)
                    fa_avg=np.append(fa_avg, self.peB)
                    faB_norm=np.append(faB_norm, part_align[int_id[m]]*self.peB)
                    alignB_norm=np.append(alignB_norm, part_align[int_id[m]])
                    rB_dist_norm = np.append(rB_dist_norm, difr_parts[m]/avg_rad)
                    thetaB_dist_norm = np.append(thetaB_dist_norm, ang_parts[m])
                    rB_dist = np.append(rB_dist, avg_rad)
                    nearest_surfaceB_arr = np.append(nearest_surfaceB_arr, avg_rad)

                # Save separation distance from the nearest interface surface
                #print(avg_rad)
                r_dist_norm = np.append(r_dist_norm, difr_parts[m]/avg_rad)
                r_dist = np.append(r_dist, avg_rad)
                align_norm=np.append(align_norm, part_align[int_id[m]])

                
                theta_dist_norm = np.append(theta_dist_norm, ang_parts[m])
        """
        plt.scatter(pos_wrap_x, pos_wrap_y, s=0.9, c=fa_norm)
        plt.xlim([0, self.lx_box])
        plt.ylim([0, self.ly_box])
        plt.show()
        """
        

        import math

        plot_dict = {'rad': {'all': r_dist_norm, 'A': rA_dist_norm, 'B': rB_dist_norm}, 'theta': {'all': theta_dist_norm, 'A': thetaA_dist_norm, 'B': thetaB_dist_norm}, 'radius': {'all': nearest_surface_arr, 'A': nearest_surfaceA_arr, 'B': nearest_surfaceB_arr}}

        def round_decimals_down(number:float, decimals:int=2):
            """
            Returns a value rounded down to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.floor(number)

            factor = 10 ** decimals
            return math.floor(number * factor) / factor

        def round_decimals_up(number:float, decimals:int=2):
            """
            Returns a value rounded up to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.ceil(number)

            factor = 10 ** decimals
            return math.ceil(number * factor) / factor

        theta_bins = np.round(np.linspace(0, 360, 121),0)
        theta_bins = np.round(np.linspace(0, 360, 181),0)

        min_rad = round_decimals_down(np.min(r_dist_norm))
        max_rad = round_decimals_up(np.max(r_dist_norm))

        max_rad_str = repr(max_rad)
        if len(max_rad_str)<4:
            max_rad_str = max_rad_str + '0'

        signif_digits, fract_digits = max_rad_str.split('.')
        fract_lastdigit = int(fract_digits[-1])

        if fract_lastdigit % 2 == 1:
            max_rad += 0.01

        min_rad_str = repr(min_rad)
        if len(min_rad_str)<4:
            min_rad_str = min_rad_str + '0'
            
        signif_digits, fract_digits = min_rad_str.split('.')
        fract_lastdigit = int(fract_digits[-1])

        if fract_lastdigit % 2 == 1:
            min_rad -= 0.01

        #rad_bins = np.round(np.linspace(min_rad, max_rad, int(round(max_rad - min_rad,2)/0.02)+1),2)

        rad_final_bins = np.round(np.linspace(0.04, 1.5, int((1.46)/0.04)+1),2)
        #rad_final_bins = np.round(np.linspace(0.02, 1.5, int((1.48)/0.02)+1),2)

        theta_final_bins = np.round(np.linspace(3, 360, 120),0)
        theta_final_bins = np.round(np.linspace(2, 360, 180),0)

        min_id = np.where(rad_final_bins==min_rad)[0]
        max_id = np.where(rad_final_bins==max_rad)[0]

        if len(max_id)==1:
            max_id = int(max_id)
            if len(min_id)==1:
                min_id = int(min_id)
                rad_bins = rad_final_bins[min_id:max_id+1]
            else:
                rad_bins = rad_final_bins[0:max_id+1]
        else:
            if len(min_id)==1:
                min_id = int(min_id)
                rad_bins = rad_final_bins[min_id:]
            else:
                rad_bins = rad_final_bins[0:]
        
        fa_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        fa_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        fa_avg_real_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        
        fa_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))


        align_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        alignA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        alignB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        num_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        num_densA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        num_densB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        part_fracA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        part_fracB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        num_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        numA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        numB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        fa_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        fa_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        
        fa_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faA_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        faB_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        fa_avg_real_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        align_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        alignA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        alignB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        num_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        num_densA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        num_densB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        part_fracA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        part_fracB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        num_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        numA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        numB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))


        


        fa_sum_final_binned_theta = np.zeros(len(theta_final_bins))
        faA_sum_final_binned_theta = np.zeros(len(theta_final_bins))
        faB_sum_final_binned_theta = np.zeros(len(theta_final_bins))


        fa_avg_final_binned_theta = np.zeros(len(theta_final_bins))
        faA_avg_final_binned_theta = np.zeros(len(theta_final_bins))
        faB_avg_final_binned_theta = np.zeros(len(theta_final_bins))

        fa_avg_real_final_binned_theta = np.zeros(len(theta_final_bins))
        
        fa_dens_final_binned_theta = np.zeros(len(theta_final_bins))
        faA_dens_final_binned_theta = np.zeros(len(theta_final_bins))
        faB_dens_final_binned_theta = np.zeros(len(theta_final_bins))


        align_final_binned_theta = np.zeros(len(theta_final_bins))
        alignA_final_binned_theta = np.zeros(len(theta_final_bins))
        alignB_final_binned_theta = np.zeros(len(theta_final_bins))

        num_dens_final_binned_theta = np.zeros(len(theta_final_bins))
        num_densA_final_binned_theta = np.zeros(len(theta_final_bins))
        num_densB_final_binned_theta = np.zeros(len(theta_final_bins))

        part_fracA_final_binned_theta = np.zeros(len(theta_final_bins))
        part_fracB_final_binned_theta = np.zeros(len(theta_final_bins))

        num_final_binned_theta = np.zeros(len(theta_final_bins))
        numA_final_binned_theta = np.zeros(len(theta_final_bins))
        numB_final_binned_theta = np.zeros(len(theta_final_bins))

        fa_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))
        faA_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))
        faB_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))
        
        fa_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))
        faA_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))
        faB_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))

        fa_avg_real_final_binned_val_theta = np.zeros(len(theta_final_bins))

        align_final_binned_val_theta = np.zeros(len(theta_final_bins))
        alignA_final_binned_val_theta = np.zeros(len(theta_final_bins))
        alignB_final_binned_val_theta = np.zeros(len(theta_final_bins))

        num_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))
        num_densA_final_binned_val_theta = np.zeros(len(theta_final_bins))
        num_densB_final_binned_val_theta = np.zeros(len(theta_final_bins))

        part_fracA_final_binned_val_theta = np.zeros(len(theta_final_bins))
        part_fracB_final_binned_val_theta = np.zeros(len(theta_final_bins))

        num_final_binned_val_theta = np.zeros(len(theta_final_bins))
        numA_final_binned_val_theta = np.zeros(len(theta_final_bins))
        numB_final_binned_val_theta = np.zeros(len(theta_final_bins))

        radius_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        
        radiusA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        radiusB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        

        avg_rad_theta = np.mean(difr_ext_surface)

        sum_id = 0
        sum_id_theta = 0
        for n_theta in range(1, len(theta_bins)):

            bin_ids_theta = np.where((theta_dist_norm <theta_bins[n_theta]) & (theta_dist_norm >= theta_bins[n_theta-1]) & (r_dist_norm < 1.1) & (r_dist_norm >= 0.3))[0]
            binA_ids_theta = np.where((thetaA_dist_norm <theta_bins[n_theta]) & (thetaA_dist_norm >= theta_bins[n_theta-1]) & (rA_dist_norm < 1.1) & (rA_dist_norm >= 0.3))[0]
            binB_ids_theta = np.where((thetaB_dist_norm <theta_bins[n_theta]) & (thetaB_dist_norm >= theta_bins[n_theta-1]) & (rB_dist_norm < 1.1) & (rB_dist_norm >= 0.3))[0]
            
            if len(bin_ids_theta)>0:

                theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]
                """
                area_radial_slice = np.pi * ((1.1*np.mean(nearest_surface_arr[bin_ids_theta]))**2 - (0.3*np.mean(nearest_surface_arr[bin_ids_theta]))**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)

                fa_avg_real_final_binned_theta[theta_final_id] = np.mean(fa_avg[bin_ids_theta])

                fa_avg_final_binned_theta[theta_final_id] = np.mean(fa_norm[bin_ids_theta])
                fa_sum_final_binned_theta[theta_final_id] = np.sum(fa_norm[bin_ids_theta])

                if len(binA_ids_theta)>0:
                    faA_sum_final_binned_theta[theta_final_id] = np.sum(faA_norm[binA_ids_theta])
                if len(binB_ids_theta)>0:
                    faB_sum_final_binned_theta[theta_final_id] = np.sum(faB_norm[binB_ids_theta])

                if len(binA_ids_theta)>0:
                    faA_avg_final_binned_theta[theta_final_id] = np.mean(faA_norm[binA_ids_theta])
                if len(binB_ids_theta)>0:
                    faB_avg_final_binned_theta[theta_final_id] = np.mean(faB_norm[binB_ids_theta])

                fa_dens_final_binned_theta[theta_final_id] = np.sum(fa_norm[bin_ids_theta]) / area_radial_slice
                if len(binA_ids_theta)>0:
                    faA_dens_final_binned_theta[theta_final_id] = np.sum(faA_norm[binA_ids_theta]) / area_radial_slice
                if len(binB_ids_theta)>0:
                    faB_dens_final_binned_theta[theta_final_id] = np.sum(faB_norm[binB_ids_theta]) / area_radial_slice

                align_final_binned_theta[theta_final_id] = np.mean(align_norm[bin_ids_theta])
                if len(binA_ids_theta)>0:
                    alignA_final_binned_theta[theta_final_id] = np.mean(alignA_norm[binA_ids_theta])
                if len(binB_ids_theta)>0:
                    alignB_final_binned_theta[theta_final_id] = np.mean(alignB_norm[binB_ids_theta])

                num_dens_final_binned_theta[theta_final_id] = len(bin_ids_theta) / area_radial_slice
                if len(binA_ids_theta)>0:
                    num_densA_final_binned_theta[theta_final_id] = len(binA_ids_theta) / area_radial_slice
                if len(binB_ids_theta)>0:
                    num_densB_final_binned_theta[theta_final_id] = len(binB_ids_theta) / area_radial_slice

                num_final_binned_theta[theta_final_id] = len(bin_ids_theta)
                if len(binA_ids_theta)>0:
                    numA_final_binned_theta[theta_final_id] = len(binA_ids_theta)
                if len(binB_ids_theta)>0:
                    numB_final_binned_theta[theta_final_id] = len(binB_ids_theta)
                """
                sum_id_theta += 1

            for n_rad in range(1, len(rad_bins)):
                
                bin_ids = np.where((theta_dist_norm <theta_bins[n_theta]) & (theta_dist_norm >= theta_bins[n_theta-1]) & (r_dist_norm < rad_bins[n_rad]) & (r_dist_norm >= rad_bins[n_rad-1]))[0]
                binA_ids = np.where((thetaA_dist_norm <theta_bins[n_theta]) & (thetaA_dist_norm >= theta_bins[n_theta-1]) & (rA_dist_norm < rad_bins[n_rad]) & (rA_dist_norm >= rad_bins[n_rad-1]))[0]
                binB_ids = np.where((thetaB_dist_norm <theta_bins[n_theta]) & (thetaB_dist_norm >= theta_bins[n_theta-1]) & (rB_dist_norm < rad_bins[n_rad]) & (rB_dist_norm >= rad_bins[n_rad-1]))[0]
                
                if len(bin_ids)>0:

                    
                    rad_final_id = np.where(rad_final_bins==rad_bins[n_rad])[0]
                    theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]

                    area_radial_slice = np.pi * ((rad_bins[n_rad]*np.mean(nearest_surface_arr[bin_ids]))**2 - (rad_bins[n_rad-1]*np.mean(nearest_surface_arr[bin_ids]))**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)

                    fa_avg_real_final_binned[rad_final_id, theta_final_id] = np.mean(fa_avg[bin_ids])
                    fa_sum_final_binned[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])
                    fa_avg_final_binned[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])

                    if len(binA_ids)>0:
                        faA_sum_final_binned[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])
                    if len(binB_ids)>0:
                        faB_sum_final_binned[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])

                    if len(binA_ids)>0:
                        faA_avg_final_binned[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                    if len(binB_ids)>0:
                        faB_avg_final_binned[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                    fa_dens_final_binned[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids]) / area_radial_slice
                    if len(binA_ids)>0:
                        faA_dens_final_binned[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids]) / area_radial_slice
                    if len(binB_ids)>0:
                        faB_dens_final_binned[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids]) / area_radial_slice

                    align_final_binned[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                    if len(binA_ids)>0:
                        alignA_final_binned[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                    if len(binB_ids)>0:
                        alignB_final_binned[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                    num_dens_final_binned[rad_final_id, theta_final_id] = len(bin_ids) / area_radial_slice
                    if len(binA_ids)>0:
                        num_densA_final_binned[rad_final_id, theta_final_id] = len(binA_ids) / area_radial_slice
                    if len(binB_ids)>0:
                        num_densB_final_binned[rad_final_id, theta_final_id] = len(binB_ids) / area_radial_slice

                    if len(binA_ids)>0:
                        part_fracA_final_binned[rad_final_id, theta_final_id] = len(binA_ids) / (len(binA_ids) + len(binB_ids))
                    if len(binB_ids)>0:
                        part_fracB_final_binned[rad_final_id, theta_final_id] = len(binB_ids) / (len(binA_ids) + len(binB_ids))

                    num_final_binned[rad_final_id, theta_final_id] = len(bin_ids)
                    if len(binA_ids)>0:
                        numA_final_binned[rad_final_id, theta_final_id] = len(binA_ids)
                    if len(binB_ids)>0:
                        numB_final_binned[rad_final_id, theta_final_id] = len(binB_ids)

                    # Average nearest cluster radius for all, A, or B particles at given (r/r_c, theta) bin
                    radius_final_binned[rad_final_id, theta_final_id] = (np.mean(r_dist[bin_ids]))
                    radiusA_final_binned[rad_final_id, theta_final_id] = (np.mean(rA_dist[binA_ids]))
                    radiusB_final_binned[rad_final_id, theta_final_id] = (np.mean(rB_dist[binB_ids]))


                    sum_id += 1
        


        unique_rad = np.unique(rad_final_bins)
        unique_theta = np.unique(theta_final_bins)

        int_fa_dens_time_theta = np.zeros(len(unique_theta))
        int_faA_dens_time_theta = np.zeros(len(unique_theta))
        int_faB_dens_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_real_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_time_theta = np.zeros(len(unique_theta))
        int_faA_avg_time_theta = np.zeros(len(unique_theta))
        int_faB_avg_time_theta = np.zeros(len(unique_theta))

        int_fa_sum_time_theta = np.zeros(len(unique_theta))
        int_faA_sum_time_theta = np.zeros(len(unique_theta))
        int_faB_sum_time_theta = np.zeros(len(unique_theta))

        int_num_dens_time_theta = np.zeros(len(unique_theta))
        int_num_densA_time_theta = np.zeros(len(unique_theta))
        int_num_densB_time_theta = np.zeros(len(unique_theta))

        int_part_fracA_time_theta = np.zeros(len(unique_theta))
        int_part_fracB_time_theta = np.zeros(len(unique_theta))

        int_align_time_theta = np.zeros(len(unique_theta))
        int_alignA_time_theta = np.zeros(len(unique_theta))
        int_alignB_time_theta = np.zeros(len(unique_theta))

        int_rad_avg_time_theta = np.zeros(len(unique_theta))

        temp_id_new = np.where((unique_rad>=0.3) & (unique_rad<=1.1))[0]
        int_num = 0
        for j in range(1, len(temp_id_new)):
            test_id = np.where(rad_final_bins==unique_rad[temp_id_new[j]])[0]
            if len(test_id)>0:

                rad_step = (unique_rad[temp_id_new[j]]-unique_rad[temp_id_new[j-1]])*radius_final_binned[temp_id_new[j],:]

                int_fa_dens_time_theta += (rad_step/2) * (fa_dens_final_binned[temp_id_new[j],:]+fa_dens_final_binned[temp_id_new[j-1],:])
                int_faA_dens_time_theta += (rad_step/2) * (faA_dens_final_binned[temp_id_new[j],:]+faA_dens_final_binned[temp_id_new[j-1],:])
                int_faB_dens_time_theta += (rad_step/2) * (faB_dens_final_binned[temp_id_new[j],:]+faB_dens_final_binned[temp_id_new[j-1],:])

                int_fa_avg_real_time_theta += (rad_step/2) * (fa_avg_real_final_binned[temp_id_new[j],:]+fa_avg_real_final_binned[temp_id_new[j-1],:])

                int_fa_avg_time_theta += (rad_step/2) * (fa_avg_final_binned[temp_id_new[j],:]+fa_avg_final_binned[temp_id_new[j-1],:])
                int_faA_avg_time_theta += (rad_step/2) * (faA_avg_final_binned[temp_id_new[j],:]+faA_avg_final_binned[temp_id_new[j-1],:])
                int_faB_avg_time_theta += (rad_step/2) * (faB_avg_final_binned[temp_id_new[j],:]+faB_avg_final_binned[temp_id_new[j-1],:])

                int_fa_sum_time_theta += (rad_step/2) * (fa_sum_final_binned[temp_id_new[j],:]+fa_sum_final_binned[temp_id_new[j-1],:])
                int_faA_sum_time_theta += (rad_step/2) * (faA_sum_final_binned[temp_id_new[j],:]+faA_sum_final_binned[temp_id_new[j-1],:])
                int_faB_sum_time_theta += (rad_step/2) * (faB_sum_final_binned[temp_id_new[j],:]+faB_sum_final_binned[temp_id_new[j-1],:])

                int_rad_avg_time_theta += radius_final_binned[temp_id_new[j],:]
                int_num += 1

                int_num_dens_time_theta += (rad_step/2) * (num_dens_final_binned[temp_id_new[j],:]+num_dens_final_binned[temp_id_new[j-1],:])
                int_num_densA_time_theta += (rad_step/2) * (num_densA_final_binned[temp_id_new[j],:]+num_densA_final_binned[temp_id_new[j-1],:])
                int_num_densB_time_theta += (rad_step/2) * (num_densB_final_binned[temp_id_new[j],:]+num_densB_final_binned[temp_id_new[j-1],:])

                int_part_fracA_time_theta += (rad_step/2) * (part_fracA_final_binned[temp_id_new[j],:]+part_fracA_final_binned[temp_id_new[j-1],:])
                int_part_fracB_time_theta += (rad_step/2) * (part_fracB_final_binned[temp_id_new[j],:]+part_fracB_final_binned[temp_id_new[j-1],:])

                int_align_time_theta += (rad_step/2) * (align_final_binned[temp_id_new[j],:]+align_final_binned[temp_id_new[j-1],:])
                int_alignA_time_theta += (rad_step/2) * (alignA_final_binned[temp_id_new[j],:]+alignA_final_binned[temp_id_new[j-1],:])
                int_alignB_time_theta += (rad_step/2) * (alignB_final_binned[temp_id_new[j],:]+alignB_final_binned[temp_id_new[j-1],:])

        avg_clust_rad_theta = int_rad_avg_time_theta / int_num

        radial_heterogeneity_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned}, 'fa_sum': {'all': fa_sum_final_binned, 'A': faA_sum_final_binned, 'B': faB_sum_final_binned}, 'fa_avg': {'all': fa_avg_final_binned, 'A': faA_avg_final_binned, 'B': faB_avg_final_binned}, 'fa_dens': {'all': fa_dens_final_binned, 'A': faA_dens_final_binned, 'B': faB_dens_final_binned}, 'align': {'all': align_final_binned, 'A': alignA_final_binned, 'B': alignB_final_binned}, 'num_dens': {'all': num_dens_final_binned, 'A': num_densA_final_binned, 'B': num_densB_final_binned}, 'part_frac': {'A': part_fracA_final_binned, 'B': part_fracB_final_binned}}
        int_radial_heterogeneity_dict = {'theta': theta_final_bins, 'radius': avg_clust_rad_theta, 'fa_avg_real': {'all': int_fa_avg_real_time_theta}, 'fa_sum': {'all': int_fa_sum_time_theta, 'A': int_faA_sum_time_theta, 'B': int_faB_sum_time_theta}, 'fa_avg': {'all': int_fa_avg_time_theta, 'A': int_faA_avg_time_theta, 'B': int_faB_avg_time_theta}, 'fa_dens': {'all': int_fa_dens_time_theta, 'A': int_faA_dens_time_theta, 'B': int_faB_dens_time_theta}, 'align': {'all': int_align_time_theta, 'A': int_alignA_time_theta, 'B': int_alignB_time_theta}, 'num_dens': {'all': int_num_dens_time_theta, 'A': int_num_densA_time_theta, 'B': int_num_densB_time_theta}, 'part_frac': {'A': int_part_fracA_time_theta, 'B': int_part_fracB_time_theta}}
        #radial_heterogeneity_dict_theta = {'theta': theta_final_bins, 'fa_avg_real': {'all': fa_avg_real_final_binned_theta}, 'fa_sum': {'all': fa_sum_final_binned_theta, 'A': faA_sum_final_binned_theta, 'B': faB_sum_final_binned_theta}, 'fa_avg': {'all': fa_avg_final_binned_theta, 'A': faA_avg_final_binned_theta, 'B': faB_avg_final_binned_theta}, 'fa_dens': {'all': fa_dens_final_binned_theta, 'A': faA_dens_final_binned_theta, 'B': faB_dens_final_binned_theta}, 'align': {'all': align_final_binned_theta, 'A': alignA_final_binned_theta, 'B': alignB_final_binned_theta}, 'num_dens': {'all': num_dens_final_binned_theta, 'A': num_densA_final_binned_theta, 'B': num_densB_final_binned_theta}}
        plot_bin_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_sum': {'all': fa_sum_final_binned_val, 'A': faA_sum_final_binned_val, 'B': faB_sum_final_binned_val}, 'fa_avg': {'all': fa_avg_final_binned_val, 'A': faA_avg_final_binned_val, 'B': faB_avg_final_binned_val}, 'fa_dens': {'all': fa_dens_final_binned_val, 'A': faA_dens_final_binned_val, 'B': faB_dens_final_binned_val}, 'align': {'all': align_final_binned_val, 'A': alignA_final_binned_val, 'B': alignB_final_binned_val}, 'num': {'all': num_final_binned_val, 'A': numA_final_binned_val, 'B': numB_final_binned_val}, 'num_dens': {'all': num_dens_final_binned_val, 'A': num_densA_final_binned_val, 'B': num_densB_final_binned_val}, 'part_frac': {'A': part_fracA_final_binned_val, 'B': part_fracB_final_binned_val}}
            
        return radial_heterogeneity_dict, int_radial_heterogeneity_dict, plot_dict, plot_bin_dict

    def radial_heterogeneity_circle(self):

        # Particle IDs for A and B particles
        A_id = np.where((self.typ==0))[0]
        B_id = np.where((self.typ==1))[0]

        # Shift x-positions of particles from -h_box, h_box to 0, l_box
        pos_wrap_x = self.pos[:,0] + self.hx_box

        # Enforce periodic boundary conditions 
        test_id = np.where(pos_wrap_x>self.lx_box)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]-self.lx_box
        test_id = np.where(pos_wrap_x<0)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]+self.lx_box

        # Shift y-positions of particles from -h_box, h_box to 0, l_box
        pos_wrap_y = self.pos[:,1] + self.hy_box

        # Enforce periodic boundary conditions 
        test_id = np.where(pos_wrap_y>self.ly_box)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]-self.ly_box
        test_id = np.where(pos_wrap_y<0)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]+self.ly_box

        # Calculate radial distance of particles from cluster's center of mass
        difx_parts = (pos_wrap_x - self.hx_box)
        dify_parts = (pos_wrap_y - self.hy_box)
        difr_parts = (difx_parts ** 2 + dify_parts ** 2) ** 0.5

        # Angular location of particles
        ang_parts = np.arctan2(dify_parts, difx_parts)*(180/np.pi)

        # Convert angles from -180, 180 to 0, 360 degrees
        neg_ang = np.where(ang_parts<=0)[0]
        if len(neg_ang)>0:
            ang_parts[neg_ang] = 180 + (180+ang_parts[neg_ang])

        # Instantiate empty array for normalized separation distance of particles from cluster's center of mass
        difr_parts_norm = np.zeros(len(difr_parts))

        r_dot_p = np.zeros(self.partNum)
        difx_arr = np.zeros(self.partNum)
        dify_arr = np.zeros(self.partNum)
        difr_arr = np.zeros(self.partNum)

        # Loop over all particles
        for h in range(0, self.partNum):

            # Shifted x- and y-position of particle
            x_pos = self.pos[h,0] + self.hx_box
            y_pos = self.pos[h,1] + self.hy_box

            # Calculate separation distance from largest cluster's CoM
            difx = self.utility_functs.sep_dist_x(x_pos, self.hx_box)
            dify = self.utility_functs.sep_dist_y(y_pos, self.hy_box)
        
            difr = (difx ** 2 + dify ** 2) ** 0.5

            difx_arr[h] = difx
            dify_arr[h] = dify
            difr_arr[h] = difr
            # Alignment toward largest cluster's CoM
            r_dot_p[h] = -difx * self.px[h] + -dify * self.py[h]

        def round_decimals_down(number:float, decimals:int=2):
            """
            Returns a value rounded down to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.floor(number)

            factor = 10 ** decimals
            return math.floor(number * factor) / factor

        def round_decimals_up(number:float, decimals:int=2):
            """
            Returns a value rounded up to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.ceil(number)

            factor = 10 ** decimals
            return math.ceil(number * factor) / factor

        # Define angular bins for this time step 
        theta_bins = np.round(np.linspace(0, 360, 181),0)
        rad_bins = np.round(np.linspace(0, int(self.hx_box), int(self.hx_box/3)),0)
        print(rad_bins)
        # Define angular bins to be same as in time-averaged case
        theta_final_bins = np.round(np.linspace(2, 360, 180),0)
        rad_final_bins = np.round(np.linspace(1, int(self.hx_box), int(self.hx_box/3)-1),0)
        print(rad_final_bins)
        stop

        # Define empty arrays for binned data given radial and angular bins
        fa_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Total aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Total aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Total aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_avg_real_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Deviation of Average body force magnitude for all particles in (r/r_c, theta) bins from steady-state

        fa_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force density for all particles in (r/r_c, theta) bins from steady-state
        faA_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged aligned body force density for A particles in (r/r_c, theta) bins from steady-state
        faB_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged aligned body force density for B particles in (r/r_c, theta) bins from steady-state

        align_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))         # Deviation of Averaged alignment for all particles in (r/r_c, theta) bins from steady-state
        alignA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged alignment for A particles in (r/r_c, theta) bins from steady-state
        alignB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged alignment for B particles in (r/r_c, theta) bins from steady-state

        num_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged number density for all particles in (r/r_c, theta) bins from steady-state
        num_densA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for A particles in (r/r_c, theta) bins from steady-state
        num_densB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for B particles in (r/r_c, theta) bins from steady-state

        part_fracA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for A particles in (r/r_c, theta) bins from steady-state
        part_fracB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for B particles in (r/r_c, theta) bins from steady-state

        num_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))           # Deviation of Total number of all particles in (r/r_c, theta) bins from steady-state
        numA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))          # Deviation of Total number of A particles in (r/r_c, theta) bins from steady-state
        numB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))          # Deviation of Total number of B particles in (r/r_c, theta) bins from steady-state

        fa_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Total aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Total aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Total aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_avg_real_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))# Average body force magnitude for all particles in (r/r_c, theta) bins for current time step

        fa_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force density for all particles in (r/r_c, theta) bins for current time step
        faA_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged aligned body force density for A particles in (r/r_c, theta) bins for current time step
        faB_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged aligned body force density for B particles in (r/r_c, theta) bins for current time step

        align_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Averaged alignment for all particles in (r/r_c, theta) bins for current time step
        alignA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged alignment for A particles in (r/r_c, theta) bins for current time step
        alignB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged alignment for B particles in (r/r_c, theta) bins for current time step

        num_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged number density for all particles in (r/r_c, theta) bins for current time step
        num_densA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for A particles in (r/r_c, theta) bins for current time step
        num_densB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for B particles in (r/r_c, theta) bins for current time step

        part_fracA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for A particles in (r/r_c, theta) bins for current time step
        part_fracB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for B particles in (r/r_c, theta) bins for current time step

        num_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Total number all particles in (r/r_c, theta) bins for current time step
        numA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Total number A particles in (r/r_c, theta) bins for current time step
        numB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Total number B particles in (r/r_c, theta) bins for current time step

        radius_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        
        radiusA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        radiusB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        
        fa_sum_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Total aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_sum_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Total aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_sum_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Total aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) from steady-state


        fa_avg_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_avg_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_avg_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) from steady-state

        fa_avg_real_final_binned_theta = np.zeros(len(theta_final_bins))        # Deviation of Average body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        
        fa_dens_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force density for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average aligned body force density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average aligned body force density for B particles in (theta) bins averaged over (r/r_c) from steady-state


        align_final_binned_theta = np.zeros(len(theta_final_bins))              # Deviation of Average alignment for all particles in (theta) bins averaged over (r/r_c) from steady-state
        alignA_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average alignment for A particles in (theta) bins averaged over (r/r_c) from steady-state
        alignB_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average alignment for B particles in (theta) bins averaged over (r/r_c) from steady-state

        num_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average number density for all particles in (theta) bins averaged over (r/r_c) from steady-state
        num_densA_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        num_densB_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for B particles in (theta) bins averaged over (r/r_c) from steady-state

        part_fracA_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        part_fracB_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for B particles in (theta) bins averaged over (r/r_c) from steady-state

        num_final_binned_theta = np.zeros(len(theta_final_bins))                # Deviation of Total number of all particles in (theta) bins averaged over (r/r_c) from steady-state
        numA_final_binned_theta = np.zeros(len(theta_final_bins))               # Deviation of Total number of A particles in (theta) bins averaged over (r/r_c) from steady-state
        numB_final_binned_theta = np.zeros(len(theta_final_bins))               # Deviation of Total number of B particles in (theta) bins averaged over (r/r_c) from steady-state

        fa_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) for current time step

        fa_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Total aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Total aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Total aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) for current time step
        
        fa_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force density for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average aligned body force density for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average aligned body force density for B particles in (theta) bins averaged over (r/r_c) for current time step

        fa_avg_real_final_binned_val_theta = np.zeros(len(theta_final_bins))    # Average body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step

        align_final_binned_val_theta = np.zeros(len(theta_final_bins))          # Average alignment for all particles in (theta) bins averaged over (r/r_c) for current time step
        alignA_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average alignment for A particles in (theta) bins averaged over (r/r_c) for current time step
        alignB_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average alignment for B particles in (theta) bins averaged over (r/r_c) for current time step

        num_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average number density for all particles in (theta) bins averaged over (r/r_c) for current time step
        num_densA_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for A particles in (theta) bins averaged over (r/r_c) for current time step
        num_densB_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for B particles in (theta) bins averaged over (r/r_c) for current time step

        part_fracA_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for A particles in (theta) bins averaged over (r/r_c) for current time step
        part_fracB_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for B particles in (theta) bins averaged over (r/r_c) for current time step

        num_final_binned_val_theta = np.zeros(len(theta_final_bins))            # Total number all particles in (theta) bins averaged over (r/r_c) for current time step
        numA_final_binned_val_theta = np.zeros(len(theta_final_bins))           # Total number A particles in (theta) bins averaged over (r/r_c) for current time step
        numB_final_binned_val_theta = np.zeros(len(theta_final_bins))           # Total number B particles in (theta) bins averaged over (r/r_c) for current time step

        radius_final_binned_theta_new = np.zeros(len(theta_final_bins))         # Average cluster radius in (theta) bins averaged over (r/r_c) for current time step

        avg_rad_theta = np.mean(difr_ext_surface)

        sum_id = 0
        sum_id_theta = 0

        # Loop over select time-averaged angular bins within current min and max angles
        for n_theta in range(1, len(theta_bins)):

            # Loop over select time-averaged radial bins within current min and max radii
            for n_rad in range(1, len(rad_bins)):

                # Find all, A, and B particle IDs currently within time-averaged bin
                bin_ids = np.where((ang_parts <theta_bins[n_theta]) & (ang_parts >= theta_bins[n_theta-1]) & (difr_arr < rad_bins[n_rad]) & (difr_arr >= rad_bins[n_rad-1]))[0]
                binA_ids = np.where((ang_parts <theta_bins[n_theta]) & (ang_parts >= theta_bins[n_theta-1]) & (difr_arr < rad_bins[n_rad]) & (difr_arr >= rad_bins[n_rad-1]) & (self.typ==0))[0]
                binB_ids = np.where((ang_parts <theta_bins[n_theta]) & (ang_parts >= theta_bins[n_theta-1]) & (difr_arr < rad_bins[n_rad]) & (difr_arr >= rad_bins[n_rad-1]) & (self.typ==1))[0]
                
                # If particles within bin, calculate property
                if len(bin_ids)>0:
                    """
                    rad_final_id = np.where(rad_final_bins==rad_bins[n_rad])[0]
                    theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]

                    area_radial_slice = np.pi * ((rad_bins[n_rad]*avg_rad_theta)**2 - (rad_bins[n_rad-1]*avg_rad_theta)**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)

                    fa_final_binned[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                    faA_final_binned[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                    faB_final_binned[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                    align_final_binned[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                    alignA_final_binned[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                    alignB_final_binned[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                    num_dens_final_binned[rad_final_id, theta_final_id] = len(bin_ids) / area_radial_slice
                    num_densA_final_binned[rad_final_id, theta_final_id] = len(binA_ids) / area_radial_slice
                    num_densB_final_binned[rad_final_id, theta_final_id] = len(binB_ids) / area_radial_slice

                    num_final_binned[rad_final_id, theta_final_id] = len(bin_ids)
                    numA_final_binned[rad_final_id, theta_final_id] = len(binA_ids)
                    numB_final_binned[rad_final_id, theta_final_id] = len(binB_ids)
                    """
                    
                    # Find ID of time-averaged bin arrays
                    rad_final_id = np.where(rad_final_bins==rad_bins[n_rad])[0]
                    theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]
                    
                    # Current area of radial slice
                    area_radial_slice = np.pi * ((rad_bins[n_rad]*np.mean(nearest_surface_arr[bin_ids]))**2 - (rad_bins[n_rad-1]*np.mean(nearest_surface_arr[bin_ids]))**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)
                    
                    # If non-zero, continue...
                    if area_radial_slice > 0:

                        # If load saved file for averages...
                        if load_save == 1:
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_avg_real_final_binned[rad_final_id, theta_final_id] = ((np.mean(fa_avg[bin_ids])) - float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_real_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_avg[bin_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0])>0:
                                fa_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(fa_norm[bin_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0]))
                                fa_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faA_norm[binA_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0]))
                                faA_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faB_norm[binB_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0]))
                                faB_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])/ area_radial_slice

                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(fa_norm[bin_ids]) - float(avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id][0]))
                                fa_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faA_norm[binA_ids]) - float(avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id][0]))
                                faA_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faB_norm[binB_ids]) - float(avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id][0]))
                                faB_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(fa_norm[bin_ids]) - float(avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faA_norm[binA_ids]) - float(avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id][0]))
                                faA_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faB_norm[binB_ids]) - float(avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id][0]))
                                faB_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['align']['all'][rad_final_id, theta_final_id][0])>0:
                                align_final_binned[rad_final_id, theta_final_id] = (np.mean(align_norm[bin_ids]) - float(avg_rad_dict['align']['all'][rad_final_id, theta_final_id][0]))
                                align_final_binned_val[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['align']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                alignA_final_binned[rad_final_id, theta_final_id] = (np.mean(alignA_norm[binA_ids]) - float(avg_rad_dict['align']['A'][rad_final_id, theta_final_id][0]))
                                alignA_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['align']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                alignB_final_binned[rad_final_id, theta_final_id] = (np.mean(alignB_norm[binB_ids]) - float(avg_rad_dict['align']['B'][rad_final_id, theta_final_id][0]))
                                alignB_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                            # Average nearest cluster radius for all, A, or B particles at given (r/r_c, theta) bin
                            radius_final_binned[rad_final_id, theta_final_id] = (np.mean(r_dist[bin_ids]))
                            radiusA_final_binned[rad_final_id, theta_final_id] = (np.mean(rA_dist[binA_ids]))
                            radiusB_final_binned[rad_final_id, theta_final_id] = (np.mean(rB_dist[binB_ids]))


                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id][0])>0:
                                num_dens_final_binned[rad_final_id, theta_final_id] = ((len(bin_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id][0]))
                                num_dens_final_binned_val[rad_final_id, theta_final_id] = (len(bin_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                num_densA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id][0]))
                                num_densA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                num_densB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id][0]))
                                num_densB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / area_radial_slice)

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                part_fracA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / (len(binA_ids) + len(binB_ids))) - float(avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id][0]))
                                part_fracA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / (len(binA_ids) + len(binB_ids)))
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                part_fracB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / (len(binA_ids) + len(binB_ids))) - float(avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id][0]))
                                part_fracB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / (len(binA_ids) + len(binB_ids)))

                        # If did not load save file for averages
                        else:
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average body force magnitude for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0])>0:
                                fa_avg_real_final_binned[rad_final_id, theta_final_id] = ((np.mean(fa_avg[bin_ids])) - float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_real_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_avg[bin_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id]>0:
                                fa_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(fa_norm[bin_ids]) - avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id])
                                fa_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                faA_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faA_norm[binA_ids]) - avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id])
                                faA_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                faB_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faB_norm[binB_ids]) - avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id])
                                faB_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id]>0:
                                fa_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(fa_norm[bin_ids]) - avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id])
                                fa_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                faA_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faA_norm[binA_ids]) - avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id])
                                faA_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                faB_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faB_norm[binB_ids]) - avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id])
                                faB_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0]>0:
                                fa_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(fa_norm[bin_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0])
                                fa_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0]>0) & (len(binA_ids)>0):
                                faA_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faA_norm[binA_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0])
                                faA_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0]>0) & (len(binB_ids)>0):
                                faB_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faB_norm[binB_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0])
                                faB_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])/ area_radial_slice

                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['align']['all'][rad_final_id, theta_final_id]>0:
                                align_final_binned[rad_final_id, theta_final_id] = (np.mean(align_norm[bin_ids]) - avg_rad_dict['align']['all'][rad_final_id, theta_final_id])
                                align_final_binned_val[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['align']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                alignA_final_binned[rad_final_id, theta_final_id] = (np.mean(alignA_norm[binA_ids]) - avg_rad_dict['align']['A'][rad_final_id, theta_final_id])
                                alignA_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['align']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                alignB_final_binned[rad_final_id, theta_final_id] = (np.mean(alignB_norm[binB_ids]) - avg_rad_dict['align']['B'][rad_final_id, theta_final_id])
                                alignB_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                            # Average nearest cluster radius for all, A, or B particles at given (r/r_c, theta) bin
                            radius_final_binned[rad_final_id, theta_final_id] = (np.mean(r_dist[bin_ids]))
                            radiusA_final_binned[rad_final_id, theta_final_id] = (np.mean(rA_dist[binA_ids]))
                            radiusB_final_binned[rad_final_id, theta_final_id] = (np.mean(rB_dist[binB_ids]))

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id]>0:
                                num_dens_final_binned[rad_final_id, theta_final_id] = ((len(bin_ids) / area_radial_slice) - avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id])
                                num_dens_final_binned_val[rad_final_id, theta_final_id] = (len(bin_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                num_densA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / area_radial_slice) - avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id])
                                num_densA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                num_densB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / area_radial_slice) - avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id])
                                num_densB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / area_radial_slice)

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                part_fracA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / (len(binB_ids) + len(binA_ids)))- avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id])
                                part_fracA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / (len(binB_ids) + len(binA_ids)))
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                part_fracB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / (len(binB_ids) + len(binA_ids))) - avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id])
                                part_fracB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / (len(binB_ids) + len(binA_ids)))

        #print(avg_rad_dict['rad'])
        #stop
        #unique_rad = np.unique(avg_rad_dict['rad'])
        
        unique_theta = np.unique(avg_rad_dict['theta'])

        int_fa_dens_time_theta = np.zeros(len(unique_theta))
        int_faA_dens_time_theta = np.zeros(len(unique_theta))
        int_faB_dens_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_real_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_time_theta = np.zeros(len(unique_theta))
        int_faA_avg_time_theta = np.zeros(len(unique_theta))
        int_faB_avg_time_theta = np.zeros(len(unique_theta))

        int_fa_sum_time_theta = np.zeros(len(unique_theta))
        int_faA_sum_time_theta = np.zeros(len(unique_theta))
        int_faB_sum_time_theta = np.zeros(len(unique_theta))

        int_num_dens_time_theta = np.zeros(len(unique_theta))
        int_num_densA_time_theta = np.zeros(len(unique_theta))
        int_num_densB_time_theta = np.zeros(len(unique_theta))

        int_part_fracA_time_theta = np.zeros(len(unique_theta))
        int_part_fracB_time_theta = np.zeros(len(unique_theta))

        int_align_time_theta = np.zeros(len(unique_theta))
        int_alignA_time_theta = np.zeros(len(unique_theta))
        int_alignB_time_theta = np.zeros(len(unique_theta))
        temp_id_new = np.where((rad_final_bins>=0.3) & (rad_final_bins<=1.1))[0]

        for j in range(1, len(temp_id_new)):
            test_id = np.where(avg_rad_dict['rad']==rad_final_bins[temp_id_new[j]])[0]
            if len(test_id)>0:

                rad_step = (rad_final_bins[temp_id_new[j]]-rad_final_bins[temp_id_new[j-1]])*radius_final_binned[temp_id_new[j],:]

                int_fa_dens_time_theta += (rad_step/2) * (fa_dens_final_binned_val[temp_id_new[j],:]+fa_dens_final_binned_val[temp_id_new[j-1],:])
                int_faA_dens_time_theta += (rad_step/2) * (faA_dens_final_binned_val[temp_id_new[j],:]+faA_dens_final_binned_val[temp_id_new[j-1],:])
                int_faB_dens_time_theta += (rad_step/2) * (faB_dens_final_binned_val[temp_id_new[j],:]+faB_dens_final_binned_val[temp_id_new[j-1],:])

                int_fa_avg_real_time_theta += (rad_step/2) * (fa_avg_real_final_binned_val[temp_id_new[j],:]+fa_avg_real_final_binned_val[temp_id_new[j-1],:])

                int_fa_avg_time_theta += (rad_step/2) * (fa_avg_final_binned_val[temp_id_new[j],:]+fa_avg_final_binned_val[temp_id_new[j-1],:])
                int_faA_avg_time_theta += (rad_step/2) * (faA_avg_final_binned_val[temp_id_new[j],:]+faA_avg_final_binned_val[temp_id_new[j-1],:])
                int_faB_avg_time_theta += (rad_step/2) * (faB_avg_final_binned_val[temp_id_new[j],:]+faB_avg_final_binned_val[temp_id_new[j-1],:])

                int_fa_sum_time_theta += (rad_step/2) * (fa_sum_final_binned_val[temp_id_new[j],:]+fa_sum_final_binned_val[temp_id_new[j-1],:])
                int_faA_sum_time_theta += (rad_step/2) * (faA_sum_final_binned_val[temp_id_new[j],:]+faA_sum_final_binned_val[temp_id_new[j-1],:])
                int_faB_sum_time_theta += (rad_step/2) * (faB_sum_final_binned_val[temp_id_new[j],:]+faB_sum_final_binned_val[temp_id_new[j-1],:])

                int_num_dens_time_theta += (rad_step/2) * (num_dens_final_binned_val[temp_id_new[j],:]+num_dens_final_binned_val[temp_id_new[j-1],:])
                int_num_densA_time_theta += (rad_step/2) * (num_densA_final_binned_val[temp_id_new[j],:]+num_densA_final_binned_val[temp_id_new[j-1],:])
                int_num_densB_time_theta += (rad_step/2) * (num_densB_final_binned_val[temp_id_new[j],:]+num_densB_final_binned_val[temp_id_new[j-1],:])

                int_part_fracA_time_theta += (rad_step/2) * (part_fracA_final_binned_val[temp_id_new[j],:]+part_fracA_final_binned_val[temp_id_new[j-1],:])
                int_part_fracB_time_theta += (rad_step/2) * (part_fracB_final_binned_val[temp_id_new[j],:]+part_fracB_final_binned_val[temp_id_new[j-1],:])

                int_align_time_theta += (rad_step/2) * (align_final_binned_val[temp_id_new[j],:]+align_final_binned_val[temp_id_new[j-1],:])
                int_alignA_time_theta += (rad_step/2) * (alignA_final_binned_val[temp_id_new[j],:]+alignA_final_binned_val[temp_id_new[j-1],:])
                int_alignB_time_theta += (rad_step/2) * (alignB_final_binned_val[temp_id_new[j],:]+alignB_final_binned_val[temp_id_new[j-1],:])

        dif_int_fa_dens_time_theta = int_fa_dens_time_theta - integrated_avg_rad_dict['fa_dens']['all'].astype('float64')
        dif_int_faA_dens_time_theta = int_faA_dens_time_theta - integrated_avg_rad_dict['fa_dens']['A']
        dif_int_faB_dens_time_theta = int_faB_dens_time_theta - integrated_avg_rad_dict['fa_dens']['B']

        dif_int_fa_avg_real_time_theta = int_fa_avg_real_time_theta - integrated_avg_rad_dict['radius']

        dif_int_fa_avg_time_theta = int_fa_avg_time_theta - integrated_avg_rad_dict['fa_avg']['all']
        dif_int_faA_avg_time_theta = int_faA_avg_time_theta - integrated_avg_rad_dict['fa_avg']['A']
        dif_int_faB_avg_time_theta = int_faB_avg_time_theta - integrated_avg_rad_dict['fa_avg']['B']

        dif_int_fa_sum_time_theta = int_fa_sum_time_theta - integrated_avg_rad_dict['fa_sum']['all']
        dif_int_faA_sum_time_theta = int_faA_sum_time_theta - integrated_avg_rad_dict['fa_sum']['A']
        dif_int_faB_sum_time_theta = int_faB_sum_time_theta - integrated_avg_rad_dict['fa_sum']['B']

        dif_int_num_dens_time_theta = int_num_dens_time_theta - integrated_avg_rad_dict['num_dens']['all']
        dif_int_num_densA_time_theta = int_num_densA_time_theta - integrated_avg_rad_dict['num_dens']['A']
        dif_int_num_densB_time_theta = int_num_densB_time_theta - integrated_avg_rad_dict['num_dens']['B']

        dif_int_part_fracA_time_theta = int_part_fracA_time_theta - integrated_avg_rad_dict['part_frac']['A']
        dif_int_part_fracB_time_theta = int_part_fracB_time_theta - integrated_avg_rad_dict['part_frac']['B']

        dif_int_align_time_theta = int_align_time_theta - integrated_avg_rad_dict['align']['all']
        dif_int_alignA_time_theta = int_alignA_time_theta - integrated_avg_rad_dict['align']['A']
        dif_int_alignB_time_theta = int_alignB_time_theta - integrated_avg_rad_dict['align']['B']





        dif_avg_int_fa_dens_time_theta = int_fa_dens_time_theta - np.mean(int_fa_dens_time_theta)
        dif_avg_int_faA_dens_time_theta = int_faA_dens_time_theta - np.mean(int_faA_dens_time_theta)
        dif_avg_int_faB_dens_time_theta = int_faB_dens_time_theta - np.mean(int_faB_dens_time_theta)

        dif_avg_int_fa_avg_real_time_theta = int_fa_avg_real_time_theta - np.mean(int_fa_avg_real_time_theta)

        dif_avg_int_fa_avg_time_theta = int_fa_avg_time_theta - np.mean(int_fa_avg_time_theta)
        dif_avg_int_faA_avg_time_theta = int_faA_avg_time_theta - np.mean(int_faA_avg_time_theta)
        dif_avg_int_faB_avg_time_theta = int_faB_avg_time_theta - np.mean(int_faB_avg_time_theta)

        dif_avg_int_fa_sum_time_theta = int_fa_sum_time_theta - np.mean(int_fa_sum_time_theta)
        dif_avg_int_faA_sum_time_theta = int_faA_sum_time_theta - np.mean(int_faA_sum_time_theta)
        dif_avg_int_faB_sum_time_theta = int_faB_sum_time_theta - np.mean(int_faB_sum_time_theta)

        dif_avg_int_num_dens_time_theta = int_num_dens_time_theta - np.mean(int_num_dens_time_theta)
        dif_avg_int_num_densA_time_theta = int_num_densA_time_theta - np.mean(int_num_densA_time_theta)
        dif_avg_int_num_densB_time_theta = int_num_densB_time_theta - np.mean(int_num_densB_time_theta)

        dif_avg_int_part_fracA_time_theta = int_part_fracA_time_theta - np.mean(int_part_fracA_time_theta)
        dif_avg_int_part_fracB_time_theta = int_part_fracB_time_theta - np.mean(int_part_fracB_time_theta)

        dif_avg_int_align_time_theta = int_align_time_theta - np.mean(int_align_time_theta)
        dif_avg_int_alignA_time_theta = int_alignA_time_theta - np.mean(int_alignA_time_theta)
        dif_avg_int_alignB_time_theta = int_alignB_time_theta - np.mean(int_alignB_time_theta)


        radial_heterogeneity_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'radius_ang': radius_final_binned_theta_new, 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned}, 'fa_avg': {'all': fa_avg_final_binned, 'A': faA_avg_final_binned, 'B': faB_avg_final_binned}, 'fa_sum': {'all': fa_sum_final_binned, 'A': faA_sum_final_binned, 'B': faB_sum_final_binned}, 'fa_dens': {'all': fa_dens_final_binned, 'A': faA_dens_final_binned, 'B': faB_dens_final_binned}, 'align': {'all': align_final_binned, 'A': alignA_final_binned, 'B': alignB_final_binned}, 'num_dens': {'all': num_dens_final_binned, 'A': num_densA_final_binned, 'B': num_densB_final_binned}, 'part_frac': {'A': part_fracA_final_binned, 'B': part_fracB_final_binned}}
        #radial_heterogeneity_dict_theta = {'theta': theta_final_bins, 'radius': radius_final_binned_theta_new, 'fa_avg_real': {'all': fa_avg_real_final_binned_theta}, 'fa_sum': {'all': fa_sum_final_binned_theta, 'A': faA_sum_final_binned_theta, 'B': faB_sum_final_binned_theta}, 'fa_avg': {'all': fa_avg_final_binned_theta, 'A': faA_avg_final_binned_theta, 'B': faB_avg_final_binned_theta}, 'fa_dens': {'all': fa_dens_final_binned_theta, 'A': faA_dens_final_binned_theta, 'B': faB_dens_final_binned_theta}, 'align': {'all': align_final_binned_theta, 'A': alignA_final_binned_theta, 'B': alignB_final_binned_theta}, 'num_dens': {'all': num_dens_final_binned_theta, 'A': num_densA_final_binned_theta, 'B': num_densB_final_binned_theta}}
        
        plot_bin_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'radius_ang': radius_final_binned_theta_new, 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned_val}, 'fa_avg': {'all': fa_avg_final_binned_val, 'A': faA_avg_final_binned_val, 'B': faB_avg_final_binned_val}, 'fa_dens': {'all': fa_dens_final_binned_val, 'A': faA_dens_final_binned_val, 'B': faB_dens_final_binned_val}, 'align': {'all': align_final_binned_val, 'A': alignA_final_binned_val, 'B': alignB_final_binned_val}, 'num': {'all': num_final_binned_val, 'A': numA_final_binned_val, 'B': numB_final_binned_val}, 'num_dens': {'all': num_dens_final_binned_val, 'A': num_densA_final_binned_val, 'B': num_densB_final_binned_val}, 'part_frac': {'A': part_fracA_final_binned_val, 'B': part_fracB_final_binned_val}, 'rad_bin': {'all': radius_final_binned, 'A': radiusA_final_binned, 'B': radiusB_final_binned}}

        dif_radial_heterogeneity_dict = {'theta': theta_final_bins, 'fa_avg_real': {'all': dif_int_fa_avg_real_time_theta}, 'fa_avg': {'all': dif_int_fa_avg_time_theta, 'A': dif_int_faA_avg_time_theta, 'B': dif_int_faB_avg_time_theta}, 'fa_sum': {'all': dif_int_fa_sum_time_theta, 'A': dif_int_faA_sum_time_theta, 'B': dif_int_faB_sum_time_theta}, 'fa_dens': {'all': dif_int_fa_dens_time_theta, 'A': dif_int_faA_dens_time_theta, 'B': dif_int_faB_dens_time_theta}, 'align': {'all': dif_int_align_time_theta, 'A': dif_int_alignA_time_theta, 'B': dif_int_alignB_time_theta}, 'num_dens': {'all': dif_int_num_dens_time_theta, 'A': dif_int_num_densA_time_theta, 'B': dif_int_num_densB_time_theta}, 'part_frac': {'A': dif_int_part_fracA_time_theta, 'B': dif_int_part_fracB_time_theta}}

        dif_avg_radial_heterogeneity_dict = {'theta': theta_final_bins, 'fa_avg_real': {'all': dif_avg_int_fa_avg_real_time_theta}, 'fa_avg': {'all': dif_avg_int_fa_avg_time_theta, 'A': dif_avg_int_faA_avg_time_theta, 'B': dif_avg_int_faB_avg_time_theta}, 'fa_sum': {'all': dif_avg_int_fa_sum_time_theta, 'A': dif_avg_int_faA_sum_time_theta, 'B': dif_avg_int_faB_sum_time_theta}, 'fa_dens': {'all': dif_avg_int_fa_dens_time_theta, 'A': dif_avg_int_faA_dens_time_theta, 'B': dif_avg_int_faB_dens_time_theta}, 'align': {'all': dif_avg_int_align_time_theta, 'A': dif_avg_int_alignA_time_theta, 'B': dif_avg_int_alignB_time_theta}, 'num_dens': {'all': dif_avg_int_num_dens_time_theta, 'A': dif_avg_int_num_densA_time_theta, 'B': dif_avg_int_num_densB_time_theta}, 'part_frac': {'A': dif_avg_int_part_fracA_time_theta, 'B': dif_avg_int_part_fracB_time_theta}}

        return radial_heterogeneity_dict, dif_radial_heterogeneity_dict, dif_avg_radial_heterogeneity_dict, plot_dict, plot_bin_dict


    def radial_heterogeneity(self, method2_align_dict, avg_rad_dict, integrated_avg_rad_dict, surface_dict, int_comp_dict, all_surface_measurements, int_dict, phase_dict, load_save = 0):
        
        # Bulk particle IDs for all, A, and B particles
        bulk_id = np.where(phase_dict['part']==0)[0]
        bulk_A_id = np.where((phase_dict['part']==0) & (self.typ==0))[0]
        bulk_B_id = np.where((phase_dict['part']==0) & (self.typ==1))[0]

        # Interface particle IDs for all, A, and B particles
        int_id = np.where(phase_dict['part']==1)[0]
        int_A_id = np.where((phase_dict['part']==1) & (self.typ==0))[0]
        int_B_id = np.where((phase_dict['part']==1) & (self.typ==1))[0]

        # Gas particle IDs for all, A, and B particles
        gas_id = np.where(phase_dict['part']==2)[0]
        gas_A_id = np.where((phase_dict['part']==2) & (self.typ==0))[0]
        gas_B_id = np.where((phase_dict['part']==2) & (self.typ==1))[0]

        # Particle IDs for A and B particles
        A_id = np.where((self.typ==0))[0]
        B_id = np.where((self.typ==1))[0]

        try:
            int_id = np.where(int_dict['part']==int_dict['largest ids'][0])[0]
            key = 'surface id ' + str(int(int_dict['largest ids'][0]))
        except:
            interface_id = int(np.where(int_comp_dict['comp']==np.max(int_comp_dict['comp']))[0])
            interface_id = int_comp_dict['ids'][interface_id]
            int_id = np.where(int_dict['part']==interface_id)[0]
            key = 'surface id ' + str(int(interface_id))
        
        # Exterior cluster surface radii of surface points
        difx_ext_surface = (surface_dict[key]['exterior']['pos']['x'] - all_surface_measurements[key]['exterior']['com']['x'])
        dify_ext_surface = (surface_dict[key]['exterior']['pos']['y'] - all_surface_measurements[key]['exterior']['com']['y'])
        difr_ext_surface = (difx_ext_surface ** 2 + dify_ext_surface ** 2) ** 0.5

        # Angular location of surface points
        ang_ext_surface = np.arctan2(dify_ext_surface, difx_ext_surface)*(180/np.pi)

        # Convert angles from -180, 180 to 0, 360 degrees
        neg_ang = np.where(ang_ext_surface<=0)[0]
        if len(neg_ang)>0:
            ang_ext_surface[neg_ang] = 180 + (180+ang_ext_surface[neg_ang])
        
        # Shift x-positions of particles from -h_box, h_box to 0, l_box
        pos_wrap_x = self.pos[int_id,0] + self.hx_box

        # Enforce periodic boundary conditions 
        test_id = np.where(pos_wrap_x>self.lx_box)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]-self.lx_box
        test_id = np.where(pos_wrap_x<0)[0]
        if len(test_id)>0:
            pos_wrap_x[test_id] = pos_wrap_x[test_id]+self.lx_box

        # Shift y-positions of particles from -h_box, h_box to 0, l_box
        pos_wrap_y = self.pos[int_id,1] + self.hy_box

        # Enforce periodic boundary conditions 
        test_id = np.where(pos_wrap_y>self.ly_box)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]-self.ly_box
        test_id = np.where(pos_wrap_y<0)[0]
        if len(test_id)>0:
            pos_wrap_y[test_id] = pos_wrap_y[test_id]+self.ly_box

        # Calculate radial distance of particles from cluster's center of mass
        difx_parts = (pos_wrap_x - all_surface_measurements[key]['exterior']['com']['x'])
        dify_parts = (pos_wrap_y - all_surface_measurements[key]['exterior']['com']['y'])
        difr_parts = (difx_parts ** 2 + dify_parts ** 2) ** 0.5

        # Angular location of particles
        ang_parts = np.arctan2(dify_parts, difx_parts)*(180/np.pi)

        # Convert angles from -180, 180 to 0, 360 degrees
        neg_ang = np.where(ang_parts<=0)[0]
        if len(neg_ang)>0:
            ang_parts[neg_ang] = 180 + (180+ang_parts[neg_ang])

        # Instantiate empty array for normalized separation distance of particles from cluster's center of mass
        difr_parts_norm = np.zeros(len(difr_parts))

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty arrays
        fa_norm = np.array([])          #Aligned body force magnitude for all particles
        faA_norm = np.array([])         #Aligned body force magnitude for A particles
        faB_norm = np.array([])         #Aligned body force magnitude for B particles

        fa_avg = np.array([])           #Body force magnitude for all particles

        r_dist_norm = np.array([])      #All particle distance from cluster CoM normalized by nearest cluster radius
        rA_dist_norm = np.array([])     #A particle distance from cluster CoM normalized by nearest cluster radius
        rB_dist_norm = np.array([])     #B particle distance from cluster CoM normalized by nearest cluster radius

        align_norm = np.array([])       #All particle alignment
        alignA_norm = np.array([])      #A particle alignment
        alignB_norm = np.array([])      #B particle alignment

        theta_dist_norm = np.array([])  # Angle around CoM for all particles
        thetaA_dist_norm = np.array([]) # Angle around CoM for A particles
        thetaB_dist_norm = np.array([]) # Angle around CoM for B particles

        r_dist = np.array([])           # Cluster radius of nearest surface for all particles
        rA_dist = np.array([])          # Cluster radius of nearest surface for A particles
        rB_dist = np.array([])          # Cluster radius of nearest surface for B particles

        nearest_surface_arr = np.array([])      # Cluster radius of nearest surface for all particles
        nearest_surfaceA_arr = np.array([])     # Cluster radius of nearest surface for A particles
        nearest_surfaceB_arr = np.array([])     # Cluster radius of nearest surface for B particles

        # Loop over particles to calculate each of their alignment, number density, and active forces
        for m in range(0, len(pos_wrap_x)):
            
            # Find nearest surfaces within +/- 1 degree of particle angular position
            near_surfaces = np.where( (ang_ext_surface>= ang_parts[m]-1.0)& (ang_ext_surface<= ang_parts[m]+1.0) )[0]
            
            # If nearest surfaces, then calculate properties
            if len(near_surfaces) > 0:

                # Calculate x, y, and r separation distance from nearest surfaces
                difx_surface_part = (surface_dict[key]['exterior']['pos']['x'][near_surfaces] - pos_wrap_x[m])
                dify_surface_part  = (surface_dict[key]['exterior']['pos']['y'][near_surfaces] - pos_wrap_y[m])
                difr_surface_part  = np.mean((difx_surface_part ** 2 + dify_surface_part ** 2) ** 0.5)

                # Calculate and save average custer radius of nearest surfaces
                avg_rad = np.mean(difr_ext_surface[near_surfaces])
                nearest_surface_arr = np.append(nearest_surface_arr, avg_rad)

                # Calculate particle's distance from cluster center of mass normalized by nearest cluster radius
                difr_parts_norm[m] = difr_parts[m]/avg_rad

                # Save particle properties if A particle
                if self.typ[int_id[m]] == 0:

                    # Aligned active force magnitude for all particles
                    fa_norm=np.append(fa_norm, part_align[int_id[m]]*self.peA)

                    # Active force magnitude for all particles
                    fa_avg=np.append(fa_avg, self.peA)

                    # Aligned active force magnitude for A particles
                    faA_norm=np.append(faA_norm, part_align[int_id[m]]*self.peA)

                    # A particle's distance from cluster center of mass normalized by nearest cluster radius
                    rA_dist_norm = np.append(rA_dist_norm, difr_parts[m]/avg_rad)

                    # Cluster radius of nearest surfaces for A particles
                    rA_dist = np.append(rA_dist, avg_rad)

                    # Alignment of A particles
                    alignA_norm=np.append(alignA_norm, part_align[int_id[m]])

                    # Angular position of A particles
                    thetaA_dist_norm = np.append(thetaA_dist_norm, ang_parts[m])

                    # Cluster radius of nearest surfaces for A particles
                    nearest_surfaceA_arr = np.append(nearest_surfaceA_arr, avg_rad)

                # Save particle properties if B particle
                else:

                    # Aligned active force magnitude for all particles
                    fa_norm=np.append(fa_norm, part_align[int_id[m]]*self.peB)

                    # Active force magnitude for all particles
                    fa_avg=np.append(fa_avg, self.peB)

                    # Aligned active force magnitude for B particles
                    faB_norm=np.append(faB_norm, part_align[int_id[m]]*self.peB)

                    # Cluster radius of nearest surfaces for B particles
                    rB_dist = np.append(rB_dist, avg_rad)

                    # Alignment of B particles
                    alignB_norm=np.append(alignB_norm, part_align[int_id[m]])

                    # B particle's distance from cluster center of mass normalized by nearest cluster radius
                    rB_dist_norm = np.append(rB_dist_norm, difr_parts[m]/avg_rad)

                    # Angular position of B particles
                    thetaB_dist_norm = np.append(thetaB_dist_norm, ang_parts[m])

                    # Cluster radius of nearest surfaces for B particles
                    nearest_surfaceB_arr = np.append(nearest_surfaceB_arr, avg_rad)

                # All particle's distance from cluster center of mass normalized by nearest cluster radius
                r_dist_norm = np.append(r_dist_norm, difr_parts[m]/avg_rad)

                # Cluster radius of nearest surfaces for all particles
                r_dist = np.append(r_dist, avg_rad)

                # Alignment of all particles
                align_norm=np.append(align_norm, part_align[int_id[m]])

                # Angular position of all particles
                theta_dist_norm = np.append(theta_dist_norm, ang_parts[m])
        import math
        
        # Define dictionary for plotting radial heterogeneity
        plot_dict = {'rad': {'all': r_dist_norm, 'A': rA_dist_norm, 'B': rB_dist_norm}, 'theta': {'all': theta_dist_norm, 'A': thetaA_dist_norm, 'B': thetaB_dist_norm}, 'radius': {'all': nearest_surface_arr, 'A': nearest_surfaceA_arr, 'B': nearest_surfaceB_arr}}


        def round_decimals_down(number:float, decimals:int=2):
            """
            Returns a value rounded down to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.floor(number)

            factor = 10 ** decimals
            return math.floor(number * factor) / factor

        def round_decimals_up(number:float, decimals:int=2):
            """
            Returns a value rounded up to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.ceil(number)

            factor = 10 ** decimals
            return math.ceil(number * factor) / factor

        # Define angular bins for this time step 
        theta_bins = np.round(np.linspace(0, 360, 181),0)

        # Find min and max particle distances cluster center of mass normalized by cluster radius
        min_rad = round_decimals_down(np.min(r_dist_norm))
        max_rad = round_decimals_up(np.max(r_dist_norm))

        # Convert maximum position to a string to make uniform length to other time steps
        max_rad_str = repr(max_rad)
        if len(max_rad_str)<4:
            max_rad_str = max_rad_str + '0'
        signif_digits, fract_digits = max_rad_str.split('.')
        fract_lastdigit = int(fract_digits[-1])

        # If odd decimal, make even decimal
        if fract_lastdigit % 2 == 1:
            max_rad += 0.01
        
        # Round to nearest 2nd decimal place
        max_rad = round(max_rad,2)

        # Convert minimum position to a string to make uniform length to other time steps
        min_rad_str = repr(min_rad)
        if len(min_rad_str)<4:
            min_rad_str = min_rad_str + '0'
        signif_digits, fract_digits = min_rad_str.split('.')
        fract_lastdigit = int(fract_digits[-1])

        # If odd decimal, make even decimal
        if fract_lastdigit % 2 == 1:
            min_rad -= 0.01

        # Round to nearest 2nd decimal place
        min_rad = round(min_rad,2)
        
        # Define radial bins for this time step 
        rad_final_bins = np.round(np.linspace(0.04, 1.5, int((1.46)/0.04)+1),2)
        
        # Define angular bins to be same as in time-averaged case
        theta_final_bins = np.round(np.linspace(2, 360, 180),0)

        # Find ID of min and maximum radial locations
        min_id = np.where(rad_final_bins==min_rad)[0]
        max_id = np.where(rad_final_bins==max_rad)[0]

        # Defines select array from the time-averaged radial position array to look at based on current min and max positions
        # If maximum ID exists, only look at time-averaged bins less than it
        if len(max_id)==1:
            max_id = int(max_id)

            # If minimum ID exists, only look at time-averaged bins larger than it
            if len(min_id)==1:
                min_id = int(min_id)
                rad_bins = rad_final_bins[min_id:max_id+1]
            else:
                rad_bins = rad_final_bins[0:max_id+1]
        else:
            # If minimum ID exists, only look at time-averaged bins larger than it
            if len(min_id)==1:
                min_id = int(min_id)
                rad_bins = rad_final_bins[min_id:]
            else:
                rad_bins = rad_final_bins[0:]

        # Define empty arrays for binned data given radial and angular bins
        fa_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_avg_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Total aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Total aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_sum_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Total aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_avg_real_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Deviation of Average body force magnitude for all particles in (r/r_c, theta) bins from steady-state

        fa_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Deviation of Averaged aligned body force density for all particles in (r/r_c, theta) bins from steady-state
        faA_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged aligned body force density for A particles in (r/r_c, theta) bins from steady-state
        faB_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged aligned body force density for B particles in (r/r_c, theta) bins from steady-state

        align_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))         # Deviation of Averaged alignment for all particles in (r/r_c, theta) bins from steady-state
        alignA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged alignment for A particles in (r/r_c, theta) bins from steady-state
        alignB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        # Deviation of Averaged alignment for B particles in (r/r_c, theta) bins from steady-state

        num_dens_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Deviation of Averaged number density for all particles in (r/r_c, theta) bins from steady-state
        num_densA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for A particles in (r/r_c, theta) bins from steady-state
        num_densB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for B particles in (r/r_c, theta) bins from steady-state

        part_fracA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for A particles in (r/r_c, theta) bins from steady-state
        part_fracB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Deviation of Averaged number density for B particles in (r/r_c, theta) bins from steady-state

        num_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))           # Deviation of Total number of all particles in (r/r_c, theta) bins from steady-state
        numA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))          # Deviation of Total number of A particles in (r/r_c, theta) bins from steady-state
        numB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))          # Deviation of Total number of B particles in (r/r_c, theta) bins from steady-state

        fa_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_avg_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Total aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Total aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_sum_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Total aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_avg_real_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))# Average body force magnitude for all particles in (r/r_c, theta) bins for current time step

        fa_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))   # Averaged aligned body force density for all particles in (r/r_c, theta) bins for current time step
        faA_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged aligned body force density for A particles in (r/r_c, theta) bins for current time step
        faB_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged aligned body force density for B particles in (r/r_c, theta) bins for current time step

        align_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))     # Averaged alignment for all particles in (r/r_c, theta) bins for current time step
        alignA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged alignment for A particles in (r/r_c, theta) bins for current time step
        alignB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))    # Averaged alignment for B particles in (r/r_c, theta) bins for current time step

        num_dens_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))  # Averaged number density for all particles in (r/r_c, theta) bins for current time step
        num_densA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for A particles in (r/r_c, theta) bins for current time step
        num_densB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for B particles in (r/r_c, theta) bins for current time step

        part_fracA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for A particles in (r/r_c, theta) bins for current time step
        part_fracB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins))) # Averaged number density for B particles in (r/r_c, theta) bins for current time step

        num_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))       # Total number all particles in (r/r_c, theta) bins for current time step
        numA_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Total number A particles in (r/r_c, theta) bins for current time step
        numB_final_binned_val = np.zeros((len(rad_final_bins), len(theta_final_bins)))      # Total number B particles in (r/r_c, theta) bins for current time step

        radius_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))        
        radiusA_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))
        radiusB_final_binned = np.zeros((len(rad_final_bins), len(theta_final_bins)))

        
        fa_sum_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Total aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_sum_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Total aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_sum_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Total aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) from steady-state


        fa_avg_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_avg_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_avg_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) from steady-state

        fa_avg_real_final_binned_theta = np.zeros(len(theta_final_bins))        # Deviation of Average body force magnitude for all particles in (theta) bins averaged over (r/r_c) from steady-state
        
        fa_dens_final_binned_theta = np.zeros(len(theta_final_bins))            # Deviation of Average aligned body force density for all particles in (theta) bins averaged over (r/r_c) from steady-state
        faA_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average aligned body force density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        faB_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average aligned body force density for B particles in (theta) bins averaged over (r/r_c) from steady-state


        align_final_binned_theta = np.zeros(len(theta_final_bins))              # Deviation of Average alignment for all particles in (theta) bins averaged over (r/r_c) from steady-state
        alignA_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average alignment for A particles in (theta) bins averaged over (r/r_c) from steady-state
        alignB_final_binned_theta = np.zeros(len(theta_final_bins))             # Deviation of Average alignment for B particles in (theta) bins averaged over (r/r_c) from steady-state

        num_dens_final_binned_theta = np.zeros(len(theta_final_bins))           # Deviation of Average number density for all particles in (theta) bins averaged over (r/r_c) from steady-state
        num_densA_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        num_densB_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for B particles in (theta) bins averaged over (r/r_c) from steady-state

        part_fracA_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for A particles in (theta) bins averaged over (r/r_c) from steady-state
        part_fracB_final_binned_theta = np.zeros(len(theta_final_bins))          # Deviation of Average number density for B particles in (theta) bins averaged over (r/r_c) from steady-state

        num_final_binned_theta = np.zeros(len(theta_final_bins))                # Deviation of Total number of all particles in (theta) bins averaged over (r/r_c) from steady-state
        numA_final_binned_theta = np.zeros(len(theta_final_bins))               # Deviation of Total number of A particles in (theta) bins averaged over (r/r_c) from steady-state
        numB_final_binned_theta = np.zeros(len(theta_final_bins))               # Deviation of Total number of B particles in (theta) bins averaged over (r/r_c) from steady-state

        fa_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_avg_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) for current time step

        fa_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Total aligned body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Total aligned body force magnitude for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_sum_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Total aligned body force magnitude for B particles in (theta) bins averaged over (r/r_c) for current time step
        
        fa_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))        # Average aligned body force density for all particles in (theta) bins averaged over (r/r_c) for current time step
        faA_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average aligned body force density for A particles in (theta) bins averaged over (r/r_c) for current time step
        faB_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average aligned body force density for B particles in (theta) bins averaged over (r/r_c) for current time step

        fa_avg_real_final_binned_val_theta = np.zeros(len(theta_final_bins))    # Average body force magnitude for all particles in (theta) bins averaged over (r/r_c) for current time step

        align_final_binned_val_theta = np.zeros(len(theta_final_bins))          # Average alignment for all particles in (theta) bins averaged over (r/r_c) for current time step
        alignA_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average alignment for A particles in (theta) bins averaged over (r/r_c) for current time step
        alignB_final_binned_val_theta = np.zeros(len(theta_final_bins))         # Average alignment for B particles in (theta) bins averaged over (r/r_c) for current time step

        num_dens_final_binned_val_theta = np.zeros(len(theta_final_bins))       # Average number density for all particles in (theta) bins averaged over (r/r_c) for current time step
        num_densA_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for A particles in (theta) bins averaged over (r/r_c) for current time step
        num_densB_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for B particles in (theta) bins averaged over (r/r_c) for current time step

        part_fracA_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for A particles in (theta) bins averaged over (r/r_c) for current time step
        part_fracB_final_binned_val_theta = np.zeros(len(theta_final_bins))      # Average number density for B particles in (theta) bins averaged over (r/r_c) for current time step

        num_final_binned_val_theta = np.zeros(len(theta_final_bins))            # Total number all particles in (theta) bins averaged over (r/r_c) for current time step
        numA_final_binned_val_theta = np.zeros(len(theta_final_bins))           # Total number A particles in (theta) bins averaged over (r/r_c) for current time step
        numB_final_binned_val_theta = np.zeros(len(theta_final_bins))           # Total number B particles in (theta) bins averaged over (r/r_c) for current time step

        radius_final_binned_theta_new = np.zeros(len(theta_final_bins))         # Average cluster radius in (theta) bins averaged over (r/r_c) for current time step

        avg_rad_theta = np.mean(difr_ext_surface)

        sum_id = 0
        sum_id_theta = 0

        # Loop over select time-averaged angular bins within current min and max angles
        for n_theta in range(1, len(theta_bins)):
            
            # All, A, or B particles within current and previous theta (2 deg) and defined limits for interface (0.3 and 1.1 of cluster radius) to reduce influence of misidentified interface
            bin_ids_theta = np.where((theta_dist_norm <theta_bins[n_theta]) & (theta_dist_norm >= theta_bins[n_theta-1]) & (r_dist_norm < 1.1) & (r_dist_norm >= 0.3))[0]
            binA_ids_theta = np.where((thetaA_dist_norm <theta_bins[n_theta]) & (thetaA_dist_norm >= theta_bins[n_theta-1]) & (rA_dist_norm < 1.1) & (rA_dist_norm >= 0.3))[0]
            binB_ids_theta = np.where((thetaB_dist_norm <theta_bins[n_theta]) & (thetaB_dist_norm >= theta_bins[n_theta-1]) & (rB_dist_norm < 1.1) & (rB_dist_norm >= 0.3))[0]
            
            # If particle exist within range, calculate statistics of them
            if len(bin_ids_theta)>0:
                
                # Output theta index for statistics data
                theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]
                """
                # Area of current angular and radial slice
                area_radial_slice = np.pi * ((1.1*np.mean(nearest_surface_arr[bin_ids_theta]))**2 - (0.3*np.mean(nearest_surface_arr[bin_ids_theta]))**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)
                
                # If denominator is non-zero, calculate deviation from steady-state of average body force magnitude for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_avg_real']['all'][theta_final_id][0])>0):
                    fa_avg_real_final_binned_theta[theta_final_id] = np.mean(fa_avg[bin_ids_theta]) - float(avg_rad_dict_theta['fa_avg_real']['all'][theta_final_id])

                # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_avg']['all'][theta_final_id][0])>0):
                    fa_avg_final_binned_theta[theta_final_id] = np.mean(fa_norm[bin_ids_theta]) - float(avg_rad_dict_theta['fa_avg']['all'][theta_final_id])

                # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_sum']['all'][theta_final_id][0])>0):
                    fa_sum_final_binned_theta[theta_final_id] = np.sum(fa_norm[bin_ids_theta]) - float(avg_rad_dict_theta['fa_sum']['all'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for A particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_sum']['A'][theta_final_id][0])>0):
                    if len(binA_ids_theta)>0:
                        faA_sum_final_binned_theta[theta_final_id] = np.sum(faA_norm[binA_ids_theta]) - float(avg_rad_dict_theta['fa_sum']['A'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for B particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_sum']['B'][theta_final_id][0])>0):
                    if len(binB_ids_theta)>0:
                        faB_sum_final_binned_theta[theta_final_id] = np.sum(faB_norm[binB_ids_theta]) - float(avg_rad_dict_theta['fa_sum']['B'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for A particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_avg']['A'][theta_final_id][0])>0):
                    if len(binA_ids_theta)>0:
                        faA_avg_final_binned_theta[theta_final_id] = np.mean(faA_norm[binA_ids_theta]) - float(avg_rad_dict_theta['fa_avg']['A'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for B particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_avg']['B'][theta_final_id][0])>0):   
                    if len(binB_ids_theta)>0:
                        faB_avg_final_binned_theta[theta_final_id] = np.mean(faB_norm[binB_ids_theta]) - float(avg_rad_dict_theta['fa_avg']['B'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_dens']['all'][theta_final_id][0])>0):
                    fa_dens_final_binned_theta[theta_final_id] = (np.sum(fa_norm[bin_ids_theta]) / area_radial_slice) - float(avg_rad_dict_theta['fa_dens']['all'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for A particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_dens']['A'][theta_final_id][0])>0):    
                    if len(binA_ids_theta)>0:
                        faA_dens_final_binned_theta[theta_final_id] = (np.sum(faA_norm[binA_ids_theta]) / area_radial_slice) - float(avg_rad_dict_theta['fa_dens']['A'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for B particles at given (theta) bin
                if (float(avg_rad_dict_theta['fa_dens']['B'][theta_final_id][0])>0):    
                    if len(binB_ids_theta)>0:
                        faB_dens_final_binned_theta[theta_final_id] = (np.sum(faB_norm[binB_ids_theta]) / area_radial_slice) - float(avg_rad_dict_theta['fa_dens']['B'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['align']['all'][theta_final_id][0])>0):
                    align_final_binned_theta[theta_final_id] = np.mean(align_norm[bin_ids_theta]) - float(avg_rad_dict_theta['align']['all'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for A particles at given (theta) bin
                if (float(avg_rad_dict_theta['align']['A'][theta_final_id][0])>0):    
                    if len(binA_ids_theta)>0:
                        alignA_final_binned_theta[theta_final_id] = np.mean(alignA_norm[binA_ids_theta]) - float(avg_rad_dict_theta['align']['A'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for B particles at given (theta) bin
                if (float(avg_rad_dict_theta['align']['B'][theta_final_id][0])>0):    
                    if len(binB_ids_theta)>0:
                        alignB_final_binned_theta[theta_final_id] = np.mean(alignB_norm[binB_ids_theta]) - float(avg_rad_dict_theta['align']['B'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged number density for all particles at given (theta) bin
                if (float(avg_rad_dict_theta['num_dens']['all'][theta_final_id][0])>0):
                    num_dens_final_binned_theta[theta_final_id] = (len(bin_ids_theta) / area_radial_slice) - float(avg_rad_dict_theta['num_dens']['all'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (theta) bin
                if (float(avg_rad_dict_theta['num_dens']['A'][theta_final_id][0])>0):    
                    if len(binA_ids_theta)>0:
                        num_densA_final_binned_theta[theta_final_id] = (len(binA_ids_theta) / area_radial_slice) - float(avg_rad_dict_theta['num_dens']['A'][theta_final_id])
                
                # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (theta) bin
                if (float(avg_rad_dict_theta['num_dens']['B'][theta_final_id][0])>0):    
                    if len(binB_ids_theta)>0:
                        num_densB_final_binned_theta[theta_final_id] = (len(binB_ids_theta) / area_radial_slice) - float(avg_rad_dict_theta['num_dens']['B'][theta_final_id])
                        
                """
                radius_final_binned_theta_new[theta_final_id] = np.mean(r_dist[bin_ids_theta])
                sum_id_theta += 1
                

            # Loop over select time-averaged radial bins within current min and max radii
            for n_rad in range(1, len(rad_bins)):

                # Find all, A, and B particle IDs currently within time-averaged bin
                bin_ids = np.where((theta_dist_norm <theta_bins[n_theta]) & (theta_dist_norm >= theta_bins[n_theta-1]) & (r_dist_norm < rad_bins[n_rad]) & (r_dist_norm >= rad_bins[n_rad-1]))[0]
                binA_ids = np.where((thetaA_dist_norm <theta_bins[n_theta]) & (thetaA_dist_norm >= theta_bins[n_theta-1]) & (rA_dist_norm < rad_bins[n_rad]) & (rA_dist_norm >= rad_bins[n_rad-1]))[0]
                binB_ids = np.where((thetaB_dist_norm <theta_bins[n_theta]) & (thetaB_dist_norm >= theta_bins[n_theta-1]) & (rB_dist_norm < rad_bins[n_rad]) & (rB_dist_norm >= rad_bins[n_rad-1]))[0]
                
                # If particles within bin, calculate property
                if len(bin_ids)>0:
                    """
                    rad_final_id = np.where(rad_final_bins==rad_bins[n_rad])[0]
                    theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]

                    area_radial_slice = np.pi * ((rad_bins[n_rad]*avg_rad_theta)**2 - (rad_bins[n_rad-1]*avg_rad_theta)**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)

                    fa_final_binned[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                    faA_final_binned[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                    faB_final_binned[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                    align_final_binned[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                    alignA_final_binned[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                    alignB_final_binned[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                    num_dens_final_binned[rad_final_id, theta_final_id] = len(bin_ids) / area_radial_slice
                    num_densA_final_binned[rad_final_id, theta_final_id] = len(binA_ids) / area_radial_slice
                    num_densB_final_binned[rad_final_id, theta_final_id] = len(binB_ids) / area_radial_slice

                    num_final_binned[rad_final_id, theta_final_id] = len(bin_ids)
                    numA_final_binned[rad_final_id, theta_final_id] = len(binA_ids)
                    numB_final_binned[rad_final_id, theta_final_id] = len(binB_ids)
                    """
                    
                    # Find ID of time-averaged bin arrays
                    rad_final_id = np.where(rad_final_bins==rad_bins[n_rad])[0]
                    theta_final_id = np.where(theta_final_bins==theta_bins[n_theta])[0]
                    
                    # Current area of radial slice
                    area_radial_slice = np.pi * ((rad_bins[n_rad]*np.mean(nearest_surface_arr[bin_ids]))**2 - (rad_bins[n_rad-1]*np.mean(nearest_surface_arr[bin_ids]))**2) * ((theta_bins[n_theta]-theta_bins[n_theta-1])/360)
                    
                    # If non-zero, continue...
                    if area_radial_slice > 0:

                        # If load saved file for averages...
                        if load_save == 1:
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_avg_real_final_binned[rad_final_id, theta_final_id] = ((np.mean(fa_avg[bin_ids])) - float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_real_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_avg[bin_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0])>0:
                                fa_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(fa_norm[bin_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0]))
                                fa_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faA_norm[binA_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0]))
                                faA_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faB_norm[binB_ids]) / area_radial_slice) - float(avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0]))
                                faB_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])/ area_radial_slice

                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(fa_norm[bin_ids]) - float(avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id][0]))
                                fa_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faA_norm[binA_ids]) - float(avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id][0]))
                                faA_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faB_norm[binB_ids]) - float(avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id][0]))
                                faB_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id][0])>0):
                                fa_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(fa_norm[bin_ids]) - float(avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                faA_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faA_norm[binA_ids]) - float(avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id][0]))
                                faA_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                faB_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faB_norm[binB_ids]) - float(avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id][0]))
                                faB_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['align']['all'][rad_final_id, theta_final_id][0])>0:
                                align_final_binned[rad_final_id, theta_final_id] = (np.mean(align_norm[bin_ids]) - float(avg_rad_dict['align']['all'][rad_final_id, theta_final_id][0]))
                                align_final_binned_val[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['align']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                alignA_final_binned[rad_final_id, theta_final_id] = (np.mean(alignA_norm[binA_ids]) - float(avg_rad_dict['align']['A'][rad_final_id, theta_final_id][0]))
                                alignA_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['align']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                alignB_final_binned[rad_final_id, theta_final_id] = (np.mean(alignB_norm[binB_ids]) - float(avg_rad_dict['align']['B'][rad_final_id, theta_final_id][0]))
                                alignB_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                            # Average nearest cluster radius for all, A, or B particles at given (r/r_c, theta) bin
                            radius_final_binned[rad_final_id, theta_final_id] = (np.mean(r_dist[bin_ids]))
                            radiusA_final_binned[rad_final_id, theta_final_id] = (np.mean(rA_dist[binA_ids]))
                            radiusB_final_binned[rad_final_id, theta_final_id] = (np.mean(rB_dist[binB_ids]))


                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id][0])>0:
                                num_dens_final_binned[rad_final_id, theta_final_id] = ((len(bin_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id][0]))
                                num_dens_final_binned_val[rad_final_id, theta_final_id] = (len(bin_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                num_densA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id][0]))
                                num_densA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                num_densB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / area_radial_slice) - float(avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id][0]))
                                num_densB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / area_radial_slice)

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id][0])>0) & (len(binA_ids)>0):
                                part_fracA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / (len(binA_ids) + len(binB_ids))) - float(avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id][0]))
                                part_fracA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / (len(binA_ids) + len(binB_ids)))
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (float(avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id][0])>0) & (len(binB_ids)>0):
                                part_fracB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / (len(binA_ids) + len(binB_ids))) - float(avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id][0]))
                                part_fracB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / (len(binA_ids) + len(binB_ids)))

                        # If did not load save file for averages
                        else:
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average body force magnitude for all particles at given (r/r_c, theta) bin
                            if float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0])>0:
                                fa_avg_real_final_binned[rad_final_id, theta_final_id] = ((np.mean(fa_avg[bin_ids])) - float(avg_rad_dict['fa_avg_real']['all'][rad_final_id, theta_final_id][0]))
                                fa_avg_real_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_avg[bin_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id]>0:
                                fa_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(fa_norm[bin_ids]) - avg_rad_dict['fa_avg']['all'][rad_final_id, theta_final_id])
                                fa_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                faA_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faA_norm[binA_ids]) - avg_rad_dict['fa_avg']['A'][rad_final_id, theta_final_id])
                                faA_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of average aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                faB_avg_final_binned[rad_final_id, theta_final_id] = (np.mean(faB_norm[binB_ids]) - avg_rad_dict['fa_avg']['B'][rad_final_id, theta_final_id])
                                faB_avg_final_binned_val[rad_final_id, theta_final_id] = np.mean(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id]>0:
                                fa_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(fa_norm[bin_ids]) - avg_rad_dict['fa_sum']['all'][rad_final_id, theta_final_id])
                                fa_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                faA_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faA_norm[binA_ids]) - avg_rad_dict['fa_sum']['A'][rad_final_id, theta_final_id])
                                faA_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of total aligned body force magnitude for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                faB_sum_final_binned[rad_final_id, theta_final_id] = (np.sum(faB_norm[binB_ids]) - avg_rad_dict['fa_sum']['B'][rad_final_id, theta_final_id])
                                faB_sum_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])

                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0]>0:
                                fa_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(fa_norm[bin_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['all'][rad_final_id, theta_final_id][0])
                                fa_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(fa_norm[bin_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0]>0) & (len(binA_ids)>0):
                                faA_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faA_norm[binA_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['A'][rad_final_id, theta_final_id][0])
                                faA_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faA_norm[binA_ids])/ area_radial_slice
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged aligned body force density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0]>0) & (len(binB_ids)>0):
                                faB_dens_final_binned[rad_final_id, theta_final_id] = ((np.sum(faB_norm[binB_ids]) / area_radial_slice) - avg_rad_dict['fa_dens']['B'][rad_final_id, theta_final_id][0])
                                faB_dens_final_binned_val[rad_final_id, theta_final_id] = np.sum(faB_norm[binB_ids])/ area_radial_slice

                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['align']['all'][rad_final_id, theta_final_id]>0:
                                align_final_binned[rad_final_id, theta_final_id] = (np.mean(align_norm[bin_ids]) - avg_rad_dict['align']['all'][rad_final_id, theta_final_id])
                                align_final_binned_val[rad_final_id, theta_final_id] = np.mean(align_norm[bin_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['align']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                alignA_final_binned[rad_final_id, theta_final_id] = (np.mean(alignA_norm[binA_ids]) - avg_rad_dict['align']['A'][rad_final_id, theta_final_id])
                                alignA_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignA_norm[binA_ids])
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged alignment for all particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['align']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                alignB_final_binned[rad_final_id, theta_final_id] = (np.mean(alignB_norm[binB_ids]) - avg_rad_dict['align']['B'][rad_final_id, theta_final_id])
                                alignB_final_binned_val[rad_final_id, theta_final_id] = np.mean(alignB_norm[binB_ids])

                            # Average nearest cluster radius for all, A, or B particles at given (r/r_c, theta) bin
                            radius_final_binned[rad_final_id, theta_final_id] = (np.mean(r_dist[bin_ids]))
                            radiusA_final_binned[rad_final_id, theta_final_id] = (np.mean(rA_dist[binA_ids]))
                            radiusB_final_binned[rad_final_id, theta_final_id] = (np.mean(rB_dist[binB_ids]))

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for all particles at given (r/r_c, theta) bin
                            if avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id]>0:
                                num_dens_final_binned[rad_final_id, theta_final_id] = ((len(bin_ids) / area_radial_slice) - avg_rad_dict['num_dens']['all'][rad_final_id, theta_final_id])
                                num_dens_final_binned_val[rad_final_id, theta_final_id] = (len(bin_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                num_densA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / area_radial_slice) - avg_rad_dict['num_dens']['A'][rad_final_id, theta_final_id])
                                num_densA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / area_radial_slice)
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                num_densB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / area_radial_slice) - avg_rad_dict['num_dens']['B'][rad_final_id, theta_final_id])
                                num_densB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / area_radial_slice)

                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for A particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id]>0) & (len(binA_ids)>0):
                                part_fracA_final_binned[rad_final_id, theta_final_id] = ((len(binA_ids) / (len(binB_ids) + len(binA_ids)))- avg_rad_dict['part_frac']['A'][rad_final_id, theta_final_id])
                                part_fracA_final_binned_val[rad_final_id, theta_final_id] = (len(binA_ids) / (len(binB_ids) + len(binA_ids)))
                            
                            # If denominator is non-zero, calculate deviation from steady-state of averaged number density for B particles at given (r/r_c, theta) bin
                            if (avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id]>0) & (len(binB_ids)>0):
                                part_fracB_final_binned[rad_final_id, theta_final_id] = ((len(binB_ids) / (len(binB_ids) + len(binA_ids))) - avg_rad_dict['part_frac']['B'][rad_final_id, theta_final_id])
                                part_fracB_final_binned_val[rad_final_id, theta_final_id] = (len(binB_ids) / (len(binB_ids) + len(binA_ids)))

        #print(avg_rad_dict['rad'])
        #stop
        #unique_rad = np.unique(avg_rad_dict['rad'])
        
        unique_theta = np.unique(avg_rad_dict['theta'])

        int_fa_dens_time_theta = np.zeros(len(unique_theta))
        int_faA_dens_time_theta = np.zeros(len(unique_theta))
        int_faB_dens_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_real_time_theta = np.zeros(len(unique_theta))

        int_fa_avg_time_theta = np.zeros(len(unique_theta))
        int_faA_avg_time_theta = np.zeros(len(unique_theta))
        int_faB_avg_time_theta = np.zeros(len(unique_theta))

        int_fa_sum_time_theta = np.zeros(len(unique_theta))
        int_faA_sum_time_theta = np.zeros(len(unique_theta))
        int_faB_sum_time_theta = np.zeros(len(unique_theta))

        int_num_dens_time_theta = np.zeros(len(unique_theta))
        int_num_densA_time_theta = np.zeros(len(unique_theta))
        int_num_densB_time_theta = np.zeros(len(unique_theta))

        int_part_fracA_time_theta = np.zeros(len(unique_theta))
        int_part_fracB_time_theta = np.zeros(len(unique_theta))

        int_align_time_theta = np.zeros(len(unique_theta))
        int_alignA_time_theta = np.zeros(len(unique_theta))
        int_alignB_time_theta = np.zeros(len(unique_theta))
        temp_id_new = np.where((rad_final_bins>=0.3) & (rad_final_bins<=1.1))[0]

        for j in range(1, len(temp_id_new)):
            test_id = np.where(avg_rad_dict['rad']==rad_final_bins[temp_id_new[j]])[0]
            if len(test_id)>0:

                rad_step = (rad_final_bins[temp_id_new[j]]-rad_final_bins[temp_id_new[j-1]])*radius_final_binned[temp_id_new[j],:]

                int_fa_dens_time_theta += (rad_step/2) * (fa_dens_final_binned_val[temp_id_new[j],:]+fa_dens_final_binned_val[temp_id_new[j-1],:])
                int_faA_dens_time_theta += (rad_step/2) * (faA_dens_final_binned_val[temp_id_new[j],:]+faA_dens_final_binned_val[temp_id_new[j-1],:])
                int_faB_dens_time_theta += (rad_step/2) * (faB_dens_final_binned_val[temp_id_new[j],:]+faB_dens_final_binned_val[temp_id_new[j-1],:])

                int_fa_avg_real_time_theta += (rad_step/2) * (fa_avg_real_final_binned_val[temp_id_new[j],:]+fa_avg_real_final_binned_val[temp_id_new[j-1],:])

                int_fa_avg_time_theta += (rad_step/2) * (fa_avg_final_binned_val[temp_id_new[j],:]+fa_avg_final_binned_val[temp_id_new[j-1],:])
                int_faA_avg_time_theta += (rad_step/2) * (faA_avg_final_binned_val[temp_id_new[j],:]+faA_avg_final_binned_val[temp_id_new[j-1],:])
                int_faB_avg_time_theta += (rad_step/2) * (faB_avg_final_binned_val[temp_id_new[j],:]+faB_avg_final_binned_val[temp_id_new[j-1],:])

                int_fa_sum_time_theta += (rad_step/2) * (fa_sum_final_binned_val[temp_id_new[j],:]+fa_sum_final_binned_val[temp_id_new[j-1],:])
                int_faA_sum_time_theta += (rad_step/2) * (faA_sum_final_binned_val[temp_id_new[j],:]+faA_sum_final_binned_val[temp_id_new[j-1],:])
                int_faB_sum_time_theta += (rad_step/2) * (faB_sum_final_binned_val[temp_id_new[j],:]+faB_sum_final_binned_val[temp_id_new[j-1],:])

                int_num_dens_time_theta += (rad_step/2) * (num_dens_final_binned_val[temp_id_new[j],:]+num_dens_final_binned_val[temp_id_new[j-1],:])
                int_num_densA_time_theta += (rad_step/2) * (num_densA_final_binned_val[temp_id_new[j],:]+num_densA_final_binned_val[temp_id_new[j-1],:])
                int_num_densB_time_theta += (rad_step/2) * (num_densB_final_binned_val[temp_id_new[j],:]+num_densB_final_binned_val[temp_id_new[j-1],:])

                int_part_fracA_time_theta += (rad_step/2) * (part_fracA_final_binned_val[temp_id_new[j],:]+part_fracA_final_binned_val[temp_id_new[j-1],:])
                int_part_fracB_time_theta += (rad_step/2) * (part_fracB_final_binned_val[temp_id_new[j],:]+part_fracB_final_binned_val[temp_id_new[j-1],:])

                int_align_time_theta += (rad_step/2) * (align_final_binned_val[temp_id_new[j],:]+align_final_binned_val[temp_id_new[j-1],:])
                int_alignA_time_theta += (rad_step/2) * (alignA_final_binned_val[temp_id_new[j],:]+alignA_final_binned_val[temp_id_new[j-1],:])
                int_alignB_time_theta += (rad_step/2) * (alignB_final_binned_val[temp_id_new[j],:]+alignB_final_binned_val[temp_id_new[j-1],:])

        dif_int_fa_dens_time_theta = int_fa_dens_time_theta - integrated_avg_rad_dict['fa_dens']['all'].astype('float64')
        dif_int_faA_dens_time_theta = int_faA_dens_time_theta - integrated_avg_rad_dict['fa_dens']['A']
        dif_int_faB_dens_time_theta = int_faB_dens_time_theta - integrated_avg_rad_dict['fa_dens']['B']

        dif_int_fa_avg_real_time_theta = int_fa_avg_real_time_theta - integrated_avg_rad_dict['radius']

        dif_int_fa_avg_time_theta = int_fa_avg_time_theta - integrated_avg_rad_dict['fa_avg']['all']
        dif_int_faA_avg_time_theta = int_faA_avg_time_theta - integrated_avg_rad_dict['fa_avg']['A']
        dif_int_faB_avg_time_theta = int_faB_avg_time_theta - integrated_avg_rad_dict['fa_avg']['B']

        dif_int_fa_sum_time_theta = int_fa_sum_time_theta - integrated_avg_rad_dict['fa_sum']['all']
        dif_int_faA_sum_time_theta = int_faA_sum_time_theta - integrated_avg_rad_dict['fa_sum']['A']
        dif_int_faB_sum_time_theta = int_faB_sum_time_theta - integrated_avg_rad_dict['fa_sum']['B']

        dif_int_num_dens_time_theta = int_num_dens_time_theta - integrated_avg_rad_dict['num_dens']['all']
        dif_int_num_densA_time_theta = int_num_densA_time_theta - integrated_avg_rad_dict['num_dens']['A']
        dif_int_num_densB_time_theta = int_num_densB_time_theta - integrated_avg_rad_dict['num_dens']['B']

        dif_int_part_fracA_time_theta = int_part_fracA_time_theta - integrated_avg_rad_dict['part_frac']['A']
        dif_int_part_fracB_time_theta = int_part_fracB_time_theta - integrated_avg_rad_dict['part_frac']['B']

        dif_int_align_time_theta = int_align_time_theta - integrated_avg_rad_dict['align']['all']
        dif_int_alignA_time_theta = int_alignA_time_theta - integrated_avg_rad_dict['align']['A']
        dif_int_alignB_time_theta = int_alignB_time_theta - integrated_avg_rad_dict['align']['B']





        dif_avg_int_fa_dens_time_theta = int_fa_dens_time_theta - np.mean(int_fa_dens_time_theta)
        dif_avg_int_faA_dens_time_theta = int_faA_dens_time_theta - np.mean(int_faA_dens_time_theta)
        dif_avg_int_faB_dens_time_theta = int_faB_dens_time_theta - np.mean(int_faB_dens_time_theta)

        dif_avg_int_fa_avg_real_time_theta = int_fa_avg_real_time_theta - np.mean(int_fa_avg_real_time_theta)

        dif_avg_int_fa_avg_time_theta = int_fa_avg_time_theta - np.mean(int_fa_avg_time_theta)
        dif_avg_int_faA_avg_time_theta = int_faA_avg_time_theta - np.mean(int_faA_avg_time_theta)
        dif_avg_int_faB_avg_time_theta = int_faB_avg_time_theta - np.mean(int_faB_avg_time_theta)

        dif_avg_int_fa_sum_time_theta = int_fa_sum_time_theta - np.mean(int_fa_sum_time_theta)
        dif_avg_int_faA_sum_time_theta = int_faA_sum_time_theta - np.mean(int_faA_sum_time_theta)
        dif_avg_int_faB_sum_time_theta = int_faB_sum_time_theta - np.mean(int_faB_sum_time_theta)

        dif_avg_int_num_dens_time_theta = int_num_dens_time_theta - np.mean(int_num_dens_time_theta)
        dif_avg_int_num_densA_time_theta = int_num_densA_time_theta - np.mean(int_num_densA_time_theta)
        dif_avg_int_num_densB_time_theta = int_num_densB_time_theta - np.mean(int_num_densB_time_theta)

        dif_avg_int_part_fracA_time_theta = int_part_fracA_time_theta - np.mean(int_part_fracA_time_theta)
        dif_avg_int_part_fracB_time_theta = int_part_fracB_time_theta - np.mean(int_part_fracB_time_theta)

        dif_avg_int_align_time_theta = int_align_time_theta - np.mean(int_align_time_theta)
        dif_avg_int_alignA_time_theta = int_alignA_time_theta - np.mean(int_alignA_time_theta)
        dif_avg_int_alignB_time_theta = int_alignB_time_theta - np.mean(int_alignB_time_theta)


        radial_heterogeneity_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'radius_ang': radius_final_binned_theta_new, 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned}, 'fa_avg': {'all': fa_avg_final_binned, 'A': faA_avg_final_binned, 'B': faB_avg_final_binned}, 'fa_sum': {'all': fa_sum_final_binned, 'A': faA_sum_final_binned, 'B': faB_sum_final_binned}, 'fa_dens': {'all': fa_dens_final_binned, 'A': faA_dens_final_binned, 'B': faB_dens_final_binned}, 'align': {'all': align_final_binned, 'A': alignA_final_binned, 'B': alignB_final_binned}, 'num_dens': {'all': num_dens_final_binned, 'A': num_densA_final_binned, 'B': num_densB_final_binned}, 'part_frac': {'A': part_fracA_final_binned, 'B': part_fracB_final_binned}}
        #radial_heterogeneity_dict_theta = {'theta': theta_final_bins, 'radius': radius_final_binned_theta_new, 'fa_avg_real': {'all': fa_avg_real_final_binned_theta}, 'fa_sum': {'all': fa_sum_final_binned_theta, 'A': faA_sum_final_binned_theta, 'B': faB_sum_final_binned_theta}, 'fa_avg': {'all': fa_avg_final_binned_theta, 'A': faA_avg_final_binned_theta, 'B': faB_avg_final_binned_theta}, 'fa_dens': {'all': fa_dens_final_binned_theta, 'A': faA_dens_final_binned_theta, 'B': faB_dens_final_binned_theta}, 'align': {'all': align_final_binned_theta, 'A': alignA_final_binned_theta, 'B': alignB_final_binned_theta}, 'num_dens': {'all': num_dens_final_binned_theta, 'A': num_densA_final_binned_theta, 'B': num_densB_final_binned_theta}}
        
        plot_bin_dict = {'rad': rad_final_bins, 'theta': theta_final_bins, 'radius': np.mean(r_dist), 'radius_ang': radius_final_binned_theta_new, 'com': {'x': all_surface_measurements[key]['exterior']['com']['x'], 'y': all_surface_measurements[key]['exterior']['com']['y']}, 'fa_avg_real': {'all': fa_avg_real_final_binned_val}, 'fa_avg': {'all': fa_avg_final_binned_val, 'A': faA_avg_final_binned_val, 'B': faB_avg_final_binned_val}, 'fa_dens': {'all': fa_dens_final_binned_val, 'A': faA_dens_final_binned_val, 'B': faB_dens_final_binned_val}, 'align': {'all': align_final_binned_val, 'A': alignA_final_binned_val, 'B': alignB_final_binned_val}, 'num': {'all': num_final_binned_val, 'A': numA_final_binned_val, 'B': numB_final_binned_val}, 'num_dens': {'all': num_dens_final_binned_val, 'A': num_densA_final_binned_val, 'B': num_densB_final_binned_val}, 'part_frac': {'A': part_fracA_final_binned_val, 'B': part_fracB_final_binned_val}, 'rad_bin': {'all': radius_final_binned, 'A': radiusA_final_binned, 'B': radiusB_final_binned}}

        dif_radial_heterogeneity_dict = {'theta': theta_final_bins, 'fa_avg_real': {'all': dif_int_fa_avg_real_time_theta}, 'fa_avg': {'all': dif_int_fa_avg_time_theta, 'A': dif_int_faA_avg_time_theta, 'B': dif_int_faB_avg_time_theta}, 'fa_sum': {'all': dif_int_fa_sum_time_theta, 'A': dif_int_faA_sum_time_theta, 'B': dif_int_faB_sum_time_theta}, 'fa_dens': {'all': dif_int_fa_dens_time_theta, 'A': dif_int_faA_dens_time_theta, 'B': dif_int_faB_dens_time_theta}, 'align': {'all': dif_int_align_time_theta, 'A': dif_int_alignA_time_theta, 'B': dif_int_alignB_time_theta}, 'num_dens': {'all': dif_int_num_dens_time_theta, 'A': dif_int_num_densA_time_theta, 'B': dif_int_num_densB_time_theta}, 'part_frac': {'A': dif_int_part_fracA_time_theta, 'B': dif_int_part_fracB_time_theta}}

        dif_avg_radial_heterogeneity_dict = {'theta': theta_final_bins, 'fa_avg_real': {'all': dif_avg_int_fa_avg_real_time_theta}, 'fa_avg': {'all': dif_avg_int_fa_avg_time_theta, 'A': dif_avg_int_faA_avg_time_theta, 'B': dif_avg_int_faB_avg_time_theta}, 'fa_sum': {'all': dif_avg_int_fa_sum_time_theta, 'A': dif_avg_int_faA_sum_time_theta, 'B': dif_avg_int_faB_sum_time_theta}, 'fa_dens': {'all': dif_avg_int_fa_dens_time_theta, 'A': dif_avg_int_faA_dens_time_theta, 'B': dif_avg_int_faB_dens_time_theta}, 'align': {'all': dif_avg_int_align_time_theta, 'A': dif_avg_int_alignA_time_theta, 'B': dif_avg_int_alignB_time_theta}, 'num_dens': {'all': dif_avg_int_num_dens_time_theta, 'A': dif_avg_int_num_densA_time_theta, 'B': dif_avg_int_num_densB_time_theta}, 'part_frac': {'A': dif_avg_int_part_fracA_time_theta, 'B': dif_avg_int_part_fracB_time_theta}}

        return radial_heterogeneity_dict, dif_radial_heterogeneity_dict, dif_avg_radial_heterogeneity_dict, plot_dict, plot_bin_dict

    def bulk_heterogeneity(self, method2_align_dict, avg_rad_dict, surface_dict, int_comp_dict, all_surface_measurements, int_dict, phase_dict, load_save = 0):
        
        # Bulk particle IDs for all, A, and B particles
        bulk_id = np.where(phase_dict['part']==0)[0]
        bulk_A_id = np.where((phase_dict['part']==0) & (self.typ==0))[0]
        bulk_B_id = np.where((phase_dict['part']==0) & (self.typ==1))[0]

        # Interface particle IDs for all, A, and B particles
        int_id = np.where(phase_dict['part']==1)[0]
        int_A_id = np.where((phase_dict['part']==1) & (self.typ==0))[0]
        int_B_id = np.where((phase_dict['part']==1) & (self.typ==1))[0]

        # Gas particle IDs for all, A, and B particles
        gas_id = np.where(phase_dict['part']==2)[0]
        gas_A_id = np.where((phase_dict['part']==2) & (self.typ==0))[0]
        gas_B_id = np.where((phase_dict['part']==2) & (self.typ==1))[0]

        # Particle IDs for A and B particles
        A_id = np.where((self.typ==0))[0]
        B_id = np.where((self.typ==1))[0]

        # Largest interface IDs for all, A, and B particles
        int_id = np.where(int_dict['part']==int_dict['largest ids'][0])[0]

        # Largest interface key for dictionary
        key = 'surface id ' + str(int(int_dict['largest ids'][0]))
        
        min_x = all_surface_measurements[key]['exterior']['com']['x']-25
        max_x = all_surface_measurements[key]['exterior']['com']['x']+25

        min_y = all_surface_measurements[key]['exterior']['com']['y']-25
        max_y = all_surface_measurements[key]['exterior']['com']['y']+25

        #min_x = self.hx_box-25
        #max_x = self.hx_box+25

        #min_y = self.hy_box-25
        #max_y = self.hy_box+25

        pos = np.copy(self.pos)
        pos[:,0] = pos[:,0] + +self.hx_box
        pos[:,1] = pos[:,1] + +self.hy_box

        
        test_id = np.where(pos[:,0]>self.lx_box)[0]
        if len(test_id)>0:
            pos[:,0] = pos[:,0]-self.lx_box
        test_id = np.where(pos[:,0]<0)[0]
        if len(test_id)>0:
            pos[:,0] = pos[:,0]+self.lx_box

        test_id = np.where(pos[:,1]>self.ly_box)[0]
        if len(test_id)>0:
            pos[:,1] = pos[:,1]-self.ly_box
        test_id = np.where(pos[:,1]<0)[0]
        if len(test_id)>0:
            pos[:,1] = pos[:,1]+self.ly_box
        
        test_id = np.where((pos[:,0]>=min_x) & (pos[:,0]<=max_x) & (pos[:,1]>=min_y) & (pos[:,1]<=max_y))[0]
        
        pos_wrap_x = pos[test_id,0]
        pos_wrap_y = pos[test_id,1]

        # Instantiate empty array (partNum) containing the average active force alignment
        # with the nearest interface surface normal
        part_align = method2_align_dict['part']['align']

        # Instantiate empty arrays
        fa_norm = np.array([])          #Aligned body force magnitude for all particles
        faA_norm = np.array([])         #Aligned body force magnitude for A particles
        faB_norm = np.array([])         #Aligned body force magnitude for B particles

        fa_avg = np.array([])           #Body force magnitude for all particles

        y_dist_norm = np.array([])      #All particle distance from cluster CoM normalized by nearest cluster radius
        yA_dist_norm = np.array([])     #A particle distance from cluster CoM normalized by nearest cluster radius
        yB_dist_norm = np.array([])     #B particle distance from cluster CoM normalized by nearest cluster radius

        align_norm = np.array([])       #All particle alignment
        alignA_norm = np.array([])      #A particle alignment
        alignB_norm = np.array([])      #B particle alignment

        x_dist_norm = np.array([])  # Angle around CoM for all particles
        xA_dist_norm = np.array([]) # Angle around CoM for A particles
        xB_dist_norm = np.array([]) # Angle around CoM for B particles

        # Loop over particles to calculate each of their alignment, number density, and active forces
        for m in range(0, len(pos_wrap_x)):

            # Save particle properties if A particle
            if self.typ[test_id[m]] == 0:

                # Aligned active force magnitude for all particles
                fa_norm=np.append(fa_norm, part_align[test_id[m]]*self.peA)

                # Active force magnitude for all particles
                fa_avg=np.append(fa_avg, self.peA)

                # Aligned active force magnitude for A particles
                faA_norm=np.append(faA_norm, part_align[test_id[m]]*self.peA)

                # A particle's distance from cluster center of mass normalized by nearest cluster radius
                xA_dist_norm = np.append(xA_dist_norm, pos_wrap_x[m])

                # Alignment of A particles
                alignA_norm=np.append(alignA_norm, part_align[test_id[m]])

                # Angular position of A particles
                yA_dist_norm = np.append(yA_dist_norm, pos_wrap_y[m])

            # Save particle properties if B particle
            else:

                # Aligned active force magnitude for all particles
                fa_norm=np.append(fa_norm, part_align[test_id[m]]*self.peB)

                # Active force magnitude for all particles
                fa_avg=np.append(fa_avg, self.peB)

                # Aligned active force magnitude for B particles
                faB_norm=np.append(faB_norm, part_align[test_id[m]]*self.peB)

                # Alignment of B particles
                alignB_norm=np.append(alignB_norm, part_align[test_id[m]])

                # B particle's distance from cluster center of mass normalized by nearest cluster radius
                xB_dist_norm = np.append(xB_dist_norm, pos_wrap_x[m])

                # Angular position of B particles
                yB_dist_norm = np.append(yB_dist_norm, pos_wrap_y[m])

            # All particle's distance from cluster center of mass normalized by nearest cluster radius
            x_dist_norm = np.append(x_dist_norm, pos_wrap_x[m])

            # Alignment of all particles
            align_norm=np.append(align_norm, part_align[test_id[m]])

            # Angular position of all particles
            y_dist_norm = np.append(y_dist_norm, pos_wrap_y[m])
        import math

        plot_dict = {'x': {'all': x_dist_norm, 'A': xA_dist_norm, 'B': xB_dist_norm}, 'y': {'all': y_dist_norm, 'A': yA_dist_norm, 'B': yB_dist_norm}}

        def round_decimals_down(number:float, decimals:int=2):
            """
            Returns a value rounded down to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.floor(number)

            factor = 10 ** decimals
            return math.floor(number * factor) / factor

        def round_decimals_up(number:float, decimals:int=2):
            """
            Returns a value rounded up to a specific number of decimal places.
            """
            if not isinstance(decimals, int):
                raise TypeError("decimal places must be an integer")
            elif decimals < 0:
                raise ValueError("decimal places has to be 0 or more")
            elif decimals == 0:
                return math.ceil(number)

            factor = 10 ** decimals
            return math.ceil(number * factor) / factor
        
        # Define radial bins for this time step 
        y_final_bins = np.round(np.linspace(min_y, max_y, 20),3)
        x_final_bins = np.round(np.linspace(min_x, max_x, 20),3)

        x_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))        # Deviation of Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        y_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        

        # Define empty arrays for binned data given radial and angular bins
        fa_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))        # Deviation of Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_avg_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))        # Deviation of Total aligned body force magnitude for all particles in (r/r_c, theta) bins from steady-state
        faA_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Total aligned body force magnitude for A particles in (r/r_c, theta) bins from steady-state
        faB_sum_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Total aligned body force magnitude for B particles in (r/r_c, theta) bins from steady-state

        fa_avg_real_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))   # Deviation of Average body force magnitude for all particles in (r/r_c, theta) bins from steady-state

        fa_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))       # Deviation of Averaged aligned body force density for all particles in (r/r_c, theta) bins from steady-state
        faA_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))      # Deviation of Averaged aligned body force density for A particles in (r/r_c, theta) bins from steady-state
        faB_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))      # Deviation of Averaged aligned body force density for B particles in (r/r_c, theta) bins from steady-state

        align_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))         # Deviation of Averaged alignment for all particles in (r/r_c, theta) bins from steady-state
        alignA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))        # Deviation of Averaged alignment for A particles in (r/r_c, theta) bins from steady-state
        alignB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))        # Deviation of Averaged alignment for B particles in (r/r_c, theta) bins from steady-state

        num_dens_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))      # Deviation of Averaged number density for all particles in (r/r_c, theta) bins from steady-state
        num_densA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))     # Deviation of Averaged number density for A particles in (r/r_c, theta) bins from steady-state
        num_densB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))     # Deviation of Averaged number density for B particles in (r/r_c, theta) bins from steady-state

        num_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))           # Deviation of Total number of all particles in (r/r_c, theta) bins from steady-state
        numA_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))          # Deviation of Total number of A particles in (r/r_c, theta) bins from steady-state
        numB_final_binned = np.zeros((len(x_final_bins), len(y_final_bins)))          # Deviation of Total number of B particles in (r/r_c, theta) bins from steady-state

        fa_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))    # Averaged aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))   # Averaged aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_avg_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))   # Averaged aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))    # Total aligned body force magnitude for all particles in (r/r_c, theta) bins for current time step
        faA_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))   # Total aligned body force magnitude for A particles in (r/r_c, theta) bins for current time step
        faB_sum_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))   # Total aligned body force magnitude for B particles in (r/r_c, theta) bins for current time step

        fa_avg_real_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))# Average body force magnitude for all particles in (r/r_c, theta) bins for current time step

        fa_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))   # Averaged aligned body force density for all particles in (r/r_c, theta) bins for current time step
        faA_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))  # Averaged aligned body force density for A particles in (r/r_c, theta) bins for current time step
        faB_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))  # Averaged aligned body force density for B particles in (r/r_c, theta) bins for current time step

        align_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))     # Averaged alignment for all particles in (r/r_c, theta) bins for current time step
        alignA_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))    # Averaged alignment for A particles in (r/r_c, theta) bins for current time step
        alignB_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))    # Averaged alignment for B particles in (r/r_c, theta) bins for current time step

        num_dens_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))  # Averaged number density for all particles in (r/r_c, theta) bins for current time step
        num_densA_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins))) # Averaged number density for A particles in (r/r_c, theta) bins for current time step
        num_densB_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins))) # Averaged number density for B particles in (r/r_c, theta) bins for current time step

        num_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))       # Total number all particles in (r/r_c, theta) bins for current time step
        numA_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))      # Total number A particles in (r/r_c, theta) bins for current time step
        numB_final_binned_val = np.zeros((len(x_final_bins), len(y_final_bins)))      # Total number B particles in (r/r_c, theta) bins for current time step

        sum_id = 0

        for xind in range(1, len(x_final_bins)):

            for yind in range(1, len(y_final_bins)):
                
                bin_ids = np.where((x_dist_norm <x_final_bins[xind]) & (x_dist_norm >= x_final_bins[xind-1]) & (y_dist_norm < y_final_bins[yind]) & (y_dist_norm >= y_final_bins[yind-1]))[0]
                binA_ids = np.where((xA_dist_norm <x_final_bins[xind]) & (xA_dist_norm >= x_final_bins[xind-1]) & (yA_dist_norm < y_final_bins[yind]) & (yA_dist_norm >= y_final_bins[yind-1]))[0]
                binB_ids = np.where((xB_dist_norm <x_final_bins[xind]) & (xB_dist_norm >= x_final_bins[xind-1]) & (yB_dist_norm < y_final_bins[yind]) & (yB_dist_norm >= y_final_bins[yind-1]))[0]
                
                if len(bin_ids)>0:

                    area_bin = (x_final_bins[xind]-x_final_bins[xind-1]) * (y_final_bins[yind]-y_final_bins[yind-1])

                    x_final_binned[xind-1, yind-1] = x_final_bins[xind-1]
                    y_final_binned[xind-1, yind-1] = y_final_bins[yind-1]

                    if (float(avg_rad_dict['fa_avg_real']['all'][xind-1, yind-1])>0):
                        fa_avg_real_final_binned[xind-1, yind-1] = np.mean(fa_avg[bin_ids]) - float(avg_rad_dict['fa_avg_real']['all'][xind-1, yind-1])
                        fa_avg_real_final_binned_val[xind-1, yind-1] = np.mean(fa_avg[bin_ids])

                    if (float(avg_rad_dict['fa_sum']['all'][xind-1, yind-1])>0):
                        fa_sum_final_binned[xind-1, yind-1] = np.sum(fa_norm[bin_ids]) - float(avg_rad_dict['fa_sum']['all'][xind-1, yind-1])
                        fa_sum_final_binned_val[xind-1, yind-1] = np.sum(fa_norm[bin_ids])

                    if (float(avg_rad_dict['fa_avg']['all'][xind-1, yind-1])>0):
                        fa_avg_final_binned[xind-1, yind-1] = np.mean(fa_norm[bin_ids]) - float(avg_rad_dict['fa_avg']['all'][xind-1, yind-1])
                        fa_avg_final_binned_val[xind-1, yind-1] = np.mean(fa_norm[bin_ids])

                    if (float(avg_rad_dict['fa_sum']['A'][xind-1, yind-1])>0) & (len(binA_ids)>0):
                        faA_sum_final_binned[xind-1, yind-1] = np.sum(faA_norm[binA_ids]) - float(avg_rad_dict['fa_sum']['A'][xind-1, yind-1])
                        faA_sum_final_binned_val[xind-1, yind-1] = np.sum(faA_norm[binA_ids])
                    if (float(avg_rad_dict['fa_sum']['B'][xind-1, yind-1])>0) & (len(binB_ids)>0):
                        faB_sum_final_binned[xind-1, yind-1] = np.sum(faB_norm[binB_ids]) - float(avg_rad_dict['fa_sum']['B'][xind-1, yind-1])
                        faB_sum_final_binned_val[xind-1, yind-1] = np.sum(faB_norm[binB_ids])

                    if (float(avg_rad_dict['fa_avg']['A'][xind-1, yind-1])>0) & (len(binA_ids)>0):
                        faA_avg_final_binned[xind-1, yind-1] = np.mean(faA_norm[binA_ids]) - float(avg_rad_dict['fa_avg']['A'][xind-1, yind-1])
                        faA_avg_final_binned_val[xind-1, yind-1] = np.mean(faA_norm[binA_ids])
                    if (float(avg_rad_dict['fa_avg']['B'][xind-1, yind-1])>0) & (len(binB_ids)>0):
                        faB_avg_final_binned[xind-1, yind-1] = np.mean(faB_norm[binB_ids]) - float(avg_rad_dict['fa_avg']['B'][xind-1, yind-1])
                        faB_avg_final_binned_val[xind-1, yind-1] = np.mean(faB_norm[binB_ids])

                    if (float(avg_rad_dict['fa_dens']['all'][xind-1, yind-1])>0):
                        fa_dens_final_binned[xind-1, yind-1] = (np.sum(fa_norm[bin_ids]) / area_bin) - float(avg_rad_dict['fa_dens']['all'][xind-1, yind-1])
                        fa_dens_final_binned_val[xind-1, yind-1] = np.sum(fa_norm[bin_ids]) / area_bin
                    if (float(avg_rad_dict['fa_dens']['A'][xind-1, yind-1])>0) & (len(binA_ids)>0):
                        faA_dens_final_binned[xind-1, yind-1] = (np.sum(faA_norm[binA_ids]) / area_bin) - float(avg_rad_dict['fa_dens']['A'][xind-1, yind-1])
                        faA_dens_final_binned_val[xind-1, yind-1] = np.sum(faA_norm[binA_ids]) / area_bin
                    if (float(avg_rad_dict['fa_dens']['B'][xind-1, yind-1])>0) & (len(binB_ids)>0):
                        faB_dens_final_binned[xind-1, yind-1] = (np.sum(faB_norm[binB_ids]) / area_bin) - float(avg_rad_dict['fa_dens']['B'][xind-1, yind-1])
                        faB_dens_final_binned_val[xind-1, yind-1] = np.sum(faB_norm[binB_ids]) / area_bin

                    if (float(avg_rad_dict['align']['all'][xind-1, yind-1])>0):
                        align_final_binned[xind-1, yind-1] = np.mean(align_norm[bin_ids]) - float(avg_rad_dict['align']['all'][xind-1, yind-1])
                        align_final_binned_val[xind-1, yind-1] = np.mean(align_norm[bin_ids])
                    if (float(avg_rad_dict['align']['A'][xind-1, yind-1])>0) & (len(binA_ids)>0):
                        alignA_final_binned[xind-1, yind-1] = np.mean(alignA_norm[binA_ids]) - float(avg_rad_dict['align']['A'][xind-1, yind-1])
                        alignA_final_binned_val[xind-1, yind-1] = np.mean(alignA_norm[binA_ids])
                    if (float(avg_rad_dict['align']['B'][xind-1, yind-1])>0) & (len(binB_ids)>0):
                        alignB_final_binned[xind-1, yind-1] = np.mean(alignB_norm[binB_ids]) - float(avg_rad_dict['align']['B'][xind-1, yind-1])
                        alignB_final_binned_val[xind-1, yind-1] = np.mean(alignB_norm[binB_ids])

                    
                    if (float(avg_rad_dict['num_dens']['all'][xind-1, yind-1])>0):
                        num_dens_final_binned[xind-1, yind-1] = (len(bin_ids) / area_bin) - float(avg_rad_dict['num_dens']['all'][xind-1, yind-1])
                        num_dens_final_binned_val[xind-1, yind-1] = len(bin_ids) / area_bin
                    if (float(avg_rad_dict['num_dens']['A'][xind-1, yind-1])>0) & (len(binA_ids)>0):
                        num_densA_final_binned[xind-1, yind-1] = (len(binA_ids) / area_bin) - float(avg_rad_dict['num_dens']['A'][xind-1, yind-1])
                        num_densA_final_binned_val[xind-1, yind-1] = len(binA_ids) / area_bin
                    if (float(avg_rad_dict['num_dens']['B'][xind-1, yind-1])>0) & (len(binB_ids)>0):
                        num_densB_final_binned[xind-1, yind-1] = (len(binB_ids) / area_bin) - float(avg_rad_dict['num_dens']['B'][xind-1, yind-1])
                        num_densB_final_binned_val[xind-1, yind-1] = len(binB_ids) / area_bin

                    sum_id += 1

        radial_heterogeneity_dict = {'x': x_final_bins, 'y': y_final_bins, 'fa_avg_real': {'all': fa_avg_real_final_binned}, 'fa_avg': {'all': fa_avg_final_binned, 'A': faA_avg_final_binned, 'B': faB_avg_final_binned}, 'fa_sum': {'all': fa_sum_final_binned, 'A': faA_sum_final_binned, 'B': faB_sum_final_binned}, 'fa_dens': {'all': fa_dens_final_binned, 'A': faA_dens_final_binned, 'B': faB_dens_final_binned}, 'align': {'all': align_final_binned, 'A': alignA_final_binned, 'B': alignB_final_binned}, 'num_dens': {'all': num_dens_final_binned, 'A': num_densA_final_binned, 'B': num_densB_final_binned}}
        
        plot_bin_dict = {'x': x_final_bins, 'y': y_final_bins, 'fa_avg_real': {'all': fa_avg_real_final_binned_val}, 'fa_avg': {'all': fa_avg_final_binned_val, 'A': faA_avg_final_binned_val, 'B': faB_avg_final_binned_val}, 'fa_dens': {'all': fa_dens_final_binned_val, 'A': faA_dens_final_binned_val, 'B': faB_dens_final_binned_val}, 'align': {'all': align_final_binned_val, 'A': alignA_final_binned_val, 'B': alignB_final_binned_val}, 'num_dens': {'all': num_dens_final_binned_val, 'A': num_densA_final_binned_val, 'B': num_densB_final_binned_val}}

        return radial_heterogeneity_dict, plot_dict, plot_bin_dict

    def radial_ang_active_measurements(self, radial_fa_dict, surface_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict):
        int_id = averaged_data_arr['int_id']
        com_radial_dict_fa = {}
        int_ids = int_dict['bin']
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            key2 = 'surface ' + str(int(int_comp_dict['ids'][m]))


            try: 
                surface_x = sep_surface_dict[key]['exterior']['pos']['x']
                surface_y = sep_surface_dict[key]['exterior']['pos']['y']
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                surface_x = sep_surface_dict[key]['interior']['pos']['x']
                surface_y = sep_surface_dict[key]['interior']['pos']['y']
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']

            try: 
                exterior_radius = all_surface_measurements[key]['exterior']['mean radius']
                exterior = 1
            except:
                exterior = 0
                exterior_radius = 0

            try:
                interior_radius = all_surface_measurements[key]['interior']['mean radius']
                interior = 1
            except:
                interior = 0
                interior_radius = 0
                
            if exterior_radius >= interior_radius:
                radius = exterior_radius
            else:
                radius = interior_radius
                
            theta_all = ((radial_fa_dict[key]['all']['theta'] + 360 ) % 360)
            theta_A = ((radial_fa_dict[key]['A']['theta'] + 360 ) % 360)
            theta_B = ((radial_fa_dict[key]['B']['theta'] + 360 ) % 360)
            
            difx = surface_x - (com_x)
            dify = surface_y - (com_y)

            difr= ( (difx )**2 + (dify)**2)**0.5
            thetar = np.arctan2(dify, difx)*(180/np.pi)
            
            thetar_ang = ((thetar + 360 ) % 360)

            #r = np.linspace(np.min(radial_fa_dict[key]['all']['r']), np.max(radial_fa_dict[key]['all']['r']), num=int((np.ceil(np.max(radial_fa_dict[key]['all']['r']) - np.min(radial_fa_dict[key]['all']['r']))+1)/2))
            
            
            r = np.linspace(0, 1.5, num=75)
            theta = np.linspace(0, 360, num=45)

            #Pressure integrand components for each value of X
            act_press_r = np.zeros((len(r)-1))
            int_press_r = np.zeros((len(r)-1))
            act_fa_r = np.zeros((len(r)-1))
            align_r = np.zeros((len(r)-1))
            num_dens_r = np.zeros((len(r)-1))
            num_r = np.zeros((len(r)-1))
            area_r = np.zeros((len(r)-1))

            act_pressA_r = np.zeros((len(r)-1))
            int_pressA_r = np.zeros((len(r)-1))
            act_faA_r = np.zeros((len(r)-1))
            alignA_r = np.zeros((len(r)-1))
            num_densA_r = np.zeros((len(r)-1))
            numA_r = np.zeros((len(r)-1))

            act_pressB_r = np.zeros((len(r)-1))
            int_pressB_r = np.zeros((len(r)-1))
            act_faB_r = np.zeros((len(r)-1))
            alignB_r = np.zeros((len(r)-1))
            num_densB_r = np.zeros((len(r)-1))
            numB_r = np.zeros((len(r)-1))

            rad_arr = np.zeros((len(r)-1))

            sum_act_press_r = np.zeros((len(r)-1)) 
            sum_int_press_r = np.zeros((len(r)-1)) 
            sum_act_fa_r = np.zeros((len(r)-1))
            sum_align_r = np.zeros((len(r)-1))
            sum_num_dens_r = np.zeros((len(r)-1))
            sum_num_r = np.zeros((len(r)-1))
            sum_area_r = np.zeros((len(r)-1))

            sum_act_pressA_r = np.zeros((len(r)-1))
            sum_int_pressA_r = np.zeros((len(r)-1))
            sum_act_faA_r = np.zeros((len(r)-1))
            sum_alignA_r = np.zeros((len(r)-1))
            sum_num_densA_r = np.zeros((len(r)-1))
            sum_numA_r = np.zeros((len(r)-1))

            sum_act_pressB_r = np.zeros((len(r)-1))
            sum_int_pressB_r = np.zeros((len(r)-1))
            sum_act_faB_r = np.zeros((len(r)-1))
            sum_alignB_r = np.zeros((len(r)-1))
            sum_num_densB_r = np.zeros((len(r)-1))
            sum_numB_r = np.zeros((len(r)-1))

            num_theta = 0
            
            #If exterior and interior surfaces defined, continue...
        
            #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
            num_theta = 0
            for j in range(1, len(theta)):

                total_area_prev = 0
                total_area_prev_slice = 0

                min_theta = theta[j-1]
                max_theta = theta[j]

                dif_theta = (max_theta - min_theta)/360

                surface_inrange = np.where((min_theta<=thetar_ang) & (thetar_ang<=max_theta))[0]
                radius_temp = np.mean(difr[surface_inrange])

                for i in range(1, len(r)):

                    #Min and max location across interface of current step
                    min_r = r[i-1]
                    max_r = r[i]

                    #Calculate area of rectangle for current step
                    total_area = (np.pi * ((max_r*radius_temp) ** 2) - total_area_prev)
                    total_area_slice = total_area * dif_theta

                    #Save total area of previous step sizes
                    total_area_prev = np.pi * ((max_r*radius_temp) ** 2)
                    total_area_prev_slice = total_area_prev * dif_theta

                    if r[i] * radius_temp <= self.hx_box:

                        #Find particles that are housed within current slice
                        parts_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['all']['r']/radius_temp)) & ((radial_fa_dict[key]['all']['r']/radius_temp)<=max_r) & (min_theta<=theta_all) & (theta_all<=max_theta))[0]
                        partsA_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['A']['r']/radius_temp)) & ((radial_fa_dict[key]['A']['r']/radius_temp)<=max_r) & (min_theta<=theta_A) & (theta_A<=max_theta))[0]
                        partsB_inrange_fa = np.where((min_r<=(radial_fa_dict[key]['B']['r']/radius_temp)) & ((radial_fa_dict[key]['B']['r']/radius_temp)<=max_r) & (min_theta<=theta_B) & (theta_B<=max_theta))[0]

                        #If at least 1 particle in slice, continue...
                        if len(parts_inrange_fa)>0:

                            #If the force is defined, continue...
                            parts_defined = np.logical_not(np.isnan(radial_fa_dict[key]['all']['fa'][parts_inrange_fa]))

                            if len(parts_defined)>0:
                                #Calculate total active force normal to interface in slice
                                act_press_r[i-1] = np.sum(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                                act_fa_r[i-1] = np.mean(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                                align_r[i-1] = np.mean(radial_fa_dict[key]['all']['align'][parts_inrange_fa][parts_defined])
                                num_dens_r[i-1] = len(parts_defined)
                                num_r[i-1] = len(parts_defined)
                                area_r[i-1] = total_area_slice
                                #If area of slice is non-zero, calculate the pressure [F/A]
                                if total_area_slice > 0:
                                    act_press_r[i-1] = act_press_r[i-1]/total_area_slice
                                    num_dens_r[i-1] = num_dens_r[i-1]/total_area_slice

                                partsA_defined = np.logical_not(np.isnan(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa]))
                                if len(partsA_defined)>0:
                                    act_pressA_r[i-1] = np.sum(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                                    act_faA_r[i-1] = np.mean(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                                    alignA_r[i-1] = np.mean(radial_fa_dict[key]['A']['align'][partsA_inrange_fa][partsA_defined])
                                    num_densA_r[i-1] = len(partsA_defined)
                                    numA_r[i-1] = len(partsA_defined)
                                    #If area of slice is non-zero, calculate the pressure [F/A]
                                    if total_area_slice > 0:
                                        act_pressA_r[i-1] = act_pressA_r[i-1]/total_area_slice
                                        num_densA_r[i-1] = num_densA_r[i-1]/total_area_slice

                                partsB_defined = np.logical_not(np.isnan(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa]))
                                if len(partsB_defined)>0:
                                    act_pressB_r[i-1] = np.sum(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                                    act_faB_r[i-1] = np.mean(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                                    alignB_r[i-1] = np.mean(radial_fa_dict[key]['B']['align'][partsB_inrange_fa][partsB_defined])
                                    #Calculate density
                                    num_densB_r[i-1] = len(partsB_defined)
                                    numB_r[i-1] = len(partsB_defined)
                                    #If area of slice is non-zero, calculate the pressure [F/A]
                                    if total_area_slice > 0:
                                        act_pressB_r[i-1] = act_pressB_r[i-1]/total_area_slice
                                        num_densB_r[i-1] = num_densB_r[i-1]/total_area_slice
                            
                                
                                
                    
                    else:

                        act_press_r[i-1] = 0
                        act_fa_r[i-1] = 0
                        align_r[i-1] = 0
                        num_dens_r[i-1] = 0
                        num_r[i-1] = 0

                        act_pressA_r[i-1] = 0
                        act_faA_r[i-1] = 0
                        alignA_r[i-1] = 0
                        num_densA_r[i-1] = 0
                        numA_r[i-1] = 0

                        act_pressB_r[i-1] = 0
                        act_faB_r[i-1] = 0
                        alignB_r[i-1] = 0
                        num_densB_r[i-1] = 0
                        numB_r[i-1] = 0

                #save this theta's measurement
                sum_act_press_r += act_press_r
                sum_act_fa_r += act_fa_r
                sum_align_r += align_r
                sum_num_dens_r += num_dens_r
                sum_num_r += num_r

                sum_act_pressA_r += act_pressA_r
                sum_act_faA_r += act_faA_r
                sum_alignA_r += alignA_r
                sum_num_densA_r += num_densA_r
                sum_numA_r += numA_r

                sum_act_pressB_r += act_pressB_r
                sum_act_faB_r += act_faB_r
                sum_alignB_r += alignB_r
                sum_num_densB_r += num_densB_r
                sum_numB_r += numB_r

                num_theta += 1

            avg_act_press_r = sum_act_press_r / num_theta
            avg_act_fa_r = sum_act_fa_r / num_theta
            avg_align_r = sum_align_r / num_theta
            avg_num_dens_r = sum_num_dens_r / num_theta
            avg_num_r = sum_num_r / num_theta

            avg_act_pressA_r = sum_act_pressA_r / num_theta
            avg_act_faA_r = sum_act_faA_r / num_theta
            avg_alignA_r = sum_alignA_r / num_theta
            avg_num_densA_r = sum_num_densA_r / num_theta
            avg_numA_r = sum_numA_r / num_theta

            avg_act_pressB_r = sum_act_pressB_r / num_theta
            avg_act_faB_r = sum_act_faB_r / num_theta
            avg_alignB_r = sum_alignB_r / num_theta
            avg_num_densB_r = sum_num_densB_r / num_theta
            avg_numB_r = sum_numB_r / num_theta


            com_radial_dict_fa[key] = {'int_id': int_id, 'current_id': int(int_comp_dict['ids'][m]), 'ext_rad': exterior_radius, 'int_rad': interior_radius, 'r': r[1:].tolist(), 'area': area_r.tolist(), 'com_x': com_x, 'com_y': com_y, 'fa_press': {'all': avg_act_press_r.tolist(), 'A': avg_act_pressA_r.tolist(), 'B': avg_act_pressB_r.tolist()}, 'fa': {'all': avg_act_fa_r.tolist(), 'A': avg_act_faA_r.tolist(), 'B': avg_act_faB_r.tolist()}, 'align': {'all': avg_align_r.tolist(), 'A': avg_alignA_r.tolist(), 'B': avg_alignB_r.tolist()}, 'num_dens': {'all': avg_num_dens_r.tolist(), 'A': avg_num_densA_r.tolist(), 'B': avg_num_densB_r.tolist()}, 'num': {'all': avg_num_r.tolist(), 'A': avg_numA_r.tolist(), 'B': avg_numB_r.tolist()}}
        return com_radial_dict_fa

    def radial_measurements3(self, radial_fa_dict, surface_dict, sep_surface_dict, int_comp_dict, all_surface_measurements, averaged_data_arr, int_dict):
        int_id = averaged_data_arr['int_id']
        com_radial_dict_fa = {}
        int_ids = int_dict['bin']
        for m in range(0, len(sep_surface_dict)):
            key = 'surface id ' + str(int(int_comp_dict['ids'][m]))
            key2 = 'surface ' + str(int(int_comp_dict['ids'][m]))

            try: 
                com_x = all_surface_measurements[key]['exterior']['com']['x']
                com_y = all_surface_measurements[key]['exterior']['com']['y']
            except:
                com_x = all_surface_measurements[key]['interior']['com']['x']
                com_y = all_surface_measurements[key]['interior']['com']['y']

            try: 
                exterior_radius = all_surface_measurements[key]['exterior']['mean radius']
                exterior = 1
            except:
                exterior = 0
                exterior_radius = 0

            try:
                interior_radius = all_surface_measurements[key]['interior']['mean radius']
                interior = 1
            except:
                interior = 0
                interior_radius = 0
                
            if exterior_radius >= interior_radius:
                radius = exterior_radius
            else:
                radius = interior_radius

            #r = np.linspace(np.min(radial_fa_dict[key]['all']['r']), np.max(radial_fa_dict[key]['all']['r']), num=int((np.ceil(np.max(radial_fa_dict[key]['all']['r']) - np.min(radial_fa_dict[key]['all']['r']))+1)/2))
            r = np.linspace(0, self.hx_box, num=100)

            #Pressure integrand components for each value of X
            int_stress_XX_r = np.zeros((len(r)-1))
            int_stress_YY_r = np.zeros((len(r)-1))
            int_stress_XY_r = np.zeros((len(r)-1))
            int_stress_YX_r = np.zeros((len(r)-1))
            int_press_r = np.zeros((len(r)-1))
            #act_fa_r = []
            #lat_r = []
            num_dens_r = np.zeros((len(r)-1))

            int_stressA_XX_r = np.zeros((len(r)-1))
            int_stressA_YY_r = np.zeros((len(r)-1))
            int_stressA_XY_r = np.zeros((len(r)-1))
            int_stressA_YX_r = np.zeros((len(r)-1))
            int_pressA_r = np.zeros((len(r)-1))

            #act_faA_r = []
            #latA_r = []
            num_densA_r = np.zeros((len(r)-1))

            int_stressB_XX_r = np.zeros((len(r)-1))
            int_stressB_YY_r = np.zeros((len(r)-1))
            int_stressB_XY_r = np.zeros((len(r)-1))
            int_stressB_YX_r = np.zeros((len(r)-1))
            int_pressB_r = np.zeros((len(r)-1))
            #act_faB_r = []
            #latB_r = []
            num_densB_r = np.zeros((len(r)-1))

            #If exterior and interior surfaces defined, continue...

            area_prev = 0

            #Pressure integrand components for each value of X
            act_press_r = np.zeros((len(r)-1))
            act_fa_r = np.zeros((len(r)-1))
            align_r = np.zeros((len(r)-1))
            num_dens_r = np.zeros((len(r)-1))
            num_r = np.zeros((len(r)-1))
            area_r = np.zeros((len(r)-1))

            act_pressA_r = np.zeros((len(r)-1))
            act_faA_r = np.zeros((len(r)-1))
            alignA_r = np.zeros((len(r)-1))
            num_densA_r = np.zeros((len(r)-1))
            numA_r = np.zeros((len(r)-1))

            act_pressB_r = np.zeros((len(r)-1))
            act_faB_r = np.zeros((len(r)-1))
            alignB_r = np.zeros((len(r)-1))
            num_densB_r = np.zeros((len(r)-1))
            numB_r = np.zeros((len(r)-1))

            rad_arr = np.zeros((len(r)-1))
            
            #If exterior and interior surfaces defined, continue...
        
            #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
            for i in range(1, len(r)):

                #Min and max location across interface of current step
                min_r = r[i-1]
                max_r = r[i]

                #Calculate area of rectangle for current step
                area = np.pi * (max_r ** 2) - area_prev

                #Save total area of previous step sizes
                area_prev = np.pi * (max_r ** 2)

                #Find particles that are housed within current slice
                parts_inrange_fa = np.where((min_r<=radial_fa_dict[key]['all']['r']) & (radial_fa_dict[key]['all']['r']<=max_r))[0]
                partsA_inrange_fa = np.where((min_r<=radial_fa_dict[key]['A']['r']) & (radial_fa_dict[key]['A']['r']<=max_r))[0]
                partsB_inrange_fa = np.where((min_r<=radial_fa_dict[key]['B']['r']) & (radial_fa_dict[key]['B']['r']<=max_r))[0]


                #If at least 1 particle in slice, continue...
                if len(parts_inrange_fa)>0:

                    #If the force is defined, continue...
                    parts_defined = np.logical_not(np.isnan(radial_fa_dict[key]['all']['fa'][parts_inrange_fa]))

                    if len(parts_defined)>0:
                        #Calculate total active force normal to interface in slice
                        act_press_r[i-1] = np.sum(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                        act_fa_r[i-1] = np.mean(radial_fa_dict[key]['all']['fa'][parts_inrange_fa][parts_defined])
                        align_r[i-1] = np.mean(radial_fa_dict[key]['all']['align'][parts_inrange_fa][parts_defined])
                        num_dens_r[i-1] = len(parts_defined)
                        num_r[i-1] = len(parts_defined)
                        area_r[i-1] = area
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_press_r[i-1] = act_press_r[i-1]/area
                            num_dens_r[i-1] = num_dens_r[i-1]/area

                        partsA_defined = np.logical_not(np.isnan(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa]))
                        if len(partsA_defined)>0:
                            act_pressA_r[i-1] = np.sum(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                            act_faA_r[i-1] = np.mean(radial_fa_dict[key]['A']['fa'][partsA_inrange_fa][partsA_defined])
                            alignA_r[i-1] = np.mean(radial_fa_dict[key]['A']['align'][partsA_inrange_fa][partsA_defined])
                            num_densA_r[i-1] = len(partsA_defined)
                            numA_r[i-1] = len(partsA_defined)
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                act_pressA_r[i-1] = act_pressA_r[i-1]/area
                                num_densA_r[i-1] = num_densA_r[i-1]/area

                        partsB_defined = np.logical_not(np.isnan(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa]))
                        if len(partsB_defined)>0:
                            act_pressB_r[i-1] = np.sum(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                            act_faB_r[i-1] = np.mean(radial_fa_dict[key]['B']['fa'][partsB_inrange_fa][partsB_defined])
                            alignB_r[i-1] = np.mean(radial_fa_dict[key]['B']['align'][partsB_inrange_fa][partsB_defined])
                            #Calculate density
                            num_densB_r[i-1] = len(partsB_defined)
                            numB_r[i-1] = len(partsB_defined)
                            #If area of slice is non-zero, calculate the pressure [F/A]
                            if area > 0:
                                act_pressB_r[i-1] = act_pressB_r[i-1]/area
                                num_densB_r[i-1] = num_densB_r[i-1]/area
            
            com_radial_dict_fa[key] = {'int_id': int_id, 'current_id': int(int_comp_dict['ids'][m]), 'ext_rad': exterior_radius, 'int_rad': interior_radius, 'r': r[1:].tolist(), 'area': area_r.tolist(), 'com_x': com_x, 'com_y': com_y, 'fa_press': {'all': act_press_r.tolist(), 'A': act_pressA_r.tolist(), 'B': act_pressB_r.tolist()}, 'fa': {'all': act_fa_r.tolist(), 'A': act_faA_r.tolist(), 'B': act_faB_r.tolist()}, 'align': {'all': align_r.tolist(), 'A': alignA_r.tolist(), 'B': alignB_r.tolist()}, 'num_dens': {'all': num_dens_r.tolist(), 'A': num_densA_r.tolist(), 'B': num_densB_r.tolist()}, 'num': {'all': num_r.tolist(), 'A': numA_r.tolist(), 'B': numB_r.tolist()}}
        return com_radial_dict_fa

    def cluster_velocity(self, prev_pos, dt_step):

        #Compute cluster parameters using system_all neighbor list
        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(self.pos))
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes
        
        min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
        clust_large = np.amax(clust_size)

 
        current_com_dict = self.plotting_utility_functs.com_view(self.pos, clp_all)

        system_all = freud.AABBQuery(self.f_box, self.f_box.wrap(prev_pos))
        cl_all=freud.cluster.Cluster()                              #Define cluster
        cl_all.compute(system_all, neighbors={'r_max': 1.0})        # Calculate clusters given neighbor list, positions,
                                                                    # and maximal radial interaction distance
        clp_all = freud.cluster.ClusterProperties()                 #Define cluster properties
        ids = cl_all.cluster_idx                                    # get id of each cluster
        clp_all.compute(system_all, ids)                            # Calculate cluster properties given cluster IDs
        clust_size = clp_all.sizes                                  # find cluster sizes

        min_size=int(self.partNum/8)                                     #Minimum cluster size for measurements to happen
        lcID = np.where(clust_size == np.amax(clust_size))[0][0]    #Identify largest cluster
        large_clust_ind_all=np.where(clust_size>min_size)           #Identify all clusters larger than minimum size
        clust_large = np.amax(clust_size)

        prev_com_dict = self.plotting_utility_functs.com_view(prev_pos, clp_all)

        difx_cluster = current_com_dict['com']['x'] - prev_com_dict['com']['x']

        difx_cluster_abs = np.abs(difx_cluster)
        if difx_cluster_abs>=self.hx_box:
            if difx_cluster < -self.hx_box:
                difx_cluster += self.lx_box
            else:
                difx_cluster -= self.lx_box
                
        dify_cluster = current_com_dict['com']['y'] - prev_com_dict['com']['y']

        dify_cluster_abs = np.abs(dify_cluster)
        if dify_cluster_abs>=self.hy_box:
            if dify_cluster < -self.hy_box:
                dify_cluster += self.ly_box
            else:
                dify_cluster -= self.ly_box

        difr_cluster = ( difx_cluster ** 2 + dify_cluster ** 2 ) ** 0.5

        vx_cluster = difx_cluster / dt_step
        vy_cluster = dify_cluster / dt_step
        vr_cluster = difr_cluster / dt_step

        orient = np.arctan2(dify_cluster, difx_cluster)
        
        cluster_velocity_dict = {'displace': {'x': difx_cluster, 'y': dify_cluster, 'r': difr_cluster, 'theta': orient}, 'velocity': {'x': vx_cluster, 'y': vy_cluster, 'r': vr_cluster, 'theta': orient / dt_step} }
        return cluster_velocity_dict
    def radial_active_measurements(self, radial_fa_dict):

        #X locations across interface for integration
        if self.hx_box<self.hy_box:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))
        elif self.hy_box<self.hx_box:
            r = np.linspace(0, self.hy_box, num=int((np.ceil(self.hy_box)+1)/3))
        else:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))

        #If exterior and interior surfaces defined, continue...

        area_prev = 0

        #Pressure integrand components for each value of X
        act_press_r = np.zeros((len(r)-1))
        act_fa_r = np.zeros((len(r)-1))
        align_r = np.zeros((len(r)-1))
        num_dens_r = np.zeros((len(r)-1))

        act_pressA_r = np.zeros((len(r)-1))
        act_faA_r = np.zeros((len(r)-1))
        alignA_r = np.zeros((len(r)-1))
        num_densA_r = np.zeros((len(r)-1))

        act_pressB_r = np.zeros((len(r)-1))
        act_faB_r = np.zeros((len(r)-1))
        alignB_r = np.zeros((len(r)-1))
        num_densB_r = np.zeros((len(r)-1))

        rad_arr = np.zeros((len(r)-1))
        #If exterior and interior surfaces defined, continue...
        
        #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
        for i in range(1, len(r)):

            #Min and max location across interface of current step
            min_r = r[i-1]
            max_r = r[i]

            #Calculate area of rectangle for current step
            area = np.pi * (max_r ** 2) - area_prev

            #Save total area of previous step sizes
            area_prev = np.pi * (max_r ** 2)

            #Find particles that are housed within current slice
            parts_inrange_fa = np.where((min_r<=radial_fa_dict['all']['r']) & (radial_fa_dict['all']['r']<=max_r))[0]
            partsA_inrange_fa = np.where((min_r<=radial_fa_dict['A']['r']) & (radial_fa_dict['A']['r']<=max_r))[0]
            partsB_inrange_fa = np.where((min_r<=radial_fa_dict['B']['r']) & (radial_fa_dict['B']['r']<=max_r))[0]


            #If at least 1 particle in slice, continue...
            if len(parts_inrange_fa)>0:

                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(radial_fa_dict['all']['fa'][parts_inrange_fa]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    act_press_r[i-1] = np.sum(radial_fa_dict['all']['fa'][parts_inrange_fa][parts_defined])
                    act_fa_r[i-1] = np.mean(radial_fa_dict['all']['fa'][parts_inrange_fa][parts_defined])
                    align_r[i-1] = np.mean(radial_fa_dict['all']['align'][parts_inrange_fa][parts_defined])
                    num_dens_r[i-1] = len(parts_defined)
                    #If area of slice is non-zero, calculate the pressure [F/A]
                    if area > 0:
                        act_press_r[i-1] = act_press_r[i-1]/area
                        num_dens_r[i-1] = num_dens_r[i-1]/area

                    partsA_defined = np.logical_not(np.isnan(radial_fa_dict['A']['fa'][partsA_inrange_fa]))
                    if len(partsA_defined)>0:
                        act_pressA_r[i-1] = np.sum(radial_fa_dict['A']['fa'][partsA_inrange_fa][partsA_defined])
                        act_faA_r[i-1] = np.mean(radial_fa_dict['A']['fa'][partsA_inrange_fa][partsA_defined])
                        alignA_r[i-1] = np.mean(radial_fa_dict['A']['align'][partsA_inrange_fa][partsA_defined])
                        num_densA_r[i-1] = len(partsA_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressA_r[i-1] = act_pressA_r[i-1]/area
                            num_densA_r[i-1] = num_densA_r[i-1]/area

                    partsB_defined = np.logical_not(np.isnan(radial_fa_dict['B']['fa'][partsB_inrange_fa]))
                    if len(partsB_defined)>0:
                        act_pressB_r[i-1] = np.sum(radial_fa_dict['B']['fa'][partsB_inrange_fa][partsB_defined])
                        act_faB_r[i-1] = np.mean(radial_fa_dict['B']['fa'][partsB_inrange_fa][partsB_defined])
                        alignB_r[i-1] = np.mean(radial_fa_dict['B']['align'][partsB_inrange_fa][partsB_defined])
                        #Calculate density
                        num_densB_r[i-1] = len(partsB_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressB_r[i-1] = act_pressB_r[i-1]/area
                            num_densB_r[i-1] = num_densB_r[i-1]/area



        com_radial_dict_fa = {'r': r[1:].tolist(), 'fa_press': {'all': act_press_r.tolist(), 'A': act_pressA_r.tolist(), 'B': act_pressB_r.tolist()}, 'fa': {'all': act_fa_r.tolist(), 'A': act_faA_r.tolist(), 'B': act_faB_r.tolist()}, 'align': {'all': align_r.tolist(), 'A': alignA_r.tolist(), 'B': alignB_r.tolist()}, 'num_dens': {'all': num_dens_r.tolist(), 'A': num_densA_r.tolist(), 'B': num_densB_r.tolist()}}
        return com_radial_dict_fa


    def radial_measurements(self, radial_stress_dict, radial_fa_dict):

        #X locations across interface for integration
        if self.hx_box<self.hy_box:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))
        elif self.hy_box<self.hx_box:
            r = np.linspace(0, self.hy_box, num=int((np.ceil(self.hy_box)+1)/3))
        else:
            r = np.linspace(0, self.hx_box, num=int((np.ceil(self.hx_box)+1)/3))

        #Pressure integrand components for each value of X
        int_stress_XX_r = np.zeros((len(r)-1))
        int_stress_YY_r = np.zeros((len(r)-1))
        int_stress_XY_r = np.zeros((len(r)-1))
        int_stress_YX_r = np.zeros((len(r)-1))
        int_press_r = np.zeros((len(r)-1))
        #act_fa_r = []
        #lat_r = []
        num_dens_r = np.zeros((len(r)-1))

        int_stressA_XX_r = np.zeros((len(r)-1))
        int_stressA_YY_r = np.zeros((len(r)-1))
        int_stressA_XY_r = np.zeros((len(r)-1))
        int_stressA_YX_r = np.zeros((len(r)-1))
        int_pressA_r = np.zeros((len(r)-1))

        #act_faA_r = []
        #latA_r = []
        num_densA_r = np.zeros((len(r)-1))

        int_stressB_XX_r = np.zeros((len(r)-1))
        int_stressB_YY_r = np.zeros((len(r)-1))
        int_stressB_XY_r = np.zeros((len(r)-1))
        int_stressB_YX_r = np.zeros((len(r)-1))
        int_pressB_r = np.zeros((len(r)-1))
        #act_faB_r = []
        #latB_r = []
        num_densB_r = np.zeros((len(r)-1))

        #If exterior and interior surfaces defined, continue...

        area_prev = 0

        #Pressure integrand components for each value of X
        act_press_r = np.zeros((len(r)-1))
        act_fa_r = np.zeros((len(r)-1))
        align_r = np.zeros((len(r)-1))
        num_dens_r = np.zeros((len(r)-1))

        act_pressA_r = np.zeros((len(r)-1))
        act_faA_r = np.zeros((len(r)-1))
        alignA_r = np.zeros((len(r)-1))
        num_densA_r = np.zeros((len(r)-1))

        act_pressB_r = np.zeros((len(r)-1))
        act_faB_r = np.zeros((len(r)-1))
        alignB_r = np.zeros((len(r)-1))
        num_densB_r = np.zeros((len(r)-1))

        rad_arr = np.zeros((len(r)-1))
        #If exterior and interior surfaces defined, continue...
        
        #For each step across interface, calculate pressure in that step's area (averaged over angle from CoM)
        for i in range(1, len(r)):

            #Min and max location across interface of current step
            min_r = r[i-1]
            max_r = r[i]

            #Calculate area of rectangle for current step
            area = np.pi * (max_r ** 2) - area_prev

            #Save total area of previous step sizes
            area_prev = np.pi * (max_r ** 2)


            #Find particles that are housed within current slice
            parts_inrange = np.where((min_r<=radial_stress_dict['all']['r']) & (radial_stress_dict['all']['r']<=max_r))[0]
            partsA_inrange = np.where((min_r<=radial_stress_dict['A']['r']) & (radial_stress_dict['A']['r']<=max_r))[0]
            partsB_inrange = np.where((min_r<=radial_stress_dict['B']['r']) & (radial_stress_dict['B']['r']<=max_r))[0]

            #Find particles that are housed within current slice
            parts_inrange_fa = np.where((min_r<=radial_fa_dict['all']['r']) & (radial_fa_dict['all']['r']<=max_r))[0]
            partsA_inrange_fa = np.where((min_r<=radial_fa_dict['A']['r']) & (radial_fa_dict['A']['r']<=max_r))[0]
            partsB_inrange_fa = np.where((min_r<=radial_fa_dict['B']['r']) & (radial_fa_dict['B']['r']<=max_r))[0]


            #If at least 1 particle in slice, continue...
            if len(parts_inrange)>0:

                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(radial_stress_dict['all']['XX'][parts_inrange]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    int_stress_XX_r[i-1] = np.sum((radial_stress_dict['all']['XX'][parts_inrange][parts_defined]))
                    int_stress_YY_r[i-1] = np.sum((radial_stress_dict['all']['YY'][parts_inrange][parts_defined]))
                    int_stress_XY_r[i-1] = np.sum((radial_stress_dict['all']['XY'][parts_inrange][parts_defined]))
                    int_stress_YX_r[i-1] = np.sum((radial_stress_dict['all']['YX'][parts_inrange][parts_defined]))
                    int_press_r[i-1] = np.sum((radial_stress_dict['all']['XX'][parts_inrange][parts_defined] + radial_stress_dict['all']['YY'][parts_inrange][parts_defined])/2)
                    #Calculate density
                    num_dens_r[i-1] = len(parts_defined)
                    #If area of slice is non-zero, calculate the pressure [F/A]
                    if area > 0:
                        int_press_r[i-1] = int_press_r[i-1]/area
                        num_dens_r[i-1] = num_dens_r[i-1]/area

                    partsA_defined = np.logical_not(np.isnan(radial_stress_dict['A']['XX'][partsA_inrange]))

                    if len(partsA_defined)>0:
                        int_stressA_XX_r[i-1] = np.sum((radial_stress_dict['A']['XX'][partsA_inrange][partsA_defined]))
                        int_stressA_YY_r[i-1] = np.sum((radial_stress_dict['A']['YY'][partsA_inrange][partsA_defined]))
                        int_stressA_XY_r[i-1] = np.sum((radial_stress_dict['A']['XY'][partsA_inrange][partsA_defined]))
                        int_stressA_YX_r[i-1] = np.sum((radial_stress_dict['A']['YX'][partsA_inrange][partsA_defined]))
                        int_pressA_r[i-1] = np.sum((radial_stress_dict['A']['XX'][partsA_inrange][partsA_defined] + radial_stress_dict['A']['YY'][partsA_inrange][partsA_defined])/2)
                        #Calculate density
                        num_densA_r[i-1] = len([partsA_defined])
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            int_pressA_r[i-1] = int_pressA_r[i-1]/area
                            num_densA_r[i-1] = num_densA_r[i-1]/area

                    partsB_defined = np.logical_not(np.isnan(radial_stress_dict['B']['XX'][partsB_inrange]))
                    if len(partsB_defined)>0:
                        int_stressB_XX_r[i-1] = np.sum((radial_stress_dict['B']['XX'][partsB_inrange][partsB_defined]))
                        int_stressB_YY_r[i-1] = np.sum((radial_stress_dict['B']['YY'][partsB_inrange][partsB_defined]))
                        int_stressB_XY_r[i-1] = np.sum((radial_stress_dict['B']['XY'][partsB_inrange][partsB_defined]))
                        int_stressB_YX_r[i-1] = np.sum((radial_stress_dict['B']['YX'][partsB_inrange][partsB_defined]))
                        int_pressB_r[i-1] = np.sum((radial_stress_dict['B']['XX'][partsB_inrange][partsB_defined] + radial_stress_dict['B']['YY'][partsB_inrange][partsB_defined])/2)
                        #Calculate density
                        num_densB_r[i-1] = len([partsB_defined])
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            int_pressB_r[i-1] = int_pressB_r[i-1]/area
                            num_densB_r[i-1] = num_densB_r[i-1]/area
                #If the force is defined, continue...
                parts_defined = np.logical_not(np.isnan(radial_fa_dict['all']['fa'][parts_inrange]))

                if len(parts_defined)>0:
                    #Calculate total active force normal to interface in slice
                    act_press_r[i-1] = np.sum(radial_fa_dict['all']['fa'][parts_inrange_fa][parts_defined])
                    act_fa_r[i-1] = np.mean(radial_fa_dict['all']['fa'][parts_inrange_fa][parts_defined])
                    align_r[i-1] = np.mean(radial_fa_dict['all']['align'][parts_inrange_fa][parts_defined])
                    num_dens_r[i-1] = len(parts_defined)
                    #If area of slice is non-zero, calculate the pressure [F/A]
                    if area > 0:
                        act_press_r[i-1] = act_press_r[i-1]/area
                        num_dens_r[i-1] = num_dens_r[i-1]/area

                    partsA_defined = np.logical_not(np.isnan(radial_fa_dict['A']['fa'][partsA_inrange_fa]))
                    if len(partsA_defined)>0:
                        act_pressA_r[i-1] = np.sum(radial_fa_dict['A']['fa'][partsA_inrange_fa][partsA_defined])
                        act_faA_r[i-1] = np.mean(radial_fa_dict['A']['fa'][partsA_inrange_fa][partsA_defined])
                        alignA_r[i-1] = np.mean(radial_fa_dict['A']['align'][partsA_inrange_fa][partsA_defined])
                        num_densA_r[i-1] = len(partsA_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressA_r[i-1] = act_pressA_r[i-1]/area
                            num_densA_r[i-1] = num_densA_r[i-1]/area

                    partsB_defined = np.logical_not(np.isnan(radial_fa_dict['B']['fa'][partsB_inrange_fa]))
                    if len(partsB_defined)>0:
                        act_pressB_r[i-1] = np.sum(radial_fa_dict['B']['fa'][partsB_inrange_fa][partsB_defined])
                        act_faB_r[i-1] = np.mean(radial_fa_dict['B']['fa'][partsB_inrange_fa][partsB_defined])
                        alignB_r[i-1] = np.mean(radial_fa_dict['B']['align'][partsB_inrange_fa][partsB_defined])
                        #Calculate density
                        num_densB_r[i-1] = len(partsB_defined)
                        #If area of slice is non-zero, calculate the pressure [F/A]
                        if area > 0:
                            act_pressB_r[i-1] = act_pressB_r[i-1]/area
                            num_densB_r[i-1] = num_densB_r[i-1]/area



        com_radial_dict_fa = {'r': r[1:].tolist(), 'fa_press': {'all': act_press_r.tolist(), 'A': act_pressA_r.tolist(), 'B': act_pressB_r.tolist()}, 'fa': {'all': act_fa_r.tolist(), 'A': act_faA_r.tolist(), 'B': act_faB_r.tolist()}, 'align': {'all': align_r.tolist(), 'A': alignA_r.tolist(), 'B': alignB_r.tolist()}, 'num_dens': {'all': num_dens_r.tolist(), 'A': num_densA_r.tolist(), 'B': num_densB_r.tolist()}}
        com_radial_dict = {'r': r[1:].tolist(), 'all': {'XX': int_stress_XX_r.tolist(), 'YY': int_stress_YY_r.tolist(), 'XY': int_stress_XY_r.tolist(), 'YX': int_stress_YX_r.tolist(), 'press': int_press_r.tolist(), 'num_dens': num_dens_r.tolist()}, 'A': {'XX': int_stressA_XX_r.tolist(), 'YY': int_stressA_YY_r.tolist(), 'XY': int_stressA_XY_r.tolist(), 'YX': int_stressA_YX_r.tolist(), 'press': int_pressA_r.tolist(), 'num_dens': num_densA_r.tolist()}, 'B': {'XX': int_stressB_XX_r.tolist(), 'YY': int_stressB_YY_r.tolist(), 'XY': int_stressB_XY_r.tolist(), 'YX': int_stressB_YX_r.tolist(), 'press': int_pressB_r.tolist(), 'num_dens': num_densB_r.tolist()}}
        return com_radial_dict, com_radial_dict_fa

    def single_msd(self, pos_prev, displace_dict):

        displace_x_typ0 = displace_dict['A']['x']
        displace_y_typ0 = displace_dict['A']['y']
        displace_r_typ0 = displace_dict['A']['mag']

        displace_x_typ1 = displace_dict['B']['x']
        displace_y_typ1 = displace_dict['B']['y']
        displace_r_typ1 = displace_dict['B']['mag']

        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]
        
        difx_typ0, dify_typ0, difr_typ0 = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)
        
        periodic_right = np.where(difx_typ0>self.hx_box)[0]
        periodic_left = np.where(difx_typ0<-self.hx_box)[0]

        periodic_top = np.where(dify_typ0>self.hy_box)[0]
        periodic_bot = np.where(dify_typ0<-self.hy_box)[0]

        stop
        #periodic_right * h_box
        difx_typ0_sim, dify_typ0_sim, difr_typ0_sim = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)

        difx_typ0_bulk, dify_typ0_bulk, difr_typ0_bulk = self.utility_functs.sep_dist_arr(self.pos[typ0ind], pos_prev[typ0ind], difxy=True)
        
        difx_typ0_bulk
        difx_typ1_sim, dify_typ1_sim, difr_typ1_sim = self.utility_functs.sep_dist_arr(self.pos[typ1ind], pos_prev[typ1ind], difxy=True)
        difx_typ1_bulk, dify_typ1_bulk, difr_typ1_bulk = self.utility_functs.sep_dist_arr(self.pos[typ1ind], pos_prev[typ1ind], difxy=True)
        
        displace_x_typ0 = np.append(displace_x_typ0, np.mean(difx_typ0_bulk))
        displace_y_typ0 = np.append(displace_y_typ0, np.mean(dify_typ0_bulk))
        displace_r_typ0 = np.append(displace_r_typ0, np.mean(difr_typ0_bulk))

        displace_x_typ1 = np.append(displace_x_typ1, np.mean(difx_typ1_bulk))
        displace_y_typ1 = np.append(displace_y_typ1, np.mean(dify_typ1_bulk))
        displace_r_typ1 = np.append(displace_r_typ1, np.mean(difr_typ1_bulk))

        x_max_typ0 = np.max(self.pos[typ0ind,0])
        y_max_typ0 = np.max(self.pos[typ0ind,1])

        x_max_typ1 = np.max(self.pos[typ1ind,0])
        y_max_typ1 = np.max(self.pos[typ1ind,1])

        msd_x_typ0 = np.zeros(len(typ0ind))
        msd_y_typ0 = np.zeros(len(typ0ind))
        msd_r_typ0 = np.zeros(len(typ0ind))

        for i in range(0, len(displace_x_typ0)):
            msd_x_typ0 = np.append(msd_x_typ0, np.sum(displace_x_typ0[i:i+1]) )
            msd_y_typ0 = np.append(msd_y_typ0, np.sum(displace_y_typ0[i:i+1]) )
            msd_r_typ0 = np.append(msd_r_typ0, np.sum(displace_r_typ0[i:i+1]) )

        for i in range(0, len(displace_x_typ1)):
            msd_x_typ1 = np.append(msd_x_typ1, np.sum(displace_x_typ1[i:i+1]) )
            msd_y_typ1 = np.append(msd_y_typ1, np.sum(displace_y_typ1[i:i+1]) )
            msd_r_typ1 = np.append(msd_r_typ1, np.sum(displace_r_typ1[i:i+1]) )

        displace_dict = {'A': {'x': displace_x_typ0, 'y': displace_y_typ0, 'mag': displace_r_typ0}, 'B': {'x': displace_x_typ1, 'y': displace_y_typ1, 'mag': displace_r_typ1} }
        msd_dict = {'A': {'x': msd_x_typ0, 'y': msd_y_typ0, 'mag': msd_r_typ0}, 'B': {'x': msd_x_typ1, 'y': msd_y_typ1, 'mag': msd_r_typ1} }
        return displace_dict, msd_dict
    def nearest_neighbors_penetrate(self):
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

        # Position and orientation arrays of type A particles in respective phase
        typ0ind = np.where(self.typ==0)[0]
        pos_A=self.pos[typ0ind]                               # Find positions of type 0 particles
        px_A=self.px[typ0ind]
        py_A=self.py[typ0ind]

        # Position and orientation arrays of type B particles in respective phase
        typ1ind = np.where(self.typ==1)[0]
        pos_B=self.pos[typ1ind]
        px_B=self.px[typ1ind]
        py_B=self.py[typ1ind]
        
        # Neighbor list query arguments to find interacting particles
        query_args = dict(mode='ball', r_min = 0.1, r_max = self.r_cut)#r_max=self.theory_functs.conForRClust(peNet_int-45., self.eps) * 1.0)

        # Locate potential neighbor particles by type in the dense phase
        system_A = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_A))
        system_B = freud.AABBQuery(self.f_box, self.f_box.wrap(pos_B))
        
        # Generate neighbor list of dense phase particles (per query args) of respective type (A or B) neighboring bulk phase reference particles of respective type (A or B)
        AA_nlist = system_A.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        AB_nlist = system_A.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        BA_nlist = system_B.query(self.f_box.wrap(pos_A), query_args).toNeighborList()
        BB_nlist = system_B.query(self.f_box.wrap(pos_B), query_args).toNeighborList()
        
        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type A bulk particles
        AA_neigh_ind = np.array([], dtype=int)
        AA_num_neigh = np.array([])
        AA_dot = np.array([])
        
        #Loop over neighbor pairings of A-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in AA_nlist.query_point_indices:
                if i not in AA_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AA_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AA_num_neigh = np.append(AA_num_neigh, len(loc))
                    AA_neigh_ind = np.append(AA_neigh_ind, int(i))

                    
                    AA_dot = np.append(AA_dot, np.sum((px_A[i]*px_A[AA_nlist.point_indices[loc]]+py_A[i]*py_A[AA_nlist.point_indices[loc]])/(((px_A[i]**2+py_A[i]**2)**0.5)*((px_A[AA_nlist.point_indices[loc]]**2+py_A[AA_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                AA_num_neigh = np.append(AA_num_neigh, 0)
                AA_neigh_ind = np.append(AA_neigh_ind, int(i))
                AA_dot = np.append(AA_dot, 0)
        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type A bulk particles
        BA_neigh_ind = np.array([], dtype=int)
        BA_num_neigh = np.array([])
        BA_dot = np.array([])

        #Loop over neighbor pairings of B-A neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_A)):
            if i in BA_nlist.query_point_indices:
                if i not in BA_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BA_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BA_num_neigh = np.append(BA_num_neigh, len(loc))
                    BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                    BA_dot = np.append(BA_dot, np.sum((px_A[i]*px_B[BA_nlist.point_indices[loc]]+py_A[i]*py_B[BA_nlist.point_indices[loc]])/(((px_A[i]**2+py_A[i]**2)**0.5)*((px_B[BA_nlist.point_indices[loc]]**2+py_B[BA_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                BA_num_neigh = np.append(BA_num_neigh, 0)
                BA_neigh_ind = np.append(BA_neigh_ind, int(i))
                BA_dot = np.append(BA_dot, 0)

        #Initiate empty arrays for finding nearest A neighboring dense particles surrounding type B bulk particles
        AB_neigh_ind = np.array([], dtype=int)
        AB_num_neigh = np.array([])
        AB_dot = np.array([])

        #Loop over neighbor pairings of A-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in AB_nlist.query_point_indices:
                if i not in AB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(AB_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    AB_num_neigh = np.append(AB_num_neigh, len(loc))
                    AB_dot = np.append(AB_dot, np.sum((px_B[i]*px_A[AB_nlist.point_indices[loc]]+py_B[i]*py_A[AB_nlist.point_indices[loc]])/(((px_B[i]**2+py_B[i]**2)**0.5)*((px_A[AB_nlist.point_indices[loc]]**2+py_A[AB_nlist.point_indices[loc]]**2)**0.5))))
                    AB_neigh_ind = np.append(AB_neigh_ind, int(i))
            else:
                #Save nearest neighbor information to array
                AB_num_neigh = np.append(AB_num_neigh, 0)
                AB_neigh_ind = np.append(AB_neigh_ind, int(i))
                AB_dot = np.append(AB_dot, 0)

        #Initiate empty arrays for finding nearest B neighboring dense particles surrounding type B bulk particles
        BB_neigh_ind = np.array([], dtype=int)
        BB_num_neigh = np.array([])
        BB_dot = np.array([])

        #Loop over neighbor pairings of B-B neighbor pairs to calculate number of nearest neighbors
        for i in range(0, len(pos_B)):
            if i in BB_nlist.query_point_indices:
                if i not in BB_neigh_ind:
                    # Find neighbors list IDs where i is reference particle
                    loc = np.where(BB_nlist.query_point_indices==i)[0]

                    #Save nearest neighbor information to array
                    BB_num_neigh = np.append(BB_num_neigh, len(loc))
                    BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                    BB_dot = np.append(BB_dot, np.sum((px_B[i]*px_B[BB_nlist.point_indices[loc]]+py_B[i]*py_B[BB_nlist.point_indices[loc]])/(((px_B[i]**2+py_B[i]**2)**0.5)*((px_B[BB_nlist.point_indices[loc]]**2+py_B[BB_nlist.point_indices[loc]]**2)**0.5))))
            else:
                #Save nearest neighbor information to array
                BB_num_neigh = np.append(BB_num_neigh, 0)
                BB_neigh_ind = np.append(BB_neigh_ind, int(i))
                BB_dot = np.append(BB_dot, 0)

        
        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with B nearest neighbors
        allB_num_neigh = AB_num_neigh + BB_num_neigh
        allB_dot = BB_dot + AB_dot

        # Save neighbor and local orientational order to arrays for all reference particles of the respective phase with A nearest neighbors
        allA_num_neigh = AA_num_neigh + BA_num_neigh
        allA_dot = AA_dot + BA_dot

        # Save neighbor and local orientational order to arrays for all B reference particles of the respective phase with all nearest neighbors
        Ball_num_neigh = np.append(BA_num_neigh, BB_num_neigh)
        Ball_dot = np.append(BA_dot, BB_dot)

        # Save neighbor and local orientational order to arrays for all A reference particles of the respective phase with all nearest neighbors
        Aall_num_neigh = np.append(AB_num_neigh, AA_num_neigh)
        Aall_dot = np.append(AB_dot, AA_dot)

        # Save neighbor, local orientational order, and position to arrays for all bulk reference particles with all nearest neighbors
        allall_num_neigh = np.append(allA_num_neigh, allB_num_neigh)
        allall_dot = np.append(allA_dot, allB_dot)
        allall_pos_x = np.append(pos_A[:,0], pos_B[:,0])
        allall_pos_y = np.append(pos_A[:,1], pos_B[:,1])

        # Average orientational order over all neighbors for each particle
        for i in range(0, len(allB_dot)):
            if allB_num_neigh[i]>0:
                allB_dot[i] = allB_dot[i]/allB_num_neigh[i]

        for i in range(0, len(allA_dot)):
            if allA_num_neigh[i]>0:
                allA_dot[i] = allA_dot[i]/allA_num_neigh[i]

        for i in range(0, len(allall_dot)):
            if allall_num_neigh[i]>0:
                allall_dot[i] = allall_dot[i]/allall_num_neigh[i]

        for i in range(0, len(Aall_dot)):
            if Aall_num_neigh[i]>0:
                Aall_dot[i] = Aall_dot[i]/Aall_num_neigh[i]

        for i in range(0, len(Ball_dot)):
            if Ball_num_neigh[i]>0:
                Ball_dot[i] = Ball_dot[i]/Ball_num_neigh[i]

        # Create output dictionary for statistical averages of total nearest neighbor numbers on each particle per phase/activity pairing
        neigh_stat_dict = {'all-all': {'mean': np.mean(allall_num_neigh), 'std': np.std(allall_num_neigh)}, 'all-A': {'mean': np.mean(allA_num_neigh), 'std': np.std(allA_num_neigh)}, 'all-B': {'mean': np.mean(allB_num_neigh), 'std': np.std(allB_num_neigh)}, 'A-A': {'mean': np.mean(AA_num_neigh), 'std': np.std(AA_num_neigh)}, 'A-B': {'mean': np.mean(AB_num_neigh), 'std': np.std(AB_num_neigh)}, 'B-B': {'mean': np.mean(BB_num_neigh), 'std': np.std(BB_num_neigh)}}


        # Create output dictionary for statistical averages of total nearest neighbor orientational correlation on each particle per phase/activity pairing
        ori_stat_dict = {'all-all': {'mean': np.mean(allall_dot), 'std': np.std(allall_dot)}, 'all-A': {'mean': np.mean(allA_dot), 'std': np.std(allA_dot)}, 'all-B': {'mean': np.mean(allB_dot), 'std': np.std(allB_dot)}, 'A-A': {'mean': np.mean(AA_dot), 'std': np.std(AA_dot)}, 'A-B': {'mean': np.mean(AB_dot), 'std': np.std(AB_dot)}, 'B-B': {'mean': np.mean(BB_dot), 'std': np.std(BB_dot)}}

        # Create output dictionary for plotting of nearest neighbor information of each particle per phase/activity pairing and their respective x-y locations
        neigh_plot_dict = {'all-all': {'neigh': allall_num_neigh, 'ori': allall_dot, 'x': allall_pos_x, 'y': allall_pos_y}, 'all-A': {'neigh': allA_num_neigh, 'ori': allA_dot, 'x': pos_A[:,0], 'y': pos_A[:,1]}, 'all-B': {'neigh': allB_num_neigh, 'ori': allB_dot, 'x': pos_B[:,0], 'y': pos_B[:,1]}, 'A-all': {'neigh': Aall_num_neigh, 'ori': Aall_dot, 'x': self.pos[:,0], 'y': self.pos[:,1]}, 'B-all': {'neigh': Ball_num_neigh, 'ori': Ball_dot, 'x': self.pos[:,0], 'y': self.pos[:,1]}}

        
        return neigh_stat_dict, ori_stat_dict, neigh_plot_dict

    def penetration_depth(self, start_dict, pos_prev, vertical_shift, dify_long):

        typ0ind = np.where(self.typ==0)[0]
        typ1ind = np.where(self.typ==1)[0]

        wall_x = np.amax(self.pos[typ0ind,0])

        if ((self.pos[typ1ind,0] <= (wall_x - 1.0)) & (self.pos[typ1ind,0] >= -(wall_x - 1.0))) | ((pos_prev[typ1ind,0] <= (wall_x - 1.0)) & (pos_prev[typ1ind,0] >= -(wall_x - 1.0))):
            if ((pos_prev[typ1ind,0] > (wall_x - 1.0)) | (pos_prev[typ1ind,0] < -(wall_x - 1.0))) & ((self.pos[typ1ind,0] <= (wall_x - 1.0)) & (self.pos[typ1ind,0] >= -(wall_x - 1.0))):
                action = 'enter'
                start_x = self.pos[typ1ind,0]
                start_y = self.pos[typ1ind,1]

            elif (pos_prev[typ1ind,0] <= (wall_x - 1.0)) & (pos_prev[typ1ind,0] >= -(wall_x - 1.0)) & ((self.pos[typ1ind,0] > (wall_x - 1.0)) & (self.pos[typ1ind,0] < -(wall_x - 1.0))): 
                action = 'exit'
                start_x = start_dict['x']
                start_y = start_dict['y']
            else:
                action = 'bulk'  
                start_x = start_dict['x']
                start_y = start_dict['y']              

            if np.abs(self.pos[typ1ind,1]-pos_prev[typ1ind,1])>self.hy_box:
                if (self.pos[typ1ind,1]<0) & (pos_prev[typ1ind,1]>0):
                    vertical_shift = dify_long + self.hy_box-start_y
                    dify_long = vertical_shift
                    start_y = -self.hy_box
                elif (self.pos[typ1ind,1]>0) & (pos_prev[typ1ind,1]<0):
                    vertical_shift = dify_long + -self.hy_box-start_y
                    dify_long = vertical_shift
                    start_y = self.hy_box
                             
            if start_x < 0:
                penetration_depth = wall_x+self.pos[typ1ind,0]
                difx = wall_x + self.pos[typ1ind,0]
            else:
                penetration_depth = wall_x-self.pos[typ1ind,0]
                difx = self.pos[typ1ind,0]  - wall_x

            #dify =  (self.pos[typ1ind,1] + vertical_shift) - start_y
            dify =  dify_long + (self.pos[typ1ind,1]- start_y)
            
            difr = (difx ** 2 + dify **2 ) ** 0.5 
            

            difx_prev = self.utility_functs.sep_dist_x(self.pos[typ1ind,0], pos_prev[typ1ind,0])
            dify_prev = self.utility_functs.sep_dist_y(self.pos[typ1ind,1], pos_prev[typ1ind,1])
            difr_prev = (difx_prev ** 2 + dify_prev ** 2 ) ** 0.5
            
            
            MSD = ( difx ** 2 + dify ** 2 )
        else:
            action = 'gas'
            MSD = 0
            dify = 0
            difx_prev = 0
            dify_prev = 0
            difr_prev = 0
            difx = 0
            difr = 0
            penetration_depth = 0
            start_x = 0
            start_y = 0
            vertical_shift = 0
            dify_long = 0

            difx_prev = self.utility_functs.sep_dist_x(self.pos[typ1ind,0], pos_prev[typ1ind,0])
            dify_prev = self.utility_functs.sep_dist_y(self.pos[typ1ind,1], pos_prev[typ1ind,1])
            difr_prev = (difx_prev ** 2 + dify_prev ** 2 ) ** 0.5

        penetration_dict = {'wall_x': wall_x, 'displace': {'x': difx_prev, 'y': dify_prev, 'r': difr_prev}, 'total_displace':{'x': difx, 'y': dify, 'r': difr, 'MSD': MSD, 'depth': penetration_depth}, 'action': action}
        start_dict = {'x': start_x, 'y': start_y}

        return penetration_dict, start_dict, vertical_shift, dify_long
    
    def phase_count(self, phase_dict):
        '''
        Purpose: Takes the phase ids of each bin and counts the number of bins of each phase

        Inputs:
        phase_dict: dictionary of arrays labeling phase of each bin and particle

        Outputs:
        count_dict: dictionary containing the number of bins of each phase ['bulk' (0), 'int' (1), or 'gas' (2)]
        '''

        # Array (NBins_x, NBins_y) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phaseBin = phase_dict['bin']

        # Array (partNum) that identifies whether bin is bulk (0), interface (1), or gas (2)
        phasePart = phase_dict['part']

        #Label individual particle phases from identified bin phases
        int_num=0
        bulk_num=0
        gas_num=0

        # Loop over all bins
        for ix in range(0, self.NBins_x):
            for iy in range(0, self.NBins_y):

                # Count number of each bin
                if phaseBin[ix][iy]==1:
                    int_num+=1
                elif phaseBin[ix][iy]==0:
                    bulk_num+=1
                elif phaseBin[ix][iy]==2:
                    gas_num+=1

        # Dictionary containing the number of bins of each phase
        count_dict = {'bulk': bulk_num, 'int': int_num, 'gas': gas_num}

        return count_dict