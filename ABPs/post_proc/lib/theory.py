
import sys
import os

from gsd import hoomd
import freud
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage

import numpy as np
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


#from symfit import parameters, variables, sin, cos, Fit

from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import utility

class theory:
    def __init__(self):
        # Set some constants
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)
        self.kT = 1.0                        # temperature
        self.threeEtaPiSigma = 1.0           # drag coefficient
        self.sigma = 1.0                     # particle diameter
        self.D_t = self.kT / self.threeEtaPiSigma      # translational diffusion constant
        self.D_r = (3.0 * self.D_t) / (self.sigma**2)  # rotational diffusion constant
        self.tauBrown = (self.sigma**2) / self.D_t     # brownian time scale (invariant)

    def compPeNet(self, xA, peA, peB):
        peNet = (peA * xA) + (peB * (1.-xA))
        return peNet

    def avgCollisionForce(self, pe):
        '''
        Purpose: Average compressive force experienced by a reference particle in the
        bulk dense phase due to neighboring active forces computed from the integral
        of possible orientations

        Inputs: Net activity of system

        Output: Average magnitude of compressive forces experienced by a bulk particle
        '''

        peCritical = 45.

        if pe < peCritical:
            pe = 0
        else:
            pe -= peCritical

        # A vector sum of the six nearest neighbors
        magnitude = 1.92

        return (magnitude * pe)

    def avgCollisionForce2(self, peA, peB, beta_A, beta_B):
        '''
        Purpose: Average compressive force experienced by a reference particle in the
        bulk dense phase due to neighboring active forces computed from the integral
        of possible orientations

        Inputs: Net activity of system

        Output: Average magnitude of compressive forces experienced by a bulk particle
        '''

        peCritical = 45

        if peA < peCritical:
            peA = 0
        else:
            peA -= peCritical

        if peB < peCritical:
            peB = 0
        else:
            peB -= peCritical

        return (peA/peB) * peA * beta_A + beta_B * peB

    # Calculate cluster radius
    def conForRClust(self, pe, eps):
        '''
        Purpose: Compute analytical radius of the custer given activity and softness

        Inputs:
            pe: net activity (peclet number)
            eps: softness (magnitude of repulsive interparticle force)

        Output: cluster radius (simulation distance units)
        '''
        out = []
        r = 1.112
        skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
        for j in skip:
            while self.ljForce(r, eps) < self.avgCollisionForce(pe):
                r -= j
            r += j
        out = r
        return out

    # Calculate cluster radius
    def conForRClust2(self, peA, peB, beta_A, beta_B, eps):
        '''
        Purpose: Compute analytical radius of the custer given activity and softness

        Inputs:
            pe: net activity (peclet number)
            eps: softness (magnitude of repulsive interparticle force)

        Output: cluster radius (simulation distance units)
        '''
        out = []
        r = 1.112
        skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
        for j in skip:
            while self.ljForce(r, eps) < self.avgCollisionForce2(peA, peB, beta_A, beta_B):
                r -= j
            r += j
        out = r
        return out

    def ljForce(self, r, eps, sigma=1.):
        '''
        Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
        and separation distance of 2 particles to compute magnitude of lennard-jones force experienced
        by each

        Inputs:
            r: Separation distance in simulation units
            epsilon: magnitude of lennard-jones potential
            sigma: particle diameter (default=1.0)

        Output: lennard jones force (dU)
        '''

        #Dimensionless distance unit
        div = (self.sigma/r)

        dU = (24. * eps / r) * ((2*(div**12)) - (div)**6)
        return dU

    # Lennard-Jones pressure
    def ljPress(self, r, pe, eps, sigma=1.):
        '''
        Purpose: Take epsilon (magnitude of lennard-jones force), sigma (particle diameter),
        activity (pe), and separation distance (r) of 2 particles to compute pressure from
        avg compressive active forces from neighbors

        Inputs:
            r: Separation distance in simulation units
            epsilon: magnitude of lennard-jones potential
            pe: activity (peclet number)
            sigma: particle diameter (default=1.0)

        Output: Analytical virial pressure (see monodisperse paper for derivation)
        '''
        #Area fraction at HCP
        phiCP = np.pi / (2. * np.sqrt(3.))

        # LJ force
        ljF = self.avgCollisionForce(pe)

        return (2. *np.sqrt(3) * ljF / r)

    # Calculate dense phase area fraction from lattice spacing
    def latToPhi(self, latIn):
        '''
        Purpose: Compute analytical area fraction of the dense phase given the lattice
        spacing.

        Inputs:
            latIn: lattice spacing

        Output: dense phase area fraction
        '''
        phiCP = np.pi / (2. * np.sqrt(3.))
        return phiCP / (latIn**2)


    #Calculate gas phase area fraction
    def compPhiG(self, pe, a, kap=4.5):
        '''
        Purpose: Compute analytical area fraction of the gas phase at steady state
        given activity and lattice spacing

        Inputs:
            pe: net activity (peclet number)
            a: lattice spacing
            kap: fitting parameter (default=4.5, shown by Redner)
            sig: particle diameter (default=1.0)

        Output: Area fraction of the gas phase at steady state
        '''
        num = 3. * (np.pi**2) * kap * self.sigma
        den = 4. * pe * a
        return num / den

    def computeTauLJ(self, epsilon):
        '''
        Purpose: Take epsilon (magnitude of lennard-jones force) and compute lennard-jones
        time unit of simulation

        Inputs: epsilon

        Output: lennard-jones time unit
        '''
        tauLJ = ((self.sigma**2) * self.threeEtaPiSigma) / epsilon
        return tauLJ

    def getLat(self, peNet, eps):
        '''
        Purpose: Take epsilon (magnitude of lennard-jones force) and net activity to
        compute lattice spacing as derived analytically (force balance of repulsive LJ force
        and compressive active force)

        Inputs:
            peNet: net activity of system
            epsilon: magnitude of lennard-jones potential

        Output: average lattice spacing of system
        '''

        #If system is passive, output cut-off radius
        if peNet == 0:
            return 2.**(1./6.)
        out = []
        r = 1.112
        skip = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

        #Loop through to find separation distance (r) where lennard-jones force (jForce)
        #approximately equals compressive active force (avgCollisionForce)
        for j in skip:
            while self.ljForce(r, eps) < self.avgCollisionForce(peNet):
                r -= j
            r += j

        return r
    def computeFLJ2(self, r, x1, y1, x2, y2, eps, l_box):
        f = (24. * eps / r) * ( (2*((self.sigma/r)**12)) - ((self.sigma/r)**6) )
        utility_functs = utility.utility(l_box)
        difx = utility_functs.sep_dist(x2, x1)
        dify = utility_functs.sep_dist(y2, y1)

        fx = f * difx / r
        fy = f * dify / r
        return fx, fy

    def computeFLJ(self, difr, difx, dify, eps):
        f = (24. * eps / difr) * ( (2*((self.sigma/difr)**12)) - ((self.sigma/difr)**6) )

        fx = f * difx / difr
        fy = f * dify / difr
        return fx, fy

    def computeFLJ_arr(self, difr, difx, dify, eps):
        fx = np.zeros(len(difr))
        fy = np.zeros(len(difr))
        for i in range(0, len(difr)):

            f = (24. * eps / difr[i]) * ( (2*((self.sigma/difr[i])**12)) - ((self.sigma/difr[i])**6) )

            fx[i] = f * difx[i] / difr[i]
            fy[i] = f * dify[i] / difr[i]
        return fx, fy

    def computeTauPerTstep(self, epsilon, mindt=0.000001):
        '''
        Purpose: Take epsilon (magnitude of lennard-jones force), and output the amount
        of Brownian time units per time step in LJ units

        Inputs:
            epsilon: magnitude of lennard-jones potential
            mindt: time step in LJ units (default=0.000001)

        Output: lennard jones force (dU)
        '''

        tstepPerTau = float(epsilon / (self.kT * mindt))
        return 1. / tstepPerTau
