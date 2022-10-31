#!/usr/bin/env python3

# Initial imports
import sys
import os

import numpy as np

import theory

class run_sim:
    def __init__(self, hoomdPath, runFor, dumpFreq, partPercA, peA, peB, partNum, intPhi, eps, aspect_ratio, seed1, seed2, seed3, seed4, seed5):

        # Read in bash arguments
        self.hoomdPath = hoomdPath # path to where you installed hoomd-blue '/.../hoomd-blue/build/'

        self.runFor = runFor           # simulation length (in tauLJ)
        self.dumpFreq = dumpFreq        # how often to dump data

        self.partPercA = partPercA      # percentage of A particles
        self.partFracA = float(partPercA) / 100.0  # fraction of particles that are A

        self.peA = peA                  # activity of A particles
        self.peB = peB                   # activity of B particles

        self.partNum = partNum           # Number of particles in system
        self.partNumA = int(self.partNum * self.partFracA)         # get the total number of A particles
        self.partNumB = int(self.partNum - self.partNumA)             # get the total number of B particles

        self.intPhi = intPhi                 # system area fraction (integer, i.e. 45, 65, etc.)
        self.phi = float(intPhi) / 100.0     # system area fraction (decimal, i.e. 0.45, 0.65, etc.)
        self.eps = eps                     # epsilon (potential well depth for LJ potential)

        self.seed1 = seed1              # seed for position
        self.seed2 = seed2                # seed for bd equilibration
        self.seed3 = seed3                # seed for initial orientations
        self.seed4 = seed4               # seed for A activity
        self.seed5 = seed5                # seed for B activity

        #aspect ratio
        self.aspect_ratio = aspect_ratio
        dim_list = aspect_ratio.split(':')
        if len(dim_list)==2:
            try:
                self.length = int(dim_list[0])
                self.width = int(dim_list[1])
            except:
                print('aspect ratio must have 2 integers separated separated by a colon, i.e. 2:1 where the x-dimension box length is twice as large as the y-dimension box width')

        # Set some constants
        self.kT = 1.0                        # temperature
        self.threeEtaPiSigma = 1.0           # drag coefficient
        self.sigma = 1.0                     # particle diameter
        self.D_t = self.kT / self.threeEtaPiSigma      # translational diffusion constant
        self.D_r = (3.0 * self.D_t) / (self.sigma**2)  # rotational diffusion constant
        self.tauBrown = (self.sigma**2) / self.D_t     # brownian time scale (invariant)
        self.r_cut = 2**(1./6.)

        self.theory_functs = theory.theory()

        # Compute parameters from activities
        if self.peA != 0:                        # A particles are NOT Brownian
            self.epsA = eps
            tauA = self.theory_functs.computeTauLJ(self.epsA)
        else:                               # A particles are Brownian
            self.epsA = eps#kT
            tauA = self.theory_functs.computeTauLJ(self.epsA)

        if self.peB != 0:                        # B particles are NOT Brownian
            self.epsB=eps
            tauB = self.theory_functs.computeTauLJ(self.epsB)
        else:                               # B particles are Brownian
            self.epsB = eps#kT
            tauB = self.theory_functs.computeTauLJ(self.epsB)

        #epsAB = (epsA + epsB + 1) / 2.0             # AB interaction well depth
        self.epsAB=self.epsA                                  # assign AB interaction well depth to same as A and B
        self.tauLJ = (tauA if (tauA <= tauB) else tauB)  # use the smaller tauLJ.  Doesn't matter since these are the same
        self.epsA = (self.epsA if (self.epsA >= self.epsB) else self.epsB)   # use the larger epsilon. Doesn't matter since these are the same

        self.dt = 0.000001 * self.tauLJ                        # timestep size.  I use 0.000001 for dt=tauLJ* (eps/10^6) generally
        self.simLength = self.runFor * self.tauBrown               # how long to run (in tauBrown)
        self.simTauLJ = self.simLength / self.tauLJ                # how long to run (in tauLJ)
        self.totTsteps = int(self.simLength / self.dt)             # how many tsteps to run
        self.numDumps = float(self.simLength / 0.025)           # dump data every 0.1 tauBrown.

        if self.dumpFreq==0:
            self.dumpFreq = float(totTsteps / numDumps)      # normalized dump frequency.
            self.dumpFreq = int(dumpFreq)                    # ensure this is an integer

        print("Brownian tau in use:"+str(self.tauBrown))
        print("Lennard-Jones tau in use:"+str(self.tauLJ))
        print("Timestep in use:"+str(self.dt))
        print("A-epsilon in use:"+str(self.epsA))
        print("B-epsilon in use:"+str(self.epsB))
        print("AB-epsilon in use:"+str(self.epsAB))
        print("Total number of timesteps:"+str(self.totTsteps))
        print("Total number of output frames:"+str(self.numDumps))
        print("File dump frequency:"+str(self.dumpFreq))

        self.beta_A = 1.0
        self.beta_B = 2.3



    def random_init(self):

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated


        # Initialize system
        hoomd.context.initialize()

        if self.length != self.width:
            area_ratio = self.length * self.width

            box_area = self.partNum/self.phi
            box_length = (box_area/area_ratio)**0.5
            lx = self.length*box_length
            ly = self.width * box_length
            set_box = hoomd.data.boxdim(Lx=lx, Ly=ly, Lz=0, dimensions=2)

            system = hoomd.deprecated.init.create_random(N = self.partNum,
                                                         name = 'A',
                                                         min_dist = 0.70,
                                                         seed = self.seed1,
                                                         box = set_box)
        else:
            #Randomly distrubte N particles with specified density phi_p (dictates simulation box size)
            #of particle type name with minimum separation distance min_dist with random seed in 2 dimensions
            system = hoomd.deprecated.init.create_random(N = self.partNum,
                                                         phi_p = self.phi,
                                                         name = 'A',
                                                         min_dist = 0.70,
                                                         seed = self.seed1,
                                                         dimensions = 2)
        # Add B-type particles
        system.particles.types.add('B')

        #Save current time step of system
        snapshot = system.take_snapshot()


        mid = int(self.partNumA)                    # starting index to assign B particles

        # Assign particles B type within the snapshot
        if self.partPercA == 0:                      # take care of all b case
            mid = 0
            for i in range(mid, self.partNum):
                system.particles[i].type = 'B'
        elif self.partPercA != 100:                  # mix of each or all A
            for i in range(mid, self.partNum):
                system.particles[i].type = 'B'

        # Assigning groups and lengths to particles
        all = hoomd.group.all()
        gA = hoomd.group.type(type = 'A', update=True)
        gB = hoomd.group.type(type = 'B', update=True)

        # Define potential between pairs
        nl = hoomd.md.nlist.cell()

        #Can change potential between particles here with hoomd.md.pair...
        lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)

        #Set parameters of pair force dependent on type of interaction
        lj.set_params(mode='shift')
        lj.pair_coeff.set('A', 'A', epsilon=self.epsA, sigma=1.0)
        lj.pair_coeff.set('A', 'B', epsilon=self.epsAB, sigma=1.0)
        lj.pair_coeff.set('B', 'B', epsilon=self.epsB, sigma=1.0)

        # General integration parameters

        #Equilibration number of time steps
        brownEquil = 1000

        # Each time step corresponds to time step size of dt
        hoomd.md.integrate.mode_standard(dt=self.dt)

        # Overdamped Langevin equations without activity at temperature kT.  Seed2 specifies translational diffusion.
        hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed2)

        #Run hoomd over brownEquil time steps
        hoomd.run(brownEquil)

        #set the activity of each type
        np.random.seed(self.seed3)                           # seed for random orientations
        angle = np.random.rand(self.partNum) * 2 * np.pi    # random particle orientation

        # Case 1: Mixture
        if self.partPercA != 0 and self.partPercA != 100:
            # First assign A-type active force vectors (w/ peA)
            activity_a = []
            for i in range(0,mid):
                x = (np.cos(angle[i])) * self.peA    # x active force vector
                y = (np.sin(angle[i])) * self.peA    # y active force vector
                z = 0                           # z active force vector
                tuple = (x, y, z)               # made into a tuple
                activity_a.append(tuple)        # add to activity A list

            # Now assign B-type active force vectors (w/ peB)
            activity_b = []
            for i in range(mid,self.partNum):
                x = (np.cos(angle[i])) * self.peB
                y = (np.sin(angle[i])) * self.peB
                z = 0
                tuple = (x, y, z)
                activity_b.append(tuple)
            # Set A-type activity in hoomd
            hoomd.md.force.active(group=gA,
                                  seed=self.seed4,
                                  f_lst=activity_a,
                                  rotation_diff=self.D_r,
                                  orientation_link=False,
                                  orientation_reverse_link=True)
            # Set B-type activity in hoomd
            hoomd.md.force.active(group=gB,
                                  seed=self.seed5,
                                  f_lst=activity_b,
                                  rotation_diff=self.D_r,
                                  orientation_link=False,
                                  orientation_reverse_link=True)
        else:
            # Case 2: All B system
            if self.partPercA == 0:
                activity_b = []
                for i in range(0,self.partNum):
                    x = (np.cos(angle[i])) * self.peB
                    y = (np.sin(angle[i])) * self.peB
                    z = 0
                    tuple = (x, y, z)
                    activity_b.append(tuple)
                hoomd.md.force.active(group=gB,
                                      seed=self.seed5,
                                      f_lst=activity_b,
                                      rotation_diff=self.D_r,
                                      orientation_link=False,
                                      orientation_reverse_link=True)
            # Case 3: All A system
            else:
                activity_a = []
                for i in range(0,self.partNum):
                    x = (np.cos(angle[i])) * self.peA
                    y = (np.sin(angle[i])) * self.peA
                    z = 0
                    tuple = (x, y, z)
                    activity_a.append(tuple)
                hoomd.md.force.active(group=gA,
                                      seed=self.seed4,
                                      f_lst=activity_a,
                                      rotation_diff=self.D_r,
                                      orientation_link=False,
                                      orientation_reverse_link=True)

        # base file name for output (specify variables that will be changed or that you care about)
        name = "random_init_pa" + str(self.peA) +\
        "_pb" + str(self.peB) +\
        "_xa" + str(self.partPercA) +\
        "_ep" + str(self.epsAB)+\
        "_phi"+str(self.intPhi)+\
        "_pNum" + str(self.partNum)+\
        "_aspect" + str(self.length) + '.' + str(self.width)

        # Actual gsd file name for output
        gsdName = name + ".gsd"

        # Remove .gsd files if they exist

        try:
            os.remove(gsdName)
        except OSError:
            pass

        #Specify how often and what to save

        # Options for what to save (dynamic)
        # 'attribute' important quantities: total particle number (N), types (types), which particles are each type (typeid),
        #  diameter of particle (diameter)
        #'property' important quantities: particle position (position), particle orientation in quaternions (orientation)
        #'momentum': I save this but it's unimportant.  It saves velocity, angular momentum, and image.  I looked into this and the v
        # velocity calculation is incorrect in my current version of hoomd.  THe newer version seems to have fixed this I believe. They mis-calculated
        # the quaternions to angles.
        #'topology' is another option for working with molecules.

        hoomd.dump.gsd(gsdName,
                       period=self.dumpFreq,
                       group=all,
                       overwrite=False,
                       phase=-1,
                       dynamic=['attribute', 'property', 'momentum'])

        #Number of time steps to run simulation for.
        hoomd.run(self.totTsteps)
    def homogeneous_cluster(self):

        import random

        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)


        # Compute lattice spacing based on each activity
        latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG
        phiG = self.theory_functs.compPhiG(peNet, latNet)

        phi_theory = self.theory_functs.latToPhi(latNet)

        Nl = int(round(self.partNum * ((phi_theory * (phiG - self.phi)) / (self.phi * (phiG - phi_theory)))))

        # Now you need to convert this to a cluster radius
        phiCP = np.pi / (2. * np.sqrt(3))

        # The area is the sum of the particle areas (normalized by close packing density of spheres)
        Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)

        curPLJ = self.theory_functs.ljPress(latNet, 500, self.eps)

        # The area for seed
        Al_real=Al

        # The cluster radius is the square root of liquid area divided by pi
        Rl = np.sqrt(Al_real / np.pi)

        alpha_max = 0.5
        I_arr = 3.0
        int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/500) * (latNet **2) * I_arr

        if int_width >= Rl:
            int_width = Rl-1.0
        # Remember!!! This is a prediction of the EQUILIBRIUM size, reduce this to seed a cluster
        # MAKE SURE that the composition of the seed has the same composition of the system
        # e.g. for xF = 0.3 the initial seed should be 30% fast 70% slow


        #print(int_width)
        #stop

        # Use latNet to space your particles
        def computeDistance(x, y):
            return np.sqrt((x**2) + (y**2))

        def interDist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def orientToOrigin(x, y, act):
            "Using similar triangles to find sides"
            x *= -1.
            y *= -1.
            hypRatio = act / np.sqrt(x**2 + y**2)
            xAct = hypRatio * x
            yAct = hypRatio * y
            return xAct, yAct

        # List of activities
        peList = [ self.peA ]
        # List of ring radii
        rList = [ 0., Rl ]
        # Depth of alignment
        #rAlign = 3.

        rAlign = int_width#*(2/3)#3.#int_width
        # List to store particle positions and types
        pos = []
        typ = []
        rOrient = []
        # z-value for simulation initialization
        z = 0.5

        for i in range(0, len(peList)):
            rMin = rList[0]             # starting distance for particle placement
            rMax = rList[1]         # maximum distance for particle placement
            ver = np.sin(60*np.pi/180)*latNet   # vertical shift between lattice rows
            hor = latNet / 2.0             # horizontal shift between lattice rows

            x = 0.
            y = 0.
            shift = 0.
            while y <= rMax:
                r = computeDistance(x, y)
                # Check if x-position is too large
                if r > (rMax + (latNet/2.)):
                    y += ver
                    shift += 1
                    if shift % 2:
                        x = hor
                    else:
                        x = 0.
                    continue

                # Whether or not particle is oriented
                if r > (rMax - rAlign):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)

                # If the loop makes it this far, append
                pos.append((x, y, z))
                typ.append(i)
                if x != 0. and y != 0.:
                    # Mirror positions, alignment and type
                    pos.append((-x, y, z))
                    pos.append((-x, -y, z))
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    typ.append(i)
                    typ.append(i)
                    typ.append(i)
                # y must be zero
                elif x != 0.:
                    pos.append((-x, y, z))
                    rOrient.append(rOrient[-1])
                    typ.append(i)
                # x must be zero
                elif y!= 0.:
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    typ.append(i)

                # Increment counter
                x += latNet

        # Update number of particles in gas and dense phase

        NLiq = len(pos)
        NGas = self.partNum - NLiq
        typ_A=0
        typ_B=0

        for i in range(0,len(typ)):
            rand_val=random.random()
            if rand_val<=self.partFracA:
                typ[i]=0
                typ_A+=1
            else:
                typ[i]=1
                typ_B+=1

        partNumB_gas=self.partNumB-typ_B
        partNumA_gas=self.partNumA-typ_A

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)
        lbox = np.sqrt(abox)
        hbox = lbox / 2.

        import utility

        utility_functs = utility.utility(lbox)

        tooClose = 0.9

        # Compute mesh

        nBins = (utility_functs.getNBins(lbox, self.r_cut))
        sizeBin = utility_functs.roundUp((lbox / nBins), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        while count < NGas:
            place = 1
            # Generate random position
            gasx = (np.random.rand() - 0.5) * lbox
            gasy = (np.random.rand() - 0.5) * lbox
            r = computeDistance(gasx, gasy)

            # Is this an HCP bin?
            if r <= (rList[-1] + (latNet/2.) + (tooClose / 2.)):
                continue

            # Are any gas particles too close?
            tmpx = gasx + hbox
            tmpy = gasy + hbox
            indx = int(tmpx / sizeBin)
            indy = int(tmpy / sizeBin)
            # Get index of surrounding bins
            lbin = indx - 1  # index of left bins
            rbin = indx + 1  # index of right bins
            bbin = indy - 1  # index of bottom bins
            tbin = indy + 1  # index of top bins
            if rbin == nBins:
                rbin -= nBins  # adjust if wrapped
            if tbin == nBins:
                tbin -= nBins  # adjust if wrapped
            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lbox
                    if h == 2 and hlist[h] == 0:
                        wrapX += lbox
                    if v == 0 and vlist[v] == -1:
                        wrapY -= lbox
                    if v == 2 and vlist[v] == 0:
                        wrapY += lbox
                    # Compute distance between particles
                    if binParts[hlist[h]][vlist[v]]:
                        for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                            # Get index of nearby particle
                            ref = binParts[hlist[h]][vlist[v]][b]
                            r = interDist(gasx, gasy,
                                          gaspos[ref][0] + wrapX,
                                          gaspos[ref][1] + wrapY)
                            # Round to 4 decimal places
                            r = round(r, 4)
                            # If too close, generate new position
                            if r <= tooClose:
                                place = 0
                                break
                    if place == 0:
                        break
                if place == 0:
                    break

            # Is it safe to append the particle?
            if place == 1:
                binParts[indx][indy].append(count)
                gaspos.append((gasx, gasy, z))
                rOrient.append(0)       # not oriented
                typ.append(0)           # final particle type, same as outer ring
                count += 1              # increment count


        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        pos = pos + gaspos

        NGas_shift=NGas
        for i in range(0,NGas):
            j=NLiq+i
            rand_val=random.random()
            xA_gas=partNumA_gas/NGas_shift
            if rand_val<=xA_gas:
                typ[j]=0
                typ_A+=1
                partNumA_gas-=1
                NGas_shift-=1
            else:
                typ[j]=1
                typ_B+=1
                partNumB_gas-=1
                NGas_shift-=1
        typ_arr=np.array(typ)
        id0=np.where(typ_arr==0)
        id1=np.where(typ_arr==1)

        x, y, z = zip(*pos)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos)

        # Get the number of types
        uniqueTyp = []
        for i in typ:
            if i not in uniqueTyp:
                uniqueTyp.append(i)
        # Get the number of each type
        particles = [ 0 for x in range(0, len(uniqueTyp)) ]
        for i in range(0, len(uniqueTyp)):
            for j in typ:
                if uniqueTyp[i] == j:
                    particles[i] += 1
        # Convert types to letter values
        unique_char_types = []
        for i in uniqueTyp:
            unique_char_types.append( chr(ord('@') + i+1) )
        char_types = []
        for i in typ:
            char_types.append( chr(ord('@') + i+1) )

        # Get a list of activities for all particles
        pe = []
        peList = [self.peA, self.peB]
        for i in typ:
            pe.append(peList[i])

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Now we make the system in hoomd
        hoomd.context.initialize()
        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = self.partNum,
                                        box = hoomd.data.boxdim(Lx=lbox,
                                                                Ly=lbox,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos[:]
        snap.particles.typeid[:] = typ[:]
        snap.particles.types[:] = char_types[:]

        # Initialize the system
        system = hoomd.init.read_snapshot(snap)
        all = hoomd.group.all()
        groups = []
        for i in unique_char_types:
            groups.append(hoomd.group.type(type=i))

        # Set particle potentials
        nl = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
        lj.set_params(mode='shift')
        for i in range(0, len(unique_char_types)):
            for j in range(i, len(unique_char_types)):
                lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        hoomd.run(brownEquil)

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, self.partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                x, y = orientToOrigin(pos[i][0], pos[i][1], pe[i])
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        hoomd.md.force.active(group=all,
                              seed=self.seed3,
                              f_lst=activity,
                              rotation_diff=self.D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)

        # Name the file from parameters
        #out = "cluster_pe"
        #for i in peList:
        #    out += str(int(i))
        #    out += "_"
        #out += "r"
        #for i in range(1, len(rList)):
        #    out += str(int(rList[i]))
        #    out += "_"
        #out += "rAlign_" + str(rAlign) + ".gsd"
        out = "homogeneous_cluster_pa" + str(int(self.peA))
        out += "_pb" + str(int(self.peB))
        out += "_phi" + str(self.intPhi)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(self.partFracA)
        out += "_pNum" + str(self.partNum)
        out += "_dtau" + "{:.1e}".format(self.dt)
        out += ".gsd"

        # Write dump

        hoomd.dump.gsd(out,
                       period=self.dumpFreq,
                       group=all,
                       overwrite=True,
                       phase=-1,
                       dynamic=['attribute', 'property', 'momentum'])

        # Run
        print('test')
        print(self.totTsteps)
        print(type(self.totTsteps))
        hoomd.run(self.totTsteps)

    def slow_bulk_cluster(self):

        import random

        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity
        latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG
        phiG = self.theory_functs.compPhiG(peNet, latNet)

        phi_theory = self.theory_functs.latToPhi(latNet)

        Nl = int(round(self.partNum * ((phi_theory * (phiG - self.phi)) / (self.phi * (phiG - phi_theory)))))

        # Now you need to convert this to a cluster radius
        phiCP = np.pi / (2. * np.sqrt(3))

        # The area is the sum of the particle areas (normalized by close packing density of spheres)
        Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
        As = (self.partNumA * np.pi * (latNet)**2) / (4*phiCP)
        Af = (self.partNumB * np.pi * (latNet)**2) / (4*phiCP)

        curPLJ = self.theory_functs.ljPress(latNet, 500, self.eps)

        # The area for seed
        Al_real=Al

        # The cluster radius is the square root of liquid area divided by pi
        Rl = np.sqrt(Al_real / np.pi)
        Rs = np.sqrt(As / np.pi)
        Rf = np.sqrt(Af / np.pi)

        alpha_max = 0.5
        I_arr = 3.0
        int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/500) * (latNet **2) * I_arr

        if int_width >= Rl:
            int_width = Rl-1.0
        # Remember!!! This is a prediction of the EQUILIBRIUM size, reduce this to seed a cluster
        # MAKE SURE that the composition of the seed has the same composition of the system
        # e.g. for xF = 0.3 the initial seed should be 30% fast 70% slow


        #print(int_width)
        #stop

        # Use latNet to space your particles
        def computeDistance(x, y):
            return np.sqrt((x**2) + (y**2))

        def interDist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def orientToOrigin(x, y, act):
            "Using similar triangles to find sides"
            x *= -1.
            y *= -1.
            hypRatio = act / np.sqrt(x**2 + y**2)
            xAct = hypRatio * x
            yAct = hypRatio * y
            return xAct, yAct

        # List of activities
        peList = [ self.peA, self.peB ]
        # List of ring radii
        rList = [ 0, Rf, Rl ]
        # Depth of alignment
        #rAlign = 3.

        rAlign = int_width#*(2/3)#3.#int_width
        # List to store particle positions and types
        pos = []
        typ = []
        rOrient = []
        # z-value for simulation initialization
        z = 0.5

        for i in range(0,len(peList)):
            rMin = rList[i]             # starting distance for particle placement
            rMax = rList[i + 1]         # maximum distance for particle placement
            ver = np.sin(60*np.pi/180)*latNet#np.sqrt(0.75) * latNet   # vertical shift between lattice rows
            hor = latNet / 2.0             # horizontal shift between lattice rows
            x = 0
            y = 0
            shift = 0
            while y < rMax:
                r = computeDistance(x, y)
                # Check if x-position is large enough
                if r <rMin: # <(rMin + (latNet / 2.)):
                    x += latNet
                    continue

                # Check if x-position is too large
                if r >rMax:#>= (rMax - (latNet/2.)):
                    y += ver
                    shift += 1
                    if shift % 2:
                        x = hor
                    else:
                        x = 0
                    continue

                # Whether or not particle is oriented
                if r > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)

                # If the loop makes it this far, append
                pos.append((x, y, z))
                typ.append(i)
                if x != 0 and y != 0:
                    # Mirror positions, alignment and type
                    pos.append((-x, y, z))
                    pos.append((-x, -y, z))
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    typ.append(i)
                    typ.append(i)
                    typ.append(i)
                # y must be zero
                elif x != 0:
                    pos.append((-x, y, z))
                    rOrient.append(rOrient[-1])
                    typ.append(i)
                # x must be zero
                elif y!= 0:
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    typ.append(i)

                # Increment counter
                x += latNet

        # Update number of particles in gas and dense phase

        NLiq = len(pos)
        NGas = self.partNum - NLiq

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)
        lbox = np.sqrt(abox)
        hbox = lbox / 2.

        import utility

        utility_functs = utility.utility(lbox)

        tooClose = 0.9

        # Compute mesh

        nBins = (utility_functs.getNBins(lbox, self.r_cut))
        sizeBin = utility_functs.roundUp((lbox / nBins), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        while count < NGas:
            place = 1
            # Generate random position
            gasx = (np.random.rand() - 0.5) * lbox
            gasy = (np.random.rand() - 0.5) * lbox
            r = computeDistance(gasx, gasy)

            # Is this an HCP bin?
            if r <= (rList[-1] + (tooClose / 2.)):
                continue

            # Are any gas particles too close?
            tmpx = gasx + hbox
            tmpy = gasy + hbox
            indx = int(tmpx / sizeBin)
            indy = int(tmpy / sizeBin)
            # Get index of surrounding bins
            lbin = indx - 1  # index of left bins
            rbin = indx + 1  # index of right bins
            bbin = indy - 1  # index of bottom bins
            tbin = indy + 1  # index of top bins
            if rbin == nBins:
                rbin -= nBins  # adjust if wrapped
            if tbin == nBins:
                tbin -= nBins  # adjust if wrapped
            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lbox
                    if h == 2 and hlist[h] == 0:
                        wrapX += lbox
                    if v == 0 and vlist[v] == -1:
                        wrapY -= lbox
                    if v == 2 and vlist[v] == 0:
                        wrapY += lbox
                    # Compute distance between particles
                    if binParts[hlist[h]][vlist[v]]:
                        for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                            # Get index of nearby particle
                            ref = binParts[hlist[h]][vlist[v]][b]
                            r = interDist(gasx, gasy,
                                          gaspos[ref][0] + wrapX,
                                          gaspos[ref][1] + wrapY)
                            # Round to 4 decimal places
                            r = round(r, 4)
                            # If too close, generate new position
                            if r <= tooClose:
                                place = 0
                                break
                    if place == 0:
                        break
                if place == 0:
                    break

            # Is it safe to append the particle?
            if place == 1:
                binParts[indx][indy].append(count)
                gaspos.append((gasx, gasy, z))
                rOrient.append(0)       # not oriented
                typ.append(1)           # final particle type, same as outer ring
                count += 1              # increment count


        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        pos = pos + gaspos
        x, y, z = zip(*pos)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos)

        # Get the number of types
        uniqueTyp = []
        for i in typ:
            if i not in uniqueTyp:
                uniqueTyp.append(i)
        # Get the number of each type
        particles = [ 0 for x in range(0, len(uniqueTyp)) ]
        for i in range(0, len(uniqueTyp)):
            for j in typ:
                if uniqueTyp[i] == j:
                    particles[i] += 1
        # Convert types to letter values
        unique_char_types = []
        for i in uniqueTyp:
            unique_char_types.append( chr(ord('@') + i+1) )
        char_types = []
        for i in typ:
            char_types.append( chr(ord('@') + i+1) )

        # Get a list of activities for all particles
        pe = []
        for i in typ:
            pe.append(peList[i])

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Now we make the system in hoomd
        hoomd.context.initialize()
        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = self.partNum,
                                        box = hoomd.data.boxdim(Lx=lbox,
                                                                Ly=lbox,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos[:]
        snap.particles.typeid[:] = typ[:]
        snap.particles.types[:] = char_types[:]

        # Initialize the system
        system = hoomd.init.read_snapshot(snap)
        all = hoomd.group.all()
        groups = []
        for i in unique_char_types:
            groups.append(hoomd.group.type(type=i))

        # Set particle potentials
        nl = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
        lj.set_params(mode='shift')
        for i in range(0, len(unique_char_types)):
            for j in range(i, len(unique_char_types)):
                lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        hoomd.run(brownEquil)

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, self.partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                x, y = orientToOrigin(pos[i][0], pos[i][1], pe[i])
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        hoomd.md.force.active(group=all,
                              seed=self.seed3,
                              f_lst=activity,
                              rotation_diff=self.D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)

        # Name the file from parameters
        #out = "cluster_pe"
        #for i in peList:
        #    out += str(int(i))
        #    out += "_"
        #out += "r"
        #for i in range(1, len(rList)):
        #    out += str(int(rList[i]))
        #    out += "_"
        #out += "rAlign_" + str(rAlign) + ".gsd"
        out = "slow_bulk_cluster_pa" + str(int(self.peA))
        out += "_pb" + str(int(self.peB))
        out += "_phi" + str(self.intPhi)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(self.partFracA)
        out += "_pNum" + str(self.partNum)
        out += "_dtau" + "{:.1e}".format(self.dt)
        out += ".gsd"

        # Write dump

        hoomd.dump.gsd(out,
                       period=self.dumpFreq,
                       group=all,
                       overwrite=True,
                       phase=-1,
                       dynamic=['attribute', 'property', 'momentum'])

        # Run
        print('test')
        print(self.totTsteps)
        print(type(self.totTsteps))
        hoomd.run(self.totTsteps)

    def fast_bulk_cluster(self):

        import random

        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity
        latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG
        phiG = self.theory_functs.compPhiG(peNet, latNet)

        phi_theory = self.theory_functs.latToPhi(latNet)

        Nl = int(round(self.partNum * ((phi_theory * (phiG - self.phi)) / (self.phi * (phiG - phi_theory)))))

        # Now you need to convert this to a cluster radius
        phiCP = np.pi / (2. * np.sqrt(3))

        # The area is the sum of the particle areas (normalized by close packing density of spheres)
        Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
        As = (self.partNumA * np.pi * (latNet)**2) / (4*phiCP)
        Af = (self.partNumB * np.pi * (latNet)**2) / (4*phiCP)

        curPLJ = self.theory_functs.ljPress(latNet, 500, self.eps)

        # The area for seed
        Al_real=Al

        # The cluster radius is the square root of liquid area divided by pi
        Rl = np.sqrt(Al_real / np.pi)
        Rs = np.sqrt(As / np.pi)
        Rf = np.sqrt(Af / np.pi)

        alpha_max = 0.5
        I_arr = 3.0
        int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/500) * (latNet **2) * I_arr

        if int_width >= Rl:
            int_width = Rl-1.0
        # Remember!!! This is a prediction of the EQUILIBRIUM size, reduce this to seed a cluster
        # MAKE SURE that the composition of the seed has the same composition of the system
        # e.g. for xF = 0.3 the initial seed should be 30% fast 70% slow


        #print(int_width)
        #stop

        # Use latNet to space your particles
        def computeDistance(x, y):
            return np.sqrt((x**2) + (y**2))

        def interDist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def orientToOrigin(x, y, act):
            "Using similar triangles to find sides"
            x *= -1.
            y *= -1.
            hypRatio = act / np.sqrt(x**2 + y**2)
            xAct = hypRatio * x
            yAct = hypRatio * y
            return xAct, yAct

        # List of activities
        peList = [ self.peA, self.peB ]
        # List of ring radii
        rList = [ 0, Rf, Rl ]
        # Depth of alignment
        #rAlign = 3.

        rAlign = int_width#*(2/3)#3.#int_width
        # List to store particle positions and types
        pos = []
        typ = []
        rOrient = []
        # z-value for simulation initialization
        z = 0.5

        for i in range(0,len(peList)):
            rMin = rList[i]             # starting distance for particle placement
            rMax = rList[i + 1]         # maximum distance for particle placement
            ver = np.sin(60*np.pi/180)*latNet#np.sqrt(0.75) * latNet   # vertical shift between lattice rows
            hor = latNet / 2.0             # horizontal shift between lattice rows
            x = 0
            y = 0
            shift = 0
            while y < rMax:
                r = computeDistance(x, y)
                # Check if x-position is large enough
                if r <rMin: # <(rMin + (latNet / 2.)):
                    x += latNet
                    continue

                # Check if x-position is too large
                if r >rMax:#>= (rMax - (latNet/2.)):
                    y += ver
                    shift += 1
                    if shift % 2:
                        x = hor
                    else:
                        x = 0
                    continue

                # Whether or not particle is oriented
                if r > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)

                # If the loop makes it this far, append
                pos.append((x, y, z))
                if i==0:
                    typ.append(1)
                else:
                    typ.append(0)
                if x != 0 and y != 0:
                    # Mirror positions, alignment and type
                    pos.append((-x, y, z))
                    pos.append((-x, -y, z))
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                        typ.append(1)
                        typ.append(1)
                    else:
                        typ.append(0)
                        typ.append(0)
                        typ.append(0)
                # y must be zero
                elif x != 0:
                    pos.append((-x, y, z))
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                    else:
                        typ.append(0)
                    #typ.append(i)
                # x must be zero
                elif y!= 0:
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                    else:
                        typ.append(0)
                    #typ.append(i)

                # Increment counter
                x += latNet

        # Update number of particles in gas and dense phase

        NLiq = len(pos)
        NGas = self.partNum - NLiq

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)
        lbox = np.sqrt(abox)
        hbox = lbox / 2.

        import utility

        utility_functs = utility.utility(lbox)

        tooClose = 0.9

        # Compute mesh

        nBins = (utility_functs.getNBins(lbox, self.r_cut))
        sizeBin = utility_functs.roundUp((lbox / nBins), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        while count < NGas:
            place = 1
            # Generate random position
            gasx = (np.random.rand() - 0.5) * lbox
            gasy = (np.random.rand() - 0.5) * lbox
            r = computeDistance(gasx, gasy)

            # Is this an HCP bin?
            if r <= (rList[-1] + (tooClose / 2.)):
                continue

            # Are any gas particles too close?
            tmpx = gasx + hbox
            tmpy = gasy + hbox
            indx = int(tmpx / sizeBin)
            indy = int(tmpy / sizeBin)
            # Get index of surrounding bins
            lbin = indx - 1  # index of left bins
            rbin = indx + 1  # index of right bins
            bbin = indy - 1  # index of bottom bins
            tbin = indy + 1  # index of top bins
            if rbin == nBins:
                rbin -= nBins  # adjust if wrapped
            if tbin == nBins:
                tbin -= nBins  # adjust if wrapped
            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lbox
                    if h == 2 and hlist[h] == 0:
                        wrapX += lbox
                    if v == 0 and vlist[v] == -1:
                        wrapY -= lbox
                    if v == 2 and vlist[v] == 0:
                        wrapY += lbox
                    # Compute distance between particles
                    if binParts[hlist[h]][vlist[v]]:
                        for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                            # Get index of nearby particle
                            ref = binParts[hlist[h]][vlist[v]][b]
                            r = interDist(gasx, gasy,
                                          gaspos[ref][0] + wrapX,
                                          gaspos[ref][1] + wrapY)
                            # Round to 4 decimal places
                            r = round(r, 4)
                            # If too close, generate new position
                            if r <= tooClose:
                                place = 0
                                break
                    if place == 0:
                        break
                if place == 0:
                    break

            # Is it safe to append the particle?
            if place == 1:
                binParts[indx][indy].append(count)
                gaspos.append((gasx, gasy, z))
                rOrient.append(0)       # not oriented
                typ.append(0)           # final particle type, same as outer ring
                count += 1              # increment count


        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        pos = pos + gaspos
        x, y, z = zip(*pos)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos)

        # Get the number of types
        uniqueTyp = []
        for i in typ:
            if i not in uniqueTyp:
                uniqueTyp.append(i)
        # Get the number of each type
        particles = [ 0 for x in range(0, len(uniqueTyp)) ]
        for i in range(0, len(uniqueTyp)):
            for j in typ:
                if uniqueTyp[i] == j:
                    particles[i] += 1
        # Convert types to letter values
        unique_char_types = []
        for i in uniqueTyp:
            unique_char_types.append( chr(ord('@') + i+1) )
        char_types = []
        for i in typ:
            char_types.append( chr(ord('@') + i+1) )

        # Get a list of activities for all particles
        pe = []
        for i in typ:
            pe.append(peList[i])

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Now we make the system in hoomd
        hoomd.context.initialize()
        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = self.partNum,
                                        box = hoomd.data.boxdim(Lx=lbox,
                                                                Ly=lbox,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos[:]
        snap.particles.typeid[:] = typ[:]
        snap.particles.types[:] = char_types[:]

        # Initialize the system
        system = hoomd.init.read_snapshot(snap)
        all = hoomd.group.all()
        groups = []
        for i in unique_char_types:
            groups.append(hoomd.group.type(type=i))

        # Set particle potentials
        nl = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
        lj.set_params(mode='shift')
        for i in range(0, len(unique_char_types)):
            for j in range(i, len(unique_char_types)):
                lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        hoomd.run(brownEquil)

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, self.partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                x, y = orientToOrigin(pos[i][0], pos[i][1], pe[i])
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        hoomd.md.force.active(group=all,
                              seed=self.seed3,
                              f_lst=activity,
                              rotation_diff=self.D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)

        # Name the file from parameters
        #out = "cluster_pe"
        #for i in peList:
        #    out += str(int(i))
        #    out += "_"
        #out += "r"
        #for i in range(1, len(rList)):
        #    out += str(int(rList[i]))
        #    out += "_"
        #out += "rAlign_" + str(rAlign) + ".gsd"
        out = "fast_bulk_cluster_pa" + str(int(self.peA))
        out += "_pb" + str(int(self.peB))
        out += "_phi" + str(self.intPhi)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(self.partFracA)
        out += "_pNum" + str(self.partNum)
        out += "_dtau" + "{:.1e}".format(self.dt)
        out += ".gsd"

        # Write dump

        hoomd.dump.gsd(out,
                       period=self.dumpFreq,
                       group=all,
                       overwrite=True,
                       phase=-1,
                       dynamic=['attribute', 'property', 'momentum'])

        # Run
        print('test')
        print(self.totTsteps)
        print(type(self.totTsteps))
        hoomd.run(self.totTsteps)

    def half_cluster(self):
        import random

        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity
        latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG
        phiG = self.theory_functs.compPhiG(peNet, latNet)

        phi_theory = self.theory_functs.latToPhi(latNet)

        Nl = int(round(self.partNum * ((phi_theory * (phiG - self.phi)) / (self.phi * (phiG - phi_theory)))))

        # Now you need to convert this to a cluster radius
        phiCP = np.pi / (2. * np.sqrt(3))

        # The area is the sum of the particle areas (normalized by close packing density of spheres)
        Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
        As = (self.partNumA * np.pi * (latNet)**2) / (4*phiCP)
        Af = (self.partNumB * np.pi * (latNet)**2) / (4*phiCP)

        curPLJ = self.theory_functs.ljPress(latNet, 500, self.eps)

        # The area for seed
        Al_real=Al

        # The cluster radius is the square root of liquid area divided by pi
        Rl = np.sqrt(Al_real / np.pi)
        Rs = np.sqrt(As / np.pi)
        Rf = np.sqrt(Af / np.pi)

        alpha_max = 0.5
        I_arr = 3.0
        int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/500) * (latNet **2) * I_arr

        if int_width >= Rl:
            int_width = Rl-1.0
        # Remember!!! This is a prediction of the EQUILIBRIUM size, reduce this to seed a cluster
        # MAKE SURE that the composition of the seed has the same composition of the system
        # e.g. for xF = 0.3 the initial seed should be 30% fast 70% slow


        #print(int_width)
        #stop

        # Use latNet to space your particles
        def computeDistance(x, y):
            return np.sqrt((x**2) + (y**2))

        def interDist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def orientToOrigin(x, y, act):
            "Using similar triangles to find sides"
            x *= -1.
            y *= -1.
            hypRatio = act / np.sqrt(x**2 + y**2)
            xAct = hypRatio * x
            yAct = hypRatio * y
            return xAct, yAct

        # List of activities
        peList = [ self.peA, self.peB ]
        # List of ring radii
        rList = [ 0, Rf, Rl ]
        # Depth of alignment
        #rAlign = 3.

        rAlign = int_width#*(2/3)#3.#int_width
        # List to store particle positions and types
        pos = []
        typ = []
        rOrient = []
        # z-value for simulation initialization
        z = 0.5

        for i in range(0,len(peList)):
            rMin = rList[i]             # starting distance for particle placement
            rMax = rList[i + 1]         # maximum distance for particle placement
            ver = np.sin(60*np.pi/180)*latNet#np.sqrt(0.75) * latNet   # vertical shift between lattice rows
            hor = latNet / 2.0             # horizontal shift between lattice rows
            x = 0
            y = 0
            shift = 0
            while y < rMax:
                r = computeDistance(x, y)
                # Check if x-position is large enough
                if r <rMin: # <(rMin + (latNet / 2.)):
                    x += latNet
                    continue

                # Check if x-position is too large
                if r >rMax:#>= (rMax - (latNet/2.)):
                    y += ver
                    shift += 1
                    if shift % 2:
                        x = hor
                    else:
                        x = 0
                    continue

                # Whether or not particle is oriented
                if r > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)

                # If the loop makes it this far, append
                pos.append((x, y, z))
                if i==0:
                    typ.append(1)
                else:
                    typ.append(0)
                if x != 0 and y != 0:
                    # Mirror positions, alignment and type
                    pos.append((-x, y, z))
                    pos.append((-x, -y, z))
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                        typ.append(1)
                        typ.append(1)
                    else:
                        typ.append(0)
                        typ.append(0)
                        typ.append(0)
                # y must be zero
                elif x != 0:
                    pos.append((-x, y, z))
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                    else:
                        typ.append(0)
                    #typ.append(i)
                # x must be zero
                elif y!= 0:
                    pos.append((x, -y, z))
                    rOrient.append(rOrient[-1])
                    if i==0:
                        typ.append(1)
                    else:
                        typ.append(0)
                    #typ.append(i)

                # Increment counter
                x += latNet

        # Update number of particles in gas and dense phase

        NLiq = len(pos)
        NGas = self.partNum - NLiq

        typ_A=0
        typ_B=0

        for i in range(0,len(typ)):
            if pos[i][0]<=0:
                typ[i]=0
                typ_A+=1
            else:
                typ[i]=1
                typ_B+=1
        gas_B=self.partNumB-typ_B
        gas_A=self.partNumA-typ_A

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)
        lbox = np.sqrt(abox)
        hbox = lbox / 2.

        import utility

        utility_functs = utility.utility(lbox)

        tooClose = 0.9

        # Compute mesh

        nBins = (utility_functs.getNBins(lbox, self.r_cut))
        sizeBin = utility_functs.roundUp((lbox / nBins), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        while count < NGas:
            place = 1
            # Generate random position
            gasx = (np.random.rand() - 0.5) * lbox
            gasy = (np.random.rand() - 0.5) * lbox
            r = computeDistance(gasx, gasy)

            # Is this an HCP bin?
            if r <= (rList[-1] + (tooClose / 2.)):
                continue

            # Are any gas particles too close?
            tmpx = gasx + hbox
            tmpy = gasy + hbox
            indx = int(tmpx / sizeBin)
            indy = int(tmpy / sizeBin)
            # Get index of surrounding bins
            lbin = indx - 1  # index of left bins
            rbin = indx + 1  # index of right bins
            bbin = indy - 1  # index of bottom bins
            tbin = indy + 1  # index of top bins
            if rbin == nBins:
                rbin -= nBins  # adjust if wrapped
            if tbin == nBins:
                tbin -= nBins  # adjust if wrapped
            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lbox
                    if h == 2 and hlist[h] == 0:
                        wrapX += lbox
                    if v == 0 and vlist[v] == -1:
                        wrapY -= lbox
                    if v == 2 and vlist[v] == 0:
                        wrapY += lbox
                    # Compute distance between particles
                    if binParts[hlist[h]][vlist[v]]:
                        for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                            # Get index of nearby particle
                            ref = binParts[hlist[h]][vlist[v]][b]
                            r = interDist(gasx, gasy,
                                          gaspos[ref][0] + wrapX,
                                          gaspos[ref][1] + wrapY)
                            # Round to 4 decimal places
                            r = round(r, 4)
                            # If too close, generate new position
                            if r <= tooClose:
                                place = 0
                                break
                    if place == 0:
                        break
                if place == 0:
                    break

            # Is it safe to append the particle?
            if place == 1:
                binParts[indx][indy].append(count)
                gaspos.append((gasx, gasy, z))
                rOrient.append(0)       # not oriented
                typ.append(0)           # final particle type, same as outer ring
                count += 1              # increment count


        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        pos = pos + gaspos

        NGas_shift=NGas

        for i in range(0,NGas):
            j=NLiq+i
            rand_val=random.random()
            xB_gas=gas_B/NGas_shift
            if rand_val<=xB_gas:
                typ[j]=1
                typ_B+=1
                gas_B-=1
                NGas_shift-=1
            else:
                typ[j]=0
                typ_A+=1
                gas_A-=1
                NGas_shift-=1
        typ_arr=np.array(typ)
        id0=np.where(typ_arr==0)
        id1=np.where(typ_arr==1)

        x, y, z = zip(*pos)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos)

        # Get the number of types
        uniqueTyp = []
        for i in typ:
            if i not in uniqueTyp:
                uniqueTyp.append(i)
        # Get the number of each type
        particles = [ 0 for x in range(0, len(uniqueTyp)) ]
        for i in range(0, len(uniqueTyp)):
            for j in typ:
                if uniqueTyp[i] == j:
                    particles[i] += 1
        # Convert types to letter values
        unique_char_types = []
        for i in uniqueTyp:
            unique_char_types.append( chr(ord('@') + i+1) )
        char_types = []
        for i in typ:
            char_types.append( chr(ord('@') + i+1) )

        # Get a list of activities for all particles
        pe = []
        for i in typ:
            pe.append(peList[i])

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Now we make the system in hoomd
        hoomd.context.initialize()
        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = self.partNum,
                                        box = hoomd.data.boxdim(Lx=lbox,
                                                                Ly=lbox,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos[:]
        snap.particles.typeid[:] = typ[:]
        snap.particles.types[:] = char_types[:]

        # Initialize the system
        system = hoomd.init.read_snapshot(snap)
        all = hoomd.group.all()
        groups = []
        for i in unique_char_types:
            groups.append(hoomd.group.type(type=i))

        # Set particle potentials
        nl = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
        lj.set_params(mode='shift')
        for i in range(0, len(unique_char_types)):
            for j in range(i, len(unique_char_types)):
                lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        hoomd.run(brownEquil)

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, self.partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                x, y = orientToOrigin(pos[i][0], pos[i][1], pe[i])
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        hoomd.md.force.active(group=all,
                              seed=self.seed3,
                              f_lst=activity,
                              rotation_diff=self.D_r,
                              orientation_link=False,
                              orientation_reverse_link=True)

        # Name the file from parameters
        #out = "cluster_pe"
        #for i in peList:
        #    out += str(int(i))
        #    out += "_"
        #out += "r"
        #for i in range(1, len(rList)):
        #    out += str(int(rList[i]))
        #    out += "_"
        #out += "rAlign_" + str(rAlign) + ".gsd"
        out = "half_and_half_cluster_pa" + str(int(self.peA))
        out += "_pb" + str(int(self.peB))
        out += "_phi" + str(self.intPhi)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(self.partFracA)
        out += "_pNum" + str(self.partNum)
        out += "_dtau" + "{:.1e}".format(self.dt)
        out += ".gsd"

        # Write dump

        hoomd.dump.gsd(out,
                       period=self.dumpFreq,
                       group=all,
                       overwrite=True,
                       phase=-1,
                       dynamic=['attribute', 'property', 'momentum'])

        # Run
        print('test')
        print(self.totTsteps)
        print(type(self.totTsteps))
        hoomd.run(self.totTsteps)


    def fast_penetrate_slow_membrane(self):

        import random

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        if self.length != self.width:
            import matplotlib.pyplot as plt
            area_ratio = self.length * self.width

            box_area = self.partNum/self.phi
            box_length = (box_area/area_ratio)**0.5
            lx = self.length*box_length
            hx = lx/2
            ly = self.width * box_length
            hy = ly/2

            #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd



            peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

            # Compute lattice spacing based on each activity
            latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
            #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
            # Compute gas phase density, phiG
            phiG = self.theory_functs.compPhiG(peNet, latNet)

            phi_theory = self.theory_functs.latToPhi(latNet)

            Nl = int(round(self.partNum * ((phi_theory * (phiG - self.phi)) / (self.phi * (phiG - phi_theory)))))

            # Now you need to convert this to a cluster radius
            phiCP = np.pi / (2. * np.sqrt(3))

            # The area is the sum of the particle areas (normalized by close packing density of spheres)
            Al = (Nl * np.pi * (latNet)**2) / (4*phiCP)
            As = (self.partNumA * np.pi * (latNet)**2) / (4*phiCP)
            Af = (self.partNumB * np.pi * (latNet)**2) / (4*phiCP)


            # The area for seed
            Al_real=Al

            if lx < ly:
                thickness = As / (lx)
            else:
                thickness = As / (ly)

            Rs = thickness / 2

            curPLJ = self.theory_functs.ljPress(latNet, 500, self.eps)
            alpha_max = 0.5
            I_arr = 3.0
            int_width = (np.sqrt(3)/(2*alpha_max)) * (curPLJ/500) * (latNet **2) * I_arr

            if int_width >= Rs:
                int_width = Rs-1.0


            # Use latNet to space your particles
            def computeDistance(x, y):
                return np.sqrt((x**2) + (y**2))

            def interDist(x1, y1, x2, y2):
                return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            def orientToOrigin(x, y, act):
                "Using similar triangles to find sides"
                x *= -1.
                y *= -1.
                hypRatio = act / np.sqrt(x**2 + y**2)
                xAct = hypRatio * x
                yAct = hypRatio * y
                return xAct, yAct

            # List of activities
            peList = [ self.peA]
            # List of ring radii
            rList = [ 0, Rs ]
            # Depth of alignment
            #rAlign = 3.

            rAlign = int_width#*(2/3)#3.#int_width
            # List to store particle positions and types
            pos = []
            typ = []
            rOrient = []
            # z-value for simulation initialization
            z = 0.5

            for i in range(0,len(peList)):

                rMin = rList[i]             # starting distance for particle placement
                rMax = rList[i + 1]         # maximum distance for particle placement

                ver = np.sin(60*np.pi/180)*latNet#np.sqrt(0.75) * latNet   # vertical shift between lattice rows
                hor = latNet / 2.0             # horizontal shift between lattice rows
                x = 0
                y = 0
                shift = 0

                if lx < ly:
                    rMax2 = hx

                    while (x <= rMax2) & (len(pos)<self.partNumA):

                        #r = computeDistance(x, y)
                        # Check if x-position is large enough
                        if y <rMin: # <(rMin + (latNet / 2.)):
                            y += latNet
                            continue

                        # Check if x-position is too large
                        if y >(rMax):#>= (rMax - (latNet/2.)):
                            x += ver
                            shift += 1
                            if shift % 2:
                                y = hor
                            else:
                                y = 0
                            continue

                        # Whether or not particle is oriented
                        if y > (rList[1]):
                            # Aligned
                            rOrient.append(1)
                        else:
                            # Random
                            rOrient.append(0)

                        # If the loop makes it this far, append
                        pos.append((x, y, z))
                        typ.append(i)

                        if x != 0 and y != 0:
                            # Mirror positions, alignment and type

                            if (len(pos)<self.partNumA):
                                pos.append((-x, y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)
                            if (len(pos)<self.partNumA):
                                pos.append((-x, -y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)
                            if (len(pos)<self.partNumA):
                                pos.append((x, -y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)

                        # y must be zero
                        elif (x != 0) & (len(pos)<self.partNumA):
                            pos.append((-x, y, z))
                            rOrient.append(rOrient[-1])
                            typ.append(i)

                            #typ.append(i)
                        # x must be zero
                        elif (y != 0) & (len(pos)<self.partNumA):
                            pos.append((x, -y, z))
                            rOrient.append(rOrient[-1])
                            typ.append(i)

                            #typ.append(i)

                        # Increment counter
                        y += latNet

                else:
                    rMax2 = hy

                    while (y <= rMax2) & (len(pos)<self.partNumA):

                        #r = computeDistance(x, y)
                        # Check if x-position is large enough
                        if x <rMin: # <(rMin + (latNet / 2.)):
                            x += latNet
                            continue

                        # Check if x-position is too large
                        if x >(rMax):#>= (rMax - (latNet/2.)):
                            y += ver
                            shift += 1
                            if shift % 2:
                                x = hor
                            else:
                                x = 0
                            continue

                        # Whether or not particle is oriented
                        if x > (rList[1]):
                            # Aligned
                            rOrient.append(1)
                        else:
                            # Random
                            rOrient.append(0)

                        # If the loop makes it this far, append
                        pos.append((x, y, z))
                        typ.append(i)

                        if x != 0 and y != 0:
                            # Mirror positions, alignment and type
                            if (len(pos)<self.partNumA):
                                pos.append((-x, y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)
                            if (len(pos)<self.partNumA):
                                pos.append((-x, -y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)
                            if (len(pos)<self.partNumA):
                                pos.append((x, -y, z))
                                rOrient.append(rOrient[-1])
                                typ.append(i)

                        # y must be zero
                        elif (x != 0) & (len(pos)<self.partNumA):
                            pos.append((-x, y, z))
                            rOrient.append(rOrient[-1])
                            typ.append(i)

                            #typ.append(i)
                        # x must be zero
                        elif (y != 0) & (len(pos)<self.partNumA):
                            pos.append((x, -y, z))
                            rOrient.append(rOrient[-1])
                            typ.append(i)

                            #typ.append(i)

                        # Increment counter
                        x += latNet

            # Update number of particles in gas and dense phase

            NLiq = len(pos)


            NGas = self.partNum - NLiq

            typ_A=0
            typ_B=0

            gas_B=self.partNumB-typ_B
            gas_A=self.partNumA-typ_A

            # Set this according to phiTotal
            areaParts = self.partNum * np.pi * (0.25)
            abox = (areaParts / self.phi)
            lbox = np.sqrt(abox)
            hbox = lbox / 2.

            import utility

            utility_functs = utility.utility(lbox)

            tooClose = 0.9

            # Compute mesh

            #if
            nBinsx = (utility_functs.getNBins(lx, self.r_cut))
            nBinsy = (utility_functs.getNBins(ly, self.r_cut))
            sizeBinx = utility_functs.roundUp((lx / nBinsx), 6)
            sizeBiny = utility_functs.roundUp((ly / nBinsy), 6)

            # Place particles in gas phase
            count = 0
            gaspos = []
            binParts = [[[] for b in range(nBinsy)] for a in range(nBinsx)]

            while count < NGas:
                place = 1
                # Generate random position
                gasx = (np.random.rand() - 0.5) * lx
                gasy = (np.random.rand() - 0.5) * ly
                if (lx <= ly) & (np.abs(gasy) <= (rList[-1] + (tooClose / 2.))):
                    continue
                elif (ly <= lx) & (np.abs(gasx) <= (rList[-1] + (tooClose / 2.))):
                    continue

                # Are any gas particles too close?
                tmpx = gasx + hx
                tmpy = gasy + hy

                if tmpx > lx:
                    tmpx -= lx
                if tmpy > ly:
                    tmpy -= ly

                indx = int(tmpx / sizeBinx)
                indy = int(tmpy / sizeBiny)
                # Get index of surrounding bins
                lbin = indx - 1  # index of left bins
                rbin = indx + 1  # index of right bins
                bbin = indy - 1  # index of bottom bins
                tbin = indy + 1  # index of top bins

                if rbin == nBinsx:
                    rbin -= nBinsx  # adjust if wrapped
                elif lbin == -1:
                    lbin += nBinsx

                if tbin == nBinsy:
                    tbin -= nBinsy  # adjust if wrapped
                elif bbin == -1:
                    bbin += nBinsy

                hlist = [lbin, indx, rbin]  # list of horizontal bin indices
                vlist = [bbin, indy, tbin]  # list of vertical bin indices

                # Loop through all bins
                for h in range(0, len(hlist)):
                    for v in range(0, len(vlist)):
                        # Take care of periodic wrapping for position
                        wrapX = 0.0
                        wrapY = 0.0
                        if h == 0 and hlist[h] == -1:
                            wrapX -= lbox
                        if h == 2 and hlist[h] == 0:
                            wrapX += lbox
                        if v == 0 and vlist[v] == -1:
                            wrapY -= lbox
                        if v == 2 and vlist[v] == 0:
                            wrapY += lbox
                        # Compute distance between particles

                        if binParts[hlist[h]][vlist[v]]:
                            for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                                # Get index of nearby particle
                                ref = binParts[hlist[h]][vlist[v]][b]
                                r = interDist(gasx, gasy,
                                              gaspos[ref][0] + wrapX,
                                              gaspos[ref][1] + wrapY)
                                # Round to 4 decimal places
                                r = round(r, 4)
                                # If too close, generate new position
                                if r <= tooClose:
                                    place = 0
                                    break
                        if place == 0:
                            break
                    if place == 0:
                        break

                # Is it safe to append the particle?
                if place == 1:
                    binParts[indx][indy].append(count)
                    gaspos.append((gasx, gasy, z))
                    rOrient.append(1)       # not oriented
                    typ.append(1)           # final particle type, same as outer ring
                    count += 1              # increment count

            ## Get each coordinate in a list
            #print("N_liq: {}").format(len(pos))
            #print("Intended N_liq: {}").format(NLiq)
            #print("N_gas: {}").format(len(gaspos))
            #print("Intended N_gas: {}").format(NGas)
            #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
            #print("Intended N: {}").format(partNum)
            pos = pos + gaspos

            NGas_shift=NGas

            for i in range(0,NGas):
                j=NLiq+i
                rand_val=random.random()
                xB_gas=gas_B/NGas_shift
                if rand_val<=xB_gas:
                    typ[j]=1
                    typ_B+=1
                    gas_B-=1
                    NGas_shift-=1
                else:
                    typ[j]=0
                    typ_A+=1
                    gas_A-=1
                    NGas_shift-=1
            typ_arr=np.array(typ)
            id0=np.where(typ_arr==0)
            id1=np.where(typ_arr==1)


            x, y, z = zip(*pos)
            ## Plot as scatter
            #cs = np.divide(typ, float(len(peList)))
            #cs = rOrient
            #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
            #ax = plt.gca()
            #ax.set_aspect('equal')
            partNum = len(pos)
            peList = [ self.peA, self.peB]
            # Get the number of types
            uniqueTyp = []
            for i in typ:
                if i not in uniqueTyp:
                    uniqueTyp.append(i)
            # Get the number of each type
            particles = [ 0 for x in range(0, len(uniqueTyp)) ]
            for i in range(0, len(uniqueTyp)):
                for j in typ:
                    if uniqueTyp[i] == j:
                        particles[i] += 1
            # Convert types to letter values
            unique_char_types = []
            for i in uniqueTyp:
                unique_char_types.append( chr(ord('@') + i+1) )
            char_types = []
            for i in typ:
                char_types.append( chr(ord('@') + i+1) )

            # Get a list of activities for all particles
            pe = []
            for i in typ:
                pe.append(peList[i])

            hoomd.context.initialize()
            set_box = hoomd.data.boxdim(Lx=lx, Ly=ly, Lz=0, dimensions=2)

            # A small shift to help with the periodic box
            snap = hoomd.data.make_snapshot(N = self.partNum,
                                            box = hoomd.data.boxdim(Lx=lx,
                                                                    Ly=ly,
                                                                    dimensions=2),
                                            particle_types = unique_char_types)

            # Set positions/types for all particles

            snap.particles.position[:] = pos[:]
            snap.particles.typeid[:] = typ[:]
            snap.particles.types[:] = char_types[:]

            # Initialize the system
            system = hoomd.init.read_snapshot(snap)
            all = hoomd.group.all()
            groups = []
            for i in unique_char_types:
                groups.append(hoomd.group.type(type=i))

            # Set particle potentials
            nl = hoomd.md.nlist.cell()
            lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
            lj.set_params(mode='shift')
            for i in range(0, len(unique_char_types)):
                for j in range(i, len(unique_char_types)):
                    lj.pair_coeff.set(unique_char_types[i],
                                      unique_char_types[j],
                                      epsilon=self.eps, sigma=self.sigma)

            # Brownian integration
            brownEquil = 10000

            hoomd.md.integrate.mode_standard(dt=self.dt)
            bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
            hoomd.run(brownEquil)

            # Set activity of each group
            np.random.seed(self.seed2)                           # seed for random orientations
            angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
            activity = []
            for i in range(0, self.partNum):
                if rOrient[i] == 0:
                    x = (np.cos(angle[i])) * pe[i]
                    y = (np.sin(angle[i])) * pe[i]
                else:
                    if lx <= ly:
                        if pos[i][1]>0:
                            x, y = (0, -pe[i])
                        else:
                            x, y = (0, pe[i])
                    else:
                        if pos[i][0]>0:
                            x, y = (-pe[i], 0)
                        else:
                            x, y = (pe[i], 0)
                z = 0.
                tuple = (x, y, z)
                activity.append(tuple)
            # Implement the activities in hoomd
            hoomd.md.force.active(group=all,
                                  seed=self.seed3,
                                  f_lst=activity,
                                  rotation_diff=self.D_r,
                                  orientation_link=False,
                                  orientation_reverse_link=True)

            # Name the file from parameters
            #out = "cluster_pe"
            #for i in peList:
            #    out += str(int(i))
            #    out += "_"
            #out += "r"
            #for i in range(1, len(rList)):
            #    out += str(int(rList[i]))
            #    out += "_"
            #out += "rAlign_" + str(rAlign) + ".gsd"
            out = "slow_membrane_pa" + str(int(self.peA))
            out += "_pb" + str(int(self.peB))
            out += "_phi" + str(self.intPhi)
            out += "_eps" + str(self.eps)
            out += "_xa" + str(self.partFracA)
            out += "_pNum" + str(self.partNum)
            out += "_dtau" + "{:.1e}".format(self.dt)
            out += ".gsd"

            # Write dump

            hoomd.dump.gsd(out,
                           period=self.dumpFreq,
                           group=all,
                           overwrite=True,
                           phase=-1,
                           dynamic=['attribute', 'property', 'momentum'])

            # Run

            hoomd.run(self.totTsteps)
