#!/usr/bin/env python3

# Initial imports
import sys
import os

import numpy as np

import theory

class run_sim:
    def __init__(self, hoomdPath, runFor, dumpFreq, partPercA, peA, peB, partNum, intPhi, eps, aspect_ratio, seed1, seed2, seed3, seed4, seed5, kT, threeEtaPiSigma, sigma, r_cut, tauLJ, epsA, epsB, dt):

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
        self.kT = kT                        # temperature
        self.threeEtaPiSigma = threeEtaPiSigma           # drag coefficient
        self.sigma = sigma                     # particle diameter
        self.D_t = self.kT / self.threeEtaPiSigma      # translational diffusion constant
        self.D_r = (3.0 * self.D_t) / (self.sigma**2)  # rotational diffusion constant
        self.tauBrown = (self.sigma**2) / (4 * self.D_t)     # brownian time scale (invariant)
        self.r_cut = r_cut

        self.theory_functs = theory.theory()

        self.epsA = epsA
        self.epsB = epsB
        self.epsA = (self.epsA if (self.epsA >= self.epsB) else self.epsB)   # use the larger epsilon. Doesn't matter since these are the same
        self.epsAB = (self.epsA + self.epsB) / 2.0             # AB interaction well depth

        self.tauLJ = tauLJ  # use the smaller tauLJ.  Doesn't matter since these are the same

        self.dt = dt 
        self.simLength = self.runFor * self.tauBrown               # how long to run (in tauBrown)
        self.simTauLJ = self.simLength / self.tauLJ                # how long to run (in tauLJ)
        self.totTsteps = int(self.simLength / self.dt)             # how many tsteps to run
        self.numDumps = float(self.simLength / self.dumpFreq)           # dump data every 0.1 tauBrown.

        #if self.dumpFreq==0:
        self.dumpFreq = float(self.totTsteps / self.numDumps)      # normalized dump frequency.
        #self.dumpFreq = int(dumpFreq)                    # ensure this is an integer

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

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

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
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Initialize system
        hoomd.context.initialize()

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

        lx_box = lbox
        ly_box = lbox
        hx_box = lx_box / 2
        hy_box = ly_box / 2
        import utility

        utility_functs = utility.utility(lx_box, ly_box)

        tooClose = 0.9

        # Compute mesh

        nBins_x = (utility_functs.getNBins(lx_box, self.r_cut))
        nBins_y = (utility_functs.getNBins(ly_box, self.r_cut))
        sizeBin_x = utility_functs.roundUp((lx_box / nBins_x), 6)
        sizeBin_y = utility_functs.roundUp((ly_box / nBins_y), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins_x)] for a in range(nBins_y)]
        while count < NGas:
            place = 1
            # Generate random position
            gasx = (np.random.rand() - 0.5) * lx_box
            gasy = (np.random.rand() - 0.5) * ly_box
            r = computeDistance(gasx, gasy)

            # Is this an HCP bin?
            if r <= (rList[-1] + (latNet/2.) + (tooClose / 2.)):
                continue

            # Are any gas particles too close?
            tmpx = gasx + hx_box
            tmpy = gasy + hy_box
            indx = int(tmpx / sizeBin_x)
            indy = int(tmpy / sizeBin_y)
            # Get index of surrounding bins
            lbin = indx - 1  # index of left bins
            rbin = indx + 1  # index of right bins
            bbin = indy - 1  # index of bottom bins
            tbin = indy + 1  # index of top bins
            if rbin == nBins_x:
                rbin -= nBins_x  # adjust if wrapped
            if tbin == nBins_y:
                tbin -= nBins_y  # adjust if wrapped
            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lx_box
                    if h == 2 and hlist[h] == 0:
                        wrapX += lx_box
                    if v == 0 and vlist[v] == -1:
                        wrapY -= ly_box
                    if v == 2 and vlist[v] == 0:
                        wrapY += ly_box
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
        print(type(pos[:]))
        print(len(pos[:]))
        print(np.shape(pos[:]))


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
                                        box = hoomd.data.boxdim(Lx=lx_box,
                                                                Ly=ly_box,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        # Set positions/types for all particles
        print(self.partNum)
        print(unique_char_types)
        print(len(pos[:]))
        print(type(pos[:]))
        print(np.shape(pos[:]))
        print(type(snap.particles.position[:]))
        print(np.shape(snap.particles.position[:]))


        snap.particles.position[:] = pos[:]
        snap.particles.typeid[:] = typ[:]
        snap.particles.types[:] = char_types[:]
        stop
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

        # Initialize system
        hoomd.context.initialize()

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

            import utility

            utility_functs = utility.utility(lx, ly)

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
                            wrapX -= lx
                        if h == 2 and hlist[h] == 0:
                            wrapX += lx
                        if v == 0 and vlist[v] == -1:
                            wrapY -= ly
                        if v == 2 and vlist[v] == 0:
                            wrapY += ly
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

    def fast_penetrate_slow_constrained_membrane(self):

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        ## Initialize system
        #hoomd.context.initialize()

        def roundUp(self, n, decimals=0):
            '''
            Purpose: Round up number of bins to account for floating point inaccuracy

            Inputs:
            n: number of bins along a given length of box

            decimals (optional): exponent of multiplier for rounding (default=0)

            Output:
            num_bins: number of bins along respective box length rounded up
            '''
            import math
            multiplier = 10 ** decimals
            num_bins = math.ceil(n * multiplier) / multiplier
            return num_bins

        import math
        area_ratio = self.length * self.width

        box_area = self.partNum/self.phi
        box_length = (box_area/area_ratio)**0.5

        latNet = self.theory_functs.phiToLat(self.phi)

        #latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        if self.length <= self.width:
            lx = self.length * box_length
            lx_part = math.ceil(lx / (latNet*np.sin(60*np.pi/180)))
            lx = lx_part * latNet

            ly = box_area / lx
            ly_part = math.ceil(ly/(latNet))
            ly = ly_part * latNet

            mem_part_width = self.partNumA / ly_part
            mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

        else:
            ly = self.width * box_length
            ly_part = math.ceil(ly / (latNet*np.sin(60*np.pi/180)))
            ly = ly_part * latNet

            lx = box_area / ly
            lx_part = math.ceil(lx/latNet)
            lx = lx_part * latNet

            mem_part_width = self.partNumA / ly_part
            mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

        hx = lx/2
        hy = ly/2

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd



        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity


        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG


        if peNet >= 50:
            phiG = self.theory_functs.compPhiG(peNet, latNet)
        else:
            phiG = 0
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

        Rs = mem_width / 2#thickness / 2

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
        rList = [ 0, Rs]
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



                while (x <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if y <rMin: # <(rMin + (latNet / 2.)):
                        y += hor#latNet
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

                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    y += latNet

            else:
                rMax2 = hy


                while (y <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if x <rMin: # <(rMin + (latNet / 2.)):
                        x += hor#latNet
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
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    x += latNet
        
        import matplotlib.pyplot as plt
        x, y, z_new = zip(*pos)

        if lx < ly:
            max_x = np.where(x == np.max(x))[0]
            min_x = np.where(x == np.min(x))[0]

            min_y_top_edge = np.min(np.array(y)[max_x])
            min_y_bot_edge = np.min(np.array(y)[min_x])

            max_y_top_edge = np.max(np.array(y)[max_x])
            max_y_bot_edge = np.max(np.array(y)[min_x])

            min_y = np.min(y)
            max_y = np.max(y)

            if (min_y_top_edge == min_y_bot_edge) & (max_y_top_edge == max_y_bot_edge):
                if min_y_top_edge == min_y:
                    y_min_new_top_edge = min_y_top_edge + hor
                else:
                    y_min_new_top_edge = min_y_top_edge - hor

                num_y = int(round(np.abs(y_min_new_top_edge * 2) / latNet))

                new_y = np.linspace(y_min_new_top_edge, np.abs(y_min_new_top_edge), num=num_y)

                new_x = np.ones(len(new_y)) * np.max(x) + ver
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_y[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)


                x, y, z_new = zip(*pos)

                pos_final = []
                y_min = np.min(y)
                y_max = np.max(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))

                x, y, z_new = zip(*pos_final)
                lx = 2 * np.max(x) + ver
                hx = lx / 2
                ly = lx * (self.width / self.length)
                hy = ly / 2

        elif ly < lx:

            max_y = np.where(y == np.max(y))[0]
            min_y = np.where(y == np.min(y))[0]

            min_x_top_edge = np.min(np.array(x)[max_y])
            min_x_bot_edge = np.min(np.array(x)[min_y])

            max_x_top_edge = np.max(np.array(x)[max_y])
            max_x_bot_edge = np.max(np.array(x)[min_y])

            min_x = np.min(x)
            max_x = np.max(x)
            
            if (min_x_top_edge == min_x_bot_edge) & (max_x_top_edge == max_x_bot_edge):
                
                if min_x_top_edge == min_x:
                    x_min_new_top_edge = min_x_top_edge + hor
                else:
                    x_min_new_top_edge = min_x_top_edge - hor

                num_x = int(round(np.abs(x_min_new_top_edge * 2) / latNet))

                new_x = np.linspace(x_min_new_top_edge, np.abs(x_min_new_top_edge), num=num_x+1)

                new_y = np.ones(len(new_x)) * np.max(y) + ver

                
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_x[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)

                x, y, z_new = zip(*pos)

                pos_final = []
                y_max = np.max(y)
                y_min = np.min(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))
                
                x, y, z_new = zip(*pos_final)

                ly = 2 * np.max(y) + ver
                hy = ly / 2
                lx = ly * (self.length / self.width)
                hx = lx / 2

        else:
            
            x, y, z_new = zip(*pos)
            pos_final = []
            for i in range(0, len(x)):
                pos_final.append((x[i], y[i] - (np.max(y) + np.min(y))/2, z_new[i]))
                if x[i] > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)
                typ.append(0)
        # Update number of particles in gas and dense phase
        
        NLiq = len(pos_final)
        if NLiq < self.partNum:
            NGas = self.partNum - NLiq
        else:
            NGas = 1
        NGas = 1
        
        typ_A=0
        typ_B=0

        gas_B=self.partNumB-typ_B
        gas_A=self.partNumA-typ_A

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)

        import utility

        utility_functs = utility.utility(lx, ly)

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
        wall_width = latNet / 2# 0.5

        tooClose = latNet / 2
        wall_distance = 0
        lx = 4 * (rList[-1] + (tooClose))
        x3, y3, z3 = zip(*pos_final)
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
        else:
            rMax_temp = np.max(y3) + latNet/2
        
        while count < NGas:
            place = 1
            # Generate random position
            #if lx > ly: 
            #    gasy = 0
            #    gasx = np.max(x3) + latNet/2
            #else:
            #    gasx = 0
            #    gasy = np.max(y3) + latNet/2
            if lx > ly:
                gas_width = rMax_temp + (hx - rMax_temp) * 0.4
                
                gasy = ver#0
                gasx = (np.random.rand() - 0.5) * lx
            else:
                gas_width = rMax_temp + (hy - rMax_temp) * 0.4
                gasx = 0
                gasy = (np.random.rand() - 0.5) * ly

            if (lx <= ly):
                if (gasy <= (rList[-1] + (tooClose))) | (gasy>=gas_width):
                    continue
            elif (ly <= lx):
                if (gasx <= (rList[-1] + (tooClose))) | (gasx>=gas_width):
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
            #elif lbin == -1:
            #    lbin += nBinsx

            if tbin == nBinsy:
                tbin -= nBinsy  # adjust if wrapped
            #elif bbin == -1:
            #    bbin += nBinsy

            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lx
                    if h == 2 and hlist[h] == 0:
                        wrapX += lx
                    if v == 0 and vlist[v] == -1:
                        wrapY -= ly
                    if v == 2 and vlist[v] == 0:
                        wrapY += ly
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
        pos_final = pos_final + gaspos
        #pos_final = gaspos
        #typ = [typ[-1]]
        x2, y2, z2 = zip(*pos_final)
        
        """
        wallpos = []
        if lx > ly: 
            y_val = -hy
            x_val = np.max(x3) + latNet/2
            while y_val < hy:
                wallpos.append((x_val, y_val, z))
                wallpos.append((-x_val, y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                y_val += 0.1
        else:
            y_val = np.max(y3) + latNet/2
            x_val = -hx
            while x_val < hx:
                wallpos.append((x_val, y_val, z))
                wallpos.append((x_val, -y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                x_val += 0.1
        
            gasy = 0
            gasx = np.max(x3) + latNet/2
        pos_final2 = pos_final + wallpos
        x, y, z = zip(*pos_final2)
        plt.scatter(x,y, s=1.0)
        plt.xlim([-hx, hx])
        plt.ylim([-hy, hy])
        plt.show()
        """
        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        #x2, y2, z2 = zip(*gaspos)
        
        typ_arr=np.array(typ)
        id0=np.where(typ_arr==0)
        id1=np.where(typ_arr==1)


        #wallstructure=wall.group()

        #wallstructure.add_plane((0,0,0),(0,2,1))
        #wallstructure.add_plane((0,0,0),(0,2,1))

        x, y, z = zip(*pos_final)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos_final)
        #peList = [ self.peA, self.peB, 0.]
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

        # Now we make the system in hoomd
        hoomd.context.initialize()
        partNum = len(pos_final)

        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = partNum,
                                        box = hoomd.data.boxdim(Lx=lx,
                                                                Ly=ly,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        #snap = hoomd.data.make_snapshot(N = self.partNum,
        #                                box = hoomd.data.boxdim(Lx=lx,
        #                                                        Ly=ly,
        #                                                        dimensions=2),
        #                                particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos_final[:]
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

        # Add wall
        wallstructure=md.wall.group()
        wallstructure2=md.wall.group()
        wallstructure4=md.wall.group()
        wallstructure5=md.wall.group()
        
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * ly ), 2)
        else:
            rMax_temp = np.max(y3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * lx ), 2)
        
        part_frac_temp = round(( (partNum - NGas) / partNum), 3)
        

        if lx > ly:
            wallstructure2.add_plane(origin=(-rMax_temp,0,0),normal=(1,0,0))
            wallstructure.add_plane(origin=(rMax_temp,0,0),normal=(-1,0,0))
            wallstructure4.add_plane(origin=(0,hy,0),normal=(0,-1,0))
            wallstructure5.add_plane(origin=(0,-hy,0),normal=(0,1,0))
        else:
            wallstructure2.add_plane(origin=(0,-rMax_temp,0),normal=(0,1,0))
            wallstructure.add_plane(origin=(0,rMax_temp,0),normal=(0,-1,0))
            wallstructure4.add_plane(origin=(hx,0,0),normal=(-1,0,0))
            wallstructure5.add_plane(origin=(-hx,0,0),normal=(1,0,0))

        lj2=md.wall.lj(wallstructure, r_cut=self.r_cut)
        lj3=md.wall.lj(wallstructure2, r_cut=self.r_cut)
        lj5=md.wall.lj(wallstructure4, r_cut=self.r_cut)
        lj6=md.wall.lj(wallstructure5, r_cut=self.r_cut)


        lj2.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj2.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj3.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj3.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj5.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj5.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj6.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj6.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        #hoomd.run(brownEquil)
        

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                if lx <= ly:
                    if pos_final[i][1]>0:
                        x, y = (0, -pe[i])
                    else:
                        x, y = (0, pe[i])
                else:
                    if pos_final[i][0]>0:
                        x, y = (-pe[i], 0)
                    else:
                        x, y = (pe[i], 0)
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        self.D_r = 0.
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
        out += "_phi" + str(phi_temp)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(part_frac_temp)
        out += "_pNum" + str(partNum)
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

    
    def fast_interior_slow_constrained_membrane(self):

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        ## Initialize system
        #hoomd.context.initialize()

        def roundUp(self, n, decimals=0):
            '''
            Purpose: Round up number of bins to account for floating point inaccuracy

            Inputs:
            n: number of bins along a given length of box

            decimals (optional): exponent of multiplier for rounding (default=0)

            Output:
            num_bins: number of bins along respective box length rounded up
            '''
            import math
            multiplier = 10 ** decimals
            num_bins = math.ceil(n * multiplier) / multiplier
            return num_bins

        import math
        area_ratio = self.length * self.width

        box_area = self.partNum/self.phi
        box_length = (box_area/area_ratio)**0.5

        latNet = self.theory_functs.phiToLat(self.phi)

        #latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        if self.length <= self.width:
            lx = self.length * box_length
            lx_part = math.ceil(lx / (latNet*np.sin(60*np.pi/180)))
            lx = lx_part * latNet

            ly = box_area / lx
            ly_part = math.ceil(ly/(latNet))
            ly = ly_part * latNet

            mem_part_width = self.partNumA / ly_part
            mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

        else:
            ly = self.width * box_length
            ly_part = math.ceil(ly / (latNet*np.sin(60*np.pi/180)))
            ly = ly_part * latNet

            lx = box_area / ly
            lx_part = math.ceil(lx/latNet)
            lx = lx_part * latNet

            mem_part_width = self.partNumA / ly_part
            mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

        hx = lx/2
        hy = ly/2

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd



        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity


        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG


        if peNet >= 50:
            phiG = self.theory_functs.compPhiG(peNet, latNet)
        else:
            phiG = 0
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

        Rs = mem_width / 2#thickness / 2

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
        rList = [ 0, Rs]
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



                while (x <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if y <rMin: # <(rMin + (latNet / 2.)):
                        y += hor#latNet
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

                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    y += latNet

            else:
                rMax2 = hy


                while (y <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if x <rMin: # <(rMin + (latNet / 2.)):
                        x += hor#latNet
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
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    x += latNet
        
        import matplotlib.pyplot as plt
        x, y, z_new = zip(*pos)

        if lx < ly:
            max_x = np.where(x == np.max(x))[0]
            min_x = np.where(x == np.min(x))[0]

            min_y_top_edge = np.min(np.array(y)[max_x])
            min_y_bot_edge = np.min(np.array(y)[min_x])

            max_y_top_edge = np.max(np.array(y)[max_x])
            max_y_bot_edge = np.max(np.array(y)[min_x])

            min_y = np.min(y)
            max_y = np.max(y)

            if (min_y_top_edge == min_y_bot_edge) & (max_y_top_edge == max_y_bot_edge):
                if min_y_top_edge == min_y:
                    y_min_new_top_edge = min_y_top_edge + hor
                else:
                    y_min_new_top_edge = min_y_top_edge - hor

                num_y = int(round(np.abs(y_min_new_top_edge * 2) / latNet))

                new_y = np.linspace(y_min_new_top_edge, np.abs(y_min_new_top_edge), num=num_y)

                new_x = np.ones(len(new_y)) * np.max(x) + ver
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_y[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)


                x, y, z_new = zip(*pos)

                pos_final = []
                y_min = np.min(y)
                y_max = np.max(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))

                x, y, z_new = zip(*pos_final)
                lx = 2 * np.max(x) + ver
                hx = lx / 2
                ly = lx * (self.width / self.length)
                hy = ly / 2

        elif ly < lx:

            max_y = np.where(y == np.max(y))[0]
            min_y = np.where(y == np.min(y))[0]

            min_x_top_edge = np.min(np.array(x)[max_y])
            min_x_bot_edge = np.min(np.array(x)[min_y])

            max_x_top_edge = np.max(np.array(x)[max_y])
            max_x_bot_edge = np.max(np.array(x)[min_y])

            min_x = np.min(x)
            max_x = np.max(x)
            
            if (min_x_top_edge == min_x_bot_edge) & (max_x_top_edge == max_x_bot_edge):
                
                if min_x_top_edge == min_x:
                    x_min_new_top_edge = min_x_top_edge + hor
                else:
                    x_min_new_top_edge = min_x_top_edge - hor

                num_x = int(round(np.abs(x_min_new_top_edge * 2) / latNet))

                new_x = np.linspace(x_min_new_top_edge, np.abs(x_min_new_top_edge), num=num_x+1)

                new_y = np.ones(len(new_x)) * np.max(y) + ver

                
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_x[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)

                x, y, z_new = zip(*pos)

                pos_final = []
                y_max = np.max(y)
                y_min = np.min(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))
                
                x, y, z_new = zip(*pos_final)

                ly = 2 * np.max(y) + ver
                hy = ly / 2
                lx = ly * (self.length / self.width)
                hx = lx / 2

        else:
            
            x, y, z_new = zip(*pos)
            pos_final = []
            for i in range(0, len(x)):
                pos_final.append((x[i], y[i] - (np.max(y) + np.min(y))/2, z_new[i]))
                if x[i] > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)
                typ.append(0)
        # Update number of particles in gas and dense phase
        for i in range(1, len(typ)):
            typ[i]=1
        
        NLiq = len(pos_final)
        if NLiq < self.partNum:
            NGas = self.partNum - NLiq
        else:
            NGas = 1
        NGas = 0
        
        
        typ_A=0
        typ_B=0

        gas_B=self.partNumB-typ_B
        gas_A=self.partNumA-typ_A

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)

        import utility

        utility_functs = utility.utility(lx, ly)

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
        wall_width = latNet / 2# 0.5

        tooClose = latNet / 2
        wall_distance = 0
        lx = 4 * (rList[-1] + (tooClose))
        
        x3, y3, z3 = zip(*pos_final)
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
        else:
            rMax_temp = np.max(y3) + latNet/2
        
        while count < NGas:
            place = 1
            # Generate random position
            #if lx > ly: 
            #    gasy = 0
            #    gasx = np.max(x3) + latNet/2
            #else:
            #    gasx = 0
            #    gasy = np.max(y3) + latNet/2
            if lx > ly:
                gas_width = rMax_temp + (hx - rMax_temp) * 0.4
                
                gasy = ver#0
                gasx = (np.random.rand() - 0.5) * lx
            else:
                gas_width = rMax_temp + (hy - rMax_temp) * 0.4
                gasx = 0
                gasy = (np.random.rand() - 0.5) * ly

            if (lx <= ly):
                if (gasy <= (rList[-1] + (tooClose))) | (gasy>=gas_width):
                    continue
            elif (ly <= lx):
                if (gasx <= (rList[-1] + (tooClose))) | (gasx>=gas_width):
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
            #elif lbin == -1:
            #    lbin += nBinsx

            if tbin == nBinsy:
                tbin -= nBinsy  # adjust if wrapped
            #elif bbin == -1:
            #    bbin += nBinsy

            hlist = [lbin, indx, rbin]  # list of horizontal bin indices
            vlist = [bbin, indy, tbin]  # list of vertical bin indices

            # Loop through all bins
            for h in range(0, len(hlist)):
                for v in range(0, len(vlist)):
                    # Take care of periodic wrapping for position
                    wrapX = 0.0
                    wrapY = 0.0
                    if h == 0 and hlist[h] == -1:
                        wrapX -= lx
                    if h == 2 and hlist[h] == 0:
                        wrapX += lx
                    if v == 0 and vlist[v] == -1:
                        wrapY -= ly
                    if v == 2 and vlist[v] == 0:
                        wrapY += ly
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
        pos_final = pos_final + gaspos
        #pos_final = gaspos
        #typ = [typ[-1]]
        x2, y2, z2 = zip(*pos_final)
        
        """
        wallpos = []
        if lx > ly: 
            y_val = -hy
            x_val = np.max(x3) + latNet/2
            while y_val < hy:
                wallpos.append((x_val, y_val, z))
                wallpos.append((-x_val, y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                y_val += 0.1
        else:
            y_val = np.max(y3) + latNet/2
            x_val = -hx
            while x_val < hx:
                wallpos.append((x_val, y_val, z))
                wallpos.append((x_val, -y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                x_val += 0.1
        
            gasy = 0
            gasx = np.max(x3) + latNet/2
        pos_final2 = pos_final + wallpos
        x, y, z = zip(*pos_final2)
        plt.scatter(x,y, s=1.0)
        plt.xlim([-hx, hx])
        plt.ylim([-hy, hy])
        plt.show()
        """
        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        #x2, y2, z2 = zip(*gaspos)
        
        typ_arr=np.array(typ)
        id0=np.where(typ_arr==0)
        id1=np.where(typ_arr==1)


        #wallstructure=wall.group()

        #wallstructure.add_plane((0,0,0),(0,2,1))
        #wallstructure.add_plane((0,0,0),(0,2,1))

        x, y, z = zip(*pos_final)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos_final)
        #peList = [ self.peA, self.peB, 0.]
        peList = [ self.peB, self.peA]
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

        # Now we make the system in hoomd
        hoomd.context.initialize()
        partNum = len(pos_final)

        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = partNum,
                                        box = hoomd.data.boxdim(Lx=lx,
                                                                Ly=ly,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        #snap = hoomd.data.make_snapshot(N = self.partNum,
        #                                box = hoomd.data.boxdim(Lx=lx,
        #                                                        Ly=ly,
        #                                                        dimensions=2),
        #                                particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos_final[:]
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

        # Add wall
        wallstructure=md.wall.group()
        wallstructure2=md.wall.group()
        wallstructure4=md.wall.group()
        wallstructure5=md.wall.group()
        
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * ly ), 2)
        else:
            rMax_temp = np.max(y3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * lx ), 2)
        
        part_frac_temp = round(( (partNum - NGas) / partNum), 3)
        

        if lx > ly:
            wallstructure2.add_plane(origin=(-rMax_temp,0,0),normal=(1,0,0))
            wallstructure.add_plane(origin=(rMax_temp,0,0),normal=(-1,0,0))
            wallstructure4.add_plane(origin=(0,hy,0),normal=(0,-1,0))
            wallstructure5.add_plane(origin=(0,-hy,0),normal=(0,1,0))
        else:
            wallstructure2.add_plane(origin=(0,-rMax_temp,0),normal=(0,1,0))
            wallstructure.add_plane(origin=(0,rMax_temp,0),normal=(0,-1,0))
            wallstructure4.add_plane(origin=(hx,0,0),normal=(-1,0,0))
            wallstructure5.add_plane(origin=(-hx,0,0),normal=(1,0,0))

        lj2=md.wall.lj(wallstructure, r_cut=self.r_cut)
        lj3=md.wall.lj(wallstructure2, r_cut=self.r_cut)
        lj5=md.wall.lj(wallstructure4, r_cut=self.r_cut)
        lj6=md.wall.lj(wallstructure5, r_cut=self.r_cut)


        lj2.force_coeff.set('A', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj2.force_coeff.set('B', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj3.force_coeff.set('A', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj3.force_coeff.set('B', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj5.force_coeff.set('A', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj5.force_coeff.set('B', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj6.force_coeff.set('A', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj6.force_coeff.set('B', sigma=wall_width, epsilon=1.0)  #plotted below in red

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        #hoomd.run(brownEquil)
        

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                if lx <= ly:
                    if pos_final[i][1]>0:
                        x, y = (0, -pe[i])
                    else:
                        x, y = (0, pe[i])
                else:
                    if pos_final[i][0]>0:
                        x, y = (-pe[i], 0)
                    else:
                        x, y = (pe[i], 0)
            z = 0.
            tuple = (x, y, z)
            activity.append(tuple)
        # Implement the activities in hoomd
        #self.D_r = 0.
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
        out = "int_membrane_pa" + str(int(self.peB))
        out += "_pb" + str(int(self.peA))
        out += "_phi" + str(phi_temp)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(part_frac_temp)
        out += "_pNum" + str(partNum)
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

    def fast_adsorb_slow_constrained_membrane(self):

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        ## Initialize system
        #hoomd.context.initialize()

        def roundUp(self, n, decimals=0):
            '''
            Purpose: Round up number of bins to account for floating point inaccuracy

            Inputs:
            n: number of bins along a given length of box

            decimals (optional): exponent of multiplier for rounding (default=0)

            Output:
            num_bins: number of bins along respective box length rounded up
            '''
            import math
            multiplier = 10 ** decimals
            num_bins = math.ceil(n * multiplier) / multiplier
            return num_bins

        import math
        area_ratio = self.length / self.width
        
        box_area = (self.partNum*(np.pi/4))/self.phi
        
        box_length = (box_area/area_ratio)**0.5
        box_width = (box_length * area_ratio)

        latNet = self.theory_functs.phiToLat(2.0)
        
        #latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)
        if self.length <= self.width:
            lx = box_width
            lx_part = math.ceil(lx / (latNet*np.sin(60*np.pi/180)))
            lx = lx_part * (latNet*np.sin(60*np.pi/180))

            ly = box_area / lx
            ly_part = math.ceil(ly/(latNet))
            ly = ly_part * latNet

            mem_part_width = 15.0
            mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

            ly = ly + mem_width
        else:
            ly = box_length
            ly_part = math.ceil(ly / (latNet*np.sin(60*np.pi/180)))
            ly = ly_part * (latNet*np.sin(60*np.pi/180))

            lx = box_area / ly
            lx_part = math.ceil(lx/latNet)
            lx = lx_part * latNet

            mem_part_width = 15.0
            mem_width = mem_part_width * latNet# (latNet*np.sin(60*np.pi/180))

            lx = lx + mem_width
        
        print(self.partNum*(np.pi/4)/((lx-mem_width)* ly))
        print(self.phi)
        
        hx = lx/2
        hy = ly/2

        #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd



        peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

        # Compute lattice spacing based on each activity


        #latNet = self.theory_functs.conForRClust2(500, 500, self.beta_A, self.beta_B, self.eps)
        # Compute gas phase density, phiG


        if peNet >= 50:
            phiG = self.theory_functs.compPhiG(peNet, latNet)
        else:
            phiG = 0
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

        Rs = mem_width / 2#thickness / 2

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
        rList = [ 0, Rs]
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



                while (x <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if y <rMin: # <(rMin + (latNet / 2.)):
                        y += hor#latNet
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

                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    y += latNet

            else:
                rMax2 = hy


                while (y <= rMax2):
                    #r = computeDistance(x, y)
                    # Check if x-position is large enough
                    if x <rMin: # <(rMin + (latNet / 2.)):
                        x += hor#latNet
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
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((-x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                    # y must be zero
                    elif (x != 0):
                        pos.append((-x, y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)
                    # x must be zero
                    elif (y != 0):
                        pos.append((x, -y, z))
                        rOrient.append(rOrient[-1])
                        typ.append(i)

                        #typ.append(i)

                    # Increment counter
                    x += latNet
        
        import matplotlib.pyplot as plt
        x, y, z_new = zip(*pos)

        
        if lx < ly:
            max_x = np.where(x == np.max(x))[0]
            min_x = np.where(x == np.min(x))[0]

            min_y_top_edge = np.min(np.array(y)[max_x])
            min_y_bot_edge = np.min(np.array(y)[min_x])

            max_y_top_edge = np.max(np.array(y)[max_x])
            max_y_bot_edge = np.max(np.array(y)[min_x])

            min_y = np.min(y)
            max_y = np.max(y)

            if (min_y_top_edge == min_y_bot_edge) & (max_y_top_edge == max_y_bot_edge):
                if min_y_top_edge == min_y:
                    y_min_new_top_edge = min_y_top_edge + hor
                else:
                    y_min_new_top_edge = min_y_top_edge - hor

                num_y = int(round(np.abs(y_min_new_top_edge * 2) / latNet))

                new_y = np.linspace(y_min_new_top_edge, np.abs(y_min_new_top_edge), num=num_y)

                new_x = np.ones(len(new_y)) * np.max(x) + ver
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_y[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)


                x, y, z_new = zip(*pos)

                pos_final = []
                y_min = np.min(y)
                y_max = np.max(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))

                x, y, z_new = zip(*pos_final)

                lx = 2 * np.max(x) + ver
                hx = lx / 2
                ly = ((self.partNum * (np.pi/4) / lx) / self.phi) + mem_width
                hy = ly / 2

        elif ly < lx:

            max_y = np.where(y == np.max(y))[0]
            min_y = np.where(y == np.min(y))[0]

            min_x_top_edge = np.min(np.array(x)[max_y])
            min_x_bot_edge = np.min(np.array(x)[min_y])

            max_x_top_edge = np.max(np.array(x)[max_y])
            max_x_bot_edge = np.max(np.array(x)[min_y])

            min_x = np.min(x)
            max_x = np.max(x)
            
            if (min_x_top_edge == min_x_bot_edge) & (max_x_top_edge == max_x_bot_edge):
                
                if min_x_top_edge == min_x:
                    x_min_new_top_edge = min_x_top_edge + hor
                else:
                    x_min_new_top_edge = min_x_top_edge - hor

                num_x = int(round(np.abs(x_min_new_top_edge * 2) / latNet))

                new_x = np.linspace(x_min_new_top_edge, np.abs(x_min_new_top_edge), num=num_x+1)

                new_y = np.ones(len(new_x)) * np.max(y) + ver

                
                for i in range(0, len(new_x)):
                    pos.append((new_x[i], new_y[i], z))
                    if new_x[i] > (rList[1]):
                        # Aligned
                        rOrient.append(1)
                    else:
                        # Random
                        rOrient.append(0)
                    typ.append(0)

                x, y, z_new = zip(*pos)

                pos_final = []
                y_max = np.max(y)
                y_min = np.min(y)
                for i in range(0, len(x)):
                    pos_final.append((x[i], y[i] - (y_max + y_min)/2, z_new[i]))
                
                x, y, z_new = zip(*pos_final)

                ly = 2 * np.max(y) + ver
                hy = ly / 2
                lx = ((self.partNum * (np.pi/4) / ly) / self.phi) + mem_width
                hx = lx / 2

        else:
            
            x, y, z_new = zip(*pos)
            pos_final = []
            for i in range(0, len(x)):
                pos_final.append((x[i], y[i] - (np.max(y) + np.min(y))/2, z_new[i]))
                if x[i] > (rList[1]):
                    # Aligned
                    rOrient.append(1)
                else:
                    # Random
                    rOrient.append(0)
                typ.append(0)
        # Update number of particles in gas and dense phase
        
        x, y, z_new = zip(*pos_final)
        
        NLiq = len(pos_final)
        if NLiq < self.partNum:
            NGas = self.partNum - NLiq
        else:
            NGas = 1

        NGas = (NLiq - self.partFracA * NLiq) / self.partFracA
        #print(round(NGas))
        #print(NLiq / (NLiq + NGas))
        #print(self.partFracA)
        #stop
        NGas = self.partNum
        
        typ_A=0
        typ_B=0

        gas_B=self.partNumB-typ_B
        gas_A=self.partNumA-typ_A

        # Set this according to phiTotal
        areaParts = self.partNum * np.pi * (0.25)
        abox = (areaParts / self.phi)

        import utility

        utility_functs = utility.utility(lx, ly)

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
        wall_width = latNet / 2# 0.5

        tooClose = latNet / 2
        wall_distance = 0
        x3, y3, z3 = zip(*pos_final)
        
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
        else:
            rMax_temp = np.max(y3) + latNet/2
        
        import utility

        utility_functs = utility.utility(lx, ly)

        while count < NGas:
            print('test')
            print(count)
            print(NGas)
            place = 1
            # Generate random position
            #if lx > ly: 
            #    gasy = 0
            #    gasx = np.max(x3) + latNet/2
            #else:
            #    gasx = 0
            #    gasy = np.max(y3) + latNet/2
            
            if lx > ly:
                gas_width = rMax_temp + (hx - rMax_temp)
                gasy = (np.random.rand() - 0.5) * ly
                gasx = (np.random.rand() - 0.5) * lx
            else:
                gas_width = rMax_temp + (hy - rMax_temp)
                gasx = (np.random.rand() - 0.5) * lx
                gasy = (np.random.rand() - 0.5) * ly

            if (lx <= ly):
                if (np.abs(gasy) <= (rMax_temp + (3*tooClose))):
                    continue
            elif (ly <= lx):
                if (np.abs(gasx) <= (rMax_temp + (3*tooClose))):
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
                    # Compute distance between particles
                    
                    if binParts[hlist[h]][vlist[v]]:
                        for b in range(0, len(binParts[hlist[h]][vlist[v]])):
                            # Get index of nearby particle
                            ref = binParts[hlist[h]][vlist[v]][b]
                            difx = utility_functs.sep_dist_x(gasx, x4[ref])
                            dify = utility_functs.sep_dist_y(gasy, y4[ref])
                            r = ( difx ** 2 + dify ** 2 ) ** 0.5                            

                            # Round to 4 decimal places
                            r = round(r, 4)

                            # If too close, generate new position
                            if r <= 3 * tooClose:
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
                x4, y4, z4 = zip(*gaspos)

        
        pos_final = pos_final + gaspos
        #pos_final = gaspos
        #typ = [typ[-1]]
        x2, y2, z2 = zip(*pos_final)
        
        """
        wallpos = []
        if lx > ly: 
            y_val = -hy
            x_val = np.max(x3) + latNet/2
            while y_val < hy:
                wallpos.append((x_val, y_val, z))
                wallpos.append((-x_val, y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                y_val += 0.1
        else:
            y_val = np.max(y3) + latNet/2
            x_val = -hx
            while x_val < hx:
                wallpos.append((x_val, y_val, z))
                wallpos.append((x_val, -y_val, z))
                rOrient.append(0)       # not oriented
                rOrient.append(0)       # not oriented
                typ.append(2)           # final particle type, same as outer ring
                typ.append(2)           # final particle type, same as outer ring
                x_val += 0.1
        
            gasy = 0
            gasx = np.max(x3) + latNet/2
        pos_final2 = pos_final + wallpos
        x, y, z = zip(*pos_final2)
        plt.scatter(x,y, s=1.0)
        plt.xlim([-hx, hx])
        plt.ylim([-hy, hy])
        plt.show()
        """
        ## Get each coordinate in a list
        #print("N_liq: {}").format(len(pos))
        #print("Intended N_liq: {}").format(NLiq)
        #print("N_gas: {}").format(len(gaspos))
        #print("Intended N_gas: {}").format(NGas)
        #print("N_liq + N_gas: {}").format(len(pos) + len(gaspos))
        #print("Intended N: {}").format(partNum)
        #x2, y2, z2 = zip(*gaspos)
        
        typ_arr=np.array(typ)
        id0=np.where(typ_arr==0)
        id1=np.where(typ_arr==1)


        #wallstructure=wall.group()

        #wallstructure.add_plane((0,0,0),(0,2,1))
        #wallstructure.add_plane((0,0,0),(0,2,1))

        x, y, z = zip(*pos_final)
        ## Plot as scatter
        #cs = np.divide(typ, float(len(peList)))
        #cs = rOrient
        #plt.scatter(x, y, s=1., c=cs, cmap='jet', edgecolors='none')
        #ax = plt.gca()
        #ax.set_aspect('equal')
        partNum = len(pos_final)
        #peList = [ self.peA, self.peB, 0.]
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

        # Now we make the system in hoomd
        hoomd.context.initialize()
        partNum = len(pos_final)

        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = partNum,
                                        box = hoomd.data.boxdim(Lx=lx,
                                                                Ly=ly,
                                                                dimensions=2),
                                        particle_types = unique_char_types)

        #snap = hoomd.data.make_snapshot(N = self.partNum,
        #                                box = hoomd.data.boxdim(Lx=lx,
        #                                                        Ly=ly,
        #                                                        dimensions=2),
        #                                particle_types = unique_char_types)

        # Set positions/types for all particles

        snap.particles.position[:] = pos_final[:]
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

        # Add wall
        wallstructure=md.wall.group()
        wallstructure2=md.wall.group()
        wallstructure4=md.wall.group()
        wallstructure5=md.wall.group()
        
        if lx > ly:
            rMax_temp = np.max(x3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * ly ), 2)
        else:
            rMax_temp = np.max(y3) + latNet/2
            phi_temp = round(( NLiq * (np.pi/4) ) / ( rMax_temp * 2 * lx ), 2)
        
        part_frac_temp = round(( (partNum - NGas) / partNum), 3)
        
        
        if lx > ly:
            wallstructure2.add_plane(origin=(-rMax_temp,0,0),normal=(1,0,0))
            wallstructure.add_plane(origin=(rMax_temp,0,0),normal=(-1,0,0))
            wallstructure4.add_plane(origin=(0,hy,0),normal=(0,-1,0))
            wallstructure5.add_plane(origin=(0,-hy,0),normal=(0,1,0))
        else:
            wallstructure2.add_plane(origin=(0,-rMax_temp,0),normal=(0,1,0))
            wallstructure.add_plane(origin=(0,rMax_temp,0),normal=(0,-1,0))
            wallstructure4.add_plane(origin=(hx,0,0),normal=(-1,0,0))
            wallstructure5.add_plane(origin=(-hx,0,0),normal=(1,0,0))

        lj2=md.wall.lj(wallstructure, r_cut=self.r_cut)
        lj3=md.wall.lj(wallstructure2, r_cut=self.r_cut)
        lj5=md.wall.lj(wallstructure4, r_cut=self.r_cut)
        lj6=md.wall.lj(wallstructure5, r_cut=self.r_cut)


        lj2.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj2.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj3.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj3.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj5.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj5.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red
        lj6.force_coeff.set('A', sigma=wall_width, epsilon=1.0)  #plotted below in red
        lj6.force_coeff.set('B', sigma=wall_width, epsilon=0.0)  #plotted below in red

        # Brownian integration
        brownEquil = 10000

        hoomd.md.integrate.mode_standard(dt=self.dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=self.kT, seed=self.seed1)
        #hoomd.run(brownEquil)
        

        # Set activity of each group
        np.random.seed(self.seed2)                           # seed for random orientations
        angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in range(0, partNum):
            if rOrient[i] == 0:
                x = (np.cos(angle[i])) * pe[i]
                y = (np.sin(angle[i])) * pe[i]
            else:
                if lx <= ly:
                    if pos_final[i][1]>0:
                        x, y = (0, -pe[i])
                    else:
                        x, y = (0, pe[i])
                else:
                    if pos_final[i][0]>0:
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
        out += "_phi" + str(self.phi)
        out += "_eps" + str(self.eps)
        out += "_xa" + str(part_frac_temp)
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

    def fast_penetrate_immobile_membrane(self):

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Initialize system
        hoomd.context.initialize()

        def roundUp(self, n, decimals=0):
            '''
            Purpose: Round up number of bins to account for floating point inaccuracy

            Inputs:
            n: number of bins along a given length of box

            decimals (optional): exponent of multiplier for rounding (default=0)

            Output:
            num_bins: number of bins along respective box length rounded up
            '''
            import math
            multiplier = 10 ** decimals
            num_bins = math.ceil(n * multiplier) / multiplier
            return num_bins

        if self.length != self.width:
            import math
            area_ratio = self.length * self.width

            box_area = self.partNum/self.phi
            box_length = (box_area/area_ratio)**0.5
            print(box_length)

            latNet = self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)

            if self.length <= self.width:
                lx = self.length * box_length
                lx_part = math.ceil(lx / (latNet*np.sin(60*np.pi/180)))
                lx = lx_part * latNet

                ly = box_area / lx
                ly_part = math.ceil(ly/(latNet))
                ly = ly_part * latNet

                mem_part_width = self.partNumA / ly_part
                mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

            else:
                ly = self.width * box_length
                ly_part = math.ceil(ly / (latNet*np.sin(60*np.pi/180)))
                ly = ly_part * latNet

                lx = box_area / ly
                lx_part = math.ceil(lx/latNet)
                lx = lx_part * latNet

                mem_part_width = self.partNumA / ly_part
                mem_width = mem_part_width * (latNet*np.sin(60*np.pi/180))

            effective_phi = self.partNum / (lx * ly)

            hx = lx/2
            hy = ly/2

            #sys.path.insert(0,self.hoomdPath)    # insert specified path to hoomdPath as first place to check for hoomd



            peNet = self.theory_functs.compPeNet(self.partFracA, self.peA, self.peB)

            # Compute lattice spacing based on each activity


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

            Rs = mem_width / 2#thickness / 2

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
            rList = [ 0, Rs]
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
                            y += hor#latNet
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
                            x += hor#latNet
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

            import matplotlib.pyplot as plt
            x, y, z = zip(*pos)

            print('x')
            print(np.max(x))
            print(np.min(x))
            print(Rs)
            print('y')
            print(np.max(y))
            print(np.min(y))
            print(hy)

            NLiq = len(pos)

            print(NLiq)
            NGas = self.partNum - NLiq

            typ_A=0
            typ_B=0

            gas_B=self.partNumB-typ_B
            gas_A=self.partNumA-typ_A

            # Set this according to phiTotal
            areaParts = self.partNum * np.pi * (0.25)
            abox = (areaParts / self.phi)

            import utility

            utility_functs = utility.utility(lx, ly)

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

                if (lx <= ly) & (np.abs(gasy) <= (rList[-1] + (tooClose /2.))):
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
                            wrapX -= lx
                        if h == 2 and hlist[h] == 0:
                            wrapX += lx
                        if v == 0 and vlist[v] == -1:
                            wrapY -= ly
                        if v == 2 and vlist[v] == 0:
                            wrapY += ly
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
            typ_arr=np.array(typ)
            id0=np.where(typ_arr==0)[0]
            id1=np.where(typ_arr==1)[0]


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
            for i in range(0, len(id1)):
                pe.append(peList[1])

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

            groupA = hoomd.group.type('A')
            groupB = hoomd.group.type('B')

            groups = []
            for i in unique_char_types:
                groups.append(hoomd.group.type(type=i))

            # Set particle potentials
            nl = hoomd.md.nlist.cell()
            lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
            lj.set_params(mode='shift')
            for i in range(0, len(unique_char_types)):
                for j in range(i, len(unique_char_types)):

                    #if (unique_char_types[j]=='B') and (unique_char_types[i]=='B'):
                    #    lj.pair_coeff.set(unique_char_types[i],
                    #                  unique_char_types[j],
                    #                  epsilon=0.0, sigma=self.sigma)
                    #else:
                    lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

            # Brownian integration
            brownEquil = 10000

            hoomd.md.integrate.mode_standard(dt=self.dt)
            bd = hoomd.md.integrate.brownian(group=groupB, kT=self.kT, seed=self.seed1)
            hoomd.run(brownEquil)

            # Set activity of each group
            np.random.seed(self.seed2)                           # seed for random orientations
            angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
            activity = []

            for i in range(0, len(id1)):
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
            hoomd.md.force.active(group=groupB,
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

    def fast_orient_penetrate_immobile_membrane(self):

        import random
        if self.hoomdPath == '/Users/nicklauersdorf/hoomd-blue/build/':
            sys.path.insert(0,self.hoomdPath)

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Initialize system
        hoomd.context.initialize()

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
            latNet = 1.4*self.theory_functs.conForRClust2(self.peA, self.peB, self.beta_A, self.beta_B, self.eps)

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
            rList = [ 0, Rs + latNet]
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
                        print('test')
                        print(len(pos))
                        print(self.partNumA)
                        #r = computeDistance(x, y)
                        # Check if x-position is large enough
                        if y <rMin: # <(rMin + (latNet / 2.)):
                            y += hor#latNet
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
                        print('test')
                        print(len(pos))
                        print(self.partNumA)
                        #r = computeDistance(x, y)
                        # Check if x-position is large enough
                        if x <rMin: # <(rMin + (latNet / 2.)):
                            x += hor#latNet
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

            print(NLiq)
            NGas = self.partNum - NLiq

            typ_A=0
            typ_B=0

            gas_B=self.partNumB-typ_B
            gas_A=self.partNumA-typ_A

            # Set this according to phiTotal
            areaParts = self.partNum * np.pi * (0.25)
            abox = (areaParts / self.phi)

            import utility

            utility_functs = utility.utility(lx, ly)

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

                if (lx <= ly) & (np.abs(gasy) <= (rList[-1] + (tooClose /2.))):
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
                            wrapX -= lx
                        if h == 2 and hlist[h] == 0:
                            wrapX += lx
                        if v == 0 and vlist[v] == -1:
                            wrapY -= ly
                        if v == 2 and vlist[v] == 0:
                            wrapY += ly
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

            typ_arr=np.array(typ)
            id0=np.where(typ_arr==0)[0]
            id1=np.where(typ_arr==1)[0]


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
            for i in range(0, len(id1)):
                pe.append(peList[1])

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

            groupA = hoomd.group.type('A')
            groupB = hoomd.group.type('B')

            groups = []
            for i in unique_char_types:
                groups.append(hoomd.group.type(type=i))

            # Set particle potentials
            nl = hoomd.md.nlist.cell()
            lj = hoomd.md.pair.lj(r_cut=self.r_cut, nlist=nl)
            lj.set_params(mode='shift')
            for i in range(0, len(unique_char_types)):
                for j in range(i, len(unique_char_types)):

                    #if (unique_char_types[j]=='B') and (unique_char_types[i]=='B'):
                    #    lj.pair_coeff.set(unique_char_types[i],
                    #                  unique_char_types[j],
                    #                  epsilon=0.0, sigma=self.sigma)
                    #else:
                    lj.pair_coeff.set(unique_char_types[i],
                                  unique_char_types[j],
                                  epsilon=self.eps, sigma=self.sigma)

            # Brownian integration
            brownEquil = 10000

            hoomd.md.integrate.mode_standard(dt=self.dt)
            bd = hoomd.md.integrate.brownian(group=groupB, kT=self.kT, seed=self.seed1)
            hoomd.run(brownEquil)

            # Set activity of each group
            np.random.seed(self.seed2)                           # seed for random orientations
            angle = np.random.rand(self.partNum) * 2 * np.pi     # random particle orientation
            activity = []
            """
            for i in range(0, len(id1)):
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
            """
            fx = []
            fy = []

            x_pos = np.array([x[0] for x in pos])
            y_pos = np.array([x[1] for x in pos])
            if lx <= ly:
                id_right = np.where((typ_arr==1) & (y_pos>0))[0]
                print(id_right)
                groupBright = hoomd.group.tag_list(name="B right", tags=id_right)
                hoomd.md.force.constant(fx=0,
                                    fy=-self.peB, fz=0,
                                    group=groupBright)
                id_left = np.where((typ_arr==1) & (y_pos<=0))[0]
                print(id_left)
                groupBleft = hoomd.group.tag_list(name="B left", tags=id_left)
                hoomd.md.force.constant(fx=0,
                                    fy=self.peB, fz=0,
                                    group=groupBleft)
            else:
                id_right = np.where((typ_arr==1) & (x_pos>0))[0]
                print(id_right)
                groupBright = hoomd.group.tag_list(name="B right", tags=id_right)
                hoomd.md.force.constant(fx=-self.peB,
                                    fy=0, fz=0,
                                    group=groupBright)
                id_left = np.where((typ_arr==1) & (x_pos<=0))[0]
                print(id_left)
                groupBleft = hoomd.group.tag_list(name="B left", tags=id_left)
                hoomd.md.force.constant(fx=self.peB,
                                    fy=0, fz=0,
                                    group=groupBleft)

            """
            for i in range(0, len(id1)):

                if lx <= ly:
                    if pos[i][1]>0:
                        fx.append(0)
                        fy.append(-pe[i])
                    else:
                        fx.append(0)
                        fy.append(pe[i])
                else:
                    if pos[i][0]>0:
                        fx.append(-pe[i])
                        fy.append(0)
                    else:
                        fx.append(pe[i])
                        fy.append(0)

                hoomd.md.force.constant(fx=fx,
                                    fy=fx,
                                    group=groupB)
            """
            # Implement the activities in hoomd
            #hoomd.md.force.active(group=groupB,
            #                      seed=self.seed3,
            #                      f_lst=activity,
            #                      rotation_diff=0,
            #                      orientation_link=False,
            #                      orientation_reverse_link=True)

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

    def fast_penetrate_stationary_membrane(self):

        import random

        import hoomd                    # import hoomd functions based on path
        from hoomd import md
        from hoomd import deprecated

        # Initialize system
        hoomd.context.initialize()

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
            groupA = hoomd.group.type(type='A')
            groupB = hoomd.group.type(type='B')
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
            #brownEquil = 10000

            hoomd.md.integrate.mode_standard(dt=self.dt)
            bd = hoomd.md.integrate.brownian(group=groupB, kT=self.kT, seed=self.seed1)
            #hoomd.run(brownEquil)

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
            out = "stationary_membrane_pa" + str(int(self.peA))
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
    def bullseye(self):

        # Initial imports

        eps = kT                                # repulsive depth
        tauLJ = computeTauLJ(eps)               # LJ time unit
        dt = 0.000001 * tauLJ                   # timestep
        dumpPerBrownian = 500.                  # number of dumps per 1 tauB
        simLength = 1.0 * tauBrown              # how long to run (in tauBrown)
        totTsteps = int(simLength / dt)         # how many tsteps to run
        numDumps = simLength * dumpPerBrownian  # total number of frames dumped
        dumpFreq = totTsteps / numDumps         # normalized dump frequency
        dumpFreq = int(dumpFreq)                # ensure this is an integer
        seed = 71996                            # a random seed
        seed2 = 2394                            # orientation seed
        seed3 = 183                             # activity seed

        # Some parameters (from command line):
        pe1 = float(sys.argv[1])        # fed in first, bullseye
        pe2 = float(sys.argv[2])        # second, second ring (and so on)
        pe3 = float(sys.argv[3])
        r1 = float(sys.argv[4])         # radius of first hcp phase
        w2 = float(sys.argv[5])
        w3 = float(sys.argv[6])
        rAlign = float(sys.argv[7])     # thickness of aligned layers
        peg = float(sys.argv[8])        # activity of gas phase
        phig = float(sys.argv[9])       # density of gas phase
            
        # List of activities
        peList = [ pe1, pe2, pe3 ]
        # List of ring radii
        rList = [ 0, r1, r1 + w2, r1 + w2 + w3 ]
        # List to store particle positions and types
        pos = []
        typ = []
        rOrient = []
        # z-value for simulation initialization
        z = 0.5

        for i in xrange(len(peList)):
            rMin = rList[i]             # starting distance for particle placement
            rMax = rList[i + 1]         # maximum distance for particle placement
            lat = computeLat(peList[i]) # activity-dependent lattice spacing
            ver = np.sqrt(0.75) * lat   # vertical shift between lattice rows
            hor = lat / 2.0             # horizontal shift between lattice rows
            
            x = 0
            y = 0
            shift = 0
            while y < rMax:
                r = computeDistance(x, y)
                # Check if x-position is large enough
                if r < rMin:
                    x += lat
                    continue
                    
                # Check if x-position is too large
                if r >= (rMax - (lat/2.)):
                    y += ver
                    shift += 1
                    if shift % 2:
                        x = hor
                    else:
                        x = 0
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
                x += lat

        # Get number of particle in gas phase
        peList.append(peg)
        lbox = 3. * rList[-1]
        hbox = lbox / 2.
        agas = (lbox**2) - (np.pi * (rList[-1]**2))
        ngas = (phig * agas) / (np.pi * (0.25))
        tooClose = 0.8

        # Make a mesh for random particle placement
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
        # Round up size of bins to account for floating point inaccuracy
        def roundUp(n, decimals=0):
            multiplier = 10 ** decimals
            return math.ceil(n * multiplier) / multiplier
        # Compute mesh
        r_cut = 2**(1./6.)
        nBins = (getNBins(lbox, r_cut))
        sizeBin = roundUp((lbox / nBins), 6)

        # Place particles in gas phase
        count = 0
        gaspos = []
        binParts = [[[] for b in range(nBins)] for a in range(nBins)]
        while count < ngas:
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
                rOrient.append(0)               # not oriented
                typ.append(len(peList) - 1)     # final particle type
                count += 1                      # increment count

        # Get each coordinate in a list
        pos = pos + gaspos
        x, y, z = zip(*pos)

        # Plot as scatter
        cs = np.divide(typ, float(len(peList)))
        cs = rOrient
        plt.scatter(x, y, s=2., c=cs, cmap='jet')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

        partNum = len(pos)
        # Get the number of types
        uniqueTyp = []
        for i in typ:
            if i not in uniqueTyp:
                uniqueTyp.append(i)
        # Get the number of each type
        particles = [ 0 for x in xrange(len(uniqueTyp)) ]
        for i in xrange(len(uniqueTyp)):
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

        # Now we make the system in hoomd
        hoomd.context.initialize()
        # A small shift to help with the periodic box
        snap = hoomd.data.make_snapshot(N = partNum,
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
        lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)
        lj.set_params(mode='shift')
        for i in xrange(len(unique_char_types)):
            for j in range(i, len(unique_char_types)):
                lj.pair_coeff.set(unique_char_types[i],
                                unique_char_types[j],
                                epsilon=eps, sigma=sigma)

        # Set activity of each group
        np.random.seed(seed2)                           # seed for random orientations
        angle = np.random.rand(partNum) * 2 * np.pi     # random particle orientation
        activity = []
        for i in xrange(partNum):
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
                            seed=seed3,
                            f_lst=activity,
                            rotation_diff=D_r,
                            orientation_link=False,
                            orientation_reverse_link=True)

        # Brownian integration
        hoomd.md.integrate.mode_standard(dt=dt)
        bd = hoomd.md.integrate.brownian(group=all, kT=kT, seed=seed)

        # Name the file from parameters
        out = "bullseye_pe"
        for i in peList:
            out += str(int(i))
            out += "_"
        out += "r"
        for i in range(1, len(rList)):
            out += str(int(rList[i]))
            out += "_"
        out += "rAlign_" + str(rAlign) + ".gsd"

        # Write dump
        hoomd.dump.gsd(out,
                    period=self.dumpFreq,
                    group=all,
                    overwrite=True,
                    phase=-1,
                    dynamic=['attribute', 'property', 'momentum'])

        # Run
        hoomd.run(self.totTsteps)