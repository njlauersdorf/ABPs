import sys, os
from gsd import hoomd
import freud, numpy as np, math
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

# Append '~/klotsa/ABPs/post_proc/lib' to path to get other module's functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory, utility

# Class of spatial binning functions
class binning:

    def __init__(self, lx_box, ly_box, partNum, NBins_x, NBins_y, peA, peB, typ, eps):
        self.theory_functs = theory.theory()

        # Total x-length of box
        self.lx_box = lx_box
        self.hx_box = self.lx_box / 2
        
        # Total y-length of box
        self.ly_box = ly_box
        self.hy_box = self.ly_box / 2

        # Number of particles
        self.partNum = partNum

        # Minimum cluster size
        self.min_size = int(self.partNum / 8)

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
        self.sizeBin_x = self.utility_functs.roundUp(self.lx_box / self.NBins_x, 6)

        # Y-length of bin
        self.sizeBin_y = self.utility_functs.roundUp(self.ly_box / self.NBins_y, 6)

        # Cut off radius of WCA potential
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)

        # Magnitude of WCA potential (softness)
        self.eps = eps

        # A type particle activity
        self.peA = peA

        # B type particle activity
        self.peB = peB

        # Array (partNum) of particle types
        self.typ = typ

    def kmeans_clustering(self)

        from numpy import unique
        from numpy import where
        from matplotlib import pyplot
        from sklearn.datasets import make_classification
        from sklearn.cluster import KMeans

        # initialize the data set we'll work with
        training_data, _ = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=4
        )

        # define the model
        kmeans_model = KMeans(n_clusters=2)

        # assign each data point to a cluster
        dbscan_result = dbscan_model.fit_predict(training_data)

        # get all of the unique clusters
        dbscan_clusters = unique(dbscan_result)

        # plot the DBSCAN clusters
        for dbscan_cluster in dbscan_clusters:
            # get data points that fall in this cluster
            index = where(dbscan_result == dbscan_clusters)
            # make the plot
            pyplot.scatter(training_data[index, 0], training_data[index, 1])

        # show the DBSCAN plot
        pyplot.show()