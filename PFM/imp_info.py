'''
Created on Feb 24, 2017

@author: lauersdorf
'''
class imp_info:
    def __init__(self):
        from numpy import array
        """
        Define all useful information about impurity species included in
        recombination spectrum calculation. May be a good place to store
        line emission information in the future. Currently, if the species
        of interest is not included in the list the user has to add the data.

        Run this program whenever a new impurity ion is added in the list.
        Data are saved in IDL file IMPURITY_INFO.SAV in the library.

        Outputs:
            IMP_I: Structure containing impurities info:
                IMP_I.ION_NAMES impurities (string)
                IMP_I.IONS      impurities (string) (lower case)
                IMP_I.N_IONS    number of available impurities
                IMP_I.(j)       structure with parameters about IMP_I.IONS[j]
                        Z          : Ionic charge
                        imp_chi    : Ionization energies for all charge states (eV)
                        imp_bound_e: Number of bound electrons for all charge states
                        chi        : Recombination energy for charge state of interest (eV)
                        bound_e    : Number of bound electrons in the lowest shell of the recombining ion.
                                    This can be partially filled
                        n_min      : Principal quantum number of the lowest shell where recombination can occur
        """

        #              0       1      2      3      4       5        6        7        8        9
        self.ion_names = ['H+1', 'D+1', 'He+2', 'B+5', 'C+6', 'N+7', 'O+8', 'Al+10', 'Al+11', 'Al+12', 'Al+13', 'Ar+18']
        self.ions = array(['h+1', 'd+1', 'he+2', 'b+5', 'c+6', 'n+7', 'o+8', 'al+10', 'al+11', 'al+12', 'al+13', 'ar+18'])


        #Ions with lowest shell partially filled
        self.part_ions = ['Al+10', 'Al+11', 'Al+12']
        
        #H+1
        
        ion_00 = [self.ion_names[0]]
        Z_00 = [1.0]
        imp_chi_00 = [13.598]
        imp_bound_e_00 = [0.0]
        chi_00 = [13.598]
        b_e_00 = [0.0]
        n_min_00 = [1.0]
        
        #D+1
        
        ion_01 = [self.ion_names[1]]
        Z_01 = [1.0]
        imp_chi_01 = [13.602]
        imp_bound_e_01 = [1.0]
        chi_01 = [13.602]
        b_e_01 = [0.0]
        n_min_01 = [1.0]
        #He+2
    
        ion_02 = [self.ion_names[2]]
        Z_02 = [2.0]
        imp_chi_02 =  [24.58741, 54.41778]
        imp_bound_e_02 = [ 1.0, 0.0]
        chi_02 = [54.51778]
        b_e_02 = [0.0]
        n_min_02 = [1.0]

        #B+5
    
        ion_03 = [self.ion_names[3]]
        Z_03 = [5.0]
        imp_chi_03 = [8.29803, 25.15484, 37.93064, 259.37521, 340.22580]
        imp_bound_e_03 = [ 0.0, 1.0, 0.0, 1.0, 0.0]
        chi_03 = [340.22580]
        b_e_03 = [0.0]
        n_min_03 = [1.0]
            
        #C+6
    
        ion_04 = [self.ion_names[4]]
        Z_04 = [6.0]
        imp_chi_04 = [11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334]
        imp_bound_e_04 = [ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        chi_04 = [489.99334]
        b_e_04 = [0.0]
        n_min_04 = [1.0]
            
        #N+7
    
        ion_05 = [self.ion_names[5]]
        Z_05 = [7.0]
        imp_chi_05 = [14.53414, 29.6013, 47.44924, 77.4735, 97.8902, 552.0718, 667.046]
        imp_bound_e_05 = [2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        chi_05 = [667.046]
        b_e_05 = [0.0]
        n_min_05 = [1.0]
            
        #O+8

        ion_06 = [self.ion_names[6]]
        Z_06 = [8.0]
        imp_chi_06 = [13.61806, 35.11730, 54.9355, 77.41353, 113.8990, 138.1197, 739.29, 871.4101]
        imp_bound_e_06 = [3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        chi_06 = [871.4101]
        b_e_06 = [0.0]
        n_min_06 = [1.0]
            
        #Al+10
    
        ion_07 = [self.ion_names[7]]
        Z_07 = [10.0]
        imp_chi_07 = [5.98577, 18.82856, 28.44765, 119.992, 153.825, 190.49, 241.76, 284.66, 330.13, 398.75, 442.005, 2085.977, 2304.1410 ]
        imp_bound_e_07 = [2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        chi_07 = [398.75]
        b_e_07 = [1.0]
        n_min_07 = [2.0]
            
        #Al+11
    
        ion_08 = [self.ion_names[8]]
        Z_08 = [11.0]
        imp_chi_08 = [ 5.98577, 18.82856, 28.44765, 119.992, 153.825, 190.49, 241.76, 284.66, 330.13, 398.75, 442.005, 2085.977, 2304.1410 ]
        imp_bound_e_08 = [ 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 ]
        chi_08 = [442.005]
        b_e_08 = [0.0]
        n_min_08 = [2.0]
                
        #Al+12

        ion_09 = [self.ion_names[9]]
        Z_09 = [12.0]
        imp_chi_09 = [ 5.98577, 18.82856, 28.44765, 119.992, 153.825, 190.49, 241.76, 284.66, 330.13, 398.75, 442.005, 2085.977, 2304.1410 ]
        imp_bound_e_09 = [ 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 ]
        chi_09 = [2085.977]
        b_e_09 = [1.0]
        n_min_09 = [1.0]
            
        #Al+13
    
        ion_10 = [self.ion_names[10]]
        Z_10 = [13.0]
        imp_chi_10 = [ 5.98577, 18.82856, 28.44765, 119.992, 153.825, 190.49, 241.76, 284.66, 330.13, 398.75, 442.005, 2085.977, 2304.1410 ]
        imp_bound_e_10 = [ 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 ]
        chi_10 = [2304.1410]
        b_e_10 = [0.0]
        n_min_10 = [1.0]
            
        #Ar+18

        ion_11 = [self.ion_names[11]]
        Z_11 = [18.0]
        imp_chi_11 = [ 15.75962, 27.62967, 40.74, 59.81, 75.02, 91.009, 124.323, 143.460, 422.45, 478.69, 538.96, 618.26, 686.10, 755.74, 854.77, 918.03, 4120.8857, 4426.2296 ]
        imp_bound_e_11 = [ 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 ]
        chi_11 = [4426.2296]
        b_e_11 = [0.0]
        n_min_11 = [1.0]

        self.Z = array([Z_00, Z_01, Z_02, Z_03, Z_04, Z_05, Z_06, Z_07, Z_08, Z_09, Z_10, Z_11])
        self.imp_chi = array([imp_chi_00, imp_chi_01, imp_chi_02, imp_chi_03, imp_chi_04, imp_chi_05, imp_chi_06, imp_chi_07, imp_chi_08, imp_chi_09, imp_chi_10, imp_chi_11])
        self.imp_bound_e = array([imp_bound_e_00, imp_bound_e_01, imp_bound_e_02, imp_bound_e_03, imp_bound_e_04, imp_bound_e_05, imp_bound_e_06, imp_bound_e_07, imp_bound_e_08, imp_bound_e_09, imp_bound_e_10, imp_bound_e_11])
        self.chi = array([chi_00, chi_01, chi_02, chi_03, chi_04, chi_05, chi_06, chi_07, chi_08, chi_09, chi_10, chi_11])
        self.b_e = array([b_e_00, b_e_01, b_e_02, b_e_03, b_e_04, b_e_05, b_e_06, b_e_07, b_e_08, b_e_09, b_e_10, b_e_11])
        self.n_min = array([n_min_00, n_min_01, n_min_02, n_min_03, n_min_04, n_min_05, n_min_06, n_min_07, n_min_08, n_min_09, n_min_10, n_min_11])


