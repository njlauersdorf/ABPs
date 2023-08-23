'''
Created on Feb 16, 2017

@author: lauersdorf
'''
#import sys
#import scipy.io
#import os
#script_dir = os.path.dirname(__file__)
#rel_file_path = os.path.join(script_dir, 'SAV')
#sys.path.append(rel_file_path)
'''
                    time1 = datetime.datetime.now()

                    time2 = datetime.datetime.now()
                    time3 = time2 - time1
                    print time3
                    '''
class sxr_x_emission:
    """
    Create a class object of the soft x-ray (sxr) emission model with a given 
    set of parameters (temperature, density, shift, etc) and detector dimensions
    (Si thickness, Be thickness). Te, density, or impurity structures, impurity densities can 
    also be added to simulate the SXR spectrum. 
  
  
    DEFAULT INSTANCES  
    ----------------

    TEMPERATURE
    te = 1250.0 eV
    alpha_te = 4
    beta_te = 1
    r0 = 0.0 m (always normalized)
    th0 = 0.0 
    del_r = 0.0 m (always normalized)
    del_th = 0.0 
    del_te = 0.0 eV
    
    DENSITY
    dens = 2.5e19 m**-3
    alpha_ne = 10.0
    beta_ne = 1.0
    ne_r0 = 0.0 m (always normalized)
    ne_th0 = 0.0
    ne_del_r =0.0 m (always normalized)
    ne_del_th = 0.0 
    del_ne = 0.0 m**-3
    del_dens = 0.0 m**-3
    del_dr = 0.0 m (always normalized)
    dr = 0.0 m (always normalized)
    
    SHIFT
    delta_h = 0.0 (same units as radius)
    delta_a = 0.0 (same units as radius)
    nm = 0.0 (%, 1 being 100%)
    en0 = 0.0eV
    
    FILTERS
    be_thick = ndarray(80): 857 micron for diodes 1-40, 421 micron for diodes 
        41-80. These default values are stored in the python file sxr_lib.py
    si_thick = ndarray(80): 35.0 micron / incident angle for each sxr diode 
        on SXR-A, SXR-B, SXR-C, and SXR-D on MST. The 80 default incident 
        angles are stored in the python file sxr_lib.py.
    tr_si = ndarray(80,n_en): Si transmission function for each diode, at each
        energy in energy integral
    tr_be = ndarray(80,n_en): Be transmission function for each diode, at each
        energy in energy integral

    LINE INTEGRAL
    n_en = 501: default number of energy points in energy integral.
    en = A linear spacing of n_en points from 500eV - 10keV. 
    delta_en = The difference between each point in the 'en' vector.
    n_s = Number of spatial points along each line of sight (default = 25)
    p = ndarray(80):p impact parameters for each sxr viewing chord on MST.
        The array of 80 default values is stored in the python file sxr_lib.py
            
    phi = ndarray(80): phi impact parameters for each sxr viewing chord on MST.
        The array of 80 default values is stored in the python file sxr_lib.py
    radius = 1
    
    rhoy = ndarray(80,n_s): rho point coordinates for each of the n_s spatial 
        integration points of each of the 80 viewing chords on MST
    tze = ndarray(80,n_s): Angle tzeta in rho coordinates for each of the 
        n_s spatial integration points for each of the 80 viewing chords on MST
    
    IMPURITIES
    ni_dens = 0.0
    alpha_ni = 0.0
    beta_ni = 0.0
    ni_dr = 0.0
    del_ni_dr = 0.0
    del_ni_dens = 0.0

    TABV
    tabv_te = None
    tabv_r_te = None
    
    tabv_dens = None
    tabv_r_dens = None
    
    tabv_ni_dens = None
    tabv_r_ni_dens = None
    
    PROFILES
    The following temperature, density, and impurity profiles are calculated 
    and stored as objects in the sxr_x_emission instance to speed up later 
    calculations.
    
    tep = ndarray(80,n_s) of the temperature profile at each spatial location 
        for each line of sight
    nep = ndarray(80,n_s) of the density profile at each spatial location for
        each line of sight
    nip = ndarray(80,n_s, number of impurities) of each impurity density at each
        spatial location for each line of sight.
    
    OPTIONAL INPUT PARAMETERS
    ---------------------------
    te: core electron temperature (eV; default=1250eV; >1)
    alpha_te: exponent of the Te profile (default=4.0; >1)
        Te(r) = Te * ( 1 - (r/a)^alpha_te )^beta_te, a=minor radius=RADIUS
    beta_te: exponent of the Te profile (default=1.0)
        Te(r) = Te * ( 1 - (r/a)^alpha_te )^beta_te, a=minor radius=RADIUS
    tabv_te: array with a Te profile to be used in the simulation,
        instead of the one defined with the two exponents alpha_te and
        beta_te. If set to 0 then these values are neglected. The array
        is saved as an attribute, tabv_te.
        (eV; default=0) 
    tabv_r_te:array of the radius of the Te profile tabv_te. It is saved as an
        attribute (geometrical coordinates. same units as radius)
    dens: core electron density (1e19 m-3; default=2.5e19m-3; >1)
    alpha_ne exponent of the density profile (default=10; >1)
        Dens(r) = Dens * ( 1 - (r/a)^Esp_Dens )^Esp_b_Dens, a=radius (minor)
    beta_ne: exponent of the density profile (default=1)
        Dens(r) = Dens * ( 1 - (r/a)^Esp_Dens )^Esp_b_Dens, a=radius (minor)
    tabv_dens: array with a density profile to be used in the simulation,
        instead of the one defined with the exponent alpha_ne. If set to 0, then
        these values are not used. 
    tabv_r_dens: array of the radius of the density profile tabv_dens.
        (geometrical coordinates, same units of radius; default=0)
    delta_a: magnetic axis shift (same units of radius; default=0)
    delta_h: LCFS shift (same units of radius; default=0)
    be_thick: array of Be thickness (microns; must be same size as si_thick,
        p, and phi) (default = in 'sxr_lib.py')
    si_thick: array of Si thickness of the detector (microns; must be same size 
        as be_thick, p, and phi) (default = in 'sxr_lib.py')
    n_en: number of energies for line integral calculations (default=501; >1)
        The value changes in the execution of the program as more points are 
        added for resolution at recombination step energies, and at a 
        discontinuity in the silicon transmission at 1839eV.
    n_s: number of points along the line of sight for line integral
        calculations (default=25; >1)
    p: array of impact parameters of the line of sight (same units of radius; 
        same size as phi, be_thick, and si_thick, default=in 'sxr_lib.py')
    phi: array of impact angles of the line of sight  (rad; must be the same 
        size as p, be_thick, and si_thick, default=in 'sxr_lib.py')
    radius: minor radius (default=1; >0)
    r0: radius of the centre of the Te island. Can be an array, if more
        than one Te island must be defined (normalized; 0-1; default=0)
    th0: poloidal angle of the centre of the Te island. Can be an array,
        if more than one Te island must be defined (rad; 0-2pi; default=0 rad)
    del_r: radial width of the Te structure. Can be an array, if more
        than one Te island must be defined (normalized; 0-1; default=0)
    del_th: poloidal width of the Te structure. Can be an array, if more
        than one Te island must be defined (rad; 0-2pi; default=0). A value
        of 0 define the island as a gaussian centered in r0,th0, a value
        different than 0 define the islands as the product of a rho-gaussian
        and a tzeta-gaussian
    del_te: increase of Te in the centre of the island. Can be an array,
        if more than one Te island must be defined (eV; default=0, which
        means that no islands are present)
    ne_r0: radius of the centre of the ne island. Can be an array, if more
        than one ne island must be defined (normalized; 0-1; default=0)
    ne_th0: poloidal angle of the centre of the ne island. Can be an array,
        if more than one ne island must be defined (rad; 0-2pi; default=0 rad)
    ne_del_r: radial width of the ne structure. Can be an array, if more
        than one ne island must be defined (normalized; 0-1; default=0)
    ne_del_th: poloidal width of the ne structure. Can be an array, if more
        than one ne island must be defined (rad; 0-2pi; default=0). A value
        of 0 define the island as a gaussian centered in ne_r0,ne_th0, a value
        different than 0 define the islands as the product of a rho-gaussian
        and a tzeta-gaussian
    del_ne: increase of ne in the centre of the island. Can be an array,
        if more than one ne island must be defined (1e19 m-3; default=0, which
        means that no islands are present)
    del_dens: increase of density in the circular sector (1e19 m-3; default=0,
        which means no increase in density)
    del_dr: radial width of the density circular sector (normalized; 0-1;
        default=0)
    dr: radius of the centre of the circular increase in density
        (normalized; 0-1; default=0)
    en_step: step recombination energy. Can be an array, if more than one
        step must be defined (eV; default=0)
    alpha_ni: array of exponents of the ni profile (default=0; >1)
        Te(r) = Te * ( 1 - (r/a)^alpha_ni )^beta_ni, a=minor radius=radius
    beta_ni: array of exponents of the ni profile (default=0; >1)
        Te(r) = Te * ( 1 - (r/a)^alpha_ni )^beta_ni, a=minor radius=radius
    ni_dr: array of radii of the centre of the circular increase in impurity density
        (normalized; 0-1; default=0)
    del_ni_dr: array of radial width of density circular sectors (default =0.0)
    del_ni_dens: array of increase in impurity density at center of circular
        ring. (10**19 m**-3, default=0)    
    ni_dens: array of core densities (10**19 m**-3) of each impurity species
    is_del_ni: increase of impurity density in the center of the island. Can be 
        an array, if more than one impurity island must be defined (1e19 m-3;
        default=0, which means that no islands are present)
    is_ni_dr: radius of the centre of the ni island. Can be an array, if more
        than one ni island must be defined (normalized; 0-1; default=0)
    is_dr: radial width of the ni structure. Can be an array, if more
        than one ne island must be defined (normalized; 0-1; default=0)
    ni_del_th: poloidal width of the ni structure. Can be an array, if more
        than one ne island must be defined (rad; 0-2pi; default=0). A value
        of 0 define the island as a gaussian centered in is_ni_dr, ni_th_dr, 
        a value different than 0 define the islands as the product of a 
        rho-gaussian and a tzeta-gaussian
    ni_th_dr: poloidal angle of the centre of the ni island. Can be an array,
        if more than one ni island must be defined (rad; 0-2pi; default=0 rad)
    en0: energy (eV) where a non-maxwellian tail is added to the xray
        spectrum (default=0)
    nm: increase of the non-maxwellian tail of the spectrum, 1 being 100%
        (default=0)
    
    Additional options:
        
    
    CALLING SEQUENCE
    ------------------
    st = sxr_x_emission(
        te=te, alpha_te=alpha_te, beta_te=beta_te, 
        r0=r0, th0=th0, del_r=del_r, del_th=del_th, del_te=del_te,
        dens=dens, alpha_ne=alpha_ne, beta_ne=beta_ne,
        ne_r0=ne_r0, ne_del_r=ne_del_r, ne_th0=ne_th0, ne_del_th=ne_del_th, del_ne=del_ne,
        del_dens=del_dens, del_dr=del_dr, dr=dr,  
        ni_dens=ni_dens, alpha_ni=alpha_ni, beta_ni=beta_ni, 
        ni_dr=ni_dr, del_ni_dr=del_ni_dr, del_ni_dens=del_ni_dens,
        is_del_ni=is_del_ni, is_ni_dr=is_ni_dr, is_dr=is_dr, ni_th_dr=ni_th_dr, ni_del_th=ni_del_th,
        imp_str=imp_str, n_imp=n_imp, 
        delta_a=delta_a, delta_h=delta_h,
        radius=radius,      
        si_thick = si_thick, be_thick = be_thick,   
        p=p, phi=phi
        )
    
    RESTRICTIONS
    -------------
    All values are single valued unless specified otherwise, in which case they 
    can be either an array or single valued (p, phi, be_thick, si_thick, any
    island structures). If more values are passed to single value parameters
    then only the first value will be used. 
    The arrays p, phi, be_thick, and si_thick must all be the same length. 
    
    An instance of the sxr_x_emission class will always need to be initialized 
    (e.g. st = sxr_x_emission(te=te, alpha_te=alpha_te, ...) before the line
    integral can be calculated with st.sxr_line_integral() )
    
    The module 'sxr_lib.py' must be installed in order to use this class.
    
    
    POSSIBLE FUTURE ADDITIONS
    --------------------------
    
    
    MODIFICATION HISTORY
    ---------------------
    July 2014 - First written by J. Johnson
    June 2016 - Updated and revised by N. Lauersdorf
       
    """

    
    def sXe_N(self, **kwargs):
        from numpy import zeros, array
        
        par = kwargs.get('par')
        par_name = kwargs.get('par_name')
        mess = ' (SXR_X_EMISSION) check impurity parameter ' + par_name.upper()\
        + ': ' + 'number of ions and number of elements not the same'
        a = kwargs.get('a', [])
        
        if not isinstance(a, float or int):
            
            n_a = len(a)

        else: 
            n_a = 1
            
        if self.n_imp == 0:
            
            if n_a != 0:
                b = float(a)
                
        else:
            if n_a == 0:
                
                n_par = len(par)
                
                if n_par == 1:
                    b = zeros(self.n_imp) + par
                    
                if n_par < self.n_imp:
                    raise ValueError( mess)
                
                else:
                    b = par[0:self.n_imp]
                    
            elif n_a == 1:
                b = zeros(self.n_imp) + a
                
            else:
                if n_a < self.n_imp:
                    raise ValueError( mess)
                
                b = a[0:self.n_imp]
                
        return array(b)

    def __init__(self, **kwargs ):
        # Import all necessary modules 
        from sxr_lib import findvalue, findvalue2, vect_0_1, sxrGeometry, getFirstValue, \
            exclude, add_float, np_array, keyword_set, idl_where, Save, Load
        from numpy import array, sqrt, where, zeros, cos, transpose, ones,\
            maximum, sin, exp, \
            log, interp, squeeze, pi, size, newaxis, einsum
        from time import time
        from Profiles import sxr_prof_te, sxr_prof_ne
        from imp_info import imp_info
        import timeit
        import datetime

        geometry = sxrGeometry()
        self.geometry_en_mu = geometry.en_mu
        #self.geometry_en = geometry.en
        self.geometry_compound = geometry.compound
        self.geometry_perc_comp_BeFeb2015 = geometry.perc_comp_BeFeb2015
        self.geometry_rho_comp = geometry.rho_comp
        self.geometry_aw = geometry.aw
        self.geometry_impact_p = geometry.impact_p
        self.geometry_impact_phi = geometry.impact_phi
        self.geometry_bethickness = geometry.bethickness
        self.geometry_angle_fact = geometry.angle_fact
        self.geometry_mu_BeFeb2015 = geometry.mu_BeFeb2015
        self.geometry_Bessy_T = geometry.Bessy_T
        self.geometry_Bessy_E = geometry.Bessy_E
        self.geometry_mu = geometry.mu
        imp = imp_info()
            
        # Import the geometric parameters from the sxrGeometry class in sxr_lib
        # this object contains the following data for 80 SXR diodes
        #     geometry.impact_p -- impact parameters 
        #     geometry.impact_phi -- impact angles 
        #     geometry.angle_fact -- x-ray incident angle on silicon diodes
        #     geometry.bethickness -- thickness of the beryllium filters  
        # The number of points for the energy integral 
        #     n_en - will change when resolution is added for energy steps and 
        #         radiation lines
        #     n_en_no_rl - does not change and is referenced later if the 
        #         radiation lines are later changed
        '''
        all_kwargs = ['n_s', 'n_corde', 'radius', 'p', 'phi', 'hor', 'te', 'alpha_te', 'beta_te', 'dens', 'alpha_ne', 'beta_ne', 'neutral_dens', 'Born', 'delta_a', 'delta_h', 'en0', 'nm', 'be_thick', 'si_thick', 'perc_comp', 'r0', 'th0', 'del_r', 'del_th', 'del_te', 'ne_r0', 'ne_th0', 'ne_del_r', 'ne_del_th', 'del_ne', 'del_dens', 'del_dr', 'dr', 'ni_dens', 'alpha_ni', 'beta_ni', 'ni_dr', 'del_ni_dr', 'del_ni_dens', 'is_del_ni', 'is_ni_dr', 'is_dr', 'ni_th_dr', 'ni_del_th', 'ni_ADAS', 'imp_str']
        '''
        if 'recycle' in kwargs:
            calc = Load()

        en0 = log(500)
        en1 = log(20000)
        
        self.en = self.geometry_en_mu

        self.n_en_mu = self.geometry_en_mu.size
        self.n_en = self.n_en_mu
        en_list = self.en.tolist()
        
        delta_en_1 = en_list[1:]
        n_en_1 = squeeze(self.n_en_mu)
        delta_en_2 = en_list[0:n_en_1-1]
        delta_en = array(delta_en_1) - array(delta_en_2)
        delta_en = delta_en.tolist()
        delta_en.append(0.0) #last element is 0 to calculate correctly the energy integral, 
        #18-Mar-2014
        self.delta_en = array(delta_en)

        self.n_qs = 5L
        
        self.n_s = 25L

        self.n_en_no_rl        = self.n_en_mu   

        self.spBe = array([ 25.0, 0.0]) #[total thickness, thickness of second foil] microns

        self.p_thick = self.spBe * long(6.305e7)
        
        self.compound = kwargs.get('Compound', self.geometry_compound)
        
        self.ind_Be = where(self.compound == 'Be')

        self.ind_noBe = where(self.compound != 'Be')

        self.n_comp = self.compound.size
        
        #PERC_COMP has been defined using the proper Be foil impurities fractional
        #abundance for Cr, Fe, Ni from Fournelle (microprobe analysis of July 2014),
        #Materion datasheet for everything else (IF-1 foils) and lower-limits
        #of Pb, Mo (Se, W, U=0).  Only first row is defined; second one has zeroes.
        #This refers to the foils used in 2010-2014, and these are the famous Be-Zr filters
        #From 5-Feb-2015 new Be foils are used, with a different composition: no Zr,
        #Fe and Ni within the maximum limit of IF-1.  THIS IS THE DEFAULT
        #
        #[ [compound of the first foil], [compounds of the second foil] ]
        
        #perc_comp_def = [ [ perc_comp_BeZr80 ], [ perc_comp_Be40 * 0.0 ] ]
        #default 2010-2014
        self.perc_comp = array( [self.geometry_perc_comp_BeFeb2015, zeros(self.geometry_perc_comp_BeFeb2015.shape)] ) #default 2015-

        self.ion_names = imp.ion_names
        self.ions = imp.ions
        self.part_ions = imp.part_ions
        self.imp_bound_e = imp.imp_bound_e
        self.imp_chi = imp.imp_chi
        Z = imp.Z
        chi = imp.chi
        b_e = imp.b_e
        n_min = imp.n_min
        
        ind_maj = where(array(self.ion_names) == 'D+1')[0]

        if len(self.imp_bound_e[ind_maj][0]) >= len(self.imp_chi[ind_maj][0]):
            length = len(self.imp_bound_e[ind_maj][0])
        else:
            length = len(self.imp_chi[ind_maj][0])
        self.maj = {'ion_names': self.ion_names[ind_maj[0]], 'Z': Z[ind_maj[0]], 'imp_chi': self.imp_chi[ind_maj[0]], 'imp_bound_e': self.imp_bound_e[ind_maj[0]], 'chi': chi[ind_maj[0]], 'b_e': b_e[ind_maj[0]], 'n_min': n_min[ind_maj[0]]}

        #CONSTANT VALUES
        self.eff_density = 1.675
        self.nominal_Be_density = 1.848
        self.rho_comp = kwargs.get('rho_comp', self.geometry_rho_comp)
        self.AW = kwargs.get('AW', self.geometry_aw)

        self.m_e = 9.10938291e-31 # kg
        self.m_i = 1.672621777e-27 # kg, assume hydrogen
        self.q = 1.602176565e-19 # C
        self.ep0 = 8.854187817e-12 # F/m
        self.c = 2.99792458e8 # m/s
        self.h = 6.626070040e-34 # Js
        self.Ry = 13.60569253 # eV
        
        self.aa = ( 1.0 / (4.0 * pi ) ) * ( self.q ** 2.0 / ( 4.0 * pi * self.ep0 ) ) ** 3.0 *\
        ( ( 32.0 * pi ** 2 ) / ( 3.0 * sqrt(3.0) * self.m_e ** 2.0 * self.c ** 3.0 ) ) *\
        sqrt( 2.0 * self.m_e / pi ) * ( 1.0 / sqrt(self.q) ) * ( 1.0 / self.h ) *\
        ( 1.0e19 ) ** 2.0

        # transmission function of the Be foil

        self.be_thick = array(self.spBe)
        self.si_thick =  100    

        sxr_Be = self.sxr_Be(self.en, self.be_thick, self.perc_comp, use_mu = 1, use_BESSY = 0, eff_mu = 0, old_foils = 0, eff_filt_dens = 0, DSX3_filt_dens = 0)
        self.T_Be = sxr_Be['T']
        
        self.perc_comp = array( [self.geometry_perc_comp_BeFeb2015, zeros(self.geometry_perc_comp_BeFeb2015.shape)] ) #default 2015-
        
        
        spt = sxr_Be['Tt']


        # response function of the detector

        self.tr_si = self.sxr_tr_Si(self.si_thick, use_mu = 1)

        self.a_si = 1.0 - self.tr_si
        
        self.sit = transpose(log( self.a_si )) 

        ### Line_integ quantities ###

        if 'n_en' in kwargs:
            if keyword_set(self.n_en):
                
                n_en = getFirstValue(kwargs, 'n_en', 501)

                en = exp( vect_0_1(n_en) * (en1 - en0) + en0) # 13-Mar-2014
                delta_en_1 = en[1:]
                delta_en_2 = en[0:n_en-1]
                delta_en = delta_en_1 - delta_en_2
                delta_en = delta_en.tolist()
                delta_en.append(0.0) #last element is 0 to calculate correctly the energy integral, 
                #18-Mar-2014
                delta_en = array(delta_en)

                en_changed = False
                
                if n_en != self.n_en:
                    en_changed = True
                else:
                    v = sum(abs(self.en - self.geometry_en_mu))
                    if v > 1.0:
                        en_changed = True
                    else:
                        en_changed = False
            
                if en_changed:
                    
                    self.n_en = n_en
                    self.en = en
                    self.delta_en = delta_en
        else:
            en_changed = False
        if 'n_s' in kwargs:
            self.n_s_original = getFirstValue(kwargs, 'n_s', 25L)
        else:
            if 'recycle' in kwargs:
                self.n_s_original = calc['n_s']
            else:
                self.n_s_original = array([25])
        self.n_s = self.n_s_original

        
        if 'n_corde' in kwargs:
            self.n_corde_original = getFirstValue(kwargs, 'n_corde', 1L)
        else:
            if 'recycle' in kwargs:
                self.n_corde_original = calc['n_corde']
            else:
                self.n_corde_original = array([1])
        self.n_corde = self.n_corde_original

        
        if 'radius' in kwargs:
            self.radius_original = getFirstValue(kwargs, 'radius', 1.0)
        else:
            if 'recycle' in kwargs:
                self.radius_original = calc['radius']
            else:
                self.radius_original = array([1.0])
        self.radius = self.radius_original

        
        if 'p' in kwargs:
            self.p_original                 = np_array(kwargs.get('p', self.geometry_impact_p))
        else:
            if 'recycle' in kwargs:
                self.p_original = calc['p']
            else:
                self.p_original = np_array(self.geometry_impact_p)
        self.p = self.p_original

        
        imp_str_changed = False
        
        #Normalizes the impact parameter P to the RADIUS: the line integral
        #is ALWAYS calculated in normalized coordinates and FUNC is always
        #a rho, tzeta function with rho in [0,1]

        if self.radius != 1:
            
            self.p = self.p / self.radius
            self.p_divided_radius = True
        else:
            self.p_divided_radius = False
        
        if 'phi' in kwargs:
            self.phi_original                 = np_array(kwargs.get('phi', self.geometry_impact_phi))
        else:
            if 'recycle' in kwargs:
                self.phi_original = calc['phi']
            else:
                self.phi_original = np_array(self.geometry_impact_phi)
        self.phi = self.phi_original


        # Factor to take into account the effective thickness of the filter
        # (flat filters)
        if 'hor' in kwargs:
            self.hor_original                 = kwargs.get('hor', 0)
        else:
            if 'recycle' in kwargs:
                self.hor_original = calc['hor']
            else:
                self.hor_original = 0
        self.hor = self.hor_original

        if self.hor == 1:
            self.f_be = abs(sin(self.phi))
        else:
            self.f_be = abs(cos(self.phi))

        # Check the number of p and phi parameters
        
        if self.p.size != self.phi.size:
            raise ValueError('Must have the same number of p impact parameters'
                                ' as phi impact parameters')
        self.n_p               = self.p.size

        ### ENERGY STEPS and ENERGY INTEGRAL OBSOLETE!!!! ###
        
        ### EMISSION ###     
        if 'te' in kwargs:
            self.te_original                 = getFirstValue(kwargs,'te', 1250.0)   
        else:
            if 'recycle' in kwargs:
                self.te_original = calc['te']
            else:
                self.te_original = array([1250])
        self.te = self.te_original

        if 'alpha_te' in kwargs:
            self.alpha_te_original                 = getFirstValue(kwargs,'alpha_te', 4.0)   
        else:
            if 'recycle' in kwargs:
                self.alpha_te_original = calc['alpha_te']
            else:
                self.alpha_te_original = array([4.0])
        self.alpha_te = self.alpha_te_original

        if 'beta_te' in kwargs:
            self.beta_te_original                 = getFirstValue(kwargs,'beta_te', 1.0)   
        else:
            if 'recycle' in kwargs:
                self.beta_te_original = calc['beta_te']
            else:
                self.beta_te_original = array([1.0])
        self.beta_te = self.beta_te_original
        
        if 'dens' in kwargs:
            self.dens_original                 = getFirstValue(kwargs,'dens', 2.5)   
        else:
            if 'recycle' in kwargs:
                self.dens_original = calc['dens']
            else:
                self.dens_original = array([2.5])
        self.dens = self.dens_original

        if 'alpha_ne' in kwargs:
            self.alpha_ne_original                 = getFirstValue(kwargs,'alpha_ne', 10.0)   
        else:
            if 'recycle' in kwargs:
                self.alpha_ne_original = calc['alpha_ne']
            else:
                self.alpha_ne_original = array([10.0])
        self.alpha_ne = self.alpha_ne_original

        if 'beta_ne' in kwargs:
            self.beta_ne_original                 = getFirstValue(kwargs,'beta_ne', 1.0)   
        else:
            if 'recycle' in kwargs:
                self.beta_ne_original = calc['beta_ne']
            else:
                self.beta_ne_original = array([1.0])
        self.beta_ne = self.beta_ne_original

        #neutral density
        #
        #I looked in Scott Eilerman's thesis - he did a little work with the neutral density profiles
        #as well. There was a plot of the neutral density profiles for PPCD on a log scale. The profiles
        #ranged between 1e10-1e12 cm^-3, so it appears to be consistent with the plot in Barbui's paper.
        #I also found a personal copy of Mark's charge state fraction where I had changed the range of
        #neutral densities to 1e9-1e12 cm^-3. I'm guessing this was the range I decided was appropriate
        #for PPCD.
        #
        #It looks like as the neutral density increases, we have more flexibility in the precision that
        #we need to get the charge state fraction to change less than 5%. In general the neutral density
        #should be defined with an error no less than 5-10%. Does this trend still hold at higher
        #neutral densities (say 1e12 cm^-3)?
        #
        #After having a lengthy conversation with Mark about neutral pressure, he convinced me that
        #the best place to start was with a flat profile of 1-3e14 m^-3. His argument is that it's pretty
        #flat in the core and mostly rises towards the edge of the machine. It tends to be increasing
        #rapidly when Te and ne are decreasing, leading to 1) less overall impurity emission from the
        #plasma and 2) the charge states Al+11 and higher not being populated. (Basically we're not
        #sensitive to where the neutral density is changing rapidly - we're sensitive where the profile
        #is mostly flat.) He suggested maybe testing 1e14, 2e14 and 3e14 flat profiles to see what effect
        #this has on the modeled SXR signal. I think this sounds reasonable, and I think those densities
        #are in the range of values in the table you have.
        #Nov-2016: tried a peaked profile of the neutral density at the edge (for rho>0.8). The effect
        #on the impurity density profiles is small, and the brightnesses changes of a few % at the edge,
        #where the values are already small. In the core there is no measurable effect
        if 'neutral_dens' in kwargs:
            self.neutral_dens_original                 = getFirstValue(kwargs,'neutral_dens', 1.0)   
        else:
            if 'recycle' in kwargs:
                self.neutral_dens_original = calc['neutral_dens']
            else:
                self.neutral_dens_original = array([1.0])
        self.neutral_dens = self.neutral_dens_original
        
        if 'Born' in kwargs:
            self.gff_Born_approx_original                 = getFirstValue(kwargs,'Born', 0.0)   
        else:
            if 'recycle' in kwargs:
                self.gff_Born_approx_original = calc['Born']
            else:
                self.gff_Born_approx_original = array([0.0])
        self.gff_Born_approx = self.gff_Born_approx_original

        if 'delta_a' in kwargs:
            self.delta_a_original                 = getFirstValue(kwargs,'delta_a', 0.0)   
        else:
            if 'recycle' in kwargs:
                self.delta_a_original = calc['delta_a']
            else:
                self.delta_a_original = array([0.0])
        self.delta_a = self.delta_a_original

        if 'delta_h' in kwargs:
            self.delta_h_original                 = getFirstValue(kwargs,'delta_h', 0.0)   
        else:
            if 'recycle' in kwargs:
                self.delta_h_original = calc['delta_h']
            else:
                self.delta_h_original = array([0.0])
        self.delta_h = self.delta_h_original

        # If at least one of the two deltas has changed, the coordinate system on
        # flux surfaces is set. Basically it calculates the coefficients of the 
        # system and puts them in COORD
        
        # Check that delta_a and delta_h satisfy necessary conditions
        # in the IDL version, this check is done in the sxr_rho subroutine
        
        self.check_delta(self.delta_a, self.delta_h)
        # Non-maxwellian 
        # energy where a non-maxwellian tail is added to the spectrum (eV) 
        if 'en0' in kwargs:
            self.en0_original                 = getFirstValue(kwargs,'en0', 0.0)   
        else:
            if 'recycle' in kwargs:
                self.en0_original = calc['en0']
            else:
                self.en0_original = array([0.0])
        self.en0 = self.en0_original  

        # increase of the non-maxwellian tail of the spectrum (%, 1 being 100%)
        if 'nm' in kwargs:
            self.nm_original                 = getFirstValue(kwargs,'nm', 0.0)   
        else:
            if 'recycle' in kwargs:
                self.nm_original = calc['nm']
            else:
                self.nm_original = array([0.0])
        self.nm = self.nm_original
        
        self.f_nm              = zeros(self.n_en)
        if self.nm:
            if self.nm != 0.0:
                ind_en = where(self.en >= self.en0)
            
                en_list = self.en.tolist()        
                self.f_nm[ind_en] = self.nm * (self.en[ind_en] - self.en0)

        #FILTER
        
        use_mu_changed = 0
        use_BESSY_changed = 0
        eff_mu_changed = 0
        of_changed = 0
        eff_filt_dens_changed = 0
        DSX3_filt_dens_changed = 0

        self.use_mu = findvalue(kwargs.get('use_mu', 1))
        self.use_BESSY = findvalue2(kwargs.get('use_BESSY', 0))
        self.eff_mu = findvalue2(kwargs.get('eff_mu', 0))
        self.old_foils = findvalue(kwargs.get('old_foils', 0))
        self.eff_filt_dens = findvalue2(kwargs.get('eff_filt_dens', 0))
        self.DSX3_filt_dens = kwargs.get('DSX3_filt_dens',0)
        
        if self.use_mu != 1:
            use_mu_changed = 1
        if self.use_BESSY != 0:
            use_BESSY_changed = 1
        if self.eff_mu != 0:
            eff_mu_changed = 1
        if self.old_foils != 0:
            of_changed = 1
        if self.eff_filt_dens != 0:
            eff_filt_dens_changed = 1
        if self.DSX3_filt_dens != 0:
            DSX3_filt_dens_changed = 1
        
        if 'be_thick' in kwargs:
            self.be_thick_original                 = np_array(kwargs.get('be_thick', self.geometry_bethickness))  
        else:
            if 'recycle' in kwargs:
                self.be_thick_original = calc['be_thick']
            else:
                self.be_thick_original = np_array(self.geometry_bethickness)
        self.be_thick = self.be_thick_original

        if 'si_thick' in kwargs:
            self.si_thick_original                 = np_array(kwargs.get('si_thick', 35.0/cos(self.geometry_angle_fact)))  
        else:
            if 'recycle' in kwargs:
                self.si_thick_original = calc['si_thick']
            else:
                self.si_thick_original = np_array(35.0/cos(self.geometry_angle_fact))
        self.si_thick = self.si_thick_original

        self.n_Be = self.be_thick.size
        sp_Be_n = zeros(self.n_Be*2)
        for i in xrange(self.n_Be):
            sp_Be_n[2*i] = 25.0
            sp_Be_n[2*i+1] = 0.0

        self.Be_changed_array = zeros(self.n_Be)
        Be_changed = False
        be_thick = zeros(2 * self.be_thick.size)
        for i in xrange(self.n_Be):
            be_thick[2*i] = self.be_thick[i]
            be_thick[2*i+1] = 0.0
        for i in xrange(self.n_Be):
            if abs(be_thick-sp_Be_n)[2*i] != 0.0:
                self.Be_changed_array[i] = 1
                Be_changed = True
                self.p_thick = long(6.305e7) * self.spBe[0]
                self.be_thick = be_thick
            else:
                pass

        pc_changed = False
        
        if 'perc_comp' in kwargs:
            self.perc_comp_original                 = np_array(kwargs.get('perc_comp'))
        else:
            if 'recycle' in kwargs:
                self.perc_comp_original = calc['perc_comp']
            else:
                self.perc_comp_original = np_array(None)
        perc_comp = self.perc_comp_original

        perc_comp_list = perc_comp.tolist()

        if perc_comp_list[0] == None:
            pc_changed = False
            
        else:
            n_perc_comp = self.perc_comp.size
            
            if n_perc_comp != self.n_comp and n_perc_comp != self.n_comp * 2L:
                raise ValueError('wrong PERC_COMP, will not be changed')
                pc_changed = False
                
            else:
                if n_perc_comp == self.n_comp:
                    perc_comp_shape = self.perc_comp.shape
                    perc_comp_zeros = zeros(perc_comp_shape)
                    pc_dum = array([self.perc_comp, perc_comp_zeros])
                    
                else:
                    pc_dum = self.perc_comp
                
                if abs(pc_dum - self.perc_comp).sum() != 0:
                    self.perc_comp = pc_dum
                    pc_changed = True
                    
                else:
                    pc_changed = False
                
                pc_dum = 0

        if en_changed or Be_changed or pc_changed or use_mu_changed or use_BESSY_changed or eff_mu_changed or of_changed or eff_filt_dens_changed or DSX3_filt_dens_changed:
            be_size = self.be_thick.size
            self.T_Be = zeros((be_size/2, 501))
            spt = zeros((be_size/2, 501))
            
            self.be_thick = self.be_thick.reshape((self.be_thick.size/2,2))
            for i in xrange(be_size/2):
                if (self.be_thick[i] == self.be_thick[0:i]).all(1).any():
                    for b in xrange(i):
                        if (self.be_thick[i] == self.be_thick[b]).any():
                            self.T_Be[i] = self.T_Be[b]
                            spt[i] = spt[b]
                else:
                    sxr_Be = self.sxr_Be(self.en, self.be_thick[i], self.perc_comp, use_mu = self.use_mu, use_BESSY = self.use_BESSY, eff_mu = self.eff_mu, old_foils = self.old_foils, eff_filt_dens = self.eff_filt_dens)
                    self.T_Be[i] = array(sxr_Be['T'])
                    spt[i] = array(sxr_Be['Tt'])
            for i in xrange(self.Be_changed_array.size):
                if self.Be_changed_array[i] == 0:
                    spt[i] = 0.0
                    self.T_Be[i] = 0.0
        
        si_thick_changed = False
        
        if isinstance(self.si_thick, float) or isinstance(self.si_thick, int):
            
            if self.si_thick == 100.0:
                si_thick_changed = False
            else:
                si_thick_changed = True
        else:
            
            if any(self.si_thick != 100.0):
                si_thick_changed_array = zeros(self.si_thick.size)
                for i in xrange(self.si_thick.size):
                    if self.si_thick[i] != 100.0:
                        si_thick_changed_array[i] = 1
                        si_thick_changed = True
                    else:
                        pass

        # Calculate the transmission of the silicon detector of thickness si_thick 

        if en_changed or si_thick_changed or use_mu_changed:
            self.tr_si = self.sxr_tr_Si(self.si_thick, use_mu = self.use_mu)
            self.a_si = 1.0 - self.tr_si
            self.sit = transpose(log(self.a_si))
       
        # Isola Te - Te island
        if 'r0' in kwargs:
            self.r0_original                 = np_array(kwargs.get('r0', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.r0_original = calc['r0']
            else:
                self.r0_original = np_array([0.0])
        self.r0 = self.r0_original

        if 'th0' in kwargs:
            self.th0_original                 = np_array(kwargs.get('th0', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.th0_original = calc['th0']
            else:
                self.th0_original = np_array([0.0])
        self.th0 = self.th0_original

        if 'del_r' in kwargs:
            self.del_r_original                 = np_array(kwargs.get('del_r', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.del_r_original = calc['del_r']
            else:
                self.del_r_original = np_array([0.0])
        self.del_r = self.del_r_original

        if 'del_th' in kwargs:
            self.del_th_original                 = np_array(kwargs.get('del_th', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.del_th_original = calc['del_th']
            else:
                self.del_th_original = np_array([0.0])
        self.del_th = self.del_th_original

        if 'del_te' in kwargs:
            self.del_te_original                 = np_array(kwargs.get('del_te', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.del_te_original = calc['del_te']
            else:
                self.del_te_original = np_array([0.0])
        self.del_te = self.del_te_original
        
        # Isola ne - ne island
        if 'ne_r0' in kwargs:
            self.ne_r0_original                 = np_array(kwargs.get('ne_r0', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.ne_r0_original = calc['ne_r0']
            else:
                self.ne_r0_original = np_array([0.0])
        self.ne_r0 = self.ne_r0_original

        if 'ne_th0' in kwargs:
            self.ne_th0_original                 = np_array(kwargs.get('ne_th0', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.ne_th0_original = calc['ne_th0']
            else:
                self.ne_th0_original = np_array([0.0])
        self.ne_th0 = self.ne_th0_original

        if 'ne_del_r' in kwargs:
            self.ne_del_r_original                 = np_array(kwargs.get('ne_del_r', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.ne_del_r_original = calc['ne_del_r']
            else:
                self.ne_del_r_original = np_array([0.0])
        self.ne_del_r = self.ne_del_r_original

        if 'ne_del_th' in kwargs:
            self.ne_del_th_original                 = np_array(kwargs.get('ne_del_th', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.ne_del_th_original = calc['ne_del_th']
            else:
                self.ne_del_th_original = np_array([0.0])
        self.ne_del_th = self.ne_del_th_original

        if 'del_ne' in kwargs:
            self.del_ne_original                 = np_array(kwargs.get('del_ne', 0.0)) 
        else:
            if 'recycle' in kwargs:
                self.del_ne_original = calc['del_ne']
            else:
                self.ne_del_ne_original = np_array([0.0])
        self.del_ne = self.del_ne_original

        # Corona ne - ne ring   
        if 'del_dens' in kwargs:
            self.del_dens_original                 =  getFirstValue(kwargs, 'del_dens', 0.0)
        else:
            if 'recycle' in kwargs:
                self.del_dens_original = calc['del_dens']
            else:
                self.del_dens_original = array([0.0])
        self.del_dens = self.del_dens_original

        if 'del_dr' in kwargs:
            self.del_dr_original                 = getFirstValue(kwargs, 'del_dr', 0.0)
        else:
            if 'recycle' in kwargs:
                self.del_dr_original = calc['del_dr']
            else:
                self.del_dr_original = array([0.0])
        self.del_dr = self.del_dr_original

        if 'dr' in kwargs:
            self.dr_original                 = getFirstValue(kwargs, 'dr', 0.0)
        else:
            if 'recycle' in kwargs:
                self.dr_original = calc['dr']
            else:
                self.dr_original = array([0.0])
        self.dr = self.dr_original

        # Main Gas

        if 'main_gas' in kwargs:
            self.main_gas_original = kwargs.get('main_gas')
            self.main_gas = self.main_gas_original
            if keyword_set(self.main_gas):
                if self.main_gas[0].upper() == 'H':
                    ind_maj = where(self.ion_names == 'H+1')
                    new_main_gas = 'H'
                elif self.main_gas[0].upper() == 'D':
                    ind_maj = where(self.ion_names == 'D+1')
                    new_main_gas = 'D'
                elif self.main_gas[0].upper() == 'HE':
                    ind_maj = where(self.ion_names == 'He+2')
                    new_main_gas = 'He'
                    
                else:
                    ind_maj = where(array(self.ion_names) == self.main_gas[0].capitalize())
                    if array(ind_maj[0]).size != 0:
                        new_main_gas = self.ion_names[ind_maj[0][0]]
                if array(ind_maj[0]).size == 0:
                    print "check MAIN_GAS, %s not in the available list of ions" % (self.main_gas[0])
                    main_gas_changed = 0
                    raise ValueError('check MAIN_GAS')
                else:
                    self.maj = {'ion_names': self.ion_names[ind_maj[0][0]], 'Z': Z[ind_maj[0][0]], 'imp_chi': self.imp_chi[ind_maj[0][0]], 'imp_bound_e': self.imp_bound_e[ind_maj[0][0]], 'chi': chi[ind_maj[0][0]], 'b_e': b_e[ind_maj[0][0]], 'n_min': n_min[ind_maj[0][0]]} 
            
                    self.main_gas = new_main_gas
            
                    main_gas_changed = 1
            else:
                main_gas_changed = 0
        else:
            main_gas_changed = 0
            self.main_gas_original = 'D'
            self.main_gas = self.main_gas_original
            
        # Prof ni
        if 'ni_dens' in kwargs:
            self.ni_dens_original                 = np_array(kwargs.get('ni_dens', 0.0))
        else:
            if 'recycle' in kwargs:
                self.ni_dens_original = calc['ni_dens']
            else:
                self.ni_dens_original = np_array([0.0])
        self.ni_dens = self.ni_dens_original

        if 'alpha_ni' in kwargs:
            self.alpha_ni_original                 = np_array(kwargs.get('alpha_ni', 0.0))
        else:
            if 'recycle' in kwargs:
                self.alpha_ni_original = calc['alpha_ni']
            else:
                self.alpha_ni_original = np_array([0.0])
        self.alpha_ni = self.alpha_ni_original  

        if 'beta_ni' in kwargs:
            self.beta_ni_original                 = np_array(kwargs.get('beta_ni', 0.0))
        else:
            if 'recycle' in kwargs:
                self.beta_ni_original = calc['beta_ni']
            else:
                self.beta_ni_original = np_array([0.0])
        self.beta_ni = self.beta_ni_original

        # Corona ni - ni ring
        if 'ni_dr' in kwargs:
            self.ni_dr_original                 = np_array(kwargs.get('ni_dr', 0.0))
        else:
            if 'recycle' in kwargs:
                self.ni_dr_original = calc['ni_dr']
            else:
                self.ni_dr_original = np_array([0.0])
        self.ni_dr = self.ni_dr_original

        if 'del_ni_dr' in kwargs:
            self.del_ni_dr_original                 = np_array(kwargs.get('del_ni_dr', 0.0))
        else:
            if 'recycle' in kwargs:
                self.del_ni_dr_original = calc['del_ni_dr']
            else:
                self.del_ni_dr_original = np_array([0.0])
        self.del_ni_dr = self.del_ni_dr_original

        if 'del_ni_dens' in kwargs:
            self.del_ni_dens_original                 = np_array(kwargs.get('del_ni_dens', 0.0))
        else:
            if 'recycle' in kwargs:
                self.del_ni_dens_original = calc['del_ni_dens']
            else:
                self.del_ni_dens_original = np_array([0.0])
        self.del_ni_dens = self.del_ni_dens_original
                
        # Isola ni - ni island
        if 'is_del_ni' in kwargs:
            self.is_del_ni_original                 = np_array(kwargs.get('is_del_ni', 0.0))
        else:
            if 'recycle' in kwargs:
                self.is_del_ni_original = calc['is_del_ni']
            else:
                self.is_del_ni_original = np_array([0.0])
        self.is_del_ni = self.is_del_ni_original

        if 'is_ni_dr' in kwargs:
            self.is_ni_dr_original                 = np_array(kwargs.get('is_ni_dr', 0.0))
        else:
            if 'recycle' in kwargs:
                self.is_ni_dr_original = calc['is_ni_dr']
            else:
                self.is_ni_dr_original = np_array([0.0])
        self.is_ni_dr = self.is_ni_dr_original

        if 'is_dr' in kwargs:
            self.is_dr_original                 = np_array(kwargs.get('is_dr', 0.0))
        else:
            if 'recycle' in kwargs:
                self.is_dr_original = calc['is_dr']
            else:
                self.is_dr_original = np_array([0.0])
        self.is_dr = self.is_dr_original

        if 'ni_th_dr' in kwargs:
            self.ni_th_dr_original                 = np_array(kwargs.get('ni_th_dr', 0.0))
        else:
            if 'recycle' in kwargs:
                self.ni_th_dr_original = calc['ni_th_dr']
            else:
                self.ni_th_dr_original = np_array([0.0])
        self.ni_th_dr = self.ni_th_dr_original

        if 'ni_del_th' in kwargs:
            self.ni_del_th_original                 = np_array(kwargs.get('ni_del_th', 0.0))
        else:
            if 'recycle' in kwargs:
                self.ni_del_th_original = calc['ni_del_th']
            else:
                self.ni_del_th_original = np_array([0.0])
        self.ni_del_th = self.ni_del_th_original

        # IMPS and RECOMB
        if 'ni_ADAS' in kwargs:
            self.ni_ADAS_original                 = kwargs.get('ni_ADAS', 0)
        else:
            if 'recycle' in kwargs:
                self.ni_ADAS_original = calc['ni_ADAS']
            else:
                self.ni_ADAS_original = 0
        self.ni_ADAS = self.ni_ADAS_original

        ni_ADAS_changed = False
        if isinstance(self.ni_ADAS, int):
            ni_ADAS = keyword_set(self.ni_ADAS)
            
            if ni_ADAS != 0:
                self.ni_ADAS = ni_ADAS
                ni_ADAS_changed = True
                
        else:
            ni_ADAS = keyword_set(ni_ADAS[0])
            
            if ni_ADAS != 0:
                self.ni_ADAS = ni_ADAS
                ni_ADAS_changed = True

        if 'imp_str' in kwargs:
            self.imp_str_original                 = kwargs.get('imp_str', [])
        else:
            if 'recycle' in kwargs:
                self.imp_str_original = calc['imp_str']
            else:
                self.imp_str_original = []
        self.imp_str = self.imp_str_original

        self.Z = self.maj['Z']
        self.chi = self.maj['chi']
        self.b_e = self.maj['b_e']
        self.n_min = self.maj['n_min']
        self.ion_name = self.maj['ion_names']
        self.imp_str = array(self.imp_str)
        self.n_imp = self.imp_str.size
        imp_str_list = self.imp_str.tolist()

        imp_str_changed = False

        if self.n_imp == 0:
            self.imp_str_o = ''
            impn = ''
            self.ind_11 = -1
            ind_11_o = -1
            self.n_imp_e = 0
            self.n_ions = 1
            self.ind_maj = 0
            self.ion_str = array(['D+1'])
        else:
            self.Z = array([])
            self.chi = array([])
            self.b_e = array([])
            self.n_min = array([])
            self.ion_name = array([])
            self.imp_str_o = []

            for item in imp_str_list:
                item = item.lower()
                item = item.replace(" ", "")
                self.imp_str_o.append(item)
            imp_str = self.imp_str_o
            imp_str = sorted(imp_str)

            ind_i = exclude(array(imp_str), array(self.ions))
            if ind_i.size == 1:
                if ind_i != -1:
                    c = 'this impurity ion:   ' + imp_str_list[ind_i] +  ' is not available in the library'
                    raise ValueError(c)
                else:
                    impn = array([])
                    for i in xrange(0, self.n_imp):
                        
                        ind = where(imp_str[i] == self.ions)
                        impn = add_float(impn, array(self.ion_names[ind[0][0]]))
                    ind_11 = where('Al+11' == impn)

                    if ind_11[0][0] > 0:
                        dum = impn[0]
                        impn[0] = impn[ind_11[0][0]]
                        impn[ind_11[0][0]] = dum
                        ind_11 = [0]

                    ind_11_o = where(self.imp_str_o == 'al+11')
            else:
                n_ind = ind_i.size
                v = str(n_ind)
                
                v = v.lstrip()
                v = v.rstrip()
                empt = []
                for item in ind_i:
                    empt.append(imp_str[item])
                mmm = array(empt) 
                
                raise ValueError('these impurity ions: %s are not available in the library' % mmm)

            self.imp_str = impn
            self.imp_str_o = self.imp_str_o
            self.imp_str_e = impn
            self.n_imp_e = self.n_imp
            
            self.ind_11 = ind_11[0]
            self.ind_11_o = ind_11_o[0]
            
            imp_str_changed = 1
        
        if ni_ADAS_changed and self.n_imp == 0:
            ni_ADAS_changed = False
            
        if self.ni_ADAS == 1 and self.ind_11 == -1:
            raise ValueError('NI_ADAS=1 but Al+11 is missing in the ion list. NI_ADAS will be ignored')
        
                
        if imp_str_changed or ni_ADAS_changed or main_gas_changed:

            if self.n_imp == 0:
                self.Z = self.maj['Z']
                self.chi = self.maj['chi']
                self.b_e = self.maj['b_e']
                self.n_min = self.maj['n_min']
                self.imp_str = self.maj['ion_names']
                self.n_imp = 1
                
            else:
                for i in xrange(0, self.n_imp):

                    imp_str_list = self.imp_str.tolist()
                    if self.imp_str.size == 1:
                        ind = where(imp_str_list == array(self.ion_names))
                        
                    else:
                        ind = where(imp_str_list[i] == array(self.ion_names))
                    numb = ind[0][0]
                    self.Z = add_float(self.Z, Z[numb])
                    self.chi = add_float(self.chi, chi[numb])
                    self.b_e = add_float(self.b_e, b_e[numb])
                    self.n_min = add_float(self.n_min, n_min[numb])

                if self.ni_ADAS and self.ind_11 != -1:
                    imp_str_e = self.imp_str
                    n_imp_e = self.n_imp

                    ind_10 = where(self.imp_str == 'Al+10')
                    ind_10 = ind_10[0]
                    ind_10 = ind_10.tolist()

                    if ind_10 == []:
                        self.Z = add_float(self.Z, Z[7])
                        self.chi = add_float(self.chi, chi[7])
                        self.b_e = add_float(self.b_e, b_e[7])
                        self.n_min = add_float(self.n_min, n_min[7])
                        
                        imp_str_e = add_float(imp_str_e, 'Al+10')
                        n_imp_e = n_imp_e + 1
                        
                    ind_12 = where(self.imp_str == 'Al+12')
                    ind_12 = ind_12[0]
                    ind_12 = ind_12.tolist()
                    
                    if ind_12 == []:
                        self.Z = add_float(self.Z, Z[8])
                        self.chi = add_float(self.chi, chi[8])
                        self.b_e = add_float(self.b_e, b_e[8])
                        self.n_min = add_float(self.n_min, n_min[8])
                        
                        imp_str_e = add_float(imp_str_e, 'Al+12')
                        n_imp_e = n_imp_e + 1
                        
                    ind_13 = where(self.imp_str == 'Al+13')
                    ind_13 = ind_13[0]
                    ind_13 = ind_13.tolist()
                    
                    if ind_13 == []:
                        self.Z = add_float(self.Z, Z[9])
                        self.chi = add_float(self.chi, chi[9])
                        self.b_e = add_float(self.b_e, b_e[9])
                        self.n_min = add_float(self.n_min, n_min[9])
                        
                        imp_str_e = add_float(imp_str_e, 'Al+13')
                        n_imp_e = n_imp_e + 1
                    
                    self.imp_str_e = imp_str_e
                    self.n_imp_e = n_imp_e
                else:
                    self.imp_str_e = self.imp_str
                    self.n_imp_e = self.n_imp

                self.ion_str = self.imp_str_e.tolist() + [self.maj['ion_names']]
                self.n_ions = self.n_imp_e + 1
                
                # Add the parameters for main gas recombination
                
                self.Z = add_float(self.Z, self.maj['Z'])
                self.chi = add_float(self.chi, self.maj['chi'])
                self.b_e = add_float(self.b_e, self.maj['b_e'])
                self.n_min = add_float(self.n_min, self.maj['n_min'])

                # NOTE on Al profiles when NI_ADAS=1
                #
                # From what I understand, using a single exp_a and exp_b for Al+11 is fine.
                # The profile behavior for the other charges states of Al is most strongly determined
                # by the temperature profile. For example, as the temperature decreases, the fractional
                # amount of Al+10 increases compared Al+11. This will likely result in a flatter profile
                # for Al+10 than Al+11. Al+12, on the other hand, starts out with a fractional amount that
                # is smaller than Al+11 and decreases as the temperature decreases. This will likely lead
                # to a more peaked profile for Al+12. This may even still be valid if you add a density ring
                # to the Al+11 density profile.
            self.ind_maj = where(array(self.ion_str) == self.maj['ion_names'])[0][0]
        
        # now that the list of impurity ions has been defined, each parameter in the
        # structures PROF_NI, ISOLA_NI e CORONA_NI should have N_IMP elements,
        # and they should be ordered as in IMP_STR
        #
        # if a single value is passed to one of these parameters an array of N_IMP
        # elements all equal to that value will be created
        #
        # if instead there are no impurities these parameters will be set to zero
        
        if self.n_imp == 0:
            
        # PROF_NI
            self.ni_dens = self.sXe_N(a = 0.0, par_name = 'ni_dens', par = self.ni_dens)
            self.alpha_ni = self.sXe_N(a = 0.0, par_name = 'ni_esp_dens', par = self.alpha_ni)
            self.beta_ni = self.sXe_N(a = 0.0, par_name = 'ni_dens', par = self.beta_ni)
            
        # ISOLA_NI
            self.is_del_ni = self.sXe_N(a = 0.0, par_name = 'is_del_ni', par = self.is_del_ni)
            self.is_ni_dr = self.sXe_N(a = 0.0, par_name = 'is_ni_dr', par = self.is_ni_dr)
            self.is_dr = self.sXe_N(a = 0.0, par_name = 'is_dr', par = self.is_dr)
            self.ni_th_dr = self.sXe_N(a = 0.0, par_name = 'ni_th_dr', par = self.ni_th_dr)
            self.ni_del_th = self.sXe_N(a = 0.0, par_name = 'ni_del_th', par = self.ni_del_th)
        
        # CORONA_NI
        
            self.ni_dr = self.sXe_N(a = 0.0, par_name = 'ni_dr', par = self.ni_dr)
            self.del_ni_dr = self.sXe_N(a = 0.0, par_name = 'del_ni_dr', par = self.del_ni_dr)
            self.del_ni_dens = self.sXe_N(a = 0.0, par_name = 'del_ni_dens', par = self.del_ni_dens)

        else:

        # PROF_NI

            self.ni_dens = self.sXe_N(a = self.ni_dens_original, par_name = 'ni_dens', par = self.ni_dens)

            self.alpha_ni = self.sXe_N(a = self.alpha_ni_original, par_name = 'alpha_ni', par = self.alpha_ni)
            self.beta_ni = self.sXe_N(a = self.beta_ni_original, par_name = 'beta_ni', par = self.beta_ni)
            
        # ISOLA_NI
            self.is_del_ni = self.sXe_N(a = self.is_del_ni_original, par_name = 'is_del_ni', par = self.is_del_ni)
            self.is_ni_dr = self.sXe_N(a = self.is_ni_dr_original, par_name = 'is_ni_dr', par = self.is_ni_dr)
            self.is_dr = self.sXe_N(a = self.is_dr_original, par_name = 'is_dr', par = self.is_dr)
            self.ni_th_dr = self.sXe_N(a = self.ni_th_dr_original, par_name = 'ni_th_dr', par = self.ni_th_dr)
            self.ni_del_th = self.sXe_N(a = self.ni_del_th_original, par_name = 'ni_del_th', par = self.ni_del_th)
        
        # CORONA_NI
            self.ni_dr = self.sXe_N(a = self.ni_dr_original, par_name = 'ni_dr', par = self.ni_dr)
            self.del_ni_dr = self.sXe_N(a = self.del_ni_dr_original, par_name = 'del_ni_dr', par = self.del_ni_dr)
            self.del_ni_dens = self.sXe_N(a = self.del_ni_dens_original, par_name = 'del_ni_dens', par = self.del_ni_dens)
        
        ### TABV ###
        # If specific profiles want to passed for te, ne, or ni, then at least 
        # two values and radial locations must be defined. If they are, then
        # they will be used instead of the alpha model parameters. 
        
        # temperature
        self.tabv_te           = np_array(kwargs.get('tabv_te',0)) 
        self.tabv_r_te_gc         = np_array(kwargs.get('tabv_r_te',0)) 
        if any(self.tabv_r_te_gc):
            self.tabv_r_te = self.sxr_rho(self.tabv_r_te_gc/self.radius, 
                                zeros(self.tabv_r_te_gc.size), norm=True)
        
        if (self.tabv_te.size >= 2.0) and (self.tabv_r_te.size >= 2.0):
            self.tabv_use_te       = True
        else:
            self.tabv_use_te        = False
    
        # density
        self.tabv_dens         = np_array(kwargs.get('tabv_dens',0)) 
        self.tabv_r_dens_gc    = np_array(kwargs.get('tabv_r_dens',0))
        
        if any(self.tabv_r_dens_gc):
            self.tabv_r_dens = self.sxr_rho(self.tabv_r_dens_gc/self.radius, 
                                    zeros(self.tabv_r_dens_gc.size), norm=True)
        
        # tabv_dens and tabv_r_dens must have at least 2 points to 
        # be able to interpolate
        if (self.tabv_dens.size >= 2.0) and (self.tabv_r_dens.size >= 2.0):
            self.tabv_use_dens     = True
        else:
            self.tabv_use_dens     = False
          
        # impurity density
        self.tabv_ni_dens      = np_array(kwargs.get('tabv_ni_dens',0))     
        self.tabv_r_ni_dens_gc = np_array(kwargs.get('tabv_r_ni_dens', 0))
        
        if any(self.tabv_r_ni_dens_gc):
            self.tabv_r_ni_dens = self.sxr_rho(
                                    self.tabv_r_ni_dens_gc/self.radius, 
                                    zeros(self.tabv_r_ni_gc.size), norm=True)
        
        # tabv_ni_dens and tabv_r_ni_dens must have at least 2 points to 
        # be able to interpolate
        if (self.tabv_ni_dens.size>=2.0) and (self.tabv_r_ni_dens.size>=2.0):
            self.tabv_use_ni_dens       = True
        else:
            self.tabv_use_ni_dens        = False     
            
        #TABV_EM rho, tzeta, and emissivity
        
        # X is defined: RH and TZ are defined as uniform arrays in [0...1]
        # and [0...2pi] (same number of elements of TABV_E_X)   
         
        self.tabv_e_x        = np_array(kwargs.get('tabv_e_x', 0))

        if any(self.tabv_e_x):
            n_tabv_e_x = self.tabv_e_x.size
            self.tabv_e_rh   = vect_0_1(n_tabv_e_x)
            self.tabv_e_tz   = vect_0_1(n_tabv_e_x, no_1 = True) * 2.0 * pi
            self.tabv_e_n    = n_tabv_e_x
        # save the RH and TZ vectors normalized to the number of elements N.
        # These will be used to speed up the calculations in the SXR_GENERIC
            
            self.rh_n = ( self.tabv_e_rh ) * ( self.tabv_e_n - 1.0 )
            self.tz_n = ( ( self.tabv_e_tz ) / ( 2.0*pi ) ) * ( self.tabv_e_n - 1.0 )
            
            x_changed = True
        else:
            x_changed = False
        
        self.tabv_e_y        = np_array(kwargs.get('tabv_e_y', 0))
        if any(self.tabv_e_y):
            n_tabv_e_y = self.tabv_e_x.size
            y_changed = True  
        else:
            y_changed = False
            n_tabv_e_y = 0
        
        # save the original EMISS array
        
        self.tabv_emiss = np_array(kwargs.get('tabv_emiss', 0))
        if any(self.tabv_emiss):
            n_tabv_emiss = self.tabv_emiss.size
            
            if n_tabv_emiss != 0:
                sz_e = self.tabv_emiss.shape
                
                if n_tabv_emiss < 4:
                    self.tabv_e_e = -99.0
                    e_changed = False
                    
                elif sz_e.size != 2:
                    e_changed = False
                    raise ValueError('TABV_EMISS should be a 2-dimension array')
                
                elif sz_e[1] != n_tabv_e_x:
                    e_changed = False
                    raise ValueError('TABV_EMISS dimension 0 (columns) should be the same\
                    as TABV_E_X')
                    
                elif sz_e[0] != n_tabv_e_y:
                    e_changed = False
                    raise ValueError('TABV_EMISS dimension 1 (rows) should be the same\
                     as TABV_E_Y')
                    
                else:
                    e_changed = True
                
            else:
                e_changed = False
        else:
            e_changed = False
        
        if x_changed or y_changed or e_changed:
            emiss = zeros((n_tabv_e_x, n_tabv_e_x))
            
            rh_t = self.tabv_e_rh * ( 1.0 - abs(self.delta_h) / self.radius) # rho NOT norm
            
            mn_x = min(self.tabv_e_x) / self.radius
            mx_x = max(self.tabv_e_x) / self.radius
            
            mn_y = min(self.tabv_e_y) / self.radius
            mx_y = max(self.tabv_e_y) / self.radius
            
            for t in xrange(n_tabv_e_x):
                tabv_e_tz_list = self.tabv_e_tz.tolist()
                x_t = rh_t * cos(tabv_e_tz_list[t]) + ( maximum(self.sxr_delta_rho(rh_t),1e-4))
                x_t = ( ( x_t - mn_x ) / ( mx_x - mn_x ) ) * ( n_tabv_e_x - 1.0 )
                
                y_t = rh_t * sin(tabv_e_tz_list[t])
                y_t = ( ( y_t - mn_y ) / (mx_y - mn_y ) ) * (n_tabv_e_y - 1.0 )


                emiss[0, ] = interp(self.tabv_emiss, x_t, y_t)
                
            self.tabv_e_e = emiss # E is EMISS mapped
        try:
            self.tabv_e_e
            
        except AttributeError:
            use_emiss = False
            
        else:
            if self.tabv_e_e.size >= 4 and self.tabv_e_rh.size >= 2:
                use_emiss = True      

               
        ### Pre-calculated quantities to speed up calculation ###
        
        ## rhoy, tze    
        # This calculation is done in the sxr_line_integral subroutine in the
        # old IDL code. Here it is calculated right away and stored, then it 
        # is only recalculated if necessary
        smin = -sqrt(1.0 - where(self.p<1.0, self.p, ones(size(self.p)))**2.0)
        smax = -smin

        self.delta_s = (smax - smin) / self.n_s

        vs = vect_0_1(self.n_s)
        step = (smax - smin - self.delta_s)
        split = smin + self.delta_s / 2.0
        calc_ns = ones(self.n_s)
        #multiplies each element of step by the vs array
        self.s = step[...,newaxis]*vs[newaxis,...] + \
            split[...,newaxis]*calc_ns[newaxis,...]

        #the diagonal of self.x is the x in the 2d array, same with the self.y
        self.x = einsum('i,j->ij', self.p*cos(self.phi), calc_ns) + \
            self.s * einsum('i,j->ij',sin(self.phi), calc_ns) 
        
        self.y = einsum('i,j->ij', self.p * sin(self.phi), calc_ns) - \
            self.s * einsum('i,j->ij',cos(self.phi), calc_ns)
        
        
        self.rhoy = self.sxr_rho(self.x,self.y)
        self.tze = self.sxr_tzeta(self.x,self.y)

        ## arrays used later in spectrum and recomb_spec methods
        
        # CALC
        self.calc_cc  = ones(self.n_en) ###################
        
        self.calc_ent = self.en

        self.calc_dent = self.delta_en

        self.calc_spt = spt

        self.calc_sit = self.sit

        self.calc_nmt = self.f_nm
        
        ## temperature, density, and impurity profiles
        if self.tabv_use_te == False:
            self.tep = sxr_prof_te(self.rhoy, self.te, self.alpha_te, self.beta_te, self.del_te, self.del_th, self.tze, self.del_r, self.r0, self.th0, self.tabv_use_te)
        else:
            self.tep = sxr_prof_te(self.rhoy, self.te, self.alpha_te, self.beta_te, self.tabv_use_te, tabv_r_te = self.tabv_r_te, tabv_te = self.tabv_te)
        if self.tabv_use_dens == False:
            self.nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens, ) 
        else:
            self.nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens, self.tabv_r_dens, self.tabv_dens) 

        parameters = {
            'te': self.te_original, 'alpha_te':self.alpha_te_original, 'beta_te':self.beta_te_original, 
               'r0':self.r0_original, 'th0':self.th0_original, 'del_r':self.del_r_original, 'del_th':self.del_th_original, 'del_te':self.del_te_original,
               
               'dens':self.dens_original, 'alpha_ne':self.alpha_ne_original, 'beta_ne':self.beta_ne_original,
               'ne_r0':self.ne_r0_original, 'ne_del_r':self.ne_del_r_original, 'ne_th0':self.ne_th0_original, 'ne_del_th':self.ne_del_th_original, 'del_ne':self.del_ne_original,
               'del_dens':self.del_dens_original, 'del_dr':self.del_dr_original, 'dr':self.dr_original,
               
               'ni_dens':self.ni_dens_original, 'alpha_ni':self.alpha_ni_original, 'beta_ni':self.beta_ni_original, 
               'ni_dr':self.ni_dr_original, 'del_ni_dr':self.del_ni_dr_original, 'del_ni_dens':self.del_ni_dens_original,
               'is_del_ni':self.is_del_ni_original, 'is_ni_dr':self.is_ni_dr_original, 'is_dr':self.is_dr_original, 'ni_th_dr':self.ni_th_dr_original, 'ni_del_th':self.ni_del_th_original,
               'imp_str':self.imp_str_original,
               
               'delta_a':self.delta_a_original, 'delta_h':self.delta_h_original,
               'radius':self.radius_original,
               
                'si_thick' : self.si_thick_original, 'be_thick':self.be_thick_original, 'n_s':self.n_s_original, 'n_corde':self.n_corde_original, 'p': self.p_original, 'phi': self.phi_original,
                'hor':self.hor_original, 'neutral_dens':self.neutral_dens_original, 'Born':self.gff_Born_approx_original, 'en0': self.en0_original, 'nm':self.nm_original, 
                'perc_comp':self.perc_comp_original, 'ni_ADAS': self.ni_ADAS_original}
        Save(parameters)
        
        
################################################################################
##################    SXR_X_EMISSION ROUTINES    ###############################
################################################################################             
             
 
    def sxr_line_integral(self):
        from numpy import einsum
        """
        Method of sxr_x_emission class to calculate the line integral along all
        of the lines of sight (p,phi). Returns the expected measured brightness 
        for each diode through each respective beryllium filter. 
        
        All of the necessary parameters are already stored in the class object.
        
        Parameters
        ----------
        save_li : bool, optional
            if True, the line inte
        
        Returns
        -------
        out : array 
            Measured brightnesses (W / m**2)

        """

        func_en = self.sxr_spectrum(self.tze, self.rhoy)['en_int']

        it = einsum('ij,j,k->j', func_en, self.delta_s, self.radius)
        
        return it
    def sxr_Zeff(self, nep=None, nip=None):
        """
        NAME:
            SXR_ZEFF
        PURPOSE:
            Calculates the Zeff
        CALLING SEQUENCE:
            val = sxr_Zeff( rho, tze, nep=nep, nip=nip )
        INPUTS:
            RHO radius of the flux surface (normalized 1=LCFS)
            TZE angle of the radius (rad)
        OPTIONAL INPUT PARAMETERS:
            NEP electron density profile
        KEYWORD PARAMETERS:
            None
        OUTPUTS:
            VAL Zeff in (rho,tze)
        OPTIONAL OUTPUT PARAMETERS:
            NIP ion density profiles, array [rho,ions] (number of
            rows=ST.IMPS.N_IMP; if no ions are specified NIP=0)
        SIDE EFFECTS:
            None
        RESTRICTIONS:
            None
        """
        from numpy import zeros, expand_dims
        from Profiles import sxr_prof_ne
        if nep:
            nep = nep
        else:
            if self.tabv_use_dens == False:
                nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens)
            else:
                nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens, self.tabv_r_dens, self.tabv_dens)
        rd = self.rhoy

        n_rd = rd.shape

        dum = zeros((n_rd[0],n_rd[1]), dtype = float)
        dum = dum


        """
        The typical form given for Zeff is
        
        Zeff = total( n_i * Z_i^2 ) / total( n_i * Z_i )                                 (1)
        
        where the sum is taken over all species, including the majority species.
        Assuming quasi-neutrality, the denominator is equivalent to the electron
        density, so eq. (1) can be written
        
        Zeff = total( n_i * Z_i^2 ) / ne                                                 (2)
        
        Because we're looking specifically at impurity species, we will separate
        out the majority species and the impurities
        
        Zeff = ( n_D * Z_D^2 + total( n_imp * Z_imp^2 ) ) / ne                           (3)
        
        Note that I have changed the range of the summation to highlight the fact
        that it only includes impurity species. n_D is the density of deuterium,
        and Z_D is the charge of deuterium (which is just 1). n_D is often taken
        to be the same as the electron density ne, but technically it is the same
        as what isn't contributed by the impurities. In particular it is given by
        
        n_D = ne - total( n_imp * Z_imp )                                                (4)
        
        where again, the sum goes over the impurity species only. This is a small
        effect but we want to include it. Noting that Z_D=1 and plugging this into
        eq. (3), gives
        
        Zeff = ( ne - total( n_imp * Z_imp ) + total( n_imp * Z_imp^2 ) ) / ne           (5)
        """
        Z_maj = self.maj['Z']

        if self.n_imp > 0:
            Z2 = dum
            den = dum

            #if self.tabv_use_ni_dens == False:
            #    nip_kk = sxr_prof_ni(self.rhoy, self.ni_dens, self.alpha_ni, self.beta_ni, self.del_ni_dens, self.ni_dr, self.del_ni_dr, self.tze, self.is_del_ni, self.is_dr, self.ni_del_th, self.is_ni_dr, self.ni_th_dr, self.tabv_use_ni_dens)
            #else:
            #    nip_kk = sxr_prof_ni(self.rhoy, self.ni_dens, self.alpha_ni, self.beta_ni, self.del_ni_dens, self.ni_dr, self.del_ni_dr, self.tze, self.is_del_ni, self.del_ni, self.is_dr, self.ni_del_th, self.is_ni_dr, self.ni_th_dr, self.tabv_use_ni_dens, en_step=self.en_step, tabv_r_ni_dens=self.tabv_r_ni_dens, tabv_ni_dens=self.tabv_ni_dens)

            for kk in xrange(self.n_imp_e):
                nip_kk = nip[:,:,kk]
                Z2 = Z2 + nip_kk * self.Z[kk] ** 2.0
                den = den + nip_kk * self.Z[kk]
            den = expand_dims(den, 2)
            Z2 = expand_dims(Z2, 2)
            

            self.nmaj = (nep - den ) / Z_maj
            Zeff = ( ( self.nmaj * Z_maj**2.0 ) + Z2 ) / nep
        else:
            self.nmaj = nep
            
            Zeff = dum + Z_maj
            
        return Zeff
    
    def sxr_spectrum(self, tze, rhoy, **kwargs):
        """
        Method to calculate the emissivity for each spatial location along each
        line of sight, for each energy in the energy spectrum (500eV to 10keV)
                
        Returns
        -------
        out : ndarray, shape ''(d0, d1, d2)''
            Emissivity per unit eV per meter (W eV**-1 m**-3)
            d0 - number of lines of sight (n_p)
            d1 - number of energy integral points (n_en)
            d2 - number of spatial integral points (n_s)
            
        """
        from numpy import add, ones, copy, power, inf, log, expand_dims, divide, dot, multiply, NaN, sqrt, reshape, log, einsum, where, exp, zeros, isnan, array, isinf, maximum, minimum, clip, sum
        from sxr_lib import idl_where, add_float, belong5
        from Profiles import sxr_prof_te, sxr_prof_ne, sxr_prof_ni, sxr_prof_nh
        import datetime
        try:
            from numexpr import evaluate
            use_numexpr = True
        except ImportError:
            use_numexpr = False

        if isinstance(rhoy, int) or isinstance(rhoy, float):
            self.rhoy = array(rhoy)
            self.rhoy = self.rhoy.reshape((1,1))
            self.tze = array(tze)
            self.tze = self.tze.reshape((1,1))
        elif isinstance(rhoy, list):
            self.rhoy = array(rhoy)
            self.rhoy = self.rhoy.reshape((len(rhoy), 1))
            self.tze = array(tze)
            self.tze = self.tze.reshape((len(tze), 1))
        else:
            self.rhoy = rhoy
            self.tze = tze

        # SXR spectrum constant
        # Remember that ion and electron densities are in 1E19 m^-3 units, so their product is 1E38
        #while the spectrum constant is 1.52E-38.  The product of the two numbers gives 1.52
        
        K_sxr = 1.52
        # Electron temperature Te and density ne
        if self.tabv_use_te == False:
            tep = sxr_prof_te(self.rhoy, self.te, self.alpha_te, self.beta_te, self.del_te, self.del_th, self.tze, self.del_r, self.r0, self.th0, self.tabv_use_te)
        else:
            tep = sxr_prof_te(self.rhoy, self.te, self.alpha_te, self.beta_te, self.del_te, self.del_th, self.tze, self.del_r, self.r0, self.th0, self.tabv_use_te, self.tabv_r_te, self.tabv_te)

        if self.tabv_use_dens == False:
            nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens)
        else:
            nep = sxr_prof_ne(self.rhoy, self.dens, self.alpha_ne, self.beta_ne, self.tze, self.del_ne, self.ne_r0, self.ne_th0, self.ne_del_r, self.ne_del_th, self.del_dens, self.dr, self.del_dr, self.tabv_use_dens, self.tabv_r_dens, self.tabv_dens)

        #Neutral density
                
        nHp = sxr_prof_nh(1.0, self.rhoy)

        # Ion density ni and main gas nmaj

        if self.n_imp_e > 0:
            if self.tabv_use_ni_dens == False:
                nip = sxr_prof_ni(self.ni_ADAS, self.ind_11, self.imp_str_o, self.imp_str_e, self.n_imp_e, self.rhoy, self.ni_dens, self.alpha_ni, self.beta_ni, self.del_ni_dens, self.ni_dr, self.del_ni_dr, self.tze, self.is_del_ni, self.is_dr, self.ni_del_th, self.is_ni_dr, self.ni_th_dr, self.tabv_use_ni_dens, nep=nep, tep=tep, nHp=nHp)
            else: 
                nip = sxr_prof_ni(self.ni_ADAS, self.ind_11, self.imp_str_o, self.imp_str_e, self.n_imp_e, self.rhoy, self.ni_dens, self.alpha_ni, self.beta_ni, self.del_ni_dens, self.ni_dr, self.del_ni_dr, self.tze, self.is_del_ni, self.is_dr, self.ni_del_th, self.is_ni_dr, self.ni_th_dr, self.tabv_use_ni_dens, nep=nep, tep=tep, nHp=nHp,  tabv_r_ni_dens=None, tabv_ni_dens=None)

            Zeff = self.sxr_Zeff(nip = nip)

        else:
            Zeff = self.sxr_Zeff()

        if self.tze.shape[1] == 1:
            self.tze = self.tze.reshape((self.tze.size, ))
            if self.rhoy.shape[1] == 1:
                self.rhoy = self.rhoy.reshape((self.rhoy.size, )) 
            n_rd = self.rhoy.shape
            
                
            rr = ones(n_rd)
            if n_rd[0] != 79:

                calc_spt = expand_dims(self.calc_spt[0], axis=0)
                calc_sit = expand_dims(self.calc_sit[:,0], axis=1)
                
                BeSi_1 = einsum('ki, j->ikj',calc_spt, rr)
                
                BeSi_2 = einsum('ik,j->ikj',calc_sit, rr)

            else:
                BeSi_1 = einsum('ki, j->ikj',self.calc_spt, rr)

                BeSi_2 = einsum('ik,j->ikj',self.calc_sit, rr)

        elif self.rhoy.shape[1] == 1:
            self.rhoy = self.rhoy.reshape((self.rhoy.size, ))    
            if self.tze.shape[1] == 1:
                self.tze = self.tze.reshape((self.tze.size, ))
            n_rd = self.rhoy.shape

            rr = ones(n_rd)
            if n_rd[0] != 79:
                calc_spt = expand_dims(self.calc_spt[0], axis=0)
                calc_sit = expand_dims(self.calc_sit[:,0], axis=1)
                
                BeSi_1 = einsum('ki, j->ikj',calc_spt, rr)
                
                BeSi_2 = einsum('ik,j->ikj',calc_sit, rr)

            else:
                BeSi_1 = einsum('ki, j->ikj',self.calc_spt, rr)

                BeSi_2 = einsum('ik,j->ikj',self.calc_sit, rr)

        else:
            n_rd = self.rhoy.shape
            rr = ones(n_rd)
            
            if n_rd[0] != 79:
                calc_spt = expand_dims(self.calc_spt[0], axis=0)
                calc_sit = expand_dims(self.calc_sit[:,0], axis=1)
                
                BeSi_1 = einsum('ki, j->ikj',calc_spt, rr)
                
                BeSi_2 = einsum('ik,j->ikj',calc_sit, rr)

            else:

                BeSi_1 = einsum('ki, kj->ijk',self.calc_spt, rr)

                BeSi_2 = einsum('ik,kj->ijk',self.calc_sit, rr)


        BeSi = BeSi_1 + BeSi_2

        BeSi_array = zeros((BeSi.shape[0], BeSi.shape[1], BeSi.shape[2], self.n_ions))

        for kk in xrange(self.n_ions):
            BeSi_array[:,:,:,kk] = BeSi

        BeSi = BeSi_array

        # Calculate the x-ray emission values for each line of sight (index i), 
        # energy (index j), and spatial point (index k). 
        # The einsum function is used equvalently to an outer product. 
        # The string input designates the dimension of each input matrix and 
        # the output matrix. The output matrix is each element of each input 
        # vector or matrix multiplied by each element of each other vector or 
        # matrix. 
        n_rd = self.rhoy.shape
        if self.rhoy.size == n_rd[0]:
            n_rd = (n_rd[0], 1)
        else:
            n_rd = self.rhoy.shape

        exp_f_b_imp = zeros((self.n_en, n_rd[1], n_rd[0], self.n_ions, ))
        exp_f_b_i_BeSi = zeros((self.n_en, n_rd[1], n_rd[0], self.n_ions, ))
        en_int_b_imp = zeros((self.n_ions, n_rd[1], n_rd[0]))

        exp_f_r_imp = zeros((self.n_qs, self.n_ions, self.n_en, n_rd[1], n_rd[0]))
        exp_f_r_i_BeSi = zeros((self.n_qs, self.n_ions, self.n_en, n_rd[1], n_rd[0]))
        en_int_r_imp = zeros((self.n_ions, n_rd[1], n_rd[0]))

        if self.gff_Born_approx:
            gff = self.sxr_gff(tep)
            calc_gff = 0

        else:
            gam2_t = self.Ry / tep
            calc_gff = 1
            
        gfb = 1.0
        
        if calc_gff:
            gam2_t = gam2_t[:,:,0]
            gam2 = zeros((self.Z.size, gam2_t.shape[0], gam2_t.shape[1]))
            for kk in xrange(self.Z.size):
                gam2[kk] = self.Z[kk] ** 2.0 * gam2_t

            gff = self.sxr_gff(gam2)
        
        if rr.ndim == 1:
            rr = rr.reshape(rr.size, 1)

        f1 = einsum('i,kj->ijk',self.en, rr) 

        f2 = einsum('i,kji->ijk',self.calc_cc, tep)  
        
        f3 = einsum('i,kji->ijk',self.calc_cc, log((nep)/sqrt(tep)))

        f7 = einsum('i,kj->ijk',self.f_nm, rr)

        f8 = f2
        
        sum_f = zeros((self.calc_cc.shape[0], gff.shape[2], gff.shape[1], gff.shape[0]))

        for kk in xrange(self.n_ions):
            if self.ind_maj == kk:
                nip_kk = self.nmaj[:,:,0]
            else:
                nip_kk = nip[:,:,kk]

            #f0 = einsum('i,kj->ijk', self.calc_cc, gff[kk])

            zip1 = nip_kk * self.Z[kk] ** 2.0
            zip1 = log(zip1)

            f9 = einsum('i,kj->ijk', self.calc_cc, zip1)

            sum_f[:,:,:,kk] = -1.0* f1/f2  + f3 + f7/f8 +f9# + log(f0)


        a = isnan(sum_f)
        b = where(a == True)
        sum_f_inf = sum_f.copy()
        sum_f_inf[b] = -inf
        
        exp_f_b_imp = K_sxr * exp( sum_f_inf.clip(min=-690.0))
        exp_f_b_i_BeSi = K_sxr * exp( (sum_f_inf + BeSi ).clip(min=-690.0))

        Z_list = self.Z.tolist()
        n_min_list = self.n_min.tolist()
        b_e_list = self.b_e.tolist()
        chi_list = self.chi.tolist()
        tep = tep.reshape((n_rd))

        ein2 = einsum('i,kj->ijk',self.calc_cc, tep)

        for kk in xrange(self.n_ions):

            for q in xrange(self.n_qs):
                if q == 0:
                            
        # Number of holes in the lowest unfilled shell
                    
                    csi = 2.0 * n_min_list[kk] ** 2.0 - b_e_list[kk]

                    chi_Te = divide(chi_list[kk],ein2)

                    chi_Te = expand_dims(chi_Te, 3)
                    
                    f_r_imp = add( ( add( sum_f ,( log( ( csi * chi_Te )/( n_min_list[kk] ** 3.0 ) ) ) ) ), ( chi_Te ) )
                    
                    ind_q = idl_where(self.en <= chi_list[kk])
                    
                else:
                    
                    csi = n_min_list[kk] + q

                    chi = ( Z_list[kk] ** 2.0 * self.Ry ) / ( csi ** 2.0 )

                    chi_Te = chi / ein2

                    chi_Te = expand_dims(chi_Te, 3)

                    f_r_imp = add(add(sum_f , ( log( ( 2.0 * chi_Te ) / csi ) ) ) , ( chi_Te ))
                    ind_q = idl_where(self.en <= chi)
                    
                '''
                a = isnan(f_r_imp)
                b = where(a == True)
                f_r_imp_inf = f_r_imp.copy()
                f_r_imp_inf[b] = inf
                print f_r_imp.shape
                '''

                exp_f_r_imp[q, kk, :,:,:] = (K_sxr * exp( (f_r_imp[:,:,:,kk]).clip(max= 690.0, min=-690.0 ) )) #gfb * 

                # Apply the transmission function of the filter and the response function of
                # the detector to the spectrum

                exp_f_r_i_BeSi[q, kk, :,:,:] = (K_sxr * exp( (f_r_imp[:,:,:,kk] + BeSi[:,:,:,kk]).clip(max=690.0, min=-690.0) )) #gfb * 
  

                if isinstance(ind_q, int)  == False:

                    ind_qq = array([])
                    for b in xrange(len(ind_q)):
                        if ind_q[b] == 1:
                            ind_qq = add_float(ind_qq, b)
                        else:
                            pass
                    for c in xrange(len(ind_qq)):
                        exp_f_r_imp[q, kk, ind_qq[c], ] = 0.0

                        exp_f_r_i_BeSi[q, kk, ind_qq[c], ] = 0.0
                indF = idl_where(isnan(f_r_imp[:,:,:,kk]).any() == 1 or isinf(f_r_imp[:,:,:,kk]).any() == 1)
                if type(indF) != int:
                    if indF != -1:
                        print 'NaN found in F_R_IMP'
                        print 'kk=' + str(kk).strip() + '       q=' + str(q).strip()



        sp_r_imp = sum( exp_f_r_imp, axis = 0)
        sp_r_i_BeSi = sum( exp_f_r_i_BeSi, axis = 0) 
        
        # energy integrals (bremsstrahlung and recombination) for each single ion
        ein = einsum('i,kj->ijk',self.calc_dent, rr)
        for kk in xrange(self.n_ions):

            en_int_b_imp[kk, ] = sum( exp_f_b_i_BeSi[:,:,:,kk] * ein, axis = 0)

            en_int_r_imp[kk, ] = sum( sp_r_i_BeSi[kk,:,:,:] * ein, axis = 0)
  
        sp_brems = sum(exp_f_b_imp, axis = 3)
        sp_b_BeSi = sum(exp_f_b_i_BeSi, axis = 3)
        
        sp_recomb = sum(sp_r_imp, axis = 0)
        sp_r_BeSi = sum(sp_r_i_BeSi, axis = 0)
        sp_tot = sp_brems + sp_recomb
        
            # The energy integral is simply the SUM over energies of the power coefficient
            
        en_int_brems = sum( sp_b_BeSi * ein, axis = 0)
        en_int_recomb = sum( sp_r_BeSi * ein, axis = 0)

        en_int = en_int_brems + en_int_recomb
        
            # Output structures
            
        spectrum = {
                        'sp_brems': sp_brems,
                        'sp_b_BeSi': sp_b_BeSi,
                        'sp_b_imp': exp_f_b_imp,
                        'sp_b_i_BeSi': exp_f_b_i_BeSi,
                        
                        'sp_recomb': sp_recomb,
                        'sp_r_BeSi': sp_r_BeSi,
                        'sp_r_imp': sp_r_imp,
                        'sp_r_i_BeSi': sp_r_i_BeSi,
                        
                        'spectrum': sp_tot,
                        'energy': self.en,  
                        
                        'en_int': en_int
                    }
            
        en_integral = {
                            'en_int_brems': en_int_brems,
                            'en_int_b_imp': en_int_b_imp,
                            
                            'en_int_recomb': en_int_recomb,
                            'en_int_r_imp': en_int_r_imp
                        }
        
        return spectrum
        #sum over impurities

    def sxr_gff(self, gam2, **kwargs):
        from numpy import array, zeros, where, log10
        from sxr_lib import getFirstValue, keyword_set
        from math import sqrt, pi
        Born = getFirstValue(kwargs, "Born", self.gff_Born_approx)
        
        gff = zeros((gam2.shape))

        if keyword_set(Born):
            gff = gff + 2.0 * sqrt(3.0) / pi
        else:
            al = array([1.43251926625281, 3.50626935257777E-1,\
                        4.36183448595035E-1, 6.03536387105599E-2,\
                        3.66626405363100E-2 ])
            bl = array([1.00000000000000, 2.92525161994346E-1,\
                        4.05566949766954E-1, 5.62573012783879E-2,\
                        3.33019373823972E-2 ])
            ah = array([1.45481634667278,-9.55399384620923E-2,\
                        1.46327814151538E-1,-1.41489406498468E-2,\
                        2.76891413242655E-3 ])
            bh = array([1.00000000000000, 3.31149751183539E-2,\
                        1.31127367293310E-1, -1.32658217746618E-2,\
                        2.74809263365693E-3 ])
            ind0 = where(gam2 < 1.0E-6)

            n0 = len(ind0[0])

            if n0 > 0:
                gff[ind0] = 1.102635 + 1.186 * sqrt(gam2[ind0]) + 0.86 * gam2[ind0]
            ind1 = where(gam2 > 1.0E10)

            n1 = len(ind1[0])

            if n1 > 0:
                gff[ind1] = 1.0 + gam2[ind1] ** (1.0/3.0)  

            g = log10(gam2)
            ind2 = where((g>= - 6) & (g <= 0.8))
            n2 = len(ind2[0])
            if n2 > 0:
                gg = g[ind2]
                P = ( ( ( al[4] * gg + al[3] ) * gg + al[2] ) * gg + al[1]) * gg + al[0]
                Q = ( ( ( bl[4] * gg + bl[3] ) * gg + bl[2] ) * gg + bl[1]) * gg + bl[0]
                gff[ind2] = P / Q
                
            ind3 = where((g >= 0.8) & (g <= 10.0))
            n3 = len(ind3[0])
            if n3 > 0:
                gg = g[ind3]
                P = ( ( ( ah[4] * gg + ah[3] ) * gg + ah[2] ) * gg + ah[1]) * gg + ah[0]
                Q = ( ( ( bh[4] * gg + bh[3] ) * gg + bh[2] ) * gg + bh[1]) * gg + bh[0]
                gff[ind3] = P / Q

        
        return gff
    
    def sxr_Be(self, en, be_thick, perc_comp_i, **kwargs):
        '''
         Transmission function of a Be foil of thickness t (microns) as a function of
         energy. Impurities can be added in the foil composition

        INPUT:
        SPESSBE thickness in microns of the foil. This can be a single value or an array
        of two elements of the type [ total thickness, thickness of the second foil ]
        (useful if the foil is a stack of filters with two different compositions).
        If USE_BESSY if set only the first value will be used.
        If not defined the program will exit with error

        OPTIONAL INPUT:
        EN energy array. This will be automatically defined by the program if the input
        value is undefined or the keyword DEFINE_E is set

        PERC_COMP chemical percent of each compound in the Be foil, one or two rows array.
        First row=first foil, second row=second foil. If it's a single row the second
        row is automatically set to zero.
        The number of columns must be the same of ST.FILTER.PERC_COMP (i.e. ST.FILTER.N_COMP,
        see SXR_X_EMISSION; this number is presently 23).
        PERC_COMP has been defined two times:
        - Using the proper Be foil impurities fractional abundance for Cr, Fe, Ni from
        J.Fournelle (microprobe analysis of July 2014), Materion datasheet for everything
        else (IF-1 foils) and lower-limits of Pb, Mo (Se, W, U, P=0). Only first row is
        defined; second one has zeroes. This refers to the foils used in 2010-2014, and
        these are the (in)famous Be-Zr filters.
        - From 5-Feb-2015 new Be foils are used, with a different composition: no Zr,
        Fe and Ni within the maximum limit of IF-1. This is the default

        USE_MU if set the program will use the tabulated values of the mass coefficients mu
        from the http://henke.lbl.gov/optical_constants/filter2.html website. This is
        the default. If set to zero will use the old Bardet formula from report
        EUR-CEA-FC-1038 (1980), and will work only for a 100% pure Be foil; or it
        will use the tabulated data from BESSY calibration (if USE_BESSY=1)

        USE_BESSY if set uses the Be foils transmission functions calibrated
        at BESSY, October 2015 (default=0). SPESSBE must be one of the measured
        foil: 857, 413, 702, 348 microns.
        ONLY FOR MST tomography

        EFF_MU if set the effective MU coefficients (determined from Bessy calibrated
        data) will be used. If OLD_FOILS=0 the effective MU of the 2015- foils will
        be used to define the transmission function, if OLD_FOILS=1 the effective MU
        of the Be+Zr foils (2012-2014) will be used. Default=0.
        ONLY FOR MST tomography

        OLD_FOILS if set the effective MU of the Be+Zr foils (2012-2014) will be used.
        Default=0.
        ONLY FOR MST tomography

        EFF_FILT_DENS if set an effective density of 1.675 g/cm3 will be used for the
        filter instead of 1.848 g/cm3 (for pure Be foils) or the weighted average
        of the densities of impurities. Default=0.
        ONLY FOR MST tomography

        NOTE1. USE_MU, USE_BESSY, EFF_MU, OLD_FOILS and EFF_FILT_DENS can be also changed using SXR_X_EMISSION
        NOTE2. Either USE_MU or USE_BESSY should be specified. If both, USE_MU has priority
        NOTE3. EFF_MU has priority on USE_MU and USE_BESSY

        DSX3_FILT_DENS if specified and different than zero this value will be used as foil
        density in the calculations. If not specified or equal to zero the nominal density
        (1.848 g/cm3) will be used. A 100% pure Be and saved beryllium MU coefficient will be considered.
        The keywords USE_MU, USE_BESSY and EFF_MU must be zero. Default=0.
        ONLY FOR DSX3 diagnostic, RFX-Mod

        OUTPUT:
        T transmission curve, same elements as EN

        TT transpose of T (will be used in SXR_X_EMISSION)
        '''
        from numpy import maximum, where, interp as interpret, squeeze, array, zeros, sum, exp, transpose, log, interp
        from sxr_lib import idl_where, keyword_set
        import datetime
        n_en = en.size

        
        
        if "define_E" in kwargs:
            define_E = kwargs.get('define_E')
            if keyword_set(define_E):
                en = self.geometry_en_mu
                interp = 0         #This flag will tell if the original mu coefficients
                                    #or BESSY data should be interpolated in the given
                                    #EN array or not.  MU are in fact defined in the
                                    #energy array SELF.EN_MU.  The same for the 
                                    #calibrated transmission functions
                                    
        elif n_en == 0:
            en = self.geometry_en_mu
            interp = 0         #This flag will tell if the original mu coefficients
                                    #or BESSY data should be interpolated in the given
                                    #EN array or not.  MU are in fact defined in the
                                    #energy array SELF.EN_MU.  The same for the 
                                    #calibrated transmission functions
                                    
        else:

            if n_en == self.n_en_mu:
                
                if sum(abs(en-self.geometry_en_mu)) <= 1.0:
                    interp = 0

                else:
                    interp = 1
                    
            else:
                interp = 1

        if 'use_mu' not in kwargs: 
            use_mu = self.use_mu
            
        else:
            use_mu = keyword_set(kwargs.get('use_mu'))
            
        if 'use_BESSY' not in kwargs: 
            use_BESSY = self.use_BESSY
            
        else:
            use_BESSY = keyword_set(kwargs.get('use_BESSY'))
            
        if 'eff_mu' not in kwargs: 
            eff_mu = self.eff_mu
            
        else:
            eff_mu = keyword_set(kwargs.get('eff_mu'))
            
        if 'old_foils' not in kwargs: 
            old_foils = self.old_foils
            
        else:
            old_foils = keyword_set(kwargs.get('old_foils'))
            
        if 'eff_filt_dens' not in kwargs: 
            eff_filt_dens = self.eff_filt_dens
        else:
            eff_filt_dens = keyword_set(kwargs.get('eff_filt_dens'))
            
        if 'DSX3_filt_dens' not in kwargs:
            DSX3 = self.DSX3_filt_dens
            
        else:
            DSX3 = keyword_set(kwargs.get('DSX3_filt_dens'))
        
        if eff_mu == 1:
            
            be_thick_list = self.be_thick.tolist()
            
            if eff_filt_dens:
                rho = self.eff_density
                
            else:
                rho = self.nominal_Be_density
                
            if old_foils:
                mu = self.mu_BeZr80
                
            else:
                mu = self.geometry_mu_BeFeb2015
    
            #Note: MU coefficients are in cm^2/g, so thickness must be converted from microns
            #to cm
    
            Tt = maximum((-1.0 * mu * rho * (be_thick_list[0] * 1e-4)), (-690.0))
            
            if interp:
                dum = interpret(en, self.en_mu, Tt)
    
            T = exp( Tt )
            
            Tt = transpose(Tt)
            dic = {'T': T, 'Tt': Tt}

    
        elif use_mu ==1:
            
            be_thick_list = be_thick.tolist()


            #check SXR_X_EMISSION. PERC_COMP has by default the Be foil impurities
            #fractional
            #abundance of the new filters (arrived in Nov 2014): no Zr, Fe, and Ni
            #within the
            #maximum limit of IF-1. Only first row is defined; second one has zeroes
        
        #PERC_COMP should have two rows, one for the 'main' Be foil and the second for
        #a different Be foil (i.e., in general, for two foils with different impuritise).
        #if it's a single row then this is a single Be foil; a second row of zeroes will
        #be automatically added.  The number of columns must be the same of 
        #SELF.PERC_COMP (i.e SELF.N_COMP)
            n_row = len(self.perc_comp[:,0])

            if n_row == 0:
                perc_comp = array([ self.perc_comp[0,]], [ self.perc_comp[0,]*0.0])
                
            else:
                perc_comp = self.perc_comp[0:2, ]
        # The same for thick_Be, it should be an array of two elements, the first is
        #the total thickness and the second the thickness of the second Be foil with
        #different impurity concentrations.  if it's a single value this is a single
        #Be foil and a thickness of zero microns will be automatically added for the
        #second value

            if self.be_thick.size == 1:
                thknssBe = array([ be_thick_list[0], 0.0])

            else:
                thknssBe = array([be_thick_list[0]-be_thick_list[1], be_thick_list[1]])
        #NOTE that from this point and on the program will define a local variable THKNSSBE
        #with the actual thicknesses of the two foil: [thickness of first filter, 
        # thickness of second filter ]
    
            n_tot_Be = 2L
    
        #use the data from the website http://henke.lbl.gov/optical_constants/filter2.html
        #and saved in the mu.sav file

            mu = zeros((n_tot_Be, self.n_en_mu), float)
            den_mu = zeros(n_tot_Be, float)
            rho = zeros(n_tot_Be, float)
            den_rho = zeros(n_tot_Be, float)
            
            en_shape = self.en.shape
            dum_null = zeros(en_shape)
            AW_list = self.AW.tolist()
            den_mu_list = den_mu.tolist()
            rho_comp_list = self.rho_comp.tolist()
            rho_list = rho.tolist()
            den_rho_list = den_rho.tolist()
            thknssBe_list = thknssBe.tolist()
        #define the total transmission function as the combination of the transmissions
        #of each impurity


            Tt = dum_null
            for b in xrange(n_tot_Be):
                #the filter is always considered as formed by two foils
                if thknssBe_list[b] != 0.0:
                    
                    #will do some calculations only if the thickness is different than 0!

                    #define the Be percent as 100%-(total percent of all impurities)
                    perc_comp[b, squeeze(self.ind_Be)] = 100.0 - sum(perc_comp[b, self.ind_noBe])

                    first = 1
                    
                    for c in xrange(self.n_comp):

                        AW_list = self.AW.tolist()
                        rho_comp_list = self.rho_comp.tolist()
                        
                        if perc_comp[b, c] != 0.0:
                            
                            if first:
                                mu[b,] = perc_comp[b,c] * AW_list[c] * self.geometry_mu[c, ]
                                den_mu_list[b] = perc_comp[b, c] * AW_list[c]
                                den_mu = array(den_mu_list)

                                rho_list[b] = perc_comp[b, c] * rho_comp_list[c]
                                rho = array(rho_list)
                                
                                den_rho_list[b] = perc_comp[b, c]
                                den_rho = array(den_rho)
                        
                                first = 0
                                
                            else:
                                mu[b, ] = mu[b, ] + perc_comp[b, c] * AW_list[c] * self.geometry_mu[c, ] 
                                                               
                                den_mu_list[b] = den_mu_list[b] + perc_comp[b, c] * AW_list[c]
                                den_mu = array(den_mu_list)
                                
                                rho_list[b] = rho_list[b] + perc_comp[b, c] * rho_comp_list[c]
                                rho = array(rho_list)
                                
                                den_rho_list[b] = den_rho_list[b] + perc_comp[b, c]
                                den_rho = array(den_rho)
                                
                    mu[b, ] = mu[b, ] / den_mu_list[b]

                    if eff_filt_dens:
                        rho_list[b] = self.eff_density
                        rho = array(rho_list)
                        
                    else:
                        rho_list[b] = rho_list[b] / den_rho_list[b]
                        rho = array(rho_list)
                        
                        #Note: MU coefficients are in cm^2/g, so thickness must be converted
                        #from microns to cm

                    dum = maximum((-1.0 * mu[b, ] * rho_list[b] * (thknssBe_list[b] * 1e-4)), (-690.0))
     
                    if interp:
                        dum = interpret(en, self.geometry_en_mu, dum)

                else:
                    dum = dum_null
                    
                Tt = Tt + dum

            T = exp( Tt )
            Tt = transpose(Tt)  
            
            dic = {'T': T, 'Tt': Tt}

        elif use_BESSY == 1:
        #Four Be foils have been calibrated:
        #SXR-A thick (857um; old foils)        SXR-A thin(413um; old foils)
        #702um (new foils) 348 um (new foils)
        #Thus thick_Be must be one of these values. if not: fatal error

            be_thick_list = self.be_thick.tolist()
            Bessy_Be = array([857.0, 413.0, 702.0, 348.0])
            indBe = idl_where(abs(self.be_thick[0] - self.Bessy.Be) < 1.5) #+/- 1.5 micron tolerance
            if isinstance(indBe, int)  == False:
                indBe = where(abs(self.be_thick[0] - self.Bessy.Be) < 1.5)
                Bessy_T = self.geometry_Bessy_T
                dum = squeeze(Bessy_T[:,indBe[0]])
                        
            else:
                if indBe == -1:
                    raise ValueError( "be_thick =" + str(be_thick_list[0], 2).strip() + 'um is not one of the\
                    calibrated foils at BESSY (857, 413, 702, 348 um)' )
        
            if interp:
                interpfunc = interpret(en, self.geometry_Bessy_E, dum)
                dum = interpfunc(en)
            
            T = dum

            Tt = transpose(log(T))
            dic = {'T': T, 'Tt': Tt}

        elif DSX3 == 1:
            
            # THIS IS ONLY FOR RFX-MOD
            # Use the specified value for the foil density. This is necessary for
            # calculations with 'mixed' foils for the DSX3 diagnostic
            
            # Note: MU coefficients are in cm^2/g, so thickness must be converted from microns to cm

            Tt = maximum( (-1.0 * self.geometry_mu[self.ind_Be, :] * self.DSX3_filt_dens[0] * (self.be_thick[0] * 1e-4 ) ), -690.0)
            
            if interp:
                Tt = interpret(en, self.geometry_en_mu, Tt)
                
            T = exp(Tt)
            Tt = transpose(Tt)
            dic = {'T': T, 'Tt': Tt}
        else:
        
        #Bardet formula, Report EUR-CEA-FC-1038 (1980)
        #This is 100% pure Be foil
        #Maintained here only for historical reasons
        
            be_thick_list = self.be_thick.tolist()
            
            Tt = maximum(( ( -1.0 * 6.305e7 * be_thick_list[0]) / (en ** 2.92)), (-690.0))
            
            T = exp( Tt )
            
            Tt = transpose( Tt )
            
            dic = {'T': T, 'Tt': Tt}
        
        return dic
    
    def sxr_tr_Si(self, si_thick, **kwargs):
        '''
        PURPOSE:
        Calculate the transmission function of a specified Si thickness
        '''
        from sxr_lib import idl_where, keyword_set
        from numpy import interp as interpelet, maximum, exp, einsum, where, zeros
                
        en = self.en
        n_en = en.size
        n_en_mu = self.geometry_en_mu.size

        if 'define_E' in kwargs:
            define_E = kwargs.get("define_E")
            
        else:
            define_E = 0
            
        if keyword_set(define_E) or n_en == 0:
            #define an array from the .sav file
            en = self.en_mu
            interp = 0
            
        else:
            
            if n_en == n_en_mu and abs(en - self.geometry_en_mu).sum() <= 1.0:
                
                interp = 0

            else:
                interp = 1

        if 'use_mu' not in kwargs: 
            use_mu = self.use_mu

        else:
            use_mu = keyword_set(kwargs.get('use_mu'))
        if use_mu:
            ind_Si = idl_where(self.compound == 'Si')
            
            if isinstance( ind_Si, int ):
                
                if ind_Si == -1:
                    
                    raise ValueError( 'GIVEN COMPOUND IS NOT FOUND IN COMPOUND ARRAY')
                
            else:
                ind_Si_index = []
                
                for index, item in enumerate(ind_Si):
                    
                    if item == 1:
                        v = index
                        ind_Si_index.append(v)
                
                if isinstance(si_thick, int) or isinstance(si_thick, float):
                    Tt = maximum((-1.0 * self.geometry_mu[ind_Si_index,:] * self.rho_comp[ind_Si_index]\
                    * (si_thick * 1e-4)), (-690.0))
                else:
                    Tt = zeros((si_thick.size,en.size))
                    for i in xrange(si_thick.size):
                        Tt[i,:] = maximum((-1.0 * self.geometry_mu[ind_Si_index,:] * self.rho_comp[ind_Si_index]\
                    * (si_thick[i] * 1e-4)), (-690.0))

            if interp:
                Tt = Tt.reshape(501,)
                Tt = interpelet(en, self.geometry_en_mu, Tt)

        else:
            
            ind_1 = where(en <= 148.0)[0]
            
            if ind_1.size != 0.0:
                Tt[:,ind_1] = exp( 
                    maximum(((-0.441*12395.0**0.96 * 14.0**3.0 * 2.33/1.0e4) * 
                    einsum('i,j->ij', si_thick, 1.0/(en[ind_1]**0.96))),-690.0))
    
            ind_2 = (where( en > 148.0 ) and where( en <= 1839.0 ))[0]
            
            if ind_2.size != 0.0:
                Tt[:,ind_2] = exp( 
                    maximum((-5.33e-4*12395.0**2.74 * 14.0**3.03 * 2.33/1.0e4 )* 
                    einsum('i,j->ij', si_thick, 1.0/(en[ind_2]**2.74)),-690.0))
    
            ind_3 = where( en >= 1839.0)[0]
            
            if ind_3.size != 0.0:
                Tt[:,ind_3] = exp( 
                    maximum((-1.38e-2*12395.0**2.79 * 14.0**2.73 * 2.33/1.0e4) * 
                    einsum('i,j->ij', si_thick, 1.0/(en[ind_3]**2.79)),-690.0))

        T = exp(Tt)

        Tt = Tt

        return T
    
    def check_delta(self, delta_a, delta_h):
        """
        Check that the values of delta_a and delta_h satisfy necessary 
        conditions. Will return an error if any condition fails and will not 
        return anything if there are no problems.
        
        Parameters
        ----------
        delta_a : number
            shift in the magnetic axis
        delta_h : number 
            shift of the last closed flux surface (LCFS)
      
        Returns
        -------
        ValueError : Only if there are problems with delta_a or delta_h
        """
        
        from numpy import zeros
        
        # Normalize delta_a and delta_h to the radius.
        delta_a = delta_a / self.radius
        delta_h = delta_h / self.radius 
        
        # Positive delta_h
        if (delta_h >= 0.0) and (delta_h <= 1.0):
            if (delta_a >= 1.0) or (delta_a <= (2.0 * delta_h - 1.0)):
                raise ValueError("Control the value of delta_a!")
                
        elif delta_h > 1.0: 
            raise ValueError( "delta_h is too big!")
            
        # negative delta_h
        if (delta_h >= -1.0) and (delta_h < 0.0 ):
            if (delta_a <= -1.0) or (delta_a >= (2.0 * delta_h + 1.0)):
                raise ValueError( "Control the value of delta_a!" )

        elif delta_h < -1.0: 
            raise ValueError( "delta_h (negative) is too big!" )
        
        # If the values of delta_a and delta_h are both okay, then add 
        # coefficients (coeff) of the coordinate system to the self object.
        coeff = zeros(3, dtype=float)
        coeff[0] = delta_a
        coeff[1] = float(0.0)
        coeff[2] = (delta_h - delta_a) / ( (1.0 - abs(delta_h) )**2.0 )
        setattr(self,'coeff', coeff)    
   
        return

    def sxr_rho(self, x, y, **kwargs):
        from sxr_lib import keyword_set, np_array, idl_where
        from numpy import zeros, sqrt, where, clip, array
        if 'delta' in kwargs:
            delta = kwargs.get('delta')
            if keyword_set(delta):
                self.delta_a = delta[0]
                self.delta_h = delta[1]
                
                delta_a = delta[0] / self.radius
                delta_h = delta[1] / self.radius
                
                if delta_h >= 0.0 and delta_h <= 1.0:
                    if delta_a >= 1.0 or delta_a <= (2.0 * delta_h - 1.0):
                        print 'controllare i valori di delta_a!'
                    else:
                        if delta_h > 1.0:
                            print 'delta_h troppo grande!'
                    if delta_h >= -1.0 and delta_h < 0.0:
                        if delta_a <= -1.0 or delta_a >= (2.0 * delta_h + 1.0):
                            print 'controllare i volri di delta_a!'
                        else:
                            if delta_h < -1.0:
                                print 'delta_h (negativo) troppo grande!'
                
                self.coeff[0] = delta_a
                self.coeff[1] = 0.0
                self.coeff[2] = (delta_h - delta_a ) / ( ( 1.0 - abs(delta_h)) ** 2.0 )
                            
        if len(kwargs) == 0:
            xd = np_array(x)
            yd = np_array(y)
                    
            delta_a = self.delta_a / self.radius
            delta_h = self.delta_h
                    
            if (delta_a - delta_h) == 0.0:
                ro_x = xd - delta_a
                ro_y = yd
                ro = sqrt( ro_x ** 2.0 + ro_y ** 2.0)
            else:
                ro = xd
                dlt = 1.0 - 4.0 * self.coeff[2] * ( -xd + self.coeff[2] * (yd ** 2.0 )\
                    + delta_a )
                ind_lt0 = where(dlt < 0.0)
                ind_ge0 = where(dlt >= 0.0)
                if ind_lt0[0] != []:
                    ind_lt0 = where(dlt < 0.0)
                    ro[ind_lt0] = 100.0
                if ind_ge0[0] != []:
                    ind_ge0 = where(dlt >= 0.0)
                    ro_x = ( -1.0 + sqrt(dlt[ind_ge0]) ) / ( 2.0 * self.coeff[2])
                    ro_y = yd[ind_ge0]
                    ro[ind_ge0] = sqrt( ro_x ** 2.0 + ro_y ** 2.0 )
            ro_norm = ro / ( 1.0 - abs(delta_h) ) 
            
            ro_norm = ro_norm.clip(min = 1.0E-5, max = 1.0)
            ro = ro.clip(min = 1.0E-5, max = (1.0-abs(delta_h) ) )
            if isinstance(ro_norm, int) or isinstance(ro_norm, float):
                ro_norm = array(ro_norm)
            return ro_norm
                
                                
    def sxr_tzeta(self, x,y):

        from numpy import zeros, arctan2, maximum, sum as npsum, array
        from sxr_lib import diff_ang, sign
        
        # Calculate the shift of rho with respect to the horizontal axis. 
        # not normalized. Equivalent to sxr_delta_rho(roy) in previous versions
        dum = self.sxr_rho(x, y)
        delta_rho = maximum(npsum(array(
                                        [dum*0.0+self.coeff[0], 
                                        dum*self.coeff[1], 
                                        (dum**2.0)*self.coeff[2]]),
                                        axis=0),1.0e-4)


        xs = x - delta_rho  

        tzed = arctan2(y,xs)

        tzed = sign(tzed) * abs(maximum(abs(tzed), 1.0e-4))
        
        tzeta = diff_ang(tzed, zeros(tzed.shape))
        return tzeta