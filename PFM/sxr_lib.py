'''
Created on Feb 16, 2017

@author: lauersdorf
'''
'''
import sys
sys.path.insert(0, '/path/to/application/app/folder')

import file
'''
def vect_0_1(n, no_1=False):
    """
    Return a numpy array of n elements between 0 and 1
    
    Parameters
    ----------
    n : int
    no_1 : bool, optional
        If True, the value 1 will not be included in the array
    
    Returns
    -------
    out : array
        n values between 0 and 1 (with or without the value 1)
                
    """
    
    from numpy import arange    
    
    if no_1:
        d = float(n)
    else:    
        d = float(n) - 1.0
    v = arange(n) / d
    
    return v

def sign(value):
    """
    Return a numpy array, the same shape as value, with the sign of each
    element (-1 if negative, 1 if positive, 0 if zero)
    
    Parameters
    ----------
    value : array_like
    
    Returns
    -------
    out : array_like
        the sign of each element of value
    """
    from numpy import array

    value = array(value)
    indx =  array(value!=0)
    value[indx] = value[indx] / abs(value[indx])
    
    return value

   
def diff_ang(ang1, ang2):
    from numpy import pi,array
    """
    Return the angle between the two input angles, oriented from the second
    to the first. ang1 or ang2 can be single valued, lists, or numpy arrays.
    If both are lists or arrays, they must be the same shape and the angle
    differences are returned elementwise. If one is an array and the other 
    is single valued, then all the angle differences are in reference to the
    single value.
        
    Parameters
    ----------
    ang1 : array_like
        the angle(s) (radians) measuring to
    ang2 : array_like
        the angle(s) (radians) measuring from 
    
    Returns
    -------
    out : array, length ''max(ang1.shape, ang2.shape)''
        the angle differences (radians)
   
    Note
    -------
    diff_ang(pi, [pi, 1.5*pi]) is equivalent to 
    diff_ang([pi, pi], [pi, 1.5*pi])
    """
    ang = array(ang1)-array(ang2)
    ang = ang % (2*pi)
    return ang

                   
def getFirstValue(kwargs, key, defaultValue):
    from numpy import array
    """
    Takes a python dictionary, kwargs, and returns the value of 'key' if it 
    is single valued and only the first value if 'key' is multivalued. Or if
    'key' is not in the dictionary, it returns 'defaultvalue'.
    
    The outputs are always returned as a numpy arrays.
        
    Parameters
    ----------
    kwargs : dict
        where the object property (key) is taken out of
    key : str
        what property to take from kwargs
    defaultValue : 
        any value or string that is returned if 'key' is not in the kwargs   
    
    Returns
    -------
    out : single valued array 
        either the fist value of 'key' if 'key' is in kwargs or 
        the first value in defaultValue if 'key' is not in kwargs
    
    """
    val = kwargs.get(key,defaultValue)
    
    try:
        if len(val)>1: 
            return np_array(val[0])
        elif len(val) == 1:
            return array(val)
    except: 
        return np_array(val)      

                      

    
def np_array(val):
    """
    Returns val as a numpy array if it is multivalued and converts val to a
    numpy array of length 1 if it is singlevalued. 
    
    Reasoning
    ---------
    Just defining val = array(val) will give a list of length 0 if val is a
    float or int but here, if val is a single value then array([val]) is 
    returned which is still a numpy array but with length 1, not length 0. 
        
    Parameters
    ----------
    val : array_like 
    
    Returns
    -------
    out : numpy array
       numpy array with a length at least 1 
     
    """
    from numpy import array
    if hasattr(val, "__len__"):
        return array(val)
    else:
        return array([val])
    
    
def idl_where(condition, complement=None):
    """
    Equivalent to the idl 'where' function. Returns an array of the same shape
    as condition, with a 1 where it is true and 0 where it is false.
        
    Parameters
    ----------
    condition : array_like, bool
        When True yield 1, when False yield 0
        If all false, return -1
    Returns
    -------
    out : ndarray, same shape as 'condition' with a 1 where 'condition' was True
         and a 0 where 'condition' was False
    """
    from numpy import where,ones,zeros
   
    indicies = where(condition, ones(condition.shape), zeros(condition.shape))    
    if indicies.any() == 0:
        indicies = -1
    return indicies
    """
    if complement:
        complement = []
        for index in indicies:
            if indicies[index] == False:
                complement.append(index)
        return complement
    return indicies
    """

def between(A, lowerLimit, upperLimit):
    """
    Returns the array (lowerLimit < A < upperLimit) so that all values
    in A less that lowerLimit are replaced with lowerLimit and all values 
    greater than upperLimit are replaced with upperLimit
        
    Parameters
    ----------
    A : array_like
        The values to apply the floor and ceiling cutoffs
    lowerLimit : number
        the floor cutoff for values of A
    upperLimit : number
        the ceiling cutoff for values of A
        
    Returns
    -------
    out : an array of the shape as 'A' with all values less than 'lowerLimit'
        changed to 'lowerLimit' and all values above 'upperLimit' changed to 
        'upperLimit'
    """
    from numpy import maximum, minimum
    
    vect = minimum(maximum(A, lowerLimit), upperLimit)
    
    return vect
    

    
class sxrGeometry:
    """
    This class contains beryllium thickness, impact parameter, impact angle, 
    and incident angle on the silicon, for each of the 80 SXR diodes.
    """
    def __init__(self):
        from numpy import array
        import scipy.io
        from time import time
    #   The minor radius of MST
        self.radius = 0.52
        
        mu_sav = scipy.io.readsav('SAV/mu.sav')
        bessy2015_sav = scipy.io.readsav('SAV/bessy2015.sav')
        '''
        Al_Charge_State_Frac_sav = scipy.io.readsav('SAV/Al_Charge_State_Frac.sav')
        ADAS = Al_Charge_State_Frac_sav['adas']
        self.mx_Te = ADAS['MX_TE']
        self.mn_Te = ADAS['MN_TE']
        self.n_Te = ADAS['N_TE']
        self.n_dens = ADAS['N_DENS']
        self.mx_dens = ADAS['MX_DENS']
        self.mn_dens = ADAS['MN_DENS']
        self.mx_denH = ADAS['MX_DENH']
        self.mn_denH = ADAS['MN_DENH']
        self.n_denH = ADAS['N_DENH']
        self.Al_csf = ADAS['AL_CSF']
        '''
        #print ADAS.dtype.names
        
        bessy2015_sav = bessy2015_sav['bessy']
        
    # The angle of incident x-rays on the silicon diodes
        self.angle_fact = array([
            0.562187, 0.455616, 0.3366750, 0.206992, 0.0698860, 0.0698860,
            0.206992, 0.336675, 0.4556160, 0.562187, 0.5621870, 0.4556160, 
            0.336675, 0.206992, 0.0698860, 0.069886, 0.2069920, 0.3366750, 
            0.455616, 0.562187, 0.3365650, 0.206874, 0.0697632, 0.0700091, 
            0.207111, 0.336785, 0.4557160, 0.562276, 0.6562570, 0.7383810, 
            0.336942, 0.207259, 0.0701416, 0.069652, 0.2067880, 0.3365040, 
            0.455478, 0.562080, 0.6560990, 0.738255, 0.5621870, 0.4556160, 
            0.336675, 0.206992,0.06988600, 0.069886, 0.2069920, 0.3366750,
            0.455616, 0.562187, 0.5621870, 0.455616, 0.3366750, 0.2069920, 
            0.0698860,0.0698860, 0.206992, 0.336675, 0.455616, 0.562187, 
            0.336565, 0.206874,0.0697632, 0.0700091, 0.207111, 0.336785, 
            0.455716, 0.562276, 0.656257, 0.738381, 0.336942, 0.207259, 
            0.0701416, 0.0696520, 0.206788, 0.336504,0.455478, 0.562080, 
            0.656099, 0.738255]) + array([ -2.98023e-07, -3.57628e-07,
            -1.78814e-07, 1.93715e-07 , 0.00000 , 0.00000, 1.93715e-07, 
            -1.78814e-07,-3.57628e-07, -2.98023e-07, -2.98023e-07, -3.57628e-07, 
            -1.78814e-07, 1.93715e-07 , 0.00000 , 0.00000, 1.93715e-07, 
            -1.78814e-07, -3.57628e-07,-2.98023e-07, 2.68221e-07, 1.93715e-07, 
            -2.98023e-08, -3.72529e-08,-1.49012e-07, 3.87430e-07, -8.94070e-08, 
            -5.96046e-08, 5.96046e-08,-5.96046e-08, 1.78814e-07, -1.04308e-07, 
            -3.72529e-08, -1.49012e-08,-3.12924e-07, -1.19209e-07, 3.87430e-07,
            3.57628e-07, 5.96046e-08,0.00000, -2.98023e-07, -3.57628e-07,
            -1.78814e-07, 1.93715e-07, 0.00000,0.00000, 1.93715e-07, 
            -1.78814e-07, -3.57628e-07, -2.98023e-07,-2.98023e-07, -3.57628e-07,
            -1.78814e-07, 1.93715e-07, 0.00000 , 0.00000, 1.93715e-07,
            -1.78814e-07, -3.57628e-07, -2.98023e-07, 2.68221e-07, 1.93715e-07,
            -2.98023e-08, -3.72529e-08, -1.49012e-07, 3.87430e-07,-8.94070e-08, 
            -5.96046e-08, 5.96046e-08, -5.96046e-08, 1.78814e-07, -1.04308e-07, 
            -3.72529e-08, -1.49012e-08, -3.12924e-07, -1.19209e-07, 3.87430e-07,
            3.57628e-07, 5.96046e-08 , 0.00000])

    # The impact parameter for each diode
        self.impact_p = array([0.455010, 0.422147, 0.379831, 0.327601, 
            0.266417, 0.198905, 0.128902, 0.0604414, 0.00326962, 0.0603420, 
            0.175870, 0.120760, 0.0576637, 0.0120282, 0.0854655, 0.158671, 
            0.227481, 0.288657, 0.340528, 0.382937, 0.191081, 0.125095, 
            0.0530729, 0.0213637, 0.0939824, 0.161066, 0.220241,0.270641,
            0.312562, 0.346947, 0.147935, 0.0803607, 0.00746861, 0.0669813,
            0.138755, 0.204269, 0.261366, 0.309410, 0.348884, 0.380865, 
            0.455010,0.422147, 0.379831, 0.327601, 0.266417, 0.198905, 
            0.128902, 0.0604414,0.00326962, 0.0603420, 0.175870, 0.120760, 
            0.0576637, 0.0120282, 0.0854655,0.158671, 0.227481, 0.288657, 
            0.340528, 0.382937, 0.191081, 0.125095,0.0530729, 0.0213637, 
            0.0939824, 0.161066, 0.220241, 0.270641, 0.312562,0.346947, 
            0.147935, 0.0803607, 0.00746861, 0.0669813, 0.138755, 0.204269,
            0.261366, 0.309410, 0.348884, 0.380865]) + array([3.27826e-07, 
            -3.57628e-07,3.27826e-07, -1.19209e-07, 1.19209e-07, -2.23517e-07,
            3.57628e-07,-3.35276e-08, -3.95812e-09, 2.23517e-08, 2.08616e-07, 
            -2.60770e-07,2.60770e-08, -3.35276e-08, 4.47035e-08, -4.47035e-07,
            1.34110e-07,4.76837e-07, 2.98023e-08, 1.49012e-07, 4.02331e-07, 
            4.02331e-07, 0.00000,4.47035e-08, -2.23517e-08, 2.53320e-07, 
            -1.49012e-07, 2.38419e-07,5.96046e-08, 4.47035e-07, 2.23517e-07, 
            -7.45058e-09, 4.19095e-09,-2.23517e-08, -2.53320e-07, 1.04308e-07,
            3.27826e-07, -2.38419e-07,-2.68221e-07, 4.47035e-07, 3.27826e-07, 
            -3.57628e-07, 3.27826e-07,-1.19209e-07, 1.19209e-07, -2.23517e-07,
            3.57628e-07, -3.35276e-08,-3.95812e-09, 2.23517e-08, 2.08616e-07, 
            -2.60770e-07, 2.60770e-08,-3.35276e-08, 4.47035e-08, -4.47035e-07, 
            1.34110e-07, 4.76837e-07,2.98023e-08, 1.49012e-07, 4.02331e-07, 
            4.02331e-07, 0.00000, 4.47035e-08,-2.23517e-08, 2.53320e-07, 
            -1.49012e-07, 2.38419e-07, 5.96046e-08,4.47035e-07, 2.23517e-07, 
            -7.45058e-09, 4.19095e-09, -2.23517e-08,-2.53320e-07, 1.04308e-07,
            3.27826e-07, -2.38419e-07, -2.68221e-07, 4.47035e-07])
   
    # The impact parameter angle phi 
        self.impact_phi = array([0.169488, 0.0629166, 6.22716, 6.09748, 
            5.96037, 5.82060,5.68349, 5.55381, 2.29328, 2.18671, 0.169488, 
            0.0629166, 6.22716, 2.95589,2.81878, 2.67901, 2.54190, 2.41222, 
            2.29328, 2.18671, 1.51466, 1.38497,1.24786, 4.24968, 4.11258, 
            3.98290, 3.86397, 3.75741, 3.66343, 3.58131,4.65663, 4.52695, 
            4.38983, 1.10845, 0.971309, 0.841593, 0.722619,0.616017, 0.521998,
            0.439842, 0.169488, 0.0629166, 6.22716, 6.09748,5.96037, 5.82060,
            5.68349, 5.55381, 2.29328, 2.18671, 0.169488,0.0629166, 6.22716, 
            2.95589, 2.81878, 2.67901, 2.54190,2.41222,2.29328,2.18671, 1.51466, 
            1.38497, 1.24786, 4.24968,4.11258, 3.98290,3.86397,3.75741, 3.66343,
            3.58131, 4.65663, 4.52695, 4.38983, 1.10845,0.971309, 0.841593, 
            0.722619, 0.616017, 0.521998, 0.439842]) + array([ -3.42727e-07,
            -3.72529e-08, 1.43051e-06, -1.43051e-06, 2.38419e-06, 4.76837e-07,
            4.29153e-06, 1.43051e-06, -1.90735e-06, -3.09944e-06, -3.42727e-07,
            -3.72529e-08, 1.43051e-06, -4.29153e-06, -4.76837e-07, -2.38419e-06,
            1.43051e-06, -1.19209e-06, -1.90735e-06, -3.09944e-06, 2.50340e-06,
            1.43051e-06, 4.76837e-07, 9.53674e-07, -4.76837e-07, 4.76837e-06,
            4.05312e-06, 4.05312e-06, 3.09944e-06, -9.53674e-07, 2.38419e-06,
            -9.53674e-07, 1.43051e-06, -4.88758e-06, 4.76837e-07, 2.98023e-07,
            -1.78814e-07, -1.78814e-07, 1.78814e-07, 2.08616e-07, -3.42727e-07,
            -3.72529e-08, 1.43051e-06, -1.43051e-06, 2.38419e-06, 4.76837e-07,
            4.29153e-06, 1.43051e-06, -1.90735e-06, -3.09944e-06, -3.42727e-07,
            -3.72529e-08, 1.43051e-06, -4.29153e-06, -4.76837e-07, -2.38419e-06,
            1.43051e-06, -1.19209e-06, -1.90735e-06, -3.09944e-06, 2.50340e-06,
            1.43051e-06, 4.76837e-07, 9.53674e-07, -4.76837e-07, 4.76837e-06,
            4.05312e-06, 4.05312e-06, 3.09944e-06, -9.53674e-07, 2.38419e-06,
            -9.53674e-07, 1.43051e-06, -4.88758e-06, 4.76837e-07, 2.98023e-07,
            -1.78814e-07, -1.78814e-07, 1.78814e-07, 2.08616e-07])
    
    # The beryllium thicknesses (in microns)
        self.bethickness = array([857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 857.00, 
            857.00, 857.00, 857.00, 421.00, 421.00, 421.00, 421.00, 421.00,
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 421.00, 
            421.00, 421.00, 421.00])
        
        self.mu = mu_sav['mu']
        self.Bessy_T = bessy2015_sav['T'][0]
        self.mu_BeFeb2015 = mu_sav['mu_befeb2015']
        self.Bessy_E = bessy2015_sav['E'][0]
        self.mu_BeZr80 = mu_sav['mu_bezr80']
        self.en_mu = mu_sav['en_mu']
        self.compound = mu_sav['compound']
        self.rho_comp = mu_sav['rho_comp']
        self.aw = mu_sav['aw']
        self.perc_comp_BeFeb2015 = mu_sav['perc_comp_befeb2015']
    


def keyword_set(argument):
    if argument == None:
        argument = -1
        return argument
    if isinstance( argument, ( int, long) ) == True:
        if argument != 0:
            keyword_argument = 1
        else:
            keyword_argument = 0
    elif len(argument) == 0:
        keyword_argument = 0
    elif len(argument) == 1 and argument[0] == 0:
        keyword_argument = 0
    else:
        keyword_argument = 1

    argument = keyword_argument 
    return argument  

def findvalue(argument):
    
    if isinstance( argument, ( int, long ) ) == 0:
        if len(argument) > 0:
            if type(argument) is not list:
                argument = argument.tolist()
                argument = keyword_set(argument[0])
            else:
                argument = keyword_set(argument[0])
        else:
            argument = 1
    else:
        argument = keyword_set(argument)
    return argument

def findvalue2(argument):
    
    if isinstance( argument, ( int, long ) ) == 0:
        if len(argument) > 0:
            if type(argument) is not list:
                argument = argument.tolist()
                argument = keyword_set(argument[0])
            else:
                argument = keyword_set(argument[0])
        else:
            argument = 0
    else:
        argument = keyword_set(argument)
    return argument
def belong5(value, vector):
    out = 0

    for i in xrange(len(vector)):
        if vector[i] == value:
            out = 1
            return out
    return out

def belong(value, vector):
    
    out = 1
    if isinstance(value, int):
        ind = idl_where(idl_size(value) == vector)
        if isinstance(ind, int) == 1:
            if ind == -1:
                out = 0
        return out
    elif isinstance(value, str):
        ind = idl_where(idl_size(value) == vector)
        if isinstance(ind, int) == 1:
            if ind == -1:
                out = 0
        return out
    else:
        for i in xrange(0, value.size):
            ind = idl_where(idl_size(value[i]) == vector)
            if isinstance( ind, int ) == 1:
                if ind == -1:
                    out = 0
        return out

def add_float(arr, line, init=0):
    '''
    Purpose: Adds a new element to a one-dimensional array extending its dimension.
    The name says "float" but in reality 'the array will continue' to contain the
    data of the same type of ARR if the array was already 'defined (ie
    LINE is converted to the type of ARR - ARR and attention that if 'string
    and LINE and 'number, or vice versa, there' error), or become an array
    type if it is defined as LINE (ARR and 'indefinite or INIT and set
    '''
    from numpy import append, array
    if arr.size == 0 or keyword_set(init):
        arr = array(line)
    else:
        arr = append(arr, line)  
    return arr
def Save(object1):
    '''
    Purpose: saves an object by pickling it
    '''
    import cPickle
    f = file('store.pckl', 'wb')
    cPickle.dump(object1, f)
    f.close()
    print 'data saved'
def Load():
    '''
    Purpose: loads saved object
    '''
    import cPickle
    f = file('store.pckl', 'rb')
    calc = cPickle.load(f)
    return calc
def add_array_line(arr, line, init=0):
    '''
    Purpose: It adds a new line to an array.  The array will continue 'to hold the data of the 
    same type if ARR the array was already 'defined or become an array of type as LINE
    if it is defined (ARR and 'indefinite or INIT e'settata). LINE must be a vector of many 
    elements how many columns of ARR
    '''
    from numpy import vstack
    if arr.size == 0 or keyword_set(init):
        arr = line
    else:
        arr = vstack([arr, line])
    return arr
def idl_size(A):    
    
    if isinstance(A, int):
        b = 'int'
    if isinstance(A, bool):
        b = 'bool'
    if isinstance(A, long):
        b = 'long'
    if isinstance(A, str):
        b = 'str'
    else:
        b = 'nothing'
    return b
    
def exclude(A, S):
    '''
    Purpose: Find the indexes of the elements of an array A that are not in
    another array B. Return the indexes of the array operation A-B
    '''
    from numpy import unique, array
    # He sets the epsilon, namely the least variation data in IDL
    
    eps = 1e-8
    
    if len(S) == 0:
        raise ValueError('providing a vector to be excluded')
    else:
        #It removes repeated by S elements, so as not to have repeated indices in IND_OUT
        ind_out = array([])

        S1 = unique(S)

        #It checks if S is of the type string, integer, or long
        v = array(['bool', 'int', 'long', 'str'])
        if belong( idl_size(A), v ):
            use_eps = 0

        else:
            use_eps = 1
            
        n_A = len(A)

        # For all the elements of A store IND_OUT indexes of the elements not present in S1
        
        for i in xrange(0, n_A):
            
            if use_eps:
                ind = idl_where( (S1 > (A[i]-eps)) and (S1 < (A[i]+eps)) )
                
            else:
                ind = idl_where( S1 == A[i])

            if not hasattr(ind, "__len__"):
                
                if ind == -1:
                    ind_out = add_float(ind_out, i)
                    
    if ind_out.size == 0:
        ind_out = array(-1)

    return ind_out
