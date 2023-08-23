'''
Created on Feb 16, 2017

@author: lauersdorf
'''
from scipy.interpolate.interpolate import RegularGridInterpolator

def sxr_prof_te(rhoy, te, alpha_te, beta_te, del_te, del_th, tze, del_r, r0, th0, tabv_use_te, tabv_r_te=None, tabv_te=None):
        """
        Calculate the temperature at each spatial point along each line of 
        sight. From either Te0, alpha, and, beta, or by interpolating the 
        Tabv_te inputs.
        
        Returns
        -------
        out : ndarray, shape ''(d0, d1)''
            Temperatures (eV)
            d0 - number of lines of sight (n_p)
            d1 - number of spatial integral points (n_s)
        """
        
        from numpy import exp, log, maximum, interp, newaxis, expand_dims
        from sxr_lib import between
        from Islands import island
        import datetime
        rd = between(rhoy, 1.0e-4, 1.0)
 
        te = te[newaxis, newaxis, ...]
        alpha_te = alpha_te[newaxis, newaxis, ...]
        beta_te = beta_te[newaxis, newaxis, ...]
        rd = expand_dims(rhoy, axis=2)
        # Calculate the base Te profile


        if not tabv_use_te:

            tep = 1.0 - exp(maximum(alpha_te * log(rd), -690.0))
            
            tep = exp(maximum(beta_te*log(maximum(tep,1.0e-4)),-690.0))

            tep = maximum(1.0e-4, (tep * te))

        else:
            tep =  maximum(1e-4, interp(rd, tabv_r_te, tabv_te))  
        

        # Add a temperature island 
        tep = maximum(tep + island(rhoy, del_te, del_th, tze, del_r, r0, th0), 1.0)
        return tep

def sxr_prof_ne(rhoy, dens, alpha_ne, beta_ne, tze, del_ne, ne_r0, ne_th0, ne_del_r, ne_del_th, del_dens, dr, del_dr, tabv_use_dens, tabv_r_dens=None, tabv_dens=None):
    """
    Calculate the density at each spatial point along each line of sight, 
    from either dens, alpha_ne, and, beta_ne, or interpolating the 
    tabv_dens input.
        
    Returns
    -------
    out : ndarray, shape ''(d0, d1)''
        Densities (10**19 m**-3)
        d0 - number of lines of sight (n_p)
        d1 - number of spatial integral points (n_s)

        
    """
    from numpy import exp, log, maximum, interp, newaxis, expand_dims
    from sxr_lib import between
    from Islands import island_ne
    from Corona import corona_ne
    rd = between(rhoy, 1.0e-4, 1.0)
    dens = dens[newaxis, newaxis, ...]
    alpha_ne = alpha_ne[newaxis, newaxis, ...]
    beta_ne = beta_ne[newaxis, newaxis, ...]
    rd = expand_dims(rhoy, axis=2)
    
        # Calculate the smooth density profile

    if not tabv_use_dens:
        rrr = 1.0 - exp(maximum(alpha_ne * log(rd), -690.0))
        prof_ne = maximum(1e-4, dens * exp(maximum(beta_ne * log(maximum(rrr, 1e-4)), -690.0)))

    else:
        prof_ne =  interp(rd, tabv_r_dens,tabv_dens)  


        # Add a ne island or an ne ring

    prof_ne = prof_ne + island_ne(rhoy, tze, del_ne, ne_r0, ne_th0, ne_del_r, ne_del_th) + corona_ne(rhoy, del_dens, dr, del_dr)

    return prof_ne
def sxr_prof_nh(neutral_dens, rhoy):
    from numpy import less_equal
    # Return a non zero nH(rho,tze) only for rho<1
    # PROF_NH is always greater than 1e-4 to avoid numerical errors
    rhoy = rhoy.clip(min=1E-4, max=None)
    rhoy = less_equal(rhoy, 1.0)
    prof_nH = neutral_dens * rhoy
    
    # TEST of a neutral density profile very peaked at the edge
    #
    #ind = where( (rd ge 0.8) and (rd le 1.0), n_ind )
    #if n_ind gt 1 then $
    #prof_nH[ind] = St.emission.neutral_dens + ( rd[ind] - 0.8d ) * 5.0d * (1.0d3 - St.emission.neutral_dens )
    
    return prof_nH

def sxr_prof_ni(ni_ADAS, ind_11, imp_str_o, imp_str_e, n_imp_e, rhoy, ni_dens, alpha_ni, beta_ni, del_ni_dens, ni_dr, del_ni_dr, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr, tabv_use_ni_dens, nep=None, tep=None, nHp=None, tabv_r_ni_dens=None, tabv_ni_dens=None):
    """
    Calculate the impurity profile for each impurity species from either the
    alpha model or interpolating the tabv_ni_dens inputs for each spatial
    location along each line of sight.
                
    Returns
        -------
    out : ndarray, shape ''(d0, d1, d2)''
        Impurity densities (10**19 m**-3) 
        d0 - number of lines of sight (n_p)
        d1 - number of spatial integral points (n_s)
        d2 - number of impurity species present
        
    """
        
    from numpy import  interp, newaxis, minimum, empty, zeros, expand_dims, expand_dims, meshgrid, zeros, where, array, log, exp, around
    from sxr_lib import between, belong5
    from Corona import sxr_corona_ni
    from Islands import sxr_island_ni
    from scipy import interpolate
    import scipy.io
    import datetime
    try:
        from numexpr import evaluate
        use_numexpr = True
    except ImportError:
        use_numexpr = False

    mx_dens = 10.00000017
    mx_Te = 3200.0
    mx_denH = 10.0
    mn_denH = 0.01
    mn_dens = 0.01
    mn_Te = 1.0
    n_denH = 201
    n_dens = 401
    n_Te = 321
    Al_Charge_State_Frac_sav = scipy.io.readsav('SAV/Al_Charge_State_Frac.sav')
    ADAS = Al_Charge_State_Frac_sav['adas']
    Al_csf = ADAS['AL_CSF']

    #alpha_ni = alpha_ni[newaxis, newaxis, ...]

    #beta_ni = beta_ni[newaxis, newaxis, ...]

    #rd = expand_dims(rhoy, axis=2)
    '''
    if tep == None:
        tep = sxr_prof_te(rhoy, te, alpha_te, beta_te, del_te, del_th, tze, del_r, r0, th0, tabv_use_te, tabv_r_te, tabv_te)
    else:
        tep = tep
    
    if nep == None:
        nep = sxr_prof_ne(rhoy, dens, alpha_ne, beta_ne, tze, del_ne, ne_r0, ne_th0, ne_del_r, ne_del_th, del_dens, dr, del_dr, tabv_use_dens, tabv_r_dens, tabv_dens)
    else:
        nep = nep
    
    if nHp == None:
        nHp = sxr_prof_nh(neutral_dens, rhoy)
    else:
        nHp = nHp
    '''

    if n_imp_e == 0:
        prof_ni = zeros((rhoy.shape))
    else:
        
        # Return a non zero ni(rho,tze) only for rho<1
        
        prof_ni = zeros((rhoy.shape[0], rhoy.shape[1],n_imp_e))
        frac_11_ok = 0
        for kk in xrange(n_imp_e):
            ind_kk = where(array(imp_str_o) == imp_str_e[kk].lower())

            if not tabv_use_ni_dens:

                if belong5(imp_str_e[kk], array(['Al+10', 'Al+12', 'Al+13'])) and ni_ADAS and ind_11 != -1:
                    if frac_11_ok == 0:
                        frac_11_ok = 1
                    else:
                        frac_11_ok = 0
                    
                    if frac_11_ok:

                        csf = sxr_csf('Al', tep, nep, nHp, Al_csf)

                        #stop
                        #tep_fi = around(((((tep.clip(min=mn_Te)).clip(max=mx_Te))-mn_Te) / (mx_Te - mn_Te)) * (n_Te - 1.0))
                        
                        #nep_fi = around(((((nep.clip(min=mn_dens)).clip(max=mx_dens))-mn_dens) / (mx_dens - mn_dens)) * (n_dens - 1.0))
                        
                        #nHp_fi = around(((((nHp.clip(min=mn_denH)).clip(max=mx_denH))-mn_denH) / (mx_denH - mn_denH)) * (n_denH - 1.0))

                        #a = zeros((3, 25))
                        #a[0] = tep_fi[0][:,0]
                        #a[1] = nep_fi[0][:,0]
                        #a[2] = nHp_fi[0]

                        #frac_10 = trilinear_interp(Al_csf[0][0].T, a)
                        #frac_11 = trilinear_interp(Al_csf[0][1].T, a)
                        #frac_12 = trilinear_interp(Al_csf[0][2].T, a)
                        #frac_13 = trilinear_interp(Al_csf[0][3].T, a)

                        #stop
                        T = 8.0 * csf[:,:,0] + 9.0 * csf[:,:,1] + 10.0 * csf[:,:,2] + 11.0 * csf[:,:,3] + 12.0 * csf[:,:,4] + 13.0 * csf[:,:,5] 

                        Al11_r = minimum((11.0 * prof_ni[:,:,ind_11]), nep[:,:,0] * (11.0 * csf[:,:,3] / T ))
                        
                        Al11_r = Al11_r / 11.0
                        
                        frac_11_ok = 1
                    
                    index = long(imp_str_e[kk][3:5])
                    if index == 8:
                        frac = csf[:,:,0]
                    elif index == 9:
                        frac = csf[:,:,1]
                    elif index == 10:
                        frac = csf[:,:,2]
                    elif index == 12:
                        frac = csf[:,:,4]
                    elif index == 13:
                        frac = csf[:,:,5]
                    
                    fatt = frac / csf[:,:,3]
                    
                    prof_ni[:,:,kk] = Al11_r * fatt

                else:
                    rd = between(rhoy, 1.0e-4, 1.0)

                    rrr = 1.0 - exp(alpha_ni[ind_kk[0]] * log(rd).clip(min = -690, max=None))

                    prof_ni[:,:,kk] = ni_dens[ind_kk[0]] * exp( beta_ni[ind_kk[0]] * log(rrr.clip(min = 1E-4, max = None)).clip(min = -690, max = None))

                    prof_ni[:,:,kk] = prof_ni[:,:,kk] + sxr_corona_ni(imp_str_o, imp_str_e, rhoy, del_ni_dens, ni_dr, del_ni_dr, is_del_ni, kk) + sxr_island_ni(rhoy, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr, imp_str_o, imp_str_e, kk)
                    # density profile of Al+11 has been already evaluated: it is always the first in the list of impurities
            else:
                raise ValueError('------- TABV for impurity densities NOT YET AVAILABLE !!! ---------')
                

    return prof_ni

                        # convert Te, electron density and neutral density profiles in fractional indices
                        # in the arrays ADAS.Te, ADAS.dens and ADAS.denH
                        
                        
    

    
    #if use_numexpr:
    #    nip = evaluate('ni_dens*(1.0-rd**alpha_ni)**beta_ni')
    #else:
    #    nip = ni_dens*(1.0-rd**alpha_ni)**beta_ni

    
    #nip = nip + sxr_corona_ni(rhoy, del_ni_dens, ni_dr, del_ni_dr, en_step=None) + sxr_island_ni(rhoy, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr)
    #return nip

def sxr_csf(element, tep, nep, nHp, Al_csf):
    from numpy import around, where, array, zeros
    import scipy.io
    import os
    import scipy.interpolate
    bessy2015_sav = scipy.io.readsav('SAV/ADAS_CSF.sav')
    c = bessy2015_sav['adas']
    d = c['elem_array'][0].tolist()
    mx_dens = 10.00000017
    mx_Te = 3200.0
    mx_denH = 10.0
    mn_denH = 0.01
    mn_dens = 0.01
    mn_Te = 1.0
    n_denH = 201
    n_dens = 401
    n_Te = 321
    #Al_Charge_State_Frac_sav = scipy.io.readsav('SAV/Al_Charge_State_Frac.sav')
    #ADAS = Al_Charge_State_Frac_sav['adas']
    #Al_csf = ADAS['AL_CSF']
    
    elem_array = c['elem_array'][0]
    
    tep_fi = around(((((tep.clip(min=mn_Te)).clip(max=mx_Te))-mn_Te) / (mx_Te - mn_Te)) * (n_Te - 1.0))
                        
    nep_fi = around(((((nep.clip(min=mn_dens)).clip(max=mx_dens))-mn_dens) / (mx_dens - mn_dens)) * (n_dens - 1.0))
                        
    #nHp_fi = around(((((nHp.clip(min=mn_denH)).clip(max=mx_denH))-mn_denH) / (mx_denH - mn_denH)) * (n_denH - 1.0))
    
    elem = element.upper()
    elem_csf = elem + '_CSF'
    s = c[elem_csf][0]


    if elem == 'B':
        csf = bilinear_interpolate(s['CSF'][0][5, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
    elif elem == 'C':
        csf = bilinear_interpolate(s['CSF'][0][6, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
    elif elem == 'N':
        csf = bilinear_interpolate(s['CSF'][0][7, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
    elif elem == 'O':
        csf = zeros((tep.shape[0],tep.shape[1], 3))
        csf[:,:,0] = bilinear_interpolate(s['CSF'][0][6, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,1] = bilinear_interpolate(s['CSF'][0][7, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,2] = bilinear_interpolate(s['CSF'][0][8, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        
    elif elem == 'AL':
        csf = zeros((tep.shape[0],tep.shape[1], 6))
        csf[:,:,0] = bilinear_interpolate(s['CSF'][0][8, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,1] = bilinear_interpolate(s['CSF'][0][9, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,2] = bilinear_interpolate(s['CSF'][0][10, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,3] = bilinear_interpolate(s['CSF'][0][11, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,4] = bilinear_interpolate(s['CSF'][0][12, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
        csf[:,:,5] = bilinear_interpolate(s['CSF'][0][13, :, :], tep_fi[:,:,0], nep_fi[:,:,0])
    s = 0
    
    return csf
    '''
    stop
    #for i in xrange(len(d)):
    #    d[i] = d[i].upper() 

    #ind = where(array(d) == elem)
    

'''
'''
def sxr_prof_ni(rhoy, ni_dens, alpha_ni, beta_ni, del_ni_dens, ni_dr, del_ni_dr, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr, tabv_use_ni_dens, en_step=None, tabv_r_ni_dens=None, tabv_ni_dens=None):
    """
    Calculate the impurity profile for each impurity species from either the
    alpha model or interpolating the tabv_ni_dens inputs for each spatial
    location along each line of sight.
                
    Returns
        -------
    out : ndarray, shape ''(d0, d1, d2)''
        Impurity densities (10**19 m**-3) 
        d0 - number of lines of sight (n_p)
        d1 - number of spatial integral points (n_s)
        d2 - number of impurity species present
        
    """
        
    from numpy import  interp, newaxis, empty, expand_dims
    from sxr_lib import between  
    from Corona import sxr_corona_ni
    from Islands import sxr_island_ni
    try:
        from numexpr import evaluate
        use_numexpr = True
    except ImportError:
        use_numexpr = False
        
        # Use the tabulated ni density values if available
    if tabv_use_ni_dens:
        # Initialize the impurity profile matrix
        rd = rhoy
        nip = empty(en_step.shape)[newaxis,...] * \
                    empty(rd.shape)[...,newaxis]

        for i in xrange(en_step.size): 
            nip[:,:,i] = interp( between(rd, 1e-4, 1.0), \
                                tabv_r_ni_dens[i,:],
                                tabv_ni_dens[i,:])  
        
        # if not, then calculate the profile
    else:
            # broadcast each array into a 3d shape to eliminate the for loop  

        ni_dens = ni_dens[newaxis, newaxis, ...]

            
        alpha_ni = alpha_ni[newaxis, newaxis, ...]

        beta_ni = beta_ni[newaxis, newaxis, ...]

        #rd = rhoy[...,...,newaxis]
        rd = expand_dims(rhoy, axis=2)
            # **If you have not installed the numexpr module, you should** #
        if use_numexpr:
            nip = evaluate('ni_dens*(1.0-rd**alpha_ni)**beta_ni')
        else:
            nip = ni_dens*(1.0-rd**alpha_ni)**beta_ni
    
   ### Old for loop calculation of the profile for each impurity species and
   ### direct calculation of the impurtiy ring structure
   #     # Initialize the impurity profile matrix
   #     nip = empty(self.en_step.shape)[newaxis,...] * \
   #              empty(rd.shape)[...,newaxis]
   #
        #nip_corona = zeros(nip.shape) 
        #for i in xrange(self.en_step.size): 
        #        
        #        ## Direct Calculation of the impurity ring.
        #        if self.del_ni_dr[i] != 0.0:
        #            nip_exp = exp(-1.0 * ((rd - self.ni_dr[i])**2.0) / 
        #                          (2.0 * self.del_ni_dr[i]**2.0))
        #        else:
        #            nip_exp = 0.0 
        #        
        #        nip_corona =  (self.del_ni_dens[i] * nip_exp)
        #        
                


    nip = nip + sxr_corona_ni(rhoy, del_ni_dens, ni_dr, del_ni_dr, en_step=None) + sxr_island_ni(rhoy, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr)
    return nip
'''
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


def bilinear_interpolate(im, x, y):
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def trilinear_interp(input_array, indices):
    import numpy as np
    output = np.empty(indices[0].shape)
    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]
    
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
#Check if xyz1 is beyond array boundary:

    x1[np.where(x1==input_array.shape[0])] = x0.max()
    y1[np.where(y1==input_array.shape[1])] = y0.max()
    z1[np.where(z1==input_array.shape[2])] = z0.max()
    

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (input_array[x0,y0,z0]*(1-x)*(1-y)*(1-z) +
             input_array[x1,y0,z0]*x*(1-y)*(1-z) +
             input_array[x0,y1,z0]*(1-x)*y*(1-z) +
             input_array[x0,y0,z1]*(1-x)*(1-y)*z +
             input_array[x1,y0,z1]*x*(1-y)*z +
             input_array[x0,y1,z1]*(1-x)*y*z +
             input_array[x1,y1,z0]*x*y*(1-z) +
             input_array[x1,y1,z1]*x*y*z)

    return output
