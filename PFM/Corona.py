'''
Created on Feb 16, 2017

@author: lauersdorf
'''

def corona_ne(rhoy, del_dens, dr, del_dr):
    """
    Density ring structure.
        
    Calculate the additional density from a density ring with increase 
    self.del_dens at a radial location of self.dr, and width self.del_dr. 
    The output gives the additional increase at each spatial location (n_s) 
    for each line of sight (n_p).
        
    Returns
    -------
    out : ndarray, shape ''(n_p, n_s)'' 
        Increase in densities (10**19 m**-3)
    """
    from numpy import expand_dims, zeros, exp
    try:
        from numexpr import evaluate
        use_numexpr = True
    except ImportError:
        use_numexpr = False

    # Return 0 if the increase in density self.del_dens is 0.
    if del_dens != 0.0:
            
        if use_numexpr:
            del_dens = expand_dims(del_dens, axis=0)
            del_dens = expand_dims(del_dens, axis=0)
            rd = expand_dims(rhoy, axis=2)
            dr = expand_dims(dr, axis=0)
            dr = expand_dims(dr, axis=0)
            del_dr = expand_dims(del_dr, axis=0)
            del_dr = expand_dims(del_dr, axis=0)
                
            cor_ne = evaluate('del_dens*exp(-1.0*((rd-dr)**2.0) / \
                                    (2.0*del_dr**2.0))')  
        else:
            del_dens = expand_dims(del_dens, axis=0)
            del_dens = expand_dims(del_dens, axis=0)
            rd = expand_dims(rhoy, axis=2)
            dr = expand_dims(dr, axis=0)
            dr = expand_dims(dr, axis=0)
            del_dr = expand_dims(del_dr, axis=0)
            del_dr = expand_dims(del_dr, axis=0)
                
            cor_ne = del_dens*exp(-1.0*((rd-dr)**2.0) / \
                                    (2.0*del_dr**2.0)) 
    else: 
        rd = expand_dims(rhoy, axis=2)
        cor_ne = zeros((rd.shape))
  
    return cor_ne

def sxr_corona_ni(imp_str_o, imp_str_e, rhoy, del_ni_dens, ni_dr, del_ni_dr, is_del_ni, kk):
    """
    Create an additional impurity ring to add to the impurity spectrum. 
    (Obsolete now that the impurity spectrum is calculated using a forward
    model)
                
    Returns
    -------
    out : ndarray, shape ''(d0, d1, d2)''
        Impurity densities (10**19 m**-3)
        d0 - number of lines of sight (n_p)
        d1 - number of spatial integral points (n_s)
        d2 - number of impurity species present
        
    """
        
    from numpy import exp,zeros, newaxis, expand_dims, where, array
    from sxr_lib import idl_where
    try:
        from numexpr import evaluate
        use_numexpr = True
    except ImportError:
        use_numexpr = False
        
    rho_shape = rhoy.shape 
    n_isole = is_del_ni.size
    ind_kk = where(array(imp_str_o) == imp_str_e[kk].lower())
    corona_tot = zeros((rho_shape[0], rho_shape[1], n_isole))
    if del_ni_dens[0] != 0.0:
        use_numexpr = 0
        if use_numexpr:
            del_ni_dens = del_ni_dens[ind_kk[0]]
            ni_dr = ni_dr[ind_kk[0]]
            del_ni_dr = del_ni_dr[ind_kk[0]]
            cor_ni = evaluate('(del_ni_dens) * exp((-1.0 * ( rd.clip(min = 1E-4) - ni_dr ) ** 2.0 ) /\
                    ( 2.0 * del_ni_dr ** 2.0 ) ).clip(min = -690) * (idl_where(rd < 1.0 ) )')

        else:   
            rd = rhoy
            cor_ni = (del_ni_dens[ind_kk[0]]) * exp((-1.0*(rd.clip(min = 1E-4) - ni_dr[ind_kk[0]] ) ** 2.0 ) /\
                    ( 2.0 * del_ni_dr[ind_kk[0]] ** 2.0 ) ).clip(min = -690) * (idl_where(rd < 1.0 ) )


    else:
        cor_ni = zeros((rhoy.shape))
        
    return cor_ni
