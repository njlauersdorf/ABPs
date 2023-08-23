'''
Created on Feb 16, 2017

@author: lauersdorf
'''

def island(rhoy, del_te, del_th, tze, del_r, r0, th0):
    """
    Temperature island structure.
        
    Calculate the increase of temperture from an island structure, at each 
    spatial location (n_s) of each line of sight (n_p). 
    The island is defined by a gaussian structure of height self.del_te,
    centered at (self.r0, self.th0), with radial width self.del_te_r, and 
    angular width self.del_th.        
                                
    Returns
    -------
    out : ndarray, shape ''(n_p, n_s)'' 
        Increase in temperatures (eV).
    """                                            
    from numpy import zeros, cos, sin, ones, exp, pi
    from sxr_lib import idl_where, diff_ang
        
    # Definitions to increase speed and readability

    rho_shape = rhoy.shape
        
    n_island = del_te.size
    island_tot = zeros((rho_shape[0], rho_shape[1], n_island))
    ind = idl_where(del_th == 0)

    if isinstance(ind, int):
        if ind != -1:
            x1 = rhoy * cos(tze)
            y1 = rhoy * sin(tze)

    # Loop over each island structure (if multiple)
    for iso in xrange(n_island):
        # make sure that the increase in temperature is not 0
        if del_te[iso] != 0.0:
            # check the width to define the type of island.
            if del_th[iso] == 0.0:
                x2 = ones(rho_shape)*r0[iso]*cos(th0[iso])

                y2 = ones(rho_shape)*r0[iso]*sin(th0[iso])

                                
                r = ( (x1-x2)**2 + (y1-y2)**2)**(1.0/2.0)

                island = del_te[iso] * \
                            exp( (-1.0 * (r**2.0) )/(2.0*del_r[iso]**2.0))           
  
            else:
                    # Calculate the part of the island in rho
                    # Old method - the gaussian is truncated at the center of 
                    # the plasma (this happens when the radial widths of the 
                    # islands are large enough.
                    
                island_rho = del_te[iso] * \
                            exp(  (-1.0 * (rhoy - r0[iso])**2.0 ) / \
                                    (2.0 * del_r[iso]**2.0) )

                    # Calculate the part of the island in tzeta               
                tze_mom = diff_ang(tze, th0[iso])
                tze_mom = tze_mom - 2.0 * pi *  idl_where(tze_mom > pi) 
    
                island_tze = exp(  (-1.0 * (tze_mom**2.0) ) / \
                                    ( 2.0 * del_th[iso]**2.0) )

                    # Calc the island as a product of the rho and tzeta parts                            
                island = island_rho * island_tze

        else:
            island = zeros(rho_shape)

        island_tot[:,:,iso] = island

    return island_tot

def island_ne(rhoy, tze, del_ne, ne_r0, ne_th0, ne_del_r, ne_del_th):
    """
    Calculate the increase of density from an island structure, at each 
    spatial location (n_s) of each line of sight (n_p). 
    The island is defined by a gaussian structure of height self.del_ne,
    and centered at (self.ne_r0, self.ne_th0), with radial width self.del_ne_r,  
    angular width self.ne_del_th. 
        
    Returns
    -------
    out : ndarray, shape ''(n_p, n_s)'' 
        Densities (10**19 m**-3)
    """
    from numpy import zeros, cos, sin, ones, exp, pi
    from sxr_lib import idl_where, diff_ang
        
    rho_shape = rhoy.shape
    n_isole = del_ne.size
    isola_tot = zeros((rho_shape[0], rho_shape[1], 1))

    # Calculate the increase for each element of del_ne
    for iso in xrange(n_isole):
        # Return nothing if the value of del_ne is 0
        if del_ne[iso] != 0.0:
            # Check the value of ne_del_th to define the type of island
            if ne_del_th[iso] == 0.0:
                x1 = rhoy * cos(tze)
                y1 = rhoy * sin(tze)
                x2 = ones(rho_shape)*ne_r0[iso]*cos(ne_th0[iso])
                y2 = ones(rho_shape)*ne_r0[iso]*sin(ne_th0[iso])
                                    
                r = ((x1-x2)**2 + (y1-y2)**2)**(1.0/2.0)
                isola = del_ne[iso] * \
                            exp((-1.0 * (r**2.0))/(2.0*ne_del_r[iso]**2.0))      

            else:
                # Calculate the part of the island in rho
                    
                # Old method - in the center of the plasma the Gaussian is
                # truncated (this happens to the radial widths of islands
                # large enough
                isola_rho = del_ne[iso] * \
                            exp((-1.0 *(rhoy - ne_r0[iso])**2.0) / 
                                    (2.0 * ne_del_r[iso]**2.0) 
                                    )
                # Calculate the part of the island in tzeta            
                tze_mom = diff_ang(tze, ne_th0[iso])

                a = idl_where(tze_mom > pi)
                if isinstance(a, int):
                    if a == -1:
                        size_tze_mom = tze_mom.shape
                        a = zeros(size_tze_mom)
                tze_mom = tze_mom - 2.0 * pi *  a
                isola_tze = exp(  (-1.0 * (tze_mom**2.0) ) / \
                                   ( 2.0 * ne_del_th[iso]**2.0) )  
                # Calculate the product of the island in rho and tzeta
                isola = isola_rho * isola_tze
        else:
            isola = zeros(rho_shape)
            
        isola_tot[:,:,0] = isola_tot[:,:,0] + isola 
    return isola_tot

def sxr_island_ni(rhoy, tze, is_del_ni, is_dr, ni_del_th, is_ni_dr, ni_th_dr, imp_str_o, imp_str_e, kk):
    """   
    Calculate the increase of impurity density from an island structure, 
    at each spatial location (n_s) of each line of sight (n_p), for each
    impurity species (n_ni). The island is defined by a gaussian structure of 
    height self.del_ni, and centered at (self.ni_r0, self.ni_th0), with 
    radial width self.del_ni_r, angular width self.ni_del_th. 
        
    Returns
    -------
    out : ndarray, shape ''(n_p, n_s, n_ni)'' 
        Impurity densities (10**19 m**-3)
    """
    from numpy import zeros, cos, sin, ones, exp, pi, where, array, expand_dims
    from sxr_lib import idl_where, diff_ang
        
    rho_shape = rhoy.shape 
    n_isole = is_del_ni.size
    isola_tot = zeros((rho_shape[0], rho_shape[1]))
    ind_kk = where(array(imp_str_o) == imp_str_e[kk].lower())

    # Calculate the increase for each element of del_ni
    #for iso in xrange(n_isole):
        # Return nothing if the value of del_ni is 0
    if is_del_ni[ind_kk] != 0.0:
            # Check the value of ni_del_th to define the type of island
        if ni_del_th[ind_kk] == 0.0:
            x1 = rhoy * cos(tze)
            y1 = rhoy * sin(tze)
            x2 = ones(rho_shape)*is_ni_dr[ind_kk]* \
                            cos(ni_th_dr[ind_kk])
            y2 = ones(rho_shape)*is_ni_dr[ind_kk]* \
                            sin(ni_th_dr[ind_kk])
                                    
            r = ((x1-x2)**2 + (y1-y2)**2)**(1.0/2.0)
            isola = is_del_ni[ind_kk] * \
                            exp((-1.0 * (r**2.0))/(2.0*is_dr[ind_kk]**2.0))      
                            
        else:
                # Calculate the part of the island in rho
                    
                # Old method - in the center of the plasma the Gaussian is
                # truncated (this happens to the radial widths of islands
                # large enough
            isola_rho = is_del_ni[ind_kk] * \
                            exp((-1.0 *(rhoy - is_ni_dr[ind_kk])**2.0) / 
                                    (2.0 * is_dr[ind_kk]**2.0) 
                                    )
                # Calculate the part of the island in tzeta            
            tze_mom = diff_ang(tze, ni_th_dr[ind_kk])
            tze_mom = tze_mom - 2.0 * pi *  idl_where(tze_mom > pi)
            isola_tze = exp(  (-1.0 * (tze_mom**2.0) ) / \
                                   ( 2.0 * ni_del_th[ind_kk]**2.0) )                       
                    
                # Calculate the product of the island in rho and tzeta
            isola = isola_rho * isola_tze
                    
    else:
        isola = zeros(rho_shape)

    return isola
