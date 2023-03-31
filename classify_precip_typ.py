# Only for data below melting layer (below 1 km approx), so only use the first radar elevation.
# Originally formulated for CAPPIs, but I could try with PPIs.

def classify_precip_typ(zh, x_coords, y_coords, cradius_relation=0, con_th=40, radius=11):
    """
    # Raint type classification
    # --------------------------

    For the classification of the precipitation type only data
    below the melting layer should be used.

    [Steiner et al. 1995]
    [https://doi.org/10.1175/1520-0450(1995)0342.0.CO;2]

    # Input
    # -----
        zh ::: reflectivity field in dbz. numpy.array (no xarray)
        x_coords/y_coords ::: x/y kartesien coord in km
        cradius_relation ::: convectiv radius intenity (0:small, 1:medium, 2:large)
        con_th ::: convective threshold in dBz

    # Output
    # ------
        precip_type ::: precipitation type
        1 = stratiform
        2 = convective

    """
    import numpy as np
    import wradlib as wrl

    def calc_convective_radiaus(mean_zh_bg, rel=cradius_relation):
        """
        Calculation of the convective radius

        Steiner et al. 1995 Fig 6.
        """

        th  = [[-100, -100, -100], [30, 25, 20], [35, 30, 25], [40, 35, 30], [45, 40, 35], [100, 100, 100]]
        rad = [1, 2, 3, 4, 5]

        convective_radius = np.ones_like(mean_zh_bg)*np.nan

        for i in range(5):
            convective_radius[(mean_zh_bg>th[i][rel]) & (mean_zh_bg<=th[i+1][rel])]=rad[i]

        return convective_radius
    
    def calc_delz(zh_bg):
        """
        Calculation of delZ
        Steiner et al. 1995 (2)
        """
        delzh = np.ones_like(zh_bg)*np.nan
        delzh[zh_bg<0]=10
        delzh[(zh_bg>=0)&(zh_bg<43)]=10 - (zh_bg[(zh_bg>=0)&(zh_bg<43)]**2)/180.
        delzh[zh_bg>=42.43]=0

        return delzh
    
    # mask zh, x, y
    zh_dummy  = np.ones_like(zh)*np.nan
    m = ~np.isnan(zh)
    # zh, x_coords, y_coords = zh[m], x_coords[m], y_coords[m]
    zh, x_coords, y_coords = zh[m], x_coords[m], y_coords[m]

    # define dummy rain typ array
    rtyp = np.ones_like(zh).ravel()*np.nan

    # Any grid poit with Z > Zh_threshold dBz is convectiv
    rtyp[zh.ravel()>con_th]=2

    #calculation of background refl.
    zh_bg = np.ones_like(zh.ravel())*np.nan

    #xy_coords
    xy_coords = np.stack((x_coords.ravel(), y_coords.ravel()), axis=1)

    # Radius in km used for calc of background refl (11km)
    r = radius

    # Calculate zh_bg
    for ix in range(xy_coords.shape[0]):
        dist = np.sqrt(((xy_coords[ix,0]-xy_coords[:,0])**2)+((xy_coords[ix,1]-xy_coords[:,1])**2) )
        zh_bg[ix] = np.nanmean(wrl.trafo.idecibel(zh.ravel()[dist<=r]))    
                                                          
    # trafo refl in decibel
    zh_bg = wrl.trafo.decibel(zh_bg)  
    
    # calc convective_radius in km 
    cr = calc_convective_radiaus(zh_bg)
    
    # calc delz
    delzh = calc_delz(zh_bg)
        
    # convectiv if exceeding peakedness
    rtyp[zh.ravel()-zh_bg >= delzh]=2
    
    
    # Define new rtype
    new_rtype = rtyp.copy()

    # define rtype for data in convective radius
    for i in range(xy_coords.shape[0]):
        if rtyp[i]==2:
            dist = np.sqrt(((xy_coords[i,0]-xy_coords[:,0])**2)+((xy_coords[i,1]-xy_coords[:,1])**2) )
            new_rtype[dist<=cr[i]] = 2
            
    # stratiform if not convective
    new_rtype[new_rtype!=2]=1

    # reshape
    precip_type = new_rtype.reshape(zh.shape)

    zh_dummy[m] = precip_type

    return zh_dummy
