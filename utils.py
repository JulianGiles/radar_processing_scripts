#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 08:38:01 2023

@author: jgiles

Various utilities for processing weather radar data.
"""

import numpy as np
import xarray as xr
from scipy import ndimage
from functools import partial
import os
import datetime as dt
import glob
from multiprocessing import Pool


def Entropy_timesteps_over_azimuth_different_vars_schneller(ds, zhlin="zhlin", zdrlin="zdrlin", rhohvnc="RHOHV_NC", kdp="KDP_ML_corrected"):

    '''
    From Tobias Scharbach
    
    Function to calculate the information Entropy, to estimate the homogenity from a sector PPi or the whole 360° PPi
    for each timestep. Values over 0.8 are considered stratiform.
    
    The dataset should have zhlin and zdrlin which are the linear form of ZH and ZDR, i.e. only positive values
    (not in DB, use wradlib.trafo.idecibel to transform from DBZ to linear)


    Parameter
    ---------
    ds : xarray.DataArray
        array with PPI data.

    Keyword Arguments
    -----------------
    zhlin : str
        Name of the ZH linear variable

    zdrlin : str
        Name of the ZDR linear variable

    rhohvnc : str
        Name of the RHOHV noise corrected variable

    kdp : str
        Name of the KDP variable. Melting layer corrected KDP is better.

    Return
    ------
    entropy_all_xr : xarray.DataArray
        DataArray with entropy values for the four input variables

    ######### Example how to calculate the min over the entropy calculated over the polarimetric variables
    Entropy = Entropy_timesteps_over_azimuth_different_vars_schneller(ds)
    
    strati = xr.concat((Entropy.entropy_zdrlin, Entropy.entropy_Z, Entropy.entropy_RHOHV, Entropy.entropy_KDP),"entropy")        
    
    min_trst_strati = strati.min("entropy")
    ds["min_entropy"] = min_trst_strati
    
    '''
    n_az = 360

    Variable_List_new_zhlin = (ds[zhlin]/(ds[zhlin].sum(("azimuth"),skipna=True)))
    entropy_zhlin = - ((Variable_List_new_zhlin * np.log10(Variable_List_new_zhlin)).sum("azimuth"))/np.log10(n_az)
    entropy_zhlin = entropy_zhlin.rename("entropy_Z")
    
    Variable_List_new_zdrlin = (ds[zdrlin]/(ds[zdrlin].sum(("azimuth"),skipna=True)))
    entropy_zdrlin = - ((Variable_List_new_zdrlin * np.log10(Variable_List_new_zdrlin)).sum("azimuth"))/np.log10(n_az)
    entropy_zdrlin = entropy_zdrlin.rename("entropy_zdrlin")
    
    
    Variable_List_new_RHOHV = (ds[rhohvnc]/(ds[rhohvnc].sum(("azimuth"),skipna=True)))
    entropy_RHOHV = - ((Variable_List_new_RHOHV * np.log10(Variable_List_new_RHOHV)).sum("azimuth"))/np.log10(n_az)
    entropy_RHOHV = entropy_RHOHV.rename("entropy_RHOHV")
    
    
    Variable_List_new_KDP = (ds[kdp]/(ds[kdp].sum(("azimuth"),skipna=True)))
    entropy_KDP = - ((Variable_List_new_KDP * np.log10(Variable_List_new_KDP)).sum("azimuth"))/np.log10(n_az)
    entropy_KDP = entropy_KDP.rename("entropy_KDP")
    

    
    entropy_all_xr = xr.merge([entropy_zhlin, entropy_zdrlin, entropy_RHOHV, entropy_KDP ])    
    
    return entropy_all_xr
        


########## Melting Layer after Wolfensberger et al 2016 but refined for PHIDP correction using Silke's style from Trömel 2019


#### Functions
    
def normalise(da, dim):
    damin = da.min(dim=dim, skipna=True, keep_attrs=True)
    damax = da.max(dim=dim, skipna=True, keep_attrs=True)
    da = (da - damin) / (damax - damin)
    return da


def sobel(da, dim):
    axis = da.get_axis_num(dim)
    func = lambda x, y: ndimage.sobel(x, y)
    return xr.apply_ufunc(func, da, axis, dask='parallelized')


def ml_normalise(ds, moments=dict(DBZH=(10., 60.), RHOHV=(0.65, 1.)), dim='height'):
    assign = {}
    for m, span in moments.items():
        v = ds[m]
        v = v.where((v >= span[0]) & (v <= span[1]))
        v = v.pipe(normalise, dim=dim)
        assign.update({f'{m}_norm': v})
    return xr.Dataset(assign)


def ml_clean(ds, zh, dim='height'):
    # removing profiles with too few data (>93% of array is nan)
    good = ds[zh+"_norm"].count(dim=dim) / ds[zh+"_norm"].sizes[dim] * 100
    return ds.where(good > 7)


def ml_combine(ds, zh, rho, dim='height'):
    comb = (1 - ds[rho+"_norm"]) * ds[zh+"_norm"]
    return comb


def ml_gradient(ds, dim='height', ywin=None):
    assign = {}
    for m in ds.data_vars:
        # step 3 sobel filter
        dy = ds[m].pipe(sobel, dim)
        # step 3 smoothing
        # currently smoothes only in y-direction (height)
        dy = dy.rolling({dim: ywin}, center=True).mean()
        assign.update({f'{m}_dy': dy})
    return ds.assign(assign)


def ml_noise_reduction(ds, dim='height', thres=0.02):
    assign = {}
    for m in ds.data_vars:
        if '_dy' in m:
            dy = ds[m].where((ds[m] > thres) | (ds[m] < -thres))
            assign.update({f"{m}": dy})
    return ds.assign(assign)



def ml_height_bottom_new(ds, moment='comb_dy', dim='height', skipna=True, drop=True):

    hgt = ds[moment].idxmax(dim=dim, skipna=skipna, fill_value=np.nan).load()

        
    ds = ds.assign(dict( 
                        mlh_bottom=hgt))
    return ds


def ml_height_top_new(ds, moment='comb_dy', dim='height',skipna=True, drop=True):#, max_height=3000): 2000 für 16.11.2014

    hgt = ds[moment].idxmin(dim=dim, skipna=skipna, fill_value=np.nan).load()

    ds = ds.assign(dict( 
                    mlh_top=hgt))
    return ds




def melting_layer_qvp_X_new(ds, moments=dict(DBZH=(10., 60.), RHOHV=(0.65, 1.), PHIDP=(-90.,-70.)), dim='height', thres=0.02, xwin=5, ywin=5, fmlh=0.3, min_h=600, all_data=False, clowres=False):
    '''
    Function to detect the melting layer based on wolfensberger et al 2015 (https://doi.org/10.1002/qj.2672) and refined for delta bump removing by K. Mühlbauer and further refined by T. Scharbach

    Parameter
    ---------
    ds : xarray.DataArray
        array with QVP data.
        
    Keyword Arguments
    -----------------
    moments : dict
        Dictionary with values for normalization of every moment.

    dim : str
        Name of the height dimension.

    thres : float
        Threshold for noise reduction in the gradients of the normalized smoothed variables. Only values over thres are kept.

    xwin : int
        Minimum number of time steps for rolling median smoothing

    ywin : int
        Minimum number of height steps for rolling mean smoothing

    fmlh : float
        Tolerance value for melting layer height limits, above and below +-(1 +- flmh) from
        calculated median top and bottom, respectively.

    min_h : int, float
        Minimum height of usable data within the polarimetric profiles, in m. This is relative to
        sea level and not relative to the altitude of the radar (in accordance to the "z" coordinate 
        from wradlib.georef.georeference_dataset). The default is 600. 

    all_data : bool
        If True, include all normalized moments in the output dataset. If False, only output 
        melting layer height values. Default is False.

    clowres : bool
        If True, use an adaptation for low resolution data (e.g. 1 km range resolution in DWD's C-band). Default is False.

    Return
    ------
    ds : xarray.DataArray
        DataArray with input data plus variables for ML top and bottom and also normalized and derived variables.
    '''    
    zh = [k for k in moments if ("zh" in k.lower()) or ("th" in k.lower())][0]
    rho = [k for k in moments if "rho" in k.lower()][0]
    phi = [k for k in moments if "phi" in k.lower()][0]

    # step 1: Filter values below min_h and normalize
    ds0 = ml_normalise(ds.where(ds[dim]>min_h), moments=moments, dim=dim)

    # step 1a
    # removing profiles with too few data (>93% of array is nan)
    good = ml_clean(ds0, zh, dim=dim)

    # step 2
    comb = ml_combine(good, zh, rho, dim=dim)
    ds0 = ds0.assign(dict(comb=comb))

    # step 3 (and 8)
    ds0 = ml_gradient(ds0, dim=dim, ywin=ywin)

    # step 4 
    ds1 = ml_noise_reduction(ds0, dim=dim, thres=thres)
    #display(ds1)
    
    # step 5
    ds2 = ml_height_bottom_new(ds1, dim=dim, drop=False)
    ds2 = ml_height_top_new(ds2, dim=dim, drop=False)

    # step 6
    while xwin >1: # try: in case there are few timesteps reduce xwin until it is possible to compute
        try:
            med_mlh_bot = ds2.mlh_bottom.rolling(time=xwin, min_periods=xwin//2, center=True).median(skipna=True)     #min_periods=xwin//2
            med_mlh_top = ds2.mlh_top.rolling(time=xwin, min_periods=xwin//2, center=True).median(skipna=True)        # min_periods=xwin//2
            break
        except ValueError:
            xwin-=2
            
    # step 7 (step 5 again)
    above = (1 + fmlh) * med_mlh_top
    below = (1 - fmlh) * med_mlh_bot
    ds3 = ds1.where((ds1[dim] >= below) & (ds1[dim] <= above), drop=False)
    ds3 = ml_height_bottom_new(ds3, dim=dim, drop=False)
    ds3 = ml_height_top_new(ds3, dim=dim, drop=False)

    # ds3.mlh_bottom and ds3.ml_bottom derived according Wolfensberger et. al.

    # step 8 already done at step 3

    # step 9

    # ml top refinement
    # cut below ml_top
    ds4 = ds0.where((ds0[dim] > ds3.mlh_top))# & (ds0[dim] < above))
    # cut above first local maxima (via differentiation and difference of signs)
    ########## rechunk, weil plötzlich nicht mehr geht, warum auch immer naja, neue Version????
    ds4 = ds4.chunk(-1)
    p4 = np.sign(ds4[zh+"_norm_dy"].differentiate(dim)).diff(dim).idxmin(dim, skipna=True)
    ds5 = ds4.where(ds4[dim] < p4)
    ds5 = ml_height_top_new(ds5, moment=phi+"_norm", dim=dim, drop=False)

    # ds5.mlh_top,ds5.ml_top, derived according Wolfensberger et. al. but used PHIDP_norm instead of DBZH_norm

    # ml bottom refinement
    ds6 = ds0.where((ds0[dim] < ds3.mlh_bottom) & (ds0[dim] > below))
    # cut below first local maxima (via differentiation and difference of signs)
    ds6 = ds6.chunk(-1)
    p4a = np.sign(ds6[phi+"_norm_dy"].differentiate(dim)).diff(dim).sortby(dim, ascending=False).idxmin(dim, skipna=True)
    
    if clowres is not True:
        ds7 = ds6.where(ds6[dim] > p4a)
    
    #idx = ds7[phi+"_norm"].argmin(dim=dim, skipna=True)
    #idx[np.isnan(idx)] = 0
    if clowres:
        hgt = ds6[phi+"_norm"].idxmin(dim=dim, skipna=True, fill_value=np.nan)
        # Julian Giles modification: instead of taking the mean, calculate the bottom height again
        # hgt = ml_height_bottom_new(ds6, moment=phi+"_norm", dim=dim, drop=False)["mlh_bottom"]
    else: 
        hgt = ds7[phi+"_norm"].idxmin(dim=dim, skipna=True, fill_value=np.nan)
    #hgt = ds7.isel({dim: idx.load()}, drop=False)[dim]
    #for i in range(0,len(idx)):
    #    if idx[i].values == 0:
    #        hgt[idx].values = np.nan    
    if clowres:
        ds7 = ds6.assign(dict( 
                              mlh_bottom=hgt))
    else:
        ds7 = ds7.assign(dict( 
                              mlh_bottom=hgt))
    # ds7.mlh_bottom ds7.ml_bottom, derived similar to Wolfensberger et. Al. but for bottom
    # uses PHIDP_norm 

    ds = ds.assign(dict(mlh_top=ds5.mlh_top, 
                        
                        mlh_bottom=ds7.mlh_bottom, 
                        
                       ),
                  )
    
    if all_data is True:
        ds = ds.assign({"comb": ds0.comb,
                        rho+"_norm": ds0[rho+"_norm"],
                        zh+"_norm": ds0[zh+"_norm"],
                        phi+"_norm": ds0[phi+"_norm"],
                        "comb_dy": ds0.comb_dy,
                        rho+"_norm_dy": ds0[rho+"_norm_dy"],
                        zh+"_norm_dy": ds0[zh+"_norm_dy"],
                        phi+"_norm_dy": ds0[phi+"_norm_dy"],
                       })
        
    ds = ds.assign_coords({dim+'_idx': ([dim], np.arange(len(ds[dim])))})

    return ds



'''
################# Example usage of melting layer detection and PHIDP processing
X_ZH = "DBTH"
X_ZDR = "UZDR"
X_RHOHV = "RHOHV"
X_PHIDP = "PHIDP"

######### Processing PHIDP for BoXPol

phi = ds[X_PHIDP].where((ds[X_RHOHV+"_NC"]>=0.9) & (ds[X_ZH+"_OC"]>=0))
start_range, off = phi.pipe(phase_offset, rng=3000)

fix_range = 750
phi_fix = ds[X_PHIDP].copy()
off_fix = off.broadcast_like(phi_fix)
phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off

window = 11
window2 = None
phi_median = phi_fix.pipe(xr_rolling, window, window2=window2, method='median', skipna=True, min_periods=3)
phi_masked = phi_median.where((ds[X_RHOHV+"_NC"] >= 0.95) & (ds[X_ZH+"_OC"] >= 0.)) 

dr = phi_masked.range.diff('range').median('range').values / 1000.

winlen = 31 # windowlen 
min_periods = 3 # min number of vaid bins
kdp = kdp_from_phidp(phi_masked, winlen, min_periods=3)
kdp1 = kdp.interpolate_na(dim='range')

winlen = 31 
phidp = phidp_from_kdp(kdp1, winlen)

assign = {X_PHIDP+"_OC_SMOOTH": phi_median.assign_attrs(ds[X_PHIDP].attrs),
  X_PHIDP+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHIDP].attrs),
  "KDP_CONV": kdp.assign_attrs(ds.KDP.attrs),
  "PHI_CONV": phidp.assign_attrs(ds[X_PHIDP].attrs),

  X_PHIDP+"_OFFSET": off.assign_attrs(ds[X_PHIDP].attrs),
  X_PHIDP+"_OC": phi_fix.assign_attrs(ds[X_PHIDP].attrs)}

ds = ds.assign(assign)
ds_qvp_ra = ds.median("azimuth")

moments={X_ZH+"_OC": (10., 60.), X_RHOHV+"_NC": (0.65, 1.), X_PHIDP+"_OC": (-0.5, 360)}
dim = 'height'
thres = 0.02
limit = None
xwin = 5
ywin = 5
fmlh = 0.3
 
ml_qvp = melting_layer_qvp_X_new(ds_qvp_ra, moments=moments, 
         dim=dim, thres=thres, xwin=xwin, ywin=ywin, fmlh=fmlh, all_data=True)

ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml': ml_qvp.mlh_top})
ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})

ml_qvp = ml_qvp.where(ml_qvp.mlh_top<10000)
nan = np.isnan(ds_plus_entropy.PHIDP_OC_MASKED)
phi2 = ds_plus_entropy["PHIDP_OC_MASKED"].where((ds_plus_entropy.z < ml_qvp.mlh_bottom) | (ds_plus_entropy.z > ml_qvp.mlh_top))#.interpolate_na(dim='range',dask_gufunc_kwargs = "allow_rechunk")

phi2 = phi2.interpolate_na(dim='range', method=interpolation_method_ML)
phi2 = xr.where(nan, np.nan, phi2)

dr = phi2.range.diff('range').median('range').values / 1000.
print("range res [km]:", dr)
# winlen in gates
# todo window length in m
winlen = 31
min_periods = 3
kdp_ml = kdp_from_phidp(phi2, winlen, min_periods=3)
ds = ds.assign({"KDP_ML_corrected": (["time", "azimuth", "range"], kdp_ml.values, ml_qvp.KDP.attrs)})
ds["KDP_ML_corrected"] = ds.KDP_ML_corrected.where((ds.KDP_ML_corrected >= 0.01) & (ds.KDP_ML_corrected <= 3)) #### maybe you dont need this filtering here

ds = ds.assign_coords({'height': ds.z})

ds = ds.assign_coords({'height_ml': ml_qvp.mlh_top})
ds = ds.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})


#################### Giagrande refinment
cut_above = ds_qvp_ra.where(ds_qvp_ra.height<ds_qvp_ra.height_ml)
cut_above = cut_above.where(ds_qvp_ra.height>ds_qvp_ra.height_ml_bottom)
#test_above = cut_above.where((cut_above.rho >=0.7)&(cut_above.rho <0.98))

min_height_ML = cut_above.rho.idxmin(dim="height")

new_cut_below_min_ML = ds_qvp_ra.where(ds_qvp_ra.height > min_height_ML)
new_cut_above_min_ML = ds_qvp_ra.where(ds_qvp_ra.height < min_height_ML)

new_cut_below_min_ML_filter = new_cut_below_min_ML.rho.where((new_cut_below_min_ML.rho>=0.97)&(new_cut_below_min_ML.rho<=1))
new_cut_above_min_ML_filter = new_cut_above_min_ML.rho.where((new_cut_above_min_ML.rho>=0.97)&(new_cut_above_min_ML.rho<=1))            


import pandas as pd
######### ML TOP Giagrande refinement
panda_below_min = new_cut_below_min_ML_filter.to_pandas()
first_valid_height_after_ml = pd.DataFrame(panda_below_min).apply(pd.Series.first_valid_index)
first_valid_height_after_ml = first_valid_height_after_ml.to_xarray()            
######### ML BOTTOM Giagrande refinement
panda_above_min = new_cut_above_min_ML_filter.to_pandas()

last_valid_height = pd.DataFrame(panda_above_min).apply(pd.Series.last_valid_index)

last_valid_height = last_valid_height.to_xarray()

ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))


ds = ds.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
ds = ds.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
'''


#################################### CFADs

def hist2d(ax, PX, PY, binsx=[], binsy=[], mode='rel_all', whole_x_range=True, cb_mode=True, qq=0.2, cmap='turbo',
           colsteps=10, mini=0, fsize=13, fcolor='black', mincounts=500, cblim=[0,26], N=False,
           cborientation="horizontal", shading='gouraud', **kwargs):
    """
    Plots the 2-dimensional distribution of two Parameters
    # Input:
    # ------
    PX = Parameter x-axis
    PY = Parameter y-axis
    binsx = [start, stop, step]
    binsy = [start, stop, step]
    mode = 'rel_all', 'abs' or 'rel_y'
            rel_all : Relative Dist. to all pixels (this is not working ATM)
            rel_y   : Relative Dist. to y-axis.
            abs     : Absolute Dist.
    whole_x_range: use the whole range of values in the x coordinate? if False, only values inside the limits of binsx will be considered
                in the calculations and the counting of valid values; which can lead to different results depending how the bin ranges are defined.
    cb_mode : plot colorbar?
    qq = percentile [0-1]. Calculates the qq and 1-qq percentiles.
    mincounts: minimum sample number to plot
    N: plot sample size?
    cborientation: orientation of the colorbar, "horizontal" or "vertical"
    shading: shading argeument for matplotlib pcolormesh. Should be 'nearest' (no interpolation) or 'gouraud' (interpolated).
    kwargs: additional arguments for matplotlib pcolormesh
    
    # Output
    # ------
    Plot of 2-dimensional distribution
    """
    
    
    import matplotlib
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    import xarray

    # Flatten the arrays
    if type(PX) == xr.core.dataarray.DataArray:
        PX_flat = PX.values.flatten()
    elif type(PX) == np.ndarray:
        PX_flat = PX.flatten()
    else:
        raise TypeError("PX should be xarray.core.dataarray.DataArray or numpy.ndarray")

    if type(PY) == xr.core.dataarray.DataArray:
        PY_flat = PY.values.flatten()
    elif type(PY) == np.ndarray:
        PY_flat = PY.flatten()
    else:
        raise TypeError("PY should be xarray.core.dataarray.DataArray or numpy.ndarray")


    matplotlib.rc('axes',edgecolor='black')

    # discret cmap
    cmap = plt.cm.get_cmap(cmap, colsteps+1).copy()
    colors = list(cmap(np.arange(colsteps+1)))
    cmap = matplotlib.colors.ListedColormap(colors[:-1], "cfad")
    # set over-color to last color of list
    cmap.set_over(colors[-1])

    # Define bins arange
    bins_px = np.arange(binsx[0], binsx[1], binsx[2])
    bins_py = np.arange(binsy[0], binsy[1], binsy[2])
    
    # Hist 2d
    if whole_x_range:
        # to consider the whole range of values in x, we extend the bins array by adding bins to -inf and inf
        # at the edges and computing the histogram with those bins into account. Then, we discard the extreme values
        bins_px_ext = np.append(np.append(bins_px[::-1], -np.inf)[::-1], np.inf).copy()
        H_ext, xe_ext, ye = np.histogram2d(PX_flat, PY_flat, bins = (bins_px_ext, bins_py))
        H = H_ext[1:-1,:]
        xe = xe_ext[1:-1]
    else:
        H, xe, ye = np.histogram2d(PX_flat, PY_flat, bins = (bins_px, bins_py))
        H_ext = H # this is for the counting part of the overall sum
    
    # Calc mean x and y (for plotting with center-based index)
    mx =0.5*(xe[0:-1]+xe[1:len(xe)])
    my =0.5*(ye[0:-1]+ye[1:len(ye)])
    
    # Calculate Percentil
    var_mean = []

    var_med = []
    var_qq1 = []
    var_qq2 = []
    var_count = []

    for i in bins_py[:-1]:
        if whole_x_range:
            PX_sub = PX_flat[(PY_flat>i)&(PY_flat<=i+binsy[2])]
        else:
            # Improved: get the subset of values in the range of bins_px and the specific range of bins_py
            # otherwise it will consider outliers that are not considered in histogram2d
            PX_sub = PX_flat[(PX_flat>bins_px[0])&(PX_flat<bins_px[-1])&(PY_flat>i)&(PY_flat<=i+binsy[2])]

        # for every bin in Y dimension, calculate statistics of values in X
        var_med.append(np.nanmedian(PX_sub))
        var_mean.append(np.nanmean(PX_sub))

        var_qq1.append(np.nanquantile(PX_sub, qq))
        var_qq2.append(np.nanquantile(PX_sub, 1-qq))
        #var_count.append(len(PX_flat[(PY_flat>i)&(PY_flat<=i+binsy[2])])) # this is counting all values per Y bin, including nans
        var_count.append(np.isfinite(PX_sub).sum()) # improved, counting only valid values (not nan, not inf)
    
    var_med = np.array(var_med)
    var_mean = np.array(var_mean)
    
    var_qq1 = np.array(var_qq1)
    var_qq2 = np.array(var_qq2)
    var_count = np.array(var_count)
    
    var_med[var_count<mincounts]=np.nan
    var_mean[var_count<mincounts]=np.nan

    var_qq1[var_count<mincounts]=np.nan
    var_qq2[var_count<mincounts]=np.nan
    
    
    # overall sum for relative distribution
    if mode=='rel_all':
        allsum = np.nansum(H_ext)
        allsum[allsum<mincounts]=np.nan
        relHa = H.T/allsum
    elif mode=='rel_y':
        allsum = np.nansum(H_ext, axis=0)
        allsum[allsum<mincounts]=np.nan
        relHa = (H/allsum).T # transpose so the y axis is temperature
        
    elif mode=='abs':
        relHa = H.T
    else:
        print('Wrong mode parameter used! Please use mode="rel_all", mode="abs" or mode="rel_y"!')
    
    RES = 100*relHa
    
    RES[RES<mini]=np.nan

    
    img = ax.pcolormesh(mx, my ,RES , cmap=cmap, vmin=cblim[0], vmax=cblim[1], shading=shading, **kwargs) #, shading="gouraud"
    ax.plot(var_mean, my, color='red', lw=2)

    ax.plot(var_qq1, my, color='red', linestyle='-.', lw=2)
    ax.plot(var_qq2, my, color='red', linestyle='-.', lw=2)

    ax.grid(color=fcolor, linestyle='-.', lw=0.5, alpha=0.9)

    ax.xaxis.label.set_color(fcolor)
    ax.tick_params(axis='both', colors=fcolor)
    
    if N==True:
        ax2 = ax.twiny()
        ax2.plot(var_count, my, color='cornflowerblue', linestyle='-', lw=2)
        #ax2.set_xlim(0,2500)
        ax2.xaxis.label.set_color('cornflowerblue')
        ax2.yaxis.label.set_color('cornflowerblue')
        ax2.tick_params(axis='both', colors='cornflowerblue')
        
        xticks = ax2.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        xticks[-1].label1.set_visible(False)

    if cb_mode==True:
        '''
        if cborientation=="horizontal":
            cax = plt.axes([0.15, -0.05, 0.70, 0.03]) #Left, bottom, length, width
        if cborientation=="vertical":
            cax = plt.axes([0.95, 0.15, 0.03, 0.70]) #Left, bottom, length, width
        '''

        cb=plt.colorbar(img, cax=None, pad=0.05, ticks=np.linspace(cblim[0],cblim[1], colsteps+1), extend = "max", orientation=cborientation)

        if mode=='abs':
            cb.ax.set_title('#', color=fcolor)
        if mode!='abs':
            cb.ax.set_title('%', color=fcolor, fontsize=fsize)
        cb.ax.tick_params(labelsize=fsize, color=fcolor)
        cbar_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color=fcolor)


        
    return img



'''
# preprocessing

KDP_DGL_max_test = []




ZH = []
ZDR = []
KDP = []
RHO = []
TT = []
E = []
ML = []
valid_values = []
ZH_ML_max = []
ZDR_ML_max = []
RHO_ML_min = []
KDP_ML_mean = []
MLTH_ML = []
BETA = []

ZH_DGL = []
ZDR_DGL = []
RHO_DGL = []
KDP_DGL = []
ZH_DGL_09 = []
ZDR_DGL_09 = []
KDP_DGL_09 = []
KDP_DGL_max= []
RHO_DGL_min= []
ZH_DGL_max= []
ZDR_DGL_max= []
# ZH_snow_list= []
# ZDR_snow_list= []
# ZH_rain_list= []
# ZDR_rain_list= []
ZH_snow = []
ZDR_snow = []
ZH_rain = []
ZDR_rain = []
ZH_sfc = []
ZDR_sfc = []

Dm_ZDR_KDP_Z_max = []
Nt_ZDR_KDP_Z_max = []
Nt_ZDR_KDP_Z_max_log = []

IWC_ZDR_KDP_Z_max = []

Dm_ZDR_KDP_Z = []
Nt_ZDR_KDP_Z_log = []
Nt_ZDR_KDP_Z = []

IWC_ZDR_KDP_Z = []
Dm_ZDR_KDP_Z_DGL_09 = []
Nt_ZDR_KDP_Z_DGL_09 = []
IWC_ZDR_KDP_Z_DGL_09 = []
Dm_ZDR_KDP_Z_DGL = []
Nt_ZDR_KDP_Z_DGL = []
IWC_ZDR_KDP_Z_DGL = []
#.where(~np.isnan(ds.height_ml))
# .where(ds.height_ml>height_ml_bottom, drop=True)
for i in range(len(qvp_nc_files)):
    #print(qvp_nc_files[i])
    
    #try:
    ds = xr.open_dataset(qvp_nc_files[i], chunks='auto')
    ds = ds.swap_dims({"index_new":"time"})
    ml_height_thres = ~np.isnan(ds.height_ml_new_gia)
    ds_thrs = ds.where(ml_height_thres == True, drop=True)
    ml_bottom_thres = ~np.isnan(ds_thrs.height_ml_bottom_new_gia)
    ds_thrs = ds_thrs.where(ml_bottom_thres == True, drop=True)
    ds_thrs = ds_thrs.where(ds_thrs.height_ml_new_gia > ds_thrs.height_ml_bottom_new_gia) 
    ds = ds_thrs
    ds["Nt_ZDR_KDP_Z_log"] = np.log10(ds.Nt_ZDR_KDP_Z/1000)
    ds["Nt_ZDR_KDP_Z"] = ds.Nt_ZDR_KDP_Z/1000
    
    ds_stat_ML = ds.where(ds.min_entropy>=0.8, drop=True)
    ds_stat_ML = ds_stat_ML.where((ds_stat_ML.height<ds_stat_ML.height_ml_new_gia) & (ds_stat_ML.height>ds_stat_ML.height_ml_bottom_new_gia))
    ds_stat_ML = ds_stat_ML.where((ds_stat_ML.min_entropy>=0.8)&(ds_stat_ML.DBTH_OC > 0 )&(ds_stat_ML.KDP_ML_corrected > 0.01)&(ds_stat_ML.RHOHV_NC > 0.7)&(ds_stat_ML.UZDR_OC > -1),  drop=True)    

    ### ML statistics
    if ds_stat_ML.time.shape[0]!=0:
        ZH_ML_max.append(ds_stat_ML.DBTH_OC.max(dim="height").values.flatten())
        ZDR_ML_max.append(ds_stat_ML.UZDR_OC.max(dim="height").values.flatten())
        RHO_ML_min.append(ds_stat_ML.RHOHV_NC.min(dim="height").values.flatten())
        KDP_ML_mean.append(ds_stat_ML.KDP_ML_corrected.mean(dim="height").values.flatten())
        mlth_ML = ds_stat_ML.height_ml_new_gia - ds_stat_ML.height_ml_bottom_new_gia
        MLTH_ML.append(mlth_ML.values.flatten())
        
        ### Silke Style
        Gradient_silke = ds.where(ds.min_entropy>=0.8)
        Gradient_silke = Gradient_silke.where((Gradient_silke.min_entropy>=0.8)&(Gradient_silke.DBTH_OC > 0 )&(Gradient_silke.KDP_ML_corrected > 0.01)&(Gradient_silke.RHOHV_NC > 0.7)&(Gradient_silke.UZDR_OC > -1))    

        DBTH_OC_gradient_Silke_ML = Gradient_silke.DBTH_OC.sel(height = Gradient_silke.height_ml_new, method="nearest")
        DBTH_OC_gradient_Silke_ML_plus_2_km = Gradient_silke.DBTH_OC.sel(height = DBTH_OC_gradient_Silke_ML.height+2000, method="nearest")
        DBTH_OC_gradient_Final = (DBTH_OC_gradient_Silke_ML_plus_2_km - DBTH_OC_gradient_Silke_ML)/2
        BETA.append(DBTH_OC_gradient_Final)
    
    ### DGL statistics
    ds_stat_DGL = ds.where((ds.temp_coord>=-20)&(ds.temp_coord<=-10), drop=True)    
    ds_stat_DGL = ds_stat_DGL.where(ds_stat_DGL.min_entropy>=0.8, drop=True)
    ds_stat_DGL = ds_stat_DGL.where((ds_stat_DGL.min_entropy>=0.8)&(ds_stat_DGL.DBTH_OC > 0 )&(ds_stat_DGL.KDP_ML_corrected > 0.01)&(ds_stat_DGL.RHOHV_NC > 0.7)&(ds_stat_DGL.UZDR_OC > -1),  drop=True)    

    if ds_stat_DGL.time.shape[0]!=0:
        ZH_DGL.append(ds_stat_DGL.DBTH_OC.values.flatten())
        ZDR_DGL.append(ds_stat_DGL.UZDR_OC.values.flatten())
        KDP_DGL_max.append(ds_stat_DGL.KDP_ML_corrected.max(dim="height").values.flatten())
        
        KDP_DGL_max_test.append(np.asarray(ds_stat_DGL.KDP_ML_corrected.max(dim="height").stack(dim=["time"]).data))

        RHO_DGL_min.append(ds_stat_DGL.RHOHV_NC.min(dim="height").values.flatten())
        ZH_DGL_max.append(ds_stat_DGL.DBTH_OC.max(dim="height").values.flatten())
        ZDR_DGL_max.append(ds_stat_DGL.UZDR_OC.max(dim="height").values.flatten())        
        ZDR_DGL_09.append(ds_stat_DGL.UZDR_OC.quantile(0.9, dim="height").values.flatten())
        ZH_DGL_09.append(ds_stat_DGL.DBTH_OC.quantile(0.9, dim="height").values.flatten())

        RHO_DGL.append(ds_stat_DGL.RHOHV_NC.values.flatten())
        
        KDP_DGL.append(ds_stat_DGL.KDP_ML_corrected.values.flatten())
        KDP_DGL_09.append(ds_stat_DGL.KDP_ML_corrected.quantile(0.9, dim="height").values.flatten())

        
 #ds_stat_ML_list_KDP_ML_corrected.append(np.asarray(ds.KDP_ML_corrected.max(dim="height").stack(dim=["height","time"]).data))
       
        
        
    ### other statistics
    
    ds_other1 = ds.where(ds.min_entropy>=0.8)
    ds_other1 = ds_other1.where((ds_other1.min_entropy>=0.8)&(ds_other1.DBTH_OC > 0 )&(ds_other1.KDP_ML_corrected > 0.01)&(ds_other1.RHOHV_NC > 0.7)&(ds_other1.UZDR_OC > -1))    
    ZH_sfc.append(ds_other1.DBTH_OC.swap_dims({"height":"range"}).isel(range = 7).values.flatten())
    ZDR_sfc.append(ds_other1.UZDR_OC.swap_dims({"height":"range"}).isel(range = 7).values.flatten())
#     ZH_snow = ds_other.DBTH_OC.sel(height = ds_other.height_ml_new, method="nearest")
#     ZDR_snow = ds_other.UZDR_OC.sel(height = ds_other.height_ml_new, method="nearest")
#     ZH_rain = ds_other.DBTH_OC.sel(height = ds_other.height_ml_bottom_new_gia, method="nearest")
#     ZDR_rain = ds_other.UZDR_OC.sel(height = ds_other.height_ml_bottom_new_gia, method="nearest")
#     ds_other = ds.where(ds.min_entropy>=0.8, drop=True)
#     ds_other = ds_other.where((ds_other.min_entropy>=0.8)&(ds_other.DBTH_OC > 0 )&(ds_other.KDP_ML_corrected > 0.001)&(ds_other.RHOHV_NC > 0.7)&(ds_other.UZDR_OC > -1),  drop=True)    

    ZH_snow.append(ds_other1.DBTH_OC.sel(height = ds_other1.height_ml_new, method="nearest").values.flatten())
    ZDR_snow.append(ds_other1.UZDR_OC.sel(height = ds_other1.height_ml_new, method="nearest").values.flatten())
    ZH_rain.append(ds_other1.DBTH_OC.sel(height = ds_other1.height_ml_bottom_new_gia, method="nearest").values.flatten())
    ZDR_rain.append(ds_other1.UZDR_OC.sel(height = ds_other1.height_ml_bottom_new_gia, method="nearest").values.flatten())
    
    
    
    
    
    ### IWC, Nt, Dm statistics    hier ist weiteres Filtern eher nicht gut i would say
    ds_microphysical = ds.where(ds.min_entropy>=0.8, drop=True)
    ds_microphysical = ds_microphysical.where((ds_microphysical.temp_coord>=-20)&(ds_microphysical.temp_coord<=-10), drop=True)
    
    Dm_ZDR_KDP_Z_DGL_09.append(ds_microphysical.Dm_ZDR_KDP_Z.quantile(0.9, dim="height").values.flatten())
    Nt_ZDR_KDP_Z_DGL_09.append(ds_microphysical.Nt_ZDR_KDP_Z.quantile(0.9, dim="height").values.flatten())
    IWC_ZDR_KDP_Z_DGL_09.append(ds_microphysical.IWC_ZDR_KDP_Z.quantile(0.9, dim="height").values.flatten())
    
    Dm_ZDR_KDP_Z_DGL.append(ds_microphysical.Dm_ZDR_KDP_Z.values.flatten())
    Nt_ZDR_KDP_Z_DGL.append(ds_microphysical.Nt_ZDR_KDP_Z.values.flatten())
    IWC_ZDR_KDP_Z_DGL.append(ds_microphysical.IWC_ZDR_KDP_Z.values.flatten())    
    
    Dm_ZDR_KDP_Z_max.append(ds_microphysical.Dm_ZDR_KDP_Z.max(dim="height").values.flatten())
    Nt_ZDR_KDP_Z_max_log.append(ds_microphysical.Nt_ZDR_KDP_Z_log.max(dim="height").values.flatten())
    Nt_ZDR_KDP_Z_max.append(ds_microphysical.Nt_ZDR_KDP_Z.max(dim="height").values.flatten())

    IWC_ZDR_KDP_Z_max.append(ds_microphysical.IWC_ZDR_KDP_Z.max(dim="height").values.flatten())    
    
    
#     ZH_snow_list.append(np.asarray(ZH_snow.squeeze().stack(dim=["height","time"]).data))
#     ZDR_snow_list.append(np.asarray(ZDR_snow.squeeze().stack(dim=["height","time"]).data))
#     ZH_rain_list.append(np.asarray(ZH_rain.squeeze().stack(dim=["height","time"]).data))
#     ZDR_rain_list.append(np.asarray(ZDR_rain.squeeze().stack(dim=["height","time"]).data))
    
    
    
    ZH.append(np.asarray(ds.DBTH_OC.squeeze().stack(dim=["height","time"]).data))
    ZDR.append(np.asarray(ds.UZDR_OC.squeeze().stack(dim=["height","time"]).data))
    RHO.append(np.asarray(ds.RHOHV_NC.squeeze().stack(dim=["height","time"]).data))
    KDP.append(np.asarray(ds.KDP_ML_corrected.squeeze().stack(dim=["height","time"]).data))
    E.append(np.asarray(ds.min_entropy.squeeze().stack(dim=["height","time"]).data))
    valid_values.append(np.asarray(ds.valid_values_min_120.squeeze().stack(dim=["height","time"]).data))
    TT.append(np.asarray((ds.temp_coord).squeeze().stack(dim=["height","time"]).data))   
    Dm_ZDR_KDP_Z.append(np.asarray(ds.Dm_ZDR_KDP_Z.squeeze().stack(dim=["height","time"]).data))
    Nt_ZDR_KDP_Z.append(np.asarray(ds.Nt_ZDR_KDP_Z.squeeze().stack(dim=["height","time"]).data))
    Nt_ZDR_KDP_Z_log.append(np.asarray(ds.Nt_ZDR_KDP_Z_log.squeeze().stack(dim=["height","time"]).data))

    IWC_ZDR_KDP_Z.append(np.asarray(ds.IWC_ZDR_KDP_Z.squeeze().stack(dim=["height","time"]).data))    
#     ZH.append(ds.DBTH_OC.values.flatten())
#     ZDR.append(ds.UZDR_OC.values.flatten())
#     RHO.append(ds.RHOHV_NC.values.flatten())
#     KDP.append(ds.KDP_ML_corrected.values.flatten())
#     TT.append(ds.temp_coord.values.flatten())
#     E.append(ds.min_entropy.values.flatten())
    #except:
    #print('Error')


ZH = np.concatenate(ZH)
ZDR = np.concatenate(ZDR)
KDP = np.concatenate(KDP)
RHO = np.concatenate(RHO)
TT = np.concatenate(TT)
E = np.concatenate(E)
valid_values = np.concatenate(valid_values)

RHO_ML_min = np.concatenate(RHO_ML_min)
ZDR_ML_max = np.concatenate(ZDR_ML_max)
ZH_ML_max = np.concatenate(ZH_ML_max)
MLTH_ML = np.concatenate(MLTH_ML)
BETA = np.concatenate(BETA)
KDP_ML_mean = np.concatenate(KDP_ML_mean)

ZH_DGL = np.concatenate(ZH_DGL)
ZDR_DGL = np.concatenate(ZDR_DGL)
RHO_DGL = np.concatenate(RHO_DGL)
KDP_DGL = np.concatenate(KDP_DGL)
ZDR_DGL_09 = np.concatenate(ZDR_DGL_09)
KDP_DGL_09 = np.concatenate(KDP_DGL_09) 


Dm_ZDR_KDP_Z_DGL_09= np.concatenate(Dm_ZDR_KDP_Z_DGL_09)
Nt_ZDR_KDP_Z_DGL_09= np.concatenate(Nt_ZDR_KDP_Z_DGL_09)
IWC_ZDR_KDP_Z_DGL_09= np.concatenate(IWC_ZDR_KDP_Z_DGL_09)
Dm_ZDR_KDP_Z_DGL= np.concatenate(Dm_ZDR_KDP_Z_DGL)
Nt_ZDR_KDP_Z_DGL= np.concatenate(Nt_ZDR_KDP_Z_DGL)
IWC_ZDR_KDP_Z_DGL= np.concatenate(IWC_ZDR_KDP_Z_DGL)

Dm_ZDR_KDP_Z_max= np.concatenate(Dm_ZDR_KDP_Z_max)
Nt_ZDR_KDP_Z_max= np.concatenate(Nt_ZDR_KDP_Z_max)
Nt_ZDR_KDP_Z_max_log= np.concatenate(Nt_ZDR_KDP_Z_max_log)

IWC_ZDR_KDP_Z_max= np.concatenate(IWC_ZDR_KDP_Z_max)

KDP_DGL_max_test = np.concatenate(KDP_DGL_max_test) 


KDP_DGL_max= np.concatenate(KDP_DGL_max)
RHO_DGL_min= np.concatenate(RHO_DGL_min)
ZH_DGL_max= np.concatenate(ZH_DGL_max)
ZDR_DGL_max= np.concatenate(ZDR_DGL_max)

Dm_ZDR_KDP_Z = np.concatenate(Dm_ZDR_KDP_Z)
Nt_ZDR_KDP_Z = np.concatenate(Nt_ZDR_KDP_Z)
IWC_ZDR_KDP_Z = np.concatenate(IWC_ZDR_KDP_Z)
Nt_ZDR_KDP_Z_log = np.concatenate(Nt_ZDR_KDP_Z_log)

ZH_snow = np.concatenate(ZH_snow)
ZDR_snow = np.concatenate(ZDR_snow)
ZH_rain = np.concatenate(ZH_rain)
ZDR_rain = np.concatenate(ZDR_rain)
ZH_sfc = np.concatenate(ZH_sfc)
ZDR_sfc = np.concatenate(ZDR_sfc)




####### Plotting example how to use the function above


mask = (Dm_ZDR_KDP_Z>0) & (IWC_ZDR_KDP_Z>0) & (E>=0.8)& (TT<=0.5)  &(TT>=-20)

smask =  (Dm_syn>0.01) & (IWC_syn>0) &  (sE_retrievals>=0.8)& (sTMP_retrievals<=0.5)  &(sTMP_retrievals>=-20)& (~np.isnan(Nt_syn_log))
#smask =    (sE_retrievals>=0.8)#& (sTMP_retrievals<=0.5)  &(sTMP_retrievals>=-20)
# Remove Color for bins with th
mth = 0

# Temp bins
tb=1#0.7

# Min counts per Temp layer
mincounts=200

#Colorbar limits
cblim=[0,10]

fig, ax = plt.subplots(3, 2, sharey=True, figsize=(20,24))#, gridspec_kw={'width_ratios': [0.47,0.53]})
hist2d(ax[0,0], Dm_ZDR_KDP_Z[mask], TT[mask], binsx=[0,10,0.1], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cb_mode=False,cmap=cm, colsteps=10, mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim)
ax[0,0].set_ylim(0.5,ytlim)
ax[0,0].set_ylabel('Temperature [°C]', fontsize=30, color='black')
ax[0,0].set_xlabel("$D_{m}$ [$mm$]", fontsize=30, color='black')

ax[0,0].text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[0,0].transAxes)
ax[0,0].set_title('Observations \n '+'N: '+str(len(Dm_ZDR_KDP_Z[mask])), fontsize=30, color='black', loc='left')


hist2d(ax[1,0], Nt_ZDR_KDP_Z_log[mask], TT[mask], binsx=[-2,2,0.1], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cb_mode=False,cmap=cm, colsteps=10, mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim)
ax[1,0].set_ylim(0.5,ytlim)
ax[1,0].set_ylabel('Temperature [°C]', fontsize=30, color='black')
ax[1,0].set_xlabel("$N_{t}$ [log$_{10}$($L^{-1}$)]", fontsize=30, color='black')
ax[1,0].xaxis.set_ticks(np.arange(-1.5, 2, 0.5))
ax[1,0].text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[1,0].transAxes)

ax2 =hist2d(ax[2,0], IWC_ZDR_KDP_Z[mask], TT[mask],  binsx=[0,0.7,0.01], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cb_mode=False,cmap=cm, colsteps=10, mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim, N=True)
ax[2,0].set_ylim(0.5,ytlim)
ax[2,0].set_ylabel('Temperature [°C]', fontsize=30, color='black')
ax[2,0].set_xlabel("IWC [$g/m^3$]", fontsize=30, color='black')

ax2.set_xlim(200,95000)
ax[2,0].text(0.05, 0.95, 'e)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[2,0].transAxes)


hist2d(ax[0,1], Dm_syn[smask], sTMP_retrievals[smask], binsx=[0,10,0.1], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cb_mode=False,cmap=cm, colsteps=10,mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim)
ax[0,1].set_ylim(0.5,ytlim)
ax[0,1].set_xlabel("$D_{m}$ [$mm$]", fontsize=30, color='black')

ax[0,1].text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[0,1].transAxes)
ax[0,1].set_title('Simulations \n '+'N: '+str(len(Dm_syn[smask])), fontsize=30, color='black', loc='left')

hist2d(ax[1,1], Nt_syn_log[smask], sTMP_retrievals[smask], binsx=[-2,2,0.1], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cb_mode=False,cmap=cm, colsteps=10, mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim)
ax[1,1].set_ylim(0.5,ytlim)

ax[1,1].set_xlabel("$N_{t}$ [log$_{10}$($L^{-1}$)]", fontsize=30, color='black')
ax[1,1].xaxis.set_ticks(np.arange(-1.5, 2, 0.5))

ax[1,1].text(0.05, 0.95, 'd)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[1,1].transAxes)

ax2 =hist2d(ax[2,1], IWC_syn[smask], sTMP_retrievals[smask], binsx=[0,0.7,0.01], binsy=[ytlim,0.5,tb],
       mode='rel_y', qq=0.2,cmap=cm, colsteps=10, mini=mth,  fsize=30,
       mincounts=mincounts, cblim=cblim, N=True)

ax[2,1].set_ylim(0.5,ytlim)
ax[2,1].text(0.05, 0.95, 'f)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=40, transform=ax[2,1].transAxes)
legend = ax[2,1].legend(title ="Sample size", loc = "upper center", labelcolor = 'cornflowerblue', frameon=False,)
plt.setp(legend.get_title(), color='cornflowerblue',fontsize=30)
ax[2,1].set_xlabel("IWC [$g/m^3$]", fontsize=30, color='black')
ax2.set_xlim(200, 11500)
box1 = ax[0,0].get_position()
box2 = ax[1,0].get_position()
box3 = ax[2,0].get_position()
box4 = ax[0,1].get_position()
box5 = ax[1,1].get_position()
box6 = ax[2,1].get_position()


ax[0,0].set_position([box1.x0, box1.y0 + box1.height * 1, box1.width * 1.15, box1.height * 1])
ax[0,1].set_position([box4.x0, box4.y0 + box4.height * 1, box4.width* 1.15, box4.height * 1])

ax[1,0].set_position([box2.x0, box2.y0 + box2.height *1., box2.width* 1.15, box2.height * 1])
ax[1,1].set_position([box5.x0, box5.y0 + box5.height *1., box5.width* 1.15, box5.height * 1])

ax[2,0].set_position([box3.x0, box3.y0 + box3.height * 0.9, box3.width* 1.15, box3.height * 1])

ax[2,1].set_position([box6.x0, box6.y0 + box6.height * 0.9, box6.width* 1.15, box6.height * 1])


'''

############## Function to load ERA5 temperature into a dataset
def interp_to_ht(ds_temp, ht):
    """
    Interpolate temperature profile from ERA5 to higher resolution
    """
    ds_temp = ds_temp.swap_dims({"lvl":"height"})
    return ds_temp.interp({"height": ht})

def attach_ERA5_TEMP(ds, site=None, path=None, convert_to_C=True):
    """
    Function to attach temperature data from ERA5 as a new coordinate of ds. It
    interpolates the temperature profile from ERA5 levels to the ds heights. By
    default it is converted to degrees C.
    
    Parameters
    ----------
    ds : xarray Dataset
        Dataset to which to attach the temperature coordinate as a function of height.
    site : str, optional
        (Short) Name of the radar site to locate the data in the default folder:
        "/automount/ags/jgiles/ERA5/hourly/"
        Either site or path must be given.
    path : str, optional
        Path to the folder where ERA5 temperature and geopotential data can be found.
        Either site or path must be given.   
    convert_to_C : bool
        If True (default), convert ERA5 temperature from K to C.

    Returns
    ----------
    ds : xarray Dataset
        Original dataset with added TEMP coordinate
    """
    if path is None:
        # Set the default path if path is None
        try:
            era5_dir = "/automount/ags/jgiles/ERA5/hourly/"+site+"/pressure_level_vars/"
            if not os.path.exists(era5_dir):
                # if path does not exist, raise an error
                raise RuntimeError("folder for site="+site+" not found!")
                
        except TypeError:
            raise TypeError("If path is not provided, site must be provided!")
    elif type(path) is str:
        era5_dir = path
    else:
        raise TypeError("path must be type str!")

    # if time is not well defined, we try to get it from variable rtime
    if ds["time"].isnull().any():
        try:
            ds.coords["time"] = ds.rtime.min(dim="azimuth", skipna=True).compute()    
        except:
            raise KeyError("Dimension time has not valid values")

    # if some coord has dimension time, reduce using median
    for coord in ["latitude", "longitude", "altitude", "elevation"]:
        if coord in ds.coords:
            if "time" in ds[coord].dims:
                ds.coords[coord] = ds.coords[coord].median("time")

    # We need the ds to be georeferenced in case it is not
    if "z" not in ds.coords:
        ds = ds.pipe(wrl.georef.georeference_dataset)

    # get times of the radar files
    startdt0 = dt.datetime.utcfromtimestamp(int(ds.time[0].values)/1e9).date()
    enddt0 = dt.datetime.utcfromtimestamp(int(ds.time[-1].values)/1e9).date() + dt.timedelta(hours=24)
        
    # transform the dates to datetimes
    startdt = dt.datetime.fromordinal(startdt0.toordinal())
    enddt = dt.datetime.fromordinal(enddt0.toordinal())
    
    # open ERA5 files
    era5_t = xr.open_mfdataset(reversed(glob.glob(era5_dir+"temperature/*"+str(startdt.year)+"*")), concat_dim="lvl", combine="nested")
    era5_g = xr.open_mfdataset(reversed(glob.glob(era5_dir+"geopotential/*"+str(startdt.year)+"*")), concat_dim="lvl", combine="nested")
    
    # add altitude coord to temperature data
    earth_r = wrl.georef.projection.get_earth_radius(ds.latitude.values)
    gravity = 9.80665
    
    era5_t.coords["height"] = (earth_r*(era5_g.z/gravity)/(earth_r - era5_g.z/gravity)).compute()
    
    # Create time dimension and concatenate
    try: # this might fail because of the issue with the time dimension in elevations that some files have
        dtslice0 = startdt.strftime('%Y-%m-%d %H')
        dtslice1 = enddt.strftime('%Y-%m-%d %H')
        temperatures = era5_t["t"].loc[{"time":slice(dtslice0, dtslice1)}].isel({"latitude":0, "longitude":0})
        if convert_to_C:
            # convert from K to C
            temp_attrs = temperatures.attrs
            temperatures = temperatures -273.15
            temp_attrs["units"] = "C"
            temperatures.attrs = temp_attrs
        
        # Interpolate to higher resolution
        hmax = 50000.
        ht = np.arange(0., hmax, 50)
                

        interp_to_ht_partial = partial(interp_to_ht, ht=ht)

        results = []
        
        with Pool() as P:
            results = P.map( interp_to_ht_partial, [temperatures[:,tt] for tt in range(len(temperatures.time)) ] )
        
        itemp_da = xr.concat(results, "time")
        
        # Fix Temperature below first measurement and above last one
        itemp_da = itemp_da.bfill(dim="height")
        itemp_da = itemp_da.ffill(dim="height")
        
        # Attempt to fill any missing timestep with adjacent data
        itemp_da = itemp_da.ffill(dim="time")
        itemp_da = itemp_da.bfill(dim="time")
        
        # Interpolate to dataset height and time, then add to dataset
        def merge_radar_profile(rds, cds):
            # cds = cds.interp({'height': rds.z}, method='linear')
            cds = cds.interp({'height': rds.z}, method='linear')
            cds = cds.interp({"time": rds.time}, method="linear")
            rds = rds.assign({"TEMP": cds})
            rds.TEMP.attrs["source"]="ERA5"
            return rds
        
        ds = ds.pipe(merge_radar_profile, itemp_da)
        
        ds.coords["TEMP"] = ds["TEMP"] # move TEMP from variable to coordinate
        return ds
    except ValueError:
        raise ValueError("!!!! ERROR: some issue when concatenating ERA5 data")




######################## NOISE CORRECTION RHOHV
## From Veli

from xhistogram.xarray import histogram
import wradlib as wrl
import dask
from dask.diagnostics import ProgressBar
import xradar


def noise_correction(ds, noise_level):
    """Calculate SNR, apply to RHOHV
    """
    # noise calculations
    snrh = ds.DBZH - 20 * np.log10(ds.range * 0.001) - noise_level - 0.033 * ds.range / 1000
    snrh = snrh.where(snrh >= 0).fillna(0)
    # attrs = wrl.io.xarray.moments_mapping['SNRH']
    attrs = xradar.model.sweep_vars_mapping['SNRH'] # moved to xradar since wradlib 1.19
    attrs.pop('gamic', None)
    snrh = snrh.assign_attrs(attrs)
    rho = ds.RHOHV * (1. + 1. / 10. ** (snrh * 0.1))
    rho = rho.assign_attrs(ds.RHOHV.attrs) 
    ds = ds.assign({'SNRH': snrh, 'RHOHV_NC': rho})
    ds = ds.assign_coords({'NOISE_LEVEL': noise_level})
    return ds


def noise_correction2(dbz, rho, noise_level):
    """
    Calculate SNR, apply to RHOHV
    Formula from Ryzhkov book page 187/203
    """
    # noise calculations
    snrh = dbz - 20 * np.log10(dbz.range * 0.001) - noise_level - 0.033 * dbz.range / 1000
    snrh = snrh.where(snrh >= 0).fillna(0)
    # attrs = wrl.io.xarray.moments_mapping['SNRH']
    attrs = xradar.model.sweep_vars_mapping['SNRH'] # moved to xradar since wradlib 1.19
    attrs.pop('gamic', None)
    snrh = snrh.assign_attrs(attrs)
    snrh.name = "SNRH"
    rho_nc = rho * (1. + 1. / 10. ** (snrh * 0.1))
    rho_nc = rho_nc.assign_attrs(rho.attrs) 
    rho_nc.name = "RHOHV_NC"
    return snrh, rho_nc

def calculate_noise_level(dbz, rho, noise=(-40, -20, 1), rho_bins=(0.9, 1.1, 0.005), snr_bins=(5., 30., .1)):
    """
    This functions calculates the noise levels and noise corrections for RHOHV, for a range of noise values.
    It returns a list of signal-to-noise and corrected rhohv arrays, as well as histograms involed in the calculations,
    a list of standard deviations for every result and the noise value with the minimum std.
    The final noise correction should be chosen based on the rn value (minumum std)
    
    The default noise range is based on BoXPol data, it may be good to extend it a bit for C-Band.
    
    The final noise level (rn) should be used with noise_correction2 one more time to get the final result.
    It may happen that the correction is too strong and we get some RHOHV values over 1. We should
    check this for some days of data and if that is the case, then select a noise level that is slightly less (about 2% less)
    """
    noise = np.arange(*noise)
    rho_bins = np.arange(*rho_bins)
    snr_bins = np.arange(*snr_bins)
    corr = [noise_correction2(dbz, rho, n) for n in noise]
    #with ProgressBar():
    #    corr = dask.compute(*corr)
    # hist = [dask.delayed(histogram)(rho0, snr0, bins=[rho_bins, snr_bins], block_size=rho.time.size) for snr0, rho0 in corr]
    hist = [histogram(rho0, snr0, bins=[rho_bins, snr_bins], block_size=rho.time.size) for snr0, rho0 in corr]
    # with ProgressBar():
    #     hist = dask.compute(*hist)
    std = [np.std(r.idxmax('RHOHV_NC_bin')).values for r in hist]
    rn = noise[np.argmin(std)]
    return corr, hist, std, rn


#### 
# Calibration of ZDR with light-rain consistency
from matplotlib import pyplot as plt

def zhzdr_lr_consistency_old(ZH, ZDR, RHO, TMP, rhohv_th=0.99, tmp_th=5, plot_correction=True):
    """
    ZH-ZDR Consistency in light rain, for ZDR calibration
    AR p.155-156
    
    """
    zdr_zh_20 = np.nanmedian(ZDR[(ZH>=19)&(ZH<21)&(RHO>rhohv_th)&(TMP>tmp_th)])
    zdr_zh_22 = np.nanmedian(ZDR[(ZH>=21)&(ZH<23)&(RHO>rhohv_th)&(TMP>tmp_th)])
    zdr_zh_24 = np.nanmedian(ZDR[(ZH>=23)&(ZH<25)&(RHO>rhohv_th)&(TMP>tmp_th)])
    zdr_zh_26 = np.nanmedian(ZDR[(ZH>=25)&(ZH<27)&(RHO>rhohv_th)&(TMP>tmp_th)])
    zdr_zh_28 = np.nanmedian(ZDR[(ZH>=27)&(ZH<29)&(RHO>rhohv_th)&(TMP>tmp_th)])
    zdr_zh_30 = np.nanmedian(ZDR[(ZH>=29)&(ZH<31)&(RHO>rhohv_th)&(TMP>tmp_th)])

    zdroffset = np.nansum([zdr_zh_20-.23, zdr_zh_22-.27, zdr_zh_24-.33, zdr_zh_26-.40, zdr_zh_28-.48, zdr_zh_30-.56])/6.
    
    if plot_correction:
        mask = (RHO>rhohv_th)&(TMP>tmp_th)
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        hist_2d(ZH[mask], ZDR[mask], bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
        plt.plot([20,22,24,26,28,30],[.23, .27, .33, .40, .48, .56], color='black')
        plt.title('Non-calibrated $Z_{DR}$')
        plt.xlabel(r'$Z_H$', fontsize=15)
        plt.ylabel(r'$Z_{DR}$', fontsize=15)
        plt.grid(which='both', color='black', linestyle=':', alpha=0.5)
        
        plt.subplot(1,2,2)
        hist_2d(ZH[mask], ZDR[mask]-zdroffset, bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
        plt.plot([20,22,24,26,28,30],[.23, .27, .33, .40, .48, .56], color='black')
        plt.title('Calibrated $Z_{DR}$')
        plt.xlabel(r'$Z_H$', fontsize=15)
        plt.ylabel(r'$Z_{DR}$', fontsize=15)
        plt.grid(which='both', color='black', linestyle=':', alpha=0.5)
        plt.legend(title=r'$\Delta Z_{DR}$: '+str(np.round(zdroffset,3))+'dB')
    
        plt.tight_layout()
        plt.show()
    
    return zdroffset

def zhzdr_lr_consistency(ds, zdr="ZDR", dbzh="DBZH", rhohv="RHOHV", rhvmin=0.99, min_h=1000, 
                         mlbottom=None, temp=None, timemode="step", plot_correction=False, plot_timestep=0):
    """
    Improved function for
    ZH-ZDR Consistency in light rain, for ZDR calibration
    AR p.155-156
    
    ds : xarray Dataset
        Dataset with ZDR, DBZH and RHOHV.
    zdr : str
        Name of the variable in ds for differential reflectivity data. Default is "ZDR".
    dbzh : str
        Name of the variable in ds for horizontal reflectivity data (in dB). Default is "DBZH".
    rhohv : str
        Name of the variable in ds for cross-correlation coefficient data. Default is "RHOHV".
    mlbottom : str, int or float
        If str: name of the ML bottom variable in ds. If None, it is assumed to 
        be "height_ml_bottom_new_gia" or "height_ml_bottom" (in that order). 
        If int or float: we assume the ML bottom is not available and we take the given
        int value as the temperature level from which to consider radar bins.
        Only gates below the melting layer bottom (i.e. the rain
        region below the melting layer) are included in the method.
    temp : str, optional
        Name of the temperature variable in ds. Only necessary if mlbottom is int.
        If None is given and mlbottom is int or float, the default name "TEMP" is used.
    rhvmin : float, optional
        Threshold on :math:`\rho_{HV}` (unitless) related to light rain.
        The default is 0.99.
    min_h : float, optional
        Minimum height of usable data within the polarimetric profiles, in m. This is relative to
        sea level and not relative to the altitude of the radar (in accordance to the "z" coordinate 
        from wradlib.georef.georeference_dataset). The default is 1000.
    timemode : str
        How to calculate the offset in case a time dimension is found. "step" calculates one offset
        per timestep. "all" calculates one offset for the whole ds. Default is "step"
    plot_correction : bool
        If True, plot the histogram showing the uncorrected vs corrected data. Default is False.
    plot_timestep : int
        In case timemode="step" and plot_correction=True, plot_timestep defines the time index to 
        be plotted. By default the first timestep (index=0) is plotted.
    
    """
    # We need the ds to be georeferenced in case it is not
    if "z" not in ds.coords:
        ds = ds.pipe(wrl.georef.georeference_dataset)
    
    if mlbottom is None:
        try:
            ml_bottom = ds["z"] < ds["height_ml_bottom_new_gia"]
        except KeyError:
            try:
                ml_bottom = ds["z"] < ds["height_ml_bottom"]
            except KeyError:
                raise KeyError("Melting layer bottom not found in ds. Provide melting layer bottom or temperature level.")
            except:
                raise RuntimeError("Something went wrong when looking for ML variable")
        except:
            raise RuntimeError("Something went wrong when looking for ML variable")
    elif type(mlbottom) is str:
        ml_bottom = ds["z"] < ds[mlbottom]
    elif type(mlbottom) in [float, int]:
        if type(temp) is str:
            ml_bottom = ds[temp] > mlbottom
        elif temp is None:
            try:
                ml_bottom = ds["TEMP"] > mlbottom
            except KeyError:
                raise KeyError("temp is not given and could not be found by default")
        else:
            raise TypeError("temp must be str or None")
    else:
        raise TypeError("mlbottom must be str, int or None")

    # filter data below ML and above min_h
    ds_fil = ds.where(ml_bottom)
    ds_fil = ds_fil.where(ds["z"]>min_h)

    if "time" in ds and timemode=="step":
        # Get dimensions to reduce (other than time)
        dims_wotime = [kk for kk in ds_fil.dims]
        while "time" in dims_wotime:
            dims_wotime.remove("time")
        
        zdr_zh_20 = (ds_fil[zdr].where((ds_fil[dbzh]>=19)&(ds_fil[dbzh]<21)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)
        zdr_zh_22 = (ds_fil[zdr].where((ds_fil[dbzh]>=21)&(ds_fil[dbzh]<23)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)
        zdr_zh_24 = (ds_fil[zdr].where((ds_fil[dbzh]>=23)&(ds_fil[dbzh]<25)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)
        zdr_zh_26 = (ds_fil[zdr].where((ds_fil[dbzh]>=25)&(ds_fil[dbzh]<27)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)
        zdr_zh_28 = (ds_fil[zdr].where((ds_fil[dbzh]>=27)&(ds_fil[dbzh]<29)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)
        zdr_zh_30 = (ds_fil[zdr].where((ds_fil[dbzh]>=29)&(ds_fil[dbzh]<31)&(ds_fil[rhohv]>rhvmin))).median(dim=dims_wotime)

        zdroffset = xr.concat([zdr_zh_20-.23, zdr_zh_22-.27, zdr_zh_24-.33, zdr_zh_26-.40, zdr_zh_28-.48, zdr_zh_30-.56], dim='dataarrays').sum(dim='dataarrays', skipna=True)/6
        
    else:
        zdr_zh_20 = (ds_fil[zdr].where((ds_fil[dbzh]>=19)&(ds_fil[dbzh]<21)&(ds_fil[rhohv]>rhvmin))).compute().median()
        zdr_zh_22 = (ds_fil[zdr].where((ds_fil[dbzh]>=21)&(ds_fil[dbzh]<23)&(ds_fil[rhohv]>rhvmin))).compute().median()
        zdr_zh_24 = (ds_fil[zdr].where((ds_fil[dbzh]>=23)&(ds_fil[dbzh]<25)&(ds_fil[rhohv]>rhvmin))).compute().median()
        zdr_zh_26 = (ds_fil[zdr].where((ds_fil[dbzh]>=25)&(ds_fil[dbzh]<27)&(ds_fil[rhohv]>rhvmin))).compute().median()
        zdr_zh_28 = (ds_fil[zdr].where((ds_fil[dbzh]>=27)&(ds_fil[dbzh]<29)&(ds_fil[rhohv]>rhvmin))).compute().median()
        zdr_zh_30 = (ds_fil[zdr].where((ds_fil[dbzh]>=29)&(ds_fil[dbzh]<31)&(ds_fil[rhohv]>rhvmin))).compute().median()

        zdroffset = xr.concat([zdr_zh_20-.23, zdr_zh_22-.27, zdr_zh_24-.33, zdr_zh_26-.40, zdr_zh_28-.48, zdr_zh_30-.56], dim='dataarrays').sum(dim='dataarrays', skipna=True)/6
    
    if plot_correction:
        try:
            mask = ds_fil[rhohv]>rhvmin
            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            if "time" in ds and timemode=="step":
                hist_2d(ds_fil[dbzh].where(mask)[plot_timestep].values, ds_fil[zdr].where(mask)[plot_timestep].values, bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
            else:
                hist_2d(ds_fil[dbzh].where(mask).values, ds_fil[zdr].where(mask).values, bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
            plt.plot([20,22,24,26,28,30],[.23, .27, .33, .40, .48, .56], color='black')
            plt.title('Non-calibrated $Z_{DR}$')
            plt.xlabel(r'$Z_H$', fontsize=15)
            plt.ylabel(r'$Z_{DR}$', fontsize=15)
            plt.grid(which='both', color='black', linestyle=':', alpha=0.5)
            
            plt.subplot(1,2,2)
            if "time" in ds and timemode=="step":
                hist_2d(ds_fil[dbzh].where(mask)[plot_timestep].values, ds_fil[zdr].where(mask)[plot_timestep].values-zdroffset[plot_timestep].values, bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
            else:
                hist_2d(ds_fil[dbzh].where(mask).values, ds_fil[zdr].where(mask).values-zdroffset, bins1=np.arange(0,40,1), bins2=np.arange(-1,3,.1))
            plt.plot([20,22,24,26,28,30],[.23, .27, .33, .40, .48, .56], color='black')
            plt.title('Calibrated $Z_{DR}$')
            plt.xlabel(r'$Z_H$', fontsize=15)
            plt.ylabel(r'$Z_{DR}$', fontsize=15)
            plt.grid(which='both', color='black', linestyle=':', alpha=0.5)
            if "time" in ds and timemode=="step":
                plt.legend(title=r'$\Delta Z_{DR}$: '+str(np.round(zdroffset[plot_timestep].values,3))+'dB')
            else:
                plt.legend(title=r'$\Delta Z_{DR}$: '+str(np.round(zdroffset,3))+'dB')
        
            plt.tight_layout()
            plt.show()
        except ValueError:
            print("No plotting: No light rain detected with current settings")
    
    # change name of the array
    zdroffset.name="ZDR_offset"
    return zdroffset


def hist_2d(A,B, bins1=35, bins2=35, mini=1, maxi=None, cmap='jet', colsteps=30, alpha=1, mode='absolute', fsize=15, colbar=True):
    """ 
    # Histogram 2d Quicklooks
    # ------------------------
    
    Plotting 2d Histogramm of two varibles
    
    # Input
    # -----
    
    A,B          ::: Variables
    bins1, bins2 ::: x, y bins
    mini, maxi   ::: min and max 
    cmap         ::: colormap
    colsteps     ::: number of cmap steps
    alpha        ::: transperency
    fsize        ::: fontsize
    mode         ::: hist mode
    
    
    # Output
    # ------
    
    2D Histogramm Plot
    
    
    ::: Hist mode:::
    absolute ::: absolute numbers
    relative ::: relative numbers
    relative_with_y ::: relative numbers of y levels
        
    """
    from matplotlib.colors import LogNorm

    # discret cmap
    cmap = plt.cm.get_cmap(cmap, colsteps)
    
    # mask array
    m=~np.isnan(A) & ~np.isnan(B)
    
    if mode=='absolute':
        
        plt.hist2d(A[m], B[m], bins=(bins1, bins2), cmap=cmap, norm=LogNorm( vmin=mini, vmax=maxi), alpha=alpha)
        if colbar==True:
          cb = plt.colorbar(shrink=1, pad=0.01)
          cb.set_label('number of samples', fontsize=fsize)
          cb.ax.tick_params(labelsize=fsize) 
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
    
    if mode=='relative':
        H, xe, ye = np.histogram2d(A[m], B[m], bins=(bins1, bins2)) 
        xm = (xe[0:-1]+ xe[1:len(xe)])/2
        ym = (ye[0:-1]+ ye[1:len(ye)])/2
        nsum = np.nansum(H)
        plt.pcolormesh(xm, ym, 100*(H.T/nsum),  cmap=cmap, norm=LogNorm( vmin=mini, vmax=maxi), alpha=alpha)
        if colbar==True:
          cb = plt.colorbar(shrink=1, pad=0.01)
          cb.set_label('%', fontsize=fsize)
          cb.ax.tick_params(labelsize=fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        
    if mode=='relative_with_y': 
        H, xe, ye = np.histogram2d(A[m], B[m], bins=(bins1, bins2)) 
        xm = (xe[0:-1]+ xe[1:len(xe)])/2
        ym = (ye[0:-1]+ ye[1:len(ye)])/2
        nsum = np.nansum(H, axis=0)
        plt.pcolormesh(xm, ym, 100*(H/nsum).T,  cmap=cmap, norm=LogNorm( vmin=mini, vmax=maxi), alpha=alpha)
        if colbar==True:
          cb = plt.colorbar(shrink=1, pad=0.01)
          cb.set_label('%', fontsize=fsize)
          cb.ax.tick_params(labelsize=fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        
# Calculate phase offset
def phase_offset(phioff, rng=3000.):
    """Calculate Phase offset.

    Parameter
    ---------
    phioff : xarray.DataArray
        differential phase array

    Keyword Arguments
    -----------------
    rng : float
        range in m to calculate system phase offset

    Return
    ------
    xarray.Dataset
        Dataset with variables PHIDP_OFFSET, start_range and stop_range
    """
    range_step = np.diff(phioff.range)[0]
    nprec = int(rng / range_step)
    if nprec % 2:
        nprec += 1

    # create binary array
    phib = xr.where(np.isnan(phioff), 0, 1)

    # take nprec range bins and calculate sum
    phib_sum = phib.rolling(range=nprec, center=True).sum(skipna=True)

    # get start range of first N consecutive precip bins
    start_range = phib_sum.idxmax(dim="range") - nprec // 2 * np.diff(phib_sum.range)[0]
    # get range of first non-nan value per ray
    #start_range = (~np.isnan(phioff)).idxmax(dim='range', skipna=True)
    # add range
    stop_range = start_range + rng
    # get phase values in specified range
    off = phioff.where((phioff.range >= start_range) & (phioff.range <= stop_range),
                       drop=True)
    # calculate nan median over range
    off = off.median(dim='range', skipna=True)
    return xr.Dataset(dict(PHIDP_OFFSET=off,
                           start_range=start_range,
                           stop_range=stop_range))

        
# PHIDP calibration
def phidp_offset_detection(ds, phidp="PHIDP", rhohv="RHOHV", dbzh="DBZH", rhohvmin=0.9,
                           dbzhmin=0., rng=3000.):
    r"""
    Calculate the offset on PHIDP.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset with PHIDP, RHOHV and DBZH.
    phidp : str
        Name of the variable for PHIDP data in ds.
    rhohv : str
        Name of the variable for RHOHV data in ds.
    dbzh : str
        Name of the variable for DBZH data in ds.
    rhohvmin : float
        Minimum value for filtering RHOHV.
    dbzhmin : float
        Minimum value for filtering DBZH.
    rng : float
        range in m to calculate system phase offset

    Returns
    ----------
    ds_offset : xarray Dataset
        xarray Dataset with the detected offset and related satatistics.

    """
    # filter
    phi = ds[phidp].where((ds[rhohv]>=rhohvmin) & (ds[dbzh]>=dbzhmin))
    # calculate offset
    phidp_offset = phi.pipe(phase_offset, rng=rng)
    return phidp_offset

        
# ZDR calibration from VPs. Adapted from Daniel Sanchez-Rivas (TowerPy) to xarray
def zdr_offset_detection_vps(ds, zdr="ZDR", dbzh="DBZH", rhohv="RHOHV", mode="median",
                        mlbottom=None, temp=None, min_h=1000, zhmin=5,
                        zhmax=30, rhvmin=0.98, minbins=2, timemode="step"):
    r"""
    Calculate the offset on :math:`Z_{DR}` using vertical profiles. Only gates 
    below the melting layer bottom (i.e. the rain region below the melting layer)
    or below the given temperature level are included in the method.


    Parameters
    ----------
    ds : xarray Dataset
        Dataset with the vertical scan. ZDR, DBZH and RHOHV are needed. Dimensions 
        should include "azimuth" and "range"
    zdr : str
        Name of the variable in ds for differential reflectivity data. Default is "ZDR".
    dbzh : str
        Name of the variable in ds for horizontal reflectivity data (in dB). Default is "DBZH".
    rhohv : str
        Name of the variable in ds for cross-correlation coefficient data. Default is "RHOHV".
    mode : str
        Method for calculating the offset from the distribution of ZDR values in light rain.
        Can be either "median" or "mean". Default is "median".
    mlbottom : str, int or float
        If str: name of the ML bottom variable in ds. If None, it is assumed to 
        be "height_ml_bottom_new_gia" or "height_ml_bottom" (in that order). 
        If int or float: we assume the ML bottom is not available and we take the given
        int value as the temperature level from which to consider radar bins.
        Only gates below the melting layer bottom (i.e. the rain
        region below the melting layer) are included in the method.
    temp : str, optional
        Name of the temperature variable in ds. Only necessary if mlbottom is int.
        If None is given and mlbottom is int or float, the default name "TEMP" is used.
    min_h : float, optional
        Minimum height of usable data within the polarimetric profiles, in m. This is relative to
        sea level and not relative to the altitude of the radar (in accordance to the "z" coordinate 
        from wradlib.georef.georeference_dataset). The default is 1000. 
    zhmin : float, optional
        Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
        The default is 5.
    zhmax : float, optional
        Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
        The default is 30.
    rhvmin : float, optional
        Threshold on :math:`\rho_{HV}` (unitless) related to light rain.
        The default is 0.98.
    minbins : float, optional
        Minimum number of :math:`Z_{DR}` bins related to light rain.
        The default is 2.
    timemode : str
        How to calculate the offset in case a time dimension is found. "step" calculates one offset
        per timestep. "all" calculates one offset for the whole ds. Default is "step"

    Returns
    ----------
    ds_offset : xarray Dataset
        xarray Dataset with the detected offset and related satatistics.

    Notes
    -----
    1. Based on the method described in [1]_ and [2]_

    References
    ----------
    .. [1] Gorgucci, E., Scarchilli, G., and Chandrasekar, V. (1999),
        A procedure to calibrate multiparameter weather radar using
        properties of the rain medium, IEEE T. Geosci. Remote, 37, 269–276,
        https://doi.org/10.1109/36.739161
    .. [2] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2022): "Calibration
        of radar differential reflectivity using quasi-vertical profiles",
        Atmos. Meas. Tech., 15, 503–520,
        https://doi.org/10.5194/amt-15-503-2022
    """
    
    # We need the ds to be georeferenced in case it is not
    if "z" not in ds.coords:
        ds = ds.pipe(wrl.georef.georeference_dataset)
    
    if mlbottom is None:
        try:
            ml_bottom = ds["z"] < ds["height_ml_bottom_new_gia"]
        except KeyError:
            try:
                ml_bottom = ds["z"] < ds["height_ml_bottom"]
            except KeyError:
                raise KeyError("Melting layer bottom not found in ds. Provide melting layer bottom or temperature level.")
            except:
                raise RuntimeError("Something went wrong when looking for ML variable")
        except:
            raise RuntimeError("Something went wrong when looking for ML variable")
    elif type(mlbottom) is str:
        ml_bottom = ds["z"] < ds[mlbottom]
    elif type(mlbottom) in [float, int]:
        if type(temp) is str:
            ml_bottom = ds[temp] > mlbottom
        elif temp is None:
            try:
                ml_bottom = ds["TEMP"] > mlbottom
            except KeyError:
                raise KeyError("temp is not given and could not be found by default")
        else:
            raise TypeError("temp must be str or None")
    else:
        raise TypeError("mlbottom must be str, int or None")
    
    # get ZDR data and filter values below ML and above min_h
    ds_zdr = ds[zdr]
    ds_zdr = ds_zdr.where(ml_bottom)
    ds_zdr = ds_zdr.where(ds["z"]>min_h)
    
    # Filter according to DBZH and RHOHV limits
    ds_zdr = ds_zdr.where(ds[dbzh]>zhmin).where(ds[dbzh]<zhmax).where(ds[rhohv]>rhvmin)

    if "time" in ds and timemode=="step":
        # Get dimensions to reduce (other than time)
        dims_wotime = [kk for kk in ds_zdr.dims]
        while "time" in dims_wotime:
            dims_wotime.remove("time")
                            
        # Filter according to the minimum number of bins limit
        ds_zdr = ds_zdr.where(ds_zdr.count(dim=dims_wotime)>minbins)
        
        # Calculate offset and others
        if mode=="median":
            zdr_offset = ds_zdr.median(dim=dims_wotime).assign_attrs(
                {"long_name":"ZDR offset from vertical profile (median)", 
                 "standard_name":"ZDR_offset_from_vertical_profile"}
                )
        elif mode=="mean":
            zdr_offset = ds_zdr.mean(dim=dims_wotime).assign_attrs(
                {"long_name":"ZDR offset from vertical profile (mean)", 
                 "standard_name":"ZDR_offset_from_vertical_profile"}
                )
        else:
            raise KeyError("mode must be either 'median' or 'mean'")

        zdr_max = ds_zdr.max(dim=dims_wotime).assign_attrs(
            {"long_name":"ZDR max from offset calculation", 
             "standard_name":"ZDR_max_from_offset_calculation"}
            )
        zdr_min = ds_zdr.min(dim=dims_wotime).assign_attrs(
            {"long_name":"ZDR min from offset calculation", 
             "standard_name":"ZDR_min_from_offset_calculation"}
            )
        zdr_std = ds_zdr.std(dim=dims_wotime).assign_attrs(
            {"long_name":"ZDR standard deviation from offset calculation", 
             "standard_name":"ZDR_std_from_offset_calculation"}
            )
        zdr_sem = ( zdr_std / ds_zdr.count(dim=dims_wotime)**(1/2) ).assign_attrs(
            {"long_name":"ZDR standard error of the mean from offset calculation", 
             "standard_name":"ZDR_sem_from_offset_calculation"}
            )

                
    else:
        # Filter according to the minimum number of bins limit
        ds_zdr = ds_zdr.where(ds_zdr.count()>minbins)

        # Calculate offset and others
        if mode=="median":
            zdr_offset = ds_zdr.compute().median().assign_attrs(
                {"long_name":"ZDR offset from vertical profile (median)", 
                 "standard_name":"ZDR_offset_from_vertical_profile"}
                )
        elif mode=="mean":
            zdr_offset = ds_zdr.compute().mean().assign_attrs(
                {"long_name":"ZDR offset from vertical profile (mean)", 
                 "standard_name":"ZDR_offset_from_vertical_profile"}
                )
        else:
            raise KeyError("mode must be either 'median' or 'mean'")

        zdr_max = ds_zdr.compute().max().assign_attrs(
            {"long_name":"ZDR max from offset calculation", 
             "standard_name":"ZDR_max_from_offset_calculation"}
            )
        zdr_min = ds_zdr.compute().min().assign_attrs(
            {"long_name":"ZDR min from offset calculation", 
             "standard_name":"ZDR_min_from_offset_calculation"}
            )
        zdr_std = ds_zdr.compute().std().assign_attrs(
            {"long_name":"ZDR standard deviation from offset calculation", 
             "standard_name":"ZDR_std_from_offset_calculation"}
            )
        zdr_sem = ( zdr_std / ds_zdr.compute().count()**(1/2) ).assign_attrs(
            {"long_name":"ZDR standard error of the mean from offset calculation", 
             "standard_name":"ZDR_sem_from_offset_calculation"}
            )

            
    # Merge results in a dataset
    ds_offset = xr.Dataset({"ZDR_offset": zdr_offset,
                            "ZDR_max_from_offset": zdr_max,
                            "ZDR_min_from_offset": zdr_min,
                            "ZDR_std_from_offset": zdr_std,
                            "ZDR_sem_from_offset": zdr_sem}
                           )
    
    return ds_offset


# This is not implemented yet
# ZDR calibration from QVPs. Adapted from Daniel Sanchez-Rivas (TowerPy) to xarray
# def offsetdetection_qvps(self, pol_profs, mlyr=None, min_h=0., max_h=5.,
#                          zhmin=0, zhmax=20, rhvmin=0.985, minbins=4,
#                          zdr_0=0.182, stats=False):
#     r"""
#     Calculate the offset on :math:`Z_{DR}` using QVPs, acoording to [1]_.

#     Parameters
#     ----------
#     pol_profs : dict
#         Profiles of polarimetric variables.
#     mlyr : class
#         Melting layer class containing the top and bottom boundaries of
#         the ML.
#     min_h : float, optional
#         Minimum height of usable data within the polarimetric profiles.
#         The default is 0.
#     max_h : float, optional
#         Maximum height of usable data within the polarimetric profiles.
#         The default is 3.
#     zhmin : float, optional
#         Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
#         The default is 0.
#     zhmax : float, optional
#         Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
#         The default is 20.
#     rhvmin : float, optional
#         Threshold on :math:`\rho_{HV}` (unitless) related to light rain.
#         The default is 0.985.
#     minbins : float, optional
#         Consecutive bins of :math:`Z_{DR}` related to light rain.
#         The default is 3.
#     zdr_0 : float, optional
#         Intrinsic value of :math:`Z_{DR}` in light rain at ground level.
#         Defaults to 0.182.
#     stats : dict, optional
#         If True, the function returns stats related to the computation of
#         the :math:`Z_{DR}` offset. The default is False.

#     Notes
#     -----
#     1. Based on the method described in [1]

#     References
#     ----------
#     .. [1] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2022): "Calibration
#         of radar differential reflectivity using quasi-vertical profiles",
#         Atmos. Meas. Tech., 15, 503–520,
#         https://doi.org/10.5194/amt-15-503-2022
#     """
#     if mlyr is None:
#         mlvl = 5
#         mlyr_thickness = 0.5
#         mlyr_bottom = mlvl - mlyr_thickness
#     else:
#         mlvl = mlyr.ml_top
#         mlyr_thickness = mlyr.ml_thickness
#         mlyr_bottom = mlyr.ml_bottom
#     if np.isnan(mlyr_bottom):
#         boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
#                           find_nearest(pol_profs.georef['profiles_height [km]'],
#                                        mlvl-mlyr_thickness)]
#     else:
#         boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
#                           find_nearest(pol_profs.georef['profiles_height [km]'],
#                                        mlyr_bottom)]
#     if boundaries_idx[1] <= boundaries_idx[0]:
#         boundaries_idx = [np.nan]
#     if np.isnan(mlvl) and np.isnan(mlyr_bottom):
#         boundaries_idx = [np.nan]

#     maxheight = find_nearest(pol_profs.georef['profiles_height [km]'],
#                              max_h)

#     if any(np.isnan(boundaries_idx)):
#         self.zdr_offset = 0
#     else:
#         profs = copy.deepcopy(pol_profs.qvps)
#         calzdr_qvps = {k: v[boundaries_idx[0]:boundaries_idx[1]]
#                        for k, v in profs.items()}

#         calzdr_qvps['ZDR [dB]'][calzdr_qvps['ZH [dBZ]'] < zhmin] = np.nan
#         calzdr_qvps['ZDR [dB]'][calzdr_qvps['ZH [dBZ]'] > zhmax] = np.nan
#         calzdr_qvps['ZDR [dB]'][calzdr_qvps['rhoHV [-]'] < rhvmin] = np.nan
#         if np.count_nonzero(~np.isnan(calzdr_qvps['ZDR [dB]'])) <= minbins:
#             calzdr_qvps['ZDR [dB]'] *= np.nan
#         calzdr_qvps['ZDR [dB]'][calzdr_qvps['rhoHV [-]']>maxheight]=np.nan

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             calzdrqvps_mean = np.nanmean(calzdr_qvps['ZDR [dB]'])
#             calzdrqvps_max = np.nanmax(calzdr_qvps['ZDR [dB]'])
#             calzdrqvps_min = np.nanmin(calzdr_qvps['ZDR [dB]'])
#             calzdrqvps_std = np.nanstd(calzdr_qvps['ZDR [dB]'])
#             calzdrqvps_sem = np.nanstd(calzdr_qvps['ZDR [dB]'])/np.sqrt(len(calzdr_qvps['ZDR [dB]']))

#         if not np.isnan(calzdrqvps_mean):
#             self.zdr_offset = calzdrqvps_mean - zdr_0
#         else:
#             self.zdr_offset = 0

#         if stats:
#             self.zdr_offset_stats = {'offset_max': calzdrqvps_max,
#                                      'offset_min': calzdrqvps_min,
#                                      'offset_std': calzdrqvps_std,
#                                      'offset_sem': calzdrqvps_sem,
#                                      }
