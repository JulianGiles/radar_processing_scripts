#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:23:05 2024

@author: jgiles

This scripts tests different tweaks to arrive to a good processed and cleaned PHIDP.
"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER?)

import datatree as dttree
import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import datetime as dt
import pandas as pd
import datetime
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
from cartopy import crs as ccrs
import xradar as xd
import cmweather
import hvplot
import hvplot.xarray
import holoviews as hv
# hv.extension("bokeh", "matplotlib") # better to put this each time this kind of plot is needed
import scipy

import panel as pn
from bokeh.resources import INLINE
from osgeo import osr

from functools import partial

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator


os.environ['WRADLIB_DATA'] = '/home/jgiles/wradlib-data-main'
# set earthdata token (this may change, only lasts a few months https://urs.earthdata.nasa.gov/users/jgiles/user_tokens)
os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTcwMzMzMjE5NywiaWF0IjoxNjk4MTQ4MTk3LCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.6DB5JJ9vdC7Vvwvaa7_mb_HbpVAh05Gz26dzdateN10C5lAd2X4a1_zClx7KkTpyoeVZSzkGSgtcd5Azc_btG0am4r2aJDGv4Zp4Vg55G4mcZMp-aTR7D520InQLMvqFacVO5wwmvfNWzMT4TyLGcXwPuX58s1oaFR5gRL9T30pXN9nEs-1aJg4LUl553PfdOvvom3q-JKXFtSTE2nLyEQOzWW36COl1aHwq6Wh4ykn4aq6ppTVAIeHdgkjtnQtxbhd9trm16fSbX9HIgG7n-drnz_v-WMeFuycMHa-zLDKnd3U3oZW6XAUq2akw2ddu6ChwoTZ4Ix2di7fudioo9Q"

import warnings
warnings.filterwarnings('ignore')

#%% Load data

ff = "/automount/realpep/upload/jgiles/dwd/*/*/2017-07-25/pro/vol5minng01/07/*allmoms*"
ds = utils.load_dwd_preprocessed(ff).pipe(wrl.georef.georeference)

#%% Kai processing

# 1) Masking
# msk_swp = ds.where(ds.DBTH >= 10).where(ds.DBTH <= 40).where(ds.RHOHV >= 0.9)
msk_swp = ds.where(ds.RHOHV >= 0.9)

# 2) Median filtering
medwin = [1,3] # We do [1,3] to only filter in the range dim. Originally by Kai: 11
# median filtering 2d
phimed = msk_swp.UPHIDP.copy()
# phimed = phimed.pipe(radarmet.filter_data, medwin)
phimed =         xr.apply_ufunc(scipy.signal.medfilt2d,
                              phimed, 
                              input_core_dims=[["azimuth","range"]],
                              output_core_dims=[["azimuth","range"]],
                              dask='parallelized',
                              kwargs=dict(kernel_size=medwin),
                              dask_gufunc_kwargs=dict(allow_rechunk=True),
                               vectorize=True
                              )

# 3) Gaussian smoothing 
# gaussian convolution 1d - smoothing
# play with sigma and kwidth
kwidth = 7 #31
sigma = 7 #10
gkern = utils.gauss_kernel(kwidth, sigma)

# phiclean = phimed.pipe(utils.smooth_data, gkern)
phiclean = xr.apply_ufunc(utils.smooth_data, phimed.compute(), kwargs=dict(kernel=gkern),
                          input_core_dims=[["azimuth","range"]], output_core_dims=[["azimuth","range"]],
                          vectorize=True)

# ALTERNATIVE) istead of 2) and 3) just do a 2D filtering (xr_rolling median) This should be the same as 2) (if window is the same)!!!
window = 7
window2 = None
phi_median = msk_swp.UPHIDP.copy().pipe(utils.xr_rolling, window, window2=window2, method='median', skipna=True, min_periods=window//2+1)

# apply also gauss kernel to phi_median?
phi_median_clean =  xr.apply_ufunc(utils.smooth_data, phi_median.compute(), kwargs=dict(kernel=gkern),
                          input_core_dims=[["azimuth","range"]], output_core_dims=[["azimuth","range"]],
                          vectorize=True)


# # 4) PHIDP offset correction
# phi_fix = phi_median.copy()
# # find start_range
# start = phi_fix.range < off.start_range
# # fix offset for ranges >= start_range
# phi_fix = phi_fix.where((phi_fix.range >= off.start_range)) - phi_offset1
# # set phi to 0 for ranges < start_range 
# phi_fix = xr.where(start, 0, phi_fix).transpose("azimuth", "range")

#%% Alexander processing

# 1) and 2) like Kai

# 3) Check that the valid intervals have sufficient length (at least equal to the moving average window)

def count_and_filter_segments(da, min_length=7):
    # Identify valid values (not NaN)
    valid_mask = ~np.isnan(da)

    # Find the indices where the valid_mask changes
    changes = np.diff(valid_mask.astype(int))
    
    # Get start and end indices of segments
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    
    # Adjust for segments that start or end at the edges
    if valid_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if valid_mask[-1]:
        end_indices = np.append(end_indices, len(valid_mask))
    
    # Calculate segment lengths
    segment_lengths = end_indices - start_indices

    # Create a mask for segments longer than min_length
    long_segment_mask = np.zeros_like(valid_mask, dtype=bool)
    for start, length in zip(start_indices, segment_lengths):
        if length >= min_length:
            long_segment_mask[start:start + length] = True

    # Apply the mask to the original DataArray
    try:
        filtered_da = da.where(xr.DataArray(long_segment_mask, dims=["range"]), drop=False)
    except:
        filtered_da = np.where(long_segment_mask, da, np.nan)

    return filtered_da

phimed_clean = xr.apply_ufunc(count_and_filter_segments, phimed.compute(), kwargs=dict(min_length=window),
                          input_core_dims=[["range"]], output_core_dims=[["range"]],
                          vectorize=True)

# 4) calculate PHIDP for the good intervals with a running average

phi_mean = phimed_clean.rolling(range=window, min_periods=window//2+1, center=True).mean(skipna=True)

# 5) fill in the gaps with linear interpolation

phi_mean_interp = phi_mean.interpolate_na(dim="range", method="linear")

# 6) fill the edges

phi_mean_interp_fill = phi_mean_interp.bfill("range").ffill("range")

#%% Comparison plots
#%% Comparison between medfilt2d and xr_rolling
ts = 35 # timestep
az = 0 # azimuth

# check that the window size is the same
msk_swp.UPHIDP.copy()[ts,az].plot(marker="x", label="masked UPHIDP")
phimed[ts, az].plot(marker="x", label="medfil (win="+str(medwin)+") masked UPHIDP")
msk_swp.UPHIDP.copy().pipe(utils.xr_rolling, 3, window2=None, method='median', skipna=True, min_periods=3)[ts,az].plot(marker=".", label="roll med masked UPHIDP"); 

plt.legend(fontsize=6)

# indeed the result is the same (be careful that both functions work in 2d and we are only trying to smooth in range)

#%% Radial comparison between methods
ts = 35 # timestep
az = 0 # azimuth

ds.UPHIDP[ts, az].plot(marker="x", label="UPHIDP")
msk_swp.UPHIDP[ts, az].plot(marker="x", label="masked UPHIDP")
phimed[ts, az].plot(marker="x", label="medfil (win="+str(medwin)+") masked UPHIDP")
phiclean[ts, az].plot(marker="x", label="Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") medfil (win="+str(medwin)+") masked UPHIDP")

phi_median[ts, az].plot(marker=".", label="roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")
phi_median_clean[ts, az].plot(marker=".", label="Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")

phi_mean_interp_fill[ts,az].plot(marker="x", label="A.Ryzhkov style medfil (win="+str(medwin)+") masked UPHIDP",
                                 ylim=(8,20), xlim=(-2,30000))

plt.legend(fontsize=6)
plt.title(str.swapcase(utils.find_loc(utils.locs, glob.glob(ff)[0]))+" "+str(phi_mean_interp_fill[ts,az].time.values)[0:19]+" azimuth: "+str(phi_mean_interp_fill[ts,az].azimuth.values))

#%% PPI plots
ts = 35 # timestep

ds.UPHIDP[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("UPHIDP")
plt.show()

msk_swp.UPHIDP[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("masked UPHIDP")
plt.show()

phimed[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("medfil (win="+str(medwin)+") masked UPHIDP")
plt.show()

phiclean[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") \n medfil (win="+str(medwin)+") masked UPHIDP")
plt.show()

phi_median[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")
plt.show()

phi_median_clean[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") \n roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")
plt.show()

phi_mean_interp_fill[ts].wrl.vis.plot(vmin=0, vmax=30)
plt.title("A.Ryzhkov style medfil (win="+str(medwin)+") masked UPHIDP")
plt.show()

#%% Calculate KDP from the different PHIDPs
winlen0=7

kdp_phimed = utils.kdp_phidp_vulpiani(phimed, winlen0, min_periods=winlen0//2+1)[1]
kdp_phiclean = utils.kdp_phidp_vulpiani(phiclean, winlen0, min_periods=winlen0//2+1)[1]
kdp_phi_median = utils.kdp_phidp_vulpiani(phi_median, winlen0, min_periods=winlen0//2+1)[1]
kdp_phi_median_clean = utils.kdp_phidp_vulpiani(phi_median_clean, winlen0, min_periods=winlen0//2+1)[1]
kdp_phi_mean_interp_fill = utils.kdp_phidp_vulpiani(phi_mean_interp_fill, winlen0, min_periods=winlen0//2+1)[1]

#%% Comparison plots KDP
#%% Radial comparison 
ts = 35 # timestep
az = 0 # azimuth

kdp_phimed[ts, az].plot(marker="x", label="KDP medfil (win="+str(medwin)+") masked UPHIDP")
kdp_phiclean[ts, az].plot(marker="x", label="KDP Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") \n medfil (win="+str(medwin)+") masked UPHIDP")

kdp_phi_median[ts, az].plot(marker=".", label="KDP roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")
kdp_phi_median_clean[ts, az].plot(marker=".", label="KDP Gauss smoothed (kwidth="+str(kwidth)+", sigma="+str(sigma)+") roll med (rangewin="+str(window)+", azwin="+str(window2)+") masked UPHIDP")

kdp_phi_mean_interp_fill[ts,az].plot(marker="x", label="KDP A.Ryzhkov style medfil (win="+str(medwin)+") masked UPHIDP",
                                 ylim=(-0.1, 0.4), xlim=(-2,30000))

plt.legend(fontsize=6)
plt.title(str.swapcase(utils.find_loc(utils.locs, glob.glob(ff)[0]))+" "+str(phi_mean_interp_fill[ts,az].time.values)[0:19]+" azimuth: "+str(phi_mean_interp_fill[ts,az].azimuth.values))

