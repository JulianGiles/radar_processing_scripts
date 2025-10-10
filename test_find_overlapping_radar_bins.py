#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:01:08 2025

@author: jgiles
"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER?)

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
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%% Load final ppis (resulting from compute_qvps_new.py)

ff1 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2016/2016-12/2016-12-01/HTY/MON_YAZ_C/12.0/MON_YAZ_C-allmoms-12.0-20162016-122016-12-01-HTY-h5netcdf.nc"
ff2 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2016/2016-12/2016-12-01/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2016-12-01-GZT-h5netcdf.nc"

ds1 = xr.open_dataset(ff1)
ds2 = xr.open_dataset(ff2)

#%% Get PPIs into the same reference system

proj = utils.get_common_projection(ds1, ds2)

ds1 = wrl.georef.georeference(ds1, crs=proj)
ds2 = wrl.georef.georeference(ds2, crs=proj)

#%% Plot PPIs together

tsel = "2016-12-01T14"

ds1.sel(time=tsel, method="nearest")["DBZH"].wrl.vis.plot(cmap="viridis")
ax = plt.gca()
ds2.sel(time=tsel, method="nearest")["DBZH"].wrl.vis.plot(ax=ax, alpha=0.4, xlim=(-100000,0), ylim=(-100000,0))

#%% Get matching volumes

mask1, mask2, mask1_nn, mask2_nn = utils.find_radar_overlap(ds1, ds2, tolerance=500, tolerance_time=4*60)

#%% Plot matched volumes

tsel = "2016-12-01T13:57"

ds1["DBZH"].where(mask1).sel(time=tsel, method="nearest").wrl.vis.plot(cmap="viridis")
ax = plt.gca()
ds2["DBZH"].where(mask2).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.4,  xlim=(-100000,0), ylim=(-100000,0))

#%% Refine the mask to valid-valid DBZH values

mask1_ref, mask2_ref = utils.find_refined_radar_overlap(ds1, ds2, mask1, mask2, var_name="DBZH",
                                                             tolerance=500, tolerance_time=4*60)

#%% Plot matched volumes with refinement

tsel = "2016-12-01T13:50"

ds1["DBZH"].where(mask1_ref).sel(time=tsel, method="nearest").wrl.vis.plot(cmap="viridis")
ax = plt.gca()
ds2["DBZH"].where(mask2_ref).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.4, xlim=(-100000,0), ylim=(-100000,0))
