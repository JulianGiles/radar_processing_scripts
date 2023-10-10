#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:36:23 2023

@author: jgiles

Plot PPIs, QVPs, line plots, etc
"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

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
import xradar as xd
import cmweather

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator



import warnings
warnings.filterwarnings('ignore')

# we define a funtion to look for loc inside a path string
def find_loc(locs, path):
    components = path.split(os.path.sep)
    for element in locs:
        for component in components:
            if element.lower() in component.lower():
                return element
    return None

locs = ["pro", "tur", "umd", "afy", "ank", "gzt", "hty", "svs"]

#%% Load data
ff="/automount/realpep/upload/jgiles/dmi/2015/2015-09/2015-09-30/ANK/MON_YAZ_K/12.0/MON_YAZ_K-allmoms-12.0-20152015-092015-09-30-ANK-h5netcdf.nc"
data=xr.open_dataset(ff)

# fix time dim in case some value is NaT
if data.time.isnull().any():
    data.coords["time"] = data["rtime"].min(dim="azimuth", skipna=True).compute()

# take time out of the coords if necessary
for coord in ["latitude", "longitude", "altitude", "elevation"]:
    if "time" in data[coord].dims:
        data.coords[coord] = data.coords[coord].min("time")


#%% Plot PPI

tsel = "2015-09-30T08:04"
datasel = data.loc[{"time": tsel}].pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "KDP"

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
datasel[mom][0].wrl.plot(x="x", y="y", cmap=cmap, norm=norm, xlim=(-25000,25000), ylim=(-25000,25000))
