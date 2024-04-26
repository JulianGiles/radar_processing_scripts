#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:25:08 2024

@author: jgiles

Script for comparing precipitation datasets at different scales.
"""

import matplotlib

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs	
import cartopy.feature 	
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from numpy import ma 
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.colors as mcolors
import os
os.chdir('/home/jgiles/radarmeteorology/notebooks/')
from matplotlib import gridspec
import scipy.stats
from scipy import signal
import gc
from dask.diagnostics import ProgressBar
import cordex as cx
import wradlib as wrl
import glob
import regionmask as rm
from cdo import Cdo

import warnings
#ignore by message
# warnings.filterwarnings("ignore", message="Default reduction dimension will be changed to the grouped dimension")
# warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
# warnings.filterwarnings("ignore", message="Setting the 'color' property will override the edgecolor or facecolor properties.")
# warnings.filterwarnings("ignore", category=FutureWarning)

try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator


proj = ccrs.PlateCarree(central_longitude=0.0)

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

#%% YEARLY analysis

#%%% Load yearly datasets

loadpath_yearly = "/automount/agradar/jgiles/gridded_data/yearly/"
paths_yearly = {
    "IMERG-V07B-monthly": loadpath_yearly+"IMERG-V07B-monthly/IMERG-V07B-monthly_precipitation_yearlysum_2000-2023.nc",
    "IMERG-V06B-monthly": loadpath_yearly+"IMERG-V06B-monthly/IMERG-V06B-monthly_precipitation_yearlysum_2000-2021.nc",
    # "IMERG-V07B-30min": loadpath_yearly+, 
    # "IMERG-V06B-30min": loadpath_yearly+, 
    "CMORPH-daily": loadpath_yearly+"CMORPH-daily/CMORPH-daily_precipitation_yearlysum_1998-2023.nc",
    "TSMP-old": loadpath_yearly+"TSMP-old/TSMP-old_precipitation_yearlysum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_yearly+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_yearlysum_2000-2022.nc",
    "ERA5-monthly": loadpath_yearly+"ERA5-monthly/ERA5-monthly_precipitation_yearlysum_1979-2020.nc",
    # "ERA5-hourly": loadpath_yearly+,
    "RADKLIM": loadpath_yearly+"RADKLIM/RADKLIM_precipitation_yearlysum_2001-2022.nc",
    "RADOLAN": loadpath_yearly+"RADOLAN/RADOLAN_precipitation_yearlysum_2006-2022.nc",
    "EURADCLIM": loadpath_yearly+"EURADCLIM/EURADCLIM_precipitation_yearlysum_2013-2020.nc",
    "GPCC-monthly": loadpath_yearly+"GPCC-monthly/GPCC-monthly_precipitation_yearlysum_2001-2020.nc",
    "GPCC-daily": loadpath_yearly+"GPCC-daily/GPCC-daily_precipitation_yearlysum_2000-2020.nc",
    "GPROF": loadpath_yearly+"GPROF/GPROF_precipitation_yearlysum_2014-2023.nc",
    "HYRAS": loadpath_yearly+"HYRAS/HYRAS_precipitation_yearlysum_1931-2020.nc", 
    }

data_yearlysum = {}

# reload the datasets
for dsname in paths_yearly.keys():
    data_yearlysum[dsname] = xr.open_dataset(paths_yearly[dsname])

#%%% Area means of the yearly values

# Means over Germany
