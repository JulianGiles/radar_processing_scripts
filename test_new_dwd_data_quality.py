#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:27:30 2023

@author: jgiles

Script to test the quality of the DWD data (specially KDP) after the
resolution upgrade in 2021
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


#%% Load new and old data

# old data
f0 = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/01/*allmoms*"
ff0 = sorted(glob.glob(f0))[0]

dwd0 = dttree.open_datatree(ff0)["sweep_1"].to_dataset()
# It may happen that some time value is missing, fix that using info in rtime
if dwd0["time"].isnull().any():
    dwd0.coords["time"] = dwd0.rtime.min(dim="azimuth", skipna=True).compute()    
    
# if some coord has dimension time, reduce using median
for coord in ["latitude", "longitude", "altitude", "elevation"]:
    if "time" in dwd0[coord].dims:
        dwd0.coords[coord] = dwd0.coords[coord].median("time")

dwd0 = dwd0.pipe(wrl.georef.polar.georeference)

# new data
f1 = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2021/2021-04/2021-04-11/pro/vol5minng01/01/*"
ff1 = sorted(glob.glob(f1))

# extract list of moments 
moments = set(fp.split("_")[-2] for fp in ff1)

# discard "allmoms" from the set if it exists
moments.discard("allmoms")

# define preprocessing function to align coords
def align(ds):
    ds["time"] = ds["time"].load().min() # reduce time in the azimuth
    ds["elevation"] = ds["elevation"].load().median() # remove elevation in time
    ds["azimuth"] = ds["azimuth"].load().round(1)
    ds = xd.util.remove_duplicate_rays(ds) # in case there are duplicate rays, remove them
    ds["time"] = np.unique(ds["time"]) # in case there are duplicate times
    return ds.set_coords(["sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"])

# dwd1 = xr.open_mfdataset(ff1, engine="odim", combine="nested", concat_dim="time", preprocess=align) 

try:
    # for every moment, open all files in folder (all timesteps) per moment into a dataset
    vardict = {} # a dict for putting a dataset per moment
    for mom in moments:
        
        # print("       Processing "+mom)
        
        # open the odim files (single moment and elevation, several timesteps)
        llmom = sorted([ff for ff in ff1 if "_"+mom+"_" in ff])
        
        vardict[mom] = xr.open_mfdataset(llmom, engine="odim", combine="nested", concat_dim="time", preprocess=align) 
        
        # It may happen that some time value is missing, fix that using info in rtime
        if vardict[mom]["time"].isnull().any():
            vardict[mom].coords["time"] = vardict[mom].rtime.min(dim="azimuth", skipna=True).compute()    
            
        # if some coord has dimension time, reduce using median
        for coord in ["latitude", "longitude", "altitude", "elevation"]:
            if "time" in vardict[mom][coord].dims:
                vardict[mom].coords[coord] = vardict[mom].coords[coord].median("time")
        
except OSError:
    pathparts = [ xx if len(xx)==8 and "20" in xx else None for xx in llmom[0].split("/") ]
    pathparts.sort(key=lambda e: (e is None, e))
    date = pathparts[0]
    print(date+" "+mom+": Error opening files. Some file is corrupt or truncated.")
    sys.exit("Script terminated early. "+date+" "+mom+": Error opening files. Some file is corrupt or truncated.")

# merge all moments
dwd1 = xr.merge(vardict.values())

dwd1 = dwd1.pipe(wrl.georef.georeference)

#%% Compare variables in PPI

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "UPHIDP"

t0 = 10
dwd0[mom][t0].wrl.plot(x="x", y="y")

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
t1 = 10
dwd1[mom][t1].wrl.plot(x="x", y="y", cmap=cmap, norm=norm)

#%% Compare variables along a ray

mom = "UPHIDP"

t0 = 10
az0 = 270
dwd0[mom][t0, az0].wrl.plot(xlim=(0, 20000), ylim=(10, 40))

t1 = 10
az1 = 90
dwd1[mom][t1, az1].wrl.plot(xlim=(0, 20000), ylim=(130, 170), marker="x")
