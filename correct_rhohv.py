#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:16 2023

@author: jgiles

Script for noise-correcting RHOHV.

"""


import os
os.chdir('/home/jgiles/')


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import datatree as dttree
import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import datetime as dt
import pandas as pd
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


#%% Set paths and options

loc="tur"
path="/automount/realpep/upload/jgiles/dwd/2017/2017-09/2017-09-30/tur/vol5minng01/01/*hd5" # for QVPs: /home/jgiles/
dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV

# get the files and check that it is not empty
files = sorted(glob.glob(path))
if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

#%% Load data

for ff in files:
    if "dwd" in ff:
        data=dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()
    else:
        data=xr.open_dataset(ff)
        
    # fix time dim in case some value is NaT
    if data.time.isnull().any():
        data.coords["time"] = data["rtime"].min(dim="azimuth", skipna=True).compute()

    # take time out of the coords if necessary
    for coord in ["latitude", "longitude", "altitude", "elevation"]:
        if "time" in data[coord].dims:
            data.coords[coord] = data.coords[coord].median("time")
    

#%% Calculate RHOHV correction
    # get DBZH name
    for X_DBZH in dbzh_names:
        if X_DBZH in data.data_vars:
            break
    
    # get RHOHV name
    for X_RHO in rhohv_names:
        if X_RHO in data.data_vars:
            break

    rho_nc = utils.calculate_noise_level(data[X_DBZH], data[X_RHO])
