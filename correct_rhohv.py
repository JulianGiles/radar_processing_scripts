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

import time
start_time = time.time()

#%% Set paths and options
# paths for testing
# path="/automount/realpep/upload/jgiles/dwd/2017/2017-09/2017-09-30/tur/vol5minng01/01/*hd5" # for QVPs: /home/jgiles/
# path="/automount/realpep/upload/jgiles/dmi//2016/2016-05/2016-05-22/AFY/VOL_B/7.0/*h5*" # for QVPs: /home/jgiles/

path0 = sys.argv[1]

if "hd5" in path0 or "h5" in path0:
    files=[path0]
elif "dwd" in path0:
    files = sorted(glob.glob(path0+"/*hd5*"))
elif "dmi" in path0:
    files = sorted(glob.glob(path0+"/*h5*"))
else:
    print("Country code not found in path")
    sys.exit("Country code not found in path.")


dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV

# get the files and check that it is not empty
if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

#%% Load data

for ff in files:
    print("processing "+ff)
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

    # check that the variables actually exist, otherwise continue
    if X_DBZH not in data.data_vars:
        print("DBZH not found in data")
        sys.exit("DBZH not found in data.")
    if X_RHO not in data.data_vars:
        print("RHOHV not found in data")
        sys.exit("RHOHV not found in data.")

    rho_nc = utils.calculate_noise_level(data[X_DBZH], data[X_RHO], noise=(-45, -15, 1))

    # get the "best" noise correction level (acoording to the min std)
    ncl = rho_nc[-1]
    
    # get index of the best correction
    bci = np.array(rho_nc[-2]).argmin()
    
    # merge into a single array
    rho_nc_out = xr.merge(rho_nc[0][bci])
    
    # add noise correction level as attribute
    rho_nc_out.attrs["noise correction level"]=ncl
    
    # Just in case, calculate again for a NCL slightly lower (2%), in case the automatically-selected one is too strong
    rho_nc2 = utils.noise_correction2(data[X_DBZH], data[X_RHO], ncl*1.02)
    
    # make a new array as before
    rho_nc_out2 = xr.merge(rho_nc2)
    rho_nc_out2.attrs["noise correction level"]=ncl*1.02

    # create saving directory if it does not exist
    if "dwd" in ff:
        country="dwd"
    elif "dmi" in ff:
        country="dmi"
    else:
        print("Country code not found in path")
        sys.exit("Country code not found in path.")
    
    ff_parts = ff.split(country)
    savepath = (country+"/rhohv_nc/").join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)

    # copy encoding from DWD to reduce file size
    rho_nc_out["RHOHV_NC"].encoding = data[X_RHO].encoding
    rho_nc_out2["RHOHV_NC"].encoding = data[X_RHO].encoding
    if country=="dwd": # special treatment for SNR since it may not be available in turkish data
        rho_nc_out["SNRH"].encoding = data["SNRHC"].encoding
        rho_nc_out2["SNRH"].encoding = data["SNRHC"].encoding
    else:
        rho_nc_dwd = xr.open_dataset(ff_parts[0]+"dwd/rhohv_nc/2015/2015-01/2015-01-01/pro/vol5minng01/00/ras07-vol5minng01_sweeph5onem_rhohv_nc_00-2015010100005900-pro-10392-hd5")
        rho_nc_out["SNRH"].encoding = rho_nc_dwd["SNRH"].encoding
        rho_nc_out2["SNRH"].encoding = rho_nc_dwd["SNRH"].encoding

    # save the arrays
    filename = ("rhohv_nc").join(savepath.split("allmoms"))
    rho_nc_out.to_netcdf(filename)
    
    filename = ("rhohv_nc_2percent").join(savepath.split("allmoms"))
    rho_nc_out2.to_netcdf(filename)
    
#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")
