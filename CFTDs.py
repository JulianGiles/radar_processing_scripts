#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:53:58 2023

@author: jgiles

Script for calculating CFTDS
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

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


import warnings
warnings.filterwarnings('ignore')

#%% Load QVPs for stratiform-case CFTDs
# This part should be run after having the QVPs computed (compute_qvps.py)

#### Get QVP file list
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps_monthly/*/*/umd/vol5minng01/07/*allmoms*"
path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*"
path_qvps2 = "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/14*/*allmoms*"

files = sorted(glob.glob(path_qvps))+sorted(glob.glob(path_qvps2))

#### Set variable names
X_DBZH = "DBZH"
X_RHOHV = "RHOHV"
X_ZDR = "ZDR_OC"
X_KDP = "KDP_ML_corrected"

if "dwd" in files[0]:
    country="dwd"
    X_TH = "TH"
elif "dmi" in files[0]:
    country="dmi"
    X_TH = "DBZH"


# there are slight differences (noise) in z coord sometimes so we have to align all datasets
# since the time coord has variable length, we cannot use join="override" so we define a function to copy
# the z coord from the first dataset into the rest with preprocessing
# There are also some time values missing, ignore those
# Some files do not have TEMP data, fill with nan
first_file = xr.open_mfdataset(files[0]) 
first_file_z = first_file.z.copy()
def fix_z_and_time(ds):
    ds.coords["z"] = first_file_z
    ds = ds.where(ds["time"].notnull(), drop=True)
    if "TEMP" not in ds.coords:
        ds.coords["TEMP"] = xr.full_like( ds["DBZH"], np.nan ).compute()
        
    return ds
    
try:
    qvps = xr.open_mfdataset(files, preprocess=fix_z_and_time)
except: 
    # if the above fails, just combine everything and fill the holes with nan (Turkish case)
    qvps = xr.open_mfdataset(files, combine="nested", concat_dim="time")

# Fill missing values in ZDR_OC with ZDR
if X_ZDR == "ZDR_OC":
    qvps[X_ZDR] = qvps[X_ZDR].where(qvps[X_ZDR].notnull(), qvps["ZDR"])
#%% Filters (conditions for stratiform)
# Filter only stratiform events (min entropy >= 0.8) and ML detected
# with ProgressBar():
#     qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull(), drop=True).compute()

# Filter only stratiform events (min entropy >= 0.8 and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull()), drop=True)
# Filter relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[X_TH] > 0 )&
                                  (qvps_strat[X_KDP] > 0)&
                                  (qvps_strat[X_RHOHV] > 0.7)&
                                  (qvps_strat[X_ZDR] > -1) &
                                  (qvps_strat[X_ZDR] < 3))

try: 
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRHC"]>10)
except:
    print("Could not filter out low SNR")

#### General statistics
values_sfc = qvps_strat_fil.isel({"z": 2})
values_snow = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_new_gia"]}, method="nearest")
values_rain = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_bottom_new_gia"]}, method="nearest")
    
#### ML statistics
# select values inside the ML
qvps_ML = qvps_strat_fil.where( (qvps_strat_fil["z"] < qvps_strat_fil["height_ml_new_gia"]) & \
                               (qvps_strat_fil["z"] > qvps_strat_fil["height_ml_bottom_new_gia"]), drop=True)

values_ML_max = qvps_ML.max(dim="z")
values_ML_min = qvps_ML.min(dim="z")
values_ML_mean = qvps_ML.mean(dim="z")
ML_thickness = qvps_ML["height_ml_new_gia"] - qvps_ML["height_ml_bottom_new_gia"]

# Silke style
# select timesteps with detected ML
gradient_silke = qvps_strat_fil.where(qvps_strat_fil["height_ml_new_gia"] > qvps_strat_fil["height_ml_bottom_new_gia"], drop=True)
gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
beta = gradient_final[X_TH] #### TH OR DBZH??


#### DGL statistics
# select values in the DGL 
qvps_DGL = qvps_strat_fil.where((qvps_strat_fil["TEMP"] >= -20)&(qvps_strat_fil["TEMP"] <= -10), drop=True)    

values_DGL_max = qvps_DGL.max(dim="z")
values_DGL_min = qvps_DGL.min(dim="z")
values_DGL_mean = qvps_DGL.mean(dim="z")

#%% CFADs Plot

# adjustment from K to C (disables now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit
ytlim=-20

# Temp bins
tb=1# degress C

# Min counts per Temp layer
mincounts=10

#Colorbar limits and step
cblim=[0,10]
colsteps=10


# Plot horizontally
# DMI
# Native worst-resolution of the data (for 1-byte moments)
# DBZH: 0.5 dB
# ZDR: 0.0625 dB
# KDP: complicated. From 0.013 at KDP approaching zero to 7.42 at extreme KDP. KDP min absolute value is 0.25 and max abs is 150 (both positive and negative)
# RHOHV: scales with a square root (finer towars RHOHV=1), so from 0.00278 at RHOHV=0.7 to 0.002 resolution at RHOHV=1
# PHIDP: 0.708661 deg
if country=="dmi":

    vars_to_plot = {"DBZH": [0, 51, 1],
                    "ZDR_OC": [-1.05, 3.1, 0.1],
                    "KDP_ML_corrected": [0, 0.51, 0.01],
                    "RHOHV": [0.9, 1.002, 0.002]}
    
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size
    
    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        rd=10 # arbitrarily large decimal position to round to (so it is actually not rounded)
        if vv == "DBZH":
            so=False
            binsx2 = [0, 51, 1]
            rd = 1 # decimal position to round to
        if vv == "ZDR_OC":
            so=True
            binsx2 = [-1, 3.1, 0.1]
            rd=1
        if vv =="RHOHV":
            so = True
            binsx2 = [0.9, 1.005, 0.005]
            rd=3
        utils.hist2d(ax[nn], qvps_strat_fil[vv].round(rd), qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                     binsx=vars_to_plot[vv], binsy=[-20,15,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps, 
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)
        
        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts
    
    
    
    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')



# DWD
if country=="dwd":
    
    vars_to_plot = {"DBZH": [0, 51, 1], 
                    "ZDR_OC": [-1, 3.1, 0.1],
                    "KDP_ML_corrected": [0, 0.51, 0.01],
                    "RHOHV": [0.9, 1.004, 0.004]}

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size
    
    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        if vv =="RHOHV":
            so = True
            binsx2 = [0.9, 1.005, 0.005]
        utils.hist2d(ax[nn], qvps_strat_fil[vv], qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                     binsx=vars_to_plot[vv], binsy=[-20,15,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps, 
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)
        
        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts
    
    
    
    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')


#%% Check particular dates
qvps.loc[{"time":"2015-09-30"}].DBZH.dropna("z", how="all").plot(x="time", ylim=(0,10000))