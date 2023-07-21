#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:16 2023

@author: jgiles

Script for calculating ZDR calibration from different methods.

1. “Bird bathing” with vertically pointing radar

2. Using light rain as a natural calibrator with intrinsic average ZDR of 0.25 dB for Z = 20 – 22 dBZ

3. NOT IMPLEMENTED! Using dry aggregated snow as a natural calibrator with intrinsic ZDR of 0.1 – 0.2 dB

4. NOT IMPLEMENTED! Bragg scattering with intrinsic ZDR equal to 0 dB

5. NOT IMPLEMENTED! Ground clutter, if stable

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

calib_type = 1 # Like the numbers from the description above
qvpelev = "07" # which QVP to use in case calib_type is 2. 07 is 12 deg in DWD.
loc="tur"
countryws="dwd" # dwd for Germany, dmi for Turkey
date="2017-09-30" # if a date, process a single date. If "", process everything.
basepath="/automount/realpep/upload/jgiles/" # for QVPs: /home/jgiles/
min_hgt = 600 # minimum height above the radar to be considered when calculating ZDR offset
phidp_names = ["UPHIDP", "PHIDP"] # names to look for the PHIDP variable, in order of preference
dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV
zdr_names = ["ZDR"]

# check that countryws is set correctly
if countryws not in ["dwd", "dmi"]:
    print("incorrect country weather service name")
    sys.exit("incorrect country weather service name")

# get path according to selected options
if calib_type==1:
    if countryws=="dwd":
        if date=="":
            path = basepath+"/"+countryws+"/*/*/*/"+loc+"/90gradstarng01/00/*hd5"
        else:
            path = basepath+"/"+countryws+"/"+date[0:4]+"/"+date[0:7]+"/"+date+"/"+loc+"/90gradstarng01/00/*hd5"
            
    elif countryws=="dmi":
        print("There are no vertical profiles in turkish data. Calibration method 1 not possible.")
        sys.exit("There are no vertical profiles in turkish data. Calibration method 1 not possible.")

elif calib_type==2:
    if countryws=="dwd":
        if date=="":
            path = basepath+"/"+countryws+"/qvps/*/*/*/"+loc+"/vol5minng01/"+qvpelev+"/*hd5"
        else:
            path = basepath+"/"+countryws+"/"+date[0:4]+"/"+date[0:7]+"/"+date+"/"+loc+"/vol5minng01/"+qvpelev+"/*hd5"
            
    elif countryws=="dmi":
        if date=="":
            path = basepath+"/"+countryws+"/qvps/*/*/*/"+loc+"/*/"+qvpelev+"/*hd5"
        else:
            path = basepath+"/"+countryws+"/"+date[0:4]+"/"+date[0:7]+"/"+date+"/"+loc+"/*/"+qvpelev+"/*hd5"

else:
    print("Calibration method not implemented. Possible options are 1 or 2")
    sys.exit("Calibration method not implemented. Possible options are 1 or 2")

# get the files and check that it is not empty
files = sorted(glob.glob(path))
if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

#%% Load data

for ff in files:
    if calib_type==1 and countryws=="dwd":
        data=dttree.open_datatree(ff)["sweep_0"].to_dataset()
    else:
        data=xr.open_dataset(ff)
        
    # fix time dim in case some value is NaT
    if data.time.isnull().any():
        data.coords["time"] = data["rtime"].min(dim="azimuth", skipna=True).compute()
    
#%% Load and attach temperature data (in case no ML is detected, and in case temperature comes from other source different than ERA5)

    data = utils.attach_ERA5_TEMP(data, site=loc)

#%% Calculate ZDR offset method 1
    if calib_type==1:
        min_height = min_hgt+data["altitude"].values

        # get PHIDP name
        for X_PHI in phidp_names:
            if X_PHI in data.data_vars:
                break
        # get DBZH name
        for X_DBZH in dbzh_names:
            if X_DBZH in data.data_vars:
                break
        
        # get RHOHV name
        for X_RHO in rhohv_names:
            if X_RHO in data.data_vars:
                break

        # get ZDR name
        for X_ZDR in zdr_names:
            if X_ZDR in data.data_vars:
                break

        # Check that all variables are present
        check_vars = [xvar not in data.data_vars for xvar in [X_PHI, X_DBZH, X_RHO, X_ZDR]]
        if any(check_vars):
            print("Not all necessary variables found in the data.")
            sys.exit("Not all necessary variables found in the data.")

        ### First we need to correct PHIDP and load corrected RHOHV
        
        # Calculate PHIDP offset
        phidp_offset = utils.phidp_offset_detection(data, phidp=X_PHI)
        off = phidp_offset["PHIDP_OFFSET"]
        start_range = phidp_offset["start_range"]
    
        # apply offset
        fix_range = 750
        phi_fix = data[X_PHI].copy()
        off_fix = off.broadcast_like(phi_fix)
        phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off
        
        data = data.assign({X_PHI+"_OC": phi_fix.assign_attrs(data[X_PHI].attrs)})
        
        # Load corrected RHOHV
        !!!!!!!!!!!!!!!!!!!!!
        
        # Calculate ML from the VP
        moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI+"_OC": (-20, 360)}
        ml = utils.melting_layer_qvp_X_new(data.median("azimuth", keep_attrs=True)\
                                           .assign_coords({"z":data["z"].median("azimuth", keep_attrs=True)})\
                                           .swap_dims({"range":"z"}), min_h=min_height,
                                           dim="z", moments=moments, all_data=True)
        
        #### Giagrande refinment
        hdim = "z"
        # get data iside the currently detected ML
        cut_above = ml.where(ml["z"]<ml.mlh_top)
        cut_above = cut_above.where(ml["z"]>ml.mlh_bottom)
        
        # get the heights with min RHOHV
        min_height_ML = cut_above[X_RHO].idxmin(dim="z") 
        
        # cut the data below and above the previous value
        new_cut_below_min_ML = ml.where(ml["z"] > min_height_ML)
        new_cut_above_min_ML = ml.where(ml["z"] < min_height_ML)
        
        # Filter out values outside some RHOHV range
        new_cut_below_min_ML_filter = new_cut_below_min_ML[X_RHO].where((new_cut_below_min_ML[X_RHO]>=0.97)&(new_cut_below_min_ML[X_RHO]<=1))
        new_cut_above_min_ML_filter = new_cut_above_min_ML[X_RHO].where((new_cut_above_min_ML[X_RHO]>=0.97)&(new_cut_above_min_ML[X_RHO]<=1))            
        
        
        ######### ML TOP Giangrande refinement
        
        notnull = new_cut_below_min_ML_filter.notnull() # this replaces nan for False and the rest for True
        first_valid_height_after_ml = notnull.where(notnull).idxmax(dim=hdim) # get the first True value, i.e. first valid value
        
        ######### ML BOTTOM Giangrande refinement
        # For this one, we need to flip the coordinate so that it is actually selecting the last valid index
        notnull = new_cut_above_min_ML_filter.notnull() # this replaces nan for False and the rest for True
        last_valid_height = notnull.where(notnull).isel({hdim:slice(None, None, -1)}).idxmax(dim=hdim) # get the first True value, i.e. first valid value (flipped)
        
        
        ml = ml.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
        ml = ml.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
        
        !!!!!!! ADD VARIABLES NAMES HERE
        zdr_offset = utils.zdr_offset_detection_vps(vert, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, mlbottom=5).compute()
        
