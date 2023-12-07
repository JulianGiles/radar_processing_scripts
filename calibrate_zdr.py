# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:31:29 2023

@author: Julian
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:16 2023

@author: jgiles

Script for calculating ZDR calibration from different methods.

1. “Bird bathing” with vertically pointing radar

2. Using light rain as a natural calibrator 

3. NOT IMPLEMENTED! Using dry aggregated snow as a natural calibrator with intrinsic ZDR of 0.1 – 0.2 dB

4. NOT IMPLEMENTED! Bragg scattering with intrinsic ZDR equal to 0 dB

5. NOT IMPLEMENTED! Ground clutter, if stable

"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.1 !! (OR GREATER?)

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
path0 = sys.argv[1]
calib_type = int(sys.argv[2]) # Like the numbers from the description above. Can be a multiple digit integer for simultaneous calculations
overwrite = True # overwrite existing files?

phidp_names = ["UPHIDP", "PHIDP"] # names to look for the PHIDP variable, in order of preference
dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV_NC", "RHOHV"] # same but for RHOHV
zdr_names = ["ZDR"]

clowres0=False # this is for the ML detection algorithm
min_hgts = utils.min_hgts
min_hgt = min_hgts["default"] # minimum height above the radar to be considered when calculating ZDR offset
if "dwd" in path0 and "90grads" in path0:
    # for the VP we need to set a higher min height because there are several bins of unrealistic values
    min_hgt = min_hgts["90grads"]
if "dwd" in path0 and "vol5minng01" in path0:
    clowres0=True
if "ANK" in path0:
    # for ANK we need higher min_hgt to avoid artifacts
    min_hgt = min_hgts["ANK"]
if "GZT" in path0:
    # for GZT we need higher min_hgt to avoid artifacts
    min_hgt = min_hgts["GZT"]

# get the files and check that it is not empty
if "hd5" in path0 or "h5" in path0:
    files=[path0]
elif "dwd" in path0:
    files = sorted(glob.glob(path0+"/*hd5*"))
elif "dmi" in path0:
    files = sorted(glob.glob(path0+"/*h5*"))
else:
    print("Country code not found in path")
    sys.exit("Country code not found in path.")

if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

# define a function to split the calibration type into separate integers
def split_digits(number):
    if not isinstance(number, int):
        raise ValueError("Input should be an integer.")

    digits_set = set([int(digit) for digit in str(number)])
    return digits_set

# get calib types set
calib_types = split_digits(calib_type)

# check that all calib_types are implemented
for cp in calib_types:
    if cp not in [1,2]:
        print("Calibration method not implemented. Possible options are 1 or 2")
        sys.exit("Calibration method not implemented. Possible options are 1 or 2")

# set the RHOHV correction location
rhoncdir = "/rhohv_nc/" # subfolder where to find the noise corrected rhohv data
rhoncfile = "*rhohv_nc_2percent*" # pattern to select the appropriate file (careful with the rhohv_nc_2percent)

# define a function to create save directory for the offset and return file save path
def make_savedir(ff, name):
    """
    ff: filepath of the original file
    name: name for the particular folder inside zdr/calibration/
    """
    if "dwd" in ff:
        country="dwd"
    elif "dmi" in ff:
        country="dmi"
    else:
        print("Country code not found in path")
        sys.exit("Country code not found in path.")

    ff_parts = ff.split(country)
    savepath = (country+"/calibration/zdr/"+name+"/").join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    return savepath


# ERA5 folder
if "jgiles" in files[0]:
    # then we are in local system
    era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif "giles1" in files[0]:
    # then we are in JSC
    era5_dir = "/p/scratch/detectrea/giles1/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below

#%% Load data

for ff in files:
    # check if the resulting calib file already exists before starting
    calib_1 = True # by default we compute as if overwrite is True
    calib_2 = True
    if not overwrite:        
        if 1 in calib_types:
            savepath = make_savedir(ff, "VP")
            if len(os.listdir(os.path.dirname(savepath))) != 0:
                calib_1 = False
        if 2 in calib_types:
            savepath = make_savedir(ff, "LR_consistency")
            if len(os.listdir(os.path.dirname(savepath))) != 0:
                calib_2 = False
    
    if all((not calib_1, not calib_2)):
        continue

    print("processing "+ff)
    if "dwd" in ff:
        # data=dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()
        data = utils.load_dwd_preprocessed(ff) # this already loads the first elev available in the files and fixes time coord

    elif "dmi" in ff:
        # data=xr.open_dataset(ff)
        data = utils.load_dmi_preprocessed(ff) # this loads DMI file and flips phidp and fixes time coord
    else:
        raise NotImplementedError("Only DWD or DMI data supported at the moment")
        
    # fix time dim and time in coords
    # data = utils.fix_time_in_coords(data)

    # flip UPHIDP and KDP in UMD data
    if "umd" in ff:
        print("Flipping phase moments in UMD")
        for vf in ["UPHIDP", "KDP"]: # Phase moments in UMD are flipped into the negatives
            attrs = data[vf].attrs.copy()
            data[vf] = data[vf]*-1
            data[vf].attrs = attrs.copy()

#%% Load noise corrected RHOHV if available
    try:
        if "dwd" in ff:
            country="dwd"
        elif "dmi" in ff:
            country="dmi"
        rhoncpath = os.path.dirname(utils.edit_str(ff, country, country+rhoncdir))
        data = utils.load_corrected_RHOHV(data, rhoncpath+"/"+rhoncfile)
                
    except OSError:
        print("No noise corrected rhohv to load: "+rhoncpath+"/"+rhoncfile)

#%% Load and attach temperature data (in case no ML is detected, and in case temperature comes from other source different than ERA5)
    
    # find loc and assign TEMP data accordingly    
    loc = utils.find_loc(utils.locs, ff)
    
    data = utils.attach_ERA5_TEMP(data, path=loc.join(era5_dir.split("loc")))

#%% Calculate ZDR offset method 1 or 2
    if 1 in calib_types or 2 in calib_types:
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
                if "_NC" in X_RHO:
                    # Check that the corrected RHOHV does not have much higher STD than the original (50% more)
                    # if that is the case we take it that the correction did not work well so we won't use it
                    if (data["RHOHV"].std()*1.5 < data[X_RHO].std()).compute():
                        # Change to the default RHOHV name 
                        X_RHO = "RHOHV"

                break

        # get ZDR name
        for X_ZDR in zdr_names:
            if X_ZDR in data.data_vars:
                break

        # Check that essential variables are present
        check_vars = [xvar not in data.data_vars for xvar in [X_DBZH, X_RHO, X_ZDR]]
        if any(check_vars):
            print("Not all necessary variables found in the data.")
            sys.exit("Not all necessary variables found in the data.")

        ### First we need to correct PHIDP 

        # Check that PHIDP is in data, otherwise skip ML detection
        if X_PHI in data.data_vars:
            
            # Calculate PHIDP offset
            min_periods = 4 # min radar bins for range rolling sum
            if clowres0:
                min_periods = 2
            phidp_offset = utils.phidp_offset_detection(data, phidp=X_PHI, rhohvmin=0.9, dbzhmin=0., 
                                                        min_height=0, rng=3000, center=True, min_periods=min_periods)
            off = phidp_offset["PHIDP_OFFSET"]
            start_range = phidp_offset["start_range"]
        
            # apply offset
            if "dwd" in ff:
                fix_range = 750
            else:
                fix_range = 200
            phi_fix = data[X_PHI].copy()
            off_fix = off.broadcast_like(phi_fix)
            phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off
            
            data = data.assign({X_PHI+"_OC": phi_fix.assign_attrs(data[X_PHI].attrs)})
                    
            try:
                # Calculate ML
                moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI+"_OC": (-20, 180)}
                data_qvp = utils.compute_qvp(data, min_thresh = {X_RHO:0.7, X_DBZH:0, X_ZDR:-1} )

                data_qvp = utils.melting_layer_qvp_X_new(data_qvp, min_h=min_height,
                                                   dim="z", moments=moments, clowres=clowres0)
                
                
                # attach temperature data again, then filter ML heights above -1 C
                data_qvp = utils.attach_ERA5_TEMP(data_qvp, path=loc.join(era5_dir.split("loc")))
                
                ## Discard possible erroneous ML values
                if "height_ml_new_gia" in data_qvp:
                    ## First, filter out ML heights that are too high (above selected isotherm)
                    isotherm = -1 # isotherm for the upper limit of possible ML values
                    z_isotherm = data_qvp.TEMP.isel(z=((data_qvp["TEMP"]-isotherm)**2).argmin("z").compute())["z"]
                    
                    data_qvp.coords["height_ml_new_gia"] = data_qvp["height_ml_new_gia"].where(data_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
                    data_qvp.coords["height_ml_bottom_new_gia"] = data_qvp["height_ml_bottom_new_gia"].where(data_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
                    
                    # Then, check that ML top is over ML bottom
                    cond_top_over_bottom = data_qvp.coords["height_ml_new_gia"] > data_qvp.coords["height_ml_bottom_new_gia"] 
                    
                    # Assign final values
                    data_qvp.coords["height_ml_new_gia"] = data_qvp["height_ml_new_gia"].where(cond_top_over_bottom).compute()
                    data_qvp.coords["height_ml_bottom_new_gia"] = data_qvp["height_ml_bottom_new_gia"].where(cond_top_over_bottom).compute()

                    data = data.assign_coords({'height_ml_new_gia': data_qvp.height_ml_new_gia.where(cond_top_over_bottom)})
                    data = data.assign_coords({'height_ml_bottom_new_gia': data_qvp.height_ml_bottom_new_gia.where(cond_top_over_bottom)})

            except:
                print("Calculating ML failed, skipping...")
        else:
            print(X_PHI+" not found in the data, skipping ML detection and below-ML offset")
        
        if 1 in calib_types and calib_1:
            # We ask for at least 1 km of consecutive ZDR values in each VP to be included in the calculation
            minbins = int(1000 / data["range"].diff("range").median())
            
            #### Calculate per timestep and full timespan (daily for dwd and dmi concat files)
            for timemode in ["step", "all"]:
                # the the file name appendage for saving the file
                fn_app = ["_timesteps" if timemode=="step" else "" ][0]
                
                if "height_ml_bottom_new_gia" in data:
                    # Calculate offset below ML
                    zdr_offset = utils.zdr_offset_detection_vps(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, 
                                                                min_h=min_height, timemode=timemode, minbins=minbins).compute()
            
                    # Copy encodings
                    zdr_offset["ZDR_offset"].encoding = data[X_ZDR].encoding
                    zdr_offset["ZDR_max_from_offset"].encoding = data[X_ZDR].encoding
                    zdr_offset["ZDR_min_from_offset"].encoding = data[X_ZDR].encoding
                    zdr_offset["ZDR_std_from_offset"].encoding = data[X_ZDR].encoding
                    zdr_offset["ZDR_sem_from_offset"].encoding = data[X_RHO].encoding
                    
                    # save the arrays
                    savepath = make_savedir(ff, "VP")
                    filename = ("zdr_offset_belowML"+fn_app).join(savepath.split("allmoms"))
                    zdr_offset.to_netcdf(filename)
                
                # calculate offset below 1 degree C
                zdr_offset = utils.zdr_offset_detection_vps(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=1, 
                                                            min_h=min_height, timemode=timemode, minbins=minbins).compute()
        
                # Copy encodings
                zdr_offset["ZDR_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_max_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_min_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_std_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_sem_from_offset"].encoding = data[X_RHO].encoding
                
                # save the arrays
                savepath = make_savedir(ff, "VP")
                filename = ("zdr_offset_below1C"+fn_app).join(savepath.split("allmoms"))
                zdr_offset.to_netcdf(filename)
    
                # calculate offset above 0 degrees C
                zdr_offset = utils.zdr_offset_detection_vps(data.where(data["TEMP"]<0), zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, 
                                                            mlbottom=-100, min_h=min_height, timemode=timemode, minbins=minbins).compute()
        
                # Copy encodings
                zdr_offset["ZDR_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_max_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_min_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_std_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_sem_from_offset"].encoding = data[X_RHO].encoding
                
                # save the arrays
                savepath = make_savedir(ff, "VP")
                filename = ("zdr_offset_above0C"+fn_app).join(savepath.split("allmoms"))
                zdr_offset.to_netcdf(filename)
    
    
                # calculate offset for the whole column
                zdr_offset = utils.zdr_offset_detection_vps(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, 
                                                            min_h=min_height, timemode=timemode, minbins=minbins).compute()
        
                # Copy encodings
                zdr_offset["ZDR_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_max_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_min_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_std_from_offset"].encoding = data[X_ZDR].encoding
                zdr_offset["ZDR_sem_from_offset"].encoding = data[X_RHO].encoding
                
                # save the arrays
                savepath = make_savedir(ff, "VP")
                filename = ("zdr_offset_wholecol"+fn_app).join(savepath.split("allmoms"))
                zdr_offset.to_netcdf(filename)

        if 2 in calib_types and calib_2:
            if "height_ml_bottom_new_gia" in data:
                # Calculate offset below ML per timestep
                zdr_offset = utils.zhzdr_lr_consistency(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="step")
                
                # Copy encodings
                zdr_offset.encoding = data[X_ZDR].encoding
                
                # save the arrays
                savepath = make_savedir(ff, "LR_consistency")
                filename = ("zdr_offset_belowML_timesteps").join(savepath.split("allmoms"))
                zdr_offset.to_netcdf(filename)
                
                # Calculate offset below ML for full timespan
                zdr_offset = utils.zhzdr_lr_consistency(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="all")
                
                # Copy encodings
                zdr_offset.encoding = data[X_ZDR].encoding
                
                # save the arrays
                savepath = make_savedir(ff, "LR_consistency")
                filename = ("zdr_offset_belowML").join(savepath.split("allmoms"))
                zdr_offset.to_netcdf(filename)

            # Calculate offset below 1 degree C per timestep
            zdr_offset = utils.zhzdr_lr_consistency(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=1, min_h=min_height, timemode="step")
            
            # Copy encodings
            zdr_offset.encoding = data[X_ZDR].encoding
            
            # save the arrays
            savepath = make_savedir(ff, "LR_consistency")
            filename = ("zdr_offset_below1C_timesteps").join(savepath.split("allmoms"))
            zdr_offset.to_netcdf(filename)
            
            # Calculate offset below 1 degree C for full timespan
            zdr_offset = utils.zhzdr_lr_consistency(data, zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=1, min_h=min_height, timemode="all")
            
            # Copy encodings
            zdr_offset.encoding = data[X_ZDR].encoding
            
            # save the arrays
            savepath = make_savedir(ff, "LR_consistency")
            filename = ("zdr_offset_below1C").join(savepath.split("allmoms"))
            zdr_offset.to_netcdf(filename)

#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")