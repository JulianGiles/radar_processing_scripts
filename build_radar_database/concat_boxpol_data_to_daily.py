#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:43:05 2023

@author: jgiles

This script takes all boxpol radar files from a folder (for one elevation) and
merges them into a single file combining all moments along all timesteps.
Then saves the resulting dataset into a new file with the same naming
style but with "any" instead of the moment name. Additionally, it saves
either a true.txt or false.txt file alongside, if the data fulfills certain
condition, as an attempt to check if there is actually something interesting
in that period of data.
"""
#### Loads packages
# NEEDS WRADLIB 2.0 !! (OR GREATER?)

import xarray as xr
import numpy as np
import sys
import glob
import xradar as xd

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jgiles')

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None

try:
    from Scripts.python.radar_processing_scripts import utils
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    # import colormap_generator

#### Computation

# DWD scan strategy for sorting elevations
scan_elevs = np.array([1.0, 2.0, 3.1, 4.5, 6.0, 8.2, 11.0, 14.0, 18.0, 28.0])

# get list of files in the folder
path = sys.argv[1]

# Load the files. This loading function already takes care of not loading a
# concatenated daily file if it already exists (if "allmoms" or "any" is in the name)
# and also aligns coordinates and fixes some expected typical issues when doing so.
# ds = utils.load_dwd_raw(path+"/ras*hd5")
ff_glob = sorted(glob.glob(path+"/n_ppi*.h5"))
ff = [fp for fp in ff_glob if "allm" not in fp]

def align_az(ds):
    azs = np.linspace(0.5, 359.5, 360)
    ds = xd.util.reindex_angle(ds, start_angle=0, stop_angle=360, angle_res=1, direction=1)
    ds = utils.align(ds)
    if (ds.azimuth.values-azs).mean() < 0.5:
        ds["azimuth"] = ds["azimuth"].copy(data=azs)
        return ds
    else:
        raise ValueError("not possible to aligne azimuth coord")

ds = xr.open_mfdataset(ff, engine="gamic", combine="nested", concat_dim="time", preprocess=align_az)

ds = utils.fix_flipped_phidp(utils.unfold_phidp(utils.fix_time_in_coords(ds)))

# check and fix how the angle variable is named if necessary
if "fixed_angle" in ds:
    # rename the variable
    ds = ds.rename({"fixed_angle": "sweep_fixed_angle"})

# Add the ZH offset and save ds as netcdf
name = ff[0].split("_")
name[-2] = name[-2][0:8]

date = "-".join([name[-2][0:4], name[-2][4:6], name[-2][6:]])
zhoff = xr.open_dataset("/automount/ags/s6toscha/FINAL_ZH_OFFSETS_ZH-ZDR_METHOD/NEU_TEST_QUANTILE_FILTERING_XBAND_PERFEKT/QUANTILE_FILTERING_XBAND_PERFEKT_MIT_ZDR_HUB2_MIT_BENUTZUNG_ZDR_GAP_FILLED/All_Points_Rolling_mean_30_days_gaps/ZH_ZDR_ZH_OFFSETS_method_rolling_mean_30_days_plus_points_20_80.nc")
zhoff0 = zhoff.sel(time=date)
ds.coords["ZH_day_offset_applied"] = float(zhoff0["ZH_day_offset_new"].values)

ds["DBZH"] = ds["DBZH"] - ds["ZH_day_offset_applied"]

for vv in ds.data_vars:
    ds[vv].encoding = {'zlib': True, 'complevel': 6}

ds.compute().to_netcdf("_".join(name))

#### EXTRA
try:
    # make a list of "valid" timesteps (those with dbzh > 5 in at least 1% of the bins)
    # this is only to somehow try to reduce the amount of data to check later, not very well tested
    valid = (ds["DBZH"]>5).sum(dim=("azimuth", "range")).compute() > ds["DBZH"][0].count().compute()*0.01
    valid = valid.time.where(valid, drop=True)

    # save the list as a txt file named true if there is any value, otherwise false
    if len(valid)>0:
        np.savetxt(path+"/true.txt", valid.values.astype(str), fmt="%s")
    else:
        np.savetxt(path+"/false.txt", valid.values.astype(str), fmt="%s")
except:
    pass # if it did not work just do nothing
