#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:43:05 2023

@author: jgiles

This script takes all dwd radar files from a folder (for one elevation) and
merges them into a single file combining all moments along all timesteps. 
Then saves the resulting dataset into a new file with the same naming
style but with "allmoms" instead of the moment name. Additionally, it saves
either a true.txt or false.txt file alongside, if the data fulfills certain 
condition, as an attempt to check if there is actually something interesting
in that period of data.
"""
#### Loads packages
# NEEDS WRADLIB 2.0 !! (OR GREATER?)

import datatree as dttree
import numpy as np
import sys
import glob

import warnings
warnings.filterwarnings('ignore')

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
scan_elevs = np.array([5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 8.0, 12.0, 17.0, 25.0])

# get list of files in the folder
path = sys.argv[1]

# Load the files. This loading function already takes care of not loading a 
# concatenated daily file if it already exists (if "allmoms" is in the name)
# and also aligns coordinates and fixes some expected typical issues when doing so.
ds = utils.load_dwd_raw(path+"/ras*hd5")

# Create a datatree
dtree = dttree.DataTree(name="root")

# check and fix how the angle variable is named if necessary
if "fixed_angle" in ds:
    # rename the variable
    ds = ds.rename({"fixed_angle": "sweep_fixed_angle"}) 

# get the sweep number according to the scan strategy
try:
    ii = int(np.where(scan_elevs == round(float(ds["sweep_fixed_angle"]),1))[0])
except:
    ii = 0

# Put the data in the data tree
dttree.DataTree(ds, name=f"sweep_{ii}", parent=dtree)

# Get the file list just for the naming of the output file
ll = sorted(glob.glob(path+"/ras*hd5"))

# Save the datatree as netcdf 
name = ll[0].split("_")
name[-2]="allmoms"
dtree.load().to_netcdf("_".join(name))
  
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
    
## To load the datatree
# ds_rel = utils.load_dwd_preprocessed("_".join(name))

## to load the datatree raw
# ds_rel = dttree.open_datatree("_".join(name))

# swp = ds_rel["sweep_1"].to_dataset()  # get a sweep as xarray dataset    
