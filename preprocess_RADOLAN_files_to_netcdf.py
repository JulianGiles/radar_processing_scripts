#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:37:15 2024

@author: jgiles

This script takes RADOLAN files and concatenates them into monthly files in netCDF format.
"""

import xarray as xr
import wradlib as wrl
import warnings
import sys
import io
from contextlib import redirect_stdout
import os

#%% Load RADOLAN files
path0 = sys.argv[1] # read monthly path from console
path1 = sys.argv[2] # save monthly file here

print("Loading "+path0)
f = io.StringIO() # a variable to catch the output of the laoding function
with warnings.catch_warnings():
    with redirect_stdout(f):
        warnings.simplefilter('ignore')
        data = wrl.io.open_radolan_mfdataset(path0+"/raa01-rw_10000-*-dwd---bin*")

#%% Concat to monthly files

print("Saving to "+path1)

data["time"].encoding["zlib"] = True
data["time"].encoding["complevel"] = 6
data["x"].encoding["zlib"] = True
data["x"].encoding["complevel"] = 6
data["y"].encoding["zlib"] = True
data["y"].encoding["complevel"] = 6
data["RW"].encoding["zlib"] = True
data["RW"].encoding["complevel"] = 6
data["RW"].encoding["_FillValue"] = 65535

if not os.path.exists(path1):
    os.makedirs(path1)

data.to_netcdf(path1+"/raa01-rw_10000-"+path0.split("/")[-1][2:]+"-dwd---.nc")