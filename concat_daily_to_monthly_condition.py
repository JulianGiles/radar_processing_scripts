#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:25:33 2023

@author: jgiles

Script for concatenating the daily to monthly files selecting only files 
with ML detected or some other condition
"""

import xarray as xr
import glob
import sys
import os
import time
import numpy as np

# path_qvps = sys.argv[1] # read path from console
# or set path here
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps/"
path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps/"

# special search criteria based on auxiliary files
special_selection = "DBZH_over_30*"

years=["2015", "2016", "2017", "2018", "2019", "2020"]
months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

def fix_coords_add_nan_ml(ds):
    # fix time dim in case some value is NaT
    if ds.time.isnull().any():
        ds.coords["time"] = ds["rtime"].min(dim="azimuth", skipna=True).compute()

    # take time out of the coords if necessary
    for coord in ["latitude", "longitude", "altitude", "elevation"]:
        if "time" in ds[coord].dims:
            ds.coords[coord] = ds.coords[coord].min("time")

    # function for adding dummy ML coordinates so the concatenation does not fail
    if "height_ml" not in ds.coords:
        ds.coords["height_ml"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_bottom" not in ds.coords:
        ds.coords["height_ml_bottom"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_new_gia" not in ds.coords:
        ds.coords["height_ml_new_gia"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_bottom_new_gia" not in ds.coords:
        ds.coords["height_ml_bottom_new_gia"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    return ds.where(ds["time"].notnull(), drop=True)

for loc in ["pro", "tur", "umd", "AFY", "ANK", "GZT", "HTY", "SVS"]:
    ## Special selection of dates based on ML_detected.txt or some other criteria
    path_special = glob.glob(path_qvps+"/**/"+loc+"/**/"+special_selection, recursive=True)
    files = []
    for ff in path_special:
        files.extend(glob.glob(os.path.dirname("/".join(ff.split("/qvps/")))+"/*allmoms*"))
    
    if len(files)>0:
        print("xxxxx Concatenating data for "+loc)
        start_time = time.time()

    else:
        continue
    
    ffs = sorted(files)
    
    for yy in years:
        for mm in months:
            ffs = sorted([ff for ff in files if "/"+yy+"-"+mm+"/" in ff])
            if len(ffs)>0:
                print(loc+" "+yy+"-"+mm)
            else:
                continue
            
            # split by elevations
            modes = set([ff.split("/"+loc+"/")[-1].split("/")[0] for ff in ffs])
            elevs = set([ff.split("/"+loc+"/")[-1].split("/")[1] for ff in ffs])
            
            for mode in modes:
                for elev in elevs:
                    ffs2 = sorted([ff for ff in ffs if "/"+mode+"/"+elev+"/" in ff])
                    
                    if len(ffs2) == 0:
                        continue
                                        
                    try:
                        qvps = xr.open_mfdataset(ffs2, combine="nested", concat_dim="time", preprocess=fix_coords_add_nan_ml)
                    except:
                        print("Failed to concatenate "+" ".join([loc, yy, mm, mode, elev]))
                    
                    if "dwd" in ffs2[0]:
                        country="dwd"
                    elif "dmi" in ffs2[0]:
                        country="dmi"
                    else:
                        print("Could not determine country to build savepath. Ignoring "+" ".join([loc, yy, mm, mode, elev]))
                        continue
                    
                    savepath1 = ffs2[0].split(country)[0]+"monthly_files_"+special_selection.replace("*", "")+"/"+loc+"/"+yy+"/"+yy+"-"+mm+"/"
                    savepath2 = ffs2[0].split("/"+loc+"/")[1]
                    savepath = savepath1 + savepath2
                    savepathdir = os.path.dirname(savepath)
        
                    if not os.path.exists(savepathdir):
                        os.makedirs(savepathdir)
                    
                    qvps.to_netcdf(savepath)

    #%% print how much time did it take
    total_time = time.time() - start_time
    print("xxxxx Concatenating data for "+loc+f" took {total_time/60:.2f} minutes to run.")