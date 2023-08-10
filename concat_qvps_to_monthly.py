#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:25:33 2023

@author: jgiles

Script for concatenating the daily QVPs to monthly files
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

years=["2015", "2016", "2017", "2018", "2019", "2020"]
months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

def add_nan_ml(ds):
    # function for adding dummy ML coordinates so the concatenation does not fail
    if "height_ml" not in ds.coords:
        ds.coords["height_ml"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_bottom" not in ds.coords:
        ds.coords["height_ml_bottom"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_new_gia" not in ds.coords:
        ds.coords["height_ml_new_gia"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    if "height_ml_bottom_new_gia" not in ds.coords:
        ds.coords["height_ml_bottom_new_gia"] = xr.full_like(ds.coords["time"], np.nan, dtype=np.float64)
    return ds

for loc in ["pro", "tur", "umd", "AFY", "ANK", "GZT", "HTY", "SVS"]:
    files = glob.glob(path_qvps+"/**/*allmoms*"+loc+"*", recursive=True)
    
    if len(files)>0:
        print("xxxxx Concatenating data for "+loc)
        start_time = time.time()

    else:
        continue
    
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
                    
                    
                    first_file = xr.open_mfdataset(ffs2[0]) 
                    first_file_z = first_file.z.copy()
                    def fix_z_and_time(ds):
                        ds.coords["z"] = first_file_z
                        ds = ds.where(ds["time"].notnull(), drop=True)
                            
                        return add_nan_ml(ds)

                    try:
                        qvps = xr.open_mfdataset(ffs2, combine="nested", concat_dim="time", preprocess=add_nan_ml)
                    except:
                        qvps = xr.open_mfdataset(ffs2, combine="nested", concat_dim="time", preprocess=fix_z_and_time)
                    
                    savepath1 = ffs2[0].split("/qvps/")[0]+"/qvps_monthly/"+yy+"/"+yy+"-"+mm+"/"
                    savepath2 = ffs2[0].split("/"+loc+"/")[1]
                    savepath = savepath1 +"/"+loc+"/"+ savepath2
                    savepathdir = os.path.dirname(savepath)
        
                    if not os.path.exists(savepathdir):
                        os.makedirs(savepathdir)
                    
                    qvps.to_netcdf(savepath)

    #%% print how much time did it take
    total_time = time.time() - start_time
    print("xxxxx Concatenating data for "+loc+f" took {total_time/60:.2f} minutes to run.")