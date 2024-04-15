#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python script
# Script to open the h5 EURADCLIM files and concatenate them into larger files, saving them to netCDF
# Adapted from: https://github.com/overeem11/EURADCLIM-tools/blob/main/AccumulateRadarHDF5ODIMListCount.py

import sys
import os
import numpy as np
from pathlib import Path
import warnings
import h5py
import natsort
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
from multiprocessing import Pool
from functools import partial
import glob
import tqdm

warnings.filterwarnings("ignore")

import time
start_time = time.time() # we will time how much it takes

import cdo

# Parameters from command line:
OutputFileName = sys.argv[1]
InputFileNames = sys.argv[2]
PathOrFileNames = sys.argv[3]

# testing
# OutputFileName = "/automount/agradar/jgiles/EURADCLIM/concat_files/RAD_OPERA_HOURLY_RAINFALL_201601.nc"
# InputFileNames = "/automount/agradar/jgiles/EURADCLIM/2016/01/"
# PathOrFileNames = "path"


if PathOrFileNames == "files":
    pathlist_temp = InputFileNames.split()
    pathlist = natsort.natsorted(pathlist_temp)

if PathOrFileNames == "path":
    pathlist_temp = Path(InputFileNames).glob('**/*.h5')
    pathlist = natsort.natsorted(pathlist_temp)

if PathOrFileNames not in ["files", "path"]:
    print("Accumulation cannot be performed. Please specify whether a list of files (files) or a directory path with files (path) is supplied!")
    sys.exit(0)

DATAFIELD_NAME = '/dataset1/data1/data'
datetime_format = "%Y%m%d%H%M%S"

with open("/automount/agradar/jgiles/EURADCLIM/CoordinatesHDF5ODIMWGS84.dat", "r") as coord_file:
    coordinates = np.loadtxt(coord_file)
    lons = coordinates[:,0]
    lats = coordinates[:,1]


def convert_to_netcdf(filepath, num, dest):
    if h5py.is_hdf5(filepath):
        f = h5py.File(filepath, "r")
        nodata = f['/dataset1/what'].attrs['nodata']
        undetect = f['/dataset1/what'].attrs['undetect']
        radardata = f[DATAFIELD_NAME][()]
        truth_table = radardata == undetect
        indices = np.where(truth_table)
        radardata[indices] = 0
        radardata[np.isnan(radardata)] = nodata
        truth_table = radardata == nodata
        indices = np.where(truth_table)
        radardata[indices] = 0
        radardata = np.expand_dims(radardata, -1)
        enddate = f['/dataset1/what'].attrs['enddate'] # we put end datetime as the timestep value
        endtime = f['/dataset1/what'].attrs['endtime']
        DateTime =[pd.to_datetime( enddate.astype(str) + endtime.astype(str) , format=datetime_format)]
        Count = f[DATAFIELD_NAME][()]
        Count[np.isnan(Count)] = nodata
        truth_table = Count != nodata
        indices = np.where(truth_table)
        Count[indices] = 1
        truth_table = Count == nodata
        indices = np.where(truth_table)
        Count[indices] = 0
        Count = np.expand_dims(Count, -1)
        f.close()
        
        ds = xr.Dataset(
        
            data_vars=dict(
        
                Precip=(["x", "y", "time"], radardata),
        
                Count=(["x", "y", "time"], Count),

                lon=(["x", "y"], lons.reshape(radardata[:,:,0].shape)),
        
                lat=(["x", "y"], lats.reshape(radardata[:,:,0].shape)),
        
            ),
        
            coords=dict(
        
                time=np.array(DateTime),
            
            ),
        
            attrs=dict(description="EURADCLIM precipitation estimates."),
        
        )
        
        ds["Precip"].attrs = {
            "long_name": "Accumulated Radar Rainfall",
            "units": "mm"
            }
    
        ds["Count"].attrs = {
            "long_name": "Number of Images with Data",
            }
        
        ds["Precip"].encoding = {
                "zlib": True,
                "complevel": 6
                }
        
        ds["Count"].encoding = {
                "zlib": True,
                "complevel": 6
                }
    
        ds["lon"].encoding = {
                "zlib": True,
                "complevel": 6
                }
    
        ds["lat"].encoding = {
                "zlib": True,
                "complevel": 6
                }
        
        # now put time as first dim
        ds = ds.transpose("time", "x", "y")
        
        dest = f"{dest}/part_{num:03d}.nc"
        ds.to_netcdf(dest)

#%%time  convert files in subfolder

print("Processing single files "+InputFileNames)

# delete all partial files in the folder, if any
to_remove = glob.glob(str(Path(OutputFileName).parent)+"/part_*")
# delete each file in the list
for file_path in to_remove:
    os.remove(file_path)

destfolder = str(Path(OutputFileName).parent)
results = []

convert_to_netcdf_partial = partial(convert_to_netcdf, dest=destfolder)
inputs = [(f, i) for i, f in enumerate(pathlist)]
with Pool() as P:
    results = P.starmap( convert_to_netcdf_partial, tqdm.tqdm(inputs, total=len(inputs)) )

#%% Concat to single file with CDO
print("Writing concatenated file")

cdo.Cdo().cat(input=destfolder+"/part*", output=OutputFileName, options="-z zip_6")

# delete all partial files in the folder, if any
to_remove = glob.glob(str(Path(OutputFileName).parent)+"/part_*")
# delete each file in the list
for file_path in to_remove:
    os.remove(file_path)


#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")

#%%
# fig = plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Plot the precipitation data

# ds["Precip"][:,:,0].plot(ax=ax, x="lon", y="lat", cmap="Blues", vmin=0, vmax=10) #, transform=ccrs.PlateCarree()) # transform for Mercator

# # plt.imshow(precip_data, extent=(lon.min(), lon.max(), lat.min(), lat.max()),
# #            origin='lower', cmap='Blues', alpha=0.7)

# # Add coastlines, gridlines, and title
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.gridlines(draw_labels=True)
# plt.title('Precipitation on {}'.format(ds['time'].values))

# # Show plot
# # plt.colorbar(label='Precipitation (mm)')
# # plt.show()