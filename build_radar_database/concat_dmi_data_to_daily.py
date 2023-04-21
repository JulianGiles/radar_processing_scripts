#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:43:05 2023

@author: jgiles
"""

# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import os
import datetime as dt
import pandas as pd
from tqdm.notebook import trange, tqdm

import warnings
warnings.filterwarnings('ignore')
import xradar as xd
import datatree as dttree

import netCDF4
import packaging

import time
start_time = time.time()

#%% Get encoding from a DWD file
dwd = xr.open_dataset("/automount/ags/jgiles/turkey_test/ras07-vol5minng01_sweeph5onem_allmoms_00-2017072700005800-pro-10392-hd5", group="sweep_0")
# display(dwd)

drop = ["szip", "zstd", "source", "chunksizes", "bzip2", "blosc", "shuffle", "fletcher32", "original_shape", "coordinates", "contiguous"]
dwd_enc = {k: {key: v.encoding[key] for key in v.encoding if key not in drop} for k, v in dwd.data_vars.items() if v.ndim == 3}
dwd_enc["PHIDP"] = dwd_enc["UPHIDP"]
dwd_enc["DBTH"] = dwd_enc["TH"]
dwd_enc["DBTV"] = dwd_enc["TV"]

#%% Import and set Dask stuff

# import dask
# from dask.distributed import Client
# # not sure if this is needed
# # client = Client(n_workers=8)
# # client
# from dask.diagnostics import ProgressBar

#%% Get files

# Get all files for one day
htypath = sorted(glob.glob("/automount/ags/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/*"))

# Create a dataframe to store the metadata of all files and then select it more easily

# Read attributes of files
radarid = []
dtime = []
taskname = []
elevation = []
nrays_expected = []
nrays_written = []
nbins = []
rlastbin = []
binlength = []
horbeamwidth = []
fpath = []

for f in htypath:
    print(".", end="")
    # Read metadata
    m = xd.io.backends.iris.IrisRawFile(f, loaddata=False)
    # Extract info
    fname = os.path.basename(f).split(".")[0]
    radarid_ = fname[0:3]
    dtimestr = fname[3:]
    dtime_ = dt.datetime.strptime(dtimestr, "%y%m%d%H%M%S")
    taskname_ = m.product_hdr["product_configuration"]["task_name"].strip()
    nbins_ = m.nbins
    rlastbin_ = m.ingest_header["task_configuration"]["task_range_info"]["range_last_bin"]/100
    binlength_ = m.ingest_header["task_configuration"]["task_range_info"]["step_output_bins"]/100
    horbeamwidth_ = round(m.ingest_header["task_configuration"]["task_misc_info"]["horizontal_beam_width"], 2)
    for i in range(10):
        try:
            nrays_expected_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["number_rays_file_expected"]
            nrays_written_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["number_rays_file_written"]    
            elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["fixed_angle"], 2)
            break
        except KeyError:
            try:
                nrays_expected_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_expected"]
                nrays_written_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_written"]    
                elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["fixed_angle"], 2)
                break
            except KeyError:
                continue
    # Append to list
    radarid.append(radarid_)
    dtime.append(dtime_)
    taskname.append(taskname_)
    elevation.append(elevation_)
    nbins.append(nbins_)
    rlastbin.append(rlastbin_)
    binlength.append(binlength_)
    #nrays_expected.append(nrays_expected_)
    #nrays_written.append(nrays_written_)
    fpath.append(f)
    horbeamwidth.append(horbeamwidth_)   

# put attributes in a dataframe
from collections import OrderedDict
df = pd.DataFrame(OrderedDict(
                  {"radarid": radarid,
                   "datetime": dtime,
                   "taskname": taskname,
                   "elevation": elevation,
                   #"nrays_expected": nrays_expected,
                   #"nrays_written": nrays_written,
                   "nbins": nbins,
                   "rlastbin": rlastbin,
                   "binlength": binlength,
                   "horbeamwidth": horbeamwidth,
                   "fpath": fpath                   
                  }))


# Let's open one scanning mode and one elevation (this will take some minutes to load)
mode = 'VOL_A'
elev = 0.

# Use the dataframe to get the paths that correspond to our selection
paths = df["fpath"].loc[df["elevation"]==elev].loc[df["taskname"]==mode]

paths = sorted(list(paths))
# print(len(paths))

# Set Engine
# engine = "netcdf4"
engine = "h5netcdf"

#%% Reading functions

# # original
# def read_single(f):
#     reindex = dict(start_angle=-0.5, stop_angle=360, angle_res=1., direction=1)
#     ds = xr.open_dataset(f, engine="iris", group="sweep_0", reindex_angle=reindex)
#     ds = ds.set_coords("sweep_mode")
#     ds = ds.rename_vars(time="rtime")
#     ds = ds.assign_coords(time=ds.rtime.min())
#     return ds

# @dask.delayed
# def process_single(f, num, dest):
#     ds = read_single(f)
#     moments = [k for k,v in ds.variables.items() if v.ndim == 2]
#     new_enc = {k: dwd_enc[k] for k in moments if k in dwd_enc}
#     shape = ds[moments[0]].shape
#     enc_new = dict(chunksizes=(1, ) + shape[1:])
#     [new_enc[k].update(enc_new) for k in new_enc]
#     dest = f"{dest}{num:03d}.nc"
#     ds.to_netcdf(dest, engine=engine, encoding=new_enc)
#     return dest

# revamped functions
def read_single(f):
    reindex = dict(start_angle=-0.5, stop_angle=360, angle_res=1., direction=1)
    ds = xr.open_dataset(f, engine="iris", group="sweep_0", reindex_angle=reindex) # not sure if sweep_0 is the name for all cases
    ds = ds.set_coords("sweep_mode")
    ds = ds.rename_vars(time="rtime")
    ds = ds.assign_coords(time=ds.rtime.min())
    ds["time"].encoding = ds["rtime"].encoding # copy also the encoding
    # fix time dtype to prevent uint16 overflow
    ds["time"].encoding["dtype"] = np.int64
    ds["rtime"].encoding["dtype"] = np.int64
    return ds

# @dask.delayed # We ditch dask to use multiprocessing below
def process_single(f, num, dest, scheme="unpacked", sdict={}):
    print(".", end="")
    ds = read_single(f)
    moments = [k for k,v in ds.variables.items() if v.ndim == 2]
    if "unpacked" in scheme:
        valid = ["dtype", "_FillValue"]
        new_enc = {k: {key: val for key, val in ds[k].encoding.items() if key in valid} for k in moments}
    else: 
        new_enc = {k: dwd_enc[k] for k in moments if k in dwd_enc}
    
    shape = ds[moments[0]].shape
    #print(shape)
    enc_new = dict(chunksizes=shape)
    enc_new.update(sdict) 
    [new_enc[k].update(enc_new) for k in new_enc]
    
    # set _FillValue according IRIS
    for mom in moments:
        if mom in ["DB_HCLASS2"]:
            continue
        # we can be smart an set "0" unconditionally 
        # as this is what IRIS No-Data value is
        new_enc[mom]["_FillValue"] = new_enc[mom]["dtype"].type(0)
        minval = new_enc[mom]["dtype"].type(0) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
        maxval = new_enc[mom]["dtype"].type(65535) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
        if mom == "PHIDP":
            # special handling for phase
            ds[mom] = ds[mom].where(ds[mom] > np.nanmin(ds[mom])).where(ds[mom] <= (maxval+180))
            ds[mom] = ds[mom].where(ds[mom] <= 180, ds[mom] - 360)
        else:
            ds[mom] = ds[mom].where(ds[mom] > minval).where(ds[mom] <= maxval)
        
    dest = f"{dest}{num:03d}.nc"
    ds.to_netcdf(dest, engine=engine, encoding=new_enc)
    return dest

#%%time  convert files in subfolder

dest = "/home/jgiles/turkey_test/test6_"
results = []
# # fill dask compute pipeline
# for i, f in tqdm(enumerate(paths)):
#     # results.append(client.compute(process_single(f, i, dest))) # some workers fail for some reason
#     results.append(process_single(f, i, dest).compute()) # this way takes about 5 min to process all paths
# # compute pipeline
# # this returns, if all results are computed
# for res in results:
#     print(res.result())    
    
# other way, much faster (about 1 min)
from multiprocessing import Pool
from functools import partial

process_single_partial = partial(process_single, dest=dest, scheme="packed")
with Pool() as P:
    results = P.starmap( process_single_partial, [(f, i) for i, f in enumerate(paths)] )
    

#%%time Reload converted files
dsr = xr.open_mfdataset(f"{dest}*", concat_dim="time", combine="nested", engine=engine)
# display(dsr)

#%% Fix encoding before writing to single file

moments = [k for k,v in dsr.variables.items() if v.ndim == 3]
shape = dsr[moments[0]].shape
enc_new= dict(chunksizes=(1, ) + shape[1:])

drop = ['szip', 'zstd', 'bzip2', 'blosc', 'coordinates']
enc = {k: {key: v.encoding[key] for key in v.encoding if key not in drop} for k, v in dsr.data_vars.items() if k in moments}
[enc[k].update(enc_new) for k in moments]
encoding = {k: enc[k] for k in moments}
# print(encoding)


#%% Write to single daily file

dsr.to_netcdf(f"{dest}-iris-test-compressed-{engine}.nc", engine=engine, encoding=encoding)



#%% TEST Load the daily file

dsrunpckd = xr.open_dataset(f"/home/jgiles/turkey_test/test4_-iris-test-compressed-{engine}.nc")
dsrpckd = xr.open_dataset(f"/home/jgiles/turkey_test/test5_-iris-test-compressed-{engine}.nc")


# test plots
dsrunpckd.rtime[0].plot(label="unpckd")
dsrpckd.rtime[0].plot(label="pckd")

np.testing.assert_allclose(dsrunpckd.DBZH.values, dsrpckd.DBZH.values)


import matplotlib.pyplot as plt
vv = "DBTH"
tt = 0
dsrunpckd[vv][tt, 250, 0:100].plot(label="unpacked")
dsrpckd[vv][tt, 250, 0:100].plot(label="packed", ls="--")
plt.legend()
plt.suptitle(vv)

