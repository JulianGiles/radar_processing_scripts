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
# import datatree as dttree

import netCDF4
import packaging

import time
start_time = time.time()


from multiprocessing import Pool
from functools import partial


#%% Set paths
htypath = sorted(glob.glob(sys.argv[1]+"/*.RAW*"))
dest = sys.argv[2]

# For testing
# htypath = sorted(glob.glob("/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/05/08/ANK/RAW/*"))
# htypath = sorted(glob.glob("/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/*"))
# htypath = sorted(glob.glob("/home/jgiles/turkey_test/AFY_20190502/*"))
# dest = "/home/jgiles/turkey_test/temp/"

# create dest if it does not exist
if not os.path.exists(dest):

    # if the demo_folder directory is not present
    # then create it.
    os.makedirs(dest)

#%% Get encoding from a DWD file
dwd = xr.open_dataset("/automount/ags/jgiles/turkey_test/ras07-vol5minng01_sweeph5onem_allmoms_00-2017072700005800-pro-10392-hd5", group="sweep_0")
# display(dwd)

# update 05/12/23: there seems to be an issue with preferred_chunks in the new xarray version, so we just drop it
# update 06/12/23: the issue is apparently fixed in the latest xarray version
drop = ["preferred_chunks", "szip", "zstd", "source", "chunksizes", "bzip2", "blosc", "shuffle", "fletcher32", "original_shape", "coordinates", "contiguous"]
dwd_enc = {k: {key: v.encoding[key] for key in v.encoding if key not in drop} for k, v in dwd.data_vars.items() if v.ndim == 3}

# manually add missing encodings
dwd_enc["PHIDP"] = dwd_enc["UPHIDP"].copy()
dwd_enc["DBTH"] = dwd_enc["TH"].copy()
dwd_enc["DBTV"] = dwd_enc["TV"].copy()
dwd_enc["DB_DBZC"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_ZDRC"] = dwd_enc["ZDR"].copy()
dwd_enc["DB_DBZC2"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_ZDRC2"] = dwd_enc["ZDR"].copy()
dwd_enc["DB_DBTE16"] = dwd_enc["DBTH"].copy()
dwd_enc["DB_DBZE16"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_DBTE8"] = dwd_enc["DBTH"].copy()
dwd_enc["DB_DBZE8"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_DBTV16"] = dwd_enc["DBTV"].copy()
dwd_enc["DB_DBZV16"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_DBTV8"] = dwd_enc["DBTV"].copy()
dwd_enc["DB_DBZV8"] = dwd_enc["DBZH"].copy()

dwd_enc["DB_SNR8"] = dwd_enc["DBZH"].copy()
dwd_enc["DB_SNR8"]["_FillValue"] = 255
dwd_enc["DB_SNR8"]["scale_factor"] = 0.5
dwd_enc["DB_SNR8"]["add_offset"] = -31.5





#%% Get files

# Get all files for one day
# htypath = sorted(glob.glob("/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/05/08/ANK/RAW/*"))
# htypath = sorted(glob.glob("/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/*"))

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
sweep_number = []


# TEST: checking the moment names to see if they are 1- or 2-byte
# taskname={}
# for f in htypath:
#     try:
#         m = xd.io.backends.iris.IrisRawFile(f, loaddata=False)
#     except ValueError:
#         # some files may be empty, ignore them
#         print("ignoring empty file: "+f)
#         continue
#     except EOFError:
#         # some files may be corrupt, ignore them
#         print("ignoring corrupt file: "+f)
#         continue
#     except OSError:
#         # some files give NULL value error, ignore them
#         print("ignoring NULL file: "+f)
#         continue
#     taskname_ = m.product_hdr["product_configuration"]["task_name"].strip()
#     for i in range(10):
#         try:
#             elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["fixed_angle"], 2)
#             mom_keys = m.data[i]["ingest_data_hdrs"].keys()
#             break
#         except KeyError:
#             try:
#                 elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["fixed_angle"], 2)
#                 mom_keys = m.data[i]["ingest_data_hdrs"].keys()
#                 break
#             except KeyError:
#                 continue
#     if taskname_ not in taskname.keys():
#         taskname[taskname_] = {"elevs":[], "moms":[]}
#     if elevation_ not in taskname[taskname_]["elevs"]:
#         taskname[taskname_]["elevs"].append(elevation_)
#     if mom_keys not in taskname[taskname_]["moms"]:
#         taskname[taskname_]["moms"].append(mom_keys)


for f in htypath:
    # print(".", end="")
    # Read metadata
    try:
        m = xd.io.backends.iris.IrisRawFile(f, loaddata=False)
    except ValueError:
        # some files may be empty, ignore them
        print("ignoring empty file: "+f)
        continue
    except EOFError:
        # some files may be corrupt, ignore them
        print("ignoring corrupt file: "+f)
        continue
    except OSError:
        # some files give NULL value error, ignore them
        print("ignoring NULL file: "+f)
        continue
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
            sweep_number_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["sweep_number"]
            break
        except KeyError:
            try:
                nrays_expected_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_expected"]
                nrays_written_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_written"]
                elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["fixed_angle"], 2)
                sweep_number_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["sweep_number"]
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
    sweep_number.append(sweep_number_)

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
                   "fpath": fpath,
                    "sweep_number": sweep_number
                  }))


# extract all possible elevations and tasknames
allelevs = df["elevation"].unique()
alltasknames = df["taskname"].unique()

# Set Engine
# engine = "netcdf4"
engine = "h5netcdf"

#%% Here starts the processing for each elev/taskname combo
for elev in allelevs:
    for mode in alltasknames:

        # Use the dataframe to get the paths that correspond to our selection
        paths = df["fpath"].loc[df["elevation"]==elev].loc[df["taskname"]==mode]

        paths = sorted(list(paths))
        # print(len(paths))

        if len(paths) == 0:
            continue

        print("processing "+str(elev)+" "+mode)

        #%% Reading functions

        # get sweep number
        sweepnr = str( df["sweep_number"].loc[df["elevation"]==elev].loc[df["taskname"]==mode].unique()[0]-1 )

        # extract the angle information for the first of the files, so we reindex accordingly all the files
        dsini = xr.open_dataset(paths[0], engine=xd.io.iris.IrisBackendEntrypoint, group="sweep_"+sweepnr, reindex_angle=False, mask_and_scale=False) # if this fails with KeyError try changing: engine=xd.io.iris.IrisBackendEntrypoint
        try:
            angle_params = xd.util.extract_angle_parameters(dsini)
            reindex = {k: v for k,v in angle_params.items() if k in ["start_angle", "stop_angle", "angle_res", "direction"]}
        except:
            # if angles could not be extracted is because something is wrong with the data, then ignore
            print("ignoring because of incorrect dims: "+str(dsini.dims))
            continue


        if "RHI" in mode:
            # make different functions for RHIs
            def read_single(f):
                # reindex = dict(start_angle=-0.5, stop_angle=360, angle_res=1., direction=1) # we moved this outside

                ds = xr.open_dataset(f, engine=xd.io.iris.IrisBackendEntrypoint, group="sweep_"+sweepnr)
                # if we put reindex inside the open_dataset I get an error because azimuth is not a dim, thus do it manually
                ds = xd.util.reindex_angle(ds, start_angle=reindex["start_angle"], stop_angle=reindex["stop_angle"],
                                           angle_res=reindex["angle_res"], direction=reindex["direction"],
                                           method="nearest", tolerance=None)

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
                try:
                    # print(".", end="")
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

                    if "unpacked" not in scheme:
                        # set _FillValue according IRIS
                        for mom in moments:
                            if mom in ["DB_HCLASS2", "DB_HCLASS"]:
                                continue

                            # here is our only assumption: at least one IRIS "zero" value is in the data
                            iris_minval = np.nanmin(ds[mom])

                            try:
                                if mom in ["DB_SNR8", "DB_SNR16"]:
                                    # these moments need special treatment

                                    # this is normally already set, but anyway, use DWD fillvalue
                                    # but: 255 is reserved in the IRIS software for areas not scanned
                                    new_enc[mom]["_FillValue"] = new_enc[mom]["dtype"].type(255)

                                    # DWD minval/maxval in Iris-space
                                    minval = new_enc[mom]["dtype"].type(0) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                                    maxval = new_enc[mom]["dtype"].type(254) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]

                                else:
                                    # this is normally already set, but anyway, use DWD fillvalue
                                    # but: 65535 is reserved in the IRIS software for areas not scanned
                                    new_enc[mom]["_FillValue"] = new_enc[mom]["dtype"].type(65535)

                                    # DWD minval/maxval in Iris-space
                                    # zero is OK for all cases
                                    # 65534 is safe for most cases
                                    minval = new_enc[mom]["dtype"].type(0) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                                    maxval = new_enc[mom]["dtype"].type(65534) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                            except KeyError:
                                print("!!! No encoding for "+mom+", skipping moment !!!")
                                continue


                            # check that minval >= iris_minval
                            if minval < iris_minval:
                                print("! WARNING: there are "+mom+" values below the IRIS minimum encoded value !")

                            # set IRIS NoData to NaN
                            ds[mom] = ds[mom].where(ds[mom] > iris_minval)

                            # special treatment of PHIDP
                            if mom == "PHIDP":
                                # [0, 360] -> [-180, 180]
                                ds[mom] -= 180

                            # clip values to DWD, set out-of-bound values to minval/maxval
                            ds[mom] = ds[mom].where((ds[mom] > minval) | np.isnan(ds[mom]), minval)
                            ds[mom] = ds[mom].where((ds[mom] < maxval) | np.isnan(ds[mom]), maxval)


                    dest = f"{dest}part_{num:03d}.nc"
                    ds.to_netcdf(dest, engine=engine, encoding=new_enc)
                    return dest
                except:
                    print("something went wrong, ignoring file "+f)


        else: # if not RHI

            # the reindex is not working correctly due to the high noise in azimuth values giving erroneous angle_res
            # and due to files having differently aligned angles ([0,..,359] or [1,...,360])
            # we fix this manually then in the read_single function
            # possible azimuth dims:
            az0 = np.arange(0,360,1)
            az05 = np.arange(0.5,360,1)
            az1 = np.arange(1,361,1)
            possazims = np.array([az0, az05, az1])


            # revamped functions
            def read_single(f):
                # reindex = dict(start_angle=-0.5, stop_angle=360, angle_res=1., direction=1) # we moved this outside

                # ds = xr.open_dataset(f, engine=xd.io.iris.IrisBackendEntrypoint, group="sweep_"+sweepnr, reindex_angle=reindex) # simple method if we did not had the issue

                # we open the file without reindex_angle
                ds = xr.open_dataset(f, engine=xd.io.iris.IrisBackendEntrypoint, group="sweep_"+sweepnr)
                azattrs = ds.coords["azimuth"].attrs.copy() # copy the attrs otherwise they may be lost later

                try:
                    # we get the differences to each of the possible azimuth arrays defined above and we choose the one with the
                    # smaller total absolute error
                    tae0 = np.nansum( np.abs(ds["azimuth"].data - az0) )
                    tae05 = np.nansum( np.abs(ds["azimuth"].data - az05) )
                    tae1 = np.nansum( np.abs(ds["azimuth"].data - az1) )
                    tae = np.array([tae0, tae05, tae1])

                    # change the coord
                    ds.coords["azimuth"] = possazims[tae.argmin()]
                    ds.coords["azimuth"].attrs = azattrs

                except ValueError: # if the previous fail due to extra rays or whatever
                    # then just use reindex_angle but with some tweaks
                    angle_params = xd.util.extract_angle_parameters(ds)
                    uniquecounts = np.unique( np.abs((ds.azimuth - np.trunc(ds.azimuth) )).round(1) , return_counts=True) # counts of decimals positions rounded to 1 decimal
                    if 0.25 < uniquecounts[0][ uniquecounts[1].argmax() ] < 0.75:
                        # if the most common decimals are .5 then align to .5 array
                        ds = xd.util._reindex_angle(ds, az05, 0.5)
                    else:
                        # if not, we align to .0
                        if angle_params["min_angle"].round() == 1. and angle_params["max_angle"].round() == 360. :
                            ds = xd.util._reindex_angle(ds, az1, 0.5)
                        else:
                            ds = xd.util._reindex_angle(ds, az0, 0.5)


                # in case the most suitable azimuth coord is [1,...,360] we need to align it to be concatenable to [0,...,359]
                if 360 in ds["azimuth"]:
                    ds = ds.roll({"azimuth":1}, roll_coords=True)
                    ds.coords["azimuth"] = az0
                    ds.coords["azimuth"].attrs = azattrs


                ds = ds.set_coords("sweep_mode")
                ds = ds.rename_vars(time="rtime")
                ds = ds.assign_coords(time=ds.rtime.min())
                ds["time"].encoding = ds["rtime"].encoding # copy also the encoding
                # fix time dtype to prevent uint16 overflow
                ds["time"].encoding["dtype"] = np.int64
                ds["rtime"].encoding["dtype"] = np.int64

                # Fixes
                # It may happen that some time value is missing, fix that using info in rtime
                if ds["time"].isnull().any():
                    ds.coords["time"] = ds.rtime.min(dim="azimuth", skipna=True).compute()

                # if some coord has dimension time, reduce using median
                for coord in ["latitude", "longitude", "altitude", "elevation"]:
                    if "time" in ds[coord].dims:
                        ds.coords[coord] = ds.coords[coord].median("time")

                return ds.dropna("azimuth", how="all")

            # @dask.delayed # We ditch dask to use multiprocessing below
            def process_single(f, num, dest, scheme="unpacked", sdict={}):
                try:
                    # print(".", end="")
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

                    if "unpacked" not in scheme:
                        # set _FillValue according IRIS
                        for mom in moments:
                            if mom in ["DB_HCLASS2", "DB_HCLASS"]:
                                continue

                            # here is our only assumption: at least one IRIS "zero" value is in the data
                            iris_minval = np.nanmin(ds[mom])

                            try:
                                if mom in ["DB_SNR8", "DB_SNR16"]:
                                    # these moments need special treatment

                                    # this is normally already set, but anyway, use DWD fillvalue
                                    # but: 255 is reserved in the IRIS software for areas not scanned
                                    new_enc[mom]["_FillValue"] = new_enc[mom]["dtype"].type(255)

                                    # DWD minval/maxval in Iris-space
                                    minval = new_enc[mom]["dtype"].type(0) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                                    maxval = new_enc[mom]["dtype"].type(254) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]

                                else:
                                    # this is normally already set, but anyway, use DWD fillvalue
                                    # but: 65535 is reserved in the IRIS software for areas not scanned
                                    new_enc[mom]["_FillValue"] = new_enc[mom]["dtype"].type(65535)

                                    # DWD minval/maxval in Iris-space
                                    # zero is OK for all cases
                                    # 65534 is safe for most cases
                                    minval = new_enc[mom]["dtype"].type(0) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                                    maxval = new_enc[mom]["dtype"].type(65534) * new_enc[mom]["scale_factor"] + new_enc[mom]["add_offset"]
                            except KeyError:
                                print("!!! No encoding for "+mom+", skipping moment !!!")
                                continue

                            # check that minval >= iris_minval
                            #if minval < iris_minval:
                                #print("! WARNING: there are "+mom+" values below the IRIS minimum encoded value !")

                            # set IRIS NoData to NaN
                            ds[mom] = ds[mom].where(ds[mom] > iris_minval)

                            # special treatment of PHIDP
                            if mom == "PHIDP":
                                # [0, 360] -> [-180, 180]
                                ds[mom] -= 180

                            # clip values to DWD, set out-of-bound values to minval/maxval
                            ds[mom] = ds[mom].where((ds[mom] > minval) | np.isnan(ds[mom]), minval)
                            ds[mom] = ds[mom].where((ds[mom] < maxval) | np.isnan(ds[mom]), maxval)


                    dest = f"{dest}part_{num:03d}.nc"
                    ds.to_netcdf(dest, engine=engine, encoding=new_enc)
                    return dest
                except:
                    print("something went wrong, ignoring file "+f)

        #%%time  convert files in subfolder

        # delete all partial files in the folder, if any
        to_remove = glob.glob(dest+"part_*")
        # delete each file in the list
        for file_path in to_remove:
            os.remove(file_path)

        # dest = "/home/jgiles/turkey_test/testank_"
        results = []

        process_single_partial = partial(process_single, dest=dest, scheme="packed")
        with Pool() as P:
            results = P.starmap( process_single_partial, [(f, i) for i, f in enumerate(paths)] )


        #%%time Reload converted files
        try:
            dsr = xr.open_mfdataset(f"{dest}part_*", concat_dim="time", combine="nested", engine=engine)
        except OSError:
            print("!!! no files to open, ignoring !!!")
            continue
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

        #%% check that the object has 360 azimuths
        if len(dsr.azimuth) != 360:
            print("The resulting array has "+str(len(dsr.azimuth))+" azimuth values instead of 360")

        #%% Write to single daily file
        date=htypath[0].split("/")[-4]
        loc=htypath[0].split("/")[-3]

        dsr.to_netcdf(f"{dest}{mode}-allm-{elev}-{date}-{loc}-{engine}.nc", engine=engine, encoding=encoding)


        # delete all partial files in the folder, if any
        to_remove = glob.glob(dest+"part_*")
        # delete each file in the list
        for file_path in to_remove:
            os.remove(file_path)

        # close dsr
        dsr.close()
        del(dsr)

#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")


