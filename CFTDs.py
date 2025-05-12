#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:53:58 2023

@author: jgiles

Script for calculating CFTDS
"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER)

import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import datetime as dt
import pandas as pd
import datetime
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
import xradar as xd
import time
import ridgeplot
import plotly
import pickle

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator



import warnings
warnings.filterwarnings('ignore')

# we define a funtion to look for loc inside a path string
def find_loc(locs, path):
    components = path.split(os.path.sep)
    for element in locs:
        for component in components:
            if element.lower() in component.lower():
                return element
    return None

locs = ["pro", "tur", "umd", "afy", "ank", "gzt", "hty", "svs"]

realpep_path = "/automount/realpep/"

#%% Load QVPs for stratiform-case CFTDs

calculate_retrievals = True # Calculate retrievals based on the QVPs?

# This part should be run after having the QVPs computed (compute_qvps.py)
start_time = time.time()
print("Loading QVPs...")

#### Get QVP file list
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# Load only events with ML detected (pre-condition for stratiform)
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/20*/*/*/pro/vol5minng01/07/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps/20*/*/*/HTY/*/*/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps/*/*/*/SVS/*/*/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps_singlefile/ML_detected/ANK/*/12*/*allmoms*"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*"
# path_qvps = [realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*",
#              realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/14*/*allmoms*"]


#### Set variable names
X_DBZH = "DBZH_AC"
X_RHO = "RHOHV_NC" # if RHOHV_NC is set here, it is then checked agains the original RHOHV in the next cell
X_ZDR = "ZDR_EC_OC_AC"
X_KDP = "KDP_ML_corrected_EC"
X_PHI = "UPHIDP_OC_MASKED"

if "dwd" in path_qvps:
    country="dwd"
    X_TH = "TH"
elif "dmi" in path_qvps:
    country="dmi"
    X_TH = "DBZH"


ff_glob = glob.glob(path_qvps)

if "dmi" in path_qvps:
    # create a function to only select the elevation closer to 10 for each date
    from collections import defaultdict
    def get_closest_elevation(paths):
        elevation_dict = defaultdict(list)
        for path in paths:
            parts = path.split('/')
            date = parts[-5]
            elevation = float(parts[-2])
            elevation_dict[date].append((elevation, path))

        result_paths = []
        for date, elevations in elevation_dict.items():
            closest_elevation_path = min(elevations, key=lambda x: abs(x[0] - 10))[1]
            result_paths.append(closest_elevation_path)

        return result_paths

    ff_glob = get_closest_elevation(ff_glob)

ff = [glob.glob(os.path.dirname(fp)+"/*allmoms*")[0] for fp in ff_glob ]

alignz = False
if "dwd" in path_qvps: alignz = True
qvps = utils.load_qvps(ff, align_z=alignz, fix_TEMP=False, fillna=False)

# Move TEMP to coordinate
if "TEMP" not in qvps.coords:
    qvps = qvps.set_coords("TEMP")


# # Load daily data
# # ## Special selection of convective dates based on DBZH_over_30.txt files
# special_selection = realpep_path+"/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/DBZH_over_30*"
# special_filter = realpep_path+"/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/ML_detected_*"
# path_special = glob.glob(special_selection)
# path_filter = glob.glob(special_filter)
# path_filter_dirs = [os.path.dirname(ff) for ff in path_filter]

# path_daily = []
# for ff in path_special:
#     if os.path.dirname(ff) in path_filter_dirs:
#         continue
#     path_daily.append(os.path.dirname("/".join(ff.split("/qvps/")))+"/*allmoms*")

# if isinstance(path_daily, str):
#     files_daily = sorted(glob.glob(path_daily))
# elif len(path_daily)==1:
#     files_daily = sorted(glob.glob(path_daily[0]))
# else:
#     files_daily = []
#     for fglob in path_daily:
#         files_daily.extend(glob.glob(fglob))

# def fix_dailys(data):
#     # fix time dim in case some value is NaT
#     if data.time.isnull().any():
#         data.coords["time"] = data["rtime"].min(dim="azimuth", skipna=True).compute()

#     # take time out of the coords if necessary
#     for coord in ["latitude", "longitude", "altitude", "elevation"]:
#         if "time" in data[coord].dims:
#             data.coords[coord] = data.coords[coord].min("time")

#     for X_PHI in ["PHIDP"]:
#         if X_PHI in data.data_vars:
#             # flip PHIDP in case it is wrapping around the edges (case for turkish radars)
#             if data[X_PHI].notnull().any():
#                 values_center = ((data[X_PHI]>-50)*(data[X_PHI]<50)).sum().compute()
#                 values_sides = ((data[X_PHI]>50)+(data[X_PHI]<-50)).sum().compute()
#                 if values_sides > values_center:
#                     data[X_PHI] = xr.where(data[X_PHI]<=0, data[X_PHI]+180, data[X_PHI]-180, keep_attrs=True).compute()

#     return data

# open_files=[]
# for ff in files_daily[0:20]:
#     if "dwd" in ff:
#         # basepath=ff.split("dwd")
#         open_files.append(fix_dailys(xr.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()))
#     else:
#         open_files.append(fix_dailys(xr.open_dataset(ff)))

# data = xr.concat(open_files, dim="time")
total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%% Filters (conditions for stratiform)
start_time = time.time()
print("Filtering stratiform conditions...")

min_entropy_thresh = 0.85
# Filter only stratiform events (min entropy >= min_entropy_thresh) and ML detected
# with ProgressBar():
#     qvps_strat = qvps.where( (qvps["min_entropy"]>=min_entropy_thresh) & (qvps.height_ml_bottom_new_gia.notnull()), drop=True).compute()

# Check that RHOHV_NC is actually better (less std) than RHOHV, otherwise just use RHOHV, on a per-day basis
std_tolerance = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_tolerance), otherwise use RHOHV
min_rho = 0.7 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results
mean_tolerance = 0.02 # 2% tolerance, for checking if RHOHV_NC is actually larger than RHOHV (overall higher values)

if "_NC" in X_RHO:
    # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
    # if that is the case we take it that the correction did not work well so we won't use it
    cond_rhohv1 = (
                    qvps[X_RHO].where(qvps[X_RHO]>min_rho).resample({"time":"D"}).std(dim=("time", "z")) < \
                    qvps["RHOHV"].where(qvps["RHOHV"]>min_rho).resample({"time":"D"}).std(dim=("time", "z"))*(1+std_tolerance)
                    ).compute()

    # Check that the corrected RHOHV have overall higher mean than the original (1 - mean_tolerance)
    # if that is the case we take it that the correction did not work well so we won't use it
    cond_rhohv2 = ( qvps[X_RHO].resample({"time":"D"}).mean(dim=("time", "z")) > \
                   qvps["RHOHV"].resample({"time":"D"}).mean(dim=("time", "z"))*(1-mean_tolerance) ).compute()

    cond_rhohv = cond_rhohv1 * cond_rhohv2

    # create an xarray.Dataarray with the valid timesteps
    valid_dates = cond_rhohv.where(cond_rhohv, drop=True).time.dt.date
    valid_datetimes = [date.values in valid_dates for date in qvps.time.dt.date]
    valid_datetimes_xr = xr.DataArray(valid_datetimes, coords={"time": qvps["time"]})

    # Redefine RHOHV_NC: keep it in the valid datetimes, put RHOHV in the rest
    qvps[X_RHO] = qvps[X_RHO].where(valid_datetimes_xr, qvps["RHOHV"])

# Conditions to clean ML height values
max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
time_window = 5 # set timestep window for the std computation (centered)
min_period = 3 # set minimum number of valid ML values in the window (centered)

cond_ML_bottom_change = abs(qvps["height_ml_bottom_new_gia"].diff("time").compute())<max_change
cond_ML_bottom_std = qvps["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_bottom_minlen = qvps["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

cond_ML_top_change = abs(qvps["height_ml_new_gia"].diff("time").compute())<max_change
cond_ML_top_std = qvps["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_top_minlen = qvps["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

# Filter only fully stratiform pixels (min entropy >= min_entropy_thresh and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=min_entropy_thresh).compute() & allcond, drop=True)
# Relaxed alternative: Filter qvps with at least 50% of stratiform pixels (min entropy >= min_entropy_thresh and ML detected)
qvps_strat_relaxed = qvps.where( ( (qvps["min_entropy"]>=min_entropy_thresh).sum("z").compute() >= qvps[X_DBZH].count("z").compute()/2 ) & allcond, drop=True)

# Filter out non relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[X_TH] > -10 )&
                                  (qvps_strat[X_KDP] > -0.1)&
                                  (qvps_strat[X_KDP] < 3)&
                                  (qvps_strat[X_RHO] > 0.7)&
                                  (qvps_strat[X_ZDR] > -1) &
                                  (qvps_strat[X_ZDR] < 3))

qvps_strat_relaxed_fil = qvps_strat_relaxed.where((qvps_strat_relaxed[X_TH] > -10 )&
                                  (qvps_strat_relaxed[X_KDP] > -0.1)&
                                  (qvps_strat_relaxed[X_KDP] < 3)&
                                  (qvps_strat_relaxed[X_RHO] > 0.7)&
                                  (qvps_strat_relaxed[X_ZDR] > -1) &
                                  (qvps_strat_relaxed[X_ZDR] < 3))

try:
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRHC"]>10)
    qvps_strat_relaxed_fil = qvps_strat_relaxed_fil.where(qvps_strat_relaxed_fil["SNRHC"]>10)
except KeyError:
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRH"]>10)
    qvps_strat_relaxed_fil = qvps_strat_relaxed_fil.where(qvps_strat_relaxed_fil["SNRH"]>10)
except:
    print("Could not filter out low SNR")

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#### Calculate retrievals
if calculate_retrievals:
    # We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil
    start_time = time.time()
    print("Calculating microphysical retrievals...")

    # to check the wavelength of each radar, in cm for DWD, in 1/100 cm for DMI ()
    # filewl = ""
    # xr.open_dataset(filewl, group="how") # DWD
    # file1 = realpep_path+"/upload/jgiles/dmi_raw/acq/OLDDATA/uza/RADAR/2015/01/01/ANK/RAW/ANK150101000008.RAW6M00"
    # xd.io.backends.iris.IrisRawFile(file1, loaddata=False).ingest_header["task_configuration"]["task_misc_info"]["wavelength"]

    Lambda = 53.1 # radar wavelength in mm (pro: 53.138, ANK: 53.1, AFY: 53.3, GZT: 53.3, HTY: 53.3, SVS:53.3)

    # We will put the final retrievals in a dict
    try: # check if exists, if not, create it
        retrievals_qvpbased
    except NameError:
        retrievals_qvpbased = {}

    for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
        print("   ... for "+stratname)
        retrievals_qvpbased[stratname] = {}
        retrievals_qvpbased[stratname][find_loc(locs, ff[0])] = utils.calc_microphys_retrievals(stratqvp, Lambda = Lambda, mu=0.33,
                                      X_DBZH=X_DBZH, X_ZDR=X_ZDR, X_KDP=X_KDP, X_TEMP="TEMP",
                                      X_PHI=X_PHI )

        # Save retrievals
        for ll in retrievals_qvpbased[stratname].keys():
            retrievals_qvpbased[stratname][ll].to_netcdf(realpep_path+"/upload/jgiles/radar_retrievals_QVPbased/"+stratname+"/"+ll+".nc")

# Check also if the retrievals are already in the QVP
try: # check if exists, if not, create it
    retrievals
except NameError:
    retrievals = {}

for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
    retrievals_namelist = [
        "lwc_zh_zdr_reimann2021",
        "lwc_zh_zdr_rhyzkov2022",
        "lwc_kdp_reimann2021",
        "lwc_ah_reimann2021",
        "lwc_hybrid_reimann2021"
        "iwc_zh_t_hogan2006",
        "iwc_zh_t_hogan2006_model",
        "iwc_zh_t_hogan2006_combined",
        "iwc_zdr_zh_kdp_carlin2021",
        "Dm_ice_zh_matrosov2019",
        "Dm_ice_zh_kdp_carlin2021",
        "Dm_ice_zdp_kdp_carlin2021",
        "Dm_hybrid_blanke2023",
        "Dm_rain_zdr_chen",
        "Dm_rain_zdr_hu2022",
        "Dm_rain_zdr_bringi2009",
        "Nt_ice_iwc_zh_t_hu2022",
        "Nt_ice_iwc_zh_t_carlin2021",
        "Nt_ice_iwc_zh_t_combined_hu2022",
        "Nt_ice_iwc_zh_t_combined_carlin2021",
        "Nt_ice_iwc_zdr_zh_kdp_hu2022",
        "Nt_ice_iwc_zdr_zh_kdp_carlin2021",
        "Nt_rain_zh_zdr_rhyzkov2020",
        ]
    retrievals[stratname] = {}
    retrievals[stratname][find_loc(locs, ff[0])] = xr.Dataset({key: stratqvp[key] for key in retrievals_namelist if key in stratqvp.data_vars})

    # Save retrievals
    if not os.path.exists(realpep_path+"/upload/jgiles/radar_retrievals/"+stratname):
        os.makedirs(realpep_path+"/upload/jgiles/radar_retrievals/"+stratname)
    for ll in retrievals[stratname].keys():
        retrievals[stratname][ll].to_netcdf(realpep_path+"/upload/jgiles/radar_retrievals/"+stratname+"/"+ll+".nc")

#### General statistics
print("Calculating statistics ...")

# We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil

z_snow_over_ML = 300 # set the height above the ML from where to consider snow. 300 m like in https://doi.org/10.1175/JAMC-D-19-0128.1
z_rain_below_ML = 300 # set the height below the ML from where to consider rain. 300 m like in https://doi.org/10.1175/JAMC-D-19-0128.1
z_grad_above_ML = 2000 # height above the ML until which to compute the gradient

# We will put the final stats in a dict
try: # check if exists, if not, create it
    stats
except NameError:
    stats = {}

for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
    print("   ... for "+stratname)

    stats[stratname] = {}

    values_sfc = stratqvp.where( (stratqvp["z"] < (stratqvp["height_ml_bottom_new_gia"]+stratqvp["z"][0])/2) ).bfill("z").isel({"z": 0}) # selects the closest value to the ground starting from below half of the ML height (with respect to the radar altitude)
    values_snow = stratqvp.where( (stratqvp["z"] > stratqvp["height_ml_new_gia"]) ).bfill("z").ffill("z").sel({"z": stratqvp["height_ml_new_gia"] + z_snow_over_ML}, method="nearest")
    values_rain = stratqvp.where( (stratqvp["z"] < stratqvp["height_ml_bottom_new_gia"]) ).ffill("z").bfill("z").sel({"z": stratqvp["height_ml_bottom_new_gia"] - z_rain_below_ML}, method="nearest")

    #### ML statistics
    # select values inside the ML
    qvps_ML = stratqvp.where( (stratqvp["z"] < stratqvp["height_ml_new_gia"]) & \
                                   (stratqvp["z"] > stratqvp["height_ml_bottom_new_gia"]), drop=True)

    values_ML_max = qvps_ML.max(dim="z")
    values_ML_min = qvps_ML.min(dim="z")
    values_ML_mean = qvps_ML.mean(dim="z")
    ML_thickness = (qvps_ML["height_ml_new_gia"] - qvps_ML["height_ml_bottom_new_gia"]).rename("ML_thickness")
    ML_bottom = qvps_ML["height_ml_bottom_new_gia"]

    ML_bottom_TEMP = stratqvp["TEMP"].sel(z=stratqvp["height_ml_bottom_new_gia"], method="nearest")
    ML_thickness_TEMP = ML_bottom_TEMP - stratqvp["TEMP"].sel(z=stratqvp["height_ml_new_gia"], method="nearest")

    #!!! Temporary solution with np.isfinite because there are -inf and inf values in ANK data
    height_ML_max = qvps_ML.where(np.isfinite(qvps_ML)).idxmax("z", skipna=True)
    height_ML_min = qvps_ML.where(np.isfinite(qvps_ML)).idxmin("z", skipna=True)

    # Silke style
    # select timesteps with detected ML
    # gradient_silke = stratqvp.where(stratqvp["height_ml_new_gia"] > stratqvp["height_ml_bottom_new_gia"], drop=True)
    # gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
    # gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
    # gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
    # beta = gradient_final[X_TH] #### TH OR DBZH??

    # Gradient above the ML
    # We select above z_snow_over_ML and below z_snow_over_ML + z_grad_above_ML
    # Then we compute the gradient as the linear fit of the valid values

    beta = stratqvp.where(stratqvp["z"] > (stratqvp["height_ml_new_gia"] + z_snow_over_ML) ) \
                        .where(stratqvp["z"] < (stratqvp["height_ml_new_gia"] + z_snow_over_ML + z_grad_above_ML) )\
                            .polyfit("z", 1, skipna=True).isel(degree=0) * 1000 # x1000 to transform the gradients to /km

    beta = beta.rename({var: var.replace("_polyfit_coefficients", "") for var in beta.data_vars})

    # Gradient below the ML
    # We select below z_rain_below_ML
    # Then we compute the gradient as the linear fit of the valid values

    beta_belowML = stratqvp.where(stratqvp["z"] < (stratqvp["height_ml_bottom_new_gia"] - z_rain_below_ML ) )\
                            .polyfit("z", 1, skipna=True).isel(degree=0) * 1000 # x1000 to transform the gradients to /km

    beta_belowML = beta_belowML.rename({var: var.replace("_polyfit_coefficients", "") for var in beta_belowML.data_vars})

    # Cloud top (3 methods)
    # Get the height value of the last not null value with a minimum of entropy 0.2 (this min entropy is to filter out random noise pixels)
    cloudtop = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2 ) \
                        .isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop")
    # Get the height value of the last value > 5 dBZ
    cloudtop_5dbz = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2).where(stratqvp[X_DBZH]>5) \
                        .isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop 5 dBZ")
    # Get the height value of the last value > 10 dBZ
    cloudtop_10dbz = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2).where(stratqvp[X_DBZH]>10) \
                        .isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop 10 dBZ")

    # Temperature of the cloud top (3 methods)
    cloudtop_TEMP = stratqvp["TEMP"].sel({"z": cloudtop}, method="nearest")
    cloudtop_TEMP_5dbz = stratqvp["TEMP"].sel({"z": cloudtop_5dbz}, method="nearest")
    cloudtop_TEMP_10dbz = stratqvp["TEMP"].sel({"z": cloudtop_10dbz}, method="nearest")


    #### DGL statistics
    # select values in the DGL
    qvps_DGL = stratqvp.where(((stratqvp["TEMP"] >= -20)&(stratqvp["TEMP"] <= -10)).compute(), drop=True)

    values_DGL_max = qvps_DGL.max(dim="z")
    values_DGL_min = qvps_DGL.min(dim="z")
    values_DGL_mean = qvps_DGL.mean(dim="z")

    #### Needle zone statistics
    # select values in the NZ
    # qvps_NZ = stratqvp.where(((stratqvp["TEMP"] >= -8)&(stratqvp["TEMP"] <= -1)).compute(), drop=True).unify_chunks()
    qvps_NZ = stratqvp.where(((stratqvp["TEMP"] >= -8)&(stratqvp["TEMP"] <= -1)).compute())

    values_NZ_max = qvps_NZ.max(dim="z")
    values_NZ_min = qvps_NZ.min(dim="z")
    values_NZ_mean = qvps_NZ.mean(dim="z")

    # Put in the dictionary
    stats[stratname][find_loc(locs, ff[0])] = {"values_sfc": values_sfc.compute().copy(deep=True).assign_attrs({"Description": "value closest to the ground (at least lower than half of the ML height)"}),
                                       "values_snow": values_snow.compute().copy(deep=True).assign_attrs({"Description": "value in snow ("" m above the ML)"}),
                                       "values_rain": values_rain.compute().copy(deep=True).assign_attrs({"Description": "value in rain ("+str(z_rain_below_ML)+" m above the ML)"}),
                                       "values_ML_max": values_ML_max.compute().copy(deep=True).assign_attrs({"Description": "maximum value within the ML"}),
                                       "values_ML_min": values_ML_min.compute().copy(deep=True).assign_attrs({"Description": "minimum value within the ML"}),
                                       "values_ML_mean": values_ML_mean.compute().copy(deep=True).assign_attrs({"Description": "mean value within the ML"}),
                                       "height_ML_max": height_ML_max.compute().copy(deep=True).assign_attrs({"Description": "height (z) of the maximum value within the ML"}),
                                       "height_ML_min": height_ML_min.compute().copy(deep=True).assign_attrs({"Description": "height (z) of the minimum value within the ML"}),
                                       "ML_thickness": ML_thickness.compute().copy(deep=True).assign_attrs({"Description": "thickness of the ML (in m)"}),
                                       "ML_bottom": ML_bottom.compute().copy(deep=True).assign_attrs({"Description": "height of the ML bottom (in m)"}),
                                       "ML_thickness_TEMP": ML_thickness_TEMP.compute().copy(deep=True).assign_attrs({"Description": "thickness of the ML (in temperature)"}),
                                       "ML_bottom_TEMP": ML_bottom_TEMP.compute().copy(deep=True).assign_attrs({"Description": "height of the ML bottom (in temperature)"}),
                                       "values_DGL_max": values_DGL_max.compute().copy(deep=True).assign_attrs({"Description": "maximum value within the DGL"}),
                                       "values_DGL_min": values_DGL_min.compute().copy(deep=True).assign_attrs({"Description": "minimum value within the DGL"}),
                                       "values_DGL_mean": values_DGL_mean.compute().copy(deep=True).assign_attrs({"Description": "mean value within the DGL"}),
                                       "values_NZ_max": values_NZ_max.compute().copy(deep=True).assign_attrs({"Description": "maximum value within the NZ"}),
                                       "values_NZ_min": values_NZ_min.compute().copy(deep=True).assign_attrs({"Description": "minimum value within the NZ"}),
                                       "values_NZ_mean": values_NZ_mean.compute().copy(deep=True).assign_attrs({"Description": "mean value within the NZ"}),
                                       "beta": beta.compute().copy(deep=True).assign_attrs({"Description": "Gradient from "+str(z_snow_over_ML)+" to "+str(z_grad_above_ML)+" m above the ML"}),
                                       "beta_belowML": beta_belowML.compute().copy(deep=True).assign_attrs({"Description": "Gradient from the first valid value to "+str(z_rain_below_ML)+" m below the ML"}),
                                       "cloudtop": cloudtop.compute().copy(deep=True).assign_attrs({"Description": "Cloud top height (highest not-null ZH value)"}),
                                       "cloudtop_5dbz": cloudtop_5dbz.compute().copy(deep=True).assign_attrs({"Description": "Cloud top height (highest ZH value > 5 dBZ)"}),
                                       "cloudtop_10dbz": cloudtop_10dbz.compute().copy(deep=True).assign_attrs({"Description": "Cloud top height (highest ZH value > 10 dBZ)"}),
                                       "cloudtop_TEMP": cloudtop_TEMP.compute().copy(deep=True).assign_attrs({"Description": "TEMP at cloud top height (highest not-null ZH value)"}),
                                       "cloudtop_TEMP_5dbz": cloudtop_TEMP_5dbz.compute().copy(deep=True).assign_attrs({"Description": "TEMP at cloud top height (highest ZH value > 5 dBZ)"}),
                                       "cloudtop_TEMP_10dbz": cloudtop_TEMP_10dbz.compute().copy(deep=True).assign_attrs({"Description": "TEMP at cloud top height (highest ZH value > 10 dBZ)"}),
        }

    # Save stats
    if not os.path.exists(realpep_path+"/upload/jgiles/radar_stats/"+stratname):
        os.makedirs(realpep_path+"/upload/jgiles/radar_stats/"+stratname)
    for ll in stats[stratname].keys():
        for xx in stats[stratname][ll].keys():
            stats[stratname][ll][xx].to_netcdf(realpep_path+"/upload/jgiles/radar_stats/"+stratname+"/"+ll+"_"+xx+".nc")


#### Calculate riming
# We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil
print("Calculating riming ...")

# We will put the final riming classification in a dict
try: # check if exists, if not, create it
    riming_classif
except NameError:
    riming_classif = {}

loc = find_loc(locs, ff[0])

# try to load the riming model. Will only work with older sklearn version (wradlib5 env)
try:
    with open('/automount/agradar/jgiles/riming_model/gbm_model_23.10.2024.pkl', 'rb') as f:
        riming_model = pickle.load(f)

    with open('/automount/agradar/jgiles/riming_model/gbm_zh_zdr_model_23.10.2024.pkl', 'rb') as f:
        riming_model_zh_zdr = pickle.load(f)

    def calc_depolarization(da, zdr="ZDR_OC", rho="RHOHV_NC"):
        return xr.apply_ufunc(wrl.dp.depolarization,
                              da[zdr], da[rho].where(da[rho]<1),
                            dask='parallelized',
        )


    for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
        print("   ... for "+stratname)

        riming_classif[stratname] = {}
        riming_classif[stratname][loc] = xr.Dataset()

        if "DR" not in stratqvp:
            DR = calc_depolarization(stratqvp, X_ZDR, X_RHO)
            assign = dict(DR = DR.assign_attrs(
                {'long_name': 'Depolarization ratio based on '+X_ZDR+' and '+X_RHO,
                 'standard_name': 'depolarization_ratio',
                 'units': 'dB'}
                ))
            riming_classif[stratname][loc] = riming_classif[stratname][loc].assign(assign)

        if "UDR" not in stratqvp:
            UDR = calc_depolarization(stratqvp, "ZDR", "RHOHV")
            assign = dict(UDR = UDR.assign_attrs(
                {'long_name': 'Depolarization ratio based on ZDR and RHOHV',
                 'standard_name': 'depolarization_ratio',
                 'units': 'dB'}
                ))
            riming_classif[stratname][loc] = riming_classif[stratname][loc].assign(assign)

        # predict riming with the model
        for XDR, XZDR, XZH in [("DR", X_ZDR, X_DBZH), ("UDR", "ZDR", "DBZH")]:

            idx = np.isfinite(riming_classif[stratname][loc][XDR].values.flatten() + \
                              stratqvp[XZDR].values.flatten() + \
                                  stratqvp[XZH].values.flatten())
            X = np.concatenate((riming_classif[stratname][loc][XDR].values.flatten()[idx].reshape(-1, 1), \
                                stratqvp[XZDR].values.flatten()[idx].reshape(-1, 1), \
                                    stratqvp[XZH].values.flatten()[idx].reshape(-1, 1)), axis=1)

            pred = riming_model.predict(X)

            pred_riming = np.zeros_like(riming_classif[stratname][loc][XDR]).flatten() + np.nan
            pred_riming[idx] = pred
            pred_riming = xr.zeros_like(riming_classif[stratname][loc][XDR]) + pred_riming.reshape(riming_classif[stratname][loc][XDR].shape)

            varname = "riming_"+XDR

            pred_riming = pred_riming.assign_attrs({
                'long_name': 'Riming prediction based on '+XDR+', '+XZDR+' and '+XZH+' with gradient boosting model',
                 'standard_name': 'riming_prediction',
                }).rename(varname)

            assign = {varname: pred_riming.copy()}
            riming_classif[stratname][loc] = riming_classif[stratname][loc].assign(assign)

            # save to file
            if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname):
                os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname)

            pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+loc+"_"+varname+".nc")

        # predict riming with the model that uses only zh and zdr
        for XZDR, XZH in [(X_ZDR, X_DBZH), ("ZDR", "DBZH")]:

            idx = np.isfinite(stratqvp[XZDR].values.flatten() + \
                              stratqvp[XZH].values.flatten())
            X = np.concatenate((stratqvp[XZDR].values.flatten()[idx].reshape(-1, 1), \
                                stratqvp[XZH].values.flatten()[idx].reshape(-1, 1)), axis=1)

            pred = riming_model_zh_zdr.predict(X)

            pred_riming = np.zeros_like(stratqvp[XZH]).flatten() + np.nan
            pred_riming[idx] = pred
            pred_riming = xr.zeros_like(stratqvp[XZH]) + pred_riming.reshape(stratqvp[XZH].shape)

            varname = "riming_"+XZDR+"_"+XZH

            pred_riming = pred_riming.assign_attrs({
                'long_name': 'Riming prediction based on '+XZDR+' and '+XZH+' with gradient boosting model',
                 'standard_name': 'riming_prediction',
                }).rename(varname)

            assign = {varname: pred_riming.copy()}
            riming_classif[stratname][loc] = riming_classif[stratname][loc].assign(assign)

            # save to file
            if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname):
                os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname)

            pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+loc+"_"+varname+".nc")

except ModuleNotFoundError:
    print("... Loading the riming model failed, trying to reload pre-calculated riming ...")
    for stratname in ["stratiform", "stratiform_relaxed"]:
        if stratname not in riming_classif.keys():
            riming_classif[stratname] = {}
        elif type(riming_classif[stratname]) is not dict:
            riming_classif[stratname] = {}
        print("Loading "+stratname+" riming classification ...")
        for ll in [loc]: # ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
            if ll not in riming_classif[stratname].keys():
                riming_classif[stratname][ll] = xr.Dataset()
            elif type(riming_classif[stratname][ll]) is not xr.Dataset:
                riming_classif[stratname][ll] = xr.Dataset()
            for xx in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_'+X_ZDR+'_'+X_DBZH,
                       ]:
                try:
                    riming_classif[stratname][ll] = riming_classif[stratname][ll].assign( xr.open_dataset(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+ll+"_"+xx+".nc") )
                    print(ll+" "+xx+" riming_classif loaded")
                except:
                    pass
            # delete entry if empty
            if not riming_classif[stratname][ll]:
                del riming_classif[stratname][ll]

# Assign
qvps_strat_fil = qvps_strat_fil.assign( riming_classif['stratiform'][loc] )
qvps_strat_relaxed_fil = qvps_strat_relaxed_fil.assign( riming_classif['stratiform_relaxed'][loc] )

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#### Save filtered QVPs to files
print("Saving filtered QVPs to files ...")
for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
    print("   ... "+stratname)
    # save to file
    if not os.path.exists(realpep_path+"/upload/jgiles/stratiform_qvps/"+stratname):
        os.makedirs(realpep_path+"/upload/jgiles/stratiform_qvps/"+stratname)

    stratqvp.to_netcdf(realpep_path+"/upload/jgiles/stratiform_qvps/"+stratname+"/"+ll+".nc")

#%% Reload QVPS
reload_qvps = False
ll = "pro"

if reload_qvps:
    print("Reloading filtered qvps")

    qvps_strat_fil = xr.open_dataset(realpep_path+"/upload/jgiles/stratiform_qvps/stratiform/"+ll+".nc")
    qvps_strat_relaxed_fil = xr.open_dataset(realpep_path+"/upload/jgiles/stratiform_qvps/stratiform_relaxed/"+ll+".nc")


#%% CFTDs Plot

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below and do not save.
auto_plot = True
savepath = "/automount/agradar/jgiles/images/CFTDs/stratiform/"

# Which to plot, qvps_strat_fil or qvps_strat_relaxed_fil
ds_to_plot = qvps_strat_fil.copy()

# Define list of seasons
selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

# adjustment from K to C (disabled now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit (only works if auto_plot=False)
ytlim=-20

# season to plot (only works if auto_plot=False)
selseas = selseaslist[0]
selmonths = selseas[1]

# Temp bins
tb=1# degress C

# Min counts per Temp layer
mincounts=100

#Colorbar limits and step
cblim=[0,10]
colsteps=10

cmaphist="Oranges"

savedict = {"custom": None} # placeholder for the for loop below, not important

# Plot horizontally
# DMI
# Native worst-resolution of the data (for 1-byte moments)
# DBZH: 0.5 dB
# ZDR: 0.0625 dB
# KDP: complicated. From 0.013 at KDP approaching zero to 7.42 at extreme KDP. KDP min absolute value is 0.25 and max abs is 150 (both positive and negative)
# RHOHV: scales with a square root (finer towards RHOHV=1), so from 0.00278 at RHOHV=0.7 to 0.002 resolution at RHOHV=1
# PHIDP: 0.708661 deg
if country=="dmi":

    vars_to_plot = {X_DBZH: [0, 45.5, 0.5],
                    X_ZDR: [-0.505, 2.05, 0.1],
                    X_KDP:  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                    "RHOHV": [0.9, 1.002, 0.002]}

    if auto_plot:
        vtp = [{X_DBZH: [0, 45.5, 0.5],
                        X_ZDR: [-0.505, 2.05, 0.1],
                        X_KDP:  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                        X_RHO: [0.9, 1.002, 0.002]},
               {"DBZH": [0, 45.5, 0.5],
                               "ZDR": [-0.505, 2.05, 0.1],
                               "KDP_CONV":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                               "RHOHV": [0.9, 1.002, 0.002]} ]
        ytlimlist = [-20, -50]
        loc = find_loc(locs, ff[0])
        add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
        savedict = {}
        for selseas in selseaslist:
            savedict.update(
                        {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+".png": [vtp[0], ytlimlist[0], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_extended.png": [vtp[0], ytlimlist[1], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr.png": [vtp[1], ytlimlist[0], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr_extended.png": [vtp[1], ytlimlist[1], selseas[1]],
                        }
                            )

    for savename in savedict.keys():
        if auto_plot:
            vars_to_plot = savedict[savename][0]
            ytlim = savedict[savename][1]
            selmonths = savedict[savename][2]

        fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

        for nn, vv in enumerate(vars_to_plot.keys()):
            so=False
            binsx2=None
            rd=10 # arbitrarily large decimal position to round to (so it is actually not rounded)
            if "DBZH" in vv:
                so=True
                binsx2 = [0, 46, 1]
                rd = 1 # decimal position to round to
            if "ZDR" in vv:
                so=True
                binsx2 = [-0.5, 2.1, 0.1]
                rd=1
            if "KDP" in vv:
                so=True #True
                binsx2 = [-0.1, 0.52, 0.02]
                rd=2
            if "RHOHV" in vv:
                so = True
                binsx2 = [0.9, 1.005, 0.005]
                rd=3

            #!!! For some reason SVS now requires rechunking here
            utils.hist2d(ax[nn], ds_to_plot[vv].chunk({"time":-1}).sel(\
                                                    time=ds_to_plot['time'].dt.month.isin(selmonths)).round(rd),
                         ds_to_plot["TEMP"].sel(\
                                             time=ds_to_plot['time'].dt.month.isin(selmonths))+adjtemp,
                         whole_x_range=True,
                         binsx=vars_to_plot[vv], binsy=[ytlim,16,tb], mode='rel_y', qq=0.2,
                         cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps,
                         fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                         cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
            ax[nn].set_ylim(15,ytlim)
            ax[nn].set_xlabel(vv, fontsize=10)

            ax[nn].tick_params(labelsize=15) #change font size of ticks
            plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts

        ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')

        if auto_plot:
            # Create savefolder
            savepath_seas = os.path.dirname(savepath+savename)
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            fig.savefig(savepath+savename, bbox_inches="tight", dpi=300)
            print("AUTO PLOT: saved "+savename)



# DWD
# plot CFTDs moments
if country=="dwd":

    vars_to_plot = {X_DBZH: [0, 46, 1],
                    X_ZDR: [-0.5, 2.1, 0.1],
                    X_KDP: [-0.1, 0.52, 0.02],
                    X_RHO: [0.9, 1.004, 0.004]}

    if auto_plot:
        vtp = [{X_DBZH: [0, 46, 1],
                        X_ZDR: [-0.5, 2.1, 0.1],
                        X_KDP:  [-0.1, 0.52, 0.02],
                        X_RHO: [0.9, 1.004, 0.004]},
               {"DBZH": [0, 46, 1],
                               "ZDR": [-0.5, 2.1, 0.1],
                               "KDP_CONV":  [-0.1, 0.52, 0.02],
                               "RHOHV": [0.9, 1.004, 0.004]} ]
        ytlimlist = [-20, -50]
        loc = find_loc(locs, ff[0])
        savedict = {}
        add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
        for selseas in selseaslist:
            savedict.update(
                        {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+".png": [vtp[0], ytlimlist[0], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_extended.png": [vtp[0], ytlimlist[1], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr.png": [vtp[1], ytlimlist[0], selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr_extended.png": [vtp[1], ytlimlist[1], selseas[1]],
                        }
                            )

    for savename in savedict.keys():
        if auto_plot:
            vars_to_plot = savedict[savename][0]
            ytlim = savedict[savename][1]
            selmonths = savedict[savename][2]

        fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

        for nn, vv in enumerate(vars_to_plot.keys()):
            so=False
            binsx2=None
            adj=1
            if "RHOHV" in vv:
                so = True
                binsx2 = [0.9, 1.005, 0.005]
            if "KDP" in vv:
                adj=1
            utils.hist2d(ax[nn], ds_to_plot[vv].sel(time=ds_to_plot['time'].dt.month.isin(selmonths))*adj,
                         ds_to_plot["TEMP"].sel(time=ds_to_plot['time'].dt.month.isin(selmonths))+adjtemp,
                         whole_x_range=True,
                         binsx=vars_to_plot[vv], binsy=[ytlim,16,tb], mode='rel_y', qq=0.2,
                         cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps,
                         fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                         cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
            ax[nn].set_ylim(15,ytlim)
            ax[nn].set_xlabel(vv, fontsize=10)

            ax[nn].tick_params(labelsize=15) #change font size of ticks
            plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts

        ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')
        if auto_plot:
            # Create savefolder
            savepath_seas = os.path.dirname(savepath+savename)
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            fig.savefig(savepath+savename, bbox_inches="tight", dpi=300)
            print("AUTO PLOT: saved "+savename)


#%% CFTDs retrievals Plot
# We assume that everything above ML is frozen and everything below is liquid

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below (selecting the first option of
# savepath_list and ds_to_plot_list) and do not save.
auto_plot = True
savepath_list = [
                "/automount/agradar/jgiles/images/CFTDs/stratiform/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_QVPbased/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_KDPpos/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_KDPpos_QVPbased/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_relaxed/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_relaxed_QVPbased/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_relaxed_KDPpos/",
                "/automount/agradar/jgiles/images/CFTDs/stratiform_relaxed_KDPpos_QVPbased/",
                 ]

# Which to plot, retrievals or retrievals_qvpbased, stratiform or stratiform_relaxed
loc = find_loc(locs, ff[0]) # by default, plot only the histograms of the currently loaded QVPs.
ds_to_plot_list = [
                    retrievals["stratiform"][loc].copy(),
                    retrievals_qvpbased["stratiform"][loc].copy(),
                    retrievals["stratiform"][loc].copy().where(qvps_strat_fil.KDP_ML_corrected>0.01),
                    retrievals_qvpbased["stratiform"][loc].copy().where(qvps_strat_fil.KDP_ML_corrected>0.01),
                    retrievals["stratiform_relaxed"][loc].copy(),
                    retrievals_qvpbased["stratiform_relaxed"][loc].copy(),
                    retrievals["stratiform_relaxed"][loc].copy().where(qvps_strat_relaxed_fil.KDP_ML_corrected>0.01),
                    retrievals_qvpbased["stratiform_relaxed"][loc].copy().where(qvps_strat_relaxed_fil.KDP_ML_corrected>0.01),
                    ]


# Define list of seasons
selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

# adjustment from K to C (disabled now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit (only works if auto_plot=False)
ytlim=-20

# season to plot (only works if auto_plot=False)
selseas = selseaslist[0]
selmonths = selseas[1]

# Select which retrievals to plot (only works if auto_plot=False)
IWC = "iwc_zdr_zh_kdp_carlin2021" # iwc_zh_t_hogan2006, iwc_zh_t_hogan2006_model, iwc_zh_t_hogan2006_combined, iwc_zdr_zh_kdp_carlin2021
LWC = "lwc_hybrid_reimann2021" # lwc_zh_zdr_reimann2021, lwc_zh_zdr_rhyzkov2022, lwc_kdp_reimann2021, lwc_ah_reimann2021, lwc_hybrid_reimann2021
Dm_ice = "Dm_ice_zdp_kdp_carlin2021" # Dm_ice_zh_matrosov2019, Dm_ice_zh_kdp_carlin2021, Dm_ice_zdp_kdp_carlin2021, Dm_hybrid_blanke2023
Dm_rain = "Dm_rain_zdr_bringi2009" # Dm_rain_zdr_chen, Dm_rain_zdr_hu2022, Dm_rain_zdr_bringi2009
Nt_ice = "Nt_ice_iwc_zdr_zh_kdp_carlin2021" # Nt_ice_iwc_zh_t_hu2022, Nt_ice_iwc_zh_t_carlin2021, Nt_ice_iwc_zh_t_combined_hu2022, Nt_ice_iwc_zh_t_combined_carlin2021, Nt_ice_iwc_zdr_zh_kdp_hu2022, Nt_ice_iwc_zdr_zh_kdp_carlin2021
Nt_rain = "Nt_rain_zh_zdr_rhyzkov2020" # Nt_rain_zh_zdr_rhyzkov2020

vars_to_plot = {"IWC/LWC [g/m^{3}]": [-0.1, 0.82, 0.02], # [-0.1, 0.82, 0.02],
                "Dm [mm]": [0, 4.1, 0.1], # [0, 3.1, 0.1],
                "Nt [log10(1/L)]": [-2, 2.1, 0.1], # [-2, 2.1, 0.1],
                }

savedict = {"custom": None} # placeholder for the for loop below, not important

for sn, savepath in enumerate(savepath_list):
    ds_to_plot = ds_to_plot_list[sn]

    if auto_plot:
        ytlimlist = [-20, -50]
        add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
        savedict = {}
        for selseas in selseaslist:
            savedict.update(
                        {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys.png": [ytlimlist[0],
                                    "iwc_zh_t_hogan2006_model", "lwc_zh_zdr_reimann2021",
                                    "Dm_ice_zh_matrosov2019", "Dm_rain_zdr_bringi2009",
                                    "Nt_ice_iwc_zh_t_carlin2021", "Nt_rain_zh_zdr_rhyzkov2020", selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_extended.png": [ytlimlist[1],
                                    "iwc_zh_t_hogan2006_model", "lwc_zh_zdr_reimann2021",
                                    "Dm_ice_zh_matrosov2019", "Dm_rain_zdr_bringi2009",
                                    "Nt_ice_iwc_zh_t_carlin2021", "Nt_rain_zh_zdr_rhyzkov2020", selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP.png": [ytlimlist[0],
                                    "iwc_zdr_zh_kdp_carlin2021", "lwc_hybrid_reimann2021",
                                    "Dm_ice_zdp_kdp_carlin2021", "Dm_rain_zdr_bringi2009",
                                    "Nt_ice_iwc_zdr_zh_kdp_carlin2021", "Nt_rain_zh_zdr_rhyzkov2020", selseas[1]],
                        selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP_extended.png": [ytlimlist[1],
                                    "iwc_zdr_zh_kdp_carlin2021", "lwc_hybrid_reimann2021",
                                    "Dm_ice_zdp_kdp_carlin2021", "Dm_rain_zdr_bringi2009",
                                    "Nt_ice_iwc_zdr_zh_kdp_carlin2021", "Nt_rain_zh_zdr_rhyzkov2020", selseas[1]],
                        }
                    )

    for savename in savedict.keys():
        if auto_plot:
            ytlim = savedict[savename][0]
            IWC = savedict[savename][1]
            LWC = savedict[savename][2]
            Dm_ice = savedict[savename][3]
            Dm_rain = savedict[savename][4]
            Nt_ice = savedict[savename][5]
            Nt_rain = savedict[savename][6]
            selmonths = savedict[savename][7]

        try:
            retreivals_merged = xr.Dataset({
                                            "IWC/LWC [g/m^{3}]": ds_to_plot[IWC].where(ds_to_plot[IWC].z > ds_to_plot.height_ml_new_gia,
                                                                              ds_to_plot[LWC].where(ds_to_plot[LWC].z < ds_to_plot.height_ml_bottom_new_gia ) ),
                                            "Dm [mm]": ds_to_plot[Dm_ice].where(ds_to_plot[Dm_ice].z > ds_to_plot.height_ml_new_gia,
                                                                              ds_to_plot[Dm_rain].where(ds_to_plot[Dm_rain].z < ds_to_plot.height_ml_bottom_new_gia ) ),
                                            "Nt [log10(1/L)]": (ds_to_plot[Nt_ice].where(ds_to_plot[Nt_ice].z > ds_to_plot.height_ml_new_gia,
                                                                              ds_to_plot[Nt_rain].where(ds_to_plot[Nt_rain].z < ds_to_plot.height_ml_bottom_new_gia ) ) ),
                })
        except KeyError:
            print("Unable to plot "+savename+". Some retrieval is not present in the dataset.")
            continue

        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15,5), width_ratios=(1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

        for nn, vv in enumerate(vars_to_plot.keys()):
            so=False
            binsx2=None
            adj=1
            if "RHOHV" in vv:
                so = True
                binsx2 = [0.9, 1.005, 0.005]
            if "KDP" in vv:
                adj=1
            #!!! For some reason SVS now requires rechunking here
            utils.hist2d(ax[nn], retreivals_merged[vv].chunk({"time":-1}).sel(time=retreivals_merged['time'].dt.month.isin(selmonths))*adj,
                         retreivals_merged["TEMP"].sel(time=retreivals_merged['time'].dt.month.isin(selmonths))+adjtemp,
                         whole_x_range=True,
                         binsx=vars_to_plot[vv], binsy=[ytlim,16,tb], mode='rel_y', qq=0.2,
                         cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps,
                         fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                         cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
            ax[nn].set_ylim(15,ytlim)
            ax[nn].set_xlabel(vv, fontsize=10)

            ax[nn].tick_params(labelsize=15) #change font size of ticks
            plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts

        ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')

        if auto_plot:
            # Create savefolder
            savepath_seas = os.path.dirname(savepath+savename)
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            fig.savefig(savepath+savename, bbox_inches="tight", dpi=300)
            print("AUTO PLOT: saved "+savename)

    if auto_plot is False:
        break

#%% Check particular dates

# Plot QVP
visdict14 = radarmet.visdict14

def plot_qvp(data, momname="DBZH", tloc=slice("2015-01-01", "2020-12-31"), plot_ml=True, plot_entropy=False, add_riming=None, **kwargs):
    mom=momname
    # norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
    # cmap = mpl.cm.get_cmap("HomeyerRainbow")
    # cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
    ticks = radarmet.visdict14[mom]["ticks"]
    cmap = visdict14[mom]["cmap"]
    # cmap = "miub2"
    norm = utils.get_discrete_norm(ticks, cmap, extend="both")

    data[momname].loc[{"time":tloc}].dropna("z", how="all").plot(x="time", cmap=cmap, norm=norm, extend="both", **kwargs)

    if plot_ml:
        try:
            data.loc[{"time":tloc}].height_ml_bottom_new_gia.plot(color="black")
            # data.loc[{"time":tloc}].height_ml_bottom_new_gia.plot(color="white",ls=":")
            data.loc[{"time":tloc}].height_ml_new_gia.plot(color="black")
            # data.loc[{"time":tloc}].height_ml_new_gia.plot(color="white",ls=":")
        except KeyError:
            print("No ML in data")
    if plot_entropy:
        try:
            data["min_entropy"].loc[{"time":tloc}].dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[min_entropy_thresh, 1], hatches=["", "X", ""], colors=[(1,1,1,0)], add_colorbar=False, extend="both")
        except:
            print("Plotting entropy failed")
    try:
        # select the times in the riming ds
        add_riming_tloc = add_riming.loc[{"time":tloc}]
        # add riming with hatches
        add_riming_tloc.where(add_riming_tloc>0.9).where(add_riming_tloc.z>add_riming_tloc.height_ml_new_gia).dropna("z", how="all").plot.contourf(x="time", levels=[min_entropy_thresh,1.1], hatches=["","**", ""], colors=[(1,1,1,0)], add_colorbar=False, extend="both")
        # add riming with color shade
        add_riming_tloc.where(add_riming_tloc>0.9).where(add_riming_tloc.z>add_riming_tloc.height_ml_new_gia).dropna("z", how="all").plot.contourf(x="time", levels=[min_entropy_thresh,1.1], colors="gray", add_colorbar=False, alpha=0.9)
    except:
        None
    plt.title(mom)

qvps_fix = qvps.copy()
# qvps_fix["KDP_ML_corrected"] = qvps_fix["KDP_ML_corrected"].where(qvps_fix.height_ml_new_gia.notnull(),  qvps_fix["KDP_CONV"])
with mpl.rc_context({'font.size': 10}):
    plot_qvp(qvps_fix, "ZDR", tloc="2020-08-02", plot_ml=True, plot_entropy=True,
              # add_riming = qvps_strat_relaxed_fil.riming_DR,
             ylim = (qvps.altitude,10000),
              xlim=[datetime.date(2020, 8, 2), datetime.date(2020, 8, 3)],
             )


# qvps_strat_fil_notime = qvps_strat_fil.copy()
# qvps_strat_fil_notime = qvps_strat_fil_notime.reset_index("time")
# plot_qvp(qvps_strat_fil_notime, "ZDR_OC", tloc="2020-07-15", plot_ml=True, plot_entropy=True, ylim=(qvps.altitude,10000))

#%% Riming statistics
#%%% reload riming estimates
locs_to_load = locs #[find_loc(locs, ff[0])] # by default, load only the histograms of the currently loaded QVPs.

try: # check if exists, if not, create it
    riming_classif
except NameError:
    riming_classif = {}

for stratname in ["stratiform", "stratiform_relaxed"]:
    if stratname not in riming_classif.keys():
        riming_classif[stratname] = {}
    elif type(riming_classif[stratname]) is not dict:
        riming_classif[stratname] = {}
    print("Loading "+stratname+" riming classification ...")
    for ll in locs_to_load: # ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
        if ll not in riming_classif[stratname].keys():
            riming_classif[stratname][ll] = xr.Dataset()
        elif type(riming_classif[stratname][ll]) is not xr.Dataset:
            riming_classif[stratname][ll] = xr.Dataset()
        for xx in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_ZDR_OC_DBZH',
                   ]:
            try:
                riming_classif[stratname][ll] = riming_classif[stratname][ll].assign( xr.open_dataset(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+ll+"_"+xx+".nc") )
                print(ll+" "+xx+" riming_classif loaded")
            except:
                pass
        # delete entry if empty
        if not riming_classif[stratname][ll]:
            del riming_classif[stratname][ll]

#%%% Plot riming histograms
locs_to_plot = locs #[find_loc(locs, ff[0])] # by default, plot only the histograms of the currently loaded QVPs.
savepath = "/automount/agradar/jgiles/images/stats_histograms/"

selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

print("Plotting riming histograms ...")
start_time = time.time()

for loc in locs_to_plot:
    print(" ... "+loc)

    for stratname in ["stratiform", "stratiform_relaxed"]:
        print(" ... ... "+stratname)

        to_plot = riming_classif[stratname][loc].chunk({"time":"auto"}).where(\
                                                           riming_classif[stratname][loc].z >= riming_classif[stratname][loc].height_ml_new_gia,
                                                        drop=True)

        for selseas in selseaslist:
            print(" ... ... ... "+selseas[0])

            # Create savefolder
            savepath_seas = savepath+stratname+"/"+selseas[0]+"/"+loc+"/"
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)

            to_plot_sel = to_plot.sel(\
                                time=to_plot['time'].dt.month.isin(selseas[1]))

            for vv in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_ZDR_OC_DBZH',
                       ]:

                try:

                    # Create temperature bins (1-degree intervals)
                    temp_bins = np.arange(-20, 1)

                    # Create an empty list to store the values
                    percentages = []
                    count = []

                    # Loop through each temperature bin
                    for i in range(len(temp_bins) - 1):
                        # Mask for the current temperature bin
                        temp_mask = (to_plot_sel.TEMP >= temp_bins[i]) & (to_plot_sel.TEMP < temp_bins[i+1])

                        # Get the data corresponding to the current temperature bin
                        data_in_bin = to_plot_sel[vv].where(temp_mask.compute(), drop=True)

                        # Calculate the percentage of 1s (ignoring NaNs)
                        total_values = np.isfinite(data_in_bin).sum()  # Total number of finite values (non-nan)
                        ones_count = (data_in_bin == 1).sum()          # Count of values equal to 1
                        percentage = (ones_count / total_values) * 100 if total_values > 0 else np.nan

                        # Append the percentage to the list
                        percentages.append(percentage.values)

                        # Append the total_values to the list
                        count.append(total_values.values)

                    # Plot the percentage against temperature
                    fig = plt.figure(figsize=(8, 6))
                    # plt.plot(percentages, temp_bins[:-1], marker='o', linestyle='-')
                    plt.step(percentages, temp_bins[:-1], where="post")
                    plt.xlabel('Percentage of rimed events [%]')
                    plt.ylabel('Temperature [°C]')
                    plt.title('Percentage of '+vv+" "+stratname+" "+selseas[0]+" "+loc)
                    plt.xlim(0, 70)
                    plt.gca().yaxis.set_inverted(True)
                    plt.grid(True)
                    ax2 = plt.twiny()
                    ax2.plot(count, temp_bins[:-1]-0.5, color="red")
                    plt.xlabel("Number of events", color="red")
                    # plt.show

                    fig.savefig(savepath_seas+"/"+loc+"_"+vv+"_vsTEMP.png",
                                    bbox_inches="tight")
                    plt.close(fig)


                    # Repeat for height above ML in the y-axis
                    # Create z bins
                    z_bins = np.arange(0, 6215, 215)

                    # Create an empty list to store the values
                    percentages = []
                    count = []

                    # Loop through each temperature bin
                    for i in range(len(z_bins) - 1):
                        # Mask for the current z bin
                        z_mask = ( (to_plot_sel.z - to_plot_sel.height_ml_new_gia) >= z_bins[i]) & ( (to_plot_sel.z - to_plot_sel.height_ml_new_gia) < z_bins[i+1])

                        # Get the data corresponding to the current z bin
                        data_in_bin = to_plot_sel[vv].where(z_mask.compute(), drop=True)

                        # Calculate the percentage of 1s (ignoring NaNs)
                        total_values = np.isfinite(data_in_bin).sum().values  # Total number of finite values (non-nan)
                        ones_count = (data_in_bin == 1).sum().values          # Count of values equal to 1
                        percentage = (ones_count / total_values) * 100 if total_values > 0 else np.nan

                        # Append the percentage to the list
                        percentages.append(percentage)

                        # Append the total_values to the list
                        count.append(total_values)

                    # Plot the percentage against height above ML
                    fig = plt.figure(figsize=(8, 6))
                    # plt.plot(percentages, temp_bins[:-1], marker='o', linestyle='-')
                    plt.step(percentages, z_bins[:-1], where="pre")
                    plt.xlabel('Percentage of rimed events [%]')
                    plt.ylabel('Height above ML [m]')
                    plt.title('Percentage of '+vv+" "+stratname+" "+selseas[0]+" "+loc)
                    plt.xlim(0, 70)
                    plt.grid(True)
                    ax2 = plt.twiny()
                    ax2.plot(count, z_bins[:-1]+107.5, color="red")
                    plt.xlabel("Number of events", color="red")
                    # plt.show

                    fig.savefig(savepath_seas+"/"+loc+"_"+vv+"_vsHeight.png",
                                    bbox_inches="tight")
                    plt.close(fig)

                except:
                    print("!!! Unable to plot "+vv+" !!!")

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%% Statistics histograms and ridgeplots
# load stats
if 'stats' not in globals() and 'stats' not in locals():
    stats = {}

for stratname in ["stratiform", "stratiform_relaxed"]:
    if stratname not in stats.keys():
        stats[stratname] = {}
    elif type(stats[stratname]) is not dict:
        stats[stratname] = {}
    print("Loading "+stratname+" stats ...")
    for ll in locs:
        if ll not in stats[stratname].keys():
            stats[stratname][ll] = {}
        elif type(stats[stratname][ll]) is not dict:
            stats[stratname][ll] = {}
        for xx in ['values_sfc', 'values_snow', 'values_rain', 'values_ML_max', 'values_ML_min', 'values_ML_mean',
                   'ML_thickness', 'ML_bottom', 'ML_thickness_TEMP', 'ML_bottom_TEMP', 'values_DGL_max', 'values_DGL_min',
                   'values_DGL_mean', 'values_NZ_max', 'values_NZ_min', 'values_NZ_mean', 'height_ML_max', 'height_ML_min',
                   'ML_bottom', 'beta', 'beta_belowML', 'cloudtop', 'cloudtop_5dbz', 'cloudtop_10dbz',
                   'cloudtop_TEMP', 'cloudtop_TEMP_5dbz', 'cloudtop_TEMP_10dbz']:
            try:
                stats[stratname][ll][xx] = xr.open_dataset(realpep_path+"/upload/jgiles/radar_stats/"+stratname+"/"+ll+"_"+xx+".nc")
                if len(stats[stratname][ll][xx].data_vars)==1:
                    # if only 1 var, convert to data array
                    stats[stratname][ll][xx] = stats[stratname][ll][xx].to_dataarray()
                if "variable" in stats[stratname][ll][xx].coords:
                    if len(stats[stratname][ll][xx]["variable"]) == 1:
                        # if there is a generic coord called "variable", remove it
                        stats[stratname][ll][xx] = stats[stratname][ll][xx].isel(variable=0)
                print(ll+" "+xx+" stats loaded")
            except:
                pass
        # delete entry if empty
        if not stats[stratname][ll]:
            del stats[stratname][ll]

# load retrievals
if 'retrievals' not in globals() and 'retrievals' not in locals():
    retrievals = {}
if 'retrievals_qvpbased' not in globals() and 'retrievals_qvpbased' not in locals():
    retrievals_qvpbased = {}

for stratname in ["stratiform", "stratiform_relaxed"]:
    if stratname not in retrievals.keys():
        retrievals[stratname] = {}
    elif type(retrievals[stratname]) is not dict:
        retrievals[stratname] = {}
    print("Loading "+stratname+" retrievals ...")
    for ll in locs:
        try:
            retrievals[stratname][ll] = xr.open_dataset(realpep_path+"/upload/jgiles/radar_retrievals/"+stratname+"/"+ll+".nc")
            print(ll+" retrievals loaded")
        except:
            pass
        # delete entry if empty
        if not retrievals[stratname][ll]:
            del retrievals[stratname][ll]

for stratname in ["stratiform", "stratiform_relaxed"]:
    if stratname not in retrievals_qvpbased.keys():
        retrievals_qvpbased[stratname] = {}
    elif type(retrievals_qvpbased[stratname]) is not dict:
        retrievals_qvpbased[stratname] = {}
    print("Loading "+stratname+" retrievals_qvpbased ...")
    for ll in locs:
        try:
            retrievals_qvpbased[stratname][ll] = xr.open_dataset(realpep_path+"/upload/jgiles/radar_retrievals_QVPbased/"+stratname+"/"+ll+".nc")
            print(ll+" retrievals_qvpbased loaded")
        except:
            pass
        # delete entry if empty
        if not retrievals_qvpbased[stratname][ll]:
            del retrievals_qvpbased[stratname][ll]

#%%% 2d histograms
locs_to_plot = locs # [find_loc(locs, ff[0])] # by default, plot only the histograms of the currently loaded QVPs.
savepath = "/automount/agradar/jgiles/images/stats_histograms/"

selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

print("Plotting histograms ...")

for loc in locs_to_plot:
    print(" ... "+loc)
    for selseas in selseaslist:
        print(" ... ... "+selseas[0])

        for stratname in ["stratiform", "stratiform_relaxed"]:
            print(" ... ... ... "+stratname)

            # Create savefolder
            savepath_seas = savepath+stratname+"/"+selseas[0]+"/"+loc+"/"
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)

            #### Get necessary variables
            MLmaxZDR = stats[stratname][loc]["values_ML_max"]["ZDR_OC"].sel(\
                                time=stats[stratname][loc]["values_ML_max"]['time'].dt.month.isin(selseas[1]))
            MLmaxKDP = stats[stratname][loc]["values_ML_max"]["KDP_ML_corrected"].sel(\
                                time=stats[stratname][loc]["values_ML_max"]['time'].dt.month.isin(selseas[1]))
            MLmaxZH = stats[stratname][loc]["values_ML_max"]["DBZH"].sel(\
                                time=stats[stratname][loc]["values_ML_max"]['time'].dt.month.isin(selseas[1]))
            MLmeanKDP = stats[stratname][loc]["values_ML_mean"]["KDP_ML_corrected"].sel(\
                                time=stats[stratname][loc]["values_ML_mean"]['time'].dt.month.isin(selseas[1]))
            ZHrain = stats[stratname][loc]["values_rain"]["DBZH"].sel(\
                                time=stats[stratname][loc]["values_rain"]['time'].dt.month.isin(selseas[1]))
            ZHsnow = stats[stratname][loc]["values_snow"]["DBZH"].sel(\
                                time=stats[stratname][loc]["values_snow"]['time'].dt.month.isin(selseas[1]))
            ZHsfc = stats[stratname][loc]["values_sfc"]["DBZH"].sel(\
                                time=stats[stratname][loc]["values_sfc"]['time'].dt.month.isin(selseas[1]))
            ZDRrain = stats[stratname][loc]["values_rain"]["ZDR_OC"].sel(\
                                time=stats[stratname][loc]["values_rain"]['time'].dt.month.isin(selseas[1]))
            ZDRsfc = stats[stratname][loc]["values_sfc"]["ZDR_OC"].sel(\
                                time=stats[stratname][loc]["values_sfc"]['time'].dt.month.isin(selseas[1]))
            deltaZH = MLmaxZH - ZHrain
            MLminRHOHV = stats[stratname][loc]["values_ML_min"]["RHOHV_NC"].sel(\
                                time=stats[stratname][loc]["values_ML_min"]['time'].dt.month.isin(selseas[1]))

            MLdepth = stats[stratname][loc]["ML_thickness"].sel(\
                                time=stats[stratname][loc]["ML_thickness"]['time'].dt.month.isin(selseas[1]))
            MLbot = stats[stratname][loc]["ML_bottom"].sel(\
                                time=stats[stratname][loc]["ML_bottom"]['time'].dt.month.isin(selseas[1]))
            betaZH = stats[stratname][loc]["beta"]["DBZH"].sel(\
                                time=stats[stratname][loc]["beta"]['time'].dt.month.isin(selseas[1]))
            cloudtop = stats[stratname][loc]["cloudtop"].sel(\
                                time=stats[stratname][loc]["cloudtop"]['time'].dt.month.isin(selseas[1]))
            cloudtop_5dbz = stats[stratname][loc]["cloudtop_5dbz"].sel(\
                                time=stats[stratname][loc]["cloudtop_5dbz"]['time'].dt.month.isin(selseas[1]))
            cloudtop_temp = stats[stratname][loc]["cloudtop_TEMP"].sel(\
                                time=stats[stratname][loc]["cloudtop_TEMP"]['time'].dt.month.isin(selseas[1]))
            cloudtop_temp_5dbz = stats[stratname][loc]["cloudtop_TEMP_5dbz"].sel(\
                                time=stats[stratname][loc]["cloudtop_TEMP_5dbz"]['time'].dt.month.isin(selseas[1]))

            #### Histograms: like Griffin et al 2020 https://doi.org/10.1175/JAMC-D-19-0128.1
            # plot histograms (2d hist) like Fig. 5
            try:
                plt.close()
                binsx = np.linspace(0.0, 4, 81)
                binsy = np.linspace(0.8, 1, 51)
                MLminRHOHVcurve = 0.97 - 0.028*binsx

                # fit our own linear regression
                idx = np.isfinite(MLmaxZDR.values) & np.isfinite(MLminRHOHV.values)
                fit = np.polyfit(MLmaxZDR.values[idx], MLminRHOHV.values[idx], 1)
                MLminRHOHVcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZDR.compute(), MLminRHOHV.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLminRHOHVcurve, c="black", label="Reference curve")
                plt.plot(binsx, MLminRHOHVcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.text(0.5, 0.82, r"$\mathregular{MLminRHOHV = 0.97 - 0.028\ MLmaxZDR }$", fontsize="xx-small")
                plt.text(0.5, 0.81, rf"$\mathregular{{MLminRHOHV_fit = {fit[1]:+.2f} {fit[0]:+.3f}\ MLmaxZDR }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_MinRHOHVinML_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + MLminRHOHV.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs MLminRHOHV due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs MLminRHOHV for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 6
            try:
                plt.close()
                binsx = np.linspace(0.0, 4, 81)
                binsy = np.linspace(-1, 3, 41)
                # MLminRHOHVcurve = 0.97 - 0.028*binsx

                utils.hist_2d(MLmaxZDR.compute(), ZDRrain.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, MLminRHOHVcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"$\mathregular{Z_{DR} \ in \ rain \ [dB]}$")
                # plt.text(0.5, 0.82, r"$\mathregular{MLminRHOHV = 0.97 - 0.028*MLmaxZDR }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZDRinRain_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + ZDRrain.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs ZDRrain due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs ZDRrain for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 7
            try:
                plt.close()
                binsx = np.linspace(-20, 50, 141)
                binsy = np.linspace(-1, 5, 61)
                # MLminRHOHVcurve = 0.97 - 0.028*binsx

                utils.hist_2d(MLmaxZH.compute(), MLmaxZDR.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, MLminRHOHVcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                # plt.text(0.5, 0.82, r"$\mathregular{MLminRHOHV = 0.97 - 0.028*MLmaxZDR }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_MaxZDRinML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + MLmaxZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs MLmaxZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZDR vs MLmaxZH for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 8
            try:
                plt.close()
                binsx = np.linspace(10, 50, 41)
                binsy = np.linspace(-4, 0, 21)
                logMLmaxKDPcurve = -3.21 + 0.05*binsx

                # scale the reference curve to C band
                lambda_s = 10 #cm
                lambda_c = 5.3 #cm
                logMLmaxKDPcurve_scaled = -3.21 + 0.05*binsx - np.log(lambda_s/lambda_c)

                # fit our own linear regression
                idx = np.isfinite(MLmaxZH.values) & np.isfinite(np.log(MLmaxKDP).values)
                fit = np.polyfit(MLmaxZH.values[idx], np.log(MLmaxKDP).values[idx], 1)
                logMLmaxKDPcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZH.compute(), np.log(MLmaxKDP).compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, logMLmaxKDPcurve, c="black", label="Reference curve")
                plt.plot(binsx, logMLmaxKDPcurve_scaled, c="darkgreen", label="Scaled reference curve")
                plt.plot(binsx, logMLmaxKDPcurve_fit, c="red", label="Fitted reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{log(Maximum \ K_{DP} \ in \ ML) \ [°/km]}$")
                plt.text(20, -3.7, r"$\mathregular{logMLmaxKDP = -3.21 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.text(20, -3.8, rf"$\mathregular{{logMLmaxKDP = {-3.21-np.log(lambda_s/lambda_c):+.2f} + 0.05\ MLmaxZH }}$", fontsize="xx-small", color="darkgreen")
                plt.text(20, -3.9, rf"$\mathregular{{logMLmaxKDP = {fit[1]:+.2f} {fit[0]:+.3f}\ MLmaxZH  }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_logMaxKDPinML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + np.log(MLmaxKDP).compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs np.log(MLmaxKDP) due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs np.log(MLmaxKDP) for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 9
            # plot a
            try:
                plt.close()
                binsx = np.linspace(-20, 50, 71)
                binsy = np.linspace(-20, 40, 61)
                ZHraincurve = -2.74 + 1.03*binsx - 0.005*binsx**2

                # fit our own linear regression
                idx = np.isfinite(MLmaxZH.values) & np.isfinite(ZHrain.values)
                fit = np.polyfit(MLmaxZH.values[idx], ZHrain.values[idx], 2)
                ZHraincurve_fit = fit[2] + fit[1]*binsx + fit[0]*binsx**2

                utils.hist_2d(MLmaxZH.compute(), ZHrain.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, ZHraincurve, c="black", label="Reference curve")
                plt.plot(binsx, ZHraincurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Z_{H} \ in \ rain \ [dBZ]}$")
                plt.text(-15, -15, r"$\mathregular{ZHrain = -2.74 + 1.03\ MLmaxZH - 0.005\ MLmaxZH^2 }$", fontsize="xx-small")
                plt.text(-15, -17, rf"$\mathregular{{ZHrain = {fit[2]:+.2f} {fit[1]:+.2f}\ MLmaxZH {fit[0]:+.3f}\ MLmaxZH^2 }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHinRain_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + ZHrain.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain for unknown reason !!! ")

            # plot b
            try:
                plt.close()
                binsx = np.linspace(-20, 50, 71)
                binsy = np.linspace(-20, 40, 61)
                ZHsnowcurve = -3.86 + 1.08*binsx - 0.0071*binsx**2

                # fit our own linear regression
                idx = np.isfinite(ZHsnow.values) & np.isfinite(ZHrain.values)
                fit = np.polyfit(ZHsnow.values[idx], ZHrain.values[idx], 2)
                ZHsnowcurve_fit = fit[2] + fit[1]*binsx + fit[0]*binsx**2

                utils.hist_2d(ZHsnow.compute(), ZHrain.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, ZHsnowcurve, c="black", label="Reference curve")
                plt.plot(binsx, ZHsnowcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Z_{H} \ in \ snow \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Z_{H} \ in \ rain \ [dBZ]}$")
                plt.text(-15, -15, r"$\mathregular{ZHrain = -3.86 + 1.08\ ZHsnow - 0.0071\ ZHsnow^2 }$", fontsize="xx-small")
                plt.text(-15, -17, rf"$\mathregular{{ZHrain = {fit[2]:+.2f} {fit[1]:+.2f}\ ZHsnow {fit[0]:+.3f}\ ZHsnow^2 }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHinRain_ZHinSnow.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZHsnow.compute().isnull() + ZHrain.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot ZHsnow vs ZHrain due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZHsnow vs ZHrain for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 10
            # plot a
            try:
                plt.close()
                binsx = np.linspace(0.8, 1, 41)
                binsy = np.linspace(-10, 20, 61)
                deltaZHcurve = 4.27 + 6.89*(1-binsx) + 341*(1-binsx)**2
                deltaZHcurve2 = -5.25 + 261.26*(1-binsx) - 974.38*(1-binsx)**2
                deltaZHcurve3 = -0.12 + 113.86*(1-binsx)

                # fit our own regressions
                idx = np.isfinite(MLminRHOHV.values) & np.isfinite(deltaZH.values)
                fit = np.polyfit((1-MLminRHOHV.values[idx]), deltaZH.values[idx], 1)
                deltaZHcurve_fit = fit[1] + fit[0]*(1-binsx)

                fit2 = np.polyfit((1-MLminRHOHV.values[idx]), deltaZH.values[idx], 2)
                deltaZHcurve_fit2 = fit2[2] + fit2[1]*(1-binsx) + fit2[0]*(1-binsx)**2

                utils.hist_2d(MLminRHOHV.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
                plt.plot(binsx, deltaZHcurve2, c="gray", label="Reference curve")
                plt.plot(binsx, deltaZHcurve3, c="gray", label="Reference curve")
                plt.plot(binsx, deltaZHcurve_fit, c="red", label="Fitted curve")
                plt.plot(binsx, deltaZHcurve_fit2, c="red", label="Fitted curve")
                plt.legend(fontsize="x-small")
                plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) \ [dBZ]´}$")
                plt.text(0.81, -1, r"$\mathregular{\Delta Z_H = 4.27 + 6.89(1-\rho _{HV}) + 341(1-\rho _{HV})^2 }$", fontsize="small")
                plt.text(0.81, -3, r"$\mathregular{\Delta Z_H = -5.25 + 261.26(1-\rho _{HV}) - 974.38(1-\rho _{HV})^2 }$", fontsize="small", color="gray")
                plt.text(0.81, -5, r"$\mathregular{\Delta Z_H = -0.12 + 113.86(1-\rho _{HV}) }$", fontsize="small", color="gray")
                plt.text(0.81, -7, rf"$\mathregular{{\Delta Z_H = {fit[1]:+.2f} {fit[0]:+.2f}\ (1-\rho _{{HV}}) }}$", fontsize="small", color="red")
                plt.text(0.81, -9, rf"$\mathregular{{\Delta Z_H = {fit2[2]:+.2f} {fit2[1]:+.2f}\ (1-\rho _{{HV}}) {fit2[0]:+.0f}\ (1-\rho _{{HV}})^2 }}$", fontsize="small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DeltaZH_MinRHOHVinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLminRHOHV.compute().isnull() + deltaZH.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLminRHOHV due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLminRHOHV for unknown reason !!! ")

            # plot b
            try:
                plt.close()
                binsx = np.linspace(0, 4, 16*5+1)
                binsy = np.linspace(-10, 20, 61)
                deltaZHcurve = 3.18 + 2.19*binsx

                # fit our own linear regression
                idx = np.isfinite(MLmaxZDR.values) & np.isfinite(deltaZH.values)
                fit = np.polyfit(MLmaxZDR.values[idx], deltaZH.values[idx], 1)
                deltaZHcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZDR.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
                plt.plot(binsx, deltaZHcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) \ [dBZ] }$")
                plt.text(2, -5, r"$\mathregular{\Delta Z_H = 3.18 + 2.19 Z_{DR} }$", fontsize="small")
                plt.text(2, -8, rf"$\mathregular{{\Delta Z_H = {fit[1]:+.2f} {fit[0]:+.2f}\ Z_{{DR}} }}$", fontsize="small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DeltaZH_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + deltaZH.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLmaxZDR due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLmaxZDR for unknown reason !!! ")

            # plot c
            try:
                plt.close()
                binsx = np.linspace(0.8, 1, 41)
                binsy = np.linspace(0, 1, 26)
                MLdepthcurve = -0.64 + 30.8*(1-binsx) - 315*(1-binsx)**2 + 1115*(1-binsx)**3

                # fit our own regression
                idx = np.isfinite(MLminRHOHV.values) & np.isfinite(MLdepth.values)
                fit = np.polyfit((1 - MLminRHOHV.values[idx]), MLdepth.values[idx]/1000, 3)
                MLdepthcurve_fit = fit[3] + fit[2]*(1-binsx) + fit[1]*(1-binsx)**2 + fit[0]*(1-binsx)**3

                utils.hist_2d(MLminRHOHV.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve")
                plt.plot(binsx, MLdepthcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.ylabel(r"Depth of ML [km]")
                plt.text(0.82, 0.2, r"$\mathregular{ML \ Depth = -0.64 + 30.8\ (1-\rho _{HV})}$" "\n" r"$\mathregular{- 315\ (1-\rho _{HV})^2 + 1115\ (1-\rho _{HV})^3}$", fontsize="xx-small")
                plt.text(0.82, 0.1, rf"$\mathregular{{ML \ Depth = {fit[3]:+.2f} {fit[2]:+.1f}\ (1-\rho _{{HV}})}}$" "\n" rf"$\mathregular{{ {fit[1]:+.0f}\ (1-\rho _{{HV}})^2 {fit[0]:+.0f}\ (1-\rho _{{HV}})^3}}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MinRHOHVinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLminRHOHV.compute().isnull() + MLdepth.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLminRHOHV due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLminRHOHV for unknown reason !!! ")

            # plot d
            try:
                plt.close()
                binsx = np.linspace(0, 4, 16*5+1)
                binsy = np.linspace(0, 1, 26)
                MLdepthcurve = 0.21 + 0.091*binsx

                # fit our own regression
                idx = np.isfinite(MLmaxZDR.values) & np.isfinite(MLdepth.values)
                fit = np.polyfit(MLmaxZDR.values[idx], MLdepth.values[idx]/1000, 1)
                MLdepthcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZDR.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve")
                plt.plot(binsx, MLdepthcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"Depth of ML [km]")
                plt.text(2.1, 0.15, r"$\mathregular{ML \ Depth = 0.21 + 0.091 Z_{DR} }$", fontsize="xx-small")
                plt.text(2.1, 0.05, rf"$\mathregular{{ML \ Depth = {fit[1]:+.2f} {fit[0]:+.3f} Z_{{DR}} }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + MLdepth.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLmaxZDR due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLmaxZDR for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 11
            try:
                plt.close()
                binsx = np.linspace(-10, 50, 31)
                binsy = np.linspace(-0, 1.2, 21)
                MLdepthcurve = 0.315 + 0.000854*binsx
                MLdepthcurve2 = -0.041 + 0.009133*binsx
                MLdepthcurve3 = 0.176 + 0.006098*binsx

                # fit our own regression
                idx = np.isfinite(MLmaxZH.values) & np.isfinite(MLdepth.values)
                fit = np.polyfit(MLmaxZH.values[idx], MLdepth.values[idx]/1000, 1)
                MLdepthcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZH.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve")
                plt.plot(binsx, MLdepthcurve2, c="gray", label="Reference curve")
                plt.plot(binsx, MLdepthcurve3, c="lightgray", label="Reference curve")
                plt.plot(binsx, MLdepthcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Depth\ of\ ML\ [km]}$")
                plt.text(11, 0.20, r"$\mathregular{MLdepth = 0.315 + 0.000854\ MLmaxZH }$", fontsize="xx-small")
                plt.text(11, 0.15, r"$\mathregular{MLdepth = -0.041 + 0.009133\ MLmaxZH }$", fontsize="xx-small", color="gray")
                plt.text(11, 0.10, r"$\mathregular{MLdepth = 0.176 + 0.006098\ MLmaxZH }$", fontsize="xx-small", color="lightgray")
                plt.text(11, 0.05, rf"$\mathregular{{MLdepth = {fit[1]:+.3f} {fit[0]:+.6f}\ MLmaxZH }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + ZHrain.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain for unknown reason !!! ")

            #### Histograms: like Troemel et al 2019 https://doi.org/10.1175/JAMC-D-19-0056.1
            # plot histograms (2d hist) like Fig. 7
            # plot only bottom left (the rest are the same as plots 8 and 9 from Griffin)
            try:
                plt.close()
                binsx = np.linspace(10, 50, 41)
                binsy = np.linspace(-4, 0, 21)
                logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # scale the reference curve to C band
                lambda_x = 3 #cm
                lambda_c = 5.3 #cm
                logMLmeanKDPcurve_scaled = -2.4 + 0.05*binsx - np.log(lambda_x/lambda_c)

                # fit our own linear regression
                idx = np.isfinite(MLmaxZH.values) & np.isfinite(np.log(MLmeanKDP).values)
                fit = np.polyfit(MLmaxZH.values[idx], np.log(MLmeanKDP).values[idx], 1)
                logMLmeanKDPcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZH.compute(), np.log(MLmeanKDP).compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                plt.plot(binsx, logMLmeanKDPcurve_scaled, c="darkgreen", label="Scaled reference curve")
                plt.plot(binsx, logMLmeanKDPcurve_fit, c="red", label="Fitted reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{log(Mean \ K_{DP} \ in \ ML) \ [°/km]}$")
                plt.text(12, -0.2, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.text(12, -0.5, rf"$\mathregular{{logMLmeanKDP = {-2.4-np.log(lambda_x/lambda_c):+.1f} + 0.05\ MLmaxZH }}$", fontsize="xx-small", color="darkgreen")
                plt.text(12, -0.8, rf"$\mathregular{{logMLmeanKDP = {fit[1]:+.1f} {fit[0]:+.2f}\ MLmaxZH  }}$", fontsize="xx-small", color="red")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_logMeanKDPinML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + np.log(MLmeanKDP).compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs np.log(MLmeanKDP) due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs np.log(MLmeanKDP) for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 9
            # plot left one
            try:
                plt.close()
                binsx = np.linspace(-0.5, 2, 51)
                binsy = np.linspace(-1, 2, 31)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(ZDRrain.values) & np.isfinite(ZDRsfc.values)
                fit = np.polyfit(ZDRrain.values[idx], ZDRsfc.values[idx], 1)
                ZDRatSfccurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(ZDRrain.compute(), ZDRsfc.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                plt.plot(binsx, ZDRatSfccurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{DR} \ in \ rain \ [dB] }$")
                plt.ylabel(r"$\mathregular{ Z_{DR} \ at \ surface \ [dB] }$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.text(0.5, -0.5, rf"$\mathregular{{ZDRatSfc = {fit[1]:+.2f} {fit[0]:+.2f}\ ZDRrain }}$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZDRatSfc_ZDRinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZDRrain.compute().isnull() + ZDRsfc.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot ZDRrain vs ZDRsfc due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZDRrain vs ZDRsfc for unknown reason !!! ")

            # plot right one
            try:
                plt.close()
                binsx = np.linspace(0, 40, 41)
                binsy = np.linspace(0, 40, 41)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(ZHrain.values) & np.isfinite(ZHsfc.values)
                fit = np.polyfit(ZHrain.values[idx], ZHsfc.values[idx], 1)
                ZHatSfccurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(ZHrain.compute(), ZHsfc.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                plt.plot(binsx, ZHatSfccurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{H} \ in \ rain \ [dBZ] }$")
                plt.ylabel(r"$\mathregular{ Z_{H} \ at \ surface \ [dBZ] }$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.text(25, 5, rf"$\mathregular{{ZHatSfc = {fit[1]:+.2f} {fit[0]:+.2f}\ ZHrain }}$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHatSfc_ZHinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZHrain.compute().isnull() + ZHsfc.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot ZHrain vs ZHsfc due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZHrain vs ZHsfc for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 10
            # plot only top left (the rest are already plotted from previous plots)
            try:
                plt.close()
                binsx = np.linspace(0, 50, 26)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(MLmaxZH.values) & np.isfinite(betaZH.values)
                fit = np.polyfit(MLmaxZH.values[idx], betaZH.values[idx], 1)
                betaZHcurve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(MLmaxZH.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                plt.plot(binsx, betaZHcurve_fit, c="red", label="Fitted curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{ Maximum \ Z_{H} \ in \ ML \ [dBZ] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.text(10, -12, rf"$\mathregular{{betaZH = {fit[1]:+.2f} {fit[0]:+.2f}\ MLmaxZH }}$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + betaZH.compute().isnull() ).all() or idx.sum()<2:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs betaZH for unknown reason !!! ")

            #### Histograms: custom

            # Plot beta vs ZHrain
            try:
                plt.close()
                binsx = np.linspace(0, 50, 26)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                utils.hist_2d(ZHrain.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{H} \ in \ rain \ [dBZ] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_ZHinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZHrain.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot ZHrain vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZHrain vs betaZH for unknown reason !!! ")

            # Plot beta vs ZDRrain
            try:
                plt.close()
                binsx = np.linspace(-0.5, 2, 51)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                utils.hist_2d(ZDRrain.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{DR} \ in \ rain \ [dB] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_ZDRinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZDRrain.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot ZDRrain vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZDRrain vs betaZH for unknown reason !!! ")

            # Plot beta vs cloudtop height
            try:
                plt.close()
                binsx = np.linspace(5, 15, 51)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(cloudtop.values/1000) & np.isfinite(betaZH.values)
                fit = np.polyfit(cloudtop.values[idx]/1000, betaZH.values[idx], 1)
                beta_cloudtop_curve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(cloudtop.compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, beta_cloudtop_curve_fit, c="red", label="Fitted curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ height \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                plt.text(9, -14, rf"$\mathregular{{betaZH = {fit[1]:+.2f} {fit[0]:+.2f}\ CloudTopHeight }}$", fontsize="x-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_CloudTopHeight.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( cloudtop.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot cloudtop vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot cloudtop vs betaZH for unknown reason !!! ")


            # Plot beta vs cloudtop height (5 dBZ)
            try:
                plt.close()
                binsx = np.linspace(5, 15, 51)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(cloudtop_5dbz.values/1000) & np.isfinite(betaZH.values)
                fit = np.polyfit(cloudtop_5dbz.values[idx]/1000, betaZH.values[idx], 1)
                beta_cloudtop_curve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(cloudtop_5dbz.compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, beta_cloudtop_curve_fit, c="red", label="Fitted curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ height \ (5 dbZ) \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                plt.text(9, -14, rf"$\mathregular{{betaZH = {fit[1]:+.2f} {fit[0]:+.2f}\ CloudTopHeight5dBZ }}$", fontsize="x-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_CloudTopHeight5dBZ.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( cloudtop_5dbz.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot cloudtop_5dbz vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot cloudtop_5dbz vs betaZH for unknown reason !!! ")

            # Plot beta vs cloudtop height TEMP
            try:
                plt.close()
                binsx = np.linspace(-40, -10, 31)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(cloudtop_temp.values) & np.isfinite(betaZH.values)
                fit = np.polyfit(cloudtop_temp.values[idx], betaZH.values[idx], 1)
                beta_cloudtop_curve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(cloudtop_temp.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, beta_cloudtop_curve_fit, c="red", label="Fitted curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ temperature \ [C] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                plt.text(-25, -14, rf"$\mathregular{{betaZH = {fit[1]:+.2f} {fit[0]:+.2f}\ CloudTopTEMP }}$", fontsize="x-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_CloudTopTEMP.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( cloudtop_temp.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot cloudtop_temp vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot cloudtop_temp vs betaZH for unknown reason !!! ")


            # Plot beta vs cloudtop height TEMP (5 dBZ)
            try:
                plt.close()
                binsx = np.linspace(-40, -10, 31)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                # fit our own regression
                idx = np.isfinite(cloudtop_temp_5dbz.values) & np.isfinite(betaZH.values)
                fit = np.polyfit(cloudtop_temp_5dbz.values[idx], betaZH.values[idx], 1)
                beta_cloudtop_curve_fit = fit[1] + fit[0]*binsx

                utils.hist_2d(cloudtop_temp_5dbz.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, beta_cloudtop_curve_fit, c="red", label="Fitted curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ temperature \ (5 dbZ) \ [C] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                plt.text(-25, -14, rf"$\mathregular{{betaZH = {fit[1]:+.2f} {fit[0]:+.2f}\ CloudTopTEMP5dBZ }}$", fontsize="x-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_CloudTopTEMP5dBZ.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( cloudtop_temp.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot cloudtop_temp vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot cloudtop_temp vs betaZH for unknown reason !!! ")


            # Plot beta vs ML depth
            try:
                plt.close()
                binsx = np.linspace(0, 1, 26)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                utils.hist_2d(MLdepth.compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Depth \ of \ ML \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_DepthML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLdepth.compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLdepth vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLdepth vs betaZH for unknown reason !!! ")

            # Plot beta vs ML height (top)
            try:
                plt.close()
                binsx = np.linspace(1.5, 4.5, 26)
                binsy = np.linspace(-15, 10, 26)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx

                utils.hist_2d((MLdepth+MLbot).compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ ML \ top \ height \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dBZ/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_MLtopheight.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( (MLdepth+MLbot).compute().isnull() + betaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot (MLdepth+MLbot) vs betaZH due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot (MLdepth+MLbot) vs betaZH for unknown reason !!! ")

#%%% ridgeplots
savepath = "/automount/agradar/jgiles/images/stats_ridgeplots/"

vars_ticks = {X_DBZH: np.arange(0, 46, 1),
                X_ZDR: np.arange(-0.5, 2.1, 0.1),
                X_KDP: np.arange(-0.1, 0.52, 0.02),
                X_RHO: np.arange(0.9, 1.004, 0.004)
                }

beta_vars_ticks = {X_DBZH: np.linspace(-15, 10, int((10--15)/1)+1 ),
                X_ZDR: np.linspace(-0.5, 1, int((1--0.5)/0.1)+1 ),
                X_KDP: np.linspace(-0.2, 0.2, int((0.2--0.2)/0.01)+1 ),
                X_RHO: np.linspace(-0.05, 0.05, int((0.05--0.05)/0.001)+1 ),
                } # the "_polyfit_coefficients" in the var names will be added below

ridge_vars = set(list(vars_ticks.keys())+list(beta_vars_ticks.keys()))

bins = {
        "ML_thickness": np.arange(0,1200,50),
        "ML_thickness_TEMP": np.arange(0, 8.5, 0.5),
        "ML_bottom": np.arange(0,4100,100),
        "ML_bottom_TEMP": np.arange(0, 9.5, 0.5),
        "values_snow": vars_ticks,
        "values_rain": vars_ticks,
        "values_DGL_mean": vars_ticks,
        "values_DGL_min": vars_ticks,
        "values_DGL_max": vars_ticks,
        "values_NZ_mean": vars_ticks,
        "values_NZ_min": vars_ticks,
        "values_NZ_max": vars_ticks,
        "values_ML_mean": vars_ticks,
        "values_ML_min": vars_ticks,
        "values_ML_max": vars_ticks,
        "values_sfc": vars_ticks,
        "cloudtop": np.arange(2000,12250,250),
        "cloudtop_5dbz": np.arange(2000,12250,250),
        "cloudtop_10dbz": np.arange(2000,12250,250),
        "beta": beta_vars_ticks,
        "beta_belowML": beta_vars_ticks,
        "cloudtop_TEMP": np.arange(-50,5,1),
        "cloudtop_TEMP_5dbz": np.arange(-50,5,1),
        "cloudtop_TEMP_10dbz": np.arange(-50,5,1),
        "deltaZH": np.arange(-5,21,1),
        "delta_z_ZHmaxML_RHOHVminML": np.arange(0,440, 40),
        }

# set a dictionary of bandwidths, this is important for the cases where the low resolution of the
# data generates a histogram with only a few intervals with data. "normal_reference" is the default
default_bandwidth_dict = {vv:"normal_reference" for vv in vars_ticks.keys()}
default_bandwidth = "normal_reference"

bandwidths = {"ML_thickness": 50,
        "ML_thickness_TEMP": default_bandwidth,
        "ML_bottom": default_bandwidth,
        "ML_bottom_TEMP": default_bandwidth,
        "values_snow": default_bandwidth_dict,
        "values_rain": default_bandwidth_dict,
        "values_DGL_mean": default_bandwidth_dict,
        "values_DGL_min": default_bandwidth_dict,
        "values_DGL_max": default_bandwidth_dict,
        "values_NZ_mean": default_bandwidth_dict,
        "values_NZ_min": default_bandwidth_dict,
        "values_NZ_max": default_bandwidth_dict,
        "values_ML_mean": default_bandwidth_dict,
        "values_ML_min": default_bandwidth_dict,
        "values_ML_max": default_bandwidth_dict,
        "values_sfc": default_bandwidth_dict,
        "cloudtop": default_bandwidth,
        "cloudtop_5dbz": default_bandwidth,
        "cloudtop_10dbz": default_bandwidth,
        "beta": default_bandwidth_dict,
        "beta_belowML": default_bandwidth_dict,
        "cloudtop_TEMP": default_bandwidth,
        "cloudtop_TEMP_5dbz": default_bandwidth,
        "cloudtop_TEMP_10dbz": default_bandwidth,
        "deltaZH": default_bandwidth,
        "delta_z_ZHmaxML_RHOHVminML": default_bandwidth,
        }
# Particular changes
bandwidths['values_sfc']['KDP_ML_corrected'] = 0.01
bandwidths['values_sfc']['RHOHV_NC'] = 0.01
bandwidths['values_snow']['RHOHV_NC'] = 0.01


order = ['tur', 'umd', 'pro', 'afy', 'ank', 'gzt', 'hty', 'svs']
# order = ['umd', 'pro', 'hty']

selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])


# Define a function to reorder the elements of the ridgeplot
def reorder_tuple_elements(data, n):
    # Extract the last n elements
    last_n = data[-n:]

    # Convert the tuple to a list to facilitate reordering
    data_list = list(data[:-n])

    # Insert each of the last n elements at the desired positions
    for i in range(n):
        target_position = i * 3 + 2
        data_list.insert(target_position, last_n[i])

    # Convert back to a tuple and return
    return tuple(data_list)

# Define a function to change the alpha value of an rgba string
def change_rgba_alpha(original_color, new_alpha):
    r, g, b, alpha = original_color.lstrip('rgba(').rstrip(')').split(',')
    return f'rgba({r}, {g}, {b}, {new_alpha})'

# Build deltaZH into stats
for stratname in ["stratiform", "stratiform_relaxed"]:
    for ll in order:
        if ll in stats[stratname].keys():
            stats[stratname][ll]["deltaZH"] = stats[stratname][ll]["values_ML_max"][X_DBZH] - stats[stratname][ll]["values_rain"][X_DBZH]

# Build delta_z_ZHmaxML_RHOHVminML into stats
for stratname in ["stratiform", "stratiform_relaxed"]:
    for ll in order:
        if ll in stats[stratname].keys():
            stats[stratname][ll]["delta_z_ZHmaxML_RHOHVminML"] = stats[stratname][ll]["height_ML_max"][X_DBZH] - stats[stratname][ll]["height_ML_min"][X_RHO]

# Plot stats ridgeplots
for selseas in selseaslist:
    print(" ... ... "+selseas[0])
    for stratname in ["stratiform", "stratiform_relaxed"]:

        print("plotting "+stratname+" stats...")

        # Create savefolder
        savepath_seas = savepath+stratname+"/"+selseas[0]+"/"
        if not os.path.exists(savepath_seas):
            os.makedirs(savepath_seas)

        order_fil = [ll for ll in order if ll in stats[stratname].keys()]

        for ss in bins.keys():
            print("...plotting "+ss)
            try:
                for vv in ridge_vars:

                    # Get the samples for each radar and filter out the radars that have zero samples.
                    samples = {loc: stats[stratname][loc][ss][vv].sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    if ss in ["beta"] and vv in ["RHOHV_NC", "RHOHV"]: # filter out unrealistic zero beta values
                        samples = {loc: samples[loc][abs(samples[loc])>0.0001] for loc in samples.keys()}

                    if ss in ["values_DGL_min", "values_ML_min", "values_rain", "values_sfc"] and vv in ["KDP_ML_corrected"]: # filter out unrealistic zero values
                        samples = {loc: samples[loc][abs(samples[loc])>0.001] for loc in samples.keys()}

                    samples = {loc.swapcase(): samples[loc] for loc in samples.keys() if len(samples[loc])>10} # filter out radars with less than 10 samples

                    fig = ridgeplot.ridgeplot(samples=samples.values(),
                                            colorscale="viridis",
                                            colormode="row-index",
                                            coloralpha=0.65,
                                            labels=samples.keys(),
                                            linewidth=2,
                                            spacing=5 / 9,
                                            # kde_points=bins[ss],
                                            bandwidth=bandwidths[ss][vv],
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=20,
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    title=ss+" "+vv,
                                    xaxis_tickvals=bins[ss][vv],
                    )

                    # Add vertical zero line
                    fig.add_vline(x=0, line_width=2, line_color="gray")

                    # Get densities data from the plot
                    densities = [ fig.data[2*i+1] for i in range(len(samples)) ]

                    # calculate means or median
                    means = [np.median(sample) for sample in samples.values()]

                    # Add a vertical line at the mean for each distribution
                    for i, mean in enumerate(means):
                        # define the bottom and top of each distribution
                        y_bot = np.array(densities[i]["y"]).min()
                        y_top = np.array(densities[i]["y"])[(np.where(np.array(densities[i]["x"]) >= mean))][0]

                        fig.add_scatter(
                            mode="lines",
                            x=[mean, mean],  # Vertical line at the mean
                            y=[y_bot , y_top],  # Set y0 and y1 based on the vertical offset
                            line=dict(color=change_rgba_alpha(densities[i]["fillcolor"], 1), width=2),
                        )

                    # We need to reorder the elements of the fig.data tuple so that
                    # the mean lines go below each distribution.
                    fig.data = reorder_tuple_elements(fig.data,len(means))

                    # save figure
                    fig.write_html(savepath_seas+"/"+ss+"_"+vv+".html")

            except KeyError:
                try:
                    samples = {loc: stats[stratname][loc][ss].sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    if ss in ["cloudtop", "cloudtop_5dbz", "cloudtop_10dbz"]: # filter out erroneous cloudtop values #!!! this will be fixed now (19.03.25) and this extra filter will not be necessary after re running the stats calculations
                        samples = {loc: samples[loc][samples[loc]<np.max(samples[loc])] for loc in samples.keys()}
                    if ss in ["cloudtop_TEMP", "cloudtop_TEMP_5dbz", "cloudtop_TEMP_10dbz"]: # filter out erroneous cloudtop values #!!! this will be fixed now (19.03.25) and this extra filter will not be necessary after re running the stats calculations
                        samples = {loc: stats[stratname][loc][ss].where(stats[stratname][loc]["".join(ss.split("_TEMP"))].sel(\
                                    time=stats[stratname][loc]["".join(ss.split("_TEMP"))]['time'].dt.month.isin(selseas[1])) <
                                                                            stats[stratname][loc]["".join(ss.split("_TEMP"))].max().values).dropna("time").values\
                                   for loc in order_fil}
                        # samples = {loc: samples[loc][samples_aux[loc]<np.max(samples_aux[loc])] for loc in samples.keys()}

                    samples = {loc.swapcase(): samples[loc] for loc in samples.keys() if len(samples[loc])>10} # filter out radars with no samples

                    fig = ridgeplot.ridgeplot(samples=samples.values(),
                                            colorscale="viridis",
                                            colormode="row-index",
                                            coloralpha=0.65,
                                            labels=samples.keys(),
                                            linewidth=2,
                                            spacing=5 / 9,
                                            # kde_points=bins[ss],
                                            bandwidth=bandwidths[ss],
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=20,
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    title=ss,
                                    xaxis_tickvals=bins[ss],
                    )

                    # Add vertical zero line
                    fig.add_vline(x=0, line_width=2, line_color="gray")

                    # Get densities data from the plot
                    densities = [ fig.data[2*i+1] for i in range(len(samples)) ]

                    # calculate means
                    means = [np.median(sample) for sample in samples.values()]

                    # Add a vertical line at the mean for each distribution
                    for i, mean in enumerate(means):
                        # define the bottom and top of each distribution
                        y_bot = np.array(densities[i]["y"]).min()
                        y_top = np.array(densities[i]["y"])[(np.where(np.array(densities[i]["x"]) >= mean))][0]

                        fig.add_scatter(
                            mode="lines",
                            x=[mean, mean],  # Vertical line at the mean
                            y=[y_bot , y_top],  # Set y0 and y1 based on the vertical offset
                            line=dict(color=change_rgba_alpha(densities[i]["fillcolor"], 1), width=2),
                        )

                    # We need to reorder the elements of the fig.data tuple so that
                    # the mean lines go below each distribution.
                    fig.data = reorder_tuple_elements(fig.data,len(means))

                    fig.write_html(savepath_seas+"/"+ss+".html")
                except:
                    print("!!! unable to plot "+stratname+" "+ss+" !!!")
            except:
                print("!!! unable to plot "+stratname+" "+ss+" !!!")

#%%% Custom plots

#%%%% Plot beta seasonality
stratname = "stratiform"


colors = ["#4c72b0",  # Deep Blue
            "#D73027",  # Bright Red
              "#E66101",  # Reddish Orange
              "#B2182B",  # Deep Crimson
              "#67001F",  # Dark Burgundy
              "#F46D43"]  # Warm Coral

colors = ["#4c72b0",  # Deep Blue
            "#000004",  # Basically black
              "#5c126e",  # Purple
              "#9b2964",  # Dark Magenta
              "#e55c30",  # Orange
              "#fbba1f"]  # Light Orange

line_styles = ["--",  # Dashed
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               "--",  # Dashed
               "-.",  # Dash-dot
               ":",  # Dotted
               (0, (3, 5, 1, 5))]  # Custom: long dash, short gap, dot, short gap

for il, loc in enumerate(locs):
    count = stats[stratname][loc]['beta']['DBZH'].groupby("time.month").count()
    stats[stratname][loc]['beta']['DBZH'].groupby("time.month").median().where(count>30).plot(
        label=loc.swapcase(), c=colors[il], ls=line_styles[il], lw=2,alpha=0.8)
plt.ylabel(r'$\beta$ [dBZ/km]')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
plt.grid(visible=True)
plt.legend()
plt.title(r'$\beta$ seasonality')

#%%%% Plot riming frequency all radars in same plot

locs_to_plot = locs #[find_loc(locs, ff[0])] # by default, plot only the histograms of the currently loaded QVPs.
# savepath = "/automount/agradar/jgiles/images/stats_histograms/"

selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

riming_class_to_plot = [
                        # 'riming_DR',
                        # 'riming_UDR',
                        # 'riming_ZDR_DBZH',
                        'riming_ZDR_OC_DBZH',
           ]

colors = ["#4c72b0",  # Deep Blue
            "#000004",  # Basically black
              "#5c126e",  # Purple
              "#9b2964",  # Dark Magenta
              "#e55c30",  # Orange
              "#fbba1f"]  # Light Orange

line_styles = ["--",  # Dashed
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               "-",  # Solid
               ]
print("Plotting riming histograms ...")


for stratname in ["stratiform", "stratiform_relaxed"]:
    print(" ... ... "+stratname)
    for vv in riming_class_to_plot:
        for selseas in selseaslist:
            print(" ... ... ... "+selseas[0])
            # Plot the percentage against temperature
            fig = plt.figure(figsize=(5, 4))

            for iloc, loc in enumerate(locs_to_plot):
                print(" ... "+loc)

                to_plot = riming_classif[stratname][loc].chunk({"time":"auto"}).where(\
                                                                   riming_classif[stratname][loc].z >= riming_classif[stratname][loc].height_ml_new_gia,
                                                                drop=True)


                # # Create savefolder
                # savepath_seas = savepath+stratname+"/"+selseas[0]+"/"+loc+"/"
                # if not os.path.exists(savepath_seas):
                #     os.makedirs(savepath_seas)

                to_plot_sel = to_plot.sel(\
                                    time=to_plot['time'].dt.month.isin(selseas[1]))


                try:

                    # Create temperature bins (1-degree intervals)
                    temp_bins = np.arange(-20, 1)

                    # Create an empty list to store the values
                    percentages = []
                    count = []

                    # Loop through each temperature bin
                    for i in range(len(temp_bins) - 1):
                        # Mask for the current temperature bin
                        temp_mask = (to_plot_sel.TEMP >= temp_bins[i]) & (to_plot_sel.TEMP < temp_bins[i+1])

                        # Get the data corresponding to the current temperature bin
                        data_in_bin = to_plot_sel[vv].where(temp_mask.compute(), drop=True)

                        # Calculate the percentage of 1s (ignoring NaNs)
                        total_values = np.isfinite(data_in_bin).sum()  # Total number of finite values (non-nan)
                        ones_count = (data_in_bin == 1).sum()          # Count of values equal to 1
                        percentage = (ones_count / total_values) * 100 if total_values > 0 else np.nan

                        # Append the percentage to the list
                        percentages.append(percentage.values)

                        # Append the total_values to the list
                        count.append(total_values.values)

                    plt.step(percentages, temp_bins[:-1], where="post",
                             label=loc.swapcase(), c=colors[iloc], lw=2, ls=line_styles[iloc])

                    # fig.savefig(savepath_seas+"/"+loc+"_"+vv+"_vsTEMP.png",
                    #                 bbox_inches="tight")
                    # plt.close(fig)


                except:
                    print("!!! Unable to plot "+vv+" !!!")

            plt.xlabel('Percentage of rimed events [%]')
            plt.ylabel('Temperature [°C]')
            plt.title('Percentage of '+vv+" "+stratname+" "+selseas[0])
            plt.xlim(0, 50)
            plt.legend()
            plt.gca().yaxis.set_inverted(True)
            plt.grid(True)
            plt.show()

#%%%% Ridgeplot of of variables in rimed vs not rimed events


savepath = "/automount/agradar/jgiles/images/stats_ridgeplots_riming/"

vars_ticks = {X_DBZH: np.arange(0, 46, 1),
                X_ZDR: np.arange(-0.5, 2.1, 0.1),
                X_KDP: np.arange(-0.1, 0.52, 0.02),
                X_RHO: np.arange(0.9, 1.004, 0.004)
                }

beta_vars_ticks = {X_DBZH: np.linspace(-15, 10, int((10--15)/1)+1 ),
                X_ZDR: np.linspace(-0.5, 1, int((1--0.5)/0.1)+1 ),
                X_KDP: np.linspace(-0.2, 0.2, int((0.2--0.2)/0.01)+1 ),
                X_RHO: np.linspace(-0.05, 0.05, int((0.05--0.05)/0.001)+1 ),
                } # the "_polyfit_coefficients" in the var names will be added below

ridge_vars = set(list(vars_ticks.keys())+list(beta_vars_ticks.keys()))

bins = {
        "ML_thickness": np.arange(0,1200,50),
        "ML_thickness_TEMP": np.arange(0, 8.5, 0.5),
        "ML_bottom": np.arange(0,4100,100),
        "ML_bottom_TEMP": np.arange(0, 9.5, 0.5),
        "values_snow": vars_ticks,
        "values_rain": vars_ticks,
        "values_DGL_mean": vars_ticks,
        "values_DGL_min": vars_ticks,
        "values_DGL_max": vars_ticks,
        "values_NZ_mean": vars_ticks,
        "values_NZ_min": vars_ticks,
        "values_NZ_max": vars_ticks,
        "values_ML_mean": vars_ticks,
        "values_ML_min": vars_ticks,
        "values_ML_max": vars_ticks,
        "values_sfc": vars_ticks,
        "cloudtop": np.arange(2000,12250,250),
        "cloudtop_5dbz": np.arange(2000,12250,250),
        "cloudtop_10dbz": np.arange(2000,12250,250),
        "beta": beta_vars_ticks,
        "beta_belowML": beta_vars_ticks,
        "cloudtop_TEMP": np.arange(-50,5,1),
        "cloudtop_TEMP_5dbz": np.arange(-50,5,1),
        "cloudtop_TEMP_10dbz": np.arange(-50,5,1),
        "deltaZH": np.arange(-5,21,1),
        "delta_z_ZHmaxML_RHOHVminML": np.arange(0,440, 40),
        }

# set a dictionary of bandwidths, this is important for the cases where the low resolution of the
# data generates a histogram with only a few intervals with data. "normal_reference" is the default
default_bandwidth_dict = {vv:"normal_reference" for vv in vars_ticks.keys()}
default_bandwidth = "normal_reference"

bandwidths = {"ML_thickness": 50,
        "ML_thickness_TEMP": default_bandwidth,
        "ML_bottom": default_bandwidth,
        "ML_bottom_TEMP": default_bandwidth,
        "values_snow": default_bandwidth_dict,
        "values_rain": default_bandwidth_dict,
        "values_DGL_mean": default_bandwidth_dict,
        "values_DGL_min": default_bandwidth_dict,
        "values_DGL_max": default_bandwidth_dict,
        "values_NZ_mean": default_bandwidth_dict,
        "values_NZ_min": default_bandwidth_dict,
        "values_NZ_max": default_bandwidth_dict,
        "values_ML_mean": default_bandwidth_dict,
        "values_ML_min": default_bandwidth_dict,
        "values_ML_max": default_bandwidth_dict,
        "values_sfc": default_bandwidth_dict,
        "cloudtop": default_bandwidth,
        "cloudtop_5dbz": default_bandwidth,
        "cloudtop_10dbz": default_bandwidth,
        "beta": default_bandwidth_dict,
        "beta_belowML": default_bandwidth_dict,
        "cloudtop_TEMP": default_bandwidth,
        "cloudtop_TEMP_5dbz": default_bandwidth,
        "cloudtop_TEMP_10dbz": default_bandwidth,
        "deltaZH": default_bandwidth,
        "delta_z_ZHmaxML_RHOHVminML": default_bandwidth,
        }
# Particular changes
bandwidths['values_sfc']['KDP_ML_corrected'] = 0.01
bandwidths['values_sfc']['RHOHV_NC'] = 0.01
bandwidths['values_snow']['RHOHV_NC'] = 0.01


order = ['tur', 'umd', 'pro', 'afy', 'ank', 'gzt', 'hty', 'svs']
# order = ['umd', 'pro', 'hty']

selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])


# Define a function to reorder the elements of the ridgeplot
def reorder_tuple_elements(data, n):
    # Extract the last n elements
    last_n = data[-n:]

    # Convert the tuple to a list to facilitate reordering
    data_list = list(data[:-n])

    # Insert each of the last n elements at the desired positions
    for i in range(n):
        target_position = i * 3 + 2
        data_list.insert(target_position, last_n[i])

    # Convert back to a tuple and return
    return tuple(data_list)

# Define a function to change the alpha value of an rgba string
def change_rgba_alpha(original_color, new_alpha):
    r, g, b, alpha = original_color.lstrip('rgba(').rstrip(')').split(',')
    return f'rgba({r}, {g}, {b}, {new_alpha})'


# Plot stats ridgeplots
for selseas in selseaslist:
    print(" ... ... "+selseas[0])
    for stratname in ["stratiform", "stratiform_relaxed"]:

        riming_class = "riming_ZDR_OC_DBZH"

        print("plotting "+stratname+" stats...")

        # Create savefolder
        savepath_seas = savepath+stratname+"/"+riming_class+"/"+selseas[0]+"/"
        if not os.path.exists(savepath_seas):
            os.makedirs(savepath_seas)

        order_fil = [ll for ll in order if ll in stats[stratname].keys()]

        for ss in bins.keys():
            print("...plotting "+ss)
            try:
                for vv in ridge_vars:

                    # Get the samples for each radar and filter out the radars that have zero samples.
                    riming_filter = { loc: riming_classif[stratname][loc].chunk({"time":"auto"}).where(\
                                                                       riming_classif[stratname][loc].z >= riming_classif[stratname][loc].height_ml_new_gia,
                                                                    ).where(\
                                                                       riming_classif[stratname][loc].z <= riming_classif[stratname][loc].height_ml_new_gia + 2000,
                                                                    ) for loc in order_fil}
                    samples_wriming = {loc: stats[stratname][loc][ss][vv].where(riming_filter[loc][riming_class].sum("z")>0).sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    samples_woriming = {loc: stats[stratname][loc][ss][vv].where(riming_filter[loc][riming_class].sum("z")==0).sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    if ss in ["beta"] and vv in ["RHOHV_NC", "RHOHV"]: # filter out unrealistic zero beta values
                        samples_wriming = {loc: samples_wriming[loc][abs(samples_wriming[loc])>0.0001] for loc in samples_wriming.keys()}
                        samples_woriming = {loc: samples_woriming[loc][abs(samples_woriming[loc])>0.0001] for loc in samples_woriming.keys()}

                    if ss in ["values_DGL_min", "values_ML_min", "values_rain", "values_sfc"] and vv in ["KDP_ML_corrected"]: # filter out unrealistic zero values
                        samples_wriming = {loc: samples_wriming[loc][abs(samples_wriming[loc])>0.001] for loc in samples_wriming.keys()}
                        samples_woriming = {loc: samples_woriming[loc][abs(samples_woriming[loc])>0.001] for loc in samples_woriming.keys()}

                    samples = {loc.swapcase(): [ samples_wriming[loc], samples_woriming[loc] ] for loc in samples_wriming.keys() if ( len(samples_wriming[loc])>10 and len(samples_woriming[loc])>10 )} # filter out radars with no samples

                    fig = ridgeplot.ridgeplot(samples=samples.values(),
                                            colorscale="viridis",
                                            # colormode="row-index",
                                            colormode="trace-index-row-wise",
                                            # colormode="trace-index",
                                            coloralpha=0.65,
                                            labels=samples.keys(),
                                            linewidth=2,
                                            spacing=5 / 9,
                                            # kde_points=bins[ss],
                                            bandwidth=bandwidths[ss][vv],
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=20,
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    title=ss,
                                    xaxis_tickvals=bins[ss][vv],
                    )

                    # Add vertical zero line
                    fig.add_vline(x=0, line_width=2, line_color="gray")

                    # Get densities data from the plot
                    densities = [ fig.data[2*i+1] for i in range(len(samples)*2) ]

                    # calculate means
                    means = []
                    for sample0 in samples.values():
                        for sample in sample0:
                            means = np.append(means, np.median(sample) )

                    # Add a vertical line at the mean for each distribution
                    for i, mean in enumerate(means):
                        # define the bottom and top of each distribution
                        y_bot = np.array(densities[i]["y"]).min()
                        y_top = np.array(densities[i]["y"])[(np.where(np.array(densities[i]["x"]) >= mean))][0]

                        fig.add_scatter(
                            mode="lines",
                            x=[mean, mean],  # Vertical line at the mean
                            y=[y_bot , y_top],  # Set y0 and y1 based on the vertical offset
                            line=dict(color=change_rgba_alpha(densities[i]["fillcolor"], 1), width=2),
                        )

                    # We need to reorder the elements of the fig.data tuple so that
                    # the mean lines go below each distribution.
                    fig.data = reorder_tuple_elements(fig.data,len(means))

                    fig.write_html(savepath_seas+"/"+ss+"_"+vv+".html")

            except KeyError:
                try:
                    riming_filter = { loc: riming_classif[stratname][loc].chunk({"time":"auto"}).where(\
                                                                       riming_classif[stratname][loc].z >= riming_classif[stratname][loc].height_ml_new_gia,
                                                                    ).where(\
                                                                       riming_classif[stratname][loc].z <= riming_classif[stratname][loc].height_ml_new_gia + 2000,
                                                                    ) for loc in order_fil}
                    samples_wriming = {loc: stats[stratname][loc][ss].where(riming_filter[loc][riming_class].sum("z")>0).sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    samples_woriming = {loc: stats[stratname][loc][ss].where(riming_filter[loc][riming_class].sum("z")==0).sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil}

                    if ss in ["cloudtop", "cloudtop_5dbz", "cloudtop_10dbz"]: # filter out erroneous cloudtop values #!!! this will be fixed now (19.03.25) and this extra filter will not be necessary after re running the stats calculations
                        samples_wriming = {loc: samples_wriming[loc][samples_wriming[loc]<np.max(samples_wriming[loc])] for loc in samples_wriming.keys()}
                        samples_woriming = {loc: samples_woriming[loc][samples_woriming[loc]<np.max(samples_woriming[loc])] for loc in samples_woriming.keys()}
                    if ss in ["cloudtop_TEMP", "cloudtop_TEMP_5dbz", "cloudtop_TEMP_10dbz"]: # filter out erroneous cloudtop values #!!! this will be fixed now (19.03.25) and this extra filter will not be necessary after re running the stats calculations
                        samples_wriming = {loc: stats[stratname][loc][ss].where(riming_filter[loc][riming_class].sum("z")>0).where(stats[stratname][loc]["".join(ss.split("_TEMP"))].sel(\
                                    time=stats[stratname][loc]["".join(ss.split("_TEMP"))]['time'].dt.month.isin(selseas[1])) <
                                                                            stats[stratname][loc]["".join(ss.split("_TEMP"))].max().values).dropna("time").values\
                                   for loc in order_fil}
                        samples_woriming = {loc: stats[stratname][loc][ss].where(riming_filter[loc][riming_class].sum("z")==0).where(stats[stratname][loc]["".join(ss.split("_TEMP"))].sel(\
                                    time=stats[stratname][loc]["".join(ss.split("_TEMP"))]['time'].dt.month.isin(selseas[1])) <
                                                                            stats[stratname][loc]["".join(ss.split("_TEMP"))].max().values).dropna("time").values\
                                   for loc in order_fil}
                        # samples = {loc: samples[loc][samples_aux[loc]<np.max(samples_aux[loc])] for loc in samples.keys()}

                    samples = {loc.swapcase(): [ samples_wriming[loc], samples_woriming[loc] ] for loc in samples_wriming.keys() if ( len(samples_wriming[loc])>10 and len(samples_woriming[loc])>10 )} # filter out radars with no samples

                    fig = ridgeplot.ridgeplot(samples=samples.values(),
                                            colorscale="viridis",
                                            # colormode="row-index",
                                            colormode="trace-index-row-wise",
                                            # colormode="trace-index",
                                            coloralpha=0.65,
                                            labels=samples.keys(),
                                            linewidth=2,
                                            spacing=5 / 9,
                                            # kde_points=bins[ss],
                                            bandwidth=bandwidths[ss],
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=20,
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    title=ss,
                                    xaxis_tickvals=bins[ss],
                    )

                    # Add vertical zero line
                    fig.add_vline(x=0, line_width=2, line_color="gray")

                    # Get densities data from the plot
                    densities = [ fig.data[2*i+1] for i in range(len(samples)*2) ]

                    # calculate means
                    means = []
                    for sample0 in samples.values():
                        for sample in sample0:
                            means = np.append(means, np.median(sample) )

                    # Add a vertical line at the mean for each distribution
                    for i, mean in enumerate(means):
                        # define the bottom and top of each distribution
                        y_bot = np.array(densities[i]["y"]).min()
                        y_top = np.array(densities[i]["y"])[(np.where(np.array(densities[i]["x"]) >= mean))][0]

                        fig.add_scatter(
                            mode="lines",
                            x=[mean, mean],  # Vertical line at the mean
                            y=[y_bot , y_top],  # Set y0 and y1 based on the vertical offset
                            line=dict(color=change_rgba_alpha(densities[i]["fillcolor"], 1), width=2),
                        )

                    # We need to reorder the elements of the fig.data tuple so that
                    # the mean lines go below each distribution.
                    fig.data = reorder_tuple_elements(fig.data,len(means))

                    fig.write_html(savepath_seas+"/"+ss+".html")
                except:
                    print("!!! unable to plot "+stratname+" "+ss+" !!!")
            except:
                print("!!! unable to plot "+stratname+" "+ss+" !!!")

#%% Checking PHIDP
# get and plot a random selection of QVPs
import random
rand_dates = [random.randint(0, len(qvps_strat.time)) for _ in range(100)]
for xx in range(100):
    qvps.where(qvps_strat.time).loc[{"time":"2017-07-25"}]["KDP_ML_corrected"][xx].plot(color="b", alpha=0.1)


# PLot a random selection of QVPs with negative KDP in the first 7 bins
qvps_negKDP = qvps.where((qvps_strat["KDP_ML_corrected"][:,0:7]<=0).all("z"), drop=True)
rand_dates = [random.randint(0, len(qvps_negKDP.time)) for _ in range(100)]
for xx in rand_dates:
    qvps_negKDP["UPHIDP_OC_MASKED"][xx].plot(color="b", alpha=0.1, ylim=(-3,3))


#%% Load offsets for exploring
paths= [realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/ANK/*/12*/*below3C_timesteps-*",
        realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/ANK/*/14*/*below3C_timesteps-*"]

files_auxlist=[]
for pp in paths:
    files_auxlist.extend(glob.glob(pp))
files_off = sorted(files_auxlist)

zdr_off_LR_below3C_timesteps = xr.open_mfdataset(files_off, combine="nested", concat_dim="time")


#%% Test ZDR calibration

# ZDR offset looks nice for a nice stratiform case
pro_vp_20170725 = xr.open_datatree(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/90gradstarng01/00/ras07-90gradstarng01_sweeph5onem_allmoms_00-2017072500041700-pro-10392-hd5")["sweep_0"].to_dataset()
if pro_vp_20170725.time.isnull().any():
    pro_vp_20170725.coords["time"] = pro_vp_20170725["rtime"].min(dim="azimuth", skipna=True).compute()
loc="pro"
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
pro_vp_20170725 = utils.attach_ERA5_TEMP(pro_vp_20170725, path=loc.join(era5_dir.split("loc")))

zdr_offset_vp_pro_20170725 = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3).compute()

zdr_offset_vp_pro_20170725_azmedian = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3, azmed=True).compute()

# Let's find a not-nice case
pro_vp_20170126 = xr.open_datatree(glob.glob(realpep_path+"/upload/jgiles/dwd/2016/2016-01/2016-01-26/pro/90gradstarng01/00/ras07-90gradstarng01*")[0])["sweep_0"].to_dataset()
if pro_vp_20170126.time.isnull().any():
    pro_vp_20170126.coords["time"] = pro_vp_20170126["rtime"].min(dim="azimuth", skipna=True).compute()
loc="pro"
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
pro_vp_20170126 = utils.attach_ERA5_TEMP(pro_vp_20170126, path=loc.join(era5_dir.split("loc")))

zdr_offset_vp_pro_20170126 = utils.zdr_offset_detection_vps(pro_vp_20170126, min_h=400, timemode="all", mlbottom=3).compute()

zdr_offset_vp_pro_20170126_azmedian = utils.zdr_offset_detection_vps(pro_vp_20170126, min_h=400, timemode="all", mlbottom=3, azmed=True).compute()

# that gives horrible values, lets see the data
pro_vp_20170126.ZDR.median("azimuth").plot(x="time", vmin=-5, vmax=5)
pro_vp_20170126.TEMP.median("azimuth").plot.contour(x="time", levels=[0,3], colors="white")

utils.zdr_offset_detection_vps(pro_vp_20170126, min_h=400, timemode="step", mlbottom=3, azmed=True).compute().ZDR_offset.plot()
# only the timesteps in the end where the ZDR reasonable values touch the ground give reasonable ZDR offset,
# lets check the distribution of ZDR values
pro_vp_20170126.ZDR.plot.hist(bins=np.arange(-10,10.1,0.1))
# the distribution is has several values around -4, probably due to the noisy values close to the ML
# lets check the data filter that goes on in the function
pro_vp_20170126.ZDR.where(pro_vp_20170126.TEMP>3).where(pro_vp_20170126["z"]>400).median("azimuth").assign_coords({"z": pro_vp_20170126["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=5, ylim=(0,2500))

pro_vp_20170126.DBZH.where(pro_vp_20170126.TEMP>3).where(pro_vp_20170126["z"]>400).median("azimuth").assign_coords({"z": pro_vp_20170126["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=70, ylim=(0,2500))


pro_vp_20170126.ZDR.where(pro_vp_20170126.TEMP>3).where(pro_vp_20170126["z"]>400).where(pro_vp_20170126["DBZH"]>5).where(pro_vp_20170126["DBZH"]<30).where(pro_vp_20170126["RHOHV"]>0.98).median("azimuth").assign_coords({"z": pro_vp_20170126["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=5, ylim=(0,2500))

pro_vp_20170126.DBZH.where(pro_vp_20170126.TEMP>3).where(pro_vp_20170126["z"]>400).where(pro_vp_20170126["DBZH"]>5).where(pro_vp_20170126["DBZH"]<30).where(pro_vp_20170126["RHOHV"]>0.98).median("azimuth").assign_coords({"z": pro_vp_20170126["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-0, vmax=30, ylim=(0,2500))

pro_vp_20170126.RHOHV.where(pro_vp_20170126.TEMP>3).where(pro_vp_20170126["z"]>400).where(pro_vp_20170126["DBZH"]>5).where(pro_vp_20170126["DBZH"]<30).where(pro_vp_20170126["RHOHV"]>0.98).median("azimuth").assign_coords({"z": pro_vp_20170126["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=0.98, vmax=1, ylim=(0,2500))


# Repeat for ANK
ank_12_20180306 = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/2018/2018-03/2018-03-06/ANK/MON_YAZ_G/14.0/*")
loc="ank"
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
ank_12_20180306 = utils.attach_ERA5_TEMP(ank_12_20180306, path=loc.join(era5_dir.split("loc")))

zdr_offset_ank_12_20180306 = utils.zhzdr_lr_consistency(ank_12_20180306, min_h=ank_12_20180306.altitude.values+300, timemode="all", mlbottom=3).compute()

# that gives horrible values, lets see the data
ank_12_20180306.ZDR.median("azimuth").plot(x="time", vmin=-5, vmax=5, ylim=(0,20000))
ank_12_20180306.TEMP.median("azimuth").plot.contour(x="time", levels=[0,3], colors="white")

# lets check the distribution of ZDR values
ank_12_20180306.ZDR.plot.hist(bins=np.arange(-10,10.1,0.1))

# lets check the data filter that goes on in the function
ank_12_20180306.ZDR.where(ank_12_20180306.TEMP>3).where(ank_12_20180306["z"]>ank_12_20180306.altitude.values+300).median("azimuth").assign_coords({"z": ank_12_20180306["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=5, ylim=(ank_12_20180306.altitude,ank_12_20180306.altitude+2500))
ank_12_20180306.DBZH.where(ank_12_20180306.TEMP>3).where(ank_12_20180306["z"]>ank_12_20180306.altitude.values+300).median("azimuth").assign_coords({"z": ank_12_20180306["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=5, ylim=(ank_12_20180306.altitude,ank_12_20180306.altitude+2500))

ank_12_20180306.ZDR.where(ank_12_20180306.TEMP>3).where(ank_12_20180306["z"]>ank_12_20180306.altitude.values+300).where((ank_12_20180306["DBZH"]>5)&(ank_12_20180306["DBZH"]<30)&(ank_12_20180306["RHOHV"]>0.98)).median("azimuth").assign_coords({"z": ank_12_20180306["z"].median("azimuth")}).swap_dims({"range":"z"}).plot(x="time", y="z", vmin=-5, vmax=5, ylim=(ank_12_20180306.altitude,ank_12_20180306.altitude+2500))


#%% Test KDP from ZPHI
ff = realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/ras07-vol5minng01_sweeph5onem_allmoms_07-2017072500033500-pro-10392-hd5"
pro20170725=xr.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()

#%% Load multiple elevations of DWD to check if there is better PHIDP
ff = realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/*/*allmoms*"

files = sorted(glob.glob(ff))

vollist = []
for fx in files:
    vollist.append(xr.open_datatree(fx)["sweep_"+fx.split("/")[-2][1]].to_dataset())
    vollist[-1].coords["elevation"] = vollist[-1].coords["elevation"].median()
    vollist[-1] = vollist[-1].expand_dims("elevation")

xx = 9
tt = np.arange(0, len(vollist[xx].time), 20)
aa = np.arange(0, 360, 30)
for tx in tt:
    for ax in aa:
        vollist[xx].UPHIDP[0, tx, ax,:].plot(color="b", alpha=0.01)

vol = xr.concat(vollist, dim="elevation")


#%% Convective events
# filter
data_fil = data.where(data[X_DBZH]>30, drop=True).where(qvps.height_ml_new_gia.isnull(), drop=True)
data_fil = data_fil.pipe(wrl.georef.georeference_dataset)

# attach temperature
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
data_fil = utils.attach_ERA5_TEMP(data_fil, path="pro".join(era5_dir.split("loc")))

# Plot CFTDs

# adjustment from K to C (disables now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit
ytlim=-20

# Temp bins
tb=1# degress C

# Min counts per Temp layer
mincounts=200

#Colorbar limits and step
cblim=[0,10]
colsteps=10




if country=="dwd":

    vars_to_plot = {"DBZH": [0, 46, 1],
                    "ZDR": [-0.5, 2.1, 0.1],
                    "KDP": [-0.1, 0.52, 0.02],
                    "RHOHV": [0.9, 1.004, 0.004]}

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        adj=1
        if "RHOHV" in vv:
            so = True
            binsx2 = [0.9, 1.005, 0.005]
        if "KDP" in vv:
            adj=1
        utils.hist2d(ax[nn], data_fil[vv]*adj, data_fil["TEMP"]+adjtemp, whole_x_range=True,
                     binsx=vars_to_plot[vv], binsy=[-20,16,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps,
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)

        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts



    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')


#%% Plot map with radars and partial beam blockage
from osgeo import osr
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy

files = [glob.glob(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/tur/vol5minng01/07/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/umd/vol5minng01/07/*allmoms*")[0],
         ]

files = [glob.glob(realpep_path+"/upload/jgiles/dmi/2015/2015-03/2015-03-03/ANK/MON_YAZ_K/12.0/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dmi/2020/2020-07/2020-07-02/AFY/VOL_B/10.0/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dmi/2016/2016-04/2016-04-07/GZT/MON_YAZ_C/12.0/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dmi/2016/2016-04/2016-04-07/HTY/MON_YAZ_C/12.0/*allmoms*")[0],
         glob.glob(realpep_path+"/upload/jgiles/dmi/2020/2020-01/2020-01-11/SVS/VOL_B/10.0/*allmoms*")[0],
         ]


# Create a Stamen terrain background instance.
terrain_map = cimgt.GoogleTiles('satellite') # This used Stamen maps before but it was moved to Stadia and not fixed in cartopy so far

# set projection
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)

CBB_list = []
for ff in files:

    # Load a sample PPI
    if "dwd" in files[0]:
        swpx = xr.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset().DBZH[0]
    if "dmi" in files[0]:
        swpx = xr.open_dataset(ff).DBZH[0]

    swpx = swpx.pipe(wrl.georef.georeference, proj=wgs84)

    # Download DEM data

    extent = wrl.zonalstats.get_bbox(swpx.x.values, swpx.y.values)
    extent

    # apply token
    os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTY5NzkwMDAyMiwiaWF0IjoxNjkyNzE2MDIyLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.4OhlJ-fTL_ii7EB2Eavyg7fPotk_U6g5ZC9ryS1RFp0cb8KGDl0ptwtifmV7A1__5FbLQlvH3MUKQg_Gq5LKTGi61bn_BBeXzRxx2Z8WJW7uuESQQH61urrbji-xwiIVo65r0tDfT0qYYulbA4X9DPBom2BHMvcvitgnvwRiQFpK8S6h7xoYLqCgHJOtATBc_2Su28qaDfH_SwRLI81iQYDnfLPhL_iWVf3bQxdObl31WD4inrST8IMSg59KMuioRRHdydE7PPsGxHWV5U2PFfRwjS1dqi0ntP_mlXoBpG-Eh-vNdaWi4KSGZA4PYN4AuTV1ijzGEzd8Qvw2aIo6Xg"
    # set location of wradlib-data, where wradlib will search for any available data
    os.environ["WRADLIB_DATA"] = "/home/jgiles/wradlib-data-main/"
    # get the tiles
    dem = wrl.io.get_srtm(extent.values())

    # DEM to spherical coords

    sitecoords = (swpx.longitude.values, swpx.latitude.values, swpx.altitude.values)
    r = swpx.range.values
    az = swpx.azimuth.values
    bw = 1
    beamradius = wrl.util.half_power_radius(r, bw)

    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(
        dem, nodata=-32768.0
    )

    rlimits = (extent["left"], extent["bottom"], extent["right"], extent["top"])
    # Clip the region inside our bounding box
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords = rastercoords[ind[1] : ind[3], ind[0] : ind[2], ...]
    rastervalues = rastervalues[ind[1] : ind[3], ind[0] : ind[2]]

    polcoords = np.dstack([swpx.x.values, swpx.y.values])
    # Map rastervalues to polar grid points
    polarvalues = wrl.ipol.cart_to_irregular_spline(
        rastercoords, rastervalues, polcoords, order=3, prefilter=False
    )

    # Partial and cumulative beam blockage
    PBB = wrl.qual.beam_block_frac(polarvalues, swpx.z.values, beamradius)
    PBB = np.ma.masked_invalid(PBB)

    CBB = wrl.qual.cum_beam_block_frac(PBB)
    CBB_xr = xr.ones_like(swpx)*CBB
    CBB_list.append(CBB_xr.rename("Beam blockage fraction").copy())

#make the plots
# https://stackoverflow.com/questions/62448828/python-cartopy-map-clip-area-outside-country-polygon
fs = 5
with mpl.rc_context({'font.size': fs}):
    fig = plt.figure()

    # create subplots
    ax = fig.add_subplot(1, 1, 1, projection=terrain_map.crs)

    # Limit the extent of the map to a small longitude/latitude range.
    # ax.set_extent([13, 15, 52, 54], crs=ccrs.Geodetic())  # [0, 20, 45, 55]
    if "dwd" in files[0]:
        ax.set_extent([6, 15, 47, 55], crs=ccrs.Geodetic())  # [0, 20, 45, 55]
    if "dmi" in files[0]:
        ax.set_extent([25, 45, 35, 42], crs=ccrs.Geodetic())  # [0, 20, 45, 55]

    # Add the Stamen data at zoom level 8.
    ax.add_image(terrain_map, 8, alpha=1)

    for nn,CBB_xr in enumerate(CBB_list):
        # Plot CBB (on ax1)
        cbarbool = False
        if nn == 0: cbarbool = True
        CBB_xr.plot(x="x", y="y", ax=ax, alpha= 0.7, vmin=0, vmax=1, cmap=mpl.cm.PuRd, transform=ccrs.PlateCarree(), add_colorbar=cbarbool)
        # ax1, cbb = wrl.vis.plot_ppi(CBB_xr, ax=ax, r=r, az=az, cmap=mpl.cm.PuRd, vmin=0, vmax=1)

        # add a marker in center of the radar
        ax.plot(CBB_xr.longitude, CBB_xr.latitude, marker='o', color='red', markersize=1,
                alpha=1, transform=ccrs.Geodetic())

    ax.coastlines(alpha=0.7, linewidth=0.5)
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    gl.xlabel_style = {'size': fs}
    gl.ylabel_style = {'size': fs}
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.4) #countries
    ax.tick_params(axis='both', labelsize=fs)

    plt.title("")

#%% Plot partial beam blockage and scan with DEM
from osgeo import osr

wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)

# Load a sample PPI
# swpx = dttree.open_datatree(realpep_path+"/upload/jgiles/dwd/2016/2016-01/2016-01-01/pro/vol5minng01/07/ras07-vol5minng01_sweeph5onem_allmoms_07-2016010100034100-pro-10392-hd5")["sweep_7"].to_dataset().DBZH[0]
swpx = xr.open_datatree(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/05/ras07-vol5minng01_sweeph5onem_allmoms_05-2017072500030000-pro-10392-hd5")["sweep_5"].to_dataset().DBZH[0]
# swpx = xr.open_dataset(realpep_path+"/upload/jgiles/dmi/2018/2018-03/2018-03-06/HTY/VOL_B/10.0/VOL_B-allmoms-10.0-2018-03-06-HTY-h5netcdf.nc").DBZH[0]
swpx = swpx.pipe(wrl.georef.georeference_dataset,  proj=wgs84)

# Download DEM data

extent = wrl.zonalstats.get_bbox(swpx.x.values, swpx.y.values)
extent

# apply fake token, data is already available
os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTY5NzkwMDAyMiwiaWF0IjoxNjkyNzE2MDIyLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.4OhlJ-fTL_ii7EB2Eavyg7fPotk_U6g5ZC9ryS1RFp0cb8KGDl0ptwtifmV7A1__5FbLQlvH3MUKQg_Gq5LKTGi61bn_BBeXzRxx2Z8WJW7uuESQQH61urrbji-xwiIVo65r0tDfT0qYYulbA4X9DPBom2BHMvcvitgnvwRiQFpK8S6h7xoYLqCgHJOtATBc_2Su28qaDfH_SwRLI81iQYDnfLPhL_iWVf3bQxdObl31WD4inrST8IMSg59KMuioRRHdydE7PPsGxHWV5U2PFfRwjS1dqi0ntP_mlXoBpG-Eh-vNdaWi4KSGZA4PYN4AuTV1ijzGEzd8Qvw2aIo6Xg"
# set location of wradlib-data, where wradlib will search for any available data
os.environ["WRADLIB_DATA"] = "/home/jgiles/wradlib-data-main/"
# get the tiles
dem = wrl.io.get_srtm(extent.values())

# DEM to spherical coords

sitecoords = (swpx.longitude.values, swpx.latitude.values, swpx.altitude.values)
r = swpx.range.values
az = swpx.azimuth.values
bw = 1
beamradius = wrl.util.half_power_radius(r, bw)

rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(
    dem, nodata=-32768.0
)

rlimits = (extent["left"], extent["bottom"], extent["right"], extent["top"])
# Clip the region inside our bounding box
ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
rastercoords = rastercoords[ind[1] : ind[3], ind[0] : ind[2], ...]
rastervalues = rastervalues[ind[1] : ind[3], ind[0] : ind[2]]

polcoords = np.dstack([swpx.x.values, swpx.y.values])
# Map rastervalues to polar grid points
polarvalues = wrl.ipol.cart_to_irregular_spline(
    rastercoords, rastervalues, polcoords, order=3, prefilter=False
)

# Partial and cumulative beam blockage
PBB = wrl.qual.beam_block_frac(polarvalues, swpx.z.values, beamradius)
PBB = np.ma.masked_invalid(PBB)

CBB = wrl.qual.cum_beam_block_frac(PBB)

# just a little helper function to style x and y axes of our maps
def annotate_map(ax, cm=None, title=""):
    xticks = ax.get_xticks()
    ticks = (xticks / 1000).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(ticks)
    yticks = ax.get_yticks()
    ticks = (yticks / 1000).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Kilometers")
    ax.set_ylabel("Kilometers")
    if not cm is None:
        plt.colorbar(cm, ax=ax)
    if not title == "":
        ax.set_title(title)
    ax.grid()

#make the plots
alt = swpx.z.values
fig = plt.figure(figsize=(15, 12))

# create subplots
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)

# azimuth angle
angle = 270

# Plot terrain (on ax1)
ax1, dem = wrl.vis.plot_ppi(
    polarvalues, ax=ax1, r=r, az=az, cmap=mpl.cm.cubehelix, vmin=0.0
)
ax1.plot(
    [0, np.sin(np.radians(angle)) * 1e5], [0, np.cos(np.radians(angle)) * 1e5], "r-"
)
ax1.plot(sitecoords[0], sitecoords[1], "ro")
annotate_map(ax1, dem, "Terrain within {0} km range".format(np.max(r / 1000.0) + 0.1))
ax1.set_xlim(-100000, 100000)
ax1.set_ylim(-100000, 100000)

# Plot CBB (on ax2)
ax2, cbb = wrl.vis.plot_ppi(CBB, ax=ax2, r=r, az=az, cmap=mpl.cm.PuRd, vmin=0, vmax=1)
annotate_map(ax2, cbb, "Beam-Blockage Fraction")
ax2.set_xlim(-100000, 100000)
ax2.set_ylim(-100000, 100000)

# Plot single ray terrain profile on ax3
(bc,) = ax3.plot(r / 1000.0, alt[angle, :], "-b", linewidth=3, label="Beam Center")
(b3db,) = ax3.plot(
    r / 1000.0,
    (alt[angle, :] + beamradius),
    ":b",
    linewidth=1.5,
    label="3 dB Beam width",
)
ax3.plot(r / 1000.0, (alt[angle, :] - beamradius), ":b")
ax3.fill_between(r / 1000.0, 0.0, polarvalues[angle, :], color="0.75")
ax3.set_xlim(0.0, np.max(r / 1000.0) + 0.1)
ax3.set_ylim(0.0, 3000)
ax3.set_xlabel("Range (km)")
ax3.set_ylabel("Altitude (m)")
ax3.grid()

axb = ax3.twinx()
(bbf,) = axb.plot(r / 1000.0, CBB[angle, :], "-g", label="BBF")
axb.spines["right"].set_color("g")
axb.tick_params(axis="y", colors="g")
axb.set_ylabel("Beam-blockage fraction", c="g")
axb.set_ylim(0.0, 1.0)
axb.set_xlim(0.0, np.max(r / 1000.0) + 0.1)


legend = ax3.legend(
    (bc, b3db, bbf),
    ("Beam Center", "3 dB Beam width", "BBF"),
    loc="upper left",
    fontsize=10,
)

#%% Test plot background map image
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


def main():
    # Create a Stamen terrain background instance.
    stamen_terrain = cimgt.Stamen('terrain-background')

    fig = plt.figure()

    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([0, 20, 45, 55], crs=ccrs.Geodetic())

    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)

    # Add a marker for the Eyjafjallajökull volcano.
    ax.plot(-19.613333, 63.62, marker='o', color='red', markersize=12,
            alpha=0.7, transform=ccrs.Geodetic())

    # Use the cartopy interface to create a matplotlib transform object
    # for the Geodetic coordinate system. We will use this along with
    # matplotlib's offset_copy function to define a coordinate system which
    # translates the text by 25 pixels to the left.
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
    text_transform = offset_copy(geodetic_transform, units='dots', x=-25)

    # Add text 25 pixels to the left of the volcano.
    # ax.text(-19.613333, 63.62, u'Eyjafjallajökull',
    #         verticalalignment='center', horizontalalignment='right',
    #         transform=text_transform,
    #         bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))
    # plt.show()


if __name__ == '__main__':
    main()
