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

import datatree as dttree
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
# This part should be run after having the QVPs computed (compute_qvps.py)
start_time = time.time()
print("Loading QVPs...")

#### Get QVP file list
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# Load only events with ML detected (pre-condition for stratiform)
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/20*/*/*/pro/vol5minng01/07/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps/2016/*/*/ANK/*/*/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
path_qvps = realpep_path+"/upload/jgiles/dmi/qvps/*/*/*/HTY/*/*/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps_singlefile/ML_detected/ANK/*/12*/*allmoms*"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*"
# path_qvps = [realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*",
#              realpep_path+"/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/14*/*allmoms*"]


#### Set variable names
X_DBZH = "DBZH"
X_RHO = "RHOHV_NC" # if RHOHV_NC is set here, it is then checked agains the original RHOHV in the next cell
X_ZDR = "ZDR_OC"
X_KDP = "KDP_ML_corrected"

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
#         open_files.append(fix_dailys(dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()))
#     else:
#         open_files.append(fix_dailys(xr.open_dataset(ff)))
        
# data = xr.concat(open_files, dim="time")
total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%% Filters (conditions for stratiform)
start_time = time.time()
print("Filtering stratiform conditions...")

# Filter only stratiform events (min entropy >= 0.8) and ML detected
# with ProgressBar():
#     qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull()), drop=True).compute()

# Check that RHOHV_NC is actually better (less std) than RHOHV, otherwise just use RHOHV, on a per-day basis
std_margin = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_margin), otherwise use RHOHV
min_rho = 0.6 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results

if "_NC" in X_RHO:
    # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
    # if that is the case we take it that the correction did not work well so we won't use it
    cond_rhohv = (
                    qvps[X_RHO].where(qvps[X_RHO]>min_rho).resample({"time":"D"}).std(dim=("time", "z")) < \
                    qvps["RHOHV"].where(qvps["RHOHV"]>min_rho).resample({"time":"D"}).std(dim=("time", "z"))*(1+std_margin) 
                    ).compute()
    
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

# Filter only fully stratiform pixels (min entropy >= 0.8 and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8).compute() & allcond, drop=True)
# Relaxed alternative: Filter qvps with at least 50% of stratiform pixels (min entropy >= 0.8 and ML detected)
qvps_strat_relaxed = qvps.where( ( (qvps["min_entropy"]>=0.8).sum("z").compute() >= qvps[X_DBZH].count("z").compute()/2 ) & allcond, drop=True)

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

#### Calculate retreivals
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
    retrievals
except NameError:
    retrievals = {}

for stratname, stratqvp in [("stratiform", qvps_strat_fil), ("stratiform_relaxed", qvps_strat_relaxed_fil)]:
    print("   ... for "+stratname)
    
    retrievals[stratname] = {}
    
    # LWC 
    lwc_zh_zdr = 10**(0.058*stratqvp[X_DBZH] - 0.118*stratqvp[X_ZDR] - 2.36) # Reimann et al 2021 (adjusted for Germany)
    lwc_zh_zdr2 = 1.38*10**(-3) *10**(0.1*stratqvp[X_DBZH] - 2.43*stratqvp[X_ZDR] + 1.12*stratqvp[X_ZDR]**2 - 0.176*stratqvp[X_ZDR]**3 ) # Ryzhkov et al 2022, used in S band
    lwc_kdp = 10**(0.568*np.log10(stratqvp[X_KDP]) + 0.06) # Reimann et al 2021(adjusted for Germany)
    
    # IWC (Collected from Blanke et al 2023)
    iwc_zh_t = 10**(0.06 * stratqvp[X_DBZH] - 0.0197*stratqvp["TEMP"] - 1.7) # empirical from Hogan et al 2006
    
    iwc_zdr_zh_kdp = xr.where(stratqvp[X_ZDR]>0.4, # Carlin et al 2021
                              4*10**(-3)*( stratqvp[X_KDP]*Lambda/( 1-wrl.trafo.idecibel(stratqvp[X_ZDR])**-1 ) ), 
                              0.031474 * ( stratqvp[X_KDP]*Lambda )**0.66 * stratqvp[X_DBZH]**0.28 ) 
    
    # Dm
    Dm_ice_zh = 1.055*stratqvp[X_DBZH]**0.271 # Matrosov et al. (2019)
    Dm_ice_zh_kdp = 0.67*( stratqvp[X_DBZH]/(stratqvp[X_KDP]*Lambda) )**(1/3) # Bukovcic et al. (2020)
    Dm_rain_zdr = 0.3015*stratqvp[X_ZDR]**3 - 1.2087*stratqvp[X_ZDR]**2 + 1.9068*stratqvp[X_ZDR] + 0.5090 # (for rain but tuned for Germany X-band, JuYu Chen, Zdr in dB, Dm in mm)
    Dm_rain_zdr2 = 0.171*stratqvp[X_ZDR]**3 - 0.725*stratqvp[X_ZDR]**2 + 1.48*stratqvp[X_ZDR] + 0.717 # (Hu and Ryzhkov 2022, used in S band data but could work for C band)
    Dm_rain_zdr3 = xr.where(stratqvp[X_ZDR]<1.25, # Bringi et al 2009 (C-band)
                            0.0203*stratqvp[X_ZDR]**4 - 0.149*stratqvp[X_ZDR]**3 + 0.221*stratqvp[X_ZDR]**2 + 0.557*stratqvp[X_ZDR] + 0.801,
                            0.0355*stratqvp[X_ZDR]**3 - 0.302*stratqvp[X_ZDR]**2 + 1.06*stratqvp[X_ZDR] + 0.684
                            )
    
    # log(Nt)
    Nt_ice_zh_iwc = (3.39 + 2*np.log10(iwc_zh_t) - 0.1*stratqvp[X_DBZH]) # (Hu and Ryzhkov 2022, different than Carlin et al 2021 only in the offset, but works better)
    Nt_rain_zh_zdr = ( -2.37 + 0.1*stratqvp[X_DBZH] - 2.89*stratqvp[X_ZDR] + 1.28*stratqvp[X_ZDR]**2 - 0.213*stratqvp[X_ZDR]**3 )# Hu and Ryzhkov 2022
    
    # Put everything together
    retrievals[stratname][find_loc(locs, ff[0])] = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
                                                             "lwc_zh_zdr2":lwc_zh_zdr2,
                                                             "lwc_kdp": lwc_kdp,
                                                             "iwc_zh_t": iwc_zh_t,
                                                             "iwc_zdr_zh_kdp": iwc_zdr_zh_kdp,
                                                             "Dm_ice_zh": Dm_ice_zh,
                                                             "Dm_ice_zh_kdp": Dm_ice_zh_kdp,
                                                             "Dm_rain_zdr": Dm_rain_zdr,
                                                             "Dm_rain_zdr2": Dm_rain_zdr2,
                                                             "Dm_rain_zdr3": Dm_rain_zdr3,
                                                             "Nt_ice_zh_iwc": Nt_ice_zh_iwc,
                                                             "Nt_rain_zh_zdr": Nt_rain_zh_zdr,
                                                             }).compute()

    # Save retrievals
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

for stratname, stratqvp in [("stratiform", qvps_strat_fil), ("stratiform_relaxed", qvps_strat_relaxed_fil)]:
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
    
    height_ML_max = qvps_ML.idxmax("z")
    height_ML_min = qvps_ML.idxmin("z")
    
    # Silke style
    # select timesteps with detected ML
    # gradient_silke = stratqvp.where(stratqvp["height_ml_new_gia"] > stratqvp["height_ml_bottom_new_gia"], drop=True)
    # gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
    # gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
    # gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
    # beta = gradient_final[X_TH] #### TH OR DBZH??
    
    # Gradient above the ML
    # First we select only above the ML. Then we interpolate possibly missing values. 
    # Then we select above z_snow_over_ML and below z_snow_over_ML + z_grad_above_ML
    # Then we compute the gradient.
    beta = stratqvp.where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                    .interpolate_na("z") \
                        .where(stratqvp["z"] > (stratqvp["height_ml_new_gia"] + z_snow_over_ML) ) \
                            .where(stratqvp["z"] < (stratqvp["height_ml_new_gia"] + z_snow_over_ML + z_grad_above_ML) )\
                                .differentiate("z").median("z") * 1000 # x1000 to transform the gradients to /km

    # Gradient below the ML
    # First we select only above the ML. Then we interpolate possibly missing values. 
    # Then we select above z_snow_over_ML and below z_snow_over_ML + z_grad_above_ML
    # Then we compute the gradient.
    beta_belowML = stratqvp.where(stratqvp["z"] < (stratqvp["height_ml_bottom_new_gia"]) ) \
                    .interpolate_na("z") \
                        .where(stratqvp["z"] < (stratqvp["height_ml_bottom_new_gia"] - z_rain_below_ML ) )\
                            .differentiate("z").median("z") * 1000 # x1000 to transform the gradients to /km
    
    # Cloud top (3 methods)
    # Get the height value of the last not null value with a minimum of entropy 0.2 (this min entropy is to filter out random noise pixels)
    cloudtop = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2 ) \
                        .notnull().isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop")
    # Get the height value of the last value > 5 dBZ
    cloudtop_5dbz = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2).where(stratqvp[X_DBZH]>5) \
                        .notnull().isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop 5 dBZ")
    # Get the height value of the last value > 10 dBZ
    cloudtop_10dbz = stratqvp[X_DBZH].where(stratqvp["z"] > (stratqvp["height_ml_new_gia"]) ) \
                        .where(stratqvp["min_entropy"] > 0.2).where(stratqvp[X_DBZH]>10) \
                        .notnull().isel(z=slice(None,None,-1)).idxmax("z").rename("cloudtop 10 dBZ")

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
    for ll in stats[stratname].keys():
        for xx in stats[stratname][ll].keys():
            stats[stratname][ll][xx].to_netcdf(realpep_path+"/upload/jgiles/radar_stats/"+stratname+"/"+ll+"_"+xx+".nc")

    ''' DEPRECATED
    # THIS IS NOW INCLUDED IN IT'S OWN SECTION DOWN BELOW
    #### Statistics from Raquel
    MLmaxZH = values_ML_max["DBZH"]
    ZHrain = values_rain["DBZH"] 
    deltaZH = MLmaxZH - ZHrain
    MLminRHOHV = values_ML_min["RHOHV_NC"]
    
    MLdepth = ML_thickness

    #### Histograms: like Ryzhkov and Krauze 2022 https://doi.org/10.1175/JTECH-D-21-0130.1
    print("   ... plotting scatterplots")
    # plot histograms (2d hist) like Fig. 10
    # plot a
    binsx = np.linspace(0.8, 1, 41)
    binsy = np.linspace(-10, 20, 61)
    deltaZHcurve = 4.27 + 6.89*(1-binsx) + 341*(1-binsx)**2 # curve from Ryzhkov and Krauze 2022 https://doi.org/10.1175/JTECH-D-21-0130.1
    
    utils.hist_2d(MLminRHOHV.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
    plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
    plt.legend()
    plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML}$")
    plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) }$")
    plt.text(0.81, -8, r"$\mathregular{\Delta Z_H = 4.27 + 6.89(1-\rho _{HV}) + 341(1-\rho _{HV})^2 }$", fontsize="small")
    plt.grid()
    fig = plt.gcf()
    fig.savefig("/automount/agradar/jgiles/images/stats_histograms/"+stratname+"/"+find_loc(locs, ff[0])+"_DeltaZH_MinRHOHVinML.png",
                bbox_inches="tight")
    plt.close(fig)
    
    # plot b
    binsx = np.linspace(0, 4, 16*5+1)
    binsy = np.linspace(-10, 20, 61)
    deltaZHcurve = 3.18 + 2.19*binsx # curve from Ryzhkov and Krauze 2022 https://doi.org/10.1175/JTECH-D-21-0130.1
    
    utils.hist_2d(values_ML_max["ZDR_OC"].compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
    plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
    plt.legend()
    plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML}$")
    plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) }$")
    plt.text(2, -8, r"$\mathregular{\Delta Z_H = 3.18 + 2.19 Z_{DR} }$", fontsize="small")
    plt.grid()
    fig = plt.gcf()
    fig.savefig("/automount/agradar/jgiles/images/stats_histograms/"+stratname+"/"+find_loc(locs, ff[0])+"_DeltaZH_MaxZDRinML.png",
                bbox_inches="tight")
    plt.close(fig)
    
    # plot c
    binsx = np.linspace(0.8, 1, 41)
    binsy = np.linspace(0, 1000, 26)
    MLdepthcurve = -0.64 + 30.8*(1-binsx) - 315*(1-binsx)**2 + 1115*(1-binsx)**3 # curve from Ryzhkov and Krauze 2022 https://doi.org/10.1175/JTECH-D-21-0130.1
    
    utils.hist_2d(MLminRHOHV.compute(), MLdepth.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
    plt.plot(binsx, MLdepthcurve*1000, c="black", label="Reference curve") # multiply curve by 1000 to change from km to m
    plt.legend()
    plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML}$")
    plt.ylabel(r"Depth of ML (m)")
    # plt.text(0.8, 800, r"$\mathregular{ML \ Depth = -0.64 + 30.8(1-\rho _{HV}) - 315(1-\rho _{HV})^2 + 1115(1-\rho _{HV})^3 }$", fontsize="xx-small")
    plt.text(0.86, 700, r"$\mathregular{ML \ Depth = -0.64 + 30.8(1-\rho _{HV})}$" "\n" r"$\mathregular{- 315(1-\rho _{HV})^2 + 1115(1-\rho _{HV})^3}$", fontsize="xx-small")
    plt.grid()
    fig = plt.gcf()
    fig.savefig("/automount/agradar/jgiles/images/stats_histograms/"+stratname+"/"+find_loc(locs, ff[0])+"_DepthML_MinRHOHVinML.png",
                bbox_inches="tight")
    plt.close(fig)
    
    # plot d
    binsx = np.linspace(0, 4, 16*5+1)
    binsy = np.linspace(0, 1000, 26)
    MLdepthcurve = 0.21 + 0.091*binsx # curve from Ryzhkov and Krauze 2022 https://doi.org/10.1175/JTECH-D-21-0130.1
    
    utils.hist_2d(values_ML_max["ZDR_OC"].compute(), MLdepth.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
    plt.plot(binsx, MLdepthcurve*1000, c="black", label="Reference curve") # multiply curve by 1000 to change from km to m
    plt.legend()
    plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML}$")
    plt.ylabel(r"Depth of ML (m)")
    # plt.text(0.8, 800, r"$\mathregular{ML \ Depth = -0.64 + 30.8(1-\rho _{HV}) - 315(1-\rho _{HV})^2 + 1115(1-\rho _{HV})^3 }$", fontsize="xx-small")
    plt.text(0.86, 700, r"$\mathregular{ML \ Depth = 0.21 + 0.091 Z_{DR} }$", fontsize="xx-small")
    plt.grid()
    fig = plt.gcf()
    fig.savefig("/automount/agradar/jgiles/images/stats_histograms/"+stratname+"/"+find_loc(locs, ff[0])+"_DepthML_MaxZDRinML.png",
                bbox_inches="tight")
    plt.close(fig)
    
    '''

#### Calculate riming
# We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil
print("Calculating riming ...")

with open('/automount/agradar/jgiles/riming_model/gbm_model_23.10.2024.pkl', 'rb') as f:
    riming_model = pickle.load(f)

with open('/automount/agradar/jgiles/riming_model/gbm_zh_zdr_model_23.10.2024.pkl', 'rb') as f:
    riming_model_zh_zdr = pickle.load(f)

def calc_depolarization(da, zdr="ZDR_OC", rho="RHOHV_NC"):   
    return xr.apply_ufunc(wrl.dp.depolarization,
                          da[zdr], da[rho].where(da[rho]<1),                         
                        dask='parallelized',                        
    )
    
# We will put the final riming classification in a dict
try: # check if exists, if not, create it
    riming_classif
except NameError:
    riming_classif = {}

for stratname, stratqvp in [("stratiform", qvps_strat_fil), ("stratiform_relaxed", qvps_strat_relaxed_fil)]:
    print("   ... for "+stratname)
    
    if "DR" not in stratqvp:
        DR = calc_depolarization(stratqvp, X_ZDR, X_RHO)
        assign = dict(DR = DR.assign_attrs(
            {'long_name': 'Depolarization ratio based on '+X_ZDR+' and '+X_RHO,
             'standard_name': 'depolarization_ratio',
             'units': 'dB'}
            ))
        stratqvp = stratqvp.assign(assign)
    if "UDR" not in stratqvp:
        UDR = calc_depolarization(stratqvp, "ZDR", "RHOHV")            
        assign = dict(UDR = UDR.assign_attrs(
            {'long_name': 'Depolarization ratio based on ZDR and RHOHV',
             'standard_name': 'depolarization_ratio',
             'units': 'dB'}
            ))
        stratqvp = stratqvp.assign(assign)
        
    # predict riming with the model
    for XDR, XZDR, XZH in [("DR", X_ZDR, X_DBZH), ("UDR", "ZDR", "DBZH")]:
        
        idx = np.isfinite(stratqvp[XDR].values.flatten() + stratqvp[XZDR].values.flatten() + stratqvp[XZH].values.flatten())
        X = np.concatenate((stratqvp[XDR].values.flatten()[idx].reshape(-1, 1), stratqvp[XZDR].values.flatten()[idx].reshape(-1, 1), stratqvp[XZH].values.flatten()[idx].reshape(-1, 1)), axis=1)
        
        pred = riming_model.predict(X)
        
        pred_riming = np.zeros_like(stratqvp[XDR]).flatten() + np.nan
        pred_riming[idx] = pred
        pred_riming = xr.zeros_like(stratqvp[XDR]) + pred_riming.reshape(stratqvp[XDR].shape)
        
        pred_riming.assign_attrs({
            'long_name': 'Riming prediction based on '+XDR+', '+XZDR+' and '+XZH+' with gradient boosting model',
             'standard_name': 'riming_prediction',
            })
        
        varname = "riming_"+XDR
        
        assign = {varname: pred_riming.copy()}
        stratqvp = stratqvp.assign(assign)

        # save to file
        if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname):
            os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname)
        
        pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+ll+"_"+varname+".nc")

    # predict riming with the model that uses only zh and zdr
    for XZDR, XZH in [(X_ZDR, X_DBZH), ("ZDR", "DBZH")]:
        
        idx = np.isfinite(stratqvp[XZDR].values.flatten() + stratqvp[XZH].values.flatten())
        X = np.concatenate((stratqvp[XZDR].values.flatten()[idx].reshape(-1, 1), stratqvp[XZH].values.flatten()[idx].reshape(-1, 1)), axis=1)
        
        pred = riming_model_zh_zdr.predict(X)
        
        pred_riming = np.zeros_like(stratqvp[XZH]).flatten() + np.nan
        pred_riming[idx] = pred
        pred_riming = xr.zeros_like(stratqvp[XZH]) + pred_riming.reshape(stratqvp[XZH].shape)
        
        pred_riming.assign_attrs({
            'long_name': 'Riming prediction based on '+XZDR+' and '+XZH+' with gradient boosting model',
             'standard_name': 'riming_prediction',
            })
        
        varname = "riming_"+XZDR+"_"+XZH
        
        assign = {varname: pred_riming.copy()}
        stratqvp = stratqvp.assign(assign)

        # save to file
        if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname):
            os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname)
        
        pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif/"+stratname+"/"+ll+"_"+varname+".nc")
        
total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

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
mincounts=200

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
# RHOHV: scales with a square root (finer towars RHOHV=1), so from 0.00278 at RHOHV=0.7 to 0.002 resolution at RHOHV=1
# PHIDP: 0.708661 deg
if country=="dmi":

    vars_to_plot = {"DBZH": [0, 45.5, 0.5], 
                    "ZDR_OC": [-0.505, 2.05, 0.1],
                    "KDP_ML_corrected":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                    "RHOHV_NC": [0.9, 1.002, 0.002]}
    
    if auto_plot:
        vtp = [{"DBZH": [0, 45.5, 0.5], 
                        "ZDR_OC": [-0.505, 2.05, 0.1],
                        "KDP_ML_corrected":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                        "RHOHV_NC": [0.9, 1.002, 0.002]},
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
            utils.hist2d(ax[nn], ds_to_plot[vv].sel(\
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
            fig.savefig(savepath+savename, bbox_inches="tight")
            print("AUTO PLOT: saved "+savename)



# DWD
# plot CFTDs moments
if country=="dwd":
    
    vars_to_plot = {"DBZH": [0, 46, 1],
                    "ZDR_OC": [-0.5, 2.1, 0.1],
                    "KDP_ML_corrected": [-0.1, 0.52, 0.02],
                    "RHOHV_NC": [0.9, 1.004, 0.004]}

    if auto_plot:
        vtp = [{"DBZH": [0, 46, 1], 
                        "ZDR_OC": [-0.5, 2.1, 0.1],
                        "KDP_ML_corrected":  [-0.1, 0.52, 0.02],
                        "RHOHV_NC": [0.9, 1.004, 0.004]},
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
            fig.savefig(savepath+savename, bbox_inches="tight")
            print("AUTO PLOT: saved "+savename)


#%% CFTDs retreivals Plot
# We assume that everything above ML is frozen and everything below is liquid

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly). 
# If False, then produce the plot as given below and do not save.
auto_plot = True 
savepath = "/automount/agradar/jgiles/images/CFTDs/stratiform/"

# Which to plot, stratiform or stratiform_relaxed
ds_to_plot = retrievals["stratiform"].copy()

loc = find_loc(locs, ff[0]) # by default, plot only the histograms of the currently loaded QVPs.

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
IWC = "iwc_zh_t" # iwc_zh_t or iwc_zdr_zh_kdp
LWC = "lwc_zh_zdr" # lwc_zh_zdr (adjusted for Germany) or lwc_zh_zdr2 (S-band) or lwc_kdp
Dm_ice = "Dm_ice_zh" # Dm_ice_zh or Dm_ice_zh_kdp
Dm_rain = "Dm_rain_zdr3" # Dm_rain_zdr, Dm_rain_zdr2 or Dm_rain_zdr3
Nt_ice = "Nt_ice_zh_iwc" # Nt_ice_zh_iwc
Nt_rain = "Nt_rain_zh_zdr" # Nt_rain_zh_zdr

vars_to_plot = {"IWC/LWC [g/m^{3}]": [-0.1, 0.82, 0.02], # [-0.1, 0.82, 0.02], 
                "Dm [mm]": [0, 3.1, 0.1], # [0, 3.1, 0.1],
                "log10(Nt) [1/L]": [-2, 2.1, 0.1], # [-2, 2.1, 0.1],
                }

savedict = {"custom": None} # placeholder for the for loop below, not important 

if auto_plot:
    ytlimlist = [-20, -50]
    add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
    savedict = {}
    for selseas in selseaslist:
        savedict.update( 
                    {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys.png": [ytlimlist[0], 
                                                           "iwc_zh_t", "lwc_zh_zdr", 
                                                           "Dm_ice_zh", "Dm_rain_zdr3", 
                                                           "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_extended.png": [ytlimlist[1],
                                                                "iwc_zh_t", "lwc_zh_zdr", 
                                                                "Dm_ice_zh", "Dm_rain_zdr3", 
                                                                "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP.png": [ytlimlist[0],
                                                               "iwc_zdr_zh_kdp", "lwc_kdp", 
                                                               "Dm_ice_zh_kdp", "Dm_rain_zdr3", 
                                                               "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP_extended.png": [ytlimlist[1],
                                                               "iwc_zdr_zh_kdp", "lwc_kdp", 
                                                               "Dm_ice_zh_kdp", "Dm_rain_zdr3", 
                                                               "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
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

    retreivals_merged = xr.Dataset({
                                    "IWC/LWC [g/m^{3}]": ds_to_plot[loc][IWC].where(ds_to_plot[loc][IWC].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][LWC].where(ds_to_plot[loc][LWC].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ),
                                    "Dm [mm]": ds_to_plot[loc][Dm_ice].where(ds_to_plot[loc][Dm_ice].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][Dm_rain].where(ds_to_plot[loc][Dm_rain].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ),
                                    "log10(Nt) [1/L]": (ds_to_plot[loc][Nt_ice].where(ds_to_plot[loc][Nt_ice].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][Nt_rain].where(ds_to_plot[loc][Nt_rain].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ) ),
        })
        
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
        utils.hist2d(ax[nn], retreivals_merged[vv].sel(time=retreivals_merged['time'].dt.month.isin(selmonths))*adj, 
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
        fig.savefig(savepath+savename, bbox_inches="tight")
        print("AUTO PLOT: saved "+savename)

#%% Check particular dates

# Plot QVP
visdict14 = radarmet.visdict14

def plot_qvp(data, momname="DBZH", tloc=slice("2015-01-01", "2020-12-31"), plot_ml=True, plot_entropy=False, **kwargs):
    mom=momname
    if "_" in momname:
        mom= momname.split("_")[0]
    # norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
    # cmap = mpl.cm.get_cmap("HomeyerRainbow")
    # cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
    ticks = radarmet.visdict14[mom]["ticks"]
    # cmap = visdict14[mom]["cmap"]
    cmap = "miub2"
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
            data["min_entropy"].loc[{"time":tloc}].dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[0.8,1], hatches=["","X"], colors="none", add_colorbar=False)
        except:
            print("Plotting entropy failed")
    plt.title(mom)

qvps_fix = qvps.copy().where(allcond)
# qvps_fix["KDP_ML_corrected"] = qvps_fix["KDP_ML_corrected"].where(qvps_fix.height_ml_new_gia.notnull(),  qvps_fix["KDP_CONV"])
with mpl.rc_context({'font.size': 10}):
    plot_qvp(qvps_fix, "ZDR_OC", tloc="2020-07-15", plot_ml=True, plot_entropy=True, ylim=(qvps.altitude,10000))


# qvps_strat_fil_notime = qvps_strat_fil.copy()
# qvps_strat_fil_notime = qvps_strat_fil_notime.reset_index("time")
# plot_qvp(qvps_strat_fil_notime, "ZDR_OC", tloc="2020-07-15", plot_ml=True, plot_entropy=True, ylim=(qvps.altitude,10000))

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
    for ll in ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
        if ll not in stats[stratname].keys():
            stats[stratname][ll] = {}
        elif type(stats[stratname][ll]) is not dict:
            stats[stratname][ll] = {}
        for xx in ['values_sfc', 'values_snow', 'values_rain', 'values_ML_max', 'values_ML_min', 'values_ML_mean', 
                   'ML_thickness', 'ML_bottom', 'ML_thickness_TEMP', 'ML_bottom_TEMP', 'values_DGL_max', 'values_DGL_min',
                   'values_DGL_mean', 'height_ML_max', 'height_ML_min', 'ML_bottom', 'beta', 'beta_belowML',
                   'cloudtop', 'cloudtop_5dbz', 'cloudtop_10dbz', 
                   'cloudtop_TEMP', 'cloudtop_TEMP_5dbz', 'cloudtop_TEMP_10dbz']:
            try:
                stats[stratname][ll][xx] = xr.open_dataset(realpep_path+"/upload/jgiles/radar_stats/"+stratname+"/"+ll+"_"+xx+".nc")
                if len(stats[stratname][ll][xx].data_vars)==1:
                    # if only 1 var, convert to data array
                    stats[stratname][ll][xx] = stats[stratname][ll][xx].to_dataarray() 
                print(ll+" "+xx+" stats loaded")
            except:
                pass
        # delete entry if empty
        if not stats[stratname][ll]:
            del stats[stratname][ll]

# load retrievals
if 'retrievals' not in globals() and 'retrievals' not in locals():
    retrievals = {}

for stratname in ["stratiform", "stratiform_relaxed"]:
    if stratname not in retrievals.keys():
        retrievals[stratname] = {}
    elif type(retrievals[stratname]) is not dict:
        retrievals[stratname] = {}
    print("Loading "+stratname+" retrievals ...")
    for ll in ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
        try:
            retrievals[stratname][ll] = xr.open_dataset(realpep_path+"/upload/jgiles/radar_retrievals/"+stratname+"/"+ll+".nc")
            print(ll+" retrievals loaded")
        except:
            pass
        # delete entry if empty
        if not retrievals[stratname][ll]:
            del retrievals[stratname][ll]

#%%% 2d histograms
locs_to_plot = [find_loc(locs, ff[0])] # by default, plot only the histograms of the currently loaded QVPs.
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

            #### Histograms: like Griffin et al 2020 https://doi.org/10.1175/JAMC-D-19-0128.1
            # plot histograms (2d hist) like Fig. 5
            try:
                plt.close()
                binsx = np.linspace(0.0, 4, 81)
                binsy = np.linspace(0.8, 1, 51)
                MLminRHOHVcurve = 0.97 - 0.028*binsx
                
                utils.hist_2d(MLmaxZDR.compute(), MLminRHOHV.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLminRHOHVcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.text(0.5, 0.82, r"$\mathregular{MLminRHOHV = 0.97 - 0.028\ MLmaxZDR }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_MinRHOHVinML_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + MLminRHOHV.compute().isnull() ).all():
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
                
                utils.hist_2d(MLmaxZH.compute(), np.log(MLmaxKDP).compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, logMLmaxKDPcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{log(Maximum \ K_{DP} \ in \ ML) \ [°/km]}$")
                plt.text(20, -3.8, r"$\mathregular{logMLmaxKDP = -3.21 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_logMaxKDPinML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + np.log(MLmaxKDP).compute().isnull() ).all():
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
                
                utils.hist_2d(MLmaxZH.compute(), ZHrain.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, ZHraincurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Z_{H} \ in \ rain \ [dBZ]}$")
                plt.text(-15, -15, r"$\mathregular{ZHrain = -2.74 + 1.03\ MLmaxZH - 0.005\ MLmaxZH^2 }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHinRain_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + ZHrain.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLmaxZH vs ZHrain for unknown reason !!! ")

            # plot b
            try:
                plt.close()
                binsx = np.linspace(-20, 50, 71)
                binsy = np.linspace(-20, 40, 61)
                ZHsnowcurve = -3.86 + 1.08*binsx - 0.0071*binsx**2
                
                utils.hist_2d(MLmaxZH.compute(), ZHsnow.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, ZHsnowcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Z_{H} \ in \ snow \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Z_{H} \ in \ rain \ [dBZ]}$")
                plt.text(-15, -15, r"$\mathregular{ZHsnow = -3.86 + 1.08\ MLmaxZH - 0.0071\ MLmaxZH^2 }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHinRain_ZHinSnow.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZHsnow.compute().isnull() + ZHrain.compute().isnull() ).all():
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
                
                utils.hist_2d(MLminRHOHV.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) \ [dBZ]´}$")
                plt.text(0.81, -8, r"$\mathregular{\Delta Z_H = 4.27 + 6.89(1-\rho _{HV}) + 341(1-\rho _{HV})^2 }$", fontsize="small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DeltaZH_MinRHOHVinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLminRHOHV.compute().isnull() + deltaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLminRHOHV due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLminRHOHV for unknown reason !!! ")
            
            # plot b
            try:
                plt.close()
                binsx = np.linspace(0, 4, 16*5+1)
                binsy = np.linspace(-10, 20, 61)
                deltaZHcurve = 3.18 + 2.19*binsx 
                
                utils.hist_2d(MLmaxZDR.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, deltaZHcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"$\mathregular{\Delta Z_H \ (MLmaxZ_H - Z_HRain) \ [dBZ] }$")
                plt.text(2, -8, r"$\mathregular{\Delta Z_H = 3.18 + 2.19 Z_{DR} }$", fontsize="small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DeltaZH_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + deltaZH.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLmaxZDR due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot deltaZH vs MLmaxZDR for unknown reason !!! ")

            # plot c
            try:
                plt.close()
                binsx = np.linspace(0.8, 1, 41)
                binsy = np.linspace(0, 1, 26)
                MLdepthcurve = -0.64 + 30.8*(1-binsx) - 315*(1-binsx)**2 + 1115*(1-binsx)**3 
                
                utils.hist_2d(MLminRHOHV.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Minimum \ \rho _{HV} \ in \ ML \ [-]}$")
                plt.ylabel(r"Depth of ML [km]")
                plt.text(0.82, 0.1, r"$\mathregular{ML \ Depth = -0.64 + 30.8\ (1-\rho _{HV})}$" "\n" r"$\mathregular{- 315\ (1-\rho _{HV})^2 + 1115\ (1-\rho _{HV})^3}$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MinRHOHVinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLminRHOHV.compute().isnull() + MLdepth.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLminRHOHV due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLminRHOHV for unknown reason !!! ")

            # plot d
            try:
                plt.close()
                binsx = np.linspace(0, 4, 16*5+1)
                binsy = np.linspace(0, 1, 26)
                MLdepthcurve = 0.21 + 0.091*binsx 
                
                utils.hist_2d(MLmaxZDR.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve") 
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{DR} \ in \ ML \ [dB]}$")
                plt.ylabel(r"Depth of ML [km]")
                plt.text(2.1, 0.05, r"$\mathregular{ML \ Depth = 0.21 + 0.091 Z_{DR} }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MaxZDRinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZDR.compute().isnull() + MLdepth.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLmaxZDR due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot MLdepth vs MLmaxZDR for unknown reason !!! ")

            # plot histograms (2d hist) like Fig. 11
            try:
                plt.close()
                binsx = np.linspace(-10, 50, 31)
                binsy = np.linspace(-0, 1.2, 21)
                MLdepthcurve = 0.315 + 0.000854*binsx
                
                utils.hist_2d(MLmaxZH.compute(), MLdepth.compute()/1000, bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, MLdepthcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{Depth\ of\ ML\ [km]}$")
                plt.text(11, 0.9, r"$\mathregular{MLdepth = 0.315 + 0.000854\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_DepthML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + ZHrain.compute().isnull() ).all():
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
                
                utils.hist_2d(MLmaxZH.compute(), np.log(MLmeanKDP).compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                plt.legend()
                plt.xlabel(r"$\mathregular{Maximum \ Z_{H} \ in \ ML \ [dBZ]}$")
                plt.ylabel(r"$\mathregular{log(Mean \ K_{DP} \ in \ ML) \ [°/km]}$")
                plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_logMeanKDPinML_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + np.log(MLmeanKDP).compute().isnull() ).all():
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
                
                utils.hist_2d(ZDRrain.compute(), ZDRsfc.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{DR} \ in \ rain \ [dB] }$")
                plt.ylabel(r"$\mathregular{ Z_{DR} \ at \ surface \ [dB] }$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZDRatSfc_ZDRinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZDRrain.compute().isnull() + ZDRsfc.compute().isnull() ).all():
                    print(" ... ... ... !!! not possible to plot ZDRrain vs ZDRsfc due to insufficient data points !!! ")
                else:
                    print(" ... ... ... !!! not possible to plot ZDRrain vs ZDRsfc for unknown reason !!! ")

            # plot right one
            try:
                plt.close()
                binsx = np.linspace(0, 40, 41)
                binsy = np.linspace(0, 40, 41)
                # logMLmeanKDPcurve = -2.4 + 0.05*binsx
                
                utils.hist_2d(ZHrain.compute(), ZHsfc.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Z_{H} \ in \ rain \ [dBZ] }$")
                plt.ylabel(r"$\mathregular{ Z_{H} \ at \ surface \ [dBZ] }$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_ZHatSfc_ZHinRain.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( ZHrain.compute().isnull() + ZHsfc.compute().isnull() ).all():
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
                
                utils.hist_2d(MLmaxZH.compute(), betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Maximum \ Z_{H} \ in \ ML \ [dBZ] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
                plt.grid()
                fig = plt.gcf()
                fig.savefig(savepath_seas+"/"+loc+"_betaZH_MaxZHinML.png",
                            bbox_inches="tight")
                plt.close(fig)
            except:
                if ( MLmaxZH.compute().isnull() + betaZH.compute().isnull() ).all():
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
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
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
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
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
                
                utils.hist_2d(cloudtop.compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ height \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
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
                
                utils.hist_2d(cloudtop_5dbz.compute()/1000, betaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
                # plt.plot(binsx, logMLmeanKDPcurve, c="black", label="Reference curve")
                # plt.legend()
                plt.xlabel(r"$\mathregular{ Cloud \ top \ height \ (5 dbZ) \ [km] }$")
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
                # plt.text(12, -0.5, r"$\mathregular{logMLmeanKDP = -2.4 + 0.05\ MLmaxZH }$", fontsize="xx-small")
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
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
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
                plt.ylabel(r"$\mathregular{ \beta \ [dB/km]}$")
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

ridge_vars = [X_DBZH, X_ZDR, X_RHO, X_KDP]

vars_ticks = {X_DBZH: np.arange(0, 46, 1), 
                X_ZDR: np.arange(-0.5, 2.1, 0.1),
                X_KDP: np.arange(-0.1, 0.52, 0.02),
                X_RHO: np.arange(0.9, 1.004, 0.004)
                }

beta_vars_ticks = {X_DBZH: np.linspace(-15, 10, int((10--15)/1)+1 ), 
                X_ZDR: np.linspace(-0.5, 1, int((1--0.5)/0.1)+1 ),
                X_KDP: np.linspace(-0.2, 0.2, int((0.2--0.2)/0.01)+1 ),
                X_RHO: np.linspace(-0.05, 0.05, int((0.05--0.05)/0.001)+1 ),
                }


bins = {"ML_thickness": np.arange(0,1200,50),
        "ML_thickness_TEMP": np.arange(0, 5.25, 0.25),
        "values_snow": vars_ticks,
        "values_rain": vars_ticks,
        "values_DGL_mean": vars_ticks,
        "values_DGL_min": vars_ticks,
        "values_DGL_max": vars_ticks,
        "values_ML_mean": vars_ticks,
        "values_ML_min": vars_ticks,
        "values_ML_max": vars_ticks,
        "values_sfc": vars_ticks,
        "cloudtop": np.arange(2000,10250,250),
        "cloudtop_5dbz": np.arange(2000,10250,250),
        "cloudtop_10dbz": np.arange(2000,10250,250),
        "beta": beta_vars_ticks,
        "cloudtop_TEMP": np.arange(-50,-20,1),
        "cloudtop_TEMP_5dbz": np.arange(-50,-20,1),
        "cloudtop_TEMP_10dbz": np.arange(-50,-20,1),
        "deltaZH": np.arange(-5,21,1),
        }



order = ['umd', 'pro', 'afy', 'ank', 'gzt', 'hty', 'svs']
order = ['umd', 'pro', 'hty']

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
            stats[stratname][ll]["deltaZH"] = stats[stratname][ll]["values_ML_max"]["DBZH"] - stats[stratname][ll]["values_rain"]["DBZH"]

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
        
                    if vv == "RHOHV_NCignore": # DEPRECATED (22.10.24), this was wrongly defined, it does nothing now.
                        order_turk = order_fil.copy()
                        order_turk.remove("pro") # THIS IS WRONG taking out DWD radars and putting them separate (because turkish radars do not have RHOHV_NC)
                        samples = [stats[stratname]["pro"][ss][vv].dropna("time").values] + [stats[stratname][loc][ss]["RHOHV"].dropna("time").values for loc in order_turk if loc in stats[stratname].keys()]
                    else:
                        samples = [stats[stratname][loc][ss][vv].sel(\
                                    time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                                   for loc in order_fil]
                        
                    fig = ridgeplot.ridgeplot(samples=samples,
                                            colorscale="viridis",
                                            colormode="row-index",
                                            coloralpha=0.65,
                                            labels=order_fil,
                                            linewidth=2,
                                            spacing=5 / 9,
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=16,
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    title=ss+" "+vv,
                                    xaxis_tickvals=bins[ss][vv],
                    )
    
                    # Add vertical zero line
                    fig.add_vline(x=0, line_width=2, line_color="gray")
                    
                    # Get densities data from the plot
                    densities = [ fig.data[2*i+1] for i in range(len(samples)) ]
                    
                    # calculate means
                    means = [np.mean(sample) for sample in samples]
                    
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
                    samples = [stats[stratname][loc][ss].sel(\
                                time=stats[stratname][loc][ss]['time'].dt.month.isin(selseas[1])).dropna("time").values\
                               for loc in order_fil]
                    fig = ridgeplot.ridgeplot(samples=samples, #bandwidth=50,
                                            colorscale="viridis",
                                            colormode="row-index",
                                            coloralpha=0.65,
                                            labels=order_fil,
                                            linewidth=2,
                                            spacing=5 / 9,
                                            )
                    fig.update_layout(
                                    height=760,
                                    width=900,
                                    font_size=16,
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
                    means = [np.mean(sample) for sample in samples]
                    
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
pro_vp_20170725 = dttree.open_datatree(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/90gradstarng01/00/ras07-90gradstarng01_sweeph5onem_allmoms_00-2017072500041700-pro-10392-hd5")["sweep_0"].to_dataset()
if pro_vp_20170725.time.isnull().any():
    pro_vp_20170725.coords["time"] = pro_vp_20170725["rtime"].min(dim="azimuth", skipna=True).compute()
loc="pro"
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
pro_vp_20170725 = utils.attach_ERA5_TEMP(pro_vp_20170725, path=loc.join(era5_dir.split("loc")))

zdr_offset_vp_pro_20170725 = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3).compute()

zdr_offset_vp_pro_20170725_azmedian = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3, azmed=True).compute()

# Let's find a not-nice case
pro_vp_20170126 = dttree.open_datatree(glob.glob(realpep_path+"/upload/jgiles/dwd/2016/2016-01/2016-01-26/pro/90gradstarng01/00/ras07-90gradstarng01*")[0])["sweep_0"].to_dataset()
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
pro20170725=dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()

#%% Load multiple elevations of DWD to check if there is better PHIDP
ff = realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/*/*allmoms*"

files = sorted(glob.glob(ff))

vollist = []
for fx in files:
    vollist.append(dttree.open_datatree(fx)["sweep_"+fx.split("/")[-2][1]].to_dataset())
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
        swpx = dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset().DBZH[0]
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
swpx = dttree.open_datatree(realpep_path+"/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/05/ras07-vol5minng01_sweeph5onem_allmoms_05-2017072500030000-pro-10392-hd5")["sweep_5"].to_dataset().DBZH[0]
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
    