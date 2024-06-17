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


#%% Load QVPs for stratiform-case CFTDs
# This part should be run after having the QVPs computed (compute_qvps.py)

#### Get QVP file list
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# Load only events with ML detected (pre-condition for stratiform)
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps/20*/*/*/pro/vol5minng01/07/ML_detected.txt"
# path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps/2016/*/*/ANK/*/*/ML_detected.txt"
# path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps/*/*/*/ANK/*/*/ML_detected.txt"
# path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps_singlefile/ML_detected/ANK/*/12*/*allmoms*"
# path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*"
# path_qvps = ["/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*",
#              "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/14*/*allmoms*"]


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
# special_selection = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/DBZH_over_30*"
# special_filter = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/ML_detected_*"
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

#%% Filters (conditions for stratiform)
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

# Filter only stratiform events (min entropy >= 0.8 and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8).compute() & allcond, drop=True)
# Filter relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[X_TH] > 0 )&
                                  (qvps_strat[X_KDP] > -0.1)&
                                  (qvps_strat[X_KDP] < 3)&
                                  (qvps_strat[X_RHO] > 0.7)&
                                  (qvps_strat[X_ZDR] > -1) &
                                  (qvps_strat[X_ZDR] < 3))

try: 
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRHC"]>10)
except KeyError:
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRH"]>10)
except:
    print("Could not filter out low SNR")

#### Calculate retreivals

# to check the wavelength of each radar, in cm for DWD, in 1/100 cm for DMI ()
# filewl = ""
# xr.open_dataset(filewl, group="how") # DWD
# file1 = "/automount/realpep/upload/jgiles/dmi_raw/acq/OLDDATA/uza/RADAR/2015/01/01/ANK/RAW/ANK150101000008.RAW6M00"
# xd.io.backends.iris.IrisRawFile(file1, loaddata=False).ingest_header["task_configuration"]["task_misc_info"]["wavelength"]

Lambda = 53.1 # radar wavelength in mm (pro: 53.138, ANK: 53.1, AFY: 53.3, GZT: 53.3, HTY: 53.3, SVS:53.3)

# LWC 
lwc_zh_zdr = 10**(0.058*qvps_strat_fil[X_DBZH] - 0.118*qvps_strat_fil[X_ZDR] - 2.36) # Reimann et al 2021 (adjusted for Germany)
lwc_zh_zdr2 = 1.38*10**(-3) *10**(0.1*qvps_strat_fil[X_DBZH] - 2.43*qvps_strat_fil[X_ZDR] + 1.12*qvps_strat_fil[X_ZDR]**2 - 0.176*qvps_strat_fil[X_ZDR]**3 ) # Ryzhkov et al 2022, used in S band
lwc_kdp = 10**(0.568*np.log10(qvps_strat_fil[X_KDP]) + 0.06) # Reimann et al 2021(adjusted for Germany)

# IWC (Collected from Blanke et al 2023)
iwc_zh_t = 10**(0.06 * qvps_strat_fil[X_DBZH] - 0.0197*qvps_strat_fil["TEMP"] - 1.7) # empirical from Hogan et al 2006

iwc_zdr_zh_kdp = xr.where(qvps_strat_fil[X_ZDR]>0.4, # Carlin et al 2021
                          4*10**(-3)*( qvps_strat_fil[X_KDP]*Lambda/( 1-wrl.trafo.idecibel(qvps_strat_fil[X_ZDR])**-1 ) ), 
                          0.031474 * ( qvps_strat_fil[X_KDP]*Lambda )**0.66 * qvps_strat_fil[X_DBZH]**0.28 ) 

# Dm
Dm_ice_zh = 1.055*qvps_strat_fil[X_DBZH]**0.271 # Matrosov et al. (2019)
Dm_ice_zh_kdp = 0.67*( qvps_strat_fil[X_DBZH]/(qvps_strat_fil[X_KDP]*Lambda) )**(1/3) # Bukovcic et al. (2020)
Dm_rain_zdr = 0.3015*qvps_strat_fil[X_ZDR]**3 - 1.2087*qvps_strat_fil[X_ZDR]**2 + 1.9068*qvps_strat_fil[X_ZDR] + 0.5090 # (for rain but tuned for Germany X-band, JuYu Chen, Zdr in dB, Dm in mm)
Dm_rain_zdr2 = 0.171*qvps_strat_fil[X_ZDR]**3 - 0.725*qvps_strat_fil[X_ZDR]**2 + 1.48*qvps_strat_fil[X_ZDR] + 0.717 # (Hu and Ryzhkov 2022, used in S band data but could work for C band)
Dm_rain_zdr3 = xr.where(qvps_strat_fil[X_ZDR]<1.25, # Bringi et al 2009 (C-band)
                        0.0203*qvps_strat_fil[X_ZDR]**4 - 0.149*qvps_strat_fil[X_ZDR]**3 + 0.221*qvps_strat_fil[X_ZDR]**2 + 0.557*qvps_strat_fil[X_ZDR] + 0.801,
                        0.0355*qvps_strat_fil[X_ZDR]**3 - 0.302*qvps_strat_fil[X_ZDR]**2 + 1.06*qvps_strat_fil[X_ZDR] + 0.684
                        )

# log(Nt)
Nt_ice_zh_iwc = (3.39 + 2*np.log10(iwc_zh_t) - 0.1*qvps_strat_fil[X_DBZH]) # (Hu and Ryzhkov 2022, different than Carlin et al 2021 only in the offset, but works better)
Nt_rain_zh_zdr = ( -2.37 + 0.1*qvps_strat_fil[X_DBZH] - 2.89*qvps_strat_fil[X_ZDR] + 1.28*qvps_strat_fil[X_ZDR]**2 - 0.213*qvps_strat_fil[X_ZDR]**3 )# Hu and Ryzhkov 2022

# Put everything together
retreivals = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
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

#### General statistics
values_sfc = qvps_strat_fil.isel({"z": 2})
values_snow = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_new_gia"]}, method="nearest")
values_rain = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_bottom_new_gia"]}, method="nearest")
    
#### ML statistics
# select values inside the ML
qvps_ML = qvps_strat_fil.where( (qvps_strat_fil["z"] < qvps_strat_fil["height_ml_new_gia"]) & \
                               (qvps_strat_fil["z"] > qvps_strat_fil["height_ml_bottom_new_gia"]), drop=True)

values_ML_max = qvps_ML.max(dim="z")
values_ML_min = qvps_ML.min(dim="z")
values_ML_mean = qvps_ML.mean(dim="z")
ML_thickness = qvps_ML["height_ml_new_gia"] - qvps_ML["height_ml_bottom_new_gia"]

# Silke style
# select timesteps with detected ML
# gradient_silke = qvps_strat_fil.where(qvps_strat_fil["height_ml_new_gia"] > qvps_strat_fil["height_ml_bottom_new_gia"], drop=True)
# gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
# gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
# gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
# beta = gradient_final[X_TH] #### TH OR DBZH??


#### DGL statistics
# select values in the DGL 
qvps_DGL = qvps_strat_fil.where((qvps_strat_fil["TEMP"] >= -20)&(qvps_strat_fil["TEMP"] <= -10), drop=True)    

values_DGL_max = qvps_DGL.max(dim="z")
values_DGL_min = qvps_DGL.min(dim="z")
values_DGL_mean = qvps_DGL.mean(dim="z")

# Put everything in a dict
try: # check if exists, if not, create it
    stats
except NameError:
    stats = {}

stats[find_loc(locs, ff[0])] = {"values_sfc": values_sfc.compute().copy(),
                                   "values_snow": values_snow.compute().copy(),
                                   "values_rain": values_rain.compute().copy(),
                                   "values_ML_max": values_ML_max.compute().copy(),
                                   "values_ML_min": values_ML_min.compute().copy(),
                                   "values_ML_mean": values_ML_mean.compute().copy(),
                                   "ML_thickness": ML_thickness.compute().copy(),
                                   "values_DGL_max": values_DGL_max.compute().copy(),
                                   "values_DGL_min": values_DGL_min.compute().copy(),
                                   "values_DGL_mean": values_DGL_mean.compute().copy(),
    }

# Save stats
for ll in stats.keys():
    for xx in stats[ll].keys():
        stats[ll][xx].to_netcdf("/automount/realpep/upload/jgiles/radar_stats/stratiform/"+ll+"_"+xx+".nc")

#%% Statistics for Raquel

#### Scatterplots:
MLmaxZH = values_ML_max["DBZH"]
ZHrain = values_rain["DBZH"] # not exactly the same definition, but close
deltaZH = MLmaxZH - ZHrain
MLminRHOHV = values_ML_min["RHOHV_NC"]

MLdepth = ML_thickness

# plot scatterplot (2d hist)
# plot1
binsx = np.linspace(0.8, 1, 41)
binsy = np.linspace(-10, 20, 61)
deltaZHcurve = 4.27 + 6.89*(1-binsx) + 341*(1-binsx)**2 # curve from Ryzhkov

utils.hist_2d(MLminRHOHV.compute(), deltaZH.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
plt.plot(binsx, deltaZHcurve, c="black")

# plot2
binsy = np.linspace(0, 1000, 26)
MLdepthcurve = -0.64 + 30.8*(1-binsx) - 315*(1-binsx)**2 + 1115*(1-binsx)**3 # curve from Ryzhkov

utils.hist_2d(MLminRHOHV.compute(), MLdepth.compute(), bins1=binsx, bins2=binsy, cmap="Blues")
plt.plot(binsx, MLdepthcurve*1000, c="black") # multiply curve by 1000 to change from km to m

#### Values:
maxZHheight = qvps_ML["DBZH"].idxmax("z") # (2)
HmaxZH = qvps_ML["height_ml_bottom_new_gia"] + 0.8*MLdepth # (2)
HMLtopZH = qvps_ML["height_ml_bottom_new_gia"] + 1.6*MLdepth # (3)

qvps_aboveML = qvps_strat_fil.where( (qvps_strat_fil["z"] > qvps_strat_fil["height_ml_new_gia"]) & \
                                    (qvps_strat_fil["z"] < 6000) , drop=True)
vertgradZHaboveML = qvps_aboveML["DBZH"].differentiate("z").median("z") 

ZHmax = values_ML_max["DBZH"].mean().compute()

Hb = qvps_ML["height_ml_bottom_new_gia"].mean().compute()
Ht = qvps_ML["height_ml_new_gia"].mean().compute()

#%% CFTDs Plot

# adjustment from K to C (disabled now because I know that all qvps have ERA5 data)
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

cmaphist="Oranges"

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
        utils.hist2d(ax[nn], qvps_strat_fil[vv].round(rd), qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                     binsx=vars_to_plot[vv], binsy=[-20,16,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps, 
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)
        
        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts
    
    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')



# DWD
# plot CFTDs moments
if country=="dwd":
    
    vars_to_plot = {"DBZH": [0, 46, 1], 
                    "ZDR_OC": [-0.5, 2.1, 0.1],
                    "KDP_ML_corrected": [-0.1, 0.52, 0.02],
                    "RHOHV_NC": [0.9, 1.004, 0.004]}

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
        utils.hist2d(ax[nn], qvps_strat_fil[vv]*adj, qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                     binsx=vars_to_plot[vv], binsy=[-20,16,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps, 
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)
        
        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts
    
    
    
    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')


#%% CFTDs retreivals Plot
# We assume that everything above ML is frozen and everything below is liquid

IWC = "iwc_zh_t" # iwc_zh_t or iwc_zdr_zh_kdp
LWC = "lwc_zh_zdr" # lwc_zh_zdr (adjusted for Germany) or lwc_zh_zdr2 (S-band) or lwc_kdp
Dm_ice = "Dm_ice_zh" # Dm_ice_zh or Dm_ice_zh_kdp
Dm_rain = "Dm_rain_zdr3" # Dm_rain_zdr, Dm_rain_zdr2 or Dm_rain_zdr3
Nt_ice = "Nt_ice_zh_iwc" # Nt_ice_zh_iwc
Nt_rain = "Nt_rain_zh_zdr" # Nt_rain_zh_zdr

retreivals_merged = xr.Dataset({
                                "IWC/LWC [g/m^{3}]": retreivals[IWC].where(retreivals[IWC].z > retreivals.height_ml_new_gia,
                                                                  retreivals[LWC].where(retreivals[LWC].z < retreivals.height_ml_bottom_new_gia ) ),
                                "Dm [mm]": retreivals[Dm_ice].where(retreivals[Dm_ice].z > retreivals.height_ml_new_gia,
                                                                  retreivals[Dm_rain].where(retreivals[Dm_rain].z < retreivals.height_ml_bottom_new_gia ) ),
                                "log10(Nt) [1/L]": (retreivals[Nt_ice].where(retreivals[Nt_ice].z > retreivals.height_ml_new_gia,
                                                                  retreivals[Nt_rain].where(retreivals[Nt_rain].z < retreivals.height_ml_bottom_new_gia ) ) ),
    })


# if country=="dwd":

vars_to_plot = {"IWC/LWC [g/m^{3}]": [-0.1, 0.82, 0.02], # [-0.1, 0.82, 0.02], 
                "Dm [mm]": [0, 3.1, 0.1], # [0, 3.1, 0.1],
                "log10(Nt) [1/L]": [-2, 2.1, 0.1], # [-2, 2.1, 0.1],
                }

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
    utils.hist2d(ax[nn], retreivals_merged[vv]*adj, retreivals_merged["TEMP"]+adjtemp, whole_x_range=True, 
                 binsx=vars_to_plot[vv], binsy=[-20,16,tb], mode='rel_y', qq=0.2,
                 cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps, 
                 fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                 cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
    ax[nn].set_ylim(15,ytlim)
    ax[nn].set_xlabel(vv, fontsize=10)
    
    ax[nn].tick_params(labelsize=15) #change font size of ticks
    plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts



ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')


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


qvps_strat_fil_notime = qvps_strat_fil.copy()
qvps_strat_fil_notime = qvps_strat_fil_notime.reset_index("time")
plot_qvp(qvps_strat_fil_notime, "ZDR_OC", tloc="2020-07-15", plot_ml=True, plot_entropy=True, ylim=(qvps.altitude,10000))

#%% Statistics histograms
import ridgeplot

# load stats
if 'stats' not in globals() and 'stats' not in locals():
    stats = {}

for ll in ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
    if ll not in stats.keys():
        stats[ll] = {}
    elif type(stats[ll]) is not dict:
        stats[ll] = {}
    for xx in ['values_sfc', 'values_snow', 'values_rain', 'values_ML_max', 'values_ML_min', 'values_ML_mean', 'ML_thickness', 'values_DGL_max', 'values_DGL_min', 'values_DGL_mean']:
        try:
            stats[ll][xx] = xr.open_dataset("/automount/realpep/upload/jgiles/radar_stats/stratiform/"+ll+"_"+xx+".nc")
            if len(stats[ll][xx].data_vars)==1:
                # if only 1 var, convert to data array
                stats[ll][xx] = stats[ll][xx].to_dataarray() 
            print(ll+" "+xx+" stats loaded")
        except:
            pass
    # delete entry if empty
    if not stats[ll]:
        del stats[ll]

ridge_vars = [X_DBZH, X_ZDR, X_RHO, X_KDP]

vars_ticks = {X_DBZH: np.arange(0, 46, 1), 
                X_ZDR: np.arange(-0.5, 2.1, 0.1),
                X_KDP: np.arange(-0.1, 0.52, 0.02),
                X_RHO: np.arange(0.9, 1.004, 0.004)
                }

bins = {"ML_thickness": np.arange(0,1200,50),
        "values_snow": vars_ticks,
        "values_rain": vars_ticks,
        "values_DGL_mean": vars_ticks,
        "values_ML_mean": vars_ticks,
        }

# for loc in stats.keys():
#     for ss in bins.keys():
#         stats[loc][ss].plot.hist(bins=bins[ss], histtype='step',
#                                                weights=np.ones_like(stats[loc][ss])*100 / len(stats[loc][ss]))


order = ['pro', 'afy', 'ank', 'gzt', 'hty', 'svs']
for ss in bins.keys():
    print("plotting "+ss)
    try: 
        for vv in ridge_vars:

            if vv == "RHOHV_NC":
                order_turk = order.copy()
                order_turk.remove("pro")
                samples = [stats["pro"][ss][vv].dropna("time").values] + [stats[loc][ss]["RHOHV"].dropna("time").values for loc in order_turk if loc in stats.keys()]
            else:
                samples = [stats[loc][ss][vv].dropna("time").values for loc in order if loc in stats.keys()]

            fig = ridgeplot.ridgeplot(samples=samples,
                                    colorscale="viridis",
                                    colormode="row-index",
                                    coloralpha=0.65,
                                    labels=order,
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
            fig.write_html("/automount/agradar/jgiles/images/ridgeplots_stats/"+ss+"_"+vv+".html")            
    except: 
        samples = [stats[loc][ss].values for loc in order if loc in stats.keys()]
        fig = ridgeplot.ridgeplot(samples=samples,
                                colorscale="viridis",
                                colormode="row-index",
                                coloralpha=0.65,
                                labels=order,
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
        fig.write_html("/automount/agradar/jgiles/images/ridgeplots_stats/"+ss+".html")
    
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
paths= ["/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/ANK/*/12*/*below3C_timesteps-*",
        "/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/ANK/*/14*/*below3C_timesteps-*"]

files_auxlist=[]
for pp in paths:
    files_auxlist.extend(glob.glob(pp))
files_off = sorted(files_auxlist)

zdr_off_LR_below3C_timesteps = xr.open_mfdataset(files_off, combine="nested", concat_dim="time")


#%% Test ZDR calibration

# ZDR offset looks nice for a nice stratiform case
pro_vp_20170725 = dttree.open_datatree("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/90gradstarng01/00/ras07-90gradstarng01_sweeph5onem_allmoms_00-2017072500041700-pro-10392-hd5")["sweep_0"].to_dataset()
if pro_vp_20170725.time.isnull().any():
    pro_vp_20170725.coords["time"] = pro_vp_20170725["rtime"].min(dim="azimuth", skipna=True).compute()
loc="pro"
era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
pro_vp_20170725 = utils.attach_ERA5_TEMP(pro_vp_20170725, path=loc.join(era5_dir.split("loc")))

zdr_offset_vp_pro_20170725 = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3).compute()

zdr_offset_vp_pro_20170725_azmedian = utils.zdr_offset_detection_vps(pro_vp_20170725, min_h=400, timemode="all", mlbottom=3, azmed=True).compute()

# Let's find a not-nice case
pro_vp_20170126 = dttree.open_datatree(glob.glob("/automount/realpep/upload/jgiles/dwd/2016/2016-01/2016-01-26/pro/90gradstarng01/00/ras07-90gradstarng01*")[0])["sweep_0"].to_dataset()
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
ank_12_20180306 = xr.open_mfdataset("/automount/realpep/upload/jgiles/dmi/2018/2018-03/2018-03-06/ANK/MON_YAZ_G/14.0/*")
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
ff = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/ras07-vol5minng01_sweeph5onem_allmoms_07-2017072500033500-pro-10392-hd5"
pro20170725=dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()

#%% Load multiple elevations of DWD to check if there is better PHIDP
ff = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/*/*allmoms*"

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

files = [glob.glob("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/tur/vol5minng01/07/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/umd/vol5minng01/07/*allmoms*")[0],
         ]

files = [glob.glob("/automount/realpep/upload/jgiles/dmi/2015/2015-03/2015-03-03/ANK/MON_YAZ_K/12.0/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dmi/2020/2020-07/2020-07-02/AFY/VOL_B/10.0/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dmi/2016/2016-04/2016-04-07/GZT/MON_YAZ_C/12.0/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dmi/2016/2016-04/2016-04-07/HTY/MON_YAZ_C/12.0/*allmoms*")[0],
         glob.glob("/automount/realpep/upload/jgiles/dmi/2020/2020-01/2020-01-11/SVS/VOL_B/10.0/*allmoms*")[0],
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
# swpx = dttree.open_datatree("/automount/realpep/upload/jgiles/dwd/2016/2016-01/2016-01-01/pro/vol5minng01/07/ras07-vol5minng01_sweeph5onem_allmoms_07-2016010100034100-pro-10392-hd5")["sweep_7"].to_dataset().DBZH[0]
swpx = dttree.open_datatree("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/05/ras07-vol5minng01_sweeph5onem_allmoms_05-2017072500030000-pro-10392-hd5")["sweep_5"].to_dataset().DBZH[0]
# swpx = xr.open_dataset("/automount/realpep/upload/jgiles/dmi/2018/2018-03/2018-03-06/HTY/VOL_B/10.0/VOL_B-allmoms-10.0-2018-03-06-HTY-h5netcdf.nc").DBZH[0]
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
    