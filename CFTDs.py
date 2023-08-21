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


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

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

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    import colormap_generator



import warnings
warnings.filterwarnings('ignore')

#%% Load QVPs for stratiform-case CFTDs
# This part should be run after having the QVPs computed (compute_qvps.py)

#### Get QVP file list
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps_singlefile/ML_detected/umd/vol5minng01/07/*allmoms*"
# path_qvps = "/automount/realpep/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps/*/*/*/ANK/*/*/*allmoms*"
# path_qvps = "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*"
# path_qvps = ["/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/12*/*allmoms*",
#              "/automount/realpep/upload/jgiles/dmi/qvps_monthly/*/*/ANK/*/14*/*allmoms*"]

# ## Special selection of dates based on ML_detected.txt or some other criteria (works but still quite slow)
# special_selection = "/automount/realpep/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/ML*"
# path_special = glob.glob(special_selection)
# path_qvps = []
# for ff in path_special:
#     path_qvps.append(os.path.dirname(ff)+"/*allmoms*")

if isinstance(path_qvps, str):
    files = sorted(glob.glob(path_qvps))
elif len(path_qvps)==1:    
    files = sorted(glob.glob(path_qvps[0]))
else:
    files = []
    for fglob in path_qvps:
        files.extend(glob.glob(fglob))

#### Set variable names
X_DBZH = "DBZH"
X_RHOHV = "RHOHV_NC"
X_ZDR = "ZDR_OC"
X_KDP = "KDP_ML_corrected"

if "dwd" in files[0]:
    country="dwd"
    X_TH = "TH"
elif "dmi" in files[0]:
    country="dmi"
    X_TH = "DBZH"

if len(files)==1:
    qvps = xr.open_mfdataset(files)
else:
    # there are slight differences (noise) in z coord sometimes so we have to align all datasets
    # since the time coord has variable length, we cannot use join="override" so we define a function to copy
    # the z coord from the first dataset into the rest with preprocessing
    # There are also some time values missing, ignore those
    # Some files do not have TEMP data, fill with nan
    first_file = xr.open_mfdataset(files[0]) 
    first_file_z = first_file.z.copy()
    def fix_z_and_time(ds):
        ds.coords["z"] = first_file_z
        ds = ds.where(ds["time"].notnull(), drop=True)
        if "TEMP" not in ds.coords:
            ds.coords["TEMP"] = xr.full_like( ds["DBZH"], np.nan ).compute()
            
        return ds
        
    try:
        qvps = xr.open_mfdataset(files, preprocess=fix_z_and_time)
    except: 
        # if the above fails, just combine everything and fill the holes with nan (Turkish case)
        qvps = xr.open_mfdataset(files, combine="nested", concat_dim="time")

# Fill missing values in ZDR_OC and RHOHV_NC with the uncorrected variables
if X_ZDR == "ZDR_OC":
    qvps[X_ZDR] = qvps[X_ZDR].where(qvps[X_ZDR].notnull(), qvps["ZDR"])
if X_RHOHV == "RHOHV_NC":
    qvps[X_RHOHV] = qvps[X_RHOHV].where(qvps[X_RHOHV].notnull(), qvps["RHOHV"])

#%% Filters (conditions for stratiform)
# Filter only stratiform events (min entropy >= 0.8) and ML detected
# with ProgressBar():
#     qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull(), drop=True).compute()

# Filter only stratiform events (min entropy >= 0.8 and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull()), drop=True)
# Filter relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[X_TH] > 0 )&
                                  (qvps_strat[X_KDP] > -0.1)&
                                  (qvps_strat[X_RHOHV] > 0.7)&
                                  (qvps_strat[X_ZDR] > -1) &
                                  (qvps_strat[X_ZDR] < 3))

try: 
    qvps_strat_fil = qvps_strat_fil.where(qvps_strat_fil["SNRHC"]>10)
except:
    print("Could not filter out low SNR")

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
gradient_silke = qvps_strat_fil.where(qvps_strat_fil["height_ml_new_gia"] > qvps_strat_fil["height_ml_bottom_new_gia"], drop=True)
gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
beta = gradient_final[X_TH] #### TH OR DBZH??


#### DGL statistics
# select values in the DGL 
qvps_DGL = qvps_strat_fil.where((qvps_strat_fil["TEMP"] >= -20)&(qvps_strat_fil["TEMP"] <= -10), drop=True)    

values_DGL_max = qvps_DGL.max(dim="z")
values_DGL_min = qvps_DGL.min(dim="z")
values_DGL_mean = qvps_DGL.mean(dim="z")

#%% CFADs Plot

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
                    "KDP":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.013],
                    "RHOHV": [0.9, 1.002, 0.002]}
    
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size
    
    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        rd=10 # arbitrarily large decimal position to round to (so it is actually not rounded)
        if vv == "DBZH":
            so=True
            binsx2 = [0, 46, 1]
            rd = 1 # decimal position to round to
        if vv == "ZDR_OC":
            so=True
            binsx2 = [-0.5, 2.1, 0.1]
            rd=1
        if vv == "KDP":
            so=True #True
            binsx2 = [-0.1, 0.51, 0.01]
            rd=2
        if "RHOHV" in vv:
            so = True
            binsx2 = [0.9, 1.005, 0.005]
            rd=3
        utils.hist2d(ax[nn], qvps_strat_fil[vv].round(rd), qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                     binsx=vars_to_plot[vv], binsy=[-20,15,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps, 
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)
        
        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts
    
    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')



# DWD
if country=="dwd":
    
    vars_to_plot = {"DBZH": [0, 46, 1], 
                    "ZDR_OC": [-0.5, 2.1, 0.1],
                    "KDP_ML_corrected": [-0.1, 0.41, 0.01],
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
                     binsx=vars_to_plot[vv], binsy=[-20,15,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps, 
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
    norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
    # cmap = mpl.cm.get_cmap("HomeyerRainbow")
    # cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
    cmap = visdict14[mom]["cmap"]

    data[momname].loc[{"time":tloc}].dropna("z", how="all").plot(x="time", cmap=cmap, norm=norm, extend="both", **kwargs)
    if plot_ml:
        try:
            data.loc[{"time":tloc}].height_ml_bottom_new_gia.plot(color="black")
            data.loc[{"time":tloc}].height_ml_bottom_new_gia.plot(color="white",ls=":")
            data.loc[{"time":tloc}].height_ml_new_gia.plot(color="black")
            data.loc[{"time":tloc}].height_ml_new_gia.plot(color="white",ls=":")
        except KeyError:
            print("No ML in data")
    if plot_entropy:
        try:
            data["min_entropy"].loc[{"time":tloc}].dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[0.8,1], hatches=["","X"], colors="none", add_colorbar=False)
        except:
            print("Plotting entropy failed")

plot_qvp(qvps, "KDP_ML_corrected", tloc="2017-07-25", plot_ml=True, plot_entropy=True, ylim=(0,10000))


qvps_strat_fil_notime = qvps_strat_fil.copy()
qvps_strat_fil_notime = qvps_strat_fil_notime.reset_index("time")
plot_qvp(qvps_strat_fil_notime, "KDP_ML_corrected", plot_ml=True, plot_entropy=True, ylim=(2000,10000))

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
