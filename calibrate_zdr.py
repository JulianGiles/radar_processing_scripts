#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:16 2023

@author: jgiles

Script for calculating ZDR calibration from vertical (birdbath) scans
"""


import os
os.chdir('/home/jgiles/')


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import datatree as dttree
import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import datetime as dt
import pandas as pd
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


#%% Set paths and load

# Load preprocessed sweep and vertical scan (testing)
date="2017-01-11"
swppath="/automount/realpep/upload/jgiles/dwd/"+date[0:4]+"/"+date[0:7]+"/"+date+"/pro/vol5minng01/01/*hd5"
vertpath="/automount/realpep/upload/jgiles/dwd/"+date[0:4]+"/"+date[0:7]+"/"+date+"/pro/90gradstarng01/00/*hd5"
qvppath="/home/jgiles/dwd/qvps/"+date[0:4]+"/"+date[0:7]+"/"+date+"/pro/vol5minng01/07/*nc"

swpfile=sorted(glob.glob(swppath))[0]
vertfile=sorted(glob.glob(vertpath))[0]
qvpfile=sorted(glob.glob(qvppath))[0]

swp=dttree.open_datatree(swpfile)["sweep_1"].to_dataset() 
vert=dttree.open_datatree(vertfile)["sweep_0"].to_dataset() 
qvp=xr.open_dataset(qvpfile)
#%% Load and attach temperature data (for ML estimation)

vert = utils.attach_ERA5_TEMP(vert, site="pro")

#%% Calculate ZDR offset

# First calculate PHIDP offset, then calculate ML
phidp_offset = utils.phidp_offset_detection(vert, phidp="UPHIDP")
off = phidp_offset["PHIDP_OFFSET"]
start_range = phidp_offset["start_range"]
    
# apply offset
fix_range = 750
phi_fix = vert["UPHIDP"].copy()
off_fix = off.broadcast_like(phi_fix)
phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off

vert = vert.assign({"UPHIDP_OC": phi_fix.assign_attrs(vert["UPHIDP"].attrs)})

# Calculate ML from the VP
moments={"DBZH": (10., 60.), "RHOHV": (0.65, 1.), "UPHIDP_OC": (-20, 360)}
ml = utils.melting_layer_qvp_X_new(vert.where(vert["z"]>700).median("azimuth", keep_attrs=True)\
                                   .assign_coords({"z":vert["z"].median("azimuth", keep_attrs=True)})\
                                   .swap_dims({"range":"z"}),
                                   dim="z", moments=moments, all_data=True)

#### Giagrande refinment
hdim = "z"
# get data iside the currently detected ML
cut_above = ml.where(ml["z"]<ml.mlh_top)
cut_above = cut_above.where(ml["z"]>ml.mlh_bottom)

# get the heights with min RHOHV
min_height_ML = cut_above["RHOHV"].idxmin(dim="z") 

# cut the data below and above the previous value
new_cut_below_min_ML = ml.where(ml["z"] > min_height_ML)
new_cut_above_min_ML = ml.where(ml["z"] < min_height_ML)

# Filter out values outside some RHOHV range
new_cut_below_min_ML_filter = new_cut_below_min_ML["RHOHV"].where((new_cut_below_min_ML["RHOHV"]>=0.97)&(new_cut_below_min_ML["RHOHV"]<=1))
new_cut_above_min_ML_filter = new_cut_above_min_ML["RHOHV"].where((new_cut_above_min_ML["RHOHV"]>=0.97)&(new_cut_above_min_ML["RHOHV"]<=1))            


######### ML TOP Giangrande refinement

notnull = new_cut_below_min_ML_filter.notnull() # this replaces nan for False and the rest for True
first_valid_height_after_ml = notnull.where(notnull).idxmax(dim=hdim) # get the first True value, i.e. first valid value

######### ML BOTTOM Giangrande refinement
# For this one, we need to flip the coordinate so that it is actually selecting the last valid index
notnull = new_cut_above_min_ML_filter.notnull() # this replaces nan for False and the rest for True
last_valid_height = notnull.where(notnull).isel({hdim:slice(None, None, -1)}).idxmax(dim=hdim) # get the first True value, i.e. first valid value (flipped)


ml = ml.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
ml = ml.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))




zdr_offset = utils.zdr_offset_detection_vps(vert, min_h=600, mlbottom=5).compute()

#%% TEST METHOD FOR VARIOUS PARAMETERS

# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different temperature levels
mom = "DBZH"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
templevels = [3,5]

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ml[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = vert["TEMP"][:, 0, :].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
qvp["height_ml_bottom_new_gia"].plot(color="white", label="ML from QVP")
qvp["height_ml_new_gia"].plot(color="white")
ml["height_ml_bottom_new_gia"].plot(color="black", label="ML from VP")
ml["height_ml_new_gia"].plot(color="black")
plt.legend()
plt.title(mom)
plt.ylabel("height [m]")

ax2=plt.subplot(gs[1], sharex=ax)
ax3 = ax2.twinx()
for nt, tv in enumerate(templevels):
    zdr_offset = utils.zdr_offset_detection_vps(vert, min_h=600, mlbottom=tv).compute()
    zdr_offset["ZDR_offset"].plot(label=str(tv), ls=[":", ":", ":"][nt], ax=ax2, ylim=(-0.2,1))
    zdr_offset["ZDR_std_from_offset"].plot(label=str(tv), ls=["--", "--", ":"][nt], ax=ax3, ylim=(-0.2,1))
ax2.set_title("")
ax3.set_title("Dotted: offset. Dashed: offset std.")
plt.legend(loc=(1.01,0.1))

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.5])
fig.colorbar(figvp, cax=cbar_ax)



# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different number of valid bins below ML
mom = "ZDR"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
templevels = [3,5]

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ml[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = vert["TEMP"][:, 0, :].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
qvp["height_ml_bottom_new_gia"].plot(color="white", label="ML from QVP")
qvp["height_ml_new_gia"].plot(color="white")
ml["height_ml_bottom_new_gia"].plot(color="black", label="ML from VP")
ml["height_ml_new_gia"].plot(color="black")
plt.legend()
plt.title(mom)
plt.ylabel("height [m]")

minbins=[10,100, 1000]
vert_ml = vert.assign_coords({"height_ml_new_gia": ml["height_ml_new_gia"], 
                              "height_ml_bottom_new_gia": ml["height_ml_bottom_new_gia"]})
ax2=plt.subplot(gs[1], sharex=ax)
ax3 = ax2.twinx()
for nt, mb in enumerate(minbins):
    zdr_offset = utils.zdr_offset_detection_vps(vert_ml, min_h=600, minbins=mb).compute()
    zdr_offset["ZDR_offset"].plot(label=str(mb), ls=[":", ":", ":"][nt], ax=ax2, ylim=(-0.2,1))
    zdr_offset["ZDR_std_from_offset"].plot(label=str(mb), ls=["--", "--", ":"][nt], ax=ax3, ylim=(-0.2,1))
ax2.set_title("")
ax3.set_title("Dotted: offset. Dashed: offset std.")
plt.legend(loc=(1.01,0.1))

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.5])
fig.colorbar(figvp, cax=cbar_ax)





# Plot only lines, no moment VP
templevels = [3,5,7]
for nt, tv in enumerate(templevels):
    zdr_offset = utils.zdr_offset_detection_vps(vert, mlbottom=tv).compute()
    zdr_offset["ZDR_offset"].plot(label=str(tv), ls=["-", ":", ":"][nt])
plt.title("")
ax = plt.gca()
plt.legend()
ax2 = ax.twinx()
fig = vert["TEMP"][:, 0, :].plot.contour(x="time", y="z", ax=ax2, levels=[0]+templevels, ylim=(0,5000))
ax2.clabel(fig)
qvp["height_ml_bottom_new_gia"].plot()
plt.title("Offsets detected as function of temperature level")
