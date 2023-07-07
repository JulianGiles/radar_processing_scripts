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
swppath="/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/01/*hd5"
vertpath="/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/90gradstarng01/00/*hd5"
qvppath="/home/jgiles/dwd/qvps/2017/2017-07/2017-07-25/pro/vol5minng01/07/*nc"

swpfile=sorted(glob.glob(swppath))[0]
vertfile=sorted(glob.glob(vertpath))[0]
qvpfile=sorted(glob.glob(qvppath))[0]

swp=dttree.open_datatree(swpfile)["sweep_1"].to_dataset() 
vert=dttree.open_datatree(vertfile)["sweep_0"].to_dataset() 
qvp=xr.open_dataset(qvpfile)
#%% Load and attach temperature data (for ML estimation)

vert = utils.attach_ERA5_TEMP(vert, site="pro")

#%% Calculate offset

zdr_offset = utils.zdr_offset_detection_vps(vert, mlbottom=5).compute()

#%% TEST METHOD FOR VARIOUS PARAMETERS

# Calculate ML from the VP
utils.melting_layer_qvp_X_new(vert.median("azimuth").swap_dims({"range":"z"}), dim="z")
!!!!!!!! ME QUEDE ACA, QUIERO CALCULAR LA ML BOTTOM DEL VP PARA APLICARLO AL METODO Y COMPARAR, PERO PRIMERO TENGO QUE ARREGLAR PHIDP
utils.phidp_offset_detection(vert, phidp="UPHIDP")
    off = phidp_offset["PHIDP_offset"]
    start_range = phidp_offset["start_range"]
    
    # apply offset
    fix_range = 750
    phi_fix = ds[X_PHIDP].copy()
    off_fix = off.broadcast_like(phi_fix)
    phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off


# Plot options
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
