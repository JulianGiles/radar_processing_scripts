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
import datetime
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

import towerpy

#%% Set paths and load

# Load preprocessed sweep and vertical scan
swppath="/automount/realpep/upload/jgiles/dwd/2017/2017-09/2017-09-09/pro/vol5minng01/01/*hd5"
vertpath="/automount/realpep/upload/jgiles/dwd/2017/2017-09/2017-09-09/pro/90gradstarng01/00/*hd5"

swpfile=sorted(glob.glob(swppath))[0]
vertfile=sorted(glob.glob(vertpath))[0]

swp=dttree.open_datatree(swpfile)["sweep_1"].to_dataset() 
vert=dttree.open_datatree(vertfile)["sweep_0"].to_dataset() 

# Load unprocessed sweep and vertical scan
# First a sweep
swp0path = "/automount/realpep/upload/jgiles/dwd_raw/2017/2017-09/2017-09-09/pro/vol5minng01/01/"
ll = sorted(glob.glob(swp0path+"/ras*hd5"))

# extract list of moments 
moments = set(fp.split("_")[-2] for fp in ll)

# discard "allmoms" from the set if it exists
moments.discard("allmoms")
