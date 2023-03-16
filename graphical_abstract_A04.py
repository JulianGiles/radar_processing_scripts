#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:01:02 2023

@author: jgiles
"""

import wradlib

import matplotlib

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs	
import cartopy.feature 	
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from numpy import ma 
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.colors as mcolors
import os
os.chdir('/home/jgiles/radarmeteorology/notebooks/')
from matplotlib import gridspec
import scipy.stats
from scipy import signal
import gc
from dask.diagnostics import ProgressBar
import cordex as cx

# load TSMP only for the grid 
tsmp = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/2002_02/*TOT_PREC*',
                             )
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))


proj = cartopy.crs.Orthographic(central_longitude=15.0, central_latitude=10, globe=None)
ax = plt.axes(projection = proj)
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
plt.gca().coastlines('50m')

# ax.add_feature(cartopy.feature.BORDERS)
# ax.gridlines( draw_labels={"bottom": "x", "left": "y"})
ax.gridlines(xlocs=np.arange(-180,180,5), ylocs=np.arange(-90,90,5), ls=(0, (1,3)), lw=1)

# set extent of map
lon_limits = [-20,50] # [25, 45] [6,15]
lat_limits = [30, 90] # [35.5, 42.5] [47, 55]
ax.set_extent([lon_limits[0], lon_limits[-1], lat_limits[0], lat_limits[-1]])
ax.set_global()


# (tsmp.TOT_PREC[0]*0).plot(ax=ax, cmap='Greys', levels = 21, vmin=-5, vmax=20,transform = rp)

# plot scatter CORDEX grid
# gs = 10
# rlongrid, rlatgrid = np.meshgrid(tsmp.rlon, tsmp.rlat)
# ax.scatter(rlongrid[::gs, ::gs], rlatgrid[::gs, ::gs], transform=rp, s=0.01)




shpfilename = cartopy.io.shapereader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = cartopy.io.shapereader.Reader(shpfilename)
        
germany = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Germany"][0]
# Display germany's shape
shape_feature = cartopy.feature.ShapelyFeature([germany.geometry], ccrs.PlateCarree(), facecolor="#226580", edgecolor='black', lw=1)
ax.add_feature(shape_feature)

turkey = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Turkey"][0]
# Display turkey's shape
shape_feature = cartopy.feature.ShapelyFeature([turkey.geometry], ccrs.PlateCarree(), facecolor="#932e2e", edgecolor='black', lw=1)
ax.add_feature(shape_feature)


# # Make figure larger
plt.gcf().set_size_inches(20, 10)

plt.savefig("/home/jgiles/sciebo/website information/map.svg")