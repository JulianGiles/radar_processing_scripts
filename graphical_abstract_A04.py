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

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator


#%% Europe map with coverage

# load TSMP only for the grid
tsmp = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/2002_02/*TOT_PREC*',
                             )

# load euradclim just for the coverage
euradclim = xr.open_dataset("/automount/agradar/jgiles/EURADCLIM/concat_files/RAD_OPERA_HOURLY_RAINFALL_202008.nc")

# load eobs just for the coverage
eobs = xr.open_dataset("/automount/ags/jgiles/E-OBS/RR/rr_ens_mean_0.25deg_reg_v29.0e.nc")

# Define EURADCLIM coverage
region_euradclim =["Portugal", "Spain", "France", "United Kingdom", "Ireland",
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra",
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
region_name = "EURADCLIM" # "Europe_EURADCLIM" # name for plots
mask = utils.get_regionmask(region_euradclim)


# Get edges of Euro CORDEX
# Compute the values to handle dask arrays
lon = tsmp.lon.compute()
lat = tsmp.lat.compute()

# Extract the edges of the lon/lat grid (all four edges of the array)
lon_bottom_edge = lon.isel(rlat=0)                    # bottom edge
lat_bottom_edge = lat.isel(rlat=0)

lon_top_edge = lon.isel(rlat=-1)                      # top edge
lat_top_edge = lat.isel(rlat=-1)

lon_left_edge = lon.isel(rlon=0).isel(rlat=slice(1, -1))  # left edge (excluding corners)
lat_left_edge = lat.isel(rlon=0).isel(rlat=slice(1, -1))

lon_right_edge = lon.isel(rlon=-1).isel(rlat=slice(1, -1))  # right edge (excluding corners)
lat_right_edge = lat.isel(rlon=-1).isel(rlat=slice(1, -1))



# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25)
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))


proj = cartopy.crs.Orthographic(central_longitude=15.0, central_latitude=40, globe=None)
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
shape_feature = cartopy.feature.ShapelyFeature([germany.geometry], ccrs.PlateCarree(), facecolor="orange", edgecolor='black', lw=1)
ax.add_feature(shape_feature)

turkey = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Turkey"][0]
# Display turkey's shape
shape_feature = cartopy.feature.ShapelyFeature([turkey.geometry], ccrs.PlateCarree(), facecolor="#E30A17", edgecolor='black', lw=1)
ax.add_feature(shape_feature)

# euradclim_countries = [country for country in reader.records() if country.attributes["NAME_LONG"] in region_euradclim+["Czech Republic"]]
# # Display EURADCLIM coverage
# for country in euradclim_countries:
#     shape_feature = cartopy.feature.ShapelyFeature([country.geometry], ccrs.PlateCarree(), facecolor="#5EA266", edgecolor='black', lw=1, alpha=0.5)
#     ax.add_feature(shape_feature)


# Step 3: Plot edges of EURO-CORDEX
ax.plot(lon_bottom_edge, lat_bottom_edge, transform=ccrs.PlateCarree(), color='#0b5394', linewidth=2, linestyle='--')
ax.plot(lon_top_edge, lat_top_edge, transform=ccrs.PlateCarree(), color='#0b5394', linewidth=2, linestyle='--')
ax.plot(lon_left_edge, lat_left_edge, transform=ccrs.PlateCarree(), color='#0b5394', linewidth=2, linestyle='--')
ax.plot(lon_right_edge, lat_right_edge, transform=ccrs.PlateCarree(), color='#0b5394', linewidth=2, linestyle='--', label='Dataset Boundary')


# # Plot euradclim coverage as points
# euradclim_valid = euradclim.Precip[0].notnull()
# ax.scatter(euradclim.lon.where(euradclim_valid).values.flatten()[::5],
#            euradclim.lat.where(euradclim_valid).values.flatten()[::5],
#            transform=ccrs.PlateCarree(), color="#1d7e75ff", zorder=1)



# # Plot eobs coverage as points
# eobs_valid = eobs.rr[0].notnull()
# eobs_valid.where(eobs_valid).plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, alpha=0.5, cmap="Oranges")
# # ax.scatter(euradclim.lon.where(euradclim_valid).values.flatten()[::5], euradclim.lat.where(euradclim_valid).values.flatten()[::5], transform=ccrs.PlateCarree())


# # Plot 60 N line
# ax.plot(np.arange(0,361), [60]*361, transform=ccrs.PlateCarree(), color="#cc0000")

# Remove title
plt.title("")

# # Make figure larger
plt.gcf().set_size_inches(20, 10)

# plt.savefig("/automount/agradar/jgiles/images/gridded_datasets_intercomparison/custom/spatial_coverage_globe.png",
#             transparent=True, bbox_inches="tight")


#%% floating grid for DETECT video
from matplotlib.collections import LineCollection

# color and width of lines
lc = "white"
lw = 2

# crop grid
lat_min = 0 #latitude index to plot only part of the grid (use 210 for around half)

proj = cartopy.crs.Orthographic(central_longitude=11.6, central_latitude=4.5, globe=None)
ax = plt.axes(projection = proj)
# ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
# plt.gca().coastlines('50m')
ax.spines['geo'].set_visible(False)
ax.axis('off')
# ax.add_feature(cartopy.feature.BORDERS)
# ax.gridlines( draw_labels={"bottom": "x", "left": "y"})
# ax.gridlines(xlocs=np.arange(-180,180,5), ylocs=np.arange(-90,90,5), ls=(0, (1,3)), lw=1)

# set extent of map
lon_limits = [-30,50] # [25, 45] [6,15]
lat_limits = [30, 90] # [35.5, 42.5] [47, 55]
ax.set_extent([lon_limits[0], lon_limits[-1], lat_limits[0], lat_limits[-1]])
# ax.set_global()

# # Step 3: Plot grid
# for n in np.arange(0, len(lon.rlat), 10):
#     ax.plot(lon.isel(rlat=n)[:-6], lat.isel(rlat=n)[:-6], transform=ccrs.PlateCarree(), color='black', linewidth=2)

# for n in np.arange(0, len(lon.rlon), 10):
#     ax.plot(lon.isel(rlon=n)[:-4], lat.isel(rlon=n)[:-4], transform=ccrs.PlateCarree(), color='black', linewidth=2)

# --- Helper Function for 3D Layers ---
def plot_at_height(ax, lons, lats, height=1.1, **kwargs):
    """
    Plots data at a simulated height above the globe.
    height: 1.0 is surface, 1.1 is 10% higher, etc.
    """
    # 1. Project Lat/Lon to Grid Coordinates (x, y)
    # transform_points returns (x, y, z), we only need x, y
    projected = ax.projection.transform_points(ccrs.PlateCarree(), np.array(lons), np.array(lats))
    x = projected[:, 0]
    y = projected[:, 1]

    # 2. Scale the coordinates radially from the center (0,0)
    x_scaled = x * height
    y_scaled = y * height

    # 3. Plot using 'transData' (tells matplotlib these are raw x,y coords)
    ax.plot(x_scaled, y_scaled, transform=ax.transData, **kwargs)

height_ = 0.03 # % of height increase
nlevels = 3
alpha_ = 0.5
linesstep = 30

# 1. Plot Latitude Lines
for n in np.arange(lat_min, len(lon.rlat), linesstep):
    lons_vals = lon.isel(rlat=n)[:-(len(lon.rlon)%linesstep)]
    lats_vals = lat.isel(rlat=n)[:-(len(lon.rlon)%linesstep)]

    # Base Layer (Surface)
    ax.plot(lons_vals, lats_vals, transform=ccrs.PlateCarree(),
            color=lc, linewidth=lw, alpha=alpha_)

    for nl in np.arange(1, nlevels):
        # Floating Layer (e.g., 20% higher)
        plot_at_height(ax, lons_vals, lats_vals, height=1+nl*height_,
                       color=lc, linewidth=lw, alpha=alpha_)

# 2. Plot Longitude Lines
for n in np.arange(0, len(lon.rlon), linesstep):
    lons_vals = lon.isel(rlon=n)[lat_min:-(len(lon.rlat)%linesstep)]
    lats_vals = lat.isel(rlon=n)[lat_min:-(len(lon.rlat)%linesstep)]

    # Base Layer
    ax.plot(lons_vals, lats_vals, transform=ccrs.PlateCarree(),
            color=lc, linewidth=lw, alpha=alpha_)

    for nl in np.arange(1, nlevels):
        # Floating Layer
        plot_at_height(ax, lons_vals, lats_vals, height=1+nl*height_,
                       color=lc, linewidth=lw, alpha=alpha_)

# --- Plot Vertical Pillars (The "Cube" edges) ---
# We find the intersection points and draw lines connecting Surface to Ceiling.
# using LineCollection is much faster than plt.plot() in a loop.

segments = []

# Create a meshgrid of indices to find intersections
# Adjust the ranges/slices ([:-6], [:-4]) to match your data exactly
rlat_indices = np.arange(lat_min, len(lon.rlat), linesstep)
rlon_indices = np.arange(0, len(lon.rlon), linesstep)

for r in rlat_indices:
    for c in rlon_indices:
        # Check if indices fall within your slice limits
        # (Assuming your previous [:-6] logic meant "stop 6 from the end")
        if c >= len(lon.rlon) - (len(lon.rlon)%linesstep) +1 or r >= len(lon.rlat) - (len(lon.rlat)%linesstep)+1:
            continue

        # Get the coordinate of the grid intersection
        # Note: syntax depends on your data structure, assuming 2D xarray here
        p_lon = float(lon.isel(rlat=r, rlon=c))
        p_lat = float(lat.isel(rlat=r, rlon=c))

        # Project to (x, y)
        p_proj = proj.transform_point(p_lon, p_lat, ccrs.PlateCarree())
        x, y = p_proj[0], p_proj[1]

        # Define the line segment: [(x_surf, y_surf), (x_ceil, y_ceil)]
        segments.append([(x, y), (x * (1+(nlevels-1)*height_), y * (1+(nlevels-1)*height_))])

# Create the collection and add to plot
lcoll = LineCollection(segments, colors=lc, linewidths=lw, alpha=alpha_, transform=ax.transData)
ax.add_collection(lcoll)

# Remove title
plt.title("")

# # Make figure larger
plt.gcf().set_size_inches(16, 9)

plt.savefig("/home/jgiles/sciebo/Summary presentations DETECT A04/DETECT videos/model_grid.png",
            transparent=True, bbox_inches="tight", dpi=300)


#%% Old map
# load TSMP only for the grid
tsmp = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/2002_02/*TOT_PREC*',
                             )

# Get edges of Euro CORDEX
# Compute the values to handle dask arrays
lon = tsmp.lon.compute()
lat = tsmp.lat.compute()

# Extract the edges of the lon/lat grid (all four edges of the array)
lon_bottom_edge = lon.isel(rlat=0)                    # bottom edge
lat_bottom_edge = lat.isel(rlat=0)

lon_top_edge = lon.isel(rlat=-1)                      # top edge
lat_top_edge = lat.isel(rlat=-1)

lon_left_edge = lon.isel(rlon=0).isel(rlat=slice(1, -1))  # left edge (excluding corners)
lat_left_edge = lat.isel(rlon=0).isel(rlat=slice(1, -1))

lon_right_edge = lon.isel(rlon=-1).isel(rlat=slice(1, -1))  # right edge (excluding corners)
lat_right_edge = lat.isel(rlon=-1).isel(rlat=slice(1, -1))



# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25)
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))


proj = cartopy.crs.Orthographic(central_longitude=15.0, central_latitude=40, globe=None)
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



# Step 3: Plot edges of EURO-CORDEX
ax.plot(lon_bottom_edge, lat_bottom_edge, transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--')
ax.plot(lon_top_edge, lat_top_edge, transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--')
ax.plot(lon_left_edge, lat_left_edge, transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--')
ax.plot(lon_right_edge, lat_right_edge, transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--', label='Dataset Boundary')


# # Make figure larger
plt.gcf().set_size_inches(20, 10)

plt.savefig("/home/jgiles/sciebo/website information/map.svg")