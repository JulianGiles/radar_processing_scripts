#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:32:35 2026

@author: jgiles

Test regridding alternatives from ICON grid to radar volume
"""

import os
import sys

# List all possible paths where this script might live
possible_paths = [
    '/home/jgiles/Scripts/python/radar_processing_scripts',              # Office PC
    '/p/scratch/detectrea2/giles1/radar_processing_scripts',             # JUWELS Scratch
]

# Find the one that exists on the current machine and add it
for script_dir in possible_paths:
    if os.path.exists(script_dir):
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        break  # Stop checking once we find the right one

import wradlib as wrl
import numpy as np
import glob
import xarray as xr
import datetime as dt
import pandas as pd
import datetime
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
from cartopy import crs as ccrs
import xradar as xd
import cmweather
import hvplot
import hvplot.xarray
import holoviews as hv
# hv.extension("bokeh", "matplotlib") # better to put this each time this kind of plot is needed
from scipy.ndimage import binary_opening
import time

import panel as pn
from bokeh.resources import INLINE
from osgeo import osr

from functools import partial

import utils
import radarmet

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pyinterp
from pyinterp.core.config import rtree

#%% Load data

# First ICON/EMVORADO
radar_volume = utils.load_emvorado_to_radar_volume("/home/jgiles/ICON_EMVORADO_testkai/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2016022100/radout/cdfin_allsim_id-017373_201602210000_201602210000_volscan", rename=True)

icon_field = utils.load_icon("/home/jgiles/ICON_EMVORADO_testkai/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2016022100/out_EU-0275_inst_DOM01_ML_20160221T000000Z.nc", "/home/jgiles/ICON_EMVORADO_testkai/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2016022100/out_EU-0275_constant_20160220T220000Z.nc")
icon_field['time'] = icon_field['time'].dt.round('1s') # round time coord to the second

icon_field_R13B5 = utils.load_icon("/home/jgiles/ICON_EMVORADO_testkai/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2016022100/out_EU-R13B5_inst_DOM01_ML_20160221T000000Z.nc", "/home/jgiles/ICON_EMVORADO_testkai/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2016022100/out_EU-R13B5_constant_20160220T220000Z.nc")
icon_field_R13B5['time'] = icon_field_R13B5['time'].dt.round('1s') # round time coord to the second

# Then the actual obs and add ERA5 data
swp18 = utils.load_dmi_preprocessed("/automount/realpep/upload/jgiles/dmi/2016/2016-02/2016-02-21/HTY/MON_YAZ_C/18.0/MON_YAZ_C-allmoms-18.0-20162016-022016-02-21-HTY-h5netcdf.nc")
swp18 = wrl.georef.georeference(swp18)
swp18 = utils.attach_ERA5_fields(swp18, path="/automount/ags/jgiles/ERA5/hourly/turkey/pressure_level_vars/",
                                 convert_to_C=False,
                       variables=["temperature", "relative_humidity"],
                       rename={"t":"TEMP", "r":"RH"}, set_as_coords=False,
                       k_n=9, pre_interpolate_z=True)

#%% Interpolate ICON to radar grid

# First my method
icon_volume = utils.icon_to_radar_volume(icon_field[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                     "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume)

icon_R13B5_volume = utils.icon_to_radar_volume(icon_field_R13B5[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                                 "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume)

# Then Kai's method
def kai_icon_to_radar(icon_ds: xr.Dataset, radar_ds: xr.Dataset) -> xr.Dataset:
    """
    Colleague's nearest-neighbor regridding logic, adapted to safely ingest
    your datasets and project both into a metric Cartesian space for accurate 3D distance.
    """

    # 1. Define a metric Cartesian projection (AEQD) centered on the radar
    site_lon = radar_ds.longitude.values.item()
    site_lat = radar_ds.latitude.values.item()
    aeqd_proj_str = f"+proj=aeqd +lon_0={site_lon} +lat_0={site_lat} +datum=WGS84"
    radar_crs = wrl.georef.ensure_crs(aeqd_proj_str)

    # 2. Re-georeference the radar volume into AEQD (meters) to ensure isotropic 3D distance
    # We drop existing x,y,z first in case they are populated with WGS84 degrees
    radar_ds_clean = radar_ds.drop_vars(["x", "y", "z"], errors="ignore")
    radar_ds_aeqd = wrl.georef.georeference(radar_ds_clean, crs=radar_crs)

    radar_x = radar_ds_aeqd["x"].values.reshape(-1)
    radar_y = radar_ds_aeqd["y"].values.reshape(-1)
    radar_z = radar_ds_aeqd["z"].values.reshape(-1)

    # radar_dims are usually ('sweep_fixed_angle', 'azimuth', 'range') or similar
    radar_dims = list(radar_ds_aeqd["x"].dims)
    radar_shape = radar_ds_aeqd["x"].shape

    # 3. Extract and prepare ICON coordinates
    n_height = icon_ds.sizes["height"]
    n_cells = icon_ds.sizes["ncells"]

    icon_lon = icon_ds["clon"].values
    if icon_ds["clon"].attrs.get("units") == "radian":
        icon_lon = np.rad2deg(icon_lon)

    icon_lat = icon_ds["clat"].values
    if icon_ds["clat"].attrs.get("units") == "radian":
        icon_lat = np.rad2deg(icon_lat)

    # Get Z coordinates (handles your choice of z_ifc or falls back to z_mc)
    z_var = "z_mc" if "z_mc" in icon_ds else "z_ifc"
    z_values = icon_ds[z_var].values
    if z_values.ndim == 3: # i.e., shape (time, height, ncells)
        z_values = z_values[0]

    # Handle the case where z_ifc has one more vertical level than height (half-levels)
    if z_values.shape[0] != n_height:
        z_values = z_values[:n_height, :]

    # 4. Reproject ICON coords to the radar's AEQD Cartesian CRS
    proj_wgs = wrl.georef.ensure_crs(4326)
    mod_x, mod_y = wrl.georef.reproject(
        icon_lon, icon_lat, trg_crs=radar_crs, src_crs=proj_wgs
    )

    # 5. Build ICON source coordinates (flattened: height outer, ncells inner)
    src = np.column_stack([
        np.tile(mod_x, n_height),
        np.tile(mod_y, n_height),
        z_values.ravel(order="C"),
    ])
    indices_all = np.arange(src.shape[0], dtype=np.int64)

    # 6. Build radar target coordinates
    trg = np.column_stack([radar_x, radar_y, radar_z])

    # 7. Query nearest ICON point using pyinterp's 3D RTree
    mesh = pyinterp.RTree3D()
    mesh.packing(src, indices_all)
    _, indices_near = mesh.query(trg, rtree.Query().with_k(1))
    indices_near = indices_near.reshape(-1).astype(np.int64)

    # Reshape indices to match the spatial dimensions of the radar volume
    icon_index = xr.DataArray(
        indices_near.reshape(radar_shape),
        dims=radar_dims,
        coords={dim: radar_ds[dim] for dim in radar_dims if dim in radar_ds.coords}
    )

    # 8. Map ICON data to the radar grid using apply_ufunc
    first_var = next(iter(icon_ds.data_vars.values()))

    icon_radar_ds = xr.apply_ufunc(
        lambda values, indices: (
            values.reshape(values.shape[:-2] + (-1,))[..., indices]
        ),
        icon_ds,
        icon_index,
        input_core_dims=[
            ["height", "ncells"],
            radar_dims,
        ],
        output_core_dims=[
            radar_dims,
        ],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[first_var.dtype],
    )

    return icon_radar_ds.transpose("time", "sweep_fixed_angle", "elevation", "azimuth", "range", missing_dims="ignore")

icon_volume_kai = kai_icon_to_radar(
    icon_field_R13B5[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh", "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_mc"]],
    radar_volume
)

icon_volume_kai = icon_volume_kai.assign({
        "latitude":radar_volume.latitude,
        "longitude":radar_volume.longitude,
        "altitude":radar_volume.altitude,
        "elevation": radar_volume.elevation}
    )


#%% Compare results
#%%% PPIs
wrl.georef.georeference(icon_volume_kai).temp[0,-2].wrl.vis.plot()

icon_R13B5_volume.temp[0,-2].wrl.vis.plot()

icon_volume.temp[0,-2].wrl.vis.plot()

#%% Height vs range
icon_volume_kai.z[-2,0].plot(label="Kai z", marker="x")

icon_volume_kai.z_mc[-2,0].plot(label="Kai z_mc", marker=".")

icon_R13B5_volume.z[-2,0].plot(ls="--", lw=3, label="Julian from R13B5")

icon_volume.z[-2,0].plot(ls=":", label="Julian from lon/lat")

radar_volume.z[-2,0].plot(ls=":", lw=2, label="radar_volume")

plt.legend()

#%% TEMP vs range
icon_volume_kai.temp[0,-2,0].plot(label="Kai z", marker="x")

icon_R13B5_volume.temp[0,-2,0].plot(ls="--", lw=3, label="Julian from R13B5")

icon_volume.temp[0,-2,0].plot(ls=":", label="Julian from lon/lat")

swp18.TEMP[0,0].plot(ls=":", lw=2, label="obs")

plt.legend()

#%% TEMP vs height

## Get the icon_field temperature profile closest to the radar site
# 1. Get the geographic coordinates of the radar
radar_lon = radar_volume.longitude.item()
radar_lat = radar_volume.latitude.item()

# 2. Compute the squared Euclidean distance (in degrees) to all ICON grid points.
# (For finding the nearest neighbor on a regional grid, this is perfectly accurate)
# Note: icon_field["lon"] and ["lat"] might be in radians depending on the dataset.
# If they are in radians, multiply radar_lon/lat by np.deg2rad() or convert icon_field to degrees first.
icon_lon = icon_field["lon"]
icon_lat = icon_field["lat"]

# Ensure we're comparing degrees to degrees
if icon_lon.attrs.get("units") == "radian":
    icon_lon = np.rad2deg(icon_lon)
    icon_lat = np.rad2deg(icon_lat)

# Calculate distance
sq_dist = (icon_lon - radar_lon)**2 + (icon_lat - radar_lat)**2

# 3. Find the 1D flat index of the minimum distance and convert it back to 2D (y, x)
min_flat_index = np.argmin(sq_dist.data).compute().item()
y_idx, x_idx = np.unravel_index(min_flat_index, sq_dist.shape)
print(f"Closest grid point is at indices y={y_idx}, x={x_idx}")

# 4. Select the vertical profile of temperature at this closest point
temp_profile = icon_field["temp"].isel(y=y_idx, x=x_idx)

# And grab the corresponding physical heights of those grid cells
height_profile = icon_field["z_mc"].isel(y=y_idx, x=x_idx)

# Plot
icon_volume_kai.temp[0,-2,0].plot(x="z", label="Kai z", marker="x")

icon_R13B5_volume.temp[0,-2,0].plot(x="z", ls="--", lw=3, label="Julian from R13B5")

icon_volume.temp[0,-2,0].plot(x="z", ls=":", label="Julian from lon/lat")

plt.plot(height_profile, temp_profile[0], ls=":", label="ICON lon/lat")

swp18.TEMP[0,0].plot(x="z", ls=":", lw=2, label="obs")

plt.legend()


#%% Improvement using apply_ufunc?

#%%% Use my current approach
start_time = time.time()
icon_volume = utils.icon_to_radar_volume(icon_field[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                     "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume)
total_time = time.time() - start_time
print(f"Total time {total_time/60:.2f} minutes.")

#%%% Use new approach
start_time = time.time()
icon_volume_f = utils.icon_to_radar_volume_faster(icon_field[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                     "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume)
total_time = time.time() - start_time
print(f"Total time {total_time/60:.2f} minutes.")

#%%% User new even faster approach reusing cache
start_time = time.time()
icon_volume_ff, cache = utils.icon_to_radar_volume_faster_faster(icon_field[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                     "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume,
                                                                 return_cache=True)
total_time = time.time() - start_time
print(f"Total time {total_time/60:.2f} minutes.")

start_time = time.time()
icon_volume_ff_ = utils.icon_to_radar_volume_faster_faster(icon_field[["temp", "pres", "rh", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                     "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume,
                                                                 indexer_cache=cache)
total_time = time.time() - start_time
print(f"Total time {total_time/60:.2f} minutes.")
