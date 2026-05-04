#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:01:08 2025

@author: jgiles
"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER?)

import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
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
import time
import re
import scipy
import statsmodels.api as sm
from scipy.ndimage import binary_opening

import panel as pn
from bokeh.resources import INLINE
from osgeo import osr

from functools import partial

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator

import dotenv

secrets = dotenv.dotenv_values("/user/jgiles/secrets.env")

os.environ['WRADLIB_DATA'] = '/home/jgiles/wradlib-data-main'
# set earthdata token (this may change, only lasts a few months)
os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = secrets['EARTHDATA_TOKEN']

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

realpep_path = "/automount/realpep/"

#%% Check which elevations would be better suited for volume matching

#%%% Load two example files
ff1 = realpep_path+"/upload/jgiles/dmi/final_ppis/2016/2016-10/2016-10-28/HTY/MON_YAZ_B/1.5/MON_YAZ_B-allmoms-1.5-20162016-102016-10-28-HTY-h5netcdf.nc"
ff2 = realpep_path+"/upload/jgiles/dmi/final_ppis/2016/2016-10/2016-10-28/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2016-10-28-GZT-h5netcdf.nc"

ds1 = xr.open_dataset(ff1)
ds2 = xr.open_dataset(ff2)

# Get PPIs into the same reference system
proj = utils.get_common_projection(ds1, ds2)

ds1 = wrl.georef.georeference(ds1, crs=proj)
ds2 = wrl.georef.georeference(ds2, crs=proj)

#%%% Check which azimuth indexes are pointing to the other radar

tsel = "2016-10-28T05"

ds1.sel(time=tsel, method="nearest")["DBZH"].isel(azimuth=[52,53]).wrl.vis.plot(alpha=0.5)
ax = plt.gca()
ds2.sel(time=tsel, method="nearest")["DBZH"].isel(azimuth=[233,234]).wrl.vis.plot(ax=ax, alpha=0.1)
ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-10000, "HTY")
ax.text(ds2.x[0,0]-10000, ds2.y[0,0]-10000, "GZT")

plt.title("DBZH "+tsel)

#%%% Define a function to plot the intersect of two scan strategies

from cycler import cycler

def _bearing_deg(lon1, lat1, lon2, lat2):
    """Forward azimuth (bearing) from point1 to point2 in degrees [0..360)."""
    lon1r, lat1r, lon2r, lat2r = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360.0) % 360.0


def _great_circle_interpolate(lon1, lat1, lon2, lat2, npts=400):
    """Great-circle interpolation (slerp) returning array (npts,2) lon/lat (deg)."""
    lon1r, lat1r, lon2r, lat2r = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    v1 = np.array([np.cos(lat1r) * np.cos(lon1r), np.cos(lat1r) * np.sin(lon1r), np.sin(lat1r)])
    v2 = np.array([np.cos(lat2r) * np.cos(lon2r), np.cos(lat2r) * np.sin(lon2r), np.sin(lat2r)])
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-8:
        lons = np.linspace(lon1, lon2, npts)
        lats = np.linspace(lat1, lat2, npts)
        return np.column_stack((lons, lats))
    t = np.linspace(0.0, 1.0, npts)
    sin_omega = np.sin(omega)
    vs = (np.sin((1.0 - t)[:, None] * omega) * v1[None, :] + np.sin(t[:, None] * omega) * v2[None, :]) / sin_omega
    vs /= np.linalg.norm(vs, axis=1)[:, None]
    lats = np.arcsin(vs[:, 2])
    lons = np.arctan2(vs[:, 1], vs[:, 0])
    return np.degrees(np.column_stack((lons, lats)))


def _haversine_m(lon1, lat1, lon2, lat2, R=6371000.0):
    """Haversine distance (meters) between two lon/lat points in degrees."""
    lon1r, lat1r, lon2r, lat2r = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def plot_dual_scan_strategy(
    ds1,
    ds2,
    elevs1,
    elevs2,
    ds1_name="Radar 1",
    ds2_name="Radar 2",
    beamwidth=1.0,
    terrain=True,
    dem=None,
    npts=400,
    cmap1="PuBu_r",
    cmap2="OrRd_r",
    pad_deg=0.02,
    show=True,
    figsize=(12, 5),
    plot_title = True,
    legend_pos = "out"
):
    """
    Plot scan strategies of two radars *facing each other* on a shared transect,
    including shaded half-power beam widths (like wradlib.vis.plot_scan_strategy).

    Parameters
    ----------
    ds1, ds2 : xarray.Dataset
        Radar datasets. Must contain `longitude`, `latitude`, `altitude` variables
        and a coordinate called `range` (units: metres).
    elevs1, elevs2 : sequence of float
        Elevation angles (deg) for radar 1 and radar 2.
    ds1_name, ds2_name : str
        Names of the radar datasets for the labels.
    beamwidth : float
        3 dB beam width in degrees (default 1.0 deg).
    terrain : bool
        If True, download / extract SRTM for the transect and plot orography.
        If False, no terrain is shown.
    dem : object, optional
        Pre-loaded DEM (returned by wrl.io.get_srtm) to avoid repeated downloads.
    npts : int
        Number of points along the great-circle transect.
    cmap1, cmap2 : str
        Matplotlib colormap names (used for the elevation fills of radar1 and radar2).
    pad_deg : float
        Small padding in degrees when requesting SRTM bbox (default 0.02 deg).
    show : bool
        If True, plt.show() is called for new figures.
    figsize : tuple
        Matplotlib figure size.
    plot_title: bool
        If True add the title.
    legend_pos: "out" or "in"
        If "out" plot the legends outside the plot area. If "in" plot them inside.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # --- read sites ---
    try:
        lon1, lat1, alt1 = float(ds1.longitude.values), float(ds1.latitude.values), float(ds1.altitude.values)
        lon2, lat2, alt2 = float(ds2.longitude.values), float(ds2.latitude.values), float(ds2.altitude.values)
    except Exception as exc:
        raise ValueError("ds1/ds2 must contain 'longitude','latitude','altitude' variables.") from exc

    # --- ranges in metres (expect ds.range in metres) ---
    try:
        r1_res_m = float(ds1.range.diff("range").median().values)
        r2_res_m = float(ds2.range.diff("range").median().values)
        r1_m = np.arange(ds1.sizes["range"]) * r1_res_m
        r2_m = np.arange(ds2.sizes["range"]) * r2_res_m
    except Exception as exc:
        raise ValueError("Could not read 'range' coordinate from ds1/ds2. Ensure a coordinate named 'range' exists (units: m).") from exc

    # --- transect lon/lats and cumulative distances (km) ---
    lonlats = _great_circle_interpolate(lon1, lat1, lon2, lat2, npts=npts)
    lons = lonlats[:, 0]; lats = lonlats[:, 1]
    # cumulative distances (meters) along the transect (haversine pairwise)
    dists_m = np.zeros(npts)
    for i in range(1, npts):
        dists_m[i] = dists_m[i - 1] + _haversine_m(lons[i - 1], lats[i - 1], lons[i], lats[i])
    distances_km = dists_m / 1000.0
    dist_total_km = distances_km[-1]

    # --- terrain extraction along transect (meters) ---
    terrain_heights_m = None
    if terrain:
        # bbox: [minlon, maxlon, minlat, maxlat] with a small pad
        bbox = [lons.min() - pad_deg, lons.max() + pad_deg, lats.min() - pad_deg, lats.max() + pad_deg]
        try:
            ds_srtm = dem if dem is not None else wrl.io.get_srtm(bbox, resolution=1)
            rastervalues, rastercoords, crs = wrl.georef.extract_raster_dataset(ds_srtm, nodata=-32768.0)
            # map raster values to transect lon/lats
            terrain_heights_m = wrl.ipol.cart_to_irregular_spline(
                rastercoords, rastervalues, lonlats, order=3, prefilter=False
            )
        except Exception as exc:
            raise RuntimeError(
                "terrain=True but SRTM extraction failed. Pass `dem` (preloaded SRTM) "
                "or supply terrain=False."
            ) from exc

    # --- helper to plot a single beam (m -> km) with shaded half-power region ---
    def _plot_beam_km(ax_local, r_m, alt_m, beamradius_m, label=None):
        # convert to km for plotting
        r_km = r_m / 1000.0
        alt_km = alt_m / 1000.0
        beam_km = beamradius_m / 1000.0
        # center & edges in black, fill uses the current color cycle
        center = ax_local.plot(r_km, alt_km, "-k", linewidth=0.6, alpha=1.0, zorder=3)
        edge1 = ax_local.plot(r_km, alt_km + beam_km, ":k", linewidth=0.6, alpha=1.0, zorder=3)
        ax_local.plot(r_km, alt_km - beam_km, ":k", linewidth=0.4, alpha=1.0, zorder=3)
        fill = ax_local.fill_between(r_km, alt_km - beam_km, alt_km + beam_km, label=label, alpha=0.45, zorder=2)
        return fill, center, edge1

    # --- prepare figure/axes ---
    fig, ax = plt.subplots(figsize=figsize)

    # plot terrain polygon
    if terrain and (terrain_heights_m is not None):
        ax.fill_between(distances_km, terrain_heights_m / 1000.0, np.min(terrain_heights_m / 1000.0) - 0.5,
                        color="0.75", zorder=1, label="Terrain")

    # --- Radar 1: azimuth pointing to radar 2 ---
    az1 = _bearing_deg(lon1, lat1, lon2, lat2)
    # use spherical_to_xyz to get altitudes and ground ranges (m)
    # note: spherical_to_xyz expects arrays in metres for ranges
    xyz1, _rad1 = wrl.georef.polar.spherical_to_xyz(r1_m, np.array([az1]), elevs1, (lon1, lat1, alt1), squeeze=True)
    # ensure xyz1 has shape (n_elevs, n_ranges, 3) if single elevs become axis
    if xyz1.ndim == 2:
        xyz1 = xyz1[np.newaxis, ...]
    # color cycle for radar1 fills
    cmap_obj1 = plt.get_cmap(cmap1)
    colors1 = [cmap_obj1(i / max(1, len(elevs1))) for i in range(len(elevs1))]
    ax.set_prop_cycle(cycler(color=colors1))
    center_last = None
    edge_last = None

    for i, el in enumerate(elevs1):
        alt = xyz1[i, ..., 2]           # alt (m)
        groundrange = np.sqrt(xyz1[i, ..., 0] ** 2 + xyz1[i, ..., 1] ** 2)  # m
        plrange = np.insert(groundrange, 0, 0.0)  # include 0 at radar
        plalt = np.insert(alt, 0, alt1)           # include radar altitude
        beamradius = wrl.util.half_power_radius(plrange, beamwidth)  # in metres
        fill, center, edge = _plot_beam_km(ax, plrange, plalt, beamradius, label=f"{el:4.1f}°")
        center_last = center
        edge_last = edge

    # mark radar 1 position and text
    ax.scatter([0.0], [alt1 / 1000.0], color="k", s=30, zorder=4)
    ax.text(0.0, alt1 / 1000.0 - 0.2, ds1_name, color="k", va="bottom", ha="right")

    # --- Radar 2: azimuth pointing to radar 1 ---
    az2 = _bearing_deg(lon2, lat2, lon1, lat1)
    xyz2, _rad2 = wrl.georef.polar.spherical_to_xyz(r2_m, np.array([az2]), elevs2, (lon2, lat2, alt2), squeeze=True)
    if xyz2.ndim == 2:
        xyz2 = xyz2[np.newaxis, ...]
    # color cycle for radar2 fills
    cmap_obj2 = plt.get_cmap(cmap2)
    colors2 = [cmap_obj2(i / max(1, len(elevs2))) for i in range(len(elevs2))]
    ax.set_prop_cycle(cycler(color=colors2))

    for i, el in enumerate(elevs2):
        alt = xyz2[i, ..., 2]
        groundrange = np.sqrt(xyz2[i, ..., 0] ** 2 + xyz2[i, ..., 1] ** 2)
        plrange = np.insert(groundrange, 0, 0.0)
        plalt = np.insert(alt, 0, alt2)
        beamradius = wrl.util.half_power_radius(plrange, beamwidth)  # m

        # plot mirrored (so radar2 looks left toward radar1)
        shifted_plrange = (dist_total_km * 1000.0) - plrange
        fill, center, edge = _plot_beam_km(ax, shifted_plrange, plalt, beamradius, label=f"{el:4.1f}°")

    # mark radar 2 position and text
    ax.scatter([dist_total_km], [alt2 / 1000.0], color="k", s=30, zorder=4)
    ax.text(dist_total_km, alt2 / 1000.0 - 0.2, ds2_name, color="k", va="bottom", ha="left")

    # --- cosmetics and legends ---
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    if uniq_handles:
        if legend_pos == "in":
            leg1 = ax.legend(uniq_handles, uniq_labels, prop={"family": "monospace"}, loc="upper right")
        else:
            leg1 = ax.legend(uniq_handles, uniq_labels, prop={"family": "monospace"}, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(leg1)

    if center_last is not None and edge_last is not None:
        legend2 = {"Center": center_last[0], "3 dB": edge_last[0]}
        if legend_pos == "in":
            ax.legend(legend2.values(), legend2.keys(), prop={"family": "monospace"}, loc="lower right")
        else:
            ax.legend(legend2.values(), legend2.keys(), prop={"family": "monospace"}, loc="lower left", bbox_to_anchor=(1.02, 0.0))

    margin_x = max(2.0, 0.05 * dist_total_km)
    ax.set_xlim(-margin_x, dist_total_km + margin_x)

    max_h = max(alt1, alt2) / 1000.0 + 2.0
    if terrain and (terrain_heights_m is not None):
        min_h = min(np.min(terrain_heights_m / 1000.0) - 0.5, 0.0)
    else:
        min_h = 0.0
    ax.set_ylim(min_h, max_h)

    ax.set_xlabel("Distance along transect (km)")
    ax.set_ylabel("Height (km a.s.l.)")
    if plot_title: ax.set_title("Dual-Radar Scan Strategies (shared transect & terrain)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


#%%% Plot scan strategies

plot_dual_scan_strategy(
    ds1, ds2,
    elevs1=[1.5, 2.2, 3.0], # [0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 4.5, 6.0, 8.0, 12.0, 18.0, 27.0, 38.0],
    elevs2=[0.4, 0.5, 0.7], # [0.5, 1.3, 4.0, 10.0, 15.0],
    ds1_name="HTY",
    ds2_name="GZT",
    terrain=True,
    figsize=(9, 4),
    plot_title=False,
    legend_pos="in"
)


#%%% Plot scan strategies individually to double check

ds1_nrays = ds1.sizes['azimuth']
ds1_nbins = ds1.sizes['range']
ds1_range_res = float(ds1.range.diff("range").median().values)
ds1_ranges = np.arange(ds1_nbins) * ds1_range_res
ds1_elevs = [0.0, 0.7, 1.5, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 25.0]
ds1_site = (float(ds1.longitude.values), float(ds1.latitude.values), float(ds1.altitude.values)) # lon, lat alt
ds1_beamwidth = 1.0

ds2_nrays = ds2.sizes['azimuth']
ds2_nbins = ds2.sizes['range']
ds2_range_res = float(ds2.range.diff("range").median().values)
ds2_ranges = np.arange(ds2_nbins) * ds2_range_res
ds2_elevs = [0.5, 1.3, 4.0, 7.0, 10.0, 15.0, 25.0, 359.8]
ds2_site = (float(ds2.longitude.values), float(ds2.latitude.values), float(ds2.altitude.values)) # lon, lat alt
ds2_beamwidth = 1.0

ax = wrl.vis.plot_scan_strategy(
    ds1_ranges, ds1_elevs, ds1_site, units="km", terrain=True, az=53
)

wrl.vis.plot_scan_strategy(
    ds2_ranges, ds2_elevs, ds2_site, units="km", terrain=True, az=233
)

#%% Look for potential dates with ERA5
# Load ERA5
era5p = xr.open_mfdataset("/automount/ags/jgiles/ERA5/hourly/europe/single_level_vars/total_precipitation/total_precipitation_year_20*.nc")
era5t = xr.open_mfdataset("/automount/ags/jgiles/ERA5/hourly/turkey/pressure_level_vars/temperature/temperature_all_*.nc")
era5g = xr.open_mfdataset("/automount/ags/jgiles/ERA5/hourly/turkey/pressure_level_vars/geopotential/geopotential_all_*.nc")

era5t = era5t.rename({"valid_time": "time"})
era5g = era5g.rename({"valid_time": "time"})

# select area and timespan
timespan = slice("2016","2020")
latsel = 36.7
lonsel = 36.45
era5p_ = era5p.sel(latitude=latsel, longitude=lonsel, method="nearest").sel(time=timespan)
era5t_ = era5t.sel(latitude=latsel, longitude=lonsel, method="nearest").sel(time=timespan)
era5g_ = era5g.sel(latitude=latsel, longitude=lonsel, method="nearest").sel(time=timespan)

# Compute z coordinate
ref_lat = 36.5 # reference latitude for Earth's radius
earth_r = wrl.georef.projection.get_earth_radius(ref_lat)
gravity = 9.80665
era5z_ = (earth_r*(era5g_.z/gravity)/(earth_r - era5g_.z/gravity)).compute()

# Look for dates with the desired isotherm at a certain height and match it to detected precip
isot = 3 # C
minh = 4000 # m
minp = 0.1 # mm/h

era5_isoth = era5z_.where((era5t_.t-273.15) < isot).min(dim="pressure_level")

era5_isotcond = era5_isoth > minh

era5_pcond = era5p_.tp*1000 > minp

# select a year to print dates that fulfill the isotherm condition
year = "2016-05"
era5_isotcond_ = era5_isotcond.sel(time=year)

for date, group in era5_isotcond_.compute().groupby("time.date"):
    # Check if any (or all) hours in this day meet the condition
    valid_hours = group.where(group, drop=True)

    if len(valid_hours.time) > 0:
        hours_str = ", ".join([pd.to_datetime(t).strftime('%H') for t in valid_hours.time.values])
        print(f"{date} | {hours_str}")

# select a year to print dates that fulfill the isotherm and precip conditions
year = "2016"
era5_isotcond_ = era5_isotcond.sel(time=year)
era5_pcond_ = era5_pcond.sel(time=year)
era5_conds = era5_isotcond_*era5_pcond_ # combine conditions

for date, group in era5_conds.compute().groupby("time.date"):
    # Check if any (or all) hours in this day meet the condition
    valid_hours = group.where(group, drop=True)

    if len(valid_hours.time) > 0:
        hours_str = ", ".join([pd.to_datetime(t).strftime('%H') for t in valid_hours.time.values])
        print(f"{date} | {hours_str}")

#%% List of dates and available elevs
"""
ML high:
    "2016-04-09", #### no valid matches: 3.0-0.7 # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-05-14",# ONLY GZT SURV 0.0 # no valid matches
    # "2016-05-15",# ONLY GZT SURV 0.0 # no valid matches
    # "2016-05-16",# ONLY GZT SURV 0.0 # no valid matches
    "2016-05-31",
    "2016-09-22", # Wet radome in GZT
    "2016-10-18", #### no valid matches: 3.0-0.5
    "2016-10-28",
    "2016-11-01", #### no valid matches: 2.2-0.5, 3.0-0.5
    # "2016-12-01", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    "2017-04-12",
    "2017-04-13",
    "2017-05-18",
    "2017-05-22", #### no valid matches: 2.2-0.5, 3.0-0.5
    # "2018-03-28",# NO POLARIMETRY OR RHOHV IN HTY
    # "2019-02-06",# NO POLARIMETRY OR RHOHV IN HTY
    # "2019-05-06",# NO POLARIMETRY OR RHOHV IN HTY
    "2019-10-17",
    "2019-10-20",
    "2019-10-21", #### no valid matches: 3.0-0.5
    "2019-10-22", #### no valid matches: 3.0-0.5
    "2019-10-28", #### no valid matches: 3.0-0.5
    "2020-03-12", #### no valid matches: 3.0-0.5
    "2020-03-13", #### no valid matches: 3.0-0.5
    # "2020-05-01", #### no valid matches: 1.5-0.5, 3.0-0.5 (all combinations)

# # X means that this date does not have ML low enough, so I removed it
# # XX means that the ML is not low enough but it is very close, I could leave it if it works.
ML in between radars:
    "2016-02-06", # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-02-21", # X # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-04-12", # X # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-05-28", # X
    # "2016-11-30", # X
    # "2016-12-01", # X
    "2016-12-14",
    "2016-12-16",
    "2016-12-20",
    "2016-12-21",
    "2016-12-22",
    # "2016-12-26", # X
    "2016-12-27",
    "2016-12-29", # XX
    "2016-12-30", # XX
    "2016-12-31",
    "2017-01-01",
    "2017-01-02",
    # "2017-01-08", #### UNUSABLE ZDR IN GZT
    # "2017-01-11", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-01-20",  #### UNUSABLE ZDR IN GZT #### no valid matches: 1.5-0.5,
    # "2017-03-04", # XX #### UNUSABLE ZDR IN GZT
    # "2017-03-16", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-03-17", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-12-24", # X
    # "2017-12-31", # X
    # "2018-01-04", # X
    # "2018-01-05", # X
    # "2018-01-14", # X
    # "2019-11-27", # X
    # "2019-12-09", # X
    # "2019-12-13", # X
    # "2019-12-14", # X
    # "2019-12-24", # X
    # "2019-12-25", # X
    # "2019-12-26", # XX
    # "2019-12-28", # X
    # "2019-12-30", # X
    # "2019-12-31", # X
    # "2020-01-02", # XX
    # "2020-01-03", # XX
    # "2020-01-04", # XX
    # "2020-01-06",
    # "2020-01-07", # XX
    "2020-01-19", # XX
    # "2020-02-22", # XX
    # "2020-02-29", # XX
    # "2020-03-17", # X
    # "2020-11-04", # X
    # "2020-11-20", # X
    # "2020-12-14", # X
"""

#%% Load the selected elevations and check

# Suitable matching elevations:
# HTY: 1.5, 2.2, 3.0 and above
# GZT: 0.5, 1.3

# ff1 = realpep_path+"/upload/jgiles/dmi/final_ppis/2017/2017-01/2017-01-02/HTY/MON_YAZ_B/1.5/MON_YAZ_B-allmoms-1.5-20172017-012017-01-02-HTY-h5netcdf.nc"
# ff2 = realpep_path+"/upload/jgiles/dmi/final_ppis/2017/2017-01/2017-01-02/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2017-01-02-GZT-h5netcdf.nc"

ff1 = realpep_path+"/upload/jgiles/dmi/final_ppis/2020/2020-03/2020-03-13/HTY/VOL_A/1.5/VOL_A-allmoms-1.5-2020-03-13-HTY-h5netcdf.nc"
ff2 = realpep_path+"/upload/jgiles/dmi/final_ppis/2020/2020-03/2020-03-13/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2020-03-13-GZT-h5netcdf.nc"

ds1 = xr.open_mfdataset(ff1)
ds2 = xr.open_mfdataset(ff2)

# Get PPIs into the same reference system
proj = utils.get_common_projection(ds1, ds2)

ds1 = wrl.georef.georeference(ds1, crs=proj)
ds2 = wrl.georef.georeference(ds2, crs=proj)

#%%% Load and apply smoothed and extrapolated ZDR offsets for dates without offset (like for GZT when the ML is too low)
# These dates should not have valid ZDR calibrations in GZT due to the low ML.
# Then, we load all daily calibrations available in the period to approximate the
# calibration with smoothing and interpolation.

print("Loading ZDR daily offsets for NaN filling")

ds1_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
ds1_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")
ds1_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
ds1_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")

ds2_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
ds2_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")
ds2_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
ds2_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")

# # plot running medians to check smoothing
# ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
# ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
# ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
# ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()

# Combine to create a single offset timeseries
ds1_zdr_offsets_lr = xr.concat([ds1_zdr_offsets_lr_ml, ds1_zdr_offsets_lr_1c], dim="variant").mean("variant")
ds1_zdr_offsets_qvp = xr.concat([ds1_zdr_offsets_qvp_ml, ds1_zdr_offsets_qvp_1c], dim="variant").mean("variant")
ds1_zdr_offsets_qvp = ds1_zdr_offsets_qvp.where(ds1_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
ds1_zdr_offsets_comb = xr.where(ds1_zdr_offsets_lr["ZDR_offset_datacount"] >= ds1_zdr_offsets_qvp["ZDR_offset_datacount"],
                            ds1_zdr_offsets_lr["ZDR_offset"],
                            ds1_zdr_offsets_qvp["ZDR_offset"]).fillna(ds1_zdr_offsets_qvp["ZDR_offset"])

ds2_zdr_offsets_lr = xr.concat([ds2_zdr_offsets_lr_ml, ds2_zdr_offsets_lr_1c], dim="variant").mean("variant")
ds2_zdr_offsets_qvp = xr.concat([ds2_zdr_offsets_qvp_ml, ds2_zdr_offsets_qvp_1c], dim="variant").mean("variant")
ds2_zdr_offsets_qvp = ds2_zdr_offsets_qvp.where(ds2_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
ds2_zdr_offsets_comb = xr.where(ds2_zdr_offsets_lr["ZDR_offset_datacount"] >= ds2_zdr_offsets_qvp["ZDR_offset_datacount"],
                            ds2_zdr_offsets_lr["ZDR_offset"],
                            ds2_zdr_offsets_qvp["ZDR_offset"]).fillna(ds2_zdr_offsets_qvp["ZDR_offset"])

# finally, smooth it out
ds1_zdr_offsets_comb_smooth = ds1_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()
ds2_zdr_offsets_comb_smooth = ds2_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()


# Add new offset/atten corrected ZDR in datasets
ds1 = ds1.assign({"ZDR_EC_OC2":
                  ds1["ZDR_EC"] - ds1_zdr_offsets_comb_smooth.sel(time=ds1.time[0].values.astype(str)[:10]).mean()} )

ds2 = ds2.assign({"ZDR_EC_OC2":
                  ds2["ZDR_EC"] - ds2_zdr_offsets_comb_smooth.sel(time=ds2.time[0].values.astype(str)[:10]).mean()} )

#%%% Plot example
tsel = "2020-03-13T08"

vv = "DBZH" # for the example plot
vmin = -10
vmax = 50
ds1.sel(time=tsel, method="nearest")[vv].wrl.vis.plot(alpha=0.5, vmin=vmin, vmax=vmax)
ax = plt.gca()
ds2.sel(time=tsel, method="nearest")[vv].wrl.vis.plot(ax=ax, alpha=0.2, vmin=vmin, vmax=vmax)
ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title(vv+" "+tsel)

#%%% Plot example over map
import shapely.geometry as sgeom
from cartopy.geodesic import Geodesic
import cartopy.io.img_tiles as cimgt

def add_range_circle(ax, lon, lat, radius_m, **kwargs):
    """Add a range circle around a radar site."""
    gd = Geodesic()
    geom = gd.circle(lon=lon, lat=lat, radius=radius_m)
    circle = sgeom.Polygon(geom)
    ax.add_geometries(
        [circle], crs=ccrs.PlateCarree(),
        **kwargs
    )

tsel = "2020-03-13T08"

vv = "DBZH" # for the example plot
vmin = -10
vmax = 50

# Choose a tile source (pick one):
tiles = cimgt.OSM()                        # OpenStreetMap
# tiles = cimgt.GoogleTiles(style="terrain")   # Google Maps (may need API key)
# tiles = cimgt.GoogleTiles(style="satellite")  # Google Satellite
# tiles = cimgt.GoogleTiles(style="street")        # road map
# tiles = cimgt.QuadtreeTiles()                    # Bing maps
# tiles = cimgt.Stamen('terrain')          # Stamen Terrain (may be deprecated)

# Use PlateCarree (lon/lat) as the display projection
map_proj = ccrs.Mercator()

fig, ax = plt.subplots(
    figsize=(5, 4),
    subplot_kw={"projection": map_proj}
)

# Add the background tiles FIRST (before other features)
ax.add_image(tiles, 7, zorder=0)  # <-- second arg is zoom level (higher = more detail, slower)

# --- Add map features ---
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.8, zorder=2)
ax.add_feature(cartopy.feature.BORDERS, linewidth=0.8, linestyle="--", zorder=2)
# ax.add_feature(cartopy.feature.LAND, facecolor="lightgray", alpha=0.3)
# ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue", alpha=0.3)
ax.gridlines(draw_labels=[ "bottom", "left"], linewidth=0.5, linestyle=":", color="gray")

# --- Plot radar data ---
# wradlib's .wrl.vis.plot() can accept a cartopy axes via the `ax` kwarg
# We need to pass the transform so it knows the data CRS
ds1_sel = ds1.sel(time=tsel, method="nearest")[vv]
ds2_sel = ds2.sel(time=tsel, method="nearest")[vv]

# Get lon/lat arrays (they should be coordinates after georeferencing)
# If your datasets have 'lon' and 'lat' coords use them directly;
# otherwise use wradlib's georef to get them
ds1_sel.wrl.vis.plot(
    ax=ax,
    crs=proj,           # the CRS the x/y data is in (from get_common_projection)
    alpha=0.3,
    vmin=vmin, vmax=vmax,
    add_colorbar=False,
    zorder=1
)
pm = ds2_sel.wrl.vis.plot(
    ax=ax,
    crs=proj,
    alpha=0.3,
    vmin=vmin, vmax=vmax,
    add_colorbar=False,
    zorder=1
)

ax.autoscale(enable=False, axis='both', tight=True)

# --- Radar site markers ---

lon1, lat1 = (float(ds1.longitude), float(ds1.latitude))
lon2, lat2 = (float(ds2.longitude), float(ds2.latitude))

ax.scatter([lon1, lon2], [lat1, lat2],
           color="black", s=50, zorder=5, transform=ccrs.PlateCarree())
ax.text(lon1, lat1 - 0.3, "HTY", transform=ccrs.PlateCarree(), fontsize=9, ha="center")
ax.text(lon2, lat2 - 0.3, "GZT", transform=ccrs.PlateCarree(), fontsize=9, ha="center")

add_range_circle(ax, lon1, lat1, ds1.range.max().values,
                 facecolor="none", edgecolor="black", linewidth=1, zorder=4)
add_range_circle(ax, lon2, lat2, ds2.range.max().values,
                 facecolor="none", edgecolor="black", linewidth=1, zorder=4)

# --- Colorbar and title ---
plt.colorbar(pm, ax=ax, label=vv, shrink=0.7)
ax.set_title(f"{vv} {tsel}")
plt.tight_layout()
plt.show()

#%% Add beam blockage

#%%% Define a function to calculate the beam blockage
def beam_blockage_from_radar_ds(ds,
                                sitecoords,
                                dem_resolution=1,
                                bw=1.0,
                                wradlib_token: str = None):
    """
    Compute PBB and CBB for a polar radar dataset `ds`.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Radar data in polar coordinates. Must have dims (azimuth, range) (for one elevation)
        or (time, azimuth, range) etc  but we only consider one elevation sweep at a time.
        It must have range, azimuth, and should allow georeferencing via wradlib.
    sitecoords : tuple (lon, lat, height_m)
        The radar site location in geographic coords and altitude (m).
    dem_resolution : int, default 3
        Resolution (arc-seconds) for SRTM fetch (1, 3, or 30).
    bw : float, default 1.0
        Beam-width scaling (for half power radius).
    wradlib_token : str, optional
        WRADLIB Earthdata bearer token (if needed for DEM access).

    Returns
    -------
    pbb_da : xarray.DataArray
      Partial beam blockage fraction (dims: azimuth × range)
    cbb_da : xarray.DataArray
      Cumulative beam blockage fraction, same dims
    """

    # If token is given, set environment (for remote DEM fetch)
    if wradlib_token is not None:
        os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = wradlib_token

    # 1. Georeference the radar sweep
    # Let ds be a 2D sweep (azimuth × range). If it has extra dims, you may select one slice.
    # Use wradlib’s xarray accessor if available:
    # e.g., if ds is a Wradlib-xarray object:
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    ds_geo = ds.pipe(wrl.georef.georeference, crs=wgs84)  # this gives ds with x, y, z coords attached

    # After georeferencing, ds_geo should have coords: x (az, range), y, z (altitude of beam center)
    # E.g., ds_geo.x.values, ds_geo.y.values, ds_geo.z.values

    xs = ds_geo.x.values     # shape (az, range)
    ys = ds_geo.y.values
    zs = ds_geo.z.values     # same shape or broadcastable (beam center altitudes)

    # 2. Determine DEM bounding box from radar footprint
    # Use wradlib.zonalstats.get_bbox which returns a dict-like:
    bbox = wrl.zonalstats.get_bbox(xs, ys)
    # bbox has keys "left", "bottom", "right", "top"
    left, bottom, right, top = bbox["left"], bbox["bottom"], bbox["right"], bbox["top"]

    # 3. Fetch DEM (SRTM) over that bounding box
    dem = wrl.io.dem.get_srtm([left, right, bottom, top], resolution=dem_resolution, merge=True)

    # 4. Extract DEM raster arrays and coords
    rastervalues, rastercoords, dem_proj = wrl.georef.extract_raster_dataset(dem, nodata=-32768.0)

    # 5. Clip DEM to radar bounding box
    rlimits = (left, bottom, right, top)
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords_clip = rastercoords[ind[1]:ind[3], ind[0]:ind[2], :]
    rastervalues_clip = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]

    # 6. Prepare polar grid (x,y) as points for interpolation
    polcoords = np.dstack((xs, ys))  # shape (az, range, 2)

    # 7. Interpolate DEM heights to the polar coords
    # The result is the “terrain height under each bin” (shape az × range)
    # polar_terrain = wrl.ipol.cart_to_irregular_spline(
    #     rastercoords_clip, rastervalues_clip, polcoords,
    #     order=3, prefilter=False
    # )

    # 7. Interpolate DEM heights to the polar coords
    # Pass the full, unclipped arrays. Coordinates falling outside the
    # downloaded DEM (i.e., the missing open ocean) will default to 0.0.
    polar_terrain = wrl.ipol.cart_to_irregular_spline(
            rastercoords, rastervalues, polcoords,
            order=3, prefilter=False
        )

    # 8. Compute beam radius (half power) for each range bin
    r = ds_geo.range.values   # 1D array (range)
    beamradius = wrl.util.half_power_radius(r, bw)  # gives 1D array (range,)

    # Broadcast to 2D: for each azimuth replicate beamradius
    beamradius2d = np.broadcast_to(beamradius, xs.shape)

    # 9. Compute partial beam blockage
    PBB = wrl.qual.beam_block_frac(polar_terrain, zs, beamradius2d)
    PBB = np.ma.masked_invalid(PBB)

    # 10. Compute cumulative beam blockage
    CBB = wrl.qual.cum_beam_block_frac(PBB)

    # 11. Wrap as xarray DataArrays
    pbb_da = xr.DataArray(PBB.filled(np.nan),
                          dims=("azimuth", "range"),
                          coords={"azimuth": ds_geo.azimuth, "range": ds_geo.range},
                          name="PBB")
    cbb_da = xr.DataArray(CBB,
                          dims=("azimuth", "range"),
                          coords={"azimuth": ds_geo.azimuth, "range": ds_geo.range},
                          name="CBB")

    return pbb_da, cbb_da

#%%% Calculate beam blockage
token = secrets['EARTHDATA_TOKEN']

ds1_pbb, ds1_cbb = beam_blockage_from_radar_ds(ds1.isel(time=0),
                                               (ds1.longitude, ds1.latitude, ds1.altitude),
                                               wradlib_token = token)

ds1 = ds1.assign({"PBB": ds1_pbb, "CBB": ds1_cbb})

ds2_pbb, ds2_cbb = beam_blockage_from_radar_ds(ds2.isel(time=0),
                                               (ds2.longitude, ds2.latitude, ds2.altitude),
                                               wradlib_token = token)

ds2 = ds2.assign({"PBB": ds2_pbb, "CBB": ds2_cbb})

# Plot beam blockage
ds1["CBB"].where(ds1["CBB"]>0.05).wrl.vis.plot(alpha=0.5, vmin=0, vmax=1, cmap=mpl.cm.PuRd)
ax = plt.gca()
ds2["CBB"].where(ds2["CBB"]>0.05).wrl.vis.plot(ax=ax, alpha=0.5, vmin=0, vmax=1, cmap=mpl.cm.PuRd,  xlim=(-100000, 100000), ylim=(-100000, 100000))
ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title("CBB "+tsel)

#%% Set parameters and filters

tsel = "2020-03-13T08" # for plots
vv = "DBZH"
SNRH_min = 15
RHOHV_min = 0.95
TEMP_min = -10
DBZH_min = 10
CBB_max = 0.05

# Apply thresholds before computing masks
dsx = utils.apply_min_max_thresh(ds1, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                        "SNRHC":SNRH_min, "SQIH":0.5,
                                        "TEMP":TEMP_min},
                                     {"CBB": CBB_max})
dsy = utils.apply_min_max_thresh(ds2, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                        "SNRHC":SNRH_min, "SQIH":0.5,
                                        "TEMP":TEMP_min},
                                     {"CBB": CBB_max})

# They consider that there is rain above the radar by looking at the
# median reflectivity in a circle 1.5km aroud each radar. Let's add this variable
dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,1500)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,1500)).compute().median(("azimuth", "range")).broadcast_like(dsy["DBZH"]) })

# Add the additional DBZH threshold
dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                     {})
dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                     {})

#%% Generate masks
mask1, mask2, idx1, idx2, matched_timesteps = utils.find_radar_overlap(dsx,dsy,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask1_ref, mask2_ref, idx1_ref, idx2_ref = utils.refine_radar_overlap(dsx,dsy,
                                                                        mask1, mask2,
                                                                        matched_timesteps,
                                                                        vv,
                                                                        tolerance=250.,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

# mask1, mask2, idx1, idx2, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx, dsy,
#                                                                     tolerance=250.,
#                                                                     tolerance_time=60*4)

# mask1_ref, mask2_ref, idx1_ref, idx2_ref, matched_timesteps = utils.refine_radar_overlap_unique_NN_pairs(dsx, dsy,
#                                                                                       idx1, idx2,
#                                                                                       vv,
#                                                                     tolerance_time=60*4,
#                                                                     z_tolerance=100.)

#%% Plot initial mask
dsx_ = dsx.copy()
dsy_ = dsy.copy()
dsx_.coords["x"] = dsx_.coords["x"]/1000
dsx_.coords["y"] = dsx_.coords["y"]/1000
dsy_.coords["x"] = dsy_.coords["x"]/1000
dsy_.coords["y"] = dsy_.coords["y"]/1000

dsx_plot = dsx_[vv].where(mask1).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-100, 100), ylim=(-100, 100), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx_.y.where(mask1).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy_.x.where(mask2).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy_.y.where(mask2).sel(time=tsel, method="nearest").values.flatten()

# ax.scatter(x1, y1, s=1, marker="o")
# ax.scatter(x2, y2, s=1, c="r", marker="x")

ax.scatter([dsx_.x[0,0], dsy_.x[0,0]], [dsx_.y[0,0], dsy_.y[0,0]], c="black")
ax.text(dsx_.x[0,0], dsx_.y[0,0]-30, "HTY")
ax.text(dsy_.x[0,0], dsy_.y[0,0]-30, "GZT")

plt.title(vv+" "+tsel)

#%% Plot initial mask (with zoom and scatter of points)
dsx_plot = dsx_[vv].where(mask1).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-10, 10), ylim=(-10, 10), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx_.y.where(mask1).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy_.x.where(mask2).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy_.y.where(mask2).sel(time=tsel, method="nearest").values.flatten()

ax.scatter(x1, y1, s=1, marker="o")
ax.scatter(x2, y2, s=1, c="r", marker="x")

plt.title(vv+" "+tsel)

#%% Plot refined masks
dsx_plot = dsx_[vv].where(mask1_ref).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2_ref).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-100, 100), ylim=(-100, 100), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx_.y.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy_.x.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy_.y.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()

# ax.scatter(x1, y1, s=1, marker="o")
# ax.scatter(x2, y2, s=1, c="r", marker="x")

ax.scatter([dsx_.x[0,0], dsy_.x[0,0]], [dsx_.y[0,0], dsy_.y[0,0]], c="black")
ax.text(dsx_.x[0,0], dsx_.y[0,0]-30, "HTY")
ax.text(dsy_.x[0,0], dsy_.y[0,0]-30, "GZT")

plt.title(vv+" "+tsel)

#%% Plot refined masks (with zoom and scatter of points)
dsx_plot = dsx_[vv].where(mask1_ref).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2_ref).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-10, 10), ylim=(-10, 10), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx_.y.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy_.x.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy_.y.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()

ax.scatter(x1, y1, s=1, marker="o")
ax.scatter(x2, y2, s=1, c="r", marker="x")

plt.title(vv+" "+tsel)

#%% Plot scatterplot
var_name="DBZH"
dsx_p, dsy_p = utils.return_unique_NN_value_pairs(dsx, dsy, mask1_ref, mask2_ref,
                                           idx1_ref, idx2_ref, matched_timesteps, var_name)

plt.scatter(dsx_p, dsy_p, alpha=0.1)
plt.plot([0,40], [0,40], c="red")
plt.xlabel("HTY")
plt.ylabel("GZT")

cc = round(np.corrcoef(dsx_p[np.isfinite(dsx_p)],dsy_p[np.isfinite(dsy_p)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name+" "+tsel)

#%% Generate masks (again for new plot, we still need previous masks too)
mask1_, mask2_, idx1_, idx2_, matched_timesteps = utils.find_radar_overlap(ds1,ds2,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask1_ref_, mask2_ref_, idx1_ref_, idx2_ref_ = utils.refine_radar_overlap(ds1,ds2,
                                                                        mask1_, mask2_,
                                                                        matched_timesteps,
                                                                        "SNRH",
                                                                        tolerance=250.,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

#%% Plot initial mask
dsx_ = dsx.copy()
dsy_ = dsy.copy()

dsx_.coords["bca"] = utils.compute_crossing_angle_cartesian(dsx_, (dsx_.x[:,0].mean(), dsx_.y[:,0].mean(), dsx_.z[:,0].mean()), (dsy_.x[:,0].mean(), dsy_.y[:,0].mean(), dsy_.z[:,0].mean()))
dsy_.coords["bca"] = utils.compute_crossing_angle_cartesian(dsy_, (dsx_.x[:,0].mean(), dsx_.y[:,0].mean(), dsx_.z[:,0].mean()), (dsy_.x[:,0].mean(), dsy_.y[:,0].mean(), dsy_.z[:,0].mean()))
bca_min = 135

dsx_.coords["x"] = dsx_.coords["x"]/1000
dsx_.coords["y"] = dsx_.coords["y"]/1000
dsy_.coords["x"] = dsy_.coords["x"]/1000
dsy_.coords["y"] = dsy_.coords["y"]/1000

dsx_plot = dsx_[vv].where(mask1_ref).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2_ref).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-100, 100), ylim=(-100, 100), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1_ref_).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).values.flatten()
y1 = dsx_.y.where(mask1_ref_).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).values.flatten()

x2 = dsy_.x.where(mask2_ref_).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).values.flatten()
y2 = dsy_.y.where(mask2_ref_).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).values.flatten()

ax.scatter(x1, y1, s=1, marker="o")
ax.scatter(x2, y2, s=1, c="r", marker="x")

ax.scatter([dsx_.x[0,0], dsy_.x[0,0]], [dsx_.y[0,0], dsy_.y[0,0]], c="black")
ax.text(dsx_.x[0,0], dsx_.y[0,0]-30, "HTY")
ax.text(dsy_.x[0,0], dsy_.y[0,0]-30, "GZT")

plt.title(vv+" "+tsel)

#%% Plot initial mask (with zoom and scatter of points)
dsx_plot = dsx_[vv].where(mask1_ref).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
dsx_plot.colorbar.set_label(dsx_[vv].units)
ax = plt.gca()
dsy_[vv].where(mask2_ref).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-7, 3), ylim=(-7, 3), add_colorbar=False)

plt.gca().set_ylabel("North-south distance from center [km]")
plt.gca().set_xlabel("West-east distance from center [km]")

x1 = dsx_.x.where(mask1_ref_).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).values.flatten()
y1 = dsx_.y.where(mask1_ref_).sel(time=tsel, method="nearest").where(dsx_["bca"]>bca_min).values.flatten()

x2 = dsy_.x.where(mask2_ref_).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).values.flatten()
y2 = dsy_.y.where(mask2_ref_).sel(time=tsel, method="nearest").where(dsy_["bca"]>bca_min).values.flatten()

ax.scatter(x1, y1, s=1, marker="o")
ax.scatter(x2, y2, s=1, c="r", marker="x")

plt.title(vv+" "+tsel)


#%% Rain path attenuation check

# One radar has to be the reference and the other must be the target, both below the ML
# Let's take GZT as reference

# Apply additional conditions
dsx_tg = dsx.where(dsx.Zm.fillna(0.) < 15)
dsy_rf = utils.apply_min_max_thresh(dsy.where(dsy.Zm.fillna(0.) < 15), {},
                                     {"PHIDP_OC_MASKED": 2})

vv = "DBZH" # Used to locate NaNs

mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx_tg,
                                                                                               dsy_rf,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref, matched_timesteps = utils.refine_radar_overlap_unique_NN_pairs(
                                                                        dsx_tg, dsy_rf,
                                                                        idx_tg, idx_rf,
                                                                        vv,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

# First look at the values
var_name="DBZH"
dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, var_name)

plt.scatter(dsx_p_tg, dsy_p_rf, alpha=0.1)
plt.plot([0,40], [0,40], c="red")
plt.xlabel("HTY")
plt.ylabel("GZT")

cc = round(np.corrcoef(dsx_p_tg[np.isfinite(dsx_p_tg)],dsy_p_rf[np.isfinite(dsy_p_rf)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

#%%% Plot scatter and boxplot of delta DBZH vs target PHI
vv = "PHIDP_OC_MASKED"

# build delta DBZH
delta_dbzh = dsx_p_tg-dsy_p_rf
delta_dbzh_flat = delta_dbzh.flatten()

dsx_p_tg_phi, dsy_p_rf_phi = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, vv)

dsx_p_tg_phi_flat = dsx_p_tg_phi.flatten()

# Scatterplot
plt.scatter(dsx_p_tg_phi_flat, delta_dbzh_flat, alpha=0.1)
#plt.plot([0,40], [0,40], c="red")
plt.xlabel(vv)
plt.ylabel("delta DBZH")

cc = round(np.corrcoef(dsx_p_tg_phi_flat[np.isfinite(dsx_p_tg_phi_flat)],
                       delta_dbzh_flat[np.isfinite(delta_dbzh_flat)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

# Box plots like in the paper
# Define bins
bins = np.arange(0, 6, 1)  # 0,1,2,3,4,5
bin_centers = bins[:-1] + 0.5

# Digitize dsx_p_tg_phi_flat into bins
bin_indices = np.digitize(dsx_p_tg_phi_flat, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh_flat[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Plot
plt.figure(figsize=(8, 5))
plt.boxplot(box_data, positions=bin_centers, widths=0.6)
plt.xlabel("dsx_p_tg_phi_flat (binned, 1° intervals)")
plt.ylabel("delta_dbzh_flat")
plt.title("Boxplots of delta_dbzh_flat vs dsx_p_tg_phi_flat bins")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bin_centers, [f"{b}-{b+1}" for b in bins[:-1]])

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

#%% Wet radome attenuation check

# One radar has to be the reference and the other must be the target, both below the ML
# Let's take GZT as reference

# Apply additional conditions
dsx_tg = dsx.where(dsx["PHIDP_OC_MASKED"] < 15)
dsy_rf = utils.apply_min_max_thresh(dsy.where(dsy.Zm.fillna(0.) < 5), {},
                                     {"PHIDP_OC_MASKED": 5})

vv = "DBZH" # Used to locate NaNs

mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx_tg,
                                                                                               dsy_rf,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref, matched_timesteps = utils.refine_radar_overlap_unique_NN_pairs(
                                                                        dsx_tg, dsy_rf,
                                                                        idx_tg, idx_rf,
                                                                        vv,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

# First look at the values
var_name="DBZH"
dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, var_name)

plt.scatter(dsx_p_tg, dsy_p_rf, alpha=0.1)
plt.plot([0,40], [0,40], c="red")
plt.xlabel("HTY")
plt.ylabel("GZT")

cc = round(np.corrcoef(dsx_p_tg[np.isfinite(dsx_p_tg)],dsy_p_rf[np.isfinite(dsy_p_rf)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

#%%% Plot scatter and boxplot of delta DBZH vs target PHI
vv = "PHIDP_OC_MASKED"

# build delta DBZH
delta_dbzh = dsx_p_tg-dsy_p_rf
delta_dbzh_flat = delta_dbzh.flatten()

dsx_p_tg_phi, dsy_p_rf_phi = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, vv)

dsx_p_tg_phi_flat = dsx_p_tg_phi.flatten()

# Scatterplot
plt.scatter(dsx_p_tg_phi_flat, delta_dbzh_flat, alpha=0.1)
#plt.plot([0,40], [0,40], c="red")
plt.xlabel(vv)
plt.ylabel("delta DBZH")

cc = round(np.corrcoef(dsx_p_tg_phi_flat[np.isfinite(dsx_p_tg_phi_flat)],
                       delta_dbzh_flat[np.isfinite(delta_dbzh_flat)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

# Box plots like in the paper
# Define bins
bins = np.arange(0, 6, 1)  # 0,1,2,3,4,5
bin_centers = bins[:-1] + 0.5

# Digitize dsx_p_tg_phi_flat into bins
bin_indices = np.digitize(dsx_p_tg_phi_flat, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh_flat[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Plot
plt.figure(figsize=(8, 5))
plt.boxplot(box_data, positions=bin_centers, widths=0.6)
plt.xlabel("dsx_p_tg_phi_flat (binned, 1° intervals)")
plt.ylabel("delta_dbzh_flat")
plt.title("Boxplots of delta_dbzh_flat vs dsx_p_tg_phi_flat bins")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bin_centers, [f"{b}-{b+1}" for b in bins[:-1]])

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

#%% ML layer attenuation check

# One radar has to be the reference above the ML and the other must be the target below the ML
# Take GZT as reference

# We need to re-do the initial thresholding changing the temperatures to negatives
# Apply thresholds before computing masks
TEMP_max = -2
dsx = utils.apply_min_max_thresh(ds1, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                        "SNRHC":SNRH_min, "SQIH":0.5,
                                        },
                                     {"CBB": CBB_max})
dsy = utils.apply_min_max_thresh(ds2, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                        "SNRHC":SNRH_min, "SQIH":0.5,
                                        },
                                     {"CBB": CBB_max})

# They consider that there is rain above the radar by looking at the
# median reflectivity in a circle 5km aroud each radar. Let's add this variable
dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })

# Let's also add the temperature value close to the radar as a coordinate
dsx = dsx.assign_coords({"Tm": ds1["TEMP"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
dsy = dsy.assign_coords({"Tm": ds2["TEMP"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })

# Add the additional DBZH threshold
dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                     {"TEMP":TEMP_max})
dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                     {"TEMP":TEMP_max})


# Apply additional conditions
#!!! Since we are not sure yet about the wet radome attenuation, we keep that filter
dsx_tg = dsx.where(dsx.Zm.fillna(0.) < 15).where(dsx.Tm>=3)
dsy_rf = dsy.where(dsy.Zm.fillna(0.) < 15).where(dsy.Tm<=TEMP_max)

vv = "DBZH" # Used to locate NaNs

mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx_tg,
                                                                                               dsy_rf,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref, matched_timesteps = utils.refine_radar_overlap_unique_NN_pairs(
                                                                        dsx_tg, dsy_rf,
                                                                        idx_tg, idx_rf,
                                                                        vv,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

# First look at the values
var_name="DBZH"
dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, var_name)

plt.scatter(dsx_p_tg, dsy_p_rf, alpha=0.1)
plt.plot([0,40], [0,40], c="red")
plt.xlabel("HTY")
plt.ylabel("GZT")

cc = round(np.corrcoef(dsx_p_tg[np.isfinite(dsx_p_tg)],dsy_p_rf[np.isfinite(dsy_p_rf)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

#%%% Plot scatter and boxplot of delta DBZH vs target PHI
vv = "PHIDP_OC_MASKED"

# build delta DBZH
delta_dbzh = dsx_p_tg-dsy_p_rf
delta_dbzh_flat = delta_dbzh.flatten()

dsx_p_tg_phi, dsy_p_rf_phi = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf, mask_tg_ref, mask_rf_ref,
                                           idx_tg_ref, idx_rf_ref, matched_timesteps, vv)

dsx_p_tg_phi_flat = dsx_p_tg_phi.flatten()

# Scatterplot
plt.scatter(dsx_p_tg_phi_flat, delta_dbzh_flat, alpha=0.1)
#plt.plot([0,40], [0,40], c="red")
plt.xlabel(vv)
plt.ylabel("delta DBZH")

cc = round(np.corrcoef(dsx_p_tg_phi_flat[np.isfinite(dsx_p_tg_phi_flat)],
                       delta_dbzh_flat[np.isfinite(delta_dbzh_flat)])[0,1],2).astype(str)
plt.text(0.5,0.1,"Corr coef (Pearson): "+cc, transform=ax.transAxes)
plt.title(var_name)

# Box plots like in the paper
# Define bins
bins = np.arange(0, 6, 1)  # 0,1,2,3,4,5
bin_centers = bins[:-1] + 0.5

# Digitize dsx_p_tg_phi_flat into bins
bin_indices = np.digitize(dsx_p_tg_phi_flat, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh_flat[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Plot
plt.figure(figsize=(8, 5))
plt.boxplot(box_data, positions=bin_centers, widths=0.6)
plt.xlabel("dsx_p_tg_phi_flat (binned, 1° intervals)")
plt.ylabel("delta_dbzh_flat")
plt.title("Boxplots of delta_dbzh_flat vs dsx_p_tg_phi_flat bins")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bin_centers, [f"{b}-{b+1}" for b in bins[:-1]])

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

#%% Let's repeat for all events
#%%% Make a dictionary with paths for each event

# Set a temporary folder where the selected values will be saved
savefolder = realpep_path+"/upload/jgiles/temp_compare_calibration_attenuation_adjacent_radars_alldates_multipleneighbors_new/"

reload = True # try to reload previous calculations?
calc = False # try to calculate if previous calculations failed?
NN = False # calculate only for nearest neighbors? If False, include all neighbors that fulfill the conditions

# First let's get all files
HTY_files = glob.glob(realpep_path+"/upload/jgiles/dmi/final_ppis/*/*/*/HTY/*/*/*allm*")
GZT_files = glob.glob(realpep_path+"/upload/jgiles/dmi/final_ppis/*/*/*/GZT/*/*/*allm*")

# Keep elevations that we want
def get_elev(path):
    return path.split("/")[-2]

HTY_elevs = ["1.5", "2.2", "3.0"]
HTY_files = [ff for ff in HTY_files if get_elev(ff) in HTY_elevs]

GZT_elevs = ["0.4", "0.5", "0.7"]
GZT_files = [ff for ff in GZT_files if get_elev(ff) in GZT_elevs]

# Define dates
ML_high_dates = [
    "2016-04-09", #### no valid matches: 3.0-0.7 # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-05-14",# ONLY GZT SURV 0.0 # no valid matches
    # "2016-05-15",# ONLY GZT SURV 0.0 # no valid matches
    # "2016-05-16",# ONLY GZT SURV 0.0 # no valid matches
    "2016-05-31",
    "2016-09-22", # Wet radome in GZT
    "2016-10-18", #### no valid matches: 3.0-0.5
    "2016-10-28",
    "2016-11-01", #### no valid matches: 2.2-0.5, 3.0-0.5
    # "2016-12-01", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    "2017-04-12",
    "2017-04-13",
    "2017-05-18",
    "2017-05-22", #### no valid matches: 2.2-0.5, 3.0-0.5
    # "2018-03-28",# NO POLARIMETRY OR RHOHV IN HTY
    # "2019-02-06",# NO POLARIMETRY OR RHOHV IN HTY
    # "2019-05-06",# NO POLARIMETRY OR RHOHV IN HTY
    "2019-10-17",
    "2019-10-20",
    "2019-10-21", #### no valid matches: 3.0-0.5
    "2019-10-22", #### no valid matches: 3.0-0.5
    "2019-10-28", #### no valid matches: 3.0-0.5
    "2020-03-12", #### no valid matches: 3.0-0.5
    "2020-03-13", #### no valid matches: 3.0-0.5
    # "2020-05-01", #### no valid matches: 1.5-0.5, 3.0-0.5 (all combinations)

    # New dates
    "2016-06-07",
    "2016-06-30",
    "2016-07-05",
    "2016-07-06",
    "2016-07-08",
    "2016-08-14",
    "2016-08-15",
    "2016-08-17",
    "2016-09-03",
    "2016-09-04",
    "2016-09-05",
    "2016-09-13",
    "2016-09-14",
    "2016-09-15",
    "2016-09-20",
    "2016-09-21",
    "2017-06-19",
    "2019-09-14",
    "2020-07-13",
    "2020-09-02",
    "2020-09-25",
    "2020-09-30",
    "2020-10-01"
]

# new clean list of dates where only dates with valid value-pairs after filtering are kept
# Dates with #< have only a few valid matches
ML_high_dates = [
    "2016-05-31", #<
    "2016-09-22", # Wet radome in GZT
    "2016-10-18", #< #### no valid matches: 3.0-0.5
    "2016-10-28",
    "2017-04-12",
    "2017-04-13", #<
    "2017-05-18", #<
    "2017-05-22", #<  #### no valid matches: 2.2-0.5, 3.0-0.5
    "2019-10-17", #<
    "2019-10-20", #<
    "2019-10-21", #### no valid matches: 3.0-0.5
    "2019-10-22", #< #### no valid matches: 3.0-0.5
    "2020-03-12", #### no valid matches: 3.0-0.5
    "2020-03-13", #### no valid matches: 3.0-0.5

    # New dates
    "2016-06-07", #<
    "2016-06-30", #<
    "2016-08-15", #<
    "2016-08-17",
    "2016-09-03",
    "2016-09-04",
    "2016-09-05", #<
    "2016-09-13",
    "2016-09-14",
    "2016-09-15",
    "2016-09-20", #<
    "2016-09-21", #<
    "2019-09-14", #<
    "2020-09-02", #<
    "2020-09-25", #<
    "2020-09-30", #<
    "2020-10-01" #<
]

# # X means that this date does not have ML low enough, so I removed it
# # XX means that the ML is not low enough but it is very close, I could leave it if it works.
ML_low_dates = [
    "2016-02-06", # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-02-21", # X # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-04-12", # X # ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
    # "2016-05-28", # X
    # "2016-11-30", # X
    # "2016-12-01", # X
    "2016-12-14",
    "2016-12-16",
    "2016-12-20",
    "2016-12-21",
    "2016-12-22",
    # "2016-12-26", # X
    "2016-12-27",
    "2016-12-29", # XX
    "2016-12-30", # XX
    "2016-12-31",
    "2017-01-01",
    "2017-01-02",
    # "2017-01-08", #### UNUSABLE ZDR IN GZT
    # "2017-01-11", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-01-20",  #### UNUSABLE ZDR IN GZT #### no valid matches: 1.5-0.5,
    # "2017-03-04", # XX #### UNUSABLE ZDR IN GZT
    # "2017-03-16", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-03-17", #### no valid matches: 1.5-0.5, 2.2-0.5, 3.0-0.5 (all combinations)
    # "2017-12-24", # X
    # "2017-12-31", # X
    # "2018-01-04", # X
    # "2018-01-05", # X
    # "2018-01-14", # X
    # "2019-11-27", # X
    # "2019-12-09", # X
    # "2019-12-13", # X
    # "2019-12-14", # X
    # "2019-12-24", # X
    # "2019-12-25", # X
    # "2019-12-26", # XX
    # "2019-12-28", # X
    # "2019-12-30", # X
    # "2019-12-31", # X
    # "2020-01-02", # XX
    # "2020-01-03", # XX
    # "2020-01-04", # XX
    # "2020-01-06", #### no valid matches: all combinations
    # "2020-01-07", # XX
    "2020-01-19", # XX
    # "2020-02-22", # XX
    # "2020-02-29", # XX
    # "2020-03-17", # X
    # "2020-11-04", # X
    # "2020-11-20", # X
    # "2020-12-14", # X

    # New dates
    "2016-12-25",
    "2016-12-26", # <= reactivated
    "2017-01-03",
    "2020-01-16",
    "2020-01-17",
    "2020-01-20",
    "2017-12-24", # <= reactivated
    "2019-12-28", # <= reactivated XXX mostly good ML detection with some ugly values
    "2019-12-31", # <= reactivated XXX mostly good ML detection with some ugly values
    "2020-01-02", # <= reactivated XXX mostly good ML detection with some ugly values
    "2020-01-03", # <= reactivated
    "2020-01-07", # <= reactivated
    # "2020-01-31", # ML not detected and far away from the 0 C line
    "2020-02-07",
    # "2020-02-08", # ML not detected and far away from the 0 C line
    "2020-02-29", # <= reactivated
    "2020-03-18",
    "2020-03-19",
    "2020-03-20",
]

# new clean list of dates where only dates with valid value-pairs after filtering are kept
ML_low_dates = [
    "2016-12-14",
    "2016-12-16",
    "2016-12-20",
    "2016-12-21",
    "2016-12-22",
    "2016-12-29",
    "2016-12-31",
    "2017-01-01",
    "2017-01-02",
    "2020-01-19", # XX

    # New dates
    "2016-12-25", # only 21 valid value pairs after filtering
    "2017-01-03",
    "2019-12-28",
    "2020-01-02", # only 21 valid value pairs after filtering
    "2020-02-07", # only 71 valid value pairs after filtering
    #"2020-03-20",  # only 95 valid value pairs after filtering and very ugly results.
]


#%%% Start the loop for dates for rain attenuation and wet radome analyses
token = secrets['EARTHDATA_TOKEN']

tsel = "2016-12-01T14" # for plots

tolerance = 250.
vv = "DBZH" # Used to locate and discard NaNs
SNRH_min = 15
RHOHV_min = 0.95
TEMP_min = 0 # preliminary filter, we can use the actual ML height later
DBZH_min = 10
CBB_max = 0.05

Zm_range = 1500. # range in m for the computation of Zm (DBZH close to radar)

vv_to_extract = ["DBZH", "DBZH_AC",
                 "ZDR_EC", "ZDR_EC_OC", "ZDR_EC_OC_AC",
                 "ZDR_EC_OCnoWR",
                 "PHIDP_OC_MASKED", #"PHIDP_OC", "PHIDP_OC_SMOOTH",
                 "Zm",
                 "TEMP", "TEMPm",
                 "TEMP_p25", "TEMP_p75", "TEMP_p50", # percentiles of TEMP along the ray
                 "ZDR_EC_OC_p25", "ZDR_EC_OC_p75", "ZDR_EC_OC_p50", # percentiles of ZDR along the ray
                 "ZDR_EC_OC_mpath", # weighted mean of ZDR along the path
                 "z", "z_beamtop",
                 "height_ml_bottom_new_gia", "height_ml_bottom_new_gia_fromqvp",
                 "RHOHV",
                 "binvol", "beam_cross_angle",
                 "riming"
                 ] # all variables to extract from the datasets, DBZH must be the first

elev_ml_bottom_fromqvp = ["10.0", "12.0", "8.0", "7.0", "15.0"] # elevations to try to load the height of the ML from QVP files, in order of preference

# Some dates do not have reliable ML heights from QVPs, replace them by NaNs
remove_ml_dates = {
    "2016-06-07": (np.nan, np.nan),
    "2016-08-15": (np.nan, np.nan),
    "2016-09-03": (np.nan, np.nan),
    "2016-09-05": (np.nan, np.nan),
    "2016-09-13": (np.nan, np.nan),
    "2016-09-14": (np.nan, np.nan),
    "2016-09-15": (np.nan, 1), # in this case HTY is bad but GZT is good
    "2016-09-21": (np.nan, 1)
    }

selected_ML_high = {vi:[] for vi in vv_to_extract}

selected_ML_high_dates = {} # to collect dates info and number of valid points

start_time = time.time()

if calc:
    if "ZDR_EC_OCnoWR" in vv_to_extract:
        # We load ZDR offsets again for alternative offset correction

        print("Loading ZDR daily offsets")

        ds1_zdr_offsets_lr_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_belowML_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_below1C_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_belowML_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_below1C_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")

        ds2_zdr_offsets_lr_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_belowML_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_below1C_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_belowML_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_below1C_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")

        # # plot running medians to check smoothing
        # ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()

        # Combine to create a single offset timeseries
        ds1_zdr_offsets_lr = ds1_zdr_offsets_lr_ml_nowr.resample(time="D").mean()\
            .fillna(ds1_zdr_offsets_lr_ml.resample(time="D").mean())\
                .fillna(ds1_zdr_offsets_lr_1c_nowr.resample(time="D").mean())\
                    .fillna(ds1_zdr_offsets_lr_1c.resample(time="D").mean())
        ds1_zdr_offsets_qvp = ds1_zdr_offsets_qvp_ml_nowr.resample(time="D").mean()\
            .fillna(ds1_zdr_offsets_qvp_ml.resample(time="D").mean())\
                .fillna(ds1_zdr_offsets_qvp_1c_nowr.resample(time="D").mean())\
                    .fillna(ds1_zdr_offsets_qvp_1c.resample(time="D").mean())
        ds1_zdr_offsets_qvp = ds1_zdr_offsets_qvp.where(ds1_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
        ds1_zdr_offsets_comb = xr.where(ds1_zdr_offsets_lr["ZDR_offset_datacount"] >= ds1_zdr_offsets_qvp["ZDR_offset_datacount"],
                                    ds1_zdr_offsets_lr["ZDR_offset"],
                                    ds1_zdr_offsets_qvp["ZDR_offset"]).fillna(ds1_zdr_offsets_lr["ZDR_offset"]).fillna(ds1_zdr_offsets_qvp["ZDR_offset"])

        ds2_zdr_offsets_lr = ds2_zdr_offsets_lr_ml_nowr.resample(time="D").mean()\
            .fillna(ds2_zdr_offsets_lr_ml.resample(time="D").mean())\
                .fillna(ds2_zdr_offsets_lr_1c_nowr.resample(time="D").mean())\
                    .fillna(ds2_zdr_offsets_lr_1c.resample(time="D").mean())
        ds2_zdr_offsets_qvp = ds2_zdr_offsets_qvp_ml_nowr.resample(time="D").mean()\
            .fillna(ds2_zdr_offsets_qvp_ml.resample(time="D").mean())\
                .fillna(ds2_zdr_offsets_qvp_1c_nowr.resample(time="D").mean())\
                    .fillna(ds2_zdr_offsets_qvp_1c.resample(time="D").mean())
        ds2_zdr_offsets_qvp = ds2_zdr_offsets_qvp.where(ds2_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
        ds2_zdr_offsets_comb = xr.where(ds2_zdr_offsets_lr["ZDR_offset_datacount"] >= ds2_zdr_offsets_qvp["ZDR_offset_datacount"],
                                    ds2_zdr_offsets_lr["ZDR_offset"],
                                    ds2_zdr_offsets_qvp["ZDR_offset"]).fillna(ds2_zdr_offsets_lr["ZDR_offset"]).fillna(ds2_zdr_offsets_qvp["ZDR_offset"])

        # finally, smooth it out (I think we dont need to smooth it)
        # ds1_zdr_offsets_comb_smooth = ds1_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()
        # ds2_zdr_offsets_comb_smooth = ds2_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()

        # Plots, for checking
        # ds1_zdr_offsets_lr.ZDR_offset.sel(time=slice("2016-12-01", "2017-01-31")).plot(marker="o")
        # ds1_zdr_offsets_qvp.ZDR_offset.sel(time=slice("2016-12-01", "2017-01-31")).plot(marker="x")
        # ds1_zdr_offsets_comb.sel(time=slice("2016-12-01", "2017-01-31")).plot(marker=".")
        # plt.grid()

    if "height_ml_bottom_new_gia_fromqvp" in vv_to_extract:
        print("Loading ML bottom heights from QVP stats")

        ds1_mlb_qvp = xr.open_dataset(realpep_path+"/upload/jgiles/radar_stats/stratiform_ML/hty_ML_bottom.nc")
        ds2_mlb_qvp = xr.open_dataset(realpep_path+"/upload/jgiles/radar_stats/stratiform_ML/gzt_ML_bottom.nc")

    if "riming" in vv_to_extract:
        ## Process the riming classification
        print("... Loading pre-calculated riming ...")
        riming_classif = {}
        X_DBZH = "DBZH_AC"
        X_ZDR = "ZDR_EC_OC_WRC_AC"
        suffix_name = "_calibration_attenuation_HTYGZT"

        for stratname in ["unfiltered"]:
            if stratname not in riming_classif.keys():
                riming_classif[stratname] = {}
            elif type(riming_classif[stratname]) is not dict:
                riming_classif[stratname] = {}
            print("Loading "+stratname+" riming classification ...")
            for ll in ["hty"]: # ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
                if ll not in riming_classif[stratname].keys():
                    riming_classif[stratname][ll] = xr.Dataset()
                elif type(riming_classif[stratname][ll]) is not xr.Dataset:
                    riming_classif[stratname][ll] = xr.Dataset()

                for xx in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_'+X_ZDR+'_'+X_DBZH,
                           ]:
                    try:
                        riming_classif[stratname][ll] = riming_classif[stratname][ll].assign( xr.open_dataset(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/"+stratname+"/"+ll+"_"+xx+".nc") )
                        print(ll+" "+xx+" riming_classif loaded")
                    except:
                        if stratname == "unfiltered":
                            pass

                # delete entry if empty
                if not riming_classif[stratname][ll]:
                    del riming_classif[stratname][ll]


        selected_ML_high_riming = {}

        for date in ML_high_dates:
            # 1. Clean the data
            riming_date = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time=date).copy().dropna("z", how="all").fillna(0)
            cleaned_mask = binary_opening(riming_date.values, structure=np.ones((3, 3))).astype(float)

            # Put the cleaned numpy array back into a DataArray so we can use xarray's spatial logic
            da_clean = riming_date.copy(data=cleaned_mask)

            # 2. Define the Melting Layer boundaries
            ml_top = da_clean.height_ml_new_gia_clean
            roi_top = ml_top + 300

            # 3. Create a boolean mask for the 300m region of interest (ROI) above the ML
            roi_mask = (da_clean.z >= ml_top) & (da_clean.z <= roi_top)

            # 4. Calculate the fraction of rimed pixels (1s) within that specific 300m layer per time step
            # Sum of 1s in the ROI
            riming_in_roi = (da_clean == 1).where(roi_mask, 0).sum(dim="z")
            # Total number of pixels in the ROI
            total_in_roi = roi_mask.sum(dim="z")

            # Calculate fraction (using .where to avoid division by zero if a profile has no valid z pixels in that range)
            riming_fraction = riming_in_roi / total_in_roi.where(total_in_roi > 0)

            # 5. Identify the profiles (times) that meet >= 80% condition
            valid_profiles = riming_fraction >= 0.8

            # 6. assign the final values
            selected_ML_high_riming[date] = valid_profiles.copy()

for date in ML_high_dates:
    print("Processing "+date)
    HTY_files0 = [ff for ff in HTY_files if date in ff]
    GZT_files0 = [ff for ff in GZT_files if date in ff]

    selected_ML_high_dates[date] = []

    for HTY_file in HTY_files0:
        for GZT_file in GZT_files0:

            # Create save folder
            sf = savefolder+"ML_high_dates/"
            if not os.path.exists(sf):
                os.makedirs(sf)

            if reload:
                dbzh_loaded = False
                for vi in vv_to_extract:
                    # if "_OCnoWR" in vi:
                    #     vi_ = "".join(vi.split("_OCnoWR"))
                    #     vi_OC = "".join(vi.split("noWR"))
                    #     selected_ML_high[vi].append( (selected_ML_high[], ) )
                    sfp_tg = sf+"_".join([vi, "tg", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                    sfp_ref = sf+"_".join([vi, "ref", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                    try:
                        # if ML heigh from QVP, check if it good according to list of dates
                        if vi == "height_ml_bottom_new_gia_fromqvp" and date in remove_ml_dates:
                            selected_ML_high[vi].append( (np.load(sfp_tg+".npy")*remove_ml_dates[date][0], np.load(sfp_ref+".npy")*remove_ml_dates[date][1] ) )
                        else:
                            selected_ML_high[vi].append( (np.load(sfp_tg+".npy"), np.load(sfp_ref+".npy") ) )
                        if vi == "DBZH":
                            dbzh_loaded = True
                            selected_ML_high_dates[date].append( ( "HTY "+HTY_file.split("/")[-2],
                                                                  "GZT "+GZT_file.split("/")[-2],
                                                                  np.isfinite(selected_ML_high[vi][-1][0]).sum()) )
                    except:
                        print(vi+": reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy \n failed")
                if dbzh_loaded:
                    continue
                if not calc:
                    print("Total fail reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy")
                    continue
                print("Total fail reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy \n attempting to calculate")

            # Load the data
            ds1 = xr.open_mfdataset(HTY_file)
            ds2 = xr.open_mfdataset(GZT_file)

            # Get PPIs into the same reference system
            proj = utils.get_common_projection(ds1, ds2)

            ds1 = wrl.georef.georeference(ds1, crs=proj)
            ds2 = wrl.georef.georeference(ds2, crs=proj)

            # Add new offset/atten corrected ZDR in datasets
            if "ZDR_EC_OCnoWR" in vv_to_extract:
                ds1 = ds1.assign({"ZDR_EC_OCnoWR":
                                  ds1["ZDR_EC"] - ds1_zdr_offsets_comb.sel(time=ds1.time[0].values.astype(str)[:10]).mean()} )

                ds2 = ds2.assign({"ZDR_EC_OCnoWR":
                                  ds2["ZDR_EC"] - ds2_zdr_offsets_comb.sel(time=ds2.time[0].values.astype(str)[:10]).mean()} )

            # Add beam blockage
            ds1_pbb, ds1_cbb = beam_blockage_from_radar_ds(ds1.isel(time=0),
                                                           (ds1.longitude, ds1.latitude, ds1.altitude),
                                                           wradlib_token = token)

            ds1 = ds1.assign({"PBB": ds1_pbb, "CBB": ds1_cbb})

            ds2_pbb, ds2_cbb = beam_blockage_from_radar_ds(ds2.isel(time=0),
                                                           (ds2.longitude, ds2.latitude, ds2.altitude),
                                                           wradlib_token = token)

            ds2 = ds2.assign({"PBB": ds2_pbb, "CBB": ds2_cbb})

            # Apply thresholds before computing masks
            dsx = utils.apply_min_max_thresh(ds1, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                                    "SNRHC":SNRH_min, "SQIH":0.5,
                                                    "TEMP":TEMP_min},
                                                 {"CBB": CBB_max})
            dsy = utils.apply_min_max_thresh(ds2, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                                    "SNRHC":SNRH_min, "SQIH":0.5,
                                                    "TEMP":TEMP_min},
                                                 {"CBB": CBB_max})

            # They consider that there is rain above the radar by looking at the
            # median reflectivity in a circle around each radar. Let's add this variable
            if "Zm" not in dsx.coords:
                dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
            if "Zm" not in dsy.coords:
                dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")).broadcast_like(dsy["DBZH"]) })

            # add the TEMP close to the radar if requested
            if "TEMPm" in vv_to_extract:
                dsx = dsx.assign_coords({"TEMPm": dsx["TEMP"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
                dsy = dsy.assign_coords({"TEMPm": dsy["TEMP"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")).broadcast_like(dsy["DBZH"]) })

            # add ML height from QVP
            if "height_ml_bottom_new_gia_fromqvp" in vv_to_extract:
                for qvp_elev in elev_ml_bottom_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(HTY_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            dsx.coords["height_ml_bottom_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_bottom_new_gia"].interp_like(dsx.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_bottom_new_gia_fromqvp" not in dsx.coords: # if that did not work, just fill with NaNs
                    dsx.coords["height_ml_bottom_new_gia_fromqvp"] = dsx.coords["height_ml_bottom_new_gia"]*np.nan

                for qvp_elev in elev_ml_bottom_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(GZT_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            dsy.coords["height_ml_bottom_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_bottom_new_gia"].interp_like(dsy.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_bottom_new_gia_fromqvp" not in dsy.coords: # if that did not work, just fill with NaNs
                    dsy.coords["height_ml_bottom_new_gia_fromqvp"] = dsy.coords["height_ml_bottom_new_gia"]*np.nan

            if "z_beamtop" in vv_to_extract:
                # we just copy the original coordinates and add half beamwidth, then georeference again
                dsx_beamtop = dsx["DBZH"].copy()
                dsx_beamtop['elevation'] = dsx_beamtop['elevation'] + 0.5
                dsx_beamtop = wrl.georef.georeference(dsx_beamtop, crs=proj)
                dsx.coords["z_beamtop"] = dsx_beamtop["z"].broadcast_like(dsx["DBZH"]).reset_coords(drop=True)

                dsy_beamtop = dsy["DBZH"].copy()
                dsy_beamtop['elevation'] = dsy_beamtop['elevation'] + 0.5
                dsy_beamtop = wrl.georef.georeference(dsy_beamtop, crs=proj)
                dsy.coords["z_beamtop"] = dsy_beamtop["z"].broadcast_like(dsy["DBZH"]).reset_coords(drop=True)

            if "binvol" in vv_to_extract:
                dsx.coords["binvol"] = dsx.range.wrl.qual.pulse_volume(dsx.range.diff("range").median(), 1)
                dsy.coords["binvol"] = dsy.range.wrl.qual.pulse_volume(dsy.range.diff("range").median(), 1)

            if "beam_cross_angle" in vv_to_extract:
                dsx.coords["beam_cross_angle"] = utils.compute_crossing_angle_cartesian(dsx, (dsx.x[:,0].mean(), dsx.y[:,0].mean(), dsx.z[:,0].mean()), (dsy.x[:,0].mean(), dsy.y[:,0].mean(), dsy.z[:,0].mean()))
                dsy.coords["beam_cross_angle"] = utils.compute_crossing_angle_cartesian(dsy, (dsx.x[:,0].mean(), dsx.y[:,0].mean(), dsx.z[:,0].mean()), (dsy.x[:,0].mean(), dsy.y[:,0].mean(), dsy.z[:,0].mean()))

            if "riming" in vv_to_extract:
                dsx.coords["riming"] = (selected_ML_high_riming[date].astype(int).interp_like(dsx.time)>0.5).astype(int)

            # Add the additional DBZH threshold
            dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                                 {})
            dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                                 {})

            # One radar has to be the reference and the other must be the target, both below the ML
            # Let's take GZT as reference

            # We will not apply additional Zm or PHIDP conditions now so we can use
            # the extracted values for both rain atten and wet radome analyses
            dsx_tg = dsx[[vv for vv in vv_to_extract if vv in dsx]].compute() # if we pre compute the variables that we want
            dsy_rf = dsy[[vv for vv in vv_to_extract if vv in dsy]].compute() # we save a lot of time (~3 times faster)

            if NN:
                mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx_tg,
                                                                                                               dsy_rf,
                                                                                    tolerance=tolerance,
                                                                                    tolerance_time=60*4)

                mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref = utils.refine_radar_overlap_unique_NN_pairs(
                                                                                        dsx_tg, dsy_rf,
                                                                                        idx_tg, idx_rf,
                                                                                        matched_timesteps,
                                                                                        vv,
                                                                                    tolerance_time=60*4,
                                                                                    z_tolerance=100.)
            else:
                mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap(dsx_tg,
                                                                                               dsy_rf,
                                                                                    tolerance=tolerance,
                                                                                    tolerance_time=60*4)

                mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref = utils.refine_radar_overlap(
                                                                                        dsx_tg, dsy_rf,
                                                                                        mask_tg, mask_rf,
                                                                                        matched_timesteps,
                                                                                        vv,
                                                                                        tolerance=tolerance,
                                                                                    tolerance_time=60*4,
                                                                                    z_tolerance=100.)

            if mask_tg_ref.sum() == 0:
                print("No matches found")
                continue # jump to next iteration if no pairs are found

            # compute percentile variables
            perc = []
            for vi in vv_to_extract:
                if re.search(r"_p\d{2}", vi):
                    perc += [re.search(r"_p\d{2}", vi).group()[-2:]]
            perc = np.unique(perc)

            for vi in vv_to_extract:
                if vi not in dsx_tg and re.search(r"_p\d{2}", vi):
                    vio = vi[:-4] # original variable name

                    # 1. Extract raw numpy arrays to avoid xarray overhead in the loop
                    dsx_tg_vals = dsx_tg[vio].values
                    dsy_rf_vals = dsy_rf[vio].values

                    # 2. Initialize output arrays with NaNs
                    # We'll create three separate arrays for the percentiles
                    shape_dsx = dsx_tg_vals.shape
                    shape_dsy = dsy_rf_vals.shape

                    dsx_tg_vals_pc = {}
                    dsy_rf_vals_pc = {}
                    for pc in perc:
                        dsx_tg_vals_pc[pc] = np.full(shape_dsx, np.nan, dtype=np.float32)
                        dsy_rf_vals_pc[pc] = np.full(shape_dsy, np.nan, dtype=np.float32)

                    # 3. Get the indices where the mask is True
                    t_idx, a_idx, r_idx = np.where(mask_tg_ref.values)
                    t_idy, a_idy, r_idy = np.where(mask_rf_ref.values)

                    # 4. Iterate only over the selected points
                    # We use a warning filter because slicing an array of pure NaNs
                    # will trigger a RuntimeWarning in np.nanpercentile
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        # for dsx
                        for t, a, r in zip(t_idx, a_idx, r_idx):
                            # Slice along the range dimension up to the current point (inclusive)
                            data_slice = dsx_tg_vals[t, a, :r+1]

                            # Calculate the percentiles, ignoring NaNs in the data
                            perc_dsx = np.nanpercentile(data_slice, [int(percx) for percx in perc])

                            # Assign the results to our output arrays
                            for pci, pc in enumerate(perc):
                                dsx_tg_vals_pc[pc][t, a, r] = perc_dsx[pci]

                        # for dsy
                        for t, a, r in zip(t_idy, a_idy, r_idy):
                            # Slice along the range dimension up to the current point (inclusive)
                            data_slice = dsy_rf_vals[t, a, :r+1]

                            # Calculate the percentiles, ignoring NaNs in the data
                            perc_dsy = np.nanpercentile(data_slice, [int(percy) for percy in perc])

                            # Assign the results to our output arrays
                            for pci, pc in enumerate(perc):
                                dsy_rf_vals_pc[pc][t, a, r] = perc_dsy[pci]

                    # 5. Reconstruct the results into a new xarray Dataset and assign to the original

                    dsx_percentiles = xr.Dataset(
                        {
                            vio+"_p"+pc: (["time", "azimuth", "range"],
                                          dsx_tg_vals_pc[pc]) for pc in perc
                        },
                        coords=dsx_tg[vio].coords
                    )

                    dsy_percentiles = xr.Dataset(
                        {
                            vio+"_p"+pc: (["time", "azimuth", "range"],
                                          dsy_rf_vals_pc[pc]) for pc in perc
                        },
                        coords=dsy_rf[vio].coords
                    )

                    dsx_tg = dsx_tg.assign( dsx_percentiles.copy() )
                    dsy_rf = dsy_rf.assign( dsy_percentiles.copy() )

                if vi not in dsx_tg and vi == "ZDR_EC_OC_mpath":
                    vio = vi.split("_mpath")[0] # original variable name

                    # 1. Extract raw numpy arrays to avoid xarray overhead in the loop
                    dsx_tg_vals = dsx_tg[vio].values
                    dsy_rf_vals = dsy_rf[vio].values

                    dsx_tg_valszhl = dsx_tg["DBZH"].pipe(wrl.trafo.idecibel).values
                    dsy_rf_valszhl = dsy_rf["DBZH"].pipe(wrl.trafo.idecibel).values

                    # 2. Initialize output arrays with NaNs
                    # We'll create three separate arrays for the percentiles
                    shape_dsx = dsx_tg_vals.shape
                    shape_dsy = dsy_rf_vals.shape

                    dsx_tg_vals_mpath = np.full(shape_dsx, np.nan, dtype=np.float32)
                    dsy_rf_vals_mpath = np.full(shape_dsy, np.nan, dtype=np.float32)

                    # 3. Get the indices where the mask is True
                    t_idx, a_idx, r_idx = np.where(mask_tg_ref.values)
                    t_idy, a_idy, r_idy = np.where(mask_rf_ref.values)

                    # 4. Iterate only over the selected points
                    # We use a warning filter because slicing an array of pure NaNs
                    # may trigger a RuntimeWarning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        # for dsx
                        for t, a, r in zip(t_idx, a_idx, r_idx):
                            # Slice along the range dimension up to the current point (inclusive)
                            data_slice = dsx_tg_vals[t, a, :r+1]
                            data_slice_zhl = dsx_tg_valszhl[t, a, :r+1]

                            # Calculate the weighted mean, ignoring NaNs in the data
                            dsx_tg_vals_mpath[t, a, r] = np.nansum(data_slice*data_slice_zhl)/np.nansum(data_slice_zhl)

                        # for dsy
                        for t, a, r in zip(t_idy, a_idy, r_idy):
                            # Slice along the range dimension up to the current point (inclusive)
                            data_slice = dsy_rf_vals[t, a, :r+1]
                            data_slice_zhl = dsy_rf_valszhl[t, a, :r+1]

                            # Calculate the weighted mean, ignoring NaNs in the data
                            dsy_rf_vals_mpath[t, a, r] = np.nansum(data_slice*data_slice_zhl)/np.nansum(data_slice_zhl)

                    # 5. Reconstruct the results into a new xarray Dataset and assign to the original

                    dsx_mpath = xr.Dataset(
                        {
                            vi: (["time", "azimuth", "range"],
                                          dsx_tg_vals_mpath)
                        },
                        coords=dsx_tg[vio].coords
                    )

                    dsy_mpath = xr.Dataset(
                        {
                            vi: (["time", "azimuth", "range"],
                                          dsy_rf_vals_mpath)
                        },
                        coords=dsy_rf[vio].coords
                    )

                    dsx_tg = dsx_tg.assign( dsx_mpath.copy() )
                    dsy_rf = dsy_rf.assign( dsy_mpath.copy() )

            # Extract variables
            for vi in vv_to_extract:
                if vi not in dsx_tg:
                    print(vi+" not found in target ds, filling with NaNs")
                    dsx_tg = dsx_tg.assign( { vi: xr.full_like(dsx_tg["DBZH"], fill_value=np.nan) } )
                if vi not in dsy_rf:
                    print(vi+" not found in reference ds, filling with NaNs")
                    dsy_rf = dsy_rf.assign( { vi: xr.full_like(dsy_rf["DBZH"], fill_value=np.nan) } )

                if NN:
                    dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf,
                                                                            mask_tg_ref, mask_rf_ref,
                                                               idx_tg_ref, idx_rf_ref,
                                                               matched_timesteps, vi)
                else:
                    dsx_p_tg, dsy_p_rf = utils.return_value_pairs(dsx_tg, dsy_rf,
                                                                  idx_tg_ref, idx_rf_ref,
                                                               matched_timesteps, vi)

                #!!! The arrays might have matched timesteps filled with NaNs (because of no valid pairs)
                # Improvement idea: remove the timesteps without valid pairs

                selected_ML_high[vi].append( (dsx_p_tg.copy(), dsy_p_rf.copy()) )

                # Save to temporary file
                sfp_tg = sf+"_".join([vi, "tg", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                sfp_ref = sf+"_".join([vi, "ref", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                np.save(sfp_tg,
                    selected_ML_high[vi][-1][0], allow_pickle=False)
                np.save(sfp_ref,
                    selected_ML_high[vi][-1][1], allow_pickle=False)

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%%% Plot boxplot of delta DBZH/ZDR vs target PHI (rain attenuation)
phi = "PHIDP_OC_MASKED"
dbzh = "DBZH" # DBZH, ZDR_EC_OC

yax = r"$Δ\mathrm{Z_{H}}\ [dBZ]$" # label for the y axis
xax = r"$\mathrm{\Phi_{DP}}\ [°]$" # label for the x axis

# we need to apply additional filters that we did not apply in the previous step
Zm_max = 15
ref_phi_max = 2

varx_range = (0, 19, 1) # start, stop, step
min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -15 # min and max limits for the y axis
ymax = 10

# extract/build necessary variables
delta_dbzh = np.concat([ (d1-d2).flatten() for d1,d2 in selected_ML_high[dbzh] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

tg_height_ml_bot = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

ref_height_ml_bot = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

tg_z = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z"] ])

ref_z = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z"] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

tg_z_beamtop = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

ref_z_beamtop = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

tg_binvol = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["binvol"] ])

ref_binvol = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["binvol"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

tg_ZDR25 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["ZDR_EC_OC_p25"] ])

ref_ZDR25 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["ZDR_EC_OC_p25"] ])

tg_ZDR75 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["ZDR_EC_OC_p75"] ])

ref_ZDR75 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["ZDR_EC_OC_p75"] ])

tg_ZDRmpath = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["ZDR_EC_OC_mpath"] ])

tg_riming = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["riming"] ])

# tg_height_ml_bot_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# ref_height_ml_bot_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa, and
# # fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = ref_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)]
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = tg_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)]
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 5000
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 5000

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_bot_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

ref_height_ml_bot_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_bot_qvp)):
    # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa
    tg_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])] = ref_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])]
    ref_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])] = tg_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_bot_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_bot_qvp[ts][:,0])
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_bot_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_bot_qvp[ts][:,0])
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_bot_qvp[ts] = pd.DataFrame(tg_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_bot_qvp[ts] = pd.DataFrame(ref_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_bot_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_bot_qvp])
ref_height_ml_bot_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_bot_qvp])

# fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 4000
ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 4000

# filter by valid values according to conditions
valid = np.isfinite(delta_dbzh) & (ref_phi<ref_phi_max) & (np.isfinite(tg_phi))\
        & (ref_Zm<Zm_max) & (tg_Zm<Zm_max)\
        & (tg_phi > varx_range[0]) & (tg_phi < varx_range[1] - varx_range[2])\
        & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMP > 3) & (ref_TEMP > 3) \
        & (tg_bca > 135) & (ref_bca > 135)\
        # & (tg_riming==0)
        # & (tg_ZDRmpath > 0) & (tg_ZDRmpath < 1)
        # & (tg_ZDR25 > -0.25) & (tg_ZDR25 < 0.25) & (tg_ZDR75 > 0) & (tg_ZDR75 < 5)
        # & (tg_binvol/ref_binvol > 0.5)
        # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\


# # And if we try to add the reverse matching? (GZT as tg and HTY as ref)
# # filter by valid values according to conditions (inverse)
# valid_ = np.isfinite(delta_dbzh) & (tg_phi<ref_phi_max) & (np.isfinite(ref_phi))\
#         & (ref_Zm<Zm_max) & (tg_Zm<Zm_max)\
#         & (ref_phi > varx_range[0]) & (ref_phi < varx_range[1] - varx_range[2])\
#         & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
#         & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
#         & (tg_TEMP > 3) & (ref_TEMP > 3)
#         # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\
#         # & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\ # loose way of avoiding the ML
#         # & (tg_TEMP > 3) & (ref_TEMP > 3)\ # loose way of avoiding the ML

# valid__ = ~valid & valid_ # when GZT is valid and HTY is not

# delta_dbzh[valid__] = delta_dbzh[valid__]*-1 # in those cases, invert the delta
# delta_dbzh = delta_dbzh[valid | valid__] # select all valids (for both radars)
# tg_phi[valid__] = ref_phi[valid__] # in those cases, assign the PHI from GZT
# tg_phi = tg_phi[valid | valid__] # select all valids (for both radars)



delta_dbzh = delta_dbzh[valid]
tg_phi = tg_phi[valid]

# Calculate best linear fit
lfit = np.polynomial.Polynomial.fit(tg_phi, delta_dbzh, 1)
lfit_str = str(lfit.convert()).replace("x", "Phi")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_phi into bins
bin_indices = np.digitize(tg_phi, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                     medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, bins)
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+dbzh+" vs "+phi+" bins")

# # add linear fit
# plt.plot([bins[0], bins[-1]], [lfit(bins[0]), lfit(bins[-1])])
# plt.text(0.95, 0.9, "Linear fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second linear fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 3)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
# plt.plot([bins[0], bins[-1]], [lfit_m(bins[0]), lfit_m(bins[-1])], c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third linear fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 3)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot([bins[0], bins[-1]], [lfit_mw(bins[0]), lfit_mw(bins[-1])], c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# Print p value and other stats
scipy.stats.linregress(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)])

# Print p value and other stats (for weighted fit)
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)]))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())

#%%% Confidence interval analysis based on different ranges (rain attenuation)
# INPUT DATA

phi = "PHIDP_OC_MASKED"
dbzh = "DBZH" # DBZH, ZDR_EC_OC

yax = r"Slope of $Δ\mathrm{Z_{H} - \Phi_{DP}}$ best fit [dBZ/°]" # label for the y axis
xax = r"Maximum $\mathrm{\Phi_{DP}}$ of fitted range [°]" # label for the x axis

weighted = True # weight the bins for the linear fittings? IQR or variance weighting (select in the code below)

# repeat filters
Zm_max = 15
ref_phi_max = 2

# extract/build necessary variables
delta_dbzh = np.concat([ (d1-d2).flatten() for d1,d2 in selected_ML_high[dbzh] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

tg_height_ml_bot = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

ref_height_ml_bot = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

tg_z = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z"] ])

ref_z = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z"] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

tg_z_beamtop = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

ref_z_beamtop = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

tg_binvol = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["binvol"] ])

ref_binvol = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["binvol"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

# tg_height_ml_bot_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# ref_height_ml_bot_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa, and
# # fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = ref_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)]
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = tg_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)]
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 5000
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 5000

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_bot_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

ref_height_ml_bot_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_bot_qvp)):
    # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa
    tg_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])] = ref_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])]
    ref_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])] = tg_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_bot_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_bot_qvp[ts][:,0])
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_bot_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_bot_qvp[ts][:,0])
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_bot_qvp[ts] = pd.DataFrame(tg_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_bot_qvp[ts] = pd.DataFrame(ref_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_bot_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_bot_qvp])
ref_height_ml_bot_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_bot_qvp])

# fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 4000
ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 4000

# filter by valid values according to conditions
valid = np.isfinite(delta_dbzh) & (ref_phi<ref_phi_max) & (np.isfinite(tg_phi))\
        & (ref_Zm<Zm_max) & (tg_Zm<Zm_max)\
        & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMP > 3) & (ref_TEMP > 3)\
        & (tg_bca > 135) & (ref_bca > 135)\
        # & (tg_binvol/ref_binvol > 0.5)
        # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\

delta_dbzh = delta_dbzh[valid]
tg_phi = tg_phi[valid]

# CONFIG
phi_ranges = [(0,18), (0,25), (0,30), (0,40), (0,50), (0,60)]   # φ ranges to test
bin_widths = [1,2,3]                                          # size of bins
B = 500                                                # bootstrap samples
ci_level = 95                                          # e.g. 95% CI
min_bin_n = 30                                          # e.g. 20 if we want to filter out low count bins

slopes_mean = {}
slopes_lowCI = {} # confidence interval based on global bootstrapping
slopes_highCI = {}
slopes2_lowCI = {} # confidence interval based on within-bin bootstrapping
slopes2_highCI = {}
phi_N = {} # Number of valid phi values in each range

# ---- FUNCTION: compute slope using bin-medians ----
def fit_binmedian_slope(phi_vals, dbzh_vals, phi_min, phi_max, bin_width, min_bin_n=0):
    """
    min_bin_n: bins with equal or less valid values than min_bin_n will be ignored.
    """

    # mask values inside selected phi range
    mask = (phi_vals >= phi_min) & (phi_vals < phi_max)
    phi_sel = phi_vals[mask]
    dbzh_sel = dbzh_vals[mask]

    # not enough samples -> return NaN
    if len(phi_sel) < 20:
        return np.nan

    # bins
    bins = np.arange(phi_min, phi_max + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2

    bin_idx = np.digitize(phi_sel, bins) - 1
    nbins = len(bins) - 1

    # compute medians
    medians = np.zeros(nbins)
    q25 = np.zeros(nbins)
    q75 = np.zeros(nbins)
    for i in range(nbins):
        vals = dbzh_sel[bin_idx == i]
        medians[i] = np.nanmedian(vals) if np.isfinite(vals).sum()>min_bin_n else np.nan
        q25[i] = np.nanquantile(vals, 0.25) if np.isfinite(vals).sum()>min_bin_n else np.nan
        q75[i] = np.nanquantile(vals, 0.75) if np.isfinite(vals).sum()>min_bin_n else np.nan
    iqr = q75-q25

    # remove empty bins
    valid = np.isfinite(medians)
    if np.sum(valid) < 2:
        return np.nan

    box_data = [dbzh_sel[bin_idx == i] for i in range(len(bins) - 1)]
    counts = [len(vals) for vals in box_data]
    variances = np.array([vals.var(ddof=1) for vals in box_data])
    weights = 1 / iqr**2 # 1 / variances # change to counts/variances for variance weighting
    weights[~np.isfinite(weights)] = 0
    w = np.sqrt(weights)

    # linear fit
    if weighted:
        p = np.polynomial.Polynomial.fit(bin_centers[valid], medians[valid], 1, w=w[valid])
    else:
        p = np.polynomial.Polynomial.fit(bin_centers[valid], medians[valid], 1)
    return p.convert().coef[1]  # slope is coef[1]

# -------------------------------------------------------------------
# MAIN LOOP OVER BIN SIZES AND PHI RANGES
# -------------------------------------------------------------------
for bin_width in bin_widths:
    print(f"Processing {bin_width}° bins ...")
    slopes_mean[bin_width] = []
    slopes_lowCI[bin_width] = []
    slopes_highCI[bin_width] = []
    slopes2_lowCI[bin_width] = []
    slopes2_highCI[bin_width] = []
    phi_N[bin_width] = []

    for (phi_min, phi_max) in phi_ranges:

        print(f"Processing φ-range {phi_min}–{phi_max}° ...")

        # ---- 1. Compute slope for the real dataset ----
        real_slope = fit_binmedian_slope(tg_phi, delta_dbzh, phi_min, phi_max, bin_width, min_bin_n=min_bin_n)

        # ---- 2. Bootstrap slopes (global resampling) ----
        boot_slopes = np.zeros(B)
        N = len(tg_phi)

        for b in range(B):
            # sample indices WITH replacement
            idx = np.random.randint(0, N, N)
            boot_phi = tg_phi[idx]
            boot_dbzh = delta_dbzh[idx]

            boot_slopes[b] = fit_binmedian_slope(
                boot_phi, boot_dbzh,
                phi_min, phi_max,
                bin_width,
                min_bin_n=min_bin_n
            )

        # remove NaNs from failed fits
        boot_slopes = boot_slopes[np.isfinite(boot_slopes)]

        # ---- 3. Compute confidence intervals ----
        low = np.percentile(boot_slopes, (100-ci_level)/2)
        high = np.percentile(boot_slopes, 100-(100-ci_level)/2)

        slopes_mean[bin_width].append(real_slope)
        slopes_lowCI[bin_width].append(low)
        slopes_highCI[bin_width].append(high)

        # ---- 4. Bootstrap slopes (within-bin resampling) ----
        mask = (tg_phi >= phi_min) & (tg_phi < phi_max)
        phi_sel = tg_phi[mask]
        dbzh_sel = delta_dbzh[mask]

        phi_N[bin_width].append(len(phi_sel))

        bins = np.arange(phi_min, phi_max + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width/2
        bin_idx = np.digitize(phi_sel, bins) - 1
        nbins = len(bin_centers)

        bin_samples = [dbzh_sel[bin_idx == i] for i in range(nbins)]

        boot_slopes = np.zeros(B)
        for b in range(B):
            # bootstrap each bin separately
            boot_medians = []
            boot_weights = []
            for vals in bin_samples:
                if np.isfinite(vals).sum()>min_bin_n:
                    idx = np.random.randint(0, len(vals), len(vals))
                    boot_vals = vals[idx]
                    boot_medians.append(np.median(boot_vals))
                    q25 = np.nanquantile(boot_vals, 0.25)
                    q75 = np.nanquantile(boot_vals, 0.75)
                    iqr = q75-q25
                    boot_counts = len(boot_vals)
                    boot_vars = boot_vals.var(ddof=1)
                    boot_weights.append(1/iqr**2) # change to boot_counts/boot_vars for variance weighting
                else:
                    boot_medians.append(np.nan)
                    boot_weights.append(np.nan)

            boot_medians = np.array(boot_medians)
            valid = np.isfinite(boot_medians)
            w = np.sqrt(np.array(boot_weights))

            if weighted:
                boot_slopes[b] = np.polynomial.Polynomial.fit(bin_centers[valid], boot_medians[valid], 1, w=w[valid]).convert().coef[1]
            else:
                boot_slopes[b] = np.polynomial.Polynomial.fit(bin_centers[valid], boot_medians[valid], 1).convert().coef[1]

        # ---- 5. Compute confidence intervals ----
        low = np.percentile(boot_slopes, (100-ci_level)/2)
        high = np.percentile(boot_slopes, 100-(100-ci_level)/2)

        slopes2_lowCI[bin_width].append(low)
        slopes2_highCI[bin_width].append(high)

# -------------------------------------------------------------------
# PLOT RESULTS
# -------------------------------------------------------------------
plt.figure(figsize=(6,3.5))
for bin_width in bin_widths:
    phi_max_values = [pr[1] for pr in phi_ranges]

    # eb1 = plt.errorbar(
    #     phi_max_values,
    #     slopes_mean[bin_width],
    #     yerr=[np.array(slopes_mean[bin_width])-np.array(slopes_lowCI[bin_width]),
    #           np.array(slopes_highCI[bin_width])-np.array(slopes_mean[bin_width])],
    #     fmt='o-', capsize=4, lw=3, capthick=3, label=f"{bin_width}° bins global bootstrap",
    #     alpha=0.5
    # )
    # eb2 = plt.errorbar(
    #     phi_max_values,
    #     slopes_mean[bin_width],
    #     yerr=[np.array(slopes_mean[bin_width])-np.array(slopes2_lowCI[bin_width]),
    #           np.array(slopes2_highCI[bin_width])-np.array(slopes_mean[bin_width])],
    #     fmt='none', capsize=4, lw=3, capthick=3, label=f"{bin_width}° bins within-bin bootstrap",
    #     ecolor=eb1[0].get_color()
    # )
    # eb2[-1][0].set_linestyle('--') # change linestyle of second error bars

    # alternative: plot only within-bin bootstrap in grayscale
    colors = ["lightgray", "gray", "black"]
    lws = [6, 4, 2]
    bini = bin_widths.index(bin_width)
    eb3 = plt.errorbar(
        phi_max_values,
        slopes_mean[bin_width],
        yerr=[np.array(slopes_mean[bin_width])-np.array(slopes2_lowCI[bin_width]),
              np.array(slopes2_highCI[bin_width])-np.array(slopes_mean[bin_width])],
        fmt='o-', capsize=lws[bini]+1, capthick=lws[bini], ms=lws[bini]+3,
        elinewidth=lws[bini], lw=lws[bini],
        label=f"{bin_width}° bins",
        color=colors[bini],
        zorder=10+bini
    )

    plt.xlabel(xax)
    plt.ylabel(yax)
    # plt.xlabel('Max φ of fitted range (°)')
    # plt.ylabel('Slope of Δ'+dbzh+'–φ (dB/°)')
    # plt.title('Bootstrap CI of Bin-Median Linear Fit Slope ('+str(B)+' Iterations)')
    # plt.legend(fontsize=6)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)

    plt.tight_layout()
    # plt.show()

    # print exact slope values
    print(f"#### {bin_width}° bins slope results ####")
    for x, n in zip(phi_max_values, slopes_mean[bin_width]):
        print("Range 0-"+str(x)+": "+str(round(n, 4)))
    print("Mean of all slopes: "+str(round(np.nanmean(slopes_mean[bin_width]), 4)))

# Add phi_N counts above x-tick labels (inside the plot area)
phi_N_ = [str(phi_N[bin_width][0])] + ["+"+str(pn0-phi_N[bin_width][0]) for pn0 in phi_N[bin_width][1:]]
for x, n in zip(phi_max_values, phi_N_):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray', zorder=20)

#%%% Plot boxplot of delta DBZH/ZDR vs target Zm (wet radome attenuation)
phi = "PHIDP_OC_MASKED"
dbzh = "DBZH_AC" # DBZH_AC, ZDR_EC_OC_AC

yax = r"$Δ\mathrm{Z_{H}}\ [dBZ]$" # label for the y axis
xax = r"$\mathrm{Z_{H}^m}\ [dBZ]$" # label for the x axis

# we need to apply additional filters that we did not apply in the previous step
ref_Zm_max = 5
ref_phi_max = 5
tg_phi_max = 15

#!!! We probably should make the bins smaller to better check the statistical significance
varx_range = (0, 45, 5) # start, stop, step
min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -15 # min and max limits for the y axis
ymax = 10

# extract/build necessary variables
delta_dbzh = np.concat([ (d1-d2).flatten() for d1,d2 in selected_ML_high[dbzh] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

tg_height_ml_bot = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

ref_height_ml_bot = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

tg_z = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z"] ])

ref_z = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z"] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

tg_z_beamtop = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

ref_z_beamtop = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

tg_binvol = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["binvol"] ])

ref_binvol = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["binvol"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

# tg_height_ml_bot_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# ref_height_ml_bot_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa, and
# # fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = ref_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)]
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = tg_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)]
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 5000
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 5000

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_bot_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

ref_height_ml_bot_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_bot_qvp)):
    # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa
    tg_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])] = ref_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])]
    ref_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])] = tg_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_bot_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_bot_qvp[ts][:,0])
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_bot_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_bot_qvp[ts][:,0])
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_bot_qvp[ts] = pd.DataFrame(tg_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_bot_qvp[ts] = pd.DataFrame(ref_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_bot_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_bot_qvp])
ref_height_ml_bot_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_bot_qvp])

# fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 4000
ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 4000

# filter by valid values according to conditions
valid = np.isfinite(delta_dbzh) & (ref_phi<ref_phi_max) & (tg_phi<tg_phi_max) & (ref_Zm<ref_Zm_max)\
        & (tg_Zm > varx_range[0]) & (tg_Zm < varx_range[1] - varx_range[2])\
        & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMP > 3) & (ref_TEMP > 3)\
        & (tg_bca > 135) & (ref_bca > 135)\
        # & (tg_binvol/ref_binvol > 0.5)
        # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\

# # And if we try to add the reverse matching? (GZT as tg and HTY as ref)
# # filter by valid values according to conditions (inverse)
# valid_ = np.isfinite(delta_dbzh) & (tg_phi<ref_phi_max) & (ref_phi<tg_phi_max) & (tg_Zm<ref_Zm_max)\
#         & (ref_Zm > varx_range[0]) & (ref_Zm < varx_range[1] - varx_range[2])\
#         & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
#         & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
#         & (tg_TEMP > 3) & (ref_TEMP > 3)\
#         & (tg_bca > 135) & (ref_bca > 135)\
#         # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\

# valid__ = ~valid & valid_ # when GZT is valid and HTY is not

# delta_dbzh[valid__] = delta_dbzh[valid__]*-1 # in those cases, invert the delta
# delta_dbzh = delta_dbzh[valid | valid__] # select all valids (for both radars)
# tg_Zm[valid__] = ref_Zm[valid__] # in those cases, assign the PHI from GZT
# tg_Zm = tg_Zm[valid | valid__] # select all valids (for both radars)


delta_dbzh = delta_dbzh[valid]
tg_Zm = tg_Zm[valid]

# Calculate best cuadratic fit
lfit = np.polynomial.Polynomial.fit(tg_Zm, delta_dbzh, 2)
lfit_str = str(lfit.convert()).replace("x", "Zm")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_Zm into bins
bin_indices = np.digitize(tg_Zm, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                 medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, bins)
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+dbzh+" vs Zm bins")

# # add cuadratic fit
# x_dense = np.linspace(bins[0], bins[-1], 100) # 100 points for a smooth curve
# plt.plot(x_dense, lfit(x_dense))
# plt.text(0.95, 0.9, "Cuadratic fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second cuadratic fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 2)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 5)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
x_dense = np.linspace(bins[0], bins[-1], 100) # 100 points for a smooth curve
# plt.plot(x_dense, lfit_m(x_dense), c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third cuadratic fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 2, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 5)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot(x_dense, lfit_m(x_dense), c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# Print p value and other stats
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x, x^2]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)], bin_centers[np.array(valid_bins)]**2))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x, x²): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())

#%%% Special handling of ZDR. Plot boxplot of delta ZDR vs target Zm (wet radome attenuation).
# same as before but we try to remove the offsets of ZDR when wet radome was affecting the radar
phi = "PHIDP_OC_MASKED"
zdr = "ZDR_EC" # not OC ZDR
zdr_to_plot = "ZDR_EC_OCnoWR_WRcorr" # ZDR_EC_OC_WRcorr (for checking the correction)
if "_new" in zdr_to_plot:
    zdr_oc = zdr_to_plot.split("_new")[0]
if "_WRcorr" in zdr_to_plot:
    zdr_oc = zdr_to_plot.split("_WRcorr")[0]

yax = r"$Δ\mathrm{Z_{DR}}\ [dB]$" # label for the y axis
xax = r"$\mathrm{Z_{H}^m}\ [dBZ]$" # label for the x axis

# we need to apply additional filters that we did not apply in the previous step
ref_Zm_max = 5
ref_phi_max = 5
tg_phi_max = 15

#!!! We probably should make the bins smaller to better check the statistical significance
varx_range = (0, 45, 5) # start, stop, step
min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

# custom atten corr based on the previous results
beta_new = 0.025 # to ignore this step set beta_new = 0

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -1 # min and max limits for the y axis
ymax = 3

# WR corr based on results
def zdr_wrc(Zm):
    Zm_ = np.where(np.nan_to_num(Zm) < 32.5,
                   np.nan_to_num(Zm),
                   32.5)
    return -0.00022*Zm_ + 0.00032*Zm_**2 # change here to adjust coefficients based on results

if "new" in zdr_to_plot:
    # Remove the timestep-based ZDR offsets and replace with daily offsets ignore the
    # wet-radome-affected timesteps (Zm high) for the target, leave the ref untouched

    selected_ML_high[zdr_oc+"_new"] = []
    for ti in range(len(selected_ML_high[zdr])):
        # get offsets and select only valid ones (no wet radome)
        tg_offsets_ti = selected_ML_high[zdr][ti][0] - selected_ML_high[zdr_oc][ti][0]
        tg_offsets_ti_valid = np.where(np.nan_to_num(selected_ML_high["Zm"][ti][0]) < ref_Zm_max,
                                       tg_offsets_ti, np.nan)
        tg_offsets_ti_ts = np.nanmean(tg_offsets_ti_valid, 1) # reduce to one offset per timestep

        # generate new ZDR offset by using a rolling mean
        tg_new_offsets_ti = pd.Series(tg_offsets_ti_ts).rolling(5, min_periods=1).mean().to_numpy()

        # fill any other remaining nan with a daily-mean offset
        tg_new_offsets_ti = np.where(np.isfinite(tg_new_offsets_ti),
                                     tg_new_offsets_ti,
                                     np.nanmean(tg_offsets_ti_ts))

        # use the new offsets where there is wet radome
        tg_zdr_oc_new_ti = np.where(np.nan_to_num(selected_ML_high["Zm"][ti][0]) < ref_Zm_max,
                                       selected_ML_high[zdr_oc][ti][0],
                                       selected_ML_high[zdr][ti][0] - np.expand_dims(tg_new_offsets_ti, 1))

        # add to the new variable
        selected_ML_high[zdr_oc+"_new"].append((tg_zdr_oc_new_ti.copy(), selected_ML_high[zdr_oc][ti][1]))

if "_WRcorr" in zdr_to_plot:
    # Correct wet-radome timesteps

    selected_ML_high[zdr_oc+"_WRcorr"] = []
    for ti in range(len(selected_ML_high[zdr])):
        # add to the new variable
        selected_ML_high[zdr_oc+"_WRcorr"].append((selected_ML_high[zdr_oc][ti][0].copy() - zdr_wrc(selected_ML_high["Zm"][ti][0].copy()),
                                                selected_ML_high[zdr_oc][ti][1]))

# extract/build necessary variables
tg_zdr = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[zdr_to_plot] ])

ref_zdr = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[zdr_to_plot] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

tg_height_ml_bot = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

ref_height_ml_bot = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

tg_z = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z"] ])

ref_z = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z"] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

tg_z_beamtop = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

ref_z_beamtop = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z_beamtop"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

# tg_height_ml_bot_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# ref_height_ml_bot_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ])

# # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa, and
# # fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = ref_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)]
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = tg_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)]
# tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 5000
# ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 5000

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_bot_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

ref_height_ml_bot_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_bot_qvp)):
    # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa
    tg_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])] = ref_height_ml_bot_qvp[ts][np.isnan(tg_height_ml_bot_qvp[ts])]
    ref_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])] = tg_height_ml_bot_qvp[ts][np.isnan(ref_height_ml_bot_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_bot_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_bot_qvp[ts][:,0])
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_bot_qvp[ts][tg_height_ml_bot_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_bot_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_bot_qvp[ts][:,0])
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_bot_qvp[ts][ref_height_ml_bot_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_bot_qvp[ts] = pd.DataFrame(tg_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_bot_qvp[ts] = pd.DataFrame(ref_height_ml_bot_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_bot_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_bot_qvp])
ref_height_ml_bot_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_bot_qvp])

# fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
tg_height_ml_bot_qvp[np.isnan(tg_height_ml_bot_qvp)] = 4000
ref_height_ml_bot_qvp[np.isnan(ref_height_ml_bot_qvp)] = 4000

if beta_new > 0:
    zdr_to_plot = zdr_to_plot+"_AC"
    delta_zdr = (tg_zdr + beta_new*tg_phi) - (ref_zdr + beta_new*ref_phi)
else:
    delta_zdr = tg_zdr - ref_zdr

# filter by valid values according to conditions
valid = np.isfinite(delta_zdr) & (ref_phi<ref_phi_max) & (tg_phi<tg_phi_max) & (ref_Zm<ref_Zm_max)\
        & (tg_Zm > varx_range[0]) & (tg_Zm < varx_range[1] - varx_range[2])\
        & (tg_z < tg_height_ml_bot_qvp) & (ref_z < ref_height_ml_bot_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMP > 3) & (ref_TEMP > 3)\
        & (tg_bca > 135) & (ref_bca > 135)\
        # & (tg_z_beamtop < tg_height_ml_bot_qvp) & (ref_z_beamtop < ref_height_ml_bot_qvp)\
        # & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\ # loose way of avoiding the ML
        # & (tg_TEMP > 3) & (ref_TEMP > 3)\ # loose way of avoiding the ML

delta_zdr = delta_zdr[valid]
tg_Zm = tg_Zm[valid]

# Calculate best cuadratic fit
lfit = np.polynomial.Polynomial.fit(tg_Zm, delta_zdr, 2)
lfit_str = str(lfit.convert()).replace("x", "Zm")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_Zm into bins
bin_indices = np.digitize(tg_Zm, bins) - 1

# Prepare data for boxplot
box_data = [delta_zdr[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                 medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, bins)
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+zdr_to_plot+" vs Zm bins")

# # add cuadratic fit
# x_dense = np.linspace(bins[0], bins[-1], 100) # 100 points for a smooth curve
# plt.plot(x_dense, lfit(x_dense))
# plt.text(0.95, 0.9, "Cuadratic fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second cuadratic fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 2)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 5)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
x_dense = np.linspace(bins[0], bins[-1], 100) # 100 points for a smooth curve
# plt.plot(x_dense, lfit_m(x_dense), c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third cuadratic fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 2, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 5)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot(x_dense, lfit_m(x_dense), c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# Print p value and other stats
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x, x^2]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)], bin_centers[np.array(valid_bins)]**2))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x, x²): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())

#%%% Start the loop for dates for ML attenuation
token = secrets['EARTHDATA_TOKEN']

# New alpha and beta values for atten correction in rain, based on the previous results.
new_alpha = 0.14
new_beta = 0.025

tsel = "2016-12-01T14" # for plots

tolerance = 250.
vv = "DBZH" # Used to locate and discard NaNs
SNRH_min = 15
RHOHV_min = 0.95
TEMP_max = 1 # for this we need a max temp, we want to select above the ML (not very precise, just a rough first estimate)
DBZH_min = 10
CBB_max = 0.05

Zm_range = 1500. # range in m for the computation of Zm (DBZH close to radar)

vv_to_extract = ["DBZH", "DBZH_AC_rain", "DBZH_AC",
                 "DBZH_AC2_rain",
                 "ZDR_EC", "ZDR_EC_AC_rain",
                 "ZDR_EC_OC", "ZDR_EC_OC_AC", "ZDR_EC_OC_AC_rain",
                 "ZDR_EC_OC_AC2_rain",
                 "ZDR_EC_OC2", "ZDR_EC_OC2_AC2_rain", # ZDR corrected with extrapolated offsets
                 "ZDR_EC_OC3", "ZDR_EC_OC3_AC2_rain", # ZDR corrected with extrapolated offsets and manual offsets for some dates
                 "PHIDP_OC_MASKED", #"PHIDP_OC",
                 "Zm",
                 "TEMP", "TEMPm", "z",
                 "height_ml_bottom_new_gia", "height_ml_new_gia",
                 "z_beambot",
                 "height_ml_new_gia_fromqvp", "height_ml_bottom_new_gia_fromqvp",
                 "PHIDP_OC_MASKED_MLbump", #"PHIDP_OC_SMOOTH_MLbump",
                 "PHIDP_OC_MASKED_MLbump_safer",
                 "PHIDP_OC_MASKED_MLbump_strict",
                 # "DBZH_AC_rain_MLmax",
                 # "DBZH_AC2_rain_MLmax",
                 # "RHOHV_MLmin",
                 "RHOHV",
                 "binvol", "beam_cross_angle",
                 "riming",
                 "range_MLbump", "range_MLbump_safer", # safer means using beam top and bottom to avoid any contact with the QVP-ML
                 "range_MLbump_strict", # strict means getting as close to the QVP-ML as possible (beam center and QVP-based ML heights)
                 ] # all variables to extract from the datasets, DBZH must be the first

elev_ml_top_fromqvp = ["10.0", "12.0", "8.0", "7.0", "15.0"] # elevations to try to load the height of the ML from QVP files, in order of preference

# Some dates do not have reliable ML heights from QVPs, replace them by NaNs
remove_ml_dates = {
    "2017-01-03": (np.nan, np.nan),
    "2020-02-07": (np.nan, np.nan),
    "2020-03-20": (np.nan, np.nan),
    }

selected_ML_low = {vi:[] for vi in vv_to_extract}

selected_ML_low_dates = {} # to collect dates info and number of valid points

start_time = time.time()

if calc:
    if "ZDR_EC_OC2" in vv_to_extract or "ZDR_EC_OC3" in vv_to_extract:
        # These dates should not have valid ZDR calibrations in GZT due to the low ML.
        # Then, we load all daily calibrations available in the period to approximate the
        # calibration with smoothing and interpolation.

        print("Loading ZDR daily offsets for NaN filling")

        ds1_zdr_offsets_lr_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_belowML_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_below1C_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_belowML_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_belowML-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_below1C_noWR-*-HTY-h5netcdf.nc")
        ds1_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/HTY/*/*/*-zdr_offset_below1C-*-HTY-h5netcdf.nc")

        ds2_zdr_offsets_lr_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_belowML_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_below1C_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_lr_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/LR_consistency/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_ml_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_belowML_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_ml = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_belowML-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_1c_nowr = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_below1C_noWR-*-GZT-h5netcdf.nc")
        ds2_zdr_offsets_qvp_1c = xr.open_mfdataset(realpep_path+"/upload/jgiles/dmi/calibration/zdr/QVP/*/*/*/GZT/*/*/*-zdr_offset_below1C-*-GZT-h5netcdf.nc")

        # # plot running medians to check smoothing
        # ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_lr_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_ml.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()
        # ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").plot(); ds2_zdr_offsets_qvp_1c.ZDR_offset.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median().plot()

        # Combine to create a single offset timeseries
        ds1_zdr_offsets_lr = ds1_zdr_offsets_lr_ml_nowr.resample(time="D").mean()\
            .fillna(ds1_zdr_offsets_lr_ml.resample(time="D").mean())\
                .fillna(ds1_zdr_offsets_lr_1c_nowr.resample(time="D").mean())\
                    .fillna(ds1_zdr_offsets_lr_1c.resample(time="D").mean())
        ds1_zdr_offsets_qvp = ds1_zdr_offsets_qvp_ml_nowr.resample(time="D").mean()\
            .fillna(ds1_zdr_offsets_qvp_ml.resample(time="D").mean())\
                .fillna(ds1_zdr_offsets_qvp_1c_nowr.resample(time="D").mean())\
                    .fillna(ds1_zdr_offsets_qvp_1c.resample(time="D").mean())
        ds1_zdr_offsets_qvp = ds1_zdr_offsets_qvp.where(ds1_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
        ds1_zdr_offsets_comb = xr.where(ds1_zdr_offsets_lr["ZDR_offset_datacount"] >= ds1_zdr_offsets_qvp["ZDR_offset_datacount"],
                                    ds1_zdr_offsets_lr["ZDR_offset"],
                                    ds1_zdr_offsets_qvp["ZDR_offset"]).fillna(ds1_zdr_offsets_lr["ZDR_offset"]).fillna(ds1_zdr_offsets_qvp["ZDR_offset"])

        ds2_zdr_offsets_lr = ds2_zdr_offsets_lr_ml_nowr.resample(time="D").mean()\
            .fillna(ds2_zdr_offsets_lr_ml.resample(time="D").mean())\
                .fillna(ds2_zdr_offsets_lr_1c_nowr.resample(time="D").mean())\
                    .fillna(ds2_zdr_offsets_lr_1c.resample(time="D").mean())
        ds2_zdr_offsets_qvp = ds2_zdr_offsets_qvp_ml_nowr.resample(time="D").mean()\
            .fillna(ds2_zdr_offsets_qvp_ml.resample(time="D").mean())\
                .fillna(ds2_zdr_offsets_qvp_1c_nowr.resample(time="D").mean())\
                    .fillna(ds2_zdr_offsets_qvp_1c.resample(time="D").mean())
        ds2_zdr_offsets_qvp = ds2_zdr_offsets_qvp.where(ds2_zdr_offsets_qvp["ZDR_offset"] < 2) # there is an extreme value in one date, lets remove it
        ds2_zdr_offsets_comb = xr.where(ds2_zdr_offsets_lr["ZDR_offset_datacount"] >= ds2_zdr_offsets_qvp["ZDR_offset_datacount"],
                                    ds2_zdr_offsets_lr["ZDR_offset"],
                                    ds2_zdr_offsets_qvp["ZDR_offset"]).fillna(ds2_zdr_offsets_lr["ZDR_offset"]).fillna(ds2_zdr_offsets_qvp["ZDR_offset"])

        # finally, smooth it out
        ds1_zdr_offsets_comb_smooth = ds1_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()
        ds2_zdr_offsets_comb_smooth = ds2_zdr_offsets_comb.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()

        # Manually adjust some offsets in GZT based on ZDR medians calculated with ZH>30, RHOHV>0.99, SNRH>20, TEMP<0
        # and taking a reference value for snow of 0.2 dB
        # ds.ZDR_EC.where(ds.DBZH>0).where(ds.DBZH<30).where(ds.RHOHV>0.99).where(ds.SNRH>20).where(ds.TEMP<0).compute().median()
        # ds.ZDR_EC.where(ds.DBZH>0).where(ds.DBZH<30).where(ds.RHOHV>0.99).where(ds.SNRH>20).where(ds.TEMP<0).compute().plot.hist(bins=30)
        # ds.ZDR_EC.where(ds.DBZH>0).where(ds.DBZH<30).where(ds.RHOHV>0.99).where(ds.SNRH>20).where(ds.TEMP<0).compute().median(("azimuth", "range")).plot(); ax2 = plt.twinx(); ds.ZDR_EC.where(ds.DBZH>0).where(ds.DBZH<30).where(ds.RHOHV>0.99).where(ds.SNRH>20).where(ds.TEMP<0).compute().count(("azimuth", "range")).plot(ax=ax2, c="orange")

        dates_to_update = { # offsets to 0.2 dB reference value
        '2016-02-06': -0.4,
        '2016-12-14': -0.32,
        '2016-12-16': -0.26,
        '2016-12-20': -0.35,
        '2016-12-21': -0.36, # New value upon revision, old value: -0.26,
        '2016-12-22': -0.32,
        '2016-12-25': -0.2,
        '2016-12-26': -0.26,
        '2016-12-27': -0.26,
        '2016-12-29': -0.5, # New value upon revision, old value: -0.32,
        '2016-12-30': -0.38,
        '2016-12-31': -0.26,
        '2017-01-01': -0.38,
        '2017-01-02': -0.32,
        '2017-01-03': -0.32,
        '2017-12-24': -0.2,
        '2019-12-28': 0.2, # New value upon revision, old value: 0.05
        '2019-12-31': -0.14,
        '2020-01-02': -0.01,
        '2020-01-03': -0.11,
        '2020-01-07': -0.11,
        '2020-01-16': -0.08,
        '2020-01-17': 0.08-0.1,
        '2020-01-19': -0.02,
        '2020-01-20': -0.17,
        '2020-01-31': -0.05,
        '2020-02-07': 0.05,
        '2020-02-08': -0.08,
        '2020-02-29': -0.08,
        '2020-03-18': 0.1,
        '2020-03-19': -0.14,
        '2020-03-20': -0.08,
        }

        ds2_zdr_offsets_comb_alt = ds2_zdr_offsets_comb.copy(deep=True)
        ds2_zdr_offsets_comb_alt.loc[list(dates_to_update.keys())] = list(dates_to_update.values())
        ds2_zdr_offsets_comb_alt_smooth = ds2_zdr_offsets_comb_alt.compute().interpolate_na("time").rolling({"time":5}, center=True, min_periods=1).median()

    if "riming" in vv_to_extract:
        ## Process the riming classification
        print("... Loading pre-calculated riming ...")
        riming_classif = {}
        X_DBZH = "DBZH_AC"
        X_ZDR = "ZDR_EC_OC_WRC_AC"
        suffix_name = "_calibration_attenuation_HTYGZT"

        for stratname in ["unfiltered"]:
            if stratname not in riming_classif.keys():
                riming_classif[stratname] = {}
            elif type(riming_classif[stratname]) is not dict:
                riming_classif[stratname] = {}
            print("Loading "+stratname+" riming classification ...")
            for ll in ["hty"]: # ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
                if ll not in riming_classif[stratname].keys():
                    riming_classif[stratname][ll] = xr.Dataset()
                elif type(riming_classif[stratname][ll]) is not xr.Dataset:
                    riming_classif[stratname][ll] = xr.Dataset()

                for xx in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_'+X_ZDR+'_'+X_DBZH,
                           ]:
                    try:
                        riming_classif[stratname][ll] = riming_classif[stratname][ll].assign( xr.open_dataset(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/"+stratname+"/"+ll+"_"+xx+".nc") )
                        print(ll+" "+xx+" riming_classif loaded")
                    except:
                        if stratname == "unfiltered":
                            pass

                # delete entry if empty
                if not riming_classif[stratname][ll]:
                    del riming_classif[stratname][ll]


        selected_ML_low_riming = {}

        for date in ML_low_dates:
            # 1. Clean the data
            riming_date = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time=date).copy().dropna("z", how="all").fillna(0)
            cleaned_mask = binary_opening(riming_date.values, structure=np.ones((3, 3))).astype(float)

            # Put the cleaned numpy array back into a DataArray so we can use xarray's spatial logic
            da_clean = riming_date.copy(data=cleaned_mask)

            # 2. Define the Melting Layer boundaries
            ml_top = da_clean.height_ml_new_gia_clean
            roi_top = ml_top + 300

            # 3. Create a boolean mask for the 300m region of interest (ROI) above the ML
            roi_mask = (da_clean.z >= ml_top) & (da_clean.z <= roi_top)

            # 4. Calculate the fraction of rimed pixels (1s) within that specific 300m layer per time step
            # Sum of 1s in the ROI
            riming_in_roi = (da_clean == 1).where(roi_mask, 0).sum(dim="z")
            # Total number of pixels in the ROI
            total_in_roi = roi_mask.sum(dim="z")

            # Calculate fraction (using .where to avoid division by zero if a profile has no valid z pixels in that range)
            riming_fraction = riming_in_roi / total_in_roi.where(total_in_roi > 0)

            # 5. Identify the profiles (times) that meet >= 80% condition
            valid_profiles = riming_fraction >= 0.8

            # 6. assign the final values
            selected_ML_low_riming[date] = valid_profiles.copy()

for date in ML_low_dates:
    print("Processing "+date)
    HTY_files0 = [ff for ff in HTY_files if date in ff]
    GZT_files0 = [ff for ff in GZT_files if date in ff]

    selected_ML_low_dates[date] = []

    for HTY_file in HTY_files0:
        for GZT_file in GZT_files0:

            # Create save folder
            sf = savefolder+"ML_low_dates/"
            if not os.path.exists(sf):
                os.makedirs(sf)

            if reload:
                dbzh_loaded = False
                for vi in vv_to_extract:
                    sfp_tg = sf+"_".join([vi, "tg", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                    sfp_ref = sf+"_".join([vi, "ref", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                    try:
                        # if ML heigh from QVP, check if it good according to list of dates
                        if vi == "height_ml_bottom_new_gia_fromqvp" and date in remove_ml_dates:
                            selected_ML_low[vi].append( (np.load(sfp_tg+".npy")*remove_ml_dates[date][0], np.load(sfp_ref+".npy")*remove_ml_dates[date][1] ) )
                        else:
                            selected_ML_low[vi].append( (np.load(sfp_tg+".npy"), np.load(sfp_ref+".npy") ) )
                        if vi == "DBZH":
                            dbzh_loaded = True
                            selected_ML_low_dates[date].append( ( "HTY "+HTY_file.split("/")[-2],
                                                                  "GZT "+GZT_file.split("/")[-2],
                                                                  np.isfinite(selected_ML_low[vi][-1][0]).sum()) )
                    except:
                        print(vi+": reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy \n failed")
                if dbzh_loaded:
                    continue
                if not calc:
                    print("Total fail reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy")
                    continue
                print("Total fail reloading \n "+sfp_tg+".npy \n or \n "+sfp_ref+".npy \n attempting to calculate")

            # Load the data
            ds1 = xr.open_mfdataset(HTY_file).set_coords("TEMP") # make sure that TEMP is a coord
            ds2 = xr.open_mfdataset(GZT_file).set_coords("TEMP")

            # Get PPIs into the same reference system
            proj = utils.get_common_projection(ds1, ds2)

            ds1 = wrl.georef.georeference(ds1, crs=proj)
            ds2 = wrl.georef.georeference(ds2, crs=proj)

            # Add new offset/atten corrected ZDR in datasets
            if "ZDR_EC_OC2" in vv_to_extract: # correct ds2 ZDR with extrapolated offsets
                ds1 = ds1.assign({"ZDR_EC_OC2":
                                  ds1["ZDR_EC"] - ds1_zdr_offsets_comb.sel(time=ds1.time[0].values.astype(str)[:10]).mean()} )

                ds2 = ds2.assign({"ZDR_EC_OC2":
                                  ds2["ZDR_EC"] - ds2_zdr_offsets_comb_smooth.sel(time=ds2.time[0].values.astype(str)[:10]).mean()} )

            if "ZDR_EC_OC3" in vv_to_extract: # correct ds2 ZDR with extrapolated offsets, some dates have manual offsets
                ds2 = ds2.assign({"ZDR_EC_OC3":
                                  ds2["ZDR_EC"] - ds2_zdr_offsets_comb_alt_smooth.sel(time=ds2.time[0].values.astype(str)[:10]).mean()} )

            vv_AC2_rain = [vv for vv in vv_to_extract if "2_rain" in vv and "_ML" not in vv]
            vv_noAC2_rain = [vv.split("_AC2_rain")[0] for vv in vv_to_extract if "_AC2_rain" in vv and "_ML" not in vv]

            if len(vv_noAC2_rain)>0:
                ds1_AC2_rain = utils.attenuation_corr_linear(ds1[["PHIDP_OC_MASKED", "PHIDP_OC"]+[vv for vv in vv_noAC2_rain if vv in ds1.data_vars]].compute(),
                                                    alpha = new_alpha, beta = new_beta, alphaml = 0, betaml = 0,
                                                   dbzh=[vv for vv in vv_noAC2_rain if "DBZH" in vv],
                                                   zdr=[vv for vv in vv_noAC2_rain if "ZDR" in vv],
                                                   phidp=["PHIDP_OC_MASKED", "PHIDP_OC"],
                                                   ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                                   temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                                   interpolate_deltabump = True )
                vars_rename = {vv: vv+"2_rain" for vv in [vv+"_AC" for vv in vv_noAC2_rain if vv in ds1.data_vars]}
                ds1 = ds1.assign(ds1_AC2_rain.rename(vars_rename)[list(vars_rename.values())])

                ds2_AC2_rain = utils.attenuation_corr_linear(ds2[["PHIDP_OC_MASKED", "PHIDP_OC"]+[vv for vv in vv_noAC2_rain if vv in ds2.data_vars]].compute(),
                                                    alpha = new_alpha, beta = new_beta, alphaml = 0, betaml = 0,
                                                   dbzh=[vv for vv in vv_noAC2_rain if "DBZH" in vv],
                                                   zdr=[vv for vv in vv_noAC2_rain if "ZDR" in vv],
                                                   phidp=["PHIDP_OC_MASKED", "PHIDP_OC"],
                                                   ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                                   temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                                   interpolate_deltabump = True )
                vars_rename = {vv: vv+"2_rain" for vv in [vv+"_AC" for vv in vv_noAC2_rain if vv in ds2.data_vars]}
                ds2 = ds2.assign(ds2_AC2_rain.rename(vars_rename)[list(vars_rename.values())])

            # add ML bump/min/max variables
            vv_bump = [vv for vv in vv_to_extract if "_MLbump" in vv and "_safer" not in vv and "_strict" not in vv]
            vv_nobump = [vv.split("_MLbump")[0] for vv in vv_to_extract if "MLbump" in vv and "_safer" not in vv and "_strict" not in vv]

            vv_bump_safer = [vv for vv in vv_to_extract if "_MLbump" in vv and "_safer" in vv]
            vv_nobump_safer = [vv.split("_MLbump")[0] for vv in vv_to_extract if "MLbump" in vv and "_safer" in vv]

            vv_bump_strict = [vv for vv in vv_to_extract if "_MLbump" in vv and "_strict" in vv]
            vv_nobump_strict = [vv.split("_MLbump")[0] for vv in vv_to_extract if "MLbump" in vv and "_strict" in vv]

            vv_min = [vv for vv in vv_to_extract if "_MLmin" in vv]
            vv_nomin = [vv.split("_MLmin")[0] for vv in vv_to_extract if "MLmin" in vv]

            vv_max = [vv for vv in vv_to_extract if "_MLmax" in vv]
            vv_nomax = [vv.split("_MLmax")[0] for vv in vv_to_extract if "MLmax" in vv]

            if len(vv_bump) > 0:

                if "range" in vv_nobump:
                    vv_nobump = ["range_" if x=="range" else x for x in vv_nobump]
                    ds1 = ds1.assign({"range_": ds1["range"].broadcast_like(ds1.DBZH)})
                    ds2 = ds2.assign({"range_": ds2["range"].broadcast_like(ds2.DBZH)})

                # for ds1
                below_ml = ds1[vv_nobump].where(ds1.z < ds1.height_ml_bottom_new_gia).where(ds1.z > ds1.height_ml_bottom_new_gia - 100)
                above_ml = ds1[vv_nobump].where(ds1.z > ds1.height_ml_new_gia).where(ds1.z < ds1.height_ml_new_gia + 100)
                below_ml_TEMP = ds1[vv_nobump].where(ds1.TEMP>3).where(ds1.TEMP<3.5).where(~ds1.height_ml_bottom_new_gia.notnull())
                above_ml_TEMP = ds1[vv_nobump].where(ds1.TEMP<-1).where(ds1.TEMP>-1-0.5).where(~ds1.height_ml_new_gia.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds1 = ds1.assign( xr.where(ds1.height_ml_bottom_new_gia.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump, vv_bump))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump, vv_bump))) ) )

                # for ds2 (by definition this will be NaN for the cases we will select, but we still need the variable for completion)
                below_ml = ds2[vv_nobump].where(ds2.z < ds2.height_ml_bottom_new_gia).where(ds2.z > ds2.height_ml_bottom_new_gia - 100)
                above_ml = ds2[vv_nobump].where(ds2.z > ds2.height_ml_new_gia).where(ds2.z < ds2.height_ml_new_gia + 100)
                below_ml_TEMP = ds2[vv_nobump].where(ds2.TEMP>3).where(ds2.TEMP<3.5).where(~ds2.height_ml_bottom_new_gia.notnull())
                above_ml_TEMP = ds2[vv_nobump].where(ds2.TEMP<-1).where(ds2.TEMP>-1-0.5).where(~ds2.height_ml_new_gia.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds2 = ds2.assign( xr.where(ds2.height_ml_bottom_new_gia.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump, vv_bump))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump, vv_bump))) ) )

            if "z_beambot" in vv_to_extract or len(vv_bump_safer) > 0:
                # we just copy the original coordinates and subtract half beamwidth, then georeference again
                ds1_beambot = ds1["DBZH"].copy()
                ds1_beambot['elevation'] = ds1_beambot['elevation'] - 0.5
                ds1_beambot = wrl.georef.georeference(ds1_beambot, crs=proj)
                ds1.coords["z_beambot"] = ds1_beambot["z"].broadcast_like(ds1["DBZH"]).reset_coords(drop=True)

                ds2_beambot = ds2["DBZH"].copy()
                ds2_beambot['elevation'] = ds2_beambot['elevation'] - 0.5
                ds2_beambot = wrl.georef.georeference(ds2_beambot, crs=proj)
                ds2.coords["z_beambot"] = ds2_beambot["z"].broadcast_like(ds2["DBZH"]).reset_coords(drop=True)

            if "z_beamtop" in vv_to_extract or len(vv_bump_safer) > 0:
                # we just copy the original coordinates and add half beamwidth, then georeference again
                ds1_beamtop = ds1["DBZH"].copy()
                ds1_beamtop['elevation'] = ds1_beamtop['elevation'] + 0.5
                ds1_beamtop = wrl.georef.georeference(ds1_beamtop, crs=proj)
                ds1.coords["z_beamtop"] = ds1_beamtop["z"].broadcast_like(ds1["DBZH"]).reset_coords(drop=True)

                ds2_beamtop = ds2["DBZH"].copy()
                ds2_beamtop['elevation'] = ds2_beamtop['elevation'] + 0.5
                ds2_beamtop = wrl.georef.georeference(ds2_beamtop, crs=proj)
                ds2.coords["z_beamtop"] = ds2_beamtop["z"].broadcast_like(ds2["DBZH"]).reset_coords(drop=True)

            # add ML height from QVP
            if "height_ml_new_gia_fromqvp" in vv_to_extract or len(vv_bump_safer) > 0 or len(vv_bump_strict) > 0:
                for qvp_elev in elev_ml_top_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(HTY_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            ds1.coords["height_ml_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_new_gia"].interp_like(ds1.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_new_gia_fromqvp" not in ds1.coords: # if that did not work, just fill with NaNs
                    ds1.coords["height_ml_new_gia_fromqvp"] = ds1.coords["height_ml_new_gia"]*np.nan

                for qvp_elev in elev_ml_top_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(GZT_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            ds2.coords["height_ml_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_new_gia"].interp_like(ds2.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_new_gia_fromqvp" not in ds2.coords: # if that did not work, just fill with NaNs
                    ds2.coords["height_ml_new_gia_fromqvp"] = ds2.coords["height_ml_new_gia"]*np.nan

            if "height_ml_bottom_new_gia_fromqvp" in vv_to_extract or len(vv_bump_safer) > 0 or len(vv_bump_strict) > 0:
                for qvp_elev in elev_ml_top_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(HTY_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            ds1.coords["height_ml_bottom_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_bottom_new_gia"].interp_like(ds1.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_bottom_new_gia_fromqvp" not in ds1.coords: # if that did not work, just fill with NaNs
                    ds1.coords["height_ml_bottom_new_gia_fromqvp"] = ds1.coords["height_ml_bottom_new_gia"]*np.nan

                for qvp_elev in elev_ml_top_fromqvp:
                    try:
                        qvp_glob = glob.glob("/".join(GZT_file.replace("final_ppis","qvps").split("/")[:-3])+"/*/"+qvp_elev+"/*.nc")
                        if len(qvp_glob)>0:
                            qvp_for_ml = xr.open_dataset(qvp_glob[0])
                            ds2.coords["height_ml_bottom_new_gia_fromqvp"] = qvp_for_ml.sel(time=date)["height_ml_bottom_new_gia"].interp_like(ds2.time, method="nearest")
                            break
                    except:
                        continue
                if "height_ml_bottom_new_gia_fromqvp" not in ds2.coords: # if that did not work, just fill with NaNs
                    ds2.coords["height_ml_bottom_new_gia_fromqvp"] = ds2.coords["height_ml_bottom_new_gia"]*np.nan

            if len(vv_bump_safer) > 0:

                if "range" in vv_nobump_safer:
                    vv_nobump_safer = ["range_" if x=="range" else x for x in vv_nobump_safer]
                    ds1 = ds1.assign({"range_": ds1["range"].broadcast_like(ds1.DBZH)})
                    ds2 = ds2.assign({"range_": ds2["range"].broadcast_like(ds2.DBZH)})

                # for ds1
                below_ml = ds1[vv_nobump_safer].where(ds1.z_beamtop < ds1.height_ml_bottom_new_gia_fromqvp).where(ds1.z_beamtop > ds1.height_ml_bottom_new_gia_fromqvp - 100)
                above_ml = ds1[vv_nobump_safer].where(ds1.z_beambot > ds1.height_ml_new_gia_fromqvp).where(ds1.z_beambot < ds1.height_ml_new_gia_fromqvp + 100)
                below_ml_TEMP = ds1[vv_nobump_safer].where(ds1.TEMP>3).where(ds1.TEMP<3.5).where(~ds1.height_ml_bottom_new_gia_fromqvp.notnull())
                above_ml_TEMP = ds1[vv_nobump_safer].where(ds1.TEMP<-1).where(ds1.TEMP>-1-0.5).where(~ds1.height_ml_new_gia_fromqvp.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds1 = ds1.assign( xr.where(ds1.height_ml_bottom_new_gia_fromqvp.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump_safer, vv_bump_safer))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump_safer, vv_bump_safer))) ) )

                # for ds2 (by definition this will be NaN for the cases we will select, but we still need the variable for completion)
                below_ml = ds2[vv_nobump_safer].where(ds2.z_beamtop < ds2.height_ml_bottom_new_gia_fromqvp).where(ds2.z_beamtop > ds2.height_ml_bottom_new_gia_fromqvp - 100)
                above_ml = ds2[vv_nobump_safer].where(ds2.z_beambot > ds2.height_ml_new_gia_fromqvp).where(ds2.z_beambot < ds2.height_ml_new_gia_fromqvp + 100)
                below_ml_TEMP = ds2[vv_nobump_safer].where(ds2.TEMP>3).where(ds2.TEMP<3.5).where(~ds2.height_ml_bottom_new_gia_fromqvp.notnull())
                above_ml_TEMP = ds2[vv_nobump_safer].where(ds2.TEMP<-1).where(ds2.TEMP>-1-0.5).where(~ds2.height_ml_new_gia_fromqvp.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds2 = ds2.assign( xr.where(ds2.height_ml_bottom_new_gia_fromqvp.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump_safer, vv_bump_safer))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump_safer, vv_bump_safer))) ) )

            if len(vv_bump_strict) > 0:

                if "range" in vv_nobump_strict:
                    vv_nobump_strict = ["range_" if x=="range" else x for x in vv_nobump_strict]
                    ds1 = ds1.assign({"range_": ds1["range"].broadcast_like(ds1.DBZH)})
                    ds2 = ds2.assign({"range_": ds2["range"].broadcast_like(ds2.DBZH)})

                # for ds1
                below_ml = ds1[vv_nobump_strict].where(ds1.z < ds1.height_ml_bottom_new_gia_fromqvp).where(ds1.z > ds1.height_ml_bottom_new_gia_fromqvp - 100)
                above_ml = ds1[vv_nobump_strict].where(ds1.z > ds1.height_ml_new_gia_fromqvp).where(ds1.z < ds1.height_ml_new_gia_fromqvp + 100)
                below_ml_TEMP = ds1[vv_nobump_strict].where(ds1.TEMP>3).where(ds1.TEMP<3.5).where(~ds1.height_ml_bottom_new_gia_fromqvp.notnull())
                above_ml_TEMP = ds1[vv_nobump_strict].where(ds1.TEMP<-1).where(ds1.TEMP>-1-0.5).where(~ds1.height_ml_new_gia_fromqvp.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds1 = ds1.assign( xr.where(ds1.height_ml_bottom_new_gia_fromqvp.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump_strict, vv_bump_strict))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump_strict, vv_bump_strict))) ) )

                # for ds2 (by definition this will be NaN for the cases we will select, but we still need the variable for completion)
                below_ml = ds2[vv_nobump_strict].where(ds2.z < ds2.height_ml_bottom_new_gia_fromqvp).where(ds2.z > ds2.height_ml_bottom_new_gia_fromqvp - 100)
                above_ml = ds2[vv_nobump_strict].where(ds2.z > ds2.height_ml_new_gia_fromqvp).where(ds2.z < ds2.height_ml_new_gia_fromqvp + 100)
                below_ml_TEMP = ds2[vv_nobump_strict].where(ds2.TEMP>3).where(ds2.TEMP<3.5).where(~ds2.height_ml_bottom_new_gia_fromqvp.notnull())
                above_ml_TEMP = ds2[vv_nobump_strict].where(ds2.TEMP<-1).where(ds2.TEMP>-1-0.5).where(~ds2.height_ml_new_gia_fromqvp.notnull())

                bump_ml = above_ml.bfill("range").head(range=1).isel(range=0) - below_ml.ffill("range").tail(range=1).isel(range=0)
                bump_ml_TEMP = above_ml_TEMP.bfill("range").head(range=1).isel(range=0) - below_ml_TEMP.ffill("range").tail(range=1).isel(range=0)

                ds2 = ds2.assign( xr.where(ds2.height_ml_bottom_new_gia_fromqvp.notnull(),
                                           bump_ml.rename(dict(zip(vv_nobump_strict, vv_bump_strict))),
                                           bump_ml_TEMP.rename(dict(zip(vv_nobump_strict, vv_bump_strict))) ) )

            if len(vv_nomin) > 0 or len(vv_nomax) > 0:
                # for ds1
                in_ml = ds1[vv_nomin+vv_nomax].where(ds1.z >= ds1.height_ml_bottom_new_gia).where(ds1.z <= ds1.height_ml_new_gia)
                in_ml_TEMP = ds1[vv_nomin+vv_nomax].where(ds1.TEMP<3).where(ds1.TEMP>-1).where(~ds1.height_ml_bottom_new_gia.notnull())

                if len(vv_nomin) > 0:
                    min_ml = in_ml[vv_nomin].min("range")
                    min_ml_TEMP = in_ml_TEMP[vv_nomin].min("range")
                    ds1 = ds1.assign( xr.where(ds1.height_ml_bottom_new_gia.notnull(),
                                               min_ml.rename(dict(zip(vv_nomin, vv_min))),
                                               min_ml_TEMP.rename(dict(zip(vv_nomin, vv_min))) ) )

                if len(vv_nomax) > 0:
                    max_ml = in_ml[vv_nomax].max("range")
                    max_ml_TEMP = in_ml_TEMP[vv_nomax].max("range")
                    ds1 = ds1.assign( xr.where(ds1.height_ml_bottom_new_gia.notnull(),
                                               max_ml.rename(dict(zip(vv_nomax, vv_max))),
                                               max_ml_TEMP.rename(dict(zip(vv_nomax, vv_max))) ) )


                # for ds2 (by definition this will be NaN for the cases we will select, but we still need the variable for completion)
                in_ml = ds2[vv_nomin+vv_nomax].where(ds2.z >= ds2.height_ml_bottom_new_gia).where(ds2.z <= ds2.height_ml_new_gia)
                in_ml_TEMP = ds2[vv_nomin+vv_nomax].where(ds2.TEMP<3).where(ds2.TEMP>-1).where(~ds2.height_ml_bottom_new_gia.notnull())

                if len(vv_nomin) > 0:
                    min_ml = in_ml[vv_nomin].min("range")
                    min_ml_TEMP = in_ml_TEMP[vv_nomin].min("range")
                    ds2 = ds2.assign( xr.where(ds2.height_ml_bottom_new_gia.notnull(),
                                               min_ml.rename(dict(zip(vv_nomin, vv_min))),
                                               min_ml_TEMP.rename(dict(zip(vv_nomin, vv_min))) ) )

                if len(vv_nomax) > 0:
                    max_ml = in_ml[vv_nomax].max("range")
                    max_ml_TEMP = in_ml_TEMP[vv_nomax].max("range")
                    ds2 = ds2.assign( xr.where(ds2.height_ml_bottom_new_gia.notnull(),
                                               max_ml.rename(dict(zip(vv_nomax, vv_max))),
                                               max_ml_TEMP.rename(dict(zip(vv_nomax, vv_max))) ) )

            # Add beam blockage
            ds1_pbb, ds1_cbb = beam_blockage_from_radar_ds(ds1.isel(time=0),
                                                           (ds1.longitude, ds1.latitude, ds1.altitude),
                                                           wradlib_token = token)

            ds1 = ds1.assign({"PBB": ds1_pbb, "CBB": ds1_cbb})

            ds2_pbb, ds2_cbb = beam_blockage_from_radar_ds(ds2.isel(time=0),
                                                           (ds2.longitude, ds2.latitude, ds2.altitude),
                                                           wradlib_token = token)

            ds2 = ds2.assign({"PBB": ds2_pbb, "CBB": ds2_cbb})

            # Apply thresholds before computing masks
            dsx = utils.apply_min_max_thresh(ds1, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                                    "SNRHC":SNRH_min, "SQIH":0.5},
                                                 {"CBB": CBB_max})
            dsy = utils.apply_min_max_thresh(ds2, {"RHOHV":RHOHV_min, "SNRH":SNRH_min,
                                                    "SNRHC":SNRH_min, "SQIH":0.5},
                                                 {"CBB": CBB_max})

            # They consider that there is rain above the radar by looking at the
            # median reflectivity in a circle aroud each radar. Let's add this variable
            if "Zm" not in dsx.coords:
                dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")) })
            if "Zm" not in dsy.coords:
                dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")) })

            # Analogously, add the TEMP close to the radar as a measure of below/above ML
            dsx = dsx.assign_coords({"TEMPm": dsx["TEMP"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")) })
            dsy = dsy.assign_coords({"TEMPm": dsy["TEMP"].sel(range=slice(0,Zm_range)).compute().median(("azimuth", "range")) })

            if "binvol" in vv_to_extract:
                dsx.coords["binvol"] = dsx.range.wrl.qual.pulse_volume(dsx.range.diff("range").median(), 1)
                dsy.coords["binvol"] = dsy.range.wrl.qual.pulse_volume(dsy.range.diff("range").median(), 1)

            if "beam_cross_angle" in vv_to_extract:
                dsx.coords["beam_cross_angle"] = utils.compute_crossing_angle_cartesian(dsx, (dsx.x[:,0].mean(), dsx.y[:,0].mean(), dsx.z[:,0].mean()), (dsy.x[:,0].mean(), dsy.y[:,0].mean(), dsy.z[:,0].mean()))
                dsy.coords["beam_cross_angle"] = utils.compute_crossing_angle_cartesian(dsy, (dsx.x[:,0].mean(), dsx.y[:,0].mean(), dsx.z[:,0].mean()), (dsy.x[:,0].mean(), dsy.y[:,0].mean(), dsy.z[:,0].mean()))

            if "riming" in vv_to_extract:
                dsx.coords["riming"] = (selected_ML_low_riming[date].astype(int).interp_like(dsx.time)>0.5).astype(int)

            # Add the additional DBZH and TEMP thresholds
            # (apply the TEMP threshold manually since it is not a variable but a coord)
            dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                                 {}).where(dsx["TEMP"] < TEMP_max)
            dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                                 {}).where(dsy["TEMP"] < TEMP_max)

            # One radar has to be the reference and the other must be the target, both below the ML
            # Let's take GZT as reference

            # We will not apply additional Zm or PHIDP conditions now so we can use
            # the extracted values for different comaparisons
            dsx_tg = dsx[[vv for vv in vv_to_extract if vv in dsx]].compute() # if we pre compute the variables that we want
            dsy_rf = dsy[[vv for vv in vv_to_extract if vv in dsy]].compute() # we save a lot of time (~3 times faster)

            if NN:
                mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx_tg,
                                                                                                               dsy_rf,
                                                                                    tolerance=tolerance,
                                                                                    tolerance_time=60*4)

                mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref = utils.refine_radar_overlap_unique_NN_pairs(
                                                                                        dsx_tg, dsy_rf,
                                                                                        idx_tg, idx_rf,
                                                                                        matched_timesteps,
                                                                                        vv,
                                                                                    tolerance_time=60*4,
                                                                                    z_tolerance=100.)
            else:
                mask_tg, mask_rf, idx_tg, idx_rf, matched_timesteps = utils.find_radar_overlap(dsx_tg,
                                                                                               dsy_rf,
                                                                                    tolerance=tolerance,
                                                                                    tolerance_time=60*4)

                mask_tg_ref, mask_rf_ref, idx_tg_ref, idx_rf_ref = utils.refine_radar_overlap(
                                                                                        dsx_tg, dsy_rf,
                                                                                        mask_tg, mask_rf,
                                                                                        matched_timesteps,
                                                                                        vv,
                                                                                        tolerance=tolerance,
                                                                                    tolerance_time=60*4,
                                                                                    z_tolerance=100.)

            if mask_tg_ref.sum() == 0:
                print("No matches found")
                continue # jump to next iteration if no pairs are found

            for vi in vv_to_extract:
                if vi not in dsx_tg:
                    print(vi+" not found in target ds, filling with NaNs")
                    dsx_tg = dsx_tg.assign( { vi: xr.full_like(dsx_tg["DBZH"], fill_value=np.nan) } )
                if vi not in dsy_rf:
                    print(vi+" not found in reference ds, filling with NaNs")
                    dsy_rf = dsy_rf.assign( { vi: xr.full_like(dsy_rf["DBZH"], fill_value=np.nan) } )

                if NN:
                    dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf,
                                                                            mask_tg_ref, mask_rf_ref,
                                                               idx_tg_ref, idx_rf_ref,
                                                               matched_timesteps, vi)
                else:
                    dsx_p_tg, dsy_p_rf = utils.return_value_pairs(dsx_tg, dsy_rf,
                                                                  idx_tg_ref, idx_rf_ref,
                                                               matched_timesteps, vi)

                selected_ML_low[vi].append( (dsx_p_tg.copy(), dsy_p_rf.copy()) )

                sfp_tg = sf+"_".join([vi, "tg", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                sfp_ref = sf+"_".join([vi, "ref", os.path.basename(HTY_file), os.path.basename(GZT_file)])
                np.save(sfp_tg,
                    selected_ML_low[vi][-1][0], allow_pickle=False)
                np.save(sfp_ref,
                    selected_ML_low[vi][-1][1], allow_pickle=False)

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%%% DEPRECATED Plot boxplot of DBZH/ZDR as vertical profiles (ML attenuation)

phi = "PHIDP_OC_MASKED"
dbzh_tg = "DBZH_AC2_rain" # ZDR_EC_OC_AC2_rain
dbzh_ref = "DBZH_AC2_rain" # ZDR_EC_OC3_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

# We only need to check that TEMPm is appropriate for each radar

# extract/build necessary variables
tg_dbzh = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg] ])

ref_dbzh = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[dbzh_ref] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_phi_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

ref_phi_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

tg_z_beambot = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

ref_z_beambot = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

tg_height_ml_top_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

ref_height_ml_top_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# We should only use height_ml_top_qvp values from tg since ref should be above the ML
# fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (np.nan_to_num(ref_phi_bump) < 1)  & (ref_phi < 5) & (ref_Zm < 5) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)\
        & np.isfinite(ref_dbzh) & (tg_height_ml_top_qvp < 1600)\
        & (tg_phi_bump > varx_range[0]) & (tg_phi_bump < varx_range[1] - varx_range[2])
        # & (tg_z_beambot > tg_height_ml_top_qvp) & (ref_z_beambot > tg_height_ml_top_qvp)\
        # & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\ # loose way of avoiding the ML
        # & (tg_TEMPm > 3) & (ref_TEMPm < 0)\ # loose way of avoiding the ML

tg_dbzh = tg_dbzh[valid]
ref_dbzh = ref_dbzh[valid]
tg_TEMP = tg_TEMP[valid]
ref_TEMP = ref_TEMP[valid]

# Box plots like in the paper
# Define bins
bins = np.arange(-16, 0, 1)
bin_centers = (bins[:-1] + bins[1:])/2

# Digitize into bins
bin_indices_tg = np.digitize(tg_TEMP, bins) - 1
bin_indices_ref = np.digitize(ref_TEMP, bins) - 1

# Prepare data for boxplot
box_data_tg = [tg_dbzh[bin_indices_tg == i] for i in range(len(bins) - 1)]
box_data_ref = [ref_dbzh[bin_indices_ref == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data_tg]

# Plot #!!! This plot still needs to be beatyfied like in the paper
plt.figure(figsize=(7, 9))
bptg = plt.boxplot(box_data_tg, positions=bin_centers-0.15, widths=0.4, showmeans=True, vert=False, label="target")
bpref = plt.boxplot(box_data_ref, positions=bin_centers+0.15, widths=0.4, showmeans=True, vert=False, label="reference")
for box in bpref['boxes']:
    # change outline color
    box.set(color='red')
plt.gca().yaxis.set_inverted(True)
plt.xlabel(dbzh_tg)
plt.ylabel(TEMP+" (binned, 1° intervals)")
plt.title("Boxplots of "+dbzh_tg+" vs "+TEMP)
plt.grid(True, linestyle="--", alpha=0.5)
plt.yticks(bin_centers, [f"{b}-{b+1}" for b in bins[:-1]])

custom_lines = [mpl.lines.Line2D([0], [0], color="black", lw=1),
                mpl.lines.Line2D([0], [0], color="red", lw=1)]
plt.legend(custom_lines, ['target', 'reference'])

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(plt.xlim()[0] + 0.01 * (plt.xlim()[1] - plt.xlim()[0]), x, # 5% above bottom
             f"{n}", ha='center', va='center', fontsize=9, color='dimgray', rotation=90)

plt.tight_layout()
plt.show()

#%%% Plot boxplot of delta DBZH/ZDR vs target PHI bump (ML attenuation)

phi = "PHIDP_OC_MASKED"
dbzh_tg = "ZDR_EC_OC_AC2_rain_WRcorr" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain
dbzh_ref = "ZDR_EC_OC3" # ZDR_EC_OC3_AC2_rain, DBZH_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

yax = r"$Δ\mathrm{Z_{DR}}\ [dB]$" # label for the y axis
xax = r"$Δ\mathrm{\Phi_{DP}^{ML}}\ [°]$" # label for the x axis

varx_range = (0, 19, 1) # start, stop, step # (0.7, 0.98, 0.02)

min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -3 # min and max limits for the y axis
ymax = 2.5

# WR corr based on results
def zdr_wrc(Zm):
    Zm_ = np.where(np.nan_to_num(Zm) < 32.5,
                   np.nan_to_num(Zm),
                   32.5)
    return -0.00022*Zm_ + 0.00032*Zm_**2 # change here to adjust coefficients based on results

if "_WRcorr" in dbzh_tg:
    # Correct wet-radome timesteps
    var_tg_ = "".join(dbzh_tg.split("_WRcorr"))
    selected_ML_low[dbzh_tg] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_tg].append((selected_ML_low[var_tg_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_tg_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

if "_WRcorr" in dbzh_ref:
    # Correct wet-radome timesteps
    var_ref_ = "".join(dbzh_ref.split("_WRcorr"))
    selected_ML_low[dbzh_ref] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_ref].append((selected_ML_low[var_ref_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_ref_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

# extract/build necessary variables
tg_dbzh = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg] ])

ref_dbzh = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[dbzh_ref] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_phi_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ]) # _MLbump_safer

ref_phi_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ]) # _MLbump_safer

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

tg_z_beambot = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

ref_z_beambot = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

# tg_height_ml_top_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# ref_height_ml_top_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# # We should only use height_ml_top_qvp values from tg since ref should be above the ML
# # fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
# tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
# ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_top_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

ref_height_ml_top_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_top_qvp)):
    # fill the NaN height_ml_top_qvp values from ref with tg
    ref_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])] = tg_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_top_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_top_qvp[ts][:,0])
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_top_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_top_qvp[ts][:,0])
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_top_qvp[ts] = pd.DataFrame(tg_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_top_qvp[ts] = pd.DataFrame(ref_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_top_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_top_qvp])
ref_height_ml_top_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_top_qvp])

# fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (np.nan_to_num(ref_phi_bump) < 1)  & (ref_phi < 5) & (ref_Zm < 5) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)\
        & (tg_height_ml_top_qvp < 1600)\
        & (tg_phi_bump > varx_range[0]) & (tg_phi_bump < varx_range[1] - varx_range[2])\
        & (tg_z_beambot > tg_height_ml_top_qvp) & (ref_z_beambot > tg_height_ml_top_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMPm > 3) & (ref_TEMPm < 0)\
        & (tg_bca > 135) & (ref_bca > 135)\


delta_dbzh = (tg_dbzh - ref_dbzh)[valid]
tg_phi_bump = tg_phi_bump[valid]

# In case we need to filter out unrealistic values
# tg_phi_bump = tg_phi_bump[delta_dbzh>-4]
# delta_dbzh = delta_dbzh[delta_dbzh>-4]

# Calculate best linear fit
lfit = np.polynomial.Polynomial.fit(tg_phi_bump, delta_dbzh, 1)
lfit_str = str(lfit.convert()).replace("x", "Phi ML bump")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_phi_bump into bins
bin_indices = np.digitize(tg_phi_bump, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                 medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, bins)
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+dbzh_tg+" vs "+phi+"_MLbump"+" bins")

# # add linear fit
# plt.plot([bins[0], bins[-1]], [lfit(bins[0]), lfit(bins[-1])])
# plt.text(0.95, 0.9, "Linear fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second linear fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 2)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
# plt.plot([bins[0], bins[-1]], [lfit_m(bins[0]), lfit_m(bins[-1])], c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third linear fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 3)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot([bins[0], bins[-1]], [lfit_mw(bins[0]), lfit_mw(bins[-1])], c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# # Print p value and other stats
# scipy.stats.linregress(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)])

# Print p value and other stats (for weighted fit)
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)]))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())

#%%% Plot boxplot of delta DBZH/ZDR vs target range ML bump (ML attenuation, analogous to previous one but for ADP)

phi = "PHIDP_OC_MASKED"
vrange = "range"
dbzh_tg = "ZDR_EC_OC_AC2_rain_WRcorr" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain
dbzh_ref = "ZDR_EC_OC3" # ZDR_EC_OC3_AC2_rain, DBZH_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

yax = r"$Δ\mathrm{Z_{DR}}\ [dB]$" # label for the y axis
xax = r"$Δ\mathrm{range^{ML}}\ [km]$" # label for the x axis

varx_range = (0, 41, 1) # start, stop, step # (0.7, 0.98, 0.02)
phi_range = (0, 19, 1)

min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -3 # min and max limits for the y axis
ymax = 2.5

# WR corr based on results
def zdr_wrc(Zm):
    Zm_ = np.where(np.nan_to_num(Zm) < 32.5,
                   np.nan_to_num(Zm),
                   32.5)
    return -0.00022*Zm_ + 0.00032*Zm_**2 # change here to adjust coefficients based on results

if "_WRcorr" in dbzh_tg:
    # Correct wet-radome timesteps
    var_tg_ = "".join(dbzh_tg.split("_WRcorr"))
    selected_ML_low[dbzh_tg] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_tg].append((selected_ML_low[var_tg_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_tg_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

if "_WRcorr" in dbzh_ref:
    # Correct wet-radome timesteps
    var_ref_ = "".join(dbzh_ref.split("_WRcorr"))
    selected_ML_low[dbzh_ref] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_ref].append((selected_ML_low[var_ref_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_ref_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

# extract/build necessary variables
tg_dbzh = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg] ])

ref_dbzh = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[dbzh_ref] ])

# tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi] ])

# ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_phi_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ]) # _MLbump_safer

ref_phi_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ]) # _MLbump_safer

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

tg_z_beambot = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

ref_z_beambot = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

tg_range_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[vrange+"_MLbump"] ])/1000 # _MLbump_safer

ref_range_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[vrange+"_MLbump"] ])/1000 # _MLbump_safer

# tg_height_ml_top_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# ref_height_ml_top_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# # We should only use height_ml_top_qvp values from tg since ref should be above the ML
# # fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
# tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
# ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_top_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

ref_height_ml_top_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_top_qvp)):
    # fill the NaN height_ml_top_qvp values from ref with tg
    ref_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])] = tg_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_top_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_top_qvp[ts][:,0])
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_top_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_top_qvp[ts][:,0])
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] > ref_m+ref_std] = np.nan

    # # Interpolate and extrapolate to fill NaNs
    # tg_height_ml_top_qvp[ts] = pd.DataFrame(tg_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    # ref_height_ml_top_qvp[ts] = pd.DataFrame(ref_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_top_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_top_qvp])
ref_height_ml_top_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_top_qvp])

# # fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
# tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
# ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (np.nan_to_num(ref_phi_bump) < 1)  & (ref_phi < 5) & (ref_Zm < 5) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)\
        & (tg_height_ml_top_qvp < 1600)\
        & (tg_range_bump > varx_range[0]) & (tg_range_bump < varx_range[1] - varx_range[2])\
        & (tg_phi_bump > phi_range[0]) & (tg_phi_bump < phi_range[1] - phi_range[2])\
        & (tg_z_beambot > tg_height_ml_top_qvp) & (ref_z_beambot > tg_height_ml_top_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMPm > 3) & (ref_TEMPm < 0)\
        & (tg_bca > 135) & (ref_bca > 135)\


delta_dbzh = (tg_dbzh - ref_dbzh)[valid]
tg_range_bump = tg_range_bump[valid]
tg_phi_bump = tg_phi_bump[valid]

# In case we need to filter out unrealistic values
# tg_range_bump = tg_range_bump[delta_dbzh>-4]
# delta_dbzh = delta_dbzh[delta_dbzh>-4]

# Calculate best linear fit
lfit = np.polynomial.Polynomial.fit(tg_range_bump, delta_dbzh, 1)
lfit_str = str(lfit.convert()).replace("x", "range ML bump")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_range_bump into bins
bin_indices = np.digitize(tg_range_bump, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                 medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, np.round(bins).astype("int")) # convert to km
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+dbzh_tg+" vs "+phi+"_MLbump"+" bins")

# # add linear fit
# plt.plot([bins[0], bins[-1]], [lfit(bins[0]), lfit(bins[-1])])
# plt.text(0.95, 0.9, "Linear fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second linear fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 2)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
# plt.plot([bins[0], bins[-1]], [lfit_m(bins[0]), lfit_m(bins[-1])], c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third linear fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 3)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot([bins[0], bins[-1]], [lfit_mw(bins[0]), lfit_mw(bins[-1])], c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# # Print p value and other stats
# scipy.stats.linregress(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)])

# Print p value and other stats (for weighted fit)
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)]))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())

#%%% Plot histogram of delta DBZH/ZDR with and without correction (also possible for combined plot with selected_ML_high events)

phi = "PHIDP_OC_MASKED"
dbzh_tg = "ZDR_EC_OC_AC2_rain_WRcorr" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain
dbzh_tg_uncorr = "ZDR_EC_OC" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain # in case we want to compare to the original values without any correction
dbzh_ref = "ZDR_EC_OC3_AC2_rain" # ZDR_EC_OC3_AC2_rain, DBZH_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

# repeat for the selected_ML_high cases (or set to None to ignore)
dbzh_tg_0 = "ZDR_EC_OC_AC_WRcorr" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain
dbzh_tg_uncorr_0 = "ZDR_EC_OC" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain # in case we want to compare to the original values without any correction
dbzh_ref_0 = "ZDR_EC_OC_AC_WRcorr" # ZDR_EC_OC3_AC2_rain, DBZH_AC2_rain

xax = r"$Δ\mathrm{Z_{DR}}\ [dB]$" # label for the x axis
unit = re.search(r"\[(.*?)\]", xax).group(1)

varx_range = (5, 19, 1) # start, stop, step # (0.7, 0.98, 0.02)

min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -3 # min and max limits for the y axis
ymax = 2.5

# ML correction coefficients
abml = -0.025

# WR corr based on results
def zdr_wrc(Zm):
    Zm_ = np.where(np.nan_to_num(Zm) < 32.5,
                   np.nan_to_num(Zm),
                   32.5)
    return -0.00022*Zm_ + 0.00032*Zm_**2 # change here to adjust coefficients based on results

if "_WRcorr" in dbzh_tg:
    # Correct wet-radome timesteps
    var_tg_ = "".join(dbzh_tg.split("_WRcorr"))
    selected_ML_low[dbzh_tg] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_tg].append((selected_ML_low[var_tg_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_tg_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

if "_WRcorr" in dbzh_ref:
    # Correct wet-radome timesteps
    var_ref_ = "".join(dbzh_ref.split("_WRcorr"))
    selected_ML_low[dbzh_ref] = []
    for ti in range(len(selected_ML_low[var_ref_])):
        # add to the new variable
        selected_ML_low[dbzh_ref].append((selected_ML_low[var_ref_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_ref_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

if dbzh_tg_0 is not None:
    if "_WRcorr" in dbzh_tg_0:
        # Correct wet-radome timesteps
        var_tg_ = "".join(dbzh_tg_0.split("_WRcorr"))
        selected_ML_high[dbzh_tg_0] = []
        for ti in range(len(selected_ML_high[var_tg_])):
            # add to the new variable
            selected_ML_high[dbzh_tg_0].append((selected_ML_high[var_tg_][ti][0].copy() - zdr_wrc(selected_ML_high["Zm"][ti][0].copy()),
                                             selected_ML_high[var_tg_][ti][1].copy() - zdr_wrc(selected_ML_high["Zm"][ti][1].copy()) ))

    if "_WRcorr" in dbzh_ref_0:
        # Correct wet-radome timesteps
        var_ref_ = "".join(dbzh_ref_0.split("_WRcorr"))
        selected_ML_high[dbzh_ref_0] = []
        for ti in range(len(selected_ML_high[var_ref_])):
            # add to the new variable
            selected_ML_high[dbzh_ref_0].append((selected_ML_high[var_ref_][ti][0].copy() - zdr_wrc(selected_ML_high["Zm"][ti][0].copy()),
                                             selected_ML_high[var_ref_][ti][1].copy() - zdr_wrc(selected_ML_high["Zm"][ti][1].copy()) ))

# ML atten correction based on results
def mlc(phi_bump):
    phi_bump_ = np.where(phi_bump >= 0, phi_bump, 0)
    return abml*phi_bump_ # change here to adjust coefficients based on results

# extract/build necessary variables (selected_ML_low cases)
tg_dbzh = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg] ])

ref_dbzh = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[dbzh_ref] ])

tg_dbzh_uncorr = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg_uncorr] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_phi_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

ref_phi_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

tg_z_beambot = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

ref_z_beambot = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

# tg_height_ml_top_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# ref_height_ml_top_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# # We should only use height_ml_top_qvp values from tg since ref should be above the ML
# # fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
# tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
# ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_top_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

ref_height_ml_top_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_top_qvp)):
    # fill the NaN height_ml_top_qvp values from ref with tg
    ref_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])] = tg_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_top_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_top_qvp[ts][:,0])
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_top_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_top_qvp[ts][:,0])
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_top_qvp[ts] = pd.DataFrame(tg_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_top_qvp[ts] = pd.DataFrame(ref_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_top_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_top_qvp])
ref_height_ml_top_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_top_qvp])

# fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (np.nan_to_num(ref_phi_bump) < 1)  & (ref_phi < 5) & (ref_Zm < 5) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)\
        & (tg_height_ml_top_qvp < 1600)\
        & (tg_phi_bump > varx_range[0])\
        & (tg_z_beambot > tg_height_ml_top_qvp) & (ref_z_beambot > tg_height_ml_top_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMPm > 3) & (ref_TEMPm < 0)\
        & (tg_bca > 135) & (ref_bca > 135)\


delta_dbzh = (tg_dbzh - ref_dbzh)[valid]
delta_dbzh_uncorr = (tg_dbzh_uncorr - ref_dbzh)[valid] # in case we want to compare to the original values without any correction
tg_phi_bump = tg_phi_bump[valid]

delta_dbzh_mlc = delta_dbzh - mlc(tg_phi_bump)

# extract/build necessary variables (selected_ML_high cases)
if dbzh_tg_0 is not None:
    tg_dbzh_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[dbzh_tg_0] ])

    tg_dbzh_uncorr_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[dbzh_tg_uncorr_0] ])

    ref_dbzh_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[dbzh_ref_0] ])

    tg_phi_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

    ref_phi_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high[phi] ])

    tg_Zm_0 = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

    ref_Zm_0 = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_high["Zm"] ]))

    tg_height_ml_bot_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

    ref_height_ml_bot_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["height_ml_bottom_new_gia"] ])

    tg_z_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["z"] ])

    ref_z_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["z"] ])

    tg_TEMP_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

    ref_TEMP_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["TEMP"] ])

    tg_RHOHV_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

    ref_RHOHV_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["RHOHV"] ])

    tg_bca_0 = np.concat([ d1.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

    ref_bca_0 = np.concat([ d2.flatten() for d1,d2 in selected_ML_high["beam_cross_angle"] ])

    # interpolate and extrapolate the ML heights for each day to fill NaNs
    tg_height_ml_bot_qvp_0 = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

    ref_height_ml_bot_qvp_0 = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_high["height_ml_bottom_new_gia_fromqvp"] ]

    for ts in range(len(tg_height_ml_bot_qvp_0)):
        # fill the NaN height_ml_bot_qvp values from tg with ref and viceversa
        tg_height_ml_bot_qvp_0[ts][np.isnan(tg_height_ml_bot_qvp_0[ts])] = ref_height_ml_bot_qvp_0[ts][np.isnan(tg_height_ml_bot_qvp_0[ts])]
        ref_height_ml_bot_qvp_0[ts][np.isnan(ref_height_ml_bot_qvp_0[ts])] = tg_height_ml_bot_qvp_0[ts][np.isnan(ref_height_ml_bot_qvp_0[ts])]

        # remove outliers (median+-std)
        tg_m = np.nanmedian(tg_height_ml_bot_qvp_0[ts][:,0])
        tg_std = np.nanstd(tg_height_ml_bot_qvp_0[ts][:,0])
        tg_height_ml_bot_qvp_0[ts][tg_height_ml_bot_qvp_0[ts] < tg_m-tg_std] = np.nan
        tg_height_ml_bot_qvp_0[ts][tg_height_ml_bot_qvp_0[ts] > tg_m+tg_std] = np.nan
        ref_m = np.nanmedian(ref_height_ml_bot_qvp_0[ts][:,0])
        ref_std = np.nanstd(ref_height_ml_bot_qvp_0[ts][:,0])
        ref_height_ml_bot_qvp_0[ts][ref_height_ml_bot_qvp_0[ts] < ref_m-ref_std] = np.nan
        ref_height_ml_bot_qvp_0[ts][ref_height_ml_bot_qvp_0[ts] > ref_m+ref_std] = np.nan

        # Interpolate and extrapolate to fill NaNs
        tg_height_ml_bot_qvp_0[ts] = pd.DataFrame(tg_height_ml_bot_qvp_0[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
        ref_height_ml_bot_qvp_0[ts] = pd.DataFrame(ref_height_ml_bot_qvp_0[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

    # finally, flatten
    tg_height_ml_bot_qvp_0 = np.concat([ds1.flatten() for ds1 in tg_height_ml_bot_qvp_0])
    ref_height_ml_bot_qvp_0 = np.concat([ds2.flatten() for ds2 in ref_height_ml_bot_qvp_0])

    # fill remaining NaNs with an arbitrarely high value so it does no undesired filtering
    tg_height_ml_bot_qvp_0[np.isnan(tg_height_ml_bot_qvp_0)] = 4000
    ref_height_ml_bot_qvp_0[np.isnan(ref_height_ml_bot_qvp_0)] = 4000

    # filter by valid values according to conditions
    valid_0 = np.isfinite(tg_dbzh_0) & np.isfinite(ref_dbzh_0) & (np.isfinite(tg_phi_0))\
            & (tg_phi_0 > varx_range[0])\
            & (tg_z_0 < tg_height_ml_bot_qvp_0) & (ref_z_0 < ref_height_ml_bot_qvp_0)\
            & (tg_RHOHV_0 > 0.97) & (ref_RHOHV_0 > 0.97)\
            & (tg_TEMP_0 > 3) & (ref_TEMP_0 > 3) \
            & (tg_bca_0 > 135) & (ref_bca_0 > 135)\
            # & (ref_phi_0<ref_phi_max)
            # & (ref_Zm_0<Zm_max) & (tg_Zm_0<Zm_max)\

    delta_dbzh_0 = (tg_dbzh_0 - ref_dbzh_0)[valid_0]
    delta_dbzh_uncorr_0 = (tg_dbzh_uncorr_0 - ref_dbzh_0)[valid_0] # in case we want to compare to the original values without any correction

    # Join both datasets

    delta_dbzh = np.concat((delta_dbzh, delta_dbzh_0))
    delta_dbzh_uncorr = np.concat((delta_dbzh_uncorr, delta_dbzh_uncorr_0))

    delta_dbzh_mlc = np.concat((delta_dbzh_mlc, delta_dbzh_0))

# --- Initialize plot ---
fig, ax = plt.subplots(figsize=(4.5, 3.5))

delta_ref = delta_dbzh_uncorr # delta_dbzh or delta_dbzh_uncorr

# --- Config ---
# bins = np.arange(-15.5, 16.5, 1)  # bin edges
bins = np.arange(-2.05, 2.15, 0.1)  # bin edges
bin_centers = bins[:-1] + np.diff(bins).mean()/2
bin_width = np.diff(bins).mean()

# --- Compute histograms (as frequency %) ---
n1, _ = np.histogram(delta_ref, bins=bins)
n2, _ = np.histogram(delta_dbzh_mlc, bins=bins)

freq1 = n1 / n1.sum() * 100
freq2 = n2 / n2.sum() * 100

# --- Stats for legend ---
mean1 = np.nanmean(delta_ref)
std1  = np.nanstd(delta_ref)
median1 = np.nanmedian(delta_ref)
mean2 = np.nanmean(delta_dbzh_mlc)
std2  = np.nanstd(delta_dbzh_mlc)
median2 = np.nanmedian(delta_dbzh_mlc)

# --- Plot histograms ---
ax.bar(bin_centers, freq1, width=bin_width,
       color="gray", alpha=1.0, label=f"Mean={mean1:.2f} {unit}\nSt_Dev={std1:.2f} {unit}\nMedian={median1:.2f} {unit}")
ax.bar(bin_centers, freq2, width=bin_width,
       color="#00000000", edgecolor="black", linewidth=0.8,
       label=f"Mean={mean2:.2f} {unit}\nSt_Dev={std2:.2f} {unit}\nMedian={median2:.2f} {unit}")

# --- Normal distribution curve fitted to delta_dbzh_mlc ---
x_dense = np.linspace(bins[0], bins[-1], 300)
pdf = scipy.stats.norm.pdf(x_dense, mean2, std2)
# Scale pdf to match histogram (frequency % with bin width 1)
pdf_scaled = pdf * bin_width * 100
ax.plot(x_dense, pdf_scaled, color="red", linewidth=1.5)

# --- Cosmetics ---
ax.set_xlabel(xax)
ax.set_ylabel("Frequency (%)")
ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(0, None)
# ax.set_xticks(np.arange(bins[0]+1, bins[-1], 2))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=8, frameon=True,
          handlelength=1.5, handleheight=1.5)

plt.tight_layout()
plt.show()

#%%% Plot QVPs with and without correction for one event (like figure 11 of Yu et al.)

#%%%% Load the files
from pathlib import Path
ff = glob.glob("/automount/realpep/upload/jgiles/dmi/qvps/2020/2020-03/2020-03-13/HTY/*/*/*h5netcdf.nc")

ds_qvps = {Path(f).parent.name: xr.open_dataset(f) for f in ff}

elevs = sorted([float(k) for k in ds_qvps.keys()])

#%%%% Plot

# 1. Define your time selection and variable
# (e.g., a specific hour, a range of indices, or the whole day)
# Example: first 10 timesteps. Change this to your specific selection.
time_selection = slice("2020-03-13T07:00", "2020-03-13T08:00")
vv = "DBZH_AC" # ZDR_EC_OC_WRC_AC, DBZH_AC
units = "dBZ"
min_count = 20

# 2. Setup the plot
plt.figure(figsize=(4, 5))
plt.title(f"Median {vv} Profiles by Elevation")
plt.xlabel(f"{vv} [{units}]")
plt.ylabel("Height (km a.s.l.)")

# 3. Create a color mapping based on your elevation range
norm = plt.Normalize(vmin=min(elevs), vmax=max(elevs))
norm = plt.Normalize(vmin=0, vmax=len(elevs))
colormap = mpl.cm.YlOrRd_r  # You can use 'plasma', 'inferno', etc.

# 4. Iterate through the sorted elevations and plot
for en, e in enumerate(elevs):
    # Access the dataset (keys in your dict are strings like '0.7')
    ds = ds_qvps[str(e)]

    # Select timesteps and calculate the median along the time dimension
    # .sel(time=...) also works if you have specific datetime strings
    profile = ds[vv].where(ds["DBZH_qvp_count"]>=min_count).sel(time=time_selection).median(dim="time")

    # Get color for this elevation
    color = colormap(norm(en))

    # Plot DBZH_AC vs z
    plt.plot(profile, ds.z/1000, label=f"{e}°", color=color, lw=2)

# 5. Add a colorbar and legend
# sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Elevation Angle')

plt.ylim(0,8)
plt.xlim(0,35)
plt.grid(True, alpha=0.3)
plt.legend(title="Elevation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%% Check how riming affects the calculations

#%%%% Load the riming data
print("... Loading pre-calculated riming ...")
riming_classif = {}
X_DBZH = "DBZH_AC"
X_ZDR = "ZDR_EC_OC_WRC_AC"
suffix_name = "_calibration_attenuation_HTYGZT"

for stratname in ["unfiltered"]:
    if stratname not in riming_classif.keys():
        riming_classif[stratname] = {}
    elif type(riming_classif[stratname]) is not dict:
        riming_classif[stratname] = {}
    print("Loading "+stratname+" riming classification ...")
    for ll in ["hty"]: # ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']:
        if ll not in riming_classif[stratname].keys():
            riming_classif[stratname][ll] = xr.Dataset()
        elif type(riming_classif[stratname][ll]) is not xr.Dataset:
            riming_classif[stratname][ll] = xr.Dataset()

        for xx in ['riming_DR', 'riming_UDR', 'riming_ZDR_DBZH', 'riming_'+X_ZDR+'_'+X_DBZH,
                   ]:
            try:
                riming_classif[stratname][ll] = riming_classif[stratname][ll].assign( xr.open_dataset(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/"+stratname+"/"+ll+"_"+xx+".nc") )
                print(ll+" "+xx+" riming_classif loaded")
            except:
                if stratname == "unfiltered":
                    pass

        # delete entry if empty
        if not riming_classif[stratname][ll]:
            del riming_classif[stratname][ll]

#%%%% Plot example
to_plot = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time="2016-12-30").copy()
to_plot.dropna("z", how="all").plot(x="time", cmap="Reds")
to_plot.height_ml_new_gia_clean.plot(x="time", c="black")
to_plot.height_ml_bottom_new_gia_clean.plot(x="time", c="black")

#%%%% Plot example (2d median filter)

to_plot = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time="2016-12-30").copy()
to_plot.dropna("z", how="all").fillna(0).rolling( time=5, z=5, min_periods=5//2+1, center=True).median().plot(x="time", cmap="Reds")
to_plot.height_ml_new_gia_clean.plot(x="time", c="black")
to_plot.height_ml_bottom_new_gia_clean.plot(x="time", c="black")

#%%%% Plot example (skimage remove_small_objects)
from skimage.morphology import remove_small_objects

to_plot = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time="2016-12-30").copy().dropna("z", how="all").fillna(0)
cleaned_mask = remove_small_objects(to_plot.astype(bool).values, min_size=10).astype(float)
to_plot.copy(data=cleaned_mask).plot(x="time", cmap="Reds")
to_plot.height_ml_new_gia_clean.plot(x="time", c="black")
to_plot.height_ml_bottom_new_gia_clean.plot(x="time", c="black")

#%%%% Plot example (skimage binary_opening) <= I think this looks the best
from scipy.ndimage import binary_opening

to_plot = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time="2016-12-30").copy().dropna("z", how="all").fillna(0)
cleaned_mask = binary_opening(to_plot.values, structure=np.ones((3, 3))).astype(float)
to_plot.copy(data=cleaned_mask).plot(x="time", cmap="Reds")
to_plot.height_ml_new_gia_clean.plot(x="time", c="black")
to_plot.height_ml_bottom_new_gia_clean.plot(x="time", c="black")
plt.ylim(1500,3000)

#%%%% Plot example (skimage binary_opening) with selection of valid riming according to conditions
from scipy.ndimage import binary_opening

# 1. Clean the data
to_plot = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time="2016-12-30").copy().dropna("z", how="all").fillna(0)
cleaned_mask = binary_opening(to_plot.values, structure=np.ones((3, 3))).astype(float)

# Put the cleaned numpy array back into a DataArray so we can use xarray's spatial logic
da_clean = to_plot.copy(data=cleaned_mask)

# 2. Define the Melting Layer boundaries
ml_top = da_clean.height_ml_new_gia_clean
roi_top = ml_top + 300

# 3. Create a boolean mask for the 300m region of interest (ROI) above the ML
roi_mask = (da_clean.z >= ml_top) & (da_clean.z <= roi_top)

# 4. Calculate the fraction of rimed pixels (1s) within that specific 300m layer per time step
# Sum of 1s in the ROI
riming_in_roi = (da_clean == 1).where(roi_mask, 0).sum(dim="z")
# Total number of pixels in the ROI
total_in_roi = roi_mask.sum(dim="z")

# Calculate fraction (using .where to avoid division by zero if a profile has no valid z pixels in that range)
riming_fraction = riming_in_roi / total_in_roi.where(total_in_roi > 0)

# 5. Identify the profiles (times) that meet your >= 80% condition
valid_profiles = riming_fraction >= 0.8

# 6. Apply the new value of 2
# Here, we update pixels to 2 IF:
#   - The time profile meets the 80% rule
#   - The altitude is above the melting layer
#   - The pixel is currently classified as riming (1)
final_da = xr.where(valid_profiles & (da_clean.z > ml_top) & (da_clean == 1), 2, da_clean)

# 7. Plotting the final result
final_da.plot(x="time", cmap="Reds")
to_plot.height_ml_new_gia_clean.plot(x="time", c="black")
to_plot.height_ml_bottom_new_gia_clean.plot(x="time", c="black")
plt.ylim(0, 3000)
plt.show()

#%%%% Plot boxplot of delta DBZH/ZDR vs target PHI bump (ML attenuation) filtered by riming
from scipy.ndimage import binary_opening

phi = "PHIDP_OC_MASKED"
dbzh_tg = "ZDR_EC_OC_AC2_rain_WRcorr" # ZDR_EC_OC_AC2_rain, DBZH_AC2_rain
dbzh_ref = "ZDR_EC_OC3" # ZDR_EC_OC3_AC2_rain, DBZH_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

yax = r"$Δ\mathrm{Z_{DR}}\ [dB]$" # label for the y axis
xax = r"$Δ\mathrm{\Phi_{DP}^{ML}}\ [°]$" # label for the x axis

varx_range = (0, 19, 1) # start, stop, step # (0.7, 0.98, 0.02)

min_bin_n = 30 # min count of valid values inside bin to be included in the fitting

sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -3 # min and max limits for the y axis
ymax = 2.5

# WR corr based on results
def zdr_wrc(Zm):
    Zm_ = np.where(np.nan_to_num(Zm) < 32.5,
                   np.nan_to_num(Zm),
                   32.5)
    return -0.00022*Zm_ + 0.00032*Zm_**2 # change here to adjust coefficients based on results

if "_WRcorr" in dbzh_tg:
    # Correct wet-radome timesteps
    var_tg_ = "".join(dbzh_tg.split("_WRcorr"))
    selected_ML_low[dbzh_tg] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_tg].append((selected_ML_low[var_tg_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_tg_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

if "_WRcorr" in dbzh_ref:
    # Correct wet-radome timesteps
    var_ref_ = "".join(dbzh_ref.split("_WRcorr"))
    selected_ML_low[dbzh_ref] = []
    for ti in range(len(selected_ML_low[var_tg_])):
        # add to the new variable
        selected_ML_low[dbzh_ref].append((selected_ML_low[var_ref_][ti][0].copy() - zdr_wrc(selected_ML_low["Zm"][ti][0].copy()),
                                         selected_ML_low[var_ref_][ti][1].copy() - zdr_wrc(selected_ML_low["Zm"][ti][1].copy()) ))

# ## Process the riming classification
# selected_ML_low_riming = {}

# for date in selected_ML_low_dates.keys():
#     # 1. Clean the data
#     riming_date = riming_classif['unfiltered']['hty']['riming_ZDR_EC_OC_WRC_AC_DBZH_AC'].sel(time=date).copy().dropna("z", how="all").fillna(0)
#     cleaned_mask = binary_opening(riming_date.values, structure=np.ones((3, 3))).astype(float)

#     # Put the cleaned numpy array back into a DataArray so we can use xarray's spatial logic
#     da_clean = riming_date.copy(data=cleaned_mask)

#     # 2. Define the Melting Layer boundaries
#     ml_top = da_clean.height_ml_new_gia_clean
#     roi_top = ml_top + 300

#     # 3. Create a boolean mask for the 300m region of interest (ROI) above the ML
#     roi_mask = (da_clean.z >= ml_top) & (da_clean.z <= roi_top)

#     # 4. Calculate the fraction of rimed pixels (1s) within that specific 300m layer per time step
#     # Sum of 1s in the ROI
#     riming_in_roi = (da_clean == 1).where(roi_mask, 0).sum(dim="z")
#     # Total number of pixels in the ROI
#     total_in_roi = roi_mask.sum(dim="z")

#     # Calculate fraction (using .where to avoid division by zero if a profile has no valid z pixels in that range)
#     riming_fraction = riming_in_roi / total_in_roi.where(total_in_roi > 0)

#     # 5. Identify the profiles (times) that meet >= 80% condition
#     valid_profiles = riming_fraction >= 0.8

#     # 6. assign the final values
#     selected_ML_low_riming[date] = valid_profiles.copy()

# # Print some stats: % of rimed profiles
# selected_ML_low_riming_sorted = dict(sorted(selected_ML_low_riming.items()))
# riming_concat = xr.concat(selected_ML_low_riming_sorted.values(), dim="time")
# riming_count = riming_concat.size
# riming_fraction = round(float(riming_concat.sum()/riming_count*100),2)
# print(f"Proportion of rimed profiles: {riming_fraction}% of {riming_count}")
# riming_count_ml = int(riming_concat.where(riming_concat.height_ml_new_gia_clean.notnull()).count())
# riming_fraction = round(float(riming_concat.where(riming_concat.height_ml_new_gia_clean.notnull()).sum()/riming_count_ml*100),2)
# print(f"Proportion of rimed profiles with ML detected: {riming_fraction}% of {riming_count_ml}")

# extract/build necessary variables
tg_dbzh = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[dbzh_tg] ])

ref_dbzh = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[dbzh_ref] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi] ])

ref_phi = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_phi_bump = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

ref_phi_bump = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[phi+"_MLbump"] ])

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

tg_RHOHV = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

ref_RHOHV = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["RHOHV"] ])

tg_z_beambot = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

ref_z_beambot = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["z_beambot"] ])

tg_bca = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

ref_bca = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["beam_cross_angle"] ])

tg_riming = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["riming"] ])

# tg_height_ml_top_qvp = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# ref_height_ml_top_qvp = np.concat([ d2.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ])

# Alternative: interpolate and extrapolate the ML heights for each day to fill NaNs
tg_height_ml_top_qvp = [ pd.DataFrame(d1).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

ref_height_ml_top_qvp = [ pd.DataFrame(d2).ffill(axis=1).values for d1,d2 in selected_ML_low["height_ml_new_gia_fromqvp"] ]

for ts in range(len(tg_height_ml_top_qvp)):
    # fill the NaN height_ml_top_qvp values from ref with tg
    ref_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])] = tg_height_ml_top_qvp[ts][np.isnan(ref_height_ml_top_qvp[ts])]

    # remove outliers (median+-std)
    tg_m = np.nanmedian(tg_height_ml_top_qvp[ts][:,0])
    tg_std = np.nanstd(tg_height_ml_top_qvp[ts][:,0])
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] < tg_m-tg_std] = np.nan
    tg_height_ml_top_qvp[ts][tg_height_ml_top_qvp[ts] > tg_m+tg_std] = np.nan
    ref_m = np.nanmedian(ref_height_ml_top_qvp[ts][:,0])
    ref_std = np.nanstd(ref_height_ml_top_qvp[ts][:,0])
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] < ref_m-ref_std] = np.nan
    ref_height_ml_top_qvp[ts][ref_height_ml_top_qvp[ts] > ref_m+ref_std] = np.nan

    # Interpolate and extrapolate to fill NaNs
    tg_height_ml_top_qvp[ts] = pd.DataFrame(tg_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values
    ref_height_ml_top_qvp[ts] = pd.DataFrame(ref_height_ml_top_qvp[ts]).interpolate(axis=0).ffill(axis=0).bfill(axis=0).values

# finally, flatten
tg_height_ml_top_qvp = np.concat([ds1.flatten() for ds1 in tg_height_ml_top_qvp])
ref_height_ml_top_qvp = np.concat([ds2.flatten() for ds2 in ref_height_ml_top_qvp])

# fill remaining NaNs with an arbitrarely low value so it does no undesired filtering
tg_height_ml_top_qvp[np.isnan(tg_height_ml_top_qvp)] = 0
ref_height_ml_top_qvp[np.isnan(ref_height_ml_top_qvp)] = 0

# # create analogous arrays for the riming classif
# dates_n = [date for date in selected_ML_low_dates.keys() for n in selected_ML_low_dates[date]]
# tg_riming = np.concat([ np.broadcast_to(selected_ML_low_riming[dates_n[n]].values[:,np.newaxis], d1.shape).flatten() for n, (d1,d2) in enumerate(selected_ML_low[dbzh_tg]) ])

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (np.nan_to_num(ref_phi_bump) < 1)  & (ref_phi < 5) & (ref_Zm < 5) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_dbzh) & np.isfinite(ref_dbzh)\
        & (tg_height_ml_top_qvp < 1600)\
        & (tg_phi_bump > varx_range[0]) & (tg_phi_bump < varx_range[1] - varx_range[2])\
        & (tg_z_beambot > tg_height_ml_top_qvp) & (ref_z_beambot > tg_height_ml_top_qvp)\
        & (tg_RHOHV > 0.97) & (ref_RHOHV > 0.97)\
        & (tg_TEMPm > 3) & (ref_TEMPm < 0)\
        & (tg_bca > 135) & (ref_bca > 135)\
        & (np.nan_to_num(tg_riming)>0.5)

delta_dbzh = (tg_dbzh - ref_dbzh)[valid]
tg_phi_bump = tg_phi_bump[valid]

# In case we need to filter out unrealistic values
# tg_phi_bump = tg_phi_bump[delta_dbzh>-4]
# delta_dbzh = delta_dbzh[delta_dbzh>-4]

# Calculate best linear fit
lfit = np.polynomial.Polynomial.fit(tg_phi_bump, delta_dbzh, 1)
lfit_str = str(lfit.convert()).replace("x", "Phi ML bump")

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_phi_bump into bins
bin_indices = np.digitize(tg_phi_bump, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Remove bins that have less than min_bin_n valid values
valid_bins = [ np.isfinite(arr).sum() >= min_bin_n  for arr in box_data ]

# Plot
plt.figure(figsize=(6, 3.5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2,
                 showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                 medianprops={"color":"black"}, meanprops={"marker":"."})
plt.xlim(bins[0], bins[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bins, bins)
# plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])
# plt.title("Boxplots of delta "+dbzh_tg+" vs "+phi+"_MLbump"+" bins")

# # add linear fit
# plt.plot([bins[0], bins[-1]], [lfit(bins[0]), lfit(bins[-1])])
# plt.text(0.95, 0.9, "Linear fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
#          horizontalalignment="right")

# add a second linear fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
lfit_m = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 2)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
# plt.plot([bins[0], bins[-1]], [lfit_m(bins[0]), lfit_m(bins[-1])], c="red")
# plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
#          horizontalalignment="right")

# add a third linear fit using the medians and IQRs of each bin
variances = np.array([vals.var(ddof=1) for vals in box_data])
iqr = np.array([np.nanquantile(vals,0.75) for vals in box_data]) - np.array([np.nanquantile(vals,0.25) for vals in box_data])
weights = 1 / iqr**2 # 1 / variances
weights[~np.isfinite(weights)] = 0
w = np.sqrt(weights)
lfit_mw = np.polynomial.Polynomial.fit(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)], 1, w=w[np.array(valid_bins)])
lfit_mw_rcoefs = np.round(lfit_mw.convert().coef, 3)
lfit_mw_rounded = np.polynomial.Polynomial(lfit_mw_rcoefs)
lfit_mw_str = str(lfit_mw_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
plt.plot([bins[0], bins[-1]], [lfit_mw(bins[0]), lfit_mw(bins[-1])], c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_mw_str+"", transform=plt.gca().transAxes,
         c="red", horizontalalignment="right")

plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers[::2], counts[::2]):
    plt.text(x, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')
for x, n in zip(bin_centers[1::2], counts[1::2]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 1% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# # Print p value and other stats
# scipy.stats.linregress(bin_centers[np.array(valid_bins)], medians[np.array(valid_bins)])

# Print p value and other stats (for weighted fit)
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x]
X = np.column_stack((np.ones_like(bin_centers[np.array(valid_bins)]), bin_centers[np.array(valid_bins)]))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.WLS(medians[np.array(valid_bins)], X, weights=weights[np.array(valid_bins)])
results = model.fit(cov_type='HC3')

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())


#%%% DEPRECATED Plot boxplot of delta DBZH/ZDR vs target ZH/RHOHV min/max in ML (ML attenuation)

varx = "DBZH_AC2_rain_MLmax" # RHOHV_MLmin
vary = "DBZH_AC2_rain" # ZDR_EC_OC_AC2_rain
TEMPm = "TEMPm"
TEMP = "TEMP"

varx_range = (10, 60, 5) # start, stop, step # (0.7, 0.93, 0.02)

# We only need to check that TEMPm is appropriate for each radar

# extract/build necessary variables
tg_vary = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[vary] ])

ref_vary = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[vary] ])

tg_varx = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[varx] ])

ref_varx = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[varx] ])

tg_Zm = np.nan_to_num(np.concat([ d1.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

ref_Zm = np.nan_to_num(np.concat([ d2.flatten() for d1,d2 in selected_ML_low["Zm"] ]))

tg_TEMPm = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

ref_TEMPm = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMPm] ])

tg_TEMP = np.concat([ d1.flatten() for d1,d2 in selected_ML_low[TEMP] ])

ref_TEMP = np.concat([ d2.flatten() for d1,d2 in selected_ML_low[TEMP] ])

tg_height_ml_top = np.concat([ d1.flatten() for d1,d2 in selected_ML_low["height_ml_new_gia"] ])

# filter by valid values according to conditions
#!!! The best filter would have ref_TEMPm < -1, but looks like no event so far
# meets this condition. So let's use PHI and Zm as an alternative for now
# valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_varx) & np.isfinite(ref_vary)
valid = (tg_TEMPm > 3) & (ref_TEMPm < 0) & np.isfinite(tg_varx) & np.isfinite(tg_vary)\
    & np.isfinite(ref_vary) & (tg_height_ml_top < 1600) & (tg_varx > varx_range[0]) & (tg_varx < varx_range[1])

delta_vary = (tg_vary - ref_vary)[valid]
tg_varx = tg_varx[valid]

# Calculate best linear fit
lfit = np.polynomial.Polynomial.fit(tg_varx, delta_vary, 1)
lfit_str = str(lfit.convert()).replace("x", varx)

# Box plots like in the paper
# Define bins
bins = np.arange(varx_range[0], varx_range[1], varx_range[2])  # 0,1,2,3,4,5
bin_centers = bins[:-1] + np.diff(bins).mean()/2

# Digitize tg_varx into bins
bin_indices = np.digitize(tg_varx, bins) - 1

# Prepare data for boxplot
box_data = [delta_vary[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Plot
plt.figure(figsize=(9, 5))
bp = plt.boxplot(box_data, positions=bin_centers, widths=np.diff(bins).mean()/2, showmeans=True)
plt.xlim(bins[0], bins[-1])
plt.xlabel(varx+" (binned)")
plt.ylabel("delta "+vary)
plt.title("Boxplots of delta "+vary+" vs "+varx+" bins")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bin_centers, [f"{round(b, 2)}-{round(b+varx_range[2], 2)}" for b in bins[:-1]])

# add linear fit
plt.plot([bins[0], bins[-1]], [lfit(bins[0]), lfit(bins[-1])])
plt.text(0.95, 0.9, "Linear fit: "+lfit_str, transform=plt.gca().transAxes, c="blue",
         horizontalalignment="right")

# add a second linear fit using the medians
medians = [line.get_ydata()[0] for line in bp['medians']]
lfit_m = np.polynomial.Polynomial.fit(bin_centers, medians, 1)
lfit_m_str = str(lfit_m.convert()).replace("x", varx)
plt.plot([bins[0], bins[-1]], [lfit_m(bins[0]), lfit_m(bins[-1])], c="red")
plt.text(0.95, 0.85, "Linear fit (medians): "+lfit_m_str, transform=plt.gca().transAxes, c="red",
         horizontalalignment="right")

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

#%% Wet radome attenuation estimation based on ZDR offsets and Zm (no radar volume matching)
# We don't get enough data points for wet radome attenuation quantification by volume matching the two radars.
# Then, we try to at least quantify the differential attenuation using ZDR offsets calculated for different Zm values.

# =============================================================================
# 1. CONFIGURATION & DISCOVERY
# =============================================================================

# Define the limits for each Wet Radome (WR) category
wr_limits = {
    "noWR": (0, 10), # noWR must be first
    "WR1015": (10, 15),
    "WR1520": (15, 20),
    "WR2025": (20, 25),
    "WR2530": (25, 30),
    "WR3035": (30, 35),
    "WR3550": (35, 50) # Assuming 50 is a reasonable max
}
wr_tags = list(wr_limits.keys())

# Plot settings
sc = False # show boxplots caps?
sf = False # show boxplots outliers?
wp = 0 # position of the whiskers as proportion of (Q3-Q1), default is 1.5

ymin = -0.15 # min and max limits for the y axis
ymax = 0.45
xminmax = np.arange(0,55,5) # min and max limits for the x axis and grid lines

yax = r"$\Delta Z_{DR}$ Offset [dB]" # label for the y axis
xax = r"$\mathrm{Z_{H}^m}\ [dBZ]$" # label for the x axis

# Pattern to find ALL offset files (all elevations, all WR modes)
# Structure: .../LR_consistency/<YYYY>/<MM>/<DD>/HTY/<MODE>/<ELEV>/*-zdr_offset_belowML_<TAG>-*.nc
offset_height_lim = "belowML"
offset_pattern = "/automount/realpep/upload/jgiles/dmi/calibration_WRtest/zdr/*/*/*/*/*/*/*/*-zdr_offset_"+offset_height_lim+"_*WR*-*.nc"

print("Step 1: Discovering files...")
files = glob.glob(offset_pattern)
print(f"Found {len(files)} offset files. Parsing metadata...")

# =============================================================================
# 2. METADATA PARSING
# =============================================================================

data_entries = []

# Regex to extract info from path
# Expected path ending: .../2016/04/09/HTY/MON_YAZ_C/12.0/filename.nc
# We need: Date, Mode, Elevation, WR_Tag
for fpath in files:
    parts = fpath.split(os.sep)

    try:
        # Extract from directory structure (assuming fixed depth from 'LR_consistency')
        # Adjust indices if your mount point depth differs
        # .../LR_consistency/YYYY/YYYY-MM/YYYY-MM-DD/HTY/MODE/ELEV/file.nc
        # parts[-5] = YYYY-MM-DD
        # parts[-3] = MODE, parts[-2] = ELEV

        loc = parts[-4]
        method = parts[-8]
        date_str = f"{parts[-5]}" # YYYY-MM-DD
        mode = parts[-3]
        elev = float(parts[-2])

        # Extract WR tag from filename
        fname = parts[-1]
        # Filename format: ...-zdr_offset_belowML_WRX-....nc
        # We can regex specifically for the tag
        match = re.search(offset_height_lim+r"_(noWR|WR\d+)-", fname)
        if match:
            tag = match.group(1)
        else:
            continue

        data_entries.append({
            "loc": loc,
            "method": method,
            "date": pd.Timestamp(date_str),
            "mode": mode,
            "elev": elev,
            "tag": tag,
            "path": fpath
        })

    except Exception as e:
        print(f"Error parsing {fpath}: {e}")
        continue

df_files = pd.DataFrame(data_entries)

# =============================================================================
# 3. LOAD ZDR OFFSETS
# =============================================================================

print("Step 2: Loading ZDR Offsets...")

offset_values = []

# We can loop through files and load the single value (or daily mean)
# Since files are small, this is reasonably fast.
for idx, row in df_files.iterrows():
    try:
        with xr.open_dataset(row['path']) as ds:
            # Resample to daily mean to handle potential sub-daily chunks
            # Assuming the file contains one day
            val = ds['ZDR_offset'].mean().item()

            offset_values.append({
                "loc": row['loc'],
                "method": row['method'],
                "date": row['date'],
                "elev": row['elev'],
                "mode": row['mode'],
                "tag": row['tag'],
                "offset": val
            })
    except Exception as e:
        print(f"Failed to load {row['path']}")

df_offsets = pd.DataFrame(offset_values)

# Pivot to get columns: date, elev, noWR, WR1015, WR1520...
# This aligns 'noWR' (baseline) with 'WRxx' for the same day/elev
df_pivot = df_offsets.pivot(index=['loc', 'method', 'date', 'elev', 'mode'], columns='tag', values='offset').reset_index()

# =============================================================================
# 4. LOAD QVP COUNTS (The Efficiency Step)
# =============================================================================
# We need to find valid timesteps for each WR category.
# Strategy: Find unique (Date, Elev, Mode) -> Find QVP -> Count timestamps

print("Step 3: Calculating valid timesteps from QVPs...")

# Unique combinations requiring QVP check
unique_combos = df_pivot[['loc', 'method', 'date', 'elev', 'mode']].drop_duplicates()

validity_dict = {} # Key: (date, elev), Value: {WR1015: count, ...}

for idx, row in unique_combos.iterrows():
    loc = row['loc']
    method = row['method']
    dt = row['date']
    elev = row['elev']
    mode = row['mode']

    # Construct QVP Search Path
    # Path: .../qvps/YYYY/YYYY-MM/YYYY-MM-DD/HTY/MODE/ELEV/*allmoms*.nc
    date_path = dt.strftime("%Y/%Y-%m/%Y-%m-%d")
    qvp_glob = f"/automount/realpep/upload/jgiles/dmi/qvps/{date_path}/{loc}/{mode}/{elev}/*allmoms*.nc"

    qvp_files = glob.glob(qvp_glob)

    if not qvp_files:
        # print(f"No QVP found for {dt.date()} {elev} {mode}")
        continue

    # Load QVP (Select only necessary variables to save memory)
    try:
        # Open first match (usually only one per day/elev)
        with xr.open_dataset(qvp_files[0]) as ds_qvp:
            zm = ds_qvp['Zm'].load() # Load into memory for fast numpy ops
            mlh = ds_qvp['height_ml_new_gia_clean'].load() # Load into memory for fast numpy ops

            counts = {}
            for tag, (low, high) in wr_limits.items():
                # Count timesteps in this range
                if "ML" in offset_height_lim:
                    count = ((zm >= low) & (zm < high) & mlh.notnull()).sum().item()
                else:
                    count = ((zm >= low) & (zm < high)).sum().item()
                counts[tag] = count

            validity_dict[(dt, elev, mode)] = counts

    except Exception as e:
        print(f"Error reading QVP {qvp_files[0]}: {e}")

# =============================================================================
# 5. CALCULATE DIFFERENCES & FILTER
# =============================================================================

print("Step 4: Processing differences...")

_ = wr_tags.pop(0) # remove 'noWR'
diff_data = {tag: [] for tag in wr_tags}
min_timesteps = 5

for idx, row in df_pivot.iterrows():
    key = (row['date'], row['elev'], row['mode'])

    # Skip if we didn't find a QVP for this day
    if key not in validity_dict:
        continue

    counts = validity_dict[key]
    baseline = row['noWR'] # The dry reference for this specific day/elev

    # If baseline is missing, we can't compare
    if pd.isna(baseline):
        continue

    for tag in wr_tags:
        # Skip if this WR tag data is missing for this day
        if tag not in row or pd.isna(row[tag]):
            continue

        # Check data quality (timesteps)
        if (counts.get(tag, 0) >= min_timesteps) & (counts.get("noWR", 0) >= min_timesteps) :
            # Calculate Difference: Wet - Dry
            diff = row[tag] - baseline
            diff_data[tag].append(diff)

# =============================================================================
# 6. PLOTTING
# =============================================================================

print("Step 5: Plotting...")

# Prepare list of arrays for boxplot
plot_data = [diff_data[tag] for tag in wr_tags]
labels = [t.replace("WR", "") for t in wr_tags]

fig, ax = plt.subplots(figsize=(6, 3.5))

# Boxplot
bin_centers = [(wr_limits[tag][1] + wr_limits[tag][0])/2 for tag in wr_tags]
bp = ax.boxplot(plot_data, positions=bin_centers,
                   widths=2.5, #labels=labels, patch_artist=True,
                   showmeans=True, showcaps=sc, showfliers=sf, whis=wp,
                   medianprops={"color":"black"}, meanprops={"marker":"."})

plt.xlim(xminmax[0], xminmax[-1])
plt.ylim(ymin, ymax)
plt.xlabel(xax)
plt.ylabel(yax)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(xminmax, xminmax)

# # Style
# colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.7)

# # Stats for legend
# counts_str = [f"N={len(d)}" for d in plot_data]
# for i, count_text in enumerate(counts_str):
#     ax.text(i+1, ax.get_ylim()[0], count_text,
#             horizontalalignment='center', verticalalignment='bottom',
#             fontsize=8, fontweight='bold')

ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# add a cuadratic fit using the medians
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
bin_centers_valid = np.concatenate(([0], np.array(bin_centers)[np.isfinite(medians)]))
medians_valid = np.concatenate(([0], medians[np.isfinite(medians)]))
lfit_m = np.polynomial.Polynomial.fit(bin_centers_valid, medians_valid, 2)
lfit_m_rcoefs = np.round(lfit_m.convert().coef, 5)
lfit_m_rounded = np.polynomial.Polynomial(lfit_m_rcoefs)
lfit_m_str = str(lfit_m_rounded.convert()).replace("x", re.sub(r'\[.*?\]', '', xax))
x_dense = np.linspace(xminmax[0], xminmax[-1], 100) # 100 points for a smooth curve
plt.plot(x_dense, lfit_m(x_dense), c="red")
plt.text(0.95, 0.85, r"Best fit: "+re.sub(r'\[.*?\]', '', yax)+"="+lfit_m_str+"", transform=plt.gca().transAxes, c="red",
         horizontalalignment="right")

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, [f"{len(d)}" for d in plot_data]):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()

# Print p value and other stats
# use alternative to scipy.optimize.curve_fit (there is no quadratic equivalent)

# We add a column for the constant (intercept) and the squared term
# Stack columns: [1, x, x^2]
X = np.column_stack((np.ones_like(bin_centers_valid),
                     bin_centers_valid,
                     bin_centers_valid**2))

# 2. Fit the model (OLS = Ordinary Least Squares)
model = sm.OLS(medians_valid, X)
results = model.fit()

# 3. Get the stats
print(f"R²: {results.rsquared:.4f}")
print(f"p-values (const, x, x²): {results.pvalues}")
print(f"Prob (F-statistic): {results.f_pvalue}")

# You can also print a comprehensive summary table
print(results.summary())


#%% TEST: ML detection over PPI or better ML from QVPs
ds = xr.open_dataset('/automount/realpep//upload/jgiles/dmi/final_ppis/2020/2020-12/2020-12-14/HTY/VOL_A/1.5/VOL_A-allmoms-1.5-2020-12-14-HTY-h5netcdf.nc')

ds0=ds.sel(time="2020-12-14T17", method="nearest").copy()

moments={"DBZH": (10., 60.), "RHOHV": (0.65, 1.), "PHIDP_OC": (-20, 180)}

ds_slice = ds.sel(time=slice("2020-12-14T17", "2020-12-14T17:10"))

# Estimate melting layer over the PPI
ds_slice2 = utils.melting_layer_qvp_X_new(ds_slice, moments=moments, dim="range", fmlh=0.3, grad_thresh=0.0001,
         xwin=3, ywin=3, min_h=200, rhohv_thresh_gia=(0.99,1), all_data=True, clowres=False)

# Plot PPI with ML values
height_ml_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==ds_slice2.height_ml_new_gia).isel(time=0).values.flatten()

height_ml_bottom_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==ds_slice2.height_ml_bottom_new_gia).isel(time=0).values.flatten()

ds_slice2.isel(time=0).RHOHV.wrl.vis.plot(vmin=0.8, vmax=1, xlim=(35,36.5), ylim=(35.5,37))
ax=plt.gca()
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        marker=".")
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        marker=".")

# Lets interpolate the missing values, wrapping around the 360 degrees, and plot again
# (this basically generates the values to fill the plotted lines)
def interpolate_cyclic(da, dim="azimuth", period=360, max_gap=10):
    coord = da[dim]
    da_ext = xr.concat(
        [da.isel({dim: -1}).assign_coords({dim: coord[-1] - period}),
         da,
         da.isel({dim: 0}).assign_coords({dim: coord[0] + period})],
        dim=dim
    )
    out = da_ext.interpolate_na(dim, max_gap=max_gap)
    return out.sel({dim: slice(coord.min(), coord.max())})

height_ml_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==interpolate_cyclic(ds_slice2.height_ml_new_gia.compute())).isel(time=0).values.flatten()

height_ml_bottom_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==interpolate_cyclic(ds_slice2.height_ml_bottom_new_gia.compute())).isel(time=0).values.flatten()

ds_slice2.isel(time=0).RHOHV.wrl.vis.plot(vmin=0.8, vmax=1, xlim=(35,36.5), ylim=(35.5,37))
ax=plt.gca()
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        marker=".")
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        marker=".")

# Let's run a rolling median to remove crap values
def rolling_wrap(da, dim="azimuth", window=5, func="median", **kwargs):
    n = window // 2
    da_ext = da.pad({dim: (n, n)}, mode="wrap", keep_attrs=True)
    rolled = getattr(da_ext.rolling({dim: window}, center=True, min_periods=n), func)(**kwargs)
    return rolled.isel({dim: slice(n, -n)})

height_ml_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==interpolate_cyclic(rolling_wrap(ds_slice2.height_ml_new_gia.compute()))).isel(time=0).values.flatten()

height_ml_bottom_new_gia_valid0 = ds_slice2.range.broadcast_like(ds_slice2.z).where(ds_slice2.range.broadcast_like(ds_slice2.z)==interpolate_cyclic(rolling_wrap(ds_slice2.height_ml_bottom_new_gia.compute()))).isel(time=0).values.flatten()

ds_slice2.isel(time=0).RHOHV.wrl.vis.plot(vmin=0.8, vmax=1, xlim=(35,36.5), ylim=(35.5,37))
ax=plt.gca()
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_new_gia_valid0)],
        marker=".")
ax.plot(ds_slice.x.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        ds_slice.y.values.flatten()[np.isfinite(height_ml_bottom_new_gia_valid0)],
        marker=".")

# What comes out of calculating the ML on the QVP?
ds_slice_qvp, ds_slice_qvp_count = utils.compute_qvp(ds_slice,
                                 min_thresh={"RHOHV":0.7, "DBZH":0, "ZDR_EC_OC":-1, "SNRH":15,
                                             "SNRHC":15, "SQIH":0.5}, output_count=True)
# adding a min count threshold is key to get rid of irrelevant QVP values that mess up the ML detection
moments={"DBZH": (10., 60.), "RHOHV": (0.65, 1.), "PHIDP_OC": (-20, 180)}

ds_slice_qvp2 = utils.melting_layer_qvp_X_new(ds_slice_qvp.where(ds_slice_qvp_count>20), moments=moments, dim="z", fmlh=0.3, grad_thresh=0.0001,
         xwin=3, ywin=3, min_h=200, rhohv_thresh_gia=(0.99,1), all_data=True, clowres=False)


ds_slice.isel(time=0).DBZH.wrl.vis.plot(vmin=0, vmax=60, xlim=(34,38), ylim=(34,38))
ax = plt.gca()
ds_slice.isel(time=0)["z"].wrl.vis.plot(ax=ax,
                      levels=[ds_slice_qvp2.isel(time=0)["height_ml_bottom_new_gia"].values,
                              ds_slice_qvp2.isel(time=0)["height_ml_new_gia"].values],
                      cmap="black",
                      func="contour")

ds_slice.isel(time=0)["TEMP"].wrl.vis.plot(ax=ax,
                      levels=[-1,6],
                      cmap="Reds",
                      func="contour")

# Plot QVP over more timesteps
ds_slice = ds.sel(time=slice("2020-12-14T15", "2020-12-14T18"))
ds_slice_qvp, ds_slice_qvp_count = utils.compute_qvp(ds_slice,
                                 min_thresh={"RHOHV":0.7, "DBZH":0, "ZDR_EC_OC":-1, "SNRH":15,
                                             "SNRHC":15, "SQIH":0.5}, output_count=True)

ds_slice_qvp2 = utils.melting_layer_qvp_X_new(ds_slice_qvp.where(ds_slice_qvp_count>20), moments=moments, dim="z", fmlh=0.3, grad_thresh=0.0001,
         xwin=5, ywin=5, min_h=200+ds_slice.altitude, rhohv_thresh_gia=(0.99,1), all_data=True, clowres=False)


ds_slice_qvp2.RHOHV.plot(x="time", vmin=0.8, vmax=1, cmap="HomeyerRainbow")
ds_slice_qvp2.height_ml_bottom_new_gia.plot(x="time", c="black")
ds_slice_qvp2.height_ml_new_gia.plot(x="time", c="black")

ds_slice0 = ds_slice.sel(time="2020-12-14T16", method="nearest")
ds_slice_qvp20 = ds_slice_qvp2.sel(time="2020-12-14T16", method="nearest")

ds_slice0.RHOHV.wrl.vis.plot(vmin=0.8, vmax=1, xlim=(34,38), ylim=(34,38))
ax = plt.gca()
ds_slice0["z"].wrl.vis.plot(ax=ax,
                      levels=[ds_slice_qvp20["height_ml_bottom_new_gia"].values,
                              ds_slice_qvp20["height_ml_new_gia"].values],
                      cmap="black",
                      func="contour")

ds_slice_qvp20.DBZH.plot()
ds_slice_qvp20.PHIDP_OC_MASKED.plot()
ax = plt.gca()
ax2 = ax.twinx()
ds_slice_qvp20.RHOHV.plot(ax=ax2)

# Try new atten corr
ds_slice0 = utils.attenuation_corr_linear(ds_slice0,
                              alpha = 0.08, beta = 0.02, alphaml = 0.16, betaml = 0.04,
                            dbzh = "DBZH", zdr = "ZDR_EC_OC", phidp = "PHIDP_OC_MASKED")

(ds_slice0.ZDR_EC_OC_AC-ds_slice0.ZDR_EC_OC).wrl.vis.plot(vmin=-1, vmax=3, xlim=(34,38), ylim=(34,38))
ax = plt.gca()
ds_slice0["z"].wrl.vis.plot(ax=ax,
                      levels=[ds_slice_qvp20["height_ml_bottom_new_gia"].values,
                              ds_slice_qvp20["height_ml_new_gia"].values],
                      cmap="black",
                      func="contour")

