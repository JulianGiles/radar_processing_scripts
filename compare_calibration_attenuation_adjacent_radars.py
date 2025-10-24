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


os.environ['WRADLIB_DATA'] = '/home/jgiles/wradlib-data-main'
# set earthdata token (this may change, only lasts a few months https://urs.earthdata.nasa.gov/users/jgiles/user_tokens)
os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTcwMzMzMjE5NywiaWF0IjoxNjk4MTQ4MTk3LCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.6DB5JJ9vdC7Vvwvaa7_mb_HbpVAh05Gz26dzdateN10C5lAd2X4a1_zClx7KkTpyoeVZSzkGSgtcd5Azc_btG0am4r2aJDGv4Zp4Vg55G4mcZMp-aTR7D520InQLMvqFacVO5wwmvfNWzMT4TyLGcXwPuX58s1oaFR5gRL9T30pXN9nEs-1aJg4LUl553PfdOvvom3q-JKXFtSTE2nLyEQOzWW36COl1aHwq6Wh4ykn4aq6ppTVAIeHdgkjtnQtxbhd9trm16fSbX9HIgG7n-drnz_v-WMeFuycMHa-zLDKnd3U3oZW6XAUq2akw2ddu6ChwoTZ4Ix2di7fudioo9Q"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%% Check which elevations would be better suited for volume matching

#%%% Load two example files
ff1 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2016/2016-10/2016-10-28/HTY/MON_YAZ_B/1.5/MON_YAZ_B-allmoms-1.5-20162016-102016-10-28-HTY-h5netcdf.nc"
ff2 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2016/2016-10/2016-10-28/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2016-10-28-GZT-h5netcdf.nc"

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
    beamwidth=1.0,
    terrain=True,
    dem=None,
    npts=400,
    cmap1="PuBu_r",
    cmap2="OrRd_r",
    pad_deg=0.02,
    show=True,
    figsize=(12, 5),
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
            ds_srtm = dem if dem is not None else wrl.io.get_srtm(bbox)
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
    ax.text(0.0, alt1 / 1000.0 + 0.2, "Radar 1", color="k", va="bottom", ha="left")

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
    ax.text(dist_total_km, alt2 / 1000.0 + 0.2, "Radar 2", color="k", va="bottom", ha="right")

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
        leg1 = ax.legend(uniq_handles, uniq_labels, prop={"family": "monospace"}, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(leg1)

    if center_last is not None and edge_last is not None:
        legend2 = {"Center": center_last[0], "3 dB": edge_last[0]}
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
    ax.set_title("Dual-Radar Scan Strategies (shared transect & terrain)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


#%%% Plot scan strategies

plot_dual_scan_strategy(
    ds1, ds2,
    elevs1=[0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 4.5, 6.0, 8.0, 12.0, 18.0, 27.0, 38.0],
    elevs2=[0.5, 1.3, 4.0, 10.0, 15.0],
    terrain=True
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

#%% List of dates and available elevs
"""
ML high:
2016-05-14 ONLY GZT SURV 0.0
2016-05-15 ONLY GZT SURV 0.0
2016-05-16 ONLY GZT SURV 0.0
2016-05-31
2016-10-28
2016-09-22
2016-11-01
2016-12-01
2017-04-13
2018-03-28 NO POLARIMETRY OR RHOHV IN HTY
2019-02-06 NO POLARIMETRY OR RHOHV IN HTY
2019-05-06 NO POLARIMETRY OR RHOHV IN HTY
2019-10-17
2019-10-20
2020-03-13

ML in between radars:
2016-02-06 ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
2016-02-21 ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
2016-04-12 ONLY GZT 0.4,0.7 AND SURV 0.5 (SURV seems to not be useful because of range res 5km)
2016-05-28
2016-11-30
2016-12-26
2016-12-29
2017-01-02
2017-11-28
2017-12-31
2018-01-04
2018-01-05
2019-11-27
2019-12-09
2019-12-13
2019-12-14
2019-12-24
2019-12-25
2020-01-02
2020-01-03
2020-01-06
2020-03-17
2020-11-04
2020-11-20
2020-12-14
"""

#%% Load the selected elevations and check

# Suitable matching elevations:
# HTY: 1.5, 2.2, 3.0 and above
# GZT: 0.5, 1.3

ff1 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2017/2017-04/2017-04-13/HTY/MON_YAZ_B/2.2/MON_YAZ_B-allmoms-2.2-2017-04-13-HTY-h5netcdf.nc"
ff2 = "/automount/realpep/upload/jgiles/dmi/final_ppis/2017/2017-04/2017-04-13/GZT/VOL_A/0.5/VOL_A-allmoms-0.5-2017-04-13-GZT-h5netcdf.nc"

ds1 = xr.open_mfdataset(ff1)
ds2 = xr.open_mfdataset(ff2)

# Get PPIs into the same reference system
proj = utils.get_common_projection(ds1, ds2)

ds1 = wrl.georef.georeference(ds1, crs=proj)
ds2 = wrl.georef.georeference(ds2, crs=proj)

tsel = "2016-12-01T14"

ds1.sel(time=tsel, method="nearest")["DBZH"].wrl.vis.plot(alpha=0.5)
ax = plt.gca()
ds2.sel(time=tsel, method="nearest")["DBZH"].wrl.vis.plot(ax=ax, alpha=0.2)
ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title("DBZH "+tsel)

#%% Add beam blockage

# Define a function to calculate the beam blockage
def beam_blockage_from_radar_ds(ds,
                                sitecoords,
                                dem_resolution=3,
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
    polar_terrain = wrl.ipol.cart_to_irregular_spline(
        rastercoords_clip, rastervalues_clip, polcoords,
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

token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTc2NTg4MjQyNiwiaWF0IjoxNzYwNjk4NDI2LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.UvbHo78icjYzHBJxtW4KxgrJ97dULLiCJFeT41ylakwLkRYEc7_xlRGLbZ_gkIAwY6H0tvSDdQHUX-as2pBOSEB8QsYS7aL7RGqzupSVXUhEHFk74rLUwKNUT22ftB_iQoKkS1KNoU6-9xiGoIg2eACPEbzg9qMc6hdRCBZKUYDY8pqPkfx2PT7fSwb2-0Jj2sk6wORnE4jk6O6nXnOMEC6ZNnH8FHnLb4PzW6U8Ig1iTkNiR3MzETI1SNo5v5pdGXT_GJcnCur4RvwBZoqBtXA60LW5XBwFW5cBOt-rCv_N3mXRJerCkgje6ikqpv-L1kYeufzBvRvgNroCfGy_dQ"

ds1_pbb, ds1_cbb = beam_blockage_from_radar_ds(ds1.isel(time=0),
                                               (ds1.longitude, ds1.latitude, ds1.altitude),
                                               wradlib_token = token)

ds1 = ds1.assign({"PBB": ds1_pbb, "CBB": ds1_cbb})

ds2_pbb, ds2_cbb = beam_blockage_from_radar_ds(ds2.isel(time=0),
                                               (ds2.longitude, ds2.latitude, ds2.altitude),
                                               wradlib_token = token)

ds2 = ds2.assign({"PBB": ds2_pbb, "CBB": ds2_cbb})

# Plot beam blockage
ds1["CBB"].where(ds1["CBB"]>0.).wrl.vis.plot(alpha=0.5, vmin=0, vmax=1, cmap=mpl.cm.PuRd)
ax = plt.gca()
ds2["CBB"].where(ds2["CBB"]>0.).wrl.vis.plot(ax=ax, alpha=0.5, vmin=0, vmax=1, cmap=mpl.cm.PuRd,  xlim=(-100000, 100000), ylim=(-100000, 100000))
ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title("CBB "+tsel)

#%% Set parameters and filters

tsel = "2016-12-01T14" # for plots
vv = "DBZH"
SNRH_min = 15
RHOHV_min = 0.95
TEMP_min = 3
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
# median reflectivity in a circle 5km aroud each radar. Let's add this variable
dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsy["DBZH"]) })

# Add the additional DBZH threshold
dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                     {})
dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                     {})

#%% Generate masks
mask1, mask2, idx1, idx2, matched_timesteps = utils.find_radar_overlap_unique_NN_pairs(dsx, dsy,
                                                                    tolerance=250.,
                                                                    tolerance_time=60*4)

mask1_ref, mask2_ref, idx1_ref, idx2_ref, matched_timesteps = utils.refine_radar_overlap_unique_NN_pairs(dsx, dsy,
                                                                                      idx1, idx2,
                                                                                      vv,
                                                                    tolerance_time=60*4,
                                                                    z_tolerance=100.)

#%% Plot initial mask
dsx[vv].where(mask1).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
ax = plt.gca()
dsy[vv].where(mask2).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-100000, 100000), ylim=(-100000, 100000))

x1 = dsx.x.where(mask1).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx.y.where(mask1).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy.x.where(mask2).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy.y.where(mask2).sel(time=tsel, method="nearest").values.flatten()

# ax.scatter(x1, y1, s=1, marker="o")
# ax.scatter(x2, y2, s=1, c="r", marker="x")

ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title(vv+" "+tsel)

#%% Plot initial mask (with zoom and scatter of points)
dsx[vv].where(mask1).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
ax = plt.gca()
dsy[vv].where(mask2).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-20000, 0), ylim=(-20000, 0))

x1 = dsx.x.where(mask1).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx.y.where(mask1).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy.x.where(mask2).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy.y.where(mask2).sel(time=tsel, method="nearest").values.flatten()

ax.scatter(x1, y1, s=1, marker="o")
ax.scatter(x2, y2, s=1, c="r", marker="x")

plt.title(vv+" "+tsel)

#%% Plot refined masks
dsx[vv].where(mask1_ref).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
ax = plt.gca()
dsy[vv].where(mask2_ref).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-100000, 100000), ylim=(-100000, 100000))

x1 = dsx.x.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx.y.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy.x.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy.y.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()

# ax.scatter(x1, y1, s=1, marker="o")
# ax.scatter(x2, y2, s=1, c="r", marker="x")

ax.scatter([ds1.x[0,0], ds2.x[0,0]], [ds1.y[0,0], ds2.y[0,0]])
ax.text(ds1.x[0,0], ds1.y[0,0]-30000, "HTY")
ax.text(ds2.x[0,0], ds2.y[0,0]-30000, "GZT")

plt.title(vv+" "+tsel)

#%% Plot refined masks (with zoom and scatter of points)
dsx[vv].where(mask1_ref).sel(time=tsel, method="nearest").wrl.vis.plot(alpha=0.5, vmin=-40, vmax=50)
ax = plt.gca()
dsy[vv].where(mask2_ref).sel(time=tsel, method="nearest").wrl.vis.plot(ax=ax, alpha=0.5, vmin=-40, vmax=50, xlim=(-20000, 0), ylim=(-20000, 0))

x1 = dsx.x.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()
y1 = dsx.y.where(mask1_ref).sel(time=tsel, method="nearest").values.flatten()

x2 = dsy.x.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()
y2 = dsy.y.where(mask2_ref).sel(time=tsel, method="nearest").values.flatten()

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

# First let's get all files
HTY_files = glob.glob("/automount/realpep/upload/jgiles/dmi/final_ppis/*/*/*/HTY/*/*/*allm*")
GZT_files = glob.glob("/automount/realpep/upload/jgiles/dmi/final_ppis/*/*/*/GZT/*/*/*allm*")

# Keep elevations that we want
def get_elev(path):
    return path.split("/")[-2]

HTY_elevs = ["1.5", "2.2", "3.0"]
HTY_files = [ff for ff in HTY_files if get_elev(ff) in HTY_elevs]

GZT_elevs = ["0.4", "0.5", "0.7"]
GZT_files = [ff for ff in GZT_files if get_elev(ff) in GZT_elevs]

# Define dates
ML_high_dates = [
    "2016-05-31",
    "2016-10-28",
    "2016-09-22",
    "2016-11-01",
    "2016-12-01",
    "2017-04-13",
    "2019-10-17",
    "2019-10-20",
    "2020-03-13",
]

ML_low_dates = [
    "2016-02-06",
    "2016-02-21",
    "2016-04-12",
    "2016-05-28",
    "2016-11-30",
    "2016-12-26",
    "2016-12-29",
    "2017-01-02",
    "2017-11-28",
    "2017-12-31",
    "2018-01-04",
    "2018-01-05",
    "2019-11-27",
    "2019-12-09",
    "2019-12-13",
    "2019-12-14",
    "2019-12-24",
    "2019-12-25",
    "2020-01-02",
    "2020-01-03",
    "2020-01-06",
    "2020-03-17",
    "2020-11-04",
    "2020-11-20",
    "2020-12-14",
    ]

#%%% Start the loop for dates for rain attenuation and wet radome analyses
import time
token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTc2NTg4MjQyNiwiaWF0IjoxNzYwNjk4NDI2LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.UvbHo78icjYzHBJxtW4KxgrJ97dULLiCJFeT41ylakwLkRYEc7_xlRGLbZ_gkIAwY6H0tvSDdQHUX-as2pBOSEB8QsYS7aL7RGqzupSVXUhEHFk74rLUwKNUT22ftB_iQoKkS1KNoU6-9xiGoIg2eACPEbzg9qMc6hdRCBZKUYDY8pqPkfx2PT7fSwb2-0Jj2sk6wORnE4jk6O6nXnOMEC6ZNnH8FHnLb4PzW6U8Ig1iTkNiR3MzETI1SNo5v5pdGXT_GJcnCur4RvwBZoqBtXA60LW5XBwFW5cBOt-rCv_N3mXRJerCkgje6ikqpv-L1kYeufzBvRvgNroCfGy_dQ"

tsel = "2016-12-01T14" # for plots

tolerance = 500.
vv = "DBZH" # Used to locate and discard NaNs
SNRH_min = 15
RHOHV_min = 0.95
TEMP_min = 3
DBZH_min = 10
CBB_max = 0.05

vv_to_extract = ["DBZH", "DBZH_AC", "DBTH", "ZDR", "ZDR_EC_OC", "ZDR_EC_OC_AC", "ZDR_EC_AC",
                 "PHIDP_OC", "PHIDP_OC_SMOOTH", "PHIDP_OC_MASKED", "KDP_ML_corrected",
                 "TEMP", "Zm"] # all variables to extract from the datasets, DBZH must be the first

selected_ML_high = {vi:[] for vi in vv_to_extract}

start_time = time.time()

for date in ML_high_dates:
    print("Processing "+date)
    HTY_files0 = [ff for ff in HTY_files if date in ff]
    GZT_files0 = [ff for ff in GZT_files if date in ff]

    for HTY_file in HTY_files0:
        for GZT_file in GZT_files0:

            # Load the data
            ds1 = xr.open_mfdataset(HTY_file)
            ds2 = xr.open_mfdataset(GZT_file)

            # Get PPIs into the same reference system
            proj = utils.get_common_projection(ds1, ds2)

            ds1 = wrl.georef.georeference(ds1, crs=proj)
            ds2 = wrl.georef.georeference(ds2, crs=proj)

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
            # median reflectivity in a circle 5km aroud each radar. Let's add this variable
            dsx = dsx.assign_coords({"Zm": dsx["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsx["DBZH"]) })
            dsy = dsy.assign_coords({"Zm": dsy["DBZH"].sel(range=slice(0,5000)).compute().median(("azimuth", "range")).broadcast_like(dsy["DBZH"]) })

            # Add the additional DBZH threshold
            dsx = utils.apply_min_max_thresh(dsx, {"DBZH":DBZH_min},
                                                 {})
            dsy = utils.apply_min_max_thresh(dsy, {"DBZH":DBZH_min},
                                                 {})

            # One radar has to be the reference and the other must be the target, both below the ML
            # Let's take GZT as reference

            # We will not apply additional Zm or PHIDP conditions now so we can use
            # the extracted values for both rain atten and wet radome analyses
            dsx_tg = dsx[vv_to_extract].compute() # if we pre compute the variables that we want
            dsy_rf = dsy[vv_to_extract].compute() # we save a lot of time (~3 times faster)

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

            if mask_tg_ref.sum() == 0: continue # jump to next iteration if no pairs are found

            for vi in vv_to_extract:
                if vi in dsx_tg and vi in dsy_rf:
                    dsx_p_tg, dsy_p_rf = utils.return_unique_NN_value_pairs(dsx_tg, dsy_rf,
                                                                            mask_tg_ref, mask_rf_ref,
                                                               idx_tg_ref, idx_rf_ref,
                                                               matched_timesteps, vi)

                    selected_ML_high[vi].append( (dsx_p_tg.copy(), dsy_p_rf.copy()) )
                else:
                    print(vi+" not found in one of the ds, filling with NaNs")
                    selected_ML_high[vi].append( (np.full_like(selected_ML_high['DBZH'][-1][0], fill_value=np.nan),
                                            np.full_like(selected_ML_high['DBZH'][-1][1], fill_value=np.nan))
                                          )

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%%% Plot boxplot of delta DBZH/ZDR vs target PHI (rain attenuation)
phi = "PHIDP_OC"
dbzh = "DBZH"

# build delta DBZH
delta_dbzh = np.concat([ (d1-d2).flatten() for d1,d2 in selected_ML_high[dbzh] ])

tg_phi = np.concat([ d1.flatten() for d1,d2 in selected_ML_high[phi] ])

# filter by valid PHI values
delta_dbzh = delta_dbzh[np.isfinite(tg_phi)]
tg_phi = tg_phi[np.isfinite(tg_phi)]

# Box plots like in the paper
# Define bins
bins = np.arange(0, 18, 1)  # 0,1,2,3,4,5
bin_centers = bins[:-1] + 0.5

# Digitize tg_phi into bins
bin_indices = np.digitize(tg_phi, bins) - 1

# Prepare data for boxplot
box_data = [delta_dbzh[bin_indices == i] for i in range(len(bins) - 1)]

# Compute counts per bin
counts = [len(vals) for vals in box_data]

# Plot
plt.figure(figsize=(8, 5))
plt.boxplot(box_data, positions=bin_centers, widths=0.6)
plt.xlabel(phi+" (binned, 1° intervals)")
plt.ylabel("delta "+dbzh)
plt.title("Boxplots of delta "+dbzh+" vs "+phi+" bins")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(bin_centers, [f"{b}-{b+1}" for b in bins[:-1]])

# Add counts above x-tick labels (inside the plot area)
for x, n in zip(bin_centers, counts):
    plt.text(x, plt.ylim()[0] + 0.01 * (plt.ylim()[1] - plt.ylim()[0]),  # 5% above bottom
             f"{n}", ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout()
plt.show()
