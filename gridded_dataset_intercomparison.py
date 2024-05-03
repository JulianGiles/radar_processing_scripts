#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:25:08 2024

@author: jgiles

Script for comparing precipitation datasets at different scales.
"""

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
import wradlib as wrl
import glob
import regionmask as rm
from cdo import Cdo
import xesmf as xe
import hvplot.xarray
import holoviews as hv

import warnings
#ignore by message
# warnings.filterwarnings("ignore", message="Default reduction dimension will be changed to the grouped dimension")
# warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
# warnings.filterwarnings("ignore", message="Setting the 'color' property will override the edgecolor or facecolor properties.")
# warnings.filterwarnings("ignore", category=FutureWarning)

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


proj = ccrs.PlateCarree(central_longitude=0.0)
proj = ccrs.Mercator(central_longitude=0.0)

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

#%% YEARLY analysis

#%%% Load yearly datasets

loadpath_yearly = "/automount/agradar/jgiles/gridded_data/yearly/"
paths_yearly = {
    "IMERG-V07B-monthly": loadpath_yearly+"IMERG-V07B-monthly/IMERG-V07B-monthly_precipitation_yearlysum_2000-2023.nc",
    "IMERG-V06B-monthly": loadpath_yearly+"IMERG-V06B-monthly/IMERG-V06B-monthly_precipitation_yearlysum_2000-2021.nc",
    # "IMERG-V07B-30min": loadpath_yearly+, 
    # "IMERG-V06B-30min": loadpath_yearly+, 
    "CMORPH-daily": loadpath_yearly+"CMORPH-daily/CMORPH-daily_precipitation_yearlysum_1998-2023.nc",
    "TSMP-old": loadpath_yearly+"TSMP-old/TSMP-old_precipitation_yearlysum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_yearly+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_yearlysum_2000-2022.nc",
    "ERA5-monthly": loadpath_yearly+"ERA5-monthly/ERA5-monthly_precipitation_yearlysum_1979-2020.nc",
    # "ERA5-hourly": loadpath_yearly+,
    "RADKLIM": loadpath_yearly+"RADKLIM/RADKLIM_precipitation_yearlysum_2001-2022.nc",
    "RADOLAN": loadpath_yearly+"RADOLAN/RADOLAN_precipitation_yearlysum_2006-2022.nc",
    "EURADCLIM": loadpath_yearly+"EURADCLIM/EURADCLIM_precipitation_yearlysum_2013-2020.nc",
    "GPCC-monthly": loadpath_yearly+"GPCC-monthly/GPCC-monthly_precipitation_yearlysum_2001-2020.nc",
    # "GPCC-daily": loadpath_yearly+"GPCC-daily/GPCC-daily_precipitation_yearlysum_2000-2020.nc",
    "GPROF": loadpath_yearly+"GPROF/GPROF_precipitation_yearlysum_2014-2023.nc",
    "HYRAS": loadpath_yearly+"HYRAS/HYRAS_precipitation_yearlysum_1931-2020.nc", 
    }

data_yearlysum = {}

# reload the datasets
for dsname in paths_yearly.keys():
    data_yearlysum[dsname] = xr.open_dataset(paths_yearly[dsname])

# Special tweaks
# RADOLAN GRID AND CRS
lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
data_yearlysum["RADOLAN"] = data_yearlysum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
data_yearlysum["RADOLAN"] = data_yearlysum["RADOLAN"].assign(crs=data_yearlysum['RADKLIM'].crs[0])
data_yearlysum["RADOLAN"].attrs["grid_mapping"] = "crs"
data_yearlysum["RADOLAN"].lon.attrs = data_yearlysum["RADKLIM"].lon.attrs
data_yearlysum["RADOLAN"].lat.attrs = data_yearlysum["RADKLIM"].lat.attrs

# EURADCLIM coords
data_yearlysum["EURADCLIM"] = data_yearlysum["EURADCLIM"].set_coords(("lon", "lat"))

# Shift HYRAS and EURADCLIM timeaxis
data_yearlysum["EURADCLIM"] = data_yearlysum["EURADCLIM"].resample({"time":"YS"}).first()
data_yearlysum["HYRAS"] = data_yearlysum["HYRAS"].resample({"time":"YS"}).mean()

# Convert all non datetime axes (cf Julian calendars) into datetime 
for dsname in paths_yearly.keys():
    try:
        data_yearlysum[dsname]["time"] = data_yearlysum[dsname].indexes['time'].to_datetimeindex()
    except:
        pass


#%%% Area means
data_to_avg = data_yearlysum # select which data to average (yearly, monthly, daily...)

region ="Germany"
rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10

data_avgreg = {}
# Means over region
to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():
    
    if dsname in ["RADOLAN", "RADKLIM", "HYRAS", "EURADCLIM"]:
        # these datasets come in equal-pixel-sized grids, so we only need to apply the average over the region
        mask = rmcountries[[region]].mask(data_to_avg[dsname])
        data_avgreg[dsname] = data_to_avg[dsname].where(mask.notnull()).mean(("x", "y")).compute()

    if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly", "CMORPH-daily", "ERA5-monthly", 
                  "GPCC-monthly", "GPCC-daily", "GPROF"]:
        # these datasets come in regular lat-lon grids, so we need to average over the region considering the area weights
        variables_to_include = [vv for vv in data_to_avg[dsname].data_vars \
                                if "lonv" not in data_to_avg[dsname][vv].dims \
                                if "latv" not in data_to_avg[dsname][vv].dims \
                                if "nv" not in data_to_avg[dsname][vv].dims]
        mask = rmcountries[[region]].mask(data_to_avg[dsname])
        if dsname in ["ERA5-monthly"]:
            data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask.notnull()), 
                                                          lon_name="longitude", lat_name="latitude").compute()
        else:
            data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask.notnull()), 
                                                          lon_name="lon", lat_name="lat").compute()

    if dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
        # we need to unrotate the TSMP grid and then average over the region considering the area weights

        grid_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
        regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]), grid_out, "conservative")
        to_add[dsname+"-EURregLonLat01deg"] = regridder(data_to_avg[dsname])
        
        mask = rmcountries[[region]].mask(to_add[dsname+"-EURregLonLat01deg"])
    
        data_avgreg[dsname] = utils.calc_spatial_mean(to_add[dsname+"-EURregLonLat01deg"].where(mask.notnull()), 
                                                      lon_name="lon", lat_name="lat").compute()
        
# add the rotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}


#%%% Simple map plot
rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10
mask = rmcountries[["Germany"]].mask(data_yearlysum["EURADCLIM"])
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_yearlysum["EURADCLIM"]["Precip"][0].where(mask.notnull(), drop=True).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# ax1.set_extent([float(a) for a in lonlat_limits])
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title("EURADCLIM")

#%%% Interannual variability area-means plot
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph"]

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

for dsname in data_avgreg.keys():
    if dsname in dsignore:
        continue
    plotted = False
    for vv in var_names:
        if vv in data_avgreg[dsname].data_vars:
            color=None
            marker=None
            if dsname in dsref:
                color = "black"
                marker = "o"
            try:
                plt.plot(data_avgreg[dsname]['time'], data_avgreg[dsname][vv], label=dsname, c=color, marker=marker)
            except TypeError:
                # try to change the time coord to datetime format
                plt.plot(data_avgreg[dsname].indexes['time'].to_datetimeindex(), data_avgreg[dsname][vv], label=dsname)
            plotted = True
    if not plotted:
        raise Warning("Nothing plotted for "+dsname)

plt.legend(ncols=3, fontsize=7)
plt.title("Area-mean annual total precip "+region+" [mm]")
plt.xlim(datetime(2000,1,1), datetime(2020,1,1))
# plt.xlim(2000, 2020)
plt.grid()

#%%% Interannual variability area-means plot (interactive html)

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph"]

dsignore = [] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)


hvplots = []
for dsname in data_avgreg.keys():
    if dsname in dsignore:
        continue
    plotted = False
    for vv in var_names:
        if vv in data_avgreg[dsname].data_vars:
            if dsname in dsref:
                color = "black"
                lw = 4
            
                hvplots.append(
                                data_avgreg[dsname][vv].hvplot.line(x='time', label=dsname).opts(color=color, line_width=lw, show_legend=True, muted_alpha=0)
                    )
            
            else:
                hvplots.append(
                                data_avgreg[dsname][vv].hvplot.line(x='time', label=dsname).opts(show_legend=True, muted_alpha=0)
                    )

            plotted = True
    if not plotted:
        raise Warning("Nothing plotted for "+dsname)

layout = hvplots[0]
for nplot in np.arange(1, len(hvplots)):
    layout = layout * hvplots[nplot]

layout.opts(title="Area-mean annual total precip "+region+" [mm]", xlabel="Time", show_grid=True, legend_position='right',
            height=600, width=1200)

# Save to HTML file
hv.save(layout, '/user/jgiles/interactive_plot.html')

#%% REGRIDDING TESTS
#%%% Simple map plot TSMP original
rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10
mask = rmcountries[["Germany"]].mask(data_yearlysum["TSMP-DETECT-Baseline"])
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_yearlysum["TSMP-DETECT-Baseline"]["TOT_PREC"][0].where(mask.notnull(), drop=True).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# ax1.set_extent([float(a) for a in lonlat_limits])
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title("TSMP-DETECT-Baseline original")

#%%% Simple map plot TSMP regridded with xESMF
ds_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
regridder = xe.Regridder(data_yearlysum["TSMP-DETECT-Baseline"].cf.add_bounds(["lon", "lat"]), ds_out, "conservative")
dr_out = regridder(data_yearlysum["TSMP-DETECT-Baseline"]["TOT_PREC"])

rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10
mask = rmcountries[["Germany"]].mask(dr_out)
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = dr_out[0].where(mask.notnull(), drop=True).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# ax1.set_extent([float(a) for a in lonlat_limits])
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title("TSMP-DETECT-Baseline unrotated with xESMF")

#%%% Simple map plot TSMP regridded with CDO
Cdo().remap("/automount/agradar/jgiles/detect_grid_specification/griddes/griddes_EURregLonLat01deg_1204x548_inclbrz_v2.txt",
            "/automount/agradar/jgiles/detect_grid_specification/rmp_weights/rmp_con_EUR-11_TO_EURregLonLat01deg_v2.nc",
            input="/automount/agradar/jgiles/gridded_data/yearly/TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_yearlysum_2000-2022.nc",
            output="/automount/agradar/jgiles/gridded_data/yearly/TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_yearlysum_2000-2022_EURregLonLat01deg.nc")
TSMP_DETECT_Baseline_precipitation_yearlysum = xr.open_dataset("/automount/agradar/jgiles/gridded_data/yearly/TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_yearlysum_2000-2022_EURregLonLat01deg.nc")
rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10
mask = rmcountries[["Germany"]].mask(TSMP_DETECT_Baseline_precipitation_yearlysum)
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = TSMP_DETECT_Baseline_precipitation_yearlysum["TOT_PREC"][0].where(mask.notnull(), drop=True).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# ax1.set_extent([float(a) for a in lonlat_limits])
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title("TSMP-DETECT-Baseline unrotated with CDO")
