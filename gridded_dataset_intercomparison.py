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
import copy
import time

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
    "GPCC-monthly": loadpath_yearly+"GPCC-monthly/GPCC-monthly_precipitation_yearlysum_1991-2020.nc",
    # "GPCC-daily": loadpath_yearly+"GPCC-daily/GPCC-daily_precipitation_yearlysum_2000-2020.nc",
    "GPROF": loadpath_yearly+"GPROF/GPROF_precipitation_yearlysum_2014-2023.nc",
    "HYRAS": loadpath_yearly+"HYRAS/HYRAS_precipitation_yearlysum_1931-2020.nc", 
    "E-OBS": loadpath_yearly+"E-OBS/E-OBS_precipitation_yearlysum_1950-2023.nc", 
    "CPC": loadpath_yearly+"CPC/CPC_precipitation_yearlysum_1979-2024.nc", 
    }

data_yearlysum = {}

# reload the datasets
print("Loading yearly datasets ...")
for dsname in paths_yearly.keys():
    print("... "+dsname)
    data_yearlysum[dsname] = xr.open_dataset(paths_yearly[dsname])

# Special tweaks
# RADOLAN GRID AND CRS
if "RADOLAN" in data_yearlysum.keys():
    lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
    data_yearlysum["RADOLAN"] = data_yearlysum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
    data_yearlysum["RADOLAN"] = data_yearlysum["RADOLAN"].assign(crs=data_yearlysum['RADKLIM'].crs[0])
    data_yearlysum["RADOLAN"].attrs["grid_mapping"] = "crs"
    data_yearlysum["RADOLAN"].lon.attrs = data_yearlysum["RADKLIM"].lon.attrs
    data_yearlysum["RADOLAN"].lat.attrs = data_yearlysum["RADKLIM"].lat.attrs

# EURADCLIM coords
if "EURADCLIM" in data_yearlysum.keys():
    data_yearlysum["EURADCLIM"] = data_yearlysum["EURADCLIM"].set_coords(("lon", "lat"))

# Shift HYRAS and EURADCLIM timeaxis
if "EURADCLIM" in data_yearlysum.keys():
    data_yearlysum["EURADCLIM"] = data_yearlysum["EURADCLIM"].resample({"time":"YS"}).first()
if "HYRAS" in data_yearlysum.keys():
    data_yearlysum["HYRAS"] = data_yearlysum["HYRAS"].resample({"time":"YS"}).mean()

# Convert all non datetime axes (cf Julian calendars) into datetime 
for dsname in paths_yearly.keys():
    try:
        data_yearlysum[dsname]["time"] = data_yearlysum[dsname].indexes['time'].to_datetimeindex()
    except:
        pass

# Special selections for incomplete extreme years
# IMERG
if "IMERG-V07B-monthly" in data_yearlysum.keys():
    data_yearlysum["IMERG-V07B-monthly"] = data_yearlysum["IMERG-V07B-monthly"].loc[{"time":slice("2001", "2022")}]
if "IMERG-V06B-monthly" in data_yearlysum.keys():
    data_yearlysum["IMERG-V06B-monthly"] = data_yearlysum["IMERG-V06B-monthly"].loc[{"time":slice("2001", "2020")}]
# CMORPH
if "CMORPH-daily" in data_yearlysum.keys():
    data_yearlysum["CMORPH-daily"] = data_yearlysum["CMORPH-daily"].loc[{"time":slice("1998", "2022")}]
# GPROF
if "GPROF" in data_yearlysum.keys():
    data_yearlysum["GPROF"] = data_yearlysum["GPROF"].loc[{"time":slice("2015", "2022")}]
# CPC
if "CPC" in data_yearlysum.keys():
    data_yearlysum["CPC"] = data_yearlysum["CPC"].loc[{"time":slice("1979", "2023")}]

colors = {
    "IMERG-V07B-monthly": "#FF6347", # Tomato
    "IMERG-V06B-monthly": "crimson", # crimson
    "CMORPH-daily": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", # DodgerBlue
    "ERA5-monthly": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

#%%% Regional averages
#%%%% Calculate area means (regional averages)
data_to_avg = data_yearlysum # select which data to average (yearly, monthly, daily...)

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland", 
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra", 
         "Monaco", "Czechia", "Slovakia", "Hungary", "Slovenia", "Romania"]#"land"
region = "Germany"
region_name = "Germany" # "Europe_EURADCLIM" # name for plots
mask = utils.get_regionmask(region)
TSMP_nudge_margin = 13 # number of gridpoints to mask out the relaxation zone at the margins

# TSMP-case: we make a specific mask to cut out the edge of the european domain + country
dsname = "TSMP-DETECT-Baseline"
mask_TSMP_nudge = False
if dsname in data_to_avg.keys():
    mask_TSMP_nudge = True # This will be used later as a trigger for this extra mask
    lon_bot = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lon.values
    lat_bot = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lat.values
    lon_top = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lon.values
    lat_top = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lat.values
    lon_right = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lon.values
    lat_right = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lat.values
    lon_left = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lon.values
    lat_left = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lat.values
    
    lon_tsmp_edge = np.concatenate((lon_bot, lon_right, lon_top[::-1], lon_left[::-1]))
    lat_tsmp_edge = np.concatenate((lat_bot, lat_right, lat_top[::-1], lat_left[::-1]))
    
    lonlat_tsmp_edge = list(zip(lon_tsmp_edge, lat_tsmp_edge))
    
    TSMP_no_nudge = rm.Regions([ lonlat_tsmp_edge ], names=["TSMP_no_nudge"], abbrevs=["TSMP_NE"], name="TSMP")
    # I did not find a way to directly combine this custom region with a predefined country region. I will 
    # have to just apply the masks consecutively

data_avgreg = {}
# Means over region
print("Calculating means over "+region_name)
to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():
    print("... "+dsname)

    if dsname in ["RADOLAN", "RADKLIM", "HYRAS", "EURADCLIM"]:
        # these datasets come in equal-pixel-sized grids, so we only need to apply the average over the region
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        data_avgreg[dsname] = data_to_avg[dsname].where(mask0.notnull()).mean(("x", "y")).compute()

    if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly", "CMORPH-daily", "ERA5-monthly", 
                  "GPCC-monthly", "GPCC-daily", "GPROF", "E-OBS", "CPC"]:
        # these datasets come in regular lat-lon grids, so we need to average over the region considering the area weights
        variables_to_include = [vv for vv in data_to_avg[dsname].data_vars \
                                if "lonv" not in data_to_avg[dsname][vv].dims \
                                if "latv" not in data_to_avg[dsname][vv].dims \
                                if "nv" not in data_to_avg[dsname][vv].dims]
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        if dsname in ["ERA5-monthly", "E-OBS"]:
            data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                          lon_name="longitude", lat_name="latitude").compute()
        else:
            data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                          lon_name="lon", lat_name="lat").compute()

    if dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
        # we need to unrotate the TSMP grid and then average over the region considering the area weights

        grid_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
        regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]), grid_out, "conservative")
        to_add[dsname+"-EURregLonLat01deg"] = regridder(data_to_avg[dsname])
        
        mask0 = mask.mask(to_add[dsname+"-EURregLonLat01deg"])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(to_add[dsname+"-EURregLonLat01deg"]).where(mask0.notnull())

        data_avgreg[dsname] = utils.calc_spatial_mean(to_add[dsname+"-EURregLonLat01deg"].where(mask0.notnull()), 
                                                      lon_name="lon", lat_name="lat").compute()
        
# add the rotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}
data_yearlysum = data_to_avg.copy()

#%%%% Simple map plot
dsname = "GPCC-monthly"
vname = "precip"
mask = utils.get_regionmask(region)
mask0 = mask.mask(data_yearlysum[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(data_yearlysum[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_yearlysum[dsname][vname][0].where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
if mask_TSMP_nudge: ax1.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(dsname)

#%%%% Simple map plot (for number of stations per gridcell)
dsname = "GPCC-monthly"
vname = "numgauge"
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Turkey/"

for yy in np.arange(2000,2021):
    ysel = str(yy)
    mask = utils.get_regionmask(region)
    mask0 = mask.mask(data_yearlysum[dsname])
    dropna = True
    if mask_TSMP_nudge: 
        mask0 = TSMP_no_nudge.mask(data_yearlysum[dsname]).where(mask0.notnull())
        dropna=False
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    cmap1 = copy.copy(plt.cm.Blues)
    cmap1.set_under("lightgray")
    plot = (data_yearlysum[dsname][vname].sel(time=ysel)/12).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", 
                                            levels=3, cmap=cmap1, vmin=1, vmax=3, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                             cbar_kwargs={'label': "", 'shrink':0.88})
    if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    plt.title(dsname+" number of stations per gridcell "+ysel)
    # save figure
    savepath_yy = savepath+ysel+"/"
    if not os.path.exists(savepath_yy):
        os.makedirs(savepath_yy)
    filename = "numgauge_"+region_name+"_"+dsname+"_"+ysel+".png"
    plt.savefig(savepath_yy+filename, bbox_inches="tight")
    # plt.show()

#%%%% Interannual variability area-means plot
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

colors = {
    "IMERG-V07B-monthly": "#FF6347", # Tomato
    "IMERG-V06B-monthly": "crimson", # crimson
    "CMORPH-daily": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", # DodgerBlue
    "ERA5-monthly": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

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
            else: 
                color = colors[dsname]
            try:
                plt.plot(data_avgreg[dsname]['time'], data_avgreg[dsname][vv], label=dsname, c=color, marker=marker)
            except TypeError:
                # try to change the time coord to datetime format
                plt.plot(data_avgreg[dsname].indexes['time'].to_datetimeindex(), data_avgreg[dsname][vv], label=dsname, c=color, marker=marker)
            plotted = True
    if not plotted:
        raise Warning("Nothing plotted for "+dsname)

plt.legend(ncols=3, fontsize=7)
plt.title("Area-mean annual total precip "+region_name+" [mm]")
plt.xlim(datetime(2000,1,1), datetime(2020,1,1))
# plt.xlim(2000, 2020)
plt.grid()

#%%%% Interannual variability area-means plot (interactive html)

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

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

layout.opts(title="Area-mean annual total precip "+region_name+" [mm]", xlabel="Time", show_grid=True, legend_position='right',
            height=600, width=1200)

# Save to HTML file
hv.save(layout, "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual/"+region_name+"/lineplots/area_mean_annual_total_precip_"+region_name+".html")

#%%%% Plot the period from each dataset
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

yticks = []
yticklabels = []

for dsn,dsname in enumerate(data_avgreg.keys()):
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
            else: 
                color = colors[dsname]
            try:
                plt.plot(data_avgreg[dsname]['time'], data_avgreg[dsname][vv]*0+dsn+1, label=dsname, c=color, marker=marker)
            except TypeError:
                # try to change the time coord to datetime format
                plt.plot(data_avgreg[dsname].indexes['time'].to_datetimeindex(), data_avgreg[dsname][vv]*0+dsn+1, label=dsname, c=color, marker=marker)
            plotted = True
            yticks.append(dsn+1)
            yticklabels.append(dsname)
    if not plotted:
        raise Warning("Nothing plotted for "+dsname)

plt.title("Period from each dataset")
plt.gca().set_xticks([str(xx) for xx in np.arange(1980, 2024)], minor=True)
plt.xlim(datetime(1980,1,1), datetime(2024,1,1))
plt.yticks(yticks, yticklabels) # set yticks and labels
plt.grid()

#%%% BIAS and ERRORS
#%%%% Bias (absolute and relative) calculation from regional averages
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

dsignore = [] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference

data_to_bias = data_avgreg

data_bias = {}
data_bias_relative = {}
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    data_bias[dsname] = data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref]
                    data_bias_relative[dsname] = (data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref])/data_to_bias[dsref[0]][vvref]*100
                    break

#%%%% Region-averages bias bar plot
# Calculate bar width based on the number of data arrays
bar_width = 0.8 / len(data_bias)

# Get time values
time_values_ref = data_to_bias[dsref[0]]['time']

#%%%%% Plotting each DataArray in the dictionary
plt.figure(figsize=(20, 6))  # Adjust figure size as needed
for idx, (key, value) in enumerate(data_bias.items()):
    value_padded = value.broadcast_like(data_to_bias[dsref[0]])
    time_values = value_padded['time']
    bar_positions = np.arange(len(time_values)) + idx * bar_width
    plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])

plt.xlabel('Time')
plt.ylabel(value.attrs['units'])
plt.title("Area-mean annual total precip BIAS with respect to "+dsref[0]+" "+region_name)
plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%%%% Plotting each DataArray in the dictionary (same but for relative bias)
plt.figure(figsize=(20, 6))  # Adjust figure size as needed
for idx, (key, value) in enumerate(data_bias_relative.items()):
    value_padded = value.broadcast_like(data_to_bias[dsref[0]])
    time_values = value_padded['time']
    bar_positions = np.arange(len(time_values)) + idx * bar_width
    plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])

plt.xlabel('Time')
plt.ylabel("%")
plt.title("Area-mean annual total precip RELATIVE BIAS with respect to "+dsref[0]+" "+region_name)
plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%%% Relative bias and errors calculation (at gridpoint level, not with the area means)
# First we need to transform EURADCLIM, RADKLIM, RADOLAN and HYRAS to regular grids
# We use the DETECT 1 km grid for this
to_add = {} # dictionary to add regridded versions
for dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
    if dsname not in data_yearlysum: continue
    print("Regridding "+dsname+" ...")

    grid_out = xe.util.cf_grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    grid_out = xe.util.grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    # # I tried to use dask for the weight generation to avoid memory crash
    # # but I did not manage to make it work: https://xesmf.readthedocs.io/en/latest/notebooks/Dask.html

    # grid_out = grid_out.chunk({"x": 50, "y": 50, "x_b": 50, "y_b": 50,})

    # # we then try parallel regridding: slower but less memory-intensive (this takes forever)
    # regridder = xe.Regridder(data_yearlysum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True)
    # to_add[dsname+"-EURregLonLat001deg"] = regridder(data_to_avg[dsname])
    # regridder.to_netcdf() # we save the weights
    # # to reuse the weigths:
    # xe.Regridder(data_yearlysum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True, weights="/path/to/weights") #!!! Can I use CDO weights here?
    # Cdo().gencon()
    # cdo gencon,/automount/ags/jgiles/IMERG_V06B/global_monthly/griddes.txt -setgrid,/automount/agradar/jgiles/TSMP/griddes_mod.txt /automount/agradar/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020.nc weights_to_IMERG.nc

    # Instead, just regrid to the reference dataset grid (this is fast)

    regridder = xe.Regridder(data_yearlysum[dsname].cf.add_bounds(["lon", "lat"]), data_yearlysum[dsref[0]], "conservative")
    to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_yearlysum[dsname])
    
# add the regridded datasets to the original dictionary
data_yearlysum = {**data_yearlysum, **to_add}
    
# Compute the biases
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline'] # datasets to ignore (because we want the regridded version)
data_to_bias = copy.copy(data_yearlysum)

to_add = {} # dictionary to add regridded versions

data_bias_map = {} # maps of yearly biases
data_bias_relative_map = {} # maps of yearly relative biases
data_abs_error_map = {} # maps of yearly absolute errors
data_bias_relative_gp = {} # yearly relative biases on a gridpoint basis (sum of gridpoint biases divided by sum of reference-data values)
data_mean_abs_error_gp = {} # yearly MAE on a gridpoint basis (sum of gridpoint abs errors divided by number of data values)
data_norm_mean_abs_error_gp = {} # yearly NMAE on a gridpoint basis (sum of gridpoint abs errors divided by sum of reference-data values)
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    print("Processing "+dsname+" ...")
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    if dsref[0]+"-grid" not in dsname: # if no regridded already, do it now
                        if "longitude" in data_to_bias[dsname].coords or "latitude" in data_to_bias[dsname].coords:
                            # if the names of the coords are longitude and latitude, change them to lon, lat
                            data_to_bias[dsname] = data_to_bias[dsname].rename({"longitude":"lon", "latitude":"lat"})
                        
                        if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bnds", "lat_bnds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])
                            
                        if dsname in ["CMORPH-daily"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bounds", "lat_bounds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        if dsname in ["GPROF", "TSMP-old-EURregLonLat01deg", "TSMP-DETECT-Baseline-EURregLonLat01deg"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        # regridder = xe.Regridder(data_to_bias[dsname].cf.add_bounds(["lon", "lat"]), data_to_bias[dsref[0]], "conservative")

                        regridder = xe.Regridder(data_to_bias[dsname], data_to_bias[dsref[0]], "conservative")

                        data0 = regridder(data_to_bias[dsname][vv])
                        to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_to_bias[dsname][vv])
                    else:
                        data0 = data_to_bias[dsname][vv]

                    data0 = data0.where(data0>0)
                    dataref = data_to_bias[dsref[0]][vvref]
                    
                    mask0 = mask.mask(data_to_bias[dsref[0]])
                    if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_bias[dsref[0]]).where(mask0.notnull())

                    data_bias_map[dsname] = ( data0 - dataref ).compute()
                    data_bias_relative_map[dsname] = ( data_bias_map[dsname] / dataref ).compute() *100
                    data_bias_map_masked = data_bias_map[dsname].where(mask0.notnull())
                    data_bias_relative_map_masked = data_bias_relative_map[dsname].where(mask0.notnull())
                    data_abs_error_map[dsname] = abs(data_bias_map[dsname])
                    data_abs_error_map_masked = data_abs_error_map[dsname].where(mask0.notnull())
                    
                    data_bias_relative_gp[dsname] = utils.calc_spatial_integral(data_bias_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100
                    
                    data_norm_mean_abs_error_gp[dsname] = utils.calc_spatial_integral(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100

                    data_mean_abs_error_gp[dsname] = utils.calc_spatial_mean(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute()
                    
                    break

# add the regridded datasets to the original dictionary
data_yearlysum = {**data_yearlysum, **to_add}

#%%%% Relative bias and error plots
#%%%%% Simple map plot
# region = "Germany" #"land" 
to_plot = data_bias_map
dsname = "TSMP-DETECT-Baseline-EURregLonLat01deg"
title = "BIAS"
yearsel = "2016"
cbarlabel = "mm" # mm
vmin = -250
vmax = 250
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
mask = utils.get_regionmask(region)
mask0 = mask.mask(to_plot[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":yearsel}].where(mask0.notnull(), drop=dropna).loc[{"lon":lonlat_slice[0], 
                                                                                      "lat":lonlat_slice[1]}].plot(x="lon", 
                                                                                                                   y="lat", 
                                                                                                                   cmap="RdBu_r", 
                                                                                    vmin=vmin, vmax=vmax, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+yearsel+"\n"+dsname+"\n "+region_name+" Ref.: "+dsref[0])

#%%%%% Simple map plot (loop)
# Like previous but for saving all plots
# region = "Germany" #"land" 
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/annual/"+region_name+"/"
period = np.arange(2000,2024)
to_plot_dict = [
            (data_bias_map, "BIAS", "mm", -250, 250),
            (data_bias_relative_map, "RELATIVE BIAS", "%", -75, 75),
           ]
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
for to_plot, title, cbarlabel, vmin, vmax in to_plot_dict:
    print("Plotting "+title)
    for dsname in to_plot.keys():
        print("... "+dsname)
        dsname_short = dsname.split("_")[0]
        mask = utils.get_regionmask(region)
        mask0 = mask.mask(to_plot[dsname])
        dropna = True
        if mask_TSMP_nudge: 
            mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
            dropna=False
        for yearsel in period:
            try:
                plt.close()
                # f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
                plot = to_plot[dsname].loc[{"time":str(yearsel)}].where(mask0.notnull(), 
                                                                        drop=dropna).loc[{"lon":lonlat_slice[0],
                                                                                          "lat":lonlat_slice[1]}].plot(x="lon",
                                                                                                                       y="lat",
                                                                                                                       cmap="RdBu_r", 
                                                                                                    vmin=vmin, vmax=vmax, 
                                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
                if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
                plot.axes.coastlines(alpha=0.7)
                plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
                plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
                plt.title(title+" "+str(yearsel)+"\n"+dsname_short+"\n "+region_name+" Ref.: "+dsref[0])
                
                # save figure
                savepath_yy = savepath+str(yearsel)+"/"
                if not os.path.exists(savepath_yy):
                    os.makedirs(savepath_yy)
                filename = "_".join([title.lower().replace(" ","_"), region_name, dsname_short,dsref[0],str(yearsel)])+".png"
                plt.savefig(savepath_yy+filename, bbox_inches="tight")
                plt.close()
            except KeyError:
                continue

#%%%%% Box plots of BIAS and ERRORS
# the box plots are made up of the yearly bias or error values, and the datasets are ordered according to their median
to_plot0 = data_bias_relative_gp.copy() # data_mean_abs_error_gp # data_bias_relative_gp # data_norm_mean_abs_error_gp
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual/"+region_name+"/boxplots/relative_bias/"
savefilename = "boxplot_relative_bias_yearly"
title = "Relative bias (yearly values) "+region_name+". Ref.: "+dsref[0]
ylabel = "%" # % # mm
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
tsel = [
        slice(None, None), # if I want to consider only certain period. Otherwise set to (None, None). Multiple options possible
        slice("2001-01-01", "2020-01-01"), 
        slice("2006-01-01", "2020-01-01"), 
        slice("2013-01-01", "2020-01-01"), 
        slice("2015-01-01", "2020-01-01")
        ] 
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

for tseln in tsel:
    to_plot = to_plot0.copy()
    savefilenamen = copy.deepcopy(savefilename)
    titlen = copy.deepcopy(title)
    
    if tseln.start is not None and tseln.stop is not None: # add specific period to title
        titlen = titlen+". "+tseln.start+" - "+tseln.stop
        savefilenamen = savefilenamen+"_"+tseln.start+"-"+tseln.stop
        
        if ignore_incomplete:
            for key in to_plot.copy().keys():
                if not (to_plot[key].time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                        to_plot[key].time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                    del(to_plot[key])
    
    # Initialize a figure and axis
    plt.figure(figsize=(1.25*(len(to_plot.keys())-len(dsignore)), 6))
    ax = plt.subplot(111)
    
    # Create a list to hold the data arrays
    plotted_arrays = []
    plotted_arrays_lengths = []
    
    # Order according to median:
    to_plot = dict(sorted(to_plot.items(), key=lambda item: item[1].sel(time=tseln).median()))
    
    # Iterate over the datasets in the dictionary
    for key, value in to_plot.items():
        if key not in dsignore:
            # Plot a box plot for each dataset
            value = value.sel(time=tseln)
            plotted_arrays.append(value.values) # values of each box
            plotted_arrays_lengths.append(len(value)) # number of values in each box
            ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6, 
                       patch_artist=True, boxprops=dict(facecolor='#b6d6e3'),
                       medianprops=dict(color="#20788e", lw=2))
    
    # Set x-axis ticks and labels with dataset names
    ax.set_xticks(range(1, len(plotted_arrays) + 1))
    ax.set_xticklabels([dsname.split("_")[0] if "_" in dsname 
                        else "-".join(dsname.split("-")[:-1]) if "EURreg" in dsname 
                        else dsname 
                        for dsname in 
                        [ds for ds in to_plot.keys() if ds not in dsignore]
                        ],
                       rotation=45, fontsize=15)
    ax.xaxis.label.set_size(15)     # change xlabel size
    ax.yaxis.label.set_size(15)     # change ylabel size
    
    ax.tick_params(axis='x', labelsize=15) # change xtick label size
    ax.tick_params(axis='y', labelsize=15) # change xtick label size
    
    # Make a secondary x axis to display the number of values in each box
    ax2 = ax.secondary_xaxis('top')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("top")
    
    ax2.set_xticks(range(1, len(plotted_arrays) + 1))
    ax2.set_xticklabels(plotted_arrays_lengths)
    ax2.set_xlabel('Number of years', fontsize= 15)
    
    # Set labels and title
    #ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(titlen, fontsize=20)
    
    # plot a reference line at zero
    plt.hlines(y=0, xmin=0, xmax=len(plotted_arrays)+1, colors='black', lw=2, zorder=0)
    plt.xlim(0.5, len(plotted_arrays) + 0.5)
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(savepath+savefilenamen+".png", bbox_inches="tight")
    plt.show()

#%%%%% Taylor diagram
# The Taylor diagram can be done by computing the stats over all gridpoints and all timesteps (spatiotemporal)
# or only doing the stats over space or time separately (for these, either temporal or spatial averages must be done first)
    
#%%%%%% Compute stats
import skill_metrics as sm
# https://github.com/PeterRochford/SkillMetrics/blob/master/Examples/taylor10.py#L123
# I cannot use skill_metrics to calculate the stats because they do not filter out 
# nan values (because of the masks) so the result is erroneous. They also do not handle weighted arrays.

mode = "" # if "spatial" then average in time and compute the diagram in space. Viceversa for "temporal"
dsref = ["GPCC-monthly"]
data_to_stat = data_yearlysum

# choose common period (only datasets that cover the whole period are included)
tslice = slice("2015","2020") # this covers all
# tslice = slice("2013","2020") # this excludes GPROF
# tslice = slice("2006","2020") # this excludes GPROF and EURADCLIM
tslice = slice("2001","2020") # this excludes GPROF, EURADCLIM and RADOLAN

ccoef = dict()
crmsd = dict()
sdev = dict()

for vv in var_names: # get the name of the desired variable in the reference dataset
    if vv in data_to_stat[dsref[0]]:
        ref_var_name = vv
        break

# Get reference dataset
ds_ref = data_to_stat[dsref[0]][ref_var_name]

# Get area weights
try:
    weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                           coords=ds_ref.to_dataset()[["lat","lon"]].coords)
except AttributeError:
    weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                           coords=ds_ref.to_dataset()[["latitude","longitude"]].coords)

# Get mask
mask = utils.get_regionmask(region)
mask_ref = mask.mask(ds_ref)
if mask_TSMP_nudge: mask_ref = TSMP_no_nudge.mask(ds_ref).where(mask_ref.notnull())
ds_ref = ds_ref.where(mask_ref.notnull())#.mean(tuple([cn for cn in ds_ref.coords if cn!="time"]))

# Normalize weights in the mask
weights = weights.where(mask_ref.notnull(), other=0.)/weights.where(mask_ref.notnull(), other=0.).sum()

for dsname in data_to_stat.keys(): # compute the stats
    if dsref[0]+"-grid" in dsname or dsname==dsref[0]:
        # get dataset
        if type(data_to_stat[dsname]) is xr.DataArray:
            ds_n = data_to_stat[dsname].where(mask_ref.notnull())
        else:
            for vv in var_names:
                if vv in data_to_stat[dsname]:
                    ds_n = data_to_stat[dsname][vv].where(mask_ref.notnull())
                    break

        # Subset period
        tslice_array = ds_ref.sel(time=tslice).time

        ds_ref_tsel = ds_ref.sel(time=tslice_array)
        try:
            ds_n_tsel = ds_n.sel(time=tslice_array)
        except KeyError:
            print(dsname+" ignored because it does not cover the selected time period")
            continue
        
        # Reduce in case mode is "spatial" or "temporal"
        if mode=="spatial":
            ds_ref_tsel = ds_ref_tsel.mean("time")
            ds_n_tsel = ds_n_tsel.mean("time")
            mode_name="Spatial"
        elif mode=="temporal":
            ds_ref_tsel = ds_ref_tsel.weighted(weights).mean([cn for cn in ds_ref_tsel.dims if cn!="time"])
            ds_n_tsel = ds_n_tsel.weighted(weights).mean([cn for cn in ds_n_tsel.dims if cn!="time"])
            mode_name="Temporal"
        else:
            mode_name="Spatiotemporal"
        
        if mode=="temporal":
            # Get Correlation Coefficient (ccoef)
                
            ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel).compute()
            
            # Get Centered Root-Mean-Square-Deviation (CRMSD)
    
            crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                        (ds_ref_tsel - ds_ref_tsel.mean()) )**2
            crmsd_1 = crmsd_0.sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
            crmsd[dsname] = np.sqrt(crmsd_1)
                            
            # Get Standard Deviation (SDEV)
            
            sdev[dsname] = ds_n_tsel.std()
        else:
            # Get Correlation Coefficient (ccoef)
    
            # could work like this but I have to update xarray to include the weights
            # ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel, weigths=weights )
            
            ccoef[dsname] = xr.corr(ds_n_tsel*weights, ds_ref_tsel*weights).compute()
            
            # Get Centered Root-Mean-Square-Deviation (CRMSD)
    
            crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                        (ds_ref_tsel - ds_ref_tsel.mean()) )**2
            crmsd_1 = crmsd_0.weighted(weights).sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
            crmsd[dsname] = np.sqrt(crmsd_1)
                            
            # Get Standard Deviation (SDEV)
            
            sdev[dsname] = ds_n_tsel.weighted(weights).std()

#%%%%%% Plot the diagram
'''
Specify individual marker label (key), label color, symbol, size, symbol face color, 
symbol edge color
'''
# Define colors for each group
color_gauges = "k"
color_radar = "r"
color_satellite = "b"
color_reanalysis = "m"
color_model = "c"

# Define marker size
markersize = 7

MARKERS = {
    "GPCC-monthly": {
        "labelColor": "k",
        "symbol": "+",
        "size": markersize,
        "faceColor": color_gauges,
        "edgeColor": color_gauges,
    },
    "HYRAS": {
        "labelColor": "k",
        "symbol": "o",
        "size": markersize,
        "faceColor": color_gauges,
        "edgeColor": color_gauges,
    },
    "E-OBS": {
        "labelColor": "k",
        "symbol": "D",
        "size": markersize,
        "faceColor": color_gauges,
        "edgeColor": color_gauges,
    },
    "CPC": {
        "labelColor": "k",
        "symbol": "X",
        "size": markersize,
        "faceColor": color_gauges,
        "edgeColor": color_gauges,
    },
    "EURADCLIM": {
        "labelColor": "k",
        "symbol": "^",
        "size": markersize,
        "faceColor": color_radar,
        "edgeColor": color_radar,
    },
    "RADOLAN": {
        "labelColor": "k",
        "symbol": "s",
        "size": markersize,
        "faceColor": color_radar,
        "edgeColor": color_radar,
    },
    "RADKLIM": {
        "labelColor": "k",
        "symbol": "v",
        "size": markersize,
        "faceColor": color_radar,
        "edgeColor": color_radar,
    },
    "IMERG-V07B-monthly": {
        "labelColor": "k",
        "symbol": "d",
        "size": markersize,
        "faceColor": color_satellite,
        "edgeColor": color_satellite,
    },
    "IMERG-V06B-monthly": {
        "labelColor": "k",
        "symbol": "<",
        "size": markersize,
        "faceColor": color_satellite,
        "edgeColor": color_satellite,
    },
    "CMORPH-daily": {
        "labelColor": "k",
        "symbol": ">",
        "size": markersize,
        "faceColor": color_satellite,
        "edgeColor": color_satellite,
    },
    "GPROF": {
        "labelColor": "k",
        "symbol": "p",
        "size": markersize,
        "faceColor": color_satellite,
        "edgeColor": color_satellite,
    },
    "ERA5-monthly": {
        "labelColor": "k",
        "symbol": "*",
        "size": markersize,
        "faceColor": color_reanalysis,
        "edgeColor": color_reanalysis,
    },
    "TSMP-old-EURregLonLat01deg": {
        "labelColor": "k",
        "symbol": "h",
        "size": markersize,
        "faceColor": color_model,
        "edgeColor": color_model,
    },
    "TSMP-DETECT-Baseline-EURregLonLat01deg": {
        "labelColor": "k",
        "symbol": "8",
        "size": markersize,
        "faceColor": color_model,
        "edgeColor": color_model,
    },
}


# Set the stats in arrays like the plotting function wants them (the reference first)

lccoef = ccoef[dsref[0]].round(3).values # we round the reference so it does not go over 1
lcrmsd = crmsd[dsref[0]].values
lsdev = sdev[dsref[0]].values
labels = [dsref[0]]

for dsname in MARKERS.keys():
    dsname_grid = dsname+"_"+dsref[0]+"-grid"
    if dsname_grid in ccoef.keys():
        lccoef = np.append(lccoef, ccoef[dsname_grid].values)
        lcrmsd = np.append(lcrmsd, crmsd[dsname_grid].values)
        lsdev = np.append(lsdev, sdev[dsname_grid].values)
        labels.append(dsname_grid.split("_")[0])

# Must set figure size here to prevent legend from being cut off
plt.figure(num=1, figsize=(8, 6))

sm.taylor_diagram(lsdev,lcrmsd,lccoef, markerLabel = labels, #markerLabelColor = 'r', 
                          markerLegend = 'on', markerColor = 'r',
                           colCOR = "black", markers = {k: MARKERS[k] for k in labels[1:]}, 
                          styleOBS = '-', colOBS = 'r', markerobs = 'o', 
                          markerSize = 7, #tickRMS = [0.0, 1.0, 2.0, 3.0],
                          tickRMSangle = 115, showlabelsRMS = 'on',
                          titleRMS = 'on', titleOBS = 'Ref: '+labels[0],
                            # checkstats = "on"
                          )

ax = plt.gca()
ax.set_title(mode_name+" Taylor Diagram over "+region_name+"\n"+
             "Area-weighted yearly gridded precipitation \n"+
             str(tslice_array[0].dt.year.values)+"-"+str(tslice_array[-1].dt.year.values),
             x=1.2, y=1,)

# Create custom legend manually (because otherwise it may end in the wrong place and cannot be specified within skillmetrics)
handles_legend = []
labels_legend = []

for labeln, paramn in MARKERS.items():
    if labeln in labels and labeln != labels[0]:
        handlen = plt.Line2D(
            [], [],
            marker=paramn['symbol'],
            color=paramn['labelColor'],
            markersize=paramn['size'],
            markerfacecolor=paramn['faceColor'],
            markeredgewidth=1.5,
            markeredgecolor=paramn['edgeColor'],
            linestyle='None',
            # axes=ax
        )
        handles_legend.append(handlen)
        labels_legend.append(labeln)

# Place the custom legend
plt.legend(handles_legend, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

# To check that the equation that defines the diagram is closed (negligible residue)
sm.check_taylor_stats(lsdev, lcrmsd, lccoef, threshold=1000000000000000000000)
# 24.05.24: the check does not close but the weighted calculations seem to be fine

#%% Seasonal analysis (interannual)

#%%% Load seasonal datasets

loadpath_seasonal = "/automount/agradar/jgiles/gridded_data/seasonal/"
loadpath_agradar = "/automount/agradar/jgiles/"
loadpath_ags = "/automount/ags/jgiles/"
paths_seasonal = {
    "IMERG-V07B-monthly": loadpath_seasonal+"/IMERG-V07B-monthly/IMERG-V07B-monthly_precipitation_seasonalsum_2000-2023.nc",
    "IMERG-V06B-monthly": loadpath_seasonal+"/IMERG-V06B-monthly/IMERG-V06B-monthly_precipitation_seasonalsum_2000-2021.nc",
    # "IMERG-V07B-30min": loadpath_monthly+, 
    # "IMERG-V06B-30min": loadpath_monthly+, 
    "CMORPH-daily": loadpath_seasonal+"CMORPH-daily/CMORPH-daily_precipitation_seasonalsum_1998-2023.nc",
    "TSMP-old": loadpath_seasonal+"TSMP-old/TSMP-old_precipitation_seasonalsum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_seasonal+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_seasonalsum_2000-2022.nc",
    "ERA5-monthly": loadpath_seasonal+"ERA5-monthly/ERA5-monthly_precipitation_seasonalsum_1979-2020.nc",
    # "ERA5-hourly": loadpath_monthly+,
    "RADKLIM": loadpath_seasonal+"RADKLIM/RADKLIM_precipitation_seasonalsum_2001-2022.nc",
    "RADOLAN": loadpath_seasonal+"RADOLAN/RADOLAN_precipitation_seasonalsum_2006-2022.nc",
    "EURADCLIM": loadpath_seasonal+"EURADCLIM/EURADCLIM_precipitation_seasonalsum_2013-2020.nc",
    "GPCC-monthly": loadpath_seasonal+"GPCC-monthly/GPCC-monthly_precipitation_seasonalsum_1991-2020.nc",
    # "GPCC-daily": ,
    "GPROF": loadpath_seasonal+"GPROF/GPROF_precipitation_seasonalsum_2014-2023.nc",
    "HYRAS": loadpath_seasonal+"HYRAS/HYRAS_precipitation_seasonalsum_1931-2020.nc", 
    "E-OBS": loadpath_seasonal+"E-OBS/E-OBS_precipitation_seasonalsum_1950-2023.nc", 
    "CPC": loadpath_seasonal+"CPC/CPC_precipitation_seasonalsum_1979-2024.nc", 
    }

data_seasonalsum = {}

# load the datasets
print("Loading monthly datasets ...")
for dsname in paths_seasonal.keys():
    print("... "+dsname)
    data_seasonalsum[dsname] = xr.open_dataset(paths_seasonal[dsname])

# Special tweaks
print("Applying tweaks ...")
# RADOLAN GRID AND CRS
if "RADOLAN" in data_seasonalsum.keys():
    lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
    data_seasonalsum["RADOLAN"] = data_seasonalsum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
    data_seasonalsum["RADOLAN"] = data_seasonalsum["RADOLAN"].assign(crs=data_seasonalsum['RADKLIM'].crs[0])
    data_seasonalsum["RADOLAN"].attrs["grid_mapping"] = "crs"
    data_seasonalsum["RADOLAN"].lon.attrs = data_seasonalsum["RADKLIM"].lon.attrs
    data_seasonalsum["RADOLAN"].lat.attrs = data_seasonalsum["RADKLIM"].lat.attrs

# EURADCLIM coords
if "EURADCLIM" in data_seasonalsum.keys():
    data_seasonalsum["EURADCLIM"] = data_seasonalsum["EURADCLIM"].set_coords(("lon", "lat"))

# Shift HYRAS and EURADCLIM timeaxis
if "EURADCLIM" in data_seasonalsum.keys():
    data_seasonalsum["EURADCLIM"]["time"] = data_seasonalsum["EURADCLIM"]["time"].get_index('time').shift(2, "M") # We place the seasonal value in the last month
    data_seasonalsum["EURADCLIM"]["time"] = data_seasonalsum["EURADCLIM"]["time"].dt.floor("D") # set to hour 0
if "HYRAS" in data_seasonalsum.keys():
    data_seasonalsum["HYRAS"]["time"] = data_seasonalsum["HYRAS"]["time"].get_index('time').shift(2, "M") # We place the seasonal value in the last month
    data_seasonalsum["HYRAS"]["time"] = data_seasonalsum["HYRAS"]["time"].dt.floor("D") # set to hour 0

# Convert all non datetime axes (cf Julian calendars) into datetime 
for dsname in data_seasonalsum.keys():
    try:
        data_seasonalsum[dsname]["time"] = data_seasonalsum[dsname].indexes['time'].to_datetimeindex()
        print(dsname+" time dimension transformed to datetime format")
    except:
        pass

# Adjustment for leap years 
for dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
    data_seasonalsum[dsname]["time"] = data_seasonalsum[dsname]["time"].get_index('time').shift(-1, "M").shift(1, "M")
    print(dsname+" adjusted for leap year")
    
# Special selections for incomplete extreme years
# EURADCLIM
if "EURADCLIM" in data_seasonalsum.keys():
    data_seasonalsum["EURADCLIM"] = data_seasonalsum["EURADCLIM"].loc[{"time":slice("2013-05", "2020-12")}]
# HYRAS
if "HYRAS" in data_seasonalsum.keys():
    data_seasonalsum["HYRAS"] = data_seasonalsum["HYRAS"].loc[{"time":slice("1931-02", "2020-11")}]

colors = {
    "IMERG-V07B-monthly": "#FF6347", # Tomato
    "IMERG-V06B-monthly": "crimson", # crimson
    "CMORPH-daily": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", # DodgerBlue
    "ERA5-monthly": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

#%%% Regional averages
#%%%% Calculate area means (regional averages)
data_to_avg = data_seasonalsum.copy() # select which data to average (yearly, monthly, daily...)

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland", 
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra", 
         "Monaco", "Czechia", "Slovakia", "Hungary", "Slovenia", "Romania"]#"land"
region = "Germany"
region_name = "Germany" # name for plots
mask = utils.get_regionmask(region)
TSMP_nudge_margin = 13 # number of gridpoints to mask out the relaxation zone at the margins

# TSMP-case: we make a specific mask to cut out the edge of the european domain + country
dsname = "TSMP-DETECT-Baseline"
mask_TSMP_nudge = False
if dsname in data_to_avg.keys():
    mask_TSMP_nudge = True # This will be used later as a trigger for this extra mask
    lon_bot = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lon.values
    lat_bot = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lat.values
    lon_top = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lon.values
    lat_top = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lat.values
    lon_right = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lon.values
    lat_right = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lat.values
    lon_left = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lon.values
    lat_left = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lat.values
    
    lon_tsmp_edge = np.concatenate((lon_bot, lon_right, lon_top[::-1], lon_left[::-1]))
    lat_tsmp_edge = np.concatenate((lat_bot, lat_right, lat_top[::-1], lat_left[::-1]))
    
    lonlat_tsmp_edge = list(zip(lon_tsmp_edge, lat_tsmp_edge))
    
    TSMP_no_nudge = rm.Regions([ lonlat_tsmp_edge ], names=["TSMP_no_nudge"], abbrevs=["TSMP_NE"], name="TSMP")
    # I did not find a way to directly combine this custom region with a predefined country region. I will 
    # have to just apply the masks consecutively

data_avgreg = {}
# Means over region
print("Calculating means over "+region_name)
to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():
    print("... "+dsname)

    if dsname in ["RADOLAN", "RADKLIM", "HYRAS", "EURADCLIM"]:
        # these datasets come in equal-pixel-sized grids, so we only need to apply the average over the region
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        with ProgressBar():
            data_avgreg[dsname] = data_to_avg[dsname].where(mask0.notnull()).mean(("x", "y")).compute()

    if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly", "CMORPH-daily", "ERA5-monthly", 
                  "GPCC-monthly", "GPCC-daily", "GPROF", "E-OBS", "CPC"]:
        # these datasets come in regular lat-lon grids, so we need to average over the region considering the area weights
        variables_to_include = [vv for vv in data_to_avg[dsname].data_vars \
                                if "lonv" not in data_to_avg[dsname][vv].dims \
                                if "latv" not in data_to_avg[dsname][vv].dims \
                                if "nv" not in data_to_avg[dsname][vv].dims]
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        if dsname in ["ERA5-monthly", "E-OBS"]:
            with ProgressBar():
                data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                              lon_name="longitude", lat_name="latitude").compute()
        else:
            with ProgressBar():
                data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                              lon_name="lon", lat_name="lat").compute()

    if dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
        # we need to unrotate the TSMP grid and then average over the region considering the area weights

        grid_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
        regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]), grid_out, "conservative")
        to_add[dsname+"-EURregLonLat01deg"] = regridder(data_to_avg[dsname])
        
        mask0 = mask.mask(to_add[dsname+"-EURregLonLat01deg"])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(to_add[dsname+"-EURregLonLat01deg"]).where(mask0.notnull())

        with ProgressBar():
            data_avgreg[dsname] = utils.calc_spatial_mean(to_add[dsname+"-EURregLonLat01deg"].where(mask0.notnull()), 
                                                          lon_name="lon", lat_name="lat").compute()
        
# add the rotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}
data_seasonalsum = data_to_avg.copy()

#%%%% Simple map plot
dsname = "GPCC-monthly"
vname = "precip"
tsel = "2015-02"
mask = utils.get_regionmask(region)
mask0 = mask.mask(data_seasonalsum[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(data_seasonalsum[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_seasonalsum[dsname][vname].sel(time=tsel).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
if mask_TSMP_nudge: ax1.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(dsname)

#%%%% Simple map plot (for number of stations per gridcell) # CHECK THIS FOR SEASONAL SUM BEFORE RUNNING!!
dsname = "GPCC-monthly"
vname = "numgauge"
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Turkey/"

for yy in np.arange(2000,2021):
    ysel = str(yy)
    mask = utils.get_regionmask(region)
    mask0 = mask.mask(data_seasonalsum[dsname])
    dropna = True
    if mask_TSMP_nudge: 
        mask0 = TSMP_no_nudge.mask(data_seasonalsum[dsname]).where(mask0.notnull())
        dropna=False
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    cmap1 = copy.copy(plt.cm.Blues)
    cmap1.set_under("lightgray")
    plot = (data_seasonalsum[dsname][vname].sel(time=ysel)/12).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", 
                                            levels=3, cmap=cmap1, vmin=1, vmax=3, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                             cbar_kwargs={'label': "", 'shrink':0.88})
    if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    plt.title(dsname+" number of stations per gridcell "+ysel)
    # save figure
    savepath_yy = savepath+ysel+"/"
    if not os.path.exists(savepath_yy):
        os.makedirs(savepath_yy)
    filename = "numgauge_"+region_name+"_"+dsname+"_"+ysel+".png"
    plt.savefig(savepath_yy+filename, bbox_inches="tight")
    # plt.show()

#%%%% Interannual variability area-means plot
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

selseaslist = [("DJF", [2]),
           ("MAM", [5]),
           ("JJA", [8]),
           ("SON", [11]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofseas", [ending_month])

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/"


for selseas in selseaslist:
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
                else: 
                    color = colors[dsname]
                try:
                    plt.plot(data_avgreg[dsname]['time'].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])), 
                             data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])), 
                             label=dsname, c=color, marker=marker)
                except TypeError:
                    # try to change the time coord to datetime format
                    plt.plot(data_avgreg[dsname].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])).indexes['time'].to_datetimeindex(), 
                             data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])), 
                             label=dsname, c=color, marker=marker)
                plotted = True
        if not plotted:
            raise Warning("Nothing plotted for "+dsname)
    
    plt.legend(ncols=3, fontsize=7)
    plt.title("Area-mean "+selseas[0]+" total precip "+region_name+" [mm]")
    plt.xlim(datetime(2000,1,1), datetime(2020,1,1))
    # plt.xlim(2000, 2020)
    plt.grid()
    # save figure
    savepath_seas = savepath+selseas[0]+"/lineplots/"
    if not os.path.exists(savepath_seas):
        os.makedirs(savepath_seas)
    filename = "area_mean_"+selseas[0]+"_total_precip_"+region_name+".png"
    plt.savefig(savepath_seas+filename, bbox_inches="tight")
    plt.close()

#%%%% Interannual variability area-means plot (interactive html)

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

selseaslist = [("DJF", [2]),
           ("MAM", [5]),
           ("JJA", [8]),
           ("SON", [11]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofseas", ending_month)

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/"


for selseas in selseaslist:
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
                                    data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])).hvplot.line(x='time', label=dsname).opts(color=color, line_width=lw, show_legend=True, muted_alpha=0)
                        )
                
                else:
                    hvplots.append(
                                    data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selseas[1])).hvplot.line(x='time', label=dsname).opts(show_legend=True, muted_alpha=0)
                        )
    
                plotted = True
        if not plotted:
            raise Warning("Nothing plotted for "+dsname)
    
    layout = hvplots[0]
    for nplot in np.arange(1, len(hvplots)):
        layout = layout * hvplots[nplot]
    
    layout.opts(title="Area-mean "+selseas[0]+" total precip "+region_name+" [mm]", xlabel="Time", show_grid=True, legend_position='right',
                height=600, width=1200)
    
    # Save to HTML file
    savepath_seas = savepath+selseas[0]+"/lineplots/"
    if not os.path.exists(savepath_seas):
        os.makedirs(savepath_seas)
    filename = "area_mean_"+selseas[0]+"_total_precip_"+region_name+".html"
    hv.save(layout, savepath_seas+filename)

#%%% BIAS and ERRORS
#%%%% Bias (absolute and relative) calculation from regional averages
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

dsignore = [] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference

data_to_bias = data_avgreg

data_bias = {}
data_bias_relative = {}
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    data_bias[dsname] = data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref]
                    data_bias_relative[dsname] = (data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref])/data_to_bias[dsref[0]][vvref]*100
                    break

#%%%% Region-averages bias bar plot

selseaslist = [("DJF", [2]),
           ("MAM", [5]),
           ("JJA", [8]),
           ("SON", [11]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofseas", ending_month)

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/"

for selseas in selseaslist:
    # Calculate bar width based on the number of data arrays
    bar_width = 0.8 / len(data_bias)
        
    # Get time values
    time_values_ref = data_to_bias[dsref[0]]['time'].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selseas[1]))
    
    # Plotting each DataArray in the dictionary
    plt.figure(figsize=(20, 6))  # Adjust figure size as needed
    for idx, (key, value) in enumerate(data_bias.items()):
        value_padded = value.sel(time=value['time'].dt.month.isin(selseas[1])).broadcast_like(data_to_bias[dsref[0]].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selseas[1])))
        time_values = value_padded['time']
        bar_positions = np.arange(len(time_values)) + idx * bar_width
        plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])
    
    plt.xlabel('Time')
    plt.ylabel(value.attrs['units'])
    plt.title("Area-mean "+selseas[0]+" total precip BIAS with respect to "+dsref[0]+" "+region_name)
    plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # save figure
    savepath_seas = savepath+selseas[0]+"/barplots/"
    if not os.path.exists(savepath_seas):
        os.makedirs(savepath_seas)
    filename = "area_mean_"+selseas[0]+"_precip_totals_bias_"+region_name+".png"
    plt.savefig(savepath_seas+filename, bbox_inches="tight")
    plt.close()

    # Plotting each DataArray in the dictionary (same but for relative bias)
    plt.figure(figsize=(20, 6))  # Adjust figure size as needed
    for idx, (key, value) in enumerate(data_bias_relative.items()):
        value_padded = value.sel(time=value['time'].dt.month.isin(selseas[1])).broadcast_like(data_to_bias[dsref[0]].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selseas[1])))
        time_values = value_padded['time']
        bar_positions = np.arange(len(time_values)) + idx * bar_width
        plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])
    
    plt.xlabel('Time')
    plt.ylabel("%")
    plt.title("Area-mean "+selseas[0]+" total precip RELATIVE BIAS with respect to "+dsref[0]+" "+region_name)
    plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # save figure
    savepath_seas = savepath+selseas[0]+"/barplots/"
    if not os.path.exists(savepath_seas):
        os.makedirs(savepath_seas)
    filename = "area_mean_"+selseas[0]+"_precip_totals_bias_relative_"+region_name+".png"
    plt.savefig(savepath_seas+filename, bbox_inches="tight")
    plt.close()

#%%%% Relative bias and errors calculation (at gridpoint level, not with the area means)
# First we need to transform EURADCLIM, RADKLIM, RADOLAN and HYRAS to regular grids
# We use the DETECT 1 km grid for this (actually, I did not manage to make that work)
lonlims = slice(TSMP_no_nudge.bounds_global[0], TSMP_no_nudge.bounds_global[2])
latlims = slice(TSMP_no_nudge.bounds_global[1], TSMP_no_nudge.bounds_global[3])

to_add = {} # dictionary to add regridded versions
for dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
    if dsname not in data_seasonalsum: continue
    print("Regridding "+dsname+" ...")

    grid_out = xe.util.cf_grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    grid_out = xe.util.grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    # # I tried to use dask for the weight generation to avoid memory crash
    # # but I did not manage to make it work: https://xesmf.readthedocs.io/en/latest/notebooks/Dask.html

    # grid_out = grid_out.chunk({"x": 50, "y": 50, "x_b": 50, "y_b": 50,})

    # # we then try parallel regridding: slower but less memory-intensive (this takes forever)
    # regridder = xe.Regridder(data_seasonalsum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True)
    # to_add[dsname+"-EURregLonLat001deg"] = regridder(data_to_avg[dsname])
    # regridder.to_netcdf() # we save the weights
    # # to reuse the weigths:
    # xe.Regridder(data_seasonalsum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True, weights="/path/to/weights") #!!! Can I use CDO weights here?
    # Cdo().gencon()
    # cdo gencon,/automount/ags/jgiles/IMERG_V06B/global_monthly/griddes.txt -setgrid,/automount/agradar/jgiles/TSMP/griddes_mod.txt /automount/agradar/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020.nc weights_to_IMERG.nc

    # Instead, just regrid to the reference dataset grid (this is fast)

    regridder = xe.Regridder(data_seasonalsum[dsname].cf.add_bounds(["lon", "lat"]), 
                             data_seasonalsum[dsref[0]].loc[{"lon": lonlims, "lat": latlims}], 
                             "conservative")
    to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_seasonalsum[dsname])
    
# add the regridded datasets to the original dictionary
data_seasonalsum = {**data_seasonalsum, **to_add}
    
# Compute the biases
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline'] # datasets to ignore (because we want the regridded version)
data_to_bias = copy.copy(data_seasonalsum)

to_add = {} # dictionary to add regridded versions

data_bias_map = {} # maps of yearly biases
data_bias_relative_map = {} # maps of yearly relative biases
data_abs_error_map = {} # maps of yearly absolute errors
data_bias_relative_gp = {} # yearly relative biases on a gridpoint basis (sum of gridpoint biases divided by sum of reference-data values)
data_mean_abs_error_gp = {} # yearly MAE on a gridpoint basis (sum of gridpoint abs errors divided by number of data values)
data_norm_mean_abs_error_gp = {} # yearly NMAE on a gridpoint basis (sum of gridpoint abs errors divided by sum of reference-data values)
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    print("Processing "+dsname+" ...")
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    if dsref[0]+"-grid" not in dsname: # if no regridded already, do it now
                        if "longitude" in data_to_bias[dsname].coords or "latitude" in data_to_bias[dsname].coords:
                            # if the names of the coords are longitude and latitude, change them to lon, lat
                            data_to_bias[dsname] = data_to_bias[dsname].rename({"longitude":"lon", "latitude":"lat"})
                        
                        if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bnds", "lat_bnds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])
                            
                        if dsname in ["CMORPH-daily"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bounds", "lat_bounds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        if dsname in ["GPROF", "TSMP-old-EURregLonLat01deg", "TSMP-DETECT-Baseline-EURregLonLat01deg"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        # regridder = xe.Regridder(data_to_bias[dsname].cf.add_bounds(["lon", "lat"]), data_to_bias[dsref[0]], "conservative")

                        regridder = xe.Regridder(data_to_bias[dsname], 
                                                 data_to_bias[dsref[0]].loc[{"lon": lonlims, "lat": latlims}], 
                                                 "conservative")

                        to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_to_bias[dsname][vv])
                        data0 = to_add[dsname+"_"+dsref[0]+"-grid"].copy()
                    else:
                        data0 = data_to_bias[dsname][vv].copy()

                    data0 = data0.where(data0>0)
                    dataref = data_to_bias[dsref[0]][vvref].loc[{"lon": lonlims, "lat": latlims}]
                    
                    mask0 = mask.mask(dataref)
                    if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

                    data_bias_map[dsname] = ( data0 - dataref ).compute()
                    data_bias_relative_map[dsname] = ( data_bias_map[dsname] / dataref ).compute() *100
                    data_bias_map_masked = data_bias_map[dsname].where(mask0.notnull())
                    data_bias_relative_map_masked = data_bias_relative_map[dsname].where(mask0.notnull())
                    data_abs_error_map[dsname] = abs(data_bias_map[dsname])
                    data_abs_error_map_masked = data_abs_error_map[dsname].where(mask0.notnull())
                    
                    data_bias_relative_gp[dsname] = utils.calc_spatial_integral(data_bias_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100
                    
                    data_norm_mean_abs_error_gp[dsname] = utils.calc_spatial_integral(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100

                    data_mean_abs_error_gp[dsname] = utils.calc_spatial_mean(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute()
                    
                    break

# add the regridded datasets to the original dictionary
data_seasonalsum = {**data_seasonalsum, **to_add}

#%%%% Relative bias and error plots
#%%%%% Simple map plot
# region = "Germany" #"land" 
to_plot = data_bias_map
dsname = "IMERG-V07B-monthly"
title = "BIAS"
yearsel = "2016-02"
cbarlabel = "mm" # mm
vmin = -250
vmax = 250
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
mask = utils.get_regionmask(region)
mask0 = mask.mask(to_plot[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":yearsel}].where(mask0.notnull(), drop=dropna).loc[{"lon":lonlat_slice[0], 
                                                                                      "lat":lonlat_slice[1]}].plot(x="lon", 
                                                                                                                   y="lat", 
                                                                                                                   cmap="RdBu_r", 
                                                                                    vmin=vmin, vmax=vmax, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+yearsel+"\n"+dsname+"\n "+region_name+" Ref.: "+dsref[0])

#%%%%% Simple map plot (loop)
# Like previous but for saving all plots !! THIS WILL PROBABLY CRASH THE MEMORY
# region = "Germany" #"land" 
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/seasons/"+region_name+"/"
period = np.arange(2000,2024)
to_plot_dict = [
            (data_bias_map, "BIAS", "mm", -250, 250),
            (data_bias_relative_map, "RELATIVE BIAS", "%", -75, 75),
           ]
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]

selseaslist = [("DJF", "02"),
           ("MAM", "05"),
           ("JJA", "08"),
           ("SON", "11")
           ] # ("nameofseas", "ending_month")

for to_plot, title, cbarlabel, vmin, vmax in to_plot_dict:
    print("Plotting "+title)
    for dsname in to_plot.keys():
        print("... "+dsname)
        dsname_short = dsname.split("_")[0]
        mask = utils.get_regionmask(region)
        mask0 = mask.mask(to_plot[dsname])
        dropna = True
        if not mask_TSMP_nudge:  # Remove/add "not" here to change the extent of the map
            mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
            dropna=False
        for yearsel in period:
            for selseas in selseaslist:
                try:
                    plt.close()
                    yearseas = str(yearsel)+"-"+str(selseas[1])
                    # f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
                    plot = to_plot[dsname].loc[{"time":yearseas}].where(mask0.notnull(), 
                                                                            drop=dropna).loc[{"lon":lonlat_slice[0],
                                                                                              "lat":lonlat_slice[1]}].plot(x="lon",
                                                                                                                           y="lat",
                                                                                                                           cmap="RdBu_r", 
                                                                                                        vmin=vmin, vmax=vmax, 
                                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                                             cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
                    # if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
                    plot.axes.coastlines(alpha=0.7)
                    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
                    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
                    plt.title(title+" "+str(yearsel)+" "+selseas[0]+"\n"+dsname_short+"\n "+region_name+" Ref.: "+dsref[0])
                    
                    # save figure
                    savepath_yy = savepath+str(yearsel)+"/"+selseas[0]+"/"
                    if not os.path.exists(savepath_yy):
                        os.makedirs(savepath_yy)
                    filename = "_".join([title.lower().replace(" ","_"), region_name, dsname_short,dsref[0],str(yearsel),selseas[0]])+".png"
                    plt.savefig(savepath_yy+filename, bbox_inches="tight")
                    plt.close()
                except KeyError:
                    continue

#%%%%% Box plots of BIAS and ERRORS
# the box plots are made up of the yearly bias or error values, and the datasets are ordered according to their median
to_plot0 = data_bias_relative_gp.copy() # data_mean_abs_error_gp # data_bias_relative_gp # data_norm_mean_abs_error_gp
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/boxplots/relative_bias/"
savefilename = "boxplot_relative_bias_" # the season name will be added at the end of savefilename
title = "relative bias "+region_name+". Ref.: "+dsref[0] # the season name will be added at the beginning of title
ylabel = "%" # % # mm
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# Override the previous with a loop for each case
savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/boxplots/"
for to_plot0, savepath, savefilename, title, ylabel in [
        (data_bias_relative_gp.copy(), 
         savepathbase + "relative_bias/",
         "boxplot_relative_bias_",
         "relative bias "+region_name+". Ref.: "+dsref[0],
         "%"),
        (data_mean_abs_error_gp.copy(), 
         savepathbase + "mean_absolute_error/",
         "boxplot_mean_absolute_error_",
         "mean absolute error "+region_name+". Ref.: "+dsref[0],
         "mm"),
        (data_norm_mean_abs_error_gp.copy(), 
         savepathbase + "normalized_mean_absolute_error/",
         "boxplot_normalized_mean_absolute_error_",
         "normalized mean absolute error "+region_name+". Ref.: "+dsref[0],
         "%"),
        ]:

    tsel = [
            slice(None, None), # if I want to consider only certain period. Otherwise set to (None, None). Multiple options possible
            slice("2001-01-01", "2020-01-01"), 
            slice("2006-01-01", "2020-01-01"), 
            slice("2013-01-01", "2020-01-01"), 
            slice("2015-01-01", "2020-01-01")
            ] 
    ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))
    selseaslist = [("DJF", [2]),
               ("MAM", [5]),
               ("JJA", [8]),
               ("SON", [11]),
               ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofseas", ending_month)
    
    for selseas in selseaslist:
        savepathn = (region_name+"/"+selseas[0]).join(copy.deepcopy(savepath).split(region_name))
        for tseln in tsel:
            to_plot = to_plot0.copy()
            savefilenamen = copy.deepcopy(savefilename)
            titlen = copy.deepcopy(title)
            titlen = selseas[0]+" "+titlen
            savefilenamen = savefilenamen+selseas[0]
            
            if tseln.start is not None and tseln.stop is not None: # add specific period to title
                titlen = titlen+". "+tseln.start+" - "+tseln.stop
                savefilenamen = savefilenamen+"_"+tseln.start+"-"+tseln.stop
                
                if ignore_incomplete:
                    for key in to_plot.copy().keys():
                        if not (to_plot[key].time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot[key].time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            del(to_plot[key])
            
            # Select the given season
            for key in to_plot.copy().keys():
                to_plot[key] = to_plot[key].sel(time=to_plot[key]['time'].dt.month.isin(selseas[1]))
            
            # Initialize a figure and axis
            plt.close()
            plt.figure(figsize=(1.25*(len(to_plot.keys())-len(dsignore)), 6))
            ax = plt.subplot(111)
            
            # Create a list to hold the data arrays
            plotted_arrays = []
            plotted_arrays_lengths = []
            
            # Order according to median:
            to_plot = dict(sorted(to_plot.items(), key=lambda item: item[1].sel(time=tseln).median()))
            
            # Iterate over the datasets in the dictionary
            for key, value in to_plot.items():
                if key not in dsignore:
                    # Plot a box plot for each dataset
                    value = value.sel(time=tseln)
                    plotted_arrays.append(value.values) # values of each box
                    plotted_arrays_lengths.append(len(value)) # number of values in each box
                    ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6, 
                               patch_artist=True, boxprops=dict(facecolor='#b6d6e3'),
                               medianprops=dict(color="#20788e", lw=2))
            
            # Set x-axis ticks and labels with dataset names
            ax.set_xticks(range(1, len(plotted_arrays) + 1))
            ax.set_xticklabels([dsname.split("_")[0] if "_" in dsname 
                                else "-".join(dsname.split("-")[:-1]) if "EURreg" in dsname 
                                else dsname 
                                for dsname in 
                                [ds for ds in to_plot.keys() if ds not in dsignore]
                                ],
                               rotation=45, fontsize=15)
            ax.xaxis.label.set_size(15)     # change xlabel size
            ax.yaxis.label.set_size(15)     # change ylabel size
            
            ax.tick_params(axis='x', labelsize=15) # change xtick label size
            ax.tick_params(axis='y', labelsize=15) # change xtick label size
            
            # Make a secondary x axis to display the number of values in each box
            ax2 = ax.secondary_xaxis('top')
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("top")
            
            ax2.set_xticks(range(1, len(plotted_arrays) + 1))
            ax2.set_xticklabels(plotted_arrays_lengths)
            ax2.set_xlabel('Number of years', fontsize= 15)
            
            # Set labels and title
            #ax.set_xlabel('')
            ax.set_ylabel(ylabel)
            ax.set_title(titlen, fontsize=20)
            
            # plot a reference line at zero
            plt.hlines(y=0, xmin=0, xmax=len(plotted_arrays)+1, colors='black', lw=2, zorder=0)
            plt.xlim(0.5, len(plotted_arrays) + 0.5)
            
            # Show the plot
            plt.grid(True)
            plt.tight_layout()
            if not os.path.exists(savepathn):
                os.makedirs(savepathn)
            plt.savefig(savepathn+savefilenamen+".png", bbox_inches="tight")
            plt.close()

#%%%%% Taylor diagram
# The Taylor diagram can be done by computing the stats over all gridpoints and all timesteps (spatiotemporal)
# or only doing the stats over space or time separately (for these, either temporal or spatial averages must be done first)
    
#%%%%%% Compute stats and plot for all seasons
import skill_metrics as sm
# https://github.com/PeterRochford/SkillMetrics/blob/master/Examples/taylor10.py#L123
# I cannot use skill_metrics to calculate the stats because they do not filter out 
# nan values (because of the masks) so the result is erroneous. They also do not handle weighted arrays.

mode = "" # if "spatial" then average in time and compute the diagram in space. Viceversa for "temporal"
dsref = ["GPCC-monthly"]
data_to_stat = data_seasonalsum

# choose common period (only datasets that cover the whole period are included)
tslice = slice("2015","2020") # this covers all
# tslice = slice("2013","2020") # this excludes GPROF
# tslice = slice("2006","2020") # this excludes GPROF and EURADCLIM
tslice = slice("2001","2020") # this excludes GPROF, EURADCLIM and RADOLAN

selseaslist = [("DJF", [2]),
           ("MAM", [5]),
           ("JJA", [8]),
           ("SON", [11]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofseas", ending_month)

# Override options above and sweep over them in a loop
for mode in ["", "spatial", "temporal"]:
    for tslice in [
        slice("2015","2020"), # this covers all
        slice("2013","2020"), # this excludes GPROF
        slice("2006","2020"), # this excludes GPROF and EURADCLIM
        slice("2001","2020"), # this excludes GPROF, EURADCLIM and RADOLAN
            ]:

        print("Plotting Taylor diagrams (mode: "+mode+", "+tslice.start+"-"+tslice.stop+")...")
        for selseas in selseaslist:
            print("... "+selseas[0])
            
            ccoef = dict()
            crmsd = dict()
            sdev = dict()
            
            for vv in var_names: # get the name of the desired variable in the reference dataset
                if vv in data_to_stat[dsref[0]]:
                    ref_var_name = vv
                    break
            
            # Get reference dataset
            ds_ref = data_to_stat[dsref[0]][ref_var_name].sel(time=data_to_stat[dsref[0]]['time'].dt.month.isin(selseas[1]))
            
            # Get area weights
            try:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["lat","lon"]].coords)
            except AttributeError:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["latitude","longitude"]].coords)
            
            # Get mask
            mask = utils.get_regionmask(region)
            mask_ref = mask.mask(ds_ref)
            if mask_TSMP_nudge: mask_ref = TSMP_no_nudge.mask(ds_ref).where(mask_ref.notnull())
            ds_ref = ds_ref.where(mask_ref.notnull())#.mean(tuple([cn for cn in ds_ref.coords if cn!="time"]))
            
            # Normalize weights in the mask
            weights = weights.where(mask_ref.notnull(), other=0.)/weights.where(mask_ref.notnull(), other=0.).sum()
            
            for dsname in data_to_stat.keys(): # compute the stats
                if dsref[0]+"-grid" in dsname or dsname==dsref[0]:
                    # get dataset
                    if type(data_to_stat[dsname]) is xr.DataArray:
                        ds_n = data_to_stat[dsname].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selseas[1])).where(mask_ref.notnull())
                    else:
                        for vv in var_names:
                            if vv in data_to_stat[dsname]:
                                ds_n = data_to_stat[dsname][vv].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selseas[1])).where(mask_ref.notnull())
                                break
            
                    # Subset period
                    tslice_array = ds_ref.sel(time=tslice).time
            
                    ds_ref_tsel = ds_ref.sel(time=tslice_array)
                    try:
                        ds_n_tsel = ds_n.sel(time=tslice_array)
                    except KeyError:
                        print(dsname+" ignored because it does not cover the selected time period")
                        continue
                    
                    # Reduce in case mode is "spatial" or "temporal"
                    if mode=="spatial":
                        ds_ref_tsel = ds_ref_tsel.mean("time")
                        ds_n_tsel = ds_n_tsel.mean("time")
                        mode_name="Spatial"
                    elif mode=="temporal":
                        ds_ref_tsel = ds_ref_tsel.weighted(weights).mean([cn for cn in ds_ref_tsel.dims if cn!="time"])
                        ds_n_tsel = ds_n_tsel.weighted(weights).mean([cn for cn in ds_n_tsel.dims if cn!="time"])
                        mode_name="Temporal"
                    else:
                        mode_name="Spatiotemporal"
                    
                    if mode=="temporal":
                        # Get Correlation Coefficient (ccoef)
                            
                        ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.std()
                    else:
                        # Get Correlation Coefficient (ccoef)
                
                        # could work like this but I have to update xarray to include the weights
                        # ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel, weigths=weights )
                        
                        ccoef[dsname] = xr.corr(ds_n_tsel*weights, ds_ref_tsel*weights).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.weighted(weights).sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.weighted(weights).std()
            
            # Plot the diagram
            savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual_by_seasons/"+region_name+"/"+selseas[0]+"/taylor_diagrams/"
            savefilename = "taylor_"+selseas[0]+"_precip_totals_"+region_name+"_"+tslice.start+"-"+tslice.stop+".png"
            if mode != "":
                savefilename = mode+"_"+savefilename
            '''
            Specify individual marker label (key), label color, symbol, size, symbol face color, 
            symbol edge color
            '''
            # Define colors for each group
            color_gauges = "k"
            color_radar = "r"
            color_satellite = "b"
            color_reanalysis = "m"
            color_model = "c"
            
            # Define marker size
            markersize = 7
            
            MARKERS = {
                "GPCC-monthly": {
                    "labelColor": "k",
                    "symbol": "+",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "HYRAS": {
                    "labelColor": "k",
                    "symbol": "o",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "E-OBS": {
                    "labelColor": "k",
                    "symbol": "D",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "CPC": {
                    "labelColor": "k",
                    "symbol": "X",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "EURADCLIM": {
                    "labelColor": "k",
                    "symbol": "^",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADOLAN": {
                    "labelColor": "k",
                    "symbol": "s",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADKLIM": {
                    "labelColor": "k",
                    "symbol": "v",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "IMERG-V07B-monthly": {
                    "labelColor": "k",
                    "symbol": "d",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "IMERG-V06B-monthly": {
                    "labelColor": "k",
                    "symbol": "<",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "CMORPH-daily": {
                    "labelColor": "k",
                    "symbol": ">",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "GPROF": {
                    "labelColor": "k",
                    "symbol": "p",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "ERA5-monthly": {
                    "labelColor": "k",
                    "symbol": "*",
                    "size": markersize,
                    "faceColor": color_reanalysis,
                    "edgeColor": color_reanalysis,
                },
                "TSMP-old-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "h",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
                "TSMP-DETECT-Baseline-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "8",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
            }
            
            
            # Set the stats in arrays like the plotting function wants them (the reference first)
            
            lccoef = ccoef[dsref[0]].round(3).values # we round the reference so it does not go over 1
            lcrmsd = crmsd[dsref[0]].values
            lsdev = sdev[dsref[0]].values
            labels = [dsref[0]]
            
            for dsname in MARKERS.keys():
                dsname_grid = dsname+"_"+dsref[0]+"-grid"
                if dsname_grid in ccoef.keys():
                    lccoef = np.append(lccoef, ccoef[dsname_grid].values)
                    lcrmsd = np.append(lcrmsd, crmsd[dsname_grid].values)
                    lsdev = np.append(lsdev, sdev[dsname_grid].values)
                    labels.append(dsname_grid.split("_")[0])
            
            # Must set figure size here to prevent legend from being cut off
            plt.close()
            plt.figure(num=1, figsize=(8, 6))
            
            sm.taylor_diagram(lsdev,lcrmsd,lccoef, markerLabel = labels, #markerLabelColor = 'r', 
                                      markerLegend = 'on', markerColor = 'r',
                                       colCOR = "black", markers = {k: MARKERS[k] for k in labels[1:]}, 
                                      styleOBS = '-', colOBS = 'r', markerobs = 'o', 
                                      markerSize = 7, #tickRMS = [0.0, 1.0, 2.0, 3.0],
                                      tickRMSangle = 115, showlabelsRMS = 'on',
                                      titleRMS = 'on', titleOBS = 'Ref: '+labels[0],
                                        # checkstats = "on"
                                      )
            
            ax = plt.gca()
            ax.set_title(mode_name+" Taylor Diagram over "+region_name+"\n"+
                         "Area-weighted "+selseas[0]+" gridded precipitation \n"+
                         str(tslice_array[0].dt.year.values)+"-"+str(tslice_array[-1].dt.year.values),
                         x=1.2, y=1,)
            
            # Create custom legend manually (because otherwise it may end in the wrong place and cannot be specified within skillmetrics)
            handles_legend = []
            labels_legend = []
            
            for labeln, paramn in MARKERS.items():
                if labeln in labels and labeln != labels[0]:
                    handlen = plt.Line2D(
                        [], [],
                        marker=paramn['symbol'],
                        color=paramn['labelColor'],
                        markersize=paramn['size'],
                        markerfacecolor=paramn['faceColor'],
                        markeredgewidth=1.5,
                        markeredgecolor=paramn['edgeColor'],
                        linestyle='None',
                        # axes=ax
                    )
                    handles_legend.append(handlen)
                    labels_legend.append(labeln)
            
            # Place the custom legend
            plt.legend(handles_legend, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
            
            # Save figure
            savepath_seas = savepath
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            plt.savefig(savepath_seas+savefilename, bbox_inches="tight")
            plt.close()
            
            # To check that the equation that defines the diagram is closed (negligible residue)
            sm.check_taylor_stats(lsdev, lcrmsd, lccoef, threshold=1000000000000000000000)
            # 24.05.24: the check does not close but the weighted calculations seem to be fine


#%% Monthly analysis

#%%% Load monthly datasets

loadpath_monthly = "/automount/agradar/jgiles/gridded_data/monthly/"
loadpath_agradar = "/automount/agradar/jgiles/"
loadpath_ags = "/automount/ags/jgiles/"
paths_monthly = {
    "IMERG-V07B-monthly": loadpath_monthly+"this path does not matter",
    "IMERG-V06B-monthly": loadpath_monthly+"this path does not matter",
    # "IMERG-V07B-30min": loadpath_monthly+, 
    # "IMERG-V06B-30min": loadpath_monthly+, 
    "CMORPH-daily": loadpath_monthly+"CMORPH-daily/CMORPH-daily_precipitation_monthlysum_1998-2023.nc",
    "TSMP-old": loadpath_monthly+"TSMP-old/TSMP-old_precipitation_monthlysum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_monthly+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_monthlysum_2000-2022.nc",
    "ERA5-monthly": loadpath_monthly+"this path does not matter",
    # "ERA5-hourly": loadpath_monthly+,
    "RADKLIM": loadpath_monthly+"RADKLIM/RADKLIM_precipitation_monthlysum_2001-2022.nc",
    "RADOLAN": loadpath_monthly+"RADOLAN/RADOLAN_precipitation_monthlysum_2006-2022.nc",
    "EURADCLIM": loadpath_monthly+"EURADCLIM/EURADCLIM_precipitation_monthlysum_2013-2020.nc",
    "GPCC-monthly": loadpath_monthly+"this path does not matter",
    # "GPCC-daily": ,
    "GPROF": loadpath_monthly+"GPROF/GPROF_precipitation_monthlysum_2014-2023.nc",
    "HYRAS": loadpath_monthly+"HYRAS/HYRAS_precipitation_monthlysum_1931-2020.nc", 
    "E-OBS": loadpath_monthly+"E-OBS/E-OBS_precipitation_monthlysum_1950-2023.nc", 
    "CPC": loadpath_monthly+"CPC/CPC_precipitation_monthlysum_1979-2024.nc", 
    }

data_monthlysum = {}

# load the datasets
print("Loading monthly datasets ...")
for dsname in paths_monthly.keys():
    print("... "+dsname)
    def preprocess_imerg(ds):
        # function to transform to accumulated monthly values
        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        ds["precipitation"] = ds["precipitation"]*days_in_month[ds.time.values[0].month-1]*24
        ds["precipitation"] = ds["precipitation"].assign_attrs(units="mm", Units="mm")
        return ds
    if dsname == "IMERG-V06B-monthly":
        data_monthlysum["IMERG-V06B-monthly"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V06B/global_monthly/3B-MO.MS.MRG.3IMERG.*.V06B.HDF5.nc4', preprocess=preprocess_imerg)\
                        .transpose('time', 'lat', 'lon', ...) # * 24*30 # convert to mm/month (approx)
    elif dsname == "IMERG-V07B-monthly":
        data_monthlysum["IMERG-V07B-monthly"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V07B/global_monthly/3B-MO.MS.MRG.3IMERG.*.V07B.HDF5.nc4', preprocess=preprocess_imerg)\
                        .transpose('time', 'lat', 'lon', ...) # * 24*30 # convert to mm/month (approx)
    elif dsname == "ERA5-monthly":
        def preprocess_era5_totprec(ds):
            # function to transform to accumulated monthly values
            days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
            ds["tp"] = ds["tp"]*days_in_month[pd.Timestamp(ds.time.values[0]).month-1]*1000
            ds["tp"] = ds["tp"].assign_attrs(units="mm", Units="mm")
            return ds
        data_monthlysum["ERA5-monthly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/monthly_averaged/single_level_vars/total_precipitation/total_precipitation_*', 
                                         preprocess=preprocess_era5_totprec)
        data_monthlysum["ERA5-monthly"] = data_monthlysum["ERA5-monthly"].assign_coords(longitude=(((data_monthlysum["ERA5-monthly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data_monthlysum["ERA5-monthly"] = data_monthlysum["ERA5-monthly"].isel(latitude=slice(None, None, -1))
    elif dsname == "GPCC-monthly":
        data_monthlysum["GPCC-monthly"] = xr.open_mfdataset("/automount/ags/jgiles/GPCC/full_data_monthly_v2022/025/*.nc")
        data_monthlysum["GPCC-monthly"] = data_monthlysum["GPCC-monthly"].isel(lat=slice(None, None, -1))
        data_monthlysum["GPCC-monthly"]["precip"] = data_monthlysum["GPCC-monthly"]["precip"].assign_attrs(units="mm", Units="mm")
    else:
        data_monthlysum[dsname] = xr.open_dataset(paths_monthly[dsname])

# Special tweaks
print("Applying tweaks ...")
# RADOLAN GRID AND CRS
if "RADOLAN" in data_monthlysum.keys():
    lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
    data_monthlysum["RADOLAN"] = data_monthlysum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
    data_monthlysum["RADOLAN"] = data_monthlysum["RADOLAN"].assign(crs=data_monthlysum['RADKLIM'].crs[0])
    data_monthlysum["RADOLAN"].attrs["grid_mapping"] = "crs"
    data_monthlysum["RADOLAN"].lon.attrs = data_monthlysum["RADKLIM"].lon.attrs
    data_monthlysum["RADOLAN"].lat.attrs = data_monthlysum["RADKLIM"].lat.attrs

# EURADCLIM coords
if "EURADCLIM" in data_monthlysum.keys():
    data_monthlysum["EURADCLIM"] = data_monthlysum["EURADCLIM"].set_coords(("lon", "lat"))

# Shift HYRAS and EURADCLIM timeaxis
if "EURADCLIM" in data_monthlysum.keys():
    data_monthlysum["EURADCLIM"]["time"] = data_monthlysum["EURADCLIM"]["time"].resample(time="MS").first()["time"] # We place the monthly value at month start
if "HYRAS" in data_monthlysum.keys():
    data_monthlysum["HYRAS"]["time"] = data_monthlysum["HYRAS"]["time"].resample(time="MS").first()["time"] # We place the monthly value at month start

# Convert all non datetime axes (cf Julian calendars) into datetime 
for dsname in data_monthlysum.keys():
    try:
        data_monthlysum[dsname]["time"] = data_monthlysum[dsname].indexes['time'].to_datetimeindex()
        print(dsname+" time dimension transformed to datetime format")
    except:
        pass
    
# Special selections for incomplete extreme years
# IMERG
if "IMERG-V07B-monthly" in data_monthlysum.keys():
    data_monthlysum["IMERG-V07B-monthly"] = data_monthlysum["IMERG-V07B-monthly"].loc[{"time":slice("2001", "2022")}]
if "IMERG-V06B-monthly" in data_monthlysum.keys():
    data_monthlysum["IMERG-V06B-monthly"] = data_monthlysum["IMERG-V06B-monthly"].loc[{"time":slice("2001", "2020")}]
# CMORPH
if "CMORPH-daily" in data_monthlysum.keys():
    data_monthlysum["CMORPH-daily"] = data_monthlysum["CMORPH-daily"].loc[{"time":slice("1998", "2022")}]
# GPROF
if "GPROF" in data_monthlysum.keys():
    data_monthlysum["GPROF"] = data_monthlysum["GPROF"].loc[{"time":slice("2015", "2022")}]
# CPC
if "CPC" in data_monthlysum.keys():
    data_monthlysum["CPC"] = data_monthlysum["CPC"].loc[{"time":slice("1979", "2023")}]

colors = {
    "IMERG-V07B-monthly": "#FF6347", # Tomato
    "IMERG-V06B-monthly": "crimson", # crimson
    "CMORPH-daily": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", # DodgerBlue
    "ERA5-monthly": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

#%%% Regional averages
#%%%% Calculate area means (regional averages)
data_to_avg = data_monthlysum.copy() # select which data to average (yearly, monthly, daily...)

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland", 
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra", 
         "Monaco", "Czechia", "Slovakia", "Hungary", "Slovenia", "Romania"]#"land"
region = "Germany"
region_name = "Germany" # name for plots
mask = utils.get_regionmask(region)
TSMP_nudge_margin = 13 # number of gridpoints to mask out the relaxation zone at the margins

# TSMP-case: we make a specific mask to cut out the edge of the european domain + country
dsname = "TSMP-DETECT-Baseline"
mask_TSMP_nudge = False
if dsname in data_to_avg.keys():
    mask_TSMP_nudge = True # This will be used later as a trigger for this extra mask
    lon_bot = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lon.values
    lat_bot = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lat.values
    lon_top = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lon.values
    lat_top = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lat.values
    lon_right = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lon.values
    lat_right = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lat.values
    lon_left = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lon.values
    lat_left = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lat.values
    
    lon_tsmp_edge = np.concatenate((lon_bot, lon_right, lon_top[::-1], lon_left[::-1]))
    lat_tsmp_edge = np.concatenate((lat_bot, lat_right, lat_top[::-1], lat_left[::-1]))
    
    lonlat_tsmp_edge = list(zip(lon_tsmp_edge, lat_tsmp_edge))
    
    TSMP_no_nudge = rm.Regions([ lonlat_tsmp_edge ], names=["TSMP_no_nudge"], abbrevs=["TSMP_NE"], name="TSMP")
    # I did not find a way to directly combine this custom region with a predefined country region. I will 
    # have to just apply the masks consecutively

data_avgreg = {}
# Means over region
print("Calculating means over "+region_name)
to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():
    print("... "+dsname)

    if dsname in ["RADOLAN", "RADKLIM", "HYRAS", "EURADCLIM"]:
        # these datasets come in equal-pixel-sized grids, so we only need to apply the average over the region
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        with ProgressBar():
            data_avgreg[dsname] = data_to_avg[dsname].where(mask0.notnull()).mean(("x", "y")).compute()

    if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly", "CMORPH-daily", "ERA5-monthly", 
                  "GPCC-monthly", "GPCC-daily", "GPROF", "E-OBS", "CPC"]:
        # these datasets come in regular lat-lon grids, so we need to average over the region considering the area weights
        variables_to_include = [vv for vv in data_to_avg[dsname].data_vars \
                                if "lonv" not in data_to_avg[dsname][vv].dims \
                                if "latv" not in data_to_avg[dsname][vv].dims \
                                if "nv" not in data_to_avg[dsname][vv].dims]
        mask0 = mask.mask(data_to_avg[dsname])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(data_to_avg[dsname]).where(mask0.notnull())
        if dsname in ["ERA5-monthly", "E-OBS"]:
            with ProgressBar():
                data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                              lon_name="longitude", lat_name="latitude").compute()
        else:
            with ProgressBar():
                data_avgreg[dsname] = utils.calc_spatial_mean(data_to_avg[dsname][variables_to_include].where(mask0.notnull()), 
                                                              lon_name="lon", lat_name="lat").compute()

    if dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
        # we need to unrotate the TSMP grid and then average over the region considering the area weights

        grid_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
        regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]), grid_out, "conservative")
        to_add[dsname+"-EURregLonLat01deg"] = regridder(data_to_avg[dsname])
        
        mask0 = mask.mask(to_add[dsname+"-EURregLonLat01deg"])
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(to_add[dsname+"-EURregLonLat01deg"]).where(mask0.notnull())

        with ProgressBar():
            data_avgreg[dsname] = utils.calc_spatial_mean(to_add[dsname+"-EURregLonLat01deg"].where(mask0.notnull()), 
                                                          lon_name="lon", lat_name="lat").compute()
        
# add the rotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}
data_monthlysum = data_to_avg.copy()

#%%%% Simple map plot
dsname = "GPCC-monthly"
vname = "precip"
tsel = "2015-02"
mask = utils.get_regionmask(region)
mask0 = mask.mask(data_monthlysum[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(data_monthlysum[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_monthlysum[dsname][vname].sel(time=tsel).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", cmap="Blues", vmin=0, vmax=1000, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
if mask_TSMP_nudge: ax1.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(dsname)

#%%%% Simple map plot (for number of stations per gridcell) # CHECK THIS FOR MONTHLY SUM BEFORE RUNNING!!
dsname = "GPCC-monthly"
vname = "numgauge"
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Turkey/"

for yy in np.arange(2000,2021):
    ysel = str(yy)
    mask = utils.get_regionmask(region)
    mask0 = mask.mask(data_monthlysum[dsname])
    dropna = True
    if mask_TSMP_nudge: 
        mask0 = TSMP_no_nudge.mask(data_monthlysum[dsname]).where(mask0.notnull())
        dropna=False
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    cmap1 = copy.copy(plt.cm.Blues)
    cmap1.set_under("lightgray")
    plot = (data_monthlysum[dsname][vname].sel(time=ysel)/12).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", 
                                            levels=3, cmap=cmap1, vmin=1, vmax=3, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                             cbar_kwargs={'label': "", 'shrink':0.88})
    if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    plt.title(dsname+" number of stations per gridcell "+ysel)
    # save figure
    savepath_yy = savepath+ysel+"/"
    if not os.path.exists(savepath_yy):
        os.makedirs(savepath_yy)
    filename = "numgauge_"+region_name+"_"+dsname+"_"+ysel+".png"
    plt.savefig(savepath_yy+filename, bbox_inches="tight")
    # plt.show()

#%%%% Interannual variability area-means plot
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

selmonthlist = [("Jan", [1]),
           ("Jul", [7]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/"


for selmonth in selmonthlist:
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
                else: 
                    color = colors[dsname]
                try:
                    plt.plot(data_avgreg[dsname]['time'].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])), 
                             data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])), 
                             label=dsname, c=color, marker=marker)
                except TypeError:
                    # try to change the time coord to datetime format
                    plt.plot(data_avgreg[dsname].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])).indexes['time'].to_datetimeindex(), 
                             data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])), 
                             label=dsname, c=color, marker=marker)
                plotted = True
        if not plotted:
            raise Warning("Nothing plotted for "+dsname)
    
    plt.legend(ncols=3, fontsize=7)
    plt.title("Area-mean "+selmonth[0]+" total precip "+region_name+" [mm]")
    plt.xlim(datetime(2000,1,1), datetime(2020,1,1))
    # plt.xlim(2000, 2020)
    plt.grid()
    # save figure
    savepath_month = savepath+selmonth[0]+"/lineplots/"
    if not os.path.exists(savepath_month):
        os.makedirs(savepath_month)
    filename = "area_mean_"+selmonth[0]+"_total_precip_"+region_name+".png"
    plt.savefig(savepath_month+filename, bbox_inches="tight")
    plt.close()

#%%%% Interannual variability area-means plot (interactive html)

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

selmonthlist = [("Jan", [1]),
           ("Jul", [7]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference (black and bold curve)

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/"


for selmonth in selmonthlist:
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
                                    data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])).hvplot.line(x='time', label=dsname).opts(color=color, line_width=lw, show_legend=True, muted_alpha=0)
                        )
                
                else:
                    hvplots.append(
                                    data_avgreg[dsname][vv].sel(time=data_avgreg[dsname]['time'].dt.month.isin(selmonth[1])).hvplot.line(x='time', label=dsname).opts(show_legend=True, muted_alpha=0)
                        )
    
                plotted = True
        if not plotted:
            raise Warning("Nothing plotted for "+dsname)
    
    layout = hvplots[0]
    for nplot in np.arange(1, len(hvplots)):
        layout = layout * hvplots[nplot]
    
    layout.opts(title="Area-mean "+selmonth[0]+" total precip "+region_name+" [mm]", xlabel="Time", show_grid=True, legend_position='right',
                height=600, width=1200)
    
    # Save to HTML file
    savepath_month = savepath+selmonth[0]+"/lineplots/"
    if not os.path.exists(savepath_month):
        os.makedirs(savepath_month)
    filename = "area_mean_"+selmonth[0]+"_total_precip_"+region_name+".html"
    hv.save(layout, savepath_month+filename)

#%%% BIAS and ERRORS
#%%%% Bias (absolute and relative) calculation from regional averages
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC"]

dsignore = [] # datasets to ignore in the plotting
dsref = ["GPCC-monthly"] # dataset to take as reference

data_to_bias = data_avgreg

data_bias = {}
data_bias_relative = {}
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    data_bias[dsname] = data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref]
                    data_bias_relative[dsname] = (data_to_bias[dsname][vv] - data_to_bias[dsref[0]][vvref])/data_to_bias[dsref[0]][vvref]*100
                    break

#%%%% Region-averages bias bar plot

selmonthlist = [("Jan", [1]),
           ("Jul", [7]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/"

for selmonth in selmonthlist:
    # Calculate bar width based on the number of data arrays
    bar_width = 0.8 / len(data_bias)
        
    # Get time values
    time_values_ref = data_to_bias[dsref[0]]['time'].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selmonth[1]))
    
    # Plotting each DataArray in the dictionary
    plt.figure(figsize=(20, 6))  # Adjust figure size as needed
    for idx, (key, value) in enumerate(data_bias.items()):
        value_padded = value.sel(time=value['time'].dt.month.isin(selmonth[1])).broadcast_like(data_to_bias[dsref[0]].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selmonth[1])))
        time_values = value_padded['time']
        bar_positions = np.arange(len(time_values)) + idx * bar_width
        plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])
    
    plt.xlabel('Time')
    plt.ylabel(value.attrs['units'])
    plt.title("Area-mean "+selmonth[0]+" total precip BIAS with respect to "+dsref[0]+" "+region_name)
    plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # save figure
    savepath_month = savepath+selmonth[0]+"/barplots/"
    if not os.path.exists(savepath_month):
        os.makedirs(savepath_month)
    filename = "area_mean_"+selmonth[0]+"_precip_totals_bias_"+region_name+".png"
    plt.savefig(savepath_month+filename, bbox_inches="tight")
    plt.close()

    # Plotting each DataArray in the dictionary (same but for relative bias)
    plt.figure(figsize=(20, 6))  # Adjust figure size as needed
    for idx, (key, value) in enumerate(data_bias_relative.items()):
        value_padded = value.sel(time=value['time'].dt.month.isin(selmonth[1])).broadcast_like(data_to_bias[dsref[0]].sel(time=data_to_bias[dsref[0]]['time'].dt.month.isin(selmonth[1])))
        time_values = value_padded['time']
        bar_positions = np.arange(len(time_values)) + idx * bar_width
        plt.bar(bar_positions, value_padded, width=bar_width, label=key, color=colors[key])
    
    plt.xlabel('Time')
    plt.ylabel("%")
    plt.title("Area-mean "+selmonth[0]+" total precip RELATIVE BIAS with respect to "+dsref[0]+" "+region_name)
    plt.xticks(np.arange(len(time_values)) + 0.4, time_values_ref.dt.year.values, rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # save figure
    savepath_month = savepath+selmonth[0]+"/barplots/"
    if not os.path.exists(savepath_month):
        os.makedirs(savepath_month)
    filename = "area_mean_"+selmonth[0]+"_precip_totals_bias_relative_"+region_name+".png"
    plt.savefig(savepath_month+filename, bbox_inches="tight")
    plt.close()

#%%%% Relative bias and errors calculation (at gridpoint level, not with the area means)
# First we need to transform EURADCLIM, RADKLIM, RADOLAN and HYRAS to regular grids
# We use the DETECT 1 km grid for this (actually, I did not manage to make that work)
lonlims = slice(TSMP_no_nudge.bounds_global[0], TSMP_no_nudge.bounds_global[2])
latlims = slice(TSMP_no_nudge.bounds_global[1], TSMP_no_nudge.bounds_global[3])

to_add = {} # dictionary to add regridded versions
for dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
    if dsname not in data_monthlysum: continue
    print("Regridding "+dsname+" ...")

    grid_out = xe.util.cf_grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    grid_out = xe.util.grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
    # # I tried to use dask for the weight generation to avoid memory crash
    # # but I did not manage to make it work: https://xesmf.readthedocs.io/en/latest/notebooks/Dask.html

    # grid_out = grid_out.chunk({"x": 50, "y": 50, "x_b": 50, "y_b": 50,})

    # # we then try parallel regridding: slower but less memory-intensive (this takes forever)
    # regridder = xe.Regridder(data_monthlysum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True)
    # to_add[dsname+"-EURregLonLat001deg"] = regridder(data_to_avg[dsname])
    # regridder.to_netcdf() # we save the weights
    # # to reuse the weigths:
    # xe.Regridder(data_monthlysum[dsname].cf.add_bounds(["lon", "lat"]), 
    #                           grid_out, 
    #                           "conservative", parallel=True, weights="/path/to/weights") #!!! Can I use CDO weights here?
    # Cdo().gencon()
    # cdo gencon,/automount/ags/jgiles/IMERG_V06B/global_monthly/griddes.txt -setgrid,/automount/agradar/jgiles/TSMP/griddes_mod.txt /automount/agradar/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020.nc weights_to_IMERG.nc

    # Instead, just regrid to the reference dataset grid (this is fast)

    regridder = xe.Regridder(data_monthlysum[dsname].cf.add_bounds(["lon", "lat"]), 
                             data_monthlysum[dsref[0]].loc[{"lon": lonlims, "lat": latlims}], 
                             "conservative")
    to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_monthlysum[dsname])
    
# add the regridded datasets to the original dictionary
data_monthlysum = {**data_monthlysum, **to_add}
    
# Compute the biases
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline'] # datasets to ignore (because we want the regridded version)
data_to_bias = copy.copy(data_monthlysum)

to_add = {} # dictionary to add regridded versions

data_bias_map = {} # maps of yearly biases
data_bias_relative_map = {} # maps of yearly relative biases
data_abs_error_map = {} # maps of yearly absolute errors
data_bias_relative_gp = {} # yearly relative biases on a gridpoint basis (sum of gridpoint biases divided by sum of reference-data values)
data_mean_abs_error_gp = {} # yearly MAE on a gridpoint basis (sum of gridpoint abs errors divided by number of data values)
data_norm_mean_abs_error_gp = {} # yearly NMAE on a gridpoint basis (sum of gridpoint abs errors divided by sum of reference-data values)
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    print("Processing "+dsname+" ...")
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            for vvref in var_names:
                if vvref in data_to_bias[dsref[0]].data_vars:
                    if dsref[0]+"-grid" not in dsname: # if no regridded already, do it now
                        if "longitude" in data_to_bias[dsname].coords or "latitude" in data_to_bias[dsname].coords:
                            # if the names of the coords are longitude and latitude, change them to lon, lat
                            data_to_bias[dsname] = data_to_bias[dsname].rename({"longitude":"lon", "latitude":"lat"})
                        
                        if dsname in ["IMERG-V07B-monthly", "IMERG-V06B-monthly"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bnds", "lat_bnds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])
                            
                        if dsname in ["CMORPH-daily"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bounds", "lat_bounds"])
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        if dsname in ["GPROF", "TSMP-old-EURregLonLat01deg", "TSMP-DETECT-Baseline-EURregLonLat01deg"]:
                            # we need to remove the default defined bounds or the regridding will fail
                            del(data_to_bias[dsname].lon.attrs["bounds"])
                            del(data_to_bias[dsname].lat.attrs["bounds"])

                        # regridder = xe.Regridder(data_to_bias[dsname].cf.add_bounds(["lon", "lat"]), data_to_bias[dsref[0]], "conservative")

                        regridder = xe.Regridder(data_to_bias[dsname], 
                                                 data_to_bias[dsref[0]].loc[{"lon": lonlims, "lat": latlims}], 
                                                 "conservative")

                        to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_to_bias[dsname][vv])
                        data0 = to_add[dsname+"_"+dsref[0]+"-grid"].copy()
                    else:
                        data0 = data_to_bias[dsname][vv].copy()

                    data0 = data0.where(data0>0)
                    dataref = data_to_bias[dsref[0]][vvref].loc[{"lon": lonlims, "lat": latlims}]
                    
                    mask0 = mask.mask(dataref)
                    if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

                    data_bias_map[dsname] = ( data0 - dataref ).compute()
                    data_bias_relative_map[dsname] = ( data_bias_map[dsname] / dataref ).compute() *100
                    data_bias_map_masked = data_bias_map[dsname].where(mask0.notnull())
                    data_bias_relative_map_masked = data_bias_relative_map[dsname].where(mask0.notnull())
                    data_abs_error_map[dsname] = abs(data_bias_map[dsname])
                    data_abs_error_map_masked = data_abs_error_map[dsname].where(mask0.notnull())
                    
                    data_bias_relative_gp[dsname] = utils.calc_spatial_integral(data_bias_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100
                    
                    data_norm_mean_abs_error_gp[dsname] = utils.calc_spatial_integral(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask0.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100

                    data_mean_abs_error_gp[dsname] = utils.calc_spatial_mean(data_abs_error_map_masked,
                                                lon_name="lon", lat_name="lat").compute()
                    
                    break

# add the regridded datasets to the original dictionary
data_monthlysum = {**data_monthlysum, **to_add}

#%%%% Relative bias and error plots
#%%%%% Simple map plot
# region = "Germany" #"land" 
to_plot = data_bias_map
dsname = "IMERG-V07B-monthly"
title = "BIAS"
yearsel = "2016-02"
cbarlabel = "mm" # mm
vmin = -250
vmax = 250
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
mask = utils.get_regionmask(region)
mask0 = mask.mask(to_plot[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":yearsel}].where(mask0.notnull(), drop=dropna).loc[{"lon":lonlat_slice[0], 
                                                                                      "lat":lonlat_slice[1]}].plot(x="lon", 
                                                                                                                   y="lat", 
                                                                                                                   cmap="RdBu_r", 
                                                                                    vmin=vmin, vmax=vmax, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+yearsel+"\n"+dsname+"\n "+region_name+" Ref.: "+dsref[0])

#%%%%% Simple map plot (loop)
# Like previous but for saving all plots !! THIS WILL PROBABLY CRASH THE MEMORY
# region = "Germany" #"land" 
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/months/"+region_name+"/"
period = np.arange(2000,2024)
to_plot_dict = [
            (data_bias_map, "BIAS", "mm", -75, 75),
            (data_bias_relative_map, "RELATIVE BIAS", "%", -75, 75),
           ]
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]

selmonthlist = [("Jan", "01"),
           ("Jul", "07"),
           ] # ("nameofmonth", "month")

for to_plot, title, cbarlabel, vmin, vmax in to_plot_dict:
    print("Plotting "+title)
    for dsname in to_plot.keys():
        print("... "+dsname)
        dsname_short = dsname.split("_")[0]
        mask = utils.get_regionmask(region)
        mask0 = mask.mask(to_plot[dsname])
        dropna = True
        if not mask_TSMP_nudge:  # Remove/add "not" here to change the extent of the map
            mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
            dropna=False
        for yearsel in period:
            for selmonth in selmonthlist:
                try:
                    plt.close()
                    yearseas = str(yearsel)+"-"+str(selmonth[1])
                    # f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
                    plot = to_plot[dsname].loc[{"time":yearseas}].where(mask0.notnull(), 
                                                                            drop=dropna).loc[{"lon":lonlat_slice[0],
                                                                                              "lat":lonlat_slice[1]}].plot(x="lon",
                                                                                                                           y="lat",
                                                                                                                           cmap="RdBu_r", 
                                                                                                        vmin=vmin, vmax=vmax, 
                                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                                             cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
                    # if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
                    plot.axes.coastlines(alpha=0.7)
                    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
                    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
                    plt.title(title+" "+str(yearsel)+" "+selmonth[0]+"\n"+dsname_short+"\n "+region_name+" Ref.: "+dsref[0])
                    
                    # save figure
                    savepath_yy = savepath+str(yearsel)+"/"+selmonth[0]+"/"
                    if not os.path.exists(savepath_yy):
                        os.makedirs(savepath_yy)
                    filename = "_".join([title.lower().replace(" ","_"), region_name, dsname_short,dsref[0],str(yearsel),selmonth[0]])+".png"
                    plt.savefig(savepath_yy+filename, bbox_inches="tight")
                    plt.close()
                except KeyError:
                    continue

#%%%%% Box plots of BIAS and ERRORS
# the box plots are made up of the yearly bias or error values, and the datasets are ordered according to their median
to_plot0 = data_bias_relative_gp.copy() # data_mean_abs_error_gp # data_bias_relative_gp # data_norm_mean_abs_error_gp
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/boxplots/relative_bias/"
savefilename = "boxplot_relative_bias_" # the season name will be added at the end of savefilename
title = "relative bias "+region_name+". Ref.: "+dsref[0] # the season name will be added at the beginning of title
ylabel = "%" # % # mm
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# Override the previous with a loop for each case
savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/boxplots/"
for to_plot0, savepath, savefilename, title, ylabel in [
        (data_bias_relative_gp.copy(), 
         savepathbase + "relative_bias/",
         "boxplot_relative_bias_",
         "relative bias "+region_name+". Ref.: "+dsref[0],
         "%"),
        (data_mean_abs_error_gp.copy(), 
         savepathbase + "mean_absolute_error/",
         "boxplot_mean_absolute_error_",
         "mean absolute error "+region_name+". Ref.: "+dsref[0],
         "mm"),
        (data_norm_mean_abs_error_gp.copy(), 
         savepathbase + "normalized_mean_absolute_error/",
         "boxplot_normalized_mean_absolute_error_",
         "normalized mean absolute error "+region_name+". Ref.: "+dsref[0],
         "%"),
        ]:

    tsel = [
            slice(None, None), # if I want to consider only certain period. Otherwise set to (None, None). Multiple options possible
            slice("2001-01-01", "2020-01-01"), 
            slice("2006-01-01", "2020-01-01"), 
            slice("2013-01-01", "2020-01-01"), 
            slice("2015-01-01", "2020-01-01")
            ] 
    ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))
    selmonthlist = [("Jan", [1]),
               ("Jul", [7]),
               ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])
    
    for selmonth in selmonthlist:
        savepathn = (region_name+"/"+selmonth[0]).join(copy.deepcopy(savepath).split(region_name))
        for tseln in tsel:
            to_plot = to_plot0.copy()
            savefilenamen = copy.deepcopy(savefilename)
            titlen = copy.deepcopy(title)
            titlen = selmonth[0]+" "+titlen
            savefilenamen = savefilenamen+selmonth[0]
            
            if tseln.start is not None and tseln.stop is not None: # add specific period to title
                titlen = titlen+". "+tseln.start+" - "+tseln.stop
                savefilenamen = savefilenamen+"_"+tseln.start+"-"+tseln.stop
                
                if ignore_incomplete:
                    for key in to_plot.copy().keys():
                        if not (to_plot[key].time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot[key].time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            del(to_plot[key])
            
            # Select the given season
            for key in to_plot.copy().keys():
                to_plot[key] = to_plot[key].sel(time=to_plot[key]['time'].dt.month.isin(selmonth[1]))
            
            # Initialize a figure and axis
            plt.close()
            plt.figure(figsize=(1.25*(len(to_plot.keys())-len(dsignore)), 6))
            ax = plt.subplot(111)
            
            # Create a list to hold the data arrays
            plotted_arrays = []
            plotted_arrays_lengths = []
            
            # Order according to median:
            to_plot = dict(sorted(to_plot.items(), key=lambda item: item[1].sel(time=tseln).median()))
            
            # Iterate over the datasets in the dictionary
            for key, value in to_plot.items():
                if key not in dsignore:
                    # Plot a box plot for each dataset
                    value = value.sel(time=tseln)
                    plotted_arrays.append(value.values) # values of each box
                    plotted_arrays_lengths.append(len(value)) # number of values in each box
                    ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6, 
                               patch_artist=True, boxprops=dict(facecolor='#b6d6e3'),
                               medianprops=dict(color="#20788e", lw=2))
            
            # Set x-axis ticks and labels with dataset names
            ax.set_xticks(range(1, len(plotted_arrays) + 1))
            ax.set_xticklabels([dsname.split("_")[0] if "_" in dsname 
                                else "-".join(dsname.split("-")[:-1]) if "EURreg" in dsname 
                                else dsname 
                                for dsname in 
                                [ds for ds in to_plot.keys() if ds not in dsignore]
                                ],
                               rotation=45, fontsize=15)
            ax.xaxis.label.set_size(15)     # change xlabel size
            ax.yaxis.label.set_size(15)     # change ylabel size
            
            ax.tick_params(axis='x', labelsize=15) # change xtick label size
            ax.tick_params(axis='y', labelsize=15) # change xtick label size
            
            # Make a secondary x axis to display the number of values in each box
            ax2 = ax.secondary_xaxis('top')
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("top")
            
            ax2.set_xticks(range(1, len(plotted_arrays) + 1))
            ax2.set_xticklabels(plotted_arrays_lengths)
            ax2.set_xlabel('Number of years', fontsize= 15)
            
            # Set labels and title
            #ax.set_xlabel('')
            ax.set_ylabel(ylabel)
            ax.set_title(titlen, fontsize=20)
            
            # plot a reference line at zero
            plt.hlines(y=0, xmin=0, xmax=len(plotted_arrays)+1, colors='black', lw=2, zorder=0)
            plt.xlim(0.5, len(plotted_arrays) + 0.5)
            
            # Show the plot
            plt.grid(True)
            plt.tight_layout()
            if not os.path.exists(savepathn):
                os.makedirs(savepathn)
            plt.savefig(savepathn+savefilenamen+".png", bbox_inches="tight")
            plt.close()

#%%%%% Taylor diagram
# The Taylor diagram can be done by computing the stats over all gridpoints and all timesteps (spatiotemporal)
# or only doing the stats over space or time separately (for these, either temporal or spatial averages must be done first)
    
#%%%%%% Compute stats and plot for all seasons
import skill_metrics as sm
# https://github.com/PeterRochford/SkillMetrics/blob/master/Examples/taylor10.py#L123
# I cannot use skill_metrics to calculate the stats because they do not filter out 
# nan values (because of the masks) so the result is erroneous. They also do not handle weighted arrays.

mode = "" # if "spatial" then average in time and compute the diagram in space. Viceversa for "temporal"
dsref = ["GPCC-monthly"]
data_to_stat = data_monthlysum

# choose common period (only datasets that cover the whole period are included)
tslice = slice("2015","2020") # this covers all
# tslice = slice("2013","2020") # this excludes GPROF
# tslice = slice("2006","2020") # this excludes GPROF and EURADCLIM
tslice = slice("2001","2020") # this excludes GPROF, EURADCLIM and RADOLAN

selmonthlist = [("Jan", [1]),
           ("Jul", [7]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

# Override options above and sweep over them in a loop
for mode in ["", "spatial", "temporal"]:
    for tslice in [
        slice("2015","2020"), # this covers all
        slice("2013","2020"), # this excludes GPROF
        slice("2006","2020"), # this excludes GPROF and EURADCLIM
        slice("2001","2020"), # this excludes GPROF, EURADCLIM and RADOLAN
            ]:

        print("Plotting Taylor diagrams (mode: "+mode+", "+tslice.start+"-"+tslice.stop+")...")
        for selmonth in selmonthlist:
            print("... "+selmonth[0])
            
            ccoef = dict()
            crmsd = dict()
            sdev = dict()
            
            for vv in var_names: # get the name of the desired variable in the reference dataset
                if vv in data_to_stat[dsref[0]]:
                    ref_var_name = vv
                    break
            
            # Get reference dataset
            ds_ref = data_to_stat[dsref[0]][ref_var_name].sel(time=data_to_stat[dsref[0]]['time'].dt.month.isin(selmonth[1]))
            
            # Get area weights
            try:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["lat","lon"]].coords)
            except AttributeError:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["latitude","longitude"]].coords)
            
            # Get mask
            mask = utils.get_regionmask(region)
            mask_ref = mask.mask(ds_ref)
            if mask_TSMP_nudge: mask_ref = TSMP_no_nudge.mask(ds_ref).where(mask_ref.notnull())
            ds_ref = ds_ref.where(mask_ref.notnull())#.mean(tuple([cn for cn in ds_ref.coords if cn!="time"]))
            
            # Normalize weights in the mask
            weights = weights.where(mask_ref.notnull(), other=0.)/weights.where(mask_ref.notnull(), other=0.).sum()
            
            for dsname in data_to_stat.keys(): # compute the stats
                if dsref[0]+"-grid" in dsname or dsname==dsref[0]:
                    # get dataset
                    if type(data_to_stat[dsname]) is xr.DataArray:
                        ds_n = data_to_stat[dsname].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selmonth[1])).where(mask_ref.notnull())
                    else:
                        for vv in var_names:
                            if vv in data_to_stat[dsname]:
                                ds_n = data_to_stat[dsname][vv].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selmonth[1])).where(mask_ref.notnull())
                                break
            
                    # Subset period
                    tslice_array = ds_ref.sel(time=tslice).time
            
                    ds_ref_tsel = ds_ref.sel(time=tslice_array)
                    try:
                        ds_n_tsel = ds_n.sel(time=tslice_array)
                    except KeyError:
                        print(dsname+" ignored because it does not cover the selected time period")
                        continue
                    
                    # Reduce in case mode is "spatial" or "temporal"
                    if mode=="spatial":
                        ds_ref_tsel = ds_ref_tsel.mean("time")
                        ds_n_tsel = ds_n_tsel.mean("time")
                        mode_name="Spatial"
                    elif mode=="temporal":
                        ds_ref_tsel = ds_ref_tsel.weighted(weights).mean([cn for cn in ds_ref_tsel.dims if cn!="time"])
                        ds_n_tsel = ds_n_tsel.weighted(weights).mean([cn for cn in ds_n_tsel.dims if cn!="time"])
                        mode_name="Temporal"
                    else:
                        mode_name="Spatiotemporal"
                    
                    if mode=="temporal":
                        # Get Correlation Coefficient (ccoef)
                            
                        ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.std()
                    else:
                        # Get Correlation Coefficient (ccoef)
                
                        # could work like this but I have to update xarray to include the weights
                        # ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel, weigths=weights )
                        
                        ccoef[dsname] = xr.corr(ds_n_tsel*weights, ds_ref_tsel*weights).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.weighted(weights).sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.weighted(weights).std()
            
            # Plot the diagram
            savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/"+selmonth[0]+"/taylor_diagrams/"
            savefilename = "taylor_"+selmonth[0]+"_precip_totals_"+region_name+"_"+tslice.start+"-"+tslice.stop+".png"
            if mode != "":
                savefilename = mode+"_"+savefilename
            '''
            Specify individual marker label (key), label color, symbol, size, symbol face color, 
            symbol edge color
            '''
            # Define colors for each group
            color_gauges = "k"
            color_radar = "r"
            color_satellite = "b"
            color_reanalysis = "m"
            color_model = "c"
            
            # Define marker size
            markersize = 7
            
            MARKERS = {
                "GPCC-monthly": {
                    "labelColor": "k",
                    "symbol": "+",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "HYRAS": {
                    "labelColor": "k",
                    "symbol": "o",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "E-OBS": {
                    "labelColor": "k",
                    "symbol": "D",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "CPC": {
                    "labelColor": "k",
                    "symbol": "X",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "EURADCLIM": {
                    "labelColor": "k",
                    "symbol": "^",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADOLAN": {
                    "labelColor": "k",
                    "symbol": "s",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADKLIM": {
                    "labelColor": "k",
                    "symbol": "v",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "IMERG-V07B-monthly": {
                    "labelColor": "k",
                    "symbol": "d",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "IMERG-V06B-monthly": {
                    "labelColor": "k",
                    "symbol": "<",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "CMORPH-daily": {
                    "labelColor": "k",
                    "symbol": ">",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "GPROF": {
                    "labelColor": "k",
                    "symbol": "p",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "ERA5-monthly": {
                    "labelColor": "k",
                    "symbol": "*",
                    "size": markersize,
                    "faceColor": color_reanalysis,
                    "edgeColor": color_reanalysis,
                },
                "TSMP-old-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "h",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
                "TSMP-DETECT-Baseline-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "8",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
            }
            
            
            # Set the stats in arrays like the plotting function wants them (the reference first)
            
            lccoef = ccoef[dsref[0]].round(3).values # we round the reference so it does not go over 1
            lcrmsd = crmsd[dsref[0]].values
            lsdev = sdev[dsref[0]].values
            labels = [dsref[0]]
            
            for dsname in MARKERS.keys():
                dsname_grid = dsname+"_"+dsref[0]+"-grid"
                if dsname_grid in ccoef.keys():
                    lccoef = np.append(lccoef, ccoef[dsname_grid].values)
                    lcrmsd = np.append(lcrmsd, crmsd[dsname_grid].values)
                    lsdev = np.append(lsdev, sdev[dsname_grid].values)
                    labels.append(dsname_grid.split("_")[0])
            
            # Must set figure size here to prevent legend from being cut off
            plt.close()
            plt.figure(num=1, figsize=(8, 6))
            
            sm.taylor_diagram(lsdev,lcrmsd,lccoef, markerLabel = labels, #markerLabelColor = 'r', 
                                      markerLegend = 'on', markerColor = 'r',
                                       colCOR = "black", markers = {k: MARKERS[k] for k in labels[1:]}, 
                                      styleOBS = '-', colOBS = 'r', markerobs = 'o', 
                                      markerSize = 7, #tickRMS = [0.0, 1.0, 2.0, 3.0],
                                      tickRMSangle = 115, showlabelsRMS = 'on',
                                      titleRMS = 'on', titleOBS = 'Ref: '+labels[0],
                                        # checkstats = "on"
                                      )
            
            ax = plt.gca()
            ax.set_title(mode_name+" Taylor Diagram over "+region_name+"\n"+
                         "Area-weighted "+selmonth[0]+" gridded precipitation \n"+
                         str(tslice_array[0].dt.year.values)+"-"+str(tslice_array[-1].dt.year.values),
                         x=1.2, y=1,)
            
            # Create custom legend manually (because otherwise it may end in the wrong place and cannot be specified within skillmetrics)
            handles_legend = []
            labels_legend = []
            
            for labeln, paramn in MARKERS.items():
                if labeln in labels and labeln != labels[0]:
                    handlen = plt.Line2D(
                        [], [],
                        marker=paramn['symbol'],
                        color=paramn['labelColor'],
                        markersize=paramn['size'],
                        markerfacecolor=paramn['faceColor'],
                        markeredgewidth=1.5,
                        markeredgecolor=paramn['edgeColor'],
                        linestyle='None',
                        # axes=ax
                    )
                    handles_legend.append(handlen)
                    labels_legend.append(labeln)
            
            # Place the custom legend
            plt.legend(handles_legend, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
            
            # Save figure
            savepath_seas = savepath
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            plt.savefig(savepath_seas+savefilename, bbox_inches="tight")
            plt.close()
            
            # To check that the equation that defines the diagram is closed (negligible residue)
            sm.check_taylor_stats(lsdev, lcrmsd, lccoef, threshold=1000000000000000000000)
            # 24.05.24: the check does not close but the weighted calculations seem to be fine


#%% Daily analysis

#%%% Load daily datasets

loadpath_daily = "/automount/agradar/jgiles/gridded_data/daily/"
loadpath_agradar = "/automount/agradar/jgiles/"
loadpath_ags = "/automount/ags/jgiles/"
paths_daily = {
    "IMERG-V07B-30min": loadpath_daily+"IMERG-V07B-30min/IMERG-V07B-30min_precipitation_dailysum_*.nc",
    "IMERG-V06B-30min": loadpath_daily+"IMERG-V06B-30min/IMERG-V06B-30min_precipitation_dailysum_*.nc",
    # "CMORPH-daily": loadpath_daily+"this one does not matter",
    "TSMP-old": loadpath_daily+"TSMP-old/TSMP-old_precipitation_dailysum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_daily+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_dailysum_2000-2022.nc",
    "ERA5-hourly": loadpath_daily+"ERA5-hourly/ERA5-hourly_precipitation_dailysum_1999-2021.nc",
    "RADKLIM": loadpath_daily+"RADKLIM/temp_serial/RADKLIM-EURregLonLat001deg_precipitation_dailysum_2001-2022_new_part*.nc",
    "RADOLAN": loadpath_daily+"RADOLAN/RADOLAN-EURregLonLat001deg_precipitation_dailysum_2006-2022.nc",
    "EURADCLIM": loadpath_daily+"EURADCLIM/temp_serial/EURADCLIM-EURregLonLat002deg_precipitation_dailysum_2013-2020_part*.nc",
    # "GPCC-daily": ,
    # "GPROF": loadpath_daily+"this one does not matter",
    "HYRAS": loadpath_daily+"HYRAS/temp_serial/HYRAS-EURregLonLat001deg_precipitation_dailysum_1930-2020_part*.nc", 
    "E-OBS": loadpath_daily+"this one does not matter", 
    # "GSMaP": loadpath_daily+"GSMaP/GSMaP_precipitation_dailysum_*.nc", 
    }

data_dailysum = {}

# load the datasets
print("Loading daily datasets ...")
for dsname in paths_daily.keys():
    print("... "+dsname)
    if dsname == "CMORPH-daily":
        data_dailysum["CMORPH-daily"] = xr.open_mfdataset('/automount/agradar/jgiles/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/*/*/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc') 
        data_dailysum["CMORPH-daily"] = data_dailysum["CMORPH-daily"].assign_coords(lon=(((data_dailysum["CMORPH-daily"].lon + 180) % 360) - 180)).sortby('lon').assign_attrs(units="mm")
        data_dailysum["CMORPH-daily"]["cmorph"] = data_dailysum["CMORPH-daily"]["cmorph"].assign_attrs(units="mm")
    elif dsname == "GPROF":
        gprof_files = sorted(glob.glob("/automount/ags/jgiles/GPM_L3/GPM_3GPROFGPMGMI_DAY.07/20*/*/*HDF5"))
        
        # We need to extract the time dimension from the metadata on each file
        # we create a function for that
        def extract_time(ds):
            # Split the string into key-value pairs
            pairs = [pair.split('=') for pair in ds.FileHeader.strip().split(';\n') if pair]
            
            # Create a dictionary from the key-value pairs
            result_dict = dict((key, value) for key, value in pairs)
            
            # Extract time and set as dim
            date_time = pd.to_datetime(result_dict["StartGranuleDateTime"])
            
            # Add it to ds as dimension
            ds = ds.assign_coords(time=date_time).expand_dims("time")
            
            # Extract if file is empty into a new coordinate
            if result_dict["EmptyGranule"] == "EMPTY":
                ds = ds.assign_coords(empty=np.array(1))
            else:
                ds = ds.assign_coords(empty=np.array(0))
            return ds
        
        gprof_attrs= xr.open_mfdataset(gprof_files, engine="h5netcdf", phony_dims='sort', preprocess=extract_time)
        
        # create a new list of filenames without the empty files
        gprof_files_noempty = [f for n,f in enumerate(gprof_files) if gprof_attrs.empty[n]!=1]
        
        # We create another function for adding the time dim to the dataset
        def add_tdim(ds):
            return ds.expand_dims("time")
        
        # Open the dataset
        data_dailysum["GPROF"] = xr.open_mfdataset(gprof_files_noempty, engine="h5netcdf", group="Grid", 
                                          concat_dim="time", combine="nested", preprocess=add_tdim)
        
        # Set the time coordinate (without empty timesteps)
        data_dailysum["GPROF"] = data_dailysum["GPROF"].assign_coords(
                                {"time": gprof_attrs.time.where(gprof_attrs.empty!=1, drop=True).drop_vars("empty")}
                                )
        
        # Fill the empty timesteps with NaN
        data_dailysum["GPROF"] = data_dailysum["GPROF"].broadcast_like(gprof_attrs.empty, exclude=[dim for dim in data_dailysum["GPROF"].dims if dim !="time"])
        
        # Remove the variables that we don't need
        data_dailysum["GPROF"] = data_dailysum["GPROF"][["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation", "npixPrecipitation", "npixTotal"]]
        
        # We need to transform the data from mm/h to mm/day
        for vv in ["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation"]:
            data_dailysum["GPROF"][vv] = data_dailysum["GPROF"][vv]*24
            data_dailysum["GPROF"][vv] = data_dailysum["GPROF"][vv].assign_attrs(units="mm", Units="mm")
    elif dsname == "E-OBS":
        data_dailysum["E-OBS"] = xr.open_dataset("/automount/ags/jgiles/E-OBS/RR/rr_ens_mean_0.25deg_reg_v29.0e.nc")
    elif dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "GSMaP"]:
        data_dailysum[dsname] = xr.open_mfdataset(paths_daily[dsname])
    else:
        if "*" in paths_daily[dsname]:
            data_dailysum[dsname] = xr.open_mfdataset(paths_daily[dsname])
        else:
            data_dailysum[dsname] = xr.open_dataset(paths_daily[dsname])
        

# Special tweaks
print("Applying tweaks ...")
# RADOLAN GRID AND CRS
if "RADOLAN" in data_dailysum.keys() and "LonLat" not in paths_daily["RADOLAN"]:
    lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
    data_dailysum["RADOLAN"] = data_dailysum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
    data_dailysum["RADOLAN"] = data_dailysum["RADOLAN"].assign(crs=data_dailysum['RADKLIM'].crs[0])
    data_dailysum["RADOLAN"].attrs["grid_mapping"] = "crs"
    data_dailysum["RADOLAN"].lon.attrs = data_dailysum["RADKLIM"].lon.attrs
    data_dailysum["RADOLAN"].lat.attrs = data_dailysum["RADKLIM"].lat.attrs

# EURADCLIM coords
if "EURADCLIM" in data_dailysum.keys():
    data_dailysum["EURADCLIM"] = data_dailysum["EURADCLIM"].set_coords(("lon", "lat"))

# Shift HYRAS and EURADCLIM timeaxis
if "EURADCLIM" in data_dailysum.keys():
    data_dailysum["EURADCLIM"]["time"] = data_dailysum["EURADCLIM"]["time"].resample(time="D").first()["time"] # We place the daily value at day start
if "HYRAS" in data_dailysum.keys(): # HYRAS is shifted 6 h (the sum is from 6 to 6 UTC)
    data_dailysum["HYRAS"]["time"] = data_dailysum["HYRAS"]["time"].resample(time="D").first()["time"] # We place the daily value at day start

# Convert all non datetime axes (cf Julian calendars) into datetime 
for dsname in data_dailysum.keys():
    try:
        data_dailysum[dsname]["time"] = data_dailysum[dsname].indexes['time'].to_datetimeindex()
        print(dsname+" time dimension transformed to datetime format")
    except:
        pass
    
# Rechunk to one chunk per timestep (unless it is already regridded to EURregLonLat)
for dsname in ["RADOLAN", "RADKLIM", "EURADCLIM", "HYRAS"]:
    if dsname in data_dailysum.keys() and "LonLat" not in paths_daily["RADOLAN"]:
        print("Rechunking "+dsname)
        new_chunks = {dim: size for dim, size in data_dailysum[dsname].dims.items()}
        new_chunks["time"] = 1
        try:
            data_dailysum[dsname] = data_dailysum[dsname].chunk(new_chunks)
        except: # rechunking may fail due to some not unified chunks, then try again
            data_dailysum[dsname] = data_dailysum[dsname].unify_chunks().chunk(new_chunks)

# Rename coordinates to lat/lon in case they are not
for dsname in data_dailysum.keys():
    if "longitude" in data_dailysum[dsname].coords:
        data_dailysum[dsname] = data_dailysum[dsname].rename({"longitude": "lon"})
    if "latitude" in data_dailysum[dsname].coords:
        data_dailysum[dsname] = data_dailysum[dsname].rename({"latitude": "lat"})

# Remove "bounds" attribute from lon and lat coords to avoid regridding fails
for dsname in data_dailysum.keys():
    if "lon" in data_dailysum[dsname].coords:
        if "bounds" in data_dailysum[dsname].lon.attrs:
            del(data_dailysum[dsname].lon.attrs["bounds"])
    if "lat" in data_dailysum[dsname].coords:
        if "bounds" in data_dailysum[dsname].lat.attrs:
            del(data_dailysum[dsname].lat.attrs["bounds"])

# Special selections for incomplete extreme years
# IMERG
if "IMERG-V07B-30min" in data_dailysum.keys():
    data_dailysum["IMERG-V07B-30min"] = data_dailysum["IMERG-V07B-30min"].loc[{"time":slice("2001", "2022")}]
if "IMERG-V06B-30min" in data_dailysum.keys():
    data_dailysum["IMERG-V06B-30min"] = data_dailysum["IMERG-V06B-30min"].loc[{"time":slice("2001", "2020")}]
# CMORPH
if "CMORPH-daily" in data_dailysum.keys():
    data_dailysum["CMORPH-daily"] = data_dailysum["CMORPH-daily"].loc[{"time":slice("1998", "2022")}]
# GPROF
if "GPROF" in data_dailysum.keys():
    data_dailysum["GPROF"] = data_dailysum["GPROF"].loc[{"time":slice("2015", "2022")}]
# CPC
if "CPC" in data_dailysum.keys():
    data_dailysum["CPC"] = data_dailysum["CPC"].loc[{"time":slice("1979", "2023")}]

colors = {
    "IMERG-V07B-30min": "#FF6347", # Tomato
    "IMERG-V06B-30min": "crimson", # crimson
    "CMORPH-daily": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", # DodgerBlue
    "ERA5-hourly": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC", "precipitationCal"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-30min", "ERA5-hourly"] # datasets to ignore in the plotting
dsref = ["E-OBS"] # dataset to take as reference (black and bold curve)

#%%%% Define region and regrid
data_to_avg = data_dailysum.copy() # select which data to process

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland", 
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra", 
         "Monaco", "Czechia", "Slovakia", "Hungary", "Slovenia", "Romania"]#"land"
region = "Germany"
region_name = "Germany" # name for plots
mask = utils.get_regionmask(region)
TSMP_nudge_margin = 13 # number of gridpoints to mask out the relaxation zone at the margins

start_time = time.time()

# TSMP-case: we make a specific mask to cut out the edge of the european domain + country
dsname = "TSMP-DETECT-Baseline"
mask_TSMP_nudge = False
if dsname in data_to_avg.keys():
    mask_TSMP_nudge = True # This will be used later as a trigger for this extra mask
    lon_bot = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lon.values
    lat_bot = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][0].lat.values
    lon_top = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lon.values
    lat_top = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][-1].lat.values
    lon_right = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lon.values
    lat_right = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,-1].lat.values
    lon_left = data_to_avg[dsname].lon[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lon.values
    lat_left = data_to_avg[dsname].lat[TSMP_nudge_margin:-TSMP_nudge_margin,TSMP_nudge_margin:-TSMP_nudge_margin][:,0].lat.values
    
    lon_tsmp_edge = np.concatenate((lon_bot, lon_right, lon_top[::-1], lon_left[::-1]))
    lat_tsmp_edge = np.concatenate((lat_bot, lat_right, lat_top[::-1], lat_left[::-1]))
    
    lonlat_tsmp_edge = list(zip(lon_tsmp_edge, lat_tsmp_edge))
    
    TSMP_no_nudge = rm.Regions([ lonlat_tsmp_edge ], names=["TSMP_no_nudge"], abbrevs=["TSMP_NE"], name="TSMP")
    # I did not find a way to directly combine this custom region with a predefined country region. I will 
    # have to just apply the masks consecutively

print("Unrotating datasets")
to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():

    if dsname in ["TSMP-DETECT-Baseline", "TSMP-old"]:
        regsavepath = paths_daily[dsname].rsplit('/', 1)[0] + '/' + paths_daily[dsname].rsplit('/', 1)[1].replace(dsname, dsname+"-EURregLonLat01deg")

        try:
            to_add[dsname+"-EURregLonLat01deg"] = xr.open_dataset(regsavepath)
            print("Previously regridded "+dsname+" was loaded")
        except FileNotFoundError:
            print("... "+dsname)
            # we need to unrotate the TSMP grid 
            
            encoding = {}
            for vv in data_to_avg[dsname].data_vars:
                valid_encodings = ['contiguous', 'complevel', 'compression', 'zlib', '_FillValue', 'shuffle', 'fletcher32', 'dtype', 'least_significant_digit']
                encoding[vv] = dict((k, data_to_avg[dsname][vv].encoding[k]) for k in valid_encodings if k in data_to_avg[dsname][vv].encoding) #!!! For now we keep the same enconding, check later if change it
    
            grid_out = xe.util.cf_grid_2d(-49.75,70.65,0.1,19.85,74.65,0.1) # manually recreate the EURregLonLat01deg grid
            regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]), grid_out, "conservative")
            to_add[dsname+"-EURregLonLat01deg"] = regridder(data_to_avg[dsname])
            to_add[dsname+"-EURregLonLat01deg"].to_netcdf(regsavepath, encoding=encoding)
            to_add[dsname+"-EURregLonLat01deg"] = xr.open_dataset(regsavepath) # reload the dataset

        # Remove "bounds" attribute from lon and lat coords to avoid regridding fails
        if "lon" in to_add[dsname+"-EURregLonLat01deg"].coords:
            if "bounds" in to_add[dsname+"-EURregLonLat01deg"].lon.attrs:
                del(to_add[dsname+"-EURregLonLat01deg"].lon.attrs["bounds"])
        if "lat" in to_add[dsname+"-EURregLonLat01deg"].coords:
            if "bounds" in to_add[dsname+"-EURregLonLat01deg"].lat.attrs:
                del(to_add[dsname+"-EURregLonLat01deg"].lat.attrs["bounds"])

# add the unrotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}
data_dailysum = data_to_avg.copy()

total_time = time.time() - start_time
print(f"Elapsed time: {total_time/60:.2f} minutes.")

#%%%% Simple map plot
dsname = "E-OBS"
vname = "rr"
tsel = "2015-02-01"
mask = utils.get_regionmask(region)
mask0 = mask.mask(data_dailysum[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(data_dailysum[dsname]).where(mask0.notnull())
    # dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_dailysum[dsname][vname].sel(time=tsel).where(mask0.notnull(), drop=dropna).plot(x="longitude", y="latitude", 
                                                                                            cmap="Blues", vmin=0, vmax=10, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# if mask_TSMP_nudge: ax1.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(dsname)

#%%%% Simple map plot (for number of stations per gridcell) # CHECK THIS FOR DAILY SUM BEFORE RUNNING!!
dsname = "GPCC-monthly"
vname = "numgauge"
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Turkey/"

for yy in np.arange(2000,2021):
    ysel = str(yy)
    mask = utils.get_regionmask(region)
    mask0 = mask.mask(data_dailysum[dsname])
    dropna = True
    if mask_TSMP_nudge: 
        mask0 = TSMP_no_nudge.mask(data_dailysum[dsname]).where(mask0.notnull())
        dropna=False
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    cmap1 = copy.copy(plt.cm.Blues)
    cmap1.set_under("lightgray")
    plot = (data_dailysum[dsname][vname].sel(time=ysel)/12).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", 
                                            levels=3, cmap=cmap1, vmin=1, vmax=3, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                             cbar_kwargs={'label': "", 'shrink':0.88})
    if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    plt.title(dsname+" number of stations per gridcell "+ysel)
    # save figure
    savepath_yy = savepath+ysel+"/"
    if not os.path.exists(savepath_yy):
        os.makedirs(savepath_yy)
    filename = "numgauge_"+region_name+"_"+dsname+"_"+ysel+".png"
    plt.savefig(savepath_yy+filename, bbox_inches="tight")
    # plt.show()

#%%% Metrics
#%%%% Metrics calculation at gridpoint level
reload_metrics = True # reload previously calculated metrics if available?
calc_if_no_reload = False # calculate metrics if not available from previously calculated files? #!!! NOT IMPLEMENTED YET!!
minpre = 1 # minimum precipitation in mm/day for a day to be considered a wet day (i.e., minimum measurable precipitation considered). Relevant for categorical metrics
timesel = slice("2016-01-01", "2016-12-31") # should be given in a slice with YYYY-MM-DD
timeperiod = "_".join([timesel.start, timesel.stop])
metricssavepath = "/automount/agradar/jgiles/gridded_data/daily_metrics/"+timeperiod+"/ref_"+dsref[0]+"/"+region_name+"/" # path to save the results of the metrics
tempsavepath = "/automount/agradar/jgiles/gridded_data/daily/temp/ref_"+dsref[0]+"/" # path so save regridded datasets
print("!! Period selected: "+timeperiod+" !!")

# Define a function to set the encoding compression
def define_encoding(data):
    encoding = {}


    for var_name, var in data.data_vars.items():
        # Combine compression and chunking settings
        encoding[var_name] = {"zlib": True, "complevel": 6}

    return encoding

# Check if temporary folder to save the regridded datasets exist, if not then create it
if not os.path.exists(tempsavepath):
    os.makedirs(tempsavepath)

# First we need to transform EURADCLIM, RADKLIM, RADOLAN and HYRAS to regular grids
# We use the DETECT 1 km grid for this (actually, I did not manage to make that work)
lonlims = slice(TSMP_no_nudge.bounds_global[0], TSMP_no_nudge.bounds_global[2])
latlims = slice(TSMP_no_nudge.bounds_global[1], TSMP_no_nudge.bounds_global[3])

to_add = {} # dictionary to add regridded versions
for dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
    if dsname not in data_dailysum: continue
    dstempsavepath = tempsavepath+dsname+"_"+dsref[0]+"-grid.nc"
    try:
        to_add[dsname+"_"+dsref[0]+"-grid"] = xr.open_dataset(dstempsavepath)
        print("Previously regridded "+dsname+" was loaded")
    except FileNotFoundError:
        
        print("Regridding "+dsname+" ...")
    
        grid_out = xe.util.cf_grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
        grid_out = xe.util.grid_2d(-49.746,70.655,0.01,19.854,74.654,0.01) # manually recreate the EURregLonLat001deg grid
        # # I tried to use dask for the weight generation to avoid memory crash
        # # but I did not manage to make it work: https://xesmf.readthedocs.io/en/latest/notebooks/Dask.html
    
        # grid_out = grid_out.chunk({"x": 50, "y": 50, "x_b": 50, "y_b": 50,})
    
        # # we then try parallel regridding: slower but less memory-intensive (this takes forever)
        # regridder = xe.Regridder(data_dailysum[dsname].cf.add_bounds(["lon", "lat"]), 
        #                           grid_out, 
        #                           "conservative", parallel=True)
        # to_add[dsname+"-EURregLonLat001deg"] = regridder(data_to_avg[dsname])
        # regridder.to_netcdf() # we save the weights
        # # to reuse the weigths:
        # xe.Regridder(data_dailysum[dsname].cf.add_bounds(["lon", "lat"]), 
        #                           grid_out, 
        #                           "conservative", parallel=True, weights="/path/to/weights") #!!! Can I use CDO weights here?
        # Cdo().gencon()
        # cdo gencon,/automount/ags/jgiles/IMERG_V06B/global_monthly/griddes.txt -setgrid,/automount/agradar/jgiles/TSMP/griddes_mod.txt /automount/agradar/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020.nc weights_to_IMERG.nc
    
        # Instead, just regrid to the reference dataset grid (this is fast)
        start_time = time.time()

        deltalonref = float(data_dailysum[dsref[0]].lon.diff("lon").median())
        deltalatref = float(data_dailysum[dsref[0]].lat.diff("lat").median())

        lonlims_src = slice(float(data_dailysum[dsname].lon.min())-deltalonref, float(data_dailysum[dsname].lon.max())+deltalonref)
        latlims_src = slice(float(data_dailysum[dsname].lat.min())-deltalatref, float(data_dailysum[dsname].lat.max())+deltalatref)

        regridder = xe.Regridder(data_dailysum[dsname].cf.add_bounds(["lon", "lat"]), 
                                 data_dailysum[dsref[0]].loc[{"lon": lonlims_src, "lat": latlims_src}], 
                                 "conservative")
        to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_dailysum[dsname].chunk(time=365), skipna=True, na_thres=1)
        
        # Save to file
        encoding = define_encoding(to_add[dsname+"_"+dsref[0]+"-grid"])
        to_add[dsname+"_"+dsref[0]+"-grid"].to_netcdf(dstempsavepath, encoding=encoding)

        # Reload
        to_add[dsname+"_"+dsref[0]+"-grid"] = xr.open_dataset(dstempsavepath)
        
        total_time = time.time() - start_time
        print(f"Regridding took {total_time/60:.2f} minutes to run.")

# add the regridded datasets to the original dictionary
data_dailysum = {**data_dailysum, **to_add}
    
# Compute the biases
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline'] # datasets to ignore (because we want the regridded version)
data_to_bias = copy.copy(data_dailysum)

to_add = {} # dictionary to add regridded versions

start_time = time.time()

metrics = {} # dictionary to store all metrics in full form (all gridpoints and all timesteps)
metrics_spatial = {} # dictionary to store all metrics averaged in time (one map)
metrics_temporal = {} # dictionary to store all metrics averaged in space (timeseries)
metrics_spatem = {} # dictionary to store all metrics averaged in space and time (one value per dataset)
metric_types = ["bias", "absolute_error", "bias_concurrent", "absolute_error_concurrent", "relative_bias_concurrent"]
metric_types2 = ["bias", "mae", "nmae", "pod", "far", "csi", "biass", "bias_concurrent", "relative_bias_concurrent", "mae_concurrent", "nmae_concurrent"]
for metric_type in metric_types:
    metrics[metric_type] = dict()
for metric_type in metric_types2:
    metrics_spatial[metric_type] = dict()
    metrics_temporal[metric_type] = dict()
    metrics_spatem[metric_type] = dict()

# We first define the reference dataset
for vvref in var_names:
    if vvref in data_to_bias[dsref[0]].data_vars:
        dataref = data_to_bias[dsref[0]][vvref].loc[{"lon": lonlims, "lat": latlims}].chunk(time=100)
        mask0 = mask.mask(dataref)
        if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

# We then loop through all the datasets we want to process
for dsname in data_to_bias.keys():
    if dsname in dsignore+dsref:
        continue
    # check that timesel is fully covered by the dsname
    if not (pd.to_datetime(timesel.start) in data_to_bias[dsname].time and pd.to_datetime(timesel.stop) in data_to_bias[dsname].time):
        print(dsname+" ignored because it does not cover the selected time period")
        continue
    print("Processing "+dsname+" ...")
    for vv in var_names:
        if vv in data_to_bias[dsname].data_vars:
            if dsref[0]+"-grid" not in dsname: # if no regridded already, do it now
                dstempsavepath = tempsavepath+dsname+"_"+dsref[0]+"-grid.nc"
                try:
                    to_add[dsname+"_"+dsref[0]+"-grid"] = xr.open_dataset(dstempsavepath)[vv]
                    print("Previously regridded "+dsname+" was loaded")
                except FileNotFoundError:
                    if "longitude" in data_to_bias[dsname].coords or "latitude" in data_to_bias[dsname].coords:
                        # if the names of the coords are longitude and latitude, change them to lon, lat
                        data_to_bias[dsname] = data_to_bias[dsname].rename({"longitude":"lon", "latitude":"lat"})
                    
                    if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min"]:
                        if "lon_bnds" in data_to_bias[dsname]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bnds", "lat_bnds"])
                        
                    if dsname in ["CMORPH-daily"]:
                        if "lon_bounds" in data_to_bias[dsname]:
                            # we need to remove the default defined bounds or the regridding will fail
                            data_to_bias[dsname] = data_to_bias[dsname].drop_vars(["lon_bounds", "lat_bounds"])

                    # regridder = xe.Regridder(data_to_bias[dsname].cf.add_bounds(["lon", "lat"]), data_to_bias[dsref[0]], "conservative")
                    print("... regridding")
                    reg_start_time = time.time()

                    regridder = xe.Regridder(data_to_bias[dsname], 
                                             data_to_bias[dsref[0]].loc[{"lon": lonlims, "lat": latlims}], 
                                             "conservative")

                    to_add[dsname+"_"+dsref[0]+"-grid"] = regridder(data_to_bias[dsname][vv], 
                                                                    skipna=True, na_thres=1).to_dataset(name=vv)
                    
                    # Save to file
                    encoding = define_encoding(to_add[dsname+"_"+dsref[0]+"-grid"])
                    to_add[dsname+"_"+dsref[0]+"-grid"].to_netcdf(dstempsavepath, encoding=encoding)
            
                    # Reload
                    to_add[dsname+"_"+dsref[0]+"-grid"] = xr.open_dataset(dstempsavepath)[vv]
                    
                    reg_total_time = time.time() - reg_start_time
                    print(f"... ... took {reg_total_time/60:.2f} minutes.")

                data0 = to_add[dsname+"_"+dsref[0]+"-grid"].copy()
                
                dsname_metric = dsname

            else:
                data0 = data_to_bias[dsname][vv].copy()
                
                dsname_metric = dsname.split("_")[0]

            data0 = data0.loc[{"time":timesel}].chunk(time=100)
            # data0 = data0.where(data0>0) # do not filter out the zero values, otherwise we will not check for detection correcly
            
            # Reload metrics
            if reload_metrics:
                print("... reloading metrics")
                for metric_type in metric_types:
                    try:
                        metric_path = "/".join([metricssavepath, metric_type, dsname_metric, "_".join([metric_type,dsname_metric])+".nc"])
                        metrics[metric_type][dsname_metric] = xr.open_dataset(metric_path)
                    except:
                        print("... ... metric "+metric_type+" could not be loaded!!")
                for metric_type in metric_types2:
                    try: # try spatem
                        metric_path = "/".join([metricssavepath, "spatem_"+metric_type, dsname_metric, "_".join(["spatem_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatem[metric_type][dsname_metric] = xr.open_dataset(metric_path)
                    except:
                        print("... ... metric spatem "+metric_type+" could not be loaded!!")
                    try: # try spatial
                        metric_path = "/".join([metricssavepath, "spatial_"+metric_type, dsname_metric, "_".join(["spatial_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatem[metric_type][dsname_metric] = xr.open_dataset(metric_path)
                    except:
                        print("... ... metric spatial "+metric_type+" could not be loaded!!")
                    try: # try temporal
                        metric_path = "/".join([metricssavepath, "temporal_"+metric_type, dsname_metric, "_".join(["temporal_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatem[metric_type][dsname_metric] = xr.open_dataset(metric_path)
                    except:
                        print("... ... metric temporal "+metric_type+" could not be loaded!!")
            else:
                # Calculte metrics
                print("... calculating metrics")
                met_start_time = time.time()

                # BIAS
                print("... ... BIAS")
                metrics["bias"][dsname_metric] = ( data0.where(mask0.notnull(), drop=True) - dataref )

                metrics_spatial["bias"][dsname_metric] = metrics["bias"][dsname_metric].mean("time").compute()

                metrics_temporal["bias"][dsname_metric] = utils.calc_spatial_mean(metrics["bias"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute()

                metrics_spatem["bias"][dsname_metric] = metrics_temporal["bias"][dsname_metric].mean("time").compute()
                
                # # relative BIAS (this may not be so useful due to divisions by zero)
                # print("... ... relative BIAS")
                # metrics["relative_bias"][dsname_metric] = ( metrics["bias"][dsname_metric] / dataref )*100

                # metrics_spatial["relative_bias"][dsname_metric] = metrics["relative_bias"][dsname_metric].mean("time").compute()

                # metrics_temporal["relative_bias"][dsname_metric] = utils.calc_spatial_mean(metrics["relative_bias"][dsname_metric],
                #                             lon_name="lon", lat_name="lat").compute()

                # metrics_spatem["relative_bias"][dsname_metric] = metrics_temporal["relative_bias"][dsname_metric].mean("time").compute()

                # AE
                print("... ... AE")
                metrics["absolute_error"][dsname_metric] = abs( metrics["bias"][dsname_metric] )

                # MAE
                print("... ... MAE")
                metrics_spatial["mae"][dsname_metric] = metrics["absolute_error"][dsname_metric].mean("time").compute()

                metrics_temporal["mae"][dsname_metric] = utils.calc_spatial_mean(metrics["absolute_error"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute()

                metrics_spatem["mae"][dsname_metric] = metrics_temporal["mae"][dsname_metric].mean("time").compute()

                # NMAE
                print("... ... NMAE")
                metrics_spatial["nmae"][dsname_metric] = ( metrics["absolute_error"][dsname_metric].sum("time", min_count=1) /
                                                   dataref.sum("time")
                                                   ).compute() *100

                metrics_temporal["nmae"][dsname_metric] = utils.calc_spatial_integral(metrics["absolute_error"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute() / \
                                                utils.calc_spatial_integral(dataref.where(mask0.notnull(), drop=True),
                                            lon_name="lon", lat_name="lat").compute() *100

                metrics_spatem["nmae"][dsname_metric] = utils.calc_spatial_integral(metrics["absolute_error"][dsname_metric],
                                            lon_name="lon", lat_name="lat").sum("time").compute() / \
                                                utils.calc_spatial_integral(dataref.where(mask0.notnull(), drop=True),
                                            lon_name="lon", lat_name="lat").sum("time").compute() *100

                # For the categorical metrics we need the following
                print("... ... Categorical metrics")
                data0_wet = (data0 > minpre).astype(bool)
                dataref_wet = (dataref > minpre).astype(bool).where(dataref.notnull())
                
                hits = data0_wet*dataref_wet.where(mask0.notnull(), drop=True)
                misses = (~data0_wet)*dataref_wet.where(mask0.notnull(), drop=True)
                false_alarms = data0_wet*(~dataref_wet).where(mask0.notnull(), drop=True)
                
                hits_spatial = hits.sum("time").compute()
                hits_temporal = hits.sum(("lon", "lat")).compute()
                hits_spatem = hits_temporal.sum("time").compute()

                misses_spatial = misses.sum("time").compute()
                misses_temporal = misses.sum(("lon", "lat")).compute()
                misses_spatem = misses_temporal.sum("time").compute()

                false_alarms_spatial = false_alarms.sum("time").compute()
                false_alarms_temporal = false_alarms.sum(("lon", "lat")).compute()
                false_alarms_spatem = false_alarms_temporal.sum("time").compute()

                # POD
                metrics_spatial["pod"][dsname_metric] = hits_spatial / (hits_spatial + misses_spatial)

                metrics_temporal["pod"][dsname_metric] = hits_temporal / (hits_temporal + misses_temporal)

                metrics_spatem["pod"][dsname_metric] = hits_spatem / (hits_spatem + misses_spatem)
                
                # FAR/POFA
                metrics_spatial["far"][dsname_metric] = false_alarms_spatial / (hits_spatial + false_alarms_spatial)

                metrics_temporal["far"][dsname_metric] = false_alarms_temporal / (hits_temporal + false_alarms_temporal)

                metrics_spatem["far"][dsname_metric] = false_alarms_spatem / (hits_spatem + false_alarms_spatem)
                
                # CSI
                metrics_spatial["csi"][dsname_metric] = hits_spatial / (hits_spatial + misses_spatial + false_alarms_spatial)

                metrics_temporal["csi"][dsname_metric] = hits_temporal / (hits_temporal + misses_temporal + false_alarms_temporal)

                metrics_spatem["csi"][dsname_metric] = hits_spatem / (hits_spatem + misses_spatem + false_alarms_spatem)

                # BIASS
                metrics_spatial["biass"][dsname_metric] = (hits_spatial + false_alarms_spatial) / (hits_spatial + misses_spatial)

                metrics_temporal["biass"][dsname_metric] = (hits_temporal + false_alarms_temporal) / (hits_temporal + misses_temporal)

                metrics_spatem["biass"][dsname_metric] = (hits_spatem + false_alarms_spatem) / (hits_spatem + misses_spatem)

                # Calculate the same initial metrics but for concurrent events (i.e., correcly detected events)
                # BIAS
                print("... ... concurrent BIAS")
                metrics["bias_concurrent"][dsname_metric] = ( data0.where(mask0.notnull(), drop=True) - dataref ).where(hits>0)

                metrics_spatial["bias_concurrent"][dsname_metric] = metrics["bias_concurrent"][dsname_metric].mean("time").compute()

                metrics_temporal["bias_concurrent"][dsname_metric] = utils.calc_spatial_mean(metrics["bias_concurrent"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute()

                metrics_spatem["bias_concurrent"][dsname_metric] = metrics_temporal["bias_concurrent"][dsname_metric].mean("time").compute()
                
                # relative BIAS (this may not be so useful due to divisions by zero)
                print("... ... concurrent relative BIAS")
                metrics["relative_bias_concurrent"][dsname_metric] = ( metrics["bias_concurrent"][dsname_metric] / dataref )*100

                metrics_spatial["relative_bias_concurrent"][dsname_metric] = metrics["relative_bias_concurrent"][dsname_metric].mean("time").compute()

                metrics_temporal["relative_bias_concurrent"][dsname_metric] = utils.calc_spatial_mean(metrics["relative_bias_concurrent"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute()

                metrics_spatem["relative_bias_concurrent"][dsname_metric] = metrics_temporal["relative_bias_concurrent"][dsname_metric].mean("time").compute()

                # AE
                print("... ... concurrent AE")
                metrics["absolute_error_concurrent"][dsname_metric] = abs( metrics["bias_concurrent"][dsname_metric] )

                # MAE
                print("... ... concurrent MAE")
                metrics_spatial["mae_concurrent"][dsname_metric] = metrics["absolute_error_concurrent"][dsname_metric].mean("time").compute()

                metrics_temporal["mae_concurrent"][dsname_metric] = utils.calc_spatial_mean(metrics["absolute_error_concurrent"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute()

                metrics_spatem["mae_concurrent"][dsname_metric] = metrics_temporal["mae_concurrent"][dsname_metric].mean("time").compute()

                # NMAE
                print("... ... concurrent NMAE")
                metrics_spatial["nmae_concurrent"][dsname_metric] = ( metrics["absolute_error_concurrent"][dsname_metric].sum("time", min_count=1) /
                                                   dataref.sum("time")
                                                   ).compute() *100

                metrics_temporal["nmae_concurrent"][dsname_metric] = utils.calc_spatial_integral(metrics["absolute_error_concurrent"][dsname_metric],
                                            lon_name="lon", lat_name="lat").compute() / \
                                                utils.calc_spatial_integral(dataref.where(mask0.notnull(), drop=True),
                                            lon_name="lon", lat_name="lat").compute() *100

                metrics_spatem["nmae_concurrent"][dsname_metric] = utils.calc_spatial_integral(metrics["absolute_error_concurrent"][dsname_metric],
                                            lon_name="lon", lat_name="lat").sum("time").compute() / \
                                                utils.calc_spatial_integral(dataref.where(mask0.notnull(), drop=True),
                                            lon_name="lon", lat_name="lat").sum("time").compute() *100

                met_total_time = time.time() - met_start_time
                print(f"... ... took {met_total_time/60:.2f} minutes.")

            break

# add the regridded datasets to the original dictionary
data_dailysum = {**data_dailysum, **to_add}
total_time = time.time() - start_time
print(f"Calculating metrics took {total_time/60:.2f} minutes to run.")

if not reload_metrics:
    # Save the metrics to files
    print("Saving the metrics to files")
    # Save metrics to files
    for metric_type, datasets in metrics.items():
        metric_dir = os.path.join(metricssavepath, metric_type)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
    
        for dsname, metric_data in datasets.items():
            ds_dir = os.path.join(metric_dir, dsname)
            if not os.path.exists(ds_dir):
                os.makedirs(ds_dir)
    
            # Convert the data to an xarray Dataset and save as NetCDF
            metric_dataset = xr.Dataset({f"{metric_type}": metric_data})
            metric_filename = os.path.join(ds_dir, f"{metric_type}_{dsname}.nc")
            metric_dataset.to_netcdf(metric_filename, encoding={metric_type: {"zlib":True, "complevel":6}})
            print(f"Saved {metric_type} for {dsname} to {metric_filename}")
    
    # Repeat the same for spatial, temporal, and spatio-temporal metrics
    for metric_dict, name in zip([metrics_spatial, metrics_temporal, metrics_spatem],
                                 ['spatial', 'temporal', 'spatem']):
        for metric_type, datasets in metric_dict.items():
            metric_dir = os.path.join(metricssavepath, f"{name}_{metric_type}")
            if not os.path.exists(metric_dir):
                os.makedirs(metric_dir)
    
            for dsname, metric_data in datasets.items():
                ds_dir = os.path.join(metric_dir, dsname)
                if not os.path.exists(ds_dir):
                    os.makedirs(ds_dir)
    
                # Convert the data to an xarray Dataset and save as NetCDF
                metric_dataset = xr.Dataset({f"{name}_{metric_type}": metric_data})
                metric_filename = os.path.join(ds_dir, f"{name}_{metric_type}_{dsname}.nc")
                metric_dataset.to_netcdf(metric_filename, encoding={f"{name}_{metric_type}": {"zlib":True, "complevel":6}})
                print(f"Saved {name}_{metric_type} for {dsname} to {metric_filename}")

#%%%% Metrics plots
#%%%%% Simple map plot
# region = "Germany" #"land" 
to_plot = metrics["bias"]
dsname = "IMERG-V07B-30min"
title = "BIAS"
timesel0 = "2016-06-01"
cbarlabel = "mm" # mm
vmin = -10
vmax = 10
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
mask = utils.get_regionmask(region)
mask0 = mask.mask(to_plot[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
    # dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":timesel0}].where(mask0.notnull(), drop=dropna).loc[{"lon":lonlat_slice[0], 
                                                                                      "lat":lonlat_slice[1]}].plot(x="lon", 
                                                                                                                   y="lat", 
                                                                                                                   cmap="RdBu_r", 
                                                                                    vmin=vmin, vmax=vmax, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
# if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+timesel0+"\n"+dsname+"\n "+region_name+" Ref.: "+dsref[0])

#%%%%% Plots in space (maps)
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/daily/"+region_name+"/maps/"
to_plot_dict = [
            (metrics_spatial['bias'], "BIAS", "mm", -1, 1, "BrBG"),
            (metrics_spatial['mae'], "MAE", "mm", 0, 3, "Reds"),
            (metrics_spatial['nmae'], "NMAE", "%", 0, 10, "Reds"),
            (metrics_spatial['pod'], "POD", "", 0.6, 1, "Blues"),
            (metrics_spatial['far'], "FAR", "", 0, 0.5, "Reds"),
            (metrics_spatial['csi'], "CSI", "", 0.4, 1, "Blues"),
            (metrics_spatial['biass'], "BIASS", "", 0.7, 1.3, "BrBG"),
            (metrics_spatial['bias_concurrent'], "Concurrent BIAS", "mm", -1, 1, "BrBG"),
            (metrics_spatial['relative_bias_concurrent'], "Concurrent relative BIAS", "%", -10, 10, "BrBG"),
            (metrics_spatial['mae_concurrent'], "Concurrent MAE", "mm", 0, 3, "Reds"),
            (metrics_spatial['nmae_concurrent'], "Concurrent NMAE", "%", 0, 10, "Reds"),
           ]
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]

for to_plot, title, cbarlabel, vmin, vmax, cmap in to_plot_dict:
    print("Plotting "+title)
    for dsname in to_plot.keys():
        print("... "+dsname)
        dsname_short = dsname.split("_")[0]
        try:
            plt.close()
            # f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
            plot = to_plot[dsname].plot(x="lon", y="lat", cmap=cmap, 
                                            vmin=vmin, vmax=vmax, 
                                                     subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                                     cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
            plot.axes.coastlines(alpha=0.7)
            plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
            plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
            plt.title(title+" "+timeperiod+"\n"+dsname_short+"\n "+region_name+" Ref.: "+dsref[0])
            
            # save figure
            savepath_period = savepath+timeperiod+"/"
            if not os.path.exists(savepath_period):
                os.makedirs(savepath_period)
            filename = "_".join([title.lower().replace(" ","_"), region_name, dsname_short,dsref[0],timeperiod])+".png"
            plt.savefig(savepath_period+filename, bbox_inches="tight")
            plt.close()
        except KeyError:
            continue

#%%%%% Box plots
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# Override the previous with a loop for each case
savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/daily/"+region_name+"/boxplots/"

to_plot0 = metrics_temporal

to_plot_dict = [ # name, title, units, reference_line, vmin, vmax, cmap
            ('bias', "BIAS", "mm", 0, -1, 1, "BrBG"), # vmin, vmax and cmap are not necessary
            ('mae', "MAE", "mm", 0, 0, 3, "Reds"),
            ('nmae', "NMAE", "%", 0, 0, 10, "Reds"),
            ('pod', "POD", "", 1, 0.6, 1, "Blues"),
            ('far', "FAR", "", 0, 0, 0.5, "Reds"),
            ('csi', "CSI", "", 1, 0.4, 1, "Blues"),
            ('biass', "BIASS", "", 1, 0.7, 1.3, "BrBG"),
            ('bias_concurrent', "Concurrent BIAS", "mm", 0, -1, 1, "BrBG"),
            ('relative_bias_concurrent', "Concurrent relative BIAS", "%", 0, -10, 10, "BrBG"),
            ('mae_concurrent', "Concurrent MAE", "mm", 0, 0, 3, "Reds"),
            ('nmae_concurrent', "Concurrent NMAE", "%", 0, 0, 10, "Reds"),
           ]

tsel = [timesel] #!!! Improvement idea: the temporal metrics could be calculated for a longer period that could then be cut down here, and the spatial and spatem metrics could be calculated for different periods all at once based on subsets
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [("DJF" , [12, 1, 2]),
           ("JJA", [6, 7, 8]),
           ("", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

print("Plotting boxplots ...")
for metric_type, title, ylabel, reference_line, vmin, vmax, cmap in to_plot_dict:
    print("... "+metric_type)
    for selmonth in selmonthlist:
        for tseln in tsel:
            to_plot = to_plot0[metric_type].copy()
            timeperiodn = timeperiod
            if tseln.start is not None and tseln.stop is not None: # add specific period to title
                timeperiodn = "_".join([timesel.start, timesel.stop])
                
                if ignore_incomplete:
                    for key in to_plot.copy().keys():
                        if not (to_plot[key].time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot[key].time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            del(to_plot[key])
                
            # Select the given season
            for key in to_plot.copy().keys():
                to_plot[key] = to_plot[key].sel(time=to_plot[key]['time'].dt.month.isin(selmonth[1]))
            
            # Initialize a figure and axis
            plt.close()
            plt.figure(figsize=(1.25*(len(to_plot.keys())-len(dsignore)), 6))
            ax = plt.subplot(111)
            
            # Create a list to hold the data arrays
            plotted_arrays = []
            plotted_arrays_lengths = []
            
            # Order according to median:
            to_plot = dict(sorted(to_plot.items(), key=lambda item: item[1].sel(time=tseln).median()))
            
            # Iterate over the datasets in the dictionary
            for key, value in to_plot.items():
                if key not in dsignore:
                    # Plot a box plot for each dataset
                    value = value.sel(time=tseln).dropna("time")
                    plotted_arrays.append(value.values) # values of each box
                    plotted_arrays_lengths.append(len(value)) # number of values in each box
                    ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6, 
                                patch_artist=True, boxprops=dict(facecolor='#b6d6e3'), showfliers=False,
                                medianprops=dict(color="#20788e", lw=2))
                    # Add the spatem value as another line (like the median)
                    ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6, 
                                patch_artist=True, boxprops=dict(facecolor='#b6d6e3'), showfliers=False, showbox=False, showcaps=False, 
                                usermedians=[float(metrics_spatem[metric_type][key])],
                                medianprops=dict(color="crimson", lw=2, ls=":"))
            
            # Set x-axis ticks and labels with dataset names
            ax.set_xticks(range(1, len(plotted_arrays) + 1))
            ax.set_xticklabels([dsname.split("_")[0] if "_" in dsname 
                                else "-".join(dsname.split("-")[:-1]) if "EURreg" in dsname 
                                else dsname 
                                for dsname in 
                                [ds for ds in to_plot.keys() if ds not in dsignore]
                                ],
                               rotation=45, fontsize=15)
            ax.xaxis.label.set_size(15)     # change xlabel size
            ax.yaxis.label.set_size(15)     # change ylabel size
            
            ax.tick_params(axis='x', labelsize=15) # change xtick label size
            ax.tick_params(axis='y', labelsize=15) # change xtick label size
            
            # # Make a secondary x axis to display the number of values in each box
            # ax2 = ax.secondary_xaxis('top')
            # ax2.xaxis.set_ticks_position("bottom")
            # ax2.xaxis.set_label_position("top")
            
            # ax2.set_xticks(range(1, len(plotted_arrays) + 1))
            # ax2.set_xticklabels(plotted_arrays_lengths)
            # ax2.set_xlabel('Number of years', fontsize= 15)
            
            # Set labels and title
            titlen = " ".join([selmonth[0], title, region_name+".", "Ref.: "+dsref[0]+".", timeperiodn])
            #ax.set_xlabel('')
            ax.set_ylabel(ylabel)
            ax.set_title(titlen, fontsize=20)
            
            # plot a reference line
            plt.hlines(y=reference_line, xmin=0, xmax=len(plotted_arrays)+1, colors='black', lw=2, zorder=0)
            
            plt.xlim(0.5, len(plotted_arrays) + 0.5)
            
            # Show and save the plot
            plt.grid(True)
            plt.tight_layout()
            
            savepathn = savepathbase+timeperiodn+"/"+metric_type+"/"
            savefilenamen = "_".join(["boxplot", metric_type, selmonth[0], timeperiodn])+".png"
            if not os.path.exists(savepathn):
                os.makedirs(savepathn)
            plt.savefig(savepathn+savefilenamen+".png", bbox_inches="tight")
            plt.close()

#%%%%% Taylor diagram
# The Taylor diagram can be done by computing the stats over all gridpoints and all timesteps (spatiotemporal)
# or only doing the stats over space or time separately (for these, either temporal or spatial averages must be done first)
    
#%%%%%% Compute stats and plot for all seasons
import skill_metrics as sm
# https://github.com/PeterRochford/SkillMetrics/blob/master/Examples/taylor10.py#L123
# I cannot use skill_metrics to calculate the stats because they do not filter out 
# nan values (because of the masks) so the result is erroneous. They also do not handle weighted arrays.

mode = "" # if "spatial" then average in time and compute the diagram in space. Viceversa for "temporal"
dsref = ["GPCC-monthly"]
data_to_stat = data_monthlysum

# choose common period (only datasets that cover the whole period are included)
tslice = slice("2015","2020") # this covers all
# tslice = slice("2013","2020") # this excludes GPROF
# tslice = slice("2006","2020") # this excludes GPROF and EURADCLIM
tslice = slice("2001","2020") # this excludes GPROF, EURADCLIM and RADOLAN

selmonthlist = [("Jan", [1]),
           ("Jul", [7]),
           ("full", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

# Override options above and sweep over them in a loop
for mode in ["", "spatial", "temporal"]:
    for tslice in [
        slice("2015","2020"), # this covers all
        slice("2013","2020"), # this excludes GPROF
        slice("2006","2020"), # this excludes GPROF and EURADCLIM
        slice("2001","2020"), # this excludes GPROF, EURADCLIM and RADOLAN
            ]:

        print("Plotting Taylor diagrams (mode: "+mode+", "+tslice.start+"-"+tslice.stop+")...")
        for selmonth in selmonthlist:
            print("... "+selmonth[0])
            
            ccoef = dict()
            crmsd = dict()
            sdev = dict()
            
            for vv in var_names: # get the name of the desired variable in the reference dataset
                if vv in data_to_stat[dsref[0]]:
                    ref_var_name = vv
                    break
            
            # Get reference dataset
            ds_ref = data_to_stat[dsref[0]][ref_var_name].sel(time=data_to_stat[dsref[0]]['time'].dt.month.isin(selmonth[1]))
            
            # Get area weights
            try:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["lat","lon"]].coords)
            except AttributeError:
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.lon.values, ds_ref.lat.values),
                                       coords=ds_ref.to_dataset()[["latitude","longitude"]].coords)
            
            # Get mask
            mask = utils.get_regionmask(region)
            mask_ref = mask.mask(ds_ref)
            if mask_TSMP_nudge: mask_ref = TSMP_no_nudge.mask(ds_ref).where(mask_ref.notnull())
            ds_ref = ds_ref.where(mask_ref.notnull())#.mean(tuple([cn for cn in ds_ref.coords if cn!="time"]))
            
            # Normalize weights in the mask
            weights = weights.where(mask_ref.notnull(), other=0.)/weights.where(mask_ref.notnull(), other=0.).sum()
            
            for dsname in data_to_stat.keys(): # compute the stats
                if dsref[0]+"-grid" in dsname or dsname==dsref[0]:
                    # get dataset
                    if type(data_to_stat[dsname]) is xr.DataArray:
                        ds_n = data_to_stat[dsname].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selmonth[1])).where(mask_ref.notnull())
                    else:
                        for vv in var_names:
                            if vv in data_to_stat[dsname]:
                                ds_n = data_to_stat[dsname][vv].sel(time=data_to_stat[dsname]['time'].dt.month.isin(selmonth[1])).where(mask_ref.notnull())
                                break
            
                    # Subset period
                    tslice_array = ds_ref.sel(time=tslice).time
            
                    ds_ref_tsel = ds_ref.sel(time=tslice_array)
                    try:
                        ds_n_tsel = ds_n.sel(time=tslice_array)
                    except KeyError:
                        print(dsname+" ignored because it does not cover the selected time period")
                        continue
                    
                    # Reduce in case mode is "spatial" or "temporal"
                    if mode=="spatial":
                        ds_ref_tsel = ds_ref_tsel.mean("time")
                        ds_n_tsel = ds_n_tsel.mean("time")
                        mode_name="Spatial"
                    elif mode=="temporal":
                        ds_ref_tsel = ds_ref_tsel.weighted(weights).mean([cn for cn in ds_ref_tsel.dims if cn!="time"])
                        ds_n_tsel = ds_n_tsel.weighted(weights).mean([cn for cn in ds_n_tsel.dims if cn!="time"])
                        mode_name="Temporal"
                    else:
                        mode_name="Spatiotemporal"
                    
                    if mode=="temporal":
                        # Get Correlation Coefficient (ccoef)
                            
                        ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.std()
                    else:
                        # Get Correlation Coefficient (ccoef)
                
                        # could work like this but I have to update xarray to include the weights
                        # ccoef[dsname] = xr.corr(ds_n_tsel, ds_ref_tsel, weigths=weights )
                        
                        ccoef[dsname] = xr.corr(ds_n_tsel*weights, ds_ref_tsel*weights).compute()
                        
                        # Get Centered Root-Mean-Square-Deviation (CRMSD)
                
                        crmsd_0 = ( (ds_n_tsel - ds_n_tsel.mean() ) - 
                                    (ds_ref_tsel - ds_ref_tsel.mean()) )**2
                        crmsd_1 = crmsd_0.weighted(weights).sum()/xr.ones_like(crmsd_0).where(crmsd_0.notnull()).sum()
                        crmsd[dsname] = np.sqrt(crmsd_1)
                                        
                        # Get Standard Deviation (SDEV)
                        
                        sdev[dsname] = ds_n_tsel.weighted(weights).std()
            
            # Plot the diagram
            savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/monthly/"+region_name+"/"+selmonth[0]+"/taylor_diagrams/"
            savefilename = "taylor_"+selmonth[0]+"_precip_totals_"+region_name+"_"+tslice.start+"-"+tslice.stop+".png"
            if mode != "":
                savefilename = mode+"_"+savefilename
            '''
            Specify individual marker label (key), label color, symbol, size, symbol face color, 
            symbol edge color
            '''
            # Define colors for each group
            color_gauges = "k"
            color_radar = "r"
            color_satellite = "b"
            color_reanalysis = "m"
            color_model = "c"
            
            # Define marker size
            markersize = 7
            
            MARKERS = {
                "GPCC-monthly": {
                    "labelColor": "k",
                    "symbol": "+",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "HYRAS": {
                    "labelColor": "k",
                    "symbol": "o",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "E-OBS": {
                    "labelColor": "k",
                    "symbol": "D",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "CPC": {
                    "labelColor": "k",
                    "symbol": "X",
                    "size": markersize,
                    "faceColor": color_gauges,
                    "edgeColor": color_gauges,
                },
                "EURADCLIM": {
                    "labelColor": "k",
                    "symbol": "^",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADOLAN": {
                    "labelColor": "k",
                    "symbol": "s",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "RADKLIM": {
                    "labelColor": "k",
                    "symbol": "v",
                    "size": markersize,
                    "faceColor": color_radar,
                    "edgeColor": color_radar,
                },
                "IMERG-V07B-monthly": {
                    "labelColor": "k",
                    "symbol": "d",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "IMERG-V06B-monthly": {
                    "labelColor": "k",
                    "symbol": "<",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "CMORPH-daily": {
                    "labelColor": "k",
                    "symbol": ">",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "GPROF": {
                    "labelColor": "k",
                    "symbol": "p",
                    "size": markersize,
                    "faceColor": color_satellite,
                    "edgeColor": color_satellite,
                },
                "ERA5-monthly": {
                    "labelColor": "k",
                    "symbol": "*",
                    "size": markersize,
                    "faceColor": color_reanalysis,
                    "edgeColor": color_reanalysis,
                },
                "TSMP-old-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "h",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
                "TSMP-DETECT-Baseline-EURregLonLat01deg": {
                    "labelColor": "k",
                    "symbol": "8",
                    "size": markersize,
                    "faceColor": color_model,
                    "edgeColor": color_model,
                },
            }
            
            
            # Set the stats in arrays like the plotting function wants them (the reference first)
            
            lccoef = ccoef[dsref[0]].round(3).values # we round the reference so it does not go over 1
            lcrmsd = crmsd[dsref[0]].values
            lsdev = sdev[dsref[0]].values
            labels = [dsref[0]]
            
            for dsname in MARKERS.keys():
                dsname_grid = dsname+"_"+dsref[0]+"-grid"
                if dsname_grid in ccoef.keys():
                    lccoef = np.append(lccoef, ccoef[dsname_grid].values)
                    lcrmsd = np.append(lcrmsd, crmsd[dsname_grid].values)
                    lsdev = np.append(lsdev, sdev[dsname_grid].values)
                    labels.append(dsname_grid.split("_")[0])
            
            # Must set figure size here to prevent legend from being cut off
            plt.close()
            plt.figure(num=1, figsize=(8, 6))
            
            sm.taylor_diagram(lsdev,lcrmsd,lccoef, markerLabel = labels, #markerLabelColor = 'r', 
                                      markerLegend = 'on', markerColor = 'r',
                                       colCOR = "black", markers = {k: MARKERS[k] for k in labels[1:]}, 
                                      styleOBS = '-', colOBS = 'r', markerobs = 'o', 
                                      markerSize = 7, #tickRMS = [0.0, 1.0, 2.0, 3.0],
                                      tickRMSangle = 115, showlabelsRMS = 'on',
                                      titleRMS = 'on', titleOBS = 'Ref: '+labels[0],
                                        # checkstats = "on"
                                      )
            
            ax = plt.gca()
            ax.set_title(mode_name+" Taylor Diagram over "+region_name+"\n"+
                         "Area-weighted "+selmonth[0]+" gridded precipitation \n"+
                         str(tslice_array[0].dt.year.values)+"-"+str(tslice_array[-1].dt.year.values),
                         x=1.2, y=1,)
            
            # Create custom legend manually (because otherwise it may end in the wrong place and cannot be specified within skillmetrics)
            handles_legend = []
            labels_legend = []
            
            for labeln, paramn in MARKERS.items():
                if labeln in labels and labeln != labels[0]:
                    handlen = plt.Line2D(
                        [], [],
                        marker=paramn['symbol'],
                        color=paramn['labelColor'],
                        markersize=paramn['size'],
                        markerfacecolor=paramn['faceColor'],
                        markeredgewidth=1.5,
                        markeredgecolor=paramn['edgeColor'],
                        linestyle='None',
                        # axes=ax
                    )
                    handles_legend.append(handlen)
                    labels_legend.append(labeln)
            
            # Place the custom legend
            plt.legend(handles_legend, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
            
            # Save figure
            savepath_seas = savepath
            if not os.path.exists(savepath_seas):
                os.makedirs(savepath_seas)
            plt.savefig(savepath_seas+savefilename, bbox_inches="tight")
            plt.close()
            
            # To check that the equation that defines the diagram is closed (negligible residue)
            sm.check_taylor_stats(lsdev, lcrmsd, lccoef, threshold=1000000000000000000000)
            # 24.05.24: the check does not close but the weighted calculations seem to be fine

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
