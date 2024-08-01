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
import copy

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

#%%% Regional averages
#%%%% Calculate area means (regional averages)
data_to_avg = data_yearlysum # select which data to average (yearly, monthly, daily...)

region ="Germany"#"land"
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
print("Calculating means over "+region)
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
data_yearlysum = data_to_avg

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
    filename = "numgauge_"+region+"_"+dsname+"_"+ysel+".png"
    plt.savefig(savepath_yy+filename, bbox_inches="tight")
    # plt.show()

#%%%% Interannual variability area-means plot
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr"]

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
plt.title("Area-mean annual total precip "+region+" [mm]")
plt.xlim(datetime(2000,1,1), datetime(2020,1,1))
# plt.xlim(2000, 2020)
plt.grid()

#%%%% Interannual variability area-means plot (interactive html)

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr"]

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
hv.save(layout, "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual/"+region+"/lineplots/area_mean_annual_total_precip_"+region+".html")

#%%%% Plot the period from each dataset
# make a list with the names of the precipitation variables
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph", "rr"]

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
             "RW", "RR", "tp", "cmorph", "rr"]

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
plt.title("Area-mean annual total precip BIAS with respect to "+dsref[0]+" "+region)
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
plt.title("Area-mean annual total precip RELATIVE BIAS with respect to "+dsref[0]+" "+region)
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
region = "Germany" #"land" 
to_plot = data_bias_relative_map
dsname = "TSMP-DETECT-Baseline-EURregLonLat01deg"
title = "BIAS"
yearsel = "2016"
cbarlabel = "%" # mm
vmin = -50
vmax = 50
mask = utils.get_regionmask(region)
mask0 = mask.mask(to_plot[dsname])
dropna = True
if mask_TSMP_nudge: 
    mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
    dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":yearsel}].where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", cmap="RdBu_r", 
                                                                                    vmin=vmin, vmax=vmax, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+yearsel+"\n"+dsname+"\n "+region+" Ref.: "+dsref[0])

#%%%%% Simple map plot (loop)
# Like previous but for saving all plots
region = "Germany" #"land" 
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Germany/"
period = np.arange(2000,2024)
to_plot_dict = [
            (data_bias_map, "BIAS", "mm", -250, 250),
            (data_bias_relative_map, "RELATIVE BIAS", "%", -75, 75),
           ]
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
                plot = to_plot[dsname].loc[{"time":str(yearsel)}].where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat", cmap="RdBu_r", 
                                                                                                    vmin=vmin, vmax=vmax, 
                                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                                         cbar_kwargs={'label': cbarlabel, 'shrink':0.88})
                if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
                plot.axes.coastlines(alpha=0.7)
                plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
                plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
                plt.title(title+" "+str(yearsel)+"\n"+dsname_short+"\n "+region+" Ref.: "+dsref[0])
                
                # save figure
                savepath_yy = savepath+str(yearsel)+"/"
                if not os.path.exists(savepath_yy):
                    os.makedirs(savepath_yy)
                filename = "_".join([title.lower().replace(" ","_"), region, dsname_short,dsref[0],str(yearsel)])+".png"
                plt.savefig(savepath_yy+filename, bbox_inches="tight")
                plt.close()
            except KeyError:
                continue

#%%%%% Box plots of BIAS and ERRORS
# the box plots are made up of the yearly bias or error values, and the datasets are ordered according to their median
to_plot0 = data_bias_relative_gp.copy() # data_mean_abs_error_gp # data_bias_relative_gp # data_norm_mean_abs_error_gp
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual/Germany/boxplots/relative_bias/"
savefilename = "boxplot_relative_bias_yearly"
title = "Relative bias (yearly values) "+region+". Ref.: "+dsref[0]
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
ax.set_title(mode_name+" Taylor Diagram over "+region+"\n"+
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
