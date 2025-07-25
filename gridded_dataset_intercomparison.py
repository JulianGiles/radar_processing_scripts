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

# Define a function to reduce the name of the datasets to the minimum
def reduce_dsname(dsname):
    if "_" in dsname:
        dsname0 = dsname.split("_")[0]
    else:
        dsname0 = dsname
    reduced_names=["IMERG-V07B", "IMERG-V06B", "CMORPH", "TSMP-old", "TSMP-DETECT", "ERA5",
                   "RADKLIM", "RADOLAN", "EURADCLIM", "GPCC", "GPROF", "HYRAS", "E-OBS",
                   "CPC", "GSMaP"]
    for reddsname in reduced_names:
        if reddsname in dsname0:
            return reddsname
    warnmsg = ("No reduced name found for "+dsname+". Returning the same input name")
    warnings.warn(warnmsg)
    return dsname

colors = {
    "IMERG-V07B-monthly": "#FF6347", "IMERG-V07B-30min": "#FF6347",    "IMERG-V07B": "#FF6347", # Tomato
    "IMERG-V06B-monthly": "crimson", "IMERG-V06B-30min": "crimson",    "IMERG-V06B": "crimson",  # crimson
    "CMORPH-daily": "#A52A2A",        "CMORPH": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", "TSMP-DETECT": "#1E90FF", # DodgerBlue
    "ERA5-monthly": "#8A2BE2",       "ERA5-hourly": "#8A2BE2",         "ERA5": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black",          "GPCC": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "GSMaP": "#FF7BA6", # PalePink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

def hex_to_rgba(hex_color, alpha=0.5):
    # Convert HEX to RGB
    rgb = mcolors.hex2color(hex_color)
    # Add the alpha (transparency) value
    rgba = rgb + (alpha,)
    return rgba

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
    # "RADKLIM": loadpath_yearly+"RADKLIM/RADKLIM_precipitation_yearlysum_2001-2022.nc",
    # "RADOLAN": loadpath_yearly+"RADOLAN/RADOLAN_precipitation_yearlysum_2006-2022.nc",
    # "EURADCLIM": loadpath_yearly+"EURADCLIM/EURADCLIM_precipitation_yearlysum_2013-2020.nc",
    "GPCC-monthly": loadpath_yearly+"GPCC-monthly/GPCC-monthly_precipitation_yearlysum_1991-2020.nc",
    # "GPCC-daily": loadpath_yearly+"GPCC-daily/GPCC-daily_precipitation_yearlysum_2000-2020.nc",
    "GPROF": loadpath_yearly+"GPROF/GPROF_precipitation_yearlysum_2014-2023.nc",
    # "HYRAS": loadpath_yearly+"HYRAS/HYRAS_precipitation_yearlysum_1931-2020.nc",
    "E-OBS": loadpath_yearly+"E-OBS/E-OBS_precipitation_yearlysum_1950-2023.nc",
    "CPC": loadpath_yearly+"CPC/CPC_precipitation_yearlysum_1979-2024.nc",
    "GSMaP": loadpath_yearly+"GSMaP/GSMaP_precipitation_yearlysum_*.nc",
    }

data_yearlysum = {}

# reload the datasets
print("Loading yearly datasets ...")
for dsname in paths_yearly.keys():
    print("... "+dsname)
    if "*" in paths_yearly[dsname]:
        data_yearlysum[dsname] = xr.open_mfdataset(paths_yearly[dsname])
    else:
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
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
region = "Turkey"
region_name = "Turkey" # "Europe_EURADCLIM" # name for plots
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
                  "GPCC-monthly", "GPCC-daily", "GPROF", "E-OBS", "CPC", "GSMaP"]:
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

dstoplot = ['IMERG-V07B-monthly', 'IMERG-V06B-monthly', 'CMORPH-daily', 'GPROF', "GSMaP",
            # 'RADKLIM', 'RADOLAN', 'EURADCLIM',
            'TSMP-old', 'TSMP-DETECT-Baseline',
            'ERA5-monthly',
            'GPCC-monthly',
            # 'HYRAS',
            'E-OBS', 'CPC']

for dsname in dstoplot:
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
                plt.plot(data_avgreg[dsname]['time'], data_avgreg[dsname][vv],
                         label=reduce_dsname(dsname), c=color, marker=marker)
            except TypeError:
                # try to change the time coord to datetime format
                plt.plot(data_avgreg[dsname].indexes['time'].to_datetimeindex(), data_avgreg[dsname][vv],
                         label=reduce_dsname(dsname), c=color, marker=marker)
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
lonlat_slice = [slice(25.5,42), slice(35.5, 42.25)] # [slice(-43.4,63.65), slice(22.6, 71.15)]
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
                # if mask_TSMP_nudge: plot.axes.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
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
to_plot0 = data_norm_mean_abs_error_gp.copy() # data_mean_abs_error_gp # data_bias_relative_gp # data_norm_mean_abs_error_gp
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/interannual/"+region_name+"/boxplots/normalized_mean_absolute_error/"
savefilename = "boxplot_normalized_mean_absolute_error_yearly"
title = "Normalized mean absolute error (yearly values) "+region_name+". Ref.: "+dsref[0]
ylabel = "%" # % # mm
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting
tsel = [
        slice(None, None), # if I want to consider only certain period. Otherwise set to (None, None). Multiple options possible
        slice("2001-01-01", "2020-01-01"),
        # slice("2006-01-01", "2020-01-01"),
        # slice("2013-01-01", "2020-01-01"),
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
            box = ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=hex_to_rgba(colors[reduce_dsname(key)], 0.2),
                                      edgecolor=colors[reduce_dsname(key)], lw = 2), #'#b6d6e3'
                        medianprops=dict(color=colors[reduce_dsname(key)], lw=2.5), #"#20788e"
                        whiskerprops=dict(color=colors[reduce_dsname(key)], lw=2),
                        capprops=dict(color=colors[reduce_dsname(key)], lw=2),
                        flierprops=dict(markeredgecolor=colors[reduce_dsname(key)], lw=2)
                        )

    # Set x-axis ticks and labels with dataset names
    ax.set_xticks(range(1, len(plotted_arrays) + 1))
    ax.set_xticklabels([reduce_dsname(dsname)
                        for dsname in
                        [ds for ds in to_plot.keys() if ds not in dsignore]
                        ],
                       rotation=45, fontsize=15)
    ax.xaxis.label.set_size(15)     # change xlabel size
    ax.yaxis.label.set_size(15)     # change ylabel size

    ax.tick_params(axis='x', labelsize=15) # change xtick label size
    ax.tick_params(axis='y', labelsize=15) # change xtick label size

    # dsnames=[dsname.split("_")[0] if "_" in dsname
    #                 else "-".join(dsname.split("-")[:-1]) if "EURreg" in dsname
    #                 else dsname
    #                 for dsname in
    #                 [ds for ds in to_plot.keys() if ds not in dsignore]
    #                 ]
    # for xtick, color in zip(ax.get_xticklabels(), [colors[dsname] for dsname in dsnames]):
    #     xtick.set_color(color)

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
    weights = xr.DataArray(utils.grid_cell_areas(ds_ref.longitude.values, ds_ref.latitude.values),
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
    "GSMaP": {
        "labelColor": "k",
        "symbol": "o",
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
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
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
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.longitude.values, ds_ref.latitude.values),
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
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
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
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.longitude.values, ds_ref.latitude.values),
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
    "CMORPH-daily": loadpath_daily+"this one does not matter",
    "TSMP-old": loadpath_daily+"TSMP-old/TSMP-old_precipitation_dailysum_2000-2021.nc",
    "TSMP-DETECT-Baseline": loadpath_daily+"TSMP-DETECT-Baseline/TSMP-DETECT-Baseline_precipitation_dailysum_2000-2022.nc",
    "ERA5-hourly": loadpath_daily+"ERA5-hourly/ERA5-hourly_precipitation_dailysum_1999-2021.nc",
    # "RADKLIM": loadpath_daily+"RADKLIM/temp_serial/RADKLIM-EURregLonLat001deg_precipitation_dailysum_2001-2022_new_part*.nc",
    # "RADOLAN": loadpath_daily+"RADOLAN/RADOLAN-EURregLonLat001deg_precipitation_dailysum_2006-2022.nc",
    # "EURADCLIM": loadpath_daily+"EURADCLIM/temp_serial/EURADCLIM-EURregLonLat002deg_precipitation_dailysum_2013-2020_part*.nc",
    # "GPCC-daily": ,
    "GPROF": loadpath_daily+"this one does not matter",
    # "HYRAS": loadpath_daily+"HYRAS/temp_serial/HYRAS-EURregLonLat001deg_precipitation_dailysum_1930-2020_part*.nc",
    "E-OBS": loadpath_daily+"this one does not matter",
    "GSMaP": loadpath_daily+"GSMaP/GSMaP_precipitation_dailysum_*.nc",
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
    "IMERG-V07B-30min": "#FF6347",    "IMERG-V07B": "#FF6347", # Tomato
    "IMERG-V06B-30min": "crimson",    "IMERG-V06B": "crimson",  # crimson
    "CMORPH-daily": "#A52A2A",        "CMORPH": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", "TSMP-DETECT": "#1E90FF", # DodgerBlue
    "ERA5-hourly": "#8A2BE2",         "ERA5": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black",          "GPCC": "black", # Black
    "GPROF": "#FF1493", # DeepPink #FF7BA6
    "GSMaP": "#FF7BA6", # PalePink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip",
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC", "precipitationCal"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-30min", "ERA5-hourly"] # datasets to ignore in the plotting
dsref = ["E-OBS"] # dataset to take as reference (black and bold curve)

colors[dsref[0]] = "black"

#%%%% Define region and regrid
data_to_avg = data_dailysum.copy() # select which data to process

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland",
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra",
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
region = "Turkey"
region_name = "Turkey" # name for plots
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
plot = data_dailysum[dsname][vname].sel(time=tsel).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat",
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
reload_metrics = False # reload previously calculated metrics if available?
calc_if_no_reload = False # calculate metrics if not available from previously calculated files? #!!! NOT IMPLEMENTED YET!!
minpre = 1 # minimum precipitation in mm/day for a day to be considered a wet day (i.e., minimum measurable precipitation considered). Relevant for categorical metrics
timesel = slice("2015-01-01", "2020-12-31") # should be given in a slice with YYYY-MM-DD
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

                    regridder = xe.Regridder(data_to_bias[dsname], #!!! To improve: select here a reduced domain, otherwise global datasets take forever
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
                        metrics[metric_type][dsname_metric] = xr.open_dataset(metric_path)[metric_type]
                    except:
                        print("... ... metric "+metric_type+" could not be loaded!!")
                for metric_type in metric_types2:
                    try: # try spatem
                        metric_path = "/".join([metricssavepath, "spatem_"+metric_type, dsname_metric, "_".join(["spatem_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatem[metric_type][dsname_metric] = xr.open_dataset(metric_path)["spatem_"+metric_type]
                    except:
                        print("... ... metric spatem "+metric_type+" could not be loaded!!")
                    try: # try spatial
                        metric_path = "/".join([metricssavepath, "spatial_"+metric_type, dsname_metric, "_".join(["spatial_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatial[metric_type][dsname_metric] = xr.open_dataset(metric_path)["spatial_"+metric_type]
                    except:
                        print("... ... metric spatial "+metric_type+" could not be loaded!!")
                    try: # try temporal
                        metric_path = "/".join([metricssavepath, "temporal_"+metric_type, dsname_metric, "_".join(["temporal_"+metric_type, dsname_metric])+".nc"])
                        metrics_temporal[metric_type][dsname_metric] = xr.open_dataset(metric_path)["temporal_"+metric_type]
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
                dataref_wet = (dataref > minpre).astype(bool)

                hits = data0_wet*dataref_wet.where(mask0.notnull(), drop=True).where(dataref.notnull())
                misses = (~data0_wet)*dataref_wet.where(mask0.notnull(), drop=True).where(dataref.notnull())
                false_alarms = data0_wet*(~dataref_wet).where(mask0.notnull(), drop=True).where(dataref.notnull())

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
to_plot = metrics["bias_concurrent"]
dsname = "GSMaP"
title = "Concurrent bias"
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
                timeperiodn = "_".join([tseln.start, tseln.stop])

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
                                patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=hex_to_rgba(colors[reduce_dsname(key)], 0.2),
                                      edgecolor=colors[reduce_dsname(key)], lw = 2), #'#b6d6e3'
                                        medianprops=dict(color=colors[reduce_dsname(key)], lw=2.5), #"#20788e"
                                        whiskerprops=dict(color=colors[reduce_dsname(key)], lw=2),
                                        capprops=dict(color=colors[reduce_dsname(key)], lw=2),
                                        flierprops=dict(markeredgecolor=colors[reduce_dsname(key)], lw=2)
                                )
                    # Add the spatem value as another line (like the median)
                    ax.boxplot(value.values, positions=[len(plotted_arrays)], widths=0.6,
                                patch_artist=True, boxprops=dict(facecolor='#b6d6e3'),
                                showfliers=False, showbox=False, showcaps=False,
                                usermedians=[float(metrics_spatem[metric_type][key])],
                                whiskerprops=dict(color=colors[reduce_dsname(key)], lw=0),
                                medianprops=dict(color="crimson", lw=2, ls=":"))

            # Set x-axis ticks and labels with dataset names
            ax.set_xticks(range(1, len(plotted_arrays) + 1))
            ax.set_xticklabels([reduce_dsname(dsname)
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
            plt.savefig(savepathn+savefilenamen, bbox_inches="tight")
            plt.close()

#%%%%% Table summary of metrics #!!! STILL WORKING ON IT!

# List of metrics and their corresponding attributes (name, title, units, reference_line, vmin, vmax, cmap, bes value)
to_plot_dict = [ # name, title, units, reference_line, vmin, vmax, cmap, best value
            ('bias', "BIAS", "mm", 0, -1, 1, "BrBG", 0.),
            ('mae', "MAE", "mm", 0, 0, 3, "Reds", 0.),
            ('nmae', "NMAE", "%", 0, 0, 10, "Reds", 0.),
            ('pod', "POD", "", 1, 0.6, 1, "Blues_r", 1.),
            ('far', "FAR", "", 0, 0, 0.5, "Reds", 0.),
            ('csi', "CSI", "", 1, 0.4, 1, "Blues_r", 1.),
            ('biass', "BIASS", "", 1, 0.7, 1.3, "BrBG", 1.),
            ('bias_concurrent', "Concurrent BIAS", "mm", 0, -1, 1, "BrBG", 0.),
            ('relative_bias_concurrent', "Concurrent relative BIAS", "%", 0, -60, 60, "BrBG", 0.),
            ('mae_concurrent', "Concurrent MAE", "mm", 0, 0, 5, "Reds", 0.),
            ('nmae_concurrent', "Concurrent NMAE", "%", 0, 0, 10, "Reds", 0.),
           ]

# List of datasets to plot, in the order we want it, do not add the reference dataset
dstoplot = [
    'IMERG-V07B-30min', 'IMERG-V06B-30min',
    "CMORPH-daily", "GPROF", "GSMaP",
    'TSMP-old-EURregLonLat01deg', 'TSMP-DETECT-Baseline-EURregLonLat01deg',
    'ERA5-hourly',
    # 'RADKLIM', 'RADOLAN', 'EURADCLIM',
    # 'HYRAS',
    ]


fontsize = 20

num_datasets = len(dstoplot)

# Adjust the figure size and grid to make the cells squared
fig, ax = plt.subplots(len(to_plot_dict), 1, figsize=(num_datasets * 2, len(to_plot_dict) * 2), gridspec_kw={'hspace': 0.})

for i, (metric, title, unit, ref_line, vmin, vmax, cmap, best) in enumerate(to_plot_dict):
    # Extract values for the current metric
    values = np.array([metrics_spatem[metric][dataset].values if hasattr(metrics_spatem[metric][dataset], 'values') else metrics_spatem[metric][dataset] for dataset in dstoplot])

    # Get the index of the maximum value
    max_idx = np.argmax(values)

    # Get the index of the best value
    best_idx = np.argmin(abs(values-best))

    # Plot the table cells with color
    for j, value in enumerate(values):
        # Determine the normalized color value based on vmin, vmax, cmap
        normalized_value = (value - vmin) / (vmax - vmin)
        normalized_value = np.clip(normalized_value, 0, 1)  # Ensure values are in [0, 1] range

        # Use matplotlib colormaps
        color_map = plt.get_cmap(cmap)
        cell_color = color_map(normalized_value)

        # Fill the cell with the calculated color
        ax[i].add_patch(plt.Rectangle((j, 0), 1, 1, color=cell_color, lw=0))

        # Annotate the cell with the value (larger font size)
        ax[i].text(j + 0.5, 0.5, f"{value:.3f}", ha='center', va='center', color='black',
                   fontweight='bold' if j == best_idx else 'normal', fontsize=fontsize)

    # Set x-ticks only for the last row
    if i == len(to_plot_dict) - 1:
        ax[i].set_xticks(np.arange(num_datasets) + 0.5)
        ax[i].set_xticklabels([reduce_dsname(ds) for ds in dstoplot], rotation=45, ha="right",
                              fontsize=fontsize, fontweight='bold')
    else:
        ax[i].set_xticks([])

    # Set y-tick labels with metric titles on the side (larger font size)
    ax[i].set_yticks([0.5])
    ax[i].set_yticklabels([f"{title} [{unit}]"], rotation=0, ha="right", va="center", fontsize=fontsize, fontweight='bold')
    ax[i].tick_params(axis='y', which='both', length=0)  # Hide tick marks

    # Set square aspect ratio for the cells
    ax[i].set_aspect('equal')

    # Set limits
    ax[i].set_xlim([0, len(dstoplot)])
    ax[i].set_ylim([0, 1])

# Adjust layout
plt.tight_layout()
plt.show()



#%%%%% PDF
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# List of datasets to plot, do not add the reference dataset, it will be added automatically at the end
dstoplot = [
    'IMERG-V07B-30min_E-OBS-grid', 'IMERG-V06B-30min_E-OBS-grid',
    "CMORPH-daily_E-OBS-grid", "GPROF_E-OBS-grid", "GSMaP_E-OBS-grid",
    'TSMP-old-EURregLonLat01deg_E-OBS-grid', 'TSMP-DETECT-Baseline-EURregLonLat01deg_E-OBS-grid',
    'ERA5-hourly_E-OBS-grid',
    # 'RADKLIM_E-OBS-grid', 'RADOLAN_E-OBS-grid', 'EURADCLIM_E-OBS-grid',
    # 'HYRAS_E-OBS-grid',
    ]

savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/daily/"+region_name+"/PDF/"

to_plot0 = data_dailysum.copy()

tsel = [timesel]
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [("DJF" , [12, 1, 2]),
           ("JJA", [6, 7, 8]),
           ("", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

bins = np.arange(1, 200)

print("Plotting PDFs ...")
for tseln in tsel:
    print("... "+"_".join([tseln.start, tseln.stop]))
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])
        for dsname in dstoplot+[dsref[0]]:
            print(dsname)
            if dsname not in dsignore:
                to_plot = to_plot0[dsname].copy()
                if type(to_plot) is xr.DataArray: to_plot = to_plot.to_dataset()
                timeperiodn = timeperiod
                if tseln.start is not None and tseln.stop is not None: # add specific period to title
                    timeperiodn = "_".join([tseln.start, tseln.stop])

                    if ignore_incomplete:
                        if not (to_plot.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            print("Ignoring "+dsname+" because it does not cover the requested period")
                            continue

                # Select the given season and mask
                to_plot = to_plot.sel(time=tseln)
                to_plot = to_plot.sel(time=to_plot['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)

                for vv in var_names:
                    if vv in to_plot.data_vars:
                        # Plot
                        to_plot[vv].where(to_plot[vv]>minpre).plot.hist(bins=bins, density=True, histtype="step",
                                              label=reduce_dsname(dsname), color=colors[reduce_dsname(dsname)])
                        plt.xscale('log')  # Set the x-axis to logarithmic scale
                        plt.xlim(minpre, 150)
                        break

        # Beautify plot
        plt.legend()
        plt.title('PDF Histograms (X-Log Scale) '+region_name+" "+selmonth[0]+" "+timeperiodn)
        plt.xlabel('Precipitation (mm/day)')
        plt.ylabel('Probability Density')

        # Save the plot
        plt.tight_layout()

        savepathn = savepathbase+timeperiodn+"/xlog/"
        savefilenamen = "_".join(["pdfs_xlog", selmonth[0], timeperiodn])+".png"
        if not os.path.exists(savepathn):
            os.makedirs(savepathn)
        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

        # Modify the y scale to log and save again
        plt.yscale('log')  # Set the x-axis to logarithmic scale
        plt.title('PDF Histograms (XY-Log Scale) '+region_name+" "+selmonth[0]+" "+timeperiodn)

        savepathn = savepathbase+timeperiodn+"/xylog/"
        savefilenamen = "_".join(["pdfs_xylog", selmonth[0], timeperiodn])+".png"
        if not os.path.exists(savepathn):
            os.makedirs(savepathn)
        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

        plt.close()

#%%%%% 2d histograms (scatterplots)
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# List of datasets to plot, do not add the reference dataset, it will be added automatically from dsref
dstoplot = [
    'IMERG-V07B-30min_E-OBS-grid', 'IMERG-V06B-30min_E-OBS-grid',
    "CMORPH-daily_E-OBS-grid", "GPROF_E-OBS-grid", "GSMaP_E-OBS-grid",
    'TSMP-old-EURregLonLat01deg_E-OBS-grid', 'TSMP-DETECT-Baseline-EURregLonLat01deg_E-OBS-grid',
    'ERA5-hourly_E-OBS-grid',
    # 'RADKLIM_E-OBS-grid', 'RADOLAN_E-OBS-grid', 'EURADCLIM_E-OBS-grid',
    # 'HYRAS_E-OBS-grid',
    ]

savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/daily/"+region_name+"/2dhist/"

to_plot0 = data_dailysum.copy()

tsel = [timesel]
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [("DJF" , [12, 1, 2]),
           ("JJA", [6, 7, 8]),
           ("", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

bins = np.arange(1, 125)

plot_mean_ds_per_bin = True
plot_mean_bias_per_bin = True
plot_mean_relbias_per_bin = True

meanbiascol = "blue" # color for the mean bias and relative bias scatters
meanbiasmarker = "o"
meanrelbiasmarker = "^"
markersize = 8
simetrical_yaxes = True # force simetrical yaxes scale in the mean bias scatters?

print("Plotting 2D histograms ...")
for tseln in tsel:
    print("... "+"_".join([tseln.start, tseln.stop]))
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])

        dsnameref = dsref[0]
        print(dsnameref)
        to_plot_ref = to_plot0[dsnameref].copy()
        if type(to_plot_ref) is xr.DataArray: to_plot_ref = to_plot_ref.to_dataset()
        timeperiodn = timeperiod
        if tseln.start is not None and tseln.stop is not None: # add specific period to title
            timeperiodn = "_".join([tseln.start, tseln.stop])

            if ignore_incomplete:
                if not (to_plot_ref.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                        to_plot_ref.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                    raise ValueError("The reference dataset selected does not cover the requested period")

        # Select the given season and mask
        to_plot_ref = to_plot_ref.sel(time=tseln)
        to_plot_ref = to_plot_ref.sel(time=to_plot_ref['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)
        if type(to_plot_ref) is xr.Dataset:
            for vv in var_names:
                if vv in to_plot_ref.data_vars:
                    to_plot_ref = to_plot_ref[vv].where(to_plot_ref[vv]>minpre)
                    break

        for dsname in dstoplot:
            print(dsname)
            if dsname not in dsignore:
                to_plot = to_plot0[dsname].copy()
                if type(to_plot) is xr.DataArray: to_plot = to_plot.to_dataset()
                timeperiodn = timeperiod
                if tseln.start is not None and tseln.stop is not None: # add specific period to title
                    timeperiodn = "_".join([tseln.start, tseln.stop])

                    if ignore_incomplete:
                        if not (to_plot.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            print("Ignoring "+dsname+" because it does not cover the requested period")
                            continue

                # Select the given season and mask
                to_plot = to_plot.sel(time=tseln)
                to_plot = to_plot.sel(time=to_plot['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)

                for vv in var_names:
                    if vv in to_plot.data_vars:
                        # Plot
                        fig = plt.figure(figsize=(7,5))
                        ax = plt.gca()
                        to_plot_ref_sel = to_plot_ref.sel(time=to_plot.time)
                        utils.hist_2d(to_plot_ref_sel.values.flatten(), # we need to sel the reference to the to_plot time in case of leap years not present
                                      to_plot[vv].where(to_plot[vv]>minpre).values.flatten(),
                                      bins1=bins, bins2=bins, mini=None, maxi=None, cmap='viridis',
                                      colsteps=30, alpha=1, mode='absolute', fsize=10, colbar=True)
                        plt.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color="black")

                        # Plot line of mean of dsname per bin
                        if plot_mean_ds_per_bin:
                            for binn in np.arange(1, len(bins)):
                                cond = to_plot[vv].where(to_plot[vv]>minpre).where(to_plot_ref_sel > bins[binn-1]).where(to_plot_ref_sel < bins[binn])
                                plt.scatter((bins[binn-1]+bins[binn])/2,
                                            cond.mean(),
                                            color="r", s=markersize
                                            )

                        ax.set_aspect('equal', adjustable='box')

                        plt.grid()

                        # Plot % of hits and R2
                        nhits = np.sum(~np.isnan(to_plot_ref_sel.values.flatten()) & ~np.isnan(to_plot[vv].where(to_plot[vv]>minpre).values.flatten()))
                        nhits_norm = nhits/ np.sum(~np.isnan(to_plot_ref_sel.values.flatten()))
                        r2 = float(xr.corr(to_plot_ref_sel, to_plot[vv].where(to_plot[vv]>minpre)))

                        plt.text(bins[-1]*0.5, bins[-1]*0.9,
                                 "hits: "+str(round(nhits_norm*100))+"%\n $R^2$: "+str(round(r2,2)),
                                 ha="center", size="large")

                        # plt.xscale('log')  # Set the x-axis to logarithmic scale
                        # plt.yscale('log')  # Set the x-axis to logarithmic scale

                        # Beautify plot
                        plt.title('2D Histogram of precip days with more than '+str(minpre)+"mm\n"+region_name+" "+selmonth[0]+" "+timeperiodn)
                        plt.xlabel(reduce_dsname(dsnameref)+' [mm/day]')
                        plt.ylabel(reduce_dsname(dsname)+' [mm/day]')


                        # Plot line of mean bias and relative bias per bin
                        if plot_mean_relbias_per_bin or plot_mean_bias_per_bin:
                            xvals = []
                            yvals = []
                            yvalsrel = []
                            for binn in np.arange(1, len(bins)):
                                xvals.append((bins[binn-1]+bins[binn])/2)
                                cond = metrics['bias_concurrent']["GSMaP"].where(to_plot[vv]>minpre).where(to_plot_ref_sel > bins[binn-1]).where(to_plot_ref_sel < bins[binn])
                                condrel = metrics['relative_bias_concurrent']["GSMaP"].where(to_plot[vv]>minpre).where(to_plot_ref_sel > bins[binn-1]).where(to_plot_ref_sel < bins[binn])
                                yvals.append(float(cond.mean()))
                                yvalsrel.append(float(condrel.mean()))

                        if plot_mean_relbias_per_bin:
                            ax2 = fig.add_axes(ax.get_position(), sharex=ax)
                            ax2.set_facecolor("None")
                            # ax2.set_aspect('equal') # This is for scaling both axes to a 1 to 1 ratio
                            ax2.plot(ax.get_xlim(), [0, 0], color=meanbiascol) # plot the 0 line
                            ax2.scatter(xvals,
                                        yvalsrel,
                                        color=meanbiascol, s=markersize, marker=meanrelbiasmarker,
                                        )
                            plt.xlim(ax.get_xlim())
                            # ax2.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
                            ax2.tick_params(bottom=0, top=0, left=0, right=1,
                                            labelbottom=1, labeltop=0, labelleft=0, labelright=1)
                            ax2.tick_params(axis="y",direction="in", pad=-30, color= meanbiascol, colors=meanbiascol, which="both")
                            ax2.yaxis.label.set_color(meanbiascol)
                            if simetrical_yaxes: ax2.set_ylim(-abs(np.array(ax2.get_ylim())).max(),
                                                              abs(np.array(ax2.get_ylim())).max())
                            ax2.set_box_aspect(1)

                        if plot_mean_bias_per_bin:
                            ax3 = fig.add_axes(ax2.get_position(), sharex=ax)
                            ax3.set_facecolor("None")
                            # ax3.set_aspect('equal') # This is for scaling both axes to a 1 to 1 ratio
                            ax3.scatter(xvals,
                                        yvals,
                                        color=meanbiascol, s=markersize, marker=meanbiasmarker,
                                        )
                            # plt.xlim(ax.get_xlim())
                            # ax3.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])
                            ax3.plot(ax.get_xlim(), [0, 0], color=meanbiascol, ls="--") # plot the 0 line
                            ax3.tick_params(bottom=0, top=0, left=0, right=1,
                                            labelbottom=1, labeltop=0, labelleft=0, labelright=1)
                            if plot_mean_relbias_per_bin: ax3.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])
                            elif simetrical_yaxes: ax3.set_ylim(-abs(np.array(ax3.get_ylim())).max(),
                                                              abs(np.array(ax3.get_ylim())).max())
                            ax3.tick_params(axis="y",direction="in", pad=-30, color= meanbiascol, colors=meanbiascol, which="both")
                            ax3.yaxis.label.set_color(meanbiascol)
                            ax3.set_box_aspect(1)

                        # Save the plot
                        # plt.tight_layout()

                        savepathn = savepathbase+timeperiodn+"/"
                        savefilenamen = "_".join(["2dhist",dsname, selmonth[0], timeperiodn])+".png"
                        if not os.path.exists(savepathn):
                            os.makedirs(savepathn)
                        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

                        plt.close()

                        break

#%%%%% Taylor diagram
# The Taylor diagram can be done by computing the stats over all gridpoints and all timesteps (spatiotemporal)
# or only doing the stats over space or time separately (for these, either temporal or spatial averages must be done first)

#%%%%%% Compute stats and plot for all seasons
import skill_metrics as sm
# https://github.com/PeterRochford/SkillMetrics/blob/master/Examples/taylor10.py#L123
# I cannot use skill_metrics to calculate the stats because they do not filter out
# nan values (because of the masks) so the result is erroneous. They also do not handle weighted arrays.

mode = "" # if "spatial" then average in time and compute the diagram in space. Viceversa for "temporal"
data_to_stat = data_dailysum
savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/daily/"+region_name+"/taylor_diagrams/"

# List of datasets to plot, do not add the reference dataset, it will be taken automatically from dsref
dstoplot = [
    'IMERG-V07B-30min_E-OBS-grid', 'IMERG-V06B-30min_E-OBS-grid',
    "CMORPH-daily_E-OBS-grid",
    'TSMP-old-EURregLonLat01deg_E-OBS-grid', 'TSMP-DETECT-Baseline-EURregLonLat01deg_E-OBS-grid',
    'ERA5-hourly_E-OBS-grid',
    # 'RADKLIM_E-OBS-grid', 'RADOLAN_E-OBS-grid', 'EURADCLIM_E-OBS-grid',
    "GPROF_E-OBS-grid",
    "GSMaP_E-OBS-grid",
    # 'HYRAS_E-OBS-grid',
    ]

selmonthlist = [("DJF" , [12, 1, 2]),
           ("JJA", [6, 7, 8]),
           ("", [1,2,3,4,5,6,7,8,9,10,11,12])] # ("nameofmonth", [month])

# Override options above and sweep over them in a loop
for mode in ["", "spatial", "temporal"]:
    for tslice in [
        slice("2015","2020"), # this covers all
        # slice("2013","2020"), # this excludes GPROF
        # slice("2006","2020"), # this excludes GPROF and EURADCLIM
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
                weights = xr.DataArray(utils.grid_cell_areas(ds_ref.longitude.values, ds_ref.latitude.values),
                                       coords=ds_ref.to_dataset()[["latitude","longitude"]].coords)

            # Get mask
            mask = utils.get_regionmask(region)
            mask_ref = mask.mask(ds_ref)
            if mask_TSMP_nudge: mask_ref = TSMP_no_nudge.mask(ds_ref).where(mask_ref.notnull())
            ds_ref = ds_ref.where(mask_ref.notnull())#.mean(tuple([cn for cn in ds_ref.coords if cn!="time"]))

            # Normalize weights in the mask
            weights = weights.where(mask_ref.notnull(), other=0.)/weights.where(mask_ref.notnull(), other=0.).sum()

            for dsname in dstoplot+dsref: # compute the stats
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
            savepath = savepathbase+"/"
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

#%% Hourly analysis

#%%% Load hourly datasets

loadpath_hourly = "/automount/agradar/jgiles/gridded_data/hourly/"
loadpath_agradar = "/automount/agradar/jgiles/"
loadpath_ags = "/automount/ags/jgiles/"
paths_hourly = {
    "IMERG-V07B-30min": loadpath_hourly+"IMERG-V07B-30min/this one does not matter",
    "IMERG-V06B-30min": loadpath_hourly+"IMERG-V06B-30min/this one does not matter",
    # "CMORPH-daily": loadpath_hourly+"this one does not matter",
    "TSMP-old": loadpath_hourly+"TSMP-old/this one does not matter",
    "TSMP-DETECT-Baseline": loadpath_hourly+"TSMP-DETECT-Baseline/this one does not matter",
    "ERA5-hourly": loadpath_hourly+"ERA5-hourly/this one does not matter",
    # "RADKLIM": loadpath_hourly+"RADKLIM/this one does not matter",
    # "RADOLAN": loadpath_hourly+"RADOLAN/this one does not matter",
    # "EURADCLIM": loadpath_hourly+"EURADCLIM/this one does not matter",
    # "GPCC-daily": ,
    # "GPROF": loadpath_hourly+"this one does not matter",
    # "HYRAS": loadpath_hourly+"HYRAS/temp_serial/HYRAS-EURregLonLat001deg_precipitation_dailysum_1930-2020_part*.nc",
    # "E-OBS": loadpath_hourly+"this one does not matter",
    "GSMaP": loadpath_hourly+"GSMaP/this one does not matter",
    }

data_hourlysum = {}

# load the datasets
print("Loading hourly datasets ...")
for dsname in paths_hourly.keys():
    print("... "+dsname)
    if dsname == "CMORPH-hourly":
        data_hourlysum["CMORPH-daily"] = xr.open_mfdataset('/automount/agradar/jgiles/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/*/*/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc')
        data_hourlysum["CMORPH-daily"] = data_hourlysum["CMORPH-daily"].assign_coords(lon=(((data_hourlysum["CMORPH-daily"].lon + 180) % 360) - 180)).sortby('lon').assign_attrs(units="mm")
        data_hourlysum["CMORPH-daily"]["cmorph"] = data_hourlysum["CMORPH-daily"]["cmorph"].assign_attrs(units="mm")
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
        data_hourlysum["GPROF"] = xr.open_mfdataset(gprof_files_noempty, engine="h5netcdf", group="Grid",
                                          concat_dim="time", combine="nested", preprocess=add_tdim)

        # Set the time coordinate (without empty timesteps)
        data_hourlysum["GPROF"] = data_hourlysum["GPROF"].assign_coords(
                                {"time": gprof_attrs.time.where(gprof_attrs.empty!=1, drop=True).drop_vars("empty")}
                                )

        # Fill the empty timesteps with NaN
        data_hourlysum["GPROF"] = data_hourlysum["GPROF"].broadcast_like(gprof_attrs.empty, exclude=[dim for dim in data_hourlysum["GPROF"].dims if dim !="time"])

        # Remove the variables that we don't need
        data_hourlysum["GPROF"] = data_hourlysum["GPROF"][["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation", "npixPrecipitation", "npixTotal"]]

        # We need to transform the data from mm/h to mm/day
        for vv in ["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation"]:
            data_hourlysum["GPROF"][vv] = data_hourlysum["GPROF"][vv]*24
            data_hourlysum["GPROF"][vv] = data_hourlysum["GPROF"][vv].assign_attrs(units="mm", Units="mm")
    elif dsname == "IMERG-V06B-30min":
        def preprocess_imerg_europe(ds):
            for vv in ds.data_vars:
                try:
                    if "mm/hr" in ds[vv].units:
                        ds[vv] = ds[vv]*0.5 # values come in mm/h but the timestep is 30 min, so we have to divide by 2
                        ds[vv] = ds[vv].assign_attrs(units="mm", Units="mm")
                except AttributeError:
                    pass
            return ds
        data_hourlysum["IMERG-V06B-30min"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V06B/europe/concat_files/*.nc4', preprocess=preprocess_imerg_europe).transpose('time', 'lat', 'lon', ...)
    elif dsname == "IMERG-V07B-30min":
        def preprocess_imerg_europe(ds):
            for vv in ds.data_vars:
                try:
                    if "mm/hr" in ds[vv].units:
                        ds[vv] = ds[vv]*0.5 # values come in mm/h but the timestep is 30 min, so we have to divide by 2
                        ds[vv] = ds[vv].assign_attrs(units="mm", Units="mm")
                except AttributeError:
                    pass
            return ds
        data_hourlysum["IMERG-V07B-30min"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V07B/europe/concat_files/*.nc4', preprocess=preprocess_imerg_europe).transpose('time', 'lat', 'lon', ...)

    elif dsname == "TSMP-old":
        def preprocess_tsmp(ds): # discard the first timestep of every monthly file (discard overlapping data)
            return ds.isel({"time":slice(1,None)})
        data_hourlysum["TSMP-old"] = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*TOT_PREC*',
                                     preprocess=preprocess_tsmp, chunks={"time":1000})

        # data["TSMP-old"] = data["TSMP-old"].assign( xr.open_mfdataset('/automount/agradar/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*WT*',
        #                              chunks={"time":1000}) )
        data_hourlysum["TSMP-old"]["time"] = data_hourlysum["TSMP-old"].get_index("time").shift(-1.5, "h") # shift the values forward to the start of the interval

    elif dsname == "TSMP-DETECT-Baseline":
        def preprocess_tsmp(ds): # discard the first timestep of every monthly file (discard overlapping data)
            return ds.isel({"time":slice(1,None)})
        data_hourlysum["TSMP-DETECT-Baseline"] = xr.open_mfdataset('/automount/ags/jgiles/DETECT_sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1/*/cosmo/TOT_PREC_ts.nc',
                                     preprocess=preprocess_tsmp, chunks={"time":1000})

        data_hourlysum["TSMP-DETECT-Baseline"]["time"] = data_hourlysum["TSMP-DETECT-Baseline"].get_index("time").shift(-0.5, "h") # shift the values forward to the start of the interval

    elif dsname == "ERA5-hourly":
        def preprocess_era5_totprec(ds):
            ds["tp"] = ds["tp"]*1000
            ds["tp"] = ds["tp"].assign_attrs(units="mm", Units="mm")
            return ds
        data_hourlysum["ERA5-hourly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/hourly/europe/single_level_vars/total_precipitation/total_precipitation_*',
                                         preprocess=preprocess_era5_totprec, chunks={"time":100})
        data_hourlysum["ERA5-hourly"] = data_hourlysum["ERA5-hourly"].assign_coords(longitude=(((data_hourlysum["ERA5-hourly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data_hourlysum["ERA5-hourly"] = data_hourlysum["ERA5-hourly"].isel(latitude=slice(None, None, -1))
        data_hourlysum["ERA5-hourly"]["time"] = data_hourlysum["ERA5-hourly"].get_index("time").shift(-1, "h") # shift the values forward to the start of the interval
    elif dsname == "EURADCLIM":
        data_hourlysum["EURADCLIM"] = xr.open_mfdataset("/automount/ags/jgiles/EURADCLIM_v2/concat_files/RAD_OPERA_HOURLY_RAINFALL_*.nc")
        data_hourlysum["EURADCLIM"] = data_hourlysum["EURADCLIM"].set_coords(("lon", "lat"))
        data_hourlysum["EURADCLIM"]["lon"] = data_hourlysum["EURADCLIM"]["lon"][0]
        data_hourlysum["EURADCLIM"]["lat"] = data_hourlysum["EURADCLIM"]["lat"][0]
    elif dsname == "RADKLIM":
        data_hourlysum["RADKLIM"] = xr.open_mfdataset("/automount/ags/jgiles/RADKLIM/20*/*.nc")

    elif dsname == "RADOLAN":
        data_hourlysum["RADOLAN"] = xr.open_mfdataset("/automount/ags/jgiles/RADOLAN/hourly/radolan/historical/concat_files/20*/*.nc")
        data_hourlysum["RADOLAN"]["RW"].attrs["unit"] = "mm"
    elif dsname == "GSMaP":
        def build_time_dim_gsmap(ds):
            # Extract the time variable and its units
            time_var = ds['Time']
            time_units = time_var.attrs['units']

            # Parse the units attribute to get the reference time
            time_origin = pd.to_datetime(time_units.split('since')[1].strip())

            # Decode the time dimension
            decoded_time = time_origin + pd.to_timedelta(time_var.values, unit='H')

            # Replace the original Time variable with the decoded time
            ds = ds.assign_coords(Time=decoded_time).rename({"Time": "time", "Longitude": "lon", "Latitude": "lat"})

            return ds
        print("... Loading skipped because of inefficiency.")
        data_hourlysum[dsname] = None
        # if "precipitation" in var_to_load:
            # data_hourlysum["GSMaP"] = xr.open_mfdataset("/automount/ags/jgiles/GSMaP/standard/v8/netcdf/*/*/*/gsmap_mvk.*.v8.*.nc",
            #                                   decode_times=False, preprocess=build_time_dim_gsmap)
    else:
        if "*" in paths_hourly[dsname]:
            data_hourlysum[dsname] = xr.open_mfdataset(paths_hourly[dsname])
        else:
            data_hourlysum[dsname] = xr.open_dataset(paths_hourly[dsname])


# Special tweaks
print("Applying tweaks ...")
# RADOLAN GRID AND CRS
if "RADOLAN" in data_hourlysum.keys() and "LonLat" not in paths_hourly["RADOLAN"]:
    lonlat_radolan = wrl.georef.rect.get_radolan_grid(900,900, wgs84=True) # these are the left lower edges of each bin
    data_hourlysum["RADOLAN"] = data_hourlysum["RADOLAN"].assign_coords({"lon":(("y", "x"), lonlat_radolan[:,:,0]), "lat":(("y", "x"), lonlat_radolan[:,:,1])})
    data_hourlysum["RADOLAN"] = data_hourlysum["RADOLAN"].assign(crs=data_hourlysum['RADKLIM'].crs[0])
    data_hourlysum["RADOLAN"].attrs["grid_mapping"] = "crs"
    data_hourlysum["RADOLAN"].lon.attrs = data_hourlysum["RADKLIM"].lon.attrs
    data_hourlysum["RADOLAN"].lat.attrs = data_hourlysum["RADKLIM"].lat.attrs

# EURADCLIM coords
if "EURADCLIM" in data_hourlysum.keys():
    data_hourlysum["EURADCLIM"] = data_hourlysum["EURADCLIM"].set_coords(("lon", "lat"))

# Convert all non datetime axes (cf Julian calendars) into datetime
for dsname in data_hourlysum.keys():
    try:
        data_hourlysum[dsname]["time"] = data_hourlysum[dsname].indexes['time'].to_datetimeindex()
        print(dsname+" time dimension transformed to datetime format")
    except:
        pass

# # Rechunk to one chunk per timestep (unless it is already regridded to EURregLonLat)
# for dsname in ["RADOLAN", "RADKLIM", "EURADCLIM", "HYRAS"]:
#     if dsname in data_hourlysum.keys() and "LonLat" not in paths_hourly["RADOLAN"]:
#         print("Rechunking "+dsname)
#         new_chunks = {dim: size for dim, size in data_hourlysum[dsname].dims.items()}
#         new_chunks["time"] = 1
#         try:
#             data_hourlysum[dsname] = data_hourlysum[dsname].chunk(new_chunks)
#         except: # rechunking may fail due to some not unified chunks, then try again
#             data_hourlysum[dsname] = data_hourlysum[dsname].unify_chunks().chunk(new_chunks)

# Rename coordinates to lat/lon in case they are not
for dsname in data_hourlysum.keys():
    try:
        if "longitude" in data_hourlysum[dsname].coords:
            data_hourlysum[dsname] = data_hourlysum[dsname].rename({"longitude": "lon"})
        if "latitude" in data_hourlysum[dsname].coords:
            data_hourlysum[dsname] = data_hourlysum[dsname].rename({"latitude": "lat"})
    except:
        continue

# Remove "bounds" attribute from lon and lat coords to avoid regridding fails
for dsname in data_hourlysum.keys():
    try:
        if "lon" in data_hourlysum[dsname].coords:
            if "bounds" in data_hourlysum[dsname].lon.attrs:
                del(data_hourlysum[dsname].lon.attrs["bounds"])
        if "lat" in data_hourlysum[dsname].coords:
            if "bounds" in data_hourlysum[dsname].lat.attrs:
                del(data_hourlysum[dsname].lat.attrs["bounds"])
    except:
        continue

# Special selections for incomplete extreme years
# IMERG
if "IMERG-V07B-30min" in data_hourlysum.keys():
    data_hourlysum["IMERG-V07B-30min"] = data_hourlysum["IMERG-V07B-30min"].loc[{"time":slice("2001", "2022")}]
if "IMERG-V06B-30min" in data_hourlysum.keys():
    data_hourlysum["IMERG-V06B-30min"] = data_hourlysum["IMERG-V06B-30min"].loc[{"time":slice("2001", "2020")}]
# CMORPH
if "CMORPH-daily" in data_hourlysum.keys():
    data_hourlysum["CMORPH-daily"] = data_hourlysum["CMORPH-daily"].loc[{"time":slice("1998", "2022")}]
# GPROF
if "GPROF" in data_hourlysum.keys():
    data_hourlysum["GPROF"] = data_hourlysum["GPROF"].loc[{"time":slice("2015", "2022")}]
# CPC
if "CPC" in data_hourlysum.keys():
    data_hourlysum["CPC"] = data_hourlysum["CPC"].loc[{"time":slice("1979", "2023")}]

colors = {
    "IMERG-V07B-30min": "#FF6347",    "IMERG-V07B": "#FF6347", # Tomato
    "IMERG-V06B-30min": "crimson",    "IMERG-V06B": "crimson",  # crimson
    "CMORPH-daily": "#A52A2A",        "CMORPH": "#A52A2A", # Brown
    "TSMP-old": "#4682B4", # SteelBlue
    "TSMP-DETECT-Baseline": "#1E90FF", "TSMP-DETECT": "#1E90FF", # DodgerBlue
    "ERA5-hourly": "#8A2BE2",         "ERA5": "#8A2BE2", # BlueViolet
    "RADKLIM": "#006400", # DarkGreen
    "RADOLAN": "#228B22", # ForestGreen
    "EURADCLIM": "#32CD32", # LimeGreen
    "GPCC-monthly": "black",          "GPCC": "black", # Black
    "GPROF": "#FF1493", # DeepPink
    "GSMaP": "#FF7BA6", # PalePink
    "HYRAS": "#FFD700", # Gold
    "E-OBS": "#FFA500", # Orange
    "CPC": "#FF8C00", # DarkOrange
    }

var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip",
             "RW", "RR", "tp", "cmorph", "rr", "hourlyPrecipRateGC", "precipitationCal"]
dsignore = [] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-30min", "ERA5-hourly"] # datasets to ignore in the plotting
dsref = ["RADKLIM-EURregLonLat025deg"] # dataset to take as reference (black and bold curve)

colors[dsref[0]] = "black"
colors["RADKLIM"] = "black"

#%%%% Define region and regrid
timesel = slice("2015-01-01", "2020-12-31") # select a slice to regrid, to save time.
data_to_avg = data_hourlysum.copy() # select which data to process

region =["Portugal", "Spain", "France", "United Kingdom", "Ireland",
         "Belgium", "Netherlands", "Luxembourg", "Germany", "Switzerland",
         "Austria", "Poland", "Denmark", "Slovenia", "Liechtenstein", "Andorra",
         "Monaco", "Czechia", "Slovakia", "Hungary", "Romania"]#"land"
region = "Turkey"
region_name = "Turkey" # name for plots
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

# We will regrid everything to a 0.25 de grid
grid_out = xe.util.cf_grid_2d(-49.875,70.75,0.25,19.725,74.475,0.25) # manually recreate an EURregLonLat025deg grid that matches ERA5
lonres = 0.25
latres = 0.25
dsignore = ["ERA5-hourly"] # We ignore ERA5 because we are regridding to a matching grid

# variables to remove because they are problematic to regrid (because of their type)
delvars = ['HQobservationTime', "MWobservationTime"]

# Define a function to get the minimum lon lat box between two datasets
def get_smaller_domain(ds1, ds2):
    """
    Returns smallest box: lonleft, lonright, latbottom, lattop
    """
    lon10 = float(ds1.lon.min())
    lon11 = float(ds1.lon.max())
    lat10 = float(ds1.lat.min())
    lat11 = float(ds1.lat.max())

    lon20 = float(ds2.lon.min())
    lon21 = float(ds2.lon.max())
    lat20 = float(ds2.lat.min())
    lat21 = float(ds2.lat.max())

    return np.max([lon10, lon20]), np.min([lon11, lon21]), np.max([lat10, lat20]), np.min([lat11, lat21])

print("Regridding datasets ...")
print("Ignoring:")
print(dsignore)

to_add = {} # dictionary to add rotated versions
for dsname in data_to_avg.keys():
    regsavepath = paths_hourly[dsname].rsplit('/', 1)[0] + '/' + dsname+"-EURregLonLat025deg"
    if dsname not in dsignore:
        if not os.path.exists(os.path.dirname(regsavepath)):
            os.makedirs(os.path.dirname(regsavepath))

        try:
            to_add[dsname+"-EURregLonLat025deg"] = xr.open_mfdataset(regsavepath+"*")
            print("Previously regridded "+dsname+" was loaded")
        except (FileNotFoundError, OSError):
            print("... "+dsname)
            if dsname == "GSMaP":
                # We will make daily files one by one
                dayfolders = glob.glob("/automount/ags/jgiles/GSMaP/standard/v8/netcdf/*/*/*")
                st = time.time()

                # Select only the folders that belong to the requested timesel (slice)
                start_date = datetime.strptime(timesel.start, "%Y-%m-%d")
                end_date = datetime.strptime(timesel.stop, "%Y-%m-%d")

                def filter_files_in_range(file_path, start_date, end_date):
                    # Extract date part from the file path
                    date_str = '/'.join(file_path.split('/')[-3:])
                    file_date = datetime.strptime(date_str, "%Y/%m/%d")

                    return start_date <= file_date <= end_date

                dayfolderssel = [f for f in dayfolders if filter_files_in_range(f, start_date, end_date)]
                dayfolderssel.sort()

                for nday, dayfolder in enumerate(dayfolderssel):
                    gsmap = xr.open_mfdataset(dayfolder+"/gsmap_mvk.*.v8.*.nc",
                                                      decode_times=False, preprocess=build_time_dim_gsmap)
                    daysdim = gsmap['time'].resample(time="D").first()["time"]
                    day = str(daysdim.values[0])[0:10]

                    if nday==0:
                        # Build regridder
                        lon0, lon1, lat0, lat1 = get_smaller_domain(grid_out, gsmap)
                        lon0 -= lonres
                        lon1 += lonres
                        lat0 -= latres
                        lat1 += latres

                        regridder = xe.Regridder(gsmap.cf.add_bounds(["lon", "lat"]),
                                                 grid_out.loc[{"lon": slice(lon0, lon1), "lat": slice(lat0, lat1)}], "conservative")

                    to_add[dsname+"-EURregLonLat025deg"] = regridder(gsmap,
                                                                     skipna=True, na_thres=1)

                    encoding = {}
                    for vv in to_add[dsname+"-EURregLonLat025deg"].data_vars:
                        valid_encodings = ['contiguous', 'complevel', 'compression', 'zlib', '_FillValue', 'shuffle', 'fletcher32', 'dtype', 'least_significant_digit']
                        encoding[vv] = dict((k, gsmap[vv].encoding[k]) for k in valid_encodings if k in gsmap[vv].encoding) #!!! For now we keep the same enconding, check later if change it
                        encoding[vv]["zlib"] = True
                        encoding[vv]["complevel"] = 6

                    to_add[dsname+"-EURregLonLat025deg"].to_netcdf(regsavepath+"_"+day+".nc", encoding=encoding)
                tt = time.time() - st
                print(f"Regridding time: {tt/60:.2f} minutes.")

            else:
                # Remove "bounds" attribute from lon and lat coords to avoid regridding fails
                if "lon" in data_to_avg[dsname].coords:
                    if "bounds" in data_to_avg[dsname].lon.attrs:
                        del(data_to_avg[dsname].lon.attrs["bounds"])
                if "lat" in data_to_avg[dsname].coords:
                    if "bounds" in data_to_avg[dsname].lat.attrs:
                        del(data_to_avg[dsname].lat.attrs["bounds"])
                # Remove bounds variables, we will do them ourselves
                for vv in ["lon_bnds", "lon_bounds", "lat_bnds", "lon_bounds", "time_bnds", "time_bounds"]+delvars:
                    if vv in data_to_avg[dsname]:
                        del(data_to_avg[dsname][vv])

                # Build regridder
                lon0, lon1, lat0, lat1 = get_smaller_domain(grid_out, data_to_avg[dsname])
                lon0 -= lonres
                lon1 += lonres
                lat0 -= latres
                lat1 += latres

                if dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
                    regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]),
                                             grid_out.loc[{"lon": slice(lon0, lon1), "lat": slice(lat0, lat1)}], "conservative")

                    to_add[dsname+"-EURregLonLat025deg"] = regridder(data_to_avg[dsname].sel(time=timesel),
                                                                     skipna=True, na_thres=1)
                elif dsname in ['TSMP-old', 'TSMP-DETECT-Baseline']:
                    regridder = xe.Regridder(data_to_avg[dsname].cf.add_bounds(["lon", "lat"]),
                                             grid_out.loc[{"lon": slice(lon0, lon1), "lat": slice(lat0, lat1)}], "conservative")

                    to_add[dsname+"-EURregLonLat025deg"] = regridder(data_to_avg[dsname].loc[{"time": timesel }],
                                                                     skipna=True, na_thres=1)

                else:
                    regridder = xe.Regridder(data_to_avg[dsname].loc[{"lon": slice(lon0, lon1),
                                                                      "lat": slice(lat0, lat1)}].cf.add_bounds(["lon", "lat"]),
                                             grid_out.loc[{"lon": slice(lon0, lon1), "lat": slice(lat0, lat1)}], "conservative")

                    to_add[dsname+"-EURregLonLat025deg"] = regridder(data_to_avg[dsname].loc[{"lon": slice(lon0, lon1),
                                                                                              "lat": slice(lat0, lat1),
                                                                                              "time": timesel }],
                                                                     skipna=True, na_thres=1)

                encoding = {}
                for vv in to_add[dsname+"-EURregLonLat025deg"].data_vars:
                    valid_encodings = ['contiguous', 'complevel', 'compression', 'zlib', '_FillValue',
                                       'shuffle', 'fletcher32', 'dtype', 'least_significant_digit',
                                       "scale_factor", "add_offset"]
                    encoding[vv] = dict((k, data_to_avg[dsname][vv].encoding[k]) for k in valid_encodings if k in data_to_avg[dsname][vv].encoding) #!!! For now we keep the same enconding, check later if change it
                    encoding[vv]["zlib"] = True
                    encoding[vv]["complevel"] = 6

                # Save files by day
                time_index = to_add[dsname+"-EURregLonLat025deg"].time.to_index()
                unique_dates = time_index.normalize().strftime('%Y-%m-%d').unique()
                date_list = unique_dates.tolist()

                st = time.time()
                for date in date_list:
                    if "IMERG" in dsname:
                        # For IMERG we also need to transform from 30-min to hourly
                        to_add[dsname+"-EURregLonLat025deg"].sel(time=date).resample(time="H").sum().to_netcdf(regsavepath+"_"+date+".nc", encoding=encoding)
                    else:
                        to_add[dsname+"-EURregLonLat025deg"].sel(time=date).to_netcdf(regsavepath+"_"+date+".nc", encoding=encoding)
                tt = time.time() - st
                print(f"Regridding time: {tt/60:.2f} minutes.")


                # for date in np.unique(to_add[dsname+"-EURregLonLat025deg"].time.dt.year.values): # save files by year
                #     to_add[dsname+"-EURregLonLat025deg"].sel(time=str(yy)).to_netcdf(regsavepath+"_"+str(yy)+".nc", encoding=encoding)

            to_add[dsname+"-EURregLonLat025deg"] = xr.open_mfdataset(regsavepath+"*") # reload the dataset

# add the unrotated datasets to the original dictionary
data_to_avg = {**data_to_avg, **to_add}
data_hourlysum = data_to_avg.copy()

total_time = time.time() - start_time
print(f"Total elapsed time: {total_time/60:.2f} minutes.")

#%%%% Fix time axes of some shifted datasets and more
for dsname in data_hourlysum.keys():
    if "EURADCLIM" in dsname:
        data_hourlysum[dsname].coords["time"] = data_hourlysum[dsname].time.dt.floor('H')
    if "RADOLAN" in dsname:
        data_hourlysum[dsname].coords["time"] = data_hourlysum[dsname].time.dt.floor('H')
        data_hourlysum[dsname] = data_hourlysum[dsname].round(1)
    if "RADKLIM" in dsname:
        data_hourlysum[dsname].coords["time"] = data_hourlysum[dsname].time.dt.floor('H')


#%%%% Simple map plot
dsname = "RADOLAN-EURregLonLat025deg"
vname = "RW"
tsel = "2015-01-02T06"
mask = utils.get_regionmask(region)
mask0 = mask.mask(data_hourlysum[dsname])
dropna = True
if mask_TSMP_nudge:
    mask0 = TSMP_no_nudge.mask(data_hourlysum[dsname]).where(mask0.notnull())
    # dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = data_hourlysum[dsname][vname].sel(time=tsel).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat",
                                                                                            cmap="Blues", vmin=0, vmax=10,
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "mm", 'shrink':0.88})
# if mask_TSMP_nudge: ax1.set_extent([-43.4, 63.65, 22.6, 71.15], crs=ccrs.PlateCarree())
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(dsname)

#%%%% Simple map plot (for number of stations per gridcell) # CHECK THIS FOR HOURLY SUM BEFORE RUNNING!!
dsname = "GPCC-monthly"
vname = "numgauge"
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/maps/Turkey/"

for yy in np.arange(2000,2021):
    ysel = str(yy)
    mask = utils.get_regionmask(region)
    mask0 = mask.mask(data_hourlysum[dsname])
    dropna = True
    if mask_TSMP_nudge:
        mask0 = TSMP_no_nudge.mask(data_hourlysum[dsname]).where(mask0.notnull())
        dropna=False
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    cmap1 = copy.copy(plt.cm.Blues)
    cmap1.set_under("lightgray")
    plot = (data_hourlysum[dsname][vname].sel(time=ysel)/12).where(mask0.notnull(), drop=dropna).plot(x="lon", y="lat",
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
minpre = 0.1 # minimum precipitation in mm/h for a timestep to be considered a wet day (i.e., minimum measurable precipitation considered). Relevant for categorical metrics
timesel = slice("2015-01-01", "2020-12-31") # should be given in a slice with YYYY-MM-DD
timeperiod = "_".join([timesel.start, timesel.stop])
metricssavepath = "/automount/agradar/jgiles/gridded_data/hourly_metrics/"+timeperiod+"/ref_"+dsref[0]+"/"+region_name+"/" # path to save the results of the metrics
print("!! Period selected: "+timeperiod+" !!")

# Define a function to set the encoding compression
def define_encoding(data):
    encoding = {}


    for var_name, var in data.data_vars.items():
        # Combine compression and chunking settings
        encoding[var_name] = {"zlib": True, "complevel": 6}

    return encoding

# Compute the biases
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline',
            "GSMaP", 'IMERG-V07B-30min', 'IMERG-V06B-30min'] # datasets to ignore (because we want the regridded version)
data_to_bias = copy.copy(data_hourlysum)

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
        dataref = data_to_bias[dsref[0]][vvref]#.chunk(time=100)
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
            data0 = data_to_bias[dsname][vv].copy()

            dsname_metric = reduce_dsname(dsname)

            data0 = data0.loc[{"time":timesel}].chunk(time=100).drop_duplicates(dim="time")
            # data0 = data0.where(data0>0) # do not filter out the zero values, otherwise we will not check for detection correcly

            if "ERA5" in dsname:
                # For ERA5 there are tiny differences in the coordinates that I need to solve, otherwise the datasets
                # will not align and the .where() will fail
                data0 = data0.interp_like(mask0, method="nearest")

            # Reload metrics
            if reload_metrics:
                print("... reloading metrics")
                for metric_type in metric_types:
                    try:
                        metric_path = "/".join([metricssavepath, metric_type, dsname_metric, "_".join([metric_type,dsname_metric])+".nc"])
                        metrics[metric_type][dsname_metric] = xr.open_dataset(metric_path)[metric_type]
                    except:
                        print("... ... metric "+metric_type+" could not be loaded!!")
                for metric_type in metric_types2:
                    try: # try spatem
                        metric_path = "/".join([metricssavepath, "spatem_"+metric_type, dsname_metric, "_".join(["spatem_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatem[metric_type][dsname_metric] = xr.open_dataset(metric_path)["spatem_"+metric_type]
                    except:
                        print("... ... metric spatem "+metric_type+" could not be loaded!!")
                    try: # try spatial
                        metric_path = "/".join([metricssavepath, "spatial_"+metric_type, dsname_metric, "_".join(["spatial_"+metric_type, dsname_metric])+".nc"])
                        metrics_spatial[metric_type][dsname_metric] = xr.open_dataset(metric_path)["spatial_"+metric_type]
                    except:
                        print("... ... metric spatial "+metric_type+" could not be loaded!!")
                    try: # try temporal
                        metric_path = "/".join([metricssavepath, "temporal_"+metric_type, dsname_metric, "_".join(["temporal_"+metric_type, dsname_metric])+".nc"])
                        metrics_temporal[metric_type][dsname_metric] = xr.open_dataset(metric_path)["temporal_"+metric_type]
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
                data0_wet = (data0 >= minpre).astype(bool)
                dataref_wet = (dataref >= minpre).astype(bool)

                hits = data0_wet*dataref_wet.where(mask0.notnull(), drop=True).where(dataref.notnull())
                misses = (~data0_wet)*dataref_wet.where(mask0.notnull(), drop=True).where(dataref.notnull())
                false_alarms = data0_wet*(~dataref_wet).where(mask0.notnull(), drop=True).where(dataref.notnull())

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

total_time = time.time() - start_time
print(f"Calculating metrics took {total_time/60:.2f} minutes to run.")

if not reload_metrics:
    # Save the metrics to files
    print("Saving the metrics to files")
    # # Save metrics to files # WE DO NOT NEED TO SAVE THE FULL metrics
    # for metric_type, datasets in metrics.items():
    #     metric_dir = os.path.join(metricssavepath, metric_type)
    #     if not os.path.exists(metric_dir):
    #         os.makedirs(metric_dir)

    #     for dsname, metric_data in datasets.items():
    #         ds_dir = os.path.join(metric_dir, dsname)
    #         if not os.path.exists(ds_dir):
    #             os.makedirs(ds_dir)

    #         # Convert the data to an xarray Dataset and save as NetCDF
    #         metric_dataset = xr.Dataset({f"{metric_type}": metric_data})
    #         metric_filename = os.path.join(ds_dir, f"{metric_type}_{dsname}.nc")
    #         metric_dataset.to_netcdf(metric_filename, encoding={metric_type: {"zlib":True, "complevel":6}})
    #         print(f"Saved {metric_type} for {dsname} to {metric_filename}")

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
#%%%%% Simple map plot # CAREFUL THAT I DID NOT SAVE-RELOAD THE FULL METRICS
# region = "Germany" #"land"
to_plot = metrics_spatial["relative_bias_concurrent"]
dsname = "IMERG-V07B"
title = "Concurrent relative BIAS"
timesel0 = ""
cbarlabel = "%" # mm
vmin = -80
vmax = 80
lonlat_slice = [slice(-43.4,63.65), slice(22.6, 71.15)]
# mask = utils.get_regionmask(region)
# mask0 = mask.mask(to_plot[dsname])
dropna = True
# if mask_TSMP_nudge:
#     mask0 = TSMP_no_nudge.mask(to_plot[dsname]).where(mask0.notnull())
#     # dropna=False
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].where(mask0.notnull(), drop=dropna).loc[{"lon":lonlat_slice[0],
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
savepath = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/hourly/"+region_name+"/maps/"
to_plot_dict = [
            (metrics_spatial['bias'], "BIAS", "mm", -1, 1, "BrBG"),
            (metrics_spatial['mae'], "MAE", "mm", 0, 1, "Reds"),
            (metrics_spatial['nmae'], "NMAE", "%", 0, 10, "Reds"),
            (metrics_spatial['pod'], "POD", "", 0.6, 1, "Blues"),
            (metrics_spatial['far'], "FAR", "", 0, 0.5, "Reds"),
            (metrics_spatial['csi'], "CSI", "", 0.4, 1, "Blues"),
            (metrics_spatial['biass'], "BIASS", "", 0.7, 1.3, "BrBG"),
            (metrics_spatial['bias_concurrent'], "Concurrent BIAS", "mm", -0.5, 0.5, "BrBG"),
            (metrics_spatial['relative_bias_concurrent'], "Concurrent relative BIAS", "%", -80, 80, "BrBG"),
            (metrics_spatial['mae_concurrent'], "Concurrent MAE", "mm", 0, 1, "Reds"),
            (metrics_spatial['nmae_concurrent'], "Concurrent NMAE", "%", 0, 80, "Reds"),
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
dsignore = ['TSMP-old-EURregLonLat025deg', 'TSMP-old'] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/hourly/"+region_name+"/boxplots/"

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

selmonthlist = [("", [1,2,3,4,5,6,7,8,9,10,11,12]),
                ("DJF" , [12, 1, 2]),
           ("JJA", [6, 7, 8]),
           ] # ("nameofmonth", [month])

print("Plotting boxplots ...")
for metric_type, title, ylabel, reference_line, vmin, vmax, cmap in to_plot_dict:
    print("... "+metric_type)
    for selmonth in selmonthlist:
        for tseln in tsel:
            to_plot = to_plot0[metric_type].copy()
            timeperiodn = timeperiod
            if tseln.start is not None and tseln.stop is not None: # add specific period to title
                timeperiodn = "_".join([tseln.start, tseln.stop])

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
            plt.savefig(savepathn+savefilenamen, bbox_inches="tight")
            plt.close()


#%%%%% PDF
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# List of datasets to plot, do not add the reference dataset, it will be added automatically at the end
dstoplot = [
    'IMERG-V07B-30min-EURregLonLat025deg', 'IMERG-V06B-30min-EURregLonLat025deg',
    'GSMaP-EURregLonLat025deg',
    'TSMP-DETECT-Baseline-EURregLonLat025deg',
    "ERA5-hourly",
    'RADOLAN-EURregLonLat025deg', 'EURADCLIM-EURregLonLat025deg',
    ]

savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/hourly/"+region_name+"/PDF/"

to_plot0 = data_hourlysum.copy()

tsel = [timesel]
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [("", [1,2,3,4,5,6,7,8,9,10,11,12]),
                ("DJF" , [12, 1, 2]),
                ("JJA", [6, 7, 8]),
           ] # ("nameofmonth", [month])

bins = np.round(np.arange(0.1, 10.1, 0.1), 1) - 0.001

dataref = to_plot0[dsref[0]]
mask0 = mask.mask(dataref)
if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

print("Plotting PDFs ...")
for tseln in tsel:
    print("... "+"_".join([tseln.start, tseln.stop]))
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])
        for dsname in dstoplot+[dsref[0]]:
            print(dsname)
            if dsname not in dsignore:
                to_plot = to_plot0[dsname].copy()

                if "ERA5" in dsname:
                    # For ERA5 there are tiny differences in the coordinates that I need to solve, otherwise the datasets
                    # will not align and the .where() will fail
                    to_plot = to_plot.interp_like(mask0, method="nearest")

                if type(to_plot) is xr.DataArray: to_plot = to_plot.to_dataset()
                timeperiodn = timeperiod
                if tseln.start is not None and tseln.stop is not None: # add specific period to title
                    timeperiodn = "_".join([tseln.start, tseln.stop])

                    if ignore_incomplete:
                        if not (to_plot.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            print("Ignoring "+dsname+" because it does not cover the requested period")
                            continue

                # Select the given season and mask
                to_plot = to_plot.sel(time=tseln)
                to_plot = to_plot.sel(time=to_plot['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)

                for vv in var_names:
                    if vv in to_plot.data_vars:
                        # Plot
                        to_plot[vv].where(to_plot[vv]>=minpre).plot.hist(bins=bins, density=True, histtype="step",
                                              label=reduce_dsname(dsname), color=colors[reduce_dsname(dsname)])
                        plt.xscale('log')  # Set the x-axis to logarithmic scale
                        plt.xlim(minpre, 10)
                        break

        # Beautify plot
        plt.legend()
        plt.title('PDF Histograms (X-Log Scale) '+region_name+" "+selmonth[0]+" "+timeperiodn)
        plt.xlabel('Precipitation (mm/h)')
        plt.ylabel('Probability Density')

        # Save the plot
        plt.tight_layout()

        savepathn = savepathbase+timeperiodn+"/xlog/"
        savefilenamen = "_".join(["pdfs_xlog", selmonth[0], timeperiodn])+".png"
        if not os.path.exists(savepathn):
            os.makedirs(savepathn)
        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

        # Modify the y scale to log and save again
        plt.yscale('log')  # Set the x-axis to logarithmic scale
        plt.title('PDF Histograms (XY-Log Scale) '+region_name+" "+selmonth[0]+" "+timeperiodn)

        savepathn = savepathbase+timeperiodn+"/xylog/"
        savefilenamen = "_".join(["pdfs_xylog", selmonth[0], timeperiodn])+".png"
        if not os.path.exists(savepathn):
            os.makedirs(savepathn)
        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

        plt.close()

#%%%%% 2d histograms (scatterplots)
dsignore = [] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# List of datasets to plot, do not add the reference dataset, it will be added automatically from dsref
dstoplot = [
    'IMERG-V07B-30min-EURregLonLat025deg', 'IMERG-V06B-30min-EURregLonLat025deg',
    'GSMaP-EURregLonLat025deg',
    'TSMP-DETECT-Baseline-EURregLonLat025deg',
    "ERA5-hourly",
    'RADOLAN-EURregLonLat025deg', 'EURADCLIM-EURregLonLat025deg',
    ]

savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/hourly/"+region_name+"/2dhist/"

to_plot0 = data_hourlysum.copy()

tsel = [timesel]
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [
    ("", [1,2,3,4,5,6,7,8,9,10,11,12]),
                ("DJF" , [12, 1, 2]),
                ("JJA", [6, 7, 8]),
           ] # ("nameofmonth", [month])

bins = np.round(np.arange(0.1, 15.1, 0.1), 1) - 0.001

dataref = to_plot0[dsref[0]]
mask0 = mask.mask(dataref)
if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

print("Plotting 2D histograms ...")
for tseln in tsel:
    print("... "+"_".join([tseln.start, tseln.stop]))
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])

        dsnameref = dsref[0]
        print(dsnameref)
        to_plot_ref = to_plot0[dsnameref].copy()
        if type(to_plot_ref) is xr.DataArray: to_plot_ref = to_plot_ref.to_dataset()
        timeperiodn = timeperiod
        if tseln.start is not None and tseln.stop is not None: # add specific period to title
            timeperiodn = "_".join([tseln.start, tseln.stop])

            if ignore_incomplete:
                if not (to_plot_ref.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                        to_plot_ref.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                    raise ValueError("The reference dataset selected does not cover the requested period")

        # Select the given season and mask
        to_plot_ref = to_plot_ref.sel(time=tseln)
        to_plot_ref = to_plot_ref.sel(time=to_plot_ref['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)
        if type(to_plot_ref) is xr.Dataset:
            for vv in var_names:
                if vv in to_plot_ref.data_vars:
                    to_plot_ref = to_plot_ref[vv].where(to_plot_ref[vv]>=minpre)
                    break

        for dsname in dstoplot:
            print(dsname)
            if dsname not in dsignore:
                to_plot = to_plot0[dsname].copy()
                if type(to_plot) is xr.DataArray: to_plot = to_plot.to_dataset()
                timeperiodn = timeperiod
                if tseln.start is not None and tseln.stop is not None: # add specific period to title
                    timeperiodn = "_".join([tseln.start, tseln.stop])

                    if ignore_incomplete:
                        if not (to_plot.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                to_plot.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                            print("Ignoring "+dsname+" because it does not cover the requested period")
                            continue

                if "ERA5" in dsname:
                    # For ERA5 there are tiny differences in the coordinates that I need to solve, otherwise the datasets
                    # will not align and the .where() will fail
                    to_plot = to_plot.interp_like(mask0, method="nearest")
                # Select the given season and mask
                to_plot = to_plot.sel(time=tseln)
                to_plot = to_plot.sel(time=to_plot['time'].dt.month.isin(selmonth[1])).where(mask0.notnull(), drop=True)

                for vv in var_names:
                    if vv in to_plot.data_vars:
                        # Plot
                        plt.figure(figsize=(7,5))
                        ax = plt.gca()
                        to_plot_ref_sel = to_plot_ref.sel(time=to_plot.time)
                        utils.hist_2d(to_plot_ref_sel.values.flatten(), # we need to sel the reference to the to_plot time in case of leap years not present
                                      to_plot[vv].where(to_plot[vv]>=minpre).values.flatten(),
                                      bins1=bins, bins2=bins, mini=None, maxi=None, cmap='viridis',
                                      colsteps=30, alpha=1, mode='absolute', fsize=10, colbar=True)
                        plt.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color="black")
                        ax.set_aspect('equal', adjustable='box')

                        plt.grid()

                        # Plot % of hits and R2
                        nhits = np.sum(~np.isnan(to_plot_ref_sel.values.flatten()) & ~np.isnan(to_plot[vv].where(to_plot[vv]>=minpre).values.flatten()))
                        nhits_norm = nhits/ np.sum(~np.isnan(to_plot_ref_sel.values.flatten()))
                        r2 = float(xr.corr(to_plot_ref_sel, to_plot[vv].where(to_plot[vv]>=minpre)))

                        plt.text(bins[-1]*0.95, bins[-1]*0.2,
                                 "hits: "+str(round(nhits_norm*100))+"%\n $R^2$: "+str(round(r2,2)),
                                 ha="right", size="large")

                        # plt.xscale('log')  # Set the x-axis to logarithmic scale
                        # plt.yscale('log')  # Set the x-axis to logarithmic scale

                        # Beautify plot
                        plt.title('2D Histogram of precip hours with more than '+str(minpre)+"mm\n"+region_name+" "+selmonth[0]+" "+timeperiodn)
                        plt.xlabel(reduce_dsname(dsnameref)+' [mm/h]')
                        plt.ylabel(reduce_dsname(dsname)+' [mm/h]')

                        # Save the plot
                        plt.tight_layout()

                        savepathn = savepathbase+timeperiodn+"/"
                        savefilenamen = "_".join(["2dhist",dsname, selmonth[0], timeperiodn])+".png"
                        if not os.path.exists(savepathn):
                            os.makedirs(savepathn)
                        plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

                        plt.close()

                        break

#%%%%% Mean diurnal cycles (amount, frequency, intensity)
reload_cycles = False
cyclessavepath0 = "/automount/agradar/jgiles/gridded_data/hourly_metrics/" # path to save the results of the metrics

dsignore = ['TSMP-old', 'TSMP-old-EURregLonLat025deg'] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# List of datasets to plot, do not add the reference dataset, it will be added automatically from dsref
dstoplot = [
    'IMERG-V07B-30min-EURregLonLat025deg', 'IMERG-V06B-30min-EURregLonLat025deg',
    'GSMaP-EURregLonLat025deg',
    'TSMP-DETECT-Baseline-EURregLonLat025deg',
    "ERA5-hourly",
    # 'RADOLAN-EURregLonLat025deg', 'EURADCLIM-EURregLonLat025deg',
    ]


to_plot0 = data_hourlysum.copy()

tsel = [timesel]
ignore_incomplete = True # flag for ignoring datasets that do not cover the complete period. Only works for specific periods (not for slice(None, None))

selmonthlist = [("", [1,2,3,4,5,6,7,8,9,10,11,12]),
                ("DJF" , [12, 1, 2]),
                ("JJA", [6, 7, 8]),
           ] # ("nameofmonth", [month])

dataref = to_plot0[dsref[0]]
mask0 = mask.mask(dataref)
if mask_TSMP_nudge: mask0 = TSMP_no_nudge.mask(dataref).where(mask0.notnull())

amount = {}
frequency = {}
intensity = {}

print("Calculating diurnal cycles ...")
for tseln in tsel:
    timeperiodn = "_".join([tseln.start, tseln.stop])
    amount[timeperiodn] = {}
    frequency[timeperiodn] = {}
    intensity[timeperiodn] = {}
    print("... "+timeperiodn)
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])
        amount[timeperiodn][selmonth[0]] = {}
        frequency[timeperiodn][selmonth[0]] = {}
        intensity[timeperiodn][selmonth[0]] = {}

        for dsname in dstoplot+dsref:
            print(dsname)
            if dsname not in dsignore:
                if reload_cycles:
                    cyclessavepath = cyclessavepath0+timeperiodn+"/diurnal_cycles/"+region_name+"/"
                    amount[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = xr.open_dataset(os.path.join(cyclessavepath, f"amount_{selmonth[0]}_{reduce_dsname(dsname)}.nc"))
                    frequency[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = xr.open_dataset(os.path.join(cyclessavepath, f"frequency_{selmonth[0]}_{reduce_dsname(dsname)}.nc"))
                    intensity[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = xr.open_dataset(os.path.join(cyclessavepath, f"intensity_{selmonth[0]}_{reduce_dsname(dsname)}.nc"))
                    print("Cycles for "+region_name+" "+timeperiodn+" "+selmonth[0]+" "+dsname+" were reloaded")
                else:
                    to_plot = to_plot0[dsname].copy()
                    if type(to_plot) is xr.DataArray: to_plot = to_plot.to_dataset()
                    if tseln.start is not None and tseln.stop is not None: # add specific period to title

                        if ignore_incomplete:
                            if not (to_plot.time[0].dt.date <= datetime.strptime(tseln.start, "%Y-%m-%d").date() and
                                    to_plot.time[-1].dt.date >= datetime.strptime(tseln.stop, "%Y-%m-%d").date()):
                                print("Ignoring "+dsname+" because it does not cover the requested period")
                                continue

                    if "ERA5" in dsname:
                        # For ERA5 there are tiny differences in the coordinates that I need to solve, otherwise the datasets
                        # will not align and the .where() will fail
                        to_plot = to_plot.interp_like(mask0, method="nearest")

                    # Select the given season and mask
                    to_plot = to_plot.sel(time=tseln)
                    to_plot = to_plot.sel(time=to_plot['time'].dt.month.isin(selmonth[1])).reindex(lat=mask0.lat, lon=mask0.lon, method="nearest").where(mask0.notnull(), drop=True)

                    for vv in var_names:
                        if vv in to_plot.data_vars:
                            st = time.time()
                            amount[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = to_plot[vv].groupby("time.hour").mean("time").compute()
                            intensity[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = to_plot[vv].where(to_plot[vv]>=minpre).groupby("time.hour").mean("time").compute()
                            frequency[timeperiodn][selmonth[0]][reduce_dsname(dsname)] = (to_plot[vv]>=minpre).groupby("time.hour").sum("time").compute()/to_plot[vv].groupby("time.hour").count("time").compute()*100

                            et = time.time() - st
                            print(f" took {et/60:.2f} minutes.")
                            # Plot
                            break


if not reload_cycles:
    # Save the diurnal cycles to files
    print("Saving the diurnal cycles to files")
    for metric_dict, name in zip([amount, intensity, frequency],
                                 ['amount', 'intensity', 'frequency']):
        for timeperiodn in metric_dict.keys():
            cyclessavepath = cyclessavepath0+timeperiodn+"/diurnal_cycles/"+region_name+"/"
            if not os.path.exists(cyclessavepath):
                os.makedirs(cyclessavepath)
            for seasname in metric_dict[timeperiodn].keys():
                for dsname in metric_dict[timeperiodn][seasname].keys():

                    # Convert the data to an xarray Dataset and save as NetCDF
                    metric_dataset = xr.Dataset({f"{name}": metric_dict[timeperiodn][seasname][dsname]})
                    metric_filename = os.path.join(cyclessavepath, f"{name}_{seasname}_{dsname}.nc")
                    metric_dataset.to_netcdf(metric_filename)
                    print(f"Saved {seasname} {name} for {dsname} to {metric_filename}")

#%%%%% Plot mean diurnal cycles (amount, frequency, intensity)
savepathbase = "/automount/agradar/jgiles/images/gridded_datasets_intercomparison/hourly/"+region_name+"/mean_diurnal_cycles/"

# List of datasets to plot, INCLUDING REFERENCE DATASET
dstoplot = [
    'IMERG-V07B', 'IMERG-V06B',
    'GSMaP',
    'TSMP-DETECT',
    "ERA5",
    # 'RADOLAN', 'EURADCLIM', "RADKLIM",
    ]

dsignore = ['TSMP-old', 'TSMP-old-EURregLonLat025deg'] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', "E-OBS", "CPC"] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

print("Plotting mean diurnal cycles ...")
for tseln in tsel:
    timeperiodn = "_".join([tseln.start, tseln.stop])
    print("... "+timeperiodn)
    for selmonth in selmonthlist:
        print("... ... "+selmonth[0])

        amount0 = amount[timeperiodn][selmonth[0]]
        frequency0 = frequency[timeperiodn][selmonth[0]]
        intensity0 = intensity[timeperiodn][selmonth[0]]

        for to_plot0, name, units in zip([amount0, frequency0, intensity0],
                                         ['amount', 'frequency', 'intensity'],
                                         ["mm", "%", "mm"]):
            for dsname in dstoplot:
                if dsname not in dsignore:
                    to_plot = to_plot0[dsname].copy()

                    # # Plot
                    # plt.figure(figsize=(7,5))
                    # ax = plt.gca()

                    to_plot.mean(("lon", "lat")).plot(label = dsname, color=colors[dsname])


            # Beautify plot
            plt.title('Diurnal cycle '+name+" \n"+region_name+" "+selmonth[0]+" "+timeperiodn)
            plt.xlabel('hour')
            plt.ylabel(units)

            plt.grid()
            plt.legend(loc=(1.05, 0.3))

            # Save the plot
            plt.tight_layout()

            # plt.show()

            savepathn = savepathbase+timeperiodn+"/"
            savefilenamen = "_".join([name,"diurnal_cycle", selmonth[0], timeperiodn])+".png"
            if not os.path.exists(savepathn):
                os.makedirs(savepathn)
            plt.savefig(savepathn+savefilenamen, bbox_inches="tight")

            plt.close()


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

# turkey = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Turkey"][0]
# # Display turkey's shape
# shape_feature = cartopy.feature.ShapelyFeature([turkey.geometry], ccrs.PlateCarree(), facecolor="#932e2e", edgecolor='black', lw=1)
# ax.add_feature(shape_feature)

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


# Plot euradclim coverage as points
euradclim_valid = euradclim.Precip[0].notnull()
ax.scatter(euradclim.lon.where(euradclim_valid).values.flatten()[::5],
           euradclim.lat.where(euradclim_valid).values.flatten()[::5],
           transform=ccrs.PlateCarree(), color="#1d7e75ff", zorder=1)



# Plot eobs coverage as points
eobs_valid = eobs.rr[0].notnull()
eobs_valid.where(eobs_valid).plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, alpha=0.5, cmap="Oranges")
# ax.scatter(euradclim.lon.where(euradclim_valid).values.flatten()[::5], euradclim.lat.where(euradclim_valid).values.flatten()[::5], transform=ccrs.PlateCarree())


# Plot 60 N line
ax.plot(np.arange(0,361), [60]*361, transform=ccrs.PlateCarree(), color="#cc0000")

# Remove title
plt.title("")

# # Make figure larger
plt.gcf().set_size_inches(20, 10)

plt.savefig("/automount/agradar/jgiles/images/gridded_datasets_intercomparison/custom/spatial_coverage_globe.png",
            transparent=True, bbox_inches="tight")

