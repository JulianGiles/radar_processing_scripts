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

# Special selections for incomplete extreme years
# IMERG
data_yearlysum["IMERG-V07B-monthly"] = data_yearlysum["IMERG-V07B-monthly"].loc[{"time":slice("2001", "2022")}]
data_yearlysum["IMERG-V06B-monthly"] = data_yearlysum["IMERG-V06B-monthly"].loc[{"time":slice("2001", "2020")}]
# CMORPH
data_yearlysum["CMORPH-daily"] = data_yearlysum["CMORPH-daily"].loc[{"time":slice("1998", "2022")}]
# GPROF
data_yearlysum["GPROF"] = data_yearlysum["GPROF"].loc[{"time":slice("2015", "2022")}]

#%%% Regional averages
#%%%% Calculate area means (regional averages)
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
data_yearlysum = data_to_avg

#%%%% Simple map plot
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

#%%%% Interannual variability area-means plot
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

#%%%% Interannual variability area-means plot (interactive html)

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

#%%% BIAS and ERRORS
#%%%% Bias (absolute and relative) calculation from regional averages
var_names = ["TOT_PREC", "precipitation", "pr", "surfacePrecipitation", "precip", "Precip", 
             "RW", "RR", "tp", "cmorph"]

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
    plt.bar(bar_positions, value_padded, width=bar_width, label=key)

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
    plt.bar(bar_positions, value_padded, width=bar_width, label=key)

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
# We use the DETECT 1 km grid for thi^s
to_add = {} # dictionary to add regridded versions
for dsname in ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM"]:
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
dsignore = ["EURADCLIM", "RADOLAN", "HYRAS", "RADKLIM", 'TSMP-old', 'TSMP-DETECT-Baseline'] # datasets to ignore 
data_to_bias = copy.deepcopy(data_yearlysum)

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
                    else:
                        data0 = data_to_bias[dsname][vv]

                    data0 = data0.where(data0>0)
                    dataref = data_to_bias[dsref[0]][vvref]
                    
                    mask = rmcountries[[region]].mask(data_to_bias[dsref[0]])

                    data_bias_map[dsname] = ( data0.where(mask.notnull()) - dataref.where(mask.notnull()) ).compute()
                    data_bias_relative_map[dsname] = ( data_bias_map[dsname] / dataref.where(mask.notnull()) ).compute() *100
                    data_abs_error_map[dsname] = abs(data_bias_map[dsname])
                    
                    data_bias_relative_gp[dsname] = utils.calc_spatial_integral(data_bias_map[dsname],
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100
                    
                    data_norm_mean_abs_error_gp[dsname] = utils.calc_spatial_integral(data_abs_error_map[dsname],
                                                lon_name="lon", lat_name="lat").compute() / \
                                                    utils.calc_spatial_integral(dataref.where(mask.notnull()),
                                                lon_name="lon", lat_name="lat").compute() *100

                    data_mean_abs_error_gp[dsname] = utils.calc_spatial_mean(data_abs_error_map[dsname],
                                                lon_name="lon", lat_name="lat").compute()
                    
                    break
                    
#%%%% Relative bias and error plots
#%%%%% Simple map plot
to_plot = data_bias_relative_map
dsname = "RADKLIM_GPCC-monthly-grid"
title = "Relative BIAS"
yearsel = "2016"
rmcountries = rm.defined_regions.natural_earth_v5_1_2.countries_10
mask = rmcountries[["Germany"]].mask(to_plot[dsname])
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = to_plot[dsname].loc[{"time":yearsel}].where(mask.notnull(), drop=True).plot(x="lon", y="lat", cmap="RdBu_r", vmin=-100, vmax=100, 
                                         subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                         cbar_kwargs={'label': "%", 'shrink':0.88})
# ax1.set_extent([float(a) for a in lonlat_limits])
plot.axes.coastlines(alpha=0.7)
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
plt.title(title+" "+yearsel+"\n"+dsname+"\n Ref.: "+dsref[0])


#%%%%% Box plots of BIAS and ERRORS
# the box plots are made up of the yearly bias or error values
to_plot = data_norm_mean_abs_error_gp # data_mean_abs_error_gp # data_bias_relative_gp
title = "Normalized mean absolute error (yearly values)"
ylabel = "%" # %
dsignore = ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', ] # ['CMORPH-daily', 'GPROF', 'HYRAS_GPCC-monthly-grid', ] #['CMORPH-daily', 'RADKLIM', 'RADOLAN', 'EURADCLIM', 'GPROF', 'HYRAS', "IMERG-V06B-monthly", "ERA5-monthly"] # datasets to ignore in the plotting

# Initialize a figure and axis
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

# Create a list to hold the data arrays
plotted_arrays = []
plotted_arrays_lengths = []

# Iterate over the datasets in the dictionary
for key, value in to_plot.items():
    if key not in dsignore:
        # Plot a box plot for each dataset
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
                   rotation=45)

# Make a secondary x axis to display the number of values in each box
ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("top")

ax2.set_xticks(range(1, len(plotted_arrays) + 1))
ax2.set_xticklabels(plotted_arrays_lengths)
ax2.set_xlabel('Number of years')

# Set labels and title
#ax.set_xlabel('')
ax.set_ylabel(ylabel)
ax.set_title(title)

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()


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
