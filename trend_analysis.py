# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

import warnings
#ignore by message
# warnings.filterwarnings("ignore", message="Default reduction dimension will be changed to the grouped dimension")
# warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
# warnings.filterwarnings("ignore", message="Setting the 'color' property will override the edgecolor or facecolor properties.")
# warnings.filterwarnings("ignore", category=FutureWarning)

proj = ccrs.PlateCarree(central_longitude=0.0)

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

#%% Load data

# Put everything on a single dictionary
data = dict()

#### GRACE GDO
# TWSA from Copernicus Global Drought Observatory (GDO), reference period 2002-2017
# https://data.jrc.ec.europa.eu/dataset/0fd62e28-241f-472c-8966-98744920e181#dataaccess
print("Loading GRACE-GDO...")
data["GRACE-GDO"] = xr.open_mfdataset('/automount/ags/jgiles/GRACE_TWS/europe/twsan_m_euu_*_m.nc')

#### GRACE GSFC
# GSFC.glb.200204_202207_RL06v2.0_OBP-ICE6GD_HALFDEGREE.nc
# For mascons, the difference between OBP and SLA is basically only over the ocean,
# where some people prefer to compare to ocean bottom pressure installations, i.e. include 
# atmospheric effects on the total vertical column. I am writing this because 
# the .5° grid is only available in the OBP variant, but SLA makes more sense for mass 
# budgets. Over continents, you should be fine with both variants, though.
# Grid: 0.5x0.5 degrees (the resolution on this one is because of the processing but the raw data has 1 degree res.)
# https://earth.gsfc.nasa.gov/geo/data/grace-mascons
print("Loading GRACE-GSFC...")
data["GRACE-GSFC"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/gsfc.glb_.200204_202207_rl06v2.0_obp-ice6gd_halfdegree.nc')

#### GRACE ITSG (Uni Bonn)
# Ben: I recommend to use ITSG, it is the longest GRACE series from spherical harmonics that
# we currently have processed there (181 months over 2003-01/2020-09). Be warned that there
# are missing months in all versions of GRACE data, as the batteries started degrading after 2011,
# and mid-of-2016 until GRACE's end in 2017 is very noisy (many people do not even touch those). 
# GRACE-FO starts in 2018, some missing months in the beginning, but very robust since then. 
# No inter-sat calibration necessary, just treat all months as one data set.
print("Loading GRACE-ITSG...")
grace_itsgdates = xr.open_mfdataset("/automount/ags/jgiles/4BenCharlotte/ITSG_UB/mmyyyy.nc")
grace_itsgyears = grace_itsgdates.year[:,0].data.astype(int)
grace_itsgmonths = grace_itsgdates.month[:,0].data.astype(int)
datetimes = np.array([datetime(y, m, 1) for y, m in zip(grace_itsgyears, grace_itsgmonths)])

def preprocess_grace_itsg(ds): # reshape dataset according to coordinates
    lats = np.unique(ds.lat)*-1
    lons = np.unique(ds.lon)
    
    ewh_reshaped = ds["ewh"].data.reshape(len(lons), len(lats)).T
    ds0 = ds.drop_vars(["lon", "lat"])
    ds0['ewh'] = xr.DataArray(ewh_reshaped, coords={"lon":lons, "lat":lats}, dims=('lat', 'lon')).isel(lat=slice(None, None, -1))
    ds0['ewh'].attrs = ds['ewh'].attrs
    return ds0.expand_dims("time")

data["GRACE-ITSG"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/ITSG_UB/ewh/*', combine="nested",
                               preprocess= preprocess_grace_itsg, concat_dim = "time")

data["GRACE-ITSG"].coords["time"] = datetimes


#### IMERG (GLOBAL MONTHLY)
print("Loading IMERG...")
def preprocess_imerg(ds):
    # function to transform to accumulated monthly values
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    ds["precipitation"] = ds["precipitation"]*days_in_month[ds.time.values[0].month-1]*24
    return ds
data["IMERG"] = xr.open_mfdataset('/automount/ags/jgiles/IMERG/global_monthly/3B-MO.MS.MRG.3IMERG.*.V06B.HDF5.nc4', preprocess=preprocess_imerg)\
                .transpose('time', 'lat', 'lon', ...) # * 24*30 # convert to mm/month (approx)


#### ERA5 (GLOBAL MONTHLY)
print("Loading ERA5...")
data["ERA5"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/monthly_averaged/single_level_vars/volumetric_soil*/volumetric_*')
data["ERA5"] = data["ERA5"].assign_coords(longitude=(((data["ERA5"].longitude + 180) % 360) - 180)).sortby('longitude')
data["ERA5"]['swvl_total'] = data["ERA5"]['swvl1']*0.07 + data["ERA5"]['swvl2']*0.21 + \
                            data["ERA5"]['swvl3']*0.72 + data["ERA5"]['swvl4']*1.89


#### TSMP
# The timestamps of accumulated P are located at the center of the corresponding interval (1:30, 4:30, ...)
# Also, every monthly file has the last timestep from the previous month because of how the data is outputted by 
# the model, so some data are overlapped. The overlaped data are negligible different (around the 5th decimal)

print("Loading TSMP...")
def preprocess_tsmp(ds): # discard the first timestep of every monthly file (discard overlapping data)
    return ds.isel({"time":slice(1,None)})
    
data["TSMP"] = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*TOT_PREC*',
                             preprocess=preprocess_tsmp, chunks={"time":1000})

data["TSMP"] = data["TSMP"].assign( xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*WT*',
                             chunks={"time":1000}) )
data["TSMP"]["time"] = data["TSMP"].get_index("time").shift(-1.5, "H") # shift the values forward to the start of the interval


data["TSMP-Ben"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/TSMP/twsa_tsmp_europe_era5_1990_2021.nc')


#### TWS reanalysis (Anne Springer)
print("Loading Springer-Rean...")
data["Springer-Rean"] = xr.open_mfdataset("/automount/ags/jgiles/springer_reanalysis/clmoas_scenario0033_ensMean.TWS.updateMat.200301-201908_monthly.nc")
data["Springer-Rean-grid"] = xr.open_mfdataset("/automount/ags/jgiles/springer_reanalysis/CORDEX0.11_grid_424x412.nc")


#%% Get common period
start_dates = []
end_dates = []
for dsname in data.keys():
    if dsname == "Springer-Rean-grid": continue
    if data[dsname].time.dtype == "O":
        datetimeindex = data[dsname].indexes['time'].to_datetimeindex()
        data[dsname]["time"] = datetimeindex
    start_dates.append( data[dsname].time[0].values )
    end_dates.append( data[dsname].time[-1].values )

# Select the period

# start_date = '2003-01-01'
# end_date = '2016-12-31'

start_date = np.array(start_dates).max()
end_date = np.array(end_dates).min()

# lon_slice = slice(-17,48)

data_timesel = dict()

for dsname in data.keys():
    if dsname == "Springer-Rean-grid": continue
    data_timesel[dsname] = data[dsname].loc[{"time": slice(start_date, end_date)}]

#%% Get trends

# xarray's polyfit gives the coefficients based in units per nanosecond, which is the basic time unit of xarray
# https://stackoverflow.com/questions/70713838/can-someone-explain-the-logic-behind-xarray-polyfit-coefficients

# Only process variables that are relevant
vars_to_keep ={
                "GRACE-GDO": ["twsan"],
                "GRACE-GSFC": ["lwe_thickness"],
                "GRACE-ITSG": ["ewh"],
                "IMERG": ["precipitation"],
                "ERA5": ["swvl_total"],
                "TSMP": ["TOT_PREC", "WT"],
                "TSMP-Ben": ["TWSA with snow", "groundwater storage", "surface water storage", "shallow water storage"],
                "Springer-Rean": ["TWS"],
                }

# For the datasets in monthly resolution
data_monthly = dict() # all data is already monthly except for TSMP
trends = dict()

for dsname in data_timesel.keys():
    # Populate dict of monthly data, leave only data that is relevant
    if dsname == "TSMP":
        data_monthly[dsname] = data_timesel[dsname][vars_to_keep[dsname]].resample({"time": "MS"}).mean().compute()
    else:
        data_monthly[dsname] = data_timesel[dsname][vars_to_keep[dsname]]
    
# TSMP gives wierd results if passing to monthly, so we go directly to yearly and then trends
data_monthly["TSMP_yearly"] = data_timesel["TSMP"][vars_to_keep[dsname]].resample({"time": "YS"}).sum().compute()

    
# Get trends
for dsname in data_monthly.keys():
    print("Calculating trends on "+dsname)
    trends[dsname] = data_monthly[dsname].polyfit(dim='time', deg=1, skipna=True).compute()


# Repeat for yearly resolution
data_yearly = dict() 
trends_yearly = dict()

for dsname in data_timesel.keys():
    data_yearly[dsname] = data_timesel[dsname][vars_to_keep[dsname]].resample({"time": "YS"}).mean().compute()

# Get trends
for dsname in data_yearly.keys():
    print("Calculating trends on "+dsname)
    trends_yearly[dsname] = data_yearly[dsname].polyfit(dim='time', deg=1, skipna=True).compute()

#%% PLOT

# Plot map of trends (remember that the coefficient is in units per nanosecond)
def plot_trend(ds, proj=ccrs.PlateCarree(central_longitude=0.0), unit_label="1/year", title="",
               lonlat_limits=None, vlims=[None, None], cmap="RdBu", gridlines=True, **kwrgs):
    '''
    ds : DataArray of polyfit_coefficients
    lonlat_limits : [lonmin, lonmax, latmin, latmax]
    '''
    nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
        
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))

    plot = (ds[0]*nanosec_to_year).plot(ax=ax1, cmap=cmap, levels = 21,
                                        vmin=vlims[0], vmax=vlims[1],
                                    cbar_kwargs={'label': unit_label, 'shrink':0.88}, **kwrgs)
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=gridlines)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    # set extent of map
    if lonlat_limits is not None:
        ax1.set_extent([lonlat_limits[0], lonlat_limits[1], lonlat_limits[2], lonlat_limits[3]], crs=ccrs.PlateCarree(central_longitude=0.0))
        
    # Set title
    plt.title(title)

# Set the limits

lon_limits = [float(data_monthly["GRACE-GDO"].lon[0]), float(data_monthly["GRACE-GDO"].lon[-1])] # [25, 45] [6,15]
lat_limits = [float(data_monthly["GRACE-GDO"].lat[0]), float(data_monthly["GRACE-GDO"].lat[-1])] # [35.5, 42.5] [47, 55]
lonlat_limits = lon_limits + lat_limits

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))


dsnames = ["GRACE-GDO", "GRACE-GSFC", "GRACE-ITSG", "IMERG", "TSMP", "TSMP-Ben", "Springer-Rean"]
unitss = ["1/year", "cm/year", "mm/year", "mm/year", "mm/year", "mm/year", "mm/year"]
varss = ["twsan", "lwe_thickness", "ewh", "precipitation", "TOT_PREC", "TWSA with snow", "TWS"]
vlimss =  [ [None, None], [-1, 1], [-5,5], [-20, 20], [-20, 20], [-10, 10], [-5, 5] ]

select = 2
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"], unit_label=units, 
           title=title, lonlat_limits=lonlat_limits, vlims=vlims)



# For Springer-Rean
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))

plot = plt.pcolormesh(data["Springer-Rean-grid"]["LONGXY"],
                      data["Springer-Rean-grid"]["LATIXY"],
                      trends_yearly[dsname][var+"_polyfit_coefficients"][0]*nanosec_to_year,
                      axes=ax1, vmin=vlims[0], vmax=vlims[1], cmap="RdBu")

plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
# set extent of map
if lonlat_limits is not None:
    ax1.set_extent([lonlat_limits[0], lonlat_limits[1], lonlat_limits[2], lonlat_limits[3]], crs=proj)
    
# Set title
plt.title(title)
plt.colorbar(levels=21)


#%% PLOT OVER GERMANY AND TURKEY with radar sites

cmap="BrBG"
namescolor="tomato"

# TURKEY GRACE with radars
# Set the limits

lon_limits = [25, 45] # [25, 45] [5,16]
lat_limits = [35.5, 42.5] # [35.5, 42.5] [47, 56]
lonlat_limits = lon_limits + lat_limits

select = 2
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"], unit_label=units, 
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
           proj=ccrs.Mercator(), transform=ccrs.PlateCarree(central_longitude=0.0))

# load radar sites of Turkey
radar_range = 18000 # in points in the plot (18000 = 350 km survallience, 8500=250 km volume task)
turk_radars = pd.read_csv("/home/jgiles/Scripts/python/turkish_radars.csv")
selected_radars = ["Hatay", "Gaziantep", "Afyonkarahisar", "Sivas", "Ankara"]
single_pol = ["Trabzon", "Samsun"]


# Plot turkish radar sites
pc=ccrs.PlateCarree(central_longitude=0.0)
plot_radar_sites = True
if plot_radar_sites:
    for sr in selected_radars: # plot selected radar ranges
        g= plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                    turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='black', marker='o', s= radar_range, alpha=0.1)
        g.set_facecolor('none')
    for sr in selected_radars: # plot selected radar locations
        plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                    turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='red', marker='x')
        name = sr
        if turk_radars[turk_radars["Name"]==sr]["Ins. Year"].values > 2018: # plot selected radar names
            name+=str(turk_radars[turk_radars["Name"]==sr]["Ins. Year"].values)
        if sr in single_pol:
            name+='(S)'
        plt.text(turk_radars[turk_radars["Name"]==sr]["Longitude"].values+0.2,
                 turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                 name, c=namescolor, transform=pc)
    for sr in turk_radars["Name"].to_list():
        if sr not in selected_radars:
            plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                        turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                        transform=pc, color='blue', marker='x')
            

# GERMANY GRACE with radars
# Set the limits

lon_limits = [5,16] # [25, 45] [5,16]
lat_limits = [47, 56] # [35.5, 42.5] [47, 56]
lonlat_limits = lon_limits + lat_limits

select = 2
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"], unit_label=units, 
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
           proj=ccrs.Mercator(), transform=ccrs.PlateCarree(central_longitude=0.0))

# load radar sites of Germany
radar_range = 5000 # in points in the plot (18000 = 350 km survallience, 8500=250 km volume task)
ger_radars = pd.read_csv("/home/jgiles/Scripts/python/german_radars.csv")
selected_radars = ["PRO", "TUR", "UMD"]
single_pol = []


# Plot german radar sites
pc=ccrs.PlateCarree(central_longitude=0.0)
plot_radar_sites = True
if plot_radar_sites:
    for sr in selected_radars: # plot selected radar ranges
        g= plt.scatter(ger_radars[ger_radars["Name"]==sr]["Longitude"].values,
                    ger_radars[ger_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='black', marker='o', s= radar_range, alpha=0.1)
        g.set_facecolor('none')
    for sr in selected_radars: # plot selected radar locations
        plt.scatter(ger_radars[ger_radars["Name"]==sr]["Longitude"].values,
                    ger_radars[ger_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='red', marker='x')
        name = sr
        if ger_radars[ger_radars["Name"]==sr]["Ins. Year"].values > 2018: # plot selected radar names
            name+=str(ger_radars[ger_radars["Name"]==sr]["Ins. Year"].values)
        if sr in single_pol:
            name+='(S)'
        plt.text(ger_radars[ger_radars["Name"]==sr]["Longitude"].values+0.2,
                 ger_radars[ger_radars["Name"]==sr]["Latitude"].values,
                 name, c=namescolor, transform=pc)
    for sr in ger_radars["Name"].to_list():
        if sr not in selected_radars:
            plt.scatter(ger_radars[ger_radars["Name"]==sr]["Longitude"].values,
                        ger_radars[ger_radars["Name"]==sr]["Latitude"].values,
                        transform=pc, color='blue', marker='x')



# TURKEY IMERG and TSMP
# Set the limits

lon_limits = [25, 45] # [25, 45] [5,16]
lat_limits = [35.5, 42.5] # [35.5, 42.5] [47, 56]
lonlat_limits = lon_limits + lat_limits

select = 3 # select IMERG
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"]*12, unit_label=units, # I think I have to multiply by 12 to get yearly trends, otherwise the values are too small
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
            proj=ccrs.Mercator(), transform=ccrs.PlateCarree(central_longitude=0.0)
           )


select = 4 # select TSMP
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"]*8*365, unit_label=units, # I think I have to multiply by 8 and 365 to get yearly trends, otherwise the values are too small
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
            proj=ccrs.Mercator(), transform=rp
           )



# GERMANY IMERG and TSMP
# Set the limits

lon_limits = [5,16] # [25, 45] [5,16]
lat_limits = [47, 56] # [35.5, 42.5] [47, 56]
lonlat_limits = lon_limits + lat_limits

select = 3 # select IMERG
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"]*12, unit_label=units, # I think I have to multiply by 12 to get yearly trends, otherwise the values are too small
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
            proj=ccrs.Mercator(), transform=ccrs.PlateCarree(central_longitude=0.0)
           )


select = 4 # select TSMP
dsname = dsnames[select]
units = unitss[select]
var = varss[select]
vlims = vlimss[select]

title = str(start_date)[0:10]+" to "+str(end_date)[0:10]+" "+var+" trend "+dsname
plot_trend(trends_yearly[dsname][var+"_polyfit_coefficients"]*8*365, unit_label=units, # I think I have to multiply by 8 and 365 to get yearly trends, otherwise the values are too small
           title=title, lonlat_limits=lonlat_limits, vlims=vlims, cmap=cmap, gridlines=False,
            proj=ccrs.Mercator(), transform=rp
           )



#%% MEAN VALUES AND DIFFERENCES

# get a common period
start_date = str(data["IMERG"].time[7].to_numpy())
end_date = str(data["IMERG"].time[-10].to_numpy())

# accumulate to yearly values
data_yearlysum = {}
data_yearlysum["IMERG"] = data["IMERG"].loc[{"time": slice(start_date, end_date)}]["precipitation"].resample({"time": "YS"}).sum().compute()
data_yearlysum["TSMP"] = data["TSMP"]["TOT_PREC"].loc[{"time": slice(start_date, end_date)}].resample({"time": "YS"}).sum().compute()

# Transform the TSMP data from rotated pole grid to rectiliniar. We will need this for the differences.
# We use CDO because I cannot find a better way.
# Thus, we save the TSMP dataset to nc, transform with CDO and reload
data_yearlysum["TSMP"].to_netcdf("/automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_"+start_date.split("-")[0]+"-"+end_date.split("-")[0]+".nc")
import subprocess
bash_code="""
cdo remapbil,/automount/ags/jgiles/IMERG/global_monthly/griddes.txt /automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020.nc /automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020_IMERGrotated0.nc
cdo sellonlatbox,-46.40,67.22,21.14,73.24 /automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020_IMERGrotated0.nc /automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020_IMERGrotated.nc
"""
result = subprocess.run(bash_code, shell=True, check=True, capture_output=True, text=True)
print(result.stdout)

data_TSMP_rot = xr.open_dataset("/automount/ags/jgiles/TSMP/postprocessed/TSMP_TOT_PREC_yearlysum_2001-2020_IMERGrotated.nc")["TOT_PREC"]

#%% Plot one year
isel=0
proj=ccrs.Mercator()
country="Turkey"
lonlat_limits_country = {"Turkey": [25, 45, 35.5, 42.5],
                         "Germany": [5, 16, 47, 56]}


for isel in np.arange(0,20):
    cmap="Greens"
    nlevs = 11
    vlims = [0,2000]
    
    # Plot TSMP
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    plot = data_yearlysum["TSMP"][isel].plot(cmap=cmap, vmin=vlims[0], vmax=vlims[1], levels=nlevs, 
                                             subplot_kws={"projection":proj}, transform=rp, 
                                             cbar_kwargs={'label': "mm", 'shrink':0.88})
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    if len(country)>0: # focus on country domain
        ax1.set_extent([lonlat_limits_country[country][0], 
                        lonlat_limits_country[country][1], 
                        lonlat_limits_country[country][2], 
                        lonlat_limits_country[country][3]])        
    getyear = str(data_yearlysum["TSMP"][isel].time.values).split("-")[0]
    plt.title("TSMP TOT_PREC "+getyear)
    dest = "/home/jgiles/sciebo/Images/TSMP_IMERG_comparison/"+getyear+"/"+country+"/"
    if not os.path.exists(dest):
        os.makedirs(dest)
    plt.savefig(dest+"TSMP_TOT_PREC_yearsum_"+getyear+"_"+country+".png",  bbox_inches='tight')
    plt.close(f) # so the figure is not displayed in Spyder
    
    # Plot IMERG
    # Get lon lat limits from TSMP
    lonlat_limits = [data_yearlysum["TSMP"][isel].lon.min(),
                     data_yearlysum["TSMP"][isel].lon.max(),
                     data_yearlysum["TSMP"][isel].lat.min(),
                     data_yearlysum["TSMP"][isel].lat.max()]
    
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    plot = data_yearlysum["IMERG"][isel].loc[{"lon":slice(lonlat_limits[0], lonlat_limits[1]),
                                              "lat":slice(lonlat_limits[2], lonlat_limits[3])}].plot(cmap=cmap, vmin=vlims[0], vmax=vlims[1], levels=nlevs, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                             cbar_kwargs={'label': "mm", 'shrink':0.88})
    # ax1.set_extent([float(a) for a in lonlat_limits])
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    if len(country)>0: # focus on country domain
        ax1.set_extent([lonlat_limits_country[country][0], 
                        lonlat_limits_country[country][1], 
                        lonlat_limits_country[country][2], 
                        lonlat_limits_country[country][3]])        
    getyear = str(data_yearlysum["IMERG"][isel].time.values).split("-")[0]
    plt.title("IMERG precipitation "+getyear)
    dest = "/home/jgiles/sciebo/Images/TSMP_IMERG_comparison/"+getyear+"/"+country+"/"
    if not os.path.exists(dest):
        os.makedirs(dest)
    plt.savefig(dest+"/IMERG_precipitation_yearsum_"+getyear+"_"+country+".png",  bbox_inches='tight')
    plt.close(f) # so the figure is not displayed in Spyder
    
    
    # Plot difference TSMP-IMERG
    # subtract the fields
    diff = data_TSMP_rot[isel] - data_yearlysum["IMERG"][isel].loc[{"lon":slice(lonlat_limits[0], lonlat_limits[1]),
                                              "lat":slice(lonlat_limits[2], lonlat_limits[3])}]
    
    # Plot diff
    cmap="BrBG"
    nlevs = 21
    vlims = [-1000,1000]
    
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    plot = diff.plot(cmap=cmap, vmin=vlims[0], vmax=vlims[1], levels=nlevs, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                            cbar_kwargs={'label': "mm", 'shrink':0.88})
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    if len(country)>0: # focus on country domain
        ax1.set_extent([lonlat_limits_country[country][0], 
                        lonlat_limits_country[country][1], 
                        lonlat_limits_country[country][2], 
                        lonlat_limits_country[country][3]])        
    getyear = str( data_TSMP_rot[isel].time.values).split("-")[0]
    plt.title("TSMP TOT_PREC - IMERG precipitation "+getyear)
    dest = "/home/jgiles/sciebo/Images/TSMP_IMERG_comparison/"+getyear+"/"+country+"/"
    if not os.path.exists(dest):
        os.makedirs(dest)
    plt.savefig(dest+"/diff_TSMP-IMERG_precipitation_yearsum_"+getyear+"_"+country+".png",  bbox_inches='tight')
    plt.close(f) # so the figure is not displayed in Spyder
    
    
    # Plot diff in % of IMERG
    cmap="BrBG"
    nlevs = 21
    vlims = [-50,50]
    
    diff2 = diff/data_yearlysum["IMERG"][isel].loc[{"lon":slice(lonlat_limits[0], lonlat_limits[1]),
                                              "lat":slice(lonlat_limits[2], lonlat_limits[3])}]*100
    
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
    plot = diff2.plot(cmap=cmap, vmin=vlims[0], vmax=vlims[1], levels=nlevs, 
                                             subplot_kws={"projection":proj}, transform=ccrs.PlateCarree(),
                                            cbar_kwargs={'label': "%", 'shrink':0.88})
    plot.axes.coastlines(alpha=0.7)
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"}, visible=False)
    plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4) #countries
    if len(country)>0: # focus on country domain
        ax1.set_extent([lonlat_limits_country[country][0], 
                        lonlat_limits_country[country][1], 
                        lonlat_limits_country[country][2], 
                        lonlat_limits_country[country][3]])        
    getyear = str( data_TSMP_rot[isel].time.values).split("-")[0]
    plt.title("TSMP TOT_PREC - IMERG precipitation "+getyear)
    dest = "/home/jgiles/sciebo/Images/TSMP_IMERG_comparison/"+getyear+"/"+country+"/"
    if not os.path.exists(dest):
        os.makedirs(dest)
    plt.savefig(dest+"/diff_TSMP-IMERG_precipitation_yearsum_"+getyear+"_"+country+"_%.png",  bbox_inches='tight')
    plt.close(f) # so the figure is not displayed in Spyder
        

#%% INTERANNUAL VARIABILITY
country="Turkey"
lonlat_limits_country = {"Turkey": [25, 45, 35.5, 42.5],
                         "Germany": [5, 16, 47, 56]}

d1 = data_TSMP_rot.loc[{"lon":slice(lonlat_limits_country[country][0],
                                             lonlat_limits_country[country][1]),
                                 "lat":slice(lonlat_limits_country[country][2],
                                             lonlat_limits_country[country][3])}].mean(["lon","lat"])

d2 = data_yearlysum["IMERG"].loc[{"lon":slice(lonlat_limits_country[country][0],
                                             lonlat_limits_country[country][1]),
                                 "lat":slice(lonlat_limits_country[country][2],
                                             lonlat_limits_country[country][3])}].mean(["lon","lat"])

plt.plot(d1.time, d1, label="TSMP")
plt.plot(d1.time, d2, label="IMERG")
plt.legend()
plt.title("Annual precip "+country+" [mm]")

plt.plot(d1.time, d1-d1.mean(), label="TSMP")
plt.plot(d1.time, d2-d2.mean(), label="IMERG")
plt.legend()
plt.title("Annual precip anomaly "+country+" [mm]")


#%% OLD CODE FROM HERE ON

#%% Get yearly and seasonal values

grace_yearly = grace.twsan.resample({'time':"YS"}).mean().compute()
imerg_europe_yearly = imerg_europe.precipitation.resample({'time':"YS"}).mean().compute()
era5_europe_yearly = era5_europe.swvl_total.resample({'time':"YS"}).mean().compute()
tsmp_yearly = tsmp.pr.resample({'time':"YS"}).mean().compute()

# for xarray seasonal resample, the value for DJF falls on D of the previous year
grace_seas = grace.twsan.resample({'time':"QS-DEC"}).mean().compute()
imerg_europe_seas = imerg_europe.precipitation.resample({'time':"QS-DEC"}).mean().compute()
era5_europe_seas = era5_europe.swvl_total.resample({'time':"QS-DEC"}).mean().compute()
tsmp_seas = tsmp.pr.resample({'time':"QS-DEC"}).mean().compute()

#%% Get trends

# xarray's polyfit gives the coefficients based in units per nanosecond, which is the basic time unit of xarray
# https://stackoverflow.com/questions/70713838/can-someone-explain-the-logic-behind-xarray-polyfit-coefficients

# for the datasets in their native resolution
grace_trend = grace.twsan.polyfit(dim='time', deg=1, skipna=True).compute()
imerg_europe_trend = imerg_europe['precipitation'].polyfit(dim='time', deg=1, skipna=True).compute()
era5_europe_trend = era5_europe.swvl_total.polyfit(dim='time', deg=1, skipna=True).compute()
tsmp_trend = tsmp.pr.polyfit(dim='time', deg=1, skipna=True).compute()

# for the datasets in yearly resample
grace_yearly_trend = grace_yearly.polyfit(dim='time', deg=1, skipna=True).compute()
imerg_europe_yearly_trend = imerg_europe_yearly.polyfit(dim='time', deg=1, skipna=True).compute()
era5_europe_yearly_trend = era5_europe_yearly.polyfit(dim='time', deg=1, skipna=True).compute()
tsmp_yearly_trend = tsmp_yearly.polyfit(dim='time', deg=1, skipna=True).compute()

# for the datasets in seasonal resample
grace_seas_trend = dict()
imerg_europe_seas_trend = dict()
era5_europe_seas_trend = dict()
tsmp_seas_trend = dict()

for month,seas in zip((12, 3, 6, 9), ('DJF', 'MAM', 'JJA', 'SON')):
    if month == 12:
        grace_seas_trend[seas] = grace_seas.loc[{'time': grace_seas.time.dt.month.isin([month])}][:-1] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        imerg_europe_seas_trend[seas] = imerg_europe_seas.loc[{'time': imerg_europe_seas.time.dt.month.isin([month])}][:-1] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        era5_europe_seas_trend[seas] = era5_europe_seas.loc[{'time': era5_europe_seas.time.dt.month.isin([month])}][:-1] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        tsmp_seas_trend[seas] = tsmp_seas.loc[{'time': tsmp_seas.time.dt.month.isin([month])}][:-1] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()

    else:
        grace_seas_trend[seas] = grace_seas.loc[{'time': grace_seas.time.dt.month.isin([month])}] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        imerg_europe_seas_trend[seas] = imerg_europe_seas.loc[{'time': imerg_europe_seas.time.dt.month.isin([month])}] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        era5_europe_seas_trend[seas] = era5_europe_seas.loc[{'time': era5_europe_seas.time.dt.month.isin([month])}] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()
        tsmp_seas_trend[seas] = tsmp_seas.loc[{'time': tsmp_seas.time.dt.month.isin([month])}] \
                                .polyfit(dim='time', deg=1, skipna=True).compute()

#%% PLOT GRACE
# For the whole monthly dataset

# first plot a single pixel to check the trend
lon_slice = slice(10,10.9)
lat_slice = slice(48,48.9)
grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].plot()
plt.plot(grace.time, 
         grace_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*grace.time.astype(float) +
         grace_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0])

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (grace_trend.polyfit_coefficients[0]*nanosec_to_year).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'1/year', 'shrink':0.88,
                                                                               })
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in TWSA \n Original monthly data')
plt.scatter(grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
            grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
            transform=ccrs.PlateCarree(), color='black', marker='x')

#%%
# For the yearly resampled data

# first plot a single pixel to check the trend
lon_slice = slice(10,10.9)
lat_slice = slice(48,48.9)
grace_yearly.loc[{'lon': lon_slice, 'lat': lat_slice}].plot()
plt.plot(grace_yearly.time,
         grace_yearly_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*grace_yearly.time.astype(float) +
         grace_yearly_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0])

#%%
eur11 = cx.cordex_domain("EUR-11", dummy='topo')

# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (grace_yearly_trend.polyfit_coefficients[0]*nanosec_to_year).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'1/year', 'shrink':0.88,
                                                                               })
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in TWSA \n Yearly resampled data')
plt.scatter(grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
            grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
            transform=ccrs.PlateCarree(), color='black', marker='x')

# plot Euro-CORDEX domain limits
lons = (eur11.lon[0,:], eur11.lon[:,0], eur11.lon[-1,:], eur11.lon[:,-1])
lats = (eur11.lat[0,:], eur11.lat[:,0], eur11.lat[-1,:], eur11.lat[:,-1])
for lon, lat in zip(lons, lats):
    plt.plot(lon, lat, transform=ccrs.PlateCarree(), color='black')

f.savefig('/user/jgiles/GRACE_TWSA_trend.png', dpi=300)

#%%
# For the seasonal resampled data

# first plot a single pixel to check the trend
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, gridspec_kw={'hspace':0.3})
for nm, seas, ax in zip((0, 1, 2, 3), ('DJF', 'MAM', 'JJA', 'SON'), (ax1, ax2, ax3, ax4)):
    lon_slice = slice(10,10.9)
    lat_slice = slice(48,48.9)
    grace_seas[nm::4].loc[{'lon': lon_slice, 'lat': lat_slice}].plot(ax=ax)
    ax.plot(grace_seas[nm::4].time,
             grace_seas_trend[seas].polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*grace_seas[nm::4].time.astype(float) + 
             grace_seas_trend[seas].polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0])
    
    ax.set_title(seas)

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

grace_seas_trend_concat = xr.concat(grace_seas_trend.values(), pd.Index([nn for nn in grace_seas_trend], name="seas"))

f = (grace_seas_trend_concat.polyfit_coefficients[:,0,...]*nanosec_to_year).plot(cmap='RdBu', levels = 21,
                                                                                 col='seas', col_wrap=2,
                                                                                figsize=(8,4),
                                                                                subplot_kws={"projection": proj},
                                                                                cbar_kwargs={'label':'1/year', 'shrink':0.88,
                                                                                                    })

draw_labels_l = {"left": "y"}
draw_labels_lb = {"bottom": "x", "left": "y"}
draw_labels_b = {"bottom": "x"}
draw_labels = [draw_labels_l, {}, draw_labels_lb, draw_labels_b]
for ln, ax in enumerate(f.axes.flat):
    ax.coastlines()
    ax.gridlines(draw_labels=draw_labels[ln])
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries

    ax.scatter(grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
                grace.twsan.loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
                transform=ccrs.PlateCarree(), color='black', marker='x')

plt.suptitle(start_date + ' to ' + end_date + ' trend in TWSA \n Seasonal resampled data', y=1.1)

#%% PLOT ERA5
# For the whole monthly dataset

# first plot a single pixel to check the trend
lon_slice = slice(10,10.2)
lat_slice = slice(48,48.2)
era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].plot()
plt.plot(era5_europe.time, 
         era5_europe_trend.polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][0,0,0]*era5_europe.time.astype(float) +
         era5_europe_trend.polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][1,0,0])


#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (era5_europe_trend.polyfit_coefficients[0]*nanosec_to_year).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'m3/m2/year', 'shrink':0.88,
                                                                               },
                                                                  vmin=-1E-2, vmax=1E-2)
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in volumetric soil water \n Original monthly data')
plt.scatter(era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].longitude.values,
            era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].latitude.values,
            transform=ccrs.PlateCarree(), color='black', marker='x')

#%%
# For the yearly resampled data

# first plot a single pixel to check the trend
lon_slice = slice(10,10.2)
lat_slice = slice(48,48.2)
era5_europe_yearly.loc[{'longitude': lon_slice, 'latitude': lat_slice}].plot()
plt.plot(era5_europe_yearly.time, 
         era5_europe_yearly_trend.polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][0,0,0]*era5_europe_yearly.time.astype(float) +
         era5_europe_yearly_trend.polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][1,0,0])

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (era5_europe_yearly_trend.polyfit_coefficients[0]*nanosec_to_year).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'m3/m2/year', 'shrink':0.88,
                                                                               },
                                                                  vmin=-1E-2, vmax=1E-2)
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in volumetric soil water \n Yearly resampled data')
plt.scatter(era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].longitude.values,
            era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].latitude.values,
            transform=ccrs.PlateCarree(), color='black', marker='x')

#%%
# For the seasonal resampled data

# first plot a single pixel to check the trend
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, gridspec_kw={'hspace':0.3})
for nm, seas, ax in zip((0, 1, 2, 3), ('DJF', 'MAM', 'JJA', 'SON'), (ax1, ax2, ax3, ax4)):
    lon_slice = slice(10,10.2)
    lat_slice = slice(48,48.2)
    era5_europe_seas[nm::4].loc[{'longitude': lon_slice, 'latitude': lat_slice}].plot(ax=ax)
    ax.plot(era5_europe_seas[nm::4].time,
             era5_europe_seas_trend[seas].polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][0,0,0]*era5_europe_seas[nm::4].time.astype(float) + 
             era5_europe_seas_trend[seas].polyfit_coefficients.loc[{'longitude': lon_slice, 'latitude': lat_slice}][1,0,0])
    
    ax.set_title(seas)

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365

era5_europe_seas_trend_concat = xr.concat(era5_europe_seas_trend.values(), pd.Index([nn for nn in era5_europe_seas_trend], name="seas"))

f = (era5_europe_seas_trend_concat.polyfit_coefficients[:,0,...]*nanosec_to_year).plot(cmap='RdBu', levels = 21,
                                                                                 col='seas', col_wrap=2,
                                                                                figsize=(8,4),
                                                                                subplot_kws={"projection": proj},
                                                                                cbar_kwargs={'label':'m3/m2/year', 'shrink':0.88,
                                                                                                    },
                                                                                vmin=-1E-2, vmax=1E-2)

draw_labels_l = {"left": "y"}
draw_labels_lb = {"bottom": "x", "left": "y"}
draw_labels_b = {"bottom": "x"}
draw_labels = [draw_labels_l, {}, draw_labels_lb, draw_labels_b]
for ln, ax in enumerate(f.axes.flat):
    ax.coastlines()
    ax.gridlines(draw_labels=draw_labels[ln])
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries

    ax.scatter(era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].longitude.values,
                era5_europe['swvl_total'].loc[{'longitude': lon_slice, 'latitude': lat_slice}].latitude.values,
                transform=ccrs.PlateCarree(), color='black', marker='x')

plt.suptitle(start_date + ' to ' + end_date + ' trend in volumetric soil water \n Seasonal resampled data', y=1.1)

#%% PLOT IMERG

# For the whole monthly dataset
convert_units = 24*30 # mm/h to mm/month
# first plot a single pixel to check the trend
lon_slice = slice(11,11.1)
lat_slice = slice(50,50.1)
(imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}]*convert_units).plot()
plt.plot(imerg_europe.time,
         imerg_europe_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*convert_units*grace.time.astype(float) + \
        imerg_europe_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0]*convert_units)


#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
convert_units = 24*30 # mm/h to mm/month

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (imerg_europe_trend.polyfit_coefficients[0]*nanosec_to_year*convert_units).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'mm/month/year', 'shrink':0.88,
                                                                               },
                                                                  vmin=-1, vmax=1)
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in precipitation \n Original monthly data')
plt.scatter(imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
            imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
            transform=ccrs.PlateCarree(), color='black', marker='x')

#%%
# For the yearly resampled data
convert_units = 24*365 # mm/h to mm/y

# first plot a single pixel to check the trend
lon_slice = slice(11,11.1)
lat_slice = slice(50,50.1)
(imerg_europe_yearly.loc[{'lon': lon_slice, 'lat': lat_slice}]*convert_units).plot()
plt.plot(imerg_europe_yearly.time, 
         imerg_europe_yearly_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*convert_units*grace_yearly.time.astype(float) +
         imerg_europe_yearly_trend.polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0]*convert_units)

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
convert_units = 24*365 # mm/h to mm/y

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=proj))
plot = (imerg_europe_yearly_trend.polyfit_coefficients[0]*nanosec_to_year*convert_units).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'mm/year', 'shrink':0.88,
                                                                               },
                                                                  vmin=-20, vmax=20)
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(start_date + ' to ' + end_date + ' trend in precipitation \n Yearly resampled data')
plt.scatter(imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
            imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
            transform=proj, color='black', marker='x')

# # set extent of map
# lon_limits = [6,15] # [25, 45] [6,15]
# lat_limits = [47, 55] # [35.5, 42.5] [47, 55]
# ax1.set_extent([lon_limits[0], lon_limits[-1], lat_limits[0], lat_limits[-1]], crs=proj)

#%%
# For the seasonal resampled data
convert_units = 24*90 # mm/h to mm/seas

# first plot a single pixel to check the trend
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, gridspec_kw={'hspace':0.3})
for nm, seas, ax in zip((0, 1, 2, 3), ('DJF', 'MAM', 'JJA', 'SON'), (ax1, ax2, ax3, ax4)):
    lon_slice = slice(11,11.1)
    lat_slice = slice(50,50.1)
    (imerg_europe_seas[nm::4].loc[{'lon': lon_slice, 'lat': lat_slice}]*convert_units).plot(ax=ax)
    ax.plot(imerg_europe_seas.time.astype('datetime64[ns]').astype(float).time,
             imerg_europe_seas_trend[seas].polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][0,0,0]*convert_units*imerg_europe_seas.time.astype('datetime64[ns]').astype(float) + 
             imerg_europe_seas_trend[seas].polyfit_coefficients.loc[{'lon': lon_slice, 'lat': lat_slice}][1,0,0]*convert_units)
    
    ax.set_title(seas)

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
convert_units = 24*90 # mm/h to mm/seas

imerg_europe_seas_trend_concat = xr.concat(imerg_europe_seas_trend.values(), pd.Index([nn for nn in imerg_europe_seas_trend], name="seas"))

f = (imerg_europe_seas_trend_concat.polyfit_coefficients[:,0,...]*nanosec_to_year*convert_units).plot(cmap='RdBu', levels = 21,
                                                                                 col='seas', col_wrap=2,
                                                                                figsize=(8,4),
                                                                                subplot_kws={"projection": proj},
                                                                                cbar_kwargs={'label':'mm/seas/year', 'shrink':0.88,
                                                                                                    },
                                                                                vmin=-10, vmax=10)

draw_labels_l = {"left": "y"}
draw_labels_lb = {"bottom": "x", "left": "y"}
draw_labels_b = {"bottom": "x"}
draw_labels = [draw_labels_l, {}, draw_labels_lb, draw_labels_b]
for ln, ax in enumerate(f.axes.flat):
    ax.coastlines()
    ax.gridlines(draw_labels=draw_labels[ln])
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries

    ax.scatter(imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lon.values,
                imerg_europe['precipitation'].loc[{'lon': lon_slice, 'lat': lat_slice}].lat.values,
                transform=ccrs.PlateCarree(), color='black', marker='x')

plt.suptitle(start_date + ' to ' + end_date + ' trend in precipitation \n Seasonal resampled data', y=1.1)

#%% PLOT TSMP

# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))
pc = ccrs.PlateCarree()

''' EXAMPLE PLOT
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))
pc = ccrs.PlateCarree()

ax = plt.axes(projection=proj) # segun la proj que pongo acá es como queda en el mapa
tsmp.pr[240].plot(ax=ax, transform=rp)
ax.coastlines('50m', linewidth=0.8)
ax.gridlines(draw_labels={"bottom": "x", "left": "y"})
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries

# In order to reproduce the extent, we can't use cartopy's smarter
# "set_extent" method, as the bounding box is computed based on a transformed
# rectangle of given size. Instead, we want to emulate the "lower left corner"
# and "upper right corner" behaviour of basemap.
# xs, ys, zs = rp.transform_points(pc,
#                                   np.array([float(grace.lon[0]), float(grace.lat[0])]),
#                                   np.array([float(grace.lon[1]), float(grace.lat[1])])).T
# ax.set_xlim(xs)
# ax.set_ylim(ys)

lon_limits = [6,15] # [25, 45] [6,15]
lat_limits = [47, 55] # [35.5, 42.5] [47, 55]
ax.set_extent([lon_limits[0], lon_limits[-1], lat_limits[0], lat_limits[-1]], crs=proj)

plt.show()


'''

#%%
# For the yearly resampled data
convert_units = 365 # mm/h to mm/y

# first plot a single pixel to check the trend
lon_point = 11
lat_point = 50
lon_transf, lat_transf = rp.transform_point(lon_point, lat_point, pc) #transform the point from pc to rp
lon_slice = slice(lon_transf, lon_transf+0.11)
lat_slice = slice(lat_transf, lat_transf+0.11)

(tsmp_yearly.loc[{'rlon': lon_slice, 'rlat': lat_slice}]*convert_units).plot()
plt.plot(tsmp_yearly.time, 
         tsmp_yearly_trend.polyfit_coefficients.loc[{'rlon': lon_slice, 'rlat': lat_slice}][0,0,0]*convert_units*grace_yearly.time.astype(float) +
         tsmp_yearly_trend.polyfit_coefficients.loc[{'rlon': lon_slice, 'rlat': lat_slice}][1,0,0]*convert_units)

#%%
# Plot a map of trends (remember that the coefficient is in units per nanosecond)
nanosec_to_year = 1e9 * 60 * 60 * 24 * 365
convert_units = 365 # mm/h to mm/y

# load radar sites of Turkey
radar_range = 18000 # in points in the plot (18000 = 350 km survallience, 8500=250 km volume task)
turk_radars = pd.read_csv("turkish_radars.csv")
selected_radars = ["Hatay", "Gaziantep", "Trabzon", "Samsun", "Sivas", "Ankara"]
single_pol = ["Trabzon", "Samsun"]

f, ax1 = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection=pc))
plot = (tsmp_yearly_trend.polyfit_coefficients[0]*nanosec_to_year*convert_units).plot(ax=ax1, cmap='RdBu', levels = 21,
                                                                  cbar_kwargs={'label':'mm/year', 'shrink':0.88,
                                                                               },
                                                                  vmin=-20, vmax=20,
                                                                  transform = rp
                                                                  )
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries´
plt.title(start_date + ' to ' + end_date + ' trend in precipitation \n Yearly resampled data')
plt.scatter(tsmp_yearly.loc[{'rlon': lon_slice, 'rlat': lat_slice}].rlon.values,
            tsmp_yearly.loc[{'rlon': lon_slice, 'rlat': lat_slice}].rlat.values,
            transform=rp, color='black', marker='x')

# Plot turkish radar sites
plot_radar_sites = True
if plot_radar_sites:
    for sr in selected_radars: # plot selected radar ranges
        plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                    turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='black', marker='o', s= radar_range, alpha=0.3)
    for sr in selected_radars: # plot selected radar locations
        plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                    turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                    transform=pc, color='red', marker='x')
        name = sr
        if turk_radars[turk_radars["Name"]==sr]["Ins. Year"].values > 2015: # plot selected radar names
            name+=str(turk_radars[turk_radars["Name"]==sr]["Ins. Year"].values)
        if sr in single_pol:
            name+='(S)'
        plt.text(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                 turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                 name, c='Orange')
    for sr in turk_radars["Name"].to_list():
        if sr not in selected_radars:
            plt.scatter(turk_radars[turk_radars["Name"]==sr]["Longitude"].values,
                        turk_radars[turk_radars["Name"]==sr]["Latitude"].values,
                        transform=pc, color='blue', marker='x')
            


# set extent of map
region_to_plot = 'turkey'
lon_limits_dict = {
                    'germany': [6, 15],
                    'turkey': [25, 45],
                    'europe': [grace.lon[0], grace.lon[-1]]
                    }

lat_limits_dict = {
                    'germany': [47, 55],
                    'turkey': [34, 42.5],
                    'europe': [grace.lat[0], grace.lat[-1]]
                    }
        

lon_limits = lon_limits_dict[region_to_plot]
lat_limits = lat_limits_dict[region_to_plot]
ax1.set_extent([lon_limits[0], lon_limits[-1], lat_limits[0], lat_limits[-1]], crs=proj)