#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:20:30 2024

@author: jgiles

Script for preprocessing gridded datasets and perform accumulations and regridding.
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

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25) 
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

#%% Load datasets to process
# It is probably better to load and process one dataset at the time

ds_to_load = [
    # "IMERG-V07B-monthly",
    # "IMERG-V06B-monthly",
    # "IMERG-V07B-30min", # do not load this unless necessary, currently the calculations are done with cdo
    # "IMERG-V06B-30min", # do not load this unless necessary, currently the calculations are done with cdo
    # "CMORPH-daily",
    # "TSMP-old",
    # "TSMP-DETECT-Baseline",
    # "TSMP-Ben",
    # "ERA5-monthly",
    # "ERA5-hourly",
    "RADKLIM",
    # "RADOLAN",
    # "EURADCLIM",
    # "GPCC-monthly",
    # "GPCC-daily",
    # "E-OBS",
    # "CPC",
    # "GPROF",
    # "HYRAS", # do not load this unless necessary, currently the calculations are done with cdo
    # "GRACE-GDO",
    # "GRACE-GSFC",
    # "GRACE-ITSG",
    # "Springer-Rean",
    ]

# Choose only 1 variable at a time to process
var_to_load =[ # this are mock up variable names, not the actual variable names from the datasets
    "precipitation",
    # "tws" # terrestrial water storage
    ]

# check that only 1 variable was selected
if len(var_to_load) != 1:
    raise ValueError("Either no variable or more than one variable was selected in 'var_to_load'")


# Put everything on a single dictionary
data = dict()

#### GRACE GDO
# TWSA from Copernicus Global Drought Observatory (GDO), reference period 2002-2017
# https://data.jrc.ec.europa.eu/dataset/0fd62e28-241f-472c-8966-98744920e181#dataaccess
if "GRACE-GDO" in ds_to_load:
    print("Loading GRACE-GDO...")
    if "tws" in var_to_load:
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
if "GRACE-GSFC" in ds_to_load:
    print("Loading GRACE-GSFC...")
    if "tws" in var_to_load:
        data["GRACE-GSFC"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/gsfc.glb_.200204_202207_rl06v2.0_obp-ice6gd_halfdegree.nc')

#### GRACE ITSG (Uni Bonn)
# Ben: I recommend to use ITSG, it is the longest GRACE series from spherical harmonics that
# we currently have processed there (181 months over 2003-01/2020-09). Be warned that there
# are missing months in all versions of GRACE data, as the batteries started degrading after 2011,
# and mid-of-2016 until GRACE's end in 2017 is very noisy (many people do not even touch those). 
# GRACE-FO starts in 2018, some missing months in the beginning, but very robust since then. 
# No inter-sat calibration necessary, just treat all months as one data set.
if "GRACE-ITSG" in ds_to_load:
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

    if "tws" in var_to_load:
        data["GRACE-ITSG"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/ITSG_UB/ewh/*', combine="nested",
                                       preprocess= preprocess_grace_itsg, concat_dim = "time")
        
        data["GRACE-ITSG"].coords["time"] = datetimes


#### IMERG V06B (GLOBAL MONTHLY)
if "IMERG-V06B-monthly" in ds_to_load:
    print("Loading IMERG-V06B-monthly...")
    if "precipitation" in var_to_load:
        def preprocess_imerg(ds):
            # function to transform to accumulated monthly values
            days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
            ds["precipitation"] = ds["precipitation"]*days_in_month[ds.time.values[0].month-1]*24
            ds["precipitation"] = ds["precipitation"].assign_attrs(units="mm", Units="mm")
            return ds

        data["IMERG-V06B-monthly"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V06B/global_monthly/3B-MO.MS.MRG.3IMERG.*.V06B.HDF5.nc4', preprocess=preprocess_imerg)\
                        .transpose('time', 'lat', 'lon', ...) # * 24*30 # convert to mm/month (approx)

#### IMERG V07B (GLOBAL MONTHLY)
if "IMERG-V07B-monthly" in ds_to_load:
    print("Loading IMERG-V07B-monthly...")
    if "precipitation" in var_to_load:
        def preprocess_imerg(ds):
            # function to transform to accumulated monthly values
            days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
            ds["precipitation"] = ds["precipitation"]*days_in_month[ds.time.values[0].month-1]*24
            ds["precipitation"] = ds["precipitation"].assign_attrs(units="mm", Units="mm")
            return ds
    
        data["IMERG-V07B-monthly"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V07B/global_monthly/3B-MO.MS.MRG.3IMERG.*.V07B.HDF5.nc4', preprocess=preprocess_imerg)\
                        .transpose('time', 'lat', 'lon', ...) # * 24*30 # convert to mm/month (approx)

#### IMERG V06B (Europe, half-hourly)
if "IMERG-V06B-30min" in ds_to_load:
    print("Loading IMERG-V06B-30min...")
    if "precipitation" in var_to_load:
        def preprocess_imerg_europe(ds):
            for vv in ds.data_vars:
                try:
                    if "mm/hr" in ds[vv].units:
                        ds[vv] = ds[vv]*0.5 # values come in mm/h but the timestep is 30 min, so we have to divide by 2
                        ds[vv] = ds[vv].assign_attrs(units="mm", Units="mm")
                except AttributeError:
                    pass
            return ds
        data["IMERG-V06B-30min"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V06B/europe/concat_files/*.nc4', preprocess=preprocess_imerg_europe).transpose('time', 'lat', 'lon', ...)

#### IMERG V07B (Europe, half-hourly)
if "IMERG-V07B-30min" in ds_to_load:
    print("Loading IMERG-V07B-30min...")
    if "precipitation" in var_to_load:
        def preprocess_imerg_europe(ds):
            for vv in ds.data_vars:
                try:
                    if "mm/hr" in ds[vv].units:
                        ds[vv] = ds[vv]*0.5 # values come in mm/h but the timestep is 30 min, so we have to divide by 2
                        ds[vv] = ds[vv].assign_attrs(units="mm", Units="mm")
                except AttributeError:
                    pass
            return ds
        data["IMERG-V07B-30min"] = xr.open_mfdataset('/automount/agradar/jgiles/IMERG_V07B/europe/concat_files/*.nc4', preprocess=preprocess_imerg_europe).transpose('time', 'lat', 'lon', ...)

#### CMORPH (global daily)
if "CMORPH-daily" in ds_to_load:
    print("Loading CMORPH-daily...")
    if "precipitation" in var_to_load:
        data["CMORPH-daily"] = xr.open_mfdataset('/automount/agradar/jgiles/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/*/*/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc') #, preprocess=preprocess_imerg_europe).transpose('time', 'lat', 'lon', ...)
        data["CMORPH-daily"] = data["CMORPH-daily"].assign_coords(lon=(((data["CMORPH-daily"].lon + 180) % 360) - 180)).sortby('lon').assign_attrs(units="mm")
        data["CMORPH-daily"]["cmorph"] = data["CMORPH-daily"]["cmorph"].assign_attrs(units="mm")

#### ERA5 (GLOBAL MONTHLY)
if "ERA5-monthly" in ds_to_load:
    print("Loading ERA5-monthly...")
    ## volumetric soil water level
    if "tws" in var_to_load:
        data["ERA5-monthly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/monthly_averaged/single_level_vars/volumetric_soil*/volumetric_*')
        data["ERA5-monthly"] = data["ERA5-monthly"].assign_coords(longitude=(((data["ERA5-monthly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data["ERA5-monthly"] = data["ERA5-monthly"]['swvl1']*0.07 + data["ERA5-monthly"]['swvl2']*0.21 + \
                                    data["ERA5-monthly"]['swvl3']*0.72 + data["ERA5-monthly"]['swvl4']*1.89
    
    ## precipitation
    if "precipitation" in var_to_load:
        def preprocess_era5_totprec(ds):
            # function to transform to accumulated monthly values
            days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
            ds["tp"] = ds["tp"]*days_in_month[pd.Timestamp(ds.time.values[0]).month-1]*1000
            ds["tp"] = ds["tp"].assign_attrs(units="mm", Units="mm")
            return ds
        data["ERA5-monthly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/monthly_averaged/single_level_vars/total_precipitation/total_precipitation_*', 
                                         preprocess=preprocess_era5_totprec)
        data["ERA5-monthly"] = data["ERA5-monthly"].assign_coords(longitude=(((data["ERA5-monthly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data["ERA5-monthly"] = data["ERA5-monthly"].isel(latitude=slice(None, None, -1))

#### ERA5 (Europe hourly)
if "ERA5-hourly" in ds_to_load:
    print("Loading ERA5-hourly...")
    ## volumetric soil water level
    if "tws" in var_to_load:
        data["ERA5-hourly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/hourly/europe/single_level_vars/volumetric_soil*/volumetric_*')
        data["ERA5-hourly"] = data["ERA5-hourly"].assign_coords(longitude=(((data["ERA5-hourly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data["ERA5-hourly"] = data["ERA5-hourly"]['swvl1']*0.07 + data["ERA5-hourly"]['swvl2']*0.21 + \
                                    data["ERA5-hourly"]['swvl3']*0.72 + data["ERA5-hourly"]['swvl4']*1.89
    
    ## precipitation
    if "precipitation" in var_to_load:
        def preprocess_era5_totprec(ds):
            ds["tp"] = ds["tp"]*1000
            ds["tp"] = ds["tp"].assign_attrs(units="mm", Units="mm")
            return ds
        data["ERA5-hourly"] = xr.open_mfdataset('/automount/ags/jgiles/ERA5/hourly/europe/single_level_vars/total_precipitation/total_precipitation_*', 
                                         preprocess=preprocess_era5_totprec, chunks={"time":100})
        data["ERA5-hourly"] = data["ERA5-hourly"].assign_coords(longitude=(((data["ERA5-hourly"].longitude + 180) % 360) - 180)).sortby('longitude')
        data["ERA5-hourly"] = data["ERA5-hourly"].isel(latitude=slice(None, None, -1))
        data["ERA5-hourly"]["time"] = data["ERA5-hourly"].get_index("time").shift(-1, "h") # shift the values forward to the start of the interval

#### TSMP
# The timestamps of accumulated P are located at the center of the corresponding interval (1:30, 4:30, ...)
# Also, every monthly file has the last timestep from the previous month because of how the data is outputted by 
# the model, so some data are overlapped. The overlaped data are negligible different (around the 5th decimal)

def preprocess_tsmp(ds): # discard the first timestep of every monthly file (discard overlapping data)
    return ds.isel({"time":slice(1,None)})

if "TSMP-old" in ds_to_load:
    print("Loading TSMP-old...")
    ## precipitation
    if "precipitation" in var_to_load:    
        data["TSMP-old"] = xr.open_mfdataset('/automount/agradar/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*TOT_PREC*',
                                     preprocess=preprocess_tsmp, chunks={"time":1000})

        # data["TSMP-old"] = data["TSMP-old"].assign( xr.open_mfdataset('/automount/agradar/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*WT*',
        #                              chunks={"time":1000}) )
        data["TSMP-old"]["time"] = data["TSMP-old"].get_index("time").shift(-1.5, "h") # shift the values forward to the start of the interval


if "TSMP-DETECT-Baseline" in ds_to_load:
    print("Loading TSMP-DETECT-Baseline...")
    ## precipitation
    if "precipitation" in var_to_load:    
        data["TSMP-DETECT-Baseline"] = xr.open_mfdataset('/automount/agradar/jgiles/DETECT_sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1/*/cosmo/TOT_PREC_ts.nc',
                                     preprocess=preprocess_tsmp, chunks={"time":1000})

        data["TSMP-DETECT-Baseline"]["time"] = data["TSMP-DETECT-Baseline"].get_index("time").shift(-0.5, "h") # shift the values forward to the start of the interval


if "TSMP-Ben" in ds_to_load:
    print("Loading TSMP-Ben...")
    ## tws
    if "tws" in var_to_load:    
        data["TSMP-Ben"] = xr.open_mfdataset('/automount/ags/jgiles/4BenCharlotte/TSMP/twsa_tsmp_europe_era5_1990_2021.nc')

#### EURADCLIM
if "EURADCLIM" in ds_to_load:
    print("Loading EURADCLIM...")
    if "precipitation" in var_to_load:    
        data["EURADCLIM"] = xr.open_mfdataset("/automount/agradar/jgiles/EURADCLIM/concat_files/RAD_OPERA_HOURLY_RAINFALL_*.nc")
        data["EURADCLIM"] = data["EURADCLIM"].set_coords(("lon", "lat"))
        data["EURADCLIM"]["lon"] = data["EURADCLIM"]["lon"][0]
        data["EURADCLIM"]["lat"] = data["EURADCLIM"]["lat"][0]

#### RADKLIM
if "RADKLIM" in ds_to_load:
    print("Loading RADKLIM...")
    if "precipitation" in var_to_load:    
        data["RADKLIM"] = xr.open_mfdataset("/automount/ags/jgiles/RADKLIM/20*/*.nc")

#### RADOLAN
if "RADOLAN" in ds_to_load:
    print("Loading RADOLAN...")
    if "precipitation" in var_to_load:
        # load raw data
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     data["RADOLAN"] = wrl.io.open_radolan_mfdataset("/automount/ags/jgiles/RADOLAN/hourly/radolan/historical/bin/20*/RW*/raa01-rw_10000-*-dwd---bin*")
        #     data["RADOLAN"]["RW"].attrs["unit"] = "mm"
        
        # load concatenated netcdf-transformed data
        data["RADOLAN"] = xr.open_mfdataset("/automount/ags/jgiles/RADOLAN/hourly/radolan/historical/concat_files/20*/*.nc")
        data["RADOLAN"]["RW"].attrs["unit"] = "mm"

#### HYRAS
if "HYRAS" in ds_to_load:
    print("Loading HYRAS...")
    if "precipitation" in var_to_load:    
        data["HYRAS"] = xr.open_mfdataset("/automount/ags/jgiles/HYRAS-PRE-DE/daily/hyras_de/precipitation/pr_hyras_1_1931_2020_v5-0_de.nc")

#### TWS reanalysis (Anne Springer)
if "Springer-Rean" in ds_to_load:
    print("Loading Springer-Rean...")
    if "tws" in var_to_load:    
        data["Springer-Rean"] = xr.open_mfdataset("/automount/ags/jgiles/springer_reanalysis/clmoas_scenario0033_ensMean.TWS.updateMat.200301-201908_monthly.nc")
        data["Springer-Rean-grid"] = xr.open_mfdataset("/automount/ags/jgiles/springer_reanalysis/CORDEX0.11_grid_424x412.nc")

#### GPCC
if "GPCC-monthly" in ds_to_load:
    print("Loading GPCC-monthly...")
    if "precipitation" in var_to_load:    
        data["GPCC-monthly"] = xr.open_mfdataset("/automount/ags/jgiles/GPCC/full_data_monthly_v2022/025/*.nc")
        data["GPCC-monthly"] = data["GPCC-monthly"].isel(lat=slice(None, None, -1))
        data["GPCC-monthly"]["precip"] = data["GPCC-monthly"]["precip"].assign_attrs(units="mm", Units="mm")

if "GPCC-daily" in ds_to_load:
    print("Loading GPCC-daily...")
    if "precipitation" in var_to_load:    
        data["GPCC-daily"] = xr.open_mfdataset("/automount/ags/jgiles/GPCC/full_data_daily_v2022/10/*.nc")
        data["GPCC-daily"] = data["GPCC-daily"].isel(lat=slice(None, None, -1))
        data["GPCC-daily"]["precip"] = data["GPCC-daily"]["precip"].assign_attrs(units="mm", Units="mm")

#### E-OBS
if "E-OBS" in ds_to_load:
    print("Loading E-OBS...")
    if "precipitation" in var_to_load:    
        data["E-OBS"] = xr.open_mfdataset("/automount/ags/jgiles/E-OBS/RR/rr_ens_mean_0.25deg_reg_v29.0e.nc")

#### CPC
if "CPC" in ds_to_load:
    print("Loading CPC...")
    if "precipitation" in var_to_load:    
        data["CPC"] = xr.open_mfdataset("/automount/ags/jgiles/CPC_global_precip/precip.????.nc")
        data["CPC"] = data["CPC"].isel(lat=slice(None, None, -1))
        data["CPC"] = data["CPC"].assign_coords(lon=(((data["CPC"].lon + 180) % 360) - 180)).sortby('lon')

#### GPROF
if "GPROF" in ds_to_load:
    print("Loading GPROF...")
    if "precipitation" in var_to_load:    

        gprof_files = sorted(glob.glob("/automount/ags/jgiles/GPM_L3/GPM_3GPROFGPMGMI.07/20*/*HDF5"))
        
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
            
            return ds
        
        gprof_attrs= xr.open_mfdataset(gprof_files, engine="h5netcdf", phony_dims='sort', preprocess=extract_time)
        
        # We create another function for adding the time dim to the dataset
        def add_tdim(ds):
            return ds.expand_dims("time")
        data["GPROF"] = xr.open_mfdataset(gprof_files, engine="h5netcdf", group="Grid", 
                                          concat_dim="time", combine="nested", preprocess=add_tdim)
        
        data["GPROF"] = data["GPROF"].assign_coords({"time": gprof_attrs.time})
        
        # We need to transform the data from mm/h to mm/month
        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        days_in_month_ds = xr.DataArray([ days_in_month[mindex] for mindex in data["GPROF"].time.dt.month.values-1], dims=["time"], coords= data["GPROF"].time.coords)
        for vv in ["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation"]:    
            data["GPROF"][vv] = data["GPROF"][vv]*days_in_month_ds*24
            data["GPROF"][vv] = data["GPROF"][vv].assign_attrs(units="mm", Units="mm")

#%% DAILY SUM
# CAREFUL WITH THIS PART, DO NOT RUN THE WHOLE CELL AT ONCE IF HANDLING MORE
# THAN ONE DATASET BECAUSE IT WILL PROBABLY CRASH OR TAKE FOREVER
# Set a path to save and load the reduced datasets
savepath = "/automount/agradar/jgiles/gridded_data/"

if len(var_to_load) == 1:
    vname = var_to_load[0]
else:
    raise ValueError("Either no variable or more than one variable was selected in 'var_to_load'")

# accumulate to daily values
print("Calculating daily sums ...")
data_dailysum = {}
for dsname in ds_to_load:
    if dsname not in ["IMERG-V07B-30min", "IMERG-V06B-30min", "HYRAS", 
                      "EURADCLIM", "GPCC-monthly", "GPCC-daily", 'GPROF']:
        print("... "+dsname)
        with ProgressBar():
            data_dailysum[dsname] = data[dsname].resample({"time": "D"}).sum().compute()

# save the processed datasets
print("Saving file ...")
for dsname in ds_to_load:
    print("... "+dsname)
    savepath_dsname = savepath+"/daily/"+dsname
    if not os.path.exists(savepath_dsname):
        os.makedirs(savepath_dsname)
    if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "HYRAS", 
                  "EURADCLIM", "GPCC-monthly", "GPCC-daily", 'GPROF']:
        # special treatment for these datasets, otherwise it will crash
        if dsname in ["HYRAS", "GPCC-daily"]:
            warnings.warn(dsname+" is already daily!")
        if dsname in ["GPCC-monthly", 'GPROF', "IMERG-V07B-monthly", "IMERG-V06B-monthly", "ERA5-monthly"]:
            warnings.warn(dsname+" is monthly!")
        if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min"]:
            warnings.warn("Do not compute daily sums from "+dsname+". Use CDO.")
        if dsname in ["IMERG-V07B-30min"]:
            data[dsname]["MWobservationTime"] = data[dsname]["MWobservationTime"].astype(data[dsname]["time"].dtype)
            if not os.path.exists(savepath_dsname+"/temp"):
                    os.makedirs(savepath_dsname+"/temp")
            for yy in np.unique(data[dsname].time.dt.year):
                print("Saving daily "+dsname+" files for "+str(yy))
                data[dsname].loc[{"time":str(yy)}].resample({"time": "D"}).sum().to_netcdf(savepath_dsname+"/temp/"+dsname+"_"+vname+"_dailysum_"+str(yy)+".nc",
                                                 encoding=dict([(vv,{"zlib":True, "complevel":6}) for vv in data[dsname].data_vars if "time_" not in vv if "Time" not in vv]))
        if dsname in ["EURADCLIM"]:
            Cdo().daysum(input="-shifttime,-1hour -cat /automount/agradar/jgiles/EURADCLIM/concat_files/RAD_OPERA_HOURLY_RAINFALL_*.nc", 
                         output="/automount/agradar/jgiles/gridded_data/daily/EURADCLIM/EURADCLIM_precipitation_dailysum_2013-2020.nc", options="-z zip_6")
    else:
        sd = str(data_dailysum[dsname].time[0].values)[0:4]
        ed = str(data_dailysum[dsname].time[-1].values)[0:4]
        data_dailysum[dsname].to_netcdf(savepath_dsname+"/"+dsname+"_"+vname+"_dailysum_"+sd+"-"+ed+".nc",
                                         encoding=dict([(vv,{"zlib":True, "complevel":6}) for vv in data_dailysum[dsname].data_vars]))

#%% MONTHLY SUM
# CAREFUL WITH THIS PART, DO NOT RUN THE WHOLE CELL AT ONCE IF HANDLING MORE
# THAN ONE DATASET BECAUSE IT WILL PROBABLY CRASH OR TAKE FOREVER
# Set a path to save and load the reduced datasets
savepath = "/automount/agradar/jgiles/gridded_data/"

if len(var_to_load) == 1:
    vname = var_to_load[0]
else:
    raise ValueError("Either no variable or more than one variable was selected in 'var_to_load'")

# accumulate to monthly values
print("Calculating monthly sums ...")
data_monthlysum = {}
for dsname in ds_to_load:
    if dsname not in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                      "EURADCLIM", "GPCC-monthly", 'GPROF']:
        print("... "+dsname)
        data_monthlysum[dsname] = data[dsname].resample({"time": "MS"}).sum().compute()
    if dsname in ["GPROF"]:
        print("... "+dsname)
        # GPROF is already monthly but we process it anyway so it is easier to read later
        data_monthlysum[dsname] = data[dsname][["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation", "npixPrecipitation", "npixTotal"]].resample({"time": "MS"}).sum().compute()

# save the processed datasets
print("Saving file ...")
for dsname in ds_to_load:
    print("... "+dsname)
    savepath_dsname = savepath+"/monthly/"+dsname
    if not os.path.exists(savepath_dsname):
        os.makedirs(savepath_dsname)
    if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                  "EURADCLIM", "GPCC-monthly"]:
        # special treatment for these datasets, otherwise it will crash
        if dsname == "HYRAS":
            Cdo().monsum(input="/automount/ags/jgiles/HYRAS-PRE-DE/daily/hyras_de/precipitation/pr_hyras_1_1931_2020_v5-0_de.nc", 
                          output="/automount/agradar/jgiles/gridded_data/monthly/HYRAS/HYRAS_precipitation_monthlysum_1931-2020.nc", options="-z zip_6")
        if dsname in ["GPCC-monthly", "ERA5-monthly"]:
            warnings.warn(dsname+" is already monthly!")
        if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly"]:
            warnings.warn("Do not compute monthly sums from "+dsname+". Use the monthly version.")
        if dsname in ["EURADCLIM"]:
            Cdo().monsum(input="/automount/agradar/jgiles/gridded_data/daily/EURADCLIM/EURADCLIM_precipitation_dailysum_2013-2020.nc", 
                         output="/automount/agradar/jgiles/gridded_data/monthly/EURADCLIM/EURADCLIM_precipitation_monthlysum_2013-2020.nc", options="-z zip_6")
    else:
        sd = str(data_monthlysum[dsname].time[0].values)[0:4]
        ed = str(data_monthlysum[dsname].time[-1].values)[0:4]
        data_monthlysum[dsname].to_netcdf(savepath_dsname+"/"+dsname+"_"+vname+"_monthlysum_"+sd+"-"+ed+".nc",
                                         encoding=dict([(vv,{"zlib":True, "complevel":6}) for vv in data_monthlysum[dsname].data_vars]))

#%% SEASONAL SUM
# CAREFUL WITH THIS PART, DO NOT RUN THE WHOLE CELL AT ONCE IF HANDLING MORE
# THAN ONE DATASET BECAUSE IT WILL PROBABLY CRASH OR TAKE FOREVER
# Set a path to save and load the reduced datasets
savepath = "/automount/agradar/jgiles/gridded_data/"

if len(var_to_load) == 1:
    vname = var_to_load[0]
else:
    raise ValueError("Either no variable or more than one variable was selected in 'var_to_load'")

# accumulate to monthly values
print("Calculating seasonal sums ...")
data_seasonsum = {}
for dsname in ds_to_load:
    if dsname not in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                      "EURADCLIM", 'GPROF']:
        print("... "+dsname)
        #!!! Resampling with QE does not work but apparently it has been fixed in the latest xarray version. Check again later
        # data_seasonsum[dsname] = data_monthlysum[dsname].resample(time="QE-DEC").sum().compute()
        data_seasonsum[dsname] = data[dsname].resample(time="QS-DEC", skipna=False).sum()
        data_seasonsum[dsname]["time"] = data_seasonsum[dsname]["time"].get_index('time').shift(3, "M") # We place the seasonal value in the last month
        # condition for filtering out the incomplete periods at the edges
        if dsname in ["CMORPH-daily", "TSMP-old", "TSMP-DETECT-Baseline", "RADKLIM", "RADOLAN", "GPCC-daily", "E-OBS", "CPC"]:
            # this datasets come in daily or hourly resolution
            cond = xr.ones_like(data[dsname].time.resample(time="MS").mean().time, dtype=int).rolling(time=3).mean().dropna("time")
        else:
            # for the rest we assume monthly resolution
            cond = xr.ones_like(data[dsname].time, dtype=int).rolling(time=3).mean().dropna("time")
        cond["time"] = cond["time"].get_index('time').shift(1, "M") # We place the seasonal value by the end of the month
        data_seasonsum[dsname] = data_seasonsum[dsname].where(cond).compute()
    if dsname in ["GPROF"]:
        print("... "+dsname)
        #!!! Resampling with QE does not work but apparently it has been fixed in the latest xarray version. Check again later
        # data_seasonsum[dsname] = data_monthlysum[dsname].resample(time="QE-DEC").sum().compute()
        data_seasonsum[dsname] = data[dsname][["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation", "npixPrecipitation", "npixTotal"]].resample(time="QS-DEC", skipna=False).sum()
        data_seasonsum[dsname]["time"] = data_seasonsum[dsname]["time"].get_index('time').shift(3, "ME") # We place the seasonal value in the last month
        # condition for filtering out the incomplete periods at the edges
        cond = xr.ones_like(data[dsname].time, dtype=int).rolling(time=3).mean().dropna("time")
        cond["time"] = cond["time"].get_index('time').shift(3, "ME") # We place the seasonal value in the last month
        data_seasonsum[dsname] = data_seasonsum[dsname].where(cond).compute()

# save the processed datasets
print("Saving file ...")
for dsname in ds_to_load:
    print("... "+dsname)
    savepath_dsname = savepath+"/seasonal/"+dsname
    if not os.path.exists(savepath_dsname):
        os.makedirs(savepath_dsname)
    if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                  "EURADCLIM"]:
        # special treatment for these datasets, otherwise it will crash
        if dsname == "HYRAS":
            Cdo().seassum(input="/automount/ags/jgiles/HYRAS-PRE-DE/daily/hyras_de/precipitation/pr_hyras_1_1931_2020_v5-0_de.nc", 
                          output="/automount/agradar/jgiles/gridded_data/seasonal/HYRAS/HYRAS_precipitation_seasonalsum_1931-2020.nc", options="-z zip_6")
        if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly"]:
            warnings.warn("Do not compute seasonal sums from "+dsname+". Use the monthly version.")
        if dsname in ["EURADCLIM"]:
            Cdo().seassum(input="/automount/agradar/jgiles/gridded_data/daily/EURADCLIM/EURADCLIM_precipitation_dailysum_2013-2020.nc", 
                         output="/automount/agradar/jgiles/gridded_data/seasonal/EURADCLIM/EURADCLIM_precipitation_seasonalsum_2013-2020.nc", options="-z zip_6")
    else:
        sd = str(data_seasonsum[dsname].time[0].values)[0:4]
        ed = str(data_seasonsum[dsname].time[-1].values)[0:4]
        data_seasonsum[dsname].to_netcdf(savepath_dsname+"/"+dsname+"_"+vname+"_seasonalsum_"+sd+"-"+ed+".nc",
                                         encoding=dict([(vv,{"zlib":True, "complevel":6}) for vv in data_seasonsum[dsname].data_vars]))

#%% YEARLY SUM
# CAREFUL WITH THIS PART, DO NOT RUN THE WHOLE CELL AT ONCE IF HANDLING MORE
# THAN ONE DATASET BECAUSE IT WILL PROBABLY CRASH OR TAKE FOREVER
# Set a path to save and load the reduced datasets
savepath = "/automount/agradar/jgiles/gridded_data/"

if len(var_to_load) == 1:
    vname = var_to_load[0]
else:
    raise ValueError("Either no variable or more than one variable was selected in 'var_to_load'")

# accumulate to yearly values
print("Calculating yearly sums ...")
data_yearlysum = {}
for dsname in ds_to_load:
    if dsname not in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                      "EURADCLIM", "GPROF"]:
        print("... "+dsname)
        data_yearlysum[dsname] = data[dsname].resample({"time": "YS"}).sum().compute()
    if dsname in ["GPROF"]:
        print("... "+dsname)
        data_yearlysum[dsname] = data[dsname][["surfacePrecipitation", "convectivePrecipitation", "frozenPrecipitation", "npixPrecipitation", "npixTotal"]].resample({"time": "YS"}).sum().compute()

# save the processed datasets
print("Saving file ...")
for dsname in ds_to_load:
    print("... "+dsname)
    savepath_dsname = savepath+"/yearly/"+dsname
    if not os.path.exists(savepath_dsname):
        os.makedirs(savepath_dsname)
    if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly", "HYRAS", 
                  "EURADCLIM"]:
        # special treatment for these datasets, otherwise it will crash
        if dsname == "HYRAS":
            Cdo().yearsum(input="/automount/ags/jgiles/HYRAS-PRE-DE/daily/hyras_de/precipitation/pr_hyras_1_1931_2020_v5-0_de.nc", 
                          output="/automount/agradar/jgiles/gridded_data/yearly/HYRAS/HYRAS_precipitation_yearlysum_1931-2020.nc", options="-z zip_6")
        if dsname in ["IMERG-V07B-30min", "IMERG-V06B-30min", "ERA5-hourly"]:
            warnings.warn("Do not compute yearly sums from "+dsname+". Use the monthly version.")
        if dsname in ["EURADCLIM"]:
            Cdo().yearsum(input="/automount/agradar/jgiles/gridded_data/monthly/EURADCLIM/EURADCLIM_precipitation_monthlysum_2013-2020.nc", 
                         output="/automount/agradar/jgiles/gridded_data/yearly/EURADCLIM/EURADCLIM_precipitation_yearlysum_2013-2020.nc", options="-z zip_6")
    else:
        sd = str(data_yearlysum[dsname].time[0].values)[0:4]
        ed = str(data_yearlysum[dsname].time[-1].values)[0:4]
        data_yearlysum[dsname].to_netcdf(savepath_dsname+"/"+dsname+"_"+vname+"_yearlysum_"+sd+"-"+ed+".nc",
                                         encoding=dict([(vv,{"zlib":True, "complevel":6}) for vv in data_yearlysum[dsname].data_vars]))
