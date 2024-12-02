#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:36:23 2023

@author: jgiles

Plot PPIs, QVPs, line plots, etc
"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER?)

import datatree as dttree
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
warnings.filterwarnings('ignore')

# To be able to plot again in the Spyder panel, run this: %matplotlib inline

#%% Load and process data

#### Load the data into an xarray dataset (ds)

abs_zdr_off_min_thresh = 0. # if ZDR_OC has more negative values than the original ZDR
# and the absolute median offset is < abs_zdr_off_min_thresh, then undo the correction (set to 0 to avoid this step)
zdr_offset_perts = False # offset correct zdr per timesteps? if False, correct with daily offset
mix_zdr_offsets = False # if True and zdr_offset_perts=False, try to
# choose between daily LR-consistency and QVP offsets based on how_mix_zdr_offset.
# If True and zdr_offset_perts=True, choose between all available timestep offsets 
# based on how_mix_zdr_offset. If False, just use the offsets according to the priority they are passed on.
how_mix_zdr_offset = "count" # how to choose between the different offsets 
# if mix_zdr_offsets = True. "count" will choose the offset that has more data points
# in its calculation (there must be a variable ZDR_offset_datacount in the loaded offset).
# "neg_overcorr" will choose the offset that generates less negative ZDR values.

min_height_key = "default" # default = 200
min_range_key = "GZT" # default = 1000

ff = "/automount/realpep/upload/jgiles/dwd/*/*/2017-07-25/pro/vol5minng01/07/*allmoms*"
ff = "/automount/realpep/upload/jgiles/dmi/*/*/2019-07-17/ANK/*F/8.0/*allmoms*"
# ff = "/automount/realpep/upload/jgiles/dmi/*/*/2020-08-09/AFY/*/10.0/*allmoms*"
# ff = "/automount/realpep/upload/jgiles/dmi/*/*/2020-11-20/HTY/*/10.0/*allmoms*"
ff = "/automount/realpep/upload/jgiles/dmi/*/*/2017-05-20/GZT/*/10.0/*allmoms*.nc"
# ff = "/automount/realpep/upload/jgiles/dmi/*/*/2020-04-30/SVS/*/10.0/*allmoms*.nc"
# ff = "/automount/realpep/upload/jgiles/dmi/*/*/2018-10-21/SVS/*/7.0/*allmoms*.nc"
# ff = '/automount/realpep/upload/jgiles/dmi/2016/2016-08/2016-08-05/AFY/VOL_B/7.0/VOL_B-allmoms-7.0-20162016-082016-08-05-AFY-h5netcdf.nc'
# ff = "/automount/realpep/upload/jgiles/dwd/*/*/2018-06-02/pro/90gradstarng01/00/*allmoms*"
# ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2021/2021-10/2021-10-30/ess/90gradstarng01/00/*"
# ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2021/2021-07/2021-07-24/ess/90gradstarng01/00/*"
# ds = utils.load_dwd_preprocessed(ff)
# ds = utils.load_dwd_raw(ff)
ds = utils.load_dmi_preprocessed(ff)
# ds = utils.load_volume(sorted(glob.glob(ff)), func=utils.load_dmi_preprocessed)

# check if we are dealing with several elevations
isvolume = len(np.unique(ds.sweep_fixed_angle))>1
if isvolume: print("More than one elevation loaded, some processing steps may be skipped or not work")

if "dwd" in ff or "DWD" in ff:
    country="dwd"
    
    if "vol5minng01" in ff:
        clowres0=True # this is for the ML detection algorithm
    else:
        clowres0=False
        
    # if "umd" in ff: # this is now done automatically with the loading functions
    #     print("Flipping phase moments in UMD")
    #     for vf in ["UPHIDP", "KDP"]: # Phase moments in UMD are flipped into the negatives
    #         attrs = ds[vf].attrs.copy()
    #         ds[vf] = ds[vf]*-1
    #         ds[vf].attrs = attrs.copy()

elif "dmi" in ff:
    country="dmi"
    clowres0=False

#### Georeference 

ds = ds.pipe(wrl.georef.georeference) 

#### Define minimum height of usable data

min_height = utils.min_hgts[min_height_key] + ds["altitude"].values

#### Define minimum range of usable data (mostly for bad PHIDP filtering)

min_range = utils.min_rngs[min_range_key]

#### Get variable names

X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(ds)

#### Load noise corrected RHOHV

# Define the rhohv corrected paths and file names or take them from the default

rhoncdir = utils.rhoncdir
rhoncfile = utils.rhoncfile

if not isvolume:
    try:
        print("Loading noise corrected RHOHV")
        rhoncpath = os.path.dirname(utils.edit_str(ff, country, country+rhoncdir))
        
        ds = utils.load_corrected_RHOHV(ds, rhoncpath+"/"+rhoncfile)
              
        # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
        # if that is the case we take it that the correction did not work well so we won't use it
        std_margin = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_margin), otherwise use RHOHV
        min_rho = 0.6 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results
    
        if ( ds["RHOHV_NC"].where(ds["RHOHV_NC"]>min_rho * (ds["z"]>min_height)).std() < ds[X_RHO].where(ds[X_RHO]>min_rho * (ds["z"]>min_height)).std()*(1+std_margin) ).compute():
            # Change the default RHOHV name to the corrected one
            X_RHO = "RHOHV_NC"                    
    
    
    except OSError:
        print("No noise corrected RHOHV to load: "+rhoncpath+"/"+rhoncfile)
    
    except ValueError:
        print("ValueError with corrected RHOHV: "+rhoncpath+"/"+rhoncfile)
else:
    print("Skipped Loading noise corrected RHOHV")

#### Load ZDR offset

# We define a custom exception to stop the next nexted loops as soon as a file is loaded
class FileFound(Exception):
    pass

# Define the offset paths and file names or take them from the default

zdroffdir = utils.zdroffdir
zdrofffile = utils.zdrofffile

if zdr_offset_perts:
    zdrofffile = utils.zdrofffile_ts
    zdr_oc_dict = {}

# Load the offsets

if not isvolume:
    try:
        print("Loading ZDR offsets")
        for zdrod in zdroffdir:
            for zdrof in zdrofffile:
                try:
                    zdroffsetpath = os.path.dirname(utils.edit_str(ff, country, country+zdrod))
                    if "/VP/" in zdrod and "/vol5minng01/" in ff:
                        elevnr = ff.split("/vol5minng01/")[-1][0:2]
                        zdroffsetpath = utils.edit_str(zdroffsetpath, "/vol5minng01/"+elevnr, "/90gradstarng01/00")
                    
                    if zdr_offset_perts:
                        # if timestep-based offsets, we collect all of them and deal with them later
                        if zdrod not in zdr_oc_dict.keys():
                            zdr_oc_dict[zdrod] = []
                        zdr_oc_dict[zdrod].append(utils.load_ZDR_offset(ds, X_ZDR, zdroffsetpath+"/"+zdrof, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True))
                        continue
                    else:
                        ds = utils.load_ZDR_offset(ds, X_ZDR, zdroffsetpath+"/"+zdrof, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True)
                    
                    # if the offset comes from LR ZH-ZDR consistency, check it against
                    # the QVP method (if available) and choose the best one based on how_mix_zdr_offset 
                    if "LR_consistency" in zdrod and mix_zdr_offsets:
                        for zdrof2 in zdrofffile:
                            try:
                                zdrod2 = [pp for pp in zdroffdir if "QVP" in pp][0]
                                zdroffsetpath_qvp = os.path.dirname(utils.edit_str(ff, country, country+zdrod2))
                                ds_qvpoc = utils.load_ZDR_offset(ds, X_ZDR, zdroffsetpath_qvp+"/"+zdrof2, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True)
                                
                                if how_mix_zdr_offset == "neg_overcorr":
                                    # calculate the count of negative values after each correction
                                    neg_count_ds_lroc = (ds[X_ZDR+"_OC"].where((ds[X_RHO]>0.99) * (ds["z"]>min_height)) < 0).sum().compute()
                                    neg_count_ds_qvpoc = (ds_qvpoc[X_ZDR+"_OC"].where((ds_qvpoc[X_RHO]>0.99) * (ds["z"]>min_height)) < 0).sum().compute()
                                    
                                    if neg_count_ds_lroc > neg_count_ds_qvpoc:
                                        # continue with the correction with less negative values
                                        print("Changing daily ZDR offset from LR_consistency to QVP")
                                        ds = ds_qvpoc
                                
                                elif how_mix_zdr_offset == "count":
                                    if "ZDR_offset_datacount" in ds and "ZDR_offset_datacount" in ds_qvpoc:
                                        # Choose the offset that has the most data points in its calculation
                                        if ds_qvpoc["ZDR_offset_datacount"] > ds["ZDR_offset_datacount"]:
                                            # continue with the correction with more data points
                                            print("Changing daily ZDR offset from LR_consistency to QVP")
                                            ds = ds_qvpoc
                                    else:
                                        print("how_mix_zdr_offset == 'count' not possible, ZDR_offset_datacount not present in all offset datasets.")
    
                                break
                            except (OSError, ValueError):
                                pass
                    
                    # calculate the count of negative values before and after correction
                    neg_count_ds = (ds[X_ZDR].where((ds[X_RHO]>0.99) * (ds["z"]>min_height)) < 0).sum().compute()
                    neg_count_ds_oc = (ds[X_ZDR+"_OC"].where((ds[X_RHO]>0.99) * (ds["z"]>min_height)) < 0).sum().compute()
                    
                    if neg_count_ds_oc > neg_count_ds and abs((ds[X_ZDR] - ds[X_ZDR+"_OC"]).compute().median()) < abs_zdr_off_min_thresh:
                        # if the correction introduces more negative values and the offset is lower than abs_zdr_off_min_thresh, then do not correct
                        ds[X_ZDR+"_OC"] = ds[X_ZDR]
                    
                    # Change the default ZDR name to the corrected one
                    X_ZDR = X_ZDR+"_OC"
                    
                    # raise the custom exception to stop the loops
                    raise FileFound 
                    
                except (OSError, ValueError):
                    pass
        
        if zdr_offset_perts:
            # Clean zdr_oc_dict of empty entries
            zdr_oc_dict = {key: value for key, value in zdr_oc_dict.items() if value}
    
            # Deal with all the timestep-based offsets
            final_zdr_oc_list = []
            if len(zdr_oc_dict) == 0:
                # No ZDR timestep-based offsets were loaded, print a message
                print("No timestep-based zdr offsets to load")
            else:
                for zdrod in zdroffdir:
                    # For the offset from each method, we merge all variants (below ML, 
                    # below 1C, etc) to have as many values as possible. In the end we have
                    # a list of final xarray dataarrays for each entry of zdroffdir.
                    if zdrod in zdr_oc_dict.keys():
                        if len(zdr_oc_dict[zdrod]) == 0:
                            del(zdr_oc_dict[zdrod])
                        elif len(zdr_oc_dict[zdrod]) == 1:
                            # zdr_oc_dict[zdrod] = zdr_oc_dict[zdrod][0]
                            final_zdr_oc_list.append(zdr_oc_dict[zdrod][0])
                        else:
                            zdr_oc_aux = zdr_oc_dict[zdrod][0].copy()
                            for zdr_oc_auxn in zdr_oc_dict[zdrod][1:]:
                                zdr_oc_aux = zdr_oc_aux.where(zdr_oc_aux[X_ZDR+"_OC"].notnull(), zdr_oc_auxn).copy()
                            final_zdr_oc_list.append(zdr_oc_aux.copy())
                            
                # we get the first correction based on priority
                final_zdr_oc = final_zdr_oc_list[0].copy()
    
                # now we pick, for each timestep, the best offset correction depending on the priority, 
                # data quality and/or possible overcorrections            
                if len(final_zdr_oc_list) > 1 and mix_zdr_offsets:
                    print("Merging valid ZDR offsets (timestep mode)")
                    if how_mix_zdr_offset == "neg_overcorr":
                        for final_zdr_ocn in final_zdr_oc_list[1:]:
                            # calculate the count of negative values for each correction
                            neg_count_final_zdr_oc = (final_zdr_oc.where((ds[X_RHO]>0.99) * (final_zdr_oc["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                            neg_count_final_zdr_ocn = (final_zdr_ocn.where((ds[X_RHO]>0.99) * (final_zdr_ocn["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                            # Are there less negatives in the first correction than the new one?
                            neg_count_final_cond = neg_count_final_zdr_oc < neg_count_final_zdr_ocn
                            # Retain first ZDR_OC where the condition is True, otherwise use the new ZDR_OC
                            final_zdr_oc = final_zdr_oc.where(neg_count_final_cond, final_zdr_ocn).copy()
    
                    elif how_mix_zdr_offset == "count":
                        for final_zdr_ocn in final_zdr_oc_list[1:]:
                            if "ZDR_offset_datacount" in final_zdr_oc and "ZDR_offset_datacount" in final_zdr_ocn:
                                # Choose the offset that has the most data points in its calculation
                                final_zdr_oc = final_zdr_oc.where(final_zdr_oc["ZDR_offset_datacount"] > final_zdr_ocn["ZDR_offset_datacount"], final_zdr_ocn).copy()
                                
                    else:
                        print("how_mix_zdr_offset = "+how_mix_zdr_offset+" is not a valid option, no mixing of offsets was done")
    
                # Get only ZDR from now on
                final_zdr_oc = final_zdr_oc[X_ZDR+"_OC"]
                
                # Compare each timestep to the original ZDR, if the offsets overcorrect and the offset is < abs_zdr_off_min_thresh, discard correction
                neg_count_final_zdr_oc = (final_zdr_oc.where((ds[X_RHO]>0.99) * (final_zdr_oc["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                neg_count_final_zdr = (ds[X_ZDR].where((ds[X_RHO]>0.99) * (ds["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                neg_count_final_cond = (neg_count_final_zdr_oc > neg_count_final_zdr) * (abs((ds[X_ZDR] - final_zdr_oc).compute().median(("range", "azimuth"))) < abs_zdr_off_min_thresh)
                
                # Set the final ZDR_OC and change the default ZDR name to the corrected one 
                ds[X_ZDR+"_OC"] = final_zdr_oc.where(~neg_count_final_cond, ds[X_ZDR]).where(final_zdr_oc.notnull(), ds[X_ZDR])
                X_ZDR = X_ZDR+"_OC"            
                            
        else:
            # If no ZDR offset was loaded, print a message
            print("No zdr offset to load: "+zdroffsetpath+"/"+zdrof)
    except FileFound:
        pass
else:
    print("Skipped Loading ZDR offsets")

#### Phase processing

interpolation_method_ML = "linear" # for interpolating PHIDP in the ML

phase_proc_params = utils.get_phase_proc_params(ff) # get default phase processing parameters

# Check that PHIDP is in data, otherwise skip ML detection
if X_PHI in ds.data_vars:
    # Set parameters according to data
    
    # for param_name in phase_proc_params[country].keys():
    #     globals()[param_name] = phase_proc_params[param_name]    
    window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia = phase_proc_params.values() # explicit alternative

    # phidp may be already preprocessed (turkish case), then only offset-correct (no smoothing) and then vulpiani
    if "PHIDP" not in X_PHI: # This is now always skipped with this definition ("PHIDP" is in both X_PHI); i.e., we apply full processing to turkish data too
        # calculate phidp offset
        ds = utils.phidp_offset_correction(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                             rng_min=1000, rng=rng, azmedian=azmedian, tolerance=(0,5)) # shorter rng, rng_min for finer turkish data
    
        phi_masked = ds[X_PHI+"_OC"].where((ds[X_RHO] >= 0.9) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )   

    else:
        ds = utils.phidp_processing(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                             rng=rng, azmedian=azmedian, tolerance=(0,5), clean_invalid=False, fillna=False)
    
        phi_masked = ds[X_PHI+"_OC_SMOOTH"].where((ds[X_RHO] >= 0.9) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

    # Assign phi_masked
    assign = { X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs) }
    ds = ds.assign(assign)
    
    # derive KDP from PHIDP (Vulpiani)

    ds = utils.kdp_phidp_vulpiani(ds, winlen0, X_PHI+"_OC_MASKED", min_periods=winlen0//2+1)    
    
    X_PHI = X_PHI+"_OC" # continue using offset corrected PHI

else:
    print(X_PHI+" not found in the data, skipping ML detection")

#### Compute QVP or RD-QVP

if isvolume:
    print("Computing RD-QVP")
    ds_qvp = utils.compute_rdqvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10,"SNRHC":10, "SQIH":0.5},
                                 max_range=30000.)
else:
    print("Computing QVP")
    ds_qvp = utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10,"SNRHC":10, "SQIH":0.5} )

# filter out values close to the ground
ds_qvp = ds_qvp.where(ds_qvp["z"]>min_height)

#### Detect melting layer

if X_PHI in ds.data_vars:
    # Define thresholds
    moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}
    
    # Calculate ML
    ds_qvp = utils.melting_layer_qvp_X_new(ds_qvp, moments=moments, dim="z", fmlh=0.3, 
             xwin=xwin0, ywin=ywin0, min_h=min_height, rhohv_thresh_gia=rhohv_thresh_gia, all_data=True, clowres=clowres0)
    
    # Assign ML values to dataset
    
    ds = ds.assign_coords({'height_ml': ds_qvp.height_ml})
    ds = ds.assign_coords({'height_ml_bottom': ds_qvp.height_ml_bottom})

    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp.height_ml_new_gia})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp.height_ml_bottom_new_gia})

#### Attach ERA5 temperature profile
loc = utils.find_loc(utils.locs, ff)
ds_qvp = utils.attach_ERA5_TEMP(ds_qvp, path=loc.join(utils.era5_dir.split("loc")))
ds = utils.attach_ERA5_TEMP(ds, path=loc.join(utils.era5_dir.split("loc")))

#### Discard possible erroneous ML values
if "height_ml_new_gia" in ds_qvp:
    ## First, filter out ML heights that are too high (above selected isotherm)
    isotherm = -1 # isotherm for the upper limit of possible ML values
    z_isotherm = ds_qvp.TEMP.isel(z=((ds_qvp["TEMP"]-isotherm)**2).argmin("z").compute())["z"]
    
    ds_qvp.coords["height_ml_new_gia"] = ds_qvp["height_ml_new_gia"].where(ds_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
    ds_qvp.coords["height_ml_bottom_new_gia"] = ds_qvp["height_ml_bottom_new_gia"].where(ds_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
    
    # Then, check that ML top is over ML bottom
    cond_top_over_bottom = ds_qvp.coords["height_ml_new_gia"] > ds_qvp.coords["height_ml_bottom_new_gia"] 
    
    # Assign final values
    ds_qvp.coords["height_ml_new_gia"] = ds_qvp["height_ml_new_gia"].where(cond_top_over_bottom).compute()
    ds_qvp.coords["height_ml_bottom_new_gia"] = ds_qvp["height_ml_bottom_new_gia"].where(cond_top_over_bottom).compute()
    
    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp.height_ml_new_gia.where(cond_top_over_bottom)})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp.height_ml_bottom_new_gia.where(cond_top_over_bottom)})

if not isvolume:
    #### Attenuation correction (NOT PROVED THAT IT WORKS NICELY ABOVE THE ML)
    if X_PHI in ds.data_vars:    
        ds = utils.attenuation_corr_linear(ds, alpha = 0.08, beta = 0.02, alphaml = 0.16, betaml = 0.022,
                                           dbzh="DBZH", zdr=["ZDR_OC", "ZDR"], phidp=["UPHIDP_OC", "PHIDP_OC"],
                                           ML_bot = "height_ml_bottom_new_gia", ML_top = "height_ml_new_gia",
                                           temp = "TEMP", temp_mlbot = 3, temp_mltop = 0, z_mlbot = 2000, dz_ml = 500,
                                           interpolate_deltabump = True )
            
        ds_qvp = ds_qvp.assign( utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})[[vv for vv in ds if "_AC" in vv]] )
        
    #### Fix KDP in the ML using PHIDP:
    if X_PHI in ds.data_vars:    
        ds = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen=winlen0, min_periods=winlen0//2+1)
    
        ds_qvp = ds_qvp.assign({"KDP_ML_corrected": utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})["KDP_ML_corrected"]})
            
    #### Classification of stratiform events based on entropy
    if X_PHI in ds.data_vars:    
        
        # calculate linear values for ZH and ZDR
        ds = ds.assign({"DBZH_lin": wrl.trafo.idecibel(ds[X_DBZH]), "ZDR_lin": wrl.trafo.idecibel(ds[X_ZDR]) })
        
        # calculate entropy
        Entropy = utils.Entropy_timesteps_over_azimuth_different_vars_schneller(ds.where(ds[X_DBZH]>0), zhlin="DBZH_lin", zdrlin="ZDR_lin", rhohvnc=X_RHO, kdp="KDP_ML_corrected")
        
        # concate entropy for all variables and get the minimum value 
        strati = xr.concat((Entropy.entropy_zdrlin, Entropy.entropy_Z, Entropy.entropy_RHOHV, Entropy.entropy_KDP),"entropy")        
        min_trst_strati = strati.min("entropy")
        
        # assign to datasets
        ds["min_entropy"] = min_trst_strati
        
        min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
        min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        ds_qvp = ds_qvp.assign({"min_entropy": min_trst_strati_qvp})
else:
    print("Ignored: Atten corr, fix KDP in the ML and entropy calculation")

#%% Test: load original ZDR offsets from files
import re

def create_zdrcal_dataset(loc, zdrcal_path="/automount/realpep/upload/jgiles/dmi/zdrcal"):
    zdr_offsets = []
    timestamps = []
    
    file_pattern = re.compile(loc+r'(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2}).zdrcal_results')
    value_pattern = re.compile(r'results\.fNewZdrOffsetEstimate_dB = ([-+]?[0-9]*\.?[0-9]+)')
    
    for filename in sorted(os.listdir(zdrcal_path)):
        match = file_pattern.match(filename)
        if match:
            # Extract timestamp from filename
            year, month, day, hour, minute, second = match.groups()
            timestamp = pd.Timestamp(f'20{year}-{month}-{day} {hour}:{minute}:{second}')

            # Read the file content
            with open(os.path.join(zdrcal_path, filename), 'r') as file:
                content = file.read()
                
            # Extract the fNewZdrOffsetEstimate_dB value
            value_match = value_pattern.search(content)
            if value_match:
                zdr_offset = float(value_match.group(1))
                timestamps.append(timestamp)
                zdr_offsets.append(zdr_offset)        
    
    # Create a DataArray with the extracted data
    zdr_offset_da = xr.DataArray(zdr_offsets, coords=[timestamps], dims=['time'])    
    
    # Create a Dataset
    zdr_offset_ds = xr.Dataset({'fNewZdrOffsetEstimate_dB': zdr_offset_da})

    return zdr_offset_ds

zdrcal = create_zdrcal_dataset("HTY")



#%% Small test (ZDR OC after atten correction)
# LR consistency correction after atten corr
zdr_offset_lr_belowML_step = utils.zhzdr_lr_consistency(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, min_h=min_height, timemode="step")
zdr_offset_lr_below1c_step = utils.zhzdr_lr_consistency(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, mlbottom=1, min_h=min_height, timemode="step")

zdr_offset_lr_combined_step = zdr_offset_lr_belowML_step.where(zdr_offset_lr_belowML_step.notnull(), zdr_offset_lr_below1c_step)

ds_qvp["ZDR_AC_OC"] = ds_qvp["ZDR_AC"] - zdr_offset_lr_combined_step

# QVP-like consistency correction after atten corr
minbins = int(1000 / ds["range"].diff("range").median())
zdr_offset_qvp_belowML_step = utils.zdr_offset_detection_qvps(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, azmed=False,
                                            min_h=min_height, timemode="step", minbins=minbins).compute()
zdr_offset_qvp_below1c_step = utils.zdr_offset_detection_qvps(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, mlbottom=1, azmed=False,
                                            min_h=min_height, timemode="step", minbins=minbins).compute()

zdr_offset_qvp_combined_step = zdr_offset_qvp_belowML_step.where(zdr_offset_qvp_belowML_step.notnull(), zdr_offset_qvp_below1c_step)

ds_qvp["ZDR_AC_OCqvp"] = ds_qvp["ZDR_AC"] - zdr_offset_qvp_combined_step["ZDR_offset"]

# QVP consistency correction after atten corr
minbins = int(1000 / ds["range"].diff("range").median())
zdr_offset_qvp_belowML_step2 = utils.zdr_offset_detection_qvps(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, azmed=True,
                                            min_h=min_height, timemode="step", minbins=minbins).compute()
zdr_offset_qvp_below1c_step2 = utils.zdr_offset_detection_qvps(ds, zdr="ZDR_AC", dbzh="DBZH_AC", rhohv=X_RHO, mlbottom=1, azmed=True,
                                            min_h=min_height, timemode="step", minbins=minbins).compute()

zdr_offset_qvp_combined_step2 = zdr_offset_qvp_belowML_step2.where(zdr_offset_qvp_belowML_step2.notnull(), zdr_offset_qvp_below1c_step2)

ds_qvp["ZDR_AC_OCqvp2"] = ds_qvp["ZDR_AC"] - zdr_offset_qvp_combined_step2["ZDR_offset"]

# Load the original offsets for plotting
zdroffsetpath = os.path.dirname(utils.edit_str(ff, country, country+"/calibration/zdr/LR_consistency/"))
zdrof = '*zdr_offset_belowML_timesteps-*'
zdr_offset_lr_belowML_step0 = xr.open_mfdataset(zdroffsetpath+"/"+zdrof)["ZDR_offset"]

zdrof = '*zdr_offset_below1C_timesteps-*'
zdr_offset_lr_below1c_step0 = xr.open_mfdataset(zdroffsetpath+"/"+zdrof)["ZDR_offset"]

zdr_offset_lr_combined_step0 = zdr_offset_lr_belowML_step0.where(zdr_offset_lr_belowML_step0.notnull(), zdr_offset_lr_below1c_step0)

zdroffsetpath = os.path.dirname(utils.edit_str(ff, country, country+'/calibration/zdr/QVP/'))
zdrof = '*zdr_offset_belowML_timesteps-*'
zdr_offset_qvp_belowML_step0 = xr.open_mfdataset(zdroffsetpath+"/"+zdrof)["ZDR_offset"]

zdrof = '*zdr_offset_below1C_timesteps-*'
zdr_offset_qvp_below1c_step0 = xr.open_mfdataset(zdroffsetpath+"/"+zdrof)["ZDR_offset"]

zdr_offset_qvp_combined_step0 = zdr_offset_qvp_belowML_step0.where(zdr_offset_qvp_belowML_step0.notnull(), zdr_offset_qvp_below1c_step0)


# Plot the offsets (does not look very good)
zdr_offset_lr_combined_step0.plot(label="LR consistency before atten corr", c="red")
zdr_offset_qvp_combined_step0.plot(label="QVP-like method before atten corr", c="orangered")
zdr_offset_lr_combined_step.plot(label="LR consistency after atten corr", c="navy", ls="--")
zdr_offset_qvp_combined_step["ZDR_offset"].plot(label="QVP-like method after atten corr", c="royalblue", ls="--")
zdr_offset_qvp_combined_step2["ZDR_offset"].plot(label="QVP method after atten corr", c="cornflowerblue", ls="--")
plt.legend()
plt.title("ZDR offset values")
plt.grid()


# Plot the differences of the offsets
(zdr_offset_lr_combined_step0-zdr_offset_lr_combined_step).plot(label="LR consistency before-after atten corr")
(zdr_offset_qvp_combined_step0-zdr_offset_qvp_combined_step["ZDR_offset"]).plot(label="QVP before-after atten corr")
plt.legend()
plt.title("ZDR offset differences before-after atten corr")
plt.grid()
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis

#%% Plot simple PPI 

tsel = "2020-03-13T10:00"
elevn = 6 # elevation index

if isvolume: # if more than one elevation, we need to select the one we want
    if tsel == "":
        datasel = ds.isel({"sweep_fixed_angle":elevn})
    else:
        datasel = ds.isel({"sweep_fixed_angle":elevn}).sel({"time": tsel}, method="nearest")
else:
    if tsel == "":
        datasel = ds
    else:
        datasel = ds.sel({"time": tsel}, method="nearest")
    
datasel = datasel.pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "KDP_ML_corrected"
xylims = 40000 # xlim and ylim (from -xylims to xylims)

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("RdBu_r")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
if mom!="VRADH": cmap = "miub2"

plot_over_map = False
plot_ML = True

crs=ccrs.Mercator(central_longitude=float(datasel["longitude"]))

if not plot_over_map:
    # plot simple PPI
    datasel[mom].wrl.vis.plot( cmap=cmap, norm=norm, xlim=(-xylims,xylims), ylim=(-xylims,xylims))
    # datasel[mom].wrl.vis.plot( cmap="Blues", vmin=0, vmax=6, xlim=(-xylims,xylims), ylim=(-xylims,xylims))
elif plot_over_map:
    # plot PPI with map coordinates
    fig = plt.figure(figsize=(10, 10))
    datasel[mom].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))
    ax = plt.gca()
    ax.add_feature(cartopy.feature.COASTLINE, linestyle='-', linewidth=1, alpha=0.4)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4)
    ax.gridlines(draw_labels=True)

if plot_ML:
    cax = plt.gca()
    datasel["z"].wrl.vis.plot(ax=cax,
                          levels=[datasel["height_ml_bottom_new_gia"], datasel["height_ml_new_gia"]], 
                          cmap="black",
                          func="contour")
    # datasel["z"].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))

if isvolume: elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).sweep_fixed_angle.values, 2))+"°"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values).split(".")[0])

#%% Plot PPI as lines 

tsel = "2020-03-13T10:00"
if tsel == "":
    datasel = ds
else:
    datasel = ds.sel({"time": tsel}, method="nearest")


mom = "PHIDP_OC_MASKED"

cmap0 = mpl.colormaps.get_cmap("Blues")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(datasel["azimuth"]))), N=len(datasel["azimuth"])+1)

from cycler import cycler
rc_cycle = {"axes.prop_cycle": cycler(color=cmap.colors)}
with plt.rc_context(rc_cycle):
    datasel[mom].plot.line(x="range", hue="azimuth", add_legend=False, xlim=(1500, 8000), ylim=(-10, 10))

# add also qvp as line
ds_qvp[mom].sel({"time": tsel}, method="nearest").plot.line(x="range", add_legend=False, c="red",
                                                            xlim=(1500, 30000), ylim=(-5,40))

plt.title(mom)
plt.show()
plt.close()

#%% Plot simple QVP 

max_height = 12000 # max height for the qvp plots

tsel = ""# slice("2017-08-31T19","2017-08-31T22")
if tsel == "":
    datasel = ds_qvp.loc[{"z": slice(0, max_height)}]
else:
    datasel = ds_qvp.loc[{"time": tsel, "z": slice(0, max_height)}]
    
# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "KDP_ML_corrected"

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
datasel[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3))
datasel["min_entropy"].dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[0.8, 1], hatches=["", "XXX", ""], colors=[(1,1,1,0)], add_colorbar=False, extend="both")
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
datasel["height_ml_new_gia"].plot(c="black")
datasel["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

# # Plot reflectivity as lines to check wet radome effect
# ax2 = plt.gca().twinx()
# ds.DBZH.mean("azimuth")[:,4:7].mean("range").plot(ax=ax2, c="dodgerblue")
# ax2.yaxis.label.set_color("dodgerblue")
# ax2.spines['left'].set_position(('outward', 60))
# ax2.tick_params(axis='y', labelcolor="dodgerblue")  # Add padding to the ticks
# ax2.yaxis.labelpad = -400  # Add padding to the y-axis label
# ax2.set_title("")

# # Plot zdrcal values
# ax3 = plt.gca().twinx()
# zdrcal.loc[{"time":slice(str(datasel.time[0].values), str(datasel.time[-1].values))}].fNewZdrOffsetEstimate_dB.plot(ax=ax3, c="magenta")
# ax3.yaxis.label.set_color("magenta")
# ax3.spines['right'].set_position(('outward', 90))
# ax3.tick_params(axis='y', labelcolor="magenta")  # Add padding to the ticks
# # ax3.yaxis.labelpad = 10  # Add padding to the y-axis label
# ax3.set_title("")

if isvolume: elevtitle = " RD-QVP"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values[0]).split(".")[0])
plt.show()
plt.close()

#%% TEST: Try different methods for stratiform classification
# lets try only ZH first
var_list = ["DBZH_lin", "DBZH"]

# Tobi's entropy (pseudo entropy)
entropy_zh_tobi = utils.calculate_pseudo_entropy(ds, var_names=var_list)

entropy_zh_tobi_over0 = utils.calculate_pseudo_entropy(ds.where(ds.DBZH>0), var_names=var_list)

# Binned normalized entropy (default 50 bins based on available data)
entropy_zh_50binned_norm = utils.calculate_binned_normalized_entropy(ds, var_names=var_list)

entropy_zh_50binned_norm_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list)

# Binned normalized entropy (default 50 bins based on available data) and empty bins deleted
entropy_zh_50binned_norm_clean = utils.calculate_binned_normalized_entropy(ds, var_names=var_list, remove_empty_bins=True)

entropy_zh_50binned_norm_clean_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, remove_empty_bins=True)

# Binned normalized entropy with "auto" bins 
entropy_zh_autobinned_norm = utils.calculate_binned_normalized_entropy(ds, var_names=var_list, bins="auto")

entropy_zh_autobinned_norm_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, bins="auto")

# Binned normalized entropy with "auto" bins and empty bins deleted
entropy_zh_autobinned_norm_clean = utils.calculate_binned_normalized_entropy(ds, var_names=var_list, bins="auto", remove_empty_bins=True)

entropy_zh_autobinned_norm_clean_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, bins="auto", remove_empty_bins=True)

# Binned normalized entropy with custom bins (between 0 and 80 dbzh)
custom_bins = np.arange(0,81,1)
entropy_zh_custombinned_norm = utils.calculate_binned_normalized_entropy(ds, var_names=var_list, bins=custom_bins)

entropy_zh_custombinned_norm_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, bins=custom_bins)

# Binned normalized entropy with custom bins (between 0 and 80 dbzh) and empty bins deleted
custom_bins = np.arange(0,81,1)
entropy_zh_custombinned_norm_clean = utils.calculate_binned_normalized_entropy(ds, var_names=var_list, bins=custom_bins, remove_empty_bins=True)

entropy_zh_custombinned_norm_clean_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, bins=custom_bins, remove_empty_bins=True)

custom_bins = np.arange(0,100000,100)
entropy_zh_custombinned_lin_norm_clean_over0 = utils.calculate_binned_normalized_entropy(ds.where(ds.DBZH>0), var_names=var_list, bins=custom_bins, remove_empty_bins=True)

# Normalized STD
std_zh = utils.calculate_std(ds, var_names=var_list)

std_zh_over0 = utils.calculate_std(ds.where(ds.DBZH>0), var_names=var_list)

std_zh_over0_below80 = utils.calculate_std(ds.where(ds.DBZH>0).where(ds.DBZH<80), var_names=var_list)

std_zh_over0_below80_propernorm = utils.calculate_std(ds.where(ds.DBZH>0).where(ds.DBZH<80), var_names=var_list, normlims=(0,80))

# Plots
def plot_qvp_strattest(stratres, vmin=0, vmax=1, ylim=(0,60000), title="", contourlevs=[0.8]):
    stratres.plot(x="time", vmin=vmin, vmax=vmax, ylim=ylim)
    stratres.plot.contour(levels=contourlevs, x="time", colors=["gray"])
    plt.title(title)
    return None

def plot_ppi_strattest(ppi, stratres, vmin=0, vmax=50, xlim=(-50000,50000), ylim=(-50000,50000), title=""):
    ppi.wrl.vis.plot(vmin=vmin, vmax=vmax, ylim=ylim, xlim=xlim)
    (ppi*0+stratres).plot(x="x", y="y", cmap="Grays", alpha=0.1, add_colorbar=False)
    plt.title(title)
    # stratres.broadcast_like(ppi).wrl.vis.plot(vmin=vmin, vmax=vmax, ylim=ylim, xlim=xlim, color="lightgray")
    return None
    
# plot_ppi_strattest(ds.DBZH.sel(time="2017-05-20T16", method="nearest"), entropy_zh_tobi.entropy_DBZH.sel(time="2017-05-20T16", method="nearest")>0.8)

#%% TEST: fix wet radome atten (following Fig 7 of https://doi.org/10.1002/qj.3366)
def zhcorr(ds, dbzh="DBZH", irange=2):
    # correction according to the fitting curve of Fig 5. https://doi.org/10.1175/JTECH-D-12-00148.1 (plotted in Fig 7 of https://doi.org/10.1002/qj.3366)
    Zm = ds[dbzh].mean("azimuth").isel(range=irange)
    Zm = Zm.where(Zm>=0)
    return 2.51 + 0.352*Zm + 0.00189*Zm**2

def zdrcorr(ds, dbzh="DBZH", zdr="ZDR", irange=2):
    # correction extracted by eye from Fig 7 of https://doi.org/10.1002/qj.3366
    Zm = ds[dbzh].mean("azimuth").isel(range=irange)
    Zm = Zm.where(Zm>=0)
    return -0.043 + 0.03018*Zm - 0.0021176*Zm**2 + 0.000056464*Zm**3

zh_offset = zhcorr(ds)
zdr_offset = zdrcorr(ds).rename("ZDR")

# Plot the values
ds["DBZH"].mean("azimuth").isel(range=2).plot(label="DBZH close to the radar")
zh_offset.plot(label="DBZH correction", ylim=(0,30));
ax2 = plt.gca().twinx()
zdr_offset.plot(label="DBZH close to the radar", ax=ax2, ylim=(0,1)); # irrelevant plot just to get the label
zdr_offset.plot(label="DBZH correction", ax=ax2, ylim=(0,1)); # irrelevant plot just to get the label
zdr_offset.plot(label="ZDR correction", ax=ax2, c="green", ylim=(0,1));
plt.legend()

#%% TEST: apply the previous fix (Plot simple QVP)
max_height = 12000 # max height for the qvp plots

tsel = ""# slice("2017-08-31T19","2017-08-31T22")
if tsel == "":
    datasel = ds_qvp.loc[{"z": slice(0, max_height)}]
else:
    datasel = ds_qvp.loc[{"time": tsel, "z": slice(0, max_height)}]

# Apply the previous correction
datasel  = datasel.assign({"ZDR_WRcorr": datasel["ZDR"]-zdr_offset.fillna(0)})
datasel = datasel.assign({"DBZH_WRcorr": datasel["DBZH"]+zh_offset.fillna(0)})

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "ZDR_WRcorr"

ticks = radarmet.visdict14["ZDR"]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
datasel[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3))
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
datasel["height_ml_new_gia"].plot(c="black")
datasel["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

# # Plot reflectivity as lines to check wet radome effect
# ax2 = plt.gca().twinx()
# ds.DBZH.mean("azimuth")[:,2].plot(ax=ax2, c="dodgerblue")
# ax2.yaxis.label.set_color("dodgerblue")
# ax2.spines['left'].set_position(('outward', 60))
# ax2.tick_params(axis='y', labelcolor="dodgerblue")  # Add padding to the ticks
# ax2.yaxis.labelpad = -400  # Add padding to the y-axis label
# ax2.set_title("")

# # Plot zdrcal values
# ax3 = plt.gca().twinx()
# zdrcal.loc[{"time":slice(str(datasel.time[0].values), str(datasel.time[-1].values))}].fNewZdrOffsetEstimate_dB.plot(ax=ax3, c="magenta")
# ax3.yaxis.label.set_color("magenta")
# ax3.spines['right'].set_position(('outward', 90))
# ax3.tick_params(axis='y', labelcolor="magenta")  # Add padding to the ticks
# # ax3.yaxis.labelpad = 10  # Add padding to the y-axis label
# ax3.set_title("")

if isvolume: elevtitle = " RD-QVP"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values[0]).split(".")[0])
plt.show()
plt.close()

#%% TEST: Test the ZDR LR offset correction again with the wet radome corrected vars
# Apply the previous correction
ds_corr  = ds.assign({"ZDR_WRcorr": ds["ZDR"]-zdr_offset.fillna(0)})
ds_corr = ds_corr.assign({"DBZH_WRcorr": ds["DBZH"]+zh_offset.fillna(0)})

zdr_offset_lr = utils.zhzdr_lr_consistency(ds_corr, zdr="ZDR", dbzh="DBZH", rhohv="RHOHV", rhvmin=0.99, min_h=200, band="C",
                         mlbottom=None, temp=None, timemode="step", plot_correction=True, plot_timestep=70)

zdr_offset_lr_wr = utils.zhzdr_lr_consistency(ds_corr, zdr="ZDR_WRcorr", dbzh="DBZH_WRcorr", rhohv="RHOHV", rhvmin=0.99, min_h=200, band="C",
                         mlbottom=None, temp=None, timemode="step", plot_correction=True, plot_timestep=70)

#%% TEST: apply the ZDR offsets with and without WR correction
max_height = 12000 # max height for the qvp plots

tsel = ""# slice("2017-08-31T19","2017-08-31T22")
if tsel == "":
    datasel = ds_qvp.loc[{"z": slice(0, max_height)}]
else:
    datasel = ds_qvp.loc[{"time": tsel, "z": slice(0, max_height)}]

# Apply the previous corrections
datasel  = datasel.assign({"ZDR_WRcorr": datasel["ZDR"]-zdr_offset.fillna(0)})
datasel = datasel.assign({"DBZH_WRcorr": datasel["DBZH"]+zh_offset.fillna(0)})

datasel  = datasel.assign({"ZDR_WRcorr_OC": datasel["ZDR_WRcorr"]-zdr_offset_lr_wr["ZDR_offset"].fillna(0)})

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "ZDR_WRcorr_OC"

ticks = radarmet.visdict14["ZDR"]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
datasel[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3))
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
datasel["height_ml_new_gia"].plot(c="black")
datasel["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

# # Plot reflectivity as lines to check wet radome effect
# ax2 = plt.gca().twinx()
# ds.DBZH.mean("azimuth")[:,2].plot(ax=ax2, c="dodgerblue")
# ax2.yaxis.label.set_color("dodgerblue")
# ax2.spines['left'].set_position(('outward', 60))
# ax2.tick_params(axis='y', labelcolor="dodgerblue")  # Add padding to the ticks
# ax2.yaxis.labelpad = -400  # Add padding to the y-axis label
# ax2.set_title("")

# # Plot zdrcal values
# ax3 = plt.gca().twinx()
# zdrcal.loc[{"time":slice(str(datasel.time[0].values), str(datasel.time[-1].values))}].fNewZdrOffsetEstimate_dB.plot(ax=ax3, c="magenta")
# ax3.yaxis.label.set_color("magenta")
# ax3.spines['right'].set_position(('outward', 90))
# ax3.tick_params(axis='y', labelcolor="magenta")  # Add padding to the ticks
# # ax3.yaxis.labelpad = 10  # Add padding to the y-axis label
# ax3.set_title("")

if isvolume: elevtitle = " RD-QVP"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values[0]).split(".")[0])
plt.show()
plt.close()


#%% Plot QVP as lines 

tsel = slice("2018-03-28T17:00","2018-03-28T18:00")
if tsel == "":
    datasel = ds_qvp
else:
    datasel = ds_qvp.sel({"time": tsel})


mom = "PHIDP_OC"

cmap0 = mpl.colormaps.get_cmap("Blues")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(datasel.time))), N=len(datasel.time)+1)

from cycler import cycler
rc_cycle = {"axes.prop_cycle": cycler(color=cmap.colors)}
with plt.rc_context(rc_cycle):
    datasel[mom].plot.line(x="range", hue="time", add_legend=False, xlim=(1000, 10000), ylim=(-10,30))

plt.title(mom)
plt.show()
plt.close()

#%% Plot fake RHI (cross section PPI) #!!!! NOT FINISHED

tsel = "2015-10-06T16:30"
azimuth = 0. # azimuth to plot

# Define projection
epsg_code=3857
# Create projection objects for georeferencing 
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(epsg_code)

# sel time
if tsel == "":
    datasel = ds
else:
    datasel = ds.sel({"time": tsel}, method="nearest")

datasel = datasel.pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "DBZH"
xylims = 50000 # xlim and ylim (from -xylims to xylims)

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"

plot_ML = True

if isvolume: # if more than one elevation, we need to select the one we want
    # Extract a single azimuth
    rec_rhi = wrl.util.cross_section_ppi(datasel, azimuth, crs=proj_utm, method="nearest")
else:
    print("Data only has one elevation.")
    
# Plot
rec_rhi[mom].plot(cmap=cmap, norm=norm, x="gr", y="z", ylim=(0,14000), xlim=(-xylims,xylims))

# THIS LAST PART WAS NOT CHECKED
if plot_ML:
    cax = plt.gca()
    datasel["z"].wrl.vis.plot(ax=cax,
                          levels=[datasel["height_ml_bottom_new_gia"], datasel["height_ml_new_gia"]], 
                          cmap="black",
                          func="contour")
    # datasel["z"].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))

if isvolume: elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).sweep_fixed_angle.values, 2))+"°"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values).split(".")[0])

#%% Load QVPs
# Load only events with ML detected (pre-condition for stratiform)
ff_ML = "/data/polara/upload/jgiles/dwd/qvps/2015/*/*/pro/vol5minng01/07/ML_detected.txt"
# ff_ML = "/automount/realpep/upload/jgiles/dmi/qvps/2018/*/*/HTY/*/*/ML_detected.txt"
ff_ML_glob = glob.glob(ff_ML)

if "dmi" in ff_ML:
    # create a function to only select the elevation closer to 10 for each date
    from collections import defaultdict
    def get_closest_elevation(paths):
        elevation_dict = defaultdict(list)
        for path in paths:
            parts = path.split('/')
            date = parts[-5]
            elevation = float(parts[-2])
            elevation_dict[date].append((elevation, path))
    
        result_paths = []
        for date, elevations in elevation_dict.items():
            closest_elevation_path = min(elevations, key=lambda x: abs(x[0] - 10))[1]
            result_paths.append(closest_elevation_path)
    
        return result_paths
    
    ff_ML_glob = get_closest_elevation(ff_ML_glob)

ff = [glob.glob(os.path.dirname(fp)+"/*allmoms*")[0] for fp in ff_ML_glob ]

alignz = False
if "dwd" in ff_ML: alignz = True
ds_qvps = utils.load_qvps(ff, align_z=alignz, fix_TEMP=True, fillna=True)


# Conditions to clean ML height values
max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
time_window = 5 # set timestep window for the std computation (centered)
min_period = 3 # set minimum number of valid ML values in the window (centered)

cond_ML_bottom_change = abs(ds_qvps["height_ml_bottom_new_gia"].diff("time").compute())<max_change
cond_ML_bottom_std = ds_qvps["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_bottom_minlen = qvps["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

cond_ML_top_change = abs(ds_qvps["height_ml_new_gia"].diff("time").compute())<max_change
cond_ML_top_std = ds_qvps["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_top_minlen = qvps["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

# reduce to daily condition
allcond_daily = allcond.resample(time="D").any().dropna("time")
allcond_daily = allcond_daily.where(allcond_daily, drop=True)

# Filter only events with clean ML (requeriment for stratiform) on a daily basis
# (not efficient and not elegant but I could not find other solution)
ds_qvps = ds_qvps.isel(time=[date.values in  allcond_daily.time.dt.date for date in ds_qvps.time.dt.date])


#%% Plot QPVs interactive, with matplotlib backend (working) fix in holoviews/plotting/mpl/raster.py
# this works with a manual fix in the holoviews files.
# In Holoviews 1.17.1, add the following to line 192 in holoviews/plotting/mpl/raster.py:
# if 'norm' in plot_kwargs: # vmin/vmax should now be exclusively in norm
#          	plot_kwargs.pop('vmin', None)
#          	plot_kwargs.pop('vmax', None)

hv.extension("matplotlib")

max_height = 12000 # max height for the qvp plots (necessary because of random high points and because of dropna in the z dim)

var_options = ['RHOHV', 'ZDR_OC', 'KDP_ML_corrected', 'ZDR', 
               # 'TH','UPHIDP',  # not so relevant
#               'UVRADH', 'UZDR',  'UWRADH', 'VRADH', 'SQIH', # not implemented yet in visdict14
               # 'WRADH', 'SNRHC', 'URHOHV', 'SNRH',
                'KDP', 'RHOHV_NC', 'UPHIDP_OC']


vars_to_plot = ['DBZH', 'KDP_ML_corrected', 'KDP', 'ZDR_OC', 'RHOHV_NC', 
                'UPHIDP_OC', 'ZDR', 'RHOHV' ]

# add missing units for PHIDP variables in turkish data (this was fixed on 28/12/23 but previous calculations have missing units)
for vv in ds_qvps.data_vars:
    if "PHIDP" in vv:
        if "units" not in ds_qvps[vv].attrs:
            ds_qvps[vv].attrs["units"] = "degrees"

visdict14 = radarmet.visdict14

# define a function for plotting a custom discrete colorbar
def cbar_hook(hv_plot, _, cmap_extend, ticklist, norm, label):
    COLORS = cmap_extend
    BOUNDS = ticklist
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax = hv_plot.handles["axis"]
    fig = hv_plot.handles["fig"]
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.35])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap_extend, norm=norm),
        cax=cbar_ax,
        extend='both',
        ticks=ticklist[1:-1],
        # spacing='proportional',
        orientation='vertical',
        label=label,
    )   


# Define the function to update plots
def update_plots(selected_day, show_ML_lines, show_min_entropy):
    selected_data = ds_qvps.sel(time=selected_day, z=slice(0, max_height)).dropna("z", how="all")
    available_vars = vars_to_plot

    plots = []

    for var in available_vars:
        ticks = visdict14[var]["ticks"]
        cmap = visdict14[var]["cmap"] # I need the cmap with extreme colors too here
        # cmap_list = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        cmap_extend = utils.get_discrete_cmap(ticks, cmap)
        ticklist = [-100]+list(ticks)+[100]
        norm = utils.get_discrete_norm(ticks, cmap_extend)

        subtitle = var
        if var == "ZDR_OC":
            # for the plot of ZDR_OC, put the value of the offset in the subtitle if it is daily
            if np.unique((selected_data["ZDR"]-selected_data["ZDR_OC"]).compute().median("z")).std() < 0.01:
                # if the std of the unique values of ZDR - ZDR_OC is < 0.1 we assume it is a daily offset
                subtitle = var+" (Offset: "+str(np.round((selected_data["ZDR"]-selected_data["ZDR_OC"]).compute().median().values,3))+")"                
            else:
                subtitle = var+" (Offset: variable per timestep)"
        if var == "DBZH": # add elevation angle to DBZH panel
            subtitle = var+" (Elevation: "+str(np.round(selected_data['sweep_fixed_angle'].mean().compute().values, 2))+"°)"

        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', title=subtitle,
            xlabel='Time', ylabel='Height (m)', colorbar=False,
            width=500, height=250, norm=norm, ylim=(None, max_height),
        ).opts(
                cmap=cmap_extend,
                color_levels=ticks.tolist(),
                clim=(ticks[0], ticks[-1]),
                xformatter = mpl.dates.DateFormatter('%H:%M'), # put only the hour in the x-axis
                hooks=[partial(cbar_hook, cmap_extend=cmap_extend, ticklist=ticklist, norm=norm, label=selected_data[var].units)],
#!!! TO DO: format tick labels so that only the hour is shown: https://discourse.holoviz.org/t/how-to-the-format-of-datetime-x-axis-ticks-in-matplotlib-backend/6344
            )

        # Add line plots for height_ml_new_gia and height_ml_bottom_new_gia
        if show_ML_lines:
            # this parts works better and simpler with holoviews directly instead of hvplot
            line1 = hv.Curve(
                (selected_data.time, selected_data["height_ml_new_gia"]),
                # line_color='black', line_width=2, line_dash='dashed', legend=False # bokeh naming?
            ).opts(color='black', linewidth=2, show_legend=False)
            line2 = hv.Curve(
                (selected_data.time, selected_data["height_ml_bottom_new_gia"]),
            ).opts(color='black', linewidth=2, show_legend=False)

            quadmesh = (quadmesh * line1 * line2)

        # Add shading for min_entropy when it's greater than 0.8
        if show_min_entropy:
            min_entropy_values = selected_data.min_entropy.where(selected_data.min_entropy>=0).dropna("z", how="all").compute()
            
            min_entropy_shading = min_entropy_values.hvplot.quadmesh(
                x='time', y='z', 
                xlabel='Time', ylabel='Height (m)', colorbar=False,
                width=500, height=250,
            ).opts(
                    cmap=['#ffffff00', "#B5B1B1", '#ffffff00'],
                    color_levels=[0, 0.8,1, 1.1],
                    clim=(0, 1.1),
                    alpha=0.8
                )
            quadmesh = (quadmesh * min_entropy_shading)

            
        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/3)]),
                         pn.Row(*plots[round(nplots/3):round(nplots/3)*2]),
                         pn.Row(*plots[round(nplots/3)*2:]),
                         )
    return gridplot
    # return pn.Row(*plots)
        

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str,
                                                value=date_range_str[0], width=600)

show_ML_lines_toggle = pn.widgets.Toggle(name='Show ML Lines', value=True)

show_min_entropy_toggle = pn.widgets.Toggle(name='Show Entropy over 0.8', value=True) 

@pn.depends(selected_day_slider.param.value, show_ML_lines_toggle, show_min_entropy_toggle)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    show_ML_lines = show_ML_lines_toggle.value
    show_min_entropy = show_min_entropy_toggle.value 
    plot = update_plots(selected_day, show_ML_lines, show_min_entropy)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
show_ML_lines_toggle.param.watch(update_plots_callback, 'value')
show_min_entropy_toggle.param.watch(update_plots_callback, 'value') 

# Create the initial plot
initial_day = str(start_date)
initial_ML_lines = True
initial_min_entropy = True 

plot_panel = pn.Row(update_plots(initial_day, initial_ML_lines, initial_min_entropy))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    pn.Row(show_ML_lines_toggle, show_min_entropy_toggle),
    plot_panel
)


layout.save("/user/jgiles/interactive_matplotlib.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)



#%% Plot QPVs interactive, with matplotlib backend, variable selector (working) fix in holoviews/plotting/mpl/raster.py
# this works with a manual fix in the holoviews files.
# In Holoviews 1.17.1, add the following to line 192 in holoviews/plotting/mpl/raster.py:
# if 'norm' in plot_kwargs: # vmin/vmax should now be exclusively in norm
#          	plot_kwargs.pop('vmin', None)
#          	plot_kwargs.pop('vmax', None)

# Plots DBZH and RHOHV_NC fixed in left panels and selectable variables in right panels
# only works for a short period of time (about 15 days), otherwise the file gets too big and it won't load in the browser

hv.extension("matplotlib")

var_options = ['RHOHV', 'ZDR_OC', 'KDP_ML_corrected', 'ZDR', 
               # 'TH','UPHIDP',  # not so relevant
#               'UVRADH', 'UZDR',  'UWRADH', 'VRADH', 'SQIH', # not implemented yet in visdict14
               # 'WRADH', 'SNRHC', 'URHOHV', 'SNRH',
                'KDP', 'RHOHV_NC', 'UPHIDP_OC']

var_starting1 = "ZDR_OC"
var_starting2 = "KDP_ML_corrected"
var_fix1 = "DBZH"
var_fix2 = "RHOHV_NC"

visdict14 = radarmet.visdict14

# define a function for plotting a custom discrete colorbar
def cbar_hook(hv_plot, _, cmap_extend, ticklist, norm, label):
    COLORS = cmap_extend
    BOUNDS = ticklist
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax = hv_plot.handles["axis"]
    fig = hv_plot.handles["fig"]
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.35])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap_extend, norm=norm),
        cax=cbar_ax,
        extend='both',
        ticks=ticklist[1:-1],
        # spacing='proportional',
        orientation='vertical',
        label=label,
    )   


# Define the function to update plots
def update_plots(selected_day, selected_var1, selected_var2, show_ML_lines, show_min_entropy):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = [ var_fix1, selected_var1, var_fix2, selected_var2 ]

    plots = []

    for var in available_vars:
        ticks = visdict14[var]["ticks"]
        cmap = visdict14[var]["cmap"] # I need the cmap with extreme colors too here
        cmap_list = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        cmap_extend = utils.get_discrete_cmap(ticks, cmap)
        ticklist = [-100]+list(ticks)+[100]
        norm = utils.get_discrete_norm(ticks, cmap_extend)

        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=False,
            width=500, height=250, norm=norm,
        ).opts(
                cmap=cmap_extend,
                color_levels=ticks.tolist(),
                clim=(ticks[0], ticks[-1]),
                hooks=[partial(cbar_hook, cmap_extend=cmap_extend, ticklist=ticklist, norm=norm, label=selected_data[var].units)],

            )

        # Add line plots for height_ml_new_gia and height_ml_bottom_new_gia
        if show_ML_lines:
            # this parts works better and simpler with holoviews directly instead of hvplot
            line1 = hv.Curve(
                (selected_data.time, selected_data["height_ml_new_gia"]),
                # line_color='black', line_width=2, line_dash='dashed', legend=False # bokeh naming?
            ).opts(color='black', linewidth=2, show_legend=False)
            line2 = hv.Curve(
                (selected_data.time, selected_data["height_ml_bottom_new_gia"]),
            ).opts(color='black', linewidth=2, show_legend=False)

            quadmesh = (quadmesh * line1 * line2)

        # Add shading for min_entropy when it's greater than 0.8
        if show_min_entropy:
            min_entropy_values = selected_data.min_entropy.dropna("z", how="all").interpolate_na(dim="z").compute()
            
            min_entropy_shading = min_entropy_values.hvplot.quadmesh(
                x='time', y='z', 
                xlabel='Time', ylabel='Height (m)', colorbar=False,
                width=500, height=250,
            ).opts(
                    cmap=['#ffffff00', "#B5B1B1", '#ffffff00'],
                    color_levels=[0, 0.8,1, 1.1],
                    clim=(0, 1.1),
                    alpha=0.8
                )
            quadmesh = (quadmesh * min_entropy_shading)

            
        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)
        

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

var1_selector = pn.widgets.Select(name='Select Variable 1', 
                                                  value=var_starting1, 
                                                  options=var_options,
                                                  # inline=True
                                                  )

var2_selector = pn.widgets.Select(name='Select Variable 2', 
                                                  value=var_starting2, 
                                                  options=var_options,
                                                  # inline=True
                                                  )

show_ML_lines_toggle = pn.widgets.Toggle(name='Show ML Lines', value=True) 
show_min_entropy_toggle = pn.widgets.Toggle(name='Show Entropy over 0.8', value=True) 


@pn.depends(selected_day_slider.param.value, var1_selector.param.value, var2_selector.param.value,
            show_ML_lines_toggle, show_min_entropy_toggle) 
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_var1 = var1_selector.value
    selected_var2 = var2_selector.value
    show_ML_lines = show_ML_lines_toggle.value
    show_min_entropy = show_min_entropy_toggle.value
    plot = update_plots(selected_day, selected_var1, selected_var2, show_ML_lines, show_min_entropy)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
var1_selector.param.watch(update_plots_callback, 'value')
var2_selector.param.watch(update_plots_callback, 'value')
show_ML_lines_toggle.param.watch(update_plots_callback, 'value')
show_min_entropy_toggle.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_var1 = var_starting1
initial_var2 = var_starting2
initial_ML_lines = True
initial_min_entropy = True

plot_panel = pn.Row(update_plots(initial_day, initial_var1, initial_var2, initial_ML_lines, initial_min_entropy))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    var1_selector,
    var2_selector,
    show_ML_lines_toggle,
    show_min_entropy_toggle,
    plot_panel
)


layout.save("/user/jgiles/interactive_matplotlib_variable_selector.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)


#%% Plot QPVs interactive, with matplotlib backend (testing) fix in holoviews/plotting/mpl/raster.py
# this works with a manual fix in the holoviews files.
# In Holoviews 1.17.1, add the following to line 192 in holoviews/plotting/mpl/raster.py:
# if 'norm' in plot_kwargs: # vmin/vmax should now be exclusively in norm
#          	plot_kwargs.pop('vmin', None)
#          	plot_kwargs.pop('vmax', None)


hv.extension("matplotlib")

var_options = ['RHOHV', 'ZDR_OC', 'KDP_ML_corrected', 'ZDR', 
               # 'TH','UPHIDP',  # not so relevant
#               'UVRADH', 'UZDR',  'UWRADH', 'VRADH', 'SQIH', # not implemented yet in visdict14
               # 'WRADH', 'SNRHC', 'URHOHV', 'SNRH',
                'KDP', 'RHOHV_NC', 'UPHIDP_OC']
var_options = ["ZDR_OC", "KDP_ML_corrected"]

# var_starting = ['DBZH', 'RHOHV_NC', 'ZDR_OC', "KDP_ML_corrected"]
var_starting1 = "ZDR_OC"
var_starting2 = "KDP_ML_corrected"
var_fix1 = "DBZH"
var_fix2 = "RHOHV_NC"

visdict14 = radarmet.visdict14

# define a function for plotting a custom discrete colorbar
def cbar_hook(hv_plot, _, cmap_extend, ticklist, norm, label):
    COLORS = cmap_extend
    BOUNDS = ticklist
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax = hv_plot.handles["axis"]
    fig = hv_plot.handles["fig"]
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.35])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap_extend, norm=norm),
        cax=cbar_ax,
        extend='both',
        ticks=ticklist[1:-1],
        # spacing='proportional',
        orientation='vertical',
        label=label,
    )   


# Define the function to update plots
def update_plots(selected_day, selected_var1, selected_var2, show_ML_lines, show_min_entropy):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = [ var_fix1, selected_var1, var_fix2, selected_var2 ]

    plots = []

    for var in available_vars:
        ticks = visdict14[var]["ticks"]
        cmap = visdict14[var]["cmap"] # I need the cmap with extreme colors too here
        cmap_list = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        cmap_extend = utils.get_discrete_cmap(ticks, cmap)
        ticklist = [-100]+list(ticks)+[100]
        norm = utils.get_discrete_norm(ticks, cmap_extend)

        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=False,
            width=500, height=250, norm=norm,
        ).opts(
                cmap=cmap_extend,
                color_levels=ticks.tolist(),
                clim=(ticks[0], ticks[-1]),
                hooks=[partial(cbar_hook, cmap_extend=cmap_extend, ticklist=ticklist, norm=norm, label=selected_data[var].units)],

            )

            
        # Add line plots for height_ml_new_gia and height_ml_bottom_new_gia
        if show_ML_lines:
            # this parts works better and simpler with holoviews directly instead of hvplot
            line1 = hv.Curve(
                (selected_data.time, selected_data["height_ml_new_gia"]),
                # line_color='black', line_width=2, line_dash='dashed', legend=False # bokeh naming?
            ).opts(color='black', linewidth=2, show_legend=False)
            line2 = hv.Curve(
                (selected_data.time, selected_data["height_ml_bottom_new_gia"]),
            ).opts(color='black', linewidth=2, show_legend=False)

            quadmesh = (quadmesh * line1 * line2)

        # Add shading for min_entropy when it's greater than 0.8
        if show_min_entropy:
            min_entropy_values = selected_data.min_entropy.dropna("z", how="all").interpolate_na(dim="z").compute()
            
            min_entropy_shading = min_entropy_values.hvplot.quadmesh(
                x='time', y='z', 
                xlabel='Time', ylabel='Height (m)', colorbar=False,
                width=500, height=250,
            ).opts(
                    cmap=['#ffffff00', "#B5B1B1", '#ffffff00'],
                    color_levels=[0, 0.8,1, 1.1],
                    clim=(0, 1.1),
                    alpha=0.8
                )
            quadmesh = (quadmesh * min_entropy_shading)
            
        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)
        

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))[0:2]

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

var1_selector = pn.widgets.Select(name='Select Variable 1', 
                                                  value=var_starting1, 
                                                  options=var_options,
                                                  # inline=True
                                                  )

var2_selector = pn.widgets.Select(name='Select Variable 2', 
                                                  value=var_starting2, 
                                                  options=var_options,
                                                  # inline=True
                                                  )

show_ML_lines_toggle = pn.widgets.Toggle(name='Show ML Lines', value=True)
show_min_entropy_toggle = pn.widgets.Toggle(name='Show Entropy over 0.8', value=True)


@pn.depends(selected_day_slider.param.value, var1_selector.param.value, var2_selector.param.value,
            show_ML_lines_toggle, show_min_entropy_toggle)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_var1 = var1_selector.value
    selected_var2 = var2_selector.value
    show_ML_lines = show_ML_lines_toggle.value
    show_min_entropy = show_min_entropy_toggle.value
    plot = update_plots(selected_day, selected_var1, selected_var2, show_ML_lines, show_min_entropy)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
var1_selector.param.watch(update_plots_callback, 'value')
var2_selector.param.watch(update_plots_callback, 'value')
show_ML_lines_toggle.param.watch(update_plots_callback, 'value')
show_min_entropy_toggle.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_var1 = var_starting1
initial_var2 = var_starting2
initial_ML_lines = True
initial_min_entropy = True

plot_panel = pn.Row(update_plots(initial_day, initial_var1, initial_var2, initial_ML_lines, initial_min_entropy))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    var1_selector,
    var2_selector,
    show_ML_lines_toggle,
    show_min_entropy_toggle,
    plot_panel
)


layout.save("/user/jgiles/interactive_matplotlib.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)


#%% Plot QPVs interactive (working)
import panel as pn
from bokeh.resources import INLINE

var_options = ['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected',
               'UVRADH', 'UZDR', 'ZDR', 'UWRADH', 'TH', 'VRADH', 'SQIH',
               'WRADH', 'UPHIDP', 'KDP', 'SNRHC', 'SQIH',
                'URHOHV', 'SNRH', 'RHOHV_NC', 'UPHIDP_OC']

var_starting = ['DBZH', 'ZDR_OC', 'KDP_ML_corrected', "RHOHV_NC"]

# Define the function to update plots
def update_plots(selected_day, selected_vars):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = selected_vars

    plots = []

    for var in available_vars:
        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', cmap='viridis', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=True
        ).opts(width=800, height=400)

        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

selected_vars_selector = pn.widgets.CheckBoxGroup(name='Select Variables', 
                                                  value=var_starting, 
                                                  options=var_options,
                                                  inline=True)

# # this works but the file is so large that it is not loading in Firefox or Chrome
# selected_vars_selector = pn.widgets.Select(name='Select Variables', 
#                                                   value="ZDR", 
#                                                   options=["ZDR", "ZDR_OC", "UZDR"],
#                                                   )


@pn.depends(selected_day_slider.param.value)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_vars = selected_vars_selector.value
    plot = update_plots(selected_day, selected_vars)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
selected_vars_selector.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_vars = var_starting
# initial_vars = "ZDR"
plot_panel = pn.Row(update_plots(initial_day, initial_vars))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    # selected_vars_selector, # works with pn.widgets.Select but creates too-large files that do not load
    plot_panel
)


# Display or save the plot as an HTML file
# pn.serve(layout)

layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)

# layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
#             states={"Select Date":date_range_str, "Select Variables": var_options}, 
#             max_states=1000, max_opts=1000)


#%% Plot QVPs interactive (testing)
from functools import partial, reduce
import panel as pn
from bokeh.resources import INLINE
from bokeh.models import FixedTicker
from bokeh.models import CategoricalColorMapper, ColorBar
from bokeh.colors import Color

var_options = ['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected',
               'UVRADH', 'UZDR', 'ZDR', 'UWRADH', 'TH', 'VRADH', 'SQIH',
               'WRADH', 'UPHIDP', 'KDP', 'SNRHC', 'SQIH',
                'URHOHV', 'SNRH', 'RHOHV_NC', 'UPHIDP_OC']

var_starting = ['DBZH', 'ZDR_OC', "ZDR", 'KDP_ML_corrected', "RHOHV_NC", "RHOHV"]
var_starting = ['DBZH', 'ZDR_OC', 'KDP_ML_corrected', "RHOHV_NC"]

visdict14 = radarmet.visdict14

# for testing with a shorter timespan
ds_qvps = ds_qvps.sel(time=slice("2015-01-02","2015-01-10"))

# Define the function to update plots
def update_plots(selected_day, selected_vars):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = selected_vars

    plots = []

    # define a function for plotting a discrete colorbar with equal color ranges
    def cbar_hook(hv_plot, _, cmap, ticklist):
        COLORS = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        BOUNDS = ticklist
        plot = hv_plot.handles["plot"]
        factors = [f"{BOUNDS[i]} - {BOUNDS[i + 1]}" for i in range(len(COLORS))]
        mapper = CategoricalColorMapper(
            palette=COLORS,
            factors=factors,
        )
        color_bar = ColorBar(color_mapper=mapper)
        plot.right[0] = color_bar


    for var in available_vars:
        # set colorbar settings
        # ticklist = list(visdict14[var]["ticks"])
        ticklist = [-100]+list(visdict14[var]["ticks"])+[100]
        norm = utils.get_discrete_norm(visdict14[var]["ticks"])
        # cmap = utils.get_discrete_cmap(visdict14[var]["ticks"], visdict14[var]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
        cmap = visdict14[var]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
        cmap_list = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]

        # quadmesh = selected_data[var].hvplot.quadmesh(
        #     x='time', y='z', cmap='viridis', title=var,
        #     xlabel='Time', ylabel='Height (m)', colorbar=True
        # ).opts(width=800, height=400, 
        #        cmap=cmap, color_levels=ticklist, clim=(ticklist[0], ticklist[-1]), 
        #         # colorbar_opts = {'ticker': FixedTicker(ticks=ticklist),},  # this changes nothing
        #         hooks=[partial(cbar_hook, cmap=cmap, ticklist=ticklist)],
        #        )

               
        quadmesh = []
        for n,cc in enumerate(cmap_list):
            # select only the data in this interval of the colorbar
            interval_data = selected_data[var].where(selected_data[var]>=ticklist[n]).where(selected_data[var]<ticklist[n+1])
            
            quadmesh.append(interval_data.hvplot.quadmesh(
                x='time', y='z', cmap='viridis', title=var,
                xlabel='Time', ylabel='Height (m)', colorbar=True
                ).opts(
                cmap=[cmap_list[n]],
                color_levels=ticklist[n:n+2],
                clim=(ticklist[n], ticklist[n+1]),
                width=800, height=400,
            ))
        
        quadmesh = reduce((lambda x, y: x * y), quadmesh) *\
            selected_data[var].hvplot.quadmesh(
                x='time', y='z', cmap='viridis', title=var,
                xlabel='Time', ylabel='Height (m)', colorbar=True
                ).opts(
                cmap=['#ffffff00'], # add a transparent layer only to recover the values when hovering with mouse
                color_levels=ticklist[0]+ticklist[-1],
                clim=(ticklist[0], ticklist[-1]),
                hooks=[partial(cbar_hook, cmap=cmap, ticklist=ticklist)],
                width=800, height=400,
            )
        
        
               
        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

selected_vars_selector = pn.widgets.CheckBoxGroup(name='Select Variables', 
                                                  value=var_starting, 
                                                  options=var_options,
                                                  inline=True)

# # this works but the file is so large that it is not loading in Firefox or Chrome
# selected_vars_selector = pn.widgets.Select(name='Select Variables', 
#                                                   value="ZDR", 
#                                                   options=["ZDR", "ZDR_OC", "UZDR"],
#                                                   )


@pn.depends(selected_day_slider.param.value)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_vars = selected_vars_selector.value
    plot = update_plots(selected_day, selected_vars)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
selected_vars_selector.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_vars = var_starting
# initial_vars = "ZDR"
plot_panel = pn.Row(update_plots(initial_day, initial_vars))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    # selected_vars_selector, # works with pn.widgets.Select but creates too-large files that do not load
    plot_panel
)


# Display or save the plot as an HTML file
# pn.serve(layout)

layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)

# layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
#             states={"Select Date":date_range_str, "Select Variables": var_options}, 
#             max_states=1000, max_opts=1000)



#%% TEST interactive plot (shows the error with the values not correctly assign to colors)
import xarray as xr
import hvplot.xarray
from bokeh.models import CategoricalColorMapper, ColorBar
hv.extension("bokeh")

COLORS = ["magenta", "green", "yellow", "orange", "red"]
BOUNDS = [-273.15, -43, -23, -3, 27, 47]
COLORS = ["magenta", "green", "yellow", "orange", "red", "purple"]
BOUNDS = [0, 230, 250, 270, 300, 320, 400]

def cbar_hook(hv_plot, _):
    plot = hv_plot.handles["plot"]
    factors = [f"{BOUNDS[i]} - {BOUNDS[i + 1]}" for i in range(len(COLORS))]
    mapper = CategoricalColorMapper(
        palette=COLORS,
        factors=factors,
    )
    color_bar = ColorBar(color_mapper=mapper)
    plot.right[0] = color_bar


ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)#-273.15
layout = pn.Column( ds.hvplot.quadmesh("lon", "lat").opts(
    cmap=COLORS,
    color_levels=BOUNDS,
    clim=(BOUNDS[0], BOUNDS[-1]),
    # hooks=[cbar_hook],
    width=800, height=400,
)
)

layout.save("/user/jgiles/interactive_test.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)

#%% Compute ZDR VP calibration

zdr_offset_belowML = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="step").compute()
zdr_offset_belowML_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="all").compute()

zdr_offset_inML = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_bottom_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="height_ml_new_gia", min_h=min_height, timemode="step").compute()
zdr_offset_inML_all = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_bottom_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="height_ml_new_gia", min_h=min_height, timemode="all").compute()

zdr_offset_aboveML = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_aboveML_all = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()

zdr_offset_whole = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_whole_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()

cond_noML = ((ds.z>ds.height_ml_new_gia) + (ds.z<ds.height_ml_bottom_new_gia)).compute()
zdr_offset_whole_noML = utils.zdr_offset_detection_vps(ds.where(cond_noML), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_whole_noML_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()


# Temporary fix because I do not have ERA5 temp downloaded for after 2020
# zdr_offset_aboveML = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
# zdr_offset_aboveML_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

# zdr_offset_whole = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
# zdr_offset_whole_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

# cond_noML = ((ds.z>ds.height_ml_new_gia) + (ds.z<ds.height_ml_bottom_new_gia)).compute()
# zdr_offset_whole_noML = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(cond_noML), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
# zdr_offset_whole_noML_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

#%% Plot ZDR VP calibration 

# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different regions (below ML, in ML, above ML)
mom = "RHOHV_NC"
visdict14 = radarmet.visdict14
cmap = utils.get_discrete_cmap(visdict14[mom]["ticks"], visdict14[mom]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
norm = utils.get_discrete_norm(visdict14[mom]["ticks"], cmap)
templevels = [-100]
date = ds.time[0].values.astype('datetime64[D]').astype(str)

offsets_to_plot = {"Below ML": zdr_offset_belowML,
                   "In ML": zdr_offset_inML,
                   "Above ML": zdr_offset_aboveML,
                   "Whole column": zdr_offset_whole,
                   "Whole column \n no ML": zdr_offset_whole_noML}

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ds_qvp[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
# plot ML limits
ds_qvp["height_ml_bottom_new_gia"].plot(color="black", label="ML") 
ds_qvp["height_ml_new_gia"].plot(color="black")
# Plot min_height
(xr.ones_like(ds_qvp["height_ml_new_gia"])*min_height).plot(color="black")
# ax.text(ds_qvp.time[0]-1, min_height, "min_height")
ax.text(-0.16, min_height/5000, "min_height", transform=ax.transAxes)
plt.legend()
plt.title(mom+" "+loc.upper()+" "+date)
plt.ylabel("height [m]")

ax2=plt.subplot(gs[1], sharex=ax)
ax3 = ax2.twinx()
for noff in offsets_to_plot.keys():
    offsets_to_plot[noff]["ZDR_offset"].plot(label=str(noff), ls="-", ax=ax2, ylim=(-0.3,1))
    ax2.set_ylabel("")
    offsets_to_plot[noff]["ZDR_std_from_offset"].plot(label=str(noff), ls="--", ax=ax3, ylim=(-0.3,1), alpha=0.5)
    ax3.set_ylabel("")
    
ax2.set_title("")
ax3.set_title("Full: offset. Dashed: offset std.")
ax3.set_yticks([],[])
ax3.legend(loc=(1.01,0))

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.4, 0.02, 0.5])
fig.colorbar(figvp, cax=cbar_ax, extend="both")


# Same as above but with separate plots for the line plots
# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different regions (below ML, in ML, above ML)
mom = "RHOHV"
visdict14 = radarmet.visdict14
cmap = utils.get_discrete_cmap(visdict14[mom]["ticks"], visdict14[mom]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
norm = utils.get_discrete_norm(visdict14[mom]["ticks"], cmap)
templevels = [-100]
date = ds.time[0].values.astype('datetime64[D]').astype(str)

offsets_to_plot = {"Below ML": zdr_offset_belowML,
                   "In ML": zdr_offset_inML,
                   "Above ML": zdr_offset_aboveML,
                   "Whole column": zdr_offset_whole,
                   "Whole column \n no ML": zdr_offset_whole_noML}

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ds_qvp[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
# plot ML limits
ds_qvp["height_ml_bottom_new_gia"].plot(color="black", label="ML") 
ds_qvp["height_ml_new_gia"].plot(color="black")
# Plot min_height
(xr.ones_like(ds_qvp["height_ml_new_gia"])*min_height).plot(color="black")
# ax.text(ds_qvp.time[0]-1, min_height, "min_height")
ax.text(-0.16, min_height/5000, "min_height", transform=ax.transAxes)
plt.legend()
plt.title(mom+" "+loc.upper()+" "+date)
plt.ylabel("height [m]")

ax2=plt.subplot(gs[1], sharex=ax)
ax3 = plt.subplot(gs[2], sharex=ax2)
for noff in offsets_to_plot.keys():
    offsets_to_plot[noff]["ZDR_offset"].plot(label=str(noff), ls="-", ax=ax2, ylim=(-0.4,0.1))
    ax2.set_ylabel("")
    offsets_to_plot[noff]["ZDR_std_from_offset"].plot(label=str(noff), ls="-", ax=ax3, ylim=(0,1))
    ax3.set_ylabel("")

ax2.set_title("")
ax3.set_title("")
ax2.text(0.5, 0.9, "Offset", transform=ax2.transAxes, horizontalalignment='center')    
ax3.text(0.5, 0.9, "Standard Dev.", transform=ax3.transAxes, horizontalalignment='center')    
# ax3.set_yticks([],[])
ax2.legend(loc=(1.01,0))

## Custom legend
# Extract the current legend handles and labels
handles = ax2.get_legend().legendHandles
labels = ax2.get_legend().get_texts()

# Modify the legend labels and add a title
new_labels = [str(round(float(offsets_to_plot[noff]["ZDR_offset"].median()), 4)) for noff in offsets_to_plot.keys()]
legend_title = "Daily offsets"

# Create a new legend with the modified handles, labels, and title
ax3.legend(handles=handles, labels=new_labels, title=legend_title, loc=(1.01,0))


fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.55, 0.02, 0.3])
fig.colorbar(figvp, cax=cbar_ax, extend="both")

#%% Timeseries of ZDR offsets

loc0 = "tur"

## Load
f_VPzdroff_below1c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_below1C_00*"
f_VPzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_belowML_00*"
f_VPzdroff_wholecol = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_wholecol_00*"

f_LRzdroff_below1c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_below1C_07*"
f_LRzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_belowML_07*"

VPzdroff_below1c = xr.open_mfdataset(f_VPzdroff_below1c)
VPzdroff_belowML = xr.open_mfdataset(f_VPzdroff_belowML)
VPzdroff_wholecol = xr.open_mfdataset(f_VPzdroff_wholecol)

LRzdroff_below1c = xr.open_mfdataset(f_LRzdroff_below1c)
LRzdroff_below1c = LRzdroff_below1c.where(abs(LRzdroff_below1c["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero
LRzdroff_belowML = xr.open_mfdataset(f_LRzdroff_belowML)
LRzdroff_belowML = LRzdroff_belowML.where(abs(LRzdroff_belowML["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero

# normalize time dim to days
VPzdroff_below1c.coords["time"] = VPzdroff_below1c.indexes["time"].normalize()
VPzdroff_belowML.coords["time"] = VPzdroff_belowML.indexes["time"].normalize()
VPzdroff_wholecol.coords["time"] = VPzdroff_wholecol.indexes["time"].normalize()

LRzdroff_below1c.coords["time"] = LRzdroff_below1c.indexes["time"].normalize()
LRzdroff_belowML.coords["time"] = LRzdroff_belowML.indexes["time"].normalize()


## Load per-timestep versions
f_VPzdroff_below1c_ts = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_below1C_times*"
f_VPzdroff_belowML_ts = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_belowML_times*"
f_VPzdroff_wholecol_ts = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_wholecol_times*"

f_LRzdroff_below1c_ts = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_below1C_times*"
f_LRzdroff_belowML_ts = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_belowML_times*"

VPzdroff_below1c_ts = xr.open_mfdataset(f_VPzdroff_below1c_ts)
VPzdroff_belowML_ts = xr.open_mfdataset(f_VPzdroff_belowML_ts)
VPzdroff_wholecol_ts = xr.open_mfdataset(f_VPzdroff_wholecol_ts)

LRzdroff_below1c_ts = xr.open_mfdataset(f_LRzdroff_below1c_ts)
LRzdroff_below1c_ts = LRzdroff_below1c_ts.where(abs(LRzdroff_below1c_ts["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero
LRzdroff_belowML_ts = xr.open_mfdataset(f_LRzdroff_belowML_ts)
LRzdroff_belowML_ts = LRzdroff_belowML_ts.where(abs(LRzdroff_belowML_ts["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero

## Plot

VPzdroff_below1c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below 1C")
VPzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below ML")
VPzdroff_wholecol["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP whole col")

LRzdroff_below1c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below 1C")
LRzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below ML")

# plt.legend()
lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle.set_sizes([6.0])

plt.ylabel("ZDR offsets")    
plt.ylim(-1,1)
plt.title(loc0.upper())


## Plot interactively

# Filter by min number of VPs/QVPs? if so set minvp > 0
minvp = 8

# put everything in the same dataset
if minvp > 0:   
    zdroffsets = xr.merge([VPzdroff_below1c["ZDR_offset"].rename("VP below 1C").where(
                                VPzdroff_below1c_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                ), 
                           VPzdroff_belowML["ZDR_offset"].rename("VP below ML").where(
                                VPzdroff_belowML_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           VPzdroff_wholecol["ZDR_offset"].rename("VP whole col").where(
                                VPzdroff_wholecol_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           LRzdroff_below1c["ZDR_offset"].rename("ZH-ZDR below 1C").where(
                                LRzdroff_below1c_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           LRzdroff_belowML["ZDR_offset"].rename("ZH-ZDR below ML").where(
                                LRzdroff_belowML_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           ],)
else:
    zdroffsets = xr.merge([VPzdroff_below1c["ZDR_offset"].rename("VP below 1C"), 
                           VPzdroff_belowML["ZDR_offset"].rename("VP below ML"),
                           VPzdroff_wholecol["ZDR_offset"].rename("VP whole col"), 
                           LRzdroff_below1c["ZDR_offset"].rename("ZH-ZDR below 1C"),
                           LRzdroff_belowML["ZDR_offset"].rename("ZH-ZDR below ML"),
                           ],)
    
# add also rolling median of the offsets
zdroffsets_rollmed = xr.merge([
                       zdroffsets["VP below 1C"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP below 1C rolling median"), 
                       zdroffsets["VP below ML"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP below ML rolling median"),
                       zdroffsets["VP whole col"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP whole col rolling median"), 
                       zdroffsets["ZH-ZDR below 1C"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below 1C rolling median"),
                       zdroffsets["ZH-ZDR below ML"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below ML rolling median"),
                       ],)



# Create an interactive scatter plot from the combined Dataset
scatter = zdroffsets.hvplot.scatter(x='time', 
                                    y=['VP below 1C', 'VP below ML', 'VP whole col', 
                                       'ZH-ZDR below 1C', 'ZH-ZDR below ML'],
                                    width=1000, height=400, size=1, muted_alpha=0)

lines = zdroffsets_rollmed.hvplot.line(x='time', 
                                    y=['VP below 1C rolling median', 'VP below ML rolling median', 'VP whole col rolling median', 
                                       'ZH-ZDR below 1C rolling median', 'ZH-ZDR below ML rolling median'],
                                    width=1000, height=400, muted_alpha=0)

# Combine both plots
overlay = (scatter * lines)
# overlay=lines

# Customize the plot (add titles, labels, etc.)
overlay.opts(title="ZDR offsets and 31 day centered rolling median (10 day min, interpolated where NaN)", xlabel="Time", ylabel="ZDR offsets", show_grid=True)

# Save the interactive plot to an HTML file
hv.save(overlay, "/user/jgiles/interactive_plot.html")


## Timeseries of ZDR VP offset above vs below ML
## Load
f_VPzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_belowML_00*"
f_VPzdroff_above0c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_above0C_00*"

VPzdroff_belowML = xr.open_mfdataset(f_VPzdroff_belowML)
VPzdroff_above0c = xr.open_mfdataset(f_VPzdroff_above0c)


## Plot

VPzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below ML")
VPzdroff_above0c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP above 0C")

# plt.legend()
lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle.set_sizes([6.0])

plt.title("")
plt.ylabel("ZDR offsets")    
plt.ylim(-1,1)

## Plot difference

(VPzdroff_belowML["ZDR_offset"]-VPzdroff_above0c["ZDR_offset"]).plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="below-above ML diff")

# add +-0.1 lines
(xr.ones_like(VPzdroff_belowML["ZDR_offset"])*0.1).plot(x="time", c="black", label="0.1")
(xr.ones_like(VPzdroff_belowML["ZDR_offset"])*-0.1).plot(x="time", c="black", label="-0.1")

# plt.legend()
lgnd = plt.legend()
lgnd.legend_handles[0].set_sizes([6.0])

plt.grid()
plt.title("")
plt.ylabel("ZDR offsets")    
plt.ylim(-0.3,0.3)

#%% Compare ZDR calibrations to RCA from Veli
f_rca_pro = "/automount/agradar/velibor/data/rca/cal_rca_"+loc0+".nc"

rca_pro = xr.open_dataset(f_rca_pro)
# normalize time
rca_pro.coords["time"] = rca_pro.indexes["time"].normalize()

## Plot

## Scatter over time plot
# Create an interactive scatter plot from the combined Dataset
scatter = zdroffsets.hvplot.scatter(x='time', 
                                    y=['VP below 1C', 'VP below ML', 'VP whole col', 
                                       'ZH-ZDR below 1C', 'ZH-ZDR below ML'],
                                    width=1000, height=400, size=1, muted_alpha=0)
scatter2 = rca_pro.hvplot.scatter(x='time', 
                                    y=["rca_dr"],
                                    width=1000, height=400, size=1, muted_alpha=0)
lines = zdroffsets_rollmed.hvplot.line(x='time', 
                                    y=['VP below 1C rolling median', 'VP below ML rolling median', 'VP whole col rolling median', 
                                       'ZH-ZDR below 1C rolling median', 'ZH-ZDR below ML rolling median'],
                                    width=1000, height=400, muted_alpha=0)

# Combine both plots
overlay = (scatter * scatter2 * lines)
# overlay=lines

# Customize the plot (add titles, labels, etc.)
overlay.opts(title="ZDR offsets and 31 day centered rolling median (10 day min, interpolated where NaN) compared to RCA", xlabel="Time", ylabel="ZDR offsets", show_grid=True)

# Save the interactive plot to an HTML file
hv.save(overlay, "/user/jgiles/interactive_plot.html")


## Scatter of ZDR calibrations vs RCA

plt.scatter(zdroffsets["VP below ML"], rca_pro.rca_dr.where(zdroffsets["VP below ML"]))

corr = xr.corr(zdroffsets["VP below ML"], rca_pro.rca_dr.where(zdroffsets["VP below ML"]), dim="time").compute()


plt.scatter(zdroffsets["VP whole col"], rca_pro.rca_dr.where(zdroffsets["VP whole col"]))

corr = xr.corr(zdroffsets["VP whole col"], rca_pro.rca_dr.where(zdroffsets["VP whole col"]), dim="time").compute()


plt.scatter(zdroffsets["ZH-ZDR below 1C"], rca_pro.rca_dr.where(zdroffsets["ZH-ZDR below 1C"]))

corr = xr.corr(zdroffsets["ZH-ZDR below 1C"], rca_pro.rca_dr.where(zdroffsets["ZH-ZDR below 1C"]), dim="time").compute()

#%% Compare ZDR offset intra-daily variability

selday = "2020-04-28"

# Plot timestep offsets
VPzdroff_below1c_ts.loc[{"time":selday}].ZDR_offset.plot(label=
                                                         "VP below 1C. Daily offset: "+
                                                         str(np.round(VPzdroff_below1c.loc[{"time":selday}].ZDR_offset.values, 2)) +
                                                         ", STD: "+
                                                         str(np.round(VPzdroff_below1c.loc[{"time":selday}].ZDR_std_from_offset.values, 2))
                                                         )
VPzdroff_belowML_ts.loc[{"time":selday}].ZDR_offset.plot(label=
                                                         "VP below ML. Daily offset: "+
                                                         str(np.round(VPzdroff_belowML.loc[{"time":selday}].ZDR_offset.values, 2)) +
                                                         ", STD: "+
                                                         str(np.round(VPzdroff_belowML.loc[{"time":selday}].ZDR_std_from_offset.values, 2))
                                                         )
VPzdroff_wholecol_ts.loc[{"time":selday}].ZDR_offset.plot(label=
                                                          "VP whole col. Daily offset: "+
                                                          str(np.round(VPzdroff_wholecol.loc[{"time":selday}].ZDR_offset.values, 2)) +
                                                          ", STD: "+
                                                          str(np.round(VPzdroff_wholecol.loc[{"time":selday}].ZDR_std_from_offset.values, 2))
                                                          )
LRzdroff_below1c_ts.loc[{"time":selday}].ZDR_offset.plot(label=
                                                         "ZH-ZDR below 1C. Daily offset: "+
                                                         str(np.round(LRzdroff_below1c.loc[{"time":selday}].ZDR_offset.values, 2)) +
                                                         ", STD: "+
                                                         str(np.round(LRzdroff_below1c_ts.loc[{"time":selday}].ZDR_offset.std().values, 2))
                                                         )
LRzdroff_belowML_ts.loc[{"time":selday}].ZDR_offset.plot(label=
                                                         "ZH-ZDR below ML. Daily offset: "+
                                                         str(np.round(LRzdroff_belowML.loc[{"time":selday}].ZDR_offset.values, 2)) +
                                                         ", STD: "+
                                                         str(np.round(LRzdroff_belowML_ts.loc[{"time":selday}].ZDR_offset.std().values, 2))
                                                         )
ax = plt.gca()
ax.set_xlim(LRzdroff_belowML_ts.loc[{"time":selday}].time[0], LRzdroff_belowML_ts.loc[{"time":selday}].time[-1])
plt.legend(fontsize=7)


#%% Create a combined smoothed offset timeseries

##########
# CONCLUSION FROM ANALYSES
##########
# it is not correct to smooth out the offsets. Just use daily offsets to calibrate

# use the zdroffsets combined-dataset from above to hierarchically select offsets.

zdroffsets_comb = xr.where(zdroffsets["VP below ML"].notnull(), zdroffsets["VP below ML"], zdroffsets["VP below 1C"])
zdroffsets_comb = xr.where(zdroffsets_comb.notnull(), zdroffsets_comb, zdroffsets["VP whole col"])
zdroffsets_comb = xr.where(zdroffsets_comb.notnull(), zdroffsets_comb, zdroffsets["ZH-ZDR below ML"])
zdroffsets_comb = xr.where(zdroffsets_comb.notnull(), zdroffsets_comb, zdroffsets["ZH-ZDR below 1C"])

zdroffsets_comb_rollmed = zdroffsets_comb.compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("smoothed combined offsets")

## Plot alongside ZDR offsets and RCA

# Create an interactive scatter plot from the combined Dataset
scatter = zdroffsets.hvplot.scatter(x='time', 
                                    y=['VP below 1C', 'VP below ML', 'VP whole col', 
                                       'ZH-ZDR below 1C', 'ZH-ZDR below ML'],
                                    width=1000, height=400, size=1, muted_alpha=0)
scatter2 = rca_pro.hvplot.scatter(x='time', 
                                    y=["rca_dr"],
                                    width=1000, height=400, size=1, muted_alpha=0)
# we combine the new offset timeseries witht the previous ones, otherwise the line will not be in the legend
zdroffsets_rollmed_extra = zdroffsets_rollmed.assign({"smoothed combined offsets":zdroffsets_comb_rollmed})
lines = zdroffsets_rollmed_extra.hvplot.line(x='time', 
                                    y=['VP below 1C rolling median', 'VP below ML rolling median', 'VP whole col rolling median', 
                                       'ZH-ZDR below 1C rolling median', 'ZH-ZDR below ML rolling median', "smoothed combined offsets"],
                                    width=1000, height=400, muted_alpha=0)

# Combine both plots
overlay = (scatter * scatter2 * lines)
# overlay=lines

# Customize the plot (add titles, labels, etc.)
overlay.opts(title="ZDR offsets and 31 day centered rolling median (10 day min, interpolated where NaN), combined smoothed offsets and RCA", xlabel="Time", ylabel="ZDR offsets", show_grid=True)

# Save the interactive plot to an HTML file
hv.save(overlay, "/user/jgiles/interactive_plot.html")

#%% Timeseries of ZDR offsets (FOR TURKISH DATA)

loc0 = "SVS"

## Load
f_LRzdroff_below1c = "/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/*/*/*-zdr_offset_below1C-*"
f_LRzdroff_belowML = "/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/*/*/*-zdr_offset_belowML-*"

f_LRzdroff_below1c_glob = glob.glob(f_LRzdroff_below1c)
f_LRzdroff_belowML_glob = glob.glob(f_LRzdroff_belowML)

# keep only the elevation closest to 10 for each date
selected_paths = []

for nn, glob0 in enumerate([f_LRzdroff_below1c_glob, f_LRzdroff_belowML_glob]):
    dates = sorted(set([ff.split("/")[-5] for ff in glob0]))
    selected_paths.append([])
    
    for date in dates:
        files = [ff for ff in glob0 if date in ff]
        closest_path = min(files, key=lambda p: abs(float(p.split('/')[-2]) - 10))
        selected_paths[nn].append(closest_path)

f_LRzdroff_below1c_glob = selected_paths[0].copy()
f_LRzdroff_belowML_glob = selected_paths[1].copy()


LRzdroff_below1c = xr.open_mfdataset(f_LRzdroff_below1c_glob)
LRzdroff_below1c = LRzdroff_below1c.where(abs(LRzdroff_below1c["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero
LRzdroff_belowML = xr.open_mfdataset(f_LRzdroff_belowML_glob)
LRzdroff_belowML = LRzdroff_belowML.where(abs(LRzdroff_belowML["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero

# normalize time dim to days
LRzdroff_below1c.coords["time"] = LRzdroff_below1c.indexes["time"].normalize()
LRzdroff_belowML.coords["time"] = LRzdroff_belowML.indexes["time"].normalize()


## Load per-timestep versions
f_LRzdroff_below1c_ts = "/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/*/*/*-zdr_offset_below1C_times*"
f_LRzdroff_belowML_ts = "/automount/realpep/upload/jgiles/dmi/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/*/*/*-zdr_offset_belowML_times*"

f_LRzdroff_below1c_ts_glob = glob.glob(f_LRzdroff_below1c_ts)
f_LRzdroff_belowML_ts_glob = glob.glob(f_LRzdroff_belowML_ts)

# keep only the elevation closest to 10 for each date
selected_paths = []

for nn, glob0 in enumerate([f_LRzdroff_below1c_ts_glob, f_LRzdroff_belowML_ts_glob]):
    dates = sorted(set([ff.split("/")[-5] for ff in glob0]))
    selected_paths.append([])
    
    for date in dates:
        files = [ff for ff in glob0 if date in ff]
        closest_path = min(files, key=lambda p: abs(float(p.split('/')[-2]) - 10))
        selected_paths[nn].append(closest_path)

f_LRzdroff_below1c_ts_glob = selected_paths[0].copy()
f_LRzdroff_belowML_ts_glob = selected_paths[1].copy()

# In case there is repeated time values, get the list of the dates to recompute
# list_to_recomp = [os.path.dirname("".join(pp.split("/calibration/zdr/LR_consistency"))) for pp in f_LRzdroff_below1c_ts_glob if (xr.open_dataset(pp).time.diff("time").astype(int)<=0).any()]
# with open(r'/user/jgiles/recomp_svs.txt', 'w') as fp:
#     for item in list_to_recomp:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#     print('Done')


LRzdroff_below1c_ts = xr.open_mfdataset(f_LRzdroff_below1c_ts_glob)
LRzdroff_below1c_ts = LRzdroff_below1c_ts.where(abs(LRzdroff_below1c_ts["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero
LRzdroff_belowML_ts = xr.open_mfdataset(f_LRzdroff_belowML_ts_glob)
LRzdroff_belowML_ts = LRzdroff_belowML_ts.where(abs(LRzdroff_belowML_ts["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero

## Plot

LRzdroff_below1c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below 1C")
LRzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below ML")

# plt.legend()
lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle.set_sizes([6.0])

plt.ylabel("ZDR offsets")    
plt.ylim(-1,1)
plt.title(loc0.upper())


## Plot interactively

# Filter by min number of VPs/QVPs? if so set minvp > 0
minvp = 8

# put everything in the same dataset
if minvp > 0:   
    zdroffsets = xr.merge([
                           LRzdroff_below1c["ZDR_offset"].rename("ZH-ZDR below 1C").where(
                                LRzdroff_below1c_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           LRzdroff_belowML["ZDR_offset"].rename("ZH-ZDR below ML").where(
                                LRzdroff_belowML_ts["ZDR_offset"].resample({"time":"D"}).count()>=minvp
                                                       ),
                           ],)
else:
    zdroffsets = xr.merge([
                           LRzdroff_below1c["ZDR_offset"].rename("ZH-ZDR below 1C"),
                           LRzdroff_belowML["ZDR_offset"].rename("ZH-ZDR below ML"),
                           ],)
    
# add also rolling median of the offsets
zdroffsets_rollmed = xr.merge([
                       zdroffsets["ZH-ZDR below 1C"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below 1C rolling median"),
                       zdroffsets["ZH-ZDR below ML"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below ML rolling median"),
                       ],)



# Create an interactive scatter plot from the combined Dataset
scatter = zdroffsets.hvplot.scatter(x='time', 
                                    y=['ZH-ZDR below 1C', 'ZH-ZDR below ML'],
                                    width=1000, height=400, size=1, muted_alpha=0)

lines = zdroffsets_rollmed.hvplot.line(x='time', 
                                    y=['ZH-ZDR below 1C rolling median', 'ZH-ZDR below ML rolling median'],
                                    width=1000, height=400, muted_alpha=0)

# Combine both plots
overlay = (scatter * lines)
# overlay=lines

# Customize the plot (add titles, labels, etc.)
overlay.opts(title="ZDR offsets and 31 day centered rolling median (10 day min, interpolated where NaN)", xlabel="Time", ylabel="ZDR offsets", show_grid=True)

# Save the interactive plot to an HTML file
hv.save(overlay, "/user/jgiles/interactive_plot.html")



#%% Check noise correction for RHOHV

X_RHO = "RHOHV"

# calculate the noise level
rho_nc = utils.calculate_noise_level(ds[X_DBZH][0], ds[X_RHO][0], noise=(-40, -20, 1))

# get the "best" noise correction level (acoording to the min std, Veli's way)
ncl = rho_nc[-1]

# get index of the best correction
bci = np.array(rho_nc[-2]).argmin()

# merge into a single array
rho_nc_out = xr.merge(rho_nc[0][bci])

# add noise correction level as attribute
rho_nc_out.attrs["noise correction level"]=ncl


# Correct rhohv just using the SNRHC from the files (from Ryzhkov and Zrnic page 186)
# we assume eta = 1
zdr_nc = wrl.trafo.idecibel(ds["SNRHC"][0]) * ds["ZDR"][0] / (wrl.trafo.idecibel(ds["SNRHC"][0]) + 1 - ds["ZDR"][0])
# zdr_nc = 1 # another approximation would be to set ZDR = 1
rho_nc2 = ds[X_RHO][0] * (1 + 1/wrl.trafo.idecibel(ds["SNRHC"][0].where(ds["SNRHC"][0]>0)) )**0.5 * (1 + zdr_nc/wrl.trafo.idecibel(ds["SNRHC"][0]) )**0.5


## Plot
# plot noise corrected RHOHV with utils.calculate_noise_level
rho_nc_out.RHOHV_NC.plot(vmin=0, vmax=1)
plt.title("Noise corrected RHOHV with own noise calc")

(rho_nc_out.RHOHV_NC>1).plot()
plt.title("Bins with noise corrected RHOHV > 1")

# plot noise corrected RHOHV with DWD SNRHC
rho_nc2.plot(vmin=0, vmax=1)
plt.title("Noise corrected RHOHV with SNRHC from files")

(rho_nc2>1).plot()
plt.title("Bins with noise corrected RHOHV > 1")

# plot the SNRH
rho_nc_out["SNRH"].plot(vmin=-60, vmax=60)
plt.title("SNRH from own noise calc")

ds["SNRHC"][0].plot(vmin=-60, vmax=60)
plt.title("SNRH from DWD")

# plot scatters of RHOHV vs SNR
plt.scatter(rho_nc[0][bci][0], rho_nc[0][bci][1], s=0.01, alpha=0.5)
plt.scatter(ds["SNRH"][0], ds[X_RHO][0], s=0.01, alpha=0.5)

# Plot ZDR and noise corrected ZDR to check if it really makes a difference (looks like it does not)

ds.ZDR[0].plot(vmin=-2, vmax=2)
plt.title("ZDR")

zdr_nc.plot(vmin=-2, vmax=2)
plt.title("noise corrected ZDR")

(ds.ZDR[0] - zdr_nc).plot(vmin=-0.1, vmax=0.1)
plt.title("Difference ZDR - noise corrected ZDR")

#%% Check noise power level in raw DWD files

ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/20*/*/*/pro/vol5minng01/05/*snrhc*"
files = glob.glob(ff)

noise_h = []
noise_v = []
eta = []
date_time = []
for f0 in files:
    aux = dttree.open_datatree(f0)
    noise_h.append(aux["how"]["radar_system"].attrs["noise_H_pw0"])
    noise_v.append(aux["how"]["radar_system"].attrs["noise_V_pw0"])
    eta.append(noise_h[-1]/noise_v[-1])
    
    datestr = aux["what"].attrs["date"]
    timestr = aux["what"].attrs["time"]
    date_time.append(datetime.datetime.strptime(datestr + timestr, "%Y%m%d%H%M%S"))

#%% Testing my version of the raw DWD data with the other one that has more moments
mydata = utils.load_dwd_raw("/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/pro/2016/2016-05-27/pro/vol5minng01/05/*")
otherdata = utils.load_dwd_raw("/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2016/2016-05/2016-05-27/pro/vol5minng01/05/*")

xr.testing.assert_allclose(mydata.SQIV[0], otherdata.SQIV[0])

mydata.DBZH[0].plot()
otherdata.DBZH[0].plot()

(mydata.UVRADH[0] - otherdata.UVRADH[0]).plot()

# check raw metadata
mydataraw = dttree.open_datatree(glob.glob("/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/pro/2016/2016-05-27/pro/vol5minng01/05/*dbzh*")[0])
otherdataraw = dttree.open_datatree(glob.glob("/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2016/2016-05/2016-05-27/pro/vol5minng01/05/*dbzh*")[0])

#%% Check ZDR offsets that come in the turkish data
# We need to load the raw files for this

dmipath = sorted(glob.glob("/automount/realpep/upload/jgiles/dmi_raw/2019/2019-09/2019-09-15/SVS/*"))

# Read attributes of files
radarid = []
dtime = []
taskname = []
elevation = []
nrays_expected = []
nrays_written = []
nbins = []
rlastbin = []
binlength = []
horbeamwidth = []
fpath = []
sweep_number = []
zdr_offset = []

for f in dmipath:
    # print(".", end="")
    # Read metadata
    try:
        m = xd.io.backends.iris.IrisRawFile(f, loaddata=False)
    except ValueError:
        # some files may be empty, ignore them
        print("ignoring empty file: "+f)
        continue
    except EOFError:
        # some files may be corrupt, ignore them
        print("ignoring corrupt file: "+f)
        continue
    except OSError:
        # some files give NULL value error, ignore them
        print("ignoring NULL file: "+f)
        continue
    # Extract info
    fname = os.path.basename(f).split(".")[0]
    radarid_ = fname[0:3]
    dtimestr = fname[3:]
    dtime_ = dt.datetime.strptime(dtimestr, "%y%m%d%H%M%S")
    taskname_ = m.product_hdr["product_configuration"]["task_name"].strip()
    nbins_ = m.nbins
    rlastbin_ = m.ingest_header["task_configuration"]["task_range_info"]["range_last_bin"]/100
    binlength_ = m.ingest_header["task_configuration"]["task_range_info"]["step_output_bins"]/100
    horbeamwidth_ = round(m.ingest_header["task_configuration"]["task_misc_info"]["horizontal_beam_width"], 2)
    zdr_offset_ = m.ingest_header["task_configuration"]["task_calib_info"]["zdr_bias"]
    for i in range(10):
        try:
            nrays_expected_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["number_rays_file_expected"]
            nrays_written_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["number_rays_file_written"]    
            elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["fixed_angle"], 2)
            sweep_number_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ"]["sweep_number"]
            break
        except KeyError:
            try:
                nrays_expected_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_expected"]
                nrays_written_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["number_rays_file_written"]    
                elevation_ = round(m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["fixed_angle"], 2)
                sweep_number_ = m.data[i]["ingest_data_hdrs"]["DB_DBZ2"]["sweep_number"]
                break
            except KeyError:
                continue
    # Append to list
    radarid.append(radarid_)
    dtime.append(dtime_)
    taskname.append(taskname_)
    elevation.append(elevation_)
    nbins.append(nbins_)
    rlastbin.append(rlastbin_)
    binlength.append(binlength_)
    #nrays_expected.append(nrays_expected_)
    #nrays_written.append(nrays_written_)
    fpath.append(f)
    horbeamwidth.append(horbeamwidth_)   
    sweep_number.append(sweep_number_)
    zdr_offset.append(zdr_offset_)

# put attributes in a dataframe
from collections import OrderedDict
df = pd.DataFrame(OrderedDict(
                  {"radarid": radarid,
                   "datetime": dtime,
                   "taskname": taskname,
                   "elevation": elevation,
                   #"nrays_expected": nrays_expected,
                   #"nrays_written": nrays_written,
                   "nbins": nbins,
                   "rlastbin": rlastbin,
                   "binlength": binlength,
                   "horbeamwidth": horbeamwidth,
                   "fpath": fpath,
                    "sweep_number": sweep_number,
                    "zdr_offset": zdr_offset
                  }))


#%% TEST data from Julian S

# ess = utils.load_dwd_preprocessed("/automount/agradar/operation_hydrometeors/data/obs/OpHymet2-case09-20210714/2021/2021-07/2021-07-14/ess/pcpng01/00/ras07-pcpng01_sweeph5onem_allmoms_00-202107140000-202107142355-ess-10410.hd5")

ess = dttree.open_datatree("/automount/agradar/operation_hydrometeors/data/obs/OpHymet2-case09-20210714/2021/2021-07/2021-07-14/ess/pcpng01/00/ras07-pcpng01_sweeph5onem_allmoms_00-202107140000-202107142355-ess-10410.hd5")["sweep_0"].to_dataset()
ess.attrs["sweep_mode"] = "azimuth_surveillance"
ess_corr = utils.fix_flipped_phidp(utils.unfold_phidp(ess.copy(), phidp_lims=(-30,30)))
ess_corr_proc = utils.phidp_processing(ess_corr.copy().pipe(wrl.georef.georeference), azmedian=True)

umd = dttree.open_datatree("/automount/agradar/operation_hydrometeors/data/obs/OpHymet2-case09-20210714/2021/2021-07/2021-07-14/umd/vol5minng01/00/ras07-vol5minng01_sweeph5onem_allmoms_00-202107140000-202107142355-umd-10356.hd5")["sweep_0"].to_dataset()
umd.attrs["sweep_mode"] = "azimuth_surveillance"
umd_corr = utils.fix_flipped_phidp(utils.unfold_phidp(umd.copy(), phidp_lims=(-30,30)))
umd_corr_proc = utils.phidp_processing(umd_corr.copy().pipe(wrl.georef.georeference), azmedian=True)


# plotting
ds=umd_corr_proc

tsel = "2021-07-14T17:45"
tsel = "2021-07-14T08"
if tsel == "":
    datasel = ds
else:
    datasel = ds.sel({"time": tsel}, method="nearest")
    
datasel = datasel.pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "UPHIDP_OC"
xylims = 100000 # xlim and ylim (from -xylims to xylims)

ticks = radarmet.visdict14[mom]["ticks"]
ticks = np.arange(-40,220,20)
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"

plot_over_map = False

if not plot_over_map:
    # plot simple PPI
    # datasel[mom].wrl.plot(x="x", y="y", cmap=cmap, norm=norm)#, xlim=(-xylims,xylims), ylim=(-xylims,xylims))
    datasel[mom].wrl.plot(x="x", y="y", cmap="RdBu_r", vmin=-180, vmax=180)
elif plot_over_map:
    # plot PPI with map coordinates
    fig = plt.figure(figsize=(10, 10))
    datasel[mom].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))
    ax = plt.gca()
    ax.add_feature(cartopy.feature.COASTLINE, linestyle='-', linewidth=1, alpha=0.4)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4)
    ax.gridlines(draw_labels=True)

plt.title(mom+". "+str(datasel.time.values).split(".")[0])

