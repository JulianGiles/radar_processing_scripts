#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:05:32 2023

@author: jgiles

This script computes the ML detection algorithm and entropy values for event classification,
then generates QVPs including sounding temperature values and saves to nc files.

"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

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
from multiprocessing import Pool
from functools import partial

import warnings
warnings.filterwarnings('ignore')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet

import time
start_time = time.time()

#%% Set paths and options. We are going to convert the data for every day of data (i.e. for every daily file)

# path0 = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/" # For testing
path0 = sys.argv[1] # read path from console
overwrite = True # overwrite existing files?

# Set the possible ZDR calibrations locations to include (in order of priority)
# The script will try to correct according to the first offset; if not available or nan it will 
# continue with the next one, and so on. Only the used offset will be outputted in the final file.
# All items in zdrofffile will be tested in each zdroffdir to load the data.
zdroffdir = ["/calibration/zdr/VP/", "/calibration/zdr/LR_consistency/"] # subfolder(s) where to find the zdr offset data
zdrofffile = ["*zdr_offset_belowML_00*", "*zdr_offset_below3C_00*", "*zdr_offset_wholecol_00*",
              "*zdr_offset_belowML_07*", "*zdr_offset_below3C_07*",
              "*zdr_offset_belowML-*", "*zdr_offset_below3C-*"] # pattern to select the appropriate file (careful with the zdr_offset_belowML_timesteps)

# set the RHOHV correction location
rhoncdir = "/rhohv_nc/" # subfolder where to find the noise corrected rhohv data
rhoncfile = "*rhohv_nc_2percent*" # pattern to select the appropriate file (careful with the rhohv_nc_2percent)

# get the files and check that it is not empty
if "hd5" in path0 or "h5" in path0:
    files=[path0]
elif "dwd" in path0:
    files = sorted(glob.glob(path0+"/*allmoms*hd5*"))
elif "dmi" in path0:
    files = sorted(glob.glob(path0+"/*allmoms*h5*"))
elif isinstance(path0, list):
    files = path0
else:
    print("Country code not found in path")
    sys.exit("Country code not found in path.")

if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

# paths to files to load manually
# 07 is the scan number for 12 degree elevation
# path = "/home/jgiles/dwd/pulled_from_detect/*/*/2017-04-12/pro/vol5minng01/07/*allmoms*"
# path = "/automount/ftp/jgiles/pulled_from_detect/*/*/*/tur/vol5minng01/07/*allmoms*"
# files = sorted(glob.glob(path))

min_hgt = 200 # minimum height above the radar to be considered when calculating ML and entropy 
if "dwd" in path0 and "90grads" in path0:
    # for the VP we need to set a higher min height because there are several bins of unrealistic values
    min_hgt = 600
if "ANK" in path0:
    # for ANK we need higher min_hgt to avoid artifacts
    min_hgt = 400
if "GZT" in path0:
    # for GZT we need higher min_hgt to avoid artifacts
    min_hgt = 300


# Use ERA5 temperature profile? If so, it does not use sounding data
era5_temp = True
if os.path.exists("/automount/ags/jgiles/ERA5/hourly/"):
    # then we are in local system
    era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/scratch/detectrea/giles1/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/scratch/detectrea/giles1/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/largedata2/detectdata/projects/A04/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/largedata2/detectdata/projects/A04/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below

# download sounding data?
download_sounding = False
if era5_temp: download_sounding = False

# Code for Sounding data (http://weather.uwyo.edu/upperair/sounding.html)
rs_id = 10868 # Close to radar site. 10393 Lindenberg close to PRO, 10868 Munich close to Turkheim

#!!!!!!!!!!!!! SOUNDING DOWNLOAD AND INCLUSION WAS DEPRECATED IN THIS NEW SCRIPT AND NOT IMPLEMENTED AGAIN 
# save raw sounding data?
save_raw_sounding = True
sounding_savepath = "/automount/ags/jgiles/soundings_wyoming/"

# names of variables
phidp_names = ["UPHIDP", "PHIDP"] # names to look for the PHIDP variable, in order of preference
dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV
zdr_names = ["ZDR"]
th_names = ["TH", "DBZH"]


# we define a funtion to look for loc inside a path string
def find_loc(locs, path):
    components = path.split(os.path.sep)
    for element in locs:
        for component in components:
            if element.lower() in component.lower():
                return element
    return None

locs = ["pro", "tur", "umd", "afy", "ank", "gzt", "hty", "svs"]

# define a function to create save directory and return file save path
def make_savedir(ff, name):
    """
    ff: filepath of the original file
    name: name for the particular folder inside 
    """
    if "dwd" in ff:
        country="dwd"
    elif "dmi" in ff:
        country="dmi"
    else:
        print("Country code not found in path")
        sys.exit("Country code not found in path.")

    ff_parts = ff.split(country)
    savepath = (country+"/qvps/"+name+"/").join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    return savepath

# define a function to split a string at a certain pattern and replace it (like in the function before but only returning the path)
def edit_str(ff, replace, name):
    """
    ff: string of file path or whatever
    replace: what string part to replace
    name: new string to put
    """

    ff_parts = ff.split(replace)
    newff = (name).join(ff_parts)
    return newff

# in case KDP is not in the dataset we defined its attributes according to DWD data:
KDP_attrs={'_Undetect': 0.0,
 'units': 'degrees per kilometer',
 'long_name': 'Specific differential phase HV',
 'standard_name': 'radar_specific_differential_phase_hv'}
#%% Load data

for ff in files:
    
    skipfile=False
    # skip files that are not volume scans (wind, surveillance, etc)
    for skipscan in ["SURVEILLANCE", "WIND", "RHI"]:
        if skipscan in ff:
            print("Skipping: no QVP computed for "+skipscan)
            skipfile=True
            break
    if skipfile:
        continue
    
    # check if the QVP file already exists before starting
    savepath = make_savedir(ff, "")
    if os.path.exists(savepath) and not overwrite:
        continue

    print("processing "+ff)
    if "dwd" in ff:
        basepath=ff.split("dwd")
        data=dttree.open_datatree(ff)["sweep_"+ff.split("/")[-2][1]].to_dataset()
    else:
        data=xr.open_dataset(ff)
        
    # fix time dim in case some value is NaT
    if data.time.isnull().any():
        data.coords["time"] = data["rtime"].min(dim="azimuth", skipna=True).compute()

    # take time out of the coords if necessary
    for coord in ["latitude", "longitude", "altitude", "elevation"]:
        if "time" in data[coord].dims:
            data.coords[coord] = data.coords[coord].min("time")

#%% Georeference
    swp = data.pipe(wrl.georef.georeference) 

#%% Check variable names and add corrections and calibrations
    min_height = min_hgt+swp["altitude"].values

    # get PHIDP name
    for X_PHI in phidp_names:
        if X_PHI in swp.data_vars:
            break
    # get DBZH name
    for X_DBZH in dbzh_names:
        if X_DBZH in swp.data_vars:
            break
    
    # get RHOHV name
    for X_RHO in rhohv_names:
        if X_RHO in swp.data_vars:
            break

    # get ZDR name
    for X_ZDR in zdr_names:
        if X_ZDR in swp.data_vars:
            break

    # get TH name
    for X_TH in th_names:
        if X_TH in swp.data_vars:
            break

    ### Load ZDR offset if available
    if "dwd" in ff:
        country="dwd"
    elif "dmi" in ff:
        country="dmi"

    # We define a custom exception to stop the next nexted loops as soon as a file is loaded
    class FileFound(Exception):
        pass

    try:
        for zdrod in zdroffdir:
            for zdrof in zdrofffile:
                try:
                    zdroffsetpath = os.path.dirname(edit_str(ff, country, country+zdrod))
                    if "/VP/" in zdrod and "/vol5minng01/" in ff:
                        elevnr = ff.split("/vol5minng01/")[-1][0:2]
                        zdroffsetpath = edit_str(zdroffsetpath, "/vol5minng01/"+elevnr, "/90gradstarng01/00")
                        
                    zdr_offset = xr.open_mfdataset(zdroffsetpath+"/"+zdrof)
                    
                    # check that the offset have a valid value. Otherwise skip
                    if zdr_offset["ZDR_offset"].isnull().all():
                        continue
                    
                    # create ZDR_OC variable
                    swp = swp.assign({X_ZDR+"_OC": swp[X_ZDR]-zdr_offset["ZDR_offset"].values})
                    swp[X_ZDR+"_OC"].attrs = swp[X_ZDR].attrs
                    
                    # Change the default ZDR name to the corrected one
                    X_ZDR = X_ZDR+"_OC"
                    
                    # raise the custom exception to stop the loops
                    raise FileFound 
                    
                except OSError:
                    pass
                
        # If no ZDR offset was loaded, print a message
        print("No zdr offset to load: "+zdroffsetpath+"/"+zdrof)
    except FileFound:
        pass
                
                            
    ### Load noise corrected RHOHV if available
    try:
        if "dwd" in ff:
            country="dwd"
        elif "dmi" in ff:
            country="dmi"
        rhoncpath = os.path.dirname(edit_str(ff, country, country+rhoncdir))
        rho_nc = xr.open_mfdataset(rhoncpath+"/"+rhoncfile)
        
        # create RHOHV_NC variable
        swp = swp.assign(rho_nc)
        swp["RHOHV_NC"].attrs["noise correction level"] = rho_nc.attrs["noise correction level"]
        
        # Check that the corrected RHOHV does not have much higher STD than the original (50% more)
        # if that is the case we take it that the correction did not work well so we won't use it
        if not (swp["RHOHV"].std()*1.5 < swp["RHOHV_NC"].std()).compute():
            # Change the default RHOHV name to the corrected one
            X_RHO = X_RHO+"_NC"
            
    except OSError:
        print("No noise corrected rhohv to load: "+rhoncpath+"/"+rhoncfile)

#%% Correct PHIDP 
    ################## Before entropy calculation we need to use the melting layer detection algorithm 
    ds = swp
    interpolation_method_ML = "linear" # for interpolating PHIDP in the ML
    
    # Check that PHIDP is in data, otherwise skip ML detection
    if X_PHI in data.data_vars:
        # Set parameters according to data
        if "dwd" in ff:
            country="dwd"
            window0=7 # number of range bins for phidp smoothing (this one is quite important!)
            winlen0=7 # size of range window (bins) for the kdp-phidp calculations
            xwin0=9 # window size (bins) for the time rolling median smoothing in ML detection
            ywin0=1 # window size (bins) for the height rolling mean smoothing in ML detection
            fix_range = 750 # range from where to consider phi values (dwd data is bad in the first bin)
        elif "dmi" in ff:
            country="dmi"
            window0=17 # number of range bins for phidp smoothing (this one is quite important!)
            winlen0=21 # size of range window (bins) for the kdp-phidp calculations
            xwin0=5 # window size (bins) for the time rolling median smoothing in ML detection
            ywin0=5 # window size (bins) for the height rolling mean smoothing in ML detection
            fix_range = 200 # range from where to consider phi values (dwd data is bad in the first bin)

        ######### Processing PHIDP
        #### fix PHIDP
        # flip PHIDP in case it is wrapping around the edges (case for turkish radars)
        if ds[X_PHI].notnull().any():
            values_center = ((ds[X_PHI]>-50)*(ds[X_PHI]<50)).sum().compute()
            values_sides = ((ds[X_PHI]>50)+(ds[X_PHI]<-50)).sum().compute()
            if values_sides > values_center:
                ds[X_PHI] = xr.where(ds[X_PHI]<=0, ds[X_PHI]+180, ds[X_PHI]-180, keep_attrs=True).compute()
        
        # filter
        phi = ds[X_PHI].where((ds[X_RHO]>=0.9) & (ds[X_DBZH]>=0) & (ds["z"]>min_height) )
        
        # phidp may be already preprocessed (turkish case), then proceed directly to masking and then vulpiani
        if "UPHIDP" not in X_PHI:
            # mask 
            phi_masked = phi.where((ds[X_RHO] >= 0.95) & (ds[X_DBZH] >= 0.))
        
        else:
            # calculate offset
            rng_offset = phi.range.diff("range").median().values * window0
            phidp_offset = phi.pipe(radarmet.phase_offset, rng=rng_offset)
            off = phidp_offset["PHIDP_OFFSET"]
            start_range = phidp_offset["start_range"]
        
            # apply offset
            phi_fix = ds[X_PHI].copy()
            off_fix = off.broadcast_like(phi_fix)
            phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off
        
            # smooth range dim
            window = window0 # window along range   <----------- this value is quite important for the quality of KDP, since phidp is very noisy
            window2 = None # window along azimuth
            phi_median = phi_fix.pipe(radarmet.xr_rolling, window, window2=window2, method='median', min_periods=round(window/2), skipna=False)
    
            # Apply additional smoothing
            gkern = radarmet.gauss_kernel(window, window)
            smooth_partial = partial(radarmet.smooth_data, kernel=gkern)
            # phiclean = radarmet.smooth_data(phi_median[0], gkern)
            phiclean = xr.apply_ufunc(smooth_partial, phi_median.compute(), 
                                      input_core_dims=[["azimuth","range"]], output_core_dims=[["azimuth","range"]],
                                      vectorize=True)
    
            # mask 
            phi_masked = phiclean.where((ds[X_RHO] >= 0.95) & (ds[X_DBZH] >= 0.))
        
        # derive KDP from PHIDP (Vulpiani)
        winlen = winlen0 # windowlen 
        
        phidp, kdp = radarmet.kdp_phidp_vulpiani(phi_masked, winlen, min_periods=winlen/2)
        
        """
        # deprecated
        # min_periods = 7 # min number of vaid bins
        kdp = radarmet.kdp_from_phidp(phi_masked, winlen, min_periods=winlen//2)
        kdp1 = kdp.interpolate_na(dim='range')
        
        # derive PHIDP from KDP (convolution method)
        winlen = winlen0
        phidp = radarmet.phidp_from_kdp(kdp1, winlen)
        """
        
        # assign new variables to dataset
        if "UPHIDP" not in X_PHI:
            assign = {
              X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs),
              "KDP_CONV": kdp.assign_attrs(KDP_attrs),
              "PHI_CONV": phidp.assign_attrs(ds[X_PHI].attrs)
              }
                
            ds = ds.assign(assign)
        
        else:
            assign = {X_PHI+"_OC_SMOOTH": phi_median.assign_attrs(ds[X_PHI].attrs),
              X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs),
              "KDP_CONV": kdp.assign_attrs(KDP_attrs),
              "PHI_CONV": phidp.assign_attrs(ds[X_PHI].attrs),
            
              X_PHI+"_OFFSET": off.assign_attrs(ds[X_PHI].attrs),
              X_PHI+"_OC": phi_fix.assign_attrs(ds[X_PHI].attrs)}
    
            ds = ds.assign(assign)
        
    else:
        print(X_PHI+" not found in the data, skipping ML detection")
    
#%% Compute QVP
    try:
        ## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
        ## Also added further filtering (TH>0, ZDR>-1)
        ds_qvp_ra = ds.where( (ds[X_RHO] > 0.7) & (ds[X_TH] > 0) & (ds[X_ZDR] > -1) ).median("azimuth", keep_attrs=True)
        ds_qvp_ra = ds_qvp_ra.assign_coords({"z": ds["z"].median("azimuth")})
        
        ds_qvp_ra = ds_qvp_ra.swap_dims({"range":"z"}) # swap range dimension for height
        
        # filter out values close to the ground
        ds_qvp_ra2 = ds_qvp_ra.where(ds_qvp_ra["z"]>min_height)
    except KeyError:
        # in case some of the variables is not present (for example in turkish data without polarimetry)
        ds_qvp_ra = ds.where(  (ds[X_TH] > 0) ).median("azimuth", keep_attrs=True)
        ds_qvp_ra = ds_qvp_ra.assign_coords({"z": ds["z"].median("azimuth")})
        
        ds_qvp_ra = ds_qvp_ra.swap_dims({"range":"z"}) # swap range dimension for height

#%% Detect melting layer
    if X_PHI in data.data_vars:
        if country=="dwd":
            moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI+"_OC": (-20, 180)}
            clowres0=True
        elif country=="dmi":
            if X_PHI+"_OC" in ds.data_vars:
                moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI+"_OC": (-20, 180)}
            else:
                moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}
            clowres0=False

        dim = 'z'
        thres = 0.02 # gradient values over thres are kept. Lower is more permissive
        xwin = xwin0 # value for the time median smoothing
        ywin = ywin0 # value for the height mean smoothing (1 for Cband)
        fmlh = 0.3
         
        ml_qvp = utils.melting_layer_qvp_X_new(ds_qvp_ra2, moments=moments, 
                 dim=dim, thres=thres, xwin=xwin, ywin=ywin, fmlh=fmlh, all_data=True, clowres=clowres0)
    
        #### Assign ML values to dataset
        
        ds = ds.assign_coords({'height_ml': ml_qvp.mlh_top})
        ds = ds.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})
        
        ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml': ml_qvp.mlh_top})
        ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})
    
        #### Giangrande refinment
        hdim = "z"
        # get data iside the currently detected ML
        cut_above = ds_qvp_ra.where(ds_qvp_ra[hdim]<ds_qvp_ra.height_ml)
        cut_above = cut_above.where(ds_qvp_ra[hdim]>ds_qvp_ra.height_ml_bottom)
        #test_above = cut_above.where((cut_above.rho >=0.7)&(cut_above.rho <0.98))
        
        # get the heights with min RHOHV
        min_height_ML = cut_above[X_RHO].idxmin(dim=hdim) 
        
        # cut the data below and above the previous value
        new_cut_below_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] > min_height_ML)
        new_cut_above_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] < min_height_ML)
        
        # Filter out values outside some RHOHV range
        new_cut_below_min_ML_filter = new_cut_below_min_ML[X_RHO].where((new_cut_below_min_ML[X_RHO]>=0.97)&(new_cut_below_min_ML[X_RHO]<=1))
        new_cut_above_min_ML_filter = new_cut_above_min_ML[X_RHO].where((new_cut_above_min_ML[X_RHO]>=0.97)&(new_cut_above_min_ML[X_RHO]<=1))            
    
    
        ######### ML TOP Giangrande refinement
        
        notnull = new_cut_below_min_ML_filter.notnull() # this replaces nan for False and the rest for True
        first_valid_height_after_ml = notnull.where(notnull).idxmax(dim=hdim) # get the first True value, i.e. first valid value
        
        ######### ML BOTTOM Giangrande refinement
        # For this one, we need to flip the coordinate so that it is actually selecting the last valid index
        notnull = new_cut_above_min_ML_filter.notnull() # this replaces nan for False and the rest for True
        last_valid_height = notnull.where(notnull).isel({hdim:slice(None, None, -1)}).idxmax(dim=hdim) # get the first True value, i.e. first valid value (flipped)
        
        
        # assign new values
        ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
        ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
        
        
        ds = ds.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
        ds = ds.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
    
#%% Attach ERA5 temperature profile
    loc = find_loc(locs, ff)
    ds_qvp_ra = utils.attach_ERA5_TEMP(ds_qvp_ra, path=loc.join(era5_dir.split("loc")))
 
#%% Fix KDP in the ML using PHIDP:
    if X_PHI in data.data_vars:    
       
        # discard possible erroneous ML values
        isotherm = -1 # isotherm for the upper limit of possible ML values
        z_isotherm = ds_qvp_ra.TEMP.isel(z=((ds_qvp_ra["TEMP"]-isotherm)**2).argmin("z").compute())["z"]
        
        ds_qvp_ra.coords["height_ml_new_gia"] = ds_qvp_ra["height_ml_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()
        ds_qvp_ra.coords["height_ml_bottom_new_gia"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()
        
        # PHIDP delta bump correction
        # get where PHIDP has nan values
        nan = np.isnan(ds[X_PHI+"_OC_MASKED"]) 
        # get PHIDP outside the ML
        phi2 = ds[X_PHI+"_OC_MASKED"].where((ds.z < ds_qvp_ra.height_ml_bottom_new_gia) | (ds.z > ds_qvp_ra.height_ml_new_gia))#.interpolate_na(dim='range',dask_gufunc_kwargs = "allow_rechunk")
        # interpolate PHIDP in ML
        phi2 = phi2.chunk(dict(range=-1)).interpolate_na(dim='range', method=interpolation_method_ML)
        # restore originally nan values
        phi2 = xr.where(nan, np.nan, phi2)
        
        # Derive KPD from the new PHIDP
        # dr = phi2.range.diff('range').median('range').values / 1000.
        # print("range res [km]:", dr)
        # winlen in gates
        # TODO: window length in m
        winlen = winlen0
        phidp_ml, kdp_ml = radarmet.kdp_phidp_vulpiani(phi2, winlen, min_periods=3)
        
        # assign to datasets
        ds = ds.assign({"KDP_ML_corrected": (["time", "azimuth", "range"], kdp_ml.fillna(ds["KDP_CONV"]).values, KDP_attrs)})
        
        #### Optional filtering:
        #ds["KDP_ML_corrected"] = ds.KDP_ML_corrected.where((ds.KDP_ML_corrected >= 0.0) & (ds.KDP_ML_corrected <= 3)) 
        
        ds = ds.assign_coords({'height': ds.z})
        
        kdp_ml_qvp = ds["KDP_ML_corrected"].where( (ds[X_RHO] > 0.7) & (ds[X_TH] > 0) & (ds[X_ZDR] > -1) ).median("azimuth", keep_attrs=True)
        kdp_ml_qvp = kdp_ml_qvp.assign_coords({"z": ds["z"].median("azimuth")})
        kdp_ml_qvp = kdp_ml_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": kdp_ml_qvp})
    
    
#%% Classification of stratiform events based on entropy
    if X_PHI in data.data_vars:    
        
        # calculate linear values for ZH and ZDR
        ds = ds.assign({"DBZH_lin": wrl.trafo.idecibel(ds[X_DBZH]), "ZDR_lin": wrl.trafo.idecibel(ds[X_ZDR]) })
        
        # calculate entropy
        Entropy = utils.Entropy_timesteps_over_azimuth_different_vars_schneller(ds, zhlin="DBZH_lin", zdrlin="ZDR_lin", rhohvnc=X_RHO, kdp="KDP_ML_corrected")
        
        # concate entropy for all variables and get the minimum value 
        strati = xr.concat((Entropy.entropy_zdrlin, Entropy.entropy_Z, Entropy.entropy_RHOHV, Entropy.entropy_KDP),"entropy")        
        min_trst_strati = strati.min("entropy")
        
        # assign to datasets
        ds["min_entropy"] = min_trst_strati
        
        min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
        min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})
        
    
                        
#%% Save dataset      
    # save file
    ds_qvp_ra.to_netcdf(savepath)

#%% If ML was detected, create a txt file for quick reference
    try:
        if ds_qvp_ra.height_ml_new_gia.notnull().any():
            with open( os.path.dirname(savepath)+'/ML_detected.txt', 'w') as f:
                f.write('')
    except:
        pass

#%% If pixels over 30 DBZH detected at some timestep in the sweep, write a txt file for reference
    try:
        valid = (ds[X_DBZH][:,:,1:]>30).sum(dim=("azimuth", "range")).compute() > ds[X_DBZH][:,:,1:].count(dim=("azimuth", "range")).compute()*0.01
        if valid.any():
            with open( os.path.dirname(savepath)+'/DBZH_over_30.txt', 'w') as f:
                f.write('')
    except:
        pass



#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")
