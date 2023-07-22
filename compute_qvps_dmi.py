#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:05:32 2023

@author: jgiles

This script computes the ML detection algorithm and entropy values for event classification,
then generates QVPs including sounding temperature values and saves to nc files.

"""

import os
os.chdir('/home/jgiles/')


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

import warnings
warnings.filterwarnings('ignore')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


#%% we are going to convert the data for every day of data (i.e. for every daily file)

# paths to files to load
# 07 is the scan number for 12 degree elevation
# path = "/home/jgiles/dwd/pulled_from_detect/*/*/2017-04-12/pro/vol5minng01/07/*allmoms*"
path = "/automount/realpep/upload/jgiles/dmi/pulled_from_detect_ank/*/*/*/ANK/*/14.0/*allmoms*"
path = "/automount/realpep/upload/jgiles/dmi/*/*/*/HTY/*/12.0/*allmoms*"
files = sorted(glob.glob(path))

# where to save the qvps
savedir = "/home/jgiles/dmi/qvps/"
# savedir = "/automount/ftp/jgiles/qvps2/"

# Use ERA5 temperature profile? If so, it does not use sounding data
era5_temp = True
era5_dir = "/automount/ags/jgiles/ERA5/hourly/hty/pressure_level_vars/"

# download sounding data?
download_sounding = False
if era5_temp: download_sounding = False

# Code for Sounding data (http://weather.uwyo.edu/upperair/sounding.html )
rs_id = "LTAU" # Close to radar site. 10393 Lindenberg close to PRO, 10868 Munich close to Turkheim. LTAU Kayseri not so close to Ankara.

# add sounding data (TEMP) to QVP?
add_sounding_to_qvp = True

# save raw sounding data?
save_raw_sounding = True
sounding_savepath = "/automount/ags/jgiles/soundings_wyoming/"

for ff in files:
    # check if the file already exists before starting
    savepath = savedir + ff.rsplit(os.sep, 1)[0].split(os.sep,6)[-1] + "/" + "qvp_"+ff.rsplit(os.sep, 1)[-1]
    if os.path.exists(savepath):
        continue

    print("Processing "+os.path.dirname(ff))
    # load data, convert to dataset, georeference
    swp = xr.open_dataset(ff) 

    try: # this might fail because of the issue with the time dimension that some files have
        swp = swp.pipe(wrl.georef.georeference_dataset)
    except ValueError:
        print("!!!! Issue with dimensions in the coordinates of the contenated files, fixing by taking time azimuth")
        for coord in ["latitude", "longitude", "altitude", "elevation"]:
            if "time" in swp[coord].dims:
                swp.coords[coord] = swp.coords[coord].median("time")
        swp = swp.pipe(wrl.georef.georeference_dataset)
        # with open(savedir+"dates_to_recompute.txt", 'a') as file:
        #     file.write(savepath.rsplit(os.sep, 5)[1]+"\n")

    ################## Before entropy calculation we need to use the melting layer detection algorithm 
    ds = swp
    interpolation_method_ML = "linear"
    
    # name the variables. Variables should be offset corrected (_OC in BoXpol) and RHOHV should be noise corrected (_NC)
    X_ZH = "DBZH"  # "DBTH"
    X_RHOHV = "RHOHV"
    X_PHIDP = "PHIDP" # "PHIDP"
    X_TH = "DBZH" # if we do not have TH
    
    # check that the variables exist in ds
    xvars = [X_ZH, X_RHOHV, X_PHIDP, X_TH]
    xvars_check = [xvar in ds.data_vars for xvar in xvars ]
    if not all( [xvar in ds.data_vars for xvar in xvars ] ):
        print( "!! Necessary variables are not in this dataset: "+ 
              " ".join([xvar for nvar,xvar in enumerate(xvars) if not xvars_check[nvar]]) +
              ". Skipping date")
        continue
    
    ######### Processing PHIDP
    #### fix PHIDP
    # filter
    phi = ds[X_PHIDP].where((ds[X_RHOHV]>=0.9) & (ds[X_ZH]>=0))
    # calculate offset
    phidp_offset = phi.pipe(radarmet.phase_offset, rng=3000)
    off = phidp_offset["PHIDP_OFFSET"]
    start_range = phidp_offset["start_range"]
    
    # apply offset
    fix_range = 750
    phi_fix = ds[X_PHIDP].copy()
    off_fix = off.broadcast_like(phi_fix)
    phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off
    
    # smooth and mask
    window = 11 # window along range   <----------- this value is quite important for the quality of KDP, since phidp is very noisy
    window2 = None # window along azimuth
    phi_median = phi_fix.pipe(radarmet.xr_rolling, window, window2=window2, method='median', skipna=True, min_periods=window//2)
    phi_masked = phi_median.where((ds[X_RHOHV] >= 0.95) & (ds[X_ZH] >= 0.))
    
    # dr = phi_masked.range.diff('range').median('range').values / 1000.
    # print("range res [km]:", dr)
    
    # derive KDP from PHIDP (convolution method)
    winlen = 31 # windowlen 
    # min_periods = 7 # min number of vaid bins
    kdp = radarmet.kdp_from_phidp(phi_masked, winlen, min_periods=winlen//2)
    kdp1 = kdp.interpolate_na(dim='range')
    
    # derive PHIDP from KDP (convolution method)
    winlen = 31
    phidp = radarmet.phidp_from_kdp(kdp1, winlen)
    
    # assign new variables to dataset
    assign = {X_PHIDP+"_OC_SMOOTH": phi_median.assign_attrs(ds[X_PHIDP].attrs),
      X_PHIDP+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHIDP].attrs),
      "KDP_CONV": kdp.assign_attrs(ds.KDP.attrs),
      "PHI_CONV": phidp.assign_attrs(ds[X_PHIDP].attrs),
    
      X_PHIDP+"_OFFSET": off.assign_attrs(ds[X_PHIDP].attrs),
      X_PHIDP+"_OC": phi_fix.assign_attrs(ds[X_PHIDP].attrs)}
    
    
    #### Compute QVP
    ## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
    ## Also added further filtering (TH>0, ZDR>-1)
    ds = ds.assign(assign)
    ds_qvp_ra = ds.where( (ds[X_RHOHV] > 0.7) & (ds[X_TH] > 0) & (ds["ZDR"] > -1) ).median("azimuth", keep_attrs=True)
    ds_qvp_ra = ds_qvp_ra.assign_coords({"z": ds["z"].median("azimuth")})
    
    ds_qvp_ra = ds_qvp_ra.swap_dims({"range":"z"}) # swap range dimension for height
    
    # filter out values close to the ground
    ds_qvp_ra2 = ds_qvp_ra.where(ds_qvp_ra["z"]>300)
    
    #### Detect melting layer
    moments={X_ZH: (10., 60.), X_RHOHV: (0.65, 1.), X_PHIDP+"_OC": (-180, 180)}
    dim = 'z'
    thres = 0.02 # gradient values over thres are kept. Lower is more permissive
    xwin = 5 # value for the time median smoothing
    ywin = 5 # value for the height mean smoothing (1 for Cband)
    fmlh = 0.3
     
    ml_qvp = utils.melting_layer_qvp_X_new(ds_qvp_ra2, moments=moments, 
             dim=dim, thres=thres, xwin=xwin, ywin=ywin, fmlh=fmlh, all_data=True, clowres=False)
    
    
    
    
    
    #### Assign ML values to dataset
    
    ds = ds.assign_coords({'height_ml': ml_qvp.mlh_top})
    ds = ds.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})
    
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml': ml_qvp.mlh_top})
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})
    
    #### Giagrande refinment
    hdim = "z"
    # get data iside the currently detected ML
    cut_above = ds_qvp_ra.where(ds_qvp_ra[hdim]<ds_qvp_ra.height_ml)
    cut_above = cut_above.where(ds_qvp_ra[hdim]>ds_qvp_ra.height_ml_bottom)
    #test_above = cut_above.where((cut_above.rho >=0.7)&(cut_above.rho <0.98))
    
    # get the heights with min RHOHV
    min_height_ML = cut_above[X_RHOHV].idxmin(dim=hdim) 
    
    # cut the data below and above the previous value
    new_cut_below_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] > min_height_ML)
    new_cut_above_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] < min_height_ML)
    
    # Filter out values outside some RHOHV range
    new_cut_below_min_ML_filter = new_cut_below_min_ML[X_RHOHV].where((new_cut_below_min_ML[X_RHOHV]>=0.97)&(new_cut_below_min_ML[X_RHOHV]<=1))
    new_cut_above_min_ML_filter = new_cut_above_min_ML[X_RHOHV].where((new_cut_above_min_ML[X_RHOHV]>=0.97)&(new_cut_above_min_ML[X_RHOHV]<=1))            
    
    
    ######### ML TOP Giangrande refinement
    
    notnull = new_cut_below_min_ML_filter.notnull() # this replaces nan for False and the rest for True
    first_valid_height_after_ml = notnull.where(notnull).idxmax(dim=hdim) # get the first True value, i.e. first valid value
    
    ######### ML BOTTOM Giangrande refinement
    # For this one, we need to flip the coordinate so that it is actually selecting the last valid index
    notnull = new_cut_above_min_ML_filter.notnull() # this replaces nan for False and the rest for True
    last_valid_height = notnull.where(notnull).isel({hdim:slice(None, None, -1)}).idxmax(dim=hdim) # get the first True value, i.e. first valid value (flipped)
    
    
    ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
    ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
    
    
    ds = ds.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
    ds = ds.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))
    
    
    #### Fix KDP in the ML using PHIDP:
    
    ds_qvp_ra3 = ds_qvp_ra.where(ds_qvp_ra.height_ml_new_gia<1000)
    
    # get where PHIDP has nan values
    nan = np.isnan(ds[X_PHIDP+"_OC_MASKED"]) 
    # get PHIDP outside the ML
    phi2 = ds[X_PHIDP+"_OC_MASKED"].where((ds.z < ds_qvp_ra3.height_ml_bottom_new_gia) | (ds.z > ds_qvp_ra3.height_ml_new_gia))#.interpolate_na(dim='range',dask_gufunc_kwargs = "allow_rechunk")
    # interpolate PHIDP in ML
    phi2 = phi2.interpolate_na(dim='range', method=interpolation_method_ML)
    # restore originally nan values
    phi2 = xr.where(nan, np.nan, phi2)
    
    # Derive KPD from the new PHIDP
    # dr = phi2.range.diff('range').median('range').values / 1000.
    # print("range res [km]:", dr)
    # winlen in gates
    # TODO: window length in m
    winlen = 31
    min_periods = 3
    kdp_ml = radarmet.kdp_from_phidp(phi2, winlen, min_periods=3)
    
    # assign to datasets
    ds = ds.assign({"KDP_ML_corrected": (["time", "azimuth", "range"], kdp_ml.values, ds_qvp_ra3.KDP.attrs)})
    
    #### Optional filtering:
    ds["KDP_ML_corrected"] = ds.KDP_ML_corrected.where((ds.KDP_ML_corrected >= 0.0) & (ds.KDP_ML_corrected <= 3)) 
    
    ds = ds.assign_coords({'height': ds.z})
    
    kdp_ml_qvp = ds["KDP_ML_corrected"].median("azimuth", keep_attrs=True)
    kdp_ml_qvp = kdp_ml_qvp.assign_coords({"z": ds["z"].median("azimuth")})
    kdp_ml_qvp = kdp_ml_qvp.swap_dims({"range":"z"}) # swap range dimension for height
    ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": kdp_ml_qvp})
    
    
    #### Classification of stratiform events based on entropy
    
    # calculate linear values for ZH and ZDR
    ds = ds.assign({"DBZH_lin": wrl.trafo.idecibel(ds["DBZH"]), "ZDR_lin": wrl.trafo.idecibel(ds["ZDR"]) })
    
    # calculate entropy
    Entropy = utils.Entropy_timesteps_over_azimuth_different_vars_schneller(ds, zhlin="DBZH_lin", zdrlin="ZDR_lin", rhohvnc=X_RHOHV, kdp="KDP_ML_corrected")
    
    # concate entropy for all variables and get the minimum value 
    strati = xr.concat((Entropy.entropy_zdrlin, Entropy.entropy_Z, Entropy.entropy_RHOHV, Entropy.entropy_KDP),"entropy")        
    min_trst_strati = strati.min("entropy")
    
    # assign to datasets
    ds["min_entropy"] = min_trst_strati
    
    min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
    min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
    ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})
    
    
    #### Download temperature profile from sounding
    if download_sounding:
        startdt0 = datetime.datetime.utcfromtimestamp(int(swp.time[0].values)/1e9).date()
        enddt0 = datetime.datetime.utcfromtimestamp(int(swp.time[-1].values)/1e9).date() + datetime.timedelta(hours=24)
        
        # transform the dates to datetimes
        startdt = datetime.datetime.fromordinal(startdt0.toordinal())
        enddt = datetime.datetime.fromordinal(enddt0.toordinal())
        
        def date_range_list(start_date, end_date):
            # Return list of datetime.date objects (inclusive) between start_date and end_date (inclusive).
            date_list = []
            curr_date = start_date
            while curr_date <= end_date:
                date_list.append(curr_date)
                curr_date += datetime.timedelta(hours=12)
            return date_list
        
        dttimes = date_range_list(startdt, enddt)
        
        
        # Create a list of xarray datasets with sounding data to then concatenate
        print("Getting temperature from sounding data")
        soundings = list()
        break_day = False # flag for stop trying for this day and continuing with next if the next step fails
        for tt in dttimes:
            if break_day:
                break
            # getting the sounding data may fail, so we try ntries times until we get it
            ntries = 100
            for nt in range(ntries):
                if break_day:
                    break
                try:
                    # Load Sounding data (http://weather.uwyo.edu/upperair/sounding.html)
                    # rs_id = 10393 # Close to radar site
                    rs_data, rs_meta = wrl.io.get_radiosonde(rs_id, tt, cols=(0,1,2))
                    
                    #rs_array = np.array( list( map(list, rs_data) ) )
                
                    # Extract temperature and height
                    stemp = rs_data['TEMP']
                    sheight = rs_data['HGHT']
                    # remove nans
                    idx = np.isfinite(stemp)
                    stemp = stemp[idx]
                    sheight = sheight[idx]
                    
                    # check that height is monotonically increasing and remove non compliant values
                    valid = np.concatenate([np.array([True]), np.sign(np.diff(sheight))>0]) # we assume the first value is valid
                    stemp = stemp[valid]
                    sheight = sheight[valid]
                    
                    stemp_da = xr.DataArray(data=stemp, dims=["height"],
                                            coords=dict(height=(["height"], sheight),       
                                                       ),
                                            attrs=dict(description="Temperature.",
                                                       units="degC",
                                                      ),
                                           )
                    
                    soundings.append(stemp_da)
                    
                    
                    if save_raw_sounding:
                        spres = rs_data['PRES']
                        spres = spres[idx] # remove nans
                        spres = spres[valid] # check monotony and remove weird values
                        
                        sounding_raw = xr.Dataset(data_vars=dict(
                                                    TEMP=(["HGHT"], stemp),
                                                    PRES=(["HGHT"], spres),
                                                    ),
                                                coords=dict(HGHT=(["HGHT"], sheight),       
                                                           ),
                                                attrs=rs_meta,
                                               )
                        # fix attrs not admitted by nc standard
                        sounding_raw.attrs["Observation time"] = str(sounding_raw.attrs["Observation time"])
                        sounding_raw.attrs["quantity"] = [varname+" "+sounding_raw.attrs["quantity"][varname] for varname in sounding_raw.attrs["quantity"].keys()]
                        
                        # make dir and save
                        os.makedirs(sounding_savepath+str(rs_id)+"/"+str(tt.year)+"/", exist_ok=True)
                        sounding_raw.to_netcdf(sounding_savepath+str(rs_id)+"/"+str(tt.year)+"/"+str(tt.date())+"T"+str(tt.hour)+".nc")
                    
                    break
                except ValueError:
                    print("No sounding data for "+str(tt)+". Attempting to fill with adjacent values")
                    # create an array of nan to be later filled with adjacent values (if there are any)
                    stemp_da = xr.DataArray(data=np.full(200, np.nan), dims=["height"],
                                            coords=dict(height=(["height"], np.arange(0,20000,100)),       
                                                       ),
                                            attrs=dict(description="Temperature.",
                                                       units="degC",
                                                      ),
                                           )
                    
                    soundings.append(stemp_da)
                    break
                
                except TimeoutError:
                    print(".. Attempt "+str(nt+1)+" failed, retrying")
                    continue
                except Exception as e:
                    if "Service Unavailable" in str(e) and nt<ntries:
                        print(".. Attempt "+str(nt+1)+" failed, retrying")
                        continue
                    elif "Service Unavailable" in str(e) and nt>=ntries:
                        print(".. All attempts failed, skipping this day of data")
                        break_day = True
                        continue                     
                    else:
                        raise(e)
                        
        if break_day:
            # if this step failed, continue with next day of data
            continue
                    
        # Create time dimension and concatenate
        timedim = xr.DataArray( pd.to_datetime(dttimes), [("time", pd.to_datetime(dttimes))] )
        try: # this might fail because of the issue with the time dimension in elevations that some files have
            temperatures = xr.concat(soundings, timedim)
            
            # Interpolate to higher resolution
            hmax = 50000.
            ht = np.arange(0., hmax, 50)
            itemp_da = temperatures.interpolate_na("height").interp({"height": ht})
            
            # Fix Temperature below first measurement and above last one
            itemp_da = itemp_da.bfill(dim="height")
            itemp_da = itemp_da.ffill(dim="height")
            
            # Attempt to fill any missing timestep with adjacent data
            itemp_da = itemp_da.ffill(dim="time")
            itemp_da = itemp_da.bfill(dim="time")
            
            # Interpolate to dataset height and time, then add to dataset
            def merge_radar_profile(rds, cds):
                # cds = cds.interp({'height': rds.z}, method='linear')
                cds = cds.interp({'height': rds.z}, method='linear')
                cds = cds.interp({"time": rds.time}, method="linear")
                rds = rds.assign({"TEMP": cds})
                rds.TEMP.attrs["source"]="soundings Wyoming"
                return rds
            
            ds_qvp_ra = ds_qvp_ra.pipe(merge_radar_profile, itemp_da)
            
            ds_qvp_ra.coords["TEMP"] = ds_qvp_ra["TEMP"] # move TEMP from variable to coordinate
            ds_qvp_ra.coords["TEMP"].attrs["source"] = "Soundings Wyoming ID: "+str(rs_id) # add source info

        except ValueError:
            print("!!!! ERROR: some issue when concatenating sounding data, ignoring date")
            with open(savedir+"/dates_to_recompute.txt", 'a') as file:
                file.write(savepath.rsplit(os.sep, 5)[1]+"\n")

    #### Use temperature from ERA5
    if era5_temp:
        # get times of the radar files
        startdt0 = datetime.datetime.utcfromtimestamp(int(swp.time[0].values)/1e9).date()
        enddt0 = datetime.datetime.utcfromtimestamp(int(swp.time[-1].values)/1e9).date() + datetime.timedelta(hours=24)
        
        # transform the dates to datetimes
        startdt = datetime.datetime.fromordinal(startdt0.toordinal())
        enddt = datetime.datetime.fromordinal(enddt0.toordinal())
        
        # open ERA5 files
        era5_t = xr.open_mfdataset(reversed(glob.glob(era5_dir+"temperature/*"+str(startdt.year)+"*")), concat_dim="lvl", combine="nested")
        era5_g = xr.open_mfdataset(reversed(glob.glob(era5_dir+"geopotential/*"+str(startdt.year)+"*")), concat_dim="lvl", combine="nested")
        
        # add altitude coord to temperature data
        earth_r = wrl.georef.projection.get_earth_radius(swp.latitude.values)
        gravity = 9.80665
        
        era5_t.coords["height"] = (earth_r*(era5_g.z/gravity)/(earth_r - era5_g.z/gravity)).compute()
        
        # Create time dimension and concatenate
        try: # this might fail because of the issue with the time dimension in elevations that some files have
            dtslice0 = startdt.strftime('%Y-%m-%d %H')
            dtslice1 = enddt.strftime('%Y-%m-%d %H')
            temperatures = era5_t["t"].loc[{"time":slice(dtslice0, dtslice1)}].isel({"latitude":0, "longitude":0})
            
            # Interpolate to higher resolution
            hmax = 50000.
            ht = np.arange(0., hmax, 50)
            
            def interp_to_ht(ds):
                ds = ds.swap_dims({"lvl":"height"})
                return ds.interp({"height": ht})
            
            results = []
            
            with Pool() as P:
                results = P.map( interp_to_ht, [temperatures[:,tt] for tt in range(len(temperatures.time)) ] )
            
            itemp_da = xr.concat(results, "time")
            
            # Fix Temperature below first measurement and above last one
            itemp_da = itemp_da.bfill(dim="height")
            itemp_da = itemp_da.ffill(dim="height")
            
            # Attempt to fill any missing timestep with adjacent data
            itemp_da = itemp_da.ffill(dim="time")
            itemp_da = itemp_da.bfill(dim="time")
            
            # Interpolate to dataset height and time, then add to dataset
            def merge_radar_profile(rds, cds):
                # cds = cds.interp({'height': rds.z}, method='linear')
                cds = cds.interp({'height': rds.z}, method='linear')
                cds = cds.interp({"time": rds.time}, method="linear")
                rds = rds.assign({"TEMP": cds})
                rds.TEMP.attrs["source"]="ERA5"
                return rds
            
            ds_qvp_ra = ds_qvp_ra.pipe(merge_radar_profile, itemp_da)
            
            ds_qvp_ra.coords["TEMP"] = ds_qvp_ra["TEMP"] # move TEMP from variable to coordinate

        except ValueError:
            print("!!!! ERROR: some issue when concatenating ERA5 data, ignoring date")
            with open(savedir+"/dates_to_recompute.txt", 'a') as file:
                file.write(savepath.rsplit(os.sep, 5)[1]+"\n")
        
            
    #### Save dataset
    print("Saving file")
    
    # create the directory if it does not exist
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    
    # save file
    ds_qvp_ra.to_netcdf(savepath)
