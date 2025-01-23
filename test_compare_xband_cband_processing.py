#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:42:27 2025

@author: jgiles

Test full processing on C-band vs X-band data
"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER?)

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
import time

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

#%% Load Data

path_xband = "/automount/radar-archiv/archiv/dkrz/output/2017-07/BoXPol_UniBonn_Radar_5min_level2_v20201127_Enigma4_18p0deg_20170725.nc"
path_cband = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/ess/vol5minng01/07/ras07-vol5minng01_sweeph5allm_any_07-20170725-ess-10410-hd5"
path_cband = "/automount/realpep/upload/jgiles/dwd/2021/2021-07/2021-07-14/ess/vol5minng01/07/ras07-vol5minng01_sweeph5allm_any_07-20210714-ess-10410-hd5"

data_xband = xr.open_dataset(path_xband)
data_cband = utils.load_dwd_preprocessed(path_cband)

# change lowercase xband vars to coords
xbandcoords = []
for dv in data_xband.data_vars.keys():
    if dv.islower():
        xbandcoords.append(dv)

data_xband = data_xband.set_coords(xbandcoords)

# georeference
data_xband = wrl.georef.georeference(data_xband)
data_cband = wrl.georef.georeference(data_cband)

#%% Apply DBZH offset to Xband
# From Veli's paper https://www.nature.com/articles/s41597-022-01656-0 Table 3

data_xband["DBZH"] = data_xband["DBZH"] -1.28

#%% Put in a dict

data = {
        "cband": data_cband,
        # "xband": data_xband
        }

qvps = {}

#%% RHOHV NC
# For Xband it is better to write this to a file and reload because it takes a lot of memory
temppath = "/automount/realpep/upload/jgiles/BoxPol/rhohv_nc/2017/2017-07/2017-07-25/18.0/rhohv_nc_2percent.nc"

for dn in data.keys():
    X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(data[dn])

    rho_nc = utils.calculate_noise_level(data[dn][X_DBZH], data[dn][X_RHO], noise=(-45, -15, 1))

    # lets do a linear fit for every noise level
    fits=[]
    for nn,rhon in enumerate(rho_nc[0]):
        merged = xr.merge(rhon)
        rhonc_snrh = xr.DataArray(merged.RHOHV_NC.values.flatten(), coords={"SNRH":merged.SNRH.values.flatten()})
        try:
            fits.append(float(rhonc_snrh.where((0<rhonc_snrh.SNRH)&(rhonc_snrh.SNRH<20)&(rhonc_snrh>0.7)).polyfit("SNRH", deg=1, skipna=True).polyfit_coefficients[0].values))
        except:
            # if it does not work, just attach nan
            fits.append(np.nan)

    # checking which fit has the slope closest to zero
    try:
        bci = np.nanargmin(np.abs(np.array(fits)))
    except ValueError:
        # if all slopes are nan, no correction is good enough, abort
        print("Could not calculate noise correction (possibly due to not enough data points). Aborting...")
        continue

    # get the best noise correction level according to bci
    ncl = np.arange(-45, -15, 1)[bci]

    # # get the "best" noise correction level (acoording to the min std)
    # ncl = rho_nc[-1]

    # # get index of the best correction
    # bci = np.array(rho_nc[-2]).argmin()

    # merge into a single array
    rho_nc_out = xr.merge(rho_nc[0][bci])

    # add noise correction level as attribute
    rho_nc_out.attrs["noise correction level"]=ncl

    # Just in case, calculate again for a NCL slightly lower (2%), in case the automatically-selected one is too strong
    rho_nc2 = utils.noise_correction2(data[dn][X_DBZH], data[dn][X_RHO], ncl*1.02)

    # make a new array as before
    rho_nc_out2 = xr.merge(rho_nc2)
    rho_nc_out2.attrs["noise correction level"]=ncl*1.02

    if dn == "xband":
        if not os.path.exists(os.path.dirname(temppath)):
            os.makedirs(os.path.dirname(temppath))

        rho_nc_out2["RHOHV_NC"].encoding = data[dn][X_RHO].encoding

        dwd_rhohv_nc_reference_file = "/automount/realpep/upload/jgiles/reference_files/reference_dwd_rhohv_nc_file/ras07-90gradstarng01_sweeph5onem_rhohv_nc_00-2015010100042300-pro-10392-hd5"
        rho_nc_dwd = xr.open_dataset(dwd_rhohv_nc_reference_file, engine="netcdf4")
        rho_nc_out2["SNRH"].encoding = rho_nc_dwd["SNRH"].encoding

        rho_nc_out2.to_netcdf(temppath)

        data[dn] = utils.load_corrected_RHOHV(data[dn], temppath)

        del(rho_nc_out2)
        del(rho_nc_out)
        del(rho_nc)
        del(fits)

    else:
        data[dn] = data[dn].assign(rho_nc_out2.copy())

#%% ZDR OC

for dn in data.keys():
    X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(data[dn])
    if X_RHO+"_NC" in data[dn].data_vars:
        X_RHO = X_RHO+"_NC"

    #### Attach ERA5 temperature profile
    loc = utils.find_loc(utils.locs, path_cband)
    data[dn] = utils.attach_ERA5_TEMP(data[dn], path=loc.join(utils.era5_dir.split("loc")))

    if dn == "xband":
        zdr_offset = utils.zhzdr_lr_consistency(data[dn], zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=3, min_h=300, timemode="step")
        minbins = int(1000 / data[dn]["range"].diff("range").median())
        zdr_offset_qvp = utils.zdr_offset_detection_qvps(data[dn], zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=3, azmed=False,
                                                    min_h=300, timemode="step", minbins=minbins).compute()
        final_zdr_oc = xr.where(zdr_offset["ZDR_offset_datacount"] > zdr_offset_qvp["ZDR_offset_datacount"], zdr_offset["ZDR_offset"], zdr_offset_qvp["ZDR_offset"]).copy()
        final_zdr_oc = final_zdr_oc.ffill("time")

        data[dn][X_ZDR+"_OC"] = data[dn][X_ZDR] - final_zdr_oc
        data[dn]["ZDR_offset"] = final_zdr_oc

    else:
        zdr_offset = utils.zhzdr_lr_consistency(data[dn], zdr=X_ZDR, dbzh=X_DBZH, rhohv=X_RHO, mlbottom=3, min_h=300, timemode="all")

        data[dn][X_ZDR+"_OC"] = data[dn][X_ZDR] - zdr_offset["ZDR_offset"].values
        data[dn]["ZDR_offset"] = zdr_offset.ZDR_offset

#%% Processing

for dn in data.keys():
    clowres0 = False
    if dn=="cband":
        if data[dn].range.diff("range").mean() <750 :
            phase_proc_params = utils.get_phase_proc_params("dwd-hres/vol5minng01")
        else:
            phase_proc_params = utils.get_phase_proc_params(path_cband) # get default phase processing parameters
            clowres0 = True
    elif dn=="xband":
        phase_proc_params = utils.get_phase_proc_params("dwd-hres/vol5minng01") # get default phase processing parameters
    else:
        raise KeyError("Check radar phase processing parameters")

    X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(data[dn])

    if X_RHO+"_NC" in data[dn].data_vars:
        X_RHO = X_RHO+"_NC"

    if X_ZDR+"_OC" in data[dn].data_vars:
        X_ZDR = X_ZDR+"_OC"


    min_height = utils.min_hgts["default"] + data[dn]["altitude"].values

    min_range = utils.min_rngs["default"]


    # Check that PHIDP is in data, otherwise skip ML detection
    if X_PHI in data[dn].data_vars:
        # Set parameters according to data

        # for param_name in phase_proc_params[country].keys():
        #     globals()[param_name] = phase_proc_params[param_name]
        window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia = phase_proc_params.values() # explicit alternative

        data[dn] = utils.phidp_processing(data[dn], X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                             rng=rng, azmedian=azmedian, tolerance=(0,5), clean_invalid=False, fillna=False)

        phi_masked = data[dn][X_PHI+"_OC_SMOOTH"].where((data[dn][X_RHO] >= 0.9) * (data[dn][X_DBZH] >= 0.) * (data[dn]["range"]>min_range) )

        # Assign phi_masked
        assign = { X_PHI+"_OC_MASKED": phi_masked.copy().assign_attrs(data[dn][X_PHI].attrs) }
        data[dn] = data[dn].assign(assign)

        # derive KDP from PHIDP (Vulpiani)

        data[dn] = utils.kdp_phidp_vulpiani(data[dn], winlen0, X_PHI+"_OC_MASKED", min_periods=winlen0//2+1)

        X_PHI = X_PHI+"_OC" # continue using offset corrected PHI

    else:
        print(X_PHI+" not found in the data, skipping ML detection")

    #### Compute QVP or RD-QVP

    print("Computing QVP")
    qvps[dn] = utils.compute_qvp(data[dn], min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10,"SNRHC":10, "SQIH":0.5} )

    # filter out values close to the ground
    qvps[dn] = qvps[dn].where(qvps[dn]["z"]>min_height)

    #### Detect melting layer

    if X_PHI in data[dn].data_vars:
        # Define thresholds
        moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}

        # Calculate ML
        qvps[dn] = utils.melting_layer_qvp_X_new(qvps[dn], moments=moments, dim="z", fmlh=0.3,
                 xwin=xwin0, ywin=ywin0, min_h=min_height, rhohv_thresh_gia=rhohv_thresh_gia, all_data=True, clowres=clowres0)

        # Assign ML values to dataset

        data[dn] = data[dn].assign_coords({'height_ml': qvps[dn].height_ml})
        data[dn] = data[dn].assign_coords({'height_ml_bottom': qvps[dn].height_ml_bottom})

        data[dn] = data[dn].assign_coords({'height_ml_new_gia': qvps[dn].height_ml_new_gia})
        data[dn] = data[dn].assign_coords({'height_ml_bottom_new_gia': qvps[dn].height_ml_bottom_new_gia})

    #### Attach ERA5 temperature profile
    loc = utils.find_loc(utils.locs, path_cband)
    qvps[dn] = utils.attach_ERA5_TEMP(qvps[dn], path=loc.join(utils.era5_dir.split("loc")))
    # data[dn] = utils.attach_ERA5_TEMP(data[dn], path=loc.join(utils.era5_dir.split("loc")))

    #### Discard possible erroneous ML values
    if "height_ml_new_gia" in qvps[dn]:
        ## First, filter out ML heights that are too high (above selected isotherm)
        isotherm = -1 # isotherm for the upper limit of possible ML values
        z_isotherm = qvps[dn].TEMP.isel(z=((qvps[dn]["TEMP"]-isotherm)**2).argmin("z").compute())["z"]

        qvps[dn].coords["height_ml_new_gia"] = qvps[dn]["height_ml_new_gia"].where(qvps[dn]["height_ml_new_gia"]<=z_isotherm.values).compute()
        qvps[dn].coords["height_ml_bottom_new_gia"] = qvps[dn]["height_ml_bottom_new_gia"].where(qvps[dn]["height_ml_new_gia"]<=z_isotherm.values).compute()

        # Then, check that ML top is over ML bottom
        cond_top_over_bottom = qvps[dn].coords["height_ml_new_gia"] > qvps[dn].coords["height_ml_bottom_new_gia"]

        # Assign final values
        qvps[dn].coords["height_ml_new_gia"] = qvps[dn]["height_ml_new_gia"].where(cond_top_over_bottom).compute()
        qvps[dn].coords["height_ml_bottom_new_gia"] = qvps[dn]["height_ml_bottom_new_gia"].where(cond_top_over_bottom).compute()

        data[dn] = data[dn].assign_coords({'height_ml_new_gia': qvps[dn].height_ml_new_gia.where(cond_top_over_bottom)})
        data[dn] = data[dn].assign_coords({'height_ml_bottom_new_gia': qvps[dn].height_ml_bottom_new_gia.where(cond_top_over_bottom)})


    #### Fix KDP in the ML using PHIDP:
    if X_PHI in data[dn].data_vars:
        data[dn] = utils.KDP_ML_correction(data[dn], X_PHI+"_MASKED", winlen=winlen0, min_periods=winlen0//2+1)

        qvps[dn] = qvps[dn].assign({"KDP_ML_corrected": utils.compute_qvp(data[dn], min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})["KDP_ML_corrected"]})

    #### Classification of stratiform events based on entropy
    if X_PHI in data[dn].data_vars:

        # calculate linear values for ZH and ZDR
        data[dn] = data[dn].assign({"DBZH_lin": wrl.trafo.idecibel(data[dn][X_DBZH]), "ZDR_lin": wrl.trafo.idecibel(data[dn][X_ZDR]) })

        # calculate entropy (Here Tobi also filters out the KDP<0.01)
        Entropy = utils.calculate_pseudo_entropy(data[dn].where(data[dn][X_DBZH]>0), dim='azimuth', var_names=["DBZH_lin", "ZDR_lin", X_RHO, "KDP_ML_corrected"], n_lowest=30)

        # concate entropy for all variables and get the minimum value
        strati = xr.concat((Entropy.entropy_DBZH_lin, Entropy.entropy_ZDR_lin, Entropy["entropy_"+X_RHO], Entropy.entropy_KDP_ML_corrected),"entropy")
        min_trst_strati = strati.min("entropy")

        # assign to datasets
        data[dn]["min_entropy"] = min_trst_strati

        min_trst_strati_qvp = min_trst_strati.assign_coords({"z": data[dn]["z"].median("azimuth")})
        min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        qvps[dn] = qvps[dn].assign({"min_entropy": min_trst_strati_qvp})


#%% Filters (conditions for stratiform)
qvps_strat = {}
qvps_strat_relaxed = {}
qvps_strat_fil = {}
qvps_strat_relaxed_fil = {}
retrievals = {}

for dn in data.keys():

    start_time = time.time()
    print("Filtering stratiform conditions...")

    X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(qvps[dn])
    X_KDP = "KDP_ML_corrected"

    if X_RHO+"_NC" in qvps[dn].data_vars:
        X_RHO = X_RHO+"_NC"

    if X_ZDR+"_OC" in qvps[dn].data_vars:
        X_ZDR = X_ZDR+"_OC"

    # Check that RHOHV_NC is actually better (less std) than RHOHV, otherwise just use RHOHV, on a per-day basis
    std_margin = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_margin), otherwise use RHOHV
    min_rho = 0.6 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results

    if "_NC" in X_RHO:
        # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
        # if that is the case we take it that the correction did not work well so we won't use it
        cond_rhohv = (
                        qvps[dn][X_RHO].where(qvps[dn][X_RHO]>min_rho).resample({"time":"D"}).std(dim=("time", "z")) < \
                        qvps[dn]["RHOHV"].where(qvps[dn]["RHOHV"]>min_rho).resample({"time":"D"}).std(dim=("time", "z"))*(1+std_margin)
                        ).compute()

        # create an xarray.Dataarray with the valid timesteps
        valid_dates = cond_rhohv.where(cond_rhohv, drop=True).time.dt.date
        valid_datetimes = [date.values in valid_dates for date in qvps[dn].time.dt.date]
        valid_datetimes_xr = xr.DataArray(valid_datetimes, coords={"time": qvps[dn]["time"]})

        # Redefine RHOHV_NC: keep it in the valid datetimes, put RHOHV in the rest
        qvps[dn][X_RHO] = qvps[dn][X_RHO].where(valid_datetimes_xr, qvps[dn]["RHOHV"])


    # Conditions to clean ML height values
    max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
    max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
    time_window = 5 # set timestep window for the std computation (centered)
    min_period = 3 # set minimum number of valid ML values in the window (centered)

    cond_ML_bottom_change = abs(qvps[dn]["height_ml_bottom_new_gia"].diff("time").compute())<max_change
    cond_ML_bottom_std = qvps[dn]["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
    # cond_ML_bottom_minlen = qvps[dn]["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

    cond_ML_top_change = abs(qvps[dn]["height_ml_new_gia"].diff("time").compute())<max_change
    cond_ML_top_std = qvps[dn]["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
    # cond_ML_top_minlen = qvps[dn]["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

    allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

    # Filter only fully stratiform pixels (min entropy >= 0.8 and ML detected)
    qvps_strat[dn] = qvps[dn].where( (qvps[dn]["min_entropy"]>=0.8).compute() & allcond, drop=True)
    # Relaxed alternative: Filter qvps with at least 50% of stratiform pixels (min entropy >= 0.8 and ML detected)
    qvps_strat_relaxed[dn] = qvps[dn].where( ( (qvps[dn]["min_entropy"]>=0.8).sum("z").compute() >= qvps[dn][X_DBZH].count("z").compute()/2 ) & allcond, drop=True)

    # Filter out non relevant values
    qvps_strat_fil[dn] = qvps_strat[dn].where((qvps_strat[dn][X_TH] > -10 )&
                                      (qvps_strat[dn][X_KDP] > -0.1)& # here Tobi filters > 0.01
                                      (qvps_strat[dn][X_KDP] < 3)&
                                      (qvps_strat[dn][X_RHO] > 0.7)&
                                      (qvps_strat[dn][X_ZDR] > -1) &
                                      (qvps_strat[dn][X_ZDR] < 3))

    qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed[dn].where((qvps_strat_relaxed[dn][X_TH] > -10 )&
                                      (qvps_strat_relaxed[dn][X_KDP] > -0.1)&
                                      (qvps_strat_relaxed[dn][X_KDP] < 3)&
                                      (qvps_strat_relaxed[dn][X_RHO] > 0.7)&
                                      (qvps_strat_relaxed[dn][X_ZDR] > -1) &
                                      (qvps_strat_relaxed[dn][X_ZDR] < 3))

    try:
        qvps_strat_fil[dn] = qvps_strat_fil[dn].where(qvps_strat_fil[dn]["SNRHC"]>10)
        qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed_fil[dn].where(qvps_strat_relaxed_fil[dn]["SNRHC"]>10)
    except KeyError:
        qvps_strat_fil[dn] = qvps_strat_fil[dn].where(qvps_strat_fil[dn]["SNRH"]>10)
        qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed_fil[dn].where(qvps_strat_relaxed_fil[dn]["SNRH"]>10)
    except:
        print("Could not filter out low SNR")

    total_time = time.time() - start_time
    print(f"took {total_time/60:.2f} minutes.")

    #### Calculate retreivals
    # We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil
    start_time = time.time()
    print("Calculating microphysical retrievals...")

    # to check the wavelength of each radar, in cm for DWD, in 1/100 cm for DMI ()
    # filewl = ""
    # xr.open_dataset(filewl, group="how") # DWD
    # file1 = realpep_path+"/upload/jgiles/dmi_raw/acq/OLDDATA/uza/RADAR/2015/01/01/ANK/RAW/ANK150101000008.RAW6M00"
    # xd.io.backends.iris.IrisRawFile(file1, loaddata=False).ingest_header["task_configuration"]["task_misc_info"]["wavelength"]

    Lambda = 53.1 # radar wavelength in mm (pro: 53.138, ANK: 53.1, AFY: 53.3, GZT: 53.3, HTY: 53.3, SVS:53.3)

    if dn == "xband":
        Lambda = 32

    # We will put the final retrievals in a dict
    try: # check if exists, if not, create it
        retrievals[dn]
    except NameError:
        retrievals[dn] = {}
    except KeyError:
        retrievals[dn] = {}

    for stratname, stratqvp in [("stratiform", qvps_strat_fil[dn].copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil[dn].copy())]:
        print("   ... for "+stratname)

        retrievals[dn][stratname] = {}

        # LWC
        lwc_zh_zdr = 10**(0.058*stratqvp[X_DBZH] - 0.118*stratqvp[X_ZDR] - 2.36) # Reimann et al 2021 eq 3.7 (adjusted for Germany)
        lwc_zh_zdr2 = 1.38*10**(-3) *10**(0.1*stratqvp[X_DBZH] - 2.43*stratqvp[X_ZDR] + 1.12*stratqvp[X_ZDR]**2 - 0.176*stratqvp[X_ZDR]**3 ) # used in S band, Ryzhkov 2022 PROM presentation https://www2.meteo.uni-bonn.de/spp2115/lib/exe/fetch.php?media=internal:uploads:all_hands_schneeferner_july2022:ryzhkov.pdf
        lwc_kdp = 10**(0.568*np.log10(stratqvp[X_KDP]) + 0.06) # Reimann et al 2021(adjusted for Germany)

        # IWC (Collected from Blanke et al 2023)
        iwc_zh_t = 10**(0.06 * stratqvp[X_DBZH] - 0.0197*stratqvp["TEMP"] - 1.7) # empirical from Hogan et al 2006 Table 2

        iwc_zdr_zh_kdp = xr.where(stratqvp[X_ZDR]>=0.4, # Carlin et al 2021 eqs 4b and 5b
                                  4*10**(-3)*( stratqvp[X_KDP]*Lambda/( 1-wrl.trafo.idecibel(stratqvp[X_ZDR])**-1 ) ),
                                  0.033 * ( stratqvp[X_KDP]*Lambda )**0.67 * wrl.trafo.idecibel(stratqvp[X_DBZH])**0.33 )

        # Dm (ice collected from Blanke et al 2023)
        Dm_ice_zh = 1.055*wrl.trafo.idecibel(stratqvp[X_DBZH])**0.271 # Matrosov et al. (2019) Fig 10 (S band)
        Dm_ice_zh_kdp = 0.67*( wrl.trafo.idecibel(stratqvp[X_DBZH])/(stratqvp[X_KDP]*Lambda) )**(1/3) # Ryzhkov and Zrnic (2019). Idk exactly where does the 0.67 approximation comes from, Blanke et al. 2023 eq 10 and Carlin et al 2021 eq 5a cite Bukovčić et al. (2018, 2020) but those two references do not show this formula.
        Dm_ice_zdp_kdp = -0.1 + 2*( (wrl.trafo.idecibel(stratqvp[X_DBZH])*(1-wrl.trafo.idecibel(stratqvp[X_ZDR])**-1 ) ) / (stratqvp[X_KDP]*Lambda) )**(1/2) # Ryzhkov and Zrnic (2019). Zdp = Z(1-ZDR**-1) from Carlin et al 2021

        Dm_rain_zdr = 0.3015*stratqvp[X_ZDR]**3 - 1.2087*stratqvp[X_ZDR]**2 + 1.9068*stratqvp[X_ZDR] + 0.5090 # (for rain but tuned for Germany X-band, JuYu Chen, Zdr in dB, Dm in mm)

        D0_rain_zdr2 = 0.171*stratqvp[X_ZDR]**3 - 0.725*stratqvp[X_ZDR]**2 + 1.48*stratqvp[X_ZDR] + 0.717 # (D0 from Hu and Ryzhkov 2022, used in S band data but could work for C band) [mm]
        D0_rain_zdr3 = xr.where(stratqvp[X_ZDR]<1.25, # D0 from Bringi et al 2009 (C-band) eq. 1 [mm]
                                0.0203*stratqvp[X_ZDR]**4 - 0.1488*stratqvp[X_ZDR]**3 + 0.2209*stratqvp[X_ZDR]**2 + 0.5571*stratqvp[X_ZDR] + 0.801,
                                0.0355*stratqvp[X_ZDR]**3 - 0.3021*stratqvp[X_ZDR]**2 + 1.0556*stratqvp[X_ZDR] + 0.6844
                                )
        mu = 0
        Dm_rain_zdr2 = D0_rain_zdr2 * (4+mu)/(3.67+mu) # conversion from D0 to Dm according to eq 4 of Hu and Ryzhkov 2022.
        Dm_rain_zdr3 = D0_rain_zdr3 * (4+mu)/(3.67+mu)

        # log(Nt)
        Nt_ice_zh_iwc = (3.39 + 2*np.log10(iwc_zh_t) - 0.1*stratqvp[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2 = (3.69 + 2*np.log10(iwc_zh_t) - 0.1*stratqvp[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_ice_zh_iwc_kdp = (3.39 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*stratqvp[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2_kdp = (3.69 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*stratqvp[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_rain_zh_zdr = ( -2.37 + 0.1*stratqvp[X_DBZH] - 2.89*stratqvp[X_ZDR] + 1.28*stratqvp[X_ZDR]**2 - 0.213*stratqvp[X_ZDR]**3 )# Hu and Ryzhkov 2022 eq. 3 [log(1/L)]

        # Put everything together
        retrievals[dn][stratname][dn] = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
                                                                 "lwc_zh_zdr2":lwc_zh_zdr2,
                                                                 "lwc_kdp": lwc_kdp,
                                                                 "iwc_zh_t": iwc_zh_t,
                                                                 "iwc_zdr_zh_kdp": iwc_zdr_zh_kdp,
                                                                 "Dm_ice_zh": Dm_ice_zh,
                                                                 "Dm_ice_zh_kdp": Dm_ice_zh_kdp,
                                                                 "Dm_ice_zdp_kdp": Dm_ice_zdp_kdp,
                                                                 "Dm_rain_zdr": Dm_rain_zdr,
                                                                 "Dm_rain_zdr2": Dm_rain_zdr2,
                                                                 "Dm_rain_zdr3": Dm_rain_zdr3,
                                                                 "Nt_ice_zh_iwc": Nt_ice_zh_iwc,
                                                                 "Nt_ice_zh_iwc2": Nt_ice_zh_iwc2,
                                                                 "Nt_ice_zh_iwc_kdp": Nt_ice_zh_iwc_kdp,
                                                                 "Nt_ice_zh_iwc2_kdp": Nt_ice_zh_iwc2_kdp,
                                                                 "Nt_rain_zh_zdr": Nt_rain_zh_zdr,
                                                                 }).compute()


#%% Filters (conditions for stratiform) Retrievals PPI-based
qvps_strat = {}
qvps_strat_relaxed = {}
qvps_strat_fil = {}
qvps_strat_relaxed_fil = {}
retrievals = {}

for dn in data.keys():

    start_time = time.time()
    print("Filtering stratiform conditions...")

    X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(data[dn])
    X_KDP = "KDP_ML_corrected"

    if X_RHO+"_NC" in data[dn].data_vars:
        X_RHO = X_RHO+"_NC"

    if X_ZDR+"_OC" in data[dn].data_vars:
        X_ZDR = X_ZDR+"_OC"

    # Check that RHOHV_NC is actually better (less std) than RHOHV, otherwise just use RHOHV, on a per-day basis
    std_margin = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_margin), otherwise use RHOHV
    min_rho = 0.6 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results

    # if "_NC" in X_RHO:
    #     # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
    #     # if that is the case we take it that the correction did not work well so we won't use it
    #     cond_rhohv = (
    #                     qvps[dn][X_RHO].where(qvps[dn][X_RHO]>min_rho).resample({"time":"D"}).std(dim=("time", "z")) < \
    #                     qvps[dn]["RHOHV"].where(qvps[dn]["RHOHV"]>min_rho).resample({"time":"D"}).std(dim=("time", "z"))*(1+std_margin)
    #                     ).compute()

    #     # create an xarray.Dataarray with the valid timesteps
    #     valid_dates = cond_rhohv.where(cond_rhohv, drop=True).time.dt.date
    #     valid_datetimes = [date.values in valid_dates for date in qvps[dn].time.dt.date]
    #     valid_datetimes_xr = xr.DataArray(valid_datetimes, coords={"time": qvps[dn]["time"]})

    #     # Redefine RHOHV_NC: keep it in the valid datetimes, put RHOHV in the rest
    #     qvps[dn][X_RHO] = qvps[dn][X_RHO].where(valid_datetimes_xr, qvps[dn]["RHOHV"])


    # Conditions to clean ML height values
    max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
    max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
    time_window = 5 # set timestep window for the std computation (centered)
    min_period = 3 # set minimum number of valid ML values in the window (centered)

    cond_ML_bottom_change = abs(data[dn]["height_ml_bottom_new_gia"].diff("time").compute())<max_change
    cond_ML_bottom_std = data[dn]["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
    # cond_ML_bottom_minlen = data[dn]["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

    cond_ML_top_change = abs(data[dn]["height_ml_new_gia"].diff("time").compute())<max_change
    cond_ML_top_std = data[dn]["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
    # cond_ML_top_minlen = data[dn]["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

    allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

    # Filter only fully stratiform pixels (min entropy >= 0.8 and ML detected)
    qvps_strat[dn] = data[dn].where( (data[dn]["min_entropy"]>=0.8).compute() & allcond, drop=True)
    # Relaxed alternative: Filter data with at least 50% of stratiform pixels (min entropy >= 0.8 and ML detected)
    # qvps_strat_relaxed[dn] = data[dn].where( ( (data[dn]["min_entropy"]>=0.8).sum("range").compute() >= data[dn][X_DBZH].count("range").compute()/2 ) & allcond, drop=True)

    # Filter out non relevant values
    qvps_strat_fil[dn] = qvps_strat[dn].where((qvps_strat[dn][X_TH] > -10 )&
                                      (qvps_strat[dn][X_KDP] > 0.01)&
                                      (qvps_strat[dn][X_KDP] < 3)&
                                      (qvps_strat[dn][X_RHO] > 0.7)&
                                      (qvps_strat[dn][X_ZDR] > -1) &
                                      (qvps_strat[dn][X_ZDR] < 3))

    # qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed[dn].where((qvps_strat_relaxed[dn][X_TH] > -10 )&
    #                                   (qvps_strat_relaxed[dn][X_KDP] > -0.1)&
    #                                   (qvps_strat_relaxed[dn][X_KDP] < 3)&
    #                                   (qvps_strat_relaxed[dn][X_RHO] > 0.7)&
    #                                   (qvps_strat_relaxed[dn][X_ZDR] > -1) &
    #                                   (qvps_strat_relaxed[dn][X_ZDR] < 3))

    # try:
    #     qvps_strat_fil[dn] = qvps_strat_fil[dn].where(qvps_strat_fil[dn]["SNRHC"]>10)
    #     # qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed_fil[dn].where(qvps_strat_relaxed_fil[dn]["SNRHC"]>10)
    # except KeyError:
    #     qvps_strat_fil[dn] = qvps_strat_fil[dn].where(qvps_strat_fil[dn]["SNRH"]>10)
    #     # qvps_strat_relaxed_fil[dn] = qvps_strat_relaxed_fil[dn].where(qvps_strat_relaxed_fil[dn]["SNRH"]>10)
    # except:
    #     print("Could not filter out low SNR")

    total_time = time.time() - start_time
    print(f"took {total_time/60:.2f} minutes.")

    #### Calculate retreivals
    # We do this for both qvps_strat_fil and relaxed qvps_strat_relaxed_fil
    start_time = time.time()
    print("Calculating microphysical retrievals...")

    # to check the wavelength of each radar, in cm for DWD, in 1/100 cm for DMI ()
    # filewl = ""
    # xr.open_dataset(filewl, group="how") # DWD
    # file1 = realpep_path+"/upload/jgiles/dmi_raw/acq/OLDDATA/uza/RADAR/2015/01/01/ANK/RAW/ANK150101000008.RAW6M00"
    # xd.io.backends.iris.IrisRawFile(file1, loaddata=False).ingest_header["task_configuration"]["task_misc_info"]["wavelength"]

    Lambda = 53.1 # radar wavelength in mm (pro: 53.138, ANK: 53.1, AFY: 53.3, GZT: 53.3, HTY: 53.3, SVS:53.3)

    if dn == "xband":
        Lambda = 32

    # We will put the final retrievals in a dict
    try: # check if exists, if not, create it
        retrievals[dn]
    except NameError:
        retrievals[dn] = {}
    except KeyError:
        retrievals[dn] = {}

    for stratname, stratqvp in [("stratiform", qvps_strat_fil[dn].copy())]:
        print("   ... for "+stratname)

        retrievals[dn][stratname] = {}

        # LWC
        lwc_zh_zdr = 10**(0.058*stratqvp[X_DBZH] - 0.118*stratqvp[X_ZDR] - 2.36) # Reimann et al 2021 eq 3.7 (adjusted for Germany)
        lwc_zh_zdr2 = 1.38*10**(-3) *10**(0.1*stratqvp[X_DBZH] - 2.43*stratqvp[X_ZDR] + 1.12*stratqvp[X_ZDR]**2 - 0.176*stratqvp[X_ZDR]**3 ) # used in S band, Ryzhkov 2022 PROM presentation https://www2.meteo.uni-bonn.de/spp2115/lib/exe/fetch.php?media=internal:uploads:all_hands_schneeferner_july2022:ryzhkov.pdf
        lwc_kdp = 10**(0.568*np.log10(stratqvp[X_KDP]) + 0.06) # Reimann et al 2021(adjusted for Germany)

        # IWC (Collected from Blanke et al 2023)
        iwc_zh_t = 10**(0.06 * stratqvp[X_DBZH] - 0.0197*stratqvp["TEMP"] - 1.7) # empirical from Hogan et al 2006 Table 2

        iwc_zdr_zh_kdp = xr.where(stratqvp[X_ZDR]>=0.4, # Carlin et al 2021 eqs 4b and 5b
                                  4*10**(-3)*( stratqvp[X_KDP]*Lambda/( 1-wrl.trafo.idecibel(stratqvp[X_ZDR])**-1 ) ),
                                  0.033 * ( stratqvp[X_KDP]*Lambda )**0.67 * wrl.trafo.idecibel(stratqvp[X_DBZH])**0.33 )

        # Dm (ice collected from Blanke et al 2023)
        Dm_ice_zh = 1.055*wrl.trafo.idecibel(stratqvp[X_DBZH])**0.271 # Matrosov et al. (2019) Fig 10 (S band)
        Dm_ice_zh_kdp = 0.67*( wrl.trafo.idecibel(stratqvp[X_DBZH])/(stratqvp[X_KDP]*Lambda) )**(1/3) # Ryzhkov and Zrnic (2019). Idk exactly where does the 0.67 approximation comes from, Blanke et al. 2023 eq 10 and Carlin et al 2021 eq 5a cite Bukovčić et al. (2018, 2020) but those two references do not show this formula.
        Dm_ice_zdp_kdp = -0.1 + 2*( (wrl.trafo.idecibel(stratqvp[X_DBZH])*(1-wrl.trafo.idecibel(stratqvp[X_ZDR])**-1 ) ) / (stratqvp[X_KDP]*Lambda) )**(1/2) # Ryzhkov and Zrnic (2019). Zdp = Z(1-ZDR**-1) from Carlin et al 2021

        Dm_rain_zdr = 0.3015*stratqvp[X_ZDR]**3 - 1.2087*stratqvp[X_ZDR]**2 + 1.9068*stratqvp[X_ZDR] + 0.5090 # (for rain but tuned for Germany X-band, JuYu Chen, Zdr in dB, Dm in mm)

        D0_rain_zdr2 = 0.171*stratqvp[X_ZDR]**3 - 0.725*stratqvp[X_ZDR]**2 + 1.48*stratqvp[X_ZDR] + 0.717 # (D0 from Hu and Ryzhkov 2022, used in S band data but could work for C band) [mm]
        D0_rain_zdr3 = xr.where(stratqvp[X_ZDR]<1.25, # D0 from Bringi et al 2009 (C-band) eq. 1 [mm]
                                0.0203*stratqvp[X_ZDR]**4 - 0.1488*stratqvp[X_ZDR]**3 + 0.2209*stratqvp[X_ZDR]**2 + 0.5571*stratqvp[X_ZDR] + 0.801,
                                0.0355*stratqvp[X_ZDR]**3 - 0.3021*stratqvp[X_ZDR]**2 + 1.0556*stratqvp[X_ZDR] + 0.6844
                                )
        mu = 0
        Dm_rain_zdr2 = D0_rain_zdr2 * (4+mu)/(3.67+mu) # conversion from D0 to Dm according to eq 4 of Hu and Ryzhkov 2022.
        Dm_rain_zdr3 = D0_rain_zdr3 * (4+mu)/(3.67+mu)

        # log(Nt)
        Nt_ice_zh_iwc = (3.39 + 2*np.log10(iwc_zh_t) - 0.1*stratqvp[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2 = (3.69 + 2*np.log10(iwc_zh_t) - 0.1*stratqvp[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_ice_zh_iwc_kdp = (3.39 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*stratqvp[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2_kdp = (3.69 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*stratqvp[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_rain_zh_zdr = ( -2.37 + 0.1*stratqvp[X_DBZH] - 2.89*stratqvp[X_ZDR] + 1.28*stratqvp[X_ZDR]**2 - 0.213*stratqvp[X_ZDR]**3 )# Hu and Ryzhkov 2022 eq. 3 [log(1/L)]

        # Put everything together
        retrievals[dn][stratname][dn] = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
                                                                 "lwc_zh_zdr2":lwc_zh_zdr2,
                                                                 "lwc_kdp": lwc_kdp,
                                                                 "iwc_zh_t": iwc_zh_t,
                                                                 "iwc_zdr_zh_kdp": iwc_zdr_zh_kdp,
                                                                 "Dm_ice_zh": Dm_ice_zh,
                                                                 "Dm_ice_zh_kdp": Dm_ice_zh_kdp,
                                                                 "Dm_ice_zdp_kdp": Dm_ice_zdp_kdp,
                                                                 "Dm_rain_zdr": Dm_rain_zdr,
                                                                 "Dm_rain_zdr2": Dm_rain_zdr2,
                                                                 "Dm_rain_zdr3": Dm_rain_zdr3,
                                                                 "Nt_ice_zh_iwc": Nt_ice_zh_iwc,
                                                                 "Nt_ice_zh_iwc2": Nt_ice_zh_iwc2,
                                                                 "Nt_ice_zh_iwc_kdp": Nt_ice_zh_iwc_kdp,
                                                                 "Nt_ice_zh_iwc2_kdp": Nt_ice_zh_iwc2_kdp,
                                                                 "Nt_rain_zh_zdr": Nt_rain_zh_zdr,
                                                                 })

        retrievals[dn][stratname][dn] = utils.compute_qvp(xr.merge([stratqvp, retrievals[dn][stratname][dn]]), min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10,"SNRHC":10, "SQIH":0.5} ).compute()

        loc = utils.find_loc(utils.locs, path_cband)
        retrievals[dn][stratname][dn] = utils.attach_ERA5_TEMP(retrievals[dn][stratname][dn], path=loc.join(utils.era5_dir.split("loc")))

#%% Plot QVPs

dn = "xband"
ds_qvp = qvps[dn]

max_height = 12000 # max height for the qvp plots

tsel = ""# slice("2017-08-31T19","2017-08-31T22")
if tsel == "":
    datasel = ds_qvp.loc[{"z": slice(0, max_height)}]
else:
    datasel = ds_qvp.loc[{"time": tsel, "z": slice(0, max_height)}]

templevels = [-100, 0]
mom = "DBZH"

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
datasel[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3))
figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=templevels)
# datasel["min_entropy"].compute().dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[0.8, 1], hatches=["", "XXX", ""], colors=[(1,1,1,0)], add_colorbar=False, extend="both")
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
datasel["height_ml_new_gia"].plot(c="black")
datasel["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

try:
    elevtitle = " "+str(np.round(data[dn]["sweep_fixed_angle"].values[0], 2))+"°"
except:
    elevtitle = " "+str(np.round(data[dn]["sweep_fixed_angle"].values, 2))+"°"

plt.title(mom+elevtitle+". "+str(datasel.time.values[0]).split(".")[0])
plt.show()
plt.close()

#%% CFTDs Plot
dn = "cband"

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below and do not save.
auto_plot = False
savepath = None

# Which to plot, qvps_strat_fil or qvps_strat_relaxed_fil
ds_to_plot = qvps_strat_fil[dn].copy()

# Define list of seasons
selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

# adjustment from K to C (disabled now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit (only works if auto_plot=False)
ytlim=-20

# season to plot (only works if auto_plot=False)
selseas = selseaslist[0]
selmonths = selseas[1]

# Temp bins
tb=1# degress C

# Min counts per Temp layer
mincounts=0

#Colorbar limits and step
cblim=[0,10]
colsteps=10

cmaphist="Oranges"

savedict = {"custom": None} # placeholder for the for loop below, not important


vars_to_plot = {"DBZH": [0, 46, 1],
                "ZDR_OC": [-0.5, 2.1, 0.1],
                "KDP_ML_corrected": [-0.1, 0.52, 0.02],
                "RHOHV_NC": [0.9, 1.004, 0.004]}

if auto_plot:
    vtp = [{"DBZH": [0, 46, 1],
                    "ZDR_OC": [-0.5, 2.1, 0.1],
                    "KDP_ML_corrected":  [-0.1, 0.52, 0.02],
                    "RHOHV_NC": [0.9, 1.004, 0.004]},
           {"DBZH": [0, 46, 1],
                           "ZDR": [-0.5, 2.1, 0.1],
                           "KDP_CONV":  [-0.1, 0.52, 0.02],
                           "RHOHV": [0.9, 1.004, 0.004]} ]
    ytlimlist = [-20, -50]
    loc = find_loc(locs, ff[0])
    savedict = {}
    add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
    for selseas in selseaslist:
        savedict.update(
                    {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+".png": [vtp[0], ytlimlist[0], selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_extended.png": [vtp[0], ytlimlist[1], selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr.png": [vtp[1], ytlimlist[0], selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_uncorr_extended.png": [vtp[1], ytlimlist[1], selseas[1]],
                    }
                        )

for savename in savedict.keys():
    if auto_plot:
        vars_to_plot = savedict[savename][0]
        ytlim = savedict[savename][1]
        selmonths = savedict[savename][2]

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        adj=1
        if "RHOHV" in vv:
            so = True
            binsx2 = [0.9, 1.005, 0.005]
        if "KDP" in vv:
            adj=1
        utils.hist2d(ax[nn], ds_to_plot[vv].sel(time=ds_to_plot['time'].dt.month.isin(selmonths))*adj,
                     ds_to_plot["TEMP"].sel(time=ds_to_plot['time'].dt.month.isin(selmonths))+adjtemp,
                     whole_x_range=True,
                     binsx=vars_to_plot[vv], binsy=[ytlim,16,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps,
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)

        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts

    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')
    if auto_plot:
        # Create savefolder
        savepath_seas = os.path.dirname(savepath+savename)
        if not os.path.exists(savepath_seas):
            os.makedirs(savepath_seas)
        fig.savefig(savepath+savename, bbox_inches="tight")
        print("AUTO PLOT: saved "+savename)


#%% CFTDs retreivals Plot
# We assume that everything above ML is frozen and everything below is liquid
dn = "cband"
loc = dn

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below and do not save.
auto_plot = False
savepath = None

# Which to plot, stratiform or stratiform_relaxed
ds_to_plot = retrievals[dn]["stratiform"].copy()

# Define list of seasons
selseaslist = [
            ("full", [1,2,3,4,5,6,7,8,9,10,11,12]),
            ("DJF", [12,1,2]),
            ("MAM", [3,4,5]),
            ("JJA", [6,7,8]),
            ("SON", [9,10,11]),
           ] # ("nameofseas", [months included])

# adjustment from K to C (disabled now because I know that all qvps have ERA5 data)
adjtemp = 0
# if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
#     print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
#     adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit (only works if auto_plot=False)
ytlim=-20

# season to plot (only works if auto_plot=False)
selseas = selseaslist[0]
selmonths = selseas[1]

# Select which retrievals to plot (only works if auto_plot=False)
IWC = "iwc_zdr_zh_kdp" # iwc_zh_t or iwc_zdr_zh_kdp
LWC = "lwc_kdp" # lwc_zh_zdr (adjusted for Germany) or lwc_zh_zdr2 (S-band) or lwc_kdp
Dm_ice = "Dm_ice_zdp_kdp" # Dm_ice_zh, Dm_ice_zh_kdp, Dm_ice_zdp_kdp
Dm_rain = "Dm_rain_zdr3" # Dm_rain_zdr, Dm_rain_zdr2 or Dm_rain_zdr3
Nt_ice = "Nt_ice_zh_iwc2_kdp" # Nt_ice_zh_iwc, Nt_ice_zh_iwc2, Nt_ice_zh_iwc_kdp, Nt_ice_zh_iwc2_kdp
Nt_rain = "Nt_rain_zh_zdr" # Nt_rain_zh_zdr

vars_to_plot = {"IWC/LWC [g/m^{3}]": [-0.1, 0.82, 0.02], # [-0.1, 0.82, 0.02],
                "Dm [mm]": [0, 3.1, 0.1], # [0, 3.1, 0.1],
                "Nt [log10(1/L)]": [-2, 2.1, 0.1], # [-2, 2.1, 0.1],
                }

savedict = {"custom": None} # placeholder for the for loop below, not important


for savename in savedict.keys():
    if auto_plot:
        ytlim = savedict[savename][0]
        IWC = savedict[savename][1]
        LWC = savedict[savename][2]
        Dm_ice = savedict[savename][3]
        Dm_rain = savedict[savename][4]
        Nt_ice = savedict[savename][5]
        Nt_rain = savedict[savename][6]
        selmonths = savedict[savename][7]

    retreivals_merged = xr.Dataset({
                                    "IWC/LWC [g/m^{3}]": ds_to_plot[loc][IWC].where(ds_to_plot[loc][IWC].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][LWC].where(ds_to_plot[loc][LWC].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ),
                                    "Dm [mm]": ds_to_plot[loc][Dm_ice].where(ds_to_plot[loc][Dm_ice].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][Dm_rain].where(ds_to_plot[loc][Dm_rain].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ),
                                    "Nt [log10(1/L)]": (ds_to_plot[loc][Nt_ice].where(ds_to_plot[loc][Nt_ice].z > ds_to_plot[loc].height_ml_new_gia,
                                                                      ds_to_plot[loc][Nt_rain].where(ds_to_plot[loc][Nt_rain].z < ds_to_plot[loc].height_ml_bottom_new_gia ) ) ),
        })

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15,5), width_ratios=(1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

    for nn, vv in enumerate(vars_to_plot.keys()):
        so=False
        binsx2=None
        adj=1
        if "RHOHV" in vv:
            so = True
            binsx2 = [0.9, 1.005, 0.005]
        if "KDP" in vv:
            adj=1
        utils.hist2d(ax[nn], retreivals_merged[vv].sel(time=retreivals_merged['time'].dt.month.isin(selmonths))*adj,
                     retreivals_merged["TEMP"].sel(time=retreivals_merged['time'].dt.month.isin(selmonths))+adjtemp,
                     whole_x_range=True,
                     binsx=vars_to_plot[vv], binsy=[ytlim,16,tb], mode='rel_y', qq=0.2,
                     cb_mode=(nn+1)/len(vars_to_plot), cmap=cmaphist, colsteps=colsteps,
                     fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot),
                     cborientation="vertical", shading="nearest", smooth_out=so, binsx_out=binsx2)
        ax[nn].set_ylim(15,ytlim)
        ax[nn].set_xlabel(vv, fontsize=10)

        ax[nn].tick_params(labelsize=15) #change font size of ticks
        plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts

    ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')

    if auto_plot:
        # Create savefolder
        savepath_seas = os.path.dirname(savepath+savename)
        if not os.path.exists(savepath_seas):
            os.makedirs(savepath_seas)
        fig.savefig(savepath+savename, bbox_inches="tight")
        print("AUTO PLOT: saved "+savename)
