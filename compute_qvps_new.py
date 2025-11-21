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


# NEEDS WRADLIB 2.0 !! (OR GREATER?)

import wradlib as wrl
import sys
import glob
import xarray as xr
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
overwrite = False # overwrite existing files?
save_processed_ppi = False # Save PPIs after processing?
save_retrievals_ppi = False # Save PPIs of microphysical retrievals? (this is pretty slow and uses substantial storage)

# ZDR offset loading parameters
abs_zdr_off_min_thresh = 0. # if ZDR_OC has more negative values than the original ZDR
# and the absolute median offset is < abs_zdr_off_min_thresh, then undo the correction (set to 0 to avoid this step)
daily_corr = True # correct ZDR with daily offset? if False, correct with timestep offsets
if "dmi" in path0:
    daily_corr = False
variability_check = True # Check the variability of the timestep offsets to decide between daily or timestep offsets?
variability_thresh = 0.1 # Threshold value of timestep-based ZDR offsets variability to use if variability_check is True.
first_offset_method_abs_priority = False # If True, give absolute priority to the offset from the first method if valid
mix_offset_variants = True # Use timestep-offsets variants to fill NaNs?
mix_daily_offset_variants = False # select the most appropriate daily offset variant based on how_mix_zdr_offset
mix_zdr_offsets = True # Select the most appropriate offset or timestep offset based on how_mix_zdr_offset
how_mix_zdr_offset = "count" # how to choose between the different offsets
# if mix_zdr_offsets = True. "count" will choose the offset that has more data points
# in its calculation (there must be a variable ZDR_offset_datacount in the loaded offset).
# "neg_overcorr" will choose the offset that generates less negative ZDR values.
abs_zdr_off_min_thresh = 0. # If abs_zdr_off_min_thresh > 0, check if offset corrected ZDR has more
# negative values than the original ZDR and if the absolute median offset
# is < abs_zdr_off_min_thresh, then undo the correction (set to 0 to avoid this step).
propagate_forward = False # Propagate timestep offsets forward to fill NaNs?
fill_with_daily_offset = True # Fill timestep NaN offsets with daily offset?

# PHIDP processing / KDP calc parameters
window0max = 25 # max value for window0 (only applied if window0 is given in meters)
winlen0max = [9, 25] # max value for winlen0 (only applied if winlen0 is given in meters)
SNRH_min = 15 # min value for SNRH thresholding. This has a significant influence in the KDP calculation and also affects the QVPs computations.

# Set the possible ZDR calibrations locations to include (in order of priority)
# The script will try to correct according to the first offset; if not available or nan it will
# continue with the next one, and so on. Only the used offset will be outputted in the final file.
# All items in zdrofffile will be tested in each zdroffdir to load the data.
# Build dictionary of offset filepaths
zdroffdir = utils.zdroffdir
zdrofffile = utils.zdrofffile
zdrofffile_ts = utils.zdrofffile_ts

# set the RHOHV correction location
rhoncdir = utils.rhoncdir  # subfolder where to find the noise corrected rhohv data
rhoncfile = utils.rhoncfile # pattern to select the appropriate file (careful with the rhohv_nc_2percent)
if type(rhoncfile) is str:
    rhoncfile = [rhoncfile]

# get the files and check that it is not empty
if "hd5" in path0 or "h5" in path0:
    files=[path0]
elif "dwd" in path0:
    files = sorted(glob.glob(path0+"/*allm*hd5*"))
elif "dmi" in path0:
    files = sorted(glob.glob(path0+"/*allm*h5*"))
elif isinstance(path0, list):
    files = path0
else:
    print("Country code not found in path")
    sys.exit("Country code not found in path.")

if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

clowres0=False # this is for the ML detection algorithm
min_hgts = utils.min_hgts
min_rngs = utils.min_rngs
min_hgt = min_hgts["default"] # minimum height above the radar to be considered
min_range = min_rngs["default"] # minimum range from which to consider data (mostly for bad PHIDP filtering)
if "dwd" in path0 and "90grads" in path0:
    # for the VP we need to set a higher min height because there are several bins of unrealistic values
    min_hgt = min_hgts["90grads"]
if "ANK" in path0:
    min_hgt = min_hgts["ANK"]
    min_range = min_rngs["ANK"]
if "GZT" in path0:
    min_hgt = min_hgts["GZT"]
    min_range = min_rngs["GZT"]
if "AFY" in path0:
    min_range = min_rngs["AFY"]
if "SVS" in path0:
    min_range = min_rngs["SVS"]
if "HTY" in path0:
    min_range = min_rngs["HTY"]

# ERA5 folder
if "dwd" in path0:
    cloc = "germany"
if "dmi" in path0:
    cloc = "turkey"
if "boxpol" in path0:
    cloc = "germany"

if os.path.exists("/automount/ags/jgiles/ERA5/hourly/"):
    # then we are in local system
    era5_dir = "/automount/ags/jgiles/ERA5/hourly/"+cloc+"/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/scratch/detectrea/giles1/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/scratch/detectrea/giles1/ERA5/hourly/"+cloc+"/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/largedata2/detectdata/projects/A04/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/largedata2/detectdata/projects/A04/ERA5/hourly/"+cloc+"/pressure_level_vars/" # dummy loc placeholder, it gets replaced below


# names of variables
phidp_names = ["UPHIDP", "PHIDP"] # names to look for the PHIDP variable, in order of preference
dbzh_names = ["DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV
zdr_names = ["ZDR"]
th_names = ["TH", "DBTH", "DBZH"]


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
    elif "boxpol" in ff:
        country="boxpol"
    else:
        print("Country code not found in path")
        sys.exit("Country code not found in path.")

    ff_parts = ff.split(country)
    savepath = (country+"/"+name+"/").join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    return savepath


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

    # check for the file DONE.txt in the savepath before starting
    savepath = make_savedir(ff, "qvps")
    if os.path.exists(os.path.dirname(savepath)+"/DONE.txt") and not overwrite:
        continue

    print("processing "+ff)
    if "dwd" in ff:
        country="dwd"
        data = utils.load_dwd_preprocessed(ff) # this already loads the first elev available in the files and fixes time coord ahd phidp unfolding and flipping
    elif "dmi" in ff:
        country="dmi"
        data = utils.load_dmi_preprocessed(ff) # this loads DMI file and flips and unfolds phidp and fixes time coord
    else:
        if "boxpol" in ff: country="boxpol"
        data=xr.open_dataset(ff)

    # flip UPHIDP and KDP in UMD data
    # if "umd" in ff: # this is now done automatically with the loading functions
    #     print("Flipping phase moments in UMD")
    #     for vf in ["UPHIDP", "KDP"]: # Phase moments in UMD are flipped into the negatives
    #         attrs = data[vf].attrs.copy()
    #         data[vf] = data[vf]*-1
    #         data[vf].attrs = attrs.copy()

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

    if "dwd" in path0 and "vol5minng01" in path0:
        if swp.range.diff("range").median() > 750:
            clowres0=True # for the ML correction algorithm

#%% Load noise corrected RHOHV if available
    for rhoncfile0 in rhoncfile:
        try:
            rhoncpath = os.path.dirname(utils.edit_str(ff, country, country+rhoncdir))
            swp = utils.load_corrected_RHOHV(swp, rhoncpath+"/"+rhoncfile0)

            if "RHOHV_NC" not in swp:
                continue

            # Check that the corrected RHOHV does not have a lot more of low values
            # if that is the case we take it that the correction did not work.
            min_rho = 0.7 # min RHOHV value for filtering
            mean_tolerance = 0.02 # 2% tolerance, for checking if RHOHV_NC is actually larger than RHOHV (overall higher values)

            if ( swp["RHOHV_NC"].where(swp["z"]>min_height).mean() > swp[X_RHO].where(swp["z"]>min_height).mean()*(1-mean_tolerance) ).compute():

                # Check that the corrected RHOHV does not have higher STD than the original (1 + std_tolerance)
                # if that is the case we take it that the correction did not work well so we won't use it
                std_tolerance = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_tolerance), otherwise use RHOHV

                if ( swp["RHOHV_NC"].where(swp[X_RHO]>min_rho * (swp["z"]>min_height)).std() < swp[X_RHO].where(swp[X_RHO]>min_rho * (swp["z"]>min_height)).std()*(1+std_tolerance) ).compute():
                    # Change the default RHOHV name to the corrected one
                    X_RHO = "RHOHV_NC"
            break

        except OSError:
            print("No noise corrected rhohv to load: "+rhoncpath+"/"+rhoncfile0)

        except ValueError:
            print("ValueError with corrected rhohv: "+rhoncpath+"/"+rhoncfile0)

#%% Correct ZDR-elevation dependency
    try:
        angle = float(swp.elevation.mean())
    except:
        angle = float(swp.sweep_fixed_angle.mean())
    try:
        swp = utils.zdr_elev_corr(swp, angle, zdr=[X_ZDR])
        X_ZDR = X_ZDR+"_EC"
    except:
        pass

#%% Load ZDR offset if available
    daily_zdr_off_paths = {}
    for offsetdir in zdroffdir:
        daily_zdr_off_paths[offsetdir] = []
        for offsetfile in zdrofffile:
            try:
                daily_zdr_off_paths[offsetdir].append(
                    utils.get_ZDR_offset_files(ff, country, offsetdir, offsetfile))
            except FileNotFoundError:
                pass

    zdr_off_paths = {}
    for offsetdir in zdroffdir:
        zdr_off_paths[offsetdir] = []
        for offsetfile in zdrofffile_ts:
            try:
                zdr_off_paths[offsetdir].append(
                    utils.get_ZDR_offset_files(ff, country, offsetdir, offsetfile))
            except FileNotFoundError:
                pass

    # Load the offsets

    try:
        swp = utils.load_best_ZDR_offset(swp, X_ZDR,
                                        zdr_off_paths=zdr_off_paths,
                                        daily_zdr_off_paths=daily_zdr_off_paths,
                                 zdr_off_name="ZDR_offset", zdr_oc_name=X_ZDR+"_OC",
                                 attach_all_vars=True,
                                 daily_corr = daily_corr, variability_check = variability_check,
                                 variability_thresh = variability_thresh,
                                 first_offset_method_abs_priority = first_offset_method_abs_priority,
                                 mix_offset_variants = mix_offset_variants,
                                 mix_daily_offset_variants = mix_daily_offset_variants,
                                 mix_zdr_offsets = mix_zdr_offsets, how_mix_zdr_offset = how_mix_zdr_offset,
                                 t_tolerance = 10,
                                 abs_zdr_off_min_thresh = abs_zdr_off_min_thresh,
                                 X_RHO=X_RHO, min_height=min_height,
                                 propagate_forward = propagate_forward,
                                 fill_with_daily_offset = fill_with_daily_offset,
                                 )
        X_ZDR = X_ZDR+"_OC"
    except:
        print("Loading ZDR offsets failed")


#%% Correct PHIDP
    ################## Before entropy calculation we need to use the melting layer detection algorithm
    ds = swp.copy()
    interpolation_method_ML = "linear" # for interpolating PHIDP in the ML

    # Check that PHIDP is in data, otherwise skip ML detection
    if X_PHI in ds.data_vars:
        # Set parameters according to data
        phase_proc_params = utils.get_phase_proc_params(ff).copy() # get default phase processing parameters
        window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia, grad_thresh = phase_proc_params.values()

        # Check if window0 and winlen0 are in m or in number of gates and apply max threshold
        rangeres = float(ds.range.diff("range").mean().compute())
        if window0 > 500:
            window0 = int(round(window0/rangeres))
            if not window0%2>0: window0 = window0 + 1
            window0 = min(window0max, window0)
        if isinstance(winlen0, list):
            if winlen0[0] > 500:
                wl0 = int(round(winlen0[0]/rangeres))
                if not wl0%2>0: wl0 = wl0 + 1
                winlen0[0] = max(winlen0max[0], wl0)
            if winlen0[1] > 500:
                wl0 = int(round(winlen0[1]/rangeres))
                if not wl0%2>0: wl0 = wl0 + 1
                winlen0[1] = min(winlen0max[1], wl0)
        else:
            if winlen0 > 500:
                winlen0 = int(round(winlen0/rangeres))
                if not winlen0%2>0: winlen0 = winlen0 + 1
                winlen0 = min(winlen0max, winlen0)

        ######### Processing PHIDP
        #### fix PHIDP

        # phidp may be already preprocessed (turkish case), then only offset-correct (no smoothing) and then vulpiani
        if "PHIDP" not in X_PHI: # This is now always skipped with this definition ("PHIDP" is in both X_PHI); i.e., we apply full processing to turkish data too
            # calculate phidp offset
            ds_phiproc = utils.phidp_offset_correction(utils.apply_min_max_thresh(ds, {"SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5}, {}, skipfullna=True),
                                               X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                                 rng_min=1000, rng=rng, azmedian=azmedian, tolerance=(0,5)) # shorter rng, rng_min for finer turkish data

            phi_masked = ds_phiproc[X_PHI+"_OC"].where((ds[X_RHO] >= 0.8) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

        else:
            # process phidp (offset and smoothing)
            ds_phiproc = utils.phidp_processing(utils.apply_min_max_thresh(ds, {"SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5}, {}, skipfullna=True),
                                        X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                                 rng=rng, azmedian=azmedian, tolerance=(0,5))

            phi_masked = ds_phiproc[X_PHI+"_OC_SMOOTH"].where((ds[X_RHO] >= 0.8) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

        # Assign new vars to ds
        ds = ds.assign(ds_phiproc[[X_PHI+"_OC_SMOOTH", X_PHI+"_OFFSET", X_PHI+"_OC"]])

        # Assign phi_masked
        assign = { X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs) }

        ds = ds.assign(assign)

        # derive KDP from PHIDP (Vulpiani)

        if isinstance(winlen0, list):
            # if winlen0 is a list, use the first value (small window) for strong rain (SR, DBZH>40) and
            # use the second value (large window) for light rain (LR, DBZH<=40)
            ds_kdpSR = utils.kdp_phidp_vulpiani(ds, winlen0[0], X_PHI+"_OC_MASKED", min_periods=max(3, int((winlen0[0] - 1) / 4)))[["KDP_CONV", "PHI_CONV"]]
            ds_kdpLR = utils.kdp_phidp_vulpiani(ds, winlen0[1], X_PHI+"_OC_MASKED", min_periods=max(3, int((winlen0[1] - 1) / 4)))[["KDP_CONV", "PHI_CONV"]]
            ds_kdp = xr.where(ds[X_DBZH]>40,
                              ds_kdpSR, ds_kdpLR)
            ds = ds.assign(ds_kdp)
        else:
            ds = utils.kdp_phidp_vulpiani(ds, winlen0, X_PHI+"_OC_MASKED", min_periods=max(3, int((winlen0 - 1) / 4)))

        X_PHI = X_PHI+"_OC" # continue using offset corrected PHI

    else:
        print(X_PHI+" not found in the data, skipping ML detection")

#%% Attach ERA5 variables
    era5_vars = ["temperature", "relative_humidity"]
    era5_vars_rename = {"t":"TEMP", "r":"RH"}
    ds = utils.attach_ERA5_fields(ds, path=era5_dir, convert_to_C=True,
                           variables=era5_vars,
                           rename=era5_vars_rename, set_as_coords=False,
                           k_n=9, pre_interpolate_z=True)

    # Save ERA5 ppis
    for vv in era5_vars_rename.values():
        ds[vv].encoding = {'zlib': True, 'complevel': 6}
    era5_ppis_path = make_savedir(ff, "ppis_era5")
    ds[[vv for vv in era5_vars_rename.values()]].to_netcdf(era5_ppis_path)

#%% Compute QVP
    ## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
    ## Also added further filtering (TH>0, ZDR>-1)
    ds_qvp_ra, ds_qvp_ra_count = utils.compute_qvp(ds, min_thresh={X_RHO:0.7, X_TH:0, X_ZDR:-1,
                                                  "SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5},
                                  output_count=True)

    # assign DBZH counts
    ds_qvp_ra = ds_qvp_ra.assign({"DBZH_qvp_count": ds_qvp_ra_count["DBZH"]})

#%% Detect melting layer
    if X_PHI in ds.data_vars:
        moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}

        ds_qvp_ra = utils.melting_layer_qvp_X_new(ds_qvp_ra.where(ds_qvp_ra_count>20), moments=moments, dim="z", fmlh=0.3, grad_thresh=grad_thresh,
                 xwin=xwin0, ywin=ywin0, min_h=min_height, rhohv_thresh_gia=rhohv_thresh_gia, all_data=True, clowres=clowres0)

        #### Assign ML values to dataset

        ds = ds.assign_coords({'height_ml': ds_qvp_ra.height_ml})
        ds = ds.assign_coords({'height_ml_bottom': ds_qvp_ra.height_ml_bottom})
        ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia})
        ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia})

    # rechunk to avoid problems
    ds = ds.chunk({"time":10, "azimuth":-1, "range":-1})

#%% Discard possible erroneous ML values
    if "height_ml_new_gia" in ds_qvp_ra:
        ## First, filter out ML heights that are too high (above selected isotherm)
        isotherm = -1 # isotherm for the upper limit of possible ML values
        # we need to fill the nans of the TEMP qvp otherwise the argmin operation will fail
        ds_qvp_ra["TEMP"] = ds_qvp_ra["TEMP"].fillna(ds["TEMP"].median("azimuth", keep_attrs=True).assign_coords({"z": ds["z"].median("azimuth", keep_attrs=True)}).swap_dims({"range":"z"}))
        z_isotherm = ds_qvp_ra.TEMP.isel(z=((ds_qvp_ra["TEMP"].fillna(100.)-isotherm)**2).argmin("z").compute())["z"]

        # ds_qvp_ra.coords["height_ml_new_gia_clean"] = ds_qvp_ra["height_ml_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()
        ds_qvp_ra.coords["height_ml_bottom_new_gia_clean"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(ds_qvp_ra["height_ml_bottom_new_gia"]<=z_isotherm.values).compute()

        # Then, check that ML top is over ML bottom
        cond_top_over_bottom = ds_qvp_ra.coords["height_ml_new_gia"] > ds_qvp_ra.coords["height_ml_bottom_new_gia_clean"]

        # Assign final values
        ds_qvp_ra.coords["height_ml_new_gia_clean"] = ds_qvp_ra["height_ml_new_gia"].where(cond_top_over_bottom).compute()
        ds_qvp_ra.coords["height_ml_bottom_new_gia_clean"] = ds_qvp_ra["height_ml_bottom_new_gia_clean"].where(cond_top_over_bottom).compute()

        ds = ds.assign_coords({'height_ml_new_gia_clean': ds_qvp_ra["height_ml_new_gia_clean"]})
        ds = ds.assign_coords({'height_ml_bottom_new_gia_clean': ds_qvp_ra["height_ml_bottom_new_gia_clean"]})

#%% Attenuation correction (NOT PROVED THAT IT WORKS NICELY ABOVE THE ML)
    if X_PHI in ds.data_vars:
        # First we calculate atten correction only in rain, as backup
        ds = utils.attenuation_corr_linear(ds, alpha = 0.08, beta = 0.02, alphaml = 0, betaml = 0,
                                           dbzh=X_DBZH, zdr=["ZDR", "ZDR_EC", "ZDR_OC", "ZDR_EC_OC"],
                                           phidp=["UPHIDP_OC_MASKED", "UPHIDP_OC", "PHIDP_OC_MASKED", "PHIDP_OC"],
                                           ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                           temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                           interpolate_deltabump = True )

        ds = ds.rename({vv: vv+"_rain" for vv in ds.data_vars if "_AC" in vv})

        # Then we calculate the atten correction both in rain and the ML (this are the final values used)
        ds = utils.attenuation_corr_linear(ds, alpha = 0.08, beta = 0.02, alphaml = 0.16, betaml = 0.022,
                                           dbzh=X_DBZH, zdr=["ZDR", "ZDR_EC", "ZDR_OC", "ZDR_EC_OC"],
                                           phidp=["UPHIDP_OC_MASKED", "UPHIDP_OC", "PHIDP_OC_MASKED", "PHIDP_OC"],
                                           ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                           temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                           interpolate_deltabump = True )

        ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min,"SNRHC":SNRH_min, "SQIH":0.5})[[vv for vv in ds if "_AC" in vv]] )
        X_DBZH = X_DBZH+"_AC"
        X_ZDR = X_ZDR+"_AC"

#%% Fix KDP in the ML using PHIDP:
    if X_PHI in ds.data_vars:

        top_tolerance = 0
        bottom_tolerance = 0
        if country == "boxpol":
            top_tolerance = 0
            bottom_tolerance = 0

        # derive KDP from PHIDP (Vulpiani)
        if isinstance(winlen0, list):
            # if winlen0 is a list, use the first value (small window) for strong rain (SR, DBZH>40) and
            # use the second value (large window) for light rain (LR, DBZH<=40)
            ds_kdpSR_mlcorr = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen0[0], min_periods=max(3, int((winlen0[0] - 1) / 4)),
                                                      mlt="height_ml_new_gia_clean",
                                                      mlb="height_ml_bottom_new_gia_clean",
                                                      top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)[["KDP_ML_corrected"]]
            ds_kdpLR_mlcorr = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen0[1], min_periods=max(3, int((winlen0[1] - 1) / 4)),
                                                      mlt="height_ml_new_gia_clean",
                                                      mlb="height_ml_bottom_new_gia_clean",
                                                      top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)[["KDP_ML_corrected"]]
            ds_kdp_mlcorr = xr.where(ds[X_DBZH]>40,
                              ds_kdpSR_mlcorr, ds_kdpLR_mlcorr)
            ds = ds.assign(ds_kdp_mlcorr)
        else:
            ds = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen=winlen0, min_periods=max(3, int((winlen0 - 1) / 4)),
                                        mlt="height_ml_new_gia_clean",
                                        mlb="height_ml_bottom_new_gia_clean",
                                         top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)

        # Mask KDP_ML_correction with PHIDP_OC_MASKED
        ds["KDP_ML_corrected"] = ds["KDP_ML_corrected"].where(ds[X_PHI+"_MASKED"].notnull())

        ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min,"SNRHC":SNRH_min, "SQIH":0.5})["KDP_ML_corrected"]})

        # Correct KDP elevation dependency
        try:
            angle = float(ds.elevation.mean())
        except:
            angle = float(ds.sweep_fixed_angle.mean())
        try:
            ds = utils.kdp_elev_corr(ds, angle, kdp=["KDP_CONV", "KDP_ML_corrected"])
            ds_qvp_ra = utils.kdp_elev_corr(ds_qvp_ra, angle, kdp=["KDP_CONV", "KDP_ML_corrected"])
            X_KDP = "KDP_ML_corrected_EC"
        except:
            pass

#%% Classification of stratiform events based on entropy
    if X_PHI in ds.data_vars:

        # calculate linear values for ZH and ZDR
        ds = ds.assign({X_DBZH+"_lin": wrl.trafo.idecibel(ds[X_DBZH]), X_ZDR+"_lin": wrl.trafo.idecibel(ds[X_ZDR]) })

        # calculate entropy
        Entropy = utils.calculate_pseudo_entropy(utils.apply_min_max_thresh(ds, {X_DBZH:0, "SNRH":SNRH_min, "SNRHC":SNRH_min,"SQIH":0.5}, {}),
                                                 dim='azimuth', var_names=[X_DBZH+"_lin", X_ZDR+"_lin", X_RHO, "KDP_ML_corrected_EC"], n_lowest=60)

        # concate entropy for all variables and get the minimum value
        strati = xr.concat((Entropy["entropy_"+X_DBZH+"_lin"], Entropy["entropy_"+X_ZDR+"_lin"],
                            Entropy["entropy_"+X_RHO], Entropy["entropy_"+"KDP_ML_corrected_EC"]),"entropy")
        min_trst_strati = strati.min("entropy")

        # assign to datasets
        ds["min_entropy"] = min_trst_strati

        min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
        min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})

#%% Calculate retrievals
    if X_PHI in ds.data_vars:
        # Calculate attenuation with the ZPHI method, only used for one of the retrievals
        ds_zphi = utils.attenuation_corr_zphi(ds , alpha = 0.08, beta = 0.02,
                                  X_DBZH="DBZH", X_ZDR=["_".join(X_ZDR.split("_")[:-1]) if "_AC" in X_ZDR else X_ZDR][0],
                                  X_PHI=X_PHI+"_MASKED")

        ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(ds_zphi, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min,"SNRHC":SNRH_min, "SQIH":0.5})[["AH"]] )

        # filter out values close to the ground
        ds_qvp_ra = ds_qvp_ra.where(ds_qvp_ra["z"]>min_height)

        print("Calculating retrievals...")

        Lambda = 53.1
        if country == "boxpol":
            Lambda = 32

        retrievals = utils.calc_microphys_retrievals(ds_zphi, Lambda = Lambda, mu=0.33,
                                      X_DBZH=X_DBZH, X_ZDR=X_ZDR, X_KDP="KDP_ML_corrected_EC", X_TEMP="TEMP",
                                      X_PHI=X_PHI+"_MASKED"
                                      ) #!!! filter out -inf inf values

        # Save retrievals
        retrievals.encoding = {'zlib': True, 'complevel': 6}
        for vv in retrievals.data_vars:
            retrievals[vv].encoding = {'zlib': True, 'complevel': 6}
        if save_retrievals_ppi:
            retrievals_path = make_savedir(ff, "ppis_retrievals")
            retrievals.to_netcdf(retrievals_path)

        # add retrievals to QVP
        attach_vars = []
        for vv in [X_RHO, X_TH, X_ZDR, "SNRH", "SNRHC", "SQIH"]:
            if vv in ds_zphi: attach_vars.append(vv)
        ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(xr.merge([retrievals, ds_zphi[attach_vars]]), min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5})[[vv for vv in retrievals.data_vars]] )

#%% Save qvp
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

#%% Save PPI
    if save_processed_ppi:
        savepath_ppi = make_savedir(ff, "final_ppis")
        if os.path.exists(os.path.dirname(savepath_ppi)+"/DONE.txt") and not overwrite:
            continue

        for vv in ds.data_vars:
            # set the encoding, try to copy original encodings
            if ds[vv].dtype == "float" or ds[vv].dtype == "float32" or ds[vv].dtype == "float64":
                if len(ds[vv].encoding) == 0:
                    try:
                        enc = ds[vv.split("_")[0]].encoding.copy()
                        if len(enc) != 0:
                            ds[vv].encoding = enc.copy()
                        else:
                            ds[vv].encoding = {'zlib': True, 'complevel': 6}
                            if ds[vv].dims == ds["DBZH"].dims:
                                ds[vv].encoding.update({k: ds["DBZH"].encoding[k] for k in ("chunksizes", "preferred_chunks", "original_shape")})
                    except:
                        ds[vv].encoding = {'zlib': True, 'complevel': 6}
                        if ds[vv].dims == ds["DBZH"].dims:
                            ds[vv].encoding.update({k: ds["DBZH"].encoding[k] for k in ("chunksizes", "preferred_chunks", "original_shape")})

        ds.to_netcdf(savepath_ppi)

        with open( os.path.dirname(savepath_ppi)+'/DONE.txt', 'w') as f:
            f.write('')

#%% Save a text file to register that the work finished correctly
    with open( os.path.dirname(savepath)+'/DONE.txt', 'w') as f:
        f.write('')

#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")
