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
save_retrievals_ppi = False # Save PPIs of microphysical retrievals? (this is pretty slow and uses substantial storage)

abs_zdr_off_min_thresh = 0. # if ZDR_OC has more negative values than the original ZDR
# and the absolute median offset is < abs_zdr_off_min_thresh, then undo the correction (set to 0 to avoid this step)
zdr_offset_perts = False # offset correct zdr per timesteps? if False, correct with daily offset
mix_zdr_offsets = True # if True and zdr_offset_perts=False, try to
# choose between daily LR-consistency and QVP offsets based on how_mix_zdr_offset.
# If True and zdr_offset_perts=True, choose between all available timestep offsets
# based on how_mix_zdr_offset. If False, just use the offsets according to the priority they are passed on.
how_mix_zdr_offset = "count" # how to choose between the different offsets
# if mix_zdr_offsets = True. "count" will choose the offset that has more data points
# in its calculation (there must be a variable ZDR_offset_datacount in the loaded offset).
# "neg_overcorr" will choose the offset that generates less negative ZDR values.

# Set the possible ZDR calibrations locations to include (in order of priority)
# The script will try to correct according to the first offset; if not available or nan it will
# continue with the next one, and so on. Only the used offset will be outputted in the final file.
# All items in zdrofffile will be tested in each zdroffdir to load the data.
zdroffdir = utils.zdroffdir
zdrofffile = utils.zdrofffile
if zdr_offset_perts:
    zdrofffile = utils.zdrofffile_ts
    zdr_oc_dict = {} # this dictionary is used later when loading all timestep-based offsets

# set the RHOHV correction location
rhoncdir = utils.rhoncdir  # subfolder where to find the noise corrected rhohv data
rhoncfile = utils.rhoncfile # pattern to select the appropriate file (careful with the rhohv_nc_2percent)

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
if "dwd" in path0 and "vol5minng01" in path0:
    clowres0=True
# Set specifics for each turkish radar
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

#%% Load noise corrected RHOHV if available
    try:
        rhoncpath = os.path.dirname(utils.edit_str(ff, country, country+rhoncdir))
        swp = utils.load_corrected_RHOHV(swp, rhoncpath+"/"+rhoncfile)

        # Check that the corrected RHOHV does not have higher STD than the original (1 + std_margin)
        # if that is the case we take it that the correction did not work well so we won't use it
        std_margin = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_margin), otherwise use RHOHV
        min_rho = 0.6 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results

        if ( swp["RHOHV_NC"].where(swp["RHOHV_NC"]>min_rho * (swp["z"]>min_height)).std() < swp[X_RHO].where(swp[X_RHO]>min_rho * (swp["z"]>min_height)).std()*(1+std_margin) ).compute():
            # Change the default RHOHV name to the corrected one
            X_RHO = "RHOHV_NC"

    except OSError:
        print("No noise corrected rhohv to load: "+rhoncpath+"/"+rhoncfile)

    except ValueError:
        print("ValueError with corrected rhohv: "+rhoncpath+"/"+rhoncfile)

#%% Load ZDR offset if available

    # We define a custom exception to stop the next nexted loops as soon as a file is loaded
    class FileFound(Exception):
        pass

    # Load the offsets
    try:
        # print("Loading ZDR offsets")
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
                        zdr_oc_dict[zdrod].append(utils.load_ZDR_offset(swp, X_ZDR, zdroffsetpath+"/"+zdrof, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True))
                        continue
                    else:
                        swp = utils.load_ZDR_offset(swp, X_ZDR, zdroffsetpath+"/"+zdrof, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True)

                    # if the offset comes from LR ZH-ZDR consistency, check it against
                    # the QVP method (if available) and choose the best one based on how_mix_zdr_offset
                    if "LR_consistency" in zdrod and mix_zdr_offsets:
                        for zdrof2 in zdrofffile:
                            try:
                                zdrod2 = [pp for pp in zdroffdir if "QVP" in pp][0]
                                zdroffsetpath_qvp = os.path.dirname(utils.edit_str(ff, country, country+zdrod2))
                                swp_qvpoc = utils.load_ZDR_offset(swp, X_ZDR, zdroffsetpath_qvp+"/"+zdrof2, zdr_oc_name=X_ZDR+"_OC", attach_all_vars=True)

                                if how_mix_zdr_offset == "neg_overcorr":
                                    # calculate the count of negative values after each correction
                                    neg_count_swp_lroc = (swp[X_ZDR+"_OC"].where((swp[X_RHO]>0.99) * (swp["z"]>min_height)) < 0).sum().compute()
                                    neg_count_swp_qvpoc = (swp_qvpoc[X_ZDR+"_OC"].where((swp_qvpoc[X_RHO]>0.99) * (swp["z"]>min_height)) < 0).sum().compute()

                                    if neg_count_swp_lroc > neg_count_swp_qvpoc:
                                        # continue with the correction with less negative values
                                        print("Changing daily ZDR offset from LR_consistency to QVP")
                                        swp = swp_qvpoc

                                elif how_mix_zdr_offset == "count":
                                    if "ZDR_offset_datacount" in swp and "ZDR_offset_datacount" in swp_qvpoc:
                                        # Choose the offset that has the most data points in its calculation
                                        if swp_qvpoc["ZDR_offset_datacount"] > swp["ZDR_offset_datacount"]:
                                            # continue with the correction with more data points
                                            print("Changing daily ZDR offset from LR_consistency to QVP")
                                            swp = swp_qvpoc
                                    else:
                                        print("how_mix_zdr_offset == 'count' not possible, ZDR_offset_datacount not present in all offset datasets.")


                                break
                            except (OSError, ValueError):
                                pass

                    # calculate the count of negative values before and after correction
                    neg_count_swp = (swp[X_ZDR].where((swp[X_RHO]>0.99) * (swp["z"]>min_height)) < 0).sum().compute()
                    neg_count_swp_oc = (swp[X_ZDR+"_OC"].where((swp[X_RHO]>0.99) * (swp["z"]>min_height)) < 0).sum().compute()

                    if neg_count_swp_oc > neg_count_swp and abs((swp[X_ZDR] - swp[X_ZDR+"_OC"]).compute().median()) < abs_zdr_off_min_thresh:
                        # if the correction introduces more negative values and the offset is lower than abs_zdr_off_min_thresh, then do not correct
                        swp[X_ZDR+"_OC"] = swp[X_ZDR]

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
                    # print("Merging valid ZDR offsets (timestep mode)")
                    if how_mix_zdr_offset == "neg_overcorr":
                        for final_zdr_ocn in final_zdr_oc_list[1:]:
                            # calculate the count of negative values for each correction
                            neg_count_final_zdr_oc = (final_zdr_oc.where((swp[X_RHO]>0.99) * (final_zdr_oc["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                            neg_count_final_zdr_ocn = (final_zdr_ocn.where((swp[X_RHO]>0.99) * (final_zdr_ocn["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
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
                neg_count_final_zdr_oc = (final_zdr_oc.where((swp[X_RHO]>0.99) * (final_zdr_oc["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                neg_count_final_zdr = (swp[X_ZDR].where((swp[X_RHO]>0.99) * (swp["z"]>min_height)) < 0).sum(("range", "azimuth")).compute()
                neg_count_final_cond = (neg_count_final_zdr_oc > neg_count_final_zdr) * (abs((swp[X_ZDR] - final_zdr_oc).compute().median(("range", "azimuth"))) < abs_zdr_off_min_thresh)

                # Set the final ZDR_OC and change the default ZDR name to the corrected one
                swp[X_ZDR+"_OC"] = final_zdr_oc.where(~neg_count_final_cond, swp[X_ZDR]).where(final_zdr_oc.notnull(), swp[X_ZDR])
                X_ZDR = X_ZDR+"_OC"

        else:
            # If no ZDR offset was loaded, print a message
            print("No zdr offset to load: "+zdroffsetpath+"/"+zdrof)
    except FileFound:
        pass


#%% Correct PHIDP
    ################## Before entropy calculation we need to use the melting layer detection algorithm
    ds = swp
    interpolation_method_ML = "linear" # for interpolating PHIDP in the ML

    # Check that PHIDP is in data, otherwise skip ML detection
    if X_PHI in ds.data_vars:
        # Set parameters according to data
        phase_proc_params = utils.get_phase_proc_params(ff) # get default phase processing parameters
        window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia, grad_thresh = phase_proc_params.values()

        ######### Processing PHIDP
        #### fix PHIDP

        # phidp may be already preprocessed (turkish case), then only offset-correct (no smoothing) and then vulpiani
        if "PHIDP" not in X_PHI: # This is now always skipped with this definition ("PHIDP" is in both X_PHI); i.e., we apply full processing to turkish data too
            # calculate phidp offset
            ds = utils.phidp_offset_correction(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                                 rng_min=1000, rng=rng, azmedian=azmedian, tolerance=(0,5)) # shorter rng, rng_min for finer turkish data

            phi_masked = ds[X_PHI+"_OC"].where((ds[X_RHO] >= 0.9) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

        else:
            # process phidp (offset and smoothing)
            ds = utils.phidp_processing(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                                 rng=rng, azmedian=azmedian, tolerance=(0,5))

            phi_masked = ds[X_PHI+"_OC_SMOOTH"].where((ds[X_RHO] >= 0.9) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

        # Assign phi_masked
        assign = { X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs) }

        ds = ds.assign(assign)

        # derive KDP from PHIDP (Vulpiani)

        ds = utils.kdp_phidp_vulpiani(ds, winlen0, X_PHI+"_OC_MASKED", min_periods=winlen0//2+1)

        X_PHI = X_PHI+"_OC" # continue using offset corrected PHI

    else:
        print(X_PHI+" not found in the data, skipping ML detection")

#%% Attach ERA5 variables
    era5_vars = ["temperature", "relative_humidity"]
    era5_vars_rename = {"t":"TEMP", "r":"RH"}
    ds = utils.attach_ERA5_fields(ds, path=era5_dir, convert_to_C=True,
                           variables=era5_vars,
                           rename=era5_vars_rename, set_as_coords=False,
                           k_n=1, pre_interpolate_z=True)

    # Save ERA5 ppis
    for vv in era5_vars_rename.values():
        ds[vv].encoding = {'zlib': True, 'complevel': 6}
    era5_ppis_path = make_savedir(ff, "ppis_era5")
    ds[[vv for vv in era5_vars_rename.values()]].to_netcdf(era5_ppis_path)

#%% Compute QVP
    ## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
    ## Also added further filtering (TH>0, ZDR>-1)
    ds_qvp_ra = utils.compute_qvp(ds, min_thresh={X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SNRHC":10, "SQIH":0.5} )

    # filter out values close to the ground
    ds_qvp_ra2 = ds_qvp_ra.where(ds_qvp_ra["z"]>min_height)

#%% Detect melting layer
    if X_PHI in ds.data_vars:
        if country=="dwd":
            moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}
        elif country=="dmi":
            moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}

        ds_qvp_ra = utils.melting_layer_qvp_X_new(ds_qvp_ra2, moments=moments, dim="z", fmlh=0.3, grad_thresh=grad_thresh,
                 xwin=xwin0, ywin=ywin0, min_h=min_height, rhohv_thresh_gia=rhohv_thresh_gia, all_data=True, clowres=clowres0)

        #### Assign ML values to dataset

        ds = ds.assign_coords({'height_ml': ds_qvp_ra.height_ml})
        ds = ds.assign_coords({'height_ml_bottom': ds_qvp_ra.height_ml_bottom})
        ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia})
        ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia})

#%% Discard possible erroneous ML values
    if "height_ml_new_gia" in ds_qvp_ra:
        ## First, filter out ML heights that are too high (above selected isotherm)
        isotherm = -1 # isotherm for the upper limit of possible ML values
        # we need to fill the nans of the TEMP qvp otherwise the argmin operation will fail
        ds_qvp_ra["TEMP"] = ds_qvp_ra["TEMP"].fillna(ds["TEMP"].median("azimuth", keep_attrs=True).assign_coords({"z": ds["z"].median("azimuth", keep_attrs=True)}).swap_dims({"range":"z"}))
        z_isotherm = ds_qvp_ra.TEMP.isel(z=((ds_qvp_ra["TEMP"].fillna(100.)-isotherm)**2).argmin("z").compute())["z"]

        ds_qvp_ra.coords["height_ml_new_gia"] = ds_qvp_ra["height_ml_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()
        ds_qvp_ra.coords["height_ml_bottom_new_gia"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()

        # Then, check that ML top is over ML bottom
        cond_top_over_bottom = ds_qvp_ra.coords["height_ml_new_gia"] > ds_qvp_ra.coords["height_ml_bottom_new_gia"]

        # Assign final values
        ds_qvp_ra.coords["height_ml_new_gia"] = ds_qvp_ra["height_ml_new_gia"].where(cond_top_over_bottom).compute()
        ds_qvp_ra.coords["height_ml_bottom_new_gia"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(cond_top_over_bottom).compute()

        ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia.where(cond_top_over_bottom)})
        ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia.where(cond_top_over_bottom)})

#%% Attenuation correction (NOT PROVED THAT IT WORKS NICELY ABOVE THE ML)
    if X_PHI in ds.data_vars:
        ds = utils.attenuation_corr_linear(ds, alpha = 0.08, beta = 0.02, alphaml = 0.16, betaml = 0.022,
                                           dbzh=X_DBZH, zdr=["ZDR_OC", "ZDR"], phidp=[X_PHI],
                                           ML_bot = "height_ml_bottom_new_gia", ML_top = "height_ml_new_gia",
                                           temp = "TEMP", temp_mlbot = 3, temp_mltop = 0, z_mlbot = 2000, dz_ml = 500,
                                           interpolate_deltabump = True )

        ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})[[vv for vv in ds if "_AC" in vv]] )

#%% Fix KDP in the ML using PHIDP:
    if X_PHI in ds.data_vars:

        ds = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen=winlen0, min_periods=winlen0//2+1)

        ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})["KDP_ML_corrected"]})

#%% Classification of stratiform events based on entropy
    if X_PHI in ds.data_vars:

        # calculate linear values for ZH and ZDR
        ds = ds.assign({"DBZH_lin": wrl.trafo.idecibel(ds[X_DBZH]), "ZDR_lin": wrl.trafo.idecibel(ds[X_ZDR]) })

        # calculate entropy
        Entropy = utils.calculate_pseudo_entropy(ds.where(ds[X_DBZH]>0), dim='azimuth', var_names=["DBZH_lin", "ZDR_lin", X_RHO, "KDP_ML_corrected"], n_lowest=60) #!!! is 60 ok?

        # concate entropy for all variables and get the minimum value
        strati = xr.concat((Entropy.entropy_DBZH_lin, Entropy.entropy_ZDR_lin, Entropy["entropy_"+X_RHO], Entropy.entropy_KDP_ML_corrected),"entropy")
        min_trst_strati = strati.min("entropy")

        # assign to datasets
        ds["min_entropy"] = min_trst_strati

        min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
        min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
        ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})

#%% Calculate retrievals
    if X_PHI in ds.data_vars:
        print("Calculating retrievals...")
        Lambda = 53.1 # radar wavelength in mm (pro: 53.138, ANK: 53.1, AFY: 53.3, GZT: 53.3, HTY: 53.3, SVS:53.3)
        X_KDP = "KDP_ML_corrected"

        # LWC
        lwc_zh_zdr = 10**(0.058*ds[X_DBZH] - 0.118*ds[X_ZDR] - 2.36) # Reimann et al 2021 eq 3.7 (adjusted for Germany)
        lwc_zh_zdr2 = 1.38*10**(-3) *10**(0.1*ds[X_DBZH] - 2.43*ds[X_ZDR] + 1.12*ds[X_ZDR]**2 - 0.176*ds[X_ZDR]**3 ) # used in S band, Ryzhkov 2022 PROM presentation https://www2.meteo.uni-bonn.de/spp2115/lib/exe/fetch.php?media=internal:uploads:all_hands_schneeferner_july2022:ryzhkov.pdf
        lwc_kdp = 10**(0.568*np.log10(ds[X_KDP]) + 0.06) # Reimann et al 2021(adjusted for Germany)

        # IWC (Collected from Blanke et al 2023)
        iwc_zh_t = 10**(0.06 * ds[X_DBZH] - 0.0197*ds["TEMP"] - 1.7) # empirical from Hogan et al 2006 Table 2

        iwc_zdr_zh_kdp = xr.where(ds[X_ZDR]>=0.4, # Carlin et al 2021 eqs 4b and 5b
                                  4*10**(-3)*( ds[X_KDP]*Lambda/( 1-wrl.trafo.idecibel(ds[X_ZDR])**-1 ) ),
                                  0.033 * ( ds[X_KDP]*Lambda )**0.67 * wrl.trafo.idecibel(ds[X_DBZH])**0.33 )

        # Dm (ice collected from Blanke et al 2023)
        Dm_ice_zh = 1.055*wrl.trafo.idecibel(ds[X_DBZH])**0.271 # Matrosov et al. (2019) Fig 10 (S band)
        Dm_ice_zh_kdp = 0.67*( wrl.trafo.idecibel(ds[X_DBZH])/(ds[X_KDP]*Lambda) )**(1/3) # Ryzhkov and Zrnic (2019). Idk exactly where does the 0.67 approximation comes from, Blanke et al. 2023 eq 10 and Carlin et al 2021 eq 5a cite Bukovčić et al. (2018, 2020) but those two references do not show this formula.
        Dm_ice_zdp_kdp = -0.1 + 2*( (wrl.trafo.idecibel(ds[X_DBZH])*(1-wrl.trafo.idecibel(ds[X_ZDR])**-1 ) ) / (ds[X_KDP]*Lambda) )**(1/2) # Ryzhkov and Zrnic (2019). Zdp = Z(1-ZDR**-1) from Carlin et al 2021

        Dm_rain_zdr = 0.3015*ds[X_ZDR]**3 - 1.2087*ds[X_ZDR]**2 + 1.9068*ds[X_ZDR] + 0.5090 # (for rain but tuned for Germany X-band, JuYu Chen, Zdr in dB, Dm in mm)

        D0_rain_zdr2 = 0.171*ds[X_ZDR]**3 - 0.725*ds[X_ZDR]**2 + 1.48*ds[X_ZDR] + 0.717 # (D0 from Hu and Ryzhkov 2022, used in S band data but could work for C band) [mm]
        D0_rain_zdr3 = xr.where(ds[X_ZDR]<1.25, # D0 from Bringi et al 2009 (C-band) eq. 1 [mm]
                                0.0203*ds[X_ZDR]**4 - 0.1488*ds[X_ZDR]**3 + 0.2209*ds[X_ZDR]**2 + 0.5571*ds[X_ZDR] + 0.801,
                                0.0355*ds[X_ZDR]**3 - 0.3021*ds[X_ZDR]**2 + 1.0556*ds[X_ZDR] + 0.6844
                                )
        mu = 0
        Dm_rain_zdr2 = D0_rain_zdr2 * (4+mu)/(3.67+mu) # conversion from D0 to Dm according to eq 4 of Hu and Ryzhkov 2022.
        Dm_rain_zdr3 = D0_rain_zdr3 * (4+mu)/(3.67+mu)

        # log(Nt)
        Nt_ice_zh_iwc = (3.39 + 2*np.log10(iwc_zh_t) - 0.1*ds[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2 = (3.69 + 2*np.log10(iwc_zh_t) - 0.1*ds[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_ice_zh_iwc_kdp = (3.39 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*ds[X_DBZH]) # (Hu and Ryzhkov 2022 eq. 10, [log(1/L)]
        Nt_ice_zh_iwc2_kdp = (3.69 + 2*np.log10(iwc_zdr_zh_kdp) - 0.1*ds[X_DBZH]) # Carlin et al 2021 eq. 7 originally in [log(1/m3)], transformed units here to [log(1/L)] by subtracting 3
        Nt_rain_zh_zdr = ( -2.37 + 0.1*ds[X_DBZH] - 2.89*ds[X_ZDR] + 1.28*ds[X_ZDR]**2 - 0.213*ds[X_ZDR]**3 )# Hu and Ryzhkov 2022 eq. 3 [log(1/L)]

        # Put everything together
        retrievals = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
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
                                })#.compute()

        # Save retrievals
        retrievals.encoding = {'zlib': True, 'complevel': 6}
        for vv in retrievals.data_vars:
            retrievals[vv].encoding = {'zlib': True, 'complevel': 6}
        if save_retrievals_ppi:
            retrievals_path = make_savedir(ff, "ppis_retrievals")
            retrievals.to_netcdf(retrievals_path)

        # add retrievals to QVP
        attach_vars = []
        for vv in [X_RHO, X_TH, X_ZDR, "SNRH", "SQIH"]:
            if vv in ds: attach_vars.append(vv)
        ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(xr.merge([retrievals, ds[attach_vars]]), min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})[[vv for vv in retrievals.data_vars]] )

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

#%% Save a text file to register that the work finished correctly
    with open( os.path.dirname(savepath)+'/DONE.txt', 'w') as f:
        f.write('')

#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")
