#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:48:08 2024

@author: jgiles

This script takes ICON+EMVORADO data and computes the ML detection algorithm and
entropy values for event classification, then generates QVPs including
temperature profiles and saves to nc files. All data given by the specified
path is loaded and processed at the same time.

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
# The files are collected from the same directory. Set the wildcards and other parameters here.

save_processed_ppi = True # Save PPIs after processing?

mom = 2 # use 1- or 2- moment scheme?
emv_wc = "*allsim_id*"
icon_wc = "*allsim_icon*"

# path0 = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/" # For testing
path0 = os.path.dirname(sys.argv[1])+"/" # read path from console
overwrite = False # overwrite existing files?

qvp_ielev=7 # elevation index to use for QVP

# get the files and check that it is not empty
files_emv = sorted(glob.glob(path0+emv_wc))
files_icon = sorted(glob.glob(path0+icon_wc))

if len(files_emv)==0:
    print("No EMVORADO files meet the selection criteria.")
    sys.exit("No EMVORADO files meet the selection criteria.")
if len(files_icon)==0:
    print("No ICON volume files meet the selection criteria.")
    sys.exit("No ICON volume files meet the selection criteria.")
if len(files_emv) != len(files_icon):
    warnings.warn("Different number of EMVORADO and ICON volume files, some timesteps will not be included.")

# ERA5 folder
if os.path.exists("/automount/ags/jgiles/ERA5/hourly/"):
    # then we are in local system
    era5_dir = "/automount/ags/jgiles/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/scratch/detectrea/giles1/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/scratch/detectrea/giles1/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below
elif os.path.exists("/p/largedata2/detectdata/projects/A04/ERA5/hourly/"):
    # then we are in JSC
    era5_dir = "/p/largedata2/detectdata/projects/A04/ERA5/hourly/loc/pressure_level_vars/" # dummy loc placeholder, it gets replaced below

# names of variables
phidp_names = ["PHIDP"] # names to look for the PHIDP variable, in order of preference
dbzh_names = ["DBZH_AC", "DBZH"] # same but for DBZH
rhohv_names = ["RHOHV"] # same but for RHOHV
zdr_names = ["ZDR_AC", "ZDR"]
th_names = ["TH", "DBTH", "DBZH"]

# default processing parameters
loc_id = files_emv[0].split("_id-")[1][0:6]
if loc_id in ["010392", "010356", "010832"]: # if dwd location
    phase_proc_params = utils.phase_proc_params["dwd"]["vol5minng01"] # get default phase processing parameters
    country = "dwd"
else:
    phase_proc_params = utils.phase_proc_params["dmi"] # get default phase processing parameters
    country = "dmi"

window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia, grad_thresh = phase_proc_params.values()

# define a function to create save directory and return file save path
def make_savedir(ff, replace=("/run/", "/run/qvps/")):
    """
    ff: filepath of the original file
    replace: part of ff to replace to create the new path
    """
    ff_parts = ff.split(replace[0])
    savepath = (replace[1]).join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    return savepath

# PHIDP processing / KDP calc parameters
window0max = 25 # max value for window0 (only applied if window0 is given in meters)
winlen0max = [9, 25] # max value for winlen0 (only applied if winlen0 is given in meters)
SNRH_min = 15 # min value for SNRH thresholding. This has a significant influence in the KDP calculation and also affects the QVPs computations.

min_hgts = utils.min_hgts
min_rngs = utils.min_rngs
min_hgt = min_hgts["default"] # minimum height above the radar to be considered
min_range = min_rngs["default"] # minimum range from which to consider data (mostly for bad PHIDP filtering)


#%% Load data

# check for the file DONE.txt in the savepath before starting
savepath = make_savedir(files_emv[0][:-25]+".nc", replace=("/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/", "/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/qvps/"))
if os.path.exists(os.path.dirname(savepath)+"/DONE.txt") and not overwrite:
    print("Files already exist, skipping: "+savepath)
    exit()

print("processing "+path0)

vol_emvorado_sim = utils.load_emvorado_to_radar_volume(files_emv, rename=True)

vol_icon_sim = xr.open_mfdataset(files_icon)

if "AHPI" in vol_emvorado_sim:
    vol_emvorado_sim["DBZH_AC"] = vol_emvorado_sim["DBZH"] + vol_emvorado_sim["AHPI"]
    vol_emvorado_sim["DBZH_AC"].attrs = vol_emvorado_sim["DBZH"].attrs
    for key in ["Description", "long_name"]:
        if key in vol_emvorado_sim["DBZH_AC"].attrs:
            vol_emvorado_sim["DBZH_AC"].attrs[key] = vol_emvorado_sim["DBZH_AC"].attrs[key] + " corrected for attenuation"

if "ADPPI" in vol_emvorado_sim:
    vol_emvorado_sim["ZDR_AC"] = vol_emvorado_sim["ZDR"] + vol_emvorado_sim["ADPPI"]
    vol_emvorado_sim["ZDR_AC"].attrs = vol_emvorado_sim["ZDR"].attrs
    for key in ["Description", "long_name"]:
        if key in vol_emvorado_sim["ZDR_AC"].attrs:
            vol_emvorado_sim["ZDR_AC"].attrs[key] = vol_emvorado_sim["ZDR_AC"].attrs[key] + " corrected for attenuation"

data = xr.merge([
                vol_emvorado_sim.isel({"sweep_fixed_angle":qvp_ielev}),
                vol_icon_sim.isel({"sweep_fixed_angle":qvp_ielev}),
                          ], compat='no_conflicts')

for coord in ["latitude", "longitude", "altitude", "elevation"]:
    if "time" in data[coord].dims:
        data.coords[coord] = data.coords[coord].mean()

#%% Georeference
swp = data.pipe(wrl.georef.georeference)

#%% Check variable names and add corrections and calibrations

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

#%% Correct ZDR-elevation dependency
try:
    angle = float(swp.elevation.mean())
except:
    angle = float(swp.sweep_fixed_angle.mean())
try:
    swp = utils.zdr_elev_corr(swp, angle, zdr=zdr_names)
    X_ZDR = X_ZDR+"_EC"
except:
    pass

ds = swp

#%% Correct PHIDP
################## Before entropy calculation we need to use the melting layer detection algorithm
interpolation_method_ML = "linear" # for interpolating PHIDP in the ML
min_height = min_hgt+swp["altitude"].values

# Check that PHIDP is in data, otherwise skip ML detection
if X_PHI in ds.data_vars:
    # Set parameters according to data
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
        ds_phiproc = utils.phidp_offset_correction(ds,
                                                   additional_thresholds=[{"SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5}, {}],
                                           X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range,
                             rng_min=1000, rng=rng, azmedian=azmedian, tolerance=(0,5)) # shorter rng, rng_min for finer turkish data

        phi_masked = ds_phiproc[X_PHI+"_OC"].where((ds[X_RHO] >= 0.8) * (ds[X_DBZH] >= 0.) * (ds["range"]>min_range) )

    else:
        # process phidp (offset and smoothing)
        ds_phiproc = utils.phidp_processing(ds,
                                additional_thresholds=[{"SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5}, {}],
                                    X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=min_height, window=window0, window2=3, fix_range=fix_range,
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

else:
    print(X_PHI+" not found in the data, skipping ML detection")


#%% Compute QVP
## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
## Also added further filtering (TH>0, ZDR>-1)
## Compared to QVPs of measured data, we don't consider here the CBB since we ran EMVORADO with the pencil beam approximation (lsmooth=.FALSE.)
ds_qvp_ra, ds_qvp_ra_count = utils.compute_qvp(ds.reset_coords("gr"),
                                               min_thresh={X_RHO:0.7, X_TH:0, X_ZDR:-1,
                                              "SNRH":SNRH_min, "SNRHC":SNRH_min, "SQIH":0.5},
                              output_count=True)

# assign DBZH counts
ds_qvp_ra = ds_qvp_ra.assign({"DBZH_qvp_count": ds_qvp_ra_count["DBZH"]})

#%% Detect melting layer
if X_PHI in ds.data_vars:
    if ds.range.diff("range").median() > 750:
        clowres0=True # for the ML correction algorithm

    moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (0, 180)} # different RHOHV limits for EMVORADO output

    ds_qvp_ra_ml = utils.melting_layer_qvp_X_new(ds_qvp_ra.where(ds_qvp_ra_count>20),
                                              moments=moments, dim="z", fmlh=0.3,
                     xwin=xwin0, ywin=ywin0, min_h=min_height,
                     rhohv_thresh_gia=(0.995, 1), # Custom rhohv_thresh_gia since RHOHV is so high in the simulations
                     all_data=True, clowres=clowres0)

    #### Assign ML values to dataset
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml': ds_qvp_ra_ml.height_ml})
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom': ds_qvp_ra_ml.height_ml_bottom})
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_new_gia': ds_qvp_ra_ml.height_ml_new_gia})
    ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra_ml.height_ml_bottom_new_gia})

    ds = ds.assign_coords({'height_ml': ds_qvp_ra_ml.height_ml})
    ds = ds.assign_coords({'height_ml_bottom': ds_qvp_ra_ml.height_ml_bottom})
    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra_ml.height_ml_new_gia})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra_ml.height_ml_bottom_new_gia})

#%% Attach ERA5 temperature profile #DISABLED: ICON HAS TEMPERATURE
# loc = utils.find_loc_code(utils.locs_code, files[0])
# ds_qvp_ra = utils.attach_ERA5_TEMP(ds_qvp_ra, path=loc.join(era5_dir.split("loc")))

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
# Analogous to the empirical corrections
if X_PHI+"_OC" in ds.data_vars:
    # Set the correction coefficients:
    # Defaults (for C band):
    alpha = alphaml = 0.08
    beta = betaml = 0.02

    # get specific coefficients according to country:
    try:
        alpha, beta, alphaml, betaml = utils.attenuation_corr_linear_coefs[country].values()
    except:
        warnings.warn("Attenuation correction: specific attenuation correction coefficients could not be determined for country="+country+". Using default coefficients for C-band")

    # First we calculate atten correction only in rain, as backup
    ds_acrain = utils.attenuation_corr_linear(ds, alpha = alpha, beta = beta, alphaml = 0, betaml = 0,
                                       dbzh="DBZH",
                                       zdr=["ZDR_EC"],
                                       phidp=["PHIDP_OC_MASKED"],
                                       ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                       temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                       interpolate_deltabump = True )

    ds_acrain = ds_acrain.rename({vv: vv+"emp_rain" for vv in ds_acrain.data_vars if vv in ["DBZH_AC", "ZDR_EC_AC"]})
    ds = ds.assign(ds_acrain[["DBZH_ACemp_rain", "ZDR_EC_ACemp_rain"]])

    # Then we calculate the atten correction both in rain and the ML (this are the final values used)
    ds_ac = utils.attenuation_corr_linear(ds, alpha = alpha, beta = beta, alphaml = alphaml, betaml = betaml,
                                       dbzh="DBZH",
                                       zdr=["ZDR_EC"],
                                       phidp=["PHIDP_OC_MASKED"],
                                       ML_bot = "height_ml_bottom_new_gia_clean", ML_top = "height_ml_new_gia_clean",
                                       temp = "TEMP", temp_mlbot = 3, temp_mltop = -1, z_mlbot = 2000, dz_ml = 500,
                                       interpolate_deltabump = True )

    ds_ac = ds_ac.rename({vv: vv+"emp" for vv in ds_ac.data_vars if vv in ["DBZH_AC", "ZDR_EC_AC"]})
    ds = ds.assign(ds_ac[["DBZH_ACemp", "ZDR_EC_ACemp"]])

    ds_qvp_ra = ds_qvp_ra.assign( utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min,"SNRHC":SNRH_min, "SQIH":0.5})[[vv for vv in ds if "_ACemp" in vv]] )

#%% Fix KDP in the ML using PHIDP:
if X_PHI+"_OC" in ds.data_vars:

    top_tolerance = 0
    bottom_tolerance = 0
    if country == "boxpol":
        top_tolerance = 0
        bottom_tolerance = 0

    # derive KDP from PHIDP (Vulpiani)
    if isinstance(winlen0, list):
        # if winlen0 is a list, use the first value (small window) for strong rain (SR, DBZH>40) and
        # use the second value (large window) for light rain (LR, DBZH<=40)
        ds_kdpSR_mlcorr = utils.KDP_ML_correction(ds, X_PHI+"_OC_MASKED", winlen0[0], min_periods=max(3, int((winlen0[0] - 1) / 4)),
                                                  mlt="height_ml_new_gia_clean",
                                                  mlb="height_ml_bottom_new_gia_clean",
                                                  top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)[["KDP_ML_corrected"]]
        ds_kdpLR_mlcorr = utils.KDP_ML_correction(ds, X_PHI+"_OC_MASKED", winlen0[1], min_periods=max(3, int((winlen0[1] - 1) / 4)),
                                                  mlt="height_ml_new_gia_clean",
                                                  mlb="height_ml_bottom_new_gia_clean",
                                                  top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)[["KDP_ML_corrected"]]
        ds_kdp_mlcorr = xr.where(ds[X_DBZH]>40,
                          ds_kdpSR_mlcorr, ds_kdpLR_mlcorr)
        ds = ds.assign(ds_kdp_mlcorr)
    else:
        ds = utils.KDP_ML_correction(ds, X_PHI+"_OC_MASKED", winlen=winlen0, min_periods=max(3, int((winlen0 - 1) / 4)),
                                    mlt="height_ml_new_gia_clean",
                                    mlb="height_ml_bottom_new_gia_clean",
                                     top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)

    # Mask KDP_ML_correction with PHIDP_OC_MASKED
    ds["KDP_ML_corrected"] = ds["KDP_ML_corrected"].where(ds[X_PHI+"_OC_MASKED"].notnull())

    ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":SNRH_min,"SNRHC":SNRH_min, "SQIH":0.5})["KDP_ML_corrected"]})

    # Correct KDP elevation dependency
    try:
        angle = float(ds.elevation.mean())
    except:
        angle = float(ds.sweep_fixed_angle.mean())
    try:
        ds = utils.kdp_elev_corr(ds, angle, kdp=["KDP_CONV", "KDP_ML_corrected"])
        ds_qvp_ra = utils.kdp_elev_corr(ds_qvp_ra, angle, kdp=["KDP_CONV", "KDP_ML_corrected"])
    except:
        pass

#%% Classification of stratiform events based on entropy
if X_PHI in ds.data_vars:

    # calculate linear values for ZH and ZDR
    ds = ds.assign({X_DBZH+"_lin": wrl.trafo.idecibel(ds[X_DBZH]), X_ZDR+"_lin": wrl.trafo.idecibel(ds[X_ZDR]) })

    # calculate entropy
    Entropy = utils.calculate_pseudo_entropy(utils.apply_min_max_thresh(ds, {X_DBZH:0, "SNRH":SNRH_min, "SNRHC":SNRH_min,"SQIH":0.5}, {}),
                                             dim='azimuth', var_names=[X_DBZH+"_lin", X_ZDR+"_lin", X_RHO, "KDP"], n_lowest=60)

    # concate entropy for all variables and get the minimum value
    strati = xr.concat((Entropy["entropy_"+X_DBZH+"_lin"], Entropy["entropy_"+X_ZDR+"_lin"],
                        Entropy["entropy_"+X_RHO], Entropy["entropy_"+"KDP"]),"entropy")
    min_trst_strati = strati.min("entropy")

    # assign to datasets
    ds["min_entropy"] = min_trst_strati

    min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
    min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
    ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})


#%% Save dataset
# we need to change the type of some arrays for this to work
# Define attribute cleanup helper
def clean_xr_attrs(dataset):
    """Cleans dataset attributes to ensure NetCDF serialization compatibility."""
    # Wrap .keys() in list() to avoid dictionary size changed during iteration errors
    for att in list(dataset.attrs.keys()):
        if isinstance(dataset.attrs[att], xr.DataArray):
            try:
                # .compute() forces Dask to evaluate the array before extracting the value
                val = dataset.attrs[att].compute().values.flatten()[0]
                if att == "station_name":
                    # Handle byte-strings if present
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    dataset.attrs[att] = str(val)
                else:
                    dataset.attrs[att] = val
            except (AttributeError, IndexError):
                pass
    return dataset

# save file
ds_qvp_ra = clean_xr_attrs(ds_qvp_ra)
ds_qvp_ra.to_netcdf(savepath)

#%% Save PPI
if save_processed_ppi:
    savepath_ppi = make_savedir(files_emv[0][:-25]+".nc", replace=("/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/", "/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/final_ppis/"))
    if os.path.exists(os.path.dirname(savepath_ppi)+"/DONE.txt") and not overwrite:
        print("Files already exist, skipping: "+savepath)
        exit()

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

    # Apply the exact same cleanup to the PPI dataset before saving
    ds = clean_xr_attrs(ds)
    ds.to_netcdf(savepath_ppi)

    with open( os.path.dirname(savepath_ppi)+'/DONE.txt', 'w') as f:
        f.write('')

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
