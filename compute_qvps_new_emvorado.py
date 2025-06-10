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

mom = 2 # use 1- or 2- moment scheme?
emv_wc = "*allsim_id*"
icon_wc = "*allsim_icon*"

# path0 = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/" # For testing
path0 = os.path.dirname(sys.argv[1])+"/" # read path from console
overwrite = True # overwrite existing files?

clowres0=False # this is for the ML detection algorithm
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
else:
    phase_proc_params = utils.phase_proc_params["dmi"] # get default phase processing parameters

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

#%% Load data

# check for the file DONE.txt in the savepath before starting
savepath = make_savedir(files_emv[0][:-25]+".nc", replace=("/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/", "/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/qvps/"))
if os.path.exists(os.path.dirname(savepath)+"/DONE.txt") and not overwrite:
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
                          ])

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

ds = swp

#%% Calculate microphysical variables
ds = utils.calc_microphys(ds, mom=mom)

#%% Compute QVP
## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
## Also added further filtering (TH>0, ZDR>-1)
ds_qvp_ra = utils.compute_qvp(ds, min_thresh={X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SNRHC":10, "SQIH":0.5} )

#%% Detect melting layer
if X_PHI in ds.data_vars:
    if ds.range.diff("range").median() > 750:
        clowres0=True # for the ML correction algorithm

    moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (0, 180)} # different RHOHV limits for EMVORADO output

    ds_qvp_ra = utils.melting_layer_qvp_X_new(ds_qvp_ra, moments=moments, dim="z", fmlh=0.3,
             xwin=xwin0, ywin=ywin0, min_h=0, rhohv_thresh_gia=(0.99, 1), all_data=True, clowres=clowres0)

    #### Assign ML values to dataset

    ds = ds.assign_coords({'height_ml': ds_qvp_ra.height_ml})
    ds = ds.assign_coords({'height_ml_bottom': ds_qvp_ra.height_ml_bottom})
    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia})

#%% Attach ERA5 temperature profile #DISABLED: ICON HAS TEMPERATURE
# loc = utils.find_loc_code(utils.locs_code, files[0])
# ds_qvp_ra = utils.attach_ERA5_TEMP(ds_qvp_ra, path=loc.join(era5_dir.split("loc")))

#%% Discard possible erroneous ML values
if "height_ml_new_gia" in ds_qvp_ra:
    ## First, filter out ML heights that are too high (above selected isotherm)
    isotherm = -1 # isotherm for the upper limit of possible ML values
    # we need to fill the nans of the TEMP qvp otherwise the argmin operation will fail
    ds_qvp_ra["TEMP"] = ds_qvp_ra["TEMP"].fillna(ds["TEMP"].median("azimuth", keep_attrs=True).assign_coords({"z": ds["z"].median("azimuth", keep_attrs=True)}).swap_dims({"range":"z"}))
    z_isotherm = ds_qvp_ra.TEMP.isel(z=((ds_qvp_ra["TEMP"]-isotherm)**2).argmin("z").compute())["z"]

    ds_qvp_ra.coords["height_ml_new_gia"] = ds_qvp_ra["height_ml_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()
    ds_qvp_ra.coords["height_ml_bottom_new_gia"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(ds_qvp_ra["height_ml_new_gia"]<=z_isotherm.values).compute()

    # Then, check that ML top is over ML bottom
    cond_top_over_bottom = ds_qvp_ra.coords["height_ml_new_gia"] > ds_qvp_ra.coords["height_ml_bottom_new_gia"]

    # Assign final values
    ds_qvp_ra.coords["height_ml_new_gia"] = ds_qvp_ra["height_ml_new_gia"].where(cond_top_over_bottom).compute()
    ds_qvp_ra.coords["height_ml_bottom_new_gia"] = ds_qvp_ra["height_ml_bottom_new_gia"].where(cond_top_over_bottom).compute()

    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia.where(cond_top_over_bottom)})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia.where(cond_top_over_bottom)})

#%% Classification of stratiform events based on entropy
if X_PHI in ds.data_vars:

    # calculate linear values for ZH and ZDR
    ds = ds.assign({X_DBZH+"_lin": wrl.trafo.idecibel(ds[X_DBZH]), X_ZDR+"_lin": wrl.trafo.idecibel(ds[X_ZDR]) })

    # calculate entropy
    Entropy = utils.calculate_pseudo_entropy(utils.apply_min_max_thresh(ds, {X_DBZH:0, "SNRH":10, "SNRHC":10,"SQIH":0.5}, {}),
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
    for att in ds_qvp_ra.attrs.keys():
        if type(ds_qvp_ra.attrs[att] is xr.DataArray):
            try:
                ds_qvp_ra.attrs[att] = ds_qvp_ra.attrs[att].values.flatten()[0]
                if att=="station_name":
                    ds_qvp_ra.attrs[att] = ds_qvp_ra.attrs[att].astype(str)
            except AttributeError:
                None
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
