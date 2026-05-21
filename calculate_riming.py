#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:51:46 2026

@author: jgiles
"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 2.0.2 !! (OR GREATER)

import numpy as np
import glob
import xarray as xr
import time
import onnxruntime as ort

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator



import warnings
warnings.filterwarnings('ignore')

# we define a funtion to look for loc inside a path string
def find_loc(locs, path):
    components = path.split(os.path.sep)
    for element in locs:
        for component in components:
            if element.lower() in component.lower():
                return element
    return None

locs = ["boxpol", "pro", "tur", "umd", "afy", "ank", "gzt", "hty", "svs"]

realpep_path = "/automount/realpep/"

suffix_name = "" # suffix to add to folder names, for special cases (like computing only a selection of cases)

print("!!!! Future proofing: maybe adding guard is needed for multiprocessing !!!!")
#if __name__ == "__main__": # set guard

#%% Load QVPs for stratiform-case CFTDs

# This part should be run after having the QVPs computed (compute_qvps.py)
start_time = time.time()
print("Loading QVPs...")

#### Get QVP file list
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_singlefile/ML_detected/pro/vol5minng01/07/*allmoms*"
# Load only events with ML detected (pre-condition for stratiform)
path_qvps = realpep_path+"/upload/jgiles/dwd/qvps/20*/*/*/pro/vol5minng01/07/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dwd/qvps_selected_for_emvorado/20*/*/*/pro/vol5minng01/07/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/dmi/qvps/20*/*/*/ANK/*/*/ML_detected.txt"
# path_qvps = realpep_path+"/upload/jgiles/boxpol/qvps/20*/*/2017-07-25/n_ppi_110deg/ML_detected.txt"

#!!! IMPORTANT !!!!
# Date 2020-10-19 for AFY at 7.0 elevation has some issue that triggers an error in the ZDR_EC_OC_AC variable
# IndexError: index 255 is out of bounds for axis 0 with size 240
# ValueError: Array chunk size or shape is unknown. Possible solution with x.compute_chunk_sizes()
# Remove it (I leave this here for future reference)

#### Set variable names
X_DBZH = "DBZH_AC"
X_RHO = "RHOHV_NC" # if RHOHV_NC is set here, it is then checked agains the original RHOHV in the next cell
X_ZDR = "ZDR_EC_OC_AC"
# X_KDP = "KDP_ML_corrected_EC"
# X_PHI = "UPHIDP_OC_MASKED"

if "dwd" in path_qvps:
    country="dwd"
    X_TH = "TH"
if "boxpol" in path_qvps:
    country="boxpol"
    X_TH = "DBTH"
    X_PHI = "PHIDP_OC_MASKED"
elif "dmi" in path_qvps:
    country="dmi"
    X_TH = "DBZH"
    X_PHI = "PHIDP_OC_MASKED"


ff_glob = glob.glob(path_qvps)

if "dmi" in path_qvps:
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
            closest_elevation_path = min(elevations, key=lambda x: abs(x[0] - 10.1))[1] # We use 10.1 to prefer elevation 12 instead of 8 if both available
            result_paths.append(closest_elevation_path)

        return result_paths

    ff_glob = get_closest_elevation(ff_glob)

try:
    ff = [glob.glob(os.path.dirname(fp)+"/*allm*")[0] for fp in ff_glob ]
except IndexError:
    ff = [glob.glob(os.path.dirname(fp)+"/*12345*")[0] for fp in ff_glob ]

alignz = False
if "dwd" in path_qvps: alignz = True
qvps = utils.load_qvps(ff, align_z=alignz, fix_TEMP=False, fillna=False)

# Move TEMP to coordinate
if "TEMP" not in qvps.coords:
    qvps = qvps.set_coords("TEMP")

# Check that RHOHV_NC is actually better (less std) than RHOHV, otherwise just use RHOHV, on a per-day basis
std_tolerance = 0.15 # std(RHOHV_NC) must be < (std(RHOHV))*(1+std_tolerance), otherwise use RHOHV
min_rho = 0.7 # min RHOHV value for filtering. Only do this test with the highest values to avoid wrong results
mean_tolerance = 0.02 # 2% tolerance, for checking if RHOHV_NC is actually larger than RHOHV (overall higher values)

if "_NC" in X_RHO:
    # Check that the corrected RHOHV does not have higher STD than the original (1 + std_tolerance)
    # if that is the case we take it that the correction did not work well so we won't use it
    cond_rhohv1 = (
                    qvps[X_RHO].where(qvps[X_RHO]>min_rho).resample({"time":"D"}).std(dim=("time", "z")) < \
                    qvps["RHOHV"].where(qvps["RHOHV"]>min_rho).resample({"time":"D"}).std(dim=("time", "z"))*(1+std_tolerance)
                    ).compute()

    # Check that the corrected RHOHV have overall higher mean than the original (1 - mean_tolerance)
    # if that is the case we take it that the correction did not work well so we won't use it
    cond_rhohv2 = ( qvps[X_RHO].resample({"time":"D"}).mean(dim=("time", "z")) > \
                   qvps["RHOHV"].resample({"time":"D"}).mean(dim=("time", "z"))*(1-mean_tolerance) ).compute()

    cond_rhohv = cond_rhohv1 * cond_rhohv2

    # create an xarray.Dataarray with the valid timesteps
    valid_dates = cond_rhohv.where(cond_rhohv, drop=True).time.dt.date
    valid_datetimes = [date.values in valid_dates for date in qvps.time.dt.date]
    valid_datetimes_xr = xr.DataArray(valid_datetimes, coords={"time": qvps["time"]})

    # Redefine RHOHV_NC: keep it in the valid datetimes, put RHOHV in the rest
    qvps[X_RHO] = qvps[X_RHO].where(valid_datetimes_xr, qvps["RHOHV"])

# data = xr.concat(open_files, dim="time")
total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%% Calculate riming
print("Calculating riming ...")
loc = find_loc(locs, ff[0])

# 1. Start an inference session with the ONNX model
riming_model = ort.InferenceSession("/automount/agradar/jgiles/riming_model/gbm_model_23.10.2024.onnx")
riming_model_zh_zdr = ort.InferenceSession("/automount/agradar/jgiles/riming_model/gbm_zh_zdr_model_23.10.2024.onnx")

# 2. Get the name of the input node (ONNX uses string names to map inputs)
input_name = riming_model.get_inputs()[0].name
label_name = riming_model.get_outputs()[0].name

input_name_zh_zdr = riming_model_zh_zdr.get_inputs()[0].name
label_name_zh_zdr = riming_model_zh_zdr.get_outputs()[0].name

# 3. Prepare the data
if "DR" not in qvps:
    DR = utils.calc_depolarization(qvps, X_ZDR, X_RHO)
    # assign = dict(DR = DR.assign_attrs(
    #     {'long_name': 'Depolarization ratio based on '+X_ZDR+' and '+X_RHO,
    #      'standard_name': 'depolarization_ratio',
    #      'units': 'dB'}
    #     ))

idx = np.isfinite(  DR.values.flatten() + \
                    qvps[X_ZDR].values.flatten() + \
                    qvps[X_DBZH].values.flatten())
X = np.concatenate((DR.values.flatten()[idx].reshape(-1, 1), \
                    qvps[X_ZDR].values.flatten()[idx].reshape(-1, 1), \
                    qvps[X_DBZH].values.flatten()[idx].reshape(-1, 1)), axis=1)

idx_ = np.isfinite( qvps[X_ZDR].values.flatten() + \
                    qvps[X_DBZH].values.flatten())
X_ = np.concatenate((qvps[X_ZDR].values.flatten()[idx_].reshape(-1, 1), \
                    qvps[X_DBZH].values.flatten()[idx_].reshape(-1, 1)), axis=1)

# 4. Run the prediction
# The run method takes a list of outputs you want (None = all) and an input dictionary
# ONNX expects the data types to match exactly what you declared during export (float32)

# First the model with DR
pred = riming_model.run([label_name], {input_name: X.astype(np.float32)})[0]

pred_riming = np.zeros_like(qvps[X_DBZH]).flatten() + np.nan
pred_riming[idx] = pred
pred_riming = xr.zeros_like(qvps[X_DBZH]) + pred_riming.reshape(qvps[X_DBZH].shape)

varname = "riming_DR_"+"_".join([X_ZDR, X_DBZH])

pred_riming = pred_riming.assign_attrs({
    'long_name': 'Riming prediction based on DR, '+X_ZDR+' and '+X_DBZH+' with gradient boosting model',
     'standard_name': 'riming_prediction',
    }).rename(varname)

# save to file
if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered"):
    os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered")

pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered/"+loc+"_"+varname+".nc")

# Then the model without DR
pred = riming_model_zh_zdr.run([label_name_zh_zdr], {input_name_zh_zdr: X_.astype(np.float32)})[0]

pred_riming = np.zeros_like(qvps[X_DBZH]).flatten() + np.nan
pred_riming[idx_] = pred
pred_riming = xr.zeros_like(qvps[X_DBZH]) + pred_riming.reshape(qvps[X_DBZH].shape)

varname = "riming_"+"_".join([X_ZDR, X_DBZH])

pred_riming = pred_riming.assign_attrs({
    'long_name': 'Riming prediction based on '+X_ZDR+' and '+X_DBZH+' with gradient boosting model',
     'standard_name': 'riming_prediction',
    }).rename(varname)

# save to file
if not os.path.exists(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered"):
    os.makedirs(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered")

pred_riming.to_netcdf(realpep_path+"/upload/jgiles/radar_riming_classif"+suffix_name+"/unfiltered/"+loc+"_"+varname+".nc")

total_time = time.time() - start_time
print(f"Total time: {total_time/60:.2f} minutes.")