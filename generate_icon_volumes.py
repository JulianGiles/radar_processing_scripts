#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:28:32 2025

@author: jgiles

This script takes ICON+EMVORADO data and generates the volumes of the variables
necessary to later compute the microphysics.

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
warnings.filterwarnings('ignore')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet

import time
start_time = time.time()

#%% Set paths and options.
overwrite = False

#path_radar = "/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/icon_2017041200/radarout/cdfin_allsim_id-010392_*"
#path_icon = "/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/icon_2017041200/out_EU-0275_inst_DOM01_ML_20170412T*Z.nc"
#path_icon_z = '/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/icon_2017041200/out_EU-0275_constant_20170411T220000Z.nc'

radar_id = sys.argv[1] # read radar id from console
path_radar = sys.argv[2] # read path to synthetic radar from console
path_icon = sys.argv[3] # read path to icon from console
path_icon_z = sys.argv[4] # read path to icon height coordinate from console
path_save = sys.argv[5] # read path to save folder from console

# attach the files wildcards
radar_wc = "cdfin_allsim_id-"+radar_id+"_*"
icon_wc = "out_EU-0275_inst_DOM01_ML_*Z.nc"
icon_z_wc = "out_EU-0275_constant_*Z.nc"

path_radar = path_radar+"/"+radar_wc
path_icon = path_icon+"/"+icon_wc
path_icon_z = path_icon_z+"/"+icon_z_wc

#%% Collect files to process

paths_radar = sorted(glob.glob(path_radar))
paths_icon = sorted(glob.glob(path_icon))
paths_icon_z = sorted(glob.glob(path_icon_z))

#%% Process each timestep
for radarpath in paths_radar:

    print("processing "+radarpath)
    partial_start_time = time.time()

    # Get path for each file
    timestep = radarpath.split(radar_wc[:-1])[1][0:12]

    try:
        iconpath = [fp for fp in paths_icon if timestep[0:8]+"T"+timestep[8:] in fp][0]
    except IndexError:
        raise IndexError("ERROR: no ICON file for timestep "+timestep)

    if len(paths_icon_z)==1:
        iconzpath = paths_icon_z[0]
    elif len(paths_icon_z)>1:
        try:
            iconzpath = [fp for fp in paths_icon_z if timestep[0:8]+"T"+timestep[8:] in fp][0]
        except IndexError:
            raise IndexError("ERROR: no ICON Z file for timestep "+timestep)
    else:
        raise IndexError("ERROR: no ICON Z file for timestep "+timestep)

    # Check if the resulting file already exists
    savename_parts = os.path.basename(radarpath).split("id-")
    savename = savename_parts[0]+"icon_id-"+radar_id+"_"+timestep

    if os.path.exists(path_save+"/"+savename) and not overwrite:
        print("... processed file already exists, skipping... ")
        continue

    # load data
    data = utils.load_emvorado_to_radar_volume(radarpath, rename=True)
    radar_volume=data.copy()

    icon_field = utils.load_icon(iconpath, glob.glob(iconzpath)[0])
    icon_field['time'] = icon_field['time'].dt.round('1s') # round time coord to the second

    # regridding to radar volume geometry
    icon_volume = utils.icon_to_radar_volume(icon_field[["temp", "pres", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
                                                         "qnc", "qni", "qnr", "qns", "qng", "qnh", "z_ifc"]], radar_volume)
    icon_volume["TEMP"] = icon_volume["temp"] - 273.15

    # calculate microphysics
    icon_volume_new = utils.calc_microphys(icon_volume, mom=2)

    # save volumes
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for vv in icon_volume_new.data_vars:
        icon_volume_new[vv].encoding = {'zlib': True, 'complevel': 6}

    icon_volume_new.to_netcdf(path_save+"/"+savename)

    # print how much time did it take
    partial_total_time = time.time() - partial_start_time
    print(f"... took {partial_total_time/60:.2f} minutes to run.")

# print how much time did it take in total
total_time = time.time() - start_time
print(f"FINISHED: Script took {total_time/60:.2f} minutes to run.")
