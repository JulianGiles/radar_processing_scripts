#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:16 2023

@author: jgiles

Script for noise-correcting RHOHV.

"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import numpy as np
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

realpep_path = "/automount/realpep/"

#%% Set paths and options
# paths for testing
# path="/automount/realpep/upload/jgiles/dwd/2017/2017-09/2017-09-30/tur/vol5minng01/01/*hd5" # for QVPs: /home/jgiles/
# path="/automount/realpep/upload/jgiles/dmi//2016/2016-05/2016-05-22/AFY/VOL_B/7.0/*h5*" # for QVPs: /home/jgiles/

path0 = sys.argv[1]
overwrite = False # overwrite existing files?

dwd_rhohv_nc_reference_file = realpep_path+"/upload/jgiles/reference_files/reference_dwd_rhohv_nc_file/ras07-90gradstarng01_sweeph5onem_rhohv_nc_00-2015010100042300-pro-10392-hd5"

dbzh_names = ["DBZH"] # names to look for the DBZH variable, in order of preference
rhohv_names = ["RHOHV"] # same but for RHOHV

if "hd5" in path0 or "h5" in path0:
    files=[path0]
elif "dwd" in path0:
    files = sorted(glob.glob(path0+"/*hd5*"))
elif "dmi" in path0:
    files = sorted(glob.glob(path0+"/*h5*"))
else:
    print("Country code not found in path")
    sys.exit("Country code not found in path.")

# get the files and check that it is not empty
if len(files)==0:
    print("No files meet the selection criteria.")
    sys.exit("No files meet the selection criteria.")

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
    savepath = (country+"/rhohv_nc/"+name+"/").join(ff_parts)
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    return savepath

#%% Load data

for ff in files:

    separator = "any"
    if "allmoms" in ff: # allmoms naming is deprecated but old files remain
        separator = "allmoms"
    if "12345" in ff: # separator for boxpol
        separator = "12345"

    # check if the QVP file already exists before starting
    savepath = make_savedir(ff, "")
    if os.path.exists(("rhohv_nc").join(savepath.split(separator))) and not overwrite:
        continue

    print("processing "+ff)
    if "dwd" in ff:
        data = utils.load_dwd_preprocessed(ff) # this already loads the first elev available in the files and fixes time coord
    elif "dmi" in ff:
        # data=xr.open_dataset(ff)
        data = utils.load_dmi_preprocessed(ff) # this loads DMI file and flips phidp and fixes time coord
    elif "boxpol" in ff:
        data = xr.open_dataset(ff)
    else:
        raise NotImplementedError("Only DWD, DMI or BoXPol data supported at the moment")

    # fix time dim and time in coords
    # data = utils.fix_time_in_coords(data)


#%% Calculate RHOHV correction
    # get DBZH name
    for X_DBZH in dbzh_names:
        if X_DBZH in data.data_vars:
            break

    # get RHOHV name
    for X_RHO in rhohv_names:
        if X_RHO in data.data_vars:
            break

    # check that the variables actually exist, otherwise continue
    if X_DBZH not in data.data_vars:
        print("DBZH not found in data")
        sys.exit("DBZH not found in data.")
    if X_RHO not in data.data_vars:
        print("RHOHV not found in data")
        sys.exit("RHOHV not found in data.")

    rho_nc = utils.calculate_noise_level(data[X_DBZH], data[X_RHO], noise=(-45, -15, 1))

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
    rho_nc2 = utils.noise_correction2(data[X_DBZH], data[X_RHO], ncl*1.02)

    # make a new array as before
    rho_nc_out2 = xr.merge(rho_nc2)
    rho_nc_out2.attrs["noise correction level"]=ncl*1.02

    # create saving directory if it does not exist
    if "dwd" in ff:
        country="dwd"
    elif "dmi" in ff:
        country="dmi"
    elif "boxpol" in ff:
        country="boxpol"
    else:
        print("Country code not found in path")
        sys.exit("Country code not found in path.")

    # copy encoding from DWD to reduce file size
    rho_nc_out["RHOHV_NC"].encoding = data[X_RHO].encoding
    rho_nc_out2["RHOHV_NC"].encoding = data[X_RHO].encoding
    if country=="dwd": # special treatment for SNR since it may not be available in turkish data
        rho_nc_out["SNRH"].encoding = data["SNRHC"].encoding
        rho_nc_out2["SNRH"].encoding = data["SNRHC"].encoding
    else:
        rho_nc_dwd = xr.open_dataset(dwd_rhohv_nc_reference_file, engine="netcdf4")
        rho_nc_out["SNRH"].encoding = rho_nc_dwd["SNRH"].encoding
        rho_nc_out2["SNRH"].encoding = rho_nc_dwd["SNRH"].encoding

    # save the arrays
    filename = ("rhohv_nc").join(savepath.split(separator))
    rho_nc_out.to_netcdf(filename)

    filename = ("rhohv_nc_2percent").join(savepath.split(separator))
    rho_nc_out2.to_netcdf(filename)

#%% print how much time did it take
total_time = time.time() - start_time
print(f"Script took {total_time/60:.2f} minutes to run.")


#%% Testing how this works
"""
# put the result into a new array to fit with a line
rhonc_snrh = xr.DataArray(rho_nc_out.RHOHV_NC.values.flatten(), coords={"SNRH":rho_nc_out.SNRH.values.flatten()})

rhonc_snrh.where((0<rhonc_snrh.SNRH)&(rhonc_snrh.SNRH<20)&(rhonc_snrh>0.7)).polyfit("SNRH", deg=1, skipna=True)

# lets do it for every noise level
fits=[]
for nn,rhon in enumerate(rho_nc[0]):
    merged = xr.merge(rhon)
    rhonc_snrh = xr.DataArray(merged.RHOHV_NC.values.flatten(), coords={"SNRH":merged.SNRH.values.flatten()})
    fits.append(float(rhonc_snrh.where((0<rhonc_snrh.SNRH)&(rhonc_snrh.SNRH<20)&(rhonc_snrh>0.7)).polyfit("SNRH", deg=1, skipna=True).polyfit_coefficients[0].values))

# checking which fit has the slope closest to zero
np.abs(np.array(fits)).argmin()
"""