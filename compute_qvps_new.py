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
overwrite = False # overwrite existing files?

# Set the possible ZDR calibrations locations to include (in order of priority)
# The script will try to correct according to the first offset; if not available or nan it will 
# continue with the next one, and so on. Only the used offset will be outputted in the final file.
# All items in zdrofffile will be tested in each zdroffdir to load the data.
zdroffdir = utils.zdroffdir
zdrofffile = utils.zdrofffile

# set the RHOHV correction location
rhoncdir = utils.rhoncdir  # subfolder where to find the noise corrected rhohv data
rhoncfile = utils.rhoncfile # pattern to select the appropriate file (careful with the rhohv_nc_2percent)

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

clowres0=False # this is for the ML detection algorithm
min_hgts = utils.min_hgts
min_hgt = min_hgts["default"] # minimum height above the radar to be considered when calculating ZDR offset
if "dwd" in path0 and "90grads" in path0:
    # for the VP we need to set a higher min height because there are several bins of unrealistic values
    min_hgt = min_hgts["90grads"]
if "dwd" in path0 and "vol5minng01" in path0:
    clowres0=True
if "ANK" in path0:
    # for ANK we need higher min_hgt to avoid artifacts
    min_hgt = min_hgts["ANK"]
if "GZT" in path0:
    # for GZT we need higher min_hgt to avoid artifacts
    min_hgt = min_hgts["GZT"]

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
    savepath = (country+"/qvps/"+name+"/").join(ff_parts)
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
    
    # check if the QVP file already exists and it is larger than 100 KiB before starting
    savepath = make_savedir(ff, "")
    if os.path.exists(savepath) and (os.stat(savepath).st_size >> 10) > 100 and not overwrite:
        continue

    print("processing "+ff)
    if "dwd" in ff:
        country="dwd"
        data = utils.load_dwd_preprocessed(ff) # this already loads the first elev available in the files and fixes time coord
    elif "dmi" in ff:
        country="dmi"
        data = utils.load_dmi_preprocessed(ff) # this loads DMI file and flips phidp and fixes time coord
    else:
        data=xr.open_dataset(ff)

    # flip UPHIDP and KDP in UMD data
    if "umd" in ff:
        print("Flipping phase moments in UMD")
        for vf in ["UPHIDP", "KDP"]: # Phase moments in UMD are flipped into the negatives
            attrs = data[vf].attrs.copy()
            data[vf] = data[vf]*-1
            data[vf].attrs = attrs.copy()

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

        # Check that the corrected RHOHV does not have much higher STD than the original (50% more)
        # if that is the case we take it that the correction did not work well so we won't use it
        if not (swp[X_RHO].std()*1.5 < swp["RHOHV_NC"].std()).compute():
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

    try:
        for zdrod in zdroffdir:
            for zdrof in zdrofffile:
                try:
                    zdroffsetpath = os.path.dirname(utils.edit_str(ff, country, country+zdrod))
                    if "/VP/" in zdrod and "/vol5minng01/" in ff:
                        elevnr = ff.split("/vol5minng01/")[-1][0:2]
                        zdroffsetpath = utils.edit_str(zdroffsetpath, "/vol5minng01/"+elevnr, "/90gradstarng01/00")
                        
                    swp = utils.load_ZDR_offset(swp, X_ZDR, zdroffsetpath+"/"+zdrof, zdr_oc_name=X_ZDR+"_OC")
                    
                    # if the offset comes from LR ZH-ZDR consistency, check it against
                    # the QVP method (if available) and choose the best one based on how 
                    # many negative values remain
                    if "LR_consistency" in zdrod:
                        for zdrof2 in zdrofffile:
                            try:
                                zdrod2 = [pp for pp in zdroffdir if "QVP" in pp][0]
                                zdroffsetpath_qvp = os.path.dirname(utils.edit_str(ff, country, country+zdrod2))
                                swp_qvpoc = utils.load_ZDR_offset(swp, X_ZDR, zdroffsetpath_qvp+"/"+zdrof2, zdr_oc_name=X_ZDR+"_OC")
                                
                                # calculate the count of negative values after each correction
                                neg_count_swp_lroc = (swp[X_ZDR+"_OC"].where(swp[X_RHO]>0.99) < 0).sum().compute()
                                neg_count_swp_qvpoc = (swp_qvpoc[X_ZDR+"_OC"].where(swp_qvpoc[X_RHO]>0.99) < 0).sum().compute()
                                
                                if neg_count_swp_lroc > neg_count_swp_qvpoc:
                                    # continue with the correction with less negative values
                                    swp = swp_qvpoc
                                                                
                                break
                            except (OSError, ValueError):
                                pass
                    
                    # calculate the count of negative values before and after correction
                    neg_count_swp = (swp[X_ZDR].where(swp[X_RHO]>0.99) < 0).sum().compute()
                    neg_count_swp_oc = (swp[X_ZDR+"_OC"].where(swp[X_RHO]>0.99) < 0).sum().compute()
                    
                    if neg_count_swp_oc > neg_count_swp and abs((swp[X_ZDR] - swp[X_ZDR+"_OC"]).compute().median()) < 0.2:
                        # if the correction introduces more negative values and the offset is lower than 0.2, then do not correct
                        swp[X_ZDR+"_OC"] = swp[X_ZDR]
                    
                    # Change the default ZDR name to the corrected one
                    X_ZDR = X_ZDR+"_OC"
                    
                    # raise the custom exception to stop the loops
                    raise FileFound 
                    
                except (OSError, ValueError):
                    pass
                
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
            fix_range = 200 # range from where to consider phi values (dmi data is bad in the first bin)

        ######### Processing PHIDP
        #### fix PHIDP
        
        
        # phidp may be already preprocessed (turkish case), then only offset-correct (no smoothing) and then vulpiani
        if "UPHIDP" not in X_PHI:
            # calculate phidp offset
            ds = utils.phidp_offset_correction(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range)
        
            phi_masked = ds[X_PHI+"_OC"].where((ds[X_RHO] >= 0.9) & (ds[X_DBZH] >= 0.) & (ds["z"]>min_height) )   
        
        else:
            # process phidp (offset and smoothing)
            ds = utils.phidp_processing(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                                 dbzhmin=0., min_height=min_height, window=window0, fix_range=fix_range)
        
            phi_masked = ds[X_PHI+"_OC_SMOOTH"].where((ds[X_RHO] >= 0.9) & (ds[X_DBZH] >= 0.) & (ds["z"]>min_height) )   

        # Assign phi_masked
        assign = { X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs) }
            
        ds = ds.assign(assign)
        
        # derive KDP from PHIDP (Vulpiani)
        
        ds = utils.kdp_phidp_vulpiani(ds, winlen0, X_PHI+"_OC_MASKED", min_periods=winlen0/2)    
        
        X_PHI = X_PHI+"_OC" # continue using offset corrected PHI
                
    else:
        print(X_PHI+" not found in the data, skipping ML detection")
    
#%% Compute QVP
    ## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
    ## Also added further filtering (TH>0, ZDR>-1)
    ds_qvp_ra = utils.compute_qvp(ds, min_thresh={X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5} )
    
    # filter out values close to the ground
    ds_qvp_ra2 = ds_qvp_ra.where(ds_qvp_ra["z"]>min_height)

#%% Detect melting layer
    if X_PHI in ds.data_vars:
        if country=="dwd":
            moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}
        elif country=="dmi":
            moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}

        ds_qvp_ra = utils.melting_layer_qvp_X_new(ds_qvp_ra2, moments=moments, dim="z", fmlh=0.3, 
                 xwin=xwin0, ywin=ywin0, min_h=min_height, all_data=True, clowres=clowres0)
    
        #### Assign ML values to dataset
        
        ds = ds.assign_coords({'height_ml': ds_qvp_ra.height_ml})
        ds = ds.assign_coords({'height_ml_bottom': ds_qvp_ra.height_ml_bottom})
        ds = ds.assign_coords({'height_ml_new_gia': ds_qvp_ra.height_ml_new_gia})
        ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp_ra.height_ml_bottom_new_gia})
    
#%% Attach ERA5 temperature profile
    loc = utils.find_loc(utils.locs, ff)
    ds_qvp_ra = utils.attach_ERA5_TEMP(ds_qvp_ra, path=loc.join(era5_dir.split("loc")))
 
#%% Discard possible erroneous ML values
    if "height_ml_new_gia" in ds_qvp_ra:
        ## First, filter out ML heights that are too high (above selected isotherm)
        isotherm = -1 # isotherm for the upper limit of possible ML values
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

#%% Fix KDP in the ML using PHIDP:
    if X_PHI in ds.data_vars:    
        
        ds = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen=winlen0, min_periods=winlen0/2)

        ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1, "SNRH":10, "SQIH":0.5})["KDP_ML_corrected"]})
    
#%% Classification of stratiform events based on entropy
    if X_PHI in ds.data_vars:    
        
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
