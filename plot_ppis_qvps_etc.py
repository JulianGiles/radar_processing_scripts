#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:36:23 2023

@author: jgiles

Plot PPIs, QVPs, line plots, etc
"""


import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None


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
import xradar as xd
import cmweather

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


#%% Load and process data

## Load the data into an xarray dataset (ds)

ff = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/*allmoms*"
# ff = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/90gradstarng01/00/*allmoms*"
# ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2021/2021-10/2021-10-30/ess/90gradstarng01/00/*"
# ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2021/2021-07/2021-07-24/ess/90gradstarng01/00/*"
ds = utils.load_dwd_preprocessed(ff)
# ds = utils.load_dwd_raw(ff)

if "dwd" in ff or "DWD" in ff:
    country="dwd"
    clowres0=True # this is for the ML detection algorithm
elif "dmi" in ff:
    country="dmi"
    clowres0=False

## Georeference 

ds = ds.pipe(wrl.georef.georeference) 

## Define minimum height of usable data

min_height = utils.min_hgts["90grads"] + ds["altitude"].values

## Get variable names

X_DBZH, X_PHI, X_RHO, X_ZDR, X_TH = utils.get_names(ds)

## Load ZDR offset

# We define a custom exception to stop the next nexted loops as soon as a file is loaded
class FileFound(Exception):
    pass

# Define the offset paths and file names or take them from the default

zdroffdir = utils.zdroffdir
zdrofffile = utils.zdrofffile

# Load the offsets

try:
    for zdrod in zdroffdir:
        for zdrof in zdrofffile:
            try:
                zdroffsetpath = os.path.dirname(utils.edit_str(ff, country, country+zdrod))
                if "/VP/" in zdrod and "/vol5minng01/" in ff:
                    elevnr = ff.split("/vol5minng01/")[-1][0:2]
                    zdroffsetpath = utils.edit_str(zdroffsetpath, "/vol5minng01/"+elevnr, "/90gradstarng01/00")
                    
                ds = utils.load_ZDR_offset(ds, X_ZDR, zdroffsetpath+"/"+zdrof)
                
                # Change the default ZDR name to the corrected one
                X_ZDR = X_ZDR+"_OC"
                
                # raise the custom exception to stop the loops
                raise FileFound 
                
            except OSError:
                pass
            
    # If no ZDR offset was loaded, print a message
    print("No zdr offset to load: "+zdroffsetpath+"/"+zdrof)
except FileFound:
    pass


## Load noise corrected RHOHV

# Define the rhohv corrected paths and file names or take them from the default

rhoncdir = utils.rhoncdir
rhoncfile = utils.rhoncfile


try:
    rhoncpath = os.path.dirname(utils.edit_str(ff, country, country+rhoncdir))
    
    ds = utils.load_corrected_RHOHV(ds, rhoncpath+"/"+rhoncfile)
    
    # Check that the corrected RHOHV does not have much higher STD than the original (50% more)
    # if that is the case we take it that the correction did not work well so we won't use it
    if not (ds[X_RHO].std()*1.5 < ds["RHOHV_NC"].std()).compute():
        # Change the default RHOHV name to the corrected one
        X_RHO = X_RHO+"_NC"
        
except OSError:
    print("No noise corrected rhohv to load: "+rhoncpath+"/"+rhoncfile)


## Phase processing

interpolation_method_ML = "linear" # for interpolating PHIDP in the ML


phase_pross_params = {
                        "dwd": {
                            "window0": 7, # number of range bins for phidp smoothing (this one is quite important!)
                            "winlen0": 7, # size of range window (bins) for the kdp-phidp calculations
                            "xwin0": 9, # window size (bins) for the time rolling median smoothing in ML detection
                            "ywin0": 1, # window size (bins) for the height rolling mean smoothing in ML detection
                            "fix_range": 750, # range from where to consider phi values (dwd data is bad in the first bin)
                        },
                        "dmi": {
                            "window0": 17,
                            "winlen0": 21,
                            "xwin0": 5,
                            "ywin0": 5,
                            "fix_range": 200,
                        },
}

# Check that PHIDP is in data, otherwise skip ML detection
if X_PHI in ds.data_vars:
    # Set parameters according to data
    
    # for param_name in phase_pross_params[country].keys():
    #     globals()[param_name] = phase_pross_params[country][param_name]    
    window0, winlen0, xwin0, ywin0, fix_range = phase_pross_params[country].values() # explicit alternative

    # phidp may be already preprocessed (turkish case), then proceed directly to masking and then vulpiani
    if "UPHIDP" not in X_PHI:
        # mask 
        phi_masked = ds[X_PHI].where((ds[X_RHO] >= 0.95) & (ds[X_DBZH] >= 0.) & (ds["z"]>min_height) )
        
        # rename X_PHI as offset corrected
        ds = ds.rename({X_PHI: X_PHI+"_OC"})

    else:
        ds = utils.phidp_processing(ds, X_PHI=X_PHI, X_RHO=X_RHO, X_DBZH=X_DBZH, rhohvmin=0.9,
                             dbzhmin=0., min_height=0, window=window0, fix_range=fix_range)
    
        phi_masked = ds[X_PHI+"_OC_SMOOTH"].where((ds[X_RHO] >= 0.95) & (ds[X_DBZH] >= 0.) & (ds["z"]>min_height) )

    # Assign phi_masked
    assign = { X_PHI+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHI].attrs) }
    ds = ds.assign(assign)
    
    # derive KDP from PHIDP (Vulpiani)

    ds = utils.kdp_phidp_vulpiani(ds, winlen0, X_PHI+"_OC_MASKED", min_periods=winlen0/2)    
    
    X_PHI = X_PHI+"_OC" # continue using offset corrected PHI

else:
    print(X_PHI+" not found in the data, skipping ML detection")

## Compute QVP

ds_qvp = utils.compute_qvp(ds, min_thresh = {X_RHO:0.7, X_TH:0, X_ZDR:-1} )

## Detect melting layer

if X_PHI in ds.data_vars:
    # Define thresholds
    moments={X_DBZH: (10., 60.), X_RHO: (0.65, 1.), X_PHI: (-20, 180)}
    
    # Calculate ML
    ds_qvp = utils.melting_layer_qvp_X_new(ds_qvp, moments=moments, 
             dim="z", xwin=xwin0, ywin=ywin0, min_h=min_height, all_data=True, clowres=clowres0)
    
    # Assign ML values to dataset
    
    ds = ds.assign_coords({'height_ml': ds_qvp.height_ml})
    ds = ds.assign_coords({'height_ml_bottom': ds_qvp.height_ml_bottom})

    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp.height_ml_new_gia})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp.height_ml_bottom_new_gia})

## Attach ERA5 temperature profile
loc = utils.find_loc(utils.locs, ff)
ds_qvp = utils.attach_ERA5_TEMP(ds_qvp, path=loc.join(utils.era5_dir.split("loc")))
ds = utils.attach_ERA5_TEMP(ds, path=loc.join(utils.era5_dir.split("loc")))

## Discard possible erroneous ML values
if "height_ml_new_gia" in ds_qvp:
    isotherm = -1 # isotherm for the upper limit of possible ML values
    z_isotherm = ds_qvp.TEMP.isel(z=((ds_qvp["TEMP"]-isotherm)**2).argmin("z").compute())["z"]
    
    ds_qvp.coords["height_ml_new_gia"] = ds_qvp["height_ml_new_gia"].where(ds_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
    ds_qvp.coords["height_ml_bottom_new_gia"] = ds_qvp["height_ml_bottom_new_gia"].where(ds_qvp["height_ml_new_gia"]<=z_isotherm.values).compute()
    
    ds = ds.assign_coords({'height_ml_new_gia': ds_qvp.height_ml_new_gia})
    ds = ds.assign_coords({'height_ml_bottom_new_gia': ds_qvp.height_ml_bottom_new_gia})

## Fix KDP in the ML using PHIDP:
if X_PHI in ds.data_vars:    
    ds = utils.KDP_ML_correction(ds, X_PHI+"_MASKED", winlen=winlen0)

    ds_qvp = ds_qvp.assign({"KDP_ML_corrected": utils.compute_qvp(ds)["KDP_ML_corrected"]})
        
## Classification of stratiform events based on entropy
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
    ds_qvp = ds_qvp.assign({"min_entropy": min_trst_strati_qvp})

#%% Plot PPI

tsel = "2015-09-30T08:04"
datasel = ds.loc[{"time": tsel}].pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


mom = "KDP"

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
datasel[mom][0].wrl.plot(x="x", y="y", cmap=cmap, norm=norm, xlim=(-25000,25000), ylim=(-25000,25000))

#%% Plot QVP


#%% Compute ZDR VP calibration

zdr_offset_belowML = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="step").compute()
zdr_offset_belowML_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, min_h=min_height, timemode="all").compute()

zdr_offset_inML = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_bottom_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="height_ml_new_gia", min_h=min_height, timemode="step").compute()
zdr_offset_inML_all = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_bottom_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="height_ml_new_gia", min_h=min_height, timemode="all").compute()

zdr_offset_aboveML = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_aboveML_all = utils.zdr_offset_detection_vps(ds.where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()

zdr_offset_whole = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_whole_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()

cond_noML = ((ds.z>ds.height_ml_new_gia) + (ds.z<ds.height_ml_bottom_new_gia)).compute()
zdr_offset_whole_noML = utils.zdr_offset_detection_vps(ds.where(cond_noML), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="step").compute()
zdr_offset_whole_noML_all = utils.zdr_offset_detection_vps(ds, zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom=-100, min_h=min_height, timemode="all").compute()


# Temporary fix because I do not have ERA5 temp downloaded for after 2020
zdr_offset_aboveML = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
zdr_offset_aboveML_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(ds.z>ds.height_ml_new_gia), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

zdr_offset_whole = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
zdr_offset_whole_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

cond_noML = ((ds.z>ds.height_ml_new_gia) + (ds.z<ds.height_ml_bottom_new_gia)).compute()
zdr_offset_whole_noML = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}).where(cond_noML), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="step").compute()
zdr_offset_whole_noML_all = utils.zdr_offset_detection_vps(ds.assign_coords({"fake_ml":ds.height_ml_bottom_new_gia+10000}), zdr="ZDR", dbzh=X_DBZH, rhohv=X_RHO, mlbottom="fake_ml", min_h=min_height, timemode="all").compute()

#%% Plot ZDR VP calibration 

# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different regions (below ML, in ML, above ML)
mom = "RHOHV_NC"
visdict14 = radarmet.visdict14
norm = utils.get_discrete_norm(visdict14[mom]["ticks"])
cmap = utils.get_discrete_cmap(visdict14[mom]["ticks"], visdict14[mom]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
templevels = [-100]
date = ds.time[0].values.astype('datetime64[D]').astype(str)

offsets_to_plot = {"Below ML": zdr_offset_belowML,
                   "In ML": zdr_offset_inML,
                   "Above ML": zdr_offset_aboveML,
                   "Whole column": zdr_offset_whole,
                   "Whole column \n no ML": zdr_offset_whole_noML}

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ds_qvp[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
# plot ML limits
ds_qvp["height_ml_bottom_new_gia"].plot(color="black", label="ML") 
ds_qvp["height_ml_new_gia"].plot(color="black")
# Plot min_height
(xr.ones_like(ds_qvp["height_ml_new_gia"])*min_height).plot(color="black")
# ax.text(ds_qvp.time[0]-1, min_height, "min_height")
ax.text(-0.16, min_height/5000, "min_height", transform=ax.transAxes)
plt.legend()
plt.title(mom+" "+loc.upper()+" "+date)
plt.ylabel("height [m]")

ax2=plt.subplot(gs[1], sharex=ax)
ax3 = ax2.twinx()
for noff in offsets_to_plot.keys():
    offsets_to_plot[noff]["ZDR_offset"].plot(label=str(noff), ls="-", ax=ax2, ylim=(-0.3,1))
    ax2.set_ylabel("")
    offsets_to_plot[noff]["ZDR_std_from_offset"].plot(label=str(noff), ls="--", ax=ax3, ylim=(-0.3,1), alpha=0.5)
    ax3.set_ylabel("")
    
ax2.set_title("")
ax3.set_title("Full: offset. Dashed: offset std.")
ax3.set_yticks([],[])
ax3.legend(loc=(1.01,0))

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.4, 0.02, 0.5])
fig.colorbar(figvp, cax=cbar_ax, extend="both")


# Same as above but with separate plots for the line plots
# Plot a moment VP, isotherms, ML bottom and calculated ZDR offset for different regions (below ML, in ML, above ML)
mom = "RHOHV"
visdict14 = radarmet.visdict14
norm = utils.get_discrete_norm(visdict14[mom]["ticks"])
cmap = utils.get_discrete_cmap(visdict14[mom]["ticks"], visdict14[mom]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
templevels = [-100]
date = ds.time[0].values.astype('datetime64[D]').astype(str)

offsets_to_plot = {"Below ML": zdr_offset_belowML,
                   "In ML": zdr_offset_inML,
                   "Above ML": zdr_offset_aboveML,
                   "Whole column": zdr_offset_whole,
                   "Whole column \n no ML": zdr_offset_whole_noML}

fig = plt.figure(figsize=(7,7))
# set height ratios for subplots
gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0) 
ax = plt.subplot(gs[0])
figvp = ds_qvp[mom].plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(0,10000), add_colorbar=False)
figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=[0]+templevels, ylim=(0,5000))
# ax = plt.gca()
ax.clabel(figcontour)
# plot ML limits
ds_qvp["height_ml_bottom_new_gia"].plot(color="black", label="ML") 
ds_qvp["height_ml_new_gia"].plot(color="black")
# Plot min_height
(xr.ones_like(ds_qvp["height_ml_new_gia"])*min_height).plot(color="black")
# ax.text(ds_qvp.time[0]-1, min_height, "min_height")
ax.text(-0.16, min_height/5000, "min_height", transform=ax.transAxes)
plt.legend()
plt.title(mom+" "+loc.upper()+" "+date)
plt.ylabel("height [m]")

ax2=plt.subplot(gs[1], sharex=ax)
ax3 = plt.subplot(gs[2], sharex=ax2)
for noff in offsets_to_plot.keys():
    offsets_to_plot[noff]["ZDR_offset"].plot(label=str(noff), ls="-", ax=ax2, ylim=(-0.4,0.1))
    ax2.set_ylabel("")
    offsets_to_plot[noff]["ZDR_std_from_offset"].plot(label=str(noff), ls="-", ax=ax3, ylim=(0,1))
    ax3.set_ylabel("")

ax2.set_title("")
ax3.set_title("")
ax2.text(0.5, 0.9, "Offset", transform=ax2.transAxes, horizontalalignment='center')    
ax3.text(0.5, 0.9, "Standard Dev.", transform=ax3.transAxes, horizontalalignment='center')    
# ax3.set_yticks([],[])
ax2.legend(loc=(1.01,0))

## Custom legend
# Extract the current legend handles and labels
handles = ax2.get_legend().legendHandles
labels = ax2.get_legend().get_texts()

# Modify the legend labels and add a title
new_labels = [str(round(float(offsets_to_plot[noff]["ZDR_offset"].median()), 4)) for noff in offsets_to_plot.keys()]
legend_title = "Daily offsets"

# Create a new legend with the modified handles, labels, and title
ax3.legend(handles=handles, labels=new_labels, title=legend_title, loc=(1.01,0))


fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.55, 0.02, 0.3])
fig.colorbar(figvp, cax=cbar_ax, extend="both")

#%% Check noise correction for RHOHV

X_RHO = "RHOHV"

# calculate the noise level
rho_nc = utils.calculate_noise_level(ds[X_DBZH][0], ds[X_RHO][0], noise=(-40, -20, 1))

# get the "best" noise correction level (acoording to the min std, Veli's way)
ncl = rho_nc[-1]

# get index of the best correction
bci = np.array(rho_nc[-2]).argmin()

# merge into a single array
rho_nc_out = xr.merge(rho_nc[0][bci])

# add noise correction level as attribute
rho_nc_out.attrs["noise correction level"]=ncl


# Correct rhohv just using the SNRHC from the files (from Ryzhkov and Zrnic page 186)
# we assume eta = 1
zdr_nc = wrl.trafo.idecibel(ds["SNRHC"][0]) * ds["ZDR"][0] / (wrl.trafo.idecibel(ds["SNRHC"][0]) + 1 - ds["ZDR"][0])
# zdr_nc = 1 # another approximation would be to set ZDR = 1
rho_nc2 = ds[X_RHO][0] * (1 + 1/wrl.trafo.idecibel(ds["SNRHC"][0].where(ds["SNRHC"][0]>0)) )**0.5 * (1 + zdr_nc/wrl.trafo.idecibel(ds["SNRHC"][0]) )**0.5


## Plot
# plot noise corrected RHOHV with utils.calculate_noise_level
rho_nc_out.RHOHV_NC.plot(vmin=0, vmax=1)
plt.title("Noise corrected RHOHV with own noise calc")

(rho_nc_out.RHOHV_NC>1).plot()
plt.title("Bins with noise corrected RHOHV > 1")

# plot noise corrected RHOHV with DWD SNRHC
rho_nc2.plot(vmin=0, vmax=1)
plt.title("Noise corrected RHOHV with SNRHC from files")

(rho_nc2>1).plot()
plt.title("Bins with noise corrected RHOHV > 1")

# plot the SNRH
rho_nc_out["SNRH"].plot(vmin=-60, vmax=60)
plt.title("SNRH from own noise calc")

ds["SNRHC"][0].plot(vmin=-60, vmax=60)
plt.title("SNRH from DWD")

# plot scatters of RHOHV vs SNR
plt.scatter(rho_nc[0][bci][0], rho_nc[0][bci][1], s=0.01, alpha=0.5)
plt.scatter(ds["SNRH"][0], ds[X_RHO][0], s=0.01, alpha=0.5)

# Plot ZDR and noise corrected ZDR to check if it really makes a difference (looks like it does not)

ds.ZDR[0].plot(vmin=-2, vmax=2)
plt.title("ZDR")

zdr_nc.plot(vmin=-2, vmax=2)
plt.title("noise corrected ZDR")

(ds.ZDR[0] - zdr_nc).plot(vmin=-0.1, vmax=0.1)
plt.title("Difference ZDR - noise corrected ZDR")

#%% Check noise power level in raw DWD files

ff = "/automount/realpep/upload/RealPEP-SPP/DWD-CBand/20*/*/*/pro/vol5minng01/05/*snrhc*"
files = glob.glob(ff)

noise_h = []
noise_v = []
eta = []
date_time = []
for f0 in files:
    aux = dttree.open_datatree(f0)
    noise_h.append(aux["how"]["radar_system"].attrs["noise_H_pw0"])
    noise_v.append(aux["how"]["radar_system"].attrs["noise_V_pw0"])
    eta.append(noise_h[-1]/noise_v[-1])
    
    datestr = aux["what"].attrs["date"]
    timestr = aux["what"].attrs["time"]
    date_time.append(datetime.datetime.strptime(datestr + timestr, "%Y%m%d%H%M%S"))
