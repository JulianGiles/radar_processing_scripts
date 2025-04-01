#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:22:35 2022

@author: jgiles
"""

import os
try:
    os.chdir('/home/jgiles/')
except FileNotFoundError:
    None
os.environ["WRADLIB_DATA"] = '/home/jgiles/wradlib-data-main'

import wradlib as wrl
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import sys
import glob
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import pandas as pd
import cartopy
import time

import xarray as xr

import hvplot
import hvplot.xarray
import holoviews as hv
# hv.extension("bokeh", "matplotlib") # better to put this each time this kind of plot is needed

import panel as pn
from bokeh.resources import INLINE
from osgeo import osr
from functools import partial

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    from Scripts.python.radar_processing_scripts.classify_precip_typ import classify_precip_typ
except ModuleNotFoundError:
    import utils
    import radarmet
    from classify_precip_typ import classify_precip_typ


#%% Initial definitions
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='10m',
    facecolor='none')


# Create dictionary for mapping variable names from original observations to EMVORADO names
var_names_map_obs = {"VRADH": "vrobs",
                 "DBZH": "zrobs",
                 "ZDR": "zdrobs",
                 "KDP": "kdpobs",
                 "UPHIDP": "phidpobs",
                 "RHOHV": "rhvobs",
                    }

var_names_map_sim = {"VRADH": "vrsim",
                 "DBZH": "zrsim",
                 "ZDR": "zdrsim",
                 "KDP": "kdpsim",
                 "UPHIDP": "phidpsim",
                 "RHOHV": "rhvsim",
                    }

#%% Load data

# Define paths
path_dwd = "/automount/realpep/upload/jgiles/dwd/2020/2020-02/2020-02-01/pro/vol5minng01/*/*-hd5"
path_dwd = "/automount/realpep/upload/jgiles/dmi/2020/2020-01/2020-01-03/HTY/VOL_*/*/*.nc"
# path_emvorado_obs = "/home/jgiles/emvorado-offline-results/output/20171028_*/radarout/cdfin_allobs_id-010392_*_volscan"
path_emvorado_sim = "/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2020010222/radout/cdfin_allsim_id-017373_*_volscan"
# path_emvorado_sim = "/automount/ags/jgiles/emvorado-offline-results/output/20170712_*/radarout/cdfin_allsim_id-010392_*_volscan"

# Load files
vol_dwd = utils.load_volume(sorted(glob.glob(path_dwd)), func=utils.load_dmi_preprocessed, verbose=True)
# vol_emvorado_obs = utils.load_emvorado_to_radar_volume(path_emvorado_obs, rename=True)
vol_emvorado_sim = utils.load_emvorado_to_radar_volume(path_emvorado_sim, rename=True)

#%% Plot PPI
tsel = "2020-01-03T20"
elevn = 7 # elevation index

datatype = "sim" # sim, obs or ori (original)
mom = "ZDR"
xylims = 180000 # xlim and ylim (from -xylims to xylims)

cities = {
  'Berlin': {'lat': 52.520008, 'lon': 13.404954},
  # Add more cities as needed
}

if datatype == "sim":
    ds = vol_emvorado_sim.copy()

elif datatype == "obs":
    ds = vol_emvorado_obs.copy()

elif datatype == "ori":
    ds = vol_dwd.copy()

else:
    raise Exception("select correct data source")

if "sweep_fixed_angle" in ds.dims:
    isvolume = True

if isvolume: # if more than one elevation, we need to select the one we want
    if tsel == "":
        datasel = ds.isel({"sweep_fixed_angle":elevn})
    else:
        datasel = ds.isel({"sweep_fixed_angle":elevn}).sel({"time": tsel}, method="nearest")
else:
    if tsel == "":
        datasel = ds
    else:
        datasel = ds.sel({"time": tsel}, method="nearest")

datasel = datasel.pipe(wrl.georef.georeference)

# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("RdBu_r")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
if mom!="VRADH": cmap = "miub2"

plot_over_map = True
plot_ML = False

crs=ccrs.Mercator(central_longitude=float(datasel["longitude"]))

if not plot_over_map:
    # plot simple PPI
    datasel[mom].wrl.vis.plot( cmap=cmap, norm=norm, xlim=(-xylims,xylims), ylim=(-xylims,xylims))
    # datasel[mom].wrl.vis.plot( cmap="Blues", vmin=0, vmax=6, xlim=(-xylims,xylims), ylim=(-xylims,xylims))
elif plot_over_map:
    # plot PPI with map coordinates
    fig = plt.figure(figsize=(5, 5))
    datasel[mom].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))
    ax = plt.gca()
    ax.add_feature(cartopy.feature.COASTLINE, linestyle='-', linewidth=1, alpha=0.4)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1, alpha=0.4)
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"})

if plot_ML:
    cax = plt.gca()
    datasel["z"].wrl.vis.plot(ax=cax,
                          levels=[datasel["height_ml_bottom_new_gia"], datasel["height_ml_new_gia"]],
                          cmap="black",
                          func="contour")
    # datasel["z"].wrl.vis.plot(fig=fig, cmap=cmap, norm=norm, crs=ccrs.Mercator(central_longitude=float(datasel["longitude"])))

# add cities
for city, coord in cities.items():
    ax.plot(coord['lon'], coord['lat'], 'ro', markersize=5, transform=ccrs.PlateCarree(), color="black")  # Plot the city as a red dot
    ax.text(coord['lon'] + 0.02, coord['lat'], city, transform=ccrs.PlateCarree(), fontsize=12, color='black')  # Add city name

if isvolume:
    try:
        elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).sweep_fixed_angle.values, 2))+"°"
    except AttributeError:
        elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).elevation.mean().values, 2))+"°"
else:
    try:
        elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
    except AttributeError:
        elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).elevation.mean().values, 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values).split(".")[0])

#%% QVP
# Azimuthally averaged profiles of a conical volume measured at elevations between 10 and 20 degrees
# use 12 deg elevation

max_height = 12000 # max height for the qvp plots
tsel = ""
elevn = 7 # elevation index
plot_ML = False

datatype = "sim" # sim, obs or ori (original)
mom = "ZDR"
xylims = 180000 # xlim and ylim (from -xylims to xylims)

if datatype == "sim":
    ds = vol_emvorado_sim.copy()

elif datatype == "obs":
    ds = vol_emvorado_obs.copy()

elif datatype == "ori":
    ds = vol_dwd.copy()

else:
    raise Exception("select correct data source")

if "sweep_fixed_angle" in ds.dims:
    isvolume = True

if isvolume: # if more than one elevation, we need to select the one we want
    if tsel == "":
        datasel = ds.isel({"sweep_fixed_angle":elevn})
    else:
        datasel = ds.isel({"sweep_fixed_angle":elevn}).sel({"time": tsel}, method="nearest")
else:
    if tsel == "":
        datasel = ds
    else:
        datasel = ds.sel({"time": tsel}, method="nearest")

datasel = datasel.pipe(wrl.georef.georeference)
if "time" in datasel.z.dims:
    datasel = datasel.assign_coords(z=datasel.z.mean('time'))

qvps = utils.compute_qvp(datasel, min_thresh={"RHOHV":0.7, "DBZH":0, "ZDR":-1, "SNRH":10, "SNRHC":10, "SQIH":0.5}).compute()


# New Colormap
colors = ["#2B2540", "#4F4580", "#5a77b1",
          "#84D9C9", "#A4C286", "#ADAA74", "#997648", "#994E37", "#82273C", "#6E0C47", "#410742", "#23002E", "#14101a"]


ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
qvps[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3), ylim=(0, max_height))
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
if plot_ML:
    qvps["height_ml_new_gia"].plot(c="black")
    qvps["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

# # Plot reflectivity as lines to check wet radome effect
# ax2 = plt.gca().twinx()
# ds.DBZH.mean("azimuth")[:,4:7].mean("range").plot(ax=ax2, c="dodgerblue")
# ax2.yaxis.label.set_color("dodgerblue")
# ax2.spines['left'].set_position(('outward', 60))
# ax2.tick_params(axis='y', labelcolor="dodgerblue")  # Add padding to the ticks
# ax2.yaxis.labelpad = -400  # Add padding to the y-axis label
# ax2.set_title("")

# # Plot zdrcal values
# ax3 = plt.gca().twinx()
# zdrcal.loc[{"time":slice(str(qvps.time[0].values), str(qvps.time[-1].values))}].fNewZdrOffsetEstimate_dB.plot(ax=ax3, c="magenta")
# ax3.yaxis.label.set_color("magenta")
# ax3.spines['right'].set_position(('outward', 90))
# ax3.tick_params(axis='y', labelcolor="magenta")  # Add padding to the ticks
# # ax3.yaxis.labelpad = 10  # Add padding to the y-axis label
# ax3.set_title("")

elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).elevation.mean().values, 2))+"°"
plt.title(mom+elevtitle+". "+str(qvps.time.values[0]).split(".")[0])
plt.show()
plt.close()

#%% ML detection
window0, winlen0, xwin0, ywin0, fix_range, rng, azmedian, rhohv_thresh_gia = utils.get_phase_proc_params(path_dwd).values()
min_height=0
clowres0=True

max_height = 12000 # max height for the qvp plots
tsel = ""
elevn = 7 # elevation index
plot_ML = False

datatype = "sim" # sim, obs or ori (original)
mom = "ZDR"
xylims = 180000 # xlim and ylim (from -xylims to xylims)

if datatype == "sim":
    ds = vol_emvorado_sim.copy()

elif datatype == "obs":
    ds = vol_emvorado_obs.copy()

elif datatype == "ori":
    ds = vol_dwd.copy()

else:
    raise Exception("select correct data source")

if "sweep_fixed_angle" in ds.dims:
    isvolume = True

if isvolume: # if more than one elevation, we need to select the one we want
    if tsel == "":
        datasel = ds.isel({"sweep_fixed_angle":elevn})
    else:
        datasel = ds.isel({"sweep_fixed_angle":elevn}).sel({"time": tsel}, method="nearest")
else:
    if tsel == "":
        datasel = ds
    else:
        datasel = ds.sel({"time": tsel}, method="nearest")

datasel = datasel.pipe(wrl.georef.georeference)
if "time" in datasel.z.dims:
    datasel = datasel.assign_coords(z=datasel.z.mean('time'))

qvps = utils.compute_qvp(datasel, min_thresh={"RHOHV":0.7, "DBZH":0, "ZDR":-1, "SNRH":10, "SNRHC":10, "SQIH":0.5})


# Define thresholds
moments={"DBZH": (10., 60.), "RHOHV": (0.65, 1.), "PHIDP": (-20, 180)}

# Calculate ML
qvps = utils.melting_layer_qvp_X_new(qvps, moments=moments, dim="z", fmlh=0.3,
         xwin=xwin0, ywin=ywin0, min_h=min_height, rhohv_thresh_gia=rhohv_thresh_gia, all_data=True, clowres=clowres0)
qvps.coords["elevation"] = qvps.coords["elevation"].mean()

#### Attach ERA5 temperature profile
loc = utils.find_loc(utils.locs, path_dwd)
qvps = utils.attach_ERA5_TEMP(qvps, path=loc.join(utils.era5_dir.split("loc")))
# ds = utils.attach_ERA5_TEMP(ds, path=loc.join(utils.era5_dir.split("loc")))


#PLOT
plot_ML=True
plot_TEMP=None #[0, 4]
ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
qvps[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3), ylim=(0, max_height))
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
if plot_ML:
    qvps["height_ml_new_gia"].plot(c="black")
    qvps["height_ml_bottom_new_gia"].plot(c="black")
if type(plot_TEMP) is list:
    qvps["TEMP"].plot.contour(levels=plot_TEMP, x="time")
plt.gca().set_ylabel("height over sea level")
elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).elevation.mean().values, 2))+"°"
plt.title(mom+elevtitle+". "+str(qvps.time.values[0]).split(".")[0])
plt.show()
plt.close()

#%% Load QVPs
# Load only events with ML detected (pre-condition for stratiform)
ff_ML = "/automount/realpep/upload/jgiles/ICON_EMVORADO_radardata/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/qvps/*/*/*/hty/vol/ML_detected.txt"
ff_ML_glob = glob.glob(ff_ML)

ff = [glob.glob(os.path.dirname(fp)+"/*allsim*")[0] for fp in ff_ML_glob ]

ds_qvps = xr.open_mfdataset(ff)

# Conditions to clean ML height values
max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
time_window = 5 # set timestep window for the std computation (centered)
min_period = 3 # set minimum number of valid ML values in the window (centered)

cond_ML_bottom_change = abs(ds_qvps["height_ml_bottom_new_gia"].diff("time").compute())<max_change
cond_ML_bottom_std = ds_qvps["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_bottom_minlen = qvps["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

cond_ML_top_change = abs(ds_qvps["height_ml_new_gia"].diff("time").compute())<max_change
cond_ML_top_std = ds_qvps["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_top_minlen = qvps["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

# reduce to daily condition
# allcond_daily = allcond.resample(time="D").any().dropna("time")
allcond_daily = allcond.resample(time="D").sum().dropna("time")
allcond_daily = allcond_daily.where(allcond_daily, drop=True)

# Filter only events with clean ML (requeriment for stratiform) on a daily basis
# (not efficient and not elegant but I could not find other solution)
ds_qvps = ds_qvps.isel(time=[date.values in  allcond_daily.time.dt.date for date in ds_qvps.time.dt.date])


#%% Plot QPVs interactive, with matplotlib backend (working) fix in holoviews/plotting/mpl/raster.py (COPIED FROM plot_ppis_qvps_etc.py)
# this works with a manual fix in the holoviews files.
# In Holoviews 1.17.1, add the following to line 192 in holoviews/plotting/mpl/raster.py:
# if 'norm' in plot_kwargs: # vmin/vmax should now be exclusively in norm
#          	plot_kwargs.pop('vmin', None)
#          	plot_kwargs.pop('vmax', None)

hv.extension("matplotlib")

max_height = 12000 # max height for the qvp plots (necessary because of random high points and because of dropna in the z dim)

var_options = ['RHOHV', 'ZDR_OC', 'KDP_ML_corrected', 'ZDR',
               # 'TH','UPHIDP',  # not so relevant
#               'UVRADH', 'UZDR',  'UWRADH', 'VRADH', 'SQIH', # not implemented yet in visdict14
               # 'WRADH', 'SNRHC', 'URHOHV', 'SNRH',
                'KDP', 'RHOHV_NC', 'UPHIDP_OC']


vars_to_plot = ['DBZH', 'KDP', 'ZDR', 'RHOHV', 'PHIDP']

visdict14 = radarmet.visdict14

# define a function for plotting a custom discrete colorbar
def cbar_hook(hv_plot, _, cmap_extend, ticklist, norm, label):
    COLORS = cmap_extend
    BOUNDS = ticklist
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax = hv_plot.handles["axis"]
    fig = hv_plot.handles["fig"]
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.35])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap_extend, norm=norm),
        cax=cbar_ax,
        extend='both',
        ticks=ticklist[1:-1],
        # spacing='proportional',
        orientation='vertical',
        label=label,
    )


# Define the function to update plots
def update_plots(selected_day, show_ML_lines, show_min_entropy):
    selected_data = ds_qvps.sel(time=selected_day, z=slice(0, max_height)).dropna("z", how="all")
    available_vars = vars_to_plot

    plots = []

    for var in available_vars:
        ticks = visdict14[var]["ticks"]
        cmap = visdict14[var]["cmap"] # I need the cmap with extreme colors too here
        # cmap_list = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        cmap_extend = utils.get_discrete_cmap(ticks, cmap)
        ticklist = [-100]+list(ticks)+[100]
        norm = utils.get_discrete_norm(ticks, cmap_extend)

        subtitle = var
        if var == "ZDR_OC":
            # for the plot of ZDR_OC, put the value of the offset in the subtitle if it is daily
            if np.unique((selected_data["ZDR"]-selected_data["ZDR_OC"]).compute().median("z")).std() < 0.01:
                # if the std of the unique values of ZDR - ZDR_OC is < 0.1 we assume it is a daily offset
                subtitle = var+" (Offset: "+str(np.round((selected_data["ZDR"]-selected_data["ZDR_OC"]).compute().median().values,3))+")"
            else:
                subtitle = var+" (Offset: variable per timestep)"
        if var == "DBZH": # add elevation angle to DBZH panel
            subtitle = var+" (Elevation: "+str(np.round(selected_data['elevation'].mean().compute().values, 2))+"°)"

        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', title=subtitle,
            xlabel='Time', ylabel='Height (m)', colorbar=False,
            width=500, height=250, norm=norm, ylim=(None, max_height),
        ).opts(
                cmap=cmap_extend,
                color_levels=ticks.tolist(),
                clim=(ticks[0], ticks[-1]),
                xformatter = mpl.dates.DateFormatter('%H:%M'), # put only the hour in the x-axis
                hooks=[partial(cbar_hook, cmap_extend=cmap_extend, ticklist=ticklist, norm=norm, label=selected_data[var].units)],
#!!! TO DO: format tick labels so that only the hour is shown: https://discourse.holoviz.org/t/how-to-the-format-of-datetime-x-axis-ticks-in-matplotlib-backend/6344
            )

        # Add line plots for height_ml_new_gia and height_ml_bottom_new_gia
        if show_ML_lines:
            # this parts works better and simpler with holoviews directly instead of hvplot
            line1 = hv.Curve(
                (selected_data.time, selected_data["height_ml_new_gia"]),
                # line_color='black', line_width=2, line_dash='dashed', legend=False # bokeh naming?
            ).opts(color='black', linewidth=2, show_legend=False)
            line2 = hv.Curve(
                (selected_data.time, selected_data["height_ml_bottom_new_gia"]),
            ).opts(color='black', linewidth=2, show_legend=False)

            quadmesh = (quadmesh * line1 * line2)

        # Add shading for min_entropy when it's greater than 0.8
        if show_min_entropy:
            min_entropy_values = selected_data.min_entropy.where(selected_data.min_entropy>=0).dropna("z", how="all").compute()

            min_entropy_shading = min_entropy_values.hvplot.quadmesh(
                x='time', y='z',
                xlabel='Time', ylabel='Height (m)', colorbar=False,
                width=500, height=250,
            ).opts(
                    cmap=['#ffffff00', "#B5B1B1", '#ffffff00'],
                    color_levels=[0, 0.9,1, 1.1],
                    clim=(0, 1.1),
                    alpha=0.8
                )
            quadmesh = (quadmesh * min_entropy_shading)


        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/3)]),
                         pn.Row(*plots[round(nplots/3):round(nplots/3)*2]),
                         pn.Row(*plots[round(nplots/3)*2:]),
                         )
    return gridplot
    # return pn.Row(*plots)


# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str,
                                                value=date_range_str[0], width=600)

show_ML_lines_toggle = pn.widgets.Toggle(name='Show ML Lines', value=True)

show_min_entropy_toggle = pn.widgets.Toggle(name='Show Entropy over 0.9', value=True)

@pn.depends(selected_day_slider.param.value, show_ML_lines_toggle, show_min_entropy_toggle)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    show_ML_lines = show_ML_lines_toggle.value
    show_min_entropy = show_min_entropy_toggle.value
    plot = update_plots(selected_day, show_ML_lines, show_min_entropy)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
show_ML_lines_toggle.param.watch(update_plots_callback, 'value')
show_min_entropy_toggle.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_ML_lines = True
initial_min_entropy = True

plot_panel = pn.Row(update_plots(initial_day, initial_ML_lines, initial_min_entropy))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    pn.Row(show_ML_lines_toggle, show_min_entropy_toggle),
    plot_panel
)


layout.save("/user/jgiles/interactive_matplotlib.html", resources=INLINE, embed=True,
            max_states=1000, max_opts=1000)

#%% Filters (conditions for stratiform)
start_time = time.time()
print("Filtering stratiform conditions...")
#### Set variable names
X_DBZH = "DBZH"
X_TH = "DBZH"
X_RHO = "RHOHV_NC" # if RHOHV_NC is set here, it is then checked agains the original RHOHV in the next cell
X_ZDR = "ZDR_OC"
X_KDP = "KDP_ML_corrected"

# Filter only stratiform events (min entropy >= 0.8) and ML detected
# with ProgressBar():
#     qvps_strat = qvps.where( (qvps["min_entropy"]>=0.8) & (qvps.height_ml_bottom_new_gia.notnull()), drop=True).compute()

# Conditions to clean ML height values
max_change = 400 # set a maximum value of ML height change from one timestep to another (in m)
max_std = 200 # set a maximum value of ML std from one timestep to another (in m)
time_window = 5 # set timestep window for the std computation (centered)
min_period = 3 # set minimum number of valid ML values in the window (centered)

cond_ML_bottom_change = abs(qvps["height_ml_bottom_new_gia"].diff("time").compute())<max_change
cond_ML_bottom_std = qvps["height_ml_bottom_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_bottom_minlen = qvps["height_ml_bottom_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

cond_ML_top_change = abs(qvps["height_ml_new_gia"].diff("time").compute())<max_change
cond_ML_top_std = qvps["height_ml_new_gia"].rolling(time=time_window, min_periods=min_period, center=True).std().compute()<max_std
# cond_ML_top_minlen = qvps["height_ml_new_gia"].notnull().rolling(time=5, min_periods=3, center=True).sum().compute()>2

allcond = cond_ML_bottom_change * cond_ML_bottom_std * cond_ML_top_change * cond_ML_top_std

# Filter only fully stratiform pixels (min entropy >= 0.8 and ML detected)
qvps_strat = qvps.where( (qvps["min_entropy"]>=0.9).compute() & allcond, drop=True)
# Relaxed alternative: Filter qvps with at least 50% of stratiform pixels (min entropy >= 0.8 and ML detected)
qvps_strat_relaxed = qvps.where( ( (qvps["min_entropy"]>=0.9).sum("z").compute() >= qvps[X_DBZH].count("z").compute()/2 ) & allcond, drop=True)

# Filter out non relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[X_TH] > -10 )&
                                  (qvps_strat[X_KDP] > -0.1)&
                                  (qvps_strat[X_KDP] < 3)&
                                  (qvps_strat[X_RHO] > 0.7)&
                                  (qvps_strat[X_ZDR] > -1) &
                                  (qvps_strat[X_ZDR] < 3))

qvps_strat_relaxed_fil = qvps_strat_relaxed.where((qvps_strat_relaxed[X_TH] > -10 )&
                                  (qvps_strat_relaxed[X_KDP] > -0.1)&
                                  (qvps_strat_relaxed[X_KDP] < 3)&
                                  (qvps_strat_relaxed[X_RHO] > 0.7)&
                                  (qvps_strat_relaxed[X_ZDR] > -1) &
                                  (qvps_strat_relaxed[X_ZDR] < 3))

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

# We will put the final retrievals in a dict
try: # check if exists, if not, create it
    retrievals
except NameError:
    retrievals = {}

for stratname, stratqvp in [("stratiform", qvps_strat_fil.copy()), ("stratiform_relaxed", qvps_strat_relaxed_fil.copy())]:
    print("   ... for "+stratname)

    retrievals[stratname] = {}

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
    Dm_ice_zdp_kdp = -0.1 + 2*( (wrl.trafo.idecibel(stratqvp[X_DBZH])*(1-wrl.trafo.idecibel(X_ZDR)**-1 ) ) / (stratqvp[X_KDP]*Lambda) )**(1/2) # Ryzhkov and Zrnic (2019). Zdp = Z(1-ZDR**-1) from Carlin et al 2021

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
    retrievals[stratname][utils.find_loc(utils.locs, ff[0])] = xr.Dataset({"lwc_zh_zdr":lwc_zh_zdr,
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
total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")

#%% CFTDs Plot

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below and do not save.
auto_plot = False
savepath = "/automount/agradar/jgiles/images/CFTDs_emvorado_cases/stratiform/"

# Which to plot, qvps_strat_fil or qvps_strat_relaxed_fil
ds_to_plot = qvps_strat_fil.copy()

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

# Plot horizontally
# DMI
# Native worst-resolution of the data (for 1-byte moments)
# DBZH: 0.5 dB
# ZDR: 0.0625 dB
# KDP: complicated. From 0.013 at KDP approaching zero to 7.42 at extreme KDP. KDP min absolute value is 0.25 and max abs is 150 (both positive and negative)
# RHOHV: scales with a square root (finer towards RHOHV=1), so from 0.00278 at RHOHV=0.7 to 0.002 resolution at RHOHV=1
# PHIDP: 0.708661 deg
loc = utils.find_loc(utils.locs, ff[0])
if loc in ['afy', 'ank', 'gzt', 'hty', 'svs']:

    vars_to_plot = {"DBZH": [0, 45.5, 0.5],
                    "ZDR_OC": [-0.505, 2.05, 0.1],
                    "KDP_ML_corrected":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                    "RHOHV_NC": [0.9, 1.002, 0.002]}

    if auto_plot:
        vtp = [{"DBZH": [0, 45.5, 0.5],
                        "ZDR_OC": [-0.505, 2.05, 0.1],
                        "KDP_ML_corrected":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                        "RHOHV_NC": [0.9, 1.002, 0.002]},
               {"DBZH": [0, 45.5, 0.5],
                               "ZDR": [-0.505, 2.05, 0.1],
                               "KDP_CONV":  [-0.1, 0.55, 0.05], # [-0.1, 0.55, 0.05],
                               "RHOHV": [0.9, 1.002, 0.002]} ]
        ytlimlist = [-20, -50]

        add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
        savedict = {}
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
            rd=10 # arbitrarily large decimal position to round to (so it is actually not rounded)
            if "DBZH" in vv:
                so=True
                binsx2 = [0, 46, 1]
                rd = 1 # decimal position to round to
            if "ZDR" in vv:
                so=True
                binsx2 = [-0.5, 2.1, 0.1]
                rd=1
            if "KDP" in vv:
                so=True #True
                binsx2 = [-0.1, 0.52, 0.02]
                rd=2
            if "RHOHV" in vv:
                so = True
                binsx2 = [0.9, 1.005, 0.005]
                rd=3
            utils.hist2d(ax[nn], ds_to_plot[vv].sel(\
                                                    time=ds_to_plot['time'].dt.month.isin(selmonths)).round(rd),
                         ds_to_plot["TEMP"].sel(\
                                             time=ds_to_plot['time'].dt.month.isin(selmonths))+adjtemp,
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



# DWD
# plot CFTDs moments
elif loc in ['pro', 'tur', 'umd', 'ess']:

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

# If auto_plot is True, then produce and save the plots automatically based on
# default configurations (only change savepath and ds_to_plot accordingly).
# If False, then produce the plot as given below and do not save.
auto_plot = False
savepath = "/automount/agradar/jgiles/images/CFTDs/stratiform/"

# Which to plot, stratiform or stratiform_relaxed
ds_to_plot = retrievals["stratiform"].copy()

loc = utils.find_loc(utils.locs, ff[0]) # by default, plot only the histograms of the currently loaded QVPs.

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
IWC = "iwc_zh_t" # iwc_zh_t or iwc_zdr_zh_kdp
LWC = "lwc_zh_zdr" # lwc_zh_zdr (adjusted for Germany) or lwc_zh_zdr2 (S-band) or lwc_kdp
Dm_ice = "Dm_ice_zh" # Dm_ice_zh, Dm_ice_zh_kdp, Dm_ice_zdp_kdp
Dm_rain = "Dm_rain_zdr3" # Dm_rain_zdr, Dm_rain_zdr2 or Dm_rain_zdr3
Nt_ice = "Nt_ice_zh_iwc" # Nt_ice_zh_iwc, Nt_ice_zh_iwc2, Nt_ice_zh_iwc_kdp, Nt_ice_zh_iwc2_kdp
Nt_rain = "Nt_rain_zh_zdr" # Nt_rain_zh_zdr

vars_to_plot = {"IWC/LWC [g/m^{3}]": [-0.1, 0.82, 0.02], # [-0.1, 0.82, 0.02],
                "Dm [mm]": [0, 3.1, 0.1], # [0, 3.1, 0.1],
                "Nt [log10(1/L)]": [-2, 2.1, 0.1], # [-2, 2.1, 0.1],
                }

savedict = {"custom": None} # placeholder for the for loop below, not important

if auto_plot:
    ytlimlist = [-20, -50]
    add_relaxed = ["_relaxed" if "relaxed" in savepath else ""][0]
    savedict = {}
    for selseas in selseaslist:
        savedict.update(
                    {selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys.png": [ytlimlist[0],
                                                           "iwc_zh_t", "lwc_zh_zdr",
                                                           "Dm_ice_zh", "Dm_rain_zdr3",
                                                           "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_extended.png": [ytlimlist[1],
                                                                "iwc_zh_t", "lwc_zh_zdr",
                                                                "Dm_ice_zh", "Dm_rain_zdr3",
                                                                "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP.png": [ytlimlist[0],
                                                               "iwc_zdr_zh_kdp", "lwc_kdp",
                                                               "Dm_ice_zh_kdp", "Dm_rain_zdr3",
                                                               "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    selseas[0]+"/"+loc+"_cftd_stratiform"+add_relaxed+"_microphys_KDP_extended.png": [ytlimlist[1],
                                                               "iwc_zdr_zh_kdp", "lwc_kdp",
                                                               "Dm_ice_zh_kdp", "Dm_rain_zdr3",
                                                               "Nt_ice_zh_iwc", "Nt_rain_zh_zdr", selseas[1]],
                    }
                )

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


#%% TEST Load Julian S ICON-EMVORADO files and compare to my workflow

ff_js = "/automount/data02/agradar/operation_hydrometeors/data/Syn_vol/20210714/ASS_2411/MAIN_2411.1/EMVO_00510000.2/120min_spinup/EMV_Vol_ESS_*.nc"
data = utils.load_emvorado_to_radar_volume(ff_js, rename=True)
radar_volume=data.copy()

ff_icon_vol_js = "/automount/data02/agradar/operation_hydrometeors/data/Syn_vol/20210714/ASS_2411/MAIN_2411.1/ICONdata/120min_spinup/ICON_Vol_ESS_?????????????????????????.nc"
icon_vol_js = utils.load_emvorado_to_radar_volume(ff_icon_vol_js, rename=True)

ff_icon_js = "/automount/data02/agradar/operation_hydrometeors/data/mod/20210714/ASS_2411/MAIN_2411.1/ICONdata/20210714150000/main0200/fc_R19B07.*.RADOLAN"
ff_icon_z_js = '/automount/data02/agradar/operation_hydrometeors/data/mod/grid/vgrd_R19B07.RADOLAN.nc'
ff_icon_hgrid_js = "/automount/data02/agradar/operation_hydrometeors/data/mod/grid/hgrd_R19B07.RADOLAN.nc"

icon_field = utils.load_icon(ff_icon_js, ff_icon_z_js)

icon_hgrid = xr.open_dataset(ff_icon_hgrid_js)

icon_field = icon_field.assign_coords({"clon":icon_hgrid["clon"], "clat":icon_hgrid["clat"]})

icon_volume = utils.icon_to_radar_volume(icon_field, radar_volume)

# test
icon_volume = utils.icon_to_radar_volume(icon_field[["u", "v", "w", "temp", "clon", "clat", "z_ifc", "z_mc"]], radar_volume)

icon_volume_new = utils.icon_to_radar_volume_new(icon_field[["u", "v", "w", "temp", "clon", "clat", "z_ifc", "z_mc"]], radar_volume)

import pyinterp
from scipy.spatial import cKDTree
start_time = time.time()

data0 = icon_field["u"].isel(time=0).where(mask, drop=True).stack(stacked=['height', 'ncells'])
mesh = pyinterp.RTree(ecef=True)
mesh.packing(src, data0)
coords, values = mesh.value(trg, k=1, within=False)


tree = cKDTree(src)
_, indices = tree.query(coords[:,0,:], k=1)  # Find nearest neighbor indices
interp_field = []
vv = "u"
icon_field_vv_masked = icon_field[vv].where(mask, drop=True).stack(stacked=['height', 'ncells'])
for it in range(len(icon_field["time"])):
    data_interp = icon_field_vv_masked.isel(time=it)[indices]
    data_interp_reshape = data_interp.values.reshape(radar_volume["x"].shape)
    interp_field.append( xr.DataArray(data_interp_reshape,
                                          coords=radar_volume["x"].coords,
                                          dims=radar_volume["x"].dims,
                                          name=vv).expand_dims(dim={"time": icon_field["time"][it].expand_dims("time")}, axis=0)
                        )

icon_volume_test = xr.merge(interp_field)

total_time = time.time() - start_time
print(f"took {total_time/60:.2f} minutes.")


#%% TEST Load Julian S ICON-EMVORADO files and apply microphysics calculations

ff_js = "/automount/data02/agradar/operation_hydrometeors/data/Syn_vol/20210714/ASS_2411/MAIN_2411.1/EMVO_00510000.2/120min_spinup/EMV_Vol_ESS_*.nc"
data = utils.load_emvorado_to_radar_volume(ff_js, rename=True)
radar_volume=data.copy()

ff_icon_vol_js = "/automount/data02/agradar/operation_hydrometeors/data/Syn_vol/20210714/ASS_2411/MAIN_2411.1/ICONdata/120min_spinup/ICON_Vol_ESS_?????????????????????????.nc"
icon_vol_js = utils.load_emvorado_to_radar_volume(ff_icon_vol_js, rename=True)

ff_icon_js = "/automount/data02/agradar/operation_hydrometeors/data/mod/20210714/ASS_2411/MAIN_2411.1/ICONdata/20210714120000/main0200/fc_R19B07.*.RADOLAN"
ff_icon_z_js = '/automount/data02/agradar/operation_hydrometeors/data/mod/grid/vgrd_R19B07.RADOLAN.nc'
ff_icon_hgrid_js = "/automount/data02/agradar/operation_hydrometeors/data/mod/grid/hgrd_R19B07.RADOLAN.nc"

icon_field = utils.load_icon(ff_icon_js, ff_icon_z_js)

icon_hgrid = xr.open_dataset(ff_icon_hgrid_js)

icon_field = icon_field.assign_coords({"clon":icon_hgrid["clon"], "clat":icon_hgrid["clat"]})

# regridding to radar volume geometry
icon_volume = utils.icon_to_radar_volume(icon_field, radar_volume)
icon_volume["TEMP"] = icon_volume["temp"] - 273.15

# calculate microphysics
icon_volume_new = utils.calc_microphys(icon_volume, mom=2)

# Make QVP
elev = 7

radar_volume_new = xr.merge([radar_volume.sel(time=slice(icon_volume_new.time[0].values, icon_volume_new.time[-1].values)),
                             icon_volume_new])
qvps = utils.compute_qvp(radar_volume_new.isel(elevation=elev), min_thresh = {"RHOHV":0.9, "DBZH":0, "ZDR":-1, "SNRH":10,"SNRHC":10, "SQIH":0.5} )

#%% TEST Load DETECT ICON-EMVORADO files and apply microphysics calculations
ff = "/automount/realpep/upload/jgiles/ICON_EMVORADO_radardata/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/2020/2020-10/2020-10-14/pro/vol/cdfin_allsim_id-010392_*_volscan"
data = utils.load_emvorado_to_radar_volume(ff, rename=True)
radar_volume=data.copy()

ff_icon = "/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2020101322/out_EU-R13B5_inst_DOM01_ML_20201013T220000Z_1h.nc"
ff_icon_z = '/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_2020101322/out_EU-R13B5_constant_20201013T220000Z.nc'

icon_field = utils.load_icon(ff_icon, ff_icon_z)
icon_field['time'] = icon_field['time'].dt.round('1s') # round time coord to the second

# regridding to radar volume geometry
icon_volume = utils.icon_to_radar_volume(icon_field[["temp", "pres", "qv", "qc", "qi", "qr", "qs", "qg", "z_ifc"]], radar_volume)
icon_volume["TEMP"] = icon_volume["temp"] - 273.15

# calculate microphysics
icon_volume_new = utils.calc_microphys(icon_volume, mom=1)

# Make QVP
elev = 7

radar_volume_new = xr.merge([radar_volume.sel(time=slice(icon_volume_new.time[0].values, icon_volume_new.time[-1].values)),
                             icon_volume_new])
qvps = utils.compute_qvp(radar_volume_new.isel(elevation=elev), min_thresh = {"RHOHV":0.9, "DBZH":0, "ZDR":-1, "SNRH":10,"SNRHC":10, "SQIH":0.5} )

#%% PLOT QVP

ds_qvp = qvps

max_height = 12000 # max height for the qvp plots

tsel = ""# slice("2017-08-31T19","2017-08-31T22")
if tsel == "":
    datasel = ds_qvp.loc[{"z": slice(0, max_height)}]
else:
    datasel = ds_qvp.loc[{"time": tsel, "z": slice(0, max_height)}]

templevels = [-100, 0]
mom = "ZDR"

ticks = radarmet.visdict14[mom]["ticks"]
cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)
# norm = mpl.colors.BoundaryNorm(ticks, cmap.N, clip=False, extend="both")
cmap = "miub2"
norm = utils.get_discrete_norm(ticks, cmap, extend="both")
datasel[mom].wrl.plot(x="time", cmap=cmap, norm=norm, figsize=(7,3))
# figcontour = ds_qvp["TEMP"].plot.contour(x="time", y="z", levels=templevels)
# datasel["min_entropy"].compute().dropna("z", how="all").interpolate_na(dim="z").plot.contourf(x="time", levels=[0.8, 1], hatches=["", "XXX", ""], colors=[(1,1,1,0)], add_colorbar=False, extend="both")
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M')) # put only the hour in the x-axis
# datasel["height_ml_new_gia"].plot(c="black")
# datasel["height_ml_bottom_new_gia"].plot(c="black")
plt.gca().set_ylabel("height over sea level")

try:
    elevtitle = " "+str(np.round(ds_qvp["elevation"].values, 2))+"°"
except:
    elevtitle = " "+str(np.round(ds_qvp["elevation"].values, 2))+"°"

plt.title(mom+elevtitle+". "+str(datasel.time.values[0]).split(".")[0])
plt.show()
plt.close()


#%% OLD CODE FROM HERE
#%% Classify stratiform/convective precip

# for observed data
momobs = vol_emvorado_obs[0]["zrobs"].sel({"time":"2017-10-20"}).pipe(wrl.georef.georeference)
pclass_obs = list()
for i in range(len(momobs["time"])):
    print(str(i+1)+"/"+str(len(momobs["time"])))
    pclass_obs.append( classify_precip_typ(momobs[i].values, momobs[i].x.values/1000, momobs[i].y.values/1000) )

pcobs = xr.ones_like(momobs)
pcobs[:] = np.array(pclass_obs)
pcobs.attrs["Unit"] = ""
pcobs.attrs["units"] = pcobs.attrs["Unit"]
pcobs.attrs["Description"] = "Observed: stratiform (blue), convective (red)"
pcobs.attrs["long_name"] = pcobs.attrs["Description"]

# for simulated data
momsim = vol_emvorado_sim[0]["zrsim"].sel({"time":"2017-10-20"}).pipe(wrl.georef.georeference)
pclass_sim = list()
for i in range(len(momsim["time"])):
    print(str(i+1)+"/"+str(len(momobs["time"])))
    pclass_sim.append( classify_precip_typ(momsim[i].values, momsim[i].x.values/1000, momsim[i].y.values/1000) )

pcsim = xr.ones_like(momsim)
pcsim[:] = np.array(pclass_sim)
pcsim.attrs["Unit"] = ""
pcsim.attrs["units"] = pcsim.attrs["Unit"]
pcsim.attrs["Description"] = "Simulated: stratiform (blue), convective (red)"
pcsim.attrs["long_name"] = pcsim.attrs["Description"]



# plot
md = [{'ticks': np.array([0.5,1.5,2.5]),
        'contours': [0, 5, 10, 15, 20, 25, 30, 35],
        'cmap': mpl.colors.ListedColormap(["lightskyblue","crimson"]),
        'name': 'Stratiform (blue), convective (red)',
        'short': ''}]

plot_timestep_map(pcobs.to_dataset(), 4, varlist=["zrobs"], md=md)

plot_timestep_map(pcsim.to_dataset(), 7, varlist=["zrsim"], md=md)

#%% Load TSMP data

data = dict()

#### TSMP
# The timestamps of accumulated P are located at the center of the corresponding interval (1:30, 4:30, ...)
# Also, every monthly file has the last timestep from the previous month because of how the data is outputted by
# the model, so some data are overlapped. The overlaped data are negligible different (around the 5th decimal)

print("Loading TSMP...")
def preprocess_tsmp(ds): # discard the first timestep of every monthly file (discard overlapping data)
    return ds.isel({"time":slice(1,None)})

data["TSMP"] = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/o.data_v01/*/*TOT_PREC*',
                             preprocess=preprocess_tsmp, chunks={"time":1000})

data["TSMP"]["time"] = data["TSMP"].get_index("time").shift(-1.5, "H")

data["TSMPraw"] = xr.open_mfdataset('/automount/ags/jgiles/TSMP/rcsm_TSMP-ERA5-eval_IBG3/rawModelOutput/ERA5Climat_EUR11_ECMWF-ERA5_analysis_FZJ-IBG3_20170701/cosmo/*2017*',
                         compat="override", coords="minimal")

#%% Plot TSMP + emvorado

swp = swp_emvorado_sim

map_proj = ccrs.Mercator(central_longitude = swp.longitude.values) # ccrs.PlateCarree(central_longitude=0.0) # ccrs.Mercator(central_longitude = swp.longitude.values)
proj_utm = wrl.georef.epsg_to_osr(32633)

proj_utm = osr.SpatialReference()
_ = proj_utm.ImportFromEPSG(32633)

# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='10m',
    facecolor='none')


# PLOT EMVORADO
timestep = 14
plot_timestep_map(swp, timestep)



# PLOT EMVORADO CAPPI
# reload the emvorado volume data before doing this
ds = xr.concat([v.assign_coords({"elevation": v.attrs.get("fixed_angle")}) for v in vol_emvorado_sim], dim="elevation", coords="minimal")

# choose variable and timestep
vv = "zrsim"
timestep = -2

# Georeference the data
ds = ds.pipe(wrl.georef.georeference, proj=map_proj)

# define elevation and azimuth angles, ranges, radar site coordinates,
elevs  = [ds.fixed_angle for ds in vol_emvorado_sim]
sitecoords = (float(vol_emvorado_sim.site.longitude),
              float(vol_emvorado_sim.site.latitude),
              float(vol_emvorado_sim.site.altitude))
polxyz = wrl.vpr.volcoords_from_polar(sitecoords, elevs,
                                      ds.azimuth.values, ds.range.values, proj_utm)


# now we define the coordinates for the 3-D grid (the CAPPI layers)

# x = np.linspace(ds.x.min(), ds.x.max(), ds.range.shape[0]*2)

# y = np.linspace(ds.y.min(), ds.y.max(), ds.range.shape[0]*2)

# z = np.arange(500.,10500.,500.)

# xyz = wrl.util.gridaspoints(z, y, x)

# gridshape = (len(z), len(y), len(x))

maxrange = ds.range.max().values
maxalt = 5000.
horiz_res = 2000.
vert_res = 250.
trgxyz, trgshape = wrl.vpr.make_3d_grid(
    sitecoords, proj_utm, maxrange, maxalt, horiz_res, vert_res
)

# containers to hold Cartesian bin coordinates and data
xyz, dt = np.array([]).reshape((-1, 3)), np.array([])
# iterate over 14 elevation angles
for i in range(len(ds.elevation)):
    # get the scan metadata for each elevation
    # where = raw["dataset%d/where" % (i + 1)]
    # what = raw["dataset%d/data1/what" % (i + 1)]
    # # define arrays of polar coordinate arrays (azimuth and range)
    # az = np.arange(0.0, 360.0, 360.0 / where["nrays"])
    # # rstart is given in km, so multiply by 1000.
    # rstart = where["rstart"] * 1000.0
    # r = np.arange(rstart, rstart + where["nbins"] * where["rscale"], where["rscale"])
    # # derive 3-D Cartesian coordinate tuples
    xyz_ = wrl.vpr.volcoords_from_polar(sitecoords, ds[vv][i]["elevation"].values, ds.azimuth.values, ds.range.values, proj_utm)
    # get the scan data for this elevation
    #   here, you can do all the processing on the 2-D polar level
    #   e.g. clutter elimination, attenuation correction, ...
    data_ = ds[vv][i, timestep]# what["offset"] + what["gain"] * raw["dataset%d/data1/data" % (i + 1)]
    # transfer to containers
    xyz, dt = np.vstack((xyz, xyz_)), np.append(dt, np.ravel(data_))

# Make the new grid interpolation
gridder = wrl.vpr.CAPPI(xyz, trgxyz, trgshape, maxrange=ds.range.max().values,
                            minelev=ds.elevation.min().values, maxelev=ds.elevation.max().values,
                            ipclass=wrl.ipol.Idw)

# gridded = np.ma.masked_invalid( gridder(dt) ).reshape(trgshape)
gridded = gridder(dt).reshape(trgshape)

# plot
trgx = trgxyz[:, 0].reshape(trgshape)[0, 0, :]
trgy = trgxyz[:, 1].reshape(trgshape)[0, :, 0]
trgz = trgxyz[:, 2].reshape(trgshape)[:, 0, 0]

# wrl.vis.plot_max_plan_and_vert(
#     trgx, trgy, trgz, gridded, unit="dBZH", levels=range(-32, 60)
# )


hgt = 2000. # choose a height to plot

f, ax1 = plt.subplots(1, 1, figsize=(5.5, 4))
plt.pcolormesh(trgx, trgy, gridded[np.nonzero(trgz==hgt)[0][0]], vmin=-10)
plt.colorbar()
plt.title(vv+" at "+str(hgt)+" m")




# PLOT TSMP
# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25)
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

f, ax1 = plt.subplots(1, 1, figsize=(5.5, 4), subplot_kw=dict(projection=map_proj))

# date = pd.to_datetime(swp.time[timestep].values)
# plot = data["TSMP"]["TOT_PREC"].loc[{ "time": str(date) }][0].plot.contourf(transform = rp, levels=5, vmin=0, vmax=0.1, cmap="YlGnBu")
plot = data["TSMP"]["TOT_PREC"].loc[{ "time": "2017-10-07 12:00" }][0].plot.contourf(transform = rp, levels=5, vmin=0.1, vmax=2, cmap="YlGnBu")

plt.gca().set_extent([11, 17, 51, 54.5])
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries



# PLOT TSMP raw
# add vcoord as coordinate for the "level" dimension
data["TSMPraw"].coords["level"] = (data["TSMPraw"]["vcoord"][0,:-1] + data["TSMPraw"]["vcoord"][0,1:]).compute().values/2
dataplot = data["TSMPraw"].sel({"level":hgt}, method="nearest")
vv= "QC" # QV specific hum, QC specific cloud liquid water content, QI specific cloud ice content, QS specific snow content , QR specific rain content
f, ax1 = plt.subplots(1, 1, figsize=(5.5, 4), subplot_kw=dict(projection=map_proj))

date = pd.to_datetime(swp.time[timestep].values)
# for the max value between elevations:
#plot = data["TSMPraw"][vv].loc[{ "time": str(date) }][0,-22:].max("level").plot.contourf(transform = rp, levels=5, vmax=1E-5)
# for a single elevation:
plot = dataplot[vv].loc[{ "time": str(date) }][0].plot.contourf(transform = rp, levels=5, vmax=1E-5, ax=ax1, cmap="YlGnBu")
# for a single elevation, changing mm/s to mm/3h (for precip rate):
# plot = (dataplot[vv].loc[{ "time": str(date) }][0]*60*60*3).plot.contourf(transform = rp, levels=5, vmax=0.1, ax=ax1, cmap="YlGnBu") #Multiplied by 60*60*3 for accumulated values in 3h
# same as before but for manually chosen date:
# plot = (dataplot[vv].loc[{ "time": "2017-07-27 18" }][0]*60*60*3).plot.contourf(transform = rp, levels=5, vmax=0.1, ax=ax1, cmap="YlGnBu") #Multiplied by 60*60*3 for accumulated values in 3h

ax1.set_extent([11, 17, 51, 54.5])
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries

#%% Plot a movie of model precip to search stratiform events

import hvplot.xarray
from bokeh.resources import INLINE

#### I tried several ways of making an html interactive plot but I got several different errors using hvplot


# first try
plot = data["TSMP"]["TOT_PREC"][0:10].hvplot.quadmesh("rlon","rlat", dynamic=True, groupby="time", projection="PlateCarree", project=True)

plot = data["TSMP"]["TOT_PREC"][0:10].hvplot.quadmesh("rlon","rlat", dynamic=True, groupby="time", projection="Mercator", project=True, geo=True, crs=rp)

hvplot.save(plot, 'test.html')


# second try
air_ds = xr.tutorial.open_dataset('air_temperature').load()

plot = air_ds.isel({"time":slice(0,10)}).hvplot.imshow(
    'lon', 'lat', 'air', cmap='viridis', projection=ccrs.Mercator())
hvplot.save(plot, 'test.html')

# third try
import holoviews as hv
hv_plot =  air_ds.isel({"time":slice(0,10)}).hvplot.quadmesh(x='lon', y='lat', dynamic=True,groupby='time', rasterize=True, cmap='viridis', projection=ccrs.Mercator())#.options(projection=ccrs.Mercator())

hv_plot =  air_ds.isel({"time":slice(0,10)}).hvplot.quadmesh(x='lon', y='lat', dynamic=False, geo=True, project=True, projection="Mercator")#.options(projection=ccrs.Mercator())
# save the plot as an HTML file
hv.save(hv_plot, 'test.html')


# fourth try, a movie

# create the plot function
# dataplot = data["TSMP"]["TOT_PREC"].sel({"time":slice("2015-01-01", "2020-12-31")})
dataplot = data["TSMP"]["TOT_PREC"].sel({"time":slice("2017-01-01", "2017-12-31")})
def plot_data(i):

    plt.subplot(111, projection=map_proj)
    # plot = data["TSMP"]["TOT_PREC"][i].plot(transform = rp, levels=5, vmin=0, vmax=0.1, cmap="YlGnBu")
    plot = dataplot[i].plot(transform = rp, norm=mpl.colors.LogNorm(vmin=0.001, vmax=10), cmap="YlGnBu")

    #plt.gca().set_extent([11, 17, 51, 54.5])
    plot.axes.coastlines()
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
    plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries


# create the animation
fig = plt.figure(figsize=(5.5, 4))
# ani = mpl.animation.FuncAnimation(fig, plot_data, frames=len(data["TSMP"]["TOT_PREC"].time), repeat=False)
ani = mpl.animation.FuncAnimation(fig, plot_data, frames=len(dataplot["time"]), repeat=False)

# save the animation as a video file
ani.save("test.mp4", writer='ffmpeg', fps=5)


#%% Plot composites over Germany
proj = ccrs.PlateCarree(central_longitude=0.0)

# Set rotated pole
# Euro-CORDEX rotated pole coordinates RotPole (198.0; 39.25)
rp = ccrs.RotatedPole(pole_longitude=198.0,
                      pole_latitude=39.25,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))

comp_tsmp = xr.open_mfdataset("/automount/realpep/upload/jgiles/ICON_EMVORADO_test/det/radout/dbzcmp_sim_*.grb2", engine = "cfgrib", concat_dim="time", combine="nested")

comp_tsmp["time"] = comp_tsmp["valid_time"]

seltime = "2022-08-14T19:00:00"
nelev = 0

comp_plot = comp_tsmp.DBZCMP_SIM.sel({"time": seltime})[nelev]

# plot single timestep
# cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
f, ax1 = plt.subplots(1, 1, figsize=(5.5, 4), subplot_kw=dict(projection=proj))
# plot = comp_plot.where(comp_plot>-999).plot(x="longitude", y="latitude", levels=visdict14["DBZH"]["ticks"], cmap=cmap, extend="both", transform=rp)
plot = comp_plot.where(comp_plot>-999).plot(ax=ax1, x="longitude", y="latitude")

# plt.gca().set_extent([4.5, 16, 46, 56]) # Germany
plt.gca().set_extent([4.5, 40, 34, 56])
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(seltime)


# make a movie
dataplot = comp_tsmp.DBZCMP_SIM.sel({"time":slice("2017-10-02", "2017-10-02 21:00")})
cmap = radarmet.get_discrete_cmap(radarmet.visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
def plot_data(i):
    dataplot2 = dataplot[i, nelev]
    plt.subplot(111, projection=ccrs.PlateCarree())
    plot = dataplot2.where(dataplot2>-999).plot(x="longitude", y="latitude", levels=radarmet.visdict14["DBZH"]["ticks"], cmap=cmap, extend="both")

    plt.gca().set_extent([4.5, 16, 46, 56])
    plot.axes.coastlines()
    plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
    plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
    plt.title(dataplot2.time.values)


# create the animation
fig = plt.figure(figsize=(5.5, 4))
ani = mpl.animation.FuncAnimation(fig, plot_data, frames=len(dataplot["time"]), repeat=True)

# save the animation as a video file
ani.save("DBZH_comp_tsmp_germany"+seltime.split(" ")[0]+".mp4", writer='ffmpeg', fps=3)


#%% Load some Turkish data (test)

htypath = [
            "/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/HTY170727000019.RAW68TW",
            # "/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/HTY170727000050.RAW68UP",
            # "/home/jgiles/turkey_test/acq/OLDDATA/uza/RADAR/2017/07/27/HTY/RAW/HTY170727000119.RAW68V4",
           ]
vol_hty3 = wrl.io.open_iris_mfdataset(htypath, reindex_angle=False)

vol_hty5 = wrl.io.open_iris_dataset(htypath[0], reindex_angle=False)
