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
import hvplot
import hvplot.xarray
import holoviews as hv

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    # from Scripts.python.radar_processing_scripts import colormap_generator
except ModuleNotFoundError:
    import utils
    import radarmet
    # import colormap_generator


os.environ['WRADLIB_DATA'] = '/home/jgiles/wradlib-data-main'
# set earthdata token (this may change, only lasts a few months https://urs.earthdata.nasa.gov/users/jgiles/user_tokens)
os.environ["WRADLIB_EARTHDATA_BEARER_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImpnaWxlcyIsImV4cCI6MTcwMzMzMjE5NywiaWF0IjoxNjk4MTQ4MTk3LCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.6DB5JJ9vdC7Vvwvaa7_mb_HbpVAh05Gz26dzdateN10C5lAd2X4a1_zClx7KkTpyoeVZSzkGSgtcd5Azc_btG0am4r2aJDGv4Zp4Vg55G4mcZMp-aTR7D520InQLMvqFacVO5wwmvfNWzMT4TyLGcXwPuX58s1oaFR5gRL9T30pXN9nEs-1aJg4LUl553PfdOvvom3q-JKXFtSTE2nLyEQOzWW36COl1aHwq6Wh4ykn4aq6ppTVAIeHdgkjtnQtxbhd9trm16fSbX9HIgG7n-drnz_v-WMeFuycMHa-zLDKnd3U3oZW6XAUq2akw2ddu6ChwoTZ4Ix2di7fudioo9Q"

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

#%% Load QVPs
ff = "/automount/realpep/upload/jgiles/dwd/qvps/2015/*/*/pro/vol5minng01/07/*allmoms*"
ds_qvps = utils.load_qvps(ff)

#%% Plot QPVs interactive (working)
import panel as pn
from bokeh.resources import INLINE

selday = "2015-01-02"

var_options = ['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected',
               'UVRADH', 'UZDR', 'ZDR', 'UWRADH', 'TH', 'VRADH', 'SQIH',
               'WRADH', 'UPHIDP', 'KDP', 'SNRHC', 'SQIH',
                'URHOHV', 'SNRH', 'RHOHV_NC', 'UPHIDP_OC']

var_starting = ['DBZH', 'ZDR_OC', 'KDP_ML_corrected', "RHOHV_NC"]

# Define the function to update plots
def update_plots(selected_day, selected_vars):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = selected_vars

    plots = []

    for var in available_vars:
        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', cmap='viridis', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=True
        ).opts(width=800, height=400)

        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

selected_vars_selector = pn.widgets.CheckBoxGroup(name='Select Variables', 
                                                  value=var_starting, 
                                                  options=var_options,
                                                  inline=True)

# # this works but the file is so large that it is not loading in Firefox or Chrome
# selected_vars_selector = pn.widgets.Select(name='Select Variables', 
#                                                   value="ZDR", 
#                                                   options=["ZDR", "ZDR_OC", "UZDR"],
#                                                   )


@pn.depends(selected_day_slider.param.value)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_vars = selected_vars_selector.value
    plot = update_plots(selected_day, selected_vars)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
selected_vars_selector.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_vars = var_starting
# initial_vars = "ZDR"
plot_panel = pn.Row(update_plots(initial_day, initial_vars))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    # selected_vars_selector, # works with pn.widgets.Select but creates too-large files that do not load
    plot_panel
)


# Display or save the plot as an HTML file
# pn.serve(layout)

layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)

# layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
#             states={"Select Date":date_range_str, "Select Variables": var_options}, 
#             max_states=1000, max_opts=1000)


#%% Plot QVPs interactive (testing)
from functools import partial
import panel as pn
from bokeh.resources import INLINE
from bokeh.models import FixedTicker
from bokeh.models import CategoricalColorMapper, ColorBar
from bokeh.colors import Color

selday = "2015-01-02"

var_options = ['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected',
               'UVRADH', 'UZDR', 'ZDR', 'UWRADH', 'TH', 'VRADH', 'SQIH',
               'WRADH', 'UPHIDP', 'KDP', 'SNRHC', 'SQIH',
                'URHOHV', 'SNRH', 'RHOHV_NC', 'UPHIDP_OC']

var_starting = ['DBZH', 'ZDR_OC', "ZDR", 'KDP_ML_corrected', "RHOHV_NC", "RHOHV"]
var_starting = ['DBZH', 'ZDR_OC', 'KDP_ML_corrected', "RHOHV_NC"]

visdict14 = radarmet.visdict14


# Define the function to update plots
def update_plots(selected_day, selected_vars):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = selected_vars

    plots = []

    # define a function for plotting a discrete colorbar with equal color ranges
    def cbar_hook(hv_plot, _, cmap, ticklist):
        COLORS = [mpl.colors.rgb2hex(cc, keep_alpha=True) for cc in cmap.colors]
        BOUNDS = ticklist
        plot = hv_plot.handles["plot"]
        factors = [f"{BOUNDS[i]} - {BOUNDS[i + 1]}" for i in range(len(COLORS))]
        mapper = CategoricalColorMapper(
            palette=COLORS,
            factors=factors,
        )
        color_bar = ColorBar(color_mapper=mapper)
        plot.right[0] = color_bar


    for var in available_vars:
        # set colorbar settings
        ticklist = list(visdict14[var]["ticks"])
        # ticklist = [-100]+list(visdict14[var]["ticks"])+[100]
        norm = utils.get_discrete_norm(visdict14[var]["ticks"])
        cmap = utils.get_discrete_cmap(visdict14[var]["ticks"], visdict14[var]["cmap"]) #mpl.cm.get_cmap("HomeyerRainbow")
        # cmap = visdict14[var]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

        quadmesh = selected_data[var].hvplot.quadmesh(
            x='time', y='z', cmap='viridis', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=True
        ).opts(width=800, height=400, 
               cmap=cmap, color_levels=ticklist, clim=(ticklist[0], ticklist[-1]), 
                colorbar_opts = {'ticker': FixedTicker(ticks=ticklist),},  # this changes nothing
               # hooks=[partial(cbar_hook, cmap=cmap, ticklist=ticklist)],
               )

        plots.append(quadmesh)

    nplots = len(plots)
    gridplot = pn.Column(pn.Row(*plots[:round(nplots/2)]),
                         pn.Row(*plots[round(nplots/2):]),
                         )
    return gridplot
    # return pn.Row(*plots)

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

date_range_str = list(np.unique([str(date0.date()) for date0 in date_range]))

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DiscreteSlider(name='Select Date', options=date_range_str, value=date_range_str[0])

selected_vars_selector = pn.widgets.CheckBoxGroup(name='Select Variables', 
                                                  value=var_starting, 
                                                  options=var_options,
                                                  inline=True)

# # this works but the file is so large that it is not loading in Firefox or Chrome
# selected_vars_selector = pn.widgets.Select(name='Select Variables', 
#                                                   value="ZDR", 
#                                                   options=["ZDR", "ZDR_OC", "UZDR"],
#                                                   )


@pn.depends(selected_day_slider.param.value)
# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = str(selected_day_slider.value)
    selected_vars = selected_vars_selector.value
    plot = update_plots(selected_day, selected_vars)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
selected_vars_selector.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = str(start_date)
initial_vars = var_starting
# initial_vars = "ZDR"
plot_panel = pn.Row(update_plots(initial_day, initial_vars))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    # selected_vars_selector, # works with pn.widgets.Select but creates too-large files that do not load
    plot_panel
)


# Display or save the plot as an HTML file
# pn.serve(layout)

layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
            max_states=1000, max_opts=1000)

# layout.save("/user/jgiles/interactive.html", resources=INLINE, embed=True, 
#             states={"Select Date":date_range_str, "Select Variables": var_options}, 
#             max_states=1000, max_opts=1000)


#%% Plot QVPs interactive (example to replicate)
import panel as pn


# Define the function to update plots
def update_plots(selected_day, selected_vars, show_height_lines, show_min_entropy):
    selected_data = ds_qvps.sel(time=selected_day)
    available_vars = selected_vars

    plots = []

    for var in available_vars:
        quadmesh = selected_data.hvplot.quadmesh(
            x='time', y='z', C=var, cmap='viridis', title=var,
            xlabel='Time', ylabel='Height (m)', colorbar=True
        ).opts(width=400, height=400)

        # Add line plots for height_ml_new_gia and height_ml_bottom_new_gia
        if show_height_lines:
            line1 = selected_data.hvplot.line(
                x='time', y='height_ml_new_gia',
                line_color='red', line_width=2, line_dash='dashed', legend=False
            )
            line2 = selected_data.hvplot.line(
                x='time', y='height_ml_bottom_new_gia',
                line_color='blue', line_width=2, line_dash='dotted', legend=False
            )
            quadmesh = (quadmesh * line1 * line2).opts(legend_position='top_left')

        # Add hatches for min_entropy when it's greater than 0.8
        if show_min_entropy:
            min_entropy_values = selected_data.min_entropy
            # min_entropy_hatches = hv.QuadMesh(min_entropy_values > 0.8, kdims=["time", "z"])
            # quadmesh = (quadmesh * min_entropy_hatches.opts(hatching='true')).opts(legend_position='top_left')

        plots.append(quadmesh)

    return pn.Row(*plots)

# Convert the date range to a list of datetime objects
date_range = pd.to_datetime(ds_qvps.time.data)
start_date = date_range.min().date()
end_date = date_range.max().date()

# Create widgets for variable selection and toggles
selected_day_slider = pn.widgets.DateSlider(name='Select Date', start=start_date, end=end_date)
selected_vars_selector = pn.widgets.CheckButtonGroup(name='Select Variables', value=['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected'], options=[w for w in list(ds_qvps.data_vars) if w.isupper()])
show_height_lines_toggle = pn.widgets.Toggle(name='Show Height Lines', value=True)
show_min_entropy_toggle = pn.widgets.Toggle(name='Show Min Entropy Hatches', value=True)

# Define the function to update plots based on widget values
def update_plots_callback(event):
    selected_day = selected_day_slider.value
    selected_vars = selected_vars_selector.value
    show_height_lines = show_height_lines_toggle.value
    show_min_entropy = show_min_entropy_toggle.value
    plot = update_plots(selected_day, selected_vars, show_height_lines, show_min_entropy)
    plot_panel[0] = plot

selected_day_slider.param.watch(update_plots_callback, 'value')
selected_vars_selector.param.watch(update_plots_callback, 'value')
show_height_lines_toggle.param.watch(update_plots_callback, 'value')
show_min_entropy_toggle.param.watch(update_plots_callback, 'value')

# Create the initial plot
initial_day = ds_qvps.time.min().data
initial_vars = ['DBZH', 'RHOHV', 'ZDR_OC', 'KDP_ML_corrected']
initial_height_lines = True
initial_min_entropy = True
plot_panel = pn.Row(update_plots(initial_day, initial_vars, initial_height_lines, initial_min_entropy))

# Create the Panel layout
layout = pn.Column(
    selected_day_slider,
    selected_vars_selector,
    show_height_lines_toggle,
    show_min_entropy_toggle,
    plot_panel
)

# Display or save the plot as an HTML file
layout.save("/user/jgiles/interactive.html")

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

#%% Timeseries of ZDR offsets

loc0 = "umd"

## Load
f_VPzdroff_below3c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_below3C_00*"
f_VPzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_belowML_00*"
f_VPzdroff_wholecol = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_wholecol_00*"

f_LRzdroff_below3c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_below3C_07*"
f_LRzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/LR_consistency/20*/*/*/"+loc0+"/vol5minng01/07/*_zdr_offset_belowML_07*"

VPzdroff_below3c = xr.open_mfdataset(f_VPzdroff_below3c)
VPzdroff_belowML = xr.open_mfdataset(f_VPzdroff_belowML)
VPzdroff_wholecol = xr.open_mfdataset(f_VPzdroff_wholecol)

LRzdroff_below3c = xr.open_mfdataset(f_LRzdroff_below3c)
LRzdroff_below3c = LRzdroff_below3c.where(abs(LRzdroff_below3c["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero
LRzdroff_belowML = xr.open_mfdataset(f_LRzdroff_belowML)
LRzdroff_belowML = LRzdroff_belowML.where(abs(LRzdroff_belowML["ZDR_offset"])>0.00049) # special filtering since the NA values are set to a fix float close to zero


## Plot

VPzdroff_below3c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below 3C")
VPzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below ML")
VPzdroff_wholecol["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP whole col")

LRzdroff_below3c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below 3C")
LRzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="ZH-ZDR below ML")

# plt.legend()
lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle.set_sizes([6.0])

plt.ylabel("ZDR offsets")    
plt.ylim(-1,1)
plt.title(loc0.upper())


## Plot interactively
# put everything in the same dataset
zdroffsets = xr.merge([VPzdroff_below3c["ZDR_offset"].rename("VP below 3C"), 
                       VPzdroff_belowML["ZDR_offset"].rename("VP below ML"),
                       VPzdroff_wholecol["ZDR_offset"].rename("VP whole col"), 
                       LRzdroff_below3c["ZDR_offset"].rename("ZH-ZDR below 3C"),
                       LRzdroff_belowML["ZDR_offset"].rename("ZH-ZDR below ML"),
                       ],)
# add also rolling median of the offsets
zdroffsets_rollmed = xr.merge([
                       zdroffsets["VP below 3C"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP below 3C rolling median"), 
                       zdroffsets["VP below ML"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP below ML rolling median"),
                       zdroffsets["VP whole col"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("VP whole col rolling median"), 
                       zdroffsets["ZH-ZDR below 3C"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below 3C rolling median"),
                       zdroffsets["ZH-ZDR below ML"].compute().interpolate_na("time").rolling(time=31, center=True, min_periods=10).median().rename("ZH-ZDR below ML rolling median"),
                       ],)



# Create an interactive scatter plot from the combined Dataset
scatter = zdroffsets.hvplot.scatter(x='time', 
                                    y=['VP below 3C', 'VP below ML', 'VP whole col', 
                                       'ZH-ZDR below 3C', 'ZH-ZDR below ML'],
                                    width=1000, height=400, size=1, muted_alpha=0)

lines = zdroffsets_rollmed.hvplot.line(x='time', 
                                    y=['VP below 3C rolling median', 'VP below ML rolling median', 'VP whole col rolling median', 
                                       'ZH-ZDR below 3C rolling median', 'ZH-ZDR below ML rolling median'],
                                    width=1000, height=400, muted_alpha=0)

# Combine both plots
overlay = (scatter * lines)
# overlay=lines

# Customize the plot (add titles, labels, etc.)
overlay.opts(title="ZDR offsets and 31 day centered rolling median (10 day min, interpolated where NaN)", xlabel="Time", ylabel="ZDR offsets", show_grid=True)

# Save the interactive plot to an HTML file
hv.save(overlay, "/user/jgiles/interactive_plot.html")


## Timeseries of ZDR VP offset above vs below ML
## Load
f_VPzdroff_belowML = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_belowML_00*"
f_VPzdroff_above0c = "/automount/realpep/upload/jgiles/dwd/calibration/zdr/VP/20*/*/*/"+loc0+"/90gradstarng01/00/*_zdr_offset_above0C_00*"

VPzdroff_belowML = xr.open_mfdataset(f_VPzdroff_belowML)
VPzdroff_above0c = xr.open_mfdataset(f_VPzdroff_above0c)


## Plot

VPzdroff_belowML["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP below ML")
VPzdroff_above0c["ZDR_offset"].plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="VP above 0C")

# plt.legend()
lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle.set_sizes([6.0])

plt.title("")
plt.ylabel("ZDR offsets")    
plt.ylim(-1,1)

## Plot difference

(VPzdroff_belowML["ZDR_offset"]-VPzdroff_above0c["ZDR_offset"]).plot.scatter(x="time", s=1, edgecolors=None, linewidths=0, label="below-above ML diff")

# add +-0.1 lines
(xr.ones_like(VPzdroff_belowML["ZDR_offset"])*0.1).plot(x="time", c="black", label="0.1")
(xr.ones_like(VPzdroff_belowML["ZDR_offset"])*-0.1).plot(x="time", c="black", label="-0.1")

# plt.legend()
lgnd = plt.legend()
lgnd.legend_handles[0].set_sizes([6.0])

plt.grid()
plt.title("")
plt.ylabel("ZDR offsets")    
plt.ylim(-0.3,0.3)


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

#%% Testing my version of the raw DWD data with the other one that has more moments
mydata = utils.load_dwd_raw("/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/pro/2016/2016-05-27/pro/vol5minng01/05/*")
otherdata = utils.load_dwd_raw("/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2016/2016-05/2016-05-27/pro/vol5minng01/05/*")

xr.testing.assert_allclose(mydata.SQIV[0], otherdata.SQIV[0])

mydata.DBZH[0].plot()
otherdata.DBZH[0].plot()

(mydata.UVRADH[0] - otherdata.UVRADH[0]).plot()

# check raw metadata
mydataraw = dttree.open_datatree(glob.glob("/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/pro/2016/2016-05-27/pro/vol5minng01/05/*dbzh*")[0])
otherdataraw = dttree.open_datatree(glob.glob("/automount/realpep/upload/RealPEP-SPP/DWD-CBand/2016/2016-05/2016-05-27/pro/vol5minng01/05/*dbzh*")[0])
