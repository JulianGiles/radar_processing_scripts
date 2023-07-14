#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:43:10 2023

@author: jgiles
"""
# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr
import os
import datetime as dt
import pandas as pd
from tqdm.notebook import trange, tqdm

import warnings
warnings.filterwarnings('ignore')
import xradar as xd
import datatree as dttree

import netCDF4
import packaging

import time
start_time = time.time()

import os
os.chdir('/home/jgiles/')

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


# Plotting libs
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
import hvplot.xarray # noqa
import hvplot
import panel as pn
import holoviews as hv
from bokeh.models.tickers import FixedTicker


#%% Load the daily file
# Set Engine
# engine = "netcdf4"
engine = "h5netcdf"

# dsrpckd = dttree.open_datatree("/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/07/ras07-vol5minng01_sweeph5onem_allmoms_07-2017072500033500-pro-10392-hd5")["sweep_7"].to_dataset()
# dsrunpckd = xr.open_dataset(f"/home/jgiles/turkey_test/test1_-iris-test-compressed-{engine}.nc")
# dsrpckd = xr.open_dataset(f"/home/jgiles/turkey_test/testank_-iris-test-compressed-{engine}.nc")
dsrpckd = xr.open_dataset(f"/automount/realpep/upload/jgiles/dmi/pulled_from_detect_ank/2015/2015-09/2015-09-30/ANK/MON_YAZ_K/12.0/MON_YAZ_K-allmoms-12.0-20152015-092015-09-30-ANK-h5netcdf.nc")
# dsrpckd = xr.open_dataset("/automount/realpep/upload/jgiles/dmi/2017/2017-12/2017-12-24/HTY/VOL_B/10.0/VOL_B-allmoms-10.0-2017-12-24-HTY-h5netcdf.nc")

for coord in ["latitude", "longitude", "altitude", "elevation"]:
    if "time" in dsrpckd[coord].dims:
        dsrpckd.coords[coord] = dsrpckd.coords[coord].median("time")


# test plots
# dsrunpckd.rtime[0].plot(label="unpckd")
dsrpckd.rtime[0].plot(label="pckd")

# np.testing.assert_allclose(dsrunpckd.DBZH.values, dsrpckd.DBZH.values)

#%% PLOT simple

vv = "DBZH" # which moment to plor
tt = 0 # which timestep to plot


# dsrunpckd[vv][tt, 250, 0:100].plot(label="unpacked")
dsrpckd[vv][tt, 250, 0:100].plot(label="packed", ls="--")
plt.legend()
plt.suptitle(vv)

#%% PLOT interactively with plotly

#### this example does not work for polar coords
def plot_2d_data(ds, var, dim1, dim2):
    data = ds[var]  # Extract the data from the xarray Dataset
    dims = (ds[dim1], ds[dim2])  # Extract the dimensions to plot along
    fig = px.imshow(data, animation_frame='time', x=dims[1], y=dims[0], color_continuous_scale='Viridis')
    fig.update_layout(title=f"2D Plot of {var} along {dim1} and {dim2}")
    return fig

fig = plot_2d_data(dsrpckd.loc[{"time": "2015-09-30 00"}].pipe(wrl.georef.georeference_dataset), vv, 'azimuth', 'range')
# fig.show()
fig.write_html("/automount/ags/jgiles/turkey_test/images/HTY_2015-09-30.html")



#### this example should work for polar coords. use Barpolar instead of imshow (but very slow to load the html file later)
# for some reason is not working, the data is not showing
# https://stackoverflow.com/questions/64918776/is-there-a-plotly-equivalent-to-matplotlib-pcolormesh
# https://github.com/plotly/plotly.py/issues/2024

ds_to_plot = dsrpckd[vv].loc[{"time": "2015-09-30 00"}][0]

azgrid, rgrid = np.meshgrid(ds_to_plot["azimuth"].values, ds_to_plot["range"].values)
azgrid = azgrid.ravel()
rgrid = rgrid.ravel()

hovertemplate = ('my range: %{r}<br>'
                 'my azimuth: %{theta}<br>'
                 'my value: %{customdata[0]:.2f}<br>'
                 '<extra></extra>')

fig = go.Figure(
    go.Barpolar(
        r=rgrid,
        theta=azgrid,
        customdata=np.vstack((ds_to_plot.values)),
        hovertemplate=hovertemplate,
        marker=dict(
            colorscale=px.colors.diverging.BrBG,
            showscale=True,
            color=ds_to_plot.values,
        )
    )
)

fig.update_layout(
    title='My Plot',
    polar=dict(
        angularaxis=dict(tickvals=np.arange(0, 360, 10),
                         direction='clockwise'),
        radialaxis_tickvals=[],
    )
)
# fig.show()
fig.write_html("/automount/ags/jgiles/turkey_test/images/HTY_2015-09-30.html")


#%% PLOT interactively with hvplot

#### PPIs
# this has problem with getting the ticks right in the colorbar
for vv in ["DBZH", "ZDR", "KDP", "PHIDP", "RHOHV"]:

    ds_to_plot = dsrpckd[vv].loc[{"time": "2015-09-30"}].pipe(wrl.georef.georeference_dataset)
    
    #### with matplotlib extension
    hvplot.extension('matplotlib')
    # set the widget on the bottom https://github.com/holoviz/hvplot/issues/519
    hv.output(widget_location='bottom') 
    
    visdict14 = radarmet.visdict14
    norm = radarmet.get_discrete_norm(visdict14[vv]["ticks"])
    ticks = visdict14[vv]["ticks"]
    cmap = visdict14[vv]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
    
    # re do the colormap because the function does not get the extremes correctly
    cmap2 = mpl.colors.ListedColormap([cc for cc in cmap.colors[1:-1]])
    cmap2.set_over(cmap.colors[-1])
    cmap2.set_under(cmap.colors[0])
    
    fig = ds_to_plot.hvplot(x="x", y="y",groupby="time", kind="quadmesh", size=(10,10), xlim=(-30000,30000), ylim=(-30000,30000), colorbar=True).opts(colorbar_opts={
                                                                                                # 'background_fill_alpha':0.1,
                                                                                                # 'bar_line_width':2,
                                                                                                # 'label_standoff':8,
                                                                                                # 'major_label_text_font_size':2,
                                                                                                # 'major_label_overrides':clabs,
                                                                                                'ticks': ticks,
                                                                                                },
                                                                                                clim =(ticks[0], ticks[-1]),
                                                                                                cmap=cmap2,
                                                                                                cbar_extend= "both",
                                                                                                # norm=norm ,
                                                                                                title=vv
    
                                                                                                )
    
    hvplot.save(fig, "/automount/ags/jgiles/turkey_test/images/ANK_"+vv+"_2015-09-30.html")
    
#### Multi-variables PPIs at the same time
# there is an issue that I cannot pass the norm argument to the plotting function to correctly plot the colorbars
# I raised an issue here: https://github.com/holoviz/hvplot/issues/1061

hvplot.extension('matplotlib')
# set the widget on the bottom https://github.com/holoviz/hvplot/issues/519
hv.output(widget_location='bottom') 

ds_to_plot = dsrpckd.loc[{"time": "2015-09-30 00"}].pipe(wrl.georef.georeference_dataset)

figs = []
visdict14 = radarmet.visdict14
visdict14["PHIDP"] = visdict14["PHI"]

for vv in ["DBZH", "ZDR", "KDP", "RHOHV", "PHIDP"]:
    norm = radarmet.get_discrete_norm(visdict14[vv]["ticks"])
    ticks = visdict14[vv]["ticks"]
    cmap = visdict14[vv]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
    
    # re do the colormap because the function does not get the extremes correctly
    cmap2 = mpl.colors.ListedColormap([cc for cc in cmap.colors[1:-1]])
    cmap2.set_over(cmap.colors[-1])
    cmap2.set_under(cmap.colors[0])
    
    figs.append(  ds_to_plot[vv].hvplot(x="x", y="y",groupby="time", kind="quadmesh",
                                    width=800,height=800 , colorbar=True).opts(colorbar_opts={
                                                                # 'ticks': ticks,
                                                                # "norm":norm
                                                                },
                                                                clim =(ticks[0], ticks[-1]),
                                                                cmap=cmap2,
                                                                cbar_extend= "both",
                                                                # norm=norm,
                                                                title=vv,
                                                                zticks=ticks,
                                                                )
        )
    


fig = figs[0]
for fign in range(len(figs)):
    if fign==0: continue
    fig = fig+figs[fign]
    
fig.cols(3)

hvplot.save(fig, "/automount/ags/jgiles/turkey_test/images/HTY_2015-09-30.html")



#### with bokeh extension
# this is quite close but not there yet, the colorbar is still not normalized according to the ticks
# It could be improved with something like this: https://discourse.holoviz.org/t/custom-discrete-colormaps/2183/2

hvplot.extension('bokeh')
ds_to_plot = dsrpckd[vv].loc[{"time": "2017-05-08 00"}].pipe(wrl.georef.georeference_dataset)
# set the widget on the bottom https://github.com/holoviz/hvplot/issues/519
hv.output(widget_location='bottom') 

visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[vv]["ticks"])
ticks = visdict14[vv]["ticks"]
cmap = visdict14[vv]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")
ticker = FixedTicker(ticks=visdict14[vv]["ticks"])


# re do the colormap because the function does not get the extremes correctly (this does nothing for the pointy ends here)
cmap2 = mpl.colors.ListedColormap([cc for cc in cmap.colors[1:-1]])
cmap2.set_over(cmap.colors[-1])
cmap2.set_under(cmap.colors[0])

fig = ds_to_plot.hvplot(x="x", y="y",groupby="time", kind="quadmesh", colorbar=True,
                        width=600,height=400  ).opts(colorbar_opts={
                                                                                            'ticker': ticker,
                                                                                            },
                                                                                            clim =(ticks[0], ticks[-1]),
                                                                                            cmap=cmap2,
                                                                                            # cbar_extend= "both",
                                                                                            # norm=norm
                                                                                            # title=vv

                                                                                            )

hvplot.save(fig, "/home/jgiles/turkey_test/images/fifth_test.html")




#### Multi variable lines along an azimuth (works but extremely slow)
hvplot.extension('bokeh')
# set the widget on the bottom https://github.com/holoviz/hvplot/issues/519
hv.output(widget_location='bottom') 

ds_to_plot = dsrpckd.loc[{"time": "2017-05-08"}]

for vv in ["DBZH", "ZDR", "KDP", "RHOHV", "PHIDP"]:
    
    fig = ds_to_plot[vv].hvplot(x="range",groupby=["time", "azimuth"], kind="line")
    hvplot.save(fig, "/home/jgiles/turkey_test/images/"+vv+"_test.html")



#%% Load QVPs
mpl.rcParams.update(mpl.rcParamsDefault)

# qvps = xr.open_dataset("/home/jgiles/dmi/qvps/2015/2015-09/2015-09-30/ANK/MON_YAZ_K/12.0/qvp_MON_YAZ_K-allmoms-12.0-20152015-092015-09-30-ANK-h5netcdf.nc")
# qvps = xr.open_dataset("/home/jgiles/dmi/qvps/2017/2017-12/2017-12-24/HTY/VOL_B/10.0/qvp_VOL_B-allmoms-10.0-2017-12-24-HTY-h5netcdf.nc")
# qvps = xr.open_dataset("/home/jgiles/dwd/qvps/2017/2017-07/2017-07-25/pro/vol5minng01/07/qvp_ras07-vol5minng01_sweeph5onem_allmoms_07-2017072500033500-pro-10392.nc")
qvps = xr.open_dataset("/home/jgiles/dwd/qvps/2017/2017-11/2017-11-05/tur/vol5minng01/07/qvp_ras07-vol5minng01_sweeph5onem_allmoms_07-2017110500033500-tur-10832.nc")
qvps = xr.open_dataset("/home/jgiles/dwd/qvps/2017/2017-11/2017-11-05/tur/vol5minng01/07/qvp_ras07-vol5minng01_sweeph5onem_allmoms_07-2017110500033500-tur-10832.nc")

#%% Plot QVP
tempfix=273.15 # to shift ERA5 data from K to C

vv="ZDR"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[vv]["ticks"])
ticks = visdict14[vv]["ticks"]
cmap = visdict14[vv]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

(qvps["ZDR"]).plot(x="time", cmap=cmap, norm=norm, extend="both", ylim=(200, 7000))
(qvps["TEMP"]-tempfix).plot.contour(x="time", levels=[3], colors="white") # 3 C line
(qvps["TEMP"]-tempfix).plot.contour(x="time", levels=[3], colors="grey", linestyles="--")
plt.title("corrected ZDR")

#%% Plot scatter
timesel="2017-07-25"

# From QVP
qvps_loc = qvps.loc[{"time": slice(timesel+" 05", timesel+" 10")}]

# we only consider points below 3 C
qvps_loc.where(qvps_loc["TEMP"]>tempfix+3).where(qvps_loc["RHOHV"]>0.99).plot.scatter(x="DBZH", y="ZDR", hue="TEMP", cmap="viridis")


# From volume
# first we need to add TEMP data
dsrpckd = dsrpckd.pipe(wrl.georef.georeference_dataset)
dsrpckd.coords["TEMP"] = qvps["TEMP"].swap_dims({"z":"range"})

vv="ZDR"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[vv]["ticks"])
ticks = visdict14[vv]["ticks"]
cmap = visdict14[vv]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

# whole PPI
isel=72

dsrpckd[vv][isel].plot(x="x", y="y", xlim=(-30000,30000), ylim=(-30000,30000), cmap=cmap, norm=norm, extend="both")

# PPI below 3 C and RHOHV>0.99
dsrpckd_fil = dsrpckd[vv][isel].where(dsrpckd["TEMP"][isel]-tempfix>3).where(dsrpckd["RHOHV"][isel]>0.99)
dsrpckd_fil.plot(x="x", y="y", xlim=(-30000,30000), ylim=(-30000,30000), cmap=cmap, norm=norm, extend="both")

# scatter
dsrpckd_loc = dsrpckd.loc[{"time": slice(timesel+" 05", timesel+" 10")}]
dsrpckd_loc_fil = dsrpckd_loc.where(dsrpckd_loc["TEMP"]>tempfix+3).where(dsrpckd_loc["RHOHV"]>0.99)
dsrpckd_loc_fil.plot.scatter(x="DBZH", y="ZDR", hue="TEMP", cmap="viridis", add_legend=False, alpha=0.1)

# another option with basic matplotlib
plt.scatter(dsrpckd_loc_fil["DBZH"], 
            dsrpckd_loc_fil["ZDR"], 
             alpha=0.1)


# scatter with only significant values
dsrpckd_loc_fil.plot.scatter(x="DBZH", y="ZDR", hue="TEMP", cmap="viridis", add_legend=False,
                                                           xlim=(0,40), ylim=(-1,2), alpha=0.1)


#%% Calibrate ZDR
qvps2=qvps.loc[{"time":slice("2017-11-05 09", "2017-11-05 11")}]
zdroffset = utils.zhzdr_lr_consistency(qvps["DBZH"][:,1:].values, qvps["ZDR"][:,1:].values, qvps["RHOHV"][:,1:].values, 
                                       (qvps["TEMP"][:,1:]-273.15).values,  tmp_th=-10, rhohv_th=0.7)
