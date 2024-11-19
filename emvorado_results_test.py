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

import xarray as xr

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
    from Scripts.python.radar_processing_scripts.classify_precip_typ import classify_precip_typ
except ModuleNotFoundError:
    import utils
    import radarmet
    from classify_precip_typ import classify_precip_typ

from osgeo import osr


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


def plot_timestep_map(ds, ts, varlist=["zrsim"], md=None):
    swp = ds.isel(time=ts)
    
    # create figure (10 inch by 8 inch)

    fig = plt.figure( figsize=(10,8),dpi=300)

    if md == None: 
        md = [radarmet.visdict14["ZH"].copy(),
              radarmet.visdict14["ZH"].copy(),
              radarmet.visdict14["ZDR"].copy(),
              radarmet.visdict14["RHOHV"].copy()]
        #better colormaps
        md[0]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
        # md[1]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
        # md[2]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
        # md[3]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
    
    axnr = 110
    if len(varlist)==2:
        axnr = 120
    if len(varlist)==4:
        axnr = 220
        
    datalist = [swp[vv] for vv in varlist] # [swp.zrsim, swp.zrsim, swp.zdrsim, swp.rhvsim]
    for nr,data in enumerate(datalist):
        #ax = fig.add_subplot(axnr+nr+1, sharex=True, sharey=True)

        norm = radarmet.get_discrete_norm(md[nr]["ticks"])
        cmap = radarmet.get_discrete_cmap(md[nr]['ticks'], md[nr]['cmap'])

        cbar_kwargs = dict(extend="both",
                          extendrect=False,
                          pad=0.05,
                          fraction=0.1,
                          label = data.units
                         )


        data.wradlib.plot(proj=ccrs.Mercator(central_longitude = swp.longitude.values),
                          fig=fig, ax=axnr+nr+1, cbar_kwargs=cbar_kwargs,
                          add_colorbar=True, norm=norm, cmap=cmap, extend='both')
        
        ax = plt.gca()
        
        dl = False
        if nr%int(str(axnr)[0]) == 0 and nr/int(str(axnr)[1]) == int(str(axnr)[0])-1: dl = ['left', 'bottom']
        elif nr%int(str(axnr)[0]) == 0: dl = ['left']
        elif nr > ( int(str(axnr)[0])*int(str(axnr)[1]) - int(str(axnr)[1]) ): dl = ['bottom'] 
        ax.gridlines(draw_labels=dl)
        ax.add_feature(states_provinces, edgecolor='black', ls=":")
        ax.add_feature(countries, edgecolor='black', )

        ax.set_title(data.name+' - '+data.attrs["long_name"], y=1)

        # plt.text(1.0, 1.05, 'azimuth', transform=ax.parasites[0].transAxes, va='bottom', ha='right')

        # not working as intended for some reason
        # plt.gca().set_xlabel("X-Range [m]")
        # plt.gca().set_ylabel("Y-Range [m]")

    fig.suptitle("sweep at {} deg elevation at {}".format(np.round(swp.elevation[0].values, decimals=1), 
                                                              swp.time.values.astype('<M8[s]')), fontsize=18)
    
    
    

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
path_dwd = "/automount/realpep/upload/jgiles/dwd/2017/2017-07/2017-07-25/pro/vol5minng01/*/*-hd5"
# path_emvorado_obs = "/home/jgiles/emvorado-offline-results/output/20171028_*/radarout/cdfin_allobs_id-010392_*_volscan"
path_emvorado_sim = "/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/iconemvorado_20170725_8radars_worked_17-07-2024/radout/cdfin_allsim_id-010392_20170725*_20170725*_volscan"
# path_emvorado_sim = "/automount/ags/jgiles/emvorado-offline-results/output/20170712_*/radarout/cdfin_allsim_id-010392_*_volscan"

# Load files
vol_dwd = utils.load_volume(sorted(glob.glob(path_dwd)), func=utils.load_dwd_preprocessed, verbose=True)
# vol_emvorado_obs = utils.load_emvorado_to_radar_volume(path_emvorado_obs, rename=True)
vol_emvorado_sim = utils.load_emvorado_to_radar_volume(path_emvorado_sim, rename=True)


#%% Plot PPI
tsel = "2017-07-25T02:00"
elevn = 0 # elevation index

datatype = "sim" # sim, obs or ori (original)
mom = "DBZH"
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

if isvolume: elevtitle = " "+str(np.round(ds.isel({"sweep_fixed_angle":elevn}).sweep_fixed_angle.values, 2))+"°"
else: elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(datasel.time.values).split(".")[0])

#%% PLOT raw data (OLD)
# extract one elevation and georreference
swp_dwd = vol_dwd.isel(sweep_fixed_angle=0).pipe(wrl.georef.georeference)
# swp_emvorado_obs = vol_emvorado_obs[1].pipe(wrl.georef.georeference)
swp_emvorado_sim = vol_emvorado_sim.isel(records=0).pipe(wrl.georef.georeference)

datatype = "ori" # sim, obs or ori (original)

for timestep in np.arange(24,36):
    
    if datatype == "sim":
        swp = swp_emvorado_sim.isel(time=timestep)
        mom = swp['DBZH'].copy() # moment to plot
        
    elif datatype == "obs":
        swp = swp_emvorado_obs.isel(time=timestep)
        mom = swp['DBZH'].copy() # moment to plot
        
    elif datatype == "ori":
        swp = swp_dwd.isel(time=timestep)
        mom = swp['DBZH'].copy() # moment to plot
        
    else:
        raise Exception("select correct data source")
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.05,0.05,0.95,0.85])
    ticks_zh = radarmet.visdict14["DBZH"]["ticks"].copy()
    
    norm = radarmet.get_discrete_norm(ticks_zh)
    cmap = radarmet.get_discrete_cmap(ticks_zh, 'HomeyerRainbow')
    cbar_extend = 'both'
    
    xlabel = 'X-Range [m]'
    ylabel = 'Y-Range [m]'
    
    if mom.sweep_mode == "rhi":
        xstr = "gr"
        ystr = "z"
    else:
        xstr = "x"
        ystr = "y"
    
    # create plot
    im = mom.plot(x=xstr, y=ystr, ax=ax,
                          norm=norm,
                          cmap=cmap,
                          add_colorbar=True,
                          add_labels=True,
                          extend=cbar_extend,
                          )
    im.colorbar.ax.tick_params(labelsize=16)
    plt.setp(im.colorbar.ax.yaxis.label, fontsize=16)
    # set_tick_params(labelsize=16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_title("{}".format(mom.attrs['long_name']), fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(str(mom.time.values))
    if mom.sweep_mode != "rhi":
        ax.set_aspect(1)
        
    # fig.savefig("/automount/user/jgiles/emvorado_test_case/"+datatype+"_dbzh_"+str(mom.time.values).split(":")[0]+".png")


#%% QVP
# Azimuthally averaged profiles of a conical volume measured at elevations between 10 and 20 degrees
# use 12 deg elevation

max_height = 12000 # max height for the qvp plots
tsel = ""
elevn = 7 # elevation index
plot_ML = False

datatype = "ori" # sim, obs or ori (original)
mom = "DBZH"
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
    datasel = datasel.assign_coords(z=datasel.z.mean('azimuth'))

qvps = utils.compute_qvp(datasel)

    
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

elevtitle = " "+str(np.round(ds["sweep_fixed_angle"].values[0], 2))+"°"
plt.title(mom+elevtitle+". "+str(qvps.time.values[0]).split(".")[0])
plt.show()
plt.close()

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
