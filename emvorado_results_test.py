#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:22:35 2022

@author: jgiles
"""

import os
os.chdir('/home/jgiles/radarmeteorology/notebooks/')
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

import xarray as xr

import wradlib as wrl
sys.path.insert(0, '../')
from radarmet import *
from read_cband_dwd import *

### Better colormaps
from colormap_generator import *

### classifier for stratiform or convective precip
from classify_precip_typ import *

def emvorado_to_radar_volume(path):
    """
    Generate a wradlib.io.RadarVolume from EMVORADO output files. 
    WARNING: The resulting volume has its elevations ordered from lower to higher and
            not according to the scan strategy
    
    Parameters
    ----------
    path : paths (str or nested sequence of paths) 
        – Either a string glob in the form "path/to/my/files/*.nc" or an 
        explicit list of files to open. Paths can be given as strings or 
        as pathlib Paths. Feeds into xarray.open_mfdataset
        
    Returns
    -------
    data_vol : :class:`wradlib.io.RadarVolume`
        Wradlib's RadarVolume

    """
    data_emvorado_xr = xr.open_mfdataset(path, concat_dim="time", combine="nested")
    data = data_emvorado_xr.rename_dims({"n_range": "range", "n_azimuth": "azimuth"})

    # we make the coordinate arrays
    range_coord = np.array([ np.arange(rs, rr*rb+rs, rr) for rr, rs, rb in 
                           zip(data.range_resolution[0], data.range_start[0], data.n_range_bins[0]) ])
    azimuth_coord = np.array([ np.arange(azs, azr*azb+azs, azr) for azr, azs, azb in 
                           zip(data.azimuthal_resolution[0], data.azimuth_start[0], data.azimuth.shape[0]*np.ones_like(data.records)) ])
    
    # create containers 
    data_vol = wrl.io.RadarVolume()
    data_records = dict()
    
    # process each elevation angle
    for rcds in data.records:
        # select elevation record
        data_records[int(rcds)] = data.sel(records = int(rcds) )

        # create time coordinate
        time_coord = xr.DataArray( [
                    datetime.datetime(int(yy), int(mm), int(dd),
                                      int(hh), int(mn), int(ss))
                    for yy,mm,dd,hh,mn,ss in
                    
                                    zip( data_records[int(rcds)].year,
                                    data_records[int(rcds)].month,
                                    data_records[int(rcds)].day,
                                    data_records[int(rcds)].hour,
                                    data_records[int(rcds)].minute,
                                    data_records[int(rcds)].second 
                                    )
                    ], dims=["time"] )

        # add coordinates for range, azimuth, time, latitude, longitude, altitude, elevation, sweep_mode
    
        data_records[int(rcds)].coords["range"] = ( ( "range"), range_coord[rcds])
        data_records[int(rcds)].coords["azimuth"] = ( ( "azimuth"), azimuth_coord[rcds])
        data_records[int(rcds)].coords["time"] = time_coord
        data_records[int(rcds)].coords["latitude"] = float( data_records[int(rcds)]["station_latitude"][0] )
        data_records[int(rcds)].coords["longitude"] = float( data_records[int(rcds)]["station_longitude"][0] )
        data_records[int(rcds)].coords["altitude"] = data_records[int(rcds)].attrs["alt_msl_true"]
        data_records[int(rcds)].coords["elevation"] = data_records[int(rcds)]["ray_elevation"]
        data_records[int(rcds)].coords["sweep_mode"] = 'azimuth_surveillance'
        
        # move some variables to attributes
        vars_to_attrs = ["station_name", "country_ID", "station_ID_national",
                         "station_longitude", "station_height",
                         "station_latitude", "range_resolution", "azimuthal_resolution",
                         "range_start", "azimuth_start", "extended_nyquist",
                         "high_nyquist", "dualPRF_ratio", "range_gate_length",
                         "n_ranges_averaged", "n_pulses_averaged", "DATE", "TIME",
                         "year", "month", "day", "hour", "minute", "second",
                         "ppi_azimuth", "ppi_elevation", "n_range_bins"
                         ]
        for vta in vars_to_attrs:
            data_records[int(rcds)].attrs[vta] = data_records[int(rcds)][vta]
            
        # add attribute "fixed_angle"
        try:
            # if one timestep
            data_records[int(rcds)].attrs["fixed_angle"] = float(data_records[int(rcds)].attrs["ppi_elevation"])
        except:
            # if multiple timesteps
            data_records[int(rcds)].attrs["fixed_angle"] = float(data_records[int(rcds)].attrs["ppi_elevation"][0])
        
        # drop variables that were moved to attrs
        data_records[int(rcds)] = data_records[int(rcds)].drop_vars(vars_to_attrs)

        # for each remaining variable add "long_name" and "units" attribute
        for vv in data_records[int(rcds)].data_vars.keys():
            try:
                data_records[int(rcds)][vv].attrs["long_name"] = data_records[int(rcds)][vv].attrs["Description"]
            except:
                print("no long_name attribute in "+vv)

            try:
                data_records[int(rcds)][vv].attrs["units"] = data_records[int(rcds)][vv].attrs["Unit"]
            except:
                print("no long_name attribute in "+vv)

        # add the elevation angle to the data volume
        data_vol.append(data_records[int(rcds)])

        
    return data_vol


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
        md = [visdict14["ZH"].copy(),
              visdict14["ZH"].copy(),
              visdict14["ZDR"].copy(),
              visdict14["RHOHV"].copy()]
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

        norm = get_discrete_norm(md[nr]["ticks"])
        cmap = get_discrete_cmap(md[nr]['ticks'], md[nr]['cmap'])

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

# Scan path for RAW Data
path_dwd = "/automount/ags/jgiles/radar_pro_test_20170725/201707*_pro/201707*_*/*vol*"
path_emvorado_obs = "/home/jgiles/emvorado-offline-results/output/20171028_*/radarout/cdfin_allobs_id-010392_*_volscan"
path_emvorado_sim = "/home/jgiles/emvorado-offline-results/output/20171028_*/radarout/cdfin_allsim_id-010392_*_volscan"

# open original measurement vol (old method)
# vol_dwd = wrl.io.open_odim(path_dwd, loader="h5py", chunks={}) # this method is deprecated, should be replaced with wrl.io.open_odim_mfdataset(file_list, chunks={})
# swp_dwd = vol_dwd[5].data.pipe(wrl.georef.georeference_dataset)
    
def open_dwd_radar_vol(path):
    # open vol (new method)
    # path can be a path with wildcards or a list of file paths created with create_dwd_filelist()
    
    if type(path) is not list:
        # If a variable is not present in all datasets this method will fail (e.g. uzdr is not present in all dwd data)
        flist = sorted(glob.glob(path))
        flist1 = np.array(flist).reshape((-1, 10)) # because there are 10 elevations
        flist2 = flist1.T
    
    else:
        flist = path
        # separate files by elevation
        nelevs = len(set([item.split("/")[-2] for item in flist]))
        flist1 = np.array(flist).reshape((nelevs, -1))
        
        # separate files by variable
        nvars = len(set([item.split("_")[-2] for item in flist1[-1]]))
        ntimes = int(flist1.shape[0]/nvars)
        
        #aux = np.array_split(flist1, nvars, axis=-1)
        
        flist2 = np.concatenate([flist1[nt::ntimes, :] for nt in range(ntimes)], axis=-1)
    
    
    vol_dwd = wrl.io.RadarVolume()
    data = list()
    
    
    for fl in flist2:

        if len(np.unique(np.array([item.split("_")[-1] for item in fl]))) > 1: # if there is more than 1 timestep

            data.append({})
            
            for vv in set([fln.split("_")[-2] for fln in fl]):
                data[-1][vv] = wrl.io.open_odim_mfdataset([fln for fln in fl if vv in fln]) #, concat_dim="time", combine="nested")

            vol_dwd.append(xr.merge(data[-1].values(), combine_attrs="override"))
            vol_dwd.sort(key=lambda x: x.time.min().values)
    
        else: # for a single timestep
            ds = wrl.io.open_odim_mfdataset(fl, concat_dim=None, combine="by_coords")
            # ds = wrl.io.open_odim_mfdataset(fl, concat_dim="time", combine="nested")
            vol_dwd.append(ds)      

    return vol_dwd

        
# open dwd data
# option 1: open files with wildcards in path
# vol_dwd = open_dwd_radar_vol(path_dwd)

# option 2: create an ordered list of files to load
# Scan path for RAW Data
realpep_path = '/automount/realpep/upload/RealPEP-SPP/DWD-CBand/'
moments = ["TH", "DBZH", "ZDR", "RHOHV", "PHIDP", "KDP", "DBZV", "DBTV", "VRADH", "VRADV", "WRADH", "WRADV"]
moments = ["DBZH", "ZDR", "RHOHV", "UPHIDP", "KDP", "TH"]
mode = "vol5minng01" #"vol5minng01" 'pcpng01' (DWD’s horizon-following precipitation scan) 
loc = "pro"
# Info on the DWD scanning strategy https://www.dwd.de/EN/ourservices/radar_products/radar_products.html
# Scans 00-05 are the volume scans (5.5°, 4.5°, 3.5°, 2.5°, 1.5° and 0.5°), the rest are 8.0°, 12.0°, 17.0° and 25.0°
SCAN = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
scan = "*" # SCAN[4]

# Start and End time
start_time = datetime.datetime(2017,7,25,0,0)
end_time = start_time + dt.timedelta(hours=24, minutes=0)

# Radardata filelist
file_list = create_dwd_filelist(path=realpep_path, 
                    starttime=start_time, 
                    endtime=end_time, 
                    moments=moments, 
                    mode=mode,
                    loc=loc,
                    scan=scan)

file_list = list(file_list)

vol_dwd = open_dwd_radar_vol(file_list)


# ##### TEST: saving with new DataTree structure (wradlib 1.19!!!)
# import datatree as dttree
# dtree = dttree.DataTree(name="root")
# for i, sw in enumerate(vol_dwd2):
#     dim0 = list(set(sw.dims) & {"azimuth", "elevation"})[0]
#     if "fixed_angle" in sw:
#         sw = sw.rename({"fixed_angle": "sweep_fixed_angle"})
#     dttree.DataTree(sw, name=f"sweep_{i}", parent=dtree)

# dtree.to_netcdf("test.nc")

# vol_dwd3 = dttree.open_datatree("test.nc")

# vol_dwd3["sweep_0"].ds # get as dataset
# swp = vol_dwd3["sweep_0"].to_dataset() # get a sweep

# ##### END TEST


# open emvorado files
vol_emvorado_obs = emvorado_to_radar_volume(path_emvorado_obs)
vol_emvorado_sim = emvorado_to_radar_volume(path_emvorado_sim)

# extract one elevation and georreference
swp_dwd = vol_dwd[4].pipe(wrl.georef.georeference_dataset)
swp_emvorado_obs = vol_emvorado_obs[1].pipe(wrl.georef.georeference_dataset)
swp_emvorado_sim = vol_emvorado_sim[1].pipe(wrl.georef.georeference_dataset)





#%% PLOT raw data
datatype = "sim" # sim, obs or ori (original)

for timestep in np.arange(8):
    
    if datatype == "sim":
        swp = swp_emvorado_sim.isel(time=timestep)
        mom = swp['zrsim'].copy() # moment to plot
        
    elif datatype == "obs":
        swp = swp_emvorado_obs.isel(time=timestep)
        mom = swp['zrobs'].copy() # moment to plot
        
    elif datatype == "ori":
        swp = swp_dwd.isel(time=timestep)
        mom = swp['DBZH'].copy() # moment to plot
        
    else:
        raise Exception("select correct data source")
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.05,0.05,0.95,0.85])
    ticks_zh = visdict14["DBZH"]["ticks"].copy()
    
    norm = get_discrete_norm(ticks_zh)
    cmap = get_discrete_cmap(ticks_zh, 'HomeyerRainbow')
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
        
    fig.savefig("/automount/user/jgiles/emvorado_test_case/"+datatype+"_dbzh_"+str(mom.time.values).split(":")[0]+".png")


#%% QVP
# Azimuthally averaged profiles of a conical volume measured at elevations between 10 and 20 degrees
# use 12 deg elevation

# create height coord
swp = vol_emvorado_sim[-3].sel({"time":"2017-10-20"}).pipe(wrl.georef.georeference_dataset)
swp = swp.assign_coords(height=swp.z.mean('azimuth'))

# Apply offsets if necessary
# add corrections here

# Calculate median
swp_median = swp.median('azimuth', skipna=True)
try:
    swp_median = swp_median.where(swp_median.DBZH > 0)
except:
    try:
        swp_median = swp_median.where(swp_median.zrsim > 0)
    except:
        swp_median = swp_median.where(swp_median.zrobs > 0)



#### Plot contour and contourf
contourf_var = 'DBZH'
contour_var = 'ZDR'


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

contourf_levels = visdict14[contourf_var]['ticks'].copy() # np.arange(0, 60, 5)
contour_levels = visdict14[contour_var]['contours'].copy() # np.arange(0, 40, 5)
cmap = visdict14[contourf_var]['cmap'].copy() # 'turbo'

cbar_kwargs = dict(extend="both",
                   extendrect=False,
                   extendfrac="auto",
                   pad=0.05,
                   fraction=0.1,
                  )

try:
    contourf =  swp_median[contourf_var]
except:
    try: 
        contourf =  swp_median[var_names_map_sim[contourf_var]]
    except:
        contourf =  swp_median[var_names_map_obs[contourf_var]]

    
contourf.plot.contourf(x="time", y="height", ax=ax, 
                              cmap=cmap, 
                              levels=contourf_levels,
                              add_colorbar=True,
                              cbar_kwargs=cbar_kwargs)

try:
    contour =  swp_median[contour_var]
except:
    try: 
        contour =  swp_median[var_names_map_sim[contour_var]]
    except:
        contour =  swp_median[var_names_map_obs[contour_var]]

contour.plot.contour(x="time", y="height", ax=ax, 
                             colors="k", 
                             levels=contour_levels,
                             add_colorbar=False)
ax.set_ylim(0, 8000)

# change title
ax.set_title(f"{contourf.time[0].values.astype('<M8[s]')} - {contourf.name}  - {swp.fixed_angle}")

#%% Classify stratiform/convective precip

# for observed data
momobs = vol_emvorado_obs[0]["zrobs"].sel({"time":"2017-10-20"}).pipe(wrl.georef.georeference_dataset)
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
momsim = vol_emvorado_sim[0]["zrsim"].sel({"time":"2017-10-20"}).pipe(wrl.georef.georeference_dataset)
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
ds = ds.pipe(georeference_dataset, proj=map_proj) 

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

comp_tsmp = xr.open_mfdataset("/home/jgiles/emvorado-offline-results/output_no-obs-all-network/*/radarout/dbzcmp_sim_*.grb2", engine = "cfgrib", concat_dim="time", combine="nested")

seltime = "2017-10-02 21:00"
nelev = 0

comp_plot = comp_tsmp.DBZCMP_SIM.sel({"time": seltime})[nelev]

# plot single timestep
cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
f, ax1 = plt.subplots(1, 1, figsize=(5.5, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))
plot = comp_plot.where(comp_plot>-999).plot(x="longitude", y="latitude", levels=visdict14["DBZH"]["ticks"], cmap=cmap, extend="both")

plt.gca().set_extent([4.5, 16, 46, 56])
plot.axes.coastlines()
plot.axes.gridlines(draw_labels={"bottom": "x", "left": "y"})
plot.axes.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, alpha=0.7) #countries
plt.title(seltime)



# make a movie
dataplot = comp_tsmp.DBZCMP_SIM.sel({"time":slice("2017-10-02", "2017-10-02 21:00")})
cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
def plot_data(i):
    dataplot2 = dataplot[i, nelev]
    plt.subplot(111, projection=ccrs.PlateCarree())
    plot = dataplot2.where(dataplot2>-999).plot(x="longitude", y="latitude", levels=visdict14["DBZH"]["ticks"], cmap=cmap, extend="both")
    
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
