#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:11:31 2022

@author: jgiles
"""
import os
os.chdir('/home/jgiles/radarmeteorology/notebooks/')
os.environ["WRADLIB_DATA"] = '/home/jgiles/wradlib-data-main'

import pyart
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

import xarray as xr

import wradlib as wrl
sys.path.insert(0, '../')
from radarmet import *
from read_cband_dwd import *

### Better colormaps
from colormap_generator import *

#%% Load data from radar sites %%time
# Load data from Ummendorf, Prötzel und Türkheim (umd, pro, tur)

DATE = ["20170727","20180923","20170725","20170725",
                "20170719","20170724","20170726",
                "20190623","20190720","20180728",
                "20180809","20190829","20190830",
                "20181202","20170720"]

LOCATIONS = ['boo', 'eis', 'fld', 'mem', 
             'neu', 'ros', 'tur', 'umd', 'drs',
             'ess', 'fbg', 'hnr', 'isn', 'nhb',
             'oft', 'pro']

# Info on the DWD scanning strategy https://www.dwd.de/EN/ourservices/radar_products/radar_products.html
# Scans 00-05 are the volume scans (5.5°, 4.5°, 3.5°, 2.5°, 1.5° and 0.5°), the rest are 8.0°, 12.0°, 17.0° and 25.0°
SCAN = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']


fmt = "%Y%m%d"

case_studies = {cs: dt.datetime.strptime(cs, fmt) for cs in DATE}

mode = "vol5minng01" #"vol5minng01" 'pcpng01' (DWD’s horizon-following precipitation scan) 


######### RUN
moments = ["TH", "DBZH", "ZDR", "RHOHV", "PHIDP", "KDP", "DBZV", "DBTV", "VRADH", "VRADV", "WRADH", "WRADV"]
moments = ["DBZH", "ZDR", "RHOHV", "UPHIDP", "KDP", "TH"]

loc = LOCATIONS[-1]
date = DATE[3]
scan = SCAN[7]

# Scan path for RAW Data
realpep_path = '/automount/realpep/upload/RealPEP-SPP/DWD-CBand/'

# Start and End time
start_time = case_studies[date] + dt.timedelta(hours=0, minutes=0)
end_time = case_studies[date] + dt.timedelta(hours=24, minutes=0)

# Radardata filelist
file_list = create_dwd_filelist(path=realpep_path, 
                    starttime=start_time, 
                    endtime=end_time, 
                    moments=moments, 
                    mode=mode,
                    loc=loc,
                    scan=scan)

file_list = list(file_list)
# open vol
vol = wrl.io.open_odim(file_list, loader="h5py", chunks={}) # this method is deprecated, should be replaced with wrl.io.open_odim_mfdataset(file_list, chunks={})
swp = vol[0].data.pipe(wrl.georef.georeference_dataset)

#%% Load data from multiple elevations (for reconstructed RHI or CAPPI)


# Read the data, a cfradial file
filename = pyart.testing.get_test_data('swx_20120520_0641.nc')
radar = pyart.io.read(filename)


start_time = case_studies[date] + dt.timedelta(hours=0, minutes=0)
end_time = case_studies[date] + dt.timedelta(hours=2, minutes=0)

file_list=[]
for scan in SCAN:
    # Radardata filelist
    file_list0 = create_dwd_filelist(path=None, 
                        starttime=start_time, 
                        endtime=end_time, 
                        moments=moments, 
                        mode=mode,
                        loc=loc,
                        scan=scan)
    
    file_list = file_list + list(file_list0)
# open vol
vol = wrl.io.open_odim(file_list, loader="h5py", chunks={}) # this method is deprecated, should be replaced with wrl.io.open_odim_mfdataset(file_list, chunks={})


######
vol1 = xr.concat([vol[x].data for x in range(len(vol))], 'Elevation')

swp = vol[0].data.pipe(wrl.georef.georeference_dataset)



#%% PLOT raw data

mom = swp['DBZH'][150].copy() # moment to plot

fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.05,0.05,0.95,0.85])
ticks_zh = visdict14["ZH"]["ticks"].copy()

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
if mom.sweep_mode != "rhi":
    ax.set_aspect(1)


#swp2['DBZH'][0].plot(x='x', y='y')

# ax, im = wrl.vis.plot_ppi(swp['DBZH'][0], fig=fig, proj='cg', cmap='turbo')
# plt.colorbar(im, shrink=0.74)

#%% ATTENUATION CORRECTION
'''
ZH_corr = ZH + alpha*PHIDP
ZDR_corr = ZDR + beta*PHIDP

X band:
alpha = 0.28; beta = 0.05 #dB/deg

C band:
alpha = 0.08; beta = 0.02 #dB/deg

For BoXPol and JuXPol:
alpha = 0.25
'''

alpha = 0.08; beta = 0.02 #dB/deg (for C band)

#### ZPHI method  (This can also be natively calculated with pyart https://arm-doe.github.io/pyart/examples/correct/plot_attenuation.html#sphx-glr-examples-correct-plot-attenuation-py 
swp0 = swp.isel(time=150).compute()

ah_fast, phical, phicorr, cphase, alpha, beta = attenuation_corr_zphi(swp0)
alphax = alpha
betax = beta

# This is only necessary when using an array of alphax (not implemented on attenuation_corr_zphi)
if len(phical.shape) == 3:
    # calculate delta sum
    delta = np.zeros((phical.shape[0], 1))
    diff = np.abs(phical - phicorr)
    #print(delta.shape, diff.shape)
    
    # find first occurrence of non NAN phi per ray
    first = np.nanargmax(~np.isnan(phiclean), axis=1)
    # find last occurrence of non NAN phi per ray
    last = phiclean.shape[1] - np.nanargmax(np.flip(~np.isnan(phiclean), axis=-1), axis=1) - 1
    
    for ray in range(phical.shape[0]):
        f = first[ray]
        l = last[ray]
        if l>f:
            for fd in range(fdphi.shape[0]):
                delta[ray, fd] = np.sum(diff[ray, f:l, fd])
                
    # get delta-sum minimum
    idx = np.argmin(delta, axis=-1)


#%% APPLY ATTENUATION CORRECTION

zhraw = swp0.DBZH #.where((swp0.range > cphase.start_range) & (swp0.range < cphase.stop_range))
zdrraw = swp0.ZDR #.where((swp0.range > cphase.start_range) & (swp0.range < cphase.stop_range))

# calculate corrections for all alphax at once
with xr.set_options(keep_attrs=True):
    #zhcorr1 = zhraw.expand_dims(dim="alpha", axis=-1) + alphax * (phical) # expand_dims only if I am trying an array of alphax
    zhcorr1 = zhraw + alphax * (phical) 
    diff = zhcorr1 - zhraw
    #zdrcorr1 = zdrraw.expand_dims(dim="alpha", axis=-1) + betax * (phical)
    zdrcorr1 = zdrraw + betax * (phical)
    zdrdiff = zdrcorr1 - zdrraw

# use only alphax corresponding to minimum delta-sum for each ray
# zhcorr2 = xr.zeros_like(zhraw)
# zdrcorr2 = xr.zeros_like(zhraw)
# for i, ix in enumerate(idx):
#     zhcorr2[i] = zhcorr1[i,:,ix]
#     zdrcorr2[i] = zdrcorr1[i,:,ix]
zhcorr2 = zhcorr1
zdrcorr2 = zdrcorr1

    
#### PLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, 
                                            figsize=(15, 12), 
                                            sharex=True, sharey=True, 
                                            squeeze=True,
                                            constrained_layout=True)
ticks1 = visdict14["ZH"]["ticks"]
plot_moment(zhraw, ticks1, fig=fig, ax=ax1)
plot_moment(zhcorr2, ticks1, fig=fig, ax=ax2)
ax2.set_title(r'Corrected $Z_{H}$', fontsize=16)

ticks2 = visdict14["ZDR"]["ticks"]
plot_moment(zdrraw, ticks2, fig=fig, ax=ax3)
plot_moment(zdrcorr2, ticks2, fig=fig, ax=ax4)
ax4.set_title(r'Corrected $Z_{DR}$', fontsize=16)

#%% KDP FROM AH VS KDP FROM PHIDP (kdp from ah should be better)
kdp = kdp_from_phidp(phicorr, winlen=11)
kdp.attrs = wrl.io.xarray.moments_mapping["KDP"]
kdp_a = xr.zeros_like(zhcorr1)
kdp_a.attrs = wrl.io.xarray.moments_mapping["KDP"]
kdp_a.data = ah_fast/alphax

#### PLOT
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
                              figsize=(10, 18), 
                              constrained_layout=True)

ticks1 = visdict14["KDP"]["ticks"]
plot_moment(kdp, ticks1, ax=ax1, fig=fig)
ax1.set_title(r'$K_{DP}$ from $\phi_{DP}$', fontsize=16)
plot_moment(kdp_a, ticks1, ax=ax2, fig=fig)
ax2.set_title(r'$K_{DP}$ from $A_{H}$', fontsize=16)
#%% RAIN RATE RETRIEVAL WITH DIFFERENT METHODS
'''
R(KDP) is suitable for hourly point estimators if:
    R > 10 mm/h at S band
    R > 5-6 mm/h at C band
    R > 2-3 mm/h at X band

X band (temp : Horizontal pol / vertical pol):
     0 C : rr_a = 49.1*ah_fast**0.87 / rr_a = 57.8*av**0.89
    10 C : rr_a = 45.5*ah_fast**0.83 / rr_a = 53.3*av**0.85
    20 C : rr_a = 43.5*ah_fast**0.79 / rr_a = 51.1*av**0.81
    30 C : rr_a = 43.0*ah_fast**0.76 / rr_a = 51.0*av**0.78

C band (temp : Horizontal pol / vertical pol):
     0 C : rr_a = 221*ah_fast**0.92 / rr_a = 281*av**0.95
    10 C : rr_a = 250*ah_fast**0.91 / rr_a = 326*av**0.94
    20 C : rr_a = 294*ah_fast**0.89 / rr_a = 393*av**0.93
    30 C : rr_a = 352*ah_fast**0.89 / rr_a = 483*av**0.93

R(A) must be complemented by R(Z) for DeltaPHIDP < 1-4 degrees and by R(KDP) in hail cells

R(ZH, ZDR) = A * ZH**0.95 * ZDR**b  with ZH in mm**6/m**3 and ZDR in dB
It is critical to use ZDR in C band because of strong diff attenuation and resonance effects
For C band:
        A = 2.37E-3 , b = 1.17  Aydin and Giridhar, 1992
        A = 3.61E-3 , b = 1.28  Scarchilli et al., 1993
        A = 3.0E-3 , b = 1.22   Zrnic et al., 2000

For BoXPol and JuXPol:
    rr1 = (dbth_corr/197.)**(1./1.49)
    Z = Z_att + 0.25 PHIDP
    rr_kdp = 16.9*kdp**0.801*np.sign(kdp)
    rr_av = 51.3*av**0.81 # for 20 C and vertical attenuation
    rr_av = 57.8*av**0.89 # for 0 C and vertical attenuation

'''

dbth_corr = wrl.trafo.idecibel(zhcorr2) # wrl.trafo.idecibel(radar_ds.DBTH_OC)

# Marshall-Palmer (Stratiform summer rain): 
rr1 = (dbth_corr/200.)**(1./1.6)

# Convective relation: 
rr2 = (dbth_corr/300.)**(1./1.4)

# R(Kdp)
kdp = swp0.KDP # radar_ds.KDP_CONV
rr_kdp = 16.9*kdp**0.801*np.sign(kdp)

#R(A) needs ZPHI
rr_a = 294*ah_fast**0.89
rr_a = xr.ones_like(rr_kdp) * rr_a

# R(ZH, ZDR) = A * ZH**0.95 * ZDR**b
rr_zh_zdr = 3.0E-3 * dbth_corr**0.95 * zdrcorr2**1.22

# Use R(Z) in light rain and R(KDP) in strong rain
lightrain = rr1.where((rr1 < 20) & (rr1 >= 0), other=0)
strongrain = rr_kdp.where((rr1 >= 20), other=0)
nanrain = xr.where((rr1 >= 0), 1, np.nan)

combined_rain = (lightrain + strongrain) * nanrain

# Plot
cbar_kwargs = dict(extend="both",
                  extendrect=False,
                  extendfrac=None,
                  pad=0.05,
                  fraction=0.2,
                 )
norm = mpl.colors.LogNorm(0.1, 100.)
cbar_kwargs.update(dict(label="Rainrate"))
cbar_extend = cbar_kwargs.get("extend", None)
cmap = "viridis"

# fig = plt.figure(figsize=(15, 12))
# ax = fig.add_subplot(111)
# lightrain.plot(x="x", y="y", ax=ax, norm=norm, cbar_kwargs=cbar_kwargs)
# strongrain.plot(x="x", y="y", ax=ax, norm=norm, add_colorbar=False)


#### PLOT
ticks = np.arange
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, 
                                            figsize=(12, 12), 
                                            sharex=True, sharey=True, 
                                            squeeze=True,
                                            constrained_layout=True)
rr1.plot(x="x", y="y", ax=ax1, norm=norm, cbar_kwargs=cbar_kwargs)
ax1.set_title(r'Marshall-Palmer', fontsize=16)

rr2.plot(x="x", y="y", ax=ax2, norm=norm, cbar_kwargs=cbar_kwargs)
ax2.set_title(r'Convective relation', fontsize=16)

rr_kdp.plot(x="x", y="y", ax=ax3, norm=norm, cbar_kwargs=cbar_kwargs)
ax3.set_title(r'R(KDP)', fontsize=16)

rr_zh_zdr.plot(x="x", y="y", ax=ax4, norm=norm, cbar_kwargs=cbar_kwargs)
ax4.set_title(r'R(ZH,ZDR)', fontsize=16)

rr_a.plot(x="x", y="y", ax=ax5, norm=norm, cbar_kwargs=cbar_kwargs)
ax5.set_title(r'R(A)', fontsize=16)

combined_rain.plot(x="x", y="y", ax=ax6, norm=norm, cbar_kwargs=cbar_kwargs)
ax6.set_title(r'Light rain: Marshall-Parlmer, Strong rain: R(KDP)', fontsize=16)


#%% PLOT COMPOSITION OF TWO RADARS

##### Load second radar close to the first one
loc2 = 'umd'

# Radardata filelist
file_list2 = create_dwd_filelist(path=None, 
                    starttime=start_time, 
                    endtime=end_time, 
                    moments=moments, 
                    mode=mode,
                    loc=loc2,
                    scan=scan)

file_list2 = list(file_list2)
# open vol
vol2 = wrl.io.open_odim(file_list2, loader="h5py", chunks={}) # this method is deprecated, should be replaced with wrl.io.open_odim_mfdataset(file_list, chunks={})
swp2 = vol2[0].data.pipe(wrl.georef.georeference_dataset)

# set scan geometry and radar coordinates
r = swp.bins[0].values
az = swp.azimuth.values
swp_sitecoords = (float(swp.longitude), float(swp.latitude))
swp2_sitecoords = (float(swp2.longitude), float(swp2.latitude))


# Data to plot:
plot_swp = swp['DBZH'][150].to_numpy()
plot_swp2 = swp2['DBZH'][150].to_numpy()

# derive UTM Zone 32 coordinates of range-bin centroids
# create osr projection using epsg number for UTM Zone 32
proj_utm = wrl.georef.epsg_to_osr(32633)

#  for swp
swp_coord = wrl.georef.spherical_to_centroids(r, az, 0, swp_sitecoords, proj=proj_utm)
swp_coord = swp_coord[..., 0:2]
swp_coord = swp_coord.reshape(-1, swp_coord.shape[-1])

# for swp2
swp2_coord = wrl.georef.spherical_to_centroids(r, az, 0, swp2_sitecoords, proj=proj_utm)
swp2_coord = swp2_coord[..., 0:2]
swp2_coord = swp2_coord.reshape(-1, swp2_coord.shape[-1])

# define target grid for composition
def bbox(*args):
    """Get bounding box from a set of radar bin coordinates"""
    x = np.array([])
    y = np.array([])
    for arg in args:
        x = np.append(x, arg[:, 0])
        y = np.append(y, arg[:, 1])
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    return xmin, xmax, ymin, ymax

xmin, xmax, ymin, ymax = bbox(swp_coord, swp2_coord)
x = np.linspace(xmin, xmax + 1000.0, 1000)
y = np.linspace(ymin, ymax + 1000.0, 1000)
grid_coords = wrl.util.gridaspoints(y, x)

# derive quality information - in this case, the pulse volume
pulse_volumes = np.tile(wrl.qual.pulse_volume(r, 1000.0, 1.0), 360)
# interpolate polar radar-data and quality data to the grid
print("Gridding swp data...")
swp_quality_gridded = wrl.comp.togrid(
    swp_coord,
    grid_coords,
    r.max() + 500.0,
    swp_coord.mean(axis=0),
    pulse_volumes,
    wrl.ipol.Nearest,
)
swp_gridded = wrl.comp.togrid(
    swp_coord,
    grid_coords,
    r.max() + 500.0,
    swp_coord.mean(axis=0),
    plot_swp.ravel(),
    wrl.ipol.Nearest,
)

print("Gridding swp2 data...")
swp2_quality_gridded = wrl.comp.togrid(
    swp2_coord,
    grid_coords,
    r.max() + 500.0,
    swp2_coord.mean(axis=0),
    pulse_volumes,
    wrl.ipol.Nearest,
)
swp2_gridded = wrl.comp.togrid(
    swp2_coord,
    grid_coords,
    r.max() + 500.0,
    swp2_coord.mean(axis=0),
    plot_swp2.ravel(),
    wrl.ipol.Nearest,
)

# compose the both radar-data based on the quality information
# calculated above
print("Composing data on a common grid...")
composite = wrl.comp.compose_weighted(
    [swp_gridded, swp2_gridded],
    [1.0 / (swp_quality_gridded + 0.001), 1.0 / (swp2_quality_gridded + 0.001)],
)
composite = np.ma.masked_invalid(composite)

#### Plotting 
plt.figure(figsize=(10, 8))
plt.subplot(111, aspect="equal")
pm = plt.pcolormesh(x, y, composite.reshape((len(x), len(y))), cmap="viridis")
plt.grid()
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.colorbar(pm, shrink=0.85)





#%% PLOT HEIGHTS OF RADAR BINS

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)
cmap = mpl.cm.viridis
swp.z.plot(x="x", y="y", cmap=cmap, ax=ax, cbar_kwargs=dict(label='Height [m]'))
ax.set_xlabel('Range [m]')
ax.set_ylabel('Range [m]')
ax.grid(True)
plt.show()



#%% PLOT RADAR MOMENTS OVER A MAP
map_proj = ccrs.Mercator(central_longitude = swp.longitude.values)

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

def plot_timestep_map(ds, ts):
    swp = ds.isel(time=ts)
    
    # create figure (10 inch by 8 inch)

    fig = plt.figure( figsize=(15,13),dpi=300)

    md = [visdict14["ZH"].copy(),
          visdict14["ZH"].copy(),
          visdict14["ZDR"].copy(),
          visdict14["RHOHV"].copy()]
    #better colormaps
    # md[0]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
    # md[1]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
    # md[2]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')
    # md[3]['cmap'] = mpl.cm.get_cmap('HomeyerRainbow')

    axnr = 220
    for nr,data in enumerate([swp.TH, swp.DBZH, swp.ZDR, swp.RHOHV]):
        #ax = fig.add_subplot(axnr+nr+1, sharex=True, sharey=True)

        norm = get_discrete_norm(md[nr]["ticks"])
        cmap = get_discrete_cmap(md[nr]['ticks'], md[nr]['cmap'])

        cbar_kwargs = dict(extend="both",
                          extendrect=False,
                          pad=0.05,
                          fraction=0.1,
                          label = data.units
                         )


        data.wradlib.plot(proj=map_proj, ax=axnr+nr+1, cbar_kwargs=cbar_kwargs,
                          add_colorbar=True, norm=norm, cmap=cmap, extend='both')
        
        ax = plt.gca()
        
        dl = False
        if nr%int(str(axnr)[0]) == 0 and nr/int(str(axnr)[1]) == int(str(axnr)[0])-1: dl = ['left', 'bottom']
        elif nr%int(str(axnr)[0]) == 0: dl = ['left']
        elif nr > ( int(str(axnr)[0])*int(str(axnr)[1]) - int(str(axnr)[1]) ): dl = ['bottom'] 
        ax.gridlines(draw_labels=dl)
        ax.add_feature(states_provinces, edgecolor='black', ls=":")
        ax.add_feature(countries, edgecolor='black', )

        plt.title(data.name+' - '+data.long_name, y=1.05)

        # plt.text(1.0, 1.05, 'azimuth', transform=ax.parasites[0].transAxes, va='bottom', ha='right')

        # not working as intended for some reason
        # plt.gca().set_xlabel("X-Range [m]")
        # plt.gca().set_ylabel("Y-Range [m]")

    fig.suptitle("sweep at {} deg elevation at {}".format(np.round(swp.elevation[0].values, decimals=1), 
                                                              swp.time.values.astype('<M8[s]')), fontsize=18)
    

plot_timestep_map(swp, 10)

#%% QVP
# Azimuthally averaged profiles of a conical volume measured at elevations between 10 and 20 degrees

# create height coord
swp = swp.assign_coords(height=swp.z.mean('azimuth'))

# Apply offsets if necessary
# add corrections here

# Calculate median
swp_median = swp.median('azimuth')
swp_median = swp_median.where(swp_median.DBZH > 0)


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

contourf =  swp_median[contourf_var]
contourf.plot.contourf(x="time", y="height", ax=ax, 
                              cmap=cmap, 
                              levels=contourf_levels,
                              add_colorbar=True,
                              cbar_kwargs=cbar_kwargs)

contour = swp_median[contour_var]
contour.plot.contour(x="time", y="height", ax=ax, 
                             colors="k", 
                             levels=contour_levels,
                             add_colorbar=False)
ax.set_ylim(0, 8000)

# change title
ax.set_title(f"{contourf.time[0].values.astype('<M8[s]')} - {loc} - {contourf.name}  - {scan}")

#%% VP vertical profile (DWD RADARS DO NOT HAVE VP)

# create height coord
swp = swp.assign_coords(height=swp.z.mean('azimuth'))

# Apply offsets if necessary
# add corrections here

# Calculate median
swp_median = swp.median('azimuth')
swp_median = swp_median.where(swp_median.DBZH > 0)


#### Plot contour and contourf
contourf_var = 'ZDR'
contour_var = 'DBZH'


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

contourf =  swp_median[contourf_var]
contourf.plot.contourf(x="time", y="height", ax=ax, 
                              cmap=cmap, 
                              levels=contourf_levels,
                              add_colorbar=True,
                              cbar_kwargs=cbar_kwargs)

contour = swp_median[contour_var]
contour.plot.contour(x="time", y="height", ax=ax, 
                             colors="k", 
                             levels=contour_levels,
                             add_colorbar=False)
ax.set_ylim(0, 8000)

# change title
ax.set_title(f"{contourf.time[0].values.astype('<M8[s]')} - {loc} - {contourf.name}  - {scan}")



#%% RHI (DWD RADARS DO NOT HAVE RHI)

# recent boxpol event
start_time = dt.datetime(2011, 6, 22, 4)
stop_time = dt.datetime(2011, 6, 22, 4, 5)

loc = "boxpol"
scan = 'rhi309'

path = get_xpol_path(start_time=start_time, loc=loc)
file_path = os.path.join(path, scan)

print(file_path)

flist = list(create_filelist(os.path.join(file_path, "*"), start_time, stop_time))

sweep_rhi_1 = wrl.io.open_gamic_mfdataset(flist, group='scan0')
sweep_rhi_1 = sweep_rhi_1.pipe(wrl.georef.georeference_dataset)
sweep_rhi_1 = sweep_rhi_1.isel(time=0)

sweep_rhi_1['DBZH'] += 2.5
sweep_rhi_1['ZDR'] += 0.8

fig = pl.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

zh_levels = np.arange(-10, 50, 5)
cmap = mpl.cm.get_cmap('turbo')

sweep_rhi_1.DBZH.plot.contourf(ax=ax1, x='gr', y='z', cmap=cmap, levels=zh_levels)
sweep_rhi_1.ZDR.plot.contourf(ax=ax2, x='gr', y='z', cmap=cmap, levels=zdr_levels)
ax1.set_ylim(0, 6000)
ax2.set_ylim(0, 6000)

#%% HYDROMETEOR CLASSIFICATION (Zrnic 2001)
# Select time
swp0 = swp.isel(time=150).compute()

# Load precipitation types
pr_types = wrl.classify.pr_types
for k, v in pr_types.items():
    print(f"{k:02} {' - '.join(v)}")

# Load Membershipfunctions Hydrometeorclassification
filename = wrl.util.get_wradlib_data_file('misc/msf_cband_v2.nc')
if 'cband' in filename:
    msf = xr.open_dataset(filename).cband.to_dataset('obs')
else:
    msf = xr.open_dataset(filename)

# Apply offsets if necessary
# add corrections here

# Load Sounding data (http://weather.uwyo.edu/upperair/sounding.html)
rs_id = 10393
rs_data, rs_meta = wrl.io.get_radiosonde(rs_id, start_time - dt.timedelta(hours=12), cols=(0,1,2))

# Extract temperature and height
stemp = rs_data['TEMP']
sheight = rs_data['HGHT']
# remove nans
idx = np.isfinite(stemp)
stemp = stemp[idx]
sheight = sheight[idx]

stemp_da = xr.DataArray(data=stemp, dims=["height"],
                        coords=dict(height=(["height"], sheight),       
                                   ),
                        attrs=dict(description="Temperature.",
                                   units="degC",
                                  ),
                       )

# Interpolate to higher resolution
hmax = 20000.
ht = np.arange(0., hmax)
itemp_da = stemp_da.interp({"height": ht})

# Fix Temperature below first measurement
itemp_da = itemp_da.bfill(dim="height")

# Get index into high res height
def merge_radar_profile(rds, cds):
    cds = cds.interp({'height': rds.z}, method='linear')
    rds = rds.assign({"TEMP": cds})
    return rds

hmc_ds = swp0.pipe(merge_radar_profile, itemp_da)

# Plot temperature profile
fig = plt.figure(figsize=(6, 5))
im = hmc_ds.TEMP.wradlib.plot_ppi(ax=111, fig=fig, proj="cg", cmap='viridis')
cgax = plt.gca()
caax = cgax.parasites[0]
paax = cgax.parasites[1]
cbar = plt.colorbar(im, ax=paax, fraction=0.046, pad=0.15)
cbar.set_label('Temperature [°C]')
caax.set_xlabel('Range [m]')
caax.set_ylabel('Range [m]')
plt.show()

# HMC workflow
# Set up independent observable ZH

msf_val = msf_index_indep_xarray(msf, swp0.DBZH.compute())

# Fuzzyfication and calculation of probability

fu = fuzzify(msf_val, hmc_ds, dict(ZH="DBZH", ZDR="ZDR", RHO="RHOHV", KDP="KDP", T="TEMP"))
fu = fu.chunk({"hmc": 1})
# weights dataset
w = xr.Dataset(dict(ZH=2., ZDR=1., RHO=1., KDP=0., TEMP=1.))
display(w)
prob = probability(fu, w)
#display(prob)
cls = classify(prob, threshold=0.)
#display(cls)


# Plot probability of HMC types

fig = plt.figure(figsize=(30,20))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 4, wspace=0.4, hspace=0.4)

prob = prob.where(prob > 0)
for i, pr in enumerate(pr_types.values()):
    if pr[0] == "NP":
        continue
    ax = fig.add_subplot(gs[i])
    prob.sel(hmc=pr[0]).plot(x="x", y="y", ax=ax, cbar_kwargs=dict(label="Probability"))
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Height [m]')
    t = ax.set_title(' - '.join(pr_types[i]))
    t.set_y(1.1) 



# Plot max probability

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cls.max("hmc").plot(x="x", y="y", cmap="cubehelix", ax=ax, cbar_kwargs=dict(label="Probability"))
ax.set_xlabel('Range [m]')
ax.set_ylabel('Range [m]')
t = ax.set_title('Hydrometeorclassification')
t.set_y(1.1) 



# Plot Classification Result

hmc = visdict14["HMC"]
ticks_hmc = hmc["ticks"]
cmap = hmc["cmap"]
norm = hmc["norm"]

hydro = cls.argmax("hmc")
hydro.attrs = dict(long_name="Hydrometeorclassification")

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
hmc = visdict14["HMC"]

# prepare colorbar
cbar_kwargs = dict(ticks=hmc["ticks"], norm=norm, extend="neither")
im = plot_moment(hydro, hmc["ticks"], ax=ax, cmap=hmc["cmap"], norm=norm, cbar_kwargs=cbar_kwargs)
# override colorbar ticklabels
labels = [pr_types[i][1] for i, _ in enumerate(pr_types)]
labels = im.colorbar.ax.set_yticklabels(labels)

display(hydro)

