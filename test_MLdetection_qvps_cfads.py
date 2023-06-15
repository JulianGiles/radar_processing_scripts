#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:31:39 2023

@author: jgiles
"""
import os
os.chdir('/home/jgiles/')


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

try:
    from Scripts.python.radar_processing_scripts import utils
    from Scripts.python.radar_processing_scripts import radarmet
except ModuleNotFoundError:
    import utils
    import radarmet


import warnings
warnings.filterwarnings('ignore')


#%% Load files
# paths to files to load
path = "/home/jgiles/dwd/pulled_from_detect/2017/2017-07/2017-07-2*/pro/vol5minng01/07/*allmoms*"
files = sorted(glob.glob(path))

# create a list to store each file as datatree
dtree = []
# load the datatrees, convert to dataset and append to the list
for ff in files:
    dtree.append( dttree.open_datatree(ff)["sweep_7"].to_dataset() )

# put everything together
swps = xr.concat(dtree, "time")
swps = swps.pipe(wrl.georef.georeference_dataset)

#%% QVP

# Azimuthally averaged profiles of a conical volume measured at elevations between 10 and 20 degrees

# create height coord
swps = swps.assign_coords(height=swps.z.mean('azimuth'))

# Apply offsets if necessary
# add corrections here
# Remove Clutter and unrealistic data 
mask = (swps["RHOHV"] < 0.8) | (swps["SNRHC"] < 5) | (swps["DBZH"] < 0) | (swps["ZDR"] < -1.5) | (swps["KDP"] < 0.5) | (swps["ZDR"] > 5) | (swps["KDP"] > 3)
swps = swps.where(~mask)


# Calculate median
qvp = swps.median('azimuth')

#%% Add temperature profile

startdt0 = datetime.datetime.utcfromtimestamp(int(swps.time[0].values)/1e9).date()
enddt0 = datetime.datetime.utcfromtimestamp(int(swps.time[-1].values)/1e9).date() + datetime.timedelta(hours=24)

# transform the dates to datetimes
startdt = datetime.datetime.fromordinal(startdt0.toordinal())
enddt = datetime.datetime.fromordinal(enddt0.toordinal())

def date_range_list(start_date, end_date):
    # Return list of datetime.date objects (inclusive) between start_date and end_date (inclusive).
    date_list = []
    curr_date = start_date
    while curr_date <= end_date:
        date_list.append(curr_date)
        curr_date += datetime.timedelta(hours=12)
    return date_list

dttimes = date_range_list(startdt, enddt)


# Create a list of xarray datasets with sounding data to then concatenate
soundings = list()
for tt in dttimes:
    # Load Sounding data (http://weather.uwyo.edu/upperair/sounding.html)
    rs_id = 10393 # Close to radar site
    rs_data, rs_meta = wrl.io.get_radiosonde(rs_id, tt, cols=(0,1,2))
    
    #rs_array = np.array( list( map(list, rs_data) ) )

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
    
    soundings.append(stemp_da)

# Create time dimension and concatenate
timedim = xr.DataArray( pd.to_datetime(dttimes), [("time", pd.to_datetime(dttimes))] )
temperatures = xr.concat(soundings, timedim)

# Interpolate to higher resolution
hmax = 20000.
ht = np.arange(0., hmax)
itemp_da = temperatures.interpolate_na("height").interp({"height": ht})

# Fix Temperature below first measurement and above last one
itemp_da = itemp_da.bfill(dim="height")
itemp_da = itemp_da.ffill(dim="height")

# Interpolate to dataset height and time, then add to dataset
def merge_radar_profile(rds, cds):
    # cds = cds.interp({'height': rds.z}, method='linear')
    cds = cds.interp({'height': rds.height}, method='linear')
    cds = cds.interp({"time": rds.time}, method="linear")
    rds = rds.assign({"TEMP": cds})
    return rds

qvp = qvp.pipe(merge_radar_profile, itemp_da)

qvp.coords["TEMP"] = qvp["TEMP"] # move TEMP from variable to coordinate

# Plot temperature vs height for checking
qvp["TEMP"].plot(x="time", y="height")

# Plot DBZH QVP vs TEMP for checking
qvp.DBZH.plot(x="time", y="TEMP", yincrease=False)

#%% Compute CFADs from QVPs (UNFINISHED)

refvar = "TEMP" # reference variable to compute CFADs

for vv in qvp.data_vars:
    qvp_bin = qvp[vv].groupby_bins("TEMP",100)
    
    np.histogram(qvp_bin, 100)


# stack the array
qvp_stacked = qvp[vv].stack({"stacked":("time", "range")})
# get rid of nans
qvp_stacked = qvp_stacked[np.isfinite(qvp_stacked)]
# calculate histogram
tempbins = np.arange(-50, 16, 1)
hist2d, xedges, yedges = np.histogram2d(qvp_stacked, qvp_stacked["TEMP"], bins=[50,tempbins], density=True)
xedges1 = (xedges[1:] + xedges[:-1])/2
yedges1 = (yedges[1:] + yedges[:-1])/2

cfad = xr.DataArray(hist2d, dims=[vv, refvar], coords={vv:xedges1, refvar:yedges1})


# plot
cfad.plot(x="DBZH", y="TEMP", yincrease=False)

# plt.gca().invert_yaxis()



#%% Testing Tobi's functions

from Scripts.python.radar_processing_scripts import utils
from Scripts.python.radar_processing_scripts import radarmet

################## Before entropy calculation we need to use the melting layer detection algorithm 
ds = swps
interpolation_method_ML = "linear"

# name the variables. Variables should be offset corrected (_OC in BoXpol) and RHOHV should be noise corrected (_NC)
X_ZH = "DBZH"  # "DBTH"
X_RHOHV = "RHOHV"
X_PHIDP = "UPHIDP" # "PHIDP"

######### Processing PHIDP
#### fix PHIDP
# filter
phi = ds[X_PHIDP].where((ds[X_RHOHV]>=0.9) & (ds[X_ZH]>=0))
# calculate offset
phidp_offset = phi.pipe(radarmet.phase_offset, rng=3000)
off = phidp_offset["PHIDP_OFFSET"]
start_range = phidp_offset["start_range"]

# apply offset
fix_range = 750
phi_fix = ds[X_PHIDP].copy()
off_fix = off.broadcast_like(phi_fix)
phi_fix = phi_fix.where(phi_fix.range >= start_range + fix_range).fillna(off_fix) - off

# smooth and mask
window = 11
window2 = None
phi_median = phi_fix.pipe(radarmet.xr_rolling, window, window2=window2, method='median', skipna=True, min_periods=3)
phi_masked = phi_median.where((ds[X_RHOHV] >= 0.95) & (ds[X_ZH] >= 0.)) 

dr = phi_masked.range.diff('range').median('range').values / 1000.
print("range res [km]:", dr)

# derive KDP from PHIDP (convolution method)
winlen = 5 # windowlen 
min_periods = 3 # min number of vaid bins
kdp = radarmet.kdp_from_phidp(phi_masked, winlen, min_periods=3)
kdp1 = kdp.interpolate_na(dim='range')

# derive PHIDP from KDP (convolution method)
winlen = 5 
phidp = radarmet.phidp_from_kdp(kdp1, winlen)

# assign new variables to dataset
assign = {X_PHIDP+"_OC_SMOOTH": phi_median.assign_attrs(ds[X_PHIDP].attrs),
  X_PHIDP+"_OC_MASKED": phi_masked.assign_attrs(ds[X_PHIDP].attrs),
  "KDP_CONV": kdp.assign_attrs(ds.KDP.attrs),
  "PHI_CONV": phidp.assign_attrs(ds[X_PHIDP].attrs),

  X_PHIDP+"_OFFSET": off.assign_attrs(ds[X_PHIDP].attrs),
  X_PHIDP+"_OC": phi_fix.assign_attrs(ds[X_PHIDP].attrs)}


#### Compute QVP
## Only data with a cross-correlation coefficient ρHV above 0.7 are used to calculate their azimuthal median at all ranges (from Trömel et al 2019).
ds = ds.assign(assign)
ds_qvp_ra = ds.where(ds[X_RHOHV] > 0.7).median("azimuth", keep_attrs=True)
ds_qvp_ra = ds_qvp_ra.assign_coords({"z": ds["z"].median("azimuth")})

ds_qvp_ra = ds_qvp_ra.swap_dims({"range":"z"}) # swap range dimension for height

# filter out values close to the ground
ds_qvp_ra2 = ds_qvp_ra.where(ds_qvp_ra["z"]>300)

#### Detect melting layer
moments={X_ZH: (10., 60.), X_RHOHV: (0.65, 1.), X_PHIDP+"_OC": (-20, 360)}
dim = 'z'
thres = 0.02 # gradient values over thres are kept. Lower is more permissive
xwin = 9 # value for the time median smoothing
ywin = 1 # value for the height mean smoothing (1 for Cband)
fmlh = 0.3
 
ml_qvp = utils.melting_layer_qvp_X_new(ds_qvp_ra2, moments=moments, 
         dim=dim, thres=thres, xwin=xwin, ywin=ywin, fmlh=fmlh, all_data=True, clowres=True)





#### Assign ML values to dataset

ds = ds.assign_coords({'height_ml': ml_qvp.mlh_top})
ds = ds.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})

ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml': ml_qvp.mlh_top})
ds_qvp_ra = ds_qvp_ra.assign_coords({'height_ml_bottom': ml_qvp.mlh_bottom})

#### Giagrande refinment
hdim = "z"
# get data iside the currently detected ML
cut_above = ds_qvp_ra.where(ds_qvp_ra[hdim]<ds_qvp_ra.height_ml)
cut_above = cut_above.where(ds_qvp_ra[hdim]>ds_qvp_ra.height_ml_bottom)
#test_above = cut_above.where((cut_above.rho >=0.7)&(cut_above.rho <0.98))

# get the heights with min RHOHV
min_height_ML = cut_above[X_RHOHV].idxmin(dim=hdim) 

# cut the data below and above the previous value
new_cut_below_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] > min_height_ML)
new_cut_above_min_ML = ds_qvp_ra.where(ds_qvp_ra[hdim] < min_height_ML)

# Filter out values outside some RHOHV range
new_cut_below_min_ML_filter = new_cut_below_min_ML[X_RHOHV].where((new_cut_below_min_ML[X_RHOHV]>=0.97)&(new_cut_below_min_ML[X_RHOHV]<=1))
new_cut_above_min_ML_filter = new_cut_above_min_ML[X_RHOHV].where((new_cut_above_min_ML[X_RHOHV]>=0.97)&(new_cut_above_min_ML[X_RHOHV]<=1))            


######### ML TOP Giagrande refinement

notnull = new_cut_below_min_ML_filter.notnull() # this replaces nan for False and the rest for True
first_valid_height_after_ml = notnull.where(notnull).idxmax(dim=hdim) # get the first True value, i.e. first valid value

######### ML BOTTOM Giagrande refinement
# For this one, we need to flip the coordinate so that it is actually selecting the last valid index
notnull = new_cut_above_min_ML_filter.notnull() # this replaces nan for False and the rest for True
last_valid_height = notnull.where(notnull).isel({hdim:slice(None, None, -1)}).idxmax(dim=hdim) # get the first True value, i.e. first valid value (flipped)


ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
ds_qvp_ra = ds_qvp_ra.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))


ds = ds.assign_coords(height_ml_new_gia = ("time",first_valid_height_after_ml.data))
ds = ds.assign_coords(height_ml_bottom_new_gia = ("time", last_valid_height.data))


#### Fix KDP in the ML using PHIDP:

ds_qvp_ra3 = ds_qvp_ra.where(ds_qvp_ra.height_ml_new_gia<1000)

# get where PHIDP has nan values
nan = np.isnan(ds[X_PHIDP+"_OC_MASKED"]) 
# get PHIDP outside the ML
phi2 = ds[X_PHIDP+"_OC_MASKED"].where((ds.z < ds_qvp_ra3.height_ml_bottom_new_gia) | (ds.z > ds_qvp_ra3.height_ml_new_gia))#.interpolate_na(dim='range',dask_gufunc_kwargs = "allow_rechunk")
# interpolate PHIDP in ML
phi2 = phi2.interpolate_na(dim='range', method=interpolation_method_ML)
# restore originally nan values
phi2 = xr.where(nan, np.nan, phi2)

# Derive KPD from the new PHIDP
dr = phi2.range.diff('range').median('range').values / 1000.
print("range res [km]:", dr)
# winlen in gates
# TODO: window length in m
winlen = 5
min_periods = 3
kdp_ml = radarmet.kdp_from_phidp(phi2, winlen, min_periods=3)

# assign to datasets
ds = ds.assign({"KDP_ML_corrected": (["time", "azimuth", "range"], kdp_ml.values, ds_qvp_ra3.KDP.attrs)})

#### Optional filtering:
ds["KDP_ML_corrected"] = ds.KDP_ML_corrected.where((ds.KDP_ML_corrected >= 0.01) & (ds.KDP_ML_corrected <= 3)) 

ds = ds.assign_coords({'height': ds.z})

kdp_ml_qvp = ds["KDP_ML_corrected"].median("azimuth", keep_attrs=True)
kdp_ml_qvp = kdp_ml_qvp.assign_coords({"z": ds["z"].median("azimuth")})
kdp_ml_qvp = kdp_ml_qvp.swap_dims({"range":"z"}) # swap range dimension for height
ds_qvp_ra = ds_qvp_ra.assign({"KDP_ML_corrected": kdp_ml_qvp})


#### Classification of stratiform events based on entropy

# calculate linear values for ZH and ZDR
ds = ds.assign({"DBZH_lin": wrl.trafo.idecibel(ds["DBZH"]), "ZDR_lin": wrl.trafo.idecibel(ds["ZDR"]) })

# calculate entropy
Entropy = utils.Entropy_timesteps_over_azimuth_different_vars_schneller(ds, zhlin="DBZH_lin", zdrlin="ZDR_lin", rhohvnc=X_RHOHV, kdp="KDP_ML_corrected")

# concate entropy for all variables and get the minimum value 
strati = xr.concat((Entropy.entropy_zdrlin, Entropy.entropy_Z, Entropy.entropy_RHOHV, Entropy.entropy_KDP),"entropy")        
min_trst_strati = strati.min("entropy")

# assign to datasets
ds["min_entropy"] = min_trst_strati

min_trst_strati_qvp = min_trst_strati.assign_coords({"z": ds["z"].median("azimuth")})
min_trst_strati_qvp = min_trst_strati_qvp.swap_dims({"range":"z"}) # swap range dimension for height
ds_qvp_ra = ds_qvp_ra.assign({"min_entropy": min_trst_strati_qvp})









#%% PLOTS


# Better colormaps
from Scripts.python.radar_processing_scripts import colormap_generator



timesel = "2019-06-12"

mom = "KDP"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
# cmap = mpl.cm.get_cmap("HomeyerRainbow")
# cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
cmap = visdict14[mom]["cmap"]

# QVP with ML heights
ds_qvp_ra["KDP"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("original KDP")

# ds_qvp_ra.height_ml_new_gia.loc[{"time":"2017-07-25"}].plot(x="time", c="magenta")
# ds_qvp_ra.height_ml_bottom_new_gia.loc[{"time":"2017-07-25"}].plot(x="time", c="teal")

ds_qvp_ra["KDP_CONV"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("KDP from PHIDP")

ds_qvp_ra["KDP_ML_corrected"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("KDP from PHIDP, ML corrected")



mom = "PHI"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
# cmap = mpl.cm.get_cmap("HomeyerRainbow")
# cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
cmap = visdict14[mom]["cmap"]

# QVP with ML heights
ds_qvp_ra["UPHIDP"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("original PHIDP")

# QVP with ML heights
ds_qvp_ra["UPHIDP_OC"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("UPHIDP OC")


mom = "DBZH"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
# cmap = mpl.cm.get_cmap("HomeyerRainbow")
# cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
cmap = visdict14[mom]["cmap"]


# QVP with ML heights
ds_qvp_ra["DBZH"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
ds_qvp_ra.height_ml.loc[{"time":timesel}].plot(x="time", c="red")
ds_qvp_ra.height_ml_bottom.loc[{"time":timesel}].plot(x="time", c="blue")
plt.title("DBZH")



# QVP with ML heights and entropy classification
mom = "DBZH"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = mpl.cm.get_cmap("HomeyerRainbow")
# cmap = get_discrete_cmap(visdict14["DBZH"]["ticks"], 'HomeyerRainbow')
# cmap = visdict14[mom]["cmap"]

ds_qvp_ra["DBZH"].loc[{"time":timesel}].plot(x="time", cmap=cmap, norm=norm, extend="both")
plt.title("DBZH")
ds_qvp_ra.height_ml_new_gia.loc[{"time":timesel}].plot(x="time", c="magenta")
ds_qvp_ra.height_ml_bottom_new_gia.loc[{"time":timesel}].plot(x="time", c="teal")

ds_qvp_ra["min_entropy"].loc[{"time":timesel}].plot.contourf(x="time", levels=[0.8,1], hatches=["", "X"], colors="none", add_colorbar=False)



# PPI

timesel = "2017-10-28 16"
timeind = 0

mom = "KDP"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

pm = ds["KDP"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("original KDP")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = ds["KDP_CONV"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("KDP from PHIDP")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = ds["KDP_ML_corrected"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("KDP from PHIDP, ML corrected")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)


mom = "DBZH"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

pm = ds["DBZH"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("DBZH")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)


mom = "TH"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

pm = ds["TH"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("TH")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)



mom = "PHI"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = mpl.cm.get_cmap("seismic") #visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow") # mpl.cm.get_cmap("seismic")

pm = ds["UPHIDP"].loc[{"time":timesel}][timeind].wradlib.plot( cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("UPHIDP")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = ds["UPHIDP_OC"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("UPHIDP OC")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = ds["UPHIDP_OC"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("UPHIDP OC")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)



mom = "RHOHV"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

pm = ds["RHOHV"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("RHOHV")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)



mom = "ZDR"
visdict14 = radarmet.visdict14
norm = radarmet.get_discrete_norm(visdict14[mom]["ticks"])
cmap = visdict14[mom]["cmap"] #mpl.cm.get_cmap("HomeyerRainbow")

pm = ds["ZDR"].loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("ZDR")
plt.text(10000, 30100, timesel+"H"+str(timeind*5)+"M") # add date and time
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)




# testing RHOHV noise correction
corr, hist, std, rn = utils.calculate_noise_level(ds["DBZH"], ds["RHOHV"], noise=(-40, -20, 1), rho_bins=(0.9, 1.1, 0.005), snr_bins=(5., 30., .1))

corr, hist, std, rn = utils.calculate_noise_level(ds["DBZH"], ds["RHOHV"], noise=(-40, -20, 1), rho_bins=(0.9, 1.1, 0.005), snr_bins=(5., 30., .1))
rhohvcorr = corr[np.argmin(std)][1]
stn = corr[np.argmin(std)][0]


pm = rhohvcorr.loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("RHOHV corrected")
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = stn.loc[{"time":timesel}][timeind].wradlib.plot( extend="both")
plt.colorbar(pm, extend="both")
plt.title("RHOHV signal to noise")
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)

pm = rhohvcorr.where(stn.loc[{"time":timesel}][timeind]>10).loc[{"time":timesel}][timeind].wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("RHOHV corrected")
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)


overone = rhohvcorr.loc[{"time":timesel}][timeind]>1
pm = overone.wradlib.plot( extend="both")
plt.colorbar(pm, extend="both")
plt.title("RHOHV corrected")
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)



# line plots
azind = 250 # azimuth index to plot

ds["KDP_CONV"].loc[{"time":timesel}][timeind, azind,].plot(xlim=(0,20000), marker="o")
ds["UPHIDP"].loc[{"time":timesel}][timeind, azind,].plot(xlim=(0,20000), ylim=(0,100), marker="o")
ds["UPHIDP_OC"].loc[{"time":timesel}][timeind, azind,].plot(xlim=(0,20000), ylim=(-5,5), marker="o")
ds["DBZH"].loc[{"time":timesel}][timeind, azind,].plot(xlim=(0,20000), ylim=(0,40), marker="o")



# testing why KDP is negative
# checking the gradient in phidp
selphidp = ds["UPHIDP_OC"].loc[{"time":"2017-07-25 T04"}][0]
pm = ((selphidp[:,1:21] - selphidp[:,0:20].to_numpy() ) < 0).wradlib.plot()
plt.colorbar(pm, extend="both")

# checking phi_masked
selphidp = phi_masked.loc[{"time":"2017-07-25 T04"}][0]
pm = selphidp.wradlib.plot(norm=norm, cmap=cmap, extend="both")
plt.colorbar(pm, extend="both")
plt.title("phi smoothed and masked")
plt.ylim(-30000, 30000)
plt.xlim(-30000, 30000)






#entropy
entropy_ppi = ds["min_entropy"].broadcast_like(ds["DBZH"])
entropy_ppi = entropy_ppi.assign_coords({"x":ds["x"], "y":ds["y"]})

pm = entropy_ppi.loc[{"time":"2017-07-25 T04"}][0].wradlib.plot( levels=[0,0.8,1])
plt.colorbar(pm, extend="both")
plt.title("min entropy on 2017-07-25 T04")



#%% TESTS

path_qvps = "/home/jgiles/dwd/qvps/*/*/2017-11-1*/pro/vol5minng01/07/*allmoms*"
files = sorted(glob.glob(path_qvps))

qvps = xr.open_mfdataset(files)


qvps0 = xr.open_mfdataset(files[0:5])
qvps1 = xr.open_mfdataset(files[5:])


qvps2 = xr.open_mfdataset(files[3])
qvps2.where(qvps2.time.notnull(), drop=True)


np.sign(qvps.z.diff("z"))

qvps.loc[{"time": "2016-04-27"}]["DBZH"].plot(x="time", vmin=0, cmap="plasma")
qvps.loc[{"time": "2016-04-27"}]["min_entropy"].plot(x="time", cmap="plasma")
qvps.loc[{"time": "2016-04-27"}]["min_entropy"].plot(x="time", vmin=0.8, cmap="plasma")


#%% CFADs Load and compute
# This part should be run after having the QVPs computed (compute_qvps.py)

#### Open QVP files
path_qvps = "/home/jgiles/dwd/qvps/*/*/*/pro/vol5minng01/07/*allmoms*"
# path_qvps = "/automount/ftp/jgiles/qvps2/*/*/*/tur/vol5minng01/07/*allmoms*"
path_qvps = "/home/jgiles/dmi/qvps/*/*/*/ANK/*/*/*allmoms*"

files = sorted(glob.glob(path_qvps))

TH = "DBZH" # set TH name, or set DBZH if TH not available, like in some turkish radars


# there are slight differences (noise) in z coord sometimes so we have to align all datasets
# since the time coord has variable length, we cannot use join="override" so we define a function to copy
# the z coord from the first dataset into the rest with preprocessing
# There are also some time values missing, ignore those
# Some files do not have TEMP data, fill with nan
first_file = xr.open_mfdataset(files[0]) 
first_file_z = first_file.z.copy()
def fix_z_and_time(ds):
    ds.coords["z"] = first_file_z
    ds = ds.where(ds["time"].notnull(), drop=True)
    if "TEMP" not in ds.coords:
        ds.coords["TEMP"] = xr.full_like( ds["DBZH"], np.nan ).compute()
        
    return ds
    
try:
    qvps = xr.open_mfdataset(files, preprocess=fix_z_and_time)
except: 
    # if the above fails, just combine everything and fill the holes with nan (Turkish case)
    qvps = xr.open_mfdataset(files, combine="nested", concat_dim="time")


#### Filters
# Filter only stratiform events (min entropy >= 0.8)
with ProgressBar():
    qvps_strat = qvps.where(qvps["min_entropy"]>=0.8, drop=True).compute()
# Filter relevant values
qvps_strat_fil = qvps_strat.where((qvps_strat[TH] > 0 )&(qvps_strat["KDP_ML_corrected"] > 0.0)&(qvps_strat["RHOHV"] > 0.7)&(qvps_strat["ZDR"] > -1))

#### General statistics
values_sfc = qvps_strat_fil.isel({"z": 2})
values_snow = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_new_gia"]}, method="nearest")
values_rain = qvps_strat_fil.sel({"z": qvps_strat_fil["height_ml_bottom_new_gia"]}, method="nearest")
    
#### ML statistics
# select values inside the ML
qvps_ML = qvps_strat_fil.where( (qvps_strat_fil["z"] < qvps_strat_fil["height_ml_new_gia"]) & \
                               (qvps_strat_fil["z"] > qvps_strat_fil["height_ml_bottom_new_gia"]), drop=True)

values_ML_max = qvps_ML.max(dim="z")
values_ML_min = qvps_ML.min(dim="z")
values_ML_mean = qvps_ML.mean(dim="z")
ML_height = qvps_ML["height_ml_new_gia"] - qvps_ML["height_ml_bottom_new_gia"]

# Silke style
# select timesteps with detected ML
gradient_silke = qvps_strat_fil.where(qvps_strat_fil["height_ml_new_gia"] > qvps_strat_fil["height_ml_bottom_new_gia"], drop=True)
gradient_silke_ML = gradient_silke.sel({"z": gradient_silke["height_ml_new_gia"]}, method="nearest")
gradient_silke_ML_plus_2km = gradient_silke.sel({"z": gradient_silke_ML["z"]+2000}, method="nearest")
gradient_final = (gradient_silke_ML_plus_2km - gradient_silke_ML)/2
beta = gradient_final[TH] #### TH OR DBZH??


#### DGL statistics
# select values in the DGL 
qvps_DGL = qvps_strat_fil.where((qvps_strat_fil["TEMP"] >= -20)&(qvps_strat_fil["TEMP"] <= -10), drop=True)    

values_DGL_max = qvps_DGL.max(dim="z")
values_DGL_min = qvps_DGL.min(dim="z")
values_DGL_mean = qvps_DGL.mean(dim="z")


#%% CFADs Plot

# adjustment from K to C
adjtemp = 0
if (qvps_strat_fil["TEMP"]>100).any(): #if there is any temp value over 100, we assume the units are Kelvin
    print("at least one TEMP value > 100 found, assuming TEMP is in K and transforming to C")
    adjtemp = -273.15 # adjustment parameter from K to C

# top temp limit
ytlim=-20

# Temp bins
tb=1# degress C

# Min counts per Temp layer
mincounts=200

#Colorbar limits and step
cblim=[0,10]
colsteps=10


# Plot horizontally
vars_to_plot = {"DBZH": [0, 51, 1], 
                "ZDR": [-1, 3.1, 0.1],
                "KDP_ML_corrected": [0, 0.51, 0.01],
                "RHOHV": [0.9, 1.005, 0.005]}


fig, ax = plt.subplots(1, 4, sharey=True, figsize=(20,5), width_ratios=(1,1,1,1.15+0.05*2))# we make the width or height ratio of the last plot 15%+0.05*2 larger to accomodate the colorbar without distorting the subplot size

for nn, vv in enumerate(vars_to_plot.keys()):

    utils.hist2d(ax[nn], qvps_strat_fil[vv], qvps_strat_fil["TEMP"]+adjtemp, whole_x_range=True, 
                 binsx=vars_to_plot[vv], binsy=[-20,15,tb], mode='rel_y', qq=0.2,
                 cb_mode=(nn+1)/len(vars_to_plot), cmap="plasma", colsteps=colsteps, 
                 fsize=20, mincounts=mincounts, cblim=cblim, N=(nn+1)/len(vars_to_plot), 
                 cborientation="vertical", shading="nearest")
    ax[nn].set_ylim(15,ytlim)
    ax[nn].set_xlabel(vv, fontsize=10)
    
    ax[nn].tick_params(labelsize=15) #change font size of ticks
    plt.rcParams.update({'font.size': 15}) #change font size of ticks for line of counts



ax[0].set_ylabel('Temperature [°C]', fontsize=15, color='black')



