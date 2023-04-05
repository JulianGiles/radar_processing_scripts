#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:43:05 2023

@author: jgiles
"""

# NEEDS WRADLIB 1.19 !! (OR GREATER?)

import datatree as dttree
import wradlib as wrl
import numpy as np
import sys
import glob
import xarray as xr

import warnings
warnings.filterwarnings('ignore')

# scan strategy for sorting elevations
scan_elevs = np.array([5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 8.0, 12.0, 17.0, 25.0])

# get list of files in the folder
path = sys.argv[1]
ll = sorted(glob.glob(path+"/ras*hd5"))

# extract list of moments 
moments = set(fp.split("_")[-2] for fp in ll)

# discard "allmoms" from the set if it exists
moments.discard("allmoms")

# for every moment, open all files in folder (all timesteps) per moment into a dataset
vardict = {} # a dict for putting a dataset per moment
for mom in moments:
    
    # print("       Processing "+mom)
    
    # open the odim files (single moment and elevation, several timesteps)
    llmom = sorted(glob.glob(path+"/ras*_"+mom+"_*hd5"))
    
    # # there is a bug with the current implementation of xradar. Re check this in future releases
    # # Looks like now it works with a temporary fix in the files
    vardict[mom] = wrl.io.open_odim_mfdataset(llmom)
    
    # if some coord has dimension time, reduce using median
    for coord in ["latitude", "longitude", "altitude", "elevation"]:
        if "time" in vardict[mom][coord].dims:
            vardict[mom].coords[coord] = vardict[mom].coords[coord].median("time")
    
    # the old method seems to still work fine
    # vardict[mom] = wrl.io.open_odim(llmom, loader="h5py", chunks={})[0].data
    
# create an empty radar volume and put the previous data inside
vol = wrl.io.RadarVolume()
vol.append(xr.merge(vardict.values()))

# Create a datatree
dtree = dttree.DataTree(name="root")

# for every elevation in the volume (there is only 1)
for i, sw in enumerate(vol):
    
    # dim0 = list(set(sw.dims) & {"azimuth", "elevation"})[0]
    
    # check and fix how the angle variable is named
    if "fixed_angle" in sw:
        # rename the variable
        sw = sw.rename({"fixed_angle": "sweep_fixed_angle"}) 
        
    # get the sweep number according to the scan strategy
    try:
        ii = int(np.where(scan_elevs == round(float(sw.attrs["fixed_angle"]),1))[0])
    except:
        ii = 0

    # Put the data in the data tree
    dttree.DataTree(sw, name=f"sweep_{ii}", parent=dtree)
    

# Save the datatree as netcdf 
name = ll[0].split("_")
name[2]="allmoms"
dtree.load().to_netcdf("_".join(name))
  
# make a list of valid timesteps (those with dbzh > 5 in at least 1% of the bins)  
valid = (vardict["dbzh"]["DBZH"][:]>5).sum(dim=("azimuth", "range")).compute() > vardict["dbzh"]["DBZH"][0].count().compute()*0.01
valid = valid.time.where(valid, drop=True)

# save the list as a txt file named true if there is any value, otherwise false
if len(valid)>0:
    np.savetxt(path+"true.txt", valid.values.astype(str), fmt="%s")
else:
    np.savetxt(path+"false.txt", valid.values.astype(str), fmt="%s")



# # to load the datatree
# vol_reload = dttree.open_datatree("_".join(name))

# vol_reload["sweep_1"].ds # get as dataset
# swp = vol_reload["sweep_1"].to_dataset() # get a sweep
    
