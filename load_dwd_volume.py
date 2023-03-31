 
import os

import re

import wradlib as wrl
import numpy as np
import sys
import glob
import datetime as dt
import datetime

import xarray as xr

sys.path.insert(0, '../')

def get_datetime_from_filename(filename, regex):
    """Get datetime from filename
    """
    fmt = "%Y%m%d%H%M%S"
    match = re.search(regex, os.path.basename(filename))
    match = ''.join(re.findall(r'[0-9]+', match.group()))
    return dt.datetime.strptime(match, fmt)

def get_moment_from_filename(filename, regex):
    """Get datetime from filename
    """
    match = re.search(regex, os.path.basename(filename))
    return match[1]

def create_dwd_filelist(path=None, starttime=dt.datetime.today(), endtime=None, mode='vol5minng01', moments=None, scan='*', loc='ESS'):
    """Create filelist from path_glob and filename dates
    """
    if path is None:
        path = f'/automount/realpep/upload/RealPEP-SPP/DWD-CBand/'
    path = path+starttime.strftime("%Y/")+ starttime.strftime("%Y-%m/")#+ starttime.strftime("%Y-%m-%d/")
    date = '{0}-{1:02d}-{2:02d}'.format(starttime.year, starttime.month, starttime.day)
    radar_path = os.path.join(path, date)
    radar_path = os.path.join(radar_path, loc)
    radar_path = os.path.join(radar_path, mode)
    radar_path = os.path.join(radar_path, scan)
    radar_path = os.path.join(radar_path, '*')
    #print(radar_path)
    #radar_path = radar_path.format(starttime.year, starttime.month, starttime.day)
    file_names = sorted(glob.glob(radar_path))

    if endtime is None:
        endtime = starttime + dt.timedelta(minutes=5)

    for fname in file_names:
        if moments is not None:
            mom = get_moment_from_filename(fname, r"sweeph5onem_(.*?)_").upper()
            if mom not in moments:
                continue
        time = get_datetime_from_filename(fname, r"\d{14}")
        if time >= starttime and time < endtime:
            yield fname



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


# Scan path for RAW Data
realpep_path = '/automount/realpep/upload/RealPEP-SPP/DWD-CBand/'
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
