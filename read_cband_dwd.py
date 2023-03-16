import glob
import os
import re
import psutil

import warnings
import gc

warnings.simplefilter('ignore')

import datetime as dt
process = psutil.Process(os.getpid())

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


'''
import pathlib
import time
import pickle
from xhistogram.xarray import histogram
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import wradlib as wrl
import xarray as xr
from matplotlib.colors import LogNorm
import pandas as pd
import scipy
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.dates import DayLocator, HourLocator, MinuteLocator,DateFormatter, drange

from scipy.integrate import cumtrapz
from scipy import stats, spatial
from scipy import ndimage

from tqdm import tqdm
import netCDF4 as nc

import hvplot
import hvplot.xarray
import holoviews as hv

'''

def memory_usage_psutil():
    # return the memory usage in MB
    rocess = psutil.Process(os.getpid())
    mem = process.memory_full_info().uss / float(1 << 20)
    return mem

def free_memory():
    mem0 = memory_usage_psutil()
    print(gc.collect())
    proc = psutil.Process()
    mem1 = memory_usage_psutil()
    print("Memory freed: {0:.5f} MB".format((mem0-mem1)))

def check_open_files(full=False):
    proc = psutil.Process()
    print(len(proc.open_files()))
    if full:
        print(proc.open_files())
        
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
        
def get_data_path_dwd(inpath):
    """ Get data path (realpep) 
    """
    return os.path.join(inpath, '{year}-{month:02d}-{day:02d}/{radar}/{mode}/{scan}')

def get_file_name_dwd():
    """ Get data path (automount) 
    """
    return ('ras07-{mode}_sweeph5onem_{moment}_{scan}-{year}{month:02d}{day:02d}'
            '{hour:02d}{mintens}[{minones0}-{minones1}]*')


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
            
