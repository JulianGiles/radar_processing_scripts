#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qvps_interactive.py

Interactive visualization of Quasi-Vertical Profile (QVP) radar data.
Displays:
  - Column 1: 4 stacked 2D QVP heatmaps (Time vs Height) for the day.
              Default variables:
                Panel 1: DBZH_AC (or DBZH, DBTH, ZH, TH)
                Panel 2: ZDR_EC_OC_WRC_AC or ZDR_EC_OC_AC (or ZDR_OC, ZDR, etc.)
                Panel 3: KDP_ML_corrected_EC (or KDP_ML_corrected, KDP)
                Panel 4: RHOHV_NC (or RHOHV)
              Each panel allows selecting any available variable, changing colormaps,
              and setting custom vmin/vmax.
              Clicking anywhere on any 2D plot selects that specific time profile.
  - Column 2: 1D Value vs Height profile for the clicked time.
              Automatically displays the 4 variables from Column 1, plus a checkbox list
              to add/remove any variable from all available variables.
              All variables share the height axis with twin x-axes, each scaled to manageable
              ranges, with customizable axis limits, line colors, and line styles.
  - Date navigation: Slider and step buttons to move across daily files or dates.

Author: Antigravity / jgiles
"""

import os
os.environ["ICECHUNK_NO_LOGS"] = "1" # Silence Icechunk Rust warnings
import sys
import glob
import re
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Icechunk support
try:
    from icechunk import Repository, local_filesystem_storage
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

# Possible paths setup (following CFTDs.py convention)
possible_paths = [
    '/home/jgiles/Scripts/python/radar_processing_scripts',              # Office PC
    '/p/scratch/detectrea2/giles1/radar_processing_scripts',             # JUWELS Scratch
]

for script_dir in possible_paths:
    if os.path.exists(script_dir) and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        break

# Attempt imports of local project modules
try:
    import utils
except ImportError:
    utils = None

try:
    import radarmet
except ImportError:
    radarmet = None

try:
    from scipy.ndimage import binary_opening
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Default path pattern following CFTDs.py
DEFAULT_REALPEP_PATH = "/automount/realpep"
DEFAULT_PATH_QVPS = f"{DEFAULT_REALPEP_PATH}/upload/jgiles/dwd/qvps/*/*/2017-07-25/pro/vol5minng01/07/*allmoms*"
DEFAULT_RIMING_DIR = f"{DEFAULT_REALPEP_PATH}/upload/jgiles/radar_riming_classif/unfiltered"

# Common colormap choices
COLORMAP_CHOICES = [
    'default (visdict)', 'viridis', 'miub2', 'HomeyerRainbow', 'HomeyerRainbow_disc',
    'SpectralExtended', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet',
    'RdBu_r', 'coolwarm', 'binary2'
]

# Distinct palette for 1D profile curves
DEFAULT_PALETTE = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#ff7f0e',  # orange
    '#17becf',  # cyan
    '#e377c2',  # pink
    '#8c564b',  # brown
    '#bcbd22',  # olive
    '#333333',  # dark gray
    '#000080',  # navy
    '#008080',  # teal
]

# Register fallback custom colormaps if missing
colors_prabhakar = np.array([
    [0.00, 1.00, 1.00], [0.00, 0.70, 0.93], [0.00, 0.00, 1.00],
    [0.50, 1.00, 0.00], [0.40, 0.80, 0.00], [0.27, 0.55, 0.00],
    [1.00, 1.00, 0.00], [0.80, 0.80, 0.00], [1.00, 0.65, 0.00],
    [1.00, 0.27, 0.00], [0.80, 0.22, 0.00], [0.55, 0.15, 0.00],
    [1.00, 0.00, 1.00], [0.58, 0.44, 0.86]
])
try:
    if "miub2" not in mpl.colormaps:
        mpl.colormaps.register(name="miub2", cmap=mcolors.ListedColormap(colors_prabhakar))
    if "binary2" not in mpl.colormaps:
        mpl.colormaps.register(name="binary2", cmap=mcolors.ListedColormap(["dimgray", "gainsboro"]))
except Exception:
    pass

LINE_STYLES = [
    ('solid', '-'),
    ('dashed', '--'),
    ('dashdot', '-.'),
    ('dotted', ':')
]
STYLE_MAP = dict(LINE_STYLES)
REV_STYLE_MAP = {v: k for k, v in LINE_STYLES}


def find_files_from_input(paths_input):
    """
    Given a string path, list of paths, glob pattern, or directory,
    return a sorted list of QVP files matching CFTDs.py conventions or simulation files.
    """
    if isinstance(paths_input, str):
        paths_input = [paths_input]

    files_found = []
    for raw_item in paths_input:
        if not raw_item:
            continue
        item = raw_item.strip()
        clean_item = item.rstrip('/')

        # 1. Glob patterns
        if any(c in item for c in ['*', '?', '[']):
            matched = glob.glob(item)
            files_found.extend([m.rstrip('/') for m in matched])
        # 2. Directly specified .zarr store (which is a directory on disk, not a file)
        elif clean_item.endswith('.zarr'):
            files_found.append(clean_item)
        # 3. Directory containing files (and not a .zarr store itself)
        elif os.path.isdir(item):
            patterns = [
                os.path.join(item, "*allsim*"),
                os.path.join(item, "*allmoms*"),
                os.path.join(item, "*allmom*"),
                os.path.join(item, "*allm*"),
                os.path.join(item, "*.zarr"),
                os.path.join(item, "*.nc"),
                os.path.join(item, "*.h5"),
                os.path.join(item, "*.hdf5"),
            ]
            dir_matched = []
            for pat in patterns:
                m = glob.glob(pat)
                if m:
                    dir_matched.extend([x.rstrip('/') for x in m])
            if dir_matched:
                files_found.extend(dir_matched)
            else:
                # If no pattern matched inside directory, check all entries
                entries = glob.glob(os.path.join(item, "*"))
                for entry in entries:
                    c_entry = entry.rstrip('/')
                    if c_entry.endswith('.zarr') or c_entry.endswith('.nc') or any(k in c_entry for k in ['allm', 'allsim']):
                        files_found.append(c_entry)
        # 4. Directly specified file
        elif os.path.isfile(item):
            # If path ends in ML_detected.txt, look for allmoms/allsim in same directory
            if os.path.basename(item) == "ML_detected.txt":
                m = glob.glob(os.path.join(os.path.dirname(item), "*allm*")) + glob.glob(os.path.join(os.path.dirname(item), "*allsim*"))
                if m:
                    files_found.extend([x.rstrip('/') for x in m])
                else:
                    files_found.append(item)
            else:
                files_found.append(item)
        # 5. Direct fallback: path passed directly (e.g. automount path or file not yet mounted)
        else:
            files_found.append(clean_item)

    # Filter ML_detected.txt if resolved to actual data files
    filtered = []
    for f in sorted(list(set(files_found))):
        c_f = f.rstrip('/')
        if os.path.basename(c_f) == "ML_detected.txt":
            m = glob.glob(os.path.join(os.path.dirname(c_f), "*allm*")) + glob.glob(os.path.join(os.path.dirname(c_f), "*allsim*"))
            if m:
                filtered.extend([x.rstrip('/') for x in m])
            else:
                filtered.append(c_f)
        else:
            filtered.append(c_f)

    # Direct fallback: if still nothing found, use the raw paths provided directly
    if not filtered:
        for raw_item in paths_input:
            if raw_item and raw_item.strip():
                filtered.append(raw_item.strip().rstrip('/'))

    return sorted(list(set(filtered)))


def extract_date_from_file(file_path):
    """
    Extract YYYY-MM-DD string from filename or parent folder path.
    """
    # Look for standard YYYY-MM-DD anywhere in the full path
    match = re.search(r"(\d{4}-\d{2}-\d{2})", file_path)
    if match:
        return match.group(1)
    # Look for YYYYMMDD in basename
    match = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(file_path))
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    # Look for YYYYMMDD anywhere in path
    match = re.search(r"(\d{4})(\d{2})(\d{2})", file_path)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


KNOWN_LOCATIONS = ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs']
DWD_STATION_IDS = {
    '10392': 'pro',
    '010392': 'pro',
    '10780': 'tur',
    '010780': 'tur',
    '10629': 'umd',
    '010629': 'umd',
}


def extract_radar_location(fpath_or_ds):
    """
    Extract 3-letter radar site location (e.g. 'pro', 'gzt', 'hty', 'ank', 'afy', 'svs', 'tur', 'umd')
    from a QVP file path, dataset, or filename.
    """
    if isinstance(fpath_or_ds, str):
        clean = fpath_or_ds.lower()
        # Check DWD station IDs
        for sid, loc_name in DWD_STATION_IDS.items():
            if sid in clean:
                return loc_name
        # Check path tokens
        parts = re.split(r'[/\\_\-]', clean)
        for loc in KNOWN_LOCATIONS:
            if loc in parts:
                return loc
        for loc in KNOWN_LOCATIONS:
            if re.search(r'[/_\-\.]' + loc + r'[/_\-\.]', clean):
                return loc
        for loc in KNOWN_LOCATIONS:
            if loc in clean:
                return loc

    if hasattr(fpath_or_ds, 'attrs'):
        for attr_k in ['radar', 'radar_id', 'location', 'site', 'station_id']:
            if attr_k in fpath_or_ds.attrs:
                val = str(fpath_or_ds.attrs[attr_k]).lower()
                for sid, loc_name in DWD_STATION_IDS.items():
                    if sid in val:
                        return loc_name
                for loc in KNOWN_LOCATIONS:
                    if loc in val:
                        return loc

    return None


def find_riming_file(loc, is_dmi=False, riming_dir=DEFAULT_RIMING_DIR):
    """
    Locate the appropriate riming NetCDF file and variable name for a given location and radar network,
    matching the decision logic in plot_ppis_qvps_etc.py.
    """
    if not loc or not os.path.isdir(riming_dir):
        return None, None

    if is_dmi or loc in ['afy', 'ank', 'gzt', 'hty', 'svs']:
        candidates = [
            'riming_ZDR_EC_OC_WRC_AC_DBZH_AC',
            'riming_DR_ZDR_EC_OC_WRC_AC_DBZH_AC',
            'riming_DR_ZDR_EC_OC_AC_DBZH_AC',
            'riming_ZDR_EC_OC_AC_DBZH_AC',
            'riming_DR',
            'riming_ZDR_DBZH',
            'riming_UDR'
        ]
    else:
        candidates = [
            'riming_DR_ZDR_EC_OC_AC_DBZH_AC',
            'riming_ZDR_EC_OC_AC_DBZH_AC',
            'riming_DR_ZDR_EC_OC_WRC_AC_DBZH_AC',
            'riming_ZDR_EC_OC_WRC_AC_DBZH_AC',
            'riming_DR',
            'riming_ZDR_DBZH',
            'riming_UDR'
        ]

    for c in candidates:
        fp = os.path.join(riming_dir, f"{loc}_{c}.nc")
        if os.path.exists(fp):
            return fp, c

    matches = glob.glob(os.path.join(riming_dir, f"{loc}_*.nc"))
    if matches:
        return matches[0], None

    return None, None


class QVPDatasetManager:
    """
    Manages loading, caching, date extraction, and dimension alignment for QVP datasets.
    Supports:
      - Multiple 1-day files (switched via date slider)
      - Single file containing 1 or multiple days
      - Automatic dimension standardization to 'time' and 'z'
    """
    def __init__(self, file_list, min_height=0.0, max_height=12000.0):
        self.file_list = sorted(file_list)
        self.min_height = min_height
        self.max_height = max_height
        self.cache = {}  # date_str -> xr.Dataset
        self.is_single_file_multi_day = False
        self.single_file_ds = None
        self.date_to_file = {}
        self.date_list = []
        self._init_date_mapping()

    def _open_raw(self, fpath):
        """
        Open a QVP data file into memory.
        Supports:
          1. utils.load_qvps (if available, as done in CFTDs.py)
          2. Icechunk zarr repository (via Repository.open and session.store)
          3. Standard Zarr store (via xr.open_zarr)
          4. NetCDF / HDF5 files (via xr.open_dataset)
        """
        clean_path = fpath.rstrip('/')

        # 1. Try utils.load_qvps if available (as used in CFTDs.py)
        if utils is not None and hasattr(utils, 'load_qvps'):
            try:
                ds = utils.load_qvps([clean_path], load=True)
                if ds is not None:
                    return ds
            except Exception:
                pass

        # 2. Check for .zarr (could be Icechunk or standard Zarr)
        if clean_path.endswith('.zarr'):
            # Try Icechunk first (standard storage for DWD/DMI radar QVP processing)
            if HAS_ICECHUNK:
                try:
                    storage = local_filesystem_storage(clean_path)
                    repo = Repository.open(storage)
                    session = repo.readonly_session("main")
                    ds = xr.open_zarr(session.store, consolidated=False)
                    return ds.load()
                except Exception:
                    pass

            # Try standard Zarr (consolidated=False, consolidated=True, default)
            for cons in [False, True, None]:
                try:
                    if cons is None:
                        ds = xr.open_zarr(clean_path)
                    else:
                        ds = xr.open_zarr(clean_path, consolidated=cons)
                    return ds.load()
                except Exception:
                    pass

        # 3. NetCDF / HDF5 files
        for eng in ['h5netcdf', 'netcdf4', None]:
            try:
                if eng is None:
                    ds = xr.open_dataset(clean_path)
                else:
                    ds = xr.open_dataset(clean_path, engine=eng)
                return ds.load()
            except Exception:
                pass

        # 4. Final attempt with generic open_zarr and open_dataset
        try:
            return xr.open_zarr(clean_path).load()
        except Exception:
            pass
        return xr.open_dataset(clean_path).load()

    def _standardize_ds(self, ds):
        # Standardize vertical dimension to 'z'
        vert_candidates = ['height', 'range', 'altitude', 'alt', 'height_above_ground', 'height_above_sea_level', 'nheight', 'z_levels', 'level']
        for vert_name in vert_candidates:
            if 'z' not in ds.dims and vert_name in ds.dims:
                ds = ds.rename({vert_name: 'z'})
            if 'z' not in ds.coords and vert_name in ds.coords:
                ds = ds.assign_coords(z=ds[vert_name])

        # Standardize time dimension to 'time'
        time_candidates = ['Time', 'datetime', 'valid_time', 'Times', 't']
        for t_name in time_candidates:
            if 'time' not in ds.dims and t_name in ds.dims:
                ds = ds.rename({t_name: 'time'})
            if 'time' not in ds.coords and t_name in ds.coords:
                ds = ds.assign_coords(time=ds[t_name])

        # Drop NaT timestamps if any
        if 'time' in ds.coords:
            valid_t = ds['time'].notnull().values
            if not valid_t.all():
                ds = ds.isel(time=valid_t)
        return ds

    def _init_date_mapping(self):
        if not self.file_list:
            return

        # If only one file, check if it contains multiple dates
        if len(self.file_list) == 1:
            try:
                ds_single = self._open_raw(self.file_list[0])
                ds_single = self._standardize_ds(ds_single)
                if 'time' in ds_single:
                    times = pd.to_datetime(ds_single.time.values)
                    unique_dates = sorted(list(set(str(t.date()) for t in times if pd.notnull(t))))
                    if len(unique_dates) > 1:
                        self.is_single_file_multi_day = True
                        self.single_file_ds = ds_single
                        self.date_list = unique_dates
                        return
                    elif len(unique_dates) == 1:
                        d_str = unique_dates[0]
                        self.date_list = [d_str]
                        self.date_to_file = {d_str: self.file_list[0]}
                        self.cache[d_str] = ds_single
                        return
            except Exception:
                pass

        date_to_files = {}
        for f in self.file_list:
            d_str = extract_date_from_file(f)
            if not d_str:
                # Open lazily to read time[0]
                try:
                    ds_meta = self._open_raw(f)
                    if 'time' in ds_meta:
                        d_str = str(pd.to_datetime(ds_meta.time.values[0]).date())
                except Exception:
                    d_str = os.path.basename(f)
            if d_str:
                date_to_files.setdefault(d_str, []).append(f)

        self.date_list = sorted(date_to_files.keys())
        self.date_to_file = {d: date_to_files[d][0] for d in self.date_list}

    def load_day(self, date_str):
        if date_str in self.cache:
            return self.cache[date_str]

        if self.is_single_file_multi_day:
            ds = self.single_file_ds.sel(time=date_str)
            self.cache[date_str] = ds
            return ds

        fpath = self.date_to_file.get(date_str, None)
        if not fpath:
            raise ValueError(f"No file found for date {date_str}")

        # Load dataset
        ds = self._open_raw(fpath)
        ds = self._standardize_ds(ds)

        # Cache dataset (keep reasonable cache size to save memory)
        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[date_str] = ds
        return ds

    def get_available_variables(self, ds):
        """
        Get all 2D data variables that have dimensions ('time', 'z') or ('z', 'time').
        """
        valid_vars = []
        for v in ds.data_vars:
            dims = ds[v].dims
            if 'time' in dims and 'z' in dims and (len(dims) == 2 or (len(dims) == 3 and ds[v].size == ds.dims['time'] * ds.dims['z'])):
                valid_vars.append(v)
        if not valid_vars:
            for v in ds.data_vars:
                if len(ds[v].dims) == 2:
                    valid_vars.append(v)
        return sorted(valid_vars)

    def get_z_coords(self, ds):
        for k in ['z', 'height', 'range', 'altitude', 'alt', 'height_above_ground', 'height_above_sea_level']:
            if k in ds.coords:
                z_vals = ds[k].values
                break
        else:
            if 'z' in ds.dims:
                z_vals = np.arange(ds.dims['z'])
            else:
                z_vals = np.arange(100)
        # If z is in kilometers (max < 50), convert to meters
        if np.nanmax(z_vals) < 50.0:
            z_vals = z_vals * 1000.0
        return z_vals

    def get_default_variables(self, available_vars):
        """
        Picks the 4 default variables specified by the user:
          Panel 1: DBZH_AC
          Panel 2: ZDR_EC_OC_WRC_AC or ZDR_EC_OC_AC
          Panel 3: KDP_ML_corrected_EC
          Panel 4: RHOHV_NC
        Falls back cleanly to related variables if not present.
        """
        p1_candidates = ['DBZH_AC', 'DBZH', 'DBTH_OC', 'DBTH', 'ZH', 'TH']
        p2_candidates = [
            'ZDR_EC_OC_WRC_AC', 'ZDR_EC_OC_AC', 'ZDR_OC_AC', 'ZDR_AC',
            'ZDR_EC_OC_WRC', 'ZDR_EC_OC', 'ZDR_OC', 'ZDR_EC', 'ZDR'
        ]
        p3_candidates = ['KDP_ML_corrected_EC', 'KDP_ML_corrected', 'KDP_CONV', 'KDP']
        p4_candidates = ['RHOHV_NC', 'RHOHV', 'URHOHV']

        def pick_first(candidates, fallback_idx=0):
            for c in candidates:
                if c in available_vars:
                    return c
            return available_vars[fallback_idx % len(available_vars)] if available_vars else "DBZH"

        v1 = pick_first(p1_candidates, 0)
        v2 = pick_first(p2_candidates, 1)
        v3 = pick_first(p3_candidates, 2)
        v4 = pick_first(p4_candidates, 3)
        return [v1, v2, v3, v4]


def get_var_vis_info(var_name, data_array=None):
    """
    Resolve colormap, discrete ticks, norm, label, and units from radarmet.visdict14
    or intelligent defaults.
    """
    vinfo = None
    if radarmet is not None and hasattr(radarmet, 'visdict14'):
        # Try direct name, then prefix, then radar family
        if var_name in radarmet.visdict14:
            vinfo = radarmet.visdict14[var_name]
        else:
            base_prefix = var_name.split("_")[0]
            if base_prefix in radarmet.visdict14:
                vinfo = radarmet.visdict14[base_prefix]
            elif "ZDR" in var_name and "ZDR" in radarmet.visdict14:
                vinfo = radarmet.visdict14["ZDR"]
            elif "KDP" in var_name and "KDP" in radarmet.visdict14:
                vinfo = radarmet.visdict14["KDP"]
            elif "RHOHV" in var_name and "RHOHV" in radarmet.visdict14:
                vinfo = radarmet.visdict14["RHOHV"]
            elif "PHIDP" in var_name and "PHIDP" in radarmet.visdict14:
                vinfo = radarmet.visdict14["PHIDP"]
            elif ("DBZH" in var_name or "DBTH" in var_name) and "DBZH" in radarmet.visdict14:
                vinfo = radarmet.visdict14["DBZH"]

    ticks = None
    cmap_obj = None
    cmap_name = "viridis"

    if vinfo is not None:
        ticks = vinfo.get("ticks", None)
        raw_cmap = vinfo.get("cmap", "viridis")
        if isinstance(raw_cmap, str):
            cmap_name = raw_cmap
            try:
                cmap_obj = mpl.colormaps.get_cmap(raw_cmap)
            except Exception:
                try:
                    cmap_obj = plt.get_cmap(raw_cmap)
                except Exception:
                    cmap_obj = mpl.colormaps.get_cmap("viridis")
                    cmap_name = "viridis"
        elif isinstance(raw_cmap, mpl.colors.Colormap):
            cmap_obj = raw_cmap
            cmap_name = getattr(raw_cmap, "name", "custom")

    if cmap_obj is None:
        cmap_name = "viridis"
        cmap_obj = mpl.colormaps.get_cmap("viridis")

    # Determine default vmin and vmax
    if ticks is not None and len(ticks) > 1:
        vmin = float(ticks[0])
        vmax = float(ticks[-1])
        # Use discrete colormap & norm if utils available
        if utils is not None and hasattr(utils, 'get_discrete_cmap'):
            try:
                cmap_obj = utils.get_discrete_cmap(ticks, cmap_obj)
                norm = utils.get_discrete_norm(ticks, cmap_obj)
            except Exception:
                norm = mpl.colors.BoundaryNorm(ticks, ncolors=len(ticks) - 1, extend='both')
        else:
            norm = mpl.colors.BoundaryNorm(ticks, ncolors=len(ticks) - 1, extend='both')
    else:
        # Fallback to dynamic data range
        if data_array is not None:
            vals = np.asarray(data_array)
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                vmin = float(np.percentile(valid, 2))
                vmax = float(np.percentile(valid, 98))
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Resolve units
    units = ""
    if data_array is not None and hasattr(data_array, 'attrs'):
        units = data_array.attrs.get("units", "")
    if not units:
        v_upper = var_name.upper()
        if any(z in v_upper for z in ["DBZH", "DBTH", "ZH", "TH"]):
            units = "dBZ"
        elif "ZDR" in v_upper:
            units = "dB"
        elif "KDP" in v_upper:
            units = "°/km"
        elif "RHOHV" in v_upper:
            units = ""
        elif "PHIDP" in v_upper or "UPHIDP" in v_upper or "PHI" in v_upper:
            units = "°"
        elif "VRADH" in v_upper or "WRADH" in v_upper:
            units = "m/s"
        elif "LWC" in v_upper or "IWC" in v_upper:
            units = "g/m³"

    return {
        "ticks": ticks,
        "cmap": cmap_obj,
        "cmap_name": cmap_name,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "units": units
    }


# ==============================================================================
# GUI Implementation using Tkinter + Matplotlib
# ==============================================================================

class QVPInteractiveAppTk:
    """
    Modern desktop GUI integrating Matplotlib visualization with Tkinter controls.
    """
    def __init__(self, dataset_mgr, initial_date=None):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk

        self.mgr = dataset_mgr
        self.dates = self.mgr.date_list
        if not self.dates:
            raise ValueError("No valid QVP files/dates found to visualize.")

        self.current_date = initial_date if initial_date in self.dates else self.dates[0]
        self.current_ds = self.mgr.load_day(self.current_date)
        self.all_vars = self.mgr.get_available_variables(self.current_ds)
        self.min_height = getattr(self.mgr, 'min_height', 0.0)
        self.max_height = getattr(self.mgr, 'max_height', 12000.0)

        # Panel settings for Column 1 (4 panels)
        defaults = self.mgr.get_default_variables(self.all_vars)
        self.col1_settings = []
        for i in range(4):
            vname = defaults[i]
            vis = get_var_vis_info(vname, self.current_ds[vname])
            self.col1_settings.append({
                'var': vname,
                'cmap_name': vis['cmap_name'],
                'vmin': vis['vmin'],
                'vmax': vis['vmax'],
                'vis': vis
            })

        # Selected 1D profile index and variables for Column 2
        self.current_time_idx = 0
        self.active_1d_vars = list(defaults)  # Start with 4 column 1 variables
        self.var_styles = {}  # var -> dict(color, linestyle, xmin, xmax)
        self._init_var_styles()

        # Build UI
        self.root = self.tk.Tk()
        self.root.title("Interactive QVP Visualizer")
        self.root.geometry("1600x950")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.vlines = []
        self.col1_axes = []
        self.col1_caxes = []
        self.cbars = [None] * 4
        self.twin_axes = []
        self.show_entropy = True
        self.show_riming = True
        self.show_ml = True
        self.col1_entropy_artists = [None] * 4
        self.col1_riming_artists = [None] * 4
        self.col1_ml_artists = [None] * 4
        self.riming_ds_cache = {}
        self.radar_location = None

        self._build_widgets()
        self.plot_all()

    def _init_var_styles(self):
        for idx, v in enumerate(self.all_vars):
            vis = get_var_vis_info(v, self.current_ds[v])
            color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            style = LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)][1]
            self.var_styles[v] = {
                'color': color,
                'linestyle': style,
                'xmin': vis['vmin'],
                'xmax': vis['vmax'],
                'units': vis['units']
            }

    def _build_widgets(self):
        # Main vertical container
        main_frame = self.ttk.Frame(self.root)
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # Top Control Bar (Date slider & Step buttons & Height Range)
        top_bar = self.ttk.LabelFrame(main_frame, text="Navigation & Settings", padding=(8, 4))
        top_bar.pack(side=self.tk.TOP, fill=self.tk.X, padx=6, pady=4)

        self.btn_prev_date = self.ttk.Button(top_bar, text="◀ Previous Day", command=self.prev_date)
        self.btn_prev_date.pack(side=self.tk.LEFT, padx=4)

        self.date_var_label = self.tk.StringVar(value=f"Date: {self.current_date} (1 of {len(self.dates)})")
        lbl_date = self.ttk.Label(top_bar, textvariable=self.date_var_label, font=("Helvetica", 10, "bold"), width=30)
        lbl_date.pack(side=self.tk.LEFT, padx=6)

        self.date_scale_val = self.tk.DoubleVar(value=self.dates.index(self.current_date))
        self.date_scale = self.ttk.Scale(
            top_bar, from_=0, to=max(1, len(self.dates) - 1),
            orient=self.tk.HORIZONTAL, variable=self.date_scale_val,
            command=self.on_date_slider_move
        )
        self.date_scale.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=8)

        self.btn_next_date = self.ttk.Button(top_bar, text="Next Day ▶", command=self.next_date)
        self.btn_next_date.pack(side=self.tk.LEFT, padx=4)

        if len(self.dates) <= 1:
            self.date_scale.configure(state="disabled")
            self.btn_prev_date.configure(state="disabled")
            self.btn_next_date.configure(state="disabled")

        # Height range controls directly on top bar
        h_bar = self.ttk.Frame(top_bar)
        h_bar.pack(side=self.tk.RIGHT, padx=6)

        self.ttk.Label(h_bar, text="Height [m]:", font=("Helvetica", 9, "bold")).pack(side=self.tk.LEFT, padx=(6, 2))
        self.ent_min_h = self.ttk.Entry(h_bar, width=6)
        self.ent_min_h.insert(0, f"{self.min_height:.0f}")
        self.ent_min_h.pack(side=self.tk.LEFT, padx=2)

        self.ttk.Label(h_bar, text="to").pack(side=self.tk.LEFT, padx=2)
        self.ent_max_h = self.ttk.Entry(h_bar, width=6)
        self.ent_max_h.insert(0, f"{self.max_height:.0f}")
        self.ent_max_h.pack(side=self.tk.LEFT, padx=2)

        self.btn_apply_h = self.ttk.Button(h_bar, text="Apply", command=self.apply_height_range)
        self.btn_apply_h.pack(side=self.tk.LEFT, padx=4)

        self.btn_entropy = self.ttk.Button(h_bar, text="Entropy: ON", command=self.toggle_entropy)
        self.btn_entropy.pack(side=self.tk.LEFT, padx=3)

        self.btn_riming = self.ttk.Button(h_bar, text="Riming: ON", command=self.toggle_riming)
        self.btn_riming.pack(side=self.tk.LEFT, padx=3)

        self.btn_ml = self.ttk.Button(h_bar, text="ML: ON", command=self.toggle_ml)
        self.btn_ml.pack(side=self.tk.LEFT, padx=3)

        self.ent_min_h.bind("<Return>", lambda e: self.apply_height_range())
        self.ent_max_h.bind("<Return>", lambda e: self.apply_height_range())

        # Content Frame: Split between Matplotlib Canvas (Left) and Sidebar Notebook (Right)
        content_frame = self.ttk.Frame(main_frame)
        content_frame.pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)

        # Right-side notebook for controls
        self.sidebar = self.ttk.Notebook(content_frame, width=420)
        self.sidebar.pack(side=self.tk.RIGHT, fill=self.tk.Y, padx=4, pady=4)

        # Matplotlib Area (Left)
        plot_frame = self.ttk.Frame(content_frame)
        plot_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        # Create Matplotlib Figure
        self.fig = mpl.figure.Figure(figsize=(12, 9), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')

        # Main GridSpec: 1 row, 2 columns (Left: 2D panels + dedicated colorbars, Right: 1D profile)
        self.gs_main = self.fig.add_gridspec(
            1, 2,
            width_ratios=[1.5, 1.0],
            left=0.07, right=0.96,
            bottom=0.08, top=0.95,
            wspace=0.28
        )

        # Left Column: 4 rows x 2 cols (Heatmap, dedicated Colorbar)
        self.gs_left = GridSpecFromSubplotSpec(
            4, 2,
            subplot_spec=self.gs_main[0, 0],
            width_ratios=[1.0, 0.035],
            wspace=0.04,
            hspace=0.24
        )

        self.col1_axes = []
        self.col1_caxes = [None] * 4
        for i in range(4):
            ax = self.fig.add_subplot(self.gs_left[i, 0], sharex=self.col1_axes[0] if i > 0 else None)
            self.col1_axes.append(ax)

        # Right Column: 1D QVP profile
        self.gs_right = GridSpecFromSubplotSpec(
            1, 1,
            subplot_spec=self.gs_main[0, 1]
        )
        orig_bbox = self.gs_right[0, 0].get_position(self.fig)
        self.ax_prof = self.fig.add_axes(orig_bbox)

        # Canvas & Toolbar
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)

        self.toolbar = self.NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # Connect click & resize events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('resize_event', lambda e: self.update_column2_profile())

        # Build Sidebar Tabs
        self._build_sidebar_tab1_panels()
        self._build_sidebar_tab2_variables()
        self._build_sidebar_tab3_styles()

    def _build_sidebar_tab1_panels(self):
        tab = self.ttk.Frame(self.sidebar, padding=6)
        self.sidebar.add(tab, text="2D QVP Panels (1-4)")

        # Scrollable container for panel configs
        c = self.tk.Canvas(tab, borderwidth=0, highlightthickness=0)
        sb = self.ttk.Scrollbar(tab, orient=self.tk.VERTICAL, command=c.yview)
        subframe = self.ttk.Frame(c)
        subframe.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0, 0), window=subframe, anchor="nw")
        c.configure(yscrollcommand=sb.set)

        c.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        sb.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self.panel_ui_elements = []

        # Global Height Range Card
        card_h = self.ttk.LabelFrame(subframe, text="Height Range (y-axis)", padding=6)
        card_h.pack(fill=self.tk.X, expand=True, pady=4, padx=2)

        row_h1 = self.ttk.Frame(card_h)
        row_h1.pack(fill=self.tk.X, pady=2)
        self.ttk.Label(row_h1, text="Min (m):", width=8).pack(side=self.tk.LEFT)
        self.ent_min_tab1 = self.ttk.Entry(row_h1, width=8)
        self.ent_min_tab1.insert(0, f"{self.min_height:.0f}")
        self.ent_min_tab1.pack(side=self.tk.LEFT, padx=2)

        self.ttk.Label(row_h1, text="Max (m):", width=8).pack(side=self.tk.LEFT, padx=(6, 0))
        self.ent_max_tab1 = self.ttk.Entry(row_h1, width=8)
        self.ent_max_tab1.insert(0, f"{self.max_height:.0f}")
        self.ent_max_tab1.pack(side=self.tk.LEFT, padx=2)

        row_h2 = self.ttk.Frame(card_h)
        row_h2.pack(fill=self.tk.X, pady=(4, 2))
        btn_apply_h_tab = self.ttk.Button(
            row_h2, text="Apply Range",
            command=lambda: self.apply_height_range(
                float(self.ent_min_tab1.get()), float(self.ent_max_tab1.get())
            )
        )
        btn_apply_h_tab.pack(side=self.tk.LEFT, padx=2)

        btn_reset_h = self.ttk.Button(
            row_h2, text="Reset (0 - 12000m)",
            command=lambda: self.apply_height_range(0.0, 12000.0)
        )
        btn_reset_h.pack(side=self.tk.LEFT, padx=4)

        row_h3 = self.ttk.Frame(card_h)
        row_h3.pack(fill=self.tk.X, pady=(4, 2))
        self.btn_entropy_tab = self.ttk.Button(row_h3, text="Entropy: ON", command=self.toggle_entropy)
        self.btn_entropy_tab.pack(side=self.tk.LEFT, padx=2)
        self.btn_riming_tab = self.ttk.Button(row_h3, text="Riming: ON", command=self.toggle_riming)
        self.btn_riming_tab.pack(side=self.tk.LEFT, padx=2)
        self.btn_ml_tab = self.ttk.Button(row_h3, text="ML: ON", command=self.toggle_ml)
        self.btn_ml_tab.pack(side=self.tk.LEFT, padx=2)

        for i in range(4):
            card = self.ttk.LabelFrame(subframe, text=f"Panel {i+1} Settings (Row {i+1})", padding=6)
            card.pack(fill=self.tk.X, expand=True, pady=4, padx=2)

            # Variable selection
            row1 = self.ttk.Frame(card)
            row1.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(row1, text="Variable:", width=10).pack(side=self.tk.LEFT)
            cb_var = self.ttk.Combobox(row1, values=self.all_vars, state="normal")
            cb_var.set(self.col1_settings[i]['var'])
            cb_var.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)

            # Colormap
            row2 = self.ttk.Frame(card)
            row2.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(row2, text="Colormap:", width=10).pack(side=self.tk.LEFT)
            cb_cmap = self.ttk.Combobox(row2, values=COLORMAP_CHOICES, state="readonly")
            cb_cmap.set(self.col1_settings[i]['cmap_name'])
            cb_cmap.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)

            # Vmin and Vmax
            row3 = self.ttk.Frame(card)
            row3.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(row3, text="vmin:", width=6).pack(side=self.tk.LEFT)
            ent_vmin = self.ttk.Entry(row3, width=8)
            ent_vmin.insert(0, f"{self.col1_settings[i]['vmin']:.2f}")
            ent_vmin.pack(side=self.tk.LEFT, padx=2)

            self.ttk.Label(row3, text="vmax:", width=6).pack(side=self.tk.LEFT, padx=(6, 0))
            ent_vmax = self.ttk.Entry(row3, width=8)
            ent_vmax.insert(0, f"{self.col1_settings[i]['vmax']:.2f}")
            ent_vmax.pack(side=self.tk.LEFT, padx=2)

            # Buttons row
            row4 = self.ttk.Frame(card)
            row4.pack(fill=self.tk.X, pady=(4, 2))

            btn_apply = self.ttk.Button(row4, text="Apply", command=lambda p=i: self.apply_panel_settings(p))
            btn_apply.pack(side=self.tk.LEFT, padx=2)

            btn_reset = self.ttk.Button(row4, text="Reset Defaults", command=lambda p=i: self.reset_panel_defaults(p))
            btn_reset.pack(side=self.tk.LEFT, padx=4)

            self.panel_ui_elements.append({
                'var': cb_var,
                'cmap': cb_cmap,
                'vmin': ent_vmin,
                'vmax': ent_vmax
            })

    def _build_sidebar_tab2_variables(self):
        tab = self.ttk.Frame(self.sidebar, padding=6)
        self.sidebar.add(tab, text="1D Variables (Checkboxes)")

        top_row = self.ttk.Frame(tab)
        top_row.pack(fill=self.tk.X, pady=2)

        self.ttk.Label(top_row, text="Filter:").pack(side=self.tk.LEFT)
        self.filter_var = self.tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._filter_variable_checkboxes())
        ent_filter = self.ttk.Entry(top_row, textvariable=self.filter_var)
        ent_filter.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=4)

        btn_clear_filter = self.ttk.Button(top_row, text="✕", width=3, command=lambda: self.filter_var.set(""))
        btn_clear_filter.pack(side=self.tk.LEFT)

        # Quick action buttons
        btn_bar = self.ttk.Frame(tab)
        btn_bar.pack(fill=self.tk.X, pady=4)

        btn_sync = self.ttk.Button(btn_bar, text="Sync with Col 1", command=self.sync_1d_with_col1)
        btn_sync.pack(side=self.tk.LEFT, padx=2)

        btn_all = self.ttk.Button(btn_bar, text="Select All", command=self.select_all_1d_vars)
        btn_all.pack(side=self.tk.LEFT, padx=2)

        btn_none = self.ttk.Button(btn_bar, text="Clear All", command=self.clear_all_1d_vars)
        btn_none.pack(side=self.tk.LEFT, padx=2)

        # Scrollable area for variable checkboxes
        c = self.tk.Canvas(tab, borderwidth=0, highlightthickness=0)
        sb = self.ttk.Scrollbar(tab, orient=self.tk.VERTICAL, command=c.yview)
        self.checkbox_container = self.ttk.Frame(c)
        self.checkbox_container.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0, 0), window=self.checkbox_container, anchor="nw")
        c.configure(yscrollcommand=sb.set)

        c.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        sb.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self.check_vars = {}
        self.check_widgets = {}
        for v in self.all_vars:
            var_state = self.tk.BooleanVar(value=(v in self.active_1d_vars))
            cb = self.ttk.Checkbutton(
                self.checkbox_container,
                text=v,
                variable=var_state,
                command=lambda vname=v: self.on_var_checkbox_toggle(vname)
            )
            cb.pack(anchor="w", pady=1, padx=4)
            self.check_vars[v] = var_state
            self.check_widgets[v] = cb

    def _filter_variable_checkboxes(self):
        query = self.filter_var.get().strip().lower()
        for v, cb in self.check_widgets.items():
            if not query or query in v.lower():
                cb.pack(anchor="w", pady=1, padx=4)
            else:
                cb.pack_forget()

    def _build_sidebar_tab3_styles(self):
        self.tab_styles = self.ttk.Frame(self.sidebar, padding=6)
        self.sidebar.add(self.tab_styles, text="1D Styles & Ranges")

        header = self.ttk.Frame(self.tab_styles)
        header.pack(fill=self.tk.X, pady=(0, 4))
        self.ttk.Label(
            header,
            text="Customize line color, style, and axis limits for active profiles:",
            wraplength=380
        ).pack(anchor="w")

        # Scrollable container for active variables styles
        c = self.tk.Canvas(self.tab_styles, borderwidth=0, highlightthickness=0)
        sb = self.ttk.Scrollbar(self.tab_styles, orient=self.tk.VERTICAL, command=c.yview)
        self.styles_container = self.ttk.Frame(c)
        self.styles_container.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0, 0), window=self.styles_container, anchor="nw")
        c.configure(yscrollcommand=sb.set)

        c.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        sb.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self._refresh_styles_tab()

    def _refresh_styles_tab(self):
        # Clear previous children
        for widget in self.styles_container.winfo_children():
            widget.destroy()

        if not self.active_1d_vars:
            lbl = self.ttk.Label(self.styles_container, text="No variables currently active in 1D profile.")
            lbl.pack(pady=10)
            return

        self.style_entries = {}
        for v in self.active_1d_vars:
            st = self.var_styles[v]
            card = self.ttk.LabelFrame(self.styles_container, text=v, padding=4)
            card.pack(fill=self.tk.X, expand=True, pady=4, padx=2)

            r1 = self.ttk.Frame(card)
            r1.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(r1, text="Color:", width=8).pack(side=self.tk.LEFT)
            cb_col = self.ttk.Combobox(r1, values=DEFAULT_PALETTE, width=12)
            cb_col.set(st['color'])
            cb_col.pack(side=self.tk.LEFT, padx=2)

            self.ttk.Label(r1, text="Style:", width=6).pack(side=self.tk.LEFT, padx=(4, 0))
            cb_style = self.ttk.Combobox(r1, values=[k for k, _ in LINE_STYLES], width=10, state="readonly")
            cb_style.set(REV_STYLE_MAP.get(st['linestyle'], 'solid'))
            cb_style.pack(side=self.tk.LEFT, padx=2)

            r2 = self.ttk.Frame(card)
            r2.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(r2, text="Min:", width=6).pack(side=self.tk.LEFT)
            ent_min = self.ttk.Entry(r2, width=8)
            ent_min.insert(0, f"{st['xmin']:.2f}")
            ent_min.pack(side=self.tk.LEFT, padx=2)

            self.ttk.Label(r2, text="Max:", width=6).pack(side=self.tk.LEFT, padx=(4, 0))
            ent_max = self.ttk.Entry(r2, width=8)
            ent_max.insert(0, f"{st['xmax']:.2f}")
            ent_max.pack(side=self.tk.LEFT, padx=2)

            btn_auto = self.ttk.Button(r2, text="Auto", width=5, command=lambda vname=v: self.auto_range_var(vname))
            btn_auto.pack(side=self.tk.LEFT, padx=4)

            self.style_entries[v] = {
                'color': cb_col,
                'style': cb_style,
                'xmin': ent_min,
                'xmax': ent_max
            }

        btn_apply_all = self.ttk.Button(
            self.styles_container,
            text="Apply All Style & Range Changes",
            command=self.apply_all_styles
        )
        btn_apply_all.pack(fill=self.tk.X, pady=8, padx=2)

    def apply_all_styles(self):
        for v, entries in self.style_entries.items():
            col = entries['color'].get().strip()
            st_name = entries['style'].get().strip()
            style_code = STYLE_MAP.get(st_name, '-')
            try:
                xmin = float(entries['xmin'].get().strip())
                xmax = float(entries['xmax'].get().strip())
                if xmin < xmax:
                    self.var_styles[v]['xmin'] = xmin
                    self.var_styles[v]['xmax'] = xmax
            except ValueError:
                pass
            self.var_styles[v]['color'] = col
            self.var_styles[v]['linestyle'] = style_code

        self.update_column2_profile()

    def auto_range_var(self, var_name):
        ds_prof = self.current_ds.isel(time=self.current_time_idx)
        vals = ds_prof[var_name].values
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin, vmax = 0.0, 1.0

        self.var_styles[var_name]['xmin'] = vmin
        self.var_styles[var_name]['xmax'] = vmax
        if var_name in self.style_entries:
            self.style_entries[var_name]['xmin'].delete(0, self.tk.END)
            self.style_entries[var_name]['xmin'].insert(0, f"{vmin:.2f}")
            self.style_entries[var_name]['xmax'].delete(0, self.tk.END)
            self.style_entries[var_name]['xmax'].insert(0, f"{vmax:.2f}")
        self.update_column2_profile()

    # --- Actions and Event Handlers ---

    def on_date_slider_move(self, val):
        idx = int(round(float(val)))
        if 0 <= idx < len(self.dates):
            target_date = self.dates[idx]
            if target_date != self.current_date:
                self.switch_date(target_date)

    def prev_date(self):
        idx = self.dates.index(self.current_date)
        if idx > 0:
            self.switch_date(self.dates[idx - 1])

    def next_date(self):
        idx = self.dates.index(self.current_date)
        if idx < len(self.dates) - 1:
            self.switch_date(self.dates[idx + 1])

    def switch_date(self, target_date):
        self.current_date = target_date
        self.date_scale_val.set(self.dates.index(target_date))
        self.date_var_label.set(f"Date: {self.current_date} ({self.dates.index(target_date)+1} of {len(self.dates)})")
        self.current_ds = self.mgr.load_day(self.current_date)
        self.all_vars = self.mgr.get_available_variables(self.current_ds)
        # Constrain time index
        if self.current_time_idx >= len(self.current_ds.time):
            self.current_time_idx = max(0, len(self.current_ds.time) - 1)
        self.plot_all()

    def apply_panel_settings(self, panel_idx):
        ui = self.panel_ui_elements[panel_idx]
        vname = ui['var'].get().strip()
        cmap_name = ui['cmap'].get().strip()
        try:
            vmin = float(ui['vmin'].get().strip())
            vmax = float(ui['vmax'].get().strip())
        except ValueError:
            self.messagebox.showerror("Invalid Input", "vmin and vmax must be numeric.")
            return

        if vmin >= vmax:
            self.messagebox.showerror("Invalid Range", "vmin must be strictly less than vmax.")
            return

        if vname not in self.all_vars:
            self.messagebox.showerror("Invalid Variable", f"Variable '{vname}' not in dataset.")
            return

        vis = get_var_vis_info(vname, self.current_ds[vname])
        if cmap_name != "default (visdict)":
            try:
                vis['cmap'] = mpl.colormaps.get_cmap(cmap_name)
                vis['cmap_name'] = cmap_name
            except Exception:
                pass
        vis['vmin'] = vmin
        vis['vmax'] = vmax
        vis['norm'] = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        self.col1_settings[panel_idx] = {
            'var': vname,
            'cmap_name': cmap_name,
            'vmin': vmin,
            'vmax': vmax,
            'vis': vis
        }

        # Automatically ensure new variable is displayed in 1D profile
        if vname not in self.active_1d_vars:
            self.active_1d_vars.append(vname)
            if vname in self.check_vars:
                self.check_vars[vname].set(True)
            self._refresh_styles_tab()

        self.update_column1_panel(panel_idx)
        self.update_column2_profile()

    def reset_panel_defaults(self, panel_idx):
        vname = self.col1_settings[panel_idx]['var']
        vis = get_var_vis_info(vname, self.current_ds[vname])
        self.col1_settings[panel_idx] = {
            'var': vname,
            'cmap_name': vis['cmap_name'],
            'vmin': vis['vmin'],
            'vmax': vis['vmax'],
            'vis': vis
        }
        ui = self.panel_ui_elements[panel_idx]
        ui['cmap'].set(vis['cmap_name'])
        ui['vmin'].delete(0, self.tk.END)
        ui['vmin'].insert(0, f"{vis['vmin']:.2f}")
        ui['vmax'].delete(0, self.tk.END)
        ui['vmax'].insert(0, f"{vis['vmax']:.2f}")
        self.update_column1_panel(panel_idx)

    def on_var_checkbox_toggle(self, var_name):
        is_checked = self.check_vars[var_name].get()
        if is_checked and var_name not in self.active_1d_vars:
            self.active_1d_vars.append(var_name)
        elif not is_checked and var_name in self.active_1d_vars:
            self.active_1d_vars.remove(var_name)
        self._refresh_styles_tab()
        self.update_column2_profile()

    def sync_1d_with_col1(self):
        col1_v = [s['var'] for s in self.col1_settings]
        for v in self.all_vars:
            state = (v in col1_v)
            self.check_vars[v].set(state)
        self.active_1d_vars = list(col1_v)
        self._refresh_styles_tab()
        self.update_column2_profile()

    def select_all_1d_vars(self):
        for v in self.all_vars:
            self.check_vars[v].set(True)
        self.active_1d_vars = list(self.all_vars)
        self._refresh_styles_tab()
        self.update_column2_profile()

    def clear_all_1d_vars(self):
        for v in self.all_vars:
            self.check_vars[v].set(False)
        self.active_1d_vars.clear()
        self._refresh_styles_tab()
        self.update_column2_profile()

    def on_canvas_click(self, event):
        if getattr(event, 'button', 1) != 1:
            return
        if hasattr(self, 'toolbar') and getattr(self.toolbar, 'mode', '') != '':
            return
        if event.inaxes in self.col1_axes and event.xdata is not None:
            time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
            idx = int(np.argmin(np.abs(time_nums - event.xdata)))
            self.set_selected_profile_index(idx)

    def set_selected_profile_index(self, idx):
        self.current_time_idx = idx
        self._update_profile_indicator_lines()
        self.update_column2_profile()

    def apply_height_range(self, min_val=None, max_val=None):
        """
        Update the vertical (y-axis) range for both Column 1 2D plots and Column 2 1D profile.
        """
        if min_val is None or max_val is None:
            try:
                min_val = float(self.ent_min_h.get().strip())
                max_val = float(self.ent_max_h.get().strip())
            except ValueError:
                self.messagebox.showerror("Invalid Input", "Min and Max height must be numbers.")
                return

        if min_val >= max_val:
            self.messagebox.showerror("Invalid Range", "Min height must be strictly less than Max height.")
            return

        self.min_height = min_val
        self.max_height = max_val

        # Synchronize entry inputs
        if hasattr(self, 'ent_min_h'):
            self.ent_min_h.delete(0, self.tk.END)
            self.ent_min_h.insert(0, f"{self.min_height:.0f}")
            self.ent_max_h.delete(0, self.tk.END)
            self.ent_max_h.insert(0, f"{self.max_height:.0f}")
        if hasattr(self, 'ent_min_tab1'):
            self.ent_min_tab1.delete(0, self.tk.END)
            self.ent_min_tab1.insert(0, f"{self.min_height:.0f}")
            self.ent_max_tab1.delete(0, self.tk.END)
            self.ent_max_tab1.insert(0, f"{self.max_height:.0f}")

        # Update y-limits of all 4 2D panels
        for ax in self.col1_axes:
            ax.set_ylim(self.min_height, self.max_height)

        # Update 1D profile
        self.update_column2_profile()
        self.canvas.draw_idle()

    # --- Plotting Routines ---

    def plot_all(self):
        for i in range(4):
            self.update_column1_panel(i)
        self._update_profile_indicator_lines()
        self.update_column2_profile()

    def update_column1_panel(self, i):
        ax = self.col1_axes[i]
        ax.cla()

        st = self.col1_settings[i]
        vname = st['var']
        vis = st['vis']

        da = self.current_ds[vname]
        if da.dims != ('time', 'z') and 'time' in da.dims and 'z' in da.dims:
            da = da.transpose('time', 'z')

        z_coords = self.mgr.get_z_coords(self.current_ds)
        time_vals = pd.to_datetime(self.current_ds.time.values)
        time_nums = mdates.date2num(time_vals)

        # Transpose so shape is (z, time) for pcolormesh(x, y, C)
        c_data = da.values.T

        mesh = ax.pcolormesh(
            time_nums, z_coords, c_data,
            cmap=vis['cmap'], norm=vis['norm'],
            shading='auto', rasterized=True
        )

        ax.set_ylim(self.min_height, self.max_height)
        ax.set_ylabel("Height (m)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Title with units
        unit_str = f" [{vis['units']}]" if vis['units'] else ""
        ax.set_title(f"Panel {i+1}: {vname}{unit_str}", fontsize=10, fontweight='bold', loc='left')

        # Formatting x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (UTC)", fontsize=9)

        # Dedicated, fixed colorbar axis (freshly created from GridSpec so it never shrinks vertically)
        if self.cbars[i] is not None:
            try:
                self.cbars[i].remove()
            except Exception:
                pass
            self.cbars[i] = None

        if self.col1_caxes[i] is not None:
            try:
                self.col1_caxes[i].remove()
            except Exception:
                pass
            self.col1_caxes[i] = None

        cax = self.fig.add_subplot(self.gs_left[i, 1])
        self.col1_caxes[i] = cax
        self.cbars[i] = self.fig.colorbar(mesh, cax=cax, extend='both')
        self.cbars[i].ax.tick_params(labelsize=8)

        # Overlay Melting layer lines if available
        self._overlay_ml_on_2d(ax, i)
        self._overlay_entropy_on_2d(ax, i)
        self._overlay_riming_on_2d(ax, i)

        self.canvas.draw_idle()

    def _overlay_ml_on_2d(self, ax, panel_idx=None):
        time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
        line_top, line_bot = None, None
        for ml_var in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
            if ml_var in self.current_ds:
                y = self.current_ds[ml_var].values
                if np.any(np.isfinite(y)):
                    line_top = ax.plot(time_nums, y, color='black', linestyle='--', linewidth=1.2, alpha=0.7, zorder=5)[0]
                    line_top.set_visible(self.show_ml)
                    break
        for ml_bot_var in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
            if ml_bot_var in self.current_ds:
                y = self.current_ds[ml_bot_var].values
                if np.any(np.isfinite(y)):
                    line_bot = ax.plot(time_nums, y, color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder=5)[0]
                    line_bot.set_visible(self.show_ml)
                    break
        if panel_idx is not None and panel_idx < len(self.col1_ml_artists):
            self.col1_ml_artists[panel_idx] = (line_top, line_bot)

    def _overlay_entropy_on_2d(self, ax, panel_idx=None):
        ent_var = None
        for cand in ['min_entropy', 'MIN_ENTROPY', 'min_Entropy', 'entropy_min']:
            if cand in self.current_ds:
                ent_var = cand
                break
        if ent_var is None:
            for v in self.current_ds.data_vars:
                if 'entropy' in v.lower():
                    ent_var = v
                    break
        if ent_var is None:
            if panel_idx is not None and panel_idx < len(self.col1_entropy_artists):
                self.col1_entropy_artists[panel_idx] = None
            return

        time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
        z_coords = self.mgr.get_z_coords(self.current_ds)
        da = self.current_ds[ent_var]
        if 'time' in da.dims and 'z' in da.dims:
            if da.dims != ('time', 'z'):
                da = da.transpose('time', 'z')
            try:
                da = da.interpolate_na(dim='z', method='linear', limit=2)
            except Exception:
                pass
            ent_vals = da.values.T
        else:
            raw = np.asarray(da.values).squeeze()
            if raw.ndim == 2:
                if raw.shape == (len(time_nums), len(z_coords)):
                    ent_vals = raw.T
                elif raw.shape == (len(z_coords), len(time_nums)):
                    ent_vals = raw
                else:
                    if panel_idx is not None and panel_idx < len(self.col1_entropy_artists):
                        self.col1_entropy_artists[panel_idx] = None
                    return
            else:
                if panel_idx is not None and panel_idx < len(self.col1_entropy_artists):
                    self.col1_entropy_artists[panel_idx] = None
                return

        mask = np.where(ent_vals > 0.85, 1.0, np.nan)
        if np.any(np.isfinite(mask)):
            cmap_gray = mcolors.ListedColormap(['#666666'])
            mesh_ent = ax.pcolormesh(
                time_nums, z_coords, mask,
                cmap=cmap_gray, vmin=0, vmax=2, alpha=0.45,
                shading='auto', rasterized=True, zorder=3
            )
            mesh_ent.set_visible(self.show_entropy)
            if panel_idx is not None and panel_idx < len(self.col1_entropy_artists):
                self.col1_entropy_artists[panel_idx] = mesh_ent
        else:
            if panel_idx is not None and panel_idx < len(self.col1_entropy_artists):
                self.col1_entropy_artists[panel_idx] = None

    def toggle_entropy(self):
        self.show_entropy = not self.show_entropy
        text = "Entropy: ON" if self.show_entropy else "Entropy: OFF"
        if hasattr(self, 'btn_entropy'):
            self.btn_entropy.configure(text=text)
        if hasattr(self, 'btn_entropy_tab'):
            self.btn_entropy_tab.configure(text=text)

        for mesh in self.col1_entropy_artists:
            if mesh is not None:
                mesh.set_visible(self.show_entropy)
        self.canvas.draw_idle()

    def _get_riming_dataset(self, loc):
        if not loc:
            return None, None
        if loc in self.riming_ds_cache:
            return self.riming_ds_cache[loc]

        fpath = self.mgr.date_to_file.get(self.current_date, '')
        if not fpath and self.mgr.file_list:
            fpath = self.mgr.file_list[0]
        is_dmi = 'dmi' in fpath.lower() or loc in ['afy', 'ank', 'gzt', 'hty', 'svs']

        rim_fp, rim_var = find_riming_file(loc, is_dmi=is_dmi, riming_dir=DEFAULT_RIMING_DIR)
        if not rim_fp or not os.path.exists(rim_fp):
            self.riming_ds_cache[loc] = (None, None)
            return None, None

        try:
            ds_rim = xr.open_dataset(rim_fp)
            if not rim_var or rim_var not in ds_rim:
                rim_candidates = [v for v in ds_rim.data_vars if 'riming' in v.lower()]
                rim_var = rim_candidates[0] if rim_candidates else list(ds_rim.data_vars.keys())[0]
            self.riming_ds_cache[loc] = (ds_rim, rim_var)
            return ds_rim, rim_var
        except Exception:
            self.riming_ds_cache[loc] = (None, None)
            return None, None

    def _overlay_riming_on_2d(self, ax, panel_idx=None):
        loc = getattr(self, 'radar_location', None)
        if not loc:
            fpath = self.mgr.date_to_file.get(self.current_date, '')
            if not fpath and self.mgr.file_list:
                fpath = self.mgr.file_list[0]
            loc = extract_radar_location(fpath)
            if not loc:
                loc = extract_radar_location(self.current_ds)
            self.radar_location = loc

        if not loc:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        ds_rim, rim_var = self._get_riming_dataset(loc)
        if ds_rim is None or rim_var is None or rim_var not in ds_rim:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        try:
            da_rim = ds_rim[rim_var].sel(time=self.current_date)
        except Exception:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        if da_rim.size == 0 or len(da_rim.time) == 0:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        if 'z' in da_rim.dims and 'time' in da_rim.dims:
            if da_rim.dims != ('time', 'z'):
                da_rim = da_rim.transpose('time', 'z')

        raw_vals = da_rim.fillna(0).values
        if HAS_SCIPY:
            try:
                cleaned = binary_opening(raw_vals, structure=np.ones((3, 3))).astype(float)
            except Exception:
                cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)
        else:
            cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)

        time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
        z_coords = self.mgr.get_z_coords(self.current_ds)

        if cleaned.shape == (len(time_nums), len(z_coords)):
            mask = np.where(cleaned.T > 0.5, 1.0, np.nan)
        else:
            try:
                da_cleaned = da_rim.copy(data=cleaned)
                da_aligned = da_cleaned.reindex(time=self.current_ds.time, method='nearest', tolerance=np.timedelta64(15, 'm'))
                if 'z' in self.current_ds.dims and 'z' in da_aligned.dims:
                    da_aligned = da_aligned.reindex(z=self.current_ds.z, method='nearest', tolerance=250.0)
                vals_aligned = da_aligned.values
                mask = np.where(vals_aligned.T > 0.5, 1.0, np.nan)
            except Exception:
                mask = np.full((len(z_coords), len(time_nums)), np.nan)
                min_t = min(cleaned.shape[0], len(time_nums))
                min_z = min(cleaned.shape[1], len(z_coords))
                mask[:min_z, :min_t] = np.where(cleaned[:min_t, :min_z].T > 0.5, 1.0, np.nan)

        mesh_rim = None
        if np.any(np.isfinite(mask)):
            cmap_gray = mcolors.ListedColormap(['#666666'])
            mesh_rim = ax.pcolormesh(
                time_nums, z_coords, mask,
                cmap=cmap_gray, vmin=0, vmax=2, alpha=0.45,
                shading='auto', rasterized=True, zorder=3
            )
            mesh_rim.set_visible(self.show_riming)

        if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
            self.col1_riming_artists[panel_idx] = mesh_rim

    def toggle_riming(self):
        self.show_riming = not self.show_riming
        text = "Riming: ON" if self.show_riming else "Riming: OFF"
        if hasattr(self, 'btn_riming'):
            self.btn_riming.configure(text=text)
        if hasattr(self, 'btn_riming_tab'):
            self.btn_riming_tab.configure(text=text)

        for mesh in self.col1_riming_artists:
            if mesh is not None:
                mesh.set_visible(self.show_riming)
        self.canvas.draw_idle()

    def toggle_ml(self):
        self.show_ml = not self.show_ml
        text = "ML: ON" if self.show_ml else "ML: OFF"
        if hasattr(self, 'btn_ml'):
            self.btn_ml.configure(text=text)
        if hasattr(self, 'btn_ml_tab'):
            self.btn_ml_tab.configure(text=text)

        for pair in self.col1_ml_artists:
            if pair is not None:
                l_top, l_bot = pair
                if l_top is not None:
                    l_top.set_visible(self.show_ml)
                if l_bot is not None:
                    l_bot.set_visible(self.show_ml)

        self.update_column2_profile()
        self.canvas.draw_idle()

    def _update_profile_indicator_lines(self):
        for line in self.vlines:
            try:
                line.remove()
            except Exception:
                pass
        self.vlines.clear()

        if 0 <= self.current_time_idx < len(self.current_ds.time):
            t_num = mdates.date2num(pd.to_datetime(self.current_ds.time.values[self.current_time_idx]))
            for ax in self.col1_axes:
                l = ax.axvline(t_num, color='red', linestyle='--', linewidth=1.6, alpha=0.9, zorder=10)
                self.vlines.append(l)
        self.canvas.draw_idle()

    def update_column2_profile(self):
        for tw in self.twin_axes:
            try:
                tw.remove()
            except Exception:
                pass
        self.twin_axes.clear()

        self.ax_prof.cla()
        self.ax_prof.grid(True, linestyle=':', alpha=0.6)

        # Dynamic shrinking of ax_prof from the bottom to make room for all bottom x-axes
        N = len(self.active_1d_vars)
        orig_bbox = self.gs_right[0, 0].get_position(self.fig)
        orig_left = orig_bbox.x0
        orig_width = orig_bbox.width
        orig_top = orig_bbox.y1
        orig_bottom = orig_bbox.y0

        fig_h = self.fig.get_size_inches()[1]
        pts_per_axis = 28.0
        step_frac = pts_per_axis / (fig_h * 72.0)

        # Move bottom up as more variables are added
        if N > 1:
            extra_bottom = (N - 1) * step_frac
            max_bottom = orig_top - 0.28
            if extra_bottom > (max_bottom - orig_bottom):
                pts_per_axis = max(18.0, (max_bottom - orig_bottom) * (fig_h * 72.0) / (N - 1))
                extra_bottom = (N - 1) * (pts_per_axis / (fig_h * 72.0))
            new_bottom = min(orig_bottom + extra_bottom, max_bottom)
        else:
            new_bottom = orig_bottom

        new_height = orig_top - new_bottom
        pos = [orig_left, new_bottom, orig_width, new_height]

        self.ax_prof._subplotspec = None
        self.ax_prof.set_position(pos)
        self.ax_prof.set_ylim(self.min_height, self.max_height)

        if not self.active_1d_vars or self.current_time_idx >= len(self.current_ds.time):
            self.ax_prof.set_title("No profile selected", fontsize=11, fontweight='bold', pad=10)
            self.ax_prof.set_ylabel("Height (m)", fontsize=10, fontweight='bold')
            self.canvas.draw_idle()
            return

        ds_prof = self.current_ds.isel(time=self.current_time_idx)
        z_coords = self.mgr.get_z_coords(self.current_ds)
        z_mask = (z_coords >= self.min_height) & (z_coords <= self.max_height)

        # 1st variable on primary axis ax_prof (at bottom offset 0)
        v0 = self.active_1d_vars[0]
        self._plot_1d_curve(self.ax_prof, ds_prof, v0, z_coords, z_mask, is_twin=False, twin_idx=0, pts_spacing=pts_per_axis)

        # Additional variables on twiny() axes, all placed at the BOTTOM with spacing
        for i, v in enumerate(self.active_1d_vars[1:], start=1):
            tw = self.ax_prof.twiny()
            tw._subplotspec = None
            tw.set_position(pos)
            tw.set_ylim(self.min_height, self.max_height)
            self.twin_axes.append(tw)
            self._plot_1d_curve(tw, ds_prof, v, z_coords, z_mask, is_twin=True, twin_idx=i, pts_spacing=pts_per_axis)

        # Overlay ML lines as horizontal dashed lines
        self._overlay_ml_on_1d(self.ax_prof, ds_prof)

        # Clean title at the top
        t_val = self.current_ds.time.values[self.current_time_idx]
        t_str = pd.to_datetime(t_val).strftime('%Y-%m-%d %H:%M:%S')
        self.ax_prof.set_title(f"QVP Profile: {t_str}", fontsize=11, fontweight='bold', pad=10)

        self.ax_prof.set_ylabel("Height (m)", fontsize=10, fontweight='bold')
        self.ax_prof.set_ylim(self.min_height, self.max_height)
        self.canvas.draw_idle()

    def _plot_1d_curve(self, ax_target, ds_prof, var_name, z_coords, z_mask, is_twin, twin_idx, pts_spacing=28.0):
        st = self.var_styles[var_name]
        color = st['color']
        linestyle = st['linestyle']
        xmin = st['xmin']
        xmax = st['xmax']
        units = st['units']
        unit_str = f" [{units}]" if units else ""

        da = ds_prof[var_name]
        if 'time' in da.dims:
            da = da.squeeze()
        vals = np.squeeze(da.values)

        # Validate x-limits
        if xmin is None or xmax is None or not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            vis = get_var_vis_info(var_name, da)
            xmin = vis.get('vmin', 0.0)
            xmax = vis.get('vmax', 1.0)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
                xmin, xmax = 0.0, 1.0
            st['xmin'] = xmin
            st['xmax'] = xmax

        # Ensure y-limit is set and clipping is enforced
        ax_target.set_ylim(self.min_height, self.max_height)

        ax_target.plot(
            vals[z_mask], z_coords[z_mask],
            color=color, linestyle=linestyle, linewidth=2.0,
            label=var_name,
            clip_on=True
        )
        ax_target.set_xlim(xmin, xmax)

        # Keep ALL x-axes scales at the bottom
        ax_target.xaxis.set_ticks_position('bottom')
        ax_target.xaxis.set_label_position('bottom')
        ax_target.tick_params(axis='x', colors=color, labelsize=8, pad=2)

        if is_twin:
            # Hide non-bottom spines
            ax_target.spines['top'].set_visible(False)
            ax_target.spines['left'].set_visible(False)
            ax_target.spines['right'].set_visible(False)

            # Offset bottom spine outward (downward)
            offset_pt = twin_idx * pts_spacing
            ax_target.spines['bottom'].set_position(('outward', offset_pt))
            ax_target.spines['bottom'].set_color(color)
            ax_target.spines['bottom'].set_linewidth(1.5)
            ax_target.set_xlabel(f"{var_name}{unit_str}", color=color, fontweight='bold', fontsize=8, labelpad=2)
        else:
            ax_target.spines['bottom'].set_color(color)
            ax_target.spines['bottom'].set_linewidth(1.5)
            ax_target.spines['bottom'].set_position(('outward', 0))
            ax_target.set_xlabel(f"{var_name}{unit_str}", color=color, fontweight='bold', fontsize=8, labelpad=2)

    def _overlay_ml_on_1d(self, ax, ds_prof):
        if not getattr(self, 'show_ml', True):
            return
        ml_top = None
        ml_bot = None
        for k in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
            if k in ds_prof and np.isfinite(ds_prof[k].values):
                ml_top = float(ds_prof[k].values)
                break
        for k in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
            if k in ds_prof and np.isfinite(ds_prof[k].values):
                ml_bot = float(ds_prof[k].values)
                break

        if ml_top is not None and self.min_height <= ml_top <= self.max_height:
            ax.axhline(ml_top, color='black', linestyle='--', linewidth=1.4, alpha=0.7)
            ax.text(
                0.02, ml_top, f" ML Top ({int(ml_top)}m)",
                color='black', transform=ax.get_yaxis_transform(),
                va='bottom', fontsize=8, fontweight='bold'
            )
        if ml_bot is not None and self.min_height <= ml_bot <= self.max_height:
            ax.axhline(ml_bot, color='black', linestyle=':', linewidth=1.4, alpha=0.7)
            ax.text(
                0.02, ml_bot, f" ML Bot ({int(ml_bot)}m)",
                color='black', transform=ax.get_yaxis_transform(),
                va='bottom', fontsize=8, fontweight='bold'
            )

    def run(self):
        self.root.mainloop()

    def on_close(self):
        self.root.quit()
        self.root.destroy()


# ==============================================================================
# Standalone Pure Matplotlib Fallback GUI (e.g. for --mpl or headless/Spyder qt)
# ==============================================================================

class QVPInteractiveAppMpl:
    """
    Pure Matplotlib interactive GUI (useful with %matplotlib qt in IPython/Spyder).
    """
    def __init__(self, dataset_mgr, initial_date=None):
        self.mgr = dataset_mgr
        self.dates = self.mgr.date_list
        if not self.dates:
            raise ValueError("No valid QVP files/dates found to visualize.")

        self.current_date = initial_date if initial_date in self.dates else self.dates[0]
        self.current_ds = self.mgr.load_day(self.current_date)
        self.all_vars = self.mgr.get_available_variables(self.current_ds)
        self.min_height = getattr(self.mgr, 'min_height', 0.0)
        self.max_height = getattr(self.mgr, 'max_height', 12000.0)

        defaults = self.mgr.get_default_variables(self.all_vars)
        self.col1_settings = []
        for i in range(4):
            vname = defaults[i]
            vis = get_var_vis_info(vname, self.current_ds[vname])
            self.col1_settings.append({
                'var': vname,
                'cmap_name': vis['cmap_name'],
                'vmin': vis['vmin'],
                'vmax': vis['vmax'],
                'vis': vis
            })

        self.current_time_idx = 0
        self.active_1d_vars = list(defaults)
        self.var_styles = {}
        for idx, v in enumerate(self.all_vars):
            vis = get_var_vis_info(v, self.current_ds[v])
            self.var_styles[v] = {
                'color': DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)],
                'linestyle': LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)][1],
                'xmin': vis['vmin'],
                'xmax': vis['vmax'],
                'units': vis['units']
            }

        self.vlines = []
        self.cbars = [None] * 4
        self.twin_axes = []
        self.show_entropy = True
        self.show_riming = True
        self.show_ml = True
        self.col1_entropy_artists = [None] * 4
        self.col1_riming_artists = [None] * 4
        self.col1_ml_artists = [None] * 4
        self.riming_ds_cache = {}
        self.radar_location = None

        self._build_figure()
        self.plot_all()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.suptitle(f"QVP Viewer - {self.current_date}", fontsize=13, fontweight='bold')

        # Main GridSpec: Left (Column 1 + dedicated colorbars), Right (Column 2 profile)
        self.gs_main = self.fig.add_gridspec(
            1, 2,
            width_ratios=[1.5, 1.0],
            left=0.06, right=0.82,
            bottom=0.09, top=0.94,
            wspace=0.28
        )

        # Left Column: 4 rows x 2 cols (Heatmap, dedicated Colorbar)
        self.gs_left = GridSpecFromSubplotSpec(
            4, 2,
            subplot_spec=self.gs_main[0, 0],
            width_ratios=[1.0, 0.035],
            wspace=0.04,
            hspace=0.24
        )

        self.col1_axes = []
        self.col1_caxes = [None] * 4
        for i in range(4):
            ax = self.fig.add_subplot(self.gs_left[i, 0], sharex=self.col1_axes[0] if i > 0 else None)
            self.col1_axes.append(ax)

        # Right Column: 1D QVP profile
        self.gs_right = GridSpecFromSubplotSpec(
            1, 1,
            subplot_spec=self.gs_main[0, 1]
        )
        orig_bbox = self.gs_right[0, 0].get_position(self.fig)
        self.ax_prof = self.fig.add_axes(orig_bbox)

        # Date Slider & Step Buttons at the bottom
        ax_prev = self.fig.add_axes([0.15, 0.02, 0.08, 0.035])
        ax_slider = self.fig.add_axes([0.25, 0.02, 0.40, 0.035])
        ax_next = self.fig.add_axes([0.67, 0.02, 0.08, 0.035])

        self.btn_prev = Button(ax_prev, '◀ Prev Day')
        self.btn_next = Button(ax_next, 'Next Day ▶')
        self.date_slider = Slider(
            ax=ax_slider, label='Date Index',
            valmin=0, valmax=max(1, len(self.dates) - 1),
            valinit=self.dates.index(self.current_date),
            valstep=1
        )
        if len(self.dates) <= 1:
            ax_prev.set_visible(False)
            ax_next.set_visible(False)
            ax_slider.set_visible(False)

        self.btn_prev.on_clicked(lambda e: self.step_date(-1))
        self.btn_next.on_clicked(lambda e: self.step_date(1))
        self.date_slider.on_changed(self.on_slider_date_change)

        # Right sidebar: Checkbox list for 1D variables
        ax_check_title = self.fig.add_axes([0.84, 0.91, 0.14, 0.03])
        ax_check_title.axis('off')
        ax_check_title.text(0.0, 0.5, "1D Profile Variables:", fontweight='bold', va='center')

        # Show first 15 variables in checkboxes, with page flip buttons
        self.page_idx = 0
        self.page_size = 14
        self.ax_checks = self.fig.add_axes([0.84, 0.35, 0.14, 0.54])
        self._rebuild_mpl_checkbuttons()

        ax_pg_prev = self.fig.add_axes([0.84, 0.30, 0.065, 0.035])
        ax_pg_next = self.fig.add_axes([0.915, 0.30, 0.065, 0.035])
        self.btn_pg_prev = Button(ax_pg_prev, '◀ Page')
        self.btn_pg_next = Button(ax_pg_next, 'Page ▶')
        self.btn_pg_prev.on_clicked(lambda e: self.flip_checkbox_page(-1))
        self.btn_pg_next.on_clicked(lambda e: self.flip_checkbox_page(1))

        # Buttons to sync with Col 1
        ax_sync = self.fig.add_axes([0.84, 0.24, 0.14, 0.04])
        self.btn_sync = Button(ax_sync, 'Sync with Col 1')
        self.btn_sync.on_clicked(lambda e: self.sync_1d_with_col1())

        # Height range controls (Min H and Max H) in sidebar
        ax_hmin_lbl = self.fig.add_axes([0.84, 0.18, 0.065, 0.03])
        ax_hmin_lbl.axis('off')
        ax_hmin_lbl.text(0.0, 0.5, "Min H (m):", fontsize=8, fontweight='bold', va='center')
        ax_hmin = self.fig.add_axes([0.91, 0.18, 0.07, 0.03])
        self.txt_min_h = TextBox(ax_hmin, '', initial=f"{self.min_height:.0f}")

        ax_hmax_lbl = self.fig.add_axes([0.84, 0.14, 0.065, 0.03])
        ax_hmax_lbl.axis('off')
        ax_hmax_lbl.text(0.0, 0.5, "Max H (m):", fontsize=8, fontweight='bold', va='center')
        ax_hmax = self.fig.add_axes([0.91, 0.14, 0.07, 0.03])
        self.txt_max_h = TextBox(ax_hmax, '', initial=f"{self.max_height:.0f}")

        ax_apply_h = self.fig.add_axes([0.84, 0.09, 0.14, 0.035])
        self.btn_apply_h = Button(ax_apply_h, 'Apply Height')
        self.btn_apply_h.on_clicked(lambda e: self.apply_height_range())
        self.txt_min_h.on_submit(lambda text: self.apply_height_range())
        self.txt_max_h.on_submit(lambda text: self.apply_height_range())

        ax_entropy = self.fig.add_axes([0.84, 0.052, 0.065, 0.032])
        self.btn_entropy = Button(ax_entropy, 'Entropy: ON')
        self.btn_entropy.on_clicked(lambda e: self.toggle_entropy())

        ax_riming = self.fig.add_axes([0.915, 0.052, 0.065, 0.032])
        self.btn_riming = Button(ax_riming, 'Riming: ON')
        self.btn_riming.on_clicked(lambda e: self.toggle_riming())

        ax_ml = self.fig.add_axes([0.84, 0.012, 0.14, 0.032])
        self.btn_ml = Button(ax_ml, 'ML: ON')
        self.btn_ml.on_clicked(lambda e: self.toggle_ml())

        # Click handler on 2D plots
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def apply_height_range(self):
        try:
            min_val = float(self.txt_min_h.text.strip())
            max_val = float(self.txt_max_h.text.strip())
        except ValueError:
            return
        if min_val >= max_val or min_val < 0:
            return
        self.min_height = min_val
        self.max_height = max_val
        for ax in self.col1_axes:
            ax.set_ylim(self.min_height, self.max_height)
        self.update_column2_profile()
        self.fig.canvas.draw_idle()

    def _rebuild_mpl_checkbuttons(self):
        self.ax_checks.cla()
        start = self.page_idx * self.page_size
        end = min(len(self.all_vars), start + self.page_size)
        page_vars = self.all_vars[start:end]
        actives = [v in self.active_1d_vars for v in page_vars]

        if page_vars:
            self.chk_widget = CheckButtons(self.ax_checks, page_vars, actives)
            self.chk_widget.on_clicked(self.on_mpl_checkbox_click)
        else:
            self.ax_checks.text(0.1, 0.5, "No variables", va='center')

    def flip_checkbox_page(self, delta):
        max_page = (len(self.all_vars) - 1) // self.page_size
        new_page = max(0, min(max_page, self.page_idx + delta))
        if new_page != self.page_idx:
            self.page_idx = new_page
            self._rebuild_mpl_checkbuttons()
            self.fig.canvas.draw_idle()

    def on_mpl_checkbox_click(self, label):
        if label in self.active_1d_vars:
            self.active_1d_vars.remove(label)
        else:
            self.active_1d_vars.append(label)
        self.update_column2_profile()

    def sync_1d_with_col1(self):
        self.active_1d_vars = [s['var'] for s in self.col1_settings]
        self._rebuild_mpl_checkbuttons()
        self.update_column2_profile()

    def step_date(self, delta):
        idx = int(self.date_slider.val) + delta
        if 0 <= idx < len(self.dates):
            self.date_slider.set_val(idx)

    def on_slider_date_change(self, val):
        idx = int(val)
        if 0 <= idx < len(self.dates):
            target_date = self.dates[idx]
            if target_date != self.current_date:
                self.current_date = target_date
                self.fig.suptitle(f"QVP Viewer - {self.current_date}", fontsize=13, fontweight='bold')
                self.current_ds = self.mgr.load_day(self.current_date)
                self.all_vars = self.mgr.get_available_variables(self.current_ds)
                if self.current_time_idx >= len(self.current_ds.time):
                    self.current_time_idx = max(0, len(self.current_ds.time) - 1)
                self.plot_all()

    def on_canvas_click(self, event):
        if getattr(event, 'button', 1) != 1:
            return
        if hasattr(self, 'toolbar') and getattr(self.toolbar, 'mode', '') != '':
            return
        if event.inaxes in self.col1_axes and event.xdata is not None:
            time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
            idx = int(np.argmin(np.abs(time_nums - event.xdata)))
            self.current_time_idx = idx
            self._update_profile_indicator_lines()
            self.update_column2_profile()

    def plot_all(self):
        for i in range(4):
            self.update_column1_panel(i)
        self._update_profile_indicator_lines()
        self.update_column2_profile()

    def update_column1_panel(self, i):
        ax = self.col1_axes[i]
        ax.cla()

        st = self.col1_settings[i]
        vname = st['var']
        vis = st['vis']

        da = self.current_ds[vname]
        if da.dims != ('time', 'z') and 'time' in da.dims and 'z' in da.dims:
            da = da.transpose('time', 'z')

        z_coords = self.mgr.get_z_coords(self.current_ds)
        time_vals = pd.to_datetime(self.current_ds.time.values)
        time_nums = mdates.date2num(time_vals)

        c_data = da.values.T

        mesh = ax.pcolormesh(
            time_nums, z_coords, c_data,
            cmap=vis['cmap'], norm=vis['norm'],
            shading='auto', rasterized=True
        )

        ax.set_ylim(self.min_height, self.max_height)
        ax.set_ylabel("Height (m)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        unit_str = f" [{vis['units']}]" if vis['units'] else ""
        ax.set_title(f"Panel {i+1}: {vname}{unit_str}", fontsize=10, fontweight='bold', loc='left')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (UTC)", fontsize=9)

        # Dedicated, fixed colorbar axis (freshly created from GridSpec so it never shrinks vertically)
        if self.cbars[i] is not None:
            try:
                self.cbars[i].remove()
            except Exception:
                pass
            self.cbars[i] = None

        if self.col1_caxes[i] is not None:
            try:
                self.col1_caxes[i].remove()
            except Exception:
                pass
            self.col1_caxes[i] = None

        cax = self.fig.add_subplot(self.gs_left[i, 1])
        self.col1_caxes[i] = cax
        self.cbars[i] = self.fig.colorbar(mesh, cax=cax, extend='both')
        self.cbars[i].ax.tick_params(labelsize=8)

        # ML lines
        time_nums = mdates.date2num(time_vals)
        line_top, line_bot = None, None
        for ml_var in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
            if ml_var in self.current_ds:
                y = self.current_ds[ml_var].values
                if np.any(np.isfinite(y)):
                    line_top = ax.plot(time_nums, y, color='black', linestyle='--', linewidth=1.2, alpha=0.7, zorder=5)[0]
                    line_top.set_visible(self.show_ml)
                    break
        for ml_bot_var in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
            if ml_bot_var in self.current_ds:
                y = self.current_ds[ml_bot_var].values
                if np.any(np.isfinite(y)):
                    line_bot = ax.plot(time_nums, y, color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder=5)[0]
                    line_bot.set_visible(self.show_ml)
                    break
        self.col1_ml_artists[i] = (line_top, line_bot)

        # Shading for min_entropy > 0.85
        ent_var = None
        for cand in ['min_entropy', 'MIN_ENTROPY', 'min_Entropy', 'entropy_min']:
            if cand in self.current_ds:
                ent_var = cand
                break
        if ent_var is None:
            for v in self.current_ds.data_vars:
                if 'entropy' in v.lower():
                    ent_var = v
                    break
        mesh_ent = None
        if ent_var is not None:
            da_ent = self.current_ds[ent_var]
            if 'time' in da_ent.dims and 'z' in da_ent.dims:
                if da_ent.dims != ('time', 'z'):
                    da_ent = da_ent.transpose('time', 'z')
                try:
                    da_ent = da_ent.interpolate_na(dim='z', method='linear', limit=2)
                except Exception:
                    pass
                ent_vals = da_ent.values.T
            else:
                raw = np.asarray(da_ent.values).squeeze()
                if raw.ndim == 2 and raw.shape == (len(time_nums), len(z_coords)):
                    ent_vals = raw.T
                elif raw.ndim == 2 and raw.shape == (len(z_coords), len(time_nums)):
                    ent_vals = raw
                else:
                    ent_vals = None
            if ent_vals is not None:
                mask = np.where(ent_vals > 0.85, 1.0, np.nan)
                if np.any(np.isfinite(mask)):
                    cmap_gray = mcolors.ListedColormap(['#666666'])
                    mesh_ent = ax.pcolormesh(
                        time_nums, z_coords, mask,
                        cmap=cmap_gray, vmin=0, vmax=2, alpha=0.45,
                        shading='auto', rasterized=True, zorder=3
                    )
                    mesh_ent.set_visible(self.show_entropy)
        self.col1_entropy_artists[i] = mesh_ent

        # Riming classification overlay
        self._overlay_riming_on_2d(ax, i)

        self.fig.canvas.draw_idle()

    def _get_riming_dataset(self, loc):
        if not loc:
            return None, None
        if loc in self.riming_ds_cache:
            return self.riming_ds_cache[loc]

        fpath = self.mgr.date_to_file.get(self.current_date, '')
        if not fpath and self.mgr.file_list:
            fpath = self.mgr.file_list[0]
        is_dmi = 'dmi' in fpath.lower() or loc in ['afy', 'ank', 'gzt', 'hty', 'svs']

        rim_fp, rim_var = find_riming_file(loc, is_dmi=is_dmi, riming_dir=DEFAULT_RIMING_DIR)
        if not rim_fp or not os.path.exists(rim_fp):
            self.riming_ds_cache[loc] = (None, None)
            return None, None

        try:
            ds_rim = xr.open_dataset(rim_fp)
            if not rim_var or rim_var not in ds_rim:
                rim_candidates = [v for v in ds_rim.data_vars if 'riming' in v.lower()]
                rim_var = rim_candidates[0] if rim_candidates else list(ds_rim.data_vars.keys())[0]
            self.riming_ds_cache[loc] = (ds_rim, rim_var)
            return ds_rim, rim_var
        except Exception:
            self.riming_ds_cache[loc] = (None, None)
            return None, None

    def _overlay_riming_on_2d(self, ax, panel_idx=None):
        loc = getattr(self, 'radar_location', None)
        if not loc:
            fpath = self.mgr.date_to_file.get(self.current_date, '')
            if not fpath and self.mgr.file_list:
                fpath = self.mgr.file_list[0]
            loc = extract_radar_location(fpath)
            if not loc:
                loc = extract_radar_location(self.current_ds)
            self.radar_location = loc

        if not loc:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        ds_rim, rim_var = self._get_riming_dataset(loc)
        if ds_rim is None or rim_var is None or rim_var not in ds_rim:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        try:
            da_rim = ds_rim[rim_var].sel(time=self.current_date)
        except Exception:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        if da_rim.size == 0 or len(da_rim.time) == 0:
            if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
                self.col1_riming_artists[panel_idx] = None
            return

        if 'z' in da_rim.dims and 'time' in da_rim.dims:
            if da_rim.dims != ('time', 'z'):
                da_rim = da_rim.transpose('time', 'z')

        raw_vals = da_rim.fillna(0).values
        if HAS_SCIPY:
            try:
                cleaned = binary_opening(raw_vals, structure=np.ones((3, 3))).astype(float)
            except Exception:
                cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)
        else:
            cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)

        time_nums = mdates.date2num(pd.to_datetime(self.current_ds.time.values))
        z_coords = self.mgr.get_z_coords(self.current_ds)

        if cleaned.shape == (len(time_nums), len(z_coords)):
            mask = np.where(cleaned.T > 0.5, 1.0, np.nan)
        else:
            try:
                da_cleaned = da_rim.copy(data=cleaned)
                da_aligned = da_cleaned.reindex(time=self.current_ds.time, method='nearest', tolerance=np.timedelta64(15, 'm'))
                if 'z' in self.current_ds.dims and 'z' in da_aligned.dims:
                    da_aligned = da_aligned.reindex(z=self.current_ds.z, method='nearest', tolerance=250.0)
                vals_aligned = da_aligned.values
                mask = np.where(vals_aligned.T > 0.5, 1.0, np.nan)
            except Exception:
                mask = np.full((len(z_coords), len(time_nums)), np.nan)
                min_t = min(cleaned.shape[0], len(time_nums))
                min_z = min(cleaned.shape[1], len(z_coords))
                mask[:min_z, :min_t] = np.where(cleaned[:min_t, :min_z].T > 0.5, 1.0, np.nan)

        mesh_rim = None
        if np.any(np.isfinite(mask)):
            cmap_gray = mcolors.ListedColormap(['#666666'])
            mesh_rim = ax.pcolormesh(
                time_nums, z_coords, mask,
                cmap=cmap_gray, vmin=0, vmax=2, alpha=0.45,
                shading='auto', rasterized=True, zorder=3
            )
            mesh_rim.set_visible(self.show_riming)

        if panel_idx is not None and panel_idx < len(self.col1_riming_artists):
            self.col1_riming_artists[panel_idx] = mesh_rim

    def toggle_riming(self):
        self.show_riming = not self.show_riming
        text = "Riming: ON" if self.show_riming else "Riming: OFF"
        if hasattr(self, 'btn_riming') and hasattr(self.btn_riming, 'label'):
            self.btn_riming.label.set_text(text)
        for mesh in self.col1_riming_artists:
            if mesh is not None:
                mesh.set_visible(self.show_riming)
        self.fig.canvas.draw_idle()

    def toggle_entropy(self):
        self.show_entropy = not self.show_entropy
        text = "Entropy: ON" if self.show_entropy else "Entropy: OFF"
        if hasattr(self, 'btn_entropy') and hasattr(self.btn_entropy, 'label'):
            self.btn_entropy.label.set_text(text)
        for mesh in self.col1_entropy_artists:
            if mesh is not None:
                mesh.set_visible(self.show_entropy)
        self.fig.canvas.draw_idle()

    def toggle_ml(self):
        self.show_ml = not self.show_ml
        text = "ML: ON" if self.show_ml else "ML: OFF"
        if hasattr(self, 'btn_ml') and hasattr(self.btn_ml, 'label'):
            self.btn_ml.label.set_text(text)
        for pair in self.col1_ml_artists:
            if pair is not None:
                l_top, l_bot = pair
                if l_top is not None:
                    l_top.set_visible(self.show_ml)
                if l_bot is not None:
                    l_bot.set_visible(self.show_ml)
        self.update_column2_profile()
        self.fig.canvas.draw_idle()

    def _update_profile_indicator_lines(self):
        for line in self.vlines:
            try:
                line.remove()
            except Exception:
                pass
        self.vlines.clear()

        if 0 <= self.current_time_idx < len(self.current_ds.time):
            t_num = mdates.date2num(pd.to_datetime(self.current_ds.time.values[self.current_time_idx]))
            for ax in self.col1_axes:
                l = ax.axvline(t_num, color='red', linestyle='--', linewidth=1.6, alpha=0.9, zorder=10)
                self.vlines.append(l)
        self.fig.canvas.draw_idle()

    def update_column2_profile(self):
        for tw in self.twin_axes:
            try:
                tw.remove()
            except Exception:
                pass
        self.twin_axes.clear()

        self.ax_prof.cla()
        self.ax_prof.grid(True, linestyle=':', alpha=0.6)

        # Dynamic shrinking of ax_prof from the bottom to make room for all bottom x-axes
        N = len(self.active_1d_vars)
        orig_bbox = self.gs_right[0, 0].get_position(self.fig)
        orig_left = orig_bbox.x0
        orig_width = orig_bbox.width
        orig_top = orig_bbox.y1
        orig_bottom = orig_bbox.y0

        fig_h = self.fig.get_size_inches()[1]
        pts_per_axis = 28.0
        step_frac = pts_per_axis / (fig_h * 72.0)

        # Move bottom up as more variables are added
        if N > 1:
            extra_bottom = (N - 1) * step_frac
            max_bottom = orig_top - 0.28
            if extra_bottom > (max_bottom - orig_bottom):
                pts_per_axis = max(18.0, (max_bottom - orig_bottom) * (fig_h * 72.0) / (N - 1))
                extra_bottom = (N - 1) * (pts_per_axis / (fig_h * 72.0))
            new_bottom = min(orig_bottom + extra_bottom, max_bottom)
        else:
            new_bottom = orig_bottom

        new_height = orig_top - new_bottom
        pos = [orig_left, new_bottom, orig_width, new_height]

        self.ax_prof._subplotspec = None
        self.ax_prof.set_position(pos)
        self.ax_prof.set_ylim(self.min_height, self.max_height)

        if not self.active_1d_vars or self.current_time_idx >= len(self.current_ds.time):
            self.ax_prof.set_title("No profile selected", fontsize=11, fontweight='bold', pad=10)
            self.ax_prof.set_ylabel("Height (m)", fontsize=10, fontweight='bold')
            self.fig.canvas.draw_idle()
            return

        ds_prof = self.current_ds.isel(time=self.current_time_idx)
        z_coords = self.mgr.get_z_coords(self.current_ds)
        z_mask = (z_coords >= self.min_height) & (z_coords <= self.max_height)

        # 1st variable on primary axis ax_prof (at bottom offset 0)
        v0 = self.active_1d_vars[0]
        self._plot_1d_curve(self.ax_prof, ds_prof, v0, z_coords, z_mask, is_twin=False, twin_idx=0, pts_spacing=pts_per_axis)

        # Additional variables on twiny() axes, all placed at the BOTTOM with spacing
        for i, v in enumerate(self.active_1d_vars[1:], start=1):
            tw = self.ax_prof.twiny()
            tw._subplotspec = None
            tw.set_position(pos)
            tw.set_ylim(self.min_height, self.max_height)
            self.twin_axes.append(tw)
            self._plot_1d_curve(tw, ds_prof, v, z_coords, z_mask, is_twin=True, twin_idx=i, pts_spacing=pts_per_axis)

        # ML lines
        if getattr(self, 'show_ml', True):
            ml_top, ml_bot = None, None
            for k in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
                if k in ds_prof and np.isfinite(ds_prof[k].values):
                    ml_top = float(ds_prof[k].values)
                    break
            for k in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
                if k in ds_prof and np.isfinite(ds_prof[k].values):
                    ml_bot = float(ds_prof[k].values)
                    break

            if ml_top is not None and self.min_height <= ml_top <= self.max_height:
                self.ax_prof.axhline(ml_top, color='black', linestyle='--', linewidth=1.4, alpha=0.7)
                self.ax_prof.text(0.02, ml_top, f" ML Top ({int(ml_top)}m)", transform=self.ax_prof.get_yaxis_transform(), va='bottom', fontsize=8, fontweight='bold')
            if ml_bot is not None and self.min_height <= ml_bot <= self.max_height:
                self.ax_prof.axhline(ml_bot, color='black', linestyle=':', linewidth=1.4, alpha=0.7)
                self.ax_prof.text(0.02, ml_bot, f" ML Bot ({int(ml_bot)}m)", transform=self.ax_prof.get_yaxis_transform(), va='bottom', fontsize=8, fontweight='bold')

        t_val = self.current_ds.time.values[self.current_time_idx]
        t_str = pd.to_datetime(t_val).strftime('%Y-%m-%d %H:%M:%S')
        self.ax_prof.set_title(f"QVP Profile: {t_str}", fontsize=11, fontweight='bold', pad=10)

        self.ax_prof.set_ylabel("Height (m)", fontsize=10, fontweight='bold')
        self.ax_prof.set_ylim(self.min_height, self.max_height)
        self.fig.canvas.draw_idle()

    def _plot_1d_curve(self, ax_target, ds_prof, var_name, z_coords, z_mask, is_twin, twin_idx, pts_spacing=28.0):
        st = self.var_styles[var_name]
        color = st['color']
        linestyle = st['linestyle']
        xmin = st['xmin']
        xmax = st['xmax']
        units = st['units']
        unit_str = f" [{units}]" if units else ""

        da = ds_prof[var_name]
        if 'time' in da.dims:
            da = da.squeeze()
        vals = np.squeeze(da.values)

        if xmin is None or xmax is None or not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            vis = get_var_vis_info(var_name, da)
            xmin = vis.get('vmin', 0.0)
            xmax = vis.get('vmax', 1.0)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
                xmin, xmax = 0.0, 1.0
            st['xmin'] = xmin
            st['xmax'] = xmax

        # Ensure y-limit is set and clipping is enforced
        ax_target.set_ylim(self.min_height, self.max_height)

        ax_target.plot(
            vals[z_mask], z_coords[z_mask],
            color=color, linestyle=linestyle, linewidth=2.0,
            label=var_name,
            clip_on=True
        )
        ax_target.set_xlim(xmin, xmax)

        # Keep ALL x-axes scales at the bottom
        ax_target.xaxis.set_ticks_position('bottom')
        ax_target.xaxis.set_label_position('bottom')
        ax_target.tick_params(axis='x', colors=color, labelsize=8, pad=2)

        if is_twin:
            # Hide non-bottom spines
            ax_target.spines['top'].set_visible(False)
            ax_target.spines['left'].set_visible(False)
            ax_target.spines['right'].set_visible(False)

            # Offset bottom spine outward (downward)
            offset_pt = twin_idx * pts_spacing
            ax_target.spines['bottom'].set_position(('outward', offset_pt))
            ax_target.spines['bottom'].set_color(color)
            ax_target.spines['bottom'].set_linewidth(1.5)
            ax_target.set_xlabel(f"{var_name}{unit_str}", color=color, fontweight='bold', fontsize=8, labelpad=2)
        else:
            ax_target.spines['bottom'].set_color(color)
            ax_target.spines['bottom'].set_linewidth(1.5)
            ax_target.spines['bottom'].set_position(('outward', 0))
            ax_target.set_xlabel(f"{var_name}{unit_str}", color=color, fontweight='bold', fontsize=8, labelpad=2)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive visualization of Quasi-Vertical Profile (QVP) radar data."
    )
    parser.add_argument(
        "paths", nargs="*", default=[DEFAULT_PATH_QVPS],
        help=f"File path(s), directory, or glob pattern to QVP data (default: {DEFAULT_PATH_QVPS})"
    )
    parser.add_argument(
        "--mpl", action="store_true",
        help="Force pure Matplotlib GUI backend instead of Tkinter."
    )
    parser.add_argument(
        "--min-height", type=float, default=0.0,
        help="Minimum height for QVP plots in meters (default: 0.0)"
    )
    parser.add_argument(
        "--max-height", type=float, default=12000.0,
        help="Maximum height for QVP plots in meters (default: 12000.0)"
    )

    args = parser.parse_args()

    print("Locating QVP files...")
    files = find_files_from_input(args.paths)
    if not files:
        print(f"No files matched by inputs: {args.paths}")
        print("Please provide a valid file path, glob pattern, or directory containing QVP data.")
        sys.exit(1)

    print(f"Found {len(files)} file(s).")
    mgr = QVPDatasetManager(files, min_height=args.min_height, max_height=args.max_height)
    print(f"Available dates ({len(mgr.date_list)}): {mgr.date_list[:5]}{'...' if len(mgr.date_list) > 5 else ''}")

    if args.mpl:
        print("Starting in pure Matplotlib mode...")
        app = QVPInteractiveAppMpl(mgr)
        plt.show()
    else:
        try:
            print("Launching interactive QVP desktop application...")
            app = QVPInteractiveAppTk(mgr)
            app.run()
        except Exception as e:
            print(f"Notice: Tkinter GUI encountered: {e}. Falling back to pure Matplotlib GUI...")
            app = QVPInteractiveAppMpl(mgr)
            plt.show()


if __name__ == "__main__":
    main()
