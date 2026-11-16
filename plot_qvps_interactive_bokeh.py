#!/usr/bin/env python3
"""
Interactive Quasi-Vertical Profile (QVP) Visualizer (Bokeh / Web Application Backend).

An alternative to plot_qvps_interactive.py utilizing Bokeh for:
  - High-performance pan & zoom in all subplots (wheel zoom, box zoom, drag pan, reset).
  - Synchronized x-axis (time) and y-axis (height) navigation across all 2D panels.
  - Floating hover overlay tooltip showing exact variable values, time, and height below the cursor.
  - Interactive profile selection by clicking (tapping) anywhere on the 2D heatmaps.
  - 1D Value vs Height profile with stacked multiple bottom x-axes, hover tooltips, and customizable ranges/styles.
  - Dynamic dataset loading supporting daily files, multi-day files, Icechunk, standard Zarr, allmoms, allsim, and NetCDF.

Usage:
  python plot_qvps_interactive_bokeh.py [paths]
  bokeh serve plot_qvps_interactive_bokeh.py --show --args [paths]
"""

import os
import sys
import glob
import re
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

# Matplotlib for colormap palette extraction
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Bokeh imports
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, HoverTool,
    Range1d, LinearAxis, Span, Label, Div,
    Select, Button, CheckboxGroup, NumericInput,
    Tabs, FixedTicker, BasicTicker
)
try:
    from bokeh.models import TabPanel
except ImportError:
    from bokeh.models import Panel as TabPanel

from bokeh.layouts import column, row, layout
from bokeh.events import Tap

# Icechunk support
try:
    from icechunk import Repository, local_filesystem_storage
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

# Possible project paths setup (following CFTDs.py convention)
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

# Default path config
DEFAULT_REALPEP_PATH = "/automount/realpep"
DEFAULT_PATH_QVPS = f"{DEFAULT_REALPEP_PATH}/upload/jgiles/dwd/qvps/*/*/2017-07-25/pro/vol5minng01/07/*allmoms*"

# Available colormap names (miub2 as top default)
CMAP_OPTIONS = [
    'miub2', 'default (visdict)', 'HomeyerRainbow', 'viridis', 'SpectralExtended',
    'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'RdBu_r', 'coolwarm', 'binary2'
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

# miub2 (Prabhakar) colors: 14 colors total
# Index 0: cyan (underflow, < vmin)
# Index 1-12: 12 discrete intervals between vmin and vmax
# Index 13: purple (overflow, > vmax)
colors_prabhakar = np.array([
    [0.00, 1.00, 1.00],  # 0: cyan (under)
    [0.00, 0.70, 0.93],  # 1: sky blue
    [0.00, 0.00, 1.00],  # 2: blue
    [0.50, 1.00, 0.00],  # 3: light green
    [0.40, 0.80, 0.00],  # 4: green
    [0.27, 0.55, 0.00],  # 5: dark green
    [1.00, 1.00, 0.00],  # 6: yellow
    [0.80, 0.80, 0.00],  # 7: dark yellow
    [1.00, 0.65, 0.00],  # 8: orange
    [1.00, 0.27, 0.00],  # 9: dark orange / red-orange
    [0.80, 0.22, 0.00],  # 10: reddish brown
    [0.55, 0.15, 0.00],  # 11: brown
    [1.00, 0.00, 1.00],  # 12: magenta
    [0.58, 0.44, 0.86]   # 13: purple (over)
])

LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']


def get_bokeh_palette(cmap_name_or_obj, n=256):
    """
    Convert a colormap name or matplotlib colormap object into a list of hex strings for Bokeh.
    """
    if cmap_name_or_obj in ['miub2', 'default (visdict)', 'colors_prabhakar']:
        return [mcolors.to_hex(c) for c in colors_prabhakar]
    if cmap_name_or_obj == 'binary2':
        return [mcolors.to_hex(c) for c in ["dimgray", "gainsboro"]]

    if isinstance(cmap_name_or_obj, str):
        try:
            cmap = mpl.colormaps.get_cmap(cmap_name_or_obj)
        except Exception:
            try:
                cmap = plt.get_cmap(cmap_name_or_obj)
            except Exception:
                cmap = mpl.colormaps.get_cmap('viridis')
    else:
        cmap = cmap_name_or_obj

    if hasattr(cmap, 'colors') and len(cmap.colors) > 0:
        return [mcolors.to_hex(c) for c in cmap.colors]
    return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]


def get_color_mapper_params(cmap_name, vmin, vmax, var_name=None):
    """
    Given colormap name, vmin, vmax, and optional variable name:
    Returns (palette, low, high, low_color, high_color, bar_ticks, major_label_overrides)
    properly aligning discrete radar colormaps (especially miub2) with tick marks and extreme under/over colors.
    """
    if cmap_name == 'default (visdict)':
        cmap_name = 'miub2'

    vmin = float(vmin)
    vmax = float(vmax)
    if vmin >= vmax:
        vmax = vmin + 1.0

    if cmap_name.lower().startswith('miub') or cmap_name == 'colors_prabhakar':
        hex_colors = [mcolors.to_hex(c) for c in colors_prabhakar]  # 14 colors
        c_under = hex_colors[0]   # cyan
        c_over = hex_colors[-1]   # purple
        c_mid = hex_colors[1:-1]  # 12 intermediate interval colors

        # Look up ticks in radarmet.visdict14 if variable name provided
        ticks = None
        if radarmet is not None and hasattr(radarmet, 'visdict14') and var_name:
            if var_name in radarmet.visdict14:
                ticks = radarmet.visdict14[var_name].get('ticks', None)
            else:
                prefix = var_name.split('_')[0]
                for k in [prefix, 'DBZH', 'ZDR', 'KDP', 'RHOHV', 'PHIDP', 'TH']:
                    if k in radarmet.visdict14 and (k in var_name or var_name.startswith(k)):
                        ticks = radarmet.visdict14[k].get('ticks', None)
                        break

        def fmt_val(v):
            return f"{int(round(v))}" if abs(v - round(v)) < 1e-4 else f"{v:g}"

        # If visdict ticks match current vmin/vmax range:
        if ticks is not None and len(ticks) >= 2:
            t_min = float(ticks[0])
            t_max = float(ticks[-1])
            if abs(vmin - t_min) < 1e-2 and abs(vmax - t_max) < 1e-2:
                diffs = np.diff(ticks)
                is_uniform = np.allclose(diffs, diffs[0], rtol=1e-2)
                if is_uniform:
                    # Uniform intervals (e.g. DBZH [-10 to 50 step 5])
                    step = float(diffs[0])
                    low = vmin - step
                    high = vmax + step
                    palette = hex_colors  # All 14 colors: [0] is [low, vmin], [13] is [vmax, high]

                    main_ticks = [float(t) for t in ticks]
                    if len(main_ticks) > 8:
                        bar_ticks = [t for i, t in enumerate(main_ticks) if i % 2 == 0 or t == vmax or t == vmin]
                    else:
                        bar_ticks = list(main_ticks)

                    all_bar_ticks = sorted(list(set([low] + bar_ticks + [high])))
                    overrides = {
                        low: f"<{fmt_val(vmin)}",
                        str(low): f"<{fmt_val(vmin)}",
                        high: f">{fmt_val(vmax)}",
                        str(high): f">{fmt_val(vmax)}"
                    }
                    for t in bar_ticks:
                        s_val = fmt_val(t)
                        overrides[t] = s_val
                        overrides[str(t)] = s_val
                        if abs(t - round(t)) < 1e-4:
                            overrides[float(int(round(t)))] = s_val
                            overrides[str(int(round(t)))] = s_val

                    return palette, low, high, c_under, c_over, all_bar_ticks, overrides
                else:
                    # Non-uniform intervals (e.g. ZDR, KDP, RHOHV)
                    avg_step = (vmax - vmin) / float(len(ticks) - 1)
                    low = vmin - avg_step
                    high = vmax + avg_step
                    N = 512
                    x_vals = np.linspace(low, high, N)
                    palette_hr = []
                    for x in x_vals:
                        if x < vmin:
                            palette_hr.append(c_under)
                        elif x >= vmax:
                            palette_hr.append(c_over)
                        else:
                            idx = np.searchsorted(ticks, x, side='right') - 1
                            idx = max(0, min(idx, len(c_mid) - 1))
                            palette_hr.append(c_mid[idx])

                    bar_ticks = sorted(list(set([low] + [float(t) for t in ticks if vmin <= t <= vmax] + [high])))
                    overrides = {
                        low: f"<{fmt_val(vmin)}",
                        str(low): f"<{fmt_val(vmin)}",
                        high: f">{fmt_val(vmax)}",
                        str(high): f">{fmt_val(vmax)}"
                    }
                    for t in ticks:
                        if vmin <= t <= vmax:
                            s_val = fmt_val(t)
                            overrides[float(t)] = s_val
                            overrides[str(float(t))] = s_val

                    return palette_hr, low, high, c_under, c_over, bar_ticks, overrides

        # Fallback for custom vmin/vmax with miub2
        step = (vmax - vmin) / 12.0
        low = vmin - step
        high = vmax + step
        palette = hex_colors
        mid_ticks = [vmin, vmin + 3 * step, vmin + 6 * step, vmin + 9 * step, vmax]
        bar_ticks = sorted(list(set([low] + mid_ticks + [high])))
        overrides = {
            low: f"<{fmt_val(vmin)}",
            str(low): f"<{fmt_val(vmin)}",
            high: f">{fmt_val(vmax)}",
            str(high): f">{fmt_val(vmax)}"
        }
        for t in mid_ticks:
            s_val = fmt_val(t)
            overrides[t] = s_val
            overrides[str(t)] = s_val
        return palette, low, high, c_under, c_over, bar_ticks, overrides

    # Continuous colormaps (e.g. viridis, HomeyerRainbow, turbo, etc.)
    palette = get_bokeh_palette(cmap_name, n=256)
    low = vmin
    high = vmax
    low_color = palette[0]
    high_color = palette[-1]

    # Generate 6 clean, evenly-spaced ticks
    t_vals = np.linspace(vmin, vmax, 6)
    bar_ticks = [float(v) for v in t_vals]

    def fmt_val(v):
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-4 else f"{v:g}"

    overrides = {float(v): fmt_val(v) for v in bar_ticks}
    for v in bar_ticks:
        overrides[str(float(v))] = fmt_val(v)
        if abs(v - round(v)) < 1e-4:
            overrides[float(int(round(v)))] = fmt_val(v)
            overrides[str(int(round(v)))] = fmt_val(v)

    return palette, low, high, low_color, high_color, bar_ticks, overrides


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
        # 2. Directly specified .zarr store (which is a directory on disk, not a regular file)
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
                entries = glob.glob(os.path.join(item, "*"))
                for entry in entries:
                    c_entry = entry.rstrip('/')
                    if c_entry.endswith('.zarr') or c_entry.endswith('.nc') or any(k in c_entry for k in ['allm', 'allsim']):
                        files_found.append(c_entry)
        # 4. Directly specified file
        elif os.path.isfile(item):
            if os.path.basename(item) == "ML_detected.txt":
                m = glob.glob(os.path.join(os.path.dirname(item), "*allm*")) + glob.glob(os.path.join(os.path.dirname(item), "*allsim*"))
                if m:
                    files_found.extend([x.rstrip('/') for x in m])
                else:
                    files_found.append(item)
            else:
                files_found.append(item)
        # 5. Direct fallback: path passed directly
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

    if not filtered:
        for raw_item in paths_input:
            if raw_item and raw_item.strip():
                filtered.append(raw_item.strip().rstrip('/'))

    return sorted(list(set(filtered)))


def extract_date_from_file(file_path):
    """
    Extract YYYY-MM-DD string from filename or parent folder path.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", file_path)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(file_path))
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = re.search(r"(\d{4})(\d{2})(\d{2})", file_path)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


class QVPDatasetManager:
    """
    Manages loading, caching, date extraction, and dimension alignment for QVP datasets.
    """
    def __init__(self, file_list, min_height=0.0, max_height=12000.0):
        self.file_list = sorted(file_list)
        self.min_height = min_height
        self.max_height = max_height
        self.cache = {}
        self.is_single_file_multi_day = False
        self.single_file_ds = None
        self.date_to_file = {}
        self.date_list = []
        self._init_date_mapping()

    def _open_raw(self, fpath):
        clean_path = fpath.rstrip('/')

        # 1. Try utils.load_qvps if available
        if utils is not None and hasattr(utils, 'load_qvps'):
            try:
                ds = utils.load_qvps([clean_path], load=True)
                if ds is not None:
                    return ds
            except Exception:
                pass

        # 2. Check for .zarr (Icechunk or standard Zarr)
        if clean_path.endswith('.zarr'):
            if HAS_ICECHUNK:
                try:
                    storage = local_filesystem_storage(clean_path)
                    repo = Repository.open(storage)
                    session = repo.readonly_session("main")
                    ds = xr.open_zarr(session.store, consolidated=False)
                    return ds.load()
                except Exception:
                    pass

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

        # 4. Final attempt
        try:
            return xr.open_zarr(clean_path).load()
        except Exception:
            pass
        return xr.open_dataset(clean_path).load()

    def _standardize_ds(self, ds):
        vert_candidates = ['height', 'range', 'altitude', 'alt', 'height_above_ground', 'height_above_sea_level', 'nheight', 'z_levels', 'level']
        for vert_name in vert_candidates:
            if 'z' not in ds.dims and vert_name in ds.dims:
                ds = ds.rename({vert_name: 'z'})
            if 'z' not in ds.coords and vert_name in ds.coords:
                ds = ds.assign_coords(z=ds[vert_name])

        time_candidates = ['Time', 'datetime', 'valid_time', 'Times', 't']
        for t_name in time_candidates:
            if 'time' not in ds.dims and t_name in ds.dims:
                ds = ds.rename({t_name: 'time'})
            if 'time' not in ds.coords and t_name in ds.coords:
                ds = ds.assign_coords(time=ds[t_name])

        if 'time' in ds.coords:
            valid_t = ds['time'].notnull().values
            if not valid_t.all():
                ds = ds.isel(time=valid_t)
        return ds

    def _init_date_mapping(self):
        if not self.file_list:
            return

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

        ds = self._open_raw(fpath)
        ds = self._standardize_ds(ds)

        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[date_str] = ds
        return ds

    def get_available_variables(self, ds):
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
        if np.nanmax(z_vals) < 50.0:
            z_vals = z_vals * 1000.0
        return z_vals

    def get_default_variables(self, available_vars):
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
    vinfo = None
    if radarmet is not None and hasattr(radarmet, 'visdict14'):
        visdict14 = radarmet.visdict14
        if var_name in visdict14:
            vinfo = visdict14[var_name]
        else:
            prefix = var_name.split("_")[0]
            for k in [prefix, 'DBZH', 'ZDR', 'KDP', 'RHOHV', 'PHIDP', 'TH']:
                if k in visdict14 and (k in var_name or var_name.startswith(k)):
                    vinfo = visdict14[k]
                    break
            if vinfo is None:
                for k in visdict14:
                    if k in var_name or var_name in k:
                        vinfo = visdict14[k]
                        break

    units = ""
    if data_array is not None and hasattr(data_array, 'attrs'):
        units = data_array.attrs.get('units', '')

    if vinfo is not None:
        raw_cmap = vinfo.get('cmap', 'miub2')
        if isinstance(raw_cmap, str):
            cmap_name = raw_cmap
            try:
                cmap = mpl.colormaps.get_cmap(raw_cmap)
            except Exception:
                try:
                    cmap = plt.get_cmap(raw_cmap)
                except Exception:
                    cmap = 'miub2'
        elif isinstance(raw_cmap, mpl.colors.Colormap):
            cmap = raw_cmap
            cmap_name = getattr(raw_cmap, 'name', 'miub2')
        else:
            cmap = 'miub2'
            cmap_name = 'miub2'

        ticks = vinfo.get('ticks', None)
        if ticks is not None and len(ticks) > 1:
            vmin = float(ticks[0])
            vmax = float(ticks[-1])
        else:
            vmin = float(vinfo.get('vmin', 0.0))
            vmax = float(vinfo.get('vmax', 1.0))
        if not units:
            units = vinfo.get('units', '')
    else:
        cmap = 'miub2'
        cmap_name = 'miub2'
        if data_array is not None:
            vals = data_array.values
            valid_vals = vals[np.isfinite(vals)]
            if len(valid_vals) > 0:
                vmin = float(np.percentile(valid_vals, 2))
                vmax = float(np.percentile(valid_vals, 98))
                if vmin >= vmax:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

    return {
        'cmap': cmap,
        'cmap_name': cmap_name,
        'vmin': vmin,
        'vmax': vmax,
        'units': units
    }


class QVPBokehApp:
    """
    Interactive Bokeh application for Quasi-Vertical Profiles (QVP).
    """
    def __init__(self, dataset_mgr, doc, initial_date=None):
        self.mgr = dataset_mgr
        self.doc = doc
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
            c_name = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
            self.col1_settings.append({
                'var': vname,
                'cmap_name': c_name,
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
                'linestyle': LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)],
                'xmin': vis['vmin'],
                'xmax': vis['vmax'],
                'units': vis['units']
            }

        # Bokeh plot references & data sources
        self.col1_plots = []
        self.col1_images = []
        self.col1_sources = []
        self.col1_mappers = []
        self.col1_colorbars = []
        self.col1_vlines = []
        self.col1_ml_renderers = []
        self.col1_entropy_renderers = []

        self.show_ml = True
        self.source_ml = ColumnDataSource(data=dict(time=[], ml_top=[], ml_bot=[]))
        self.update_ml_source()

        self.show_entropy = True
        self.source_entropy = ColumnDataSource(data=dict(image=[np.zeros((10, 10))], x=[0], y=[0], dw=[1], dh=[1]))
        self.entropy_mapper = LinearColorMapper(
            palette=['#505050'],
            low=0.0,
            high=1.0,
            nan_color='rgba(0,0,0,0)'
        )
        self.update_entropy_source()

        self.source_1d = ColumnDataSource(data=dict(z=[]))
        self.p_prof = None
        self.prof_layout = column([], width=525)

        # Build app layout
        self._build_ui()
        self.update_all()

    def _extract_ml_series(self, ds):
        time_vals = pd.to_datetime(ds.time.values)
        t_ms = (time_vals.astype('int64') // 10**6).values

        y_top = np.full(len(t_ms), np.nan)
        for ml_var in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
            if ml_var in ds:
                vals = np.asarray(ds[ml_var].values, dtype=float).squeeze()
                if vals.ndim == 0 and np.isfinite(vals):
                    y_top = np.full(len(t_ms), float(vals))
                    break
                elif vals.ndim == 1 and np.any(np.isfinite(vals)):
                    y_top = vals
                    break

        y_bot = np.full(len(t_ms), np.nan)
        for ml_bot_var in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
            if ml_bot_var in ds:
                vals = np.asarray(ds[ml_bot_var].values, dtype=float).squeeze()
                if vals.ndim == 0 and np.isfinite(vals):
                    y_bot = np.full(len(t_ms), float(vals))
                    break
                elif vals.ndim == 1 and np.any(np.isfinite(vals)):
                    y_bot = vals
                    break

        return t_ms, y_top, y_bot

    def update_ml_source(self):
        t_ms, y_top, y_bot = self._extract_ml_series(self.current_ds)
        self.source_ml.data = dict(time=t_ms, ml_top=y_top, ml_bot=y_bot)

    def toggle_melting_layer(self):
        self.show_ml = not self.show_ml
        if self.show_ml:
            self.btn_toggle_ml.label = "Melting Layer: ON"
            self.btn_toggle_ml.button_type = "primary"
        else:
            self.btn_toggle_ml.label = "Melting Layer: OFF"
            self.btn_toggle_ml.button_type = "default"

        for r_top, r_bot in self.col1_ml_renderers:
            r_top.visible = self.show_ml
            r_bot.visible = self.show_ml

        self._build_column2_plot()

    def _extract_entropy_mask(self, ds):
        """
        Extract 2D boolean mask where min_entropy > 0.85 and compute its spatial bounds.
        Returns (mask, x0, y0, dw, dh) where mask is (z, time) with 1.0 for min_entropy > 0.85 and nan elsewhere.
        """
        ent_var = None
        for cand in ['min_entropy', 'MIN_ENTROPY', 'min_Entropy', 'entropy_min']:
            if cand in ds:
                ent_var = cand
                break
        if ent_var is None:
            for v in ds.data_vars:
                if 'entropy' in v.lower():
                    ent_var = v
                    break

        z_coords = self.mgr.get_z_coords(ds)
        time_vals = pd.to_datetime(ds.time.values)
        t_ms = (time_vals.astype('int64') // 10**6).values

        if len(t_ms) > 1:
            dt = float(t_ms[1] - t_ms[0])
            x0 = float(t_ms[0]) - dt / 2.0
            dw = float(t_ms[-1] - t_ms[0]) + dt
        elif len(t_ms) == 1:
            dw = 300000.0
            x0 = float(t_ms[0]) - dw / 2.0
        else:
            x0, dw = 0.0, 1.0

        y0 = float(np.nanmin(z_coords))
        dh = float(np.nanmax(z_coords) - y0)

        if ent_var is None:
            mask = np.full((len(z_coords), max(1, len(t_ms))), np.nan)
            return mask, x0, y0, dw, dh

        da = ds[ent_var]
        if 'time' in da.dims and 'z' in da.dims:
            if da.dims != ('time', 'z'):
                da = da.transpose('time', 'z')
            try:
                da = da.interpolate_na(dim='z', method='linear', limit=2)
            except Exception:
                pass
            vals = da.values  # (time, z)
            mask = np.where(vals.T > 0.85, 1.0, np.nan)
        else:
            raw = np.asarray(da.values).squeeze()
            if raw.ndim == 2:
                if raw.shape == (len(t_ms), len(z_coords)):
                    mask = np.where(raw.T > 0.85, 1.0, np.nan)
                elif raw.shape == (len(z_coords), len(t_ms)):
                    mask = np.where(raw > 0.85, 1.0, np.nan)
                else:
                    mask = np.full((len(z_coords), max(1, len(t_ms))), np.nan)
            else:
                mask = np.full((len(z_coords), max(1, len(t_ms))), np.nan)

        return mask, x0, y0, dw, dh

    def update_entropy_source(self):
        mask, x0, y0, dw, dh = self._extract_entropy_mask(self.current_ds)
        self.source_entropy.data = dict(image=[mask], x=[x0], y=[y0], dw=[dw], dh=[dh])

    def toggle_entropy(self):
        self.show_entropy = not self.show_entropy
        if self.show_entropy:
            self.btn_toggle_entropy.label = "Entropy: ON"
            self.btn_toggle_entropy.button_type = "primary"
        else:
            self.btn_toggle_entropy.label = "Entropy: OFF"
            self.btn_toggle_entropy.button_type = "default"

        for r_ent in self.col1_entropy_renderers:
            r_ent.visible = self.show_entropy

    def _build_ui(self):
        # 1. Top Bar Controls: Date Selector, Height Range, ML Toggle
        self.sel_date = Select(title="Date:", value=self.current_date, options=self.dates, width=115)
        self.sel_date.on_change('value', lambda attr, old, new: self.on_date_changed(new))

        self.btn_prev_date = Button(label="◀ Prev", width=55, margin=(20, 2, 0, 2))
        self.btn_next_date = Button(label="Next ▶", width=55, margin=(20, 2, 0, 2))
        self.btn_prev_date.on_click(lambda: self.step_date(-1))
        self.btn_next_date.on_click(lambda: self.step_date(1))

        if len(self.dates) <= 1:
            self.sel_date.disabled = True
            self.btn_prev_date.disabled = True
            self.btn_next_date.disabled = True

        self.inp_min_h = NumericInput(title="Min H:", value=self.min_height, width=75)
        self.inp_max_h = NumericInput(title="Max H:", value=self.max_height, width=75)
        self.btn_apply_h = Button(label="Apply", button_type="primary", width=75, margin=(20, 2, 0, 2))
        self.btn_apply_h.on_click(self.apply_height_range)

        self.btn_toggle_ml = Button(label="Melting Layer: ON", button_type="primary", width=235, margin=(4, 2, 2, 2))
        self.btn_toggle_ml.on_click(self.toggle_melting_layer)

        self.btn_toggle_entropy = Button(
            label="Entropy: ON" if self.show_entropy else "Entropy: OFF",
            button_type="primary" if self.show_entropy else "default",
            width=235, margin=(2, 2, 4, 2)
        )
        self.btn_toggle_entropy.on_click(self.toggle_entropy)

        date_controls = row(self.btn_prev_date, self.sel_date, self.btn_next_date)
        height_controls = row(self.inp_min_h, self.inp_max_h, self.btn_apply_h)

        # 2. Sidebar Tabs
        # Tab 1: 2D Panels (1-4) controls
        panel_widgets = []
        self.panel_ui = []
        for i in range(4):
            lbl = Div(text=f"<b>Panel {i+1} Configuration</b>", margin=(6, 0, 2, 0))
            st = self.col1_settings[i]
            sel_v = Select(title="Variable:", value=st['var'], options=self.all_vars, width=235)
            sel_c = Select(title="Colormap:", value=st['cmap_name'] if st['cmap_name'] in CMAP_OPTIONS else 'miub2', options=CMAP_OPTIONS, width=235)
            inp_vmin = NumericInput(title="vmin:", value=st['vmin'], width=110)
            inp_vmax = NumericInput(title="vmax:", value=st['vmax'], width=110)
            btn_apply = Button(label="Apply Changes", button_type="success", width=110)
            btn_reset = Button(label="Reset Defaults", width=110)

            idx_cap = i
            sel_v.on_change('value', lambda attr, old, new, idx=idx_cap: self.on_panel_var_selected(idx, new))
            btn_apply.on_click(lambda idx=idx_cap: self.apply_panel_config(idx))
            btn_reset.on_click(lambda idx=idx_cap: self.reset_panel_defaults(idx))

            self.panel_ui.append({
                'var': sel_v, 'cmap': sel_c, 'vmin': inp_vmin, 'vmax': inp_vmax
            })
            panel_widgets.extend([lbl, sel_v, sel_c, row(inp_vmin, inp_vmax), row(btn_apply, btn_reset)])

        tab1_layout = column(panel_widgets, width=245)
        tab1 = TabPanel(child=tab1_layout, title="2D Panels (1-4)")

        # Tab 2: 1D Profile Variables checkboxes
        active_indices = [self.all_vars.index(v) for v in self.active_1d_vars if v in self.all_vars]
        self.chk_1d = CheckboxGroup(labels=self.all_vars, active=active_indices, width=235)
        self.chk_1d.on_change('active', lambda attr, old, new: self.on_checkbox_change(new))

        btn_sel_all = Button(label="Select All", width=70)
        btn_clr_all = Button(label="Clear All", width=70)
        btn_sync = Button(label="Sync Col 1", button_type="primary", width=85)
        btn_sel_all.on_click(self.select_all_1d_vars)
        btn_clr_all.on_click(self.clear_all_1d_vars)
        btn_sync.on_click(self.sync_1d_with_col1)

        tab2_layout = column(
            Div(text="<b>Active Variables in 1D Profile:</b>"),
            row(btn_sel_all, btn_clr_all, btn_sync),
            self.chk_1d,
            width=245
        )
        tab2 = TabPanel(child=tab2_layout, title="1D Variables")

        # Tab 3: Styles & Ranges for active variables
        self.style_controls_box = column([], width=245)
        tab3 = TabPanel(child=self.style_controls_box, title="1D Styles & Ranges")

        tabs_sidebar = Tabs(tabs=[tab1, tab2, tab3], width=250)

        # 3. Column 1: 4 2D Heatmaps (linked x_range & y_range for zoom/pan)
        self._build_column1_plots()

        # 4. Column 2: 1D QVP Profile
        self._build_column2_plot()

        # Overall Layout
        sidebar_column = column(
            Div(text="<h4 style='margin:0 0 6px 0; color:#1a365d;'>QVP Navigation & Settings</h4>"),
            date_controls,
            height_controls,
            self.btn_toggle_ml,
            self.btn_toggle_entropy,
            Div(text="<hr style='margin:6px 0;'>"),
            tabs_sidebar,
            width=255
        )

        col1_layout = column(self.col1_plots, width=930)

        self.app_layout = row(
            sidebar_column,
            col1_layout,
            self.prof_layout,
            spacing=14
        )
        self.doc.add_root(self.app_layout)
        self.doc.title = f"Interactive QVP Visualizer - {self.current_date}"

    def on_panel_var_selected(self, panel_idx, new_var):
        if not new_var or new_var not in self.current_ds:
            return
        vis = get_var_vis_info(new_var, self.current_ds[new_var])
        ui = self.panel_ui[panel_idx]
        ui['vmin'].value = vis['vmin']
        ui['vmax'].value = vis['vmax']
        c_name = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
        ui['cmap'].value = c_name

    def _build_column1_plots(self):
        self.col1_plots.clear()
        self.col1_images.clear()
        self.col1_sources.clear()
        self.col1_mappers.clear()
        self.col1_colorbars.clear()
        self.col1_vlines.clear()
        self.col1_ml_renderers.clear()
        self.col1_entropy_renderers.clear()

        shared_x = None
        shared_y = None

        for i in range(4):
            st = self.col1_settings[i]
            vname = st['var']
            vis = st['vis']

            palette, low, high, low_color, high_color, bar_ticks, overrides = get_color_mapper_params(
                st['cmap_name'], st['vmin'], st['vmax'], var_name=vname
            )
            mapper = LinearColorMapper(
                palette=palette, low=low, high=high,
                low_color=low_color, high_color=high_color,
                nan_color='rgba(0,0,0,0)'
            )
            self.col1_mappers.append(mapper)

            unit_str = f" [{vis['units']}]" if vis['units'] else ""
            tools = "pan,box_zoom,wheel_zoom,reset,tap"

            h_plot = 270 if i == 3 else (250 if i == 0 else 245)
            if i == 0:
                p = figure(
                    title=f"Panel 1: {vname}{unit_str}",
                    x_axis_type="datetime",
                    tools=tools,
                    width=920, height=h_plot,
                    toolbar_location="above",
                    active_drag="pan", active_scroll="wheel_zoom"
                )
                shared_x = p.x_range
                shared_y = p.y_range
                p.y_range.start = self.min_height
                p.y_range.end = self.max_height
            else:
                p = figure(
                    title=f"Panel {i+1}: {vname}{unit_str}",
                    x_axis_type="datetime",
                    x_range=shared_x,
                    y_range=shared_y,
                    tools=tools,
                    width=920, height=h_plot,
                    toolbar_location="above",
                    active_drag="pan", active_scroll="wheel_zoom"
                )

            p.yaxis.axis_label = "Height (m)"
            p.title.text_font_size = "10.5pt"
            p.title.text_font_style = "bold"
            p.xaxis.axis_label = "Time (UTC)" if i == 3 else ""

            # Data Source & Image
            src = ColumnDataSource(data=dict(image=[np.zeros((10, 10))], x=[0], y=[0], dw=[1], dh=[1]))
            img_renderer = p.image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=mapper, source=src)

            # Hover Tool: Real-time value overlay directly below cursor!
            hover = HoverTool(
                renderers=[img_renderer],
                tooltips=[
                    ("Time", "$x{%Y-%m-%d %H:%M}"),
                    ("Height", "$y{0} m"),
                    ("Value", f"@image{{0.00}}{unit_str}")
                ],
                formatters={"$x": "datetime"},
                mode="mouse"
            )
            p.add_tools(hover)

            # Gray shading overlay for min_entropy > 0.85
            r_ent = p.image(
                image='image', x='x', y='y', dw='dw', dh='dh',
                color_mapper=self.entropy_mapper,
                source=self.source_entropy,
                global_alpha=0.45,
                name="Entropy Shading"
            )
            r_ent.visible = self.show_entropy
            self.col1_entropy_renderers.append(r_ent)

            # ColorBar on the right
            ticker = FixedTicker(ticks=list(bar_ticks)) if bar_ticks else BasicTicker()
            cbar = ColorBar(
                color_mapper=mapper, title=unit_str, width=14, padding=4, label_standoff=4,
                ticker=ticker, major_label_overrides=dict(overrides) if overrides else {}
            )
            p.add_layout(cbar, 'right')
            self.col1_colorbars.append(cbar)

            # Red vertical line indicating selected profile
            vline = Span(location=0, dimension='height', line_color='red', line_dash='dashed', line_width=1.8, line_alpha=0.9)
            p.add_layout(vline)
            self.col1_vlines.append(vline)

            # Tap listener on 2D plot to select profile timestamp
            p.on_event(Tap, self.on_tap_col1)

            # Melting Layer Lines (Top: dashed, Bottom: dotted)
            r_top = p.line(
                x='time', y='ml_top', source=self.source_ml,
                line_color='black', line_dash='dashed', line_width=1.8,
                line_alpha=0.85, name="ML Top"
            )
            r_bot = p.line(
                x='time', y='ml_bot', source=self.source_ml,
                line_color='black', line_dash='dotted', line_width=1.8,
                line_alpha=0.85, name="ML Bottom"
            )
            r_top.visible = self.show_ml
            r_bot.visible = self.show_ml
            self.col1_ml_renderers.append((r_top, r_bot))

            self.col1_plots.append(p)
            self.col1_sources.append(src)
            self.col1_images.append(img_renderer)

    def _build_column2_plot(self):
        """
        Build the 1D Profile plot with multi-axis bottom layout and real-time hover tooltips.
        """
        if self.current_time_idx < len(self.current_ds.time):
            t_val = self.current_ds.time.values[self.current_time_idx]
            t_str = pd.to_datetime(t_val).strftime('%Y-%m-%d %H:%M:%S')
        else:
            t_str = "No Selection"

        tools = "pan,box_zoom,wheel_zoom,reset,hover"
        p = figure(
            title=f"QVP Profile: {t_str}",
            y_range=Range1d(start=self.min_height, end=self.max_height),
            tools=tools,
            width=520, height=1015,
            toolbar_location="above",
            active_drag="pan", active_scroll="wheel_zoom"
        )
        p.yaxis.axis_label = "Height (m)"
        p.yaxis.axis_label_text_font_style = "bold"
        p.title.text_font_size = "11pt"
        p.title.text_font_style = "bold"
        p.grid.grid_line_dash = "dotted"

        if not self.active_1d_vars or self.current_time_idx >= len(self.current_ds.time):
            p.xaxis.axis_label = "No variable selected"
            self.p_prof = p
            self.prof_layout.children = [p]
            return

        ds_prof = self.current_ds.isel(time=self.current_time_idx)
        z_coords = self.mgr.get_z_coords(self.current_ds)
        z_mask = (z_coords >= self.min_height) & (z_coords <= self.max_height)

        # Populate source data
        src_dict = {'z': z_coords[z_mask]}
        for v in self.active_1d_vars:
            da = ds_prof[v]
            if 'time' in da.dims:
                da = da.squeeze()
            vals = np.squeeze(da.values)
            src_dict[v] = vals[z_mask]
        self.source_1d.data = src_dict

        # Variable 0 on primary bottom axis
        v0 = self.active_1d_vars[0]
        st0 = self.var_styles[v0]
        unit_str0 = f" [{st0['units']}]" if st0['units'] else ""
        p.x_range = Range1d(start=st0['xmin'], end=st0['xmax'])
        p.xaxis.axis_label = f"{v0}{unit_str0}"
        p.xaxis.axis_label_text_color = st0['color']
        p.xaxis.major_label_text_color = st0['color']
        p.xaxis.axis_line_color = st0['color']
        p.xaxis.major_tick_line_color = st0['color']
        p.xaxis.axis_label_text_font_style = "bold"

        r0 = p.line(x=v0, y='z', source=self.source_1d, color=st0['color'],
                    line_dash=st0['linestyle'], line_width=2.5, name=v0)

        # Subsequent variables on extra bottom axes
        renderers = [r0]
        for v in self.active_1d_vars[1:]:
            st = self.var_styles[v]
            unit_str = f" [{st['units']}]" if st['units'] else ""

            p.extra_x_ranges[v] = Range1d(start=st['xmin'], end=st['xmax'])
            ax_twin = LinearAxis(
                x_range_name=v,
                axis_label=f"{v}{unit_str}",
                axis_label_text_color=st['color'],
                major_label_text_color=st['color'],
                axis_line_color=st['color'],
                major_tick_line_color=st['color'],
                axis_label_text_font_style="bold"
            )
            p.add_layout(ax_twin, 'below')

            r = p.line(x=v, y='z', source=self.source_1d, color=st['color'],
                       line_dash=st['linestyle'], line_width=2.5, x_range_name=v, name=v)
            renderers.append(r)

        # Hover Tool for 1D profile
        hover = p.select_one(HoverTool)
        if hover:
            hover.renderers = renderers
            hover.tooltips = [
                ("Variable", "$name"),
                ("Value", "$x{0.00}"),
                ("Height", "$y{0} m")
            ]
            hover.mode = "mouse"

        # Melting Layer (ML) lines on 1D profile
        if self.show_ml:
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
                p.add_layout(Span(location=ml_top, dimension='width', line_color='black', line_dash='dashed', line_width=1.5))
                p.add_layout(Label(x=10, y=ml_top, x_units='screen', text=f"ML Top ({int(ml_top)}m)",
                                   text_font_size="8pt", text_font_style="bold", text_color="black"))

            if ml_bot is not None and self.min_height <= ml_bot <= self.max_height:
                p.add_layout(Span(location=ml_bot, dimension='width', line_color='black', line_dash='dotted', line_width=1.5))
                p.add_layout(Label(x=10, y=ml_bot, x_units='screen', text=f"ML Bot ({int(ml_bot)}m)",
                                   text_font_size="8pt", text_font_style="bold", text_color="black"))

        self.p_prof = p
        self.prof_layout.children = [p]

    def update_all(self):
        for i in range(4):
            self.update_column1_panel(i)
        self.update_vlines()
        self._build_column2_plot()
        self._refresh_styles_tab()

    def update_column1_panel(self, i):
        st = self.col1_settings[i]
        vname = st['var']
        vis = st['vis']

        da = self.current_ds[vname]
        if da.dims != ('time', 'z') and 'time' in da.dims and 'z' in da.dims:
            da = da.transpose('time', 'z')

        z_coords = self.mgr.get_z_coords(self.current_ds)
        time_vals = pd.to_datetime(self.current_ds.time.values)
        t_ms = (time_vals.astype('int64') // 10**6).values

        # Image 2D array: rows = height, cols = time
        c_data = da.values.T

        if len(t_ms) > 1:
            dt = float(t_ms[1] - t_ms[0])
            x0 = float(t_ms[0]) - dt / 2.0
            dw = float(t_ms[-1] - t_ms[0]) + dt
        elif len(t_ms) == 1:
            dw = 300000.0
            x0 = float(t_ms[0]) - dw / 2.0
        else:
            x0, dw = 0.0, 1.0

        y0 = float(np.nanmin(z_coords))
        dh = float(np.nanmax(z_coords) - y0)

        # Update data source
        self.col1_sources[i].data = dict(
            image=[c_data],
            x=[x0],
            y=[y0],
            dw=[dw],
            dh=[dh]
        )

        # Update color mapper & colorbar
        palette, low, high, low_color, high_color, bar_ticks, overrides = get_color_mapper_params(
            st['cmap_name'], st['vmin'], st['vmax'], var_name=vname
        )
        self.col1_mappers[i].palette = palette
        self.col1_mappers[i].low = low
        self.col1_mappers[i].high = high
        self.col1_mappers[i].low_color = low_color
        self.col1_mappers[i].high_color = high_color

        unit_str = f" [{vis['units']}]" if vis['units'] else ""
        self.col1_plots[i].title.text = f"Panel {i+1}: {vname}{unit_str}"

        # 1. Update existing colorbar model in-place so listeners trigger
        cbar = self.col1_colorbars[i]
        if hasattr(cbar, 'ticker') and hasattr(cbar.ticker, 'ticks') and bar_ticks:
            cbar.ticker.ticks = list(bar_ticks)
        if overrides is not None:
            cbar.major_label_overrides = dict(overrides)
        cbar.title = unit_str

        # 2. Swap ColorBar layout on plot to force full BokehJS layout and ticker re-creation
        p = self.col1_plots[i]
        p.right = [r for r in p.right if r != cbar]
        ticker = FixedTicker(ticks=list(bar_ticks)) if bar_ticks else BasicTicker()
        new_cbar = ColorBar(
            color_mapper=self.col1_mappers[i],
            title=unit_str,
            width=14,
            padding=4,
            label_standoff=4,
            ticker=ticker,
            major_label_overrides=dict(overrides) if overrides else {}
        )
        p.add_layout(new_cbar, 'right')
        self.col1_colorbars[i] = new_cbar

    def update_vlines(self):
        time_vals = pd.to_datetime(self.current_ds.time.values)
        t_ms = (time_vals.astype('int64') // 10**6).values
        if 0 <= self.current_time_idx < len(t_ms):
            t_sel = float(t_ms[self.current_time_idx])
            for vline in self.col1_vlines:
                vline.location = t_sel

    def on_tap_col1(self, event):
        """
        Handle tap / click on 2D heatmaps to select the profile timestamp.
        """
        clicked_ms = event.x
        time_vals = pd.to_datetime(self.current_ds.time.values)
        t_ms = (time_vals.astype('int64') // 10**6).values
        idx = int(np.argmin(np.abs(t_ms - clicked_ms)))
        self.current_time_idx = idx
        self.update_vlines()
        self._build_column2_plot()

    def on_date_changed(self, new_date):
        if new_date in self.dates and new_date != self.current_date:
            self.current_date = new_date
            self.doc.title = f"Interactive QVP Visualizer - {self.current_date}"
            self.current_ds = self.mgr.load_day(self.current_date)
            self.update_ml_source()
            self.update_entropy_source()
            self.all_vars = self.mgr.get_available_variables(self.current_ds)
            if self.current_time_idx >= len(self.current_ds.time):
                self.current_time_idx = max(0, len(self.current_ds.time) - 1)
            self.update_all()

    def step_date(self, delta):
        idx = self.dates.index(self.current_date) + delta
        if 0 <= idx < len(self.dates):
            self.sel_date.value = self.dates[idx]

    def apply_height_range(self):
        min_v = float(self.inp_min_h.value)
        max_v = float(self.inp_max_h.value)
        if min_v >= max_v or min_v < 0:
            return
        self.min_height = min_v
        self.max_height = max_v

        # Update 2D panel y-range
        if self.col1_plots:
            self.col1_plots[0].y_range.start = self.min_height
            self.col1_plots[0].y_range.end = self.max_height

        self._build_column2_plot()

    def apply_panel_config(self, panel_idx):
        ui = self.panel_ui[panel_idx]
        vname = ui['var'].value
        cmap_name = ui['cmap'].value
        vmin = float(ui['vmin'].value)
        vmax = float(ui['vmax'].value)

        vis = get_var_vis_info(vname, self.current_ds[vname])
        vis['cmap_name'] = cmap_name
        vis['vmin'] = vmin
        vis['vmax'] = vmax

        self.col1_settings[panel_idx] = {
            'var': vname,
            'cmap_name': cmap_name,
            'vmin': vmin,
            'vmax': vmax,
            'vis': vis
        }

        if vname not in self.active_1d_vars:
            self.active_1d_vars.append(vname)
            if vname in self.all_vars:
                active_set = set(self.chk_1d.active)
                active_set.add(self.all_vars.index(vname))
                self.chk_1d.active = sorted(list(active_set))

        self.update_column1_panel(panel_idx)
        self._build_column2_plot()
        self._refresh_styles_tab()

    def reset_panel_defaults(self, panel_idx):
        ui = self.panel_ui[panel_idx]
        vname = ui['var'].value
        vis = get_var_vis_info(vname, self.current_ds[vname])
        c_choice = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
        self.col1_settings[panel_idx] = {
            'var': vname,
            'cmap_name': c_choice,
            'vmin': vis['vmin'],
            'vmax': vis['vmax'],
            'vis': vis
        }
        ui['cmap'].value = c_choice
        ui['vmin'].value = vis['vmin']
        ui['vmax'].value = vis['vmax']

        self.update_column1_panel(panel_idx)

    def on_checkbox_change(self, active_indices):
        self.active_1d_vars = [self.all_vars[idx] for idx in active_indices if idx < len(self.all_vars)]
        self._build_column2_plot()
        self._refresh_styles_tab()

    def select_all_1d_vars(self):
        self.chk_1d.active = list(range(len(self.all_vars)))
        self.active_1d_vars = list(self.all_vars)
        self._build_column2_plot()
        self._refresh_styles_tab()

    def clear_all_1d_vars(self):
        self.chk_1d.active = []
        self.active_1d_vars = []
        self._build_column2_plot()
        self._refresh_styles_tab()

    def sync_1d_with_col1(self):
        col1_v = [s['var'] for s in self.col1_settings]
        active_indices = [self.all_vars.index(v) for v in col1_v if v in self.all_vars]
        self.chk_1d.active = active_indices
        self.active_1d_vars = list(col1_v)
        self._build_column2_plot()
        self._refresh_styles_tab()

    def _refresh_styles_tab(self):
        widgets = [Div(text="<b>Active Curve Range & Style:</b>", margin=(4, 0, 4, 0))]
        for v in self.active_1d_vars:
            st = self.var_styles[v]
            lbl = Div(text=f"<b>{v}</b>", margin=(6, 0, 2, 0))
            inp_xmin = NumericInput(title="xmin:", value=st['xmin'], width=110)
            inp_xmax = NumericInput(title="xmax:", value=st['xmax'], width=110)
            sel_dash = Select(title="Line Style:", value=st['linestyle'], options=LINE_STYLES, width=110)
            sel_col = Select(title="Color:", value=st['color'], options=DEFAULT_PALETTE, width=110)

            def make_handler(var_name, xmin_w, xmax_w, dash_w, col_w):
                def apply_style():
                    self.var_styles[var_name]['xmin'] = float(xmin_w.value)
                    self.var_styles[var_name]['xmax'] = float(xmax_w.value)
                    self.var_styles[var_name]['linestyle'] = dash_w.value
                    self.var_styles[var_name]['color'] = col_w.value
                    self._build_column2_plot()
                return apply_style

            handler = make_handler(v, inp_xmin, inp_xmax, sel_dash, sel_col)
            btn_upd = Button(label=f"Update {v}", button_type="primary", width=225)
            btn_upd.on_click(handler)

            widgets.extend([lbl, row(inp_xmin, inp_xmax), row(sel_col, sel_dash), btn_upd])

        self.style_controls_box.children = widgets


# ==============================================================================
# Main Entry Point & Server Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Quasi-Vertical Profile (QVP) Visualizer (Bokeh Backend)."
    )
    parser.add_argument(
        "paths", nargs="*", default=[DEFAULT_PATH_QVPS],
        help=f"File path(s), directory, or glob pattern to QVP data (default: {DEFAULT_PATH_QVPS})"
    )
    parser.add_argument(
        "--port", type=int, default=5006,
        help="Port to run the Bokeh server on (default: 5006)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not automatically open the web browser."
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

    # Launch Bokeh Server
    try:
        from bokeh.server.server import Server
        from tornado.ioloop import IOLoop
    except ImportError:
        print("Error: Bokeh and Tornado must be installed to run the interactive server.")
        print("Please run within an environment with Bokeh installed (e.g. conda activate wradlib7).")
        sys.exit(1)

    def bkapp(doc):
        QVPBokehApp(mgr, doc)

    # Find free port starting at requested port
    port = args.port
    server = None
    for p in range(port, port + 20):
        try:
            server = Server({'/': bkapp}, io_loop=IOLoop.current(), port=p, allow_websocket_origin=['*'])
            port = p
            break
        except OSError:
            continue

    if server is None:
        server = Server({'/': bkapp}, io_loop=IOLoop.current(), port=0, allow_websocket_origin=['*'])
        port = server.port

    print(f"\n=================================================================")
    print(f" Interactive QVP Visualizer running at: http://localhost:{port}/")
    print(f" Features: Native Pan/Zoom, Hover Value Overlays, Profile Selection")
    print(f" Press Ctrl+C in terminal to stop.")
    print(f"=================================================================\n")

    server.start()

    if not args.no_browser:
        server.io_loop.add_callback(server.show, "/")

    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print("\nStopping Bokeh server...")
        server.stop()


# If executed via `bokeh serve plot_qvps_interactive_bokeh.py --show --args [paths]`
if __name__.startswith('bokeh_app'):
    paths = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_PATH_QVPS]
    files = find_files_from_input(paths)
    mgr = QVPDatasetManager(files)
    QVPBokehApp(mgr, curdoc())
elif __name__ == "__main__":
    main()
