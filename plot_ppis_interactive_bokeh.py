#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Plan Position Indicator (PPI) Visualizer (Bokeh Backend)

Provides a browser-based interactive GUI with native Zoom, Pan, and real-time
Hover Value Overlays directly under the cursor.

Layout:
- Left Sidebar: Date, Timestep (slider + selector), Elevation, Max Range,
  Melting Layer / Entropy / Riming toggles (auto-disabled when unavailable),
  and configuration tabs for the 3 PPI panels and radial line plots.
- Row 1 (Top): 3 linked polar-coordinate PPI subplots (default: DBZH, ZDR, KDP)
  with range rings, azimuth spokes, independent variables/colormaps/colorbars,
  optional ML/Entropy/Riming overlays, and click-to-select radial line.
- Row 2 (Bottom): 1 wide multi-axis radial line-plot subplot (default: DBZH,
  ZDR, KDP, RHOHV) along the selected azimuth, with optional ML/Entropy/Riming
  indicators and customizable curve ranges, colors, and line styles.

Usage:
    python plot_ppis_interactive_bokeh.py [path_or_glob_or_dir ...] [--port 5006] [--no-browser]
    bokeh serve plot_ppis_interactive_bokeh.py --show --args [paths...]
"""

import os
os.environ["ICECHUNK_NO_LOGS"] = "1"
import sys
import glob
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, HoverTool,
    CustomJSHover, Range1d, LinearAxis, Span, Label, BoxAnnotation,
    Select, Button, NumericInput, CheckboxGroup, Div, Slider,
    Tabs, TabPanel, FixedTicker, BasicTicker
)
from bokeh.layouts import column, row
from bokeh.events import Tap

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")

try:
    from icechunk import Repository, local_filesystem_storage
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

try:
    import wradlib as wrl
    HAS_WRADLIB = True
except ImportError:
    wrl = None
    HAS_WRADLIB = False

# Local project scripts directory setup
possible_paths = [
    '/home/jgiles/Scripts/python/radar_processing_scripts',
    '/p/scratch/detectrea2/giles1/radar_processing_scripts',
]
for script_dir in possible_paths:
    if os.path.exists(script_dir) and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        break

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

# Default path config
DEFAULT_REALPEP_PATH = "/automount/realpep"
DEFAULT_PATH_PPIS = f"{DEFAULT_REALPEP_PATH}/upload/jgiles/dwd/*/*/2017-07-25/pro/vol5minng01/07/*allmoms*"
DEFAULT_RIMING_DIR = f"{DEFAULT_REALPEP_PATH}/upload/jgiles/radar_riming_classif/unfiltered"

# Available colormap names (miub2 as top default)
CMAP_OPTIONS = [
    'miub2', 'default (visdict)', 'HomeyerRainbow', 'viridis', 'SpectralExtended',
    'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'RdBu_r', 'coolwarm', 'binary2'
]

# Distinct palette for 1D radial curves
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

KNOWN_LOCATIONS = ['pro', 'umd', 'tur', 'afy', 'ank', 'gzt', 'hty', 'svs', 'ess', 'bo']
DWD_STATION_IDS = {
    '10392': 'pro',
    '010392': 'pro',
    '10780': 'tur',
    '010780': 'tur',
    '10629': 'umd',
    '010629': 'umd',
    '10410': 'ess',
    '010410': 'ess',
    '17373': 'hty',
    '017373': 'hty',
}

# Metadata/non-plottable variable names to exclude from PPI moment selection
NON_MOMENT_VARS = {
    'sweep_number', 'prt_mode', 'follow_mode', 'sweep_fixed_angle',
    'sweep_mode', 'ray_azimuth', 'ray_elevation', 'forecast_seconds',
    'forecast_minutes', 'crs_wkt', 'rtime', 'rays', 'bins',
    'latitude', 'longitude', 'altitude', 'elevation', 'x', 'y', 'z', 'gr'
}


# ==============================================================================
# Colormap & Colorbar Helpers (reused from plot_qvps_interactive_bokeh.py)
# ==============================================================================

def get_bokeh_palette(cmap_name_or_obj, n=256):
    """
    Convert a colormap name or matplotlib colormap object into a list of hex strings for Bokeh.
    """
    if cmap_name_or_obj in ['miub2', 'default (visdict)', 'colors_prabhakar']:
        return [mcolors.to_hex(c) for c in colors_prabhakar]
    if cmap_name_or_obj == 'binary2':
        return [mcolors.to_hex('dimgray'), mcolors.to_hex('gainsboro')]

    cmap = None
    if isinstance(cmap_name_or_obj, mpl.colors.Colormap):
        cmap = cmap_name_or_obj
    elif isinstance(cmap_name_or_obj, str):
        try:
            cmap = mpl.colormaps.get_cmap(cmap_name_or_obj)
        except Exception:
            try:
                cmap = plt.get_cmap(cmap_name_or_obj)
            except Exception:
                cmap = None

    if cmap is None:
        return [mcolors.to_hex(c) for c in colors_prabhakar]

    colors = cmap(np.linspace(0, 1, n))
    return [mcolors.to_hex(c) for c in colors]


def get_color_mapper_params(cmap_name, vmin, vmax, var_name=None):
    """
    Compute LinearColorMapper parameters and exact colorbar tick configuration.
    Handles 'miub2' (Prabhakar 14-color discrete palette) with extend='both' semantics,
    supporting both uniform and non-uniform visdict14 ticks.

    Returns:
        palette (list of hex), low (float), high (float),
        low_color (hex), high_color (hex),
        bar_ticks (list of float or None), overrides (dict or None)
    """
    is_miub2 = cmap_name in ['miub2', 'default (visdict)', 'colors_prabhakar']

    def fmt_val(v):
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-4 else f"{v:g}"

    if is_miub2:
        hex_colors = [mcolors.to_hex(c) for c in colors_prabhakar]
        c_under = hex_colors[0]
        c_mid = hex_colors[1:13]
        c_over = hex_colors[13]

        ticks = None
        if var_name is not None and radarmet is not None and hasattr(radarmet, 'visdict14'):
            vd = radarmet.visdict14
            vinfo = None
            if var_name in vd:
                vinfo = vd[var_name]
            else:
                prefix = var_name.split("_")[0]
                for k in [prefix, 'DBZH', 'ZDR', 'KDP', 'RHOHV', 'PHIDP', 'TH']:
                    if k in vd and (k in var_name or var_name.startswith(k)):
                        vinfo = vd[k]
                        break
                if vinfo is None:
                    for k in vd:
                        if k in var_name or var_name in k:
                            vinfo = vd[k]
                            break
            if vinfo is not None and 'ticks' in vinfo:
                ticks = np.array(vinfo['ticks'], dtype=float)

        if ticks is not None and len(ticks) == 13:
            if abs(vmin - ticks[0]) < 1e-4 and abs(vmax - ticks[-1]) < 1e-4:
                diffs = np.diff(ticks)
                if np.allclose(diffs, diffs[0], rtol=1e-3):
                    step = float(diffs[0])
                    low = float(vmin - step)
                    high = float(vmax + step)
                    palette = hex_colors
                    bar_ticks = [low] + [float(t) for t in ticks] + [high]
                    overrides = {
                        low: f"<{fmt_val(vmin)}",
                        str(low): f"<{fmt_val(vmin)}",
                        high: f">{fmt_val(vmax)}",
                        str(high): f">{fmt_val(vmax)}"
                    }
                    for t in ticks:
                        s_val = fmt_val(t)
                        overrides[float(t)] = s_val
                        overrides[str(float(t))] = s_val
                    return palette, low, high, c_under, c_over, bar_ticks, overrides
                else:
                    n_bins = 280
                    pad = (vmax - vmin) / 12.0
                    low = float(vmin - pad)
                    high = float(vmax + pad)
                    x_vals = np.linspace(low, high, n_bins, endpoint=False) + (high - low) / (2.0 * n_bins)
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

    # Continuous colormaps
    palette = get_bokeh_palette(cmap_name, n=256)
    low = vmin
    high = vmax
    low_color = palette[0]
    high_color = palette[-1]

    t_vals = np.linspace(vmin, vmax, 6)
    bar_ticks = [float(v) for v in t_vals]
    overrides = {float(v): fmt_val(v) for v in bar_ticks}
    for v in bar_ticks:
        overrides[str(float(v))] = fmt_val(v)
        if abs(v - round(v)) < 1e-4:
            overrides[float(int(round(v)))] = fmt_val(v)
            overrides[str(int(round(v)))] = fmt_val(v)

    return palette, low, high, low_color, high_color, bar_ticks, overrides


# ==============================================================================
# File Discovery, Location Detection & Riming Helpers
# ==============================================================================

def find_files_from_input(paths_input):
    """
    Given a string path, list of paths, glob pattern, or directory,
    return a sorted list of PPI files (DWD preprocessed/raw, DMI, Icechunk zarr, EMVORADO, BoXPol, etc.).
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
        # 2. Directly specified .zarr store
        elif clean_item.endswith('.zarr'):
            files_found.append(clean_item)
        # 3. Directory containing PPI files
        elif os.path.isdir(item):
            patterns = [
                os.path.join(item, "*allsim_id*"),
                os.path.join(item, "*allmoms*"),
                os.path.join(item, "*allmom*"),
                os.path.join(item, "*allm*"),
                os.path.join(item, "*.zarr"),
                os.path.join(item, "*.hd5"),
                os.path.join(item, "*.h5"),
                os.path.join(item, "*.nc"),
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
                    base = os.path.basename(c_entry)
                    if base in ("DONE.txt", "ML_detected.txt", "DBZH_over_30.txt"):
                        continue
                    if os.path.isfile(c_entry) or c_entry.endswith('.zarr'):
                        files_found.append(c_entry)
        # 4. Directly specified file
        elif os.path.isfile(item):
            files_found.append(item)
        # 5. Direct fallback (e.g. automount path)
        else:
            files_found.append(clean_item)

    # Filter out marker text files and non-volume scans (SURVEILLANCE, WIND, RHI) if mixed with real data
    filtered = []
    for f in sorted(list(set(files_found))):
        c_f = f.rstrip('/')
        base = os.path.basename(c_f)
        if base in ("DONE.txt", "ML_detected.txt", "DBZH_over_30.txt") or base.endswith(".txt"):
            continue
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


def extract_radar_location(fpath_or_ds):
    """
    Extract 3-letter radar site location (e.g. 'pro', 'gzt', 'hty', 'ank', 'afy', 'svs', 'tur', 'umd')
    from a PPI file path, dataset, or filename.
    """
    if isinstance(fpath_or_ds, str):
        clean = fpath_or_ds.lower()
        for sid, loc_name in DWD_STATION_IDS.items():
            if sid in clean:
                return loc_name
        parts = re.split(r'[/\\_\-\.]', clean)
        for loc in KNOWN_LOCATIONS:
            if loc in parts:
                return loc
        for loc in KNOWN_LOCATIONS:
            if re.search(r'[/_\-\.]' + loc + r'[/_\-\.]', clean):
                return loc

    if hasattr(fpath_or_ds, 'attrs'):
        for attr_k in ['radar', 'radar_id', 'location', 'site', 'station_id', 'station_name']:
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


def detect_ppi_file_type(file_list):
    """
    Inspect a file path (or list of paths for one date/elevation) and determine
    which loading strategy to use, mirroring utils.py, compute_qvps_new.py,
    compute_qvps_new_emvorado.py, and plot_ppis_qvps_etc.py.

    Returns one of:
        'icechunk_zarr', 'emvorado_sim', 'dwd_preprocessed', 'dwd_raw',
        'dmi_preprocessed', 'generic'
    """
    if not file_list:
        return 'generic'
    f0 = file_list[0].rstrip('/')
    f0_low = f0.lower()
    base_low = os.path.basename(f0_low)

    # 1. Icechunk / Zarr PPI store (e.g. final_ppis/*.zarr)
    if f0_low.endswith('.zarr'):
        return 'icechunk_zarr'

    # 2. EMVORADO simulation files (*allsim_id* or emvorado folder, non-zarr)
    if 'allsim_id' in base_low or ('emvorado' in f0_low and 'allsim' in base_low and not f0_low.endswith('.zarr')):
        return 'emvorado_sim'

    # 3. DWD files
    if '/dwd/' in f0_low or '/dwd-' in f0_low or 'dwd-cband' in f0_low or 'vol5minng' in f0_low or 'pcpng01' in f0_low or '90gradstarng' in f0_low:
        if 'allm' in base_low:
            return 'dwd_preprocessed'
        else:
            return 'dwd_raw'

    # 4. DMI (Turkish radar) files
    if '/dmi/' in f0_low or any(f'/{s}/' in f0_low for s in ['ank', 'gzt', 'afy', 'svs', 'hty']):
        return 'dmi_preprocessed'

    # 5. Generic / BoXPol / NetCDF / HDF5
    return 'generic'


def extract_elev_folder_label(fpath):
    """
    Extract an elevation folder label (like '07', '10.0', '7.0') if present in parent path.
    """
    parent = os.path.basename(os.path.dirname(fpath.rstrip('/')))
    if re.match(r'^\d+(\.\d+)?$', parent):
        return parent
    return None


# ==============================================================================
# PPI Dataset Manager
# ==============================================================================

class PPIDatasetManager:
    """
    Manages discovery, smart loading, georeferencing, elevation selection,
    and date mapping for PPI radar datasets (DWD obs, DMI obs, Icechunk zarr
    final_ppis, EMVORADO simulations, BoXPol, and generic NetCDF/HDF5).
    """
    def __init__(self, file_list):
        self.file_list = sorted(file_list)
        self.cache = {}  # (date_str, elev_key) -> xr.Dataset (georeferenced single-elevation sweep over time)
        self.date_list = []
        # date_str -> dict(elev_label -> list_of_files)
        self.date_elev_files = {}
        self.date_to_file = {}  # representative filepath per date
        self.is_single_file_multi_day = False
        self.single_file_ds = None
        self._init_date_mapping()

    def _init_date_mapping(self):
        if not self.file_list:
            return

        # Group files by date (and by elevation folder if multiple elevation folders exist on same day)
        date_groups = {}
        for f in self.file_list:
            d_str = extract_date_from_file(f)
            if not d_str:
                d_str = "Dataset"
            date_groups.setdefault(d_str, []).append(f)

        # If a single file was provided, inspect if it contains multiple days
        if len(self.file_list) == 1:
            fp = self.file_list[0]
            try:
                ds_raw = self._load_files_raw([fp])
                ds_raw = self._standardize_ds(ds_raw)
                if 'time' in ds_raw.coords and ds_raw.sizes.get('time', 0) > 1:
                    times = pd.to_datetime(ds_raw.time.values)
                    unique_dates = sorted(list(set(str(t.date()) for t in times if pd.notnull(t))))
                    if len(unique_dates) > 1:
                        self.is_single_file_multi_day = True
                        self.single_file_ds = ds_raw
                        self.date_list = unique_dates
                        for d in unique_dates:
                            self.date_to_file[d] = fp
                            self.date_elev_files[d] = {'default': [fp]}
                        return
                    elif len(unique_dates) == 1:
                        d_str = unique_dates[0]
                        self.date_list = [d_str]
                        self.date_to_file[d_str] = fp
                        self.date_elev_files[d_str] = {'default': [fp]}
                        self.cache[(d_str, 'raw')] = ds_raw
                        return
            except Exception as e:
                print(f"Notice during initial inspection of {fp}: {e}")

        self.date_list = sorted(date_groups.keys())
        for d in self.date_list:
            flist = sorted(date_groups[d])
            self.date_to_file[d] = flist[0]
            ftype = detect_ppi_file_type(flist)

            # Group by elevation subfolder only when multiple elevation folders exist for obs files
            if ftype in ('dwd_preprocessed', 'dmi_preprocessed', 'icechunk_zarr', 'generic') and len(flist) > 1:
                elev_map = {}
                for fp in flist:
                    el = extract_elev_folder_label(fp) or 'default'
                    elev_map.setdefault(el, []).append(fp)
                if len(elev_map) > 1:
                    self.date_elev_files[d] = elev_map
                else:
                    self.date_elev_files[d] = {'default': flist}
            else:
                # For EMVORADO multi-timestep or DWD raw single-moment files, keep all day files together
                self.date_elev_files[d] = {'default': flist}

    def _load_files_raw(self, files):
        """
        Dispatch to the appropriate loader based on file type, location, and obs vs model.
        """
        ftype = detect_ppi_file_type(files)
        f0 = files[0].rstrip('/')
        print(f"Loading PPI ({ftype}): {len(files)} file(s) -> {os.path.basename(f0)}")

        # 1. Icechunk / Zarr PPI
        if ftype == 'icechunk_zarr':
            if utils is not None and hasattr(utils, 'load_icechunk_zarr_ppi'):
                try:
                    return utils.load_icechunk_zarr_ppi(files)
                except Exception as e:
                    print(f"  load_icechunk_zarr_ppi fallback ({e})...")
            if HAS_ICECHUNK:
                try:
                    dsets = []
                    for ff in files:
                        repo = Repository.open(local_filesystem_storage(ff))
                        session = repo.readonly_session("main")
                        dsets.append(xr.open_zarr(session.store, consolidated=False))
                    return dsets[0] if len(dsets) == 1 else xr.concat(dsets, dim='time', join='outer')
                except Exception:
                    pass
            try:
                return xr.open_zarr(f0, consolidated=False)
            except Exception:
                return xr.open_zarr(f0)

        # 2. EMVORADO simulation files (*allsim_id*)
        if ftype == 'emvorado_sim':
            if utils is not None and hasattr(utils, 'load_emvorado_to_radar_volume'):
                ds = utils.load_emvorado_to_radar_volume(files if len(files) > 1 else files[0], rename=True)
                # Compute attenuation-corrected DBZH_AC and ZDR_AC if AHPI/ADPPI present (as in compute_qvps_new_emvorado.py)
                if "AHPI" in ds and "DBZH" in ds and "DBZH_AC" not in ds:
                    ds["DBZH_AC"] = ds["DBZH"] + ds["AHPI"]
                    ds["DBZH_AC"].attrs = dict(ds["DBZH"].attrs)
                if "ADPPI" in ds and "ZDR" in ds and "ZDR_AC" not in ds:
                    ds["ZDR_AC"] = ds["ZDR"] + ds["ADPPI"]
                    ds["ZDR_AC"].attrs = dict(ds["ZDR"].attrs)
                return ds
            return xr.open_mfdataset(files, concat_dim='time', combine='nested', compat='no_conflicts')

        # 3. DWD preprocessed
        if ftype == 'dwd_preprocessed':
            if utils is not None and hasattr(utils, 'load_dwd_preprocessed'):
                return utils.load_dwd_preprocessed(files)
            dt = xr.open_datatree(f0, chunks={})
            return dt.descendants[0].to_dataset() if len(dt.descendants) >= 1 else dt.to_dataset()

        # 4. DWD raw (single-moment ODIM files)
        if ftype == 'dwd_raw':
            if utils is not None and hasattr(utils, 'load_dwd_raw'):
                try:
                    return utils.load_dwd_raw(files)
                except Exception as e:
                    print(f"  load_dwd_raw fallback ({e})...")

        # 5. DMI preprocessed
        if ftype == 'dmi_preprocessed':
            if utils is not None and hasattr(utils, 'load_dmi_preprocessed'):
                try:
                    return utils.load_dmi_preprocessed(files)
                except Exception as e:
                    print(f"  load_dmi_preprocessed fallback ({e})...")

        # 6. Generic / BoXPol / fallback chain
        for opener in (
            lambda: xr.open_dataset(f0) if len(files) == 1 else xr.open_mfdataset(files, compat='no_conflicts'),
            lambda: xr.open_dataset(f0, engine='h5netcdf'),
            lambda: xr.open_dataset(f0, engine='odim'),
            lambda: (lambda dt: dt.descendants[0].to_dataset() if len(dt.descendants) >= 1 else dt.to_dataset())(xr.open_datatree(f0, chunks={})),
        ):
            try:
                return opener()
            except Exception:
                continue

        raise RuntimeError(f"Could not open PPI file(s): {files[:3]}")

    def _standardize_ds(self, ds):
        """
        Standardize dimension and coordinate names to ('time', 'azimuth', 'range')
        (with optional 'sweep_fixed_angle' if multi-elevation volume).
        """
        rename_dims = {}
        if 'azimuth' not in ds.dims:
            for cand in ['n_azimuth', 'az', 'azim', 'ray']:
                if cand in ds.dims:
                    rename_dims[cand] = 'azimuth'
                    break
        if 'range' not in ds.dims:
            for cand in ['n_range', 'rng', 'bin', 'gate']:
                if cand in ds.dims:
                    rename_dims[cand] = 'range'
                    break
        if 'time' not in ds.dims:
            for cand in ['Time', 'datetime', 'valid_time', 'Times']:
                if cand in ds.dims:
                    rename_dims[cand] = 'time'
                    break
        if rename_dims:
            ds = ds.rename_dims(rename_dims)

        # Ensure 'time' dimension exists (e.g. single-timestep EMVORADO file)
        if 'time' not in ds.dims:
            if 'time' in ds.coords:
                t_val = ds.coords['time'].values
                ds = ds.expand_dims({'time': [t_val] if np.ndim(t_val) == 0 else t_val})
            else:
                ds = ds.expand_dims({'time': [np.datetime64('NaT')]})

        # Drop invalid NaT times if any
        if 'time' in ds.coords and ds.sizes.get('time', 0) > 1:
            valid_t = pd.notnull(pd.to_datetime(ds['time'].values))
            if not valid_t.all() and valid_t.any():
                ds = ds.isel(time=valid_t)

        # Sort by time if monotonic check fails
        if 'time' in ds.coords and ds.sizes.get('time', 0) > 1:
            try:
                t_idx = pd.to_datetime(ds['time'].values)
                if not t_idx.is_monotonic_increasing:
                    ds = ds.sortby('time')
            except Exception:
                pass

        return ds

    def _ensure_georeferenced(self, ds):
        """
        Ensure sweep dataset has 'x', 'y', 'z', 'gr' coordinates (in meters).
        Uses wradlib.georef.georeference when available, with an analytical 4/3-Earth fallback.
        """
        if all(c in ds.coords for c in ('x', 'y', 'z')):
            if 'gr' not in ds.coords:
                gr = np.hypot(ds['x'], ds['y'])
                ds = ds.assign_coords(gr=gr)
            return ds

        # Clean scalar site coords that might carry a time dim (prevents wradlib errors)
        for coord in ['latitude', 'longitude', 'altitude', 'elevation']:
            if coord in ds.coords and 'time' in ds[coord].dims:
                ds = ds.assign_coords({coord: ds[coord].mean('time')})

        if HAS_WRADLIB:
            try:
                ds_geo = ds.pipe(wrl.georef.georeference)
                if 'gr' not in ds_geo.coords and 'x' in ds_geo.coords and 'y' in ds_geo.coords:
                    ds_geo = ds_geo.assign_coords(gr=np.hypot(ds_geo['x'], ds_geo['y']))
                return ds_geo
            except Exception as e:
                print(f"  wradlib georeference fallback ({e})...")

        # Analytical 4/3 Earth radius fallback
        rng = np.asarray(ds['range'].values, dtype=float)
        az = np.asarray(ds['azimuth'].values, dtype=float)
        elev_deg = self.get_elevation_deg(ds)
        alt_m = self.get_altitude_m(ds)

        re_eff = (4.0 / 3.0) * 6371000.0
        elev_rad = np.radians(elev_deg)
        az_rad = np.radians(az)[:, None]
        r2d = rng[None, :]

        z_2d = alt_m + np.sqrt(r2d**2 + re_eff**2 + 2.0 * r2d * re_eff * np.sin(elev_rad)) - re_eff
        gr_2d = re_eff * np.arcsin((r2d * np.cos(elev_rad)) / (re_eff + z_2d - alt_m))
        x_2d = gr_2d * np.sin(az_rad)
        y_2d = gr_2d * np.cos(az_rad)

        ds = ds.assign_coords(
            x=(('azimuth', 'range'), x_2d),
            y=(('azimuth', 'range'), y_2d),
            z=(('azimuth', 'range'), z_2d),
            gr=(('azimuth', 'range'), gr_2d),
        )
        return ds

    def get_elevation_options(self, date_str):
        """
        Return list of (elev_key, label) available for the given date.
        """
        elev_map = self.date_elev_files.get(date_str, {'default': []})
        if len(elev_map) > 1:
            def sort_key(k):
                try:
                    return (0, float(k))
                except Exception:
                    return (1, k)
            keys = sorted(elev_map.keys(), key=sort_key)
            return [(k, f"Folder {k}") for k in keys]

        # Single folder/file: inspect raw dataset for sweep_fixed_angle dim
        ds_raw = self._get_raw_day(date_str, 'default')
        if 'sweep_fixed_angle' in ds_raw.dims and ds_raw.sizes['sweep_fixed_angle'] > 1:
            n_el = ds_raw.sizes['sweep_fixed_angle']
            if 'elevation' in ds_raw.coords and ds_raw['elevation'].ndim == 1 and len(ds_raw['elevation']) == n_el:
                vals = ds_raw['elevation'].values
                return [(f"idx:{i}", f"{float(vals[i]):.1f}° (idx {i})") for i in range(n_el)]
            elif 'sweep_fixed_angle' in ds_raw.coords:
                vals = ds_raw['sweep_fixed_angle'].values
                return [(f"idx:{i}", f"{float(vals[i]):.1f}° (idx {i})") for i in range(n_el)]
            return [(f"idx:{i}", f"Elev idx {i}") for i in range(n_el)]

        elev_deg = self.get_elevation_deg(ds_raw)
        return [('default', f"{elev_deg:.1f}°")]

    def _get_raw_day(self, date_str, folder_key='default'):
        cache_key = (date_str, f"raw:{folder_key}")
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.is_single_file_multi_day and self.single_file_ds is not None:
            ds = self.single_file_ds.sel(time=date_str)
            self.cache[cache_key] = ds
            return ds

        if (date_str, 'raw') in self.cache and folder_key == 'default':
            ds = self.cache[(date_str, 'raw')]
            self.cache[cache_key] = ds
            return ds

        files = self.date_elev_files.get(date_str, {}).get(folder_key, [])
        if not files:
            raise ValueError(f"No PPI files found for date {date_str} ({folder_key})")

        ds = self._load_files_raw(files)
        ds = self._standardize_ds(ds)
        if len(self.cache) > 10:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = ds
        return ds

    def load_sweep(self, date_str, elev_key='default'):
        """
        Load and georeference a single-elevation PPI dataset for (date_str, elev_key).
        """
        cache_key = (date_str, f"geo:{elev_key}")
        if cache_key in self.cache:
            return self.cache[cache_key]

        if elev_key.startswith('idx:'):
            idx = int(elev_key.split(':', 1)[1])
            ds_raw = self._get_raw_day(date_str, 'default')
            ds = ds_raw.isel(sweep_fixed_angle=idx)
        else:
            ds_raw = self._get_raw_day(date_str, elev_key)
            if 'sweep_fixed_angle' in ds_raw.dims:
                # Default to index 7 for EMVORADO volumes (matching compute_qvps_new_emvorado.py qvp_ielev=7) if available
                default_idx = 7 if ds_raw.sizes['sweep_fixed_angle'] > 7 else 0
                ds = ds_raw.isel(sweep_fixed_angle=default_idx)
            else:
                ds = ds_raw

        ds = self._ensure_georeferenced(ds)
        if len(self.cache) > 10:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = ds
        return ds

    def get_elevation_deg(self, ds):
        for k in ['elevation', 'sweep_fixed_angle']:
            if k in ds.coords or k in ds.data_vars:
                try:
                    val = np.asarray(ds[k].values, dtype=float)
                    val = val[np.isfinite(val)]
                    if len(val) > 0:
                        return float(np.median(val))
                except Exception:
                    pass
        if 'fixed_angle' in ds.attrs:
            try:
                return float(ds.attrs['fixed_angle'])
            except Exception:
                pass
        return 0.0

    def get_altitude_m(self, ds):
        if 'altitude' in ds.coords or 'altitude' in ds.data_vars:
            try:
                val = np.asarray(ds['altitude'].values, dtype=float)
                val = val[np.isfinite(val)]
                if len(val) > 0:
                    return float(np.median(val))
            except Exception:
                pass
        return 0.0

    def get_available_variables(self, ds):
        valid_vars = []
        for v in ds.data_vars:
            if v in NON_MOMENT_VARS:
                continue
            dims = ds[v].dims
            if 'azimuth' in dims and 'range' in dims:
                if ds[v].dtype.kind in ('f', 'i', 'u'):
                    valid_vars.append(v)
        return sorted(valid_vars)

    def get_default_variables(self, available_vars):
        p1_candidates = ['DBZH_AC', 'DBZH_ACemp', 'DBZH', 'DBTH_OC', 'DBTH', 'ZH', 'TH']
        p2_candidates = [
            'ZDR_EC_OC_WRC_AC', 'ZDR_EC_OC_AC', 'ZDR_EC_AC', 'ZDR_AC_EC',
            'ZDR_OC_AC', 'ZDR_AC', 'ZDR_EC_OC_WRC', 'ZDR_EC_OC', 'ZDR_OC', 'ZDR_EC', 'ZDR'
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

    def get_sweep_geometry_1d(self, ds):
        """
        Return 1D sweep geometry arrays:
            azimuth_deg (N_az,), range_km (N_r,), gr_km (N_r,), z_m (N_r,)
        """
        az = np.asarray(ds['azimuth'].values, dtype=float)
        rng_m = np.asarray(ds['range'].values, dtype=float)
        if np.nanmax(rng_m) < 500.0:
            rng_m = rng_m * 1000.0
        range_km = rng_m / 1000.0

        # Ground range (in km)
        if 'gr' in ds.coords:
            gr_raw = np.asarray(ds['gr'].values, dtype=float)
            if gr_raw.ndim == 2:
                gr_m = np.nanmedian(gr_raw, axis=0) if gr_raw.shape == (len(az), len(rng_m)) else np.nanmedian(gr_raw, axis=1)
            else:
                gr_m = gr_raw
            gr_km = gr_m / 1000.0 if np.nanmax(gr_m) > 500.0 else gr_m
        elif 'x' in ds.coords and 'y' in ds.coords:
            x_v = np.asarray(ds['x'].values, dtype=float)
            y_v = np.asarray(ds['y'].values, dtype=float)
            gr_2d = np.hypot(x_v, y_v)
            gr_m = np.nanmedian(gr_2d, axis=0) if gr_2d.shape == (len(az), len(rng_m)) else np.nanmedian(gr_2d, axis=1)
            gr_km = gr_m / 1000.0 if np.nanmax(gr_m) > 500.0 else gr_m
        else:
            elev_rad = np.radians(self.get_elevation_deg(ds))
            gr_km = range_km * np.cos(elev_rad)

        # Height z (in meters)
        if 'z' in ds.coords:
            z_raw = np.asarray(ds['z'].values, dtype=float)
            if z_raw.ndim == 2:
                z_m = np.nanmedian(z_raw, axis=0) if z_raw.shape == (len(az), len(rng_m)) else np.nanmedian(z_raw, axis=1)
            else:
                z_m = z_raw
            if np.nanmax(z_m) < 50.0:
                z_m = z_m * 1000.0
        else:
            elev_rad = np.radians(self.get_elevation_deg(ds))
            z_m = self.get_altitude_m(ds) + rng_m * np.sin(elev_rad)

        return az, range_km, gr_km, z_m


def get_var_vis_info(var_name, data_array=None, compute_percentiles=True):
    """
    Resolve colormap, vmin, vmax, and units for a radar variable using radarmet.visdict14.
    If compute_percentiles is False and the variable is not in visdict14, returns safe defaults
    without triggering lazy dask reads on large 3D PPI datasets.
    """
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
        from_visdict = True
    else:
        cmap = 'miub2'
        cmap_name = 'miub2'
        from_visdict = False
        if compute_percentiles and data_array is not None:
            vals = np.asarray(data_array.values)
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
        'units': units,
        'from_visdict': from_visdict
    }


def _contiguous_intervals(mask_1d, edges_1d):
    """
    Given a 1D boolean mask of length N and bin edges of length N+1,
    return (starts, ends) lists for contiguous True intervals.
    """
    starts, ends = [], []
    in_seg = False
    seg_start = 0
    for i, flag in enumerate(mask_1d):
        if flag and not in_seg:
            in_seg = True
            seg_start = i
        elif not flag and in_seg:
            in_seg = False
            starts.append(float(edges_1d[seg_start]))
            ends.append(float(edges_1d[i]))
    if in_seg:
        starts.append(float(edges_1d[seg_start]))
        ends.append(float(edges_1d[len(mask_1d)]))
    return starts, ends


def _compute_bin_edges(centers):
    """
    Compute bin edges (length N+1) from 1D sorted bin centers (length N).
    """
    c = np.asarray(centers, dtype=float)
    if len(c) == 0:
        return np.array([0.0])
    if len(c) == 1:
        return np.array([max(0.0, c[0] - 0.5), c[0] + 0.5])
    mids = 0.5 * (c[:-1] + c[1:])
    first = max(0.0, c[0] - 0.5 * (c[1] - c[0]))
    last = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate([[first], mids, [last]])


# ==============================================================================
# Interactive Bokeh PPI Application
# ==============================================================================

class PPIBokehApp:
    """
    Interactive Bokeh application for Polar Plan Position Indicators (PPI)
    with linked 3-panel top row and click-to-select radial line plot bottom row.
    """
    def __init__(self, dataset_mgr, doc, initial_date=None, initial_time_idx=0, initial_max_range=None):
        self.mgr = dataset_mgr
        self.doc = doc
        self.dates = self.mgr.date_list
        if not self.dates:
            raise ValueError("No valid PPI files/dates found to visualize.")

        self.current_date = initial_date if initial_date in self.dates else self.dates[0]
        self.elev_options = self.mgr.get_elevation_options(self.current_date)
        # Default elevation: prefer index 7 for multi-elev EMVORADO if available, else index 0
        self.current_elev_key = self.elev_options[7][0] if len(self.elev_options) > 7 and self.elev_options[0][0].startswith('idx:') else self.elev_options[0][0]
        self.current_ds = self.mgr.load_sweep(self.current_date, self.current_elev_key)
        self.all_vars = self.mgr.get_available_variables(self.current_ds)
        if not self.all_vars:
            raise ValueError("No plottable 2D (azimuth, range) radar variables found in dataset.")

        self.n_times = int(self.current_ds.sizes.get('time', 1))
        self.current_time_idx = max(0, min(int(initial_time_idx), self.n_times - 1))

        # Sweep 1D geometry & max range
        self.az_deg, self.range_km, self.gr_km, self.z_m = self.mgr.get_sweep_geometry_1d(self.current_ds)
        self.elev_deg = self.mgr.get_elevation_deg(self.current_ds)
        self.alt_m = self.mgr.get_altitude_m(self.current_ds)
        self.data_max_range_km = float(np.ceil(np.nanmax(self.gr_km))) if len(self.gr_km) > 0 else 60.0
        if initial_max_range is not None and initial_max_range > 0:
            self.max_range_km = float(initial_max_range)
        else:
            self.max_range_km = min(self.data_max_range_km, 150.0) if self.data_max_range_km > 150.0 else self.data_max_range_km
            if self.max_range_km <= 0:
                self.max_range_km = 60.0

        self.current_az_idx = 0
        self.radial_x_mode = "Range (km)"  # or "Ground Range (km)" or "Height (m)"
        self._slice_cache = {}  # var_name -> 2D numpy array (azimuth, range) for current timestep
        self._polar_lut = None

        # Default variables: Top row uses first 3 (DBZH, ZDR, KDP); Bottom row uses first 4 (+ RHOHV)
        defaults = self.mgr.get_default_variables(self.all_vars)
        self.ppi_settings = []
        for i in range(3):
            vname = defaults[i]
            da_slice = self._get_var_slice_2d(vname)
            vis = get_var_vis_info(vname, self.current_ds[vname], compute_percentiles=False)
            if not vis['from_visdict']:
                vis = get_var_vis_info(vname, xr.DataArray(da_slice, attrs=self.current_ds[vname].attrs), compute_percentiles=True)
            c_name = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
            self.ppi_settings.append({
                'var': vname,
                'cmap_name': c_name,
                'vmin': vis['vmin'],
                'vmax': vis['vmax'],
                'vis': vis
            })

        self.active_1d_vars = list(defaults)
        self.var_styles = {}
        self._init_var_styles()

        # Overlay availability & toggle states
        self.riming_ds_cache = {}
        self.riming_day_cache = {}  # (loc, date_str) -> (times_ns, z_vals_m, cleaned_2d) or None
        self.radar_location = None

        self.show_ml = True
        self.show_entropy = True
        self.show_riming = True
        self.has_ml = False
        self.has_entropy = False
        self.has_riming = False

        # Bokeh plot containers & shared sources
        self.ppi_plots = []
        self.ppi_sources = []
        self.ppi_images = []
        self.ppi_mappers = []
        self.ppi_colorbars = []
        self.ppi_hovers = []
        self.ppi_ml_renderers = []
        self.ppi_entropy_renderers = []
        self.ppi_riming_renderers = []

        self.source_elev = ColumnDataSource(data=dict(elev_deg=[self.elev_deg], alt_m=[self.alt_m]))
        self.source_radial_line = ColumnDataSource(data=dict(x=[0.0, 0.0], y=[0.0, self.max_range_km]))
        self.source_ml_top = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_ml_bot = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_entropy_rings = ColumnDataSource(data=dict(x=[], y=[], inner_radius=[], outer_radius=[]))
        self.source_riming_rings = ColumnDataSource(data=dict(x=[], y=[], inner_radius=[], outer_radius=[]))
        self.source_grid_rings = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.source_grid_spokes = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[]))
        self.source_grid_labels = ColumnDataSource(data=dict(x=[], y=[], text=[]))
        self.source_1d = ColumnDataSource(data=dict(x=[]))

        self.radial_layout = column([], width=1555)
        self.p_radial = None
        self._updating_time_widgets = False

        # Assess overlay availability before building UI
        self._detect_overlay_availability()

        self._build_ui()
        self._rebuild_polar_lut()
        self._update_polar_grid()
        self.update_all()

    # --------------------------------------------------------------------------
    # Data & Style Helpers
    # --------------------------------------------------------------------------

    def _init_var_styles(self):
        for idx, v in enumerate(self.all_vars):
            vis = get_var_vis_info(v, self.current_ds[v], compute_percentiles=False)
            needs_calc = (not vis['from_visdict'])
            if needs_calc and v in self.active_1d_vars:
                sl = self._get_var_slice_2d(v)
                vis = get_var_vis_info(v, xr.DataArray(sl, attrs=self.current_ds[v].attrs), compute_percentiles=True)
                needs_calc = False
            self.var_styles[v] = {
                'color': DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)],
                'linestyle': LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)],
                'ymin': vis['vmin'],
                'ymax': vis['vmax'],
                'units': vis['units'],
                'needs_calc': needs_calc,
            }

    def _ensure_var_style_computed(self, v):
        if v not in self.var_styles:
            idx = self.all_vars.index(v) if v in self.all_vars else 0
            sl = self._get_var_slice_2d(v)
            vis = get_var_vis_info(v, xr.DataArray(sl, attrs=self.current_ds[v].attrs), compute_percentiles=True)
            self.var_styles[v] = {
                'color': DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)],
                'linestyle': LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)],
                'ymin': vis['vmin'],
                'ymax': vis['vmax'],
                'units': vis['units'],
                'needs_calc': False,
            }
        elif self.var_styles[v].get('needs_calc', False):
            sl = self._get_var_slice_2d(v)
            vis = get_var_vis_info(v, xr.DataArray(sl, attrs=self.current_ds[v].attrs), compute_percentiles=True)
            self.var_styles[v]['ymin'] = vis['vmin']
            self.var_styles[v]['ymax'] = vis['vmax']
            self.var_styles[v]['needs_calc'] = False

    def _get_var_slice_2d(self, var_name):
        """
        Return 2D numpy float array of shape (len(azimuth), len(range)) for current timestep.
        """
        if var_name in self._slice_cache:
            return self._slice_cache[var_name]

        da = self.current_ds[var_name]
        if 'time' in da.dims:
            da = da.isel(time=self.current_time_idx)
        if 'azimuth' in da.dims and 'range' in da.dims and da.dims != ('azimuth', 'range'):
            da = da.transpose('azimuth', 'range')
        arr = np.asarray(da.values, dtype=float)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        self._slice_cache[var_name] = arr
        return arr

    def _get_time_options(self):
        times = pd.to_datetime(self.current_ds.time.values)
        opts = []
        for idx, t in enumerate(times):
            if pd.notnull(t):
                opts.append((str(idx), t.strftime('%H:%M:%S')))
            else:
                opts.append((str(idx), f"Step {idx}"))
        return opts

    def _get_current_time_str(self):
        if 0 <= self.current_time_idx < self.n_times:
            t_val = pd.to_datetime(self.current_ds.time.values[self.current_time_idx])
            if pd.notnull(t_val):
                return t_val.strftime('%Y-%m-%d %H:%M:%S UTC')
        return f"{self.current_date} (Step {self.current_time_idx})"

    # --------------------------------------------------------------------------
    # Polar-to-Cartesian Lookup Table & Polar Grid
    # --------------------------------------------------------------------------

    def _rebuild_polar_lut(self, grid_n=520):
        max_r = float(self.max_range_km)
        step = (2.0 * max_r) / grid_n
        coords = np.linspace(-max_r + step / 2.0, max_r - step / 2.0, grid_n)
        xx, yy = np.meshgrid(coords, coords)
        rr = np.hypot(xx, yy)
        az = (np.degrees(np.arctan2(xx, yy))) % 360.0

        az_arr = np.asarray(self.az_deg, dtype=float) % 360.0
        az_order = np.argsort(az_arr)
        az_sorted = az_arr[az_order]

        idx_hi = np.searchsorted(az_sorted, az, side='left')
        idx_lo = (idx_hi - 1) % len(az_sorted)
        idx_hi_mod = idx_hi % len(az_sorted)

        d_lo = np.abs(((az - az_sorted[idx_lo] + 180.0) % 360.0) - 180.0)
        d_hi = np.abs(((az - az_sorted[idx_hi_mod] + 180.0) % 360.0) - 180.0)
        nearest_sorted_az = np.where(d_lo <= d_hi, idx_lo, idx_hi_mod)
        az_idx = az_order[nearest_sorted_az]

        gr_arr = np.asarray(self.gr_km, dtype=float)
        r_order = np.argsort(gr_arr)
        gr_sorted = gr_arr[r_order]
        r_hi = np.searchsorted(gr_sorted, rr, side='left')
        r_hi_clip = np.clip(r_hi, 0, len(gr_sorted) - 1)
        r_lo_clip = np.clip(r_hi - 1, 0, len(gr_sorted) - 1)
        nearest_sorted_r = np.where(
            np.abs(rr - gr_sorted[r_lo_clip]) <= np.abs(rr - gr_sorted[r_hi_clip]),
            r_lo_clip, r_hi_clip
        )
        r_idx = r_order[nearest_sorted_r]

        dr = float(np.nanmedian(np.diff(gr_sorted))) if len(gr_sorted) > 1 else 1.0
        r_min_valid = max(0.0, float(gr_sorted[0]) - dr * 0.5)
        r_max_valid = float(gr_sorted[-1]) + dr * 0.5

        daz = float(np.nanmedian(np.diff(az_sorted))) if len(az_sorted) > 1 else 1.0
        min_az_dist = np.minimum(d_lo, d_hi)
        valid_mask = (rr >= r_min_valid) & (rr <= r_max_valid) & (min_az_dist <= max(daz * 1.5, 2.0))

        self._polar_lut = {
            'az_idx': az_idx,
            'r_idx': r_idx,
            'valid_mask': valid_mask,
            'x0': -max_r,
            'y0': -max_r,
            'dw': 2.0 * max_r,
            'dh': 2.0 * max_r,
            'max_r_km': max_r,
        }

    def _polar_to_cartesian(self, data_2d):
        lut = self._polar_lut
        if lut is None:
            return np.full((10, 10), np.nan)
        az_idx = np.clip(lut['az_idx'], 0, data_2d.shape[0] - 1)
        r_idx = np.clip(lut['r_idx'], 0, data_2d.shape[1] - 1)
        return np.where(lut['valid_mask'], data_2d[az_idx, r_idx], np.nan)

    def _update_polar_grid(self):
        """
        Update range rings, azimuth spokes, and cardinal/range labels for current max_range_km.
        """
        max_r = float(self.max_range_km)
        # Pick nice ring spacing yielding ~3 to 6 rings
        for step in (2, 5, 10, 15, 20, 25, 30, 50, 75, 100):
            n_rings = int(max_r // step)
            if 2 <= n_rings <= 6:
                ring_step = step
                break
        else:
            ring_step = max(1.0, round(max_r / 4.0))

        radii = [float(r) for r in np.arange(ring_step, max_r + 1e-6, ring_step) if r <= max_r]
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        self.source_grid_rings.data = dict(
            xs=[(r * sin_t).tolist() for r in radii],
            ys=[(r * cos_t).tolist() for r in radii],
        )

        outer_r = radii[-1] if radii else max_r
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        x0s, y0s, x1s, y1s = [], [], [], []
        for a in angles_deg:
            rad = np.radians(a)
            x0s.append(0.0)
            y0s.append(0.0)
            x1s.append(float(outer_r * np.sin(rad)))
            y1s.append(float(outer_r * np.cos(rad)))
        self.source_grid_spokes.data = dict(x0=x0s, y0=y0s, x1=x1s, y1=y1s)

        lbl_x, lbl_y, lbl_t = [], [], []
        # Range labels along 45 deg (NE)
        s45 = np.sin(np.radians(45))
        for r in radii:
            lbl_x.append(float(r * s45))
            lbl_y.append(float(r * s45))
            lbl_t.append(f"{int(r)} km" if abs(r - round(r)) < 1e-3 else f"{r:g} km")
        # Cardinal indicators near outer edge
        c_r = max_r * 0.92
        for txt, cx, cy in [("N", 0.0, c_r), ("E", c_r, 0.0), ("S", 0.0, -c_r), ("W", -c_r, 0.0)]:
            lbl_x.append(float(cx))
            lbl_y.append(float(cy))
            lbl_t.append(txt)
        self.source_grid_labels.data = dict(x=lbl_x, y=lbl_y, text=lbl_t)

    # --------------------------------------------------------------------------
    # Overlay Detection & Extraction (ML, Entropy, Riming)
    # --------------------------------------------------------------------------

    def _find_ml_var_names(self, ds):
        top_var, bot_var = None, None
        for k in ['height_ml_new_gia_clean', 'height_ml_new_gia', 'height_ml']:
            if k in ds.coords or k in ds.data_vars:
                vals = np.asarray(ds[k].values, dtype=float)
                if np.any(np.isfinite(vals)):
                    top_var = k
                    break
        for k in ['height_ml_bottom_new_gia_clean', 'height_ml_bottom_new_gia', 'height_ml_bottom']:
            if k in ds.coords or k in ds.data_vars:
                vals = np.asarray(ds[k].values, dtype=float)
                if np.any(np.isfinite(vals)):
                    bot_var = k
                    break
        return top_var, bot_var

    def _find_entropy_var_name(self, ds):
        for cand in ['min_entropy', 'MIN_ENTROPY', 'min_Entropy', 'entropy_min']:
            if cand in ds.data_vars or cand in ds.coords:
                return cand
        for v in ds.data_vars:
            if 'entropy' in v.lower():
                return v
        return None

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

    def _get_riming_day_cleaned(self):
        """
        Load and morphologically clean the daily (time, z) riming mask for the current date & location.
        Returns (times_ns, z_vals_m, cleaned_2d) or None if unavailable.
        """
        fpath = self.mgr.date_to_file.get(self.current_date, '')
        if not fpath and self.mgr.file_list:
            fpath = self.mgr.file_list[0]
        loc = extract_radar_location(fpath) or extract_radar_location(self.current_ds)
        self.radar_location = loc
        if not loc:
            return None

        key = (loc, self.current_date)
        if key in self.riming_day_cache:
            return self.riming_day_cache[key]

        ds_rim, rim_var = self._get_riming_dataset(loc)
        if ds_rim is None or rim_var is None or rim_var not in ds_rim:
            self.riming_day_cache[key] = None
            return None

        try:
            da_rim = ds_rim[rim_var].sel(time=self.current_date)
        except Exception:
            self.riming_day_cache[key] = None
            return None

        if da_rim.size == 0 or da_rim.sizes.get('time', 0) == 0:
            self.riming_day_cache[key] = None
            return None

        if 'z' in da_rim.dims and 'time' in da_rim.dims and da_rim.dims != ('time', 'z'):
            da_rim = da_rim.transpose('time', 'z')

        raw_vals = da_rim.fillna(0).values
        if HAS_SCIPY:
            try:
                cleaned = binary_opening(raw_vals, structure=np.ones((3, 3))).astype(float)
            except Exception:
                cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)
        else:
            cleaned = np.where(raw_vals > 0.5, 1.0, 0.0)

        times_ns = pd.to_datetime(da_rim.time.values).astype('int64').values
        z_vals = np.asarray(da_rim['z'].values, dtype=float)
        if np.nanmax(z_vals) < 50.0:
            z_vals = z_vals * 1000.0

        res = (times_ns, z_vals, cleaned)
        self.riming_day_cache[key] = res
        return res

    def _detect_overlay_availability(self):
        """
        Check availability of Melting Layer, Entropy, and Riming for the current dataset/date.
        """
        top_v, bot_v = self._find_ml_var_names(self.current_ds)
        self.ml_top_var = top_v
        self.ml_bot_var = bot_v
        self.has_ml = (top_v is not None or bot_v is not None)

        self.entropy_var = self._find_entropy_var_name(self.current_ds)
        self.has_entropy = (self.entropy_var is not None)

        rim_day = self._get_riming_day_cleaned()
        self.has_riming = (rim_day is not None)

    def _sync_overlay_button_states(self):
        """
        Enable or disable the ML, Entropy, and Riming buttons based on availability.
        """
        # Melting Layer
        if self.has_ml:
            self.btn_toggle_ml.disabled = False
            self.btn_toggle_ml.label = "Melting Layer: ON" if self.show_ml else "Melting Layer: OFF"
            self.btn_toggle_ml.button_type = "primary" if self.show_ml else "default"
        else:
            self.btn_toggle_ml.disabled = True
            self.btn_toggle_ml.label = "Melting Layer: N/A"
            self.btn_toggle_ml.button_type = "default"

        # Entropy
        if self.has_entropy:
            self.btn_toggle_entropy.disabled = False
            self.btn_toggle_entropy.label = "Entropy: ON" if self.show_entropy else "Entropy: OFF"
            self.btn_toggle_entropy.button_type = "primary" if self.show_entropy else "default"
        else:
            self.btn_toggle_entropy.disabled = True
            self.btn_toggle_entropy.label = "Entropy: N/A"
            self.btn_toggle_entropy.button_type = "default"

        # Riming
        if self.has_riming:
            self.btn_toggle_riming.disabled = False
            self.btn_toggle_riming.label = "Riming: ON" if self.show_riming else "Riming: OFF"
            self.btn_toggle_riming.button_type = "primary" if self.show_riming else "default"
        else:
            self.btn_toggle_riming.disabled = True
            self.btn_toggle_riming.label = "Riming: N/A"
            self.btn_toggle_riming.button_type = "default"

        for r_top, r_bot in self.ppi_ml_renderers:
            r_top.visible = (self.has_ml and self.show_ml)
            r_bot.visible = (self.has_ml and self.show_ml)
        for r_ent in self.ppi_entropy_renderers:
            r_ent.visible = (self.has_entropy and self.show_entropy)
        for r_rim in self.ppi_riming_renderers:
            r_rim.visible = (self.has_riming and self.show_riming)

    def _get_current_ml_heights(self):
        """
        Return (ml_top_m, ml_bot_m) floats or None for the current timestep.
        """
        if not self.has_ml:
            return None, None
        ml_top, ml_bot = None, None
        if self.ml_top_var is not None:
            da = self.current_ds[self.ml_top_var]
            val = float(da.isel(time=self.current_time_idx).values) if 'time' in da.dims else float(da.values)
            if np.isfinite(val):
                ml_top = val
        if self.ml_bot_var is not None:
            da = self.current_ds[self.ml_bot_var]
            val = float(da.isel(time=self.current_time_idx).values) if 'time' in da.dims else float(da.values)
            if np.isfinite(val):
                ml_bot = val
        return ml_top, ml_bot

    def _height_to_gr_km(self, z_target_m):
        if z_target_m is None or not np.isfinite(z_target_m) or len(self.z_m) < 2:
            return None
        z_arr = self.z_m
        gr_arr = self.gr_km
        valid = np.isfinite(z_arr) & np.isfinite(gr_arr)
        z_v = z_arr[valid]
        gr_v = gr_arr[valid]
        if len(z_v) < 2:
            return None
        order = np.argsort(z_v)
        z_v = z_v[order]
        gr_v = gr_v[order]
        if z_target_m < z_v[0] or z_target_m > z_v[-1]:
            return None
        return float(np.interp(z_target_m, z_v, gr_v))

    def _height_to_range_km(self, z_target_m):
        if z_target_m is None or not np.isfinite(z_target_m) or len(self.z_m) < 2:
            return None
        valid = np.isfinite(self.z_m) & np.isfinite(self.range_km)
        z_v = self.z_m[valid]
        r_v = self.range_km[valid]
        if len(z_v) < 2:
            return None
        order = np.argsort(z_v)
        z_v = z_v[order]
        r_v = r_v[order]
        if z_target_m < z_v[0] or z_target_m > z_v[-1]:
            return None
        return float(np.interp(z_target_m, z_v, r_v))

    def _update_ml_overlay(self):
        ml_top, ml_bot = self._get_current_ml_heights()
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        r_top = self._height_to_gr_km(ml_top)
        if r_top is not None and r_top > 0:
            self.source_ml_top.data = dict(
                x=(r_top * np.sin(theta)).tolist(),
                y=(r_top * np.cos(theta)).tolist()
            )
        else:
            self.source_ml_top.data = dict(x=[], y=[])

        r_bot = self._height_to_gr_km(ml_bot)
        if r_bot is not None and r_bot > 0:
            self.source_ml_bot.data = dict(
                x=(r_bot * np.sin(theta)).tolist(),
                y=(r_bot * np.cos(theta)).tolist()
            )
        else:
            self.source_ml_bot.data = dict(x=[], y=[])

    def _get_current_entropy_mask_1d(self):
        """
        Return 1D boolean array of length len(range) where min_entropy > 0.85 at current timestep, or None.
        """
        if not self.has_entropy or self.entropy_var is None:
            return None
        da = self.current_ds[self.entropy_var]
        if 'time' in da.dims:
            da = da.isel(time=self.current_time_idx)
        if 'azimuth' in da.dims:
            da = da.median('azimuth')
        vals = np.asarray(da.values, dtype=float).squeeze()
        if vals.ndim != 1 or len(vals) != len(self.gr_km):
            return None
        return vals > 0.85

    def _update_entropy_overlay(self):
        mask_1d = self._get_current_entropy_mask_1d()
        if mask_1d is None or not np.any(mask_1d):
            self.source_entropy_rings.data = dict(x=[], y=[], inner_radius=[], outer_radius=[])
            return
        gr_edges = _compute_bin_edges(self.gr_km)
        r_in, r_out = _contiguous_intervals(mask_1d, gr_edges)
        self.source_entropy_rings.data = dict(
            x=[0.0] * len(r_in),
            y=[0.0] * len(r_in),
            inner_radius=r_in,
            outer_radius=r_out
        )

    def _get_current_riming_mask_1d(self):
        """
        Return 1D boolean array of length len(range) where riming is detected at current timestep, or None.
        """
        if not self.has_riming:
            return None
        rim_day = self._get_riming_day_cleaned()
        if rim_day is None:
            return None
        times_ns, z_vals_m, cleaned_2d = rim_day

        if self.current_time_idx >= self.n_times:
            return None
        t_cur_ns = int(pd.to_datetime(self.current_ds.time.values[self.current_time_idx]).value)
        dt_ns = np.abs(times_ns - t_cur_ns)
        best_i = int(np.argmin(dt_ns))
        # Tolerance: 15 minutes
        if dt_ns[best_i] > 15 * 60 * 10**9:
            return None

        rim_profile_z = cleaned_2d[best_i, :]  # shape (len(z_vals_m),)
        if not np.any(rim_profile_z > 0.5):
            return np.zeros(len(self.z_m), dtype=bool)

        # Map z_m of PPI range gates onto z_vals_m via nearest-neighbor within 250 m
        z_order = np.argsort(z_vals_m)
        z_sorted = z_vals_m[z_order]
        rim_sorted = rim_profile_z[z_order]

        hi = np.searchsorted(z_sorted, self.z_m, side='left')
        hi_c = np.clip(hi, 0, len(z_sorted) - 1)
        lo_c = np.clip(hi - 1, 0, len(z_sorted) - 1)
        d_lo = np.abs(self.z_m - z_sorted[lo_c])
        d_hi = np.abs(self.z_m - z_sorted[hi_c])
        nearest = np.where(d_lo <= d_hi, lo_c, hi_c)
        min_dist = np.minimum(d_lo, d_hi)

        dz = float(np.nanmedian(np.diff(z_sorted))) if len(z_sorted) > 1 else 250.0
        tol = max(250.0, dz * 0.75)
        mask_1d = (min_dist <= tol) & (rim_sorted[nearest] > 0.5)
        return mask_1d

    def _update_riming_overlay(self):
        mask_1d = self._get_current_riming_mask_1d()
        if mask_1d is None or not np.any(mask_1d):
            self.source_riming_rings.data = dict(x=[], y=[], inner_radius=[], outer_radius=[])
            return
        gr_edges = _compute_bin_edges(self.gr_km)
        r_in, r_out = _contiguous_intervals(mask_1d, gr_edges)
        self.source_riming_rings.data = dict(
            x=[0.0] * len(r_in),
            y=[0.0] * len(r_in),
            inner_radius=r_in,
            outer_radius=r_out
        )

    def toggle_melting_layer(self):
        if not self.has_ml:
            return
        self.show_ml = not self.show_ml
        self._sync_overlay_button_states()
        self._build_radial_plot()

    def toggle_entropy(self):
        if not self.has_entropy:
            return
        self.show_entropy = not self.show_entropy
        self._sync_overlay_button_states()
        self._build_radial_plot()

    def toggle_riming(self):
        if not self.has_riming:
            return
        self.show_riming = not self.show_riming
        self._sync_overlay_button_states()
        self._build_radial_plot()

    # --------------------------------------------------------------------------
    # UI Construction
    # --------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Date selector
        self.sel_date = Select(title="Date:", value=self.current_date, options=self.dates, width=120)
        self.sel_date.on_change('value', lambda attr, old, new: self.on_date_changed(new))

        self.btn_prev_date = Button(label="◀", width=45, margin=(20, 2, 0, 2))
        self.btn_next_date = Button(label="▶", width=45, margin=(20, 2, 0, 2))
        self.btn_prev_date.on_click(lambda: self.step_date(-1))
        self.btn_next_date.on_click(lambda: self.step_date(1))
        if len(self.dates) <= 1:
            self.sel_date.disabled = True
            self.btn_prev_date.disabled = True
            self.btn_next_date.disabled = True

        # 2. Timestep selector & slider
        time_opts = self._get_time_options()
        self.sel_time = Select(title="Time (UTC):", value=str(self.current_time_idx), options=time_opts, width=120)
        self.sel_time.on_change('value', lambda attr, old, new: self.on_time_select_changed(new))

        self.btn_prev_time = Button(label="◀", width=45, margin=(20, 2, 0, 2))
        self.btn_next_time = Button(label="▶", width=45, margin=(20, 2, 0, 2))
        self.btn_prev_time.on_click(lambda: self.step_time(-1))
        self.btn_next_time.on_click(lambda: self.step_time(1))

        self.slider_time = Slider(
            start=0, end=max(1, self.n_times - 1), value=self.current_time_idx, step=1,
            show_value=False, width=230, margin=(2, 4, 4, 4)
        )
        self.slider_time.on_change('value', lambda attr, old, new: self.on_time_slider_changed(new))
        if self.n_times <= 1:
            self.sel_time.disabled = True
            self.btn_prev_time.disabled = True
            self.btn_next_time.disabled = True
            self.slider_time.disabled = True

        # 3. Elevation selector
        self.sel_elev = Select(
            title="Elevation:", value=self.current_elev_key,
            options=self.elev_options, width=230
        )
        self.sel_elev.on_change('value', lambda attr, old, new: self.on_elev_changed(new))
        if len(self.elev_options) <= 1:
            self.sel_elev.disabled = True

        # 4. Max Range & Azimuth controls
        self.inp_max_r = NumericInput(title="Max Range (km):", value=self.max_range_km, width=140)
        self.btn_apply_r = Button(label="Apply", button_type="primary", width=80, margin=(20, 2, 0, 4))
        self.btn_apply_r.on_click(self.apply_max_range)

        self.inp_az = NumericInput(
            title="Radial Azimuth (°):",
            value=round(float(self.az_deg[self.current_az_idx]), 1),
            mode="float", width=110
        )
        self.btn_az_prev = Button(label="◀", width=35, margin=(20, 2, 0, 2))
        self.btn_az_next = Button(label="▶", width=35, margin=(20, 2, 0, 2))
        self.btn_az_apply = Button(label="Set", width=40, margin=(20, 2, 0, 2))
        self.btn_az_prev.on_click(lambda: self.step_azimuth(-1))
        self.btn_az_next.on_click(lambda: self.step_azimuth(1))
        self.btn_az_apply.on_click(self.apply_azimuth_input)

        self.sel_radial_x = Select(
            title="Radial X-Axis:", value=self.radial_x_mode,
            options=["Range (km)", "Ground Range (km)", "Height (m)"], width=230
        )
        self.sel_radial_x.on_change('value', lambda attr, old, new: self.on_radial_x_changed(new))

        # 5. Overlay toggle buttons (auto-disabled when unavailable)
        self.btn_toggle_ml = Button(label="Melting Layer: ON", button_type="primary", width=230, margin=(4, 2, 2, 2))
        self.btn_toggle_ml.on_click(self.toggle_melting_layer)

        self.btn_toggle_entropy = Button(label="Entropy: ON", button_type="primary", width=112, margin=(2, 2, 4, 2))
        self.btn_toggle_entropy.on_click(self.toggle_entropy)

        self.btn_toggle_riming = Button(label="Riming: ON", button_type="primary", width=112, margin=(2, 2, 4, 2))
        self.btn_toggle_riming.on_click(self.toggle_riming)

        # 6. Sidebar Tabs
        # Tab 1: PPI Panels (1-3)
        panel_widgets = []
        self.panel_ui = []
        for i in range(3):
            lbl = Div(text=f"<b>PPI Panel {i+1} Configuration</b>", margin=(6, 0, 2, 0))
            st = self.ppi_settings[i]
            sel_v = Select(title="Variable:", value=st['var'], options=self.all_vars, width=230)
            sel_c = Select(
                title="Colormap:",
                value=st['cmap_name'] if st['cmap_name'] in CMAP_OPTIONS else 'miub2',
                options=CMAP_OPTIONS, width=230
            )
            inp_vmin = NumericInput(title="vmin:", value=st['vmin'], mode="float", width=110)
            inp_vmax = NumericInput(title="vmax:", value=st['vmax'], mode="float", width=110)
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

        tab1_layout = column(panel_widgets, width=240)
        tab1 = TabPanel(child=tab1_layout, title="PPI Panels (1-3)")

        # Tab 2: Radial 1D Variables
        active_indices = [self.all_vars.index(v) for v in self.active_1d_vars if v in self.all_vars]
        self.chk_1d = CheckboxGroup(labels=self.all_vars, active=active_indices, width=230)
        self.chk_1d.on_change('active', lambda attr, old, new: self.on_checkbox_change(new))

        btn_sel_all = Button(label="All", width=55)
        btn_clr_all = Button(label="Clear", width=65)
        btn_sync = Button(label="Sync PPIs", button_type="primary", width=95)
        btn_sel_all.on_click(self.select_all_1d_vars)
        btn_clr_all.on_click(self.clear_all_1d_vars)
        btn_sync.on_click(self.sync_1d_with_ppis)

        tab2_layout = column(
            Div(text="<b>Active Variables in Radial Plot:</b>"),
            row(btn_sel_all, btn_clr_all, btn_sync),
            self.chk_1d,
            width=240
        )
        tab2 = TabPanel(child=tab2_layout, title="Radial Vars")

        # Tab 3: Radial Curve Styles & Ranges
        self.style_controls_box = column([], width=240)
        tab3 = TabPanel(child=self.style_controls_box, title="Radial Styles")

        tabs_sidebar = Tabs(tabs=[tab1, tab2, tab3], width=248)

        # Build Row 1 (3 polar PPIs) and Row 2 (radial line plot)
        self._build_ppi_plots()
        self._sync_overlay_button_states()

        # Assemble Layout
        sidebar_column = column(
            Div(text="<h4 style='margin:0 0 4px 0; color:#1a365d;'>PPI Navigation & Settings</h4>"),
            row(self.btn_prev_date, self.sel_date, self.btn_next_date),
            row(self.btn_prev_time, self.sel_time, self.btn_next_time),
            self.slider_time,
            self.sel_elev,
            row(self.inp_max_r, self.btn_apply_r),
            row(self.inp_az, self.btn_az_prev, self.btn_az_next, self.btn_az_apply),
            self.sel_radial_x,
            self.btn_toggle_ml,
            row(self.btn_toggle_entropy, self.btn_toggle_riming),
            Div(text="<hr style='margin:5px 0;'>"),
            tabs_sidebar,
            width=255
        )

        row1_ppis = row(self.ppi_plots, spacing=6)
        main_plots_column = column(row1_ppis, self.radial_layout, spacing=8)

        self.app_layout = row(
            sidebar_column,
            main_plots_column,
            spacing=12
        )
        self.doc.add_root(self.app_layout)
        self.doc.title = f"Interactive PPI Visualizer - {self.current_date}"

    def _build_ppi_plots(self):
        self.ppi_plots.clear()
        self.ppi_sources.clear()
        self.ppi_images.clear()
        self.ppi_mappers.clear()
        self.ppi_colorbars.clear()
        self.ppi_hovers.clear()
        self.ppi_ml_renderers.clear()
        self.ppi_entropy_renderers.clear()
        self.ppi_riming_renderers.clear()

        shared_x = Range1d(start=-self.max_range_km, end=self.max_range_km)
        shared_y = Range1d(start=-self.max_range_km, end=self.max_range_km)

        az_formatter = CustomJSHover(code="""
            const x = special_vars.x;
            const y = special_vars.y;
            let az = Math.atan2(x, y) * 180.0 / Math.PI;
            if (az < 0) az += 360.0;
            return az.toFixed(1) + "°";
        """)
        rng_formatter = CustomJSHover(args=dict(elev_src=self.source_elev), code="""
            const x = special_vars.x;
            const y = special_vars.y;
            const gr = Math.hypot(x, y);
            const elev_deg = elev_src.data['elev_deg'][0] || 0.0;
            const cos_e = Math.max(0.05, Math.cos(elev_deg * Math.PI / 180.0));
            const r_slant = gr / cos_e;
            return r_slant.toFixed(2) + " km (gr: " + gr.toFixed(2) + " km)";
        """)
        hgt_formatter = CustomJSHover(args=dict(elev_src=self.source_elev), code="""
            const x = special_vars.x;
            const y = special_vars.y;
            const gr_m = Math.hypot(x, y) * 1000.0;
            const elev_deg = elev_src.data['elev_deg'][0] || 0.0;
            const alt_m = elev_src.data['alt_m'][0] || 0.0;
            const elev_rad = elev_deg * Math.PI / 180.0;
            const r_m = gr_m / Math.max(0.05, Math.cos(elev_rad));
            const Re = (4.0 / 3.0) * 6371000.0;
            const z = alt_m + Math.sqrt(r_m * r_m + Re * Re + 2.0 * r_m * Re * Math.sin(elev_rad)) - Re;
            return Math.round(z) + " m";
        """)
        xy_formatter = CustomJSHover(code="""
            const x = special_vars.x;
            const y = special_vars.y;
            return "(" + x.toFixed(1) + ", " + y.toFixed(1) + ") km";
        """)

        for i in range(3):
            st = self.ppi_settings[i]
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
            self.ppi_mappers.append(mapper)

            unit_str = f" [{vis['units']}]" if vis['units'] else ""
            tools = "pan,box_zoom,wheel_zoom,reset"

            p = figure(
                title=f"PPI {i+1}: {vname}{unit_str}",
                x_range=shared_x,
                y_range=shared_y,
                frame_width=405,
                frame_height=405,
                match_aspect=True,
                tools=tools,
                toolbar_location="above",
                active_drag="pan",
                active_scroll="wheel_zoom"
            )
            p.xaxis.axis_label = "x (km)"
            p.yaxis.axis_label = "y (km)" if i == 0 else ""
            p.title.text_font_size = "10.5pt"
            p.title.text_font_style = "bold"
            p.grid.grid_line_alpha = 0.25

            # Main polar PPI image
            src = ColumnDataSource(data=dict(
                image=[np.full((10, 10), np.nan)],
                x=[-self.max_range_km], y=[-self.max_range_km],
                dw=[2.0 * self.max_range_km], dh=[2.0 * self.max_range_km]
            ))
            img_renderer = p.image(
                image='image', x='x', y='y', dw='dw', dh='dh',
                color_mapper=mapper, source=src
            )

            # Real-time hover overlay
            hover = HoverTool(
                renderers=[img_renderer],
                tooltips=[
                    ("Value", f"@image{{0.00}}{unit_str}"),
                    ("Azimuth", "$x{custom}"),
                    ("Range", "$y{custom}"),
                    ("Height", "$sx{custom}"),
                    ("(x, y)", "$sy{custom}"),
                ],
                formatters={
                    "$x": az_formatter,
                    "$y": rng_formatter,
                    "$sx": hgt_formatter,
                    "$sy": xy_formatter,
                },
                mode="mouse"
            )
            p.add_tools(hover)
            self.ppi_hovers.append(hover)

            # Polar coordinate grid overlay: range rings (multi_line, zero fill), azimuth spokes, and labels
            p.multi_line(
                xs='xs', ys='ys',
                source=self.source_grid_rings,
                line_color='#444444', line_alpha=0.35,
                nonselection_line_alpha=0.35,
                line_dash='dotted', line_width=1.0
            )
            p.segment(
                x0='x0', y0='y0', x1='x1', y1='y1',
                source=self.source_grid_spokes,
                line_color='#444444', line_alpha=0.30,
                nonselection_line_alpha=0.30,
                line_dash='dotted', line_width=1.0
            )
            p.text(
                x='x', y='y', text='text',
                source=self.source_grid_labels,
                text_font_size="7.5pt", text_color="#333333",
                text_alpha=0.75, nonselection_text_alpha=0.75,
                text_align="center", text_baseline="middle"
            )

            # Gray shading overlay for min_entropy > 0.85
            r_ent = p.annulus(
                x='x', y='y', inner_radius='inner_radius', outer_radius='outer_radius',
                source=self.source_entropy_rings,
                fill_color='#666666', fill_alpha=0.45,
                nonselection_fill_alpha=0.45, line_alpha=0.0,
                name="Entropy Shading"
            )
            r_ent.visible = (self.has_entropy and self.show_entropy)
            self.ppi_entropy_renderers.append(r_ent)

            # Gray shading overlay for riming classification
            r_rim = p.annulus(
                x='x', y='y', inner_radius='inner_radius', outer_radius='outer_radius',
                source=self.source_riming_rings,
                fill_color='#666666', fill_alpha=0.45,
                nonselection_fill_alpha=0.45, line_alpha=0.0,
                name="Riming Shading"
            )
            r_rim.visible = (self.has_riming and self.show_riming)
            self.ppi_riming_renderers.append(r_rim)

            # Melting Layer rings (Top: black dashed, Bottom: black dotted)
            r_top = p.line(
                x='x', y='y', source=self.source_ml_top,
                line_color='black', line_dash='dashed', line_width=1.8,
                line_alpha=0.9, name="ML Top"
            )
            r_bot = p.line(
                x='x', y='y', source=self.source_ml_bot,
                line_color='black', line_dash='dotted', line_width=1.8,
                line_alpha=0.9, name="ML Bottom"
            )
            r_top.visible = (self.has_ml and self.show_ml)
            r_bot.visible = (self.has_ml and self.show_ml)
            self.ppi_ml_renderers.append((r_top, r_bot))

            # Selected radial indicator line (red dashed ray from radar origin)
            p.line(
                x='x', y='y', source=self.source_radial_line,
                line_color='red', line_dash='dashed', line_width=2.0, line_alpha=0.95
            )

            # ColorBar on the right
            ticker = FixedTicker(ticks=list(bar_ticks)) if bar_ticks else BasicTicker()
            cbar = ColorBar(
                color_mapper=mapper, title=unit_str, width=14, padding=4, label_standoff=4,
                ticker=ticker, major_label_overrides=dict(overrides) if overrides else {}
            )
            p.add_layout(cbar, 'right')
            self.ppi_colorbars.append(cbar)

            # Click / tap listener to select radial azimuth
            p.on_event(Tap, self.on_tap_ppi)

            self.ppi_plots.append(p)
            self.ppi_sources.append(src)
            self.ppi_images.append(img_renderer)

    def _build_radial_plot(self):
        """
        Build Row 2 wide multi-axis radial line plot for the selected azimuth.
        """
        az_val = float(self.az_deg[self.current_az_idx]) if len(self.az_deg) > 0 else 0.0
        t_str = self._get_current_time_str()

        if self.radial_x_mode == "Ground Range (km)":
            x_arr = self.gr_km
            x_label = "Ground Range (km)"
            x_max = self.max_range_km
            x_unit = "km"
        elif self.radial_x_mode == "Height (m)":
            x_arr = self.z_m
            x_label = "Height (m)"
            # Height corresponding to max_range_km
            valid_g = np.isfinite(self.gr_km) & np.isfinite(self.z_m)
            if np.any(valid_g):
                x_max = float(np.interp(self.max_range_km, self.gr_km[valid_g], self.z_m[valid_g]))
            else:
                x_max = float(np.nanmax(self.z_m)) if len(self.z_m) > 0 else 12000.0
            x_unit = "m"
        else:
            x_arr = self.range_km
            x_label = "Range (km)"
            cos_e = max(0.05, float(np.cos(np.radians(self.elev_deg))))
            x_max = self.max_range_km / cos_e
            x_unit = "km"

        tools = "pan,box_zoom,wheel_zoom,reset,hover"
        p = figure(
            title=f"Radial Profile  |  Azimuth: {az_val:.1f}°  |  Elev: {self.elev_deg:.1f}°  |  {t_str}",
            x_range=Range1d(start=0.0, end=max(1.0, x_max)),
            tools=tools,
            width=1550, height=430,
            toolbar_location="above",
            active_drag="pan", active_scroll="wheel_zoom"
        )
        p.xaxis.axis_label = x_label
        p.xaxis.axis_label_text_font_style = "bold"
        p.title.text_font_size = "11pt"
        p.title.text_font_style = "bold"
        p.grid.grid_line_dash = "dotted"

        if not self.active_1d_vars:
            p.yaxis.axis_label = "No variable selected"
            self.p_radial = p
            self.radial_layout.children = [p]
            return

        # Populate ColumnDataSource along selected radial
        src_dict = {
            'x': np.asarray(x_arr, dtype=float),
            'range_km': np.asarray(self.range_km, dtype=float),
            'gr_km': np.asarray(self.gr_km, dtype=float),
            'z_m': np.asarray(self.z_m, dtype=float),
        }
        for v in self.active_1d_vars:
            self._ensure_var_style_computed(v)
            arr2d = self._get_var_slice_2d(v)
            if 0 <= self.current_az_idx < arr2d.shape[0]:
                src_dict[v] = arr2d[self.current_az_idx, :]
            else:
                src_dict[v] = np.full(len(x_arr), np.nan)
        self.source_1d.data = src_dict

        # Variable 0 on primary left y-axis
        v0 = self.active_1d_vars[0]
        st0 = self.var_styles[v0]
        unit_str0 = f" [{st0['units']}]" if st0['units'] else ""
        p.y_range = Range1d(start=st0['ymin'], end=st0['ymax'])
        p.yaxis.axis_label = f"{v0}{unit_str0}"
        p.yaxis.axis_label_text_color = st0['color']
        p.yaxis.major_label_text_color = st0['color']
        p.yaxis.axis_line_color = st0['color']
        p.yaxis.major_tick_line_color = st0['color']
        p.yaxis.axis_label_text_font_style = "bold"

        r0 = p.line(
            x='x', y=v0, source=self.source_1d,
            color=st0['color'], line_dash=st0['linestyle'],
            line_width=2.3, name=v0
        )
        renderers = [r0]

        # Subsequent variables alternate between right and left extra y-axes for readability
        for idx_v, v in enumerate(self.active_1d_vars[1:], start=1):
            st = self.var_styles[v]
            unit_str = f" [{st['units']}]" if st['units'] else ""
            side = 'right' if (idx_v % 2 == 1) else 'left'

            p.extra_y_ranges[v] = Range1d(start=st['ymin'], end=st['ymax'])
            ax_twin = LinearAxis(
                y_range_name=v,
                axis_label=f"{v}{unit_str}",
                axis_label_text_color=st['color'],
                major_label_text_color=st['color'],
                axis_line_color=st['color'],
                major_tick_line_color=st['color'],
                axis_label_text_font_style="bold"
            )
            p.add_layout(ax_twin, side)

            r = p.line(
                x='x', y=v, source=self.source_1d,
                color=st['color'], line_dash=st['linestyle'],
                line_width=2.3, y_range_name=v, name=v
            )
            renderers.append(r)

        # Hover tool on radial curves
        hover = p.select_one(HoverTool)
        if hover:
            hover.renderers = renderers
            hover.tooltips = [
                ("Variable", "$name"),
                ("Value", "$y{0.00}"),
                ("Range", "@range_km{0.2f} km"),
                ("Ground Range", "@gr_km{0.2f} km"),
                ("Height", "@z_m{0} m"),
            ]
            hover.mode = "mouse"

        # Entropy gray shaded bands along radial
        if self.has_entropy and self.show_entropy:
            ent_mask = self._get_current_entropy_mask_1d()
            if ent_mask is not None and np.any(ent_mask):
                x_edges = _compute_bin_edges(x_arr)
                starts, ends = _contiguous_intervals(ent_mask, x_edges)
                for xs, xe in zip(starts, ends):
                    p.add_layout(BoxAnnotation(
                        left=xs, right=xe, fill_color='#666666', fill_alpha=0.25, line_alpha=0
                    ))

        # Riming gray shaded bands along radial
        if self.has_riming and self.show_riming:
            rim_mask = self._get_current_riming_mask_1d()
            if rim_mask is not None and np.any(rim_mask):
                x_edges = _compute_bin_edges(x_arr)
                starts, ends = _contiguous_intervals(rim_mask, x_edges)
                for xs, xe in zip(starts, ends):
                    p.add_layout(BoxAnnotation(
                        left=xs, right=xe, fill_color='#666666', fill_alpha=0.25, line_alpha=0
                    ))

        # Melting Layer vertical markers on radial plot
        if self.has_ml and self.show_ml:
            ml_top, ml_bot = self._get_current_ml_heights()
            for z_ml, label_prefix, dash in [
                (ml_top, "ML Top", "dashed"),
                (ml_bot, "ML Bot", "dotted"),
            ]:
                if z_ml is None:
                    continue
                if self.radial_x_mode == "Height (m)":
                    x_loc = z_ml
                elif self.radial_x_mode == "Ground Range (km)":
                    x_loc = self._height_to_gr_km(z_ml)
                else:
                    x_loc = self._height_to_range_km(z_ml)
                if x_loc is not None and 0.0 <= x_loc <= x_max:
                    p.add_layout(Span(
                        location=x_loc, dimension='height',
                        line_color='black', line_dash=dash, line_width=1.6
                    ))
                    y_lbl_offset = 10 if label_prefix == "ML Top" else 26
                    p.add_layout(Label(
                        x=x_loc, y=y_lbl_offset, y_units='screen',
                        text=f" {label_prefix} ({int(round(z_ml))}m)",
                        text_font_size="8pt", text_font_style="bold", text_color="black"
                    ))

        self.p_radial = p
        self.radial_layout.children = [p]

    # --------------------------------------------------------------------------
    # Update Methods
    # --------------------------------------------------------------------------

    def update_all(self):
        self.source_elev.data = dict(elev_deg=[self.elev_deg], alt_m=[self.alt_m])
        for i in range(3):
            self.update_ppi_panel(i)
        self._update_ml_overlay()
        self._update_entropy_overlay()
        self._update_riming_overlay()
        self._sync_overlay_button_states()
        self.update_radial_line()
        self._build_radial_plot()
        self._refresh_styles_tab()

    def update_ppi_panel(self, i):
        st = self.ppi_settings[i]
        vname = st['var']
        vis = st['vis']

        data_2d = self._get_var_slice_2d(vname)
        img = self._polar_to_cartesian(data_2d)
        lut = self._polar_lut

        self.ppi_sources[i].data = dict(
            image=[img],
            x=[lut['x0']],
            y=[lut['y0']],
            dw=[lut['dw']],
            dh=[lut['dh']]
        )

        # Update color mapper & colorbar
        palette, low, high, low_color, high_color, bar_ticks, overrides = get_color_mapper_params(
            st['cmap_name'], st['vmin'], st['vmax'], var_name=vname
        )
        self.ppi_mappers[i].palette = palette
        self.ppi_mappers[i].low = low
        self.ppi_mappers[i].high = high
        self.ppi_mappers[i].low_color = low_color
        self.ppi_mappers[i].high_color = high_color

        unit_str = f" [{vis['units']}]" if vis['units'] else ""
        self.ppi_plots[i].title.text = f"PPI {i+1}: {vname}{unit_str}"

        # Update hover tooltip unit string
        hover = self.ppi_hovers[i]
        hover.tooltips = [
            ("Value", f"@image{{0.00}}{unit_str}"),
            ("Azimuth", "$x{custom}"),
            ("Range", "$y{custom}"),
            ("Height", "$sx{custom}"),
            ("(x, y)", "$sy{custom}"),
        ]

        # Swap ColorBar on plot so ticks and overrides fully refresh in BokehJS
        cbar = self.ppi_colorbars[i]
        if hasattr(cbar, 'ticker') and hasattr(cbar.ticker, 'ticks') and bar_ticks:
            cbar.ticker.ticks = list(bar_ticks)
        if overrides is not None:
            cbar.major_label_overrides = dict(overrides)
        cbar.title = unit_str

        p = self.ppi_plots[i]
        p.right = [r for r in p.right if r != cbar]
        ticker = FixedTicker(ticks=list(bar_ticks)) if bar_ticks else BasicTicker()
        new_cbar = ColorBar(
            color_mapper=self.ppi_mappers[i],
            title=unit_str,
            width=14,
            padding=4,
            label_standoff=4,
            ticker=ticker,
            major_label_overrides=dict(overrides) if overrides else {}
        )
        p.add_layout(new_cbar, 'right')
        self.ppi_colorbars[i] = new_cbar

    def update_radial_line(self):
        if len(self.az_deg) == 0:
            return
        az = float(self.az_deg[self.current_az_idx])
        rad = np.radians(az)
        r_end = float(min(self.max_range_km, self.data_max_range_km))
        self.source_radial_line.data = dict(
            x=[0.0, float(r_end * np.sin(rad))],
            y=[0.0, float(r_end * np.cos(rad))]
        )

    # --------------------------------------------------------------------------
    # Event Handlers
    # --------------------------------------------------------------------------

    def on_tap_ppi(self, event):
        """
        Click anywhere on any PPI to select that radial (azimuth).
        """
        if event.x is None or event.y is None:
            return
        if abs(event.x) < 1e-9 and abs(event.y) < 1e-9:
            return
        clicked_az = float(np.degrees(np.arctan2(event.x, event.y)) % 360.0)
        self._select_azimuth_deg(clicked_az)

    def _select_azimuth_deg(self, target_az_deg):
        az_arr = np.asarray(self.az_deg, dtype=float) % 360.0
        diff = np.abs(((az_arr - (target_az_deg % 360.0) + 180.0) % 360.0) - 180.0)
        self.current_az_idx = int(np.argmin(diff))
        self.inp_az.value = round(float(self.az_deg[self.current_az_idx]), 1)
        self.update_radial_line()
        self._build_radial_plot()

    def step_azimuth(self, delta):
        if len(self.az_deg) == 0:
            return
        self.current_az_idx = (self.current_az_idx + delta) % len(self.az_deg)
        self.inp_az.value = round(float(self.az_deg[self.current_az_idx]), 1)
        self.update_radial_line()
        self._build_radial_plot()

    def apply_azimuth_input(self):
        try:
            val = float(self.inp_az.value)
        except (TypeError, ValueError):
            return
        self._select_azimuth_deg(val)

    def on_radial_x_changed(self, new_mode):
        self.radial_x_mode = new_mode
        self._build_radial_plot()

    def _set_time_index(self, new_idx):
        new_idx = max(0, min(int(new_idx), self.n_times - 1))
        if new_idx == self.current_time_idx:
            return
        self.current_time_idx = new_idx
        self._slice_cache.clear()

        self._updating_time_widgets = True
        try:
            self.sel_time.value = str(self.current_time_idx)
            self.slider_time.value = self.current_time_idx
        finally:
            self._updating_time_widgets = False

        for i in range(3):
            self.update_ppi_panel(i)
        self._update_ml_overlay()
        self._update_entropy_overlay()
        self._update_riming_overlay()
        self._build_radial_plot()

    def on_time_select_changed(self, new_val):
        if self._updating_time_widgets:
            return
        try:
            self._set_time_index(int(new_val))
        except (TypeError, ValueError):
            pass

    def on_time_slider_changed(self, new_val):
        if self._updating_time_widgets:
            return
        self._set_time_index(int(new_val))

    def step_time(self, delta):
        self._set_time_index(self.current_time_idx + delta)

    def _reload_sweep_and_refresh(self):
        self._slice_cache.clear()
        self.current_ds = self.mgr.load_sweep(self.current_date, self.current_elev_key)
        self.all_vars = self.mgr.get_available_variables(self.current_ds)
        self.n_times = int(self.current_ds.sizes.get('time', 1))
        self.current_time_idx = max(0, min(self.current_time_idx, self.n_times - 1))

        self.az_deg, self.range_km, self.gr_km, self.z_m = self.mgr.get_sweep_geometry_1d(self.current_ds)
        self.elev_deg = self.mgr.get_elevation_deg(self.current_ds)
        self.alt_m = self.mgr.get_altitude_m(self.current_ds)
        self.data_max_range_km = float(np.ceil(np.nanmax(self.gr_km))) if len(self.gr_km) > 0 else 60.0
        if self.current_az_idx >= len(self.az_deg):
            self.current_az_idx = 0

        # Sync time controls
        self._updating_time_widgets = True
        try:
            self.sel_time.options = self._get_time_options()
            self.sel_time.value = str(self.current_time_idx)
            self.slider_time.end = max(1, self.n_times - 1)
            self.slider_time.value = self.current_time_idx
            disabled_t = (self.n_times <= 1)
            self.sel_time.disabled = disabled_t
            self.btn_prev_time.disabled = disabled_t
            self.btn_next_time.disabled = disabled_t
            self.slider_time.disabled = disabled_t
        finally:
            self._updating_time_widgets = False

        # Ensure selected variables exist in the new dataset
        defaults = self.mgr.get_default_variables(self.all_vars)
        for i in range(3):
            self.panel_ui[i]['var'].options = self.all_vars
            if self.ppi_settings[i]['var'] not in self.all_vars:
                v_new = defaults[i]
                vis = get_var_vis_info(v_new, self.current_ds[v_new], compute_percentiles=False)
                if not vis['from_visdict']:
                    vis = get_var_vis_info(v_new, xr.DataArray(self._get_var_slice_2d(v_new), attrs=self.current_ds[v_new].attrs), compute_percentiles=True)
                c_name = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
                self.ppi_settings[i] = {
                    'var': v_new, 'cmap_name': c_name, 'vmin': vis['vmin'], 'vmax': vis['vmax'], 'vis': vis
                }
                self.panel_ui[i]['var'].value = v_new
                self.panel_ui[i]['cmap'].value = c_name
                self.panel_ui[i]['vmin'].value = vis['vmin']
                self.panel_ui[i]['vmax'].value = vis['vmax']

        for v in self.all_vars:
            if v not in self.var_styles:
                idx = self.all_vars.index(v)
                vis = get_var_vis_info(v, self.current_ds[v], compute_percentiles=False)
                self.var_styles[v] = {
                    'color': DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)],
                    'linestyle': LINE_STYLES[(idx // len(DEFAULT_PALETTE)) % len(LINE_STYLES)],
                    'ymin': vis['vmin'],
                    'ymax': vis['vmax'],
                    'units': vis['units'],
                    'needs_calc': not vis['from_visdict'],
                }

        self.active_1d_vars = [v for v in self.active_1d_vars if v in self.all_vars]
        if not self.active_1d_vars:
            self.active_1d_vars = list(defaults)
        self.chk_1d.labels = self.all_vars
        self.chk_1d.active = [self.all_vars.index(v) for v in self.active_1d_vars if v in self.all_vars]

        self._detect_overlay_availability()
        self._rebuild_polar_lut()
        self._update_polar_grid()
        self.update_all()

    def on_date_changed(self, new_date):
        if new_date in self.dates and new_date != self.current_date:
            self.current_date = new_date
            self.doc.title = f"Interactive PPI Visualizer - {self.current_date}"
            self.elev_options = self.mgr.get_elevation_options(self.current_date)
            valid_keys = [k for k, _ in self.elev_options]
            if self.current_elev_key not in valid_keys:
                self.current_elev_key = self.elev_options[0][0]
            self.sel_elev.options = self.elev_options
            self.sel_elev.value = self.current_elev_key
            self.sel_elev.disabled = (len(self.elev_options) <= 1)
            self._reload_sweep_and_refresh()

    def step_date(self, delta):
        idx = self.dates.index(self.current_date) + delta
        if 0 <= idx < len(self.dates):
            self.sel_date.value = self.dates[idx]

    def on_elev_changed(self, new_elev_key):
        if new_elev_key != self.current_elev_key:
            self.current_elev_key = new_elev_key
            self._reload_sweep_and_refresh()

    def apply_max_range(self):
        try:
            new_r = float(self.inp_max_r.value)
        except (TypeError, ValueError):
            return
        if new_r <= 0:
            return
        self.max_range_km = new_r
        self._rebuild_polar_lut()
        self._update_polar_grid()

        if self.ppi_plots:
            self.ppi_plots[0].x_range.start = -self.max_range_km
            self.ppi_plots[0].x_range.end = self.max_range_km
            self.ppi_plots[0].y_range.start = -self.max_range_km
            self.ppi_plots[0].y_range.end = self.max_range_km

        for i in range(3):
            self.update_ppi_panel(i)
        self.update_radial_line()
        self._build_radial_plot()

    def on_panel_var_selected(self, panel_idx, new_var):
        if not new_var or new_var not in self.current_ds:
            return
        vis = get_var_vis_info(new_var, self.current_ds[new_var], compute_percentiles=False)
        if not vis['from_visdict']:
            sl = self._get_var_slice_2d(new_var)
            vis = get_var_vis_info(new_var, xr.DataArray(sl, attrs=self.current_ds[new_var].attrs), compute_percentiles=True)
        ui = self.panel_ui[panel_idx]
        ui['vmin'].value = vis['vmin']
        ui['vmax'].value = vis['vmax']
        c_name = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
        ui['cmap'].value = c_name

    def apply_panel_config(self, panel_idx):
        ui = self.panel_ui[panel_idx]
        vname = ui['var'].value
        cmap_name = ui['cmap'].value
        try:
            vmin = float(ui['vmin'].value)
            vmax = float(ui['vmax'].value)
        except (TypeError, ValueError):
            return
        if vmin >= vmax:
            return

        vis = get_var_vis_info(vname, self.current_ds[vname], compute_percentiles=False)
        vis['cmap_name'] = cmap_name
        vis['vmin'] = vmin
        vis['vmax'] = vmax

        self.ppi_settings[panel_idx] = {
            'var': vname,
            'cmap_name': cmap_name,
            'vmin': vmin,
            'vmax': vmax,
            'vis': vis
        }

        self.update_ppi_panel(panel_idx)

    def reset_panel_defaults(self, panel_idx):
        ui = self.panel_ui[panel_idx]
        vname = ui['var'].value
        vis = get_var_vis_info(vname, self.current_ds[vname], compute_percentiles=False)
        if not vis['from_visdict']:
            sl = self._get_var_slice_2d(vname)
            vis = get_var_vis_info(vname, xr.DataArray(sl, attrs=self.current_ds[vname].attrs), compute_percentiles=True)
        c_choice = vis['cmap_name'] if vis['cmap_name'] in CMAP_OPTIONS else 'miub2'
        self.ppi_settings[panel_idx] = {
            'var': vname,
            'cmap_name': c_choice,
            'vmin': vis['vmin'],
            'vmax': vis['vmax'],
            'vis': vis
        }
        ui['cmap'].value = c_choice
        ui['vmin'].value = vis['vmin']
        ui['vmax'].value = vis['vmax']
        self.update_ppi_panel(panel_idx)

    def on_checkbox_change(self, active_indices):
        self.active_1d_vars = [self.all_vars[idx] for idx in active_indices if idx < len(self.all_vars)]
        self._build_radial_plot()
        self._refresh_styles_tab()

    def select_all_1d_vars(self):
        self.chk_1d.active = list(range(len(self.all_vars)))
        self.active_1d_vars = list(self.all_vars)
        self._build_radial_plot()
        self._refresh_styles_tab()

    def clear_all_1d_vars(self):
        self.chk_1d.active = []
        self.active_1d_vars = []
        self._build_radial_plot()
        self._refresh_styles_tab()

    def sync_1d_with_ppis(self):
        ppi_v = [s['var'] for s in self.ppi_settings]
        active_indices = [self.all_vars.index(v) for v in ppi_v if v in self.all_vars]
        self.chk_1d.active = active_indices
        self.active_1d_vars = list(ppi_v)
        self._build_radial_plot()
        self._refresh_styles_tab()

    def _refresh_styles_tab(self):
        widgets = [Div(text="<b>Active Radial Curve Range & Style:</b>", margin=(4, 0, 4, 0))]
        for v in self.active_1d_vars:
            self._ensure_var_style_computed(v)
            st = self.var_styles[v]
            lbl = Div(text=f"<b>{v}</b>", margin=(6, 0, 2, 0))
            inp_ymin = NumericInput(title="ymin:", value=st['ymin'], mode="float", width=110)
            inp_ymax = NumericInput(title="ymax:", value=st['ymax'], mode="float", width=110)
            sel_dash = Select(title="Line Style:", value=st['linestyle'], options=LINE_STYLES, width=110)
            sel_col = Select(title="Color:", value=st['color'], options=DEFAULT_PALETTE, width=110)

            def make_handler(var_name, ymin_w, ymax_w, dash_w, col_w):
                def apply_style():
                    self.var_styles[var_name]['ymin'] = float(ymin_w.value)
                    self.var_styles[var_name]['ymax'] = float(ymax_w.value)
                    self.var_styles[var_name]['linestyle'] = dash_w.value
                    self.var_styles[var_name]['color'] = col_w.value
                    self._build_radial_plot()
                return apply_style

            handler = make_handler(v, inp_ymin, inp_ymax, sel_dash, sel_col)
            btn_upd = Button(label=f"Update {v}", button_type="primary", width=225)
            btn_upd.on_click(handler)

            widgets.extend([lbl, row(inp_ymin, inp_ymax), row(sel_col, sel_dash), btn_upd])

        self.style_controls_box.children = widgets


# ==============================================================================
# Main Entry Point & Server Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Plan Position Indicator (PPI) Visualizer (Bokeh Backend)."
    )
    parser.add_argument(
        "paths", nargs="*", default=[DEFAULT_PATH_PPIS],
        help=f"File path(s), directory, or glob pattern to PPI data (default: {DEFAULT_PATH_PPIS})"
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
        "--time-idx", type=int, default=0,
        help="Initial timestep index to display (default: 0)"
    )
    parser.add_argument(
        "--max-range", type=float, default=None,
        help="Maximum ground range to display in km (default: auto from sweep)"
    )

    args = parser.parse_args()

    print("Locating PPI files...")
    files = find_files_from_input(args.paths)
    if not files:
        print(f"No files matched by inputs: {args.paths}")
        print("Please provide a valid file path, glob pattern, or directory containing PPI data.")
        sys.exit(1)

    print(f"Found {len(files)} file(s).")
    mgr = PPIDatasetManager(files)
    print(f"Available dates ({len(mgr.date_list)}): {mgr.date_list[:5]}{'...' if len(mgr.date_list) > 5 else ''}")

    try:
        from bokeh.server.server import Server
        from tornado.ioloop import IOLoop
    except ImportError:
        print("Error: Bokeh and Tornado must be installed to run the interactive server.")
        print("Please run within an environment with Bokeh installed (e.g. conda activate wradlib7).")
        sys.exit(1)

    def bkapp(doc):
        PPIBokehApp(
            mgr, doc,
            initial_time_idx=args.time_idx,
            initial_max_range=args.max_range
        )

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
    print(f" Interactive PPI Visualizer running at: http://localhost:{port}/")
    print(f" Features: Linked Polar PPIs, Hover Overlays, Click-to-Select Radial")
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


if __name__.startswith('bokeh_app'):
    paths = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_PATH_PPIS]
    files = find_files_from_input(paths)
    mgr = PPIDatasetManager(files)
    PPIBokehApp(mgr, curdoc())
elif __name__ == "__main__":
    main()
