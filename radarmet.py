import os
import re
import glob
import datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import wradlib as wrl
import scipy
import h5py
from osgeo import gdal, osr
from scipy.spatial import cKDTree


def get_xpol_path(path=None, start_time=dt.datetime.today(), loc='boxpol'):
    """Create path of BoXPol/JuXPol radar data files.

    Parameter
    ---------
    path : str
        Path to root folder of radar data,
        defaults to None (/automount/radar/scans or /automount/radar-archiv/scans)
    start_time : datetime.datetime
        datetime - object to select correct folder
    loc : str
        "boxpol" or "juxpol" (not case sensitive)

    Return
    ------
    radar_path : str
        Path to radar data

    """
    loc = "" if loc.lower()[0:2] == "bo" else "_juelich"
    if path is None:
        ins = "-archiv" if start_time < dt.datetime(2015, 1, 1) else ""
        path = f"/automount/radar{ins}/scans{loc}"
        if not os.path.exists(path):
            path = os.environ["RADARMET_DATA"]
    radar_path = os.path.join(path, "{0}/{0}-{1:02}/{0}-{1:02d}-{2:02d}")
    return radar_path.format(start_time.year, start_time.month, start_time.day)


def get_file_date_regex(filename):
    """Get regex from filename.
    """
    # regex for ""%Y-%m-%d--%H:%M:%S"
    reg0 = r"\d{4}.\d{2}.\d{2}..\d{2}.\d{2}.\d{2}"
    # regex for "%Y%m%d%H%M%S"
    reg1 = r"\d{14}"
    match = re.search(reg0, os.path.basename(filename))
    return reg1 if match is None else reg0


def get_datetime_from_filename(filename, regex):
    """Get datetime from filename.
    """
    fmt = "%Y%m%d%H%M%S"
    match = re.search(regex, os.path.basename(filename))
    match = "".join(re.findall(r"[0-9]+", match.group()))
    return dt.datetime.strptime(match, fmt)


def create_filelist(path_glob, starttime, endtime):
    """Create filelist from path_glob and filename dates
    """
    file_names = sorted(glob.glob(path_glob))
    regex = get_file_date_regex(file_names[0])
    for fname in file_names:
        time = get_datetime_from_filename(fname, regex)
        if time >= starttime and time < endtime:
            yield fname


def get_discrete_cmap(ticks, colors, bad="white", over=None, under=None):
    """Create discrete colormap.

    Parameters
    ----------
    ticks : int | sequence
        number of ticks or sequence of ticks
    colors : colormap | sequence
        colormap or sequence of colors
    bad : color
    over : color
    under : color

    Returns
    -------
    matplotlib.colors.ListedColormap
    """
    ticks = ticks if isinstance(ticks, int) else len(ticks)
    if isinstance(colors, (str, mpl.colors.Colormap)):
        cmap = mpl.cm.get_cmap(colors)
        colors = cmap(np.linspace(0, 1, ticks + 1))
    cmap = mpl.colors.ListedColormap(colors[1:-1])
    if over is None:
        over = colors[-1]
    if under is None:
        under = colors[0]
    cmap.set_under(under)
    cmap.set_over(over)
    cmap.set_bad(color=bad)
    return cmap


def get_discrete_norm(ticks):
    """Return discrete boundary norm.

    Parameters
    ----------
    ticks : sequence
        sequence of ticks

    Returns
    -------
    matplotlib.colors.BoundaryNorm
    """
    return mpl.colors.BoundaryNorm(ticks, len(ticks) - 1)


def plot_reference_colorbar(ticks, cmap, ax, **kwargs):
    """Plot reference colorbar

    Parameters
    ----------
    ticks : sequence
        sequence of ticks
    cmap : cmap instance
    ax : axes instance

    Keyword Arguments
    -----------------
    Additonal Keywords for colorbar

    Returns
    -------
    matplotlib.colorbar instance

    """
    vmin = ticks[0] - np.diff(ticks)[0]
    vmax = ticks[-1] + np.diff(ticks)[-1]
    vmax2 = ticks[-1] + 2 * np.diff(ticks)[-1]
    y = [vmin] + list(ticks) + [vmax] + [vmax2]
    x = np.arange(2)
    data = np.repeat(np.array(y, dtype=np.float), 2).reshape(-1, 2)
    data[-2, :] = np.nan
    norm = get_discrete_norm(ticks)
    pm = ax.pcolormesh(x, y, data,
                       cmap=cmap, norm=norm,
                       )
    cb = plt.colorbar(pm, ax=ax,
                      ticks=ticks,
                      **kwargs,
                      )
    return cb


colors_prabhakar = np.array([[0.00, 1.00, 1.00],
                             [0.00, 0.70, 0.93],
                             [0.00, 0.00, 1.00],
                             [0.50, 1.00, 0.00],
                             [0.40, 0.80, 0.00],
                             [0.27, 0.55, 0.00],
                             [1.00, 1.00, 0.00],
                             [0.80, 0.80, 0.00],
                             [1.00, 0.65, 0.00],
                             [1.00, 0.27, 0.00],
                             [0.80, 0.22, 0.00],
                             [0.55, 0.15, 0.00],
                             [1.00, 0.00, 1.00],
                             [0.58, 0.44, 0.86]])

cmap_prabhakar = mpl.colors.ListedColormap(colors_prabhakar)
mpl.cm.register_cmap("miub2", cmap=cmap_prabhakar)

visdict14 = dict(ZH=dict(ticks=np.arange(-10,55,5),
                         contours=[0, 5, 10, 15, 20, 25, 30, 35],
                         cmap=cmap_prabhakar,
                         name=r'Horizontal Reflectivity (dBz)',
                         short='$\mathrm{\mathsf{Z_{H}}}$'),
                DBZH=dict(ticks=np.arange(-10,55,5),
                         contours=[0, 5, 10, 15, 20, 25, 30, 35],
                         cmap=cmap_prabhakar,
                         name=r'Horizontal Reflectivity (dBz)',
                         short='$\mathrm{\mathsf{Z_{H}}}$'),
                TH=dict(ticks=np.arange(-10,55,5),
                         contours=[0, 5, 10, 15, 20, 25, 30, 35],
                         cmap=cmap_prabhakar,
                         name=r'Total Reflectivity (dBz)',
                         short='$\mathrm{\mathsf{Z_{H}}}$'),
                DBTH=dict(ticks=np.arange(-10,55,5),
                         contours=[0, 5, 10, 15, 20, 25, 30, 35],
                         cmap=cmap_prabhakar,
                         name=r'Total Reflectivity (dBz)',
                         short='$\mathrm{\mathsf{Z_{H}}}$'),
                 ZDR=dict(ticks=np.array([-1., -0.1, 0, 0.1, 0.2, 0.3, 0.4, .5, 0.6, 0.8, 1.0, 2., 3.0]),
                          contours=np.array([-1, -0.3, 0]),
                          cmap=cmap_prabhakar,
                          name=r'Differential Reflectivity (dB)',
                          short='$\mathrm{\mathsf{Z_{DR}}}$'),
                 RHOHV=dict(ticks=np.array([.7, .8, .85, .9, .92, .94, .95, .96, .97, .98, .99, .995, .998]),
                            cmap=cmap_prabhakar,
                            name=r'Crosscorrelation Coefficient ()',
                            short='$\mathrm{\mathsf{RHO_{HV}}}$'),
                 KDP=dict(ticks=np.array([-0.5, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]),
                          cmap=cmap_prabhakar,
                          name=r'Specific Differental Phase (°/km)',
                          short='$\mathrm{\mathsf{K_{DP}}}$'),
                 TEMP=dict(contours=[5, 0, -5, -10, -15, -20, -25, -30, -35, -40]),
                 PHI=dict(ticks=np.array([0,5,10,15,20,25,30,40,50,60,70,80,90]),
                          cmap=cmap_prabhakar,
                          name=r'Differental Phase (°)',
                          short='$\mathrm{\mathsf{\Phi_{DP}}}$'),
                 PHIDP=dict(ticks=np.array([0,5,10,15,20,25,30,40,50,60,70,80,90]),
                          cmap=cmap_prabhakar,
                          name=r'Differental Phase (°)',
                          short='$\mathrm{\mathsf{\Phi_{DP}}}$'),
                 UPHIDP=dict(ticks=np.array([0,5,10,15,20,25,30,40,50,60,70,80,90]),
                          cmap=cmap_prabhakar,
                          name=r'Differental Phase (°)',
                          short='$\mathrm{\mathsf{\Phi_{DP}}}$'),
                 HMC=dict(norm=mpl.colors.BoundaryNorm(np.arange(-0.5, 11 + 0.6, 1), 12),
                          ticks=np.arange(0, 11 + 1),
                          bounds=np.arange(-0.5, 11 + 0.6, 1),
                          cmap=mpl.colors.ListedColormap(['LightBlue', 'Blue', 'Lime', 'Black', 'Red','Yellow', \
                                                          'Fuchsia', 'LightPink', 'Cyan', 'Gray', 'DarkOrange','White']),
                          name=r'HMC - Zrnic et al. 2001 ',
                          short=r'$\HMC_{Z}$',
                          long_name=r'Hydrometeorclass',
                          labels=['Light Rain', 'Moderate Rain', 'Heavy Rain', 'Large Drops', 'Hail', 'Rain/Hail', \
                                  'Graupel/Hail', 'Dry Snow', 'Wet Snow', 'H Crystals','V Crystals','No Rain'],
                          labels_short=['LR', 'MR', 'HR', 'LD', 'HL', 'RH', 'GH', 'DS', 'WS', 'HC', 'VC', 'NR'])
                )


def plot_moment(mom, ticks, fig=None, ax=None, cmap=None, norm=None, cbar_kwargs=None):
    xlabel = 'X-Range [m]'
    ylabel = 'Y-Range [m]'

    if not ax.is_last_row():
        xlabel = ''
    if not ax.is_first_col():
        ylabel = ''

    # colorbar kwargs
    cbarkwargs = dict(extend="both",
                      extendrect=False,
                      extendfrac='auto',
                      pad=0.05,
                      fraction=0.1,
                     )
    
    if cbar_kwargs is not None:
        cbarkwargs.update(cbar_kwargs)

    cbar_extend = cbarkwargs.get("extend", None)

    # get norm
    if norm is None:
        norm = get_discrete_norm(ticks)
    # define cmap
    if cmap is None:
        cmap = get_discrete_cmap(ticks, colors_prabhakar)
    
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
                          cbar_kwargs=cbarkwargs,
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
    return im


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for _, sp in ax.spines.items():
        sp.set_visible(False)


def set_spine_direction(ax, direction):
    if direction in ["right", "left"]:
        ax.yaxis.set_ticks_position(direction)
        ax.yaxis.set_label_position(direction)
    elif direction in ["top", "bottom"]:
        ax.xaxis.set_ticks_position(direction)
        ax.xaxis.set_label_position(direction)
    else:
        raise ValueError("Unknown Direction: %s" % (direction,))

    ax.spines[direction].set_visible(True)


def create_lineplot(fig, subplot=111, xlabel=None, ylabel=None):

    if xlabel is None:
        xlabel = 'Range Bins'

    if ylabel is None:
        ylabel = ''

    host = fig.add_subplot(subplot)

    ax = host
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def add_axis(host, ylabel=None, pos=1.0):
    if ylabel is None:
        ylabel=''
    ax = host.twinx()
    host.spines["right"].set_visible(False)
    ax.spines["right"].set_position(("axes", pos))
    make_patch_spines_invisible(ax)
    set_spine_direction(ax, "right")
    ax.set_ylabel(ylabel)

    return ax


# phase processing
def filter_data(data, medwin):
    data.values = scipy.signal.medfilt2d(data.values, [1, medwin])
    return data

def filter_data_np(data, medwin):
    data = scipy.signal.medfilt2d(data, [1, medwin])
    return data

def gauss_kernel(width, sigma):
    dirac = np.zeros(width)
    dirac[int(width / 2)] = 1
    return scipy.ndimage.gaussian_filter1d(dirac, sigma=sigma)


def convolve(data, kernel, mode='same'):
    mask = np.isnan(data)
    out = np.convolve(np.where(mask, 0, data), kernel, mode=mode) / np.convolve(~mask, kernel, mode=mode)
    return out


def get_peaks(da, height=0):
    def process_peaks(arr, height=0):
        # Apply find_peaks
        arr = arr.copy()
        peaks, _ = scipy.signal.find_peaks(arr, height=height)
        try:
            peak = peaks[0]
        except IndexError:
            peak=0
        return peak
    return xr.apply_ufunc(process_peaks,
                          da,
                          input_core_dims=[["PHIDP_bin"]],
                          output_core_dims=[[]],
                          #output_sizes={"peaks": len(da)},
                          output_dtypes=((int)),
                          dask='parallelized',
                          vectorize=True,
                          kwargs=dict(height=height),
                          dask_gufunc_kwargs=dict(allow_rechunk=True),
                         )


def smooth_data(data, kernel):
    res = data.copy()
    for i, dat in enumerate(data.values):
        res[i] = convolve(dat, kernel)
    return res


def phase_offset(phioff, rng=3000.):
    """Calculate Phase offset.

    Parameter
    ---------
    phioff : xarray.DataArray
        differential phase array

    Keyword Arguments
    -----------------
    rng : float
        range in m to calculate system phase offset

    Return
    ------
    xarray.Dataset
        Dataset with variables PHIDP_OFFSET, start_range and stop_range
    """
    range_step = np.diff(phioff.range)[0]
    nprec = int(rng / range_step)
    if nprec % 2:
        nprec += 1

    # create binary array
    phib = xr.where(np.isnan(phioff), 0, 1)

    # take nprec range bins and calculate sum
    phib_sum = phib.rolling(range=nprec, center=True).sum(skipna=True)

    # get start range of first N consecutive precip bins
    start_range = phib_sum.idxmax(dim="range") - nprec // 2 * np.diff(phib_sum.range)[0]
    # get range of first non-nan value per ray
    #start_range = (~np.isnan(phioff)).idxmax(dim='range', skipna=True)
    # add range
    stop_range = start_range + rng
    # get phase values in specified range
    off = phioff.where((phioff.range >= start_range) & (phioff.range <= stop_range),
                       drop=True)
    # calculate nan median over range
    off = off.median(dim='range', skipna=True)
    return xr.Dataset(dict(PHIDP_OFFSET=off,
                           start_range=start_range,
                           stop_range=stop_range))


def kdp_from_phidp(da, winlen, min_periods=2):
    """Derive KDP from PHIDP (based on convolution filter).

    Parameter
    ---------
    da : xarray.DataArray
        array with differential phase data
    winlen : int
        size of window in range dimension

    Keyword Arguments
    -----------------
    min_periods : int
        minimum number of valid bins

    Return
    ------
    kdp : xarray.DataArray
        DataArray with specific differential phase values
    """
    dr = da.range.diff('range').median('range').values / 1000.
    print("range res [km]:", dr)
    print("processing window [km]:", dr * winlen)
    return xr.apply_ufunc(wrl.dp.kdp_from_phidp,
                          da,
                          input_core_dims=[["range"]],
                          output_core_dims=[["range"]],
                          dask='parallelized',
                          kwargs=dict(winlen=winlen, dr=dr, min_periods=min_periods),
                          dask_gufunc_kwargs=dict(allow_rechunk=True),
                          )


def phidp_from_kdp(da, winlen):
    """Derive PHIDP from KDP.

    Parameter
    ---------
    da : xarray.DataArray
        array with specific differential phase data
    winlen : int
        size of window in range dimension

    Return
    ------
    phi : xarray.DataArray
        DataArray with differential phase values
    """
    dr = da.range.diff('range').median('range').values / 1000.
    print("range res [km]:", dr)
    print("processing window [km]:", dr * winlen)
    return xr.apply_ufunc(scipy.integrate.cumtrapz,
                          da,
                          input_core_dims=[["range"]],
                          output_core_dims=[["range"]],
                          dask='parallelized',
                          kwargs=dict(dx=dr, initial=0.0, axis=-1),
                          ) * 2


def kdp_phidp_vulpiani(da, winlen, min_periods=2):
    """Derive KDP from PHIDP (based on Vulpiani).

    ParameterRHOHV_NC
    ---------
    da : xarray.DataArray
        array with differential phase data
    winlen : int
        size of window in range dimension

    Keyword Arguments
    -----------------
    min_periods : int
        minimum number of valid bins

    Return
    ------
    kdp : xarray.DataArray
        DataArray with specific differential phase values
    """
    dr = da.range.diff('range').median('range').values / 1000.
    print("range res [km]:", dr)
    print("processing window [km]:", dr * winlen)
    return xr.apply_ufunc(wrl.dp.process_raw_phidp_vulpiani,
                          da,
                          input_core_dims=[["range"]],
                          output_core_dims=[["range"], ["range"]],
                          dask='parallelized',
                          kwargs=dict(winlen=winlen, dr=dr,
                                      min_periods=min_periods),
                          dask_gufunc_kwargs=dict(allow_rechunk=True),
                          )


def xr_rolling(da, window, window2=None, method="mean", min_periods=2, **kwargs):
    """Apply rolling function `method` to 2D datasets

    Parameter
    ---------
    da : xarray.DataArray
        array with data to apply rolling function
    window : int
        size of window in range dimension

    Keyword Arguments
    -----------------
    window2 : int
        size of window in azimuth dimension
    method : str
        function name to apply
    min_periods : int
        minimum number of valid bins
    **kwargs : dict
        kwargs to feed to rolling function

    Return
    ------
    da_new : xarray.DataArray
        DataArray with applied rolling function
    """
    prng = window // 2
    srng = slice(prng, -prng)
    da_new = da.pad(range=prng, mode='reflect', reflect_type='odd')

    dim = dict(range=window)
    isel = dict(range=srng)

    if window2 is not None:
        paz = window2 // 2
        saz = slice(paz, -paz)
        da_new = da_new.pad(azimuth=paz, mode="wrap")
        dim.update(dict(azimuth=window2))
        isel.update(dict(azimuth=saz))

    rolling = da_new.rolling(dim=dim, center=True, min_periods=min_periods)

    da_new = getattr(rolling, method)(**kwargs)
    da_new = da_new.isel(**isel)
    return da_new


# Hydrometeor Classification

def msf_index_indep_xarray(msf_ds, obs):
    def wrap_digitize(data, bins=None):
        return np.digitize(data, bins)

    idp = msf_ds.idp.values
    bins = np.append(idp, idp[-1] + (idp[-1] - idp[-2]))
    idx = xr.apply_ufunc(wrap_digitize,
                         obs,
                         dask='parallelized',
                         kwargs=dict(bins=bins),
                         output_dtypes=['i4']) - 1
    # select bins
    idx = xr.where((idx >= 0) & (idx < bins.shape[0] - 2), idx, 0)
    return msf_ds.isel(idp=idx)


def trapezoid(msf, obs):
    ones = ((obs >= msf[..., 1]) & (obs <= msf[..., 2]))
    zeros = ((obs < msf[..., 0]) | (obs > msf[..., 3]))
    lower = ((obs >= msf[..., 0]) & (obs < msf[..., 1]))
    higher = ((obs > msf[..., 2]) & (obs <= msf[..., 3]))

    obs_lower = obs - msf[..., 0]
    msf_lower = msf[..., 1] - msf[..., 0]
    low = (obs_lower / msf_lower)

    obs_higher = obs - msf[..., 3]
    msf_higher = msf[..., 2] - msf[..., 3]
    high = (obs_higher / msf_higher)

    ret = xr.zeros_like(obs)  # * np.nan
    # ret = ret.where(zeros, 0).where(ones, 1).where(lower, low).where(higher, high)
    ret = xr.where(ones, 1, ret)
    ret = xr.where(lower, low, ret)
    ret = xr.where(higher, high, ret)
    return ret  # .where(ret == np.nan, 0)


def fuzzify(msf_ds, hmc_ds, msf_obs_mapping):
    fuzz_ds = xr.Dataset()
    for mf, hm in msf_obs_mapping.items():
        obs = hmc_ds[hm]
        msf = msf_ds[mf]
        fuzz_ds = fuzz_ds.assign({mf: trapezoid(msf, obs)})
    return fuzz_ds.transpose("hmc", ...)


def probability(data, weights):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing containing the membership probability values.
    weights : :class:`numpy:numpy.ndarray`
        Array of length (observables) containing the weights for
        each observable.

    Returns
    -------
    out : xarray.DataArray
        Array containing weighted hmc-membership probabilities.
    """
    out = data.to_array(dim="obs")
    w = weights.to_array(dim="obs")
    out = (out * w).sum("obs") / w.sum("obs")
    return out  # .transpose("hmc", ...)


def classify(data, threshold=0.0):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : np.ndarray
        Array which is of size (hmc-class, data.shape), containing the
        weighted hmc-membership probability values.

    Keyword Arguments
    -----------------
    threshold : float
        Threshold value where probability is considered no precip,
        defaults to 0

    Returns
    -------
    out : xr.DataArray
        DataArray containing containing probability scores.
        No precip is added on the top.
    """
    # handle no precipitation
    nop = xr.where(data.sum("hmc") / len(data.hmc) <= threshold, 1, 0)
    nop = nop.assign_coords({"hmc": "NP"}).expand_dims(dim="hmc", axis=-1)
    return xr.concat([data, nop], dim="hmc")


####### Satellite ###################

def convert_gpmrefl_grband_dfr(refl_gpm, radar_band=None):
    """
    Convert GPM reflectivity to ground radar band using the DFR relationship
    found in Louf et al. (2019) paper.
    Parameters:
    ===========
    refl_gpm:
        Satellite reflectivity field.
    radar_band: str
        Possible values are 'S', 'C', or 'X'
    Return:
    =======
    refl:
        Reflectivity conversion from Ku-band to ground radar band
    """
    if radar_band == "S":
        cof = np.array(
            [2.01236803e-07, -6.50694273e-06, 1.10885533e-03, -6.47985914e-02,
             -7.46518423e-02])
        dfr = np.poly1d(cof)
    elif radar_band == "C":
        cof = np.array(
            [1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01,
             5.33556919e-01])
        dfr = np.poly1d(cof)
    elif radar_band == "X":
        # Use of C band DFR relationship multiply by ratio
        cof = np.array(
            [1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01,
             5.33556919e-01])
        dfr = 3.2 / 5.5 * np.poly1d(cof)
    else:
        raise ValueError(f"Radar reflectivity band ({radar_band}) not supported.")

    return refl_gpm + dfr(refl_gpm)


import copy
import numpy as np


def convert_sat_refl_to_gr_band(refp, zp, zbb, bbwidth, radar_band='S'):
    """
    Convert the satellite reflectivity to S, C, or X-band using the Cao et al.
    (2013) method.
    Parameters
    ==========
    refp:
        Satellite reflectivity field.
    zp:
        Altitude.
    zbb:
        Bright band height.
    bbwidth:
        Bright band width.
    radar_band: str
        Possible values are 'S', 'C', or 'X'
    Return
    ======
    refp_ss:
        Stratiform reflectivity conversion from Ku-band to S-band
    refp_sh:
        Convective reflectivity conversion from Ku-band to S-band
    """
    # Set coefficients for conversion from Ku-band to S-band
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Snow
    as0 = [4.78e-2, 4.12e-2, 8.12e-2, 1.59e-1, 2.87e-1, 4.93e-1, 8.16e-1, 1.31e+0,
           2.01e+0, 2.82e+0, 1.74e-1]
    as1 = [1.23e-2, 3.66e-3, 2.00e-3, 9.42e-4, 5.29e-4, 5.96e-4, 1.22e-3, 2.11e-3,
           3.34e-3, 5.33e-3, 1.35e-2]
    as2 = [-3.50e-4, 1.17e-3, 1.04e-3, 8.16e-4, 6.59e-4, 5.85e-4, 6.13e-4, 7.01e-4,
           8.24e-4, 1.01e-3, -1.38e-3]
    as3 = [-3.30e-5, -8.08e-5, -6.44e-5, -4.97e-5, -4.15e-5, -3.89e-5, -4.15e-5,
           -4.58e-5, -5.06e-5, -5.78e-5, 4.74e-5]
    as4 = [4.27e-7, 9.25e-7, 7.41e-7, 6.13e-7, 5.80e-7, 6.16e-7, 7.12e-7, 8.22e-7,
           9.39e-7, 1.10e-6, 0]
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Hail
    ah0 = [4.78e-2, 1.80e-1, 1.95e-1, 1.88e-1, 2.36e-1, 2.70e-1, 2.98e-1, 2.85e-1,
           1.75e-1, 4.30e-2, 8.80e-2]
    ah1 = [1.23e-2, -3.73e-2, -3.83e-2, -3.29e-2, -3.46e-2, -2.94e-2, -2.10e-2,
           -9.96e-3, -8.05e-3, -8.27e-3, 5.39e-2]
    ah2 = [-3.50e-4, 4.08e-3, 4.14e-3, 3.75e-3, 3.71e-3, 3.22e-3, 2.44e-3, 1.45e-3,
           1.21e-3, 1.66e-3, -2.99e-4]
    ah3 = [-3.30e-5, -1.59e-4, -1.54e-4, -1.39e-4, -1.30e-4, -1.12e-4, -8.56e-5,
           -5.33e-5, -4.66e-5, -7.19e-5, 1.90e-5]
    ah4 = [4.27e-7, 1.59e-6, 1.51e-6, 1.37e-6, 1.29e-6, 1.15e-6, 9.40e-7, 6.71e-7,
           6.33e-7, 9.52e-7, 0]

    refp_ss = np.zeros(refp.shape) + np.NaN  # snow
    refp_sh = np.zeros(refp.shape) + np.NaN  # hail
    zmlt = zbb + bbwidth / 2.  # APPROXIMATION!
    zmlb = zbb - bbwidth / 2.  # APPROXIMATION!
    ratio = (zp - zmlb) / (zmlt - zmlb)

    iax, iay = np.where(ratio >= 1)
    # above melting layer
    if len(iax) > 0:
        dfrs = as0[10] + as1[10] * refp[iax, iay] + as2[10] * refp[iax, iay] ** 2 + as3[
            10] * refp[iax, iay] ** 3 + as4[10] * refp[iax, iay] ** 4
        dfrh = ah0[10] + ah1[10] * refp[iax, iay] + ah2[10] * refp[iax, iay] ** 2 + ah3[
            10] * refp[iax, iay] ** 3 + ah4[10] * refp[iax, iay] ** 4
        refp_ss[iax, iay] = refp[iax, iay] + dfrs
        refp_sh[iax, iay] = refp[iax, iay] + dfrh

    ibx, iby = np.where(ratio <= 0)
    if len(ibx) > 0:  # below the melting layer
        dfrs = as0[0] + as1[0] * refp[ibx, iby] + as2[0] * refp[ibx, iby] ** 2 + as3[
            0] * refp[ibx, iby] ** 3 + as4[0] * refp[ibx, iby] ** 4
        dfrh = ah0[0] + ah1[0] * refp[ibx, iby] + ah2[0] * refp[ibx, iby] ** 2 + ah3[
            0] * refp[ibx, iby] ** 3 + ah4[0] * refp[ibx, iby] ** 4
        refp_ss[ibx, iby] = refp[ibx, iby] + dfrs
        refp_sh[ibx, iby] = refp[ibx, iby] + dfrh

    imx, imy = np.where((ratio > 0) & (ratio < 1))
    if len(imx) > 0:  # within the melting layer
        ind = np.round(ratio[imx, imy]).astype(int)[0]
        dfrs = as0[ind] + as1[ind] * refp[imx, imy] + as2[ind] * refp[imx, imy] ** 2 + \
               as3[ind] * refp[imx, imy] ** 3 + as4[ind] * refp[imx, imy] ** 4
        dfrh = ah0[ind] + ah1[ind] * refp[imx, imy] + ah2[ind] * refp[imx, imy] ** 2 + \
               ah3[ind] * refp[imx, imy] ** 3 + ah4[ind] * refp[imx, imy] ** 4
        refp_ss[imx, imy] = refp[imx, imy] + dfrs
        refp_sh[imx, imy] = refp[imx, imy] + dfrh

    # Jackson Tan's fix for C-band
    if radar_band == 'C':
        deltas = 5.3 / 10.0 * (refp_ss - refp)
        refp_ss = refp + deltas
        deltah = 5.3 / 10.0 * (refp_sh - refp)
        refp_sh = refp + deltah
    elif radar_band == 'X':
        deltas = 3.2 / 10.0 * (refp_ss - refp)
        refp_ss = refp + deltas
        deltah = 3.2 / 10.0 * (refp_sh - refp)
        refp_sh = refp + deltah

    return refp_ss, refp_sh


def convert_to_Ku(refg, zg, zbb, radar_band='S'):
    '''
    From Liao and Meneghini (2009)
    Parameters
    ==========
    refg:
        Ground radar reflectivity field.
    zg:
        Altitude.
    zbb:
        Bright band height.
    bbwidth:
        Bright band width.
    radar_band: str
        Possible values are 'S', 'C', or 'X'
    Returns
    =======
    refg_ku:
        Ground radar reflectivity field converted to Ku-band.
    '''

    refg_ku = np.zeros(refg.shape) + np.NaN
    idx = np.where(zg >= zbb)

    #  Above bright band
    if len(idx) > 0:
        refg_ku[idx] = 0.185074 + 1.01378 * refg[idx] - 0.00189212 * refg[idx] ** 2

    # Below bright band
    ibx = np.where(zg < zbb)
    if len(ibx) > 0:
        refg_ku[ibx] = -1.50393 + 1.07274 * refg[ibx] + 0.000165393 * refg[ibx] ** 2

    #  Jackson Tan's fix for C-band
    if radar_band == 'C':
        delta = (refg_ku - refg) * 5.3 / 10.0
        refg_ku = refg + delta
    elif radar_band == 'X':
        delta = (refg_ku - refg) * 3.2 / 10.0
        refg_ku = refg + delta

    return refg_ku


def read_gpm(filename, bbox):
    """
    Read and organize the SR data (DPR).

    # Input:
    # ------
    filename ::: path to satellite radar data
    bbox     ::: ground radar bounding box

    # Output:
    # ------
    gpm_data ::: satellite data dict

    """
    from netCDF4 import Dataset

    scan = 'NS'  # NS, MS, HS
    pr_data = Dataset(filename, mode="r")
    lon = pr_data[scan].variables['Longitude']
    lat = pr_data[scan].variables['Latitude']

    poly = [[bbox['left'], bbox['bottom']],
            [bbox['left'], bbox['top']],
            [bbox['right'], bbox['top']],
            [bbox['right'], bbox['bottom']],
            [bbox['left'], bbox['bottom']]]
    mask = wrl.zonalstats.get_clip_mask(np.dstack((lon[:], lat[:])), poly)
    mask = np.nonzero(np.count_nonzero(mask, axis=1))
    lon = lon[mask]
    lat = lat[mask]

    # Height of DPR
    # dpr_alt = np.array(pr_data[scan]['navigation'].variables['dprAlt'][mask])
    # print(dpr_alt)

    year = pr_data[scan]['ScanTime'].variables['Year'][mask]
    month = pr_data[scan]['ScanTime'].variables['Month'][mask]
    dayofmonth = pr_data[scan]['ScanTime'].variables['DayOfMonth'][mask]
    # dayofyear = pr_data[scan]['ScanTime'].variables['DayOfYear'][mask]
    hour = pr_data[scan]['ScanTime'].variables['Hour'][mask]
    minute = pr_data[scan]['ScanTime'].variables['Minute'][mask]
    second = pr_data[scan]['ScanTime'].variables['Second'][mask]
    # secondofday = pr_data[scan]['ScanTime'].variables['SecondOfDay'][mask]
    millisecond = pr_data[scan]['ScanTime'].variables['MilliSecond'][mask]
    date_array = zip(year, month, dayofmonth,
                     hour, minute, second,
                     millisecond.astype(np.int32) * 1000)
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in
         date_array])

    # DPR Altitude in m
    dpr_alt = pr_data[scan]['navigation'].variables['dprAlt'][mask]
    # print ('dpr_alt: ', dpr_alt)

    sfc = pr_data[scan]['PRE'].variables['landSurfaceType'][mask]
    pflag = pr_data[scan]['PRE'].variables['flagPrecip'][mask]
    # print(np.unique(pflag))

    # bbflag = pr_data[scan]['CSF'].variables['flagBB'][mask]
    zbb = pr_data[scan]['CSF'].variables['heightBB'][mask]
    # print(zbb.dtype)
    bbwidth = pr_data[scan]['CSF'].variables['widthBB'][mask]

    qbb = pr_data[scan]['CSF'].variables['qualityBB'][mask]
    qtype = pr_data[scan]['CSF'].variables['qualityTypePrecip'][mask]
    ptype = pr_data[scan]['CSF'].variables['typePrecip'][mask]

    quality = pr_data[scan]['scanStatus'].variables['dataQuality'][mask]

    #### REFL KU
    refl = pr_data[scan]['SLV'].variables['zFactorCorrected'][mask]

    # dummy for extending MS swath to NS
    dummy = np.zeros((refl.shape[0], 12, refl.shape[2])) * np.nan

    ##### REFL KA
    refl_ka = pr_data['MS']['SLV'].variables['zFactorCorrected'][mask]
    refl_ka = np.concatenate((dummy, refl_ka, dummy), axis=1)
    # print(refl.shape, refl_ka.shape)

    #### KA and DFR
    refl_ku_m = pr_data['NS']['PRE'].variables['zFactorMeasured'][mask]

    refl_ka_m = pr_data['MS']['PRE'].variables['zFactorMeasured'][mask]
    refl_ka_m = np.concatenate((dummy, refl_ka_m, dummy), axis=1)

    # "Maske"
    refl_ka[refl_ka < 0] = np.nan
    refl_ka_m[refl_ka_m < 0] = np.nan
    refl_ku_m[refl_ku_m < 0] = np.nan
    # dfr = wrl.trafo.idecibel(refl_ku_m) - wrl.trafo.idecibel(refl_ka_m)
    dfr = refl_ku_m - refl_ka_m
    # dfr[np.isnan()]

    # print('DFR: ', dfr.shape)
    clutter = pr_data['NS']['PRE'].variables['binClutterFreeBottom'][mask]

    zenith = pr_data[scan]['PRE'].variables['localZenithAngle'][mask]

    temp = pr_data['NS']['DSD'].variables['phase'][mask]
    temp = temp.copy().astype(float)
    temp[temp == 100] = np.nan
    temp[temp == 200] = np.nan
    temp[temp == 255] = np.nan
    temp[temp == 125] = np.nan
    temp[temp == 175] = np.nan

    temp[temp < 100] = temp[temp < 100] - 100.
    temp[temp > 200] = temp[temp > 200] - 200.

    pr_data.close()

    # Check for bad data
    if max(quality) != 0:
        raise ValueError('GPM contains Bad Data')

    pflag = pflag.astype(np.int8)

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError('GPM Dimensions do not match! '
                         'Needed 3, given {0}'.format(ndim))

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    refl = np.flip(refl, axis=-1)
    refl_ka = np.flip(refl_ka, axis=-1)
    dfr = np.flip(dfr, axis=-1)

    # Change pflag=1 to pflag=2 to be consistent with 'Rain certain' in TRMM
    pflag[pflag == 1] = 2

    # Simplify the precipitation types
    ptype = (ptype / 1e7).astype(np.int16)

    # Simplify the surface types
    imiss = (sfc == -9999)
    sfc = (sfc / 1e2).astype(np.int16) + 1
    sfc[imiss] = 0

    # Set a quality indicator for the BB and precip type data
    # TODO: Why is the `quality` variable overwritten?

    quality = np.zeros((nscan, nray), dtype=np.uint8)

    i1 = ((qbb == 0) | (qbb == 1)) & (qtype == 1)
    quality[i1] = 1

    i2 = ((qbb > 1) | (qtype > 2))
    quality[i2] = 2

    gpm_data = {}
    gpm_data.update({'nscan': nscan, 'nray': nray, 'nbin': nbin,
                     'date': pr_time, 'lon': lon, 'lat': lat,
                     'pflag': pflag, 'ptype': ptype, 'zbb': zbb,
                     'bbwidth': bbwidth, 'sfc': sfc, 'quality': quality,
                     'refl': refl, 'zenith': zenith, 'dpr_alt': dpr_alt,
                     'refl_ka': refl_ka,
                     'dfr': dfr, 'temp': temp})

    return gpm_data


def read_gr_sweep(gr_file_path, offset_z=0, offset_phi=0, offset_zdr=0):
    """
    Read and organize the GR data for one sweep.The following code reads data
    in ODIM H5 format. If your GR data is in some other format
    respective adaptions are needed.

    # Input:
    # ------
    gr_file_path ::: path to graund radar sweep data
    offset_z     ::: Refl. offset in dB
    offset_phi   ::: System PHI
    offset_zdr   ::: Diff. Refl offst in dB

    # Output:
    # ------
    ds0_masked ::: Sweep Data with PPB Quality Index
    More description in the data itself.

    """

    def noise_correction(ds, noise_level):
        # noise calculations
        snr = ds.DBZH - 20 * np.log10(
            ds.range * 0.001) - noise_level - 0.033 * ds.range / 1000
        rho = ds.RHOHV * np.sqrt(1. + 1. / 10. ** (snr * 0.1))
        return ds.assign({'SNR': snr, 'RHOHV': rho})

    def xr_rolling(da, window, method='mean'):
        da_new = da.pad(range=window // 2, mode='reflect', reflect_type='odd')
        da_new = getattr(da_new.rolling(range=window, center=True), method)()
        da_new = da_new.isel(range=slice(window // 2, -window // 2))
        return da_new

    # read one ppi data
    # umstellen open_odim_dataset
    vol = wrl.io.open_gamic_dataset(gr_file_path)
    print(vol)
    ds0 = vol[0]

    # Offsets and Noise Processing
    # ------------------------------
    # keep uncorrected RHOHV
    ds0 = ds0.assign({'URHOHV': ds0.RHOHV})

    # add the offsets
    ds0['DBZH'] += offset_z
    ds0['PHIDP'] += offset_phi
    ds0['ZDR'] += offset_zdr

    # RHOHV noise correction
    # ds0 = ds0.pipe(noise_correction, -22)
    # ds0 = ds0.pipe(noise_correction, -22)

    # georeference data
    ds0 = ds0.pipe(wrl.georef.georeference_dataset)

    ## Phase Processing
    ## -----------------
    window = 13
    phi_ma = ds0.PHIDP.pipe(xr_rolling, window, method='median')

    # mask phi
    phi_masked = phi_ma.where((ds0.RHOHV >= 0.95) & (ds0.DBZH >= 0.))

    # not used for moment
    first = np.isfinite(phi_masked).argmax(dim='range')
    last = phi_masked.sizes['range'] - np.isfinite(
        phi_masked.sortby(['range'], ascending=False)).argmax(dim='range')

    # nur bei neuem wradlib
    kdp = xr.apply_ufunc(wrl.dp.kdp_from_phidp,
                         phi_masked,
                         input_core_dims=[["range"]],
                         output_core_dims=[["range"]],
                         kwargs={'winlen': 21, 'dr': 0.125, 'min_periods': 3}, )

    phidp = xr.apply_ufunc(scipy.integrate.cumtrapz,
                           kdp,
                           input_core_dims=[["range"]],
                           output_core_dims=[["range"]],
                           kwargs=dict(dx=0.125, initial=0.0, axis=-1), ) * 2

    phidp2, kdp2 = xr.apply_ufunc(wrl.dp.process_raw_phidp_vulpiani,
                                  ds0.PHIDP.where(
                                      (ds0.RHOHV >= 0.95) & (ds0.DBZH >= 0.)),
                                  input_core_dims=[["range"]],
                                  output_core_dims=[["range"], ["range"]],
                                  kwargs=dict(winlen=15, dr=0.125, th1=-0.5, th2=5,
                                              th3=-80, min_periods=3))

    ds0 = ds0.assign({'PHI_SMOOTH': phidp})
    ds0 = ds0.assign({'PHI_MASKED': phi_masked})
    ds0 = ds0.assign({'KDP1_NEW': kdp})
    ds0 = ds0.assign({'PHI1_NEW': phidp})
    ds0 = ds0.assign({'KDP2_NEW': kdp2})
    ds0 = ds0.assign({'PHI2_NEW': phidp2})

    ds0_masked = ds0.where(ds0.RHOHV >= 0.7)

    ds0_masked = ds0_masked.assign({'height': ds0.z})

    sitecoords = (
    ds0_masked.longitude.data, ds0_masked.latitude.data, ds0_masked.altitude.data)

    # BB and Q
    nrays = 360  # number of rays
    nbins = ds0_masked.range.shape[0]  # number of range bins
    el = ds0_masked.elevation.data[0]  # vertical antenna pointing angle (deg)
    bw = 1.0  # half power beam width (deg)
    range_res = ds0_masked.range.data[1] - ds0_masked.range.data[
        0]  # range resolution (meters)

    r = np.arange(nbins) * range_res
    beamradius = wrl.util.half_power_radius(r, bw)

    coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
    coords = wrl.georef.spherical_to_proj(coord[..., 0],
                                          coord[..., 1],
                                          coord[..., 2], sitecoords)
    lon = coords[..., 0]
    lat = coords[..., 1]
    alt = coords[..., 2]

    polcoords = coords[..., :2]

    rlimits = (lon.min(), lat.min(), lon.max(), lat.max())

    ds = wrl.io.open_raster(
        '/automount/ftp/radar/wradlib-data/geo/bonn_dem.tif')
    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds,
                                                                         nodata=-32768.)

    # Clip the region inside our bounding box
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
    rastervalues = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]

    # Map rastervalues to polar grid points
    polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoords, rastervalues,
                                                    polcoords, order=3,
                                                    prefilter=False)

    PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
    PBB = np.ma.masked_invalid(PBB)
    CBB = wrl.qual.cum_beam_block_frac(PBB)

    def QBBF(CBB):
        """
        Quality Index for PBB
        Crisologo 2018
        """

        Qbbf = np.zeros_like(CBB) * np.nan
        Qbbf[CBB < 0.1] = 1
        Qbbf[CBB > 0.5] = 0
        Qbbf[(CBB <= 0.5) & (CBB >= 0.1)] = (
                    1 - (CBB[(CBB <= 0.5) & (CBB >= 0.1)] - 0.1) / 0.4)

        return Qbbf

    # ds0_masked = ds0_masked.assign({'QBBF': QBBF(CBB)})
    ds0_masked = ds0_masked.assign({"QBBF": (
    ["azimuth", "range"], QBBF(CBB))})

    return ds0_masked


def get_bbox(gr_data):
    """
    Calculation of bounding box around the ground radar

    # Input:
    # ------
    gr_data ::: graund radar sweep data

    # Output:
    # ------
    bbox ::: Coords for bounding box

    """
    # number of rays in gr sweep
    nray_gr = gr_data.rays.data.shape[0]
    # gate length (meters)
    dr_gr = gr_data.range.data[1] - gr_data.range.data[0]
    # number of gates in gr beam
    ngate_gr = gr_data.rays.data.shape[1]
    # elevation of sweep (degree)
    elev_gr = gr_data.elevation.data[0]

    # Longitude of GR
    lon0_gr = gr_data.longitude.data
    # Latitude of GR
    lat0_gr = gr_data.latitude.data
    # Altitude of GR (meters)
    alt0_gr = gr_data.altitude.data

    # spherical to proj
    coord = wrl.georef.sweep_centroids(nray_gr, dr_gr, ngate_gr, elev_gr)
    coords = wrl.georef.spherical_to_proj(coord[..., 0],
                                          np.degrees(coord[..., 1]),
                                          coord[..., 2],
                                          (lon0_gr, lat0_gr, alt0_gr))
    lon = coords[..., 0]
    lat = coords[..., 1]

    bbox = wrl.zonalstats.get_bbox(lon, lat)

    return bbox


def volume_matching(sr_data, gr_data, bw_sr, dr_sr, zt, platf, gr_radar_band='X'):
    """
    # Input
    # ------

    sr_data ::: SR Datafile
    gr_data ::: GR Datafile
    bw_sr   ::: SR beam width
    dr_sr   ::: SR gate length (meters)
    zt      ::: SR orbit height (meters)
    platf   ::: SR platform/product: one out of ["gpm", "trmm"]
    gr_radar_band ::: GR band: "S", "C" or "X"

    # Output
    # -------

    Dict with :
      SR(KU)  ::: SR Ku-Reflectivity in dBz
      SR(X)   ::: SR Ku-Reflectivity (converted to X) in dBz
      SR(KA)  ::: SR Ka-Reflectivity in dBz
      DFR     ::: SR Dual-frequency ratio in dB
      GR(X)   ::: GR X-Reflectivity in dBz
      X       ::: x location in m
      Y       ::: y location in m
      Z       ::: z location in m
      RHO     ::: cross-correlatine coeff.
      KDP     ::: spez. differential phase in °/km
      ZDR     ::: differential reflectivity in dB

    More Information:
    https://docs.wradlib.org/en/stable/notebooks/match3d/wradlib_match_workflow.html

    """
    ######### GR data and att
    # number of rays in gr sweep
    nray_gr = gr_data.rays.data.shape[0]
    # number of gates in gr beam
    ngate_gr = gr_data.rays.data.shape[1]
    # elevation of sweep (degree)
    elev_gr = gr_data.elevation.data[0]
    # gate length (meters)
    dr_gr = gr_data.range.data[1] - gr_data.range.data[0]
    # sweep datetime stamp
    date_gr = gr_data.time
    # range of first gate
    r0_gr = gr_data.range.data[0]
    # azimuth angle of first beam
    a0_gr = gr_data.azimuth.data[0]
    # Longitude of GR
    lon0_gr = gr_data.longitude.data
    # Latitude of GR
    lat0_gr = gr_data.latitude.data
    # Altitude of GR (meters)
    alt0_gr = gr_data.altitude.data
    # Beam width of GR (degree)
    bw_gr = 1.
    # reflectivity array of sweep
    ref_gr = gr_data.DBZH.data
    # rhohv array of sweep
    rho_gr = gr_data.URHOHV.data
    # kdp array of sweep
    kdp_gr = gr_data.KDP1_NEW.data
    # zdr array of sweep
    zdr_gr = gr_data.ZDR.data
    # zdr array of sweep
    pbb_gr = gr_data.QBBF.data

    # Threshold
    TH = 0  #######SM
    ref_gr[ref_gr <= TH] = np.nan
    zdr_gr[ref_gr <= TH] = np.nan
    rho_gr[ref_gr <= TH] = np.nan
    kdp_gr[ref_gr <= TH] = np.nan

    # Extract relevant SR data and meta-data
    # Longitudes of SR scans
    sr_lon = sr_data['lon']
    # Latitudes of SR scans
    sr_lat = sr_data['lat']
    # Precip flag
    pflag = sr_data['pflag']
    # Number of scans on SR data
    nscan_sr = sr_data['nscan']
    # Number of rays in one SR scan
    nray_sr = sr_data['nray']
    # Number of gates in one SR ray
    ngate_sr = sr_data['nbin']

    # Georeferencing
    # Set fundamental georeferencing parameters
    # Calculate equivalent earth radius
    wgs84 = wrl.georef.get_default_projection()
    re1 = wrl.georef.get_earth_radius(lat0_gr, wgs84)

    a = wgs84.GetSemiMajor()
    b = wgs84.GetSemiMinor()

    # Set up aeqd-projection gr-centered
    rad = wrl.georef.proj4_to_osr(('+proj=aeqd +lon_0={lon:f} ' +
                                   '+lat_0={lat:f} +a={a:f} ' +
                                   '+b={b:f}').format(lon=lon0_gr,
                                                      lat=lat0_gr,
                                                      a=a, b=b))
    re2 = wrl.georef.get_earth_radius(lat0_gr, rad)

    ## Georeference GR data
    # create gr range and azimuth arrays
    rmax_gr = r0_gr + ngate_gr * dr_gr
    r_gr = np.arange(0, ngate_gr) * dr_gr + dr_gr / 2.
    az_gr = np.arange(0, nray_gr) - a0_gr

    # create gr polar grid and calculate aeqd-xyz coordinates
    gr_polargrid = np.meshgrid(r_gr, az_gr)
    gr_xyz, rad = wrl.georef.spherical_to_xyz(gr_polargrid[0], gr_polargrid[1], elev_gr,
                                              (lon0_gr, lat0_gr, alt0_gr),
                                              squeeze=True)

    # create gr poygon array in aeqd-xyz-coordinates
    gr_poly, rad1 = wrl.georef.spherical_to_polyvert(r_gr, az_gr, elev_gr,
                                                     (lon0_gr, lat0_gr, alt0_gr))
    gr_poly.shape = (nray_gr, ngate_gr, 5, 3)

    # get radar domain (outer ring)
    gr_domain = gr_xyz[:, -1, 0:2]
    gr_domain = np.vstack((gr_domain, gr_domain[0]))

    ## Georeference SR data
    sr_x, sr_y = wrl.georef.reproject(sr_lon, sr_lat,
                                      projection_source=wgs84,
                                      projection_target=rad)
    sr_xy = np.dstack((sr_x, sr_y))

    ## Subset relevant SR data¶
    # Create ZonalData for spatial subsetting (inside GR range domain)
    # get precip indexes
    # precip_mask = (pflag == 2) & wrl.zonalstats.get_clip_mask(sr_xy, gr_domain, rad)
    precip_mask = (pflag >= 2) & wrl.zonalstats.get_clip_mask(sr_xy, gr_domain, rad)

    ## SR Parallax Correction
    # use localZenith Angle
    alpha = sr_data['zenith']
    beta = abs(-17.04 + np.arange(nray_sr) * bw_sr)

    # Correct for parallax, get 3D-XYZ-Array
    #   xyzp_sr: Parallax corrected xyz coordinates
    #   r_sr_inv: range array from ground to SR platform
    #   zp: SR bin altitudes
    xyp_sr, r_sr_inv, z_sr = wrl.georef.correct_parallax(sr_xy, ngate_sr, dr_sr, alpha)
    xyzp_sr = np.concatenate((xyp_sr, z_sr[..., np.newaxis]),
                             axis=-1)

    ## Compute spherical coordinates of SR bins with regard to GR
    r_sr, az_sr, elev_sr = wrl.georef.xyz_to_spherical(xyzp_sr, alt0_gr, proj=rad)
    # TODO: hardcoded 1.0=elev_gr!!!!!!!!!!!!
    mask = (elev_sr > (elev_gr - bw_gr / 2.)) & (elev_sr < (elev_gr + bw_gr / 2.))

    ## Compute SR and GR pulse volumes
    # Calculate distance from orbit rs
    rs = wrl.georef.dist_from_orbit(zt, alpha, beta, r_sr_inv, re1)

    ## SR pulse volume
    # Small anngle approximation
    vol_sr2 = np.pi * dr_sr * rs ** 2 * np.radians(bw_sr / 2.) ** 2

    # Or using wradlib's native function
    vol_sr = wrl.qual.pulse_volume(rs, dr_sr, bw_sr)

    ## GR pulse volume
    # GR pulse volumes
    #   along one beam
    vol_gr = wrl.qual.pulse_volume(r_gr, dr_gr, bw_gr)
    #   with shape (nray_gr, ngate_gr)
    vol_gr = np.repeat(vol_gr, nray_gr).reshape((nray_gr, ngate_gr), order="F")

    ##Calculate horizontal and vertical dimensions Rs and Ds of SR bins
    Rs = 0.5 * (1 + np.cos(np.radians(alpha)))[:, :, np.newaxis] * rs * np.tan(
        np.radians(bw_sr / 2.))
    Ds = dr_sr / np.cos(np.radians(alpha))
    Ds = np.broadcast_to(Ds[..., np.newaxis], Rs.shape)

    ## Median Brightband Width/Height
    ratio, ibb = wrl.qual.get_bb_ratio(sr_data['zbb'], sr_data['bbwidth'],
                                       sr_data['quality'], z_sr)
    zbb = sr_data['zbb'].copy()
    zbb[~ibb] = np.nan

    ## Convert SR Ku reflectivities to S-band
    # Based on Cao et.al (2013)
    ref_sr = sr_data['refl'].filled(np.nan)
    ref_sr_ss = np.zeros_like(ref_sr) * np.nan
    ref_sr_sh = np.zeros_like(ref_sr) * np.nan

    a_s, a_h = (wrl.trafo.KuBandToS.snow, wrl.trafo.KuBandToS.hail)

    ia = (ratio >= 1)
    ref_sr_ss[ia] = ref_sr[ia] + wrl.util.calculate_polynomial(ref_sr[ia], a_s[:, 10])
    ref_sr_sh[ia] = ref_sr[ia] + wrl.util.calculate_polynomial(ref_sr[ia], a_h[:, 10])
    ib = (ratio <= 0)
    ref_sr_ss[ib] = ref_sr[ib] + wrl.util.calculate_polynomial(ref_sr[ib], a_s[:, 0])
    ref_sr_sh[ib] = ref_sr[ib] + wrl.util.calculate_polynomial(ref_sr[ib], a_h[:, 0])
    im = (ratio > 0) & (ratio < 1)
    ind = np.round(ratio[im] * 10).astype(np.int)
    ref_sr_ss[im] = ref_sr[im] + wrl.util.calculate_polynomial(ref_sr[im], a_s[:, ind])
    ref_sr_sh[im] = ref_sr[im] + wrl.util.calculate_polynomial(ref_sr[im], a_h[:, ind])

    # Jackson Tan's fix for C-band
    if gr_radar_band == 'C':
        print('SR reflectivity is converted to C-band')
        deltas = (ref_sr_ss - ref_sr) * 5.3 / 10.0
        ref_sr_ss = ref_sr + deltas
        deltah = (ref_sr_sh - ref_sr) * 5.3 / 10.0
        ref_sr_sh = ref_sr + deltah

    if gr_radar_band == 'X':
        print('SR reflectivity is converted to X-band')
        deltas = (ref_sr_ss - ref_sr) * 3.2 / 10.0
        ref_sr_ss = ref_sr + deltas
        deltah = (ref_sr_sh - ref_sr) * 3.2 / 10.0
        ref_sr_sh = ref_sr + deltah

    ref_sr_ss[ref_sr < 0] = np.nan

    ## Matching SR/GR
    # Identify which SR rays actually intersect with the GR sweep

    # Based on the above criteria (in radar range, precipitating SR profile)
    # and based on SR elevation angle (with regard to GR).

    # First assumption: no valid SR bins (all False)
    valid = np.asarray(elev_sr, dtype=np.bool) == False
    # SR is inside GR range and is precipitating
    iscan = precip_mask.nonzero()[0]
    iray = precip_mask.nonzero()[1]
    valid[iscan, iray] = True
    # SR bins intersect with GR sweep
    valid = valid & (elev_sr >= (elev_gr - bw_gr / 2.)) & (
                elev_sr <= (elev_gr + bw_gr / 2.))
    # Number of matching SR bins per profile
    nvalids = np.sum(valid, axis=2)
    # scan and ray indices for profiles with at least one valid bin
    vscan, vray = np.where(nvalids > 0)
    # number of profiles with at least one valid bin
    nprof = len(vscan)

    ## Averaging SR parameters
    # average coordinates
    xyz_v1 = xyzp_sr.copy()
    xyz_v1[~valid] = np.nan
    xyz_c1 = xyzp_sr.filled(0)
    xyz_c1[~valid] = 0
    c = np.count_nonzero(xyz_c1[..., 0], axis=2)
    ntotsr = c[vscan, vray]
    xyz_m1 = np.nanmean(xyz_v1, axis=2)
    xyz = xyz_m1[vscan, vray]

    # approximate Rs
    rs_v1 = Rs.copy()
    rs_v1[~valid] = np.nan
    rs_m1 = np.nanmax(rs_v1, axis=2)
    rs_prof = rs_m1[vscan, vray]
    ds = rs_prof

    # approximate Ds
    ds_v1 = Ds.copy()
    ds_v1[~valid] = np.nan
    ds_m1 = np.nansum(ds_v1, axis=2)
    ds_prof = ds_m1[vscan, vray]
    dz = ds_prof

    # approximate Vs
    vs_v1 = vol_sr.copy()
    vs_v1[~valid] = np.nan
    vs_m1 = np.nansum(vs_v1, axis=2)
    vs_prof = vs_m1[vscan, vray]
    volsr1 = vs_prof

    ## Calculate spherical coordinates of SR sample volume with respect to GR
    r_sr, az_sr, el_rs = wrl.georef.xyz_to_spherical(xyz, alt0_gr, proj=rad)

    ## Average SR reflectivity
    # unconverted
    ref_sr_1 = wrl.trafo.idecibel(ref_sr)
    ref_sr_1[~valid] = np.nan
    refsr1a = np.nanmean(ref_sr_1, axis=2)[vscan, vray]
    refsr1a = wrl.trafo.decibel(refsr1a)

    # converted for stratiform
    ref_sr_2 = wrl.trafo.idecibel(ref_sr_ss)
    ref_sr_2[~valid] = np.nan
    refsr2a = np.nanmean(ref_sr_2, axis=2)[vscan, vray]
    refsr2a = wrl.trafo.decibel(refsr2a)

    # converted for convective
    ref_sr_3 = wrl.trafo.idecibel(ref_sr_sh)
    ref_sr_3[~valid] = np.nan
    refsr3a = np.nanmean(ref_sr_3, axis=2)[vscan, vray]
    refsr3a = wrl.trafo.decibel(refsr3a)

    # Combined conversion
    ref_sr_com = np.ones_like(ref_sr) * np.nan
    ref_sr_com[sr_data['ptype'] == 1] = ref_sr_ss[sr_data['ptype'] == 1]
    ref_sr_com[sr_data['ptype'] == 2] = ref_sr_sh[sr_data['ptype'] == 2]
    ref_sr_4 = wrl.trafo.idecibel(ref_sr_com)
    ref_sr_4[~valid] = np.nan
    refsr4a = np.nanmean(ref_sr_4, axis=2)[vscan, vray]
    refsr4a = wrl.trafo.decibel(refsr4a)

    # Ka
    ref_ka = sr_data['refl_ka'].filled(np.nan)
    ref_sr_5 = wrl.trafo.idecibel(ref_ka)
    ref_sr_5[~valid] = np.nan
    refsr5a = np.nanmean(ref_sr_5, axis=2)[vscan, vray]
    refsr5a = wrl.trafo.decibel(refsr5a)

    # # DFR
    ref_sr_6 = sr_data['dfr'].filled(np.nan)
    ref_sr_6[~valid] = np.nan
    refsr6a = np.nanmean(ref_sr_6, axis=2)[vscan, vray]

    ## Identify which GR bins actually intersect with the SR footprint
    ## Create Zonal Data Source of matching SR profiles

    zds = wrl.zonalstats.DataSource(xyz[:, 0:2].reshape(-1, 2), rad)

    ## Create SR beam polygons using Rs
    tmp_trg_lyr = zds.ds.GetLayer()
    trg_poly = []
    for i, feat in enumerate(tmp_trg_lyr):
        geom = feat.GetGeometryRef().Buffer(rs_prof[i])
        poly = wrl.georef.ogr_to_numpy(geom)
        trg_poly.append(poly)

    ## Create ZonalDataPoly for GR and SR profiles
    print('Create ZonalDataPoly for GR and SR profiles.')
    print('This process can take a few seconds... ')
    # zdp = wrl.zonalstats.ZonalDataPoly(gr_poly[..., 0:2].reshape(-1, 5, 2), trg_poly, srs=rad)
    # zdp.dump_vector('m3d_zonal_poly_{0}'.format(platf))
    # obj3 = wrl.zonalstats.ZonalStatsPoly('m3d_zonal_poly_{0}'.format(platf))

    zdp = wrl.zonalstats.ZonalDataPoly(gr_poly[..., 0:2].reshape(-1, 5, 2), trg_poly,
                                       srs=rad)

    obj3 = wrl.zonalstats.ZonalStatsPoly(zdp)

    ## Average GR volume and reflectivity
    volgr1 = np.array([np.sum(vol_gr.ravel()[obj3.ix[i]])
                       for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])

    ref_gr_i = wrl.trafo.idecibel(ref_gr.ravel())
    refgr1a = np.array([np.nanmean(ref_gr_i[obj3.ix[i]])
                        for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])
    refgr1a = wrl.trafo.decibel(refgr1a)

    rho_gr_i = rho_gr.ravel()
    rhogr1a = np.array([np.nanmean(rho_gr_i[obj3.ix[i]])
                        for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])

    zdr_gr_i = zdr_gr.ravel()
    zdrgr1a = np.array([np.nanmean(zdr_gr_i[obj3.ix[i]])
                        for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])

    kdp_gr_i = kdp_gr.ravel()
    kdpgr1a = np.array([np.nanmean(kdp_gr_i[obj3.ix[i]])
                        for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])

    pbb_gr_i = pbb_gr.ravel()
    pbbgr1a = np.array([np.nanmin(pbb_gr_i[obj3.ix[i]])
                        for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])

    match_dict = {'SR(KU)': refsr1a,
                  'SR(X)': refsr4a,
                  'SR(KA)': refsr5a,
                  'DFR': refsr6a,
                  'GR(X)': refgr1a,
                  'X': xyz[..., 0],
                  'Y': xyz[..., 1],
                  'Z': xyz[..., 2],
                  'RHO': rhogr1a,
                  'KDP': kdpgr1a,
                  'ZDR': zdrgr1a,
                  'QI(PBB)': pbbgr1a}

    return match_dict


def plot_unmatched_refl(sr_data, gr_data):
    """
    Plotting unmatched SR and GR Reflectivity

    # Input:
    # ------
    gr_data ::: graund radar sweep data
    sr_data ::: satellite data

    # Output:
    # ------
    Plot of SR Refl. and GR Refl.

    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(sr_data['lon'], sr_data['lat'], c=sr_data['refl'][:, :, 5], s=10,
                cmap='jet', vmin=0, vmax=40)
    cbar = plt.colorbar(ticks=np.arange(0, 42, 2))
    cbar.set_label('SR Reflectivity (dBz)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('Longitude (°)', fontsize=12)
    plt.ylabel('Latitude (°)', fontsize=12)
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.title('GPM', fontsize=12, loc='left')

    plt.grid(lw=0.25, color='grey')

    plt.subplot(1, 2, 2)
    plt.scatter(gr_data.x.data / 1e3, gr_data.y.data / 1e3, c=gr_data.DBZH.data, s=10,
                cmap='jet', vmin=0, vmax=40)
    cbar = plt.colorbar(ticks=np.arange(0, 42, 2))
    cbar.set_label('GR Reflectivity (dBz)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('x (km)', fontsize=12)
    plt.ylabel('y (km)', fontsize=12)
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.title('BoXPol', fontsize=12, loc='left')
    plt.grid(lw=0.25, color='grey')
    plt.tight_layout()


def plot_matched_refl(res):
    """
    Plotting matched SR and GR Reflectivity

    # Input:
    # ------
    res ::: matching result

    # Output:
    # ------
    Plot of SR Refl.,  GR Refl., SR-GR Refl. and Scatterplot

    """
    xmin, xmax = np.nanmin(res['X'] / 1e3), np.nanmax(res['X'] / 1e3)
    ymin, ymax = np.nanmin(res['Y'] / 1e3), np.nanmax(res['Y'] / 1e3)

    plt.figure(figsize=(30, 5))
    plt.subplot(1, 4, 1)
    plt.scatter(res['X'] / 1e3, res['Y'] / 1e3, c=res['SR(X)'], cmap='jet', vmin=0,
                vmax=40)
    cbar = plt.colorbar(ticks=np.arange(0, 42, 2))
    cbar.set_label('SR Reflectivity (dBz)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('x (km)', fontsize=12)
    plt.ylabel('y (km)', fontsize=12)
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(lw=0.25, color='grey')

    plt.subplot(1, 4, 2)
    plt.scatter(res['X'] / 1e3, res['Y'] / 1e3, c=res['GR(X)'], cmap='jet', vmin=0,
                vmax=40)
    cbar = plt.colorbar(ticks=np.arange(0, 42, 2))
    cbar.set_label('GR Reflectivity (dBz)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('x (km)', fontsize=12)
    plt.ylabel('y (km)', fontsize=12)
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(lw=0.25, color='grey')

    plt.subplot(1, 4, 3)
    plt.scatter(res['X'] / 1e3, res['Y'] / 1e3, c=res['SR(X)'] - res['GR(X)'],
                cmap='seismic', vmin=-10, vmax=10)
    cbar = plt.colorbar(ticks=np.arange(-10, 12, 2))
    cbar.set_label('SR-GR Reflectivity (dB)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('x (km)', fontsize=12)
    plt.ylabel('y (km)', fontsize=12)
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(lw=0.25, color='grey')

    plt.subplot(1, 4, 4)
    plt.scatter(res['SR(X)'], res['GR(X)'], c='grey', s=10)
    plt.xlabel('SR Reflectivity (dBz)', fontsize=12)
    plt.ylabel('GR Reflectivity (dBz)', fontsize=12)
    plt.plot([0, 40], [0, 40], color='black')
    a = plt.xticks(fontsize=12)
    a = plt.yticks(fontsize=12)
    plt.grid(lw=0.25, color='grey')


def dpr_swath_contour(sr_grid):
    """

    Function for dpr swath grid contour

    ## sr_grid[:, 0]        # - complete left column
    ## sr_grid[-1,1:-1]     # - upper edge without left and right corner
    ## sr_grid[:, -1][::-1] # - complete right column (backward)
    ## sr_grid[0,0:-1]      # - lower edge (without right corner)

    Input:
    ------
    sr_grid ::: Space-borne grid x,y

    Output:
    -------
    trg ::: Contour of the space-borne grid

    """

    trg = np.r_[
        sr_grid[:, 0], sr_grid[-1, 1:-1], sr_grid[:, -1][::-1], sr_grid[0, 0:-1][::-1]]

    return trg


def idx_sr_in_gr_area(_bingrid, _gpm_xy):
    """
    Funktion:
    ---------
    Search for SR Footprint (Index) that are located in the GR scan area

    Input:
    ------
    _bingrid ::: Binary-grid of the GR scan area

    _gpm_xy ::: GPM footptint coordinates

    Output:
    -------

    _gpm_xy_outer_idx ::: index of all footprints not included in radolan scan area
    _gpm_xy_inner_idx ::: idex of all footprints included in radolan area but
                           not scanned by ground radars

    """
    _gpm_xy = _gpm_xy.copy()
    # Bestimmen von Eckpunkten bei RADOLAN RY GRID!
    # HARDCODED: but ok for regula RADOLAN Produkts
    xmin, xmax = -523.4621669218559, 375.5378330781441
    ymin, ymax = -4658.644724265572, -3759.644724265572

    # Radolangitter um 1 gridpoit zu allen seiten erweitern
    _xex, _yex = np.arange(xmin - 1, xmax + 2, 1), np.arange(ymin - 1, ymax + 2, 1)
    _xxx, _yyy = np.meshgrid(_xex, _yex)

    # Bin grid bestimmen
    _bingrid = _bingrid.copy()
    _bingrid[_bingrid >= 0] = 1
    _bingrid[_bingrid < 0] = 0

    # Gittererweiterungen zusammenstellen
    ## rn_tb ::: bin gird top and bottom
    ## rn_lr ::: bin gird left and right
    rn_tb = np.zeros(900)
    rn_lr = np.zeros(902)

    # Griderweiterungen einsetzen
    # erst oben und unten
    rn1 = np.c_[rn_tb, _bingrid, rn_tb]
    # dann rechts
    rn2 = np.vstack((rn1, rn_lr))
    # dann rechts
    rn3 = np.vstack((rn_lr, rn2))

    from skimage import measure
    # Contouren des Bin-grids erstellen
    contours = measure.find_contours(rn3, 0, positive_orientation='high',
                                     fully_connected='high')
    ### print(len(contours), ' detected Polygons')
    # print("---------------------")
    # for j in range(len(contours)):
    #    print(len(contours[j]))
    # print("---------------------")

    # bestiimung der Listen für Polygon Points
    _xx = []
    _yy = []

    # suche nach dem größten Polygon... dieser sollte der RADOLAN UMRISS sein
    for i in range(len(contours)):
        # Polygone mit 3 oder weniger points ergeben keine nutzbare Fläche
        if len(_xxx[contours[i][:, 0].astype(int), contours[i][:, 1].astype(int)]) <= 3:
            print('not relevant polygon removed')
        else:
            # PolygonPoints bestimmen (RADOLAN RAND POINTS)
            _xx.append(
                _xxx[contours[i][:, 0].astype(int), contours[i][:, 1].astype(int)])
            _yy.append(
                _yyy[contours[i][:, 0].astype(int), contours[i][:, 1].astype(int)])

    # sortiern nach der länge, das letzt element ist das größte und sollte somit der Radolarand sein
    _xx.sort(key=len)
    _yy.sort(key=len)

    # Outter polygon erstellen
    _xy = np.vstack((_xx[-1].ravel(), _yy[-1].ravel())).transpose()

    # Suche nach den GPM footprints im Scanngebiet von RADOLAN mit ZONALSTATS
    zdpoly = wrl.zonalstats.ZonalDataPoly(_gpm_xy, [_xy])
    _gpm_xy_outer_idx = zdpoly.get_source_index(0)

    # Array for inner Polygons
    _gpm_xy_inner_idx = np.array([])

    # Entferne Inner Polygons
    # Auch nur wenn es weitere Polygone gibt
    if len(_xx) > 1:

        for inner_poly_index in range(len(_xx) - 1):
            ### print ('Polygon size: ', len(_xx[inner_poly_index].ravel()))

            _xy_inner = np.vstack((_xx[inner_poly_index].ravel(),
                                   _yy[inner_poly_index].ravel())).transpose()

            _zdpoly = wrl.zonalstats.ZonalDataPoly(_gpm_xy, [_xy_inner])

            _inner_idx = _zdpoly.get_source_index(0)

            _gpm_xy_inner_idx = np.append(_gpm_xy_inner_idx, _inner_idx)

            ### if _gpm_xy_inner_idx.size==0:
            ### print ('inner polygons do not match with SR grid')

            ### else:
            ### print('match idx with inner polygon: ', _gpm_xy_inner_idx.shape)

    _idx_r = ~np.isin(_gpm_xy_outer_idx, _gpm_xy_inner_idx)

    return _gpm_xy_outer_idx[_idx_r]


def dpr_antenna_weighting(r):
    """
    Funktion: explained in
    Watters, D., A. Battaglia, K. Mroz, and F. Tridon, 0:
    Validation of the GPM Version-5 Surface Rainfall
    Products over Great Britain and Ireland.
    J. Hydrometeor., 0, https://doi.org/10.1175/JHM-D-18-0051.1

    and

    Mroz, K., A. Battaglia, T. J. Lang, D. J. Cecil,
    S. Tanelli, and F. Tridon, 2017:  Hail-detection
    algorithm for the GPM Core Observatory satellite sensors.
    Journal of Applied Meteorology and Climatology,56 (7), 1939–1957

    Input:
    ------
    r ::: Distance to the center point of the footprint

    Output:
    -------
    omega ::: Weighting for the point with the distance

    """

    omega = np.exp(-(r / 2.5) ** 2. * np.log10(4.))

    return omega


def ipoli_radi_w_extended(xy_cut, gpm_xy, rwdata_cut, dpr_footprint, k=25):
    """
    ...
    """
    # cKDTree radolan
    tree = cKDTree(xy_cut, balanced_tree=False)

    # cKDTree gpm dpr
    tree_gpm = cKDTree(gpm_xy, balanced_tree=True)

    dists, ix = tree.query(gpm_xy, k=k,
                           n_jobs=-1)  # k maximal possible ry pixel in dpr footprint
    ix2 = tree.query_ball_point(gpm_xy, dpr_footprint)

    ry_pns_w = []
    ry_pns_raw = []
    ry_weights = []
    ry_x = []
    ry_y = []

    for i in range(ix.shape[0]):
        # i is all points in one dpr footprint
        # distancen for all i
        index = np.isin(ix[i, :], ix2[i])
        res1 = np.nansum(dpr_antenna_weighting(dists[i, :][index]) * rwdata_cut.ravel()[
            ix[i, :][index]])
        res2 = np.nansum(dpr_antenna_weighting(dists[i, :][index]))

        ry_pns_w.append(res1 / res2)

        ry_pns_raw.append(rwdata_cut.ravel()[ix[i, :][index]])
        ry_weights.append(dpr_antenna_weighting(dists[i, :][index]))
        ry_x.append(xy_cut[:, 0].ravel()[ix[i, :][index]])
        ry_y.append(xy_cut[:, 1].ravel()[ix[i, :][index]])

    return np.array(ry_pns_w), np.array(ry_pns_raw), np.array(ry_weights), np.array(
        ry_x), np.array(ry_y)


def ipoli_radi(xy_cut, gpm_xy, rwdata_cut, dpr_footprint, k=25, calc='mean'):
    """
    calc: mean, max, min
    """
    # cKDTree radolan
    tree = cKDTree(xy_cut, balanced_tree=False)

    # cKDTree gpm dpr
    tree_gpm = cKDTree(gpm_xy, balanced_tree=True)

    dists, ix = tree.query(gpm_xy, k=k,
                           n_jobs=-1)  # k maximal possible ry pixel in dpr footprint
    ix2 = tree.query_ball_point(gpm_xy, dpr_footprint)

    ry_par = []

    if calc == 'mean':

        for i in range(ix.shape[0]):
            # i is all points in one dpr footprint
            # distancen for all i
            index = np.isin(ix[i, :], ix2[i])
            res1 = np.nanmean(rwdata_cut.ravel()[ix[i, :][index]])
            ry_par.append(res1)

    elif calc == 'max':

        for i in range(ix.shape[0]):
            # i is all points in one dpr footprint
            # distancen for all i
            index = np.isin(ix[i, :], ix2[i])
            res1 = np.nanmax(rwdata_cut.ravel()[ix[i, :][index]])
            ry_par.append(res1)

    elif calc == 'min':

        for i in range(ix.shape[0]):
            # i is all points in one dpr footprint
            # distancen for all i
            index = np.isin(ix[i, :], ix2[i])
            res1 = np.nanmin(rwdata_cut.ravel()[ix[i, :][index]])
            ry_par.append(res1)

    elif calc == 'std':

        for i in range(ix.shape[0]):
            # i is all points in one dpr footprint
            # distancen for all i
            index = np.isin(ix[i, :], ix2[i])
            res1 = np.nanstd(rwdata_cut.ravel()[ix[i, :][index]])
            ry_par.append(res1)

    else:
        print('Wrong calc parameter declaration!')

    return np.array(ry_par)


def gpm_scan_time(filename, scanswath='NS'):
    from netCDF4 import Dataset
    pr_data = Dataset(filename, mode="r")
    year = pr_data[scanswath]['ScanTime'].variables['Year']
    month = pr_data[scanswath]['ScanTime'].variables['Month']
    dayofmonth = pr_data[scanswath]['ScanTime'].variables['DayOfMonth']
    # dayofyear = pr_data['NS']['ScanTime'].variables['DayOfYear'][mask]
    hour = pr_data[scanswath]['ScanTime'].variables['Hour']
    minute = pr_data[scanswath]['ScanTime'].variables['Minute']
    second = pr_data[scanswath]['ScanTime'].variables['Second']
    # secondofday = pr_data['NS']['ScanTime'].variables['SecondOfDay'][mask]
    millisecond = pr_data[scanswath]['ScanTime'].variables['MilliSecond']
    date_array = zip(year, month, dayofmonth,
                     hour, minute, second,
                     millisecond)
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in date_array])

    # ttt = gpm_scan_time(gpm_file)
    s_n = {'NS': 49, 'HS': 24, 'MS': 25}

    time_array = np.array(s_n[scanswath] * [pr_time]).T

    pr_data.close()

    return time_array


def get_time_of_gpm2(gpm_time):
    """einfache mittlere zeit"""
    iii = int(len(np.array(gpm_time['Year'])) / 2)
    gpm_year = np.array(gpm_time['Year'])[iii]
    gpm_month = np.array(gpm_time['Month'])[iii]
    gpm_day = np.array(gpm_time['DayOfMonth'])[iii]
    gpm_hour = np.array(gpm_time['Hour'])[iii]
    gpm_min = np.array(gpm_time['Minute'])[iii]
    gpm_sek = np.array(gpm_time['Second'])[iii]
    gpm_dt = dt.datetime(gpm_year, gpm_month, gpm_day, gpm_hour, gpm_min,
                         gpm_sek).strftime("%Y.%m.%d -- %H:%M:%S")
    return gpm_dt


def calc_height(bin_range_number, elipsoid_offset, localZenithAngle, scan_swath='NS'):
    # Function for height calculation from bins.....
    # GPM DPR ATBD Level 2 2017 Awaka et al S. 21

    if scan_swath in ('NS', 'MS'):
        rangeBinSize = 125.
        binEllipsoid = 176.
    else:
        rangeBinSize = 250.
        binEllipsoid = 88.

    calc_1 = ((binEllipsoid - bin_range_number) * rangeBinSize + elipsoid_offset)

    calc_h = calc_1 * np.cos(np.deg2rad(localZenithAngle))

    return calc_h


def calc_bin(height, elipsoid_offset, localZenithAngle, scan_swath='NS'):
    # Function for height calculation from bins.....
    # GPM DPR ATBD Level 2 2017 Awaka et al S. 21

    if scan_swath in ('NS', 'MS'):
        rangeBinSize = 125.
        binEllipsoid = 176.
    else:
        rangeBinSize = 250.
        binEllipsoid = 88.

    res0 = (height / np.cos(np.deg2rad(localZenithAngle))) - elipsoid_offset
    res1 = res0 / rangeBinSize
    res_bin = binEllipsoid - res1

    return res_bin


def compare_grsr(gpm_file, sc='NS'):
    """
    Comparison of SR an GR data.
    SR Data = DPR V7-20170308 V05A
    GR Data = RADOLAN (RX=Reflecitivtiy in dBz,
                       RY=Rain rate in mm/h)

    Ref: Pejcic et al 2020
    https://doi.org/10.1127/METZ/2020/1039

    # Input:
    # -----
    gpm_file ::: Path to DPR file
    sc       ::: Scan NS, MS or HS

    # Output:
    # ------
    Matched SR GR Data in dictionary

    x      ::: RAW x coords in km
    y      ::: RAW y coords in km
    gr_rr  ::: RAW rain rate GR (RY)
    gr_ref ::: RAW reflectivity GR (RX)
    X      ::: Matched x coords in km
    Y      ::: Matched y coords in km
    GR_RR  ::: Rain rate in mm/h (GR)
    GR_REF ::: Reflectivity in dBz (GR)
    SR_RR  ::: Rain rate in mm/h (SR)
    SR_REF ::: Reflectivity in dBz (SR)
    SR_PIA ::: PIA in dB (SR)
    SR_Dm  ::: Dm in mm (SR)
    SR_Nw  ::: Nw in 10 log10(m^-3 mm^-1) (SR)
    SR_STH ::: Storm top height in m (SR)
    SR_BBH ::: Bright band height in m (SR)
    SR_BBW ::: Bright band width in m (SR)
    SR_CFB ::: Clutter free bottom in m (SR)
    SR_LZA ::: Local zenith angle in °
    SR_TYP ::: Precipitation Type (strati = 1.,conv = 2.,other=3.)
    SR_PHA ::: Precipitation phase (liquid=1, solid=0, mixed=[0,1])
    SR_LST ::: Land surface type (0-99 : Ocean, 100 - 199 : Land, 200 - 299 : Coast, 300 - 399 : Inland water)
    SR_ELE ::: Elevation in m
    SR_ANV ::: Anvil (0: no Anvil, 1: Anvil no rain, 2: Anvil with rain)
    SR_HIP ::: Heavy Ice Precipitation (see ATBD)
    GR_ALT ::: Mean GR Scan altitude in m

    """

    import warnings
    warnings.filterwarnings('ignore')

    print('GPM file path: ')
    print('--------------')
    print(gpm_file)

    # DPR Parameters
    dpr_para = {'NS': 176, 'HS': 88, 'MS': 176}

    # Import radolan grid src
    radolan_xy = wrl.georef.get_radolan_grid(900, 900)
    radolan_xy = radolan_xy.reshape(-1, radolan_xy.shape[-1])
    zd = wrl.zonalstats.DataSource(radolan_xy, name='src')

    # Determine projection
    proj_stereo = wrl.georef.create_osr("dwd-radolan")
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)

    # Import RADOLAN Heights
    rhmax = np.load("/automount/ftp/radar/wradlib-data/radolan/alt/RY_H_max.npy",
                    encoding='bytes', allow_pickle=True)
    rhmin = np.load("/automount/ftp/radar/wradlib-data/radolan/alt/RY_H_min.npy",
                    encoding='bytes', allow_pickle=True)
    rhmean = np.load("/automount/ftp/radar/wradlib-data/radolan/alt/RY_H_mean.npy",
                     encoding='bytes', allow_pickle=True)

    # radolan_height_std_cut = rhstd.ravel()[ry_idx2]
    radolan_beam_width = np.load('/automount/ftp/radar/wradlib-data/radolan/alt/RY_C_min.npy',
                                 encoding='bytes', allow_pickle=True)

    dem_slope = np.load('/automount/ftp/radar/wradlib-data/radolan/alt/slope_array.npy',
                        encoding='bytes', allow_pickle=True)

    print('GR Heights and Beam widths imported!')

    # gpm scantimes
    gpm_scan_times = gpm_scan_time(gpm_file, scanswath=sc)

    # read gpm file with importatnt parameters
    gpmdpr = h5py.File(gpm_file, 'r')

    # GPM navigation and time
    gpm_lat = np.array(gpmdpr[sc]['Latitude'])
    gpm_lon = np.array(gpmdpr[sc]['Longitude'])
    gpm_time = gpmdpr[sc]['ScanTime']

    try:
        # gpm_zeit = get_time_of_gpm(gpm_lon, gpm_lat, gpm_time)
        gpm_zeit = get_time_of_gpm2(gpm_time)

    except ValueError:
        pass
        print('____________ValueError____________')
    else:
        # gpm_zeit = get_time_of_gpm(gpm_lon, gpm_lat, gpm_time)
        ht, mt = gpm_zeit[14:16], str(int(round(float(gpm_zeit[17:19]) / 5.0) * 5.0))
        year, ye, m, d = gpm_zeit[0:4], gpm_zeit[2:4], gpm_zeit[5:7], gpm_zeit[8:10]

        if mt == '0':
            mt = '00'
        if mt == '5':
            mt = '05'
        if mt == '60':
            mt = '55'
            ht = str(int(ht) + 1)
            if ht == '24':
                d = str(int(d) + 1)

        print('Overpass time: ', gpm_zeit)

        r_pro = 'ry'
        # try:

        r_pfad = ('/automount/radar/dwd/' + r_pro + '/' + str(year) + '/' + str(
            year) + '-' +
                  str(m) + '/' + str(year) + '-' + str(m) + '-' + str(
                    d) + '/raa01-' + r_pro + '_10000-' +
                  str(ye) + str(m) + str(d) + str(ht) + str(mt) + '-dwd---bi*')
        print('Path to GR rain rate (RADOLAN RY): ')
        print('----------------------------------')
        print(r_pfad)

        z_pro = 'rx'

        z_pfad = ('/automount/radar/dwd/' + z_pro + '/' + str(year) + '/' + str(
            year) + '-' +
                  str(m) + '/' + str(year) + '-' + str(m) + '-' + str(
                    d) + '/raa01-' + z_pro + '_10000-' +
                  str(ye) + str(m) + str(d) + str(ht) + str(mt) + '-dwd---bi*')

        print('Path to GR reflectivity (RADOLAN RX):')
        print('-------------------------------------')
        print(z_pfad)

        ## TRY: gibt es eine RADOLAN Datei zum Overpass!?
        ## read radolan ry file data and attributes
        try:
            rwdata, rwattrs = wrl.io.read_radolan_composite(glob.glob(r_pfad)[0])

            radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
            x = radolan_grid_xy[:, :, 0]
            y = radolan_grid_xy[:, :, 1]
            rwdata = np.ma.masked_equal(rwdata,
                                        -9999) * 12  # Um auf mm/h zu kommen, beim Einlesen sind es mm/5min

            radolan_scan_time = rwattrs['datetime']

            zwdata, zwattrs = wrl.io.read_radolan_composite(glob.glob(z_pfad)[0])
            zwdata = np.ma.masked_equal(zwdata, -9999) / 2 - 32.5
        except:
            print('RY File Missing!')

        else:
            # copy data for binary grid
            bingrid = rwdata.copy()
            # bingridz = zwdata.copy()

            # Precipitation data in RADOLAN region
            # Posible precip products:
            # precipRateNearSurface
            dpr_pns = np.array(gpmdpr[sc]['SLV']['precipRateNearSurface'])
            dpr_pns[dpr_pns == -9999.9] = np.nan

            # precipRateAve24
            dpr_pav = np.array(gpmdpr[sc]['SLV']['precipRateAve24'])
            dpr_pav[dpr_pav == -9999.9] = np.nan
            # precip rate 3d
            dpr_p3d = np.array(gpmdpr[sc]['SLV']['precipRate'])
            dpr_p3d[dpr_p3d == -9999.9] = np.nan

            # Brightband height
            dpr_bbh = np.array(gpmdpr[sc]['CSF']['heightBB'], dtype=float)
            dpr_bbh[dpr_bbh == -9999.9] = np.nan

            # Bright-band width
            dpr_bbw = np.array(gpmdpr[sc]['CSF']['widthBB'], dtype=float)
            dpr_bbw[dpr_bbw == -9999.9] = np.nan

            # Bright-band Top/Bot
            dpr_bot = np.array(gpmdpr[sc]['CSF']['binBBBottom'], dtype=float)
            dpr_bot[dpr_bot == -9999] = np.nan
            dpr_bot[dpr_bot == -1111] = np.nan
            dpr_bot[dpr_bot == 0] = np.nan

            dpr_top = np.array(gpmdpr[sc]['CSF']['binBBTop'], dtype=float)
            dpr_top[dpr_top == -9999] = np.nan
            dpr_top[dpr_top == -1111] = np.nan
            dpr_top[dpr_top == 0] = np.nan

            # FreezingLevelHeight
            dpr_flh = np.array(gpmdpr[sc]['VER']['heightZeroDeg'], dtype=float)
            dpr_flh[dpr_flh == -9999.9] = np.nan

            # PhaseNearSurface
            dpr_pha = np.array(gpmdpr[sc]['SLV']['phaseNearSurface'], dtype=float)
            # dpr_pha = dpr_pha/100
            dpr_pha[dpr_pha == 255] = np.nan

            # typePrecip
            dpr_typ = np.array(gpmdpr[sc]['CSF']['typePrecip'], dtype=float)

            # StromTopHeight
            dpr_sth = np.array(gpmdpr[sc]['PRE']['heightStormTop'], dtype=float)
            dpr_sth[dpr_sth == -9999.9] = np.nan

            # Elevation
            dpr_ele = np.array(gpmdpr[sc]['PRE']['elevation'], dtype=float)
            dpr_ele[dpr_ele == -9999] = np.nan

            # Clutterfreebottom
            dpr_cfb_bin = np.array(gpmdpr[sc]['PRE']['binClutterFreeBottom'],
                                   dtype=float)
            dpr_cfb_bin[dpr_cfb_bin == -9999] = np.nan

            dpr_eof = np.array(gpmdpr[sc]['PRE']['ellipsoidBinOffset'], dtype=float)
            dpr_eof[dpr_eof == -9999] = np.nan

            dpr_lza = np.array(gpmdpr[sc]['PRE']['localZenithAngle'], dtype=float)
            dpr_lza[dpr_lza == -9999] = np.nan

            ## Berechnung der clutterfree bottom height
            dpr_cfb = calc_height(dpr_cfb_bin, dpr_eof, dpr_lza, scan_swath=sc)
            dpr_bbt = calc_height(dpr_top, dpr_eof, dpr_lza, scan_swath=sc)
            dpr_bbb = calc_height(dpr_bot, dpr_eof, dpr_lza, scan_swath=sc)

            # Landsurfacetyp
            dpr_lst = np.array(gpmdpr[sc]['PRE']['landSurfaceType'], dtype=float)
            dpr_lst[dpr_lst == -9999] = np.nan

            ## Flags
            # HeavyIcePrecipitatio
            dpr_hip = np.array(gpmdpr[sc]['CSF']['flagHeavyIcePrecip'], dtype=float)
            dpr_hip[dpr_hip == -99] = np.nan
            # flagShallowRain
            dpr_shr = np.array(gpmdpr[sc]['CSF']['flagShallowRain'], dtype=float)
            dpr_shr[dpr_shr == -9999] = np.nan

            # flagBB
            dpr_flb = np.array(gpmdpr[sc]['CSF']['flagBB'], dtype=float)

            # Anvil
            dpr_anv = np.array(gpmdpr[sc]['CSF']['flagAnvil'], dtype=float)
            dpr_anv[dpr_anv == -99] = np.nan

            # 3D Phase
            dpr_3dpha = np.array(gpmdpr[sc]['DSD']['phase'], dtype=float)
            dpr_3dpha[dpr_3dpha == 255] = np.nan
            dpr_3dpha = dpr_3dpha // 100

            # PiaFinal
            dpr_pia = np.array(gpmdpr[sc]['SLV']['piaFinal'], dtype=float)
            dpr_pia[dpr_pia == -9999] = np.nan

            # DSD
            dpr_ref = np.array(gpmdpr[sc]['SLV']['zFactorCorrectedNearSurface'],
                               dtype=float)
            dpr_ref[dpr_ref < -9999.9] = np.nan

            # DSD
            dpr_dsd = np.array(gpmdpr[sc]['SLV']['paramDSD'], dtype=float)
            dpr_dsd[dpr_dsd < -9999.9] = np.nan
            dpr_Nw = dpr_dsd[:, :, :, 0]
            dpr_Dm = dpr_dsd[:, :, :, 1]

            # GPM coordinate projection
            gpm_x, gpm_y = wrl.georef.reproject(gpm_lon, gpm_lat,
                                                    projection_target=proj_stereo,
                                                    projection_source=proj_wgs)

            print('SR GR Matching started!')
            print('-----------------------')
            ## Remove all GPM footprint that are not in scan area of radolan
            ## -------------------------------------------------------------

            # GPM Koordinaten ravel
            gpm_xy = np.vstack((gpm_x.ravel(), gpm_y.ravel())).transpose()

            # index for GPM footprints in scan area
            oi_idx = idx_sr_in_gr_area(bingrid, gpm_xy)

            # check if idx is empty
            if len(oi_idx) == 0:
                print("____No Index of Overpass in Radolan region____")
                pass


            else:
                # entferne alle Indizes auserhalb des äußeren Polygons
                gpm_xy = gpm_xy[oi_idx]
                gpm_pns = dpr_pns.ravel()[oi_idx]
                # gpm_pes = dpr_pes.ravel()[oi_idx]
                gpm_pav = dpr_pav.ravel()[oi_idx]
                gpm_bbh = dpr_bbh.ravel()[oi_idx]
                gpm_bbw = dpr_bbw.ravel()[oi_idx]
                gpm_pha = dpr_pha.ravel()[oi_idx]
                gpm_typ = dpr_typ.ravel()[oi_idx]
                gpm_sth = dpr_sth.ravel()[oi_idx]
                gpm_ele = dpr_ele.ravel()[oi_idx]
                gpm_cfb = dpr_cfb.ravel()[oi_idx]
                gpm_lst = dpr_lst.ravel()[oi_idx]
                gpm_tim = gpm_scan_times.ravel()[oi_idx]

                # Parameters only for new pns product
                dpr_p3d = np.reshape(dpr_p3d, [dpr_pns.shape[0] * dpr_pns.shape[1],
                                               dpr_para[sc]])
                gpm_p3d = dpr_p3d[oi_idx, :]

                # 3d phase for adjusting
                dpr_3dpha = np.reshape(dpr_3dpha,
                                       [dpr_3dpha.shape[0] * dpr_3dpha.shape[1],
                                        dpr_para[sc]])
                gpm_ph3d = dpr_3dpha[oi_idx, :]

                # Parameters only for new DSD 3D product
                dpr_Dm = np.reshape(dpr_Dm,
                                    [dpr_pns.shape[0] * dpr_pns.shape[1], dpr_para[sc]])
                gpm_Dm = dpr_Dm[oi_idx, :]

                dpr_Nw = np.reshape(dpr_Nw,
                                    [dpr_pns.shape[0] * dpr_pns.shape[1], dpr_para[sc]])
                gpm_Nw = dpr_Nw[oi_idx, :]

                gpm_cfb = dpr_cfb.ravel()[oi_idx]
                gpm_eof = dpr_eof.ravel()[oi_idx]
                gpm_cfb_bin = dpr_cfb_bin.ravel()[oi_idx]

                gpm_lza = dpr_lza.ravel()[oi_idx]
                gpm_bbb = dpr_bbb.ravel()[oi_idx]
                gpm_bbt = dpr_bbt.ravel()[oi_idx]
                gpm_flh = dpr_flh.ravel()[oi_idx]
                gpm_hip = dpr_hip.ravel()[oi_idx]
                gpm_shr = dpr_shr.ravel()[oi_idx]
                gpm_anv = dpr_anv.ravel()[oi_idx]
                gpm_flb = dpr_flb.ravel()[oi_idx]
                gpm_pia = dpr_pia.ravel()[oi_idx]
                gpm_ref = dpr_ref.ravel()[oi_idx]

                ## Remove all RADOLAN Points not included in GPM DPR swath
                ## -------------------------------------------------------
                # Contour from ORIGINAL DPR Overpass
                dpr_contour_x = dpr_swath_contour(gpm_x)
                dpr_contour_y = dpr_swath_contour(gpm_y)

                # create dpr polygon of dpr xy contours
                dpr_xy_poly = np.vstack(
                    (dpr_contour_x.ravel(), dpr_contour_y.ravel())).transpose()

                # load regular radolan grid and overlie with dpr contours
                zd_poly_radolan = wrl.zonalstats.ZonalDataPoint(zd, [dpr_xy_poly],
                                                                buf=2.5)

                # get radolan index in dpr swath
                ry_idx = zd_poly_radolan.get_source_index(0)

                # get radolan index in dpr swath AND in binary grid
                rwtest = ~np.ma.masked_less(rwdata, 0).mask
                zwtest = ~np.ma.masked_less(zwdata, 0).mask

                radolan_idx = np.flatnonzero(rwtest.ravel())

                ry_idx2 = np.intersect1d(ry_idx, radolan_idx, assume_unique=True)

                # Radolan xy array
                xy_radolan = np.vstack((x.ravel(), y.ravel())).transpose()

                # Extract all affected RADOLAN grid points
                rwdata_cut = rwdata.ravel()[ry_idx2]
                zwdata_cut = zwdata.ravel()[ry_idx2]
                x_cut = xy_radolan[..., 0].ravel()[ry_idx2].copy()
                y_cut = xy_radolan[..., 1].ravel()[ry_idx2].copy()

                # extrect all Height points (need to have the same shape as radolan grid)
                radolan_height_min_cut = rhmin.ravel()[ry_idx2]
                radolan_height_max_cut = rhmax.ravel()[ry_idx2]
                # radolan_height_std_cut = rhstd.ravel()[ry_idx2]
                radolan_height_mean_cut = rhmean.ravel()[ry_idx2]

                radolan_beam_width_cut = radolan_beam_width.ravel()[ry_idx2]

                # Compare all cutted radolan x and y
                xy_cut = np.vstack((x_cut.ravel(), y_cut.ravel())).transpose()

                # Dpr footprint radius
                dpr_footprint = 2.5  # radius in km

                ## -------------------------------------------
                # Interpolation with antena weighted function
                ## -------------------------------------------
                # ry_pns_w = ipoli_radi_w(xy_cut, gpm_xy, rwdata_cut, dpr_footprint, k=25)
                # zwdata_cut[zwdata_cut<15]=np.nan
                ry_pns_w, ry_pns_raw, ry_pns_aweight, ry_x, ry_y = ipoli_radi_w_extended(
                    xy_cut, gpm_xy, rwdata_cut, dpr_footprint, k=25)
                rx_pns_w, rx_pns_raw, rx_pns_aweight, rx_x, rx_y = ipoli_radi_w_extended(
                    xy_cut, gpm_xy, zwdata_cut, dpr_footprint, k=25)

                # Interpolation of other parameters on dpr grid
                radolan_height_min_cut_ipoli = ipoli_radi(xy_cut, gpm_xy,
                                                          radolan_height_min_cut,
                                                          dpr_footprint, k=25,
                                                          calc='mean')
                radolan_height_max_cut_ipoli = ipoli_radi(xy_cut, gpm_xy,
                                                          radolan_height_max_cut,
                                                          dpr_footprint, k=25,
                                                          calc='mean')
                radolan_height_mean_cut_ipoli = ipoli_radi(xy_cut, gpm_xy,
                                                           radolan_height_mean_cut,
                                                           dpr_footprint, k=25,
                                                           calc='mean')
                radolan_height_std_cut_ipoli = ipoli_radi(xy_cut, gpm_xy,
                                                          radolan_height_min_cut,
                                                          dpr_footprint, k=25,
                                                          calc='std')
                radolan_beam_width_cut_ipoli = ipoli_radi(xy_cut, gpm_xy,
                                                          radolan_beam_width_cut,
                                                          dpr_footprint, k=25,
                                                          calc='mean')

                ## Create adjusted pns and phase product from p3d
                ## ----------------------------------------------

                # delta h calculation
                # delta_h = radolan_height_min_cut_ipoli - gpm_cfb
                # Height over NN
                delta_h = radolan_height_min_cut_ipoli - (gpm_cfb - gpm_ele)

                # new bin height
                new_bin = calc_bin(gpm_cfb + delta_h, gpm_eof, gpm_lza)
                # round for bin
                new_bin = np.ceil(new_bin)
                # error calc with nan removal
                new_bin[new_bin < 0] = np.nan

                # Beachtung des local zenith angle
                # gpm_bin_in_beam = beamwidth / sin(lza)
                # rbwi = radolan_beam_width_cut_ipoli / np.cos(np.deg2rad(gpm_lza))
                rbwi = radolan_beam_width_cut_ipoli
                # column calculation ! Half column!
                rcbin = np.round((rbwi / 2.) / 125., 0)

                # adjusted precip product
                gpm_app = []
                gpm_aDm = []
                gpm_aNw = []

                # adjusted phase
                # gpm_aph = []

                # number of bins with same phase (ice,liquid, mixed) in one column
                gpm_lbn = []
                gpm_ibn = []
                gpm_mbn = []

                for j in range(len(new_bin)):

                    if np.isnan(new_bin[j]):
                        # Check if bin is nan
                        gpm_app.append(np.nan)
                        gpm_lbn.append(np.nan)
                        gpm_mbn.append(np.nan)
                        gpm_ibn.append(np.nan)


                    else:
                        if (new_bin[j] - 1) >= dpr_para[sc]:
                            # Check if bin index bigger than array
                            # happen by round bin
                            gpm_app.append(np.nan)
                            gpm_lbn.append(np.nan)
                            gpm_mbn.append(np.nan)
                            gpm_ibn.append(np.nan)
                        else:
                            nbin = int(new_bin[j]) - 1
                            cbin = rcbin[j]

                            # Averaged Rainrate in gpm cpolumn (later Antenna)
                            start_bin = nbin - int(cbin)
                            end_bin = nbin + int(cbin) + 1
                            if start_bin < 0:
                                start_bin = 0
                            if end_bin >= dpr_para[sc]:
                                end_bin = dpr_para[sc] - 1

                            # Mean Precip rate in colum... later antena function C-band
                            rr = gpm_p3d[j, start_bin: end_bin]
                            rr = np.nanmean(rr)

                            Dm = gpm_Dm[j, start_bin: end_bin]
                            Dm = np.nanmean(Dm)

                            Nw = gpm_Nw[j, start_bin: end_bin]
                            Nw = np.nanmean(Nw)

                            phase = gpm_ph3d[j, start_bin: end_bin]

                            # Phase bin ratios... later also weighting!
                            idx_l = phase == 2
                            idx_i = phase == 0
                            idx_m = phase == 1

                            gpm_app.append(rr)
                            gpm_aDm.append(Dm)
                            gpm_aNw.append(Nw)
                            # gpm_aph.append(gpm_ph3d[j,nbin])
                            gpm_lbn.append(sum(idx_l))
                            gpm_ibn.append(sum(idx_i))
                            gpm_mbn.append(sum(idx_m))

                gpm_app = np.array(gpm_app)
                gpm_lbn = np.array(gpm_lbn)
                gpm_ibn = np.array(gpm_ibn)
                gpm_mbn = np.array(gpm_mbn)
                gpm_aDm = np.array(gpm_aDm)
                gpm_aNw = np.array(gpm_aNw)

                # remove invalide values
    gpm_sth[gpm_sth < 0] = np.nan
    gpm_bbw[gpm_bbw < 0] = np.nan
    gpm_bbh[gpm_bbh < 0] = np.nan

    # pha = gpm_pha.copy().astype(float)
    # pha[pha==255]=np.nan
    # pha = pha//100

    # Dominant Precip type
    typ = gpm_typ.copy()
    typ = typ // 10000000

    # Phase score
    pha = gpm_lbn / (gpm_lbn + gpm_ibn + gpm_mbn)

    res = {'x': x,
           'y': y,
           'gr_rr': rwdata,
           'gr_ref': zwdata,
           'X': gpm_xy[:, 0],
           'Y': gpm_xy[:, 1],
           'GR_RR': ry_pns_w,
           'GR_REF': rx_pns_w,
           'SR_RR': gpm_app,
           'SR_REF': gpm_ref,
           'SR_PIA': gpm_pia,
           'SR_Dm': gpm_aDm,
           'SR_Nw': gpm_aNw,
           'SR_STH': gpm_sth,
           'SR_BBH': gpm_bbh,
           'SR_BBW': gpm_bbw,
           'SR_CFB': gpm_cfb,
           'SR_LZA': gpm_lza,
           'SR_TYP': typ,
           'SR_PHA': pha,
           'SR_LST': gpm_lst,
           'SR_ELE': gpm_ele,
           'SR_ANV': gpm_anv,
           'SR_HIP': gpm_hip,
           'GR_ALT': radolan_height_mean_cut_ipoli}

    gpmdpr.close()

    return res


def hist_2d(A, B, binsx=35, binsy=35, mini=1, maxi=None, cmap='jet', colsteps=30,
            alpha=1, fsize=15, colbar=True):
    """
    # Histogram 2d Quicklooks
    # ------------------------

    Plotting 2d Histogramm of two varibles

    # Input
    # -----

    A,B          ::: Variables
    bins1, bins2 ::: x, y bins
    mini, maxi   ::: min and max
    cmap         ::: colormap
    colsteps     ::: number of cmap steps
    alpha        ::: transperency
    fsize        ::: fontsize

    # Output
    # ------

    2D Histogramm Plot


    ::: Hist mode:::
    absolute ::: absolute numbers
    relative ::: relative numbers
    relative_with_y ::: relative numbers of y levels

    """
    from matplotlib.colors import LogNorm

    # discret cmap
    cmap = plt.cm.get_cmap(cmap, colsteps)

    # mask array
    m = ~np.isnan(A) & ~np.isnan(B)

    plt.hist2d(A[m], B[m], bins=(binsx, binsy), cmap=cmap,
               norm=LogNorm(vmin=mini, vmax=maxi), alpha=alpha)
    if colbar == True:
        cb = plt.colorbar(shrink=1, pad=0.01)
        cb.set_label('number of samples', fontsize=fsize)
        cb.ax.tick_params(labelsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)




def attenuation_corr_zphi(swp0 , alpha = 0.08, beta = 0.02, medwin = 7, kwidth = 7, sigma = 10):
    '''
    ZPHI method for attenuation correction. Needs functions from radarmet.py

    Parameters:
        * swp0(xarray.Dataset): single sweep (PPI for a single timestep) of radar data,
                                must contain ZH (or DBZH), PHIDP (or UPHIDP),
                                RHOHV, optionally ZDR
        * alpha(float): alpha value for X or C band, default value is for C band
        * beta(float): beta value for X or C band, default value is for C band
        * medwin(int): value for median filtering 2d
        * kwidth(int): value for gaussian convolution 1d - smoothing
        * sigma(int): value for gaussian convolution 1d - smoothing

    Returns:
        * ah_fast(numpy.array): Attenuation (horizontal)
        * phical(numpy.array): PHI CAL from attenuation
        * phicorr(xarray.DataArray): PHIDP masked, filtered, smoothed and offset corrected
        * cphase(xarray.Dataset): Corrected phase phi
        * alpha(float)
        * beta(float)

    ZH_corr = ZH + alpha*PHIDP
    ZDR_corr = ZDR + beta*PHIDP

    X band:
    alpha = 0.28; beta = 0.05 #dB/deg

    C band:
    alpha = 0.08; beta = 0.02 #dB/deg

    For BoXPol and JuXPol:
    alpha = 0.25
    '''

    #### ZPHI method
    # copy moments

    if 'PHIDP' in swp0.data_vars:
        phidp = 'PHIDP'
    #elif 'PHIDP_OC'in swp0.data_vars: # if PHIDP is already corrected there are steps that should be avoided
    #    phidp = 'PHIDP_OC'
    #elif 'PHIDP_OC_MASKED'in swp0.data_vars:
    #    phidp = 'PHIDP_OC_MASKED'
    #elif 'PHIDP_OC_SMOOTH'in swp0.data_vars:
    #    phidp = 'PHIDP_OC_SMOOTH'
    elif 'UPHIDP'in swp0.data_vars:
        phidp = 'UPHIDP'
    else:
        raise AttributeError('PHIDP could not be found in Dataset')

    if 'DBZH' in swp0.data_vars:
        dbzh = 'DBZH'
    elif 'ZH'in swp0.data_vars:
        dbzh = 'ZH'
    else:
        raise AttributeError('ZH could not be found in Dataset')

    swp_msk = swp0.copy()

    dr_m = swp0.range.diff('range').median()
    dr_km = dr_m / 1000.
    scantime = swp0.time.values.astype('<M8[s]')
    # mask uh and rho
    try:
        swp_msk = swp0.where((swp0['DBTH'] >= -10.))
    except:
        swp_msk = swp0.where((swp0[dbzh] >= -10.)) # or DBTH
    swp_msk = swp_msk.where(swp_msk.RHOHV > 0.40)
    swp_msk = swp_msk.where(swp_msk.range > dr_m * 5)

    phi_masked = swp_msk[phidp].copy()

    # median filtering 2d
    #medwin = 7
    phimed = filter_data(phi_masked, medwin=medwin)

    # gaussian convolution 1d - smoothing
    # play with sigma and kwidth
    #kwidth = 7
    #sigma = 10
    gkern = gauss_kernel(kwidth, sigma)
    phiclean = smooth_data(phimed, gkern)
    #phiclean = np.ma.array(phiclean, mask=res_mask)

    # median over all first range bins, broadcasted to all rays
    poffset = phase_offset(phiclean, rng=3000.)
    offset = poffset.PHIDP_OFFSET
    offset1 = np.ones(360) * np.nanmedian(offset)
    #print(np.nanmedian(offset1))

    phicorr = phiclean - offset1[:, np.newaxis]



    cphase = phase_zphi(phiclean, rng=1000.)
    #cphase.first.plot(label="first")
    #cphase.last.plot(label="last")
    dphi = cphase.last - cphase.first
    dphi = dphi.where(dphi>=0).fillna(0)

    alphax = alpha
    betax = beta
    bx = 0.78 # Average value representative of X or C band, see https://doi.org/10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2
    # need to expand alphax to dphi-shape
    fdphi = 10 ** (0.1 * bx * alphax * dphi) - 1

    zhraw = swp0[dbzh].where((swp0.range > cphase.start_range) & (swp0.range < cphase.stop_range))

    zax = zhraw.pipe(wrl.trafo.idecibel).fillna(0)

    za = zax ** bx

    # set masked to zero for integration
    za_zero = za.fillna(0)

    from scipy.integrate import cumtrapz
    iza_x = 0.46 * bx * cumtrapz(za_zero.values, axis=1, initial=0, dx=dr_km.values)
    iza = np.max(iza_x, axis=1)[:, None] - iza_x

    # we need some np.newaxis voodoo here, to get the correct dimensions (this version is for an array of alphax)
    #iza_fdphi = iza[np.newaxis, ...] / fdphi[..., np.newaxis]
    #iza_first = np.array([iza_fdphi[:, ray, first[ray]]
    #                      for ray in range(za.shape[0])])

    iza_fdphi = iza / fdphi.to_numpy()[..., np.newaxis]

    iza_first = np.array([iza_fdphi[ray, cphase.first_idx[ray]] for ray in range(za.shape[0])])

    ah_fast = ( za / (iza_first[:, None] + iza) ).compute()

    ah_fast = np.ma.masked_invalid(ah_fast).filled(0) # replace nans by zeros

    phical = 2 * cumtrapz(ah_fast/alphax, axis=1, dx=dr_km.values, initial=0)

    return ah_fast, phical, phicorr, cphase, alpha, beta


def phase_zphi(phi, rng=1000.):
    range_step = np.diff(phi.range)[0]
    nprec = int(rng / range_step)
    if nprec % 2:
        nprec += 1

    # create binary array
    phib = xr.where(np.isnan(phi), 0, 1)

    # take nprec range bins and calculate sum
    phib_sum = phib.rolling(range=nprec, center=True).sum(skipna=True)

    offset = nprec // 2 * np.diff(phib_sum.range)[0]
    offset_idx = nprec // 2
    start_range = phib_sum.idxmax(dim="range") - offset
    start_range_idx = phib_sum.argmax(dim="range") - offset_idx
    stop_range = phib_sum[:, ::-1].idxmax(dim="range") - offset
    stop_range_idx = len(phib_sum.range) - (phib_sum[:, ::-1].argmax(dim="range") - offset_idx) - 2
    # get phase values in specified range
    first = phi.where((phi.range >= start_range) & (phi.range <= start_range + rng),
                       drop=True).quantile(0.05, dim='range', skipna=True)
    last = phi.where((phi.range >= stop_range - rng) & (phi.range <= stop_range),
                       drop=True).quantile(0.95, dim='range', skipna=True)


    return xr.Dataset(dict(phib=phib_sum,
                           offset=offset,
                           offset_idx=offset_idx,
                           start_range=start_range,
                           stop_range=stop_range,
                           first=first,
                           first_idx=start_range_idx,
                           last=last,
                           last_idx=stop_range_idx,
                          ))

######## BETTER COLORMAPS FOR RADAR
# copied from https://github.com/EVS-ATMOS/CVD-colormaps/blob/master/code/colormaps.py
# Creating a dictionary containing the LangRainbow12 colormap values.
LangRainbow12_data = {
    'blue': [
        (0.0, 0.97000000000000008, 0.97000000000000008),
        (0.090909090909090912, 0.95599999999999996, 0.95599999999999996),
        (0.18181818181818182, 0.94500000000000006, 0.94500000000000006),
        (0.27272727272727271, 0.93700000000000006, 0.93700000000000006),
        (0.36363636363636365, 0.93199999999999994, 0.93199999999999994),
        (0.45454545454545459, 0.92999999999999994, 0.92999999999999994),
        (0.54545454545454541, 0.14900000000000002, 0.14900000000000002),
        (0.63636363636363635, 0.060000000000000053, 0.060000000000000053),
        (0.72727272727272729, 0.042000000000000037, 0.042000000000000037),
        (0.81818181818181823, 0.027000000000000024, 0.027000000000000024),
        (0.90909090909090917, 0.015000000000000013, 0.015000000000000013),
        (1.0, 0.0060000000000000053, 0.0060000000000000053)],
    'green': [
        (0.0, 0.82999999999999996, 0.82999999999999996),
        (0.090909090909090912, 0.7240000000000002, 0.7240000000000002),
        (0.18181818181818182, 0.64799999999999991, 0.64799999999999991),
        (0.27272727272727271, 0.67660000000000009, 0.67660000000000009),
        (0.36363636363636365, 0.76879999999999971, 0.76879999999999971),
        (0.45454545454545459, 0.92999999999999983, 0.92999999999999983),
        (0.54545454545454541, 0.93100000000000005, 0.93100000000000005),
        (0.63636363636363635, 0.75929999999999997, 0.75929999999999997),
        (0.72727272727272729, 0.54600000000000004, 0.54600000000000004),
        (0.81818181818181823, 0.35999999999999999, 0.35999999999999999),
        (0.90909090909090917, 0.20500000000000002, 0.20500000000000002),
        (1.0, 0.08415600000000005, 0.08415600000000005)],
    'red': [
        (0.0, 0.89999999999999991, 0.89999999999999991),
        (0.090909090909090912, 0.77039999999999997, 0.77039999999999997),
        (0.18181818181818182, 0.61499999999999999, 0.61499999999999999),
        (0.27272727272727271, 0.50300000000000011, 0.50300000000000011),
        (0.36363636363636365, 0.38800000000000012, 0.38800000000000012),
        (0.45454545454545459, 0.27000000000000024, 0.27000000000000024),
        (0.54545454545454541, 0.93099999999999983, 0.93099999999999983),
        (0.63636363636363635, 0.89999999999999991, 0.89999999999999991),
        (0.72727272727272729, 0.79800000000000004, 0.79800000000000004),
        (0.81818181818181823, 0.69299999999999995, 0.69299999999999995),
        (0.90909090909090917, 0.58500000000000008, 0.58500000000000008),
        (1.0, 0.4740000000000002, 0.4740000000000002)]
}


# Creating a dictionary of the Homeyer colormap values.
def yuv_rainbow_24(nc):
    path1 = np.linspace(0.8*np.pi, 1.8*np.pi, nc)
    path2 = np.linspace(-0.33*np.pi, 0.33*np.pi, nc)

    y = np.concatenate([np.linspace(0.3, 0.85, nc*2//5),
                        np.linspace(0.9, 0.0, nc - nc*2//5)])
    u = 0.40*np.sin(path1)
    v = 0.55*np.sin(path2) + 0.1

    rgb_from_yuv = np.array([[1, 0, 1.13983],
                             [1, -0.39465, -0.58060],
                             [1, 2.03211, 0]])
    cmap_dict = {'blue': [], 'green': [], 'red': []}
    for i in range(len(y)):
        yuv = np.array([y[i], u[i], v[i]])
        rgb = rgb_from_yuv.dot(yuv)
        red_tuple = (i/(len(y)-1), rgb[0], rgb[0])
        green_tuple = (i/(len(y)-1), rgb[1], rgb[1])
        blue_tuple = (i/(len(y)-1), rgb[2], rgb[2])
        cmap_dict['blue'].append(blue_tuple)
        cmap_dict['red'].append(red_tuple)
        cmap_dict['green'].append(green_tuple)
    return cmap_dict


data_dir = os.path.split(__file__)[0]
bal_rgb_vals = np.genfromtxt(os.path.join(data_dir, 'balance-rgb.txt'))

blue_to_red = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }


# Making a dictionary of all the colormap dictionaries.
better_colormaps = {
        'HomeyerRainbow': yuv_rainbow_24(15),
        'LangRainbow12': LangRainbow12_data,
        'Blue_to_red' : blue_to_red,
        'balance' : bal_rgb_vals}

def vertical_interpolation(vol, elevs=None, method="nearest"):
    """
    Vertically interpolate volume data
    
    elevs: iterable of elevations to which interpolate the data. Defaults to None, which does no interpolation and returns a stacked array of the data.
    method: method for interpolation, defaults to "nearest"
    """
    time = vol[0].time
    dsx = xr.concat([v.drop(["time","rtime"]).assign_coords({"elevation": v.attrs.get("fixed_angle")}) for v in vol], dim="elevation")
    dsx = dsx.transpose("time", "elevation", "azimuth", "range")
    if elevs is not None:
        new_elev=elevs
        dsx = dsx.interp(elevation=new_elev, method=method)
    dsx = dsx.assign_coords({"time": time})
    return dsx


def georeference_dataset(obj, **kwargs):
    """Georeference Dataset.

        .. versionadded:: 1.5

    This function adds georeference data to xarray Dataset/DataArray `obj`.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.Dataset` or :py:class:`xarray:xarray.DataArray`

    Keyword Arguments
    -----------------
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`, :py:class:`cartopy.crs.CRS` or None
        If GDAL OSR SRS, output is in this projection, else AEQD.
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    ----------
    obj : :py:class:`xarray:xarray.Dataset` or :py:class:`xarray:xarray.DataArray`
    """
    proj = kwargs.pop("proj", "None")
    re = kwargs.pop("re", None)
    ke = kwargs.pop("ke", 4.0 / 3.0)

    # adding xyz aeqd-coordinates
    site = (
        obj.coords["longitude"].values,
        obj.coords["latitude"].values,
        obj.coords["altitude"].values,
    )

    if site == (0.0, 0.0, 0.0):
        re = 6378137.0

    # create meshgrid to overcome dimension problem with spherical_to_xyz
    r, az = np.meshgrid(obj["range"], obj["azimuth"])
    
    # GDAL OSR, convert to this proj
    if isinstance(proj, osr.SpatialReference):
        xyz = wrl.georef.polar.spherical_to_proj(
            r, az, obj["elevation"], site, proj=proj, re=re, ke=ke
        )
    # other proj, convert to aeqd
    elif proj:
        xyz, dst_proj = wrl.georef.polar.spherical_to_xyz(
            r, az, obj["elevation"], site, re=re, ke=ke, squeeze=True
        )
    # proj, convert to aeqd and add offset
    else:
        xyz, dst_proj = wrl.georef.polar.spherical_to_xyz(
            r, az, obj["elevation"], site, re=re, ke=ke, squeeze=True
        )
        xyz += np.array(site).T
    
    #print(xyz.ndim, xyz.shape)
    
    # calculate center point
    # use first range bins
    ax = tuple(range(xyz.ndim - 2))
    center = np.mean(xyz[..., 0, :], axis=ax)

    # calculate ground range
    gr = np.sqrt((xyz[..., 0] - center[0]) ** 2 + (xyz[..., 1] - center[1]) ** 2)

    # dimension handling
    dim0 = obj["azimuth"].dims[-1]
    #print(dim0)
    #print(obj["elevation"].dims)
    if obj["elevation"].dims:
        dimlist = list(obj["elevation"].dims)
    else:
        dimlist = list(obj["azimuth"].dims)
    #print(dimlist)
    ########## THIS IS DIFFERENT WITH RESPECT TO wrl.georef.georeference_dataset
    # xyz is an array of cartesian coordinates for every spherical coordinate,
    # so the possible dimensions are (elevations, azimuths, range, 3)
    # for 2d, it either has (elevations, range, 3) or (azimuths, range, 3) dimensions
    # for 3d, the only option is the full (elevations, azimuths, range, 3) dimensions
    # Thus, i think adding this two lines for the 3d case will not break other functionalities
    if xyz.ndim > 3:
        dimlist += ["azimuth"]
    ##########
    dimlist += ["range"]

    #print(dimlist, xyz.shape)
    
    # add xyz, ground range coordinates
    obj.coords["x"] = (dimlist, xyz[..., 0])
    obj.coords["y"] = (dimlist, xyz[..., 1])
    obj.coords["z"] = (dimlist, xyz[..., 2])
    obj.coords["gr"] = (dimlist, gr)

    # adding rays, bins coordinates
    if obj.sweep_mode == "azimuth_surveillance":
        bins, rays = np.meshgrid(obj["range"], obj["azimuth"], indexing="xy")
    else:
        bins, rays = np.meshgrid(obj["range"], obj["elevation"], indexing="xy")
    obj.coords["rays"] = ([dim0, "range"], rays)
    obj.coords["bins"] = ([dim0, "range"], bins)

    return obj