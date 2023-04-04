from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from swarmpal.utils.exceptions import PalError


def _get_tfa_meta(datatree):
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    if not pal_processes_meta.get("TFA_Preprocess"):
        raise PalError("Must first run tfa.processes.Preprocess")
    return pal_processes_meta


def _get_active_dataset_window(datatree, meta=None, clip_times=True, tlims=None):
    """Get the dataset, subselected to the analysis window"""
    pal_processes_meta = meta if meta else _get_tfa_meta(datatree)
    tfa_preprocess_meta = pal_processes_meta.get("TFA_Preprocess")
    subtree = datatree[tfa_preprocess_meta.get("dataset")]
    # Get the analysis time window if present
    dataset_palmeta = subtree.swarmpal.pal_meta.get(".", {})
    window = dataset_palmeta.get("analysis_window")
    # Slice out the relevant part of the dataset
    if tlims:
        subset_ds = subtree.ds.sel({"TFA_Time": slice(tlims[0], tlims[1])})
    elif clip_times and window:
        subset_ds = subtree.ds.sel({"TFA_Time": slice(window[0], window[1])})
    else:
        subset_ds = subtree.ds
    return subset_ds


def time_series(datatree, ax=None, clip_times=True, tlims=None):
    """Plot the time series of the active variable

    Parameters
    ----------
    datatree : DataTree
        A datatree from the TFA toolbox
    ax : AxesSubplot, optional
        Axes onto which to plot
    clip_times : bool, optional
        Clip to the analysis window, by default True
    tlims : tuple(str, str), optional
        Tuple of ISO strings to limit the plot to

    Returns
    -------
    fig, ax
    """
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    ds = _get_active_dataset_window(
        datatree, meta=meta, clip_times=clip_times, tlims=tlims
    )
    da = ds["TFA_Variable"]
    da_origin_name = meta["TFA_Preprocess"]["active_variable"]
    use_magnitude = meta["TFA_Preprocess"]["use_magnitude"]
    units = ds[da_origin_name].attrs.get("units")
    # Build figure
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    da.plot.line(x="TFA_Time", ax=ax)
    # Adjust axes
    da_label = f"|{da_origin_name}|" if use_magnitude else da_origin_name
    ytext = f"TFA: {da_label}"
    ytext = f"{ytext} ({units})" if units else ytext
    ax.set_ylabel(ytext)
    ax.set_xlabel("Time")
    ax.grid()
    return fig, ax


def spectrum(
    datatree, ax=None, clip_times=True, tlims=None, log=True, levels=None, **kwargs
):
    """Plot the dynamic spectrum of the result of the wavelet transform

    Parameters
    ----------
    datatree : DataTree
        A datatree from the TFA toolbox
    ax : AxesSubplot, optional
        Axes onto which to plot
    clip_times : bool, optional
        Clip to the analysis window, by default True
    tlims : tuple(str, str), optional
        Tuple of ISO strings to limit the plot to
    log : bool, optional
        Logarithmic scale, by default True
    levels : ndarray, optional
        Override the levels used in the colorbar

    Returns
    -------
    fig, ax
    """
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    ds = _get_active_dataset_window(
        datatree, meta=meta, clip_times=clip_times, tlims=tlims
    )
    da = ds["wavelet_power"]
    # Create new DataArray to be plotted
    if log:
        da = np.log10(da)
        da.name = "log(wavelet_power)"
    da = da.assign_coords({"Frequency": 1000 / da["scale"]})
    da["Frequency"].attrs = {
        "units": "mHz",
    }
    # Identify levels to use in colorbar (can be overridden)
    if levels is None:
        lower, upper = np.min(da), np.max(da)
        levels = np.linspace(lower, upper, 20)
    # Identify other settings to use in plot (can be overridden)
    cmap = kwargs.pop("cmap", "jet")
    cbar_kwargs = kwargs.pop(
        "cbar_kwargs",
        {"location": "right", "format": "%.1f"},
    )
    # Build figure
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    da.plot.contourf(
        x="TFA_Time",
        y="Frequency",
        cmap=cmap,
        levels=levels,
        extend="both",
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )
    # Adjust axes
    ax.set_xlabel("Time")
    return fig, ax


def _get_wave_index(datatree, clip_times=True, tlims=None):
    """Evaluate wave index from wavelet power"""
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    ds = _get_active_dataset_window(
        datatree, meta=meta, clip_times=clip_times, tlims=tlims
    )
    da = ds["wavelet_power"]
    return xr.DataArray(
        data=np.nansum(da, axis=0),
        coords={"TFA_Time": da["TFA_Time"]},
        name="wave_index",
    )


def wave_index(datatree, ax=None, clip_times=True, tlims=None):
    """Plot the index of wave activity

    Parameters
    ----------
    datatree : DataTree
        A datatree from the TFA toolbox
    ax : AxesSubplot, optional
        Axes onto which to plot
    clip_times : bool, optional
        Clip to the analysis window, by default True
    tlims : tuple(str, str), optional
        Tuple of ISO strings to limit the plot to
    """
    da = _get_wave_index(datatree, clip_times=clip_times, tlims=tlims)
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    da.plot.line(x="TFA_Time", ax=ax)
    ax.set_xlabel("Time")
    ax.grid()
    return fig, ax


def quicklook(datatree):
    """Returns a figure overviewing relevant contents of the data"""
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    time_series(datatree, ax=ax1)
    try:
        wave_index(datatree, ax=ax2)
        spectrum(datatree, ax=ax3)
    except KeyError:
        pass
    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    return fig, (ax1, ax2, ax3)
