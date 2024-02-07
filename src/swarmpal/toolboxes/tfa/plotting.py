from __future__ import annotations

import logging

import matplotlib.dates as mdt
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec

from swarmpal.utils.exceptions import PalError

logger = logging.getLogger(__name__)


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


def _add_secondary_x_axes(
    dataset=None, ax=None, varnames=("Latitude", "Longitude"), timevar="Timestamp"
):
    """Add a number of secondary x-axes at the top of the plot"""
    # Restrict to those which are available in the data
    varnames_available = set(varnames).intersection(set(dataset.data_vars))
    for x in set(varnames).difference(varnames_available):
        logger.warning(f" Skipping {x}: not available in data")
    varnames = [v for v in varnames if v in varnames_available]
    if len(varnames) == 0:
        return ax
    # Identify the times at each xtick location
    t = mdt.num2date(ax.get_xticks())
    t = [_t.replace(tzinfo=None) for _t in t]

    def add_xaxis(varname="Latitude", timevar=timevar, yposition=1.0):
        # Identify and format the variable value at the tick locations
        x = dataset[varname].sel({timevar: t}, method="nearest").data
        x = [f"{_x:.1f}" for _x in x]
        # Add the secondary xaxis at the given y-position
        ax_new = ax.secondary_xaxis(yposition)
        ax_new.set_xlabel(
            varname,
            position=(-0.02, 0),
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_new.transAxes,
        )
        ax_new.set_xticks(ax.get_xticks())
        ax_new.set_xticklabels(x, rotation=30, horizontalalignment="left")
        return ax_new

    # Place the first secondary x-axis
    ax_new = add_xaxis(varname=varnames[0], yposition=1.02)
    # Find the height of the placed labels, in figure coordinates
    bbox = ax_new.get_xticklabels()[0].get_window_extent(
        renderer=ax.figure.canvas.get_renderer()
    )
    bbox = bbox.transformed(ax.figure.dpi_scale_trans.inverted())
    label_height = bbox.height
    # Recalculate that height as a fraction of main axes height
    bbox_ax = ax.get_window_extent()
    bbox_ax = bbox_ax.transformed(ax.figure.dpi_scale_trans.inverted())
    label_height = label_height / bbox_ax.height
    # Increase it to account for the extra height taken by label rotation(?)
    label_height *= 1.5
    # Use that height to intelligently place the other x-axes
    if len(varnames) > 1:
        for i, varname in enumerate(varnames[1:]):
            new_y = 1.02 + (i + 1) * label_height
            ax_new = add_xaxis(varname=varname, yposition=new_y)

    return ax


def time_series(
    datatree,
    varname="TFA_Variable",
    ax=None,
    clip_times=True,
    tlims=None,
    extra_x=("QDLat", "MLT"),
    **kwargs,
):
    """Plot the time series of the active variable, or any other variable

    Parameters
    ----------
    datatree : DataTree
        A datatree from the TFA toolbox
    varname : str
        Select a variable from within the dataset
    ax : AxesSubplot, optional
        Axes onto which to plot
    clip_times : bool, optional
        Clip to the analysis window, by default True
    tlims : tuple(str, str), optional
        Tuple of ISO strings to limit the plot to
    extra_x : tuple(str), optional
        Variables to add as extra x-axes

    Returns
    -------
    fig, ax
    """
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    timevar = meta["TFA_Preprocess"]["timevar"]
    ds = _get_active_dataset_window(
        datatree, meta=meta, clip_times=clip_times, tlims=tlims
    )
    if varname == "TFA_Variable":
        da = ds["TFA_Variable"]
        da_origin_name = meta["TFA_Preprocess"]["active_variable"]
        use_magnitude = meta["TFA_Preprocess"]["use_magnitude"]
    else:
        da = ds[varname]
        da_origin_name = da.name
        use_magnitude = False
    units = ds[da_origin_name].attrs.get("units")
    # Build figure
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    mainvar_timevar = "TFA_Time" if "TFA_Time" in da.coords else timevar
    da.plot.line(x=mainvar_timevar, ax=ax, **kwargs)
    # Add the extra x-axes as required
    if extra_x:
        ax = _add_secondary_x_axes(ds, ax, varnames=extra_x, timevar=timevar)
    # Adjust axes
    da_label = f"|{da_origin_name}|" if use_magnitude else da_origin_name
    ytext = f"TFA: {da_label}"
    ytext = f"{ytext} ({units})" if units else ytext
    ax.set_ylabel(ytext)
    ax.set_xlabel("Time")
    ax.grid()
    return fig, ax


def spectrum(
    datatree,
    ax=None,
    clip_times=True,
    tlims=None,
    log=True,
    levels=None,
    extra_x=("QDLat", "MLT"),
    **kwargs,
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
    extra_x : tuple(str), optional
        Variables to add as extra x-axes

    Returns
    -------
    fig, ax
    """
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    timevar = meta["TFA_Preprocess"]["timevar"]
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
    # Add the extra x-axes as required
    if extra_x:
        ax = _add_secondary_x_axes(ds, ax, varnames=extra_x, timevar=timevar)
    # Adjust axes
    ax.set_xlabel("Time")
    return fig, ax


def _get_wave_index(
    dataset,
):
    """Evaluate wave index from wavelet power"""
    da = dataset["wavelet_power"]
    return xr.DataArray(
        data=np.nansum(da, axis=0),
        coords={"TFA_Time": da["TFA_Time"]},
        name="wave_index",
    )


def wave_index(
    datatree, ax=None, clip_times=True, tlims=None, extra_x=("QDLat", "MLT"), **kwargs
):
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
    extra_x : tuple(str), optional
        Variables to add as extra x-axes

    Returns
    -------
    fig, ax
    """
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    timevar = meta["TFA_Preprocess"]["timevar"]
    ds = _get_active_dataset_window(
        datatree, meta=meta, clip_times=clip_times, tlims=tlims
    )
    da = _get_wave_index(ds)
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    da.plot.line(x="TFA_Time", ax=ax, **kwargs)
    ax.set_xlabel("Time")
    ax.grid()
    # Add the extra x-axes as required
    if extra_x:
        ax = _add_secondary_x_axes(ds, ax, varnames=extra_x, timevar=timevar)
    return fig, ax


def quicklook(
    datatree,
    clip_times=True,
    tlims=None,
    extra_x=("QDLat", "MLT"),
):
    """Returns a figure overviewing relevant contents of the data

    Parameters
    ----------
    datatree : DataTree
        A datatree from the TFA toolbox
    clip_times : bool, optional
        Clip to the analysis window, by default True
    tlims : tuple(str, str), optional
        Tuple of ISO strings to limit the plot to
    extra_x : tuple(str), optional
        Variables to add as extra x-axes

    Returns
    -------
    fig, axes
    """
    fig = plt.figure(figsize=(15, 6))
    fig.set_clip_on(False)
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    time_series(datatree, ax=ax1, clip_times=clip_times, tlims=tlims, extra_x=extra_x)
    try:
        wave_index(
            datatree,
            ax=ax2,
            clip_times=clip_times,
            tlims=tlims,
            extra_x=None,
        )
        spectrum(datatree, ax=ax3, clip_times=clip_times, tlims=tlims, extra_x=extra_x)
    except KeyError:
        pass
    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    return fig, (ax1, ax2, ax3)
