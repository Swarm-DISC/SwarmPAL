from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from swarmpal.utils.exceptions import PalError


def _get_tfa_meta(datatree):
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    if not pal_processes_meta.get("TFA_Preprocess"):
        raise PalError("Must first run tfa.processes.Preprocess")
    return pal_processes_meta


def _get_active_dataset(datatree, meta=None):
    pal_processes_meta = meta if meta else _get_tfa_meta(datatree)
    tfa_preprocess_meta = pal_processes_meta.get("TFA_Preprocess")
    return datatree[tfa_preprocess_meta.get("dataset")].ds


def time_series(datatree, ax=None):
    """Plot the time series of the active variable"""
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    ds = _get_active_dataset(datatree, meta=meta)
    da = ds["TFA_Variable"]
    da_origin_name = meta["TFA_Preprocess"]["active_variable"]
    units = ds[da_origin_name].attrs.get("units")
    # Build figure
    fig, ax = (None, ax) if ax else plt.subplots(1, 1)
    da.plot.line(x="Timestamp", ax=ax)
    # Adjust axes
    ytext = f"TFA: {da_origin_name}"
    ytext = f"{ytext} ({units})" if units else ytext
    ax.set_ylabel(ytext)
    ax.grid()
    return fig, ax


def spectrum(datatree, log=True, clip_times=True, levels=None, ax=None, **kwargs):
    """Plot the dynamic spectrum of the result of the wavelet transform"""
    # Extract relevant DataArray (da) and info
    meta = _get_tfa_meta(datatree)
    ds = _get_active_dataset(datatree, meta=meta)
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
        x="Timestamp",
        y="Frequency",
        cmap=cmap,
        levels=levels,
        extend="both",
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )
    return fig, ax


def quicklook(datatree):
    """Returns a figure overviewing relevant contents of the data"""
    ...
