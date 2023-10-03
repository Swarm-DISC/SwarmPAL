from __future__ import annotations

import logging
from contextlib import contextmanager

import cartopy.crs as ccrs
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.colors import Normalize

from swarmpal.utils.exceptions import PalError

logger = logging.getLogger(__name__)

__all__ = (
    "plot_analysed_pass",
    "quicklook",
    "quicklook_animated",
)


@contextmanager
def _disable_mpl_interactive_mode():
    """Used to temporarily disable interactive plotting mode"""
    initial_state = plt.isinteractive()  # Get the initial state
    try:
        plt.ioff()  # Turn off interactive mode during the context
        yield
    finally:
        if initial_state:
            plt.ion()  # Restore interactive mode if it was initially on
        else:
            plt.ioff()  # Restore interactive mode if it was initially off


def _get_dsecs_meta(datatree, check_analysis=False):
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    if not pal_processes_meta.get("DSECS_Preprocess"):
        raise PalError("Must first run dsecs.processes.Preprocess")
    if check_analysis and pal_processes_meta.get("DSECS_Analysis") is None:
        raise PalError("Must first run dsecs.processes.Analysis")
    return pal_processes_meta


def plot_analysed_pass(datatree, pass_no=0):
    """Plot a figure showing currents from one orbital pass

    Parameters
    ----------
    datatree : DataTree
        A datatree processed with the DSECS toolbox
    pass_no : int
        A number between 0 and x, specifying the pass to plot

    Returns
    -------
    fig, axes
    """

    # Select the inputs we'll need for the figure
    pal_processes_meta = _get_dsecs_meta(datatree, check_analysis=True)
    datasat_name_alpha = pal_processes_meta["DSECS_Preprocess"]["dataset_alpha"]
    datasat_name_charlie = pal_processes_meta["DSECS_Preprocess"]["dataset_charlie"]
    data_a = datatree[datasat_name_alpha]
    data_c = datatree[datasat_name_charlie]
    data_currents = datatree[f"DSECS_output/{pass_no}/currents"]

    # Create a figure and axes with an orthographic projection
    # centred around the spacecraft longitude midpoint
    # central_lon = data_a["Longitude"].median().data
    central_lon = data_currents["Longitude"].isel(y=0).median().data
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        subplot_kw={
            "projection": ccrs.Orthographic(
                central_longitude=central_lon, central_latitude=0
            )
        },
    )
    # Set common view on each subplot, with spacecraft tracks
    for ax in axes:
        ax.set_global()
        ax.coastlines(color="purple", alpha=0.5)
        ax.scatter(
            data_a["Longitude"],
            data_a["Latitude"],
            color="grey",
            transform=ccrs.PlateCarree(),
            s=0.1,
            alpha=0.1,
        )
        ax.scatter(
            data_c["Longitude"],
            data_c["Latitude"],
            color="grey",
            transform=ccrs.PlateCarree(),
            s=0.1,
            alpha=0.1,
        )

    # Plot arrows for each of the DF and CF currents
    # # or a single central slice?:
    # slicenum = int(data_currents.dims["y"]/2)
    # current_slice = data_currents.isel(y=slicenum)
    for slicenum in range(data_currents.dims["y"]):
        current_slice = data_currents.isel(y=slicenum)
        axes[0].quiver(
            current_slice["Longitude"],
            current_slice["Latitude"],
            current_slice["JEastDf"],
            current_slice["JNorthDf"],
            transform=ccrs.PlateCarree(),
            scale=1000,
            color="blue",
            pivot="tail",
            width=0.0005,
        )
        axes[0].set_title("Horizontal DF current")
        axes[1].quiver(
            current_slice["Longitude"],
            current_slice["Latitude"],
            current_slice["JEastCf"],
            current_slice["JNorthCf"],
            transform=ccrs.PlateCarree(),
            scale=1000,
            color="blue",
            pivot="tail",
            width=0.0005,
        )
        axes[1].set_title("Horizontal CF current")

    # Set color min/max range, based on 99th percentile of data
    # Symmetric around 0, and rounded up to nearest 10
    vmin, vmax = np.nanquantile(data_currents["Jr"], (0.01, 0.99))
    vminmax = np.ceil(np.max(np.abs((vmin, vmax))) / 10) * 10
    norm = Normalize(vmin=-vminmax, vmax=vminmax)
    axes[2].pcolormesh(
        data_currents["Longitude"],
        data_currents["Latitude"],
        data_currents["Jr"],
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu",
        norm=norm,
    )
    axes[2].set_title("Radial current")

    return fig, axes


def quicklook(datatree, show_now=True):
    """Returns a dict of figures overviewing the outputs of the analysis

    Parameters
    ----------
    datatree : DataTree
        A datatree from the DSECS toolbox

    Returns
    -------
    dict[int, [fig, axes]]
    """

    try:
        _ = _get_dsecs_meta(datatree, check_analysis=True)
    except PalError:
        raise PalError("No quicklook available before analysis has been run")

    # Identify number of analysed passes and generate a fig for each one
    num_passes = len(datatree["DSECS_output"].children)
    fig_collection = {i: plot_analysed_pass(datatree, i) for i in range(num_passes)}

    return fig_collection


def quicklook_animated(datatree):
    """Creates an animation of quicklook plots, using ipywidgets

    Parameters
    ----------
    datatree : DataTree
        A datatree from the DSECS toolbox
    """

    # Identify number of analysed passes and prerender the figures for each
    num_passes = len(datatree["DSECS_output"].children)
    with _disable_mpl_interactive_mode():
        fig_collection = quicklook(datatree)

    # Get the figure associated with a particular pass
    def get_pass_figure(pass_no):
        fig, _ = fig_collection.get(pass_no)
        return fig

    # Generate widgets to use for the output and control
    output = widgets.Output()

    def update_figure(frame):
        with output:
            clear_output(wait=True)
            display(get_pass_figure(frame))

    play = widgets.Play(
        value=0,
        min=0,
        max=num_passes - 1,
        step=1,
        interval=1000,
        description="Press play",
        disabled=False,
        playing=False,
        repeat=True,
    )
    slider = widgets.IntSlider(value=0, min=0, max=num_passes - 1)
    # Link the widgets and create the display
    widgets.jslink((play, "value"), (slider, "value"))
    play.observe(lambda change: update_figure(change["new"]), "value")
    return widgets.VBox(
        (
            widgets.HBox([play, slider]),
            output,
        )
    )
