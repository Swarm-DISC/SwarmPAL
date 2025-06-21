from __future__ import annotations

import datetime as dt
import logging
from contextlib import contextmanager
from functools import wraps

import cartopy.crs as ccrs
import ipywidgets as widgets
import matplotlib as mpl
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


def _turn_off_interactive_mode(func):
    """Used to temporarily disable interactive plotting mode"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        initially_interactive = mpl.is_interactive()
        plt.ioff()  # Turn off interactive mode before calling the function
        result = func(*args, **kwargs)
        if initially_interactive:
            plt.ion()  # Turn interactive mode back on after the function call
        return result

    return wrapper


def _get_dsecs_meta(datatree, check_analysis=False):
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    if not pal_processes_meta.get("DSECS_Preprocess"):
        raise PalError("Must first run dsecs.processes.Preprocess")
    if check_analysis and pal_processes_meta.get("DSECS_Analysis") is None:
        raise PalError("Must first run dsecs.processes.Analysis")
    return pal_processes_meta


def _get_dsecs_pass_time_interval(datatree, pass_no=0):
    """Extract time start and end of a given pass"""
    s = datatree[f"DSECS_output/{pass_no}"]["currents"].attrs["Time interval"]
    t1, t2 = s.split(" - ")
    t1 = dt.datetime.fromisoformat(t1.split(".")[0])
    t2 = dt.datetime.fromisoformat(t2.split(".")[0])
    return t1, t2


@_turn_off_interactive_mode
def plot_analysed_pass(datatree, pass_no=0, extent="global"):
    """Plot a figure showing currents from one orbital pass

    Parameters
    ----------
    datatree : DataTree
        A datatree processed with the DSECS toolbox
    pass_no : int
        A number between 0 and x, specifying the pass to plot
    extent : str | tuple, default "global"
        "global" or "automatic", or a tuple to be provided to ax.set_extent()

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Select the inputs we'll need for the figure
    pal_processes_meta = _get_dsecs_meta(datatree, check_analysis=True)
    dataset_name_alpha = pal_processes_meta["DSECS_Preprocess"]["dataset_alpha"]
    dataset_name_charlie = pal_processes_meta["DSECS_Preprocess"]["dataset_charlie"]
    data_a = datatree[dataset_name_alpha]
    data_c = datatree[dataset_name_charlie]
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
        if extent == "global":
            ax.set_global()
        elif extent == "automatic":
            min_lon = data_currents["Longitude"].min() - 10
            max_lon = data_currents["Longitude"].max() + 10
            ax.set_extent([min_lon, max_lon, -50, 50])
        else:
            ax.set_extent(extent)
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
            pivot="tail",
            angles="uv",
            scale=0.0001,
            scale_units="xy",
            width=10000,
            units="xy",
            color="blue",
        )
        axes[0].set_title("Horizontal DF current")
        axes[1].quiver(
            current_slice["Longitude"],
            current_slice["Latitude"],
            current_slice["JEastCf"],
            current_slice["JNorthCf"],
            transform=ccrs.PlateCarree(),
            pivot="tail",
            angles="uv",
            scale=0.0001,
            scale_units="xy",
            width=10000,
            units="xy",
            color="blue",
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

    # Add time start and end of pass, and dataset sources
    # TODO: Add product version numbers
    title_text = f"{dataset_name_alpha}\n{dataset_name_charlie}"
    t1, t2 = _get_dsecs_pass_time_interval(datatree, pass_no=pass_no)
    title_text += f"\nStart: {t1.isoformat()}\nEnd: {t2.isoformat()}"
    fig.suptitle(title_text, x=0.9, ha="right", va="bottom")

    plt.close()

    return fig


@_turn_off_interactive_mode
def quicklook(datatree, frame_select="all"):
    """Returns figures overviewing the outputs of the analysis

    Parameters
    ----------
    datatree : DataTree
        A datatree from the DSECS toolbox
    frame_select : str, default "all"
        "all", "odd", "even" to limit the frame numbers displayed

    Returns
    -------
    dict[int, matplotlib.figure.Figure]
    """

    try:
        _ = _get_dsecs_meta(datatree, check_analysis=True)
    except PalError:
        raise PalError("No quicklook available before analysis has been run")

    # Identify number of analysed passes and generate a fig for each one
    num_passes = len(datatree["DSECS_output"].children)
    # Config to select which frames to generate
    if frame_select == "all":
        frames = range(num_passes)
    elif frame_select == "odd":
        frames = [i for i in range(num_passes) if i % 2 != 0]
    elif frame_select == "even":
        frames = [i for i in range(num_passes) if i % 2 == 0]
    else:
        raise ValueError("frame_select should be 'all', 'odd', or 'even'")

    fig_collection = {}
    for i in frames:
        fig_collection[i] = plot_analysed_pass(datatree, i)

    return fig_collection


@_turn_off_interactive_mode
def quicklook_animated(datatree, frame_select="all"):
    """Creates an animation of quicklook plots, using ipywidgets

    Parameters
    ----------
    datatree : DataTree
        A datatree from the DSECS toolbox
    frame_select : str, default "all"
        "all", "odd", "even" to limit the frame numbers displayed
    """

    # Prerender the figures for each frame
    fig_collection = quicklook(datatree, frame_select=frame_select)

    # Get the figure associated with a particular pass
    def get_pass_figure(pass_no):
        fig = fig_collection.get(pass_no)
        return fig

    # Generate widgets to use for the output and control
    output = widgets.Output()

    def update_figure(frame):
        with output:
            clear_output(wait=True)
            display(get_pass_figure(frame))

    # Select frames according to odd/even
    frames = tuple(fig_collection.keys())
    frame_start = frames[0]
    frame_end = frames[-1]
    frame_step = frames[1] - frames[0]
    slider = widgets.IntSlider(
        value=frame_start, min=frame_start, max=frame_end, step=frame_step
    )
    play = widgets.Play(
        value=frame_start,
        min=frame_start,
        max=frame_end,
        step=frame_step,
        interval=1200,
        description="Press play",
        disabled=False,
        playing=False,
        repeat=True,
    )
    # Link the widgets and create the display
    widgets.jslink((play, "value"), (slider, "value"))
    play.observe(lambda change: update_figure(change["new"]), "value")

    return widgets.VBox(
        (
            widgets.HBox([play, slider]),
            output,
        )
    )
