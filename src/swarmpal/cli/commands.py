from __future__ import annotations

import os
import sys

import click
import yaml
from xarray import open_datatree

import swarmpal
from swarmpal.express import fac_single_sat as _fac_single_sat
from swarmpal.schema import is_iso8601_datetime
from swarmpal.utils.configs import SPACECRAFT_TO_MAGLR_DATASET
from swarmpal.utils.queries import last_available_time as _last_available_time


def _update_times(start_time, end_time, dataset_config):
    """Helper to update start and end times in a dataset configuration."""
    if not is_iso8601_datetime(start_time):
        click.echo(
            f"The start time should be in ISO8601 format. Got '{start_time}' instead.",
            err=True,
        )
        sys.exit(1)
    if not is_iso8601_datetime(end_time):
        click.echo(
            f"The end time should be in ISO8601 format. Got '{end_time}' instead.",
            err=True,
        )
        sys.exit(1)

    for dataset in dataset_config["data_params"]:
        # Vires format
        if "start_time" in dataset:
            dataset["start_time"] = start_time
        if "end_time" in dataset:
            dataset["end_time"] = end_time
        # HAPI format
        if "start" in dataset:
            dataset["start"] = start_time
        if "stop" in dataset:
            dataset["stop"] = end_time


def _read_config(filename, times):
    """Helper function to read and validate YAML config files"""
    with open(filename) as f:
        datasets = yaml.safe_load(f)
    if times is not None:
        _update_times(times[0], times[1], datasets)
    return datasets


@click.group()
def cli():
    pass


@cli.command()
def spacecraft():
    """List names of available spacecraft"""
    spacecraft = list(SPACECRAFT_TO_MAGLR_DATASET.keys())
    click.echo("\n".join(spacecraft))


@cli.command(add_help_option=True)
@click.option(
    "--spacecraft", required=True, help="Check available with: swarmpal spacecraft"
)
@click.option("--time_start", required=True, help="ISO 8601 time")
@click.option("--time_end", required=True, help="ISO 8601 time")
@click.option("--grade", required=True, help="'OPER' or 'FAST'")
@click.option("--to_cdf_file", required=True, help="Output CDF file")
def fac_single_sat(
    spacecraft: str, time_start: str, time_end: str, grade: str, to_cdf_file: str
):
    """Execute FAC single-satellite processor"""
    return _fac_single_sat(spacecraft, time_start, time_end, grade, to_cdf_file)


@cli.command(add_help_option=True)
@click.argument("collection")
def last_available_time(collection):
    """UTC of last available data for a collection, e.g. SW_FAST_MAGA_LR_1B"""
    time = _last_available_time(collection)
    click.echo(time.isoformat())


def _check_overwrite(file: click.Path, overwrite: bool):
    if os.path.exists(file.name) and not overwrite:
        click.echo(
            f"Output file '{file.name}' already exists. Use --overwrite to replace it.",
            err=True,
        )
        sys.exit(1)


@cli.command(add_help_option=True, short_help="Fetch datasets from Vires or Hapi")
@click.option(
    "--time",
    nargs=2,
    type=str,
    help="Override the start and end times in the configuration file",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output file if it already exists",
)
@click.argument("config", type=click.File("r"))
@click.argument("out", type=click.File("w"))
def fetch_data(time, overwrite, config: click.File, out: click.Path):
    """Fetch data described in yaml file CONFIG and save the resulting DataTree in NetCDF file OUT"""
    _check_overwrite(out, overwrite)
    dataset_config = _read_config(config.name, time)
    data = swarmpal.fetch_data(dataset_config)
    data.to_netcdf(out.name)


@cli.command(
    add_help_option=True,
    short_help="Process datasets in batch mode",
    # help="Process datasets in batch mode for a given CONFIG file in yaml format",
)
@click.option(
    "--time",
    nargs=2,
    type=str,
    help="Override the start and end times in the configuration file",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output file if it already exists",
)
@click.argument("config", type=click.File("r"))
@click.argument("out", type=click.File("w"))
def batch(time, overwrite, config: click.File, out: click.Path):
    """Run SwarmPAL in batch mode. The datasets and processes need to be specified in YAML file and
    passed as the first CONFIG argument. The results are written to NetCDF files specified by OUT."""

    _check_overwrite(out, overwrite)

    dataset_config = _read_config(config.name, time)
    data = swarmpal.fetch_data(dataset_config)

    # Apply processes
    swarmpal.apply_processes(data, dataset_config.get("process_params", []))

    # Save the results as a NetCDF file
    data.to_netcdf(out.name)


@cli.command()
@click.argument("file", type=click.File("r"))
@click.argument("out", type=click.File("w"))
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output file if it already exists",
)
def quicklook(overwrite, file: click.File, out: click.Path):
    """Create a quicklook plot if possible"""
    _check_overwrite(out, overwrite)
    data = open_datatree(file.name)
    fig = swarmpal.quicklook(data)
    fig.savefig(out.name)
