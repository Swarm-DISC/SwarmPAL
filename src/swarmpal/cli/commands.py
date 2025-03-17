from __future__ import annotations

import click
import yaml

import swarmpal
from swarmpal.express import fac_single_sat as _fac_single_sat
from swarmpal.utils.configs import SPACECRAFT_TO_MAGLR_DATASET
from swarmpal.utils.queries import last_available_time as _last_available_time


def _read_config(filename):
    """Helper function to read and validate YAML config files"""
    with open(filename) as f:
        datasets = yaml.safe_load(f)
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


@cli.command(add_help_option=True, short_help="Fetch datasets from Vires or Hapi")
@click.argument("config", type=click.File("r"))
@click.argument("out", type=click.File("w"))
def fetch_data(config, out):
    """Fetch data described in yaml file CONFIG and save the resulting DataTree in NetCDF file OUT"""

    dataset_config = _read_config(config.name)
    data = swarmpal.fetch_data(dataset_config)
    data.to_netcdf(out.name)


@cli.command(
    add_help_option=True,
    short_help="Process datasets in batch mode",
    # help="Process datasets in batch mode for a given CONFIG file in yaml format",
)
@click.argument("config", type=click.File("r"))
@click.argument("out", type=click.File("w"))
def batch(config: click.File, out: click.Path):
    """Run SwarmPAL in batch mode. The datasets and processes need to be specified in YAML file and
    passed as the first CONFIG argument. The results are written to NetCDF files specified by OUT."""

    dataset_config = _read_config(config.name)
    data = swarmpal.fetch_data(dataset_config)

    # Apply processes
    for process_spec in dataset_config.get("process_params", []):
        process_name = process_spec.pop("process_name")
        process = swarmpal.make_process(process_name=process_name, config=process_spec)
        data = process(data)

    # Save the results as a NetCDF file
    data.to_netcdf(out.name)
