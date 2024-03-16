from __future__ import annotations

import click

from swarmpal.toolboxes.fac.presets import fac_single_sat as preset_fac_single_sat
from swarmpal.utils.configs import SPACECRAFT_TO_MAGLR_DATASET


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
@click.option("--grade", required=True, help="'OPER' or 'FAST'")
@click.option("--time_start", required=True, help="ISO 8601 time")
@click.option("--time_end", required=True, help="ISO 8601 time")
@click.option("--output", required=True, help="Output CDF file")
def fac_single_sat(
    spacecraft: str,
    grade: str,
    time_start: str,
    time_end: str,
    output: str,
):
    return preset_fac_single_sat(spacecraft, grade, time_start, time_end, output)
