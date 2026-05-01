from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
import yaml
from click.testing import CliRunner

from swarmpal.cli import cli
from swarmpal.cli.commands import _update_times

from .io.test_paldata import fetch_pal_meta_checks

# Unit tests for swarmpal.cli using the click library's recommendation at
#  https://click.palletsprojects.com/en/stable/testing/


@pytest.fixture
def cli_runner():
    return CliRunner()


def _format_cmd_non_zero_message(cmd_args):
    return f"CLI non-zero exit code: swarmpal {' '.join(cmd_args)}"


def test_cli_spacecraft(cli_runner):
    cmd_args = ["spacecraft"]
    result = cli_runner.invoke(cli, cmd_args)
    assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
    known_spacecraft = [
        "Swarm-A",
        "Swarm-B",
        "Swarm-C",
        "CHAMP",
        "CryoSat-2",
        "GRACE-A",
        "GRACE-B",
        "GRACE-FO-1",
        "GRACE-FO-2",
        "GOCE",
    ]
    for spacecraft in known_spacecraft:
        assert spacecraft in result.output
    assert len(known_spacecraft) == len(result.output.strip().split("\n"))


@pytest.mark.remote
def test_cli_last_available_time(cli_runner):
    cmd_args = ["last-available-time", "SW_FAST_MAGA_LR_1B"]
    result = cli_runner.invoke(cli, cmd_args)
    assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
    # Check that the output (without the trailing newline) parses to a time
    datetime.strptime(result.output.strip(), "%Y-%m-%dT%H:%M:%S.%f")


@pytest.mark.remote
def test_cli_fac_single_sat(cli_runner, tmp_path):
    cmd_args = [
        "fac-single-sat",
        "--spacecraft",
        "Swarm-A",
        "--time_start",
        "2016-01-01T00:00:00",
        "--time_end",
        "2016-01-01T01:00:00",
        "--to_cdf_file",
        "output.cdf",
        "--grade",
        "FAST",
    ]
    with cli_runner.isolated_filesystem(tmp_path):
        result = cli_runner.invoke(cli, cmd_args)
        assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
        assert os.path.exists("output.cdf")


test_yaml_data_params = """
data_params:
  - provider: vires
    collection: "SW_OPER_MAGA_LR_1B"
    measurements: ["B_NEC"]
    models: ["CHAOS"]
    start_time: "2016-01-01T00:00:00"
    end_time: "2016-01-01T00:30:00"
    server_url: "https://vires.services/ows"
"""
test_yaml_process_params = """
process_params:
  - process_name: FAC_single_sat
    dataset: SW_OPER_MAGA_LR_1B
    model_varname: B_NEC_CHAOS
    measurement_varname: B_NEC
    inclination_limit: 30
    time_jump_limit: 6
"""


@pytest.mark.remote
def test_cli_fetch_data(cli_runner, tmp_path):
    output_filename = "output.nc4"
    input_filename = "input.yaml"
    cmd_args = ["fetch-data", input_filename, output_filename]

    with cli_runner.isolated_filesystem(tmp_path):
        with open(input_filename, "w") as f:
            f.write(test_yaml_data_params)

        config = yaml.safe_load(test_yaml_data_params)

        result = cli_runner.invoke(cli, cmd_args)
        assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
        assert os.path.exists(output_filename)

        ds = xr.open_datatree(output_filename)
        assert "Spacecraft" in ds["/SW_OPER_MAGA_LR_1B"]
        fetch_pal_meta_checks(
            ds.swarmpal.pal_meta["SW_OPER_MAGA_LR_1B"], config["data_params"][0]
        )


@pytest.mark.remote
def test_cli_batch(cli_runner, tmp_path):
    output_filename = "output.nc4"
    input_filename = "input.yaml"
    cmd_args = ["batch", input_filename, output_filename]

    with cli_runner.isolated_filesystem(tmp_path):
        with open(input_filename, "w") as f:
            f.write(test_yaml_data_params)
            f.write(test_yaml_process_params)

        result = cli_runner.invoke(cli, cmd_args)
        assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
        assert os.path.exists(output_filename)


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    [
        ("a", "b"),
        ("2016-01-01T00:00:00", "b"),
        ("a", "2016-01-01T00:00:00"),
    ],
)
def test_update_times_fails(start_time, end_time, cli_runner, tmp_path):
    output_filename = "output.nc4"
    input_filename = "input.yaml"
    cmd_args = [
        "batch",
        "--time",
        start_time,
        end_time,
        input_filename,
        output_filename,
    ]

    with cli_runner.isolated_filesystem(tmp_path):
        with open(input_filename, "w") as f:
            f.write(test_yaml_data_params)

        result = cli_runner.invoke(cli, cmd_args)
        assert "time should be in ISO8601 format." in result.output
        assert result.exit_code == 1


test_yaml_data_params_multiple = """
data_params:
  - provider: vires
    collection: "SW_OPER_MAGA_LR_1B"
    measurements: ["B_NEC"]
    models: ["Model = CHAOS"]
    auxiliaries: ["QDLat"]
    start_time: "2016-03-18T11:00:00"
    end_time: "2016-03-18T11:30:00"
    server_url: https://vires.services/ows
  - provider: vires
    collection: SW_OPER_MAGC_LR_1B
    measurements: ["B_NEC"]
    models: ["Model = CHAOS"]
    auxiliaries: ["QDLat"]
    start_time: "2016-03-18T11:00:00"
    end_time: "2016-03-18T11:30:00"
    server_url: https://vires.services/ows
"""
test_yaml_data_params_hapi = """
data_params:
  - provider: hapi
    dataset: "SW_OPER_MAGA_LR_1B"
    parameters: "F,B_NEC"
    start: "2016-01-01T00:00:00"
    stop: "2016-01-01T00:00:10"
    server: "https://vires.services/hapi"
"""


@pytest.mark.parametrize(
    ("start_time", "end_time", "config_content"),
    [
        ("2017-01-01T00:00:00", "2017-01-01T00:30:00", test_yaml_data_params),
        ("2023-01-01T00:00:00", "2023-01-01T00:30:00", test_yaml_data_params_multiple),
        ("2022-01-01T00:00:00", "2022-01-01T00:30:00", test_yaml_data_params_hapi),
    ],
)
def test_cli_update_times_helper(start_time, end_time, config_content):
    datasets = yaml.safe_load(config_content)
    _update_times(start_time, end_time, datasets)
    for dataset in datasets["data_params"]:
        if dataset["provider"] == "vires":
            assert dataset["start_time"] == start_time
            assert dataset["end_time"] == end_time
        if dataset["provider"] == "hapi":
            assert dataset["start"] == start_time
            assert dataset["stop"] == end_time


@pytest.mark.remote
@pytest.mark.parametrize(
    ("start_time", "end_time", "config_content"),
    [
        ("2017-01-01T00:00:00", "2017-01-01T00:30:00", test_yaml_data_params),
        ("2023-01-01T00:00:00", "2023-01-01T00:30:00", test_yaml_data_params_multiple),
        ("2022-01-01T00:00:00", "2022-01-01T00:30:00", test_yaml_data_params_hapi),
    ],
)
def test_cli_update_times(start_time, end_time, config_content, cli_runner, tmp_path):
    output_filename = "output.nc4"
    input_filename = "input.yaml"
    cmd_args = [
        "fetch-data",
        "--time",
        start_time,
        end_time,
        input_filename,
        output_filename,
    ]

    with cli_runner.isolated_filesystem(tmp_path):
        with open(input_filename, "w") as f:
            f.write(config_content)

        result = cli_runner.invoke(cli, cmd_args)
        assert result.exit_code == 0, result.output

        ds = xr.open_datatree(output_filename)
        for dataset in ds.children:
            timestamps = ds[dataset]["Timestamp"].to_numpy()
            assert np.all(np.datetime64(start_time) <= timestamps)
            assert np.all(np.datetime64(end_time) >= timestamps)
