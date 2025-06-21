from __future__ import annotations

import os
from datetime import datetime

import pytest
from click.testing import CliRunner

from swarmpal.cli import cli

# Unit tests for swarmpal.cli using the click library's recommendation at
#  https://click.palletsprojects.com/en/stable/testing/


@pytest.fixture()
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


@pytest.mark.remote()
def test_cli_last_available_time(cli_runner):
    cmd_args = ["last-available-time", "SW_FAST_MAGA_LR_1B"]
    result = cli_runner.invoke(cli, cmd_args)
    assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
    # Check that the output (without the trailing newline) parses to a time
    datetime.strptime(result.output.strip(), "%Y-%m-%dT%H:%M:%S.%f")


@pytest.mark.remote()
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


@pytest.mark.remote()
def test_cli_fetch_data(cli_runner, tmp_path):
    output_filename = "output.nc4"
    input_filename = "input.yaml"
    cmd_args = ["fetch-data", input_filename, output_filename]

    with cli_runner.isolated_filesystem(tmp_path):
        with open(input_filename, "w") as f:
            f.write(test_yaml_data_params)

        result = cli_runner.invoke(cli, cmd_args)
        assert result.exit_code == 0, _format_cmd_non_zero_message(cmd_args)
        assert os.path.exists(output_filename)


@pytest.mark.remote()
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
