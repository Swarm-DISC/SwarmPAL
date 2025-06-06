from __future__ import annotations

import numpy as np
import pytest
from xarray import DataTree

import swarmpal.toolboxes
from swarmpal.toolboxes import dsecs

from ..test_data import load_test_config, load_test_datatree


@pytest.mark.dsecs()
def test_by_name():
    """The DSECS toolbox was added to the toolboxes lookup dictionary"""
    preprocess = swarmpal.make_process("DSECS_Preprocess")
    assert isinstance(preprocess, dsecs.processes.Preprocess)

    analysis = swarmpal.make_process("DSECS_Analysis")
    assert isinstance(analysis, dsecs.processes.Analysis)


@pytest.mark.cached()
@pytest.mark.dsecs()
def test_dsecs_basic():
    input_data = DataTree.from_dict(
        {
            "SW_OPER_MAGA_LR_1B": load_test_datatree(
                "test_dsecs_basic.nc4", group="SW_OPER_MAGA_LR_1B"
            ),
            "SW_OPER_MAGC_LR_1B": load_test_datatree(
                "test_dsecs_basic.nc4", group="SW_OPER_MAGC_LR_1B"
            ),
        }
    )

    assert "SW_OPER_MAGA_LR_1B" in input_data
    assert "SW_OPER_MAGC_LR_1B" in input_data

    dataset_meta = load_test_config("test_dsecs_basic")
    for config in dataset_meta["process_params"]:
        process_name = config.pop("process_name")
        process = swarmpal.make_process(process_name=process_name, config=config)
        input_data = process(input_data)

    assert "DSECS_output" in input_data
    assert len(input_data["DSECS_output"]) == 1

    test_data = DataTree.from_dict(
        {
            "DSECS_output": load_test_datatree(
                "test_dsecs_basic.nc4", group="DSECS_output"
            ),
        }
    )

    variables = [
        "JEastDf",
        "JNorthDf",
        "Jr",
        "JEastCf",
        "JNorthCf",
        "JEastTotal",
        "JNorthTotal",
    ]
    for variable in variables:
        assert variable in input_data["DSECS_output/0/currents"]
        assert len(test_data["DSECS_output/0/currents"][variable]) == len(
            input_data["DSECS_output/0/currents"][variable]
        )

        diff = (
            test_data["DSECS_output/0/currents"][variable]
            - input_data["DSECS_output/0/currents"][variable]
        ).to_numpy()
        assert np.all((np.abs(diff) < 1e-7) | np.isnan(diff))
