from __future__ import annotations

import numpy as np
import pytest

import swarmpal.toolboxes
from swarmpal.io import create_paldata
from swarmpal.toolboxes import fac

from ..test_data import load_test_config, load_test_dataset


def test_by_name():
    """FAC_single_sat was added to the toolboxes lookup dictionary"""
    process = swarmpal.make_process("FAC_single_sat")
    assert isinstance(process, fac.processes.FAC_single_sat)


@pytest.mark.cached()
def test_fac_singlesat_swarm():
    """Basic test that the FAC process is applied"""
    input_dataset = "SW_OPER_MAGA_LR_1B"
    input_data = create_paldata(
        load_test_dataset("test_fac_singlesat_swarm.nc4", group=input_dataset)
    )
    assert input_dataset in input_data

    dataset_meta = load_test_config("test_fac_singlesat_swarm")
    process_config = dataset_meta["toolboxes"]["FAC_single_sat"]
    process = fac.processes.FAC_single_sat(config=process_config)

    data = process(input_data)
    assert input_dataset in data

    variables = ["IRC", "FAC"]
    for variable in variables:
        assert variable in data["PAL_FAC_single_sat"]
        assert len(data["PAL_FAC_single_sat"][variable]) == len(
            input_data["PAL_FAC_single_sat"][variable]
        )

        # The difference between test data and the recent calculation should be small or NaN
        diffs = (
            data["PAL_FAC_single_sat"][variable]
            - input_data["PAL_FAC_single_sat"][variable]
        ).to_numpy()
        assert np.all((diffs < 1e-10) | np.isnan(diffs))
