from __future__ import annotations

import pytest

from swarmpal.io import PalDataItem, create_paldata
from swarmpal.toolboxes import fac


@pytest.mark.remote()
def test_fac_singlesat_swarm():
    """Basic test that the FAC process is applied"""
    data_params_fac_input_swarm = {
        "collection": "SW_OPER_MAGA_LR_1B",
        "measurements": ["B_NEC"],
        "models": ["CHAOS"],
        "start_time": "2016-01-01T00:00:00",
        "end_time": "2016-01-01T00:30:00",
        "server_url": "https://vires.services/ows",
        "options": {
            "asynchronous": False,
            "show_progress": False,
        },
    }
    data = create_paldata(PalDataItem.from_vires(**data_params_fac_input_swarm))
    process_config = {
        "dataset": "SW_OPER_MAGA_LR_1B",
        "model_varname": "B_NEC_CHAOS",
        "measurement_varname": "B_NEC",
        "inclination_limit": 30,
        "time_jump_limit": 6,
    }
    process = fac.processes.FAC_single_sat(config=process_config)
    data = process(data)
    assert "IRC" in data["PAL_FAC_single_sat"]
    assert "FAC" in data["PAL_FAC_single_sat"]
