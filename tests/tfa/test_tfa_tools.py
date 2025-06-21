from __future__ import annotations

import numpy as np
import pytest
from xarray import DataTree

import swarmpal.toolboxes
from swarmpal import apply_processes
from swarmpal.toolboxes import tfa

from ..test_data import load_test_config, load_test_datatree


def test_by_name():
    """FAC processes were added to the toolboxes lookup dictionary"""
    tfa_processes = {
        "TFA_Preprocess": tfa.processes.Preprocess,
        "TFA_Clean": tfa.processes.Clean,
        "TFA_Filter": tfa.processes.Filter,
        "TFA_Wavelet": tfa.processes.Wavelet,
        "TFA_WaveDetection": tfa.processes.WaveDetection,
    }
    for process_name, process in tfa_processes.items():
        new_process = swarmpal.make_process(process_name)
        assert isinstance(new_process, process)


@pytest.mark.cached()
def test_tfa_basic():
    product_name = "SW_OPER_MAGA_LR_1B"
    input_data = DataTree.from_dict(
        {
            "SW_OPER_MAGA_LR_1B": load_test_datatree(
                "test_tfa_basic.nc4", group=product_name
            ),
        }
    )

    assert product_name in input_data
    assert "PAL_meta" not in input_data.attrs

    dataset_meta = load_test_config("test_tfa_basic")
    input_data = apply_processes(input_data, dataset_meta["process_params"])

    assert "PAL_meta" in input_data.attrs
    assert "TFA_Time" in input_data[product_name]

    test_data = load_test_datatree("test_tfa_basic.nc4")

    variables = [
        "TFA_Variable",
        "wavelet_power",
        "scale",
        "B_NEC_res_Model",
    ]

    for variable in variables:
        assert variable in input_data[product_name]
        assert len(test_data[product_name][variable]) == len(
            input_data[product_name][variable]
        )

        diff = (
            test_data[product_name][variable] - input_data[product_name][variable]
        ).to_numpy()

        assert np.all((np.abs(diff) < 1e-10) | np.isnan(diff))
