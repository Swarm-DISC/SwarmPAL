from __future__ import annotations

import numpy as np
import pytest
from xarray import DataTree

import swarmpal.toolboxes
from swarmpal import apply_processes
from swarmpal.toolboxes import tfa

from ..test_data import load_test_config, load_test_datatree


def test_tfa_by_name():
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
        assert new_process.process_name == process_name


@pytest.mark.cached
def test_tfa_basic():
    """Test a basic application of the TFA toolbox to a Swarm data product.

    In the test dataset the TFA toolbox output is written to the /PAL_TFA group.
    The unit test will rerun the analysis, place the new results in /PAL_TFA_TEST and
    compare to the original results.
    """
    product_name = "SW_OPER_MAGA_LR_1B"
    original_output_group = "PAL_TFA"
    test_output_group = "PAL_TFA_TEST"
    data = DataTree.from_dict(
        {
            product_name: load_test_datatree("test_tfa_basic.nc4", group=product_name),
            original_output_group: load_test_datatree(
                "test_tfa_basic.nc4", group=original_output_group
            ),
        }
    )

    assert product_name in data
    assert "PAL_meta" not in data.attrs

    dataset_meta = load_test_config("test_tfa_basic")
    for process in dataset_meta["process_params"]:
        process["output_dataset"] = test_output_group

    print(dataset_meta["process_params"][0])
    data = apply_processes(data, dataset_meta["process_params"])

    # Test pal_meta
    assert "PAL_meta" in data.attrs
    assert test_output_group in data.swarmpal.pal_meta["."]["output_datasets"]
    assert "PAL_meta" in data[test_output_group].attrs

    # Reload the config, because make process calls .pop on 'process_name'
    dataset_meta = load_test_config("test_tfa_basic")
    for process in dataset_meta["process_params"]:
        process_name = process["process_name"]
        assert process_name in data.swarmpal.pal_meta[test_output_group]
        assert process_name in data[test_output_group].swarmpal.pal_meta["."]

    assert "TFA_Time" in data[test_output_group]

    variables = [
        "TFA_Variable",
        "wavelet_power",
        "scale",
    ]

    # Test accuracy to machine precision
    for variable in variables:
        assert variable in data[test_output_group]
        assert len(data[test_output_group][variable]) == len(
            data[original_output_group][variable]
        )

        diff = (
            data[test_output_group][variable] - data[original_output_group][variable]
        ).to_numpy()

        assert np.all((np.abs(diff) < 1e-10) | np.isnan(diff))
