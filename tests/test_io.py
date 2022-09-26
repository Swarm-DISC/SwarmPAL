from __future__ import annotations

import pytest
from xarray import Dataset

from swarmx.io import ExternalData, MagExternalData, ViresDataFetcher


@pytest.mark.remote
def test_ViresDataFetcher():
    parameters = {
        "collection": "SW_OPER_MAGA_LR_1B",
        "measurements": ["F"],
        "models": ["IGRF"],
        "auxiliaries": ["QDLat"],
        "sampling_step": "PT1M",
        "start_time": "2022-01-01T00:00:00",
        "end_time": "2022-01-01T00:01:00",
        "kwargs": dict(asynchronous=False, show_progress=False)
    }
    v = ViresDataFetcher(parameters=parameters)
    ds = v.fetch_data()
    if not isinstance(ds, Dataset):
        _type = type(ds)
        raise TypeError(f"Returned data should be xarray.Dataset, not {_type}")
    expected_vars = [
        "Spacecraft",
        "Radius",
        "Latitude",
        "Longitude",
        "F",
        "F_IGRF",
        "QDLat",
    ]
    if set(expected_vars) != set(ds.data_vars):
        raise ValueError(f"Erroneous data_vars: {ds.data_vars}")


@pytest.mark.remote
def test_ExternalData():
    start = "2022-01-01T00:00:00"
    end = "2022-01-01T00:01:00"
    ExternalData.COLLECTIONS = [f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"]
    ExternalData.DEFAULTS["measurements"] = ["F", "B_NEC", "Flags_B"]
    ExternalData.DEFAULTS["model"] = "CHAOS"
    ExternalData.DEFAULTS["auxiliaries"] = ["QDLat", "QDLon", "MLT"]
    ExternalData.DEFAULTS["sampling_step"] = None
    d = ExternalData(
        collection="SW_OPER_MAGA_LR_1B",
        model="None",
        start_time=start,
        end_time=end,
        viresclient_kwargs=dict(asynchronous=False, show_progress=False),
    )
    ds = d.xarray
    if not isinstance(ds, Dataset):
        _type = type(ds)
        raise TypeError(f"Returned data should be xarray.Dataset, not {_type}")
    vars_to_check = {"B_NEC"}
    vars_to_check_nonpresence = {"B_NEC_Model"}
    returned_vars = set(ds.data_vars)
    assert vars_to_check.issubset(returned_vars)
    assert not vars_to_check_nonpresence.issubset(returned_vars)


@pytest.mark.remote
def test_MagExternalData():
    start = "2022-01-01T00:00:00"
    end = "2022-01-01T00:01:00"
    d = MagExternalData(
        collection="SW_OPER_MAGA_LR_1B",
        model="IGRF",
        start_time=start,
        end_time=end,
        viresclient_kwargs=dict(asynchronous=False, show_progress=False),
    )
    ds = d.xarray
    if not isinstance(ds, Dataset):
        _type = type(ds)
        raise TypeError(f"Returned data should be xarray.Dataset, not {_type}")
    vars_to_check = {"B_NEC", "B_NEC_Model"}
    returned_vars = set(ds.data_vars)
    assert vars_to_check.issubset(returned_vars)
