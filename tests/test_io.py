from __future__ import annotations

from xarray import Dataset

from swarmx.io import MagData, ViresDataFetcher


def test_ViresDataFetcher():
    parameters = {
        "collection": "SW_OPER_MAGA_LR_1B",
        "measurements": ["F"],
        "models": ["IGRF"],
        "auxiliaries": ["QDLat"],
        "sampling_step": "PT1M",
    }
    v = ViresDataFetcher(parameters=parameters)
    start = "2022-01-01T00:00:00"
    end = "2022-01-01T00:10:00"
    ds = v.fetch_data(start, end, asynchronous=False, show_progress=False)
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


def test_MagData():
    d = MagData(collection="SW_OPER_MAGA_LR_1B", model="IGRF")
    start = "2022-01-01T00:00:00"
    end = "2022-01-01T00:10:00"
    d.fetch(start, end, asynchronous=False, show_progress=False)
    ds = d.xarray
    if not isinstance(ds, Dataset):
        _type = type(ds)
        raise TypeError(f"Returned data should be xarray.Dataset, not {_type}")
