from __future__ import annotations

from datetime import datetime, timedelta
from os.path import join as join_path

import numpy as np
import pytest
from pandas import to_datetime as to_pandas_datetime
from xarray import Dataset, DataTree, open_dataset

from swarmpal.io._paldata import PalDataItem, create_paldata

from ..test_data import load_test_dataset, get_local_filename


def vires_checks(item):
    # Type checks
    assert isinstance(item.xarray, Dataset)
    assert isinstance(item.datatree, DataTree)
    assert isinstance(item.dataset_name, str)
    # Sanity checks
    # Collection
    assert item.dataset_name == "SW_OPER_MAGA_LR_1B"
    # Do we have all the expected variables?
    assert len(item.xarray.keys()) == 8
    assert "B_NEC_IGRF" in item.xarray
    assert "Timestamp" in item.xarray
    assert "Longitude" in item.xarray
    assert "Latitude" in item.xarray
    assert "Radius" in item.xarray
    assert "B_NEC" in item.xarray
    assert "NEC" in item.xarray
    assert "F" in item.xarray
    # Time interval matches [start,end]_time
    assert item.xarray["Timestamp"].to_numpy()[0] >= np.datetime64(
        "2016-01-01T00:00:00"
    )
    assert item.xarray["Timestamp"].to_numpy()[-1] <= np.datetime64(
        "2016-01-01T00:00:10"
    )
    # Every entry in 'Spacecraft' should be 'A'
    assert np.all(np.unique(item.xarray["Spacecraft"]) == ["A"])


def hapi_checks(item):
    # Type checks
    assert isinstance(item.xarray, Dataset)
    assert isinstance(item.datatree, DataTree)
    assert isinstance(item.dataset_name, str)
    # Sanity checks
    # Collection
    assert item.dataset_name == "SW_OPER_MAGA_LR_1B"
    # Do we have all the expected variables?
    assert len(item.xarray.keys()) == 2
    assert "B_NEC" in item.xarray
    assert "F" in item.xarray
    # Time interval matches [start,end]_time
    assert item.xarray["Timestamp"].to_numpy()[0] >= np.datetime64(
        "2016-01-01T00:00:00"
    )
    assert item.xarray["Timestamp"].to_numpy()[-1] <= np.datetime64(
        "2016-01-01T00:00:10"
    )


@pytest.mark.cached()
def test_paldataitem_hapi():
    item = load_test_dataset("test_paldataitem_hapi.nc4", group="SW_OPER_MAGA_LR_1B")
    hapi_checks(item)


@pytest.mark.cached()
def test_paldataitem_file():
    dataset_filename = get_local_filename("test_paldataitem_vires.nc4")
    item = PalDataItem.from_file(dataset_filename, group="SW_OPER_MAGA_LR_1B")
    vires_checks(item)
    assert isinstance(item.xarray, Dataset)


@pytest.mark.cached()
def test_paldataitem_manual():
    dataset_filename = get_local_filename("test_paldataitem_vires.nc4")
    ds = open_dataset(dataset_filename, group="SW_OPER_MAGA_LR_1B")
    item = PalDataItem.from_manual(ds)
    item.dataset_name = "SW_OPER_MAGA_LR_1B"
    vires_checks(item)
    assert isinstance(item.xarray, Dataset)


@pytest.mark.cached()
def test_time_pad_vires():
    start_time = datetime(2016, 1, 1, 0, 0, 0)
    end_time = datetime(2016, 1, 1, 0, 0, 10)
    dt0 = timedelta(seconds=3)
    dt1 = timedelta(seconds=5)
    pdi = load_test_dataset("test_time_pad_vires.nc4", group="SW_OPER_MAGA_LR_1B")
    # Check that the start and end times of data are offset correctly
    t0, t1 = to_pandas_datetime(pdi.xarray["Timestamp"][[0, -1]])
    assert t0 == start_time - dt0
    assert t1 == end_time + dt1 - timedelta(seconds=1)
    # # Check that analysis window matches the given start, end times
    # add this when finalised. NB. must deserialise the PAL_meta
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][0] == start_time.isoformat()
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][1] == end_time.isoformat()


@pytest.mark.cached()
def test_time_pad_hapi():
    start_time = datetime(2016, 1, 1, 0, 0, 0)
    end_time = datetime(2016, 1, 1, 0, 0, 10)
    dt0 = timedelta(seconds=3)
    dt1 = timedelta(seconds=5)
    pdi = load_test_dataset("test_time_pad_hapi.nc4", group="SW_OPER_MAGA_LR_1B")
    # Check that the start and end times of data are offset correctly
    t0, t1 = to_pandas_datetime(pdi.xarray["Timestamp"][[0, -1]])
    assert t0 == start_time - dt0
    assert t1 == end_time + dt1 - timedelta(seconds=1)
    # # Check that analysis window matches the given start, end times
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][0] == start_time.isoformat()
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][1] == end_time.isoformat()


@pytest.mark.cached()
@pytest.fixture()
def paldata_item_MAGA():
    return load_test_dataset(
        "fixture_paldata_item_MAGA.nc4", group="SW_OPER_MAGA_LR_1B"
    )


@pytest.mark.cached()
@pytest.fixture()
def paldata_item_MAGB():
    return load_test_dataset(
        "fixture_paldata_item_MAGB.nc4", group="SW_OPER_MAGB_LR_1B"
    )


@pytest.mark.cached()
def test_create_paldata(paldata_item_MAGA, paldata_item_MAGB):
    # Test basic paldata
    data = create_paldata(paldata_item_MAGA)
    assert isinstance(data, DataTree)
    assert data.name == "paldata"
    assert "SW_OPER_MAGA_LR_1B" in data.children
    assert data.swarmpal.pal_meta["SW_OPER_MAGA_LR_1B"]
    assert isinstance(paldata_item_MAGA.datatree, DataTree)
    # assert not paldata_item_MAGA.datatree == data
    # Test paldata with two entries
    data = create_paldata(paldata_item_MAGA, paldata_item_MAGB)
    assert data.name == "paldata"
    assert "SW_OPER_MAGA_LR_1B" in data.children
    assert "SW_OPER_MAGB_LR_1B" in data.children
    assert data.swarmpal.pal_meta["SW_OPER_MAGA_LR_1B"]
    assert data.swarmpal.pal_meta["SW_OPER_MAGB_LR_1B"]
    # Test paldata with manually defined entries
    data = create_paldata(
        **{
            "path1/alpha": paldata_item_MAGA,
            "path2/bravo": paldata_item_MAGB,
        }
    )
    assert data.name == "paldata"
    assert data["path1/alpha"]
    assert data["path2/bravo"]
    # assert isinstance(data["path1/alpha"], DataTree)
    # assert data["path1/alpha"].name == "SW_OPER_MAGA_LR_1B"
    # assert isinstance(data["path2/bravo"], DataTree)
    assert data.swarmpal.pal_meta["path1/alpha"]
    assert data.swarmpal.pal_meta["path2/bravo"]
