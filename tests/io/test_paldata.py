from datetime import datetime, timedelta
from os.path import join as join_path

import pytest
from datatree import DataTree
from pandas import to_datetime as to_pandas_datetime
from xarray import Dataset, open_dataset

from swarmpal.io.paldata import PalDataItem, create_paldata


@pytest.mark.remote
def test_paldataitem_vires():
    params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:00:10",
    )
    vires_params = dict(
        **params,
        models=["IGRF"],
        filters=["(Longitude > 92.8) AND (Latitude < -72.57)"],
        options=dict(asynchronous=False, show_progress=False),
    )
    item = PalDataItem.from_vires(**vires_params)
    item.initialise()
    assert isinstance(item.xarray, Dataset)
    assert isinstance(item.datatree, DataTree)
    return item.xarray


@pytest.mark.remote
@pytest.fixture
def xarray_data_file(tmp_path):
    ds = test_paldataitem_vires()
    target_output = join_path(tmp_path, "test_data.nc")
    ds.to_netcdf(target_output)
    return target_output


@pytest.mark.remote
def test_paldataitem_hapi():
    params = dict(
        dataset="SW_OPER_MAGA_LR_1B",
        parameters="F,B_NEC",
        start="2016-01-01T00:00:00",
        stop="2016-01-01T00:00:10",
    )
    hapi_params = dict(
        **params,
        server="https://vires.services/hapi",
        options=dict(logging=True),
    )
    item = PalDataItem.from_hapi(**hapi_params)
    item.initialise()
    assert isinstance(item.xarray, Dataset)
    assert isinstance(item.datatree, DataTree)


def test_paldataitem_file(xarray_data_file):
    item = PalDataItem.from_file(xarray_data_file)
    assert isinstance(item.xarray, Dataset)


def test_paldataitem_manual(xarray_data_file):
    ds = open_dataset(xarray_data_file)
    item = PalDataItem.from_manual(ds)
    assert isinstance(item.xarray, Dataset)


@pytest.mark.remote
def test_time_pad_vires():
    start_time = datetime(2016, 1, 1, 0, 0, 0)
    end_time = datetime(2016, 1, 1, 0, 0, 10)
    dt0 = timedelta(seconds=3)
    dt1 = timedelta(seconds=5)
    vires_params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time=start_time,
        end_time=end_time,
        pad_times=[dt0, dt1],
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    pdi = PalDataItem.from_vires(**vires_params)
    # Check that the start and end times of data are offset correctly
    t0, t1 = to_pandas_datetime(pdi.xarray["Timestamp"][[0, -1]])
    assert t0 == start_time - dt0
    assert t1 == end_time + dt1 - timedelta(seconds=1)
    # # Check that analysis window matches the given start, end times
    # add this when finalised. NB. must deserialise the PAL_meta
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][0] == start_time.isoformat()
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][1] == end_time.isoformat()


@pytest.mark.remote
def test_time_pad_hapi():
    start_time = datetime(2016, 1, 1, 0, 0, 0)
    end_time = datetime(2016, 1, 1, 0, 0, 10)
    dt0 = timedelta(seconds=3)
    dt1 = timedelta(seconds=5)
    hapi_params = dict(
        dataset="SW_OPER_MAGA_LR_1B",
        parameters="F,B_NEC",
        start=start_time.isoformat(),
        stop=end_time.isoformat(),
        pad_times=[dt0, dt1],
        server="https://vires.services/hapi",
    )
    pdi = PalDataItem.from_hapi(**hapi_params)
    # Check that the start and end times of data are offset correctly
    t0, t1 = to_pandas_datetime(pdi.xarray["Timestamp"][[0, -1]])
    assert t0 == start_time - dt0
    assert t1 == end_time + dt1 - timedelta(seconds=1)
    # # Check that analysis window matches the given start, end times
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][0] == start_time.isoformat()
    # assert pdi.xarray.attrs["PAL_meta"]["analysis_window"][1] == end_time.isoformat()


@pytest.mark.remote
@pytest.fixture
def paldata_item_MAGA():
    data_params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["B_NEC"],
        models=["IGRF"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:01:00",
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    pdi = PalDataItem.from_vires(**data_params)
    pdi.initialise()
    return pdi


@pytest.mark.remote
@pytest.fixture
def paldata_item_MAGB():
    data_params = dict(
        collection="SW_OPER_MAGB_LR_1B",
        measurements=["B_NEC"],
        models=["IGRF"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:01:00",
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    pdi = PalDataItem.from_vires(**data_params)
    pdi.initialise()
    return pdi


@pytest.mark.remote
def test_create_paldata(paldata_item_MAGA, paldata_item_MAGB):
    # Test basic paldata
    data = create_paldata(paldata_item_MAGA)
    assert data.name == "paldata"
    assert "SW_OPER_MAGA_LR_1B" in data.children
    assert data.swarmpal.pal_meta["SW_OPER_MAGA_LR_1B"]
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
    assert data.swarmpal.pal_meta["path1/alpha"]
    assert data.swarmpal.pal_meta["path2/bravo"]
