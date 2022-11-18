from os.path import join as join_path

import pytest
from xarray import Dataset, open_dataset

from swarmpal.io.paldata import PalData, PalDataItem


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


@pytest.mark.remote
def test_paldataitem_hapi():
    params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:00:10",
    )
    hapi_params = dict(
        **params,
        server_url="https://vires.services/hapi",
        options=dict(logging=True),
    )
    item = PalDataItem.from_hapi(**hapi_params)
    item.initialise()
    assert isinstance(item.xarray, Dataset)
    return item.xarray


@pytest.mark.remote
@pytest.fixture
def xarray_data_file(tmp_path):
    ds = test_paldataitem_hapi()
    target_output = join_path(tmp_path, "test_data.nc")
    ds.to_netcdf(target_output)
    return target_output


def test_paldataitem_file(xarray_data_file):
    item = PalDataItem.from_file(xarray_data_file)
    assert isinstance(item.xarray, Dataset)


def test_paldataitem_manual(xarray_data_file):
    ds = open_dataset(xarray_data_file)
    item = PalDataItem.from_manual(ds)
    assert isinstance(item.xarray, Dataset)


def test_paldata(xarray_data_file):
    mypal = PalData(
        PalDataItem.from_file(xarray_data_file),
        PalDataItem.from_file(xarray_data_file),
    )
    mypal.initialise()
    assert isinstance(mypal[0].xarray, Dataset)
    assert isinstance(mypal[1].xarray, Dataset)
    mypal = PalData(
        one=PalDataItem.from_file(xarray_data_file),
        two=PalDataItem.from_file(xarray_data_file),
    )
    assert isinstance(mypal["one"].xarray, Dataset)
    assert isinstance(mypal["two"].xarray, Dataset)
