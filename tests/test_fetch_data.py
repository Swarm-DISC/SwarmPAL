from __future__ import annotations

import numpy as np
import pytest
from xarray import DataTree

from swarmpal import fetch_data
from swarmpal.io import PalDataItem

from .io.test_paldata import hapi_checks, vires_checks


@pytest.mark.remote()
def test_fetch_data_vires(tmp_path):
    data_spec = dict(
        data_params=[
            dict(
                provider="vires",
                collection="SW_OPER_MAGA_LR_1B",
                measurements=["F", "B_NEC"],
                models=["IGRF"],
                start_time="2016-01-01T00:00:00",
                end_time="2016-01-01T00:00:10",
                filters=["(Longitude > 92.8) AND (Latitude < -72.57)"],
                server_url="https://vires.services/ows",
            )
        ]
    )

    item = fetch_data(data_spec)
    assert isinstance(item, DataTree)
    palitem = PalDataItem.from_manual(item["/SW_OPER_MAGA_LR_1B"].to_dataset())
    palitem.dataset_name = "SW_OPER_MAGA_LR_1B"
    vires_checks(palitem)
    # Test to make sure ViRES client data can be saved to NetCDF; had
    # problems with writing CategoricalDType variables in the past
    palitem.xarray.to_netcdf(tmp_path / "test_fetch_data_vires.nc4")


@pytest.mark.remote()
def test_fetch_data_hapi():
    data_spec = dict(
        data_params=[
            dict(
                provider="hapi",
                dataset="SW_OPER_MAGA_LR_1B",
                parameters="F,B_NEC",
                start="2016-01-01T00:00:00",
                stop="2016-01-01T00:00:10",
                server="https://vires.services/hapi",
            )
        ]
    )

    item = fetch_data(data_spec)
    assert isinstance(item, DataTree)
    dataitem = PalDataItem.from_manual(item["/SW_OPER_MAGA_LR_1B"].to_dataset())
    dataitem.dataset_name = "SW_OPER_MAGA_LR_1B"
    hapi_checks(dataitem)


@pytest.mark.remote()
def test_pad_times():
    data_spec = dict(
        data_params=[
            dict(
                provider="vires",
                collection="SW_OPER_MAGA_LR_1B",
                measurements=["F", "B_NEC"],
                start_time="2016-01-01T00:00:00",
                end_time="2016-01-01T00:00:10",
                pad_times=["0:00:03", "0:00:05"],
                server_url="https://vires.services/ows",
            )
        ]
    )
    item = fetch_data(data_spec)
    palitem = PalDataItem.from_manual(item["/SW_OPER_MAGA_LR_1B"].to_dataset())
    palitem.dataset_name = "SW_OPER_MAGA_LR_1B"

    assert "Timestamp" in palitem.xarray
    assert palitem.xarray["Timestamp"].to_numpy()[0] >= np.datetime64(
        "2015-12-31T23:59:57"
    )
    assert palitem.xarray["Timestamp"].to_numpy()[-1] <= np.datetime64(
        "2016-01-01T00:00:14"
    )
