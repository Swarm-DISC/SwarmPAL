import pytest
from xarray import Dataset

from swarmpal.io._datafetchers import HapiDataFetcher, ViresDataFetcher, get_fetcher


def test_get_fetcher():
    get_fetcher("vires")
    get_fetcher("hapi")
    get_fetcher("file")
    get_fetcher("manual")


@pytest.mark.remote
def test_vires_fetcher():
    params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:00:10",
    )
    vires_params = dict(
        **params,
        server_url="https://vires.services/ows",
        models=["IGRF"],
        filters=["(Longitude > 92.8) AND (Latitude < -72.57)"],
        options=dict(asynchronous=False, show_progress=False),
    )
    vires_fetcher = ViresDataFetcher(**vires_params)
    vires_dataset = vires_fetcher.fetch_data()
    assert isinstance(vires_dataset, Dataset)


@pytest.mark.remote
def test_hapi_fetcher():
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
    # TODO: find source of warning (which is filtered out here)
    # PytestUnraisableExceptionWarning: Exception ignored in:
    # unclosed <ssl.SSLSocket:ResourceWarning
    hapi_fetcher = HapiDataFetcher(**hapi_params)
    hapi_dataset = hapi_fetcher.fetch_data()
    assert isinstance(hapi_dataset, Dataset)
