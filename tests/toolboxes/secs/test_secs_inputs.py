import datetime as dt

import pytest

try:
    from swarmx.toolboxes.secs import SecsInputs, SecsInputSingleSat
except ImportError:
    pass


@pytest.mark.dsecs
@pytest.mark.remote
def test_SecsInputSingleSat():
    start = dt.datetime.fromisoformat("2022-01-01T00:00:00")
    end = dt.datetime.fromisoformat("2022-01-01T00:01:00")
    inputs = SecsInputSingleSat(
        collection="SW_OPER_MAGA_LR_1B",
        model="IGRF",
        start_time=start,
        end_time=end,
        viresclient_kwargs=dict(asynchronous=False, show_progress=False),
    )
    assert len(inputs.xarray["Timestamp"]) == 60
    assert "B_NEC" in inputs.xarray.data_vars
    assert "B_NEC_Model" in inputs.xarray.data_vars


@pytest.mark.dsecs
@pytest.mark.remote
def test_SecsInputs():
    start = dt.datetime.fromisoformat("2022-01-01T00:00:00")
    end = dt.datetime.fromisoformat("2022-01-01T00:01:00")
    inputs = SecsInputs(
        model="IGRF",
        start_time=start,
        end_time=end,
        viresclient_kwargs=dict(asynchronous=False, show_progress=False),
    )
    for s in (inputs.s1, inputs.s2):
        assert len(s.xarray["Timestamp"]) == 60
        assert "B_NEC" in s.xarray.data_vars
        assert "B_NEC_Model" in s.xarray.data_vars
        assert "ApexLatitude" in s.xarray.data_vars
        assert "ApexLongitude" in s.xarray.data_vars
