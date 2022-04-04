import datetime as dt

from swarmx.toolboxes.fac import FacInputs, fac_single_sat


def test_fac_single_sat():
    start = dt.datetime.fromisoformat("2022-01-01T00:00:00")
    end = dt.datetime.fromisoformat("2022-01-01T00:01:00")
    fac_inputs = FacInputs(
        collection="SW_OPER_MAGA_LR_1B", model="IGRF",
        start_time=start, end_time=end,
        viresclient_kwargs=dict(asynchronous=False, show_progress=False)
    )
    output = fac_single_sat(fac_inputs)
    assert output["time"].shape == output["fac"].shape
    assert (end-start).seconds == (len(output["time"]) + 1)
