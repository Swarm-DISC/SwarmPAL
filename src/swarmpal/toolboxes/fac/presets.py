from __future__ import annotations

from datatree import DataTree

from swarmpal.io import PalDataItem, create_paldata
from swarmpal.toolboxes import fac
from swarmpal.utils.configs import SPACECRAFT_TO_MAGLR_DATASET
from swarmpal.utils.exceptions import PalError


def fac_single_sat(
    spacecraft: str,
    grade: str,
    time_start: str,
    time_end: str,
    output: str,
) -> DataTree:
    """Execute FAC single-satellite processor"""
    try:
        input_dataset = SPACECRAFT_TO_MAGLR_DATASET[spacecraft]
    except KeyError:
        raise PalError(
            f"Invalid 'spacecraft'. Must be one of {SPACECRAFT_TO_MAGLR_DATASET.keys()}"
        )
    if grade == "OPER":
        pass
    elif grade == "FAST":
        input_dataset = input_dataset.replace("OPER", "FAST")
    else:
        raise PalError("Invalid 'grade'. Must be one of 'OPER', 'FAST'")
    # Fetch data and apply process
    data_params = dict(
        collection=input_dataset,
        measurements=["B_NEC", "Flags_F", "Flags_B", "Flags_q"],
        models=["CHAOS"],
        start_time=time_start,
        end_time=time_end,
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    data = create_paldata(PalDataItem.from_vires(**data_params))
    process = fac.processes.FAC_single_sat(
        config={
            "dataset": input_dataset,
            "model_varname": "B_NEC_CHAOS",
            "measurement_varname": "B_NEC",
        },
    )
    data = process(data)
    if output:
        data.swarmpal.to_cdf(file_name=output, leaf="PAL_FAC_single_sat")
    return data
