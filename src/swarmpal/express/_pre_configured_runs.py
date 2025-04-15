from __future__ import annotations

from xarray import DataTree

from swarmpal.io import PalDataItem, create_paldata
from swarmpal.toolboxes import fac
from swarmpal.utils.configs import SPACECRAFT_TO_MAGLR_DATASET
from swarmpal.utils.exceptions import PalError


def fac_single_sat(
    spacecraft: str,
    time_start: str,
    time_end: str,
    grade: str = "OPER",
    to_cdf_file: str | None = None,
) -> DataTree:
    """Execute FAC single-satellite processor

    Parameters
    ----------
    spacecraft : str
        Spacecraft to use
    time_start : str
        Starting time of the analysis
    time_end : str
        Ending time of the analysis
    grade : str, optional
        OPER or FAST processing chain, by default "OPER""
    to_cdf_file : str | None, optional
        Path for CDF file to generate, by default None

    Returns
    -------
    DataTree

    Raises
    ------
    PalError
        When inputs are invalid
    """
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
    if spacecraft in ["Swarm-A", "Swarm-B", "Swarm-C"]:
        measurements = ["B_NEC", "Flags_F", "Flags_B", "Flags_q"]
    else:
        measurements = ["B_NEC"]
    data_params = dict(
        collection=input_dataset,
        measurements=measurements,
        models=["CHAOS"],
        start_time=time_start,
        end_time=time_end,
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    data = create_paldata(PalDataItem.from_vires(**data_params))
    time_jump_limit = 1 if "Swarm" in spacecraft else 10
    process = fac.processes.FAC_single_sat(
        config={
            "dataset": input_dataset,
            "model_varname": "B_NEC_CHAOS",
            "measurement_varname": "B_NEC",
            "time_jump_limit": time_jump_limit,
        },
    )
    data = process(data)
    if to_cdf_file:
        data.swarmpal.to_cdf(file_name=to_cdf_file, leaf="PAL_FAC_single_sat")
    return data
