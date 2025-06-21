from __future__ import annotations

from datetime import timedelta

from xarray import DataTree

from swarmpal.io._paldata import PalDataItem, create_paldata
from swarmpal.schema import validate as _validate

processes_by_name = {}


def make_process(process_name=None, config={}):
    """Instantiates a PalProcess object with a given name and a configuration.

    Parameters
    ----------
    process_name: Str
        The name of the process to apply.
        Must be one of ['FAC_single_sat'].
    config: dict
        Parameters passed to the Toolbox' process.
    """
    if process_name not in processes_by_name:
        if process_name == "FAC_single_sat":
            from swarmpal.toolboxes.fac.processes import FAC_single_sat  # noqua(I001)

            processes_by_name["FAC_single_sat"] = FAC_single_sat

        elif process_name == "DSECS_Preprocess":
            from swarmpal.toolboxes.dsecs.processes import Preprocess

            processes_by_name["DSECS_Preprocess"] = Preprocess

        elif process_name == "DSECS_Analysis":
            from swarmpal.toolboxes.dsecs.processes import Analysis

            processes_by_name["DSECS_Analysis"] = Analysis

        elif process_name == "TFA_Preprocess":
            from swarmpal.toolboxes.tfa.processes import Preprocess as TFA_Preprocess

            processes_by_name["TFA_Preprocess"] = TFA_Preprocess

        elif process_name == "TFA_Clean":
            from swarmpal.toolboxes.tfa.processes import Clean as TFA_Clean

            processes_by_name["TFA_Clean"] = TFA_Clean

        elif process_name == "TFA_Filter":
            from swarmpal.toolboxes.tfa.processes import Filter as TFA_Filter

            processes_by_name["TFA_Filter"] = TFA_Filter

        elif process_name == "TFA_Wavelet":
            from swarmpal.toolboxes.tfa.processes import Wavelet as TFA_Wavelet

            processes_by_name["TFA_Wavelet"] = TFA_Wavelet

        elif process_name == "TFA_WaveDetection":
            from swarmpal.toolboxes.tfa.processes import (
                WaveDetection as TFA_WaveDetection,
            )

            processes_by_name["TFA_WaveDetection"] = TFA_WaveDetection

        else:
            raise ValueError(
                f"Unknown process {process_name}. Must be one of ['FAC_single_sat', 'DSECS_Preprocess', 'DSECS_Analysis']"
            )

    return processes_by_name[process_name](config=config)


def apply_process(data, process_name=None, config={}):
    """Create a SwarmPAL process and apply it on the given data.

    Parameters
    ----------
    data: DataTree
        the data on which the process will be applied to.
    process_name: Str
        the name of the process to apply. See ... for a list of Toolboxes and their Processes.
    config: dict
        parameters passed to the Toolbox.
    """
    process = make_process(process_name=process_name, config=config)
    return process(data)


def apply_processes(data, process_params):
    """Apply a list of processes to a dataset.

    Parameters
    ----------
    data: DataTree
        the data on which the process will be applied to.
    process_params:
        a list of processes to apply to the input data.
    """
    for config in process_params:
        process_name = config.pop("process_name")
        apply_process(data, process_name=process_name, config=config)
    return data


def _str_to_timedelta(time):
    """Convert strings that match 'HH:MM:SS' to datetime.timedelta objects."""
    hours, minutes, seconds = (int(part) for part in time.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _fetch_dataset(provider="", config={}, options=None):
    """Helper that downloads a single dataset from a data provider with a specified configuration.
    Parameters
    ----------
    provider: Str
        The name of the data provider. Must be one of ['vires', 'hapi'].
    config: List
        A configuration passed to `create_paldata`.
        TODO: describe the 'schema'
    options: dict or None
        The option passed to `create_paldata`. When None, the following defaults are used:

            .. list-table::
                :widths: 30 70
                :header-rows: 1

                * - Provider
                  - Default Options
                * - `vires`
                  - `dict(asynchronous=False, show_progress=False)`
                * - `hapi`
                  - `dict(logging=False)`
    """
    # Convert pad_times from strings to timedelta objects
    if "pad_times" in config:
        config["pad_times"] = [_str_to_timedelta(time) for time in config["pad_times"]]

    # Download the data
    if provider == "vires":
        options = options or dict(asynchronous=False, show_progress=False)
        return create_paldata(PalDataItem.from_vires(options=options, **config))

    if provider == "hapi":
        options = options or dict(logging=False)
        return create_paldata(PalDataItem.from_hapi(options=options, **config))

    raise ValueError(
        f"Unknown provider {provider}. Provider must be one of ['vires', 'hapi']."
    )


def fetch_data(configurations):
    """Downloads list of datasets and returns a unified DataTree.

    Parameters
    ----------
    configurations: List
        A list of configurations passed to `create_paldata`.
        TODO: describe the 'schema'
    Returns
    -------

    """

    _validate(configurations)

    data = DataTree()
    for dataset_config in configurations["data_params"]:
        provider = dataset_config.pop("provider")
        options = dataset_config.pop("options", {})
        item = _fetch_dataset(provider=provider, options=options, config=dataset_config)
        for key, dt in item.children.items():
            data[key] = dt
    return data
