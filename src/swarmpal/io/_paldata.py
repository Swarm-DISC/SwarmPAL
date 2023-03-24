"""
PalData tools for containing data
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from os import PathLike
from re import match as regex_match

from datatree import DataTree, register_datatree_accessor
from pandas import to_datetime as to_pandas_datetime
from xarray import DataArray, Dataset

from swarmpal.io._datafetchers import DataFetcherBase, get_fetcher
from swarmpal.utils.exceptions import PalError

logger = logging.getLogger(__name__)


class PalDataItem:
    """An Item (i.e. dataset) that will be stored within a PalData object

    Intended use is through the `.from_...` creator tools

    Parameters
    ----------
    fetcher: DataFetcherBase

    Examples
    --------
    >>> from swarmpal.io import PalDataItem
    >>>
    >>> # Create the item with a configuration
    >>> params = dict(
    >>>     collection="SW_OPER_MAGA_LR_1B",
    >>>     measurements=["F", "B_NEC"],
    >>>     start_time="2016-01-01T00:00:00",
    >>>     end_time="2016-01-02T00:00:00",
    >>>     server_url="https://vires.services/ows",
    >>> )
    >>> item = PalDataItem.from_vires(**params)
    >>> # "Initialise" - triggers the expensive part, fetching the data
    >>> item.initialise()
    >>> # Data is available as an xarray.Dataset
    >>> item.xarray
    >>> # or as a DataTree
    >>> item.datatree
    """

    def __init__(self, fetcher: DataFetcherBase) -> None:
        self.xarray = None
        self._fetcher = fetcher

    @property
    def dataset_name(self) -> str:
        """Name of the dataset, used as the datatree label"""
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

    @property
    def xarray(self) -> Dataset | None:
        """xarray.Dataset containing the data, generated if not already present"""
        if self._xarray is None:
            self.initialise()
        return self._xarray

    @xarray.setter
    def xarray(self, xarray_dataset: Dataset | None) -> None:
        self._xarray = xarray_dataset

    @property
    def datatree(self) -> DataTree:
        """A datatree containing the dataset labelled with the dataset name"""
        return DataTree(data=self.xarray, name=self.dataset_name)

    @property
    def analysis_window(self) -> tuple[datetime]:
        """The start and end times of the analysis window (considering optional padding"""
        try:
            return self._analysis_window
        except AttributeError:
            # Use the dataset time bounds if analysis_window hasn't been set
            t = to_pandas_datetime(self.xarray["Timestamp"][[0, -1]].data)
            t = t.to_pydatetime()
            self.analysis_window = tuple(t)
        return self._analysis_window

    @analysis_window.setter
    def analysis_window(self, analysis_window: tuple[datetime]):
        self._analysis_window = analysis_window

    @property
    def magnetic_models(self) -> dict:
        """Dictionary of model names and details"""
        # Get xarray attribute data and then parse it
        #  input looks like: ['Model = IGRF(max_degree=13,min_degree=1)']
        mlist = self.xarray.attrs.get("MagneticModels", [])
        # xarray will write a single-item list[str] as str, so catch that
        mlist = [mlist] if isinstance(mlist, str) else mlist
        models = {}
        for model_string in mlist:
            # Gives, e.g. ('Model', 'IGRF(max_degree=13,min_degree=1)')
            name, detail = regex_match(r"(.*) = (.*)", model_string).groups()
            models[name] = detail
        return models

    def _serialise_pal_metadata(self):
        def _format_handler(x):
            if isinstance(x, datetime):
                return x.isoformat()
            raise TypeError("Unknown type")

        meta = {
            "analysis_window": self.analysis_window,
            "magnetic_models": self.magnetic_models,
        }
        return json.dumps(meta, default=_format_handler)

    def initialise(self):
        """Trigger the fetching of the data and attach PAL metadata"""
        self.xarray = self._fetcher.fetch_data()
        self.xarray.attrs["PAL_meta"] = self._serialise_pal_metadata()

    @staticmethod
    def _ensure_datetime(times: tuple[datetime | str]) -> tuple[datetime]:
        """Converts times to datetimes if they are not already"""
        if isinstance(times[0], str):
            times = tuple(map(datetime.fromisoformat, times))
        return times

    @staticmethod
    def _datetime_to_str(times: tuple[datetime]) -> tuple[str]:
        """Convert datetime objects to iso strings"""
        return tuple(map(lambda x: x.isoformat(), times))

    @staticmethod
    def _get_start_end_times(params: dict) -> tuple[dict, tuple[datetime]]:
        """Get start and end times from the job parameters

        Accounts for difference between VirES ("start_time", "end_time"),
        and HAPI ("start", "stop")
        """
        if "start_time" in params.keys():
            start = params["start_time"]
            end = params["end_time"]
        elif "start" in params.keys():
            start = params["start"]
            end = params["stop"]
        return PalDataItem._ensure_datetime((start, end))

    @staticmethod
    def _update_start_end_times(params: dict, start: str, end: str):
        """Update the job parameters with new (start, end) times"""
        if "start_time" in params.keys():
            params["start_time"] = start
            params["end_time"] = end
        elif "start" in params.keys():
            params["start"] = start
            params["stop"] = end
        return params

    @staticmethod
    def _pad_times(params: dict) -> tuple[dict, tuple[datetime]]:
        """Use job parameters to adjust time window

        Returns the modified params dictionary, as well as the analysis window
        """
        # Store original times as the analysis window
        analysis_window = PalDataItem._get_start_end_times(params)
        pad_times = params.pop("pad_times", None)
        if pad_times:
            # Adjust original times according to the pad values
            times_to_fetch = (
                analysis_window[0] - pad_times[0],
                analysis_window[1] + pad_times[1],
            )
        else:
            times_to_fetch = analysis_window
        # Convert times back to ISO strings and use to update the job parameters
        times_to_fetch = PalDataItem._datetime_to_str(times_to_fetch)
        params = PalDataItem._update_start_end_times(params, *times_to_fetch)
        return (params, analysis_window)

    @staticmethod
    def from_vires(**params) -> PalDataItem:
        """Create PalDataItem from VirES source

        TODO: Detail params given by swarmpal.io.datafetchers.ViresParameters
        """
        params, analysis_window = PalDataItem._pad_times(params)
        fetcher = get_fetcher("vires")(**params)
        pdi = PalDataItem(fetcher)
        pdi.analysis_window = analysis_window
        pdi.dataset_name = params.get("collection")
        return pdi

    @staticmethod
    def from_hapi(**params) -> PalDataItem:
        """Create PalDataItem from HAPI source

        TODO: Detail params given by swarmpal.io.datafetchers.HapiParameters
        """
        params, analysis_window = PalDataItem._pad_times(params)
        fetcher = get_fetcher("hapi")(**params)
        pdi = PalDataItem(fetcher)
        pdi.analysis_window = analysis_window
        pdi.dataset_name = params.get("dataset")
        return pdi

    @staticmethod
    def from_file(
        filename: PathLike | None = None, group: str | None = None, **params
    ) -> PalDataItem:
        """Create a PalDataItem from a file

        Parameters
        ----------
        filename : PathLike
            Path to the (netCDF) file to load
        group : str
            Group name within the (netCDF) file

        Returns
        -------
        PalDataItem
        """
        if filename:
            params["filename"] = filename
        if group:
            params["group"] = group
        fetcher = get_fetcher("file")(**params)
        return PalDataItem(fetcher)

    @staticmethod
    def from_manual(xarray_dataset: Dataset | None = None, **params) -> PalDataItem:
        """Create a PalDataItem manually from an existing xarray Dataset

        Parameters
        ----------
        xarray_dataset : Dataset
            An existing xarray.Dataset

        Returns
        -------
        PalDataItem
        """
        if xarray_dataset:
            params["xarray_dataset"] = xarray_dataset
        fetcher = get_fetcher("manual")(**params)
        return PalDataItem(fetcher)


class PalData:
    """Holds an Xarray DataTree of PAL inputs and outputs

    - PalData can be created from PalDataItem objects passed to .set_inputs.
    - PalData can be created from existing datatree (e.g. stored on disk)
        See https://xarray-datatree.readthedocs.io/en/latest/io.html
    - The data is stored within the .datatree property.

    Parameters
    ----------
    name: str
        Provide a name to label the datatree
    datatree: DataTree
        Provide an existing DataTree directly instead

    Examples
    --------
    >>> from swarmpal.io import PalData, PalDataItem
    >>>
    >>> params = dict(
    >>>     collection="SW_OPER_MAGA_LR_1B",
    >>>     measurements=["F", "B_NEC"],
    >>>     start_time="2016-01-01T00:00:00",
    >>>     end_time="2016-01-02T00:00:00",
    >>>     server_url="https://vires.services/ows",
    >>> )
    >>> my_pal = PalData(name="trial")
    >>> my_pal.set_inputs(
    >>>     SwarmAlpha=PalDataItem.from_vires(**params)
    >>> )
    >>> # Access data directly through the datatree
    >>> my_pal.datatree["inputs/SwarmAlpha"]
    """

    def __init__(self, name: str | None = None, datatree: DataTree | None = None):
        self.datatree = datatree if datatree else DataTree(name=name)

    def set_inputs(self, **paldataitems: PalDataItem):
        """Supply named input datasets through PalDataItem

        Parameters
        ----------
        **paldataitems_kw: PalDataItem
            Provide instances of PalDataItem as keyword arguments
        """
        if self.datatree:
            logger.warn(" Resetting contents of PalData")
            self.datatree = DataTree(name=self.datatree.name)
        for name, item in paldataitems.items():
            # Use paldataitems to populate the DataTree; triggers download
            self.datatree[f"inputs/{name}"] = DataTree(item.xarray)
            # TODO: Validate and extract PAL meta as a property


class PalMeta:
    @staticmethod
    def serialise(meta: dict) -> str:
        def _format_handler(x):
            if isinstance(x, datetime):
                return x.isoformat()
            raise TypeError("Unknown type")

        return json.dumps(meta, default=_format_handler)

    @staticmethod
    def deserialise(meta: str) -> dict:
        return json.loads(meta)


@register_datatree_accessor("swarmpal")
class PalDataTreeAccessor:
    """Provide custom attributes and methods on XArray DataTrees for SwarmPAL functionality.

    See e.g. https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py
    """

    def __init__(self, datatree) -> None:
        self._datatree = datatree

    def apply(self, palprocess: PalProcess) -> DataTree:
        return palprocess(self._datatree)

    @property
    def pal_meta(self) -> dict:
        pal_metadata_set = {}
        for datatree in (self._datatree, *self._datatree.descendants):
            pal_meta = datatree.attrs.get("PAL_meta", "{}")
            pal_meta = PalMeta.deserialise(pal_meta)
            treepath = datatree.relative_to(self._datatree)
            pal_metadata_set[treepath] = pal_meta
        return pal_metadata_set

    @property
    def magnetic_model_names(self) -> list[str]:
        """List of the model names used in the dataset"""
        magnetic_model_names = set()
        for _, pal_meta in self.pal_meta.items():
            models = pal_meta.get("magnetic_models", {})
            for model_name, _ in models.items():
                magnetic_model_names.add(model_name)
        return list(magnetic_model_names)

    @property
    def magnetic_model_name(self) -> str:
        """Model name if one and only one has been set"""
        models = self.magnetic_model_names
        if len(models) == 0:
            raise PalError("No models identified")
        if len(models) > 1:
            raise PalError("More than one model available")
        return models[0]

    def magnetic_residual(self, model: str = "") -> DataArray:
        """Magnetic data-model residual in NEC frame"""
        if not self._datatree.is_leaf:
            raise PalError("This is not a leaf node")
        if not model:
            model = self.magnetic_model_name
        try:
            B_NEC = self._datatree["B_NEC"]
            B_NEC_mod = self._datatree[f"B_NEC_{model}"]
        except KeyError:
            raise PalError(f"One of B_NEC or B_NEC_{model} is not available")
        residual = B_NEC - B_NEC_mod
        residual.attrs = {
            "units": "nT",
            "description": "Magnetic field vector data-model residual, NEC frame",
        }
        return residual


def create_paldata(
    *paldataitems: PalDataItem, **paldataitems_kw: PalDataItem
) -> DataTree:
    """Generates a Datatree from a number of PalDataItems

    Returns
    -------
    Datatree
        A Datatree containing Datasets defined from each PalDataItem

    Examples
    --------
    >>> from swarmpal.io import create_paldata, PalDataItem
    >>>
    >>> # Parameters to control a particular data request
    >>> data_params = dict(
    >>>     collection="SW_OPER_MAGA_LR_1B",
    >>>     measurements=["B_NEC"],
    >>>     models=["IGRF"],
    >>>     start_time="2016-01-01T00:00:00",
    >>>     end_time="2016-01-01T03:00:00",
    >>>     server_url="https://vires.services/ows",
    >>>     options=dict(asynchronous=False, show_progress=False),
    >>> )
    >>> # Create the datatree from a list of items
    >>> data = create_paldata(
    >>>     PalDataItem.from_vires(**data_params)
    >>> )
    >>> # Create the datatree from labelled items
    >>> data = create_paldata(
    >>>     one=PalDataItem.from_vires(**data_params),
    >>>     two=PalDataItem.from_vires(**data_params),
    >>> )
    """
    # Assign each PalDataItem.datatree as a child in the tree
    fulltree = DataTree(name="paldata")
    names = [pdi.dataset_name for pdi in paldataitems]
    if len(set(names)) != len(names):
        raise PalError("Duplicate dataset names found; use kwargs instead")
    for item in paldataitems:
        subtree = item.datatree
        subtree.parent = fulltree
    # Assign each PalDataItem.datatree in user-specified location
    for name, item in paldataitems_kw.items():
        fulltree[f"{name}"] = item.datatree
    return fulltree


class PalProcess(ABC):
    """Abstract class to define processes to act on datatrees"""

    def __init__(
        self, config: dict | None = None, active_tree: str = "/", inplace: bool = True
    ):
        self._active_tree = active_tree
        config = config if config else {}
        self.set_config(**config)
        if not inplace:
            raise NotImplementedError(
                "Haven't figured this out yet - is it possible to do without a deep copy?"
            )

    @property
    @abstractmethod
    def process_name(self) -> str:
        return "PalProcess"

    @abstractmethod
    def set_config(self, **kwargs) -> None:
        self.config = dict(**kwargs)

    @property
    def active_tree(self) -> str:
        """Defines which branch of the datatree will be used"""
        return self._active_tree

    @property
    def config(self) -> dict:
        """Dictionary that configures the process behaviour"""
        return self._config

    @config.setter
    def config(self, config: dict) -> None:
        self._config = config

    def __call__(self, datatree) -> DataTree:
        """Run the process, defined in _call, to update the datatree"""
        # Select the active branch to work on and detach it
        if self.active_tree != "/":
            subtree = datatree[self.active_tree].copy()[self.active_tree]
            datatree[self.active_tree].orphan()
            subtree.orphan()
        else:
            subtree = datatree
        # Check metadata to see if this has already been run
        procname = self.process_name
        subtree_root_pal_meta = subtree.swarmpal.pal_meta["."]
        if procname in subtree_root_pal_meta.keys():
            logger.warn(f" Rerunning {procname}: May overwrite existing data")
        # Apply process to create updated datatree
        subtree = self._call(subtree)
        # Update metadata with details of the applied process
        subtree_root_pal_meta[procname] = self.config
        subtree.attrs["PAL_meta"] = PalMeta.serialise(subtree_root_pal_meta)
        # Update the full tree with the modified subtree
        if self.active_tree != "/":
            subtree.parent = datatree
        return datatree

    @abstractmethod
    def _call(self, datatree) -> DataTree:
        ...
