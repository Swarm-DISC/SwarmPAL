"""
PalData tools for containing data
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as packages_metadata
import json
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from os import PathLike
from re import match as regex_match

from cdflib.xarray import xarray_to_cdf
from pandas import to_datetime as to_pandas_datetime
from xarray import DataArray, Dataset, DataTree, register_datatree_accessor
from xarray.core.extension_array import PandasExtensionArray

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
        """Create a new datatree containing only this dataset; labelled with the dataset name."""
        return DataTree(dataset=self.xarray, name=self.dataset_name)

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
            if isinstance(x, (datetime, date)):
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
        # HOTFIX for https://github.com/ESA-VirES/VirES-Python-Client/issues/112
        # Convert all instances of xarray.core.extension_array.PandasExtentionArray to numpy.ndarray
        for var in self.xarray.variables:
            if isinstance(self.xarray[var].data, PandasExtensionArray):
                self.xarray[var].data = self.xarray[var].data.to_numpy()

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

        Parameters
        ----------
        collection : str
        measurements : list[str]
        start_time : str | datetime
        end_time : str | datetime
        models : list[str]
        auxiliaries : list[str]
        sampling_step : str
        filters : list[str]
        options : dict
        server_url : str
            defaults to "https://vires.services/ows"
        pad_times : tuple[timedelta]
            This is handled specially by SwarmPAL and not passed to viresclient
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

        Parameters
        ----------
        server : str
        dataset : str
        parameters : str
        start : str
        stop : str
        options : dict
        pad_times : tuple[timedelta]
            This is handled specially by SwarmPAL and not passed to hapiclient
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
        pdi = PalDataItem(fetcher)
        pdi.dataset_name = params.get("dataset_name", group)
        pdi.initialise()
        return pdi

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
        pdi = PalDataItem(fetcher)
        pdi.dataset_name = "manual"
        return pdi


class PalMeta:
    @staticmethod
    def serialise(meta: dict) -> str:
        def _format_handler(x):
            if isinstance(x, (datetime, date)):
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

    def to_cdf(self, file_name: str, leaf: str, istp_check: bool = False) -> None:
        """Write one leaf of the datatree to a CDF file

        Parameters
        ----------
        file_name : str
            Name of the file to create
        leaf : str
            Location within the datatree
        """
        # Identify dataset to use
        ds = self._datatree[leaf].ds.copy()
        # Adjust metadata (CDF global attrs)
        # Extra PAL_meta from the parent node
        pal_meta = self._datatree[leaf].parent.swarmpal.pal_meta["."]
        versions = f"swarmpal-{packages_metadata.version('swarmpal')} [cdflib-{packages_metadata.version('cdflib')}]"
        ds.attrs.update(
            {
                "CREATOR": versions,
                "CREATED": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "TITLE": file_name,
                "PAL_meta": PalMeta.serialise(pal_meta),
            }
        )
        # NB: cdflib will write Timestamp as type CDF_TT2000 not CDF_EPOCH
        xarray_to_cdf(xarray_dataset=ds, file_name=file_name, istp=istp_check)


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
    children = {}
    # Assign each PalDataItem.datatree as a child in the tree
    names = [pdi.dataset_name for pdi in paldataitems]
    if len(set(names)) != len(names):
        raise PalError("Duplicate dataset names found; use kwargs instead")
    for item in paldataitems:
        subtree = item.datatree
        children[item.dataset_name] = subtree
    # Assign each PalDataItem.datatree in user-specified location
    for name, item in paldataitems_kw.items():
        children[f"{name}"] = item.datatree
    # DataTree(children={'a/b': ...}) causes an infinite loop, but from_dict seems to work.
    # See https://github.com/pydata/xarray/issues/9978
    # fulltree = DataTree(name="paldata", children=children)
    fulltree = DataTree.from_dict(children)
    fulltree.name = "paldata"
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
