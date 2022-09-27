"""
Classes for holding data and interacting with the VirES Server
"""
from __future__ import annotations

from datetime import datetime
from textwrap import dedent

from numpy import ndarray
from viresclient import SwarmRequest
from xarray import Dataset, open_dataset

__all__ = ("ViresDataFetcher", "ExternalData", "MagExternalData")

DEFAULTS = {"VirES_server": "https://vires.services/ows"}


class ViresDataFetcher:
    """Connects to and retrieves data from VirES through viresclient

    Parameters
    ----------
    url : str
        Server URL, defaults to "https://vires.services/ows"
    parameters : dict
        Parameters to pass to viresclient

    Examples
    --------
    >>> from swarmpal.io import ViresDataFetcher
    >>> # Initialise request
    >>> v = ViresDataFetcher(
    >>>     parameters={
    >>>         'collection': 'SW_OPER_MAGA_LR_1B',
    >>>         'measurements': ['F', 'B_NEC', 'Flags_B'],
    >>>         'models': ['CHAOS'],
    >>>         'auxiliaries': ['QDLat', 'QDLon'],
    >>>         'sampling_step': None
    >>>     }
    >>> )
    >>> # Fetch data and extract as xarray.Dataset
    >>> ds = v.fetch_data("2022-01-01", "2022-01-02")
    """

    VIRES_URL = "https://vires.services/ows"

    def __init__(self, url: str | None = None, parameters: dict | None = None) -> None:
        self.url = self._url() if url is None else url
        if parameters is None:
            raise TypeError("Must supply parameters")
        elif isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError(f"Invalid parameters: {parameters}")
        self._initialise_request()
        return None

    @classmethod
    def _url(cls):
        return cls.VIRES_URL

    @property
    def parameters(self) -> dict:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        required_parameters = {
            "collection",
            "measurements",
            "auxiliaries",
            "sampling_step",
            "models",
            "start_time",
            "end_time",
            "kwargs",
        }
        if not isinstance(parameters, dict) or required_parameters != set(
            parameters.keys()
        ):
            message = dedent(
                f"""Invalid parameters: {parameters}
            Should contain {required_parameters}"""
            )
            raise TypeError(message)
        self._parameters = parameters

    def _initialise_request(self) -> None:
        collection = self.parameters.get("collection")
        measurements = self.parameters.get("measurements")
        auxiliaries = self.parameters.get("auxiliaries")
        sampling_step = self.parameters.get("sampling_step")
        models = self.parameters.get("models")
        self.vires = SwarmRequest(self.url)
        self.vires.set_collection(collection)
        self.vires.set_products(
            measurements=measurements,
            models=models,
            auxiliaries=auxiliaries,
            sampling_step=sampling_step,
        )

    def fetch_data(self) -> Dataset:
        data = self.vires.get_between(
            self.parameters.get("start_time"),
            self.parameters.get("end_time"),
            **self.parameters.get("kwargs"),
        )
        ds = data.as_xarray()
        return ds


class ExternalData:
    """Fetches and loads data from external sources, e.g. VirES

    Parameters
    ----------
    collection : str
        One of ExternalData.COLLECTIONS
    model : str
        VirES-compatible model specification. Defaults to "CHAOS" (i.e. full CHAOS model)
    start_time : str | datetime
    end_time : str | datetime
    pad_times: list[datetime.timedelta]
        Extend the requested time window by these two amounts
    source : str
        Defaults to "vires" (only one possible currently)
    parameters : dict
        Override the parameters passed to ViresDataFetcher
    viresclient_kwargs: dict
        Pass extra kwargs to viresclient

    Notes
    -----
    The model variable in the returned data will be renamed to "Model" rather than, e.g., "CHAOS"

    Examples
    --------
    >>> from swarmpal.io import ExternalData
    >>> # Customise the class (if not using a subclass)
    >>> ExternalData.COLLECTIONS = [f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"]
    >>> ExternalData.DEFAULTS["measurements"] = ["F", "B_NEC", "Flags_B"]
    >>> ExternalData.DEFAULTS["model"] = "CHAOS"
    >>> ExternalData.DEFAULTS["auxiliaries"] = ["QDLat", "QDLon", "MLT"]
    >>> ExternalData.DEFAULTS["sampling_step"] = None
    >>> # Request data
    >>> d = ExternalData(
    >>>     collection="SW_OPER_MAGA_LR_1B", model="None",
    >>>     start_time="2022-01-01", end_time="2022-01-02",
    >>>     viresclient_kwargs=dict(asynchronous=True, show_progress=True)
    >>> )
    >>> # Access data stored in memory as xarray.Dataset
    >>> d.xarray
    # The returned dataset will contain "B_NEC" and "B_NEC_Model"
    """

    # To be overwritten in subclasses
    COLLECTIONS: list[str] = []
    DEFAULTS: dict = {
        "measurements": list(),
        "model": "",
        "auxiliaries": list(),
        "sampling_step": None,
        "pad_times": None,
    }

    def __init__(
        self,
        source: str = "vires",
        collection: str | None = None,
        model: str | None = None,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
        pad_times: list[datetime.timedelta] | None = None,
        parameters: dict | None = None,
        viresclient_kwargs: dict | None = None,
        initialise: bool = True,
    ) -> None:
        viresclient_kwargs = {} if viresclient_kwargs is None else viresclient_kwargs
        # Convert to datetimes so that we can use timedelta given by pad_times
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
            end_time = datetime.fromisoformat(end_time)
        # Store the unpadded time window
        self.analysis_window = [start_time, end_time]
        # Preferentially use the currently set pad_times, else use the default
        pad_times = pad_times if pad_times else self._default_pad_times()
        # Extend the requested time period according to pad_times
        if pad_times:
            start_time = start_time - pad_times[0]
            end_time = end_time + pad_times[1]
        # Initialise the properties
        self.xarray = None
        self.source = source
        self.magnetic_model_name = model
        # Prepare access to external data source given
        if source in ("manual", "swarmpal_file"):
            pass
        elif source == "vires":
            # Validate some inputs
            if collection not in self._supported_collections():
                message = dedent(
                    f"""Unsupported collection: {collection}
                Choose from {self._supported_collections()}
                """
                )
                raise ValueError(message)
            # Prepare the VirES Data Fetcher
            default_parameters = self._prepare_parameters(
                collection=collection, model=model
            )
            parameters = default_parameters if parameters is None else parameters
            parameters["start_time"] = start_time
            parameters["end_time"] = end_time
            parameters["kwargs"] = viresclient_kwargs
            self.fetcher = ViresDataFetcher(parameters=parameters)
            if initialise:
                self.initialise()

    @classmethod
    def _supported_collections(cls) -> list:
        return cls.COLLECTIONS

    @classmethod
    def _default_pad_times(cls) -> list[datetime.timedelta] | None:
        return cls.DEFAULTS.get("pad_times", None)

    @classmethod
    def _prepare_parameters(cls, collection: str = None, model: str = None) -> dict:
        """Return parameters compatible with ViresDataFetcher"""
        model = cls.DEFAULTS["model"] if model is None else model
        model_list = None if model == "None" else [f"Model = {model}"]
        return {
            "collection": collection,
            "measurements": cls.DEFAULTS["measurements"],
            "models": model_list,
            "auxiliaries": cls.DEFAULTS["auxiliaries"],
            "sampling_step": cls.DEFAULTS["sampling_step"],
        }

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, source):
        allowed_sources = ("manual", "swarmpal_file", "vires")
        if source in allowed_sources:
            self._source = source
        else:
            raise ValueError(
                f"Invalid source '{source}', must be one of: {allowed_sources}"
            )

    @property
    def analysis_window(self) -> list[datetime]:
        return self._analysis_window

    @analysis_window.setter
    def analysis_window(self, time_pair: list[datetime]):
        self._analysis_window = time_pair

    @property
    def xarray(self) -> Dataset:
        if self._xarray:
            return self._xarray
        else:
            raise AttributeError("xarray not set. Run .initialise() to fetch the data")

    @xarray.setter
    def xarray(self, xarray_dataset: Dataset | None):
        self._xarray = xarray_dataset

    @property
    def magnetic_model_name(self):
        return self._magnetic_model_name

    @magnetic_model_name.setter
    def magnetic_model_name(self, name):
        self._magnetic_model_name = name

    def initialise(self, xarray_or_file: Dataset | str | None = None):
        """Load the data

        Parameters
        ----------
        xarray_or_file : Dataset | str | None, optional
            Optionally supply an xarray.Dataset or a file name, by default None
        """
        if xarray_or_file:
            if isinstance(xarray_or_file, Dataset):
                self.xarray = xarray_or_file.copy()
            else:
                self.xarray = open_dataset(xarray_or_file)
        else:
            # Fetch the data
            self.xarray = self.fetcher.fetch_data()

    def get_array(self, variable: str) -> ndarray:
        """Extract numpy array from dataset"""
        ds = self.xarray
        available_vars = list(ds.dims) + list(ds.data_vars)
        if variable not in available_vars:
            raise ValueError(
                f"'{variable}' not found in dataset containing: {available_vars}"
            )
        return ds.get(variable).data  # type: ignore

    def append_array(
        self, varname, data, dims=("Timestamp",), units=None, description=None
    ):
        """Append a new variable to the dataset

        Parameters
        ----------
        varname: str
            Name to give to the data variable
        data: ndarray
            Array of data, of same dimensions as dims
        dims: tuple, default=("Timestamp",)
            Dimension names
        units: str
            Units to attach to the data
        description: str
            Description to attach to the data
        """
        self.xarray = self.xarray.assign(
            {
                varname: ("Timestamp", data),
            }
        )
        if units:
            self.xarray[varname].attrs["units"] = units
        if description:
            self.xarray[varname].attrs["description"] = description
        return self

    def to_file(self, filepath):
        """Save the current data to a file"""
        self.xarray.to_netcdf(filepath)


class MagExternalData(ExternalData):
    """Demo class for accessing magnetic data

    Examples
    --------
    >>> d = MagExternalData(
    >>>     collection="SW_OPER_MAGA_LR_1B", model="IGRF",
    >>>     start_time="2022-01-01", end_time="2022-01-02",
    >>>     viresclient_kwargs=dict(asynchronous=True, show_progress=True)
    >>> )
    >>> d.xarray  # Returns xarray of data
    >>> d.get_array("B_NEC")  # Returns numpy array
    """

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["F", "B_NEC", "Flags_B"],
        "model": "IGRF",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }
