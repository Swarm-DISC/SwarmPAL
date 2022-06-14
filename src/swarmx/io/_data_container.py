"""
Classes for holding data and interacting with the VirES Server
"""
from __future__ import annotations

from datetime import datetime
from textwrap import dedent

from numpy import ndarray
from viresclient import SwarmRequest
from xarray import Dataset

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
    >>> from swarmx.io import ViresDataFetcher
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

    def fetch_data(
        self, start_time: str | datetime, end_time: str | datetime, **kwargs
    ) -> Dataset:
        data = self.vires.get_between(start_time, end_time, **kwargs)
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
    >>> from swarmx.io import ExternalData
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
    }

    def __init__(
        self,
        collection: str,
        model: str,
        start_time: str | datetime,
        end_time: str | datetime,
        source: str = "vires",
        parameters: dict | None = None,
        viresclient_kwargs: dict | None = None,
    ) -> None:
        viresclient_kwargs = {} if viresclient_kwargs is None else viresclient_kwargs
        if collection not in self._supported_collections():
            message = dedent(
                f"""Unsupported collection: {collection}
            Choose from {self._supported_collections()}
            """
            )
            raise ValueError(message)
        # Prepare access to external data source given
        if source == "vires":
            default_parameters = self._prepare_parameters(
                collection=collection, model=model
            )
            parameters = default_parameters if parameters is None else parameters
            self.fetcher = ViresDataFetcher(parameters=parameters)
            # Fetch the data
            self.xarray = self.fetcher.fetch_data(
                start_time, end_time, **viresclient_kwargs
            )
        else:
            raise NotImplementedError("Only the VirES source is configured")

    @classmethod
    def _supported_collections(cls) -> list:
        return cls.COLLECTIONS

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
    def xarray(self) -> Dataset:
        return self._xarray

    @xarray.setter
    def xarray(self, xarray_dataset: Dataset):
        self._xarray = xarray_dataset

    def get_array(self, variable: str) -> ndarray:
        """Extract numpy array from dataset"""
        ds = self.xarray
        available_vars = list(ds.dims) + list(ds.data_vars)
        if variable not in available_vars:
            raise ValueError(
                f"'{variable}' not found in dataset containing: {available_vars}"
            )
        return ds.get(variable).data  # type: ignore

    def append_array(self, varname, data, dims=("Timestamp",)):
        """Append a new variable to the dataset

        Parameters
        ----------
        varname: str
            Name to give to the data variable
        data: ndarray
            Array of data, of same dimensions as dims
        dims: tuple, default=("Timestamp",)
            Dimension names
        """
        self.xarray = self.xarray.assign(
            {
                varname: ("Timestamp", data),
            }
        )
        return self


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
