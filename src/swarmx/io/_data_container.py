from __future__ import annotations

from datetime import datetime
from textwrap import dedent

from numpy import array
from viresclient import SwarmRequest
from xarray import Dataset

DEFAULTS = {"VirES_server": "https://vires.services/ows"}


class DataFetcher:
    pass


class ViresDataFetcher(DataFetcher):
    """Connects to and retrieves data from VirES through viresclient

    Example usage::

        from swarmx.io import VirESDataFetcher
        # Initialise request
        v = VirESDataFetcher(
            parameters={
                'collection': 'SW_OPER_MAGA_LR_1B',
                'measurements': ['F', 'B_NEC', 'Flags_B'],
                'models': ['CHAOS'],
                'auxiliaries': ['QDLat', 'QDLon'],
                'sampling_step': None
            }
        )
        # Fetch data and extract as xarray.Dataset
        ds = v.fetch_data("2022-01-01", "2022-01-02")

    Args:
        url (str): Server URL, defaults to "https://vires.services/ows"
        parameters (dict): Parameters to pass to viresclient
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


class Data:
    pass


class MagData(Data):
    """Fetches and loads magnetic data products

    Example usage::

        from swarmx.io import MagData
        # Prepare data
        d = MagData(collection="SW_OPER_MAGA_LR_1B", model="CHAOS")
        d.fetch("2022-01-01", "2022-01-02")
        # Access data stored in memory as xarray.Dataset
        d.xarray

    The returned dataset will contain "B_NEC" and "B_NEC_Model"

    Args:
        collection (str): One of MagData.COLLECTIONS
        model (str): VirES-compatible model specification
        source (str): Defaults to "vires"
        parameters (dict):
            If supplied, overrides what is supplied to ViresDataFetcher
    """

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
    ]

    def __init__(
        self,
        collection: str | None = None,
        model: str | None = None,
        source: str = "vires",
        parameters: dict | None = None,
    ) -> None:
        if collection not in self._supported_collections():
            message = dedent(
                f"""Unsupported collection: {collection}
            Choose from {self._supported_collections()}
            """
            )
            raise ValueError(message)
        if source == "vires":
            default_parameters = self._prepare_parameters(
                collection=collection, model=model
            )
            parameters = default_parameters if parameters is None else parameters
            self.fetcher = ViresDataFetcher(parameters=parameters)
        else:
            raise NotImplementedError("Only the VirES source is configured")

    @classmethod
    def _supported_collections(cls) -> list:
        return cls.COLLECTIONS

    @staticmethod
    def _prepare_parameters(collection: str = None, model: str = None) -> dict:
        model = "CHAOS" if model is None else model
        measurements = ["F", "B_NEC", "Flags_B"]
        auxiliaries = ["QDLat", "QDLon"]
        sampling_step = None
        return {
            "collection": collection,
            "measurements": measurements,
            "models": [f"Model = {model}"],
            "auxiliaries": auxiliaries,
            "sampling_step": sampling_step,
        }

    @property
    def xarray(self) -> Dataset:
        return self._xarray

    @xarray.setter
    def xarray(self, xarray_dataset: Dataset):
        self._xarray = xarray_dataset

    def fetch(
        self, start_time: str | datetime, end_time: str | datetime, **kwargs
    ) -> Dataset:
        """Fetch data from source and store Dataset in .xarray attribute

        Args:
            start_time (str / datetime)
            end_time (str / datetime)
        """
        self.xarray = self.fetcher.fetch_data(start_time, end_time, **kwargs)
        return self.xarray

    def get_array(self, variable: str) -> array:
        """Extract numpy array from dataset"""
        ds = self.xarray
        available_vars = list(ds.dims) + list(ds.data_vars)
        if variable not in available_vars:
            raise ValueError(
                f"'{variable}' not found in dataset containing: {available_vars}"
            )
        return ds.get(variable).data
