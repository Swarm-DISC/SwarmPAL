"""
Tools to connect to the outside world and get/create xarray Datasets
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from os.path import exists as path_exists

from hapiclient import hapi, hapitime2datetime
from numpy.typing import ArrayLike
from pandas import to_datetime as to_pandas_datetime
from viresclient import SwarmRequest
from xarray import Dataset, open_dataset


@dataclass
class Parameters:
    """Control which dataset is accessed, and how the fetcher behaves"""

    ...


@dataclass
class ViresParameters(Parameters):
    collection: str
    measurements: list[str]
    start_time: str
    end_time: str
    server_url: str = "https://vires.services/ows"
    models: list[str] = field(default_factory=list)
    auxiliaries: list[str] = field(default_factory=list)
    sampling_step: str | None = None
    filters: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)


@dataclass
class HapiParameters(Parameters):
    collection: str
    measurements: list[str]
    start_time: str
    end_time: str
    server_url: str = "https://vires.services/hapi"
    options: dict = field(default_factory=dict)


@dataclass
class FileParameters(Parameters):
    filename: PathLike
    group: str | None = None


@dataclass
class ManualParameters(Parameters):
    ...


class DataFetcherBase(ABC):
    """Interface with an external data source"""

    @property
    @abstractmethod
    def source(self) -> str:
        """String to identify the data source type (e.g. 'vires', 'hapi')"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """Set of parameters to control how/what data is accessed"""
        ...

    @abstractmethod
    def fetch_data(self) -> Dataset:
        """Command to get data as an xarray Dataset"""
        ...


class ViresDataFetcher(DataFetcherBase):
    """Connects to and retrieves data from VirES through viresclient"""

    @property
    def source(self) -> str:
        return "vires"

    @property
    def parameters(self) -> ViresParameters:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        self._parameters = ViresParameters(**parameters)

    def __init__(self, **parameters) -> None:
        """Create connection to VirES and initialise the request

        Available parameters are listed in ViresParameters

        Next run .fetch_data() to trigger the request
        """
        self.parameters = parameters
        self.vires_request = self._initialise_request()

    def _initialise_request(self) -> SwarmRequest:
        """Use the set parameters to initialise the request"""
        vires_request = SwarmRequest(self.parameters.server_url)
        vires_request.set_collection(self.parameters.collection)
        vires_request.set_products(
            measurements=self.parameters.measurements,
            models=self.parameters.models,
            auxiliaries=self.parameters.auxiliaries,
            sampling_step=self.parameters.sampling_step,
        )
        for filter in self.parameters.filters:
            vires_request.add_filter(filter)
        return vires_request

    def fetch_data(self) -> Dataset:
        """Process the request on VirES and load an xarray Dataset"""
        return self.vires_request.get_between(
            self.parameters.start_time,
            self.parameters.end_time,
            **self.parameters.options,
        ).as_xarray()


class HapiDataFetcher(DataFetcherBase):
    """Connects to and retrieves data from a HAPI server through hapiclient"""

    @property
    def source(self) -> str:
        return "hapi"

    @property
    def parameters(self) -> HapiParameters:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        self._parameters = HapiParameters(**parameters)

    def __init__(self, **parameters) -> None:
        """Prepare inputs for hapi & test connection

        Available parameters are listed in HapiParameters

        Next run .fetch_data() to trigger the request
        """
        self.parameters = parameters
        self._get_hapi_info()

    @staticmethod
    def _hapi_to_xarray(data: ArrayLike, meta: dict) -> Dataset:
        # Separate the time variable name from the other varnames
        #  (assuming time variable comes first in the list)
        varnames = [p["name"] for p in meta["parameters"]]
        timevar, varnames = varnames[0], varnames[1:]
        # Generate dimension labels for each parameter
        dims = ()
        for p in meta["parameters"][1:]:
            n_extra_dims = len(p.get("size", []))
            extra_dims = (f"{p['name']}_dim{i+1}" for i in range(n_extra_dims))
            dims = (*dims, ("Timestamp", *extra_dims))
        # Convert time data to timezone-naive DatetimeIndex
        tdata = to_pandas_datetime(hapitime2datetime(data[timevar]))
        tdata = tdata.tz_convert("UTC").tz_convert(None)
        # Assuming we now have ordered lists of varnames, dims,
        # assemble a Dataset from the data & meta
        ds = Dataset(
            data_vars={
                timevar: (timevar, tdata),
                **{_name: (_dim, data[_name]) for _name, _dim, in zip(varnames, dims)},
            }
        )
        # Assign metadata for each data variable
        for p in meta["parameters"][1:]:
            ds[p["name"]].attrs = {
                "units": p.get("units"),
                "description": p.get("description"),
            }
        return ds

    def _get_hapi_info(self) -> dict:
        # Get info response from HAPI server
        return hapi(
            self.parameters.server_url,
            self.parameters.collection,
            ",".join(self.parameters.measurements),
            **self.parameters.options,
        )

    def fetch_data(self) -> Dataset:
        """Make a HAPI query and load an xarray Dataset"""
        data, meta = hapi(
            self.parameters.server_url,
            self.parameters.collection,
            ",".join(self.parameters.measurements),
            self.parameters.start_time,
            self.parameters.end_time,
            **self.parameters.options,
        )
        return self._hapi_to_xarray(data, meta)


class FileDataFetcher(DataFetcherBase):
    @property
    def source(self) -> str:
        return "file"

    @property
    def parameters(self) -> FileParameters:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        self._parameters = FileParameters(**parameters)

    def __init__(self, filename: PathLike, group: str | None = None) -> None:
        self.parameters = dict(filename=filename, group=group)
        if not path_exists(self.parameters.filename):
            raise FileNotFoundError(self.parameters.filename)

    def fetch_data(self) -> Dataset:
        kwargs = {"filename_or_obj": self.parameters.filename}
        if self.parameters.group:
            kwargs["group"] = self.parameters.group
        return open_dataset(**kwargs)


class ManualDataFetcher(DataFetcherBase):
    @property
    def source(self) -> str:
        return "manual"

    @property
    def parameters(self) -> FileParameters:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        self._parameters = ManualParameters(**parameters)

    def __init__(self, xarray_dataset: Dataset) -> None:
        self.parameters = dict()
        if not isinstance(xarray_dataset, Dataset):
            raise ValueError("Given data must be xarray.Dataset")
        self._xarray = xarray_dataset.copy()

    def fetch_data(self) -> Dataset:
        return self._xarray


def get_fetcher(source) -> DataFetcherBase:
    fetchers = {
        "vires": ViresDataFetcher,
        "hapi": HapiDataFetcher,
        "file": FileDataFetcher,
        "manual": ManualDataFetcher,
    }
    try:
        return fetchers[source]
    except KeyError:
        raise KeyError(f"Data source '{source}' not found")


if __name__ == "__main__":
    params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-02T00:00:10",
    )
    hapi_params = dict(
        **params,
        server_url="https://vires.services/hapi",
    )
    vires_params = dict(
        **params,
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
        models=["IGRF"],
    )
    vires_data = get_fetcher("vires")(**vires_params).fetch_data()
    hapi_data = get_fetcher("hapi")(**hapi_params).fetch_data()
