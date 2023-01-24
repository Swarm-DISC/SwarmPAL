"""
PalData tools for containing data
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from os import PathLike
from re import match as regex_match

from datatree import DataTree
from pandas import to_datetime as to_pandas_datetime
from xarray import Dataset

from swarmpal.io.datafetchers import DataFetcherBase, get_fetcher

logging.basicConfig(level=logging.WARN)


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
    >>> # Data is available within the .xarray attribute
    >>> item.xarray
    """

    def __init__(self, fetcher: DataFetcherBase) -> None:
        self.xarray = None
        self._fetcher = fetcher

    @property
    def xarray(self) -> Dataset | None:
        """Points to the xarray.Dataset contained within this object"""
        if self._xarray is None:
            self.initialise()
        return self._xarray

    @xarray.setter
    def xarray(self, xarray_dataset: Dataset | None) -> None:
        self._xarray = xarray_dataset

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
    def _pad_times(params: dict) -> tuple[dict, tuple[datetime]]:
        """Use job parameters to adjust time window

        Returns the modified params dictionary, as well as the analysis window
        """
        # Store original times as the analysis window
        analysis_window = (params["start_time"], params["end_time"])
        analysis_window = PalDataItem._ensure_datetime(analysis_window)
        pad_times = params.pop("pad_times", None)
        if pad_times:
            # Adjust original times according to the pad values
            fetch_times = (
                analysis_window[0] - pad_times[0],
                analysis_window[1] + pad_times[1],
            )
        else:
            fetch_times = analysis_window
        # Convert times back to ISO strings
        fetch_times = PalDataItem._datetime_to_str(fetch_times)
        params["start_time"], params["end_time"] = fetch_times
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
            logging.warn("Resetting contents of PalData")
            self.datatree = DataTree(name=self.datatree.name)
        for name, item in paldataitems.items():
            # Use paldataitems to populate the DataTree; triggers download
            self.datatree[f"inputs/{name}"] = DataTree(item.xarray)
            # TODO: Validate and extract PAL meta as a property


if __name__ == "__main__":
    # Prepare parameters for fetching data
    params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["F", "B_NEC"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-02T00:00:00",
    )
    hapi_params = dict(
        **params,
        server_url="https://vires.services/hapi",
    )
    vires_params = dict(
        **params,
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
        # models=["IGRF"]
    )
    # Create PalData from two inputs, one from vires, one from hapi
    mypal1 = PalData()
    mypal1.set_inputs(
        vires_source=PalDataItem.from_vires(**vires_params),
        hapi_source=PalDataItem.from_hapi(**hapi_params),
    )
