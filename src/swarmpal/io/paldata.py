"""
PalData tools for containing data
"""
from __future__ import annotations

from datetime import datetime
from os import PathLike

from pandas import to_datetime as to_pandas_datetime
from xarray import Dataset

from swarmpal.io.datafetchers import DataFetcherBase, get_fetcher


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

    def initialise(self):
        """Trigger the fetching of the data"""
        self.xarray = self._fetcher.fetch_data()

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
    """Holds multiple PalDataItem objects and connects to toolboxes

    PalData is created given PalDataItem objects a *args or **kwargs
    Each item is then accessible like `PalData[0]` or, e.g., `PalData["item_0"]`

    Parameters
    ----------
    *paldataitems: PalDataItem
        Provide instances of PalDataItem as the arguments
    **kwargs: PalDataItem
        Provide instances of PalDataItem as keyword arguments

    Examples
    --------
    >>> from swarmpal.io import PalData, PalDataItem
    >>>
    >>> mypal = PalData(
    >>>     PalDataItem.from_file("file_1.nc"),
    >>>     PalDataItem.from_file("file_2.nc"),
    >>> )
    >>> mypal.initialise()
    >>> # The two datasets are available via:
    >>> mypal[0].xarray, mypal[1].xarray
    >>>
    >>> mypal = PalData(
    >>>     item_1=PalDataItem.from_file("file_1.nc"),
    >>>     item_2=PalDataItem.from_file("file_2.nc"),
    >>> )
    >>> mypal.initialise()
    >>> # The two datasets are available via:
    >>> mypal["item_1"].xarray, mypal["item_2"].xarray
    """

    def __init__(self, *paldataitems: PalDataItem, **paldataitems_kw: PalDataItem):
        self._registry = {}
        if paldataitems and paldataitems_kw:
            raise ValueError("Do not supply both args and kwargs")
        for item in (*paldataitems, *paldataitems_kw.values()):
            if not isinstance(item, PalDataItem):
                raise TypeError(f"{item} should be a PalDataItem")
        for key, item in enumerate(paldataitems):
            self._registry[key] = item
        for key, item in paldataitems_kw.items():
            self._registry[key] = item

    def __getitem__(self, x):
        try:
            return self._registry[x]
        except KeyError:
            raise KeyError(f"Item '{x}' not found in PalData")

    def __iter__(self):
        yield from self._registry.values()

    def initialise(self):
        """Trigger the fetching of the data"""
        for paldataitem in self._registry.values():
            paldataitem.initialise()


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
        kwargs=dict(asynchronous=False, show_progress=False),
        # models=["IGRF"]
    )
    # Create PalData from two inputs, one from vires, one from hapi
    mypal1 = PalData(
        PalDataItem.from_vires(**vires_params),
        PalDataItem.from_hapi(**hapi_params),
    )
    mypal1.initialise()
