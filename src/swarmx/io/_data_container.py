from textwrap import dedent
from viresclient import SwarmRequest


DEFAULTS = {
    "VirES_server": "https://vires.services/ows"
}


class DataFetcher:
    pass


class ViresDataFetcher(DataFetcher):

    def __init__(self, url=None, parameters=None):
        self.url = DEFAULTS.get("VirES_server") if url is None else url
        self.parameters = parameters
        self._initialise_request()

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        required_parameters = set(
            ['collection', 'measurements', 'auxiliaries', 'sampling_step', 'models']
        )
        if not isinstance(parameters, dict) or required_parameters != set(parameters.keys()):
            message = dedent(f"""Invalid parameters: {parameters}
            Should contain {required_parameters}""")
            raise TypeError(message)
        self._parameters = parameters

    def _initialise_request(self):
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
            sampling_step=sampling_step
        )
        return None

    def fetch_data(self, start_time, end_time, **kwargs):
        data = self.vires.get_between(start_time, end_time, **kwargs)
        ds = data.as_xarray()
        return ds


class Data:
    pass


class MagData(Data):

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"]
    ]

    def __init__(self, collection=None, model=None, source="vires", parameters=None):
        if collection not in self._supported_collections:
            message = dedent(f"""Unsupported collection: {collection}
            Choose from {self._supported_collections}
            """)
            raise ValueError(message)
        if source == "vires":
            default_parameters = self._prepare_parameters(collection=collection, model=model)
            parameters = default_parameters if parameters is None else parameters
            self.fetcher = ViresDataFetcher(parameters=parameters)
        else:
            raise NotImplementedError("Only the VirES source is configured")

    @classmethod
    @property
    def _supported_collections(cls):
        return cls.COLLECTIONS

    @staticmethod
    def _prepare_parameters(collection=None, model=None):
        collection = "SW_OPER_MAGA_LR_1B" if collection is None else collection
        model = "CHAOS" if model is None else model
        measurements = ["F", "B_NEC", "Flags_B"]
        auxiliaries = ["QDLat", "QDLon"]
        sampling_step = None
        return {
            "collection": collection,
            "measurements": measurements,
            "models": [model],
            "auxiliaries": auxiliaries,
            "sampling_step": sampling_step
        }

    @property
    def xarray(self):
        return self._xarray

    @xarray.setter
    def xarray(self, xarray_dataset):
        self._xarray = xarray_dataset

    def fetch(self, start_time, end_time, **kwargs):
        self.xarray = self.fetcher.fetch_data(start_time, end_time, **kwargs)
        return self.xarray
