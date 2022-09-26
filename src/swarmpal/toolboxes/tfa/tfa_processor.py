"""TODO Docstring for TFA
"""

import datetime as dt
from abc import ABC, abstractmethod

import numpy as np

from swarmpal.io import ExternalData
from swarmpal.toolboxes.tfa import tfalib


class TfaMagInputs(ExternalData):
    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["B_NEC", "F"],
        "model": "'CHAOS-Core' + 'CHAOS-Static'",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
        "pad_times": [dt.timedelta(hours=3), dt.timedelta(hours=3)],
    }


class TfaEfiInputs(ExternalData):
    COLLECTIONS = [
        *[f"SW_OPER_EFI{x}_LP_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": [],
        "model": "'CHAOS-Core' + 'CHAOS-Static'",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }


class TfaProcessor:
    def __init__(self, input_data, params=None):
        self.input_data = input_data
        self.params = (
            params
            if params
            else {
                "freq_lims": [0.020, 0.100],
                "lat_lims": None,
                "maglat_lims": [-60, 60],
            }
        )

    @property
    def analysis_window(self):
        """list[datetime]: pair of unpadded times"""
        return self.input_data.analysis_window

    def apply(self, process):
        """Apply a TFA Process"""
        pass

    def plot(self):
        pass

    def image(self):
        pass

    def wave_index(self):
        pass


class TFA_Process(ABC):
    params = None

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def apply(self, target):
        return

    # append the parameters to the "meta" dictionary of the target TFA_Data object
    def append_params(self, target):
        target.meta[self.__name__] = self.params


class Cadence(TFA_Process):
    __name__ = "Cadence"

    def __init__(self, params=None):
        if params is None:
            self.params = {"Sampling_Rate": 86400, "Interp": False}
        else:
            self.params = params

    def apply(self, target):
        target.t, target.X = tfalib.constant_cadence(
            target.t, target.X, self.params["Sampling_Rate"], self.params["Interp"]
        )[0:2]
        self.append_params(target)

        return target


class Cleaning(TFA_Process):
    __name__ = "Cleaning"

    def __init__(self, params=None):
        if params is None:
            self.params = {"Window_Size": 50, "Method": "iqr", "Multiplier": 6}
        else:
            self.params = params

    def apply(self, target):
        inds = tfalib.outliers(
            target.X,
            self.params["Window_Size"],
            method=self.params["Method"],
            multiplier=self.params["Multiplier"],
        )
        target.X[inds] = np.NaN

        # interpolate cleaned values and pre-existing gaps
        N, D = target.X.shape
        t_ind = np.arange(N)
        for i in range(D):
            x = np.reshape(target.X[:, i], (N,))
            nonNaN = ~np.isnan(x)
            y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
            target.X[:, i] = y

        self.append_params(target)

        return target


class Filtering(TFA_Process):
    __name__ = "Filtering"

    def __init__(self, params=None):
        if params is None:
            self.params = {"Sampling_Rate": 1, "Cutoff": 20 / 1000}
        else:
            self.params = params

    def apply(self, target):
        target.X = tfalib.filter(
            target.X, self.params["Sampling_Rate"], self.params["Cutoff"]
        )
        self.append_params(target)

        return target


class Wavelet(TFA_Process):
    __name__ = "Wavelet"

    def __init__(self, params=None):
        if params is None:
            self.params = {
                "Time_Step": 1,
                "Min_Scale": 1000 / 100,
                "Max_Scale": 1000 / 20,
                "dj": 0.1,
            }
        else:
            self.params = params

        self.params["Wavelet_Function"] = "Morlet"
        self.params["Wavelet_Param"] = 6.2036
        self.params["Wavelet_Norm_Factor"] = 0.74044116

    def apply(self, target):

        N, D = target.X.shape
        target.s = tfalib.wavelet_scales(
            self.params["Min_Scale"], self.params["Max_Scale"], self.params["dj"]
        )
        M = len(target.s)
        target.W = np.full((M, N, D), np.NaN)

        for i in range(D):
            wave = tfalib.wavelet_transform(
                np.reshape(target.X[:, i], (N,)),
                dx=self.params["Time_Step"],
                minScale=self.params["Min_Scale"],
                maxScale=self.params["Max_Scale"],
                dj=self.params["dj"],
            )[0]
            norm = tfalib.wavelet_normalize(
                np.abs(wave) ** 2,
                target.s,
                dx=self.params["Time_Step"],
                dj=self.params["dj"],
                wavelet_norm_factor=0.74044116,
            )
            target.W[:, :, i] = norm

        self.append_params(target)

        return target


if __name__ == "__main__":
    # Initialise access to inputs
    inputs = TfaMagInputs(
        collection="SW_OPER_MAGA_LR_1B",
        model="IGRF",
        start_time=dt.datetime(2015, 6, 23, 0, 0, 0),
        end_time=dt.datetime(2015, 6, 23, 5, 0, 0),
        viresclient_kwargs={"asynchronous": False, "show_progress": False},
    )
    # Initialise processor
    processor = TfaProcessor(inputs)
