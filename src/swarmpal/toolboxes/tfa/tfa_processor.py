"""TFA process tools

Authors
-------
constantinos@noa.gr
"""

import datetime as dt
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
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
    def __init__(self, input_data, X_varname="B_NEC", params=None):
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
        self.t = self._time_in_seconds(self.input_data.xarray["Timestamp"])
        self.X = self.input_data.xarray[X_varname].data.copy()
        self.meta = {"General": self.params}

    @staticmethod
    def _time_in_seconds(timestamps):
        """Time in seconds from beginning of data"""
        return (timestamps - timestamps[0]).dt.seconds.data

    @property
    def analysis_window(self):
        """list[datetime]: pair of unpadded times"""
        return self.input_data.analysis_window

    @property
    def analysis_window_indexes(self):
        """tuple[int]: indexes where analysis starts and ends"""
        try:
            # Use precalculated if available
            idx0, idx1 = self._idx0, self._idx1
        except AttributeError:
            # Identify indexes at the given times from analysis_window
            window_dt64 = np.array(self.analysis_window).astype("datetime64[ns]")
            timestamps = self.input_data.xarray["Timestamp"].data
            idx0 = np.argwhere(timestamps == window_dt64[0])
            idx1 = np.argwhere(timestamps == window_dt64[1])
            idx0, idx1 = idx0[0, 0], idx1[0, 0]
            # Store for next usage
            self._idx0, self._idx1 = idx0, idx1
        return idx0, idx1

    @property
    def t(self):
        """Time in seconds from t0"""
        return self._t

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def X(self):
        """Parameter being analysed"""
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    def apply(self, process):
        """Apply a TFA Process"""
        process.apply(self)

    def plot(self, data="Series"):
        if data == "Series":
            x = self.X
        elif data == "Index":
            x = self.wave_index
        elif data == "QD":
            pass

        D = self.X.shape[1]

        plt.figure()
        for i in range(D):
            plt.subplot(D, 1, i + 1)
            plt.plot(self.t, x[:, i], "-b")
            plt.xlim(self.analysis_window_indexes)
            plt.grid(True)
            # if i == 0:
            #     plt.title(self.label)

    def image(self):
        M, N, D = self.W.shape
        freqs = 1000 / self.s
        # freq_lims = [freqs[0], freqs[-1]]
        # yticks = np.hstack((np.arange(1,10), np.arange(10,200,20),
        #                    np.arange(200,1000,200), np.arange(1000,10000,1000)))
        # yticks = np.append(yticks, freq_lims)
        # yticklabels = ['%.0f'%i for i in yticks]

        plt.figure()
        for i in range(D):
            m = np.max([np.log10(np.min(self.W[::, :, i])), -6])
            x = np.log10(np.max(self.W[::, :, i]))

            plt.subplot(D, 1, i + 1)
            plt.contourf(
                self.t,
                freqs,
                np.log10(self.W[::, :, i]),
                cmap="jet",
                levels=np.linspace(m, x, 20),
                extend="min",
            )
            plt.xlim(self.analysis_window_indexes)
            # plt.yticks(ticks=yticks, labels=yticklabels)
            # plt.ylim(freq_lims)
            plt.ylabel("Freq (mHz)")
            cbh = plt.colorbar(orientation="vertical")
            cb_ticks = cbh.get_ticks()
            cbh.set_ticks(cb_ticks)
            cbh.set_ticklabels(["%.2f" % i for i in cb_ticks])
            # if i == 0:
            #     plt.title(self.label)

    def _evaluate_wave_index(self):
        N, D = self.X.shape
        _I = np.full((N, D), np.NaN)

        for i in range(D):
            _I[:, i] = np.sum(self.W[:, :, i], axis=0)

        return _I

    @property
    def wave_index(self):
        """Evaluate the wave index"""
        try:
            # Use precalculated if available
            return self._I
        except AttributeError:
            self._I = self._evaluate_wave_index()
        return self._I


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
            self.params = {"Sampling_Rate": 1, "Interp": False}
        else:
            self.params = params

    def apply(self, target):
        target.t, target.X, _ = tfalib.constant_cadence(
            target.t, target.X, self.params["Sampling_Rate"], self.params["Interp"]
        )
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
    # Apply a process
    processor.apply(Cadence)
