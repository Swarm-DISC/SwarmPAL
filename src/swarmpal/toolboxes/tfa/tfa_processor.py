"""TODO Docstring for TFA
"""


import datetime as dt
import logging
import re
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from swarmpal.io import ExternalData
from swarmpal.toolboxes.tfa import tfalib


class TfaInput(ExternalData):
    """
    This extends the ExternalData class, so it uses the same inputs, plus a
    few more to taylor the data to the ones that are being used by the TFA
    tool.

    Parameters
    ----------
    collection : str
        One of ExternalData.COLLECTIONS
    start_time : datetime
        The starting time for the analysis
    end_time : datetime
        The ending time for the analysis
    initialise : bool
        Set to True to download the requested data. If False, the TfaInput
        onject will be initialized, but without any actual data inside
    varname : str
        The name of the variable to be analysed, e.g. "F" for the magnetic
        field magnitude by the ASM instrument, or "B_NEC" for the vector
        field from the VFM instrument, etc.
    sampling_time : str
        String that specifies the time between two consecutive measurements
        to be given according to the ISO 8601 standard for time intervals
        e.g. PT1S for one second resolution, PT0.02S for 0.02 seconds,
        PT1M for one minute resolution etc
    remove_chaos : bool
        Set to True to allow a subsequent call of subtract_chaos() to
        remove the CHAOS-Core and CHAOS-Static field model from the data.
        It only applies to magnetic field variables.
    """

    # default values for the class parameters
    TIME_LIMS = []
    COLLECTION = "SW_OPER_MAGB_LR_1B"
    VARNAME = "B_NEC"
    SAMPLING_TIME = 1
    FLAGNAME = "Flags_B"
    MAX_FLAG_VAL = 30
    REMOVE_CHAOS = True

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
        *[f"SW_OPER_EFI{x}_LP_1B" for x in "ABC"],
        *[f"SW_OPER_IBI{x}TMS_2F" for x in "ABC"],
        *[f"SW_OPER_FAC{x}TMS_2F" for x in "ABC"],
        *[f"SW_EXPT_EFI{x}_TCT02" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["B_NEC", "Flags_B"],
        "model": "None",  # "'CHAOS-Core' + 'CHAOS-Static'" or "None"
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": "PT1S",
        "pad_times": [dt.timedelta(hours=3), dt.timedelta(hours=3)],
    }

    def flag_cleaning(self, varname=None, flagname=None, max_val=None):
        """Set flagged values to NaN"""
        # if no inputs are given, default to the class variables
        if varname is None:
            varname = self.VARNAME
        if flagname is None:
            flagname = self.FLAGNAME
        if max_val is None:
            max_val = self.MAX_FLAG_VAL

        inds_to_remove = self.xarray[flagname] > max_val
        if ("NEC" in self.VARNAME) or ("XYZ" in self.VARNAME):
            # 3-D case
            self.xarray[varname][inds_to_remove, :] = np.NaN
        else:
            # 1-D case
            self.xarray[varname][inds_to_remove] = np.NaN

    def calculate_magnitude(self, input_varname=None, output_varname=None):
        """If requested, calculate the magnitude of a vector variable"""
        # only run if vector field was given as input
        if ("NEC" in self.VARNAME) or ("XYZ" in self.VARNAME):
            # if no inputs are given, default to the class variables
            if input_varname is None:
                input_varname = self.VARNAME
            if output_varname is None:
                output_varname = "B"

            X = np.sqrt(np.sum(self.xarray[input_varname] ** 2, 1))
            self.append_array(output_varname, X.data, dims=("Timestamp"))
        else:
            logging.warn(
                "TfaInput: Will not calculate magnitude! Input field is not in vector form."
            )

    def subtract_chaos(self, field_varname=None, model_varname=None):
        """Remove the CHAOS Internal and Static field from the data"""
        # only run if class variable is  true
        if self.REMOVE_CHAOS:
            # if no inputs are given, default to the class variables
            if field_varname is None:
                field_varname = self.VARNAME
            if model_varname is None:
                model_varname = self.VARNAME + "_Model"

            B_res = self.xarray[field_varname].data - self.xarray[model_varname].data
            self.xarray[field_varname].data = B_res
        else:
            logging.warn(
                "TfaInput: Cannot subtract CHAOS if the 'remove_chaos' parameter has not been set to True"
            )

    def convert_to_mfa(self):
        """Convert field from NEC to MFA coordinates"""
        # only run for NEC variables
        if "NEC" in self.VARNAME:
            B_MFA = tfalib.mfa(
                self.xarray["B_NEC"].data, self.xarray["B_NEC_Model"].data
            )

            # Use the append_array() function when it gets updated
            # self.append_array("B_MFA", B_MFA, dims=("Timestamp", "MFA"))
            #
            # For now, its being done manually
            self.xarray = self.xarray.assign_coords({"MFA": ["M", "F", "A"]})
            self.xarray = self.xarray.assign({"B_MFA": (("Timestamp", "MFA"), B_MFA)})
        else:
            logging.warn("TfaInput: Cannot run Conversion to MFA for non-NEC inputs")

    def __init__(
        self, varname="B_NEC", remove_chaos=False, sampling_time="PT1S", *args, **kwargs
    ):
        # Select variable and set flag and maximum acceptable flag value
        self.VARNAME = varname
        if self.VARNAME == "B_NEC":
            self.FLAGNAME = "Flags_B"
            self.MAX_FLAG_VAL = 30
        elif self.VARNAME == "F":
            self.FLAGNAME = "Flags_F"
            self.MAX_FLAG_VAL = 63
        elif self.VARNAME == "E_NEC":
            self.FLAGNAME = "Flags_TII"
            self.MAX_FLAG_VAL = 23
        elif self.VARNAME == "n":
            self.FLAGNAME = "Flags_LP"
            self.MAX_FLAG_VAL = 63
        elif self.VARNAME == "Bubble_Probability":
            self.FLAGNAME = "Flags_F"
            self.MAX_FLAG_VAL = 1000
        elif self.VARNAME == "FAC":
            self.FLAGNAME = "Flags_F"
            self.MAX_FLAG_VAL = 1000
        elif self.VARNAME == "EEF":
            self.FLAGNAME = "Flags"
            self.MAX_FLAG_VAL = 1000
        elif self.VARNAME == "Eh_XYZ":
            self.FLAGNAME = "Quality_flags"
            self.MAX_FLAG_VAL = 0
        elif self.VARNAME == "Ev_XYZ":
            self.FLAGNAME = "Quality_flags"
            self.MAX_FLAG_VAL = 0
        else:
            sys.exit("TfaInput: unrecognized 'varname' input: " + varname)

        self.DEFAULTS["measurements"] = [self.VARNAME, self.FLAGNAME]

        # Check if CHAOS field removal has been requested and if yes, add
        # it to the input parameters so that it will be retrieved
        if remove_chaos:
            self.REMOVE_CHAOS = True
            self.DEFAULTS["model"] = "'CHAOS-Core' + 'CHAOS-Static'"
        else:
            self.REMOVE_CHAOS = False
            self.DEFAULTS["model"] = "None"

        # set pad_times to either 3 hours, if Low-res data), or 5 minutes if
        # high-res
        self.COLLECTION = kwargs["collection"]
        if "HR" in self.COLLECTION:
            self.DEFAULTS["pad_times"] = [
                dt.timedelta(minutes=5),
                dt.timedelta(minutes=5),
            ]
        else:
            self.DEFAULTS["pad_times"] = [dt.timedelta(hours=3), dt.timedelta(hours=3)]

        # Parse sampling_time argument to float and set the class variable
        self.SAMPLING_TIME = isotime2num(sampling_time)
        self.DEFAULTS["sampling_step"] = sampling_time

        # set time limits (user-requeste) and pad for retrieval
        self.TIME_LIMS = [kwargs["start_time"], kwargs["end_time"]]
        kwargs["start_time"] -= self.DEFAULTS["pad_times"][0]
        kwargs["end_time"] += self.DEFAULTS["pad_times"][-1]

        if "EXPT" in self.COLLECTION:

            from viresclient import SwarmRequest

            SERVER_URL = "https://vires.services/ows"
            request = SwarmRequest(SERVER_URL)
            request.set_collection(self.COLLECTION)

            if self.VARNAME == "Eh_XYZ":
                var_list = ["Ehx", "Ehy", "Ehz", "Quality_flags"]
            elif self.VARNAME == "Ev_XYZ":
                var_list = ["Evx", "Evy", "Evz", "Quality_flags"]
            request.set_products(measurements=var_list)

            data = request.get_between(
                kwargs["start_time"].strftime("%Y-%m-%dT%H:%M:%S"),
                kwargs["end_time"].strftime("%Y-%m-%dT%H:%M:%S"),
            )

            self.xarray = data.as_xarray()
            v = self.VARNAME[0:2]
            E_XYZ = np.vstack(
                (self.xarray[v + "x"], self.xarray[v + "y"], self.xarray[v + "z"])
            ).T
            self.xarray = self.xarray.assign_coords({"XYZ": ["X", "Y", "Z"]})
            self.xarray = self.xarray.assign(
                {self.VARNAME: (("Timestamp", "XYZ"), E_XYZ)}
            )

        elif "AUX_OBS" in self.COLLECTION:

            from viresclient import SwarmRequest

            SERVER_URL = "https://vires.services/ows"
            request = SwarmRequest(SERVER_URL)
            request.set_collection(self.COLLECTION)
            request.set_products(measurements=[self.VARNAME, "Quality"])
            self.FLAGNAME = "Quality"
            self.MAX_FLAG_VAL = 1

            data = request.get_between(
                kwargs["start_time"].strftime("%Y-%m-%dT%H:%M:%S"),
                kwargs["end_time"].strftime("%Y-%m-%dT%H:%M:%S"),
            )

            self.xarray = data.as_xarray()

        else:
            super().__init__(*args, **kwargs)


def isotime2num(iso_str):
    """Converts ISO duration to its numeric value in seconds"""
    # pairs of ISO characters and their values in seconds
    date_vals = (("Y", 31536000), ("M", 2592000), ("D", 86400))
    time_vals = (("H", 3600), ("M", 60), ("S", 1))

    duration = 0.0

    # isolate part between P and T chars, i.e. the date part
    s_date = re.findall("P(.+)T", iso_str)
    # check of values of years, months and days and add their values to duration
    if s_date:
        for (c, m) in date_vals:
            match = re.findall(r"([\d\.]+)" + c, s_date[0])
            if match:
                duration += float(match[0]) * m

    # isolate part between after the T char, i.e. the time part
    s_time = re.findall("T(.+)", iso_str)
    # check of values of hours, mins and secs and add their values to duration
    if s_time:
        for (c, m) in time_vals:
            match = re.findall(r"([\d\.]+)" + c, s_time[0])
            if match:
                duration += float(match[0]) * m

    return duration


class TfaProcessor:
    input_data: TfaInput
    X_name: str
    X_label: str
    params: dict
    segment_index: list

    """
    Data to be used by the TFA tools and functions

    Parameters
    ----------
    input_data : TfaInput object
        The object containing the data
    active_variable: dict
        Dictionary of the form {"varname": "F"} which specifies the name of
        the variable that will be used for further processing. If this
        variable is a vector, an additional key is required containing the
        component of this vector that will be processed, e.g.
        {"varname": "B_NEC", "component": 0}. Component number uses the
        Python convention of starting from zero, so 0 is the first component
        (N in the NEC case), 1 is the second (E) and 2 is the third (C).
    """

    def __init__(self, input_data, active_variable, params=None):
        self.X_name = None
        self.X_label = None
        self.input_data = input_data
        self.segment_index = []
        self.Max_Segment = 0

        self.params = (
            params
            if params
            else {
                "General": {
                    "freq_lims": [0.020, 0.100],
                    "lat_lims": None,
                    "maglat_lims": [-75, 75],
                }
            }
        )

        # specify the variable that will be processed
        self.X_name = active_variable["varname"]
        self.X_label = self.X_name

        X = self.input_data.xarray[self.X_name].data
        # if its a vector, isolate the specified component
        if "component" in active_variable.keys():
            Xi = active_variable["component"]
            X = X[:, Xi]
            self.X_label += "_" + str(Xi + 1)

        # get Timestamps and convert to seconds
        t_old = self.input_data.xarray.get("Timestamp").data
        t_old_sec = (t_old - t_old[0]) / np.timedelta64(
            1, "s"
        )  # converts diff to seconds

        # apply "constant_candence"
        new_t_sec, new_X = tfalib.constant_cadence(
            t_old_sec, X, 1 / self.input_data.SAMPLING_TIME, interp=False
        )[0:2]
        new_t = t_old[0] + new_t_sec * np.timedelta64(1, "s")

        # insert new values to xarray Dataset
        self.input_data.xarray = self.input_data.xarray.assign_coords({"time": new_t})
        self.input_data.xarray = self.input_data.xarray.assign({"X": (("time"), new_X)})

        # self.segment_index = self.create_segment_index()

    @property
    def analysis_window(self):
        """list[datetime]: pair of unpadded times"""
        return self.input_data.TIME_LIMS

    def apply(self, process):
        """Apply a TFA Process"""
        process.apply(self)

    def create_segment_index(self, lat_lims=None):
        """Create an array with the segment index for each time"""

        if not (lat_lims is None):
            if hasattr(lat_lims, "__len__") and (not isinstance(lat_lims, str)):
                if len(lat_lims) == 2:
                    self.params["General"]["maglat_lims"] = lat_lims
                elif len(lat_lims) == 1:
                    self.params["General"]["maglat_lims"] = [
                        -np.abs(lat_lims),
                        +np.abs(lat_lims),
                    ]
                else:
                    logging.warn(
                        "create_segment_index: 'lat_lims' not a 2-element \
                          vector - using default value"
                    )
            else:
                self.params["General"]["maglat_lims"] = [
                    -np.abs(lat_lims),
                    +np.abs(lat_lims),
                ]

        # interpolate mag_lat to new "time"
        old_t = (
            self.input_data.xarray["Timestamp"].data
            - self.input_data.xarray["Timestamp"].data[0]
        ).astype(np.float64)
        new_t = (
            self.input_data.xarray["time"].data - self.input_data.xarray["time"].data[0]
        ).astype(np.float64)
        mlat = np.interp(new_t, old_t, self.input_data.xarray["QDLat"])

        ind = np.arange(len(mlat))
        mlat_bool = (mlat > self.params["General"]["maglat_lims"][0]) & (
            mlat < self.params["General"]["maglat_lims"][1]
        )
        d = np.hstack(([0], np.diff(ind[mlat_bool])))
        c = np.cumsum(d > 1)

        si = np.full(mlat.shape, np.NaN)
        si[mlat_bool] = c

        # remove segments that are in the padded time intervals
        tlims = np.array(self.analysis_window).astype(np.datetime64)
        si[
            (self.input_data.xarray["time"].data < tlims[0])
            | (self.input_data.xarray["time"].data > tlims[-1])
        ] = np.NaN
        si = si - np.nanmin(si)

        self.segment_index = si
        self.Max_Segment = np.nanmax(si)

    def get_segment_inds_and_lims(self, segment):
        if segment is None:
            inds = np.full(self.input_data.xarray["time"].data.shape, True)
        else:
            if segment > self.Max_Segment:
                segment = self.Max_Segment
            inds = np.full(self.input_data.xarray["time"].data.shape, False)
            inds[self.segment_index == segment] = True

        t_min = np.nanmin(self.input_data.xarray["time"].data[inds])
        t_max = np.nanmax(self.input_data.xarray["time"].data[inds])

        return (inds, [t_min, t_max])

    def plotX(self, full=False, segment=None):
        """Plot the active variable time series"""

        (inds, [t_min, t_max]) = self.get_segment_inds_and_lims(segment)

        # plt.figure()
        plt.plot(
            self.input_data.xarray["time"].data[inds],
            self.input_data.xarray["X"].data[inds],
        )
        plt.title(self.input_data.COLLECTION)
        if self.X_label[0] == "E":
            y_label = self.X_label + " (mV/m)"
        else:
            y_label = self.X_label + " (nT)"
        plt.ylabel(y_label)
        plt.grid(True)
        if not full and segment is None:
            plt.xlim(self.input_data.TIME_LIMS)
        elif not full:
            plt.xlim([t_min, t_max])
        elif full:
            pass

    def image(self, full=False, segment=None, cbar_lims=None, log=True):
        """
        Plot the dynamic spectrum of the result of the wavelet transform

        The wavelet process must have been successfully applied first.
        """
        (inds, [t_min, t_max]) = self.get_segment_inds_and_lims(segment)

        freqs = 1000 / self.input_data.xarray["scale"].data

        if cbar_lims is None:
            m = np.max([np.log10(np.min(self.input_data.xarray["wavelet_power"])), -6])
            x = np.log10(np.max(self.input_data.xarray["wavelet_power"]))
        else:
            m, x = cbar_lims
        cb_ticks = np.arange(np.ceil(m), np.floor(x))

        # plt.figure()
        # plt.subplot(D,1,i+1)
        if log:
            plt.contourf(
                self.input_data.xarray["time"].data[inds],
                freqs,
                np.log10(self.input_data.xarray["wavelet_power"][:, inds]),
                cmap="jet",
                levels=np.linspace(m, x, 20),
                extend="both",
            )
        else:
            plt.contourf(
                self.input_data.xarray["time"].data[inds],
                freqs,
                self.input_data.xarray["wavelet_power"][:, inds],
                cmap="jet",
                levels=np.linspace(m, x, 20),
                extend="both",
            )
        # plt.yticks(ticks=yticks, labels=yticklabels)
        # plt.ylim(freq_lims)
        plt.ylabel("Freq (mHz)")
        cbh = plt.colorbar(orientation="horizontal", shrink=1, aspect=50)
        # cb_ticks = cbh.get_ticks()
        cbh.set_ticks(cb_ticks)
        cbh.set_ticklabels(["%.1f" % i for i in cb_ticks], fontsize=8)
        if not full and segment is None:
            plt.xlim(self.input_data.TIME_LIMS)
        elif not full:
            plt.xlim([t_min, t_max])
        # plt.title(self.input_data.COLLECTION)

    def plotAUX(self, full=False, segment=None):
        """Plot Mag.Lat. and MLT time series"""

        [t_min, t_max] = self.get_segment_inds_and_lims(segment)[1]

        # plt.figure()
        # fig,ax = plt.subplots()
        plt.plot(
            self.input_data.xarray["Timestamp"].data,
            self.input_data.xarray["QDLat"].data,
            "-b",
            label="QD Lat",
        )
        plt.plot(0, 0, "-r", label="MLT")  # this is just for the legend
        ax = plt.gca()
        ax.set_ylabel("QDLat (deg)")
        ax.set_ylim([-90, 90])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax2 = ax.twinx()
        ax2.plot(
            self.input_data.xarray["Timestamp"].data,
            self.input_data.xarray["MLT"].data,
            "-r",
            label="MLT",
        )
        ax2.set_ylabel("MLT (hr)")
        ax2.set_ylim([0, 24])
        ax2.set_yticks([0, 6, 12, 18, 24])
        # ax.title(self.input_data.COLLECTION)
        ax.grid(True)
        ax.legend()
        if not full and segment is None:
            ax.set_xlim(self.input_data.TIME_LIMS)
        elif not full:
            ax.set_xlim([t_min, t_max])
        elif full:
            pass

    def plotI(self, full=False, segment=None):
        """
        Plot the wave index time series.

        The wavelet process must have been successfully applied first and then
        the wave_index() function has to be executed to produce the index.
        Optionally, the user can run the wave_detection() function, before the
        wave_index() to remove parts of the signal that have been identified
        as suspicious false positives (e.g. spikes) or that might be related
        to ESF signatures (Plasma Bubbles).
        """

        (inds, [t_min, t_max]) = self.get_segment_inds_and_lims(segment)

        # plt.figure()
        plt.plot(
            self.input_data.xarray["time"].data[inds],
            self.input_data.xarray["wavelet_index"].data[inds],
        )
        plt.title(self.input_data.COLLECTION)
        plt.ylabel("Wave Index")
        plt.grid(True)
        if not full and segment is None:
            plt.xlim(self.input_data.TIME_LIMS)
        elif not full:
            plt.xlim([t_min, t_max])
        elif full:
            pass

    def wave_index(self):
        """
        Produce the index of wave activity for the frequencies that were used
        in the wavelet process.

        The wavelet process must have been successfully applied first.
        """
        if "wavelet_power" in self.input_data.xarray:
            wavindex = np.nansum(self.input_data.xarray["wavelet_power"].data, 0)
            self.input_data.xarray = self.input_data.xarray.assign(
                {"wavelet_index": (("time"), wavindex)}
            )
        else:
            logging.warn(
                "wave_index(): No wavelet array 'wavelet_power' found! Must apply the Wavelet function first!"
            )

    def interp_nans(self):
        """Interpolate NaN values by a piecewise linear interpolation scheme"""
        i = np.arange(len(self.input_data.xarray["X"]))
        nonNaNinds = np.where(~np.isnan(self.input_data.xarray["X"].data))
        self.input_data.xarray["X"].data = np.interp(
            i, i[nonNaNinds], self.input_data.xarray["X"].data[nonNaNinds]
        )

    def wave_detection(self, threshold=0):
        """
        Remove parts of the wavelet spectrum that might not be true waves.

        The wavelet process must have been successfully applied first.

        This function removes (sets to NaN) parts of the wavelet spectrum that
        might be due to spikes, data gaps, ESFs or trailing parts of wave
        activity from either above or below the range of frequencies that were
        used to perform the wavelet transform.
        """
        if "wavelet_power" in self.input_data.xarray:
            # remove points below the threshold
            threshInds = self.input_data.xarray["wavelet_power"].data < threshold
            self.input_data.xarray["wavelet_power"].data[threshInds] = np.NaN

            # remove points that are outside the segments
            # (if segments have been defined)
            if len(self.segment_index) > 0:
                outInds = np.isnan(self.segment_index)
                self.input_data.xarray["wavelet_power"].data[:, outInds] = np.NaN

            # find peak frequency for each time and exclude events with peak
            # at the edges of the frequency range
            maxInds = np.argmax(self.input_data.xarray["wavelet_power"].data, 0)
            self.input_data.xarray["wavelet_power"].data[
                :, np.where(maxInds == 0)
            ] = np.NaN
            nFreqs = len(self.input_data.xarray["scale"])
            self.input_data.xarray["wavelet_power"].data[
                :, np.where(maxInds == nFreqs - 1)
            ] = np.NaN

            # remove according to IBI (only for Swarm!)
            if self.input_data.COLLECTION[0:2] == "SW":
                sat = self.input_data.COLLECTION[11]  # get sat char

                from viresclient import SwarmRequest

                SERVER_URL = "https://vires.services/ows"
                request = SwarmRequest(SERVER_URL)
                request.set_collection("SW_OPER_IBI%sTMS_2F" % sat)
                request.set_products(
                    measurements=["Bubble_Probability"], sampling_step="PT1S"
                )

                logging.warn("wave_detection: Retrieving IBI L2 product")
                data = request.get_between(
                    self.analysis_window[0].strftime("%Y-%m-%dT%H:%M:%S"),
                    self.analysis_window[1].strftime("%Y-%m-%dT%H:%M:%S"),
                )

                ibi = data.as_xarray()
                # interpolate to the times of the wavelet_power array
                bubble = np.interp(
                    self.input_data.xarray["time"].data.astype(np.float64),
                    ibi["Timestamp"].data.astype(np.float64),
                    ibi["Bubble_Probability"].data.astype(np.float64),
                )

                bubbleInds = np.where(bubble > 0.20)
                self.input_data.xarray["wavelet_power"].data[:, bubbleInds] = np.NaN
                return bubble

        else:
            logging.warn(
                "wave_detection(): No wavelet array 'wavelet_power' found! Must apply the Wavelet function first!"
            )


class TFA_Process(ABC):
    params = None

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def apply(self, target):
        return

    # append the parameters to the "meta" dictionary of the target TFA_Data object
    def append_params(self, target):
        target.params[self.__name__] = self.params


class Cleaning(TFA_Process):
    __name__ = "Cleaning"

    def __init__(self, params=None):
        if params is None:
            self.params = {"Window_Size": 10, "Method": "iqr", "Multiplier": 0.5}
        else:
            self.params = params

    def apply(self, target):
        inds = tfalib.outliers(
            target.input_data.xarray["X"].data,
            self.params["Window_Size"],
            method=self.params["Method"],
            multiplier=self.params["Multiplier"],
        )
        target.input_data.xarray["X"].data[inds] = np.NaN

        # interpolate cleaned values and pre-existing gaps
        s = target.input_data.xarray["X"].data.shape
        if len(s) == 1:
            N = s[0]
            t_ind = np.arange(N)
            x = target.input_data.xarray["X"].data
            nonNaN = ~np.isnan(x)
            y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
            target.input_data.xarray["X"].data = y
        else:
            N, D = target.input_data.xarray["X"].data.shape
            t_ind = np.arange(N)
            for i in range(D):
                x = np.reshape(target.input_data.xarray["X"].data[:, i], (N,))
                nonNaN = ~np.isnan(x)
                y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
                target.input_data.xarray["X"].data = y

        self.append_params(target)

        return target


class Filtering(TFA_Process):
    __name__ = "Filtering"

    def __init__(self, params=None):
        if params is None:
            self.params = None
        else:
            self.params = params

    def apply(self, target):
        if self.params is None:
            self.params = {
                "Sampling_Rate": 1 / target.input_data.SAMPLING_TIME,
                "Cutoff_Frequency": 20 / 1000,
            }
        else:
            self.params["Sampling_Rate"] = 1 / target.input_data.SAMPLING_TIME

            if "Cutoff_Scale" in self.params:
                self.params["Cutoff_Frequency"] = 1 / self.params["Cutoff_Scale"]

        target.input_data.xarray["X"].data = tfalib.filter(
            target.input_data.xarray["X"].data,
            self.params["Sampling_Rate"],
            self.params["Cutoff_Frequency"],
        )
        self.append_params(target)

        return target


class Wavelet(TFA_Process):
    __name__ = "Wavelet"

    def __init__(self, params=None):
        if params is None:
            self.params = None
        else:
            if "Min_Frequency" in params and "Max_Frequency" in params:
                if params["Min_Frequency"] < params["Max_Frequency"]:
                    self.params = params
                else:
                    logging.warn("Min_Frequency must be smaller than Max_Frequency")
            elif "Min_Frequency" in params or "Max_Frequency" in params:
                logging.warn(
                    "The limits must both be in either frequency or scale,\
                      no combinations allowed."
                )
            else:
                if params["Min_Scale"] < params["Max_Scale"]:
                    self.params = params
                else:
                    logging.warn("Min_Scale must be smaller than Max_Scale")

    def apply(self, target):
        if self.params is None:
            self.params = {
                "Time_Step": target.input_data.SAMPLING_TIME,
                "Min_Scale": 1000 / 100,
                "Max_Scale": 1000 / 1,
                "dj": 0.1,
            }
        else:
            self.params["Time_Step"] = target.input_data.SAMPLING_TIME

            if "Min_Frequency" in self.params and "Max_Frequency" in self.params:
                if self.params["Max_Frequency"] <= 1 / (2 * self.params["Time_Step"]):
                    self.params["Min_Scale"] = 1 / self.params["Max_Frequency"]
                    self.params["Max_Scale"] = 1 / self.params["Min_Frequency"]
                else:
                    logging.warn(
                        "Max_Frequency needs to be smaller than 1/(2*Time_Step)"
                    )

            else:
                if self.params["Min_Scale"] < 2 * self.params["Time_Step"]:
                    logging.warn("Min_Scale needs to be bigger or equal to 2*Time_Step")

        self.params["Wavelet_Function"] = "Morlet"
        self.params["Wavelet_Param"] = 6.2036
        self.params["Wavelet_Norm_Factor"] = 0.74044116

        s = tfalib.wavelet_scales(
            self.params["Min_Scale"], self.params["Max_Scale"], self.params["dj"]
        )

        wave = tfalib.wavelet_transform(
            target.input_data.xarray["X"].data,
            dx=self.params["Time_Step"],
            minScale=self.params["Min_Scale"],
            maxScale=self.params["Max_Scale"],
            dj=self.params["dj"],
        )[0]
        norm = tfalib.wavelet_normalize(
            np.abs(wave) ** 2,
            s,
            dx=self.params["Time_Step"],
            dj=self.params["dj"],
            wavelet_norm_factor=0.74044116,
        )

        # delete old ones, if they exist, first! This is necessary for multiple
        # applications of the Wavelet() process, otherwise it conlficts with
        # the variables that are already in the xarray
        if "wavelet_power" in target.input_data.xarray:
            target.input_data.xarray = target.input_data.xarray.drop("wavelet_power")
        if "scale" in target.input_data.xarray:
            target.input_data.xarray = target.input_data.xarray.drop("scale")

        # insert new values to xarray Dataset
        target.input_data.xarray = target.input_data.xarray.assign_coords({"scale": s})
        target.input_data.xarray = target.input_data.xarray.assign(
            {"wavelet_power": (("scale", "time"), norm)}
        )

        self.append_params(target)

        return target
