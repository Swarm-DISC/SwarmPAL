from __future__ import annotations

import numpy as np
from datatree import DataTree
from xarray import DataArray, Dataset

from swarmpal.io import PalProcess
from swarmpal.toolboxes.tfa import tfalib
from swarmpal.utils.exceptions import PalError

FLAG_THRESHOLDS = {
    "B_NEC": {"flag_name": "Flags_B", "max_val": 30},
    "F": {"flag_name": "Flags_F", "max_val": 63},
    "E_NEC": {"flag_name": "Flags_TII", "max_val": 23},
    "n": {"flag_name": "Flags_LP", "max_val": 63},
    "Bubble_Probability": {"flagn_ame": "Flags_F", "max_val": 1000},
    "FAC": {"flag_name": "Flags_F", "max_val": 1000},
    "EEF": {"flag_name": "Flags", "max_val": 1000},
    "Eh_XYZ": {"flag_name": "Quality_flags", "max_val": 0},
    "Ev_XYZ": {"flag_name": "Quality_flags", "max_val": 0},
}


class Preprocess(PalProcess):
    """

    Notes
    -----
    Required config parameters:
    dataset
    active_variable
    Optional config parameters:
    remove_model
    model
    convert_to_mfa
    clean_by_flags
    clean_varname
    clean_flagname
    clean_maxval
    """

    @property
    def process_name(self) -> str:
        return "TFA_Preprocess"

    def set_config(
        self,
        dataset: str = "",
        active_variable: str = "",
        active_component: int | None = None,
        sampling_rate: float = 1,
        remove_model: bool = False,
        model: str = "",
        convert_to_mfa: bool = False,
        use_magnitude: bool = False,
        clean_by_flags: bool = False,
        flagclean_varname: str = "",
        flagclean_flagname: str = "",
        flagclean_maxval: int | None = None,
    ) -> None:
        self.config = dict(
            dataset=dataset,
            active_variable=active_variable,
            active_component=active_component,
            sampling_rate=sampling_rate,
            remove_model=remove_model,
            model=model,
            convert_to_mfa=convert_to_mfa,
            use_magnitude=use_magnitude,
            clean_by_flags=clean_by_flags,
            flagclean_varname=flagclean_varname,
            flagclean_flagname=flagclean_flagname,
            flagclean_maxval=flagclean_maxval,
        )

    @property
    def active_variable(self):
        return self.config.get("active_variable", "")

    @property
    def active_component(self):
        return self.config.get("active_component", "")

    def _call(self, datatree: DataTree) -> DataTree:
        self._validate_inputs(datatree)
        # Select the datatree to work on
        self.subtree = datatree[self.config.get("dataset")]
        # Prepare data depending on the content
        if self.active_variable in ("Eh_XYZ", "Ev_XYZ"):
            ds = self._prep_efi_expt_data(self.subtree.ds)
        else:
            ds = self._prep_magnetic_data(self.subtree.ds)
        # Optionally clean according to flag values
        if self.config.get("clean_by_flags", False):
            ds = self._flag_cleaning(ds)
        # Identify and assign "TFA_Variable" (to be used in next processes)
        if self.active_component is not None:
            da = ds[self.active_variable][:, self.active_component].copy(deep=True)
        elif self.config["use_magnitude"]:
            da = (ds[self.active_variable] ** 2).sum(axis=1).pipe(np.sqrt)
        else:
            da = ds[self.active_variable].copy(deep=True)
        da = da.rename({"Timestamp": "TFA_Time"})
        da = self._constant_cadence(da)
        ds = ds.assign({"TFA_Variable": da, "TFA_Time": da["TFA_Time"]})
        # Assign dataset back into the datatree to return
        self.subtree = self.subtree.assign(ds.copy())
        self.subtree.parent = datatree
        return datatree

    def _validate_inputs(self, datatree):
        """Some checks that the inputs and config are valid"""
        dataset = self.config.get("dataset")
        active_variable = self.config.get("active_variable")
        active_component = self.config.get("active_component")
        use_magnitude = self.config.get("use_magnitude")
        if not all((dataset, active_variable)):
            raise PalError("TFA Preprocess: dataset and/or active_variable not set")
        # Catch the cases with special names that aren't initially available in the dataset (they are set later)
        if any(x in active_variable for x in ("B_NEC_res_", "MFA", "Eh_XYZ", "Ev_XYZ")):
            target_shape = (len(datatree[dataset]["Timestamp"]), 3)
        else:
            target_shape = datatree[dataset][active_variable].shape
        # Check if active_component is set appropriately, according to the shape of the active_variable
        if (
            (len(target_shape) > 1)
            and (active_component is None)
            and (not use_magnitude)
        ):
            raise PalError("TFA Preprocess: active_component not set")
        if (len(target_shape) == 1) and (active_component is not None):
            raise PalError("TFA Preprocess: active_component set, but no vector found")

    def _prep_magnetic_data(self, ds: Dataset) -> Dataset:
        """Subtract model and/or rotate to MFA"""
        remove_model = self.config.get("remove_model", False)
        convert_to_mfa = self.config.get("convert_to_mfa", False)
        # Identify model name from config or from PAL meta
        model = self.config.get("model", "")
        model = model if model else self.subtree.swarmpal.magnetic_model_name
        # Optionally assign residuals to dataset
        if remove_model:
            ds = ds.assign(
                {"B_NEC_res_Model": self.subtree.swarmpal.magnetic_residual(model)},
            )
        # Optionally rotate to MFA (Mean-field aligned coordinates)
        if convert_to_mfa:
            if remove_model:
                B_MFA = tfalib.mfa(
                    ds["B_NEC_res_Model"].data, ds[f"B_NEC_{model}"].data
                )
            else:
                B_MFA = tfalib.mfa(ds["B_NEC"].data, ds[f"B_NEC_{model}"].data)
            ds = ds.assign_coords({"MFA": [0, 1, 2]})
            ds = ds.assign({"B_MFA": (("Timestamp", "MFA"), B_MFA)})
            ds["B_MFA"].attrs = {
                "units": "nT",
                "description": "Magnetic field in Mean-field aligned coordinates",
            }
        return ds

    def _prep_efi_expt_data(self, ds: Dataset) -> Dataset:
        """Assign the Eh_XYZ or Ev_XYZ vector data variable"""
        # Validate input data
        available_vars = set(ds.data_vars)
        vectors = {
            "Eh_XYZ": ("Ehx", "Ehy", "Ehz"),
            "Ev_XYZ": ("Evx", "Evy", "Evz"),
        }
        vectors = vectors[self.active_variable]
        required_vars = {*vectors, "Quality_flags"}
        if not required_vars.issubset(available_vars):
            raise PalError(f"Not all available: {required_vars}")
        # Create and assign the vector parameter
        E_XYZ = np.vstack([ds[i] for i in vectors]).T
        ds = ds.assign_coords({"XYZ": ["X", "Y", "Z"]})
        ds = ds.assign({self.active_variable: (("Timestamp", "XYZ"), E_XYZ)})
        return ds

    def _flag_cleaning(self, ds):
        """Set values to NaN where flags exceed a threshold"""
        varname = self.config.get("flagclean_varname", None)
        flagname = self.config.get("flagclean_flagname", None)
        max_val = self.config.get("flagclean_maxval", None)
        # Use default parameters if none given in config
        varname = varname if varname else self.active_variable
        flagname = flagname if flagname else FLAG_THRESHOLDS[varname]["flag_name"]
        max_val = max_val if max_val else FLAG_THRESHOLDS[varname]["max_val"]
        # Set flagged values to NaN
        inds_to_remove = ds[flagname] > max_val
        ds[varname][inds_to_remove, ...] = np.NaN
        return ds

    def _constant_cadence(self, da):
        """Convert array to that of constant cadence"""
        # Convert time to seconds for tfalib.constant_cadence
        t_old = da["TFA_Time"].data
        t_old_sec = (t_old - t_old[0]) / np.timedelta64(1, "s")
        new_t_sec, new_X = tfalib.constant_cadence(
            t_old_sec, da.data, self.config["sampling_rate"], interp=False
        )[0:2]
        new_t = t_old[0] + (new_t_sec * 1e9).astype("timedelta64[ns]")
        # Assign into new array to return
        da_new = DataArray(
            data=new_X,
            dims=("TFA_Time",),
        )
        da_new = da_new.assign_coords({"TFA_Time": new_t})
        da_new.attrs = {
            "units": da.attrs.get("units", ""),
            "description": da.attrs.get("description", ""),
        }
        da_new["TFA_Time"].attrs = {
            "units": da["TFA_Time"].attrs.get("units", ""),
            "description": da["TFA_Time"].attrs.get("description", ""),
        }
        return da_new


def _get_tfa_active_subtree(datatree):
    """Returns the relevant subtree when Preprocess has been applied"""
    # Scan the tree based on previous preprocess application
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    tfa_preprocess_meta = pal_processes_meta.get("TFA_Preprocess")
    if not tfa_preprocess_meta:
        raise PalError("Must first run tfa.processes.Preprocess")
    return datatree[tfa_preprocess_meta.get("dataset")]


def _get_sampling_rate(datatree):
    """Get the sampling rate set by Preprocess"""
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    tfa_preprocess_meta = pal_processes_meta.get("TFA_Preprocess")
    return tfa_preprocess_meta["sampling_rate"]


class Clean(PalProcess):
    """Clean TFA_Variable by removing outliers and interpolate gaps"""

    @property
    def process_name(self) -> str:
        return "TFA_Clean"

    def set_config(
        self,
        window_size: int = 10,
        method: str = "iqr",
        multiplier: float = 0.5,
    ) -> None:
        self.config = dict(
            window_size=window_size,
            method=method,
            multiplier=multiplier,
        )

    def _call(self, datatree) -> DataTree:
        # Identify the DataArray to modify
        subtree = _get_tfa_active_subtree(datatree)
        target_var = subtree["TFA_Variable"]
        # Apply cleaning routine inplace
        self._clean_variable(target_var)
        return datatree

    def _clean_variable(self, target_var) -> DataArray:
        # Remove outliers
        inds = tfalib.outliers(
            target_var.data,
            self.config.get("window_size"),
            method=self.config.get("method"),
            multiplier=self.config.get("multiplier"),
        )
        target_var.data[inds] = np.NaN
        # Interpolate over gaps
        s = target_var.data.shape
        if len(s) == 1:
            N = s[0]
            t_ind = np.arange(N)
            x = target_var.data
            nonNaN = ~np.isnan(x)
            y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
            target_var.data = y
        else:
            N, D = target_var.data.shape
            t_ind = np.arange(N)
            for i in range(D):
                x = np.reshape(target_var.data[:, i], (N,))
                nonNaN = ~np.isnan(x)
                y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
                target_var.data[:, i] = y
        return target_var


class Filter(PalProcess):
    """Filtering"""

    @property
    def process_name(self) -> str:
        return "TFA_Filtering"

    def set_config(
        self,
        sampling_rate: float | None = None,
        cutoff_frequency: float = 20 / 1000,
    ) -> None:
        self.config = dict(
            sampling_rate=sampling_rate,
            cutoff_frequency=cutoff_frequency,
        )

    def _call(self, datatree) -> DataTree:
        self._configure(datatree)
        # Identify the DataArray to modify
        subtree = _get_tfa_active_subtree(datatree)
        target_var = subtree["TFA_Variable"]
        # Apply filtering routine inplace
        target_var = self._filter(target_var)
        return datatree

    def _configure(self, datatree):
        if self.config["sampling_rate"] is None:
            self.config["sampling_rate"] = _get_sampling_rate(datatree)

    def _filter(self, target_var) -> DataArray:
        target_var.data = tfalib.filter(
            target_var.data,
            self.config.get("sampling_rate"),
            self.config.get("cutoff_frequency"),
        )
        return target_var


class Wavelet(PalProcess):
    """Apply wavelet analysis"""

    @property
    def process_name(self) -> str:
        return "TFA_Wavelet"

    def set_config(
        self,
        sampling_rate: float | None = None,
        min_frequency: float | None = None,
        max_frequency: float | None = None,
        min_scale: float | None = None,
        max_scale: float | None = None,
        dj: float = 0.1,
    ) -> None:
        self.config = dict(
            sampling_rate=sampling_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_scale=min_scale,
            max_scale=max_scale,
            dj=dj,
        )

    def _call(self, datatree: DataTree) -> DataTree:
        self._configure(datatree)
        # Identify the DataArray to modify
        subtree = _get_tfa_active_subtree(datatree)
        target_var = subtree["TFA_Variable"]
        # Apply wavelet routine
        norm, scale = self._wavelets(target_var)
        # Assign new array to dataset
        subtree = subtree.assign_coords({"scale": scale})
        subtree["wavelet_power"] = DataArray(
            data=norm,
            dims=("scale", "TFA_Time"),
        )
        subtree.parent = datatree
        return datatree

    def _configure(self, datatree):
        if self.config["min_scale"] is None:
            self.config["min_scale"] = 1 / self.config["max_frequency"]
            self.config["max_scale"] = 1 / self.config["min_frequency"]
        if self.config["sampling_rate"] is None:
            self.config["sampling_rate"] = _get_sampling_rate(datatree)

    def _wavelets(self, target_var: DataArray):
        scale = tfalib.wavelet_scales(
            self.config.get("min_scale"),
            self.config.get("max_scale"),
            self.config.get("dj"),
        )
        wave = tfalib.wavelet_transform(
            target_var.data,
            dx=1 / self.config.get("sampling_rate"),
            minScale=self.config.get("min_scale"),
            maxScale=self.config.get("max_scale"),
            dj=self.config.get("dj"),
        )[0]
        norm = tfalib.wavelet_normalize(
            np.abs(wave) ** 2,
            scale,
            dx=1 / self.config.get("sampling_rate"),
            dj=self.config.get("dj"),
            wavelet_norm_factor=0.74044116,
        )
        return norm, scale


class WaveDetection(PalProcess):
    """Screen out potential false waves

    Removes part of the wavelet spectrum that might be due to spikes, data gaps, ESFs or trailing parts of wave
    activity from either above or below the range of frequencies that were
    used to perform the wavelet transform.
    """

    @property
    def process_name(self) -> str:
        return "TFA_WaveDetection"

    def set_config(
        self,
    ):
        ...

    def _call(self, datatree):
        ...

    def _attach_ibi(self):
        ...
