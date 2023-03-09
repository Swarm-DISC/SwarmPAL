import numpy as np
from datatree import DataTree
from xarray import Dataset

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
    def process_name(self):
        return "Preprocess"

    def set_config(
        self,
        dataset: str = "",
        active_variable: str = "",
        remove_model: bool = False,
        model: str = "",
        convert_to_mfa: bool = False,
        clean_by_flags: bool = False,
        clean_varname: str = "",
        clean_flagname: str = "",
        clean_maxval: int | None = None,
    ) -> None:
        self.config = dict(
            dataset=dataset,
            active_variable=active_variable,
            remove_model=remove_model,
            model=model,
            convert_to_mfa=convert_to_mfa,
            clean_by_flags=clean_by_flags,
            clean_varname=clean_varname,
            clean_flagname=clean_flagname,
            clean_maxval=clean_maxval,
        )

    @property
    def active_variable(self):
        return self.config.get("active_variable", "")

    def _call(self, datatree: DataTree) -> DataTree:
        self._validate_inputs()
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
        # Assign dataset back into the datatree to return
        self.subtree = self.subtree.assign(ds.copy())
        self.subtree.parent = datatree
        return datatree

    def _validate_inputs(self):
        """Some checks that the inputs and config are valid"""
        dataset = self.config.get("dataset")
        active_variable = self.config.get("active_variable")
        if not all((dataset, active_variable)):
            raise PalError("dataset and/or active_variable not set")

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
        varname = self.config.get("clean_varname", None)
        flagname = self.config.get("clean_flagname", None)
        max_val = self.config.get("clean_maxval", None)
        # Use default parameters if none given in config
        varname = varname if varname else self.active_variable
        flagname = flagname if flagname else FLAG_THRESHOLDS[varname]["flag_name"]
        max_val = max_val if max_val else FLAG_THRESHOLDS[varname]["max_val"]
        # Set flagged values to NaN
        inds_to_remove = ds[flagname] > max_val
        ds[varname][inds_to_remove, ...] = np.NaN
        return ds
