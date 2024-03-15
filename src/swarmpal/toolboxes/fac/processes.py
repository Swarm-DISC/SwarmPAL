from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from datatree import DataTree, register_datatree_accessor
from numpy import stack
from xarray import Dataset

from swarmpal.io import PalProcess
from swarmpal.toolboxes.fac.fac_algorithms import fac_single_sat_algo

logger = logging.getLogger(__name__)

__all__ = (
    "FAC_single_sat",
    "PalFacDataTreeAccessor",
)


class FAC_single_sat(PalProcess):
    """Provides the process for the classic single-satellite FAC algorithm"""

    @property
    def process_name(self):
        return "FAC_single_sat"

    def set_config(
        self,
        dataset: str = "SW_OPER_MAGA_LR_1B",
        model_varname: str = "B_NEC_CHAOS",
        measurement_varname: str = "B_NEC",
        inclination_limit: float = 30,
        time_jump_limit: int = 1,
        include_auxiliaries: bool = True,
    ) -> None:
        """Configures the process

        Parameters
        ----------
        dataset : str, optional
            Dataset to use, by default "SW_OPER_MAGA_LR_1B"
        model_varname : str, optional
            Name of the magnetic model predictions, by default "B_NEC_CHAOS"
        measurement_varname : str, optional
            Name of the measurements, by default "B_NEC"
        inclination_limit : float, optional
            Limit of inclination for FAC validity (in degrees), by default 30
        time_jump_limit : int, optional
            Maximum allowable time step in data for FAC validity (in seconds), by default 1
        include_auxiliaries : bool, optional
            Whether to include e.g. Latitude, Longitude, Flags, etc, by default True
        """
        self.config = dict(
            dataset=dataset,
            model_varname=model_varname,
            measurement_varname=measurement_varname,
            inclination_limit=inclination_limit,
            time_jump_limit=time_jump_limit,
            include_auxiliaries=include_auxiliaries,
        )

    def _call(self, datatree):
        # Identify inputs for algorithm
        subtree = datatree[self.config.get("dataset")]
        dataset_in = subtree.ds
        # Apply algorithm
        fac_results = fac_single_sat_algo(
            time=self._get_time(dataset_in),
            positions=self._get_positions(dataset_in),
            B_res=self._get_B_res(dataset_in),
            B_model=self._get_B_model(dataset_in),
            inclination_limit=self.config.get("inclination_limit"),
            time_jump_limit=self.config.get("time_jump_limit"),
        )
        # Insert a new output dataset with these results
        ds_out = Dataset(
            data_vars={
                "Timestamp": ("Timestamp", fac_results["time"]),
                "FAC": ("Timestamp", fac_results["fac"]),
                "IRC": ("Timestamp", fac_results["irc"]),
            }
        )
        ds_out["FAC"].attrs = {"units": "uA/m2"}
        ds_out["IRC"].attrs = {"units": "uA/m2"}
        if self.config.get("include_auxiliaries"):
            ds_out = self._append_aux(dataset_in, ds_out)
        datatree["PAL_FAC_single_sat"] = DataTree(data=ds_out)
        return datatree

    def _validate(self):
        ...

    def _get_time(self, dataset):
        return dataset.get("Timestamp").data.astype("datetime64[ns]")

    def _get_positions(self, dataset):
        return stack(
            [
                dataset.get("Latitude").data,
                dataset.get("Longitude").data,
                dataset.get("Radius").data,
            ],
            axis=1,
        )

    def _get_B_res(self, dataset):
        measurement_varname = self.config.get("measurement_varname", "B_NEC")
        model_varname = self.config.get("model_varname", "B_NEC_Model")
        return dataset.get(measurement_varname).data - dataset.get(model_varname).data

    def _get_B_model(self, dataset):
        model_varname = self.config.get("model_varname", "B_NEC_Model")
        return dataset.get(model_varname).data

    def _append_aux(self, ds_in, ds_out):
        """Extract auxiliary information from inputs and add to output dataset"""
        aux_in = set(ds_in.data_vars)
        aux_desired = {
            "Latitude",
            "Longitude",
            "Radius",
            "Flags_F",
            "Flags_B",
            "Flags_q",
        }
        aux_matched = aux_desired.intersection(aux_in)
        aux_missing = aux_desired.difference(aux_in)
        logging.warning(f"Missing auxiliaries: {aux_missing}")
        ds_in_interpd = ds_in[list(aux_matched)].interp_like(ds_out, method="nearest")
        ds_out = ds_out.assign(
            {aux_name: ds_in_interpd[aux_name] for aux_name in aux_matched}
        )
        return ds_out


@register_datatree_accessor("swarmpal_fac")
class PalFacDataTreeAccessor:
    def __init__(self, datatree) -> None:
        self._datatree = datatree

    def quicklook(self, active_tree="."):
        fig, axes = plt.subplots(nrows=2, sharex=True)
        # TODO: refactor to be able to identify active tree
        process_config = self._datatree.swarmpal.pal_meta[active_tree]["FAC_single_sat"]
        dataset = process_config.get("dataset")
        self._datatree[f"{active_tree}/PAL_FAC_single_sat"]["IRC"].plot.line(ax=axes[0])
        self._datatree[f"{active_tree}/PAL_FAC_single_sat"]["FAC"].plot.line(ax=axes[1])
        axes[0].set_xlabel("")
        axes[0].grid()
        axes[1].grid()
        fig.suptitle(f"{dataset}")
        return fig, axes
