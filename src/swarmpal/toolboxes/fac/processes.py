from __future__ import annotations

import matplotlib.pyplot as plt
from datatree import DataTree, register_datatree_accessor
from numpy import stack
from xarray import Dataset

from swarmpal.io import PalProcess
from swarmpal.toolboxes.fac.fac_algorithms import fac_single_sat_algo

__all__ = (
    "FAC_singlesat",
    "PalFacDataTreeAccessor",
)


class FAC_singlesat(PalProcess):
    """Provides the process for the classic single-satellite FAC algorithm

    Notes
    -----
    Expected config parameters:
    dataset
    model_varname
    measurement_varname
    """

    @property
    def process_name(self):
        return "FAC_singlesat"

    def set_config(
        self,
        dataset: str = "SW_OPER_MAGA_LR_1B",
        model_varname: str = "B_NEC_CHAOS",
        measurement_varname: str = "B_NEC",
        inclination_limit: float = 30,
    ) -> None:
        self.config = dict(
            dataset=dataset,
            model_varname=model_varname,
            measurement_varname=measurement_varname,
        )

    def _call(self, datatree):
        # Identify inputs for algorithm
        subtree = datatree[self.config.get("dataset")]
        dataset = subtree.ds
        time = self._get_time(dataset)
        positions = self._get_positions(dataset)
        B_res = self._get_B_res(dataset)
        B_model = self._get_B_model(dataset)
        # Apply algorithm
        fac_results = fac_single_sat_algo(
            time=time, positions=positions, B_res=B_res, B_model=B_model
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
        subtree["PAL:FAC_output"] = DataTree(data=ds_out)
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


@register_datatree_accessor("swarmpal_fac")
class PalFacDataTreeAccessor:
    def __init__(self, datatree) -> None:
        self._datatree = datatree

    def quicklook(self, active_tree="."):
        fig, axes = plt.subplots(nrows=2, sharex=True)
        # TODO: refactor to be able to identify active tree
        process_config = self._datatree.swarmpal.pal_meta[active_tree]["FAC_singlesat"]
        dataset = process_config.get("dataset")
        self._datatree[f"{active_tree}/{dataset}/PAL:FAC_output"]["IRC"].plot.line(
            ax=axes[0]
        )
        self._datatree[f"{active_tree}/{dataset}/PAL:FAC_output"]["FAC"].plot.line(
            ax=axes[1]
        )
        axes[0].set_xlabel("")
        axes[0].grid()
        axes[1].grid()
        return fig, axes
