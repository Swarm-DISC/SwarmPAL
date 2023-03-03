import matplotlib.pyplot as plt
from datatree import DataTree, register_datatree_accessor
from numpy import stack
from xarray import Dataset

from swarmpal.io import PalProcess
from swarmpal.toolboxes.fac import fac_single_sat_algo


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
        time_out = fac_results["time"]
        fac_out = fac_results["fac"]
        # Insert a new output dataset with these results
        ds_out = Dataset(
            data_vars={
                "Timestamp": ("Timestamp", time_out),
                "FAC": ("Timestamp", fac_out),
            }
        )
        ds_out["FAC"].attrs = {"units": "uA/m2"}
        subtree["output"] = DataTree(data=ds_out)
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
        measurement_varname = process_config.get("measurement_varname", "B_NEC")
        model_varname = process_config.get("model_varname", "B_NEC_Model")
        dataset = process_config.get("dataset")
        residual = (
            self._datatree[f"{active_tree}/{dataset}"][measurement_varname]
            - self._datatree[f"{active_tree}/{dataset}"][model_varname]
        )
        residual.diff(dim="Timestamp").plot.line(ax=axes[0], hue="NEC")
        self._datatree[f"{active_tree}/{dataset}/output"]["FAC"].plot.line(ax=axes[1])
        axes[0].set_xlabel("")
        axes[0].set_ylabel(r"d/dt($\Delta$B)")
        axes[0].grid()
        axes[1].grid()
        return fig, axes
