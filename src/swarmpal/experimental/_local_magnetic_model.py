from __future__ import annotations

import chaosmagpy as cp
import pooch
from numpy import vstack
from numpy.typing import ArrayLike
from pandas import to_datetime
from xarray import DataArray

from swarmpal.io import PalProcess

LATEST_CHAOS = {
    "CORE": "CHAOS-8.2_core.shc",
    "CORE_EXTRAPOLATED": "CHAOS-8.2_core_extrapolated.shc",
    "STATIC": "CHAOS-8.2_static.shc",
}

CHAOS_REGISTRY = {
    "CHAOS-8.2_core.shc": "md5:c41b0111f14763f8722d2e54e7367c71",
    "CHAOS-8.2_core_extrapolated.shc": "md5:a905f4a90ad1f6603030f64eccdeb106",
    "CHAOS-8.2_static.shc": "md5:079ba5811d89d9bd66ccf0084891d689",
}

CHAOS_BASE_URL = "https://zenodo.org/records/14893049/files/"


def fetch_chaos_file(filename):
    return pooch.retrieve(
        url=f"{CHAOS_BASE_URL}{filename}?download=1",
        known_hash=CHAOS_REGISTRY[filename],
    )


def fetch_latest_chaos_files():
    core_file = fetch_chaos_file(LATEST_CHAOS["CORE"])
    core_extrapolated_file = fetch_chaos_file(LATEST_CHAOS["CORE_EXTRAPOLATED"])
    static_file = fetch_chaos_file(LATEST_CHAOS["STATIC"])
    return {
        "core": core_file,
        "core_extrapolated": core_extrapolated_file,
        "static": static_file,
    }


def evaluate_chaos(
    longitude: ArrayLike, latitude: ArrayLike, radius: ArrayLike, time: ArrayLike
):
    chaos_files = fetch_latest_chaos_files()
    model_core = cp.load_CHAOS_shcfile(chaos_files["core"])
    # # TODO: Add support for extrapolated core, and static
    # model_core_extrapolated = cp.load_CHAOS_shcfile(chaos_files["core_extrapolated"])
    # model_static = cp.load_CHAOS_shcfile(chaos_files["static"])
    # #
    # Prepare inputs for ChaosMagPy
    # For simplicity, fix the epoch for evaluation at the midpoint
    t_mid = to_datetime(time[int(len(time) / 2)].values)
    epoch_mjd = cp.data_utils.mjd2000(t_mid.year, t_mid.month, t_mid.day)
    radius_km = radius / 1000
    theta = 90 - latitude
    phi = longitude
    # Evaluate the model
    B_radius, B_theta, B_phi = model_core.synth_values_tdep(
        time=epoch_mjd, radius=radius_km, theta=theta, phi=phi
    )
    # Convert to B_NEC
    return vstack((-B_theta, B_phi, -B_radius)).T


class LocalForwardMagneticModel(PalProcess):
    """Compute a magnetic model locally and append it to a datatree"""

    @property
    def process_name(self):
        return "LocalForwardMagneticModel"

    def set_config(self, dataset="SW_OPER_MAGA_LR_1B", model_descriptor="CHAOS-Core"):
        self.config = dict(dataset=dataset, model_descriptor=model_descriptor)

    def _call(self, datatree):
        subtree = datatree[f"{self.config.get('dataset')}"]
        ds = subtree.ds
        if self.config.get("model_descriptor") == "CHAOS-Core":
            B_NEC_Model = evaluate_chaos(
                longitude=ds["Longitude"],
                latitude=ds["Latitude"],
                radius=ds["Radius"],
                time=ds["Timestamp"],
            )
        else:
            raise NotImplementedError(
                f"Model descriptor {self.config.get('model_descriptor')} not implemented"
            )
        da = DataArray(
            data=B_NEC_Model,
            coords=ds["B_NEC"].coords,
            dims=ds["B_NEC"].dims,
        )
        da.attrs = {
            "units": "nT",
            "description": "Locally-computed forward model of the magnetic field",
        }
        ds = ds.assign(
            {
                f"B_NEC_{self.config.get('model_descriptor')}": da,
            }
        )
        # Assign dataset back into the datatree to return
        subtree = subtree.assign(ds)
        datatree[self.config.get("dataset")] = subtree
        return datatree
