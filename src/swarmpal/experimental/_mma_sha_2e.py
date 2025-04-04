""""Experimental demo of MMA_SHA_2E 

See https://github.com/natgomezperez/MMA-2E"""

import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import xarray as xr

from swarmpal.io import PalProcess
from swarmpal.experimental.upstream_MMA_2E.MMA_2E.utils import configuration as MMA_2E_config
from swarmpal.experimental.upstream_MMA_2E.MMA_2E.qmatrix import estimate_SH_coefficients_1D as MMA_2E_estimate_SH_coefficients_1D
from swarmpal.experimental.upstream_MMA_2E.MMA_2E.utils.Coord_Trans import get_MagLat as MMA_2E_get_MagLat
from swarmpal.experimental._mjd2000 import mjd2000


class MMA_SHA_2E(PalProcess):
    @property
    def process_name(self):
        return "MMA_SHA_2E"

    def set_config(self, datasets=None, local_time_limit=6.0, max_gm_lat=60.0, min_gm_lat=0.0):
        self.config = dict(
            datasets=datasets,
            local_time_limit=local_time_limit,
            max_gm_lat=max_gm_lat,
            min_gm_lat=min_gm_lat,
        )
    

    @staticmethod
    def _extract_simplified_dataframe(ds):
        """Turn a dataset into the dataframe that the MMA code works with"""
        B_res_NEC = ds.swarmpal.magnetic_residual()
        return pd.DataFrame(
            {
                "t": mjd2000(ds["Timestamp"]),
                "r": ds["Radius"] / 1000,  # Rad in km
                "phi": ds["Longitude"],
                "theta": 90 - ds["Latitude"],  # Colatitude in degrees
                "B_rtp_1": -B_res_NEC.data[:, 2],  # Rad is -Center
                "B_rtp_2": -B_res_NEC.data[:, 0],  # Theta is -North
                "B_rtp_3": B_res_NEC.data[:, 1],
                "sat": ds["Spacecraft"],
                # 'MLT'    : ds["MLT"],
                "time": ds["Timestamp"],
            }
        )
    
    @staticmethod
    def _convert_longitude_to_local_time(longitude_deg, t):
        longitude_deg = (longitude_deg / 360) + (t % 1)
        return (longitude_deg % 1) * 24

    def _merge_and_select_data(self, datatree):
        """Merge the datasets and return a subselected dataframe for the MMA code to work with"""
        # Use the datasets specified in the config or all datasets in the datatree
        labels = self.config["datasets"]
        labels = labels if labels else datatree.keys()
        df = pd.concat(
            [self._extract_simplified_dataframe(datatree[label]) for label in labels],
            ignore_index=True,
        )
        # Local time and magnetic latitude masks from the process configuration
        lt_limit = self.config["local_time_limit"]
        min_gm_lat = self.config["min_gm_lat"]
        max_gm_lat = self.config["max_gm_lat"]
        # Apply the masks
        local_time = self._convert_longitude_to_local_time(df["phi"], df["t"])
        mask_lt = np.logical_or(local_time < lt_limit, local_time > (24 - lt_limit))
        lat_mag = MMA_2E_get_MagLat(90 - df.theta, df.phi, df.time)
        mask_maglat = np.logical_and(
            np.abs(lat_mag) >= min_gm_lat, np.abs(lat_mag) <= max_gm_lat
        )
        return df[np.logical_and(mask_lt, mask_maglat)]


    @staticmethod
    def _clean_data(df):
        threshold = 15
        outliers = pd.Series(data=False, index=df.index)
        for vi in range(1, 4):
            vec_c = "B_rtp_" + str(vi)
            z = np.abs(scipy_stats.zscore(df[vec_c]))
            outliers = (z > threshold) | outliers
        return df[~outliers]


    def _call(self, datatree):
        # Preprocess data
        df = self._clean_data(self._merge_and_select_data(datatree))
        # Perform analysis
        params = MMA_2E_config.BasicConfig()
        params.fullreset()
        ds = MMA_2E_estimate_SH_coefficients_1D(df, params)
        datatree["MMA_SHA_2E"] = xr.DataTree(ds)
        return datatree
