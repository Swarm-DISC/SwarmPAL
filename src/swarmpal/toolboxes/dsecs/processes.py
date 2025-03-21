from __future__ import annotations

from apexpy import Apex
from numpy import cos, deg2rad, sin
from pyproj import CRS, Transformer
from xarray import DataTree

from swarmpal.io import PalProcess
from swarmpal.toolboxes.dsecs.dsecs_algorithms import _DSECS_steps
from swarmpal.utils.exceptions import PalError


class Preprocess(PalProcess):
    """Prepare data for input to DSECS analysis"""

    @property
    def process_name(self):
        return "DSECS_Preprocess"

    def set_config(
        self,
        dataset_alpha: str = "SW_OPER_MAGA_LR_1B",
        dataset_charlie: str = "SW_OPER_MAGC_LR_1B",
    ):
        self._config = dict(
            dataset_alpha=dataset_alpha,
            dataset_charlie=dataset_charlie,
        )

    def _call(self, datatree):
        # Identify inputs for algorithm
        _alpha = self.config.get("dataset_alpha")
        _charlie = self.config.get("dataset_charlie")
        ds_alpha = datatree[_alpha].ds
        ds_charlie = datatree[_charlie].ds
        # Append ApexLatitude and ApexLongitude
        ds_alpha = self._append_apex_coords(ds_alpha)
        ds_charlie = self._append_apex_coords(ds_charlie)
        # Update datatree with the updated datasets
        datatree[_alpha] = datatree[_alpha].assign(ds_alpha)
        datatree[_charlie] = datatree[_charlie].assign(ds_charlie)
        return datatree

    @staticmethod
    def _append_apex_coords(ds):
        date = ds["Timestamp"].data[0].astype("M8[ms]").astype("O")
        # Evaluate apex coordinates
        mlat, mlon = Preprocess._calc_apex_coords(
            date,
            ds["Latitude"].data,
            ds["Longitude"].data,
            ds["Radius"].data,
        )
        ds = ds.assign(
            {
                "ApexLatitude": ("Timestamp", mlat),
                "ApexLongitude": ("Timestamp", mlon),
            }
        )
        return ds

    @staticmethod
    def _spherical_geocentric_to_geodetic(lat, lon, rad):
        """Convert from geocentric coordinates to geodetic lat/lon and height

        Parameters
        ----------
        lat : array_like
            Geocentric latitude (degrees)
        lon : array_like
            Geocentric longitude (degrees)
        rad : array_like
            Geocentric radius (metres)

        Returns
        -------
        lat: ndarray
            Geodetic latitude (degrees)
        lon: ndarray
            Geodetic longitude (degrees)
        alt: ndarray
            Geodetic altitude (metres)
        """
        # Convert to Cartesian x,y,z
        s_p = sin(deg2rad(lon))
        c_p = cos(deg2rad(lon))
        s_t = sin(deg2rad(90 - lat))
        c_t = cos(deg2rad(90 - lat))
        x = rad * c_p * s_t
        y = rad * s_p * s_t
        z = rad * c_t
        # Use PROJ to make the transformation
        ecef = CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84")
        lla = CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84")
        transformer = Transformer.from_crs(ecef, lla)
        lon, lat, alt = transformer.transform(x, y, z, radians=False)
        return lat, lon, alt

    @staticmethod
    def _calc_apex_coords(date, lat, lon, rad):
        """Calculate Apex coordinates

        Parameters
        ----------
        date: datetime
            Epoch date to use for Apex calculations
        lat : array_like
            Geocentric latitude (degrees)
        lon : array_like
            Geocentric longitude (degrees)
        rad : array_like
            Geocentric radius (metres)

        Returns
        -------
        mlat: ndarray
            Mangnetic apex latitude (degrees)
        mlon: ndarray
            Magnetic apex longitude (degrees)
        """
        A = Apex(date)
        geod_lat, lon, height = Preprocess._spherical_geocentric_to_geodetic(
            lat, lon, rad
        )
        mlat, mlon = A.convert(geod_lat, lon, "geo", "apex", height=height / 1e3)
        return mlat, mlon


def _get_dsecs_active_subtrees(datatree):
    """Returns the relevant subtrees (i.e. Alpha, Charlie)"""
    # Scan the tree based on previous preprocess application
    pal_processes_meta = datatree.swarmpal.pal_meta.get(".", {})
    dsecs_preprocess_meta = pal_processes_meta.get("DSECS_Preprocess")
    if not dsecs_preprocess_meta:
        raise PalError("Must first run dsecs.processes.Preprocess")
    _alpha = dsecs_preprocess_meta.get("dataset_alpha")
    _charlie = dsecs_preprocess_meta.get("dataset_charlie")
    return datatree[_alpha], datatree[_charlie]


class Analysis(PalProcess):
    """Run the DSECS analysis"""

    @property
    def process_name(self):
        return "DSECS_Analysis"

    def set_config(self):
        self._config = dict()

    def _call(self, datatree):
        # Identify inputs for algorithm
        dt_alpha, dt_charlie = _get_dsecs_active_subtrees(datatree)
        ds_alpha = dt_alpha.ds
        ds_charlie = dt_charlie.ds
        # Apply analysis
        dsecs_output = _DSECS_steps(ds_alpha, ds_charlie)
        # Store outputs into the datatree
        for i, output in enumerate(dsecs_output):
            if output["current_densities"] is not None:
                datatree[f"DSECS_output/{i}/currents"] = DataTree(
                    dataset=output["current_densities"]
                )
                datatree[f"DSECS_output/{i}/Fit_Alpha"] = DataTree(
                    dataset=output["magnetic_fit_Alpha"]
                )
                datatree[f"DSECS_output/{i}/Fit_Charlie"] = DataTree(
                    dataset=output["magnetic_fit_Charlie"]
                )
        return datatree
