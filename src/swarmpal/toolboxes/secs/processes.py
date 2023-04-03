from apexpy import Apex
from datatree import DataTree
from numpy import cos, deg2rad, sin
from pyproj import CRS, Transformer

from swarmpal.io import PalProcess
from swarmpal.toolboxes.secs.dsecs_algorithms import _DSECS_steps

# from swarmpal.toolboxes.secs.dsecs_algorithms import dsecsdata


class DSECS_Process(PalProcess):
    """Provides the process for the DSECS algorithm for Alpha-Charlie

    Notes
    -----
    Expected config parameters:
    dataset_alpha
    dataset_charlie
    model_varname
    measurement_varname
    """

    @property
    def process_name(self):
        return "DSECS_Process"

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
        # Prepare for input to DSECS tools

        dsecs_output = _DSECS_steps(ds_alpha, ds_charlie)
        # SwA_list = get_eq(ds_alpha)  # lists of suitable segments
        # SwC_list = get_eq(ds_charlie)
        # Apply DSECS algorithm to each segment in turn
        # for SwA, SwC in zip(SwA_list, SwC_list):
        #    case = dsecsdata()
        #    case.populate(SwA, SwC)
        #    res, data, mapping = case.fit1D_df()
        #    break
        # Store outputs into the datatree
        for i, output in enumerate(dsecs_output):
            if output["current_densities"] is not None:
                datatree["output_" + str(i)] = DataTree(
                    data=output["current_densities"]
                )
                datatree["Fit_Alpha_" + str(i)] = DataTree(
                    data=output["magnetic_fit_Alpha"]
                )
                datatree["Fit_Charlie_" + str(i)] = DataTree(
                    data=output["magnetic_fit_Charlie"]
                )
        return datatree

    @staticmethod
    def _append_apex_coords(ds):
        date = ds["Timestamp"].data[0].astype("M8[ms]").astype("O")
        # Evaluate apex coordinates
        mlat, mlon = DSECS_Process._calc_apex_coords(
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
        geod_lat, lon, height = DSECS_Process._spherical_geocentric_to_geodetic(
            lat, lon, rad
        )
        mlat, mlon = A.convert(geod_lat, lon, "geo", "apex", height=height / 1e3)
        return mlat, mlon
