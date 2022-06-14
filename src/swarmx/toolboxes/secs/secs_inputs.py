from apexpy import Apex
from numpy import deg2rad, sin, cos
from pyproj import CRS, Transformer

from swarmx.io import ExternalData


class SecsInputSingleSat(ExternalData):
    """Accessing external data: magnetic data from one satellite"""

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["B_NEC", "Flags_B"],
        "model": "'CHAOS-Core' + 'CHAOS-Static'",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }


class SecsInputs:
    """Accessing external data: inputs to SECS algorithm

    Examples
    --------
    >>> d = SecsInputs(
    >>>     start_time="2022-01-01", end_time="2022-01-02",
    >>>     model="'CHAOS-Core' + 'CHAOS-Static'",
    >>>     viresclient_kwargs=dict(asynchronous=True, show_progress=True)
    >>> )
    >>> d.s1.xarray  # Returns xarray of data from satellite 1 (Alpha)
    >>> d.s2.xarray  # Returns xarray of data from satellite 2 (Charlie)
    >>> d.s1.get_array("B_NEC")  # Returns numpy array
    """

    def __init__(
        self,
        spacecraft_pair="Alpha-Charlie",
        start_time=None,
        end_time=None,
        model="'CHAOS-Core' + 'CHAOS-Static'",
        viresclient_kwargs=None,
    ):
        if spacecraft_pair != "Alpha-Charlie":
            raise NotImplementedError("Only the Alpha-Charlie pair are configured")
        # Fetch external data
        inputs_1 = SecsInputSingleSat(
            collection="SW_OPER_MAGA_LR_1B",
            model=model,
            start_time=start_time,
            end_time=end_time,
            viresclient_kwargs=viresclient_kwargs,
        )
        inputs_2 = SecsInputSingleSat(
            collection="SW_OPER_MAGC_LR_1B",
            model=model,
            start_time=start_time,
            end_time=end_time,
            viresclient_kwargs=viresclient_kwargs,
        )
        # Store datasets as properties
        self.s1 = inputs_1
        self.s2 = inputs_2
        # Append Apex coordinates to each dataset
        for s in (self.s1, self.s2):
            mlat, mlon = self._calc_apex_coords_from_inputs(s)
            s.append_array("ApexLatitude", mlat, dims=("Timestamp",))
            s.append_array("ApexLongitude", mlon, dims=("Timestamp",))

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
        s_t = sin(deg2rad(90-lat))
        c_t = cos(deg2rad(90-lat))
        x = rad*c_p*s_t
        y = rad*s_p*s_t
        z = rad*c_t
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
        geod_lat, lon, height = SecsInputs._spherical_geocentric_to_geodetic(
            lat, lon, rad
        )
        mlat, mlon = A.convert(geod_lat, lon, 'geo', 'apex', height=height/1e3)
        return mlat, mlon

    @staticmethod
    def _calc_apex_coords_from_inputs(s):
        """Calculate Apex coordinates from SecsInputSingleSat

        Parameters
        ----------
        s: SecsInputSingleSat
            Object that contains a xarray.Dataset from a single satellite

        Returns
        -------
        mlat: ndarray
            Mangnetic apex latitude (degrees)
        mlon: ndarray
            Magnetic apex longitude (degrees)
        """
        # Get current date as datetime
        date = s.get_array("Timestamp")[0].astype('M8[ms]').astype('O')
        # Evaluate apex coordinates
        mlat, mlon = SecsInputs._calc_apex_coords(
            date,
            s.get_array("Latitude"),
            s.get_array("Longitude"),
            s.get_array("Radius"),
        )
        return mlat, mlon
