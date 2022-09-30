import numpy as np
import datetime as dt
from swarmpal.toolboxes.secs import SecsInputs
import swarmpal.toolboxes.secs.aux_tools as auto
from swarmpal.toolboxes.secs.aux_tools import (
    sub_Swarm_grids,
    sph2sph,
    sub_FindLongestNonZero,
)

# from sub_fit_1D_DivFree import SwarmMag2J_test_fit_1D_DivFree
import xarray as xr

from swarmpal.toolboxes.secs.dsecs_algorithms import SECS_1D_DivFree_magnetic


def get_data_slices(
    t1=dt.datetime(2016, 3, 18, 11, 3, 0),
    t2=dt.datetime(2016, 3, 18, 11, 40, 0),
    model="IGRF",
):
    """_summary_

    Parameters
    ----------
    t1 : _type_, optional
        _description_, by default dt.datetime(2016, 3, 18, 11, 3, 0)
    t2 : _type_, optional
        _description_, by default dt.datetime(2016, 3, 18, 11, 40, 0)
    model : str, optional
        _description_, by default 'IGRF'

    Returns
    -------
    _type_
        _description_
    """

    inputs = SecsInputs(
        start_time=t1,
        end_time=t2,
        model="IGRF",
    )

    SwA = auto.get_eq(inputs.s1.xarray)

    SwC = auto.get_eq(inputs.s2.xarray)

    # SwA,SwC = getUnitVectors(SwA,SwC)

    return SwA, SwC


def analyze(dataA, dataC):

    return


def getUnitVectors(SwA, SwC):

    # Geographical components of unit vector in the main field direction
    # from sub_ReadMagDataLR, check again model and signs
    tmp = np.sqrt(
        SwA["B_NEC_Model"].sel(NEC="N") ** 2
        + SwA["B_NEC_Model"].sel(NEC="E") ** 2
        + SwA["B_NEC_Model"].sel(NEC="C") ** 2
    )
    SwA["ggUvT"] = -SwA["B_NEC_Model"].sel(NEC="N") / tmp  # south
    SwA["ggUvP"] = SwA["B_NEC_Model"].sel(NEC="E") / tmp  # east
    SwA["UvR"] = -SwA["B_NEC_Model"].sel(NEC="C") / tmp  # radial

    tmp = np.sqrt(
        SwC["B_NEC_Model"].sel(NEC="N") ** 2
        + SwC["B_NEC_Model"].sel(NEC="E") ** 2
        + SwC["B_NEC_Model"].sel(NEC="C") ** 2
    )
    SwC["ggUvT"] = -SwC["B_NEC_Model"].sel(NEC="N") / tmp  # south
    SwC["ggUvP"] = SwC["B_NEC_Model"].sel(NEC="E") / tmp  # east
    SwC["UvR"] = -SwC["B_NEC_Model"].sel(NEC="C") / tmp  # radial

    return SwA, SwC



class grid2D:
    def __init__(self):
        self.ggLat = np.array([])
        self.ggLon = np.array([])
        self.ggLat = np.array([])
        self.ggLon = np.array([])
        self.magLat = np.array([])
        self.magLon = np.array([])
        self.angle2D = np.array([])
        self.diff2lon2D = np.array([])
        self.diff2lat2D = np.array([])

    def create(
        self, lat1, lon1, lat2, lon2, dlat, lonRat, extLat, extLon, poleLat, poleLon
    ):
        """_summary_

        Parameters
        ----------
        lat1 : _type_
            _description_
        lon1 : _type_
            _description_
        lat2 : _type_
            _description_
        lon2 : _type_
            _description_
        dlat : _type_
            _description_
        lonRat : _type_
            _description_
        extLat : _type_
            _description_
        extLon : _type_
            _description_
        """
        self.ggLat, self.ggLon, self.angle2D, self.diff2lon2D, self.diff2lat2D = sub_Swarm_grids(
            lat1,
            lon1,
            lat2,
            lon2,
            dlat,
            lonRat,
            extLat,
            extLon,
        )

        self.magLat, self.magLon, _, _ = sph2sph(
            poleLat, poleLon, self.ggLat, self.ggLon, [], []
        )
        self.magLon = self.magLon % 360



class grid1D:
    def __init__(self):

        self.lat = np.array([])
        self.diff2 = np.array([])

    def create(self, lat1, lat2, dlat, extLat):
        """

        Parameters
        ----------
        lat1 : _type_
            _description_
        lat2 : _type_
            _description_
        dlat : _type_
            _description_
        extLat : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        self.lat, self.diff2 = auto.sub_Swarm_grids_1D(lat1, lat2, dlat, extLat)



class dsecsgrid:
    """ """
    def __init__(self):
        self.out = grid2D()
        self.secs2D = grid2D()
        self.secs1Ddf = grid1D()
        self.secs1Dcf = grid1D()
        self.outputlimitlat = 40
        self.Re =6371
        self.Ri = 6371 + 130
        self.poleLat = 90.0
        self.poleLon = 0.0
        self.dlatOut = 0.5  # resolution in latitude
        self.lonRatioOut = 2
        self.extLatOut = 0
        self.extLonOut = 3
        self.extLat1D = 1

    def FindPole(self, SwA):

        # define search grid
        dlat = 0.5  # latitude step [degree]
        minlat = 60  # latitude range
        maxlat = 90
        dlon = 5  # longitude step [degree]
        minlon = 0  # longitude range
        maxlon = 360 - 0.1 * dlon
        # create [Nlon,Nlat] matrices
        latP, lonP = np.mgrid[
            minlat : maxlat + dlat : dlat, minlon : maxlon + dlon : dlon
        ]
        latP = latP.flatten()
        lonP = lonP.flatten()

        # format error matrix
        errMat = np.full_like(latP, np.nan)

        # Could Limit the latitude range of the Swarm-A measurement points used in the optimization
        indA = np.nonzero(abs(SwA["Latitude"].data) < 100)

        # Loop over possible pole locations
        for n in range(len(latP)):
            # Rotate the main field unit vector at Swarm-A measurement points to the system
            # whose pole is at (latP(n), lonP(n)).
            lat, _, Bt, Bp = sph2sph(
                latP[n],
                lonP[n],
                SwA["Latitude"].data[indA],
                SwA["Longitude"].data[indA],
                -SwA["unit_B_NEC_Model"].sel(NEC="N").data[indA],
                SwA["unit_B_NEC_Model"].sel(NEC="E").data[indA],
            )
            Br = -SwA["unit_B_NEC_Model"].sel(NEC="C").data[indA]

            # Remove points that are very close to this pole location (otherwise 1/sin is problematic)
            ind = np.nonzero(abs(lat) < 89.5)
            lat = lat[ind]
            Bt = Bt[ind]  # theta=south
            Bp = Bp[ind]  # east
            Br = Br[ind]  # radial

            # Calculate unit vector in dipole system centered at (latP(n), lonP(n)).
            tmp = 1 + 3 * np.sin(np.radians(lat)) ** 2
            BxD = np.cos(np.radians(lat)) / tmp  # north
            ByD = np.zeros_like(lat)  # east
            BzD = 2 * np.sin(np.radians(lat)) / tmp  # down

            # Difference between the measured unit vectors and dipole unit vectors,
            # averaged over all points
            errMat[n] = np.nanmean((Bt + BxD) ** 2 + (Bp - ByD) ** 2 + (Br + BzD) ** 2)

        # Find pole location with minimum error
        ind = np.argmin(errMat)
        self.poleLat = latP[ind]
        self.poleLon = lonP[ind]

        ###skipped plotting routine here (see sub_FindPole.m), also flattened latP,lonP and errMat


    # self.poleLat, self.poleLon = dsecsgrid._sub_FindPole(SwA)

    def create(self, SwA, SwC):
        """
        Parameters
        ----------
        SwA : _type_
            _description_
        SwC : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        # Make grid around [-X,X] latitudes
        limitOutputLat = self.outputlimitlat
        ind = np.nonzero(abs(SwA["Latitude"].data) <= limitOutputLat)
        lat1 = SwA["Latitude"].data[ind]
        lon1 = SwA["Longitude"].data[ind]
        ind = np.nonzero(abs(SwC["Latitude"].data) <= limitOutputLat)
        lat2 = SwC["Latitude"].data[ind]
        lon2 = SwC["Longitude"].data[ind]

        # result should also be xarray
        # check with Heikki, sub_Swarm_grids from sub_Swarm_grids_2D.m returns more than 2 things
        self.out.create(
            lat1,
            lon1,
            lat2,
            lon2,
            self.dlatOut,
            self.lonRatioOut,
            self.extLatOut,
            self.extLonOut,
            self.poleLat,
            self.poleLon,
        )

        self.secs2D.create(
            SwA["Latitude"].data,
            SwA["Longitude"].data,
            SwC["Latitude"].data,
            SwC["Longitude"].data,
            self.dlatOut,
            self.lonRatioOut,
            self.extLatOut,
            self.extLonOut,
            self.poleLat,
            self.poleLon,
        )

        self.secs1Ddf.create(
            SwA["magLat"], SwC["magLat"], self.dlatOut, self.extLat1D
        )
        trackA = getLocalDipoleFPtrack(SwA["magLat"].data, SwA["Radius"].data*1e-3, self.Ri)
        trackC = getLocalDipoleFPtrack(SwC["magLat"].data, SwC["Radius"].data*1e-3, self.Ri)
        self.secs1Dcf.create(trackA, trackC, self.dlatOut, self.extLat1D)

        # self.ggLat1D,_ = auto.sub_Swarm_grids_1D(lat1,lat2,)


def getLocalDipoleFPtrack(latB, rB, Ri):
    """

    Parameters
    ----------
    SwA : _type_
        _description_
    SwC : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    # Use the LOCAL DIPOLE footpoints in grid construction
    thetaB = (90 - latB) / 180 * np.pi
    thetaFP = np.arcsin(
        np.sqrt(Ri / rB) * np.sin(thetaB)
    )  # co-latitude of the footpoints, mapped to the northern hemisphere
    ind = np.argwhere(thetaB > np.pi / 2)
    thetaFP[ind] = np.pi - thetaFP[ind]
    latFP = 90 - thetaFP / np.pi * 180
    track = np.arange(np.min(np.abs(latFP)), np.max(1 + np.abs(latFP)), 0.2)

    return track


def mag_transform_dsecs(SwA, SwC, pole_lat, pole_lon):
    """_summary_

    Parameters
    ----------
    SwA : _type_
        _description_
    SwC : _type_
        _description_
    pole_lat : _type_
        _description_
    pole_lon : _type_
        _description_
    """

    _, _, auvt, auvp = sph2sph(
        pole_lat,
        pole_lon,
        SwA["Latitude"].data,
        SwA["Longitude"].data,
        SwA["ggUvT"].data,
        SwA["ggUvP"].data,
    )  # unit vector along SwA magnetic field
    _, _, cuvt, cuvp = sph2sph(
        pole_lat,
        pole_lon,
        SwC["Latitude"].data,
        SwC["Longitude"].data,
        SwC["ggUvT"].data,
        SwC["ggUvP"].data,
    )  # unit vector along SwC magnetic field
    amaglat, amaglon, amagbt, amagbp = sph2sph(
        pole_lat,
        pole_lon,
        SwA["Latitude"].data,
        SwA["Longitude"].data,
        -SwA["B_NEC"].sel(NEC="N").data,
        SwA["B_NEC"].sel(NEC="E").data,
    )  # SwA locations & data
    cmaglat, cmaglon, cmagbt, cmagbp= sph2sph(
        pole_lat,
        pole_lon,
        SwC["Latitude"].data,
        SwC["Longitude"].data,
        -SwC["B_NEC"].sel(NEC="N").data,
        SwC["B_NEC"].sel(NEC="E").data,
    ) 
    
    
    SwA=SwA.assign({"magUvT":("Timestamp",auvt), "magUvP":("Timestamp",auvp),
     "magLat" : ("Timestamp",amaglat), "magLon" :("Timestamp",amaglon), "magBt":("Timestamp",amagbt),
      "magBp" : ("Timestamp",amagbt)})
    SwC=SwC.assign({"magUvT":("Timestamp",cuvt), "magUvP":("Timestamp",cuvp),
     "magLat" : ("Timestamp",cmaglat), "magLon" :("Timestamp",cmaglon), "magBt":("Timestamp",cmagbt),
      "magBp" : ("Timestamp",cmagbt)})

    # SwC locations & data
    SwA["magLon"] = SwA["magLon"] % 360
    SwC["magLon"] = SwC["magLon"] % 360

    return SwA, SwC


def trim_data(SwA, SwC):
    """ """

    Vx = np.gradient(SwA["magLat"])  # northward velocity [deg/step]
    Vy = np.gradient(SwA["magLon"]) * np.cos(
        np.radians(SwA["magLat"])
    )  # eastward velocity [deg/step]
    _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
    SwA = SwA.sel(Timestamp=SwA["Timestamp"][ind])

    Vx = np.gradient(SwC["magLat"])
    Vy = np.gradient(SwC["magLon"]) * np.cos(np.radians(SwC["magLat"]))
    _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))

    ### replace with indexing whole dataset
    SwC = SwC.sel(Timestamp=SwC["Timestamp"][ind])

    return SwA, SwC


def sub_load_real_data(result):

    # Radius of the Earth and radius of the ionospheric current layer
    result["Re"] = 6371  # default 6371 km
    result["Ri"] = result["Re"] + 130  # default Re+130
    ###Not needed because we already have apex coordinates
    # FPalt = result["Ri"] - result["Re"]

    # Select the latitude range abs(lat) <= limit where the satellite data is taken.
    limitSwarmLat = 60

    # Get Swarm A and C data
    inputs = SecsInputs(
        start_time=dt.datetime(2016, 3, 18, 11, 3, 0),
        end_time=dt.datetime(2016, 3, 18, 11, 40, 0),
        model="IGRF",
    )

    SwA = inputs.s1.xarray
    SwA = SwA.where(abs(SwA["Latitude"]) <= limitSwarmLat, drop=True)

    SwC = inputs.s2.xarray
    SwC = SwC.where(abs(SwA["Latitude"]) <= limitSwarmLat, drop=True)

    # Define the output grid.
    DlatOut = 0.5  # resolution in latitude
    LonRatioOut = 2  # ratio (satellite separation)/(grid resolution) in longitude
    ExtLatOut = 0  # Number of points to extend outside satellite data area in latitude
    ExtLonOut = 3  # Number of points to extend outside satellite data area in longitude

    # Make grid around [-X,X] latitudes
    limitOutputLat = 40
    ind = np.nonzero(abs(SwA["Latitude"].data) <= limitOutputLat)
    lat1 = SwA["Latitude"].data[ind]
    lon1 = SwA["Longitude"].data[ind]
    ind = np.nonzero(abs(SwC["Latitude"].data) <= limitOutputLat)
    lat2 = SwC["Latitude"].data[ind]
    lon2 = SwC["Longitude"].data[ind]

    ###result should also be xarray
    ###check with Heikki, sub_Swarm_grids from sub_Swarm_grids_2D.m returns more than 2 things
    result["ggLat"], result["ggLon"], _, _, _ = sub_Swarm_grids(
        lat1, lon1, lat2, lon2, DlatOut, LonRatioOut, ExtLatOut, ExtLonOut
    )
    # Transpose into [Nlon,Nlat] matrices
    ### need to add this to dataarray somehow
    result["ggLat"] = np.transpose(result["ggLat"])
    result["ggLon"] = np.transpose(result["ggLon"])

    return result, SwA, SwC


def sub_rotate(result, SwA, SwC, suunta):

    latP = result["PoleLat"]
    lonP = result["PoleLon"]

    if suunta == "geo2mag":
        # Rotate input coordinates and horizontal part of input vectors to the local dipole system.
        result["magLat"], result["magLon"], _, _ = sph2sph(
            latP, lonP, result["ggLat"], result["ggLon"], [], []
        )  # output grid
        result["magLon"] = result["magLon"] % 360
        _, _, SwA["magUvT"], SwA["magUvP"] = sph2sph(
            latP,
            lonP,
            SwA["Latitude"].data,
            SwA["Longitude"].data,
            SwA["ggUvT"].data,
            SwA["ggUvP"].data,
        )  # unit vector along SwA magnetic field
        _, _, SwC["magUvT"], SwC["magUvP"] = sph2sph(
            latP,
            lonP,
            SwC["Latitude"].data,
            SwC["Longitude"].data,
            SwC["ggUvT"].data,
            SwC["ggUvP"].data,
        )  # unit vector along SwC magnetic field
        SwA["magLat"], SwA["magLon"], SwA["magBt"], SwA["magBp"] = sph2sph(
            latP,
            lonP,
            SwA["Latitude"].data,
            SwA["Longitude"].data,
            SwA["B_NEC"].sel(NEC="N").data,
            SwA["B_NEC"].sel(NEC="E").data,
        )  # SwA locations & data
        SwC["magLat"], SwC["magLon"], SwC["magBt"], SwC["magBp"] = sph2sph(
            latP,
            lonP,
            SwC["Latitude"].data,
            SwC["Longitude"].data,
            SwC["B_NEC"].sel(NEC="N").data,
            SwC["B_NEC"].sel(NEC="E").data,
        )  # SwC locations & data
        SwA["magLon"] = SwA["magLon"] % 360
        SwC["magLon"] = SwC["magLon"] % 360

        # Our analysis requires that satellites move mostly in the north/south direction (problems in grid generation etc),
        # but that may not be the case in the magnetic coordinate system. Solve this by selecting latitude range where satellite's
        # eastward velocity is sufficiently small.
        # First Swarm-A
        Vx = np.gradient(SwA["magLat"])  # northward velocity [deg/step]
        Vy = np.gradient(SwA["magLon"]) * np.cos(
            np.radians(SwA["magLat"])
        )  # eastward velocity [deg/step]
        _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
        # SwA["apexLat"] = SwA["apexLat"][ind]
        # SwA.r=SwA.r(ind);      SwA.ggLat=SwA.ggLat(ind);  SwA.ggLon=SwA.ggLon(ind);  SwA.dn=SwA.dn(ind);
        # SwA.Br=SwA.Br(ind);    SwA.ggBt=SwA.ggBt(ind);    SwA.ggBp=SwA.ggBp(ind);    SwA.Bpara=SwA.Bpara(ind);
        # SwA.UvR=SwA.UvR(ind);  SwA.ggUvT=SwA.ggUvT(ind);  SwA.ggUvP=SwA.ggUvP(ind);
        # SwA.magLat=SwA.magLat(ind);      SwA.magLon=SwA.magLon(ind);
        # SwA.magBt=SwA.magBt(ind);        SwA.magBp=SwA.magBp(ind);
        # SwA.magUvT=SwA.magUvT(ind);      SwA.magUvP=SwA.magUvP(ind);
        ### replace with indexing whole dataset
        SwA = SwA.sel(Timestamp=SwA["Timestamp"][ind])

        # Then Swarm-C
        Vx = np.gradient(SwC["magLat"])
        Vy = np.gradient(SwC["magLon"]) * np.cos(np.radians(SwC["magLat"]))
        _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
        # SwC.apexLat=SwC.apexLat(ind);
        # SwC.r=SwC.r(ind);      SwC.ggLat=SwC.ggLat(ind);  SwC.ggLon=SwC.ggLon(ind);  SwC.dn=SwC.dn(ind);
        # SwC.Br=SwC.Br(ind);    SwC.ggBt=SwC.ggBt(ind);    SwC.ggBp=SwC.ggBp(ind);    SwC.Bpara=SwC.Bpara(ind);
        # SwC.UvR=SwC.UvR(ind);  SwC.ggUvT=SwC.ggUvT(ind);  SwC.ggUvP=SwC.ggUvP(ind);
        # SwC.magLat=SwC.magLat(ind);      SwC.magLon=SwC.magLon(ind);
        # SwC.magBt=SwC.magBt(ind);        SwC.magBp=SwC.magBp(ind);
        # SwC.magUvT=SwC.magUvT(ind);      SwC.magUvP=SwC.magUvP(ind);
        ### replace with indexing whole dataset
        SwC = SwC.sel(Timestamp=SwC["Timestamp"][ind])

    elif suunta == "mag2geo":
        # Rotate analysis results back to geographic.
        lat = result["magLat"].data
        lon = result["magLon"].data
        _, _, result["df1dGgJt"], result["df1dGgJp"] = sph2sph(
            latP, 0, lat, lon, result["df1dMagJt"].data, result["df1dMagJp"].data
        )  # J from 1D DF SECS
        _, _, result["df2dGgJt"], result["df2dGgJp"] = sph2sph(
            latP, 0, lat, lon, result["df2dMagJt"].data, result["df2dMagJp"].data
        )  # J from 2D DF SECS
        _, _, result["cf1dDipGgJt"], result["cf1dDipGgJp"] = sph2sph(
            latP, 0, lat, lon, result["cf1dDipMagJt"].data, result["cf1dDipMagJp"].data
        )  # J from dipolar 1D CF SECS
        _, _, result["cf2dDipGgJt"], result["cf2dDipGgJp"] = sph2sph(
            latP, 0, lat, lon, result["cf2dDipMagJt"].data, result["cf2dDipMagJp"].data
        )  # J from dipolar 2D CF SECS
        _, _, result["remoteCf2dDipGgJt"], result["remoteCf2dDipGgJp"] = sph2sph(
            latP,
            0,
            lat,
            lon,
            result["remoteCf2dDipMagJt"].data,
            result["remoteCf2dDipMagJp"].data,
        )  # J from remote dipolar 2D CF SECS

        lat = SwA["magLat"].data
        lon = SwA["magLon"].data
        _, _, SwA["df1dGgBt"], SwA["df1dGgBp"] = sph2sph(
            latP, 0, lat, lon, SwA["df1dMagBt"].data, SwA["df1dMagBp"].data
        )  # B at SwA from 1D DF SECS
        _, _, SwA["df2dGgBt"], SwA["df2dGgBp"] = sph2sph(
            latP, 0, lat, lon, SwA["df2dMagBt"].data, SwA["df2dMagBp"].data
        )  # B at SwA from 2D DF SECS
        _, _, SwA["cf1dDipGgBt"], SwA["cf1dDipGgBp"] = sph2sph(
            latP, 0, lat, lon, SwA["cf1dDipMagBt"].data, SwA["cf1dDipMagBp"].data
        )  # B at SwA from dipolar 1D CF SECS
        _, _, SwA["cf2dDipGgBt"], SwA["cf2dDipGgBp"] = sph2sph(
            latP, 0, lat, lon, SwA["cf2dDipMagBt"].data, SwA["cf2dDipMagBp"].data
        )  # B at SwA from dipolar 2D CF SECS
        _, _, SwA["remoteCf2dDipGgBt"], SwA["remoteCf2dDipGgBp"] = sph2sph(
            latP,
            0,
            lat,
            lon,
            SwA["remoteCf2dDipMagBt"].data,
            SwA["remoteCf2dDipMagBp"].data,
        )  # B at SwA from remote dipolar 2D CF SECS

        lat = SwC["magLat"].data
        lon = SwC["magLon"].data
        _, _, SwC["df1dGgBt"], SwC["df1dGgBp"] = sph2sph(
            latP, 0, lat, lon, SwC["df1dMagBt"].data, SwC["df1dMagBp"].data
        )
        _, _, SwC["df2dGgBt"], SwC["df2dGgBp"] = sph2sph(
            latP, 0, lat, lon, SwC["df2dMagBt"].data, SwC["df2dMagBp"].data
        )
        _, _, SwC["cf1dDipGgBt"], SwC["cf1dDipGgBp"] = sph2sph(
            latP, 0, lat, lon, SwC["cf1dDipMagBt"].data, SwC["cf1dDipMagBp"].data
        )
        _, _, SwC["cf2dDipGgBt"], SwC["cf2dDipGgBp"] = sph2sph(
            latP, 0, lat, lon, SwC["cf2dDipMagBt"].data, SwC["cf2dDipMagBp"].data
        )
        _, _, SwC["remoteCf2dDipGgBt"], SwC["remoteCf2dDipGgBp"] = sph2sph(
            latP,
            0,
            lat,
            lon,
            SwC["remoteCf2dDipMagBt"].data,
            SwC["remoteCf2dDipMagBp"].data,
        )

    else:
        raise ValueError("Direction must be either geo2mag or mag2geo.")

    return result, SwA, SwC



class dsecsdata:

    def __init__(self):
        self.lonB = np.array([])
        self.latB = np.array([])
        self.rB = np.array([])
        self.Bt = np.array([])
        self.Bp = np.array([])
        self.Br = np.array([])
        self.Bpara = np.array([])
        self.uvR = np.array([])
        self.uvT = np.array([])
        self.uvP = np.array([])
        self.grid: dsecsgrid = dsecsgrid()
        self.alpha: float = 1e-5
        self.epsSVD: float = 4e-4
    

    def populate(self, SwA, SwC, grid):
        self.latB = np.concatenate((SwA["magLat"], SwC["magLat"]))
        self.lonB = np.concatenate((SwA["magLon"], SwC["magLon"]))
        self.rB = np.concatenate((SwA["Radius"], SwC["Radius"]))*1e-3
        B = np.concatenate((SwA["B_NEC_res"], SwC["B_NEC_res"]))
        self.Bt = -B[:, 0]
        self.Bp = B[:, 1]
        self.Br = -B[:, 2]
        self.Bpara = np.concatenate((SwA["B_para_res"], SwC["B_para_res"]))
        self.grid = grid
        self.uvR = np.concatenate((SwA["UvR"],SwC["UvR"]))
        self.uvT = np.concatenate((SwA["magUvT"],SwC["magUvT"]))
        self.uvP = np.concatenate((SwA["magUvP"],SwC["magUvP"]))
        

    def fit1D(self):

        matBr, matBt = SECS_1D_DivFree_magnetic(
            self.latB, self.grid.secs1Ddf.lat, self.rB, self.grid.Ri, 500
        )

        N1d = len(self.grid.secs1Ddf.lat)
        print(N1d)
        y = self.Bpara #measurement, parallel magnetic field
        print(matBr.shape)
        print(self.uvR.shape)
        A = (matBr.T * self.uvR).T + (matBt.T * self.uvT).T
        regmat = self.grid.secs1Ddf.diff2 
        x = auto.sub_inversion(A,regmat,self.epsSVD,self.alpha,y)
        return x,y,A



#
#def test_1D_DF_fit(SwA, SwC, grid):
#
#    data = dsecsdata()
#    data.populate(SwA,SwC, grid)
#
#    # Fit 1D DF SECS
#    SwA, SwC, result = SwarmMag2J_test_fit_1D_DivFree(SwA, SwC, result)
#

# Swarm_B2J_real_analyze()
