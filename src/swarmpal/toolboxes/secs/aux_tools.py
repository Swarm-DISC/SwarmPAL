import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def sph2sph(latP, lonP, latOld, lonOld, VtOld, VpOld):
    """Change coordinates and (horizontal part of) vectors from one spherical
        coordinate system to another.

    Parameters
    ----------
    latP, lonP : float
        Latitude and longitude of the new pole in the old coordinates, [deg]
    latOld, lonOld : array
        Latitudes and longitudes of the data points to be transformed in the
        old coordinate system, [deg]
    VtOld, VpOld : array
        Theta- and phi-components of a vector field at locations
        (latOld,lonOld). If you want to transform only coordinates make these
        empty  arrays (np.array([])).

    Returns
    -------
    latNew, lonNew : array
        Coordinates of the points (latOld,lonOld) in the new system, [deg],
        same size as input parameters. Note that the old pole has longitude 0
        and longitude is between 0 and 360.
    VtNew, VpNew : array
        Theta- and phi-components of the vector field in the new system. If no
        input vector field was given, these are empty vectors [].
    """

    # If pole of the new coordinate system is NaN, return NaN
    if np.any(np.isnan([latP, lonP])):
        latNew = np.full_like(latOld, np.nan)
        lonNew = np.full_like(lonOld, np.nan)
        VtNew = np.full_like(VtOld, np.nan)
        VpNew = np.full_like(VpOld, np.nan)

        return latNew, lonNew, VtNew, VpNew

    # If pole of the new coordinate system is close enough to the old one, do
    # nothing
    if 90.0 - latP < 0.01:
        return latOld, lonOld, VtOld, VpOld

    # latitude --> co-latitude and deg --> rad
    thetaP = np.radians(90 - latP)
    phiP = np.radians(lonP)
    thetaO = np.radians(90 - latOld)
    phiO = np.radians(lonOld)

    # Latitude of the data points in the new system
    cosThetaPrime = np.cos(thetaO) * np.cos(thetaP) + np.sin(thetaO) * np.sin(
        thetaP
    ) * np.cos(phiO - phiP)
    latNew = 90 - np.degrees(np.arccos(cosThetaPrime))

    # Find those data points that are NOT too close to the new coordinate pole.
    # Points too close will get NaN in all other output parameters except
    # latNew, which is set to 90.
    # Except those points that have latOld or lonOld NaN, they are put to NaN.
    ind = (90 - latNew) > 0.01
    latNew[~ind] = 90
    ii = np.isnan(latOld + lonOld)
    latNew[ii] = np.nan
    sinThetaPrime = np.full_like(latOld, np.nan)  # np.nan + latOld
    # this affects all other output variables
    sinThetaPrime[ind] = np.sqrt(1 - cosThetaPrime[ind] ** 2.0)

    # Longitude of the data points in the new system
    # [old pole has longitude 0]
    sinA = np.sin(thetaO) * np.sin(phiP - phiO) / sinThetaPrime
    cosA = (np.cos(thetaO) - np.cos(thetaP) * cosThetaPrime) / (
        np.sin(thetaP) * sinThetaPrime
    )

    lonNew = np.degrees(np.arctan2(sinA, cosA))
    lonNew[lonNew < 0] += 360

    # Horizontal vector components in the new system
    if len(VtOld) == len(latOld):
        sinC = np.sin(thetaP) * np.sin(phiP - phiO) / sinThetaPrime
        cosC = (np.cos(thetaP) - np.cos(thetaO) * cosThetaPrime) / (
            np.sin(thetaO) * sinThetaPrime
        )
        VtNew = cosC * VtOld - sinC * VpOld
        VpNew = cosC * VpOld + sinC * VtOld
    else:
        # return empty variables
        VtNew = np.array([])
        VpNew = np.array([])

    return latNew, lonNew, VtNew, VpNew


def sub_FindLongestNonZero(x):
    """Find the longest sequence of consecutive non-zero elements in an array.
        Return the sequence and indices as y=x[ind].

    Parameters
    ----------
    x : array
        Input array

    Returns
    -------
    y : array
        Longest sequence of consecutive non-zero elements
    ind : array
        Indices of the sequence
    """

    # pad with zeros and find zero elements
    x_aux = np.pad(x, (1, 1), "constant", constant_values=0)
    zpos = np.nonzero(x_aux == 0)[0]

    # Find the longest jump in zero positions
    grpidx = np.argmax(np.diff(zpos))

    # indices and values in the longest sequence
    ind = np.arange(zpos[grpidx], zpos[grpidx + 1] - 1)
    y = x[ind]

    return y, ind


def sub_inversion(secsMat, regMat, epsSVD, alpha, magVec):
    """Solve matrix equation sysMat*resultVec = dataVec, using truncated
        singular value decomposition.

    Parameters
    ----------
    secsMat : ndarray
        Matrix giving the magnetic field from SECS amplitudes
    regMat : ndarray
        Regularization matrix for the SECS amplitudes
    epsSVD : float
        Pre-defined truncation parameter so that all singular values smaller
        than epsSVD*(largest singular value) will be ignored
    alpha : float
        Parameter scaling the amount of regularization
    magVec : array
        Magnetic measurements to be fitted

    Returns
    -------
    result_Vec : array
        Vector of SECS amplitudes
    """

    # Combine into one matrix and add zero constraint on the data vector
    if regMat.size == 0 or np.isnan(alpha):
        sysMat = secsMat
        dataVec = magVec
    else:
        sysMat = np.concatenate((secsMat, alpha * regMat))

        dataVec = np.concatenate((magVec, np.zeros(regMat.shape[0])))

    # Calculate SVD
    logger.info(
        f"\n  Calculating SVD of a [{sysMat.shape[0]},{sysMat.shape[1]}] " "matrix ... "
    )
    # works for 3x3 matrix, check what input matrix is and check again
    svdU, svdS, svdVh = np.linalg.svd(sysMat, full_matrices=False)
    svdV = svdVh.T
    logger.info("done\n")

    # Calculate the inverse matrix
    lkmS = len(svdS)
    slim = epsSVD * svdS[0]
    ss = 1.0 / svdS
    ind = np.nonzero(svdS <= slim)
    ss[ind] = 0

    logger.info(
        f"epsilon = {epsSVD}, singular values range from {svdS[0]} to "
        f"{svdS[lkmS - 1]} \n"
    )
    logger.info(
        f"--> {len(ind)} values smaller than {slim} deleted (of {lkmS} " "values)\n\n"
    )

    # Calculate the result vector
    resultVec = svdU.conj().T @ dataVec

    # works with test example but should check again with real data
    resultVec = np.diagflat(ss)[:lkmS, :lkmS] @ resultVec

    resultVec = svdV @ resultVec

    return resultVec


def sub_LonInterp(lat, lon, intLat, method, ekstra=np.nan):
    """Robust interpolation of longitude by interpolating the sin and cos
    of the angle. This way a 360 deg jump does not matter.

    Parameters
    ----------
    lat, lon : array
        original coordinates [degree]
    intLat : array
        Latitudes where lon is interpolated to [degree]
    method : str
        interpolation method

    Returns
    -------
    intLon : array
        Interpolated longitudes at latitudes intLat [degree],
        NOTE: -180 <= intlon <= 180
    """

    phi = np.radians(lon)

    # check if this is always used with extrapolation
    # interpolate sin and cos
    fC = interp1d(lat, np.cos(phi), kind=method, fill_value=ekstra, bounds_error=False)
    fS = interp1d(lat, np.sin(phi), kind=method, fill_value=ekstra, bounds_error=False)

    intC = fC(intLat)
    intS = fS(intLat)

    # calculate interpolated longitude and convert to degree.
    # NOTE:  -180 <= intlon <= 180
    intLon = np.degrees(np.arctan2(intS, intC))

    return intLon


def sub_Swarm_grids_1D(lat1, lat2, Dlat1D, ExtLat1D):
    """Make 1D grid around the 2 Swarm satellite paths.

    Parameters
    ----------
    lat1, lat2 : array
        Geographic latitudes of the satellites' paths, [degree]
    Dlat1D : int or float
        1D grid spacing in latitudinal direction, [degree]
    ExtLat1D : int or float
        Number of points to extend the 1D grid outside data area in latitudinal
        direction

    Returns
    -------
    lat1D : array
        Geographic latitudes of the 1D grid
    mat1Dsecond : ndarray
        2nd gradient matrix
    """

    # Latitudinal extent of the satellite data, only that part where there is
    # data from both satellites.
    maxlat = min(np.nanmax(lat1), np.nanmax(lat2))
    minlat = max(np.nanmin(lat1), np.nanmin(lat2))

    # Create 1D grid.
    # Limit grid to latitudes -89 < lat2Dsecs < 89.
    lat1D = np.arange(
        minlat - ExtLat1D * Dlat1D, maxlat + (ExtLat1D + 1) * Dlat1D, Dlat1D
    )
    ind = np.nonzero((lat1D > -89) & (lat1D < 89))
    lat1D = lat1D[ind]

    # Create 2nd gradient matrix
    apu = np.ones(len(lat1D))
    mat1Dsecond = np.diag(apu[1:], -1) - 2 * np.diag(apu) + np.diag(apu[1:], 1)

    # [unit 1/deg^2]  (NOTE: not quite same as north component of nabla^2)
    mat1Dsecond = mat1Dsecond[1:-1, :] / Dlat1D**2

    return lat1D, mat1Dsecond


def sub_Swarm_grids(lat1, lon1, lat2, lon2, Dlat2D, LonRatio, ExtLat2D, ExtLon2D):
    """Make 2D grids around the 2 Swarm satellite paths.

    Parameters
    ----------
    lat1, lat2 : array
        Geographic latitudes of the satellites's paths, [degree]
    lon1, lon2 : array
        Geographic longitudes of the satellites's paths, [degree]
    Dlat2D : float
        2D grid spacing in latitudinal direction, [degree]
    LonRatio : float
        Ratio between the the satellite separation and longitudinal spacing of
        the 2D grid, lonRatio=mean(lon1,lon2)/dlon at each latitude separately
    ExtLat2D, ExtLon2D : int
        Number of points to extend the 2D grid outside data area in latitudinal
        and longitudinal directions

    Returns
    -------
    lat2D, lon2D : ndarray
        Geographic latitudes and longitudes of the 2D grid
    angle2D : ndarray
         Half-angle of such a spherical cap that has same area as the 2D grid
         cell [radian]
    dLon2D : ndarray
        _description_
    mat2DsecondLat : ndarray
        2D second gradient matrix in latitude
    _type_
        _description_
    """

    # Latitudinal extent of the satellite data, only that part where there is
    # data from both satellites.
    maxlat = min(np.nanmax(lat1), np.nanmax(lat2))
    minlat = max(np.nanmin(lat1), np.nanmin(lat2))

    # Average satellite path, and the satellite spacing in longitudinal direction.
    ind1 = np.nonzero((lat1 >= minlat) & (lat1 <= maxlat))
    # ind2 = np.nonzero((lat2 >= minlat) & (lat2 <= maxlat))
    latA = lat1[ind1]
    lonEro = sub_LonInterp(lat2, lon2, lat1[ind1], "linear", "extrapolate") - lon1[ind1]
    lonEro[lonEro > 180] -= 360
    lonEro[lonEro < -180] += 360
    lonA = lon1[ind1] + lonEro / 2

    # Latitudes of the 2D grid.
    # Limit grids to latitudes -89 < lat2Dsecs < 89.
    apulat = np.arange(
        (minlat - ExtLat2D * Dlat2D), (maxlat + (ExtLat2D + 1) * Dlat2D), Dlat2D
    )
    ind = np.nonzero((apulat > -89) & (apulat < 89))
    apulat = apulat[ind]

    # Number of points in the 2D grid
    Nlat = len(apulat)
    Nlon = 1 + 2 * ExtLon2D

    # Preformat the 2D variables.
    lat2D = np.full((Nlat, Nlon), np.nan)
    lon2D = np.full((Nlat, Nlon), np.nan)
    angle2D = np.full((Nlat, Nlon), np.nan)
    dLon2D = np.full((Nlat, Nlon), np.nan)

    # Make the 2D grid
    for n in range(Nlat):
        lat2D[n, :] = apulat[n]

        # Average satellite longitude at this latitude, and the
        # grid spacing in longitudinal direction
        apulona = sub_LonInterp(latA, lonA, apulat[n], "linear", "extrapolate")

        # check interpolation
        apuf = interp1d(latA, lonEro, kind="linear", fill_value="extrapolate")
        apulonero = apuf(apulat[n])
        # apulonero = interp1(latA,lonEro,apulat[n],'linear','extrap')

        Dlon = np.abs(apulonero) / LonRatio

        # 2D longitudes at this latitude
        lon2D[n, :] = (
            apulona + np.arange(-(Nlon - 1) / 2.0, (Nlon - 1) / 2.0 + 1.0) * Dlon
        )
        dLon2D[n, :] = Dlon

        # Half-angle of such a spherical cap that has same area as the 2D grid
        # cells at this latitude
        theta = np.radians(90 - apulat[n])
        angle2D[n, :] = np.arccos(
            1
            - Dlon
            / 360
            * (
                np.cos(theta - Dlat2D / 360 * np.pi)
                - np.cos(theta + Dlat2D / 360 * np.pi)
            )
        )

    # Make 2D second gradient matrix in latitude.
    # NOTE: it might be more accurate to make the gradient matric in along-track direction, not latitude.
    # Here we assume that poles are listed along latitude at each longitude.
    # i.e. [(lat1,lon1) (lat2,lon1) ... (latN,lon1) (lat1,lon2) ...]
    # This corresponds to lat2D(:) according to the above construction
    apu = np.ones(Nlat)

    mat2DsecondLat = np.zeros(((Nlat - 2) * Nlon, Nlon * Nlat))
    apumat = np.diag(apu[1:], -1) - 2 * np.diag(apu) + np.diag(apu[1:], 1)
    # This gives 2nd deriv. for one column of lat2D
    apumat = apumat[1:-1, :] / Dlat2D**2

    for n in range(Nlon):
        i1 = np.arange(Nlat - 2) + (n - 1) * (Nlat - 2)
        i2 = np.arange(Nlat) + (n - 1) * Nlat
        i1, i2 = np.ix_(i1, i2)
        mat2DsecondLat[i1, i2] = apumat

    return lat2D, lon2D, angle2D, dLon2D, mat2DsecondLat


def get_eq(ds, QD_filter_max=60):
    """Splits data into a list of pieces suitable for DSECS analysis latitude.

    Parameters
    ----------
    ds : _type_
        _description_
    QD_filter_max : int, optional
        _description_, by default 60

    Returns
    -------
    _type_
        _description_
    """
    # mask= (np.abs(ds.QDLat) > QD_filter_max) | (np.abs(ds.QDLat) < QD_filter_min)
    mask = np.abs(ds.QDLat) > QD_filter_max

    # ovals=np.ma.flatnotmasked_contiguous(np.ma.masked_array(mask,mask=mask))
    # return ovals,ds
    ovals = np.ma.flatnotmasked_contiguous(np.ma.masked_array(mask, mask=mask))
    # return ovals,ds
    out = []
    for d in ovals:
        out.append(ds.isel(Timestamp=d))
        out[-1] = out[-1].assign(unit_B_NEC_Model=_normalizev(out[-1]["B_NEC_Model"]))
        out[-1] = out[-1].assign(
            {
                "B_para_res": (
                    "Timestamp",
                    np.einsum(
                        "ij,ij->i",
                        out[-1]["B_NEC"] - out[-1]["B_NEC_Model"],
                        out[-1]["unit_B_NEC_Model"],
                    ),
                )
            }
        )
        out[-1] = out[-1].assign(B_NEC_res=out[-1]["B_NEC"] - out[-1]["B_NEC_Model"])
    return out


def _normalizev(v):
    """Creates an unit vector.

    Parameters
    ----------
    v : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return (v.T / np.linalg.norm(v, axis=1)).T


def sub_points_along_fieldline(thetaSECS, Rsecs, L, minD):
    """Calculate points to be used in the Biot-Savart integral along field line
            in function SECS_2D_CurlFree_AntiSym_magnetic_lineintegral

        Parameters
        ----------
        thetaSECS : float
            Co-latitude of the CF SECS, [radian]
        Rsecs : float
            Radius of the sphere where CF SECS is located, [km]
        L : float
            L-value of the field line starting from the CF SECS pole, [km]
    %       NOTE: this is in kilometers, so really L*Rsecs;
        minD : float
            minimum horizontal distance between the CF SECS pole and footpoints of
            the points where magnetic field is needed, [km], SCALAR

        Returns
        -------
        array
            co-latitudes of the integration points (= end points of the current
            elements), [radian]
    """

    # Minimum length of integration steps (NOTE: actually the minimum average length)
    minStep = 10  # step in km

    # Adjust step according to horizontal distance
    # NOTE: maybe should increase more rapidly than linearly
    step = min(200, max(minStep, 0.1 * minD))

    # Length of the field line from one ionosphere to the other.
    x = np.pi / 2 - thetaSECS
    s = L * abs(
        np.sin(x) * np.sqrt(3 * np.sin(x) ** 2 + 1)
        + 1 / np.sqrt(3) * np.arcsinh(np.sqrt(3) * np.sin(x))
    )

    # Number of steps and step size in co-latitude assuming uniform horizontally adjusted step length
    Nint = int(np.ceil(s / step))
    dt0 = abs(2 * thetaSECS - np.pi) / Nint

    # Take larger steps at high altitudes.
    # Altitude of Swarm is 450-520 km.
    tmp = np.full((Nint,), np.nan)
    tmp[0] = min(thetaSECS, np.pi - thetaSECS)  # always start from north hemisphere
    n = 0
    while tmp[n] < np.pi / 2:  # stop just before reaching the equator
        h = L * np.sin(tmp[n]) ** 2 - Rsecs  # height [km]
        # adjust step according to altitude
        # NOTE: maybe should increase more rapidly than linearly
        dt = dt0 * min(500, max(1, h / 300 - 2))  # start to increase above 900 km
        tmp[n + 1] = tmp[n] + dt
        n += 1

    # Make north and south hemispheres symmetric
    tmp = tmp[:n]
    t = np.hstack([tmp, np.pi / 2, np.pi - tmp[::-1]])

    # Flip if start from south hemisphere
    if thetaSECS > np.pi / 2:
        t = t[::-1]

    return t
