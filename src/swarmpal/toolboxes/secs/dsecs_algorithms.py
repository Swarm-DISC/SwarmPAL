"""Algorithms for low latitude spherical elementary current system analysis.

Adapted from MatLab code by Heikki Vanhamäki. 

"""

import numpy as np
import scipy
from swarmpal.toolboxes.secs import aux_tools as auto



def sub_fit_1D_DivFree(SwA,SwC,result):

    #extract radii of the Earth and ionospheric current layer
    Re = result["Re"]
    Ri = result["Ri"]

    #Combine data from the two satellites.
    ###add .values.flatten() if we use xarray
    latB = np.vstack(SwA["magLat"].flatten(), SwC["magLat"].flatten())
    rB = np.vstack(SwA["r"].flatten(), SwC["r"].flatten())

    #Fit is done to the parallel magnetic disturbance.
    uvR = np.vstack(SwA["UvR"].flatten(), SwC["UvR"].flatten())
    uvT = np.vstack(SwA["magUvT"].flatten(), SwC["magUvT"].flatten())
    Bpara = np.vstack(SwA["Bpara"].flatten(), SwC["Bpara"].flatten())

    #Calculate 1D SECS grid in a separate sub-function
    Dlat1D = 0.5    #latitude spacing [degree]
    ExtLat1D = 5    #Number of points to extend outside the data range
    lat1D, mat1Dsecond = auto.sub_Swarm_grids_1D(SwA["magLat"],SwC["magLat"],Dlat1D,ExtLat1D)
    lat1D = lat1D.flatten()
    N1d = len(lat1D)

    #Calculate B-matrices that give magnetic field from SECS amplitudes and form the field-aligned matrix.
    matBr, matBt = SECS_1D_DivFree_magnetic(latB,lat1D,rB,Ri,500)
    matBpara = np.tile(uvR,(1,N1d)) * matBr + np.tile(uvT,(1,N1d)) * matBt

    #Fit the 1D DF SECS to the field-aligned magnetic disturbance.
    #Use zero constraint on the 2nd latitudinal derivative.
    alpha = 1e-5
    epsSVD = 4e-4
    Idf = sub_inversion(matBpara, mat1Dsecond, epsSVD, alpha, Bpara)  #scaling factors of the 1D DF SECS

    #Calculate the eastward current produced by the 1D DF SECS.
    latJ = result["magLat"].flatten()
    matJp = SECS_1D_DivFree_vector(latJ,lat1D,Ri)
    ###add to result xarray
    result["df1dMagJp"]= np.reshape(matJp @ Idf, np.shape(result["magLat"]),order='F')
    result["df1dMagJt"] = np.zeros(np.shape(result["magLat"]))

    #Calculate the magnetic field produced by the 1D DF SECS.
    Na = len(SwA["magLat"].flatten())
    sizeA = np.shape(SwA["magLat"])
    sizeC = np.shape(SwC["magLat"])
    apu = matBr @ Idf
    ###add to SwA and SwC xarray?
    SwA["df1dBr"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dBr"] = np.reshape(apu[Na:], sizeC, order='F')
    apu = matBt @ Idf
    SwA["df1dMagBt"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dMagBt"] = np.reshape(apu[Na:], sizeC, order='F')
    SwA["df1dMagBp"] = np.zeros(sizeA)
    SwC["df1dMagBp"] = np.zeros(sizeC)
    apu = matBpara @ Idf
    SwA["df1dBpara"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dBpara"] = np.reshape(apu[Na:], sizeC, order='F')

    return SwA,SwC,result

def _legPol_n1(n, x, ind=False):
    """Evaluate associated Legendre polynomials (l, m=1).

    Parameters
    ----------
    n : int
        max degree of the polynomials
    x : ndarray
        points at which to evaluate the polynomials, -1 <= x >= 1
    ind : bool, optional
        Return only the nth degree polynomial, by default False

    Returns
    -------
    ndarray
        (len(x),n) shaped array of the polynomial values.
    """

    v = np.array([])

    Pa = np.zeros((len(x), n))
    Pa[:, 0] = np.sqrt(1 - x**2)
    Pa[:, 1] = 3 * x * Pa[:, 0]

    if n > 2:
        for j in range(2, n):
            Pa[:, j] = (
                2 * x * Pa[:, j - 1]
                - Pa[:, j - 2]
                + (x * Pa[:, j - 1] - Pa[:, j - 2]) / (j - 1)
            )
    if ind or n == 1:
        return -1 * Pa[:, n]
    else:
        return -1 * Pa


def _legPol(n, x, ind=False):
    """Evaluate Legendre polynomials.

    Parameters
    ----------
    n : int
        max degree of the polynomials
    x : ndarray
        1D array of points at which to evaluate the polynomials, -1 <= x >= 1
    ind : bool, optional
        Return only the nth degree polynomial, by default False

    Returns
    -------
    ndarray
        (len(x),n+1) shaped array of the polynomial values.
    """

    P = np.zeros((len(x), n + 1))
    P[:, 0] = 1
    P[:, 1] = x

    for i in range(2, n + 1):
        P[:, i] = (
            2 * x * P[:, i - 1] - P[:, i - 2] - (x * P[:, i - 1] - P[:, i - 2]) / i
        )

    if ind or n == 0:
        return P[:, n]
    else:
        return P


def SECS_1D_DivFree_vector(latV, latSECS, ri):
    """Calculates the matrix for 1D divergence free current density.

    Parameters
    ----------
    latV : ndarray
        Latitudes of locations where vectors is calculated, in degrees
    latSECS : ndarray
        Latitudes of the 1D SECS poles, in degrees
    ri : int
        Assumed radius of the spherical shell where the currents flow (km).

    Returns
    -------
    ndarray
       2D array that maps the 1D elementary current amplitudes to currents.
    """

    # ravel and reassign

    latV = latV.ravel()
    latSECS = latSECS.ravel()

    nv = len(latV)
    nsecs = len(latSECS)
    matVphi = np.zeros((nv, nsecs)) + np.nan

    theta_secs = (90.0 - latSECS) * np.pi / 180
    theta_v = (90.0 - latV) * np.pi / 180

    # Separation of the 1D SECS

    if (nsecs) > 1:
        d_theta = np.nan + theta_secs
        d_theta[0] = theta_secs[0] - theta_secs[1]
        d_theta[1:] = (theta_secs[:-1] - theta_secs[1:]) / 2
        d_theta[-1] = theta_secs[-2] - theta_secs[-1]
    else:
        d_theta = np.arrya([0.5 / 180 * np.pi])

    # limit for near pole area [radians]
    limit = np.abs(d_theta) / 2

    # Loop over vector positions and SECS positions

    for i in range(nv):
        ct = np.cos(theta_v[i])
        st = np.sin(theta_v[i])
        for j in range(nsecs):
            if (np.abs(theta_secs[j]) - theta_v[i]) >= limit[j]:
                # Current at co-latittude theta_v[i] caused by 1D SECS at theataSECS[j]
                # see equation (4) of Vanhamaki et al. 2003
                matVphi[i, j] = (ct + np.sign(theta_v[i] - theta_secs[j])) / st
            else:
                # Linear interpolation near 1D SECS pole
                north = -np.tan((theta_secs[j] - limit[j]) / 2)
                south = 1 / np.tan((theta_secs[j] + limit[j]) / 2)
                diff = theta_v[i] - theta_secs[j]
                change = (south - north) / (2 * limit[j])
                matVphi[i, j] = 0.5 * (north + south) + diff * change

    return matVphi.squeeze() / (2 * ri)


def SECS_1D_DivFree_magnetic(latB, latSECS, rb, rsecs, M):
    """Calculates the matrix for magnetic field from 1D divergence free current.

    Parameters
    ----------
    latB : ndarray
        Array of latitudes where the magnetic field is calculated [degree]
    latSECS : ndarray
        Array of latitudes where 1D CF SECS are located [degree]
    rb : ndarray
        Array of radial distances of the points where the magnetic field is calculated, scalar or vector, [km]
    rsecs : float
        Radius of the sphere where the 1D CF SECS are located, scalar, [km]
    M : int
        argest order of the Legendre polynomials

    Returns
    -------
    ndarray
        2D array to map the elementary current system amplitudes to radial magnetic field
    ndarray
        2D array to map the elementary current system amplitudes to meridional magnetic field
    """

    # if rb is scale, use same rb for all points
    rb = rb + 0 * latB

    # ravel
    latB = latB.ravel()
    latSECS = latSECS.ravel()
    rb = rb.ravel()

    t = np.where(rb <= rsecs, rb / rsecs, rsecs / rb)
    M = np.abs(np.round(M))
    thetaSECS = (90 - latSECS) * np.pi / 180
    thetaB = (90 - latB) * np.pi / 180

    # First radial magnetic field. See equations (6) and (B6) in Vanhamäki et al. (2003).
    # Assume [Rb]=km, [B]=nT, [Idf]=A.
    # There is a common factor of 0.2*pi/Rb,         if Rb<Rsecs,
    #                            0.2*pi*Rsecs/Rb^2, if Rb>Rsecs.
    # Series of (ratio of radii)^m, m=1...M

    t_aux = np.zeros((len(latB), M))
    t_aux[:, 0] = 0.2 * np.pi / rb * t * np.where(rb < rsecs, 1, rsecs / rb)

    for i in range(1, M):
        t_aux[:, i] = t * t_aux[:, i - 1]

    # Legendre polynomials of cos(thetaSECS) and cos(thetaB) for m=1...M

    PtS = _legPol(M, np.cos(thetaSECS), False)
    PtS = PtS[:, 1 : M + 2]
    PtB = _legPol(M, np.cos(thetaB), False)
    PtB = PtB[:, 1 : M + 2]
    # Calculate the sum in eqs (6) and (B6) in Vanhamäki et al. (2003) up to M.
    matBradial = (t_aux * PtB) @ PtS.T

    # Then theta-component. See equations (7) and (B7) in Vanhamäki et al. (2003).
    # Assume [Rb]=km and [B]=nT when unit of scaling factor is Ampere.
    # There is a common factor of 0.2*pi/Rb,         if Rb<Rsecs,
    #                           -0.2*pi*Rsecs/Rb^2, if Rb>Rsecs.
    # Series of (ratio of radii)^m/{m or m+1} , m=1...M

    t_aux = np.zeros((len(latB), M))
    t_aux[:, 0] = 0.2 * np.pi / rb * t * np.where(rb <= rsecs, 1, -rsecs / rb)
    above = rb > rsecs
    for i in range(1, M):
        t_aux[:, i] = t * t_aux[:, i - 1]
        t_aux[:, i - 1] = t_aux[:, i - i] / (i + above)
    t_aux[:, M - 1] = t_aux[:, M - 1] / (M + above)
    # Associated Legendre polynomials of cos(thetaB) for m=1...M
    PtBa = _legPol_n1(M, np.cos(thetaB), False)
    # Calculate the sum in eqs (7) and (B7) in Vanhamäki et al. (2003) up to M.
    matBtheta = (t_aux * PtBa) @ PtS.T

    return matBradial.squeeze(), matBtheta.squeeze()


def SECS_1D_CurlFree_vector(latV, latSECS, ri):
    """Calculates the array relating SECS amplitudes to curl free current.

    Parameters
    ----------
    latV : ndarray
        Array of latitudes where the current is calculated.
    latSECS : ndarray
        Array of latitudes of the SECS poles.
    ri : float
        Assumed radius of the spherical shell where the currents flow (km).

    Returns
    -------
    ndarray
       2D array to map the elementary current amplitudes to div free current.
    """
    return SECS_1D_DivFree_vector(latV, latSECS, ri)


def SECS_1D_CurlFree_magnetic(latB, latSECS, rb, rsecs, geometry):
    """Calculates the array that maps the CF SECS amplitudes to magnetic field.

    Parameters
    ----------
    latB : ndarray
        Latitudes where the magnetic field is calculated.
    latSECS : ndarray
        Latitudes of the SECS poles
    rb : ndarray
        Array of radial distances of the points where the magnetic field is calculated, scalar or vector, [km]
    rsecs : float
        Radius of the sphere where the calculation takes place, [km], scalar
    geometry : string
        If 'radial', assume radial field lines. Else, assume dipolar field.

    Returns
    -------
    ndarray
        2D array that maps the SECS amplitudes to magnetic field measurements.
    """

    # if rb is scale, use same rb for all points
    latB = latB.ravel()
    latSECS = latSECS.ravel()
    rb = rb + 0 * latB

    # For each individual CF SECS we have
    #  Bphi(r,theta) = f(r) * mu0 * Jtheta(rsecs,FPtheta),  rb>rsecs
    #  Bphi(r,theta) = 0,                                   rb<rsecs
    # Here FPtheta is co-latitude of the magnetic footpoint in the ionosphere,
    #  FPtheta = theta,                                spherical FAC
    #  sin(FPtheta)/sqrt(Rsecs) = sin(theta)/sqrt(r),  dipole FAC.
    # Similarly, the scaling function f(r) depens on the geometry,
    #  f(r)=rsecs/rb,        spherical FAC
    #  f(r)=(rsecs/rb)^1.5,  dipole FAC
    # For derivation of these results see appendix B of Juusola et al. (2006).
    # This means that
    #  (B of CF SECS) = A*(radial unit vector) \times (J of CF SECS)
    # where A=f(r)*mu0 or A=0. Moreover, we have relationship
    #  (J of CF SECS) = - (radial unit vector) \times (J of DF SECS)
    # so that
    #  (B of a CF SECS) = A*(J of a DF SECS).

    if geometry == "radial":
        FPlat = latB
    else:
        theta = (90.0 - latB) / 180 * np.pi
        aux = np.sqrt(rsecs / rb) * np.sin(theta)
        aux[np.abs(aux) > 1] = 0
        fptheta = np.arcsin(aux)
        fptheta = np.where(theta > np.pi / 2, np.pi - fptheta, fptheta)
        fplat = 90.0 - fptheta / np.pi * 180

    # Calculate current densoty at footpoints
    matBphi = SECS_1D_DivFree_vector(fplat, latSECS, rsecs)

    # Convert current density at footpoint to magnetic field at satellite.
    # mu0=4*pi*1e-7 and if scaling factors are in [A], radii in [km] and
    # magnetic field in [nT]  -->  extra factor of 1e6.
    # "-" sign because positive scaling factor means FAC towards the ionosphere,
    # which is -r direction. This is consistent with SECS_1D_CurlFree_vector.t

    if geometry == "radial":
        # assume radial FAC
        fr = -rsecs / rb * 0.4 * np.pi
    else:
        fr = -np.power(rsecs / rb, 1.5) * 0.4 * np.pi

    for n in range(len(latB)):
        if rb[n] > rsecs:
            matBphi[n, :] = matBphi[n, :] * fr[n]
        else:
            matBphi[n, :] = 0

    return matBphi.squeeze()


def SECS_2D_DivFree_vector(thetaV, phiV, thetaSECS, phiSECS, radius, limitangle):
    """Calculates 2D DF SECS matrices for currents densities.

    Function for calculating matrices matVtheta and matVphi which give the theta- and
    phi-components of a vector field from the scaling factors of div-free spherical elementary
    surrent systems (DF SECS).

    Parameters
    ----------
    thetaV : ndarray
        theta coordinate of points where the vector is calculated
    phiV : ndarray
        phi coordinate of points where the vector is calculated
    thetaSECS : ndarray
        theta coordinates od SECS poles
    phiSECS : ndarray
        phi coordinates of points where the vector is calculated
    radius : ndarray
        Radius of the sphere where the calculation takes place, [km], scalar
    limitangle : ndarray
        Half-width of the uniformly distributed SECS, [radian], scalar or Nsecs-dimensional vector.

    Returns
    -------
    ndarray
        2D array to map the elementary current system amplitudes to theta component of current density.
    ndarray
        2D array to map the elementary current system amplitudes to phi component of current density.
    """
    # ravel and reassign
    thetaV = thetaV.ravel()
    phiV = phiV.ravel()
    thetaSECS = thetaSECS.ravel()
    phiSECS = phiSECS.ravel()
    limitangle = limitangle.ravel()

    # number of points where V is calculated and scaling factors are given
    Nv = len(thetaV)
    Nsecs = len(thetaSECS)
    matVtheta = np.zeros((Nv, Nsecs)) + np.nan
    matVphi = np.zeros((Nv, Nsecs)) + np.nan

    # if LimitAngle is scalar, use it for every SECS
    if len(limitangle) == 1:
        limitangle = limitangle + np.zeros(thetaSECS.shape)

    # This is a common factor in all components

    ComFac = 1.0 / (4 * np.pi * radius)

    # loop over vector field positions
    for n in range(Nv):
        # cosine of co-latitude in the SECS-centered system
        # See Eq. (A5) and Fig. 14 of Vanhamäki et al.(2003)
        CosThetaPrime = np.cos(thetaV[n]) * np.cos(thetaSECS) + np.sin(
            thetaV[n]
        ) * np.sin(thetaSECS) * np.cos(phiSECS - phiV[n])

        # sin and cos of angle C, multiplied by sin(theta').
        # See Eqs. (A2)-(A5) and Fig. 14 of Vanhamäki et al.(2003)

        sinC = np.sin(thetaSECS) * np.sin(phiSECS - phiV[n])
        cosC = (np.cos(thetaSECS) - np.cos(thetaV[n]) * CosThetaPrime) / np.sin(
            thetaV[n]
        )
        # Find those SECS poles that are far away from the calculation point
        distant = CosThetaPrime < np.cos(limitangle)

        # vector field proportional to cot(0.5*CosThetaPrime), see Eq. (2) of Vanhamäki et al.(2003)
        dummy = ComFac / (1 - CosThetaPrime[distant])
        matVtheta[n, distant] = dummy * sinC[distant]
        matVphi[n, distant] = dummy * cosC[distant]

        # Assume that the curl of a DF SECS is uniformly distributed inside LimitAngle
        # field proportional to a*tan(0.5*CosThetaPrime), where a=cot(0.5*LimitAngle)^2

        dummy = (
            ComFac
            * 1
            / np.tan(0.5 * limitangle[~distant]) ** 2
            / (1 + CosThetaPrime[~distant])
        )
        matVtheta[n, ~distant] = dummy * sinC[~distant]
        matVphi[n, ~distant] = dummy * cosC[~distant]

    return matVtheta.squeeze(), matVphi.squeeze()


def SECS_2D_DivFree_magnetic(thetaB, phiB, thetaSECS, phiSECS, rb, rsecs):
    """Calculates the array that maps the SECS amplitudes to magnetic field.

    Parameters
    ----------
    thetaB : ndarray
        Theta coordinate of the points where magnetic field is calculated.
    phiB : ndarray
        Phi coordinate of the points where magnetic field is calculated.
    thetaSECS : ndarray
        Theta coordinate of the SECS poles.
    phiSECS : ndarray
        Phi coordinate of the SECS poles.
    rb : ndarray
        Geocentric radius of the points where the magnetic field is calculated.
    rsecs : float
        Assumed radius of the spherical shell where the currents flow (km).

    Returns
    -------
    ndarray
        2D array to map the elementary current system amplitudes to radial component of magnetic field.
    ndarray
        2D array to map the elementary current system amplitudes to theta component of magnetic field.
    ndarray
        2D array to map the elementary current system amplitudes to phi component of magnetic field.
    """

    # ravel and reassign
    thetaB = thetaB.ravel()
    phiB = phiB.ravel()
    thetaSECS = thetaSECS.ravel()
    phiSECS = phiSECS.ravel()
    rb = rb.ravel()

    # number of points where B is calculated and scaling factors are given

    Nb = len(thetaB)
    Nsecs = len(thetaSECS)
    matBradial = np.zeros((Nb, Nsecs)) + np.nan
    matBtheta = np.zeros((Nb, Nsecs)) + np.nan
    matBphi = np.zeros((Nb, Nsecs)) + np.nan

    # If Rb is scalar, use same radius for all points.

    rb = rb + 0 * thetaB

    # Ratio of the radii, smaller/larger
    ratio = np.minimum(rb, [rsecs]) / np.maximum(rb, [rsecs])

    # There is a common factor mu0/(4*pi)=1e-7. Also 1/Rb is a common factor
    # If scaling factors are in [A], radii in [km] and magnetic field in [nT]  --> extra factor of 1e6

    factor = 0.1 / rb

    # loop over B positions

    for n in range(Nb):
        # cos and square of sin of co-latitude in the SECS-centered system
        # See Eq. (A5) and Fig. 14 of Vanhamäki et al.(2003)
        CosThetaPrime = np.cos(thetaB[n]) * np.cos(thetaSECS) + np.sin(
            thetaB[n]
        ) * np.sin(thetaSECS) * np.cos(phiSECS - phiB[n])
        Sin2ThetaPrime = 1 - CosThetaPrime**2

        #  %sin and cos of angle C, divided by sin(theta').
        # See Eqs. (A2)-(A5) and Fig. 14 of Vanhamäki et al.(2003)
        # sinC = np.zeros(CosThetaPrime.shape)
        # cosC = np.zeros(CosThetaPrime.shape)
        sinC = np.sin(thetaSECS) * np.sin(phiSECS - phiB[n]) / Sin2ThetaPrime
        cosC = (np.cos(thetaSECS) - np.cos(thetaB[n]) * CosThetaPrime) / (
            np.sin(thetaB[n] * Sin2ThetaPrime)
        )
        sinC = np.where(Sin2ThetaPrime <= 1e-10, 0, sinC)
        cosC = np.where(Sin2ThetaPrime <= 1e-10, 0, cosC)

        # auxiliary variable
        auxroot = np.sqrt(1 - 2 * ratio[n] * CosThetaPrime + ratio[n] ** 2)

        if rb[n] < rsecs:
            auxVertical = 1
            # See Eq. 10 of Amm and Viljanen 1999.
            auxHorizontal = -factor[n] * (
                (ratio[n] - CosThetaPrime) / auxroot + CosThetaPrime
            )
        elif rb[n] > rsecs:
            auxVertical = ratio[n]
            # See eq. (A8) of Amm and Viljanen (1999)
            auxHorizontal = -factor[n] * (1 - ratio[n] * CosThetaPrime) / (auxroot - 1)
        else:
            # Actually horizontal field is not well defined but this is the average.
            # See eqs. (10) and (A8) of Amm and Viljanen (1999).
            auxVertical = 1
            auxHorizontal = -factor[n] * (auxroot + CosThetaPrime - 1) / 2

        # See Eqs. (9) and (A7) of Amm and Viljanen (1999).
        matBradial[n, :] = auxVertical * factor[n] * (1 / auxroot - 1)
        matBtheta[n, :] = auxHorizontal * cosC
        matBphi[n, :] = -auxHorizontal * sinC

    return matBradial.squeeze(), matBtheta.squeeze(), matBphi.squeeze()


def SECS_2D_CurlFree_antisym_vector(
    thetaV, phiV, thetaSECS, phiSECS, radius, limitangle
):
    """Calculates the mapping from antisymmetric dipolar CF SECS to current density.

    Parameters
    ----------
    thetaV : ndarray
        theta coordinate of points where the vector is calculated
    phiV : ndarray
        phi coordinate of points where the vector is calculated
    thetaSECS : ndarray
        theta coordinates od SECS poles
    phiSECS : ndarray
        phi coordinates of points where the vector is calculated
    radius : ndarray
        Radius of the sphere where the calculation takes place, [km], scalar
    limitangle : ndarray
        Half-width of the uniformly distributed SECS, [radian], scalar or Nsecs-dimensional vector.


    Returns
    -------
    ndarray
        2D array that maps the SECS amplitudes to the theta component of the current density.
    ndarray
        2D array that maps the SECS amplitudes to the phi component of the current density.
    """

    # vector field of a CF SECS = - (radial unit vector) \times (vector field of a DF SECS)
    Vp, Vt = SECS_2D_DivFree_vector(
        thetaV, phiV, thetaSECS, phiSECS, radius, limitangle
    )
    Vp = -Vp

    # Anti-SECS
    [aVp, aVt] = SECS_2D_DivFree_vector(
        thetaV, phiV, np.pi - thetaSECS, phiSECS, radius, limitangle
    )
    aVp = -aVp

    matVtheta = Vt - aVt
    matVphi = Vp - aVp

    return matVtheta.squeeze(), matVphi.squeeze()


def SECS_2D_CurlFree_antisym_magnetic(
    thetaB, phiB, thetaSECS, phiSECS, rb, rsecs, limitangle
):
    """Calculates the mapping from antisymmetric dipolar CF SECS to magnetic field.

    Parameters
    ----------
    thetaB : ndarray
        Theta coordinate of the points where magnetic field is calculated.
    phiB : ndarray
        Phi coordinate of the points where magnetic field is calculated.
    thetaSECS : ndarray
        Theta coordinate of the SECS poles.
    phiSECS : ndarray
        Phi coordinate of the SECS poles.
    rb : ndarray
        Geocentric radius of the points where the magnetic field is calculated.
    rsecs : float
        Assumed radius of the spherical shell where the currents flow (km).
    limitangle : ndarray
        Half-width of the uniformly distributed SECS, [radian], scalar or Nsecs-dimensional vector.

    Returns
    -------
    ndarray
        2D array that maps the SECS amplitudes to the theta component of the current density.
    ndarray
        2D array that maps the SECS amplitudes to the phi component of the current density.

    """

    # ravel and reassign

    thetaB = thetaB.ravel()
    phiB = phiB.ravel()
    rb = rb.ravel()
    thetaSECS = thetaSECS.ravel()
    phiSECS = phiSECS.ravel()
    rsecs = rsecs
    limitangle = limitangle.ravel()

    Nb = len(thetaB)
    Nsecs = len(thetaSECS)
    matBradial = np.nan + np.zeros((Nb, Nsecs))
    matBtheta = np.nan + np.zeros((Nb, Nsecs))
    matBphi = np.nan + np.zeros((Nb, Nsecs))

    # If rb is scalar, make a it a vector
    rb = rb + 0 * thetaB

    # co-latitudes of the anti-SECS at the conjugate point
    athetaSECS = np.pi - thetaSECS

    aux = rsecs**2 / rb
    bp, bt = SECS_2D_DivFree_magnetic(thetaB, phiB, thetaSECS, phiSECS, aux, rsecs)[1:]
    bt = -bt
    abp, abt = SECS_2D_DivFree_magnetic(thetaB, phiB, athetaSECS, phiSECS, aux, rsecs)[
        1:
    ]
    abt = -abt
    aux = aux / rb

    # loop over secs poles
    for n in range(Nsecs):

        # small number with dimension distance ^3 to avoid division by zero
        smallr = (
            1.1 * rsecs * limitangle[n]
        )  # Optimal choice probably depends on altitude, but ignore it here

        # First B of the horizontal current
        # this is the same as the horizontal magnetic field of a 2d df SECS, suitably scaled and rotated by 90 degrees.
        # aux = rsecs**2/rb
        # bp,bt = SECS_2D_DivFree_magnetic(thetaB, phiB, thetaSECS[n],phiSECS[n],aux,rsecs)[1:]
        # bt = -bt
        # abp,abt = SECS_2D_DivFree_magnetic(thetaB,phiB,athetaSECS[n],phiSECS[n],aux,rsecs)[1:]
        # abt = -abt
        # aux = aux/rb
        matBtheta[:, n] = (bt[:, n] - abt[:, n]) * aux
        matBphi[:, n] = (bp[:, n] - abp[:, n]) * aux

        # then dipolar line currents by very simple numerical integration
        bx, by, bz = secs_2d_curlFree_antisym_lineintegral(
            thetaB, phiB, rb, thetaSECS[n], phiSECS[n], rsecs, smallr
        )

        # add to the magnetic field of the horizontal current
        matBradial[:, n] = (
            bx * np.sin(thetaB) * np.cos(phiB)
            + by * np.sin(thetaB) * np.sin(phiB)
            + bz * np.cos(thetaB)
        )
        matBtheta[:, n] = (
            matBtheta[:, n]
            + bx * np.cos(thetaB) * np.cos(phiB)
            + by * np.cos(thetaB) * np.sin(phiB)
            - bz * np.sin(thetaB)
        )
        matBphi[:, n] = matBphi[:, n] - bx * np.sin(phiB) + by * np.cos(phiB)

    return matBradial.squeeze(), matBtheta.squeeze(), matBphi.squeeze()


def secs_2d_curlFree_antisym_lineintegral(
    thetaB, phiB, thetaSECS, phiSECS, rb, rsecs, smallr
):
    """_summary_

    Parameters
    ----------
    thetaB : ndarray
        Theta coordinate of the points where magnetic field is calculated.
    phiB : ndarray
        Phi coordinate of the points where magnetic field is calculated.
    thetaSECS : ndarray
        Theta coordinate of the SECS poles.
    phiSECS : ndarray
        Phi coordinate of the SECS poles.
    rb : ndarray
        Geocentric radius of the points where the magnetic field is calculated.
    rsecs : float
        Assumed radius of the spherical shell where the currents flow (km).
    smallr : ndarray
        Limit of "small" radius [km]

    Returns
    -------
    ndarray
        _description_
    ndarray
        _description_
    ndarray
        _description_
    """

    # ravel and reassign

    # thetaSECS = thetaSECS.ravel()
    # phiSECS = phiSECS.ravel()

    # Need the theta-angles of the integration points and the integration step
    L = rsecs / np.sin(thetaSECS) ** 2

    if 0:
        # fixed number of integration steps
        Nint = int(10e3)
    else:
        step = 10  # step in km
        # length of field line from one ionosphere to the other
        x = np.pi / 2 - thetaSECS
        s = L * np.abs(
            np.sin(x) * np.sqrt(3 * np.sin(x) ** 2 - 1)
            + 1 / np.sqrt(3) * np.arcsinh(np.sqrt(3) * np.sin(x))
        )
        Nint = np.ceil(s / step)

    dt = (
        2 * thetaSECS - np.pi
    ) / Nint  # negative if SECS pole at the northern hemisphere
    t = (np.pi - thetaSECS) + (np.arange(1, Nint + 1, dtype=np.double) - 0.5) * dt

    # xyz - coordinates of the integration points (= locations of the current elements)
    # Make sure these are ROW vectors
    x = L * np.sin(t) ** 3 * np.cos(phiSECS)
    y = L * np.sin(t) ** 3 * np.sin(phiSECS)
    z = -L * np.sin(t) ** 2 * np.cos(t)

    # xyz coordinates of the field points

    x0 = rb * np.sin(thetaB) * np.cos(phiB)
    y0 = rb * np.sin(thetaB) * np.sin(phiB)
    z0 = rb * np.cos(thetaB)

    # xyz  components of the current elements
    dlx = 3 * L * dt * np.cos(t) * np.sin(t) ** 2 * np.cos(phiSECS)
    dly = 3 * L * dt * np.cos(t) * np.sin(t) ** 2 * np.sin(phiSECS)
    dlz = -L * dt * np.sin(t) * (1 - 3 * np.cos(t) ** 2)

    # |distances between current elements and field points|^3
    diffx = (np.expand_dims(x0, -1) - x).T  # (x0[:,np.newaxis] -x).T
    diffy = (np.expand_dims(y0, -1) - x).T  # (y0[:,np.newaxis] -y).T
    diffz = (np.expand_dims(z0, -1) - x).T  # (z0[:,np.newaxis] -z).T

    if 1:
        tt = np.sqrt(diffx * diffx + diffy * diffx + diffz * diffx)
        root = tt * tt * tt
    # root=_calc_root(diffx,diffy,diffz)
    # Remove singularity by setting a lower lomit to the distance between current elements and field points.
    # Use the small number
    root = np.where(
        root < smallr * smallr * smallr, smallr, root
    )  # np.maximum(root,smallr**3)
    # xyz components of the magnetic field
    # now there is a common factor mu0/(4*pi) = 1e.7
    # If scaling factors as in [A] , radii in [km] and magnetic field in [nT] --> extra factor of 1e6
    # Make sure the products are (Nb, Nint) matrices and sum along the rows to integrate

    xsum = (dly * diffz.T - dlz * diffy.T) / root.T
    ysum = (dlx * diffz.T - dlz * diffx.T) / root.T
    zsum = (dlx * diffy.T - dly * diffx.T) / root.T

    bx = 0.1 * np.sum(xsum.T).T
    by = 0.1 * np.sum(ysum.T).T
    bz = 0.1 * np.sum(zsum.T).T

    return bx, by, bz


# @jit(parallel=True,nopython=True)
def _calc_root(x, y, z):
    """_summary_

    Parameters
    ----------
    x : ndarray
        _description_
    y : ndarray
        _description_
    z : ndarray
        _description_

    Returns
    -------
    ndarray
        _description_
    """

    tt = np.sqrt(x * x + y * y + z * z)
    root = tt * tt * tt

    return root
