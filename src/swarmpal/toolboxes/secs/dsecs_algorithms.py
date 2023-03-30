"""Algorithms for low latitude spherical elementary current system analysis.

Adapted from MatLab code by Heikki Vanhamäki.

"""


import logging
import apexpy
import numpy as np
import pandas
import xarray as xr

import swarmpal.toolboxes.secs.aux_tools as auto
from swarmpal.toolboxes.secs import SecsInputs
from swarmpal.toolboxes.secs.aux_tools import (
    sph2sph,
    sub_FindLongestNonZero,
    sub_Swarm_grids,
)

logger = logging.getLogger(__name__)


def _DSECS_steps(SwAin, SwCin):
    """DSECS analysis for Vires xarray input.

    Parameters
    ----------
    SwAin : xarray from SecsInput
        Input data for Swarm Alpha
    SwCin : xarray from SecsInput
        Input data for Swarm Charlie

    Returns
    -------
    List of dicts
        Each entry contains the original data ("original_data), the resulting current densities ("current_densities"), the magnetic fit for Swarm A and C ("magnetic_Fit_Alpha", "magnetic_Fit_Charlie") and the dsecsdata object ("case").

    """

    SwA_list, ovals = auto.get_eq(SwAin)
    SwC_list, _ = auto.get_eq(SwCin, ovals=ovals)

    out = []

    try:

        for SwA, SwC in zip(SwA_list, SwC_list):
            case = dsecsdata()
            logger.info("Populating data object.")
            case.populate(SwA, SwC)
            if case.flag == 0:
                logger.info("Starting analysis.")
                case.analyze()
                logger.info("Formatting results.")
                _, currents, afit, cfit = case.dump()
                # resdict = res.to_dict()
            else:
                logger.info("Analysis failed")
                currents = None
                afit = None
                cfit = None
            loopres = {
                "original_data": (SwA, SwC),
                "current_densities": currents,
                "magnetic_fit_Alpha": afit,
                "magnetic_fit_Charlie": cfit,
                "case": case,
            }
            out.append(loopres)
    except Exception as e:
        logger.warn(e)

    return out


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

    Pa = np.zeros((len(x), n))
    Pa[:, 0] = np.sqrt(1 - x**2)
    Pa[:, 1] = 3 * x * Pa[:, 0]

    if n > 2:
        for j in range(2, n):
            Pa[:, j] = (
                2 * x * Pa[:, j - 1]
                - Pa[:, j - 2]
                + (x * Pa[:, j - 1] - Pa[:, j - 2]) / (j)
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

    # flatten and reassign

    latV = latV.flatten()
    latSECS = latSECS.flatten()

    nv = len(latV)
    nsecs = len(latSECS)
    matVphi = np.zeros((nv, nsecs)) + np.nan

    theta_secs = (90.0 - latSECS) * np.pi / 180
    theta_v = (90.0 - latV) * np.pi / 180

    # Separation of the 1D SECS

    if (nsecs) > 1:
        d_theta = np.nan + theta_secs
        d_theta[0] = theta_secs[0] - theta_secs[1]
        d_theta[1:-1] = (theta_secs[:-2] - theta_secs[2:]) / 2
        d_theta[-1] = theta_secs[-2] - theta_secs[-1]
    else:
        d_theta = np.array([0.5 / 180 * np.pi])

    # limit for near pole area [radians]
    limit = np.abs(d_theta) / 2

    # Loop over vector positions and SECS positions

    for i in range(nv):
        ct = np.cos(theta_v[i])
        st = np.sin(theta_v[i])
        for j in range(nsecs):
            if (np.abs(theta_secs[j] - theta_v[i])) >= limit[j]:
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

    # flatten
    latB = latB.flatten()
    latSECS = latSECS.flatten()
    rb = rb.flatten()

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
    t_aux[:, 0] = 0.2 * np.pi / rb * t * np.where(rb <= rsecs, 1, rsecs / rb)

    for i in range(1, M):
        t_aux[:, i] = t * t_aux[:, i - 1]

    # Legendre polynomials of cos(thetaSECS) and cos(thetaB) for m=1...M

    PtS = _legPol(M, np.cos(thetaSECS), False)
    # PtS = PtS[:, 1 : M + 2]
    PtS = PtS[:, 1:]
    PtB = _legPol(M, np.cos(thetaB), False)
    # PtB = PtB[:, 1 : M + 2]
    PtB = PtB[:, 1:]
    # Calculate the sum Flinein eqs (6) and (B6) in Vanhamäki et al. (2003) up to M.

    matBradial = (t_aux * PtB) @ PtS.T

    # Then theta-component. See equations (7) and (B7) in Vanhamäki et al. (2003).
    # Assume [Rb]=km and [B]=nT when unit of scaling factor is Ampere.
    # There is a common factor of 0.2*pi/Rb,         if Rb<Rsecs,
    #                           -0.2*pi*Rsecs/Rb^2, if Rb>Rsecs.
    # Series of (ratio of radii)^m/{m or m+1} , m=1...M

    t_aux = np.zeros((len(latB), M))
    t_aux[:, 0] = 0.2 * np.pi / rb * t * np.where(rb <= rsecs, 1, -rsecs / rb)
    above = rb > rsecs
    for i in range(0, M - 1):
        t_aux[:, i + 1] = t * t_aux[:, i]
        t_aux[:, i] = t_aux[:, i] / (i + 1 + above)
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
    latB = latB.flatten()
    rb = rb.flatten()
    latSECS = latSECS.flatten()
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
        fplat = latB
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
    # flatten and reassign
    thetaV = thetaV.flatten()
    phiV = phiV.flatten()
    thetaSECS = thetaSECS.flatten()
    phiSECS = phiSECS.flatten()
    limitangle = limitangle.flatten()

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

    # flatten and reassign
    thetaB = thetaB.flatten()
    phiB = phiB.flatten()
    thetaSECS = thetaSECS.flatten()
    phiSECS = phiSECS.flatten()
    rb = rb.flatten()

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
        ind = np.nonzero(Sin2ThetaPrime > 1e-10)
        sinC = np.zeros(CosThetaPrime.shape)
        cosC = np.zeros(CosThetaPrime.shape)
        sinC[ind] = (
            np.sin(thetaSECS[ind])
            * np.sin(phiSECS[ind] - phiB[n])
            / Sin2ThetaPrime[ind]
        )
        cosC[ind] = (
            np.cos(thetaSECS[ind]) - np.cos(thetaB[n]) * CosThetaPrime[ind]
        ) / (np.sin(thetaB[n]) * Sin2ThetaPrime[ind])

        # sinC = np.where(Sin2ThetaPrime <= 1e-10, 0, sinC)
        # cosC = np.where(Sin2ThetaPrime <= 1e-10, 0, cosC)

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
            auxHorizontal = -factor[n] * (
                (1 - ratio[n] * CosThetaPrime) / (auxroot) - 1
            )
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

    # flatten and reassign

    thetaB = thetaB.flatten()
    phiB = phiB.flatten()
    rb = rb.flatten()
    thetaSECS = thetaSECS.flatten()
    phiSECS = phiSECS.flatten()
    rsecs = rsecs
    limitangle = limitangle.flatten()

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
    _, bp, bt = SECS_2D_DivFree_magnetic(thetaB, phiB, thetaSECS, phiSECS, aux, rsecs)
    # bt = -bt
    _, abp, abt = SECS_2D_DivFree_magnetic(
        thetaB, phiB, athetaSECS, phiSECS, aux, rsecs
    )
    # abt = -abt
    aux = aux / rb
    a2 = -bt + abt
    matBtheta = (a2.T * aux).T
    a2 = bp - abp
    matBphi = (a2.T * aux).T

    # loop over secs poles
    for n in range(Nsecs):
        # small number with dimension distance ^3 to avoid division by zero
        smallr = (
            1.1 * rsecs * limitangle[n]
        )  # Optimal choice probably depends on altitude, but ignore it here

        # First B of the horizontal current
        # this is the same as the horizontal magnetic field of a 2d df SECS, suitably scaled and rotated by 90 degrees.

        bx, by, bz = secs_2d_curlFree_antisym_lineintegral(
            thetaB, phiB, thetaSECS[n], phiSECS[n], rb, rsecs, smallr
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
    """Line integral for DSECS 2d curl free.

    Parameters
    ----------
    thetaB : ndarray
        Theta coordinate of the points where magnetic field is calculated.
    phiB : ndarray
        Phi coordinate of the points where magnetic field is calculated.
    thetaSECS : float
        Theta coordinate of the SECS pole in radian.
    phiSECS : float
        Phi coordinate of the SECS pole in radian.
    rb : ndarray
        Geocentric radius of the points where the magnetic field is calculated.
    rsecs : float
        Assumed radius of the spherical shell where the currents flow (km).
    smallr : ndarray
        Limit of "small" radius [km]

    Returns
    -------
    bx, by, bz : ndarray
        Cartesian components of the magnetic field at the magnetometer locations [nT]
    """

    # flatten and reassign

    # thetaSECS = thetaSECS.flatten()
    # phiSECS = phiSECS.flatten()

    # Need the theta-angles of the integration points and the integration step
    L = rsecs / np.sin(thetaSECS) ** 2

    # xyz coordinates of the field points
    x0 = rb * np.sin(thetaB) * np.cos(phiB)
    y0 = rb * np.sin(thetaB) * np.sin(phiB)
    z0 = rb * np.cos(thetaB)

    # Need minimum horizontal distance between the CF SECS and footpoints of the B-field points
    # Map B-field points to Rsecs
    thetaFP = np.arcsin(
        np.sqrt(rsecs / rb) * np.sin(thetaB)
    )  # co-latitude of the footpoints, mapped to the northern hemisphere

    ind = np.nonzero(thetaB > np.pi / 2)

    thetaFP[ind] = np.pi - thetaFP[ind]

    # Horizontal distances as  cosine of co-latitude in the SECS-centered system
    # See Eq. (A5) and Fig. 14 of Vanhamäki et al.(2003)
    tmp = np.cos(thetaFP) * np.cos(thetaSECS) + np.sin(thetaFP) * np.sin(
        thetaSECS
    ) * np.cos(phiSECS - phiB)

    minD = rsecs * np.arccos(np.max(tmp))

    # Calculate the theta-angles of the points used in the integration
    # make sure t is ROW vector
    t = auto.sub_points_along_fieldline(thetaSECS, rsecs, L, minD)

    # xyz - coordinates of the integration points (= locations of the current elements)
    # Make sure these are ROW vectors
    x = L * np.sin(t) ** 3 * np.cos(phiSECS)
    y = L * np.sin(t) ** 3 * np.sin(phiSECS)
    z = L * np.sin(t) ** 2 * np.cos(t)

    # xyz  components of the current elements
    dlx = -np.diff(x)
    dly = -np.diff(y)
    dlz = -np.diff(z)

    # xyz-coordinates of the mid-points
    x = (x[:-1] + x[1:]) / 2
    y = (y[:-1] + y[1:]) / 2
    z = (z[:-1] + z[1:]) / 2

    # |distances between current elements and field points|^3
    diffx = (np.expand_dims(x0, -1) - x).T  # (x0[:,np.newaxis] -x).T
    diffy = (np.expand_dims(y0, -1) - y).T  # (y0[:,np.newaxis] -y).T
    diffz = (np.expand_dims(z0, -1) - z).T  # (z0[:,np.newaxis] -z).T

    tt = diffx * diffx + diffy * diffy + diffz * diffz
    root = np.sqrt(tt * tt * tt)

    # Remove singularity by setting a lower lomit to the distance between current elements and field points.
    # Use the small number
    root = np.where(root < smallr * smallr * smallr, smallr**3, root)

    # xyz components of the magnetic field
    # now there is a common factor mu0/(4*pi) = 1e.7
    # If scaling factors as in [A] , radii in [km] and magnetic field in [nT] --> extra factor of 1e6
    # Make sure the products are (Nb, Nint) matrices and sum along the rows to integrate

    xsum = (dly * diffz.T - dlz * diffy.T) / root.T
    ysum = (dlx * diffz.T - dlz * diffx.T) / root.T
    zsum = (dlx * diffy.T - dly * diffx.T) / root.T

    bx = 0.1 * np.sum(xsum, axis=1)
    by = -0.1 * np.sum(ysum, axis=1)
    bz = 0.1 * np.sum(zsum, axis=1)

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


def get_data(
    t1,
    t2,
    model="IGRF",
):
    """Fetches Swarm data.
    Parameters
    ----------
    t1 : datetime
        Start date
    t2 : datetime
        End date
    model : str, optional
        Magnetic field model, by default 'IGRF'
    Returns
    -------
    SwA, SwC : xarray
        Swarm A and C data as xarray returned from Vires.
    """

    inputs = SecsInputs(
        start_time=t1,
        end_time=t2,
        model=model,
    )

    # SwA = auto.get_eq(inputs.s1.xarray)

    # SwC = auto.get_eq(inputs.s2.xarray)

    # SwA,SwC = getUnitVectors(SwA,SwC)

    return inputs.s1.xarray, inputs.s2.xarray


def getUnitVectors(SwA, SwC):
    """Calculates the magnetic unit vectors.
    Parameters
    ----------
    SwA, SwC : xarray
        Swarm A and C datasets.
    Returns
    -------
    SwA, SwC : xarray
        Swarm A and C datasets including magnetic unit vectors 'ggUvT', 'ggUvP' and 'UvR'.
    """

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
    """Class for 2D DSECS grids"""

    def __init__(self, origin="geo"):
        """Initializes the grid.
        Parameters
        ----------
        origin : str
            parameter controlling if the grid is created in geogrpahic or magnetic coordinates.
            'geo' or 'mag'.

        """
        self.ggLat = np.array([])
        self.ggLon = np.array([])
        self.magLat = np.array([])
        self.magLon = np.array([])
        self.angle2D = np.array([])
        self.diff2lon2D = np.array([])
        self.diff2lat2D = np.array([])
        self.origin = origin

    def create(
        self, lat1, lon1, lat2, lon2, dlat, lonRat, extLat, extLon, poleLat, poleLon
    ):
        """Initializes the 2D grid from Swarm data.
        Parameters
        ----------
        lat1, lat2 : ndarray
            Swarm A and C latitudes, [degree].
        lon1, lon2 : ndarray
            Swarm A and C longitudes, [degree].
        dlat : float
            2D grid spacing in latitudinal direction, [degree].
        lonRat : int or float
             Ratio between the satellite separation and longitudinal spacing of the 2D grid.
        extLat, extLon : int
            Number of points to extend the 2D grid outside data area in latitudinal and longitudinal directions.
        """

        Lat, Lon, self.angle2D, self.diff2lon2D, self.diff2lat2D = sub_Swarm_grids(
            lat1,
            lon1,
            lat2,
            lon2,
            dlat,
            lonRat,
            extLat,
            extLon,
        )
        Lon = Lon % 360
        if self.origin == "geo":
            self.ggLat, self.ggLon = Lat, Lon
            self.magLat, self.magLon, _, _ = sph2sph(
                poleLat, poleLon, self.ggLat, self.ggLon, [], []
            )
            self.magLon = self.magLon % 360
        elif self.origin == "mag":
            self.magLat, self.magLon = Lat, Lon
            self.ggLat, self.ggLon, _, _ = sph2sph(
                poleLat, 0, self.magLat, self.magLon, [], []
            )
            self.ggLon = self.ggLon % 360


class grid1D:
    """Simple class to hold a 1D lat grid"""

    def __init__(self):
        self.lat = np.array([])
        self.diff2 = np.array([])

    def create(self, lat1, lat2, dlat, extLat):
        """Initializes the 1D grid from Swarm data.
        Parameters
        ----------
        lat1, lat2 : ndarray
            Swarm A and C latitudes ,[degree].
        dlat : int or float
             1D grid spacing in latitudinal direction, [degree].
        extLat : int
            Number of points to extend the 1D grid outside data area in latitudinal direction.
        """

        self.lat, self.diff2 = auto.sub_Swarm_grids_1D(lat1, lat2, dlat, extLat)


class dsecsgrid:
    """Class for all the grids needed in DSECS analysis"""

    def __init__(self):
        """Initialize all the necessary parameters."""
        self.out = grid2D(origin="geo")
        self.secs2Ddf = grid2D(origin="mag")
        self.secs1Ddf = grid1D()
        self.secs1DcfNorth = grid1D()
        self.secs1DcfSouth = grid1D()
        self.secs2DcfNorth = grid2D(origin="mag")
        self.secs2DcfSouth = grid2D(origin="mag")
        self.secs2DcfRemoteNorth = grid2D(origin="mag")
        self.secs2DcfRemoteSouth = grid2D(origin="mag")
        self.outputlimitlat = 40
        self.Re = 6371
        self.Ri = 6371 + 130
        self.poleLat = 90.0
        self.poleLon = 0.0
        self.dlatOut = 0.5  # resolution in latitude
        self.lonRatioOut = 2
        self.extLatOut = 0
        self.extLonOut = 3
        self.extLat1D = 5
        self.indsN = dict()
        self.insdS = dict()
        self.extLat2D = 5
        self.extLon2D = 7
        self.cfremoteN = 3
        self.flag = 0
        self.test = dict()

    def FindPole(self, SwA):
        """Finds the best location for a local magnetic dipole pole based on Swarm measurements.
        Parameters
        ----------
        SwA : xarray
            Swarm A dataset.
        """
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

    def create(self, SwA, SwC):
        """Initializes the 1D and 2D grids from Swarm data.
        Parameters
        ----------
        SwA, SwC : xarray
            Swarm A and C datasets.
        """

        # Make grid around [-X,X] latitudes
        limitOutputLat = self.outputlimitlat
        ind = np.nonzero(abs(SwA["Latitude"].data) <= limitOutputLat)
        if len(ind[0]) == 0:
            logger.warn("No data within analysis area.")
            self.flag = 1
            return
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

        self.secs2Ddf.create(
            SwA["magLat"].data,
            SwA["magLon"].data,
            SwC["magLat"].data,
            SwC["magLon"].data,
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D,
            self.extLon2D,
            self.poleLat,
            self.poleLon,
        )

        self.secs1Ddf.create(SwA["magLat"], SwC["magLat"], self.dlatOut, self.extLat1D)

        indsN = dict()
        indsS = dict()
        for dat, sat in zip([SwA, SwC], ["A", "C"]):
            indsN[sat] = np.nonzero(dat["ApexLatitude"].data > 0)
            indsS[sat] = np.nonzero(dat["ApexLatitude"].data <= 0)

        if (
            len(indsN["A"]) == 0
            or len(indsN["C"]) == 0
            or len(indsS["A"]) == 0
            or len(indsS["C"]) == 0
        ):
            logger.warn("No data from both hemispheres.")
            self.flag = 1
            return

        self.indsN = indsN
        self.insdS = indsS

        # Create CF grids
        # 1D
        trackAN = getLocalDipoleFPtrack1D(
            SwA["magLat"].data[indsN["A"]],
            SwA["Radius"].data[indsN["A"]] * 1e-3,
            self.Ri,
        )
        trackCN = getLocalDipoleFPtrack1D(
            SwC["magLat"].data[indsN["C"]],
            SwC["Radius"].data[indsN["C"]] * 1e-3,
            self.Ri,
        )
        trackAS = getLocalDipoleFPtrack1D(
            SwA["magLat"].data[indsS["A"]],
            SwA["Radius"].data[indsS["A"]] * 1e-3,
            self.Ri,
        )
        trackCS = getLocalDipoleFPtrack1D(
            SwC["magLat"].data[indsS["C"]],
            SwC["Radius"].data[indsS["C"]] * 1e-3,
            self.Ri,
        )
        self.secs1DcfNorth.create(trackAN, trackCN, self.dlatOut, self.extLat1D)
        self.secs1DcfSouth.create(trackAS, trackCS, self.dlatOut, self.extLat1D)
        # self.secs2DcfNorth.create()
        # self.secs2DcfNorth.create()
        # 2D

        # North first
        trackALatN, trackALonN = getLocalDipoleFPtrack2D(
            SwA["magLat"].data[indsN["A"]],
            SwA["magLon"].data[indsN["A"]],
            SwA["Radius"].data[indsN["A"]] * 1e-3,
            self.Ri,
        )
        trackCLatN, trackCLonN = getLocalDipoleFPtrack2D(
            SwC["magLat"].data[indsN["C"]],
            SwC["magLon"].data[indsN["C"]],
            SwC["Radius"].data[indsN["C"]] * 1e-3,
            self.Ri,
        )
        temp = np.array([np.min([np.min(trackALatN), np.min(trackCLatN)])])

        self.secs2DcfNorth.create(
            np.concatenate((temp - 1, trackALatN)),
            np.insert(trackALonN, 0, trackALonN[0]),
            np.concatenate((temp - 1, trackCLatN)),
            np.insert(trackCLonN, 0, trackCLonN[0]),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D,
            self.extLon2D,
            self.poleLat,
            self.poleLon,
        )

        # get outer shell for remote curl free locations
        Rlat, Rlon, Rangle, _, _ = auto.sub_Swarm_grids(
            np.concatenate((temp - 1, trackALatN)),
            np.insert(trackALonN, 0, trackALonN[0]),
            np.concatenate((temp - 1, trackCLatN)),
            np.insert(trackCLonN, 0, trackCLonN[0]),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D + self.cfremoteN,
            self.extLon2D + self.cfremoteN,
        )
        Rlon = Rlon % 360

        N1, N2 = self.secs2DcfNorth.ggLat.shape
        mask = np.ones(Rlat.shape)

        mask[
            self.cfremoteN : self.cfremoteN + N1, self.cfremoteN : self.cfremoteN + N2
        ] = 0

        self.secs2DcfRemoteNorth.magLat = Rlat[mask > 0]
        self.secs2DcfRemoteNorth.magLon = Rlon[mask > 0]
        self.secs2DcfRemoteNorth.angle2D = Rangle[mask > 0]

        # South
        trackALatS, trackALonS = getLocalDipoleFPtrack2D(
            SwA["magLat"].data[indsS["A"]],
            SwA["magLon"].data[indsS["A"]],
            SwA["Radius"].data[indsS["A"]] * 1e-3,
            self.Ri,
        )
        trackCLatS, trackCLonS = getLocalDipoleFPtrack2D(
            SwC["magLat"].data[indsS["C"]],
            SwC["magLon"].data[indsS["C"]],
            SwC["Radius"].data[indsS["C"]] * 1e-3,
            self.Ri,
        )

        temp = np.array([np.max([np.max(trackALatS), np.max(trackCLatS)])])
        self.secs2DcfSouth.create(
            np.concatenate((temp + 1, trackALatS)),
            np.insert(trackALonS, 0, trackALonS[0]),
            np.concatenate((temp + 1, trackCLatS)),
            np.insert(trackCLonS, 0, trackCLonS[0]),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D,
            self.extLon2D,
            self.poleLat,
            self.poleLon,
        )

        Rlat, Rlon, Rangle, _, _ = auto.sub_Swarm_grids(
            np.concatenate((temp + 1, trackALatS)),
            np.insert(trackALonS, 0, trackALonS[0]),
            np.concatenate((temp + 1, trackCLatS)),
            np.insert(trackCLonS, 0, trackCLonS[0]),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D + self.cfremoteN,
            self.extLon2D + self.cfremoteN,
        )
        Rlon = Rlon % 360

        N1, N2 = self.secs2DcfSouth.ggLat.shape
        mask = np.ones(Rlat.shape)
        mask[
            self.cfremoteN : self.cfremoteN + N1, self.cfremoteN : self.cfremoteN + N2
        ] = 0
        self.secs2DcfRemoteSouth.magLat = Rlat[mask > 0]
        self.secs2DcfRemoteSouth.magLon = Rlon[mask > 0]
        self.secs2DcfRemoteSouth.angle2D = Rangle[mask > 0]

        # self.ggLat1D,_ = auto.sub_Swarm_grids_1D(lat1,lat2,)


def getLocalDipoleFPtrack1D(latB, rB, Ri):
    """Finds the local dipole footpoints for the 1D curl-free grid creation.
    Parameters
    ----------
    latB : ndarrar
        Magnetic latitude of the satellite, [degree].
    rB : ndarray
        Geocentric radius of the satellite, [km].
    Ri : int or float
        Assumed radius of the spherical shell where the currents flow, [km].
    Returns
    -------
    track : ndarray
        Latitude of the local dipole footpoints of the satellite, [degree].
    """

    # Use the LOCAL DIPOLE footpoints in grid construction
    thetaB = (90 - latB) / 180 * np.pi
    thetaFP = np.arcsin(
        np.sqrt(Ri / rB) * np.sin(thetaB)
    )  # co-latitude of the footpoints, mapped to the northern hemisphere
    ind = np.nonzero(thetaB > np.pi / 2)
    thetaFP[ind] = np.pi - thetaFP[ind]
    latFP = 90 - thetaFP / np.pi * 180
    track = np.arange(np.min(np.abs(latFP)), np.max(1 + np.abs(latFP)), 0.2)

    return track


def getLocalDipoleFPtrack2D(latB, lonB, rB, Ri):
    """Finds the local dipole footpoints for the 2D curl-free grid creation.
    Parameters
    ----------
    latB : ndarray
        Magnetic latitude of the satellite, [degree].
    lonB : ndarray
        Magnetic longitude of the satellite, [degree].
    rB : ndarray
        Geocentric radius of the satellite, [km].
    Ri : int or float
        Assumed radius of the spherical shell where the currents flow, [km].
    Returns
    -------
    latFP, lonFP : ndarray
        Latidue and longitude of the local dipole footpoints of the satellite, [degree].
    """

    latB = np.squeeze(latB)
    lonB = np.squeeze(lonB)
    rB = np.squeeze(rB)
    thetaB = (90 - latB.flatten()) / 180 * np.pi

    thetaFP = np.arcsin(
        np.sqrt(Ri / rB) * np.sin(thetaB)
    )  # co-latitude of the footpoints, mapped to the northern hemisphere

    ind = np.nonzero(thetaB > np.pi / 2)

    thetaFP[ind] = np.pi - thetaFP[ind]

    latFP = 90 - thetaFP / np.pi * 180
    lonFP = lonB
    sorted = np.argsort(np.abs(latFP))
    latFP = latFP[sorted]
    lonFP = lonFP[sorted]
    # probe which hemisphere is being processed
    if np.mean(latB) > 0:

        inds = latFP > 0
        latFP = latFP[inds]
        lonFP = lonFP[inds]
    else:
        inds = latFP < 0
        latFP = latFP[inds]
        lonFP = lonFP[inds]

    return latFP, lonFP


def mag_transform_dsecs(SwA, SwC, pole_lat, pole_lon):
    """Rotates the data to a magnetic coordinate systems.
    Parameters
    ----------
    SwA, SwC : xarray
        Swarm A and C datasets.
    pole_lat, pole_lon : float
        Latitude and longitude of the new pole in the old coordinates, [degree].
    Returns
    -------
    SwA, SwC : xarray
        Swarm A and C datasets including data rotated to the magnetic coordinate system.
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
        -SwA["B_NEC_res"].sel(NEC="N").data,
        SwA["B_NEC_res"].sel(NEC="E").data,
    )  # SwA locations & data
    cmaglat, cmaglon, cmagbt, cmagbp = sph2sph(
        pole_lat,
        pole_lon,
        SwC["Latitude"].data,
        SwC["Longitude"].data,
        -SwC["B_NEC_res"].sel(NEC="N").data,
        SwC["B_NEC_res"].sel(NEC="E").data,
    )

    SwA = SwA.assign(
        {
            "magUvT": ("Timestamp", auvt),
            "magUvP": ("Timestamp", auvp),
            "magLat": ("Timestamp", amaglat),
            "magLon": ("Timestamp", amaglon),
            "magBt": ("Timestamp", amagbt),
            "magBp": ("Timestamp", amagbp),
        }
    )
    SwC = SwC.assign(
        {
            "magUvT": ("Timestamp", cuvt),
            "magUvP": ("Timestamp", cuvp),
            "magLat": ("Timestamp", cmaglat),
            "magLon": ("Timestamp", cmaglon),
            "magBt": ("Timestamp", cmagbt),
            "magBp": ("Timestamp", cmagbp),
        }
    )

    # SwC locations & data
    SwA["magLon"] = SwA["magLon"] % 360
    SwC["magLon"] = SwC["magLon"] % 360

    return SwA, SwC


def trim_data(SwA, SwC):
    """Finds a period with suitable spaceraft velocity for analysis.
    Parameters
    ----------
    SwA, SwC : xarray
        Swarm A and C datasets.
    Returns
    -------
    SwA, SwC : xarray
        Swarm A and C datasets trimmed to the suitable period.
    """

    Vx = np.gradient(SwA["magLat"])  # northward velocity [deg/step]
    Vy = np.gradient(SwA["magLon"]) * np.cos(
        np.radians(SwA["magLat"])
    )  # eastward velocity [deg/step]
    _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
    SwA = SwA.sel(Timestamp=SwA["Timestamp"][ind])

    Vx = np.gradient(SwC["magLat"])
    Vy = np.gradient(SwC["magLon"]) * np.cos(np.radians(SwC["magLat"]))
    _, ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))

    # replace with indexing whole dataset
    SwC = SwC.sel(Timestamp=SwC["Timestamp"][ind])

    return SwA, SwC


def get_exclusion_zone(SwA, SwC):

    apexcrossing_a = np.where(
        np.sign(SwA.ApexLatitude)[:-1].data != np.sign(SwA.ApexLatitude)[1:].data
    )[0][0]
    apexcrossing_c = np.where(
        np.sign(SwC.ApexLatitude)[:-1].data != np.sign(SwC.ApexLatitude)[1:].data
    )[0][0]

    date = pandas.to_datetime(SwA.Timestamp).mean().to_pydatetime()
    apex_out = apexpy.Apex(date=date)
    alat, alon = apex_out.convert(
        SwA.ApexLatitude, SwA.ApexLongitude, "apex", "geo", height=130
    )
    clat, clon = apex_out.convert(
        SwC.ApexLatitude, SwC.ApexLongitude, "apex", "geo", height=130
    )

    valsA = alat[apexcrossing_a : apexcrossing_a + 2]
    valsC = clat[apexcrossing_c : apexcrossing_c + 2]

    minval = np.max([np.min(valsA), np.min(valsC)])
    maxval = np.min([np.max(valsA), np.max(valsC)])

    return maxval, minval


class dsecsdata:
    """Class for DSECS variables and fitting procedures"""

    def __init__(self):
        """Initialize all the necessary variables."""
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
        self.alpha1Ddf: float = 1e-5
        self.epsSVD1Ddf: float = 4e-4
        self.alpha1Dcf: float = 20e-5
        self.epsSVD1Dcf: float = 100e-4
        self.alpha2D: float = 50e-5
        self.epsSVD2D: float = 10e-4
        self.epsSVD2dcf: float = 20e-4
        self.alpha2Dcf: float = 10e-1
        self.df1D = np.ndarray([])
        self.df2D = np.ndarray([])
        self.cf1D = np.ndarray([])
        self.cf2D = np.ndarray([])
        self.df2dBr = np.ndarray([])
        self.df2dBp = np.ndarray([])
        self.df2dBt = np.ndarray([])
        self.df1dBt = np.ndarray([])
        self.df1dBr = np.ndarray([])
        self.matBr1Ddf = np.ndarray([])
        self.matBt1Ddf = np.ndarray([])
        self.df1DJp = np.ndarray([])
        self.matBpara1D = np.ndarray([])
        self.matBpara2D = np.ndarray([])
        self.apexlats = np.ndarray([])
        self.cf1dDipMagJp = np.ndarray([])
        self.cf1dDipMagJt = np.ndarray([])
        self.cf1dDipJr = np.ndarray([])
        self.cf1dDipBr = np.ndarray([])
        self.cf1dDipMagBt = np.ndarray([])
        self.cf1dDipMagBp = np.ndarray([])
        self.cf2dDipMagJp = np.ndarray([])
        self.cf2dDipMagJt = np.ndarray([])
        self.cf2dDipJr = np.ndarray([])
        self.cf2dDipBr = np.ndarray([])
        self.cf2dDipMagBt = np.ndarray([])
        self.cf2dDipMagBp = np.ndarray([])
        self.matBp2Ddf = np.ndarray([])
        self.matBt2Ddf = np.ndarray([])
        self.matBr2Ddf = np.ndarray([])
        self.df2dMagJt = np.ndarray([])
        self.df2dMagJp = np.ndarray([])
        self.remoteCf2dDipMagBr = np.ndarray([])
        self.remoteCf2dDipMagBt = np.ndarray([])
        self.remoteCf2dDipMagBp = np.ndarray([])
        self.remoteCf2dDipMagJp = np.ndarray([])
        self.remoteCf2dDipMagJt = np.ndarray([])
        self.remoteCf2dDipMagJr = np.ndarray([])
        self.test = dict()
        self.flag = 0
        self.QDlats = np.ndarray([])
        self.apexcrossingA: float = 0
        self.apexcrossingC: float = 0
        self.exclusionmax = 0
        self.exclusionmin = 0
        self.SwA = None
        self.SwC = None

    def populate(self, SwA, SwC):
        """Initializes a DSECS analaysis case from Swarm data.
        Parameters
        ----------
        SwA, SwC : xarray
            Swarm A and C datasets.
        """

        # initialize grid
        grid = dsecsgrid()
        grid.FindPole(SwA)

        # calculate additional variables
        SwA, SwC = getUnitVectors(SwA, SwC)

        SwA, SwC = mag_transform_dsecs(SwA, SwC, grid.poleLat, grid.poleLon)
        # trim the data if needed

        SwA, SwC = trim_data(SwA, SwC)

        # create grid
        self.test["swa"] = SwA
        self.test["swc"] = SwC
        grid.create(SwA, SwC)

        if grid.flag != 0:
            logger.warn("Could not create grid. No analysis performed.")
            self.flag = 1
            return

        self.latB = np.concatenate((SwA["magLat"], SwC["magLat"]))
        self.lonB = np.concatenate((SwA["magLon"], SwC["magLon"]))
        self.apexlats = np.concatenate((SwA["ApexLatitude"], SwC["ApexLatitude"]))
        self.QDlats = np.concatenate((SwA["QDLat"], SwC["QDLat"]))
        self.rB = np.concatenate((SwA["Radius"], SwC["Radius"])) * 1e-3
        B = np.concatenate((SwA["B_NEC_res"], SwC["B_NEC_res"]))
        self.Bt = -B[:, 0]
        self.Bp = B[:, 1]
        self.Br = -B[:, 2]
        self.magBp = np.concatenate((SwA["magBp"], SwC["magBp"]))
        self.magBt = np.concatenate((SwA["magBt"], SwC["magBt"]))
        self.Bpara = np.concatenate((SwA["B_para_res"], SwC["B_para_res"]))
        self.grid = grid
        self.uvR = np.concatenate((SwA["UvR"], SwC["UvR"]))
        self.uvT = np.concatenate((SwA["magUvT"], SwC["magUvT"]))
        self.uvP = np.concatenate((SwA["magUvP"], SwC["magUvP"]))

        outproto = self.grid.out.magLat
        dataproto = self.latB
        self.cf1dDipMagJp = 0 * np.empty_like(outproto)
        self.cf1dDipMagJt = 0 * np.empty_like(outproto)
        self.cf1dDipJr = 0 * np.empty_like(outproto)
        self.cf1dDipBr = 0 * np.empty_like(dataproto)
        self.cf1dDipMagBt = 0 * np.empty_like(dataproto)
        self.cf1dDipMagBp = 0 * np.empty_like(dataproto)
        self.cf1dDipBr = 0 * np.empty_like(dataproto)
        self.cf1dDipMagBt = 0 * np.empty_like(dataproto)
        self.cf1dDipMagBp = 0 * np.empty_like(dataproto)
        self.cf2dDipMagJp = 0 * np.empty_like(outproto)
        self.cf2dDipMagJt = 0 * np.empty_like(outproto)
        self.cf2dDipJr = 0 * np.empty_like(outproto)
        self.cf2dDipBr = 0 * np.empty_like(dataproto)
        self.cf2dDipMagBt = 0 * np.empty_like(dataproto)
        self.cf2dDipMagBp = 0 * np.empty_like(dataproto)
        self.remoteCf2dDipMagBr = 0 * np.empty_like(dataproto)
        self.remoteCf2dDipMagBt = 0 * np.empty_like(dataproto)
        self.remoteCf2dDipMagBp = 0 * np.empty_like(dataproto)
        self.remoteCf2dDipMagJp = 0 * np.empty_like(outproto)
        self.remoteCf2dDipMagJt = 0 * np.empty_like(outproto)
        self.remoteCf2dDipMagJr = 0 * np.empty_like(outproto)

        self.exclusionmax, self.exclusionmin = get_exclusion_zone(SwA, SwC)
        self.SwA = SwA
        self.SwC = SwC

    def fit1D_df(self):
        """1D divergence-free fitting."""

        self.matBr1Ddf, self.matBt1Ddf = SECS_1D_DivFree_magnetic(
            self.latB, self.grid.secs1Ddf.lat, self.rB, self.grid.Ri, 500
        )

        y = self.Bpara  # measurement, parallel magnetic field

        self.matBpara1D = (self.matBr1Ddf.T * self.uvR).T + (
            self.matBt1Ddf.T * self.uvT
        ).T  #

        regmat = self.grid.secs1Ddf.diff2  # regularization
        x = auto.sub_inversion(
            self.matBpara1D, regmat, self.epsSVD1Ddf, self.alpha1Ddf, y
        )
        matJp = SECS_1D_DivFree_vector(
            self.grid.out.magLat, self.grid.secs1Ddf.lat, self.grid.Ri
        )
        self.df1DJp = np.reshape(matJp @ x, self.grid.out.ggLat.shape)
        self.df1dBr = self.matBr1Ddf @ x
        self.df1dBt = self.matBt1Ddf @ x
        self.df1D = x

        return x, y, self.matBpara1D

    def fit2D_df(self):
        """2D divergence-free fitting."""

        # Calculate B-matrices and form the field-aligned matrix
        thetaB = (90 - self.latB) / 180 * np.pi
        phiB = self.lonB / 180 * np.pi
        theta2D = (90 - self.grid.secs2Ddf.magLat) / 180 * np.pi
        phi2D = self.grid.secs2Ddf.magLon / 180 * np.pi

        matBr2D, matBt2D, matBp2Ddf = SECS_2D_DivFree_magnetic(
            thetaB, phiB, theta2D, phi2D, self.rB, self.grid.Ri
        )
        # N2d = np.size(self.grid.secs2D.lat)

        self.matBpara2D = (
            (matBr2D.T * self.uvR).T
            + (matBt2D.T * self.uvT).T
            + (matBp2Ddf.T * self.uvP).T
        )

        # Remove field explained by the 1D DF SECS (must have been fitted earlier).

        Bpara2D = self.Bpara - self.matBpara1D @ self.df1D

        regmat = self.grid.secs2Ddf.diff2lat2D  # regularization
        self.df2D = auto.sub_inversion(
            self.matBpara2D, regmat, self.epsSVD2D, self.alpha2D, Bpara2D
        )

        # Calculate the current produced by the 2D DF SECS.
        thetaJ = (90 - self.grid.out.magLat.flatten()) / 180 * np.pi
        phiJ = self.grid.out.magLon.flatten() / 180 * np.pi
        matJt, matJp = SECS_2D_DivFree_vector(
            thetaJ, phiJ, theta2D, phi2D, self.grid.Ri, self.grid.secs2Ddf.angle2D
        )
        self.df2dMagJt = np.reshape(matJt @ self.df2D, self.grid.out.magLat.shape)
        self.df2dMagJp = np.reshape(matJp @ self.df2D, self.grid.out.magLat.shape)

        # Calculate the magnetic field produced by the 2D DF SECS
        # still split into SwA and SwC
        self.df2dBr = matBr2D @ self.df2D
        self.df2dBp = matBp2Ddf @ self.df2D
        self.df2dBt = matBt2D @ self.df2D
        self.matBr2Ddf = matBr2D
        self.matBp2Ddf = matBp2Ddf
        self.matBt2Ddf = matBt2D
        # self.df2dBp np.ndarray([]) = self.matBp2Ddf @ self.df2D
        self.df2dBpara = self.matBpara2D @ self.df2D

        return self.df2D, Bpara2D, self.matBpara2D, self.df2dBr

    def analyze(self):
        """Perform the DSECS analysis steps."""

        self.fit1D_df()
        self.fit2D_df()
        self.fit1D_cf()
        self.fit2D_cf()

    def fit1D_cf(self):
        """1D curl-free fitting."""
        # split data into hemispheres
        # ind = np.nonzero(.............)
        indN = np.nonzero(self.apexlats > 0)
        indRN = np.nonzero(self.grid.out.magLat > 0)
        indS = np.nonzero(self.apexlats <= 0)
        indRS = np.nonzero(self.grid.out.magLat <= 0)
        Bp = self.magBp - self.df2dBp
        for ind, rind, gridhem in zip(
            [indN, indS],
            [indRN, indRS],
            [self.grid.secs1DcfNorth, self.grid.secs1DcfSouth],
        ):
            latB_hem = self.latB[ind]
            rB_hem = self.rB[ind]

            Bphem = Bp[ind].flatten()
            #        #Remove effect of 1D & 2D DF currents (must have been fitted earlier)
            # a=SwA.magBp(indA)-SwA.df1dMagBp(indA)-SwA.df2dMagBp(indA);
            # SwA.df1dMagBp = 0 by definition
            #        #Calculate the B-matrix that gives magnetic field at Swarm orbit from the SECS.
            # The symmetric system is a combination of northern and southern hemisphere systems.
            matBp = SECS_1D_CurlFree_magnetic(
                latB_hem, gridhem.lat, rB_hem, self.grid.Ri, 0
            ) - SECS_1D_CurlFree_magnetic(
                latB_hem, -gridhem.lat, rB_hem, self.grid.Ri, 0
            )
            #        #Fit the dipolar 1D CF SECS. Use zero constraint on the 2nd latitudinal derivative.
            regmat = gridhem.diff2  # regularization
            cf1D = auto.sub_inversion(
                matBp, regmat, self.epsSVD1Dcf, self.alpha1Dcf, Bphem
            )
            # Calculate the theta-current (southward) produced by the 1D CF SECS.
            latJ = self.grid.out.magLat.flatten()
            matJt = SECS_1D_CurlFree_vector(
                latJ, gridhem.lat, self.grid.Ri
            ) - SECS_1D_CurlFree_vector(latJ, -gridhem.lat, self.grid.Ri)
            tmp = np.reshape(matJt @ cf1D, self.grid.out.magLat.shape)
            self.cf1dDipMagJt[rind] = tmp[rind]
            #        #Calculate the radial current produced by the 1D CF SECS using finite differences.
            # For FAC we should scale by 1/sin(inclination).
            dlat = self.grid.dlatOut  # step size in latitude
            dtheta = dlat / 180 * np.pi
            thetaJ = (90 - latJ) / 180 * np.pi
            apuNorth = (
                SECS_1D_CurlFree_vector(latJ + dlat, gridhem.lat, self.grid.Ri)
                - SECS_1D_CurlFree_vector(latJ + dlat, -gridhem.lat, self.grid.Ri)
            ) @ cf1D
            apuSouth = (
                SECS_1D_CurlFree_vector(latJ - dlat, gridhem.lat, self.grid.Ri)
                - SECS_1D_CurlFree_vector(latJ - dlat, -gridhem.lat, self.grid.Ri)
            ) @ cf1D
            tmp = -(
                np.sin(thetaJ + dtheta) * apuSouth - np.sin(thetaJ - dtheta) * apuNorth
            ) / (
                2 * dtheta * self.grid.Ri * np.sin(thetaJ)
            )  # radial current = -div
            tmp = np.reshape(
                tmp * 1000, self.grid.out.magLat.shape
            )  # A/km^2 --> nA/m^2
            self.cf1dDipJr[rind] = tmp[rind]
            #        #Calculate the magnetic field produced by the 1D CF SECS at the Swarm satellites.
            # There is only eastward field, so the other components remain zero (as formatted).
            # need to split into SwA and SwC
            self.cf1dDipMagBp[ind] = matBp @ cf1D
        return cf1D, Bp, matBp, Bphem

    def fit2D_cf(self):
        """2D curl-free fitting."""
        # grid generation (also remote grid) missing
        # Fit is done to all components of the magnetic disturbance.
        # Remove field explained by the 1D & 2D DF SECS and 1D CF SECS (dipolar)
        Br = self.Br - self.df1dBr - self.df2dBr
        Bt = self.magBt - self.df1dBt - self.df2dBt - self.cf1dDipMagBt
        # df1dMagBp = 0 by definition
        Bp = self.magBp - self.df2dBp - self.cf1dDipMagBp

        thetaB = (90 - self.latB) / 180 * np.pi
        phiB = self.lonB / 180 * np.pi

        indN = np.nonzero(self.apexlats > 0)
        indRN = np.nonzero(self.grid.out.magLat > 0)
        indS = np.nonzero(self.apexlats <= 0)
        indRS = np.nonzero(self.grid.out.magLat <= 0)

        # add indN and indRN to class
        for ind, rind, gridhem, remote_hem, label in zip(
            [indN, indS],
            [indRN, indRS],
            [self.grid.secs2DcfNorth, self.grid.secs2DcfSouth],
            [self.grid.secs2DcfRemoteNorth, self.grid.secs2DcfRemoteSouth],
            ["north", "south"],
        ):
            # need to flatten these?
            Br_hem = Br[ind]
            Bt_hem = Bt[ind]
            Bp_hem = Bp[ind]
            thetaB_hem = thetaB[ind]
            phiB_hem = phiB[ind]
            rB_hem = self.rB[ind]

            # Calculate B-matrices.
            theta2D = (90 - gridhem.magLat) / 180 * np.pi
            phi2D = gridhem.magLon / 180 * np.pi
            # should rename those to cf and df?
            matBr2D, matBt2D, matBp2D = SECS_2D_CurlFree_antisym_magnetic(
                thetaB_hem,
                phiB_hem,
                theta2D,
                phi2D,
                rB_hem,
                self.grid.Ri,
                gridhem.angle2D,
            )

            # ask Heikki if we should keep this switch
            # if result.remoteCFdip:
            # if True:
            # B-matrices for remote grid
            # split remote grid into north and south
            remoteTheta = (90 - remote_hem.magLat) / 180 * np.pi
            remotePhi = remote_hem.magLon / 180 * np.pi

            (
                remoteMatBr2D,
                remoteMatBt2D,
                remoteMatBp2D,
            ) = SECS_2D_CurlFree_antisym_magnetic(
                thetaB_hem,
                phiB_hem,
                remoteTheta,
                remotePhi,
                rB_hem,
                self.grid.Ri,
                remote_hem.angle2D,
            )
            # Fit scaling factors of the remote 2D CF SECS
            remoteEps = 0.5

            remoteIcf = auto.sub_inversion(
                np.vstack((remoteMatBr2D, remoteMatBt2D, remoteMatBp2D)),
                np.array([]),
                remoteEps,
                np.nan,
                np.concatenate((Br_hem, Bt_hem, Bp_hem)).squeeze(),
            )
            self.test[label + "_dataremote"] = np.concatenate(
                (Br_hem, Bt_hem, Bp_hem)
            ).squeeze()
            self.test[label + "_remotex"] = remoteIcf
            self.test[label + "_remotefit"] = (
                np.vstack((remoteMatBr2D, remoteMatBt2D, remoteMatBp2D)) @ remoteIcf
            )

            # Remove effect from the measured B

            Br_hem = Br_hem.squeeze() - (remoteMatBr2D @ remoteIcf).squeeze()
            Bt_hem = Bt_hem.squeeze() - (remoteMatBt2D @ remoteIcf).squeeze()
            Bp_hem = Bp_hem.squeeze() - (remoteMatBp2D @ remoteIcf).squeeze()

            # Fit the (local) 2D CF SECS. Use zero constraint on the 2nd latitudinal derivative.
            regmat = gridhem.diff2lat2D  # regularization

            Icf = auto.sub_inversion(
                np.vstack((matBr2D, matBt2D, matBp2D)),
                regmat,
                self.epsSVD2dcf,
                self.alpha2Dcf,
                np.concatenate((Br_hem, Bt_hem, Bp_hem)).squeeze(),
            )
            self.test[label + "_data"] = np.concatenate(
                (Br_hem, Bt_hem, Bp_hem)
            ).squeeze()
            # Calculate the horizontal current produced by the 2D CF SECS.
            thetaJ = (90 - self.grid.out.magLat.flatten()) / 180 * np.pi
            phiJ = self.grid.out.magLon.flatten() / 180 * np.pi
            matJt, matJp = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiJ, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
            )
            tmp = np.reshape(matJt @ Icf, self.grid.out.magLat.shape)
            self.cf2dDipMagJt[rind] = tmp[rind]
            tmp = np.reshape(matJp @ Icf, self.grid.out.magLat.shape)
            self.cf2dDipMagJp[rind] = tmp[rind]
            self.test[label + "_x"] = Icf
            self.test[label + "_fit"] = np.vstack((matBr2D, matBt2D, matBp2D)) @ Icf
            self.test[label + "_matJp"] = matJp
            self.test[label + "_matJt"] = matJt
            self.test[label + "_matX"] = np.vstack((matBr2D, matBt2D, matBp2D))
            self.test[label + "_remotematX"] = np.vstack(
                (remoteMatBr2D, remoteMatBt2D, remoteMatBp2D)
            )
            self.test[label + "_difflat"] = regmat
            self.test[label + "dats"] = [thetaB_hem, phiB_hem, theta2D, phi2D]
            # Calculate the radial current produced by the 2D CF SECS using finite differences.
            # For FAC we should scale by 1/sin(inclination).
            thetaNorth = thetaJ - self.grid.dlatOut / 180 * np.pi
            thetaSouth = thetaJ + self.grid.dlatOut / 180 * np.pi
            matJtNorth, _ = SECS_2D_CurlFree_antisym_vector(
                thetaNorth, phiJ, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
            )
            matJtSouth, _ = SECS_2D_CurlFree_antisym_vector(
                thetaSouth, phiJ, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
            )

            phiEast = phiJ + self.grid.dlatOut / 180 * np.pi
            phiWest = phiJ - self.grid.dlatOut / 180 * np.pi
            _, matJpEast = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiEast, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
            )
            _, matJpWest = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiWest, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
            )
            aux1 = (
                np.sin(thetaSouth) * (matJtSouth @ Icf)
                - np.sin(thetaNorth) * (matJtNorth @ Icf)
            ) / (thetaSouth - thetaNorth)
            aux2 = ((matJpEast - matJpWest) @ Icf) / (phiEast - phiWest)
            tmp = -(aux1 + aux2) / (
                self.grid.Ri * np.sin(thetaJ)
            )  # radial current = -div
            tmp = np.reshape(
                tmp * 1000, self.grid.out.magLat.shape
            )  # A/km^2 --> nA/m^2
            self.cf2dDipJr[rind] = tmp[rind]

            # Calculate the magnetic field produced by the 2D CF SECS.

            np.put(self.cf2dDipBr, ind, matBr2D @ Icf)
            np.put(self.cf2dDipMagBt, ind, matBt2D @ Icf)
            np.put(self.cf2dDipMagBp, ind, matBp2D @ Icf)

            # keep this switch?
            # if result.remoteCFdip:
            # if True:
            # Current produced by the remote SECS
            remoteAngle = remote_hem.angle2D
            matJt, matJp = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiJ, remoteTheta, remotePhi, self.grid.Ri, remoteAngle
            )
            tmp = np.reshape(matJt @ remoteIcf, self.grid.out.magLat.shape)
            self.remoteCf2dDipMagJt[rind] = tmp[rind]
            tmp = np.reshape(matJp @ remoteIcf, self.grid.out.magLat.shape)
            self.remoteCf2dDipMagJp[rind] = tmp[rind]
            matJtNorth, _ = SECS_2D_CurlFree_antisym_vector(
                thetaNorth, phiJ, remoteTheta, remotePhi, self.grid.Ri, remoteAngle
            )
            matJtSouth, _ = SECS_2D_CurlFree_antisym_vector(
                thetaSouth, phiJ, remoteTheta, remotePhi, self.grid.Ri, remoteAngle
            )
            _, matJpEast = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiEast, remoteTheta, remotePhi, self.grid.Ri, remoteAngle
            )
            _, matJpWest = SECS_2D_CurlFree_antisym_vector(
                thetaJ, phiWest, remoteTheta, remotePhi, self.grid.Ri, remoteAngle
            )
            aux1 = (
                np.sin(thetaSouth) * (matJtSouth @ remoteIcf)
                - np.sin(thetaNorth) * (matJtNorth @ remoteIcf)
            ) / (thetaSouth - thetaNorth)
            aux2 = ((matJpEast - matJpWest) @ remoteIcf) / (phiEast - phiWest)
            tmp = -(aux1 + aux2) / (self.grid.Ri * np.sin(thetaJ))
            tmp = np.reshape(tmp * 1000, self.grid.out.magLat.shape)
            self.remoteCf2dDipMagJr[rind] = tmp[rind]

            self.test[label + "_lats"] = self.grid.out.magLat[rind]
            self.test[label + "_rind"] = rind

            # Then their magnetic field
            np.put(self.remoteCf2dDipMagBr, ind, remoteMatBr2D @ remoteIcf)
            np.put(self.remoteCf2dDipMagBt, ind, remoteMatBt2D @ remoteIcf)
            np.put(self.remoteCf2dDipMagBp, ind, remoteMatBp2D @ remoteIcf)

        return (
            Br,
            Bt,
            Bp,
            thetaB_hem,
            phiB_hem,
            theta2D,
            phi2D,
            rB_hem,
            self.grid.Ri,
            gridhem.angle2D,
            np.concatenate((Br_hem, Bt_hem, Bp_hem)).squeeze(),
            np.vstack((matBr2D, matBt2D, matBp2D)),
            regmat,
        )  # no idea what we need to return from all of the above

    def dump(self):
        """Dump the currents to xarrays.
        Returns
        -------
        dsmag : xarray of the results in the magnetic coordinate system.
        dsgeo : xarray of the results in geographic coordinate system.
        fitA : xarray with fitted magnetic field for Swarm A
        fitC : xarray with fitted magnetic field for Swarm C
        """

        # Find the point where

        MagJthetaDf = (
            # self.cf1dDipMagJt
            +self.df2dMagJt
            # + self.cf2dDipMagJt
            # + self.remoteCf2dDipMagJt
        )
        MagJphiDf = (
            self.df1DJp
            + self.df2dMagJp  # + self.cf2dDipMagJp + self.remoteCf2dDipMagJp
        )

        MagJthetaCf = (
            self.cf1dDipMagJt
            # + self.df2dMagJt
            + self.cf2dDipMagJt
            + self.remoteCf2dDipMagJt
        )
        MagJphiCf = (
            self.cf2dDipMagJp
            + self.remoteCf2dDipMagJp
            # self.df1DJp + #self.df2dMagJp + self.cf2dDipMagJp + self.remoteCf2dDipMagJp
        )

        badlats = np.nonzero(
            (self.grid.out.ggLat < self.exclusionmax)
            & (self.grid.out.ggLat > self.exclusionmin)
        )

        MagJphiCf[badlats] = np.nan
        MagJthetaCf[badlats] = np.nan

        # MagJtheta = (
        #    self.cf1dDipMagJt
        #    + self.df2dMagJt
        #    + self.cf2dDipMagJt
        #    + self.remoteCf2dDipMagJt
        # )
        # MagJphi = (
        #    self.df1DJp + self.df2dMagJp + self.cf2dDipMagJp + self.remoteCf2dDipMagJp
        # )

        # MagJtheta = MagJthetaCf + MagJthetaDf
        # MagJphi = MagJphiDf + MagJphiCf

        Brfit = self.df1dBr + self.df2dBr + self.cf2dDipBr + self.remoteCf2dDipMagBr
        Btfit = self.df1dBt + self.df2dBt + self.cf2dDipMagBt + self.remoteCf2dDipMagBt
        Bpfit = (
            self.df2dBp
            + self.cf1dDipMagBp
            + self.cf2dDipMagBt
            + self.remoteCf2dDipMagBp
        )

        _, _, geoJthetaDf, geoJphiDf = sph2sph(
            self.grid.poleLat,
            0,
            self.grid.out.magLat,
            self.grid.out.magLon,
            MagJthetaDf,
            MagJphiDf,
        )

        _, _, geoJthetaCf, geoJphiCf = sph2sph(
            self.grid.poleLat,
            0,
            self.grid.out.magLat,
            self.grid.out.magLon,
            MagJthetaCf,
            MagJphiCf,
        )

        geoJtheta = geoJthetaCf + geoJthetaDf
        geoJphi = geoJphiDf + geoJphiCf

        Jr = self.cf1dDipJr + self.cf2dDipJr + self.remoteCf2dDipMagJr
        Jr[badlats] = np.nan
        dsmag = xr.Dataset(
            data_vars=dict(
                JphiDf=(["x", "y"], MagJphiDf),
                JthetaDf=(["x", "y"], MagJthetaDf),
                Jr=(["x", "y"], Jr),
                JphiCf=(["x", "y"], MagJphiCf),
                JthetaCf=(["x", "y"], MagJthetaCf),
            ),
            coords=dict(
                Latitude=(["x", "y"], self.grid.out.magLon),
                Longitude=(["x", "y"], self.grid.out.magLat),
            ),
            attrs={
                "description": "DSECS result mag",
                "Ionospheric height (km)": self.grid.Ri,
            },
        )

        dsgeo = xr.Dataset(
            data_vars=dict(
                JEastDf=(
                    ["x", "y"],
                    geoJphiDf,
                    {
                        "unit": "A/km",
                        "description": "Div free Eastward current densoty",
                    },
                ),
                JNorthDf=(
                    ["x", "y"],
                    -geoJthetaDf,
                    {
                        "unit": "A/km",
                        "description": "Div free Northward current density",
                    },
                ),
                Jr=(
                    ["x", "y"],
                    Jr,
                    {"unit": "nA/m^2", "description": "Radial current density"},
                ),
                JEastCf=(
                    ["x", "y"],
                    geoJphiCf,
                    {
                        "unit": "A/km",
                        "description": "Curl free Eastward current density",
                    },
                ),
                JNorthCf=(
                    ["x", "y"],
                    -geoJthetaCf,
                    {
                        "unit": "A/km",
                        "description": "Curl free Northward current density",
                    },
                ),
                JEastTotal=(
                    ["x", "y"],
                    geoJphi,
                    {"unit": "A/km", "description": "Total Eastward current density"},
                ),
                JNorthTotal=(
                    ["x", "y"],
                    -geoJtheta,
                    {"unit": "A/km", "description": "Total Northward current density"},
                ),
            ),
            coords=dict(
                Longitude=(["x", "y"], self.grid.out.ggLon, {"unit": "deg"}),
                Latitude=(["x", "y"], self.grid.out.ggLat, {"unit": "deg"}),
            ),
            attrs={
                "description": "DSECS result geographic coordinates",
                "ionospheric height (km)": self.grid.Ri,
                "Time interval": np.datetime_as_string(self.SwA.Timestamp[0])
                + " - "
                + np.datetime_as_string(self.SwA.Timestamp[-1]),
                "Mean time": np.datetime_as_string(np.mean(self.SwA.Timestamp)),
            },
        )

        Brfit = self.df1dBr + self.df2dBr + self.cf2dDipBr + self.remoteCf2dDipMagBr
        Btfit = self.df1dBt + self.df2dBt + self.cf2dDipMagBt + self.remoteCf2dDipMagBt
        Bpfit = (
            self.df2dBp
            + self.cf1dDipMagBp
            + self.cf2dDipMagBp
            + self.remoteCf2dDipMagBp
        )

        geoObsLat, geoObsLon, geoBtfit, geoBpfit = sph2sph(
            self.grid.poleLat,
            0,
            self.latB,
            self.lonB,
            Btfit,
            Bpfit,
        )

        Na = len(self.SwA.Latitude)
        BrFitA = Brfit[:Na]
        BtFitA = geoBtfit[:Na]
        BpFitA = geoBpfit[:Na]
        BrFitC = Brfit[Na:]
        BtFitC = geoBtfit[Na:]
        BpFitC = geoBpfit[Na:]

        B_NEC_FitA = np.array([-BtFitA, BpFitA, -BrFitA]).T
        B_NEC_FitC = np.array([-BtFitC, BpFitC, -BrFitC]).T

        fitA = xr.Dataset(
            data_vars=dict(
                B_NEC_fit=(["Timestamp", "NEC"], B_NEC_FitA, {"unit": "nT"}),
                B_NEC_data=(
                    ["Timestamp", "NEC"],
                    self.SwA["B_NEC_res"].data,
                    {"unit": "nT"},
                ),
                Longitude=(["Timestamp"], self.SwA.Longitude.data, {"unit": "deg"}),
                Latitude=(["Timestamp"], self.SwA.Latitude.data, {"unit": "deg"}),
                Radius=(["Timestamp"], self.SwA.Radius.data, {"unit": "m"}),
            ),
            coords=dict(
                Timestamp=(
                    ["Timestamp"],
                    self.SwA.Timestamp.data,
                    {"description": "UTC time"},
                ),
                NEC=(
                    ["NEC"],
                    ["N", "E", "C"],
                    {"description": "North East Center frame"},
                ),
            ),
            attrs={
                "description": "DSECS magnetic field fit.",
                "satellite": "Swarm Alpha",
            },
        )

        fitC = xr.Dataset(
            data_vars=dict(
                B_NEC_fit=(["Timestamp", "NEC"], B_NEC_FitC, {"unit": "nT"}),
                B_NEC_data=(
                    ["Timestamp", "NEC"],
                    self.SwC["B_NEC_res"].data,
                    {"unit": "nT"},
                ),
                Longitude=(["Timestamp"], self.SwC.Longitude.data, {"unit": "deg"}),
                Latitude=(["Timestamp"], self.SwC.Latitude.data, {"unit": "deg"}),
                Radius=(["Timestamp"], self.SwC.Radius.data, {"unit": "m"}),
            ),
            coords=dict(
                Timestamp=(
                    ["Timestamp"],
                    self.SwA.Timestamp.data,
                    {"description": "UTC time"},
                ),
                NEC=(
                    ["NEC"],
                    ["N", "E", "C"],
                    {"description": "North East Center frame"},
                ),
            ),
            attrs={
                "description": "DSECS magnetic field fit.",
                "satellite": "Swarm Charlie",
            },
        )

        return dsmag, dsgeo, fitA, fitC
