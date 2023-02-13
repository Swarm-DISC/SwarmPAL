"""Algorithms for low latitude spherical elementary current system analysis.

Adapted from MatLab code by Heikki Vanhamäki.

"""

import numpy as np
import swarmpal.toolboxes.secs.aux_tools as auto
from swarmpal.toolboxes.secs import SecsInputs
from swarmpal.toolboxes.secs.aux_tools import (
    sph2sph,
    sub_FindLongestNonZero,
    sub_Swarm_grids,
)


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
    t_aux[:, 0] = 0.2 * np.pi / rb * t * np.where(rb <= rsecs, 1, rsecs / rb)

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
    rb = rb.ravel()
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
        ind = np.argwhere(Sin2ThetaPrime > 1e-10)
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

    # ravel and reassign

    # thetaSECS = thetaSECS.ravel()
    # phiSECS = phiSECS.ravel()

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
    diffy = (np.expand_dims(y0, -1) - x).T  # (y0[:,np.newaxis] -y).T
    diffz = (np.expand_dims(z0, -1) - x).T  # (z0[:,np.newaxis] -z).T

    tt = np.sqrt(diffx * diffx + diffy * diffy + diffz * diffz)
    root = tt * tt * tt

    # Remove singularity by setting a lower lomit to the distance between current elements and field points.
    # Use the small number
    root = np.where(root < smallr * smallr * smallr, smallr, root)

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


def get_data_slices(
    t1,
    t2,
    model="IGRF",
):
    """Get data and split it into slices suitable for DSECS analysis.

    Parameters
    ----------
    t1 : _type_, optional
        _description_
    t2 : _type_, optional
        _description_
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
        model=model,
    )

    SwA = auto.get_eq(inputs.s1.xarray)

    SwC = auto.get_eq(inputs.s2.xarray)

    # SwA,SwC = getUnitVectors(SwA,SwC)

    return SwA, SwC


def getUnitVectors(SwA, SwC):
    """Get the magnetic unit vectors."""

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
        """Initialize from data

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
            self.magLon = self.magLon % 360


class grid1D:
    """Simple class to hold a 1D lat grid"""

    def __init__(self):
        self.lat = np.array([])
        self.diff2 = np.array([])

    def create(self, lat1, lat2, dlat, extLat):
        """Initialize from data.

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
    """Class for all the grids needed in DSECS analysis"""

    def __init__(self):
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
        self.extLat1D = 1
        self.indsN = dict()
        self.insdS = dict()
        self.extLat2D = 5
        self.extLon2D = 7
        self.cfremoteN = 3

    def FindPole(self, SwA):
        """Find the best pole location for the analysis"""
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
        """Initialize the grids from data.


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
        for dat, sat in zip([SwA, SwC], ["A", "C"], strict=True):
            indsN[sat] = np.argwhere(dat["ApexLatitude"].data > 0)
            indsS[sat] = np.argwhere(dat["ApexLatitude"].data <= 0)

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
        temp = np.min([np.min(trackALatN), np.min(trackCLatN)])
        self.secs2DcfNorth.create(
            np.append(temp - 1, trackALatN),
            np.append(trackALonN[0], trackALonN),
            np.append(temp - 1, trackCLatN),
            np.append(trackCLonN[0], trackCLonN),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D,
            self.extLon2D,
            self.poleLat,
            self.poleLon,
        )

        # get outer shell for remote curl free locations
        Rlat, Rlon, Rangle, _, _ = auto.sub_Swarm_grids(
            np.append(temp - 1, trackALatN),
            np.append(trackALonN[0], trackALonN),
            np.append(temp - 1, trackCLatN),
            np.append(trackCLonN[0], trackCLonN),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D + self.cfremoteN,
            self.extLon2D + self.cfremoteN,
            self.poleLat,
            self.poleLon,
        )

        N1, N2 = self.secs2DcfNorth.shape
        mask = np.ones(Rlat.shape)
        mask[self.cfremoteN :, self.cfremoteN :][:N1, :N2] = 0
        ind = np.argwhere(mask)
        self.secs2DcfRemoteNorth.magLat = Rlat[ind]
        self.secs2DcfRemoteNorth.maglon = Rlon[ind]
        self.secs2DcfRemoteNorth.angle2D = Rangle[ind]

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
        temp = np.max([np.max(trackALatN), np.max(trackCLatN)])
        self.secs2DcfSouth.create(
            np.append(temp + 1, trackALatS),
            np.append(trackALonS[0], trackALonS),
            np.append(temp + 1, trackCLatS),
            np.append(trackCLonS[0], trackCLonS),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D,
            self.extLon2D,
            self.poleLat,
            self.poleLon,
        )

        Rlat, Rlon, Rangle, _, _ = auto.sub_Swarm_grids(
            np.append(temp + 1, trackALatS),
            np.append(trackALonS[0], trackALonS),
            np.append(temp + 1, trackCLatS),
            np.append(trackCLonS[0], trackCLonS),
            self.dlatOut,
            self.lonRatioOut,
            self.extLat2D + self.cfremoteN,
            self.extLon2D + self.cfremoteN,
            self.poleLat,
            self.poleLon,
        )

        N1, N2 = self.secs1DcfSouth.shape
        mask = np.ones(Rlat.shape)
        mask[self.cfremoteN :, self.cfremoteN :][:N1, :N2] = 0
        ind = np.argwhere(mask)
        self.secs2DcfRemoteSouth.magLat = Rlat[ind]
        self.secs2DcfRemoteSouth.magLon = Rlon[ind]
        self.secs2DcfRemoteSouth.angle2D = Rangle[ind]

        # self.ggLat1D,_ = auto.sub_Swarm_grids_1D(lat1,lat2,)


def getLocalDipoleFPtrack1D(latB, rB, Ri):
    """Get the local dipole footpoints for the CF grid creation.

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


def getLocalDipoleFPtrack2D(latB, lonB, rB, Ri):
    thetaB = (90 - latB) / 180 * np.pi
    thetaFP = np.arcsin(
        np.sqrt(Ri / rB) * np.sin(thetaB)
    )  # co-latitude of the footpoints, mapped to the northern hemisphere
    ind = np.argwhere(thetaB > np.pi / 2)
    thetaFP[ind] = np.pi - thetaFP[ind]
    latFP = 90 - thetaFP / np.pi * 180
    lonFP = lonB
    sorted = np.argsort(np.abs(latFP))
    latFP = latFP[sorted]
    lonFP = lonFP[sorted]
    # probe which hemisphere is being processed
    if np.mean(latB) > 0:
        latFP = latFP[latFP > 0]
        lonFP = lonFP[latFP > 0]
    else:
        latFP = latFP[latFP < 0]
        lonFP = lonFP[lonFP < 0]

    return latFP, lonFP


def mag_transform_dsecs(SwA, SwC, pole_lat, pole_lon):
    """Rotate the data to magnetic coordinate systems.

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
            "magBp": ("Timestamp", amagbt),
        }
    )
    SwC = SwC.assign(
        {
            "magUvT": ("Timestamp", cuvt),
            "magUvP": ("Timestamp", cuvp),
            "magLat": ("Timestamp", cmaglat),
            "magLon": ("Timestamp", cmaglon),
            "magBt": ("Timestamp", cmagbt),
            "magBp": ("Timestamp", cmagbt),
        }
    )

    # SwC locations & data
    SwA["magLon"] = SwA["magLon"] % 360
    SwC["magLon"] = SwC["magLon"] % 360

    return SwA, SwC


def trim_data(SwA, SwC):
    """Find periods with suitable spaceraft velocity for analysis."""

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


class dsecsdata:
    """Class for DSECS variables and fitting procedures"""

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
        self.alpha1Ddf: float = 1e-5
        self.epsSVD1Ddf: float = 4e-4
        self.alpha1Dcf: float = 20e-5
        self.epsSVD1Dcf: float = 100e-4
        self.alpha2D: float = 50e-5
        self.epsSVD2D: float = 10e-4
        self.df1D: np.ndarray([])
        self.df2D: np.ndarray([])
        self.cf1D: np.ndarray([])
        self.cf2D: np.ndarray([])
        self.matBr1D: np.ndarray([])
        self.matBt1D: np.ndarray([])
        self.matBpara1D: np.ndarray([])
        self.matBpara2D: np.ndarray([])
        self.apexlats: np.ndarray([])
        self.cf1dDipMagJp = np.ndarray([])
        self.cf1dDipMagJt = np.ndarray([])
        self.cf1dDipJr = np.ndarray([])
        self.cf1dDipBr = np.ndarray([])
        self.cf1dDipMagBt = np.ndarray([])
        self.cf1dDipMagBp = np.ndarray([])
        self.cf1dDipBr = np.ndarray([])
        self.cf1dDipMagBt = np.ndarray([])
        self.cf1dDipMagBp = np.ndarray([])
        self.cf2dDipMagJp = np.ndarray([])
        self.cf2dDipMagJt = np.ndarray([])
        self.cf2dDipJr = np.ndarray([])
        self.cf2dDipBr = np.ndarray([])
        self.cf2dDipMagBt = np.ndarray([])
        self.cf2dDipMagBp = np.ndarray([])
        self.cf2dDipBr = np.ndarray([])
        self.cf2dDipMagBt = np.ndarray([])
        self.cf2dDipMagBp = np.ndarray([])
        self.matBp2Ddf = np.ndarray([])
        self.matBt2Ddf = np.ndarray([])
        self.matBr2Ddf = np.ndarray([])

    def populate(self, SwA, SwC):
        """Initialize a DSECS analaysis case from data"""

        # initialize grid
        grid = dsecsgrid()
        grid.FindPole(SwA)

        # calculate additional variables
        SwA, SwC = getUnitVectors(SwA, SwC)

        SwA, SwC = mag_transform_dsecs(SwA, SwC, grid.poleLat, grid.poleLon)
        # trim the data if needed

        SwA, SwC = trim_data(SwA, SwC)

        # create grid

        grid.create(SwA, SwC)

        self.latB = np.concatenate((SwA["magLat"], SwC["magLat"]))
        self.lonB = np.concatenate((SwA["magLon"], SwC["magLon"]))
        self.apexlats = np.concatenate((SwA["ApexLatitude"], SwC["ApexLatitude"]))
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
        self.cf1dDipMagJp = np.empty_like(outproto)
        self.cf1dDipMagJt = np.empty_like(outproto)
        self.cf1dDipJr = np.empty_like(outproto)
        self.cf1dDipBr = np.empty_like(dataproto)
        self.cf1dDipMagBt = np.empty_like(dataproto)
        self.cf1dDipMagBp = np.empty_like(dataproto)
        self.cf1dDipBr = np.empty_like(dataproto)
        self.cf1dDipMagBt = np.empty_like(dataproto)
        self.cf1dDipMagBp = np.empty_like(dataproto)
        self.cf2dDipMagJp = np.empty_like(outproto)
        self.cf2dDipMagJt = np.empty_like(outproto)
        self.cf2dDipJr = np.empty_like(outproto)
        self.cf2dDipBr = np.empty_like(dataproto)
        self.cf2dDipMagBt = np.empty_like(dataproto)
        self.cf2dDipMagBp = np.empty_like(dataproto)
        self.cf2dDipBr = np.empty_like(dataproto)
        self.cf2dDipMagBt = np.empty_like(dataproto)
        self.cf2dDipMagBp = np.empty_like(dataproto)
        self.test = dict()

    def fit1D_df(self):
        """1D divergence-free fit for data.

        Returns
        -------
        _type_
            _description_
        """

        self.matBr1D, self.matBt1D = SECS_1D_DivFree_magnetic(
            self.latB, self.grid.secs1Ddf.lat, self.rB, self.grid.Ri, 500
        )

        y = self.Bpara  # measurement, parallel magnetic field

        self.matBpara1D = (self.matBr1D.T * self.uvR).T + (
            self.matBt1D.T * self.uvT
        ).T  #

        regmat = self.grid.secs1Ddf.diff2  # regularization
        x = auto.sub_inversion(
            self.matBpara1D, regmat, self.epsSVD1Ddf, self.alpha1Ddf, y
        )
        self.df1D = x
        return x, y, self.matBpara1D

    def fit2D_df(self):
        """2D divergence-free fit for data."""
        # Calculate B-matrices and form the field-aligned matrix
        thetaB = (90 - self.latB) / 180 * np.pi
        phiB = self.lonB / 180 * np.pi
        theta2D = (90 - self.grid.secs2Ddf.magLat) / 180 * np.pi
        phi2D = self.grid.secs2Ddf.magLon / 180 * np.pi

        matBr2D, matBt2D, self.matBp2Ddf = SECS_2D_DivFree_magnetic(
            thetaB, phiB, theta2D, phi2D, self.rB, self.grid.Ri
        )
        # N2d = np.size(self.grid.secs2D.lat)

        self.matBpara2D = (
            (matBr2D.T * self.uvR).T
            + (matBt2D.T * self.uvT).T
            + (self.matBp2Ddf.T * self.uvP).T
        )

        # Remove field explained by the 1D DF SECS (must have been fitted earlier).
        Bpara2D = self.Bpara - self.matBpara1D @ self.df1D

        regmat = self.grid.secs2Ddf.diff2lat2D  # regularization
        self.df2D = auto.sub_inversion(
            self.matBpara2D, regmat, self.epsSVD2D, self.alpha2D, Bpara2D
        )

        # Calculate the current produced by the 2D DF SECS.
        thetaJ = (90 - self.grid.out.magLat.ravel()) / 180 * np.pi
        phiJ = self.grid.out.magLon.ravel() / 180 * np.pi
        matJt, matJp = SECS_2D_DivFree_vector(
            thetaJ, phiJ, theta2D, phi2D, self.grid.Ri, self.grid.secs2Ddf.angle2D
        )
        # df2dMagJt = np.reshape(matJt @ self.df2D, self.grid.out.magLat.shape)
        # df2dMagJp = np.reshape(matJp @ self.df2D, self.grid.out.magLat.shape)

        # Calculate the magnetic field produced by the 2D DF SECS
        df2dBr = matBr2D @ self.df2D
        # df2dMagBt = matBt2D @ self.df2D
        # df2dMagBp = self.matBp2Ddf @ self.df2D
        # df2dBpara = self.matBpara2D @ self.df2D

        return self.df2D, Bpara2D, self.matBpara2D, df2dBr

    def fit1d_cf(self):
        """1D curl-free fit for data."""
        # split data into hemispheres
        # ind = np.nonzero(.............)
        indN = np.argwhere(self.apexlats > 0)
        indRN = np.argwhere(self.grid.out.magLat > 0)
        indS = np.argwhere(self.apexlats <= 0)
        indRS = np.argwhere(self.grid.out.magLat <= 0)

        for ind, rind, gridhem in zip(
            [indN, indS],
            [indRN, indRS],
            [self.grid.secs1DcfNorth, self.grid.secs1DcfSouth],
            strict=True,
        ):
            latB_hem = self.latB[ind]
            rB_hem = self.rB[ind]
            Bp = self.magBp - self.matBp2Ddf @ self.df2D
            Bphem = Bp[ind].ravel()
            #        #Remove effect of 1D & 2D DF currents (must have been fitted earlier)
            # a=SwA.magBp(indA)-SwA.df1dMagBp(indA)-SwA.df2dMagBp(indA);
            # SwA.df1dMagBp = 0 by definition
            Bp = self.magBp - self.matBp2Ddf @ self.df2D
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
            #        #Calculate the theta-current (southward) produced by the 1D CF SECS.
            latJ = self.grid.out.magLat.ravel()
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
            self.cf1dDipMagBp[ind] = (matBp @ cf1D).reshape((-1, 1))
        return cf1D, Bp, matBp, Bphem


def fit2d_cf(self):
    """2D curl-free fit for data."""
    # Fit is done to all components of the magnetic disturbance.
    # Remove field explained by the 1D & 2D DF SECS and 1D CF SECS (dipolar)
    Br = self.Br - self.df1dBr - self.df2dBr - self.cf1dDipBr
    Bt = self.magBt - self.df1dMagBt - self.df2dMagBt - self.cf1dDipMagBt
    # df1dMagBp = 0 by definition
    Bp = self.magBp - self.df2dMagBp - self.cf1dDipMagBp

    thetaB = (90 - self.latB) / 180 * np.pi
    phiB = self.lonB / 180 * np.pi

    for ind, rind, gridhem, remote_hem in zip(
        [self.indN, self.indS],
        [self.indRN, self.indRS],
        [self.grid.secs2DcfNorth, self.grid.secs2DcfSouth],
        [self.grid.secs2dcfRemoteNorth, self.grid.secs2dcfRemoteSouth],
    ):
     
        Br_hem = Br[ind]
        Bt_hem = Bt[ind]
        Bp_hem = Bp[ind]
        thetaB_hem = thetaB[ind]
        phiB_hem = phiB[ind]
        rB_hem = self.rB[ind]

        # Calculate B-matrices.
        theta2D = (90 - gridhem.lat) / 180 * np.pi
        phi2D = gridhem.lon / 180 * np.pi
   
        matBr2D, matBt2D, matBp2D = SECS_2D_CurlFree_antisym_magnetic(
            thetaB_hem, phiB_hem, theta2D, phi2D, rB_hem, self.grid.Ri, gridhem.angle2D
        )

        remoteTheta = (90 - remote_hem.lat) / 180 * np.pi
        remotePhi = remote_hem.lon / 180 * np.pi
        remoteMatBr2D, remoteMatBt2D, remoteMatBp2D = SECS_2D_CurlFree_antisym_magnetic(
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
            [],
            remoteEps,
            np.nan,
            np.vstack((Br_hem, Bt_hem, Bp_hem)),
        )
        # Remove effect from the measured B
        Br_hem = Br_hem - remoteMatBr2D @ remoteIcf
        Bt_hem = Bt_hem - remoteMatBt2D @ remoteIcf
        Bp_hem = Bp_hem - remoteMatBp2D @ remoteIcf

        # Fit the (local) 2D CF SECS. Use zero constraint on the 2nd latitudinal derivative.
        regmat = gridhem.diff2lat2D  # regularization
        Icf = auto.sub_inversion(
            np.vstack((matBr2D, matBt2D, matBp2D)),
            regmat,
            self.epsSVD2Dcf,
            self.alpha2Dcf,
            np.vstack((Br_hem, Bt_hem, Bp_hem)),
        )

        # Calculate the horizontal current produced by the 2D CF SECS.
        thetaJ = (90 - self.grid.out.magLat.ravel()) / 180 * np.pi
        phiJ = self.grid.out.magLon.ravel() / 180 * np.pi
        matJt, matJp = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiJ, theta2D, phi2D, self.grid.Ri, gridhem.angle2D
        )
        tmp = np.reshape(matJt @ Icf, self.grid.out.magLat.shape)
        self.cf2dDipMagJt[rind] = tmp[rind]
        tmp = np.reshape(matJp @ Icf, self.grid.out.magLat.shape)
        self.cf2dDipMagJp[rind] = tmp[rind]

        # Calculate the radial current produced by the 2D CF SECS using finite differences.
        # For FAC we should scale by 1/sin(inclination).
        thetaNorth = thetaJ - self.dlatOut / 180 * np.pi
        thetaSouth = thetaJ + self.dlatOut / 180 * np.pi
        matJtNorth, _ = SECS_2D_CurlFree_antisym_vector(
            thetaNorth, phiJ, theta2D, phi2D, self.Ri, gridhem.angle2D
        )
        matJtSouth, _ = SECS_2D_CurlFree_antisym_vector(
            thetaSouth, phiJ, theta2D, phi2D, self.Ri, gridhem.angle2D
        )

        phiEast = phiJ + self.dlatOut / 180 * np.pi
        phiWest = phiJ - self.dlatOut / 180 * np.pi
        _, matJpEast = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiEast, theta2D, phi2D, self.Ri, gridhem.angle2D
        )
        _, matJpWest = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiWest, theta2D, phi2D, self.Ri, gridhem.angle2D
        )
        aux1 = (
            np.sin(thetaSouth) * (matJtSouth @ Icf)
            - np.sin(thetaNorth) * (matJtNorth @ Icf)
        ) / (thetaSouth - thetaNorth)
        aux2 = ((matJpEast - matJpWest) @ Icf) / (phiEast - phiWest)
        tmp = -(aux1 + aux2) / (self.Ri * np.sin(thetaJ))  # radial current = -div
        tmp = np.reshape(tmp * 1000, self.grid.out.magLat.shape)  # A/km^2 --> nA/m^2
        self.cf2dDipJr[rind] = tmp[rind]

        # Calculate the magnetic field produced by the 2D CF SECS.
        # cf2dDipBr = matBr2D @ Icf
        # cf2dDipMagBt = matBt2D @ Icf
        # cf2dDipMagBp = matBp2D @ Icf

        # Current produced by the remote SECS
        remoteAngle = remote_hem.angle2D
        matJt, matJp = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiJ, remoteTheta, remotePhi, self.Ri, remoteAngle
        )
        tmp = np.reshape(matJt @ remoteIcf, self.grid.out.magLat.shape)
        # remoteCf2dDipMagJt[rind] = tmp[rind]
        tmp = np.reshape(matJp @ remoteIcf, self.grid.out.magLat.shape)
        self.remoteCf2dDipMagJp[rind] = tmp[rind]
        matJtNorth, _ = SECS_2D_CurlFree_antisym_vector(
            thetaNorth, phiJ, remoteTheta, remotePhi, self.Ri, remoteAngle
        )
        matJtSouth, _ = SECS_2D_CurlFree_antisym_vector(
            thetaSouth, phiJ, remoteTheta, remotePhi, self.Ri, remoteAngle
        )
        _, matJpEast = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiEast, remoteTheta, remotePhi, self.Ri, remoteAngle
        )
        _, matJpWest = SECS_2D_CurlFree_antisym_vector(
            thetaJ, phiWest, remoteTheta, remotePhi, self.Ri, remoteAngle
        )
        aux1 = (
            np.sin(thetaSouth) * (matJtSouth @ remoteIcf)
            - np.sin(thetaNorth) * (matJtNorth @ remoteIcf)
        ) / (thetaSouth - thetaNorth)
        aux2 = ((matJpEast - matJpWest) @ remoteIcf) / (phiEast - phiWest)
        tmp = -(aux1 + aux2) / (self.Ri * np.sin(thetaJ))
        tmp = np.reshape(tmp * 1000, self.grid.out.magLat.shape)
        self.remoteCf2dDipJr[rind] = tmp[rind]

        # Then their magnetic field
        self.remoteCf2dDipBr = remoteMatBr2D @ remoteIcf
        self.remoteCf2dDipMagBt = remoteMatBt2D @ remoteIcf
        self.remoteCf2dDipMagBp = remoteMatBp2D @ remoteIcf

    return
