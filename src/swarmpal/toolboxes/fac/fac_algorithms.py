"""Tools to evaluate FACs using the single-satellite method

Given input containing magnetic field measurements and model predictions
(from swarmpal.io.MagData), compute the FACs

Adapted from code by Ask Neve Gamby (https://github.com/Swarm-DISC/SwarmPyFAC)

"""


from __future__ import annotations

from numpy import (
    abs,
    apply_along_axis,
    arctan2,
    cos,
    deg2rad,
    diff,
    nan,
    pi,
    sin,
    stack,
    tile,
)
from numpy.linalg import norm
from scipy.interpolate import splev, splrep

from swarmpal.io import ExternalData

MU_0 = 4.0 * pi * 10 ** (-7)


def _means(x):
    return (x[1:] + x[:-1]) * 0.5


def _spherical_delta(p):
    """Compute the change of spherical positions as an NEC vector"""
    r_means = _means(p[:, 2])
    return stack(
        [
            r_means * sin(deg2rad(diff(p[:, 0]))),
            r_means * sin(deg2rad(diff(p[:, 1]))) * cos(deg2rad(_means(p[:, 0]))),
            -diff(p[:, 2]),
        ],
        axis=1,
    )


def _NEC_to_VSC(v):
    """Construct a function which transforms vectors from NEC to VSC frame"""
    angles = -arctan2(v[:, 0] - v[:, 1], v[:, 0] + v[:, 1])
    sines = sin(angles)
    cosines = cos(angles)

    def transform(v):
        return stack(
            [
                cosines * v[:, 0] + sines * v[:, 1],
                -sines * v[:, 0] + cosines * v[:, 1],
                v[:, 2],
            ],
            axis=1,
        )

    return transform


def _radial_current(dB, v, dt):
    """Calculate radial currents from magnetic residuals and spacecraft velocity"""
    # Compute curl
    change = dB[:, 1] / v[:, 0] - dB[:, 0] / v[:, 1]
    return (-0.001 / (2 * MU_0)) * change / dt


def _interpolate_data(t_source, t_target):
    """Interpolate data from original sampling on t_source to new sampling on t_target"""

    def _func(d):
        B_spline_rep = splrep(t_source, d)
        return splev(t_target, B_spline_rep)

    return _func


def _inclination(v):
    return arctan2(v[:, 2], norm(v[:, :2], axis=1))


def fac_single_sat_algo(
    time=None, positions=None, B_res=None, B_model=None, inclination_limit=30.0
):
    """Compute field-aligned current (FAC) from numpy arrays

    Parameters
    ----------
    time : array_like
        Array of datetime64[ns]
    positions : array_like
        Nx3 array of positions (Latitude, Longitude, Radius) in units of (degrees, degrees, metres)
    B_res : array_like
        Nx3 array of magnetic field residuals in nanoTesla
    B_model : array_like
        Nx3 array of magnetic field model predictions in nanoTesla
    inclination_limit : float, optional

    Returns
    -------
    dict
        {"time": array_like, "fac": array_like}
        Times and FAC estimates at those times
    """
    # Convert time (datetime64[ns]) to seconds
    time_seconds = time.astype(float) / 1e9
    # Array of positions accounting for local time
    pos_ltl = positions.copy()
    pos_ltl[:, 1] = (pos_ltl[:, 1] + 180 + time_seconds / 86400 * 360) % 360 - 180
    # Evaluate velocity in NEC frame
    dt = diff(time_seconds)
    v = _spherical_delta(pos_ltl) / tile(dt, (3, 1)).T
    # Transform to VSC frame
    NEC2VSC = _NEC_to_VSC(v)
    v_VSC = NEC2VSC(v)
    B1 = NEC2VSC(B_res[:-1])
    B2 = NEC2VSC(B_res[1:])
    # Compute radial current
    irc = _radial_current(B2 - B1, v_VSC, dt)
    # Interpolate magnetic model predictions (B_model) onto time midpoints
    target_time = _means(time_seconds)
    interpolator = _interpolate_data(time_seconds, target_time)
    B_interpolated = apply_along_axis(interpolator, 0, B_model)
    # Convert radial current according to the inclination of B_model
    inclination = _inclination(B_interpolated)
    fac = -irc / sin(inclination)
    # Screen out positions where the inclination is too extreme
    #  or where dt > 1 second
    reject = abs(inclination) < deg2rad(inclination_limit)
    reject &= dt > 1
    fac[reject] = nan
    # Convert new time back to datetime64
    target_time = (target_time * 1e9).astype("datetime64[ns]")
    return {"time": target_time, "fac": fac}


def fac_single_sat(fac_inputs, **kwargs):
    """Compute field-aligned current (FAC) from FacInputs object

    Parameters
    ----------
    fac_inputs : FacInputs

    Returns
    -------
    dict
        {"time": array_like, "fac": array_like}
        Times and FAC estimates at those times

    Examples
    --------
    >>> from swarmpal.toolboxes.fac import FacInputs, fac_single_sat
    >>> fac_inputs = FacInputs(
    >>>     collection="SW_OPER_MAGA_LR_1B", model="IGRF"
    >>>     start_time="2022-01-01", end_time="2022-01-02"
    >>> )
    >>> output = fac_single_sat(fac_inputs)
    """
    # Extract necessary data from the input
    time = fac_inputs.get_array("Timestamp").astype("datetime64[ns]")
    # Nx3 Array of positions(theta, phi, r)
    positions = stack(
        [
            fac_inputs.get_array("Latitude"),
            fac_inputs.get_array("Longitude"),
            fac_inputs.get_array("Radius"),
        ],
        axis=1,
    )
    # Magnetic field vector residual and model values
    B_res = fac_inputs.get_array("B_NEC") - fac_inputs.get_array("B_NEC_Model")
    B_model = fac_inputs.get_array("B_NEC_Model")

    fac_results = fac_single_sat_algo(
        time=time, positions=positions, B_res=B_res, B_model=B_model, **kwargs
    )
    return fac_results


class FacInputs(ExternalData):
    """Accessing external data: inputs to FAC algorithm

    Examples
    --------
    >>> d = FacInputs(
    >>>     collection="SW_OPER_MAGA_LR_1B", model="IGRF",
    >>>     start_time="2022-01-01", end_time="2022-01-02",
    >>>     viresclient_kwargs=dict(asynchronous=True, show_progress=True)
    >>> )
    >>> d.xarray  # Returns xarray of data
    >>> d.get_array("B_NEC")  # Returns numpy array
    """

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["F", "B_NEC", "Flags_B"],
        "model": "IGRF",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }


if __name__ == "__main__":
    fac_inputs = FacInputs(
        collection="SW_OPER_MAGA_LR_1B",
        model="IGRF",
        start_time="2022-01-01",
        end_time="2022-01-02",
        viresclient_kwargs=dict(asynchronous=True, show_progress=True),
    )
    output = fac_single_sat(fac_inputs)
