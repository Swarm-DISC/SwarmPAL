"""TODO Docstring for TFA
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt

from swarmx.io import ExternalData

TFA_DEFAULTS = {
    "F_CUTOFF": 3 / 1000,
    "FREQS": np.arange(1, 50) / 1000,
    "OMEGA": 6,
    "LEVELS": np.arange(-4, 4, 0.5),
}


class TfaMagInputs(ExternalData):
    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
        *[f"SW_OPER_MAG{x}_HR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["F"],
        "model": "'CHAOS-Core' + 'CHAOS-Static'",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }


def pre_process(inputs, f_cutoff=TFA_DEFAULTS["F_CUTOFF"]):
    """Filtering"""
    # Extract inputs we need
    X = inputs.get_array("F")
    X_model = inputs.get_array("F_Model")
    # Create and apply filter
    b, a = butter(5, f_cutoff, btype="highpass", analog=False, output="ba", fs=1)
    filtered = filtfilt(b, a, X - np.mean(X))
    filtered1 = filtfilt(b, a, X - X_model)
    return {"filtered": filtered, "filtered1": filtered1}


def wavelet_power(x, dt=1, freqs=TFA_DEFAULTS["FREQS"], omega=TFA_DEFAULTS["OMEGA"]):
    """Calculate wavelet power"""
    N = len(x)
    Fx = fft(x)
    wk_pos = np.arange(0, np.floor(N / 2)) * (2 * np.pi) / (N * dt)
    L = len(wk_pos)
    fourier_factor = 4 * np.pi / (omega + np.sqrt(2 + omega**2))
    N_freqs = len(freqs)
    W = np.zeros((N_freqs, N))
    for i in range(N_freqs):
        s = 1 / (fourier_factor * freqs[i])
        psi = np.pi ** (-1 / 4) * np.exp(-((s * wk_pos - omega) ** 2) / 2)
        psi = psi * np.sqrt(2 * np.pi * s / dt)
        full_psi = np.hstack((psi, np.zeros(N - L)))
        W[i, :] = np.abs(ifft(Fx * full_psi)) ** 2
    return W


def plot_filtered(inputs, outputs):
    """Plot the filtered input data"""
    t = inputs.get_array("Timestamp")
    F = inputs.get_array("F")
    mlat = inputs.get_array("QDLat")
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].plot(t, F, "-")
    axes[0].set_ylabel("|B| (nT)")
    axes[1].plot(t, outputs["filtered1"], "-r")
    axes[1].set_ylabel("|B| filtered (nT)")
    axes[2].plot(t, mlat, "-r")
    axes[2].set_ylabel("MLAT (deg)")
    for ax in axes:
        ax.grid()
    return fig


def plot_wavelet_power(
    inputs, power, freqs=TFA_DEFAULTS["FREQS"], levels=TFA_DEFAULTS["LEVELS"]
):
    """Plot the wavelet power"""
    fig, ax = plt.subplots(1, 1)
    ax.contourf(
        inputs.get_array("Timestamp"),
        freqs * 1000,
        np.log10(power),
        levels=levels,
        cmap=plt.cm.jet,
        extend="both",
    )
    # add colorbar axes first, then:
    # plt.colorbar(im, ticks=range(-10,10,1), label=r"$log_{10}(Wavelet Power)$")
    ax.set_ylabel("Frequency (mHz)")
    return fig


if __name__ == "__main__":
    inputs = TfaMagInputs(
        collection="SW_OPER_MAGA_LR_1B",
        model="'CHAOS-Core' + 'CHAOS-Static'",
        start_time=dt.datetime(2015, 6, 23, 0, 0, 0),
        end_time=dt.datetime(2015, 6, 23, 5, 0, 0),
    )
    inputs_filtered = pre_process(inputs)
    plot_filtered(inputs, inputs_filtered)
    W1 = wavelet_power(inputs_filtered["filtered1"])
    plot_wavelet_power(inputs, W1)
