"""TFA library

Authors:
constantinos@noa.gr
"""

import sys

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import scipy.signal as signal
from scipy.fft import fft, ifft

R_E = 6371.2  # reference Earth radius in km


def constant_cadence(t_obs, x_obs, sampling_rate, interp=False):
    """
    Set data points to a new time-series with constant sampling time.

    Even though data are supposed to be provided at constant time steps,
    e.g. as times t = 1, 2, 3, ... etc, they are often not, as they can
    be given instead at times t = 1.01, 2.03, 2.99, etc or data points
    along with their time stampts might be missing entirely, or even be
    duplicated one or more times. This function corrects these errors,
    by producing a new array with timestamps at exactly the requested
    cadence and either moves the existing values to their new proper
    place, or interpolates the new values at the new time stamps based
    on the old ones, depending on the value of the `interp` parameter.

    `t_obs` is a one-dimensional array with the  time in seconds
    `x_obs` is a one or two-dimensioanl array with real values
    `sampling_rate` is a real number, given in Hz

    Note: Gaps in the data will NOT be filled, even when `interp` is True.


    Parameters
    ----------
    t_obs : (N,) array_like
        A 1-D array of real values.
    x_obs : (...,N,...) array_like
        A N-D array of real values. The length of `x_obs` along the first
        axis must be equal to the length of `t_obs`.
    sampling_rate: float
        The sampling rate of the output series in Hz (eg 2 means two
        measurements per second, i.e. a time step of 0.5 sec between each).
    interp: bool, optional
        If False the function will move the existing data values to their
        new time stamps. If True, it will interpolate the values at the
        new time stamps based on the original values, using a linear
        interpolation scheme.


    Returns
    ----------
    t_rec : (N,) array_like
        A 1-D array of the new time values, set at constant cadence.
    x_rec : (...,N,...) array_like
        A N-D array of real values, with the values of `x_obs` set at constant
        cadence.
    nn_mask: (...,N,...) array_like, bool
        A N-D array of boolean values


    Examples
    --------
    >>> import numpy as np
    >>> import tfalib
    >>> import matplotlib.pyplot as plt
    >>> N = 10 # No Points to generate
    >>> FS = 50 # sampling rate in Hz
    >>> # create time vector
    >>> t = np.arange(N) * (1/FS)
    >>> # add some small deviations in time, just to complicate things
    >>> n = 0.1 * np.random.randn(N) * (1/FS)
    >>> t = 12.81 + t + n
    >>> # produce x values from a simple linear relation
    >>> x = 10 + 0.01 * t
    >>> # remove some values at random
    >>> inds_to_rem = np.random.permutation(np.arange(1,N-1))[:int(N/4)]
    >>> t_obs = np.delete(t, inds_to_rem)
    >>> x_obs = np.delete(x, inds_to_rem)
    >>> (t_rec, x_rec, nn) = tfalib.constant_cadence(t_obs, x_obs, FS, False)
    >>> (t_int, x_int, nn) = tfalib.constant_cadence(t_obs, x_obs, FS, True)
    >>> plt.figure(3)
    >>> plt.plot(t_obs, x_obs, '--xk', t_rec, x_rec, 'or', t_int, x_int, '+b')
    >>> plt.legend(('Original Points', 'Moved', 'Interpolated'))
    >>> plt.grid(True)
    >>> plt.show()
    """

    if len(t_obs.shape) > 1:
        sys.exit(
            "constant_cadence: ERROR: `t_obs` argument must be 1-dimensional array"
        )

    if len(x_obs.shape) > 2:
        sys.exit(
            "constant_cadence: ERROR: `x_obs` argument must be 1 or 2-dimensional array"
        )

    N = len(t_obs)
    if len(x_obs.shape) == 2:
        multiDim = True
        transp = False

        if x_obs.shape[0] != N and x_obs.shape[1] != N:
            sys.exit(
                "constant_cadence: ERROR: `x_obs` must have the same length as `t_obs`"
            )
        elif x_obs.shape[0] != N and x_obs.shape[1] == N:
            x_obs = x_obs.T
            transp = True

        M = x_obs.shape[1]  # number of variables

    elif len(x_obs.shape) == 1:
        multiDim = False

    dt = 1 / sampling_rate
    time_range = np.max(t_obs) - np.min(t_obs)
    time_rec_N = np.ceil(time_range / dt)
    # init_t = np.round(t_obs[0]/dt) * dt
    init_t = t_obs[0]
    inds = np.abs(np.round((t_obs - init_t) / dt)).astype(int)

    t_rec = np.arange(init_t, init_t + time_rec_N * dt + 1e-6, dt)

    if multiDim:
        x_rec = np.full((N, M), np.NaN)
        x_rec[inds, :] = x_obs
    else:
        x_rec = np.full(t_rec.shape, np.NaN)
        x_rec[inds] = x_obs

    nonnan_mask = ~np.isnan(x_rec)

    if interp:
        # scipy interpolation that also extrapolates a bit if is needed
        # for the final point (default)
        f = interpolate.interp1d(t_obs, x_obs, kind="linear", fill_value="extrapolate")
        x_int = f(t_rec)

        # numpy interpolation, without extrapolation of final value
        # x_int = np.interp(t_rec, t_obs, x_obs)

        x_int[~nonnan_mask] = np.NaN

        if multiDim and transp:
            x_int = x_int.T
            nonnan_mask = nonnan_mask.T

        return t_rec, x_int, nonnan_mask
    else:
        if multiDim and transp:
            x_rec = x_rec.T
            nonnan_mask = nonnan_mask.T

        return t_rec, x_rec, nonnan_mask


def moving_mean_and_stdev(x, window_size, unbiased_std=True):
    """
    Calculate moving average and moving st.dev

    Parameters
    ----------
    x: (...,N,...) array_like
        A 1-D or 2-D array of real values. If 2-D then each column is being
        treated separately.
    window_size: int
        The size of the rolling window (in number of points)
    unbiased_std: bool, optional
        If True the unbiased estimator of the standard deviation will be used,
        i.e dividing by N-1. If False the standard deviation will be computed
        by dividing by N.


    Returns
    ----------
    moving_mean: (...,N,...) array_like
        Moving mean, the same size as `x`.
    moving_stdev: (...,N,...) array_like
        Moving standard deviation, the same size as `x`.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import tfalib
    >>> N = 1000
    >>> t = np.linspace(0, 2*np.pi, N)
    >>> x = np.sin(2*np.pi*t/np.pi) + 0.1*np.random.randn(N)
    >>> m, s = tfalib.moving_mean_and_stdev(x, 50)
    >>> plt.plot(t, x, 'xk', t, m, '-b', t, m + 3*s, '-r', t, m - 3*s, '-r')
    >>> plt.show()
    """

    original_shape = x.shape

    if len(original_shape) > 2:
        sys.exit(
            "moving_mean_and_stdev: ERROR: `x` argument must be 1- or 2-dimensional array"
        )

    if len(original_shape) == 1:
        x = np.reshape(x, (-1, 1))  # turn it to a single column

    N, M = x.shape

    if window_size >= N:
        sys.exit(
            "moving_mean_and_stdev: ERROR: `window_size` cannot be equal or larger than the length of the data series in `x`"
        )

    # initialize outputs
    moving_mean = np.full(x.shape, np.NaN)
    moving_stdev = np.full(x.shape, np.NaN)

    for i in range(M):
        # convolve() works with 1-D series so use the appropriate dimensionality
        x1 = np.reshape(x[:, i], (N,))

        # count non-NaNs
        nonNaNs = ~np.isnan(x1)
        moving_N = np.convolve(nonNaNs, np.ones(window_size), "same")

        # remove NaNs (set to zero)
        x1[~nonNaNs] = 0

        # calculate moving mean
        m = np.convolve(x1, np.ones(window_size), "same") / moving_N

        # calculate moving mean of sum of squares
        s = np.convolve(x1**2, np.ones(window_size), "same") / moving_N

        stdev = np.sqrt(s - m**2)

        # use the unbiased 1/(N-1) factor instead of 1/N for the st.dev.
        if unbiased_std:
            stdev *= np.sqrt(moving_N / (moving_N - 1))

        # replace NaNs that were removed previously
        x1[~nonNaNs] = np.NaN

        moving_mean[:, i] = m
        moving_stdev[:, i] = stdev

    if len(original_shape) == 1:
        moving_mean = np.reshape(moving_mean, (-1,))
        moving_stdev = np.reshape(moving_stdev, (-1,))

    return moving_mean, moving_stdev


def moving_q25_and_q75(x, window_size):
    """
    Calculate moving 25th and 75th percentiles

    The difference between these two percentiles is called the inter-quartile
    range (iqr) and can be used for outlier detection, i.e. accept only points
    that lie within the region from q25 - 1.5*iqr up to q75 + 1.5*iqr and
    discard the rest.

    NOTE: It is recommended to use an odd integer number for window_size

    Parameters
    ----------
    x: (...,N,...) array_like
        A 1-D or 2-D array of real values. If 2-D then each column is being
        treated separately.
    window_size: int
        The size of the rolling window (in number of points)


    Returns
    ----------
    moving_q25: (...,N,...) array_like
        Moving 25th percentile, the same size as `x`.
    moving_q75: (...,N,...) array_like
        Moving 75th percentile, the same size as `x`.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import tfalib
    >>> N = 1000
    >>> t = np.linspace(0, 2*np.pi, N)
    >>> x = np.sin(2*np.pi*t/np.pi) + 0.1*np.random.randn(N)
    >>> q25, q75 = tfalib.moving_q25_and_q75(x, 50)
    >>> iqr = q75 - q25
    >>> plt.plot(t, x, 'xk', t, q25 - 1.5*iqr, '-r', t, q75 + 1.5*iqr, '-r')
    >>> plt.show()
    """

    original_shape = x.shape

    if len(original_shape) > 2:
        sys.exit(
            "moving_mean_and_stdev: ERROR: `x` argument must be 1- or 2-dimensional array"
        )

    N = original_shape[0]

    if window_size >= N:
        sys.exit(
            "moving_mean_and_stdev: ERROR: `window_size` cannot be equal or larger than the length of the data series in `x`"
        )

    # use Pandas rolling quartile functionality
    D = pd.DataFrame(x)
    moving_window = D.rolling(
        window_size, min_periods=int(window_size / 2)
    )  # accept windows of minimum W/2 valid points
    moving_q25 = moving_window.quantile(0.25, interpolation="linear").to_numpy()
    moving_q75 = moving_window.quantile(0.75, interpolation="linear").to_numpy()

    moving_q25 = np.roll(moving_q25, -int(window_size / 2), axis=0)
    moving_q75 = np.roll(moving_q75, -int(window_size / 2), axis=0)

    if len(original_shape) == 1:
        moving_q25 = np.reshape(moving_q25, (-1,))
        moving_q75 = np.reshape(moving_q75, (-1,))

    return moving_q25, moving_q75


def outliers(x, window_size, method="iqr", multiplier=np.NaN):
    """
    Find statistical outliers in data

    This uses a moving window to identify outliers, based on how larger or
    smaller data points are from their neighbours within the window. Two
    methods are used:

    `normal`: Assumes Gaussian distribution. Calculates the meand and st.dev.
    inside a window of length `window_size` and flags as outliers points that
    lie below/above the window mean +/- M times that st.dev, with M being
    defined by the `multiplier` parameter.

    `iqr`: As above, but using the quartiles q25 and q75 and the inter-quartile
    range iqr, to define the zone of acceptable measurements. Outliers will lie
    below q25 - M*iqr or above q75 + M*iqr, with M being the `multiplier`
    parameter.

    `multiplier` can be either a single float or a list of two numbers, in
    which case, the first will be used to define the lower limit and the second
    the upper one. If you want to search only for e.g. high outliers, then set
    the first element of `multiplier` as numpy.Inf so that it will include all
    values.



    Parameters
    ----------
    x: (...,N,...) array_like
        A 1-D or 2-D array of real values. If 2-D then each column is being
        treated separately.
    window_size: int
        The size of the rolling window (in number of points)
    method: string
        Can be either 'normal' or 'iqr' and signifies the method used
    multiplier: float or list (of two floats)
        The number that indicates the spread of the zone of accepted values


    Returns
    ----------
    outlier_inds: (...,N,...) array_like
        Boolean array, the same size as `x` with True where an outlier has been
        detected and False elsewhere.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import tfalib
    >>> N = 1000
    >>> t = np.linspace(0, 2*np.pi, N)
    >>> x = np.sin(2*np.pi*t/np.pi) + np.random.randn(N)
    >>> M = 100 # number of outliers
    >>> A = 5   # intensity of outliers
    >>> spkinds = np.random.permutation(N)[0:M]
    >>> x[spkinds[0:int(np.floor(M/2))]] += A # first half to be increased
    >>> x[spkinds[int(np.ceil(M/2)):]] -= A # second half to be decreased
    >>> outlier_inds = tfalib.outliers(x, 25, method = 'iqr', multiplier = 1.5)
    >>> plt.plot(t, x, 'xk', t[outlier_inds], x[outlier_inds], 'or')
    >>> plt.show()
    """

    if not (type(multiplier) == list or type(multiplier) == np.ndarray):
        multiplier = [multiplier, multiplier]

    inds = np.full(x.shape, False)

    if method == "normal":
        if np.all(np.isnan(multiplier)):
            multiplier = [3, 3]

        m, s = moving_mean_and_stdev(x, window_size)
        inds[x < m - multiplier[0] * s] = True
        inds[x > m + multiplier[1] * s] = True

    elif method == "iqr":
        if np.all(np.isnan(multiplier)):
            multiplier = [1.5, 1.5]

        q25, q75 = moving_q25_and_q75(x, window_size)
        iqr = q75 - q25
        inds[x < q25 - multiplier[0] * iqr] = True
        inds[x > q75 + multiplier[1] * iqr] = True

    else:
        sys.exit(
            "outliers: ERROR: `method` not recognized! Choose between 'normal' or 'iqr'"
        )

    return inds


def filter(x, sampling_rate, cutoff):
    """
    High-pass filter the data

    This is just a wrapper of the Chebysev Type II filter of SciPy. The way it
    works is that the lowpass filtered version of the series is being produced,
    by means of cheby2() and then it is subtracted from the data series, so
    that the high-pass component remains.


    Parameters
    ----------
    x: (...,N,...) array_like
        A 1-D or 2-D array of real values. If 2-D then each column is being
        treated separately.
    sampling_rate: float
        The sampling rate of the data, i.e. the reciprocal of the time step
    cutoff: float
        The cutoff frequency that the filter will use. Sinusoidal waveforms
        with frequencies below this cutoff will be reduced in amplitude
        (ideally to zero, but frequencies close to the cutoff will be less
        affected), while those with frequencies above this cutoff will remain
        unchanged.


    Returns
    ----------
    filtered: (...,N,...) array_like
        Array, the same size as `x` with the result of the filtering process.


    Examples
    --------
    >>> import numpy as np
    >>> import tfalib
    >>> import matplotlib.pyplot as plt
    >>> T = np.arange(0, 3600, 0.5)
    >>> Y = np.sin(2*np.pi*T/500)*(np.exp(-(T/1000)**2)) + np.sin(2*np.pi*T/250)*(np.exp(-((T-np.max(T))/1000)**2))
    >>> F = tfalib.filter(Y, 2, 3/1000)
    >>> plt.figure(1)
    >>> plt.plot(T, Y, color=[.5,.5,.5], linewidth=5)
    >>> plt.plot(T, F, '-r')
    >>> plt.legend(('Original Series', 'High-Pass Filtered'))
    >>> plt.grid(True)
    >>> plt.show()
    """

    # this is just a wrapper of Scipy's filtering functionality
    sos = signal.cheby2(
        7, 10, cutoff, btype="lowpass", analog=False, output="sos", fs=sampling_rate
    )
    F = x - signal.sosfiltfilt(sos, x, axis=0)

    return F


def morlet_wave(N=600, scale=1, dx=0.01, omega=6.203607835633639, roll=True, norm=True):
    """
    Generate a morlet wave-function to be used with the wavelet_tranform()

    This generates the comlex-conjugate, scaled and time-reversed form of the
    Morlet wavelet, so that it can be immediately used in the wavelet transform
    function.


    Parameters
    ----------
    N: integer
        Number of points to generate
    scale: float
        The scale, i.e. period of the generated waveform
    dx: float
        The time step of the data, i.e. the reciprocal of the sampling rate
    omega: float
        The omega_zero parameter of the Morlet function. The default value is
        6.2036 which is the value for which the wavelet scales directly
        correspond to the Fourier periods
    roll: boolean
        If `False`, the signal is generated as is, centered at zero. If `True`,
        it is translated so zero becomes the first element in the time series
        and the part of the wavelet that corresponds to negative x values is
        folded back at the end of the series. Use `False` to plot and see the
        wavelet, but `True` to use it with the wavelet transform!
     norm: boolean
         If `True` the function is normalized by multiplication with the factor
         sqrt(dx/scale), so that its sum of squares is 1 and sum of squares of
         FFT coefficients is N. Use `True` with the wavelet transform!


    Returns
    ----------
    wavelet: (N,) array_like
        1-D array with the complex values of the Morlet wavelet
    x: (N,) array_like
        1-D array with the `x` values that correspond to the wavelet (use for
        plotting only, otherwise ignore)
    wavelet_norm_factor: float
        A number to be used for the normalization of the result of the wavelet
        transform.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import fft
    >>> import tfalib
    >>> import matplotlib.pyplot as plt
    >>> N_wave = 600
    >>> s_wave = 50
    >>> dx_wave = .5
    >>> m, m_x, m_norm = tfalib.morlet_wave(N_wave, s_wave, dx_wave, roll=False, norm=True)
    >>> plt.figure()
    >>> plt.plot(m_x, np.real(m), '-b', m_x, np.imag(m), '-r')
    >>> plt.grid(True)
    >>> plt.show()
    >>> # Test wavelet function's properties
    >>> print('Wavelet Integral = %f + i %f (should be zero)'%(np.trapz(np.real(m), dx=dx_wave), np.trapz(np.imag(m), dx=dx_wave)))
    >>> print('Sum of squares = %f (should be 1)'%np.sum(np.abs(m)**2))
    >>> print('Sum of squares of FFT = %f (should be N)'%np.sum(np.abs(fft(m, norm='backward'))**2))
    """

    # omega=6.203607835633639 for scales == fourier_periods

    x = np.arange(-(N // 2) * dx, np.ceil(N / 2) * dx, dx)
    eta = -x / scale
    y = 0.7511255444649425 * np.exp(-1j * omega * eta) * np.exp(-(eta**2) / 2)

    # to ensure Sum(w**2) = 1 and Sum(FFT(w)**2) = N
    if norm:
        y *= np.sqrt(dx / scale)

    # roll so that it starts at zero and folds back at the end of the series
    if roll:
        y = np.roll(y, N // 2)

    # normalization factor for wavelet application
    wavelet_norm_factor = 0.74044116  # for omega = 6.20360...
    if omega == 6:
        wavelet_norm_factor = 0.776  # for omega = 6
    # ... add other cases as necesssary

    return y, x, wavelet_norm_factor


def wavelet_scales(minScale, maxScale, dj):
    M = np.log2(maxScale / minScale)
    scales = minScale * np.power(2, np.arange(0, M, dj))
    return scales


def wavelet_transform(x, dx, minScale, maxScale, wavelet_function=morlet_wave, dj=0.1):
    """
    Apply the wavelet transform on time series data.


    Parameters
    ----------
    x: (N,) Array like
        Input time series
    dx: float
        The time step of the data, i.e. the reciprocal of the sampling rate
    wavelet_function: function
        The wavelet mother function to use in the transform
    minScale: float
        The smallest scale to use for the wavelet transform
    maxScale: float
        The largest scale to use for the wavelet transform
    dj: float
        The step size to use for generating the scales that will be used for
        the wavelet transform. Scales are generated using the form:
            scales = minScale * np.power(2, np.arange(0, M, dj))
        with M being given by np.log2(maxScale/minScale)+dj, ensuring that the
        maximum scale will be equal to maxScale

    Returns
    ----------
    wave_mat: (M,N) array_like
        2-D array with the complex values of the wavelet transform. Each row is
        a different scale and each column a different moment in time
    scales: (M,) array_like
        1-D array with the values of the scales that were used for the wavelet
        transform


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import tfalib
    >>> fs = 8
    >>> T = np.arange(0, 10000, 1/fs);
    >>> N = len(T)
    >>> dj=0.1
    >>> W, scales = tfalib.wavelet_transform(X, 1/fs, tfalib.morlet_wave, 2, 1000, dj)
    >>> Wsq = np.abs(W)**2
    >>> log2scales = np.log2(scales)
    >>> plt.figure()
    >>> plt.imshow(Wsq[91:0:-1,:], aspect='auto',
    >>>            extent=[T[0], T[-1], log2scales[0], log2scales[-1]])
    >>> plt.yticks(np.arange(log2scales[0],log2scales[-1]+dj),
    >>>            labels=2**np.arange(log2scales[0],log2scales[-1]+dj))
    """

    N = len(x)
    scales = wavelet_scales(minScale, maxScale, dj)
    N_scales = len(scales)
    wave_mat = np.full((N_scales, N), 0 + 0 * 1j)
    Fx = fft(x, norm="backward")
    for i in range(N_scales):
        s = scales[i]
        w = wavelet_function(N, s, dx)[0]
        Fw = fft(w, norm="backward")
        c = ifft(Fx * Fw, norm="backward")

        wave_mat[i, :] = c

    return wave_mat, scales


def wavelet_normalize(wave_sq_matrix, scales, dx, dj, wavelet_norm_factor):
    """
    Apply a normalization to the squared magnitude of the output of the wavelt
    transform so that its results are compatible with the FFT.


    Parameters
    ----------
    wave_sq_matrix: (M,N) Array like
        The square of the magnitude of the output of the wavelet transform
    scales: (M,) array_like
        1-D array with the values of the scales that were used for the wavelet
        transform
    dx: float
        The time step of the data, i.e. the reciprocal of the sampling rate
    dj: float
        The step size to use for generating the scales that will be used for
        the wavelet transform. Scales are generated using the form:
            scales = minScale * np.power(2, np.arange(0, M, dj))
        with M being given by np.log2(maxScale/minScale)+dj, ensuring that the
        maximum scale will be equal to maxScale
    wavelet_norm_factor: float
        The wavelet-specific normalization factor that needs to be applied

    Returns
    ----------
    normalized_wave_sq_matrix: (M,N) Array like
        The normalized square of the magnitude of the output of the wavelet
        transform
    """
    return (
        (2 * dx * dj / wavelet_norm_factor)
        * wave_sq_matrix
        / np.reshape(scales, (-1, 1))
    )


def magn(X):
    """
    Return the row-wise magnitude of elements in 2D array 'X' as a single-column array.
    """
    return np.reshape(np.sqrt(np.sum(X**2, axis=1)), (-1, 1))


def mfa(B_NEC, B_MEAN_NEC, R_NEC=None):
    """ """

    # if no positional vector is given just assume the direction (0,0,-1) in NEC
    # coordinates, i.e. radial outwards (the magnitude is not necessary, only
    # the direction matters in order to compute its cross product with the mean
    # field component
    if R_NEC is None:
        R_NEC = np.zeros(B_NEC.shape)
        R_NEC[:, 2] = -1

    MFA = np.full(B_NEC.shape, np.NaN)

    # create the unitary vector of the mean field
    B_MEAN = magn(B_MEAN_NEC)
    B_MEAN_UNIT = B_MEAN_NEC / B_MEAN

    # find the field along the mean field direction
    MFA[:, 2] = np.sum(B_NEC * B_MEAN_UNIT, axis=1)

    # find the direction of the azimuthal component
    B_AZIM = np.cross(B_MEAN_UNIT, R_NEC)
    B_AZIM_UNIT = B_AZIM / magn(B_AZIM)

    # find the field along the azimuthal direction
    MFA[:, 1] = np.sum(B_NEC * B_AZIM_UNIT, axis=1)

    # find the direction of the poloidal component
    B_POL_UNIT = np.cross(B_AZIM_UNIT, B_MEAN_UNIT)
    # no need to normalize as this is already the cross product of two unitary vectors

    MFA[:, 0] = np.sum(B_NEC * B_POL_UNIT, axis=1)

    # test that magnitude is conserved
    # magn(B_NEC) / magn(MFA) == 1 True!

    return MFA
