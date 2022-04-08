# -*- coding: utf-8 -*-
"""
# INSERT ESA PROJECT BLOCK #

@author: constantinos@noa.gr
"""

import sys
import numpy as np
import scipy.interpolate as interpolate



def constant_cadence(t_obs, x_obs, sampling_rate, interp = False):
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
    on the old ones.
    
    `t_obs` is a one-dimensional array with the  time in seconds
    `x_obs` is a one or two-dimensioanl array with real values
    `sampling_rate` is a real number, given in Hz
    
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
        sys.exit("constant_cadence: ERROR: `t_obs` argument must be 1-dimensional array")
    
    if len(x_obs.shape) > 2: 
        sys.exit("constant_cadence: ERROR: `x_obs` argument must be 1 or 2-dimensional array")
    
    N = len(t_obs)
    if len(x_obs.shape) == 2:
        multiDim = True
        
        if x_obs.shape[0] != N and x_obs.shape[1] != N:
            sys.exit("constant_cadence: ERROR: `x_obs` must have the same length as 1 t_obs`")
        elif x_obs.shape[0] != N and x_obs.shape[1] == N:
            x_obs = x_obs.T
            transp = True
        
        M = x_obs.shape[1] # number of variables
        
    elif len(x_obs.shape) == 1:
        multiDim = False
    
    dt = 1/sampling_rate
    time_range = np.max(t_obs) - np.min(t_obs)
    time_rec_N = np.ceil((time_range / dt))
    # init_t = np.round(t_obs[0]/dt) * dt
    init_t = t_obs[0]
    inds = np.abs(np.round((t_obs - init_t)/dt)).astype(int)
    
    t_rec = np.arange(init_t, init_t + time_rec_N*dt + 1E-6, dt)
    
    if multiDim:
        x_rec = np.full((N,M), np.NaN)
        x_rec[inds, :] = x_obs
    else:
        x_rec = np.full(t_rec.shape, np.NaN)
        x_rec[inds] = x_obs
        
    nonnan_mask = ~np.isnan(x_rec)
    
    if interp:
        # scipy interpolation that also extrapolates a bit if is needed
        # for the final point (default)
        f = interpolate.interp1d(t_obs, x_obs, kind='linear', 
                                       fill_value = 'extrapolate')
        x_int = f(t_rec)
        
        # Numpy interpolation, without extrapolation of final value
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
    
    