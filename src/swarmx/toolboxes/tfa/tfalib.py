# -*- coding: utf-8 -*-
"""
# INSERT ESA PROJECT BLOCK #

@author: constantinos@noa.gr
"""

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
    and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.
    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    """
    
    dt = 1/sampling_rate
    time_range = np.max(t_obs) - np.min(t_obs)
    time_rec_N = np.ceil((time_range / dt))
    # init_t = np.round(t_obs[0]/dt) * dt
    init_t = t_obs[0]
    inds = np.abs(np.round((t_obs - init_t)/dt)).astype(int)
    
    t_rec = np.arange(init_t, init_t + time_rec_N*dt + 1E-6, dt)
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
        
        return t_rec, x_int, nonnan_mask
    else:
        return t_rec, x_rec, nonnan_mask
    
    