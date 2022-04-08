# -*- coding: utf-8 -*-
"""


@author: constantinos@noa.gr
"""

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '16'
plt.rcParams['lines.markersize'] = '10'
plt.rcParams['lines.markeredgewidth'] = 2.0
plt.rcParams['markers.fillstyle'] = 'none'

#%% Create sample time series to test despiking/cleaning/filtering etc

N = 100000 # approx No seconds in a day
t = np.arange(N)

# creata a slowly varying component 
slow = 40000 + 10000 * np.sin(2*np.pi*t/5000) * (1 + np.abs(np.cos(2*np.pi*t/(2*N))))

# create a fast varying component
fast = 5*np.sin(2*np.pi*t/600) # fast variation

# add random spikes
spike_pos = np.random.permutation(N)[:int(N/100)] # positions of 1000 spikes
x = slow + fast
x[spike_pos] = x[spike_pos] + 5

# plot
plt.figure(1)
plt.plot(t, x)
plt.grid(True)

#%% Test preprocessing routines

N = 10 # No Points to generate
FS = 50 # sampling rate in Hz

# create time vector
t = np.arange(N) * (1/FS)
# add some small deviations in time, just to complicate things
n = 0.1 * np.random.randn(N) * (1/FS)
t = 12.81 + t + n

# produce x values from a simple linear relation
x = 10 + 0.01 * t

plt.figure(2)
plt.plot(t, x, '-xk')
plt.grid(True)

# remove some values at random
inds_to_rem = np.random.permutation(np.arange(1,N-1))[:int(N/4)]
t_obs = np.delete(t, inds_to_rem)
x_obs = np.delete(x, inds_to_rem)

plt.figure(3)
plt.plot(t_obs, x_obs, '--xk')
plt.grid(True)

# reconstruct the original (approximately as the small deviations 
# will not be reproduced)

time_range = np.max(t_obs) - np.min(t_obs)
time_rec_N = np.ceil((time_range / (1/FS)))
# init_t = np.round(t_obs[0]/(1/FS)) * (1/FS)
init_t = t_obs[0]
inds = np.abs(np.round((t_obs - init_t)/(1/FS))).astype(int)

t_rec = np.arange(init_t, init_t + time_rec_N*(1/FS) + 1E-6, (1/FS))
x_rec = np.full(t_rec.shape, np.NaN)
x_rec[inds] = x_obs
nonnan_mask = ~np.isnan(x_rec)

f = interpolate.interp1d(t_obs, x_obs, kind='linear', 
                               fill_value = 'extrapolate')
x_int = f(t_rec)
# x_int = np.interp(t_rec, t_obs, x_obs)
x_int[~nonnan_mask] = np.NaN


plt.figure(3)
plt.plot(t_rec, x_rec, 'or', t_rec, x_int, '+b')
plt.grid(True)
#%%










# set data to constant cadence
# assuming time is in milliseconds
t = np.array([1., 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]) * 1000 
noise = np.random.randn(len(t)) * 10
t = t + noise
sampling_rate = 2
x = np.linspace(0,1,len(t))

dt = 1/sampling_rate

time_range = np.ceil((np.max(t) - np.min(t))/1000)/dt
time_ind = np.arange(time_range)

indices = np.round((t - np.min(t))/1000/dt)
new_x = np.full(time_ind.shape, np.NaN)
new_x[indices.astype(int)] = x









