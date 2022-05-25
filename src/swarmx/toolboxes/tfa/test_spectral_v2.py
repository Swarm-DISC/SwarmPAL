# -*- coding: utf-8 -*-
"""
Created on Thu May 12 19:00:39 2022

@author: magek
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import rfft, ifft, rfftfreq, fft, ifft
import matplotlib.pyplot as plt
import tfalib

plt.close('all')

fs = 8
T = np.arange(0, 10000, 1/fs);
N = len(T)

# TO DO: Built a better test signal
# Signal with a rapidly changing frequency is hard to describe via Fourier !!!

# fr = np.linspace(100/1000, 110/1000, N) # this creates issues ! 
# fr = 1/64
# fr = np.full(T.shape, 1.00)
# for i in range(10):
    # fr[np.arange(2000) + i*2000] = (i+1)*10/1000
# ampl = 7.25
# X = ampl * np.sin(2*np.pi*T * fr)
X = 0*np.sin(2*np.pi*T/256) + 5*np.sin(2*np.pi*T/64)

plt.figure()
plt.plot(T, X)
plt.grid(True)

print('Signal Energy = %f'%(np.trapz(X**2, dx=1/fs)))
print('Signal Variance = %f'%(np.var(X)))

#%% Generate Wavelet function

N_wave = 600
s_wave = 50
dx_wave = .5
m, m_t, m_norm = tfalib.morlet_wave(N_wave, s_wave, dx_wave, roll=False, norm=True)
plt.figure()
plt.plot(m_t, np.real(m), '-b', m_t, np.imag(m), '-r')
plt.grid(True)

#%% Test wavelet function's properties

print('Wavelet Integral = %f + i %f (should be zero)'%(np.trapz(np.real(m), dx=dx_wave), np.trapz(np.imag(m), dx=dx_wave)))
print('Sum of squares = %f (should be 1)'%np.sum(np.abs(m)**2))
print('Sum of sqares of FFT = %f (should be N)'%np.sum(np.abs(fft(m, norm='backward'))**2)) 

#%% Apply the Wavelet Transform
dj=0.1
W, scales = tfalib.wavelet_transform(X, 1/fs, tfalib.morlet_wave, 2, 1000, dj)
Wsq = np.abs(W)**2
Wn = tfalib.wavelet_normalize(Wsq, scales, 1/fs, dj, m_norm)
log2scales = np.log2(scales)

plt.figure()
plt.imshow(Wsq[91:0:-1,:], aspect='auto',
           extent=[T[0], T[-1], log2scales[0], log2scales[-1]])
plt.yticks(np.arange(log2scales[0],log2scales[-1]+dj), 
           labels=2**np.arange(log2scales[0],log2scales[-1]+dj))


plt.figure()
plt.plot(scales, (Wn[:,N//2]))

#%% Compare against a windowed FFT analysis

window_size = 8000
window_step = window_size//10

i_end = np.arange(window_size, N, window_step)
N_win = len(i_end)
FFT_freqs = rfftfreq(window_size, 1/float(fs))[1:] # ignore the first, i.e. 0th freq
N_freqs = len(FFT_freqs)

F = np.full((len(FFT_freqs), N_win), np.NaN)

for i in range(N_win):
    window = X[i_end[i]-window_size:i_end[i]]
    fft_power = np.abs(rfft(window)[1:]) / (window_size/2)
    # fft_power = np.abs(fft(window)[1:N_freqs+1]) / (window_size/2)
    F[:,i] = fft_power**2
  # plt.plot(FFT_freqs, fft_power)

T_win = T[i_end - int(window_size/2)]

plt.figure()
plt.plot(T_win, np.sum(F, axis=0), '*k', T, np.sum(Wn, axis=0), '-b')
plt.ylim([0, np.max(np.sum(F, axis=0))*1.1])


