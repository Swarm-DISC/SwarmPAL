# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:38:28 2022

@author: constantinos@noa.gr
"""

# from viresclient import set_token
from viresclient import SwarmRequest
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

#set_token("https://vires.services/ows")

SAT = 'A'
TYPE = 'MAG'
RES = 'LR'
FIELD = 'F'
START_TIME = dt.datetime(2015, 6, 23,    0, 0, 0)
END_TIME =   dt.datetime(2015, 6, 23,    5, 0, 0)
xlims = [dt.datetime(2015, 6, 23, 2, 51, 0), 
         dt.datetime(2015, 6, 23, 3, 38, 0)]

COLLECTION = "SW_OPER_" + TYPE + SAT + "_" + RES + "_1B"

request = SwarmRequest()
request.set_collection("SW_OPER_MAGA_LR_1B")
request.set_products(measurements=[FIELD],
                     models = ["CHAOS-Core", "CHAOS-Static"],
                     auxiliaries=["QDLat", "QDLon", "MLT"],
                     residuals=False,
                     sampling_step="PT1S")

data = request.get_between(start_time = START_TIME, end_time = END_TIME)
df = data.as_dataframe()
D = df.index
T = md.date2num(D)
X = df[FIELD].to_numpy()
Chaos_C = df['F_CHAOS-Core'].to_numpy()
Chaos_S = df['F_CHAOS-Static'].to_numpy()
Chaos = Chaos_C + Chaos_S
MLat = df['QDLat'].to_numpy()

#%% Cleaning & Pre-processing


#%% Filtering
from scipy import signal

F_CUTOFF = 3/1000

b, a = signal.butter(5, F_CUTOFF, btype='highpass', analog=False, output='ba', fs=1)
filtered = signal.filtfilt(b, a, X - np.mean(X))
filtered1 = signal.filtfilt(b, a, X - Chaos)
# filtered = signal.sosfilt(sos, X[::-1])
# filtered = -signal.sosfilt(sos, filtered[::-1])
# filtered1 = signal.sosfilt(sos, X - Chaos)

# b, a = signal.cheby2(5, 10, F_CUTOFF, btype='lowpass', analog=False, output='ba', fs=1)
# filtered = X - signal.filtfilt(b, a, X)
# filtered1 = X - Chaos - signal.filtfilt(b, a, X - Chaos)


#% 



plt.figure(1)
plt.subplot(3,1,1)
plt.plot(D, X, '-')
plt.xlim(xlims)
plt.ylim([30000, 60000])
plt.grid(True)
plt.ylabel('B (nT)')
plt.title(r'Swarm-A $|B|^{ASM}$ Field')
plt.subplot(3,1,2)
plt.plot(D, filtered1, '-r')
plt.xlim(xlims)
# plt.ylim([-30, 35])
plt.grid(True)
plt.ylabel('B (nT)')
plt.title('Filtered Field')
plt.subplot(3,1,3)
plt.plot(D, MLat, '-r')
plt.xlim(xlims)
plt.ylim([-90, 90])
plt.yticks([-90,-60,-30,0,30,60,90])
plt.grid(True)
plt.ylabel('Mag.Lat (deg)')
# plt.title('Ancillary')

#%% Wavelet

from scipy.fft import fft, ifft

FREQS = np.arange(1,50)/1000
DT = 1
OMEGA = 6

def wavelet_power(x, dt, freqs, omega = 6):
    N = len(x)
    Fx = fft(x)
    wk_pos = np.arange(0, np.floor(N/2)) * (2*np.pi)/(N*dt)
    L = len(wk_pos)
    fourier_factor = 4*np.pi / (omega + np.sqrt(2 + omega^2))
    N_freqs = len(freqs)
    
    W = np.zeros((N_freqs,N))
    
    for i in range(N_freqs):
        s = 1 / (fourier_factor * freqs[i])
        psi = np.pi**(-1/4) * np.exp(-((s * wk_pos - omega)**2)/2)
        psi = psi * np.sqrt(2*np.pi*s/dt)
        full_psi = np.hstack((psi, np.zeros((N-L))))
        W[i,:] = np.abs(ifft(Fx * full_psi))**2
        
    return W



#%%

import matplotlib.cm as cm

W = wavelet_power(filtered, DT, FREQS, OMEGA)
W1 = wavelet_power(filtered1, DT, FREQS, OMEGA)

levels = np.arange(-4, 4, .5)
plt.rcParams.update({'font.size':16})
    
fig = plt.figure(2)
# ax = plt.subplot(2,1,1)
# im = ax.contourf(D, FREQS*1000, np.log10(W), levels, cmap=cm.jet, extend='both')
# plt.colorbar(im, ticks=range(-10,10,1), label=r"$log_{10}(Wavelet Power)$")
# plt.xlim(xlims)
# # plt.xlabel('time')
# plt.ylabel('freq (mHz)')
# plt.title('High-pass Filtered only')

ax = plt.subplot(1,1,1)
im = ax.contourf(D, FREQS*1000, np.log10(W1), levels, cmap=cm.jet, extend='both')
plt.colorbar(im, ticks=range(-10,10,1), label=r"$log_{10}(Wavelet Power)$")
plt.xlim(xlims)
plt.xlabel('time')
plt.ylabel('freq (mHz)')
plt.title('Wavelet Spectrum')



# plt.figure(3)
# plt.semilogy(D, np.mean(W, axis=0), 'r-', D, np.mean(W1, axis=0), 'g-')
# plt.xlim(xlims)
# plt.xlabel('time')
# plt.grid(True)
# plt.legend(('High-pass Filtered only', 'Filtered Chaos Residual'))
