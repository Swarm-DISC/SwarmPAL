# -*- coding: utf-8 -*-
"""
# INSERT ESA PROJECT BLOCK #

@author: constantinos@noa.gr
"""

from  tfa_classes import TFA_Data
from  tfa_classes import Cadence, Cleaning, Filtering, Wavelet


tfa = TFA_Data()

# tfa.meta['General']['Field'] = 'B_NEC'
# tfa.meta['General']['Component'] = [True, True, True]
# tfa.meta['General']['Magn'] = True

tfa.meta['General']['Field'] = 'B_NEC'
tfa.meta['General']['Component'] = [True, True, True]
tfa.meta['General']['Magn'] = False

# Data Retrieval (also CHAOS subtraction and Flag-based cleaning)
tfa.retrieve_data()
# tfa.X = np.reshape(np.sin(2*np.pi*tfa.t/0.00016550925925925926), (-1,1))

# Set to constant cadence
cadence_params = {'Sampling_Rate': 86400, 'Interp': False}
cadence = Cadence(cadence_params)
cadence.apply(tfa)

# Statistical Cleaning
cleaning_params = {'Window_Size': 50, 'Method': 'iqr', 'Multiplier': 6}
cleaning = Cleaning(cleaning_params)
cleaning.apply(tfa)

# Filtering
filtering_params = {'Sampling_Rate': 1, 'Cutoff': 20/1000}
filtering = Filtering(filtering_params)
filtering.apply(tfa)

tfa.plot('Series')

# Wavelet
wavelet_params = {'Time_Step': 1, 'Min_Scale': 1000/100, 
               'Max_Scale': 1000/10, 'dj': 0.1}
wavelet = Wavelet(wavelet_params)
wavelet.apply(tfa)

tfa.image()

# Wave Index
tfa.wave_index()

tfa.plot('Index')
