# -*- coding: utf-8 -*-
"""
# INSERT ESA PROJECT BLOCK #

@author: constantinos@noa.gr
"""

import datetime as dt
from collections import OrderedDict
from abc import ABC, abstractmethod
from viresclient import SwarmRequest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import tfalib



class TFA_Data():
    d = None
    t = None
    X = None
    DF = None
    s = None
    W = None
    I = None
    label = None
    meta = OrderedDict()
    
    
    
    def __init__(self, params=None):
        if params is None:
            params = {'Mission': 'SW', 'Source': 'A', 'Type': 'MAGx_LR', 
                      'Field': 'B_NEC', 'Component': [True, True, True], 'Magn': False,
                      'time_lims': [dt.datetime(2020, 3, 14), dt.datetime(2020, 3, 14, 0, 59, 59)],
                      'freq_lims': [0.020, 0.100], 'lat_lims': None, 
                      'maglat_lims': [-60, 60],
                      'Models': ['CHAOS-Core', 'CHAOS-Static']
                      }
            self.meta = {'General': params}
        else:
            self.meta = params
        
        
        
    def retrieve_data(self):
        if self.meta['General']['Mission'] == 'SW':
            request = SwarmRequest()
            # construct file_type, i.e. MAGA_LR etc
            file_type = self.meta['General']['Type'].replace('x', 
                                                         self.meta['General']['Source'])
            level = '1B'
            request.set_collection("SW_OPER_" + file_type + "_" + level)
            
            
            # decide sampling rate
            if self.meta['General']['Type'] == 'MAGx_LR':
                sampling_time = 'PT1S'
            elif self.meta['General']['Type'] == 'MAGx_HR':
                sampling_time = 'PT0.019S'
            elif self.meta['General']['Type'] == 'EFIx_LP':
                sampling_time = 'PT0.5S'
            else:
                pass
            self.meta['General']['Sampling_Time'] = sampling_time
            
            # calculate CHAOS models to subtract from measurements
            models = None
            if 'Models' in self.meta['General'].keys():
                models = self.meta['General']['Models']
            
            varname = self.meta['General']['Field']
            mfa = False
            if varname.lower() == 'b_mfa':
                varname = 'B_NEC'
                mfa = True
                
            # flagname = 'lalaalla'
            
            # create request
            request.set_products(measurements=[varname],
                                  models = models,
                                  residuals=False,
                                  auxiliaries=["QDLat", "QDLon", "MLT", 
                                              'Latitude', 'Longitude', 'Radius'],
                                  sampling_step=sampling_time)
            
            
            # request.set_range_filter ( ... ) 
            # do multiple passes for each e.g. flag value

            
            # Extend time-lims by three hours before and after the given times
            # to account for edge effects. All plots and results will use the
            # input time_lims, but data retrieval and processing will operate
            # on the extended time interval
            margin = dt.timedelta(hours=3)
            if 'HR' in self.meta['General']['Type']:
                margin = dt.timedelta(minutes=5)
            extended_time_lims = [self.meta['General']['time_lims'][0] - margin,
                                self.meta['General']['time_lims'][1] + margin]
            
            
            # perform the request
            res = request.get_between(start_time = extended_time_lims[0],
                                      end_time = extended_time_lims[1])
            
            # get result and assign values to local variables
            df = res.as_dataframe()
            D = df.index
            self.t = md.date2num(D)
            self.X = np.stack(df[varname].to_numpy())
            self.DF = df
            self.d = md.num2date(self.t)
            self.label = self.meta['General']['Mission'] + '-' + self.meta['General']['Source'] +\
                ' ' + file_type + ' ' + self.meta['General']['Field']
            
            #
            # flag-based cleaning here as well! Add the flag name as an extra 
            # variable in the inputs
            #
            
            
            # subtract CHAOS models       
            # 
            # If both CHAOS-Core & CHAOS-Static are present, do I have to 
            # subtract both sequentially, or does the Static include the core 
            # field somehow???
            # 
            model_field = np.full(self.X.shape, 0.0)
            if 'Models' in self.meta['General'].keys():
                for i, m in enumerate(models):
                    model_field += np.stack(df[varname + '_' + m].to_numpy())
                self.X -= model_field
           
            
           # perform Mean Field Alligned transformation (if needed)
            if mfa:
                self.X = tfalib.mfa(self.X, model_field)
            
                                    
            # isolate the appropriate components according to input parameters
            dims = sum(self.meta['General']['Component'])
            if dims == 1:
                self.X = np.reshape(self.X , (-1, 1))
            else:
                self.X = np.reshape(self.X , (-1, 3))
                self.X = self.X[:, self.meta['General']['Component']]


            # calculate magnitude of selected components (if requested)
            if self.meta['General']['Magn']:
                self.X = tfalib.magn(self.X)
    

    def plot(self, data='Series'):
        if data == 'Series':
            x = self.X
        elif data == 'Index':
            x = self.I
        elif data=='QD':
            pass
        
        D = self.X.shape[1]
        
        plt.figure()
        for i in range(D):
            plt.subplot(D,1,i+1)
            plt.plot(self.d, x[:,i], '-b')
            plt.xlim(self.meta['General']['time_lims'])
            plt.grid(True)
            if i == 0:
                plt.title(self.label)
    
    
    def image(self):
        M, N, D = self.W.shape
        freqs = 1000/self.s
        # freq_lims = [freqs[0], freqs[-1]]
        # yticks = np.hstack((np.arange(1,10), np.arange(10,200,20), 
        #                    np.arange(200,1000,200), np.arange(1000,10000,1000)))
        # yticks = np.append(yticks, freq_lims)
        # yticklabels = ['%.0f'%i for i in yticks]

        plt.figure()
        for i in range(D):
            m = np.max([np.log10(np.min(self.W[::, :, i])), -6])
            x = np.log10(np.max(self.W[::, :, i]))
            
            plt.subplot(D,1,i+1)
            plt.contourf(self.d, freqs, np.log10(self.W[::, :, i]), 
                         cmap='jet', levels=np.linspace(m, x, 20), 
                         extend='min')
            plt.xlim(self.meta['General']['time_lims'])
            # plt.yticks(ticks=yticks, labels=yticklabels)
            # plt.ylim(freq_lims)
            plt.ylabel('Freq (mHz)')
            cbh = plt.colorbar(orientation='vertical')
            cb_ticks = cbh.get_ticks()
            cbh.set_ticks(cb_ticks)
            cbh.set_ticklabels(['%.2f'%i for i in cb_ticks])
            if i == 0:
                plt.title(self.label)

    
    
    def wave_index(self):
        N, D = self.X.shape
        self.I = np.full((N,D), np.NaN)
        
        for i in range(D):
            self.I[:,i] = np.sum(self.W[:,:,i], axis=0)
            


class TFA_Process(ABC):
    params = None
    
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def apply(self, target):
        return
    
    # append the parameters to the "meta" dictionary of the target TFA_Data object
    def append_params(self, target):
        target.meta[self.__name__] = self.params



class Cadence(TFA_Process):
    __name__ = 'Cadence'
    
    def __init__(self, params=None):
        if params is None:
            self.params = {'Sampling_Rate': 86400, 'Interp': False}
        else:
             self.params = params
    
    def apply(self, target):
        target.t, target.X = tfalib.constant_cadence(target.t, target.X, 
                        self.params['Sampling_Rate'], self.params['Interp'])[0:2]
        self.append_params(target)
        
        return target
        
        
        
class Cleaning(TFA_Process):
    __name__ = 'Cleaning'
    
    def __init__(self, params=None):
        if params is None:
            self.params = {'Window_Size': 50, 'Method': 'iqr', 'Multiplier': 6}
        else:
             self.params = params
    
    def apply(self, target):
        inds = tfalib.outliers(target.X, self.params['Window_Size'], 
                        method = self.params['Method'], 
                        multiplier = self.params['Multiplier'])
        target.X[inds] = np.NaN
        
        # interpolate cleaned values and pre-existing gaps
        N, D = target.X.shape
        t_ind = np.arange(N)
        for i in range(D):
            x = np.reshape(target.X[:,i], (N,))
            nonNaN = ~ np.isnan(x)
            y = np.interp(t_ind, t_ind[nonNaN], x[nonNaN])
            target.X[:,i] = y
        
        self.append_params(target)
        
        return target



class Filtering(TFA_Process):
    __name__ = 'Filtering'
    
    def __init__(self, params=None):
        if params is None:
            self.params = {'Sampling_Rate': 1, 'Cutoff': 20/1000}
        else:
             self.params = params
    
    def apply(self, target):
        target.X = tfalib.filter(target.X, self.params['Sampling_Rate'], 
                                 self.params['Cutoff'])
        self.append_params(target)
        
        return target
    
    

class Wavelet(TFA_Process):
    __name__ = 'Wavelet'
    
    def __init__(self, params=None):
        if params is None:
            self.params = {'Time_Step': 1, 'Min_Scale': 1000/100, 
                           'Max_Scale': 1000/20, 'dj': 0.1}
        else:
             self.params = params
         
        self.params['Wavelet_Function'] = 'Morlet'
        self.params['Wavelet_Param'] = 6.2036
        self.params['Wavelet_Norm_Factor'] = 0.74044116
    
    def apply(self, target):
        
        N, D = target.X.shape
        target.s = tfalib.wavelet_scales(self.params['Min_Scale'], 
                                       self.params['Max_Scale'], 
                                       self.params['dj'])
        M = len(target.s)
        target.W = np.full((M, N, D),  np.NaN)
        
        for i in range(D):
            wave = tfalib.wavelet_transform(np.reshape(target.X[:,i], (N,)), 
                        dx = self.params['Time_Step'], 
                        minScale = self.params['Min_Scale'], 
                        maxScale = self.params['Max_Scale'], 
                        dj = self.params['dj'])[0]
            norm = tfalib.wavelet_normalize(np.abs(wave)**2, target.s, 
                          dx = self.params['Time_Step'], 
                          dj = self.params['dj'], 
                          wavelet_norm_factor = 0.74044116)
            target.W[:,:,i] = norm
        
        self.append_params(target)
        
        return target

    
    




