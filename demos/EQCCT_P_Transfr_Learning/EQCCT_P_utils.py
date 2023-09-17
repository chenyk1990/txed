#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://github.com/smousavi05/EQTransformer/tree/master/EQTransformer/core
# A modified Version
from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import tensorflow
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, Activation, LSTM, Conv1D, InputSpec
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from obspy.signal.trigger import trigger_onset
import matplotlib
from numpy import NAN
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DataGenerator(tensorflow.keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Name of hdf5 file containing waveforms data.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    phase_window: int, fixed=40
        The number of samples (window) around each phaset.
            
    shuffle: bool, default=True
        Shuffeling the list.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    label_type: str, default=gaussian 
        Labeling type: 'gaussian', 'triangle', or 'box'.
             
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.

    coda_ratio: {float, 0.4}, default=0.4
        % of S-P time to extend event/coda envelope past S pick.       
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.

    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_P': y2}: outputs including three separate numpy arrays as labels for P.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 phase_window= 40, 
                 shuffle=True, 
                 norm_mode = 'max',
                 label_type = 'gaussian',                 
                 augmentation = False, 
                 add_event_r = None,
                 add_gap_r = None,
                 coda_ratio = 0.4,
                 shift_event_r = None,
                 add_noise_r = None, 
                 drop_channe_r = None, 
                 scale_amplitude_r = None, 
                 pre_emphasis = True,
                 **kwargs):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type       
        self.augmentation = augmentation   
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r
        self.coda_ratio = coda_ratio
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channe_r = drop_channe_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation:
            return 2*int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.augmentation:
            indexes = self.indexes[index*self.batch_size//2:(index+1)*self.batch_size//2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y2 = self.__data_generation(list_IDs_temp)
        return ({'input': X}, {'picker_P': y2})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
        return data_noisy   
         
    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
       
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      
    
    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y2 = np.zeros((self.batch_size, self.dim, 1))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            additions = None
            dataset = fl.get(str(ID))
            data = np.array(dataset['data'])
            
            if ID.split('_')[-1] == 'EV':
                spt = int(dataset.attrs['p_arrival_sample']);
                sst = int(dataset.attrs['s_arrival_sample']);
                coda_end = int(dataset.attrs['coda_end_sample']);
                snr = dataset.attrs['snr_db'];
                    
                
           
            ## augmentation 
            if self.augmentation == True:                 
                if i <= self.batch_size//2:   
                    if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                                       
                    if self.norm_mode:                    
                        data = self._normalize(data, self.norm_mode)  
                else:                  
                    if dataset.attrs['trace_category'] == 'earthquake_local':                   
                        if self.shift_event_r:
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r); 
                            
                        if self.add_event_r:
                            data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r); 
                                
                        if self.add_noise_r:
                            data = self._add_noise(data, snr, self.add_noise_r);
    
                        if self.drop_channe_r:    
                            data = self._drop_channel(data, snr, self.drop_channe_r);
                            data = self._adjust_amplitude_for_multichannels(data)  
                                    
                        if self.scale_amplitude_r:
                            data = self._scale_amplitude(data, self.scale_amplitude_r); 
                                    
                        if self.pre_emphasis:  
                            data = self._pre_emphasis(data) 
                                    
                        if self.norm_mode:    
                            data = self._normalize(data, self.norm_mode)                            
                                    
                    elif dataset.attrs['trace_category'] == 'noise':
                        if self.drop_channe_r:    
                            data = self._drop_channel_noise(data, self.drop_channe_r);
                            
                        if self.add_gap_r:    
                            data = self._add_gaps(data, self.add_gap_r)
                            
                        if self.norm_mode: 
                            data = self._normalize(data, self.norm_mode) 

            elif self.augmentation == False:  
                if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                    data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                     
                if self.norm_mode:                    
                    data = self._normalize(data, self.norm_mode)                          

            X[i, :, :] = data                                       

            ## labeling 

            if self.label_type  == 'triangle' and dataset.attrs['trace_category'] == 'earthquake_local':                      
                sd = None    
                if spt and sst: 
                    sd = sst - spt                      

                if spt and (spt-20 >= 0) and (spt+21 < self.dim):
                    y2[i, spt-20:spt+21, 0] = self._label()
                elif spt and (spt+21 < self.dim):
                    y2[i, 0:spt+spt+1, 0] = self._label(a=0, b=spt, c=2*spt)
                elif spt and (spt-20 >= 0):
                    pdif = self.dim - spt
                    y2[i, spt-pdif-1:self.dim, 0] = self._label(a=spt-pdif, b=spt, c=2*pdif)          

                if additions: 
                    add_spt = additions[0];
                    add_sst = additions[1];
                    add_sd = None
                    if add_spt and add_sst: 
                        add_sd = add_sst - add_spt                     
                   

                    if add_spt and (add_spt-20 >= 0) and (add_spt+21 < self.dim):
                        y2[i, add_spt-20:add_spt+21, 0] = self._label()
                    elif add_spt and (add_spt+21 < self.dim):
                        y2[i, 0:add_spt+add_spt+1, 0] = self._label(a=0, b=add_spt, c=2*add_spt)
                    elif add_spt and (add_spt-20 >= 0):
                        pdif = self.dim - add_spt
                        y2[i, add_spt-pdif-1:self.dim, 0] = self._label(a=add_spt-pdif, b=add_spt, c=2*pdif)
    
             

        fl.close() 
                           
        return X, y2.astype('float32')




class DataGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_P': y2}: outputs including three separate numpy arrays as labels for P.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data    


    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = fl.get(str(ID))
            data = np.array(dataset['data'])              

          
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X




def data_reader( list_IDs, 
                 file_name, 
                 dim=6000, 
                 n_channels=3, 
                 norm_mode='max',
                 augmentation=False, 
                 add_event_r=None,
                 add_gap_r=None,
                 coda_ratio=0.4,
                 shift_event_r=None,                                  
                 add_noise_r=None, 
                 drop_channe_r=None, 
                 scale_amplitude_r=None, 
                 pre_emphasis=True,
                 **kwargs):   
    
    """ 
    
    For pre-processing and loading of data into memory. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 datasets.
            
    dim: int, default=6000
        Dimension of input traces, in sample. 
           
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.

    coda_ratio: {float, 0.4}, default=0.4
        % of S-P time to extend event/coda envelope past S pick.
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.
            
    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input { 'picker_P': y2}: outputs including three separate numpy arrays as labels for P.
            
    Note
    -----
    Label type is fixed to box.
    
        
    """  
    
    def _normalize( data, mode = 'max'):
        'Normalize waveforms in each batch'
          
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude( data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel( data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])   
        else:
            data_noisy = data
        return data_noisy    
         
    def _adjust_amplitude_for_multichannels(data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(a=0, b=20, c=40): 
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y
    
    def _add_event(data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
        
        added = np.copy(data)
        additions = spt_secondEV = sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr >= 10.0) and (data.shape[0]-s_p-21-coda_end) > 20: 
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added                
        return data, additions 



    def _shift_event(data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end   
    
    def _pre_emphasis( data, pre_emphasis=0.97):
        'apply the pre_emphasis'
        
        for ch in range(n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    fl = h5py.File(file_name, 'r')

    if augmentation:
        X = np.zeros((2*len(list_IDs), dim, n_channels))
        y2 = np.zeros((2*len(list_IDs), dim, 1))
    else:
        X = np.zeros((len(list_IDs), dim, n_channels))
        y2 = np.zeros((len(list_IDs), dim, 1))

    # Generate data
    pbar = tqdm(total=len(list_IDs)) 
    for i, ID in enumerate(list_IDs):
        pbar.update()

        additions = None
        dataset = fl.get(str(ID))
        
        if ID.split('_')[-1] == 'EV':            
            data = np.array(dataset)                    
            spt = int(dataset.attrs['p_arrival_sample']);
            sst = int(dataset.attrs['s_arrival_sample']);
            coda_end = int(dataset.attrs['coda_end_sample']);
            snr = dataset.attrs['snr_db'];
                    
        elif ID.split('_')[-1] == 'NO':
            data = np.array(dataset)
           
        if augmentation:                 
            if dataset.attrs['trace_category'] == 'earthquake_local':                   
                data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r/2); 
            if norm_mode: 
                data1 = _normalize(data, norm_mode)   
                          
            if dataset.attrs['trace_category'] == 'earthquake_local':
                if shift_event_r and spt:
                    data, spt, sst, coda_end = _shift_event(data, spt, sst, coda_end, snr, shift_event_r);  
                          
                if add_event_r:
                    data, additions = _add_event(data, spt, sst, coda_end, snr, add_event_r); 
                    
                if drop_channe_r:    
                    data = _drop_channel(data, snr, drop_channe_r);
                  #  data = _adjust_amplitude_for_multichannels(data); 
                          
                if scale_amplitude_r:
                    data = _scale_amplitude(data, scale_amplitude_r); 
                    
                if pre_emphasis:  
                    data = _pre_emphasis(data);

                if add_noise_r:
                    data = _add_noise(data, snr, add_noise_r);
                    
                if norm_mode:    
                    data2 = _normalize(data, norm_mode); 
                     
                            
            if dataset.attrs['trace_category'] == 'noise':
                if drop_channe_r:    
                    data = _drop_channel_noise(data, drop_channe_r);
                if add_gap_r:    
                    data = _add_gaps(data, add_gap_r)                    
                if norm_mode:                    
                    data2 = _normalize(data, norm_mode) 
                    
            X[i, :, :] = data1 
            X[len(list_IDs)+i, :, :] = data2                                      

            if dataset.attrs['trace_category'] == 'earthquake_local': 

                if spt and (spt-20 >= 0) and (spt+21 < dim):
                    y2[i, spt-20:spt+21, 0] = _label()
                    y2[len(list_IDs)+i, spt-20:spt+21, 0] = _label()                   
                elif spt and (spt+21 < dim):
                    y2[i, 0:spt+spt+1, 0] = _label(a=0, b=spt, c=2*spt)
                    y2[len(list_IDs)+i, 0:spt+spt+1, 0] = _label(a=0, b=spt, c=2*spt)                   
                elif spt and (spt-20 >= 0):
                    pdif = dim - spt
                    y2[i, spt-pdif-1:dim, 0] = _label(a=spt-pdif, b=spt, c=2*pdif)
                    y2[len(list_IDs)+i, spt-pdif-1:dim, 0] = _label(a=spt-pdif, b=spt, c=2*pdif)
            
                     
                sd = sst - spt     
                if additions: 
                    add_spt = additions[0];
                    print(add_spt)
                    add_sst = additions[1];
                    add_sd = add_sst - add_spt 
                    
                    if add_spt and (add_spt-20 >= 0) and (add_spt+21 < dim):
                        y2[len(list_IDs)+i, add_spt-20:add_spt+21, 0] = _label()  
                    elif add_spt and (add_spt+21 < dim):
                        y2[len(list_IDs)+i, 0:add_spt+add_spt+1, 0] = _label(a=0, b=add_spt, c=2*add_spt)
                    elif add_spt and (add_spt-20 >= 0):
                        pdif = dim - add_spt
                        y2[len(list_IDs)+i, add_spt-pdif-1:dim, 0] = _label(a=add_spt-pdif, b=add_spt, c=2*pdif) 
                        

    fl.close()                           
    return X.astype('float32'), y2.astype('float32')







def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind




def picker(args, yh3, yh3_std, spt=None):
    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
        
    yh3 : 1D array
        P arrival probabilities.  
        
    yh3_std : 1D array
        P arrival standard deviations.  
        
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
                           
                
    """               
        


    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]



    sP_arr = _detect_peaks(yh3, mph=args['P_threshold'], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]

            if  args['estimate_uncertainty'] and sauto:
                P_uncertainty = np.round(yh3_std[int(sauto)], 3)

            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        perrorall.append(int(spt- swave))  
        Ppickall.append(int(swave))
        Pproball.append(int(np.max(so)))
    except:
        perrorall.append(None)
        Ppickall.append(None)
        Pproball.append(None)


    #Ppickall = np.array(Ppickall)
    #perrorall = np.array(perrorall)  
    #Pproball = np.array(Pproball)
    
    return Ppickall, perrorall, Pproball





def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   

    
    

def f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def normalize(data, mode='std'):
    
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """   
    
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
    
 


def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



