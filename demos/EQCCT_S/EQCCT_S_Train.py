from __future__ import print_function
from __future__ import division


import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
import tensorflow
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
from tensorflow import keras
import time
from os import listdir
import platform
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
import contextlib
import sys
import warnings
from scipy import signal
from matplotlib.lines import Line2D
from obspy import read
from os.path import join
import json
import pickle
import faulthandler; faulthandler.enable()
import obspy
import logging
from obspy.signal.trigger import trigger_onset
from tensorflow.keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten, Reshape, multiply
from tensorflow.keras.layers import concatenate, GRU, Input, LSTM, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Add, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, SeparableConv1D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten, UpSampling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import layers, models, optimizers
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, GlobalAveragePooling1D
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose, Bidirectional, GRU, LSTM, Input,Dense, SpatialDropout1D, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation, Add
import numpy as np
from tensorflow.keras.layers import Input,Dense, Add, UpSampling1D, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation
from tensorflow.keras import optimizers, Model
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
from numpy.random import seed
import math
from tensorflow.keras.regularizers import l2
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False




def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
           
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))


import tensorflow as tf

w1 = 6000
w2 = 3
drop_rate = 0.2
stochastic_depth_rate = 0.1

positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (w1, w2)
num_classes = 1
input_shape = (6000, 3)
image_size = 6000  # We'll resize input images to this size
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    





def convF1(inpt, D1, fil_ord, Dr):
    '''
    encode = BatchNormalization()(inpt)    
    encode = Activation(tf.nn.gelu')(encode)
    encode = SpatialDropout1D(Dr)(encode, training=True)
    encode  = Conv1D(D1,  fil_ord, strides =(1), padding='same')(encode)
    
     
    encode  = Conv1D(D1,  fil_ord, strides =(1), padding='same')(inpt)
    encode = BatchNormalization()(encode)    
    encode = Activation(tf.nn.gelu')(encode)
    encode = Dropout(Dr)(encode)
    '''
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = Activation(tf.nn.gelu')(inpt)
    pre = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = BatchNormalization()(pre)    
    pre = Activation(tf.nn.gelu)(pre)
    
    #shared_conv = Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf  = Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = BatchNormalization()(inf)    
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf,inpt])
    
    inf1  = Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = BatchNormalization()(inf1)  
    inf1 = Activation(tf.nn.gelu)(inf1)    
    encode = Dropout(Dr)(inf1)

    return encode

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded

def create_cct_modelP(inputs):

    inputs1 = convF1(inputs,   10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = layers.Reshape((6000,1,40))(inputs1)

    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        #encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        #attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelS(inputs):

    inputs1 = convF1(inputs,   10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation






import keras
from keras.models import load_model
from keras import backend as K
from keras.layers import Input
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
import os
import shutil
import multiprocessing
from EqT_utils_S import DataGenerator, _lr_schedule, data_reader
import datetime
from tqdm import tqdm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# https://github.com/smousavi05/EQTransformer/tree/master/EQTransformer/core
# A modified Version
def trainer1(input_hdf5=None,
            output_name=None,                
            input_dimention=(6000, 3),
            shuffle=True, 
            label_type='gaussian',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3, 
            drop_channel_r=0.5,
            add_gap_r=0.2,
            scale_amplitude_r=None,
            pre_emphasis=False,                
            mode='generator',
            batch_size=200,
            epochs=200, 
            monitor='val_loss',
            patience=12,
            multi_gpu=False,
            number_of_gpus=4,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True):
        
    """
    
    Generate a model and train it.  
    
    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of data with NumPy arrays containing 3 component waveforms each 1 min long.


    output_name: str, default=None
        Output directory.
        
    input_dimention: tuple, default=(6000, 3)
        OLoss types for S picking respectively. 

    shuffle: bool, default=True
        To shuffle the list prior to the training.

    label_type: str, default='triangle'
        Labeling type. 'gaussian', 'triangle', or 'box'. 

    normalization_mode: str, default='std'
        Mode of normalization for data preprocessing, 'max': maximum amplitude among three components, 'std', standard deviation. 

    augmentation: bool, default=True
        If True, data will be augmented simultaneously during the training.

    add_event_r: float, default=0.6
        Rate of augmentation for adding a secondary event randomly into the empty part of a trace.

    shift_event_r: float, default=0.99
        Rate of augmentation for randomly shifting the event within a trace.
      
    add_noise_r: float, defaults=0.3 
        Rate of augmentation for adding Gaussian noise with different SNR into a trace.       
        
    drop_channel_r: float, defaults=0.4 
        Rate of augmentation for randomly dropping one of the channels.

    add_gap_r: float, defaults=0.2 
        Add an interval with zeros into the waveform representing filled gaps.       
        
    scale_amplitude_r: float, defaults=None
        Rate of augmentation for randomly scaling the trace. 
              
    pre_emphasis: bool, defaults=False
        If True, waveforms will be pre-emphasized. Defaults to False.  
        
          
    mode: str, defaults='generator'
        Mode of running. 'generator', or 'preload'. 
         
    batch_size: int, default=200
        Batch size.
          
    epochs: int, default=200
        The number of epochs.
          
    monitor: int, default='val_loss'
        The measure used for monitoring.
           
    patience: int, default=12
        The number of epochs without any improvement in the monitoring measure to automatically stop the training.          
        
    multi_gpu: bool, default=False
        If True, multiple GPUs will be used for the training. 
           
    number_of_gpus: int, default=4
        Number of GPUs uses for multi-GPU training.
           
    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None. 
         
    gpu_limit: float, default=None
        Set the maximum percentage of memory usage for the GPU.
        
    use_multiprocessing: bool, default=True
        If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction. 

        
    """     


    args = {
    "input_hdf5": input_hdf5,
    "output_name": output_name,
    "input_dimention": input_dimention,
    "shuffle": shuffle,
    "label_type": label_type,
    "normalization_mode": normalization_mode,
    "augmentation": augmentation,
    "add_event_r": add_event_r,
    "shift_event_r": shift_event_r,
    "add_noise_r": add_noise_r,
    "add_gap_r": add_gap_r,
    "drop_channel_r": drop_channel_r,
    "scale_amplitude_r": scale_amplitude_r,
    "pre_emphasis": pre_emphasis,
    "mode": mode,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,           
    "multi_gpu": multi_gpu,
    "number_of_gpus": number_of_gpus,           
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "use_multiprocessing": use_multiprocessing
    }
                       
    def train(args):
        """ 
        
        Performs the training.
    
        Parameters
        ----------
        args : dic
            A dictionary object containing all of the input parameters. 

        Returns
        -------
        history: dic
            Training history.  
            
        model: 
            Trained model.
            
        start_training: datetime
            Training start time. 
            
        end_training: datetime
            Training end time. 
            
        save_dir: str
            Path to the output directory. 
            
        save_models: str
            Path to the folder for saveing the models.  
            
        training size: int
            Number of training samples.
            
        validation size: int
            Number of validation samples.  
            
        """    

        
        save_dir, save_models=_make_dir(args['output_name'])
        training, validation=_split(args, save_dir)
        callbacks=_make_callback(args, save_models)
        model=_build_model(args)
        model.summary()  
        
        if args['gpuid']:           
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            tf.Session(config=tf.ConfigProto(log_device_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
            K.tensorflow_backend.set_session(tf.Session(config=config))
            
        start_training = time.time()                  
            
        if args['mode'] == 'generator': 
            
            params_training = {'file_name': str(args['input_hdf5']), 
                              'dim': args['input_dimention'][0],
                              'batch_size': args['batch_size'],
                              'n_channels': args['input_dimention'][-1],
                              'shuffle': args['shuffle'],  
                              'norm_mode': args['normalization_mode'],
                              'label_type': args['label_type'],                          
                              'augmentation': args['augmentation'],
                              'add_event_r': args['add_event_r'], 
                              'add_gap_r': args['add_gap_r'],  
                              'shift_event_r': args['shift_event_r'],                            
                              'add_noise_r': args['add_noise_r'], 
                              'drop_channe_r': args['drop_channel_r'],
                              'scale_amplitude_r': args['scale_amplitude_r'],
                              'pre_emphasis': args['pre_emphasis']}    
                        
            params_validation = {'file_name': str(args['input_hdf5']),  
                                 'dim': args['input_dimention'][0],
                                 'batch_size': args['batch_size'],
                                 'n_channels': args['input_dimention'][-1],
                                 'shuffle': False,  
                                 'norm_mode': args['normalization_mode'],
                                 'augmentation': False}         

            training_generator = DataGenerator(training, **params_training)
            validation_generator = DataGenerator(validation, **params_validation) 

            print('Started training in generator mode ...') 
            print(training_generator)
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          use_multiprocessing=args['use_multiprocessing'],
                                          workers=multiprocessing.cpu_count(),    
                                          callbacks=callbacks, 
                                          epochs=args['epochs'],verbose=2)
        else:
            print('Please specify training_mode !', flush=True)
        end_training = time.time()  
        
        return history, model, start_training, end_training, save_dir, save_models, len(training), len(validation)
                  
    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)  
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)





def _make_dir(output_name):
    
    """ 
    
    Make the output directories.

    Parameters
    ----------
    output_name: str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir: str
        Full path to the output directory.
        
    save_models: str
        Full path to the model directory. 
        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
        save_models = os.path.join(save_dir, 'models')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
    return save_dir, save_models



def _build_model(args): 
    
    """ 
    
    Build and compile the model.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
               
    Returns
    -------   
    model: 
        Compiled model.
        
    """       
    


    # Model EQCCT
    inputs = layers.Input(shape=input_shape,name='input')
    
    featuresS = create_cct_modelS(inputs)
    featuresS = Reshape((6000,1))(featuresS)

    logits  = Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)
    model = Model(inputs=[inputs], outputs=[logits])    
    

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc',f1,precision, recall])    
    
    
   # model.load_weights('./ModelPS/test_trainer_021.h5')
  
    return model  
    

def _split(args, save_dir):
    
    """ 
    
    Split the list of input data into training, validation, and test set.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_dir: str
       Path to the output directory. 
              
    Returns
    -------   
    training: str
        List of trace names for the training set. 
    validation : str
        List of trace names for the validation set. 
                
    """       
    
    #df = pd.read_csv(args['input_csv'])
    #ev_list = df.trace_name.tolist()    
    #np.random.shuffle(ev_list)     
    #training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    #validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
    #                        int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
    #test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    
    training = np.load('train_EQ_Texas.npy')
    validation = np.load('valid_EQ_Texas.npy')
    test = np.load('test_EQ_Texas.npy')
    #training = training[0:1000]
    
    np.save(save_dir+'/test', test) 
    np.save(save_dir+'/training', training)  
    np.save(save_dir+'/validation', validation)  

    return training, validation 



def _make_callback(args, save_models):
    
    """ 
    
    Generate the callback.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_models: str
       Path to the output directory for the models. 
              
    Returns
    -------   
    callbacks: obj
        List of callback objects. 
        
        
    """    
    
    m_name=str(args['output_name'])+'_{epoch:03d}.h5'   
    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'], 
                                           patience=args['patience']) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'], 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True,
                                  save_weights_only=True)  
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks
 
    


def _pre_loading(args, training, validation):
    
    """ 
    
    Load data into memory.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    training: str
        List of trace names for the training set. 
        
    validation: str
        List of trace names for the validation set. 
              
    Returns
    -------   
    training_generator: obj
        Keras generator for the training set. 
        
    validation_generator: obj
        Keras generator for the validation set. 
        
        
    """   
    
    training_set={}
    fl = h5py.File(args['input_hdf5'], 'r')   
    
    print('Loading the training data into the memory ...')
    pbar = tqdm(total=len(training)) 
    for ID in training:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get(str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get(str(ID))
        training_set.update( {str(ID) : dataset})  

    print('Loading the validation data into the memory ...', flush=True)            
    validation_set={}
    pbar = tqdm(total=len(validation)) 
    for ID in validation:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get(str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get(str(ID))
        validation_set.update( {str(ID) : dataset})  
   
    params_training = {'dim':args['input_dimention'][0],
                       'batch_size': args['batch_size'],
                       'n_channels': args['input_dimention'][-1],
                       'shuffle': args['shuffle'],  
                       'norm_mode': args['normalization_mode'],
                       'label_type': args['label_type'],
                       'augmentation': args['augmentation'],
                       'add_event_r': args['add_event_r'], 
                       'add_gap_r': args['add_gap_r'],                         
                       'shift_event_r': args['shift_event_r'],  
                       'add_noise_r': args['add_noise_r'], 
                       'drop_channe_r': args['drop_channel_r'],
                       'scale_amplitude_r': args['scale_amplitude_r'],
                       'pre_emphasis': args['pre_emphasis']}  

    params_validation = {'dim': args['input_dimention'][0],
                         'batch_size': args['batch_size'],
                         'n_channels': args['input_dimention'][-1],
                         'shuffle': False,  
                         'norm_mode': args['normalization_mode'],
                         'augmentation': False}  
    
    training_generator = PreLoadGenerator(training, training_set, **params_training)  
    validation_generator = PreLoadGenerator(validation, validation_set, **params_validation) 
    
    return training_generator, validation_generator  




def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args): 

    """ 
    
    Write down the training results.

    Parameters
    ----------
    history: dic
        Training history.  
   
    model: 
        Trained model.  

    start_training: datetime
        Training start time. 

    end_training: datetime
        Training end time.    
         
    save_dir: str
        Path to the output directory. 

    save_models: str
        Path to the folder for saveing the models.  
      
    training_size: int
        Number of training samples.    

    validation_size: int
        Number of validation samples. 

    args: dic
        A dictionary containing all of the input parameters. 
              
    Returns
    -------- 
    ./output_name/history.npy: Training history.    

    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.

    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.         

    ./output_name/X_learning_curve_loss.png: The learning curve of loss.  
        
        
    """   
    
    np.save(save_dir+'/history',history.history)
    ''' 
    hist_df = pd.DataFrame(history.history)
    hist_csv = 'save_dir+'/history.csv'
    with open(hist_csv,mode='w') as f:
        hist_df.to_csv(f)
    '''
    #model.save(save_dir+'/final_model.h5')
    #model.to_json()   
    #model.save_weights(save_dir+'/model_weights.h5')


    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        the_file.write('input_dimention: '+str(args['input_dimention'])+'\n')     
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        the_file.write('mode of training: '+str(args['mode'])+'\n')   
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')   
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n') 
        the_file.write('multi_gpu: '+str(args['multi_gpu'])+'\n')
        the_file.write('number_of_gpus: '+str(args['number_of_gpus'])+'\n') 
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('label_type: '+str(args['label_type'])+'\n')
        the_file.write('augmentation: '+str(args['augmentation'])+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')               
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('add_event_r: '+str(args['add_event_r'])+'\n')
        the_file.write('add_noise_r: '+str(args['add_noise_r'])+'\n')   
        the_file.write('shift_event_r: '+str(args['shift_event_r'])+'\n')                            
        the_file.write('drop_channel_r: '+str(args['drop_channel_r'])+'\n')            
        the_file.write('scale_amplitude_r: '+str(args['scale_amplitude_r'])+'\n')            
        the_file.write('pre_emphasis: '+str(args['pre_emphasis'])+'\n')


trainer1(input_hdf5= '/home/omar/EQCCT_Texas/Texas_Test_WithNoise_Final.h5',
        output_name='test_trainer_S',                
        shuffle=True, 
        label_type='triangle',
        normalization_mode='std',
        augmentation=True,
        add_event_r=0.5,
        shift_event_r=0.99,
        add_noise_r=0.5, 
        drop_channel_r=0.1,
        add_gap_r=None,
        scale_amplitude_r=None,
        pre_emphasis=False,               
        mode='generator',
        batch_size=40,
        epochs=50, 
        patience=10,
        gpuid=None,
        gpu_limit=None)

