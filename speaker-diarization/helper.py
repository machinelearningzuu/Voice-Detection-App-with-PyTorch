import pickle
import librosa
import os, random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize

from variables import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def save_weights(weight_file, weights):
    file = open(weight_file, 'wb')
    pickle.dump(weights, file)
     
def load_weights(weight_file):
    file = open(weight_file, 'rb')
    obj = pickle.load(file)
    return obj

def VAD(wavData):
    ste = librosa.feature.rms(
                            y=wavData,
                            hop_length=hop_length
                             ).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    vad = (ste>thresh).astype('bool')
    vad = vad.reshape(-1,)
    return vad

def MFCC(wavData):
    mfcc = librosa.feature.mfcc(
                             y=wavData, 
                             sr=sample_rate, 
                             n_mfcc=n_mfcc,
                             hop_length=hop_length
                             ).T
    return mfcc

def LOGMEL(wavData):
    mel = librosa.feature.melspectrogram(
                                         wavData, 
                                         sr=sample_rate, 
                                         n_mels=n_mels,
                                         hop_length=hop_length
                                         ).T

    return librosa.amplitude_to_db(mel)

def process_mfcc(vad, mfcc):
    len_vad = vad.shape[0]
    len_mfcc = mfcc.shape[0]

    if len_mfcc > len_vad:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif len_mfcc < len_vad:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:]
    return vad, mfcc

def feature_extraction(wavFile, print_shapes = True):
    wavData = librosa.load(wavFile,sr=sample_rate)[0]
    vad = VAD(wavData)
    mfcc = MFCC(wavData)
    
    if print_shapes:
        print("VAD O/P shape  : {}".format(vad.shape))
        print("MFCC O/P shape : {}\n".format(mfcc.shape))

    vad, mfcc = process_mfcc(vad, mfcc)
    if print_shapes:
        print("Processed VAD O/P shape  : {}".format(vad.shape))
        print("Processed MFCC O/P shape : {}\n".format(mfcc.shape))
    
    return vad, mfcc

def plot_metrics(GMMfit, mfcc, max_components = 50):
    models = [GMMfit(mfcc, nth_component) for nth_component in np.arange(1, max_components)]
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(1, max_components), [m.bic(mfcc) for m in models], label='BIC')
    plt.plot(np.arange(1, max_components), [m.aic(mfcc) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('GMM n_components for an audio file')