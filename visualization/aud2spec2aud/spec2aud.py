import librosa
import librosa.display
import numpy as np
import scipy.signal
from scipy.io.wavfile import read,write
import math
import pandas as pd
import os
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import argparse
import sox
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--mag-path',type=str,help='Path to the spectogram magnitude')
parser.add_argument('--data-path',type=str,help='Path to the pickled dictionary containing phase, mean,std')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('-sp','--save-path',type=str,help='Path to save the spectogram/audio')
args = parser.parse_args()


# Example Mag Path: /media/data_dump/hemant/rachit/scripts/Robust_ASR/spectogram_audio/spectogram/sample-101535.npy
# Example Phase Path: /media/data_dump/hemant/rachit/scripts/Robust_ASR/spectogram_audio/phases/sample-101535.npy

def magphase_to_spect(mag,phase,mean,std):
    mag = mag[:phase.shape[0],:phase.shape[1]]
    print(mean,std)
    mag = mag*std + mean
    mag = np.expm1(mag) 
    D = np.multiply(mag,phase)
    return D

def spect_to_aud(D,hop_length,win_length,window,sample_rate,save_path,tfm):
    reconstructed_audio = librosa.istft(D,hop_length=hop_length,win_length=win_length,window=window)
    tfm.build_file(input_array=reconstructed_audio,sample_rate_in=sample_rate,output_filepath=save_path)

if __name__ == "__main__":
    mag_path = args.mag_path
    data_path = args.data_path
    window_stride = args.window_stride
    window_size = args.window_size
    sample_rate = args.sample_rate
    save_path = args.save_path
    window = scipy.signal.hamming
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    with open(data_path, 'rb') as handle:
        data_dict = pickle.load(handle)
    mag = np.load(mag_path)
    phase =data_dict["phase"]
    mean=data_dict["mean"]
    std = data_dict["std"]
    D = magphase_to_spect(mag,phase,mean,std)
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000,bitdepth=16,n_channels=1)
    spect_to_aud(D,hop_length,win_length,window,sample_rate,save_path+mag_path.split("/")[-1][:-4]+".wav",tfm)
