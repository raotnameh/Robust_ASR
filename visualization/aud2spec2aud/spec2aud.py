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

parser = argparse.ArgumentParser()
parser.add_argument('--mag-path',type=str,help='Path to the spectogram magnitude')
parser.add_argument('--phase-path',type=str,help='Path to the spectogram magnitude')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('-sp','--save-path',type=str,help='Path to save the spectogram/audio')
args = parser.parse_args()


# Example Mag Path: /media/data_dump/hemant/rachit/scripts/Robust_ASR/spectogram_audio/spectogram/sample-101535.npy
# Example Phase Path: /media/data_dump/hemant/rachit/scripts/Robust_ASR/spectogram_audio/phases/sample-101535.npy

def magphase_to_spect(mag,phase):
    mag = mag[:phase.shape[0],:phase.shape[1]]
    mag = np.expm1(mag) 
    D = np.multiply(mag,phase)
    return D

def spect_to_aud(D,hop_length,win_length,window,sample_rate,save_path):
    reconstructed_audio = librosa.istft(D,hop_length=hop_length,win_length=win_length,window=window)
    write(save_path,sample_rate,reconstructed_audio)

if __name__ == "__main__":
    mag_path = args.mag_path
    phase_path = args.phase_path
    window_stride = args.window_stride
    window_size = args.window_size
    sample_rate = args.sample_rate
    save_path = args.save_path
    window = scipy.signal.hamming
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    mag = np.load(mag_path)
    phase = np.load(phase_path)
    D = magphase_to_spect(mag,phase)
    spect_to_aud(D,hop_length,win_length,window,sample_rate,save_path+mag_path.split("/")[-1][:-4]+".wav")