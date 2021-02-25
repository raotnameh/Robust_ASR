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
parser.add_argument('-ap','--audio-path',type=str,help='Path to the audio')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('-sp','--save-path',type=str,help='Path to save the spectogram/audio')
args = parser.parse_args()

#Example Audio Path: /media/data_dump/hemant/rachit/dataset/downloaded_Data/cv_v1_data/train7/wav/sample-101535.wav

def load_audio(path):
    sample_rate, sound = read(path)
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    sound = 2*((sound - sound.min()) / (sound.max() - sound.min())) - 1
    return sound

def aud_to_spect(original_audio,n_fft,hop_length,win_length,window,save_path):
    original = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length,win_length=win_length, window=window)
    spect, phase = librosa.magphase(original)
    spect = np.log1p(spect)
    np.save(save_path,phase)
    return spect,phase

if __name__ == "__main__":
    audio_path = args.audio_path
    window_stride = args.window_stride
    window_size = args.window_size
    sample_rate = args.sample_rate
    save_path = args.save_path
    window = scipy.signal.hamming
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    original_audio = load_audio(audio_path)
    aud_to_spect(original_audio,n_fft,hop_length,win_length,window,save_path+audio_path.split("/")[-1][:-4])