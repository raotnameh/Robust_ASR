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


class Aud2Spec2Aud():
    """
        audio_path: Path to the audio that you want to play with
        window_stride: Stride for the fft window (default: 0.01)
        window_size: Size for the fft window (default: 0.02)
        sample_rate: Rate at which audio is sampled to form a spectrogram (default: Hamming Window)
        normalize: Whether the mag needs to be denormalized or not (default: False) 
    """

    def __init__(self,audio_path,window_stride=0.01,window_size=0.02,sample_rate=16000,window= scipy.signal.hamming,normalize=False):
        self.audio_path = audio_path
        self.window_size = window_size
        self.window_stride = window_stride
        self.sample_rate = sample_rate
        self.n_fft = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)
        self.original_audio = self.load_audio(self.audio_path)
        self.window = window
        self.win_length = self.n_fft
        self.normalize = normalize
        self.mag, self.phase,self.mean,self.std = self.forward()

    def load_audio(self,path):
        sample_rate, sound = read(path)
        sound = sound.astype('float32') / 32767  # normalize audio
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # multiple channels, average
        sound = 2*((sound - sound.min()) / (sound.max() - sound.min())) - 1
        return sound

    def magphase_to_spect(self,new_mag):
        new_mag = new_mag*self.std + self.mean
        new_mag = np.expm1(new_mag) 
        D = np.multiply(new_mag,self.phase)
        return D

    def forward(self):
        """
            Function to convert an audio to a spectrogram.
            Input -> 
            Returns-> mag: spectrogram magnitude, it's a numpy array
                      phase: spectrogram phase, it's a numpy array
                      mean: magnitude mean
                      std: magnitude std
        """
        original = librosa.stft(self.original_audio, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.win_length, window=self.window)
        spect, phase = librosa.magphase(original)
        mag = np.log1p(spect)
        mean = 0
        std = 1
        if self.normalize:
            mean = mag.mean()
            std = mag.std()
            mag = mag - mean
            mag = mag*std
        return mag,phase,mean,std
    
    def backward(self,new_mag,save_path):
        """
            Function to convert a magnitude of a spectrogram to an audio file.
            Input -> new_mag: magnitude of the spectrogram that you need to convert (note that the shape of this magnitude should match the shape of the input spectrogram magnitude)
                     save_path: Path to save the audio file. 
            Returns -> 
        """
        if new_mag.shape != self.mag.shape:
            raise ValueError(f"Expected Input Shape for the magnitude spectogram to be: {self.mag.shape}. Got: {new_mag.shape}")
        D = self.magphase_to_spect(new_mag)
        tfm = sox.Transformer()
        tfm.convert(samplerate=16000,bitdepth=16,n_channels=1)
        reconstructed_audio = librosa.istft(D,hop_length=self.hop_length,win_length=self.win_length,window=self.window)
        tfm.build_file(input_array=reconstructed_audio,sample_rate_in=self.sample_rate,output_filepath=save_path)

# if __name__ == "__main__":
#     a2sa = Aud2Spec2Aud("/media/data_dump/hemant/rachit/dataset/downloaded_Data/librispeech/LibriSpeech/wav/test_clean-672-122797-0033.wav")
#     mag = np.load("testing.npy")
#     a2sa.backward(mag,"testing.wav")
