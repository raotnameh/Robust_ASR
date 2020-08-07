import wave, struct, math, random
from pydub import AudioSegment
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument("fileAddr",help = "The file for which downsampling and bitrate adjustment needs to be done",type= str)
args = parser.parse_args()
f = args.fileAddr
path = os.getcwd()+ f
# print("ye hai bhai: "+ os.getcwd())
song = AudioSegment.from_wav(path).set_frame_rate(16000)

song.export(os.getcwd()+ f, format='wav', bitrate='256k')

# print("Done!")