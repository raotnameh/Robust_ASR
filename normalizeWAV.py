import xml.etree.ElementTree as ET
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment
from tqdm import tqdm
tqdm.pandas()
import sox
from soundfile import SoundFile
import csv
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Normalize WAV files.')
parser.add_argument('--source-path',help = 'path to source directory which contains mp3 files')
parser.add_argument('--dest-path',help = 'path to destination directory which will contain final wav files')

args = parser.parse_args()
src = args.source_path
dst = args.dest_path

def correctPaths(x):
    x = x.split('/')
    x[0] = '/media/data_dump/hemant/microsoft/data_2'
    return '/'.join(x)

def normaliseWAV(song):
#    print(src+song,dst+song[:-3]+"wav")
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000,bitdepth=16,n_channels=1)
    tfm.build(src+song,dst+song[:-3]+"wav")

def splitWav(line):
    line = line.strip().split()
    sourcePath = line[1]
    destinationPath = line[0]
    start = float(line[2])
    end = float(line[3])
    sourcePath = d[sourcePath]
    destinationPath = '/media/data_dump/hemant/microsoft/data_2/Train_Dev/dataset/wav/'+destinationPath+'.wav'
    song = AudioSegment.from_wav(sourcePath)
    start *=1000
    end *=1000
    sliced = song[start:end]
    sliced.export(destinationPath, format="wav")

def checkAudio(wavFile):
    myfile = SoundFile(''+wavFile,"r")
    sRate = myfile.samplerate
    channels = myfile.channels
    bitdepth = myfile.subtype
    if sRate!= 16000 or channels!=1 or bitdepth != "PCM_16":
        print(wavFile)
        print(str(myfile.samplerate)+" "+ str(myfile.channels)+ " "+str(myfile.subtype))
    myfile.close()

if __name__ == '__main__':
    #print(src,dst)
    songs = os.listdir(src)
    with ProcessPoolExecutor(max_workers=24) as executor:
        list(tqdm((executor.map(normaliseWAV, songs)), total=len(songs)))

