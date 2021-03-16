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
parser.add_argument('--src-path',help = 'path to source directory which contains mp3 files')
parser.add_argument('--dst-path',help = 'path to destination directory which will contain final wav files')

args = parser.parse_args()
src = args.src_path
dst = args.dst_path

folder = "_".join(src.split("/")[-2].split("-"))

def correctPaths(x):
    x = x.split('/')
    x[0] = '/media/data_dump/hemant/microsoft/data_2'
    return '/'.join(x)

def normaliseWAV(song):
    if song[-3:] == "txt":
        return
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000,bitdepth=16,n_channels=1)
    tfm.build(src+song,dst+folder+"-"+song[:-4]+"wav")

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

#def librispeech

if __name__ == '__main__':
    speakers = os.listdir(src)
    old_src = src
    cnt = 0
    for speaker in tqdm(speakers):
        src = old_src
        chapters = os.listdir(f"{src}{speaker}/") 
        for chapter in chapters:
             src = old_src
             src = f"{src}{speaker}/{chapter}/"
             songs = os.listdir(src)
             cnt += len(songs)-1
             with ProcessPoolExecutor(max_workers=48) as executor:
                 executor.map(normaliseWAV, songs)
    print(cnt)
