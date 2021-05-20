import argparse
from pydub import AudioSegment
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description='Concatenate audio files')
parser.add_argument('--src-dir',help = 'path to source directory which contains wav files')
parser.add_argument('--dst-dir',help = 'path to destination directory which will contain final wav files')
parser.add_argument('--thresh', type=int, help='time threshold above which each audio should be')
parser.add_argument('--format', type=str, help='audio file format')

args = parser.parse_args()
src = args.src_dir
dst = args.dst_dir
thresh = args.thresh
format_ = args.format

files = os.listdir(src)

def add_to_time(f):
    global time_dict
    time = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', src+f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time = float(time.stdout)
    return [f,time]

def audio_creation(f):
    if format_=="mp3":
        sound1 = AudioSegment.from_mp3(src+f[0][0])
        sound2 = AudioSegment.from_mp3(src+f[1][0])
    elif format_ == "wav":
        sound1 = AudioSegment.from_wav(src+f[0][0])
        sound2 = AudioSegment.from_wav(src+f[1][0])
    combined_sounds = sound1+sound2
    combined_sounds.export(f[2], format=format_)    


with ProcessPoolExecutor(max_workers=32) as executor:
    lst = list(tqdm((executor.map(add_to_time, files)), total=len(files)))

concat_aud = []
above_aud = []
for i in lst:
    if i[1]<thresh:
        concat_aud.append([i[0],i[1]])
    else:
        above_aud.append([i[0],i[1]])

concat_aud.sort(key=lambda x: x[1],reverse =True)
make_audio = []
print(len(concat_aud))
for i in tqdm(range(0, len(concat_aud)-1,2)):
    x = concat_aud[i]
    y = concat_aud[i+1]
    if x[1]+y[1]<thresh:
        break
    make_audio.append((x,y,dst+x[0][:-4]+"-"+y[0]))

print(len(make_audio))
with ProcessPoolExecutor(max_workers=32) as executor:
    tqdm((executor.map(audio_creation, make_audio)), total=len(make_audio))
