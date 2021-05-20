import argparse
from pydub import AudioSegment
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import subprocess
from shutil import copyfile

parser = argparse.ArgumentParser(description='Concatenate audio files')
parser.add_argument('--src-dir',help = 'path to source directory which contains wav files')
parser.add_argument('--dst-dir',help = 'path to destination directory which will contain final wav files')
parser.add_argument('--thresh', type=int, help='time threshold above which each audio should be')
parser.add_argument('--format', type=str, help='audio file format')
parser.add_argument('--num-workers', type=int, help='number of workers')

args = parser.parse_args()
src = args.src_dir
dst = args.dst_dir
thresh = args.thresh
format_ = args.format
num_workers = args.num_workers

files = os.listdir(src)

def add_to_time(f):
    time = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', src+f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time = float(time.stdout)
    return [f,time]

def audio_creation(f):
    lst = []
    if format_=="mp3":
        for i in f[0]:
            sound = AudioSegment.from_mp3(src+i[0])
            lst.append(sound)
    elif format_ == "wav":
        for i in f[0]:
            sound = AudioSegment.from_mp3(src+i[0])
            lst.append(sound)

    combined_sounds = lst[0]
    for i in range(1, len(lst)):
        combined_sounds = combined_sounds + lst[i]
    combined_sounds.export(f[1], format=format_)

def copy_file(f):
    copyfile(src+f[0], dst+f[0])

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    lst = list(tqdm((executor.map(add_to_time, files)), total=len(files)))

print("total audios: ",len(lst))
concat_aud = []
above_aud = []
for i in lst:
    if i[1]<thresh:
        concat_aud.append([i[0],i[1]])
    else:
        above_aud.append([i[0],i[1]])

print("Length of Audios above threshold: ",len(above_aud), " Length of audios below threshold:  ",len(concat_aud))

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    tqdm((executor.map(copy_file, above_aud)), total=len(above_aud))

print("done saving")
concat_aud.sort(key=lambda x: x[1],reverse =True)
make_audio = []
l = 0
r =len(concat_aud)-1
while (l<r):
    lst = [concat_aud[l],concat_aud[r]]
    summ = lst[0][1]+lst[1][1]
    strng = lst[0][0][:-4] + "-"+lst[0][0]
    while summ<thresh and l<r-1:
        r -=1
        lst.append(concat_aud[r])
        summ += lst[-1][1]
        strng = strng[:-4]+"-" +lst[-1][0]
    if summ <thresh and l>=r-1:
        break
    make_audio.append((lst,dst+strng))
    l+=1
    r-=1
print("Concatenated Audio Files (number): ",len(make_audio))
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    tqdm((executor.map(audio_creation, make_audio)), total=len(make_audio))
