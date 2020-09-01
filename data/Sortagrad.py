import glob
from tqdm.auto import tqdm

import subprocess

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video.split(',')[0]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [input_video,float(result.stdout)]

files = glob.glob("t.csv")
print(files)
max_duration = 15.0
min_duration = 0.5

def csv_(dummy):
    dummy = [i for i in tqdm(dummy) if i[-1] <= max_duration and i[-1] >= min_duration]
    dummy = sorted(dummy, key = lambda x: x[-1])
    a = ''
    for i in tqdm(dummy):
        a+=i[0]
    with open('t_sorted.csv', "w") as f:
        f.write(a)


import pandas, os
from concurrent.futures import ProcessPoolExecutor
for file in files:
    with open(file, "r") as f:
        csv = f.readlines()

    def run(get_length, wav):
            with ProcessPoolExecutor(max_workers=48) as executor:
                results = list(tqdm((executor.map(get_length, wav)), total=len(wav)))
            return results
    print("starting the processes")
    temp = run(get_length, csv)
    print("saving the files")
    save_ = csv_(temp)
    duration = [i[-1] for i in temp]

    print(f"{file} info: \nmin = {min(duration)} seconds\nmax = {max(duration)} seconds\ntotal = {round(sum(duration)/3600,3)} Hrs and {round(sum(duration)/60,3)} Mins")
    print(f"percentage of files less than {max_duration} seconds and greater than {min_duration} seconds: , {len([i for i in duration if i <max_duration])/len(duration)}")
    print(f"percentage of files less than {min_duration} seconds: , {len([i for i in duration if i <min_duration])/len(duration)}")
    print(f"Number files removed from the csv are: {len([i for i in duration if i >max_duration]) + len([i for i in duration if i <min_duration]) }")
