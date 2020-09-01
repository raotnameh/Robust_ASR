import glob
from tqdm.auto import tqdm

import subprocess

def get_length(input_video):
    with open(input_video,"r") as f:
        return f.read()

files = glob.glob("t.csv")
print(files)

def csv_(dummy):
    dummy = sorted(dummy, key = lambda x: len(x))
    a = ''
    for i in tqdm(dummy):
        a+=i
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
