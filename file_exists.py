import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Check if file exists or not.')
parser.add_argument('--src-path',help = 'path to CSV')

args = parser.parse_args()
src = args.src_path

l = []
def check(wav,txt,label,changeCSV,l):
    if not (os.path.exists(wav) and os.path.exists(txt)):
        changeCSV = True
        return
    l.append([wav,txt,label])

df = pd.read_csv(src,header=None)
change = False
df.progress_apply(lambda x: check(x[0],x[1],x[2],change,l),axis=1)
print(len(l),df.shape)
if (len(l)!=df.shape[0]):
    with  open(src,"w") as f:
        writer = csv.writer(f)
        writer.writerows(l)
