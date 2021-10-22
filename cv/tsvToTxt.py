import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import math
import argparse
import os
import json

parser = argparse.ArgumentParser(description='converting tsv data to txt files.')
parser.add_argument('--src-dir',help = 'path to cv-corpus en folder')
parser.add_argument('--label-path',help = 'path to json file which contains labels')

args = parser.parse_args()
src = args.src_dir
if src[-1] !="/":
    src +="/"
dst = src + "txt/"
if not os.path.exists(dst):
    os.makedirs(dst)
label_file = args.label_path

def replaceUnwanted(txt,path,labels):
    try:
        txt = txt.upper()
        unwanted = set(txt) - labels
        for i in unwanted:
            txt = txt.replace(i,"")
        with open(src+"txt/"+path[:-3]+"txt", "w") as f:
            f.write(txt.upper())
    except AttributeError:
        pass

print(f"\n[TXT FILE CREATION] \t Creating txt files in directory: {dst}\n")

l = ["dev.tsv","invalidated.tsv","other.tsv","test.tsv","train.tsv","validated.tsv"]
with open(label_file) as lf:
    labels = set(json.load(lf))

for i in l:
    df =pd.read_csv(src+i,sep="\t")
    print(src+i)
    df.dropna(subset=["path","sentence"])
    df.progress_apply(lambda x: replaceUnwanted(x["sentence"],x["path"],labels),axis=1)
