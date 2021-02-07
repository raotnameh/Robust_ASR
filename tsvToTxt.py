import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import math
import argparse

parser = argparse.ArgumentParser(description='converting tsv data to txt files.')
parser.add_argument('--src-dir',help = 'path to cv-corpus en folder')

args = parser.parse_args()
src = args.src_dir

def replaceUnwanted(txt,path):
    try:
        txt = txt.upper()
        labels = set(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," ","'"])
        unwanted = set(txt) - labels
        for i in unwanted:
            txt = txt.replace(i,"")
        with open(src+"txt/"+path[:-3]+"txt", "w") as f:
            f.write(txt.upper())
    except AttributeError:
        pass

l = ["dev.tsv","invalidated.tsv","other.tsv","test.tsv","train.tsv","validated.tsv"]

for i in l:
    df =pd.read_csv(src+i,sep="\t")
    print(df)
    df.dropna(subset=["path","sentence"])
    df.progress_apply(lambda x: replaceUnwanted(x["sentence"],x["path"]),axis=1)
