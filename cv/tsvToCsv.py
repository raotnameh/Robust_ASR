import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time  
import concurrent.futures
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Create Respective CSVs with given labels.')
parser.add_argument('--src-dir',help = 'path to cv corpus lang folder')
#parser.add_argument('--csv-dir',help = 'path to final csv directory')
parser.add_argument('--label',help='label to create csv for', type=str ,default=None)

args = parser.parse_args()
src = args.src_dir
if src[-1]!="/":
    src += "/"
dst = src + "csvs/"
if not os.path.exists(dst):
    os.makedirs(dst)
label = args.label


print(f"\n[CSV CREATION] \t Creating csvs in directory: {dst}\n")

lst = ['dev.tsv','train.tsv',"test.tsv",'invalidated.tsv','validated.tsv','other.tsv']
#lst = ['invalidated.tsv','validated.tsv','other.tsv','train.tsv']
#lst = ["test.tsv"]
l = []
for j in lst:
    df = pd.read_csv(src+j,'\t')
    if label:
        df1 = df[~df[label].isnull()]
    else:
        df1 = df
    prePath = src
    print(j)
    for i in tqdm(range(len(df1))):
        x = df1.iloc[i]
        wav = prePath+'wav/'+x['path'][:-3]+'wav' 
        txt = prePath+'txt/'+x['path'][:-3]+'txt'
        if label:
            a = x[label]
            l.append([wav,txt,a])
        else:
            l.append([wav,txt])

    finDf = pd.DataFrame(l)
    finDf.drop_duplicates(inplace=True)
    finDf.to_csv(dst+j[:-3]+"csv",index= False, header = False)
