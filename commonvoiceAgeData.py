import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time  
import concurrent.futures
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Create Respective CSVs with given labels.')
parser.add_argument('--src-path',help = 'path to cv corpus en folder')
parser.add_argument('--csv-path',help = 'path to final csv to be created')
parser.add_argument('--label',help='label to create csv for')

args = parser.parse_args()
src = args.src_path
dst = args.csv_path
label = args.label

lst = ['dev.tsv']
#lst = ['invalidated.tsv','validated.tsv','other.tsv','train.tsv']
#lst = ["test.tsv"]
l = []
for i in lst:
    df = pd.read_csv(src+i,'\t')
    df1 = df[~df[label].isnull()]
    prePath = src
    print(i)
    for i in tqdm(range(len(df1))):
        x = df1.iloc[i]
        wav = prePath+'wav/'+x['path'][:-3]+'wav' 
        txt = prePath+'txt/'+x['path'][:-3]+'txt'
        age = x['age']
        l.append([wav,txt,age])

finDf = pd.DataFrame(l)
finDf.drop_duplicates(inplace=True)
finDf.to_csv(dst,index= False, header = False)
