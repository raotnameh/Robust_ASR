import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser(description='converting tsv data to txt files.')
parser.add_argument('--speakertxt-path',help = 'path to file speaker.txt in Librispeech data')
parser.add_argument('--csvs-path',help = 'path to folder where final csvs are placed')

args = parser.parse_args()

preTxt = args.speakertxt_path
preCsv = args.csvs_path


f = open(preTxt,"r")

lines = f.readlines()
f.close()
prePath = preCsv
files = os.listdir(prePath)
id_gender = {}
for line in lines:
    t = line.split("|")
    id_gender[int(t[0].strip())] = t[1].strip()

for f in files:
    df = pd.read_csv(prePath+f,header=None)
#    print(df)
    df[2] = df[2].map(id_gender)
    df.to_csv(prePath+f,header=False,index=False)
#    print(df)
#    break
