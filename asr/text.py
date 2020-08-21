import glob, os
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(description='to create the combined text file using a csv for training the LM')
parser.add_argument('--path',
                    help='path to the csv')
parser.add_argument('--save',
                    help='path to save the txt')
args = parser.parse_args()

with open(args.path ,"r") as f:
    out = f.readlines()
    
txt = [i.split(',')[-1].strip() for i in out]

a = ''
for i in tqdm(txt):
    with open(i,"r") as f:
        a+= f.read().upper() + '\n'
        
with open(args.save + ".txt","w") as f:
        f.write(a)
