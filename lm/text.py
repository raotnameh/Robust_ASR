import glob, os
from tqdm.auto import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor


parser = argparse.ArgumentParser(description='to create the combined text file using a csv for training the LM')
parser.add_argument('--path',
                    help='path to the csv')
parser.add_argument('--save',
                    help='path to save the txt')
args = parser.parse_args()

def get_txt(i):
    with open(i,"r") as f:
        return f.read().upper()
        
with open(args.path ,"r") as f:
    out = f.readlines()
    
txt = [i.split(',')[-2].strip() for i in out]


def txt_save(dummy):
    a = ''
    for i in dummy:
        a+=i + '\n'
    return a

def run(get_txt, txt):
    with ProcessPoolExecutor(max_workers=64) as executor:
        results = list(tqdm((executor.map(get_txt, txt)), total=len(txt)))
    return results

print("starting the processes")
temp = set(run(get_txt, txt))
print("saving the files")
with open(args.save + ".txt","w") as f:
        f.write(txt_save(temp))
