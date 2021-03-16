import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import math
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser(description='converting tsv data to txt files.')
parser.add_argument('--src-path',help = 'path to folder like test_other etc')
parser.add_argument('--dst-path',help = 'path to destn saving txts')

args = parser.parse_args()
src = args.src_path
dst = args.dst_path
folder = "_".join(src.split("/")[-2].split("-"))

def replaceUnwanted(txt):
    x = txt.split()
    song = x[0]
    txt = " ".join(x[1:])
    try:
        txt = txt.upper()
        labels = set(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," ","'"])
        unwanted = set(txt) - labels
        for i in unwanted:
            txt = txt.replace(i,"")
        with open(dst+folder+"-"+song+".txt", "w") as f:
            f.write(txt.upper())
    except AttributeError:
        pass

if __name__ == '__main__':
    speakers = os.listdir(src)
    old_src = src
    cnt = 0
    for speaker in tqdm(speakers):
        src = old_src
        chapters = os.listdir(f"{src}{speaker}/") 
        for chapter in chapters:
            src = old_src
            src = f"{src}{speaker}/{chapter}/"
            songs = os.listdir(src)
            cnt += len(songs)-1
            for txt in songs:
                if txt[-3:] =="txt":
                    src +=txt
                    with open(src,"r") as f:
                        lines = f.readlines()
                    with ProcessPoolExecutor(max_workers=48) as executor:
                         executor.map(replaceUnwanted, lines)
            print(cnt)
