import pandas as pd
import os
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import argparse
parser = argparse.ArgumentParser(description='User Label Mapping')

parser.add_argument('--csv-path',type=str, help='Input CSV path')
parser.add_argument('--label',type=str, help='User defined label')
parser.add_argument('--save-path',type=str, help='Path to save csv')

args = parser.parse_args()
csv_path = args.csv_path
label = args.label
save_path = args.save_path

def speaker_map(x,label):
    x[2] = label
    return x

df = pd.read_csv(csv_path,header=None)
df[2] = np.nan
df = df.progress_apply(lambda x: speaker_map(x,label),axis=1)
df.to_csv(save_path,header=False,index=False)

