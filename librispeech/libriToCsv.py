import os
import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(description='converting tsv data to txt files.')
parser.add_argument('--wav-path',help = 'path to folder containing final wav files')
parser.add_argument('--txt-path',help = 'path to folder containing final txt files')
parser.add_argument('--csv-path',help = 'path to folder where final csvs need to be placed')

args = parser.parse_args()
preWav = args.wav_path
preTxt = args.txt_path
preCsv = args.csv_path

#preWav = "/media/data_dump/hemant/rachit/dataset/downloaded_Data/librispeech/LibriSpeech/wav/"
#preTxt = "/media/data_dump/hemant/rachit/dataset/downloaded_Data/librispeech/LibriSpeech/txt/"
#preCsv = "/media/data_dump/hemant/rachit/dataset/downloaded_Data/librispeech/LibriSpeech/csv/individual_segregated_speaker/"
files = os.listdir(preWav)

d = {}

for file in files:
    split_info = file.split("-")
    csv_type = split_info[0]
    wav = preWav + file
    txt = preTxt + file[:-3]+"txt"
    speaker = split_info[1]
    try:
        d[csv_type].append([wav,txt,speaker])
    except:
        d[csv_type]= [[wav,txt,speaker]]

#print(len(d["train_clean_360"]))
for i in d:
#    my_df = pd.DataFrame(d[i])
   # print(d[i][:5])
   # break
    #print(d[i][:100])
    #break
    with open(preCsv+i+".csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(d[i])
