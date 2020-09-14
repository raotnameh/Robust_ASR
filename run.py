import argparse
import os

def run(alphaValue,cnt):
    command = "python train.py --enco-modules 1 --enco-res --train-manifest data/csvs/train_sorted.csv --val-manifest data/csvs/dev_sorted.csv --cuda --rnn-type gru --hidden-layers 5 --opt-level O1 --loss-scale 1.0 --hidden-size 1024 --epochs 50 --lr 0.001 --batch-size 32 --gpu-rank 1 --update-rule 1 --exp-name /media/data_dump/hemant/rachit/Robust_ASR/save/train_Asr_experimentation/experiment_2/exp_"+str(cnt) +" --mw-alpha " + str(alphaValue) + " --train-asr"

    os.system(command)

#alphaValues = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#alphaValues = [0.1]
alphaValues = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
for value in alphaValues:
    run(value, value)
