import argparse
import os

def run(alphaValue,cnt):
    command = "python train.py --enco-modules 2 --dummy --enco-res --train-manifest data/csvs/train_sorted_EN_US.csv --val-manifest data/csvs/dev_sorted.csv --cuda --rnn-type gru --hidden-layers 3 --momentum 0.94 --opt-level O1 --loss-scale 1.0 --hidden-size 1024 --epochs 100 --lr 0.0001 --batch-size 24 --gpu-rank 1 --dummy  --checkpoint --update-rule 4 --mw-beta 0.1 --mw-gamma 0.1 --disc-modules 1 --disc-res --exp-name /media/data_dump/hemant/rachit/Robust_ASR/save/train_Asr_experimentation/experiment_1/exp_"+str(cnt) +" --mw-alpha " + str(alphaValue)

    os.system(command)

alphaValues = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
#alphaValues = [0.1]
cnt = 1
for value in alphaValues:
    run(value, cnt)
    cnt+=1
