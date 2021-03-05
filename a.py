import argparse
import os

file = "train"
batch = 112

#train_ = "data/csvs/train7_sorted.csv"
train_ = "csvs/train_sorted_EN_US.csv"
test_ = "csvs/dev_sorted_EN_US.csv"
#test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "0"
def run(save):
    command = f"python {file}.py --train-manifest {train_} --val-manifest {test_} --batch-size {batch} --num-workers 16 --epochs 40 --cuda --lr 0.0015  --gpu-rank {gpu_rank} --exp-name enus_40/{save} --alpha 0.0001 --beta 0.0 --gamma 0.00 --fp16 --update-rule 2 --train-asr"

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr_asr")

