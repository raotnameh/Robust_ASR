import argparse
import os

file = "train"
batch = 64

#train_ = "data/csvs/train7_sorted.csv"
train_ = "csvs/train_sorted_EN_US.csv"
test_ = "csvs/dev_sorted_EN_US.csv"
#test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "0"
def run(save):
    command = f"python {file}.py --train-manifest {train_} --val-manifest {test_} --batch-size {batch} --epochs 500 --cuda --lr 0.0015 --checkpoint --gpu-rank {gpu_rank} --exp-name save/{save} --alpha 0.0001 --beta 0.1 --gamma 0.01 --fp16 --update-rule 5"

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr_0.0001_0.1_0.01")

