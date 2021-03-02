import argparse
import os

file = "train"
batch = 32#64

#train_ = "data/csvs/train7_sorted.csv"
train_ = "csvs/train_sorted_EN_US.csv"
test_ = "csvs/dev_sorted_EN_US.csv"
#test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "1"
def run(save):
    command = f"python {file}.py --train-manifest {train_} --val-manifest {test_} --batch-size {batch} --epochs 500 --cuda --lr 0.0015 --checkpoint --gpu-rank {gpu_rank} --exp-name save/{save} --alpha 0.0001 --beta 0.01 --gamma 0.001 --fp16 "

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr_0.0001_0.01_0.001")

