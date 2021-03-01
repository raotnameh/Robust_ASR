import argparse
import os

file = "train"
batch = 128

#train_ = "data/csvs/train7_sorted.csv"
train_ = "out.csv"
test_ = "outd.csv"
#test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "0"
def run(save):
    command = f"python {file}.py --num-workers 16 --train-manifest {train_} --val-manifest {test_} --cuda --epochs 500 --lr 0.0015 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --exp-name save/{save} --alpha 0 --train-asr --fp16"

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr_v2")

