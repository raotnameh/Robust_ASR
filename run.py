import argparse
import os

file = "train"
batch = 32#64#128

#train_ = "data/csvs/train7_sorted.csv"
train_ = "50k.csv"
test_ = "../out_test.csv"
#test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "3"
def run(save):
    command = f"python {file}.py --num-workers 16 --train-manifest {train_} --val-manifest {test_} --cuda --epochs 500 --lr 0.005 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --exp-name save/{save} --mw-alpha 0 --train-asr --fp16"

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr_big_5")

