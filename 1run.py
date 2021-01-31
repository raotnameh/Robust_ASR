import argparse
import os

file = "train_horovod"
batch = 32 

train_ = "data/csvs/train7_sorted.csv"
test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "1"
def run(save):
    command = f"horovodrun -np 1 -H localhost:1 python {file}.py --num-workers 8 --enco-modules 2 --enco-res --train-manifest {train_} --val-manifest {test_} --cuda --epochs 50 --lr 0.0001 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --exp-name save_1/{save} --mw-alpha 0 --train-asr --dummy" 

    print('\n\n',command,'\n\n')
    os.system(command)


run(f"asr")

