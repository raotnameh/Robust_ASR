import argparse
import os

file = "train_fp16"
batch = 32

train_ = "data/csvs/train7_sorted.csv"
test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = 1
def run(alpha,save):
    command = f"python {file}.py --opt-level O2 --enco-modules 2 --enco-res --train-manifest {train_} --val-manifest {test_} --cuda --rnn-type gru --hidden-layers 5 --hidden-size 1024 --epochs 50 --lr 0.0001 --patience 10 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --update-rule 2 --exp-name save/{save} --mw-alpha {alpha} --num-epochs 50 --train-asr"

    print('\n\n',command,'\n\n')
    os.system(command)

alpha =  [i/100 for i in range(0,100,10)][:7]
print(alpha)
import time 
a = time.time()

for i in alpha:
    run(i,f"alpha_{i}")
    break

print(time.time() - a)
