import argparse
import os

file = "train_fp16"
batch = 32

train_ = "data/csvs/train7_sorted.csv"
test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = 1
def run(alpha,save):
    command = f"python {file}.py --opt-level O2 --enco-modules 2 --enco-res --train-manifest {train_} --val-manifest {test_} --cuda --rnn-type gru --hidden-layers 5 --hidden-size 1024 --epochs 50 --lr 0.0001 --patience 10 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --update-rule 2 --exp-name save/{save} --mw-alpha {alpha} --num-epochs 50 --train-asr --dummy"

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

python train_horovod.py --train-manifest data/csvs/train.csv --val-manifest data/csvs/dev.csv --cuda --version V1.0
--enco-modules 2 --enco-res
--forg-modules 2 --forg-res
--disc-modules 2 --disc-res --update-rule 2
--rnn-type gru --hidden-layers 5 --hidden-size 1024
--epochs 50 --lr 0.001 --patience 10 --batch-size 32
--checkpoint --checkpoint-per-batch 1000 --continue-from save/models/ckpt_final.pth --finetune
--gpu-rank 0,1,2,4 --learning-anneal 0.95
--exp-name save/ --num-epochs 50 --seed 123456
--mw-alpha 0.00001 --mw-beta 0.6 --mw-gamma 0.2
--train-asr --dummy --fp16

