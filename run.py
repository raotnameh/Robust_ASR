import argparse
import os

file = "train_horovod"
batch = 320

train_ = "data/csvs/train7_sorted.csv"
test_ = "data/csvs/dev4_sorted.csv"
gpu_rank = "0,1,2,3,4"
def run(beta,gamma,save):
    command = f"horovodrun -np 5 -H localhost:5 python {file}.py --num-workers 12 --enco-modules 2 --enco-res --train-manifest {train_} --val-manifest {test_} --cuda --rnn-type gru --hidden-layers 5 --hidden-size 1024 --epochs 50 --lr 0.001 --batch-size {batch} --checkpoint --gpu-rank {gpu_rank} --update-rule 2 --exp-name save/{save} --mw-alpha 0.0001 --mw-beta {beta} --mw-gamma {gamma} --forg-modules 2 --forg-res --num-epochs 10 --patience 10 --checkpoint"

    print('\n\n',command,'\n\n')
    os.system(command)


beta = [0.1, 0.2, 0.001, 0.5, 0.0001, 1e-05, 0.01]
gamma = [0.6, 0.1, 0.2, 0.001, 0.01, 0.4]
print("Number of experiments :",len(beta)*len(gamma))
import time 
a = time.time()
for g in sorted(gamma):
    for b in sorted(beta):
        run(f"{b}",f"{g}",f"0.0001_{b}_{g}")

print(time.time() - a)
