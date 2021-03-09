import argparse
import os

file = "train"
batch = 64

train_ = "csvs/train_sorted_EN_US.csv"
test_ = "csvs/dev_sorted_EN_US.csv"
gpu_rank = "0"

def run(beta,gamma,save):
    command = f"python {file}.py --train-manifest {train_} --val-manifest {test_} --batch-size {batch} --epochs 20 --cuda --lr 0.0015 --checkpoint --gpu-rank {gpu_rank} --exp-name enus_20/{save} --alpha 0.0001 --beta {beta} --gamma {gamma} --fp16 --update-rule 2"
    

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
