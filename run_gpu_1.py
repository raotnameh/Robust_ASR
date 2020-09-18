import argparse
import os

def run(disc_modules):
    command = '''python train.py \
    --enco-modules 1 --enco-res \
    --disc-modules {} --disc-res \
    --train-manifest data/csvs/train_sorted.csv --val-manifest data/csvs/dev_sorted.csv \
    --cuda --gpu-rank 2 \
    --rnn-type gru --hidden-layers 5 --opt-level O1 --loss-scale 1.0 --hidden-size 1024 \
    --epochs 50 --lr 0.001 --batch-size 32 --update-rule 1 \
    --exp-name /media/data_dump/asr/Robust_ASR/save/train_Asr_experimentation/experiment_2/exp_disc_modules_{} \
    --mw-alpha 0.1'''.format(disc_modules, disc_modules)

    os.system(command)

disc_modules_values = [1, 3]
for disc_modules in disc_modules_values:
    run(disc_modules)
