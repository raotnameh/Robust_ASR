learning = [0.015, 0.0015, 0.00015]
anneal = [0.9, 0.95, 0.99]

import os

for lr in learning:
    for an in anneal:
        command = f"horovodrun -np  2 -H localhost:2 python train_horovod.py --train-manifest csvs/accent_cv/train/train7_sorted.csv --val-manifest csvs/accent_cv/dev/dev4_sorted.csv --batch-size 128  --num-workers 12 --epochs 100 --cuda --lr {lr}  --gpu-rank 0,1 --exp-name save/modified_fnet/enus/asr/classifier/classifier_{lr}_{an}  --fp16 --hyper-rate 1.25 --beta 0.1 --gamma 0.1 --update-rule 1 --num-epochs 1  --patience 1000 --learning-anneal {an} --spec-augment --warmup save/modified_fnet/enus/asr/models/ckpt_final.pth --finetune-disc"
        os.system(command)


