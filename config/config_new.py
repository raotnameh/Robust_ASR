# import json

# with open('labels.json') as label_file:
#     labels = str(''.join(json.load(label_file)))


def prepare_info(layers):
    hyperparameters = ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','batch_norm']
    info = []

    for l in layers:
        dummy = {}
        for r,hyper in enumerate(hyperparameters): 
            dummy[hyper] = l[r]
        info.append(dummy)

    return info

def configPre():
    layers = [
        [1,11,2,256,0.2,1,True],
    ]

    return prepare_info(layers)

    
def configE():
    layers = [
        [5,11,1,256,0.2,1,True],
        [5,11,1,256,0.2,1,True],
        [5,11,1,256,0.2,1,True],
        [5,11,1,256,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [1,13,1,512,0.2,1,True],
        [1,15,1,684,0.4,1,True],
        [1,17,1,684,0.4,1,True],
        [1,19,1,684,0.4,1,True],
        [1,21,1,740,0.4,1,True],
        [1,23,1,740,0.4,1,True],
        [1,25,1,740,0.4,1,True],
        ]

    return prepare_info(layers)

def configD():
    layers = [
        [1,11,1,512,0.2,1,True],
        [5,19,1,256,0.2,1,True],
        [1,11,2,161,0.0,1,False],
    ]

    return prepare_info(layers)

def configP(labels=29):
    layers = [
        [1,29,1,896,0.4,2,True],
        [1,1,1,1024,0.4,1,True],
        [1,1,1,labels,0.0,1,False],
    ]

    return prepare_info(layers)

def configFN():
    layers = [
        # [1,11,1,512,0.3,1,True],
        [1,1,1,640,0.0,1,False],
    ]

    return prepare_info(layers)
 

def configDM():
    layers = [
        [1,29,1,1024,0.4,2,True],
    ]

    return prepare_info(layers)

# f"python {file}.py --train-manifest {train_} --val-manifest {test_} --batch-size {batch_size} --epochs 500 --cuda --lr 0.0015 --checkpoint --gpu-rank {gpu_rank} --exp-name save/{save} --alpha 0.0001 --beta 0.2 --gamma 0.6 --fp16"

# def configE():
#     layers = [
#         [5,11,1,256,0.2,1,True],
#         [5,11,1,256,0.2,1,True],
#         # [1,33,1,384,0.2,1,True],
#         [5,11,1,384,0.2,1,True],
#         [5,11,1,384,0.2,1,True],
#         # [1,33,1,384,0.2,1,True],
#         [5,11,1,512,0.2,1,True],
#         [5,11,1,512,0.2,1,True],
#         # [1,33,1,512,0.2,1,True],
#         ]

#     return prepare_info(layers)




# python test.py --test-manifest csvs/accent_cv/test/test2.csv --model-path /home/hemant/updated_v2/save/modified_fnet/enus/asr/models/ckpt_599.pth --gpu-rank 3 --cuda --batch-size 64 --decoder beam --lm-path lm/enus.binary --beam-width 256 --alpha 1.5 --beta 4.714