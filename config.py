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
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','batchnorm']
        [1,11,2,256,0.2,1,True],
    ]

    return prepare_info(layers)
    
def configE():
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','batchnorm']
        [5,11,1,256,0.2,1,True],
        [5,11,1,256,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [5,11,1,384,0.2,1,True],
        [5,11,1,512,0.2,1,True],
        [5,11,1,512,0.2,1,True],
        [5,11,1,640,0.3,1,True],
        [5,11,1,640,0.3,1,True],
        [5,11,1,768,0.3,1,True],
        [5,11,1,768,0.3,1,True],
    ]

    return prepare_info(layers)


def configDecoder(labels=29):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','batchnorm']
        [1,29,1,896,0.4,2,True],
        [1,1,1,1024,0.4,1,True],
        [1,1,1,161,0.0,1,False],
    ]

    return prepare_info(layers)
