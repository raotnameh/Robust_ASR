# import json

# with open('labels.json') as label_file:
#     labels = str(''.join(json.load(label_file)))


def prepare_info(layers):
    hyperparameters = ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
    info = []

    for l in layers:
        dummy = {}
        for r,hyper in enumerate(hyperparameters): 
            dummy[hyper] = l[r]
        info.append(dummy)

    return info

def configE():
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
        [1,11,2,256,0.2,1],
        [1,11,2,512,0.2,1],
        [10,11,1,512,0.3,1],
        [10,11,1,738,0.3,1],
    ]

    return prepare_info(layers)

def configF():
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
        [1,11,2,256,0.2,1],
        [1,11,2,512,0.2,1],
        [20,11,1,512,0.3,1],
        [1,11,1,738*1,0.3,1],
    ]

    return prepare_info(layers)

def configR():
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
        [1,11,2,738,0.2,1],
        [5,11,1,512,0.3,1],
        [1,11,2,256,0.2,1],
        [1,11,1,1,0.0,1],
    ]

    return prepare_info(layers)

def configD(out=2):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
        [5,11,1,738,0.4,1],
        [1,1,1,738,0.4,2],
        [2,1,1,1024,0.4,1],
        [1,11,1,out,0.0,1],
    ]

    return prepare_info(layers)

def configP(labels=29):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation']
        [10,11,1,738,0.4,1],
        [1,1,1,738,0.4,2],
        [1,1,1,1024,0.4,1],
        [1,1,1,labels,0.0,1],
    ]

    return prepare_info(layers)
