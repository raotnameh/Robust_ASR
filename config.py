# import json

# with open('labels.json') as label_file:
#     labels = str(''.join(json.load(label_file)))


def prepare_info(layers):
    hyperparameters = ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
    info = []

    for l in layers:
        dummy = {}
        for r,hyper in enumerate(hyperparameters): 
            dummy[hyper] = l[r]
        info.append(dummy)

    return info


def configPre(nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [1,11,2,256,0.2,1,nonlinear],
        [1,11,2,384,0.2,1,nonlinear],
    ]

    return prepare_info(layers)
    
def configE(nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [10,11,1,384,0.3,1,nonlinear],
        [10,11,1,512,0.3,1,nonlinear],
    ]

    return prepare_info(layers)


def configP(labels=29,nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [10,11,1,768,0.4,1,nonlinear],
        [1,23,1,1024,0.4,2,nonlinear],
        [1,1,1,2048,0.4,1,nonlinear],
        [1,1,1,labels,0.0,1,nonlinear],
    ]

    return prepare_info(layers)


def configF(classes=1,nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [20,11,1,384,0.3,1,nonlinear],
        [1,11,1,512*classes,0.3,1,nonlinear],
    ]

    return prepare_info(layers)

def configR(nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [1,11,2,640,0.2,1,nonlinear],
        [5,11,1,640,0.3,1,nonlinear],
        [1,11,2,512,0.2,1,nonlinear],
        [5,11,1,512,0.3,1,nonlinear],
        [1,11,1,161,0.0,1,nonlinear],
    ]

    return prepare_info(layers)

def configD(nonlinear = 1):
    layers = [
        # ['sub_blocks', 'kernel_size','stride','out_channels','dropout','dilation','nonlinear']
        [5,5,1,768,0.4,2,nonlinear],
        [1,1,1,1024,0.4,2,nonlinear],
        # [1,11,1,out,0.0,1,nonlinear],
    ]

    return prepare_info(layers)

