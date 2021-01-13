# import json

# with open('labels.json') as label_file:
#     labels = str(''.join(json.load(label_file)))

def config(labels=29, sub_blocks=5):
    info = [
        {
            'sub_blocks': 1,
            'kernel_size': 11,
            'stride': 2,
            'out_channels': 1024,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 11,
            'stride': 1,
            'out_channels': 512,
            'dropout': 0.2,
            'dilation': 1,
        }

    ]

    return info


def configD(labels=29, sub_blocks=5):
    info = [
        {
            'sub_blocks': 1,
            'kernel_size': 11,
            'stride': 2,
            'out_channels': 512,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 11,
            'stride': 1,
            'out_channels': 1024,
            'dropout': 0.2,
            'dilation': 1,
        }
    ]

    return info