# import json

# with open('labels.json') as label_file:
#     labels = str(''.join(json.load(label_file)))

def config(labels=29, sub_blocks=5):
    info = [
        {
            'sub_blocks': 1,
            'kernel_size': 11,
            'stride': 2,
            'out_channels': 256,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 11,
            'stride': 1,
            'out_channels': 256,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 13,
            'stride': 1,
            'out_channels': 384,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 17,
            'stride': 1,
            'out_channels': 512,
            'dropout': 0.2,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 21,
            'stride': 1,
            'out_channels': 640,
            'dropout': 0.3,
            'dilation': 1,
        },
        {
            'sub_blocks': sub_blocks,
            'kernel_size': 25,
            'stride': 1,
            'out_channels': 768,
            'dropout': 0.3,
            'dilation': 1,
        },
        {
            'sub_blocks': 1,
            'kernel_size': 29,
            'stride': 1,
            'out_channels': 896,
            'dropout': 0.4,
            'dilation': 2,
        },
        {
            'sub_blocks': 1,
            'kernel_size': 1,
            'stride': 1,
            'out_channels': 1024,
            'dropout': 0.4,
            'dilation': 1,
        },
        {
            'sub_blocks': 1,
            'kernel_size': 1,
            'stride': 1,
            'out_channels': len(labels),
            'dropout': 0.4,
            'dilation': 1,
        }

    ]

    return info