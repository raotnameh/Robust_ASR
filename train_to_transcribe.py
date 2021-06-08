import torch
import argparse
import os

parser = argparse.ArgumentParser(description='test to transcribe convert the model')
parser.add_argument('--path', type=str, help='enter the full path to the model checkpoints')
parser.add_argument('--save', type=str, help='enter the path and file name to same, example: save/ckpt_final.pth')
parser.add_argument('--half', default='store_true', help='save the weights in 16 bytes')

args = parser.parse_args()

package = torch.load(args.path, map_location='cpu')
models = package['models']
del models['decoder']
for i in models:
    if args.half: models[i] = models[i][0].half()
    else: models[i] = models[i][0].half()
print(models.keys())
updated_package = {'models': models, 'audio_conf': package['audio_conf'], 'labels': package['labels']}
torch.save(updated_package,args.save)