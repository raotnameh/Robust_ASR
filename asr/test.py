import argparse
import os
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from opts import add_decoder_args, add_inference_args
from utils import load_model, shorten_target
import numpy as np

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=32, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default="test", help="Saves output of model from test to this file_path")
parser.add_argument('--gpu', dest='gpu', type=str, help='GPU to be used', required=True)
parser = add_decoder_args(parser)

def evaluate_acc(test_loader,device,model,save_output=None, verbose=False, half=False):
    model.eval()
    conv_params = model.conv_params
    correct,total = 0,0
    for _, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes, time_durs = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        new_targets = []

        if half:
            inputs = inputs.half()

        out, output_sizes = model(inputs, input_sizes)
        new_timesteps = out.size(1)

        for idx,size in enumerate(target_sizes.data.cpu().numpy()):
            new_size = size.item()
            for key in conv_params:
                params = conv_params[key]
                new_size = int((new_size + 2*params['padding'] - params['time_kernel'])/params['stride'] + 1)
            prev = 0
            time_dur = time_durs.data.cpu().numpy()[idx]
            new_target = targets.data.numpy()[prev:size.item()]
            new_target = shorten_target(new_target,new_size,time_dur)
            #new_target += [0]*(new_timesteps-len(new_target))
            len_target = len(new_target)
            prev = size.item()
            new_target = torch.Tensor(new_target).to(torch.long).to(device)
            correct+= float((out[idx].argmax(dim=1)[:len_target]==new_target).sum())
            total+= len_target

        output_data = []
        
        if save_output is not None:
            output_data.append((out.cpu(),targets))
    
    accuracy = correct/total
    return accuracy*100,output_data

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #device = torch.device("cuda:0" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)
    
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      labels=model.labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    acc, output_data = evaluate_acc(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    print('Test Summary \tAverage WER {acc:.3f}\t'.format(acc=acc))
    if args.save_output is not None:
        torch.save(output_data, args.save_output)
