import argparse
import os
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
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


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=None, verbose=False, half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data

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
            new_target += [0]*(new_timesteps-len(new_target))
            prev = size.item()
            new_targets.append(new_target)

        new_targets = torch.Tensor(new_targets).to(torch.long).to(device)
        output_data = []
        
        correct += float((out.argmax(dim=2)==new_targets).sum())
        total += new_targets.size(0)*new_targets.size(1)
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
    
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels,
                                 lm_path=args.lm_path,
                                 alpha=args.alpha,
                                 beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n,
                                 cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width,
                                 num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      labels=model.labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    wer, cer, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    if args.save_output is not None:
        torch.save(output_data, args.save_output)
