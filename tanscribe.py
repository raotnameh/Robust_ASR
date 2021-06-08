import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from data.data_loader import SpectrogramDataset, AudioDataLoader, accent
from decoder import GreedyDecoder
from opts import add_decoder_args

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_decoder_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=64, type=int, help='Number of workers used in dataloading')
parser.add_argument('--gpu-rank', dest='gpu_rank', type=str, help='GPU to be used', required=True)
parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda or not')
# checkpoint loading args
parser.add_argument('--model-path', type=str, help='Path to "model" directory, where weights of component of models are saved', required=True)
# decoder args
parser.add_argument('--decoder', default='greedy', help='type of decoder to use.')



def load_model_components(device, args):
    package = torch.load(args.model_path, map_location="cpu")
    models = package['models']
    [i.eval() for i in models.values()]
    pre, encoder_model, asr_model = models['preprocessing'], models['encoder'], models['predictor']

    model_components = [pre, encoder_model, asr_model]
    for i in range(len(model_components)):
        model_components[i].eval()
        model_components[i].to(device)

    return model_components, package['audio_conf'], package['labels'] # e, f, d, asr


# model compnents: e, f, d, asr
def forward_call(model_components, inputs, input_sizes, device,args): 
    # preprocessing pass
    x_, updated_lengths = model_components[0](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
    # encoder pass.
    z, updated_lengths = model_components[1](x_, updated_lengths) 
    # predictor pass.
    asr_out, asr_out_sizes = model_components[2](z, updated_lengths)
    return asr_out.softmax(2).float(), asr_out_sizes


def evaluate(test_loader, device, model_components, target_decoder, decoder, args):

    with torch.no_grad():
        total_cer, total_wer, num_tokens, num_chars = 0,0,0,0
 
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i==len(test_loader)-2:
                break
            if data is None:
                continue
            # Data loading
            inputs, targets, input_percentages, target_sizes, accents = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            
            # Forward pass
            asr_out, asr_out_sizes= forward_call(model_components, inputs, input_sizes, device,args)
            
            # Predictor metric
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
            decoded_output, _ = target_decoder.decode(asr_out, asr_out_sizes)
            target_strings = decoder.convert_to_strings(split_targets)

            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                # print(transcript, reference)
                wer_inst = target_decoder.wer(transcript, reference)
                cer_inst = target_decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference.replace(' ', ''))

        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

    print('Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{args.gpu_rank}" if args.cuda else "cpu")

    model_components, audio_conf, labels  = load_model_components(device, args)

    print("Loaded models successfully...")

    base_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder
        decoder = BeamCTCDecoder(labels,
                                 lm_path=args.lm_path,
                                 alpha=args.alpha,
                                 beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n,
                                 cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width,
                                 num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      use_noise=False,
                                      manifest_filepath=args.test_manifest,
                                      labels=labels,
                                      normalize=True,
                                      )
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    evaluate(test_loader=test_loader,
            device=device,
            model_components=model_components,
            target_decoder=decoder,
            decoder = base_decoder,
            args = args)
