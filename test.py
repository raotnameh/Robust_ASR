import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from utils import load_model_components

from data.data_loader import accent as accent_dict

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=32, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default="test", help="Saves output of model from test to this file_path")
parser.add_argument('--gpu', dest='gpu', type=str, help='GPU to be used', required=True)
parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda or not')
parser.add_argument('--half', action='store_true', default=False, help='whether to use half precision or not')
# checkpoint loading args
parser.add_argument('--model-path', dest='model_path', type=str, help='Path to "model" directory, where weights of component of models are saved', required=True)
parser.add_argument('--f', dest='forget', action='store_true', required=True, help='Whether to include forget module or not')
parser.add_argument('--d', dest='discriminate', action='store_true', required=True, help='Whether to include discriminator module or not')
parser.add_argument('--ckpt-id', dest='ckpt_id', default='final', help='checkpoint id to load from model_path directory for component modules')
# decoder args
parser.add_argument('--decoder', dest='decoder', default='greedy', help='type of decoder to use.')
# audio conf
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')

# model compnents: e, f, d, asr
def forward_call(model_components, inputs, inputs_sizes): 
    z, updated_lengths = model_components[0](inputs, inputs_sizes)
    a = z
    if model_components[1] is not None: # forget net is present
        m = model_components[1](inputs, inputs_sizes)
        z = z * m
    else: m = None
    disc_out = model_components[2](z) if model_components[2] is not None else None # discrmininator is present or not
    asr_out, asr_out_sizes = model_components[3](z, updated_lengths)
    return (asr_out, asr_out_sizes), disc_out, a, z, updated_lengths, m


def evaluate(test_loader, device, model_components, target_decoder, save_output=None, verbose=False, half=False):
    eps = 0.0000000001
    accent = list(accent_dict.values())
    dict_z = []
    dict_z_ = []
    dict_m = []
    with torch.no_grad():
        total_cer, total_wer, num_tokens, num_chars = eps, eps, eps, eps
        conf_mat = np.ones((len(accent), len(accent)))*eps # ground-truth: dim-0; predicted-truth: dim-1;
        tps, fps, tns, fns = np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps # class-wise TP, FP, TN, FN
        length, num = eps, eps
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
            (asr_out, asr_out_sizes), disc_out, z, z_, updated_lengths, m = forward_call(model_components, inputs, input_sizes.type(torch.LongTensor).to(device))
            
            #saving z and z_
            dict_z.append([z,accents,updated_lengths]) 
            dict_z_.append([z_,accents,updated_lengths])
            if m is not None: dict_m.append(m)
            # Predictor metric
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
            decoded_output, _ = target_decoder.decode(asr_out, asr_out_sizes)
            target_strings = target_decoder.convert_to_strings(split_targets)

            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer_inst = target_decoder.wer(transcript, reference)
                cer_inst = target_decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference.replace(' ', ''))

            wer = float(total_wer) / num_tokens
            cer = float(total_cer) / num_chars
    
            if disc_out is not None:
                # Discriminator metrics: fill in the confusion matrix.
                out, predicted = torch.max(disc_out, 1)
                for j in range(len(accents)):
                    if accents[j] == predicted[j].item():
                        num = num + 1
                    conf_mat[accents[j], predicted[j].item()] += 1
                length = length + len(accents)

    if model_components[2] is not None: # if discriminator is present.
        # Discriminator metrics: compute metrics using confustion metrics.
        for acc_type in range(len(accent)):
            tps[acc_type] = conf_mat[acc_type, acc_type]
            fns[acc_type] = np.sum(conf_mat[acc_type, :]) - tps[acc_type]
            fps[acc_type] = np.sum(conf_mat[:, acc_type]) - tps[acc_type]
            tns[acc_type] = np.sum(conf_mat) - tps[acc_type] - fps[acc_type] - fns[acc_type]
        class_wise_precision, class_wise_recall = tps/(tps+fps), tps/(fns+tps)
        class_wise_f1 = 2 * class_wise_precision * class_wise_recall / (class_wise_precision + class_wise_recall)
        macro_precision, macro_recall, macro_accuracy = np.mean(class_wise_precision), np.mean(class_wise_recall), np.mean((tps+tns)/(tps+fps+fns+tns))
        micro_precision, micro_recall, micro_accuracy = tps.sum()/(tps.sum()+fps.sum()), tps.sum()/(fns.sum()+tps.sum()), (tps.sum()+tns.sum())/(tps.sum()+tns.sum()+fns.sum()+fps.sum())
        micro_f1, macro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall), 2*macro_precision*macro_recall/(macro_precision+macro_recall)
        print('Discriminator accuracy (micro) {acc: .3f}\t'
              'Discriminator precision (micro) {pre: .3f}\t'
              'Discriminator recall (micro) {rec: .3f}\t'
              'Discriminator F1 (micro) {f1: .3f}\t'.format(acc_ = num/length *100 , acc=micro_accuracy, pre=micro_precision, rec=micro_recall, f1=micro_f1))
    
    print('Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    #saving z and z_
    torch.save(dict_z,f"{args.save_output}/z.pth") 
    torch.save(dict_z_,f"{args.save_output}/z_.pth")
    torch.save(dict_m,f"{args.save_output}/m.pth")

labels = ['_',"'",'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_components = load_model_components(device, args.model_path, args.forget, args.discriminate, args.ckpt_id, args.half)
    
    #Creating the configuration apply to the audio
    audio_conf = dict(sample_rate=args.sample_rate,
                        window_size=args.window_size,
                        window_stride=args.window_stride,
                        window=args.window,
                        noise_dir=args.noise_dir,
                        noise_prob=args.noise_prob,
                        noise_levels=(args.noise_min, args.noise_max))

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
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None
    
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    evaluate(test_loader=test_loader,
                                     device=device,
                                     model_components=model_components,
                                     target_decoder=decoder,
                                     save_output=False,
                                     verbose=False,
                                     half=args.half)

    # print('Test Summary \t'
    #       'Average WER {wer:.3f}\t'
    #       'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    # if args.save_output is not None:
    #     torch.save(output_data, args.save_output)
