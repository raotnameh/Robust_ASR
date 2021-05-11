import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from data.data_loader import SpectrogramDataset, AudioDataLoader, accent
from decoder import GreedyDecoder
from utils import load_model_components
from opts import add_decoder_args

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_decoder_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=64, type=int, help='Number of workers used in dataloading')
parser.add_argument('--save-representation', default=False, help="Saves outtput representations z, z_, and m")
parser.add_argument('--save-output', default=False, help="Saves output of model from test to this file_path")
parser.add_argument('--gpu-rank', dest='gpu_rank', type=str, help='GPU to be used', required=True)
parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda or not')
# checkpoint loading args
parser.add_argument('--model-path', type=str, help='Path to "model" directory, where weights of component of models are saved', required=True)
parser.add_argument('--visual-path', type=str, help='Path to where the meta info for visualization will be saved')
parser.add_argument('--forget-net', action='store_true', required=False, help='Whether to include forget module or not')
parser.add_argument('--disc' , action='store_true', required=False, help='Whether to include discriminator module or not')
parser.add_argument('--use-decoder' , action='store_true', required=False, help='Whether to include decoder or not')
parser.add_argument('--visual' , action='store_true', required=False, help='Whether to save meta info for visualization')
# decoder args
parser.add_argument('--decoder', default='greedy', help='type of decoder to use.')

# model compnents: e, f, d, asr
def forward_call(model_components, inputs, input_sizes, device,args): 
    # preprocessing pass
    x_, updated_lengths = model_components[5](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
    # encoder pass.
    z, updated_lengths = model_components[0](x_, updated_lengths) 
    a = z
    # decoder pass
    decoder_out_e, decoder_out_f = None, None
    if args.use_decoder: decoder_out_e, _ = model_components[4](z, updated_lengths)
    # forget net forward pass.
    if model_components[1] is not None: 
        m, _ = model_components[1](z, updated_lengths)
        z = z * m
        if args.use_decoder:  decoder_out_f, _ = model_components[4](z, updated_lengths)
    else: 
        m = None
        decoder_out_f = None
    # dicriminator net forward pass.
    disc_out = model_components[2](z) if model_components[2] is not None else None # discrmininator is present or not
    # asr forward pass.
    asr_out, asr_out_sizes = model_components[3](z, updated_lengths)
    return asr_out.softmax(2).float(), asr_out_sizes, disc_out, decoder_out_e, decoder_out_f, a, z, updated_lengths, m


def evaluate(test_loader, accent_dict, device, model_components, target_decoder, decoder, args):
    eps = 0.0000000001
    accent = list(accent_dict.values()) # Corresponds to number of accents in training set. (num of output classes in discriminator)
    dict_z = []
    dict_z_ = []
    dict_m = []
    dict_d = []
    output_data = []
    with torch.no_grad():
        total_cer, total_wer, num_tokens, num_chars = eps, eps, eps, eps
        conf_mat = np.ones((len(accent), len(accent)))*eps # ground-truth: dim-0; predicted-truth: dim-1;
        acc_weights = np.ones((len(accent)))*eps
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
            asr_out, asr_out_sizes, disc_out, decoder_out_e, decoder_out_f, z, z_, updated_lengths, m = forward_call(model_components, inputs, input_sizes, device,args)
            #saving z and z_
            if args.save_representation:
                dict_z.append([z.cpu(),accents,updated_lengths]) 
                dict_z_.append([z_.cpu(),accents,updated_lengths])
                if m is not None: dict_m.append([m.cpu(),accents,updated_lengths])
                if args.use_decoder: dict_d.append([inputs.cpu(),decoder_out_e.cpu(),decoder_out_f.cpu(),accents,updated_lengths])
            
            # Predictor metric
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
            decoded_output, _ = target_decoder.decode(asr_out, asr_out_sizes)
            target_strings = decoder.convert_to_strings(split_targets)
            
            if args.save_output is not None:
                # add output to data array, and continue
                output_data.append((asr_out.cpu(), asr_out_sizes.cpu(), target_strings))

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
    
            if disc_out is not None:
                # Discriminator metrics: fill in the confusion matrix.
                out, predicted = torch.max(disc_out, 1)
                for j in range(len(accents)):
                    acc_weights[accents[j]] += 1
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
        weighted_precision, weighted_recall = ((acc_weights / acc_weights.sum()) * class_wise_precision).sum(), ((acc_weights / acc_weights.sum()) * class_wise_recall).sum()
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        micro_precision, micro_recall, micro_accuracy = tps.sum()/(tps.sum()+fps.sum()), tps.sum()/(fns.sum()+tps.sum()), (tps.sum()+tns.sum())/(tps.sum()+tns.sum()+fns.sum()+fps.sum())
        micro_f1, macro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall), 2*macro_precision*macro_recall/(macro_precision+macro_recall)
        print('Discriminator accuracy (micro) {acc: .3f}\t'
              'Discriminator precision (micro) {pre: .3f}\t'
              'Discriminator recall (micro) {rec: .3f}\t'
              'Discriminator F1 (micro) {f1: .3f}\t'.format(acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))
    
    print('Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

    # saving
    if args.save_representation:
        torch.save(dict_z,f"{args.save_representation}/z.pth") 
        torch.save(dict_z_,f"{args.save_representation}/z_.pth")
        torch.save(dict_m,f"{args.save_representation}/m.pth")
        torch.save(dict_d,f"{args.save_representation}/d.pth")
    if args.save_output: 
        torch.save(output_data, f"{args.save_output}/out.pth")

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{args.gpu_rank}" if args.cuda else "cpu")

    model_components, accent_dict, package = load_model_components(device, args)
    if args.forget_net:
        assert model_components[1] is not None, "forget net not found in checkpoint"
    if args.disc:
        assert model_components[2] is not None, "discriminate net not found in checkpoint"
    print("Loaded models successfully...")

    #Loading the configuration apply to the audio and labels of asr
    audio_conf = package['audio_conf']
    labels = package['labels']

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
                                      manifest_filepath=args.test_manifest,
                                      labels=labels,
                                      normalize=True,
                                      audio_recreation=args.visual,
                                      metadata_reacreation_path=args.visual_path)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    evaluate(test_loader=test_loader,
            accent_dict = accent_dict,
            device=device,
            model_components=model_components,
            target_decoder=decoder,
            decoder = base_decoder,
            args = args)
