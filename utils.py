import torch
import torch.distributed as dist
import pandas as pd

import os

import time
import json
import numpy as np
from tqdm.auto import tqdm

def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model_components(device, args,test=True):
    package = torch.load(args.model_path, map_location="cpu")
    models = package['models']
    [i[0].eval() for i in models.values()]
    if not test: return package['labels']
    pre, encoder_model, asr_model = models['preprocessing'][0], models['encoder'][0], models['predictor'][0]
    decoder = models['decoder'][0] if args.use_decoder else None
    forget_model = models['forget_net'][0] if args.forget_net else None
    disc_model = models['discriminator'][0] if args.disc else None

    model_components = [encoder_model, forget_model, disc_model, asr_model, decoder, pre]
    for i in range(len(model_components)):
        if model_components[i] is not None:
            model_components[i].eval()
            model_components[i].to(device)
    return model_components, package['accent_dict'], package # e, f, d, asr


class Decoder_loss():
    def __init__(self,criteria):
        self.loss = criteria

    def forward(self,target, output, WIDTHS,device):

        #create mask for padded instances
        mask = torch.arange(target.shape[2]).expand(len(WIDTHS), target.shape[2]) < WIDTHS.unsqueeze(1)
        mask = mask.unsqueeze(1)
        mask = torch.repeat_interleave(mask, target.shape[1], 1).to(device)

        #limit output to input shape
        output_inter = output[:,:,:target.shape[2]]

        #do element-wise multiplication to zero padded instances
        outputs = output_inter*mask

        #calculate loss
        loss_ = self.loss(outputs, target)

        return loss_

def weights_(args, accent_dict):
    accent_counts = pd.read_csv(args.train_manifest, header=None).iloc[:,[-1]].apply(pd.value_counts).to_dict()[2]#
    accent_counts = {str(i):j for i,j in accent_counts.items()}
    # print(sorted(accent_counts))
    # print(sorted(accent_dict))
    # exit()
    # # [len(accent_dict)]
    disc_loss_weights = torch.zeros(len(accent_dict))
    for count, i in enumerate(accent_dict):
        if accent_dict[i] == count: disc_loss_weights[count] =  accent_counts[i]
        else: print(f"error in weighted loss")
    return torch.sum(disc_loss_weights) / disc_loss_weights

def validation(test_loader,GreedyDecoder, models, args,accent,device,loss_save,labels,eps=0.0000000001):
    [i[0].eval() for i in models.values()]
    total_cer, total_wer, num_tokens, num_chars = eps, eps, eps, eps
    conf_mat = np.ones((len(accent), len(accent)))*eps # ground-truth: dim-0; predicted-truth: dim-1;
    acc_weights = np.ones((len(accent)))*eps
    length, num = eps, eps
    #Decoder used for evaluation
    target_decoder = GreedyDecoder(labels)
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        if args.dummy and i%2 == 1: break

        # Data loading
        inputs, targets, input_percentages, target_sizes, accents = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        
        # Forward pass
        if not args.train_asr:
            # Forward pass
            x_, updated_lengths_ = models['preprocessing'][0](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
            z, updated_lengths = models['encoder'][0](x_, updated_lengths_) # Encoder network
            m, updated_lengths = models['forget_net'][0](z,updated_lengths_) # Forget network
            z_ = z * m # Forget Operation
            discriminator_out = models['discriminator'][0](z_, updated_lengths) # Discriminator network
            asr_out, asr_out_sizes = models['predictor'][0](z_, updated_lengths) # Predictor network
        else:
            # Forward pass                  
            x_, updated_lengths = models['preprocessing'][0](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
            z,updated_lengths = models['encoder'][0](x_, updated_lengths) # Encoder network
            asr_out, asr_out_sizes = models['predictor'][0](z, updated_lengths) # Predictor network
    
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

        
        if not args.train_asr:
            # Discriminator metrics: fill in the confusion matrix.
            out, predicted = torch.max(discriminator_out, 1)
            for j in range(len(accents)):
                acc_weights[accents[j]] += 1
                if accents[j] == predicted[j].item():
                    num = num + 1
                conf_mat[accents[j], predicted[j].item()] += 1
            length = length + len(accents)
    
    # add comment janvijay
    tps, fps, tns, fns = np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps # class-wise TP, FP, TN, FN
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

    # TODO
    # if state == 'test':
    #     return wer, cer, num, length,  weighted_precision, weighted_recal, weighted_f1
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    
    return wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy


def Normalize(input):
    '''
    Normalize the input such that it only varies in the direction and not the magnitude
    '''
