import torch
import torch.distributed as dist
import pandas as pd

import os

import time
import json
import numpy as np
from tqdm.auto import tqdm

import horovod.torch as hvd

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

def validation(test_loader,GreedyDecoder, models, args,accent,device,labels,finetune=False,mtl=False,eps=0.0000000001):
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
            if finetune or mtl: z_ = z
            else: 
                # print('using forget net')
                m, updated_lengths = models['forget_net'][0](z,updated_lengths_) # Forget network
                # print(m)
                z_ = z * m # Forget Operation
            
            discriminator_out = models['discriminator'][0](z_, updated_lengths) # Discriminator network
            asr_out, asr_out_sizes = models['predictor'][0](z_, updated_lengths) # Predictor network

            
        else:
            # Forward pass                  
            x_, updated_lengths = models['preprocessing'][0](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
            z,updated_lengths = models['encoder'][0](x_, updated_lengths) # Encoder network
            asr_out, asr_out_sizes = models['predictor'][0](z, updated_lengths) # Predictor network
    
        asr_out = asr_out.softmax(2).float()
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

def finetune_disc(models,disc_train_loader,device,args,scaler,disc_train_sampler,writer,test_loader, GreedyDecoder,accent,labels,save_folder):

    if hvd.rank() == 0:
        
        if args.warmup: package = torch.load(args.warmup, map_location=(f"cuda:0" if args.cuda else "cpu"))
        else: package = torch.load(args.continue_from, map_location=(f"cuda:0" if args.cuda else "cpu"))

        with torch.no_grad():
            wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,labels,finetune=True,eps=0.0000000001)
        
        # Logging to tensorboard.
        writer.add_scalar('Finetune/Average-WER', wer, 0)
        writer.add_scalar('Finetune/Average-CER', cer, 0)
        writer.add_scalar('Finetune/Discriminator-Accuracy', num/length *100, 0)
        writer.add_scalar('Finetune/Discriminator-Precision', weighted_precision, 0)
        writer.add_scalar('Finetune/Discriminator-Recall', weighted_recall, 0)
        writer.add_scalar('Finetune/Discriminator-F1', weighted_f1, 0)
        
        print('Validation Summary Epoch: [{0}]\t'
                'Average WER {wer:.3f}\t'
                'Average CER {cer:.3f}\t'
                'Accuracy {acc_: .3f}\t'
                'Discriminator accuracy (micro) {acc: .3f}\t'
                'Discriminator precision (micro) {pre: .3f}\t'
                'Discriminator recall (micro) {rec: .3f}\t'
                'Discriminator F1 (micro) {f1: .3f}\t'.format(0, wer=wer, cer=cer, acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))
    if args.test_disc: exit()
    for epoch in range(args.epochs):
        
        [i[0].train() for i in models.values()] # putting all the models in training state
        start_epoch_time = time.time()
        d_avg_loss, d_counter = 0, 0
        for i, (data) in enumerate(disc_train_loader):
            
            # Data loading
            inputs_, targets_, input_percentages_, target_sizes_, accents_ = data
            input_sizes_ = input_percentages_.mul_(int(inputs_.size(3))).int()
            inputs_ = inputs_.to(device)
            accents_ = torch.tensor(accents_).to(device)
                       
            d_counter += 1
            [m[-1].zero_grad() for m in models.values() if m is not None] #making graidents zero

            with torch.cuda.amp.autocast(enabled=True if args.fp16 else False):
                # Forward pass
                with torch.no_grad(): 
                    x_, updated_lengths_ = models['preprocessing'][0](inputs_.squeeze(dim=1),input_sizes_.type(torch.LongTensor).to(device))
                    z, updated_lengths = models['encoder'][0](x_,updated_lengths_) # Encoder network
                discriminator_out = models['discriminator'][0](z, updated_lengths) # Discriminator network
                # Loss             
                discriminator_loss = models['discriminator'][1](discriminator_out, accents_)

            d_loss = discriminator_loss.item()
            valid_loss, error = check_loss(discriminator_loss, d_loss)
            if valid_loss:
                scaler.scale(discriminator_loss).backward()
                for i_ in ['discriminator']: 
                    models[i_][-1].synchronize()
                    with models[i_][-1].skip_synchronize():
                        scaler.step(models[i_][-1])
                scaler.update()
            else: 
                print(error)
                print("Skipping grad update")
                d_loss = 0.0
            
            d_avg_loss += d_loss

            if hvd.rank() == 0:
                if not args.silent: print(f"Epoch: [{epoch+1}][{i+1}/{len(disc_train_sampler)}]\t\t\t\t\t Discriminator Loss: {round(d_loss,4)} ({round(d_avg_loss/d_counter,4)})")
                # Logging to tensorboard.
                writer.add_scalar('Finetune/Discriminator-finetune-loss', d_loss, len(disc_train_sampler)*epoch+i+1) # Discriminator's training loss in the current main - iteration.
            
        epoch_time = time.time() - start_epoch_time
        if hvd.rank() == 0:
            print('Training discriminator Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'D average Loss {2}\t'.format(epoch + 1, epoch_time, round(d_avg_loss/d_counter,4)))

        if hvd.rank() == 0:
            with torch.no_grad():
                wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,labels,finetune=True,eps=0.0000000001)
            
            # Logging to tensorboard.
            writer.add_scalar('Finetune/Average-WER', wer, epoch+1)
            writer.add_scalar('Finetune/Average-CER', cer, epoch+1)
            writer.add_scalar('Finetune/Discriminator-Accuracy', num/length *100, epoch+1)
            writer.add_scalar('Finetune/Discriminator-Precision', weighted_precision, epoch+1)
            writer.add_scalar('Finetune/Discriminator-Recall', weighted_recall, epoch+1)
            writer.add_scalar('Finetune/Discriminator-F1', weighted_f1, epoch+1)
            
            print('Validation Summary Epoch: [{0}]\t'
                    'Average WER {wer:.3f}\t'
                    'Average CER {cer:.3f}\t'
                    'Accuracy {acc_: .3f}\t'
                    'Discriminator accuracy (micro) {acc: .3f}\t'
                    'Discriminator precision (micro) {pre: .3f}\t'
                    'Discriminator recall (micro) {rec: .3f}\t'
                    'Discriminator F1 (micro) {f1: .3f}\t'.format(epoch + 1, wer=wer, cer=cer, acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))

            # saving
            print("Updating the final model!")
            save = {}
            for s_ in models.keys(): 
                save[s_] = []
                save[s_].append(models[s_][0]) 
                save[s_].append(models[s_][1]) 
                save[s_].append(models[s_][2].state_dict()) 
            package['models'] = save
            torch.save(package, os.path.join(save_folder, f"ckpt_final_{epoch}.pth"))
            del save
                        
        # anneal lr
        dummy_lr = None
        for i in models:
            for g in models[i][-1].param_groups:
                if dummy_lr is None: dummy_lr = g['lr']
                if g['lr'] >= 1e-6:
                    g['lr'] = g['lr'] * args.learning_anneal
            print(f"Learning rate of {i} annealed to: {g['lr']} from {dummy_lr}")
            dummy_lr = None
        
        if not args.no_shuffle:
            print("Shuffling batches...")
            disc_train_sampler.shuffle(epoch)



