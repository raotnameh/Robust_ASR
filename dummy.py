import argparse
import json
import os
import random
import time, math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
from collections import OrderedDict
import pandas as pd
from config import *

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from data.data_loader import get_accents
from decoder import GreedyDecoder
from model import *

from utils import reduce_tensor, check_loss, Decoder_loss, validation, weights_


parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--version', default='V1.0', help='experiment version')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=2, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
parser.add_argument('--patience', dest='patience', default=10, type=int, 
                    help='Break the training loop if the WER does not decrease for a certain number of epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=0.95, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--dummy', action='store_true',
                    help='do a dummy loop') 
parser.add_argument('--num-epochs', default=1, type=int,
                    help='choosing the number of iterations to train the discriminator in each training iteration')   
parser.add_argument('--exp-name', dest='exp_name', required=True, help='Location to save experiment\'s chekpoints and log-files.')

# Model arguements
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--enco-modules', dest='enco_modules', default=1, type=int,
                    help='Number of convolutional modules in Encoder net')
parser.add_argument('--enco-res', dest='enco_res', action='store_true', default= False,
                    help='Whether to keep in residual connections in encoder network')
parser.add_argument('--disc-modules', dest='disc_modules', default=1, type=int,
                    help='Number of convolutional modules in Encoder net')
parser.add_argument('--disc-res', dest='disc_res', action='store_true', default= False,
                    help='Whether to keep in residual connections in encoder network') 
parser.add_argument('--forg-modules', dest='forg_modules', default=1, type=int,
                    help='Number of convolutional modules in Encoder net')
parser.add_argument('--forg-res', dest='forg_res', action='store_true', default= False,
                    help='Whether to keep in residual connections in forget network')      
parser.add_argument('--update-rule', default=2, type=int,
                    help='train the discriminator k times')
parser.add_argument('--train-asr', action='store_true',
                    help='training only the ASR')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')

# Mixed precision training
parser.add_argument('--fp16', action='store_true',
                    help='training using fp16')

# Hyper parameters for the loss functions
parser.add_argument('--mw-alpha', type= float, default= 1,
                    help= 'weight for reconstruction loss')
parser.add_argument('--mw-beta', type= float, default= 1,
                    help= 'weight for discriminator loss')              
parser.add_argument('--mw-gamma', type= float, default= 1,
                    help= 'weight for regularisation')             

# noise arguments
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.gpu_rank: os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_rank
    version_ = args.version
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False#True
    accent_dict = get_accents(args.train_manifest) 
    accent = list(accent_dict.values())

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Gpu setting
    device = torch.device("cuda" if args.cuda else "cpu")

    #Where to save the models and training's metadata
    save_folder = os.path.join(args.exp_name, 'models')
    tbd_logs = os.path.join(args.exp_name, 'tbd_logdir')
    loss_save = os.path.join(args.exp_name, 'train.log')
    config_save = os.path.join(args.exp_name, 'config.json')
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    # save the experiment configuration.
    with open(config_save, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Instantiating tensorboard writer.
    writer = SummaryWriter(tbd_logs)

    wer_results = torch.Tensor(args.epochs)
    best_wer, best_cer = None, None
    d_avg_loss, p_avg_loss, p_d_avg_loss, start_epoch, amp_state = 0, 0, 0, 0, None
    poor_cer_list = []
    eps = 0.0000000001 # epsilon value
    start_iter = 0
    
    a = ""
    #Loading the labels
    with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))
    #Creating the configuration apply to the audio
    audio_conf = dict(sample_rate=args.sample_rate,
                        window_size=args.window_size,
                        window_stride=args.window_stride,
                        window=args.window,
                        noise_dir=args.noise_dir,
                        noise_prob=args.noise_prob,
                        noise_levels=(args.noise_min, args.noise_max))

    models = {} # All the models with their loss and optimizer are saved in this dict
    
    # Preprocessing
    pre = Encoder(161,configPre())
    pre_optimizer = torch.optim.Adam(pre.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    models['pre'] = [pre, None, pre_optimizer]

    # ASR
    print(len(labels))
    asr = Decoder(configE()[-1]['out_channels'],configDecoder()).to(device)
    asr_optimizer = torch.optim.Adam(asr.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    criterion = nn.CTCLoss(reduction='none')#CTCLoss()
    models['predictor'] = [asr, criterion, asr_optimizer]

    # Encoder and Decoder
    encoder = Encoder(configPre()[-1]['out_channels'],configE())
    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    models['encoder'] = [encoder, None, e_optimizer]

    # Lr scheduler
    scheduler = []
    for i in models.keys():
        models[i][0].to(device)
        scheduler.append(torch.optim.lr_scheduler.MultiplicativeLR(models[i][-1], lr_lambda=lambda epoch:args.learning_anneal, verbose=True))

    
    # Printing the models
    if not args.silent: print(nn.Sequential(OrderedDict( [(k,v[0]) for k,v in models.items()] )))
    
    #Creating the dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    disc_train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    disc_train_sampler = BucketingSampler(disc_train_dataset, batch_size=args.batch_size)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    disc_train_loader = AudioDataLoader(disc_train_dataset,
                                   num_workers=args.num_workers, batch_sampler=disc_train_sampler)
    
    disc_train_sampler.shuffle(start_epoch)
    disc_ = iter(disc_train_loader)

    test_loader = AudioDataLoader(test_dataset, batch_size=int(1.5*args.batch_size),
                                  num_workers=args.num_workers)

    if args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    # Printing the parameters of all the different modules 
    if not args.silent: [print(f"Number of parameters for {i[0]} in Million is: {get_param_size(i[1][0])/1000000}") for i in models.items()]
    accent_list = sorted(accent, key=lambda x:accent[x])
    a += f"epoch,epoch_time,wer,cer,acc,"
    for accent_type in accent_list:
        a += f"precision_{accent_type},"
    for accent_type in accent_list:
        a += f"recall_{accent_type},"
    for accent_type in accent_list:
        a += f"f1_{accent_type},"
    a += "d_avg_loss,p_avg_loss\n"
    
    # To choose the number of times update the discriminator
    update_rule = args.update_rule
    prob = np.geomspace(1, update_rule*100000, num=update_rule)[::-1]
    prob /= np.sum(prob)
    prob_ = [0 for i in prob]
    prob_[-1] = 1
    print(f"Initial Probability to udpate to the discrimiantor: {prob}")
    diff = np.array([ prob[i] - prob[-1-i] for i in range(len(prob))])
    diff /= len(train_sampler)*args.num_epochs

    #reading weights for different losses
    alpha = args.mw_alpha
    beta = args.mw_beta
    gamma = args.mw_gamma

    scaler = torch.cuda.amp.GradScaler(enabled=True if args.fp16 else False)
    for epoch in range(start_epoch, args.epochs):
        [i[0].train() for i in models.values()] # putting all the models in training state
        start_epoch_time = time.time()
        p_counter, d_counter = eps, eps

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            if args.dummy and i%2 == 1: break

            # Data loading
            inputs, targets, input_percentages, target_sizes, accents = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)

            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                package = {'models': models , 'start_epoch': epoch + 1, 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': i, 'accent_dict': accent_dict, 'version': version_, 'train.log': a}
                torch.save(package, os.path.join(save_folder, f"ckpt_{epoch+1}_{i+1}.pth"))
            
            if args.train_asr: # Only trainig the ASR component
                
                [m[-1].zero_grad() for m in models.values() if m[-1] is not None] #making graidents zero
                p_counter += 1
                with torch.cuda.amp.autocast(enabled=True if args.fp16 else False):
                    # Forward pass
                    x_, updated_lengths = models['pre'][0](inputs.squeeze(),input_sizes.type(torch.LongTensor).to(device))
                    z,updated_lengths = models['encoder'][0](x_, updated_lengths) # Encoder network
                    asr_out, asr_out_sizes = models['predictor'][0](z, updated_lengths) # Predictor network
                    asr_out = asr_out.transpose(0, 1) # TxNxH
                    # asd = asr_out.log_softmax(2).float()
                    # lo_ = models['predictor'][1](asr_out.log_softmax(2).float().contiguous(), targets.contiguous(), asr_out_sizes.contiguous(), target_sizes.contiguous())
                    # df_ = torch.mean(lo_)
                
                    asr_loss = torch.mean(models['predictor'][1](asr_out.log_softmax(2).float(), targets, asr_out_sizes, target_sizes))
                    # asr_loss = torch.mean(models['predictor'][1](asr_out.contiguous().log_softmax(2).float(), targets, asr_out_sizes, target_sizes).contiguous())
                    loss = asr_loss


                p_loss = loss.item()
                valid_loss, error = check_loss(loss, p_loss)
                if valid_loss:
                    scaler.scale(loss).backward()
                    for i_ in models.keys():
                        scaler.step(models[i_][-1])
                    scaler.update()
                else: 
                    print(error)
                    print("Skipping grad update")
                    p_loss = 0.0
                

                p_avg_loss += p_loss
                writer.add_scalar('Train/Predictor-Per-Iteration-Loss', p_loss, len(train_sampler)*epoch+i+1) # Predictor-loss in the current iteration.
                writer.add_scalar('Train/Predictor-Avergae-Loss-Cur-Epoch', p_avg_loss/p_counter, len(train_sampler)*epoch+i+1)
                print(f"Epoch: [{epoch+1}][{i+1}/{len(train_sampler)}]\t predictor Loss: {round(p_loss,4)} ({round(p_avg_loss/p_counter,4)})") 
                # continue

        d_avg_loss /= d_counter
        p_avg_loss /= p_counter
        epoch_time = time.time() - start_epoch_time
        start_iter = 0
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'D/P average Loss {2}, {3}\t'.format(epoch + 1, epoch_time, round(d_avg_loss,4),round(p_avg_loss,4)))
        
        d_avg_loss, p_avg_loss, p_d_avg_loss = 0, 0, 0

        start_ter = 0
        with torch.no_grad():
            wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,loss_save,labels,eps=0.0000000001)
        
        a += f"{epoch},{epoch_time},{wer},{cer},{num/length *100},"
    
        for idx, accent_type in enumerate(accent_list):
            a += f"{class_wise_precision[idx]},"
        for idx, accent_type in enumerate(accent_list):
            a += f"{class_wise_recall[idx]},"
        for idx, accent_type in enumerate(accent_list):
            a += f"{class_wise_f1[idx]},"
        a += f"{d_avg_loss},{p_avg_loss},{alpha},{beta},{gamma}\n"

        with open(loss_save, "w") as f:
            f.write(a)
        # Logging to tensorboard.
        writer.add_scalar('Validation/Average-WER', wer, epoch+1)
        writer.add_scalar('Validation/Average-CER', cer, epoch+1)
        writer.add_scalar('Validation/Discriminator-Accuracy', num/length *100, epoch+1)
        writer.add_scalar('Validation/Discriminator-Precision', weighted_precision, epoch+1)
        writer.add_scalar('Validation/Discriminator-Recall', weighted_recall, epoch+1)
        writer.add_scalar('Validation/Discriminator-F1', weighted_f1, epoch+1)
        
        print('Validation Summary Epoch: [{0}]\t'
                'Average WER {wer:.3f}\t'
                'Average CER {cer:.3f}\t'
                'Accuracy {acc_: .3f}\t'
                'Discriminator accuracy (micro) {acc: .3f}\t'
                'Discriminator precision (micro) {pre: .3f}\t'
                'Discriminator recall (micro) {rec: .3f}\t'
                'Discriminator F1 (micro) {f1: .3f}\t'.format(epoch + 1, wer=wer, cer=cer, acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))

        # saving
        if best_wer is None or best_wer > wer:
            best_wer = wer
            print("Updating the final model!")
            package = {'models': models , 'start_epoch': epoch+1, 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': None, 'accent_dict': accent_dict, 'version': version_, 'train.log': a}
            torch.save(package, os.path.join(save_folder, f"ckpt_final.pth"))
            
        if args.checkpoint:
            package = {'models': models , 'start_epoch': epoch+1, 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': None, 'accent_dict': accent_dict, 'version': version_, 'train.log': a}
            torch.save(package, os.path.join(save_folder, f"ckpt_{epoch+1}.pth"))

        # Exiting criteria
        terminate_train = False
        if best_cer is None or best_cer > cer:
            best_cer = cer
            poor_cer_list = []
        else:
            poor_cer_list.append(cer)
            if len(poor_cer_list) >= args.patience:
                print("Exiting training loop...")
                terminate_train = True
        # if terminate_train:
        #     break

        d_avg_loss, p_avg_loss, p_d_avg_loss = 0, 0, 0

        # anneal lr
        for i in scheduler:
            i.step()

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

    writer.close()

# TODO
# logger fo rtraining combine the losses
