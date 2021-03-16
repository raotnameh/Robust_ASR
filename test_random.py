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
import horovod.torch as hvd
from config import *

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.data_loader import get_accents
from decoder import GreedyDecoder
from model import *
from utils import *

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
parser.add_argument('--disc-kl-loss', action='store_true',
                    help='use kl divergence loss for discriminator')
parser.add_argument('--early-val', default=100, type=int,
                    help='Doing an early validation step')                    
                    
# Model arguements
parser.add_argument('--update-rule', default=2, type=int,
                    help='train the discriminator k times')
parser.add_argument('--train-asr', action='store_true',
                    help='training only the ASR')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')

# Hyper parameters for the loss functions
parser.add_argument('--alpha', type= float, default= 1,
                    help= 'weight for reconstruction loss')
parser.add_argument('--beta', type= float, default= 1,
                    help= 'weight for discriminator loss')              
parser.add_argument('--gamma', type= float, default= 1,
                    help= 'weight for regularisation')             

# input augments
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

# Mixed precision training
parser.add_argument('--fp16', action='store_true',
                    help='training using fp16')
parser.add_argument('--using-new', action='store_true',
                    help='using-new-weights')                    

if __name__ == '__main__':
    args = parser.parse_args()
    # args.fp16 = False # bugs with multi gpu training
    if args.gpu_rank: os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_rank
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    version_ = args.version
    #Lables for the discriminator
    accent_dict = get_accents(args.train_manifest) 
    accent = list(accent_dict.values())

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Gpu setting
    device = torch.device("cuda" if args.cuda else "cpu")

    
    if args.continue_from:
        package = torch.load(args.continue_from, map_location=(f"cuda" if args.cuda else "cpu"))
        models = package['models'] 
        labels, audio_conf, version_, start_iter = package['labels'], package['audio_conf'], package['version'], package['start_iter']
        audio_conf['noise_dir'] = args.noise_dir
        audio_conf['noise_prob'] = args.noise_prob
        audio_conf['noise_levels'] = (args.noise_min, args.noise_max)
        del models['discriminator']
        del models['decoder']

        asr = Predictor(configE()[-1]['out_channels'],configP(labels=len(labels)))
        asr_optimizer = torch.optim.Adam(asr.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
        criterion = nn.CTCLoss(reduction='none')#CTCLoss()
        models['predictor'] = [asr, criterion, asr_optimizer]
        
        a = ""
        start_iter = 0
        version_ = args.version
        print("loaded models succesfully")

    if not args.silent: 
        [print(f"Number of parameters for {i[0]} in Million is: {get_param_size(i[1][0])/1000000}") for i in models.items()]
        print(f"Total number of parameter is: {sum([get_param_size(i[1][0])/1000000 for i in models.items()])}")
        print(models.keys())
    
    #Creating the dataset
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    
    test_loader = AudioDataLoader(test_dataset, batch_size=4*int(args.batch_size),
                                  num_workers=args.num_workers,pin_memory=True)

    for i in models.keys():
        models[i][0].to(device)
    with torch.no_grad():
        print("start-validation")
        wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,labels,eps=0.0000000001)
    
    epoch = 0
    print('Validation Summary Epoch: [{0}]\t'
            'Average WER {wer:.3f}\t'
            'Average CER {cer:.3f}\t'
            'Accuracy {acc_: .3f}\t'
            'Discriminator accuracy (micro) {acc: .3f}\t'
            'Discriminator precision (micro) {pre: .3f}\t'
            'Discriminator recall (micro) {rec: .3f}\t'
            'Discriminator F1 (micro) {f1: .3f}\t'.format(epoch + 1, wer=wer, cer=cer, acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))
