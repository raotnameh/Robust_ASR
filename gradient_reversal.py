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
from config.config_small import *
# from config.config_jasper import *
#from config.config_large import *

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
parser.add_argument('--finetune-disc', dest='finetune_disc', action='store_true',
                    help='Finetune the discriminator from checkpoint "continue_from"')
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
parser.add_argument('--early-val', default=10e-10, type=int,
                    help='Doing an early validation step')                    
parser.add_argument('--warmup', default=None, help='start-from from checkpoint model')
parser.add_argument('--use-decoder', action='store_true', help='To use decoder or no')
parser.add_argument('--test-disc', dest='test_disc', action='store_true',
                    help='Test the discriminator from checkpoint ')
# Model arguements
parser.add_argument('--update-rule', default=2, type=float,
                    help='train the discriminator k times')
parser.add_argument('--train-asr', action='store_true',
                    help='training only the ASR')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')

# Hyper parameters for the loss functions
parser.add_argument('--alpha', type= float, default= 1e-5,
                    help= 'weight for reconstruction loss')
parser.add_argument('--beta', type= float, default= 1e-5,
                    help= 'weight for discriminator loss')              
parser.add_argument('--gamma', type= float, default= 1e-5,
                    help= 'weight for regularisation') 
parser.add_argument('--hyper-rate', type= float, default= 1.1,
                    help= 'Hyper parameter rate')             

# input augments
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.25,type=float, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.1,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.25,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')

# Mixed precision training
parser.add_argument('--fp16', action='store_true',
                    help='training using fp16')                  


class GradientReversalLayer(torch.autograd.Function):
	"""
	Implement the gradient reversal layer for the convenience of domain adaptation neural network.
	The forward part is the identity function while the backward part is the negative function."""
	@staticmethod
	def forward(self, inputs):
		return inputs
	@staticmethod
	def backward(self, grad_output):
		# grad_input = grad_output.clone()
		# grad_input = -1 * grad_input
		return -1 * grad_output #grad_input

def grad_reverse(x):
	return GradientReversalLayer.apply(x)


if __name__ == '__main__':
    args = parser.parse_args()
    # args.fp16 = False # bugs with multi gpu training
    if args.gpu_rank: os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_rank
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    # Initializing horovod distributed training
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    version_ = args.version
    #Lables for the discriminator
    accent_dict = get_accents(args.train_manifest) 
    accent = list(accent_dict.values())

    # Lr scaler for multi gpu training
    lr_scaler = hvd.size()
    args.lr = args.lr * lr_scaler
    if args.lr > 0.025: args.lr = 0.025
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Gpu setting
    device = torch.device("cuda" if args.cuda else "cpu")

    # Where to save the models and training metadata
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
    d_avg_loss, p_avg_loss, p_d_avg_loss, start_epoch = 0, 0, 0, 0
    poor_cer_list = []
    eps = 0.0000000000001 # epsilon value
    start_iter = 0
    
    if args.continue_from:
        package = torch.load(args.continue_from, map_location=(f"cuda" if args.cuda else "cpu"))
        models = package['models']
        a = ''
        labels, audio_conf, version_, start_iter = package['labels'], package['audio_conf'], package['version'], package['start_iter']
        audio_conf['noise_dir'] = args.noise_dir
        audio_conf['noise_prob'] = args.noise_prob
        audio_conf['noise_levels'] = (args.noise_min, args.noise_max)
    
        if not args.train_asr: # if adversarial training.
            assert 'discriminator' in models.keys(), "discriminator not found in checkpoint loaded"
        else: 
            try: 
                print("Deleting the forget_net and discriminator")
                del models['forget_net']
                del models['discriminator']
            except: pass
        
        if not args.finetune: # If continuing training after the last epoch.
            
            dummy = {i:models[i][-1] for i in models}
            for i in models:
                models[i][-1] = torch.optim.Adam(models[i][0].parameters(), lr=package['lr'],weight_decay=1e-4,amsgrad=True)
                models[i][-1].load_state_dict(dummy[i])
            del dummy

            start_epoch = package['start_epoch']  # Index start at 0 for training
            if start_iter is None:
                # start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
                best_wer = package['best_wer']
                best_cer = package['best_cer']
                poor_cer_list = package['poor_cer_list']
                a = package['train.log']
        else:
            start_iter = 0
            a = ""
            version_ = args.version
            for i in models:
                if i == "forget_net": models[i][-1] = torch.optim.Adam(models[i][0].parameters(), lr=100*args.lr,weight_decay=1e-4,amsgrad=True)    
                else: models[i][-1] = torch.optim.Adam(models[i][0].parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
        # print(best_cer, best_wer, audio_conf,start_iter)
        print("loaded models succesfully")
    else:
        if hvd.rank() == 0: 
            a = ""
            accent_list = sorted(accent, key=lambda x:accent[x])
            a += f"epoch,epoch_time,wer,cer,acc,precision,recall,f1,d_avg_loss,p_avg_loss\n"
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
        if not args.warmup:
            models = {} # All the models with their loss and optimizer are saved in this dict
            
            # Preprocessing
            pre = Pre(161,configPre()) # 161 comes from the spectrogram feature for each time step.
            pre_optimizer = torch.optim.Adam(pre.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
            models['preprocessing'] = [pre, None, pre_optimizer]

            # Encoder and Decoder
            encoder = Encoder(configPre()[-1]['out_channels'],configE())
            e_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
            models['encoder'] = [encoder, None, e_optimizer]
            if args.use_decoder:
                print("Using the Decoder")
                decoder = Decoder(configE()[-1]['out_channels'],configD())
                d_optimizer = torch.optim.Adam(decoder.parameters(),lr=args.lr,weight_decay=1e-4,amsgrad=True)
                dec_loss = Decoder_loss(nn.MSELoss())
                models['decoder'] = [decoder, dec_loss, d_optimizer]
            
            # ASR
            asr = Predictor(configE()[-1]['out_channels'],configP(labels=len(labels)))
            asr_optimizer = torch.optim.Adam(asr.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
            criterion = nn.CTCLoss(reduction='none')#CTCLoss()
            models['predictor'] = [asr, criterion, asr_optimizer]
        
        if not args.train_asr:
            # Discriminator
            discriminator = Discriminator(configFN()[-1]['out_channels'],configDM(),classes=len(accent))
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
            # Weighted loss depending on the class count
            disc_loss_weights = weights_(args, accent_dict).to(device)   
            dis_loss = nn.CrossEntropyLoss(weight=disc_loss_weights)
            models['discriminator'] = [discriminator, dis_loss, discriminator_optimizer]

    if hvd.rank() == 0:
        if not args.silent: 
            # Printing the models
            print(nn.Sequential(OrderedDict( [(k,v[0]) for k,v in models.items()] )))
             # Printing the parameters of all the different modules 
            for i in models.items():
                print(f"Number of parameters for {i[0]} in Million is: {get_param_size(i[1][0])/1000000}") 
            print(f"Total number of parameter is: {sum([get_param_size(i[1][0])/1000000 for i in models.items()])}")
            print(f"Initial learning rate: {print(models['encoder'][-1].param_groups[0]['lr'])}")
    
    #Creating the dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    if not args.train_asr: 
        disc_train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, use_noise=False, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                num_replicas=hvd.size(), rank=hvd.rank())
    if not args.train_asr: 
        disc_train_sampler = DistributedBucketingSampler(disc_train_dataset, batch_size=args.batch_size,
                                                num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler,pin_memory=True)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers,pin_memory=True)

    if not args.train_asr: 
        disc_train_loader = AudioDataLoader(disc_train_dataset,
                                   num_workers=args.num_workers, batch_sampler=disc_train_sampler,pin_memory=True)    
        disc_train_sampler.shuffle(start_epoch)
        disc_ = iter(disc_train_loader)
    
    if args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)
    # exit()
    # HVD multi gpu training
    for i in models.keys():
        models[i][0].to(device)
        hvd.broadcast_parameters(models[i][0].state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(models[i][-1], root_rank=0)
        models[i][-1] = hvd.DistributedOptimizer(models[i][-1], named_parameters=models[i][0].named_parameters())
    
    
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.fp16 else False) # fp16 training

    # Finetuning the discriminator.
    if args.finetune_disc and (not args.train_asr) and (args.warmup or args.continue_from):
        print("Finetuning discriminator")
        finetune_disc(models,disc_train_loader,device,args,scaler,disc_train_sampler,writer,test_loader, GreedyDecoder,accent,labels,save_folder)
        exit()

    for epoch in range(start_epoch, args.epochs):
        [i[0].train() for i in models.values()] # putting all the models in training state
        start_epoch_time = time.time()
        p_counter, d_counter = eps, eps
        if alpha <= 1.0: alpha = alpha * args.hyper_rate
        if beta <= args.update_rule: beta = beta * args.hyper_rate
        # else: beta = args.beta
        if gamma <= args.update_rule: gamma = gamma * args.hyper_rate
              
        if hvd.rank() == 0 : print(alpha,beta,gamma)
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            if args.dummy and i%2 == 1: break

            # Data loading
            inputs, targets, input_percentages, target_sizes, accents = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
        
            if hvd.rank() == 0 :
                if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                    save = {}
                    for s_ in models.keys():
                        save[s_] = []
                        save[s_].append(models[s_][0]) 
                        save[s_].append(models[s_][1]) 
                        save[s_].append(models[s_][2].state_dict()) 
                    package = {'models': save , 'start_epoch': epoch , 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': i, 'accent_dict': accent_dict, 'version': version_, 'train.log': a, 'audio_conf': audio_conf, 'labels': labels, 'lr':args.lr * (args.learning_anneal**(epoch+1))}
                    torch.save(package, os.path.join(save_folder, f"ckpt_{epoch+1}_{i+1}.pth"))
                    del save

            if i % args.early_val+1 == args.early_val and args.early_val < len(train_sampler):
                if hvd.rank() == 0 :
                    with torch.no_grad():
                        wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,loss_save,labels,eps=0.0000000001)
                    print('Validation Summary Epoch: [{0}]\t'
                            'Average WER {wer:.3f}\t'
                            'Average CER {cer:.3f}\t'
                            'Accuracy {acc_: .3f}\t'
                            'Discriminator accuracy (micro) {acc: .3f}\t'
                            'Discriminator precision (micro) {pre: .3f}\t'
                            'Discriminator recall (micro) {rec: .3f}\t'
                            'Discriminator F1 (micro) {f1: .3f}\t'.format(epoch + 1, wer=wer, cer=cer, acc_ = num/length *100 , acc=micro_accuracy, pre=weighted_precision, rec=weighted_recall, f1=weighted_f1))

                    # Logging to tensorboard.
                    writer.add_scalar('Validation/Average-WER', wer, epoch+1)
                    writer.add_scalar('Validation/Average-CER', cer, epoch+1)
                    writer.add_scalar('Validation/Discriminator-Accuracy', num/length *100, epoch+1)
                    writer.add_scalar('Validation/Discriminator-Precision', weighted_precision, epoch+1)
                    writer.add_scalar('Validation/Discriminator-Recall', weighted_recall, epoch+1)
                    writer.add_scalar('Validation/Discriminator-F1', weighted_f1, epoch+1)
                    
                [i_[0].train() for i_ in models.values()] # putting all the models in training state
            
            
            accents = torch.tensor(accents).to(device)
            
            [m[-1].zero_grad() for m in models.values() if m is not None] #making graidents zero
            p_counter += 1

            with torch.cuda.amp.autocast(enabled=True if args.fp16 else False):
                # Forward pass
                x_, updated_lengths_ = models['preprocessing'][0](inputs.squeeze(dim=1),input_sizes.type(torch.LongTensor).to(device))
                z, updated_lengths = models['encoder'][0](x_, updated_lengths_) # Encoder network
                
                discriminator_out = models['discriminator'][0](grad_reverse(z), updated_lengths) # Discriminator network
                asr_out, asr_out_sizes = models['predictor'][0](z, updated_lengths) # Predictor network
                # Loss                
                discriminator_loss = models['discriminator'][1](discriminator_out, accents) * beta
                p_d_loss = discriminator_loss.item()    
        
                asr_out = asr_out.transpose(0, 1)  # TxNxH
                asr_loss = torch.mean(models['predictor'][1](asr_out.log_softmax(2).float(), targets, asr_out_sizes, target_sizes))  # average the loss by minibatch
                
            loss = asr_loss + discriminator_loss 

            p_loss = loss.item()
            valid_loss, error = check_loss(loss, p_loss)
            if valid_loss:
                scaler.scale(loss).backward()
                for i_ in models.keys():
                    models[i_][-1].synchronize()
                    # if i_ == "discriminator": continue
                    with models[i_][-1].skip_synchronize():
                        scaler.step(models[i_][-1])
                scaler.update()
                p_avg_loss += asr_loss.item()
                p_d_avg_loss += p_d_loss
            else: 
                print(error)
                print("Skipping grad update")
                p_avg_loss += 0.0
                p_d_avg_loss += 0.0
            
            if hvd.rank() == 0:
                 # Logging to tensorboard and train.log.
                writer.add_scalar('Train/Predictor-Avergae-Loss-Cur-Epoch', p_avg_loss/p_counter, len(train_sampler)*epoch+i+1) # Average predictor-loss uptil now in current epoch.
                writer.add_scalar('Train/Discriminator-Avergae-Loss-Cur-Epoch', p_d_avg_loss/p_counter, len(train_sampler)*epoch+i+1) # Average Dummy Disctrimintaor loss uptil now in current epoch.
                if not args.silent: print(f"Epoch: [{epoch+1}][{i+1}/{len(train_sampler)}]\t predictor Loss: {round(p_loss,4)} ({round(p_avg_loss/p_counter,4)})\t dummy_discriminator Loss: {round(p_d_loss,8)} ({round(p_d_avg_loss/p_counter,8)})") 
            
        d_avg_loss /= d_counter
        p_avg_loss /= p_counter
        epoch_time = time.time() - start_epoch_time
        start_iter = 0
        if hvd.rank() == 0:
            print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'D/P average Loss {2}, {3}\t'.format(epoch + 1, epoch_time, round(d_avg_loss,4),round(p_avg_loss,4)))
        
        # anneal lr
        dummy_lr = None
        for i in models:
            for g in models[i][-1].param_groups:
                if dummy_lr is None: dummy_lr = g['lr']
                if g['lr'] >= 1e-5:
                    g['lr'] = g['lr'] * args.learning_anneal
            print(f"Learning rate of {i} annealed to: {g['lr']} from {dummy_lr}")
            dummy_lr = None
        
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

        if hvd.rank() == 0:
            with torch.no_grad():
                wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,labels,mtl=True,eps=0.0000000001)
            
            f"epoch,epoch_time,wer,cer,acc,precision,recall,f1,d_avg_loss,p_avg_loss\n"
            a += f"{epoch},{epoch_time},{wer},{cer},{num/length *100},{weighted_precision},{weighted_recall},{weighted_f1},{d_avg_loss},{p_avg_loss},{args.alpha},{args.beta},{args.gamma}\n"

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
                save = {}
                for s_ in models.keys():
                    save[s_] = []
                    save[s_].append(models[s_][0]) 
                    save[s_].append(models[s_][1]) 
                    save[s_].append(models[s_][2].state_dict()) 
                package = {'models': save , 'start_epoch': epoch+1, 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': None, 'accent_dict': accent_dict, 'version': version_, 'train.log': a, 'audio_conf': audio_conf, 'labels': labels, 'lr':args.lr * (args.learning_anneal**(epoch+1))}
                torch.save(package, os.path.join(save_folder, f"ckpt_final.pth"))
                del save
               
            if args.checkpoint:
                save = {}
                for s_ in models.keys():
                    save[s_] = []
                    save[s_].append(models[s_][0]) 
                    save[s_].append(models[s_][1]) 
                    save[s_].append(models[s_][2].state_dict()) 
                package = {'models': save , 'start_epoch': epoch+1, 'best_wer': best_wer, 'best_cer': best_cer, 'poor_cer_list': poor_cer_list, 'start_iter': None, 'accent_dict': accent_dict, 'version': version_, 'train.log': a, 'audio_conf': audio_conf, 'labels': labels, 'lr':args.lr * (args.learning_anneal**(epoch+1))}
                torch.save(package, os.path.join(save_folder, f"ckpt_{epoch+1}.pth"))
                del save

            # Exiting criteria
            terminate_train = False
            if best_cer is None or best_cer > wer:
                best_cer = wer
                poor_cer_list = []
            else:
                poor_cer_list.append(wer)
                if len(poor_cer_list) >= args.patience:
                    print("Exiting training loop...")
                    terminate_train = True
            if terminate_train:
                break

            d_avg_loss, p_avg_loss, p_d_avg_loss, p_d_avg_loss = 0, 0, 0, 0

    writer.close()

# if (i+1)%500 == 0 and i !=1:
#                 if hvd.rank() == 0: 
#                     with torch.no_grad():
#                         wer, cer, num, length,  weighted_precision, weighted_recall, weighted_f1, class_wise_precision, class_wise_recall, class_wise_f1, micro_accuracy = validation(test_loader, GreedyDecoder, models, args,accent,device,loss_save,labels,eps=0.0000000001)
#                         print('Validation Summary Epoch: [{0}]\t'
#                                 'Average WER {wer:.3f}\t'
#                                 'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
#                 [h_[0].train() for h_ in models.values()] # putting all the models in training state



