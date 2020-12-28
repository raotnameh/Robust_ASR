import argparse
import json
import os
import random
import time, math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
# from warpctc_pytorch import CTCLoss
from collections import OrderedDict
import pandas as pd

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.data_loader import accent as accent_dict
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns, ForgetNet, Encoder, Decoder, DiscimnateNet

from utils import reduce_tensor, check_loss, Decoder_loss

parser = argparse.ArgumentParser(description='DeepSpeech training')
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
parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('--patience', dest='patience', default=10, type=int, help='Patience epochs.')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
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
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--weights', default='', help='Continue from checkpoint model')
parser.add_argument('--update-rule', default=2, type=int,
                    help='out classes of the discriminator')
parser.add_argument('--train-asr', action='store_true',
                    help='training only the ASR')
parser.add_argument('--dummy', action='store_true',
                    help='do a dummy loop') 
parser.add_argument('--num-epochs', default=1, type=int,
                    help='choosing the number of iterations to train the discriminator in each training iteration')   
parser.add_argument('--mw-alpha', type= float, default= 1,
                    help= 'weight for reconstruction loss')
parser.add_argument('--mw-beta', type= float, default= 1,
                    help= 'weight for discriminator loss')              
parser.add_argument('--mw-gamma', type= float, default= 1,
                    help= 'weight for regularisation')             

parser.add_argument('--exp-name', dest='exp_name', required=True, help='Location to save experiment\'s chekpoints and log-files.')


def to_np(x):
    return x.cpu().numpy()

def setup(rank, world_size):
    print("setting up process!")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')

def cleanup():
    dist.destroy_process_group()

def train_fn(rank, world_size):
    print(f"Running the training on rank : {rank}.")
    setup(rank, world_size)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args()
    accent = list(accent_dict.values())
    if rank==0:
        #Where to save the models and training's metadata
        save_folder = os.path.join(args.exp_name, 'models')
        tbd_logs = os.path.join(args.exp_name, 'tbd_logdir')
        loss_save = os.path.join(args.exp_name, 'train.log')
        config_save = os.path.join(args.exp_name, 'config.json')
        os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

        # save the experiment configuration.
        with open(config_save, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    best_wer, best_cer = None, None
    d_avg_loss, p_avg_loss, p_d_avg_loss, start_epoch = 0, 0, 0, 0
    eps = 0.0000000001 # epsilon value
    start_iter = 0

    with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

    # Creating the configuration apply to the audio
    audio_conf = dict(sample_rate=args.sample_rate,
                        window_size=args.window_size,
                        window_stride=args.window_stride,
                        window=args.window,
                        noise_dir=args.noise_dir,
                        noise_prob=args.noise_prob,
                        noise_levels=(args.noise_min, args.noise_max))

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    models = {} # All the models with their loss and optimizer are saved in this dict

    # Different modules used with parameters, optimizer and loss 

    # ASR : Model
    asr = DeepSpeech(rnn_hidden_size=args.hidden_size,
                        nb_layers=args.hidden_layers,
                        labels=labels,
                        rnn_type=supported_rnns[rnn_type],
                        audio_conf=audio_conf,
                        bidirectional=args.bidirectional)
    asr = asr.to(rank)
    ddp_asr = DDP(asr, device_ids=[rank])

    # ASR : Loss & Optimiser
    asr_optimizer = torch.optim.Adam(asr.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    criterion = nn.CTCLoss()
    models['predictor'] = [asr, criterion, asr_optimizer] 

    # Encoder : Model
    encoder = Encoder(num_modules = args.enco_modules, residual_bool = args.enco_res)
    encoder = encoder.to(rank)
    ddp_encoder = DDP(encoder, device_ids=[rank])

    # Encoder : Loss & Optimiser
    models['encoder'] = [encoder, None, None]

    # Decoder : Model
    decoder = Decoder()
    decoder = decoder.to(rank)
    ddp_decoder = DDP(decoder, device_ids=[rank])

    # Decoder : Loss & Optimiser
    dec_loss = Decoder_loss(nn.MSELoss())
    ed_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),
                                    lr=args.lr,weight_decay=1e-4,amsgrad=True)
    models['decoder'] = [decoder, dec_loss, ed_optimizer] 

    if not args.train_asr:
        # Forget Network : Model
        fnet = ForgetNet(num_modules = args.forg_modules, residual_bool = args.forg_res, hard_mask_bool = True)
        fnet = fnet.to(rank)
        ddp_fnet = DDP(fnet, device_ids=[rank])

        # Forget Network : Loss & Optimiser
        fnet_optimizer = torch.optim.Adam(fnet.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
        models['forget_net'] = [fnet, None, fnet_optimizer]

    if not args.train_asr:
        # Discriminator : Model
        discriminator = DiscimnateNet(classes=len(accent),num_modules=args.disc_modules,residual_bool=args.disc_res)
        discriminator = discriminator.to(rank)
        ddp_fnet = DDP(discriminator, device_ids=[rank])

        # Discriminator : Loss & Optmiser
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
        accent_counts = pd.read_csv(args.train_manifest, header=None).iloc[:,[-1]].apply(pd.value_counts).to_dict()
        disc_loss_weights = torch.zeros(len(accent)) + eps
        for accent_type_f in accent_counts:
            if isinstance(accent_counts[accent_type_f], dict):
                for accent_type_in_f in accent_counts[accent_type_f]:
                    if accent_type_in_f in accent_dict:
                        disc_loss_weights[accent_dict[accent_type_in_f]] += accent_counts[accent_type_f][accent_type_in_f]
        disc_loss_weights = torch.sum(disc_loss_weights) / disc_loss_weights     
        dis_loss = nn.CrossEntropyLoss(weight=disc_loss_weights.to(rank))
        models['discrimator'] = [discriminator, dis_loss, discriminator_optimizer] 
        
    print(f"Model for rank {rank}:\n", nn.Sequential(OrderedDict([(k,v[0]) for k,v in models.items()])))

    #Creating the dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    disc_train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size, num_replicas=world_size, rank=rank)
    disc_train_sampler = DistributedBucketingSampler(disc_train_dataset, batch_size=args.batch_size, num_replicas=world_size, rank=rank)

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
    [print(f"Number of parameters for {i[0]} for rank {rank} in Million is: {DeepSpeech.get_param_size(i[1][0])/1000000}") for i in models.items()]
    accent_list = sorted(accent, key=lambda x:accent[x])
    
    if rank==0:
        # Instantiating tensorboard writer.
        writer = SummaryWriter(tbd_logs)

        # Initialising dumped string.
        a = f"epoch,wer,cer,acc,"
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
    print(f"Initial Probability to udpate to the discrimiantor for rank {rank}: {prob}")
    diff = np.array([prob[i] - prob[-1-i] for i in range(len(prob))])
    diff /= len(train_sampler)*args.num_epochs

    # Reading weights for different losses
    alpha = args.mw_alpha
    beta = args.mw_beta
    gamma = args.mw_gamma

    # Train Loop
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
            inputs = inputs.to(rank)

            # if args.num_epochs > epoch: prob -= diff
            # else: prob = prob_ 
            # update_rule = np.random.choice(args.update_rule, 1, p=prob) + 1
            # print(f"Update rule for rank {rank}: {update_rule}")
            update_rule = 2
            d_avg_loss_iter = eps
            
            for k in range(int(update_rule)): #updating the discriminator only  
                d_counter += 1
                [m[-1].zero_grad() for m in models.values() if m[-1] is not None] #making graidents zero

                # Data loading
                try: inputs_, targets_, input_percentages_, target_sizes_, accents_ = next(disc_)
                except:
                    disc_train_sampler.shuffle(start_epoch)
                    disc_ = iter(disc_train_loader)
                    inputs_, targets_, input_percentages_, target_sizes_, accents_ = next(disc_)

                input_sizes_ = input_percentages_.mul_(int(inputs_.size(3))).int()
                inputs_ = inputs_.to(rank)
                accents_ = torch.tensor(accents_).to(rank)
                # Forward pass
                z,updated_lengths = encoder(inputs_,input_sizes_.type(torch.LongTensor)) # Encoder network
                m = fnet(inputs_,input_sizes_.type(torch.LongTensor).to(rank)) # Forget network
                z_ = z * m # Forget Operation
                discriminator_out = discriminator(z_) # Discriminator network
                # Loss
                discriminator_loss = dis_loss(discriminator_out, accents_)
                d_loss = discriminator_loss.item()
                d_avg_loss += d_loss
                d_avg_loss_iter += d_loss
                
                discriminator_loss.backward()
                discriminator_optimizer.step()

                print(f"Rank {rank} Epoch: [{epoch+1}][{i+1,k+1}/{len(train_sampler)}]\t\t\t\t\t Discriminator Loss: {round(d_loss,4)} ({round(d_avg_loss/d_counter,4)})")

            if rank==0:
                # Logging to tensorboard.
                writer.add_scalar('Train/Discriminator-Avergae-Loss-Cur-Epoch', d_avg_loss/d_counter, len(train_sampler)*epoch+i+1) # Discriminator's training loss in the current main - iteration.
                writer.add_scalar('Train/Discriminator-Per-Iteration-Loss', d_avg_loss_iter/(update_rule+eps), len(train_sampler)*epoch+i+1) # Discriminator's training loss in the current main - iteration.
            
            # Random labels for adversarial learning of the predictor network                
            # Shuffling the elements of a list s.t. elements are not same at the same indices
            dummy = [] 
            for acce in accents:
                while True:
                    d = random.randint(0,len(accent)-1)
                    if acce != d:
                        dummy.append(d)
                        break
            accents = torch.tensor(dummy).to(rank)

            [m[-1].zero_grad() for m in models.values() if m[-1] is not None] #making graidents zero
            p_counter += 1
            
            # Forward pass
            z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor)) # Encoder network
            decoder_out = decoder(z) # Decoder network
            m = fnet(inputs,input_sizes.type(torch.LongTensor).to(rank)) # Forget network
            z_ = z * m # Forget Operation
            discriminator_out = discriminator(z_) # Discriminator network
            asr_out, asr_out_sizes = asr(z_, updated_lengths) # Predictor network
            # Loss
            discriminator_loss = dis_loss(discriminator_out, accents) * beta
            p_d_loss = discriminator_loss.item()
            p_d_avg_loss += p_d_loss
            mask_regulariser_loss = (m * (1-m)).mean() * gamma

            asr_out = asr_out.transpose(0, 1)  # TxNxH
            asr_loss = criterion(asr_out, targets, asr_out_sizes, target_sizes)
            asr_loss = asr_loss / updated_lengths.size(0)  # average the loss by minibatch
            decoder_loss = dec_loss.forward(inputs, decoder_out, input_sizes,rank) * alpha
            loss = asr_loss + decoder_loss + mask_regulariser_loss
            p_loss = loss.item()
            p_avg_loss += p_loss

            discriminator_loss.backward(retain_graph=True)
            ed_optimizer.zero_grad()
            loss.backward()
            ed_optimizer.step()
            asr_optimizer.step()
            fnet_optimizer.step()

            if rank==0:
                # Logging to tensorboard and train.log.
                writer.add_scalar('Train/Predictor-Per-Iteration-Loss', p_loss, len(train_sampler)*epoch+i+1) # Predictor-loss in the current iteration.
                writer.add_scalar('Train/Predictor-Avergae-Loss-Cur-Epoch', p_avg_loss/p_counter, len(train_sampler)*epoch+i+1) # Average predictor-loss uptil now in current epoch.
                writer.add_scalar('Train/Dummy-Discriminator-Per-Iteration-Loss', p_d_loss, len(train_sampler)*epoch+i+1) # Dummy Disctrimintaor loss in the current iteration.
                writer.add_scalar('Train/Dummy-Discriminator-Avergae-Loss-Cur-Epoch', p_d_avg_loss/p_counter, len(train_sampler)*epoch+i+1) # Average Dummy Disctrimintaor loss uptil now in current epoch.
            
            print(f"Epoch: [{epoch+1}][{i+1}/{len(train_sampler)}]\t predictor Loss: {round(p_loss,4)} ({round(p_avg_loss/p_counter,4)})\t dummy_discriminator Loss: {round(p_d_loss,4)} ({round(p_d_avg_loss/p_counter,4)})") 

        d_avg_loss /= d_counter
        p_avg_loss /= p_counter
        epoch_time = time.time() - start_epoch_time
        start_iter = 0
        print('Rank {4} Training Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'D/P average Loss {2}, {3}\t'.format(epoch + 1, epoch_time, round(d_avg_loss,4),round(p_avg_loss,4), rank))

        dist.barrier()

        if rank == 0:
            print("Starting Validation on Rank 0...")
            start_ter = 0
            with torch.no_grad():
                total_cer, total_wer, num_tokens, num_chars = eps, eps, eps, eps
                conf_mat = np.ones((len(accent), len(accent)))*eps # ground-truth: dim-0; predicted-truth: dim-1;
                tps, fps, tns, fns = np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps, np.ones((len(accent)))*eps # class-wise TP, FP, TN, FN
                acc_weights = np.ones((len(accent)))*eps
                length, num = eps, eps
                #Decoder used for evaluation
                target_decoder = GreedyDecoder(labels)
                for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    if args.dummy and i%2 == 1: break

                    # Data loading
                    inputs, targets, input_percentages, target_sizes, accents = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                    inputs = inputs.to(rank)
                    
                    # Forward pass
                    if not args.train_asr:
                        z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor)) # Encoder network
                        m = fnet(inputs,input_sizes.type(torch.LongTensor).to(rank)) # Forget network
                        z_ = z * m # Forget Operation
                        discriminator_out = discriminator(z_) # Discriminator network
                        asr_out, asr_out_sizes = asr(z_, updated_lengths) # Predictor network
                    else:
                        z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor)) # Encoder network
                        decoder_out = decoder(z) # Decoder network
                        asr_out, asr_out_sizes = asr(z, updated_lengths) # Predictor network

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
        
                    if not args.train_asr:
                        # Discriminator metrics: fill in the confusion matrix.
                        out, predicted = torch.max(discriminator_out, 1)
                        for j in range(len(accents)):
                            acc_weights[accents[j]] += 1
                            if accents[j] == predicted[j].item():
                                num = num + 1
                            conf_mat[accents[j], predicted[j].item()] += 1
                        length = length + len(accents)

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

            
            a += f"{epoch},{wer},{cer},{num/length *100},"
            
            for idx, accent_type in enumerate(accent_list):
                a += f"{class_wise_precision[idx]},"
            for idx, accent_type in enumerate(accent_list):
                a += f"{class_wise_recall[idx]},"
            for idx, accent_type in enumerate(accent_list):
                a += f"{class_wise_f1[idx]},"
            a += f"{d_avg_loss},{p_avg_loss},{alpha},{beta},{gamma}\n"

            with open(loss_save, "w") as f:
                f.write(a)

            d_avg_loss, p_avg_loss, p_d_avg_loss = 0, 0, 0

            package = {'models': models}
            torch.save(package, os.path.join(save_folder, f"ckpt_{epoch+1}.pth"))

        dist.barrier() # to make sure that processes with rank!=0 start after validation.

        # Annealing learning rate.
        for j in models.values():
            if j[-1]:
                for g in j[-1].param_groups:
                    g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to on rank {}: {lr:.6f}'.format(rank, lr=g['lr']))

        # Shuffling batches in train_sampler.
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

    cleanup()

def run_train(function, world_size):
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print("number of gpus available:", n_gpus)
    run_train(train_fn, n_gpus)