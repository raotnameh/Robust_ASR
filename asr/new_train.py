import argparse
import json
import os
import random
import time, math
import sys
import traceback

import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
#from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from logger import VisdomLogger, TensorBoardLogger
from new_model import DeepSpeech
from test import evaluate_acc #make a function evaluate accuracy
from utils import reduce_tensor, check_loss, shorten_target

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
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
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
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
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--weights', default='', help='Continue from checkpoint model')
#parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def to_np(x):
    return x.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    args = parser.parse_args()
    args.num_workers = 2
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:"+str(args.gpu_rank) if args.cuda else "cpu")
    torch.cuda.set_device(int(args.gpu_rank))
    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
    loss_results, acc_results = torch.zeros(args.epochs), torch.zeros(args.epochs)
    #print(loss_results)
    best_acc = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = model.labels
        audio_conf = model.audio_conf
        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            loss_results, acc_results = package['loss_results'], package['acc_results']
            best_acc = acc_results[start_epoch]
            if main_proc and args.visdom:  # Add previous scores to visdom graph
                visdom_logger.load_previous_values(start_epoch, package)
            if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values(start_epoch, package)
    else:
        with open(args.labels_path) as label_file:
            labels = json.load(label_file)

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))
    
    model = DeepSpeech(labels=labels,audio_conf=audio_conf)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    try:
        model.load_state_dict(torch.load(args.weights)['state_dict'], strict = False)
        print('using weights')
    except:pass
    model = model.to(device)
    parameters = model.parameters()
    #optimizer = torch.optim.SGD(parameters, lr=args.lr,
     #                           momentum=args.momentum, nesterov=True, weight_decay=1e-4)
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                             weight_decay=1e-4,amsgrad=True)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)
    if args.distributed:
        model = DistributedDataParallel(model)
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    conv_params = model.conv_params

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            try:
                if i == len(train_sampler):
                    break
                inputs, targets, input_percentages, target_sizes, time_durs = data
                #print(time_durs.data)
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                # measure data loading time
                data_time.update(time.time() - end)
                inputs = inputs.to(device)
            
                out, output_sizes = model(inputs, input_sizes)#NxTxH
                out = out.transpose(1, 2)  # NxHxT
                new_timesteps = out.size(2)
                new_targets = []
                #print(target_sizes)
                for idx,size in enumerate(target_sizes.data.cpu().numpy()):
                    new_size = size.item()
                    for key in conv_params:
                        params = conv_params[key]
                        new_size = int((new_size + 2*params['padding'] - params['time_kernel'])/params['stride'] + 1)
                    prev = 0
                    time_dur = time_durs.data.cpu().numpy()[idx]
                    new_target = list(targets.data.numpy()[prev:size.item()])
                    new_target = shorten_target(new_target,new_size,time_dur)
                    new_target += [0]*(new_timesteps-len(new_target))
                    prev = size.item()
                    new_targets.append(new_target)
    
                new_targets = torch.Tensor(new_targets).to(torch.long).to(device)
    
                #change either out or targets to match speech
                #print(output_sizes)
                #print(target_sizes)
                #print(out.size())
                #print(new_targets.size())
                float_out = out.float()  # ensure float32 for loss
                #print(float_out.size())
                loss = criterion(float_out, new_targets).to(device)
                loss = loss / inputs.size(0)  # average the loss by minibatch
                    
                if math.isnan(loss.item()):
                    continue

                if args.distributed:
                    loss = loss.to(device)
                    loss_value = reduce_tensor(loss, args.world_size).item()
                else:
                    loss_value = loss.item()

                    # Check to ensure valid loss was calculated
                valid_loss, error = check_loss(loss, loss_value)
                if valid_loss:
                        #optimizer.zero_grad()
                        # compute gradient
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    if i%1 == 0:
                        print('optimizer step')
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    print(error)
                    print('Skipping grad update')
                    loss_value = 0

                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if not args.silent:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
                if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                    file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                    print("Saving checkpoint model to %s" % file_path)
                    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                    loss_results=loss_results,
                                                    acc_results=acc_results, avg_loss=avg_loss),file_path)
                del loss, out, float_out 

            except Exception as err: print(output_sizes)

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        with torch.no_grad():
            acc, output_data = evaluate_acc(test_loader=test_loader,
                                            device=device,
                                            model=model)
        print(loss_results)
        loss_results[epoch] = avg_loss
        acc_results[epoch] = acc
        print('Validation Summary Epoch: [{0}]\t'
              'Average Accuracy {acc:.3f}\t'.format(epoch + 1, acc=acc))

        values = {
            'loss_results': loss_results,
            'acc_results': acc_results
            }
        #print(values['acc_results'])
        if args.visdom and main_proc:
            visdom_logger.update(epoch, values)
        if args.tensorboard and main_proc:
            tensorboard_logger.update(epoch, values, model.named_parameters())
            values = {
                'Avg Train Loss': avg_loss,
                'Avg ACC': acc
            }

        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth' % (save_folder, epoch + 1)
            try:torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                acc_results=acc_results, avg_loss=avg_loss),file_path)
            except:torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                acc_results=acc_results, avg_loss=avg_loss),file_path)
        # anneal lr
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if main_proc and (best_acc is None or best_acc < acc):
            print("Found better validated model, saving to %s" % args.model_path)
            try:torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                acc_results=acc_results, avg_loss=avg_loss),args.model_path)
            except:torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                acc_results=acc_results, avg_loss=avg_loss),args.model_path)
            best_acc = acc
            avg_loss = 0

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)



