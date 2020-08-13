import argparse
import json
import os
import random
import time, math

import numpy as np
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

from utils import reduce_tensor, check_loss

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--weights', default='', help='Continue from checkpoint model')


def to_np(x):
    return x.cpu().numpy()



if __name__ == '__main__':
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Gpu setting
    device = torch.device("cuda:"+str(args.gpu_rank) if args.cuda else "cpu")
    torch.cuda.set_device(int(args.gpu_rank))
    
    #Where to save the models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    wer_results = torch.Tensor(args.epochs)
    best_wer = None
    avg_loss, start_epoch = 0, 0
    
    with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                        window_size=args.window_size,
                        window_stride=args.window_stride,
                        window=args.window,
                        noise_dir=args.noise_dir,
                        noise_prob=args.noise_prob,
                        noise_levels=(args.noise_min, args.noise_max))
    
    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    asr = DeepSpeech(rnn_hidden_size=args.hidden_size,
                        nb_layers=args.hidden_layers,
                        labels=labels,
                        rnn_type=supported_rnns[rnn_type],
                        audio_conf=audio_conf,
                        bidirectional=args.bidirectional)
    decoder = GreedyDecoder(labels)

    try:
        package = torch.load(args.weights)
        asr.load_state_dict(package['state_dict'], strict = True)
        best_wer = package['wer_results']
        print('using weights')
    except:pass

    
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    
    #Different modules used with parameters, optimizer and, loss.
    asr = asr.to(device)
    asr_parameters = asr.parameters()
    asr_optimizer = torch.optim.Adam(asr_parameters, lr=args.lr,weight_decay=1e-4,amsgrad=True)
    criterion = CTCLoss()

    print(asr)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(asr))
    
    for epoch in range(start_epoch, args.epochs):
        
        asr.train()
        start_epoch_time = time.time()

        for i, (data) in enumerate(train_loader):
            if i == len(train_sampler):
                break
            
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            
            if math.isnan(loss.item()):
                continue

            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                # compute gradient
                loss.backward()
                if i%2 == 0 and i != 0: #Num of steps to combine for the grad update.
                   print('optimizer step')
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            avg_loss += loss_value
            # measure elapsed time
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t
                      'Loss: {3} ({4})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler),loss_value, avg_loss))

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'Average Loss {2}\t'.format(epoch + 1, epoch_time, avg_loss))
        avg_loss = 0

        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        
        if args.checkpoint:
            file_path = '%s/deepspeech_%d.pth' % (save_folder, epoch + 1)
            try:torch.save(DeepSpeech.serialize(model.module, epoch=epoch,
                                wer_results=wer),file_path)
            except: torch.save(DeepSpeech.serialize(model, epoch=epoch,
                                wer_results=wer),file_path)
        
        # anneal lr
        for g in asr_optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if best_wer is None or best_wer > wer:
            print("Found better validated model, saving to %s" % args.model_path)
            try:torch.save(DeepSpeech.serialize(model.module, epoch=epoch,wer_results=wer),
                       args.model_path)
            except:torch.save(DeepSpeech.serialize(model, epoch=epoch,wer_results=wer),
                       args.model_path)
            best_wer = wer

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
