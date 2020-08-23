import argparse
import json
import os
import random
import time, math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, accent
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns, ForgetNet, Encoder, Decoder, DiscimnateNet

from utils import reduce_tensor, check_loss, decoder_loss

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
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--weights', default='', help='Continue from checkpoint model')
parser.add_argument('--update-rule', default=2, type=int,
                    help='out classes of the discriminator')


def to_np(x):
    return x.cpu().numpy()



if __name__ == '__main__':
    args = parser.parse_args()
    accent = list(accent.values())

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Gpu setting
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.cuda.set_device(int(args.gpu_rank))

    #Where to save the models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    wer_results = torch.Tensor(args.epochs)
    best_wer = None
    d_avg_loss, p_avg_loss, p_d_avg_loss, start_epoch = 0, 0, 0, 0
    
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
    

    #Creating the ASR model block
    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    #Decoder used for evaluation

    #Try loading the information from the last checkpoint
    try:
        package = torch.load(args.weights)
        asr.load_state_dict(package['state_dict'], strict = True)
        best_wer = package['wer_results']
        print('using weights')
    except:pass

    
    #Creating the dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.augment,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=int(1.5*args.batch_size),
                                  num_workers=args.num_workers)

    if args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    
    #Different modules used with parameters, optimizer and, loss.
    
    #asr
    asr = DeepSpeech(rnn_hidden_size=args.hidden_size,
                        nb_layers=args.hidden_layers,
                        labels=labels,
                        rnn_type=supported_rnns[rnn_type],
                        audio_conf=audio_conf,
                        bidirectional=args.bidirectional)
    asr = asr.to(device)
    asr_optimizer = torch.optim.Adam(asr.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    criterion = CTCLoss()

    # Encoder and Decoder
    encoder = Encoder()
    encoder = encoder.to(device)

    decoder = Decoder()
    decoder =decoder.to(device)
    #dec_loss = 

    ed_optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),
                                     lr=args.lr,weight_decay=1e-4,amsgrad=True)

    # Forget_net
    fnet = ForgetNet()
    fnet = fnet.to(device)
    fnet_optimizer = torch.optim.Adam(fnet.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)

    # Discriminator
    discriminator = DiscimnateNet(classes=len(accent))
    discriminator = discriminator.to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr,weight_decay=1e-4,amsgrad=True)
    dis_loss = nn.CrossEntropyLoss()
    
    # Models
    models = {
                'predictor': [asr, criterion, asr_optimizer],
                'encoder': [encoder, None, None],
                'decoder': [decoder, dec_loss, ed_optimizer],
                'forget_net': [fnet, None, fnet_optimizer],
                'discrimator': [discriminator, dis_loss, discriminator_optimizer],
                }
    [i[-1].zero_grad() for i in models.values() if i[-1] is not None] #making gridents zero

    # Printing the models
    print(nn.Sequential(encoder,
                        fnet,
                        decoder,
                        asr,
                        discriminator
    ))

    # Printing the parameters of all the different modules 
    [print(f"Number of parameters for {i[0]} in Million is: {DeepSpeech.get_param_size(i[1][0])/1000000}") for i in models.items()]
    a = f"epoch,wer,cer,accuracy,d_avg_loss, p_avg_loss\n"
    
    for epoch in range(start_epoch, args.epochs):
        [i[0].train() for i in models.values()] # putting all the models in training state
        start_epoch_time = time.time()
        p_counter, d_counter = 0, 0

        for i, (data) in enumerate(train_loader):
            if i == len(train_sampler):
                break

            # Data loading
            inputs, targets, input_percentages, target_sizes, accents = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)

            if i%(args.update_rule+1) == 0: #updating the discriminator only

                [i[-1].zero_grad() for i in models.values() if i[-1] is not None] #making graidents zero
                accents = torch.tensor(accents).to(device)
                d_counter += 1
                # Forward pass
                z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor).to(device)) # Encoder network
                m = fnet(inputs,input_sizes.type(torch.LongTensor).to(device)) # Forget network
                z_ = z * m # Forget Operation
                discriminator_out = discriminator(z_) # Discriminator network
                # Loss
                discriminator_loss = dis_loss(discriminator_out, accents) 
                d_loss = discriminator_loss.item()
                d_avg_loss += d_loss
                
                discriminator_loss.backward()
                discriminator_optimizer.step()

                #[i[-1].zero_grad() for i in models.values() if i[-1] is not None] #making graidents zero
                print(f"Epoch: [{epoch+1}][{i+1}/{len(train_sampler)}]\t\t\t\t\t Discriminator Loss: {round(d_loss,4)} ({round(d_avg_loss/d_counter,4)})")

            else: #random labels for adversarial learning of the predictor network
                
                [i[-1].zero_grad() for i in models.values() if i[-1] is not None] #making graidents zero
                accents = torch.tensor(random.choices(accent,k=len(accents))).to(device)
                p_counter += 1
                # Forward pass
                z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor).to(device)) # Encoder network
                decoder_out = decoder(z) # Decoder network
                m = fnet(inputs,input_sizes.type(torch.LongTensor).to(device)) # Forget network
                z_ = z * m # Forget Operation
                discriminator_out = discriminator(z_) # Discriminator network
                asr_out, asr_out_sizes = asr(z_, updated_lengths) # Predictor network
                # Loss
                # decoder_loss = decoder_loss(decoder_out,inputs)
                discriminator_loss = dis_loss(discriminator_out, accents)
                p_d_loss = discriminator_loss.item()
                p_d_avg_loss += p_d_loss

                asr_out = asr_out.transpose(0, 1)  # TxNxH
                asr_loss = criterion(asr_out.float(), targets, asr_out_sizes.cpu(), target_sizes)
                asr_loss = asr_loss / updated_lengths.size(0)  # average the loss by minibatch
                loss = asr_loss #+ decoder_loss
                p_loss = loss.item()
                p_avg_loss += p_loss

                discriminator_loss.backward(retain_graph=True)
                ed_optimizer.zero_grad()
                loss.backward()
                ed_optimizer.step()
                asr_optimizer.step()
                fnet_optimizer.step()

                #[i[-1].zero_grad() for i in models.values() if i[-1] is not None] #making graidents zero
                print(f"Epoch: [{epoch+1}][{i+1}/{len(train_sampler)}]\t predictor Loss: {round(p_loss,4)} ({round(p_avg_loss/p_counter,4)})\t dummy_discriminator Loss: {round(p_d_loss,4)} ({round(p_d_avg_loss/p_counter,4)})") 
        
            # if i%10 == 9:
            #        break
        d_avg_loss /= d_counter
        p_avg_loss /= p_counter
        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {1}\t'
              'D/P average Loss {2}, {3}\t'.format(epoch + 1, epoch_time, round(d_avg_loss,4),round(p_avg_loss,4)))

        with torch.no_grad():
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            length, num = 0, 0
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                # Data loading
                inputs, targets, input_percentages, target_sizes, accents = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                inputs = inputs.to(device)

                z,updated_lengths = encoder(inputs,input_sizes.type(torch.LongTensor).to(device)) # Encoder network
                m = fnet(inputs,input_sizes.type(torch.LongTensor).to(device)) # Forget network
                z_ = z * m # Forget Operation
                discriminator_out = discriminator(z_) # Discriminator network
                asr_out, asr_out_sizes = asr(z_, updated_lengths) # Predictor network

                # Predictor metric
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size
                target_decoder = GreedyDecoder(labels)
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
     
                # Discriminator metric
                out , predicted = torch.max(discriminator_out, 1)
                for j in range(len(accents)):
                    if accents[j] == predicted[j].item():
                        num = num + 1
                length = length + len(accents)

                # if i%10 == 9:
                #    break

        print('Validation Summary Epoch: [{0}]\t'
                'Average WER {wer:.3f}\t'
                'Average CER {cer:.3f}\t'
                'Discriminator accuracy {acc: .3f}\t'.format(epoch + 1, wer=wer, cer=cer, acc = num/length *100 ))

        
        a += f"{epoch},{wer},{cer},{num/length *100},{d_avg_loss},{p_avg_loss}\n"

        with open("loss.txt", "w") as f:
            f.write(a)

        d_avg_loss, p_avg_loss, p_d_avg_loss = 0, 0, 0
        
        if args.checkpoint:
            for k,v in models.items():
                torch.save(v[0],f"{args.save_folder}{k}_{epoch+1}.pth")

        
        # anneal lr
        for g in asr_optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        # if best_wer is None or best_wer > wer:
        #     print("Found better validated model, saving to %s" % args.model_path)
        #     try:torch.save(DeepSpeech.serialize(model.module, epoch=epoch,wer_results=wer),
        #                args.model_path)
        #     except:torch.save(DeepSpeech.serialize(model, epoch=epoch,wer_results=wer),
        #                args.model_path)
        #     best_wer = wer

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
# Training Summary Epoch: [1]     Time taken (s): 193.08977723121643      D/P average Loss 1.7909, 133.1424
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:41<00:00,  3.45it/s]
# Validation Summary Epoch: [1]   Average WER 1.053       Average CER 0.732       Discriminator accuracy  55.517
# Epoch: [2][1/3862]                                       Discriminator Loss: 1.7652 (1.7652)
