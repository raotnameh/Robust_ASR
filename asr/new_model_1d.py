import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SequenceWise(nn.Module): #Same as TimeDistributed
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(1) - length) > 0:
                    mask[i].narrow(1, length, mask[i].size(1) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class DeepSpeech(nn.Module): #Language Recognizer Module
    def __init__(self, labels="abc", audio_conf=None):
        super(DeepSpeech, self).__init__()

        if audio_conf is None:
            audio_conf = {}
        self.version = '0.0.1'
        self.audio_conf = audio_conf or {}
        self.labels = labels

        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)

        self.conv = MaskConv(nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=4, padding=4),#T-->T/2
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=4, padding=4),#T/2-->T/4
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, kernel_size=9, stride=4, padding=4),#T/4-->T/8
            nn.LeakyReLU()
        ))
        #rnn_input_size = int(math.floor(rnn_input_size + 2 * 1 - 3) / 2 + 1) - For Reference
        rnn_input_size = 256
        print('input size for TimeDistributed Dense Layer',rnn_input_size)

        fully_connected = nn.Sequential(
            nn.Linear(rnn_input_size, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes, bias=False)
        )

        self.conv_params = {'conv1':{'time_kernel':9,'stride':4,'padding':4},
                'conv2':{'time_kernel':9,'stride':4,'padding':4},
                'conv3':{'time_kernel':9,'stride':4,'padding':4},}

        self.fc = SequenceWise(fully_connected)

        self.inference_softmax = InferenceBatchSoftmax()
    
    def forward(self,x,lengths):
        lengths = lengths.cpu().int()
        #print(x.shape)i
        output_lengths = self.get_seq_lens(lengths)

        x, _ = self.conv(x, lengths)

        sizes = x.size()
        #print(x.shape)
        #x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        #print(x.shape)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        #x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        #print(x.shape)
        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths
    
    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv1d:
                seq_len = ((seq_len + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) // m.stride[0] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(labels=package['labels'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(labels=package['labels'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,acc_results=None,avg_loss=None, meta=None):
        package = {
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['acc_results'] = acc_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
    
