import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from contextlib import ExitStack
import math
from collections import OrderedDict


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
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
        :param x: The input of size BxCxDxT
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
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x



class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None,
                 bidirectional=True, context=20):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self.version = '0.0.1'
        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.rnn_type = rnn_type
        self.audio_conf = audio_conf or {}
        self.labels = labels
        self.bidirectional = bidirectional

        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)

        # self.conv = MaskConv(nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True)
        # ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        #print(rnn_input_size)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        #print(rnn_input_size)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        #print(rnn_input_size)
        rnn_input_size *= 32
        #print(rnn_input_size)

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        # print(lengths)
        # output_lengths = self.get_seq_lens(lengths)
        # print(output_lengths)
        # x, _ = self.conv(x, output_lengths)
        # print("shape: ", x.shape)
        output_lengths = lengths

        sizes = x.size()
        #print(x.shape)
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        #print(x.shape)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        #print(x.shape)

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        #exit()
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
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        package = {
            'hidden_size': model.hidden_size,
            'hidden_layers': model.hidden_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'state_dict': model.state_dict(),
            'bidirectional': model.bidirectional,
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
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
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


class ForgetNet(nn.Module):
    def __init__(self, num_modules, residual_bool, hard_mask_bool):
        super(ForgetNet, self).__init__()
        self.residual = residual_bool
        self.apply_hard_mask = hard_mask_bool
        # self.conv_1 = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        # self.conv_2 = nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        # self.batch_norm_1 = nn.BatchNorm2d(32)
        # self.batch_norm_2 = nn.BatchNorm2d(32)
        assert(num_modules >= 1)
        self.modules_list = []
        for i in range(num_modules):
            if i==0:
                stride_1, stride_2, input_channels = (2, 2), (2, 1), 1
                self.modules_list.append(
                nn.ModuleDict({
                'conv_1': nn.Conv2d(input_channels, 32, kernel_size=(41, 11), stride=stride_1, padding=(20, 5)),
                'batch_norm_1': nn.BatchNorm2d(32),
                'conv_2': nn.Conv2d(32, 32, kernel_size=(21, 11), stride=stride_2, padding=(10, 5)),
                'batch_norm_2': nn.BatchNorm2d(32),
                })
                )
            else:
                stride_1, input_channels = 1, 32
                self.modules_list.append(
                nn.ModuleDict({
                'conv_1': nn.Conv1d(input_channels, 32, kernel_size=41, stride=stride_1, padding=20),
                'batch_norm_1': nn.BatchNorm1d(32),
                'batch_norm_2': nn.BatchNorm1d(32),
                })
                )
        self.modules_list = nn.ModuleList(self.modules_list)
        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def hard_mask(self, x, widths):
        '''
            Applies hard-mask to padded regions.
        '''
        widths = (widths + 2 * 5 - 11) // 2 + 1 # widths = (widths + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        widths = (widths + 2 * 5 - 11) // 1 + 1
        mask = torch.zeros((x.shape[0], x.shape[3]+1), dtype=x.dtype, device=x.device)
        mask = 1.0 - torch.cumsum(mask.scatter_(1, widths.reshape(x.shape[0], 1), 1.0), dim=-1)
        mask = mask[:, :-1].view(mask.shape[0], 1, 1, mask.shape[-1]-1).repeat(1, x.shape[1], x.shape[2], 1)
        return mask * x

    def forward(self, input_x, widths):
        # x = self.hard_tanh(self.batch_norm_1(self.conv_1(input_x)))
        # x = self.sigmoid(self.batch_norm_2(self.conv_2(x)))
        for i, module_dict in enumerate(self.modules_list):

            if i == 0:
                x = self.hard_tanh(module_dict['batch_norm_1'](module_dict['conv_1'](input_x)))
                x = module_dict['conv_2'](x)
            else:
                x = self.hard_tanh(module_dict['batch_norm_1'](module_dict['conv_1'](x)))

            if i != 0 and self.residual: # Residual connection
                x = x + input_x

            if (len(self.modules_list) != i+1):
                if i==0:
                    non_lin = self.hard_tanh
                    input_x = x = non_lin(module_dict['batch_norm_2'](x))
                    x_s = x.shape
                    x = torch.reshape(x, (x_s[0], x_s[1], x_s[2]*x_s[3]))
                    input_x_s = input_x.shape
                    input_x = torch.reshape(input_x, (input_x_s[0], input_x_s[1], input_x_s[2]*input_x_s[3]))
                else:
                    non_lin = self.hard_tanh
                    input_x = x = non_lin(module_dict['batch_norm_2'](x))

            else:
                non_lin = self.sigmoid
                input_x = x = non_lin(module_dict['batch_norm_2'](x))


        if self.apply_hard_mask:
            #x = x.unsqueeze(2)
            sizes = x.shape
            x = x.view(sizes[0], sizes[1], 41, -1)
            x = self.hard_mask(x, widths)
        return x

class DiscimnateNet(nn.Module):
    def __init__(self,classes, residual_bool, num_modules=None):
        super(DiscimnateNet, self).__init__()

        self.adaptive_pooling = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_1 = torch.nn.Linear(1024, classes)
        self.sigmoid = nn.Sigmoid()

        self.base = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(21, 11), stride=(1, 1), padding=(10, 5)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=(5, 11), stride=(1, 1), padding=(2, 5)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    )
        
        cnns = []
        for x in range(num_modules):
            cnn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(5, 11), stride=(1, 1), padding=(2, 5)),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2, inplace=True),
                                )
            cnns.append(('%d' % (x + 1), cnn))
        self.cnns = nn.Sequential(OrderedDict(cnns))

        self.residual = residual_bool


    def forward(self, input_x):
        x =  self.base(input_x)
        # print("Shape: "+str(input_x.shape))
        for cnn in self.cnns:
            input_x = cnn(x)
            if self.residual: # Residual connection
                x = x+input_x
            else: 
                x = input_x

        x = self.adaptive_pooling(x)
        x = self.sigmoid(self.linear_1(torch.flatten(x, start_dim=1)))
        return x

class Encoder(nn.Module):
    def __init__(self, num_modules, residual_bool):
        super(Encoder, self).__init__()
        
        assert(num_modules >= 1)
        self.modules_list = []
        for i in range(num_modules):
            if i==0:
                stride_1, stride_2, input_channels = (2, 2), (2, 1), 1
                self.modules_list.append(
                nn.ModuleDict({
                'conv_1': nn.Conv2d(input_channels, 32, kernel_size=(41, 11), stride=stride_1, padding=(20, 5)),
                'batch_norm_1': nn.BatchNorm2d(32),
                'conv_2': nn.Conv2d(32, 32, kernel_size=(21, 11), stride=stride_2, padding=(10, 5)),
                'batch_norm_2': nn.BatchNorm2d(32),
                })
                )
            else:
                stride_1, input_channels = 1, 32
                self.modules_list.append(
                nn.ModuleDict({
                'conv_1': nn.Conv1d(input_channels, 32, kernel_size=41, stride=stride_1, padding=20),
                'batch_norm_1': nn.BatchNorm1d(32),
                'batch_norm_2': nn.BatchNorm1d(32),
                })
                )
        self.modules_list = nn.ModuleList(self.modules_list)
        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)
        self.residual = residual_bool

        
    def forward(self, input_x, lengths):

        for i, module_dict in enumerate(self.modules_list):

            if i==0:
                x = self.hard_tanh(module_dict['batch_norm_1'](module_dict['conv_1'](input_x)))
                x = module_dict['conv_2'](x)
            else:
                x = self.hard_tanh(module_dict['batch_norm_1'](module_dict['conv_1'](x)))

            if i != 0 and self.residual: # Residual connection
                x = x + input_x

            if i==0:
                input_x = x = self.hard_tanh(module_dict['batch_norm_2'](x))
                x_s = x.shape
                x = torch.reshape(x, (x_s[0], x_s[1], x_s[2]*x_s[3]))
                input_x_s = input_x.shape
                input_x = torch.reshape(input_x, (input_x_s[0], input_x_s[1], input_x_s[2]*input_x_s[3]))
            else:
                input_x = x = self.hard_tanh(module_dict['batch_norm_2'](x))

            #input_x = x = self.hard_tanh(x)
            #x_s = x.shape
        sizes = x.shape
        x = x.view(sizes[0], sizes[1], 41, -1)
        return x, self.get_seq_lens(lengths)


    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        # for m in self.conv.modules():
        for m in self.modules_list[0].values():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.trans_conv_1 = nn.ConvTranspose2d(32, 32, kernel_size=(21, 11), stride=(2,1), padding = (10,5), output_padding = (1,0))
        self.trans_conv_2 = nn.ConvTranspose2d(32, 1, kernel_size=(41, 11), stride=(2,2), padding = (20,5),  output_padding = (1,1))
        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)

    def forward(self,x):
        x = self.hard_tanh(self.trans_conv_1(x))
        x = self.hard_tanh(self.trans_conv_2(x))
        return x

##################################################
##################################################

class Block(nn.Module):

    def __init__(self, kernel = None, stride = 1, dilation = 1, out_ch = None, inp_ch = None,
                dropout_ = None, num_sub_blocks = None):
        super(Block, self).__init__()

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.out_ch = out_ch
        self.dropout_ = dropout_
        self.num_sub_blocks = num_sub_blocks

        # sub-blocks
        assert(num_sub_blocks - 1 >= 1)
        self.modules_list = []
        for i in range(num_sub_blocks - 1):

            if i==0:
                self.modules_list.append(
                        nn.ModuleDict({
                            'conv': nn.Conv1d(inp_ch, out_ch, kernel_size=kernel, stride=stride, padding = 5),
                            'batch_norm': nn.BatchNorm1d(out_ch),
                            'relu': nn.ReLU(0.2),
                            'dropout': nn.Dropout(dropout_),
                        }))
            else:
                self.modules_list.append(
                        nn.ModuleDict({
                            'conv': nn.Conv1d(out_ch, out_ch, kernel_size=kernel, stride=stride, padding = 5),
                            'batch_norm': nn.BatchNorm1d(out_ch),
                            'relu': nn.ReLU(0.2),
                            'dropout': nn.Dropout(dropout_),
                        }))
        self.modules_list = nn.ModuleList(self.modules_list)
        # final sub blocks
        self.convf = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, stride=stride, dilation= dilation, padding = 5)
        self.batchnormf = nn.BatchNorm1d(out_ch)
        self.reluf = nn.ReLU(0.2)
        self.dropoutf = nn.Dropout(dropout_)

        # 1x1 norm
        self.conv11 = nn.Conv1d(inp_ch, out_ch, kernel_size=1)
        self.batchnorm11 = nn.BatchNorm1d(out_ch)

    def forward(self, x):

        inputx = x

        for i, module_dict in enumerate(self.modules_list):

            x = module_dict['dropout'](module_dict['relu'](module_dict['batch_norm'](module_dict['conv'](x))))

        x1 = self.batchnorm11(self.conv11(inputx)) 
        x = self.dropoutf(self.reluf(self.batchnormf(self.convf(x)) + x1))

        return x

class PrePostBlock(nn.Module):

    def __init__(self, kernel = None, stride = 1, dilation = 1, out_ch = None, inp_ch = None,
                dropout_ = None):
        super(PrePostBlock, self).__init__()

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.out_ch = out_ch
        self.dropout_ = dropout_

        self.conv = nn.Conv1d(inp_ch, out_ch, kernel_size=kernel, stride=stride, padding = 5)
        self.batchnorm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(0.2)
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x):

        x = self.dropout(self.relu(self.batchnorm(self.conv(x))))

        return x

class Jasper(nn.Module):
    def __init__(self, labels="abc", num_sub_block=None, num_blocks=None, vocab_size = None, audio_conf=None,
                 out_chs = None):
        super(Jasper, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self.version = '0.0.1'
        self.audio_conf = audio_conf or {}
        self.labels = labels
        self.vocab_size = vocab_size
        self.out_chs = out_chs
        self.num_blocks = num_blocks
        self.num_sub_block = num_sub_block

        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size *= 32

        # Pre-processing block
        self.convPre =  PrePostBlock(kernel = 11, stride = 2, out_ch = 256, inp_ch = input_size, dropout_ = 0.2)

        # Mid blocks
        assert(num_blocks >= 1)
        self.modules_list = []
        prev_ch = 256
        for i in range(num_blocks):

            if i==0:
                self.modules_list.append(
                        nn.ModuleDict({
                            'block': Block(kernel = 11, stride = 1, dilation = 1, out_ch = out_chs[i], inp_ch = prev_ch,
                                           dropout_ = 0.2, num_sub_blocks = num_sub_block)
                        }))
            else:
                self.modules_list.append(
                        nn.ModuleDict({
                            'block': Block(kernel = 11, stride = 1, dilation = 1, out_ch = out_chs[i], inp_ch = prev_ch,
                                           dropout_ = 0.2, num_sub_blocks = num_sub_block)
                        }))
            prev_ch = out_chs[i]

        self.modules_list = nn.ModuleList(self.modules_list)

        # Post blocks
        self.convPos1 =  PrePostBlock(kernel = 11, out_ch = 896, inp_ch = prev_ch, dropout_ = 0.4)

        self.convPos2 =  PrePostBlock(kernel = 1, out_ch = 1024, inp_ch = 896, dropout_ = 0.4)

        self.convPos3 =  nn.Conv1d(1024, vocab_size, kernel_size=1)
       

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = lengths

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        #print(x.shape)
        #x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # Forward pass
        # Pre block
        x = self.convPre(x)
        # Mid blocks
        for i, module_dict in enumerate(self.modules_list):

            x = module_dict['block'](x)

        # Post block
        x = self.convPos1(x)
        x = self.convPos2(x)
        x = self.convPos3(x) 

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
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

    '''@classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        return model'''

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        package = {
            'hidden_size': model.hidden_size,
            'hidden_layers': model.hidden_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'state_dict': model.state_dict(),
            'bidirectional': model.bidirectional,
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
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
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

###############################################
###############################################

class Model(nn.Module):
    def __init__(self):
        self.forget_net = ForgetNet()
        self.discimnate_net = DiscimnateNet()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.predict_net = PredictNet()
        self.train_pred_with_discriminate = False
    
    def forward(self, mode, mini_batch):
        with ExitStack() as stack:
            if not mode.startswith('train'):
                no_grad = torch.no_grad()
            encoded_spect = self.encoder(mini_batch['spect'])
            mask = self.forget_net(mini_batch['spect'], mini_batch['widths'])
            masked_encoded_spect = mask * encoded_spect
        if mode=='train_disc':
            disc_out = self.discimnate_net(masked_encoded_spect)
            if self.train_pred_with_discriminate:
                pred_out = self.predict_net(masked_encoded_spect, mini_batch['widths'])
                return disc_out, pred_out
            return disc_out
        elif mode=='train_rest':
            deco_out = self.decoder(encoded_spect)
            disc_out = self.discimnate_net(masked_encoded_spect)
            pred_out = self.predict_net(masked_encoded_spect, mini_batch['widths'])
            return deco_out, disc_out, pred_out
        else: # for test
            with torch.no_grad():
                pred_out = self.predict_net(masked_encoded_spect, mini_batch['widths'])
            return pred_out

if __name__ == '__main__':

    MAX_W = 130 # MAX_W is variable with audio length; 0.2 second, 20 millisecond // 41 (minimum) input to forget
    MIN_W = 41
    H = 161 # H is variable with sampling rate; // 161 - 16khz
    B_SZ = 24

    # Creating random input and widths
    X = torch.rand((B_SZ, 1, H, MAX_W))
    WIDTHS = torch.LongTensor(B_SZ).random_(MIN_W, MAX_W)
    WIDTHS[0] = MAX_W

    print("INPUT", X.shape)
    forget_net = ForgetNet(8, 1, 1)
    M = forget_net(X, WIDTHS)
    print("FORGET OUT", M.shape)

    encoder_net = Encoder(8, 1)
    Z = encoder_net(X, WIDTHS)
    print("ENCODER OUT", Z[0].shape)
    print("ENCODER OUT", Z[1].shape)

    decoder_net = Decoder()
    X_bar = decoder_net(Z[0])
    print("DECODER OUT", X_bar.shape)

    Z_bar = Z[0] * M
    discriminator = DiscimnateNet(2, 1, 1)
    Prob = discriminator(Z_bar)
    print("DISCRIMINATOR OUT", Prob.shape)

    asr = Jasper(num_sub_block=2, num_blocks=2, vocab_size = 1000, out_chs = [512, 1024])
    output = asr(Z_bar, Z[1])
    print("ASR OUT", output[0].shape)
    print("ASR OUT", output[1].shape)


# if __name__ == '__main__':
#     import os.path
#     import argparse

#     parser = argparse.ArgumentParser(description='DeepSpeech model information')
#     parser.add_argument('--model-path', default='models/deepspeech_final.pth',
#                         help='Path to model file created by training')
#     args = parser.parse_args()
#     package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
#     model = DeepSpeech.load_model(args.model_path)

#     print("Model name:         ", os.path.basename(args.model_path))
#     print("DeepSpeech version: ", model.version)
#     print("")
#     print("Recurrent Neural Network Properties")
#     print("  RNN Type:         ", model.rnn_type.__name__.lower())
#     print("  RNN Layers:       ", model.hidden_layers)
#     print("  RNN Size:         ", model.hidden_size)
#     print("  Classes:          ", len(model.labels))
#     print("")
#     print("Model Features")
#     print("  Labels:           ", model.labels)
#     print("  Sample Rate:      ", model.audio_conf.get("sample_rate", "n/a"))
#     print("  Window Type:      ", model.audio_conf.get("window", "n/a"))
#     print("  Window Size:      ", model.audio_conf.get("window_size", "n/a"))
#     print("  Window Stride:    ", model.audio_conf.get("window_stride", "n/a"))

#     if package.get('loss_results', None) is not None:
#         print("")
#         print("Training Information")
#         epochs = package['epoch']
#         print("  Epochs:           ", epochs)
#         print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
#         print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
#         print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))


