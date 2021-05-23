import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np

class block_B(nn.Module):
    def __init__(self, sub_blocks, kernel_size=11, dilation=1, stride=1, in_channels=32, out_channels=256, dropout=0.2,batch_norm=True,name='pre'):
        super(block_B, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) * dilation
        self.sub_blocks = sub_blocks

        self.layers = nn.ModuleList()
        if sub_blocks == 1:
            self.layers.append(
                nn.Sequential(
                    OrderedDict([
                    (f'conv_{name}',nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, bias=False)),
                    ])
                ) 
            )
            if batch_norm:
                self.layers.append(
                        nn.Sequential(
                            OrderedDict([
                                (f'batchnorm_{name}', nn.BatchNorm1d(out_channels)),
                                (f'relu_{name}', nn.ReLU()),
                                (f'dropout_{name}', nn.Dropout(p=dropout)),
                            ])
                    )
                )
        else:
            for i in range(sub_blocks - 1):
                if i == 0:
                    self.layers.append(
                        nn.Sequential(
                            OrderedDict([
                            (f'conv_{name}',nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, bias=False)),
                            (f'batchnorm_{name}', nn.BatchNorm1d(out_channels)),
                            (f'relu_{name}', nn.ReLU()),
                            (f'dropout_{name}',nn.Dropout(p=dropout)),
                            ])
                        )
                    )
                else: 
                    self.layers.append(
                        nn.Sequential(
                            OrderedDict([
                            (f'conv_{name}',nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, bias=False)),
                            (f'batchnorm_{name}', nn.BatchNorm1d(out_channels)),
                            (f'relu_{name}', nn.ReLU()),
                            (f'dropout_{name}',nn.Dropout(p=dropout)),
                            ])
                        )
                    )

            self.layers.append(
                nn.Sequential(
                    OrderedDict([
                    (f'conv_{name}',nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, bias=False)),
                    (f'batchnorm_{name}', nn.BatchNorm1d(out_channels)),
                    ])
                )
            )
            self.conv = nn.Sequential(
                OrderedDict([
                    (f'conv_{name}', nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)),
                    (f'batchnorm_{name}', nn.BatchNorm1d(out_channels)),
                ])
                ) 
            self.last = nn.Sequential(
                OrderedDict([
                    (f'relu_{name}', nn.ReLU()),
                    (f'dropout_{name}', nn.Dropout(p=dropout)),
                ])
                )       
            # if batch_norm:
            #     self.last = nn.Sequential(
            #     OrderedDict([
            #         (f'relu_{name}', nn.ReLU()),
            #         (f'dropout_{name}', nn.Dropout(p=dropout)),
            #     ])
            #     )
                
            # else: self.last = nn.Sequential()
                
    def forward(self, x, lengths):
        lengths = (lengths/self.stride + 0.5).to(lengths.dtype)
        # y = x # 32,161,148 # (batch_size, channels, seq_len)
        if self.sub_blocks != 1: initial = self.conv(x) 
        for i in range(len(self.layers)):
            # x = self.layers[i](x)
            x = self.maskconv1d(self.layers[i](x), lengths)
        if self.sub_blocks != 1: 
            # x = self.last(x + initial)
            x = self.maskconv1d(self.last(x + initial), lengths)
        # exit()
        return x, lengths

    def maskconv1d(self,x,lengths):
        # print(x.shape)
        max_len = x.size(2)
        idxs = torch.arange(max_len).to(lengths.dtype).to(lengths.device).expand(len(lengths), max_len)
        mask = idxs >= lengths.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        # print(mask.shape)
        # print(x.shape)
        
        del mask, idxs, max_len 
        return x

class block_Deco(nn.Module):
    def __init__(self, sub_blocks, kernel_size=11, dilation=1, stride=1, in_channels=32, out_channels=256, dropout=0.2, nonlinear=2, batch_norm = True):
        super(block_Deco, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = ((kernel_size) // 2) * dilation
        self.sub_blocks = sub_blocks
        self.dilation = dilation
        if self.stride >= 2:
            output_padding = self.stride - 1
        else:
            output_padding = 0

        self.layers = nn.ModuleList()
        if sub_blocks == 1:
            self.layers.append(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, output_padding=output_padding),
            )
            if batch_norm:
              self.layers.append(
                  nn.Sequential(
                      nn.BatchNorm1d(out_channels),
                      nn.ReLU(),
                      nn.Dropout(p=dropout),
                          )
              )
        else:
            for i in range(sub_blocks - 1):
                if i == 0:
                    self.layers.append(
                        nn.Sequential(
                            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, output_padding=output_padding),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                                )
                    )
                else: 
                    self.layers.append(
                        nn.Sequential(
                            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, output_padding=output_padding),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                                )
                    )
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, output_padding=output_padding),
                    nn.BatchNorm1d(out_channels),
                )
            )
            self.conv = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=dilation, output_padding=output_padding),
                nn.BatchNorm1d(out_channels),
                )
            self.last = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropout),
                )

    def forward(self, x, lengths):
        y = x # 32,1,148
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        if self.sub_blocks != 1:
            y = self.last(y + self.conv(x))
        return y, lengths
    
    # def forward(self, x, lengths):
    #     y = x # 32,1,148
    #     for i in range(len(self.layers)):
    #         #y = self.layers[i](y)
    #         y, lengths = self.mask_conv(self.layers[i](y), lengths)
    #     if self.sub_blocks != 1: 
    #         #y = self.last(y + self.conv(x))
    #         y = self.last(y + self.conv(x))
    #         y, lengths = self.mask_conv(y, lengths)

    #     return y, lengths

    def mask_conv(self, x, x_lengths):
        
        # get max size of the time steps
        max_len = x.size(2)

        # create vector with time step indexes
        idxs = torch.arange(max_len, dtype=x_lengths.dtype, device=x_lengths.device)

        # create boolean mask using the tim step indexes
        mask = idxs.expand(x_lengths.size(0), max_len) >= x_lengths.unsqueeze(1)

        # fill mask with batch with 0 for padded time steps
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)

        # get the lengths to pass to the next layer
        x_lengths = self.get_seq_len(x_lengths)
        return x, x_lengths

    def get_seq_len(self, lengths):
        return ((lengths + 2 * self.padding - self.dilation
                * (self.kernel_size - 1) - 1) // self.stride + 1)



class Pre(nn.Module):
    def __init__(self,in_channels,info):
        super(Pre, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_B(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],batch_norm=info[i]['batch_norm'],name='pre',
                    )
            )
            in_channels = info[i]['out_channels']
        
    def forward(self, x, lengths):
        for i in range(len(self.layers)):
            # print(i, "-------",x.shape)
            x, lengths = self.layers[i](x, lengths,)    
        return x, lengths


class Forget(torch.nn.Module):
  def __init__(self,in_channels):
    super(Forget, self).__init__()
    self.linear = torch.randn((in_channels, 1), requires_grad=True)
    
  def forward(self, x):
    print(self.linear)
    return self.linear*x

class Encoder(nn.Module):
    def __init__(self,in_channels,info):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_B(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],batch_norm=info[i]['batch_norm'],name='encoder'
                    )
            )
            in_channels = info[i]['out_channels']

    def forward(self, x, lengths):
        for i in range(len(self.layers)):
            # print(i, "-------",x.shape)
            x, lengths = self.layers[i](x, lengths,)
        return x, lengths 

class Predictor(nn.Module):
    def __init__(self,in_channels,info):
        super(Predictor, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_B(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],batch_norm=info[i]['batch_norm'],name='predictor'
                    )
            )
            in_channels = info[i]['out_channels']

    def forward(self, x, lengths):
        for i in range(len(self.layers)):
            # print(i, "-------",x.shape)
            x, lengths = self.layers[i](x, lengths,)
        return x.permute(0,2,1), lengths # batch_size, seq_length,classes


class disc_last(nn.Module):
    def __init__(self,info,classes,name='disc'):
        super(disc_last, self).__init__()
        self.disc_last = nn.Sequential(
                        OrderedDict([
                            (f'adaptive_{name}',nn.AdaptiveAvgPool1d(8)),
                            (f"flatten_{name}", nn.Flatten()),
                            (f'linear_{name}',torch.nn.Linear(info[-1]['out_channels']*8, classes)),
                        ])
        )

    def forward(self,x,lengths):
        return self.disc_last(x), lengths

class Discriminator(nn.Module):
    def __init__(self,in_channels,info,classes):
        super(Discriminator, self).__init__()

        self.disc = nn.ModuleList()
        for i in range(len(info)):
            self.disc.append(
                block_B(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],batch_norm=info[i]['batch_norm'],name='disc',
                    )
            )
            in_channels = info[i]['out_channels']

        self.disc.append(disc_last(info,classes,name='disc'))

        
    def forward(self, x, lengths):
        for i in range(len(self.disc)):
            x, lengths = self.disc[i](x, lengths,)
        return x # batch_size, classes,seq_length 


class Decoder(nn.Module):
    def __init__(self,in_channels,info):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_Deco(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],batch_norm=info[i]['batch_norm'],
                    )
            )
            in_channels = info[i]['out_channels']

    def forward(self, x, lengths):
        for i in range(len(self.layers)):
            # print(i, "-------",x.shape)
            x, lengths = self.layers[i](x, lengths)

        return x, lengths

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

if __name__ == '__main__':
    
    import json

    device = torch.device("cpu")
    with open('labels.json') as label_file:
        labels = str(''.join(json.load(label_file)))
    from config import *
    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('experiment_1')
    
    pre = Pre(161,configPre()).to(device)
    # print(pre)
    print(f"Number of parameters in preprocessor is: {get_param_size(pre)/1000000}")

    encoder = Encoder(configPre()[-1]['out_channels'],configE()).to(device)
    # print(encoder)
    print(f"Number of parameters in encoder is: {get_param_size(encoder)/1000000}")

    forget_net = Encoder(configPre()[-1]['out_channels'],configFN()).to(device)
    # print(forget_net)
    print(f"Number of parameters in forget net is: {get_param_size(forget_net)/1000000}")

    # decoder = Decoder(configE()[-1]['out_channels'],configDecoder()).to(device)
    # print(decoder)
    # print(f"Number of parameters in decoder is: {get_param_size(decoder)/1000000}")

    discriminator = Discriminator(configFN()[-1]['out_channels'],configDM(),100)
    # print(discriminator)
    print(f"Number of parameters in discriminator is: {get_param_size(discriminator)/1000000}")

    # exit()
    # asr = Encoder(configE()[-1]['out_channels'],configP(),transpose=True).to(device)
    # # print(asr)
    # print(f"Number of parameters in asr is: {get_param_size(asr)/1000000}")

    # exit()
    # for 15 second audios a batch size of 14 works
    B_SZ = 1

    X = torch.rand((B_SZ, 161, 80)).to(device)
    WIDTHS = torch.LongTensor(B_SZ).random_(1600,1610 ).to(device)

    print(f"input shape is {X.shape}")
    Z1, L = pre(X, lengths=WIDTHS)  
    print("PRE OUT", Z1.shape)  
    Z2, L = encoder(Z1, lengths=L)    
    print("ENCODER OUT", Z2.shape)
    # Z, L = decoder(Z2,lengths=L)
    # for i in Z:
    #     for j in i:
    #         print(j)
    # print("DECODER OUT",Z.shape)

    # writer.add_graph(encoder,(Z1,L))
    # writer.close()
    

    M, L_ = forget_net(Z1, lengths=L)    
    print("FORGET-NET OUT", M.shape)
    
    Z_ = Z2 * M

    D, _ = discriminator(Z_, lengths=WIDTHS)    
    
    print("DISCRIMINATOR OUT", D.shape)
    
    # print(Z_.shape)
    # R = decoder(Z_)
    # print("DECODER OUT", R.shape)


    exit()
    P, L_ = asr(Z, lengths=L)    
    print("ASR OUT", P.shape)
    # print(L_)
