import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class block_B(nn.Module):
    def __init__(self, sub_blocks, kernel_size=11, dilation=1, stride=1, in_channels=32, out_channels=256, dropout=0.2, nonlinear=2):
        super(block_B, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) * dilation
        self.sub_blocks = sub_blocks

        self.layers = nn.ModuleList()
        if sub_blocks == 1:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                        )
            )
        else:
            for i in range(sub_blocks - 1):
                if i == 0:
                    self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                            nn.BatchNorm1d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Dropout(p=dropout, inplace=True),
                                )
                    )
                else: 
                    self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                            nn.BatchNorm1d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Dropout(p=dropout, inplace=True),
                                )
                    )
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                )
            )
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                )
            self.last = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                )
        
        self.layers_nonlinear = nn.ModuleList()
        for i in range(nonlinear):
            self.layers_nonlinear.append(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, )
            )

    def forward(self, x, lengths):
        y = x # 32,1,148
        for i in range(len(self.layers)):
            y, lengths = self.maskconv1d(self.layers[i](y), lengths, self.stride)
        if self.sub_blocks != 1: 
            y, lengths = self.maskconv1d(self.last(y + self.conv(x)), lengths, self.stride)
        for i in range(len(self.layers_nonlinear)):
            y, lengths = self.maskconv1d(self.layers_nonlinear[i](y), lengths, stride=1)
        return y, lengths #lengths

    def maskconv1d(self,x,lengths, stride):
        lengths = (lengths/stride + 0.5).to(lengths.dtype)
        max_len = x.size(2)
        idxs = torch.arange(max_len).to(lengths.dtype).to(lengths.device).expand(len(lengths), max_len)
        mask = idxs >= lengths.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        
        del mask, idxs, max_len 
        return x, lengths

class block_Deco(nn.Module):
    def __init__(self, sub_blocks, kernel_size=11, dilation=1, stride=1, in_channels=32, out_channels=256, dropout=0.2, nonlinear=2):
        super(block_Deco, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) * dilation
        self.sub_blocks = sub_blocks

        self.layers = nn.ModuleList()
        if sub_blocks == 1:
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                        )
            )
        else:
            for i in range(sub_blocks - 1):
                if i == 0:
                    self.layers.append(
                        nn.Sequential(
                            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                            nn.BatchNorm1d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Dropout(p=dropout, inplace=True),
                                )
                    )
                else: 
                    self.layers.append(
                        nn.Sequential(
                            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                            nn.BatchNorm1d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Dropout(p=dropout, inplace=True),
                                )
                    )
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(out_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                )
            )
            self.conv = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                )
            self.last = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                )
        
        self.layers_nonlinear = nn.ModuleList()
        for i in range(nonlinear):
            self.layers_nonlinear.append(
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=1, stride=1, )
            )

    def forward(self, x):
        y = x # 32,1,148
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        if self.sub_blocks != 1: 
            y = self.last(y + self.conv(x))
        for i in range(len(self.layers_nonlinear)):
            y = self.layers_nonlinear[i](y)
        return y #lengths


        
class Encoder(nn.Module):
    def __init__(self,info):
        super(Encoder, self).__init__()

        in_channels = 1024

        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_B(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels, nonlinear=0,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],
                    )
            )
            in_channels = info[i]['out_channels']
        
    def forward(self, x, lengths,):
        for i in range(len(self.layers)):
            print(i, "-------",x.shape)
            x, lengths = self.layers[i](x, lengths,)
            
        return x, lengths

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


class Decoder(nn.Module):
    def __init__(self,info):
        super(Decoder, self).__init__()

        in_channels = 512

        self.layers = nn.ModuleList()
        for i in range(len(info)):
            self.layers.append(
                block_Deco(info[i]['sub_blocks'], kernel_size=info[i]['kernel_size'], dilation=info[i]['dilation'],
                    stride=info[i]['stride'], in_channels=in_channels, nonlinear=0,
                    out_channels=info[i]['out_channels'], dropout=info[i]['dropout'],
                    )
            )
            in_channels = info[i]['out_channels']
        
    def forward(self, x,):
        for i in range(len(self.layers)):
            print(i, "-------",x.shape)
            x = self.layers[i](x)
            
        return x


if __name__ == '__main__':

    import json

    device = torch.device("cuda:0")
    with open('labels.json') as label_file:
        labels = str(''.join(json.load(label_file)))
    print(len(labels))
    from config import config, configD

    encoder = Encoder(config(labels)).to(device)
    print(encoder)
    print(f"Number of parameters for in Million is: {get_param_size(encoder)/1000000}")
    # exit()
    B_SZ = 128


    decoder = Decoder(configD(labels)).to(device)
    print(decoder)
    print(f"Number of parameters for in Million is: {get_param_size(decoder)/1000000}")
    # exit()
    B_SZ = 128

    from tqdm.auto import tqdm
    # print(X.shape, WIDTHS)
    #for i in tqdm(range(2) ):
        # Creating random input and widths
    X = torch.rand((B_SZ, 1024, 120)).to(device)
    print("Input Size: ", X.shape)
    WIDTHS = torch.LongTensor(B_SZ).random_(1500,1501 ).to(device)
    Z, L = encoder(X, lengths=WIDTHS)
    Z_ = decoder(Z)

    # print(Z)
    print("ENCODER OUT", Z.shape)
    print("DECODER OUT", Z_.shape)



