import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import ExitStack

class ForgetNet(nn.Module):
    def __init__(self):
        super(ForgetNet, self).__init__()
        self.apply_hard_mask = True
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(32)
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

    def forward(self, x, widths):
        x = self.hard_tanh(self.batch_norm_1(self.conv_1(x)))
        x = self.sigmoid(self.batch_norm_2(self.conv_2(x)))
        if self.apply_hard_mask:
            x = self.hard_mask(x, widths)
        return x

class DiscimnateNet(nn.Module):
    def __init__(self):
        super(DiscimnateNet, self).__init__()
        self.conv_1 = nn.Conv2d(32, 64, kernel_size=(21, 11), stride=(1, 1), padding=(10, 5))
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5))
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(5, 11), stride=(1, 1), padding=(2, 5))
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_1 = torch.nn.Linear(1024, 1)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.batch_norm_2 = nn.BatchNorm2d(128)
        self.batch_norm_3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.batch_norm_1(self.conv_1(x)))
        x = self.leaky_relu(self.batch_norm_2(self.conv_2(x)))
        x = self.leaky_relu(self.batch_norm_3(self.conv_3(x)))
        x = self.leaky_relu(self.adaptive_pooling(x))
        x = self.sigmoid(self.linear_1(torch.flatten(x, start_dim=1)))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)

    def forward(self,x):
        x = self.hard_tanh(self.batch_norm_1(self.conv_1(x)))
        x = self.hard_tanh(self.batch_norm_2(self.conv_2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.trans_conv_1 = nn.ConvTranspose2d(32, 32, kernel_size=(21, 11), stride=(2,1), padding = (10,5))
        self.trans_conv_2 = nn.ConvTranspose2d(32, 1, kernel_size=(41, 11), stride=(2,2), padding = (20,5),  output_padding = (0,1))
        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)

    def forward(self,x):
        x = self.hard_tanh(self.trans_conv_1(x))
        x = self.hard_tanh(self.trans_conv_2(x))
        return x

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

    MAX_W = 100 # MAX_W is variable with audio length; 0.2 second, 20 millisecond // 41 (minimum) input to forget
    MIN_W = 41
    H = 81 # H is variable with sampling rate; // 161 - 16khz
    B_SZ = 32

    # Creating random input and widths
    X = torch.rand((B_SZ, 1, H, MAX_W))
    WIDTHS = torch.LongTensor(B_SZ).random_(MIN_W, MAX_W)
    WIDTHS[0] = MAX_W

    print("INPUT", X.shape)
    forget_net = ForgetNet()
    M = forget_net(X, WIDTHS)
    print("MASK OUT", M.shape)

    encoder_net = Encoder()
    Z = encoder_net(X)
    print("ENCODER OUT", Z.shape)

    decoder_net = Decoder()
    X_bar = decoder_net(Z)
    print("DECODER OUT", X_bar.shape)

    Z_bar = Z * M
    discriminator = DiscimnateNet()
    Prob = discriminator(Z_bar)
    print("DISCRIMINATOR OUT", Prob.shape)
