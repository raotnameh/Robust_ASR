import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import torchaudio
import numpy as np

device = torch.device("cuda:0")


x = torch.from_numpy(s.reshape(1,1,20000)).to(device)


st = time.time()
Spec = torchaudio.transforms.Spectrogram(n_fft=320, win_length=320, hop_length=160).to(device)
spec_out = Spec(x)
spec_out = np.log(spec_out.cpu().numpy()+1)
print(time.time()-st, "seconds.")


import librosa
st = time.time()

D = librosa.stft(s, n_fft=320, hop_length=160,
                 win_length=320)

spect, phase = librosa.magphase(D)

spect = torch.FloatTensor(spect)

mean = spect.mean()
std = spect.std()
spect.add_(-mean)
spect.div_(std)

spect = np.log(spect.numpy()+1)


print(time.time()-st, "seconds.")

print(spec_out[0,0,:,:].shape)
print(spect.shape)
print(spec_out[0,0,:,:]/spect)