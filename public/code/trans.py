import os
import sys


import pickle
import json
import os.path
from data.data_loader import SpectrogramParser
import torch
from decoder import GreedyDecoder
import argparse

from tqdm import tqdm
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model
from decoder import BeamCTCDecoder
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("fileAddr",help = "The file for which the prediction needs to be made",type= str)
args = parser.parse_args()
prepath = os.getcwd()
device = torch.device("cpu")
half = False
model = load_model(device, prepath+ "/public/models/deepspeech_final.pth", True).type(torch.FloatTensor)

decoder = BeamCTCDecoder(model.labels, lm_path=prepath+"/public/models/lm.binary", alpha=0.47, beta=0.28,
                        beam_width=2048, num_processes=12)
spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

def transcribe(audio_path, spect_parser, model, decoder, device, use_half):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    with open("out.txt","wb") as f:
        pickle.dump(out.cpu().detach().numpy(),f)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets


def trans_sopi(
    audio_path="/media/data_dump/hemant/rachit/audioRecordJS/public/uploads/h.wav"):
    decoded_output, decoded_offsets = transcribe(audio_path=audio_path,
                                                spect_parser=spect_parser,
                                                model=model,
                                                decoder=decoder,
                                                device=device,
                                                use_half=half)

    return decoded_output[0][0]

if __name__ == "__main__":
    f = prepath + args.fileAddr 
    print(trans_sopi(f))
