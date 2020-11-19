import torch
import torch.distributed as dist

from model import DeepSpeech

import os

def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model_components(device, model_path, forget, discriminator, ckpt_id, use_half):
    package = torch.load(model_path, map_location="cpu")
    models = package['models']
    encoder_model, asr_model = models['encoder'][0], models['predictor'][0]
    forget_model = None if 'forget_net' not in models else models['forget_net'][0]
    disc_model = None if 'discrimator' not in models else models['discrimator'][0]

    model_components = [encoder_model, forget_model, disc_model, asr_model]
    for i in range(len(model_components)):
        if model_components[i] is None:
            continue
        model_components[i].eval()
        model_components[i] = model_components[i].to(device)
        if use_half:
            model_components[i] = model_components[i].half()
    return model_components # e, f, d, asr

def load_model(device, model_path, use_half):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


class Decoder_loss():
    def __init__(self,criteria):
        self.loss = criteria

    def forward(self,target, output, WIDTHS,device):

        #create mask for padded instances
        mask = torch.arange(max(WIDTHS)).expand(len(WIDTHS), max(WIDTHS)) < WIDTHS.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = torch.repeat_interleave(mask, target.shape[2], 2).to(device)
    
        #limit output to input shape
        output_inter = output[:,:,:target.shape[2],:target.shape[3]]

        #do element-wise multiplication to zero padded instances
        outputs = output_inter*mask

        #calculate loss
        loss_ = self.loss(outputs, target)

        return loss_
