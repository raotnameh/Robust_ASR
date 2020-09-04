import torch
import torch.distributed as dist
import torch.nn as nn
from model import DeepSpeechRNN


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


def load_model(device, model_path, use_half):
    model = DeepSpeechRNN.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def shorten_target(target,new_size,time_dur):
    ratio = new_size/len(target)
    new_time_dur = time_dur*ratio
    new_target = list(target)
    new_target = new_target[:int(new_time_dur[0])] + \
            new_target[int(time_dur[0]):int(time_dur[0])+int(new_time_dur[1])] + \
            new_target[int(time_dur[1]):int(time_dur[1])+int(new_time_dur[2])]
    while(len(new_target)<new_size):
        new_target.append(new_target[-1])
    return new_target

def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        #print(m.bias)
        #if m.bias is not None and isinstance(m,nn.Conv2d):
        #    torch.nn.init.xavier_uniform_(m.bias)
