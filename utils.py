import torch
import torch.distributed as dist

from model import DeepSpeech


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
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model

def decoder_loss(target, output, WIDTHS, criteria):

    #create mask for padded instances
    mask = torch.arange(max(WIDTHS)).expand(len(WIDTHS), max(WIDTHS)) < WIDTHS.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(2)
    mask = torch.repeat_interleave(mask, target.shape[2], 2)

    #limit output to input shape
    output_inter = output[:,:,:target.shape[2],:target.shape[3]]

    #do element-wise multiplication to zero padded instances
    outputs = output_inter*mask

    #calculate loss
    loss_ = criteria(outputs, target)

    return loss_

