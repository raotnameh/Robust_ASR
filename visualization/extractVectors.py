import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from os import listdir
import os
import sys

parser = argparse.ArgumentParser(description='Extract the embedding vectors')
parser.add_argument('--path', help='path to torch embedding vector .pth file')
parser.add_argument('--save', help='path to save the npy files')



def extract2(z, ml, batch_sz):
    
    '''
    This extraction routine iterates thorugh all the embeddings
    and cuts the length in excess of the global minimumm length
    and appends as new embedding/example having same label

    params:
        input:
            z : input embeddings
            ml: global minimum length
            batch_sz : batch size with which the model was saved
        
        outputs:
            z_m : updated embeddings
            y: label of all embeddings
            
    '''

    y = []
    i = 0
    
    #loop through each batch
    for _ in tqdm(z):
        
        if i != 0:
            z_temp = np.array(_[0])
            dim_len = z_temp.shape[3]

            # loop through each example in the batch
            # currently the batch contains 2 elements
            for idx in range(0,batch_sz):
                
                # check if the 4th dim is bigger
                # than our minimum length
                if dim_len > ml:
                    # we floor so that we can cut off
                    # extra after N*ml
                    l = int(np.floor(dim_len/ml))
                    l_d = 0
                    # if bigger then extract chunks "ml"
                    # and keep appending
                    for kk in range(l):
                        z_m = np.concatenate([z_m,z_temp[idx,:,:,l_d:l_d + ml][np.newaxis,:].reshape(1, -1)])
                        l_d = l_d + ml
                        y = y + [_[1][idx]]
                else:
                    # if not bigger then just append
                    z_m = np.concatenate([z_m,z_temp[idx,:,:,:ml][np.newaxis,:].reshape(1, -1)])
                    y = y + [_[1][idx]]

        else:
            z_temp = np.array(_[0])
            
            z_m = z_temp[0,:,:,:ml][np.newaxis,:].reshape(1, -1)
            y = y + [_[1][0]]

            z_m = np.concatenate([z_m,z_temp[0,:,:,:ml][np.newaxis,:].reshape(1, -1)])
            y = y + [_[1][1]]
        

        i = i + 1

    return np.array(z_m), np.array(y)

def extract(z, ml):
    
    '''
    This extraction routine iterates thorugh all the embeddings
    and cuts the length in excess of the global minimumm length

    params:
        input:
            z : input embeddings
            ml: global minimum length
        
        outputs:
            z_m : updated embeddings
            y: label of all embeddings
            
    '''
    y = []
    i = 0
    for _ in tqdm(z):

        if i != 0:
            z_temp = _[0].reshape(_[0].shape[0],-1).cpu()
            z_m = np.concatenate([z_m,z_temp[:,:ml]])
            
        else:
            z_temp = _[0].reshape(_[0].shape[0],-1).cpu()
            z_m = z_temp[:,:ml]
            
        y = y + _[1]

        i = i + 1

    return z_m, np.array(y)


def min_len1(z):

    length = []

    for _ in tqdm(z):

        length.append(_[2][0].item())
        length.append(_[2][1].item())
    
    min_ = np.argmin(np.array(length))

    return length[min_]


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    save_path = args.save

    try:
        z_path = path + '/' + 'z.pth'
        _z_path = path + '/' + 'z_.pth'
        z1 = torch.load(z_path,map_location='cpu')
        z_1 = torch.load(_z_path,map_location='cpu')

        # get batch size

        batch_size = z1[0][0].shape[0]

        min_z = min_len1(z1)
        min_z_ = min_len1(z_1)

        z, y = extract2(z1, min_z, batch_size)
        z_, y_ = extract2(z_1, min_z_, batch_size)
        np.save(save_path+"/feat_z.npy", z)
        np.save(save_path+"/labels_z.npy", y)

        np.save(save_path+"/feat_z_.npy", z_)
        np.save(save_path+"/labels_z_.npy", y_)

        print("Done!" )
    except:
        print("Something wrong! I'm continuing!")
        
