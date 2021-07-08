import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch.optim as optim
import numpy as np
import time
import random
import sys
import pickle
from tqdm.auto import tqdm

from dataloader_initializations import *
from models_original import *
from training import *

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
        return params
for lr in [0.015]:
    log_filename = f'cnn{lr}.txt'
    f = open("grid_search.txt", "w")

    #configuring logger
    MSGFORMAT = '%(name)s -- %(asctime)s :: %(levelname)s : %(message)s'
    DATEFMT = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(level = logging.DEBUG, format = MSGFORMAT, datefmt = DATEFMT, filename = log_filename, filemode = 'w')
    logger = logging.getLogger(__name__)

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    #Log Metadata
    metadata = "CNN based" #model_name + ',' + str(bidirectionality) + ',' + str(emb_size) + ',' + str(num_lstm) + ',' + str(hidden_size)
    logger.info(metadata)

    emb_size = 128
    channels = 128

    #initialize model
    model = CNN(vocab_size, emb_size, channels)        
    print(f"Total number of parameters: {get_param_size(model)/1000000}") 

    opt = optim.Adam(model.parameters(), lr = lr,weight_decay=1e-6,amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)

    print(model)
    logger.info(model)

    max_valid_acc = 0
    max_test_acc = 0
    for n in range(200):

        model, train_acc, train_loss = train_model(train_loader, model, opt, criterion, device)
        valid_acc = test_model(valid_loader, model, opt, criterion, device)
        test_acc = test_model(test_loader, model, opt, criterion, device,"test")

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            max_test_acc = test_acc

        #Log results
        logger.info('Epoch:' + str(n))
        logger.info('Training Accuracy: ' + str(train_acc) + '  --- train loss: ' + str(train_loss))
        logger.info('Validation Accuracy: ' + str(valid_acc)) 
        logger.info('Test Accuracy: ' + str(test_acc))
        logger.info('Max Test Acc = ' + str(max_test_acc) + '\n')

        for g in opt.param_groups:
            if g['lr'] >= 1e-6:
                g['lr'] = g['lr'] * 0.95
        print(f"Learning rate is annealed to: {g['lr']}")
        