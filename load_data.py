# -*- coding: utf-8 -*-


import numpy as np

def load_data(args):
    data_document = np.load('train.npy', allow_pickle=True).item()
    valid_data = data_document['dev_data']
    train_data = data_document['train_data']
    test_data = data_document['test_data']

    return train_data, valid_data, test_data
