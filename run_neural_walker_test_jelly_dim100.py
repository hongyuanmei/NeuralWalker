# -*- coding: utf-8 -*-
"""
run file for neural walker

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.trainers as trainers
import modules.data_processers as data_processers

import run_model

dtype=theano.config.floatX

#
input_trainer = {
    'log_file': './log.test.jelly.dim100.txt',
    'path_rawdata': None,
    #
    'max_epoch': 100,
    'dim_model': 100,
    'path_start_model': None,
    'save_file_path': './test.jelly.dim100.models/',
    'optimizer': 'adam',
    'maps_train': [
        'grid', 'l'
    ]
}

#TODO: start training
run_model.train_model(input_trainer)
