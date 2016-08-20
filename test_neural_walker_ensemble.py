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
input_tester = {
    #'log_file': './log.dim100.txt',
    'path_rawdata': None,
    #
    #'max_epoch': 50,
    #'dim_model': 100,
    'set_path_model': [
        './test.grid.dim100.090.0.models/model62.pkl',
        './test.grid.dim100.090.1.models/model16.pkl',
        './test.grid.dim100.090.2.models/model48.pkl',
        './test.grid.dim100.090.3.models/model54.pkl',
        './test.grid.dim100.090.4.models/model26.pkl'
    ],
    'map_test': 'grid',
    'file_save': './test.grid.dim100.090.ensemble.results.pkl'
}

#TODO: start training
run_model.test_model_ensemble(input_tester)
