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

file_save = './test.l.dim100.results.pkl'

with open(file_save, 'rb') as f:
    results = pickle.load(f)

success_results = [
    result for result in results if result['success'] == True
]

print "# of samples and success samples are : ", (
    len(results), len(success_results)
)
print "success rate is : ", round(1.0*len(success_results) / len(results), 4)

list_idx_tosee = [
    0, 5, 10, 15, 20
]

for idx_tosee in list_idx_tosee:
    result = success_results[idx_tosee]
    print " "
    print "the reference path is : "
    print result['path_ref']
    print "the generated path is : "
    print result['path_gen']
    print "the destination is : "
    print result['pos_destination']
    print "the current position is : "
    print result['pos_current']
    print " "
