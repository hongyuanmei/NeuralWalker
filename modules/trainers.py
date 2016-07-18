# -*- coding: utf-8 -*-
"""
the modules for Neural Walker model
we use :
1) bi-directional LSTM as encoder
2) multi-input neural aligner
3) LSTM as decoder
4) deep-output layer

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
import utils
import models
import optimizers

dtype=theano.config.floatX


class NeuralWalkerTrainer(object):
    #
    def __init__(self, model_settings):
        print "building trainer ... "
        self.seq_lang = tensor.ivector(name='seq_lang')
        self.seq_world = tensor.matrix(
            name='seq_world', dtype=dtype
        )
        # shape -- len_path * dim_raw_world_input
        self.seq_action = tensor.ivector(name='seq_action')
        #
        self.model_settings = model_settings
        #
        self.neural_walker = models.NeuralWalker(
            model_settings = self.model_settings
        )
        self.neural_walker.compute_loss(
            self.seq_lang, self.seq_world,
            self.seq_action
        )
        #
        #
        assert(
            self.model_settings['optimizer'] == 'adam' or self.model_settings['optimizer'] == 'sgd'
        )
        if self.model_settings['optimizer'] == 'adam':
            self.optimizer = optimizers.Adam()
        else:
            self.optimizer = optimizers.SGD()
        #
        self.optimizer.compute_updates(
            self.neural_walker.params,
            self.neural_walker.grad_params
        )
        #
        self.model_learn = theano.function(
            inputs = [
                self.seq_lang, self.seq_world,
                self.seq_action
            ],
            outputs = self.neural_walker.cost,
            updates = self.optimizer.updates
        )
        #
        self.model_dev = theano.function(
            inputs = [
                self.seq_lang, self.seq_world,
                self.seq_action
            ],
            outputs = self.neural_walker.cost,
        )
        #
        self.get_model = self.neural_walker.get_model
        self.save_model = self.neural_walker.save_model
