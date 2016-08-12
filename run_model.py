# -*- coding: utf-8 -*-
"""
run the neural walker model

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
import modules.beam_search as beam_search

dtype=theano.config.floatX

#TODO: function to train seq2seq models
def train_model(input_trainer):
    '''
    this function is called to train model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_trainer['random_seed']#12345
    )
    #
    os.system('mkdir -p '+input_trainer['save_file_path'])
    #
    log_dict = {
        'log_file': input_trainer['log_file'],
        'save_file_path': input_trainer['save_file_path'],
        'mode': 'create', 'compile_time': None,
        #
        'max_dev_rate': -1.0,
        #
        'max_epoch': input_trainer['max_epoch'],
        #'size_batch': input_trainer['size_batch'],
        'tracked': {
            'track_cnt': 0,
            'train_loss': None, 'dev_loss': None,
            #
            'dev_rate': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcess(
        path_rawdata=input_trainer['path_rawdata']
    )
    #
    #TODO: build another data process for Greedy search, i.e., gs
    ##
    bs_settings = {
        'size_beam': 1, # greedy search
        'path_model': None,
        'trained_model': None, # trained model will be assigned
        'dim_lang': data_process.dim_lang,
        'map': None
    }
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'dim_lang': data_process.dim_lang,
        'dim_world': data_process.dim_world,
        'dim_action': data_process.dim_action,
        'dim_model': input_trainer['dim_model'],
        'optimizer': input_trainer['optimizer'],
        'drop_out_rate': input_trainer['drop_out_rate']
    }

    trainer = trainers.NeuralWalkerTrainer(
        model_settings = model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        err = 0.0
        num_steps = 0
        #TODO: shuffle the training data and train this epoch
        ##
        train_start = time.time()
        #
        for name_map in input_trainer['maps_train']:
            max_steps = len(
                data_process.dict_data['train'][name_map]
            )
            for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
                data_process.process_one_data(
                    idx_data, name_map, 'train'
                )
                cost_numpy = trainer.model_learn(
                    data_process.seq_lang_numpy,
                    data_process.seq_world_numpy,
                    data_process.seq_action_numpy
                )
                err += cost_numpy
                print "training i-th out of N in map : ", (idx_data, max_steps, name_map)
            #
            num_steps += max_steps
        #
        train_err = err / num_steps
        #
        log_dict['tracked']['track_cnt'] += 1
        log_dict['tracked']['train_loss'] = round(train_err, 3)
        train_end = time.time()
        log_dict['tracked']['train_time'] = round(
            train_end - train_start, 0
        )
        #
        #
        print "validating ... "
        #
        err = 0.0
        num_steps = 0
        dev_start = time.time()
        #
        for name_map in input_trainer['maps_train']:
            max_steps = len(
                data_process.dict_data['dev'][name_map]
            )
            for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
                data_process.process_one_data(
                    idx_data, name_map, 'dev'
                )
                cost_numpy = trainer.model_dev(
                    data_process.seq_lang_numpy,
                    data_process.seq_world_numpy,
                    data_process.seq_action_numpy
                )
                err += cost_numpy
                print "validating i-th out of N in map : ", (idx_data, max_steps, name_map)
            #
            num_steps += max_steps
        #
        dev_err = err / num_steps
        #
        log_dict['tracked']['dev_loss'] = round(dev_err, 3)
        #TODO: get beam search result, beam = 1
        #
        bs_settings['trained_model'] = trainer.get_model()
        #bs = beam_search.BeamSearchNeuralWalker(
        #    bs_settings
        #)
        #
        cnt_success = 0
        num_steps = 0
        #
        for name_map in input_trainer['maps_train']:
            max_steps = len(
                data_process.dict_data['dev'][name_map]
            )
            #
            bs_settings['map'] = data_process.maps[
                data_process.map2idx[name_map]
            ]
            bs = beam_search.BeamSearchNeuralWalker(
                bs_settings
            )
            #
            for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
                data_process.process_one_data(
                    idx_data, name_map, 'dev'
                )
                bs.set_encoder(
                    data_process.seq_lang_numpy,
                    data_process.seq_world_numpy
                )
                pos_start, pos_end = data_process.get_pos(
                    idx_data, name_map, 'dev'
                )
                bs.init_beam(
                    numpy.copy(pos_start), numpy.copy(pos_end)
                )
                bs.search_func()
                #
                if bs.check_pos_end():
                    cnt_success += 1
                #
                bs.refresh_state()
                #
            #
            num_steps += max_steps
            #
        #
        success_rate = round(1.0 * cnt_success / num_steps, 4)
        log_dict['tracked']['dev_rate'] = success_rate
        #
        dev_end = time.time()
        log_dict['tracked']['dev_time'] = round(
            dev_end - dev_start, 0
        )
        #
        #
        if log_dict['tracked']['dev_rate'] > log_dict['max_dev_rate']:
            save_file = log_dict['save_file_path'] + 'model' + str(log_dict['tracked']['track_cnt']) + '.pkl'
            trainer.save_model(save_file)
        #
        data_process.track_log(log_dict)
        #
    print "finish training"
    # function finished
# training finished


def test_model(input_tester):
    '''
    this function is called to test model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(12345)
    #

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcess(
        path_rawdata=input_tester['path_rawdata']
    )
    #
    #TODO: build another data process for Greedy search, i.e., gs
    ##
    bs_settings = {
        'size_beam': 1, # greedy search
        'path_model': input_tester['path_model'],
        'trained_model': None,
        'dim_lang': data_process.dim_lang,
        'map': data_process.maps[
            data_process.map2idx[input_tester['map_test']]
        ]
    }
    #
    #TODO: build the model
    print "building model ... "
    #
    bs = beam_search.BeamSearchNeuralWalker(
        bs_settings
    )
    #
    name_map = input_tester['map_test']
    #
    cnt_success = 0
    num_steps = len(
        data_process.dict_data['dev'][name_map]
    ) + len(
        data_process.dict_data['train'][name_map]
    )
    #
    bs = beam_search.BeamSearchNeuralWalker(
        bs_settings
    )
    #
    bs_results = []
    #
    for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'dev'
        )
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'dev'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        bs.refresh_state()
        #
    #
    #
    for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'train'
        )
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'train'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        #
        bs.refresh_state()
        ##
    #
    #
    success_rate = round(1.0 * cnt_success / num_steps, 4)
    #
    if input_tester['file_save'] != None:
        print "saving results ... "
        assert('.pkl' in input_tester['file_save'])
        with open(input_tester['file_save'], 'wb') as f:
            pickle.dump(bs_results, f)
    else:
        print "No need to save results"
    #
    print "the # of paths in this map is : ", (num_steps, name_map)
    print "the success_rate is : ", success_rate
    #
    print "finish testing !!! "
