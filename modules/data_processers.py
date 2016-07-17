# -*- coding: utf-8 -*-
"""
data_processers

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

dtype=theano.config.floatX

class DataProcess(object):
    '''
    this class process raw data into the model-friendly format
    and save them when neccessary
    '''
    def __init__(self, path_rawdata=None):
        #
        print "initialize the data processer ... "
        if path_rawdata:
            self.path_rawdata = path_rawdata
        else:
            self.path_rawdata = './data/'
        #
        #
        with open(self.path_rawdata+'databag3.pickle','r') as f:
            raw_data = pickle.load(f)
        with open(self.path_rawdata+'valselect.pickle', 'r') as f:
            devset = pickle.load(f)
        with open(self.path_rawdata+'stat.pickle', 'r') as f:
            stats = pickle.load(f)
        with open(self.path_rawdata+'mapscap1000.pickle', 'r') as f:
            self.maps = pickle.load(f)
            # maps is a list
        #
        self.lang2idx = stats['word2ind']
        self.dim_lang = stats['volsize']
        #
        self.dim_world = 78
        self.dim_action = 4
        # pre-defined world representations
        # 6 + 3 * (8+3+6+1) = 78
        #
        #self.names_map = raw_data.keys()
        self.names_map = ['grid', 'jelly', 'l']
        '''
        names_map should be ['grid', 'jelly', 'l']
        '''
        #
        self.dict_data = {
            'train': {},
            'dev': {}
        }
        #
        for name_map in self.names_map:
            self.dict_data['train'][name_map] = []
            self.dict_data['dev'][name_map] = []
            for idx_data, data in enumerate(raw_data[name_map]):
                if idx_data in devset[name_map]:
                    self.dict_data['dev'][name_map].append(data)
                else:
                    self.dict_data['train'][name_map].append(data)
        #
        self.map2idx = {
            'grid': 0, 'jelly': 1, 'l': 2
        }
        self.idx2map = {
            0: 'grid', 1: 'jelly', 2: 'l'
        }
        #
        self.seq_lang_numpy = None
        self.seq_world_numpy = None
        self.seq_action_numpy = None
        #

    #
    def get_left_and_right(self, direc_current):
        # direc_current can be 0 , 90, 180, 270
        # it is the current facing direction
        assert(direc_current == 0 or direc_current == 90 or direc_current == 180 or direc_current == 270)
        left = direc_current - 90
        if left == -90:
            left = 270
        right = direc_current + 90
        if right == 360:
            right = 0
        behind = direc_current + 180
        if behind == 360:
            behind = 0
        elif behind == 450:
            behind = 90
        else:
            print "impossible direction !!! "
        return left, right, behind


    #
    def get_pos(self, idx_data, name_map, tag_split):
        one_data = self.dict_data[tag_split][name_map][idx_data]
        path_one_data = one_data['cleanpath']
        return path_one_data[0], path_one_data[-1]

    #
    def process_one_data(self, idx_data, name_map, tag_split):
        # process the data with id = idx_data
        # in map[name_map]
        # with tag = tag_split, i.e., 'train' or 'dev'
        one_data = self.dict_data[tag_split][name_map][idx_data]
        list_word_idx = [
            self.lang2idx[w] for w in one_data['instruction'] if w in self.lang2idx
        ]
        self.seq_lang_numpy = numpy.array(
            list_word_idx, dtype=numpy.int32
        )
        # seq_lang finished
        #
        self.seq_world_numpy = numpy.zeros(
            (len(one_data['cleanpath']), self.dim_world),
            dtype=dtype
        )
        idx_map = self.map2idx[
            one_data['map'].lower()
        ]
        nodes = self.maps[idx_map]['nodes']
        for idx_pos, pos in enumerate(one_data['cleanpath']):
            x_current, y_current, direc_current = pos[0], pos[1], pos[2]
            #
            count_pos_found = 0
            #
            for idx_node, node in enumerate(nodes):
                if node['x'] == x_current and node['y'] == y_current:
                    # find this position in the map
                    # so we can get its feature
                    count_pos_found += 1
                    #
                    left_current, right_current, behind_current = self.get_left_and_right(
                        direc_current
                    )
                    '''
                    note:
                    for node, we keep it as [0,..,1,..,0] one hoc
                    but for all directions, the last entry of feature tags if this way is walkable:
                    1 -- it is blocked
                    '''
                    feat_node = numpy.cast[dtype](
                        node['objvec']
                    )
                    feat_forward = numpy.cast[dtype](
                        node['capfeat'][direc_current]
                    )
                    feat_left = numpy.cast[dtype](
                        node['capfeat'][left_current]
                    )
                    feat_right = numpy.cast[dtype](
                        node['capfeat'][right_current]
                    )
                    feat_behind = numpy.cast[dtype](
                        node['capfeat'][behind_current]
                    )
                    self.seq_world_numpy[idx_pos, :] = numpy.copy(
                        numpy.concatenate(
                            (feat_node, feat_forward, feat_left, feat_right, feat_behind),
                            axis=0
                        )
                    )
            assert(count_pos_found > 0)
            # have to find this position in this map
        #
        self.seq_action_numpy = numpy.zeros(
            (len(one_data['action']), ),
            dtype=numpy.int32
        )
        for idx_action, one_hot_vec_action in enumerate(one_data['action']):
            self.seq_action_numpy[idx_action] = numpy.argmax(
                one_hot_vec_action
            )
        # finished processing !
    #
    #
    def creat_log(self, log_dict):
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        print "creating training log file ... "
        with open(log_dict['log_file'], 'w') as f:
            f.write('This the training log file. \n')
            f.write('It tracks some statistics in the training process ... ')
            f.write('Before training, the compilation time is '+str(log_dict['compile_time'])+' sec ... \n')
            f.write('Things that need to be tracked : \n')
            for the_key in log_dict['tracked']:
                f.write(the_key+' ')
            f.write('\n\n')
        #

    def continue_log(self, log_dict):
        print "continue tracking log ... "
        with open(log_dict['log_file'], 'a') as f:
            for the_key in log_dict['tracked']:
                f.write(the_key+' is '+str(log_dict['tracked'][the_key])+' \n')
            #
            # early_stop applied due to success_rate
            #
            if log_dict['max_dev_rate'] < log_dict['tracked']['dev_rate']:
                f.write('This is a new best model ! \n')
                log_dict['max_dev_rate'] = log_dict['tracked']['dev_rate']
            f.write('\n')

    def track_log(self, log_dict):
        #print "recording training log ... "
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        assert(log_dict['mode']=='create' or log_dict['mode']=='continue')
        if log_dict['mode'] == 'create':
            self.creat_log(log_dict)
        else:
            self.continue_log(log_dict)


#'''

if __name__ == '__main__':
    #
    data_process = DataProcess(
        path_rawdata=None
    )
    data_process.read_rawdata()
    #data_process.preorder_to_tree(26, 'dev')
