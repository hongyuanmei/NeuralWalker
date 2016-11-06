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
import datetime
import argparse
__author__ = 'Hongyuan Mei'

dtype=theano.config.floatX


#

def main():

    parser = argparse.ArgumentParser(
        description='Trainning model ... '
    )
    #
    '''
    modify here accordingly ...
    '''
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    #
    parser.add_argument(
        '-fd', '--FileData', required=False,
        help='Path of the dataset'
    )
    #
    parser.add_argument(
        '-fp', '--FilePretrain', required=True,
        help='File of pretrained model'
    )
    parser.add_argument(
        '-mt', '--MapTest', required=False,
        help='Test Map'
    )
    #parser.add_argument(
    #    '-sr', '--SaveResults', required=False,
    #    help='Save results ? True or False'
    #)
    parser.add_argument(
        '--saveresults', dest='saveresults',
        action='store_true'
    )
    parser.add_argument(
        '--no-saveresults', dest='saveresults',
        action='store_false'
    )
    parser.set_defaults(saveresults=False)
    #
    args = parser.parse_args()
    #
    #print "args.saveresults : ", args.saveresults
    #
    if args.FileData == None:
        args.FileData = None
    #
    assert(args.FilePretrain != None)
    args.PathPretrain = os.path.abspath(args.PathPretrain)
    if args.MapTest == None:
        args.MapTest = 'l'
    else:
        args.MapTest = str(args.MapTest)
    if args.saveresults == False:
        file_save = None
    else:
        pretrain = args.FilePretrain
        tag_save = pretrain.split('track_')[1].split('/model.pkl')[0]
        file_save = './results/result_' + tag_save + '.pkl'
    #
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    #
    print ("FileData is : %s" % args.FileData )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("MapTest is : %s" % str(args.MapTest) )
    print ("Save Results ? : %s" % file_save )
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'FileData': args.FileData,
        'FilePretrain': args.FilePretrain,
        'MapTest': args.MapTest
    }
    #
    input_tester = {
        'path_rawdata': args.FileData,
        'path_model': args.FilePretrain,
        'args': dict_args,
        'map_test': args.MapTest,
        'file_save': file_save
    }
    #
    run_model.test_model(input_tester)
    #

if __name__ == "__main__": main()
