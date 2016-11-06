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
    parser.add_argument(
        '-fd', '--FileData', required=False,
        help='Path of the dataset'
    )
    #
    parser.add_argument(
        '-d', '--DimModel', required=False,
        help='Dimension of LSTM model '
    )
    parser.add_argument(
        '-s', '--Seed', required=False,
        help='Seed of random state'
    )
    #
    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', required=False,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-op', '--Optimizer', required=False,
        help='Optimizer of training'
    )
    #
    parser.add_argument(
        '-do', '--DropOut', required=False,
        help='Drop-out rate'
    )
    #
    parser.add_argument(
        '-m1', '--Map1', required=False,
        help='First Train Map'
    )
    parser.add_argument(
        '-m2', '--Map2', required=False,
        help='Second Train Map'
    )
    #
    args = parser.parse_args()
    #
    if args.FileData == None:
        args.FileData = None
    #
    if args.MaxEpoch == None:
        args.MaxEpoch = numpy.int32(2)
    else:
        args.MaxEpoch = numpy.int32(args.MaxEpoch)
    if args.Optimizer == None:
        args.Optimizer = 'adam'
    else:
        args.Optimizer = args.Optimizer
    #
    if args.DimModel == None:
        args.DimModel = numpy.int32(100)
    else:
        args.DimModel = numpy.int32(args.DimModel)
    if args.Seed == None:
        args.Seed = numpy.int32(12345)
    else:
        args.Seed = numpy.int32(args.Seed)
    if args.DropOut == None:
        args.DropOut = numpy.float32(0.9)
    else:
        args.DropOut = numpy.float32(args.DropOut)
    #
    if args.Map1 == None:
        args.Map1 = 'grid'
    else:
        args.Map1 = str(args.Map1)
    if args.Map2 == None:
        args.Map2 = 'jelly'
    else:
        args.Map2 = str(args.Map2)
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    tag_model = '_PID='+str(id_process)+'_TIME='+time_current
    #
    path_track = './tracks/track' + tag_model + '/'
    file_log = os.path.abspath(
        path_track + 'log.txt'
    )
    path_save = path_track
    command_mkdir = 'mkdir -p ' + os.path.abspath(
        path_track
    )
    os.system(command_mkdir)
    #
    ## show values ##
    '''
    '--FileData', '--DimModel', '--Seed'
    '--FilePretrain', '--MaxEpoch', '--Optimizer'
    '--DropOut', '--Map1', '--Map2'
    '''
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    #
    print ("FileData is : %s" % args.FileData )
    print ("DimModel is : %s" % str(args.DimModel) )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("Optimizer is : %s" % args.Optimizer )
    print ("DropOut is : %s" % str(args.DropOut) )
    print ("Map1 is : %s" % str(args.Map1) )
    print ("Map2 is : %s" % str(args.Map2) )
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'FileData': args.FileData,
        'DimModel': args.DimModel,
        'Seed': args.Seed,
        'FilePretrain': args.FilePretrain,
        'MaxEpoch': args.MaxEpoch,
        'Optimizer': args.Optimizer,
        'DropOut': args.DropOut,
        'Map1': args.Map1,
        'Map2': args.Map2
    }
    #
    input_trainer = {
        'random_seed': args.Seed,
        'path_rawdata': args.FileData,
        'path_start_model': args.FilePretrain,
        'drop_out_rate': args.DropOut,
        'max_epoch': args.MaxEpoch,
        'dim_model': args.DimModel,
        'optimizer': args.Optimizer,
        'save_file_path': path_save,
        'log_file': file_log,
        'args': dict_args,
        'maps_train': [
            args.Map1, args.Map2
        ]
    }
    #
    run_model.train_model(input_trainer)
    #

if __name__ == "__main__": main()
