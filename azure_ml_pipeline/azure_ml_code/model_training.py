import tempfile
import os
import argparse
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch

from tsai.data.preparation import SlidingWindowSplitter
from tsai.data.preprocessing import TSStandardize
from tsai.tslearner import TSForecaster, TSRegressor
from tsai.callback.core import EarlyStoppingCallback, SaveModelCallback
from tsai.models.RNN import GRU
import tsai.data.external #check_data()
import tsai.data.validation #get_splits()
import tsai.metrics
# from fastai.learner import *

from azureml.core import Run

def init():
    global args
    
    print('init')
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_dir', required=True)
    arg_parser.add_argument('--regression_window_length', type=int, required=True)
    arg_parser.add_argument('--regression_stride', type=int, required=True)
    arg_parser.add_argument('--regression_horizon', type=int, required=True)
    arg_parser.add_argument('--batch_size', type=int, required=True)
    arg_parser.add_argument('--max_epochs', type=int, required=True)
    arg_parser.add_argument('--arch', required=True)
    arg_parser.add_argument('--hidden_size', type=int, required=True)
    arg_parser.add_argument('--n_layers', type=int, required=True)
    arg_parser.add_argument('--bias', required=True)
    arg_parser.add_argument('--rnn_dropout', type=float, required=True)
    arg_parser.add_argument('--bidirectional', required=True)
    arg_parser.add_argument('--fc_dropout', type=float, required=True)
    arg_parser.add_argument('--min_delta', type=float, required=True)
    arg_parser.add_argument('--patience', type=int, required=True)
    
    args, unknown_args = arg_parser.parse_known_args()

def run(mini_batch, mini_batch_context):
    
    print('tsai version: ', tsai.__version__)
    print('pytorch version: ', torch.__version__)
    
    input_path_train = '/'.join(mini_batch[0].split('/')[0:-1])
    print('input file path (train):', input_path_train)
    
    target = mini_batch_context.partition_key_value['target']
    task = mini_batch_context.partition_key_value['task']
    print('target:', target)
    print('task:', task)
    
    table = pq.read_table(input_path_train)
    df_train = table.to_pandas()
    df_train = df_train.set_index('Time', drop=True)
    df_train = df_train.sort_values(by='Time')
    
    tag_X = [s for s in df_train.columns if s != target]
    tag_y = target
    tags = tag_X + [tag_y]
        
    if task == 'regression':
        
        window_length = args.regression_window_length
        stride = args.regression_stride
        horizon = args.regression_horizon
        Xt_train, yt_train = SlidingWindowSplitter(window_length, stride=stride, horizon=horizon, get_x=list(range(len(tag_X))), get_y=len(tag_X))(df_train[tags])
        
        splits = tsai.data.validation.get_splits(yt_train, valid_size=.1, stratify=False, random_state=123, shuffle=True)
        
        batch_tfms = [TSStandardize(by_var=True, verbose=False)]
        arch = args.arch
        arch_config = {'hidden_size':args.hidden_size, 'n_layers':args.n_layers, 'bias':bool(args.bias), 'rnn_dropout':args.rnn_dropout, 'bidirectional':bool(args.bidirectional), 'fc_dropout':args.fc_dropout}
        bs = args.batch_size
        metrics=[tsai.metrics.mae, tsai.metrics.rmse]
        cbs = [EarlyStoppingCallback(min_delta=args.min_delta, patience=args.patience), SaveModelCallback(min_delta=args.min_delta)]
        
        learn = TSRegressor(Xt_train, yt_train, splits=splits, batch_tfms=batch_tfms, arch=arch, arch_config=arch_config, bs=bs, metrics=metrics, cbs=cbs)
        lr_max = learn.lr_find(start_lr=1e-07, end_lr=1)
        
        learn.fit_one_cycle(n_epoch=args.max_epochs, lr_max=lr_max)
        
        valid_mae = learn.validate()[1]
        valid_mse = learn.validate()[2]
        print('MAE Validation: ', valid_mae)
        print('MSE Validation: ', valid_mse)
        
    learn.export(os.path.join(args.output_dir, task + '_' + target + '_model.tsai'))

    return [target + '|' + task + '|' + str(valid_mse) for _ in mini_batch]