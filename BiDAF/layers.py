# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/7/15
@function:
"""
import tensorflow as tf
from tensorflow.contrib import cudnn_rnn

def get_cudnn_cell(rnn_type,hidden_size,layer_num=1,direction="bidirectional",dropout=0.0):
    if rnn_type.endswith("lstm"):
        cell=cudnn_rnn.CudnnLSTM(num_layers=layer_num,num_units=hidden_size,direction=direction,dropout=dropout)

    elif rnn_type.endswith("gru"):
        cell=cudnn_rnn.CudnnGRU(num_layers=layer_num,num_units=hidden_size,direction=direction,dropout=dropout)

    elif rnn_type.endswith("rnn"):
        cell=cudnn_rnn.CudnnRNNTanh(num_layers=layer_num,num_units=hidden_size,direction=direction,dropout=dropout)
    else:
        raise NotImplementedError("Unsuported rnn type: {}".format(rnn_type))
    return cell