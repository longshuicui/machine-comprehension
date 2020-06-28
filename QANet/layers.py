# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :  2020/06/27
@function:
"""
import tensorflow as tf


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def conv(inputs,output_size,bias=None,activation=None,kernel_size=1,name="conv",reuse=None):
    with tf.variable_scope(name,reuse=reuse):
        shapes=inputs.shape.as_list()
        if len(shapes)>4:
            raise NotImplementedError
        elif len(shapes)==4:
            filter_shape=[1,kernel_size,shapes[-1],output_size]
            bias_shape=[1,1,1,output_size]
            strides=[1,1,1,1]
        else:
            filter_shape=[kernel_size,shapes[-1],output_size]
            bias_shape=[1,1,output_size]
            strides=1

        conv_func=tf.nn.conv1d if len(shapes)==3 else tf.nn.conv2d

        kernel_=tf.get_variable(name="kernel_",shape=filter_shape,dtype=tf.float32,regularizer=regularizer,
                                initializer=initializer_relu() if activation is not None else initializer())
        outputs=conv_func(inputs,kernel_,strides,"VALID")

        if bias:
            bias_=tf.get_variable(name="bias_",shape=bias_shape,regularizer=regularizer,
                                  initializer=tf.zeros_initializer())
            outputs+=bias_

        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x,size=None,activation=None,num_layers=2,scope="highway",dropout=0.0,reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        if size is None:
            size=x.shape.as_list()[-1]
        else:
            x=conv(x,size,name="input_project",reuse=reuse)

        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)

        return x