# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :  2020/06/27
@function:
"""
import tensorflow as tf
import math


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


def residual_block(inputs,num_blocks,num_conv_layers,kernel_size,mask=None,num_filters=128,input_projection=False,
                   num_heads=8,scope="res_block",reuse=None,bias=True,dropout=0.0):
    """
    residual block
    Args:
        inputs: inputs embedding [batch_size, seq_len, dimension]
        num_blocks: the count of residual block
        num_conv_layers: conv layers in residual block
        kernel_size:
        mask: sequence mask
        num_filters: the cnt of kernel
        input_projection: whether input projection or not
        num_heads: self-attention
        scope: scope
        reuse: context and question will use same network
        bias: bias
        dropout: if train, will use dropout
    Returns:
        outputs:[batch_size, seq_len, dimension]
    """
    with tf.variable_scope(scope,reuse=reuse):
        if input_projection:
            # 输入映射
            inputs=conv(inputs,num_filters,name="input_projection",reuse=reuse)
        outputs=inputs
        sublayer=1
        total_sublayers=(num_conv_layers+2)*num_blocks # 总子层数
        for i in range(num_blocks):
            # 增加时间步， position embedding
            outputs=add_timing_signal_1d(outputs)
            # conv block 包括 layernorm+conv
            outputs,sublayer=conv_block(outputs,
                                        num_conv_layers=num_conv_layers,
                                        kernel_size=kernel_size,
                                        num_filters=num_filters,
                                        scope="encoder_block_%d"%i,
                                        reuse=reuse,
                                        bias=bias,
                                        dropout=dropout,
                                        sublayers=(sublayer,total_sublayers))
            # self-attention block 包括layernorm + self-att
            outputs,sublayer=self_attention_block(outputs,
                                                  num_filters=num_filters,
                                                  mask=mask,
                                                  num_heads=num_heads,
                                                  scope="self_attention_layer_%d"%i,
                                                  reuse=reuse,
                                                  bias=bias,
                                                  dropout=dropout,
                                                  sublayers=(sublayer,total_sublayers))
        return outputs


def self_attention_block(inputs,num_filters,mask=None,num_heads=8,scope="self_attention_ffn",reuse=None,
                         bias=True,dropout=0.0,sublayers=(1,1)):
    with tf.variable_scope(scope,reuse=reuse):
        l,L=sublayers
        # self attention
        outputs=layer_norm(inputs,scope="layer_norm_1",reuse=reuse)
        outputs=tf.nn.dropout(outputs,keep_prob=1.0-dropout)
        outputs=multihead_attention(queries=outputs,
                                    units=num_filters,
                                    num_heads=num_heads,
                                    reuse=reuse,
                                    mask=mask,
                                    bias=bias,
                                    dropout=dropout)
        residual=layer_dropout(outputs,inputs,dropout*float(l)/L)
        l+=1

        # ffn
        outputs=layer_norm(residual,scope="layer_norm_2", reuse=reuse)
        outputs=tf.nn.dropout(outputs,1.0-dropout)
        # 这里也将全连接换成卷积操作
        outputs=conv(outputs,num_filters,bias=True,activation=tf.nn.relu,name="ffn_1",reuse=reuse)
        outputs=conv(outputs,num_filters,bias=True,name="ffn_2",reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l+=1
        return outputs, l


def multihead_attention(queries,units,num_heads,memory=None,scope="multi_head_attention",
                        reuse=None,mask=None,bias=True,dropout=0.0):
    with tf.variable_scope(scope,reuse=reuse):
        if memory is None:
            memory=queries
        # 这里用的卷积代替原先的全连接，提高计算速度和减小参数量
        memory=conv(memory,2*units,name="memory_projection",reuse=reuse)
        query=conv(queries,units,name="query_projection",reuse=reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, 2, axis=2)]
        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head ** -0.5
        x = dot_product_attention(Q, K, V,
                                  bias=bias,
                                  mask=mask,
                                  scope="dot_product_attention",
                                  reuse=reuse, dropout=dropout)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


def conv_block(inputs,num_conv_layers,kernel_size,num_filters,scope="conv_block",reuse=None,bias=True,
               dropout=0.0,sublayers=(1,1)):
    with tf.variable_scope(scope,reuse=reuse):
        outputs=tf.expand_dims(inputs,2)
        l,L=sublayers
        for i in range(num_conv_layers):
            residual=outputs  # origin
            # layer norm
            outputs=layer_norm(outputs,scope="layer_norm_%d"%i,reuse=reuse)
            if i%2 ==0:
                outputs=tf.nn.dropout(outputs,keep_prob=1.0-dropout)
            # conv
            outputs=depthwise_separable_convolution(inputs=outputs,
                                                    kernel_size=(kernel_size,1),
                                                    num_filters=num_filters,
                                                    scope="depthwise_conv_layers_%d"%i,
                                                    bias=bias,
                                                    reuse=reuse)
            outputs=layer_dropout(outputs,residual,dropout*float(l)/L)
            l+=1
        return tf.squeeze(outputs,2), l


def depthwise_separable_convolution(inputs,kernel_size,num_filters,scope="ds_conv",bias=True,reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        shapes=inputs.shape.as_list()
        depthwise_filter=tf.get_variable(name="depthwise_filter",
                                         shape=[kernel_size[0],kernel_size[1],shapes[-1],1],
                                         dtype=tf.float32,
                                         regularizer=regularizer,
                                         initializer=initializer_relu())
        pointwise_filter=tf.get_variable(name="pointwise_filter",
                                         shape=[1,1,shapes[-1],num_filters],
                                         dtype=tf.float32,
                                         regularizer=regularizer,
                                         initializer=initializer_relu())
        outputs=tf.nn.separable_conv2d(inputs,depthwise_filter,pointwise_filter,strides=[1,1,1,1],padding="SAME")
        if bias:
            b=tf.get_variable(name="bias",
                              shape=outputs.shape[-1],
                              regularizer=regularizer,
                              initializer=tf.zeros_initializer())
            outputs+=b
        outputs=tf.nn.relu(outputs)
        return outputs


def layer_norm(x,filters=None,epsilon=1e-6,scope=None,reuse=None):
    if filters is None:
        filters=x.get_shape()[-1]
    with tf.variable_scope(scope,default_name="layer_norm",values=[x],reuse=reuse):
        scale=tf.get_variable(name="layer_norm_scale",shape=[filters],
                              regularizer=regularizer,
                              initializer=tf.ones_initializer())
        bias=tf.get_variable(name="layer_norm_bias",shape=[filters],
                             regularizer=regularizer,
                             initializer=tf.zeros_initializer())
        mean=tf.reduce_mean(x,axis=-1,keep_dims=True)
        variance=tf.reduce_mean(tf.square(x-mean),axis=-1,keep_dims=True)
        norm_x=(x-mean)*tf.rsqrt(variance+epsilon)
        return norm_x*scale+bias


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


def dot_product_attention(q,k,v,bias,mask = None,scope=None,reuse = None,dropout = 0.0):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                    logits.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x  if x is not None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)


def mask_logits(inputs, mask, mask_value = -1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret,[0,2,1,3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret