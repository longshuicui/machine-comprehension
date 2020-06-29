# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :  2020/06/26
@function:
"""
import tensorflow as tf
import numpy as np

from collections import defaultdict
from layers import *

params=defaultdict(
    batch_size=32,
    passage_length=400,  # context length
    question_length=30,  # question length
    num_heads=1,  # the number of head about self attention
    character_limit=16, # the max number of char
    dimension=128,
    dimension_char=32,
    vocab_size=50000,
    char_vocab_size=100,
    embedding_dropout=0.3,
    residual_dropout=0.2,
    highway_dropout=0.3
)

class QANet(object):
    """QAnet model"""
    def __init__(self,params):
        self.params=params

    def input_placeholders(self):
        # 这里先用占位符
        context=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="context")
        question=tf.placeholder(tf.int32,[None,self.params["question_length"]],name="question")
        context_char=tf.placeholder(tf.int32,[None,self.params["passage_length"],self.params["character_limit"]],name="context_char")
        question_char=tf.placeholder(tf.int32,[None,self.params["question_length"],self.params["character_limit"]],name="question_char")
        y1=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="answer_start_index")
        y2=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="answer_end_index")
        return context, question, context_char, question_char, y1, y2


    def forward(self,features,labels=None,mode=None):
        c_mask=tf.sequence_mask(features["context"],maxlen=self.params["passage_length"],dtype=tf.float32)
        q_mask=tf.sequence_mask(features["question"],maxlen=self.params["question_length"],dtype=tf.float32)
        c_len=tf.reduce_sum(tf.cast(c_mask,tf.int32),axis=1)
        q_len=tf.reduce_sum(tf.cast(q_mask,tf.int32),axis=1)

        # embedding
        context_emb, question_emb=self.embedding(features,mode)
        # highway
        context_emb, question_emb=self.highway_layer(context_emb,question_emb,mode)
        # embedding encoder
        c,q=self.encoder_layer(context_emb,question_emb,c_mask,q_mask,mode)
        # C2Q and Q2C attention from BiDAF
        print(c.shape)
        print(q.shape)












    def embedding(self,features,mode):
        PL=self.params["passage_length"]
        QL=self.params["question_length"]
        dropout= self.params["embedding_dropout"] if mode=="train" else 0.0

        with tf.variable_scope("embedding"):
            word_mat = tf.get_variable(name="word_mat",
                                       initializer=tf.random_normal_initializer(),
                                       shape=[self.params["vocab_size"],self.params["dimension"]],
                                       dtype=tf.float32)
            char_mat = tf.get_variable(name="char_mat",
                                       initializer=tf.random_normal_initializer(),
                                       shape=[self.params["char_vocab_size"],self.params["dimension_char"]],
                                       dtype=tf.float32)

        # char level embedding
        context_char_emb=tf.reshape(tf.nn.embedding_lookup(char_mat,features["c_char"]),
                                    shape=[-1,self.params["character_limit"],self.params["dimension_char"]])
        question_char_emb=tf.reshape(tf.nn.embedding_lookup(char_mat,features["q_char"]),
                                     shape=[-1, self.params["character_limit"],self.params["dimension_char"]])

        # dropout
        context_char_emb=tf.nn.dropout(context_char_emb,keep_prob=1.0-dropout)
        question_char_emb=tf.nn.dropout(question_char_emb,keep_prob=1.0-dropout)

        # put char embedding through a cnn, concat representations
        context_char_emb=conv(inputs=context_char_emb,
                              output_size=self.params["dimension_char"],
                              bias=True,activation=tf.nn.relu,
                              kernel_size=5,
                              name="char_conv",
                              reuse=None)
        question_char_emb=conv(inputs=question_char_emb,
                               output_size=self.params["dimension_char"],
                               kernel_size=5,
                               name="char_conv",
                               reuse=True)

        context_char_emb=tf.reduce_max(context_char_emb,axis=1)
        question_char_emb=tf.reduce_max(question_char_emb,axis=1)

        context_char_emb=tf.reshape(context_char_emb,shape=[-1,PL,context_char_emb.shape[-1]])
        question_char_emb=tf.reshape(question_char_emb,shape=[-1,QL,question_char_emb.shape[-1]])

        # word level embedding
        context_emb=tf.nn.embedding_lookup(word_mat,features["context"])
        question_emb=tf.nn.embedding_lookup(word_mat,features["question"])

        context_emb=tf.nn.dropout(context_emb, keep_prob=1.0-dropout)
        question_emb=tf.nn.dropout(question_emb, keep_prob=1.0-dropout)

        context_emb=tf.concat([context_emb,context_char_emb],axis=-1)
        question_emb=tf.concat([question_emb,question_char_emb],axis=-1)

        return context_emb,question_emb


    def highway_layer(self,c_emb,q_emb,mode):
        context_emb = highway(c_emb,
                              size=self.params["dimension"],
                              scope="highway",
                              dropout=self.params["highway_dropout"] if mode == "train" else 0.0,
                              reuse=None)
        question_emb = highway(q_emb,
                               size=self.params["dimension"],
                               scope="highway",
                               dropout=self.params["highway_dropout"] if mode == "train" else 0.0,
                               reuse=True)
        return context_emb,question_emb


    def encoder_layer(self,c_emb,q_emb,c_mask,q_mask,mode):
        with tf.variable_scope("embedding_encoder_layer"):
            c=residual_block(c_emb,
                             num_blocks=1,
                             num_conv_layers=4,
                             kernel_size=7,
                             mask=c_mask,
                             num_filters=self.params["dimension"],
                             num_heads=self.params["num_heads"],
                             scope="encoder_residual_block",
                             bias=False,
                             dropout=self.params["residual_dropout"] if mode=="train" else 0.0)
            q=residual_block(q_emb,
                             num_blocks=1,
                             num_conv_layers=4,
                             kernel_size=7,
                             mask=q_mask,
                             num_filters=self.params["dimension"],
                             num_heads=self.params["num_heads"],
                             scope="encoder_residual_block",
                             bias=False,
                             dropout=self.params["residual_dropout"] if mode == "train" else 0.0,
                             reuse=True) # q和c使用相同的网络参数
            return c,q










if __name__ == '__main__':
    model=QANet(params)
    context, question, context_char, question_char, y1, y2=model.input_placeholders()
    features={"context":context,
              "question":question,
              "c_char":context_char,
              "q_char":question_char,
              "y1":y1,
              "y2":y2}
    model.forward(features)

