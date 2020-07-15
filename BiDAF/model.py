# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/7/15
@function: bidaf model
"""
import tensorflow as tf
from collections import defaultdict

params=defaultdict(

)

class BiDAF:
    """bidirectional attention flow"""
    def __init__(self,params):
        self.params=params


    def build_graph(self,features,labels=None,mode=None):


        pass


    def create_placeholders(self):
        c=tf.placeholder(tf.int32,[None,None],name="c")
        c_len=tf.placeholder(tf.int32,[None],name="c_len")
        q=tf.placeholder(tf.int32,[None,None],name="q")
        q_len=tf.placeholder(tf.int32,[None],name="q_len")
        y1=tf.placeholder(tf.int32,[None,None],name="start_index")
        y2=tf.placeholder(tf.int32,[None,None],name="end_index")
        return c,c_len,q,q_len,y1,y2


    def embedding(self,features,):
        """这里没有使用char level的特征"""
        with tf.variable_scope("embedding"):
            emb=tf.get_variable(name="embedding",
                                shape=[self.params["vocab_size"],self.params["embedding_size"]],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(),
                                trainable=True) # 这里可以使用预训练的向量
            c_emb=tf.nn.embedding_lookup(emb,features["context"])
            q_emb=tf.nn.embedding_lookup(emb,features["question"])
        return c_emb,q_emb



        pass
