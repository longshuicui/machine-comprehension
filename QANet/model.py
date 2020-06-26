# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :  2020/06/26
@function:
"""
import tensorflow.compat.v1 as tf
import numpy as np
from collections import defaultdict


params=defaultdict(
    batch_size=32,
    passage_length=400,  # context length
    question_length=30,  # question length
    num_heads=1,  # the number of head about self attention
    character_limit=16, # the max number of char
    dimension=128,
    dimension_char=32,
    vocab_size=50000
)

class QANet(object):
    """QAnet model"""
    def __init__(self,params):
        self.params=params

    def input_placeholders(self):
        context=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="context")
        question=tf.placeholder(tf.int32,[None,self.params["question_length"]],name="question")
        context_char=tf.placeholder(tf.int32,[None,self.params["passage_length"],self.params["character_limit"]],name="context_char")
        question_char=tf.placeholder(tf.int32,[None,self.params["question_length"],self.params["character_limit"]],name="question_char")
        y1=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="answer_start_index")
        y2=tf.placeholder(tf.int32,[None,self.params["passage_length"]],name="answer_end_index")
        return context, question, context_char, question_char, y1, y2


    def embedding(self):
        with tf.variable_scope("embedding"):
            word_mat = tf.get_variable(name="word_mat",
                                       initializer=tf.random_normal_initializer(),
                                       shape=[self.params["vocab_size"],self.params["dimension"]],
                                       dtype=tf.float32)




if __name__ == '__main__':
    model=QANet(params)
    print(model.embedding())
    print(1)
