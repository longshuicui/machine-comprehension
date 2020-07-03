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
    attention_dropout=0.2,
    encoder_dropout=0.2,
    highway_dropout=0.3,
    learning_rate=0.01,
    max_gradient=5.0
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
        if mode!=tf.estimator.ModeKeys.TRAIN:
            self.params["embedding_dropout"]=0.0
            self.params["residual_dropout"]=0.0
            self.params["attention_dropout"]=0.0
            self.params["encoder_dropout"]=0.0

        c_mask=tf.cast(tf.cast(features["context"],tf.bool),tf.float32)
        q_mask=tf.cast(tf.cast(features["question"],tf.bool),tf.float32)
        # embedding
        context_emb, question_emb=self.embedding(features)
        # highway
        context_emb, question_emb=self.highway_layer(context_emb,question_emb)
        # embedding encoder
        c,q=self.encoder_layer(context_emb,question_emb,c_mask,q_mask)
        # C2Q and Q2C attention from BiDAF
        attention_outputs=self.context_to_query_attention_layer(c,q,c_mask,q_mask)
        # model encoder
        outputs=self.model_encoder_layer(attention_outputs,c_mask)
        # logits
        start_logits, end_logits=self.output_layer(outputs,c_mask)
        if mode==tf.estimator.ModeKeys.PREDICT:
            return start_logits,end_logits
        loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=start_logits, labels=features["y1"])
        loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=end_logits, labels=features["y2"])
        loss = tf.reduce_mean(loss1 + loss2)
        return start_logits, end_logits, loss


    def embedding(self,features):
        PL=self.params["passage_length"]
        QL=self.params["question_length"]

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
        context_char_emb=tf.nn.dropout(context_char_emb,keep_prob=1.0-self.params["embedding_dropout"])
        question_char_emb=tf.nn.dropout(question_char_emb,keep_prob=1.0-self.params["embedding_dropout"])

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

        context_emb=tf.nn.dropout(context_emb, keep_prob=1.0-self.params["embedding_dropout"])
        question_emb=tf.nn.dropout(question_emb, keep_prob=1.0-self.params["embedding_dropout"])

        context_emb=tf.concat([context_emb,context_char_emb],axis=-1)
        question_emb=tf.concat([question_emb,question_char_emb],axis=-1)

        return context_emb,question_emb


    def highway_layer(self,c_emb,q_emb):
        context_emb = highway(c_emb,
                              size=self.params["dimension"],
                              scope="highway",
                              dropout=self.params["highway_dropout"],
                              reuse=None)
        question_emb = highway(q_emb,
                               size=self.params["dimension"],
                               scope="highway",
                               dropout=self.params["highway_dropout"],
                               reuse=True)
        return context_emb,question_emb


    def encoder_layer(self,c_emb,q_emb,c_mask,q_mask):
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
                             dropout=self.params["residual_dropout"])
            q=residual_block(q_emb,
                             num_blocks=1,
                             num_conv_layers=4,
                             kernel_size=7,
                             mask=q_mask,
                             num_filters=self.params["dimension"],
                             num_heads=self.params["num_heads"],
                             scope="encoder_residual_block",
                             bias=False,
                             dropout=self.params["residual_dropout"],
                             reuse=True) # q和c使用相同的网络参数
            return c,q


    def context_to_query_attention_layer(self,c,q,c_mask,q_mask):
        with tf.variable_scope("context_to_query_attention_layer"):
            S=optimized_trilinear_for_attention([c,q],
                                                c_maxlen=self.params["passage_length"],
                                                q_maxlen=self.params["question_length"],
                                                input_keep_prob=1.0-self.params["attention_dropout"])
            mask_q=tf.expand_dims(q_mask,1)
            S_=tf.nn.softmax(mask_logits(S,mask=mask_q))
            mask_c=tf.expand_dims(c_mask,2)
            S_T=tf.transpose(tf.nn.softmax(mask_logits(S,mask=mask_c),dim=1),[0,2,1])
            c2q=tf.matmul(S_,q)
            q2c=tf.matmul(tf.matmul(S_,S_T),c)
            attention_outputs=tf.concat([c,c2q,c*c2q,c*q2c],axis=-1)
            return attention_outputs


    def model_encoder_layer(self,inputs,c_mask):
        outputs=[conv(inputs,self.params["dimension"],name="input_projection")]
        for i in range(3):
            if i%2 == 0:
                outputs[i]=tf.nn.dropout(outputs[i],keep_prob=1.0-self.params["encoder_dropout"])
            output=residual_block(outputs[i],
                                  num_blocks=7,
                                  num_conv_layers=2,
                                  kernel_size=5,
                                  mask=c_mask,
                                  num_filters=self.params["dimension"],
                                  num_heads=self.params["num_heads"],
                                  scope="model_encoder",
                                  reuse=True if i>0 else None,
                                  bias=False,
                                  dropout=self.params["residual_dropout"])
            outputs.append(output)
        return outputs


    def output_layer(self,model_enc,c_mask):
        with tf.variable_scope("output_layer"):
            # start index
            s_inp=tf.concat([model_enc[1],model_enc[2]],axis=-1)
            start_logits=conv(s_inp,1,bias=False,name="start_pointer")
            start_logits=tf.squeeze(start_logits,axis=-1)
            start_logits=mask_logits(start_logits,mask=c_mask)
            # end index
            e_inp=tf.concat([model_enc[1],model_enc[3]],axis=-1)
            end_logits=conv(e_inp,1,bias=False,name="end_pointer")
            end_logits=tf.squeeze(end_logits,axis=-1)
            end_logits=mask_logits(end_logits,mask=c_mask)

            return start_logits, end_logits


def create_train_op(loss,params):
    global_step=tf.train.get_or_create_global_step()
    lr=tf.minimum(params["learning_rate"],0.001/tf.log(999.)*tf.log(tf.cast(global_step,tf.float32)+1))
    optimizer=tf.train.AdamOptimizer(learning_rate=lr,beta1=0.8,beta2=0.999)
    grads=optimizer.compute_gradients(loss)
    gradients,variables=zip(*grads)
    clip_grads,_=tf.clip_by_global_norm(gradients,params["max_gradient"])
    train_op=optimizer.apply_gradients(zip(clip_grads,variables),global_step=global_step)
    return train_op,global_step






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

    variables=tf.trainable_variables()
    for name,var in variables:
        print(name)

