# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/7/3
@function: preprocess
"""
import spacy
import json
import random
import logging
import unicodedata
import numpy as np
import tensorflow as tf
from collections import Counter,defaultdict
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

nlp=spacy.blank("en")

def word_tokenizer(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]


def convert_idx(text,tokens):
    current=0
    spans=[]
    for token in tokens:
        current=text.find(token,current)
        if current<0:
            logger.info(f"Token {token} not found")
            raise Exception()
        spans.append((current,current+len(token)))
        current+=len(token)
    return spans


def process_file(filename,data_type,word_counter=None,char_counter=None):
    logger.info(f"Generating {data_type} examples ...")
    examples=[]
    eval_examples={}
    total=0
    source=json.load(open(filename,"r",encoding="utf8"))
    for article in source["data"]:
        for para in article["paragraphs"]:
            context=para["context"].replace("''",'" ').replace("``",'" ')
            context_tokens=word_tokenizer(context)
            context_chars=[list(token) for token in context_tokens]
            spans=convert_idx(context,context_tokens)
            for token in context_tokens:
                word_counter[token]+=len(para["qas"])
                for char in token:
                    char_counter[char]+=len(para["qas"])
            for qa in para["qas"]:
                total+=1
                ques=qa["question"].replace("''",'" ').replace("``",'" ')
                ques_tokens=word_tokenizer(ques)
                ques_chars=[list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token]+=1
                    for char in token:
                        char_counter[char]+=1
                y1s,y2s=[],[]
                answer_texts=[]
                for answer in qa["answers"]:
                    answer_text=answer["text"]
                    answer_start=answer["answer_start"] # char start
                    answer_end=answer_start+len(answer_text) # char end
                    answer_texts.append(answer_text)
                    answer_span=[]
                    for idx, span  in enumerate(spans):
                        if not (answer_end<=span[0] or answer_start>=span[1]):
                            answer_span.append(idx)
                    y1,y2=answer_span[0],answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                # 每个问题对应一个实例
                example={"context_tokens":context_tokens,
                         "context_chars":context_chars,
                         "ques_tokens":ques_tokens,
                         "ques_chars":ques_chars,
                         "y1s":y1s,
                         "y2s":y2s,
                         "id":total}
                examples.append(example)
                eval_examples[str(total)]={"context":context,"spans":spans,"answers":answer_texts,"uuid":qa["id"]}
    random.shuffle(examples)
    logger.info(f"{len(examples)} questions in total.")
    return examples,eval_examples


def get_embedding(counter, data_type, limit=-1,emb_file=None, size=None, vec_size=None):
    logger.info("Generating {} embedding ...".format(data_type))
    embedding_dict={}
    filter_elements=[k for k,v in counter.items() if v>limit]  # 选择词频大于limit的词
    if emb_file is not None:
        pass
    else:
        assert vec_size is not None
        for token in filter_elements:
            embedding_dict[token]=[np.random.normal(scale=0.1) for _ in range(vec_size)]
        logger.info("{} tokens have corresponding embedding vector".format(len(filter_elements)))

    PAD="<PAD>"
    UNK="<UNK>"
    token2idx_dict={token:idx for idx,token in enumerate(embedding_dict.keys(),2)}
    token2idx_dict[PAD]=0
    token2idx_dict[UNK]=1
    embedding_dict[PAD]=[0. for _ in range(vec_size)]
    embedding_dict[UNK]=[0. for _ in range(vec_size)]
    idx2token_dict={idx:token for token,idx in token2idx_dict.items()}
    idx2emb_dict={idx:embedding_dict[token] for token,idx in token2idx_dict.items()}
    emb_mat=[idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat,token2idx_dict


def convert_to_features(data,word2idx_dict,char2idx_dict):
    example={}

    pass


def build_features(config,examples,data_type,out_file,word2idx,char2idx,is_test=False):
    para_limit=config.test_para_limit if is_test else config.para_limit
    ques_limit=config.test_ques_limit if is_test else config.ques_limit
    ans_limit=100 if is_test else config.ans_limit
    char_limit=config.char_limit
    def filter_func(example):
        return len(example["context_tokens"])>para_limit or \
               len(example["ques_tokens"])>ques_limit or \
               (example["y2s"][0]-example["y1s"][0])>ans_limit

    def _get_word_idx(word):
        for each in (word,word.lower(),word.capitalize(),word.upper()):
            if each in word2idx:
                return word2idx[each]
        return 1

    def _get_char_idx(char):
        return char2idx[char] if char in char2idx else 1

    logger.info("Processing {} examples ...".format(data_type))
    writer=tf.io.TFRecordWriter(path=out_file)
    meta={}
    total=0
    total_=0
    for i,example in enumerate(examples):
        if i%1000==0:
            logger.info("{}".format(example["y1s"]))
            logger.info("{}".format(example["y2s"]))
        if len(example["y1s"])==0 or len(example["y2s"])==0:
            continue
        total_+=1
        if filter_func(example): continue # 过滤掉大于定值的句子
        total+=1
        context_idx=np.zeros([para_limit],dtype=np.int32)
        context_char_idx=np.zeros([para_limit,char_limit],dtype=np.int32)
        question_idx=np.zeros([ques_limit],dtype=np.int32)
        question_char_idx=np.zeros([ques_limit,char_limit],dtype=np.int32)
        y1=np.zeros([para_limit],dtype=np.int32)
        y2=np.zeros([para_limit],dtype=np.int32)

        # 段落词转索引
        for j, token in enumerate(example["context_tokens"]):
            context_idx[j]=_get_word_idx(token)
        # 问题词转索引
        for j, token in enumerate(example["ques_tokens"]):
            question_idx[j]=_get_word_idx(token)
        # 段落字符转索引
        for j, token in enumerate(example["context_chars"]):
            for k, char in enumerate(token):
                if k==char_limit:break
                context_char_idx[j,k]=_get_char_idx(char)
        # 问题字符转索引
        for j, token in enumerate(example["ques_chars"]):
            for k,char in enumerate(token):
                if k==char_limit:break
                question_char_idx[j,k]=_get_char_idx(char)

        start,end=example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end]=1,1

        features=defaultdict()
        features["context_idx"]=tf.train.Feature(int64_list=tf.train.Int64List(value=context_idx.tolist()))
        features["ques_idx"]=tf.train.Feature(int64_list=tf.train.Int64List(value=question_idx.tolist()))
        features["context_char_idx"]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idx.tostring()]))
        features["question_char_idx"]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[question_char_idx.tostring()]))
        features["y1"]=tf.train.Feature(int64_list=tf.train.Int64List(value=y1.tolist()))
        features["y2"]=tf.train.Feature(int64_list=tf.train.Int64List(value=y2.tolist()))
        features["id"]=tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))

        record=tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(record.SerializeToString())

    logger.info(f"Build {total}/{total_} instances of features.")
    meta["total"]=total
    writer.close()
    return meta


def save(filename,obj,message=None):
    if message is not None:
        logger.info("Saving {}...".format(message))
        with open(filename,"w",encoding="utf8") as file:
            json.dump(obj,file)


def preprocess(config):
    word_counter = Counter()
    char_counter = Counter()
    train_examples, train_eval = process_file(config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(config.test_file, "test", word_counter,char_counter)

    word_emb_mat, word2idx = get_embedding(word_counter,"token", vec_size=config.embedding_size)
    char_emb_mat, char2idx = get_embedding(char_counter, "char", vec_size=config.char_embedding_size)

    train_meta = build_features(config,train_examples,"train",config.train_output_file,word2idx,char2idx)
    dev_meta = build_features(config,dev_examples,"dev",config.dev_output_file, word2idx, char2idx)
    test_meta = build_features(config,test_examples,"test",config.test_output_file,word2idx,char2idx)

    save(config.word_emb_file,word_emb_mat,message="word_embedding")
    save(config.char_emb_file,char_emb_mat,message="char_embedding")

    save(config.word_dict,word2idx,message="word dictionary")
    save(config.char_dict,char2idx,message="char dictionary")


class Config:
    embedding_size=128
    char_embedding_size=64
    para_limit=400
    ques_limit=30
    char_limit=16
    test_para_limit=400
    test_ques_limit=30
    ans_limit=32

    train_file="../data/train-v2.0.json"
    dev_file="../data/dev-v2.0.json"
    test_file="../data/dev-v2.0.json"
    train_output_file="../output/train.tfrecord"
    dev_output_file="../output/dev.tfrecord"
    test_output_file="../output/test.tfrecord"
    word_emb_file="../output/embedding.word"
    char_emb_file="../output/embedding.char"
    word_dict="../output/vocab.word"
    char_dict="../output/vocab.char"





if __name__ == '__main__':
    config=Config()
    preprocess(config)