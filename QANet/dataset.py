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
import numpy as np
from collections import Counter
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


    pass

def load_vocab_file(counter):
    vocab={"<pad>":0,"<unk>":1}
    for i, t in enumerate(counter.most_common()):
        if t not in vocab:
            vocab[t]=i+2
    return vocab






if __name__ == '__main__':
    word_counter=Counter()
    char_counter=Counter()
    train_examples,train_eval=process_file("../data/train.json","train",word_counter,char_counter)
    word2index=load_vocab_file(word_counter)
    char2index=load_vocab_file(char_counter)