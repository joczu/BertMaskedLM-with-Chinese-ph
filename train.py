#!/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 Tuya NLP. All rights reserved.
#   
#   FileName：test.py
#   Author  ：rentc(桑榆)
#   DateTime：2021/4/22
#   Desc    ：
#
# ================================================================
import torch
import logging
import argparse
from transformers import AdamW
import torch.utils.data as Data
from data_process import MyDataset
from tokenizer.tokenizer import MyTokenizer
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertForMaskedLM


def parse_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_path', type=str, default="./train_data/train", help='training data path')
    parser.add_argument('-s', '--model_save_path', type=str, default='./model', help='model file save path ')
    parser.add_argument('-v', '--vocab_path', type=str, default='./vocab/vocab.txt', help='vocab file path')
    parser.add_argument('-e', '--each_steps', type=int, default=100, help='save model after train N steps ')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('-p', '--epoch', type=int, default=3, help='epoch')
    args = parser.parse_args()
    return args


# get arguments
args = parse_args()
# 训练数据 按行处理
data_path = args.train_data_path
model_save_path = args.model_save_path
vocab_path = args.vocab_path
batch_size = args.batch_size
epoch = args.epoch
each_steps = args.each_steps
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig(
    vocab_size=60105,  # 305
    hidden_size=768 // 4,
    num_hidden_layers=12 // 4,
    num_attention_heads=12 // 4,
    intermediate_size=3072 // 4,
    max_position_embeddings=512 // 4,
    type_vocab_size=2,
    pad_token_id=0,
    return_dict=True
)
mt = MyTokenizer(vocab_path)
model = BertForMaskedLM(configuration)

train_dataset = MyDataset(data_path, n_raws=1000, shuffle=True)

step = 0
for _ in range(epoch):
    train_dataset.initial()
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    for _, data in enumerate(train_iter):
        print(data)
        # x = ["中国 人民 [MASK] 世界 长度 X 经 a b c a 2 工 了"] * 32
        inputs = mt.tokenize(data, max_length=128)
        # r = bert_model(**inputs)
        labels = mt.tokenize(data, max_length=128)["input_ids"]

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        optimizer = AdamW(model.parameters(), lr=1e-5)
        optimizer.state.get("")
        print(loss)
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        #
        # num_warmup_steps = 1000
        # num_train_steps = 10000
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % each_steps == 0:
            model.save_pretrained(model_save_path)
        # scheduler.step()
