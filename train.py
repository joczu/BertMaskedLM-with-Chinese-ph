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
import re
from transformers import AdamW
import torch.utils.data as Data
from data_process import MyDataset
from tokenizer.tokenizer import MyTokenizer
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertForMaskedLM
import os
import time
import pdb


def parse_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_path', type=str, default="./train_data/train", help='training data path')
    parser.add_argument('-s', '--model_save_path', type=str, default='./model', help='model file save path ')
    parser.add_argument('-v', '--vocab_path', type=str, default='./vocab/vocab.txt', help='vocab file path')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('-e', '--epoch', type=int, default=2, help='epoch')
    parser.add_argument('--each_steps', type=int, default=100, help='save model after train N steps ')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--usegpu', action='store_true',default=False, help="use gpu or not")
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--curepoch',type=int, help="what epoch you want to begin train model")
    parser.add_argument('--curstep',type=int, help="where step you want to begin train model")
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=args.log_path,mode='w')
#file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 创建一个handler，用于将日志输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
#console.setFormatter(formatter)
logger.addHandler(console)


# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig(
    vocab_size=68181,  # SougouBertVocab共68181个词汇，它们是过滤了Sougou语料词频不大于6*1e-7的词后，与Bert自带的vocab取交集
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

if not args.load_model:
    model = BertForMaskedLM(configuration)
else:
    model = BertForMaskedLM.from_pretrained(args.load_model)
multi_gpu = False
if args.usegpu==True and args.device:
    if len(args.device)==1:
        device=torch.device(int(args.device))
        model = model.to(device)
    elif len(args.device)>1 and torch.cuda.device_count()>1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        device_ids=[int(i) for i in args.device.split(',')]
        model = torch.nn.DataParallel(model,device_ids=device_ids)
        multi_gpu = True
else:
    device = torch.device("cpu")

train_dataset = MyDataset(data_path, n_raws=1000, shuffle=False)

time0 = time.time()
optimizer = AdamW(model.parameters(), lr=0.5*1e-5)

logger.info("The train model is : %s"%args.load_model)
logger.info("The logger information is saved in : %s"%args.log_path)

if args.curepoch:
    curepoch = args.curepoch
else:
    curepoch = -1
if args.curstep:
    curstep = args.curstep
else:
    curstep = -1
    
    
for ee in range(epoch):
    if ee < curepoch:
        continue
    step = 0    
    train_dataset.initial()
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("\t epoch = %d"%ee)
    for gg, data in enumerate(train_iter):
        step += 1
        if gg < curstep:
            continue
        inputs,labels = mt.tokenize(data, max_length=100, p_mask = 0.15)
        # r = bert_model(**inputs)
        # labels = mt.tokenize(data, max_length=128)["input_ids"]
        if args.usegpu==True and args.device:
            inputs = inputs.to(device)
            labels = labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.mean() if multi_gpu else outputs.loss
        logits = outputs.logits
        optimizer.state.get("")
        if gg%(0.1*each_steps)==0:
            time1 = time.time()
            logger.info('\t batch = %d \t loss = %.5f \t cost_time = %.3fs'%(gg,loss.item(),time1-time0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % each_steps == 0:
            if args.model_name:
                model_save_path1 = os.path.join(model_save_path,'%s_step_%d.bin'%(args.model_name,step))
            else:
                model_save_path1 = os.path.join(model_save_path,'model_step_%d.bin'%(step))
            if hasattr(model,'module'):
                model.module.save_pretrained(model_save_path1)
            else:
                model.save_pretrained(model_save_path1)
    if args.model_name:
        model_save_path2 = os.path.join(model_save_path,'%s_epoch_%d.bin'%(args.model_name,ee))
    else:
        model_save_path2 = os.path.join(model_save_path,'model_epoch_%d.bin'%ee)
    print()
    if hasattr(model,'module'):
        model.module.save_pretrained(model_save_path2)
    else:
        model.save_pretrained(model_save_path2)