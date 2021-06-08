#!/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 Tuya NLP. All rights reserved.
#   
#   FileName：test.py
#   Author  ：rentc(桑榆)
#   DateTime：2021/4/24
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


def parser_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_data_path', type=str, default="./test/SougouTest.txt", help='training data path')
    parser.add_argument('-v', '--vocab_path', type=str, default='./vocab/SougouBertVocab.txt', help='vocab file path')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--each_steps', type=int, default=100, help='save model after train N steps ')
    parser.add_argument('--usegpu', action='store_true',default=False, help='use GPU or not')
    parser.add_argument('--device', type=str, default=1)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')
    args = parser.parse_args()
    return args

args = parser_args()
if not args.load_model:
    raise ValueError('test.py need a trained model')
else:
    model = BertForMaskedLM.from_pretrained(args.load_model)
multi_gpu = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# formatter = logging.Formatter(
#     '%(asctime)s - %(levelname)s - %(message)s')

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler(filename=args.log_path,mode='w')
#file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 创建一个handler，用于将日志输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
#console.setFormatter(formatter)
logger.addHandler(console)


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

test_dataset = MyDataset(args.test_data_path, n_raws=1000, shuffle=False)
mt = MyTokenizer(args.vocab_path)
time0 = time.time()
logger.info("The test model is : %s"%args.load_model)
logger.info("The logger information is saved in : %s"%args.log_path)

step = 0    
test_dataset.initial()
test_iter = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
correct = 0
total = 0
for gg, data in enumerate(test_iter):
    inputs,labels = mt.tokenize(data, max_length=100, p_mask = 0.15)
    if args.usegpu==True and args.device:
        inputs = inputs.to(device)
        labels = labels.to(device)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss.mean() if multi_gpu else outputs.loss
    logits = outputs.logits
    loss.backward()
    step += 1
    labels2 = (labels != -100)
    #pdb.set_trace()
    indices = torch.arange(0,labels.shape[1])
    for k in range(labels2.shape[0]):
        curl = labels2[k]
        if curl.sum()>0:
            curind = indices[curl]
        else:
            continue
        mask_token_logits = logits[k,curind,:]
        word_idx = torch.topk(mask_token_logits,5).indices
        real_idx = labels[k][labels[k]!=-100]
        for k2 in range(len(real_idx)):
            if real_idx[k2].item() in word_idx[k2]:
                correct += 1
    total += torch.sum(labels2).item()
    accuracy = correct/total
    if gg%(args.each_steps)==0:
        time1 = time.time()
        logger.info('\t batch = %d \t loss = %.5f \t accuracy = %.1f%% \t cost_time = %.3fs'%\
              (gg,loss.item(),accuracy*100,time1-time0))  
    if gg>=15:
        break
    
    








