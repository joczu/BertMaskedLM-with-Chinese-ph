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
import math
import pdb
from scipy import sparse

def parser_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_data_path', type=str, default="./test/SougouTest.txt", help='test data path')
    parser.add_argument('-v', '--vocab_path', type=str, default='./vocab/SougouBertVocab.txt', help='vocab file path')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--each_steps', type=int, default=100, help='save model after train N steps ')
    parser.add_argument('--usegpu', action='store_true',default=False, help='use GPU or not')
    parser.add_argument('--device', type=str, default=1)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--perplexity',type=str,default='')
    parser.add_argument('--lastone_token',action='store_true', default=False,help='let the last word in a sentence be [MASK]')
    args = parser.parse_args()
    return args

args = parser_args()
if not args.load_model:
    raise ValueError('test.py need a trained model')
else:
    model = BertForMaskedLM.from_pretrained(args.load_model)
multi_gpu = False


if args.perplexity:
    para_matrix = sparse.dok_matrix(sparse.load_npz(args.perplexity))

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
#pdb.set_trace()

mt = MyTokenizer(args.vocab_path)
time0 = time.time()
logger.info("The test model is : %s"%args.load_model)
logger.info("The logger information is saved in : %s"%args.log_path)

step = 0    
test_dataset.initial()
total_samplers = len(test_dataset)
test_iter = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
correct_5 = 0
correct_3 = 0
correct_1 = 0
prep_storge = []
total = 0

def top_accuracy(mask_token_logits,real_idx,k):
    res = 0
    word_idx = torch.topk(mask_token_logits,k).indices   # top5
    for k2 in range(len(real_idx)):
        if real_idx[k2].item() in word_idx[k2]:
            res += 1
    return res
def perplexity(sen, logit, pm):
    '''
    sen : 经过bert分词化的句子，tensor格式
    label : 标注句子中被mask的地方
    pm : 参数矩阵
    # ======
    perplexity = (p(w_1|w_0) * p(w_2|w_1) * ... * p(w_n|w_{n-1}))^(-1/2)
    '''
    preplexity = 0
    #pdb.set_trace()
    sen_flag = sen != 103
    logit = torch.argmax(logit,axis=-1)
    fix_sen = torch.where(sen_flag, sen, logit)
    fix_sen = fix_sen[fix_sen !=0 ]  # 去除补码位
    for i in range(1,len(fix_sen)):
        try:
            value = math.log(pm[fix_sen[i-1],fix_sen[i]])
        except ValueError:
            value = -1000   # python中的log可达到的最小值为math.log(1e-323)=-743.74..再小就会变成0，变成math.log(0)引发错误。
        preplexity += value
    log_preplexity = preplexity*(-0.5)
    return log_preplexity
    
for gg, data in enumerate(test_iter):
    pdb.set_trace()
    if not args.lastone_token:
        inputs,labels = mt.tokenize(data, max_length=100, p_mask = 0.15)
    else:
        inputs,labels = mt.tokenize(data, max_length=100, p_mask = 0, islastone=True)
    if args.usegpu==True and args.device:
        inputs = inputs.to(device)
        labels = labels.to(device)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss.mean() if multi_gpu else outputs.loss
    logits = outputs.logits
    step += 1
    labels2 = (labels != -100)
    #pdb.set_trace()
    indices = torch.arange(0,labels.shape[1])
    
    for k in range(labels2.shape[0]):
        curl = labels2[k]
        curlogit = logits[k,:,:].to('cpu')
        curprep = perplexity(inputs["input_ids"][k].to('cpu'),curlogit,para_matrix)
        prep_storge.append(curprep)
        if curl.sum()>0:
            curind = indices[curl]
        else:
            continue
        mask_token_logits = logits[k,curind,:]
        real_idx = labels[k][labels[k]!=-100]
        correct_5 += top_accuracy(mask_token_logits,real_idx,k=5)
        correct_3 += top_accuracy(mask_token_logits,real_idx,k=3)
        correct_1 += top_accuracy(mask_token_logits,real_idx,k=1)

    total += torch.sum(labels2).item()
    top5 = correct_5/total
    top3 = correct_3/total
    top1 = correct_1/total
    mean_prep = sum(prep_storge[-1*args.batch_size:])/args.batch_size
    if gg%(args.each_steps)==0:
        time1 = time.time()
        ratio = (gg+1)*args.batch_size/total_samplers
        logger.info('\t batch = %d \t complete = %.3f%% \t loss = %.3f \t top1 = %.1f%% \t top3 = %.1f%% \n\t top5 = %.1f%% \t mean_log_preplexity = %.1f \t cost_time = %.1fs\n'%(gg,ratio*100,loss.item(),top1*100,top3*100,top5*100,mean_prep,time1-time0))
    
with open('./log_preplexity.txt','w',encoding='utf8') as f:
    f.write(str(prep_storge))
    








