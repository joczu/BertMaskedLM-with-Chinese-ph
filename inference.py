#!/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 Tuya NLP. All rights reserved.
#   
#   FileName：inference.py
#   Author  ：rentc(桑榆)
#   DateTime：2021/4/23
#   Desc    ：
#
# ================================================================

from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch
import jieba
from tokenizer.tokenizer import MyTokenizer
import pdb
import random
import argparse

tokenizer = MyTokenizer("./vocab/SougouBertVocab.txt")
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="../bert_model")


parser = argparse.ArgumentParser()
parser.add_argument("-m","--mode",type=int,default=1,help="模式为0，表示使用随机mask进行推断，模式为1，表示使用指定mask进行推断")
parser.add_argument("-p","--p_mask",type=float,default=0.15,help="当mode=0时,此参数生效，用来表示随机生成mask的概率")
parser.add_argument("--model",type=str,help="载入训练好的模型")
args = parser.parse_args()

# model = BertForMaskedLM.from_pretrained('bert-base-chinese', cache_dir="../bert_model")
if not args.model:
    raise ValueError("you need a trained model")
model = BertForMaskedLM.from_pretrained(args.model)

data_m0 = ['但同时享受了佛教名山宗教的“红利”','我也一度以为用制片人的钱是应该的',
'由于小区在建设时是人工开挖而成的','她毕业于金日成综合大学',
'吉林市累计完成棚户区改造建筑面积６１０万平方米','把抗震救灾作为“四群”教育实行干部直接联系群众制度的第一线',
'专业框可以输入“不限”查询不限制专业的职位','敬请投资者留意投资风险',
'其中像国美等商家也频频结合自己的庆典、开业等活动','将茶文化带进厦门','并在逐步实施']

data_m1 = ["共产党 就是 好 [MASK]","[MASK] 就像 园丁 一样 抚育着 学生","你想要 干 [MASK]","中国 [MASK]","涂鸦 [MASK]","人工 [MASK]","我 爱 [MASK]"]
data_m1 = ['开 所有 [MASK]','我要 困 了 我 要 [MASK]','你 会 不会 爱 [MASK]','关闭 客 关闭 餐厅 [MASK]']
real = ["开所有灯光","我要困了我要睡觉","你会不会爱我","关闭客关闭餐厅灯"]


mode = args.mode
if mode==0:
    data2 = data_m0
    p=args.p_mask
elif mode==1:
    data2 = data_m1
    p=0
else:
    raise ValueError('mode must in [0,1]')


for i in data2:
    if mode == 1:
        inputs,labels = tokenizer.tokenize([i],max_length=50,p_mask=p)  # 特殊预测
        outputs = model(**inputs, return_dict=False)
    else:
        cut_i = jieba.lcut(i)
        inputs,labels = tokenizer.tokenize([' '.join(cut_i)],max_length=50,p_mask=p)
        outputs = model(**inputs, labels=labels, return_dict=True)
        loss = outputs.loss
        logits = outputs.logits
        #pdb.set_trace()
        labels2 = (labels != -100).squeeze()
        indices = torch.arange(0,labels.shape[1])
        labels3 = indices[labels2]
        print('loss:',loss.item())
        mask_token_logits = logits[0, labels3, :]
        word_idx = torch.topk(mask_token_logits, 1).indices.tolist()
        #pdb.set_trace()
        words = [tokenizer.tokenizer.convert_ids_to_tokens(j) for j in word_idx]
    # print(outputs)
    if mode == 1:
        outputs = outputs[0]
        #pdb.set_trace()
        mask_ind = torch.arange(0,inputs['input_ids'].shape[1])
        mask_ind = mask_ind[inputs['input_ids'][0]==103]
        mask_token_logits = outputs[0, mask_ind, :]
        word_idx = torch.topk(mask_token_logits, 5).indices.tolist()
        words = [tokenizer.tokenizer.convert_ids_to_tokens(j) for j in word_idx]
        print('输入: ',inputs['input_ids'])
        print('真实: ',i)
        print('预测: ',words)
    else:
        print('输入:',inputs['input_ids'])
        print('真实:',' '.join(cut_i))
        labels4 = labels3 - 1
        need_pre = [[cut_i[j]] for j in labels4]
        print('预测:',list(zip(need_pre,words)))
        print('\n')
    #break
# BertModel

# model = BertModel.from_pretrained("./model/Sougoumodel_step_1700000.bin")
# outputs = model(**inputs, return_dict=True)
# print()
# print(torch.topk(outputs.last_hidden_state[0, -2, :], 5))
