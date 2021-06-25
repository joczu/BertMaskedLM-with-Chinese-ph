#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:04:55 2021

@author: shizi
"""


import pdb
import jieba
import torch
import numpy as np
from scipy import sparse
file_train = "./train_data/SougouTrain.txt"
file_test = "./test/SougouTest.txt"
vocab_file = "./vocab/SougouBertVocab.txt"

with open(file_train,'r',encoding='utf8') as f:
    string_train = f.read()
with open(file_test,'r',encoding='utf8') as f:
    string_test = f.read()
string = string_train+string_test
string = string_test
del string_train
del string_test
string = string.split('\n')
    
from transformers import BertTokenizer
tokenizer = BertTokenizer(vocab_file,
                         do_lower_case=True,
                         do_basic_tokenize=True,
                         never_split=None,
                         unk_token="[UNK]",
                         sep_token="[SEP]",
                         pad_token="[PAD]",
                         cls_token="[CLS]",
                         mask_token="[MASK]",
                         tokenize_chinese_chars=False,
                         strip_accents=None)
word_num = tokenizer.vocab_size
parameters = np.zeros([word_num,word_num],dtype=float)
sigw = np.zeros(word_num,dtype=float)
for i in range(len(string)):
    curstring = jieba.lcut(string[i])
    curstring = ' '.join(curstring)
    curstring = tokenizer(curstring,
                          truncation=True,
                          padding=True,
                          max_length=100)
    curstring = curstring['input_ids']
    sigw[curstring[0]] += 1
    for k in range(1,len(curstring)):
        parameters[curstring[k-1],curstring[k]] += 1
        sigw[curstring[k]] += 1
    if i % 10**4==0:
        print(i)

#pdb.set_trace()
# 由于矩阵过大，68000*68000左右，无法直接归一化计算频率，因此使用稀疏矩阵，
# 稀疏矩阵中，coo_matrix，对于已经知道矩阵的元素的情况下，是比较有效的，它虽无法进行增量处理，但
#     能够有效计算行和，列和。

sigw_sum = np.sum(sigw)
sigw = sigw/sigw_sum

parameters = sparse.lil_matrix(parameters)
parameters_sum = parameters.sum(axis=1)
for ind,k in enumerate(parameters.data):
    if len(k):
        parameters.data[ind] = [i/parameters_sum[ind,0] for i in k]
# valid
parameters_sum = parameters.sum(axis=1)
parameters_sum = parameters_sum[parameters_sum != 0]  # 扣除每行和中不为0的，那么剩下的就都为1
print('error = ',np.abs(np.sum(parameters_sum-1)+np.sum(sigw)-1))

# save sparse matrix
parameters = sparse.coo_matrix(parameters)   # lil_matrix类型的稀疏矩阵不可储存，需要变换成其他格式
sparse.save_npz('./para_ph.npz',parameters)
np.save('./para_sig.npy',sigw)
print('pharse sparse matrix has saved at ./para_ph.npz')
print('sigword matrix has save at ./para_sig.npy')

# load sparse matrix, if you need
# sparse_matrix = sparse.load_npz('./para.npsz')
# sparse_matrix = sparse.lil_matrix(sparse_matrix)




    