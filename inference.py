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

from tokenizer.tokenizer import MyTokenizer

tokenizer = MyTokenizer("vocab/vocab.txt")
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="../bert_model")
# model = BertForMaskedLM.from_pretrained('bert-base-chinese', cache_dir="../bert_model")
model = BertForMaskedLM.from_pretrained("./model")

inputs = tokenizer.tokenize(["中国 [MASK]"])
labels = tokenizer.tokenize(["中国 人民"])["input_ids"]
outputs = model(**inputs, labels=labels, return_dict=True)
print(outputs)
loss = outputs.loss
logits = outputs.logits
print(loss, logits)
mask_token_logits = logits[0, -2, :]
word_idx = torch.topk(mask_token_logits, 5).indices.tolist()
words = tokenizer.tokenizer.convert_ids_to_tokens(word_idx)
print(words)

# BertModel

model = BertModel.from_pretrained("./model")
outputs = model(**inputs, return_dict=True)
torch.topk(outputs.last_hidden_state[0, -2, :], 5)
