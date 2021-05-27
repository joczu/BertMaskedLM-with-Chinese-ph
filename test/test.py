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
from transformers import AdamW
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertForMaskedLM
from transformers import get_linear_schedule_with_warmup

# Initializing a BERT bert-base-uncased style configuration
from tokenizer.tokenizer import MyTokenizer
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="../bert_model")
model = BertForMaskedLM.from_pretrained('bert-base-chinese', cache_dir="../bert_model")

inputs = tokenizer("中华人民共和[MASK]", return_tensors="pt")
print(inputs)
labels = tokenizer("中华人民共和国", return_tensors="pt")["input_ids"]

outputs = model(**inputs, labels=labels, return_dict=True)
loss = outputs.loss
logits = outputs.logits
print(loss, logits)
mask_token_logits = logits[0, -2, :]
word_idx = torch.topk(mask_token_logits, 5).indices.tolist()
words = tokenizer.convert_ids_to_tokens(word_idx)
print(words)
