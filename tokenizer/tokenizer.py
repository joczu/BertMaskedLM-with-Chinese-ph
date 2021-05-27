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
import jieba
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertForMaskedLM


class MyTokenizer:
    def __init__(self, vocab_path):
        self.tokenizer = BertTokenizer(vocab_path,
                                       do_lower_case=True,
                                       do_basic_tokenize=True,
                                       never_split=None,
                                       unk_token="[UNK]",
                                       sep_token="[SEP]",
                                       pad_token="[PAD]",
                                       cls_token="[CLS]",
                                       mask_token="[MASK]",
                                       tokenize_chinese_chars=False,
                                       strip_accents=None
                                       )

    def tokenize(self, word_list, max_length=10, truncation=True, padding=True):
        inputs = self.tokenizer(word_list,
                                return_tensors="pt",
                                truncation=truncation,
                                padding=padding,
                                max_length=max_length)
        return inputs
        # labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]


if __name__ == '__main__':
    mt = MyTokenizer("../vocab/vocab.txt")
    p = mt.tokenize(["中国 a", "中国 人民 [MASK] 安全 他们 [MASK]"], max_length=5)
    p2 = mt.tokenizer("中国 人民 [MASK] 安全 他们 [MASK]", return_tensors="pt")
    print(p)
    print(p2)
