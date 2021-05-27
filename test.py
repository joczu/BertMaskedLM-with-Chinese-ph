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
import transformers
from transformers import default_data_collator, DataCollatorForLanguageModeling

DataCollatorForLanguageModeling.tokenizer
with open("./vocab/vocab.txt", "r") as f:
    for i in range(100):
        print(f.readline())
