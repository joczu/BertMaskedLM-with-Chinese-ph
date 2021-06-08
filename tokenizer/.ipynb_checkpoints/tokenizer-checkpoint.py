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
import pdb

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

    def tokenize(self, word_list,  p_mask:float, max_length=20, truncation=True, padding=True):
        inputs = self.tokenizer(word_list,
                                return_tensors="pt",
                                truncation=truncation,
                                padding=padding,
                                max_length=max_length)
        #pdb.set_trace()
        if p_mask == 0:
            labels = None
        else:
            inputs['input_ids'],labels = self.collate_fn(inputs['input_ids'], p_mask=p_mask)
        return inputs,labels
            # labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    
    def collate_fn(self, inputs, p_mask:float = 0.15):
        '''
            collate_fn 函数用来进一步校正数据，安装bert模型的mask原则作出masked的inputs与labels。
            其中，inputs有15%的几率被mask，而所有的mask中，有80%的几率用'[mask]'进行替代，有10%用随机词进行替代，剩下10%保留原形
        '''
        labels = inputs.clone()
        #pdb.set_trace()
        
         # 特殊码标记，Bert.tokenizer特殊码指 '[UNK]','[CLS]','[PAD]','[SEP]','[MASK]'. 如果这个tag为特殊码，则为1，否则为0
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]   
        special_tokens_mask = torch.tensor(special_tokens_mask,dtype=torch.bool)
        probability_matrix = torch.full(labels.shape, p_mask)
        probability_matrix.masked_fill_(special_tokens_mask, value = 0.)  # 特殊码的位置被标记为0，让特殊码一定不会变动
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels
        
        
    

if __name__ == '__main__':
    mt = MyTokenizer("../vocab/vocab.txt")
    sen = [' '.join(jieba.lcut('我也一度以为用制片人的钱是应该的')),' '.join(jieba.lcut('她毕业于金日成综合大学'))]
    #sen2 = jieba.lcut('她毕业于金日成综合大学')
    #pdb.set_trace()
    p,l = mt.tokenize(sen)
    #p2,l2 = mt.tokenizer([' '.join(sen2)])
    print('原句为：', ' '.join(sen), '\n分词后： ',p['input_ids'], '\n标签: ',l )
    #print('\n')
    #print('原句为：', ' '.join(sen2), '\n分词后： ',p2['input_ids'], '\n标签: ',l2 )
    #print(p2)
