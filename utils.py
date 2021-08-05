#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 18:11:01 2021

@author: shizi
"""

from torch.nn import CrossEntropyLoss
from math import log
import numpy as np
import torch
import matplotlib.pyplot as plt
import jieba
import torch.utils.data as Data
from collections import namedtuple
from typing import Any
import pdb
import random
import datetime
UNK = '[UNK]'
pad_id = 0
PAD = '[PAD]'
from transformers import BertConfig
from transformers import BertTokenizer, BertForMaskedLM
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

#%% 

def calculate_loss_and_accuracy(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    # pdb.set_trace()
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size], GPT2模型的输出有两个元素的元组，取0就是取这组输出的特征
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits.contiguous()     ## .contiguous 在内存上连续, .shape = [batchsize,token_len-1,voca_size]
                                                        ## shift_logits 是从0开始，到n-2结束，与shift_labels的[1,n-1]一一对应，
                                                        ## 每个维度，比如说shift_logits[0,0,:]代表该batch的outputs的第0个样本的第
                                                        ## 的第0个词对应的向量，他们刻画了输出的分布，越大的数值越有可能表示下一个token
                                                        ## 这个分布对应的是下一个token的预测，去arg_max(shift_logits[0,0,:])就能得到
                                                        ## 这个logit下预测出的下一个token的id，因为arg_max取得是下标，而这个下标对应的是词典
                                                        ## 中某个词的位置。
    shift_labels = labels.contiguous()  ## .shape = [batchsize, token_len-1]

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),  ## 把平面的batch压缩成向量
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy

def calculate_last_logits_and_loss(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]
    
    def split_input_label(inp):
        findlabel = inp==102
        _,seplabel = findlabel.max(1)
        seplabel -= 1
        seplabel.unsqueeze_(1)
        labels = torch.gather(inp,1,seplabel)
        corres = seplabel - 1
        return labels, corres
    
    labels,corres_inp = split_input_label(inp=labels)
    logits_size = list(logits.shape)
    logits_size[1] = 1
    #special_logits = logits[..., corres_inp, :].contiguous().to(device)
    corres_inp = corres_inp.unsqueeze(-1).expand(*logits_size)
    special_logits = torch.gather(logits,1,corres_inp)
    loss_fct = CrossEntropyLoss(reduction='mean')
    special_logits = special_logits.view(-1,logits_size[-1])
    labels = labels.view(-1)
    loss = loss_fct(special_logits,labels)
    return special_logits, labels, loss


def bigrams_perplexity(sentence, ph_dict, sig_dict, islog=False, smooth=5*1e-8):
    """
    bigrams_perplexity can calculate the perplexity of a sentence by bigrams lm.

    Parameters
    ----------
    sentence : Type|list -> ['而且', '如果', '真的', '是', '名单', '已经', '出来', '了']
    ph_matrix : Type|Torch.Tensor(sparse_coo_tensor); Shape|[len(word2idx),len(word2idx)]
    sig_matrix: Type|Torch.Tensor(sparse_coo_tensor); Shape|[len(word2idx)]
    word2idx: Type|dict 
    Returns
    -------
    perplexity | Type:float
    """
    length = len(sentence)
    ppl = log(sig_dict[sentence[0]]) if sentence[0] in sig_dict else log(smooth)
    for i in range(1,length):
        if sentence[i-1] in ph_dict:
            if sentence[i] in ph_dict[sentence[i-1]]:
                ppl += log(ph_dict[sentence[i-1]][sentence[i]])
            else:
                ppl += log(smooth)
        else:
            ppl += log(smooth)
    ppl = -1/length * ppl
    
    if islog==False:
        ppl = np.exp(ppl)

    return ppl

def bigrams_tokenizer(sentence, word2idx):
    '''
    bigrams_tokenizer can tokenize a sentence by word2idx

    Parameters
    ----------
    sentence : Type | list 
        ex1-> ['而且', '如果', '真的', '是', '名单', '已经', '出来', '了']
        ex2-> ['而', '且', '如', '果', '真', '的', '是', '名', '单', '已', '经', 
               '出', '来', '了']
    word2idx : Type|dict; Describie|The corresding dict than can switch from 
               string list to int list
    
    Returns
    -------
    sen: Type | list 
        ex1 -> [512, 200, 1566, 7, 1826, 77, 677, 5]
        ex2 -> [121, 599, 197, 245, 485, 0, 6, 108, 332, 149, 49, 22, 35, 8]
    '''
    sen = [word2idx[i] if i in word2idx else word2idx[UNK] for i in sentence]
    return sen
    
def draw_ppl_distribution(ppl:np.ndarray, filename:str=None, 
                          group:int = 30, isLog = True):
    '''

    Parameters
    ----------
    ppl : np.ndarray
        DESCRIPTION: 获取已经收集好的困惑度，用于绘图
    filename : str, optional
        DESCRIPTION：用于设置图的title以及保存文件的名字，应该是'./dir1/dir2/name.png' 
    group : int, optional
        DESCRIPTION：默认值为30，根据group的值对ppl按大小顺序进行分组
    isLog : TYPE, optional
        DESCRIPTION. 为真则绘制对数ppl的分布，否则绘制原版ppl分布。
    Returns
    -------
    None ： 没有返回值，但是能够得到绘制的图形。

    '''
    assert type(ppl)==np.ndarray
    fig, ax = plt.subplots(figsize=(12,8),dpi=128)
    ks = ax.hist(ppl, bins=group, density = True, histtype="bar", rwidth=1,alpha=0.75)
    for i in ks[2].get_children():
        i.set_linewidth(1)
        i.set_edgecolor((0.,0.,0.4,0.75))
    ax.set_ylabel(r'$\frac{frequent}{dis(group)}$', fontsize = 16)
    ax.set_xlabel('logperplexity' if isLog else 'perplexity', fontsize = 16)
    gdis = ks[1][1] - ks[1][0]
    ax.plot(ks[1][:-1]+0.5*gdis,ks[0],'k--',linewidth = 1.5)
    mean = ppl.mean()
    var = ppl.var()
    
    
    annloc_x = ax.get_xticks()[-3]
    annloc_y = ax.get_yticks()[-2]
    y_dis = ax.get_yticks()[-1]-ax.get_yticks()[-2]
    x_dis = ax.get_xticks()[-1]-ax.get_xticks()[-2]
    
    if isLog:
        ax.annotate(text=r'$\mathbb{E}~[\xi]$=%.2f'%(mean), 
                    xy=(annloc_x-0.5*x_dis,annloc_y-0.5*y_dis), fontsize = 16)
        ax.annotate(text=r'$\mathbb{D}~[\xi]$=%.2f'%(var), 
                    xy=(annloc_x-0.5*x_dis,annloc_y-0.9*y_dis), fontsize = 16)
    else:
        ax.annotate(text=r'$\mathbb{E}~[\xi]$=%.2e'%(mean), 
                    xy=(annloc_x-0.5*x_dis,annloc_y-0.5*y_dis), fontsize = 16)
        ax.annotate(text=r'$\mathbb{D}~[\xi]$=%.2e'%(var), 
                    xy=(annloc_x-0.5*x_dis,annloc_y-0.9*y_dis), fontsize = 16)

    if filename:
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        name += '_distribution'
        plt.title(name, fontsize=18)
    else:
        pass
    
    if filename:
        plt.savefig(filename)
    else:
        plt.savefig('./perplexity_distribution.png')
    
    return None  


def draw_boxvioplot(ppl,filename:str=None, group:int=30, isLog=True, Tvalue:float=0.99):
    dist, interv = np.histogram(ppl,bins=group)
    distcum = dist.cumsum()
    if isLog == False:
        Tindex = 0
        while True:
            if distcum[Tindex]/distcum[-1]>Tvalue:
                break
            else:
                Tindex += 1
        #remvalue = distcum[-1] - distcum[Tindex]
        dist = dist[:Tindex+1]
        interv = interv[:Tindex+2]
    ppl2 = ppl[ppl<=interv[-1]]
    fig, ax = plt.subplots(figsize=[12,8],dpi=128)
    box = ax.boxplot(ppl2,vert=False,widths=0.5)
    mline = box['medians'][0] 
    mline.set_linewidth(2)
    mline.set_color('red')
    m_xloc = mline.get_data()[0][0]
    
    boxes = box['boxes'][0]
    boxes.set_linewidth(2)
    boxes_data = boxes.get_data()
    lu_locate = (boxes_data[0][1],boxes_data[1][1])
    ax.annotate(text = 'median=%.2f'%m_xloc,xy=(lu_locate[0],lu_locate[1]+0.02),fontsize=16)
    #mediansline = 
    ax.violinplot(ppl2,vert=False)
    ax.set_xlabel('logperplexity' if isLog else 'perplexity',fontsize=20)
    ax.set_xticklabels(labels=map(lambda x:int(x),ax.axes.get_xticks()),fontdict={'fontsize':20})
    ax.set_yticklabels(labels=['boxplot'],fontdict={'fontsize':20})
    
    if filename:
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        name += '_boxplot'
        ax.set_title(name,fontsize=20)
    else:
        ax.set_title('boxplot',fontsize=20)
    if filename:
        name = filename.replace('perplexity','boxplot')
        plt.savefig(name)
    else:
        plt.savefig(name+'.png')
    plt.show()




def top_accuracy(mask_token_logits,real_idx,k):

    res = 0
    word_idx = torch.topk(mask_token_logits,k).indices   # top5
    for k2 in range(len(real_idx)):
        if real_idx[k2].item() in word_idx[k2]:
            res += 1
    return res

def ppl_based_union_ngrams_lm(sen, logit, pm, smooth=1e-10, islog=True):
    '''
    describe: 该方法的原理是使将要预测的内容使用对应的语言模型进行替代，然后得到校正的句子，
              再用bigrams的困惑度公式进行计算。这个原理需要训练好的语言模型，bigrams参数。
    ----------------------------
    sen : 经过bert分词化的句子，tensor格式
    label : 标注句子中被mask的地方
    pm : 参数矩阵
    ------------------
    perplexity = (p(w_1|w_0) * p(w_2|w_1) * ... * p(w_n|w_{n-1}))^(-1/n)
    '''

    #pdb.set_trace()
    sen_flag = sen != 103
    logit = torch.argmax(logit,axis=-1)
    fix_sen = torch.where(sen_flag, sen, logit)
    fix_sen = fix_sen[fix_sen !=0 ]  # 去除补码位
    if islog:
        perplexity = 0
        for i in range(1,len(fix_sen)):
            if fix_sen[i-1] in pm:
                if fix_sen[i] in pm[fix_sen[i-1]]:
                    value = log(pm[fix_sen[i-1],fix_sen[i]]) 
                else:
                    value = log(smooth)
            else:
                value = log(smooth)
            perplexity += value
        perplexity = -1/len(sen)*perplexity
    else:
        preplexity = 1
        for i in range(1,len(fix_sen)):
            if fix_sen[i-1] in pm:
                if fix_sen[i] in pm[fix_sen[i-1]]:
                    value = pm[fix_sen[i-1],fix_sen[i]]
                else:
                    value = smooth
            else:
                value = smooth
            preplexity *= value
        perplexity = perplexity**(-1/len(sen))
    return preplexity


      
def common_method(temp, loop, args, corpus, drop, ph, phh):
        
    for i in range(len(corpus)):
        temp = corpus[i]
        if len(temp) > args.minlength:
            corpus[i] = [j for j in temp if j.isalnum() == True]
            if len(corpus[i]) <= args.minlength:
                drop.append(i)
            else:
                for j in loop:
                    if j not in ph:
                        ph[corpus[i][j]] = 1
                    else:
                        ph[corpus[i][j]] += 1
                    if j > 0:
                        preword = corpus[i][j - 1]
                        curword = corpus[i][j]
                        if preword not in phh:
                            phh[preword] = {curword: 1}
                        else:
                            if curword not in phh[preword][curword]:
                                phh[preword][curword] = 1
                            else:
                                phh[preword][curword] += 1
        else:
            drop.append(i)
    return None

def judge_change(dic, isdouble=False):
    assert type(dic) == dict
    stop = np.random.randint(low=10,high=100)
    ischange = 0
    if isdouble==False:
        for i in dic.values():
            if type(i)==str:
                ischange += 1
            if stop < 0:
                break
            stop -= 1
        if ischange >= 10:
            for k,v in dic.items():
                dic[k] = float(v)
    else:
        for k,v in dic.items():
            for k1,v1 in dic[k].items():
                if type(v1) == str:
                    ischange += 1
                if stop < 0:
                    break
            stop -= 1
        if ischange >= 10:
            for k,v in dic.items():
                for k1,v1 in dic[k].items():
                    dic[k][k1] = float(v1)
    return dic

#%%

class BertTokenTool:
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
                                       strip_accents=None,
                                       )

    def tokenize(self, word_list,  p_mask:float, max_length=300, truncation=True, 
                 padding=True, islastone=False):
        inputs = self.tokenizer(word_list,
                                return_tensors="pt",
                                truncation=truncation,
                                padding=padding,
                                max_length=max_length)
        if p_mask == 0:
            if islastone:
                inputs['input_ids'],labels = self.collate_fn2(inputs['input_ids'])
            else:
                labels = None
        else:
            inputs['input_ids'],labels = self.collate_fn(inputs['input_ids'], p_mask=p_mask)
        return inputs,labels
    
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
    
    def collate_fn2(self,inputs):
        labels = inputs.clone()
        maskid = [i-1 for val in labels.tolist() for i in range(len(val)) 
                  if val[i]==self.tokenizer.sep_token_id]
        for i in range(len(inputs)):
            ind = maskid[i]
            inputs[i][ind] = self.tokenizer.mask_token_id
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if j != maskid[i]:
                    labels[i][j] = -100
                else:
                    pass
        return inputs, labels  
    
class GptTokenTool:
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
                                       strip_accents=None,
                                       )
    def tokenize(self, word_list, max_length=300, truncation=True, 
                 padding=True):
        
        inputs = self.tokenizer(word_list,
                        return_tensors="pt",
                        truncation=truncation,
                        padding=padding,
                        max_length=max_length)
        inputs = inputs['input_ids']
        inputs, labels = inputs[...,:-1], inputs[...,1:]
        return inputs, labels
    

class MyDataset(Data.Dataset):
    def __init__(self, file_path, n_raws=1000, shuffle=False):
        """
        file_path: the path to the dataset file
        nraws: each time put nraws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        total_count = 0
        # get the count of all samples
        with open(file_path, 'r') as f:
            for _ in f:
                total_count += 1
        self.file_path = file_path
        self.total_count = total_count
        self.n_raws = n_raws
        self.shuffle = shuffle
        self.file_input = None
        self.samples = []

    def mask(self, text):
        return text

    def initial(self):
        self.file_input = open(self.file_path, 'r')
        self.samples = []
        # put nraw samples into memory
        for _ in range(self.n_raws):
            raw = self.file_input.readline()
            if raw:
                self.samples.append(" ".join(jieba.lcut(raw.strip())))
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return self.total_count

    def __getitem__(self, item):
        idx = self.index[0]
        sample = self.samples[idx]
        # self.index = self.index[1:]
        self.index = self.index[1:]
        
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:
            # all the samples in the memory have been used, need to get the new samples
            for _ in range(self.n_raws):
                raw = self.file_input.readline()
                if raw:
                    self.samples.append(" ".join(jieba.lcut(raw.strip())))
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)
        return sample

def day_month(datetime_now):
    '''
    day_month return something like '7-30' datetime
    Parameters
    ----------
    datetime_now : TYPE -> datetime.datetime
    
    Returns
    -------
    day: Type -> str
    
    Example:
        day = day_month(datetime.now())
        # day = '7-30'
    '''    
    day = datetime_now.day
    month = datetime_now.month
    return str(month) + '-' + str(day)


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data




    
