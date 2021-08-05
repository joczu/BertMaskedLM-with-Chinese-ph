#!/usr/bin/env python
# coding=utf-8
"""
   Copyright (C) 2019 Tuya NLP. All rights reserved.   
   FileName：train.py
   Author  ：shizi
   DateTime：2021/7/14
   Desc    ：该脚本只支持bert和gpt2的训练。经过代码抽象后，要求当修改model_name=bert时，
             能够自动调用bert分词，训练Bert模型；当修改model_name=gpt2时，能够自动调用
             gpt分词，训练gpt模型.
             目前不支持char模式。
   Vocab   : SougouBertVocab共68181个词汇，它们是过滤了Sougou语料词频不大于6*1e-7的词
             后，与bert-base-chinese自带的vocab取并集
"""

import torch
import logging
import argparse
from transformers import AdamW
import torch.utils.data as Data
from transformers import GPT2Config,BertConfig
from transformers import GPT2LMHeadModel,BertForMaskedLM
from utils import (calculate_loss_and_accuracy,GptTokenTool,BertTokenTool,
                   MyDataset,day_month,ModelWrapper)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time
import pdb
PAD = '[PAD]'
pad_id = 0
logger = None

def parse_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str,default='bert', 
                        help='model name = gpt2 or bert')
    parser.add_argument('--trainfile', type=str, 
                        default="./testdata/stard1w.txt", 
                        help='training data path')
    parser.add_argument('--savemodel', type=str, 
                        default='./modelfile/MN/', 
                        help='model file save path')
    parser.add_argument('--mode',type=str,
                        default='word',
                        help='train model based on word or char')
    parser.add_argument('--corpusname',type=str,
                        default='sougou',
                        help='trainging corpus name')
    parser.add_argument('--vocabpath', type=str, 
                        default='./vocab/SougouBertVocab.txt', 
                        help='vocabulary file')
    parser.add_argument('--batchsize', type=int, default=2, 
                        help='set your batch_size')
    parser.add_argument('--epoch', type=int, default=1, 
                        help='epoch')
    parser.add_argument('--showstep', type=int, default=100, 
                        help='''during training, save and show model after 
                        train N steps ''')
    parser.add_argument('--usegpu', type=int,default=0, 
                        help="usegpu = 1 if use else 0")
    parser.add_argument('--device', type=str, default='0',
                        help="""if usegpu and only a sigle gpu, you can ignore 
                        the item, but if you use multi gpu, you should set the 
                        item='0,1' for two pieces of gpu, item='0,1,2' for 
                        three pieces of gpu and so on""")
    parser.add_argument('--loadmodel', type=str, default=None,
                        help="your trained model file address")
    parser.add_argument('--log', type=str, default='./log/MN/CN_DT_train_log.txt')
    parser.add_argument('--tensorboard',type=str,default='./tensorboard/MN/CN_DT_train',
                        help='tensorboard directory')
    parser.add_argument('--curepoch',type=str, default=None, 
                        help="what epoch you want to begin train model")
    parser.add_argument('--curstep',type=str, default=None, 
                        help="where step you want to begin train model")
    args = parser.parse_args()
    return args

# get arguments
args = parse_args()
# 训练数据 按行处理
assert args.modelname in ['gpt2','bert'], 'model_name must be gpt2 or bert'
assert args.usegpu in [0,1],'usegpu is equal 0 or 1'
assert args.mode in ['word','char'], 'mode is word or char'

day = day_month(datetime.now())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logfile = args.log
corpusname = args.trainfile.split('/')[-1]
corpusname = corpusname.split('.')[0]
logfile = logfile.replace('MN',args.modelname)
logfile = logfile.replace('CN',corpusname).replace('DT',day)
file_handler = logging.FileHandler(filename=logfile,mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)

tb_dir = args.tensorboard.replace('MN',args.modelname)
tb_dir = tb_dir.replace('CN',corpusname).replace('DT',day)
if os.path.exists(tb_dir):
    files = os.listdir(tb_dir)
    for i in files:
        os.remove(os.path.join(tb_dir,i))
tb_writer = SummaryWriter(log_dir = tb_dir,filename_suffix='tb')

if args.modelname=='gpt2':
    configuration = GPT2Config(
                    vocab_size = 68181,
                    bos_token_id = 68180,
                    eos_token_id = 68180,
                    n_embd = 768 // 4,
                    n_layer= 12 // 4,
                    n_head= 12 // 4,
                    n_ctx = 1024//4,
                    n_positions = 1024 // 4)
    tokenizer = GptTokenTool(args.vocabpath)
    if not args.loadmodel:
        model = GPT2LMHeadModel(configuration)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.loadmodel)
elif args.modelname=='bert':
    configuration = BertConfig(
                    vocab_size=68181,
                    hidden_size=768 // 4,
                    num_hidden_layers=12 // 4,
                    num_attention_heads=12 // 4,
                    intermediate_size=3072 // 4,
                    max_position_embeddings=512 // 4,
                    type_vocab_size=2,
                    pad_token_id=0,
                    return_dict=True)
    tokenizer = BertTokenTool(args.vocabpath)
    if not args.loadmodel:
        model = BertForMaskedLM(configuration)
    else:
        model = BertForMaskedLM.from_pretrained(args.loadmodel)
transfer = tokenizer.tokenizer.convert_ids_to_tokens

multi_gpu = False
if bool(args.usegpu)==True and args.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert str(device)=='cuda','your machine need own a gpu card at least'
    udevice = list(map(int,args.device.split(',')))
    udevice = [i for i in udevice if type(i)==int]
    if len(udevice)==1:
        sdevice=torch.device(udevice[0])
        model = model.to(sdevice)
    elif len(udevice)>1 and torch.cuda.device_count()>1:
        model = model.to(device)
        device_ids=[int(i) for i in udevice]
        model = torch.nn.DataParallel(model,device_ids=device_ids)
        multi_gpu = True
else:
    device = torch.device("cpu")

trainset = MyDataset(args.trainfile, n_raws=1000, shuffle=True)

time0 = time.time()
optimizer = AdamW(model.parameters(), lr= 1e-5)

logger.info("The Initial Date = %s"%day)
logger.info("%s is training which based on corpus %s"%
            (args.modelname,args.trainfile))
logger.info("The log information is saved in : %s"%logfile)

curepoch = int(args.curepoch) if args.curepoch else -1
curstep = int(args.curstep) if args.curepoch else -1

#%%
def bertshow_predict_vs_actual(inputs, labels, outputs, 
                           transfer = tokenizer.tokenizer.convert_ids_to_tokens):
    '''
    function show_predict_vs_actual can generate markdown format string for 
        better showing our train or test effect.
    Parameters
    ----------
    inputs : TYPE --> your dataloader element.
        DESCRIPTION. --> empty
    labels : TYPE --> your dataloader element. 
        DESCRIPTION.
    outputs : TYPE --> result that your inputs and labels pass through model. 
        DESCRIPTION.
    transfer : TYPE --> function
        DESCRIPTION. The default is tokenizer.tokenizer.convert_ids_to_tokens.
    Returns
    -------
    info2 : TYPE --> string 
        DESCRIPTION : info2 is markdown string, it can't identified as table by
        tensorboard.
    '''
    
    ind = torch.ones_like(labels)
    for kk in range(args.batchsize):
        ind[kk] *= kk
    ind = torch.where(labels != -100, ind, labels)
    ind = ind[ind != -100]
    ind = [i.item() for i in ind]
    score = torch.softmax(outputs.logits[labels != -100],-1)
    score = score.max(-1).values
    score = score.tolist()
    masked_pre = outputs.logits[labels != -100].max(-1).indices
    masked_label = labels[labels != -100]
    info = list(zip(ind,transfer(masked_pre),score,transfer(masked_label)))
    translist = list(map(transfer, inputs['input_ids']))
    translist = [[j for j in i if j != '[PAD]'] for i in translist]
    for i0,i1 in enumerate(translist):
        i1.append(' :')
        for j0,j1,j2,j3 in info:
            if i0==j0:
                i1.append((j1,j2,j3))
    info2 = ''    
    for i in translist:
        flag = 0
        keyind = i.index(' :')
        if keyind == len(i)-1:
            first = '&nbsp;&nbsp;'.join(i[:keyind])
            jm = ''
            flag -= 1
        else:
            first = '&nbsp;&nbsp;'.join(i[:keyind])
            second = i[keyind+1:]
            jm = ''
            for kt,(j0,j1,j2) in enumerate(second):
                if j0 != j2:
                    flag += 1 
                if kt==0:
                    jm += '| **PREDICT** | **ACTUAL** |\n| :---: | :---: |\n'
                jm += '| %s(%.3f%%) | %s  |\n'%(j0,j1*100,j2)
                
                    # jm += '||<strong>预测:%s'%j0 + '(prob:%.5f)'%j1 + ' vs 实际:%s</strong>'%j2 
        if flag == 0:
            info2 += '<p><strong>Sentence ✅:</strong>&nbsp;&nbsp;'+first +'</p>' + jm
        elif flag>0:
            info2 += '<p><strong>Sentence ❌:</strong>&nbsp;&nbsp;'+first +'</p>' + jm                            
        elif flag<0:
            info2 += '<p><strong>Sentence ⭕️:</strong>&nbsp;&nbsp;'+first +'</p>' + jm                                                       
    return info2

def gptshow_predict_vs_actual(inputs, labels, outputs, 
                           transfer = tokenizer.tokenizer.convert_ids_to_tokens):
    
    translist = list(map(transfer, inputs))
    # labelist  = list(map(transfer, labels))
    # translist = [[j for j in i if j !='[PAD]'] for i in translist]
    # labelist  = [[j for j in i if j !='[PAD]'] for i in labelist]
    _,pre = outputs.logits.max(-1)
    prelist = list(map(transfer, pre))
    info = ''
    for i,j in zip(translist,prelist):
        info += ' '.join(i) + '  \n' + ' '.join(j)
        info += '***'
    return info
    
#%%
trainset.initial()

def train(model, 
          corpus, 
          epochs = args.epoch, 
          modelname = args.modelname, 
          batchs = args.batchsize,
          maxlength = 100):
    
    runloss = 0.
    runacc = 0.
    speci_var = 1
    
    for ee in range(epochs):
        if ee < curepoch:
            continue
        train_iter = Data.DataLoader(dataset=corpus, batch_size=batchs, shuffle=True)
        logger.info("epoch = %d"%ee)
        
        for gg, data in enumerate(train_iter):
            if gg < curstep:
                continue
            if modelname == 'gpt2':
                inputs,labels = tokenizer.tokenize(data, max_length=maxlength)
                if str(device)=='cuda':
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model.forward(input_ids = inputs)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=labels)
            elif modelname == 'bert':
                inputs,labels = tokenizer.tokenize(data, max_length=maxlength, p_mask=0.15)
                if str(device)=='cuda':
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                masked_label = labels[labels != -100]
                masked_pre = outputs.logits[labels != -100].max(-1).indices
                if masked_label.numel() == 0:
                    accuracy = 0
                else:
                    accuracy = (torch.sum(masked_pre==masked_label)/masked_label.numel()).item()
                loss = outputs.loss.mean() if multi_gpu else outputs.loss
                if gg%(0.1*args.showstep)==9:
                    info2 = bertshow_predict_vs_actual(inputs, labels, outputs)
                    tb_writer.add_text('predict-vs-actural',info2,ee*len(train_iter)+gg)
                    tb_writer.close()
            if speci_var == 1:
                model_wr = ModelWrapper(model)
                if modelname == 'gpt2':
                    tb_writer.add_graph(model_wr, inputs)
                elif modelname == 'bert':
                    tb_writer.add_graph(model_wr, inputs['input_ids'])
                tb_writer.close()
                speci_var = 0
            runloss += loss.item()
            runacc += accuracy
            optimizer.state.get("")
            if gg%(0.1*args.showstep)==9:
                time1 = time.time()
                logger.info('\t batch = %d \t loss = %.5f \t acc = %.3f \t cost_time = %.3fs'%(
                             gg,loss.item(),accuracy,time1-time0))
                tb_writer.add_scalar('%s-train-loss'%args.modelname,runloss/0.1/args.showstep,ee*len(train_iter)+gg)
                tb_writer.add_scalar('%s-train-acc'%args.modelname,runacc/0.1/args.showstep,ee*len(train_iter)+gg)
                runloss = 0.0
                runacc = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if gg>100:
                break
            
            if gg % args.showstep == 0:
                save_path1 = os.path.join(args.savemodel.replace('MN',args.modelname),
                                          '%s_%s_%s_step_%d.bin'%(args.modelname,
                                                                  args.corpusname,args.mode,gg))
                if hasattr(model,'module'):
                    model.module.save_pretrained(save_path1)
                else:
                    model.save_pretrained(save_path1)
        save_path2 = os.path.join(args.savemodel.replace('MN',args.modelname),
                                  '%s_%s_%s_epoch_%d.bin'%(args.modelname,
                                                           args.corpusname,args.mode,ee))
        if hasattr(model,'module'):
            model.module.save_pretrained(save_path2)
        else:
            model.save_pretrained(save_path2)
        logger.info('we get model %s'%save_path2)
    logger.info('tensorboard information has been recorded in %s'%tb_dir)
    logger.info('training done!')

train(model=model, corpus=trainset)
tb_writer.close()




       
        
