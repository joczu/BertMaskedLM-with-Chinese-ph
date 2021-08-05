#!/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 Tuya NLP. All rights reserved.
#   
#   FileName：test.py
#   Author  ：shizi
#   DateTime：2021/7/15
#   Desc    ：对训练好的模型进行效果测试，能够统一Bert和Gpt2模型测试。
#
# ================================================================
import torch
import logging
import argparse
import torch.utils.data as Data
from datetime import datetime
from transformers import GPT2LMHeadModel,BertForMaskedLM
from utils import (calculate_loss_and_accuracy,GptTokenTool,
                  BertTokenTool,MyDataset,top_accuracy,day_month)
from utils import ppl_based_union_ngrams_lm as perplexity
from utils import calculate_last_logits_and_loss as cllal
from sklearn import decomposition
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import tensorflow as tf
import numpy as np
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import os
import time
import pdb

def parser_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname',type=str,default='bert',
                        help='input your test model name, bert or gpt2')
    parser.add_argument('--loadmodel', type=str, 
                        default='./modelfile/bert/bert_sougou_word_step_1000.bin',
                        help="load trained model for test its profiler.")
    parser.add_argument('--testfile', type=str, 
                        default="./testdata/stard1w.txt", 
                        help='test data path')
    parser.add_argument('--vocabpath', type=str, 
                        default='./vocab/SougouBertVocab.txt', 
                        help='vocab file path')
    parser.add_argument('--batchsize', type=int, default=2, help='batch_size')
    parser.add_argument('--showstep', type=int, default=100, help='save model after train N steps ')
    parser.add_argument('--usegpu', type=int,default=0, help='use GPU or not')
    parser.add_argument('--device', type=str, default='0',
                        help="""if usegpu and only a sigle gpu, you can ignore 
                        the item, but if you use multi gpu, you should set the 
                        item='0,1' for two pieces of gpu, item='0,1,2' for 
                        three pieces of gpu and so on""")
    parser.add_argument('--log', type=str, default='./log/MN/CN_DT_test_log.txt')
    parser.add_argument('--tensorboard',type=str,default='./tensorboard/MN/CN_DT_test',
                        help='tensorboard directory')
    parser.add_argument('--demodata',type=str,default='./test/demodata.xls',
                        help='sure your demo data file address')
    parser.add_argument('--save_demo',type=str,
                        default='./demo/MN/MN_CN_MO_DT_L_perplexity.xls',
                        help='''draw the (log) perplexity distribution''')
    parser.add_argument('--isppl',type=int,default=0,
                        help='whether or not calculate perplexity')
    parser.add_argument('--ppltype',type=str,default='log',
                        help='ppltype = log or ori')
    parser.add_argument('--islastone',type=int,default=0,
                        help='if test the generate lastone task')
    args = parser.parse_args()
    return args

args = parser_args()
assert args.loadmodel,'test.py need a trained model'
assert args.usegpu in [0,1],'usegpu is equal 0 or 1'
assert args.isppl in [0,1], 'isppl is equal 0 or 1'
assert args.islastone in [0,1], 'islastone is equal 0 or 1'
assert args.modelname in args.loadmodel, "let your args.modelname in args.loadmodel"

day = day_month(datetime.now())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logfile = args.log
corpusname = args.testfile.split('/')[-1]
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
tb_writer = SummaryWriter(log_dir = tb_dir)

if args.modelname=='gpt2':
    model = GPT2LMHeadModel.from_pretrained(args.loadmodel)
    tokenizer = GptTokenTool(args.vocabpath)
else:
    model = BertForMaskedLM.from_pretrained(args.loadmodel)
    tokenizer = BertTokenTool(args.vocabpath)
    
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


testset = MyDataset(args.testfile, n_raws=500, shuffle=False)
time0 = time.time()
logger.info("Test is begining at %s"%day)
logger.info("The test model is : %s"%args.loadmodel)
logger.info("The logger information is saved in : %s"%logfile)

testset.initial()
total_samplers = len(testset)
test_iter = Data.DataLoader(dataset=testset, batch_size=args.batchsize)


#%% 增加句子表示功能  已完成，可视化有效
# ori = '即想 办法 在 有限 时间 内 增加 运动 速度'
# sen,_ = tokenizer.tokenize(ori, max_length=100)
# output = model(sen)
# logit = output.logits.squeeze()
# sen_logit = logit.mean(0,keepdim=True)
# sen_logit2 = (sen_logit + 0.1)/2
# ssen = torch.cat([sen_logit,sen_logit2],dim=0)
# tb_writer.add_embedding(ssen,['sentence1','sentence2'],global_step=1)
# tb_writer.close()



#%% 测试
top5 = 0
top3 = 0
top1 = 0
ppl = [] if args.isppl else None
tot = 0

#sentence_num = 1000
PCA = decomposition.PCA(128)
sen_vector = []
sen_label = []
transfer = tokenizer.tokenizer.convert_ids_to_tokens
# 由于bert的masked是随机的，并且一个batch内有的sample可能有两个masked，有的可能只有一个，
# 不好结构化，因此面对bert的topk测试，使用了枚举法。
# bert的困惑度不好搞，目前不支持困惑度。
for gg, data in enumerate(test_iter):
    ratio = (gg+1)*args.batchsize/total_samplers
    if args.modelname == 'gpt2':
        inputs,labels = tokenizer.tokenize(data, max_length=100)
        if str(device)=='cuda':
            inputs, lables = inputs.to(device), labels.to(device)
        outputs = model.forward(input_ids = inputs)
        if args.islastone:
            last_logits, last_labels, loss = cllal(outputs, labels)
            top5 += top_accuracy(last_logits, last_labels, 5)
            top3 += top_accuracy(last_logits, last_labels, 3)
            top1 += top_accuracy(last_logits, last_labels, 1)
            tot += last_labels.numel()
            if gg%(args.showstep)==99:
                time1 = time.time()
                logger.info('\t batch = %d \t complete = %.3f%% \t loss = %.3f \
                            \t top1 = %.1f%% \t top3 = %.1f%% \t top5 = %.1f%% \
                            \t cost_time = %.1fs\n'.replace('  ','')%(
                            gg,ratio*100,loss.item(),top1*100/tot,top3*100/tot,
                            top5*100/tot,time1-time0))
                tb_writer.add_scalar('%s-lastone-loss'%args.modelname,loss.item(),gg)
                tb_writer.add_scalar('%s-lastone-top1'%args.modelname,top1/tot,gg)
                tb_writer.add_scalar('%s-lastone-top3'%args.modelname,top3/tot,gg)
                tb_writer.add_scalar('%s-lastone-top5'%args.modelname,top5/tot,gg)
                tb_writer.close()
        else:
            if gg%args.showstep==99:
                time1 = time.time()
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=labels)
                logger.info('\t batch = %d \t complete = %.3f%% \t loss = %.3f \
                              \t acc = %.2f%% \t cost_time=%.1fs'.replace('  ','')%(
                            gg,ratio*100,loss.item(),accuracy*100,time1-time0))
                tb_writer.add_scalar('%s-testloss',loss.item(),gg)
                tb_writer.add_scalar('%s-testacc',accuracy,gg)
                tb_writer.close()
    elif args.modelname == 'bert':
        if args.islastone:
            inputs, labels = tokenizer.tokenize(data, max_length=100, p_mask = 0, islastone=True)
            inputs = inputs.to(device) if str(device)=='cuda' else inputs
            labels = labels.to(device) if str(device)=='cuda' else labels
        else:
            inputs, labels = tokenizer.tokenize(data, max_length=100, p_mask = 0.15)
            inputs = inputs.to(device) if str(device)=='cuda' else inputs
            labels = labels.to(device) if str(device)=='cuda' else labels
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.mean() if multi_gpu else outputs.loss
        masked_label = labels[labels != -100]
        #pdb.set_trace()
        masked_pre5 = outputs.logits[labels != -100].topk(5).indices
        top5 += sum([masked_label[i].item() in masked_pre5[i] for i in range(len(masked_label))])
        top3 += sum([masked_label[i].item() in masked_pre5[i,:3] for i in range(len(masked_label))])
        top1 += sum([masked_label[i].item() in masked_pre5[i,:1] for i in range(len(masked_label))])
        tot += masked_label.numel()
        if gg%args.showstep==99:
            time1 = time.time()
            logger.info('\t batch = %d \t complete = %.1f%% \t loss = %.3f \t top1 = %.1f%% \t top3 = %.1f%%'
                        '\t top5 = %.1f%% \t cost_time = %.1fs'%(gg,ratio*100,loss.item(),top1*100/tot,
                          top3*100/tot,top5*100/tot,time1-time0))
            tb_writer.add_scalar('%s-testloss'%args.modelname,loss.item(),gg)
            tb_writer.add_scalar('%s-testtop1'%args.modelname,top1/tot,gg)
            tb_writer.add_scalar('%s-testtop3'%args.modelname,top3/tot,gg)
            tb_writer.add_scalar('%s-testtop5'%args.modelname,top5/tot,gg)
            tb_writer.close()
            
        vinput = tokenizer.tokenizer(data,return_tensors="pt",max_length=100,
                                     padding=True,truncation=True)
        vout = model(**vinput)
        vlogits = vout.logits.mean(1)  # batch * L * V -> batch * V
        # PCA.fit(vlogits.detach().numpy())
        # print(PCA.explained_variance_ratio_.sum())
        sen_vector.append(vlogits)
        sen_label.append(data)
        if gg>1000:
            break
# svector = np.vstack(sen_vector)

svector = torch.stack([j for i in sen_vector for j in i])
PCA.fit(svector.detach().numpy())
print(PCA.explained_variance_ratio_.sum())
svector2 = PCA.transform(svector.detach().numpy())
slabel = [j for i in sen_label for j in i]

svector2 = svector2[:200]
slabel = slabel[:200]
tb_writer.add_embedding(mat=svector2,metadata=slabel,global_step=1)
tb_writer.close()


#%%  

data = np.load('./蚂蚁金服.npy')
txt = '/Users/desktop/dataset/句子对数据/蚂蚁金服m.txt'
with open('/Users/shizi/desktop/dataset/句子对数据/蚂蚁金服m.txt','r') as f:
    data_l = f.read()
    data_l = data_l.split('/n')



