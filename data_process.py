#!/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 Tuya NLP. All rights reserved.
#   
#   FileName：data_process.py
#   Author  ：rentc(桑榆)
#   DateTime：2021/4/24
#   Desc    ：
#
# ================================================================

import jieba
import torch.utils.data as Data
import random


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


if __name__ == "__main__":
    data_path = "./train_data/train"
    batch_size = 4
    nraws = 1000
    epoch = 1
    train_dataset = MyDataset(data_path, nraws, shuffle=False)
    for _ in range(epoch):
        train_dataset.initial()
        train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size)
        for _, data in enumerate(train_iter):
            print(data)
