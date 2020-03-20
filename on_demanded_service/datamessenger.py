import numpy as np
from threading import Thread
import random

class DataMessenger():
    def __init__(self, meta, batch_size, train_on):
        self.Bucket = meta['bucket']
        self.train_num = meta['train_n']
        self.test_num = meta['test_n']
        self.batch_size = batch_size
        #self.train_set = list(meta['train'].values())
        #self.test_set = list(meta['test'].values())
        #random.shuffle(self.train_set)
        #random.shuffle(self.test_set)
        self.key = 0

        self.train_on = train_on
        #self.autogen = AutoCaching(self.train_set, self.Bucket, "True", self.train_num)
        self.on_epoch_end()

    def __len__(self):
        if self.train_on == True:
            return int(self.train_num / self.batch_size)
        else:
            return int(self.test_num / self.batch_size)

    def __getitem__(self, index):
        ranges = self.selected_range[index * self.batch_size: (index+1) * self.batch_size]
        keys = []
        for i in range(len(ranges)):
            keys.append(self.key)
            self.key += 1
        return ranges, keys

    def on_epoch_end(self):
        if self.train_on == True:
            self.selected_range = np.arange(self.train_num)
        else:
            self.selected_range = np.arange(self.test_num)
