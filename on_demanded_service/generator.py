#import tensorflow as tf
#tf.keras.utils.Sequence
import time
import tensorflow
from PIL import Image
from threading import Thread
import numpy as np
import autoaugment
import json
import os, sys
import elasticache_auto_discovery
from pymemcache.client.hash import HashClient
from tensorflow.keras.utils import to_categorical
import random

class PluckCache(tensorflow.keras.utils.Sequence):
    # In Cifar 10 mean and std has [120.70756512369792, 64.15007589112136]
    def __init__(self,
                 batch_size=64,
                 Host='default.url.ElasticacheService:11211',
                 normalization=[120.70756512369792, 64.15007589112136],
                 image_size=32,
                 data_num=50000, #cifar-10 has a got images 50,000.
                 num_class=10
                 ):
        self.Host = Host # memcached host
        nodes = elasticache_auto_discovery.discover(self.Host)
        nodes = map(lambda x: (x[0], int(x[2])), nodes)
        self.memcached = HashClient(nodes, use_pooling=True, max_pool_size=5)
        self.batch_size = batch_size
        self.normalization = [120.70756512369792, 64.15007589112136]
        self.data_num = data_num
        self.size = image_size
        self.num_class = num_class
        self.on_epoch_end()
        # prepare for batch_data object.
        self.X = np.zeros((self.batch_size, self.size, self.size, 3),dtype=np.uint8) # images
        self.Y = np.zeros((self.batch_size, ), dtype=np.uint8) # labels

    def memget(self, key):
        x = self.memcached.get(str(key))
        decode = np.frombuffer(x, dtype=np.uint8)
        decode = decode.reshape(self.size, self.size, 4)
        decode = decode.transpose(2,0,1)

        y = decode[-1][0][0]
        x = decode[:3]
        x = x.transpose(1,2,0)
        del decode
        return x, y

    def __len__(self):
        return int(self.data_num / self.batch_size)

    def __getitem__(self, index):
        iterTime = time.time()
        ranges = self.selected_range[index * self.batch_size: (index+1) * self.batch_size]
        images, labels = self._generate(ranges)
        return images, labels

    def on_epoch_end(self):
        self.selected_range = np.arange(self.data_num)
        random.shuffle(self.selected_range)

    def _thread_generator(self, ranges):
        for i, key in enumerate(ranges):
            x, y = self.memget(key)
            self.X[i] = x
            self.Y[i] = y
        return

    def _worker(self, ranges):
        th = Thread(target=self._thread_generator, args=(ranges,))
        th.start()
        th.join()
        return

    def _generate(self, ranges):
        self._worker(ranges)
        onehot = to_categorical(self.Y, self.num_class)
        X = (self.X - self.normalization[0]) / (self.normalization[1] + 1e-7)
        return X, onehot
