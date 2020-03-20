import boto3
import os, sys
from threading import Thread
import json
import time
import random

s3cli = boto3.client('s3')
s3res = boto3.resource('s3')
faas = boto3.client('lambda')
class PrePareCacheService():
    def __init__(self, meta, host, lambda_handler, train_on):
        self.Bucket = meta['bucket']
        self.train_num = meta['train_n']
        self.test_num = meta['test_n']
        #self.batch_size = batch_size
        self.train_set = list(meta['train'].values())
        self.test_set = list(meta['test'].values())
        random.shuffle(self.train_set)
        random.shuffle(self.test_set)
        self.host = host
        self.train_on = train_on
        self.lambda_handler = lambda_handler

    def _generate_trainset(self, idx, key):
        '''
        idx : train dataset index number
        key : to store key name in memcached.
        '''
        _,_, label, imgname = self.train_set[idx].split("/")
        parameter = {"bucket_name": self.Bucket,
                     "cache_idx" : str(key),
                     "object_path" : self.train_set[idx],
                     "label_name" : str(label),
                     "host" : self.host}
        #print(parameter)
        res = faas.invoke(FunctionName=self.lambda_handler,
                          InvocationType="Event",
                          Payload=json.dumps(parameter))
        return

    def _generate_testset(self, idx, key):
        _,_, label, imgname = self.test_set[idx].split("/")
        parameter = {"bucket_name": self.Bucket,
                     "cache_idx" : str(key),
                     "object_path" : self.test_set[idx],
                     "label_name" : str(label),
                     "host" : self.host}
        #print(parameter)
        res = faas.invoke(FunctionName=self.lambda_handler,
                          InvocationType="Event",
                          Payload=json.dumps(parameter))

        return


    def _train_worker(self, ranges, keys):
        for idx, key in zip(ranges, keys):
            self._generate_trainset(idx, key)

        return

    def _test_worker(self, ranges, keys):
        for idx, key in zip(ranges, keys):
            self._generate_testset(idx, key)

        return

    def thread_run(self, ranges, keys):
        #print("Made by hjkim Threading to generate data")
        if self.train_on == True:
            th1 = Thread(target=self._train_worker, args=(ranges, keys))
        else:
            th1 = Thread(target=self._test_worker, args=(ranges, keys))
        th1.start()
        th1.join()
        start = time.time()
        #print("[{}]Done ! ranges : {}".format(time.time() - start, ranges))
        return 200
