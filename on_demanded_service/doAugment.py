from datamessenger import DataMessenger
from autocaching import AutoCaching
from threading import Thread
import boto3
import json
import numpy as np
import os, sys
import time

def do_augmentation(Host='aws.elasticache.endpoint:11211', Train_on=True, invocation_name="lambda", r=3):

    s3cli = boto3.client('s3')
    cifar10_meta_name = "./cifar10.meta.s3"
    Bucket = "datasetBucket"
    meta_path = "cifar10/cifar10.meta.s3"
    s3cli.download_file(Bucket, meta_path, cifar10_meta_name)
    with open(cifar10_meta_name, 'r') as meta:
        meta = meta.read()
    meta = json.loads(meta)

    messenger = DataMessenger(meta, 512, Train_on)
    autocache = AutoCaching(meta, Host, invocation_name, Train_on)

    count = 0
    augment_time = time.time()
    length = messenger.__len__()
    rush = r

    labels_dict = {}
    for j in range(rush):
        for i in range(length):
            setTime = time.time()
            ranges, keys = messenger[i] # prepare a set of train/test data
            autocache.autogen_run(ranges, keys)
            print("augment set time : {}".format(time.time() - setTime))
    print("[{}]Done makes data number {}".format(time.time() - augment_time, count))
    return count
