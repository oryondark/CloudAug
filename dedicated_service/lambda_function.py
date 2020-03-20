import numpy as np
from PIL import Image
import autoaugment
import boto3
import json
import os, sys

#AWS Elasticache Service - Auto discovery
import elasticache_auto_discovery
from pymemcache.client.hash import HashClient

s3 = boto3.client('s3')
FaaS = boto3.client('lambda')
policy = autoaugment.CIFAR10Policy()

def lambda_handler(event, context):
    '''
    summary:
    요청을 통해 전송되는 데이터는 Json 데이터이며,
    S3 object 안에 미리 전처리하기 위한 이미지 파일이 존재합니다.
    해당 이미지 파일의 URL 정보들은 특수한 CSV file로 드릴 것 입니다.
    Json format은 다음을 준수해주세요.

    {
        object_path : s3_path,
        bucekt : bucket_name,
        cache_idx :  to_cache_index_number ( for search index )
        label_name : image_label
        host : endpoint_for_memcached
    }


    '''

    bucket_name = event['bucket_name']
    cache_idx = int(event['cache_idx'])
    object_path = event['object_path']
    label = int(event['label_name'])
    elasticache_host = event['host']
    print("set host : {}".format(elasticache_host))
    print("set bucket : {}".format(bucket_name))

    elasticache_config_endpoint = elasticache_host
    nodes = elasticache_auto_discovery.discover(elasticache_config_endpoint)
    nodes = map(lambda x: (x[0], int(x[2])), nodes)
    print(nodes)
    memcached = HashClient(nodes, use_pooling=True, max_pool_size=5)

    print("To set index number in memory : ".format(cache_idx))
    tmp = '/tmp/image_dump.png'
    s3cli = boto3.client('s3')
    s3cli.download_file(bucket_name, object_path, tmp)

    image = Image.open(tmp)
    image = policy(image)
    image = np.array(image, dtype=np.uint8)

    y = np.zeros((32,32,1))
    y[0][0][0] = label
    y.astype(np.uint8)

    concated = np.concatenate((image, y), axis=2)
    print("concatenated shape : {}".format(concat.shape))
    concated = concated.astype(np.uint8)
    concated = concated.tobytes()
    
    #caching
    '''
    Decode usage:
    decoded = np.frombuffer(concat)
    decode = decoded.transpose(2,0,1)
    label = decoded[3][0][0]
    image = decoded[:3]
    image = image.transpose(1,2,0)
    '''
    ret = memcached.set(str(cache_idx), concat) # idx : key, concat : value
    return json.dumps({'state':ret})
