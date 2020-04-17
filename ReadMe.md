# Memcached strategy with Serverless Computing

### AWS Cloud Environment
1) AWS Elasticache(Memcached) in VPC Network <br>
2) EC2 g3s.xlarge can connect to Memcached on VPC <br>
3) AWS Lambda <br>

### Frameworks & Library
1) Tensorflow 1.14 gpu <br>
2) pymemcache for access memcached
3) numpy, tensorboard(for visualization)
4) threading

### Usage
[train_test.py](on_demanded_service/train_test.py)
```python
# WideResNet
model = ResNet(input_shape=(32,32,3), depth=28, factor=10, num_classes=10)

# AutoAugment and Caching
Train_HOST = 'Train_dataset_elasticache_address:11211'
#invocate AWS Lambda
data_num = do_augmentation(Train_HOST, True, 'augment_AWSlambda_name') # store for your train data set.

#make Generator
train_gen = PluckCache(batch_size, Train_HOST, num_class=10)

#start train
model.fit_generator(train_gen)
```

### Contributor
Hyunjune Kim. - email is '4u_olion@naver.com'<br>
Kyungyong Lee. - my professor is him, an assistant professor in Kookmin University.<br>
