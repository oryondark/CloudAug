import tensorflow
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
print(tensorflow.__version__)

from generator import PluckCache
from tensorflow.keras.datasets import cifar10 # Keras official library
from tensorflow.keras.optimizers import Adam
from doAugment import *
import json

'''
layer  output  filter
conv1 32 x 32 3x3x16
conv2 32 x 32 3x3x16xfactor
conv3 16 x 16 3x3x32xfactor
conv4 8 x 8   3x3x64xfactor
avg   1 x 1   8x8
'''
def wideBlock(x, output_ch, dropout, stride):
    x = BatchNormalization()(x)
    x = Conv2D(output_ch, 3, padding="SAME", strides=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = Conv2D(output_ch, 3, padding="SAME", strides=stride, activation="relu")(x)
    return x

def wideLayer(x, output_ch, num_of_block, rate,stride):
    strides = [stride] + [1]*(num_of_block - 1)
    for i, s in enumerate(strides):
        y = wideBlock(x, output_ch, rate, s)
        if s != 1:
            o = Conv2D(output_ch, 1, padding="SAME", strides=stride)(x)
            y = tensorflow.keras.layers.add([o, y])
        y = Activation('relu')(y)
    return y


def ResNet(input_shape, depth, factor, num_classes):
    assert ((depth-4)%6 ==0), 'should be 6n+4'
    n = int((depth-4) / 6) # number of blocks
    outchannel_per_steps = [16*factor, 32*factor, 64*factor] # wide factors
    drop_rate = 0.3

    batch = Input(shape=input_shape) # inputs

    x = Conv2D(16, 3, strides=1, padding="SAME", input_shape=(None,32,32,3))(batch) # No use ReLU

    ##### Wide ResNet #####
    x = wideLayer(x, outchannel_per_steps[0], n, drop_rate, 1)
    x = wideLayer(x, outchannel_per_steps[1], n, drop_rate, 2)
    x = wideLayer(x, outchannel_per_steps[2], n, drop_rate, 2)
    #####     End     #####

    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x) # outputs
    model = Model(inputs=batch, outputs=x)

    return model

# create model
model = ResNet(input_shape=(32,32,3), depth=28, factor=10, num_classes=10)
model.summary()

# data augment & caching
Train_HOST = 'Train_dataset_elasticache_address:11211'
data_num = do_augmentation(Train_HOST, True, 'hjkim_autoaugment') # store for your train data set.
Test_HOST = 'Train_dataset_elasticache_address:11211'
data_num = do_augmentation(HOST, False, 'hjkim_validata_caching') # store test data set.
batch_size = 64


# start train
train_gen = PluckCache(batch_size, Train_HOST, num_class=10)
test_gen = PluckCache(batch_size*2, Test_HOST, num_class=10)

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


#opt_rms = tensorflow.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule(0)), metrics=['accuracy'])
train_time = time.time()
visuallization = tensorflow.keras.callbacks.TensorBoard(log_dir='./logs/serverlessautoaug', batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)
model.fit_generator(train_gen,\
                    steps_per_epoch=150000 // batch_size,epochs=125,\
                    verbose=1, validation_data=test_gen, \
                    callbacks=[visuallization, LearningRateScheduler(lr_schedule)])
print("learntime = {}".format(time.time() - train_time))
