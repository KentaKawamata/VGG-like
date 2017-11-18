from __future__ import print_function
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import sys
import numpy as np

 
#(Z_train, a_train), (Z_test, a_test) = cifar10.load_data()
#print(Z_train.shape)


def unpickle(f):
    import _pickle as pickle
    fp = open(f, 'rb')
    
    if sys.version_info.major==2:
        data = pickle.load(fp)

    elif sys.version_info.major==3:
        data = pickle.load(fp, encoding='latin-1')

    fp.close()
    return data

def load_cifar10(datadir):
    train_data = []
    train_target = []

    # 訓練データをロード
    for i in range(1, 6):
        d = unpickle("%s/data_batch_%d" % (datadir, i))
        train_data.extend(d["data"])
        train_target.extend(d["labels"])

    # テストデータをロード
    d = unpickle("%s/test_batch" % (datadir))
    test_data = d["data"]
    test_target = d["labels"]

    #print(train_data.shape)
    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)

    print(train_data.shape)
    # 画像のピクセル値を0-1に正規化
    #train_data /= 255.0
    #test_data /= 255.0

    return train_data, test_data, train_target, test_target

if __name__=="__main__":

    num_classes = 10

    # CIFAR-10データをロード
    print("load CIFAR-10 dataset")
   
    X_train, X_test, y_train, y_test = load_cifar10("cifar-10-batches-py")

    print(X_train.shape)

    # クラスラベル（0-9）をone-hotエンコーディング形式に変換
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    X_train = X_train.reshape((len(X_train), 48, 48, 3))
    X_test = X_test.reshape((len(X_test), 48, 48, 3))

    # 画像のピクセル値を0-1に正規化
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    print(X_train.shape)
    print(Y_train.shape)    
