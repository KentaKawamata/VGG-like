from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, BatchNormalization
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import sys
import numpy as np
from keras.layers.advanced_activations import PReLU

def plot_history(history):

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

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

    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)

    # 画像のピクセル値を0-1に正規化
    #train_data /= 255.0
    #test_data /= 255.0

    return train_data, test_data, train_target, test_target

def VGG16(shape):
    
    model = Sequential()
    model.add(InputLayer(input_shape=shape))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
     
    model.add(Flatten())
    model.add(Dense(512))
    model.add(PReLU())
    model.add(Dense(512))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
     
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    return model

if __name__=="__main__":
    
    batch = 128
    num_classes = 10
    epochs = 100
    row, col = 32, 32
    input_shape = (row, col, 3)

    # CIFAR-10データをロード
    print("load CIFAR-10 dataset")
   
    x_train, x_test, y_train, y_test = load_cifar10("cifar-10-batches-py")

    # クラスラベル（0-9）をone-hotエンコーディング形式に変換
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    X_train = x_train.reshape((len(x_train), row, col, 3))
    X_test = x_test.reshape((len(x_test), row, col, 3))
    X_train /= 255
    X_test /= 255

    model = VGG16(input_shape)
    early = EarlyStopping()
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    #history = model.fit(X_train, Y_train, batch_size=batch, epochs=epochs, validation_data=(X_test, Y_test), callbacks=[early])
    history = model.fit(X_train, Y_train, batch_size=batch, epochs=epochs, validation_data=(X_test, Y_test))
    plot_history(history)

    print('save the architecture of a CNN model')
    json_string = model.to_json()
    open('vgg_model.json', 'w').write(json_string)
    print('save weight datasets!!')
    model.save_weights('vgg_model_weights.h5')
