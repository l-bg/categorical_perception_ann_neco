'''
Categorical Perception: A Groundwork for Deep Learning
Laurent Bonnasse-Gahot & Jean-Pierre Nadal
Python script for reproducing the results presented in Figure D.2 (Classification accuracy on the CIFAR-10 image dataset (test set) using a multi-layer perceptron with varying levels of dropout.).
'''

import os
import random
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1,
                    help='seed')
args = parser.parse_args()

seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_classes = 10
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train = tensorflow.keras.utils.to_categorical(y_train, n_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, n_classes)

input_shape = x_train.shape[1:]

n_epochs = 1000
n_hid = 1024
batch_size = 128
n_layers = 2

dropout_rg = np.linspace(0., 0.7, 8)

def create_model(n_hid, n_layers, dropout_layers):
    input_x = Input(shape=input_shape)
    x = Dropout(dropout_layers[0])(input_x)
    for k in range(n_layers):
        x = Dense(n_hid, activation='relu')(x)
        x = Dropout(dropout_layers[k+1])(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(input_x, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model


df_results = []
for layer_i in range(n_layers+1):
    print('layer = {}'.format(layer_i))
    for dropout in dropout_rg:
        print(('--- p = {}'.format(dropout)))
        dropout_layers = 0.2*np.ones(n_layers + 1)
        dropout_layers[layer_i] = dropout
        model = create_model(n_hid=n_hid, n_layers=n_layers, dropout_layers=dropout_layers)
        history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=n_epochs,
                validation_data=(x_test, y_test),
                verbose=0)

        df = pd.DataFrame(history.history)
        df['epoch'] = df.index + 1
        df['dropout'] = dropout
        df['layer'] = layer_i
        df_results.append(df)


if not os.path.exists('csv'):
    os.makedirs('csv')

pd.concat(df_results).to_csv('csv/cifar10_mlp_dropout_seed_{}.csv'.format(seed),
                             index=False)
