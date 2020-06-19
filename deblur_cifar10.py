'''
First example of deblurring using as single layer Dense model.
'''

import os

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pr

# Constants
batch_size = 100
epochs     = 100

# Input image dimensions
img_rows, img_cols, channels = 32, 32, 3
inputs  = img_rows * img_cols * channels
outputs = img_rows * img_cols * channels

save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'single_dense.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.

dataset = np.concatenate([x_train, x_test], axis=0)
target  = pr.blur_input(dataset, k_size=10 )

# MODEL CREATION
inp   = Input(shape=inputs)
x     = Dense(units=32*32*3, activation='linear')(inp)
model = Model(inp, x)

opt = tf.keras.optimizers.SGD(lr=0.01)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error', 'binary_crossentropy'])
