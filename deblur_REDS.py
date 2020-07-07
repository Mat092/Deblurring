import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import imageio

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import SSIM, PSNR, SSIM_loss, SSIM_multiscale_loss, MIX, SSIM_PSNR
from callbacks import CustomCB
from generator import DataGenerator
from preprocessing import read_REDS

batch_size = 32
epochs     = 100

model_name = 'conv4_16filt_reds_3x3'
save_path  = os.path.join(os.getcwd(), 'models', model_name + '.h5')

x_train, x_test, y_train, y_test = read_REDS(test_size=0.1)

train = DataGenerator(x_train, y_train, batch_size=batch_size)
test  = DataGenerator(x_test,  y_test,  batch_size=batch_size)

# read a sample patch to retrieve image size
h, w, c = imageio.imread(x_train[0]).shape

inp   = Input(shape=(None, None, c))
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=16, padding='same', activation='linear')(inp)
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=16, padding='same', activation='linear')(x)
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=16, padding='same', activation='linear')(x)
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=3 , padding='same', activation='sigmoid')(x)
model = Model(inp, x)

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001, # keras standard params
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-7
                               )

# opt = tf.keras.optimizers.SGD(learning_rate=1., momentum=.9, nesterov=True) # keras standard params

metrics = ['mean_squared_error', 'mean_absolute_error', PSNR, SSIM, MIX]

model.compile(optimizer=opt, loss='mean_squared_error', loss_weights=None, metrics=metrics)

saveback = ModelCheckpoint(filepath=save_path,
                           monitor='val_loss',
                           save_best_only=True,
                           save_weight_only=False,
                           save_freq='epoch',
                           )

datafile   = os.path.join(os.getcwd(), 'data', 'hist_{}.csv'.format(model_name))

# custom back saves all the info we need in pandas dataframe.
customback = CustomCB(datafile)

history = model.fit(x=train,
                    y=None,
                    epochs=epochs,
                    validation_data=test,
                    verbose=1,
                    shuffle=True,
                    callbacks=[saveback, customback],
                    )
