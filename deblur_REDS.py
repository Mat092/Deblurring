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
epochs     = 50

cwd = os.getcwd()

x_train, x_test, y_train, y_test = read_REDS(test_size=0.1)

train = DataGenerator(x_train, y_train, batch_size=batch_size)
test  = DataGenerator(x_test,  y_test,  batch_size=batch_size)

metrics = ['mean_squared_error', 'mean_absolute_error', PSNR, SSIM, MIX]

h, w, c = imageio.imread(x_train[0]).shape

def create_model_conv(conv_num, filters):

  inp = Input((None, None, c))
  x = Conv2D(kernel_size=(3,3), strides=(1,1), filters=filters, padding='same', activation='linear')(inp)

  for _ in range(conv_num):
    x = Conv2D(kernel_size=(3,3), strides=(1,1), filters=filters, padding='same', activation='linear')(x)

  out = Conv2D(kernel_size=(3,3), strides=(1,1), filters=3, padding='same', activation='sigmoid')(x)

  return Model(inp, out)

# objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX}#, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
# model   = tf.keras.models.load_model(save_path, custom_objects=objects)

def train_model_and_save(model, model_name, loss):

  save_path  = os.path.join(cwd, 'models', model_name + '.h5')

  opt = tf.keras.optimizers.Adam(learning_rate=0.001, # keras standard params
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-7
                                 )

  model.compile(optimizer=opt, loss=loss, loss_weights=None, metrics=metrics)

  saveback = ModelCheckpoint(filepath=save_path,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weight_only=False,
                             save_freq='epoch',
                             )

  datafile   = os.path.join(os.getcwd(), 'data', 'hist_{}.csv'.format(model_name))
  customback = CustomCB(datafile)

  model.fit(x=train,
            epochs=epochs,
            validation_data=test,
            verbose=1,
            callbacks=[saveback, customback]
            )


# model = create_model_conv(0, 8)
# model_name = 'conv2_8filt_mse_reds'
# train_model_and_save(model, model_name, 'mean_squared_error')

# model = create_model_conv(0, 8)
# model_name = 'conv2_8filt_mae_reds'
# train_model_and_save(model, model_name, 'mean_absolute_error')

# model = create_model_conv(0, 8)
# model_name = 'conv2_8filt_ssim_reds'
# train_model_and_save(model, model_name, SSIM_loss)
#
# model = create_model_conv(0, 16)
# model_name = 'conv2_16filt_mse_reds'
# train_model_and_save(model, model_name, 'mean_squared_error')
#
# model = create_model_conv(0, 16)
# model_name = 'conv2_16filt_mae_reds'
# train_model_and_save(model, model_name, 'mean_absolute_error')
#
# model = create_model_conv(0, 16)
# model_name = 'conv2_16filt_ssim_reds'
# train_model_and_save(model, model_name, SSIM_loss)

model = create_model_conv(4, 8)
model_name = 'conv6_8filt_mse_reds'
train_model_and_save(model, model_name, 'mean_squared_error')

# model = create_model_conv(4, 8)
# model_name = 'conv6_8filt_mae_reds'
# train_model_and_save(model, model_name, 'mean_absolute_error')
#
# model = create_model_conv(4, 8)
# model_name = 'conv6_8filt_ssim_reds'
# train_model_and_save(model, model_name, SSIM_loss)
