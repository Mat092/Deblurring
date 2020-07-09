'''
First example of deblurring using simple models.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Activation, Flatten, Input, Lambda, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import SSIM, PSNR, SSIM_loss, SSIM_multiscale_loss, MIX, SSIM_PSNR
import preprocessing as pr
from callbacks import CustomCB

# Constants
batch_size = 64
epochs     = 50
size       = 10000   # number of images used

cwd = os.getcwd()

dataset = np.load('data/cifar10.npy')[:size]
blurred = np.load('data/cifar10_blur_sigma0-3.npy')[:size]

x_train, x_test, y_train, y_test = pr.train_test(blurred[:size], dataset[:])

metrics = ['mean_squared_error', 'mean_absolute_error', PSNR, SSIM, MIX]

def create_model_unetlike():

  inp = Input(shape=(32, 32, 3))
  conv1 = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=8 , padding='same', activation='relu')(inp)
  mp1   = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
  # 16x16x8
  conv2 = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=16, padding='same', activation='relu')(mp1)
  mp2   = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)
  # 8x8x16
  conv3 = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=48, padding='same', activation='relu')(mp2)
  # 8x8x48
  ps2   = Lambda(lambda x : tf.nn.depth_to_space(x, 2, data_format='NHWC'))(conv3)
  # 16x16x12
  conc2 = Concatenate(axis=-1)([conv2, ps2])
  conv4 = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=12, padding='same', activation='relu')(conc2)
  ps1   = Lambda(lambda x : tf.nn.depth_to_space(x, 2, data_format='NHWC'))(conv4)
  # 32x32x3
  conc1 = Concatenate(axis=-1)([conv1, ps1])
  conv5 = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=3, padding='same', activation='sigmoid')(conc1)

  return Model(inp, conv5)

def create_model_conv(conv_num, filters):

  inp = Input((32, 32, 3))
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

  model.fit(x=x_train, y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1,
            shuffle=True,
            callbacks=[saveback, customback]
            )

objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss } # 'SSIM_PSNR' : SSIM_PSNR}

# Already have mse
# model_name = 'unetlike_mse_cifar'
# model = create_model_unetlike()
# model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
# train_model_and_save(model, model_name, 'mean_squared_error')

# model_name = 'unetlike_mae_cifar'
model = create_model_unetlike()
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_squared_error')

model_name = 'unetlike_ssim_cifar'
# model = create_model_unetlike()
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, SSIM_loss)

model_name = 'conv2_16filt_mse_cifar'
# model = create_model_conv(0, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_squared_error')

model_name = 'conv2_16filt_mae_cifar'
# model = create_model_conv(0, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_absolute_error')

model_name = 'conv2_16filt_ssim_cifar'
# model = create_model_conv(0, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, SSIM_loss)

model_name = 'conv2_32filt_mse_cifar'
# model = create_model_conv(0, 32)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_squared_error')

model_name = 'conv2_32filt_mae_cifar'
# model = create_model_conv(0, 32)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_absolute_error')

model_name = 'conv2_32filt_ssim_cifar'
# model = create_model_conv(0, 32)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, SSIM_loss)

model_name = 'conv6_16filt_mse_cifar'
# model = create_model_conv(4, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_squared_error')

model_name = 'conv6_16filt_mae_cifar'
# model = create_model_conv(4, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, 'mean_absolute_error')

model_name = 'conv6_16filt_ssim_cifar'
# model = create_model_conv(4, 16)
model = tf.keras.models.load_model(os.path.join(cwd, 'models', model_name + '.h5'), custom_objects=objects)
train_model_and_save(model, model_name, SSIM_loss)
