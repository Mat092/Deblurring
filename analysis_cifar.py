import imageio
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
import seaborn as sns

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

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


def change_model(model_path):

  # create a new model and load the weights for marshall compatiblity.
  model = create_model_unetlike()
  model.load_weights(model_path)

  return model


if __name__ == '__main__':

  cwd = os.getcwd()

  # Models trained of cifar.
  models = ['unetlike_mse_cifar', 'unetlike_mae_cifar', 'unetlike_ssim_cifar',
            'conv2_16filt_mse_cifar', 'conv2_16filt_mae_cifar', 'conv2_16filt_ssim_cifar',
            'conv2_32filt_mse_cifar', 'conv2_32filt_mae_cifar', 'conv2_32filt_ssim_cifar',
            'conv6_16filt_mse_cifar', 'conv6_16filt_mae_cifar', 'conv6_16filt_ssim_cifar'
  ]

  # Loading Test model
  model_name = models[9]
  model_path = os.path.join(cwd, 'models', model_name + '.h5')
  model_data = os.path.join(cwd, 'data', 'hist_' + model_name + '.csv')

  # Loading Training data for metrics
  data = pd.read_csv(model_data)

  # Loading datasets (new images)
  orig_path = os.path.join(cwd, 'data', 'cifar10.npy')
  blur_path = os.path.join(cwd, 'data', 'cifar10_blur_sigma0-3.npy')

  # Load original and blurred datasets, and transform into tf variables
  size = 8000
  orig = np.load(orig_path)[15000: 15000 + size]
  blur = np.load(blur_path)[15000: 15000 + size]

  # Loading Model
  objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}

  if 'unetlike' in model_name :
    loss = model_name.split('_')[1]

    if loss == 'mse':
      loss = 'mean_squared_error'

    elif loss == 'mae':
      loss = 'mean_absolute_error'

    elif loss == 'ssim':
      loss = SSIM_loss

    model = change_model(model_path)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

  else :
    model = tf.keras.models.load_model(model_path, custom_objects=objects)

  # Predict the output for new CIFAR-10
  pred = model.predict(x=blur)

  pred.shape

  # Compute commons metrics
  mse_pred_orig = tf.keras.losses.MSE(orig.reshape(-1, 32*32*3), pred.reshape(-1, 32*32*3)).numpy()
  mse_blur_orig = tf.keras.losses.MSE(orig.reshape(-1, 32*32*3), blur.reshape(-1, 32*32*3)).numpy()

  psnr_pred_orig = PSNR(tf.Variable(orig), tf.Variable(pred)).numpy()
  psnr_blur_orig = PSNR(tf.Variable(orig), tf.Variable(blur)).numpy()

  ssim_pred_orig = SSIM(tf.Variable(orig), tf.Variable(pred)).numpy()
  ssim_blur_orig = SSIM(tf.Variable(orig), tf.Variable(blur)).numpy()

  mask = psnr_blur_orig < 30

  mse_pred_orig[mask].mean()
  psnr_pred_orig[mask].mean()
  ssim_pred_orig[mask].mean()

  (pred[idx]-blur[idx]).min()
  plt.imshow(pred[7014]-blur[7014]);

  psnr_diff = (psnr_pred_orig - psnr_blur_orig)
  mask1 = psnr_diff > 0

  plt.imshow(orig[mask][651]);
  plt.imshow(blur[mask][651]);
  plt.imshow(pred[mask][651]);

  ssim_diff = (ssim_pred_orig - ssim_blur_orig)
  mask2 = ssim_diff > 0

  mask = mask1 & mask2

  psnr_pred_orig.mean()
  psnr_blur_orig.mean()

  plt.scatter(ssim_diff[mask], psnr_diff[mask], marker='.');
