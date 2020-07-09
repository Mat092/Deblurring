import imageio
import os
import random

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

def draw_images(n_clip):

  blur_path = '/media/mattia/70EA383EEA3802C0/Users/Mattia/REDS/train_blur/train/train_blur'
  sharp_path = blur_path.replace('blur', 'sharp')

  clips  = np.random.randint(low=0, high=240, size=(n_clips,))
  frames = np.random.randint(low=0, high=100, size=(n_clips,))

  blur  = [imageio.imread(os.path.join(blur_path,  f'{clip:03}', f'{frame:08}.png')) for clip, frame in zip(clips, frames)]
  sharp = [imageio.imread(os.path.join(sharp_path, f'{clip:03}', f'{frame:08}.png')) for clip, frame in zip(clips, frames)]

  return np.asarray(blur, dtype='float32') / 255., np.asarray(sharp, dtype='float32') / 255.


def draw_patches(n_clips):

  cwd = os.getcwd()

  blur_path  = os.path.join(cwd, 'data', 'redREDS', 'blur')
  sharp_path = os.path.join(cwd, 'data', 'redREDS', 'sharp')

  # chose n_clips random patches
  blur_names  = random.choices(os.listdir(blur_path), k=n_clips)
  sharp_names = [blur_name.replace('blur', 'sharp') for blur_name in blur_names]

  # load same images
  blur  = [imageio.imread(os.path.join(blur_path, name))  for name in blur_names ]
  sharp = [imageio.imread(os.path.join(sharp_path, name)) for name in sharp_names]

  return np.asarray(blur, dtype='float32') / 255., np.asarray(sharp, dtype='float32') / 255.

if __name__ == '__main__':

  n_clips = 300 # max = 240

  cwd = os.getcwd()

  # Loading Test model
  model_name = 'conv2_16filt_mse_reds'
  model_path = os.path.join(cwd, 'models', model_name + '.h5')
  data_path  = os.path.join(cwd, 'data', 'hist_' + model_name + '.csv')

  sns.set_style('darkgrid')

  data = pd.read_csv(data_path)

  data

  if True :

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

    ax1.plot(range(50), data['PSNR'], label='train PSNR')
    ax1.plot(range(50), data['val_PSNR'], label='test PSNR')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('PSNR',  fontsize=15)
    ax1.legend(fontsize=15)

    ax2.plot(range(50), data['SSIM'], label='test SSIM')
    ax2.plot(range(50), data['val_SSIM'], label='test SSIM')
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('SSIM',  fontsize=15)
    ax2.legend(fontsize=15)

    ax3.plot(range(50), data['mean_squared_error'], label='test MSE')
    ax3.plot(range(50), data['val_mean_squared_error'], label='test MSE')
    ax3.set_xlabel('Epoch', fontsize=15)
    ax3.set_ylabel('MSE',  fontsize=15)
    ax3.legend(fontsize=15)

    ax4.plot(range(50), data['mean_absolute_error'], label='test MAE')
    ax4.plot(range(50), data['val_mean_absolute_error'], label='test MAE')
    ax4.set_xlabel('Epoch', fontsize=15)
    ax4.set_ylabel('MAE',  fontsize=15)
    ax4.legend(fontsize=15)

    plt.tight_layout()
    image_name = 'train_plot_reds_conv2'
    plt.savefig(os.path.join(os.getcwd(), 'images', image_name))

  objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
  model   = load_model(model_path, custom_objects=objects)

  # estrae N immagini casuali
  # blur, orig = draw_images(n_clips)

  # Estrae N patch casuali
  blur, orig = draw_patches(n_clips)

  pred = model.predict(x=blur)

  batch, h, w, c = blur.shape

  mse_pred_orig = tf.keras.losses.MSE(orig.reshape(batch, h*w*c), pred.reshape(batch, h*w*c)).numpy()
  mse_blur_orig = tf.keras.losses.MSE(orig.reshape(batch, h*w*c), blur.reshape(batch, h*w*c)).numpy()

  psnr_pred_orig = PSNR(tf.Variable(orig), tf.Variable(pred)).numpy()
  psnr_blur_orig = PSNR(tf.Variable(orig), tf.Variable(blur)).numpy()

  ssim_pred_orig = SSIM(tf.Variable(orig), tf.Variable(pred)).numpy()
  ssim_blur_orig = SSIM(tf.Variable(orig), tf.Variable(blur)).numpy()

  # MI SERVE IL BEST MODEL PER ESTRARRE IMMAGINI

  mask = psnr_blur_orig < 30

  mse_pred_orig[mask].mean()
  psnr_pred_orig[mask].mean()
  ssim_pred_orig[mask].mean()

  idx = (ssim_pred_orig - ssim_blur_orig)

  psnr_pred_orig[6]

  psnr_pred_orig[idx > 0]

  plt.imshow(orig[6])
  plt.imshow(blur[6])
  plt.imshow(pred[6])
