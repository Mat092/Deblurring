import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

if __name__ == '__main__':

  cwd = os.getcwd()

  orig_path = os.path.join(cwd, 'data', 'cifar10.npy')
  blur_path = os.path.join(cwd, 'data', 'cifar10_blur_sigma0-3.npy')

  # Load original and blurred datasets, and transform into tf variables
  size = 8000
  orig = np.load(orig_path)[:size]
  blur = np.load(blur_path)[:size]

  # Loading Test model
  model_name = 'unetlike_mse_cifar.h5'
  model_path = os.path.join(cwd, 'models', model_name)

  objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
  model = tf.keras.models.load_model(model_path, custom_objects=objects)

  # Predict the output for CIFAR10
  pred    = model.predict(x=blur)
  metrics = model.evaluate(x=blur, y=orig)

  psnr_pred_orig = PSNR(tf.Variable(orig), tf.Variable(pred)).numpy()
  psnr_blur_orig = PSNR(tf.Variable(orig), tf.Variable(blur)).numpy()

  ssim_pred_orig = SSIM(tf.Variable(orig), tf.Variable(pred)).numpy()
  ssim_blur_orig = SSIM(tf.Variable(orig), tf.Variable(blur)).numpy()

  # mask = psnr_blur_orig < 30

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
