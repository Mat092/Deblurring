import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style('darkgrid')

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

if __name__ == '__main__':

  cwd = os.getcwd()

  orig_path = os.path.join(cwd, 'data', 'cifar10.npy')
  blur_path = os.path.join(cwd, 'data', 'cifar10_blur_sigma0-3.npy')

  # Load original and blurred datasets, and transform into tf variables
  size = 8000
  orig = np.load(orig_path)[:size]
  blur = np.load(blur_path)[:size]

  # # Loading Test model
  # model_name = 'unetlike_mse_cifar.h5'
  # model_path = os.path.join(cwd, 'models', model_name)
  #
  # objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
  # model = tf.keras.models.load_model(model_path, custom_objects=objects)
  #
  # # Predict the output for CIFAR10
  # pred    = model.predict(x=blur)
  # metrics = model.evaluate(x=blur, y=orig)
  #
  # psnr_pred_orig = PSNR(tf.Variable(orig), tf.Variable(pred)).numpy()
  # psnr_blur_orig = PSNR(tf.Variable(orig), tf.Variable(blur)).numpy()
  #
  # ssim_pred_orig = SSIM(tf.Variable(orig), tf.Variable(pred)).numpy()
  # ssim_blur_orig = SSIM(tf.Variable(orig), tf.Variable(blur)).numpy()
  #
  # # mask = psnr_blur_orig < 30
  #
  # psnr_pred_orig.mean()
  # psnr_blur_orig.mean()
  #
  # psnr_diff = (psnr_pred_orig - psnr_blur_orig)
  # mask1 = psnr_diff > 0
  #
  # ssim_diff = (ssim_pred_orig - ssim_blur_orig)
  # mask2 = ssim_diff > 0

#%%
pred = np.load( os.path.join(cwd, 'data', 'cifar10_unet_pred.npy'))

# idx = ssim_diff.argmax()
for idx in [6271, 3822, 7014]:
    fig, ax = plt.subplots(nrows=1, ncols = 3, figsize=(12,8))
    for a, source in zip(ax, [orig, blur, pred]):
      a.imshow(source[idx]);
      a.set_xticks([])
      a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(cwd, 'images', f'CIFAR_{idx}.png'), bbox_inches = 'tight')

#%%
    fig, ax = plt.subplots(nrows=1, ncols = 3, figsize=(12,8))
    for n, (cm, a) in enumerate(zip(['Reds', 'Greens', 'Blues'], ax)):
      a.imshow((pred[idx]-blur[idx])[:,:,n], cmap = cm);
      a.set_xticks([])
      a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(cwd, 'images', f'channel_var_CIFAR_{idx}.png'), bbox_inches = 'tight')

#%%
# idx = psnr_diff.argmax()
idx = 7014
fig, ax = plt.subplots(nrows=1, ncols = 3, figsize=(12,8))
for a, source in zip(ax, [orig, blur, pred]):
  a.imshow(source[idx]);
  a.set_xticks([])
  a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cwd, 'images', 'best_psnr_CIFAR.png'), bbox_inches = 'tight')

#%%
fig, ax = plt.subplots(nrows=1, ncols = 3, figsize=(12,8))
for n, (cm, a) in enumerate(zip(['Reds', 'Greens', 'Blues'], ax)):
  a.imshow((pred[idx]-blur[idx])[:,:,n], cmap = cm);
  a.set_xticks([])
  a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cwd, 'images', 'channel_var_psnr_CIFAR.png'), bbox_inches = 'tight')





# plt.scatter(ssim_diff[mask], psnr_diff[mask], marker='.');
