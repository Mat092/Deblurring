'''
Small script to produce some graph for analysis of obtained data
'''

import imageio
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

# TODO : I'm very open about plot style
sns.set_style('darkgrid')

model_name = 'conv4_32filt_reds'

datafile   = os.path.join(os.getcwd(), 'data', 'hist_' + model_name + '.csv')

data = pd.read_csv(datafile)

# add column 'epoch' to dataframe
num_rows = data.index[-1] + 1
num_cols = len(data.columns)

epoch_range = np.arange(num_rows) + 1
data['epoch'] = epoch_range

data.columns

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))

for i, name in enumerate(data.columns[[4, 10]]) :

  data.plot(ax=ax, label=name, y=name, x='epoch')

# #%% Some visual evaluation for CIFAR10 Models
#
# model_name = 'netlike_mse_cifar'
#
# modelfile = os.path.join(os.getcwd(), 'models', model_name + '.h5')
#
# modelfile
#
# objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
# model   = tf.keras.models.load_model(modelfile, custom_objects=objects)
#
# img_num  = 55000 # which image do u want?
#
# # reshape because predict and evaluate want 4 dim
# orig = np.load('data/cifar10.npy')[img_num].reshape(1, 32, 32, 3)
# blur = np.load('data/cifar10_blurred_ksize3.npy')[img_num].reshape(1, 32, 32, 3)
#
# # loss, mse, mae, psnr, ssim, mix = model.evaluate(x=blur, y=orig)
# loss, mse, mae, psnr, ssim, mix = model.evaluate(x=blur, y=orig)
# pred = model.predict(blur)[0]
#
# # Shows the Incredible result SIDE BY SIDE
# def big_plot():
#
#   fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#
#   fig.suptitle('Metrics -> loss : {:.5}, mse : {:.5}, mae : {:.5}, PSNR : {:.3}. SSIM : {:.3}, MIX : {:.3}'.format(loss, mse, mae, psnr, ssim, mix))
#
#   ax1.imshow(orig[0])
#   ax1.set_xticks([])
#   ax1.set_yticks([])
#   ax1.set_title('Original Image')
#
#   ax2.imshow(blur[0])
#   ax2.set_xticks([])
#   ax2.set_yticks([])
#   ax2.set_title('Blurred Image')
#
#   ax3.imshow(pred)
#   ax3.set_xticks([])
#   ax3.set_yticks([])
#   ax3.set_title('Predicted Image')
#
#   plt.show();
#
# big_plot();
#
# #%% Some Visual Evaluations for REDS models.
#
# blur_path = '/media/mattia/70EA383EEA3802C0/Users/Mattia/REDS/train_blur/train/train_blur'
# sharp_path = blur_path.replace('blur', 'sharp')
#
# saving_dir = os.path.join(os.getcwd(), 'images')
#
# model_cifar = 'netlike_mse_cifar.h5'
# model_reds  = 'conv4_16filt_reds.h5'
#
# model_cifar_path = os.path.join(os.getcwd(), 'models', model_cifar)
# model_reds_path  = os.path.join(os.getcwd(), 'models', model_reds)
#
# objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
# model1  = tf.keras.models.load_model(model_cifar_path, custom_objects=objects)
# model2  = tf.keras.models.load_model(model_reds_path,  custom_objects=objects)
#
# clip_num = np.random.randint(low=0, high=249)
# frame    = np.random.randint(low=0, high=100)
#
# blur_image_path  = os.path.join(blur_path, f'{clip_num:03}', f'{frame:08}.png')
# sharp_image_path = os.path.join(sharp_path, f'{clip_num:03}', f'{frame:08}.png')
#
# blur  = (np.asarray(imageio.imread(blur_image_path), dtype='float32') / 255.).reshape(1, 720, 1280, 3)
# sharp = (np.asarray(imageio.imread(sharp_image_path), dtype='float32') / 255.).reshape(1, 720, 1280, 3)
#
# blur_tf  = tf.Variable(blur)
# sharp_tf = tf.Variable(sharp)
#
# blur_psnr = PSNR(blur_tf, sharp_tf)[0]
# blur_ssim = SSIM(blur_tf, sharp_tf)[0]
#
# loss, mse, mae, psnr, ssim, mix = model1.evaluate(x=blur, y=sharp)
# pred = model1.predict(x=blur)
#
# def big_plot_reds():
#
#   fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
#
#   ax1.imshow(sharp[0])
#   ax1.set_xticks([])
#   ax1.set_yticks([])
#   ax1.set_title('Original Image')
#
#   ax2.imshow(blur[0])
#   ax2.set_xticks([])
#   ax2.set_yticks([])
#   ax2.set_title('Blurred Image')
#
#   ax3.imshow(pred[0])
#   ax3.set_xticks([])
#   ax3.set_yticks([])
#   ax3.set_title('Predicted Image')
#
#   plt.show();
#
# print('Metrics -> loss : {:.5}, mse : {:.5}, mae : {:.5}, PSNR : {:.3}. SSIM : {:.3}, MIX : {:.3}'.format(loss, mse, mae, psnr, ssim, mix))
# print(f'Metrics on blur : PSNR : {blur_psnr:.3}, SSIM : {blur_ssim:.3}')
# big_plot_reds();
