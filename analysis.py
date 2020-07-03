'''
Small script to produce some graph for analysis of obtained data
'''

import os

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

# TODO : I'm very open about plot style
sns.set_style('darkgrid')

model_name = 'tripleConv_ssim_sigmoid_16filters'

datafile   = os.path.join(os.getcwd(), 'data', 'hist_' + model_name + '.csv')

data = pd.read_csv(datafile)

data.

# add column 'epoch' to dataframe
num_rows = data.index[-1] + 1
num_cols = len(data.columns)

epoch_range = np.arange(num_rows) + 1
data['epoch'] = epoch_range

data

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))

for i, name in enumerate(['loss', 'val_loss']) :

  data.plot(ax=ax, label=name, y=name, x='epoch')

#%% Some visual evaluation

model_name = 'conv5_ssim_psnr_32filters'

modelfile = os.path.join(os.getcwd(), 'models', model_name + '.h5')

objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
model   = tf.keras.models.load_model(modelfile, custom_objects=objects)

img_num  = 10101 # which image do u want?

# reshape because predict and evaluate want 4 dim
orig = np.load('data/cifar10.npy')[img_num].reshape(1, 32, 32, 3)
blur = np.load('data/cifar10_blurred_ksize3.npy')[img_num].reshape(1, 32, 32, 3)

# loss, mse, mae, psnr, ssim, mix = model.evaluate(x=blur, y=orig)
loss, mse, mae, psnr, ssim, mix = model.evaluate(x=blur, y=orig)
pred = model.predict(blur)[0]

# Shows the Incredible result SIDE BY SIDE
def big_plot():

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

  fig.suptitle('Metrics -> loss : {:.5}, mse : {:.5}, mae : {:.5}, PSNR : {:.3}. SSIM : {:.3}, MIX : {:.3}'.format(loss, mse, mae, psnr, ssim, mix))

  ax1.imshow(orig[0])
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1.set_title('Original Image')

  ax2.imshow(blur[0])
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_title('Blurred Image')

  ax3.imshow(pred)
  ax3.set_xticks([])
  ax3.set_yticks([])
  ax3.set_title('Predicted Image')

  plt.show();

big_plot();
