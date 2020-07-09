import os
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

# TODO : I'm very open about plot style
sns.set_style('darkgrid')

model_name = 'unetlike_mse_cifar'

datafile   = os.path.join(os.getcwd(), 'data', 'hist_' + model_name + '.csv')

data = pd.read_csv(datafile)
data
# add column 'epoch' to dataframe
num_rows = data.index[-1] + 1
num_cols = len(data.columns)

epoch_range = np.arange(num_rows) + 1
data['epoch'] = epoch_range

for cn in data.columns:
    print(cn)

#%%
columns = data.columns[[1,2,3,4,7,8,9,10]]
names = ['PSNR', 'SSIM', 'MAE', 'MSE', 'test_PSNR', 'test_SSIM', 'test_MAE', 'test_MSE']
n_plot = len(columns)//2

n_plot

fig, ax = plt.subplots(ncols=n_plot//2, nrows=n_plot//2, figsize=(5*n_plot, 10))
# plt.suptitle('Metrics during U-like Net training', fontsize=35)
for i, name in enumerate(columns) :
  a = ax.flatten()[i%n_plot]
  data.plot(ax=a, label=names[i], y=name, x='epoch')
  a.legend(fontsize=15)
plt.tight_layout()
plt.savefig('images/Lplot_U-like.png')
