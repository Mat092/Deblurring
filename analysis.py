'''
Small script to produce some graph for analysis of obtained data
'''

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO : I'm very open about plot style
sns.set_style('darkgrid')

datafile = 'data/history_single_conv_try.csv'

data = pd.read_csv(datafile)

# This line get rid of unnamed: 0, but if in "to_csv" there is index=False, no problem
if 'Unnamed: 0' in data.columns:
  data = data.drop('Unnamed: 0', axis=1)

data

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

model = tf.keras.models.load_model('models/doubleConv_mse.h5')

img_num  = 1000 # which image do u want?

# reshape because predict and evaluate want 4 dim
orig = np.load('data/cifar10.npy')[img_num].reshape(1, 32, 32, 3)
blur = np.load('data/cifar10_blurred_ksize3.npy')[img_num].reshape(1, 32, 32, 3)

metr = model.evaluate(x=blur, y=orig)
pred = model.predict(blur)[0]

# Show the Incredible result SIDE BY SIDE
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

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
