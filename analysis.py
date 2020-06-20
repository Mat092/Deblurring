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

datafile = 'data/history_single_conv_try'

data = pd.read_csv(datafile)

data


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

model = tf.keras.models.load_model('models/single_conv_try.h5')

dataset = np.load('data/cifar10.npy')[:100]
blurred = np.load('data/cifar10_blurred_ksize3.npy')[:100]

pred = model.predict(blurred)

plt.imshow(dataset[0])

plt.imshow(blurred[0])

np.max(blurred[0] - pred[0])


plt.imshow(pred[0])
