'''
First example of deblurring using simple models.
'''

import os

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pr
from metrics import PSNR
from callbacks import CustomCB

# Constants
batch_size = 100
epochs     = 100

size = 10000

save_dir    = os.path.join(os.getcwd(), 'models')
model_name  = 'double_conv_8filters_PSNR'
model_fname = save_dir + f'/{model_name}.h5'

# dataset preprocessing TODO : Save the two dataset for faster loading time?

# dataset = pr.cifar_download_and_scale()
# blurred = pr.blur_input(dataset, k_size=3)
#
# np.save('data/cifar10' , dataset)
# np.save('data/cifar10_blurred_ksize3', blurred)

dataset = np.load('data/cifar10.npy')[:size]
blurred = np.load('data/cifar10_blurred_ksize3.npy')[:size]

x_train, x_test, y_train, y_test = pr.train_test(blurred, dataset)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

inp   = Input(shape=(32, 32, 3))
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=8, padding='same')(inp)
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=3, padding='same')(x)
model = Model(inp, x)

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001, # keras standard params
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-7
                               )

metrics = ['mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy']

model.compile(optimizer=opt, loss=PSNR, loss_weights=[-1,], metrics=metrics)

saveback = ModelCheckpoint(filepath=model_fname,
                           monitor='val_loss',
                           save_best_only=True,
                           save_weight_only=False,
                           save_freq='epoch',
                           )

customback = CustomCB()

history = model.fit(x=x_train, y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    shuffle=True,
                    callbacks=[saveback], # customback], TODO : custom callback
                    )

names = ['loss', 'val_loss'] + metrics
df = pd.DataFrame({name : history.history[name] for name in names})

df.to_csv(f'data/history_{model_name}.csv', header=True, float_format='%g', index=False)
