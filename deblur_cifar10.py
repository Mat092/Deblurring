'''
First example of deblurring using simple models.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import SSIM, PSNR, SSIM_loss, SSIM_multiscale_loss, MIX
import preprocessing as pr
from callbacks import CustomCB

# Constants
batch_size = 100
epochs     = 100

size = 30000

model_name = 'doubleConv_mix'
save_path  = os.path.join(os.getcwd(), 'models', model_name + '.h5')

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

# TODO : Maybe use different activation Function to help training? (Relu, squashing function ...)
inp   = Input(shape=(32, 32, 3))
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=8, padding='same', activation='relu')(inp)
x     = Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=3, padding='same', activation='sigmoid')(x)
model = Model(inp, x)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, # keras standard params
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-7
                               )

metrics = ['mean_squared_error', 'mean_absolute_error', PSNR, SSIM, MIX]

model.compile(optimizer=opt, loss=MIX, loss_weights=None, metrics=metrics)

saveback = ModelCheckpoint(filepath=save_path,
                           monitor='val_loss',
                           save_best_only=True,
                           save_weight_only=False,
                           save_freq='epoch',
                           )

datafile   = os.path.join(os.getcwd(), 'data', 'hist_{}.csv'.format(model_name))
customback = CustomCB(datafile)

history = model.fit(x=x_train, y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    shuffle=True,
                    callbacks=[saveback, customback]
                    )

# Already saved in Custom Callbacks
# df = pd.DataFrame({name : history.history[name] for name in history.history})
#
# df.to_csv(datafile, header=True, float_format='%g', index=False)
