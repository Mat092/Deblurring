from __future__ import print_function
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
import os
import numpy as np
import preprocessing as pr
'''
Trying a simple CNN model on a gaussian blurred CIFAR10.
ETA: 100 epochs * 30s = 50min
'''
#%%-----------------------------------------------------------------------------
batch_size = 100
epochs     = 100

# Input image dimensions
img_rows, img_cols, channels = 32, 32, 3
input_shape = (img_rows, img_cols, channels)
inputs  = img_rows * img_cols * channels
outputs = img_rows * img_cols * channels

save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'single_conv.h5'

# The data, split between train and test sets:
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.

gssk = 5

y_train = x_train
x_train = pr.blur_input(x_train, k_size=gssk )

y_test = x_test
x_test = pr.blur_input(x_test, k_size=gssk )

model = Sequential()
model.add(Conv2D(3, kernel_size=(gssk, gssk),
                 activation='relu',
                 padding='same',
                 input_shape=input_shape))

model.compile(loss='mean_squared_error',
              optimizer=Adadelta(),
              metrics=['mean_absolute_error', 'binary_crossentropy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.save_weights(model_name)

score = model.evaluate(x_test, y_test, verbose=0)
if len(score) == 3:
  print('Test loss (MSE):', score[0])
  print('Test MAE:', score[1])
  print('Test BCE:', score[2])
else: print(score)
