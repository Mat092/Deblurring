'''
Some function to ease preprocessing operation
'''

import os

from tensorflow.keras.datasets import cifar10

from sklearn.model_selection import train_test_split as tts

import numpy as np

# TODO : need to comment that for a second sorry (conflict with tensorflow gpu)
# import cv2
#
# def blur_input(dataset, k_size=3, sigma=None):
#
#   # kernel size should be odd for cv2.
#   k = k_size if k_size % 2 else k_size-1
#
#   # TODO : Should do a better control on how sigma values are passed
#   if sigma is None:
#     sigma = np.random.uniform(low=0., high=3., size=len(dataset))
#
#   # If sigmaY is 0, the value from sigmaX is used. TODO : border Type?
#   # TODO : It is possible to do this with multiprocessing for speed up, but maybe not necessary.
#   y = [cv2.GaussianBlur(img, ksize=(k,k), sigmaX=s, sigmaY=0., borderType=cv2.BORDER_DEFAULT)
#        for img, s in zip(dataset, sigma) ]
#
#   return np.asarray(y, dtype='float32')


def cifar_download_and_scale():
  '''
  Download and scale the CIFAR_10 dataset from tensorflow.
  '''

  # Input image dimensions
  img_rows, img_cols, channels = 32, 32, 3
  inputs  = img_rows * img_cols * channels
  outputs = img_rows * img_cols * channels

  # The data, split between train and test sets:
  (x_train, _), (x_test, _) = cifar10.load_data()

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
  x_test  = x_test.reshape( x_test.shape[0] , img_rows, img_cols, channels)

  # normalize to [0., 1.] range
  x_train = x_train.astype('float32') / 255.
  x_test  = x_test.astype('float32')  / 255.

  # Concatenate, we don't need the split yet
  dataset = np.concatenate([x_train, x_test], axis=0)

  return dataset


def train_test(input, target, test_size=0.1):
  '''
  Simple wrap of train_test_split from scikit.
  '''

  x_train, x_test, y_train, y_test = tts(input, target,
                                        test_size=test_size,
                                        train_size=None,
                                        random_state=None,
                                        shuffle=True,
                                        stratify=None
                                        )

  return x_train, x_test, y_train, y_test


if __name__ == '__main__':

  # Small Visual Test for blurring a random image

  from matplotlib import pyplot as plt

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

  img = cv2.imread('images/lenna.png', 4)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  input = blur_input([img]*10, k_size=20)

  ax1.imshow(img/255)
  ax1.set_xticks([])
  ax1.set_yticks([])

  ax2.imshow(input[0]/255)
  ax2.set_xticks([])
  ax2.set_yticks([])

  plt.show();
