'''
Some function to ease preprocessing operation
'''

import os

from tensorflow.keras.datasets import cifar10

import numpy as np
from scipy.ndimage import gaussian_filter
import random

def create_target(dataset, mode='constant', cval=0.):

  y = []

  for i, x in enumerate(dataset) :

    sigma = np.random.uniform(low=0., high=3.)
    blurred = gaussian_filter(input=x, sigma=sigma, mode=mode, cval=cval)

    y.append(blurred)


  return np.asarray(y, dtype='float32')


if __name__ == '__main__':

  pass
