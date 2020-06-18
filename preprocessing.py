'''
Some function to ease preprocessing operation
'''

import os
import cv2
import numpy as np

def blur_input(dataset, k_size = 4, sigma=None):
  k = k_size if k_size % 2 else k_size-1

  #Should do a better control on how sigma values are passed
  if sigma is None:
    sigma = np.random.uniform(low=0., high=3., size=len(dataset))

  y = [cv2.GaussianBlur(img,ksize=(k,k),sigmaX=0)
       for img, s in zip(dataset, sigma) ]

  return np.asarray(y, dtype='float32')


if __name__ == '__main__':
  from matplotlib import pyplot as plt

  img = cv2.imread('lenna.png', 4)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  input = blur_input([img]*7, k_size = 20)
  plt.imshow(input[0]/255)
