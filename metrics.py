'''
Defining custom metrics for model evaluation
Usable from keras function

From :
https://keras.io/api/metrics/#creating-custom-metrics
'''

import tensorflow as tf

import numpy as np

def PSNR(y_true, y_pred):

  '''
  Peak Signal to Noise Ration defined here :

  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  '''

  batch_size = y_pred.shape[0]

  diff = (y_true - y_pred).reshape(batch_size, -1)

  # TODO : all axis? or for one image at a time?
  mse  = np.mean(diff*diff, axis=-1) # axis = -1 -> reshape needed above
  psnr = 10 * np.log10(1. / mse)

  # average psnr in the batch.
  mean_psnr = np.mean(psnr)

  return mean_psnr


def SSIM(y_true, y_pred):

  '''
  Structural SIMiliraty index defined here :

  https://en.wikipedia.org/wiki/Structural_similarity
  '''

  ssim = 1.

  return ssim


if __name__ == '__main__':
  pass
