'''
Defining custom metrics for model evaluation
Usable from keras function

From :
https://keras.io/api/metrics/#creating-custom-metrics
'''

import tensorflow as tf

import numpy as np

def mean_PSNR(y_true, y_pred):

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


def PSNR(y_true, y_pred):
  '''
  Peak Signal to Noise Ration defined here :

  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

  The higher, the better. range [0, inf)
  '''
  return tf.image.psnr(y_true, y_pred, max_val=1.)


def SSIM_loss(y_true, y_pred):

  '''
  Structural SIMiliraty index defined here :

  https://en.wikipedia.org/wiki/Structural_similarity

  The lower, the better. range [0, 1]
  '''
  return 1 - tf.image.ssim(y_true, y_pred, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


def SSIM_multiscale_loss(y_true, y_pred):
  '''
  Multiscale SSIM Reference in :

  arxiv 1511.08861v3 - "Loss Function for Image Restoration with Neural Networks, H. Zhao et al."
  '''
  return 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


def SSIM(y_true, y_pred):
  '''
  Structural SIMiliraty index defined here :

  https://en.wikipedia.org/wiki/Structural_similarity

  The higher, the better. range [0, 1]
  '''
  return tf.image.ssim(y_true, y_pred, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


def SSIM_multiscale(y_true, y_pred):
  '''
  Multiscale SSIM Reference in :

  arxiv 1511.08861v3 - "Loss Function for Image Restoration with Neural Networks, H. Zhao et al."
  '''
  return tf.image.ssim_multiscale( y_true, y_pred, max_val=1., filter_size=5, filter_sigma=1.5, k1=0.01, k2=0.03)


def MIX(y_true, y_pred):
  '''
  linear combination of multiscale sim and l1, defined :

  arxiv 1511.08861v3 - "Loss Function for Image Restoration with Neural Networks, H. Zhao et al."
  '''

  alpha = 0.84

  # TODO: Missing a term I don't fully understand
  return alpha * SSIM_loss(y_true, y_pred) + (1. - alpha) * tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred), axis=[1, 2, 3])

  # tf.keras.losses.MAE(y_true, y_pred)


if __name__ == '__main__':
  pass
