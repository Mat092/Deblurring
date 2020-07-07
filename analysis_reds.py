import imageio
import os

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

def image_name(n_clip, frame, sharp):

  blur_path = '/media/mattia/70EA383EEA3802C0/Users/Mattia/REDS/train_blur/train/train_blur'
  sharp_path = blur_path.replace('blur', 'sharp')

  if sharp :
    return os.path.join(sharp_path, f'{n_clip:03}', f'{frame:08}.png')
  else :
    return os.path.join(blur_path, f'{n_clip:03}', f'{frame:08}.png')

def change_model(model):

  x = Input((None, None, 3))
  y = model(x)
  model = Model(x, y)

  return model

if __name__ == '__main__':

  n_clips = 2 # max = 240

  cwd = os.getcwd()

  # The images are stored in Windows in my PC
  blur_path = '/media/mattia/70EA383EEA3802C0/Users/Mattia/REDS/train_blur/train/train_blur'
  sharp_path = blur_path.replace('blur', 'sharp')

  # Loading Test model
  model_name = 'conv4_32filt_reds.h5'
  model_path = os.path.join(cwd, 'models', model_name)

  objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
  model   = load_model(model_path, custom_objects=objects)
  model   = change_model(model)

  model.summary()

  frames = np.random.randint(low=0, high=100, size=(n_clips,))

  blur = np.asarray([imageio.imread(image_name(n, fr, sharp=False)) for n,fr in enumerate(frames)], dtype='float32') / 255.
  orig = np.asarray([imageio.imread(image_name(n, fr, sharp=True )) for n,fr in enumerate(frames)], dtype='float32') / 255.

  pred = model.predict(x=blur)

  plt.imshow(pred[0])

  psnr_pred_orig = PSNR(tf.Variable(orig), tf.Variable(pred)).numpy()
  psnr_blur_orig = PSNR(tf.Variable(orig), tf.Variable(blur)).numpy()

  ssim_pred_orig = SSIM(tf.Variable(orig), tf.Variable(pred)).numpy()
  ssim_blur_orig = SSIM(tf.Variable(orig), tf.Variable(blur)).numpy()

  psnr_pred_orig.mean()
  psnr_blur_orig.mean()
