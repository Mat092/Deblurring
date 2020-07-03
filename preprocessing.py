'''
Some function to ease preprocessing operation
'''

import os

from tensorflow.keras.datasets import cifar10

from sklearn.model_selection import train_test_split as tts

import numpy as np

from matplotlib import pyplot as plt

# TODO : need to comment that for a second sorry (conflict with tensorflow-gpu in miniconda)
import cv2

def blur_input(dataset, sigma=None):

  # TODO : Should do a better control on how sigma values are passed
  if sigma is None:
    sigma = np.random.uniform(low=0., high=3., size=len(dataset))

  # If sigmaY is 0, the value from sigmaX is used. TODO : border Type?
  # TODO : It is possible to do this with multiprocessing for speed up, but maybe not necessary.
  y = [cv2.GaussianBlur(img, ksize=(0,0), sigmaX=s, sigmaY=0., borderType=cv2.BORDER_DEFAULT)
       for img, s in zip(dataset, sigma) ]

  return np.asarray(y, dtype='float32')


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


def transform(data):
  '''
  fourier transform of ta batch of images in dataset and blurred
  '''
  return np.fft.fftn(data, s=None, axes=(1, 2, 3), norm=None)


def anti_tranform(data):
  '''
  Compute the anti-fourier transform of a batch of images
  '''
  return np.fft.ifftn(data, s=None, axes=(1, 2, 3), norm=None)


if __name__ == '__main__':

  # Blur and save the dataset.
  datafile = os.path.join(os.getcwd(), 'data', 'cifar10.npy')
  outfile  = os.path.join(os.getcwd(), 'data', 'cifar10_blur_sigma0-3.npy')

  dataset  = np.load(datafile)

  # Blur and save
  # blurred  = blur_input(dataset)
  # np.save(outfile, blurred)

  blurred = np.load(outfile)

  def show_image(inpt1, inpt2):

    img_num = np.random.randint(low=0, high=len(dataset-1))

    orig = inpt1[img_num]
    blur = inpt2[img_num]

    # Small Visual Test for blurring

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    ax1.imshow(orig)
    ax1.set_title('Original Image')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(blur)
    ax2.set_title('Blurred Image')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show();

  # show_image(dataset, blurred)

  # Some Fourier transformation.

  outfile_ft  = os.path.join(os.getcwd(), 'data', 'cifar10_FT.npy')
  blurfile_ft = os.path.join(os.getcwd(), 'data', 'cifar10_blur_sigma0-3_FT.npy')

  ft = transform(dataset[:20000])

  ft_shift = np.log(np.abs(ft))

  # shift = np.fft.fftshift(ft)

  plt.imshow(ft_shift[0] / ft_shift[0].max())

  data = np.real(anti_tranform(ft))

  plt.imshow(data[0])

  np.abs(data - dataset[:100]).max()

  ft_blur = transform(blurred[:])

  np.save(outfile_ft,  ft_shift)
  np.save(blurfile_ft, ft_blur)

#%%
