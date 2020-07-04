import os
import imageio
import math

import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

  def __init__ (self, x_set, y_set, batch_size):

    self.x_set = x_set
    self.y_set = y_set

    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x_set) / self.batch_size)

  def __getitem__(self, idx):

    batch_x = self.x_set[idx * self.batch_size : (idx + 1) * self.batch_size]
    batch_y = self.y_set[idx * self.batch_size : (idx + 1) * self.batch_size]

    x_images = np.asarray([imageio.imread(path) for path in batch_x], dtype='float32') / 255.
    y_images = np.asarray([imageio.imread(path) for path in batch_y], dtype='float32') / 255.

    return (x_images, y_images)

  def on_epoch_end(self, ):

    pass # insert shuffling


if __name__ == '__main__':
  pass
