from tensorflow.keras.callbacks import Callback

import pandas as pd

class CustomCB(Callback):

  '''
  This object can be passed to different keras function and can
  store a lot of different informations, usefull for posterior data analysis

  Empty for now though.

  List of possibility at :
  https://www.tensorflow.org/guide/keras/custom_callback
  '''
  def __init__ (self, datafile):
    self.data     = pd.DataFrame()
    self.datafile = datafile

  def on_batch_begin(self, batch, logs={}):
    pass

  def on_batch_end(self, batch, logs={}):
    pass

  def on_epoch_begin(self, epoch, logs={}):
    pass

  def on_epoch_end(self, epoch, logs={}):
    '''
    At the end of each epoch save the history data.
    '''

    print('Sono in epoch end, ', logs)

    self.data = self.data.append(logs, ignore_index=True)
    self.data.to_csv(self.datafile, header=True, float_format='%g', index=False)
    print('Saved History at {}'.format(self.datafile))


if __name__ == '__main__':
  pass
