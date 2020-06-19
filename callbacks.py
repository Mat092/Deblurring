from tensorflow.keras.callbacks import Callback

class CustomCB(Callback):

  '''
  This object can be passed to different keras function and can
  store a lot of different informations, usefull for posterior data analysis

  Empty for now though.

  List of possibility at :
  https://www.tensorflow.org/guide/keras/custom_callback
  '''

  def on_batch_begin(self, batch, logs={}):
    pass

  def on_batch_end(self, batch, logs={}):
    pass

  def on_epoch_begin(self, epoch, logs={}):
    pass

  def on_epoch_end(self, epoch, logs={}):
    pass

  # ...

if __name__ == '__main__':
  pass
