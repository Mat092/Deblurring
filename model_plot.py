import tensorflow as tf
from metrics import PSNR, SSIM, MIX, SSIM_loss, SSIM_PSNR

model_name = 'models/unetlike_mse_cifar.h5'

objects = {'PSNR' : PSNR, 'SSIM' : SSIM, 'MIX' : MIX, 'SSIM_loss' : SSIM_loss, 'SSIM_PSNR' : SSIM_PSNR}
model = tf.keras.models.load_model(model_name, custom_objects=objects)

model.summary()

tf.keras.utils.plot_model(model, "images/U-like_model_CIFAR.png", show_shapes=True)
