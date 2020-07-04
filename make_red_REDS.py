import os

import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt

path = '/media/mattia/70EA383EEA3802C0/Users/Mattia/REDS/'

sharp_path = path + 'train_sharp/train/train_sharp/'
blur_path  = path + 'train_blur/train/train_blur/'
saving_dir = os.path.join(os.getcwd(), 'data', 'redREDS/')

img = cv2.imread(sharp_path + '{:03}/{:08}.png'.format(0, 0))
img_blur = cv2.imread(blur_path + '{:03}/{:08}.png'.format(0, 0))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imwrite(saving_dir + '/prova.png', img)

n_clips  = 240
n_frames = 100
n_subdiv = 5

im_h, im_w, im_c = img.shape

# Numbers of patches in h and 2 dimension
patch_h, patch_w = np.array((im_h, im_w)) // n_subdiv

# Starting point of every patches in the image
orig_patches = list(itertools.product(patch_h * np.arange(n_subdiv), patch_w * np.arange(n_subdiv)))

for n_clip in range(n_clips):

  # Select a random Frame from every clip
  rn_frame = np.random.randint(0, n_frames)

  # reading sharp image
  sh_img = cv2.imread(sharp_path + f'{n_clip:03}/{rn_frame:08}.png', 4)

  # reading blur image
  bl_img = cv2.imread(blur_path  + f'{n_clip:03}/{rn_frame:08}.png', 4)

  for n_patch, (y, x) in enumerate(orig_patches):

    # Divide images by patches, and save.
    sharp_name = saving_dir + f'sharp/{n_clip:03}_{rn_frame:03}_{n_patch:02}:{n_subdiv * n_subdiv:02}_sharp.png'
    cv2.imwrite(sharp_name, sh_img[y : y + patch_h, x : x + patch_w, :])

    blur_name  = saving_dir + f'blur/{n_clip:03}_{rn_frame:03}_{n_patch:02}:{n_subdiv * n_subdiv:02}_blur.png'
    cv2.imwrite(blur_name, bl_img[y : y + patch_h, x : x + patch_w, :])
