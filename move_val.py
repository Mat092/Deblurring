import os
import shutil
import random

redsdir = os.path.join(os.getcwd(), 'data', 'redREDS')

sharpdir = os.path.join(redsdir, 'sharp')
blurdir  = os.path.join(redsdir, 'blur')

valsharp  = os.path.join(redsdir, 'val_sharp')
blursharp = os.path.join(redsdir, 'val_blur')

sharp_paths = os.listdir(sharpdir)
blur_paths  = os.listdir(blurdir)

sharp_paths = sorted(sharp_paths)
blur_paths  = sorted(blur_paths)

test_size = 0.1
test_num = int(len(sharp_paths) * test_size)

for i in range(test_num):

  # Same names selection
  sharp_name = random.choice(sharp_paths)
  blur_name  = sharp_name.split('sharp')[0] + 'blur.png'

  sharp_path = os.path.join(sharpdir, sharp_name)
  blur_path  = os.path.join(blurdir,   blur_name)

  new_sharp_path = os.path.join(redsdir, 'val_sharp', sharp_name)
  new_blur_path  = os.path.join(redsdir, 'val_blur',  blur_name)

  os.rename(sharp_path, new_sharp_path)
  os.rename(blur_path,  new_blur_path )

  sharp_paths.remove(sharp_name)
  blur_paths.remove(blur_name)
