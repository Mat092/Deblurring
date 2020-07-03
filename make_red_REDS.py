import numpy as np
import cv2
import itertools

sharp_path = 'path tp /train_sharp'
blur_path  = 'path tp /train_blur'
saving_dir = 'path tp /redREDS'

n_clips  = 240
n_frames = 1000

im_h, im_w = (720, 1280)
n_subdiv = 5
patch_h, patch_w = np.array(img.shape[:-1]) // n_subdiv
orig_patches = list(itertools.product(patch_h*np.arange(n_subdiv), patch_w*np.arange(n_subdiv)))

for n_clip in range(n_clips):
    rn_frame = np.random.randint(0, n_frames)

    #reading sharp image
    sh_img = cv2.imread(sharp_path + f'/{n_clip:03}/{rn_frame:08}.png' + frame, 4)
    sh_img = cv2.cvtColor(sh_img, cv2.COLOR_BGR2RGB)

    #reading blur image
    bl_img = cv2.imread(blur_path + f'/{n_clip:03}/{rn_frame:08}.png' + frame, 4)
    bl_img = cv2.cvtColor(bl_img, cv2.COLOR_BGR2RGB)

    for n_patch, (y, x) in enumerate(orig_patches):
        sharp_name = saving_dir + f'/{n_clip:03}_{rn_frame:08}_{n_patch:02}:{n_subdiv*n_subdiv:02}_sharp.png''
        cv2.imwrite(sharp_name, sh_img[y : y + patch_h, x : x + patch_w, :])

        blur_name  = saving_dir + f'/{n_clip:03}_{rn_frame:08}_{n_patch:02}:{n_subdiv*n_subdiv:02}_blur.png''
        cv2.imwrite(sharp_name, bl_img[y : y + patch_h, x : x + patch_w, :])
