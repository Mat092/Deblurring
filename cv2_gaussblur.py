import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Plotting different gaussian smoothing of the same image, with different kernel size.
'''

img = cv2.imread('lenna.png', 4)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

n_kernel = 5
k_sizes = np.arange(2,2+n_kernel)*2 +1

plt.figure(figsize=(6*(n_kernel+1),8))

#Original
plt.subplot(1, n_kernel+1, 1),plt.imshow(img),plt.title('Original', fontsize = 20)
plt.xticks([]), plt.yticks([])

#Different kernels
for i,k in enumerate(k_sizes):
  blur = cv2.blur(img,(k,k))
  plt.subplot(1, n_kernel+1, 2+i),plt.imshow(blur),plt.title(f'Kernel ({k},{k})', fontsize = 20)
  plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig('lenna_gauss_kernel.png', dpi=300, bbox_inches='tight' )
plt.show()
