import cv2
import numpy as np
from matplotlib import pyplot as plt


def filter2d(img, kernel, mode = 'convol'):
    padded_arr = np.pad(img, 1, mode = 'constant')
    H_k, W_k = kernel.shape
    H_p, W_p = padded_arr.shape
    output_img = np.zeros(padded_arr.shape)
    #Cross Correlation
    if mode == 'cross':
        kernel = np.flip(kernel)
    for i in range(H_p-2):
        for j in range(W_p-2): 
            output_img[i,j] = np.sum(padded_arr[i:H_k+i, j:W_k+j] * kernel)
    return output_img[:H_p-2, :W_p-2]

kernel = np.ones((3,3), dtype=int)/9
# kernel = np.ones((5,5), dtype=int)/25

img = cv2.imread(r'C:\UIT\HK5\introCV_CS231\HW\HW1\HW1.jpg', 1)
img_bgr = cv2.resize(img, (540, 540))
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

h,s,v = cv2.split(img_hsv)
v1 = filter2d(v, kernel, mode='convol').astype(np.uint8)
result_hsv = cv2.merge((h, s, v1))
result_rgb = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)


# result = filter2d(img_bgr, kernel, mode = 'convol')
plt.subplot(121).set_title('Original')
plt.imshow(img_rgb)
# plt.xticks([]), plt.yticks([])
plt.subplot(122).set_title('Averaging')
plt.imshow(result_rgb)
# plt.xticks([]), plt.yticks([])
plt.show()