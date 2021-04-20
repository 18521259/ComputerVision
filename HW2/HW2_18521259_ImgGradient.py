import cv2
import numpy as np

img_name = 'HW2.jpg'
img = cv2.imread(r'C:\UIT\HK5\introCV_CS231\HW\HW2\HW2.jpg', 0)

#đạo hàm theo trục x (gradien theo hướng x)
dx = img[:,1:] - img[:,:-1]
#đạo hàm theo trục y (gradien theo hướng y)
dy = img[1:,:] - img[:-1,:]
G = dx[1:,:]**2 + dy[:,1:]**2
#độ lớn của gradien
G_img = np.sqrt(dx[1:,:]**2 + dy[:,1:]**2)
G_img = G_img.astype(np.uint8)

cv2.imshow('Original', img)
cv2.imshow('Gradient', G_img)
cv2.imshow('Grad', G) 
print(G_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('Gradient_{}'.format(img_name), G_img)