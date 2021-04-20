import cv2
import numpy as np

def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    new_img *= 255
    return new_img
    
img = cv2.imread('test.jpg',0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

print('[Before Scale] laplacian image min-max:', np.min(laplacian), '-', np.max(laplacian))

cv2.imshow('before_scale-laplacian.jpg', laplacian)
cv2.imshow('before_scale-sobelx.jpg', sobelx)
cv2.imshow('before_scale-sobely.jpg', sobely)
cv2.imshow('before_scale-sobelxy.jpg', sobelxy)

laplacian = scale_to_0_255(laplacian)
sobelx =scale_to_0_255(sobelx)
sobely = scale_to_0_255(sobely)
sobelxy = scale_to_0_255(sobelxy)
print('[After Scale] laplacian image min-max:', np.min(laplacian), '-', np.max(laplacian))

cv2.imshow('laplacian.jpg', laplacian)
cv2.imshow('sobelx.jpg', sobelx)
cv2.imshow('sobely.jpg', sobely)
cv2.imshow('sobelxy.jpg', sobelxy)

cv2.waitKey(0)
cv2.destroyAllWindows()