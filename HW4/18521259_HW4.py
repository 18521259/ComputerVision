import cv2
import numpy as np

def calc_hist(img_vec):
    hist = np.zeros(256, np.int)
    values, counts = np.unique(img_vec, return_counts=True)
    print(values, "\n", counts)
    for i, val in enumerate(values):
        hist[val] = counts[i]
    return hist

def hist_equal_gray(_img):
    hist_img = _img.copy()
    values, counts = np.unique(_img, return_counts=True)
    print(values)
    print(counts)
    cdf = np.zeros(len(counts), np.int)
    for i in range(len(cdf)):
        cdf[i] = np.sum(counts[:i]) + counts[i]
    hv = np.zeros(len(values), np.int)
    for i in range(len(hv)):
        hv[i] = round( ((cdf[i] - np.min(cdf))/(np.max(cdf) - np.min(cdf))*255) )
    print(hv)
    transform = dict(zip(values, hv))
    for i in range(len(hist_img)):
        for j in range(len(hist_img[0])):
            if hist_img[i][j] in transform.keys():
                hist_img[i][j] = transform.get(hist_img[i][j])

    return hist_img


img_bgr = cv2.imread(r'C:\UIT\HK6\HW-20210420T132435Z-001\HW\HW4\test3.jpg', 1)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(img_hsv)
v1 = hist_equal_gray(v)
result_hsv = cv2.merge([h, s, v1])
result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

# hist = calc_hist(img_gray)
# print(hist)

cv2.imshow('original', img_bgr)
cv2.imshow('img', result_bgr)
# cv2.imwrite('result.png', result_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()