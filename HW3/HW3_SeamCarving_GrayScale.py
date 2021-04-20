from tqdm import trange
import numpy as np
import numba
import cv2
from scipy.ndimage.filters import convolve

def calc_gradient(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

    img = img.astype('float32')
    gradient = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    return gradient

def crop_c(img, scale_c):
    r, c = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

@numba.jit
def carve_column(img):
    r, c = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    img = img[mask].reshape((r, c - 1))
    return img

@numba.jit
def minimum_seam(img):
    r, c = img.shape
    gradient = calc_gradient(img)

    M = gradient.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


"""main()"""
# Choose edge to be carved: <r/c> (row/column)
axis = 'r'
# 0 < scale < 1
scale = 0.5

in_filename = 'test.jpg'
out_filename = 'result2.jpg'

img = cv2.imread(in_filename, 0)

if axis == 'r':
    out = crop_r(img, scale)
elif axis == 'c':
    out = crop_c(img, scale)

cv2.imshow('result', out)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(out_filename, out)