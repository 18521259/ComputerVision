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
    filter_du = np.stack([filter_du] * 3, axis = -1)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis = -1)

    img = img.astype(np.float32)
    convolved = np.abs(convolve(img, filter_du)) + np.abs(convolve(img, filter_dv))

    gradient = convolved.sum(axis = -1)

    return gradient

def crop_c(img, scale_c):
    r, c, _ = img.shape
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
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=-1)
    img = img[mask].reshape((r, c - 1, 3))
    return img

@numba.jit
def minimum_seam(img):
    r, c, _ = img.shape
    gradient = calc_gradient(img)

    M = gradient.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

"""main()"""
# Choose edge to be carved: <r/c> (row/column)
axis = 'c'
# 0 < scale < 1
scale = 0.5

in_filename = 'sample.jpg'
out_filename = 'result.jpg'

img = cv2.imread(in_filename)

if axis == 'r':
    out = crop_r(img, scale)
elif axis == 'c':
    out = crop_c(img, scale)

cv2.imshow(out_filename, out)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(out_filename, out)