import numpy as np
import cv2 as cv


def backward_energy(img):
    from scipy.ndimage.filters import convolve
    filter_du = np.array([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ])

    filter_dv = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ])

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype('float32')
    grads = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    return grads


def backward_energy_opencv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grads_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grads_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    grads = np.absolute(grads_x) + np.absolute(grads_y)

    return grads


def minimum_seam(img):
    r, c, _ = img.shape
    M = backward_energy_opencv(img)

    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        pre_i = i - 1

        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[pre_i, j:j+2])
                absolute_idx = idx + j

            elif j == c - 1:
                idx = np.argmin(M[pre_i, j-1:j+1])
                absolute_idx = idx + j - 1

            else:
                idx = np.argmin(M[i - 1, j-1:j+2])
                absolute_idx = idx + j - 1

            backtrack[i, j] = absolute_idx
            M[i, j] += M[pre_i, absolute_idx]

    return M, backtrack


def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img
