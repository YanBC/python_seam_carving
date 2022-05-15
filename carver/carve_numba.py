import numpy as np
import cv2 as cv
from numba import jit


@jit
def forward_energy(im):
    h, w = im.shape[:2]
    im = cv.cvtColor(im.astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return energy


def backward_energy(im):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    grads_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grads_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    grads = np.absolute(grads_x) + np.absolute(grads_y)

    return grads


@jit
def get_minimum_seam(im):
    h, w = im.shape[:2]
    M = forward_energy(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for row in range(1, h):
        for col in range(0, w):
            if col == 0:
                idx = np.argmin(M[row-1, col:col+3])
                absolute_idx = idx + col

            elif col == w - 1:
                idx = np.argmin(M[row-1, col-2:col+1])
                absolute_idx = idx + col - 2

            else:
                idx = np.argmin(M[row-1, col-1:col+2])
                absolute_idx = idx + col - 1
            backtrack[row, col] = absolute_idx
            M[row, col] += M[row-1, absolute_idx]

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    col = np.argmin(M[-1])
    for row in range(h-1, -1, -1):
        boolmask[row, col] = False
        seam_idx.append(col)
        col = backtrack[row, col]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


def carve_column(im):
    seam_idx, boolmask = get_minimum_seam(im)

    im = remove_seam(im, boolmask)

    return im
