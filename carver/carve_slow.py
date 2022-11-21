import numpy as np
import cv2 as cv
from typing import Tuple


# maximun possible value with backward energy
_MAX_PIXEL_BACKWARD_ENERGY = 8 * 255 * 2


def _backward_energy(img):
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


def _backward_energy_opencv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grads_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grads_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    grads = np.absolute(grads_x) + np.absolute(grads_y)

    return grads


def _minimum_seam(energy):
    r, c = energy.shape
    M = energy.astype(np.float32)

    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        pre_i = i - 1

        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[pre_i, j:j+3])
                absolute_idx = idx + j

            elif j == c - 1:
                idx = np.argmin(M[pre_i, j-2:j+1])
                absolute_idx = idx + j - 2

            else:
                idx = np.argmin(M[pre_i, j-1:j+2])
                absolute_idx = idx + j - 1

            backtrack[i, j] = absolute_idx
            M[i, j] += M[pre_i, absolute_idx]

    return M, backtrack


def _get_keep(min_idx: int, backtrack: np.ndarray) -> np.ndarray:
    r, _ = backtrack.shape
    keep = np.ones_like(backtrack, dtype=np.bool)
    j = min_idx
    for i in reversed(range(r)):
        keep[i, j] = False
        j = backtrack[i, j]
    return keep


def carve_column(img:np.ndarray, mask_remove:np.ndarray=None, mask_protect:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    _img = img.copy()
    # watch out for the protected area
    _mask_protect = mask_protect.astype(np.float32) if mask_protect is not None else None
    # perform object removal
    _mask_remove = mask_remove.astype(np.float32) if mask_remove is not None else None

    r, c, _ = img.shape
    energy = _backward_energy_opencv(img)
    if _mask_remove is not None:
        energy -= _mask_remove * _MAX_PIXEL_BACKWARD_ENERGY * r
    if _mask_protect is not None:
        energy += _mask_protect * _MAX_PIXEL_BACKWARD_ENERGY * r
    M, backtrack = _minimum_seam(energy)
    min_idx = np.argmin(M[-1])
    keep = _get_keep(min_idx, backtrack)

    keep3c = np.stack([keep] * 3, axis=2)
    _img = _img[keep3c].reshape((r, c - 1, 3))
    if _mask_remove is not None:
        _mask_remove = _mask_remove[keep].reshape((r, c - 1))
    if _mask_protect is not None:
        _mask_protect = _mask_protect[keep].reshape((r, c - 1))

    return _img, _mask_remove, _mask_protect


class Core:
    def __init__(self) -> None:
        pass

    def carve_column(self, img:np.ndarray, mask_remove:np.ndarray=None, mask_protect:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _img, _mask_remove, _mask_protect = carve_column(img, mask_remove, mask_protect)
        return _img, _mask_remove, _mask_protect
