import sys
import argparse

from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
import cv2 as cv



def e1(img):
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
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    return convolved

def e1_opencv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grads_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grads_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    grads = np.absolute(grads_x) + np.absolute(grads_y)

    return grads


def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = e1_opencv(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
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






def opts_parser():
    p = argparse.ArgumentParser()
    p.add_argument('axis', choices=['r', 'c'], help='resize in rows(r) or columns(c)')
    p.add_argument('scale', type=float, help='resize scale')
    p.add_argument('src', help='path to source image')
    p.add_argument('--des', default='res.jpg', help='where to store the resized image')
    return p.parse_args()

def main():
    opts = opts_parser()

    which_axis = opts.axis
    scale = opts.scale
    in_filename = opts.src
    out_filename = opts.des

    img = imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    
    imwrite(out_filename, out)






if __name__ == '__main__':
    # main()

    import time
    img = cv.imread('star.jpg')

    start = time.time()
    for i in range(100):
        tmp = e1_opencv(img)
    print(f"{time.time() - start}")