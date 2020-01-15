import numpy as np
import argparse
import cv2 as cv

from carver import carve_column


def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c):
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

    img = cv.imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    
    cv.imwrite(out_filename, out)


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    print(time.time() - start)