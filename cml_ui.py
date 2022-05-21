import numpy as np
import argparse
import cv2 as cv
from carver import Engine


def opts_parser():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to source image')
    p.add_argument('-c', '--col', type=int, help='target width to be converted into')
    p.add_argument('-r', '--row', type=int, help='target height to be converted into')
    p.add_argument('--des', default='res.jpg', help='where to store the resized image')
    return p.parse_args()


def main():
    opts = opts_parser()

    target_width = opts.col
    target_height = opts.row
    in_filename = opts.src
    out_filename = opts.des

    img = cv.imread(in_filename)
    assert img is not None, f'Failed to read image {in_filename}'
    if target_width is None or target_height is None:
        print(f"image {in_filename} has shape {img.shape}")
        print('Please specify target height (-r) and width (-c)')
        return

    carving_engine = Engine()
    out = carving_engine.run(img, target_width, target_height)

    if out is not None:
        cv.imwrite(out_filename, out)
    else:
        return


if __name__ == '__main__':
    # import time
    # start = time.time()
    main()
    # print(time.time() - start)
