import numpy as np
import argparse
import cv2 as cv
from carver import Engine


def opts_parser():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='Path to source image')
    p.add_argument('-c', '--col', type=int, help='Target width')
    p.add_argument('-r', '--row', type=int, help='Target height')
    p.add_argument('--des', default='res.jpg', help='Where to store the resized image. Default to res.jpg')
    p.add_argument('--protect', help="Path to the protected mask image file; Providing a protect mask will improve the result since seam carving guarantee not carving anything in the proteced area but it's optional")
    p.add_argument('--remove', help='Path to the removal mask image file; Seam carving will perform object removal and remove all the masked area if this flag is specify')
    return p.parse_args()


def read_image(path: str) -> np.ndarray:
    img = cv.imread(path)
    assert img is not None, f'Failed to read image {path}'
    return img


def main():
    opts = opts_parser()

    target_width = opts.col
    target_height = opts.row
    in_filename = opts.src
    out_filename = opts.des
    protect = opts.protect
    remove = opts.remove

    img = read_image(in_filename)
    if (target_width is None or target_height is None) and remove is None:
        print(f"image {in_filename} has shape {img.shape}")
        print('Please specify target height (-r) and width (-c)')
        return

    mask_protect = None
    mask_remove = None
    if protect is not None:
        mask_protect = read_image(protect)
        assert mask_protect.shape == img.shape, f"image sizes do not match, src {img.shape}, protect {mask_protect.shape}"
    if remove is not None:
        mask_remove = read_image(remove)
        assert mask_remove.shape == img.shape, f"image sizes do not match, src {img.shape}, remove {mask_remove.shape}"

    carving_engine = Engine()
    if remove is not None:
        out = carving_engine.remove(img, mask_remove, mask_protect)
    else:
        out = carving_engine.resize(img, target_width, target_height, mask_protect)

    if out is not None:
        cv.imwrite(out_filename, out)
    else:
        return


if __name__ == '__main__':
    main()
