import numpy as np
import argparse
from cv2 import resize as imresize
from imageio import imread, imwrite


# try:
#     from carver.carve_fast import carve_column
# except ModuleNotFoundError:
#     from carver.carve_slow import carve_column
#     print('#@@@@@@@@@@@@@@@@@@@#')
#     print('Using slow version seam carving')
#     print('#@@@@@@@@@@@@@@@@@@@#')

from carver.carve_numba import carve_column




class Engine():
    def __init__(self):
        self.MAX_WIDTH = 5000
        self.MAX_HEIGHT = 5000

    def _crop_c(self, img, num):
        r, c, _ = img.shape

        for i in range(num):
            img = carve_column(img)

        return img

    def _crop_r(self, img, num):
        img = np.rot90(img, 1, (0, 1))
        img = self._crop_c(img, num)
        img = np.rot90(img, 3, (0, 1))
        return img


    def run(self, img, target_width, target_height):
        if target_width > self.MAX_WIDTH or target_height > self.MAX_HEIGHT:
            print(f'target height and width must be less than {self.MAX_HEIGHT} and {self.MAX_WIDTH} respectively')
            return None

        img_h, img_w, _ = img.shape
        scale_h = target_height / img_h
        scale_w = target_width / img_w
        scale = max(scale_h, scale_w)

        img_scaled = imresize(img, (0,0), fx=scale, fy=scale)
        rows_tobe_carved = img_scaled.shape[0] - target_height
        cols_tobe_carved = img_scaled.shape[1] - target_width

        out = self._crop_r(img_scaled, rows_tobe_carved)
        out = self._crop_c(out, cols_tobe_carved)

        return out





def opts_parser():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to source image')
    p.add_argument('width', type=int, help='target width to be converted into')
    p.add_argument('height', type=int, help='target height to be converted into')
    p.add_argument('--des', default='res.jpg', help='where to store the resized image')
    return p.parse_args()



def main():
    opts = opts_parser()

    target_width = opts.width
    target_height = opts.height
    in_filename = opts.src
    out_filename = opts.des

    carving_engine = Engine()

    img = imread(in_filename)

    out = carving_engine.run(img, target_width, target_height)
    
    if out is not None:
        imwrite(out_filename, out)
    else:
        return


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    print(time.time() - start)