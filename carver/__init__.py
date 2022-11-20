import numpy as np
import cv2 as cv
from typing import List


USE_GPU = True
try:
    from carver.carve_cuda import Core
except ModuleNotFoundError as e:
    print('#@@@@@@@@@@@@@@@@@@@#')
    print(e)
    print('Failed to import carve_cuda.py, using CPU instead')
    print('#@@@@@@@@@@@@@@@@@@@#')
    USE_GPU = False
if not USE_GPU:
        from carver.carve_slow import Core


def prepare_mask(mask: np.ndarray, scale: float = None) -> np.ndarray:
    '''
    mask: shape=(height, width, 3), dtype=np.uint8

    return: shape=(height, width), dtype=np.uint8
    '''
    _mask = mask[:, :, 0]
    if scale is not None:
        _mask = cv.resize(_mask, (0,0), fx=scale, fy=scale)
    _mask = np.greater(_mask, 0).astype(np.uint8)
    return _mask


class Engine():
    def __init__(self):
        self.MAX_WIDTH = 5000
        self.MAX_HEIGHT = 5000
        self.core = Core()

    def _crop_c(self, img:np.ndarray, num:int, mask_remove:np.ndarray=None, mask_protect:np.ndarray=None) -> List[np.ndarray]:
        r, c, _ = img.shape

        for i in range(num):
            img, mask_remove, mask_protect, = self.core.carve_column(img, mask_remove, mask_protect)

        return img

    def _crop_r(self, img:np.ndarray, num:int, mask_remove:np.ndarray=None, mask_protect:np.ndarray=None) -> List[np.ndarray]:
        img = np.rot90(img, 1, (0, 1))
        if mask_remove is not None:
            mask_remove = np.rot90(mask_remove, 1, (0, 1))
        if mask_protect is not None:
            mask_protect = np.rot90(mask_protect, 1, (0, 1))

        img = self._crop_c(img, num, mask_remove, mask_protect)

        img = np.rot90(img, 3, (0, 1))
        if mask_remove is not None:
            mask_remove = np.rot90(mask_remove, 3, (0, 1))
        if mask_protect is not None:
            mask_protect = np.rot90(mask_protect, 3, (0, 1))
        return img

    def remove(self, img:np.ndarray, mask_remove:np.ndarray, mask_protect:np.ndarray=None) -> np.ndarray:

        mask_remove = prepare_mask(mask_remove)
        if mask_protect is not None:
            mask_protect = prepare_mask(mask_protect)

        # count = 0
        if mask_protect is None:
            while np.any(mask_remove > 0):
                img, mask_remove, mask_protect = self.core.carve_column(img, mask_remove, mask_protect)
                # count += 1
                # print(count)
        else:
            curr_mask_sum = mask_remove.sum()
            past_mask_sum = np.inf
            while curr_mask_sum < past_mask_sum:
                img, mask_remove, mask_protect = self.core.carve_column(img, mask_remove, mask_protect)
                past_mask_sum, curr_mask_sum = curr_mask_sum, mask_remove.sum()
                # count += 1
                # print(count)

        return img

    def resize(self, img:np.ndarray, target_width:int, target_height:int, mask_protect:np.ndarray=None) -> np.ndarray:
        if target_width > self.MAX_WIDTH or target_height > self.MAX_HEIGHT:
            print(f'target height and width must be less than {self.MAX_HEIGHT} and {self.MAX_WIDTH} respectively')
            return None

        img_h, img_w, _ = img.shape
        scale_h = target_height / img_h
        scale_w = target_width / img_w
        scale = max(scale_h, scale_w)

        img_scaled = cv.resize(img, (0,0), fx=scale, fy=scale)
        rows_tobe_carved = img_scaled.shape[0] - target_height
        cols_tobe_carved = img_scaled.shape[1] - target_width

        if mask_protect is not None:
            mask_protect = prepare_mask(mask_protect)

        out = self._crop_r(img_scaled, rows_tobe_carved, None, mask_protect)
        out = self._crop_c(out, cols_tobe_carved, None, mask_protect)

        return out
