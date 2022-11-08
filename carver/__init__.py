import numpy as np
import cv2 as cv


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


class Engine():
    def __init__(self):
        self.MAX_WIDTH = 5000
        self.MAX_HEIGHT = 5000
        self.core = Core()

    def _crop_c(self, img:np.ndarray, num:int) -> np.ndarray:
        r, c, _ = img.shape

        for i in range(num):
            img = self.core.carve_column(img)

        return img

    def _crop_r(self, img:np.ndarray, num:int) -> np.ndarray:
        img = np.rot90(img, 1, (0, 1))
        img = self._crop_c(img, num)
        img = np.rot90(img, 3, (0, 1))
        return img

    def run(self, img:np.ndarray, target_width:int, target_height:int) -> np.ndarray:
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

        out = self._crop_r(img_scaled, rows_tobe_carved)
        out = self._crop_c(out, cols_tobe_carved)

        return out
