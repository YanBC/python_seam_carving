import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from typing import Tuple
import cv2 as cv
import os


#######################
# Meta Data
#######################
cuda.init()
gpu_id = 0
block_x = 32
block_y = 32
data_type = np.float32
data_size = np.ones(1, dtype=data_type).nbytes
current_dir = os.path.dirname(__file__)
cuda_codes = os.path.join(current_dir, "carve.cu")


#######################
# Functions
#######################
# unused, for documentation only
def minimum_seam(image:np.ndarray, width:int, height:int) -> Tuple[np.ndarray, int]:
    global block_x, block_y, data_type, data_size, gpu_id

    cuda_ctx = cuda.Device(gpu_id).make_context()
    with open(cuda_codes) as f:
        _codes = f.read()
    mod = SourceModule(_codes)
    try:
        # cuda stream and funtions
        stream = cuda.Stream()
        bgr2gray = mod.get_function("bgr2gray")
        sobel_abs = mod.get_function("sobel_abs")
        min_energy_at_row = mod.get_function("min_energy_at_row")
        get_min_index = mod.get_function("get_min_index")

        # host data
        image_host = image.astype(data_type)
        backtrack_host = cuda.pagelocked_zeros(shape=(height, width), dtype=np.int32)
        min_index_host = cuda.pagelocked_zeros(shape=(1,), dtype=np.int32)

        # device data
        image_device = cuda.mem_alloc(image_host.nbytes)
        gray_image_device = cuda.mem_alloc(width * height * data_size)
        energy_device = cuda.mem_alloc(width * height * data_size)
        backtrack_device = cuda.mem_alloc(backtrack_host.nbytes)
        min_index_device = cuda.mem_alloc(4)       # int32

        # workflow
        cuda.memcpy_htod_async(image_device, image_host, stream)
        bgr2gray(
                gray_image_device,
                image_device,
                np.int32(width),
                np.int32(height),
                block=(block_x, block_y, 1),
                grid=(width // block_x + 1, height // block_y + 1, 1),
                stream=stream)
        sobel_abs(
                energy_device,
                gray_image_device,
                np.int32(width),
                np.int32(height),
                block=(block_x, block_y, 1),
                grid=(width // block_x + 1, height // block_y + 1, 1),
                stream=stream)
        for row in range(1, height):
            min_energy_at_row(
                energy_device,
                backtrack_device,
                np.int32(width),
                np.int32(row),
                block=(block_x, 1, 1),
                grid=(width // block_x + 1, 1, 1),
                stream=stream)
        get_min_index(
                energy_device,
                min_index_device,
                np.int32(width),
                np.int32(height),
                block=(1, 1, 1),
                grid=(1, 1, 1),
                stream=stream)
        cuda.memcpy_dtoh_async(backtrack_host, backtrack_device, stream)
        cuda.memcpy_dtoh_async(min_index_host, min_index_device, stream)

        # run
        stream.synchronize()

        # get result
        return backtrack_host.reshape(height, width), min_index_host[0]

    finally:
        cuda_ctx.pop()


# unused, for documentation only
def carve_column(img):
    image = np.ascontiguousarray(img)
    r, c = image.shape[0:2]

    backtrack_m, min_index = minimum_seam(image, c, r)
    mask = np.ones((r, c), dtype=np.bool)

    j = min_index
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack_m[i, j]

    mask = np.stack([mask] * 3, axis=2)
    image = image[mask].reshape((r, c - 1, 3))
    return image


# core function
class Core:
    def __init__(self, gpu_id:int=0) -> None:
        self._cuda_ctx = cuda.Device(gpu_id).make_context()
        try:
            with open(cuda_codes) as f:
                _codes = f.read()
            self._mod = SourceModule(_codes)

            # cuda stream and funtions
            self._stream = cuda.Stream()
            self._bgr2gray = self._mod.get_function("bgr2gray")
            self._sobel_abs = self._mod.get_function("sobel_abs")
            self._min_energy_at_row = self._mod.get_function("min_energy_at_row")
            self._get_min_index = self._mod.get_function("get_min_index")

        finally:
            self._cuda_ctx.pop()

    def _minimum_seam(self, image:np.ndarray, width:int, height:int) -> np.ndarray:
        global block_x, block_y, data_type, data_size

        self._cuda_ctx.push()
        try:
            # host data
            image_host = image.astype(data_type)
            backtrack_host = cuda.pagelocked_zeros(shape=(height, width), dtype=np.int32)
            min_index_host = cuda.pagelocked_zeros(shape=(1,), dtype=np.int32)

            # device data
            image_device = cuda.mem_alloc(image_host.nbytes)
            gray_image_device = cuda.mem_alloc(width * height * data_size)
            energy_device = cuda.mem_alloc(width * height * data_size)
            backtrack_device = cuda.mem_alloc(backtrack_host.nbytes)
            min_index_device = cuda.mem_alloc(4)       # int32

            # workflow
            cuda.memcpy_htod_async(image_device, image_host, self._stream)
            self._bgr2gray(
                    gray_image_device,
                    image_device,
                    np.int32(width),
                    np.int32(height),
                    block=(block_x, block_y, 1),
                    grid=(width // block_x + 1, height // block_y + 1, 1),
                    stream=self._stream)
            self._sobel_abs(
                    energy_device,
                    gray_image_device,
                    np.int32(width),
                    np.int32(height),
                    block=(block_x, block_y, 1),
                    grid=(width // block_x + 1, height // block_y + 1, 1),
                    stream=self._stream)
            for row in range(1, height):
                self._min_energy_at_row(
                    energy_device,
                    backtrack_device,
                    np.int32(width),
                    np.int32(row),
                    block=(block_x, 1, 1),
                    grid=(width // block_x + 1, 1, 1),
                    stream=self._stream)
            self._get_min_index(
                    energy_device,
                    min_index_device,
                    np.int32(width),
                    np.int32(height),
                    block=(1, 1, 1),
                    grid=(1, 1, 1),
                    stream=self._stream)
            cuda.memcpy_dtoh_async(backtrack_host, backtrack_device, self._stream)
            cuda.memcpy_dtoh_async(min_index_host, min_index_device, self._stream)

            # run
            self._stream.synchronize()

            # get result
            return backtrack_host.reshape(height, width), min_index_host[0]

        finally:
            self._cuda_ctx.pop()

    def carve_column(self, img:np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(img)
        r, c = image.shape[0:2]

        backtrack_m, min_index = self._minimum_seam(image, c, r)
        mask = np.ones((r, c), dtype=np.bool)

        j = min_index
        for i in reversed(range(r)):
            mask[i, j] = False
            j = backtrack_m[i, j]

        mask = np.stack([mask] * 3, axis=2)
        image = image[mask].reshape((r, c - 1, 3))
        return image
