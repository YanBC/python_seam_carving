import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from typing import Tuple
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
max_pixel_backward_energy = 16 * 255 * 4092


#######################
# Core
#######################
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
            self._add_mask_by_factor = self._mod.get_function("add_mask_by_factor")

        finally:
            self._cuda_ctx.pop()

    def _get_keep(self, min_idx: int, backtrack: np.ndarray) -> np.ndarray:
        r, _ = backtrack.shape
        keep = np.ones_like(backtrack, dtype=np.bool)
        j = min_idx
        for i in reversed(range(r)):
            keep[i, j] = False
            j = backtrack[i, j]
        return keep

    def _minimum_seam(
                self,
                image:np.ndarray,
                r_mask:np.ndarray,
                p_mask:np.ndarray,
                width:int,
                height:int
        ) -> Tuple[np.ndarray, np.int32]:
        global block_x, block_y, data_size, max_pixel_backward_energy

        self._cuda_ctx.push()
        try:
            # host data
            image_host = image
            r_mask_host = r_mask
            p_mask_host = p_mask
            backtrack_host = cuda.pagelocked_zeros(shape=(height, width), dtype=np.int32)
            min_index_host = cuda.pagelocked_zeros(shape=(1,), dtype=np.int32)

            # device data
            image_device = cuda.mem_alloc(image_host.nbytes)
            if r_mask_host is not None:
                r_mask_device = cuda.mem_alloc(r_mask_host.nbytes)
            if p_mask_host is not None:
                p_mask_device = cuda.mem_alloc(p_mask_host.nbytes)
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
            if r_mask_host is not None:
                cuda.memcpy_htod_async(r_mask_device, r_mask_host, self._stream)
                self._add_mask_by_factor(
                    energy_device,
                    r_mask_device,
                    np.float32(-1 * max_pixel_backward_energy * height),
                    np.int32(width),
                    np.int32(height),
                    block=(block_x, block_y, 1),
                    grid=(width // block_x + 1, height // block_y + 1, 1),
                    stream=self._stream)
            if p_mask_host is not None:
                cuda.memcpy_htod_async(p_mask_device, p_mask_host, self._stream)
                self._add_mask_by_factor(
                    energy_device,
                    p_mask_device,
                    np.float32(max_pixel_backward_energy * height),
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

    def carve_column(
                self,
                img:np.ndarray,
                mask_remove:np.ndarray=None,
                mask_protect:np.ndarray=None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        global data_type

        r, c = img.shape[0:2]
        _img = img.astype(data_type)
        _img = np.ascontiguousarray(_img)
        if mask_protect is None:
            _mask_protect = None
        else:
            _mask_protect = mask_protect.astype(data_type)
            _mask_protect = np.ascontiguousarray(_mask_protect)

        if mask_remove is None:
            _mask_remove = None
        else:
            _mask_remove = mask_remove.astype(data_type)
            _mask_remove = np.ascontiguousarray(_mask_remove)

        backtrack_m, min_index = self._minimum_seam(
                _img, _mask_remove, _mask_protect, c, r)
        keep = self._get_keep(min_index, backtrack_m)

        keep3c = np.stack([keep] * 3, axis=2)
        _img = _img[keep3c].reshape((r, c - 1, 3))
        if _mask_remove is not None:
            _mask_remove = _mask_remove[keep].reshape((r, c - 1))
        if _mask_protect is not None:
            _mask_protect = _mask_protect[keep].reshape((r, c - 1))

        return _img, _mask_remove, _mask_protect
