import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from typing import Tuple


#######################
# Meta Data
#######################
cuda.init()
gpu_id = 0
block_x = 32
block_y = 32
data_type = np.float32
data_size = np.ones(1, dtype=data_type).nbytes


#######################
# Functions
#######################
def minimum_seam(image:np.ndarray, width:int, height:int) -> Tuple[np.ndarray, int]:
    global mod, block_x, block_y, data_type, data_size, gpu_id

    cuda_ctx = cuda.Device(gpu_id).make_context()
    mod = SourceModule("""
            __global__ void bgr2gray(float *g_odata, float *g_idata, int width, int height) {
                // printf("%d, %d\\n", width, height);
                int des_x = blockIdx.x * blockDim.x + threadIdx.x;
                int des_y = blockIdx.y * blockDim.y + threadIdx.y;
                if (des_x >= width || des_y >= height)
                    return;
                int des_id = des_y * width + des_x;
                int src_r_id = des_id * 3;
                g_odata[des_id] = 0.299 * g_idata[src_r_id] + 0.587 * g_idata[src_r_id+1] + 0.114 * g_idata[src_r_id+2];
            }


            __global__ void sobel_abs(float *g_odata, float *g_idata, int width, int height) {
                int des_x = blockIdx.x * blockDim.x + threadIdx.x;
                int des_y = blockIdx.y * blockDim.y + threadIdx.y;
                if (des_x >= width || des_y >= height)
                    return;

                int index = des_y * width + des_x;
                float value_x = 0;
                float value_y = 0;

                if (des_x == 0 || des_x == width - 1) {
                    value_x = 0;
                }
                else {
                    value_x = -2 * g_idata[index - 1] + 2 * g_idata[index + 1];
                    if (des_y != 0) {
                        value_x += -1 * g_idata[index - width - 1] + 1 * g_idata[index - width + 1];
                    }
                    if (des_y != height - 1) {
                        value_x += -1 * g_idata[index + width - 1] + 1 * g_idata[index + width + 1];
                    }
                }

                if (des_y == 0 || des_y == height - 1) {
                    value_y = 0;
                }
                else {
                    value_y = -2 * g_idata[index - width] + 2 * g_idata[index + width];
                    if (des_x != 0) {
                        value_y += -1 * g_idata[index - width - 1] + 1 * g_idata[index + width - 1];
                    }
                    if (des_x != width - 1) {
                        value_y += -1 * g_idata[index - width + 1] + 1 * g_idata[index + width + 1];
                    }
                }

                g_odata[index] = sqrt(value_x * value_x + value_y * value_y);
            }


            __device__ int arg_min(float *arr, int size) {
                int min_offset = 0;
                float min_val = arr[0];
                for (int i = 1; i < size; i++) {
                    if (arr[i] < min_val) {
                        min_offset = i;
                        min_val = arr[i];
                    }
                }
                return min_offset;
            }

            __device__ int get_array_index(int col, int row, int width) {
                return row * width + col;
            }

            __global__ void min_energy_at_row(float *energy_m, int *backtrack_m, int width, int row) {
                int col = threadIdx.x + blockIdx.x * blockDim.x;
                if (col >= width) {
                    return;
                }

                int shift_col;
                if (col == 0) {
                    shift_col = 0;
                } else if (col == width - 1) {
                    shift_col = -2;
                } else {
                    shift_col = -1;
                }
                int head = get_array_index(col + shift_col, row - 1, width);
                int min_offset = arg_min(energy_m + head, 3);
                int min_col = col + shift_col + min_offset;

                int min_index = get_array_index(min_col, row - 1, width);
                int current_index = get_array_index(col, row, width);
                energy_m[current_index] += energy_m[min_index];
                backtrack_m[current_index] = min_col;
            }

            __global__ void get_min_index(float *energy_m, int *index, int width, int height) {
                int offset = width * (height - 1);
                *index = arg_min(energy_m + offset, width);
            }
    """)

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
