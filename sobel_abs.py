import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2 as cv


cuda.init()
device = cuda.Device(0)
cuda_ctx = device.make_context()
stream = cuda.Stream()


try:
    mod = SourceModule("""
        __global__ void sobel_abs_x(float *g_odata, float *g_idata, int width, int height) {
            int des_x = blockIdx.x * blockDim.x + threadIdx.x;
            int des_y = blockIdx.y * blockDim.y + threadIdx.y;
            if (des_x >= width || des_y >= height)
                return;

            int index = des_y * width + des_x;
            float value = 0;

            if (des_x == 0 || des_x == width - 1) {
                value = 0;
            }
            else {
                value = -2 * g_idata[index - 1] + 2 * g_idata[index + 1];
                if (des_y != 0) {
                    value += -1 * g_idata[index - width - 1] + 1 * g_idata[index - width + 1];
                }
                if (des_y != height - 1) {
                    value += -1 * g_idata[index + width - 1] + 1 * g_idata[index + width + 1];
                }
            }
            if (value <= 0) {
                g_odata[index] = -1 * value;
            } else {
                g_odata[index] = value;
            }
        }

        __global__ void sobel_abs_y(float *g_odata, float *g_idata, int width, int height) {
            int des_x = blockIdx.x * blockDim.x + threadIdx.x;
            int des_y = blockIdx.y * blockDim.y + threadIdx.y;
            if (des_x >= width || des_y >= height)
                return;

            int index = des_y * width + des_x;
            float value = 0;

            if (des_y == 0 || des_y == height - 1) {
                value = 0;
            }
            else {
                value = -2 * g_idata[index - width] + 2 * g_idata[index + width];
                if (des_x != 0) {
                    value += -1 * g_idata[index - width - 1] + 1 * g_idata[index + width - 1];
                }
                if (des_x != width - 1) {
                    value += -1 * g_idata[index - width + 1] + 1 * g_idata[index + width + 1];
                }
            }
            if (value <= 0) {
                g_odata[index] = -1 * value;
            } else {
                g_odata[index] = value;
            }
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
    """)
    sobel_abs_x = mod.get_function("sobel_abs_x")
    sobel_abs_y = mod.get_function("sobel_abs_y")
    sobel_abs = mod.get_function("sobel_abs")

    np.random.seed(0)
    image = np.random.randint(0, 255, (500, 500))
    image_h, image_w = image.shape[:2]
    block_x = 32
    block_y = 32

    arr_in = image.astype(np.float32)
    arr_out = cuda.pagelocked_zeros(shape=(image_h, image_w), dtype=np.float32)
    height, width = np.asarray(arr_in.shape[:2]).astype(np.int32)

    arr_in_gpu = cuda.mem_alloc(arr_in.nbytes)
    arr_out_gpu = cuda.mem_alloc(arr_out.nbytes)

    cuda.memcpy_htod_async(arr_in_gpu, arr_in, stream=stream)

    sobel_abs(arr_out_gpu, arr_in_gpu, width, height, block=(block_x, block_y, 1), grid=(image_w // block_x + 1, image_h // block_y + 1), stream=stream)

    cuda.memcpy_dtoh_async(arr_out, arr_out_gpu, stream=stream)

    stream.synchronize()
    # print(arr_out)

    gpu_out = arr_out.copy()
    opencv_out_x = cv.Sobel(image.astype(np.float64), cv.CV_64F, 1, 0, ksize=3).astype(np.float32)
    opencv_out_y = cv.Sobel(image.astype(np.float64), cv.CV_64F, 0, 1, ksize=3).astype(np.float32)
    opencv_out = np.sqrt(opencv_out_x * opencv_out_x + opencv_out_y * opencv_out_y)
    error = np.abs(gpu_out - opencv_out)
    print(error)
    print(f"max error: {np.max(error)}")
    print(f"mean error: {np.mean(error)}")

finally:
    cuda_ctx.pop()
