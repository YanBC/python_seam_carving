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
    """)
    bgr2gray = mod.get_function("bgr2gray")

    image = cv.imread("./images/rem.jpeg")
    # image = np.random.randint(0, 255, (500, 500, 3))
    image_h, image_w = image.shape[:2]
    block_x = 32
    block_y = 32

    arr_in = image.astype(np.float32)
    arr_out = cuda.pagelocked_zeros(shape=(image_h, image_w), dtype=np.float32)
    height, width = np.asarray(arr_in.shape[:2]).astype(np.int32)

    arr_in_gpu = cuda.mem_alloc(arr_in.nbytes)
    arr_out_gpu = cuda.mem_alloc(arr_out.nbytes)
    # width_gpu = cuda.mem_alloc(width.nbytes)
    # height_gpu = cuda.mem_alloc(height.nbytes)

    cuda.memcpy_htod_async(arr_in_gpu, arr_in, stream=stream)
    # cuda.memcpy_htod_async(width_gpu, width, stream=stream)
    # cuda.memcpy_htod_async(height_gpu, height, stream=stream)

    bgr2gray(arr_out_gpu, arr_in_gpu, width, height, block=(block_x, block_y, 1), grid=(image_w // block_x + 1, image_h // block_y + 1), stream=stream)

    cuda.memcpy_dtoh_async(arr_out, arr_out_gpu, stream=stream)

    stream.synchronize()
    arr_out = arr_out.astype(np.uint8)

    # opencv_out = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # print(arr_out.astype(np.float32) - opencv_out.astype(np.float32))

    cv.imwrite("./images/rem_gray.jpeg", arr_out)

finally:
    cuda_ctx.pop()
