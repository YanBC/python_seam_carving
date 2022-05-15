import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2 as cv
from carver.carve_slow import minimum_seam


cuda.init()
device = cuda.Device(0)
cuda_ctx = device.make_context()
stream = cuda.Stream()


try:
    mod = SourceModule("""
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
    """)
    min_energy_at_row = mod.get_function("min_energy_at_row")

    np.random.seed(0)
    energy_h = 5
    energy_w = 32
    block_shape = (32, 1, 1)
    grid_shape = (energy_w // 32 + 1, 1, 1)
    energy = np.random.randn(energy_h, energy_w)

    energy_host = energy.copy().astype(np.float32)
    backtrack_host = cuda.pagelocked_zeros(shape=(energy_h, energy_w), dtype=np.int32)

    energy_device = cuda.mem_alloc(energy_host.nbytes)
    backtrack_device = cuda.mem_alloc(backtrack_host.nbytes)

    cuda.memcpy_htod_async(energy_device, energy_host, stream)
    for row in range(1, energy_h):
        min_energy_at_row(energy_device, backtrack_device, np.int32(energy_w), np.int32(row), block=block_shape, grid=grid_shape, stream=stream)
    cuda.memcpy_dtoh_async(backtrack_host, backtrack_device, stream)
    cuda.memcpy_dtoh_async(energy_host, energy_device, stream)

    stream.synchronize()

    energy_gpu = energy_host.copy()
    backtrack_gpu = backtrack_host.copy()
    energy_cpu, backtrack_cpu = minimum_seam(energy)
    energy_error = np.abs(energy_cpu - energy_gpu)
    print(energy_error)
    print(f"energy error sum: {energy_error.sum()}")
    print(f"energy error max: {energy_error.max()}")
    print(f"energy error mean: {energy_error.mean()}")

    backtrack_error = np.abs(backtrack_cpu - backtrack_gpu)
    print(backtrack_error)
    print(f"backtrack error sum: {backtrack_error.sum()}")
    print(f"backtrack error max: {backtrack_error.max()}")
    print(f"backtrack error mean: {backtrack_error.mean()}")

finally:
    cuda_ctx.pop()
