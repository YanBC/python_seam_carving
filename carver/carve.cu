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
