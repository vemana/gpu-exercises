#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__constant__ float d_filter_const[9];

__global__ void reference_conv2d_kernel(const float* a, float* c, long long width, long long height) {
    __shared__ float sdata[18][18];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    int smem_row = ty + 1;
    int smem_col = tx + 1;
    
    if (row < height && col < width) {
        sdata[smem_row][smem_col] = a[row * width + col];
    } else {
        sdata[smem_row][smem_col] = 0.0f;
    }
    
    if (ty == 0) {
        int g_row = row - 1;
        sdata[0][smem_col] = (g_row >= 0 && col < width) ? a[g_row * width + col] : 0.0f;
    }
    if (ty == blockDim.y - 1) {
        int g_row = row + 1;
        sdata[smem_row + 1][smem_col] = (g_row < height && col < width) ? a[g_row * width + col] : 0.0f;
    }
    if (tx == 0) {
        int g_col = col - 1;
        sdata[smem_row][0] = (row < height && g_col >= 0) ? a[row * width + g_col] : 0.0f;
    }
    if (tx == blockDim.x - 1) {
        int g_col = col + 1;
        sdata[smem_row][smem_col + 1] = (row < height && g_col < width) ? a[row * width + g_col] : 0.0f;
    }
    
    if (tx == 0 && ty == 0) {
        sdata[0][0] = (row - 1 >= 0 && col - 1 >= 0) ? a[(row - 1) * width + (col - 1)] : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        sdata[0][smem_col + 1] = (row - 1 >= 0 && col + 1 < width) ? a[(row - 1) * width + (col + 1)] : 0.0f;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        sdata[smem_row + 1][0] = (row + 1 < height && col - 1 >= 0) ? a[(row + 1) * width + (col - 1)] : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        sdata[smem_row + 1][smem_col + 1] = (row + 1 < height && col + 1 < width) ? a[(row + 1) * width + (col + 1)] : 0.0f;
    }
    
    __syncthreads();
    
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                float pixel = sdata[smem_row + fy][smem_col + fx];
                float weight = d_filter_const[(fy + 1) * 3 + (fx + 1)];
                sum += pixel * weight;
            }
        }
        c[row * width + col] = sum;
    }
}

std::vector<LaunchConfig> launch_reference_conv2d(const float* a, const float* filter, float* c, long long width, long long height) {
    global_tracer.trace("Entering launch_reference_conv2d");

    cudaMemcpyToSymbol(d_filter_const, filter, 9 * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    global_tracer.trace("Launching reference_conv2d_kernel");
    reference_conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, width, height);

    global_tracer.trace("Exiting launch_reference_conv2d");
    return {{"reference_conv2d_kernel", (const void*)reference_conv2d_kernel, static_cast<long long>(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), static_cast<int>(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
