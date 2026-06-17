#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void max_pool2d_kernel(const float* input, float* output, long long channels, long long height, long long width) {
    long long out_height = height / 2;
    long long out_width = width / 2;

    long long x = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long y = blockIdx.y * (long long)blockDim.y + threadIdx.y;
    long long c = blockIdx.z * (long long)blockDim.z + threadIdx.z;

    if (x < out_width && y < out_height && c < channels) {
        // float max_val = -1e9;
        // student implements max pooling here
        // output[(c * out_height + y) * out_width + x] = max_val;
    }
}

std::vector<LaunchConfig> launch_max_pool2d(const float* input, float* output, long long channels, long long height, long long width) {
    global_tracer.trace("Entering launch_max_pool2d");
    
    long long out_height = height / 2;
    long long out_width = width / 2;

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z
    );
    
    global_tracer.trace("Launching max_pool2d_kernel");
    max_pool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, channels, height, width);
    
    global_tracer.trace("Exiting launch_max_pool2d");
    
    return {{"max_pool2d_kernel", (const void*)max_pool2d_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
