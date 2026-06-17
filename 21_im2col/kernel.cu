#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void im2col_kernel(const float* input, float* output, long long C, long long H, long long W) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total_elements = C * H * W;
    if (idx < total_elements) {
        // student implements im2col here
    }
}

std::vector<LaunchConfig> launch_im2col(const float* input, float* output, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_im2col");
    
    int threadsPerBlock = 256;
    long long total_elements = C * H * W;
    long long blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching im2col_kernel");
    im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, C, H, W);
    
    global_tracer.trace("Exiting launch_im2col");
    
    return {{"im2col_kernel", (const void*)im2col_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
