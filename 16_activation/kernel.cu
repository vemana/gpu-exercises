#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void silu_kernel(const float* input, float* output, long long size) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < size) {
        // float x = input[idx];
        // student implements SiLU here
        // output[idx] = x / (1.0f + expf(-x));
    }
}

std::vector<LaunchConfig> launch_activation(const float* input, float* output, long long size) {
    global_tracer.trace("Entering launch_activation");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching SiLU kernel");
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    
    global_tracer.trace("Exiting launch_activation");
    
    return {{"silu_kernel", (const void*)silu_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
