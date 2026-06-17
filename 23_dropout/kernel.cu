#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

// Note: For Dropout PRNG, use a simple deterministic hash function like PCG Hash
// to avoid the overhead of cuRAND and make verification perfectly reproducible.

__global__ void dropout_kernel(const float* input, float* output, long long size, float drop_prob, float scale) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < size) {
        // student implements dropout here
    }
}

std::vector<LaunchConfig> launch_dropout(const float* input, float* output, long long size, float drop_prob) {
    global_tracer.trace("Entering launch_dropout");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    float scale = 1.0f / (1.0f - drop_prob);
    
    global_tracer.trace("Launching dropout_kernel");
    dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size, drop_prob, scale);
    
    global_tracer.trace("Exiting launch_dropout");
    
    return {{"dropout_kernel", (const void*)dropout_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
