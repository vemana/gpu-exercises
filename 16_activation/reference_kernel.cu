

#include "reference_kernel.h"

#include <cmath>
#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void silu_reference_kernel(const float* input, float* output, long long size) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

std::vector<LaunchConfig> launch_reference_activation(const float* input, float* output, long long size) {
    global_tracer.trace("Entering launch_reference_activation");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching SiLU reference kernel");
    silu_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    
    global_tracer.trace("Exiting launch_reference_activation");
    
    return {{"silu_reference_kernel", (const void*)silu_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
