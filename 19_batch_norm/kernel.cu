

#include "kernel.h"

#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void batch_norm_kernel(const float* input, float* output, long long N, long long C, long long H, long long W) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_batch_norm(const float* input, float* output, long long N, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_batch_norm");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = C;
    
    global_tracer.trace("Launching batch_norm_kernel");
    batch_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, C, H, W);
    
    global_tracer.trace("Exiting launch_batch_norm");
    
    return {{"batch_norm_kernel", (const void*)batch_norm_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
